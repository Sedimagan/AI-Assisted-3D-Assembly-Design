"""
shape_generator.py — Phase 3 missing-component shape generation.

Given a partial assembly and a predicted missing-component type (from
NodeRanker) plus a target region (from surface_analyzer.py's open-joint
detector), produce a 3D shape for the missing part:
  - ShapeRetriever:       nearest-neighbour lookup in the part bank, zero training
  - ConditionalShapeVAE:  32^3 occupancy-grid VAE conditioned on the frozen
                          Phase-1 encoder's context vector, for parts with no
                          good retrieval match
  - HybridShapeGenerator: orchestrates the two with a confidence threshold

Mirrors NodeRanker's design: the Phase-1 encoder stays frozen, only this
module's own small parameters are trained (train_shape_gen.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh

from part_bank import PartBank, canonicalize, devoxelize, voxelize, VOXEL_RES

COND_DIM = 64 + 8 + 3 + 2  # ctx(64) + type_onehot(8) + target_bbox(3) + neighbor_scale(2)


# ── Conditional VAE ──────────────────────────────────────────────────────────────

class ConditionalShapeVAE(nn.Module):
    """
    Encoder: 32^3 occupancy -> 3D-CNN (4 conv blocks, 32->64->128->256) -> mu, logvar
    Decoder: [latent Z, cond] -> 3D-deconv -> 32^3 occupancy logits
    """

    def __init__(self, res: int = VOXEL_RES, latent_dim: int = 128,
                 cond_dim: int = COND_DIM):
        super().__init__()
        self.res = res
        self.latent_dim = latent_dim
        self.cond_dim = cond_dim

        def conv_block(cin, cout, stride=2):
            return nn.Sequential(
                nn.Conv3d(cin, cout, kernel_size=4, stride=stride, padding=1),
                nn.BatchNorm3d(cout),
                nn.LeakyReLU(0.2, inplace=True),
            )

        # 32 -> 16 -> 8 -> 4 -> 2
        self.enc = nn.Sequential(
            conv_block(1, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            conv_block(128, 256),
        )
        self._enc_spatial = res // 16  # 2 for res=32
        enc_flat = 256 * self._enc_spatial ** 3
        self.fc_mu = nn.Linear(enc_flat, latent_dim)
        self.fc_logvar = nn.Linear(enc_flat, latent_dim)

        self.fc_dec = nn.Linear(latent_dim + cond_dim, enc_flat)

        def deconv_block(cin, cout, activate=True):
            layers = [nn.ConvTranspose3d(cin, cout, kernel_size=4, stride=2, padding=1)]
            if activate:
                layers += [nn.BatchNorm3d(cout), nn.LeakyReLU(0.2, inplace=True)]
            return nn.Sequential(*layers)

        self.dec = nn.Sequential(
            deconv_block(256, 128),
            deconv_block(128, 64),
            deconv_block(64, 32),
            deconv_block(32, 1, activate=False),  # raw logits, 2 -> 4 -> 8 -> 16 -> 32
        )

    def encode(self, vox: torch.Tensor):
        # vox: (B, 1, res, res, res)
        h = self.enc(vox).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, cond: torch.Tensor):
        h = self.fc_dec(torch.cat([z, cond], dim=-1))
        h = h.view(-1, 256, self._enc_spatial, self._enc_spatial, self._enc_spatial)
        return self.dec(h)  # (B, 1, res, res, res) logits

    def forward(self, vox: torch.Tensor, cond: torch.Tensor):
        mu, logvar = self.encode(vox)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z, cond)
        return recon_logits, mu, logvar

    @torch.no_grad()
    def generate(self, cond: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """cond: (cond_dim,) or (1, cond_dim). Returns (n_samples, res, res, res) probs."""
        if cond.dim() == 1:
            cond = cond.unsqueeze(0)
        cond = cond.expand(n_samples, -1)
        z = torch.randn(n_samples, self.latent_dim, device=cond.device)
        logits = self.decode(z, cond)
        return torch.sigmoid(logits).squeeze(1)  # (n_samples, res, res, res)


def vae_loss(recon_logits, target_vox, mu, logvar, beta_kl: float = 0.05,
             lambda_dice: float = 0.5):
    """BCE + soft-Dice on occupancy + KL divergence."""
    recon_logits = recon_logits.squeeze(1)
    bce = F.binary_cross_entropy_with_logits(recon_logits, target_vox)

    probs = torch.sigmoid(recon_logits)
    dims = tuple(range(1, probs.dim()))
    inter = (probs * target_vox).sum(dim=dims)
    union = probs.sum(dim=dims) + target_vox.sum(dim=dims)
    dice = 1.0 - (2.0 * inter + 1e-6) / (union + 1e-6)
    dice = dice.mean()

    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return bce + lambda_dice * dice + beta_kl * kl, {
        "bce": bce.item(), "dice": dice.item(), "kl": kl.item(),
    }


# ── Retrieval ────────────────────────────────────────────────────────────────────

class ShapeRetriever:
    """Thin wrapper over PartBank.query() + fit_to_bbox()."""

    def __init__(self, bank: PartBank):
        self.bank = bank

    def retrieve(self, comp_type: Optional[str], category: Optional[str],
                 target_extents, exclude_assemblies: Optional[set] = None,
                 ) -> tuple[Optional[trimesh.Trimesh], float, Optional[str]]:
        hits = self.bank.query(comp_type, category, target_extents, top_k=1,
                                exclude_assemblies=exclude_assemblies)
        if not hits:
            return None, 0.0, None
        hit = hits[0]
        mesh = self.bank.load_mesh(hit.part_id)
        fitted = fit_to_bbox(mesh, target_extents)
        return fitted, hit.fit_score, hit.part_id


def fit_to_bbox(mesh: trimesh.Trimesh, target_extents) -> trimesh.Trimesh:
    """Rescale a canonical (unit-max-extent, origin-centered) part mesh onto a
    target bounding box, matching sorted extents (longest axis to longest)."""
    mesh = mesh.copy()
    extents = mesh.extents
    order = np.argsort(extents)          # mesh axis indices, shortest..longest
    target = np.sort(np.asarray(target_extents, dtype=np.float64))  # shortest..longest

    scale_per_axis = np.ones(3)
    for rank, axis in enumerate(order):
        scale_per_axis[axis] = target[rank] / max(extents[axis], 1e-9)

    mesh.apply_transform(np.diag([*scale_per_axis, 1.0]))
    return mesh


# ── Conditioning vector ──────────────────────────────────────────────────────────

def build_conditioning_vector(
    ctx: torch.Tensor,           # (64,) mean-pooled partial-graph embedding
    comp_type_idx: int,          # index into COMP_TYPES (0..7)
    target_bbox_norm,            # (3,) normalized target extents
    neighbor_scale,               # (2,) e.g. [mean neighbor volume, mean neighbor extent]
    n_types: int = 8,
) -> torch.Tensor:
    device = ctx.device
    type_oh = torch.zeros(n_types, device=device)
    type_oh[comp_type_idx] = 1.0
    bbox_t = torch.as_tensor(target_bbox_norm, dtype=torch.float32, device=device)
    nbr_t = torch.as_tensor(neighbor_scale, dtype=torch.float32, device=device)
    return torch.cat([ctx, type_oh, bbox_t, nbr_t])  # (77,)


def estimate_target_bbox(open_joint_extents, part_bank: PartBank,
                          comp_type: Optional[str], category: Optional[str]) -> np.ndarray:
    """Primary: spatial extent of the open-joint region. Fallback: median
    extents of same-type parts in this category, from the part bank."""
    if open_joint_extents is not None and np.all(np.asarray(open_joint_extents) > 1e-6):
        return np.asarray(open_joint_extents, dtype=np.float32)

    candidates = [e for e in part_bank.index
                  if comp_type is None or e["comp_type"] == comp_type]
    if category is not None:
        same_cat = [e for e in candidates if e["category"] == category]
        if same_cat:
            candidates = same_cat
    if not candidates:
        return np.array([0.1, 0.1, 0.3], dtype=np.float32)

    bboxes = np.array([c["bbox"] for c in candidates], dtype=np.float32)
    return np.median(bboxes, axis=0)


# ── Hybrid orchestrator ──────────────────────────────────────────────────────────

@dataclass
class ShapeResult:
    mesh: trimesh.Trimesh
    source: str            # "retrieved" | "generated"
    confidence: float
    part_id: Optional[str]
    placement: np.ndarray   # (3,) translation to open-joint centroid


class HybridShapeGenerator:
    def __init__(self, retriever: ShapeRetriever, vae: Optional[ConditionalShapeVAE],
                 device, retrieval_tau: float = 0.6):
        self.retriever = retriever
        self.vae = vae
        self.device = device
        self.retrieval_tau = retrieval_tau

    def generate(
        self,
        comp_type: Optional[str],
        category: Optional[str],
        target_extents,
        cond_vec: Optional[torch.Tensor],
        placement_centroid,
        mode: str = "auto",
    ) -> Optional[ShapeResult]:
        placement = np.asarray(placement_centroid, dtype=np.float32)

        if mode in ("auto", "retrieve"):
            mesh, conf, part_id = self.retriever.retrieve(comp_type, category, target_extents)
            if mesh is not None and (mode == "retrieve" or conf >= self.retrieval_tau):
                return ShapeResult(mesh, "retrieved", conf, part_id, placement)

        if mode == "retrieve":
            return None  # forced retrieval requested but nothing found

        if self.vae is None or cond_vec is None:
            return None

        with torch.no_grad():
            occ = self.vae.generate(cond_vec.to(self.device), n_samples=1)[0]
        mesh = devoxelize(occ.cpu().numpy(), threshold=0.5)
        if mesh is None:
            return None
        mesh = fit_to_bbox(mesh, target_extents)
        conf = float(occ.max().item())
        return ShapeResult(mesh, "generated", conf, None, placement)
