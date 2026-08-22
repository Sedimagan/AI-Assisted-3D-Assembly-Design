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
                 normal_hint=None,
                 ) -> tuple[Optional[trimesh.Trimesh], float, Optional[str]]:
        hits = self.bank.query(comp_type, category, target_extents, top_k=1,
                                exclude_assemblies=exclude_assemblies)
        if not hits:
            return None, 0.0, None
        hit = hits[0]
        mesh = self.bank.load_mesh(hit.part_id)
        fitted = fit_to_bbox(mesh, target_extents, normal_hint=normal_hint, comp_type=comp_type)
        return fitted, hit.fit_score, hit.part_id


# Types whose meaningful orientation axis is unambiguous from the type alone
# -- used to skip the geometric rod-vs-disk heuristic entirely when we
# already know the answer (see _joint_axis_index).
_FLAT_JOINT_TYPES = {"washer", "thin_plate", "thick_plate"}
_ROD_JOINT_TYPES  = {"bolt", "long_shaft", "short_shaft"}

# A hole's own bbox depth is usually just its visible recess/counterbore,
# not the full material thickness a bolt's shaft needs to span to protrude
# out the far side -- true thickness isn't threaded through this far, so
# this is a deliberate approximation: make the shaft longer than the raw
# hole depth by this factor. Bolt-only; washers/nuts are unaffected.
BOLT_PROTRUSION_FACTOR = 1.6

# Extra fraction of the head's own reach added on top of "just touching the
# surface", so the head visibly clears it rather than sitting exactly flush.
# How far above the surface the head sits, as a fraction of the head's own
# reach from the mesh's centroid (its "how far the head extends" distance)
# -- 0.15 means the head pokes out roughly 15% of that distance above the
# surface, with the shaft's remainder (100% - 15% = 85% of that reach, plus
# everything on the tip side) pulled back through/below it.
_BOLT_HEAD_CLEARANCE_FRACTION = 0.15


def _bolt_end_spreads(mesh: trimesh.Trimesh, joint_axis: int) -> tuple[float, float]:
    """RMS spread of vertices (perpendicular to joint_axis -- i.e. their
    typical radius from the joint axis) in the top vs bottom 20% band of
    the mesh's extent along joint_axis. Shared basis for both
    _bolt_head_sign (which end is wider) and _bolt_shaft_radius (how wide
    the narrow end actually is, in absolute terms)."""
    coords = mesh.vertices[:, joint_axis]
    lo, hi = float(coords.min()), float(coords.max())
    span = hi - lo
    if span < 1e-9:
        return 0.0, 0.0
    band = 0.2 * span
    other_axes = [a for a in range(3) if a != joint_axis]

    def _spread(mask: np.ndarray) -> float:
        pts = mesh.vertices[mask][:, other_axes]
        if len(pts) == 0:
            return 0.0
        return float(np.sqrt((pts ** 2).sum(axis=1)).mean())

    return _spread(coords >= hi - band), _spread(coords <= lo + band)  # (top, bottom)


def _bolt_head_sign(mesh: trimesh.Trimesh, joint_axis: int) -> int:
    """Which end of a rod-like mesh, along its joint_axis local coordinate,
    is the wider "head" end vs the narrower shaft/thread end -- a bolt head
    is flanged/wider than its shaft. Returns +1 if the head is on the
    positive-axis side, -1 if on the negative side. Used to orient a bolt
    head-outward (see rotate_to_target_axis's head_sign param) rather than
    leaving head vs tip to whatever canonicalize() happened to produce."""
    top_spread, bot_spread = _bolt_end_spreads(mesh, joint_axis)
    return 1 if top_spread >= bot_spread else -1


def _bolt_shaft_radius(mesh: trimesh.Trimesh, joint_axis: int, head_sign: int) -> float:
    """Typical cross-sectional radius of a bolt's *shaft* specifically (not
    its head), measured at the tip end -- the band opposite head_sign.
    Used so fit_to_bbox scales a bolt by its shaft diameter matching the
    hole diameter (the physically correct constraint: the shaft is what
    passes through the hole), rather than by its overall bounding-box
    width, which is dominated by the head and would size the head to
    exactly the hole's diameter instead of visibly wider than it -- the
    bug reported after the first orientation fix ("bolts still look like
    they're inside the hole")."""
    top_spread, bot_spread = _bolt_end_spreads(mesh, joint_axis)
    return bot_spread if head_sign >= 0 else top_spread


def bolt_head_clearance_offset(mesh: trimesh.Trimesh, normal_hint,
                                comp_type: Optional[str]) -> np.ndarray:
    """For a bolt already oriented head-outward (fit_to_bbox/rotate_to_target_axis
    orient it that way when comp_type == "bolt"), how far to shift its placement
    along normal_hint so the head clears the hole's entry surface instead of
    being centered on it. canonicalize() centers every part at its own
    centroid, which by default puts that centroid (roughly the shaft's
    middle) at the hole's surface centroid -- burying about half the head
    below the surface instead of resting it above. Returns a zero vector for
    non-bolt types or a degenerate normal_hint (nothing to add to placement)."""
    if comp_type != "bolt" or normal_hint is None:
        return np.zeros(3)
    n = np.asarray(normal_hint, dtype=np.float64)
    norm = np.linalg.norm(n)
    if norm < 1e-6:
        return np.zeros(3)
    n = n / norm
    proj = mesh.vertices @ n
    head_extent = float(proj.max())  # how far the head reaches from centroid, now outward
    # Naive placement (mesh centroid at the surface centroid, zero shift)
    # puts the head's outer tip at +head_extent above the surface -- pull
    # the whole mesh back by (most of) that distance so the tip instead
    # ends up just at/above the surface, dragging the shaft/tip end further
    # through and past it on the far side.
    return -n * head_extent * (1.0 - _BOLT_HEAD_CLEARANCE_FRACTION)


def _joint_axis_index(extents: np.ndarray, comp_type: Optional[str] = None) -> int:
    """Which of a mesh's 3 local axes is its "joint axis" -- the one
    rotate_to_target_axis will align with normal_hint: the longest extent
    for a rod-like part (bolt/shaft), the shortest for a flat/disk-like part
    (washer). Shared by fit_to_bbox and rotate_to_target_axis so both agree
    on which axis is which -- see fit_to_bbox's docstring for why that
    agreement matters.

    comp_type, when it's one of the unambiguous types above, decides this
    directly -- the caller already knows whether this is a bolt or a washer,
    so there's no need to re-infer "rod vs disk" from the mesh's own extent
    proportions. Falls back to that geometric heuristic (comparing the
    *middle* extent to shortest/longest, not just shortest-vs-longest, since
    both a 5x5x20 bolt and a 5x20x20 washer have the same shortest/longest
    ratio 0.25) only for types not in either set above (e.g. "nut", "body"),
    where there isn't a clear a-priori answer.

    Using comp_type when available isn't just a convenience -- the
    geometric heuristic has a real failure mode the type-based path avoids:
    it ties (and misclassifies a rod as flat) whenever the *other* two
    extents happen to be close to equal, which is the NORMAL case for any
    round hole's in-plane footprint (its two in-plane dimensions are both
    ~diameter). Confirmed on a real Tool_Post upload: a bolt's target bbox
    was [16, 16, 10.6] (16x16 round footprint, 10.6 deep) -- shortest=10.6,
    mid=long=16 exactly, so long-mid=0 always loses to mid-short, tripping
    "is_flat" for every such hole regardless of the true component type.
    Fixed 2026-08-22.
    """
    if comp_type in _FLAT_JOINT_TYPES:
        order = np.argsort(extents)
        return order[0]
    if comp_type in _ROD_JOINT_TYPES:
        order = np.argsort(extents)
        return order[-1]
    order = np.argsort(extents)  # shortest .. longest axis indices
    short, mid, long = extents[order[0]], extents[order[1]], extents[order[2]]
    is_flat = (long - mid) < (mid - short)  # middle extent closer to longest => disk
    return order[0] if is_flat else order[-1]


def fit_to_bbox(mesh: trimesh.Trimesh, target_extents, normal_hint=None,
                 comp_type: Optional[str] = None) -> trimesh.Trimesh:
    """Rescale a canonical (unit-max-extent, origin-centered) part mesh onto a
    target bounding box, then (if normal_hint is given) rotate it so its
    joint axis points along that direction — see rotate_to_target_axis.
    Scaling and rotation are done together, in that order, so both agree on
    which mesh axis is the "joint axis" (computed once, from the mesh's
    pristine pre-scale proportions — see the note below on why that
    agreement matters).

    If normal_hint is given and (as is the norm in this corpus) axis-aligned,
    the target dimension *along that world axis* is treated as the hole's
    true depth/bore-axis extent and matched to the mesh's own joint axis
    (see _joint_axis_index). The other two target dimensions (in-plane,
    perpendicular to normal_hint) are matched by rank to the mesh's
    remaining two axes.

    Without normal_hint, falls back to matching purely by numeric rank
    (shortest target value -> mesh's shortest axis, etc.) -- correct only
    when the hole's own depth happens to be its largest bbox dimension (a
    genuinely deep hole/pocket). For a hole that's *wider than it is deep*
    (a shallow counterbore or a stubby fastener's recess), rank-matching
    assigns the hole's small depth value to the mesh's shortest axis
    regardless of whether that's actually the depth-facing axis -- crushing
    a bolt's shaft length down to the hole's small depth and ballooning its
    cross-section out to the hole's large diameter: a squat, proportion-
    inverted part that visibly looks rotated ~90° from correct. Confirmed
    exactly this on a real Tool_Post upload (a hole 16x16 in-plane but only
    ~10.6 deep) -- fixed 2026-08-22.

    Why scale and rotate must share one joint-axis decision: computing it
    separately in each step (as this used to do, calling rotate_to_target_axis
    as a distinct second pass) re-derives "which axis is longest" from the
    mesh's extents *after* scaling — but scaling a shallow/wide target can
    make the joint axis numerically the *shortest* of the three (its true
    role is depth, and depth was small here), so a second, independent
    re-derivation picks a different, wrong axis than the one just scaled as
    the joint axis. Computing it once, before scaling, and reusing that same
    index for the rotation avoids the two steps disagreeing.
    """
    mesh = mesh.copy()
    extents = mesh.extents
    order = np.argsort(extents)          # mesh axis indices, shortest..longest
    target = np.asarray(target_extents, dtype=np.float64)
    scale_per_axis = np.ones(3)

    depth_axis = None
    if normal_hint is not None:
        n = np.asarray(normal_hint, dtype=np.float64)
        if np.linalg.norm(n) > 1e-6:
            depth_axis = int(np.argmax(np.abs(n)))  # dominant world axis

    joint_axis = _joint_axis_index(extents, comp_type)  # from PRE-scale extents
    # head_sign must be read off the mesh's local vertex layout before
    # scaling changes it -- scaling is per-axis-positive so it can't flip
    # which end is which, but easiest to keep this unambiguous by computing
    # it right alongside joint_axis, from the same pre-scale mesh.
    head_sign = _bolt_head_sign(mesh, joint_axis) if comp_type == "bolt" else None

    if depth_axis is not None:
        depth_value = target[depth_axis]
        if comp_type == "bolt":
            # See BOLT_PROTRUSION_FACTOR -- the hole's own depth is usually
            # just its visible recess, not the full material thickness a
            # shaft needs to span to protrude out the far side.
            depth_value = depth_value * BOLT_PROTRUSION_FACTOR
        scale_per_axis[joint_axis] = depth_value / max(extents[joint_axis], 1e-9)

        inplane_target = np.sort(np.delete(target, depth_axis))     # 2 values, ascending
        inplane_mesh_axes = [a for a in order if a != joint_axis]   # already ascending

        if comp_type == "bolt":
            # A bolt's HEAD must end up wider than the hole (so it can't
            # pass through and rests on the surface) -- but the shaft must
            # match the hole's diameter (so it passes through). Matching
            # the mesh's overall bounding-box width (dominated by the
            # head) to the hole's diameter, as the else-branch below does,
            # sizes the head to *exactly* the hole's diameter instead of
            # visibly wider -- confirmed as the cause of a real "the bolts
            # still look like they're inside the hole" report after the
            # first orientation fix. Scale instead by the shaft's own
            # (narrower) radius specifically, applying the SAME factor to
            # both in-plane axes (a uniform, non-distorting scale) so the
            # mesh's natural head:shaft width ratio is preserved rather
            # than flattened out -- the head then naturally comes out
            # wider than the hole by whatever margin the retrieved part's
            # own geometry has.
            shaft_radius = _bolt_shaft_radius(mesh, joint_axis, head_sign)
            target_shaft_diam = float(np.mean(inplane_target))  # hole is ~round
            if shaft_radius > 1e-9:
                uniform_scale = target_shaft_diam / (2.0 * shaft_radius)
                for axis in inplane_mesh_axes:
                    scale_per_axis[axis] = uniform_scale
            else:
                for rank, axis in enumerate(inplane_mesh_axes):
                    scale_per_axis[axis] = inplane_target[rank] / max(extents[axis], 1e-9)
        else:
            for rank, axis in enumerate(inplane_mesh_axes):
                scale_per_axis[axis] = inplane_target[rank] / max(extents[axis], 1e-9)
    else:
        target_sorted = np.sort(target)  # shortest..longest
        for rank, axis in enumerate(order):
            scale_per_axis[axis] = target_sorted[rank] / max(extents[axis], 1e-9)

    mesh.apply_transform(np.diag([*scale_per_axis, 1.0]))

    if normal_hint is not None:
        mesh = rotate_to_target_axis(mesh, normal_hint, joint_axis=joint_axis,
                                      head_sign=head_sign)

    return mesh


def rotate_to_target_axis(mesh: trimesh.Trimesh, normal_hint,
                           comp_type: Optional[str] = None,
                           joint_axis: Optional[int] = None,
                           head_sign: Optional[int] = None) -> trimesh.Trimesh:
    """Rotate a canonically OBB-aligned part (see part_bank.canonicalize) so
    it points along the real target joint's axis, instead of keeping
    whatever orientation it was canonicalized in. See _joint_axis_index for
    which axis gets aligned (longest for a rod, shortest for a disk, and why
    passing comp_type through matters). No-op if normal_hint is missing/
    degenerate.

    joint_axis: pass this through when the caller already knows which axis
    is the joint axis (e.g. fit_to_bbox, which must compute it from the
    mesh's pre-scale proportions to stay consistent with its own scaling —
    see fit_to_bbox's docstring). Bypasses re-deriving it from this mesh's
    current extents, which can disagree after scaling has changed which
    axis numerically reads as longest/shortest.

    head_sign: for a bolt (see _bolt_head_sign), which end of joint_axis is
    the head. Without this, aligning the joint axis to normal_hint only
    fixes *which line* the shaft lies along, not which end points which way
    -- the head could just as easily end up pointing into the hole as out of
    it, since align_vectors has no notion of "head" or "tip". normal_hint
    points outward (away from material, see surface_analyzer.py), so when
    given, the head is rotated to face +normal_hint specifically, not just
    "aligned with its line".
    """
    if normal_hint is None:
        return mesh
    target = np.asarray(normal_hint, dtype=np.float64)
    norm = np.linalg.norm(target)
    if norm < 1e-6:
        return mesh
    target = target / norm

    align_axis_idx = (joint_axis if joint_axis is not None
                       else _joint_axis_index(mesh.extents, comp_type))

    source = np.zeros(3)
    source[align_axis_idx] = head_sign if head_sign is not None else 1.0

    rot = trimesh.geometry.align_vectors(source, target)
    mesh = mesh.copy()
    mesh.apply_transform(rot)
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


FASTENER_TYPES = {"bolt", "washer", "nut"}


class HybridShapeGenerator:
    def __init__(self, retriever: ShapeRetriever, vae: Optional[ConditionalShapeVAE],
                 device, retrieval_tau: float = 0.6, retrieval_tau_fastener: Optional[float] = None):
        self.retriever = retriever
        self.vae = vae
        self.device = device
        self.retrieval_tau = retrieval_tau
        # Fasteners are highly standardized shapes and the part bank already
        # holds hundreds of real examples — accept a real (if imperfect)
        # retrieval match more readily than a VAE hallucination for these
        # types specifically. Falls back to the global tau if unset.
        self.retrieval_tau_fastener = (
            retrieval_tau_fastener if retrieval_tau_fastener is not None else retrieval_tau
        )

    def generate(
        self,
        comp_type: Optional[str],
        category: Optional[str],
        target_extents,
        cond_vec: Optional[torch.Tensor],
        placement_centroid,
        mode: str = "auto",
        normal_hint=None,
    ) -> Optional[ShapeResult]:
        placement = np.asarray(placement_centroid, dtype=np.float32)
        tau = self.retrieval_tau_fastener if comp_type in FASTENER_TYPES else self.retrieval_tau

        if mode in ("auto", "retrieve"):
            # retrieve() already scales AND rotates via fit_to_bbox -- no
            # separate rotate_to_target_axis call needed (or wanted: it
            # would double-rotate).
            mesh, conf, part_id = self.retriever.retrieve(
                comp_type, category, target_extents, normal_hint=normal_hint,
            )
            if mesh is not None and (mode == "retrieve" or conf >= tau):
                head_offset = bolt_head_clearance_offset(mesh, normal_hint, comp_type)
                return ShapeResult(mesh, "retrieved", conf, part_id, placement + head_offset)

        if mode == "retrieve":
            return None  # forced retrieval requested but nothing found

        if self.vae is None or cond_vec is None:
            return None

        with torch.no_grad():
            occ = self.vae.generate(cond_vec.to(self.device), n_samples=1)[0]
        mesh = devoxelize(occ.cpu().numpy(), threshold=0.5)
        if mesh is None:
            return None
        # fit_to_bbox scales AND rotates -- see its docstring for why the two
        # must be done together (no separate rotate_to_target_axis call here).
        mesh = fit_to_bbox(mesh, target_extents, normal_hint=normal_hint, comp_type=comp_type)
        conf = float(occ.max().item())
        head_offset = bolt_head_clearance_offset(mesh, normal_hint, comp_type)
        return ShapeResult(mesh, "generated", conf, None, placement + head_offset)
