"""
dataset.py — AssemblyDataset
Loads STEP files from source_dir, converts each assembly to an attributed
graph (nodes = solid bodies, edges = shared surfaces / mate constraints),
and caches the processed PyG Data objects.

Falls back to synthetic graphs when no STEP files are found.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.transforms import RandomLinkSplit

# 8-class component type vocabulary (index used for one-hot encoding)
COMP_TYPES = ["body", "fastener", "bearing", "shaft", "plate", "housing", "gear", "other"]
MATE_TYPES = ["coincident", "concentric", "parallel", "tangent", "fixed", "other"]


# ── STEP file parser ──────────────────────────────────────────────────────────

def _parse_step(step_path: str) -> Optional[Data]:
    """
    Parse a single STEP file into a PyG Data object.

    Node features  (13-dim):
        [0:8]  component-type one-hot (8 classes, default = 'body')
        [8]    normalised volume
        [9]    normalised surface area (estimated from bbox)
        [10]   bbox Δx
        [11]   bbox Δy
        [12]   bbox Δz

    Edge features  (2-dim):
        [0]    mate type encoded (0=coincident … 5=other, normalised to [0,1])
        [1]    weight (1.0 for detected contacts)
    """
    import gmsh

    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("assembly")

    try:
        gmsh.merge(step_path)
        gmsh.model.occ.synchronize()

        volumes = gmsh.model.occ.getEntities(3)   # list of (dim=3, tag) per solid body
        if len(volumes) < 2:
            return None

        # Fragment volumes against each other to share boundary surface tags for contact detection
        gmsh.model.occ.fragment(volumes, [])
        gmsh.model.occ.synchronize()
        
        # Re-fetch volumes since fragmentation may update tags
        volumes = gmsh.model.occ.getEntities(3)

        # ── Node features ──────────────────────────────────────────────────
        node_feats: List[List[float]] = []
        body_surfs: List[frozenset]   = []

        raw_vols, raw_sas = [], []
        for dim, tag in volumes:
            bbox = gmsh.model.occ.getBoundingBox(dim, tag)   # xmin..zmax
            vol  = gmsh.model.occ.getMass(dim, tag)
            dx, dy, dz = bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]
            sa = 2.0 * (dx*dy + dy*dz + dz*dx)              # box surface-area approx.
            raw_vols.append(vol)
            raw_sas.append(sa)

            # Surface tags bounding this body (for mate detection)
            bnd = gmsh.model.getBoundary([(dim, tag)], oriented=False, combined=True)
            body_surfs.append(frozenset(abs(s[1]) for s in bnd if s[0] == 2))

        # Normalise geometry features across the assembly
        vol_max = max(raw_vols) or 1.0
        sa_max  = max(raw_sas)  or 1.0
        bboxes  = []
        for dim, tag in volumes:
            bbox = gmsh.model.occ.getBoundingBox(dim, tag)
            bboxes.append((bbox[3]-bbox[0], bbox[4]-bbox[1], bbox[5]-bbox[2]))

        bbox_max = max(max(b) for b in bboxes) or 1.0

        for i, (dim, tag) in enumerate(volumes):
            type_oh = [0.0] * 8
            type_oh[0] = 1.0                                  # default: "body"
            feat = (
                type_oh
                + [raw_vols[i] / vol_max,
                   raw_sas[i]  / sa_max,
                   bboxes[i][0] / bbox_max,
                   bboxes[i][1] / bbox_max,
                   bboxes[i][2] / bbox_max]
            )
            node_feats.append(feat)

        # ── Edge detection (shared bounding surface = mate contact) ────────
        src, dst, eattr = [], [], []
        n = len(volumes)

        for i in range(n):
            for j in range(i + 1, n):
                if body_surfs[i] & body_surfs[j]:
                    # bidirectional
                    src  += [i, j];  dst  += [j, i]
                    eattr += [[0.0, 1.0], [0.0, 1.0]]   # coincident contact

        if not src:
            # No shared surfaces → fully connect as fallback
            for i in range(n):
                for j in range(n):
                    if i != j:
                        src.append(i); dst.append(j)
                        eattr.append([1.0, 1.0])          # "other" contact

        x          = torch.tensor(node_feats, dtype=torch.float)
        edge_index = torch.tensor([src, dst],  dtype=torch.long)
        edge_attr  = torch.tensor(eattr,        dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    except Exception as exc:
        print(f"    [skip] {Path(step_path).name}: {exc}")
        return None

    finally:
        gmsh.finalize()


# ── Synthetic data fallback ───────────────────────────────────────────────────

def _synthetic_graph(n_nodes: int = None) -> Data:
    """Generate one random assembly graph for testing when no STEP files exist."""
    rng      = random.Random()
    n        = n_nodes or rng.randint(4, 20)
    node_dim = 13

    x = torch.zeros(n, node_dim)
    for i in range(n):
        t = rng.randint(0, 7)
        x[i, t] = 1.0                                    # type one-hot
        x[i, 8:] = torch.rand(5)                         # geometry features

    # Random connected graph
    src, dst, eattr = [], [], []
    perm = list(range(n)); rng.shuffle(perm)
    for k in range(n - 1):
        i, j = perm[k], perm[k + 1]
        mt = rng.randint(0, 5) / 5.0
        src += [i, j]; dst += [j, i]
        eattr += [[mt, 1.0], [mt, 1.0]]

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr  = torch.tensor(eattr,      dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _generate_synthetic(n: int = 300) -> List[Data]:
    print(f"  Generating {n} synthetic assembly graphs for training…")
    return [_synthetic_graph() for _ in range(n)]


# ── Main dataset class ────────────────────────────────────────────────────────

class AssemblyDataset(InMemoryDataset):
    """
    PyG InMemoryDataset wrapping either real STEP files or synthetic graphs.

    Usage
    -----
    ds = AssemblyDataset(source_dir="...", processed_dir="data/processed")
    """

    def __init__(
        self,
        source_dir: str,
        processed_dir: str,
        force_reload: bool = False,
        transform=None,
        pre_transform=None,
    ):
        self.source_dir = Path(source_dir)
        if force_reload:
            # Check both the raw processed_dir and PyG's nested processed_dir/processed path
            for fname in ["data.pt", "processed/data.pt"]:
                proc = Path(processed_dir) / fname
                if proc.exists():
                    proc.unlink()
        super().__init__(str(processed_dir), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0],
                                            weights_only=False)

    @property
    def raw_file_names(self): return []

    @property
    def processed_file_names(self): return ["data.pt"]

    def download(self): pass

    def process(self):
        step_exts = {".step", ".stp", ".STEP", ".STP"}
        step_files = [p for p in self.source_dir.rglob("*") if p.suffix in step_exts]

        graphs: List[Data] = []
        if step_files:
            print(f"  Found {len(step_files)} STEP file(s) in {self.source_dir}")
            for sf in sorted(step_files):
                print(f"  Parsing: {sf.name}")
                g = _parse_step(str(sf))
                if g is not None and g.num_nodes >= 2:
                    graphs.append(g)
            print(f"  Parsed {len(graphs)} valid assembly graph(s).")
        else:
            print(f"  No STEP files found in {self.source_dir}.")

        if len(graphs) < 10:
            graphs += _generate_synthetic(max(100, 300 - len(graphs)))

        data, slices = self.collate(graphs)
        torch.save((data, slices), self.processed_paths[0])
        print(f"  Saved {len(graphs)} graphs → {self.processed_paths[0]}")


# ── Split helper ──────────────────────────────────────────────────────────────

def get_splits(dataset: AssemblyDataset, cfg: dict):
    """
    Split the dataset into train / val / test lists, then apply
    RandomLinkSplit to each individual graph.

    RandomLinkSplit must be applied per-graph (it expects a Data object,
    not an InMemoryDataset).  Graphs with fewer than 8 directed edges are
    skipped to ensure the split can always allocate at least one val/test edge.
    """
    n       = len(dataset)
    n_test  = max(1, int(n * cfg["data"]["test_ratio"]))
    n_val   = max(1, int(n * cfg["data"]["val_ratio"]))
    n_train = max(1, n - n_val - n_test)

    torch.manual_seed(42)
    perm = torch.randperm(n).tolist()

    splitter = RandomLinkSplit(
        num_val                    = 0.1,
        num_test                   = 0.1,
        is_undirected              = True,
        add_negative_train_samples = True,
        neg_sampling_ratio         = cfg["training"]["neg_ratio"],
    )

    def _transform(indices: list, split_idx: int) -> List[Data]:
        result = []
        for i in indices:
            data = dataset[i]
            if data.edge_index.size(1) < 8:   # need enough edges to split
                continue
            try:
                train_d, val_d, test_d = splitter(data)
                result.append([train_d, val_d, test_d][split_idx])
            except Exception:
                continue
        return result

    train_data = _transform(perm[:n_train],             0)
    val_data   = _transform(perm[n_train:n_train+n_val], 1)
    test_data  = _transform(perm[n_train+n_val:],        2)

    print(f"  Splits — train: {len(train_data)}  val: {len(val_data)}"
          f"  test: {len(test_data)}  (skipped graphs with <8 edges)")
    return train_data, val_data, test_data
