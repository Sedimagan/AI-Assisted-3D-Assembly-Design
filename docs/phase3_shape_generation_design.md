# Phase 3 Design — Missing-Component Shape Generation

**Status:** design only, not implemented. Phase 3 scope per the Phase 2 First Review deck
(`Review_files/Phase2_First_Review_Parthasarathy_Perumal.html`, slides "Path A vs Path B").

**Goal:** given a partial assembly (e.g. a Bench Vice with its handle missing), the system
already answers *something is missing* (LinkPredictor), *it's a handle-type part* (NodeRanker),
and *it belongs here* (Octree open-surface detector). This design adds the last step:
**produce the 3D shape of the missing part** — coarse ("high-level shape") always, refined
detail when retrieval finds a good match.

**Approach:** hybrid **Retrieval-first (Path A) + Voxel-VAE fallback (Path B-lite)** —
sized for the M1 hardware and the current ~131-assembly corpus, not a research-scale
diffusion model.

---

## 1. Pipeline overview

```
Partial assembly STEP
  │
  ├─ dataset.py  ────────────────  22-dim graph  (existing)
  ├─ AssemblyGNN (frozen)  ──────  node embeddings z (N,64)  (existing)
  ├─ NodeRanker  ────────────────  predicted missing type, e.g. "shaft/handle"  (existing)
  ├─ surface_analyzer.py  ───────  open-joint regions (centroid + extent)  (existing)
  │
  ▼  NEW — Stage 2: conditioning
  cond = concat[ ctx(64) · type_onehot(8) · target_bbox(3) · neighbor_scale(2) ]   (77-dim)
  │
  ▼  NEW — Stage 3: hybrid generation
  ShapeRetriever (part bank)  ──→  best-match mesh, fitted to target bbox
        │ (if retrieval confidence < τ, or user asks for "generated" mode)
        ▼
  ConditionalShapeVAE  ──→  32³ occupancy grid ──→ marching cubes ──→ coarse mesh
  │
  ▼  Stage 4: placement & display
  translate/scale mesh to open-joint centroid → ghost overlay in Streamlit 3D viewer
```

Everything above the "NEW" markers already exists and is reused frozen — same principle
as `train_ranker.py`: **never touch `best_serving.pt`**.

---

## 2. New files

| File | Purpose |
|---|---|
| `back_end/part_bank.py` | Build + query the part library (all bodies from all parsed assemblies, normalized) |
| `back_end/shape_generator.py` | `ConditionalShapeVAE`, `ShapeRetriever`, `HybridShapeGenerator`, voxelization utils |
| `back_end/train_shape_gen.py` | Leave-one-part-out training loop for the VAE (mirrors `train_ranker.py`) |
| `back_end/data/part_bank/` | Generated: `index.json` + one `.npz` per part (verts/faces/voxels/metadata) |
| `back_end/checkpoints/shape_vae.pt` | Generated: VAE weights + encoder staleness guard |

Touched (small, additive edits only): `back_end/infer.py`, `back_end/api.py`,
`front_end/app.py`, `back_end/config.yaml`.

---

## 3. `part_bank.py` — the part library

Every parsed assembly already yields per-body trimesh meshes during `dataset.py`'s
`_build_trimesh()` pass — the part bank persists what is currently thrown away.

```python
# Key entry points

def build_part_bank(source_dir: str, out_dir: str = "data/part_bank") -> None:
    """
    Re-walk the parsed corpus (reuses graph_cache / re-parses via _parse_step).
    For each solid body in each assembly:
      1. extract trimesh mesh (existing _build_trimesh)
      2. canonicalize: center at origin, PCA-align axes, scale to unit max-extent
         (store the original scale + transform so retrieval can undo it)
      3. voxelize at 32³ (trimesh.voxel) for VAE training targets
      4. infer comp_type (existing _infer_type_from_geometry)
      5. save <part_id>.npz: verts, faces, voxels, scale, comp_type,
         category, source_assembly
    Write index.json: [{part_id, comp_type, category, scale, n_verts, ...}]
    """

class PartBank:
    def __init__(self, bank_dir: str): ...        # loads index.json lazily
    def query(self, comp_type: str, category: str | None,
              target_extents: np.ndarray, top_k: int = 5) -> list[PartHit]:
        """
        Filter by comp_type (+ category when known), rank by bbox-aspect-ratio
        similarity to target_extents. Returns hits with a fit_score in [0,1].
        """
    def load_mesh(self, part_id: str) -> trimesh.Trimesh: ...
```

Expected size: 126 parseable assemblies × ~12 bodies avg ≈ **1,500 parts**, a few hundred MB
of `.npz` — fine on local disk, `data/` is already git-ignored.

---

## 4. `shape_generator.py` — core module

```python
class ConditionalShapeVAE(nn.Module):
    """
    Encoder:  32³ occupancy → 3D-CNN (4 conv blocks, 32→64→128→256) → μ, σ (latent 128)
    Decoder:  [latent 128 ‖ cond 77] → 3D-deconv → 32³ occupancy logits
    cond = [ctx(64) ‖ type_onehot(8) ‖ target_bbox_norm(3) ‖ neighbor_scale(2)]
    ~2-3M params — comparable to the current AssemblyGNN, trains on MPS.
    """
    def forward(self, vox, cond): ...            # returns recon_logits, mu, logvar
    def generate(self, cond, n_samples=1): ...   # sample z ~ N(0,1), decode

class ShapeRetriever:
    """Thin wrapper over PartBank.query() + fit_to_bbox() (scale + PCA-orient)."""
    def retrieve(self, comp_type, category, target_extents) -> tuple[trimesh.Trimesh, float]:
        ...  # returns (fitted mesh, confidence = fit_score)

class HybridShapeGenerator:
    """
    The single object infer.py talks to.
      result = hsg.generate(graph, gnn, ranker_out, open_joints, mode="auto")
    mode: "auto"      → retrieval if confidence ≥ τ (config, default 0.6), else VAE
          "retrieve"  → force Path A
          "generate"  → force VAE (coarse)
    Returns ShapeResult: mesh (trimesh), source ("retrieved"|"generated"),
                         confidence, part_id | None, placement transform
    """

# utils
def voxelize(mesh, res=32) -> np.ndarray: ...
def devoxelize(occ, threshold=0.5) -> trimesh.Trimesh: ...   # marching cubes (skimage)
def estimate_target_bbox(open_joints, part_bank, comp_type, category) -> np.ndarray:
    """
    Target extents for the missing part:
      primary  — spatial extent of the open-joint cluster (octree region)
      fallback — median extents of same-type parts in this category (part bank)
    """
```

**Conditioning vector construction** (mirrors `NodeRanker`'s context exactly):

```python
z    = gnn(g.x, g.edge_index, g.edge_attr)      # frozen encoder, (N, 64)
ctx  = z.mean(0)                                 # same mean-pool as NodeRanker
cond = torch.cat([ctx, type_onehot, bbox_norm, neighbor_scale])   # (77,)
```

---

## 5. `train_shape_gen.py` — training loop

Mirrors `train_ranker.py`'s structure (frozen encoder, separate checkpoint, cheap epochs):

```
1. Load frozen encoder from checkpoints/best_serving.pt   (assert it exists)
2. Build/refresh part bank if data/part_bank/index.json missing
3. For each epoch, for each training graph:
     for n_per_graph sampled nodes v:
        target_vox = part_bank voxels of body v          (32³)
        partial    = graph with node v removed           (existing remove-node util
                                                          from train_ranker.py)
        cond       = build_cond(partial, true_type(v), bbox(v), neighbors(v))
        recon, mu, logvar = vae(target_vox, cond)
        loss = BCE(recon, target_vox) + λ_dice·DiceLoss + β·KL(mu, logvar)
4. Val metric: voxel IoU + Chamfer distance (sampled surface points)
5. Save checkpoints/shape_vae.pt with:
     {"vae": state_dict, "encoder_trained_at": ..., "encoder_auc": ...}
   — same staleness guard as node_ranker.pt: MUST be retrained whenever
     best_serving.pt is promoted.
```

Data volume: 126 assemblies × ~12 bodies = **~1,500 leave-one-out pairs/epoch**
(×8 with 90° rotation augmentation). Epochs are minutes on MPS, not hours — the encoder
is frozen and voxel grids are tiny.

**Realistic targets** (report honestly, as with NodeRanker):

| Metric | Target | Baseline to beat |
|---|---|---|
| Voxel IoU (val) | ≥ 0.35 | mean-shape-per-type prior (~0.20-0.25) |
| Chamfer (unit-normalized) | ≤ 0.08 | same prior |
| Retrieval fit_score@1 | ≥ 0.6 on held-out assemblies | random same-type pick |

---

## 6. Integration points

### `infer.py` (additive)

```python
def load_shape_generator(bank_dir, vae_path, gnn, device) -> HybridShapeGenerator | None:
    # None if artifacts missing → callers degrade gracefully (same as load_ranker)

def generate_missing_shape(hsg, gnn, graph, ranker_result, open_joints,
                           mode="auto") -> ShapeResult | None:
```

Called **after** `predict_next_component()` — the top-1 ranked type feeds the condition.

### `api.py` (additive)

- startup: load `HybridShapeGenerator` if `checkpoints/shape_vae.pt` + part bank exist;
  `GET /health` gains `"shape_gen_loaded"`.
- new `POST /generate/missing-part` → `{verts, faces, source, confidence, transform}`
  (JSON mesh; the Streamlit client renders Plotly `Mesh3d` from it directly).
- `POST /analyze/step` response gains optional `generated_part` block.

### `front_end/app.py`

The Streamlit subprocess script (`_run_inference()`) — **not** the API — is where the UI
actually calls inference (established in the NodeRanker wiring). Add after the
"🧭 AI-Ranked Next Component" section:

- new section **"🪄 Generated Missing Part"**: source badge (retrieved from `<part_id>` /
  AI-generated coarse), confidence bar
- 3D viewer: render `ShapeResult.mesh` as a **translucent ghost** (`opacity=0.45`,
  distinct color, e.g. violet `#8b5cf6`) positioned at the open-joint centroid,
  with its own legend entry — same overlay pattern as the lime-green open-joint patches.

### `config.yaml` (additive)

```yaml
shape_gen:
  voxel_res:      32
  latent_dim:     128
  cond_dim:       77
  epochs:         60
  lr:             1.0e-3
  beta_kl:        0.05
  lambda_dice:    0.5
  n_per_graph:    8
  retrieval_tau:  0.6        # below this fit_score, fall back to VAE
  part_bank_dir:  "data/part_bank"
```

---

## 7. Build order & verification

| Step | Deliverable | Verify by |
|---|---|---|
| 1 | `part_bank.py` + built bank | spot-check 10 parts render correctly in trimesh; index counts match corpus |
| 2 | `voxelize`/`devoxelize` round-trip | IoU(mesh → vox → mesh) > 0.8 on sample parts |
| 3 | `ShapeRetriever` alone wired into Streamlit | bench-vice-minus-handle demo retrieves a handle-like part — **first demoable milestone, no training needed** |
| 4 | `ConditionalShapeVAE` + smoke train (2 epochs) | loss decreases, generate() produces non-empty grids |
| 5 | full VAE training run | IoU/Chamfer vs. mean-shape baseline table |
| 6 | `HybridShapeGenerator` + API + UI | end-to-end: upload partial STEP → ghost part appears |

Step 3 is the key de-risking move: **retrieval needs zero training** and already delivers
the Prof. Siwal demo ("system proposes the missing handle's shape"). The VAE then upgrades
it from "copies a known handle" to "generates a handle-ish shape" for unmatched cases.

## 8. Known risks / open questions

- **Voxel 32³ is coarse** — long thin parts (handles!) survive PCA-alignment + per-axis
  normalization reasonably well, but verify early on real handle bodies (step 2).
- **Placement orientation**: open-joint centroid gives position; orientation uses the
  retrieved/generated part's PCA axes aligned to the joint region's principal axis —
  approximate, acceptable for a ghost preview, stated as such.
- **Part bank leakage**: for honest eval, retrieval/VAE val splits must exclude parts
  whose *source assembly* is in the val split (split by assembly ID, as train.py already
  does for graphs).
- **Encoder dependency**: `shape_vae.pt` and the part-bank *embeddings* (if added later
  for semantic retrieval) inherit the NodeRanker staleness rule — retrain after any
  `best_serving.pt` promotion.
- **`.DS_Store`/cache noise**: part-bank builder must reuse `dataset.py`'s existing
  exclusion rules (UUID stems, category filter) so bank contents match training reality.
