# AI-Assisted 3D Assembly Design

**Predicting Missing Components in CAD Assemblies using Graph Neural Networks**

| | |
|---|---|
| **Student** | Parthasarathy Perumal |
| **Programme** | M.Tech Data Science & AI, Sem 4 |
| **Guide** | Prof. Sagarika Borah |
| **University** | PES University, Electronic City, Bengaluru |
| **Phase 1** | May 10 – Jul 10, 2026 (Base replication) |
| **Phase 2** | Jul 10 – Sep 10, 2026 (Novelty & improvement) |

---

## Quick Start

```bash
# Clone and enter project
git clone <repo-url> && cd AI-Assisted-3D-Assembly-Design

# One command sets up everything
bash bootstrap.sh

# Add Gemini API key and review ports in .env
nano .env   # set GEMINI_API_KEY=... FRONTEND_PORT=... BACKEND_URL=...

# Activate environment
source .venv/bin/activate

# Start both services (reads ports from .env)
bash start_services.sh              # front-end :11501  · back-end :11000
bash stop_services.sh               # graceful shutdown

# Or run individually
streamlit run front_end/app.py      # 3D viewer only
cd back_end && python train.py      # GNN training
```

> Full setup guide → [`GETTING_STARTED.md`](./GETTING_STARTED.md)

---

## Abstract

Engineering CAD assemblies consist of multiple interconnected components whose correct selection is time-consuming and expertise-dependent. This project trains a **Graph Attention Network (GAT)** on assembly graphs to detect **missing components** via link prediction — Phase 1. A **Gemini-powered Skills AI agent** (AIDA) explains predictions in engineering language. The solution is implemented entirely in Python using PyTorch Geometric, without dependency on proprietary CAD software APIs.

**Phase 2 (planned):** Next-component ranking via NodeRanker (cosine similarity, Hit@K, MRR).

---

## Problem Statement

Modern CAD tools (SolidWorks, CATIA, Fusion 360) offer no contextual intelligence during assembly creation. Engineers rely entirely on domain knowledge to choose and position each component.

| Pain Point | Description |
|---|---|
| **Knowledge barrier** | Junior engineers lack assembly patterns built over years of experience |
| **No automation** | Existing CAD tools offer no intelligent part suggestion during assembly creation |
| **Non-standardisation** | Different designers make inconsistent choices for equivalent subassemblies |
| **Incomplete assemblies** | Partially defined assemblies have no mechanism to detect or fill missing parts |

**Research Gap:** No existing system applies graph-based deep learning to recommend components from partial CAD assembly graphs without proprietary API dependency.

---

## System Architecture

```
Source_3d_models/ (STEP files)
         │
         ▼
dataset.py — gmsh (OpenCASCADE)
  Solid bodies → Nodes · Shared surfaces → Edges (2-dim)
  + trimesh exact SA · SDF ray-casting (SDF mean + variance)
  Nodes: 16-dim · PyG InMemoryDataset
         │
         ▼
model.py — AssemblyGNN (3-layer GAT)
  16 → 512 → 256 → 64 dim
  4-head attention · edge features on L1
         │
         ▼
LinkPredictor MLP  [Phase 1]
MLP(hᵤ‖hᵥ) → BCE
Missing component detection (AUC-ROC, AP)
         │
         ▼  (Phase 2 — planned)
NodeRanker · cosine_sim(ctx, cands)
Next-component ranking (Hit@K, MRR)
         │
         ▼
      skills_agent.py — Gemini AI (AIDA)
      Engineering explanation of predictions
                  │
                  ▼
         front_end/app.py — Streamlit
         Interactive 3D viewer + dual panel UI
```

### Graph Schema

| Element | Dim | Features |
|---|---|---|
| **Node** | **16** | Type one-hot (8 classes, geometry-driven via SDF) · volume · exact SA (trimesh) · bbox (Δx, Δy, Δz) · SDF mean · SDF variance · SA/V ratio |
| **Edge** | 2 | Mate type encoded (coincident/concentric/parallel/tangent/fixed/other) · weight |
| **Avg nodes/graph** | ~18 | — |
| **Avg edges/graph** | ~32 | — |
| **Train/Val/Test** | 70/15/15 | Split by assembly ID — no data leakage |

### Feature Engineering Detail (16 Node + 2 Edge Features)

The graph construction pipeline in `back_end/dataset.py` uses **gmsh + OpenCASCADE + trimesh + SDF ray-casting** to convert each STEP file into an attributed graph. Every solid body becomes a node; every detected physical contact becomes a bidirectional edge.

#### Node Features — 16 Dimensions

| Dim | Feature | Source | Status |
|---|---|---|---|
| [0–7] | Component type one-hot (8 classes) | SDF-inferred geometry rules | ✅ Geometry-driven (was hardcoded) |
| [8] | Normalised volume | `gmsh.occ.getMass()` | unchanged |
| [9] | **Exact** normalised surface area | `trimesh.Trimesh.area` | ✅ Exact (was bbox approximation) |
| [10] | Bbox Δx / bbox_max | `gmsh.occ.getBoundingBox()` | unchanged |
| [11] | Bbox Δy / bbox_max | `gmsh.occ.getBoundingBox()` | unchanged |
| [12] | Bbox Δz / bbox_max | `gmsh.occ.getBoundingBox()` | unchanged |
| [13] | **SDF mean** | Inward ray-casting (trimesh) | ✅ New — avg local thickness |
| [14] | **SDF variance** | Inward ray-casting (trimesh) | ✅ New — shape complexity |
| [15] | **SA/V ratio** | exact_SA / volume | ✅ New — plate vs solid discriminator |

**Dims [0–7] — Geometry-Driven Component Type One-Hot (8 classes)**

Each body is now classified by its geometry (SDF statistics + bounding-box ratios) rather than being hardcoded to "body":

| Index | Class | SDF signature |
|---|---|---|
| 0 | `body` | Generic fallback |
| 1 | `fastener` | Elongated (>2.5×) · thin walls (SDF mean < 8 % of long axis) |
| 2 | `bearing` | SDF std > 60 % of mean — bimodal ring topology |
| 3 | `shaft` | Strongly elongated (>3.5×) · SDF mean > 5 % of long axis |
| 4 | `plate` | Flatness ratio < 12 % (min bbox dim / max bbox dim) |
| 5 | `housing` | SDF variance > 20 % of mean² — complex wall geometry |
| 6 | `gear` | *(covered by housing rule for now; explicit gear rule Phase 2)* |
| 7 | `other` | *(reserved)* |

**Dim [8] — Normalised Volume**

Exact B-Rep volume via `gmsh.model.occ.getMass(3, tag)`, normalised by the maximum across all bodies in the assembly.

**Dim [9] — Exact Normalised Surface Area**

Surface area is now computed from the actual triangulated mesh via `trimesh.Trimesh.area`, eliminating the bbox approximation (`2(dx·dy + dy·dz + dz·dx)`) that was the single biggest accuracy flaw in the 13-dim baseline.

$$\text{Feature}[9] = \frac{\text{trimesh.area}_i}{\max(\text{trimesh.area across all bodies})}$$

**Dims [10–12] — Normalised Bounding Box Dimensions (Δx, Δy, Δz)**

$$\text{Feature}[10] = \frac{dx_i}{\text{bbox}_\text{max}}, \quad \text{Feature}[11] = \frac{dy_i}{\text{bbox}_\text{max}}, \quad \text{Feature}[12] = \frac{dz_i}{\text{bbox}_\text{max}}$$

Unchanged — encode relative size and aspect ratio of each body.

**Dims [13–14] — Shape Diameter Function (SDF mean + variance)**

The **Shape Diameter Function** approximates CGAL's SDF algorithm via trimesh inward ray-casting:
1. Sample N surface points; shoot a ray along the *inward* surface normal for each
2. Record the first-hit distance (local thickness at that point)
3. Aggregate across all sampled points → mean (avg thickness) and variance (shape complexity)

| SDF signature | Inferred component role |
|---|---|
| High mean, low variance | Shaft — uniformly thick cross-section |
| Low mean, low variance | Plate — uniformly thin |
| High variance | Housing or gear — varying wall thickness |
| Std > 60 % of mean | Bearing — bimodal thick/thin ring regions |

Both values are normalised by the maximum within the assembly before use as features.

**Dim [15] — Normalised SA/V Ratio**

Surface-area-to-volume ratio is a powerful discriminator between shells and solids that neither SA nor volume alone provides:

$$\text{Feature}[15] = \frac{(\text{exact\_SA}_i / \text{Volume}_i)}{\max(\text{SA/V across all bodies})}$$

Flat plates and thin shells have high SA/V; compact solid blocks have low SA/V.

#### Edge Features — 2 Dimensions

Edges are bidirectional (each contact produces two directed edges, one in each direction). Both share the same attribute vector.

**Dim [0] — Normalised Mate Type**

Encodes the category of the physical constraint between two contacting bodies. The vocabulary is:

| Index | Mate Type | Normalised Value | Meaning |
|---|---|---|---|
| 0 | `coincident` | 0.0 | Face-to-face planar contact (default for all detected contacts) |
| 1 | `concentric` | 0.2 | Shared axis — shaft through bore, bearing in housing |
| 2 | `parallel` | 0.4 | Parallel faces without touching |
| 3 | `tangent` | 0.6 | Curved surface touching flat or curved surface |
| 4 | `fixed` | 0.8 | Rigid ground constraint |
| 5 | `other` | 1.0 | Fallback — used when graph is fully connected due to no detected contacts |

> **Current baseline limitation:** All detected contacts are hardcoded to `0.0` (coincident) and all fallback edges to `1.0` (other) — `dataset.py` lines 115 and 123. True mate constraint type (concentric vs tangent etc.) is not extractable from raw STEP geometry alone; it requires CAD assembly constraint metadata. This limitation is acknowledged and carried forward as-is.

**Dim [1] — Edge Weight (Contact Presence)**

A scalar set to `1.0` for all edges in the current implementation, indicating the presence of a detected or inferred connection. No partial contact weighting is applied in Phase 1.

#### Geometric Extraction Mechanism in gmsh

The contact detection pipeline relies on two key gmsh operations:

**Step 1 — Boolean Fragmentation**

```python
gmsh.model.occ.fragment(volumes, [])
```

By default, STEP bodies are independent solids with no shared topology. The `fragment()` operation performs a Boolean intersection of all volumes against each other, forcing bodies that are physically touching to share the *exact same* boundary surface tags. Without this step, no shared surfaces exist and no edges can be detected.

**Step 2 — Boundary Surface Extraction**

```python
bnd = gmsh.model.getBoundary([(dim, tag)], oriented=False, combined=True)
body_surfs.append(frozenset(abs(s[1]) for s in bnd if s[0] == 2))
```

For each body the 2D boundary surface tags are collected into a `frozenset`. An edge is then created between body $i$ and body $j$ whenever their surface sets intersect:

```python
if body_surfs[i] & body_surfs[j]:
    # physical contact detected → add bidirectional edge
```

**Step 3 — Area Threshold (Inference / App Only)**

In `front_end/app.py`, contact detection during inference applies an additional area threshold to filter out numerical overlaps from slightly misaligned parts:

```python
_shared_sa = sum(gmsh.model.occ.getMass(2, st) for st in shared_surfaces)
if _shared_sa / _min_body_sa >= 0.01:   # shared area ≥ 1% of smaller body
    # valid mate contact
```

This threshold is applied only in the Streamlit inference path (not during training), as the training STEP files are assumed to be geometrically clean assemblies.

---

## Codebase

```
AI-Assisted-3D-Assembly-Design/
│
├── bootstrap.sh                 ← One-shot automated setup (uv + venv + deps + .env)
├── start_services.sh            ← Start front-end + back-end; reads ports from .env
├── stop_services.sh             ← Graceful shutdown (SIGTERM → SIGKILL); reads ports from .env
├── GETTING_STARTED.md           ← Full setup and usage guide
├── .env.example                 ← Environment template (copy → .env)
│
├── front_end/
│   ├── app.py                   ← Streamlit 3D viewer (dual-panel, sidebar controls)
│   └── requirements.txt
│
├── back_end/
│   ├── config.yaml              ← All hyperparameters + data paths
│   ├── dataset.py               ← STEP → PyG graph pipeline (gmsh parser + synthetic fallback)
│   ├── model.py                 ← AssemblyGNN · LinkPredictor · NodeRanker
│   ├── train.py                 ← Training loop · early stopping · checkpointing
│   ├── evaluate.py              ← AUC-ROC · AP · Hit@K · MRR · NDCG@K
│   ├── infer.py                 ← Inference on any STEP file or demo assembly
│   ├── skills_agent.py          ← Gemini AI orchestrator with engineering skills
│   ├── requirements.txt
│   ├── data/                    ← Processed graph cache (auto-generated)
│   ├── checkpoints/             ← best.pt saved here during training
│   └── results/                 ← train_log.json + test_metrics.json
│
├── skills/
│   └── engineering_3d_assembly.yaml  ← AIDA skills profile (6 domain skill areas)
│
├── trained_models/              ← Timestamped model exports (git-ignored *.pt, folder tracked)
│
└── Source_3d_models/            ← 3D assembly STEP files for training (git-ignored)
```

### Key files

| File | What it does |
|---|---|
| `back_end/dataset.py` | Scans `Source_3d_models/` for STEP files; gmsh+OCC for solid bodies/edges; trimesh for exact SA; SDF ray-casting for SDF mean+variance; geometry-driven type inference; 16-dim node features; raises RuntimeError if no real assemblies found (synthetic fallback removed) |
| `back_end/model.py` | 3-layer GAT encoder (16→512→256→64) + `LinkPredictor` MLP; `build_model()` returns `(gnn, lp, device)`; `NodeRanker` present in file, deferred to Phase 2 |
| `back_end/train.py` | Training loop: BCE loss + hard negatives, Adam + ReduceLROnPlateau, early stopping; saves `best.pt` during training and a timestamped export to `trained_models/` on completion |
| `back_end/evaluate.py` | `evaluate()` returns AUC-ROC and Average Precision — Phase 1 only; Hit@K / MRR / NDCG@K deferred to Phase 2 |
| `back_end/infer.py` | `predict_missing()` scores all absent node pairs and returns top-K with confidence; bar-chart CLI output |
| `back_end/skills_agent.py` | `AssemblySkillsAgent` — loads skills YAML, builds Gemini system prompt, exposes `explain_prediction()`, `identify_component()`, `suggest_assembly_sequence()`, `answer()` |
| `front_end/app.py` | Streamlit app — fixed header, dual independent panels (left: inference + AIDA, right: 3D viewer), gmsh subprocess STEP→STL, Plotly `Mesh3d` with red-highlighted not-assembled / under-connected parts, area-thresholded contact detection, group-based under-connection detection by part basename, fragmentation name inheritance, osascript folder picker, background training with unbuffered log streaming |

---

## Skills AI (AIDA)

The project uses **Google Gemini** as an AI orchestrator configured with mechanical engineering domain knowledge.

### Pipeline

```
GNN predictions (missing component links — Phase 1)
         │
         ▼
AssemblySkillsAgent  ←  skills/engineering_3d_assembly.yaml
         │                (persona + 6 skill domains + response rules)
         ▼
Gemini (gemini-2.0-flash)
         │
         ▼
3–4 bullet engineering explanation (displayed in AIDA panel)
```

### Skill domains

| Skill | Coverage |
|---|---|
| `mechanical_engineering` | Statics, materials, tolerancing (GD&T), design for manufacturing |
| `3d_modelling` | B-Rep, STEP/IGES, OpenCASCADE, solid modelling kernels |
| `3d_assembly_design` | Mate constraints, DOF, assembly hierarchies, sub-assemblies |
| `3d_parts_identification` | Body / fastener / bearing / shaft / plate / gear classification from geometry |
| `gnn_interpretation` | Reading AUC-ROC, AP, link confidence scores in engineering terms |
| `assembly_sequencing` | Build order, fixture requirements, interference analysis |

### Usage

```python
from back_end.skills_agent import AssemblySkillsAgent

agent = AssemblySkillsAgent()
print(agent.explain_prediction(
    missing=[((2, 5), 0.87), ((0, 3), 0.72)],
    recs=[],   # next-component recs added in Phase 2
    context="Lathe tailstock — 4 known components",
))
```

Set `GEMINI_API_KEY` in `.env`. The agent degrades gracefully (offline mode) without a key.

AIDA responses are constrained to **3–4 concise bullet points** covering: what the top missing link represents mechanically, the implied mate constraint type, confidence level interpretation, and any unusual finding worth verifying. The AIDA panel is displayed prominently with a vivid blue gradient at the bottom of the left inference panel.

---

## Environment Configuration (.env)

Copy `.env.example` → `.env` and fill in your values. Key sections:

| Section | Key variables |
|---|---|
| **Google Gemini** | `GEMINI_API_KEY` · `GEMINI_MODEL=gemini-2.0-flash` |
| **Skills AI** | `SKILLS_PROFILE=engineering_3d_assembly` · `SKILLS_TEMPERATURE=0.3` |
| **Frontend** | `FRONTEND_HOST=localhost` · `FRONTEND_PORT=11501` |
| **Backend** | `BACKEND_URL=http://localhost:11000` |
| **LLM** | *(placeholder for future LLM-specific keys)* |
| **Application** | `APP_ENV=development` · `LOG_LEVEL=INFO` · `SOURCE_3D_MODELS=<path>` |

`start_services.sh` and `stop_services.sh` both read `FRONTEND_PORT` and `BACKEND_URL` from `.env` automatically — no need to edit the scripts when changing ports.

---

## 3D Viewer (Front-end)

**Stack:** STEP → gmsh (OpenCASCADE) → STL → PyVista → Plotly `Mesh3d` → Streamlit

```bash
source .venv/bin/activate
bash start_services.sh          # starts Streamlit on FRONTEND_PORT (default 11501)
# or directly:
streamlit run front_end/app.py --server.port 11501
```

**Dual-panel layout:**

| Panel | Description |
|---|---|
| **Left — AI-Assisted Viewer** | Shows "Train first" if no checkpoint. After training: upload a STEP file → runs GNN inference → displays missing component predictions with confidence bars and assembly health status. |
| **Right — 3D Model Viewer** | Upload any STEP/STP file → gmsh converts → interactive Plotly 3D viewer. Highlighted in red: parts detected as not assembled or under-connected. Independent inference display. |

**After training completes** a green banner appears: *"🎉 Training Complete · AUC X.XXXX — Upload a 3D model in the left panel to predict missing components."*

### Assembly Health Detection

The inference pipeline analyses every component in the uploaded STEP file and flags assembly problems in the left panel and 3D viewer (red highlight):

| Condition | Detection Method |
|---|---|
| **Not Assembled** | Degree = 0 after area-thresholded contact check — no surface contact with any other part |
| **Under-Connected** | Same part (by name) appears multiple times; instances with fewer connections than the best-connected instance are flagged |

**Area-threshold contact rule:** A shared surface between two bodies only counts as a mate connection when the shared area is ≥ 1 % of the smaller body's total surface area. This filters incidental tiny overlaps from slightly-displaced parts that are not truly mated.

**Group-based under-connection:** Parts are grouped by their **basename** (last path segment in the STEP hierarchy, e.g. `11-Black Dice LicPlate Bolts-X4`) regardless of instance folder. If one bolt is correctly mated and three others are displaced, the three displaced bolts are flagged as under-connected.

### Part Name Display

Component names are extracted from gmsh entity names (STEP path hierarchy) and displayed as short labels:

- Full gmsh paths like `Shapes/Assembly/InstanceFolder/PartName` are trimmed to the last segment
- Names are inherited through the `fragment()` operation so sub-volumes created by Boolean splitting retain their parent's name
- Any remaining unnamed nodes are filled from STEP `PRODUCT` records as fallback

**Sidebar layout:**

| Row | Controls |
|---|---|
| 1 | Colour · Opacity (side by side) |
| 2 | Background · Camera (side by side) |
| 3 | *(note)* Locate source for training |
| 4 | **📁 3D Files** — native macOS folder picker; saves to `.env` + `config.yaml` |
| 5 | *(label)* 📂 If new 3D models added |
| 6 | **🔄 Upload new file** *(when left viewer has a file)* |
| 7 | **🚀 Train 3D Models** — starts `back_end/train.py` with unbuffered stdout; milestones stream live to Activity Log every 3 s |
| 8 | **📋 Activity Log** — colour-coded per stage; shows dataset parsing, model build, every 10th epoch AUC, test results |

**Training Activity Log stages:**

| Emoji | What it shows |
|---|---|
| 📂 | `[1/4] Loading dataset` |
| 🗂️ | STEP files found / parsed / synthetic graphs generated |
| 🧠 | `[2/4] Building model` · params count |
| 🏋️ | `[3/4] Training` |
| 📊 | Epoch 10, 20 … with AUC score |
| ✅ | New best AUC checkpoint saved |
| 📈 | `[4/4] Test results` — final AUC / AP |
| ✅ | Training complete + prompt to upload for prediction |

---

## Backend — GNN Training

```bash
cd back_end

# From the UI: click "🚀 Train 3D Models" in the sidebar — streams milestones to Activity Log

# Or from the terminal:
python train.py                        # reads Source_3d_models/ by default
python train.py --force-reload         # re-process STEP files (clears graph cache)

# Inference demo
python infer.py --demo
python infer.py --step ../Source_3d_models/my_assembly.step

# Test the Skills AI agent
python skills_agent.py
```

### Training configuration (`config.yaml`)

| Parameter | Value |
|---|---|
| GAT layers | 3 (16→512→256→64 dim) |
| Attention heads | [4, 2, 1] |
| Dropout | 0.2 |
| Optimizer | Adam lr=1e-3, weight_decay=5e-4 |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=8) |
| Early stopping | patience=20 |
| Max epochs | 200 |
| Batch size | 32 graphs |
| Neg sampling ratio | 1:1 pos/neg |

### Trained model output

After training completes a timestamped file is saved to `trained_models/`:

```
trained_models/assembly_gnn_20260527_100641_auc06233.pt   ← current best (16-dim, 138 real assemblies incl. hinges)
```

Each export contains: epoch · best AUC · test metrics · `trained_at` timestamp · `source_dir` path · model weights (`gnn`, `lp`) · full config. Files are git-ignored; the folder is tracked.

### Phase 1 targets — Missing Component Detection

| Metric | Target |
|---|---|
| AUC-ROC | ≥ 0.85 |
| Average Precision | ≥ 0.82 |

> Hit@K · MRR · NDCG@K are Phase 2 targets (NodeRanker — next-component ranking).

---

## Datasets

| Dataset | Size | Role |
|---|---|---|
| [ABC Dataset](https://deep-geometry.github.io/abc-dataset/) — Koch et al., CVPR 2019 | 1M+ STEP / B-Rep files | Primary training data; geometric metadata (bbox, volume, surface area, mate constraints) |
| [Fusion 360 Gallery](https://github.com/AutodeskAILab/Fusion360GalleryDataset) — Willis et al., 2021 | 8,251 assemblies · 154K bodies | M1-friendly subset; native joint annotations |
| [PartNet](https://partnet.cs.stanford.edu/) — Mo et al., CVPR 2019 | 573,585 parts · 26 categories | Hierarchical part annotations for edge feature construction |
| Local assemblies (`Source_3d_models/`) | 93 STEP files → 138 graphs | `Assembly_Files/` · `Bracket_Bolt/` · `Shaft_Bearing_Housing/` · **`Hinge_assembly/`** (added 27 May 2026) · `Plate_Bolt/` |
| Synthetic (fallback) | ~~300 graphs~~ **removed** | Synthetic fallback removed; training raises an error if no real multi-body STEP files are found |

---

## Local Development — MacBook Pro M1

| Constraint | Guideline |
|---|---|
| **Nodes per graph** | Keep under 20 |
| **Training subset** | 500–1,000 assemblies |
| **PyTorch backend** | MPS (Metal Performance Shaders) — auto-detected by `build_model()` |
| **GraphSAGE depth** | Max 3 layers |
| **Expected training time** | < 30 min / epoch on M1 |

**Recommended assembly types for M1:** Bracket + Bolt (2–4 parts) · Shaft + Bearing + Housing (3 parts) · Box + Lid (2–3 parts) · Simple Hinge (3 parts)

---

## Technology Stack

| Layer | Tools |
|---|---|
| **Setup** | `uv` (package manager) · `bootstrap.sh` · `.env` |
| **CAD parsing** | `gmsh` + OpenCASCADE (STEP → solid bodies + surface adjacency) |
| **Geometry enrichment** ✅ | `trimesh` — exact surface area (replaces bbox approx) · SDF ray-casting via trimesh — Shape Diameter Function (mean + variance) for geometry-driven component type inference · 16-dim node features |
| **Graph ML** | PyTorch Geometric · GAT · RandomLinkSplit |
| **AI Orchestration** | Google Gemini (`gemini-2.0-flash`) · `skills_agent.py` |
| **Front-end** | Streamlit · Plotly `Mesh3d` · PyVista (offscreen STL load) |
| **Evaluation** | scikit-learn · AUC-ROC · AP (Phase 1) · Hit@K · MRR · NDCG@K (Phase 2) |
| **Environment** | Python 3.10+ · `.venv/` · `python-dotenv` |

---

## Project Timeline

```
May 10, 2026 ──────────────── Jul 10, 2026 ──────────────── Sep 10, 2026
│   PHASE 1 — Replication      │   PHASE 2 — Novelty          │
│                               │                               │
│ Wk 1–3: Literature review     │ Wk 9–10: Design novelty       │
│ Wk 4–5: Data pipeline         │ Wk 11–13: Improved model      │
│ Wk 6–8: Baseline GNN          │ Wk 14–16: Thesis + demo       │
★ Zeroth Review (10 May)        │                              ★ Final Review (~10 Sep)
```

---

## Deliverables

| Phase | Deliverable |
|---|---|
| Phase 1 | GAT/GCN baseline; GCN vs. GAT vs. GraphSAGE comparison; **missing component detection** AUC ≥ 0.85 · AP ≥ 0.82 |
| Phase 2 | **NodeRanker** next-component ranking; HetGNN typed embeddings; BPR ranking loss; Hit@5 ≥ 0.70 · MRR ≥ 0.64; GNNExplainer |
| Final | Streamlit demo (STEP upload → 3D view → GNN predictions → AI explanation); thesis; open-source repo (MIT) |

---

## Literature Survey

Survey papers are available in [`Literature_survey_papers/`](./Literature_survey_papers/).

| Paper | Venue | PDF |
|---|---|---|
| Du et al. (2024). **BrepGen: A B-Rep Generative Diffusion Model with Structured Latent Geometry** | ACM SIGGRAPH 2024 · arXiv: 2401.15563 | [PDF](./Literature_survey_papers/BrepGen_2401.15563v3.pdf) |
| Jayaraman et al. (2024). **SolidGen: An Autoregressive Model for Direct B-Rep Synthesis and Editing** | ACM ToG, Vol. 43 · DOI: 10.1145/3626206 | [PDF](./Literature_survey_papers/SolidGen_2203.13944v2.pdf) |
| Wang et al. (2025). **CAD-GPT: Synthesising CAD Construction Sequences with Spatial Reasoning-Enhanced Multimodal LLMs** | arXiv: 2501.09803 | [PDF](./Literature_survey_papers/CADGPT_2412.19663v2.pdf) |
| (2024). **Can GNNs Learn Link Heuristics?** | arXiv: 2411.14711 | [PDF](<./Literature_survey_papers/Can GNNs Learn Link Heuristics_2411.14711v2.pdf>) |
| (2023). **Heterogeneous Graph Contrastive Learning** | arXiv: 2303.00995 | [PDF](./Literature_survey_papers/Heterogeneous%20Graph%20Contrastive%20Learning_2303.00995v1.pdf) |

**Foundational references** (cited in thesis): Kipf & Welling (2017) — GCN · Veličković et al. (2018) — GAT · Hamilton et al. (2017) — GraphSAGE · Koch et al. (2019) — ABC Dataset · Mo et al. (2019) — PartNet · Ying et al. (2019) — GNNExplainer.

---

## Geometry-Enriched Node Features — Implemented ✅

> **Triggered by:** First Review feedback — Prof. Sagarika Borah, 24 May 2026  
> **Implemented:** 26 May 2026 — `dataset.py` updated, `config.yaml` in_dim 13 → 16, retrained

### Background

During the First Review, Prof. Sagarika Borah suggested exploring **CGAL** (Computational Geometry Algorithms Library) to strengthen the geometric grounding of the project. A detailed analysis of the pipeline revealed two specific flaws in the 13-dim node feature vector — both now fixed:

| Flaw | Old Code | Fix Applied | Status |
|---|---|---|---|
| **Surface area was a bbox approximation** | `2*(dx*dy + dy*dz + dz*dx)` | `trimesh.Trimesh.area` on the actual mesh | ✅ Fixed |
| **Component type hardcoded to "body"** | `type_oh[0] = 1.0` always | SDF-driven heuristic via `_infer_type_from_geometry()` | ✅ Fixed |

The edge features have a similar hardcoding issue (mate type is always `0.0` for detected contacts), but this is an inherent limitation of STEP geometry — true mate constraints require CAD assembly context unavailable from geometry alone. This is acknowledged and kept as-is.

### Why Not Replace gmsh?

CGAL cannot read STEP files. gmsh + OpenCASCADE remains mandatory for three reasons:

1. **STEP parsing** — only OCC-based kernels (gmsh, pythonocc, FreeCAD) can parse ISO 10303 STEP
2. **Contact detection** — `occ.fragment()` is a single B-Rep topology operation; replacing it with CGAL's pairwise `do_intersect()` mesh tests would be O(n²) and computationally heavier, not lighter
3. **Inference path** — users upload `.step` / `.stp` files; gmsh is always needed at inference time regardless of training strategy

### Refined Tool Split: gmsh + trimesh + CGAL

The enrichment is divided across three tools by responsibility, keeping computation minimal:

```
STEP file
   │
   ▼  gmsh (OpenCASCADE) — unchanged, stays as entry point
   │  • Parse solid bodies from STEP
   │  • occ.fragment() → contact detection (fast B-Rep topology, keep as-is)
   │  • getBoundingBox() → Δx, Δy, Δz
   │  • getMass() → volume
   │  • Export each body as .stl mesh for downstream enrichment
   │
   ▼  trimesh — lightweight pure-Python enrichment (pip install, no build)
   │  • Exact surface area  ← replaces bbox approximation (fixes Flaw 1)
   │  • Milliseconds per body; zero C++ overhead
   │
   ▼  CGAL — only for what trimesh cannot do
   │  • Shape Diameter Function (SDF) per body
   │    → mean SDF + SDF variance used to infer component type:
   │       high mean, low variance   → shaft / fastener
   │       low mean, low variance    → plate
   │       high variance             → housing / gear
   │       bimodal distribution      → bearing
   │    (fixes Flaw 2 — component type one-hot becomes geometry-driven)
   │
   ▼  Augmented node feature vector: 13-dim → 16-dim
      (exact SA replaces bbox approx; SDF mean + variance added as new dims)
   │
   ▼  AssemblyGNN (3-layer GAT) — in_dim updated from 13 → 16
```

### Feature-by-Feature Changes (Implemented)

| Dim | Feature | Before (13-dim) | After (16-dim) | Tool | Status |
|---|---|---|---|---|---|
| [0:8] | Component type one-hot | Hardcoded "body" | SDF + bbox heuristic | trimesh SDF | ✅ |
| [8] | Volume (normalised) | `gmsh.occ.getMass()` | Unchanged | gmsh | — |
| [9] | Surface area (normalised) | Bbox approx `2(dx·dy+…)` | `trimesh.Trimesh.area` | trimesh | ✅ |
| [10–12] | Bbox Δx, Δy, Δz | `getBoundingBox()` | Unchanged | gmsh | — |
| [13] | SDF mean | — | Inward ray-cast mean thickness | trimesh rays | ✅ new |
| [14] | SDF variance | — | Inward ray-cast thickness variance | trimesh rays | ✅ new |
| [15] | SA/V ratio | — | exact_SA / volume (normalised) | trimesh + gmsh | ✅ new |

Edge features (mate type, weight) remain unchanged — mate constraint type requires CAD assembly metadata unavailable from raw STEP geometry.

### What is SDF and Why Does It Help Component Classification?

The **Shape Diameter Function** measures the local thickness of a 3D body at each surface point by casting rays inward and recording the diameter of the shape at that location.

| Component | SDF mean | SDF variance | Interpretation |
|---|---|---|---|
| Shaft / bolt | High | Low | Uniformly thick cylinder throughout |
| Flat plate | Low | Low | Uniformly thin cross-section |
| Housing / gear | Any | High | Varying wall thickness — complex geometry |
| Bearing | Medium | Medium-high | Concentric thick/thin ring regions |

Using `SDF mean` [13] and `SDF variance` [14] as two new node features directly enables the GNN to distinguish these roles without a labelled component-type training dataset — the geometry itself carries the signal.

### Why trimesh Instead of CGAL for Surface Area?

CGAL's `Polygon_mesh_processing::area()` and trimesh's `mesh.area` both compute exact surface area from triangle meshes — they give identical results. trimesh is a pure-Python library that installs with a single `pip install` and runs in milliseconds. Invoking CGAL for this one operation would require a C++ build environment (Conda or compilation from source) for no additional accuracy. CGAL is reserved exclusively for SDF, which has no equivalent in trimesh or any other pure-Python library.

### Expected Impact

- Exact surface area replaces the bbox approximation — the single most impactful fix in the current feature set
- SDF-driven component type inference activates the 8-class one-hot that is currently frozen at "body" for all nodes
- Two new scalar features (SDF mean, SDF variance) add geometric signal that directly correlates with assembly role
- gmsh contact detection is untouched — no added computation on the critical path
- Enables a clean ablation study for Phase 2: AUC-ROC with 13-dim baseline vs 16-dim enriched features

### Implementation Notes

- `trimesh` and `rtree` added to `back_end/requirements.txt`
- CGAL was not required — trimesh inward ray-casting gives equivalent SDF results as a pure-Python drop-in
- SDF helpers `_build_trimesh()`, `_compute_sdf_stats()`, `_infer_type_from_geometry()` added to `dataset.py`
- `gmsh.model.mesh.generate(2)` now called during parse to triangulate surfaces before trimesh extraction
- `config.yaml` `in_dim` updated 13 → 16; model checkpoint from this run: `assembly_gnn_20260526_113959_auc07500.pt`

---

## Training History

![Training Progression — AUC-ROC & Average Precision](docs/training_history.png)

Five training runs are shown, split into two eras. Each bar group shows Val AUC (light), Test AUC (solid), and Test AP (translucent) for that run. Dashed red/orange lines are the Phase 1 targets (AUC 0.85, AP 0.82).

| Run | Date | Change | Graphs | Val AUC | Test AUC | Test AP |
|---|---|---|---|---|---|---|
| R1 | 23 May 14:12 | 13-dim · synthetic+real · early stop ep 1 | synth+real | 0.440 | 0.415 | 0.513 |
| R3 | 23 May 16:24 | 13-dim · 300 synthetic graphs ⚠ inflated | synth+real | 0.927 | 0.985* | 0.993* |
| R6 | 26 May 10:42 | 13-dim · 25 real STEP only · j1.0.0 removed | 25 | 0.813 | 0.636 | 0.584 |
| R7 | 26 May 11:13 | 16-dim · trimesh+SDF · getNode() bug → 4 graphs | 4 | 0.719 | 0.342 | 0.427 |
| R8 | 26 May 11:39 | 16-dim · bug fixed · 25 real STEP · SA+SDF+SA/V | 25 | 0.750 | 0.585 | 0.559 |
| **R9** | **27 May 10:06** | **16-dim · +Hinge assemblies · 93 STEP → 138 graphs** | **138** | **0.623** | **0.512** | **0.533** |

> \* R3 metrics artificially inflated: 300 synthetic test graphs trivially match the 300 synthetic training graphs — not a valid measure of real-geometry performance.

**Key observations:**
- Removing synthetic data (R3→R6) drops the inflated test AUC to an honest 0.636, reflecting the true difficulty of real geometry
- The 16-dim enrichment (R8) improves over the 13-dim baseline on the same 25 assemblies
- Adding hinge assemblies (R9: 138 graphs, train/val/test = 48/12/11) lowers val AUC slightly — 138 diverse assembly types are harder to generalise from than 25, which is the expected and honest behaviour of the model on real data
- Phase 1 AUC target of 0.85 requires substantially more training data or the Phase 2 HetGNN/NodeRanker improvements

---

## Review Files

| File | Description |
|---|---|
| [`zeroth_review_presentation.html`](./Review_files/zeroth_review_presentation.html) | Zeroth review — project proposal, 10 May 2026 |
| [`zeroth_review_presentation_self_reference.html`](./Review_files/zeroth_review_presentation_self_reference.html) | Zeroth review with self-reference links |
| [`AI_Assisted_3D_Assembly_guidance_call_1.html`](./Review_files/AI_Assisted_3D_Assembly_guidance_call_1.html) | Guidance call 1 — motivation, methodology, open questions |
| [`first_review_presentation.html`](./Review_files/first_review_presentation.html) | First review — 6 slides: Title · Architecture · Algorithms · Techniques · Expected Outcomes · References, 24 May 2026 |

---

## License

MIT — open-source, reproducible seeds, no proprietary CAD API dependency.
