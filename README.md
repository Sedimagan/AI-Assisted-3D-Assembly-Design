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
  Solid bodies → Nodes (13-dim)
  Shared surfaces → Edges (2-dim)
  PyG InMemoryDataset
         │
         ▼
model.py — AssemblyGNN (3-layer GAT)
  13 → 512 → 256 → 64 dim
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
| **Node** | 13 | Type one-hot (8 classes: body/fastener/bearing/shaft/plate/housing/gear/other) · volume · surface area · bbox (x, y, z) |
| **Edge** | 2 | Mate type encoded (coincident/concentric/parallel/tangent/fixed/other) · weight |
| **Avg nodes/graph** | ~18 | — |
| **Avg edges/graph** | ~32 | — |
| **Train/Val/Test** | 70/15/15 | Split by assembly ID — no data leakage |

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
└── Source_3d_models/            ← Drop .step / .stp files here
```

### Key files

| File | What it does |
|---|---|
| `back_end/dataset.py` | Scans `Source_3d_models/` for STEP files; uses gmsh + OpenCASCADE to extract solid bodies (nodes) and shared surfaces (edges); falls back to 300 synthetic graphs if folder is empty |
| `back_end/model.py` | 3-layer GAT encoder (13→512→256→64) + `LinkPredictor` MLP; `build_model()` returns `(gnn, lp, device)`; `NodeRanker` present in file, deferred to Phase 2 |
| `back_end/train.py` | Training loop: BCE loss + hard negative sampling, Adam + ReduceLROnPlateau, early stopping, checkpoint saving |
| `back_end/evaluate.py` | `evaluate()` returns AUC-ROC and Average Precision — Phase 1 only; Hit@K / MRR / NDCG@K deferred to Phase 2 |
| `back_end/infer.py` | `predict_missing()` scores all absent node pairs and returns top-K with confidence; bar-chart CLI output |
| `back_end/skills_agent.py` | `AssemblySkillsAgent` — loads skills YAML, builds Gemini system prompt, exposes `explain_prediction()`, `identify_component()`, `suggest_assembly_sequence()`, `answer()` |
| `front_end/app.py` | Streamlit app — fixed header, session-state file upload, gmsh subprocess conversion (STEP → STL), Plotly `Mesh3d` viewer, dual-panel layout, activity log sidebar |

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
Engineering explanation in plain language
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

**Features:**
- Fixed header with project title, student/guide/university info
- File uploader hidden after upload; sidebar "Upload new file" button to reset
- STEP conversion runs in a **subprocess** (bypasses gmsh signal-handler thread restriction)
- Conversion result cached with `@st.cache_data` — sidebar changes (colour, opacity, camera) are instant
- Dual-panel layout: left = uploaded model, right = AI-Assisted viewer (coming soon)
- Sidebar activity log with colour-coded status entries
- Auto-scrolls to viewer after conversion; success banner hides on first pan/zoom

**Sidebar controls:** part colour · background · camera preset (Isometric/Top/Front/Side) · axis grid · opacity · upload new file

---

## Backend — GNN Training

```bash
cd back_end

# First run (synthetic data — no STEP files needed)
python train.py

# Re-process STEP files after adding to Source_3d_models/
python train.py --force-reload

# Inference demo
python infer.py --demo
python infer.py --step ../Source_3d_models/my_assembly.step

# Test the Skills AI agent
python skills_agent.py
```

### Training configuration (`config.yaml`)

| Parameter | Value |
|---|---|
| GAT layers | 3 (13→512→256→64 dim) |
| Attention heads | [4, 2, 1] |
| Dropout | 0.2 |
| Optimizer | Adam lr=1e-3, weight_decay=5e-4 |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=8) |
| Early stopping | patience=20 |
| Max epochs | 200 |
| Batch size | 32 graphs |
| Neg sampling ratio | 1:1 pos/neg |

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
| Synthetic (fallback) | 300 graphs | Auto-generated when `Source_3d_models/` is empty |

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
