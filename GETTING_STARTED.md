# Getting Started

**AI-Assisted 3D Assembly Design**
Predicting Missing & Next Components in CAD Assemblies using Graph Neural Networks

| | |
|---|---|
| **Student** | Parthasarathy Perumal |
| **Guide** | Prof. Sagarika Borah |
| **Programme** | M.Tech DS & AI, Sem 4 — PES University, Electronic City, Bengaluru |

---

## Prerequisites

| Requirement | Minimum | Recommended | Notes |
|---|---|---|---|
| **Python** | 3.10 | 3.11 or 3.12 | Must be on `$PATH` |
| **RAM** | 8 GB | 16 GB | 16 GB+ for large STEP files |
| **Storage** | 4 GB | 8 GB | Models + dataset cache |
| **OS** | macOS 12+ | macOS 14 (M1/M2/M3) | Linux also supported |
| **Internet** | Required at setup | — | For package downloads |
| **Gemini API key** | Optional | Required for AI features | Free tier available |

> **macOS M1/M2/M3 note:** All packages install as native ARM wheels via `uv`. No Rosetta required.

---

## Automated Setup (Recommended)

One command installs everything:

```bash
bash bootstrap.sh
```

The script runs **7 steps** and takes ~3 minutes on a fresh machine:

| Step | What it does |
|---|---|
| **1** | Detects Python 3.10+ (`python3.12` → `python3.10` → `python3`) |
| **2** | Checks write permissions on all project directories |
| **3** | Installs **`uv`** — ultra-fast package manager (10–100× faster than pip) |
| **4** | Creates isolated virtual environment at `.venv/` |
| **5** | Installs all dependencies (front-end, back-end, AI/Gemini, shared) |
| **6** | Creates `.env` from `.env.example` with your local paths pre-filled |
| **7** | Validates the Skills AI profile (`skills/engineering_3d_assembly.yaml`) |

### Bootstrap flags

```bash
bash bootstrap.sh --force-venv   # delete and recreate .venv from scratch
bash bootstrap.sh --skip-uv      # skip uv check (use existing installation)
bash bootstrap.sh --no-color     # plain output (for CI / log files)
```

---

## API Keys Setup

After bootstrap, open `.env` and fill in your keys:

```bash
nano .env          # or: code .env  /  open -e .env
```

### Gemini API key (required for AI features)

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Click **Create API key**
3. Copy the key into `.env`:

```env
GEMINI_API_KEY=AIzaSy...your-key-here...
GEMINI_MODEL=gemini-2.0-flash
```

> The free tier gives 15 requests/minute and 1 million tokens/day — more than enough for development.

---

## Running the Project

Activate the environment first in every new terminal session:

```bash
source .venv/bin/activate
```

### Front-end: 3D Assembly Viewer

```bash
streamlit run front_end/app.py
```

Opens at `http://localhost:8501`.  Upload any `.step` / `.stp` file to view it interactively.

### Back-end: GNN Training

```bash
cd back_end
python train.py                        # train on synthetic data (no STEP files needed)
python train.py --force-reload         # re-process STEP files from Source_3d_models/
```

Training output:
```
[1/4] Loading dataset …
[2/4] Building model …  (total params: 186,432)
[3/4] Training …
  Ep   1/200  loss=0.6931  AUC=0.512  AP=0.498  Hit@5=0.221  MRR=0.134
  Ep   2/200  loss=0.6478  AUC=0.601  ...
  ✓ New best AUC=0.601  checkpoint saved.
  ...
[4/4] Test evaluation …
```

### Back-end: Inference + AI Explanation

```bash
cd back_end
python infer.py --demo                          # synthetic demo assembly
python infer.py --step ../Source_3d_models/x.step   # real STEP file
```

### Skills AI Agent (standalone test)

```bash
cd back_end
python skills_agent.py
```

---

## Adding 3D Models

Place `.step` or `.stp` files anywhere inside:

```
Source_3d_models/
├── brackets/
│   └── bracket_001.step
├── shafts/
│   └── shaft_bearing_housing.step
└── your_assembly.stp
```

Then re-run training with `--force-reload` to reprocess:

```bash
cd back_end && python train.py --force-reload
```

The parser (powered by gmsh + OpenCASCADE) automatically:
- Detects all solid bodies → nodes
- Detects shared surfaces between bodies → edges (mate constraints)
- Extracts 13-dim node features (type, volume, surface area, bbox)
- Falls back to 300 synthetic graphs if no valid STEP files are found

---

## Project Structure

```
AI-Assisted-3D-Assembly-Design/
│
├── bootstrap.sh                 ← Automated setup script
├── GETTING_STARTED.md           ← This file
├── .env.example                 ← Environment template (copy → .env)
├── .env                         ← Your local config (git-ignored)
├── .venv/                       ← Virtual environment (git-ignored)
│
├── front_end/
│   ├── app.py                   ← Streamlit 3D viewer
│   └── requirements.txt
│
├── back_end/
│   ├── config.yaml              ← Hyperparameters + data paths
│   ├── requirements.txt
│   ├── dataset.py               ← STEP → PyG graph pipeline
│   ├── model.py                 ← 3-layer GAT + LinkPredictor + NodeRanker
│   ├── train.py                 ← Training loop
│   ├── evaluate.py              ← AUC, AP, Hit@K, MRR, NDCG
│   ├── infer.py                 ← Inference on partial assemblies
│   ├── skills_agent.py          ← Gemini AI orchestrator
│   ├── data/                    ← Processed graph cache (auto-generated)
│   ├── checkpoints/             ← best.pt saved here
│   └── results/                 ← train_log.json + test_metrics.json
│
├── skills/
│   └── engineering_3d_assembly.yaml   ← Skills AI profile
│
├── Source_3d_models/            ← Drop your STEP files here
├── Literature_survey_papers/
└── Review_files/
```

---

## Skills AI System

The project uses **Google Gemini** as an AI orchestrator with domain-specific engineering skills.

### How it works

```
STEP file
    ↓ gmsh (OpenCASCADE)
Assembly graph (nodes + edges)
    ↓ AssemblyGNN (3-layer GAT)
Node embeddings (64-dim)
    ↓  ┌─────────────────────┐  ┌──────────────────────┐
       │  LinkPredictor MLP  │  │  NodeRanker (cosine) │
       └─────────────────────┘  └──────────────────────┘
       Missing component scores   Next component ranking
    ↓
AssemblySkillsAgent (Gemini)
    ↓ skills/engineering_3d_assembly.yaml
AI explanation in engineering language
```

### Skills profile

The file `skills/engineering_3d_assembly.yaml` defines the AI's persona and six domain skill areas:

| Skill | What it covers |
|---|---|
| `mechanical_engineering` | Statics, materials, tolerancing, GD&T |
| `3d_modelling` | B-Rep, STEP/IGES, OpenCASCADE, solid modelling |
| `3d_assembly_design` | Mate constraints, DOF, assembly hierarchies |
| `3d_parts_identification` | Body, fastener, bearing, shaft, plate, gear classification |
| `gnn_interpretation` | Reading AUC, Hit@K, confidence scores in engineering terms |
| `assembly_sequencing` | Build order, fixture requirements, interference analysis |

### Adding new skills

1. Edit `skills/engineering_3d_assembly.yaml` — add a new entry under `skills:`
2. Alternatively create a new profile YAML and set `SKILLS_PROFILE=your_profile` in `.env`
3. No code changes needed — `skills_agent.py` loads the profile dynamically

---

## Manual Setup (if bootstrap fails)

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env

# 2. Create and activate venv
uv venv .venv --python python3.11
source .venv/bin/activate

# 3. Install dependencies
uv pip install streamlit gmsh pyvista plotly
uv pip install torch torch-geometric numpy scikit-learn tqdm PyYAML
uv pip install google-generativeai python-dotenv

# 4. Create .env
cp .env.example .env
# Edit .env and add GEMINI_API_KEY
```

---

## Troubleshooting

| Error | Fix |
|---|---|
| `Python 3.10+ not found` | Install via Homebrew: `brew install python@3.11` |
| `uv: command not found` | `export PATH="$HOME/.local/bin:$PATH"` then rerun |
| `STEP conversion failed: signal only works in main thread` | Expected in Streamlit — gmsh runs in subprocess automatically |
| `module 'cascadio' has no attribute 'convert'` | Project uses `gmsh` now, not `cascadio` — reinstall with `uv pip install gmsh` |
| `GEMINI_API_KEY not set` | Add key to `.env` — AI features degrade gracefully without it |
| `torch_geometric not found` | Run `uv pip install torch-geometric` separately |
| Sidebar shows scrollbar | Reduce browser zoom level to 90% |

---

## License

MIT — open-source, reproducible seeds, no proprietary CAD API dependency.
