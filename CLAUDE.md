# CLAUDE.md

Guidance for Claude Code (or any Claude agent) working in this repository.

## Project

**AI-Assisted 3D Assembly Design** — predicting missing and next components in CAD assemblies using Graph Neural Networks. MTech Data Science & AI individual project, PES University Bengaluru (Parthasarathy Perumal, guide Prof. Sagarika Borah for Phase 1 / Prof. Gaurav Siwal for Phase 2).

- Phase 1 (May–Jul 2026, complete): missing-component detection via link prediction.
- Phase 2 (Jul–Sep 2026, in progress): next-component ranking via NodeRanker, plus a planned heterogeneous GNN and GNNExplainer integration.

Current git branch: `ph2-node-ranker-review1`.

## Setup & running

```bash
bash bootstrap.sh                 # one-shot setup: uv, .venv, deps, .env, skills validation
source .venv/bin/activate         # activate before any command below

bash start_services.sh            # front-end :11501 + back-end :11000 (reads .env)
bash stop_services.sh             # graceful shutdown

streamlit run front_end/app.py    # front-end only
cd back_end && python train.py    # GNN training only
```

Required env var: `GEMINI_API_KEY` in `.env` (copy from `.env.example`). AI explanation features degrade gracefully without it; everything else still works.

## Architecture

```
STEP file (Source_3d_models/)
  → dataset.py: gmsh + OpenCASCADE + trimesh + SDF ray-casting
    → assembly graph: 22-dim nodes, 6-dim edges (PyG InMemoryDataset)
  → model.py: AssemblyGNN, 3-layer GAT (22 → 128 → 64), heads [8,4,1]
    → LinkPredictor MLP        → missing-component detection (Phase 1, AUC/AP)
    → NodeRanker (cosine sim)  → next-component ranking (Phase 2, Hit@K/MRR/NDCG@K)
  → assembly_templates.py (AssemblyTemplateDB) — per-category component-type frequency templates
  → surface_analyzer.py (Octree open-surface detector) — flags likely locations of missing parts on the 3D model
  → skills_agent.py — Gemini-based agent ("AIDA") explains predictions in engineering language,
    persona/skills defined in skills/engineering_3d_assembly.yaml
  → back_end/api.py (FastAPI) + front_end/app.py (Streamlit dual-panel UI)
```

Key files in `back_end/`:
- `dataset.py` — STEP → graph pipeline (node/edge feature extraction)
- `model.py` — `AssemblyGNN`, `LinkPredictor`, `NodeRanker`
- `train.py` — Phase 1 training loop (5-fold CV, early stopping, checkpoint promotion gating)
- `train_ranker.py` — Phase 2 NodeRanker training (leave-one-node-out task, BPR loss); loads the frozen Phase 1 encoder, trains only its own projection layer
- `evaluate.py` — AUC, AP, Hit@K, MRR, NDCG@K
- `infer.py` — inference on partial assemblies (`--demo` for synthetic, `--step <file>` for real STEP input)
- `skills_agent.py` — Gemini AI orchestrator (AIDA)
- `config.yaml` — all hyperparameters and data paths; **the authoritative source of truth for feature dims, training config, and ranker config** — check this before trusting numbers in README/GETTING_STARTED, which can lag behind

Checkpoints: `back_end/checkpoints/` (serving model, e.g. `best_serving.pt`) and `back_end/../trained_models/` (timestamped exports, filename encodes AUC, e.g. `assembly_gnn_<timestamp>_auc<value>.pt`). NodeRanker uses a **separate** checkpoint (`node_ranker.pt`) — the Phase 1 serving checkpoint is never touched by ranker training; re-run the ranker if Phase 1 is ever retrained.

## Conventions & things to know

- No test suite, linter, or formatter config currently in the repo (no pytest/flake8/pyproject.toml) — don't assume `pytest` or `ruff` exist; check before invoking.
- `config.yaml` node/edge feature dims (22/6) are current; older docs (README, GETTING_STARTED) may reference earlier dims (e.g. 21 or 13) — trust `config.yaml` and recent git log/commit messages over prose docs when they disagree.
- Training data lives under `Source_3d_models/` (drop `.step`/`.stp` files anywhere inside; re-run with `--force-reload` to reprocess). Falls back to synthetic graphs if no valid STEP files are found.
- `mask_ratio` in `config.yaml` controls edge masking fraction for partial-graph training (Phase 1); `ranker.n_per_graph` controls leave-one-node-out samples per graph per epoch (Phase 2).
- Ports and Gemini/skills config are all driven from `.env` (see `.env.example` for the full list) — don't hardcode ports; read from env.
- `skills/engineering_3d_assembly.yaml` defines AIDA's persona and domain skill areas — extend by editing this YAML, not by changing `skills_agent.py` (it loads profiles dynamically via `SKILLS_PROFILE`).
- Presentation/report artifacts (slide decks, speaker notes, LaTeX report) live in `Review_files/` and the repo root — these are generated deliverables, not source of truth for current model state; check `config.yaml`, `results/`, and git log instead.
