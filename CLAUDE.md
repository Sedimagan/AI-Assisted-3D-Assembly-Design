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

bash start_services.sh            # front-end :8501 + back-end :8000 (from .env)
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

Key files in `front_end/`: `app.py` (UI), `gnn_background.py` (animated backdrop), `dark_theme.py` (inline-colour mapper). Key files in `back_end/`:
- `dataset.py` — STEP → graph pipeline (node/edge feature extraction)
- `model.py` — `AssemblyGNN`, `LinkPredictor`, `NodeRanker`
- `train.py` — Phase 1 training loop (5-fold CV, early stopping, checkpoint promotion gating)
- `train_ranker.py` — Phase 2 NodeRanker training (leave-one-node-out task, BPR loss); loads the frozen Phase 1 encoder, trains only its own projection layer
- `evaluate.py` — AUC, AP, Hit@K, MRR, NDCG@K
- `infer.py` — inference on partial assemblies (`--demo` for synthetic, `--step <file>` for real STEP input)
- `skills_agent.py` — Gemini AI orchestrator (AIDA)
- `slot_detector.py` — WHERE a missing part goes: finds empty holes on the host bodies (plus n-fold symmetry for seats hidden under fitted parts). Tool-post calibrated and gated behind `looks_like_toolpost()`; other categories fall through to unlabelled slots.
- `rebuild.py` — WHAT it looks like: single source of geometry for BOTH viewer groups ("Suggested Shapes" and "Reconstructed"). Copies a surviving instance from the upload when one exists; otherwise takes exact geometry from the most similar corpus tool posts. `build_supplement.py` builds `data/part_bank_supplement/` (springs).
- `config.yaml` — all hyperparameters and data paths; **the authoritative source of truth for feature dims, training config, and ranker config** — check this before trusting numbers in README/GETTING_STARTED, which can lag behind

Checkpoints: `back_end/checkpoints/` (serving model, e.g. `best_serving.pt`) and `back_end/../trained_models/` (timestamped exports, filename encodes AUC, e.g. `assembly_gnn_<timestamp>_auc<value>.pt`). NodeRanker uses a **separate** checkpoint (`node_ranker.pt`) — the Phase 1 serving checkpoint is never touched by ranker training; re-run the ranker if Phase 1 is ever retrained.

## Conventions & things to know

- No test suite, linter, or formatter config currently in the repo (no pytest/flake8/pyproject.toml) — don't assume `pytest` or `ruff` exist; check before invoking.
- `config.yaml` node/edge feature dims (22/6) are current; older docs (README, GETTING_STARTED) may reference earlier dims (e.g. 21 or 13) — trust `config.yaml` and recent git log/commit messages over prose docs when they disagree.
- Training data lives under `Source_3d_models/` (drop `.step`/`.stp` files anywhere inside; re-run with `--force-reload` to reprocess). Falls back to synthetic graphs if no valid STEP files are found.
- Phase 1 partial-graph/edge masking is done by `RandomLinkSplit` in `dataset.py`'s `get_splits()` (`num_val`/`num_test`/`disjoint_train_ratio`), not by a `mask_ratio` config key — an earlier `mask_ratio` entry in `config.yaml` was dead (never read by any code) and was removed 2026-08; `ranker.n_per_graph` controls leave-one-node-out samples per graph per epoch (Phase 2).
- **Ports come from `.env`: front-end `:8501`, back-end `:8000`** (`FRONTEND_PORT` and `BACKEND_URL`). `start_services.sh` / `stop_services.sh` fall back to the same values when `.env` is absent. Verify what is actually bound with `lsof -nP -iTCP -sTCP:LISTEN | grep Python`.
- Historical note, because it explains older logs and any long-running process you may find on `:8000`: until 2026-09-17 both scripts loaded `.env` via `source <(...)` process substitution, which silently sets nothing under bash 3.2.57 (macOS stock bash, which `#!/usr/bin/env bash` resolves to here). `.env` was therefore ignored outright and the 8501/8000 fallbacks always won. Both scripts now write the filtered lines to a temp file and source that instead, so `.env` is genuinely applied. `.env` was then set to `BACKEND_URL=http://localhost:8000` so the effective ports are unchanged by the fix.
- That bug never affected Gemini/skills config: `api.py` and `skills_agent.py` call `python-dotenv`'s `load_dotenv()` themselves, so `GEMINI_API_KEY` and friends always loaded. Only shell-level vars were affected.
- `front_end/app.py` does **not** call the back-end over HTTP. It loads checkpoints from disk and runs inference in a subprocess (avoiding gmsh + PyTorch signal-handler conflicts). `BACKEND_URL` is read only by the start/stop scripts. The FastAPI service is an independent API surface, not a dependency of the Streamlit app — which is why the port mismatch above went unnoticed for so long.
- Don't hardcode ports in code; read from env.
- **The part bank is a lossy proxy, not real geometry.** `data/part_bank/` stores heavily decimated meshes: measured on the side lever, the bank copy has 336 faces enclosing 3,102 mm^3 against the real solid's 13,936 faces and 9,024 mm^3 (34% of the volume). Plain primitives survive that; complex parts do not. So `rebuild.py` uses the bank only to decide WHICH design to use, then re-reads that design's real solid from the corpus STEP file (`exact_mesh()`, cached in `data/part_bank_exact/`) and falls back to the coarse bank mesh only if the STEP is unavailable. Do not build anything accuracy-sensitive directly on bank meshes.
- **The bank also silently excludes springs** (its task-18 gate rejects non-watertight meshes, and a helical spring's swept surface never is) — zero of the 11,954 bank parts are spring-sized. Hence the separate supplementary library above.
- **Bank `center` is a part's position in its ORIGINAL assembly frame.** Never use it as an absolute position: that "matches" any model drawn from the corpus by reading back its own pose. `rebuild.template_center()` uses it only as an offset from a host body the upload also contains, for the two parts with no geometric constraint (central shaft, knob); sibling tool posts agree on those offsets to ~0.05 mm.
- **Retrieve by similar assembly, not by vote.** The corpus has two different side-lever designs, three assemblies each; a plain majority vote tied and picked the wrong one (10.9 mm shape error). `similar_assemblies()` scores bank assemblies against the upload's size signatures — the right design lives in assemblies scoring 0.86, the wrong one in assemblies scoring 0.0.
- `Test_3D_models/Partial_tool_post.step` is derived from `Source_3d_models/.../Tool_Post_10` (so the source IS in the bank). Score against the original with `Review_files/toolpost_reconstruction/eval_rebuild.py`, which also runs a leave-one-out variant excluding Tool_Post_10; the two give identical numbers because siblings 25/26 supply the same designs. Current result: mean position error 0.28 mm, shape 0.39 mm, against a ~0.2-0.6 mm noise floor. If the test file is edited (its springs were removed on 2026-09-17), re-run it.
- **The UI is a dark (navy) theme with a live GNN backdrop; the theme is switched by `.streamlit/config.toml` `[theme] base` alone.** Three pieces cooperate, all in `front_end/`:
  - `dark_theme.py` — `install()` (called at the top of `app.py`, before any HTML is emitted) wraps `st.markdown` / `DeltaGenerator.markdown` so every `unsafe_allow_html` snippet has its inline CSS colours mapped to dark equivalents by role (text lightened / greys flipped, pale backgrounds -> dark tinted surfaces, pale borders -> subtle lines; hue kept so red/green/blue status colours keep their meaning; saturated badge fills untouched). The app's ~70 inline colours are authored for a LIGHT page and are never edited per theme; on the light theme `install()` is a no-op. Gotcha: `st.markdown(html)` is called as `(html, ...)` but `slot.markdown(html)` as `(self, html, ...)` — the wrapper transforms the first *string* argument (it once assumed position 0 and silently skipped the Activity Log). A snippet starting `<!--nodark-->` is passed through.
  - `gnn_background.py` — `inject()` draws the animated graph scene on a fixed `<canvas>` (z-index -1) injected into the PARENT page by a zero-height component iframe (same pattern as the auto-scroll script at the end of `app.py`), makes Streamlit's containers transparent, and styles the sidebar/header as frosted glass. It has both a dark and a light palette (chosen from `theme.base`) and keeps `--gnn-sidebar-w` current so the fixed `#proj-header` lines up with the real sidebar edge (it used to be hard-coded `left:21rem`, ~36px off, which only became visible over a non-white page). Bump `VERSION` when editing its JS, otherwise an already-open page keeps running the old script. Respects `prefers-reduced-motion` (one static frame).
  - `app.py` — the 3D viewer's axis/legend text colour is derived from the viewer *background the user picked* (`_fg`), not from the page theme (a white scene on a dark page would otherwise get pale axis text).
  Editing the app's own HTML still means writing light-page colours; they are mapped at render time. Streamlit only reloads `.streamlit/config.toml` and new modules on a restart: `bash stop_services.sh --frontend-only && bash start_services.sh --frontend-only` (the front-end does not hot-reload script changes in this setup). `st.components.v1.html` (used for the backdrop injector and the auto-scroll) is deprecated in Streamlit 1.57 with a removal date already past — migrating to `st.iframe` needs a check that the iframe can still reach `window.parent`.
- `skills/engineering_3d_assembly.yaml` defines AIDA's persona and domain skill areas — extend by editing this YAML, not by changing `skills_agent.py` (it loads profiles dynamically via `SKILLS_PROFILE`).
- Presentation/report artifacts (slide decks, speaker notes, LaTeX report) live in `Review_files/` and the repo root — these are generated deliverables, not source of truth for current model state; check `config.yaml`, `results/`, and git log instead.
