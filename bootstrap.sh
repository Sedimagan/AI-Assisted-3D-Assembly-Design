#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  bootstrap.sh — One-shot setup for AI-Assisted 3D Assembly Design
#  PES University, Electronic City, Bengaluru
#
#  Usage:  bash bootstrap.sh
#  Flags:  --skip-uv        skip uv installation check
#          --force-venv     delete and recreate .venv even if it exists
#          --no-color       disable colour output
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
if [[ "${*:-}" == *"--no-color"* ]] || [[ ! -t 1 ]]; then
  RED="" GREEN="" YELLOW="" BLUE="" CYAN="" BOLD="" DIM="" NC=""
else
  RED='\033[0;31m' GREEN='\033[0;32m' YELLOW='\033[1;33m'
  BLUE='\033[0;34m' CYAN='\033[0;36m' BOLD='\033[1m'
  DIM='\033[2m' NC='\033[0m'
fi

info()    { echo -e "${BLUE}  $*${NC}"; }
success() { echo -e "${GREEN}  ✓ $*${NC}"; }
warn()    { echo -e "${YELLOW}  ⚠  $*${NC}"; }
error()   { echo -e "${RED}  ✗ $*${NC}" >&2; exit 1; }
step()    { echo -e "\n${CYAN}${BOLD}[$1/$TOTAL_STEPS] $2${NC}"; }

TOTAL_STEPS=7
SKIP_UV=false
FORCE_VENV=false
[[ "${*:-}" == *"--skip-uv"*   ]] && SKIP_UV=true
[[ "${*:-}" == *"--force-venv"* ]] && FORCE_VENV=true

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "${CYAN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║       AI-Assisted 3D Assembly Design — Bootstrap         ║"
echo "  ║  Predicting Missing & Next Components via Graph AI        ║"
echo "  ║  PES University, Electronic City, Bengaluru               ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Project root = directory containing this script
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJ_ROOT"
info "Project root: $PROJ_ROOT"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Python version check
# ─────────────────────────────────────────────────────────────────────────────
step 1 "Checking Python version (≥ 3.10 required)"

PYTHON_CMD=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$cmd" &>/dev/null; then
    VER=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
    MAJOR="${VER%%.*}"; MINOR="${VER##*.}"
    if [[ "$MAJOR" -ge 3 && "$MINOR" -ge 10 ]]; then
      PYTHON_CMD="$cmd"; break
    fi
  fi
done

if [[ -z "$PYTHON_CMD" ]]; then
  error "Python 3.10+ not found.\n  Install from https://python.org or via Homebrew: brew install python@3.11"
fi
PY_VER=$("$PYTHON_CMD" --version 2>&1)
success "Found $PY_VER at $(command -v "$PYTHON_CMD")"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Write permissions
# ─────────────────────────────────────────────────────────────────────────────
step 2 "Checking write permissions"

for dir in "$PROJ_ROOT" "$PROJ_ROOT/front_end" "$PROJ_ROOT/back_end" \
           "$PROJ_ROOT/back_end/data" "$PROJ_ROOT/back_end/checkpoints" \
           "$PROJ_ROOT/back_end/results"; do
  mkdir -p "$dir"
  if [[ ! -w "$dir" ]]; then
    error "No write permission: $dir\n  Try: chmod -R u+w \"$PROJ_ROOT\""
  fi
done
success "Write permissions OK for all project directories"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Install uv (ultra-fast pip replacement)
# ─────────────────────────────────────────────────────────────────────────────
step 3 "Installing uv package manager"

if $SKIP_UV; then
  warn "--skip-uv flag set; skipping uv installation"
elif command -v uv &>/dev/null; then
  success "uv already installed: $(uv --version)"
else
  info "Downloading and installing uv…"
  if command -v curl &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget &>/dev/null; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  else
    error "Neither curl nor wget found. Install one and retry."
  fi
  # Add uv to PATH for this session
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  # Source uv env if present
  [[ -f "$HOME/.local/bin/env" ]]  && source "$HOME/.local/bin/env"  || true
  [[ -f "$HOME/.cargo/env"    ]]  && source "$HOME/.cargo/env"       || true

  if ! command -v uv &>/dev/null; then
    error "uv installation failed. Add ~/.local/bin to your PATH and retry."
  fi
  success "uv installed: $(uv --version)"
fi

UV_CMD=$(command -v uv 2>/dev/null || echo "")
[[ -z "$UV_CMD" ]] && error "uv not found in PATH. Re-run without --skip-uv."

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Create virtual environment
# ─────────────────────────────────────────────────────────────────────────────
step 4 "Creating virtual environment (.venv/)"

if $FORCE_VENV && [[ -d ".venv" ]]; then
  warn "--force-venv: removing existing .venv/"
  rm -rf .venv
fi

if [[ -d ".venv" ]]; then
  warn ".venv/ already exists — skipping creation (use --force-venv to rebuild)"
else
  uv venv .venv --python "$PYTHON_CMD" --seed
  success "Virtual environment created: $(pwd)/.venv"
fi

# Activate
# shellcheck disable=SC1091
source .venv/bin/activate
success "Virtual environment activated: $VIRTUAL_ENV"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Install all dependencies
# ─────────────────────────────────────────────────────────────────────────────
step 5 "Installing dependencies (uv pip — ultra fast)"

info "Installing shared + AI dependencies…"
uv pip install \
  "python-dotenv>=1.0.0" \
  "PyYAML>=6.0" \
  "google-generativeai>=0.7.0" \
  "numpy>=1.24.0"

info "Installing front-end dependencies…"
uv pip install \
  "streamlit>=1.35.0" \
  "gmsh>=4.11.0" \
  "pyvista>=0.43.0" \
  "plotly>=5.0.0"

info "Installing back-end / ML dependencies…"
uv pip install \
  "torch>=2.0.0" \
  "scikit-learn>=1.3.0" \
  "tqdm>=4.65.0"

# torch-geometric needs special handling (no universal wheel)
info "Installing PyTorch Geometric…"
uv pip install \
  "torch-geometric>=2.3.0" \
  "torch-scatter" "torch-sparse" "torch-cluster" \
  --find-links "https://data.pyg.org/whl/torch-$(python -c 'import torch; print(torch.__version__.split("+")[0])')+cpu.html" \
  2>/dev/null || uv pip install "torch-geometric>=2.3.0"  # fallback without extras

success "All dependencies installed"

# Sanity check
python -c "import streamlit, gmsh, pyvista, plotly, torch, torch_geometric, google.generativeai" \
  2>/dev/null && success "Import check passed" \
  || warn "One or more packages could not be imported — check output above"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Create .env from template
# ─────────────────────────────────────────────────────────────────────────────
step 6 "Configuring environment (.env)"

if [[ -f ".env" ]]; then
  warn ".env already exists — skipping (edit it manually to update keys)"
else
  cp .env.example .env
  success ".env created from .env.example"
fi

# Patch SOURCE_3D_MODELS to absolute path of this project
sed -i.bak \
  "s|SOURCE_3D_MODELS=.*|SOURCE_3D_MODELS=$PROJ_ROOT/Source_3d_models|" \
  .env && rm -f .env.bak

success "SOURCE_3D_MODELS path set to: $PROJ_ROOT/Source_3d_models"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Verify skills profiles
# ─────────────────────────────────────────────────────────────────────────────
step 7 "Verifying skills profiles"

SKILLS_DIR="$PROJ_ROOT/skills"
SKILL_COUNT=$(find "$SKILLS_DIR" -name "*.yaml" 2>/dev/null | wc -l | tr -d ' ')

if [[ "$SKILL_COUNT" -gt 0 ]]; then
  success "Found $SKILL_COUNT skill profile(s) in skills/"
  find "$SKILLS_DIR" -name "*.yaml" | while read -r f; do
    info "  • $(basename "$f")"
  done
else
  warn "No skill profiles found in skills/ — run will use fallback responses"
fi

# Verify skills_agent.py can load the profile (no Gemini key needed)
python -c "
import sys; sys.path.insert(0, 'back_end')
import yaml; from pathlib import Path
p = Path('skills/engineering_3d_assembly.yaml')
d = yaml.safe_load(p.read_text())
print('  Profile:', d['name'])
print('  Skills: ', ', '.join(s['id'] for s in d['skills']))
" 2>/dev/null && success "Skills profile validated" \
  || warn "Could not validate skills profile — check skills/ directory"

# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
echo -e "\n${GREEN}${BOLD}  ══════════════════════════════════════════════════════════"
echo    "  ✓  Bootstrap complete!"
echo -e "  ══════════════════════════════════════════════════════════${NC}\n"

echo -e "${BOLD}  Next steps:${NC}"
echo -e "  ${DIM}1.${NC} Add your Gemini API key:"
echo -e "     ${BOLD}nano .env${NC}  →  set GEMINI_API_KEY=<your-key>"
echo -e "     Get a key at: ${CYAN}https://aistudio.google.com/app/apikey${NC}\n"
echo -e "  ${DIM}2.${NC} Activate the environment (new terminal sessions):"
echo -e "     ${BOLD}source .venv/bin/activate${NC}\n"
echo -e "  ${DIM}3.${NC} Run the front-end viewer:"
echo -e "     ${BOLD}streamlit run front_end/app.py${NC}\n"
echo -e "  ${DIM}4.${NC} Train the GNN backend:"
echo -e "     ${BOLD}cd back_end && python train.py${NC}\n"
echo -e "  ${DIM}5.${NC} Run inference + AI explanation:"
echo -e "     ${BOLD}cd back_end && python infer.py --demo${NC}\n"
echo -e "  ${DIM}6.${NC} Test the Skills AI agent:"
echo -e "     ${BOLD}cd back_end && python skills_agent.py${NC}\n"
echo -e "${DIM}  Deactivate venv when done: deactivate${NC}\n"
