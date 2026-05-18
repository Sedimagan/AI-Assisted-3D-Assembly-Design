#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  start_services.sh — Start front-end and back-end services
#
#  Usage:  bash start_services.sh
#  Flags:  --no-color   plain output
#          --frontend-only
#          --backend-only
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Colour ────────────────────────────────────────────────────────────────────
if [[ "${*}" == *"--no-color"* ]] || [[ ! -t 1 ]]; then
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

# ── Config ────────────────────────────────────────────────────────────────────
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS_DIR="$PROJ_ROOT/.pids"
LOGS_DIR="$PROJ_ROOT/.logs"
VENV="$PROJ_ROOT/.venv/bin/activate"

FRONTEND_PORT=8501
BACKEND_PORT=8000
STARTUP_DELAY=4          # seconds to wait between frontend and backend start

START_FRONTEND=true
START_BACKEND=true
[[ "${*}" == *"--frontend-only"* ]] && START_BACKEND=false
[[ "${*}" == *"--backend-only"*  ]] && START_FRONTEND=false

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "${CYAN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║       AI-Assisted 3D Assembly Design — Services          ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Pre-flight checks ─────────────────────────────────────────────────────────
cd "$PROJ_ROOT"

if [[ ! -f "$VENV" ]]; then
  error ".venv not found. Run:  bash bootstrap.sh"
fi

# shellcheck disable=SC1090
source "$VENV"
success "Virtual environment activated"

mkdir -p "$PIDS_DIR" "$LOGS_DIR"

# ── Port availability check ───────────────────────────────────────────────────
check_port() {
  local port=$1 name=$2
  if lsof -iTCP:"$port" -sTCP:LISTEN &>/dev/null; then
    warn "Port $port already in use ($name may already be running)"
    warn "Run  bash stop_services.sh  to clear it first."
    return 1
  fi
  return 0
}

$START_FRONTEND && check_port $FRONTEND_PORT "Front-end" || true
$START_BACKEND  && check_port $BACKEND_PORT  "Back-end"  || true

# ── Start front-end (Streamlit) ───────────────────────────────────────────────
start_frontend() {
  info "Starting front-end  →  http://localhost:${FRONTEND_PORT}"
  nohup streamlit run "$PROJ_ROOT/front_end/app.py" \
    --server.port "$FRONTEND_PORT" \
    --server.headless true \
    --server.fileWatcherType none \
    > "$LOGS_DIR/frontend.log" 2>&1 &
  echo $! > "$PIDS_DIR/frontend.pid"
  success "Front-end started  (PID $(cat "$PIDS_DIR/frontend.pid"))  →  http://localhost:${FRONTEND_PORT}"
}

# ── Start back-end (FastAPI / uvicorn) ────────────────────────────────────────
start_backend() {
  info "Starting back-end   →  http://localhost:${BACKEND_PORT}"
  cd "$PROJ_ROOT/back_end"
  nohup python -m uvicorn api:app \
    --host 0.0.0.0 \
    --port "$BACKEND_PORT" \
    --log-level info \
    > "$LOGS_DIR/backend.log" 2>&1 &
  echo $! > "$PIDS_DIR/backend.pid"
  cd "$PROJ_ROOT"
  success "Back-end started   (PID $(cat "$PIDS_DIR/backend.pid"))   →  http://localhost:${BACKEND_PORT}"
}

# ── Launch sequence ───────────────────────────────────────────────────────────
if $START_FRONTEND && $START_BACKEND; then
  start_frontend
  info "Waiting ${STARTUP_DELAY}s before starting back-end…"
  sleep "$STARTUP_DELAY"
  start_backend
elif $START_FRONTEND; then
  start_frontend
elif $START_BACKEND; then
  start_backend
fi

# ── Health poll (give services a moment to bind) ──────────────────────────────
sleep 2
echo ""
echo -e "${BOLD}  Service URLs${NC}"
echo -e "  ${DIM}─────────────────────────────────────────────────${NC}"

if $START_FRONTEND; then
  if lsof -iTCP:$FRONTEND_PORT -sTCP:LISTEN &>/dev/null; then
    echo -e "  ${GREEN}●${NC} Front-end  (Streamlit)  →  ${CYAN}http://localhost:${FRONTEND_PORT}${NC}"
  else
    echo -e "  ${YELLOW}○${NC} Front-end  (Streamlit)  →  still starting…  check .logs/frontend.log"
  fi
fi

if $START_BACKEND; then
  if lsof -iTCP:$BACKEND_PORT -sTCP:LISTEN &>/dev/null; then
    echo -e "  ${GREEN}●${NC} Back-end   (FastAPI)    →  ${CYAN}http://localhost:${BACKEND_PORT}${NC}"
    echo -e "     API docs              →  ${CYAN}http://localhost:${BACKEND_PORT}/docs${NC}"
  else
    echo -e "  ${YELLOW}○${NC} Back-end   (FastAPI)    →  still starting…  check .logs/backend.log"
  fi
fi

echo ""
echo -e "  ${DIM}Logs:  .logs/frontend.log   .logs/backend.log${NC}"
echo -e "  ${DIM}Stop:  bash stop_services.sh${NC}"
echo ""
