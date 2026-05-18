#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  stop_services.sh — Stop front-end and back-end services
#
#  Usage:  bash stop_services.sh
#  Flags:  --no-color
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

# ── Config ────────────────────────────────────────────────────────────────────
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDS_DIR="$PROJ_ROOT/.pids"

FRONTEND_PORT=8501
BACKEND_PORT=8000
GRACE_PERIOD=3     # seconds before SIGKILL

STOP_FRONTEND=true
STOP_BACKEND=true
[[ "${*}" == *"--frontend-only"* ]] && STOP_BACKEND=false
[[ "${*}" == *"--backend-only"*  ]] && STOP_FRONTEND=false

# ── Banner ────────────────────────────────────────────────────────────────────
echo -e "${CYAN}${BOLD}"
echo "  ╔══════════════════════════════════════════════════════════╗"
echo "  ║     AI-Assisted 3D Assembly Design — Stop Services       ║"
echo "  ╚══════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# ── Kill helper ───────────────────────────────────────────────────────────────
# kill_service <name> <pid_file> <port>
kill_service() {
  local name="$1" pid_file="$2" port="$3"
  local killed=false

  # 1. Try PID file first
  if [[ -f "$pid_file" ]]; then
    local pid
    pid=$(cat "$pid_file")
    if kill -0 "$pid" 2>/dev/null; then
      info "Stopping $name (PID $pid) …"
      kill -TERM "$pid" 2>/dev/null || true
      # Wait for graceful shutdown
      local waited=0
      while kill -0 "$pid" 2>/dev/null && [[ $waited -lt $GRACE_PERIOD ]]; do
        sleep 1; (( waited++ ))
      done
      # Force kill if still alive
      if kill -0 "$pid" 2>/dev/null; then
        warn "$name did not stop in ${GRACE_PERIOD}s — sending SIGKILL"
        kill -KILL "$pid" 2>/dev/null || true
        sleep 1
      fi
      killed=true
    else
      warn "$name PID $pid not found (already stopped?)"
    fi
    rm -f "$pid_file"
  fi

  # 2. Fallback: kill by port (catches orphaned processes)
  if lsof -iTCP:"$port" -sTCP:LISTEN &>/dev/null; then
    local port_pids
    port_pids=$(lsof -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)
    if [[ -n "$port_pids" ]]; then
      info "Clearing port $port (PIDs: $port_pids) …"
      echo "$port_pids" | xargs kill -TERM 2>/dev/null || true
      sleep "$GRACE_PERIOD"
      if lsof -iTCP:"$port" -sTCP:LISTEN &>/dev/null; then
        echo "$port_pids" | xargs kill -KILL 2>/dev/null || true
      fi
      killed=true
    fi
  fi

  if $killed; then
    success "$name stopped"
  else
    warn "$name was not running"
  fi
}

# ── Stop services ─────────────────────────────────────────────────────────────
$STOP_BACKEND  && kill_service "Back-end  (FastAPI)"    "$PIDS_DIR/backend.pid"  "$BACKEND_PORT"
$STOP_FRONTEND && kill_service "Front-end (Streamlit)"  "$PIDS_DIR/frontend.pid" "$FRONTEND_PORT"

# ── Clean up empty pids dir ───────────────────────────────────────────────────
if [[ -d "$PIDS_DIR" ]] && [[ -z "$(ls -A "$PIDS_DIR")" ]]; then
  rmdir "$PIDS_DIR"
fi

# ── Final status ──────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}  Port status after shutdown${NC}"
echo -e "  ${DIM}─────────────────────────────────────────────────${NC}"

for port_name in "$FRONTEND_PORT:Front-end" "$BACKEND_PORT:Back-end"; do
  port="${port_name%%:*}"; name="${port_name##*:}"
  if lsof -iTCP:"$port" -sTCP:LISTEN &>/dev/null; then
    echo -e "  ${YELLOW}●${NC} $name  port $port  still in use"
  else
    echo -e "  ${GREEN}○${NC} $name  port $port  free"
  fi
done

echo ""
echo -e "  ${DIM}Logs retained in .logs/  —  restart with:  bash start_services.sh${NC}"
echo ""
