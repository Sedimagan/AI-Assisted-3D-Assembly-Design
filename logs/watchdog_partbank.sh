#!/usr/bin/env bash
# Auto-resume watchdog for the Ph2-final-design part-bank rebuild.
# part_bank.py has no resume support at all (single monolithic pass over all
# assemblies, no fold/checkpoint structure) -- on death this restarts the
# whole build from scratch.
set -u
cd "$(dirname "$0")/../back_end"
LOG="../logs/ph2_final_partbank.log"
RESUME_LOG="../logs/watchdog_resume_events.log"
source ../.venv/bin/activate

echo "$(date '+%Y-%m-%d %H:%M:%S')  [watchdog-pb] started, watching $LOG" >> "$RESUME_LOG"

while true; do
    sleep 60

    if grep -q "Wrote .*index.json" "$LOG" 2>/dev/null; then
        echo "$(date '+%Y-%m-%d %H:%M:%S')  [watchdog-pb] completion marker found — run complete, exiting watchdog." >> "$RESUME_LOG"
        break
    fi

    if pgrep -f "part_bank.py" > /dev/null 2>&1; then
        continue
    fi

    echo "$(date '+%Y-%m-%d %H:%M:%S')  [watchdog-pb] part_bank.py not running and no completion marker — restarting from scratch" >> "$RESUME_LOG"

    nohup python3 -u part_bank.py --categories Bench_vice C_Clamps Pipe_vice Gate_Valve Press_Tool Tool_Post Crane_hook >> "$LOG" 2>&1 &
    echo "$(date '+%Y-%m-%d %H:%M:%S')  [watchdog-pb] relaunched, PID $!" >> "$RESUME_LOG"

    sleep 20
done
