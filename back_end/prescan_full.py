"""
prescan_full.py — Full-pipeline pre-scan using actual _parse_step_with_timeout.
Quarantines any assembly that hangs or errors. Run before training.
Usage: python back_end/prescan_full.py
"""
import subprocess, sys, os
from pathlib import Path

TIMEOUT   = 150          # seconds per file (covers 120s parse + 30s overhead)
BASE      = Path("/Users/mbp/Documents/MTECH/Sem4/Individual_project/AI_Assisted_3D_Assembly_Design/AI-Assisted-3D-Assembly-Design/Source_3d_models/Best_models_for_training")
ME_DIR    = BASE / "Mechanical_Engineering_Medium_Nodes_Medium_Edges"
QDIR      = BASE / "_quarantine_stall"
PYTHON    = sys.executable
BACKEND   = str(Path(__file__).parent)

PROBE = f"""
import sys, io
sys.path.insert(0, {repr(BACKEND)})
import torch
from dataset import _parse_step_with_timeout
step = sys.argv[1]
data, status = _parse_step_with_timeout(step, timeout_secs=90)
if status == "ok" and data is not None:
    print(f"OK {{data.num_nodes}} {{data.num_edges}}")
elif status == "timeout":
    print("TIMEOUT")
    sys.exit(2)
else:
    print(f"SKIP {{status}}")
    sys.exit(3)
"""

QDIR.mkdir(exist_ok=True)

dirs = sorted(d for d in ME_DIR.iterdir() if d.is_dir())
total = len(dirs)
quarantined = []
ok_count = 0
skip_count = 0

print(f"Full-pipeline pre-scan: {total} assemblies, {TIMEOUT}s outer timeout each …", flush=True)

for i, d in enumerate(dirs, 1):
    step = d / "assembly.step"
    if not step.exists():
        skip_count += 1
        print(f"  [{i}/{total}] SKIP (no assembly.step): {d.name}", flush=True)
        continue

    try:
        result = subprocess.run(
            [PYTHON, "-c", PROBE, str(step)],
            timeout=TIMEOUT,
            capture_output=True, text=True
        )
        out = result.stdout.strip()
        if result.returncode == 0 and out.startswith("OK"):
            ok_count += 1
            print(f"  [{i}/{total}] OK  {d.name}  ({out})", flush=True)
        elif result.returncode == 2:
            print(f"  [{i}/{total}] TIMEOUT-inner  {d.name} — quarantining", flush=True)
            dest = QDIR / d.name
            if not dest.exists():
                d.rename(dest)
            quarantined.append(d.name)
        else:
            err = (result.stderr or out)[:100].replace('\n', ' ')
            skip_count += 1
            print(f"  [{i}/{total}] SKIP  {d.name}: {err}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"  [{i}/{total}] HANG  {d.name} (>{TIMEOUT}s outer) — quarantining", flush=True)
        dest = QDIR / d.name
        if not dest.exists():
            d.rename(dest)
        quarantined.append(d.name)

print(f"\nDone. OK={ok_count}  skip={skip_count}  quarantined={len(quarantined)}")
remaining = len(list(ME_DIR.iterdir()))
print(f"Remaining in ME dir: {remaining}")
if quarantined:
    print("Newly quarantined:")
    for q in quarantined:
        print(f"  {q}")
