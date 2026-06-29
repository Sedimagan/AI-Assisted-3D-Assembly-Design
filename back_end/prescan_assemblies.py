"""
prescan_assemblies.py — Pre-scan STEP files with hard subprocess timeout.
Quarantines any assembly that hangs or errors within TIMEOUT seconds.
Run: python prescan_assemblies.py
"""
import subprocess, sys, os, json
from pathlib import Path

TIMEOUT   = 90          # seconds per file
BASE      = Path("/Users/mbp/Documents/MTECH/Sem4/Individual_project/AI_Assisted_3D_Assembly_Design/AI-Assisted-3D-Assembly-Design/Source_3d_models/Best_models_for_training")
ME_DIR    = BASE / "Mechanical_Engineering_Medium_Nodes_Medium_Edges"
QDIR      = BASE / "_quarantine_stall"
PYTHON    = sys.executable

PROBE = """
import sys, gmsh, warnings
warnings.filterwarnings('ignore')
step = sys.argv[1]
gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 0)
gmsh.model.occ.importShapes(step)
gmsh.model.occ.synchronize()
vols = gmsh.model.getEntities(3)
print(f"OK {len(vols)}")
gmsh.finalize()
"""

QDIR.mkdir(exist_ok=True)

dirs = sorted(d for d in ME_DIR.iterdir() if d.is_dir())
total = len(dirs)
quarantined = []
ok_count = 0

print(f"Pre-scanning {total} assemblies with {TIMEOUT}s timeout each …", flush=True)

for i, d in enumerate(dirs, 1):
    step = d / "assembly.step"
    if not step.exists():
        print(f"  [{i}/{total}] SKIP (no assembly.step): {d.name}", flush=True)
        continue

    try:
        result = subprocess.run(
            [PYTHON, "-c", PROBE, str(step)],
            timeout=TIMEOUT,
            capture_output=True, text=True
        )
        if result.returncode == 0 and result.stdout.strip().startswith("OK"):
            ok_count += 1
            print(f"  [{i}/{total}] OK  {d.name}", flush=True)
        else:
            err = (result.stderr or result.stdout)[:120].replace('\n', ' ')
            print(f"  [{i}/{total}] ERR {d.name}: {err}", flush=True)
            dest = QDIR / d.name
            if not dest.exists():
                d.rename(dest)
                quarantined.append(d.name)
    except subprocess.TimeoutExpired:
        print(f"  [{i}/{total}] HANG {d.name} (>{TIMEOUT}s) — quarantining", flush=True)
        dest = QDIR / d.name
        if not dest.exists():
            d.rename(dest)
        quarantined.append(d.name)

print(f"\nDone. OK={ok_count}  quarantined={len(quarantined)}")
print(f"Remaining: {len(list(ME_DIR.iterdir()))} assemblies")
if quarantined:
    print("Quarantined:")
    for q in quarantined:
        print(f"  {q}")
