import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime
from pathlib import Path

# Project root — resolved once here so all helpers can reference it
_PROJ_ROOT  = Path(__file__).resolve().parent.parent
_CKPT_PATH  = _PROJ_ROOT / "back_end" / "checkpoints" / "best.pt"
_STEP_CACHE = _PROJ_ROOT / ".logs" / "inference_input.step"

os.environ["DISPLAY"] = ":99"
import pyvista as pv
pv.OFF_SCREEN = True

import streamlit as st
import streamlit.components.v1
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI-Assisted 3D Assembly Design",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Fixed header + global CSS ─────────────────────────────────────────────────
st.markdown(
    """
    <style>
    #proj-header {
        position: fixed;
        top: 2.875rem;
        left: 21rem;   /* Streamlit sidebar width — centers within the main content area */
        right: 0;
        z-index: 998;
        background: #0d1828;
        border-bottom: 1px solid #1e3a5f;
        box-shadow: 0 2px 8px rgba(0,0,0,0.4);
        padding: 0.85rem 1.5rem 0.3rem;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        gap: 0.2rem;
    }
    div.block-container {
        padding-top: 8.5rem !important;
    }
    /* Left viewer border — box-shadow renders outside the element box so all
       four sides (including bottom) are always fully visible */
    [data-testid="stPlotlyChart"] {
        box-shadow: 0 0 0 1.5px #2a4060;
        border-radius: 8px;
        margin-bottom: 4px;   /* keeps the shadow from being clipped by the row below */
    }
    /* Success alert: push down slightly */
    [data-testid="stAlert"] {
        margin-top: 0.6rem !important;
        transition: opacity 0.5s ease;
    }
    /* ── Sidebar: pull everything to the top ── */
    section[data-testid="stSidebar"] > div:first-child {
        padding-top: 0 !important;
    }
    section[data-testid="stSidebar"] > div:first-child > div:first-child {
        padding-top: 0 !important;
        margin-top:  0 !important;
    }
    /* Zero-out the automatic gap Streamlit injects around every widget */
    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {
        margin-top:    0 !important;
        margin-bottom: 0 !important;
        padding-top:   0 !important;
        padding-bottom:0 !important;
    }
    section[data-testid="stSidebar"] .element-container {
        margin:  0 !important;
        padding: 0 !important;
    }
    /* Tighter dividers */
    section[data-testid="stSidebar"] hr {
        margin: 0.2rem 0 !important;
    }
    /* Smaller widget labels */
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] label {
        font-size: 0.72rem !important;
        line-height: 1.1 !important;
        margin-bottom: 0 !important;
    }
    /* Shrink the colour-picker swatch */
    section[data-testid="stSidebar"] [data-testid="stColorPicker"] {
        padding: 0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stColorPicker"] button {
        width:  1.6rem !important;
        height: 1.6rem !important;
        min-width:  1.6rem !important;
        min-height: 1.6rem !important;
        border-radius: 4px !important;
    }
    /* Compact selectbox */
    section[data-testid="stSidebar"] [data-testid="stSelectbox"] > div > div {
        min-height: 1.8rem !important;
        font-size:  0.77rem !important;
        padding: 0.15rem 0.5rem !important;
    }
    /* Slim slider */
    section[data-testid="stSidebar"] [data-testid="stSlider"] {
        padding-top:    0.1rem !important;
        padding-bottom: 0.1rem !important;
    }
    section[data-testid="stSidebar"] [data-testid="stSlider"] > div {
        padding: 0 !important;
    }
    /* Compact checkbox */
    section[data-testid="stSidebar"] .stCheckbox {
        margin:  0 !important;
        padding: 0 !important;
    }
    /* Compact upload button */
    section[data-testid="stSidebar"] [data-testid="stButton"] button {
        padding: 0.15rem 0.75rem !important;
        font-size: 0.77rem !important;
        line-height: 1.4 !important;
    }
    </style>

    <div id="proj-header">
        <span style="color:#ffffff;font-size:1.5rem;font-weight:700;letter-spacing:0.4px;margin-top:0.4rem;display:block;">
            🔩 AI-Assisted 3D Assembly Design
        </span>
        <span style="color:#a0b8cc;font-size:0.82rem;letter-spacing:0.2px;">
            <strong>Student:</strong> Parthasarathy Perumal &nbsp;·&nbsp;
            <strong>Guide:</strong> Prof. Sagarika Borah &nbsp;·&nbsp;
            M.Tech DS &amp; AI, Sem 4 &nbsp;·&nbsp;
            PES University, Electronic City, Bengaluru
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
if "uploaded_bytes" not in st.session_state:
    st.session_state.uploaded_bytes    = None
    st.session_state.uploaded_name     = None
if "activity_log" not in st.session_state:
    st.session_state.activity_log      = []
    st.session_state.activity_log.append(
        f"{datetime.now().strftime('%H:%M:%S')}  🟢  App initialized"
    )
if "mesh_logged" not in st.session_state:
    st.session_state.mesh_logged        = False
if "inference_result" not in st.session_state:
    st.session_state.inference_result   = None
if "inference_done_for" not in st.session_state:
    st.session_state.inference_done_for = ""
if "pred_bytes" not in st.session_state:
    st.session_state.pred_bytes = None
if "pred_name"  not in st.session_state:
    st.session_state.pred_name  = None
if "training_just_done" not in st.session_state:
    st.session_state.training_just_done = False
if "last_train_auc" not in st.session_state:
    st.session_state.last_train_auc = None

# Initialise source_3d_dir from .env, falling back to the default folder
if "source_3d_dir" not in st.session_state:
    _default_src = str(_PROJ_ROOT / "Source_3d_models")
    _env_file = _PROJ_ROOT / ".env"
    if _env_file.exists():
        for _line in _env_file.read_text().splitlines():
            if _line.startswith("SOURCE_3D_MODELS="):
                _default_src = _line.split("=", 1)[1].strip()
                break
    st.session_state.source_3d_dir = _default_src

# ── Log helper ────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    entry = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
    last  = st.session_state.activity_log[-1] if st.session_state.activity_log else ""
    if last.split("  ", 1)[-1] != msg:          # skip exact consecutive duplicates
        st.session_state.activity_log.append(entry)

# ── Activity log renderer ─────────────────────────────────────────────────────
def render_log(entries: list) -> str:
    def colour(e: str) -> str:
        if "❌" in e:                                        return "#ff6b6b"
        if "✅" in e:                                        return "#6bffb8"
        if any(x in e for x in ("⚙️", "🧠", "⏸️")):       return "#6bbbff"
        if any(x in e for x in ("📂", "📐", "📦", "🗂️")): return "#ffd06b"
        if any(x in e for x in ("📊", "📈")):               return "#a0e0ff"
        if any(x in e for x in ("🏋️", "🔹")):              return "#c0c0e0"
        if "🚀" in e:                                        return "#ffaa44"
        return "#aaaacc"

    rows = "".join(
        f'<div style="color:{colour(e)};font-size:0.67rem;'
        f'font-family:monospace;line-height:1.7;word-break:break-all;">{e}</div>'
        for e in entries
    )
    return (
        '<div style="background:#070e18;border-radius:6px;padding:0.5rem 0.6rem;'
        'max-height:220px;overflow-y:auto;">' + rows + "</div>"
    )

# ── Inference helpers ─────────────────────────────────────────────────────────

def _run_inference(step_bytes: bytes) -> dict:
    """
    Write STEP bytes to a cache file then run missing-component prediction
    in a subprocess (avoids gmsh + PyTorch signal-handler conflicts).
    Returns a dict with keys: n_nodes, n_edges, missing_links  or  error.
    """
    _STEP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _STEP_CACHE.write_bytes(step_bytes)

    back_end = str(_PROJ_ROOT / "back_end")
    script = textwrap.dedent(f"""\
        import sys, json, io, contextlib
        sys.path.insert(0, {repr(back_end)})
        from dataset import _parse_step
        from infer import load_checkpoint, predict_missing

        graph = _parse_step({repr(str(_STEP_CACHE))})
        if graph is None or graph.num_nodes < 2:
            print(json.dumps({{"error": "Need ≥ 2 solid bodies for inference"}}))
            sys.exit(0)

        with contextlib.redirect_stdout(io.StringIO()):
            gnn, lp, device, cfg = load_checkpoint({repr(str(_CKPT_PATH))})

        results = predict_missing(gnn, lp, graph, device, top_k=5)
        out = {{
            "n_nodes": int(graph.num_nodes),
            "n_edges": int(graph.edge_index.size(1)),
            "missing_links": [
                {{"src": int(u), "dst": int(v), "confidence": float(s)}}
                for (u, v), s in results
            ]
        }}
        print(json.dumps(out))
    """)
    r = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=180,
    )
    # Parse the last JSON line from stdout
    for line in reversed(r.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                return json.loads(line)
            except Exception:
                pass
    return {"error": (r.stderr.strip()[:300] or "Inference subprocess failed")}


def _right_panel_html(result: dict | None, ckpt_exists: bool) -> str:
    """Return the HTML for the right viewer panel based on inference state."""
    PANEL = (
        "border:1.5px solid #2a4060;border-radius:8px;height:260px;"
        "background:#f8f9fb;"
    )
    CENTER = (
        "display:flex;flex-direction:column;align-items:center;"
        "justify-content:center;text-align:center;padding:14px;"
    )

    if result and "error" not in result:
        missing = result.get("missing_links", [])
        n_nodes = result.get("n_nodes", 0)
        n_edges = result.get("n_edges", 0)

        if missing:
            rows = ""
            for lk in missing:
                u, v, s = lk["src"], lk["dst"], lk["confidence"]
                pct = int(s * 100)
                col = "#4caf82" if s >= 0.8 else "#5b9bd5" if s >= 0.5 else "#e08850"
                rows += (
                    f'<div style="display:flex;align-items:center;gap:8px;'
                    f'font-size:0.72rem;margin-bottom:6px;">'
                    f'<span style="color:#555;white-space:nowrap;min-width:90px;">'
                    f'Node {u} ↔ Node {v}</span>'
                    f'<div style="flex:1;background:#dce8f0;border-radius:3px;height:7px;">'
                    f'<div style="width:{pct}%;background:{col};border-radius:3px;height:7px;"></div>'
                    f'</div>'
                    f'<span style="color:{col};font-weight:700;min-width:36px;text-align:right;">'
                    f'{s:.3f}</span></div>'
                )
        else:
            rows = "<p style='color:#4caf82;font-size:0.78rem;'>✓ Assembly appears complete — no missing links found.</p>"

        return (
            f'<div style="{PANEL}padding:14px;overflow-y:auto;">'
            f'<p style="font-size:0.72rem;color:#888;margin:0 0 6px;">'
            f'{n_nodes} components · {n_edges} known connections</p>'
            f'<p style="font-size:0.82rem;font-weight:600;color:#333;margin:0 0 10px;">'
            f'🔍 Missing Component Predictions</p>'
            f'{rows}</div>'
        )

    if result and "error" in result:
        return (
            f'<div style="{PANEL}{CENTER}">'
            f'<div style="font-size:1.8rem;">⚠️</div>'
            f'<div style="font-size:0.8rem;color:#e05555;margin-top:0.4rem;">Inference error</div>'
            f'<div style="font-size:0.68rem;color:#aaa;margin-top:0.3rem;max-width:180px;">'
            f'{result["error"][:120]}</div></div>'
        )

    if not ckpt_exists:
        return (
            f'<div style="{PANEL}{CENTER}">'
            f'<div style="font-size:2rem;">🧠</div>'
            f'<div style="font-size:0.88rem;font-weight:600;color:#5b9bd5;margin-top:0.5rem;">'
            f'No trained model yet</div>'
            f'<div style="font-size:0.75rem;color:#999;margin-top:0.3rem;">'
            f'Click "Train 3D Models" to begin</div></div>'
        )

    return (
        f'<div style="{PANEL}{CENTER}">'
        f'<div style="font-size:2rem;">🔍</div>'
        f'<div style="font-size:0.88rem;font-weight:600;color:#5b9bd5;margin-top:0.5rem;">'
        f'Upload a STEP file to predict</div>'
        f'<div style="font-size:0.75rem;color:#999;margin-top:0.3rem;">'
        f'Missing components will appear here</div></div>'
    )


# ── Source folder helper ──────────────────────────────────────────────────────
def _pick_source_folder() -> str | None:
    """Open a native macOS folder-picker; return the chosen path or None."""
    result = subprocess.run(
        ["osascript", "-e",
         'set p to POSIX path of (choose folder with prompt '
         '"Select the folder containing your 3D model files (.step / .stp)")'
         ],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode == 0:
        return result.stdout.strip().rstrip("/") or None
    return None


def _save_source_dir(path: str) -> None:
    """Persist the chosen folder to .env and back_end/config.yaml."""
    # .env
    env_path = _PROJ_ROOT / ".env"
    if env_path.exists():
        txt = env_path.read_text()
        txt = re.sub(r"^SOURCE_3D_MODELS=.*$", f"SOURCE_3D_MODELS={path}",
                     txt, flags=re.MULTILINE)
        env_path.write_text(txt)
    # config.yaml
    cfg_path = _PROJ_ROOT / "back_end" / "config.yaml"
    if cfg_path.exists():
        txt = cfg_path.read_text()
        txt = re.sub(r'^(\s*source_dir:\s*).*$', f'\\1"{path}"',
                     txt, flags=re.MULTILINE)
        cfg_path.write_text(txt)


# ── Training helpers ──────────────────────────────────────────────────────────
_TRAIN_SCRIPT = _PROJ_ROOT / "back_end" / "train.py"
_TRAIN_LOG    = _PROJ_ROOT / ".logs" / "training.log"

# Patterns that mark a training milestone worth showing in the activity log
_MILESTONE_RE = re.compile(
    r"\[1/4\]|\[2/4\]|\[3/4\]|\[4/4\]"        # stage markers
    r"|Found \d+|No STEP files|Parsing:|Parsed \d+|Generating \d+|Saved \d+"
    r"|graphs loaded|graphs →"                   # dataset lines
    r"|Model built|total params"                 # model build
    r"|✓ New best|New best AUC|checkpoint saved" # best model
    r"|Early stop"                               # early stopping
    r"|── Test|auc\s*:|ap\s*:|Results saved|Best checkpoint"  # test results
    r"|Traceback|Error|Exception"                # errors
)
_EPOCH_RE   = re.compile(r"Ep\s+(\d+)/")
_AUC_RE     = re.compile(r"AUC=([0-9.]+)")
_TEST_AUC   = re.compile(r"auc\s*:\s*([0-9.]+)", re.IGNORECASE)


def _emoji_for(line: str) -> str:
    if "[1/4]" in line:                            return "📂"
    if "[2/4]" in line:                            return "🧠"
    if "[3/4]" in line:                            return "🏋️"
    if "[4/4]" in line:                            return "📊"
    if any(x in line for x in ("Found", "Parsing", "Parsed", "Generating", "Saved", "graphs")):
        return "🗂️"
    if any(x in line for x in ("Model built", "total params")):
        return "⚙️"
    if any(x in line for x in ("✓", "New best", "checkpoint")):
        return "✅"
    if "Early stop" in line:                       return "⏸️"
    if any(x in line.lower() for x in ("auc", "ap :", "results saved", "best checkpoint")):
        return "📈"
    if any(x in line for x in ("Error", "Traceback", "Exception")):
        return "❌"
    return "🔹"


def _is_training() -> bool:
    pid = st.session_state.get("training_pid")
    if not pid:
        return False
    try:
        os.kill(pid, 0)   # signal 0 = existence check only
        return True
    except OSError:
        return False


def _start_training() -> None:
    _TRAIN_LOG.parent.mkdir(parents=True, exist_ok=True)
    lf = open(_TRAIN_LOG, "w")          # file handle passed to Popen; child inherits fd
    proc = subprocess.Popen(
        [sys.executable, str(_TRAIN_SCRIPT)],
        cwd=str(_TRAIN_SCRIPT.parent),
        stdout=lf, stderr=lf,
    )
    lf.close()                           # parent can close; child keeps its fd
    st.session_state.training_pid     = proc.pid
    st.session_state.training_log_pos = 0
    log(f"🚀  Training started (PID {proc.pid})")


def _poll_training() -> None:
    """Read new lines from training log and push milestones to activity log."""
    if not _TRAIN_LOG.exists():
        return
    with open(_TRAIN_LOG, "r", errors="replace") as f:
        f.seek(st.session_state.get("training_log_pos", 0))
        new_text = f.read()
        st.session_state.training_log_pos = f.tell()

    for line in new_text.splitlines():
        line = line.strip()
        if not line:
            continue

        # Capture test AUC for the completion banner
        ta = _TEST_AUC.search(line)
        if ta:
            st.session_state.last_train_auc = float(ta.group(1))

        # Always log milestone lines
        if _MILESTONE_RE.search(line):
            log(f"{_emoji_for(line)}  {line}")
            continue

        # Log every 10th epoch (and capture best AUC per epoch)
        em = _EPOCH_RE.search(line)
        if em and int(em.group(1)) % 10 == 0:
            am = _AUC_RE.search(line)
            auc_tag = f"  AUC={am.group(1)}" if am else ""
            log(f"📊  Epoch {em.group(1)}{auc_tag}")

    # Mark done when process exits
    if not _is_training() and st.session_state.get("training_pid"):
        auc_str = (f"  (AUC {st.session_state.last_train_auc:.4f})"
                   if st.session_state.last_train_auc else "")
        log(f"✅  Training complete{auc_str} — upload a 3D model to predict missing components")
        st.session_state.training_pid       = None
        st.session_state.training_just_done = True


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<p style='font-size:0.8rem;font-weight:700;margin:0 0 0.2rem;'>⚙️ Viewer Settings</p>",
        unsafe_allow_html=True,
    )
    _cc, _co = st.columns([1, 1.5])
    mesh_color = _cc.color_picker("Colour", "#5b9bd5")
    opacity    = _co.slider("Opacity", 0.1, 1.0, 1.0, 0.05)
    show_grid  = True  # grid always on
    _col_bg, _col_cam = st.columns(2)
    bg_color    = _col_bg.selectbox("Background", ["white", "black", "grey", "lightgrey"])
    view_preset = _col_cam.selectbox("Camera", ["Isometric", "Top", "Front", "Side"])

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.68rem;color:#3a5878;margin:0 0 0.15rem;'>"
        "Locate source for training</p>",
        unsafe_allow_html=True,
    )
    if st.button("📁 3D Files", use_container_width=True):
        chosen = _pick_source_folder()
        if chosen:
            st.session_state.source_3d_dir = chosen
            _save_source_dir(chosen)
            log(f"📁  Source folder set: {Path(chosen).name}")
            st.rerun()

    # Show label + full path in tooltip
    st.markdown(
        f"<p style='font-size:0.65rem;color:#2a4060;margin:0.05rem 0 0;"
        f"overflow:hidden;text-overflow:ellipsis;white-space:nowrap;"
        f"' title='{st.session_state.source_3d_dir}'>📂 If new 3D models added</p>",
        unsafe_allow_html=True,
    )

    if st.session_state.uploaded_bytes is not None:
        if st.button("🔄 Upload new file", use_container_width=True):
            log("🔄  New file upload requested")
            st.session_state.uploaded_bytes   = None
            st.session_state.uploaded_name    = None
            st.session_state.mesh_logged      = False
            st.session_state.inference_result = None
            st.session_state.inference_done_for = ""
            st.rerun()

    # ── Train button ──────────────────────────────────────────────────────────
    _poll_training()   # read any new log lines on every rerun

    if _is_training():
        st.markdown(
            "<p style='font-size:0.72rem;color:#4caf82;margin:0.2rem 0;'>"
            "⚙️ Training in progress…</p>",
            unsafe_allow_html=True,
        )
        if st.button("🔄 Refresh status", use_container_width=True):
            st.rerun()
    else:
        if st.button("🚀 Train 3D Models", use_container_width=True):
            _start_training()
            st.rerun()

    # Activity Log — always visible, compact heading
    st.markdown(
        "<p style='font-size:0.78rem;font-weight:600;margin:0.25rem 0 0.15rem;'>📋 Activity Log</p>",
        unsafe_allow_html=True,
    )
    log_slot = st.empty()

# ── Conversion helpers ────────────────────────────────────────────────────────
def _gmsh_convert(step_path: str, stl_path: str) -> None:
    """Run gmsh in a subprocess to avoid signal-handler thread restriction."""
    script = textwrap.dedent(f"""\
        import gmsh
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 5.0)
        gmsh.model.add("assembly")
        gmsh.merge({repr(step_path)})
        gmsh.model.mesh.generate(2)
        gmsh.write({repr(stl_path)})
        gmsh.finalize()
    """)
    r = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "gmsh conversion failed")


@st.cache_data(show_spinner=False)
def load_mesh(file_bytes: bytes) -> dict:
    """STEP → triangulated mesh data. Cached so sidebar changes skip re-conversion."""
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "model.step")
        tp = os.path.join(d, "model.stl")
        with open(sp, "wb") as f:
            f.write(file_bytes)
        _gmsh_convert(sp, tp)
        mesh  = pv.read(tp).triangulate()
        faces = mesh.faces.reshape(-1, 4)
        return dict(
            verts   = mesh.points.copy(),
            i       = faces[:, 1].copy(),
            j       = faces[:, 2].copy(),
            k       = faces[:, 3].copy(),
            bounds  = mesh.bounds,
            n_pts   = mesh.n_points,
            n_cells = mesh.n_cells,
        )

# ── Always-visible dual-panel layout ─────────────────────────────────────────
# ── Training completion banner ────────────────────────────────────────────────
if st.session_state.training_just_done and _CKPT_PATH.exists():
    auc_tag = (f" · AUC {st.session_state.last_train_auc:.4f}"
               if st.session_state.last_train_auc else "")
    st.success(
        f"🎉 **Training Complete{auc_tag}** — Model ready!  "
        f"Upload a 3D model in the right panel to predict missing components.",
        icon="✅",
    )

# ── Always-visible dual-panel layout ─────────────────────────────────────────
st.markdown("### 3D Viewers")
col_left, col_right = st.columns(2)

CAMERAS = {
    "Isometric": dict(eye=dict(x=1.5, y=1.5, z=1.5)),
    "Top":       dict(eye=dict(x=0,   y=0,   z=2.5)),
    "Front":     dict(eye=dict(x=0,   y=-2.5, z=0 )),
    "Side":      dict(eye=dict(x=2.5, y=0,    z=0 )),
}
BG_MAP = {"white":"#ffffff","black":"#000000","grey":"#808080","lightgrey":"#d3d3d3"}

# ── LEFT: 3D viewer (view only — not connected to inference) ──────────────────
with col_left:
    st.markdown(
        "<p style='font-size:0.8rem;font-weight:600;margin:0 0 0.2rem;color:#444;'>"
        "📁 3D Model Viewer</p>",
        unsafe_allow_html=True,
    )

    if st.session_state.uploaded_bytes is None:
        raw = st.file_uploader(
            "Upload a STEP / STP file", type=["step", "stp"],
            help="Converted via gmsh then rendered interactively.",
        )
        if raw:
            kb = len(raw.getvalue()) / 1024
            log(f"📂  File received: {raw.name}  ({kb:.1f} KB)")
            st.session_state.uploaded_bytes = raw.getvalue()
            st.session_state.uploaded_name  = raw.name
            st.rerun()
        else:
            st.markdown(
                '<div style="text-align:center;padding:2rem 0;color:#aaa;height:260px;'
                'display:flex;flex-direction:column;align-items:center;justify-content:center;">'
                '<div style="font-size:2.5rem;">📂</div>'
                '<p style="font-size:0.85rem;margin-top:0.4rem;">Upload a STEP file to view</p>'
                '</div>',
                unsafe_allow_html=True,
            )
    else:
        if not st.session_state.mesh_logged:
            log("⚙️  Starting STEP → STL conversion via gmsh…")

        with st.spinner("Converting STEP → STL …"):
            try:
                m = load_mesh(st.session_state.uploaded_bytes)
            except Exception as exc:
                log(f"❌  Conversion failed: {exc}")
                st.error(f"Conversion failed: {exc}")
                m = None

        if m:
            if not st.session_state.mesh_logged:
                log("✅  STL mesh generated")
                b = m["bounds"]
                log(f"📐  {m['n_pts']:,} pts · {m['n_cells']:,} faces")
                log(f"📦  Bbox: {b[1]-b[0]:.2f} × {b[3]-b[2]:.2f} × {b[5]-b[4]:.2f}")
                log("🎨  Building 3D viewer…")
                st.session_state.mesh_logged = True

            verts = m["verts"]
            fig = go.Figure(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=m["i"], j=m["j"], k=m["k"],
                color=mesh_color, opacity=opacity, flatshading=True,
                lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3,
                              roughness=0.5, fresnel=0.2),
                lightposition=dict(x=100, y=100, z=100),
            ))
            fig.update_layout(
                scene=dict(
                    bgcolor=BG_MAP[bg_color], aspectmode="data",
                    xaxis=dict(showgrid=show_grid, title="X"),
                    yaxis=dict(showgrid=show_grid, title="Y"),
                    zaxis=dict(showgrid=show_grid, title="Z"),
                ),
                scene_camera=CAMERAS[view_preset],
                margin=dict(l=0, r=0, b=0, t=0),
                paper_bgcolor="rgba(0,0,0,0)",
                height=255,
            )
            st.plotly_chart(fig, use_container_width=True)
            b = m["bounds"]
            st.markdown(
                f'<div style="font-size:0.7rem;color:#888;margin-top:-0.4rem;">'
                f'✅ {st.session_state.uploaded_name} &nbsp;·&nbsp; '
                f'{m["n_pts"]:,} pts · {m["n_cells"]:,} faces</div>',
                unsafe_allow_html=True,
            )

# ── RIGHT: Prediction panel — active only after training is complete ───────────
with col_right:
    st.markdown(
        "<p style='font-size:0.8rem;font-weight:600;margin:0 0 0.2rem;color:#444;'>"
        "🤖 AI-Assisted 3D Assembly Design Viewer</p>",
        unsafe_allow_html=True,
    )

    if not _CKPT_PATH.exists():
        # Training not done yet
        st.markdown(_right_panel_html(None, False), unsafe_allow_html=True)

    elif st.session_state.pred_bytes is None:
        # Trained model ready — show prediction uploader
        pred_file = st.file_uploader(
            "📂 Upload 3D model for predicting missing components",
            type=["step", "stp"],
            key="pred_uploader",
            help="The trained GNN will identify which component connections are missing.",
        )
        if pred_file:
            st.session_state.pred_bytes = pred_file.getvalue()
            st.session_state.pred_name  = pred_file.name
            log(f"🔍  Prediction file received: {pred_file.name}")
            st.rerun()

    else:
        # Run inference (once per file)
        if st.session_state.inference_done_for != st.session_state.pred_name:
            with st.spinner("🔍 Predicting missing components…"):
                log("🔍  Running GNN inference…")
                _res = _run_inference(st.session_state.pred_bytes)
                st.session_state.inference_result   = _res
                st.session_state.inference_done_for = st.session_state.pred_name
                if "error" not in _res:
                    _n = len(_res.get("missing_links", []))
                    log(f"✅  Prediction done — {_n} missing link(s) found")
                else:
                    log(f"⚠️  Inference: {_res.get('error','')[:80]}")

        st.markdown(
            _right_panel_html(st.session_state.inference_result, True),
            unsafe_allow_html=True,
        )
        if st.button("🔄 Predict another model", key="reset_pred",
                     use_container_width=True):
            st.session_state.pred_bytes         = None
            st.session_state.pred_name          = None
            st.session_state.inference_result   = None
            st.session_state.inference_done_for = ""
            st.rerun()

log("✅  3D viewer ready")

# ── Final activity log render ─────────────────────────────────────────────────
log_slot.markdown(render_log(st.session_state.activity_log[-12:]),
                  unsafe_allow_html=True)

# ── Auto-scroll + hide success alert on first viewer interaction ──────────────
if _is_training():
    streamlit.components.v1.html(
        "<script>setTimeout(function(){"
        "window.parent.document.querySelector('[data-testid=\"stRefreshButton\"] button')"
        "?.click();"           # try Streamlit's internal refresh if available
        "},4000);</script>",
        height=0,
    )

streamlit.components.v1.html(
    """<script>
    setTimeout(function () {
        var doc = window.parent.document;

        // Auto-scroll to bottom
        var main = doc.querySelector('section.main');
        if (main) main.scrollTo({ top: main.scrollHeight, behavior: 'smooth' });

        // Hide success alert when user first pans or zooms the 3D viewer
        var alertEl = doc.querySelector('[data-testid="stAlert"]');
        var plotDiv = doc.querySelector('.js-plotly-plot');

        if (plotDiv && alertEl) {
            var dismiss = function () {
                alertEl.style.transition = 'opacity 0.5s ease';
                alertEl.style.opacity = '0';
                setTimeout(function () { alertEl.style.display = 'none'; }, 500);
            };
            // plotly_relayout fires on any camera move (pan, zoom, rotate)
            plotDiv.on('plotly_relayout', dismiss);
        }
    }, 500);
    </script>""",
    height=0,
)
