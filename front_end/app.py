import os
import subprocess
import sys
import tempfile
import textwrap
from datetime import datetime

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
        padding: 0.55rem 1.5rem;
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
        <span style="color:#ffffff;font-size:1.5rem;font-weight:700;letter-spacing:0.4px;">
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
    st.session_state.mesh_logged       = False

# ── Log helper ────────────────────────────────────────────────────────────────
def log(msg: str) -> None:
    entry = f"{datetime.now().strftime('%H:%M:%S')}  {msg}"
    last  = st.session_state.activity_log[-1] if st.session_state.activity_log else ""
    if last.split("  ", 1)[-1] != msg:          # skip exact consecutive duplicates
        st.session_state.activity_log.append(entry)

# ── Activity log renderer ─────────────────────────────────────────────────────
def render_log(entries: list) -> str:
    def colour(e: str) -> str:
        if "❌" in e: return "#ff6b6b"
        if "✅" in e: return "#6bffb8"
        if any(x in e for x in ("⚙️", "🎨", "🔄")): return "#6bbbff"
        if any(x in e for x in ("📂", "📐", "📦")): return "#ffd06b"
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

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<p style='font-size:0.8rem;font-weight:700;margin:0 0 0.2rem;'>⚙️ Viewer Settings</p>",
        unsafe_allow_html=True,
    )
    mesh_color  = st.color_picker("Part colour", "#5b9bd5")
    bg_color    = st.selectbox("Background", ["white", "black", "grey", "lightgrey"])
    view_preset = st.selectbox("Camera preset", ["Isometric", "Top", "Front", "Side"])
    show_grid   = st.checkbox("Show axis grid", value=False)
    opacity     = st.slider("Opacity", 0.1, 1.0, 1.0, 0.05)

    st.markdown("---")
    if st.session_state.uploaded_bytes is not None:
        if st.button("🔄 Upload new file", use_container_width=True):
            log("🔄  New file upload requested")
            st.session_state.uploaded_bytes = None
            st.session_state.uploaded_name  = None
            st.session_state.mesh_logged    = False
            st.rerun()

    # Activity Log — always visible, compact heading
    st.markdown(
        "<p style='font-size:0.78rem;font-weight:600;margin:0.35rem 0 0.2rem;'>📋 Activity Log</p>",
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

# ── File uploader (hidden after upload) ───────────────────────────────────────
uploader_slot = st.empty()

if st.session_state.uploaded_bytes is None:
    with uploader_slot.container():
        c1, c2 = st.columns([2, 1])
        with c1:
            raw = st.file_uploader(
                "Upload a STEP / STP file", type=["step", "stp"],
                help="Converted locally via gmsh then rendered with Plotly.",
            )
        with c2:
            st.info("**Supported:** `.step`, `.stp`\n\nInteractive 3D — zoom, rotate, pan.")

    if raw:
        kb = len(raw.getvalue()) / 1024
        log(f"📂  File received: {raw.name}  ({kb:.1f} KB)")
        st.session_state.uploaded_bytes = raw.getvalue()
        st.session_state.uploaded_name  = raw.name
        st.rerun()
    else:
        st.markdown(
            """<div style="text-align:center;padding:3rem 0;color:#888;">
                <div style="font-size:3rem;">📂</div>
                <p style="margin-top:0.5rem;">Upload a STEP file above to get started</p>
            </div>""",
            unsafe_allow_html=True,
        )
        log_slot.markdown(render_log(st.session_state.activity_log[-12:]),
                          unsafe_allow_html=True)
        st.stop()

# ── Convert & display ─────────────────────────────────────────────────────────
uploader_slot.empty()

if not st.session_state.mesh_logged:
    log("⚙️  Starting STEP → STL conversion via gmsh…")

with st.spinner("Converting STEP → STL mesh …"):
    try:
        m = load_mesh(st.session_state.uploaded_bytes)
    except Exception as exc:
        log(f"❌  Conversion failed: {exc}")
        log_slot.markdown(render_log(st.session_state.activity_log[-12:]),
                          unsafe_allow_html=True)
        st.error(f"Conversion failed: {exc}")
        st.stop()

if not st.session_state.mesh_logged:
    log("✅  STL mesh generated successfully")
    b = m["bounds"]
    log(f"📐  {m['n_pts']:,} points · {m['n_cells']:,} faces")
    log(f"📦  Bbox: {b[1]-b[0]:.2f} × {b[3]-b[2]:.2f} × {b[5]-b[4]:.2f}")
    log("🎨  Building Plotly 3D viewer…")
    st.session_state.mesh_logged = True

# Stats
st.success(f"✅ **{st.session_state.uploaded_name}** loaded successfully.")
b = m["bounds"]
st.markdown(
    f"""<div style="display:flex;gap:1.5rem;margin:0.3rem 0 0.6rem;font-size:0.75rem;color:#555;">
        <span><strong>Points:</strong> {m['n_pts']:,}</span>
        <span><strong>Faces:</strong> {m['n_cells']:,}</span>
        <span><strong>Bbox (x×y×z):</strong>
              {b[1]-b[0]:.2f} × {b[3]-b[2]:.2f} × {b[5]-b[4]:.2f}</span>
    </div>""",
    unsafe_allow_html=True,
)

# ── Plotly figure ─────────────────────────────────────────────────────────────
CAMERAS = {
    "Isometric": dict(eye=dict(x=1.5, y=1.5, z=1.5)),
    "Top":       dict(eye=dict(x=0,   y=0,   z=2.5)),
    "Front":     dict(eye=dict(x=0,   y=-2.5, z=0 )),
    "Side":      dict(eye=dict(x=2.5, y=0,    z=0 )),
}
BG_MAP = {"white":"#ffffff","black":"#000000","grey":"#808080","lightgrey":"#d3d3d3"}

verts = m["verts"]
fig = go.Figure(go.Mesh3d(
    x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
    i=m["i"], j=m["j"], k=m["k"],
    color=mesh_color, opacity=opacity, flatshading=True,
    lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3, roughness=0.5, fresnel=0.2),
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
    height=340,
)

# ── Dual viewer layout ────────────────────────────────────────────────────────
st.markdown("### 3D Viewers")
col_left, col_right = st.columns(2)

with col_left:
    st.markdown(
        f"<p style='font-size:0.8rem;font-weight:600;margin:0 0 0.25rem;color:#444;'>"
        f"📁 {st.session_state.uploaded_name}</p>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.markdown(
        "<p style='font-size:0.8rem;font-weight:600;margin:0 0 0.25rem;color:#444;'>"
        "🤖 AI-Assisted 3D Assembly Design Viewer</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """<div style="
            border:1.5px solid #2a4060;
            border-radius:8px;
            height:346px;
            display:flex;
            flex-direction:column;
            align-items:center;
            justify-content:center;
            text-align:center;
            background:#f8f9fb;
            color:#aaa;
        ">
            <div style="font-size:2.2rem;">🚧</div>
            <div style="font-size:0.95rem;font-weight:600;color:#5b9bd5;margin-top:0.5rem;">
                Coming Soon
            </div>
            <div style="font-size:0.78rem;color:#999;margin-top:0.3rem;">
                Working on it…
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

log("✅  3D viewer ready")

# ── Final activity log render ─────────────────────────────────────────────────
log_slot.markdown(render_log(st.session_state.activity_log[-12:]),
                  unsafe_allow_html=True)

# ── Auto-scroll + hide success alert on first viewer interaction ──────────────
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
