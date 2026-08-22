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
_CKPT_SERVING = _PROJ_ROOT / "back_end" / "checkpoints" / "best_serving.pt"
_CKPT_FALLBACK = _PROJ_ROOT / "back_end" / "checkpoints" / "best_overall.pt"
_CKPT_PATH  = _CKPT_SERVING if _CKPT_SERVING.exists() else _CKPT_FALLBACK
_STEP_CACHE   = _PROJ_ROOT / ".logs" / "inference_input.step"
_TMPL_DB_PATH = _PROJ_ROOT / "back_end" / "data" / "assembly_templates.json"
_RANKER_PATH  = _PROJ_ROOT / "back_end" / "checkpoints" / "node_ranker.pt"
_SHAPE_VAE_PATH = _PROJ_ROOT / "back_end" / "checkpoints" / "shape_vae.pt"
_PART_BANK_DIR  = _PROJ_ROOT / "back_end" / "data" / "part_bank"

if not os.environ.get("DISPLAY"):
    os.environ["DISPLAY"] = ":99"
import pyvista as pv
pv.OFF_SCREEN = True

# Load .env for local dev; on Streamlit Cloud, secrets come from st.secrets
try:
    from dotenv import load_dotenv
    load_dotenv(_PROJ_ROOT / ".env")
except ImportError:
    pass

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
            <strong>Guide:</strong> Prof. Sagarika Borah (Phase 1) &nbsp;·&nbsp;
            Prof. Gaurav Siwal (Phase 2) &nbsp;·&nbsp;
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
if "last_train_ap" not in st.session_state:
    st.session_state.last_train_ap = None
if "last_cv_summary" not in st.session_state:
    st.session_state.last_cv_summary = None
if "aida_explanation" not in st.session_state:
    st.session_state.aida_explanation   = None
if "aida_explain_for" not in st.session_state:
    st.session_state.aida_explain_for   = ""

# Results-panel sections: each is independently collapsible/expandable.
# Purely a text-panel display convenience — the 3D viewer's Original/Result
# toggle is what controls whether overlays show, independent of these.
# Default: all collapsed — user opts in to whichever section(s) they want.
_PANEL_SECTION_KEYS = [
    "assembly_type", "not_assembled", "potentially_missing",
    "template_missing", "next_component", "open_joints", "suggested_shapes",
]
for _psk in _PANEL_SECTION_KEYS:
    if f"exp_{_psk}" not in st.session_state:
        st.session_state[f"exp_{_psk}"] = False

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
        'height:calc(100vh - 420px);min-height:200px;overflow-y:auto;">'
        + rows + "</div>"
    )

# ── Inference helpers ─────────────────────────────────────────────────────────

def _run_inference(step_bytes: bytes,
                   source_dir: str = "",
                   uploaded_name: str = "") -> dict:
    """
    Write STEP bytes to a cache file then run missing-component prediction
    in a subprocess (avoids gmsh + PyTorch signal-handler conflicts).
    Returns a dict with keys: n_nodes, n_edges, missing_links,
    centroids, part_names, node_degrees, potentially_missing  or  error.
    """
    _STEP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _STEP_CACHE.write_bytes(step_bytes)

    back_end = str(_PROJ_ROOT / "back_end")
    script = textwrap.dedent(f"""\
        import sys, os, json, io, contextlib, math
        sys.path.insert(0, {repr(back_end)})
        import torch
        from torch_geometric.data import Data
        from dataset import _parse_step
        from infer import load_checkpoint, predict_missing, load_ranker, predict_next_component
        from explainer import explain_missing_link, explain_next_component

        graph = _parse_step({repr(str(_STEP_CACHE))})
        n_bodies = graph.num_nodes if graph is not None else 0
        if n_bodies < 2:
            # ── Single-body (or empty) upload: geometry analysis + template match ──
            import gmsh as _sb_gmsh
            from dataset import _build_trimesh, _compute_shape_signals, _classify_component_type, COMP_TYPES as _CT
            _sb_gmsh.initialize()
            _sb_gmsh.option.setNumber("General.Terminal", 0)
            _sb_gmsh.model.add("sb_analysis")
            _sb_vols_data = []
            _sb_signals = None
            try:
                _sb_gmsh.merge({repr(str(_STEP_CACHE))})
                _sb_gmsh.model.occ.synchronize()
                _sb_vols = _sb_gmsh.model.occ.getEntities(3)
                _sb_mesh_ok = True
                try:
                    _sb_gmsh.model.mesh.generate(2)
                except Exception:
                    _sb_mesh_ok = False
                for _svi, (_sv_d, _sv_t) in enumerate(_sb_vols):
                    _sv_bb  = _sb_gmsh.model.occ.getBoundingBox(_sv_d, _sv_t)
                    _sv_vol = _sb_gmsh.model.occ.getMass(_sv_d, _sv_t)
                    _sv_dx  = _sv_bb[3] - _sv_bb[0]
                    _sv_dy  = _sv_bb[4] - _sv_bb[1]
                    _sv_dz  = _sv_bb[5] - _sv_bb[2]
                    _sv_nm  = _sb_gmsh.model.getEntityName(_sv_d, _sv_t)
                    _sb_vols_data.append({{
                        "vol": _sv_vol, "dx": _sv_dx, "dy": _sv_dy, "dz": _sv_dz,
                        "centroid": [(_sv_bb[0]+_sv_bb[3])/2,
                                     (_sv_bb[1]+_sv_bb[4])/2,
                                     (_sv_bb[2]+_sv_bb[5])/2],
                        "name": _sv_nm.strip() if _sv_nm else "",
                    }})
                    if _svi == 0:
                        # Only the first body feeds classification/NodeRanker below.
                        _sv_center = [(_sv_bb[0]+_sv_bb[3])/2,
                                      (_sv_bb[1]+_sv_bb[4])/2,
                                      (_sv_bb[2]+_sv_bb[5])/2]
                        try:
                            _sv_com = _sb_gmsh.model.occ.getCenterOfMass(_sv_d, _sv_t)
                        except Exception:
                            _sv_com = None
                        _sv_bnd = _sb_gmsh.model.getBoundary([(_sv_d, _sv_t)], oriented=False, combined=True)
                        _sv_surf_tags = [abs(_s[1]) for _s in _sv_bnd if _s[0] == 2]
                        _sv_tm = _build_trimesh(_sv_surf_tags) if _sb_mesh_ok else None
                        _sb_signals = _compute_shape_signals(
                            _sv_tm, _sv_vol, (_sv_dx, _sv_dy, _sv_dz), _sv_center, _sv_com,
                            face_count=len(_sv_surf_tags),
                        )
            except Exception:
                pass
            finally:
                _sb_gmsh.finalize()

            if not _sb_vols_data:
                print(json.dumps({{"error": "No solid bodies found in the uploaded STEP file."}}))
                sys.exit(0)

            _sv      = _sb_vols_data[0]
            _sb_sa   = 2.0 * (_sv["dx"]*_sv["dy"] + _sv["dy"]*_sv["dz"] + _sv["dz"]*_sv["dx"])
            if _sb_signals is not None:
                _sb_tidx, _, _, _ = _classify_component_type(_sb_signals)
            else:
                _sb_tidx = _CT.index("body")  # geometry extraction failed — safe fallback
            _sb_type = _CT[_sb_tidx]
            _sb_name = _sv["name"] if _sv["name"] else "Part 1"

            _tmpl_sb = None; _miss_sb = []
            try:
                from assembly_templates import AssemblyTemplateDB as _ATDB
                _db_sb = _ATDB({repr(str(_TMPL_DB_PATH))})
                if _db_sb.load():
                    _tm_sb, _tc_sb = _db_sb.match(
                        [_sb_type], name_hints=[_sb_name, {repr(uploaded_name)}],
                    )
                    if _tm_sb:
                        _tmpl_sb = {{"label": _tm_sb["label"], "category": _tm_sb["category"],
                                     "confidence": round(_tc_sb, 3),
                                     "n_assemblies": _tm_sb.get("n_assemblies", 0)}}
                        _miss_sb = _db_sb.get_missing([_sb_type], _tm_sb)
            except Exception:
                pass

            # ── Open surface detection (Octree — Borah & Borah 2020 concept) ──
            _open_surfs_sb = []
            try:
                from surface_analyzer import analyze_open_surfaces as _aos_sb
                _open_surfs_sb = _aos_sb({repr(str(_STEP_CACHE))})
            except Exception:
                pass

            # ── Next-component ranking (Phase 2, NodeRanker) ──────────────────
            _next_comp_sb = []
            if os.path.exists({repr(str(_RANKER_PATH))}):
                try:
                    _sb_bbox_max = max(_sv["dx"], _sv["dy"], _sv["dz"], 1e-9)
                    _sb_ext = sorted([_sv["dx"], _sv["dy"], _sv["dz"]])
                    _sb_elong = _sb_ext[2] / (_sb_ext[1] + 1e-9)
                    _sb_flat  = _sb_ext[0] / (_sb_ext[2] + 1e-9)
                    _sb_axy = _sv["dx"] / (_sv["dy"] + 1e-9)
                    _sb_ayz = _sv["dy"] / (_sv["dz"] + 1e-9)
                    _sb_spher = min(1.0, (math.pi ** (1/3))
                                    * ((6 * _sv["vol"]) ** (2/3)) / (_sb_sa + 1e-9))
                    _sb_sav = _sb_sa / (_sv["vol"] + 1e-9)
                    _sb_oh = [0.0] * 8
                    _sb_oh[_sb_tidx] = 1.0
                    _sb_feat = _sb_oh + [
                        min(13.8, math.log1p(max(0.0, _sv["vol"]))),
                        min(11.5, math.log1p(max(0.0, _sb_sa))),
                        _sv["dx"] / _sb_bbox_max, _sv["dy"] / _sb_bbox_max, _sv["dz"] / _sb_bbox_max,
                        _sb_elong, _sb_flat, _sb_axy, _sb_ayz, _sb_spher,
                        0.0, 0.0, _sb_sav, 0.0,
                    ]
                    _sb_graph = Data(
                        x=torch.tensor([_sb_feat], dtype=torch.float),
                        edge_index=torch.zeros(2, 0, dtype=torch.long),
                        edge_attr=torch.zeros(0, 6, dtype=torch.float),
                    )
                    with contextlib.redirect_stdout(io.StringIO()):
                        _gnn_sb, _lp_sb, _dev_sb, _ = load_checkpoint({repr(str(_CKPT_PATH))})
                        _nr_sb, _protos_sb, _, _ = load_ranker({repr(str(_RANKER_PATH))}, _gnn_sb, _dev_sb)
                    _next_comp_sb = [
                        {{"type": t, "score": float(s)}}
                        for t, s in predict_next_component(_gnn_sb, _nr_sb, _protos_sb, _sb_graph, _dev_sb, top_k=5)
                    ]
                except Exception:
                    _next_comp_sb = []

            print(json.dumps({{
                "n_nodes": 1, "n_edges": 0, "missing_links": [],
                "centroids":   [_sv["centroid"]],
                "part_names":  [_sb_name],
                "node_degrees": [0],
                "potentially_missing": [],
                "assembly_match":       _tmpl_sb,
                "template_missing":     _miss_sb,
                "open_surfaces":        _open_surfs_sb,
                "next_component_suggestions": _next_comp_sb,
                "generated_parts":      [],
                "single_body_analysis": True,
            }}))
            sys.exit(0)

        with contextlib.redirect_stdout(io.StringIO()):
            gnn, lp, device, cfg = load_checkpoint({repr(str(_CKPT_PATH))})

        results = predict_missing(gnn, lp, graph, device, top_k=5)

        # ── GNNExplainer: why is the top missing-link prediction what it is? ──
        # Post-hoc, no retraining -- optimises a small edge/feature mask against
        # the already-loaded frozen encoder+head. ~100 steps, a couple seconds.
        _link_explanation = None
        if results:
            try:
                (_eu, _ev), _ = results[0]
                _link_explanation = explain_missing_link(
                    gnn, lp, graph, _eu, _ev, device, epochs=100,
                )
            except Exception:
                _link_explanation = None

        # ── Next-component ranking (Phase 2, NodeRanker) ──────────────────────
        _next_comp = []
        _rank_explanation = None
        if os.path.exists({repr(str(_RANKER_PATH))}):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    _nr, _protos, _, _ = load_ranker({repr(str(_RANKER_PATH))}, gnn, device)
                _next_comp = [
                    {{"type": t, "score": float(s)}}
                    for t, s in predict_next_component(gnn, _nr, _protos, graph, device, top_k=5)
                ]
                if _next_comp:
                    try:
                        _rank_explanation = explain_next_component(
                            gnn, _nr, _protos, graph, _next_comp[0]["type"], device, epochs=100,
                        )
                    except Exception:
                        _rank_explanation = None
            except Exception:
                _next_comp = []

        # One gmsh pass: centroids + names + area-thresholded contact degrees
        # A connection only counts when shared surface area >= 1 % of the smaller
        # body's total surface area — this filters out incidental tiny overlaps
        # from slightly-displaced parts that are not truly mated.
        import gmsh, re as _re
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        try:
            gmsh.merge({repr(str(_STEP_CACHE))})
            gmsh.model.occ.synchronize()
            vols = gmsh.model.occ.getEntities(3)

            # Record names before fragmentation so children can inherit them
            _pre_frag_names = {{}}
            for _d, _t in vols:
                _n = gmsh.model.getEntityName(_d, _t)
                _pre_frag_names[(_d, _t)] = _n.strip() if _n else ""

            _inherited = {{}}
            if vols:
                _fout, _fmap = gmsh.model.occ.fragment(vols, [])
                gmsh.model.occ.synchronize()
                # Propagate each parent's name to all child volumes it created
                for _si, (_d, _t) in enumerate(vols):
                    _pname = _pre_frag_names[(_d, _t)]
                    if _pname and _si < len(_fmap):
                        for _cd, _ct in _fmap[_si]:
                            if _cd == 3:
                                _inherited[(_cd, _ct)] = _pname
                vols = gmsh.model.occ.getEntities(3)

            centroids    = []
            raw_names    = []
            _surf_sets   = []
            _surf_totals = []
            for dim, tag in vols:
                bbox = gmsh.model.occ.getBoundingBox(dim, tag)
                centroids.append([(bbox[0]+bbox[3])/2, (bbox[1]+bbox[4])/2, (bbox[2]+bbox[5])/2])
                _en = gmsh.model.getEntityName(dim, tag)
                if _en and _en.strip():
                    raw_names.append(_en.strip())
                elif (dim, tag) in _inherited:
                    raw_names.append(_inherited[(dim, tag)])
                else:
                    raw_names.append("")
                _bnd  = gmsh.model.getBoundary([(dim, tag)], oriented=False, combined=True)
                _stgs = frozenset(abs(_s[1]) for _s in _bnd if _s[0] == 2)
                _surf_sets.append(_stgs)
                _surf_totals.append(max(sum(gmsh.model.occ.getMass(2, _st) for _st in _stgs), 1e-12))

            _CONTACT_THRESH = 0.01   # shared area must be >= 1 % of smaller body's area
            _nv   = len(vols)
            _adeg = [0] * _nv
            for _ii in range(_nv):
                for _jj in range(_ii + 1, _nv):
                    _sh = _surf_sets[_ii] & _surf_sets[_jj]
                    if _sh:
                        _shared_sa = sum(gmsh.model.occ.getMass(2, _st) for _st in _sh)
                        _min_sa    = min(_surf_totals[_ii], _surf_totals[_jj])
                        if _shared_sa / _min_sa >= _CONTACT_THRESH:
                            _adeg[_ii] += 1
                            _adeg[_jj] += 1
            node_degrees = _adeg

        except Exception:
            centroids    = []
            raw_names    = []
            _es = graph.edge_index[0].tolist()
            _fb = [0] * int(graph.num_nodes)
            for _s in _es: _fb[_s] += 1
            node_degrees = _fb
        finally:
            gmsh.finalize()

        # Fallback: fill any still-empty names from STEP PRODUCT records
        if any(not n for n in raw_names):
            try:
                with open({repr(str(_STEP_CACHE))}, 'r', errors='replace') as _f:
                    _txt = _f.read()
                _matches = _re.findall(r"PRODUCT\s*\(\s*'([^']*)'", _txt, _re.IGNORECASE)
                _snames  = [m.strip() for m in _matches if m.strip()]
                _pi = 0
                for _i in range(len(raw_names)):
                    if not raw_names[_i]:
                        if _pi < len(_snames):
                            raw_names[_i] = _snames[_pi]
                        _pi += 1
            except Exception:
                pass

        part_names = [n if n else f"Part {{i+1}}" for i, n in enumerate(raw_names)]

        # ── Auto-reference: find best-matching source assembly ────────────────
        # Normalise filename (strip test prefixes / version suffixes) so
        # "disp_bolt_License Plate Bracket Assembly_3.step" matches
        # "License Plate Bracket Assembly.STEP" in source folder.
        import os as _os

        def _norm(s):
            s = s.lower()
            if '.' in s:
                s = s.rsplit('.', 1)[0]
            for _pfx in ('disp_bolt_', 'disp_', 'wo_bolt_', 'wo_',
                         'test_', 'modified_', 'without_', 'missing_'):
                if s.startswith(_pfx):
                    s = s[len(_pfx):]
                    break
            while s and s[-1].isdigit():
                s = s[:-1]
            return s.rstrip('_ ').strip()

        def _bn(s):
            if s and '/' in s:
                return s.rstrip('/').split('/')[-1].strip()
            return s or ''

        _src_dir    = {repr(source_dir)}
        _test_norm  = _norm({repr(uploaded_name)})
        _ref_path   = None
        _best_score = 0.0

        if _os.path.isdir(_src_dir):
            for _sf in _os.listdir(_src_dir):
                if not _sf.lower().endswith(('.step', '.stp')):
                    continue
                _sn = _norm(_sf)
                if _sn == _test_norm:
                    _ref_path = _os.path.join(_src_dir, _sf)
                    break
                _tw = set(_test_norm.replace('_', ' ').split())
                _sw = set(_sn.replace('_', ' ').split())
                _sc = len(_tw & _sw) / max(len(_tw | _sw), 1)
                if _sc > _best_score:
                    _best_score = _sc
                    if _sc >= 0.5:
                        _ref_path = _os.path.join(_src_dir, _sf)

        potentially_missing = []
        if _ref_path and _ref_path != {repr(str(_STEP_CACHE))}:
            _rnames = []
            _rctrs  = []
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)
            try:
                gmsh.merge(_ref_path)
                gmsh.model.occ.synchronize()
                _rvols = gmsh.model.occ.getEntities(3)
                _rpre = {{}}
                for _rd, _rt in _rvols:
                    _rn = gmsh.model.getEntityName(_rd, _rt)
                    _rpre[(_rd, _rt)] = _rn.strip() if _rn else ""
                _rinh = {{}}
                if _rvols:
                    _rfo, _rfm = gmsh.model.occ.fragment(_rvols, [])
                    gmsh.model.occ.synchronize()
                    for _ri, (_rd, _rt) in enumerate(_rvols):
                        _rpn = _rpre[(_rd, _rt)]
                        if _rpn and _ri < len(_rfm):
                            for _rcd, _rct in _rfm[_ri]:
                                if _rcd == 3:
                                    _rinh[(_rcd, _rct)] = _rpn
                    _rvols = gmsh.model.occ.getEntities(3)
                for _rd, _rt in _rvols:
                    _rb = gmsh.model.occ.getBoundingBox(_rd, _rt)
                    _rctrs.append([(_rb[0]+_rb[3])/2, (_rb[1]+_rb[4])/2, (_rb[2]+_rb[5])/2])
                    _ren = gmsh.model.getEntityName(_rd, _rt)
                    if _ren and _ren.strip():
                        _rnames.append(_ren.strip())
                    elif (_rd, _rt) in _rinh:
                        _rnames.append(_rinh[(_rd, _rt)])
                    else:
                        _rnames.append("")
            except Exception:
                pass
            finally:
                gmsh.finalize()

            # Jaccard-match test vs reference component counts by basename
            def _grp(names, ctrs):
                g = {{}}
                for _n, _c in zip(names, ctrs):
                    _k = _bn(_n)
                    if _k: g.setdefault(_k, []).append(_c)
                return g

            _tg = _grp(part_names, centroids)
            _rg = _grp(_rnames,    _rctrs)

            for _nm, _rcs in _rg.items():
                _tcs = _tg.get(_nm, [])
                if len(_tcs) >= len(_rcs):
                    continue
                _used = set()
                for _tc in _tcs:
                    _bi, _bd = None, float('inf')
                    for _i, _rc in enumerate(_rcs):
                        if _i in _used: continue
                        _d = sum((_a-_b)**2 for _a, _b in zip(_tc, _rc))
                        if _d < _bd: _bd, _bi = _d, _i
                    if _bi is not None: _used.add(_bi)
                for _i, _rc in enumerate(_rcs):
                    if _i not in _used:
                        potentially_missing.append({{"name": _nm, "centroid": _rc}})

        # ── Template matching ──────────────────────────────────────────────────
        _tmpl_match = None; _tmpl_missing = []
        try:
            from assembly_templates import AssemblyTemplateDB as _ATDB2
            from dataset import COMP_TYPES as _CT2
            _db2 = _ATDB2({repr(str(_TMPL_DB_PATH))})
            if _db2.load():
                _tidxs  = graph.x[:, :8].argmax(dim=1).tolist()
                _ptypes = [_CT2[int(j)] for j in _tidxs]
                _name_hints = part_names + [{repr(uploaded_name)}]
                _tm2, _tc2 = _db2.match(_ptypes, name_hints=_name_hints)
                if _tm2:
                    _tmpl_match = {{"label": _tm2["label"], "category": _tm2["category"],
                                    "confidence": round(_tc2, 3),
                                    "n_assemblies": _tm2.get("n_assemblies", 0)}}
                    _tmpl_missing = _db2.get_missing(_ptypes, _tm2)
        except Exception:
            pass

        # ── Open surface detection (Octree — Borah & Borah 2020 concept) ──────
        # Free surfaces (not shared between any two bodies after fragment) that
        # form a significant fraction of their parent body's total surface area
        # are clustered via Octree into spatial regions — each region is a
        # potential location where a missing component should be assembled.
        _open_surfs = []
        try:
            from surface_analyzer import analyze_open_surfaces as _aos
            _open_surfs = _aos({repr(str(_STEP_CACHE))})
        except Exception:
            pass

        # ── Shape generation for missing components (Phase 3) ─────────────────
        # Every detected hole-shaped open surface (is_hole=True, from
        # surface_analyzer's per-hole detection — a real bolt-hole pattern,
        # not a one-off bore) is evaluated against all three fastener types
        # directly via PartBank's shape-fit score. This is no longer capped by
        # the template's per-type missing-count prediction, which is only an
        # approximate per-category median and — when the template match itself
        # is wrong or low-confidence — can be wildly off (observed: predicted
        # 1 missing bolt for an assembly with 12 real empty bolt holes). The
        # template match still drives the text-only "Expected Components
        # Missing" panel (_tmpl_missing, unfiltered) for non-fastener types,
        # where geometric hole-detection doesn't apply.
        _generated_parts = []
        _svae_path = {repr(str(_SHAPE_VAE_PATH))}
        _pbank_dir = {repr(str(_PART_BANK_DIR))}
        _MIN_FASTENER_FIT = 0.3
        if (_open_surfs and os.path.exists(_svae_path)
                and os.path.exists(os.path.join(_pbank_dir, "index.json"))):
            try:
                from infer import load_shape_generator, generate_missing_shape
                from shape_generator import flush_near_face_offset

                with contextlib.redirect_stdout(io.StringIO()):
                    _hsg = load_shape_generator(_pbank_dir, _svae_path, device,
                                                 retrieval_tau_fastener=0.4)

                if _hsg is not None:
                    _bank = _hsg.retriever.bank
                    _gen_category = _tmpl_match["category"] if _tmpl_match else None
                    _hole_surfs = [s for s in _open_surfs if s.get("is_hole")]
                    _hole_surfs.sort(key=lambda s: s.get("area_ratio", 0), reverse=True)

                    # Fastener sequence, per user spec (2026-08-22): a hole
                    # gets a bolt, full stop -- never an independently-best-
                    # fit-scored nut or washer sitting alone at the hole
                    # itself (nuts/washers don't occupy a hole; a nut threads
                    # onto a bolt already in place, a washer sits under it).
                    # If (and only if) the hole is a through-hole, ALSO add a
                    # washer immediately beyond the exit face, then a nut
                    # immediately beyond the washer -- both threaded onto the
                    # same bolt's shaft, in that fixed order. A blind hole
                    # gets the bolt alone.
                    def _unit(_v):
                        _len_sq = sum(_c * _c for _c in _v)
                        if _len_sq <= 1e-12:
                            return None
                        _len = _len_sq ** 0.5
                        return [_c / _len for _c in _v]

                    def _face_centroid(_centroid, _bbox, _normal_u, _use_max):
                        # In-plane (XY) coords from _centroid; the along-
                        # normal coordinate replaced with the bbox's own
                        # extreme on the requested side -- same fix as the
                        # entry-centroid correction below, generalised to
                        # pick either extreme.
                        _c = list(_centroid)
                        _dom = max(range(3), key=lambda _i: abs(_normal_u[_i]))
                        _c[_dom] = _bbox[_dom + 3] if _use_max else _bbox[_dom]
                        return _c

                    for _surf in _hole_surfs:
                        _bbox = _surf.get("bbox")
                        if not _bbox:
                            continue
                        _extents = [_bbox[3] - _bbox[0], _bbox[4] - _bbox[1], _bbox[5] - _bbox[2]]

                        _hits = _bank.query("bolt", _gen_category, _extents, top_k=1)
                        if not _hits or _hits[0].fit_score <= _MIN_FASTENER_FIT:
                            continue
                        _bolt_score = _hits[0].fit_score

                        # "centroid" is the MIDDLE of the hole recess's own
                        # depth (bbox midpoint along the bore axis), not its
                        # outer/entry face -- confirmed directly: a hole with
                        # bbox z=[85,115] (normal_hint=[0,0,1], i.e. pointing
                        # +z/outward) has centroid z=100, not 115. Anchoring
                        # bolt placement to the centroid put the head/shaft
                        # transition at the recess's midpoint, burying the
                        # head partway into the hole instead of resting it on
                        # the true surface -- reported as "the head is inside
                        # the hole" even after shape_bolt_head's flush-to-
                        # -centroid placement was verified numerically
                        # correct *relative to that (wrong) reference point*.
                        # Replace just the along-normal coordinate with the
                        # bbox's own outer extreme (max if normal points
                        # positive along that axis, min if negative); the
                        # other two in-plane coordinates stay as centroid's,
                        # since those should already be reasonably centered.
                        _nrm = _surf.get("normal_hint")
                        _nrm_u = _unit(_nrm) if _nrm else None
                        _entry_centroid = (
                            _face_centroid(_surf["centroid"], _bbox, _nrm_u, _nrm_u[max(range(3), key=lambda _i: abs(_nrm_u[_i]))] > 0)
                            if _nrm_u else list(_surf["centroid"])
                        )

                        _sr = generate_missing_shape(
                            _hsg, gnn, graph, "bolt",
                            open_joint_extents=_extents,
                            open_joint_centroid=_entry_centroid,
                            category=_gen_category, device=device,
                            normal_hint=_nrm,
                        )
                        if _sr is None:
                            continue
                        _verts = (_sr.mesh.vertices + _sr.placement).tolist()
                        _faces = _sr.mesh.faces.tolist()
                        _generated_parts.append({{
                            "type":        "bolt",
                            "fit_score":   round(float(_bolt_score), 3),
                            "source":      _sr.source,
                            "confidence":  round(float(_sr.confidence), 3),
                            "part_id":     _sr.part_id,
                            "vertices":    _verts,
                            "triangles":   _faces,
                            "surface_body_idx": _surf.get("body_idx", 0),
                        }})

                        if not _surf.get("is_through") or not _nrm_u:
                            continue
                        _exit_coord = _surf.get("exit_face_coord")
                        if _exit_coord is None:
                            continue
                        _exit_normal = [-_c for _c in _nrm_u]
                        _dom = max(range(3), key=lambda _i: abs(_nrm_u[_i]))
                        _cursor = list(_surf["centroid"])
                        _cursor[_dom] = _exit_coord  # start right at the exit face

                        for _seq_type in ("washer", "nut"):
                            _seq_hits = _bank.query(_seq_type, _gen_category, _extents, top_k=1)
                            if not _seq_hits:
                                break  # bank has nothing for this type -- stop the sequence here
                            _seq_sr = generate_missing_shape(
                                _hsg, gnn, graph, _seq_type,
                                open_joint_extents=_extents,
                                open_joint_centroid=_cursor,
                                category=_gen_category, device=device,
                                normal_hint=_exit_normal,
                            )
                            if _seq_sr is None:
                                break
                            _flush = flush_near_face_offset(_seq_sr.mesh, _exit_normal)
                            _placement = _seq_sr.placement + _flush
                            _seq_verts = (_seq_sr.mesh.vertices + _placement).tolist()
                            _generated_parts.append({{
                                "type":        _seq_type,
                                "fit_score":   round(float(_seq_hits[0].fit_score), 3),
                                "source":      _seq_sr.source,
                                "confidence":  round(float(_seq_sr.confidence), 3),
                                "part_id":     _seq_sr.part_id,
                                "vertices":    _seq_verts,
                                "triangles":   _seq_sr.mesh.faces.tolist(),
                                "surface_body_idx": _surf.get("body_idx", 0),
                            }})
                            # Advance the cursor past this part's own thickness
                            # (measured post-generation, in local coords) so
                            # the next part in the sequence starts flush
                            # against THIS one's far face, not overlapping it.
                            import numpy as _np_seq
                            _seq_proj = _seq_sr.mesh.vertices @ _np_seq.array(_exit_normal)
                            _thickness = float(_seq_proj.max() - _seq_proj.min())
                            _cursor[_dom] = _cursor[_dom] + _exit_normal[_dom] * _thickness
            except Exception:
                _generated_parts = []

        out = {{
            "n_nodes": int(graph.num_nodes),
            "n_edges": int(graph.edge_index.size(1)),
            "missing_links": [
                {{"src": int(u), "dst": int(v), "confidence": float(s)}}
                for (u, v), s in results
            ],
            "centroids":            centroids,
            "part_names":           part_names,
            "node_degrees":         node_degrees,
            "potentially_missing":  potentially_missing,
            "assembly_match":       _tmpl_match,
            "template_missing":     _tmpl_missing,
            "open_surfaces":        _open_surfs,
            "next_component_suggestions": _next_comp,
            "generated_parts":      _generated_parts,
            "single_body_analysis": False,
            "link_explanation":     _link_explanation,
            "next_component_explanation": _rank_explanation,
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


def _placeholder_panel_html(result: dict | None, ckpt_exists: bool) -> str | None:
    """HTML for panel placeholder states (error / no-model / idle).

    Returns None when `result` is a valid, error-free inference result — the
    caller should then render it as collapsible sections via
    `_build_panel_sections()` instead of a single HTML blob.
    """
    PANEL = (
        "border:1.5px solid #2a4060;border-radius:8px;height:400px;"
        "background:#f8f9fb;"
    )
    CENTER = (
        "display:flex;flex-direction:column;align-items:center;"
        "justify-content:center;text-align:center;padding:14px;"
    )

    if result and "error" not in result:
        return None

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


def _build_panel_sections(result: dict) -> tuple[str, list[dict]]:
    """Build the header line + a list of independent, collapsible sections
    for a valid (error-free) inference result.

    Each section dict has: key, icon, title, html. `key` matches the
    `exp_<key>` session-state flag that the 3D viewer reads to decide
    whether to draw the matching overlay (see the col_right viewer code).
    """
    if True:
        n_nodes      = result.get("n_nodes", 0)
        n_edges      = result.get("n_edges", 0)
        part_names   = result.get("part_names", [])
        node_degrees   = result.get("node_degrees", [])
        assembly_match = result.get("assembly_match")
        tmpl_missing   = result.get("template_missing", [])
        single_body    = result.get("single_body_analysis", False)
        open_surfs     = result.get("open_surfaces", [])
        next_comp      = result.get("next_component_suggestions", [])
        gen_parts      = result.get("generated_parts", [])
        link_expl      = result.get("link_explanation")
        rank_expl      = result.get("next_component_explanation")

        def _basename(s):
            if s and "/" in s:
                return s.rstrip("/").split("/")[-1].strip()
            return s or ""

        def _pname(idx):
            if idx < len(part_names) and part_names[idx]:
                raw = part_names[idx]
                short = _basename(raw)
                return short[:36] + "…" if len(short) > 36 else short
            return f"Part {idx + 1}"

        not_assembled   = []   # degree=0 — no contact at all
        under_connected = []   # degree>0 but < group max — not properly mated

        if not single_body:
            # Group nodes by part basename; flag isolated or under-connected
            _grp: dict = {}
            for _gi, _gn in enumerate(part_names):
                _key = _basename(_gn) if _gn else f"__solo_{_gi}"
                _grp.setdefault(_key, []).append(_gi)

            for _gn, _gnodes in _grp.items():
                _degs = [node_degrees[_i] for _i in _gnodes if _i < len(node_degrees)]
                _max  = max(_degs) if _degs else 0
                for _gi in _gnodes:
                    if _gi >= len(node_degrees):
                        continue
                    _d = node_degrees[_gi]
                    if len(_gnodes) > 1:
                        if _d == 0:
                            not_assembled.append(_gi)
                        elif _d < _max:
                            under_connected.append(_gi)
                    else:
                        if _d == 0:
                            not_assembled.append(_gi)

        sections: list[dict] = []

        # ── Section 0: Assembly type identification ───────────────────────────
        if assembly_match:
            _am_conf = assembly_match.get("confidence", 0)
            _am_pct  = int(_am_conf * 100)
            _am_col  = "#4caf82" if _am_conf >= 0.6 else "#5b9bd5" if _am_conf >= 0.35 else "#999"
            _am_n    = assembly_match.get("n_assemblies", 0)
            # Was hardcoded black — invisible against a dark theme's near-black
            # background. Follow Streamlit's actual active theme instead of
            # guessing, so it stays correct regardless of how the user set it
            # (system preference or the app's own ☰ > Settings theme picker).
            _am_label_col = "#e5e7eb" if st.context.theme.type == "dark" else "#000000"
            sections.append({
                "key": "assembly_type", "icon": "🔮", "title": "Assembly Type Identified",
                "html": (
                    f'<span style="color:{_am_label_col};font-size:0.78rem;font-weight:600;">'
                    f'{assembly_match["label"]}</span>'
                    f'<span style="background:{_am_col};color:#fff;font-size:0.65rem;'
                    f'font-weight:700;padding:1px 6px;border-radius:3px;margin-left:8px;">'
                    f'{_am_pct}%</span>'
                    f'<span style="color:#777;font-size:0.66rem;margin-left:6px;">'
                    f'· {_am_n} training samples</span>'
                ),
            })

        # ── Section 1: not assembled / under-connected nodes ──────────────────
        all_flagged = not_assembled + under_connected
        if all_flagged:
            _sec_html = ""
            # Group by (name, label) so repeated instances show as one entry + count
            _seen_flagged: dict = {}
            for i in all_flagged:
                _lbl = ("no connections in assembly"
                        if i in not_assembled else "under-connected — not properly mated")
                _key = (_pname(i), _lbl)
                _seen_flagged[_key] = _seen_flagged.get(_key, 0) + 1
            for (_nm, _lbl), _cnt in _seen_flagged.items():
                _cnt_badge = (
                    f'<span style="background:#fde8e8;color:#c0392b;font-size:0.62rem;'
                    f'font-weight:700;padding:1px 5px;border-radius:3px;margin-left:4px;">'
                    f'×{_cnt}</span>' if _cnt > 1 else ""
                )
                _sec_html += (
                    f'<div style="display:flex;align-items:center;gap:6px;'
                    f'font-size:0.72rem;margin-bottom:5px;">'
                    f'<span style="color:#ef4444;font-size:0.78rem;">⚠</span>'
                    f'<span style="color:#cc2222;font-weight:600;">{_nm}</span>'
                    f'{_cnt_badge}'
                    f'<span style="color:#888;font-size:0.68rem;">— {_lbl}</span>'
                    f'</div>'
                )
            sections.append({
                "key": "not_assembled", "icon": "🔴",
                "title": "Not Assembled / Not Properly Mated", "html": _sec_html,
            })

        # ── Section 3: potentially missing components (auto reference match) ────
        pot_missing = result.get("potentially_missing", [])
        if pot_missing:
            _sec_html = ""
            for mc in pot_missing:
                mn = mc["name"]
                mn_s = (mn[:36] + "…") if len(mn) > 36 else mn
                _sec_html += (
                    f'<div style="display:flex;align-items:center;gap:6px;'
                    f'font-size:0.72rem;margin-bottom:5px;">'
                    f'<span style="color:#f97316;font-size:0.78rem;">❓</span>'
                    f'<span style="color:#c2440e;font-weight:600;">{mn_s}</span>'
                    f'<span style="color:#888;font-size:0.68rem;">'
                    f'— not found in assembly</span>'
                    f'</div>'
                )
            sections.append({
                "key": "potentially_missing", "icon": "🔍",
                "title": "Potentially Missing Components", "html": _sec_html,
            })

        # ── Section 4: template-based missing components ──────────────────────
        if tmpl_missing:
            _sec_html = ""
            for _mc in tmpl_missing:
                _mc_cnt = _mc.get("count", 1)
                _mc_lbl = _mc.get("label", _mc.get("type", "component"))
                _mc_sfx = f" ×{_mc_cnt}" if _mc_cnt > 1 else ""
                _sec_html += (
                    f'<div style="display:flex;align-items:center;gap:6px;'
                    f'font-size:0.72rem;margin-bottom:5px;">'
                    f'<span style="color:#14b8a6;font-size:0.78rem;">➕</span>'
                    f'<span style="color:#0d9488;font-weight:600;">'
                    f'{_mc_lbl}{_mc_sfx}</span>'
                    f'</div>'
                )
            sections.append({
                "key": "template_missing", "icon": "📋",
                "title": "Expected Components Missing", "html": _sec_html,
            })

        # ── Section 4b: AI-ranked next component (Phase 2, NodeRanker) ────────
        if next_comp:
            _sec_html = ""
            for _nc in next_comp[:5]:
                _nc_type  = str(_nc.get("type", "")).capitalize()
                _nc_score = _nc.get("score", 0.0)
                _nc_pct   = int(max(0.0, (_nc_score + 1) / 2) * 100)
                _nc_col   = ("#4caf82" if _nc_score >= 0.5 else
                             "#8b5cf6" if _nc_score >= 0.0 else "#999")
                _sec_html += (
                    f'<div style="display:flex;align-items:center;gap:6px;'
                    f'font-size:0.72rem;margin-bottom:4px;">'
                    f'<span style="color:#5b21b6;font-weight:600;width:70px;'
                    f'flex-shrink:0;">{_nc_type}</span>'
                    f'<div style="flex:1;background:#eee;border-radius:3px;'
                    f'height:8px;overflow:hidden;">'
                    f'<div style="width:{_nc_pct}%;height:100%;background:{_nc_col};"></div>'
                    f'</div>'
                    f'<span style="color:#777;font-size:0.65rem;width:44px;'
                    f'text-align:right;flex-shrink:0;">{_nc_score:+.3f}</span>'
                    f'</div>'
                )
            sections.append({
                "key": "next_component", "icon": "🧭",
                "title": "AI-Ranked Next Component (GNN)", "html": _sec_html,
            })

        # ── Section 4c/4d: GNNExplainer — why the top predictions came out this
        # way. Post-hoc explanation of the already-trained model (no retraining):
        # which prior connections and which geometric features most influenced
        # the top missing-link / next-component prediction above.
        def _explanation_html(expl: dict, target_label: str) -> str:
            html = (
                f'<div style="font-size:0.72rem;color:#444;font-weight:600;'
                f'margin-bottom:5px;">{target_label}</div>'
            )
            edges = expl.get("contributing_edges", [])
            if edges:
                html += (
                    '<div style="font-size:0.65rem;color:#888;margin-bottom:2px;">'
                    'Most influential existing connections</div>'
                )
                for _e in edges[:5]:
                    _s, _d = _e["edge"]
                    html += (
                        f'<div style="font-size:0.68rem;color:#555;margin-bottom:2px;">'
                        f'&nbsp;&nbsp;{_pname(_s)} ({_e["src_type"]}) '
                        f'— {_pname(_d)} ({_e["dst_type"]})'
                        f'<span style="color:#aaa;"> · importance {_e["importance"]:.2f}</span>'
                        f'</div>'
                    )
            feats = expl.get("contributing_features")
            if feats:
                for _node_key, _flist in feats.items():
                    if not _flist:
                        continue
                    html += (
                        f'<div style="font-size:0.65rem;color:#888;margin:4px 0 2px;">'
                        f'Most influential features on node {_node_key}</div>'
                    )
                    html += '<div style="font-size:0.68rem;color:#555;">&nbsp;&nbsp;' + (
                        ", ".join(f'{_f["feature"]} ({_f["importance"]:.2f})' for _f in _flist[:5])
                    ) + '</div>'
            nodes = expl.get("contributing_nodes")
            if nodes:
                html += (
                    '<div style="font-size:0.65rem;color:#888;margin:4px 0 2px;">'
                    'Most influential existing components</div>'
                )
                html += '<div style="font-size:0.68rem;color:#555;">&nbsp;&nbsp;' + (
                    ", ".join(f'{_pname(_n["node"])} ({_n["type"]}, {_n["importance"]:.2f})'
                              for _n in nodes[:5])
                ) + '</div>'
            return html

        if link_expl:
            _lu, _lv = link_expl["target"]["edge"]
            _lbl = f'Missing link {_pname(_lu)} → {_pname(_lv)}  (confidence {link_expl["target"]["confidence"]:.0%})'
            sections.append({
                "key": "link_explanation", "icon": "🧩",
                "title": "Why This Missing Link? (GNNExplainer)",
                "html": _explanation_html(link_expl, _lbl),
            })

        if rank_expl:
            _rt = str(rank_expl["target"]["type"]).capitalize()
            _lbl = f'Next component: {_rt}  (score {rank_expl["target"]["score"]:+.3f})'
            sections.append({
                "key": "rank_explanation", "icon": "🧩",
                "title": "Why This Next Component? (GNNExplainer)",
                "html": _explanation_html(rank_expl, _lbl),
            })

        # ── Section 5: Open surface joints (Octree spatial analysis) ─────────
        if open_surfs:
            _sec_html = (
                '<p style="font-size:0.65rem;color:#65a30d;margin:0 0 6px;">'
                'Lime green mesh in 3D viewer — click a legend entry to isolate each joint</p>'
            )
            for _os in open_surfs:
                _os_bi  = _os.get("body_idx", 0)
                _os_ar  = _os.get("area_ratio", 0)
                _os_pct = int(_os_ar * 100)
                _sec_html += (
                    f'<div style="display:flex;align-items:flex-start;gap:6px;'
                    f'font-size:0.72rem;margin-bottom:6px;">'
                    f'<span style="display:inline-block;width:10px;height:10px;'
                    f'background:#84cc16;border-radius:2px;margin-top:2px;flex-shrink:0;"></span>'
                    f'<div>'
                    f'<span style="color:#3f6212;font-weight:600;">'
                    f'⬡ Body {_os_bi + 1}  ({_os_pct}%)</span>'
                    f'<span style="background:#d9f99d;color:#3f6212;font-size:0.63rem;'
                    f'padding:1px 5px;border-radius:3px;margin-left:6px;">'
                    f'{_os_pct}% of body area</span>'
                    f'<br><span style="color:#4d7c0f;font-size:0.67rem;">'
                    f'This area needs components to be assembled</span>'
                    f'</div></div>'
                )
            sections.append({
                "key": "open_joints", "icon": "⬡",
                "title": "Open Assembly Joints Detected", "html": _sec_html,
            })

        # ── Section 5b: Suggested missing-part shapes (Phase 3, shape gen) ────
        if gen_parts:
            _sec_html = (
                '<p style="font-size:0.65rem;color:#0284c7;margin:0 0 6px;">'
                'Ghost mesh in 3D viewer (bolt=red, nut=dark red, washer=brown)'
                ' — best-effort shape + location, not verified</p>'
            )
            _GP_COLORS_PANEL = {"bolt": "#ef4444", "nut": "#7f1d1d", "washer": "#92400e"}
            for _gp in gen_parts:
                _gp_type = str(_gp.get("type", "component")).capitalize()
                _gp_src  = _gp.get("source", "generated")
                _gp_conf = _gp.get("confidence", 0.0)
                _gp_fit  = _gp.get("fit_score", 0.0)
                _gp_icon = "🔄" if _gp_src == "retrieved" else "✨"
                _gp_badge_txt = "Retrieved from part bank" if _gp_src == "retrieved" else "AI-generated (VAE)"
                _gp_pct = int(max(0.0, min(1.0, _gp_conf)) * 100)
                _gp_swatch = _GP_COLORS_PANEL.get(str(_gp.get("type", "")).lower(), "#38bdf8")
                _sec_html += (
                    f'<div style="display:flex;align-items:flex-start;gap:6px;'
                    f'font-size:0.72rem;margin-bottom:6px;">'
                    f'<span style="display:inline-block;width:10px;height:10px;'
                    f'background:{_gp_swatch};border-radius:2px;margin-top:2px;flex-shrink:0;"></span>'
                    f'<div>'
                    f'<span style="color:#0369a1;font-weight:600;">'
                    f'{_gp_icon} {_gp_type}</span>'
                    f'<span style="background:#e0f2fe;color:#0369a1;font-size:0.63rem;'
                    f'padding:1px 5px;border-radius:3px;margin-left:6px;">'
                    f'{_gp_badge_txt}</span>'
                    f'<br><span style="color:#0284c7;font-size:0.67rem;">'
                    f'Location fit {_gp_fit:.0%} · shape confidence {_gp_pct}%</span>'
                    f'</div></div>'
                )
            sections.append({
                "key": "suggested_shapes", "icon": "🪄",
                "title": "Suggested Missing-Part Shapes", "html": _sec_html,
            })

        header = f'{n_nodes} components · {n_edges} known connections'
        return header, sections


# ── AIDA explanation helper ───────────────────────────────────────────────────

def _run_aida_explain(inference_result: dict) -> str:
    """Build a three-category assembly summary and ask AIDA to explain it."""
    missing      = inference_result.get("missing_links", [])
    n_nodes      = inference_result.get("n_nodes", 0)
    n_edges      = inference_result.get("n_edges", 0)
    part_names   = inference_result.get("part_names", [])
    node_degrees = inference_result.get("node_degrees", [])
    open_surfs   = inference_result.get("open_surfaces", [])
    gen_parts    = inference_result.get("generated_parts", [])
    back_end     = str(_PROJ_ROOT / "back_end")

    def _basename(s):
        if s and "/" in s:
            return s.rstrip("/").split("/")[-1].strip()
        return s or ""

    def _pname(i):
        if i < len(part_names) and part_names[i]:
            short = _basename(part_names[i])
            return short[:48] + "…" if len(short) > 48 else short
        return f"Part {i+1}"

    # Category 1 — replicate the exact grouping logic from _build_panel_sections
    _grp: dict = {}
    for _gi, _gn in enumerate(part_names):
        _key = _basename(_gn) if _gn else f"__solo_{_gi}"
        _grp.setdefault(_key, []).append(_gi)

    not_assembled:   list = []
    under_connected: list = []
    for _gnodes in _grp.values():
        _degs = [node_degrees[_i] for _i in _gnodes if _i < len(node_degrees)]
        _max  = max(_degs) if _degs else 0
        for _gi in _gnodes:
            if _gi >= len(node_degrees):
                continue
            _d = node_degrees[_gi]
            if len(_gnodes) > 1:
                if _d == 0:
                    not_assembled.append(_gi)
                elif _d < _max:
                    under_connected.append(_gi)
            else:
                if _d == 0:
                    not_assembled.append(_gi)

    # Group by unique name for a compact AIDA summary
    def _group_by_name(indices, label):
        counts: dict = {}
        for i in indices:
            n = _pname(i)
            counts[n] = counts.get(n, 0) + 1
        return [{"name": n, "count": c, "reason": label} for n, c in counts.items()]

    not_mated = (
        _group_by_name(not_assembled,   "no connections in assembly") +
        _group_by_name(under_connected, "under-connected — not properly mated")
    )

    # Category 2 — AI Predicted Missing Links
    missing_named = [
        {"src": _pname(m["src"]), "dst": _pname(m["dst"]),
         "confidence": round(m["confidence"], 3)}
        for m in missing
    ]

    # Category 3 — Open Assembly Joints (Octree surface analysis)
    open_joints = [
        {"body": m.get("body_idx", 0) + 1,
         "area_pct": int(m.get("area_ratio", 0) * 100)}
        for m in open_surfs
    ]

    # Category 4 — Suggested Missing-Part Shapes (Phase 3, shape generation)
    suggested_shapes = [
        {"type": str(g.get("type", "component")).capitalize(),
         "source": "retrieved from part bank" if g.get("source") == "retrieved" else "AI-generated (VAE)",
         "confidence": round(g.get("confidence", 0.0), 3),
         "fit_score": round(g.get("fit_score", 0.0), 3)}
        for g in gen_parts
    ]

    script = textwrap.dedent(f"""\
        import sys
        sys.path.insert(0, {repr(back_end)})
        try:
            from skills_agent import AssemblySkillsAgent
            agent = AssemblySkillsAgent()

            not_mated    = {repr(not_mated)}
            missing_lnks = {repr(missing_named)}
            open_joints  = {repr(open_joints)}
            sugg_shapes  = {repr(suggested_shapes)}
            n_nodes      = {n_nodes}
            n_edges      = {n_edges}

            # ── Category 1 text — compact one-line summary per unique part type ─
            if not_mated:
                total_flagged = sum(m['count'] for m in not_mated)
                parts_summary = ", ".join(
                    f"{{m['name']}} (x{{m['count']}}: {{m['reason']}})"
                    for m in not_mated
                )
                nm_lines = "  " + str(total_flagged) + " flagged — " + parts_summary
            else:
                nm_lines = "  (none)"

            # ── Category 2 text ───────────────────────────────────────────────
            if missing_lnks:
                ml_lines = "\\n".join(
                    f"  - {{m['src']}} ↔ {{m['dst']}} — confidence {{m['confidence']:.3f}}"
                    for m in missing_lnks
                )
            else:
                ml_lines = "  (none)"

            # ── Category 3 text ───────────────────────────────────────────────
            if open_joints:
                oj_lines = "\\n".join(
                    f"  - Body {{j['body']}}: {{j['area_pct']}}% of body surface area is open"
                    for j in open_joints
                )
            else:
                oj_lines = "  (none)"

            # ── Category 4 text ───────────────────────────────────────────────
            if sugg_shapes:
                ss_lines = "\\n".join(
                    f"  - {{s['type']}} — {{s['source']}}, location fit "
                    f"{{s['fit_score']:.0%}}, shape confidence {{s['confidence']:.0%}}"
                    for s in sugg_shapes
                )
            else:
                ss_lines = "  (none)"

            prompt = (
                "You are AIDA, an AI Design Assistant for 3D mechanical assembly analysis.\\n"
                "Assembly: " + str(n_nodes) + " components, " + str(n_edges) + " known connections.\\n\\n"
                "DATA:\\n"
                "1. Not Assembled / Not Properly Mated: " + nm_lines + "\\n"
                "2. AI Predicted Missing Links: " + ml_lines + "\\n"
                "3. Open Assembly Joints (Octree): " + oj_lines + "\\n"
                "4. Suggested Missing-Part Shapes: " + ss_lines + "\\n\\n"
                "STRICT RULES — follow exactly:\\n"
                "- Write ALL FIVE sections. Do NOT stop after section 1.\\n"
                "- Section 1: write EXACTLY 2 bullets — one for the 'no connections' group, one for the 'under-connected' group. Do NOT write a bullet per part name.\\n"
                "- Section 2: write one bullet per predicted link (max 3 bullets total).\\n"
                "- Section 3: write one bullet per open joint body (max 4 bullets total).\\n"
                "- Section 4: write one bullet per suggested shape (max 4 bullets total), noting whether it was retrieved from the part bank or AI-generated, and its confidence.\\n"
                "- Section 5: write 2 sentences max.\\n"
                "- Plain text bullets only. No markdown bold or italic.\\n\\n"
                "=== 1. Not Assembled / Not Properly Mated ===\\n\\n"
                "=== 2. AI Predicted Missing Links ===\\n\\n"
                "=== 3. Open Assembly Joints Detected ===\\n\\n"
                "=== 4. Suggested Missing-Part Shapes ===\\n\\n"
                "=== Overall Recommendation ==="
            )
            print(agent._ask(prompt, max_tokens=2500))
        except Exception as e:
            print(f"[AIDA offline] {{e}}")
    """)
    r = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, timeout=90,
    )
    return (r.stdout.strip() or r.stderr.strip()[:300]
            or "AIDA explanation unavailable.")


# ── Training helpers ──────────────────────────────────────────────────────────
_TRAIN_LOG    = _PROJ_ROOT / ".logs" / "training.log"

# Patterns that mark a training milestone worth showing in the activity log
_MILESTONE_RE = re.compile(
    r"\[1/4\]|\[2/4\]|\[3/4\]|\[4/4\]"        # stage markers
    r"|Found \d+|No STEP files|Parsing:|Parsed \d+|Generating \d+|Saved \d+"
    r"|graphs loaded|graphs →|Splits —"           # dataset lines
    r"|Model built|total params"                 # model build
    r"|✓ New best|New best AUC|checkpoint saved" # best model
    r"|Early stop"                               # early stopping
    r"|── Test|auc\s*:|ap\s*:|Results saved|Best checkpoint"  # test results
    r"|FOLD \d+|Fold \d+ test|CV Summary|Mean AUC|Mean AP|Best overall"  # CV
    r"|Traceback|Error|Exception"                # errors
)
_EPOCH_RE   = re.compile(r"Ep\s+(\d+)/")
_AUC_RE     = re.compile(r"AUC=([0-9.]+)")
_TEST_AUC   = re.compile(r"auc\s*:\s*([0-9.]+)", re.IGNORECASE)
_TEST_AP    = re.compile(r"ap\s*:\s*([0-9.]+)",  re.IGNORECASE)
_FOLD_TEST  = re.compile(r"Fold\s+(\d+)\s+test\s*—\s*AUC=([0-9.]+)\s+AP=([0-9.]+)", re.IGNORECASE)
_MEAN_AUC   = re.compile(r"Mean AUC\s*=\s*([0-9.]+)\s*[±+/-]+\s*([0-9.]+)", re.IGNORECASE)
_MEAN_AP    = re.compile(r"Mean AP\s*=\s*([0-9.]+)\s*[±+/-]+\s*([0-9.]+)",  re.IGNORECASE)


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
    if "FOLD" in line:                             return "🔁"
    if "Fold" in line and "test" in line.lower():  return "📊"
    if "CV Summary" in line:                       return "📋"
    if "Mean AUC" in line or "Mean AP" in line:    return "📈"
    if "Best overall" in line:                     return "🏆"
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

        # Capture test AUC / AP for the completion banner
        ta = _TEST_AUC.search(line)
        if ta:
            st.session_state.last_train_auc = float(ta.group(1))
        pa = _TEST_AP.search(line)
        if pa:
            st.session_state.last_train_ap = float(pa.group(1))

        # Capture per-fold test results into cv_summary
        ft = _FOLD_TEST.search(line)
        if ft:
            cv = st.session_state.last_cv_summary or {"fold_aucs": [], "fold_aps": []}
            cv["fold_aucs"].append(float(ft.group(2)))
            cv["fold_aps"].append(float(ft.group(3)))
            st.session_state.last_cv_summary = cv

        # Capture mean AUC ± std
        ma = _MEAN_AUC.search(line)
        if ma:
            cv = st.session_state.last_cv_summary or {"fold_aucs": [], "fold_aps": []}
            cv["mean_auc"] = float(ma.group(1))
            cv["std_auc"]  = float(ma.group(2))
            st.session_state.last_cv_summary = cv

        # Capture mean AP ± std
        mp = _MEAN_AP.search(line)
        if mp:
            cv = st.session_state.last_cv_summary or {"fold_aucs": [], "fold_aps": []}
            cv["mean_ap"] = float(mp.group(1))
            cv["std_ap"]  = float(mp.group(2))
            st.session_state.last_cv_summary = cv

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
        cv = st.session_state.last_cv_summary
        if cv and cv.get("mean_auc"):
            auc_str = (f"  (mean AUC {cv['mean_auc']:.4f}±{cv.get('std_auc',0):.4f}"
                       f"  mean AP {cv['mean_ap']:.4f}±{cv.get('std_ap',0):.4f})")
        else:
            auc_str = (f"  (AUC {st.session_state.last_train_auc:.4f}"
                       + (f"  AP {st.session_state.last_train_ap:.4f}" if st.session_state.last_train_ap else "")
                       + ")"
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
        "<p style='font-size:0.8rem;font-weight:600;margin:0 0 0.2rem;color:#444;'>"
        "🤖 AI-Assisted 3D Assembly Design Viewer</p>",
        unsafe_allow_html=True,
    )

    # ── Model metrics badge (persistent, reads from test_metrics.json) ──────
    _metrics_file  = _PROJ_ROOT / "back_end" / "results" / "test_metrics.json"
    if _CKPT_PATH.exists() and _metrics_file.exists():
        try:
            _m = json.loads(_metrics_file.read_text())
            _m_auc = _m.get("auc", 0)
            _m_ap  = _m.get("ap",  0)

            # Read val AUC from serving checkpoint (lightweight metadata only)
            _srv_val_auc = None
            try:
                import torch as _torch
                _srv_meta = _torch.load(str(_CKPT_SERVING), map_location="cpu", weights_only=False)
                _srv_val_auc = _srv_meta.get("auc")
            except Exception:
                pass

            # Primary badges: best-fold metrics (val AUC + test AUC/AP)
            _auc_col = "#16a34a" if _m_auc >= 0.70 else ("#d97706" if _m_auc >= 0.55 else "#dc2626")
            _ap_col  = "#16a34a" if _m_ap  >= 0.70 else ("#d97706" if _m_ap  >= 0.55 else "#dc2626")
            _val_col = "#16a34a" if (_srv_val_auc or 0) >= 0.70 else ("#d97706" if (_srv_val_auc or 0) >= 0.55 else "#dc2626")
            _val_badge = (
                f'<span style="background:{_val_col};color:#fff;font-size:0.58rem;'
                f'font-weight:700;padding:1px 5px;border-radius:10px;white-space:nowrap;">'
                f'Val AUC {_srv_val_auc:.3f}</span>'
            ) if _srv_val_auc is not None else ""
            st.markdown(
                '<div style="display:flex;gap:4px;align-items:center;'
                'margin:0 0 4px;flex-wrap:nowrap;overflow-x:auto;">'
                '<span style="font-size:0.58rem;color:#6b7280;white-space:nowrap;">Best model:</span>'
                + _val_badge +
                f'<span style="background:{_auc_col};color:#fff;font-size:0.58rem;'
                f'font-weight:700;padding:1px 5px;border-radius:10px;white-space:nowrap;">'
                f'Test AUC {_m_auc:.3f}</span>'
                f'<span style="background:{_ap_col};color:#fff;font-size:0.58rem;'
                f'font-weight:700;padding:1px 5px;border-radius:10px;white-space:nowrap;">'
                f'Test AP {_m_ap:.3f}</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        except Exception:
            pass

    # ── Code version badge — last commit + working-tree state, so a stale
    # browser session (cached inference_result, see _run_inference's
    # "once per file" gate) is visibly distinguishable from a fresh code
    # change. Recomputed on every rerun, not cached, so it's always current.
    try:
        import subprocess as _sp
        _git_dir = str(_PROJ_ROOT)
        _commit_ts = _sp.run(
            ["git", "log", "-1", "--format=%cd", "--date=format:%Y-%m-%d %H:%M"],
            cwd=_git_dir, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        _commit_hash = _sp.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_git_dir, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        _dirty = _sp.run(
            ["git", "status", "--porcelain"],
            cwd=_git_dir, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        _branch = _sp.run(
            ["git", "branch", "--show-current"],
            cwd=_git_dir, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        _remote_url = _sp.run(
            ["git", "remote", "get-url", "origin"],
            cwd=_git_dir, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        # Normalise to a clickable https URL regardless of how origin is
        # configured (https://...git, git@host:owner/repo.git, etc.)
        _repo_url = _remote_url
        if _repo_url.startswith("git@"):
            _repo_url = _repo_url.replace(":", "/", 1).replace("git@", "https://", 1)
        if _repo_url.endswith(".git"):
            _repo_url = _repo_url[:-4]

        if _commit_ts and _commit_hash:
            _dirty_note = " · uncommitted changes present" if _dirty else ""
            st.markdown(
                f'<div style="font-size:0.6rem;color:#9ca3af;margin:0 0 2px;">'
                f'Code last updated: {_commit_ts}  (commit {_commit_hash}){_dirty_note}'
                f'</div>',
                unsafe_allow_html=True,
            )
        if _repo_url:
            _branch_note = f' · branch <code>{_branch}</code>' if _branch else ""
            st.markdown(
                f'<div style="font-size:0.6rem;color:#9ca3af;margin:0 0 4px;">'
                f'Hosted from <a href="{_repo_url}" target="_blank" '
                f'style="color:#6b7280;">{_repo_url.split("github.com/")[-1]}</a>'
                f'{_branch_note}'
                f'</div>',
                unsafe_allow_html=True,
            )
    except Exception:
        pass

    st.markdown("---")

    # Training can still run from the terminal (see CLAUDE.md); this keeps
    # the activity log in sync with that log file even without the button.
    _poll_training()   # read any new log lines on every rerun

    # Activity Log — always visible, given the freed-up space below Viewer
    # Settings now that the source-folder / upload-new-file / train buttons
    # have been removed from this sidebar.
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
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.0)
        gmsh.option.setNumber("Mesh.Optimize", 1)
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
    """STEP → list of triangulated mesh bodies. Cached so sidebar changes skip re-conversion."""
    with tempfile.TemporaryDirectory() as d:
        sp = os.path.join(d, "model.step")
        with open(sp, "wb") as f:
            f.write(file_bytes)
        
        # Mesh and extract individual bodies inside a helper to avoid thread signal constraints
        import subprocess, sys, json, textwrap
        
        script = textwrap.dedent(f"""\
            import gmsh, json, sys
            
            def run():
                # ── Try Fast Path ──────────────────────────────────────────
                gmsh.initialize()
                gmsh.option.setNumber("General.Terminal", 0)
                gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.0)
                gmsh.option.setNumber("Mesh.Optimize", 1)
                gmsh.model.add("assembly")
                try:
                    gmsh.merge({repr(sp)})
                    gmsh.model.occ.synchronize()
                    vols = gmsh.model.occ.getEntities(3)
                    gmsh.model.mesh.generate(2)
                    
                    bodies = []
                    xmin, ymin, zmin, xmax, ymax, zmax = 1e9, 1e9, 1e9, -1e9, -1e9, -1e9
                    
                    for idx, (dim, tag) in enumerate(vols):
                        v_bbox = gmsh.model.occ.getBoundingBox(dim, tag)
                        xmin = min(xmin, v_bbox[0]); ymin = min(ymin, v_bbox[1]); zmin = min(zmin, v_bbox[2])
                        xmax = max(xmax, v_bbox[3]); ymax = max(ymax, v_bbox[4]); zmax = max(zmax, v_bbox[5])
                        
                        cx = (v_bbox[0] + v_bbox[3]) / 2.0
                        cy = (v_bbox[1] + v_bbox[4]) / 2.0
                        cz = (v_bbox[2] + v_bbox[5]) / 2.0
                        
                        bnd = gmsh.model.getBoundary([(dim, tag)], oriented=False, combined=True)
                        surf_tags = [abs(s[1]) for s in bnd if s[0] == 2]
                        
                        all_coords = []
                        all_node_tags = {{}}
                        triangles = []
                        
                        for s_tag in surf_tags:
                            node_tags, coords, _ = gmsh.model.mesh.getNodes(2, s_tag, includeBoundary=True)
                            coords = coords.reshape(-1, 3)
                            for t, coord in zip(node_tags, coords):
                                if t not in all_node_tags:
                                    all_node_tags[t] = len(all_coords)
                                    all_coords.append(coord.tolist())
                                    
                            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, s_tag)
                            for el_type, el_nodes in zip(elem_types, elem_node_tags):
                                if el_type == 2:
                                    el_nodes = el_nodes.reshape(-1, 3)
                                    for tri in el_nodes:
                                        triangles.append([
                                            all_node_tags[tri[0]],
                                            all_node_tags[tri[1]],
                                            all_node_tags[tri[2]]
                                        ])
                                        
                        bodies.append({{
                            "idx": idx,
                            "verts": all_coords,
                            "triangles": triangles,
                            "centroid": [cx, cy, cz],
                        }})
                    
                    print(json.dumps({{
                        "bodies": bodies,
                        "bounds": [xmin, xmax, ymin, ymax, zmin, zmax]
                    }}))
                    return
                except Exception as e:
                    pass
                finally:
                    gmsh.finalize()
                    
                # ── Fallback Path: Mesh Individually ─────────────────────────
                gmsh.initialize()
                gmsh.option.setNumber("General.Terminal", 0)
                gmsh.model.add("master")
                try:
                    gmsh.merge({repr(sp)})
                    gmsh.model.occ.synchronize()
                    vols = gmsh.model.occ.getEntities(3)
                    
                    v_info = []
                    xmin, ymin, zmin, xmax, ymax, zmax = 1e9, 1e9, 1e9, -1e9, -1e9, -1e9
                    for dim, tag in vols:
                        v_bbox = gmsh.model.occ.getBoundingBox(dim, tag)
                        xmin = min(xmin, v_bbox[0]); ymin = min(ymin, v_bbox[1]); zmin = min(zmin, v_bbox[2])
                        xmax = max(xmax, v_bbox[3]); ymax = max(ymax, v_bbox[4]); zmax = max(zmax, v_bbox[5])
                        cx = (v_bbox[0] + v_bbox[3]) / 2.0
                        cy = (v_bbox[1] + v_bbox[4]) / 2.0
                        cz = (v_bbox[2] + v_bbox[5]) / 2.0
                        v_info.append((dim, tag, v_bbox, [cx, cy, cz]))
                except Exception as e:
                    print(json.dumps({{"error": "Failed to read STEP file geometry: " + str(e)}}))
                    return
                finally:
                    gmsh.finalize()
                    
                bodies = []
                for idx, (dim, tag, bbox, centroid) in enumerate(v_info):
                    gmsh.initialize()
                    gmsh.option.setNumber("General.Terminal", 0)
                    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 2.0)
                    gmsh.option.setNumber("Mesh.Optimize", 1)
                    gmsh.model.add(f"body_{{idx}}")
                    try:
                        gmsh.merge({repr(sp)})
                        gmsh.model.occ.synchronize()
                        curr_vols = gmsh.model.occ.getEntities(3)
                        
                        to_remove = [v for i, v in enumerate(curr_vols) if i != idx]
                        if to_remove:
                            gmsh.model.occ.remove(to_remove, recursive=True)
                            gmsh.model.occ.synchronize()
                            
                        gmsh.model.mesh.generate(2)
                        
                        # Extract mesh
                        rem_dim, rem_tag = curr_vols[idx]
                        bnd = gmsh.model.getBoundary([(rem_dim, rem_tag)], oriented=False, combined=True)
                        surf_tags = [abs(s[1]) for s in bnd if s[0] == 2]
                        
                        all_coords = []
                        all_node_tags = {{}}
                        triangles = []
                        
                        for s_tag in surf_tags:
                            node_tags, coords, _ = gmsh.model.mesh.getNodes(2, s_tag, includeBoundary=True)
                            coords = coords.reshape(-1, 3)
                            for t, coord in zip(node_tags, coords):
                                if t not in all_node_tags:
                                    all_node_tags[t] = len(all_coords)
                                    all_coords.append(coord.tolist())
                                    
                            elem_types, elem_tags, elem_node_tags = gmsh.model.mesh.getElements(2, s_tag)
                            for el_type, el_nodes in zip(elem_types, elem_node_tags):
                                if el_type == 2:
                                    el_nodes = el_nodes.reshape(-1, 3)
                                    for tri in el_nodes:
                                        triangles.append([
                                            all_node_tags[tri[0]],
                                            all_node_tags[tri[1]],
                                            all_node_tags[tri[2]]
                                        ])
                        bodies.append({{
                            "idx": idx,
                            "verts": all_coords,
                            "triangles": triangles,
                            "centroid": centroid,
                        }})
                    except Exception:
                        # Coarse fallback
                        x0, y0, z0, x1, y1, z1 = bbox
                        verts = [
                            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]
                        ]
                        triangles = [
                            [0, 1, 2], [0, 2, 3],
                            [4, 5, 6], [4, 6, 7],
                            [0, 1, 5], [0, 5, 4],
                            [1, 2, 6], [1, 6, 5],
                            [2, 3, 7], [2, 7, 6],
                            [3, 0, 4], [3, 4, 7]
                        ]
                        bodies.append({{
                            "idx": idx,
                            "verts": verts,
                            "triangles": triangles,
                            "centroid": centroid,
                        }})
                    finally:
                        gmsh.finalize()
                        
                print(json.dumps({{
                    "bodies": bodies,
                    "bounds": [xmin, xmax, ymin, ymax, zmin, zmax]
                }}))

            run()
        """)
        
        r = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=120,
        )
        
        # Parse output
        out = None
        for line in reversed(r.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                try:
                    out = json.loads(line)
                    break
                except Exception:
                    pass
        if not out or "error" in out:
            raise RuntimeError(out.get("error") if out else (r.stderr.strip() or "gmsh mesh generation failed"))
            
        n_pts = sum(len(b["verts"]) for b in out["bodies"])
        n_cells = sum(len(b["triangles"]) for b in out["bodies"])
        out["n_pts"] = n_pts
        out["n_cells"] = n_cells
        return out

# ── Always-visible dual-panel layout ─────────────────────────────────────────
# ── Training completion banner ────────────────────────────────────────────────
if st.session_state.training_just_done and _CKPT_PATH.exists():
    _cv = st.session_state.last_cv_summary
    if _cv and _cv.get("mean_auc"):
        _n = len(_cv.get("fold_aucs", []))
        _banner_tag = (f" · {_n}-Fold CV  mean AUC {_cv['mean_auc']:.4f}±{_cv.get('std_auc',0):.4f}"
                       f"  mean AP {_cv['mean_ap']:.4f}±{_cv.get('std_ap',0):.4f}")
    else:
        _auc_tag = (f" · AUC {st.session_state.last_train_auc:.4f}"
                    if st.session_state.last_train_auc else "")
        _ap_tag  = (f" · AP {st.session_state.last_train_ap:.4f}"
                    if st.session_state.last_train_ap else "")
        _banner_tag = _auc_tag + _ap_tag
    st.success(
        f"🎉 **Training Complete{_banner_tag}** — Model ready!  "
        f"Upload a 3D model in the left panel to predict missing components.",
        icon="✅",
    )

# ── Always-visible dual-panel layout ─────────────────────────────────────────
col_left, col_right = st.columns(2)

CAMERAS = {
    "Isometric": dict(eye=dict(x=1.5, y=1.5, z=1.5)),
    "Top":       dict(eye=dict(x=0,   y=0,   z=2.5)),
    "Front":     dict(eye=dict(x=0,   y=-2.5, z=0 )),
    "Side":      dict(eye=dict(x=2.5, y=0,    z=0 )),
}
BG_MAP = {"white":"#ffffff","black":"#000000","grey":"#808080","lightgrey":"#d3d3d3"}

# ── RIGHT: 3D viewer (view only — not connected to inference) ─────────────────
with col_right:
    st.markdown(
        "<p style='font-size:0.8rem;font-weight:600;margin:0 0 0.2rem;color:#444;'>"
        "Results</p>",
        unsafe_allow_html=True,
    )

    if st.session_state.uploaded_bytes is None:
        st.markdown(
            '<div style="text-align:center;padding:0.5rem 0 2rem;color:#aaa;height:260px;'
            'display:flex;flex-direction:column;align-items:center;justify-content:flex-start;padding-top:1.5rem;">'
            '<div style="font-size:2.5rem;">📂</div>'
            '<p style="font-size:0.85rem;margin-top:0.4rem;">Upload a STEP file in the '
            'Input panel to view it here</p>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        # "Show parts" is now a native "🧩 Parts" legend group (click to
        # toggle, same as Open Joints / Suggested Shapes / etc).
        _vt_col1, _vt_col2 = st.columns([1, 1])
        with _vt_col1:
            _viewer_mode = st.segmented_control(
                "View", options=["Original", "Result"], default="Result",
                key="viewer_mode", label_visibility="collapsed",
            ) or "Result"
        with _vt_col2:
            _highlight_choice = st.segmented_control(
                "Highlight", options=["Highlight: On", "Highlight: Off"],
                default="Highlight: On", key="highlight_mode",
                label_visibility="collapsed",
            ) or "Highlight: On"
        _show_result = _viewer_mode == "Result"
        _highlight_not_assembled = _highlight_choice == "Highlight: On"

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

            import numpy as np
            fig = go.Figure()
            
            # Map GNN node centroids to viewer body centroids by minimum distance
            # ("Original" mode skips this — plain mesh, no result-derived labels)
            body_to_gnn = {b["idx"]: [] for b in m["bodies"]}
            if (
                _show_result
                and st.session_state.inference_result
                and "centroids" in st.session_state.inference_result
                and len(st.session_state.inference_result["centroids"]) > 0
            ):
                gnn_centroids = st.session_state.inference_result["centroids"]
                for g_idx, g_center in enumerate(gnn_centroids):
                    best_b_idx = None
                    min_dist = float('inf')
                    for b in m["bodies"]:
                        b_center = b["centroid"]
                        dist = sum((g_center[k] - b_center[k])**2 for k in range(3))
                        if dist < min_dist:
                            min_dist = dist
                            best_b_idx = b["idx"]
                    if best_b_idx is not None:
                        body_to_gnn[best_b_idx].append(g_idx)

            # Determine which nodes to highlight as "not assembled"
            # Rules (applied in combination):
            #   1. Isolated nodes (degree=0) — no contact at all
            #   2. Under-connected nodes — same part name as other instances but
            #      fewer connections (e.g. 3 displaced bolts vs 1 correct bolt)
            # Fallback: GNN predictions when neither rule fires
            highlighted_gnn_nodes = set()
            if (
                _show_result
                and st.session_state.inference_result
                and st.session_state.inference_done_for == st.session_state.uploaded_name
            ):
                _ir      = st.session_state.inference_result
                _degrees = _ir.get("node_degrees", [])
                _pnames  = _ir.get("part_names", [])

                # Group node indices by part basename (last "/" segment)
                def _bn(s):
                    if s and "/" in s:
                        return s.rstrip("/").split("/")[-1].strip()
                    return s or ""

                _grp: dict = {}
                for _gi, _gn in enumerate(_pnames):
                    _key = _bn(_gn) if _gn else f"__solo_{_gi}"
                    _grp.setdefault(_key, []).append(_gi)

                for _gn, _gnodes in _grp.items():
                    _degs_in_grp = [_degrees[_i] for _i in _gnodes if _i < len(_degrees)]
                    _max_deg = max(_degs_in_grp) if _degs_in_grp else 0
                    for _gi in _gnodes:
                        if _gi < len(_degrees):
                            if len(_gnodes) > 1:
                                # Multi-instance part: flag any under-connected instance
                                if _degrees[_gi] < _max_deg:
                                    highlighted_gnn_nodes.add(_gi)
                            else:
                                # Unique part: flag only if completely isolated
                                if _degrees[_gi] == 0:
                                    highlighted_gnn_nodes.add(_gi)

            # Trace indices per legend group, so the "Select all" buttons
            # below can restyle a whole group visible again in one click
            # without disturbing any other group's individually-toggled
            # state (that state lives only in the browser's Plotly widget —
            # Streamlit has no visibility into it, so this has to be a
            # client-side restyle, not a rerun).
            _group_trace_indices: dict = {
                "not_assembled": [], "parts": [], "potentially_missing": [],
                "open_joints": [], "suggested_shapes": [],
            }

            _first_not_assembled_legend = True
            _first_parts_legend = True
            for b in m["bodies"]:
                idx = b["idx"]
                verts = np.array(b["verts"])
                triangles = np.array(b["triangles"])

                if len(verts) == 0 or len(triangles) == 0:
                    continue
                
                mapped_gnn = body_to_gnn.get(idx, [])
                is_highlighted = (
                    _highlight_not_assembled
                    and any(g in highlighted_gnn_nodes for g in mapped_gnn)
                )

                _inf_pnames = (st.session_state.inference_result or {}).get("part_names", [])
                def _vname(g):
                    return (_inf_pnames[g] if g < len(_inf_pnames) and _inf_pnames[g]
                            else f"Part {g+1}")

                def _short(s, maxlen=24):
                    # Strip path prefixes like "Shapes/Assembly/Part/Part" → "Part"
                    s = s.rstrip("/").split("/")[-1].strip()
                    return s[:maxlen] + "…" if len(s) > maxlen else s

                def _wrap(s, width=22):
                    if len(s) <= width:
                        return s
                    cut = max(s.rfind(" ", 0, width), s.rfind("-", 0, width))
                    if cut <= 0:
                        cut = width
                    rest = s[cut:].lstrip()
                    return s[:cut].rstrip() + "<br>" + (rest[:width-1] + "…" if len(rest) > width else rest)

                if is_highlighted:
                    b_color  = "#f59e0b"
                    h_gnn    = [g for g in mapped_gnn if g in highlighted_gnn_nodes]
                    name     = ", ".join(_wrap(_short(_vname(g))) for g in h_gnn) + " ⚠"
                    show_leg = True
                else:
                    b_color  = mesh_color
                    name     = (_short(_vname(mapped_gnn[0])) if mapped_gnn else f"Part {idx+1}")
                    # Every part gets its own legend line (not just the
                    # first) so each is individually selectable, matching
                    # groupclick="toggleitem" below.
                    show_leg = True

                _legendgroup_kwargs = {}
                if is_highlighted:
                    _legendgroup_kwargs["legendgroup"] = "not_assembled"
                    if _first_not_assembled_legend:
                        _legendgroup_kwargs["legendgrouptitle"] = dict(text="🟠 Not Assembled")
                        _first_not_assembled_legend = False
                else:
                    # Grouped under one "🧩 Parts" header, but each body still
                    # gets its own legend line so it can be selected/hidden
                    # individually (groupclick="toggleitem" below).
                    _legendgroup_kwargs["legendgroup"] = "parts"
                    if _first_parts_legend:
                        _legendgroup_kwargs["legendgrouptitle"] = dict(text="🧩 Parts")
                        _first_parts_legend = False

                fig.add_trace(go.Mesh3d(
                    x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                    i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
                    color=b_color, opacity=opacity, flatshading=False,
                    name=name,
                    showlegend=show_leg,
                    hoverinfo="name",
                    lighting=dict(ambient=0.6, diffuse=0.9, specular=0.5,
                                   roughness=0.3, fresnel=0.4),
                    lightposition=dict(x=200, y=300, z=400),
                    **_legendgroup_kwargs,
                ))
                _group_trace_indices["not_assembled" if is_highlighted else "parts"].append(
                    len(fig.data) - 1
                )

            # Orange cross markers at estimated locations of missing components
            _pot = ((st.session_state.inference_result or {}).get("potentially_missing", [])
                    if _show_result else [])
            for _pot_i, _mc in enumerate(_pot):
                _cx, _cy, _cz = _mc["centroid"]
                _mn = _mc["name"]
                _ms = (_mn[:26] + "…") if len(_mn) > 26 else _mn
                fig.add_trace(go.Scatter3d(
                    x=[_cx], y=[_cy], z=[_cz],
                    mode="markers",
                    marker=dict(symbol="cross", size=14, color="#f97316",
                                line=dict(color="#ffffff", width=2)),
                    name=f"❓ {_ms}",
                    showlegend=True,
                    legendgroup="potentially_missing",
                    legendgrouptitle=(dict(text="❓ Potentially Missing") if _pot_i == 0 else None),
                    hovertext=f"Missing component:<br>{_mn}",
                    hoverinfo="text",
                ))
                _group_trace_indices["potentially_missing"].append(len(fig.data) - 1)

            # ── Amber mesh patches for open joints (Octree surface analysis) ──
            # Each amber surface = an open mating joint where a missing component
            # should be assembled (derived from Borah & Borah 2020 Octree concept).
            _open_surfs_view = ((st.session_state.inference_result or {}).get("open_surfaces", [])
                                if _show_result else [])
            for _osi, _os in enumerate(_open_surfs_view):
                _ov = _os.get("vertices", [])
                _ot = _os.get("triangles", [])
                if not _ov or not _ot:
                    continue
                _ov_arr = np.array(_ov)
                _ot_arr = np.array(_ot)
                if len(_ov_arr) < 3 or len(_ot_arr) < 1:
                    continue
                _os_bi = _os.get("body_idx", 0)
                _os_ar = _os.get("area_ratio", 0)
                fig.add_trace(go.Mesh3d(
                    x=_ov_arr[:, 0], y=_ov_arr[:, 1], z=_ov_arr[:, 2],
                    i=_ot_arr[:, 0], j=_ot_arr[:, 1], k=_ot_arr[:, 2],
                    color="#84cc16",
                    opacity=0.82,
                    flatshading=True,
                    name=f"⬡ Body {_os_bi + 1}  ({int(_os_ar * 100)}%)",
                    showlegend=True,
                    legendgroup="open_joints",
                    legendgrouptitle=(dict(text="⬡ Open Joints") if _osi == 0 else None),
                    hovertext=(
                        f"This area needs components to be assembled<br>"
                        f"Body {_os_bi + 1} · {_os_ar:.0%} of body surface area"
                    ),
                    hoverinfo="text",
                    lighting=dict(ambient=0.7, diffuse=0.6, specular=0.1,
                                  roughness=0.8, fresnel=0.1),
                    lightposition=dict(x=100, y=100, z=100),
                ))
                _group_trace_indices["open_joints"].append(len(fig.data) - 1)

            # ── Ghost overlay: AI-suggested missing-part shapes (Phase 3) ──────
            # Red translucent mesh at the matched open-joint location —
            # best-effort shape + placement, not a verified/exact fit.
            _gen_parts_view = ((st.session_state.inference_result or {}).get("generated_parts", [])
                               if _show_result else [])
            for _gp_i, _gp in enumerate(_gen_parts_view):
                _gpv = _gp.get("vertices", [])
                _gpt = _gp.get("triangles", [])
                if not _gpv or not _gpt:
                    continue
                _gpv_arr = np.array(_gpv)
                _gpt_arr = np.array(_gpt)
                if len(_gpv_arr) < 3 or len(_gpt_arr) < 1:
                    continue
                _gp_type = str(_gp.get("type", "component")).capitalize()
                _gp_src  = _gp.get("source", "generated")
                _gp_conf = _gp.get("confidence", 0.0)
                _gp_fit  = _gp.get("fit_score", 0.0)
                _gp_icon = "🔄" if _gp_src == "retrieved" else "✨"
                # Fastener sequence color coding (user spec, 2026-08-22):
                # bolt=red, nut=dark red, washer=brown; anything else keeps
                # the original red so non-fastener suggested shapes are
                # unaffected.
                _GP_COLORS = {"bolt": "#ef4444", "nut": "#7f1d1d", "washer": "#92400e"}
                _gp_color = _GP_COLORS.get(str(_gp.get("type", "")).lower(), "#ef4444")
                fig.add_trace(go.Mesh3d(
                    x=_gpv_arr[:, 0], y=_gpv_arr[:, 1], z=_gpv_arr[:, 2],
                    i=_gpt_arr[:, 0], j=_gpt_arr[:, 1], k=_gpt_arr[:, 2],
                    color=_gp_color,
                    opacity=0.40,
                    flatshading=True,
                    name=f"{_gp_icon} Suggested {_gp_type}",
                    showlegend=True,
                    legendgroup="suggested_shapes",
                    legendgrouptitle=(dict(text="🪄 Suggested Shapes") if _gp_i == 0 else None),
                    hovertext=(
                        f"AI-suggested {_gp_type.lower()}<br>"
                        f"{'Retrieved from part bank' if _gp_src == 'retrieved' else 'AI-generated (VAE)'}"
                        f" · location fit {_gp_fit:.0%} · shape confidence {_gp_conf:.0%}"
                    ),
                    hoverinfo="text",
                    lighting=dict(ambient=0.9, diffuse=0.3, specular=0.6,
                                  roughness=0.2, fresnel=0.6),
                    lightposition=dict(x=150, y=200, z=300),
                ))
                _group_trace_indices["suggested_shapes"].append(len(fig.data) - 1)

            # "Select all" buttons — one per group that actually has traces
            # this run — client-side restyle (no Streamlit rerun) so they
            # don't disturb whatever the user has individually toggled
            # elsewhere in the legend.
            _select_all_labels = {
                "not_assembled": "🟠 All not-assembled",
                "parts": "🧩 All parts",
                "potentially_missing": "❓ All missing",
                "open_joints": "⬡ All joints",
                "suggested_shapes": "🪄 All suggested",
            }
            _select_all_buttons = [
                dict(
                    label=_select_all_labels[_grp_key],
                    method="restyle",
                    # args/args2 = toggle: first click selects all, next
                    # click deselects all, alternating on each press.
                    args=[{"visible": True}, _grp_indices],
                    args2=[{"visible": False}, _grp_indices],
                )
                for _grp_key, _grp_indices in _group_trace_indices.items()
                if _grp_indices
            ]

            fig.update_layout(
                scene=dict(
                    bgcolor=BG_MAP[bg_color], aspectmode="data",
                    xaxis=dict(showgrid=show_grid, title="X"),
                    yaxis=dict(showgrid=show_grid, title="Y"),
                    zaxis=dict(showgrid=show_grid, title="Z"),
                ),
                scene_camera=CAMERAS[view_preset],
                margin=dict(l=0, r=0, b=(28 if _select_all_buttons else 0), t=0),
                paper_bgcolor="rgba(0,0,0,0)",
                height=400,
                showlegend=True,
                legend=dict(
                    x=0.01, y=0.99,
                    xanchor="left", yanchor="top",
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor="rgba(0,0,0,0)",
                    borderwidth=0,
                    font=dict(size=9, color="#000000"),
                    itemsizing="constant",
                    # "toggleitem" so each entry can still be selected/hidden
                    # individually — grouping is just visual organization
                    # (headers), not a forced all-or-nothing click.
                    groupclick="toggleitem",
                    grouptitlefont=dict(size=9, color="#000000"),
                ),
                updatemenus=(
                    [
                        dict(
                            type="buttons",
                            direction="right",
                            x=0.0, y=-0.06,
                            xanchor="left", yanchor="top",
                            showactive=False,
                            bgcolor="rgba(255,255,255,0.9)",
                            bordercolor="#2a4060",
                            borderwidth=1,
                            pad=dict(l=3, r=3, t=1, b=1),
                            font=dict(size=8, color="#2a4060"),
                            buttons=_select_all_buttons,
                        )
                    ]
                    if _select_all_buttons else []
                ),
            )

            st.plotly_chart(fig, use_container_width=True)

            b = m["bounds"]
            st.markdown(
                f'<div style="font-size:0.7rem;color:#888;margin-top:-0.4rem;">'
                f'✅ {st.session_state.uploaded_name} &nbsp;·&nbsp; '
                f'{m["n_pts"]:,} pts · {m["n_cells"]:,} faces</div>',
                unsafe_allow_html=True,
            )

# ── LEFT: Prediction panel — active only after training is complete ────────────
with col_left:
    st.markdown(
        "<p style='font-size:0.8rem;font-weight:600;margin:0 0 0.2rem;color:#444;'>"
        "Input</p>",
        unsafe_allow_html=True,
    )
    if not _CKPT_PATH.exists():
        # Training not done yet
        st.markdown(_placeholder_panel_html(None, False), unsafe_allow_html=True)

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
            # Synchronize to the right viewer so they don't have to upload twice
            st.session_state.uploaded_bytes = st.session_state.pred_bytes
            st.session_state.uploaded_name  = st.session_state.pred_name
            st.session_state.mesh_logged    = False
            st.rerun()

    else:
        # Run inference (once per file)
        if st.session_state.inference_done_for != st.session_state.pred_name:
            with st.spinner("🔍 Predicting missing components…"):
                log("🔍  Running GNN inference…")
                _res = _run_inference(
                    st.session_state.pred_bytes,
                    source_dir=st.session_state.source_3d_dir,
                    uploaded_name=st.session_state.pred_name or "",
                )
                st.session_state.inference_result   = _res
                st.session_state.inference_done_for = st.session_state.pred_name
                if "error" not in _res:
                    _n = len(_res.get("missing_links", []))
                    log(f"✅  Prediction done — {_n} missing link(s) found")
                else:
                    log(f"⚠️  Inference: {_res.get('error','')[:80]}")
                st.rerun()

        _ph = _placeholder_panel_html(st.session_state.inference_result, True)
        if _ph is not None:
            st.markdown(_ph, unsafe_allow_html=True)
        else:
            _panel_header, _panel_sections = _build_panel_sections(
                st.session_state.inference_result
            )
            with st.container(height=400, border=True):
                st.markdown(
                    f'<p style="font-size:0.72rem;color:#888;margin:0 0 4px;">'
                    f'{_panel_header}</p>',
                    unsafe_allow_html=True,
                )
                if _panel_sections:
                    for _sec in _panel_sections:
                        with st.expander(
                            f'{_sec["icon"]} {_sec["title"]}',
                            expanded=False,
                            key=f'exp_{_sec["key"]}',
                        ):
                            st.markdown(_sec["html"], unsafe_allow_html=True)
                else:
                    st.markdown(
                        "<p style='color:#4caf82;font-size:0.78rem;'>"
                        "✓ Assembly appears complete — no issues found.</p>",
                        unsafe_allow_html=True,
                    )
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("🔄 Predict another", key="reset_pred", use_container_width=True):
                st.session_state.pred_bytes         = None
                st.session_state.pred_name          = None
                st.session_state.inference_result   = None
                st.session_state.inference_done_for = ""
                st.session_state.aida_explanation   = None
                st.session_state.aida_explain_for   = ""
                # Synchronize reset to the left viewer
                st.session_state.uploaded_bytes     = None
                st.session_state.uploaded_name      = None
                st.session_state.mesh_logged        = False
                st.rerun()
        with btn_col2:
            if st.button("🗑️ Reset All & Log", key="reset_main_log", use_container_width=True):
                st.session_state.pred_bytes         = None
                st.session_state.pred_name          = None
                st.session_state.inference_result   = None
                st.session_state.inference_done_for = ""
                st.session_state.aida_explanation   = None
                st.session_state.aida_explain_for   = ""
                st.session_state.uploaded_bytes     = None
                st.session_state.uploaded_name      = None
                st.session_state.mesh_logged        = False
                st.session_state.activity_log       = []
                st.rerun()

# ── AIDA panel — below dual viewer ───────────────────────────────────────────
_inf = st.session_state.inference_result
_pred_name = st.session_state.pred_name or ""

# Trigger explanation once per prediction result
if (
    _inf and "error" not in _inf
    and st.session_state.aida_explain_for != _pred_name
    and _pred_name
):
    with st.spinner("🤖 AIDA is preparing the engineering explanation…"):
        _expl = _run_aida_explain(_inf)
        st.session_state.aida_explanation = _expl
        st.session_state.aida_explain_for = _pred_name
        log("🤖  AIDA explanation ready")

# Render AIDA panel
_expl_text = st.session_state.aida_explanation
if _expl_text:
    import html as _html
    _lines = []
    for _ln in _expl_text.splitlines():
        _esc = _html.escape(_ln)
        if _esc.startswith("===") and _esc.endswith("==="):
            # Section heading — bright sky-blue, bold
            _lines.append(
                f'<p style="font-size:0.82rem;font-weight:700;color:#7dd3fc;'
                f'margin:10px 0 4px;">{_esc}</p>'
            )
        elif _esc.startswith(("* ", "- ", "• ")):
            # Bullet — white text
            _bullet = _esc[2:]
            _lines.append(
                f'<p style="font-size:0.85rem;color:#e0f2fe;line-height:1.7;'
                f'margin:2px 0 2px 8px;">• {_bullet}</p>'
            )
        elif _esc.strip() == "":
            _lines.append('<div style="height:4px;"></div>')
        else:
            # Plain text (non-bullet body lines) — same bright white
            _lines.append(
                f'<p style="font-size:0.85rem;color:#e0f2fe;line-height:1.7;'
                f'margin:2px 0;">{_esc}</p>'
            )
    _body = "".join(_lines)
elif _inf and "error" in _inf:
    _body = '<p style="font-size:0.84rem;color:#fca5a5;margin:0;">Inference error — no explanation available.</p>'
else:
    _body = (
        '<p style="font-size:0.84rem;color:#7dd3fc;font-style:italic;margin:0;">'
        'Upload a 3D model to the left panel — AIDA will explain the missing '
        'component predictions in engineering language once inference is complete.'
        '</p>'
    )

st.markdown(
    f"""
    <div style="
        background:linear-gradient(135deg,#0a1628,#0f2744);
        border:1px solid #1d4ed8;
        border-left:4px solid #38bdf8;
        border-radius:10px;
        margin-top:0.8rem;
        overflow:hidden;
        box-shadow:0 0 18px rgba(56,189,248,0.15);
    ">
        <div style="
            padding:0.6rem 1.2rem;
            border-bottom:1px solid #1e3a6e;
            display:flex;
            align-items:baseline;
            gap:0.6rem;
            background:linear-gradient(90deg,#0c3566,#0f2744);
        ">
            <span style="color:#38bdf8;font-size:0.95rem;font-weight:700;letter-spacing:0.3px;">🤖 AIDA explains</span>
            <span style="color:#93c5fd;font-size:0.76rem;font-weight:500;">
                Gemini AI · engineering interpretation of GNN predictions
            </span>
        </div>
        <div style="padding:1.0rem 1.2rem;min-height:70px;">
            {_body}
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

log("✅  3D viewer ready")

# ── Final activity log render ─────────────────────────────────────────────────
log_slot.markdown(render_log(st.session_state.activity_log[-30:]),
                  unsafe_allow_html=True)

# ── Auto-poll training log every 3 s while training is active ─────────────────
if _is_training():
    import time
    time.sleep(3)
    st.rerun()

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
