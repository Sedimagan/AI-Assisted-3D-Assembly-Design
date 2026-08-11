"""
api.py — Back-end REST API
Exposes GNN inference and Skills AI endpoints over HTTP.

Run:  uvicorn back_end.api:app --reload --port 8000
  or: python back_end/api.py
"""

from __future__ import annotations

import math
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import uvicorn
import yaml
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from model import build_model
from dataset import _parse_step, _synthetic_graph, COMP_TYPES
from infer import predict_missing, load_ranker, predict_next_component
from skills_agent import AssemblySkillsAgent
from surface_analyzer import analyze_open_surfaces

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI-Assisted 3D Assembly Design API",
    description="GNN inference + Gemini Skills AI for CAD assembly completion",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup: load model + agent ───────────────────────────────────────────────

_gnn = _lp = _device = None
_agent: Optional[AssemblySkillsAgent] = None
_ckpt_serving  = Path(__file__).parent / "checkpoints" / "best_serving.pt"
_ckpt_fallback = Path(__file__).parent / "checkpoints" / "best_overall.pt"
_ckpt_path = _ckpt_serving if _ckpt_serving.exists() else _ckpt_fallback

_nr = _type_prototypes = _nr_comp_types = None
_ranker_loaded = False
_ranker_path = Path(__file__).parent / "checkpoints" / "node_ranker.pt"

@app.on_event("startup")
def startup():
    global _gnn, _lp, _device, _agent
    global _nr, _type_prototypes, _nr_comp_types, _ranker_loaded

    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    mc = cfg["model"]

    _gnn, _lp, _device = build_model(
        in_dim=mc["in_dim"], out_dim=mc["out_dim"],
        hidden=mc["hidden_dim"], heads=mc["heads"],
        dropout=mc["dropout"], edge_dim=mc["edge_dim"],
        encoder_type=mc.get("encoder_type", "rgat"),
    )

    if _ckpt_path.exists():
        ckpt = torch.load(_ckpt_path, map_location=_device, weights_only=False)
        _gnn.load_state_dict(ckpt["gnn"])
        _lp.load_state_dict(ckpt["lp"])
        _gnn.eval(); _lp.eval()
        print(f"  ✓ Checkpoint loaded (val AUC={ckpt['auc']:.4f})")
    else:
        print("  ⚠  No checkpoint found — using untrained model.")

    if _ranker_path.exists():
        try:
            _nr, _type_prototypes, _nr_comp_types, _ = load_ranker(
                str(_ranker_path), _gnn, _device
            )
            _ranker_loaded = True
            print("  ✓ NodeRanker loaded")
        except Exception as e:
            print(f"  ⚠  NodeRanker failed to load — {e}")
    else:
        print("  ⚠  No NodeRanker checkpoint found — run train_ranker.py to enable it.")

    _agent = AssemblySkillsAgent()
    print(f"  ✓ Skills AI agent ready ({_agent.status()})")


# ── Schemas ───────────────────────────────────────────────────────────────────

class ComponentNode(BaseModel):
    type_index: int = 0           # 0–7 maps to COMP_TYPES
    volume:       float = 0.5
    surface_area: float = 0.5
    bbox_x:       float = 1.0
    bbox_y:       float = 1.0
    bbox_z:       float = 1.0

class AssemblyGraph(BaseModel):
    nodes: List[ComponentNode]
    edges: List[Tuple[int, int]]  # (src, dst) pairs
    top_k: int = 5

class ExplainRequest(BaseModel):
    missing: List[Tuple[Tuple[int, int], float]]
    recommendations: List[Tuple[str, float]]
    context: str = ""

class IdentifyRequest(BaseModel):
    volume:       float
    surface_area: float
    bbox:         Tuple[float, float, float]
    n_mates:      int = 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_pyg(req: AssemblyGraph):
    """Build a 22-dim node / 6-dim edge PyG graph matching dataset.py's schema."""
    from torch_geometric.data import Data
    x_rows = []
    for nd in req.nodes:
        oh = [0.0] * 8
        oh[nd.type_index] = 1.0
        dx, dy, dz = nd.bbox_x, nd.bbox_y, nd.bbox_z
        bbox_max = max(dx, dy, dz, 1e-9)
        ext = sorted([dx, dy, dz])
        elongation = ext[2] / (ext[1] + 1e-9)
        flatness = ext[0] / (ext[2] + 1e-9)
        aspect_xy = dx / (dy + 1e-9)
        aspect_yz = dy / (dz + 1e-9)
        vol = nd.volume
        sa = nd.surface_area
        sphericity = min(1.0, (math.pi ** (1/3))
                         * ((6 * vol) ** (2/3))
                         / (sa + 1e-9))
        sav = sa / (vol + 1e-9)
        feat = oh + [
            min(13.8, math.log1p(max(0.0, vol))),   # [8]
            min(11.5, math.log1p(max(0.0, sa))),     # [9]
            dx / bbox_max,                            # [10]
            dy / bbox_max,                            # [11]
            dz / bbox_max,                            # [12]
            elongation,                               # [13]
            flatness,                                 # [14]
            aspect_xy,                                # [15]
            aspect_yz,                                # [16]
            sphericity,                               # [17]
            0.0,                                      # [18] SDF mean
            0.0,                                      # [19] SDF variance
            sav,                                      # [20] SA/V
            0.0,                                      # [21] log1p(n_holes)
        ]
        x_rows.append(feat)
    x = torch.tensor(x_rows, dtype=torch.float)
    if req.edges:
        src = [e[0] for e in req.edges]
        dst = [e[1] for e in req.edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]] * len(src),
                                 dtype=torch.float)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr = torch.zeros(0, 6, dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":      "ok",
        "model_loaded": _ckpt_path.exists(),
        "ranker_loaded": _ranker_loaded,
        "agent":        _agent.status() if _agent else None,
        "comp_types":   COMP_TYPES,
    }


@app.post("/predict/missing")
def api_predict_missing(req: AssemblyGraph):
    """Predict missing component connections in a partial assembly."""
    if _gnn is None:
        raise HTTPException(503, "Model not loaded")
    graph = _to_pyg(req)
    results = predict_missing(_gnn, _lp, graph, _device, top_k=req.top_k)
    return {"missing_links": [
        {"src": u, "dst": v, "confidence": s} for (u, v), s in results
    ]}


@app.post("/predict/next-component")
def api_predict_next_component(req: AssemblyGraph):
    """Rank candidate next-component types by cosine similarity to the
    partial assembly's context vector (Phase 2, NodeRanker)."""
    if _nr is None:
        raise HTTPException(503, "NodeRanker not loaded — run train_ranker.py first")
    graph = _to_pyg(req)
    results = predict_next_component(_gnn, _nr, _type_prototypes, graph, _device)
    return {"next_component_suggestions": [
        {"type": t, "score": s} for t, s in results
    ]}


@app.post("/explain")
def api_explain(req: ExplainRequest):
    """Use Gemini Skills AI to explain GNN predictions in engineering terms."""
    if _agent is None:
        raise HTTPException(503, "Skills agent not ready")
    explanation = _agent.explain_prediction(
        missing=req.missing,
        recs=req.recommendations,
        context=req.context,
    )
    return {"explanation": explanation, "agent": _agent.status()}


@app.post("/identify")
def api_identify(req: IdentifyRequest):
    """Identify component type from geometric features using Skills AI."""
    if _agent is None:
        raise HTTPException(503, "Skills agent not ready")
    result = _agent.identify_component(
        volume=req.volume,
        surface_area=req.surface_area,
        bbox=req.bbox,
        n_mates=req.n_mates,
    )
    return {"identification": result}


@app.post("/analyze/step")
async def api_analyze_step(file: UploadFile = File(...)):
    """
    Full Phase 1 analysis of a STEP file upload.

    Runs both the GNN link predictor and the Octree open-surface detector,
    returning combined results in a single response.
    """
    if not file.filename:
        raise HTTPException(400, "No file uploaded")
    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".step", ".stp"}:
        raise HTTPException(400, f"Expected .step/.stp file, got {suffix}")

    tmp_dir = tempfile.mkdtemp(prefix="assembly_")
    tmp_path = Path(tmp_dir) / file.filename
    try:
        content = await file.read()
        tmp_path.write_bytes(content)

        open_surfaces = []
        try:
            raw_surfaces = analyze_open_surfaces(str(tmp_path))
            for s in raw_surfaces:
                open_surfaces.append({
                    "centroid":    s.get("centroid", [0, 0, 0]),
                    "area":       s.get("area", 0),
                    "area_ratio":  s.get("area_ratio", 0),
                    "body_idx":    s.get("body_idx", -1),
                    "vertices":    s.get("vertices", []),
                    "triangles":   s.get("triangles", []),
                    "normal_hint": s.get("normal_hint", [0, 0, 1]),
                    "bbox":        s.get("bbox"),
                    "is_hole":     s.get("is_hole", False),
                })
        except Exception as e:
            open_surfaces = []
            surf_error = str(e)
        else:
            surf_error = None

        missing_links = []
        next_component_suggestions = []
        graph_info = None
        try:
            graph = _parse_step(str(tmp_path))
            if graph is not None and graph.num_nodes >= 1:
                graph_info = {
                    "num_nodes": graph.num_nodes,
                    "num_edges": graph.edge_index.size(1),
                }
                if graph.num_nodes >= 2:
                    if _gnn is not None and _lp is not None:
                        results = predict_missing(
                            _gnn, _lp, graph, _device, top_k=10,
                        )
                        missing_links = [
                            {"src": u, "dst": v, "confidence": s}
                            for (u, v), s in results
                        ]
                    else:
                        graph_info["warning"] = "Model not loaded — no predictions"
                if _ranker_loaded:
                    nc_results = predict_next_component(
                        _gnn, _nr, _type_prototypes, graph, _device, top_k=5,
                    )
                    next_component_suggestions = [
                        {"type": t, "score": s} for t, s in nc_results
                    ]
        except Exception as e:
            graph_info = {"error": str(e)}

        return {
            "filename":      file.filename,
            "open_surfaces": open_surfaces,
            "surf_error":    surf_error,
            "missing_links": missing_links,
            "next_component_suggestions": next_component_suggestions,
            "graph":         graph_info,
        }
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
        try:
            Path(tmp_dir).rmdir()
        except OSError:
            pass


@app.get("/docs-summary")
def docs_summary():
    return {
        "endpoints": {
            "GET  /health":                 "Service + model status",
            "POST /predict/missing":        "Missing component link prediction (JSON graph)",
            "POST /predict/next-component": "Next-component type ranking (JSON graph, NodeRanker)",
            "POST /analyze/step":           "Full STEP file analysis (GNN + ranker + open surfaces)",
            "POST /explain":                "Gemini AI explanation of predictions",
            "POST /identify":               "Component type identification from geometry",
        }
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
