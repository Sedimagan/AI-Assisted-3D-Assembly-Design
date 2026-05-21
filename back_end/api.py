"""
api.py — Back-end REST API
Exposes GNN inference and Skills AI endpoints over HTTP.

Run:  uvicorn back_end.api:app --reload --port 8000
  or: python back_end/api.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:
    pass

from model import build_model
from dataset import _synthetic_graph, COMP_TYPES
from infer import predict_missing
from skills_agent import AssemblySkillsAgent

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
_ckpt_path = Path(__file__).parent / "checkpoints" / "best.pt"

@app.on_event("startup")
def startup():
    global _gnn, _lp, _device, _agent

    cfg_path = Path(__file__).parent / "config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    mc = cfg["model"]

    _gnn, _lp, _device = build_model(
        in_dim=mc["in_dim"], out_dim=mc["out_dim"],
        hidden=mc["hidden_dim"], heads=mc["heads"],
        dropout=mc["dropout"], edge_dim=mc["edge_dim"],
    )

    if _ckpt_path.exists():
        ckpt = torch.load(_ckpt_path, map_location=_device, weights_only=False)
        _gnn.load_state_dict(ckpt["gnn"])
        _lp.load_state_dict(ckpt["lp"])
        _gnn.eval(); _lp.eval()
        print(f"  ✓ Checkpoint loaded (val AUC={ckpt['auc']:.4f})")
    else:
        print("  ⚠  No checkpoint found — using untrained model.")

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
    from torch_geometric.data import Data
    n = len(req.nodes)
    x_rows = []
    for nd in req.nodes:
        oh = [0.0] * 8; oh[nd.type_index] = 1.0
        x_rows.append(oh + [nd.volume, nd.surface_area,
                             nd.bbox_x, nd.bbox_y, nd.bbox_z])
    x = torch.tensor(x_rows, dtype=torch.float)
    if req.edges:
        src = [e[0] for e in req.edges]
        dst = [e[1] for e in req.edges]
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.ones(len(src), 2)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr  = torch.zeros(0, 2)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":      "ok",
        "model_loaded": _ckpt_path.exists(),
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


@app.get("/docs-summary")
def docs_summary():
    return {
        "endpoints": {
            "GET  /health":          "Service + model status",
            "POST /predict/missing": "Missing component link prediction (Phase 1)",
            "POST /explain":         "Gemini AI explanation of predictions",
            "POST /identify":        "Component type identification from geometry",
        }
    }


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
