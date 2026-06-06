"""
skills_agent.py — Gemini AI Orchestrator
Loads a skills profile from skills/<profile>.yaml and exposes domain-aware
methods for interpreting GNN predictions, explaining assemblies, and
orchestrating the full pipeline.

Usage
-----
from skills_agent import AssemblySkillsAgent

agent = AssemblySkillsAgent()
reply = agent.explain_prediction(missing=[(( 2, 5), 0.87)], recs=[("bearing", 0.83)])
print(reply)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import yaml

# Load .env from project root (two levels up from back_end/)
try:
    from dotenv import load_dotenv
    _env = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(_env)
except ImportError:
    pass  # python-dotenv not installed; rely on shell environment

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


# ── Skills loader ─────────────────────────────────────────────────────────────

def _load_skills(profile: str) -> dict:
    skills_dir = Path(__file__).resolve().parents[1] / "skills"
    path = skills_dir / f"{profile}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Skills profile not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def _build_system_prompt(skills: dict) -> str:
    lines = [skills["persona"].strip(), "\n## Domain Skills\n"]
    for s in skills.get("skills", []):
        lines.append(f"### {s['name']}")
        lines.append(s["description"].strip())
        kw = ", ".join(s.get("keywords", [])[:8])
        if kw:
            lines.append(f"Key concepts: {kw}")
        lines.append("")
    lines.append("## Response Rules")
    for rule in skills.get("response_rules", []):
        lines.append(f"- {rule}")
    return "\n".join(lines)


# ── Agent ─────────────────────────────────────────────────────────────────────

class AssemblySkillsAgent:
    """
    Gemini-backed AI agent with mechanical engineering and 3D assembly skills.
    All public methods return plain-text strings safe to display in the UI.
    """

    def __init__(self, profile: str | None = None):
        self.profile = profile or os.getenv("SKILLS_PROFILE", "engineering_3d_assembly")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
        self.temperature = float(os.getenv("SKILLS_TEMPERATURE", "0.3"))
        self.max_tokens  = int(os.getenv("SKILLS_MAX_OUTPUT_TOKENS", "300"))

        skills = _load_skills(self.profile)
        self._system_prompt = _build_system_prompt(skills)
        self._agent_name    = skills.get("name", "AIDA")

        self._model = None
        if _GENAI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            self._model = genai.GenerativeModel(
                model_name         = self.model_name,
                system_instruction = self._system_prompt,
                generation_config  = {
                    "temperature":        self.temperature,
                    "max_output_tokens":  self.max_tokens,
                    "top_p":              float(os.getenv("SKILLS_TOP_P", "0.9")),
                },
            )
        else:
            if not _GENAI_AVAILABLE:
                print("[SkillsAgent] google-generativeai not installed. "
                      "Run: pip install google-generativeai")
            elif not os.getenv("GEMINI_API_KEY"):
                print("[SkillsAgent] GEMINI_API_KEY not set in .env")

    # ── Internal helper ───────────────────────────────────────────────────────

    def _ask(self, prompt: str, max_tokens: int = 1800) -> str:
        if self._model is None:
            return (
                "[SkillsAgent offline] Gemini is not configured. "
                "Set GEMINI_API_KEY in .env to enable AI explanations."
            )
        try:
            import google.generativeai as genai
            cfg = genai.types.GenerationConfig(max_output_tokens=max_tokens)
            resp = self._model.generate_content(prompt, generation_config=cfg)
            return resp.text.strip()
        except Exception as exc:
            return f"[SkillsAgent error] {exc}"

    # ── Public API ────────────────────────────────────────────────────────────

    def explain_prediction(
        self,
        missing: List[Tuple[Tuple[int, int], float]],
        recs:    List[Tuple[str, float]],
        context: str = "",
    ) -> str:
        """
        Explain GNN link-prediction and ranking results in engineering terms.

        Parameters
        ----------
        missing : list of ((src, dst), confidence) — predicted missing edges
        recs    : list of (component_type, score) — ranked next-component suggestions
        context : optional free-text description of the assembly
        """
        missing_txt = "\n".join(
            f"  - {u} ↔ {v} — confidence {s:.3f}"
            for (u, v), s in missing
        ) or "  (none)"

        recs_txt = "\n".join(
            f"  {i+1}. {comp:<14} score={score:.3f}"
            for i, (comp, score) in enumerate(recs)
        ) or "  (none)"

        prompt = f"""
GNN predictions for a partial CAD assembly.
{f"Context: {context}" if context else ""}

Missing links (absent but predicted):
{missing_txt}

Next-component recommendations:
{recs_txt}

Respond with exactly 3–4 concise bullet points (one sentence each):
• What the top missing link likely represents mechanically
• What mate constraint it suggests (coincident / concentric / etc.)
• Whether the confidence level is high/medium/low and what action to take
• (Optional) One unusual finding worth verifying in CAD

No prose paragraphs. No headings. Bullets only. Maximum 4 lines total.
"""
        return self._ask(prompt)

    def identify_component(
        self,
        volume: float,
        surface_area: float,
        bbox: Tuple[float, float, float],
        n_mates: int = 0,
    ) -> str:
        """Identify likely component type from geometric features."""
        dx, dy, dz = bbox
        aspect_xy = max(dx, dy) / (min(dx, dy) + 1e-9)
        aspect_z  = dz / (max(dx, dy) + 1e-9)

        prompt = f"""
A CAD component has the following geometric properties:
- Volume: {volume:.4f} (normalised)
- Surface area: {surface_area:.4f} (normalised)
- Bounding box: {dx:.3f} × {dy:.3f} × {dz:.3f}
- XY aspect ratio: {aspect_xy:.2f}
- Z aspect ratio:  {aspect_z:.2f}
- Number of mate connections in assembly: {n_mates}

Based on these features, what type of mechanical component is this most likely to be?
Choose from: body, fastener, bearing, shaft, plate, housing, gear, other.
Provide a brief engineering justification (2–3 sentences).
"""
        return self._ask(prompt)

    def suggest_assembly_sequence(
        self,
        components: List[str],
        mate_edges: List[Tuple[int, int]],
    ) -> str:
        """Suggest an optimal assembly sequence given components and their connections."""
        comp_list = "\n".join(f"  {i}: {c}" for i, c in enumerate(components))
        edge_list = ", ".join(f"({u}→{v})" for u, v in mate_edges) or "none"

        prompt = f"""
A mechanical assembly contains the following components:
{comp_list}

Mate connections (edges in the assembly graph):
{edge_list}

Please suggest:
1. An optimal assembly sequence (which component to fix first, then attach next, etc.)
2. Any sub-assemblies that should be built independently.
3. Potential accessibility or interference issues to watch for.
Keep the answer practical and concise.
"""
        return self._ask(prompt)

    def answer(self, question: str) -> str:
        """Free-form Q&A using the full skills context."""
        return self._ask(question)

    def status(self) -> dict:
        """Return configuration status for display in UI."""
        return {
            "profile":    self.profile,
            "model":      self.model_name,
            "online":     self._model is not None,
            "agent_name": self._agent_name,
        }


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    agent = AssemblySkillsAgent()
    print("Agent status:", agent.status())
    print("\n── explain_prediction demo ──")
    print(agent.explain_prediction(
        missing=[((2, 5), 0.87), ((0, 3), 0.72)],
        recs=[("bearing", 0.83), ("shaft", 0.71), ("fastener", 0.65)],
        context="Lathe tailstock partial assembly with 4 known components.",
    ))
