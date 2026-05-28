"""
assembly_templates.py — Assembly Completeness Model

Learns per-category component-type distributions from Source_3d_models/ and
uses them to:
  1. Identify which assembly type a partial (or single-component) upload belongs to
  2. Report what components are potentially missing based on the expected template

The model is purely geometry-driven — no neural features required.
It loads type histograms from the processed training dataset (data.pt + sources.json)
and groups them by top-level source folder (Hinge_assembly, Shaft_Bearing_Housing …).

Workflow
--------
  build()      ← called once from train.py after training, results cached
  load()       ← called at inference time; returns False if cache missing
  match()      ← given a list of component-type strings, returns best template + confidence
  get_missing() ← diff present vs template to list missing components
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

COMP_TYPES = [
    "body", "fastener", "bearing", "shaft",
    "plate", "housing", "gear", "other",
]

TYPE_LABELS = {
    "body":     "structural body",
    "fastener": "fastener  (bolt / screw / nut)",
    "bearing":  "bearing",
    "shaft":    "shaft / spindle / pin",
    "plate":    "plate  (hinge leaf / bracket / flange)",
    "housing":  "housing / block",
    "gear":     "gear",
    "other":    "component",
}

# Human-readable labels for source folder categories
_CATEGORY_LABELS: Dict[str, str] = {
    "hinge_assembly":        "Hinge Assembly",
    "shaft_bearing_housing": "Shaft + Bearing + Housing Assembly",
    "bracket_bolt":          "Bracket + Bolt Assembly",
    "plate_bolt":            "Plate + Bolt Assembly",
    "assembly_files":        "Mechanical Assembly",
}


def _category_from_path(source_path: str) -> str:
    """
    Derive assembly category from the top-level subfolder name under
    Source_3d_models/.  Normalises spaces/hyphens to underscores.
    """
    p = Path(source_path)
    # Walk up until we find the folder directly inside Source_3d_models
    for i, part in enumerate(p.parts):
        if part.lower().replace(" ", "_") in (
            "source_3d_models", "source_3dmodels", "3dmodels",
        ):
            if i + 1 < len(p.parts) - 1:
                return p.parts[i + 1].lower().replace(" ", "_").replace("-", "_")
    # Fallback: immediate parent
    return p.parent.name.lower().replace(" ", "_").replace("-", "_")


def _friendly_label(category: str) -> str:
    return _CATEGORY_LABELS.get(
        category,
        category.replace("_", " ").title(),
    )


class AssemblyTemplateDB:
    """
    Database of per-category component-type distribution templates.

    Each template records the median count of each component type
    (plate, shaft, fastener, etc.) across all training assemblies in that
    category.  At inference time, an uploaded partial assembly is matched
    against templates and missing components are reported.

    Usage
    -----
    db = AssemblyTemplateDB("back_end/data/assembly_templates.json")
    if not db.load():
        db.build("back_end/data/processed")
        db.save()

    template, conf = db.match(["plate", "shaft"])
    missing = db.get_missing(["plate"], template)
    """

    def __init__(self, cache_path: str):
        self.cache_path = Path(cache_path)
        self.templates: List[Dict] = []

    # ── Build ──────────────────────────────────────────────────────────────────

    def build(self, processed_dir: str) -> None:
        """
        Load data.pt + sources.json from the processed dataset directory,
        group graphs by assembly category, and compute median component-type
        distributions per category.
        """
        proc      = Path(processed_dir) / "processed"
        data_pt   = proc / "data.pt"
        src_json  = proc / "sources.json"

        if not data_pt.exists():
            raise FileNotFoundError(f"data.pt not found at {data_pt}. Run training first.")
        if not src_json.exists():
            raise FileNotFoundError(
                f"sources.json not found at {src_json}. "
                "Re-run training with --force-reload to regenerate the processed dataset."
            )

        data_bundle, slices = torch.load(data_pt, weights_only=False)
        source_paths: List[str] = json.loads(src_json.read_text())

        x_slices = slices["x"].tolist()      # length = n_graphs + 1
        n_graphs  = len(source_paths)

        category_entries: Dict[str, List[Dict]] = {}

        for i in range(n_graphs):
            start = x_slices[i]
            end   = x_slices[i + 1]
            x_i   = data_bundle.x[start:end]  # [n_nodes, 16]

            type_indices = x_i[:, :8].argmax(dim=1).tolist()
            type_counts  = Counter(COMP_TYPES[j] for j in type_indices)
            category     = _category_from_path(source_paths[i])

            category_entries.setdefault(category, []).append({
                "file":         Path(source_paths[i]).name,
                "type_counts":  dict(type_counts),
                "total_bodies": end - start,
            })

        self.templates = []
        for cat in sorted(category_entries):
            entries = category_entries[cat]

            # For each type, median count across assemblies that contain it
            merged: Dict[str, int] = {}
            for t in COMP_TYPES:
                vals    = [e["type_counts"].get(t, 0) for e in entries]
                nonzero = [v for v in vals if v > 0]
                if nonzero:
                    merged[t] = round(statistics.median(nonzero))

            merged = {k: v for k, v in merged.items() if v > 0}

            self.templates.append({
                "category":         cat,
                "label":            _friendly_label(cat),
                "component_counts": merged,
                "total_bodies":     round(
                    statistics.median([e["total_bodies"] for e in entries])
                ),
                "examples":         [e["file"] for e in entries[:5]],
                "n_assemblies":     len(entries),
            })

        print(
            f"  [TemplateDB] Built {len(self.templates)} category templates "
            f"from {n_graphs} training assemblies."
        )
        for t in self.templates:
            print(f"    {t['label']:40s}  bodies≈{t['total_bodies']}  "
                  f"dist={t['component_counts']}")

    # ── Match ──────────────────────────────────────────────────────────────────

    def match(
        self,
        present_types: List[str],
        min_confidence: float = 0.10,
    ) -> Tuple[Optional[Dict], float]:
        """
        Find the best-matching template for a partial assembly described by a
        list of inferred component-type strings.

        Scoring:
          overlap / max(total_present, total_expected)
          + 25 % bonus when every present type fits inside the template
            (no unexpected components)

        Returns (template_dict, confidence ∈ [0,1]) or (None, 0.0).
        """
        if not self.templates or not present_types:
            return None, 0.0

        present_counts = Counter(present_types)
        best_tmpl, best_score = None, 0.0

        for tmpl in self.templates:
            expected = tmpl["component_counts"]
            if not expected:
                continue

            overlap = sum(
                min(present_counts.get(t, 0), expected.get(t, 0))
                for t in COMP_TYPES
            )
            union = max(sum(present_counts.values()), sum(expected.values()))
            score = overlap / union if union > 0 else 0.0

            # Bonus: all present components are expected in this assembly type
            extra = sum(
                max(0, present_counts.get(t, 0) - expected.get(t, 0))
                for t in COMP_TYPES
            )
            if extra == 0 and overlap > 0:
                score = min(score * 1.25, 1.0)

            if score > best_score:
                best_score = score
                best_tmpl  = tmpl

        if best_score < min_confidence:
            return None, 0.0

        return best_tmpl, round(min(best_score, 1.0), 3)

    # ── Missing components ────────────────────────────────────────────────────

    def get_missing(
        self,
        present_types: List[str],
        template: Dict,
    ) -> List[Dict]:
        """
        Compare present component types against the template's expected counts
        and return a list of missing component descriptors:
            [{"type": "shaft", "count": 1, "label": "shaft / spindle / pin"}, ...]
        """
        present_counts = Counter(present_types)
        missing = []
        for t, needed in template["component_counts"].items():
            have = present_counts.get(t, 0)
            if have < needed:
                missing.append({
                    "type":  t,
                    "count": needed - have,
                    "label": TYPE_LABELS.get(t, t),
                })
        return missing

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.templates, indent=2))
        print(f"  [TemplateDB] Saved {len(self.templates)} templates → {self.cache_path}")

    def load(self) -> bool:
        if self.cache_path.exists():
            self.templates = json.loads(self.cache_path.read_text())
            return True
        return False
