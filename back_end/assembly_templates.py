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
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from dataset import COMP_TYPES  # single source of truth — see dataset.py

TYPE_LABELS = {
    "long_shaft":  "long shaft / spindle",
    "short_shaft": "short shaft / pin",
    "thick_plate": "thick plate  (bracket / flange)",
    "thin_plate":  "thin plate  (leaf / gasket)",
    "bolt":        "bolt / screw",
    "washer":      "washer",
    "nut":         "nut",
    "body":        "structural body",
}

# Human-readable labels for source folder categories
_CATEGORY_LABELS: Dict[str, str] = {
    "hinge_assembly":        "Hinge Assembly",
    "shaft_bearing_housing": "Shaft + Bearing + Housing Assembly",
    "bracket_bolt":          "Bracket + Bolt Assembly",
    "plate_bolt":            "Plate + Bolt Assembly",
    "assembly_files":        "Mechanical Assembly",
}


_CONTAINER_DIRS = {
    "source_3d_models", "source_3dmodels", "3dmodels",
    "best_models_for_training",
}


def _category_from_path(source_path: str) -> str:
    """
    Derive assembly category from the folder immediately under the corpus
    root.  Walks past known container directories (Source_3d_models/,
    Best_models_for_training/) so it lands on the real category folder
    regardless of nesting depth — e.g.
    .../Source_3d_models/Best_models_for_training/Bench_vice/Bench_vice_01/...
    → "bench_vice", not "best_models_for_training".
    Normalises spaces/hyphens to underscores.
    """
    p = Path(source_path)
    parts_norm = [part.lower().replace(" ", "_").replace("-", "_") for part in p.parts]
    for i, norm in enumerate(parts_norm):
        if norm in _CONTAINER_DIRS:
            j = i + 1
            while j < len(parts_norm) - 1 and parts_norm[j] in _CONTAINER_DIRS:
                j += 1
            if j < len(p.parts) - 1:
                return parts_norm[j]
    # Fallback: immediate parent
    return p.parent.name.lower().replace(" ", "_").replace("-", "_")


def _friendly_label(category: str) -> str:
    return _CATEGORY_LABELS.get(
        category,
        category.replace("_", " ").title(),
    )


# Spelling variants worth checking alongside the literal category name when
# matching it against a filename/part name (see AssemblyTemplateDB.match's
# name_hints) -- "vice" is the corpus's own spelling, "vise" is the common US
# alternate a real upload's filename might use instead.
_SPELLING_ALIASES = {"vice": "vise"}


def _normalize_hint_text(text: str) -> str:
    """Lowercase + collapse separators, so 'Tool_post_No_bolts.step' and
    'Tool-Post' both compare equal to the category phrase 'tool post'."""
    for ch in "_-.":
        text = text.replace(ch, " ")
    return " ".join(text.lower().split())


def _category_phrase_variants(category: str) -> List[str]:
    """The category's own name as a normalized phrase, plus any spelling
    alias variant (e.g. 'pipe vice' -> also 'pipe vise')."""
    base = category.replace("_", " ").lower()
    variants = {base}
    for a, b in _SPELLING_ALIASES.items():
        if a in base:
            variants.add(base.replace(a, b))
    return list(variants)


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
        name_hints: Optional[List[str]] = None,
        min_confidence: float = 0.10,
        min_parts_for_type_signal: int = 5,
    ) -> Tuple[Optional[Dict], float]:
        """
        Find the best-matching template for a partial assembly described by a
        list of inferred component-type strings.

        Scoring: cosine similarity between the present-type count vector and
        each template's expected-type count vector (across COMP_TYPES), +25%
        bonus when every present type fits inside the template (no unexpected
        components). Cosine compares *proportions*, not absolute counts, so
        it doesn't care how large a category's template is in total —
        essential here since uploads are frequently partial (missing
        components is the whole point of this feature) and templates vary
        ~4x in total expected body count (C_Clamps ≈10 vs Gate_Valve ≈58).

        Previously used overlap / max(total_present, total_expected): for
        any partial upload, total_present < total_expected for every
        template, so that denominator reduced to total_expected — which
        systematically favoured small-total templates (e.g. C_Clamps,
        Pipe_Vice) over large-total ones (e.g. Tool_Post) even when the
        large template's raw overlap was actually higher. Confirmed this
        reproduced a real user-reported bug: a partial Tool_Post upload
        matched to "Pipe Vice" (score 0.538) over the correct "Tool Post"
        (score 0.430) despite Tool Post having more overlapping components
        (11 vs 7) — fixed 2026-08-17.

        name_hints: optional list of strings (real part names, the uploaded
        filename) — checked for the category's own name appearing as a
        phrase (e.g. "tool post", "gate valve", underscore/hyphen-insensitive,
        with vice/vise as an alias pair) as a secondary signal. A hit adds a
        large flat bonus, decisive enough to override a weak/ambiguous
        geometric signal — real filenames and part names are often literally
        named after the assembly type, which is a far more direct and
        reliable signal than component-type composition alone, especially
        for the sparse-upload case min_parts_for_type_signal exists for.

        min_parts_for_type_signal: below this many detected components, the
        type-composition signal alone is too sparse to trust for a confident
        classification — returns (None, 0.0) unless the winning template also
        has a name_hints hit. Added 2026-08-17 after a real 4-body upload
        (Test_3D_models/Tool_post_No_bolts.step — Tool Holder/washer/plate/nut,
        no shafts or bolts survived) confidently (cosine bonus capped at 1.0)
        matched "Crane Hook" instead of "Tool Post": Crane Hook's template
        happens to be heavily washer/nut-dominated, which coincidentally
        aligned in *direction* with this sparse 4-part vector even though the
        absolute signal was far too thin to discriminate reliably. Rather than
        chase ever-more-specific geometric heuristics for one file, this makes
        the system honestly represent uncertainty when there isn't enough
        signal — unless the filename/part-names hint rescues it directly.

        Returns (template_dict, confidence ∈ [0,1]) or (None, 0.0).
        """
        if not self.templates or not present_types:
            return None, 0.0

        present_counts = Counter(present_types)
        present_vec = [present_counts.get(t, 0) for t in COMP_TYPES]
        present_norm = math.sqrt(sum(v * v for v in present_vec))
        hint_text = _normalize_hint_text(" ".join(name_hints)) if name_hints else ""

        best_tmpl, best_score, best_had_hint = None, 0.0, False

        for tmpl in self.templates:
            expected = tmpl["component_counts"]
            if not expected:
                continue

            expected_vec = [expected.get(t, 0) for t in COMP_TYPES]
            expected_norm = math.sqrt(sum(v * v for v in expected_vec))
            if present_norm == 0 or expected_norm == 0:
                continue
            dot = sum(p * e for p, e in zip(present_vec, expected_vec))
            score = dot / (present_norm * expected_norm)

            # Bonus: all present components are expected in this assembly type.
            # Not clamped to 1.0 here -- two templates can both legitimately
            # earn the bonus (both cosine >= 0.8, both extra == 0), and
            # clamping during comparison collapsed their real difference into
            # an artificial tie decided by iteration order (alphabetical by
            # category) rather than which is the closer match. Only the final
            # returned confidence is clamped, below.
            extra = sum(
                max(0, present_counts.get(t, 0) - expected.get(t, 0))
                for t in COMP_TYPES
            )
            if extra == 0 and dot > 0:
                score = score * 1.25

            has_hint = bool(hint_text) and any(
                phrase in hint_text
                for phrase in _category_phrase_variants(tmpl["category"])
            )
            if has_hint:
                score += 1.0   # decisive -- see name_hints docstring above

            if score > best_score:
                best_score    = score
                best_tmpl     = tmpl
                best_had_hint = has_hint

        if len(present_types) < min_parts_for_type_signal and not best_had_hint:
            return None, 0.0

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
