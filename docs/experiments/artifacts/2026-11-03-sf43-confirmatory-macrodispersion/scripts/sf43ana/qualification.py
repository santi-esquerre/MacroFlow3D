"""Qualification gate of the robust sign at the REALIZED sizes (decision H4 with H13 floors against H15 coverage).

The coverage harness ``sf43cov`` (profile ``sf43``) qualifies, with the G5.3 criterion, (i) the bootstrap-t sampling limit of the production
mean on a grid of sizes ``n`` and (ii) the whole robust-sign pipeline ``RL(a_j) > 0`` at several PROFILES of arm sizes per structure ``K``
(``K`` = number of budget components: 3 at sigma^2 0.25 / 1, 4 at 2.25 / 4).  Its ``qualification.json`` is the document the
pre-registration names (key ``qualification``, hashed like the coverage summary).

A robust sign at an amplitude is TESTABLE only if

  (a) its H13 floors hold (``gate4.design_from_arms``: this is not repeated here);
  (b) the pipeline of its structure ``K`` is qualified at the LARGEST profile whose every arm size (production, ladder, width cells,
      ``h_max`` passes) is at most the REALIZED size of that arm (the sizes of the design arrays the test actually uses); and
  (c) the sampling limit is qualified at the largest grid size ``n`` at most the realized production ``N``.

Otherwise its p-value is 1 (the hypothesis stays in the fixed Holm family) with status ``not qualified at the realized sizes``; without a
qualification document every robust sign has ``p = 1`` and status ``no qualification document`` (the smoke run may have none).

ASSUMPTION (stated, not tested): coverage qualified at a smaller size carries over to larger sizes (the coverage error of the studentized
procedures decreases with the sample sizes).  The rule uses the largest eligible profile only; a smaller qualified profile does not rescue a
larger eligible profile that is not qualified.  A profile or grid size is qualified only if EVERY simulated cell of it passed (all
distributions, bias scenarios, tails); an incomplete document (cells missing) qualifies nothing.
"""
import hashlib
import json
import os

NO_DOC = "no qualification document"
NOT_QUALIFIED = "not qualified at the realized sizes"
ARM_KEYS = ("production", "ladder", "width", "h_max")


def load(path):
    """-> (document dict | None, info dict ``{path, found, sha256, plan_hash, all_cells_complete}``).  ``path`` None / missing file: no document."""
    info = {"path": path, "found": False, "sha256": None, "plan_hash": None, "all_cells_complete": None}
    if not path or not os.path.isfile(path):
        return None, info
    with open(path, "rb") as f:
        raw = f.read()
    doc = json.loads(raw.decode())
    info.update(found=True, sha256=hashlib.sha256(raw).hexdigest(), plan_hash=doc.get("plan_hash"), all_cells_complete=doc.get("all_cells_complete"))
    return doc, info


def _all(flags):
    if not flags:
        return None
    if any(f is False for f in flags):
        return False
    return True if all(f is True for f in flags) else None


def pipeline_profiles(doc):
    """Profiles of the document's ``pipeline`` cells: ``[{K, arm_sizes {production, ladder, width, h_max}, qualified (True | False | None), cells}]``,
    one per distinct (K, sizes), ``qualified`` = every cell of the profile qualified."""
    groups = {}
    for c in doc.get("pipeline", []):
        a = c["arm_sizes"]
        sizes = {"production": int(c["N"]), "ladder": int(a["ladder"]), "width": int(a["width"]), "h_max": int(a["h_max"])}
        groups.setdefault((int(c["K"]),) + tuple(sizes[k] for k in ARM_KEYS), {"K": int(c["K"]), "arm_sizes": sizes, "flags": []})["flags"].append(c.get("qualified"))
    return [{"K": g["K"], "arm_sizes": g["arm_sizes"], "qualified": _all(g["flags"]), "cells": len(g["flags"])} for _, g in sorted(groups.items())]


def sampling_grid(doc):
    """``{n: qualified}`` of the bootstrap-t sampling limit (side ``lower``, every distribution of the grid point)."""
    out = {}
    for key, byd in doc.get("sampling", {}).items():
        out[int(key.split("=")[1])] = _all([e["lower"]["qualified"] for e in byd.values()])
    return out


def realized_sizes(design):
    """Realized arm sizes of a ``gate4.Design``: production N, the paired ladder, the smallest width cell, the ``h_max`` pairs (None for a
    component the structure does not have: it constrains no profile)."""
    width = [a.size for a in (design.wide_fine, design.prod_fine, design.wide_coarse, design.prod_coarse) if a is not None]
    return {"production": int(design.prod.size) if design.prod is not None else 0,
            "ladder": int(design.levels.shape[0]) if design.levels is not None else None,
            "width": int(min(width)) if width else None,
            "h_max": int(design.h16.size) if design.h16 is not None else None}


def check(doc, K, sizes):
    """-> (status | None, detail).  ``status`` None: testable by the rule; else ``NO_DOC`` / ``NOT_QUALIFIED``.  ``detail`` documents the profile
    and the grid size used."""
    detail = {"K": K, "realized_sizes": dict(sizes)}
    if doc is None:
        return NO_DOC, detail
    profiles = [p for p in pipeline_profiles(doc) if p["K"] == K and all(sizes[k] is None or p["arm_sizes"][k] <= sizes[k] for k in ARM_KEYS)]
    chosen = max(profiles, key=lambda p: (sum(p["arm_sizes"].values()), tuple(p["arm_sizes"][k] for k in ARM_KEYS)), default=None)
    grid = sampling_grid(doc)
    g = max((n for n in grid if n <= sizes["production"]), default=None)
    complete = doc.get("all_cells_complete") is True
    detail.update(pipeline_profile=chosen, sampling_grid_n=g, sampling_qualified=None if g is None else grid[g], document_complete=complete)
    ok = complete and chosen is not None and chosen["qualified"] is True and g is not None and grid[g] is True
    reasons = []
    if not complete:
        reasons.append("the qualification document is incomplete (all_cells_complete is not true)")
    if chosen is None:
        reasons.append("no pipeline profile of structure K = %d has every arm size at most the realized size" % K)
    elif chosen["qualified"] is not True:
        reasons.append("the largest eligible pipeline profile %s is not qualified" % (chosen["arm_sizes"],))
    if g is None:
        reasons.append("no sampling grid size at most the realized production N = %d" % sizes["production"])
    elif grid[g] is not True:
        reasons.append("the bootstrap-t sampling limit is not qualified at n = %d" % g)
    detail["reasons"] = reasons
    return (None if ok else NOT_QUALIFIED), detail
