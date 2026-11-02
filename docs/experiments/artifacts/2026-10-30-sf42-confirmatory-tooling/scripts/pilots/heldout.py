"""Held-out check of the committed cost model on the J1B grid pilots (R6): predicted against measured seconds per realization.

    python3 -m pilots.heldout --data <R/pilots> --repo <repo root> --out <R/pilots/j1b_heldout.json>

The J1B points are NOT added to the fit (``cost_projection.json`` is unchanged).  Two models predict them: the SF-39-only fit and the
refit with the J1 / J2 / J3 / J3B pilot points of R4 (``pilots.cost.points_from_runs``, the model the committed projection uses).
``rel`` = (model - observed) / observed, the sign convention of ``cost_projection.json``; the observed time is the code's own
per-realization time (``cost.realization_seconds``).
"""
import argparse
import json

from mcstat import costmodel as cm

from . import cost


def heldout(data, repo):
    old, _ = cm.fit_sf39(repo)
    new = old.with_points(cost.points_from_runs(data))      # J1, J2, J3, J3B: the committed refit
    rows = []
    for p in cost.points_from_runs(data, jobs=("J1B",)):
        a = p.arm
        rows.append({"label": a.label, "code": a.code, "cells": a.cells, "sigma2": a.sigma2, "observed_s": p.gpu_s_per_real,
                     "sf39_model_s": old.predict(a), "rel_sf39_model": (old.predict(a) - p.gpu_s_per_real) / p.gpu_s_per_real,
                     "in_sf39_range": bool(old.in_range(a)), "refit_model_s": new.predict(a),
                     "rel_refit_model": (new.predict(a) - p.gpu_s_per_real) / p.gpu_s_per_real, "in_refit_range": bool(new.in_range(a))})
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pilots.heldout")
    ap.add_argument("--data", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    rows = heldout(a.data, a.repo)
    with open(a.out, "w") as f:
        json.dump({"note": "J1B points held out of the fit; rel = (model - observed) / observed", "rows": rows}, f, indent=1)
    for r in rows:
        print("%-28s observed %7.2f s  SF-39-only %7.2f (%+.3f)  refit %7.2f (%+.3f)" % (
            r["label"], r["observed_s"], r["sf39_model_s"], r["rel_sf39_model"], r["refit_model_s"], r["rel_refit_model"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
