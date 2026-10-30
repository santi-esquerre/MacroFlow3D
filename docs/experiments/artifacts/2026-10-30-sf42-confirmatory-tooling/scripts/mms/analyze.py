#!/usr/bin/env python3
"""SF-42 N2: tables, observed orders and the acceptance-D verdict of the MMS study.

    python3 analyze.py <out_dir> [--min-grids 4] [--threshold 1.8] [--eps 0.5 1 1.5 2]

Reads the mms_ny*_eps*_rtol*.json of mms_dirichlet, writes <out_dir>/mms_summary.md and mms_summary.json and
prints the markdown. Observed order of a pair: p = log(e_coarse/e_fine) / log(ny_fine/ny_coarse) (= log2 for
ratio-2 ladders).

Verdict D (spec acceptance D): at every eps, the observed order of each GATED metric is >= threshold over each of
the two finest grid pairs. Gated metrics: head L2 and head max (all cells), interior face flux L2, interior face
flux max restricted to faces at least two cells from the Dirichlet faces. Reported, not gated: Dirichlet faces and
first interior layer (order near 1 expected). Verdict INCOMPLETE if fewer than min_grids grids or an expected eps is
missing or any run did not converge; the algebraic-error control (rtol 1e-12 vs 1e-10 at the finest grid) is
reported with its largest relative change of any reported error.
"""
import glob
import json
import math
import os
import sys

# (key in the JSON, norm key, label, gated)
METRICS = [
    ("head", "l2_abs", "head L2", True),
    ("head", "max_abs", "head max", True),
    ("flux_interior", "l2_abs", "interior flux L2", True),
    ("flux_far", "max_abs", "flux max (>=2 cells from Dirichlet faces)", True),
    ("flux_interior", "max_abs", "interior flux max (all interior faces)", False),
    ("flux_dirichlet", "l2_abs", "Dirichlet faces L2", False),
    ("flux_dirichlet", "max_abs", "Dirichlet faces max", False),
    ("flux_first_layer", "l2_abs", "first interior layer L2", False),
    ("flux_first_layer", "max_abs", "first interior layer max", False),
]
REL_CONTROL_TOL = 1e-2  # algebraic control: relative change of any error between rtol 1e-12 and 1e-10


def observed_order(e_coarse, e_fine, ny_coarse, ny_fine):
    """log(e_c/e_f)/log(ny_f/ny_c); nan for a non-positive or non-finite error."""
    if not (e_coarse > 0 and e_fine > 0 and math.isfinite(e_coarse) and math.isfinite(e_fine)):
        return float("nan")
    return math.log(e_coarse / e_fine) / math.log(ny_fine / ny_coarse)


def orders(series):
    """series: list of (ny, error) sorted by ny -> list of (ny_c, ny_f, p)."""
    return [(a[0], b[0], observed_order(a[1], b[1], a[0], b[0])) for a, b in zip(series, series[1:])]


def gate_metric(series, threshold):
    """Pass iff the two finest pairs both have p >= threshold. Needs >= 3 grids."""
    ps = orders(series)
    if len(ps) < 2:
        return {"orders": [p for _, _, p in ps], "pass": None}
    last = [p for _, _, p in ps[-2:]]
    return {"orders": [p for _, _, p in ps], "last_two": last, "pass": all(p >= threshold for p in last)}


def load(out_dir):
    runs = []
    for f in sorted(glob.glob(os.path.join(out_dir, "mms_ny*_eps*_rtol*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        if d.get("schema") == "sf42-mms-dirichlet-1":
            runs.append(d)
    return runs


def verdict(runs, expected_eps=(0.5, 1.0, 1.5, 2.0), min_grids=4, threshold=1.8, main_rtol=1e-10):
    main = [r for r in runs if math.isclose(r["pcg_rtol"], main_rtol, rel_tol=1e-9)]
    out = {"threshold": threshold, "min_grids": min_grids, "per_eps": {}, "problems": []}
    all_pass = True
    incomplete = False
    for eps in expected_eps:
        rs = sorted((r for r in main if math.isclose(r["eps"], eps, rel_tol=1e-12)), key=lambda r: r["ny"])
        entry = {"grids": [r["ny"] for r in rs], "metrics": {}}
        out["per_eps"][repr(float(eps))] = entry
        if len(rs) < max(min_grids, 3):
            incomplete = True
            out["problems"].append("eps=%g: %d grids (< %d)" % (eps, len(rs), max(min_grids, 3)))
        for r in rs:
            if not r["pcg_converged"]:
                incomplete = True
                out["problems"].append("eps=%g ny=%d: PCG not converged" % (eps, r["ny"]))
        for key, nk, label, gated in METRICS:
            series = [(r["ny"], r[key][nk]) for r in rs]
            g = gate_metric(series, threshold)
            g["gated"] = gated
            g["errors"] = [e for _, e in series]
            entry["metrics"][label] = g
            if gated and g["pass"] is not True:
                if g["pass"] is None:
                    incomplete = True
                else:
                    all_pass = False
    out["verdict"] = "INCOMPLETE" if incomplete else ("PASS" if all_pass else "FAIL")
    out["algebraic_control"] = algebraic_control(runs, main_rtol)
    return out


def algebraic_control(runs, main_rtol=1e-10):
    """Finest grid, per eps: largest relative change of any reported error between the tighter and the main rtol."""
    res = {}
    main = [r for r in runs if math.isclose(r["pcg_rtol"], main_rtol, rel_tol=1e-9)]
    tight = [r for r in runs if r["pcg_rtol"] < main_rtol * 0.5]
    for t in tight:
        m = [r for r in main if r["ny"] == t["ny"] and math.isclose(r["eps"], t["eps"], rel_tol=1e-12)]
        if not m:
            continue
        m = m[0]
        worst, where = 0.0, None
        for key, nk, label, _ in METRICS:
            a, b = m[key][nk], t[key][nk]
            rel = abs(a - b) / max(abs(a), 1e-300)
            if rel >= worst:
                worst, where = rel, label
        res["ny=%d eps=%g rtol=%g" % (t["ny"], t["eps"], t["pcg_rtol"])] = {
            "max_rel_change": worst, "metric": where, "negligible": worst < REL_CONTROL_TOL}
    return res


def fmt(x):
    return "%.3e" % x


def markdown(runs, v):
    main = [r for r in runs if math.isclose(r["pcg_rtol"], 1e-10, rel_tol=1e-9)]
    L = ["# MMS order study of the Dirichlet head solve (SF-42 N2)", "",
         "Verdict D: **%s** (threshold %.2f over the two finest grid pairs; gated: head L2 and max, interior flux L2, "
         "flux max at least 2 cells from the Dirichlet faces)." % (v["verdict"], v["threshold"]), ""]
    if v["problems"]:
        L += ["Problems: " + "; ".join(v["problems"]), ""]
    for eps in sorted({r["eps"] for r in main}):
        rs = sorted((r for r in main if r["eps"] == eps), key=lambda r: r["ny"])
        L += ["## eps = %g" % eps, "",
              "| ny | nx | mg | PCG it | conv | rel res | " + " | ".join(m[2] for m in METRICS) + " |",
              "|" + "---|" * (6 + len(METRICS))]
        for r in rs:
            L.append("| %d | %d | %d | %d | %s | %s | %s |" % (
                r["ny"], r["nx"], r["mg_levels"], r["pcg_iterations"], r["pcg_converged"], fmt(r["residual_rel"]),
                " | ".join(fmt(r[k][nk]) for k, nk, _, _ in METRICS)))
        L += ["", "Observed orders (consecutive pairs):", "",
              "| metric | gated | " + " | ".join("%d-%d" % (a["ny"], b["ny"]) for a, b in zip(rs, rs[1:])) + " | rel. error (finest) |",
              "|" + "---|" * (3 + max(len(rs) - 1, 0))]
        for key, nk, label, gated in METRICS:
            ps = orders([(r["ny"], r[key][nk]) for r in rs])
            rel = rs[-1][key][nk.replace("_abs", "_rel")]
            L.append("| %s | %s | %s | %s |" % (label, "yes" if gated else "no", " | ".join("%.2f" % p for _, _, p in ps), fmt(rel)))
        L.append("")
    ac = v["algebraic_control"]
    L += ["## Algebraic-error control (rtol 1e-12 against 1e-10, finest grid)", ""]
    if not ac:
        L.append("No rtol 1e-12 run found.")
    for k, d in ac.items():
        L.append("- %s: max relative change %.2e (%s), %s" % (k, d["max_rel_change"], d["metric"],
                                                              "negligible" if d["negligible"] else "NOT negligible"))
    return "\n".join(L) + "\n"


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--min-grids", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=1.8)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0])
    a = ap.parse_args(argv)
    runs = load(a.out_dir)
    if not runs:
        print("no mms JSON in %s" % a.out_dir, file=sys.stderr)
        return 2
    v = verdict(runs, a.eps, a.min_grids, a.threshold)
    md = markdown(runs, v)
    with open(os.path.join(a.out_dir, "mms_summary.md"), "w") as f:
        f.write(md)
    with open(os.path.join(a.out_dir, "mms_summary.json"), "w") as f:
        json.dump(v, f, indent=1, sort_keys=True)
    print(md)
    return 0 if v["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
