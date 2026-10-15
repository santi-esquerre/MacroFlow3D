#!/usr/bin/env python3
"""SF-35 cross-construction digest (criterion D): elliptic SF-33 solution vs the traced labels.

  xcons_digest.py raw/xcons/eps<E>_N<N>.json ... [--sf33 <SF-33 artifact dir>] [--out analysis/xcons.md]

Inputs: driver JSON of `inlet_slab --construct trace --solve` (metrics.i1o4 = metrics of the Newton-Krylov solution
against the traced labels as psi_or: e_psi, e_psi1, e_psi2, a_psi1, a_psi2; r_F of the solve from
continuation.stages[-1].newton.r_F; xcons = per-plane RMS differences) and the SF-33 full-precision references
(orchestrator reading rule 1: the 1e-6 criterion is evaluated against the FULL-PRECISION values, not the printed digits):
  (0.25, 32)  raw/ladder_0.25_c4/N32.json   (0.25, 64)  raw/ladder_0.25_c4/N64.json   (0.5, 32)  raw/prod32_05/N32.json
each at metrics.i1o4.{e_psi, e_psi1, e_psi2, a_psi1, a_psi2}.

Verdict D (spec): the --solve runs converge (r_F <= 1e-10) and e_psi / e_psi1 / e_psi2 reproduce SF-33 within 1e-6
relative (|new - ref| / |ref|); a_psi1, a_psi2 observed order 32 -> 64 at eps 0.25 >= 2.5 (SF-33: 3.07 / 2.87).
A point without a reference (e.g. a local smoke at N = 16) is listed, never judged.
"""
import argparse
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import fnum, get, load_json, md_table, n_of, num, order, write_out  # noqa: E402

REFS = {
    (0.25, 32): "raw/ladder_0.25_c4/N32.json",
    (0.25, 64): "raw/ladder_0.25_c4/N64.json",
    (0.5, 32): "raw/prod32_05/N32.json",
}
KEYS = ("e_psi", "e_psi1", "e_psi2", "a_psi1", "a_psi2")
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SF33 = os.path.join(os.path.dirname(os.path.dirname(HERE)), "2026-10-06-sf33-gpu-inlet-labels")


def load_ref(sf33, key):
    p = os.path.join(sf33, REFS[key])
    try:
        js = load_json(p)
    except (OSError, ValueError):
        return None, p
    m = get(js, "metrics", "i1o4", default={}) or {}
    return {k: num(m.get(k)) for k in KEYS}, p


def solve_rF(js):
    stages = get(js, "continuation", "stages", default=[]) or []
    if stages:
        return num(get(stages[-1], "newton", "r_F"))
    return float("nan")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="driver JSON of the --solve runs (eps<E>_N<N>.json)")
    ap.add_argument("--sf33", default=DEFAULT_SF33, help="SF-33 artifact directory (default: sibling artifact)")
    ap.add_argument("--out", help="markdown output path (also printed)")
    a = ap.parse_args(argv)

    runs = {}
    for f in a.files:
        js = load_json(f)
        eps = num(js.get("eps"))
        if math.isnan(eps):
            m = re.search(r"eps([0-9.]+)_N", os.path.basename(f))
            eps = float(m.group(1)) if m else float("nan")
        runs[(round(eps, 12), n_of(f, js))] = (f, js)

    L = ["# SF-35 cross-construction (D): elliptic (SF-33 Newton-Krylov) vs traced labels", ""]
    L.append("Verdict D: every `--solve` run converges (`r_F <= 1e-10`) and `e_psi`, `e_psi1`, `e_psi2` reproduce the "
             "SF-33 full-precision values within 1e-6 relative; `a_psi1`, `a_psi2` observed order 32 -> 64 at eps 0.25 "
             ">= 2.5 (SF-33: 3.07 / 2.87).")
    L.append("")
    L.append("References loaded (SF-33 full precision, `metrics.i1o4`):")
    L.append("")
    refs = {}
    for key in sorted(REFS):
        r, p = load_ref(a.sf33, key)
        refs[key] = r
        if r is None:
            L.append("- eps=%g N=%d: **%s not readable**" % (key[0], key[1], p))
        else:
            L.append("- eps=%g N=%d (%s): e_psi=%.17g e_psi1=%.17g e_psi2=%.17g a_psi1=%.17g a_psi2=%.17g"
                     % (key[0], key[1], REFS[key], r["e_psi"], r["e_psi1"], r["e_psi2"], r["a_psi1"], r["a_psi2"]))
            print("REFERENCE eps=%g N=%d e_psi=%.17g e_psi1=%.17g e_psi2=%.17g" %
                  (key[0], key[1], r["e_psi"], r["e_psi1"], r["e_psi2"]))
    L.append("")

    L.append("## Per run")
    L.append("")
    rows, states = [], []
    for key in sorted(runs):
        f, js = runs[key]
        m = get(js, "metrics", "i1o4", default=None)
        rf = solve_rF(js)
        sst = js.get("solve_status", "?")
        ref = refs.get(key)
        conv = (sst == "converged") and (not math.isnan(rf)) and rf <= 1e-10
        if m is None:
            rows.append((key[0], key[1], js.get("trace_status", "?"), sst, fnum(rf), "-", "-", "-", "-", "-", "-", "-", "no solve metrics"))
            states.append("not_evaluable")
            continue
        vals = {k: num(m.get(k)) for k in KEYS}
        rel = {}
        if ref is not None:
            for k in ("e_psi", "e_psi1", "e_psi2"):
                rel[k] = abs(vals[k] - ref[k]) / abs(ref[k]) if ref[k] != 0.0 else float("inf")
        if ref is None:
            verdict = "not_evaluable (no SF-33 reference at this (eps, N))"
            states.append("not_evaluable")
        else:
            ok = conv and all((not math.isnan(v)) and v <= 1e-6 for v in rel.values())
            verdict = "pass" if ok else "FAIL"
            states.append("pass" if ok else "fail")
        rows.append((key[0], key[1], js.get("trace_status", "?"), sst, fnum(rf), fnum(vals["e_psi"]), fnum(vals["e_psi1"]), fnum(vals["e_psi2"]),
                     fnum(rel.get("e_psi", float("nan"))), fnum(rel.get("e_psi1", float("nan"))),
                     fnum(rel.get("e_psi2", float("nan"))), "conv" if conv else "NOT CONVERGED", verdict))
    L.append(md_table(["eps", "N", "trace status", "solve status", "r_F (solve)", "e_psi", "e_psi1", "e_psi2",
                       "rel diff e_psi", "rel diff e_psi1", "rel diff e_psi2", "r_F <= 1e-10", "verdict D (value part)"],
                      rows))
    L.append("")

    L.append("## Observed order of a_psi (32 -> 64, eps = 0.25)")
    L.append("")
    k32, k64 = (0.25, 32), (0.25, 64)
    ord_state = "not_evaluable"
    if k32 in runs and k64 in runs:
        m32 = get(runs[k32][1], "metrics", "i1o4", default={}) or {}
        m64 = get(runs[k64][1], "metrics", "i1o4", default={}) or {}
        o1 = order(m32.get("a_psi1"), m64.get("a_psi1"))
        o2 = order(m32.get("a_psi2"), m64.get("a_psi2"))
        r1 = order(refs[k32]["a_psi1"], refs[k64]["a_psi1"]) if refs.get(k32) and refs.get(k64) else float("nan")
        r2 = order(refs[k32]["a_psi2"], refs[k64]["a_psi2"]) if refs.get(k32) and refs.get(k64) else float("nan")
        L.append(md_table(["quantity", "a_psi(32)", "a_psi(64)", "observed order", "SF-33 order (from its JSON)", ">= 2.5"],
                          [("a_psi1", fnum(m32.get("a_psi1")), fnum(m64.get("a_psi1")), fnum(o1, "%.2f"), fnum(r1, "%.2f"),
                            "pass" if (not math.isnan(o1) and o1 >= 2.5) else "FAIL"),
                           ("a_psi2", fnum(m32.get("a_psi2")), fnum(m64.get("a_psi2")), fnum(o2, "%.2f"), fnum(r2, "%.2f"),
                            "pass" if (not math.isnan(o2) and o2 >= 2.5) else "FAIL")]))
        ord_state = "pass" if (not math.isnan(o1) and not math.isnan(o2) and o1 >= 2.5 and o2 >= 2.5) else "fail"
    else:
        L.append("Both (0.25, 32) and (0.25, 64) are needed: not_evaluable.")
    L.append("")

    L.append("## XCONS per plane (RMS over plane j of psi_solve - psi_trace, periodic parts)")
    L.append("")
    for key in sorted(runs):
        xc = get(runs[key][1], "xcons", default=None)
        if not xc:
            continue
        L.append("eps = %g, N = %d" % key)
        L.append("")
        L.append(md_table(["j", "x1", "a_psi1_j", "a_psi2_j"],
                          [(p["j"], fnum(p["x1"], "%.4f"), fnum(p["a_psi1_j"]), fnum(p["a_psi2_j"])) for p in xc]))
        L.append("")

    if any(s == "fail" for s in states) or ord_state == "fail":
        verdict = "FAIL"
    elif states and all(s == "pass" for s in states) and ord_state == "pass":
        verdict = "pass"
    else:
        verdict = "not_evaluable"
    L.append("**Verdict D: %s** (value part: %s; a_psi order part: %s)" % (verdict, ",".join(states) or "-", ord_state))
    L.append("")
    write_out("\n".join(L), a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
