#!/usr/bin/env python3
"""SF-35 positive-control digest (criteria A1-A4).

  controls_digest.py --raw raw/controls --exports exports/controls [--ctest-log logs/controls/ctest.log]
                     [--out analysis/controls.md]

Inputs (written by `run_campaign.sh <build> controls`):
  raw/<part>/{cosx2_N<N>,k1_N<N>,control2d_N<N>}.json   driver --summary JSON of `--construct trace`
  exports/<part>/<name>/psi1.npy, psi2.npy               saved labels, shape (N+1, N, N) = [plane j][m2][m3], x = m/N
  ctest.log                                              `ctest -R 'inlet_slab_(controls|trace16)'` (A4 and the fast tests)

Criteria (spec "Acceptance thresholds"):
  A1  k = 1: labels affine to <= 1e-13 on every plane: max |psi1 - x2|, max |psi2 - x3|  (the feet are not exported;
      affine labels psi1 = x2, psi2 = x3 on every plane is the observable consequence of feet = (x2, x3)).
  A2  k = exp(0.5 cos 2 pi x2): |psi_i(j,m2,m3) - psi_i(0,m2,m3)| <= 1e-10 on every plane; e_v and r_F of these labels
      decrease with observed order >= 3.5 on each pair 16->32 and 32->64.  Pre-registered reading: both pairs are
      evaluated; a 16->32 miss with a 32->64 pass is reported as `pre-asymptotic`, a 32->64 miss is `fail`.
  A3  control2d, eps = 0.5: max |psi2 - x3| <= 1e-10 on every plane.
  A4  synthetic failure field: asserted inside the ctest binary (inlet_slab_trace16 / inlet_slab_controls); the ctest
      outcome and each test's time (spec: <= 60 s) are reported here.
Deterministic; numpy only.
"""
import argparse
import glob
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import fnum, get, load_json, md_table, n_of, num, order, write_out  # noqa: E402


def load_labels(d):
    p1 = np.load(os.path.join(d, "psi1.npy"))
    p2 = np.load(os.path.join(d, "psi2.npy"))
    return p1, p2


def max_abs(a):
    """max |a|, NaN if any NaN (a failed seed must not pass silently)."""
    return float("nan") if np.isnan(a).any() else float(np.max(np.abs(a)))


def affine_defects(p1, p2):
    n = p1.shape[1]
    x2 = (np.arange(n) / n)[None, :, None]
    x3 = (np.arange(n) / n)[None, None, :]
    return max_abs(p1 - x2), max_abs(p2 - x3)


def straightness(p1, p2):
    d1 = max_abs(p1 - p1[0:1])
    d2 = max_abs(p2 - p2[0:1])
    return d1, d2


def ctest_results(path):
    res = []
    try:
        with open(path, errors="replace") as f:
            for line in f:
                m = re.search(r"Test\s+#\d+:\s+(\S+)\s+\.+\s*(\S+(?:\s\S+)?)\s+([0-9.]+)\s+sec", line)
                if m:
                    res.append((m.group(1), m.group(2).strip(), float(m.group(3))))
    except OSError:
        pass
    return res


def by_name(raw, prefix):
    out = {}
    for f in sorted(glob.glob(os.path.join(raw, prefix + "_N*.json"))):
        js = load_json(f)
        out[n_of(f, js)] = (f, js)
    return out


def pair_orders(vals, ns):
    return [(a, b, order(vals[a], vals[b])) for a, b in zip(ns[:-1], ns[1:]) if b == 2 * a]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="raw/<part> directory with the driver JSON")
    ap.add_argument("--exports", required=True, help="exports/<part> directory with the saved labels")
    ap.add_argument("--ctest-log", help="ctest log of the fast tests (A4 + timing)")
    ap.add_argument("--out", help="markdown output path (also printed)")
    a = ap.parse_args(argv)

    L = ["# SF-35 positive controls (A1-A4)", ""]
    overall = []

    # ---- A1 -----------------------------------------------------------------------------
    L += ["## A1: k = 1 (Y = 0 cells, eps = 0.5)", ""]
    k1 = by_name(a.raw, "k1")
    rows = []
    for n, (f, js) in sorted(k1.items()):
        d = os.path.join(a.exports, "k1_N%d" % n)
        if not os.path.isdir(d):
            rows.append((n, js.get("status"), "no labels exported", "-", "not_evaluable"))
            overall.append(("A1", "not_evaluable"))
            continue
        p1, p2 = load_labels(d)
        d1, d2 = affine_defects(p1, p2)
        ok = (not math.isnan(d1 + d2)) and max(d1, d2) <= 1e-13
        rows.append((n, js.get("status"), fnum(d1), fnum(d2), "pass" if ok else "FAIL"))
        overall.append(("A1", "pass" if ok else "fail"))
    L.append(md_table(["N", "status", "max|psi1 - x2|", "max|psi2 - x3|", "A1 (<= 1e-13)"], rows) if rows
             else "No k1 run found: not_evaluable.")
    if not rows:
        overall.append(("A1", "not_evaluable"))
    L.append("")

    # ---- A2 -----------------------------------------------------------------------------
    L += ["## A2: k = exp(0.5 cos 2 pi x2) (Y_cell = cos(2 pi (j + 1/2) h), eps = 0.5)", ""]
    cs = by_name(a.raw, "cosx2")
    rows, ev, rf, ns = [], {}, {}, []
    st2 = []
    for n, (f, js) in sorted(cs.items()):
        d = os.path.join(a.exports, "cosx2_N%d" % n)
        ev[n] = num(get(js, "metrics", "trace", "e_v"))
        rf[n] = num(get(js, "residual", "r_F"))
        if os.path.isdir(d):
            p1, p2 = load_labels(d)
            s1, s2 = straightness(p1, p2)
            ok = (not math.isnan(s1 + s2)) and max(s1, s2) <= 1e-10
            st2.append(ok)
            rows.append((n, js.get("status"), fnum(s1), fnum(s2), "pass" if ok else "FAIL", fnum(ev[n]), fnum(rf[n])))
            ns.append(n)
        else:
            rows.append((n, js.get("status"), "no labels", "-", "not_evaluable", fnum(ev[n]), fnum(rf[n])))
            st2.append(None)
            ns.append(n)
    if rows:
        L.append(md_table(["N", "status", "max|dpsi1(j)-dpsi1(0)|", "max|dpsi2(j)-dpsi2(0)|",
                           "straight (<= 1e-10)", "e_v", "r_F"], rows))
        L.append("")
        orows = []
        verdicts = {}
        for qn, vals in (("e_v", ev), ("r_F", rf)):
            po = pair_orders(vals, ns)
            for (x, y, o) in po:
                orows.append((qn, "%d->%d" % (x, y), fnum(o, "%.2f"), "pass" if (not math.isnan(o) and o >= 3.5) else "miss"))
            if len(po) < 2:
                verdicts[qn] = "not_evaluable (fewer than two pairs)"
            else:
                first = (not math.isnan(po[0][2])) and po[0][2] >= 3.5
                last = (not math.isnan(po[-1][2])) and po[-1][2] >= 3.5
                if first and last:
                    verdicts[qn] = "pass"
                elif (not first) and last:
                    verdicts[qn] = "pass, pre-asymptotic first pair (reported, not tuned)"
                else:
                    verdicts[qn] = "FAIL (last pair < 3.5)"
        L.append(md_table(["quantity", "pair", "observed order", ">= 3.5"], orows))
        L.append("")
        floor = [qn for qn, vals in (("e_v", ev), ("r_F", rf)) if any(abs(v) < 1e-13 for v in vals.values() if not math.isnan(v))]
        for qn, v in verdicts.items():
            L.append("- A2 order verdict %s: **%s**" % (qn, v))
        if floor:
            L.append("- note: %s reaches < 1e-13 at some N (roundoff floor): the observed order there is not meaningful; "
                     "the verdict above is the literal rule." % ", ".join(floor))
        straight_ok = None if any(s is None for s in st2) else all(st2)
        L.append("- A2 straightness verdict: **%s**" % ("not_evaluable" if straight_ok is None else ("pass" if straight_ok else "FAIL")))
        orders_pass = all(v.startswith("pass") for v in verdicts.values())
        orders_ne = any(v.startswith("not_evaluable") for v in verdicts.values())
        if straight_ok is None or orders_ne:
            overall.append(("A2", "not_evaluable"))
        else:
            overall.append(("A2", "pass" if (straight_ok and orders_pass) else "fail"))
    else:
        L.append("No cosx2 run found: not_evaluable.")
        overall.append(("A2", "not_evaluable"))
    L.append("")

    # ---- A3 -----------------------------------------------------------------------------
    L += ["## A3: control2d (psi2 = x3), eps = 0.5", ""]
    c2 = by_name(a.raw, "control2d")
    rows = []
    for n, (f, js) in sorted(c2.items()):
        d = os.path.join(a.exports, "control2d_N%d" % n)
        if not os.path.isdir(d):
            rows.append((n, js.get("status"), "no labels", "not_evaluable"))
            overall.append(("A3", "not_evaluable"))
            continue
        p1, p2 = load_labels(d)
        n_ = p2.shape[1]
        x3 = (np.arange(n_) / n_)[None, None, :]
        d2 = max_abs(p2 - x3)
        ok = (not math.isnan(d2)) and d2 <= 1e-10
        rows.append((n, js.get("status"), fnum(d2), "pass" if ok else "FAIL"))
        overall.append(("A3", "pass" if ok else "fail"))
    if rows:
        L.append(md_table(["N", "status", "max|psi2 - x3|", "A3 (<= 1e-10)"], rows))
    else:
        L.append("No control2d run found: not_evaluable.")
        overall.append(("A3", "not_evaluable"))
    L.append("")

    # ---- A4 / ctest ---------------------------------------------------------------------
    L += ["## A4 and fast tests (ctest)", ""]
    ct = ctest_results(a.ctest_log) if a.ctest_log else []
    if ct:
        rows = []
        for name, res, t in ct:
            ok = res.startswith("Passed") and t <= 60.0
            rows.append((name, res, "%.2f" % t, "pass" if ok else "FAIL"))
            overall.append(("A4/ctest:" + name, "pass" if ok else "fail"))
        L.append(md_table(["test", "result", "time (s)", "passed and <= 60 s"], rows))
    else:
        L.append("No ctest result lines parsed from `%s`: not_evaluable." % a.ctest_log)
        overall.append(("A4/ctest", "not_evaluable"))
    L.append("")
    L += ["## Summary", ""]
    L.append(md_table(["criterion", "verdict"], overall))
    L.append("")
    write_out("\n".join(L), a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
