#!/usr/bin/env python3
"""Max |dpsi| between two saved label exports (informative comparison E, sigma2 2.25 eps 1 vs sigma2 1 eps 1.5).

  labels_diff.py DIR_A DIR_B [--out analysis/info_sigma_eps.md] [--expect 1e-12]

Each DIR holds psi1.npy, psi2.npy ((N+1, N, N) float64, slab layout) and labels.json (from --save-labels).
NaN labels (failed seeds) in either export are counted and excluded from the max (the NaN masks must coincide, else
`nan_mismatch` is reported).  The expectation (<= 1e-12) is recorded in the output; it is not a pass/fail gate of the
spec (E: recorded, no threshold) but the digest states whether it was met.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import fnum, md_table, rel, write_out  # noqa: E402


def load(d):
    meta = {}
    p = os.path.join(d, "labels.json")
    if os.path.exists(p):
        with open(p) as f:
            meta = json.load(f)
    return np.load(os.path.join(d, "psi1.npy")), np.load(os.path.join(d, "psi2.npy")), meta


def diff(a, b):
    na, nb = np.isnan(a), np.isnan(b)
    mism = int(np.count_nonzero(na != nb))
    both = ~(na | nb)
    m = float(np.max(np.abs(a[both] - b[both]))) if both.any() else float("nan")
    return m, int(np.count_nonzero(na)), int(np.count_nonzero(nb)), mism


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir_a")
    ap.add_argument("dir_b")
    ap.add_argument("--out")
    ap.add_argument("--expect", type=float, default=1e-12, help="expected bound (default 1e-12)")
    a = ap.parse_args(argv)
    try:
        a1, a2, ma = load(a.dir_a)
        b1, b2, mb = load(a.dir_b)
    except OSError as ex:
        print("ERROR: %s" % ex, file=sys.stderr)
        return 2
    if a1.shape != b1.shape:
        print("ERROR: shapes differ %s vs %s" % (a1.shape, b1.shape), file=sys.stderr)
        return 2
    d1, n1a, n1b, m1 = diff(a1, b1)
    d2, n2a, n2b, m2 = diff(a2, b2)
    worst = np.nanmax([d1, d2]) if not (np.isnan(d1) and np.isnan(d2)) else float("nan")
    L = ["# SF-35 informative comparison: same exp(eps Y) amplitude through two parametrizations", ""]
    L.append("- A: `%s` (field %s, eps %s)" % (rel(a.dir_a), ma.get("field", {}).get("sigma2", "?"), ma.get("eps", "?")))
    L.append("- B: `%s` (field %s, eps %s)" % (rel(a.dir_b), mb.get("field", {}).get("sigma2", "?"), mb.get("eps", "?")))
    L.append("")
    L.append(md_table(["label", "max|dpsi|", "NaN in A", "NaN in B", "NaN mask mismatches"],
                      [("psi1", fnum(d1), n1a, n1b, m1), ("psi2", fnum(d2), n2a, n2b, m2)]))
    L.append("")
    met = (not np.isnan(worst)) and worst <= a.expect and m1 == 0 and m2 == 0
    L.append("Expected `<= %.0e` (recorded, no threshold in the spec): **%s** (max over both labels: %s)" % (
        a.expect, "met" if met else "NOT met", fnum(worst)))
    L.append("")
    write_out("\n".join(L), a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
