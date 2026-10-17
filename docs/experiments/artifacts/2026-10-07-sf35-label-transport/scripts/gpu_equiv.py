#!/usr/bin/env python3
"""SF-35 F (V100 part): host vs GPU tracer equivalence from the exports and the driver JSON.

  gpu_equiv.py --case N HOST.json GPU.json HOST_EXPORT_DIR GPU_EXPORT_DIR [--case ...] [--out analysis/gpu.md]
  gpu_equiv.py --not-built [--out analysis/gpu.md]      (the binary rejected --tracer gpu: record and exit 0)

Per case: max|dpsi1|, max|dpsi2| between the two exports (criterion <= 1e-12), per-plane nfev and status-histogram
equality from the JSON (transport.status_hist_per_plane, planes[].nfev), both statuses, and wall times (driver
`timing.trace` and `timing.total`).  If nfev / histograms differ, the number of planes that differ is reported; the 1e-12
bound remains the criterion.  Deterministic; numpy only.
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import fnum, get, load_json, md_table, num, write_out  # noqa: E402


def maxdiff(a, b):
    na, nb = np.isnan(a), np.isnan(b)
    mism = int(np.count_nonzero(na != nb))
    both = ~(na | nb)
    return (float(np.max(np.abs(a[both] - b[both]))) if both.any() else float("nan")), mism


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", nargs=5, action="append", metavar=("N", "HOST_JSON", "GPU_JSON", "HOST_DIR", "GPU_DIR"),
                    help="one grid: N, host JSON, gpu JSON, host export dir, gpu export dir")
    ap.add_argument("--not-built", action="store_true", help="record that the GPU tracer is not built")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    L = ["# SF-35 F: host vs GPU tracer", ""]
    if a.not_built:
        L.append("The binary rejected `--tracer gpu` (exit 1, \"not built\"): the GPU tracer node is not part of this "
                 "build; criterion F is not_evaluable here (non-blocking).")
        L.append("")
        write_out("\n".join(L), a.out)
        return 0
    if not a.case:
        ap.error("need --case ... or --not-built")
    rows, states, drows = [], [], []
    for n, hj, gj, hd, gd in a.case:
        try:
            H, G = load_json(hj), load_json(gj)
            h1, h2 = np.load(os.path.join(hd, "psi1.npy")), np.load(os.path.join(hd, "psi2.npy"))
            g1, g2 = np.load(os.path.join(gd, "psi1.npy")), np.load(os.path.join(gd, "psi2.npy"))
        except (OSError, ValueError) as ex:
            rows.append((n, "-", "-", "-", "-", "-", "-", "not_evaluable (%s)" % ex))
            states.append("not_evaluable")
            continue
        d1, m1 = maxdiff(h1, g1)
        d2, m2 = maxdiff(h2, g2)
        hs = get(H, "transport", "status_hist_per_plane", default=[]) or []
        gs = get(G, "transport", "status_hist_per_plane", default=[]) or []
        nh = [p.get("nfev") for p in (H.get("planes") or [])]
        ng = [p.get("nfev") for p in (G.get("planes") or [])]
        hist_diff = sum(1 for x, y in zip(hs, gs) if x != y) + abs(len(hs) - len(gs))
        nfev_diff = sum(1 for x, y in zip(nh, ng) if x != y) + abs(len(nh) - len(ng))
        ok = (not math.isnan(max(d1, d2))) and max(d1, d2) <= 1e-12 and m1 == 0 and m2 == 0
        states.append("pass" if ok else "fail")
        rows.append((n, fnum(d1), fnum(d2), m1 + m2, "%s / %s" % (H.get("status"), G.get("status")),
                     "identical" if nfev_diff == 0 and hist_diff == 0 else "%d planes nfev, %d planes hist differ" % (nfev_diff, hist_diff),
                     "pass" if ok else "FAIL", ""))
        drows.append((n, fnum(get(H, "timing", "trace"), "%.2f"), fnum(get(G, "timing", "trace"), "%.2f"),
                      fnum(get(H, "timing", "total"), "%.2f"), fnum(get(G, "timing", "total"), "%.2f"),
                      fnum(num(get(H, "timing", "trace")) / num(get(G, "timing", "trace")), "%.2f")))
    L.append(md_table(["N", "max|dpsi1|", "max|dpsi2|", "NaN mismatches", "status host / gpu",
                       "per-plane nfev and status histograms", "<= 1e-12", "note"], rows))
    L.append("")
    L.append("## Wall times (driver `timing`, seconds)")
    L.append("")
    L.append(md_table(["N", "trace host", "trace gpu", "total host", "total gpu", "trace speedup host/gpu"], drows))
    L.append("")
    v = "FAIL" if "fail" in states else ("pass" if states and all(s == "pass" for s in states) else "not_evaluable")
    L.append("**Verdict F (max |dpsi| host vs gpu <= 1e-12 at the grids run): %s**" % v)
    L.append("")
    write_out("\n".join(L), a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
