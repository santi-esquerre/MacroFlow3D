"""Estimator-noise table (SF-42 G): per amplitude, alpha_TP and alpha_TAP of the consumer production pass (hmax_div 8, or -1 for the
B arms, which have no hmax passes), weights flux, set perplane: mean, across-realization SD, split-half particle share (sf42red
definition) in basis pos and disp, and the SD ratio pos/disp.  Usage: python3 noise_summary.py <scripts dir> <out dir> <set>...
(reads <out dir>/<set>.csv written by `sf42red reduce`; prints the table and the window coverage; writes nothing).  Signs and
magnitudes of alpha_T are not interpreted: this is a design input (which basis has the lower noise)."""
import csv
import math
import os
import sys

import numpy as np

scripts, outdir, sets = sys.argv[1], sys.argv[2], sys.argv[3:]
sys.path.insert(0, scripts)
sys.dont_write_bytecode = True
from sf42red import reduce as red  # noqa: E402

print("set,estimator,basis,n,mean,sd,splitshare_fraction,splitshare_flag,sd_ratio_pos_over_disp")
for s in sets:
    with open(os.path.join(outdir, s + ".csv")) as f:
        rows = [r for r in csv.DictReader(f) if r["code"] == "consumer" and r["family"] == "plane" and r["weights"] == "flux"
                and r["set"] == "perplane" and r["hmax_div"] in ("8", "-1")]
    cov = sorted({(r["d_first"], r["d_last"], r["n_planes"], r["d_lo"], r["d_hi"]) for r in rows})
    print("# %s window coverage (d_first, d_last, n_planes, d_lo, d_hi): %s; rows %d" % (s, cov, len(rows)))
    for e in ("alpha_TP", "alpha_TAP"):
        sd = {}
        for b in ("pos", "disp"):
            rs = [r for r in rows if r["basis"] == b]
            v = np.array([float(r[e]) for r in rs])
            A = np.array([float(r[e + "_A"]) for r in rs])
            B = np.array([float(r[e + "_B"]) for r in rs])
            ok = np.isfinite(v) & np.isfinite(A) & np.isfinite(B)
            sh = red.splithalf_fraction(A[ok], B[ok], v[ok])
            sd[b] = float(v[ok].std(ddof=1))
            print("%s,%s,%s,%d,%.10g,%.10g,%.10g,%d,%s" % (s, e, b, ok.sum(), v[ok].mean(), sd[b], sh["fraction"], sh["flag"],
                                                           "%.10g" % (sd["pos"] / sd["disp"]) if b == "disp" else ""))
