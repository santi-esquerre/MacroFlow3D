#!/usr/bin/env python3
"""PROBE (SF-35-feas): per-plane e_v_j, e_i1_j, e_i2_j at local x1/ell = M (ell = 1/4) and observed orders.

  planes.py [--shift S] COARSE.json FINE.json     (N_fine = 2 N_coarse; S = original-x1 offset of the
                                                   new inlet in cube units, labelling only)
Plane j has x1 = j / N, i.e. x1/ell = 4 j / N, so j = M N / 4 (same x1 on both grids: j_f = 2 j_c).
"""
import argparse
import json
import math

ap = argparse.ArgumentParser()
ap.add_argument("--shift", type=float, default=0.0)
ap.add_argument("coarse")
ap.add_argument("fine")
a = ap.parse_args()
MS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]


def load(p):
    with open(p) as f:
        d = json.load(f)
    return d["N"], {pl["j"]: pl for pl in d["planes"]}, d


def order(c, f):
    if c is None or f is None or c <= 0 or f <= 0:
        return float("nan")
    return math.log2(c / f)


nc, pc, dc = load(a.coarse)
nf, pf, df = load(a.fine)
assert nf == 2 * nc, (nc, nf)
print(f"coarse N={nc} status={dc.get('status')}  fine N={nf} status={df.get('status')}  shift={a.shift} (cube units; orig x1/ell = local + {a.shift*4:g})")
print("x1/ell(local)  x1/ell(orig)  | e_v(N%d)  e_v(N%d)  ord | e_i1(N%d) e_i1(N%d) ord | e_i2(N%d) e_i2(N%d) ord" % (nc, nf, nc, nf, nc, nf))
for m in MS:
    jc = round(m * nc / 4.0)
    jf = 2 * jc
    if jc not in pc or jf not in pf:
        continue
    row = [f"{m:6.2f}", f"{m + 4*a.shift:6.2f}"]
    s = f"{row[0]}         {row[1]}        |"
    for key in ("e_v", "e_i1", "e_i2"):
        c, f = pc[jc][key], pf[jf][key]
        s += f" {c:.3e} {f:.3e} {order(c, f):6.2f} |"
    print(s)
