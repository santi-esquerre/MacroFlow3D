#!/usr/bin/env python3
"""Post-hoc analysis (NOT pre-registered): how the grid resolution the Lagrangian consumer needs grows with sigma^2.

Reads only versioned SF-37 outputs (no new runs) and prints the tables of
docs/experiments/2026-10-23-resolution-vs-variance.md.

Usage (from the repository root):
  python3 -I docs/experiments/artifacts/2026-10-23-resolution-vs-variance/scripts/resolution_vs_variance.py \
      docs/experiments/artifacts/2026-10-19-sf37-lagrangian-cube-verification \
      > docs/experiments/artifacts/2026-10-23-resolution-vs-variance/analysis/resolution_vs_variance.txt
"""
import csv
import hashlib
import math
import random
import statistics as st
import sys
from collections import defaultdict

A = sys.argv[1]
V = f"{A}/analysis/v1v2"
LH = (8, 10, 16, 20, 32, 64)          # SF-37 candidate lambda/h (N = 4 lambda/h, reference N = 512 = lambda/h 128)
S2 = ("1.0", "2.25", "4.0")
THR = 0.05                            # SF-37 relative threshold (det branch of RR5)


def rd(name):
    with open(f"{V}/{name}") as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def sha(name):
    with open(f"{V}/{name}", "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def rel(r):
    return float(r["D"]) / abs(float(r["q_ref"]))


print("# Post-hoc, not pre-registered. Inputs (sha256):")
for n in ("lambda_h_cells.csv", "v2_cells.csv", "d8_shadow_lambda_h.csv"):
    print(f"#   {n} {sha(n)}")

# ---------------------------------------------------------------- 1. exact statistics of K = exp(Y)
# Y Gaussian, C(r) = s2 exp(-(r/lambda)^2).  Cov K(r) = exp(2 mu + s2) (exp(C(r)) - 1)  (lognormal identity)
# rho_K(r) = (exp(C(r)) - 1) / (exp(s2) - 1);  Var dY/dx_i = -C''(0) = 2 s2 / lambda^2.
def rho_k(r, s2):
    return (math.exp(s2 * math.exp(-r * r)) - 1) / (math.exp(s2) - 1)


def efold_k(s2):  # rho_K(r) = 1/e, r in units of lambda
    u = math.log(1 + (math.exp(s2) - 1) / math.e) / s2
    return math.sqrt(-math.log(u))


# self-check of the lognormal identity by Monte Carlo at s2 = 1, r = 0.5 lambda (fixed seed)
rng, n, s2c, r0 = random.Random(20261023), 200000, 1.0, 0.5
rho_y = math.exp(-r0 * r0)
k1, k2 = [], []
for _ in range(n):
    y1 = rng.gauss(0, 1)
    y2 = rho_y * y1 + math.sqrt(1 - rho_y ** 2) * rng.gauss(0, 1)
    k1.append(math.exp(math.sqrt(s2c) * y1))
    k2.append(math.exp(math.sqrt(s2c) * y2))
m1, m2 = st.fmean(k1), st.fmean(k2)
prod = [(a - m1) * (b - m2) for a, b in zip(k1, k2)]
mc = st.fmean(prod) / (st.pstdev(k1) * st.pstdev(k2))
se = st.pstdev(prod) / math.sqrt(n) / (st.pstdev(k1) * st.pstdev(k2))
assert abs(mc - rho_k(r0, s2c)) < 4 * se, (mc, rho_k(r0, s2c), se)
print(f"\n## 1. Exact field statistics (lognormal identity MC check at s2=1, r=0.5 lambda: "
      f"MC {mc:.4f} vs formula {rho_k(r0, s2c):.4f}, SE {se:.4f}: OK)")
print("s2    sigma  rms(dY/dx_i)*lambda  K_efold/lambda  K_microscale/Y_microscale  K(+2s)/K(-2s)")
for s2 in (1.0, 2.25, 4.0, 6.25):
    s = math.sqrt(s2)
    print(f"{s2:<5} {s:<6.3f} {math.sqrt(2) * s:<20.3f} {efold_k(s2):<15.3f} "
          f"{math.sqrt((1 - math.exp(-s2)) / s2):<26.3f} {math.exp(4 * s):.0f}")

# ---------------------------------------------------------------- 2. SF-37 lambda/h cells
rows = rd("lambda_h_cells.csv")
assert len(rows) == 3 * 3 * 6 * 2 * 7, len(rows)
for r in rows:  # the reading reproduces every SF-37 verdict
    ok = abs(float(r["D"])) <= max(float(r["thr_rel"]), float(r["thr_2se"]))
    assert ok == (r["verdict"] == "pass"), r
print(f"\n## 2. SF-37 lambda/h cells: {len(rows)} cells, every SF-37 verdict reproduced from |D| <= max(thr_rel, thr_2se)")


def cells(s2, lh=None, **kw):
    return [r for r in rows if r["sigma2"] == s2 and (lh is None or r["lambda_over_h"] == str(lh))
            and all(r[k] == v for k, v in kw.items())]


def seedmean_max(s2, lh):  # max over (series, plane) of |mean over the 3 seeds of the signed relative error|
    d = defaultdict(list)
    for r in cells(s2, lh):
        d[(r["series"], r["plane"])].append(rel(r))
    return max(abs(st.fmean(v)) for v in d.values())


W = {(s2, lh): max(abs(rel(r)) for r in cells(s2, lh)) for s2 in S2 for lh in LH}
MED = {(s2, lh): st.median(abs(rel(r)) for r in cells(s2, lh)) for s2 in S2 for lh in LH}
SM = {(s2, lh): seedmean_max(s2, lh) for s2 in S2 for lh in LH}

print("\n### 2a. Relative error |Q(N)-Q(512)|/|Q(512)| by sigma^2 and lambda/h (42 cells each: 3 seeds x 2 Q x 7 planes)")
print("s2    lambda/h  worst    median   seed-mean(max)  sigma_T2 cells with D<0  var_tau cells with D<0  SF-37 fails")
for s2 in S2:
    for lh in LH:
        neg = {q: sum(1 for r in cells(s2, lh, series=q) if float(r["D"]) < 0) for q in ("sigma_T2", "var_tau")}
        fails = sum(1 for r in cells(s2, lh) if r["verdict"] != "pass")
        print(f"{s2:<5} {lh:<9} {W[s2, lh]:<8.4f} {MED[s2, lh]:<8.4f} {SM[s2, lh]:<15.4f} "
              f"{neg['sigma_T2']:>2}/21{'':<18} {neg['var_tau']:>2}/21{'':<17} {fails}")

print("\n### 2b. Worst relative error per seed (all Q, planes) by lambda/h")
for s2 in S2:
    for seed in ("3101", "3102", "3103"):
        vals = " ".join(f"{lh}:{max(abs(rel(r)) for r in cells(s2, lh, seed=seed)):.4f}" for lh in LH)
        print(f"s2 {s2:<5} seed {seed}  {vals}")

print("\n### 2c. Worst cell location (seed, Q, plane x1; x1 = 16 is 64 lambda) at lambda/h 16 and 32")
for s2 in S2:
    for lh in (16, 32):
        w = max(cells(s2, lh), key=lambda r: abs(rel(r)))
        print(f"s2 {s2:<5} lambda/h {lh:<3} seed {w['seed']} {w['series']:<8} x1 {w['plane']:<4} rel {rel(w):+.4f}")

planes = sorted({r["plane"] for r in rows}, key=float)
print("\n### 2d. Median |relative error| of sigma_T2 over seeds by plane at lambda/h 16 (growth with travel distance)")
print("s2    " + " ".join(f"x1={p:<6}" for p in planes) + " slope d ln(err)/d ln(x1) over x1 2..16")
for s2 in S2:
    m = [st.median(abs(rel(r)) for r in cells(s2, 16, series="sigma_T2", plane=p)) for p in planes]
    xs = [math.log(float(p)) for p in planes if float(p) >= 2]
    ys = [math.log(v) for p, v in zip(planes, m) if float(p) >= 2]
    xm, ym = st.fmean(xs), st.fmean(ys)
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / sum((x - xm) ** 2 for x in xs)
    print(f"{s2:<5} " + " ".join(f"{v:<9.4f}" for v in m) + f" {slope:.2f}")


def required(errs, thr):
    """lambda/h where the error curve crosses thr (log-log linear between candidates; below 8 extrapolated)."""
    for i in range(len(LH)):
        if all(e <= thr for e in errs[i:]):
            if i == 0:
                s = (math.log(errs[2]) - math.log(errs[0])) / (math.log(16) - math.log(8))
                return math.exp(math.log(8) + (math.log(thr) - math.log(errs[0])) / s), "extrapolated below 8"
            x0, x1 = math.log(LH[i - 1]), math.log(LH[i])
            y0, y1 = math.log(errs[i - 1]), math.log(errs[i])
            return math.exp(x0 + (math.log(thr) - y0) * (x1 - x0) / (y1 - y0)), f"between {LH[i-1]} and {LH[i]}"
    return float("nan"), "none"


print("\n### 2e. Continuous lambda/h at which the worst relative error reaches 5 % (det branch of the SF-37 rule) and gamma = sqrt(2) sigma h / lambda there")
req = {}
for s2 in S2:
    v, how = required([W[s2, lh] for lh in LH], THR)
    req[s2] = v
    print(f"s2 {s2:<5} lambda/h_req {v:6.2f} ({how}); gamma {math.sqrt(2 * float(s2)) / v:.3f}")
for s2, lh in zip(S2, (8, 16, 32)):
    print(f"SF-37 production value s2 {s2:<5} lambda/h {lh:<3} gamma {math.sqrt(2 * float(s2)) / lh:.3f}")

print("\n### 2f. Threshold-free exponent beta in lambda/h_req ~ sigma^beta: beta = ln(err(s2=4)/err(s2=a)) / (p ln(2/sqrt(a))), p = 2")
for name, M in (("worst", W), ("seed-mean", SM), ("median", MED)):
    out = []
    for lh in (16, 32):
        b1 = math.log(M["4.0", lh] / M["1.0", lh]) / (2 * math.log(2))
        b2 = math.log(M["4.0", lh] / M["2.25", lh]) / (2 * math.log(2 / 1.5))
        out.append(f"lambda/h {lh}: beta(1->4) {b1:.2f}, beta(2.25->4) {b2:.2f}")
    print(f"{name:<10} " + " | ".join(out))

print("\n### 2g. Cell-level ordering in sigma^2 (same seed, Q, plane, lambda/h; 126 cells per Q)")
E = {(r["sigma2"], r["seed"], r["series"], r["plane"], r["lambda_over_h"]): abs(rel(r)) for r in rows}
keys = {k[1:] for k in E}
for q in ("sigma_T2", "var_tau"):
    kq = [k for k in keys if k[1] == q]
    up = sum(1 for k in kq if E[("1.0",) + k] < E[("2.25",) + k] < E[("4.0",) + k])
    up14 = sum(1 for k in kq if E[("1.0",) + k] < E[("4.0",) + k])
    print(f"{q:<8} err(1) < err(2.25) < err(4): {up}/{len(kq)}   err(1) < err(4): {up14}/{len(kq)}")

# ---------------------------------------------------------------- 3. observed order of the V2 ladder (128 -> 256 -> 512)
v2 = rd("v2_cells.csv")
print("\n## 3. V2 observed order p (triple 128/256/512) per sigma^2 and Q")
for s2 in S2:
    for q in ("sigma_T2", "var_tau"):
        p = [float(r["p"]) for r in v2 if r["sigma2"] == s2 and r["series"] == q and r["p"] not in ("", "nan")]
        inside = sum(1 for x in p if 1.5 <= x <= 2.5)
        print(f"s2 {s2:<5} {q:<8} n {len(p):>2}  median {st.median(p):.2f}  min {min(p):.2f}  max {max(p):.2f}  in [1.5, 2.5]: {inside}/{len(p)}")

# ---------------------------------------------------------------- 4. D8 shadow lambda/h of the SF-39 estimator series (same rule)
d8 = rd("d8_shadow_lambda_h.csv")
eps = {"10": "1", "15": "2.25", "20": "4"}
print("\n## 4. D8 shadow lambda/h (monotone: smallest candidate passing with every larger one), same RR5 rule")
series = ("var_x1", "dvar_x1_dt", "var_x2", "var_x3", "dvar_x2_dt", "dvar_x3_dt", "dsigma_T2_dx1")
print("s2    " + " ".join(f"{s:<14}" for s in series))
for e, s2 in eps.items():
    out = []
    for s in series:
        st_by = {int(r["lambda_over_h"]): r["status"] for r in d8 if r["eps_code"] == e and r["series"] == s}
        mono = next((lh for lh in LH if all(st_by[x] == "pass" for x in LH if x >= lh)), None)
        out.append(f"{mono!s:<14}")
    print(f"{s2:<5} " + " ".join(out))
