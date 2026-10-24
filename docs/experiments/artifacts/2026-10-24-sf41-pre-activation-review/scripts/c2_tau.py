"""C2 design emulation: DP5(4) plane-pass tau against an independent quadrature of 1/v1 on a
layered field (numpy; not SF-41 evidence).

Why 1-D suffices: samples depend on x1 only, so the transverse gradients of both splines vanish
up to roundoff (<= 1e-16 relative); |g| = |g1|, dx1/ds = g1/|g| = 1 and the position error
estimate of the arclength step (sum of the e_i, which add to 0, times h) is rounding-level, far
below tol = 1e-8: every step is accepted with the clamp factor 5 and h = h_max after the warm-up
(0.1, 0.5, 1 x h_max).  tau rides passively (dtau/ds = 1/(k|g|)), so the tau error is the DP5
quadrature error at the FIXED step h_max = h/8 plus the Henon-landing steps; no adaptivity.
Run: OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 15 taskset -c 10-15 python3 -I c2_tau.py
"""
import importlib.util
import math
import os

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("c_common", os.path.join(here, "c_common.py"))
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

TOL = 1e-8


def run_case(label, n, h, lnk_fn, planes, kind_f, kind_lnk, medium, quiet=False):
    """medium 'a' (periodic exact head, zero-mean gauge) or 'b' (Dirichlet exact discrete head)."""
    x = (np.arange(n) + 0.5) * h
    lnk = lnk_fn(x)
    if medium == "a":
        f, q = cc.layered_f_periodic(lnk, h)
    else:
        f, q = cc.layered_f_dirichlet(lnk, h)
    sf = cc.Spline1D(kind_f, n, h, f)
    sk = cc.Spline1D(kind_lnk, n, h, lnk)

    def field(xx):
        _, df = sf(xx)
        y, _ = sk(xx)
        return 1.0 + df, math.exp(y)

    taus, nsteps = cc.plane_pass_1d(field, planes, h / 8.0, TOL)
    rels, refs = [], []
    for X, tau in zip(planes, taus):
        ref = cc.tau_quadrature(field, 0.0, X, h)
        refs.append(ref)
        rels.append(abs(tau - ref) / abs(ref))
    if quiet:
        return max(rels)
    v1 = np.array([field(xx)[0] * field(xx)[1] for xx in np.linspace(0.0, n * h, 4001)[:-1]])
    print(f"{label:62s} n={n:4d} h=1/{round(1/h):<3d} planes={planes}")
    print(f"    q={q:.6f}  v1 range [{v1.min():.4f}, {v1.max():.4f}]  accepted_steps={nsteps}")
    print("    tau rel per plane: " + " ".join(f"{r:.3e}" for r in rels))
    print(f"    MAX tau rel = {max(rels):.3e}   (gate 1e-9: {'PASS' if max(rels) <= 1e-9 else 'FAIL'})")
    return max(rels)


def cos06(x):
    return 0.5 * np.cos(2 * math.pi * 0.6 * x + 0.3)


def cos10(x):
    return 0.5 * np.cos(2 * math.pi * x + 0.3)


def f2prof(x):
    return 0.5 * np.cos(2 * math.pi * x)


P2 = [0.25, 0.5, 1.0, 1.5, 1.75]
P4 = [0.25, 0.5, 1.0, 2.0, 3.0, 3.75]
PF2 = [0.25, 0.5, 1.0, 2.0, 3.0]
res = {}

print("=== (i) SF-38 F2 reproduction: periodic medium (a), ln k = 0.5 cos(2 pi x1), periodic splines")
print("    measured on V100 (SF-38 F2): tau rel 4.771e-10 at 64x32x32 and 128x32x32 (h = 1/32);")
print("    SF-36 A2: 4.6e-9 at 16^3 (h = 1/16), 4.77e-10 at 32^3.   Cube L1 = 1 (16, 32 cells) added.")
for n, h in [(16, 1 / 16), (32, 1 / 32), (64, 1 / 32), (128, 1 / 32)]:
    res[("F2", n, round(1 / h))] = run_case("F2 periodic, periodic splines", n, h, f2prof, PF2,
                                            "per", "per", "a")
print()
print("=== (ii) C2 candidate: medium (b), ln k = 0.5 cos(2 pi 0.6 x1 + 0.3), not-a-knot f and ln k")
res[("C2", 64, 32)] = run_case("C2 cand L1=2", 64, 1 / 32, cos06, P2, "nak", "nak", "b")
res[("C2", 128, 32)] = run_case("C2 cand L1=4", 128, 1 / 32, cos06, P4, "nak", "nak", "b")
print()
print("=== (iii) variants")
res[("V1", 64, 32)] = run_case("V1 PERIODIC ln k spline, nak f, 0.6-profile L1=2", 64, 1 / 32, cos06, P2,
                               "nak", "per", "b")
res[("V1", 128, 32)] = run_case("V1 PERIODIC ln k spline, nak f, 0.6-profile L1=4", 128, 1 / 32, cos06,
                                P4, "nak", "per", "b")
res[("V2", 64, 32)] = run_case("V2 profile 0.5cos(2 pi x1+0.3), nak/nak L1=2", 64, 1 / 32, cos10, P2,
                               "nak", "nak", "b")
res[("V2", 128, 32)] = run_case("V2 profile 0.5cos(2 pi x1+0.3), nak/nak L1=4", 128, 1 / 32, cos10, P4,
                                "nak", "nak", "b")
res[("V3", 32, 16)] = run_case("V3 h=1/16 candidate L1=2", 32, 1 / 16, cos06, P2, "nak", "nak", "b")
res[("V3", 64, 16)] = run_case("V3 h=1/16 candidate L1=4", 64, 1 / 16, cos06, P4, "nak", "nak", "b")
res[("V4", 64, 32)] = run_case("V4 F2 profile 0.5cos(2 pi x1), medium (b), nak/nak L1=2", 64, 1 / 32,
                               f2prof, P2, "nak", "nak", "b")
res[("V4", 128, 32)] = run_case("V4 F2 profile 0.5cos(2 pi x1), medium (b), nak/nak L1=4", 128, 1 / 32,
                                f2prof, P4, "nak", "nak", "b")
print()
print("=== (iv) sensitivity of the candidate to frequency nu and phase phi (ln k = 0.5 cos(2 pi nu x1 + phi)),")
print("    h = 1/32, L1 = 2, not-a-knot f and ln k, planes of C2 (L1 = 2); MAX tau rel only")
for nu in (0.3, 0.6, 0.9, 1.5):
    for phi in (0.3, 1.0):
        r = run_case("scan", 64, 1 / 32, lambda x, nu=nu, phi=phi: 0.5 * np.cos(2 * math.pi * nu * x + phi),
                     P2, "nak", "nak", "b", quiet=True)
        print(f"    nu={nu:3.1f} phi={phi:3.1f}: {r:.3e}")
print()
print("=== (v) amplitude of the scan profile (h = 1/32, L1 = 2, nak/nak, C2 planes): MAX tau rel")
for amp in (0.5, 0.25):
    for nu, phi in ((0.6, 0.3), (0.6, 1.0), (0.9, 1.0)):
        r = run_case("scan", 64, 1 / 32, lambda x, a=amp, nu=nu, phi=phi: a * np.cos(2 * math.pi * nu * x + phi),
                     P2, "nak", "nak", "b", quiet=True)
        print(f"    amp={amp:4.2f} nu={nu:3.1f} phi={phi:3.1f}: {r:.3e}")
print()
print("=== (v-b) phase neighbourhood of the candidate (nu = 0.6, amp 0.5, h = 1/32, L1 = 2 and L1 = 4): MAX tau rel")
for phi in (0.1, 0.2, 0.3, 0.4, 0.5):
    r2 = run_case("scan", 64, 1 / 32, lambda x, phi=phi: 0.5 * np.cos(2 * math.pi * 0.6 * x + phi), P2,
                  "nak", "nak", "b", quiet=True)
    r4 = run_case("scan", 128, 1 / 32, lambda x, phi=phi: 0.5 * np.cos(2 * math.pi * 0.6 * x + phi), P4,
                  "nak", "nak", "b", quiet=True)
    print(f"    phi={phi:3.1f}: L1=2 {r2:.3e}   L1=4 {r4:.3e}")
print()
print("=== (vi) alignment dependence: ONE plane list (C2 candidate, L1 = 2, h = 1/32): signed tau - tau_quad")
print("    (the arclength grid is 0.1, 0.5, 1, 1, .. x h_max from the seed, i.e. offset 0.6 h_max from the")
print("    spline knots at multiples of h/2 = 4 h_max; a landing re-aligns the grid; hence the error is a")
print("    sum of knot-straddling DP5 errors whose sign/size varies with the plane list)")
_x = (np.arange(64) + 0.5) / 32
_lnk = cos06(_x)
_f, _q = cc.layered_f_dirichlet(_lnk, 1 / 32)
_sf, _sk = cc.Spline1D("nak", 64, 1 / 32, _f), cc.Spline1D("nak", 64, 1 / 32, _lnk)
_fld = lambda xx: (1.0 + _sf(xx)[1], math.exp(_sk(xx)[0]))
for pl in ([0.2], [0.25], [0.3], [0.5], [0.6], [1.0], [0.0625, 0.25], [0.125, 0.25], [0.25, 0.5]):
    tt, _ = cc.plane_pass_1d(_fld, pl, 1 / 256, TOL)
    print(f"    planes={str(pl):16s} " + " ".join(f"{t - cc.tau_quadrature(_fld, 0.0, X, 1 / 32):+.2e}"
                                               for X, t in zip(pl, tt)))
