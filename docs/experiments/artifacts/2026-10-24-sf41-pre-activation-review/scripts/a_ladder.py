"""SF-41 pre-activation design emulation for threshold A (DESIGN evidence, numpy; NOT SF-41 evidence).

Spline emulated: uniform cubic B-spline on the SF-28 cell-centred lattice, NOT-A-KNOT closure in x1 on the raw samples
(no ghost samples), periodic in x2/x3 (see a_spline.py).  Prefilter variants: x line solve = reduced tridiagonal | dense LU |
sparse LU; y/z = FFT ("GPU-like") | cyclic tridiagonal ("CPU-mirror-like").

Selection criterion for the threshold-A order-ladder test field (fixed in this file BEFORE any order is computed, applied to
every candidate, no candidate dropped):
  boxes (nx,ny,nz) = (32,16,16), (64,32,32), (128,64,64)  (L1 = 2, h = 1/16, 1/32, 1/64);
  fixed physical point set: 10000 random points in [0,L1]x[0,1)^2 + 2000 points within one coarsest cell (1/16) of each x face
  (all inside [0,L1]);
  CRIT-1 (exercises the x closure): at the finest level the max-norm error of the value and of the Hessian is attained at a
         point within 1/16 of an x face;
  CRIT-2: predicted min orders over 16->32 and 32->64 (value max, value RMS, gradient max, gradient RMS, Hessian max) are
         >= gate + 0.05, gates 3.8 / 3.8 / 2.8 / 2.8 / 1.8.
  A candidate passes iff CRIT-1 and CRIT-2 hold.  Recommended = the passing candidate with the largest minimum margin
  (min over the five orders of order - (gate + 0.05)); if none passes the script says so.
Norms: gradient error = Euclidean norm per point; Hessian error = component-wise max over the 6 components.

usage: python3 -I a_ladder.py   (no arguments)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.interpolate import CubicSpline
from a_spline import (prefilter_np, prefilter_periodic_fft, prefilter_periodic_cyc, nak_x, evaluate3, evaluate1,
                      gaussian_field)

GATES = {"val_max": 3.8, "val_rms": 3.8, "grad_max": 2.8, "grad_rms": 2.8, "hess_max": 1.8}
MARGIN_PAD = 0.05
TWOPI = 2 * np.pi
COARSE = 1.0 / 16.0


def hdr(s):
    print()
    print("#" * 100)
    print("# " + s)
    print("#" * 100)


# ------------------------------------------------------------------ analytic candidates
def g_exp07(x):
    g = np.exp(0.7 * x)
    return g, 0.7 * g, 0.49 * g


def g_expsin(x):
    e = np.exp(x)
    s, c = np.sin(3 * x), np.cos(3 * x)
    return e * s, e * (s + 3 * c), e * (-8 * s + 6 * c)


def g_nat(x):
    a, b = TWOPI * 1.1, TWOPI * 0.6
    return (np.sin(a * x + 0.3) + 0.5 * np.cos(b * x),
            a * np.cos(a * x + 0.3) - 0.5 * b * np.sin(b * x),
            -a * a * np.sin(a * x + 0.3) - 0.5 * b * b * np.cos(b * x))


CANDIDATES = [  # (name, g, amplitude a of the factor 1 + a sin(2 pi x2) cos(2 pi x3))
    ("C1 exp(0.7 x1)*(1+0.3 S)", g_exp07, 0.3),
    ("C2 exp(x1) sin(3 x1)*(1+0.3 S)", g_expsin, 0.3),
    ("C3 exp(x1) sin(3 x1)*(1+0.05 S)", g_expsin, 0.05),
    ("C4 exp(x1) sin(3 x1)*(1+0.01 S)", g_expsin, 0.01),
    ("C5 exp(0.7 x1)*(1+0.01 S)", g_exp07, 0.01),
    ("C6 natural1D(x1)*(1+0.01 S)", g_nat, 0.01),
]


def exact3(gfun, a, pts):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    g0, g1, g2 = gfun(x)
    sy, cy, sz, cz = np.sin(TWOPI * y), np.cos(TWOPI * y), np.sin(TWOPI * z), np.cos(TWOPI * z)
    q = 1 + a * sy * cz
    qy, qz = a * TWOPI * cy * cz, -a * TWOPI * sy * sz
    qyy = qzz = -a * TWOPI**2 * sy * cz
    qyz = -a * TWOPI**2 * cy * sz
    v = g0 * q
    g = np.column_stack([g1 * q, g0 * qy, g0 * qz])
    H = np.column_stack([g2 * q, g0 * qyy, g0 * qzz, g1 * qy, g1 * qz, g0 * qyz])
    return v, g, H


def samples3(gfun, a, nx, ny, nz, h):
    x = (np.arange(nx) + 0.5) * h
    y = (np.arange(ny) + 0.5) * h
    z = (np.arange(nz) + 0.5) * h
    return gfun(x)[0][:, None, None] * (1 + a * np.sin(TWOPI * y)[None, :, None] * np.cos(TWOPI * z)[None, None, :])


def ladder_points(L1, seed=777, nrand=10000, nnear=2000):
    rng = np.random.default_rng(seed)
    px = np.concatenate([rng.uniform(0, L1, nrand), rng.uniform(0, COARSE, nnear), rng.uniform(L1 - COARSE, L1, nnear)])
    px = np.clip(px, 0.0, L1)
    return np.column_stack([px, rng.uniform(0, 1, len(px)), rng.uniform(0, 1, len(px))])


def margin_points(L1, h, seed=4242, ndraw=4):
    r = (np.arange(32) + 0.5) / 64.0                    # fixed x/h offsets in (0, 1/2)
    rng = np.random.default_rng(seed)
    xs = np.concatenate([-h * r, L1 + h * r])
    xs = np.repeat(xs, ndraw)
    return np.column_stack([xs, rng.uniform(0, 1, len(xs)), rng.uniform(0, 1, len(xs))])


def norms(ev, ex):
    ev_v, ev_g, ev_H = ev
    ex_v, ex_g, ex_H = ex
    ev_ = ev_v - ex_v
    eg_ = np.linalg.norm(ev_g - ex_g, axis=1)
    eH_ = np.max(np.abs(ev_H - ex_H), axis=1)
    return {"val_max": np.abs(ev_).max(), "val_rms": np.sqrt(np.mean(ev_**2)),
            "grad_max": eg_.max(), "grad_rms": np.sqrt(np.mean(eg_**2)), "hess_max": eH_.max()}, ev_, eH_


def orders(vals):
    return [np.log2(vals[i] / vals[i + 1]) for i in range(len(vals) - 1)]


# ------------------------------------------------------------------ 0. validation against scipy
def section0():
    hdr("0. Validation of the 1-D not-a-knot emulation against scipy.interpolate.CubicSpline(bc_type='not-a-knot')")
    print("raw cell-centred samples, evaluation on [-h/2, L1 + h/2) incl. the two margins (scipy extrapolates with the end polynomial);")
    print("differences normalised by max|scipy| of the same derivative order; solver variants reduced | dense | sparse")
    rng = np.random.default_rng(31)
    for name, gfun in (("natural1D", g_nat), ("expsin", g_expsin)):
        for n in (32, 64, 128):
            h = 1.0 / 16 if n == 32 else (1.0 / 32 if n == 64 else 1.0 / 64)
            L1 = n * h
            x = (np.arange(n) + 0.5) * h
            xs = rng.uniform(-h / 2, L1 + h / 2, 5000)
            xs = xs[(xs / h - 0.5 >= -1) & (xs / h - 0.5 < n)]
            cs = CubicSpline(x, gfun(x)[0], bc_type="not-a-knot")
            ref = [cs(xs), cs(xs, 1), cs(xs, 2)]
            line = f"{name:10s} n={n:4d} h=1/{int(round(1 / h)):<3d} "
            for m in ("reduced", "dense", "sparse"):
                C = nak_x(gfun(x)[0][:, None], m)[:, 0]
                out = evaluate1(C, xs, h, n)
                d = [np.abs(o - r).max() / np.abs(r).max() for o, r in zip(out, ref)]
                line += f"| {m}: v={d[0]:.2e} d1={d[1]:.2e} d2={d[2]:.2e} "
            print(line)
    # 3-D (y/z constant) against 1-D, and node interpolation
    n, ny, h = 64, 32, 1.0 / 32
    x = (np.arange(n) + 0.5) * h
    f3 = np.broadcast_to(g_expsin(x)[0][:, None, None], (n, ny, ny)).copy()
    pts = ladder_points(n * h)
    for yz in ("fft", "cyc"):
        c3 = prefilter_np(f3, yz, "reduced")
        v, g, H = evaluate3(c3, pts, h, n, False)
        C1 = nak_x(g_expsin(x)[0][:, None], "reduced")[:, 0]
        v1, g1, H1 = evaluate1(C1, pts[:, 0], h, n)
        print(f"3-D (y/z constant, yz={yz}) vs 1-D: value {np.abs(v - v1).max() / np.abs(v1).max():.2e} "
              f"d/dx {np.abs(g[:, 0] - g1).max() / np.abs(g1).max():.2e} d2/dx2 {np.abs(H[:, 0] - H1).max() / np.abs(H1).max():.2e} "
              f"| max |d/dy|,|d/dz| {np.abs(g[:, 1:]).max():.2e} max |other Hessian| {np.abs(H[:, 1:]).max():.2e}")
    rng = np.random.default_rng(5)
    f = rng.standard_normal((32, 16, 16))
    h = 1.0 / 16
    gx, gy, gz = np.meshgrid((np.arange(32) + .5) * h, (np.arange(16) + .5) * h, (np.arange(16) + .5) * h, indexing="ij")
    nodes = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    for yz in ("fft", "cyc"):
        for xs_ in ("reduced", "dense", "sparse"):
            c3 = prefilter_np(f, yz, xs_)
            v = evaluate3(c3, nodes, h, 32, False)[0]
            print(f"node interpolation (random 32x16x16 samples, yz={yz}, x={xs_}): max|s(x_ijk)-f_ijk|/max|f| = "
                  f"{np.abs(v - f.ravel()).max() / np.abs(f).max():.2e}   (gate 1e-12)")


# ------------------------------------------------------------------ 1. 1-D natural field and random 3-mode fields
def ladder1d(gfun, L1, cells, pts_x):
    out = []
    for cpu in cells:
        n = int(round(L1 * cpu))
        h = 1.0 / cpu
        x = (np.arange(n) + 0.5) * h
        C = nak_x(gfun(x)[0][:, None], "reduced")[:, 0]
        v, d1, d2 = evaluate1(C, pts_x, h, n)
        e = gfun(pts_x)
        out.append([np.abs(v - e[0]).max(), np.abs(d1 - e[1]).max(), np.abs(d2 - e[2]).max()])
    return np.array(out)            # (levels, 3)


def pts1d(L1, seed=777):
    rng = np.random.default_rng(seed)
    return np.clip(np.concatenate([rng.uniform(0, L1, 10000), rng.uniform(0, COARSE, 2000),
                                   rng.uniform(L1 - COARSE, L1, 2000)]), 0.0, L1)


def section_a():
    hdr("(a) reproduction of the risk numbers: 1-D natural field, max norms, near-face points (1e4 random + 2000 per face, all in [0,L1])")
    print("field f(x) = sin(2 pi 1.1 x + 0.3) + 0.5 cos(2 pi 0.6 x); ladder 16/32/64 cells per unit length (+128 for the extra pair)")
    print("gates value/grad/Hessian = 3.8 / 2.8 / 1.8; order = log2(e_h / e_{h/2}); min over the pairs 16->32 and 32->64")
    for L1 in (1.0, 2.0):
        e = ladder1d(g_nat, L1, (16, 32, 64, 128), pts1d(L1))
        print(f"L1={L1:g}: max errors  (value, grad, Hessian) per level:")
        for cpu, row in zip((16, 32, 64, 128), e):
            print(f"   {cpu:4d} cells/unit  {row[0]:.4e} {row[1]:.4e} {row[2]:.4e}")
        o = [np.log2(e[:-1, k] / e[1:, k]) for k in range(3)]
        mn = [min(o[k][0], o[k][1]) for k in range(3)]
        print(f"   orders 16->32  : {o[0][0]:.4f} {o[1][0]:.4f} {o[2][0]:.4f}")
        print(f"   orders 32->64  : {o[0][1]:.4f} {o[1][1]:.4f} {o[2][1]:.4f}")
        print(f"   orders 64->128 : {o[0][2]:.4f} {o[1][2]:.4f} {o[2][2]:.4f}")
        print(f"   MIN over 16->32,32->64 (value/grad/Hess): {mn[0]:.4f} / {mn[1]:.4f} / {mn[2]:.4f}"
              f"   vs gates 3.8/2.8/1.8: " + " ".join("PASS" if m >= g else "FAIL" for m, g in zip(mn, (3.8, 2.8, 1.8))))
    hdr("(a) 300 random 3-mode fields: f = sum_{m=1..3} a_m sin(kappa_m x + phi_m), kappa_m/2pi ~ U(0.2,0.6), phi_m ~ U(0,2pi)")
    print("the amplitude distribution of the prior review is not recorded: four choices are run (first = the baseline used in the summary); RNG seed 20261024")
    print("counts (of 300) of fields failing the gates 3.8/2.8/1.8 (value/grad/Hess, max norm): min over 16->32,32->64 | the same with gate+0.05 | 64->128 alone")
    for amp in ("U(0.5,1)", "U(0,1)", "1", "N(0,1)"):
        for L1 in (1.0, 2.0):
            r = random_counts(L1, amp)
            print(f"amp {amp:9s} L1={L1:g}: {r[0][0]}/{r[0][1]}/{r[0][2]} | {r[1][0]}/{r[1][1]}/{r[1][2]} | {r[2][0]}/{r[2][1]}/{r[2][2]}"
                  f"   (smallest error at 128 cells/unit over the fields: {r[3][0]:.2e} {r[3][1]:.2e} {r[3][2]:.2e}, round-off check)")


def random_counts(L1, amp):
    gates = np.array([3.8, 2.8, 1.8])
    rng = np.random.default_rng(20261024)
    px = pts1d(L1)
    fail_min = np.zeros(3, int)
    fail_last = np.zeros(3, int)
    fail_min_pad = np.zeros(3, int)
    e128min = np.full(3, np.inf)
    for _ in range(300):
        kap = TWOPI * rng.uniform(0.2, 0.6, 3)
        ph = rng.uniform(0, TWOPI, 3)
        am = {"U(0.5,1)": lambda: rng.uniform(0.5, 1.0, 3), "U(0,1)": lambda: rng.uniform(0.0, 1.0, 3),
              "1": lambda: np.ones(3), "N(0,1)": lambda: rng.standard_normal(3)}[amp]()

        def gfun(x, kap=kap, ph=ph, am=am):
            return (sum(a * np.sin(k * x + p) for a, k, p in zip(am, kap, ph)),
                    sum(a * k * np.cos(k * x + p) for a, k, p in zip(am, kap, ph)),
                    sum(-a * k * k * np.sin(k * x + p) for a, k, p in zip(am, kap, ph)))
        e = ladder1d(gfun, L1, (16, 32, 64, 128), px)
        o = np.log2(e[:-1] / e[1:])                    # (3 pairs, 3 quantities)
        mn = np.minimum(o[0], o[1])
        fail_min += (mn < gates)
        fail_min_pad += (mn < gates + MARGIN_PAD)
        fail_last += (o[2] < gates)
        e128min = np.minimum(e128min, e[3])
    return fail_min, fail_min_pad, fail_last, e128min


# ------------------------------------------------------------------ (b) selection
BOXES = [(32, 16, 16), (64, 32, 32), (128, 64, 64)]


def eval_candidate(gfun, a, yz="fft", x="reduced"):
    L1 = 2.0
    pts = ladder_points(L1)
    ex = exact3(gfun, a, pts)
    errs, locs = [], []
    for (nx, ny, nz) in BOXES:
        h = 1.0 / ny
        c = prefilter_np(samples3(gfun, a, nx, ny, nz, h), yz, x)
        nrm, ev_, eH_ = norms(evaluate3(c, pts, h, nx, False), ex)
        errs.append(nrm)
        locs.append((pts[np.argmax(np.abs(ev_)), 0], pts[np.argmax(eH_), 0]))
    return errs, locs, L1


def section_b():
    hdr("(b) threshold-A test field selection: 3-D not-a-knot ladder on boxes (32,16,16), (64,32,32), (128,64,64); L1 = 2; 14000 points")
    print(f"criterion fixed before computing: CRIT-1 max-norm value AND Hessian error at the finest level within {COARSE:g} of an x face;")
    print(f"CRIT-2 min orders over 16->32, 32->64 >= gate + {MARGIN_PAD} (gates {GATES})")
    results = []
    for name, gfun, a in CANDIDATES:
        errs, locs, L1 = eval_candidate(gfun, a)
        keys = list(GATES)
        o = {k: orders([e[k] for e in errs]) for k in keys}
        mins = {k: min(o[k]) for k in keys}
        margins = {k: mins[k] - (GATES[k] + MARGIN_PAD) for k in keys}
        dist = [min(x, L1 - x) for x in locs[-1]]
        crit1 = all(d <= COARSE + 1e-12 for d in dist)
        crit2 = all(m >= 0 for m in margins.values())
        minmargin = min(margins.values())
        print()
        print(f"candidate {name}")
        for k in keys:
            print(f"   {k:9s} errors h=1/16,1/32,1/64: {errs[0][k]:.4e} {errs[1][k]:.4e} {errs[2][k]:.4e}   "
                  f"orders {o[k][0]:.4f} {o[k][1]:.4f}  min {mins[k]:.4f}  gate+0.05 {GATES[k] + MARGIN_PAD:.2f}  margin {margins[k]:+.4f}")
        print(f"   finest-level argmax location: value at x1={locs[-1][0]:.5f} (distance to nearest face {dist[0]:.5f}), "
              f"Hessian at x1={locs[-1][1]:.5f} (distance {dist[1]:.5f})")
        print(f"   CRIT-1 {'PASS' if crit1 else 'FAIL'}   CRIT-2 {'PASS' if crit2 else 'FAIL'}   minimum margin {minmargin:+.4f}   "
              f"=> {'PASSES' if crit1 and crit2 else 'does not pass'}")
        results.append((name, gfun, a, crit1 and crit2, minmargin, errs, o, mins, locs))
    passing = [r for r in results if r[3]]
    print()
    if passing:
        rec = max(passing, key=lambda r: r[4])
        print(f"LARGEST-MARGIN PICK under CRIT-1/CRIT-2 (the original rule; NOT the recorded recommendation, see below): {rec[0]}   (minimum margin {rec[4]:+.4f})")
    else:
        rec = max(results, key=lambda r: r[4])
        print("NO CANDIDATE PASSED both criteria.  Stand-in for (c)-(f) (largest minimum margin, passing or not): "
              f"{rec[0]}   (minimum margin {rec[4]:+.4f})")
    name, gfun, a, ok, mm, errs, o, mins, locs = rec
    print(f"predicted orders of {name} (4 decimals):")
    for k in GATES:
        print(f"   {k:9s} 16->32 {o[k][0]:.4f}   32->64 {o[k][1]:.4f}   min {mins[k]:.4f}")
    print("error values per level (h = 1/16, 1/32, 1/64):")
    for k in GATES:
        print(f"   {k:9s} {errs[0][k]:.6e} {errs[1][k]:.6e} {errs[2][k]:.6e}")
    rec_e11 = select_e11(results, rec)
    return rec, rec_e11


def select_e11(results, rule_pick):
    """Recorded recommendation (owner decision E11): a passing candidate whose x-closure share of the finest-level value and
    Hessian error is >= 0.9.  Tie-breaker among those: largest minimum margin.  The shares are the (b2) diagnostic, computed after
    the CRIT-1/CRIT-2 orders were seen (disclosed)."""
    print()
    print("RECORDED RECOMMENDATION")
    print("  selected by the orchestrator per owner decision E11: the passing candidate whose x-closure share of the finest-level value and")
    print("  Hessian error is >= 0.9; C1 was the largest-margin pick under CRIT-1/CRIT-2, which did not test the share; the share diagnostic b2")
    print("  was computed after the orders were seen and this is disclosed.")
    qual = []
    for r in results:
        if not r[3]:
            continue
        e0, _, _ = eval_candidate(r[1], 0.0)
        sv = e0[-1]["val_max"] / r[5][-1]["val_max"]
        sh = e0[-1]["hess_max"] / r[5][-1]["hess_max"]
        ok = sv >= 0.9 and sh >= 0.9
        print(f"  passing candidate {r[0]:34s} x share value {sv:.3f} Hessian {sh:.3f}  -> {'qualifies' if ok else 'does not qualify'}")
        if ok:
            qual.append(r)
    pick = max(qual, key=lambda r: r[4])
    print(f"  qualifying: {', '.join(r[0].split()[0] for r in qual)}; tie-breaker (largest minimum margin) and the orchestrator's selection: {pick[0]}"
          f"   (minimum margin {pick[4]:+.4f})")
    name, gfun, a, ok, mm, errs, o, mins, locs = pick
    print(f"  predicted orders of {name} (6 significant digits), all five norms:")
    for k in GATES:
        print(f"   {k:9s} 16->32 {o[k][0]:.6g}   32->64 {o[k][1]:.6g}   min {mins[k]:.6g}   gate+0.05 {GATES[k] + MARGIN_PAD:.2f}")
    print("  error values per level (h = 1/16, 1/32, 1/64; 6 significant digits):")
    for k in GATES:
        print(f"   {k:9s} {errs[0][k]:.5e} {errs[1][k]:.5e} {errs[2][k]:.5e}")
    print(f"  finest-level argmax location: value at x1={locs[-1][0]:.5f}, Hessian at x1={locs[-1][1]:.5f}")
    return pick


def section_b2():
    hdr("(b2) POST-HOC DIAGNOSTIC (added after the CRIT-1/CRIT-2 results were seen; does not alter the selection above)")
    print("share of the x closure in the 3-D error: the same g(x1) with the y/z factor removed (a = 0), same 3-D pipeline, same points;")
    print("'x-only' errors at the finest level and the min orders of the a = 0 ladder against gate + 0.05")
    pts = ladder_points(2.0)
    for name, gfun, a in CANDIDATES:
        e_a, _, _ = eval_candidate(gfun, a)
        e_0, _, _ = eval_candidate(gfun, 0.0)
        keys = list(GATES)
        o0 = {k: orders([e[k] for e in e_0]) for k in keys}
        m0 = {k: min(o0[k]) for k in keys}
        ok0 = all(m0[k] >= GATES[k] + MARGIN_PAD for k in keys)
        print(f"{name:34s} finest val_max total {e_a[-1]['val_max']:.3e} x-only {e_0[-1]['val_max']:.3e} (x share {e_0[-1]['val_max'] / e_a[-1]['val_max']:.3f}); "
              f"hess_max total {e_a[-1]['hess_max']:.3e} x-only {e_0[-1]['hess_max']:.3e} (x share {e_0[-1]['hess_max'] / e_a[-1]['hess_max']:.3f})")
        print(f"{'':34s} x-only min orders " + " ".join(f"{k}={m0[k]:.4f}" for k in keys) + f"  -> x-only ladder {'passes' if ok0 else 'does not pass'} gate+0.05")


# ------------------------------------------------------------------ (c) x-only near-face sub-ladder
def section_c(rec):
    name, gfun, a = rec[0], rec[1], rec[2]
    hdr(f"(c) x-only near-face sub-ladder for {name}: 1-D profile g(x1), L1 = 2, 4000 points within 1/16 of the faces (2000 each), 16/32/64/128 cells/unit")
    L1 = 2.0
    rng = np.random.default_rng(909)
    px = np.clip(np.concatenate([rng.uniform(0, COARSE, 2000), rng.uniform(L1 - COARSE, L1, 2000)]), 0.0, L1)
    cells = (16, 32, 64, 128)
    errs = []
    for cpu in cells:
        n = int(round(L1 * cpu))
        h = 1.0 / cpu
        x = (np.arange(n) + 0.5) * h
        C = nak_x(gfun(x)[0][:, None], "reduced")[:, 0]
        v, d1, d2 = evaluate1(C, px, h, n)
        e = gfun(px)
        errs.append([np.abs(v - e[0]).max(), np.sqrt(np.mean((v - e[0]) ** 2)), np.abs(d1 - e[1]).max(),
                     np.sqrt(np.mean((d1 - e[1]) ** 2)), np.abs(d2 - e[2]).max()])
    errs = np.array(errs)
    labels = ["val_max", "val_rms", "grad_max", "grad_rms", "hess_max"]
    for cpu, row in zip(cells, errs):
        print(f"   {cpu:4d} cells/unit  " + " ".join(f"{lab}={v:.5e}" for lab, v in zip(labels, row)))
    o = np.log2(errs[:-1] / errs[1:])
    for i, pair in enumerate(("16->32", "32->64", "64->128")):
        print(f"   orders {pair:8s}: " + " ".join(f"{lab}={v:.6g}" for lab, v in zip(labels, o[i])))
    print("   (recorded quantity; no gate)")


# ------------------------------------------------------------------ (d) polynomial reproduction
def section_d():
    hdr("(d) reproduction of p(x1) = (x1 - L1/2)^2 and of a full cubic through the prefilter (y/z constant)")
    print("cubic = 1 + 0.5 (x1 - L1/2) - 0.7 (x1 - L1/2)^2 + 0.3 (x1 - L1/2)^3; boxes (64,16,16) and (128,16,16), h = 1/16, L1 = 4, 8")
    print("points: 10000 random in [0,L1]x[0,1)^2 + 2000 per face within 1/16 + margin points at fixed x/h in [-h/2,0) and [L1, L1+h/2)")
    print("normwise relative: max |err| / max |exact| for value; Euclidean per point for the gradient; component max for the Hessian")
    for nx in (64, 128):
        ny = nz = 16
        h = 1.0 / 16
        L1 = nx * h
        rng = np.random.default_rng(1234)
        pts = np.vstack([np.column_stack([rng.uniform(0, L1, 10000), rng.uniform(0, 1, 10000), rng.uniform(0, 1, 10000)]),
                         np.column_stack([rng.uniform(0, COARSE, 2000), rng.uniform(0, 1, 2000), rng.uniform(0, 1, 2000)]),
                         np.column_stack([rng.uniform(L1 - COARSE, L1, 2000), rng.uniform(0, 1, 2000), rng.uniform(0, 1, 2000)]),
                         margin_points(L1, h)])
        s = pts[:, 0] - L1 / 2
        polys = {
            "p=(x-L1/2)^2": (lambda t: t**2, lambda t: 2 * t, lambda t: 2 + 0 * t),
            "cubic": (lambda t: 1 + 0.5 * t - 0.7 * t**2 + 0.3 * t**3, lambda t: 0.5 - 1.4 * t + 0.9 * t**2, lambda t: -1.4 + 1.8 * t),
        }
        x = (np.arange(nx) + 0.5) * h - L1 / 2
        for pname, (p0, p1, p2) in polys.items():
            ex_v = p0(s)
            ex_g = np.column_stack([p1(s), 0 * s, 0 * s])
            ex_H = np.column_stack([p2(s)] + [0 * s] * 5)
            f = np.broadcast_to(p0(x)[:, None, None], (nx, ny, nz)).copy()
            for yz in ("fft", "cyc"):
                for xm in ("reduced", "dense"):
                    c = prefilter_np(f, yz, xm)
                    v, g, H = evaluate3(c, pts, h, nx, False)
                    rv = np.abs(v - ex_v).max() / np.abs(ex_v).max()
                    rg = np.linalg.norm(g - ex_g, axis=1).max() / np.linalg.norm(ex_g, axis=1).max()
                    rH = np.abs(H - ex_H).max() / np.abs(ex_H).max()
                    rHxx = np.abs(H[:, 0] - ex_H[:, 0]).max() / np.abs(ex_H[:, 0]).max()
                    rHoff = np.abs(H[:, 1:]).max() / np.abs(ex_H[:, 0]).max()
                    print(f"nx={nx:4d} {pname:13s} yz={yz:3s} x={xm:7s}: rel value={rv:.3e} gradient={rg:.3e} Hessian={rH:.3e}"
                          f" [xx only {rHxx:.3e}; spurious y/z/cross components {rHoff:.3e}]  (gate 1e-12)")


# ------------------------------------------------------------------ (e) partition of unity
def section_e():
    hdr("(e) partition of unity: constant 3.0, boxes (32,16,16), (64,32,32), (128,64,64); max abs error of value and of the gradient")
    print("point sets: IN = ladder point set inside [0,L1] (14000, incl. near-face); MARGIN = fixed x/h offsets r=(j+0.5)/64, j=0..31, at x=-h r and L1+h r")
    for yz in ("fft", "cyc"):
        for xm in ("dense", "reduced", "sparse"):
            for (nx, ny, nz) in BOXES:
                h = 1.0 / ny
                L1 = nx * h
                f = np.full((nx, ny, nz), 3.0)
                c = prefilter_np(f, yz, xm)
                row = f"yz={yz:3s} x={xm:7s} h=1/{ny:<3d} "
                for lab, pts in (("IN", ladder_points(L1)), ("MARGIN", margin_points(L1, h))):
                    v, g, H = evaluate3(c, pts, h, nx, False)
                    row += f"| {lab}: value {np.abs(v - 3.0).max():.2e} gradient {np.abs(g).max():.2e} Hessian {np.abs(H).max():.2e} "
                print(row)
    print()
    print("margin points fixed in PHYSICAL space leave the domain on finer grids: x_m = -1/128 (i.e. -h16/8)")
    for ny in (16, 32, 64, 128, 256):
        h = 1.0 / ny
        r = (-1.0 / 128) / h
        print(f"   h=1/{ny:<3d}: x_m/h = {r:+.4f}   inside the evaluation domain [-h/2, L1+h/2)? {'yes' if r >= -0.5 else 'NO'}")


# ------------------------------------------------------------------ (f) CPU vs GPU surrogate
def section_f(recs):
    hdr("(f) CPU-vs-GPU surrogate: y/z FFT (reference, 'GPU-like') vs cyclic tridiagonal ('CPU-mirror-like'); same reduced not-a-knot x solve")
    print("full pipeline = prefilter + evaluation; differences relative to the max of the reference (gradient Euclidean, Hessian component max)")
    print("point set: ladder points inside [0,L1] + fixed-x/h margin points; gate (SF-41 A): <= 1e-13")
    print("noise gain of the x line solve: rms(C_k) / rms(e) for unit white noise e on the samples (reduced = dense), by coefficient index k")
    rng = np.random.default_rng(3)
    for n in (32, 128, 1664):
        e = rng.standard_normal((n, 2048))
        C = nak_x(e, "reduced")
        gain = np.sqrt((C**2).mean(axis=1)) / np.sqrt((e**2).mean())
        print(f"   n={n:5d}: k=-2..3: " + " ".join(f"{g:.2f}" for g in gain[:6]) + f" | k=n-4..n+1: " + " ".join(f"{g:.2f}" for g in gain[-6:])
              + f" | middle {gain[n // 2]:.3f}")
    fields = [(f"field {r[0]}", r[1], r[2]) for r in recs] + [("stress: Gaussian field lambda/h = 8, seed 6001", None, None)]
    for label, gfun, a in fields:
        print(f"-- {label}")
        dco = []
        for (nx, ny, nz) in BOXES:
            h = 1.0 / ny
            L1 = nx * h
            if gfun is not None:
                f = samples3(gfun, a, nx, ny, nz, h)
            else:
                f = gaussian_field(nx, ny, nz, h, 8 * h, 6001)
            pts = np.vstack([ladder_points(L1), margin_points(L1, h)])
            cg = prefilter_np(f, "fft", "reduced")
            cc = prefilter_np(f, "cyc", "reduced")
            eg = evaluate3(cg, pts, h, nx, False)
            ec = evaluate3(cc, pts, h, nx, False)
            dcoef = np.abs(cc - cg).max() / np.abs(cg).max()
            dco.append((ny, dcoef))
            xp = pts[:, 0]
            regions = {"interior(0.5<=x<=L1-0.5)": (xp >= 0.5) & (xp <= L1 - 0.5),
                       "near-face(<=1/16)": ((xp >= 0) & (xp <= COARSE)) | ((xp >= L1 - COARSE) & (xp <= L1)),
                       "margin": (xp < 0) | (xp > L1), "ALL": np.ones(len(xp), bool)}
            sv, sg, sH = np.abs(eg[0]).max(), np.linalg.norm(eg[1], axis=1).max(), np.abs(eg[2]).max()
            print(f"   h=1/{ny:<3d} NP (coefficients {dcoef:.2e}): " + " | ".join(
                f"{rn}: v {np.abs(ec[0][m] - eg[0][m]).max() / sv:.1e} g {np.linalg.norm(ec[1][m] - eg[1][m], axis=1).max() / sg:.1e} "
                f"H {np.abs(ec[2][m] - eg[2][m]).max() / sH:.1e}" for rn, m in regions.items()))
            # control: the periodic SF-28 pair (FFT vs cyclic tridiagonal), same samples
            pp = prefilter_periodic_fft(f)
            pc = prefilter_periodic_cyc(f)
            ptsin = ladder_points(L1)
            ptsin = ptsin[(ptsin[:, 0] >= 0.5) & (ptsin[:, 0] <= L1 - 0.5)]   # periodic spline of non-periodic data blows up at the faces
            ep, epc = evaluate3(pp, ptsin, h, nx, True), evaluate3(pc, ptsin, h, nx, True)
            qv = np.abs(epc[0] - ep[0]).max() / np.abs(ep[0]).max()
            qg = np.linalg.norm(epc[1] - ep[1], axis=1).max() / np.linalg.norm(ep[1], axis=1).max()
            qH = np.abs(epc[2] - ep[2]).max() / np.abs(ep[2]).max()
            print(f"            periodic control (SF-28 pair, FFT vs cyclic tridiagonal, interior points 0.5<=x<=L1-0.5): value {qv:.2e} gradient {qg:.2e} Hessian {qH:.2e}")
        print("   coefficient difference (yz cyc vs yz fft; same x solve) / max|c|: " + "  ".join(f"h=1/{n}: {d:.3e}" for n, d in dco)
              + "   (host-style vs device-style evaluation of the same coefficients is the identical numpy code; not separable here)")


def main():
    print("# SF-41 pre-activation, threshold A design emulations (numpy; NOT SF-41 evidence)")
    section0()
    section_a()
    rec, rec_e11 = section_b()
    section_b2()
    section_c(rec)
    section_c(rec_e11)
    section_d()
    section_e()
    section_f([rec, rec_e11])


if __name__ == "__main__":
    main()
