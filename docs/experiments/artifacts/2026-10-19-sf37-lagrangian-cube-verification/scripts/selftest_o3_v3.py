#!/usr/bin/env python3
"""Self-test of o3.py and digest_v3.py (exit 0 iff every check passes; one PASS/FAIL line per check).

  selftest_o3_v3.py [--keep DIR]

(a) the o3 c-stencil equals label_diagnostics.cross (random labels, <= 1e-15 abs; chunked planes included);
(b) f_half, N_half, d_max, location equal label_diagnostics.concentration on the same arrays; chunk-size invariance;
(c) top-q marking: ceil(q n) in exact arithmetic, ties broken by flat index;
(d) e_v^(q): 0 for exact k = 1 labels; small and second order for an analytic smooth label pair with the exact c as vD;
(e) O3a / O3b verdict rules on synthetic series (accept, reject, unresolved, shrinking, not shrinking, not_evaluable);
(f) O3c geometry on a planted case (periodic wrap, unwrapped landings, renormalization, pooled, x1 = 1 structurally empty);
(g) V3a interpolation: exact (<= 1e-12) for a bicubic-representable periodic function, 4th order for a smooth trig function,
    unwrapped landings, D5 proxy;
(h) criterion C by orders_digest.py in place (real SF-35 seed-3001 summaries, plus altered copies) and verdict parsing;
(i) V3b: D_P, SE_D and the D6 decomposition on synthetic run pairs with known answers; classification and noise-limited rules;
(j) end-to-end `discovered` runs of o3.py and digest_v3.py on a tiny synthetic campaign tree; `full` design on it (nothing passes);
(k) py_compile of the scripts and absence of numpy-2-removed API names.
Never reads seeds 3101-3103 (synthetic data only).  Python 3 + numpy + scipy only.
"""
import argparse
import csv
import importlib.util
import json
import math
import os
import py_compile
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sf37lib as L  # noqa: E402
import o3 as O3  # noqa: E402
import digest_v3 as DV  # noqa: E402
import selftest_sf37 as S37  # noqa: E402  (synthetic run writer and independent moment loops)

FAILS = []
RESULTS = []
SF35 = os.path.join("docs", "experiments", "artifacts", "2026-10-07-sf35-label-transport")


def check(name, cond, detail=""):
    ok = bool(cond)
    RESULTS.append((name, ok))
    if not ok:
        FAILS.append(name)
    print("%s  %s%s" % ("PASS" if ok else "FAIL", name, ("  [" + detail + "]") if detail else ""))
    return ok


def find_repo():
    d = HERE
    for _ in range(8):
        if os.path.isfile(os.path.join(d, SF35, "scripts", "label_diagnostics.py")):
            return d
        d = os.path.dirname(d)
    return None


REPO = find_repo()


def load_ld():
    spec = importlib.util.spec_from_file_location("label_diagnostics", os.path.join(REPO, SF35, "scripts", "label_diagnostics.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def slurp(p):
    with open(p) as f:
        return f.read()


def csv_rows(p):
    lines = [ln for ln in slurp(p).splitlines() if not ln.startswith("#")]
    rd = list(csv.reader(lines))
    hdr = rd[0]
    return hdr, [dict(zip(hdr, r)) for r in rd[1:]]


# --------------------------------------------------------------------------------------------------------------
# synthetic labels
# --------------------------------------------------------------------------------------------------------------
def random_labels(n, seed, amp=0.05):
    rng = np.random.default_rng(seed)
    x = np.arange(n, dtype=np.float64) / n
    u1 = amp * rng.standard_normal((n + 1, n, n))
    u2 = amp * rng.standard_normal((n + 1, n, n))
    return x[None, :, None] + u1, x[None, None, :] + u2


def analytic_labels(N, amp=1.0):
    """psi1 = x2 + U1, psi2 = x3 + U2 with smooth periodic U (x1-dependent) and the exact c = grad psi1 x grad psi2."""
    x = np.arange(N + 1, dtype=np.float64) / N
    y = np.arange(N, dtype=np.float64) / N
    X, Y, Z = np.meshgrid(x, y, y, indexing="ij")
    a, b, tp = 0.08 * amp, 0.06 * amp, 2.0 * np.pi
    U1 = a * np.sin(tp * Y) * np.cos(tp * Z) * (1 + X)
    U1x = a * np.sin(tp * Y) * np.cos(tp * Z)
    U1y = a * tp * np.cos(tp * Y) * np.cos(tp * Z) * (1 + X)
    U1z = -a * tp * np.sin(tp * Y) * np.sin(tp * Z) * (1 + X)
    U2 = b * np.cos(tp * (Y + Z)) * X * (1 - X)
    U2x = b * np.cos(tp * (Y + Z)) * (1 - 2 * X)
    U2y = -b * tp * np.sin(tp * (Y + Z)) * X * (1 - X)
    U2z = U2y
    g1 = (U1x, 1 + U1y, U1z)
    g2 = (U2x, U2y, 1 + U2z)
    c = (g1[1] * g2[2] - g1[2] * g2[1], g1[2] * g2[0] - g1[0] * g2[2], g1[0] * g2[1] - g1[1] * g2[0])
    return Y + U1, Z + U2, c


def planar_labels(N):
    """x1-independent labels psi1 = y + a sin(2 pi y) cos(2 pi z), psi2 = z + b cos(2 pi (y + z)) on the (N+1, N, N) lattice."""
    y = np.arange(N, dtype=np.float64) / N
    Y, Z = np.meshgrid(y, y, indexing="ij")
    p1 = Y + 0.08 * np.sin(2 * np.pi * Y) * np.cos(2 * np.pi * Z)
    p2 = Z + 0.06 * np.cos(2 * np.pi * (Y + Z))
    return np.broadcast_to(p1, (N + 1, N, N)).copy(), np.broadcast_to(p2, (N + 1, N, N)).copy()


def planar_value(y, z):
    ym, zm = np.mod(y, 1.0), np.mod(z, 1.0)
    return (y + 0.08 * np.sin(2 * np.pi * ym) * np.cos(2 * np.pi * zm), z + 0.06 * np.cos(2 * np.pi * (ym + zm)))


# --------------------------------------------------------------------------------------------------------------
def test_stencil_and_concentration(ld):
    n = 8
    p1, p2 = random_labels(n, 11)
    ref = ld.cross(p1, p2, n)
    mine = O3.cross_planes(p1, p2, n, 1, n - 1, 1)
    d = max(float(np.abs(a - b).max()) for a, b in zip(ref, mine))
    check("(a) c-stencil: cross_planes (all interior planes) equals label_diagnostics.cross", d <= 1e-15, "max abs diff %.2e" % d)
    part = O3.cross_planes(p1, p2, n, 2, 3, 2)  # planes 2, 4, 6
    d2 = max(float(np.abs(a[1::2][:3] - b).max()) for a, b in zip(ref, part))
    check("(a) c-stencil: strided chunk (planes 2,4,6) equals the corresponding planes of label_diagnostics.cross", d2 <= 1e-15,
          "max abs diff %.2e" % d2)
    # (b) concentration equality on a coarse/fine pair
    nc = 8
    N = 2 * nc
    c1, c2 = random_labels(nc, 21)
    f1, f2 = random_labels(N, 22)
    conc = ld.concentration(c1, c2, f1, f2, nc)
    err2, v2, _, _ = O3.pair_arrays(c1, c2, f1, f2, N)
    m = O3.pair_metrics(err2, v2)
    check("(b) f_half equals label_diagnostics.concentration frac exactly (and N_half)", m["f_half"] == conc["frac"] and m["N_half"] == conc["nhalf"],
          "frac %.8g vs %.8g, N_half %d vs %d" % (m["f_half"], conc["frac"], m["N_half"], conc["nhalf"]))
    rel = lambda a, b: abs(a - b) / abs(b)
    check("(b) RMS|c_N|, d_max, RMS d and the d_max location equal label_diagnostics",
          rel(m["rms_c_N"], conc["vrms"]) <= 1e-15 and rel(m["d_max"], conc["emax"]) <= 1e-15 and rel(m["rms_d_all"], conc["rms"]) <= 1e-15
          and m["d_max_loc"] == conc["loc"] and m["n_common"] == conc["nvert"])
    err2b, v2b, _, _ = O3.pair_arrays(c1, c2, f1, f2, N, chunk=1)
    check("(b) chunked evaluation (1 coarse plane per chunk) is bitwise identical to the default", np.array_equal(err2, err2b) and np.array_equal(v2, v2b))
    d1 = np.sqrt(err2).ravel() / math.sqrt(v2.mean())
    check("(b) fraction with d > 0.1 matches a direct count", abs(m["frac_d_gt_0.1"] - float(np.mean(d1 > 0.1))) == 0.0)


def test_marking():
    check("(c) ceil(q n) exact: n = 1000 -> 1, 1001 -> 2, 258048 -> 259, 2080768 -> 2081, 16711680 -> 16712",
          [O3.ceil_q(n) for n in (1000, 1001, 258048, 2080768, 16711680)] == [1, 2, 259, 2081, 16712])
    d = np.ones(5000)
    d[[7, 3, 9, 100, 4, 8]] = 2.0
    mk = O3.mark_top_q(d)
    check("(c) ties broken by flat index: six tied maxima, ceil(q n) = 5 -> indices 3,4,7,8,9 in rank order", mk.tolist() == [3, 4, 7, 8, 9], str(mk.tolist()))
    d2 = np.zeros(2500)
    mk2 = O3.mark_top_q(d2)
    check("(c) all-equal d, n = 2500 -> ceil(2.5) = 3 first indices", mk2.tolist() == [0, 1, 2])
    d3 = np.arange(3000, dtype=np.float64)
    d3[[10, 2999]] = 5000.0
    check("(c) distinct maxima ranked by d: largest 3 of 3000", O3.mark_top_q(d3).tolist() == [10, 2999, 2998])


def test_ev(tmp):
    # exact k = 1 labels
    n, N = 8, 16
    c1, c2 = (np.broadcast_to((np.arange(n, dtype=float) / n)[None, :, None], (n + 1, n, n)).copy(),
              np.broadcast_to((np.arange(n, dtype=float) / n)[None, None, :], (n + 1, n, n)).copy())
    f1 = np.broadcast_to((np.arange(N, dtype=float) / N)[None, :, None], (N + 1, N, N)).copy()
    f2 = np.broadcast_to((np.arange(N, dtype=float) / N)[None, None, :], (N + 1, N, N)).copy()
    vD = (np.ones((N + 1, N, N)), np.zeros((N + 1, N, N)), np.zeros((N + 1, N, N)))
    err2, v2, dv2, vv2 = O3.pair_arrays(c1, c2, f1, f2, N, vD)
    m = O3.pair_metrics(err2, v2, dv2, vv2)
    check("(d) e_v^(q) = 0 for exact k = 1 labels (psi1 = x2, psi2 = x3, vD = e1)", m["e_v_q"] == 0.0 and m["e_v_all"] == 0.0 and m["v_rms_common_vD"] == 1.0)
    # analytic smooth pair
    vals = {}
    for N in (16, 32, 64):
        cp1, cp2, _ = analytic_labels(N // 2)
        fp1, fp2, cex = analytic_labels(N)
        err2, v2, dv2, vv2 = O3.pair_arrays(cp1, cp2, fp1, fp2, N, cex)
        mm = O3.pair_metrics(err2, v2, dv2, vv2)
        vals[N] = mm["e_v_q"]
    o1, o2 = L.order_pair(vals[16], vals[32]), L.order_pair(vals[32], vals[64])
    check("(d) analytic labels with vD = exact c: e_v^(q) small and second order (orders 16->32, 32->64 within [1.9, 2.1])",
          vals[64] < 1e-3 and 1.9 <= o1 <= 2.1 and 1.9 <= o2 <= 2.1, "e_v^(q) %s, orders %.3f %.3f" % (
              ", ".join("%.3e" % vals[k] for k in (16, 32, 64)), o1, o2))
    va, _, _, _ = O3.verdict_o3a((16, 32, 64), vals)
    check("(d) the O3a rule accepts that series", va == "accept")


def test_verdicts():
    ch = (128, 256, 512)
    V = lambda a, b, c: O3.verdict_o3a(ch, {128: a, 256: b, 512: c})[0]
    check("(e) O3a accept: [1, .25, .0625] (order 2)", V(1.0, 0.25, 0.0625) == "accept")
    check("(e) O3a accept at the order-1.8 boundary", V(1.0, 0.5, 0.5 / 2 ** 1.8 * 0.9999999) == "accept")
    check("(e) O3a unresolved just below order 1.8 (not flat)", V(1.0, 0.5, 0.5 / 2 ** 1.79) == "unresolved")
    check("(e) O3a reject: change 8 % between 256 and 512, e > 1e-12", V(1.0, 0.5, 0.46) == "reject")
    check("(e) O3a reject also when not decreasing (e(512) > e(256) by 5 %)", V(1.0, 0.5, 0.525) == "reject")
    check("(e) O3a unresolved: slow decrease (order 1)", V(1.0, 0.6, 0.3) == "unresolved")
    check("(e) O3a unresolved: flat change < 10 % but e(512) <= 1e-12 is not a reject", V(1e-12, 5e-13, 4.8e-13) == "unresolved")
    check("(e) O3a unresolved: non-monotone with a large last order", V(1.0, 0.1, 0.5) == "unresolved")
    check("(e) O3a unresolved: first refinement increases, last order large", V(0.1, 0.5, 0.01) == "unresolved")
    check("(e) O3a not_evaluable: missing grid", O3.verdict_o3a(ch, {128: 1.0, 256: 0.25})[0] == "not_evaluable")
    check("(e) O3a not_evaluable: NaN value (never unresolved)", V(1.0, float("nan"), 0.1) == "not_evaluable")
    check("(e) O3a not_evaluable: fewer than two grids", O3.verdict_o3a((512,), {512: 1.0})[0] == "not_evaluable")
    B = lambda a, b, c: O3.verdict_o3b(ch, {128: a, 256: b, 512: c})[0]
    check("(e) O3b shrinking: strictly decreasing f_half", B(0.1, 0.05, 0.02) == "shrinking")
    check("(e) O3b not shrinking: a non-decrease on the last refinement", B(0.1, 0.05, 0.06) == "not shrinking")
    check("(e) O3b not shrinking: equal values", B(0.1, 0.1, 0.05) == "not shrinking")
    check("(e) O3b not_evaluable: missing / NaN", B(0.1, float("nan"), 0.02) == "not_evaluable" and O3.verdict_o3b(ch, {128: 0.1})[0] == "not_evaluable")


def write_planted_run(d, x23, w, planes):
    os.makedirs(d, exist_ok=True)
    np.save(os.path.join(d, "planes_x23.npy"), x23)
    np.save(os.path.join(d, "weights.npy"), w)
    with open(os.path.join(d, "track.json"), "w") as f:
        json.dump({"planes": list(planes), "times": [1.0]}, f)


def test_o3c(tmp):
    N = 32
    nc = N // 2
    planes = [0.25, 0.5, 1.0]
    nan = float("nan")
    # particles: x23[p, plane, (x2, x3)]
    x23 = np.full((6, 3, 2), np.nan)
    x23[0, 0] = (0.99, 0.30)        # near mark (jc=4: (0, 5) at (0, 0.3125)); wraps in x2
    x23[1, 0] = (0.50, 0.50)        # far
    x23[2, 0] = (0.06, 0.375)       # exactly on the boundary dx3 = 2/N (inclusive)
    x23[3, 0] = (0.07, 0.30)        # dx2 = 0.07 > 2/N = 0.0625
    x23[4, 0] = (1.01, 2.30)        # unwrapped landing, near after reduction mod 1
    # particle 5 does not contribute at plane 0.25
    x23[:, 1] = np.array([[0.30, 0.10], [0.50, 0.50], [0.10, 0.10], [0.80, 0.80], [0.60, 0.40], [0.51, 0.49]])
    x23[:, 2] = x23[:, 1]
    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    d = os.path.join(tmp, "planted")
    write_planted_run(d, x23, w, planes)
    run = L.RunData(d)
    marked = np.array([np.ravel_multi_index((3, 0, 5), (nc - 1, nc, nc)), np.ravel_multi_index((7, 8, 8), (nc - 1, nc, nc))])
    rows = {str(r["x1"]): r for r in O3.o3c_fractions(run, marked, N)}
    # plane 0.25: contributors 0..4 (weights 1..5, W = 15); hits: 0, 2, 4 -> weight 1 + 3 + 5 = 9
    f25 = rows["0.25"]["frac"]
    check("(f) O3c plane 0.25: periodic wrap, boundary-inclusive, unwrapped landing, renormalized over contributors (9/15)",
          abs(f25 - 9.0 / 15.0) <= 1e-15 and rows["0.25"]["n"] == 5 and rows["0.25"]["n_hit"] == 3, "%.6f" % f25)
    # plane 0.5 (jc = 8): mark (8, 8) at (0.5, 0.5): particle 1 (0.5,0.5) and particle 5 (0.51,0.49) are within 0.0625; weights 2 + 6 of 21
    f5 = rows["0.5"]["frac"]
    check("(f) O3c plane 0.5: marked vertex hits particles 1 and 5 (8/21)", abs(f5 - 8.0 / 21.0) <= 1e-15 and rows["0.5"]["n_hit"] == 2, "%.6f" % f5)
    # pooled: contributing at both planes = particles 0..4 (W = 15); marked at either: 0, 2, 4 (plane 0.25) and 1 (plane 0.5) -> 1+3+5+2 = 11
    fp = rows["pooled"]["frac"]
    check("(f) O3c pooled: particles contributing at both planes, marked at either (11/15)", abs(fp - 11.0 / 15.0) <= 1e-15 and rows["pooled"]["n"] == 5, "%.6f" % fp)
    check("(f) O3c x1 = 1 is reported structurally empty", rows["1.0"]["status"] == "structurally_empty")
    B = O3.marks_on_plane(marked, N, 4)
    check("(f) marks_on_plane extracts only the marks of that coarse plane", B.sum() == 1 and B[0, 5])
    # no marks on the plane -> fraction 0, status ok
    rows2 = {str(r["x1"]): r for r in O3.o3c_fractions(run, np.array([], dtype=np.int64), N)}
    check("(f) a plane without marks has weighted fraction 0", rows2["0.25"]["frac"] == 0.0 and rows2["0.25"]["status"] == "ok")


def b3(t):
    t = np.abs(t)
    return np.where(t < 1, (4 - 6 * t ** 2 + 3 * t ** 3) / 6.0, np.where(t < 2, (2 - t) ** 3 / 6.0, 0.0))


def spline_eval(c, u2, u3):
    N = c.shape[0]
    f2, f3 = np.floor(u2).astype(np.int64), np.floor(u3).astype(np.int64)
    val = np.zeros_like(u2)
    for i in range(-1, 3):
        for k in range(-1, 3):
            val += b3(u2 - (f2 + i)) * b3(u3 - (f3 + k)) * c[(f2 + i) % N, (f3 + k) % N]
    return val


def test_interp():
    rng = np.random.default_rng(5)
    N = 12
    c = rng.standard_normal((N, N))
    tmp = (np.roll(c, 1, 0) + 4 * c + np.roll(c, -1, 0)) / 6.0
    U = (np.roll(tmp, 1, 1) + 4 * tmp + np.roll(tmp, -1, 1)) / 6.0
    x2 = rng.uniform(-2.0, 3.0, 200)
    x3 = rng.uniform(-2.0, 3.0, 200)
    truth = spline_eval(c, np.mod(x2, 1.0) * N, np.mod(x3, 1.0) * N)
    got = DV.interp_U(U, N, x2, x3)
    check("(g) V3a interpolation exact (<= 1e-12) for a bicubic-representable periodic function (incl. unwrapped points)",
          float(np.abs(got - truth).max()) <= 1e-12, "max abs diff %.2e" % float(np.abs(got - truth).max()))
    # 4th order on trig function
    errs = {}
    pts2, pts3 = rng.uniform(0, 1, 400), rng.uniform(0, 1, 400)
    f = lambda y, z: np.sin(2 * np.pi * y) * np.cos(4 * np.pi * z) + 0.3 * np.cos(2 * np.pi * (y + z))
    for Nn in (16, 32, 64):
        g = np.arange(Nn) / Nn
        Y, Z = np.meshgrid(g, g, indexing="ij")
        errs[Nn] = float(np.abs(DV.interp_U(f(Y, Z), Nn, pts2, pts3) - f(pts2, pts3)).max())
    o1, o2 = math.log2(errs[16] / errs[32]), math.log2(errs[32] / errs[64])
    check("(g) V3a interpolation 4th-order accurate for a smooth trig function (observed orders 16->32, 32->64 >= 3.6)", o1 >= 3.6 and o2 >= 3.6,
          "errors %s, orders %.2f %.2f" % (", ".join("%.2e" % errs[k] for k in (16, 32, 64)), o1, o2))
    # label_error: unwrapped landings and psi_p = exact label
    Nn = 32
    p1, p2 = planar_labels(Nn)
    yu = rng.uniform(-1.5, 2.5, 300)
    zu = rng.uniform(-1.5, 2.5, 300)
    pp = np.stack(planar_value(yu, zu), axis=1)
    e1, e2, err = DV.label_error(p1[5], p2[5], Nn, yu, zu, pp)
    e1w, e2w, errw = DV.label_error(p1[5], p2[5], Nn, np.mod(yu, 1.0), np.mod(zu, 1.0), np.stack(planar_value(np.mod(yu, 1.0), np.mod(zu, 1.0)), axis=1))
    check("(g) label_error: unwrapped landing equals the reduced one (psi_p shifted accordingly) and is a small interpolation error",
          float(np.abs(err - errw).max()) <= 1e-12 and float(err.max()) < 5e-4, "max err %.2e, wrap diff %.2e" % (float(err.max()), float(np.abs(err - errw).max())))
    check("(g) label_error: |e| is the Euclidean norm of (e1, e2)", float(np.abs(err - np.sqrt(e1 ** 2 + e2 ** 2)).max()) == 0.0)
    # D5 proxy decreases 4th order with N
    px = {}
    for Nn in (8, 16, 32):
        a1, a2 = planar_labels(Nn)
        b1, b2 = planar_labels(2 * Nn)
        px[Nn] = DV.interp_proxy((a1, a2), (b1, b2), Nn, 0, 0)
    check("(g) D5 interpolation proxy finite, positive and decreasing with N", all(np.isfinite(list(px.values()))) and px[8] > px[16] > px[32] > 0,
          ", ".join("%d: %.2e" % (k, px[k]) for k in px))
    # classification helpers
    check("(g) V3a concordance rule: decreasing RMS 64>128>256 concordant; a non-decrease on 128->256 not concordant; missing not_evaluable",
          DV.classify_v3a({64: 3.0, 128: 2.0, 256: 1.0}, (64, 128, 256)) == "concordant"
          and DV.classify_v3a({64: 3.0, 128: 2.0, 256: 2.0}, (64, 128, 256)) == "not concordant"
          and DV.classify_v3a({64: 3.0, 128: float("nan"), 256: 1.0}, (64, 128, 256)) == "not_evaluable")
    check("(g) V3a reference states: not available > not verified > verified",
          DV.reference_label(False, "accept") == "reference not available" and DV.reference_label(True, "reject") == "reference not verified"
          and DV.reference_label(True, "accept") == "reference verified"
          and DV.final_v3a("reference not verified", "concordant") == "reference not verified"
          and DV.final_v3a("reference verified", "not concordant") == "not concordant")


def test_criterion_c(tmp):
    script = DV.find_orders_digest()
    check("(h) orders_digest.py found from the repository root", script is not None and os.path.isfile(script))
    raw = os.path.join(REPO, SF35, "raw", "matrix_eps10")
    files = [os.path.join(raw, "N%d.json" % n) for n in (64, 128, 256)]
    out = os.path.join(tmp, "cdir")
    os.makedirs(out, exist_ok=True)
    v, rc, note = DV.run_criterion_c(script, "1", files, (64, 128, 256), os.path.join(out, "c.md"), os.path.join(out, "c_planes.csv"))
    check("(h) criterion C in place on the SF-35 seed-3001 eps = 1 summaries (64/128/256): verdict parsed, = accept", v == "accept" and rc == 0, "verdict %s rc %s %s" % (v, rc, note))
    check("(h) criterion C wrote its report and planes csv", os.path.isfile(os.path.join(out, "c.md")) and os.path.isfile(os.path.join(out, "c_planes.csv")))
    # altered copies
    alt = os.path.join(tmp, "alt")
    os.makedirs(alt, exist_ok=True)
    for n in (64, 128, 256):
        js = json.load(open(os.path.join(raw, "N%d.json" % n)))
        if n == 256:
            js["metrics"]["trace"]["e_v"] = js["metrics"]["trace"]["e_v"] * 0.0 + json.load(open(os.path.join(raw, "N128.json")))["metrics"]["trace"]["e_v"] * 0.97
        with open(os.path.join(alt, "N%d.json" % n), "w") as f:
            json.dump(js, f)
    v2, _, _ = DV.run_criterion_c(script, "1", [os.path.join(alt, "N%d.json" % n) for n in (64, 128, 256)], (64, 128, 256),
                                  os.path.join(out, "c2.md"), os.path.join(out, "c2.csv"))
    check("(h) criterion C: an e_v that stops converging (3 % change 128 -> 256) gives reject", v2 == "reject", v2)
    v3, _, _ = DV.run_criterion_c(script, "1", files[:2], (64, 128, 256), os.path.join(out, "c3.md"), os.path.join(out, "c3.csv"))
    check("(h) criterion C: a missing grid gives not_evaluable (never accept)", v3 == "not_evaluable", v3)
    v4, _, n4 = DV.run_criterion_c(script, "1", [], (64, 128, 256), os.path.join(out, "c4.md"), os.path.join(out, "c4.csv"))
    check("(h) criterion C: no summary at all gives not_evaluable", v4 == "not_evaluable")
    check("(h) verdict parser reads '**Amplitude verdict (C): accept**' and ignores the rule text",
          DV.parse_c_verdict("Amplitude verdict: `accept` iff\n**Amplitude verdict (C): unresolved**\n") == "unresolved" and DV.parse_c_verdict("nothing") is None)
    # stage identity parsing
    t1 = "SETUP  SF-19: mg_levels=3 converged=1 G=(1.5e+00, -2e-02, 3.0e-03) mean_flux\n  inlet: vmin=1.234560e-01 mean(U plane 0)=1 Q0-1=0 v_rms=1.307292e+00\n"
    t2 = "TRACK field=gaussian inlet_vmin=1.234560e-01\n  SF-19: G=(1.5e+00, -2e-02, 3.0e-03)\n  inlet: vmin=1.2e-01 v_rms=1.307292e+00"
    a, b = DV.parse_stage_identity(t1), DV.parse_stage_identity(t2)
    check("(h) stage identity parsing: G, inlet_vmin, v_rms at print precision; identical / absent / DIFFERENT reported",
          a["G"] == ("1.5e+00", "-2e-02", "3.0e-03") and DV.compare_identity(a, b, "G") == "identical"
          and DV.compare_identity(a, b, "v_rms") == "identical"
          and DV.compare_identity(a, DV.parse_stage_identity("x"), "G") == "absent in the forward log"
          and DV.compare_identity(DV.parse_stage_identity(None), DV.parse_stage_identity(None), "G") == "absent in both logs"
          and DV.compare_identity(a, DV.parse_stage_identity("inlet_vmin=9.9e-01"), "inlet_vmin").startswith("DIFFERENT"))
    codes_dir = os.path.join(tmp, "codes", "labels")
    os.makedirs(codes_dir, exist_ok=True)
    with open(os.path.join(codes_dir, "exit_codes_labels-o3.tsv"), "w") as f:
        f.write("case\trc\twall_s\tend_utc\neps15_s1/N64\t21\t10\tx\neps15_s1/N128\t19\t3\ty\n")
    man = L.Manifest(os.path.join(tmp, "codes"))
    ex = O3.read_exit_codes(os.path.join(tmp, "codes"), man)
    check("(h) exit-code tsv parsed (header skipped)", ex == {"eps15_s1/N64": 21, "eps15_s1/N128": 19}, str(ex))


# --------------------------------------------------------------------------------------------------------------
# V3b synthetic runs
# --------------------------------------------------------------------------------------------------------------
def write_pollock_csvs(d, w, x23, tau, tx, planes, times, st_p):
    Np, P = tau.shape
    T = tx.shape[1]
    with open(os.path.join(d, "pollock_moments_planes.csv"), "w") as f:
        f.write("x1,contributing,non_ok,mean_x2,var_x2,mean_x3,var_x3,mean_t,var_t\n")
        for m in range(P):
            sel = [math.isfinite(x23[p, m, 0]) and math.isfinite(x23[p, m, 1]) for p in range(Np)]
            m2, v2, cnt = S37.loop_mean_var(w, x23[:, m, 0], sel)
            m3, v3, _ = S37.loop_mean_var(w, x23[:, m, 1], sel)
            mt, vt, _ = S37.loop_mean_var(w, tau[:, m], sel)
            f.write("%.15e,%d,%d,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e\n" % (planes[m], cnt, int((st_p != 0).sum()), m2, v2, m3, v3, mt, vt))
    with open(os.path.join(d, "pollock_moments_time.csv"), "w") as f:
        f.write("# format_version=1\n")
        f.write("t,mean_x,mean_y,mean_z,var_x,var_y,var_z,active\n")
        for n in range(T):
            sel = [all(math.isfinite(tx[p, n, k]) for k in range(3)) for p in range(Np)]
            ms, vs = [], []
            cnt = 0
            for k in range(3):
                m_, v_, cnt = S37.loop_mean_var(w, tx[:, n, k], sel)
                ms.append(m_)
                vs.append(v_)
            f.write("%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,%d\n" % (times[n], ms[0], ms[1], ms[2], vs[0], vs[1], vs[2], cnt))


def write_v3b_run(d, truth, planes, times, M, scale, weights_mode="same", grid_noise=0.0, noise_seed=0, nonactive=0):
    """O6 files via the SF-37 synthetic writer + Pollock files = scaled O6 positions (known D_P)."""
    S37.write_fake_run(d, truth, planes, times, M, grid_noise=grid_noise, noise_seed=noise_seed)
    ld = lambda n: np.load(os.path.join(d, n))
    w6, x23, tau, tx = ld("weights.npy"), ld("planes_x23.npy"), ld("planes_tau.npy"), ld("times_x.npy")
    stp, stt = ld("status_plane.npy"), ld("status_time.npy")
    wp = w6.copy() if weights_mode == "same" else (w6 ** 2) / (w6 ** 2).sum()
    px, ptx = x23 * scale, tx * scale
    stp2, stt2 = np.zeros_like(stp), np.zeros_like(stt)  # Pollock's own status codes (0 ok; 12, 14, 15 non-active)
    if nonactive:
        stp2[:nonactive] = 12
        stt2[:nonactive] = 14
    np.save(os.path.join(d, "pollock_weights.npy"), wp)
    np.save(os.path.join(d, "pollock_status_plane.npy"), stp2)
    np.save(os.path.join(d, "pollock_status_time.npy"), stt2)
    np.save(os.path.join(d, "pollock_planes_x23.npy"), px)
    np.save(os.path.join(d, "pollock_planes_t.npy"), tau)
    np.save(os.path.join(d, "pollock_times_x.npy"), ptx)
    write_pollock_csvs(d, wp, px, tau, ptx, planes, times, stp2)
    return w6, wp, x23, px


def test_v3b(tmp):
    M, planes, times = 16, L.PLANES, L.TIMES
    P, T = len(planes), len(times)
    truth = S37.fake_truth(77, M, P, T)
    d1 = os.path.join(tmp, "v3b_same")
    s = 1.1
    w6, wp, x23, px = write_v3b_run(d1, truth, planes, times, M, s, "same")
    man = L.Manifest(tmp)
    o6 = L.RunData(d1, man)
    pr = DV.PollockRun(d1, man, "P", "pollock")
    ph = DV.PollockRun(d1, man, "Ph", "o6")
    check("(i) PollockRun maps the pollock_* files onto the RunData interface (Np, weights, planes, masks)",
          pr.complete and pr.Np == M * M and np.array_equal(pr.w, wp) and np.array_equal(ph.w, w6)
          and np.array_equal(pr.mem("planes_x23.npy"), px, equal_nan=True) and pr.P == P and np.array_equal(pr.plane_mask(), o6.plane_mask()))
    idx = L.load_bootstrap(manifest=man)
    counts = L.counts_from_idx(idx, 16)
    tot = L.paired_stats(pr, o6, counts)
    s2o = tot["sigma_T2"]["b"]
    exp = (s * s - 1.0) * s2o
    check("(i) D_P = sigma_T^2(Pollock) - sigma_T^2(O6) equals (s^2-1) sigma_T^2(O6) for positions scaled by s (<= 1e-12 rel)",
          float(np.abs(tot["sigma_T2"]["D"] - exp).max() / np.abs(exp).max()) <= 1e-12, "%.2e" % float(np.abs(tot["sigma_T2"]["D"] - exp).max() / np.abs(exp).max()))
    # naive SE_D at plane 2 from the committed index matrix
    m = 2
    ids = L.block_ids(M * M, 4)
    mult = counts[:, ids]  # (R, Np)
    mask = o6.plane_mask()[:, m]
    ww = mult * (w6 * mask)[None, :]
    Wt = ww.sum(axis=1)
    var = 0.0
    for k in (0, 1):
        x = x23[:, m, k]
        xx = np.where(mask, x, 0.0)
        mu = (ww * xx[None, :]).sum(axis=1) / Wt
        var = var + 0.5 * ((ww * (xx[None, :] - mu[:, None]) ** 2).sum(axis=1) / Wt)
    se_naive = float(np.std((s * s - 1.0) * var, ddof=1))
    se_lib = float(tot["sigma_T2"]["se"][m])
    check("(i) SE_D of the paired block bootstrap equals the naive resample recomputation (<= 1e-8 rel)", abs(se_lib - se_naive) / se_naive <= 1e-8,
          "lib %.6e naive %.6e" % (se_lib, se_naive))
    wpart = L.paired_stats(pr, ph, counts)["sigma_T2"]
    tpart = L.paired_stats(ph, o6, counts)["sigma_T2"]
    check("(i) D6 identical weights: weight part exactly 0 (D and SE_D), trajectory part = D_P", float(np.abs(wpart["D"]).max()) == 0.0
          and float(np.abs(wpart["se"]).max()) == 0.0 and float(np.abs(tpart["D"] - tot["sigma_T2"]["D"]).max()) <= 1e-14)
    ok, worst, non_ok, cols = DV.pollock_moments_check(pr)
    check("(i) Pollock moments reproduce pollock_moments_planes/time.csv (<= 1e-12 rel) and the non_ok column", ok and non_ok is True, "worst %.2e" % worst)
    # different Pollock weights
    d2 = os.path.join(tmp, "v3b_diffw")
    w6b, wpb, x23b, pxb = write_v3b_run(d2, truth, planes, times, M, 1.05, "different", grid_noise=0.0, noise_seed=3, nonactive=7)
    o6b = L.RunData(d2, man)
    prb = DV.PollockRun(d2, man, "Pb", "pollock")
    phb = DV.PollockRun(d2, man, "Phb", "o6")
    totb = L.paired_stats(prb, o6b, counts)["sigma_T2"]
    wb = L.paired_stats(prb, phb, counts)["sigma_T2"]
    tb = L.paired_stats(phb, o6b, counts)["sigma_T2"]
    sel = o6b.plane_mask()
    ok_dec = float(np.abs(wb["D"] + tb["D"] - totb["D"]).max()) <= 1e-12
    # independent weight-part value at plane 1: sigma^2(P-traj, P-w) - sigma^2(P-traj, O6-w) by explicit loops
    m = 1
    sl = sel[:, m]
    sigw = 0.5 * sum(L.seq_wmean_var(wpb[sl], pxb[sl, m, k])[1] for k in (0, 1))
    sigo = 0.5 * sum(L.seq_wmean_var(w6b[sl], pxb[sl, m, k])[1] for k in (0, 1))
    check("(i) D6 different weights: weight part + trajectory part = D_P (<= 1e-12) and weight part equals the explicit-loop value",
          ok_dec and abs(wb["D"][m] - (sigw - sigo)) <= 1e-12 * abs(sigw), "decomp %s, w-part %.6e vs %.6e" % (ok_dec, wb["D"][m], sigw - sigo))
    st = DV.status_summary(prb, "plane")
    check("(i) Pollock non-active count and weight per pass (7 particles with code 12)", st["n_non_active"] == 7 and st["by_code"] == {12: 7}
          and abs(st["w_non_active"] - float(wpb[:7].sum())) <= 1e-15)
    stt = DV.status_summary(prb, "time")
    check("(i) Pollock non-active count in the time pass (code 14)", stt["n_non_active"] == 7 and stt["by_code"] == {14: 7})
    comp = L.paired_stats(prb, o6b, counts, 4, prb.plane_mask() & o6b.plane_mask(), prb.time_mask() & o6b.time_mask())
    check("(i) D_P on the common contributors is computed (finite, equals D_P when the masks coincide)",
          np.isfinite(comp["sigma_T2"]["D"]).all() and float(np.abs(comp["sigma_T2"]["D"] - totb["D"]).max()) <= 1e-12)
    # classification rules
    ns = (64, 128, 256, 512)
    C = lambda D, ref=1.0, se=0.001: DV.v3b_classify(ns, dict(zip(ns, D)), ref, se)[0]
    check("(i) V3b concordant: |D_P| decreasing on every refinement and |D_P(512)| <= max(5 %, 2 SE_D)", C([0.4, 0.2, 0.1, 0.04]) == "concordant")
    check("(i) V3b concordant through the 2 SE_D branch (|D_P(512)| = 0.09 <= 2 x 0.05)", C([0.5, 0.3, 0.2, 0.09], 1.0, 0.05) == "concordant")
    check("(i) V3b discordant: not decreasing on the last refinement", C([0.4, 0.2, 0.1, 0.11]) == "discordant")
    check("(i) V3b discordant: decreasing but |D_P(512)| above both thresholds", C([0.9, 0.6, 0.4, 0.3]) == "discordant")
    check("(i) V3b sign change is judged on |D_P| (signed values 0.4, -0.2, 0.1, -0.04)", C([0.4, -0.2, 0.1, -0.04]) == "concordant")
    check("(i) V3b not_evaluable: a missing N or NaN", C([0.4, 0.2, float("nan"), 0.04]) == "not_evaluable"
          and DV.v3b_classify(ns, {64: 0.1, 128: 0.05}, 1.0, 0.1)[0] == "not_evaluable")
    check("(i) noise-limited flag: |D(N) - D(2N)| < 2 x the larger SE_D",
          DV.noise_limited(0.10, 0.02, 0.07, 0.01) is True and DV.noise_limited(0.10, 0.01, 0.05, 0.01) is False
          and DV.noise_limited(0.1, float("nan"), 0.05, 0.01) is None)
    sl = DV.lsq_slope([0.25, 0.5, 1.0, 2.0], [0.1 * x + 0.3 for x in (0.25, 0.5, 1.0, 2.0)])
    check("(i) slope of D_P against x1 by least squares", abs(sl - 0.1) <= 1e-12)
    # consistency helper
    a, b = os.path.join(tmp, "fa.bin"), os.path.join(tmp, "fb.bin")
    open(a, "wb").write(b"abc")
    open(b, "wb").write(b"abc")
    same = DV.files_identical(a, b)
    open(b, "wb").write(b"abd")
    check("(i) files_identical: True / False / None (missing)", same is True and DV.files_identical(a, b) is False and DV.files_identical(a, a + "x") is None)


# --------------------------------------------------------------------------------------------------------------
# end-to-end tree
# --------------------------------------------------------------------------------------------------------------
def save(path, arr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, arr)


def build_tree(root):
    M, planes, times = 16, L.PLANES, L.TIMES
    P, T = len(planes), len(times)
    raw64 = os.path.join(REPO, SF35, "raw", "matrix_eps10")
    tsv_o3 = ["case\trc\twall_s\tend_utc"]
    tsv_10 = ["case\trc\twall_s\tend_utc"]
    # ---- O3 part: eps 15 (exact vD) and 20 (biased vD), seed 7001; eps 15 seed 7002 with exit 19 at N=64
    for e, bias in ((15, 1.0), (20, 1.05)):
        truth = S37.fake_truth(900 + e, M, P, T)
        for N in (8, 16, 32, 64):
            p1, p2, _ = analytic_labels(N, 1.0 if e == 15 else 1.5)
            d = os.path.join(root, "labels", "exports", "eps%d_s7001" % e, "N%d" % N)
            save(os.path.join(d, "psi1.npy"), p1)
            save(os.path.join(d, "psi2.npy"), p2)
            tsv_o3.append("eps%d_s7001/N%d\t21\t1.0\tx" % (e, N))
        for N in (16, 32, 64):
            _, _, c = analytic_labels(N, 1.0 if e == 15 else 1.5)
            run = os.path.join(root, "v2", L.v2_name(e, 7001, N))
            S37.write_fake_run(run, truth, planes, times, M, grid_noise=0.0, noise_seed=N)
            vd = os.path.join(run, "vd")
            for k in range(3):
                save(os.path.join(vd, "vD%d.npy" % (k + 1)), c[k] * bias)
            with open(os.path.join(vd, "vD.json"), "w") as f:
                json.dump({"v_rms": float(np.sqrt(np.mean(c[0] ** 2 + c[1] ** 2 + c[2] ** 2)))}, f)
    d = os.path.join(root, "labels", "exports", "eps15_s7002")
    for N in (8, 16, 32, 64):
        p1, p2, _ = analytic_labels(N)
        save(os.path.join(d, "N%d" % N, "psi1.npy"), p1)
        save(os.path.join(d, "N%d" % N, "psi2.npy"), p2)
        tsv_o3.append("eps15_s7002/N%d\t%d\t1.0\tx" % (N, 19 if N == 64 else 21))
    for N in (16, 32, 64):
        truth = S37.fake_truth(1234, M, P, T)
        run = os.path.join(root, "v2", L.v2_name(15, 7002, N))
        S37.write_fake_run(run, truth, planes, times, M)
        _, _, c = analytic_labels(N)
        for k in range(3):
            save(os.path.join(run, "vd", "vD%d.npy" % (k + 1)), c[k])
    # ---- V3a part: eps 10, seed 7001 verified reference; seed 7002 inlet backflow (labels NaN, exit 17 at N=8)
    rng = np.random.default_rng(31)
    for N in (8, 16, 32):
        p1, p2 = planar_labels(N)
        d = os.path.join(root, "labels", "exports", "eps10_s7001", "N%d" % N)
        save(os.path.join(d, "psi1.npy"), p1)
        save(os.path.join(d, "psi2.npy"), p2)
        tsv_10.append("eps10_s7001/N%d\t21\t1.0\tx" % N)
        js = json.load(open(os.path.join(raw64, "N%d.json" % {8: 64, 16: 128, 32: 256}[N])))
        js["N"] = N
        js["exit_code"] = 0
        os.makedirs(os.path.join(root, "labels", "raw", "eps10_s7001"), exist_ok=True)
        with open(os.path.join(root, "labels", "raw", "eps10_s7001", "N%d.json" % N), "w") as f:
            json.dump(js, f)
        stage = ("SETUP  SF-19: mg_levels=3 converged=1 G=(1.5e+00, -2e-02, 3.0e-03) mean_flux\n  inlet: vmin=1.234560e-01 mean(U) v_rms=1.307292e+00\n"
                 "TRACE_STAGE inlet_vmin=1.234560e-01 Q0=1 v_rms=1.307292e+00\n")
        os.makedirs(os.path.join(root, "labels", "logs", "eps10_s7001"), exist_ok=True)
        with open(os.path.join(root, "labels", "logs", "eps10_s7001", "N%d.log" % N), "w") as f:
            f.write(stage if N != 32 else "no stage lines\n")
        fw = stage.replace("1.234560e-01", "9.999990e-02") if N == 16 else stage
        os.makedirs(os.path.join(root, "v3a"), exist_ok=True)
        with open(os.path.join(root, "v3a", DV.v3a_name(10, 7001, N) + ".log"), "w") as f:
            f.write(fw)
        # forward run: all planes at one set of unwrapped landings; labels constant along x1 lines (x1-independent field)
        truth = S37.fake_truth(55, M, 4, T)
        run = os.path.join(root, "v3a", DV.v3a_name(10, 7001, N))
        S37.write_fake_run(run, truth, DV.V3A_PLANES, times, M)
        yu = rng.uniform(-1.0, 2.0, M * M)
        zu = rng.uniform(-1.0, 2.0, M * M)
        x23 = np.load(os.path.join(run, "planes_x23.npy"))
        fin = np.isfinite(x23[:, :, 0])
        for m in range(4):
            x23[:, m, 0] = np.where(fin[:, m], yu, np.nan)
            x23[:, m, 1] = np.where(fin[:, m], zu, np.nan)
        np.save(os.path.join(run, "planes_x23.npy"), x23)
        np.save(os.path.join(run, "labels.npy"), np.stack(planar_value(yu, zu), axis=1))
        np.save(os.path.join(run, "seeds.npy"), np.stack([np.zeros(M * M), yu, zu], axis=1))
    # seed 7002: inlet backflow
    tsv_10.append("eps10_s7002/N8\t17\t1.0\tx")
    for N in (8, 16, 32):
        truth = S37.fake_truth(56, M, 4, T)
        run = os.path.join(root, "v3a", DV.v3a_name(10, 7002, N))
        S37.write_fake_run(run, truth, DV.V3A_PLANES, times, M)
    # ---- V3b part: eps 10/15/20 seed 7001, N 16/32/64 (Pollock = O6 scaled by 1 + 0.05 (16/N)^2); v2 twins carry the same O6 files
    for e in (10, 15, 20):
        truth = S37.fake_truth(300 + e, M, P, T)
        for N in (16, 32, 64):
            d = os.path.join(root, "v3b", DV.v3b_name(e, 7001, N))
            write_v3b_run(d, truth, planes, times, M, 1.0 + 0.05 * (16.0 / N) ** 2, "same", grid_noise=0.5 * (8.0 / N) ** 2, noise_seed=N + e)
            if e == 10 and N == 64:
                continue  # no v2 twin: 'v2 run absent'
            v2d = os.path.join(root, "v2", L.v2_name(e, 7001, N))
            os.makedirs(v2d, exist_ok=True)
            for fn in DV.O6_CONSISTENCY_FILES + ("track.json",):
                shutil.copy(os.path.join(d, fn), os.path.join(v2d, fn))
    # corrupt one O6 file of the eps 2, N = 32 v2 twin
    v2w = os.path.join(root, "v2", L.v2_name(20, 7001, 32), "weights.npy")
    np.save(v2w, np.load(v2w) * 1.0000001)
    with open(os.path.join(root, "labels", "exit_codes_labels-o3.tsv"), "w") as f:
        f.write("\n".join(tsv_o3) + "\n")
    with open(os.path.join(root, "labels", "exit_codes_labels-eps10.tsv"), "w") as f:
        f.write("\n".join(tsv_10) + "\n")


def test_e2e(tmp):
    root = os.path.join(tmp, "camp")
    build_tree(root)
    out3 = os.path.join(tmp, "out_o3")
    r = O3.main([root, out3, "--design", "discovered"])
    check("(j) o3.py (discovered, synthetic tree) exits 0 and writes cells, verdicts, O3c, D7, summary, manifest",
          r == 0 and all(os.path.isfile(os.path.join(out3, f)) for f in ("o3_cells.csv", "o3_verdicts.csv", "o3c.csv", "o3_d7.csv", "o3_summary.md", "input_manifest_o3.csv")))
    hdr, rows = csv_rows(os.path.join(out3, "o3_verdicts.csv"))
    byk = {(r_["eps_code"], r_["seed"]): r_ for r_ in rows}
    check("(j) O3a: exact vD (eps 1.5) -> accept; vD biased by 5 % (eps 2) -> reject; exit 19 at N=64 (seed 7002) -> not_evaluable",
          byk[("15", "7001")]["O3a"] == "accept" and byk[("20", "7001")]["O3a"] == "reject" and byk[("15", "7002")]["O3a"] == "not_evaluable",
          "%s / %s / %s" % (byk[("15", "7001")]["O3a"], byk[("20", "7001")]["O3a"], byk[("15", "7002")]["O3a"]))
    check("(j) O3b verdicts are in the allowed set and not_evaluable follows O3a's missing data",
          all(r_["O3b"] in ("shrinking", "not shrinking") for r_ in rows if r_["seed"] == "7001") and byk[("15", "7002")]["O3b"] == "not_evaluable")
    check("(j) the not_evaluable reason names the construction exit", "exit 19" in slurp(os.path.join(out3, "o3_verdicts.csv")))
    hdr, crow = csv_rows(os.path.join(out3, "o3c.csv"))
    check("(j) O3c rows: planes 0.25, 0.5, pooled and the structurally empty outlet for every evaluable (eps, seed, N)",
          sum(1 for r_ in crow if r_["x1"] == "1.0" and r_["status"] == "structurally_empty") >= 9
          and any(r_["x1"] == "pooled" and r_["status"] == "ok" for r_ in crow) and any(r_["x1"] == "0.25" and r_["status"] == "ok" for r_ in crow))
    hdr, cells = csv_rows(os.path.join(out3, "o3_cells.csv"))
    check("(j) blind output: o3_cells.csv carries only the O3 columns (no per-plane label metrics)",
          not any(("plane" in h or "x1_" in h) for h in hdr if h not in ("d_max_x1",)))
    out_a = os.path.join(tmp, "out_v3")
    r = DV.main([root, out_a, "--design", "discovered"])
    files = ("v3_summary.md", "v3a_cells.csv", "v3a_verdicts.csv", "v3a_criterionC.csv", "v3a_stage_identity.csv", "v3a_constructor.csv", "v3a_control_x0.csv",
             "v3b_cells.csv", "v3b_verdicts.csv", "v3b_noise_limited.csv", "v3b_slopes.csv", "v3b_pollock_status.csv", "v3b_time_snapshots.csv",
             "v3b_consistency.csv", "input_manifest_v3.csv", "criterionC_s7001.md", "criterionC_s7001_planes.csv")
    check("(j) digest_v3.py (discovered, synthetic tree) exits 0 and writes every output", r == 0 and all(os.path.isfile(os.path.join(out_a, f)) for f in files),
          ",".join(f for f in files if not os.path.isfile(os.path.join(out_a, f))))
    hdr, vrows = csv_rows(os.path.join(out_a, "v3a_verdicts.csv"))
    s1 = [r_ for r_ in vrows if r_["seed"] == "7001"]
    s2 = [r_ for r_ in vrows if r_["seed"] == "7002"]
    check("(j) V3a seed 7001: criterion C accept, reference verified, interpolation error decreases -> concordant at all 4 planes",
          len(s1) == 4 and all(r_["criterion_C"] == "accept" and r_["reference"] == "reference verified" and r_["classification"] == "concordant" for r_ in s1),
          str([(r_["x1"], r_["classification"], r_["rms_N8"], r_["rms_N16"], r_["rms_N32"]) for r_ in s1]))
    check("(j) V3a seed 7002 (inlet backflow, exit 17, labels NaN): reference not available", len(s2) == 4 and all(r_["classification"] == "reference not available" for r_ in s2),
          str([r_["classification"] for r_ in s2]))
    hdr, idr = csv_rows(os.path.join(out_a, "v3a_stage_identity.csv"))
    idm = {r_["N"]: r_ for r_ in idr if r_["seed"] == "7001"}
    check("(j) stage identity: identical at N=8, inlet_vmin DIFFERENT at N=16, 'absent in the trace log' at N=32",
          idm["8"]["G"] == "identical" and idm["16"]["inlet_vmin"].startswith("DIFFERENT") and idm["32"]["G"] == "absent in the trace log",
          str((idm["8"]["G"], idm["16"]["inlet_vmin"], idm["32"]["G"])))
    hdr, ctl = csv_rows(os.path.join(out_a, "v3a_control_x0.csv"))
    c7001 = [r_ for r_ in ctl if r_["seed"] == "7001"]
    check("(j) V3a control x1 = 0 recorded for every N of seed 7001", len(c7001) == 3 and all(float(r_["rms_unweighted"]) < 1e-2 for r_ in c7001))
    hdr, v3b = csv_rows(os.path.join(out_a, "v3b_verdicts.csv"))
    check("(j) V3b: 3 amplitudes x 7 planes, every cell concordant (|D_P| = (s^2 - 1) sigma^2 decreasing, final <= 5 %)",
          len(v3b) == 21 and all(r_["classification"] == "concordant" for r_ in v3b), str(set(r_["classification"] for r_ in v3b)))
    hdr, cons = csv_rows(os.path.join(out_a, "v3b_consistency.csv"))
    cm = {(r_["eps_code"], r_["N"]): r_ for r_ in cons}
    check("(j) consistency: O6 files bitwise identical to the v2 twin (eps 1.5), DIFFERENT where corrupted (eps 2, N=32), 'v2 run absent' where no twin, Pollock moments reproduced",
          cm[("15", "32")]["O6_files_vs_v2"] == "bitwise identical" and cm[("20", "32")]["O6_files_vs_v2"].startswith("DIFFERENT")
          and cm[("20", "16")]["O6_files_vs_v2"] == "bitwise identical" and cm[("10", "64")]["O6_files_vs_v2"] == "v2 run absent"
          and all(r_["pollock_moments_le_1e-12"] == "yes" for r_ in cons),
          str({k: v["O6_files_vs_v2"] for k, v in cm.items()}))
    md = slurp(os.path.join(out_a, "v3_summary.md"))
    check("(j) the summary states verdict and reason per cell", "V3a" in md and "V3b" in md and "concordant" in md and "reference not available" in md)
    # full design on the same tree: seeds 3101-3103 absent -> nothing can pass
    out_f3 = os.path.join(tmp, "out_o3_full")
    O3.main([root, out_f3, "--design", "full"])
    hdr, fr = csv_rows(os.path.join(out_f3, "o3_verdicts.csv"))
    check("(j) o3.py full design on a tree without the pre-registered runs: every O3a/O3b verdict is not_evaluable", len(fr) == 6
          and all(r_["O3a"] == "not_evaluable" and r_["O3b"] == "not_evaluable" for r_ in fr))
    out_fa = os.path.join(tmp, "out_v3_full")
    DV.main([root, out_fa, "--design", "full"])
    hdr, fa = csv_rows(os.path.join(out_fa, "v3a_verdicts.csv"))
    hdr, fb = csv_rows(os.path.join(out_fa, "v3b_verdicts.csv"))
    check("(j) digest_v3.py full design without the pre-registered runs: V3a never concordant, V3b all not_evaluable",
          len(fa) == 12 and not any(r_["classification"] == "concordant" for r_ in fa) and len(fb) == 63 and all(r_["classification"] == "not_evaluable" for r_ in fb))
    # determinism
    out_a2 = os.path.join(tmp, "out_v3b")
    DV.main([root, out_a2, "--design", "discovered"])
    same = True
    for fn in ("v3a_cells.csv", "v3b_cells.csv", "v3b_verdicts.csv", "v3a_verdicts.csv"):
        a_ = [ln for ln in slurp(os.path.join(out_a, fn)).splitlines() if not ln.startswith("#")]
        b_ = [ln for ln in slurp(os.path.join(out_a2, fn)).splitlines() if not ln.startswith("#")]
        same = same and a_ == b_
    check("(j) digest_v3.py is deterministic (identical csv on repeat)", same)
    out32 = os.path.join(tmp, "out_o3b")
    O3.main([root, out32, "--design", "discovered"])
    check("(j) o3.py is deterministic (identical verdict and cell csv on repeat)",
          [ln for ln in slurp(os.path.join(out3, "o3_cells.csv")).splitlines() if not ln.startswith("#")]
          == [ln for ln in slurp(os.path.join(out32, "o3_cells.csv")).splitlines() if not ln.startswith("#")])


def test_api():
    banned = ["np.trapz", "np.trapezoid", "np.in1d", "np.product", "np.cumproduct", "np.float_", "np.alltrue", "np.sometrue",
              "np.row_stack", "np.asfarray", "np.round_", "np.NaN", "np.Inf", "np.infty", "np.unicode_", "np.string_", "np.bool8"]
    hits = []
    for fn in ("o3.py", "digest_v3.py"):
        src = slurp(os.path.join(HERE, fn))
        for b in banned:
            if b in src:
                hits.append("%s:%s" % (fn, b))
    check("(k) numpy API: no numpy-2-removed / numpy-1.26-absent names in o3.py and digest_v3.py", not hits, ",".join(hits))
    for fn in ("o3.py", "digest_v3.py", "selftest_o3_v3.py"):
        try:
            py_compile.compile(os.path.join(HERE, fn), doraise=True, cfile=os.path.join(tempfile.gettempdir(), "o3v3_%s.pyc" % fn))
            ok = True
        except py_compile.PyCompileError:  # pragma: no cover
            ok = False
        check("(k) py_compile %s" % fn, ok)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--keep", default=None, help="keep the temporary directory here")
    args = ap.parse_args(argv)
    if REPO is None:
        print("FAIL  repository root not found (SF-35 scripts)")
        return 1
    tmp = args.keep or tempfile.mkdtemp(prefix="o3v3_selftest_")
    os.makedirs(tmp, exist_ok=True)
    v = L.versions()
    print("selftest_o3_v3: python %s numpy %s scipy %s; tmp %s" % (v["python"], v["numpy"], v["scipy"], tmp))
    try:
        ld = load_ld()
        test_stencil_and_concentration(ld)
        test_marking()
        test_ev(tmp)
        test_verdicts()
        test_o3c(tmp)
        test_interp()
        test_criterion_c(tmp)
        test_v3b(tmp)
        test_e2e(tmp)
        test_api()
    finally:
        if args.keep is None:
            shutil.rmtree(tmp, ignore_errors=True)
    print("selftest_o3_v3: %d checks, %d failed" % (len(RESULTS), len(FAILS)))
    for f in FAILS:
        print("  FAILED: " + f)
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
