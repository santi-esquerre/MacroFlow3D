"""C3 design emulation: periodicity of v = k (G + grad f) under a shift by +1 in x2 / x3 on a medium (b)
style written field (not-a-knot in x1, periodic in x2, x3), numpy; not SF-41 evidence.

Point set of SF-38 F3 (tests/lagrangian_labels/long_box_tests.cu case_F3): 4096 points, x1 uniform in
[0, L1], x2, x3 uniform in [0, 1); i % 4 == 1 pushes x2 to the edge list, == 2 pushes x3, == 3 pushes x1
to a quarter-edge near a box face and x2, x3 to the edge list; edge = {0, h/4, h/2, h, 1-h, 1-h/2,
1-1e-9, 1}.  The metric is the F3 one: per component |v(x + e_k) - v(x)| <= 1e-14 and <= 1e-14 |v(x)|.
numpy has no FMA and uses a different summation order than the host/device evaluator: the numbers are
a MAGNITUDE estimate of the roundoff, not a bit-level prediction (except the dyadic block, where the
shift is exact in floating point, hence bitwise equality is expected for ANY evaluator that reduces
the coordinate first, as the SF-28 evaluator does).
Run: OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 nice -n 15 taskset -c 10-15 python3 -I c3_shift.py
"""
import importlib.util
import math
import os

import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("c_common", os.path.join(here, "c_common.py"))
cc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cc)

TP = 2.0 * math.pi
EDGE_H = None


# variant: (f kind, a, transverse ln k amplitude, x1-profile amplitude of ln k)
VARIANTS = {
    "V0": ("2d", 0.5, 0.3, 0.0),   # as specified
    "V1": ("3d", 0.5, 0.3, 0.5),   # x3-dependent harmonic f, non-periodic x1 profile in ln k
    "V2": ("2d", 0.1, 0.1, 0.0),   # V0 with small amplitudes (v1 >= 1 - 2 pi a > 0.37 everywhere)
    "V3": ("3d", 0.1, 0.1, 0.2),   # V1 with small amplitudes
}


def build_field(variant, nx, ny, h):
    kind, a, ak, a1 = VARIANTS[variant]
    x1 = ((np.arange(nx) + 0.5) * h)[:, None, None]
    x2 = ((np.arange(ny) + 0.5) * h)[None, :, None]
    x3 = ((np.arange(ny) + 0.5) * h)[None, None, :]
    if kind == "2d":
        f = a * np.exp(-TP * x1) * np.cos(TP * x2) + 0.0 * x3
    else:
        f = a * np.exp(-TP * math.sqrt(2.0) * x1) * np.cos(TP * x2) * np.cos(TP * x3)
    lnk = ak * np.sin(TP * x2) * np.cos(TP * x3) + a1 * np.cos(TP * 0.6 * x1 + 0.3) + 0.0 * x1
    axes = [("nak", nx, h), ("per", ny, h), ("per", ny, h)]
    return cc.Spline(axes, f), cc.Spline(axes, lnk)


def velocity(sf, sk, pts):
    _, gr = sf.eval(pts)
    y, _ = sk.eval(pts)
    k = np.exp(y)
    g = gr.copy()
    g[:, 0] += 1.0  # G1 = 1
    return k[:, None] * g


def f3_points(nx, h, npts, seed):
    Lx = nx * h
    rng = np.random.default_rng(seed)
    edge = np.array([0.0, 0.25 * h, 0.5 * h, h, 1.0 - h, 1.0 - 0.5 * h, 1.0 - 1e-9, 1.0])
    pts = np.empty((npts, 3))
    for i in range(npts):
        p = [Lx * rng.random(), rng.random(), rng.random()]
        if i % 4 == 1:
            p[1] = edge[rng.integers(8)]
        if i % 4 == 2:
            p[2] = edge[rng.integers(8)]
        if i % 4 == 3:
            e0 = edge[rng.integers(8)] * 0.25
            p[0] = e0 if rng.integers(2) == 0 else Lx - e0
            p[1] = edge[rng.integers(8)]
            p[2] = edge[rng.integers(8)]
        pts[i] = p
    return pts


def shift_stats(sf, sk, pts, lines):
    v0 = velocity(sf, sk, pts)
    vn = np.linalg.norm(v0, axis=1)
    out = {}
    for ax, name in ((1, "x2"), (2, "x3")):
        q = pts.copy()
        q[:, ax] += 1.0
        d = np.abs(velocity(sf, sk, q) - v0)
        out[name] = d
        viol = int(np.sum(~((d <= 1e-14) & (d <= 1e-14 * vn[:, None]))))
        lines.append(f"    shift +1 in {name}: max abs per component (v1, v2, v3) = "
                     f"{d[:, 0].max():.3e} {d[:, 1].max():.3e} {d[:, 2].max():.3e}; max rel (d/|v|) = "
                     f"{(d / vn[:, None]).max(axis=0)[0]:.3e} {(d / vn[:, None]).max(axis=0)[1]:.3e} "
                     f"{(d / vn[:, None]).max(axis=0)[2]:.3e}; components violating F3 metric: {viol}")
    return v0, vn


def main():
    h = 1.0 / 16.0
    print("C3 design emulation (numpy), NOT SF-41 evidence.  h = 1/16, G = (1,0,0).")
    print("V0: f = 0.5 cos(2 pi x2) exp(-2 pi x1) ; ln k = 0.3 sin(2 pi x2) cos(2 pi x3)   [as specified]")
    print("V1: f = 0.5 cos(2 pi x2) cos(2 pi x3) exp(-2 pi sqrt2 x1) (harmonic, x3-dependent) ;")
    print("    ln k = 0.3 sin(2 pi x2) cos(2 pi x3) + 0.5 cos(2 pi 0.6 x1 + 0.3)  [exercises ln k wrap in x2,x3 AND")
    print("    a non-periodic x1 profile in the not-a-knot ln k spline]")
    print("V2: V0 with a = 0.1 and ln k amplitude 0.1 ;  V3: V1 with f amplitude 0.1, ln k amplitudes 0.1 (transverse), 0.2 (x1)")
    print("point set: 4096 F3-style points, rng seed 20261022 (numpy default_rng PCG64; NOT the F3 mt19937 stream)")
    print()
    for variant in ("V0", "V1", "V2", "V3"):
        for nx in (64, 128):
            ny = 16
            sf, sk = build_field(variant, nx, ny, h)
            lines = []
            pts = f3_points(nx, h, 4096, 20261022)
            v0, vn = shift_stats(sf, sk, pts, lines)
            print(f"{variant} {nx}x{ny}x{ny}: |v| in [{vn.min():.3f}, {vn.max():.3f}]; v3 max |.| = "
                  f"{np.abs(v0[:, 2]).max():.3e}")
            print("  random + edge points:")
            for l in lines:
                print(l)
            # dyadic block: k / 2^16
            rng = np.random.default_rng(777)
            dy = np.empty((4096, 3))
            dy[:, 0] = rng.integers(0, int(nx * h * 65536) + 1, 4096) / 65536.0
            dy[:, 1] = rng.integers(0, 65536, 4096) / 65536.0
            dy[:, 2] = rng.integers(0, 65536, 4096) / 65536.0
            vd = velocity(sf, sk, dy)
            ndiff = {}
            for ax, name in ((1, "x2"), (2, "x3")):
                q = dy.copy()
                q[:, ax] += 1.0
                assert np.all(q[:, ax] - 1.0 == dy[:, ax])  # the shift is exact in floating point
                ndiff[name] = int(np.sum(velocity(sf, sk, q) != vd))
            print(f"  dyadic points k/2^16 (4096): components with v(x+e) != v(x) bitwise: x2 shift {ndiff['x2']}, "
                  f"x3 shift {ndiff['x3']}  (expected 0 / 0)")
            print()


if __name__ == "__main__":
    main()
