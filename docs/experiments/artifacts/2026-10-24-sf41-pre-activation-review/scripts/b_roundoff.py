"""SF-41 pre-activation design probe for threshold B (DESIGN emulation, numpy; NOT SF-41 evidence).

Question: when the SAME periodic samples are represented by (P) the periodic cubic B-spline (SF-28) and (NP) the spline
non-periodic in x1 (not-a-knot closure on the raw cell-centred samples, periodic in x2/x3), how large is the relative difference
of value, gradient and Hessian at points >= 2 lambda from both x faces, and how does the end effect decay with the knot distance?
Threshold B of SF-41: <= 1e-11 on 1664x128x128, lambda/h = 32.

Pairs (reference P vs candidate NP):
  P1  P = 3-D FFT prefilter (GPU-like)            vs NP = FFT in y/z + x line solve by sparse LU of the full (n+2) system
                                                          (this is the original probe of the scratchpad: probe_B_roundoff.py)
  P2  P = 3-D FFT prefilter (GPU-like)            vs NP = cyclic-tridiagonal y/z line solves + reduced tridiagonal x solve
  P3  P = cyclic-tridiagonal prefilter, all axes   vs NP = cyclic-tridiagonal y/z + reduced tridiagonal x solve (CPU mirror vs CPU)
Drivers (field f whose samples are represented):
  head1  linearised head fluctuation of a Gaussian-covariance Y (SF-18-like spectrum): f = irfft( i kx Yhat / k^2 )  [probe driver]
  head2  same linearisation with the nonlinear driver Z = Y + (Y^2 - 1)/2
  Klike  K-like field exp(2 Y) - mean (sigma_Y^2 = 4 lognormal conductivity; broadband spectrum)

usage: python3 -I b_roundoff.py   (no arguments; the configuration list is below)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from a_spline import (gaussian_field, prefilter_periodic_fft, prefilter_periodic_cyc, prefilter_np, evaluate3)

CONFIGS = [  # (nx, ny, nz, lambda/h, seeds)
    ((1664, 128, 128), 32.0, (6001,)),
    ((416, 32, 32), 32.0, (6001, 6002)),
    ((416, 32, 32), 8.0, (6001, 6002)),   # same physical lambda = 1/4 and L1 = 13 as the big box, coarser grid
]
DRIVERS = ("head1", "head2", "Klike")
NPTS = 100000


def head_of(Z, h):
    nx, ny, nz = Z.shape
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=h)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=h)
    kz = 2 * np.pi * np.fft.rfftfreq(nz, d=h)
    k2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    Zk = np.fft.rfftn(Z)
    with np.errstate(divide="ignore", invalid="ignore"):
        hk = np.where(k2 > 0, 1j * kx[:, None, None] * Zk / k2, 0.0)
    return np.fft.irfftn(hk, s=(nx, ny, nz))


def driver_field(name, Y, h):
    if name == "head1":
        return 2.0 * head_of(Y, h)
    if name == "head2":
        return 2.0 * head_of(Y + 0.5 * (Y * Y - 1.0), h)
    K = np.exp(2.0 * Y)
    return K - K.mean()


def rel(a, b):
    """value, gradient (Euclidean per point), Hessian (component-wise) relative differences, b = reference."""
    (va, ga, Ha), (vb, gb, Hb) = a, b
    rv = np.max(np.abs(va - vb)) / np.max(np.abs(vb))
    rg = np.max(np.linalg.norm(ga - gb, axis=1)) / np.max(np.linalg.norm(gb, axis=1))
    rH = np.max(np.abs(Ha - Hb)) / np.max(np.abs(Hb))
    return rv, rg, rH


def profile_report(label, cp, cn, i2, nx):
    """end-effect profile of the coefficient difference |c_np,i - c_p,i| (max over y,z) / max|c_p|."""
    scale = np.max(np.abs(cp))
    diff = np.max(np.abs(cn[2:2 + nx] - cp), axis=(1, 2)) / scale          # per knot i = 0..nx-1
    floor = diff[i2: nx - i2].max()
    out = [f"  [{label}] coef interior floor (knots {i2}..{nx - i2 - 1}) = {floor:.3e}"]
    for side, d in (("west", diff), ("east", diff[::-1])):
        prof = d[:24]
        out.append(f"  [{label}] {side} profile knots 0..23: " + " ".join(f"{p:.2e}" for p in prof))
        k = 0
        while k < 120 and d[k] > 1e3 * floor:
            k += 1                                                         # knots exceeding 1e3 x floor, contiguous from the face
        txt = f"  [{label}] {side}: {k} contiguous knots exceed 1e3 x floor ({1e3 * floor:.2e})"
        if k >= 4:
            i = np.arange(k)
            s_all = np.polyfit(i, np.log(d[:k]), 1)[0]
            s_tail = np.polyfit(i[3:], np.log(d[3:k]), 1)[0]
            txt += f"; fitted per-knot factor all={np.exp(s_all):.4f} tail(knots>=3)={np.exp(s_tail):.4f}"
        else:
            txt += "; fit n/a (< 4 knots)"
        out.append(txt)
    return "\n".join(out)


def run(shape, lam_over_h, seed):
    nx, ny, nz = shape
    h = 1.0 / ny
    ell = lam_over_h * h
    L1 = nx * h
    i2 = int(np.ceil(2 * ell / h))
    rng = np.random.default_rng(12345)
    px = rng.uniform(2 * ell, L1 - 2 * ell, NPTS)
    py = rng.uniform(0, 1, NPTS)
    pz = rng.uniform(0, 1, NPTS)
    xi = (np.arange(i2, nx - i2) + 0.5) * h                                # cell centres at distance >= 2 lambda
    lines = [(3, 5), (ny // 2, nz // 2 + 1), (ny - 2, nz // 3)]
    cx = np.concatenate([xi for _ in lines])
    cy = np.concatenate([np.full(len(xi), (j + 0.5) * h) for j, _ in lines])
    cz = np.concatenate([np.full(len(xi), (k + 0.5) * h) for _, k in lines])
    pts = np.column_stack([np.concatenate([px, cx]), np.concatenate([py, cy]), np.concatenate([pz, cz])])
    print(f"=== grid={nx}x{ny}x{nz} lambda/h={lam_over_h:g} (ell={ell:g}, L1={L1:g}) seed={seed} "
          f"points={len(pts)} (random {NPTS} + cell centres of 3 x1 lines {len(cx)}), distance >= 2 lambda from both x faces")
    Y = gaussian_field(nx, ny, nz, h, ell, seed)
    for drv in DRIVERS:
        f = driver_field(drv, Y, h)
        cpf = prefilter_periodic_fft(f)
        cnf = prefilter_np(f, "fft", "sparse")
        evp = evaluate3(cpf, pts, h, nx, True)
        res = {"P1": (cpf, cnf, rel(evaluate3(cnf, pts, h, nx, False), evp))}
        del cnf
        cn2 = prefilter_np(f, "cyc", "reduced")
        res["P2"] = (cpf, cn2, rel(evaluate3(cn2, pts, h, nx, False), evp))
        cpc = prefilter_periodic_cyc(f)
        res["P3"] = (cpc, cn2, rel(evaluate3(cn2, pts, h, nx, False), evaluate3(cpc, pts, h, nx, True)))
        print(f"-- driver {drv}  (max|f| = {np.abs(f).max():.3e})")
        for key, (cp, cn, (rv, rg, rH)) in res.items():
            verdict = "within" if max(rv, rg, rH) <= 1e-11 else "ABOVE"
            print(f"REL {key} value={rv:.3e} grad={rg:.3e} hess={rH:.3e}  ({verdict} the B gate 1e-11)")
        for key in ("P1", "P2", "P3"):
            cp, cn, _ = res[key]
            print(profile_report(key, cp, cn, i2, nx))
        del cpf, cn2, cpc, res, f


def main():
    print("# SF-41 pre-activation, threshold B round-off / end-effect probe (design emulation, numpy; NOT SF-41 evidence)")
    for shape, lam, seeds in CONFIGS:
        for seed in seeds:
            run(shape, lam, seed)


if __name__ == "__main__":
    main()
