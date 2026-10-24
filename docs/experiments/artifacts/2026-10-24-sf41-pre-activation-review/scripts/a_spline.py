"""Shared numpy emulation of the SF-28 periodic cubic B-spline and of the SF-41 candidate spline
(not-a-knot in x1, periodic in x2/x3).  DESIGN emulation, not SF-41 evidence.

Conventions (SF-28 header): cell-centred lattice x_i = (i + 1/2) h, coefficients on the same lattice,
t = x/h - 1/2, i0 = floor(t), u = t - i0, weights on c_{i0-1..i0+2}; interpolation c_{i-1} + 4 c_i + c_{i+1} = 6 f_i.

Non-periodic x1 closure (decided): unknowns c_{-1}..c_{n} (n+2); not-a-knot rows at x_1 and x_{n-2}; n interpolation rows on the
raw samples (no ghost samples); two continuation coefficients c_{-2}, c_{n+1} (4th difference = 0 at x_0 and x_{n-1}) so that the
evaluation is defined on [-h/2, L1 + h/2).  Stored with offset 2: C[k + 2] = c_k, shape (n + 4, ...).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import solve_banded

NAK = np.array([-1.0, 4.0, -6.0, 4.0, -1.0])


def symbol(n):
    m = np.arange(n)
    return 2.0 / 3.0 + np.cos(2 * np.pi * m / n) / 3.0


def symbol_half(n):
    """symbol on the rfft half spectrum m = 0..n//2 (the symbol is even in m)."""
    m = np.arange(n // 2 + 1)
    return 2.0 / 3.0 + np.cos(2 * np.pi * m / n) / 3.0


def weights(u):
    w = np.stack([(1 - u) ** 3 / 6, (3 * u**3 - 6 * u**2 + 4) / 6, (-3 * u**3 + 3 * u**2 + 3 * u + 1) / 6, u**3 / 6])
    dw = np.stack([-(1 - u) ** 2 / 2, (3 * u**2 - 4 * u) / 2, (-3 * u**2 + 2 * u + 1) / 2, u**2 / 2])
    d2w = np.stack([1 - u, 3 * u - 2, 1 - 3 * u, u])
    return w, dw, d2w


# ---------------------------------------------------------------- periodic prefilters
def prefilter_periodic_fft(f):
    """GPU-like: 3-D real FFT, divide by the per-axis symbols, inverse."""
    nx, ny, nz = f.shape
    F = np.fft.rfftn(f)
    F /= symbol(nx)[:, None, None] * symbol(ny)[None, :, None] * symbol_half(nz)[None, None, :]
    return np.fft.irfftn(F, s=(nx, ny, nz))


def cyc_axis(f, axis):
    """CPU-mirror-like: cyclic tridiagonal c_{i-1} + 4 c_i + c_{i+1} = 6 f_i along `axis` (tridiagonal solve + Sherman-Morrison)."""
    g = np.moveaxis(f, axis, 0)
    shp = g.shape
    n = shp[0]
    rhs = 6.0 * g.reshape(n, -1)
    gamma = -4.0
    ab = np.zeros((3, n))
    ab[0, 1:] = 1.0
    ab[2, :-1] = 1.0
    ab[1, :] = 4.0
    ab[1, 0] = 4.0 - gamma
    ab[1, -1] = 4.0 - 1.0 / gamma
    y = solve_banded((1, 1), ab, rhs)
    u = np.zeros(n)
    u[0] = gamma
    u[-1] = 1.0
    z = solve_banded((1, 1), ab, u)
    fac = (y[0] + y[-1] / gamma) / (1.0 + z[0] + z[-1] / gamma)
    y -= fac * z[:, None]
    return np.moveaxis(y.reshape(shp), 0, axis)


def prefilter_periodic_cyc(f):
    return cyc_axis(cyc_axis(cyc_axis(f, 0), 1), 2)


# ---------------------------------------------------------------- not-a-knot line solve along x
def nak_matrix(n):
    rows, cols, vals = [], [], []
    for j, v in enumerate(NAK):
        rows.append(0); cols.append(j); vals.append(v)            # c_{-1}..c_3
    for i in range(n):
        for d, v in zip((-1, 0, 1), (1.0 / 6, 4.0 / 6, 1.0 / 6)):
            rows.append(i + 1); cols.append(i + 1 + d); vals.append(v)
    for j, v in enumerate(NAK):
        rows.append(n + 1); cols.append(n - 3 + j); vals.append(v)  # c_{n-4}..c_n
    return sp.csc_matrix((vals, (rows, cols)), shape=(n + 2, n + 2))


def _continue(c, n):
    """c: (n+2, M) holding c_{-1}..c_n -> (n+4, M) with c_{-2}, c_{n+1}."""
    out = np.empty((n + 4,) + c.shape[1:])
    out[1:-1] = c
    out[0] = 4 * c[0] - 6 * c[1] + 4 * c[2] - c[3]
    out[-1] = 4 * c[-1] - 6 * c[-2] + 4 * c[-3] - c[-4]
    return out


def nak_x(g, method="reduced"):
    """g: (n, M) raw samples -> (n+4, M) coefficients c_{-2}..c_{n+1}.  method: reduced | dense | sparse.

    reduced: the two not-a-knot rows eliminate to c_1 = (8 f_1 - f_0 - f_2)/6 (and the mirror c_{n-2}); c_2..c_{n-3} then solve
    a plain [1 4 1] tridiagonal system; c_0, c_{-1}, c_{n-1}, c_n follow by back-substitution.
    """
    n = g.shape[0]
    g = g.reshape(n, -1)
    if method in ("dense", "sparse"):
        rhs = np.zeros((n + 2, g.shape[1]))
        rhs[1:-1] = g
        if method == "dense":
            c = np.linalg.solve(nak_matrix(n).toarray(), rhs)
        else:
            c = spla.splu(nak_matrix(n)).solve(rhs)
        return _continue(c, n)
    assert n >= 8
    c = np.zeros((n + 2, g.shape[1]))                              # row k + 1 holds c_k
    c[1 + 1] = (8 * g[1] - g[0] - g[2]) / 6                        # c_1
    c[n - 2 + 1] = (8 * g[n - 2] - g[n - 1] - g[n - 3]) / 6        # c_{n-2}
    m = n - 4                                                      # unknowns c_2..c_{n-3}
    rhs = 6.0 * g[2 : n - 2]
    rhs[0] -= c[2]
    rhs[-1] -= c[n - 1]
    ab = np.zeros((3, m))
    ab[0, 1:] = 1.0
    ab[1, :] = 4.0
    ab[2, :-1] = 1.0
    c[3 : n - 1] = solve_banded((1, 1), ab, rhs)
    c[1] = 6 * g[1] - 4 * c[2] - c[3]                              # c_0
    c[0] = 6 * g[0] - 4 * c[1] - c[2]                              # c_{-1}
    c[n] = 6 * g[n - 2] - 4 * c[n - 1] - c[n - 2]                  # c_{n-1}
    c[n + 1] = 6 * g[n - 1] - 4 * c[n] - c[n - 1]                  # c_n
    return _continue(c, n)


def prefilter_np(f, yz="fft", x="reduced"):
    """Non-periodic in x1 (not-a-knot, raw samples), periodic in x2/x3.  Returns (nx+4, ny, nz).
    yz: 'fft' (GPU-like) or 'cyc' (CPU-mirror-like cyclic tridiagonal line solves)."""
    nx, ny, nz = f.shape
    if yz == "fft":
        F = np.fft.rfft2(f, axes=(1, 2))
        F /= symbol(ny)[None, :, None] * symbol_half(nz)[None, None, :]
        g = np.fft.irfft2(F, s=(ny, nz), axes=(1, 2))
    else:
        g = cyc_axis(cyc_axis(f, 1), 2)
    return nak_x(g.reshape(nx, -1), x).reshape(nx + 4, ny, nz)


# ---------------------------------------------------------------- evaluation
def evaluate3(c, pts, h, nx, periodic_x):
    """value, gradient (n,3), Hessian (n,6: xx yy zz xy xz yz).  Cell size h on all axes; ny, nz = c.shape[1:]."""
    ny, nz = c.shape[1], c.shape[2]
    idx, W = [], []
    for a in range(3):
        n = nx if a == 0 else (ny if a == 1 else nz)
        x = pts[:, a]
        if a > 0 or periodic_x:
            L = n * h
            x = x - L * np.floor(x / L)
            x = np.where(x >= L, x - L, x)
            x = np.where(x < 0, x + L, x)
        t = x / h - 0.5
        i0 = np.floor(t).astype(np.int64)
        u = t - i0
        w, dw, d2w = weights(u)
        ii = np.stack([i0 - 1 + k for k in range(4)])
        if a == 0 and not periodic_x:
            assert i0.min() >= -1 and i0.max() <= nx - 1, "x outside [-h/2, L1 + h/2)"
            ii = ii + 2
        else:
            ii = np.mod(ii, n)
        idx.append(ii)
        W.append((w, dw / h, d2w / h**2))
    N = len(pts)
    v = np.zeros(N); g = np.zeros((N, 3)); H = np.zeros((N, 6))
    wx, wy, wz = W
    for p in range(4):
        for q in range(4):
            for r in range(4):
                cv = c[idx[0][p], idx[1][q], idx[2][r]]
                v += wx[0][p] * wy[0][q] * wz[0][r] * cv
                g[:, 0] += wx[1][p] * wy[0][q] * wz[0][r] * cv
                g[:, 1] += wx[0][p] * wy[1][q] * wz[0][r] * cv
                g[:, 2] += wx[0][p] * wy[0][q] * wz[1][r] * cv
                H[:, 0] += wx[2][p] * wy[0][q] * wz[0][r] * cv
                H[:, 1] += wx[0][p] * wy[2][q] * wz[0][r] * cv
                H[:, 2] += wx[0][p] * wy[0][q] * wz[2][r] * cv
                H[:, 3] += wx[1][p] * wy[1][q] * wz[0][r] * cv
                H[:, 4] += wx[1][p] * wy[0][q] * wz[1][r] * cv
                H[:, 5] += wx[0][p] * wy[1][q] * wz[1][r] * cv
    return v, g, H


def evaluate1(C, x, h, n):
    """1-D value, first, second derivative of the not-a-knot spline; C: (n+4,) coefficients c_{-2}..c_{n+1}."""
    t = x / h - 0.5
    i0 = np.floor(t).astype(np.int64)
    assert i0.min() >= -1 and i0.max() <= n - 1
    u = t - i0
    w, dw, d2w = weights(u)
    cc = np.stack([C[i0 - 1 + k + 2] for k in range(4)])
    return (w * cc).sum(0), (dw * cc).sum(0) / h, (d2w * cc).sum(0) / h**2


def gaussian_field(nx, ny, nz, h, ell, seed):
    """SF-18-like periodic Gaussian-covariance field, unit variance."""
    rng = np.random.default_rng(seed)
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=h)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=h)
    kz = 2 * np.pi * np.fft.rfftfreq(nz, d=h)
    k2 = kx[:, None, None] ** 2 + ky[None, :, None] ** 2 + kz[None, None, :] ** 2
    Yk = np.fft.rfftn(rng.standard_normal((nx, ny, nz))) * np.sqrt(np.exp(-k2 * ell**2 / 4.0))
    Y = np.fft.irfftn(Yk, s=(nx, ny, nz))
    Y /= Y.std()
    return Y
