"""Shared helpers of the SF-41 pre-activation DESIGN emulations C2/C3/C4 (numpy; not SF-41 evidence).

Loaded by path from c2_tau.py / c3_shift.py / c4_lobe.py (python3 -I does not put the script
directory on sys.path).  Everything here mirrors, in numpy float64 without FMA, the conventions
of src/numerics/interpolation/PeriodicTricubicBSpline.cuh (uniform cubic B-spline, cell-centred
lattice x_i = (i + 1/2) h) and of the DP5(4) arclength stepper of
src/physics/particles/lagrangian_labels/LagrangianTrackerDevice.cuh.
"""
import math

import numpy as np

# ---------------------------------------------------------------------------------------------
# Splines: separable cubic B-spline, per axis 'per' (periodic, SF-28) or 'nak' (not-a-knot on the
# raw samples; unknowns c_{-1}..c_n; continuation c_{-2} = 4c_{-1}-6c_0+4c_1-c_2 and symmetric).
# Extended coefficient array per axis has n + 4 entries, index j + 2 for j = -2 .. n + 1.
# ---------------------------------------------------------------------------------------------


def axis_matrix(kind, n):
    """P (n+4, n): extended coefficients as a linear map of the n cell-centred samples."""
    if kind == "per":
        A = np.zeros((n, n))
        for i in range(n):
            A[i, (i - 1) % n] += 1.0 / 6.0
            A[i, i] += 4.0 / 6.0
            A[i, (i + 1) % n] += 1.0 / 6.0
        Ainv = np.linalg.inv(A)
        P = np.zeros((n + 4, n))
        for j in range(-2, n + 2):
            P[j + 2] = Ainv[j % n]
        return P
    assert kind == "nak"
    m = n + 2  # unknowns c_{-1}..c_n, index j + 1
    A = np.zeros((m, m))
    for i in range(n):  # interpolation rows
        A[i, i] = 1.0 / 6.0
        A[i, i + 1] = 4.0 / 6.0
        A[i, i + 2] = 1.0 / 6.0
    A[n, 0:5] = [-1.0, 4.0, -6.0, 4.0, -1.0]  # left not-a-knot row
    A[n + 1, m - 5:m] = [-1.0, 4.0, -6.0, 4.0, -1.0]  # right (mirror)
    B = np.zeros((m, n))
    B[np.arange(n), np.arange(n)] = 1.0  # rhs = f_i on rows 0..n-1, 0 on the two NAK rows
    S = np.linalg.solve(A, B)  # (m, n): c_{-1..n} from samples
    P = np.zeros((n + 4, n))
    P[1:n + 3] = S  # c_{-1}..c_n at ext index 1..n+2
    P[0] = 4 * S[0] - 6 * S[1] + 4 * S[2] - S[3]  # c_{-2}
    P[n + 3] = 4 * S[m - 1] - 6 * S[m - 2] + 4 * S[m - 3] - S[m - 4]  # c_{n+1}
    return P


class Spline:
    """N-D separable spline; axes = [(kind, n, h), ...]; samples of shape (n1, .., nd)."""

    def __init__(self, axes, samples):
        self.axes = axes
        c = np.asarray(samples, dtype=float)
        assert c.shape == tuple(a[1] for a in axes)
        for d, (kind, n, h) in enumerate(axes):
            P = axis_matrix(kind, n)
            c = np.moveaxis(np.tensordot(P, c, axes=([1], [d])), 0, d)
        self.ce = c

    def _stencil(self, d, x):
        kind, n, h = self.axes[d]
        x = np.asarray(x, dtype=float)
        if kind == "per":
            L = n * h
            xr = x - L * np.floor(x / L)
            xr = np.where(xr >= L, xr - L, xr)
            xr = np.where(xr < 0.0, xr + L, xr)
        else:
            xr = x
        t = xr / h - 0.5
        ft = np.floor(t)
        i0 = ft.astype(np.int64)
        assert i0.min() >= -1 and i0.max() <= n - 1, "evaluation outside [-h/2, L+h/2)"
        u = t - ft
        v = 1.0 - u
        u2 = u * u
        u3 = u2 * u
        sixth = 1.0 / 6.0
        w = np.stack([v * v * v * sixth, (3.0 * u3 - 6.0 * u2 + 4.0) * sixth,
                      (-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) * sixth, u3 * sixth], axis=-1)
        inv_h = 1.0 / h
        dw = np.stack([-0.5 * v * v * inv_h, 0.5 * (3.0 * u2 - 4.0 * u) * inv_h,
                       0.5 * (-3.0 * u2 + 2.0 * u + 1.0) * inv_h, 0.5 * u2 * inv_h], axis=-1)
        idx = i0[..., None] + np.arange(1, 5)  # ext index of c_{i0-1+a} is i0 + 1 + a
        return idx, w, dw

    def eval(self, pts):
        """pts: (P, nd) -> value (P,), gradient (P, nd)."""
        pts = np.asarray(pts, dtype=float)
        nd = len(self.axes)
        st = [self._stencil(d, pts[:, d]) for d in range(nd)]
        shp = [(-1,) + (1,) * d + (4,) + (1,) * (nd - 1 - d) for d in range(nd)]
        blk = self.ce[tuple(st[d][0].reshape(shp[d]) for d in range(nd))]  # (P, 4, .., 4)
        sub = "abc"[:nd]
        ops = ["p" + sub[d] for d in range(nd)]

        def contract(arrs):
            return np.einsum(",".join(ops) + ",p" + sub + "->p", *arrs, blk)

        val = contract([st[d][1] for d in range(nd)])
        grad = np.stack([contract([st[d][2 if d == g else 1] for d in range(nd)]) for g in range(nd)],
                        axis=-1)
        return val, grad


class Spline1D:
    """Pure-python scalar 1-D spline (fast path of the tracker / quadrature)."""

    def __init__(self, kind, n, h, samples):
        self.kind, self.n, self.h, self.L = kind, n, h, n * h
        self.ce = (axis_matrix(kind, n) @ np.asarray(samples, dtype=float)).tolist()

    def __call__(self, x):
        h = self.h
        if self.kind == "per":
            L = self.L
            xr = x - L * math.floor(x / L)
            if xr >= L:
                xr -= L
            if xr < 0.0:
                xr += L
        else:
            xr = x
        t = xr / h - 0.5
        ft = math.floor(t)
        i0 = int(ft)
        assert -1 <= i0 <= self.n - 1, "evaluation outside [-h/2, L+h/2)"
        u = t - ft
        v = 1.0 - u
        u2 = u * u
        u3 = u2 * u
        s6 = 1.0 / 6.0
        c0, c1, c2, c3 = self.ce[i0 + 1:i0 + 5]
        val = (c0 * (v * v * v * s6) + c1 * ((3.0 * u3 - 6.0 * u2 + 4.0) * s6)
               + c2 * ((-3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0) * s6) + c3 * (u3 * s6))
        d = (c0 * (-0.5 * v * v) + c1 * (0.5 * (3.0 * u2 - 4.0 * u))
             + c2 * (0.5 * (-3.0 * u2 + 2.0 * u + 1.0)) + c3 * (0.5 * u2)) / h
        return val, d


# ---------------------------------------------------------------------------------------------
# DP5(4) (Dormand-Prince) in the exact operation order of dp5_step (state arrays (P, D)).
# ---------------------------------------------------------------------------------------------
c2, c3, c4, c5 = 1 / 5, 3 / 10, 4 / 5, 8 / 9
a21 = 1 / 5
a31, a32 = 3 / 40, 9 / 40
a41, a42, a43 = 44 / 45, -56 / 15, 32 / 9
a51, a52, a53, a54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
a61, a62, a63, a64, a65 = 9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656
b1, b3, b4, b5, b6 = 35 / 384, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84
e1, e3, e4, e5, e6, e7 = (71 / 57600, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40)


def dp5_step(rhs, t, t_end, y, k1, h, n_err, tol):
    """One DP5(4) step; rhs(t, y) -> dy (P, D).  Returns y5, k7, err (P,)."""
    k2 = rhs(t + c2 * h, y + h * (a21 * k1))
    k3 = rhs(t + c3 * h, y + h * (a31 * k1 + a32 * k2))
    k4 = rhs(t + c4 * h, y + h * (a41 * k1 + a42 * k2 + a43 * k3))
    k5 = rhs(t + c5 * h, y + h * (a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4))
    k6 = rhs(t_end, y + h * (a61 * k1 + a62 * k2 + a63 * k3 + a64 * k4 + a65 * k5))
    y5 = y + h * (b1 * k1 + b3 * k3 + b4 * k4 + b5 * k5 + b6 * k6)
    k7 = rhs(t_end, y5)
    e = h * (e1 * k1 + e3 * k3 + e4 * k4 + e5 * k5 + e6 * k6 + e7 * k7)
    err = np.max(np.abs(e[..., :n_err]), axis=-1) / tol if n_err > 0 else np.zeros(y.shape[:-1])
    return y5, k7, err


def step_factor(err):
    err = np.asarray(err, dtype=float)
    with np.errstate(divide="ignore"):
        f = np.minimum(5.0, np.maximum(0.2, 0.9 * np.power(np.where(err > 0.0, err, 1.0), -0.2)))
    return np.where(err > 0.0, f, 5.0)


# ---------------------------------------------------------------------------------------------
# Exact discrete heads of the layered medium (harmonic face K, 1-D, potential Phi = G1 x + f).
# ---------------------------------------------------------------------------------------------
def harm(a, b):
    return 2.0 / (1.0 / a + 1.0 / b)


def layered_f_periodic(lnk, h, G1=1.0):
    """Periodic medium (a): f_i with constant face flux q = G1 / mean(1/Kf); zero-mean gauge."""
    K = np.exp(lnk)
    Kf = harm(K, np.roll(K, -1))  # face i+1/2
    q = G1 / np.mean(1.0 / Kf)
    df = h * (q / Kf - G1)  # f_{i+1} - f_i
    f = np.concatenate([[0.0], np.cumsum(df[:-1])])
    return f - f.mean(), q


def layered_f_dirichlet(lnk, h, G1=1.0):
    """Medium (b): west head G1 L1, east head 0, Dirichlet faces at h/2 with coefficient 2 K_cell.

    Returns f_i = -H_i - G1 x_i + G1 L1 (cell centres) and the constant face flux q."""
    n = len(lnk)
    L = n * h
    K = np.exp(lnk)
    Kf = harm(K[:-1], K[1:])
    q = G1 * L / (h / (2 * K[0]) + np.sum(h / Kf) + h / (2 * K[-1]))
    H = np.empty(n)
    H[0] = G1 * L - q * h / (2 * K[0])
    for i in range(n - 1):
        H[i + 1] = H[i] - q * h / Kf[i]
    assert abs(H[-1] - q * h / (2 * K[-1])) < 1e-12 * G1 * L  # east Dirichlet closes the chain
    x = (np.arange(n) + 0.5) * h
    return -H - G1 * x + G1 * L, q


# ---------------------------------------------------------------------------------------------
# Faithful scalar plane pass (sequential planes, Henon landing) for a layered (1-D) field.
# field(x) -> (g1, k).  State (x1, tau); positions-only error norm (x1).
# ---------------------------------------------------------------------------------------------
def plane_pass_1d(field, planes, h_max, tol, min_step=1e-14, max_arclength=1e30):
    def rhs_arc(s, y):
        g1, k = field(float(y[0, 0]))
        assert g1 > 0.0 and k > 0.0
        gn = math.sqrt(g1 * g1)
        return np.array([[g1 / gn, 1.0 / (k * gn)]])

    def rhs_land(x1, y):  # y = (s, tau); independent variable x1
        g1, k = field(float(x1))
        assert g1 > 0.0, "backflow in landing"
        gn = math.sqrt(g1 * g1)
        return np.array([[gn / g1, 1.0 / (k * g1)]])

    def land(x1a, ya, X):
        t, y = x1a, ya.copy()
        k1 = rhs_land(t, y)
        dt_mag = min(h_max, abs(X - t))
        while True:
            remaining = X - t
            last = dt_mag >= abs(remaining)
            dt = remaining if last else dt_mag
            t_end = X if last else t + dt
            y5, k7, err = dp5_step(rhs_land, t, t_end, y, k1, dt, 0, tol)
            fac = step_factor(err)[0]
            if err[0] <= 1.0:
                y, k1 = y5, k7
                if last:
                    return y
                t = t_end
                dt_mag = min(h_max, abs(dt) * fac)
            else:
                dt_mag = abs(dt) * fac
                assert dt_mag >= min_step

    y = np.array([[0.0, 0.0]])
    s = 0.0
    f = rhs_arc(0.0, y)
    out = []
    m = 0
    X = planes[0]
    h = 0.1 * h_max
    nsteps = 0
    while True:
        y5, k7, err = dp5_step(rhs_arc, s, s + h, y, f, h, 1, tol)
        fac = step_factor(err)[0]
        if not err[0] <= 1.0:
            h = h * fac
            assert h >= min_step
            continue
        if 1.0 * (y5[0, 0] - X) >= 0.0:
            yl = land(y[0, 0], np.array([[s, y[0, 1]]]), X)
            y = np.array([[X, yl[0, 1]]])
            s = yl[0, 0]
            out.append(float(y[0, 1]))
            f = rhs_arc(s, y)
            if m == len(planes) - 1:
                return out, nsteps
            m += 1
            X = planes[m]
            h = min(h_max, h * fac)
            continue
        nsteps += 1
        y, f = y5, k7
        s += h
        h = min(h_max, h * fac)


def tau_quadrature(field_g1k, a, b, h):
    """Independent adaptive quadrature of 1/v1 on [a, b], split at the half-cell lattice
    (all spline breakpoints lie on it); scipy QUADPACK, tight tolerances per piece."""
    from scipy.integrate import quad

    def inv_v(x):
        g1, k = field_g1k(x)
        return 1.0 / (k * g1)

    step = 0.5 * h
    x = a
    j = math.floor(a / step) + 1
    total = 0.0
    while x < b:
        nxt = min(b, j * step)
        if nxt > x:
            val, _ = quad(inv_v, x, nxt, epsabs=1e-15, epsrel=1e-14, limit=200)
            total += val
        x = nxt
        j += 1
    return total
