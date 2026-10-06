"""SF-29 shared metrics and the one-line parseable CASE print format.

Vertex grid: x1 = j/N (j = 0..N, axis 0), x2 = m2/N, x3 = m3/N (periodic, m = 0..N-1).
Labels are psi1 = x2 + u1, psi2 = x3 + u2 with u_i periodic in (x2, x3); all FD operate on the
periodic parts plus the exact affine gradient, so the label jumps never enter a difference.

Finite differences (`order` argument of label_gradients / fd_metrics):
  order=2 (default, unchanged since N1): centered in x2, x3 (periodic); in x1 centered on planes
          1..N-1 and second-order one-sided on the inlet (forward) and outlet (backward) planes.
  order=4 (C-i4): centered 4th order in x2, x3 (-u_{+2} + 8u_{+1} - 8u_{-1} + u_{-2})/(12h); in x1
          centered 4th order on planes 2..N-2 and 5-point 4th-order skewed (planes 1, N-1) or
          one-sided (planes 0, N) stencils (Fornberg weights, fd_weights below).

Metrics (SF-29 UNDERSTAND record section 3; e_psi normalization corrected in C-i4):
  e_v    = RMS(c - v_D) / RMS(v_D),        c = grad_h psi1 x grad_h psi2   (RMS over vector norms)
  a_psii = RMS(psi_i - psi_i^or)                                  (absolute, per label)
  e_psii = a_psii / den(i),  den_i = RMS(psi_i^or - affine_i), den_ref = max(den_1, den_2):
           den(i) = den_i if den_i > 1e-6 den_ref, else den_ref    (a label whose oracle periodic part
           is negligible -- control2d: psi2 = Q0 x3, periodic part ~ (Q0 - 1) ~ 1e-13 -- is normalized
           by the other label's; k = 1, den_ref = 0: absolute RMS).  Before C-i4 the threshold was
           1e-12 den_ref, which control2d (den ratio ~1e-11) passed, so e_psi was roundoff / roundoff.
  e_psi  = max(e_psi1, e_psi2)
  e_i    = RMS(v_D . grad_h psi_i) / (RMS|v_D| RMS|grad_h psi_i|)
  e_div  = RMS(div_h c) / RMS(v_D)            (div_h with the same `order`)
  min_c, p0.1, p1, p5, p50: min and percentiles of |c| (the same of |v_D| are printed by callers).
No regularization of |c| anywhere.
"""
import numpy as np


def affine(N):
    x = np.arange(N) / float(N)
    X2 = np.broadcast_to(x[None, :, None], (N + 1, N, N))
    X3 = np.broadcast_to(x[None, None, :], (N + 1, N, N))
    return X2, X3


def periodic_parts(psi1, psi2):
    N = psi1.shape[1]
    X2, X3 = affine(N)
    return psi1 - X2, psi2 - X3


def d1_fd(u, h):
    d = np.empty_like(u)
    d[1:-1] = (u[2:] - u[:-2]) / (2 * h)
    d[0] = (-3 * u[0] + 4 * u[1] - u[2]) / (2 * h)
    d[-1] = (3 * u[-1] - 4 * u[-2] + u[-3]) / (2 * h)
    return d


def dp_fd(u, h, axis):
    return (np.roll(u, -1, axis=axis) - np.roll(u, 1, axis=axis)) / (2 * h)


def grad_fd(u, h):
    return d1_fd(u, h), dp_fd(u, h, 1), dp_fd(u, h, 2)


def fd_weights(z, x0, m):
    """Fornberg (1988) finite-difference weights: w[k, i] approximates d^k/dx^k at x0 by sum_i w[k, i] f(z[i]),
    k = 0..m.  Exact for polynomials of degree < len(z)."""
    z = np.asarray(z, dtype=float)
    n = z.size
    c = np.zeros((m + 1, n))
    c1 = 1.0; c4 = z[0] - x0
    c[0, 0] = 1.0
    for i in range(1, n):
        mn = min(i, m); c2 = 1.0; c5 = c4; c4 = z[i] - x0
        for j in range(i):
            c3 = z[i] - z[j]; c2 = c2 * c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[k, i] = c1 * (k * c[k - 1, i - 1] - c5 * c[k, i - 1]) / c2
                c[0, i] = -c1 * c5 * c[0, i - 1] / c2
            for k in range(mn, 0, -1):
                c[k, j] = (c4 * c[k, j] - k * c[k - 1, j]) / c3
            c[0, j] = c4 * c[0, j] / c3
        c1 = c2
    return c


# 4th-order centered weights (offsets -2..2), first and second derivative (times h, h^2)
C4_D1 = np.array([1.0, -8.0, 0.0, 8.0, -1.0]) / 12.0
C4_D2 = np.array([-1.0, 16.0, -30.0, 16.0, -1.0]) / 12.0


def x1_stencils4(N, deriv):
    """4th-order x1 stencils on the vertex planes 0..N for derivative `deriv` (1 or 2).

    Returns {plane: (first_plane, weights * h^deriv)} for the boundary planes (0, 1, N-1, N) and the centered
    weights for planes 2..N-2.  d1: 5 points (planes 0..4 for 0, 1; N-4..N for N-1, N); d11: 6 points
    (planes 0..5 / N-5..N), exact for polynomials up to degree 4 / 5."""
    npts = 5 if deriv == 1 else 6
    out = {}
    for p in (0, 1):
        z = np.arange(npts, dtype=float)
        out[p] = (0, fd_weights(z, float(p), deriv)[deriv])
    for p in (N - 1, N):
        z = np.arange(N - npts + 1, N + 1, dtype=float)
        out[p] = (N - npts + 1, fd_weights(z - (N - npts + 1), float(p - (N - npts + 1)), deriv)[deriv])
    return out, (C4_D1 if deriv == 1 else C4_D2)


def d1_fd4(u, h):
    """4th-order x1 derivative of an (N+1, ...) array on all planes 0..N (see x1_stencils4)."""
    N = u.shape[0] - 1
    bnd, cw = x1_stencils4(N, 1)
    d = np.empty_like(u)
    d[2:N - 1] = sum(cw[k] * u[k:N - 3 + k] for k in range(5))
    for p, (s, w) in bnd.items():
        d[p] = sum(w[k] * u[s + k] for k in range(w.size))
    return d / h


def dp_fd4(u, h, axis):
    """4th-order centered periodic first derivative along `axis`."""
    return (-np.roll(u, -2, axis=axis) + 8 * np.roll(u, -1, axis=axis) - 8 * np.roll(u, 1, axis=axis)
            + np.roll(u, 2, axis=axis)) / (12 * h)


def grad_fd4(u, h):
    return d1_fd4(u, h), dp_fd4(u, h, 1), dp_fd4(u, h, 2)


def label_gradients(psi1, psi2, order=2):
    """grad_h psi_i (tuples of 3 arrays) from vertex labels (periodic parts differenced); order 2 or 4."""
    N = psi1.shape[1]; h = 1.0 / N
    u1, u2 = periodic_parts(psi1, psi2)
    gf = grad_fd if order == 2 else grad_fd4
    g1 = list(gf(u1, h)); g1[1] = g1[1] + 1.0
    g2 = list(gf(u2, h)); g2[2] = g2[2] + 1.0
    return tuple(g1), tuple(g2)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def rms_vec(v):
    return float(np.sqrt(np.mean(v[0]**2 + v[1]**2 + v[2]**2)))


E_PSI_DEN_REL = 1e-6     # a label with den_i <= E_PSI_DEN_REL * den_ref is normalized by den_ref (module docstring)


def e_psi_parts(psi1, psi2, psi_or):
    """Per-label label errors: (e_psi1, e_psi2), (a_psi1, a_psi2), (den used 1, den used 2)."""
    N = psi1.shape[1]
    a = affine(N)
    den = [rms(psi_or[i] - a[i]) for i in range(2)]
    dref = max(den)
    e = []; ab = []; used = []
    for i in range(2):
        p = (psi1, psi2)[i]
        d = den[i] if den[i] > E_PSI_DEN_REL * dref else dref
        ai = rms(p - psi_or[i])
        ab.append(ai); used.append(d)
        e.append(ai / d if d > 0 else ai)
    return tuple(e), tuple(ab), tuple(used)


def fd_metrics(psi1, psi2, vD, psi_or=None, order=2):
    """Metrics of vertex labels against the reference velocity vD (3 arrays) and oracle labels.
    order = 2 (default) or 4: finite-difference order of grad_h psi_i (hence c, e_v, e_i) and of div_h."""
    if order not in (2, 4):
        raise ValueError("order must be 2 or 4")
    N = psi1.shape[1]; h = 1.0 / N
    g1, g2 = label_gradients(psi1, psi2, order)
    c = cross(g1, g2)
    vrms = rms_vec(vD)
    out = {}
    out["e_v"] = rms_vec(tuple(c[i] - vD[i] for i in range(3))) / vrms
    for name, g in (("e_i1", g1), ("e_i2", g2)):
        out[name] = rms(vD[0] * g[0] + vD[1] * g[1] + vD[2] * g[2]) / (vrms * rms_vec(g))
    if order == 2:
        div = d1_fd(c[0], h) + dp_fd(c[1], h, 1) + dp_fd(c[2], h, 2)
    else:
        div = d1_fd4(c[0], h) + dp_fd4(c[1], h, 1) + dp_fd4(c[2], h, 2)
    out["e_div"] = rms(div) / vrms
    cn = np.sqrt(c[0]**2 + c[1]**2 + c[2]**2)
    vn = np.sqrt(vD[0]**2 + vD[1]**2 + vD[2]**2)
    out["min_c"] = float(cn.min())
    pc = np.percentile(cn, [0.1, 1.0, 5.0, 50.0])
    pv = np.percentile(vn, [0.1, 1.0, 5.0, 50.0])
    out["p0.1"], out["p1"], out["p5"], out["p50"] = (float(x) for x in pc)
    out["vD_min"] = float(vn.min())
    out["vD_p"] = tuple(float(x) for x in pv)
    out["v_rms"] = vrms
    out["order"] = order
    if psi_or is not None:
        e, ab, used = e_psi_parts(psi1, psi2, psi_or)
        out["e_psi1"], out["e_psi2"] = e
        out["a_psi1"], out["a_psi2"] = ab
        out["den_psi"] = used
        out["e_psi"] = max(e)
    else:
        out["e_psi"] = float("nan")
    return out


def _f(x):
    if x is None:
        return "nan"
    if isinstance(x, (int, np.integer)):
        return "%d" % x
    return "%.3e" % x


def case_line(field, eps, N, cand, m, r_F=None, its=None, t=0.0):
    """The one-line parseable metrics format shared by every SF-29 script.  Since C-i4 the per-label errors are
    appended after the `t=` field (` | e_psi1=<..> e_psi2=<..>`) when the metrics dict has them; lines without
    the suffix (all earlier outputs) parse unchanged."""
    line = ("CASE field=%s eps=%g N=%d cand=%s | r_F=%s its=%s | e_v=%s e_psi=%s e_i=(%s,%s) e_div=%s "
            "min_c=%s p0.1=%s p1=%s p5=%s p50=%s | t=%.1f"
            % (field, eps, N, cand, _f(r_F), _f(its), _f(m.get("e_v")), _f(m.get("e_psi")), _f(m.get("e_i1")),
               _f(m.get("e_i2")), _f(m.get("e_div")), _f(m.get("min_c")), _f(m.get("p0.1")), _f(m.get("p1")),
               _f(m.get("p5")), _f(m.get("p50")), t))
    if m.get("e_psi1") is not None and m.get("e_psi2") is not None:
        line += " | e_psi1=%s e_psi2=%s" % (_f(m["e_psi1"]), _f(m["e_psi2"]))
    return line


def print_case(field, eps, N, cand, m, r_F=None, its=None, t=0.0):
    print(case_line(field, eps, N, cand, m, r_F, its, t), flush=True)


def parse_case_line(line):
    """Inverse of case_line -> dict (strings converted to float where possible)."""
    if not line.startswith("CASE "):
        raise ValueError("not a CASE line")
    out = {}
    for tok in line[5:].replace("|", " ").split():
        if "=" not in tok:
            continue
        k, v = tok.split("=", 1)
        if v.startswith("("):
            out[k] = tuple(float(x) for x in v.strip("()").split(","))
            continue
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def orders(errs, Ns):
    """Observed orders p = log(e_c/e_f)/log(N_f/N_c) over consecutive grids."""
    return [float(np.log(errs[i] / errs[i + 1]) / np.log(Ns[i + 1] / float(Ns[i]))) for i in range(len(Ns) - 1)]
