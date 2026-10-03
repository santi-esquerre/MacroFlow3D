"""SF-29 shared metrics and the one-line parseable CASE print format.

Vertex grid: x1 = j/N (j = 0..N, axis 0), x2 = m2/N, x3 = m3/N (periodic, m = 0..N-1).
Labels are psi1 = x2 + u1, psi2 = x3 + u2 with u_i periodic in (x2, x3); all FD operate on the
periodic parts plus the exact affine gradient, so the label jumps never enter a difference.

Second-order finite differences: centered in x2, x3 (periodic); in x1 centered on planes
1..N-1 and second-order one-sided on the inlet (forward) and outlet (backward) planes.

Metrics (SF-29 UNDERSTAND record section 3):
  e_v   = RMS(c - v_D) / RMS(v_D),        c = grad_h psi1 x grad_h psi2   (RMS over vector norms)
  e_psi = max_i RMS(psi_i - psi_i^or) / RMS(psi_i^or - affine_i)
          (if RMS(psi_i^or - affine_i) <= 1e-12 x the other label's, e.g. control2d psi2 = x3, that
           label is normalized by the larger periodic part of the pair; k = 1: absolute RMS)
  e_i   = RMS(v_D . grad_h psi_i) / (RMS|v_D| RMS|grad_h psi_i|)
  e_div = RMS(div_h c) / RMS(v_D)
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


def label_gradients(psi1, psi2):
    """grad_h psi_i (tuples of 3 arrays) from vertex labels (periodic parts differenced)."""
    N = psi1.shape[1]; h = 1.0 / N
    u1, u2 = periodic_parts(psi1, psi2)
    g1 = list(grad_fd(u1, h)); g1[1] = g1[1] + 1.0
    g2 = list(grad_fd(u2, h)); g2[2] = g2[2] + 1.0
    return tuple(g1), tuple(g2)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def rms_vec(v):
    return float(np.sqrt(np.mean(v[0]**2 + v[1]**2 + v[2]**2)))


def fd_metrics(psi1, psi2, vD, psi_or=None):
    """Metrics of vertex labels against the reference velocity vD (3 arrays) and oracle labels."""
    N = psi1.shape[1]; h = 1.0 / N
    g1, g2 = label_gradients(psi1, psi2)
    c = cross(g1, g2)
    vrms = rms_vec(vD)
    out = {}
    out["e_v"] = rms_vec(tuple(c[i] - vD[i] for i in range(3))) / vrms
    for name, g in (("e_i1", g1), ("e_i2", g2)):
        out[name] = rms(vD[0] * g[0] + vD[1] * g[1] + vD[2] * g[2]) / (vrms * rms_vec(g))
    div = d1_fd(c[0], h) + dp_fd(c[1], h, 1) + dp_fd(c[2], h, 2)
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
    if psi_or is not None:
        a = affine(N)
        den = [rms(psi_or[i] - a[i]) for i in range(2)]
        e = []
        for i in range(2):
            p = (psi1, psi2)[i]
            # a label whose oracle periodic part vanishes (control2d: psi2 = x3 exactly) is normalized
            # by the larger periodic part of the pair instead of by ~0 (stated in the docstring)
            d = den[i] if den[i] > 1e-12 * max(den) else max(den)
            e.append(rms(p - psi_or[i]) / d if d > 0 else rms(p - psi_or[i]))
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
    """The one-line parseable metrics format shared by every SF-29 script."""
    return ("CASE field=%s eps=%g N=%d cand=%s | r_F=%s its=%s | e_v=%s e_psi=%s e_i=(%s,%s) e_div=%s "
            "min_c=%s p0.1=%s p1=%s p5=%s p50=%s | t=%.1f"
            % (field, eps, N, cand, _f(r_F), _f(its), _f(m.get("e_v")), _f(m.get("e_psi")), _f(m.get("e_i1")),
               _f(m.get("e_i2")), _f(m.get("e_div")), _f(m.get("min_c")), _f(m.get("p0.1")), _f(m.get("p1")),
               _f(m.get("p5")), _f(m.get("p50")), t))


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
