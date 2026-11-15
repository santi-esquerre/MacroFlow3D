"""Definitions of the SF-43 per-realization plane estimators, written from decision G4 / the sf42red table contract.

Independent of every analysis package: standard library and numpy only (nothing from `sf43ana`, `mcstat`, `sf43camp`, `sf42red`).
Slopes use `math.fsum` (exact summation), a different arithmetic path from the reducer's numpy dot products.

Plane moments are dicts {n, mt, vt, mu: (m2, m3), C: (c22, c33, c23 or None)}, biased and weighted over the plane's own set; `None`
stands for an empty plane.  `c23 is None` means the source has no cross covariance (the tracker's moment CSV has only the diagonal).
"""
import math

NAN = float("nan")
REL_TOL = 1e-9


class Undefined(Exception):
    """The estimators are undefined (the reducer writes NaN estimates for the row)."""


def slope(x, y):
    """Ordinary least-squares slope of y on x."""
    n = len(x)
    if any(not math.isfinite(v) for v in list(x) + list(y)):
        raise Undefined("non-finite value in a slope")
    xm, ym = math.fsum(x) / n, math.fsum(y) / n
    sxx = math.fsum((a - xm) ** 2 for a in x)
    if not sxx > 0.0:
        raise Undefined("degenerate abscissa")
    return math.fsum((a - xm) * (b - ym) for a, b in zip(x, y)) / sxx


def window_planes(x1, x0, ell, d_lo, d_hi):
    """Indices of the planes with d = (x1 - x0)/ell in [d_lo, d_hi], relative tolerance 1e-9 (absolute below |bound| 1); >= 3."""
    if any(b <= a for a, b in zip(x1, x1[1:])):
        raise Undefined("planes not strictly increasing")
    tl, th = REL_TOL * max(1.0, abs(d_lo)), REL_TOL * max(1.0, abs(d_hi))
    idx = [k for k, x in enumerate(x1) if d_lo - tl <= (x - x0) / ell <= d_hi + th]
    if len(idx) < 3:
        raise Undefined("fewer than 3 planes in the window")
    return idx


def _unit(v):
    n = math.sqrt(math.fsum(c * c for c in v))
    if not (n > 0.0 and math.isfinite(n)):
        raise Undefined("degenerate direction")
    return [c / n for c in v]


def _dot(a, b):
    return math.fsum(p * q for p, q in zip(a, b))


def estimate(x1w, mom, ell):
    """Estimators on the window planes x1w with per-plane moments `mom`.  Returns {column: value} (alpha_TAP is NaN when the source
    has no cross covariance).  Raises Undefined where the reducer writes NaN estimates for the row.
    q = (x1_last - x1_first, mu_last - mu_first) over the window, e_L = q/|q|, tilt = atan2(|e_L x e_1|, e_L . e_1); the transverse
    pair is the Gram-Schmidt of e2, e3 against e_L; alpha_TP = slope[(var x2 + var x3)/2]/(2 ell); alpha_TAP the same with the
    variances along that pair; U_r = 1/slope[mean tau]; alpha_LP = U_r^2 slope[var tau]/(2 ell)."""
    if any(m is None for m in mom):
        raise Undefined("empty particle set at a window plane")
    eL = _unit([x1w[-1] - x1w[0], mom[-1]["mu"][0] - mom[0]["mu"][0], mom[-1]["mu"][1] - mom[0]["mu"][1]])
    tilt = math.atan2(math.hypot(eL[1], eL[2]), eL[0])
    e2, e3 = [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]
    u1 = _unit([a - _dot(e2, eL) * b for a, b in zip(e2, eL)])
    u2 = _unit([c - _dot(e3, eL) * b - _dot(e3, u1) * a for a, b, c in zip(u1, eL, e3)])
    sig = [(m["C"][0] + m["C"][1]) / 2.0 for m in mom]
    out = {"alpha_TP": 0.5 * slope(x1w, sig) / ell, "sigT2_first": sig[0], "sigT2_last": sig[-1], "tilt": tilt, "alpha_TAP": NAN}
    if all(m["C"][2] is not None for m in mom):
        def var_u(u, m):
            return u[1] * u[1] * m["C"][0] + 2.0 * u[1] * u[2] * m["C"][2] + u[2] * u[2] * m["C"][1]
        out["alpha_TAP"] = 0.5 * slope(x1w, [(var_u(u1, m) + var_u(u2, m)) / 2.0 for m in mom]) / ell
    s_mt = slope(x1w, [m["mt"] for m in mom])
    if not s_mt > 0.0:
        raise Undefined("non-positive slope of the mean crossing time")
    out["U_r"] = 1.0 / s_mt
    out["alpha_LP"] = 0.5 * out["U_r"] ** 2 * slope(x1w, [m["vt"] for m in mom]) / ell
    return out


def close(a, b, rtol, atol):
    """(ok, rel): NaN == NaN; otherwise |a-b| <= max(atol, rtol max(|a|,|b|)); rel = |a-b| / max(|a|, |b|, atol/rtol)."""
    if math.isnan(a) and math.isnan(b):
        return True, 0.0
    if math.isnan(a) or math.isnan(b):
        return False, math.inf
    d = abs(a - b)
    m = max(abs(a), abs(b))
    return d <= max(atol, rtol * m), (0.0 if d == 0.0 else d / max(m, atol / rtol))
