"""SF-39 C-STATS part 1: intervals, equivalence, margins and estimation helpers.

Pure functions on numpy arrays / floats (preregistration.md sections 4, 5, 9; decisions F4-F12 of
2026-10-27).  No file I/O.  Student-t is primary; BCa (B = 5000, a fresh
``numpy.random.default_rng(20261027)`` per call) is a sensitivity.  Alpha units: lambda.
"""
from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
from scipy import stats as _st

BCA_SEED = 20261027
BCA_RESAMPLES = 5000
NUM_CUT = 2.5e-3  # physical-bounded / numerical cut (units of lambda), also the margin floor scale


# ----------------------------------------------------------------------------------------------
# Student / Welch intervals
# ----------------------------------------------------------------------------------------------
def _tcrit(level: float, sided: str, dof: float) -> float:
    if sided == "two":
        return float(_st.t.ppf(0.5 * (1.0 + level), dof))
    if sided in ("upper", "lower"):
        return float(_st.t.ppf(level, dof))
    raise ValueError(f"sided must be 'two'|'upper'|'lower', got {sided!r}")


def _bounds(est: float, half: float, sided: str):
    if sided == "two":
        return est - half, est + half
    if sided == "upper":  # upper confidence bound: (-inf, hi]
        return -np.inf, est + half
    return est - half, np.inf  # 'lower': [lo, inf)


def t_interval(x, level: float = 0.90, sided: str = "two"):
    """Student t interval on the mean: (mean, lo, hi, n, sd), n - 1 d.o.f.; NaN bounds if n < 2."""
    x = np.asarray(x, dtype=float)
    n = int(x.size)
    if n < 2:
        return (float(x.mean()) if n else np.nan), np.nan, np.nan, n, np.nan
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    lo, hi = _bounds(mean, _tcrit(level, sided, n - 1) * sd / np.sqrt(n), sided)
    return mean, float(lo), float(hi), n, sd


def welch_interval(a, b, level: float = 0.90):
    """Difference of means a - b, Welch-Satterthwaite d.o.f.: (diff, lo, hi, nu, se)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    na, nb = a.size, b.size
    if na < 2 or nb < 2:
        return float(a.mean() - b.mean()), np.nan, np.nan, np.nan, np.nan
    va, vb = a.var(ddof=1) / na, b.var(ddof=1) / nb
    diff = float(a.mean() - b.mean())
    se = float(np.sqrt(va + vb))
    if se == 0.0:
        return diff, diff, diff, np.nan, 0.0
    nu = (va + vb) ** 2 / (va**2 / (na - 1) + vb**2 / (nb - 1))
    half = _tcrit(level, "two", nu) * se
    return diff, diff - half, diff + half, float(nu), se


def paired_interval(x, y, level: float = 0.90, sided: str = "two"):
    """Paired difference x - y (aligned arrays) as one t-sample."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != y.shape:
        raise ValueError("paired arrays must have the same shape (align by seed first)")
    return t_interval(x - y, level, sided)


def align_by_seed(seeds_a, vals_a, seeds_b, vals_b):
    """Restrict two seed-labelled arrays to their common seeds, ordered by seed: (seeds, a, b)."""
    pa = {int(s): i for i, s in enumerate(seeds_a)}
    pb = {int(s): i for i, s in enumerate(seeds_b)}
    common = sorted(pa.keys() & pb.keys())
    va, vb = np.asarray(vals_a, float), np.asarray(vals_b, float)
    return (np.array(common), va[[pa[s] for s in common]], vb[[pb[s] for s in common]])


# ----------------------------------------------------------------------------------------------
# ANCOVA (Gate 5 sensitivity)
# ----------------------------------------------------------------------------------------------
class Ancova(NamedTuple):
    a: float      # adjusted mean = intercept at c = 0 (sigma^2 = nominal)
    se_a: float
    dof: int      # n - 2
    b: float      # slope
    s_e: float
    n: int


def ancova_adjusted(alpha_par2, sigma2_par2, sigma2_nominal: float) -> Ancova:
    """OLS alpha = a + b c + e, c = sigma2 - nominal; SE_a = s_e sqrt(1/n + cbar^2/S_cc)."""
    y = np.asarray(alpha_par2, float)
    c = np.asarray(sigma2_par2, float) - sigma2_nominal
    n = y.size
    if n < 3:
        raise ValueError("ANCOVA needs n >= 3")
    cbar, ybar = c.mean(), y.mean()
    scc = float(((c - cbar) ** 2).sum())
    if scc == 0.0:
        raise ValueError("ANCOVA covariate is constant")
    b = float(((c - cbar) * (y - ybar)).sum() / scc)
    a = float(ybar - b * cbar)
    resid = y - a - b * c
    s_e = float(np.sqrt((resid**2).sum() / (n - 2)))
    se_a = s_e * float(np.sqrt(1.0 / n + cbar**2 / scc))
    return Ancova(a, se_a, n - 2, b, s_e, n)


def ancova_vs(arm_L, alpha_par2, sigma2_par2, sigma2_nominal: float, level: float = 0.90):
    """Interval of (mean(arm_L) - a): (diff, lo, hi, nu, anc)."""
    anc = ancova_adjusted(alpha_par2, sigma2_par2, sigma2_nominal)
    L = np.asarray(arm_L, float)
    nL = L.size
    if nL < 2:
        return float(L.mean() - anc.a), np.nan, np.nan, np.nan, anc
    vL = L.var(ddof=1) / nL
    va = anc.se_a**2
    diff = float(L.mean() - anc.a)
    se = np.sqrt(va + vL)
    if se == 0.0:
        return diff, diff, diff, np.nan, anc
    nu = (va + vL) ** 2 / (va**2 / anc.dof + vL**2 / (nL - 1))
    half = _tcrit(level, "two", nu) * se
    return diff, diff - half, diff + half, float(nu), anc


# ----------------------------------------------------------------------------------------------
# BCa bootstrap (sensitivity)
# ----------------------------------------------------------------------------------------------
_ALT = {"two": "two-sided", "upper": "less", "lower": "greater"}  # 'upper' = upper confidence bound


def _bca(data, statistic, level, sided):
    """(lo, hi, degenerate).  Degenerate / failing input never raises: NaN bounds and flag True."""
    nan = (np.nan, np.nan, True)
    data = tuple(np.asarray(d) for d in data)
    if any(d.size < 2 for d in data):
        return nan
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with np.errstate(all="ignore"):
                res = _st.bootstrap(
                    data, statistic, vectorized=True, axis=-1, paired=False,
                    n_resamples=BCA_RESAMPLES, confidence_level=level, method="BCa",
                    alternative=_ALT[sided], random_state=np.random.default_rng(BCA_SEED))
        dist = np.asarray(res.bootstrap_distribution)
        lo, hi = float(res.confidence_interval.low), float(res.confidence_interval.high)
    except Exception:  # noqa: BLE001 - contract: never raise on degenerate input
        return nan
    if np.ptp(dist) == 0 or np.isnan(lo) or np.isnan(hi) or any("egenerate" in str(w.message) for w in caught):
        return nan
    return lo, hi, False


def bca_interval(sample, level: float = 0.90, sided: str = "two"):
    """BCa of the mean of one sample (a paired quantity is passed as its per-realization difference)."""
    return _bca((sample,), lambda x, axis: np.mean(x, axis=axis), level, sided)


def bca_welch(a, b, level: float = 0.90, sided: str = "two"):
    """BCa of mean(a) - mean(b), two independent samples."""
    return _bca((a, b), lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis), level, sided)


def _ols_intercept(idx, c, y):
    cc, yy = c[idx], y[idx]
    cm, ym = cc.mean(-1), yy.mean(-1)
    scc = ((cc - cm[..., None]) ** 2).sum(-1)
    b = ((cc - cm[..., None]) * (yy - ym[..., None])).sum(-1) / scc
    return ym - b * cm


def bca_ancova_index(alpha_par2, sigma2_par2, sigma2_nominal: float, level: float = 0.90, sided: str = "two"):
    """BCa of the ANCOVA adjusted mean a, resampling (alpha_r, sigma2_r) jointly (index sample)."""
    y = np.asarray(alpha_par2, float)
    c = np.asarray(sigma2_par2, float) - sigma2_nominal
    return _bca((np.arange(y.size),), lambda idx, axis: _ols_intercept(idx, c, y), level, sided)


def bca_ancova_vs(arm_L, alpha_par2, sigma2_par2, sigma2_nominal: float, level: float = 0.90, sided: str = "two"):
    """BCa of mean(arm_L) - a: Par2 pairs resampled jointly (index sample), arm_L independent."""
    y = np.asarray(alpha_par2, float)
    c = np.asarray(sigma2_par2, float) - sigma2_nominal
    return _bca((np.arange(y.size), np.asarray(arm_L, float)),
                lambda idx, L, axis: np.mean(L, axis=axis) - _ols_intercept(idx, c, y), level, sided)


# ----------------------------------------------------------------------------------------------
# Margins, equivalence, half-width
# ----------------------------------------------------------------------------------------------
def margin_transverse(alpha_mean: float) -> float:
    return max(0.05 * abs(alpha_mean), 1.25e-3)


def margin_lam(alpha_mean: float) -> float:
    return 0.025 * alpha_mean


def margin(est: str, alpha_mean: float) -> float:
    """A margin of estimator est in {'TAM', 'TP', 'LAM'} at the given ensemble mean."""
    if est in ("TAM", "TP"):
        return margin_transverse(alpha_mean)
    if est == "LAM":
        return margin_lam(alpha_mean)
    raise ValueError(f"unknown estimator {est!r}")


def m_h(est: str, mean_2x: float, mean_prod: float) -> float:
    """h-ladder margin: min of the margins at the 2x-rung mean and at the production mean."""
    return min(margin(est, mean_2x), margin(est, mean_prod))


_LABELS = {"verdict": ("converged", "not converged", "inconclusive"),
           "gate5": ("agree", "disagree", "inconclusive")}


def equivalence(ci_lo: float, ci_hi: float, m: float, kind: str = "verdict") -> str:
    """inside [-m, m] (closed) -> first label; entirely outside -> second; else (also NaN) -> third."""
    inside, outside, other = _LABELS[kind]
    if ci_lo >= -m and ci_hi <= m:
        return inside
    if ci_lo > m or ci_hi < -m:
        return outside
    return other


def halfwidth_target(est: str, alpha_mean: float) -> float:
    if est in ("TAM", "TP"):
        return max(0.1 * abs(alpha_mean), 2.5e-3)
    if est == "LAM":
        return 0.05 * alpha_mean
    raise ValueError(f"unknown estimator {est!r}")


def halfwidth_check(x, target: float, level: float = 0.95):
    """B: 95 % two-sided t half-width of the mean vs target: (mean, halfwidth, target, met)."""
    mean, lo, hi, n, sd = t_interval(x, level, "two")
    hw = 0.5 * (hi - lo)
    return mean, hw, target, bool(hw <= target)


# ----------------------------------------------------------------------------------------------
# Reporting-only helpers
# ----------------------------------------------------------------------------------------------
def richardson_guard(q_half: float, q_prod: float, q_2x: float):
    """Order-2 extrapolation of the production error, only under the specification's guard.

    Rungs (means): 1/2 rung q_half, production q_prod, 2x rung q_2x.  D_c = q_prod - q_half,
    D_f = q_2x - q_prod.  Observed p = log2(D_c / D_f) when monotone (D_c D_f > 0); applied iff
    p in [1.5, 2.5], then the production error is -D_f 2^2/(2^2-1) (Q_prod - Q_exact for e ~ h^2).
    Otherwise only the bound |D_f| is reported.  Reporting only, never a correction.
    """
    dc, df = q_prod - q_half, q_2x - q_prod
    monotone = dc * df > 0
    p = float(np.log2(dc / df)) if monotone else np.nan
    applied = bool(monotone and 1.5 <= p <= 2.5)
    return {"p_obs": p, "monotone": bool(monotone), "applied": applied,
            "prod_error": (-4.0 * df / 3.0) if applied else None, "bound": abs(df)}


def splithalf_fraction(alpha_A, alpha_B, alpha_r):
    """mean_r[(aA - aB)^2]/4 over the sample variance (ddof=1) of alpha_r; flag if > 0.25."""
    aA, aB, a = (np.asarray(v, float) for v in (alpha_A, alpha_B, alpha_r))
    comp = float(np.mean((aA - aB) ** 2) / 4.0)
    var = float(a.var(ddof=1))
    frac = comp / var if var > 0 else (np.inf if comp > 0 else np.nan)
    return {"component": comp, "variance": var, "fraction": float(frac), "flag": bool(frac > 0.25)}


def delta_half(d, level: float = 0.90):
    """F7: per-realization Delta_half; 'decreasing' iff the two-sided CI lies above 0."""
    mean, lo, hi, n, sd = t_interval(d, level)
    return {"mean": mean, "lo": lo, "hi": hi, "n": n, "decreasing": bool(lo > 0)}


def rho_readings(rho_xb, m: float, S: float):
    """F8 readings from per-realization rho(X_b) (gating-free): resolved / limited / decorrelated."""
    r = np.asarray(rho_xb, float)
    p95, med = float(np.percentile(r, 95)), float(np.median(r))
    return {"p95": p95, "median": med,
            "pathwise_resolved": bool(p95 <= 0.1 * m / S),
            "pathwise_limited": bool(med >= np.sqrt(0.2 * m / S)),
            "decorrelated": bool(med >= 0.5)}


def _ols(x, y):
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    tot = ((y - y.mean()) ** 2).sum()
    return {"intercept": float(beta[0]), "slope": float(beta[1]), "r2": float(1 - (res**2).sum() / tot) if tot > 0 else np.nan,
            "n": int(x.size)}


def rho_growth_fits(x1, rho_matrix, x0: float, ell: float):
    """F8 growth: OLS of log median rho on x1 (exponential) and on log(x1 - x0) (power law).

    rho_matrix[r, plane]; planes with x1 >= x0 + ell and a positive finite median only.
    """
    x1 = np.asarray(x1, float)
    med = np.nanmedian(np.asarray(rho_matrix, float), axis=0)
    keep = (x1 >= x0 + ell - 1e-12 * max(1.0, abs(x0 + ell))) & np.isfinite(med) & (med > 0)
    x, ly = x1[keep], np.log(med[keep])
    if x.size < 3:
        return {"exp": None, "power": None, "n": int(x.size)}
    return {"exp": _ols(x, ly), "power": _ols(np.log(x - x0), ly), "n": int(x.size)}


def width_check(wide, narrow, level: float = 0.90):
    """F10: Welch CI of mean(wide) - mean(narrow) vs +-10 % of the wider arm's mean (no floor)."""
    diff, lo, hi, nu, se = welch_interval(wide, narrow, level)
    m = 0.10 * abs(float(np.mean(wide)))
    return {"diff": diff, "lo": lo, "hi": hi, "nu": nu, "margin": m,
            "label": equivalence(lo, hi, m, "gate5")}


def delta_pollock(alpha_pollock, alpha_consumer, level: float = 0.90):
    """F11: paired CI of Delta_P = alpha(Pollock) - alpha(consumer)."""
    return paired_interval(alpha_pollock, alpha_consumer, level)


def diff_in_diff(dp_a, dp_b, level: float = 0.90):
    """F11: per-seed (Delta_P of rung a) - (Delta_P of rung b), aligned arrays, paired t interval."""
    return paired_interval(dp_a, dp_b, level)
