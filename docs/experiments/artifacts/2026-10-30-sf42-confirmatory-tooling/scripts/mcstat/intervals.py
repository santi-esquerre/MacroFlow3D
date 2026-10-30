"""Per-realization slope estimators (decision G4) and intervals / one-sided limits for a mean (decision G5.3, G5.4).

Conventions
-----------
* Samples live on the LAST axis; leading axes are independent replicates (vectorized: the coverage simulator passes arrays
  of shape ``(replicates, n)``).
* "Lower limit at tail ``tau``" is ``L(tau)`` with ``P(L(tau) > mu) <= tau``; "upper limit" ``U(tau)`` with
  ``P(U(tau) < mu) <= tau``.  ``tau`` may be a scalar (output without the tail axis) or a 1-D array (output gets a trailing tail
  axis).  Any tail in (0, 1) down to 1e-4 and below is supported (``_util.t_isf``, a refined ``t.isf``, is used, never ``1 - ppf``).
* ``sided``: ``'two'`` (tails ``(1 - level) / 2`` each), ``'lower'`` (only ``lo``, tail ``1 - level``; ``hi = +inf``), ``'upper'``
  (only ``hi``; ``lo = -inf``).
* Standard deviations are sample standard deviations (``ddof = 1``); degrees of freedom ``n - 1`` (Student) or Welch-Satterthwaite.
* Bootstrap work uses ``numpy.random.default_rng(seed)`` and resamples seeds with replacement (documented below).  No
  ``scipy.stats.bootstrap`` (its keyword API changed across the supported scipy versions).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy import stats as _st

from ._util import as_tails, bootstrap_means, get_rng, lcb_sorted, t_isf, ucb_sorted


# ----------------------------------------------------------------------------------------------------------------------
# Per-realization slope estimators (decision G4)
# ----------------------------------------------------------------------------------------------------------------------
def ols_slope(d, y):
    """Least-squares slope of ``y`` against ``d`` (decision G4 "per-realization least-squares slope").

    ``d``: (P,) plane distances (travel from the injection plane); ``y``: (..., P) one moment per plane (e.g. the transverse
    crossing variance ``(var x2 + var x3) / 2`` or the crossing-time variance).  Returns shape ``y.shape[:-1]``.
    ``slope = sum((d - dbar) (y - ybar)) / sum((d - dbar)^2)``.  Within-realization OLS standard errors are never used (G4).
    """
    d = np.asarray(d, dtype=float)
    y = np.asarray(y, dtype=float)
    if d.ndim != 1 or y.shape[-1] != d.size:
        raise ValueError("y must have the plane axis (length P) last, d must be 1-D with P entries")
    if d.size < 2:
        raise ValueError("a slope needs at least two planes")
    dc = d - d.mean()
    sdd = float(dc @ dc)
    if sdd == 0.0:
        raise ValueError("the plane distances are all equal")
    return (y - y.mean(axis=-1, keepdims=True)) @ dc / sdd


def alpha_T_from_slope(slope):
    """G4 ``alpha_TP`` / ``alpha_TAP``: half the slope (against ``d``) of the transverse crossing variance.

    For a transverse variance growing as ``sigma_T^2 = 2 alpha_T d`` (``sigma_T^2 = (var x2 + var x3) / 2``), ``slope = 2 alpha_T``.
    """
    return 0.5 * np.asarray(slope, dtype=float)


def alpha_L_from_slope(slope, U):
    """G4 ``alpha_LP``: ``(U_r^2 / 2) * slope`` of the crossing-TIME variance against ``d``.

    Travel-time estimand: ``var(t at d) ~ 2 alpha_L d / U^2``.  ``U`` (scalar or one per replicate) is the realization's mean
    crossing speed over the window (see ``mean_crossing_speed``).  The estimand is the mean over realizations of these values,
    NOT ``U_bar^2/2`` times the mean slope (decision G4).
    """
    return 0.5 * np.asarray(U, dtype=float) ** 2 * np.asarray(slope, dtype=float)


def mean_crossing_speed(d, t_mean, method="ols"):
    """Realization's mean crossing speed ``U_r`` over the window from the mean crossing time at each plane.

    ``d``: (P,) distances; ``t_mean``: (..., P) crossing-time mean per plane.
    Interpretation (decision G4 says only "the realization's mean crossing speed over the window"):
    ``method='ols'``      ``U_r = 1 / slope_OLS(t_mean against d)`` (the least-squares travel speed, consistent with the slope
                          estimators used for the variances);
    ``method='endpoint'`` ``U_r = (d_last - d_first) / (t_mean[last] - t_mean[first])``.
    """
    d = np.asarray(d, dtype=float)
    t_mean = np.asarray(t_mean, dtype=float)
    if method == "ols":
        return 1.0 / ols_slope(d, t_mean)
    if method == "endpoint":
        return (d[-1] - d[0]) / (t_mean[..., -1] - t_mean[..., 0])
    raise ValueError("method must be 'ols' or 'endpoint'")


def orthogonal_variance(var2, var3, cov23, drift2, drift3):
    """Variance of the transverse crossing position orthogonal to the realization's mean crossing displacement (``alpha_TAP``).

    ``n = (-drift3, drift2) / |drift|`` is the unit vector orthogonal to the drift direction ``(drift2, drift3)`` (the realization's
    mean transverse crossing displacement; the reader chooses its definition, e.g. the least-squares tilt of the mean position
    against ``d``); the result is ``n^T Sigma n = (drift3^2 var2 + drift2^2 var3 - 2 drift2 drift3 cov23) / |drift|^2``.
    NaN where the drift vector is zero (direction undefined).
    """
    var2, var3, cov23, d2, d3 = (np.asarray(v, dtype=float) for v in (var2, var3, cov23, drift2, drift3))
    nrm2 = d2 * d2 + d3 * d3
    with np.errstate(divide="ignore", invalid="ignore"):
        return (d3 * d3 * var2 + d2 * d2 * var3 - 2.0 * d2 * d3 * cov23) / np.where(nrm2 > 0, nrm2, np.nan)


# ----------------------------------------------------------------------------------------------------------------------
# Student, paired, Welch
# ----------------------------------------------------------------------------------------------------------------------
class Interval(NamedTuple):
    mean: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    n: int
    sd: np.ndarray


def _bounds(mean, half_lo, half_hi):
    return mean - half_lo, mean + half_hi


def student_interval(x, level=0.90, sided="two"):
    """Student-t interval of the mean of ``x`` (..., n): formula ``mean -/+ t_{1 - tau, n - 1} sd / sqrt(n)``."""
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    mean = x.mean(axis=-1)
    if n < 2:
        nan = np.full_like(mean, np.nan)
        return Interval(mean, nan, nan, n, nan)
    sd = x.std(axis=-1, ddof=1)
    se = sd / np.sqrt(n)
    if sided == "two":
        q = t_isf(0.5 * (1.0 - level), n - 1)
        return Interval(mean, mean - q * se, mean + q * se, n, sd)
    q = t_isf(1.0 - level, n - 1)
    if sided == "lower":
        return Interval(mean, mean - q * se, np.full_like(mean, np.inf), n, sd)
    if sided == "upper":
        return Interval(mean, np.full_like(mean, -np.inf), mean + q * se, n, sd)
    raise ValueError("sided must be 'two', 'lower' or 'upper'")


def student_limit(x, tail, side="lower"):
    """One-sided Student limit of the mean of ``x`` (..., n) at tail ``tau`` (scalar or 1-D array): ``mean -/+ t_{1 - tau} se``.

    ``side='lower'`` gives ``L(tau)`` (``P(L > mu) <= tau``, increasing in ``tau``), ``'upper'`` gives ``U(tau)``.  Output shape
    ``x.shape[:-1] + (T,)`` (no tail axis for a scalar tail).  ``n < 2`` gives NaN.
    """
    tau, scalar = as_tails(tail)
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    mean = x.mean(axis=-1)[..., None]
    if n < 2:
        out = np.full(mean.shape[:-1] + tau.shape, np.nan)
        return out[..., 0] if scalar else out
    se = (x.std(axis=-1, ddof=1) / np.sqrt(n))[..., None]
    q = t_isf(tau, n - 1)
    if side == "lower":
        out = mean - q * se
    elif side == "upper":
        out = mean + q * se
    else:
        raise ValueError("side must be 'lower' or 'upper'")
    return out[..., 0] if scalar else out


def paired_interval(x, y, level=0.90, sided="two"):
    """Paired difference ``x - y`` (aligned per seed, same shape) as one Student sample."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("paired arrays must have the same shape (align by seed first)")
    return student_interval(x - y, level, sided)


class Welch(NamedTuple):
    diff: np.ndarray
    se: np.ndarray
    nu: np.ndarray
    lo: np.ndarray
    hi: np.ndarray


def welch_contrast(mean1, sd1, n1, mean2, sd2, n2, c1=1.0, c2=-1.0, level=0.90, sided="two"):
    """Welch-Satterthwaite interval of the contrast ``c1 mu1 + c2 mu2`` from summary statistics (broadcasting over arrays).

    ``diff = c1 m1 + c2 m2``; ``v_i = c_i^2 s_i^2 / n_i``; ``se = sqrt(v1 + v2)``; ``nu = (v1 + v2)^2 / (v1^2/(n1-1) + v2^2/(n2-1))``;
    interval ``diff -/+ t_{nu} se``.  ``c1 = 1, c2 = -1`` is the ordinary difference of means (SF-39 Gate 5); ``c2 = -0.9`` is the
    ratio-consistent contrast ``mu_L - 0.9 mu_P`` of the post-review evaluation.
    """
    m1, s1, n1, m2, s2, n2 = (np.asarray(v, dtype=float) for v in (mean1, sd1, n1, mean2, sd2, n2))
    v1 = (c1 * s1) ** 2 / n1
    v2 = (c2 * s2) ** 2 / n2
    diff = c1 * m1 + c2 * m2
    se = np.sqrt(v1 + v2)
    with np.errstate(divide="ignore", invalid="ignore"):
        nu = (v1 + v2) ** 2 / (v1 ** 2 / (n1 - 1) + v2 ** 2 / (n2 - 1))
    if sided == "two":
        q = t_isf(0.5 * (1.0 - level), nu)
        return Welch(diff, se, nu, diff - q * se, diff + q * se)
    q = t_isf(1.0 - level, nu)
    if sided == "lower":
        return Welch(diff, se, nu, diff - q * se, np.full_like(diff, np.inf))
    if sided == "upper":
        return Welch(diff, se, nu, np.full_like(diff, -np.inf), diff + q * se)
    raise ValueError("sided must be 'two', 'lower' or 'upper'")


def welch_from_summary(mean1, sd1, n1, mean2, sd2, n2, level=0.90, sided="two"):
    """Welch interval of ``mu1 - mu2`` from summary statistics (see ``welch_contrast``)."""
    return welch_contrast(mean1, sd1, n1, mean2, sd2, n2, 1.0, -1.0, level, sided)


def welch(a, b, level=0.90, sided="two", c1=1.0, c2=-1.0):
    """Welch interval of the contrast ``c1 mean(a) + c2 mean(b)`` of two independent samples (last axis; leading axes batch)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return welch_contrast(a.mean(-1), a.std(-1, ddof=1), a.shape[-1], b.mean(-1), b.std(-1, ddof=1), b.shape[-1],
                          c1, c2, level, sided)


def welch_limit(a, b, tail, side="lower"):
    """One-sided Welch limit of ``mean(a) - mean(b)`` at tail ``tau`` (scalar or 1-D array of tails)."""
    tau, scalar = as_tails(tail)
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    w = welch_contrast(a.mean(-1), a.std(-1, ddof=1), a.shape[-1], b.mean(-1), b.std(-1, ddof=1), b.shape[-1])
    q = t_isf(tau, w.nu[..., None])
    out = w.diff[..., None] - q * w.se[..., None] if side == "lower" else w.diff[..., None] + q * w.se[..., None]
    return out[..., 0] if scalar else out


def margin_multiple(lo, hi, m):
    """``max(|lo|, |hi|) / m``: how many margins ``m`` the interval ``[lo, hi]`` reaches from 0 (the SF-39 F4 ratio of the
    twin-difference interval to the equivalence margin; the interval lies inside ``+-k m`` iff the ratio is at most ``k``)."""
    return np.maximum(np.abs(lo), np.abs(hi)) / m


# ----------------------------------------------------------------------------------------------------------------------
# Bootstrap-t (studentized) for a mean
# ----------------------------------------------------------------------------------------------------------------------
class BootT(NamedTuple):
    """Studentized bootstrap distribution of a mean: ``tstar`` is sorted ascending along its last axis."""
    mean: np.ndarray
    se: np.ndarray
    tstar: np.ndarray
    B: int


def bootstrap_t(x, B=2000, rng=None, seed=None, share_draws=False):
    """Bootstrap-t distribution of the mean of ``x`` (..., n).

    Resampling: each of the ``B`` resamples draws ``n`` seeds with replacement from the ``n`` observed values (independent
    resamples for every replicate row of ``x``); ``t*_b = (mean*_b - mean) / (sd*_b / sqrt(n))`` with ``sd*_b`` the sample
    standard deviation (ddof = 1) of resample ``b``.  A resample with ``sd* = 0`` gets ``t* = 0`` (equal mean) or ``+-inf``.
    ``rng`` (a ``numpy.random.Generator``) or ``seed`` selects the stream: ``np.random.default_rng(seed)``.
    ``share_draws=False`` (default) draws independent resample indices for every replicate row (a Python loop over rows);
    ``share_draws=True`` uses ONE index matrix for the whole batch (a single matrix product, fast): the per-replicate
    bootstrap-t limit is still a valid limit, but the replicates' errors become dependent through the shared draws, so a
    miss-rate estimate over a batch has a larger Monte Carlo error than the binomial one: use many independent batches.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[-1]
    rng = get_rng(rng, seed)
    mean = x.mean(axis=-1)
    se = x.std(axis=-1, ddof=1) / np.sqrt(n)
    xc = x - mean[..., None]
    m, s2 = bootstrap_means((xc, xc * xc), B, rng, share_draws)
    var = np.maximum(n / (n - 1.0) * (s2 - m * m), 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = m / (np.sqrt(var) / np.sqrt(n))
    t = np.where(np.sqrt(var) > 0.0, t, np.where(m == 0.0, 0.0, np.sign(m) * np.inf))
    t.sort(axis=-1)
    return BootT(mean, se, t, B)


def bootstrap_t_limit(bt, tail, side="lower"):
    """One-sided bootstrap-t limit at tail ``tau`` from a ``BootT``.

    ``L(tau) = mean - q_upper(t*, tau) se`` and ``U(tau) = mean - q_lower(t*, tau) se`` with the rank rule of ``_util``;
    ``-inf`` / ``+inf`` when ``B`` is too small to resolve the tail (``tau < 1 / (B + 1)``).  Output as ``student_limit``.
    """
    tau, scalar = as_tails(tail)
    mean, se = bt.mean[..., None], bt.se[..., None]
    if side == "lower":
        q = ucb_sorted(bt.tstar, tau)
        with np.errstate(invalid="ignore"):
            out = np.where(np.isinf(q), -np.inf, mean - q * se)
    elif side == "upper":
        q = lcb_sorted(bt.tstar, tau)
        with np.errstate(invalid="ignore"):
            out = np.where(np.isinf(q), np.inf, mean - q * se)
    else:
        raise ValueError("side must be 'lower' or 'upper'")
    out = np.where(se == 0.0, mean, out)
    return out[..., 0] if scalar else out


def bootstrap_t_interval(x, level=0.90, sided="two", B=2000, rng=None, seed=None, share_draws=False):
    """Bootstrap-t interval of the mean (``sided`` as in ``student_interval``)."""
    bt = bootstrap_t(x, B, rng, seed, share_draws)
    if sided == "two":
        tau = 0.5 * (1.0 - level)
        lo, hi = bootstrap_t_limit(bt, tau, "lower"), bootstrap_t_limit(bt, tau, "upper")
    elif sided == "lower":
        lo, hi = bootstrap_t_limit(bt, 1.0 - level, "lower"), np.full_like(bt.mean, np.inf)
    elif sided == "upper":
        lo, hi = np.full_like(bt.mean, -np.inf), bootstrap_t_limit(bt, 1.0 - level, "upper")
    else:
        raise ValueError("sided must be 'two', 'lower' or 'upper'")
    return Interval(bt.mean, lo, hi, np.asarray(x).shape[-1], bt.se * np.sqrt(np.asarray(x).shape[-1]))


# ----------------------------------------------------------------------------------------------------------------------
# Studentized upper bounds of |Delta| (SF-42 N6c; decision G5.3, G5.4)
# ----------------------------------------------------------------------------------------------------------------------
# The budget components B_k(t) are upper confidence bounds, at tail ``t``, of ``|Delta|`` for a linear contrast of arm means
# ``Delta = sum_g E[u_g]`` (``u_g``: the per-seed combination of the arms of independent pairing group ``g``).  The percentile
# bootstrap of ``|Delta*|`` is not studentized and under-covers at small ``n`` even on normal data (the coverage simulation of
# N6b: ``n = 30``, ``P(t_{n-1} < -z sqrt((n-1)/n))`` instead of ``Phi(-z)``); the two methods below are studentized.
#
#   U(t) = max(UCL_t, -LCL_t),    UCL_t, LCL_t = the one-sided limits of ``Delta`` at tail ``t``.
#
# Validity.  If ``Delta >= 0`` then ``|Delta| = Delta`` and ``U(t) >= UCL_t``, so the miss ``|Delta| > U(t)`` implies
# ``Delta > UCL_t`` (probability ``t`` for an exact limit).  If ``Delta < 0`` then ``|Delta| = -Delta`` and ``U(t) >= -LCL_t``, so
# the miss implies ``-Delta > -LCL_t``, i.e. ``Delta < LCL_t`` (probability ``t``).  One of the two cases holds, hence
# ``P(|Delta| > U(t)) <= t`` whenever the one-sided limits are exact; for finite ``n`` and skewed data they are approximate, which
# the SF-42 coverage simulation measures.
#
# * ``method='student'``      ``U(t) = |Delta_hat| + t_{1-t, nu} se`` (UCL and -LCL differ only in the sign of ``Delta_hat``), with the
#                             Welch-Satterthwaite standard error over the independent groups (one group: one-sample ``t`` on the
#                             per-seed combinations, ``nu = n - 1``).
# * ``method='bootstrap-t'``  ``T*_b = (Delta*_b - Delta_hat) / se*_b`` over joint resamples of the seeds (independent resamples of
#                             every group, ``se*_b`` recomputed on the resample), ``UCL_t = Delta_hat - se T*_{(t)}``,
#                             ``LCL_t = Delta_hat - se T*_{(1-t)}`` with the rank rule of ``_util``; ``+inf`` when ``B`` cannot
#                             resolve the tail (never an optimistic value).
#
# Both are second-order accurate for smooth functions of means under their regularity conditions; neither is exact for skewed data.
BOUND_METHODS = ("percentile", "student", "bootstrap-t")


def check_bound_method(method):
    if method not in BOUND_METHODS:
        raise ValueError("method must be one of %s, got %r" % (BOUND_METHODS, method))
    return method


def welch_groups(groups):
    """``(Delta_hat, se, nu)`` of ``Delta = sum_g mean(u_g)`` over independent groups ``u_g`` of shape ``(..., n_g)``.

    ``se^2 = sum_g v_g`` with ``v_g = s_g^2 / n_g`` (``s_g`` the ddof = 1 standard deviation) and the Welch-Satterthwaite degrees of
    freedom ``nu = (sum v_g)^2 / sum(v_g^2 / (n_g - 1))`` (exactly ``n - 1`` for one group; ``sum(n_g - 1)`` where every ``v_g`` is 0,
    where ``se = 0`` makes ``nu`` irrelevant).  Extra leading axes (e.g. the pairs of a ladder) are carried along.
    """
    est = se2 = den = 0.0
    dof = 0.0
    for u in groups:
        u = np.asarray(u, dtype=float)
        n = u.shape[-1]
        if n < 2:
            raise ValueError("a studentized bound needs at least 2 seeds in every pairing group")
        v = u.var(axis=-1, ddof=1) / n
        est = est + u.mean(axis=-1)
        se2 = se2 + v
        den = den + v * v / (n - 1.0)
        dof += n - 1.0
    if len(groups) == 1:
        nu = np.full(np.shape(est), dof)
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            nu = np.where(den > 0.0, se2 * se2 / den, dof)
    return est, np.sqrt(se2), nu


def studentize_sorted(num, var_of_mean):
    """Sorted (last axis) ``T* = num / sqrt(var_of_mean)``; a resample with zero variance gets ``0`` (``num = 0``) or ``+-inf``."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = num / np.sqrt(var_of_mean)
        t = np.where(var_of_mean > 0.0, t, np.where(num == 0.0, 0.0, np.sign(num) * np.inf))
    return np.sort(t, axis=-1)


def _var_of_mean(mc, s2c, n):
    """Variance of the mean of a resample from its centred first and second moments: ``n/(n-1) (s2c - mc^2) / n`` (floored at 0)."""
    return np.maximum(n / (n - 1.0) * (s2c - mc * mc), 0.0) / n


def tstar_from_moments(mc, s2c, n):
    """Sorted bootstrap-t statistics of ONE group from the resample means ``mc`` and second moments ``s2c`` of the CENTRED values
    (``u - mean(u)``): ``T* = mc / sqrt(var*/n)``, ``var*`` the ddof = 1 variance of the resample."""
    return studentize_sorted(mc, _var_of_mean(mc, s2c, n))


def boot_tstar(groups, B, rng, share_draws=False, inner_ndim=0):
    """Sorted ``T*`` (..., B) of ``Delta = sum_g mean(u_g)``: resamples of the seeds, independent between groups, ``T* = (Delta* -
    Delta_hat) / se*`` with ``se*^2 = sum_g var*_g / n_g`` recomputed on every resample."""
    num = var = 0.0
    for u in groups:
        u = np.asarray(u, dtype=float)
        n = u.shape[-1]
        uc = u - u.mean(axis=-1, keepdims=True)
        mc, s2c = bootstrap_means((uc, uc * uc), B, rng, share_draws, inner_ndim)
        num = num + mc
        var = var + _var_of_mean(mc, s2c, n)
    return studentize_sorted(num, var)


def abs_upper_bound(delta, se, tails, nu=None, tstar=None):
    """``U(t) = max(UCL_t, -LCL_t)``, the upper confidence bound of ``|Delta|`` at tail(s) ``t`` (see the section comment).

    ``delta``, ``se``: shape ``(...)``; ``tails``: scalar or 1-D (output ``(..., T)``, no tail axis for a scalar).  Student when
    ``tstar`` is None (``nu`` required): ``|delta| + t_{1-t, nu} se``.  Bootstrap-t when the sorted ``tstar`` (..., B) is given:
    ``UCL_t = delta - se T*_{(t)}``, ``LCL_t = delta - se T*_{(1-t)}``, ``+inf`` for a tail that ``B`` cannot resolve.  ``se = 0``
    gives ``|delta|``.
    """
    tau, scalar = as_tails(tails)
    d, s = np.asarray(delta, dtype=float)[..., None], np.asarray(se, dtype=float)[..., None]
    if tstar is None:
        if nu is None:
            raise ValueError("nu is required for the Student bound")
        nu = np.asarray(nu, dtype=float)
        # one degrees-of-freedom value (single pairing group): the quantiles depend on the tail only (t_isf is the slow part)
        q = t_isf(tau, nu.flat[0]) if nu.size and np.all(nu == nu.flat[0]) else t_isf(tau, nu[..., None])
        out = np.abs(d) + q * s
    else:
        with np.errstate(invalid="ignore"):  # 0 * inf where se = 0 and the tail is unresolved: replaced below
            ucl = d - s * lcb_sorted(tstar, tau)
            lcl = d - s * ucb_sorted(tstar, tau)
            out = np.where(s == 0.0, np.abs(d), np.maximum(ucl, -lcl))
    return out[..., 0] if scalar else out


# ----------------------------------------------------------------------------------------------------------------------
# BCa (reporting only: decision G5.3 "BCa is reported, never a verdict")
# ----------------------------------------------------------------------------------------------------------------------
class BCa(NamedTuple):
    lo: float
    hi: float
    z0: float
    accel: float
    degenerate: bool


def bca_interval(x, level=0.90, sided="two", B=5000, seed=20261030, statistic=None):
    """BCa bootstrap interval of ``statistic`` (default: the mean) of a 1-D sample.  Reporting only.

    Percentile bootstrap with bias correction ``z0 = Phi^-1(#(theta* < theta_hat) / B)`` (ties count half) and jackknife
    acceleration ``a = sum(d^3) / (6 (sum d^2)^1.5)``, ``d = theta_jack_mean - theta_jack_i``; adjusted levels
    ``Phi(z0 + (z0 + z_alpha) / (1 - a (z0 + z_alpha)))``, linear-interpolation quantiles.  ``statistic(x, axis=-1)`` must be
    vectorized.  Degenerate inputs (n < 3, constant bootstrap distribution, ``z0`` infinite) return NaN with ``degenerate=True``.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("bca_interval takes a 1-D sample (pass differences for paired data)")
    stat = statistic or (lambda a, axis=-1: np.mean(a, axis=axis))
    n = x.size
    nan = BCa(np.nan, np.nan, np.nan, np.nan, True)
    if n < 3:
        return nan
    rng = np.random.default_rng(seed)
    th = float(stat(x, axis=-1))
    boots = np.asarray(stat(x[rng.integers(0, n, size=(B, n))], axis=-1))
    if np.ptp(boots) == 0.0:
        return nan
    frac = np.mean(boots < th) + 0.5 * np.mean(boots == th)
    if not 0.0 < frac < 1.0:
        return nan
    z0 = float(_st.norm.ppf(frac))
    jk = np.array([float(stat(np.delete(x, i), axis=-1)) for i in range(n)])
    d = jk.mean() - jk
    den = 6.0 * (d @ d) ** 1.5
    a = float((d ** 3).sum() / den) if den > 0 else 0.0
    if sided == "two":
        alphas = np.array([0.5 * (1.0 - level), 0.5 * (1.0 + level)])
    elif sided == "lower":
        alphas = np.array([1.0 - level, 1.0])
    elif sided == "upper":
        alphas = np.array([0.0, level])
    else:
        raise ValueError("sided must be 'two', 'lower' or 'upper'")
    out = [np.nan, np.nan]
    for j, al in enumerate(alphas):
        if al <= 0.0:
            out[j] = -np.inf
        elif al >= 1.0:
            out[j] = np.inf
        else:
            za = _st.norm.ppf(al)
            adj = _st.norm.cdf(z0 + (z0 + za) / (1.0 - a * (z0 + za)))
            out[j] = float(np.quantile(boots, adj))
    return BCa(out[0], out[1], z0, a, False)
