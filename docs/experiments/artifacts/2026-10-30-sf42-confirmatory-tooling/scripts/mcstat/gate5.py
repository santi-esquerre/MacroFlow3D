"""Gate 5, matched and paired (decision G5.5): equivalence of ``log(mean L / mean P)`` within ``+-log(1.10)``.

Data: per seed ``r`` the estimate of the Lagrangian consumer ``L_r`` and of Par2 ``P_r`` on the SAME SF-18 field and grid
(positive quantities, e.g. the plane-based ``alpha_LP``).  Estimand ``theta = log(mu_L / mu_P)`` (a ratio of means, not a mean of
ratios).

* estimate ``theta_hat = log(mean L / mean P)``;
* standard error from the linearization ``e_r = L_r / mean(L) - P_r / mean(P)`` (mean of ``e`` is exactly 0), ``se = sd(e) / sqrt(n)``
  (ddof = 1), ``t`` on ``n - 1`` degrees of freedom;
* ``level`` CI (default 90 %) ``theta_hat -/+ t_{(1 + level)/2, n-1} se``; verdict ``agree`` (CI inside ``+-delta``), ``disagree``
  (CI entirely outside ``[-delta, delta]``), ``inconclusive``; ``delta = log(margin_ratio)``, ``margin_ratio = 1.10``;
* TOST: ``p1 = P(T_{n-1} > (theta_hat + delta) / se)``, ``p2 = P(T_{n-1} > (delta - theta_hat) / se)``, ``p_tost = max(p1, p2)``;
  ``p_tost <= alpha`` iff the ``1 - 2 alpha`` CI lies strictly inside ``(-delta, delta)`` (tested);
* ``disagree`` is descriptive (outside the Holm family, decision G5.5); only the TOST p-value enters Holm (G5.6).

Power model (analytic).  The linearization makes ``theta_hat ~ N(theta, sigma_e^2 / n)`` and ``s_e^2 ~ sigma_e^2 chi2_{n-1} /
(n - 1)`` approximately independent, with ``sigma_e^2 = cv_L^2 + cv_P^2 - 2 rho cv_L cv_P`` (CV = sd / mean, ``rho`` the pairing
correlation).  The TOST rejects iff ``-delta + c s_e < theta_hat < delta - c s_e``, ``c = t_{1 - alpha, n-1} / sqrt(n)``, so
``power = E_s[ Phi((delta - c s - theta) sqrt(n) / sigma) - Phi((-delta + c s - theta) sqrt(n) / sigma) ]_+`` evaluated by
quadrature over the chi-square density (Gauss-Legendre, 600 nodes, checked against adaptive quadrature in the tests; an exact two-variable integral under the stated independence; no closed form such as a
noncentral t exists because the acceptance region is a triangle in ``(theta_hat, s)``).  A Monte Carlo check of the full pipeline
(bivariate normal ``L, P`` through ``gate5_paired``) is part of the tests: the approximation error comes only from the
linearization and the independence of ``theta_hat`` and ``s_e``; measured in the tests it is 0.001 at a true ratio 1 and about 0.012
at a true ratio 0.95 (CVs 0.2-0.3, n = 40-100): use the analytic power for planning, not as an exact probability.

Bootstrap-t variant (``method='bootstrap-t'``, SF-42 N6c).  The Student TOST above fails the coverage criterion of G5.3 under the stress
family (skewness 1.94, excess kurtosis 4.7) for ``n <= 170`` (N6b).  The studentized bootstrap replaces the Student reference
distribution by the resampled one: the seed pairs ``(L_r, P_r)`` are resampled jointly (the pairing is kept), and each resample
recomputes ``theta* = log(mean L* / mean P*)`` and the linearized ``se*`` (same formula as above, on the resample), giving
``T*_b = (theta*_b - theta_hat) / se*_b``.  One-sided p-values of the two TOST nulls (rank form of the Monte Carlo p-value, which is
exactly the rank rule of the confidence limits below)::

    p1 = (1 + #{T* >= (theta_hat + delta) / se}) / (B + 1)        H01: theta <= -delta
    p2 = (1 + #{T* <= (theta_hat - delta) / se}) / (B + 1)        H02: theta >= +delta

so ``p_tost = max(p1, p2) >= 1 / (B + 1)`` and a level ``alpha < 1 / (B + 1)`` can never be met (conservative, never optimistic);
the ``level`` CI is ``theta_hat - se T*_{(upper tau)}, theta_hat - se T*_{(lower tau)}`` with ``tau = (1 - level) / 2`` and the rank
rule of ``_util`` (infinite when ``B`` cannot resolve ``tau``).  ``p_tost <= alpha`` iff the ``1 - 2 alpha`` bootstrap-t CI lies
strictly inside ``(-delta, delta)`` (tested).  The verdict rule is unchanged (G5.5).  There is no analytic power for this variant
(the rejection region depends on the resampled distribution): power and sample size for ``'bootstrap-t'`` are obtained by
simulation, i.e. by running ``gate5_paired(..., method='bootstrap-t')`` on paired data generated at the planning ratio, CVs and
correlation (``covproc.run_tost`` generates exactly that bivariate structure; its ``tost:gate5/boott`` coverage cells use the two
boundary ratios and so measure the SIZE), while ``tost_power`` / ``gate5_sample_size`` remain the Student planning approximations
(for the bootstrap-t variant they are a guide, not its power).
"""
from __future__ import annotations

import math

import numpy as np
from scipy import special as _sp
from scipy import stats as _st

from ._util import bootstrap_means, get_rng, lcb_sorted, t_isf, ucb_sorted
from .intervals import studentize_sorted
from .sizing import corr_lower, mean_lower, sd_upper

MARGIN_RATIO = 1.10
LN_MARGIN = math.log(MARGIN_RATIO)
_GL_X, _GL_W = np.polynomial.legendre.leggauss(600)  # Gauss-Legendre rule of the power quadrature


class Gate5Result(dict):
    """dict with attribute access: theta, se, dof, lo, hi, verdict, p_tost, ratio, ratio_lo, ratio_hi, n, p1, p2."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None


def _boot_tstar(L, P, B, rng, share_draws):
    """Sorted ``T* = (theta* - theta_hat) / se*`` (..., B) of the joint seed-pair bootstrap (see the module docstring).

    Works on the data normalised by their means (``Ln = L / mean L``, ``Pn = P / mean P``): ``theta* - theta_hat = log(ml* / mp*)``
    with the resample means ``ml*``, ``mp*`` of ``Ln``, ``Pn``, and the linearized contrast of the resample is ``Ln / ml* - Pn / mp*``
    (mean exactly 0), whose second moment follows from the resample means of ``Ln^2``, ``Pn^2`` and ``Ln Pn``.
    """
    n = L.shape[-1]
    Ln, Pn = L / L.mean(axis=-1, keepdims=True), P / P.mean(axis=-1, keepdims=True)
    ml, mp, sl, sp, slp = bootstrap_means((Ln, Pn, Ln * Ln, Pn * Pn, Ln * Pn), B, rng, share_draws)
    if np.any(ml <= 0.0) or np.any(mp <= 0.0):  # the log ratio of the resample means is undefined: never skipped silently
        raise ValueError("a bootstrap resample has a non-positive mean of L or P: the log-ratio statistic is undefined")
    ee = sl / (ml * ml) - 2.0 * slp / (ml * mp) + sp / (mp * mp)  # mean of e*^2
    var_of_mean = np.maximum(n / (n - 1.0) * ee, 0.0) / n  # se*^2
    return studentize_sorted(np.log(ml / mp), var_of_mean)


def gate5_paired(L, P, level=0.90, margin_ratio=MARGIN_RATIO, method="student", B=4000, rng=None, seed=None, share_draws=False):
    """Paired log-ratio equivalence of two codes on the same seeds (arrays (..., n); leading axes are replicates).

    ``method='student'`` (default): Student reference distribution (module docstring).  ``method='bootstrap-t'``: studentized seed-pair
    bootstrap with ``B`` resamples, ``rng`` (a ``numpy.random.Generator``) or ``seed`` required, ``share_draws`` as in
    ``intervals.bootstrap_t``; ``dof`` is ``None`` and the result carries ``B``.
    """
    if method not in ("student", "bootstrap-t"):
        raise ValueError("method must be 'student' or 'bootstrap-t', got %r" % (method,))
    L = np.asarray(L, dtype=float)
    P = np.asarray(P, dtype=float)
    if L.shape != P.shape:
        raise ValueError("L and P must be aligned by seed (same shape)")
    n = L.shape[-1]
    if n < 3:
        raise ValueError("Gate 5 needs at least 3 paired seeds")
    mL, mP = L.mean(axis=-1), P.mean(axis=-1)
    if np.any(mL <= 0) or np.any(mP <= 0):
        raise ValueError("the log-ratio of means needs positive means")
    e = L / mL[..., None] - P / mP[..., None]
    se = e.std(axis=-1, ddof=1) / math.sqrt(n)
    theta = np.log(mL / mP)
    delta = math.log(margin_ratio)
    if method == "student":
        q = t_isf(0.5 * (1.0 - level), n - 1)
        lo, hi = theta - q * se, theta + q * se
        with np.errstate(divide="ignore", invalid="ignore"):
            p1 = _st.t.sf((theta + delta) / se, n - 1)
            p2 = _st.t.sf((delta - theta) / se, n - 1)
        dof = n - 1
    else:
        ts = _boot_tstar(L, P, B, get_rng(rng, seed), share_draws)
        tau = 0.5 * (1.0 - level)
        with np.errstate(divide="ignore", invalid="ignore"):  # 0 * inf where se = 0 and the tail is unresolved
            lo = np.where(se == 0.0, theta, theta - se * ucb_sorted(ts, tau))
            hi = np.where(se == 0.0, theta, theta - se * lcb_sorted(ts, tau))
            t1, t2 = (theta + delta) / se, (theta - delta) / se
        p1 = (1.0 + (ts >= t1[..., None]).sum(axis=-1)) / (B + 1.0)
        p2 = (1.0 + (ts <= t2[..., None]).sum(axis=-1)) / (B + 1.0)
        dof = None
    verdict = np.where((lo >= -delta) & (hi <= delta), "agree",
                       np.where((lo > delta) | (hi < -delta), "disagree", "inconclusive"))
    p = np.maximum(p1, p2)
    out = Gate5Result(theta=theta, se=se, dof=dof, lo=lo, hi=hi, verdict=verdict.item() if verdict.ndim == 0 else verdict,
                      p_tost=p, ratio=np.exp(theta), ratio_lo=np.exp(lo), ratio_hi=np.exp(hi), n=n,
                      p1=p1, p2=p2)
    if method == "bootstrap-t":
        out["B"] = B
    return out


def sd_e(cv_L, cv_P, corr):
    """Standard deviation of the linearized per-seed contrast ``e_r``: ``sqrt(cv_L^2 + cv_P^2 - 2 corr cv_L cv_P)``."""
    v = cv_L ** 2 + cv_P ** 2 - 2.0 * corr * cv_L * cv_P
    if v < 0:
        raise ValueError("inconsistent CVs / correlation")
    return math.sqrt(v)


def tost_power(n, sigma_e, delta=LN_MARGIN, theta=0.0, alpha=0.05):
    """Analytic TOST power (model in the module docstring) at true log-ratio ``theta`` for ``n`` paired seeds.

    ``alpha`` is the level of EACH one-sided test (Holm-adjusted: ``0.05 / m``); the CI has level ``1 - 2 alpha``.
    """
    if n < 3:
        raise ValueError("n >= 3")
    if delta <= 0:
        return 0.0
    if sigma_e <= 0:
        return 1.0 if abs(theta) < delta else 0.0
    df = n - 1
    c = t_isf(alpha, df) / math.sqrt(n)
    psi = sigma_e / math.sqrt(n)
    wstar = df * (delta / (c * sigma_e)) ** 2  # bracket vanishes for c s >= delta
    hi = min(wstar, _st.chi2.isf(1e-13, df))
    w = 0.5 * hi * (_GL_X + 1.0)  # 600-point Gauss-Legendre rule on [0, hi]; the integrand is smooth there
    s = sigma_e * np.sqrt(w / df)
    br = _sp.ndtr((delta - c * s - theta) / psi) - _sp.ndtr((-delta + c * s - theta) / psi)
    val = 0.5 * hi * float(np.sum(_GL_W * np.maximum(br, 0.0) * _st.chi2.pdf(w, df)))
    return float(min(1.0, max(0.0, val)))


def gate5_power(n, cv_L, cv_P, corr, alpha=0.05, ratio=1.0, margin_ratio=MARGIN_RATIO):
    """TOST power at a true ratio ``mu_L / mu_P = ratio`` (CVs and correlation as in ``sd_e``)."""
    return tost_power(n, sd_e(cv_L, cv_P, corr), math.log(margin_ratio), math.log(ratio), alpha)


def gate5_sample_size(cv_L, cv_P, corr, alpha, power=0.8, ratio=1.0, margin_ratio=MARGIN_RATIO, n_min=3, n_max=200000):
    """Smallest ``n`` whose TOST power at ``ratio`` is at least ``power`` (G5.5: ratio 1, alpha = Holm worst level ``0.05 / m``).

    The power is increasing in ``n`` at the planning ratios used (checked on a grid in the tests), so an exponential search plus
    bisection is exact.  Returns ``(n, reached)``; ``reached`` is False if ``n_max`` does not reach the power.
    """
    sigma = sd_e(cv_L, cv_P, corr)
    d, th = math.log(margin_ratio), math.log(ratio)

    def ok(n):
        return tost_power(n, sigma, d, th, alpha) >= power

    if ok(n_min):
        return n_min, True
    hi = max(2 * n_min, 8)
    while not ok(hi):
        hi *= 2
        if hi > n_max:
            return n_max, False
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi, True


def gate5_conservative_inputs(sd_L, sd_P, mean_L, mean_P, n_pilot, corr=None, conf=0.8, deflate_means=True):
    """G5.5 / G5.2 conservative planning inputs from a pilot of ``n_pilot`` paired seeds.

    ``cv = sd_upper(sd, n, conf) / mean_lower(mean, sd, n, conf)`` (80 % upper bound of the SD over the 80 % lower bound of the
    mean; ``deflate_means=False`` keeps the plain mean) and the pairing correlation at its ``conf`` lower bound; ``corr=None``
    (not calibrated on a disjoint seed range) gives 0.  Returns ``(cv_L, cv_P, corr_lower)``.
    """
    if deflate_means:
        mL = float(mean_lower(mean_L, sd_L, n_pilot, conf))
        mP = float(mean_lower(mean_P, sd_P, n_pilot, conf))
    else:
        mL, mP = mean_L, mean_P
    if mL <= 0 or mP <= 0:
        raise ValueError("deflated mean not positive: CV undefined")
    r = 0.0 if corr is None else float(corr_lower(corr, n_pilot, conf))
    return float(sd_upper(sd_L, n_pilot, conf)) / mL, float(sd_upper(sd_P, n_pilot, conf)) / mP, r
