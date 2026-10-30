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
"""
from __future__ import annotations

import math

import numpy as np
from scipy import special as _sp
from scipy import stats as _st

from ._util import t_isf
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


def gate5_paired(L, P, level=0.90, margin_ratio=MARGIN_RATIO):
    """Paired log-ratio equivalence of two codes on the same seeds (arrays (..., n); leading axes are replicates)."""
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
    q = t_isf(0.5 * (1.0 - level), n - 1)
    lo, hi = theta - q * se, theta + q * se
    verdict = np.where((lo >= -delta) & (hi <= delta), "agree",
                       np.where((lo > delta) | (hi < -delta), "disagree", "inconclusive"))
    with np.errstate(divide="ignore", invalid="ignore"):
        p1 = _st.t.sf((theta + delta) / se, n - 1)
        p2 = _st.t.sf((delta - theta) / se, n - 1)
    p = np.maximum(p1, p2)
    out = Gate5Result(theta=theta, se=se, dof=n - 1, lo=lo, hi=hi, verdict=verdict.item() if verdict.ndim == 0 else verdict,
                      p_tost=p, ratio=np.exp(theta), ratio_lo=np.exp(lo), ratio_hi=np.exp(hi), n=n,
                      p1=p1, p2=p2)
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
