"""Sample-size rules (decision G5.2) and the Gate 4 power model with joint allocation (G5.2, "arms that feed the error budget are
allocated jointly with the production arms for a stated Gate 4 power").

Conservative inputs (G5.2): standard deviations inflated to their 80 % upper confidence bound, means deflated to their 80 % lower
bound, the pairing correlation (Gate 5) taken at its 80 % lower bound (0 unless calibrated on a disjoint seed range).
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from scipy import optimize as _opt
from scipy import stats as _st

from ._util import t_isf

CONF80 = 0.80


# ----------------------------------------------------------------------------------------------------------------------
# One-sided confidence bounds of the planning inputs
# ----------------------------------------------------------------------------------------------------------------------
def sd_upper(sd, n, conf=CONF80):
    """Upper one-sided ``conf`` confidence bound of a standard deviation from ``n`` values (normal data):
    ``sd * sqrt((n - 1) / chi2_{1 - conf, n - 1})`` (``chi2_q`` the lower ``q``-quantile)."""
    sd = np.asarray(sd, dtype=float)
    return sd * np.sqrt((n - 1) / _st.chi2.ppf(1.0 - conf, n - 1))


def mean_lower(mean, sd, n, conf=CONF80):
    """Lower one-sided ``conf`` confidence bound of a mean: ``mean - t_{conf, n - 1} sd / sqrt(n)``."""
    return np.asarray(mean, dtype=float) - t_isf(1.0 - conf, n - 1) * np.asarray(sd, dtype=float) / math.sqrt(n)


def corr_lower(r, n, conf=CONF80):
    """Lower one-sided ``conf`` confidence bound of a correlation (Fisher): ``tanh(atanh(r) - z_conf / sqrt(n - 3))``."""
    if n <= 3:
        raise ValueError("a correlation bound needs n > 3")
    r = np.clip(np.asarray(r, dtype=float), -0.999999, 0.999999)
    return np.tanh(np.arctanh(r) - _st.norm.ppf(conf) / math.sqrt(n - 3))


# ----------------------------------------------------------------------------------------------------------------------
# Absolute half-width targets
# ----------------------------------------------------------------------------------------------------------------------
def n_for_halfwidth(sd, half, level=0.95, n_min=2, n_max=10 ** 7):
    """Smallest ``n`` with ``t_{(1 + level)/2, n - 1} * sd / sqrt(n) <= half`` (two-sided ``level`` Student half-width).

    The half-width is decreasing in ``n`` (checked in the tests), so an exponential search plus bisection is exact.
    Returns ``n_max`` if the target is not reached by it.
    """
    if not (sd >= 0 and half > 0):
        raise ValueError("need sd >= 0 and half > 0")

    def ok(n):
        return t_isf(0.5 * (1.0 - level), n - 1) * sd / math.sqrt(n) <= half

    if ok(n_min):
        return n_min
    hi = max(n_min * 2, 4)
    while not ok(hi):
        hi *= 2
        if hi >= n_max:
            return n_max
    lo = hi // 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return hi


def n_absolute_target(sd, n_pilot, half, level=0.95, conf=CONF80, n_min=2):
    """G5.2 absolute precision target: ``n_for_halfwidth(sd_upper(sd, n_pilot, conf), half, level)``."""
    return n_for_halfwidth(float(sd_upper(sd, n_pilot, conf)), half, level, n_min)


def n_relative_target(mean, sd, n_pilot, rel, level=0.95, conf=CONF80, n_min=2):
    """G5.2 target as a fraction ``rel`` of the mean: half-width ``rel * mean_lower`` (mean deflated to its ``conf`` lower bound),
    SD inflated to its ``conf`` upper bound.  Raises ``ValueError`` if the deflated mean is not positive."""
    m = float(mean_lower(abs(mean), sd, n_pilot, conf))
    if m <= 0:
        raise ValueError("the deflated mean is not positive: a relative target is undefined")
    return n_absolute_target(sd, n_pilot, rel * m, level, conf, n_min)


def clamp_resize(n_required, n_orig, max_factor=2.0):
    """G5.2 blinded re-estimation rule: upward only and at most ``max_factor`` times the original ``N``."""
    return int(min(max(int(n_required), int(n_orig)), math.floor(max_factor * n_orig)))


# ----------------------------------------------------------------------------------------------------------------------
# Gate 4 power model and joint allocation
# ----------------------------------------------------------------------------------------------------------------------
# Planning model (normal approximations; every assumption is listed here because the allocation inherits it):
#   RL = L(a/2) - sum_k B_k(a/(2K)) (decision G5.4), with
#     L(a/2)        = ybar - q_{a/2} s_p / sqrt(n_p)                (production arm: n_p seeds, per-seed sd s_p),
#     B_k(a/(2K))   = bias_k + slack_k + q_{a/(2K)} s_k / sqrt(n_k) + noise_k, noise_k ~ N(0, s_k^2/n_k)
#                     (component k: n_k seeds of its per-seed contrast, s_k the sd of the BOUND scale contrast, i.e. already
#                     multiplied by its extrapolation factor; slack_k >= 0 the planned excess of the bound over the true bias,
#                     e.g. the 1.25 safety factor, which does not shrink with n_k),
#   with q_x the upper x-quantile of the standard normal (or of Student with the arm's degrees of freedom: ``use_t``).
#   The production mean carries the biases: E[ybar] = alpha + sum_k bias_k, so (the bias estimates being unbiased and of the
#   same sign as the production bias, the case the test must withstand) E[RL] = alpha - q s_p/sqrt(n_p) - sum_k (slack_k +
#   q_k s_k/sqrt(n_k)) and Var[RL] = s_p^2/n_p + sum_k s_k^2/n_k (independent pieces; sd estimation noise, non-normality and the
#   dependence between pieces that share seeds are ignored; the union bound makes the TEST valid under dependence, the
#   power approximation is not).  Power(a) = P(RL > 0) = Phi(E[RL] / sqrt(Var[RL])).
def _gate4_moments(alpha_true, a, n_prod, sd_prod, comps, use_t):
    K = len(comps)
    def q(tau, n):
        return t_isf(tau, n - 1) if use_t else _st.norm.isf(tau)
    mean = alpha_true - q(a / 2.0, n_prod) * sd_prod / math.sqrt(n_prod)
    var = sd_prod ** 2 / n_prod
    for c in comps:
        n = c["n"]
        mean -= c.get("slack", 0.0) + q(a / (2.0 * K), n) * c["sd"] / math.sqrt(n)
        var += c["sd"] ** 2 / n
    return mean, var


def gate4_power(alpha_true, a, n_prod, sd_prod, comps, use_t=True):
    """Planning power ``P(RL_e(a) > 0)`` under the model above.

    ``comps``: list of dicts ``{'n': seeds, 'sd': sd of the bound-scale contrast, 'slack': planned excess (default 0)}`` (one per
    budget component).  ``use_t`` uses Student quantiles with ``n - 1`` degrees of freedom (default) instead of normal ones.
    """
    mean, var = _gate4_moments(alpha_true, a, n_prod, sd_prod, comps, use_t)
    return float(_st.norm.cdf(mean / math.sqrt(var)))


def allocate_gate4(alpha_plan, a, sd_prod, cost_prod, comps, power=0.8, n_prod_min=2, use_t=True, max_steps=100000):
    """Seeds for the production arm and the budget components minimizing total cost at a stated Gate 4 power.

    Minimizes ``cost_prod n_p + sum_k cost_k n_k`` subject to the planning power >= ``power`` (model: see above).  ``comps``:
    list of dicts ``{'sd', 'cost' (GPU-s per seed of that component's extra arms), 'slack' (default 0), 'n_min' (default 2)}``.
    With normal quantiles the constraint set is convex in the ``n`` (sum of convex terms under a norm), so SLSQP on the continuous
    relaxation (variables ``n / 100``) finds the optimum; the result is rounded up and then incremented greedily (largest power
    gain per cost) until the power with the requested quantile family is reached.  Returns ``(n_prod, [n_k], power_achieved, cost)``.
    Not a coverage guarantee: it inherits the model's assumptions.
    """
    K = len(comps)
    if alpha_plan <= 0:
        raise ValueError("the planning alpha must be positive for Gate 4 to be powered")
    costs = np.array([cost_prod] + [c["cost"] for c in comps], dtype=float)
    sds = np.array([sd_prod] + [c["sd"] for c in comps], dtype=float)
    nmin = np.array([n_prod_min] + [c.get("n_min", 2) for c in comps], dtype=float)
    slack = sum(c.get("slack", 0.0) for c in comps)
    qs = np.array([_st.norm.isf(a / 2.0)] + [_st.norm.isf(a / (2.0 * K))] * K)
    zp = _st.norm.ppf(power)

    def g(x):  # >= 0 feasible
        n = x * 100.0
        return alpha_plan - slack - float(np.sum(qs * sds / np.sqrt(n))) - zp * math.sqrt(float(np.sum(sds ** 2 / n)))

    x0 = np.maximum(nmin, 400.0) / 100.0
    while g(x0) < 0 and x0.max() < 1e9:
        x0 = x0 * 4.0
    if g(x0) < 0:
        raise ValueError("the power target is unreachable: alpha_plan <= slack")
    with warnings.catch_warnings():  # scipy 1.11 warns when SLSQP clips an iterate to the bounds (harmless: the clipped point is used)
        warnings.simplefilter("ignore", RuntimeWarning)
        res = _opt.minimize(lambda x: float(np.sum(costs * x)) * 100.0, x0, method="SLSQP",
                            bounds=[(m / 100.0, None) for m in nmin], constraints=[{"type": "ineq", "fun": g}],
                            options={"maxiter": 500, "ftol": 1e-12})
    n = np.maximum(np.ceil(np.maximum(res.x * 100.0, nmin) - 1e-9), nmin).astype(int)

    def pw(nn):
        cs = [dict(c, n=int(nn[i + 1])) for i, c in enumerate(comps)]
        return gate4_power(alpha_plan, a, int(nn[0]), sd_prod, cs, use_t)

    steps = 0
    while pw(n) < power and steps < max_steps:
        best, gain = 0, -1.0
        p0 = pw(n)
        for j in range(K + 1):
            n[j] += 1
            gj = (pw(n) - p0) / costs[j]
            n[j] -= 1
            if gj > gain:
                best, gain = j, gj
        n[best] += 1
        steps += 1
    if pw(n) < power:
        raise RuntimeError("allocation did not reach the power target within max_steps")
    return int(n[0]), [int(v) for v in n[1:]], pw(n), float(np.sum(costs * n))
