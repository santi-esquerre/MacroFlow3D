"""Inference of addendum 1 (``preregistration-addendum-1.md`` section 2): ratios of means, IC90, descriptive.

* ``paired_ratio(x, y)``: ``theta = log(mean x / mean y)`` on aligned finite seeds; IC90 = ``mcstat.gate5.gate5_paired(method="bootstrap-t",
  B=2000, seed=20261104)`` (used by every rule); its ``method="student"`` interval is reported beside it, never used by a rule.
* ``log_contrast(series, signs)``: the same studentized seed bootstrap (same rank rule, same stream) for ``theta = sum_k s_k log mean(x_k)``
  with the linearized per-seed contrast ``e = sum_k s_k x_k / mean(x_k)``; with two series (+1, -1) it IS ``gate5_paired`` (bit for bit,
  unit test).  The paired ratio of ratios ``m_far / m_main`` is the four-series case (J3 far +, D far -, J3 main -, D main +).
* ``unpaired_ratio(g1, g2)``: ``theta_1 - theta_2``, ``se = sqrt(se_1^2 + se_2^2)`` (each group linearized on its own seeds),
  Welch-Satterthwaite ``nu``, ``theta +- t_{0.95, nu} se`` ("Welch sobre el log").
* ``within(ci, (a, b))`` is ``lo >= a and hi <= b`` (closed).

A result is a plain dict; one that cannot be computed (``n < 3`` pairs, a non-positive mean) is ``{"evaluable": False, "n", "reason"}``,
never an exception.
"""
import math

import numpy as np

from mcstat._util import bootstrap_means, get_rng, lcb_sorted, t_isf, ucb_sorted
from mcstat.gate5 import gate5_paired
from mcstat.intervals import studentize_sorted

LEVEL, B, SEED, MIN_N = 0.90, 2000, 20261104, 3


def not_evaluable(n, reason):
    return {"evaluable": False, "n": int(n), "reason": reason}


def within(ci, bounds):
    return bool(ci[0] >= bounds[0] and ci[1] <= bounds[1])


def _check(series):
    n = int(np.asarray(series[0]).size)
    if n < MIN_N:
        return not_evaluable(n, "fewer than %d common finite seeds" % MIN_N)
    if any(not (float(np.mean(s)) > 0.0) for s in series):
        return not_evaluable(n, "a non-positive mean (the log of a ratio of means is undefined)")
    return None


def paired_ratio(x, y):
    """``mean x / mean y`` over the aligned seeds: bootstrap-t IC90 (rules) and Student IC90 (reported)."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    bad = _check([x, y])
    if bad:
        return bad
    try:
        bt = gate5_paired(x, y, level=LEVEL, method="bootstrap-t", B=B, seed=SEED)
        st = gate5_paired(x, y, level=LEVEL, method="student")
    except ValueError as e:  # e.g. a resample with a non-positive mean: reported, not raised
        return not_evaluable(x.size, str(e))
    return {"evaluable": True, "n": int(x.size), "method": "gate5_paired bootstrap-t, B=%d, seed=%d" % (B, SEED), "theta": float(bt.theta),
            "se": float(bt.se), "ratio": float(bt.ratio), "ci90": [float(bt.ratio_lo), float(bt.ratio_hi)],
            "student_ci90": [float(st.ratio_lo), float(st.ratio_hi)], "mean_num": float(x.mean()), "mean_den": float(y.mean())}


def log_contrast(series, signs):
    """Studentized seed bootstrap of ``theta = sum_k s_k log mean(x_k)`` (series aligned on the same seeds, ``signs`` +-1)."""
    xs = [np.asarray(s, dtype=float) for s in series]
    bad = _check(xs)
    if bad:
        return bad
    n, K = xs[0].size, len(xs)
    means = [x.mean() for x in xs]
    xn = [x / m for x, m in zip(xs, means)]
    e = 0.0
    for s, v in zip(signs, xn):
        e = e + s * v
    se = e.std(ddof=1) / math.sqrt(n)
    num, den = 1.0, 1.0
    for s, m in zip(signs, means):
        if s > 0:
            num = num * m
        else:
            den = den * m
    theta = np.log(num / den)
    prods = {(j, k): xn[j] * xn[k] for j in range(K) for k in range(j + 1)}
    out = bootstrap_means(tuple(xn) + tuple(prods.values()), B, get_rng(None, SEED))
    m_b, s_b = out[:K], dict(zip(prods, out[K:]))
    pn, pd = 1.0, 1.0
    for s, m in zip(signs, m_b):
        if s > 0:
            pn = pn * m
        else:
            pd = pd * m
    if np.any(pn <= 0.0) or np.any(pd <= 0.0):
        return not_evaluable(n, "a bootstrap resample has a non-positive mean")
    ee = 0.0
    for j in range(K):
        for k in range(j):
            ee = ee + (2.0 * signs[j] * signs[k]) * s_b[(j, k)] / (m_b[j] * m_b[k])
        ee = ee + s_b[(j, j)] / (m_b[j] * m_b[j])
    ts = studentize_sorted(np.log(pn / pd), np.maximum(n / (n - 1.0) * ee, 0.0) / n)
    tau = 0.5 * (1.0 - LEVEL)
    with np.errstate(divide="ignore", invalid="ignore"):
        lo = theta if se == 0.0 else theta - se * ucb_sorted(ts, tau)
        hi = theta if se == 0.0 else theta - se * lcb_sorted(ts, tau)
    return {"evaluable": True, "n": int(n), "method": "log-contrast studentized seed bootstrap, B=%d, seed=%d" % (B, SEED),
            "theta": float(theta), "se": float(se), "ratio": float(np.exp(theta)), "ci90": [float(np.exp(lo)), float(np.exp(hi))]}


def _group(series, signs):
    """(theta, se, n) of one group of series paired within the group (the linearization of ``paired_ratio``)."""
    xs = [np.asarray(s, dtype=float) for s in series]
    n = xs[0].size
    e = 0.0
    for s, x in zip(signs, xs):
        e = e + s * x / x.mean()
    return sum(s * math.log(x.mean()) for s, x in zip(signs, xs)), float(e.std(ddof=1) / math.sqrt(n)), n


def unpaired_ratio(g1, g2):
    """Ratio of two independent group contrasts; ``g = (series, signs)`` (e.g. ``([far, main], [1, -1])`` for F, ``([x], [1])`` for a level)."""
    for g in (g1, g2):
        bad = _check(list(g[0]))
        if bad:
            return bad
    t1, s1, n1 = _group(*g1)
    t2, s2, n2 = _group(*g2)
    theta, v = t1 - t2, s1 * s1 + s2 * s2
    se = math.sqrt(v)
    if v == 0.0:
        nu, lo, hi = float("nan"), theta, theta
    else:
        nu = v * v / (s1 ** 4 / (n1 - 1) + s2 ** 4 / (n2 - 1))
        q = float(t_isf(0.5 * (1.0 - LEVEL), nu))
        lo, hi = theta - q * se, theta + q * se
    return {"evaluable": True, "n": [int(n1), int(n2)], "method": "Welch on the log (Satterthwaite nu)", "theta": theta, "se": se, "nu": nu,
            "ratio": math.exp(theta), "ci90": [math.exp(lo), math.exp(hi)]}
