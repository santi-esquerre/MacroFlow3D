"""Procedures qualified by the coverage simulator (SF-42 spec item 8, acceptance F; decision G5.3-G5.6).

A procedure maps a parameter dict, a standardized distribution (``covdist``), a random stream and a number ``m`` of replicates to
miss counts per nominal tail.  A *miss* is the event the procedure promises to keep below its tail: a one-sided limit on the wrong
side of the truth, an upper bound of a bias below the bias, a test that rejects at the boundary of its null.  The criterion
(``coverage.py``) is: the 95 % one-sided Clopper-Pearson upper limit of the miss probability is at most 1.25 times the tail.

Tail levels (derived from G5.4 / G5.6, all from the Holm sequence of the primary family: m = 7 tests, SF-43 spec item 6-7)
---------------------------------------------------------------------------------------------------------------------------
* Holm levels ``a_j = 0.05 / (7 - j)``, j = 0..6: 7.14e-3, 8.33e-3, 0.01, 0.0125, 0.01667, 0.025, 0.05.
* Sampling limit ``L_e(a / 2)`` of the robust-sign test at level ``a_j``: tails ``a_j / 2`` = 3.571e-3, 4.167e-3, 5e-3, 6.25e-3,
  8.33e-3, 0.0125, 0.025 (the worst is 3.57e-3), plus 0.05 (each side of a 90 % interval).
* Budget components ``B_k(a / (2K))`` with ``K`` = 3 (``E_h, E_W, E_int``) or 4 (+ the width x lambda/h interaction): tails
  ``a_j / 6`` (worst 1.190e-3) and ``a_j / 8`` (worst 8.929e-4): 14 values, 13 distinct (``a_5 / 6 = a_4 / 8``).  ``E_h`` itself uses its inputs at half of these
  (``p_L`` at ``t / 2``, down to 4.46e-4), which fixes the bootstrap size ``B``.
* TOST (Gate 5): each one-sided test at level ``a_j`` (Holm-adjusted, worst 7.14e-3); the 90 % CI of the log ratio uses the two
  0.05 tails (the level ``a_6 = 0.05`` row).
* Whole robust-sign pipeline: ``P(RL_e(a_j) > alpha_true) <= a_j`` at the seven Holm levels.

Replicates per cell: ``ceil(175 / smallest tail)`` (G5.3), so every tail of the cell has at least 175 expected misses at nominal.

Bootstrap size.  The percentile / studentized bootstrap limits use the rank rule of ``_util`` ((B + 1) quantiles): a tail below
``1 / (B + 1)`` returns an infinite limit.  The simulator takes ``B = ceil(boot_per_tail / tail_resolved)`` (default 10 resamples
beyond the quantile) rounded up to 1000, with ``tail_resolved`` the smallest tail the object resolves (``t`` for a contrast, ``t / 2``
for the ladder, ``tau`` for bootstrap-t) and counts the infinite limits (``unresolved``; zero by construction).

Arm structure of the pipeline (representative of SF-43; sizes in ``budget_arm_size``)
------------------------------------------------------------------------------------
Production arm: ``N`` seeds, per-seed sd 1, true mean ``mu = beta_h + beta_int + beta_W`` (``alpha_true = 0``).  Biases are in units of
the production standard error ``s_p = 1 / sqrt(N)`` and have the sign that makes the test anti-conservative (production above the
truth).  ``n_b`` seeds for every budget arm; paired differences have sd ``SD_PAIR = 0.35`` (the largest sd(diff)/sd of the SF-39
grid ladders is 0.34 (alpha_LAM, sigma^2 = 2.25); typical 0.05-0.2).

* ``E_h``: ratio-2 ladder of 3 levels paired by seed (levels 5/10/20 lambda/h), production level ``ip`` (1 = middle, 2 = finest),
  level errors ``beta_h * 2^(-order (i - ip))``, level noise sd ``SD_PAIR / sqrt 2`` (differences have sd 0.35).
* ``E_int``: paired h_max passes 16 and 8, errors ``beta_int`` and ``2 beta_int`` (order 1, the assumption behind the factor 2).
* ``E_W`` (K = 3): the wide arm (``n_b`` seeds, mean ``beta_h + beta_int``, unpaired) against the production arm (all N seeds).
* K = 4: ``E_W`` from the coarse-resolution pair (wide, production-width on the same ``n_b`` seeds as the fine production subset,
  width effect ``beta_W / 2`` at the coarse resolution) and ``E_WxH`` from the 2 x 2 (the change of the width effect between the
  resolutions, ``beta_W / 2``); the production width effect at the fine resolution is ``beta_W``, bounded by ``E_W + E_WxH``.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from . import budget as bd
from . import gate5 as g5
from . import intervals as iv

FWER = 0.05
HOLM_M = 7
KS = (3, 4)
SD_PAIR = 0.35
RES = (5.0, 10.0, 20.0)  # lambda/h of the ladder (only the ratio 2 matters)


def holm_levels(m=HOLM_M, fwer=FWER):
    """``fwer / m, fwer / (m - 1), ..., fwer`` (increasing): the levels of the Holm step-down for ``m`` tests."""
    return np.array([fwer / (m - j) for j in range(m)])


def sampling_tails():
    return np.unique(np.concatenate([holm_levels() / 2.0, [FWER]]))


def component_tails():
    return np.unique(np.concatenate([holm_levels() / (2.0 * K) for K in KS]))


def budget_arm_size(N):
    """Seeds of every budget arm for a production arm of ``N`` seeds: half of the production seeds, between 20 and 200
    (SF-43 sizes the budget arms jointly with the production ones; ladders, passes and widths are subsets)."""
    return int(min(200, max(20, N // 2)))


def boot_size(tail, boot_per_tail):
    """Bootstrap resamples that leave ``boot_per_tail`` of them beyond the ``tail`` quantile, rounded up to 1000."""
    return int(math.ceil(boot_per_tail / tail / 1000.0) * 1000)


def _count(mask):
    return mask.sum(axis=0).astype(np.int64)


# ----------------------------------------------------------------------------------------------------------------------
# Sampling limits of a mean (Student, bootstrap-t), lower and upper separately
# ----------------------------------------------------------------------------------------------------------------------
def run_sampling(p, dist, rng, m):
    tails = sampling_tails()
    x = dist.draw(rng, (m, p["n"]))  # true mean 0
    if p["method"] == "student":
        lo, up = iv.student_limit(x, tails, "lower"), iv.student_limit(x, tails, "upper")
    elif p["method"] == "boott":
        bt = iv.bootstrap_t(x, p["B"], rng, share_draws=p["share"])
        lo, up = iv.bootstrap_t_limit(bt, tails, "lower"), iv.bootstrap_t_limit(bt, tails, "upper")
    else:
        raise ValueError("unknown sampling method %r" % (p["method"],))
    rows = {"lower": _count(lo > 0.0), "upper": _count(up < 0.0)}
    extra = {"unresolved": _count(np.isinf(lo) | np.isinf(up))}
    return rows, extra


# ----------------------------------------------------------------------------------------------------------------------
# ContrastBound: UCB of |Delta| (paired, unpaired, 2 x 2 interaction)
# ----------------------------------------------------------------------------------------------------------------------
def run_contrast(p, dist, rng, m):
    tails = component_tails()
    n, kind, B, share = p["n"], p["kind"], p["B"], p["share"]
    sd = SD_PAIR
    if kind == "paired":  # E_int style: one pairing group; Delta = mean of the per-seed difference u
        delta = p["kappa"] * sd / math.sqrt(n)
        u = delta + sd * dist.draw(rng, (m, n))
        cb = bd.e_int(u, np.zeros_like(u), factor=1.0, B=B, rng=rng, share_draws=share)
    elif kind == "unpaired":  # E_W style: two independent arms of unit sd
        delta = p["kappa"] * math.sqrt(2.0 / n)
        cb = bd.e_w(delta + dist.draw(rng, (m, n)), dist.draw(rng, (m, n)), B=B, rng=rng, share_draws=share)
    elif kind == "interaction":  # 2 x 2: two paired differences, one per width, compared; Delta = their difference
        delta = p["kappa"] * sd * math.sqrt(2.0 / n)
        wf = delta + sd * dist.draw(rng, (m, n))
        pf = sd * dist.draw(rng, (m, n))
        cb = bd.e_interaction(wf, np.zeros_like(wf), pf, np.zeros_like(pf), B=B, rng=rng, share_draws=share)
    else:
        raise ValueError("unknown contrast kind %r" % (kind,))
    b = cb.bound(tails)
    return {"bound": _count(b < delta)}, {"unresolved": _count(np.isinf(b))}


# ----------------------------------------------------------------------------------------------------------------------
# LadderBound: E_h as a bound of the production level's |bias|
# ----------------------------------------------------------------------------------------------------------------------
def ladder_errors(order, ip, K=3, nonmono=False):
    """Level errors in units of the production error: ``2^(-order (i - ip))`` (error ``C h^order``, ratio 2), with alternating sign
    for ``nonmono`` so that the differences have no real order."""
    i = np.arange(K) - ip
    return 2.0 ** (-order * i) * ((-1.0) ** i if nonmono else 1.0)


def run_ladder(p, dist, rng, m):
    tails = component_tails()
    n, ip, B, share = p["n"], p["ip"], p["B"], p["share"]
    f = ladder_errors(p["order"], ip, nonmono=p["scenario"] == "nonmono")
    sd_level = SD_PAIR / math.sqrt(2.0)
    e_p = p["kappa"] * sd_level / math.sqrt(n)  # production bias = kappa standard errors of a level mean
    levels = (e_p * f)[None, None, :] + sd_level * dist.draw(rng, (m, n, 3))
    lb = bd.LadderBound(levels, RES, ip, safety=p.get("safety", 1.25), B=B, rng=rng, share_draws=share)  # safety: test control
    b = lb.bound(tails)
    used = lb.details(tails)["order_branch_used"]
    return {"bound": _count(b < e_p)}, {"unresolved": _count(np.isinf(b)), "order_branch": _count(used)}


# ----------------------------------------------------------------------------------------------------------------------
# Gate 5 TOST at the equivalence boundary, paired data
# ----------------------------------------------------------------------------------------------------------------------
def run_tost(p, dist, rng, m):
    n, rho, cv = p["n"], p["rho"], p["cv"]
    a = holm_levels()
    # z_L = A c + B u, z_P = A c - B u: unit variances, correlation rho, and z_L - z_P = 2 B u carries the stress distribution
    A, Bc = math.sqrt((1.0 + rho) / 2.0), math.sqrt((1.0 - rho) / 2.0)
    c, u = dist.draw(rng, (m, n)), dist.draw(rng, (m, n))
    zl, zp = A * c + Bc * u, A * c - Bc * u
    P = 1.0 + cv * zp
    ratio = g5.MARGIN_RATIO
    rows = {}
    for name, r in (("ratio_hi", ratio), ("ratio_lo", 1.0 / ratio)):  # true ratio at the boundary +delta / -delta
        res = g5.gate5_paired(r * (1.0 + cv * zl), P)
        rows[name] = _count(res.p_tost[:, None] <= a[None, :])
    return rows, {}


# ----------------------------------------------------------------------------------------------------------------------
# Whole robust-sign pipeline
# ----------------------------------------------------------------------------------------------------------------------
def run_pipeline(p, dist, rng, m):
    a = holm_levels()
    N, nb, K, share = p["N"], p["nb"], p["K"], p["share"]
    s_p = 1.0 / math.sqrt(N)
    e_h, e_i, w_f = p["b_h"] * s_p, p["b_int"] * s_p, p["b_w"] * s_p
    x = (e_h + e_i + w_f) + dist.draw(rng, (m, N))  # production arm, true alpha = 0
    if p["sampling"] == "student":
        samp = bd.StudentMean(x)
    else:
        samp = bd.BootstrapTMean(x, p["B_s"], rng=rng, share_draws=share)
    ip = p["ip"]
    lev = (e_h * ladder_errors(p["order"], ip))[None, None, :] + SD_PAIR / math.sqrt(2.0) * dist.draw(rng, (m, nb, 3))
    comps = [bd.LadderBound(lev, RES, ip, B=p["B_l"], rng=rng, share_draws=share)]
    a16 = np.full((m, nb), e_i)
    a8 = 2.0 * e_i + SD_PAIR * dist.draw(rng, (m, nb))
    comps.append(bd.e_int(a16, a8, B=p["B_c"], rng=rng, share_draws=share))
    base = e_h + e_i
    if K == 3:
        comps.append(bd.e_w(base + dist.draw(rng, (m, nb)), x, B=p["B_c"], rng=rng, share_draws=share))
    elif K == 4:
        w_c = 0.5 * w_f
        wide_f = base + dist.draw(rng, (m, nb))
        wide_c = wide_f + SD_PAIR * dist.draw(rng, (m, nb))
        prod_f = x[:, :nb]
        prod_c = prod_f - (w_f - w_c) + SD_PAIR * dist.draw(rng, (m, nb))
        comps.append(bd.e_w(wide_c, prod_c, B=p["B_c"], rng=rng, share_draws=share))
        comps.append(bd.e_interaction(wide_f, wide_c, prod_f, prod_c, B=p["B_c"], rng=rng, share_draws=share))
    else:
        raise ValueError("K must be 3 or 4")
    rl = bd.RobustLimit(samp, comps).lower(a)
    return {"RL": _count(rl > 0.0)}, {"nonfinite": _count(~np.isfinite(rl))}


# ----------------------------------------------------------------------------------------------------------------------
# BCa (reporting only)
# ----------------------------------------------------------------------------------------------------------------------
BCA_LEVELS = (0.95, 0.90)  # two-sided levels: tails 0.025 and 0.05 on each side


def run_bca(p, dist, rng, m):
    x = dist.draw(rng, (m, p["n"]))
    lo = np.empty((m, len(BCA_LEVELS)))
    hi = np.empty_like(lo)
    degenerate = np.zeros(len(BCA_LEVELS), dtype=np.int64)
    for i in range(m):
        for j, lev in enumerate(BCA_LEVELS):
            r = iv.bca_interval(x[i], level=lev, sided="two", B=p["B"], seed=int(rng.integers(2 ** 31)))
            lo[i, j], hi[i, j] = r.lo, r.hi
            degenerate[j] += int(r.degenerate)
    return {"lower": _count(lo > 0.0), "upper": _count(hi < 0.0)}, {"degenerate": degenerate}


class Proc(NamedTuple):
    tails: object  # callable(params) -> 1-D array of nominal tails (the same order as the count arrays)
    run: object  # callable(params, dist, rng, m) -> (rows, extra)


PROCS = {
    "sampling": Proc(lambda p: sampling_tails(), run_sampling),
    "contrast": Proc(lambda p: component_tails(), run_contrast),
    "ladder": Proc(lambda p: component_tails(), run_ladder),
    "tost": Proc(lambda p: holm_levels(), run_tost),
    "pipeline": Proc(lambda p: holm_levels(), run_pipeline),
    "bca": Proc(lambda p: np.array([0.025, 0.05]), run_bca),
}
