"""The ``fieller`` procedure of the SF-43 coverage harness (registered into ``mcstat.covproc.PROCS``).

One cell = (distribution, ``n``).  Each replicate draws the per-seed differences ``D_r`` of the Fieller form AT THE BOUNDARY of the null
(mean 0, variance 1, the standardized distribution ``covdist.get_dist(dist)``: at the margin ``theta = 1.10`` or ``1/1.10`` the
differences ``L_r - theta P_r`` have exactly the distribution of the SF-42 ``sampling`` cells, independent of the CVs and of the pairing
correlation, pre-activation review section 3.e) and applies the one-sample tests of ``sf43ana.gate5.onesample_p``.  A MISS is the event
the procedure promises to keep below its tail: the test rejects its null although the true mean is on the boundary.  Rows (counts per
tail, 16 tails = the Holm levels ``fwer / (m - j)`` of the primary family and their halves):

    ``<proc>/hi``   p_hi <= tail   (H0: mean >= 0 against smaller values; the ``D_hi = L - 1.10 P`` test)
    ``<proc>/lo``   p_lo <= tail   (H0: mean <= 0 against larger values;  the ``D_lo = L - P / 1.10`` test)

for ``<proc>`` in ``student`` (descriptive), ``hall``, ``boott``.  The bootstrap-t at half the level is the row ``boott`` read at the
tails ``a_j / 2``.  Both skew directions are covered by the distributions ``stress`` (positive skewness) and ``stress-`` (its reflection),
so every side meets the anti-conservative direction.  ``B = 0`` skips the bootstrap (a test aid).  The bootstrap draws one resampling
matrix per batch of ``m`` replicates (``share_draws``): replicates of a batch are dependent through it, which the SF-42 evaluation
corrects by the design effect (``<= 1 + (m - 1) / B``, 1.001 for ``m = 50``, ``B = 33599``).

Ties: a rank p-value is a multiple of ``1 / (B + 1)`` and can equal a tail exactly (``B + 1 = 33600``); the comparison ``p <= tail`` is made
with a relative tolerance of 1e-9 so that a floating point representation of the tail never decides it.
"""
from __future__ import annotations

from sf43cov import mcpath  # noqa: F401

import math

import numpy as np

from mcstat import budget as bd
from mcstat import covproc
from mcstat import intervals as iv

from sf43ana import gate5 as g5

TIE_TOL = 1e-9


def fieller_tails(p):
    """Sorted distinct tails: the Holm levels of the ``family_m`` family and their halves."""
    main = g5.holm_levels(int(p["family_m"]), float(p["fwer"]))
    return np.unique(np.concatenate([main, main / 2.0]))


def _counts(pv, tails):
    return (pv[:, None] <= tails[None, :] * (1.0 + TIE_TOL)).sum(axis=0).astype(np.int64)


def run_fieller(p, dist, rng, m):
    tails = fieller_tails(p)
    x = dist.draw(rng, (m, int(p["n"])))
    rows = {}
    procs = [("student", {}), ("hall", {"ref": p.get("ref", "t")})]
    if int(p.get("B", 0)) > 0:
        procs.append(("boott", {"B": int(p["B"]), "rng": rng, "share_draws": True}))
    for name, kw in procs:
        ph, pl = g5.onesample_p(x, name, **kw)
        rows[name + "/hi"], rows[name + "/lo"] = _counts(ph, tails), _counts(pl, tails)
    return rows, {}


def register():
    covproc.PROCS["fieller"] = covproc.Proc(fieller_tails, run_fieller)


register()


# ----------------------------------------------------------------------------------------------------------------------
# The robust-sign pieces of H15 at the m = 8 Holm levels (mcstat's own procedures are fixed at m = 7)
# ----------------------------------------------------------------------------------------------------------------------
SD_PAIR = covproc.SD_PAIR
RES = covproc.RES


def holm_tails(p):
    return covproc.holm_levels(int(p["family_m"]), float(p["fwer"]))


def run_sf43_sampling(p, dist, rng, m):
    """Bootstrap-t sampling limits of a mean (the production limit of the robust sign) at the Holm levels and their halves, ``m = 8``:
    rows ``lower`` (miss: lower limit above the true mean 0) and ``upper`` (upper limit below it), as ``mcstat.covproc.run_sampling``."""
    tails = fieller_tails(p)
    x = dist.draw(rng, (m, int(p["n"])))
    bt = iv.bootstrap_t(x, int(p["B"]), rng, share_draws=bool(p["share"]))
    lo, up = iv.bootstrap_t_limit(bt, tails, "lower"), iv.bootstrap_t_limit(bt, tails, "upper")
    return ({"lower": covproc._count(lo > 0.0), "upper": covproc._count(up < 0.0)},
            {"unresolved": covproc._count(np.isinf(lo) | np.isinf(up))})


def run_sf43_pipeline(p, dist, rng, m):
    """The whole robust-sign pipeline at the Holm levels of the ``family_m`` family with PER-ARM sizes (``mcstat.covproc.run_pipeline``
    uses one size for all budget arms and ``m = 7``): production ``N`` (bootstrap-t sampling limit, ``sampling='boott'``), ladder ``n_l``
    seeds, ``h_max`` passes ``n_i``, width cells ``n_w``; ``K`` = 3 (``E_h``, ``E_int``, ``E_W``) or 4 (+ ``E_WxH``); all components
    Student-studentized (``comp``), true alpha 0, biases ``b_h, b_int, b_w`` in standard errors of the production mean (the
    anti-conservative sign).  Row ``RL``: ``RL(a_j) > 0`` at the Holm levels."""
    a = holm_tails(p)
    N, nl, ni, nw, K, share = int(p["N"]), int(p["n_l"]), int(p["n_i"]), int(p["n_w"]), int(p["K"]), bool(p["share"])
    cm = p.get("comp", "student")
    s_p = 1.0 / math.sqrt(N)
    e_h, e_i, w_f = p["b_h"] * s_p, p["b_int"] * s_p, p["b_w"] * s_p
    x = (e_h + e_i + w_f) + dist.draw(rng, (m, N))
    samp = bd.StudentMean(x) if p["sampling"] == "student" else bd.BootstrapTMean(x, int(p["B_s"]), rng=rng, share_draws=share)
    ip = int(p["ip"])
    lev = (e_h * covproc.ladder_errors(p["order"], ip))[None, None, :] + SD_PAIR / math.sqrt(2.0) * dist.draw(rng, (m, nl, 3))
    comps = [bd.LadderBound(lev, RES, ip, B=int(p["B_l"]), rng=rng, share_draws=share, method=cm)]
    a16 = np.full((m, ni), e_i)
    a8 = 2.0 * e_i + SD_PAIR * dist.draw(rng, (m, ni))
    comps.append(bd.e_int(a16, a8, B=int(p["B_c"]), rng=rng, share_draws=share, method=cm))
    base = e_h + e_i
    if K == 3:
        comps.append(bd.e_w(base + dist.draw(rng, (m, nw)), x, B=int(p["B_c"]), rng=rng, share_draws=share, method=cm))
    elif K == 4:
        w_c = 0.5 * w_f
        wide_f = base + dist.draw(rng, (m, nw))
        wide_c = wide_f + SD_PAIR * dist.draw(rng, (m, nw))
        prod_f = x[:, :nw]
        prod_c = prod_f - (w_f - w_c) + SD_PAIR * dist.draw(rng, (m, nw))
        comps.append(bd.e_w(wide_c, prod_c, B=int(p["B_c"]), rng=rng, share_draws=share, method=cm))
        comps.append(bd.e_interaction(wide_f, wide_c, prod_f, prod_c, B=int(p["B_c"]), rng=rng, share_draws=share, method=cm))
    else:
        raise ValueError("K must be 3 or 4")
    rl = bd.RobustLimit(samp, comps).lower(a)
    return {"RL": covproc._count(rl > 0.0)}, {"nonfinite": covproc._count(~np.isfinite(rl))}


def register_sf43():
    covproc.PROCS["sf43_sampling"] = covproc.Proc(fieller_tails, run_sf43_sampling)
    covproc.PROCS["sf43_pipeline"] = covproc.Proc(holm_tails, run_sf43_pipeline)


register_sf43()
