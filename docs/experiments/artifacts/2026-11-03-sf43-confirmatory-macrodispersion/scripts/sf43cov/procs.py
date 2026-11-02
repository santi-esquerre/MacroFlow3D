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

import numpy as np

from mcstat import covproc

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
