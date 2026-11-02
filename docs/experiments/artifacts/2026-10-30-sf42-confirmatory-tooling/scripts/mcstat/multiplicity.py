"""Multiplicity (decision G5.6): Holm step-down.

The primary family of SF-43 is the robust sign at ``sigma^2`` 1, 2.25, 4 (three p-values) and Gate 5 at the four amplitudes (four
TOST p-values), Holm at a family-wise 5 %.  The intersection-union combination of two estimators of one amplitude is
``budget.intersection_union`` (``max(p_TP, p_TAP)``).
"""
from __future__ import annotations

import numpy as np


def holm(pvalues, alpha=0.05):
    """Holm (1979) step-down procedure.

    Sort ``p_(1) <= ... <= p_(m)``; adjusted ``p~_(i) = min(1, max_{j <= i} (m - j + 1) p_(j))`` (running maximum, so adjusted
    values are monotone); reject hypothesis ``i`` iff ``p~_i <= alpha``.  Equivalent to the sequential rule "reject ``p_(1)``
    if ``p_(1) <= alpha / m``, then ``p_(2)`` if ``p_(2) <= alpha / (m - 1)``, ..., stop at the first failure".  Strongly
    controls the family-wise error rate at ``alpha`` under arbitrary dependence.  Returns ``(adjusted, reject)`` in the input
    order.  NaN p-values are never rejected and count as 1 (they still count toward ``m``).
    """
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError("holm expects a 1-D array of p-values")
    m = p.size
    if m == 0:
        return p.copy(), np.zeros(0, dtype=bool)
    pp = np.where(np.isnan(p), 1.0, p)
    order = np.argsort(pp, kind="stable")
    scaled = (m - np.arange(m)) * pp[order]
    adj_sorted = np.minimum(1.0, np.maximum.accumulate(scaled))
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj, adj <= alpha
