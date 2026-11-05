"""Holm step-down over a FIXED family (decision G5.6 with H4).

The family is fixed in the pre-registration (``Prereg.family_ids``); ``m`` is its size whatever can be tested.  A hypothesis that
cannot be tested (arm below its floor, missing input, Gate 5 p-value not supplied) enters with ``p = 1`` and is never rejected.

Procedure: order by ``p`` ascending (ties: family order); step ``i`` (1-based) uses the level ``fwer / (m - i + 1)``; the step
rejects when the hypothesis is testable and, for a hypothesis with an ``exact`` callable (the robust sign), ``exact(level)`` is
true -- the robust limit is evaluated AT the Holm level, not read off the p-value lattice (the lattice overshoots the infimum by up
to 1.16 %) --, else ``p <= level``; the first non-rejection stops the procedure.  Comparisons of a p-value with a level carry the relative tolerance ``rtol`` (1e-9: a rank
p-value that equals the level up to representation is a rejection, H4).  The lattice decision (``p <= level``) and the
Holm-adjusted p-value (``max_{j <= i} (m - j + 1) p_(j)``, capped at 1) are reported as well.
"""


def holm(hyps, fwer=0.05, rtol=1e-9):
    """``hyps``: list of dicts ``{id, p, testable, exact (callable level -> bool, optional)}`` in family order.
    Returns (list of per-hypothesis result dicts in family order, info)."""
    m = len(hyps)
    order = sorted(range(m), key=lambda i: (hyps[i]["p"] if hyps[i]["testable"] else 1.0, i))
    res = [None] * m
    stopped, cum = False, 0.0
    for rank, i in enumerate(order):
        h = hyps[i]
        p = float(h["p"]) if h["testable"] else 1.0
        level = fwer / (m - rank)
        cum = max(cum, (m - rank) * p)
        lattice = bool(h["testable"] and p <= level * (1.0 + rtol))
        if stopped or not h["testable"]:
            rej = False
        elif h.get("exact") is not None:
            rej = bool(h["exact"](level))
        else:
            rej = lattice
        if not rej:
            stopped = True
        res[i] = {"id": h["id"], "p": p, "testable": bool(h["testable"]), "rank": rank + 1, "level": level, "rejected": rej,
                  "lattice_rejected": lattice and not stopped_before(res, order, rank, rtol), "p_holm_adjusted": min(1.0, cum),
                  "decided_by": "untestable" if not h["testable"] else "exact" if h.get("exact") is not None else "p-value"}
    return res, {"m": m, "fwer": fwer, "n_rejected": sum(r["rejected"] for r in res)}


def stopped_before(res, order, rank, rtol=1e-9):
    """True if a lattice step-down would already have stopped before this rank (reporting of the lattice decision only)."""
    return any(not (res[j]["testable"] and res[j]["p"] <= res[j]["level"] * (1.0 + rtol)) for j in order[:rank])
