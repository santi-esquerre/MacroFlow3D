"""Default-method results with fixed seeds (``golden_default_methods.json``): recorded from the code BEFORE the studentized options
were added (SF-42 N6c), to prove that the defaults (percentile contrast / ladder bounds, Student TOST) did not change.

``cases()`` recomputes them; ``python3 -m mcstat.tests._golden <path>`` rewrites the JSON (only to be done knowingly).
"""
import json
import sys

import numpy as np

from mcstat import budget as bd
from mcstat import gate5, intervals as iv

TAILS = [1e-3, 3e-3, 1e-2, 0.05]
A_LEVELS = [0.01, 0.05, 0.3]


def _arr(x):
    return np.asarray(x, dtype=float).tolist()


def cases():
    out = {}
    rng = np.random.default_rng(20261030)
    u = rng.normal(0.4, 1.0, size=(5, 30))
    v = rng.normal(0.0, 1.0, size=(5, 30))
    w = rng.normal(0.1, 1.5, size=(5, 25))
    for share in (False, True):
        tag = "share" if share else "indep"
        cb = bd.e_int(u, v, B=999, seed=1, share_draws=share)
        out["e_int_" + tag] = {"estimate": _arr(cb.estimate), "bound": _arr(cb.bound(TAILS))}
        cb = bd.e_w(u, w, B=999, seed=2, share_draws=share)
        out["e_w_" + tag] = {"estimate": _arr(cb.estimate), "bound": _arr(cb.bound(TAILS))}
        cb = bd.e_interaction(u[:, :25], v[:, :25], w, 0.5 * w + 0.1 * v[:, :25], B=999, seed=3, share_draws=share)
        out["e_wxh_" + tag] = {"estimate": _arr(cb.estimate), "bound": _arr(cb.bound(TAILS))}
    # ladders: order branch (monotone power law with noise), finest production level, fallback (non-monotone), 4-level fallback
    errs = 0.05 * (np.array([5.0, 10.0, 20.0]) / 10.0) ** (-2.0)
    lev = rng.normal(0, 1, size=(4, 40, 1)) + errs[None, None, :] + rng.normal(0, 0.01, size=(4, 40, 3))
    for ip in (1, 2):
        lb = bd.LadderBound(lev, [5.0, 10.0, 20.0], ip, B=999, seed=4)
        d = lb.details(TAILS)
        out["ladder_ip%d" % ip] = {"bound": _arr(lb.bound(TAILS)), "order_hat": _arr(lb.order_hat), "p_L": _arr(d["p_L"]),
                                   "used": np.asarray(d["order_branch_used"]).astype(int).tolist()}
    nm = rng.normal(0, 1, size=(4, 40, 1)) + 0.05 * np.array([0.0, 1.0, 0.5])[None, None, :] + rng.normal(0, 0.02, size=(4, 40, 3))
    lb = bd.LadderBound(nm, [5.0, 10.0, 20.0], 1, B=999, seed=5, share_draws=True)
    out["ladder_fallback"] = {"bound": _arr(lb.bound(TAILS)), "used": np.asarray(lb.details(TAILS)["order_branch_used"]).astype(int).tolist()}
    lev4 = rng.normal(0, 1, size=(3, 30, 1)) + 0.02 * np.array([0.0, 1.0, 0.1, 0.9])[None, None, :] + rng.normal(0, 0.02, size=(3, 30, 4))
    lb = bd.LadderBound(lev4, [4.0, 8.0, 16.0, 32.0], 2, B=999, seed=6)
    out["ladder_4level"] = {"bound": _arr(lb.bound(TAILS)), "range_hat": _arr(lb.range_hat)}
    # robust limit and p-value
    x = rng.normal(0.5, 1.0, size=(3, 60))
    ladder = bd.LadderBound(lev[:3], [5.0, 10.0, 20.0], 1, B=999, seed=7)
    comps = [ladder, bd.e_int(u[:3], v[:3], B=999, seed=8), bd.e_w(u[:3], w[:3], B=999, seed=9)]
    rl = bd.RobustLimit(bd.StudentMean(x), comps)
    pv = rl.p_value()
    out["robust"] = {"RL": _arr(rl.lower(A_LEVELS)), "p": _arr(pv.p)}
    # Gate 5 (Student) and bootstrap-t of a mean
    L = rng.lognormal(0.0, 0.3, size=(4, 40))
    P = L * rng.lognormal(0.02, 0.1, size=(4, 40))
    r = gate5.gate5_paired(L, P)
    out["gate5"] = {k: _arr(r[k]) for k in ("theta", "se", "lo", "hi", "p_tost", "p1", "p2")}
    out["gate5"]["verdict"] = [str(s) for s in r["verdict"]]
    bt = iv.bootstrap_t(x, B=499, seed=10)
    out["bootstrap_t"] = {"lower": _arr(iv.bootstrap_t_limit(bt, TAILS, "lower")), "upper": _arr(iv.bootstrap_t_limit(bt, TAILS, "upper"))}
    return out


if __name__ == "__main__":
    with open(sys.argv[1], "w") as f:
        json.dump(cases(), f, indent=1, sort_keys=True)
