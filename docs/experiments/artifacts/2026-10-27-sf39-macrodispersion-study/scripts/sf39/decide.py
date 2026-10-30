"""SF-39 C-STATS part 2: decision logic (F4 production-h verdict, rule D, amplitude label, Gate 5).

Exactly the specification's text as clarified by preregistration.md section 4 ("D (rule D)"): rule D is
evaluated per ladder; branch 2 is existential over qualifying differences; branch 3 is as written (no
"other's point estimate" clause); the amplitude label needs the identical label of both estimators.
All intervals are 90 % two-sided Student t unless stated.  Units of lambda.
"""
from __future__ import annotations

import numpy as np

from .stats import (NUM_CUT, ancova_vs, equivalence, t_interval, welch_interval)

DISCORDANT = "unresolved (estimator-discordant)"
TWO_P15 = 2.0**1.5


def _ci(x, level=0.90):
    mean, lo, hi, n, _ = t_interval(x, level)
    return {"mean": mean, "lo": lo, "hi": hi, "n": n}


def production_h_verdict(D_c, D_f, D_tw, m: float):
    """F4.  D_c = prod - 1/2 rung, D_f = 2x - prod (paired by seed, same length), D_tw = prod - twin.

    converged iff narrow verdict converged AND CI(D_tw) inside +-2^1.5 m AND the 90 % CI of
    s (2^1.5 D_f - D_c) is not entirely above 0 (s = sign of mean D_c, +1 if exactly 0);
    not converged iff the narrow verdict is not converged; else inconclusive.
    """
    D_c, D_f, D_tw = (np.asarray(v, float) for v in (D_c, D_f, D_tw))
    if D_c.shape != D_f.shape:
        raise ValueError("D_c and D_f must be paired by seed (same length)")
    f = _ci(D_f)
    narrow = equivalence(f["lo"], f["hi"], m)
    tw = _ci(D_tw)
    tw_inside = bool(tw["lo"] >= -TWO_P15 * m and tw["hi"] <= TWO_P15 * m)
    s = -1.0 if D_c.mean() < 0 else 1.0
    order = _ci(s * (TWO_P15 * D_f - D_c))
    order_ok = not (order["lo"] > 0)
    if narrow == "converged" and tw_inside and order_ok:
        verdict = "converged"
    elif narrow == "not converged":
        verdict = "not converged"
    else:
        verdict = "inconclusive"
    return {"verdict": verdict, "narrow": narrow, "narrow_ci": f, "tw_inside": tw_inside, "tw_ci": tw,
            "order_ok": order_ok, "order_ci": order, "s": s, "m": m}


def hmax_verdict(D_hmax, m: float):
    """h_max verdict: equivalence of the 90 % CI of (h/16 - h/8) against +-m (m at the h/16 mean)."""
    c = _ci(D_hmax)
    return {"verdict": equivalence(c["lo"], c["hi"], m), "ci": c, "m": m}


def rule_d(ladders: dict, prod_values):
    """Gate 4 classification of one transverse estimator (rule D, branches 1-5 in order).

    ladders = {name: {'verdict': str, 'diffs': {dname: array}, 'finer': {dname: array}}}, normally
    'h' (diffs D_f, D_tw; finer: 2x-rung values, production values) and 'hmax' (diff d16_8; finer: h/16
    values).  Each branch is evaluated over the ladders in dict order (a branch fires if it holds for
    either) and, within a ladder, over its differences.  The ladder verdict and the refinement
    difference of one branch always belong to the same ladder; "the other" difference is the rest of
    that ladder (vacuous when it has one difference).
    Returns {'label', 'branch', 'ladder', 'difference', 'U', 'trace', 'prod_ci'}.
    """
    info = {}
    for lname, lad in ladders.items():
        d = {k: _ci(v) for k, v in lad["diffs"].items()}
        fin = {k: _ci(v) for k, v in lad.get("finer", {}).items()}
        info[lname] = (lad["verdict"], d, fin)

    def others(d, k):
        return [v for kk, v in d.items() if kk != k]

    trace = []

    def result(label, branch, lname=None, dname=None, U=None):
        return {"label": label, "branch": branch, "ladder": lname, "difference": dname, "U": U,
                "trace": trace, "prod_ci": prod_ci}

    prod_ci = _ci(prod_values)

    # 1. grid-limited, lower bound
    for lname, (verdict, d, fin) in info.items():
        for k, c in d.items():
            ok = verdict != "converged" and c["lo"] > 0 and all(not (o["mean"] < 0) for o in others(d, k))
            trace.append({"branch": 1, "ladder": lname, "difference": k, "holds": bool(ok)})
            if ok:
                return result("unresolved (grid-limited, lower bound)", 1, lname, k)
    # 2. numerical (existential over qualifying differences)
    for lname, (verdict, d, fin) in info.items():
        for k, c in d.items():
            if verdict == "not converged" and c["hi"] < 0 and all(not (o["mean"] > 0) for o in others(d, k)):
                f = fin[k]
                vanishes = bool((f["lo"] <= 0 <= f["hi"]) or f["hi"] <= NUM_CUT)
                trace.append({"branch": 2, "ladder": lname, "difference": k, "qualifies": True, "holds": vanishes,
                              "finer_ci": f})
                if vanishes:
                    return result("numerical", 2, lname, k)
            else:
                trace.append({"branch": 2, "ladder": lname, "difference": k, "qualifies": False, "holds": False})
    # 3. grid-limited, upper bound (as written: no clause on the other difference)
    for lname, (verdict, d, fin) in info.items():
        for k, c in d.items():
            ok = verdict != "converged" and c["hi"] < 0
            trace.append({"branch": 3, "ladder": lname, "difference": k, "holds": bool(ok)})
            if ok:
                return result("unresolved (grid-limited, upper bound)", 3, lname, k)
    # 4. physical iff every ladder verdict converged
    both = all(v == "converged" for v, _, _ in info.values())
    trace.append({"branch": 4, "holds": bool(both), "prod_ci": prod_ci})
    if both:
        if prod_ci["lo"] > 0:
            return result("physical (positive)", 4)
        if prod_ci["hi"] < 0:
            return result("physical (negative)", 4)
        if prod_ci["hi"] <= NUM_CUT:
            return result("physical (bounded)", 4, U=prod_ci["hi"])
        return result("unresolved (imprecise)", 4)
    return result("unresolved", 5)


def amplitude_label(label_TAM: str, label_TP: str) -> str:
    """Common label (sub-label included) of the two transverse estimators, else discordant."""
    return label_TAM if label_TAM == label_TP else DISCORDANT


def gate5_verdict(L_alpha, par2_alpha, L_alpha_fv=None, par2_sigma2=None, sigma2_nominal=None):
    """Gate 5: Welch 90 % CI of mean(L) - mean(Par2) against the fixed margin 0.10 mean(Par2).

    Optional sensitivities: (i) L_alpha_fv = Lagrangian alpha with FV qbar normalization (same
    margin); (ii) ANCOVA (needs par2_sigma2, sigma2_nominal): L - a, margin 0.10 a.
    Labels agree / disagree / inconclusive.
    """
    def one(L, P):
        diff, lo, hi, nu, se = welch_interval(L, P)
        m = 0.10 * abs(float(np.mean(P)))
        return {"diff": diff, "lo": lo, "hi": hi, "nu": nu, "margin": m,
                "verdict": equivalence(lo, hi, m, "gate5")}

    out = {"primary": one(L_alpha, par2_alpha)}
    if L_alpha_fv is not None:
        out["fv_qbar"] = one(L_alpha_fv, par2_alpha)
    if par2_sigma2 is not None:
        diff, lo, hi, nu, anc = ancova_vs(L_alpha, par2_alpha, par2_sigma2, sigma2_nominal)
        m = 0.10 * abs(anc.a)
        out["ancova"] = {"diff": diff, "lo": lo, "hi": hi, "nu": nu, "margin": m, "a": anc.a, "se_a": anc.se_a,
                         "verdict": equivalence(lo, hi, m, "gate5")}
    return out


_ROW_KEYS = ("amplitude", "estimator", "decision_point", "verdict", "ci_lo", "ci_hi", "n", "note")


def decision_row(amplitude, estimator, decision_point, verdict, ci=None, n=None, note=""):
    """One row of the section-10 table; ci = (lo, hi) or None."""
    lo, hi = ci if ci is not None else (None, None)
    return dict(zip(_ROW_KEYS, (amplitude, estimator, decision_point, verdict, lo, hi, n, note)))


def decision_table_rows(entries):
    """entries: iterable of dicts with the keys of decision_row (ci given as 'ci' tuple or ci_lo/ci_hi)."""
    rows = []
    for e in entries:
        e = dict(e)
        ci = e.pop("ci", None) or (e.pop("ci_lo", None), e.pop("ci_hi", None))
        if ci == (None, None):
            ci = None
        rows.append(decision_row(e["amplitude"], e["estimator"], e["decision_point"], e["verdict"],
                                 ci, e.get("n"), e.get("note", "")))
    return rows
