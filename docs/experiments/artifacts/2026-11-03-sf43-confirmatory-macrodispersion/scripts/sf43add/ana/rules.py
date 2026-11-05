"""The pre-registered rules of addendum 1 (``preregistration-addendum-1.md`` section 3, thresholds read from ``prereg_addendum.json``).

Conditions are evaluated generically from the JSON (``[lhs, op, rhs]``): ``lhs`` is ``name`` (the point ratio), ``abs(name-1)``, ``ci(name)`` or
``ci(name).lo|hi``; ``op`` one of ``>= <= < >`` or ``within`` (closed: ``lo >= a and hi <= b``).  Every condition of every class is evaluated and stored
with its value and truth (``None`` = the quantity was not evaluable, counted as false); the first class whose conditions all hold wins.
IC90 of the rules are the bootstrap-t ones (``stats.paired_ratio``); the Student ones are reported beside them.
"""
import re

import numpy as np

from . import stats
from .data import LP, TP_FAR, TP_MAIN, common_all, pair

QUANTILES = (0, 5, 25, 50, 75, 95, 100)
J3_CLASSES = ("lambda/h = 5 equivalente", "lambda/h = 5 no equivalente")


def ratio(va, ka, vb, kb, label=""):
    """Paired ratio ``mean(a)/mean(b)`` on the common finite seeds of two views (an absent view: not evaluable)."""
    if va is None or vb is None:
        return stats.not_evaluable(0, "arm not available (%s)" % label)
    _, x, y, info = pair(va, ka, vb, kb)
    r = stats.paired_ratio(x, y)
    r["pairing"] = info
    return r


# ----------------------------------------------------------------------------------------------------------------------------------
# conditions
# ----------------------------------------------------------------------------------------------------------------------------------
_LHS = re.compile(r"(?:(abs)\(([\w/]+)-1\)|ci\(([\w/]+)\)(?:\.(lo|hi))?|([\w/]+))$")


def eval_cond(Q, lhs, op, rhs):
    """-> {"lhs", "op", "rhs", "value", "truth"} for one condition against the quantities ``Q`` (name -> result dict)."""
    m = _LHS.match(lhs)
    if not m:
        raise ValueError("unsupported condition %r" % (lhs,))
    absn, name, ci_name, side, plain = m.groups()
    q = Q.get(name or ci_name or plain)
    out = {"lhs": lhs, "op": op, "rhs": rhs, "value": None, "truth": None}
    if q is None or not q.get("evaluable"):
        return out
    if absn:
        v = abs(q["ratio"] - 1.0)
    elif ci_name:
        v = q["ci90"] if side is None else q["ci90"][0 if side == "lo" else 1]
    else:
        v = q["ratio"]
    out["value"] = v
    out["truth"] = bool({">=": lambda: v >= rhs, "<=": lambda: v <= rhs, "<": lambda: v < rhs, ">": lambda: v > rhs,
                         "within": lambda: stats.within(v, rhs)}[op]())
    return out


def first_match(Q, classes):
    """-> (class name or None, {class name: [conditions]}); a class with no ``all`` list is the fall-through and matches last."""
    conds, hit = {}, None
    for c in classes:
        if "all" not in c:
            continue
        conds[c["name"]] = [eval_cond(Q, *t) for t in c["all"]]
        if hit is None and all(x["truth"] is True for x in conds[c["name"]]):
            hit = c["name"]
    return hit, conds


def _fall_entry(classes):
    return [c for c in classes if "all" not in c][-1]


def _fall_through(classes):
    return _fall_entry(classes)["name"]


# ----------------------------------------------------------------------------------------------------------------------------------
# J1
# ----------------------------------------------------------------------------------------------------------------------------------
def completeness(v):
    """Mean, fraction of complete realizations (``n_last == n_traced``) and quantiles of ``n_last / n_traced`` of a view (None: no counts)."""
    c = None if v is None else v.completeness()
    if c is None:
        return {"evaluable": False, "reason": "no crossing counts in the reduced tables" if v is not None else "arm not available"}
    ok = np.isfinite(c)
    if not ok.any():
        return {"evaluable": False, "reason": "no finite completeness", "n": 0}
    comp = v.complete()
    return {"evaluable": True, "n": int(ok.sum()), "mean": float(c[ok].mean()), "fraction_complete": float(comp[ok].mean()),
            "quantiles": {str(q): float(np.percentile(c[ok], q)) for q in QUANTILES}}


def manipulation(spec, comp):
    """Manipulation check on the horizon arm: mean completeness and fraction of complete realizations above the pre-registered minima."""
    if not comp.get("evaluable"):
        return dict(comp, arm=spec["arm"], passed=None)
    ok = comp["mean"] >= spec["mean_completeness_min"] and comp["fraction_complete"] >= spec["fraction_complete_min"]
    return {"evaluable": True, "arm": spec["arm"], "n": comp["n"], "mean": comp["mean"], "fraction_complete": comp["fraction_complete"],
            "mean_completeness_min": spec["mean_completeness_min"], "fraction_complete_min": spec["fraction_complete_min"], "passed": bool(ok)}


def evaluate_j1(add, views, repro_result, label):
    """J1 at one sample size.  ``views``: arm id -> View (frozen arms already restricted to the seeds of the label; an absent arm gives
    "not evaluable" entries)."""
    j1 = add["j1"]
    Q = {n: ratio(views.get(a), LP, views.get(b), LP, "%s/%s" % (a, b)) for n, (a, b) in j1["quantities"].items()}
    comp = {a: completeness(views.get(a)) for a in ("s4_g5p", "j1_base10", "j1_H2", "j1_H3", "j1_D", "j1_DH3")}
    spec = j1["manipulation"]
    man = manipulation(spec, comp[spec["arm"]])
    rep, qb = j1["representativity"], Q[j1["representativity"]["quantity"]]
    fz = rep["frozen"]
    repres = {"quantity": rep["quantity"], "frozen": fz, "result": qb}
    if qb.get("evaluable"):
        repres.update(frozen_ratio_in_bootstrap_t_ci=bool(qb["ci90"][0] <= fz["ratio"] <= qb["ci90"][1]),
                      frozen_ratio_in_student_ci=bool(qb["student_ci90"][0] <= fz["ratio"] <= qb["student_ci90"][1]),
                      student_ci_overlap_with_frozen=bool(qb["student_ci90"][0] <= fz["student_ci90"][1] and fz["student_ci90"][0] <= qb["student_ci90"][1]))
    hit, conds = first_match(Q, j1["classes"])
    missing = [n for n in ("r_H", "r_D", "q_H3", "q_DH3") if not Q[n].get("evaluable")]
    if not repro_result or not repro_result.get("evaluated"):
        status, cls, why = "not classified", None, "reproduction check (3.1.1) not evaluated"
    elif not repro_result["passed"]:
        status, cls, why = "not classified", None, "reproduction check (3.1.1) failed"
    elif man["passed"] is None:
        status, cls, why = "not classified", None, "manipulation check not evaluable (%s)" % man.get("reason", "")
    elif not man["passed"]:
        status, cls, why = "classified", spec["on_fail"], "manipulation check failed: classes 1-3 not assigned"
    elif missing:
        status, cls, why = "not evaluable", None, "not evaluable: %s" % ", ".join(missing)
    else:
        status, cls, why = "classified", hit or _fall_through(j1["classes"]), "first match of the pre-registered order" if hit else "no earlier class holds (fall-through)"
    # Amendment A1 (pre-run): the fall-through class's `override_if` (here ci(q_DH3).lo > 1.10) forces it over a class 1-3 match; classification path only
    fall = _fall_entry(j1["classes"])
    ov_conds = [eval_cond(Q, *t) for t in fall.get("override_if", [])]
    holds = bool(ov_conds) and all(c["truth"] is True for c in ov_conds)
    before = cls if status == "classified" and man["passed"] is True else None
    applied = before is not None and holds and before != fall["name"]
    if applied:
        why = "Amendment A1 override (%s) gives %s over the first match %s" % ("; ".join("%s %s %s" % tuple(t) for t in fall["override_if"]), fall["name"], before)
        cls = fall["name"]
    return {"label": label, "quantities": Q, "completeness": comp, "manipulation": man, "representativity": repres, "conditions": conds,
            "status": status, "class": cls, "reason": why, "class_before_override": before,
            "override": {"conditions": ov_conds, "holds": holds, "applied": bool(applied), "note": fall.get("override_note")}}


def j8_inputs(add, j1_result):
    """Which IC90 (bootstrap-t) of r_H, r_D, q_H3, q_DH3 contain, in their interior, a threshold the rule compares the quantity with."""
    out = {}
    for name, thr in add["j1"]["j8_trigger"]["thresholds"].items():
        q = j1_result["quantities"][name]
        if q.get("evaluable"):
            lo, hi = q["ci90"]
            out[name] = {"ci90": [lo, hi], "thresholds": thr, "contained": [x for x in thr if lo < x < hi]}
        else:
            out[name] = {"ci90": None, "thresholds": thr, "contained": [], "reason": q["reason"]}
    return {"quantities": out, "any_contained": any(v["contained"] for v in out.values())}


# ----------------------------------------------------------------------------------------------------------------------------------
# J3, J9
# ----------------------------------------------------------------------------------------------------------------------------------
def _present(v):
    return v is not None and v.N > 0


def evaluate_j3(add, views):
    j3 = add["j3"]
    v, ref = views.get(j3["arm"]), views.get(j3["reference"])
    Q = {"m_main": ratio(v, TP_MAIN, ref, TP_MAIN, "j3/s1"), "m_far": ratio(v, TP_FAR, ref, TP_FAR, "j3/s1"),
         "F5": ratio(v, TP_FAR, v, TP_MAIN, "j3 far/main")}
    if _present(v) and _present(ref):
        _, (jf, df, jm, dm) = common_all([(v, TP_FAR), (ref, TP_FAR), (v, TP_MAIN), (ref, TP_MAIN)])
        Q["m_far/m_main"] = stats.log_contrast([jf, df, jm, dm], [1, -1, -1, 1])
    else:
        Q["m_far/m_main"] = stats.not_evaluable(0, "arm not available")
    conds = [eval_cond(Q, k, "within", b) for k, b in j3["equivalent"].items()]
    eq = all(c["truth"] is True for c in conds)
    return {"quantities": Q, "conditions": conds, "equivalent": eq, "class": J3_CLASSES[0] if eq else J3_CLASSES[1],
            "reason": "all three IC90 within their margins" if eq else
            ("an IC90 is outside its margin" if all(c["truth"] is not None for c in conds) else "j3_l5 missing or not evaluable")}


def evaluate_j9(add, views, j3_result):
    j9, ref, v, j3v = add["j9"], views.get("s1_dist"), views.get(add["j9"]["arm"]), views.get(add["j3"]["arm"])
    n_done = int((np.isfinite(v.get(TP_MAIN)) & np.isfinite(v.get(TP_FAR))).sum()) if v is not None else 0
    Q = {"F48": ratio(v, TP_FAR, v, TP_MAIN, "j9 far/main"), "F5": ratio(j3v, TP_FAR, j3v, TP_MAIN, "j3 far/main"),
         "F24": ratio(ref, TP_FAR, ref, TP_MAIN, "s1 far/main")}
    for name, other in (("F48/F5", j3v), ("F48/F24", ref)):
        if _present(v) and _present(other):
            Q[name] = stats.unpaired_ratio(_far_main(v), _far_main(other))
        else:
            Q[name] = stats.not_evaluable(0, "arm not available")
    desc = {}
    for tag, key in (("main", TP_MAIN), ("far", TP_FAR)):
        if _present(v) and _present(j3v):
            x, y = v.get(key), j3v.get(key)
            desc["W48/W24 %s" % tag] = stats.unpaired_ratio(([x[np.isfinite(x)]], [1]), ([y[np.isfinite(y)]], [1]))
        else:
            desc["W48/W24 %s" % tag] = stats.not_evaluable(0, "arm not available")
    hit, conds = first_match(Q, j9["classes"])
    if n_done < j9["min_completed"]:
        cls, why = j9["classes"][0]["name"], "fewer than %d j9_w48 realizations completed (precedence)" % j9["min_completed"]
    else:
        cls, why = hit or _fall_through(j9["classes"]), "first match of the pre-registered order" if hit else "no earlier class holds (fall-through)"
    return {"n_completed": n_done, "min_completed": j9["min_completed"], "quantities": Q, "conditions": conds, "class": cls, "reason": why,
            "F24_frozen": j9["compare"]["F24_frozen"], "descriptive_levels": desc, "j3_equivalent": j3_result["equivalent"],
            "j3_qualification": None if j3_result["equivalent"] else "lambda/h = 5 grid bias is a caveat: J9 is reported with no width conclusion"}


def _far_main(v):
    """Group ``(series, signs)`` of the per-realization far/main contrast of a view on its finite seeds."""
    _, (a, b) = common_all([(v, TP_FAR), (v, TP_MAIN)])
    return [a, b], [1, -1]
