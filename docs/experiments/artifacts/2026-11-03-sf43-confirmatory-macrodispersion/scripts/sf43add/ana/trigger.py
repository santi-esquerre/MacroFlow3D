"""``j8-trigger``: the conditional extension J8 (``preregistration-addendum-1.md`` section 5), read only AFTER the freeze.

J8 runs iff ALL of: (a) J1 passed the reproduction check; (b) at least one IC90 (bootstrap-t) of r_H, r_D, q_H3, q_DH3 contains in its interior a
threshold the rule compares that quantity with; (c) GPU-h used + 1.2 x the projected J8 cost <= the 8 GPU-h cap; (d) the projected end of the J8 job
is not after the hard end (2026-11-07T03:00Z).  Prints ``J8 RUN`` | ``J8 SKIP`` and one line per condition (quantity names and thresholds, no interval).
"""
import json
from datetime import datetime, timezone

FACTOR = 1.2


def _utc(s):
    t = datetime.fromisoformat(s.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def evaluate(S, used, projected, end_utc):
    """-> (run: bool, [(label, ok, text)]) from an addendum summary dict."""
    cap, hard = float(S["config"]["budget_gpu_h_cap"]), S["config"]["hard_end_utc"]
    ins = S.get("j8_trigger_inputs")
    a = S["reproduction"].get("passed") is True
    contained = {k: v["contained"] for k, v in (ins or {}).get("quantities", {}).items() if v["contained"]}
    cost = used + FACTOR * projected
    conds = [("a", a, "J1 reproduction check %s" % ("passed" if a else "did not pass (or was not evaluated)")),
             ("b", bool(contained), "thresholds inside an IC90: %s" % (", ".join("%s %s" % (k, v) for k, v in contained.items()) if contained else
                                                                         "none" if ins else "J1 n=60 not evaluated")),
             ("c", cost <= cap, "GPU-h used %.3f + %.1f x projected %.3f = %.3f %s cap %.3f" % (used, FACTOR, projected, cost, "<=" if cost <= cap else ">", cap)),
             ("d", _utc(end_utc) <= _utc(hard), "projected end %s %s hard end %s" % (end_utc, "<=" if _utc(end_utc) <= _utc(hard) else ">", hard))]
    return all(c[1] for c in conds), conds


def main(a):
    with open(a.summary) as f:
        S = json.load(f)
    run, conds = evaluate(S, a.gpu_h_used, a.j8_gpu_h_projected, a.j8_end_utc)
    print("J8 RUN" if run else "J8 SKIP")
    for label, ok, text in conds:
        print("(%s) %s: %s" % (label, "ok" if ok else "NOT MET", text))
    return 0
