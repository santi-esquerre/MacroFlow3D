"""post hoc (2026-11-04) N5: ``alpha_LP``.

(a) the G5.4 error-budget table of the FROZEN rules (``sf43ana.gate4.RobustTest``: E_h with its fallback, E_W, E_WxH, E_int; the levels of
    ``report_levels`` = [0.05, 0.00625], the labels and seeds of the frozen run) applied to ``alpha_LP`` instead of the transverse estimators, at
    the four amplitudes, displacement basis (primary) and position basis; as a reproduction check the same code on ``alpha_TP`` / ``alpha_TAP``
    must give the committed ``budgets`` of results/analysis/summary.json (max relative difference reported).
(b) the paired ratio ``alpha_LP(s4_l16) / alpha_LP(s4_prod)`` (lambda/h 16 vs 32 at sigma^2 = 4) on the common seeds, production selection
    (consumer, plane, flux weights, per-plane set; both bases), ``mcstat.gate5.gate5_paired`` Student and bootstrap-t (B 2000, seed 20261104)
    90 % intervals, leave-one-out influence of every seed on the ratio of means (top 5 listed).
"""
import json
import os
import sys

import numpy as np

from common import ART, LABEL, dump, load, outdir
from mcstat import gate5 as mg5
from sf43ana.gate4 import RobustTest, design_from_arms
from sf43ana.gate5 import arm_series, pair_by_seed
from sf43ana.loader import build_data

LEVELS = np.array([0.05, 0.00625])
COMPS = ("E_h", "E_W", "E_WxH", "E_int")


def f(x):
    return [float(v) for v in np.ravel(x)]


def budget(pr, arms, amp, est, basis):
    d = design_from_arms(pr, amp, arms, est, basis)
    t = RobustTest(d, pr.null_value(amp), pr.raw, "%s/%s/%s" % (amp, est, basis))  # label = the frozen run's (seeds the bootstraps)
    if not t.testable:
        return {"testable": False, "reasons": t.missing}
    rep = t.report(LEVELS)
    m = float(d.prod.mean())
    out = {"testable": True, "N_production": int(d.prod.size), "production_mean": m, "K": int(rep["K"]), "levels": f(LEVELS),
           "L_a_over_2": f(rep["L_a_over_2"]), "RL": f(rep["RL"]), "components": {}}
    tot = np.zeros(2)
    for k in COMPS:
        if k not in rep["components"]:
            continue
        c = rep["components"][k]
        e = np.ravel(c["enforced"]).astype(float)
        tot += e
        det = c.get("details") or {}
        entry = {"enforced": f(e), "percent_of_mean": f(100.0 * e / m)}
        if k == "E_h":
            entry.update(order_branch_used=[bool(v) for v in np.ravel(det.get("order_branch_used"))], gate_ok=det.get("gate_ok"),
                         order_branch_value=f(det.get("order_branch_value")), fallback_value=f(det.get("fallback_value")),
                         p_L=f(det.get("p_L")), p_use=f(det.get("p_use")), order_hat=det.get("order_hat"), delta_hat=det.get("delta_hat"),
                         fallback_fires=[bool(not u) for u in np.ravel(det.get("order_branch_used"))] if det.get("gate_ok") else "gate failed")
        else:
            entry.update(estimate=det.get("estimate"), se=det.get("se"))
        out["components"][k] = entry
    out["sum_components"] = f(tot)
    out["sum_percent_of_mean"] = f(100.0 * tot / m)
    dom = [max(out["components"], key=lambda k: out["components"][k]["enforced"][i]) for i in range(2)]
    out["dominant_component"] = dom
    out["ladder_means"] = f(d.levels.mean(axis=0)) if d.levels is not None else None
    out["ladder_resolutions"] = list(d.res) if d.res is not None else None
    return out, rep


def rel(a, b):
    a, b = np.ravel(np.asarray(a, float)), np.ravel(np.asarray(b, float))
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def part_a(pr, arms):
    out = {"label": LABEL, "levels": f(LEVELS), "table": {}, "reproduction_alpha_T": {}}
    with open(os.path.join(ART, "results", "analysis", "summary.json")) as fh:
        frozen = json.load(fh)["budgets"]
    for basis in ("disp", "pos"):
        for amp in pr.amps:
            if not pr.amp(amp).get("components"):
                continue
            r = budget(pr, arms, amp, "alpha_LP", basis)
            out["table"].setdefault(basis, {})[amp] = r[0] if isinstance(r, tuple) else r
    for amp in ("s025", "s1", "s225", "s4"):
        for est in ("alpha_TP", "alpha_TAP"):
            r, rep = budget(pr, arms, amp, est, "disp")
            fz = frozen[amp]["disp"][est]
            dev = {"RL": rel(rep["RL"], fz["RL"]), "L_a_over_2": rel(rep["L_a_over_2"], fz["L_a_over_2"])}
            for k, c in rep["components"].items():
                dev[k] = rel(c["enforced"], fz["components"][k]["enforced"])
            out["reproduction_alpha_T"]["%s/%s" % (amp, est)] = dev
    return out


def part_b(pr, arms, ledger):
    data = build_data(pr, arms, ledger)
    out = {"label": LABEL, "B": 2000, "seed": 20261104, "level": 0.90, "bases": {}}
    for basis, col in (("disp", "alpha_LP:disp"), ("pos", "alpha_LP")):
        s1, v1 = arm_series(data, "s4_l16", col)
        s2, v2 = arm_series(data, "s4_prod", col)
        seeds, L, P, info = pair_by_seed(s1, v1, s2, v2)
        n = L.size
        st = mg5.gate5_paired(L, P, level=0.90)
        bt = mg5.gate5_paired(L, P, level=0.90, method="bootstrap-t", B=2000, rng=np.random.default_rng(np.random.SeedSequence([20261104, 5])))
        ratio = L.mean() / P.mean()
        loo = np.array([np.delete(L, i).mean() / np.delete(P, i).mean() for i in range(n)])
        infl = loo - ratio
        top = np.argsort(-np.abs(infl))[:5]
        out["bases"][basis] = {
            "column": col, "arms": {"numerator": "s4_l16 (lambda/h 16)", "denominator": "s4_prod (lambda/h 32)"}, "pairing": info, "n": int(n),
            "mean_l16": float(L.mean()), "mean_prod": float(P.mean()), "ratio_of_means": float(ratio),
            "student": {"ratio_lo": float(st.ratio_lo), "ratio_hi": float(st.ratio_hi), "se_log": float(st.se), "dof": int(st.dof)},
            "bootstrap_t": {"ratio_lo": float(bt.ratio_lo), "ratio_hi": float(bt.ratio_hi), "se_log": float(bt.se), "B": 2000},
            "loo_ratio_min": float(loo.min()), "loo_ratio_max": float(loo.max()), "loo_max_abs_change": float(np.abs(infl).max()),
            "top5_influence": [{"seed": int(seeds[i]), "alpha_LP_l16": float(L[i]), "alpha_LP_prod": float(P[i]), "ratio_without_seed": float(loo[i]),
                                "change": float(infl[i]), "change_percent": float(100 * infl[i] / ratio)} for i in top],
            "sd_l16": float(L.std(ddof=1)), "sd_prod": float(P.std(ddof=1)), "corr": float(np.corrcoef(L, P)[0, 1]),
            "skewness_log_ratio_terms_note": "per-seed ratio L_r/P_r: min %.3f median %.3f max %.3f" % (
                (L / P).min(), np.median(L / P), (L / P).max())}
    return out


def main():
    pr, arms, ledger = load(sys.argv[1])
    d = outdir("n5")
    a = part_a(pr, arms)
    dump(os.path.join(d, "n5a_budget_table.json"), a)
    b = part_b(pr, arms, ledger)
    dump(os.path.join(d, "n5b_l16_vs_prod.json"), b)
    for basis, tb in a["table"].items():
        for amp, r in tb.items():
            if not r.get("testable"):
                print(basis, amp, "untestable", r.get("reasons"))
                continue
            cs = "  ".join("%s %s" % (k, ["%.2f%%" % x for x in c["percent_of_mean"]]) for k, c in r["components"].items())
            print(basis, amp, "N=%d mean=%.4g" % (r["N_production"], r["production_mean"]), cs, "SUM", ["%.2f" % x for x in r["sum_percent_of_mean"]],
                  "dom", r["dominant_component"], "E_h order used", r["components"]["E_h"]["order_branch_used"], "gate_ok", r["components"]["E_h"]["gate_ok"])
    print("reproduction (max rel dev vs committed budgets):", json.dumps({k: max(v.values()) for k, v in a["reproduction_alpha_T"].items()}))
    for basis, r in b["bases"].items():
        print(basis, "n=%d ratio %.4f Student [%.4f, %.4f] boot-t [%.4f, %.4f]" % (r["n"], r["ratio_of_means"], r["student"]["ratio_lo"], r["student"]["ratio_hi"],
              r["bootstrap_t"]["ratio_lo"], r["bootstrap_t"]["ratio_hi"]), "LOO max |change| %.4f" % r["loo_max_abs_change"], "corr %.3f" % r["corr"])
        for t in r["top5_influence"]:
            print("    ", t)


if __name__ == "__main__":
    main()
