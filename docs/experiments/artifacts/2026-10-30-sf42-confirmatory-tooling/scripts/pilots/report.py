"""Local report generator of the SF-42 R4 pilots from the pulled host data (design inputs; no interpretation of any alpha).

    python3 -m pilots.report j1|j1b|j2|j3|j3b --data <dir with J1, J2, J3, hostref.json> --out <R/pilots dir>

``--data`` layout (copied from ``~/sf42_campaign/pilots``): ``<job>/<run id>/RUN.json``, ``<job>/reduced/<run id>.csv``,
``hostref.json`` (SF-39 PCG iteration reference, planning SDs).  Outputs: ``j1_grid.{json,csv}``, ``j2_particles.{json,csv}``,
``j3_pairing.json`` in ``--out``.  The cost fit and the SF-43 projection are made by ``pilots.report cost`` (see ``cost_report``).
"""
import argparse
import csv
import json
import math
import os
import re
import sys

import numpy as np
from scipy import stats

from seedlattice import rule

from mcstat import costmodel as cm

from . import analyze as an
from . import cost as costmod
from . import design

AMP_KEY = {"0.25": "s025", "1": "s1", "2.25": "s225", "4": "s4"}
RATIO_LIMIT = 3.0


def load_runs(data, job):
    out = []
    jd = os.path.join(data, job)
    for name in sorted(os.listdir(jd)) if os.path.isdir(jd) else []:
        p = os.path.join(jd, name, "RUN.json")
        if os.path.exists(p):
            with open(p) as f:
                out.append(json.load(f))
    return out


def write_csv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(cols)
        for r in rows:
            w.writerow([("%.10g" % r[c]) if isinstance(r.get(c), float) else r.get(c, "") for c in cols])


# ----------------------------------------------------------------------------------------------------------------------
def j1(data, out, job="J1", name="j1_grid"):
    with open(os.path.join(data, "hostref.json")) as f:
        ref = json.load(f)["pcg"]
    rows = []
    for r in load_runs(data, job):
        row = {"id": r["id"], "code": r["code"], "grid": "%dx%dx%d" % (r["nx"], r["ny"], r["nz"]), "lh": r["lh"], "amp": r["amp"],
               "M": r["M"], "seed0": r["seed0"], "exit": r["exit"], "wall_s": r["wall_s"], "peak_gpu_mib": r["peak_gpu_mib_own_pid"]}
        if r["code"] == "consumer":
            recs = [x for x in r["realizations"] if x.get("iterations") is not None]
            row["pcg_iterations"] = recs[0]["iterations"] if recs else None
            row["verdict"] = recs[0]["verdict"] if recs else "no_solver_record"
            sec = (recs[0].get("seconds") or {}) if recs else {}
            row["flow_stage_s"] = sec.get("stage")
            row["track_s"], row["prepare_s"] = sec.get("track"), sec.get("prepare")
            row["total_s_runner"] = sec.get("total")
            eb = [int(m.group(1)) for line in ((recs[0].get("mem_lines_stage") or []) + [recs[0].get("mem_line_bridge") or ""]) if line
                  for m in [re.search(r"exact_bytes=(\d+)", line)] if m] if recs else []
            row["runner_exact_bytes_max_mib"] = max(eb) / 2 ** 20 if eb else None
        else:
            sv = r.get("par2_solves") or []
            row["pcg_iterations"] = sv[0]["iterations"] if sv else None
            row["verdict"] = ("converged" if sv and sv[0]["converged"] == "YES" else "not_converged") if sv else "no_solver_record"
            st = r.get("par2_stage_times") or []
            if st:
                row["flow_stage_s"], row["solve_s"] = st[0]["flow_stage_s"], st[0]["solve_s"]
                row["track_s"] = st[0]["transport_s"]
        m = ref[AMP_KEY[r["amp"]]]["production"]["median"]
        row["sf39_median_iterations"] = m
        row["ratio_to_sf39_median"] = row["pcg_iterations"] / m if row["pcg_iterations"] is not None else None
        row["different_grid_required"] = bool(row["ratio_to_sf39_median"] is not None and row["ratio_to_sf39_median"] > RATIO_LIMIT) or \
            row["verdict"] != "converged" or r["exit"] != 0
        rows.append(row)
    cols = ["id", "code", "grid", "lh", "amp", "M", "seed0", "exit", "wall_s", "flow_stage_s", "solve_s", "prepare_s", "track_s", "total_s_runner", "peak_gpu_mib", "runner_exact_bytes_max_mib",
            "pcg_iterations", "verdict", "sf39_median_iterations", "ratio_to_sf39_median", "different_grid_required"]
    write_csv(os.path.join(out, name + ".csv"), rows, cols)
    with open(os.path.join(out, name + ".json"), "w") as f:
        json.dump({"rule": "iterations > %g x SF-39 production median (A+B arms, same amplitude) or not converged: requires a different grid" % RATIO_LIMIT,
                   "sf39_reference": ref, "rows": rows}, f, indent=1)
    return rows


# ----------------------------------------------------------------------------------------------------------------------
def j2(data, out):
    with open(os.path.join(data, "hostref.json")) as f:
        sd = json.load(f)["planning_sd"]
    runs = load_runs(data, "J2")
    rows, res = [], {}
    for amp in design.EPS:
        mine = sorted([r for r in runs if r["amp"] == amp and r["exit"] == 0], key=lambda r: r["M"])
        if len(mine) < 2:
            continue
        nx, ny, lh = mine[0]["nx"], mine[0]["ny"], mine[0]["lh"]
        adm = rule.admissible(ny, design.window(nx, ny, lh, amp), range(1, 1025))
        tabs = {r["M"]: an.read_table(os.path.join(data, "J2", "reduced", r["id"] + ".csv")) for r in mine}
        res[amp] = {"grid": "%dx%dx%d" % (nx, ny, ny), "M": [r["M"] for r in mine], "estimators": {},
                    "consumer_crossing": {M: crossing_fraction(t, "consumer", "flux") for M, t in tabs.items()}}
        for est in an.EST:
            for basis in (("pos", "disp") if est != "alpha_LP" else ("pos",)):
                comps = {M: an.component_rows(t, est, basis) for M, t in tabs.items()}
                V = {"pos": sd[AMP_KEY[amp]][est + "_pos"]["sd"] ** 2, "disp": sd[AMP_KEY[amp]][est + "_disp"]["sd"] ** 2}
                Vb = V[basis]
                cs = [comps[M]["component"] for M in sorted(comps)]
                proj, fit = an.project_M(sorted(comps), cs, Vb, adm)
                rec = {"basis": basis, "planning_sd": math.sqrt(Vb), "exponent": fit["exponent"], "exponent_se": fit["se"],
                       "k_fixed": fit["k_fixed"], "by_M": {}, "M_0.25_fixed": proj[0.25]["M_fixed"], "M_0.25_fit": proj[0.25]["M_fit"],
                       "M_0.10_fixed": proj[0.10]["M_fixed"], "M_0.10_fit": proj[0.10]["M_fit"]}
                for M in sorted(comps):
                    c = comps[M]
                    rec["by_M"][M] = {"n": c["n"], "component": c["component"], "own_variance": c["variance"], "own_fraction": c["fraction"],
                                      "own_flag": c["flag"], "share_planning": c["component"] / Vb}
                    rows.append({"amp": amp, "estimator": est, "basis": basis, "M": M, "n": c["n"], "component": c["component"],
                                 "own_variance": c["variance"], "own_fraction": c["fraction"], "own_flag": int(c["flag"]),
                                 "planning_sd": math.sqrt(Vb), "share_planning": c["component"] / Vb})
                res[amp]["estimators"]["%s/%s" % (est, basis)] = rec
        # M_plan: the 0.25 conclusion with the pos denominator, the largest over the three estimators; same with the disp denominator
        for tag, bases in (("pos", ("pos",)), ("disp", ("disp", "pos"))):
            cand = [v for k, v in res[amp]["estimators"].items() if v["basis"] in bases and (tag == "pos" or v["basis"] == "disp" or k.startswith("alpha_LP"))]
            for thr in ("0.25", "0.10"):
                vals = [v["M_%s_fixed" % thr] for v in cand]
                res[amp]["M_%s_%s" % (thr, tag)] = None if any(x is None for x in vals) else max(vals)
        res[amp]["admissible_M_up_to_1024"] = adm[:40]
    cols = ["amp", "estimator", "basis", "M", "n", "component", "own_variance", "own_fraction", "own_flag", "planning_sd", "share_planning"]
    write_csv(os.path.join(out, "j2_components.csv"), rows, cols)
    with open(os.path.join(out, "j2_particles.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


# ----------------------------------------------------------------------------------------------------------------------
def crossing_fraction(rows, code="par2", weights=None):
    """Plane-crossing completeness of a reduced table: min over realizations of n_last / n_traced (reached the last plane) and of
    n_first / n_traced, for the plane-family ``perplane`` position-basis rows of ``code`` (``weights``: filter, e.g. 'flux' for the
    consumer; the counts do not depend on it)."""
    rs = [r for r in rows if r["code"] == code and r["family"] == "plane" and r["basis"] == "pos" and r["set"] == "perplane"
          and (weights is None or r["weights"] == weights)]
    if not rs:
        raise ValueError("no plane rows of code %r" % code)
    return {"n_realizations": len(rs),
            "min_last_over_injected": min(float(r["n_last"]) / float(r["n_traced"]) for r in rs),
            "min_first_over_injected": min(float(r["n_first"]) / float(r["n_traced"]) for r in rs)}


def j3(data, out, par2_job="J3", name="j3_pairing.json"):
    """Gate 5 pairing statistics; ``par2_job`` = 'J3B' pairs the J3 consumer tables with the longer-horizon Par2 runs."""
    runs = load_runs(data, "J3")
    pruns = load_runs(data, par2_job)
    res = {}
    for amp in design.EPS:
        c = [r for r in runs if r["amp"] == amp and r["code"] == "consumer" and r["exit"] == 0]
        p = [r for r in pruns if r["amp"] == amp and r["code"] == "par2" and r["exit"] == 0]
        if not (c and p):
            continue
        ct = an.read_table(os.path.join(data, "J3", "reduced", c[0]["id"] + ".csv"))
        pt = an.read_table(os.path.join(data, par2_job, "reduced", p[0]["id"] + ".csv"))
        try:
            res[amp] = an.pairing(ct, pt)
        except ValueError as e:
            res[amp] = {"error": str(e)}
            continue
        res[amp]["grid"] = "%dx%dx%d" % (c[0]["nx"], c[0]["ny"], c[0]["nz"])
        res[amp]["M_consumer"] = c[0]["M"]
        res[amp]["par2_horizon_factor"] = p[0].get("par2_horizon_factor", 1.2)
        res[amp]["par2_n_steps"] = p[0]["par2_n_steps"]
        res[amp]["par2_crossing"] = crossing_fraction(pt)
        res[amp]["consumer_crossing"] = crossing_fraction(ct, "consumer", "unif")
    with open(os.path.join(out, name), "w") as f:
        json.dump(res, f, indent=1)
    return res


# ----------------------------------------------------------------------------------------------------------------------
BUDGET_GPU_H = 150.0     # decision G10


def scenario_totals(proj_rows, prow, anchor, rms, max_res, reserve=0.05):
    """Totals (GPU-h, reserve included) of one projection under the SF-39-style sensitivities.  ``rms`` / ``max_res``: {code: refit rms /
    largest |residual|}.  ``anchor``: {(code, cells, nx): mean observed / refit ratio of the pilot points of that geometry}: an arm on a
    geometry that was measured is multiplied by it (the refit leaves 10-27 % residuals on some large grids), others keep the model value.
    Sensitivities, not the STOP rule's value: total_minus_1rms / total_plus_1rms (each code's refit rms), total_plus_max_residual,
    total_anchored and total_anchored_plus_1rms (anchored, then each code's rms applied on top)."""
    g = 1.0 + reserve

    def tot(f):
        return g * sum(r["gpu_h"] * f(r["code"]) for r in proj_rows)

    anch = [r["gpu_h"] * anchor.get((w["arm"].code, w["arm"].cells, w["arm"].nx), 1.0) for r, w in zip(proj_rows, prow)]
    return {"total_minus_1rms": tot(lambda c: 1.0 - rms[c]), "total_plus_1rms": tot(lambda c: 1.0 + rms[c]),
            "total_plus_max_residual": tot(lambda c: 1.0 + max_res[c]), "total_anchored": g * sum(anch),
            "total_anchored_plus_1rms": g * sum(a * (1.0 + rms[r["code"]]) for a, r in zip(anch, proj_rows))}


def stop_rule(base, all_proj, budget):
    """Owner STOP rule (decision G10): the projected total at M_plan, or its +1 rms value, above ``budget`` => STOP_BUDGET.  The other
    scenarios and the anchored / +max-residual values are reported beside it as sensitivities and do not enter the rule."""
    checked = {"total_M_plan": base["total_gpu_h"], "total_M_plan_plus_1rms": base["total_plus_1rms"]}
    stop = any(v > budget for v in checked.values())
    return {"rule": "STOP_BUDGET if the projected total at M_plan or its +1 rms value exceeds the budget", "budget_gpu_h": budget,
            "checked": checked, "stop_budget": bool(stop), "verdict": "STOP_BUDGET" if stop else "WITHIN_BUDGET",
            "margin_gpu_h": {k: budget - v for k, v in checked.items()},
            "sensitivities_not_the_rule": {name: {k: p[k] for k in ("total_gpu_h", "total_plus_1rms", "total_plus_max_residual",
                                                                     "total_anchored", "total_anchored_plus_1rms")}
                                           for name, p in all_proj.items()}}


def residual_diagnostics(data, model, sdr_ref):
    """Evidence for the sign of the refit residuals of the consumer pilot points (hypothesis checks, not a model): per point the mean PCG
    iterations of the run against SF-39's production median at the amplitude, the realization count, and the first realization's time
    against the mean of the others; and the correlation of the refit residual with the iteration ratio."""
    runs = {"%s/%s" % (r["job"], r["id"]): r for job in ("J1", "J2", "J3") for r in load_runs(data, job) if r["code"] == "consumer" and r["exit"] == 0}
    rel = {x["label"]: x["rel"] for x in model.residuals() if x["code"] == "lagrangian"}
    rows = []
    for key, r in sorted(runs.items()):
        recs = [x for x in r["realizations"] if x.get("iterations") is not None]
        it = sum(x["iterations"] for x in recs) / len(recs)
        tot = [x["seconds"]["total"] for x in recs]
        med = sdr_ref[AMP_KEY[r["amp"]]]["production"]["median"]
        rows.append({"id": r["id"], "job": r["job"], "amp": r["amp"], "nr": r["nr"], "ny": r["ny"], "mg_levels": recs[0].get("mg_levels"), "rel_refit": rel[key], "mean_pcg_iterations": it,
                     "sf39_median_iterations": med, "iteration_ratio": it / med,
                     "first_over_rest_time": tot[0] / (sum(tot[1:]) / len(tot[1:])) if len(tot) > 1 else None})
    x, y = np.array([r["iteration_ratio"] for r in rows]), np.array([r["rel_refit"] for r in rows])
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    one = [r["rel_refit"] for r in rows if r["nr"] == 1]
    many = [r["rel_refit"] for r in rows if r["nr"] > 1]
    fr = [r["first_over_rest_time"] for r in rows if r["first_over_rest_time"] is not None]
    return {"points": rows, "n_negative": int((y < 0).sum()), "n": len(rows),
            "pearson_iteration_ratio_vs_residual": float(np.corrcoef(x, y)[0, 1]),
            "spearman_iteration_ratio_vs_residual": float(np.corrcoef(rx, ry)[0, 1]),
            "mean_residual_single_realization_runs": float(np.mean(one)) if one else None, "n_single": len(one),
            "mean_residual_multi_realization_runs": float(np.mean(many)) if many else None, "n_multi": len(many),
            "median_first_over_rest_time": float(np.median(fr)) if fr else None,
            "note": "residual = (refit - observed) / observed: negative = the refit is below the observation"}


def par2_horizon_cost_ratio(data):
    """J3B / J3 (same seeds, grids and amplitudes) of the Par2 per-realization time, of its transport stage and of the step count."""
    j3 = {r["id"]: r for r in load_runs(data, "J3") if r["code"] == "par2" and r["exit"] == 0}
    out = {}
    for r in load_runs(data, "J3B"):
        b = j3.get(r["id"][:-3])
        if b is None or r["exit"] != 0:
            continue
        out[r["id"]] = {"realization_time_ratio": costmod.realization_seconds(r) / costmod.realization_seconds(b),
                        "transport_time_ratio": sum(x["transport_s"] for x in r["par2_stage_times"]) / sum(x["transport_s"] for x in b["par2_stage_times"]),
                        "step_ratio": r["par2_n_steps"] / b["par2_n_steps"], "wall_ratio": r["wall_s"] / b["wall_s"]}
    return out


def cost_report(data, out, repo, j2_json, j3_json, hostref_json):
    """J4: refit with the pilot walls, residuals of the new points under the SF-39-only model, SF-43 projections (M_plan, SF-39 density,
    share 0.10, variant B) with the +-1 rms sensitivity."""
    old, _ = cm.fit_sf39(repo)
    pts = costmod.points_from_runs(data)
    new = old.with_points(pts)
    fit = {"sf39_only": old.summary(), "with_pilots": new.summary(), "n_new_points": len(pts)}
    # residuals of the new points under the SF-39-only model (Par2 pilot points against SF-39's own 'par2' fit, cg_rtol 1e-6)
    # on two bases: 'realization' (the code's own per-realization time, used for the refit) and 'wall' (process wall, start-up included)
    res = []
    for basis in ("realization", "wall"):
        for p in costmod.points_from_runs(data, basis=basis):
            a = p.arm
            arm = a if a.family == "lagrangian" else cm.Arm("par2", a.cells, a.sigma2, a.particles, a.nx, 0.0, 0, a.label)
            pred = old.predict(arm)
            res.append({"basis": basis, "label": a.label, "code": a.code, "n_real": p.n_real, "observed_s": p.gpu_s_per_real,
                        "sf39_model_s": pred, "rel_error_sf39_model": (pred - p.gpu_s_per_real) / p.gpu_s_per_real,
                        "in_sf39_range": old.in_range(arm)})
    labels = {p.arm.label for p in pts}
    resid_new = [{"label": r["label"], "code": r["code"], "rel_refit": r["rel"], "observed_s": r["observed"], "refit_s": r["predicted"]}
                 for r in new.residuals() if r["label"] in labels]
    with open(j2_json) as f:
        j2r = json.load(f)
    with open(j3_json) as f:
        j3r = json.load(f)
    with open(hostref_json) as f:
        hostref = json.load(f)
    sdr = hostref["planning_sd"]
    amps = list(costmod.AMPS)
    M_plan = {a: (j2r[a]["M_0.25_pos"] or 1024) for a in amps}
    M_010 = {a: (j2r[a]["M_0.10_pos"] or 1024) for a in amps}
    M_sf39 = {a: int(round(float(design.target_M(a)))) for a in amps}
    g5 = {a: j3r[a]["plan_lower"]["n"] for a in amps}
    j3b_path = os.path.join(os.path.dirname(j3_json), "j3b_pairing.json")      # horizon-corrected Par2 (amplitudes 1, 2.25, 4)
    g5b = dict(g5)
    if os.path.exists(j3b_path):
        with open(j3b_path) as f:
            for a, v in json.load(f).items():
                g5b[a] = v["plan_lower"]["n"]
    g5b = {a: max(30, n) for a, n in g5b.items()}
    ratio = {a: sdr[AMP_KEY[a]]["alpha_TP_disp"]["sd"] / sdr[AMP_KEY[a]]["alpha_TP_pos"]["sd"] for a in amps}

    def scale(fam, amp, n):
        return an.scale_n(n, 1.0, ratio[amp]) if fam in ("production", "E_W") else n

    n_old = len(old.points)
    ratios = {}
    for pt, rs in zip(new.points[n_old:], new.residuals()[n_old:]):
        ratios.setdefault((pt.arm.code, pt.arm.cells, pt.arm.nx), []).append(1.0 / (1.0 + rs["rel"]))
    anchor = {k: sum(v) / len(v) for k, v in ratios.items()}
    summ = fit["with_pilots"]
    rms = {c: v["rms_rel"] for c, v in summ.items()}
    max_res = {c: v["max_abs_rel"] for c, v in summ.items()}
    out_proj = {}
    for name, M, sc, gn, tag in (("M_plan", M_plan, None, g5, costmod.PAR2_H36_TAG), ("sf39_density", M_sf39, None, g5, costmod.PAR2_H36_TAG),
                                 ("M_plan_share010", M_010, None, g5, costmod.PAR2_H36_TAG),
                                 ("variant_B_M_plan", M_plan, scale, g5, costmod.PAR2_H36_TAG),
                                 ("M_plan_gate5_J3B_floor30", M_plan, None, g5b, costmod.PAR2_H36_TAG),
                                 ("M_common_128_128_86_68", dict(M_plan, **{"0.25": 128, "1": 128}), None, g5, costmod.PAR2_H36_TAG),
                                 ("M_plan_par2_horizon1.2", M_plan, None, g5, costmod.PAR2_TAG)):
        prow = costmod.plan(M, gn, scale=sc, par2_tag=tag)
        proj = costmod.project(new, prow)
        proj["M"] = M
        proj["lagrangian_subtotal_gpu_h"] = sum(r["gpu_h"] for r in proj["rows"] if r["code"] == "lagrangian")
        proj["par2_subtotal_gpu_h"] = sum(r["gpu_h"] for r in proj["rows"] if r["code"] != "lagrangian")
        proj.update(scenario_totals(proj["rows"], prow, anchor, rms, max_res))
        out_proj[name] = proj
    summary = {"fit": fit, "new_points_under_sf39_model": res, "new_points_residuals_refit": resid_new, "projections": out_proj,
               "M_plan": M_plan, "M_share010": M_010, "M_sf39": M_sf39, "gate5_n_lower_bound_corr": g5, "gate5_n_J3B_floor30": g5b,
               "anchor_factors": {"%s|%d|%d" % k: v for k, v in anchor.items()}, "variant_B_sd_ratio_disp_over_pos": ratio,
               "residual_diagnostics": residual_diagnostics(data, new, hostref["pcg"]),
               "par2_horizon_cost_ratio": par2_horizon_cost_ratio(data), "budget_gpu_h": BUDGET_GPU_H}
    summary["stop_rule"] = stop_rule(out_proj["M_plan"], out_proj, BUDGET_GPU_H)
    summary["stop_budget"] = summary["stop_rule"]["stop_budget"]
    with open(os.path.join(out, "cost_projection.json"), "w") as f:
        json.dump(summary, f, indent=1)
    rows = [dict(r, variant=name) for name, pr in out_proj.items() for r in pr["rows"]]
    write_csv(os.path.join(out, "cost_projection_rows.csv"), rows,
              ["variant", "family", "amp", "label", "code", "n", "gpu_s_per_real", "gpu_h", "gpu_h_per_100", "in_range"])
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pilots.report")
    ap.add_argument("what", choices=("j1", "j1b", "j2", "j3", "j3b", "cost"))
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", help="repository root (cost)")
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    if a.what == "cost":
        r = cost_report(a.data, a.out, a.repo, os.path.join(a.out, "j2_particles.json"), os.path.join(a.out, "j3_pairing.json"),
                        os.path.join(a.data, "hostref.json"))
        print(json.dumps({k: v["total_gpu_h"] for k, v in r["projections"].items()}, indent=1), "stop_budget=%s" % r["stop_budget"])
        return 0
    if a.what == "j1b":
        r = j1(a.data, out=a.out, job="J1B", name="j1b_grid")
    elif a.what == "j3b":
        r = j3(a.data, a.out, "J3B", "j3b_pairing.json")
    else:
        r = {"j1": j1, "j2": j2, "j3": j3}[a.what](a.data, a.out)
    print(json.dumps(r, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
