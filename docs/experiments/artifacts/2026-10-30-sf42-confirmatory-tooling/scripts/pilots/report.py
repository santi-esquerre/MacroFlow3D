"""Local report generator of the SF-42 R4 pilots from the pulled host data (design inputs; no interpretation of any alpha).

    python3 -m pilots.report j1|j2|j3 --data <dir with J1, J2, J3, hostref.json> --out <R/pilots dir>

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
import statistics
import sys

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
def j1(data, out):
    with open(os.path.join(data, "hostref.json")) as f:
        ref = json.load(f)["pcg"]
    rows = []
    for r in load_runs(data, "J1"):
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
    write_csv(os.path.join(out, "j1_grid.csv"), rows, cols)
    with open(os.path.join(out, "j1_grid.json"), "w") as f:
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
        res[amp] = {"grid": "%dx%dx%d" % (nx, ny, ny), "M": [r["M"] for r in mine], "estimators": {}}
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
def j3(data, out):
    runs = load_runs(data, "J3")
    res = {}
    for amp in design.EPS:
        c = [r for r in runs if r["amp"] == amp and r["code"] == "consumer" and r["exit"] == 0]
        p = [r for r in runs if r["amp"] == amp and r["code"] == "par2" and r["exit"] == 0]
        if not (c and p):
            continue
        ct = an.read_table(os.path.join(data, "J3", "reduced", c[0]["id"] + ".csv"))
        pt = an.read_table(os.path.join(data, "J3", "reduced", p[0]["id"] + ".csv"))
        try:
            res[amp] = an.pairing(ct, pt)
        except ValueError as e:
            res[amp] = {"error": str(e)}
            continue
        res[amp]["grid"] = "%dx%dx%d" % (c[0]["nx"], c[0]["ny"], c[0]["nz"])
        res[amp]["M_consumer"] = c[0]["M"]
    with open(os.path.join(out, "j3_pairing.json"), "w") as f:
        json.dump(res, f, indent=1)
    return res


# ----------------------------------------------------------------------------------------------------------------------
def cost_report(data, out, repo, j2_json, j3_json, hostref_json):
    """J4: refit with the pilot walls, residuals of the new points under the SF-39-only model, SF-43 projections (M_plan, SF-39 density,
    share 0.10, variant B) with the +-1 rms sensitivity."""
    old, _ = cm.fit_sf39(repo)
    pts = costmod.points_from_runs(data)
    new = old.with_points(pts)
    fit = {"sf39_only": old.summary(), "with_pilots": new.summary(), "n_new_points": len(pts)}
    # residuals of the new points under the SF-39-only model (Par2 pilot points against SF-39's own 'par2' fit, cg_rtol 1e-6)
    res = []
    for p in pts:
        a = p.arm
        arm = a if a.family == "lagrangian" else cm.Arm("par2", a.cells, a.sigma2, a.particles, a.nx, 0.0, 0, a.label)
        pred = old.predict(arm)
        res.append({"label": a.label, "code": a.code, "n_real": p.n_real, "observed_s": p.gpu_s_per_real, "sf39_model_s": pred,
                    "rel_error_sf39_model": (pred - p.gpu_s_per_real) / p.gpu_s_per_real, "in_sf39_range": old.in_range(arm)})
    labels = {p.arm.label for p in pts}
    resid_new = [{"label": r["label"], "code": r["code"], "rel_refit": r["rel"], "observed_s": r["observed"], "refit_s": r["predicted"]}
                 for r in new.residuals() if r["label"] in labels]
    with open(j2_json) as f:
        j2r = json.load(f)
    with open(j3_json) as f:
        j3r = json.load(f)
    with open(hostref_json) as f:
        sdr = json.load(f)["planning_sd"]
    amps = list(costmod.AMPS)
    M_plan = {a: (j2r[a]["M_0.25_pos"] or 1024) for a in amps}
    M_010 = {a: (j2r[a]["M_0.10_pos"] or 1024) for a in amps}
    M_sf39 = {a: int(round(float(design.target_M(a)))) for a in amps}
    g5 = {a: j3r[a]["plan_lower"]["n"] for a in amps}
    ratio = {a: sdr[AMP_KEY[a]]["alpha_TP_disp"]["sd"] / sdr[AMP_KEY[a]]["alpha_TP_pos"]["sd"] for a in amps}

    def scale(fam, amp, n):
        return an.scale_n(n, 1.0, ratio[amp]) if fam in ("production", "E_W") else n

    out_proj = {}
    rms_l = fit["with_pilots"]["lagrangian"]["rms_rel"]
    rms_p = fit["with_pilots"][costmod.PAR2_TAG]["rms_rel"]
    for name, M, sc in (("M_plan", M_plan, None), ("sf39_density", M_sf39, None), ("M_plan_share010", M_010, None),
                        ("variant_B_M_plan", M_plan, scale)):
        proj = costmod.project(new, costmod.plan(M, g5, scale=sc))
        lag_h = sum(r["gpu_h"] for r in proj["rows"] if r["code"] == "lagrangian")
        par_h = sum(r["gpu_h"] for r in proj["rows"] if r["code"] != "lagrangian")
        proj["M"] = M
        proj["lagrangian_subtotal_gpu_h"], proj["par2_subtotal_gpu_h"] = lag_h, par_h
        proj["total_plus_1rms"] = (lag_h * (1 + rms_l) + par_h * (1 + rms_p)) * 1.05
        proj["total_minus_1rms"] = (lag_h * (1 - rms_l) + par_h * (1 - rms_p)) * 1.05
        out_proj[name] = proj
    summary = {"fit": fit, "new_points_under_sf39_model": res, "new_points_residuals_refit": resid_new, "projections": out_proj,
               "M_plan": M_plan, "M_share010": M_010, "M_sf39": M_sf39, "gate5_n_lower_bound_corr": g5, "variant_B_sd_ratio_disp_over_pos": ratio,
               "budget_gpu_h": 150.0}
    summary["stop_budget"] = bool(out_proj["M_plan"]["total_gpu_h"] > 150.0 or out_proj["M_plan"]["total_plus_1rms"] > 150.0)
    with open(os.path.join(out, "cost_projection.json"), "w") as f:
        json.dump(summary, f, indent=1)
    rows = [dict(r, variant=name) for name, pr in out_proj.items() for r in pr["rows"]]
    write_csv(os.path.join(out, "cost_projection_rows.csv"), rows,
              ["variant", "family", "amp", "label", "code", "n", "gpu_s_per_real", "gpu_h", "gpu_h_per_100", "in_range"])
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pilots.report")
    ap.add_argument("what", choices=("j1", "j2", "j3", "cost"))
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
    r = {"j1": j1, "j2": j2, "j3": j3}[a.what](a.data, a.out)
    print(json.dumps(r, indent=1, default=str)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
