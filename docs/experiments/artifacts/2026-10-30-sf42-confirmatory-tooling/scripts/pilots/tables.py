"""Markdown tables of the committed pilot outputs (``R/pilots``) for the experiment note: python3 -m pilots.tables <R/pilots dir> <table>.

Tables: j1, j2, j2m (M conclusions), j2cross (consumer crossing completeness), j3, cost_fit, cost_new (residuals of the new points under the SF-39-only model), proj (SF-43
projection), proj_rows (GPU-h per arm and per 100 realizations at M_plan), stop (STOP rule), resid (diagnostics of the refit residuals).  No interpretation of any alpha is made or printed.
"""
import csv
import json
import os
import sys


def _json(d, name):
    with open(os.path.join(d, name)) as f:
        return json.load(f)


def _csv(d, name):
    with open(os.path.join(d, name)) as f:
        return list(csv.DictReader(f))


def g(x, n=4):
    return "%.*g" % (n, x) if isinstance(x, float) else str(x)


def j1(d):
    rows = _csv(d, "j1_grid.csv")
    out = ["| run | code | grid | sigma^2 | M | PCG iters | verdict | SF-39 median | ratio | flow stage (s) | wall (s) | peak GPU MiB (own PID) | runner exact MiB | rule |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        out.append("| `%s` | %s | %s | %s | %s | %s | %s | %s | %.2f | %.1f | %.1f | %s | %s | %s |" % (
            r["id"], "consumer" if r["code"] == "consumer" else "Par2", r["grid"], r["amp"], r["M"] if r["code"] == "consumer" else "-",
            r["pcg_iterations"], r["verdict"], r["sf39_median_iterations"], float(r["ratio_to_sf39_median"]), float(r["flow_stage_s"]),
            float(r["wall_s"]), r["peak_gpu_mib"], "%.0f" % float(r["runner_exact_bytes_max_mib"]) if r["runner_exact_bytes_max_mib"] else "-",
            "different grid" if r["different_grid_required"] == "True" else "ok"))
    return "\n".join(out)


def j2(d):
    rows = _csv(d, "j2_components.csv")
    out = ["| sigma^2 | estimator (basis) | M | n | component `mean_r[(a_A-a_B)^2]/4` | own split-half share (6 real.) | planning SD (R3) | share of planning variance |",
           "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["estimator"] == "alpha_TAP":
            continue    # identical to alpha_TP within 4e-4 relative (see the note); kept in j2_components.csv
        out.append("| %s | %s (%s) | %s | %s | %.3e | %.3e | %.3e | %.3e |" % (r["amp"], r["estimator"], r["basis"], r["M"], r["n"], float(r["component"]),
                                                                          float(r["own_fraction"]), float(r["planning_sd"]), float(r["share_planning"])))
    return "\n".join(out)


def j2m(d):
    r = _json(d, "j2_particles.json")
    out = ["| sigma^2 | grid | M tested | estimator (basis) | fitted exponent of M (se) | M at share <= 0.25 (fixed M^-2 law / fit) | M at share <= 0.10 (fixed / fit) |",
           "|---|---|---|---|---|---|---|"]
    for amp, v in r.items():
        for k, e in v["estimators"].items():
            if k.startswith("alpha_TAP"):
                continue
            out.append("| %s | %s | %s | %s | %.2f (%.2f) | %s / %s | %s / %s |" % (amp, v["grid"], ",".join(str(m) for m in v["M"]), k.replace("/", " ("
                       ) + ")", e["exponent"], e["exponent_se"], e["M_0.25_fixed"], e["M_0.25_fit"], e["M_0.10_fixed"], e["M_0.10_fit"]))
    out += ["", "| sigma^2 | M_plan (0.25, pos denominator; max over the three estimators) | same, disp denominator | M at 0.10 (pos) | M at 0.10 (disp) |", "|---|---|---|---|---|"]
    for amp, v in r.items():
        out.append("| %s | %s | %s | %s | %s |" % (amp, v["M_0.25_pos"], v["M_0.25_disp"], v["M_0.10_pos"], v["M_0.10_disp"]))
    return "\n".join(out)


def j3(d):
    out = []
    for f, tag in (("j3_pairing.json", "J3 (SF-39 horizon factor 1.2)"), ("j3b_pairing.json", "J3B (horizon factor 3.6)")):
        r = _json(d, f)
        out += ["**%s**" % tag, "", "| sigma^2 | grid | n pairs | mean L | mean P | CV L | CV P | corr | corr 80 % LB | Par2 steps | min Par2 crossed/injected at the last plane | min consumer reached/seeded | plan n (LB corr) | plan n (corr 0) |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for a, v in r.items():
            out.append("| %s | %s | %d | %.4g | %.4g | %.3f | %.3f | %.3f | %.3f | %d | %.4f | %.4f | %d | %d |" % (
                a, v["grid"], v["n_pairs"], v["mean_L"], v["mean_P"], v["cv_L"], v["cv_P"], v["corr"], v["corr_lower80"], v["par2_n_steps"],
                v["par2_crossing"]["min_last_over_injected"], v["consumer_crossing"]["min_last_over_injected"], v["plan_lower"]["n"],
                v["plan_zero"]["n"]))
        out.append("")
    return "\n".join(out)


def cost_fit(d):
    r = _json(d, "cost_projection.json")
    out = ["| code | fit | points | rms rel | max abs rel | cells range | sigma^2 range | particles range | coefficients |", "|---|---|---|---|---|---|---|---|---|"]
    for tag, key in (("SF-39 only", "sf39_only"), ("with pilots", "with_pilots")):
        for code, v in r["fit"][key].items():
            rg = v["ranges"]
            out.append("| %s | %s | %d | %.3f | %.3f | %.3g - %.3g | %g - %g | %g - %g | %s |" % (
                code, tag, v["n_points"], v["rms_rel"], v["max_abs_rel"], rg["cells"][0], rg["cells"][1], rg["sigma2"][0], rg["sigma2"][1],
                rg["particles"][0], rg["particles"][1], ", ".join("%s=%.3g" % kv for kv in v["coef"].items())))
    return "\n".join(out)


def cost_new(d):
    r = _json(d, "cost_projection.json")
    out = ["| point | code | observed s/real | SF-39-only model | relative error of the model | refit residual | in SF-39 range |", "|---|---|---|---|---|---|---|"]
    refit = {(x["label"], x["code"], round(x["observed_s"], 6)): x["rel_refit"] for x in r["new_points_residuals_refit"]}
    for x in r["new_points_under_sf39_model"]:
        if x["basis"] != "realization":
            continue
        rf = refit.get((x["label"], x["code"], round(x["observed_s"], 6)))
        out.append("| `%s` | %s | %.1f | %.1f | %+.3f | %s | %s |" % (x["label"], x["code"], x["observed_s"], x["sf39_model_s"], x["rel_error_sf39_model"],
                                                                    "%+.3f" % rf if rf is not None else "-", "yes" if x["in_sf39_range"] else "no"))
    return "\n".join(out)


def proj(d):
    r = _json(d, "cost_projection.json")
    fams = ["production", "E_h", "E_W", "hmax", "gate5_consumer", "gate5_par2", "g7_par2", "distance"]
    out = ["| variant | " + " | ".join(fams) + " | reserve 5 % | total GPU-h | -1 rms | +1 rms | +max residual | anchored | anchored +1 rms |", "|---|" + "---|" * (len(fams) + 7)]
    for name, p in r["projections"].items():
        out.append("| %s | %s | %.1f | **%.1f** | %.1f | %.1f | %.1f | %.1f | %.1f |" % (
            name, " | ".join("%.1f" % p["by_family"][f] for f in fams), p["reserve_gpu_h"], p["total_gpu_h"], p["total_minus_1rms"], p["total_plus_1rms"],
            p["total_plus_max_residual"], p["total_anchored"], p["total_anchored_plus_1rms"]))
    return "\n".join(out)


def proj_rows(d):
    r = _json(d, "cost_projection.json")
    out = ["| family | sigma^2 | arm | n | s/realization | GPU-h | GPU-h per 100 | in calibrated range |", "|---|---|---|---|---|---|---|---|"]
    for x in r["projections"]["M_plan"]["rows"]:
        out.append("| %s | %s | %s | %d | %.1f | %.2f | %.3f | %s |" % (x["family"], x["amp"], x["label"].split(" ", 2)[2], x["n"], x["gpu_s_per_real"], x["gpu_h"],
                                                           x["gpu_h_per_100"], "yes" if x["in_range"] == "True" or x["in_range"] is True else "**no**"))
    return "\n".join(out)


def j2cross(d):
    r = _json(d, "j2_particles.json")
    out = ["| sigma^2 | M | realizations | min over realizations of particles reaching the last plane / seeded | same, first plane |", "|---|---|---|---|---|"]
    for amp, v in r.items():
        for M, c in v["consumer_crossing"].items():
            out.append("| %s | %s | %d | %.4f | %.4f |" % (amp, M, c["n_realizations"], c["min_last_over_injected"], c["min_first_over_injected"]))
    return "\n".join(out)


def stop(d):
    r = _json(d, "cost_projection.json")["stop_rule"]
    out = ["budget %.0f GPU-h (G10); verdict **%s**" % (r["budget_gpu_h"], r["verdict"]), "", "| value checked by the rule | GPU-h | margin to the budget |", "|---|---|---|"]
    for k, v in r["checked"].items():
        out.append("| %s | %.1f | %.1f |" % (k, v, r["margin_gpu_h"][k]))
    return "\n".join(out)


def resid(d):
    r = _json(d, "cost_projection.json")["residual_diagnostics"]
    out = ["| point | job | sigma^2 | ny | MG levels | realizations | mean PCG iterations | SF-39 median | ratio | refit residual | first / rest time |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for p in sorted(r["points"], key=lambda p: (p["job"], p["id"])):
        out.append("| `%s` | %s | %s | %d | %s | %d | %.1f | %g | %.2f | %+.3f | %s |" % (
            p["id"], p["job"], p["amp"], p["ny"], p["mg_levels"], p["nr"], p["mean_pcg_iterations"], p["sf39_median_iterations"], p["iteration_ratio"],
            p["rel_refit"], "%.2f" % p["first_over_rest_time"] if p["first_over_rest_time"] is not None else "-"))
    out += ["", "negative residuals %d of %d; Pearson / Spearman of residual with the iteration ratio %.2f / %.2f; mean residual single-realization runs %+.3f (%d), "
            "multi-realization runs %+.3f (%d); median first / rest realization time %.2f" % (
                r["n_negative"], r["n"], r["pearson_iteration_ratio_vs_residual"], r["spearman_iteration_ratio_vs_residual"],
                r["mean_residual_single_realization_runs"], r["n_single"], r["mean_residual_multi_realization_runs"], r["n_multi"], r["median_first_over_rest_time"])]
    return "\n".join(out)


if __name__ == "__main__":
    print({"j1": j1, "j2": j2, "j2m": j2m, "j3": j3, "cost_fit": cost_fit, "cost_new": cost_new, "proj": proj, "proj_rows": proj_rows,
       "j2cross": j2cross, "stop": stop, "resid": resid}[sys.argv[2]](sys.argv[1]))
