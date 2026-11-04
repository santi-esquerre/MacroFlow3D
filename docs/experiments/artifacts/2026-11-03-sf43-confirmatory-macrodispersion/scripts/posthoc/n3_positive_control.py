"""post hoc (2026-11-04) N3: the positive control (pc_s004, sigma^2 = 0.04) plane by plane.

Uses the FROZEN code of ``sf43ana.positive_control`` (``read_planes``, ``x22_box``, the box first-order prediction) on the committed planes
tables.  Quantities per weights variant (flux, unif) in the DISPLACEMENT basis (the only basis for which the prediction exists):
``second_moment`` (what ``R`` and X22 are) and ``centred`` (what the estimand ``alpha_TP`` is the slope of).  Per plane d: measured, predicted,
ratio; excess = measured - predicted against d: OLS slope over the window with a 90 % REALIZATION-bootstrap percentile interval (B 2000,
seed 20261104; the resampled realizations' mean profiles are re-fitted), the d at which the OLS line crosses zero, stability under trimming,
the ratio at the first and last plane.  Check: R (flux, second moment) must reproduce the frozen summary (1.145 [1.122, 1.168]).
"""
import json
import os
import sys

import numpy as np

from common import ART, LABEL, ci, dump, load, ols_slope, outdir
from sf43ana.positive_control import planes_path, read_planes, x22_box
from sf43ana.loader import TableCache

B, SEED, LEVEL = 2000, 20261104, 0.90
TRIMS = [(25, 94), (30, 90), (35, 85), (40, 80), (25, 60), (60, 94), (25, 80), (40, 94)]


def profiles(pr, arms, ledger, w):
    arm = arms["pc_s004"]
    pc = pr.raw["positive_control"]
    tables = TableCache(os.path.join(sys.argv[1], "camp", "reduced"), pr, ledger)
    cfg = arm.cfg
    ell = float(next(iter(tables.table(cfg["source"], arm.segments[0], "main").values()))["ell"])
    cache, V2, VC, dgrid = {}, [], [], None
    for i, s in enumerate(arm.seeds):
        seg = arm.segments[i]
        if seg not in cache:
            cache[seg] = read_planes(planes_path(tables, cfg["source"], seg, "main"), cfg["code"], w, "disp", cfg["hmax_div"])
        rec = cache[seg][int(s) - ledger.seg_seed0[(cfg["source"], seg)]]
        dgrid = rec[0] if dgrid is None else dgrid
        assert np.allclose(rec[0], dgrid)
        V2.append(rec[2] / ell ** 2)
        VC.append(rec[3] / ell ** 2)
    pred = float(pc["sigma2"]) * x22_box(dgrid, float(pc["length_lambda"]), float(pc["width_lambda"]), float(pc["kmax"]))
    return dgrid, np.array(V2), np.array(VC), pred, ell


def line_stats(d, M, pred, rows, idx):
    """OLS of the excess mean profile over rows; returns slope, intercept at d=0 (extrapolated), zero-crossing d."""
    x = d[rows]
    ex = M.mean(0)[rows] - pred[rows]
    sl = ols_slope(x, ex)
    ic = ex.mean() - sl * x.mean()
    return sl, ic, (-ic / sl if sl != 0 else float("nan"))


def analyse(d, M, pred, rng):
    n = M.shape[0]
    idx = rng.integers(0, n, size=(B, n))
    Mb = M[idx].mean(1)  # (B, planes)
    m = M.mean(0)
    res = {"n": int(n), "d": d, "measured": m, "predicted": pred, "ratio": m / pred}
    rl, rh = ci(Mb / pred, LEVEL)
    res["ratio_lo"], res["ratio_hi"] = rl, rh
    res["excess"] = m - pred
    trims = {}
    for lo, hi in TRIMS:
        rows = (d >= lo - 1e-6) & (d <= hi + 1e-6)
        sl, ic, z = line_stats(d, M, pred, rows, None)
        bs = np.array([ols_slope(d[rows], Mb[b][rows] - pred[rows]) for b in range(B)])
        bi = np.array([(Mb[b][rows] - pred[rows]).mean() - bs[b] * d[rows].mean() for b in range(B)])
        bz = -bi / bs
        trims["%g-%g" % (lo, hi)] = {"planes": int(rows.sum()), "excess_slope": sl, "excess_slope_ci90": ci(bs, LEVEL), "excess_intercept_d0": ic,
                                     "zero_crossing_d": z, "zero_crossing_ci90_percentile": ci(bz[np.isfinite(bz)], LEVEL),
                                     "share_bootstrap_slopes_positive": float((bs > 0).mean()),
                                     "ratio_first_plane": float(m[rows][0] / pred[rows][0]), "ratio_last_plane": float(m[rows][-1] / pred[rows][-1]),
                                     "ratio_fit_first": float((pred[rows][0] + ic + sl * d[rows][0]) / pred[rows][0]),
                                     "ratio_fit_last": float((pred[rows][-1] + ic + sl * d[rows][-1]) / pred[rows][-1])}
    res["trims"] = trims
    full = (d >= 25 - 1e-6) & (d <= 94 + 1e-6)
    i0, i1 = int(np.argmin(abs(d - 25))), int(np.argmin(abs(d - 94)))
    iA = int(np.argmin(abs(d - 80)))
    res["ratio_first_last"] = {"d_first": d[i0], "d_last": d[i1], "ratio_first": m[i0] / pred[i0], "ratio_last": m[i1] / pred[i1],
                               "ratio_first_ci90": ci(Mb[:, i0] / pred[i0], LEVEL), "ratio_last_ci90": ci(Mb[:, i1] / pred[i1], LEVEL),
                               "last_minus_first_ci90": ci(Mb[:, i1] / pred[i1] - Mb[:, i0] / pred[i0], LEVEL)}
    res["prediction_fall_80_to_94_percent"] = 100.0 * (pred[i1] / pred[iA] - 1.0)
    res["window_mean_ratio_R_like"] = float(m[full].mean() / pred[full].mean())
    res["window_mean_ratio_ci90"] = ci(Mb[:, full].mean(1) / pred[full].mean(), LEVEL)
    sm = ols_slope(d[full], m[full])
    ps = ols_slope(d[full], pred[full])
    res["window_slope"] = {"measured": sm, "predicted": ps, "ratio": sm / ps}
    return res


def main():
    pr, arms, ledger = load(sys.argv[1])
    out = {"label": LABEL, "arm": "pc_s004", "B": B, "seed": SEED, "level": LEVEL, "basis": "displacement (the prediction exists only there)",
           "interval_method": "realization bootstrap (percentile, B 2000) of the mean profiles, re-fitted per resample", "variants": {}}
    for w in ("flux", "unif"):
        d, V2, VC, pred, ell = profiles(pr, arms, ledger, w)
        out["ell"] = ell
        out["N"] = int(V2.shape[0])
        for name, M in (("second_moment", V2), ("centred", VC)):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, 3, 0 if w == "flux" else 1, 0 if name == "second_moment" else 1]))
            out["variants"]["%s/%s" % (w, name)] = analyse(d, M, pred, rng)
    # reproduction check against the frozen summary (R, flux, second moment, window mean)
    with open(os.path.join(ART, "results", "analysis", "summary.json")) as f:
        froz = json.load(f)["positive_control"]["R"]
    v = out["variants"]["flux/second_moment"]
    out["reproduction"] = {"R_frozen": froz["value"], "R_frozen_interval": froz["interval"], "R_here": v["window_mean_ratio_R_like"],
                           "R_here_ci90": v["window_mean_ratio_ci90"], "abs_diff_value": abs(froz["value"] - v["window_mean_ratio_R_like"])}
    dump(os.path.join(outdir("n3"), "n3_positive_control.json"), out)
    r = out["reproduction"]
    print("R frozen %.6f %s  here %.6f %s" % (r["R_frozen"], r["R_frozen_interval"], r["R_here"], r["R_here_ci90"]))
    for k, v in out["variants"].items():
        t = v["trims"]["25-94"]
        print(k, "ratio first/last %.3f/%.3f" % (v["ratio_first_last"]["ratio_first"], v["ratio_first_last"]["ratio_last"]),
              "slope %.3e %s" % (t["excess_slope"], t["excess_slope_ci90"]), "zero-crossing d=%.1f" % t["zero_crossing_d"],
              "fall80-94 %.2f%%" % v["prediction_fall_80_to_94_percent"], "fit ratio %.3f->%.3f" % (t["ratio_fit_first"], t["ratio_fit_last"]))
        for tk, tv in v["trims"].items():
            print("    trim %-6s slope %.3e [%.3e, %.3e] zero %.1f" % (tk, tv["excess_slope"], *tv["excess_slope_ci90"], tv["zero_crossing_d"]))


if __name__ == "__main__":
    main()
