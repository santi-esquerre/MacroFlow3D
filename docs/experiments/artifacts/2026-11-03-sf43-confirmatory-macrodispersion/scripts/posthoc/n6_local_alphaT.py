"""post hoc (2026-11-04) N6: local ``alpha_T`` against distance on ``s1_dist``, ``s225_dist``, ``s4_dist``.

Estimand as the primary one: consumer, flux weights, displacement basis, per-plane moments; sigma_T^2(d) = (var_x2 + var_x3)/2 of the
particles that crossed plane d (per realization, planes table ``*.planes.csv``, all 170 planes d = 25..194 are in the ``d25-94`` file),
divided by ell^2 -> lambda^2; ``alpha_TP`` over a window = 1/2 x OLS slope of sigma_T^2 on d (lambda), per realization, then averaged.
Check: the window [25, 94] reproduces the frozen table's ``alpha_TP`` per seed.

Outputs per arm: ensemble sigma_T^2(d) (mean, SE over realizations) per plane (CSV); local alpha_TP in 9 consecutive windows of 17 lambda
[25,42] ... [161,178] (both end planes included, as the frozen main window [25, 94]) and its ratio to the main-window value, with 90 %
realization-bootstrap percentile intervals (B 2000, seed 20261104); the linear trend of the 9 local values against the window centres, as
% of the main-window value per 100 lambda (realization bootstrap, the ratio re-formed per resample); the same quantity from the global
quadratic fit of sigma_T^2 over d >= 25 (the preview's definition); the descriptive saturation fit sigma_T^2(d) = s0^2 + s_psi^2 (1 - exp(-d/zeta))
of the ensemble-mean curve over all planes (variable projection: for each zeta on a log grid [1, 1e5] lambda, non-negative least squares for
(s0^2, s_psi^2); unweighted, because the plane values are strongly correlated), with the zeta of 2000 realization-bootstrap refits: the
one-sided 95 % LOWER bound is the 5 % bootstrap quantile (percentile bootstrap; zeta at the grid end 1e5 means 'linear within the data').
No statement about asymptotic regimes.
"""
import csv
import os
import sys

import numpy as np
from scipy.optimize import nnls

from common import LABEL, dump, load, ci, outdir
from sf43ana.loader import TableCache
from sf43ana.positive_control import planes_path, read_planes

B, SEED = 2000, 20261104
WIN = [(25 + 17 * i, 25 + 17 * (i + 1)) for i in range(9)]  # 25-42 ... 161-178
ZGRID = np.logspace(0, 5, 501)


def planes_sigma(pr, arms, ledger, scratch, aid):
    arm = arms[aid]
    tables = TableCache(os.path.join(scratch, "camp", "reduced"), pr, ledger)
    cfg = arm.cfg
    ell = float(next(iter(tables.table(cfg["source"], arm.segments[0], "main").values()))["ell"])
    cache, rows, d0 = {}, [], None
    for i, s in enumerate(arm.seeds):
        seg = arm.segments[i]
        if seg not in cache:
            cache[seg] = read_planes(planes_path(tables, cfg["source"], seg, "main"), cfg["code"], "flux", "disp", cfg["hmax_div"])
        rec = cache[seg][int(s) - ledger.seg_seed0[(cfg["source"], seg)]]
        d0 = rec[0] if d0 is None else d0
        assert np.allclose(rec[0], d0)
        rows.append(rec[3] / ell ** 2)
    return d0, np.array(rows), ell


def alpha_local(d, S, lo, hi):
    k = (d >= lo - 1e-6) & (d <= hi + 1e-6)
    x = d[k] - d[k].mean()
    return 0.5 * (S[..., k] @ x) / (x @ x)


def fit_zeta(d, y):
    """-> (zeta, s0sq, spsi_sq, ssr) minimising the SSR over the zeta grid with non-negative linear coefficients.  y: (..., planes)."""
    best = None
    for z in ZGRID:
        A = np.stack([np.ones_like(d), 1.0 - np.exp(-d / z)], axis=1)
        coef, rn = nnls(A, y)
        if best is None or rn < best[3]:
            best = (z, coef[0], coef[1], rn)
    return best


def analyse(d, S, ell, aid, ref_alpha):
    n = S.shape[0]
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 6, sum(map(ord, aid))]))
    idx = rng.integers(0, n, size=(B, n))
    main = alpha_local(d, S, 25, 94)
    out = {"arm": aid, "N": int(n), "ell": ell, "reproduction_main_vs_frozen_table_max_rel": float(np.max(np.abs(main - ref_alpha) / np.abs(ref_alpha))),
           "main_alpha_TP_25_94": float(main.mean()), "windows": [], "label": LABEL}
    loc = np.stack([alpha_local(d, S, lo, hi) for lo, hi in WIN], axis=1)  # (n, 9)
    mean_loc = loc.mean(0)
    Mb = main[idx].mean(1)  # (B,)
    Lb = loc[idx].mean(1)  # (B, 9)
    for j, (lo, hi) in enumerate(WIN):
        r = Lb[:, j] / Mb
        a, b = ci(r)
        out["windows"].append({"window": [lo, hi], "centre": 0.5 * (lo + hi), "alpha_TP": float(mean_loc[j]), "alpha_TP_ci90": [float(x) for x in ci(Lb[:, j])],
                               "ratio_to_main": float(mean_loc[j] / main.mean()), "ratio_ci90": [float(a), float(b)]})
    cen = np.array([0.5 * (lo + hi) for lo, hi in WIN])
    out["local_to_main_range"] = [float(min(w["ratio_to_main"] for w in out["windows"])), float(max(w["ratio_to_main"] for w in out["windows"]))]

    def trend(L, M):  # % of main per 100 lambda
        xc = cen - cen.mean()
        return 100.0 * (L @ xc) / (xc @ xc) / M * 100.0

    t_hat = float(trend(mean_loc, main.mean()))
    t_b = trend(Lb, Mb)
    out["trend_windows_percent_per_100"] = {"estimate": t_hat, "ci90": [float(x) for x in ci(t_b)]}
    # global quadratic fit of the per-realization curves over d >= 25: alpha_T'(d) = c (per lambda), trend = 100 c / main
    k = d >= 25 - 1e-6
    V = np.vander(d[k], 3)
    coef = np.linalg.lstsq(V, S[:, k].T, rcond=None)[0]  # (3, n)
    c2 = coef[0]
    out["trend_quadratic_percent_per_100"] = {"estimate": float(100 * 100 * c2.mean() / main.mean()),
                                             "ci90": [float(x) for x in ci(100 * 100 * c2[idx].mean(1) / Mb)]}
    # ensemble curve
    m, se = S.mean(0), S.std(0, ddof=1) / np.sqrt(n)
    out["curve"] = {"d": d, "sigma_T2_mean": m, "sigma_T2_se": se}
    z, s0, sp, ssr = fit_zeta(d, m)
    zb = np.empty(B)
    atb = 0
    for b in range(B):
        zz = fit_zeta(d, S[idx[b]].mean(0))
        zb[b] = zz[0]
    out["saturation_fit"] = {"model": "s0^2 + s_psi^2 (1 - exp(-d/zeta))", "planes": [float(d[0]), float(d[-1])], "n_planes": int(d.size), "zeta": float(z),
                             "s0_sq": float(s0), "s_psi_sq": float(sp), "rms_residual": float(np.sqrt(ssr ** 2 / d.size)),
                             "zeta_at_grid_end": bool(z >= ZGRID[-1] * 0.999),
                             "zeta_lower_bound_95_one_sided_bootstrap": float(np.quantile(zb, 0.05)), "zeta_boot_quantiles_5_25_50_75_95": [float(x) for x in np.quantile(zb, [0.05, 0.25, 0.5, 0.75, 0.95])],
                             "share_boot_fits_at_grid_end": float((zb >= ZGRID[-1] * 0.999).mean()),
                             "zeta_ci90_percentile": [float(x) for x in ci(zb)]}
    return out


def main():
    scratch = sys.argv[1]
    pr, arms, ledger = load(scratch)
    od = outdir("n6")
    summary = {}
    for aid in ("s1_dist", "s225_dist", "s4_dist"):
        d, S, ell = planes_sigma(pr, arms, ledger, scratch, aid)
        res = analyse(d, S, ell, aid, arms[aid].get("main", "disp", "alpha_TP"))
        dump(os.path.join(od, "n6_%s.json" % aid), res)
        with open(os.path.join(od, "n6_%s_curve.csv" % aid), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["d_lambda", "sigma_T2_mean_lambda2", "sigma_T2_se"])
            w.writerows(zip(d, res["curve"]["sigma_T2_mean"], res["curve"]["sigma_T2_se"]))
        summary[aid] = res
        print(aid, "N", res["N"], "repro max rel %.2e" % res["reproduction_main_vs_frozen_table_max_rel"], "main %.4e" % res["main_alpha_TP_25_94"])
        print("   local/main", ["%.3f" % w["ratio_to_main"] for w in res["windows"]], "range", res["local_to_main_range"])
        print("   trend windows %.2f%% %s per 100 lambda;  quadratic %.2f%% %s" % (res["trend_windows_percent_per_100"]["estimate"], res["trend_windows_percent_per_100"]["ci90"],
              res["trend_quadratic_percent_per_100"]["estimate"], res["trend_quadratic_percent_per_100"]["ci90"]))
        sf = res["saturation_fit"]
        print("   zeta %.1f lower95 %.1f at_end %s share_end %.3f s0^2 %.4g s_psi^2 %.4g" % (sf["zeta"], sf["zeta_lower_bound_95_one_sided_bootstrap"], sf["zeta_at_grid_end"], sf["share_boot_fits_at_grid_end"], sf["s0_sq"], sf["s_psi_sq"]))


if __name__ == "__main__":
    main()
