"""SF-42 N3: compare the finite-volume Darcy face fluxes with the pseudo-spectral reference.

Usage (from the `scripts/` directory of the artifact root):
  python3 -m spectral.compare run --fv-root FV --out OUT [--sigma2 1 4] [--seeds 420001 420002]
                                  [--ns 16 32 40 64 128 256] [--ref-m 0.25:64,1:64,2.25:96,4:128]
                                  [--gen-n 64] [--tol 1e-12] [--workers W] [--keep-ref-faces]
  python3 -m spectral.compare scan --out OUT --sigma2 4 --seed 420001 --ms 64 128 256 [--ns 16 64 256] [--band B]
  python3 -m spectral.compare summarize --out OUT

FV inputs (written by `apps/sf42_periodic_flow`): FV/<tag>_seed<seed>/N<N>/{Y,U,V,W}.npy + report.json,
<tag> = s<sigma2> with '.' -> 'p'.  For every (sigma2, seed) one reference pair (M, 2M) is solved on the
continuous SF-18 field of a generating grid `gen-n` (a band-limited field whose modes |m_d| <= gen-n/2 - 1 contain
every mode of the FV grids with N <= gen-n and, for smaller N, differ only by modes with variance <= 4e-17, reported
per level); the exact face averages of both references on every FV grid are compared with the FV dumps.  Quantities
per level: relative L2 error of the FV face fluxes (x-faces, all faces), max norm relative to the rms flux,
reference self-convergence (M against 2M on the same faces), observed orders, the generator match (Y dump against
`sf18.generate`) and the applied scales.  `summarize` applies acceptance E of SF-42.

Restart safe: a finished (sigma2, seed) case has OUT/cases/<tag>_seed<seed>.json and is skipped; the solved
reference faces are cached under OUT/ref/ until the case is written.
"""
import argparse
import glob
import json
import os
import shutil
import sys
import time

import numpy as np

from . import darcy, sf18

ELL = 0.25
DEFAULT_SIGMA2 = (0.25, 1.0, 2.25, 4.0)
DEFAULT_SEEDS = (420001, 420002)
DEFAULT_NS = (16, 32, 40, 64, 128, 256)
DEFAULT_REF_M = {0.25: 64, 1.0: 64, 2.25: 96, 4.0: 128}  # the finer reference is 2M (local scans M 32/48/64: see SF-42 N3 report)
DEFAULT_BAND = darcy.DEFAULT_BAND
ORDER_MIN = 1.5
SELFCONV_RATIO = 10.0
REPORT_LH = (10, 16, 32)  # lambda / h whose error is recorded


def tag(sigma2):
    return ("s%g" % sigma2).replace(".", "p")


def eps_of(sigma2):
    return float(np.sqrt(sigma2))


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# --- reference ------------------------------------------------------------------------------------
def solve_reference(sigma2, seed, M, ns, cache_dir, gen_n, tol, maxit, workers, band=DEFAULT_BAND):
    """Solve the pseudo-spectral problem on M^3 and return {N: (U, V, W)} plus the solver summary."""
    sfile = os.path.join(cache_dir, "summary.json")
    names = [os.path.join(cache_dir, "N%d_%s.npy" % (N, c)) for N in ns for c in "UVW"]
    if os.path.exists(sfile) and all(os.path.exists(p) for p in names):
        with open(sfile) as fh:
            summ = json.load(fh)
        if summ.get("band") == band and summ.get("tol") == tol and summ.get("gen_n") == gen_n:
            log("reference M=%d: cached" % M)
            return {N: tuple(np.load(os.path.join(cache_dir, "N%d_%s.npy" % (N, c))) for c in "UVW") for N in ns}, summ
    os.makedirs(cache_dir, exist_ok=True)
    gen = sf18.generate(gen_n, sigma2=1.0, ell=ELL, seed=seed, workers=workers)
    Y = sf18.field_on_grid(gen, M, offset=0.0, workers=workers)
    K = np.exp(eps_of(sigma2) * Y)
    t0 = time.time()
    ref = darcy.Reference(K, tol=tol, maxit=maxit, workers=workers, band=band)
    del K, Y
    log("reference M=%d: PCG %s iterations, %d matvecs, %.0f s, G=%s" %
        (M, [c["iterations"] for c in ref.correctors], ref.matvecs, time.time() - t0, np.array2string(ref.G, precision=12)))
    faces = {}
    for N in ns:
        faces[N] = darcy.face_fluxes(ref.q, N)
        for c, a in zip("UVW", faces[N]):
            np.save(os.path.join(cache_dir, "N%d_%s.npy" % (N, c)), a)
    summ = ref.summary()
    summ.update({"gen_n": gen_n, "band": band, "applied_scale_gen": gen.applied_scale, "raw_variance_gen": gen.raw_variance})
    with open(sfile, "w") as fh:
        json.dump(summ, fh, indent=1)
    return faces, summ


# --- one case --------------------------------------------------------------------------------------
def load_fv(fv_dir):
    with open(os.path.join(fv_dir, "report.json")) as fh:
        rep = json.load(fh)
    arrs = {c: np.load(os.path.join(fv_dir, "%s.npy" % c)) for c in "YUVW"}
    N = rep["n"]
    expect = {"Y": (N, N, N), "U": (N, N, N + 1), "V": (N, N + 1, N), "W": (N + 1, N, N)}
    for c, shp in expect.items():
        if arrs[c].shape != shp:
            raise ValueError("%s: %s has shape %s, expected %s" % (fv_dir, c, arrs[c].shape, shp))
    return rep, arrs


def run_case(sigma2, seed, ns, M, fv_root, out, gen_n, tol, maxit, workers, keep_ref_faces, check_y=True, band=DEFAULT_BAND):
    cfile = os.path.join(out, "cases", "%s_seed%d.json" % (tag(sigma2), seed))
    if os.path.exists(cfile):
        log("case %s seed %d: done (skipped)" % (tag(sigma2), seed))
        return
    case_dir = "%s_seed%d" % (tag(sigma2), seed)
    avail = [N for N in ns if os.path.exists(os.path.join(fv_root, case_dir, "N%d" % N, "report.json"))]
    missing = [N for N in ns if N not in avail]
    if not avail:
        log("case %s seed %d: no FV level available, skipped" % (tag(sigma2), seed))
        return
    if missing:
        log("case %s seed %d: FV levels missing: %s" % (tag(sigma2), seed, missing))
    cache = os.path.join(out, "ref", case_dir)
    faces_a, summ_a = solve_reference(sigma2, seed, M, avail, os.path.join(cache, "M%d" % M), gen_n, tol, maxit, workers, band)
    faces_b, summ_b = solve_reference(sigma2, seed, 2 * M, avail, os.path.join(cache, "M%d" % (2 * M)), gen_n, tol, maxit, workers, band)
    gen = sf18.generate(gen_n, sigma2=1.0, ell=ELL, seed=seed, workers=workers)
    levels = []
    for N in avail:
        rep, arrs = load_fv(os.path.join(fv_root, case_dir, "N%d" % N))
        fv = (arrs["U"], arrs["V"], arrs["W"])
        e = darcy.flux_errors(fv, faces_b[N])  # the finer reference is the target
        sc = darcy.flux_errors(faces_a[N], faces_b[N])  # reference self-convergence on the same faces
        lev = {"N": N, "lambda_over_h": N * ELL, "fv_error": e, "ref_selfconv": sc,
               "ratio_fv_over_selfconv": e["rel_l2_all"] / max(sc["rel_l2_all"], 1e-300),
               "fv_converged": bool(rep["all_correctors_converged"]), "fv_iterations": [c["iterations"] for c in rep["correctors"]],
               "fv_rtol": rep["pcg_rtol"], "fv_G": rep["G"], "fv_K_eff": rep["K_eff"], "fv_applied_scale": rep["sf18"]["applied_scale"],
               "fv_raw_variance": rep["sf18"]["raw_variance"], "fv_seconds": rep["seconds"]["flow"],
               "G_rel_diff_vs_ref": [abs(rep["G"][i] - summ_b["G"][i]) / abs(summ_b["G"][0]) for i in range(3)],
               "Keff_xx_rel_diff_vs_ref": abs(rep["K_eff"][0][0] - summ_b["K_eff"][0][0]) / summ_b["K_eff"][0][0],
               "field_variance_outside_fv_modes": sf18.mode_variance_outside(gen, N)}
        if check_y:
            py = sf18.generate(N, sigma2=1.0, ell=ELL, seed=seed, workers=workers)
            lev["generator_match"] = {"max_abs_dY_over_std": float(np.max(np.abs(py.y - arrs["Y"])) / np.std(arrs["Y"])),
                                      "applied_scale_rel_diff": py.applied_scale / rep["sf18"]["applied_scale"] - 1.0,
                                      "raw_variance_rel_diff": py.raw_variance / rep["sf18"]["raw_variance"] - 1.0,
                                      "Y_variance_gpu": float(np.var(arrs["Y"]))}
        levels.append(lev)
        log("case %s seed %d N=%d: FV err all=%.3e x=%.3e | selfconv %.3e | ratio %.1f" %
            (tag(sigma2), seed, N, e["rel_l2_all"], e["rel_l2_x"], sc["rel_l2_all"], lev["ratio_fv_over_selfconv"]))
    scales = [l["fv_applied_scale"] for l in levels]
    ns_done = [l["N"] for l in levels]
    err_all = [l["fv_error"]["rel_l2_all"] for l in levels]
    err_x = [l["fv_error"]["rel_l2_x"] for l in levels]
    case = {"sigma2": sigma2, "eps": eps_of(sigma2), "seed": seed, "ell": ELL, "gen_n": gen_n, "ref_M": [M, 2 * M], "ref_band": band, "ref_tol": tol,
            "reference": {"M": summ_a, "2M": summ_b}, "levels": levels,
            "orders_all": darcy.observed_orders(ns_done, err_all) if len(levels) > 1 else [],
            "orders_x": darcy.observed_orders(ns_done, err_x) if len(levels) > 1 else [],
            "applied_scale_spread_rel": float((max(scales) - min(scales)) / min(scales)), "missing_levels": missing}
    os.makedirs(os.path.dirname(cfile), exist_ok=True)
    with open(cfile, "w") as fh:
        json.dump(case, fh, indent=1)
    if not keep_ref_faces:
        shutil.rmtree(cache, ignore_errors=True)


# --- acceptance E -------------------------------------------------------------------------------------
def verdict_E(ns, err_all, selfconv_all, order_min=ORDER_MIN, ratio_min=SELFCONV_RATIO):
    """Acceptance E of SF-42 on one ladder: reference converged at every level (self-convergence at least
    `ratio_min` below the FV error), FV error strictly decreasing, observed order over the finest pair >= order_min."""
    if len(ns) < 2:
        raise ValueError("a ladder needs at least two levels")
    orders = darcy.observed_orders(ns, err_all)
    ratios = [e / max(s, 1e-300) for e, s in zip(err_all, selfconv_all)]
    out = {"monotone": all(err_all[i + 1] < err_all[i] for i in range(len(ns) - 1)),
           "order_finest_pair": orders[-1], "order_ok": orders[-1] >= order_min,
           "min_ratio_fv_over_selfconv": min(ratios), "reference_ok": min(ratios) >= ratio_min,
           "finest_N": ns[-1]}
    out["pass"] = bool(out["monotone"] and out["order_ok"] and out["reference_ok"])
    return out


def summarize(out):
    files = sorted(glob.glob(os.path.join(out, "cases", "*.json")))
    cases = []
    for f in files:
        with open(f) as fh:
            cases.append(json.load(fh))
    by_sigma = {}
    for c in cases:
        by_sigma.setdefault(c["sigma2"], []).append(c)
    summary = {"order_min": ORDER_MIN, "selfconv_ratio_min": SELFCONV_RATIO, "sigma2": {}}
    md = ["# SF-42 N3 pseudo-spectral cross-check of the FV velocity\n",
          "Relative L2 error of the FV face fluxes (all faces) against the exact face averages of the pseudo-spectral "
          "reference (finer of the pair M, 2M); `selfconv` = reference M against 2M on the same faces.\n"]
    for s2 in sorted(by_sigma):
        cs = sorted(by_sigma[s2], key=lambda c: c["seed"])
        entry = {"seeds": {}}
        md.append("\n## sigma^2 = %g (eps = %g), reference pair M = %d / %d\n" % (s2, eps_of(s2), cs[0]["ref_M"][0], cs[0]["ref_M"][1]))
        all_pass = True
        for c in cs:
            ns = [l["N"] for l in c["levels"]]
            ea = [l["fv_error"]["rel_l2_all"] for l in c["levels"]]
            ex = [l["fv_error"]["rel_l2_x"] for l in c["levels"]]
            sc = [l["ref_selfconv"]["rel_l2_all"] for l in c["levels"]]
            v = verdict_E(ns, ea, sc) if len(ns) > 1 else {"pass": False, "note": "single level"}
            vx = verdict_E(ns, ex, [l["ref_selfconv"]["rel_l2_x"] for l in c["levels"]]) if len(ns) > 1 else {}
            fv_ok = all(l["fv_converged"] for l in c["levels"])
            v["fv_all_converged"] = fv_ok
            v["x_faces"] = vx
            v["complete_ladder"] = not c["missing_levels"]
            all_pass = all_pass and v["pass"] and fv_ok and v["complete_ladder"]
            entry["seeds"][str(c["seed"])] = {"N": ns, "err_all": ea, "err_x": ex, "selfconv_all": sc,
                                              "orders_all": c["orders_all"], "orders_x": c["orders_x"], "verdict": v,
                                              "applied_scale_spread_rel": c["applied_scale_spread_rel"],
                                              "err_at_lambda_over_h": {str(lh): next((e for n, e in zip(ns, ea) if abs(n * ELL - lh) < 1e-9), None) for lh in REPORT_LH}}
            md.append("\n### seed %d (FV rtol %g; applied-scale spread over the ladder %.2e)\n" % (c["seed"], c["levels"][0]["fv_rtol"], c["applied_scale_spread_rel"]))
            md.append("| N | lambda/h | err all | err x | selfconv all | FV/selfconv | order (all) | FV its | gen. match max\\|dY\\|/std |")
            md.append("|---|---|---|---|---|---|---|---|---|")
            for i, l in enumerate(c["levels"]):
                o = "" if i == 0 else "%.2f" % c["orders_all"][i - 1]
                gm = l.get("generator_match", {}).get("max_abs_dY_over_std")
                md.append("| %d | %g | %.3e | %.3e | %.3e | %.1f | %s | %s | %s |" %
                          (l["N"], l["lambda_over_h"], l["fv_error"]["rel_l2_all"], l["fv_error"]["rel_l2_x"], l["ref_selfconv"]["rel_l2_all"],
                           l["ratio_fv_over_selfconv"], o, "/".join(str(i_) for i_ in l["fv_iterations"]), "n/a" if gm is None else "%.1e" % gm))
            md.append("\nverdict E: monotone=%s, order(finest pair)=%.2f (>= %g: %s), min FV/selfconv=%.1f (>= %g: %s), FV converged=%s, "
                      "complete ladder=%s => **%s**" % (v["monotone"], v["order_finest_pair"], ORDER_MIN, v["order_ok"], v["min_ratio_fv_over_selfconv"],
                                                          SELFCONV_RATIO, v["reference_ok"], fv_ok, v["complete_ladder"],
                                                          "PASS" if (v["pass"] and fv_ok and v["complete_ladder"]) else "FAIL"))
        entry["pass"] = all_pass
        summary["sigma2"][str(s2)] = entry
        md.append("\nsigma^2 = %g: %s\n" % (s2, "PASS" if all_pass else "FAIL"))
    summary["pass"] = bool(summary["sigma2"]) and all(e["pass"] for e in summary["sigma2"].values())
    with open(os.path.join(out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(out, "summary.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    log("summary: %s" % ("PASS" if summary["pass"] else "FAIL"))
    return summary


def scan(sigma2, seed, ms, ns, out, gen_n, tol, maxit, workers, band):
    """Reference self-convergence ladder WITHOUT FV data: solve at every M of `ms` (increasing) and report the relative
    L2 difference of the exact face averages on the FV grids `ns` between consecutive resolutions.  Used to choose
    the pair (M, 2M) of `run` per sigma^2 before the main study."""
    root = os.path.join(out, "scan", "%s_seed%d" % (tag(sigma2), seed))
    faces, summ = {}, {}
    for M in ms:
        faces[M], summ[M] = solve_reference(sigma2, seed, M, ns, os.path.join(root, "M%d" % M), gen_n, tol, maxit, workers, band)
    rows = []
    for a, b in zip(ms[:-1], ms[1:]):
        for N in ns:
            e = darcy.flux_errors(faces[a][N], faces[b][N])
            rows.append({"M_from": a, "M_to": b, "N": N, "rel_l2_all": e["rel_l2_all"], "rel_l2_x": e["rel_l2_x"]})
            log("scan %s seed %d: M %d -> %d on the N=%d faces: rel L2 all %.3e (x %.3e)" % (tag(sigma2), seed, a, b, N, e["rel_l2_all"], e["rel_l2_x"]))
    res = {"sigma2": sigma2, "seed": seed, "band": band, "tol": tol, "ms": list(ms), "ns": list(ns), "rows": rows,
           "solves": {str(M): {k: summ[M][k] for k in ("mcut", "G", "K_eff", "matvecs", "seconds", "correctors")} for M in ms}}
    with open(os.path.join(out, "scan_%s_seed%d.json" % (tag(sigma2), seed)), "w") as fh:
        json.dump(res, fh, indent=1)
    return res


def parse_ref_m(text):
    out = {}
    for item in text.split(","):
        k, v = item.split(":")
        out[float(k)] = int(v)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--fv-root", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--sigma2", type=float, nargs="+", default=list(DEFAULT_SIGMA2))
    r.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    r.add_argument("--ns", type=int, nargs="+", default=list(DEFAULT_NS))
    r.add_argument("--ref-m", type=parse_ref_m, default=dict(DEFAULT_REF_M))
    r.add_argument("--gen-n", type=int, default=64)
    r.add_argument("--tol", type=float, default=1e-12)
    r.add_argument("--band", type=float, default=DEFAULT_BAND)
    r.add_argument("--maxit", type=int, default=2000)
    r.add_argument("--workers", type=int, default=os.cpu_count())
    r.add_argument("--keep-ref-faces", action="store_true")
    r.add_argument("--no-check-y", action="store_true")
    c = sub.add_parser("scan")
    c.add_argument("--out", required=True)
    c.add_argument("--sigma2", type=float, required=True)
    c.add_argument("--seed", type=int, default=DEFAULT_SEEDS[0])
    c.add_argument("--ms", type=int, nargs="+", required=True)
    c.add_argument("--ns", type=int, nargs="+", default=list(DEFAULT_NS))
    c.add_argument("--gen-n", type=int, default=64)
    c.add_argument("--tol", type=float, default=1e-12)
    c.add_argument("--maxit", type=int, default=2000)
    c.add_argument("--band", type=float, default=DEFAULT_BAND)
    c.add_argument("--workers", type=int, default=os.cpu_count())
    s = sub.add_parser("summarize")
    s.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        for s2 in a.sigma2:
            if s2 not in a.ref_m:
                raise SystemExit("no reference resolution for sigma2=%g (use --ref-m)" % s2)
        for s2 in a.sigma2:
            for seed in a.seeds:
                run_case(s2, seed, a.ns, a.ref_m[s2], a.fv_root, a.out, a.gen_n, a.tol, a.maxit, a.workers, a.keep_ref_faces,
                         check_y=not a.no_check_y, band=a.band)
        summarize(a.out)
    elif a.cmd == "scan":
        scan(a.sigma2, a.seed, sorted(a.ms), a.ns, a.out, a.gen_n, a.tol, a.maxit, a.workers, a.band)
    else:
        summarize(a.out)


if __name__ == "__main__":
    main()
