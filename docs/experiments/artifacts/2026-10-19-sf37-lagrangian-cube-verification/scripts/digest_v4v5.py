#!/usr/bin/env python3
"""SF-37 digest of V4 (Lagrangian e_v), V5 (covector stretching, pair decorrelation) and the diagnostics D4, D3 (passive I).

  digest_v4v5.py ROOT OUTDIR [--design full|discovered]

ROOT follows the campaign layout C-LAYOUT.  V4 reads the v2 runs (all N); V5 and the passive-I check read the v1 runs
(covectors on, seed_v.npy).  `full` = pre-registered design with completeness (missing runs -> `not_evaluable`, never
`pass`); `discovered` = the design implied by the runs found.  Thresholds: V4 order >= 1.8 on 256 -> 512 and a strict
decrease on 64 -> 128 -> 256 -> 512 (RR4, error convention log2(e_N / e_2N)); V5 has no threshold.
Outputs: v4v5_summary.md, csv files, input_manifest_v4v5.csv.  python3 + numpy + scipy.
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sf37lib as L  # noqa: E402

NAN = float("nan")
X1_MAX = 4.0


def k2(e, s, n):
    return ("v2", L.v2_name(e, s, n))


def k1(e, s, n, tol, h):
    return ("v1", L.v1_name(e, s, n, tol, h))


def chain_of(design):
    nr = design.n_ref
    return (nr // 8, nr // 4, nr // 2, nr)


# -------------------------------------------------------------------------------------------------------------
# per-plane e_v statistics
# -------------------------------------------------------------------------------------------------------------
def ev_arrays(run, m):
    """flux weights, e = 1 - exp(-I), I, tau over the contributors of plane m (index order)."""
    sel = run.plane_mask()[:, m]
    w = run.w[sel]
    I = run.mem("planes_I.npy")[sel, m]
    tau = run.mem("planes_tau.npy")[sel, m]
    e = 1.0 - np.exp(-I)
    return w, e, I, tau


def wrms(w, v):
    W = L.seq_sum(w)
    if not (W > 0.0):
        return NAN
    return math.sqrt(L.seq_sum(w * v * v) / W)


def trimmed_rms(w, v, q=0.999):
    """RMS over the particles with |v| <= the weighted q-quantile of |v| (the top 1-q of the weight is dropped)."""
    av = np.abs(v)
    cut = L.wquantile(av, w, q)
    if not math.isfinite(cut):
        return NAN
    sel = av <= cut
    return wrms(w[sel], v[sel])


def ev_stats(run, m):
    w, e, I, tau = ev_arrays(run, m)
    out = {"n": int(w.size)}
    ae, aI = np.abs(e), np.abs(I)
    out["rms_e"] = wrms(w, e)
    out["rms_I"] = wrms(w, I)
    out["rms_e_trim"] = trimmed_rms(w, e)
    for q, tag in ((0.5, "p50"), (0.9, "p90"), (0.99, "p99"), (0.999, "p999"), (1.0, "max")):
        out["e_" + tag] = L.wquantile(ae, w, q)
        out["I_" + tag] = L.wquantile(aI, w, q)
    ok = np.isfinite(e) & (w > 0)
    tot = float((w[ok] * e[ok] ** 2).sum())
    out["share_top10"] = NAN
    out["share_top0.1pct_weight"] = NAN
    if tot > 0:
        contrib = np.where(ok, w * e * e, 0.0)
        top = np.argsort(-contrib, kind="stable")[:10]
        out["share_top10"] = float(contrib[top].sum() / tot)
        o = np.argsort(-np.where(ok, ae, -np.inf), kind="stable")
        cw = np.cumsum(np.where(ok[o], w[o], 0.0))
        k = int(np.searchsorted(cw, 1e-3 * cw[-1], side="left")) + 1
        out["share_top0.1pct_weight"] = float(contrib[o[:k]].sum() / tot)
    # correlation of |I| with tau (flux-weighted Pearson, unweighted Spearman)
    out["corr_absI_tau_w"] = NAN
    out["spearman_absI_tau"] = NAN
    good = np.isfinite(aI) & np.isfinite(tau) & (w > 0)
    if good.sum() > 2:
        ww = w[good] / w[good].sum()
        a, b = aI[good], tau[good]
        ma, mb = (ww * a).sum(), (ww * b).sum()
        va, vb = (ww * (a - ma) ** 2).sum(), (ww * (b - mb) ** 2).sum()
        if va > 0 and vb > 0:
            out["corr_absI_tau_w"] = float((ww * (a - ma) * (b - mb)).sum() / math.sqrt(va * vb))
        from scipy import stats as sps
        out["spearman_absI_tau"] = float(sps.spearmanr(a, b)[0])
    return out


def log2_ratio_se(ra, rb, m, counts, g=4):
    """Paired block-bootstrap SE of log2(RMS_a / RMS_b) at plane m (RMS^2 = sum w e^2 / sum w per resample)."""
    ids = L.block_ids(ra.Np, g)
    nb = g * g
    ms = []
    for r in (ra, rb):
        sel = r.plane_mask()[:, m]
        w = np.where(sel, r.w, 0.0)
        e = np.where(sel, 1.0 - np.exp(-r.mem("planes_I.npy")[:, m]), 0.0)
        S = np.bincount(ids, weights=w * e * e, minlength=nb)
        Wb = np.bincount(ids, weights=w, minlength=nb)
        with np.errstate(divide="ignore", invalid="ignore"):
            ms.append((counts @ S) / (counts @ Wb))
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = 0.5 * np.log2(ms[0] / ms[1])
    lr = lr[np.isfinite(lr)]
    return float(np.std(lr, ddof=1)) if lr.size > 1 else NAN


def v4_verdict(vals, chain_order_idx=(2, 3), thr=L.V4_ORDER):
    """vals = RMS at the chain (64, 128, 256, 512). -> (verdict, decreases, order)."""
    if len(vals) < 4 or not all(math.isfinite(v) for v in vals):
        return "not_evaluable", False, NAN
    dec = all(vals[i + 1] < vals[i] for i in range(len(vals) - 1))
    p = L.order_pair(vals[chain_order_idx[0]], vals[chain_order_idx[1]])
    return ("pass" if (dec and math.isfinite(p) and p >= thr) else "fail"), dec, p


def main(argv=None):
    ap = argparse.ArgumentParser(description="SF-37 V4/V5 digest")
    ap.add_argument("root")
    ap.add_argument("outdir")
    ap.add_argument("--design", choices=("full", "discovered"), default="full")
    args = ap.parse_args(argv)
    argv_text = "digest_v4v5.py %s %s --design %s" % (args.root, args.outdir, args.design)

    manifest = L.Manifest(args.root)
    camp = L.Campaign(args.root, manifest)
    idx16 = L.load_bootstrap(manifest=manifest)
    design = L.Design.full() if args.design == "full" else L.Design.discovered(camp)
    eng = L.PairEngine(camp, idx16)
    counts16 = eng.counts(4)
    W = L.OutWriter(args.outdir, manifest, "digest_v4v5", argv_text)
    chain = chain_of(design)
    md = W.md_header("SF-37 digest V4 / V5 / D3 (passive I) / D4",
                     ["design: `%s` (eps codes %s, seeds %s, v2 N %s, v1 N %s); V4 chain %s" %
                      (design.kind, list(design.eps_codes), list(design.seeds), list(design.v2_ns), list(design.v1_ns),
                       list(chain))])

    # ---- completeness
    exp = [x for x in design.expected_runs()]
    missing = [(p, n, camp.exit_codes(p).get(n)) for p, n in exp if camp.run(p, n) is None]
    W.csv("completeness.csv", ["part", "run", "status", "exit_code"],
          [[p, n, "missing" if camp.run(p, n) is None else "present", camp.exit_codes(p).get(n, "")] for p, n in exp])
    md += ["", "## Completeness", "",
           "Expected runs: %d; missing: %d (missing runs make dependent cells `not_evaluable`, never `pass`)." %
           (len(exp), len(missing))]
    if missing:
        md += ["", L.md_table(["part", "run", "exit code"], [(m[0], m[1], "" if m[2] is None else m[2]) for m in missing[:200]])]

    # ---- V4
    v4rows, v4md, d4rows, d4se = [], [], [], []
    for e in design.eps_codes:
        for s in design.seeds:
            runs = {n: camp.run(*k2(e, s, n)) for n in design.v2_ns}
            ref_run = next((r for r in runs.values() if r is not None), None)
            planes = ref_run.planes if ref_run is not None else np.array(design.planes)
            for m, x1 in enumerate(planes):
                if x1 > X1_MAX + 1e-12:
                    continue
                stats = {n: (ev_stats(r, m) if r is not None else None) for n, r in runs.items()}
                rms = {n: (stats[n]["rms_e"] if stats[n] else NAN) for n in design.v2_ns}
                chain_vals = [rms.get(n, NAN) for n in chain]
                verdict, dec, p = v4_verdict(chain_vals)
                # D4 robustness: the same rule on RMS I and on the 99.9 %-trimmed RMS
                shadow = {}
                for key in ("rms_I", "rms_e_trim"):
                    cv = [(stats[n][key] if stats.get(n) else NAN) for n in chain]
                    shadow[key] = v4_verdict(cv)
                trip = L.order_triple(*[rms.get(n, NAN) for n in design.n_triple])
                se = NAN
                if runs.get(chain[2]) is not None and runs.get(chain[3]) is not None:
                    se = log2_ratio_se(runs[chain[2]], runs[chain[3]], m, counts16)
                row = [e, L.sigma2_of(e), s, x1] + [rms.get(n, NAN) for n in design.v2_ns] + [dec, p, verdict]
                v4rows.append(row)
                d4se.append([e, s, x1, p, se, trip, shadow["rms_I"][0], shadow["rms_I"][2], shadow["rms_e_trim"][0],
                             shadow["rms_e_trim"][2]])
                for n in design.v2_ns:
                    st = stats[n]
                    if st is None:
                        continue
                    d4rows.append([e, s, n, x1, st["n"], st["rms_e"], st["rms_I"], st["rms_e_trim"], st["e_p50"],
                                   st["e_p90"], st["e_p99"], st["e_p999"], st["e_max"], st["I_p50"], st["I_p90"],
                                   st["I_p99"], st["I_p999"], st["I_max"], st["share_top10"], st["share_top0.1pct_weight"],
                                   st["corr_absI_tau_w"], st["spearman_absI_tau"]])
    hdr4 = ["eps_code", "sigma2", "seed", "plane"] + ["rms_ev_N%d" % n for n in design.v2_ns] + ["decreases_on_chain", "order_256_512", "verdict"]
    W.csv("v4_cells.csv", hdr4, v4rows)
    W.csv("d4_percentiles.csv", ["eps_code", "seed", "N", "plane", "n_contributors", "rms_e", "rms_I", "rms_e_trim999",
                                 "abs_e_p50", "abs_e_p90", "abs_e_p99", "abs_e_p999", "abs_e_max", "abs_I_p50",
                                 "abs_I_p90", "abs_I_p99", "abs_I_p999", "abs_I_max", "share_top10",
                                 "share_top0.1pct_weight", "corr_absI_tau_flux_weighted", "spearman_absI_tau"], d4rows)
    W.csv("d4_robustness.csv", ["eps_code", "seed", "plane", "order_256_512_rms_e", "se_bootstrap_log2_rms_ratio",
                                "triple_order_rms_e", "v4_rule_on_rms_I", "order_rms_I", "v4_rule_on_rms_e_trim999",
                                "order_rms_e_trim999"], d4se)
    md += ["", "## V4 - Lagrangian e_v = 1 - exp(-I)", "",
           "Rule: at every amplitude, seed and plane x1 <= 4, the flux-weighted RMS e_v decreases strictly on every "
           "refinement %s and `log2(RMS(%d)/RMS(%d)) >= 1.8` (RR4 error convention)." %
           (" -> ".join(str(n) for n in chain), chain[2], chain[3]), ""]
    amp = []
    for e in design.eps_codes:
        sub = [r for r in v4rows if r[0] == e]
        amp.append([L.sigma2_of(e), len(sub), sum(1 for r in sub if r[-1] == "fail"),
                    sum(1 for r in sub if r[-1] == "not_evaluable"),
                    "fail" if any(r[-1] == "fail" for r in sub) else ("not_evaluable" if (not sub or any(r[-1] == "not_evaluable" for r in sub)) else "pass")])
    md += [L.md_table(["sigma2", "cells", "fail", "not_evaluable", "V4 verdict"], amp), "", "### V4 cells", "",
           L.md_table(["sigma2", "seed", "plane"] + ["RMS e_v N%d" % n for n in design.v2_ns] + ["strict decrease", "order %d->%d" % (chain[2], chain[3]), "verdict"],
                      [[r[1], r[2], r[3]] + r[4:] for r in v4rows]), ""]
    nsh = [sum(1 for r in d4se if r[6] == "pass"), sum(1 for r in d4se if r[8] == "pass"), len(d4se)]
    md += ["### D4 - robustness (recorded, no thresholds)", "",
           "The V4 rule re-run on RMS I passes in %d of %d cells and on the 99.9 %%-trimmed RMS e_v in %d of %d cells. "
           "Weighted percentiles of |e_v| and |I| (50/90/99/99.9/max), RMS I, the share of sum w e^2 of the top 10 particles and "
           "of the top 0.1 %% of the weight (ranked by |e_v|), the correlation of |I| with tau, the bootstrap SE of "
           "log2(RMS_%d/RMS_%d) and the triple order: `d4_percentiles.csv`, `d4_robustness.csv`." %
           (nsh[0], nsh[2], nsh[1], nsh[2], chain[2], chain[3]), ""]

    # ---- D3 passive-I check (v1 runs)
    d3 = []
    for e in design.eps_codes:
        for s in design.seeds:
            for n in design.v1_ns:
                ra, rh = camp.run(*k1(e, s, n, L.TOL_PRIMARY, 8)), camp.run(*k1(e, s, n, L.TOL_REF, 32))
                planes = (rh.planes if rh is not None else (ra.planes if ra is not None else np.array(design.planes)))
                for m, x1 in enumerate(planes):
                    if x1 > X1_MAX + 1e-12:
                        continue
                    if ra is None or rh is None:
                        d3.append([e, s, n, x1, NAN, NAN, NAN])
                        continue
                    a = ev_stats(ra, m)["rms_e"]
                    b = ev_stats(rh, m)["rms_e"]
                    d3.append([e, s, n, x1, a, b, abs(a - b) / b if b else NAN])
    W.csv("d3_passive_I.csv", ["eps_code", "seed", "N", "plane", "rms_ev_h8_1e-8", "rms_ev_h32_1e-10", "rel_diff"], d3)
    d3ord = []
    for e in design.eps_codes:
        for s in design.seeds:
            ra, rb = camp.run(*k1(e, s, chain[2], L.TOL_REF, 32)), camp.run(*k1(e, s, chain[3], L.TOL_REF, 32))
            planes = rb.planes if rb is not None else np.array(design.planes)
            for m, x1 in enumerate(planes):
                if x1 > X1_MAX + 1e-12:
                    continue
                if ra is None or rb is None:
                    d3ord.append([e, s, x1, NAN, NAN, NAN, "not_evaluable"])
                    continue
                qa, qb = ev_stats(ra, m)["rms_e"], ev_stats(rb, m)["rms_e"]
                p = L.order_pair(qa, qb)
                d3ord.append([e, s, x1, qa, qb, p, "pass" if (math.isfinite(p) and p >= L.V4_ORDER) else "fail"])
    W.csv("d3_v4_order_on_reference_runs.csv", ["eps_code", "seed", "plane", "rms_ev_256", "rms_ev_512", "order_256_512",
                                                "shadow_order_ge_1.8"], d3ord)
    rel = [r[6] for r in d3 if r[6] == r[6]]
    md += ["## D3 - passive-I check (recorded)", "",
           "`|RMS e_v(h/8, 1e-8) - RMS e_v(h/32, 1e-10)| / RMS e_v(h/32, 1e-10)` at x1 <= 4 on the v1 runs: %s over %d cells "
           "(max %s). V4 order %d -> %d on the reference runs (shadow, order only): %d of %d cells >= 1.8. "
           "Detail: `d3_passive_I.csv`, `d3_v4_order_on_reference_runs.csv`." %
           ("median %s" % L.fnum(float(np.median(rel))) if rel else "n/a", len(rel), L.fnum(max(rel)) if rel else "n/a",
            chain[2], chain[3], sum(1 for r in d3ord if r[6] == "pass"), len(d3ord)), ""]

    # ---- V5 covector statistics
    cov_rows, rate_rows, slope_rows = [], [], []
    seedv_missing = []
    for e in design.eps_codes:
        for s in design.seeds:
            for n in design.v1_ns:
                for hdiv, tol in ((8, L.TOL_PRIMARY), (32, L.TOL_REF)):
                    key = k1(e, s, n, tol, hdiv)
                    r = camp.run(*key)
                    if r is None or not r.exists("planes_cov.npy"):
                        continue
                    cov = r.read_array("planes_cov.npy")
                    mp = r.plane_mask()
                    tau = r.mem("planes_tau.npy")
                    w = r.w
                    a0 = b0 = None
                    if r.exists("seed_v.npy"):
                        v = r.read_array("seed_v.npy").astype(np.float64)
                        a0 = np.sqrt(v[:, 1] ** 2 + v[:, 0] ** 2)
                        b0 = np.sqrt((v[:, 2] / v[:, 0]) ** 2 + 1.0)
                    else:
                        seedv_missing.append(r.name)
                    med_L = {"a": [], "b": []}
                    med_tau = []
                    for m, x1 in enumerate(r.planes):
                        sel = mp[:, m]
                        ws = w[sel]
                        ones = np.ones(int(sel.sum()))
                        na = np.sqrt((cov[sel, m, 0:3] ** 2).sum(axis=1))
                        nb_ = np.sqrt((cov[sel, m, 3:6] ** 2).sum(axis=1))
                        row = [e, s, n, "tol%s_h%d" % (L.fmt_tol(tol), hdiv), x1, int(sel.sum())]
                        for nm_, vv in (("a", na), ("b", nb_)):
                            row += [L.wquantile(vv, ones, q) for q in (0.5, 0.9, 0.99)] + [float(vv.max()) if vv.size else NAN]
                            row += [L.wquantile(vv, ws, q) for q in (0.5, 0.9, 0.99)]
                        cov_rows.append(row)
                        tsel = tau[sel, m]
                        rr = [e, s, n, "tol%s_h%d" % (L.fmt_tol(tol), hdiv), x1]
                        med_tau.append(float(np.median(tsel)) if tsel.size else NAN)
                        for nm_, vv, v0 in (("a", na, a0), ("b", nb_, b0)):
                            if v0 is None or not sel.any():
                                rr += [NAN] * 5
                                med_L[nm_].append(NAN)
                                continue
                            lg = np.log(vv / v0[sel])
                            rate = lg / tsel
                            rr += [L.wquantile(rate, ones, q) for q in (0.01, 0.5, 0.9, 0.99)] + [float(rate.max())]
                            med_L[nm_].append(float(np.median(lg)))
                        rate_rows.append(rr)
                    mt = np.array(med_tau)
                    for nm_ in ("a", "b"):
                        ml = np.array(med_L[nm_])
                        for i in range(len(mt) - 1):
                            dtau = mt[i + 1] - mt[i]
                            dlt = math.log(mt[i + 1]) - math.log(mt[i]) if (mt[i] > 0 and mt[i + 1] > 0) else NAN
                            slope_rows.append([e, s, n, "tol%s_h%d" % (L.fmt_tol(tol), hdiv), nm_, r.planes[i], r.planes[i + 1],
                                               mt[i], mt[i + 1], ml[i], ml[i + 1],
                                               (ml[i + 1] - ml[i]) / dtau if dtau else NAN,
                                               (ml[i + 1] - ml[i]) / dlt if (dlt == dlt and dlt != 0) else NAN])
    cov_hdr = ["eps_code", "seed", "N", "setting", "plane", "n_contributors"]
    for nm_ in ("a", "b"):
        cov_hdr += ["|%s|_p50" % nm_, "|%s|_p90" % nm_, "|%s|_p99" % nm_, "|%s|_max" % nm_, "|%s|_p50_fluxw" % nm_,
                    "|%s|_p90_fluxw" % nm_, "|%s|_p99_fluxw" % nm_]
    W.csv("v5_covector_norms.csv", cov_hdr, cov_rows)
    rate_hdr = ["eps_code", "seed", "N", "setting", "plane"]
    for nm_ in ("a", "b"):
        rate_hdr += ["rate_%s_p01" % nm_, "rate_%s_p50" % nm_, "rate_%s_p90" % nm_, "rate_%s_p99" % nm_,
                     "rate_%s_max" % nm_]
    W.csv("v5_rates.csv", rate_hdr, rate_rows)
    W.csv("v5_local_slopes.csv", ["eps_code", "seed", "N", "setting", "covector", "plane_i", "plane_j", "median_tau_i",
                                  "median_tau_j", "median_log_ratio_i", "median_log_ratio_j", "slope_vs_tau",
                                  "slope_vs_log_tau"], slope_rows)

    # ---- V5 decorrelation fits
    def fits_for(pair_specs, e, label):
        out = []
        for s in design.seeds:
            for n in design.v1_ns:
                ka, kb = pair_specs(e, s, n)
                a, b = camp.run(*ka), camp.run(*kb)
                if a is None or b is None:
                    out.append((s, n, {"name": "insufficient", "n": 0, "rss_exp": NAN, "rss_alg": NAN,
                                       "slope_exp": NAN, "slope_alg": NAN}, "missing run"))
                    continue
                sp = L.pair_separation(a, b, "plane")
                out.append((s, n, L.fit_decorrelation(b.planes, sp["rms"], sp["rel"]), ""))
        return out

    primary = lambda e, s, n: (k1(e, s, n, L.TOL_PRIMARY, 8), k1(e, s, n, L.TOL_REF, 8))
    fallbacks = [("FALLBACK (h8,1e-10) vs (h32,1e-10)", lambda e, s, n: (k1(e, s, n, L.TOL_REF, 8), k1(e, s, n, L.TOL_REF, 32))),
                 ("FALLBACK (h8,1e-8) vs (h32,1e-10)", lambda e, s, n: (k1(e, s, n, L.TOL_PRIMARY, 8), k1(e, s, n, L.TOL_REF, 32))),
                 ("FALLBACK (1e-6,1e-10) at h8", lambda e, s, n: (k1(e, s, n, 1e-6, 8), k1(e, s, n, L.TOL_REF, 8)))]
    fit_rows, name_rows = [], []
    for e in design.eps_codes:
        res = fits_for(primary, e, "primary")
        for s, n, f, note in res:
            fit_rows.append([e, s, n, "primary (1e-8 vs 1e-10 at h8)", f["n"], f["rss_exp"], f["rss_alg"], f["slope_exp"], f["slope_alg"], f["name"], note])
        nm = L.amplitude_name([f["name"] for _, _, f, _ in res])
        fb_txt = []
        for lab, spec in fallbacks:
            rf = fits_for(spec, e, lab)
            for s, n, f, note in rf:
                fit_rows.append([e, s, n, lab, f["n"], f["rss_exp"], f["rss_alg"], f["slope_exp"], f["slope_alg"], f["name"], note])
            fb_txt.append("%s: %s" % (lab, L.amplitude_name([f["name"] for _, _, f, _ in rf])))
        name_rows.append([L.sigma2_of(e), sum(1 for _, _, f, _ in res if f["name"] in ("exponential", "algebraic")), len(res), nm,
                          "; ".join(fb_txt) if nm == "insufficient" else "(fallbacks recorded in v5_fits.csv)"])
    W.csv("v5_fits.csv", ["eps_code", "seed", "N", "pair", "eligible_planes", "rss_exponential", "rss_algebraic",
                          "slope_exponential_per_x1", "slope_algebraic_per_logx1", "better_fit", "note"], fit_rows)
    md += ["## V5 - covector stretching and pair decorrelation (recorded, no threshold)", "",
           "Fits of `log RMS(dx_T)` on the planes where `1e-12 < RMS(dx_T) < 0.1 sigma_T` for the pair (1e-8, 1e-10) at h/8 "
           "(>= 3 eligible planes, else `insufficient`): exponential (on x1) vs algebraic (on log x1), the better residual sum of "
           "squares names the fit; amplitude name = majority over the evaluable (seed, N) cases. Fallback pairs are reported "
           "as FALLBACK in the CSV and never as the named result.", "",
           L.md_table(["sigma2", "evaluable cases", "cases", "named fit", "fallback note"], name_rows), "",
           "### Fits per case", "",
           L.md_table(["eps_code", "seed", "N", "pair", "eligible planes", "RSS exp", "RSS alg", "slope exp", "slope alg", "better fit", "note"],
                      fit_rows), ""]
    md += ["### Covector norms per plane (primary setting and reference; unweighted over contributors, flux-weighted in the CSV)", ""]
    sel_rows = [r for r in cov_rows if r[3] == "tol1e-8_h8"]
    md.append(L.md_table(["sigma2", "seed", "N", "plane", "n", "|a| p50", "|a| p90", "|a| p99", "|a| max", "|b| p50", "|b| p90",
                          "|b| p99", "|b| max"],
                         [[L.sigma2_of(r[0]), r[1], r[2], r[4], r[5], r[6], r[7], r[8], r[9], r[13], r[14], r[15], r[16]] for r in sel_rows]))
    md += ["", "Finite-time rates `log(|a|/|a0|)/tau`, `log(|b|/|b0|)/tau`, and the local slopes of the median `log(|a|/|a0|)` vs tau and vs log tau: "
           "`v5_rates.csv`, `v5_local_slopes.csv`."]
    if seedv_missing:
        md += ["", "`seed_v.npy` absent for %d v1 runs (rates and slopes `nan` there): %s" % (len(seedv_missing), ", ".join(sorted(set(seedv_missing))[:20]))]

    W.md("v4v5_summary.md", "\n".join(md))
    nman, dig = W.write_manifest("input_manifest_v4v5.csv")
    with open(os.path.join(args.outdir, "v4v5_summary.md"), "a") as f:
        f.write("\n## Input manifest\n\n%d input files (path, sha256, bytes in `input_manifest_v4v5.csv`); sha256 of the manifest "
                "listing: `%s`.\n" % (nman, dig))
    print("digest_v4v5: wrote %d files to %s (missing runs: %d)" % (len(W.written), args.outdir, len(missing)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
