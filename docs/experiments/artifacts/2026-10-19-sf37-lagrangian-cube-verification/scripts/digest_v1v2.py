#!/usr/bin/env python3
"""SF-37 digest of V1 (tolerance), V2 (grid), the production lambda/h and the diagnostics D1-D3, D8, D9.

  digest_v1v2.py ROOT OUTDIR [--design full|discovered]

ROOT follows the campaign layout C-LAYOUT (ROOT/<part>/<run>/ particle directories, <run>.log, <run>.json,
exit_codes.tsv).  `full` = the pre-registered design with a completeness check (a missing run makes the dependent
cells `not_evaluable`, never `pass`); `discovered` = the design implied by the runs found.  Thresholds are the
pre-registered ones of the SF-37 activation row (sf37lib.py); nothing is tuned.  Outputs in OUTDIR: v1v2_summary.md
(every V1 / V2 cell with verdict and branch, the lambda/h table, the diagnostics summaries), csv files with the
full detail and input_manifest_v1v2.csv (sha256 of every file read).  python3 + numpy + scipy.
"""
import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sf37lib as L  # noqa: E402

NAN = float("nan")
BR = {"det": "d", "order": "o", "stat": "s", "fail": "F", "not_evaluable": "?"}


def k2(e, s, n):
    return ("v2", L.v2_name(e, s, n))


def k1(e, s, n, tol, h):
    return ("v1", L.v1_name(e, s, n, tol, h))


def blank_v1(reason):
    return {"verdict": "not_evaluable", "branch": "not_evaluable", "binding": "", "thr_rel": NAN,
            "thr_stat": NAN, "z": NAN, "reason": reason}


class Ctx:
    def __init__(self, camp, design, eng):
        self.camp = camp
        self.D = design
        self.eng = eng

    def run(self, key):
        return self.camp.run(*key)


# -------------------------------------------------------------------------------------------------------------
# V1, RR15
# -------------------------------------------------------------------------------------------------------------
def do_v1(ctx):
    D = ctx.D
    cells, ladder, vac, shadow, ident_rows = [], [], [], [], []
    h8 = 8
    for e in D.eps_codes:
        for s in D.seeds:
            for n in D.v1_ns:
                kp, kr, kh = k1(e, s, n, L.TOL_PRIMARY, h8), k1(e, s, n, L.TOL_REF, h8), k1(e, s, n, L.TOL_REF, 32)
                ra, rb, rh = ctx.run(kp), ctx.run(kr), ctx.run(kh)
                ps = ctx.eng.pair(kp, kr)
                sep = L.pair_separation(ra, rb, "plane") if (ra and rb) else None
                far = L.far_index(sep["rel"]) if sep else None
                planes = list(rb.planes) if rb else list(D.planes)
                for m, pl in enumerate(planes):
                    if ps is None:
                        c = blank_v1("missing run")
                        qa = qr = Dd = se = NAN
                    else:
                        S = ps["sigma_T2"]
                        qa, qr, Dd, se = S["a"][m], S["b"][m], S["D"][m], S["se"][m]
                        c = L.v1_cell(Dd, qr, se)
                        c["reason"] = ""
                    cells.append({
                        "eps_code": e, "sigma2": L.sigma2_of(e), "seed": s, "N": n, "plane": pl, "q_1e-8": qa,
                        "q_1e-10": qr, "D": Dd, "se_D": se, "z": c["z"], "thr_rel": c["thr_rel"],
                        "thr_2se": c["thr_stat"], "binding": c["binding"], "branch": c["branch"],
                        "verdict": c["verdict"], "reason": c["reason"],
                        "ms_over_sigma_T2": sep["ratio"][m] if sep else NAN,
                        "rms_over_sigma_T": sep["rel"][m] if sep else NAN,
                        "far": (far is not None and m >= far),
                        "stat2se_over_qref": (c["thr_stat"] / abs(qr)) if (ps is not None and qr) else NAN})
                # ladder: A) each tol vs (1e-10, h8); B) each h8 setting vs the reference (1e-10, h32)
                pairs = []
                for t in D.v1_tols:
                    if t != L.TOL_REF:
                        pairs.append(("tol_vs_1e-10_h8", k1(e, s, n, t, h8), kr, "tol%s_h8" % L.fmt_tol(t), "tol1e-10_h8"))
                for t in D.v1_tols:
                    pairs.append(("h8_vs_ref_h32", k1(e, s, n, t, h8), kh, "tol%s_h8" % L.fmt_tol(t), "tol1e-10_h32"))
                for kind, ka, kb, la, lb in pairs:
                    pst = ctx.eng.pair(ka, kb)
                    a_, b_ = ctx.run(ka), ctx.run(kb)
                    sp = L.pair_separation(a_, b_, "plane") if (a_ and b_) else None
                    fp = L.far_index(sp["rel"]) if sp else None
                    fp_txt = ("%g" % b_.planes[fp]) if (sp is not None and fp is not None) else "none"
                    for qn in L.PLANE_SERIES:
                        for m, pl in enumerate(planes):
                            if pst is None:
                                vals = (NAN, NAN, NAN, NAN)
                                cc = blank_v1("missing run")
                            else:
                                S = pst[qn]
                                vals = (S["a"][m], S["b"][m], S["D"][m], S["se"][m])
                                cc = L.v1_cell(vals[2], vals[1], vals[3])
                            ladder.append([e, s, n, kind, la, lb, qn, pl, vals[0], vals[1], vals[2], vals[3],
                                           cc["z"], cc["branch"], cc["verdict"],
                                           sp["ratio"][m] if sp else NAN, sp["rel"][m] if sp else NAN, fp_txt])
                # shadow classifications (RR15): pure h_max effect and the primary against the reference
                for la, ka in (("tol1e-10_h8", kr), ("tol1e-8_h8", kp)):
                    pst = ctx.eng.pair(ka, kh)
                    a_ = ctx.run(ka)
                    sp = L.pair_separation(a_, rh, "plane") if (a_ and rh) else None
                    fp = L.far_index(sp["rel"]) if sp else None
                    if pst is None:
                        shadow.append([e, s, n, la + "_vs_tol1e-10_h32", "not_evaluable", NAN, NAN, "n/a", "missing run"])
                        continue
                    cc = [L.v1_cell(pst["sigma_T2"]["D"][m], pst["sigma_T2"]["b"][m], pst["sigma_T2"]["se"][m])
                          for m in range(len(planes))]
                    nfail = sum(1 for x in cc if x["verdict"] == "fail")
                    nne = sum(1 for x in cc if x["verdict"] == "not_evaluable")
                    verdict = "fail" if nfail else ("not_evaluable" if nne else "pass")
                    shadow.append([e, s, n, la + "_vs_tol1e-10_h32", verdict, nfail, float(np.nanmax(sp["ratio"])) if sp else NAN,
                                   ("%g" % rh.planes[fp]) if (sp is not None and fp is not None) else "none", ""])
                # RR15 vacuity: counters and bitwise identity against (1e-10, h8)
                rows_here = []
                for t in list(D.v1_tols) + ["ref_h32"]:
                    if t == "ref_h32":
                        key, lab = kh, "tol1e-10_h32"
                    else:
                        key, lab = k1(e, s, n, t, h8), "tol%s_h8" % L.fmt_tol(t)
                    r = ctx.run(key)
                    if r is None:
                        rows_here.append((t, lab, None))
                        continue
                    trk = r.track()
                    cp, ct = L.pass_counters(trk, "plane_pass"), L.pass_counters(trk, "time_pass")
                    if rb is not None:
                        idt = L.bitwise_identical_particles(r, rb)
                        stid = L.statuses_identical(r, rb)
                        wfrac = float(rb.w[idt].sum() / rb.w.sum()) if rb.w.sum() > 0 else NAN
                        nid = int(idt.sum())
                        pst = ctx.eng.pair(key, kr)
                        firstD = "none"
                        if pst is not None:
                            nz = np.nonzero(pst["sigma_T2"]["D"] != 0.0)[0]
                            if nz.size:
                                firstD = "%g" % planes[int(nz[0])]
                    else:
                        wfrac, nid, stid, firstD = NAN, -1, False, "n/a"
                    rows_here.append((t, lab, (cp, ct, wfrac, nid, stid, firstD, r.Np)))
                    vac.append([e, s, n, lab, cp["nfev"], cp["accepted_steps"], cp["rejected_steps"], cp["landing_steps"],
                                cp["landing_rejected"], ct["nfev"], ct["accepted_steps"], ct["rejected_steps"],
                                ct["landing_steps"], ct["landing_rejected"], wfrac, nid, r.Np, stid, firstD])
                prim = [x for x in rows_here if x[0] == L.TOL_PRIMARY]
                ident_rows.append((e, s, n, prim[0][2] if prim else None))
    return cells, ladder, vac, shadow, ident_rows


def v1_label(cells_sub, ident):
    """Verdict + RR15 label of one (eps, N, seed) block of V1 cells."""
    if not cells_sub:
        return "not_evaluable"
    if any(c["verdict"] == "fail" for c in cells_sub):
        return "not met"
    if any(c["verdict"] == "not_evaluable" for c in cells_sub):
        return "not_evaluable"
    if ident is not None and ident[3] == ident[6] and ident[4]:  # n_ident == Np and statuses identical
        return "met, vacuous (tol-independent step sequence)"
    return "met"


# -------------------------------------------------------------------------------------------------------------
# V2 and lambda/h
# -------------------------------------------------------------------------------------------------------------
def v2_block(ctx, series_names, e, s):
    """V2 cells for one (eps, seed) over the given series names."""
    D = ctx.D
    nc, nm, nf = D.n_triple
    pcm = ctx.eng.pair(k2(e, s, nc), k2(e, s, nm))
    pmf = ctx.eng.pair(k2(e, s, nm), k2(e, s, nf))
    rm, rf = ctx.run(k2(e, s, nm)), ctx.run(k2(e, s, nf))
    sep = L.pair_separation(rm, rf, "plane") if (rm and rf) else None
    far = L.far_index(sep["rel"]) if sep else None
    out = []
    for qn in series_names:
        if pcm is None or pmf is None:
            n_ax = len(L.series_axis_labels(qn, D.planes, D.times))
            labels = L.series_axis_labels(qn, D.planes, D.times)
            for i in range(n_ax):
                out.append({"series": qn, "i": i, "label": labels[i], "q_c": NAN, "q_m": NAN, "q_f": NAN,
                            "cell": L.v2_cell(NAN, NAN, NAN, NAN), "se_mf": NAN, "far": False,
                            "reason": "missing run"})
            continue
        run_f = ctx.run(k2(e, s, nf))
        labels = L.series_axis_labels(qn, run_f.planes, run_f.times)
        for i in range(len(labels)):
            qc, qm = pcm[qn]["a"][i], pcm[qn]["b"][i]
            qf = pmf[qn]["b"][i]
            cell = L.v2_cell(qc, qm, qf, pmf[qn]["se"][i], pcm[qn]["se"][i])
            isfar = (far is not None and L.SERIES_KIND[qn] == "plane" and i >= far)
            out.append({"series": qn, "i": i, "label": labels[i], "q_c": qc, "q_m": qm, "q_f": qf, "cell": cell,
                        "se_mf": pmf[qn]["se"][i], "far": isfar, "reason": ""})
    return out


def do_v2(ctx):
    D = ctx.D
    cells = []
    for e in D.eps_codes:
        for s in D.seeds:
            for c in v2_block(ctx, L.PLANE_SERIES, e, s):
                c.update({"eps_code": e, "seed": s})
                cells.append(c)
    return cells


def cand_status(ctx, e, n, series_names, planes_only=True):
    """Status of one lambda/h candidate N at amplitude e: missing / pass / fail, with its cells."""
    D = ctx.D
    rows = []
    missing = False
    for s in D.seeds:
        pr = ctx.eng.pair(k2(e, s, n), k2(e, s, D.n_ref))
        if pr is None:
            missing = True
            continue
        run_f = ctx.run(k2(e, s, D.n_ref))
        for qn in series_names:
            labels = L.series_axis_labels(qn, run_f.planes, run_f.times)
            for i in range(len(labels)):
                cc = L.lambda_h_cell(pr[qn]["a"][i], pr[qn]["b"][i], pr[qn]["se"][i])
                rows.append((s, qn, i, labels[i], pr[qn]["a"][i], pr[qn]["b"][i], cc))
    if missing:
        return "missing", rows
    ok = all(r[6]["verdict"] == "pass" for r in rows) and len(rows) > 0
    return ("pass" if ok else "fail"), rows


def do_lambda_h(ctx, series_names):
    """-> per-eps dict: {cand: (status, rows)}, production values per eps and overall."""
    D = ctx.D
    res = {}
    prod = {}
    cands = [c[1] for c in D.candidates]
    for e in D.eps_codes:
        res[e] = {}
        for (n, lh) in D.candidates:
            res[e][lh] = (n,) + cand_status(ctx, e, n, series_names)
        st = {lh: res[e][lh][1] for lh in cands}
        prod[e] = L.production_lambda_h(cands, st)
    return res, prod


def study_value(prod, which):
    vals = [prod[e][which] for e in prod]
    if not vals or any(v is None for v in vals):
        return None
    return max(vals)


# -------------------------------------------------------------------------------------------------------------
# main digest
# -------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="SF-37 V1/V2/lambda-h digest")
    ap.add_argument("root")
    ap.add_argument("outdir")
    ap.add_argument("--design", choices=("full", "discovered"), default="full")
    args = ap.parse_args(argv)
    argv_text = "digest_v1v2.py %s %s --design %s" % (args.root, args.outdir, args.design)

    manifest = L.Manifest(args.root)
    camp = L.Campaign(args.root, manifest)
    idx16 = L.load_bootstrap(manifest=manifest)
    design = L.Design.full() if args.design == "full" else L.Design.discovered(camp)
    eng = L.PairEngine(camp, idx16)
    ctx = Ctx(camp, design, eng)
    W = L.OutWriter(args.outdir, manifest, "digest_v1v2", argv_text)
    md = W.md_header("SF-37 digest V1 / V2 / production lambda/h / D1-D3, D8, D9",
                     ["design: `%s` (eps codes %s, seeds %s, v2 N %s, v1 N %s, N_ref %d)" %
                      (design.kind, list(design.eps_codes), list(design.seeds), list(design.v2_ns),
                       list(design.v1_ns), design.n_ref)])

    # ---- completeness
    exp = design.expected_runs()
    comp_rows, missing = [], []
    mism = []
    for part, name in exp:
        r = camp.run(part, name)
        code = camp.exit_codes(part).get(name)
        comp_rows.append([part, name, "present" if r else "missing", "" if code is None else code])
        if r is None:
            missing.append((part, name, code))
        elif design.kind == "full":
            if not (np.array_equal(r.planes, np.array(design.planes)) and np.array_equal(r.times, np.array(design.times))):
                mism.append(name)
    W.csv("completeness.csv", ["part", "run", "status", "exit_code"], comp_rows)
    md += ["", "## Completeness", "",
           "Expected runs: %d; present: %d; missing: %d. Missing runs make every dependent cell `not_evaluable` "
           "(never `pass`)." % (len(exp), len(exp) - len(missing), len(missing))]
    if missing:
        md.append("")
        md.append(L.md_table(["part", "run", "exit code"], [(m[0], m[1], "" if m[2] is None else m[2]) for m in missing[:200]]))
        if len(missing) > 200:
            md.append("... %d more in completeness.csv" % (len(missing) - 200))
    if mism:
        md += ["", "Runs whose planes/times differ from the pre-registered lists: %s" % ", ".join(mism)]
    bad_codes = []
    for part in ("v2", "v1"):
        for name, code in sorted(camp.exit_codes(part).items()):
            if code != 0:
                bad_codes.append((part, name, code))
    if bad_codes:
        md += ["", "Runs with a non-zero exit code (runs with exit 22/23 still contribute their contributors):", "",
               L.md_table(["part", "run", "exit"], bad_codes)]

    # ---- V1
    v1cells, ladder, vac, shadow, ident_rows = do_v1(ctx)
    hdr = ["eps_code", "sigma2", "seed", "N", "plane", "q_1e-8", "q_1e-10", "D", "se_D", "z", "thr_rel", "thr_2se",
           "binding", "branch", "verdict", "reason", "ms_over_sigma_T2", "rms_over_sigma_T", "far", "stat2se_over_qref"]
    W.csv("v1_cells.csv", hdr, [[c[h] for h in hdr] for c in v1cells])
    W.csv("v1_ladder.csv", ["eps_code", "seed", "N", "pair_kind", "test", "reference", "series", "plane", "q_test",
                            "q_ref", "D", "se_D", "z", "v1_branch", "v1_verdict", "ms_over_sigma_T2",
                            "rms_over_sigma_T", "first_plane_rms_gt_0.1_sigma_T"], ladder)
    W.csv("rr15_vacuity.csv", ["eps_code", "seed", "N", "setting", "plane_nfev", "plane_accepted", "plane_rejected",
                               "plane_landing", "plane_landing_rejected", "time_nfev", "time_accepted",
                               "time_rejected", "time_landing", "time_landing_rejected",
                               "ident_weight_fraction_vs_1e-10_h8", "n_ident", "Np", "statuses_identical",
                               "first_plane_D_nonzero"], vac)
    W.csv("rr15_shadow.csv", ["eps_code", "seed", "N", "pair", "v1_inequality", "n_fail", "max_ms_over_sigma_T2",
                              "first_plane_rms_gt_0.1_sigma_T", "reason"], shadow)

    md += ["", "## V1 - tolerance (source B)", "",
           "Rule: at each N, amplitude, seed and plane, `|sigma_T^2(1e-8) - sigma_T^2(1e-10)| <= max(1e-3 "
           "|sigma_T^2(1e-10)|, 2 SE_D)` on the h/8 runs. Branch: `det` = relative branch, `stat` = 2 SE_D branch, "
           "`fail`. `far` = plane at or after the first plane with RMS(dx_T) > 0.1 sigma_T for the pair.", ""]
    summ = []
    for e in design.eps_codes:
        for n in design.v1_ns:
            for s in design.seeds:
                sub = [c for c in v1cells if c["eps_code"] == e and c["N"] == n and c["seed"] == s]
                ident = [x[3] for x in ident_rows if x[0] == e and x[1] == s and x[2] == n]
                lab = v1_label(sub, ident[0] if ident else None)
                nf = sum(1 for c in sub if c["verdict"] == "fail")
                ne = sum(1 for c in sub if c["verdict"] == "not_evaluable")
                far_cells = [c for c in sub if c["far"]]
                stat_far = bool(far_cells) and all(c["branch"] == "stat" for c in far_cells)
                summ.append([L.sigma2_of(e), n, s, len(sub), nf, ne, lab, "yes" if stat_far else "no"])
    md += [L.md_table(["sigma2", "N", "seed", "cells", "fail", "not_evaluable", "V1 outcome (RR15 label)",
                       "passes only via 2 SE_D at every far plane"], summ), ""]
    amp_rows = []
    for e in design.eps_codes:
        sub = [c for c in v1cells if c["eps_code"] == e]
        nf = sum(1 for c in sub if c["verdict"] == "fail")
        ne = sum(1 for c in sub if c["verdict"] == "not_evaluable")
        vv = "fail" if nf else ("not_evaluable" if ne or not sub else "pass")
        labs = [r[6] for r in summ if r[0] == L.sigma2_of(e)]
        vac_all = bool(labs) and all(x.startswith("met, vacuous") for x in labs)
        amp_rows.append([L.sigma2_of(e), len(sub), nf, ne, vv + (" (vacuous: tol-independent step sequence)" if vv == "pass" and vac_all else "")])
    md += ["### V1 per amplitude", "", L.md_table(["sigma2", "cells", "fail", "not_evaluable", "verdict"], amp_rows), "",
           "### V1 cells (verdict and branch for every cell)", ""]
    md.append(L.md_table(["sigma2", "seed", "N", "plane", "Q(1e-8)", "Q(1e-10)", "D", "SE_D", "z", "rel thr", "2SE_D",
                          "branch", "verdict", "MS/sigma_T2", "far"],
                         [[L.sigma2_of(c["eps_code"]), c["seed"], c["N"], c["plane"], c["q_1e-8"], c["q_1e-10"], c["D"],
                           c["se_D"], c["z"], c["thr_rel"], c["thr_2se"], c["branch"], c["verdict"],
                           c["ms_over_sigma_T2"], c["far"]] for c in v1cells]))
    md += ["", "### RR15 shadow classifications (recorded, not blocking)", "",
           L.md_table(["eps_code", "seed", "N", "pair", "V1 inequality", "fail planes", "max MS/sigma_T2",
                       "first plane RMS>0.1 sigma_T", "note"], shadow), "",
           "Counters, bitwise identity against (h/8, 1e-10) and the first plane with D != 0 per setting: "
           "`rr15_vacuity.csv`; the whole tolerance / h_max ladder: `v1_ladder.csv`."]

    # ---- V2
    v2cells = do_v2(ctx)
    hdr2 = ["eps_code", "sigma2", "seed", "series", "plane", "q_c", "q_m", "q_f", "D_mf", "se_mf", "p", "z", "thr_rel",
            "thr_2se", "order_ok", "det_ok", "stat_ok", "branch", "verdict", "richardson", "order_from_noise_flag", "far"]
    rows2 = []
    for c in v2cells:
        ce = c["cell"]
        rows2.append([c["eps_code"], L.sigma2_of(c["eps_code"]), c["seed"], c["series"], c["label"], c["q_c"], c["q_m"],
                      c["q_f"], ce["D"], c["se_mf"], ce["p"], ce["z"], ce["thr_rel"], ce["thr_stat"], ce["order_ok"],
                      ce["det_ok"], ce["stat_ok"], ce["branch"], ce["verdict"], ce["richardson"], ce["noise_flag"],
                      c["far"]])
    W.csv("v2_cells.csv", hdr2, rows2)
    nc, nm, nf_ = design.n_triple
    md += ["", "## V2 - grid (source A)", "",
           "Rule (triple %d -> %d -> %d): observed order >= 1.5, or `|Q(%d) - Q(%d)| <= max(0.02 |Q(%d)|, 2 SE_D)`, "
           "for `Q` in {sigma_T^2, var tau}. Branch = first satisfied of `det` (relative), `order`, `stat` (2 SE_D); "
           "all three booleans are in `v2_cells.csv`." % (nc, nm, nf_, nm, nf_, nf_), ""]
    amp2 = []
    for e in design.eps_codes:
        sub = [c for c in v2cells if c["eps_code"] == e]
        nfl = sum(1 for c in sub if c["cell"]["verdict"] == "fail")
        nne = sum(1 for c in sub if c["cell"]["verdict"] == "not_evaluable")
        vv = "fail" if nfl else ("not_evaluable" if nne or not sub else "pass")
        statonly = []
        for s in design.seeds:
            for qn in L.PLANE_SERIES:
                fc = [c for c in sub if c["seed"] == s and c["series"] == qn and c["far"]]
                statonly.append(bool(fc) and all(c["cell"]["branch"] == "stat" for c in fc))
        amp2.append([L.sigma2_of(e), len(sub), nfl, nne, vv, "%d / %d" % (sum(statonly), len(statonly))])
    md += [L.md_table(["sigma2", "cells", "fail", "not_evaluable", "V2 verdict", "(seed, Q) passing only via 2 SE_D at every far plane"], amp2),
           "", "### V2 cells (verdict and branch for every cell)", ""]
    md.append(L.md_table(["sigma2", "seed", "Q", "plane", "Q(%d)" % nc, "Q(%d)" % nm, "Q(%d)" % nf_, "order p", "|D|", "rel thr",
                          "2SE_D", "branch", "verdict", "Richardson res.", "order from noise"],
                         [[r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[10], abs(r[8]) if r[8] == r[8] else r[8], r[12], r[13],
                           r[17], r[18], r[19], r[20]] for r in rows2]))

    # ---- production lambda/h
    lh_series = L.PLANE_SERIES
    lres, lprod = do_lambda_h(ctx, lh_series)
    lrows, lsum = [], []
    for e in design.eps_codes:
        for lh, (n, status, rows) in lres[e].items():
            nfail = sum(1 for r in rows if r[6]["verdict"] != "pass")
            worst = NAN
            for r in rows:
                cc = r[6]
                if cc["verdict"] != "not_evaluable":
                    den = max(cc["thr_rel"], cc["thr_stat"])
                    ratio = abs(cc["D"]) / den if den > 0 else (0.0 if cc["D"] == 0 else float("inf"))
                    worst = ratio if not (worst == worst) else max(worst, ratio)
                lrows.append([e, L.sigma2_of(e), r[0], n, lh, r[1], r[3], r[4], r[5], cc["D"], cc["thr_rel"], cc["thr_stat"],
                              cc["z"], cc["branch"], cc["verdict"]])
            lsum.append([L.sigma2_of(e), n, lh, status, len(rows), nfail, worst])
    W.csv("lambda_h_cells.csv", ["eps_code", "sigma2", "seed", "N", "lambda_over_h", "series", "plane", "q_N", "q_ref",
                                 "D", "thr_rel", "thr_2se", "z", "branch", "verdict"], lrows)
    md += ["", "## Production lambda/h (RR5)", "",
           "Candidate passes iff `|Q(N) - Q(%d)| <= max(0.05 |Q(%d)|, 2 SE_D(N, %d))` at every plane, Q and seed "
           "(contributors). `missing` = a needed run has no data. Primary = smallest passing candidate (literal "
           "spec); monotone = smallest candidate such that it and every larger candidate pass." %
           (design.n_ref, design.n_ref, design.n_ref), "",
           L.md_table(["sigma2", "N", "lambda/h", "status", "cells", "cells not passing", "worst abs(D)/max(thr)"], lsum), ""]
    prod_rows = []
    for e in design.eps_codes:
        p, m = lprod[e]
        prod_rows.append([L.sigma2_of(e), "none <= %g" % design.candidates[-1][1] if (p is None and design.candidates) else p,
                          "none <= %g" % design.candidates[-1][1] if (m is None and design.candidates) else m,
                          "DIFFERENT" if p != m else "same"])
    sp, sm = study_value(lprod, 0), study_value(lprod, 1)
    lastc = design.candidates[-1][1] if design.candidates else NAN
    md += [L.md_table(["sigma2", "primary (literal)", "monotone variant", "primary vs monotone"], prod_rows), "",
           "**Study value (max over amplitudes of the primary): %s**; monotone variant: %s%s." %
           ("none <= %g" % lastc if sp is None else "%g" % sp, "none <= %g" % lastc if sm is None else "%g" % sm,
            "; the two DIFFER (report to the owner)" if sp != sm else ""),
           "", "Per-cell compact view (branch per plane: d=det, s=stat, F=fail, ?=not evaluable):", ""]
    comp = []
    for e in design.eps_codes:
        for lh, (n, status, rows) in lres[e].items():
            for s in design.seeds:
                for qn in lh_series:
                    br = "".join(BR[r[6]["branch"]] for r in rows if r[0] == s and r[1] == qn)
                    comp.append([L.sigma2_of(e), n, lh, s, qn, status if not br else br])
    md.append(L.md_table(["sigma2", "N", "lambda/h", "seed", "Q", "branches by plane"], comp))

    # ---- D1
    d1rows = []

    def add_d1(crit, e, s, key, qn, plane, D_, se_, qref, branch, verdict, thr_rel, thr_stat, z_):
        d1rows.append([crit, e, s, key, qn, plane, D_, se_, z_, thr_rel, thr_stat,
                       (thr_stat / abs(qref)) if (qref and qref == qref and thr_stat == thr_stat) else NAN,
                       bool(thr_stat > thr_rel) if (thr_stat == thr_stat and thr_rel == thr_rel) else False, branch, verdict])

    for c in v1cells:
        add_d1("V1", c["eps_code"], c["seed"], "N%d" % c["N"], "sigma_T2", c["plane"], c["D"], c["se_D"], c["q_1e-10"],
               c["branch"], c["verdict"], c["thr_rel"], c["thr_2se"], c["z"])
    for c in v2cells:
        ce = c["cell"]
        add_d1("V2", c["eps_code"], c["seed"], "%d-%d" % (nm, nf_), c["series"], c["label"], ce["D"], c["se_mf"], c["q_f"],
               ce["branch"], ce["verdict"], ce["thr_rel"], ce["thr_stat"], ce["z"])
    for r in lrows:
        add_d1("lambda_h", r[0], r[2], "N%d" % r[3], r[5], r[6], r[9], NAN, r[8], r[13], r[14], r[10], r[11], r[12])
    # seed-pooled z
    groups = {}
    for r in d1rows:
        groups.setdefault((r[0], r[1], r[3], r[4], r[5]), []).append(r[8])
    d1out = []
    for r in d1rows:
        zs = [z for z in groups[(r[0], r[1], r[3], r[4], r[5])] if z == z and math.isfinite(z)]
        pooled = sum(zs) / math.sqrt(len(zs)) if zs else NAN
        d1out.append(r + [pooled])
    W.csv("d1_cells.csv", ["criterion", "eps_code", "seed", "key", "series", "plane", "D", "se_D", "z", "thr_rel",
                           "thr_2se", "thr_2se_over_abs_qref", "stat_regime", "branch", "verdict", "seed_pooled_z"], d1out)
    from scipy import stats as sps
    pt = 2.0 * (1.0 - sps.t.cdf(2.0, 15))
    d1s = []
    for crit in ("V1", "V2", "lambda_h"):
        sub = [r for r in d1out if r[0] == crit]
        nstat = sum(1 for r in sub if bool(r[12]))
        nfail = sum(1 for r in sub if r[14] == "fail")
        nfail_stat = sum(1 for r in sub if r[14] == "fail" and bool(r[12]))
        d1s.append([crit, len(sub), nstat, "%.3g" % pt, "%.2f / %.2f / %.2f" % (nstat * 0.05, nstat * pt, nstat * 0.065),
                    nfail, nfail_stat])
    md += ["", "## D1 - multiplicity (recorded, no thresholds)", "",
           "Cells in the statistical regime = cells with 2 SE_D greater than the relative threshold. Expected chance "
           "exceedances per statistical-regime cell: 2 (1 - t_cdf(2; 15 dof)) = %.4f (range 5 %% .. 6.5 %% shown)." % pt, "",
           L.md_table(["criterion", "cells", "statistical-regime cells", "t15 exceedance prob.",
                       "expected chance exceedances (5% / t15 / 6.5%)", "failing cells", "failing cells in stat regime"], d1s), ""]
    # sign agreement
    fail_rows = [r for r in d1out if r[14] == "fail" and r[6] == r[6]]
    by_plane = {}
    for r in fail_rows:
        by_plane.setdefault((r[0], r[1], r[3], r[4], r[5]), []).append(math.copysign(1, r[6]))
    tot_g = sum(1 for v in by_plane.values() if len(v) >= 2)
    agree_g = sum(1 for v in by_plane.values() if len(v) >= 2 and len(set(v)) == 1)
    by_seed = {}
    for r in d1out:
        by_seed.setdefault((r[0], r[1], r[2], r[3], r[4]), []).append(r)
    tot_a = agree_a = 0
    for lst in by_seed.values():
        for i in range(len(lst) - 1):
            if lst[i][14] == "fail" and lst[i + 1][14] == "fail" and lst[i][6] == lst[i][6] and lst[i + 1][6] == lst[i + 1][6]:
                tot_a += 1
                agree_a += int(math.copysign(1, lst[i][6]) == math.copysign(1, lst[i + 1][6]))
    md += ["Sign agreement of failing cells: across seeds (same criterion, amplitude, N, Q, plane; >= 2 failing seeds) "
           "%d of %d groups agree; across adjacent planes both failing (same seed) %d of %d pairs agree. "
           "(%d failing cells in total.)" % (agree_g, tot_g, agree_a, tot_a, len(fail_rows)), ""]
    # SE sensitivity and ESS
    sens, ess_rows = [], []
    for e in design.eps_codes:
        for s in design.seeds:
            for n in design.v1_ns:
                pr16 = ctx.eng.pair(k1(e, s, n, L.TOL_PRIMARY, 8), k1(e, s, n, L.TOL_REF, 8), 4)
                if pr16 is None:
                    continue
                r4 = ctx.eng.pair(k1(e, s, n, L.TOL_PRIMARY, 8), k1(e, s, n, L.TOL_REF, 8), 2)
                r64 = ctx.eng.pair(k1(e, s, n, L.TOL_PRIMARY, 8), k1(e, s, n, L.TOL_REF, 8), 8)
                for m, pl in enumerate(ctx.run(k1(e, s, n, L.TOL_REF, 8)).planes):
                    sens.append(["V1", e, s, "N%d" % n, "sigma_T2", pl, pr16["sigma_T2"]["se"][m], r4["sigma_T2"]["se"][m],
                                 r64["sigma_T2"]["se"][m]])
            pr16 = ctx.eng.pair(k2(e, s, design.n_triple[1]), k2(e, s, design.n_triple[2]), 4)
            if pr16 is not None:
                r4 = ctx.eng.pair(k2(e, s, design.n_triple[1]), k2(e, s, design.n_triple[2]), 2)
                r64 = ctx.eng.pair(k2(e, s, design.n_triple[1]), k2(e, s, design.n_triple[2]), 8)
                rf = ctx.run(k2(e, s, design.n_triple[2]))
                for qn in L.PLANE_SERIES:
                    for m, pl in enumerate(rf.planes):
                        sens.append(["V2", e, s, "%d-%d" % design.n_triple[1:], qn, pl, pr16[qn]["se"][m],
                                     r4[qn]["se"][m], r64[qn]["se"][m]])
            for key in [k2(e, s, n) for n in design.v2_ns] + [k1(e, s, n, t, 8) for n in design.v1_ns for t in (L.TOL_PRIMARY, L.TOL_REF)]:
                r = ctx.run(key)
                if r is None:
                    continue
                mp = r.plane_mask()
                qs = L.observed_series(r)
                pcore = L._plane_core(r)
                for m, pl in enumerate(r.planes):
                    ess_rows.append([key[1], pl, L.effective_sample_size(r.w[mp[:, m]]), int(mp[:, m].sum()),
                                     pcore["mean_tau"][m], pcore["var_tau"][m], qs["sigma_T2"][m]])
    W.csv("d1_se_sensitivity.csv", ["criterion", "eps_code", "seed", "key", "series", "plane", "se_16_blocks",
                                    "se_4_blocks", "se_64_blocks"], sens)
    W.csv("d1_ess.csv", ["run", "plane", "ess", "n_contributors", "mean_tau", "var_tau", "sigma_T2"], ess_rows)
    ratios = [(x[7] / x[6], x[8] / x[6]) for x in sens if x[6] and x[6] == x[6] and x[6] > 0]
    if ratios:
        r4m = np.nanmedian([r[0] for r in ratios])
        r64m = np.nanmedian([r[1] for r in ratios])
        md += ["SE_D sensitivity (median over %d cells of SE_D(4 blocks)/SE_D(16) and SE_D(64 blocks)/SE_D(16)): %.3g, %.3g "
               "(detail `d1_se_sensitivity.csv`; effective sample sizes `d1_ess.csv`; seed-pooled z in `d1_cells.csv`)." %
               (len(ratios), r4m, r64m), ""]

    # ---- D2
    d2 = []
    for (part, name) in exp:
        txt = camp.log_text(part, name)
        r = camp.run(part, name)
        info = L.parse_field_line(txt) if txt else None
        nfev = NAN
        if r is not None:
            trk = r.track()
            cp = L.pass_counters(trk, "plane_pass")
            if cp["nfev"] is not None and len(trk["planes"]):
                nfev = cp["nfev"] / (r.Np * trk["planes"][-1])
        pn = L.parse_run_name(name)
        if info is None:
            d2.append([name, pn[2], pn[3], NAN, NAN, NAN, nfev])
        else:
            d2.append([name, pn[2], pn[3], info.get("applied_scale", NAN), info.get("raw_variance", NAN),
                       info.get("final_variance", NAN), nfev])
    W.csv("d2_pairing.csv", ["run", "seed", "N", "applied_scale", "raw_variance", "final_variance",
                             "nfev_per_particle_per_unit_length"], d2)
    spread = []
    for seed in design.seeds:
        scs = [r[3] for r in d2 if r[1] == seed and r[3] == r[3]]
        rvs = [r[4] for r in d2 if r[1] == seed and r[4] == r[4]]
        if scs:
            ds = max(abs(x - scs[0]) / abs(scs[0]) for x in scs) if scs[0] else NAN
            dr = max(abs(x - rvs[0]) / abs(rvs[0]) for x in rvs) if rvs and rvs[0] else NAN
            spread.append([seed, len(scs), ds, dr, "equal <= 1e-12" if (ds <= 1e-12 and dr <= 1e-12) else "DIFFERENT"])
        else:
            spread.append([seed, 0, NAN, NAN, "no FIELD line found"])
    nf_rows = []
    for n in design.v2_ns:
        vv = [r[6] for r in d2 if r[2] == n and r[0].startswith("v2_") and r[6] == r[6]]
        nf_rows.append([n, len(vv), float(np.mean(vv)) if vv else NAN])
    md += ["", "## D2 - pairing (recorded)", "",
           "FIELD-line `applied_scale` / `raw_variance` across all runs of a seed (relative spread):", "",
           L.md_table(["seed", "runs with a FIELD line", "applied_scale rel. spread", "raw_variance rel. spread", "equal?"], spread),
           "", "Plane-pass field evaluations per particle per unit length (mean over v2 runs):", "",
           L.md_table(["N", "runs", "nfev / particle / unit length"], nf_rows), ""]

    # ---- D3
    d3 = []
    for e in design.eps_codes:
        for s in design.seeds:
            pgrid = ctx.eng.pair(k2(e, s, design.n_triple[1]), k2(e, s, design.n_triple[2]))
            rf = ctx.run(k2(e, s, design.n_triple[2]))
            for n in design.v1_ns:
                p1 = ctx.eng.pair(k1(e, s, n, L.TOL_PRIMARY, 8), k1(e, s, n, L.TOL_REF, 32))
                for qn in L.PLANE_SERIES:
                    for m, pl in enumerate(rf.planes if rf else design.planes):
                        a1 = abs(p1[qn]["D"][m]) if p1 else NAN
                        a2 = abs(pgrid[qn]["D"][m]) if pgrid else NAN
                        d3.append([e, s, n, qn, pl, a1, a2, a1 / a2 if (a2 and a2 == a2) else NAN])
    W.csv("d3_attribution.csv", ["eps_code", "seed", "N", "series", "plane", "abs_Q(h8,1e-8)-Q(h32,1e-10)",
                                 "abs_Q(256)-Q(512)", "ratio"], d3)
    d3ref = []
    nm_, nf2 = design.n_triple[1], design.n_triple[2]
    for e in design.eps_codes:
        for s in design.seeds:
            pr = ctx.eng.pair(k1(e, s, nm_, L.TOL_REF, 32), k1(e, s, nf2, L.TOL_REF, 32))
            rr = ctx.run(k1(e, s, nf2, L.TOL_REF, 32))
            for qn in L.PLANE_SERIES:
                for m, pl in enumerate(rr.planes if rr else design.planes):
                    if pr is None:
                        d3ref.append([e, s, qn, pl, NAN, NAN, NAN, NAN, "not_evaluable", "not_evaluable"])
                        continue
                    cc = L.lambda_h_cell(pr[qn]["a"][m], pr[qn]["b"][m], pr[qn]["se"][m], rel=L.V2_REL)
                    d3ref.append([e, s, qn, pl, pr[qn]["a"][m], pr[qn]["b"][m], pr[qn]["D"][m], pr[qn]["se"][m],
                                  cc["branch"], cc["verdict"]])
    W.csv("d3_v2_on_reference_runs.csv", ["eps_code", "seed", "series", "plane", "q_256", "q_512", "D", "se_D", "branch",
                                          "verdict_shadow_inequality"], d3ref)
    nshadow = collections_count([r[9] for r in d3ref])
    md += ["", "## D3 - attribution (recorded)", "",
           "Tolerance/h_max effect `|Q(h/8,1e-8) - Q(h/32,1e-10)|` against the grid effect `|Q(%d) - Q(%d)|`: `d3_attribution.csv`. "
           "V2 inequality (2 %% / 2 SE_D) on the reference runs (h/32, 1e-10) at N %d / %d (shadow, no triple): %s "
           "(`d3_v2_on_reference_runs.csv`). The passive-I check is in `digest_v4v5.py`." % (nm_, nf2, nm_, nf2, nshadow), ""]

    # ---- D8
    d8_series = L.TIME_SERIES + L.DPLANE_SERIES + L.DTIME_SERIES
    d8v1, d8v2 = [], []
    for e in design.eps_codes:
        for s in design.seeds:
            for n in design.v1_ns:
                pr = ctx.eng.pair(k1(e, s, n, L.TOL_PRIMARY, 8), k1(e, s, n, L.TOL_REF, 8))
                rr = ctx.run(k1(e, s, n, L.TOL_REF, 8))
                for qn in d8_series:
                    labels = L.series_axis_labels(qn, rr.planes if rr else design.planes, rr.times if rr else design.times)
                    for i, lab in enumerate(labels):
                        if pr is None:
                            c = blank_v1("missing run")
                            vals = (NAN, NAN, NAN)
                        else:
                            c = L.v1_cell(pr[qn]["D"][i], pr[qn]["b"][i], pr[qn]["se"][i])
                            vals = (pr[qn]["D"][i], pr[qn]["b"][i], pr[qn]["se"][i])
                        d8v1.append([e, s, n, qn, lab, vals[1], vals[0], vals[2], c["z"], c["branch"], c["verdict"]])
        for s in design.seeds:
            for c in v2_block(ctx, d8_series, e, s):
                ce = c["cell"]
                d8v2.append([e, s, c["series"], c["label"], c["q_c"], c["q_m"], c["q_f"], ce["D"], c["se_mf"], ce["p"],
                             ce["branch"], ce["verdict"]])
    W.csv("d8_shadow_v1.csv", ["eps_code", "seed", "N", "series", "axis", "q_1e-10", "D", "se_D", "z", "branch", "verdict"], d8v1)
    W.csv("d8_shadow_v2.csv", ["eps_code", "seed", "series", "axis", "q_c", "q_m", "q_f", "D_mf", "se_mf", "p", "branch",
                               "verdict"], d8v2)
    d8lh_rows, d8lh_sum = [], []
    for qn in d8_series:
        res, prod = do_lambda_h(ctx, (qn,))
        for e in design.eps_codes:
            for lh, (n, status, rows) in res[e].items():
                d8lh_rows.append([e, qn, n, lh, status, sum(1 for r in rows if r[6]["verdict"] != "pass"), len(rows)])
            p, m_ = prod[e]
            d8lh_sum.append([L.sigma2_of(e), qn, "none" if p is None else p, "none" if m_ is None else m_])
    W.csv("d8_shadow_lambda_h.csv", ["eps_code", "series", "N", "lambda_over_h", "status", "cells_not_passing", "cells"], d8lh_rows)
    # budgets
    bud = []
    for e in design.eps_codes:
        for s in design.seeds:
            for n in design.v1_ns:
                kr, kh = k1(e, s, n, L.TOL_REF, 8), k1(e, s, n, L.TOL_REF, 32)
                specs = [("tol", k1(e, s, n, t, 8), kr, "tol%s_h8" % L.fmt_tol(t)) for t in design.v1_tols if t != L.TOL_REF]
                specs.append(("h_max", kr, kh, "tol1e-10_h8_vs_h32"))
                specs.append(("tol+h_max", k1(e, s, n, L.TOL_PRIMARY, 8), kh, "tol1e-8_h8_vs_ref"))
                for kind, ka, kb, lab in specs:
                    a_, b_ = ctx.run(ka), ctx.run(kb)
                    if a_ is None or b_ is None:
                        continue
                    for ax, kd in (("plane", "plane"), ("time", "time")):
                        sp = L.pair_separation(a_, b_, kd)
                        xs = b_.planes if kd == "plane" else b_.times
                        for i, x in enumerate(xs):
                            bud.append([kind, lab, e, s, n, ax, x, sp["rel"][i]])
            for n in design.v2_ns:
                n2 = 2 * n
                if n2 in design.v2_ns:
                    a_, b_ = ctx.run(k2(e, s, n)), ctx.run(k2(e, s, n2))
                    if a_ is None or b_ is None:
                        continue
                    for kd in ("plane", "time"):
                        sp = L.pair_separation(a_, b_, kd)
                        xs = b_.planes if kd == "plane" else b_.times
                        for i, x in enumerate(xs):
                            bud.append(["grid", "N%d_vs_N%d" % (n, n2), e, s, n, kd, x, sp["rel"][i]])
    W.csv("d8_budget.csv", ["kind", "pair", "eps_code", "seed", "N", "axis", "x", "rms_dxT_over_sigma_T"], bud)
    grow = []
    for (part, name) in exp:
        r = camp.run(part, name)
        if r is None:
            continue
        o = L.observed_series(r)
        for qn, xs in (("sigma_T2", r.planes), ("var_tau", r.planes), ("var_x1", r.times), ("var_x2", r.times),
                       ("var_x3", r.times)):
            v = o[qn]
            for i in range(len(xs)):
                for j in range(i + 1, len(xs)):
                    if abs(xs[j] / xs[i] - 2.0) < 1e-12 and v[i] > 0 and v[j] > 0:
                        grow.append([name, qn, xs[i], xs[j], math.log(v[j] / v[i]) / math.log(2.0)])
    W.csv("d8_growth.csv", ["run", "series", "x_from", "x_to", "local_growth_exponent"], grow)
    c8 = {}
    for r in d8v1:
        c8.setdefault(r[3], collections_counter()).update([r[10]])
    c8b = {}
    for r in d8v2:
        c8b.setdefault(r[2], collections_counter()).update([r[11]])
    md += ["", "## D8 - shadow classifications on the SF-39 estimator quantities (recorded, not blocking)", "",
           "Series: var x_k(t), Delta var_k / Delta t between consecutive times, Delta sigma_T^2 / Delta x1 between "
           "consecutive planes. V1 (N %s) and V2 verdict counts per series:" % list(design.v1_ns), "",
           L.md_table(["series", "V1 pass", "V1 fail", "V1 n/e", "V2 pass", "V2 fail", "V2 n/e"],
                      [[qn, c8.get(qn, {}).get("pass", 0), c8.get(qn, {}).get("fail", 0), c8.get(qn, {}).get("not_evaluable", 0),
                        c8b.get(qn, {}).get("pass", 0), c8b.get(qn, {}).get("fail", 0), c8b.get(qn, {}).get("not_evaluable", 0)]
                       for qn in d8_series]), "",
           "Shadow production lambda/h per amplitude and series (primary, monotone):", "",
           L.md_table(["sigma2", "series", "primary", "monotone"], d8lh_sum), "",
           "Cells: `d8_shadow_v1.csv`, `d8_shadow_v2.csv`, `d8_shadow_lambda_h.csv`; RMS(dx_T)/sigma_T budgets for the tol, "
           "h_max and grid pairs: `d8_budget.csv`; local growth exponents: `d8_growth.csv`.", ""]

    # ---- D9
    nonc, cap, comm, bit = [], [], [], []
    for (part, name) in exp:
        r = camp.run(part, name)
        if r is None:
            continue
        wsum = r.w.sum()
        mp, mt = r.plane_mask(), r.time_mask()
        for m, pl in enumerate(r.planes):
            nonc.append([name, "plane", pl, 1.0 - float(r.w[mp[:, m]].sum() / wsum)])
        for m, tt in enumerate(r.times):
            nonc.append([name, "time", tt, 1.0 - float(r.w[mt[:, m]].sum() / wsum)])
        for which, f in (("plane", "status_plane.npy"), ("time", "status_time.npy")):
            st = r.mem(f)
            msk = st == 2
            cap.append([name, which, int(msk.sum()), float(r.w[msk].sum() / wsum)] + [int((st == k).sum()) for k in range(7)])
    W.csv("d9_noncontributing.csv", ["run", "axis", "x", "noncontributing_weight_fraction"], nonc)
    W.csv("d9_caphits.csv", ["run", "pass", "n_status2_cap", "weight_status2"] + ["n_status%d" % k for k in range(7)], cap)
    for e in design.eps_codes:
        for s in design.seeds:
            for n in design.v1_ns:
                ra, rb = ctx.run(k1(e, s, n, L.TOL_PRIMARY, 8)), ctx.run(k1(e, s, n, L.TOL_REF, 8))
                if ra is not None and rb is not None:
                    cp = ra.plane_mask() & rb.plane_mask()
                    ct = ra.time_mask() & rb.time_mask()
                    oa, ob = L.observed_series(ra, cp, ct), L.observed_series(rb, cp, ct)
                    pr = ctx.eng.pair(k1(e, s, n, L.TOL_PRIMARY, 8), k1(e, s, n, L.TOL_REF, 8))
                    for qn in L.PLANE_SERIES:
                        for m, pl in enumerate(rb.planes):
                            Dc = oa[qn][m] - ob[qn][m]
                            D0 = pr[qn]["D"][m]
                            comm.append(["V1", e, s, "N%d" % n, qn, pl, D0, Dc, abs(Dc - D0)])
                rv = ctx.run(k2(e, s, n))
                if ra is not None and rv is not None:
                    pos = all(np.array_equal(np.ascontiguousarray(ra.mem(f)).view("<i8"), np.ascontiguousarray(rv.mem(f)).view("<i8"))
                              for f in ("planes_x23.npy", "planes_tau.npy", "planes_I.npy", "times_x.npy", "times_I.npy"))
                    bit.append([e, s, n, pos, L.statuses_identical(ra, rv)])
                else:
                    bit.append([e, s, n, "not_evaluable", "not_evaluable"])
        for s in design.seeds:
            nc_, nm2, nf3 = design.n_triple
            ra, rb = ctx.run(k2(e, s, nm2)), ctx.run(k2(e, s, nf3))
            if ra is not None and rb is not None:
                cp = ra.plane_mask() & rb.plane_mask()
                ct = ra.time_mask() & rb.time_mask()
                oa, ob = L.observed_series(ra, cp, ct), L.observed_series(rb, cp, ct)
                pr = ctx.eng.pair(k2(e, s, nm2), k2(e, s, nf3))
                for qn in L.PLANE_SERIES:
                    for m, pl in enumerate(rb.planes):
                        Dc = oa[qn][m] - ob[qn][m]
                        D0 = pr[qn]["D"][m]
                        comm.append(["V2", e, s, "%d-%d" % (nm2, nf3), qn, pl, D0, Dc, abs(Dc - D0)])
    W.csv("d9_common_contributors.csv", ["criterion", "eps_code", "seed", "key", "series", "plane", "D_own_contributors",
                                         "D_common_contributors", "abs_change"], comm)
    W.csv("d9_bitwise_v1_vs_v2.csv", ["eps_code", "seed", "N", "records_bitwise_identical", "statuses_identical"], bit)
    wmax = max([r[3] for r in nonc], default=NAN)
    nbit_ok = sum(1 for r in bit if r[3] is True and r[4] is True)
    md += ["", "## D9 - contributors and reproducibility (recorded)", "",
           "Largest non-contributing weight fraction over all runs/planes/times: %s. Cap hits (status 2 = cap_exceeded) "
           "per run and pass: `d9_caphits.csv`. V1/V2 differences recomputed on common contributors: "
           "`d9_common_contributors.csv`. Bitwise identity of the V1 (1e-8, h/8, covectors on) run and the V2 primary "
           "run (positions, tau, I, statuses): %d of %d (eps, seed, N) cases identical (`d9_bitwise_v1_vs_v2.csv`)." %
           (L.fnum(wmax), nbit_ok, len(bit)), ""]

    # ---- manifest
    W.md("v1v2_summary.md", "\n".join(md))
    nman, dig = W.write_manifest("input_manifest_v1v2.csv")
    with open(os.path.join(args.outdir, "v1v2_summary.md"), "a") as f:
        f.write("\n## Input manifest\n\n%d input files (path, sha256, bytes in `input_manifest_v1v2.csv`); sha256 of the "
                "manifest listing: `%s`.\n" % (nman, dig))
    print("digest_v1v2: wrote %d files to %s (missing runs: %d)" % (len(W.written), args.outdir, len(missing)))
    return 0


def collections_counter():
    import collections
    return collections.Counter()


def collections_count(items):
    c = collections_counter()
    c.update(items)
    return ", ".join("%s=%d" % (k, v) for k, v in sorted(c.items())) or "none"


if __name__ == "__main__":
    sys.exit(main())
