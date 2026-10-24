#!/usr/bin/env python3
"""SF-38 N9: POST-HOC (NOT pre-registered) readings of the medium A/B evidence, computed only from the imported raw evidence.

  posthoc_ab.py RAW_DIR OUT_DIR      reads RAW_DIR (= artifact `raw/`), writes OUT_DIR/posthoc_ab.md
  posthoc_ab.py --selftest           assert-based checks on hand-computed examples (no raw files)

Stdlib only; deterministic (a rerun on the same raw/ regenerates a byte-identical file). Nothing here changes a pre-registered verdict,
threshold or analysis file. No alpha statement: S_T and S_tau appear only as the rule's statistics. 1 MiB = 1048576 B.
Sections: P1 pairing efficiency, P2 point estimates vs margins, P3 paired boundary-layer profile, P4 mean-flow forcing and tilt,
P5 memory of medium (b) at the production grid, P6 recorded Pollock non_ok / x-exit evidence.
"""
import csv
import json
import math
import os
import re
import statistics as st
import sys

STATS = ("S_T", "S_tau", "V1", "theta_2", "theta_3")
AMPS = (("1", "ab_s1"), ("4", "ab_s4"))
DIST = (0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10, 20)
MIB = 1048576.0
MEM_LEAN_BPC = 68.57       # memory audit 2026-10-22 section 10 (estimate from a code count, not measured)
PROD_CELLS = 3328 * 384 * 384
CEILING_MIB = 30000.0


def read(path):
    with open(path, errors="replace") as f:
        return f.read()


def toks(line):
    return dict(re.findall(r"(\w+)=(\S+)", line))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def pearson(x, y):
    mx, my = st.mean(x), st.mean(y)
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def gmean(v):
    return math.exp(sum(math.log(a) for a in v) / len(v))


def thickness(pts, ok):
    """pts: [(distance, meanD, SE)] ascending in distance; smallest distance d such that every plane at distance >= d passes ok(). None if
    even the farthest plane fails. 0.0 if all pass."""
    last_bad = None
    for k, p in enumerate(pts):
        if not ok(p):
            last_bad = k
    if last_bad is None:
        return pts[0][0]
    return pts[last_bad + 1][0] if last_bad + 1 < len(pts) else None


def f4(x):
    return "%.4g" % x


# ---------------------------------------------------------------------------------------------------------------- loading
def load_amp(raw, d):
    base = os.path.join(raw, "sf38_campaign", "ab", d, "realizations")
    R = []
    for r in range(10):
        p = os.path.join(base, "r_%04d" % r)
        e1 = json.load(open(os.path.join(p, "e1.json")))
        pr = json.load(open(os.path.join(p, "primary.json")))
        prof = {}
        lay = {}
        for m in "ab":
            with open(os.path.join(p, "e1_%s.csv" % m)) as f:
                rows = list(csv.DictReader(f))
            pl = [x for x in rows if x["kind"] == "plane"]
            ly = [x for x in rows if x["kind"] == "layer"]
            prof[m] = ([float(x["x1"]) for x in pl], [float(x["u_var"]) for x in pl], float(pl[0]["u_mean"]))
            lay[m] = ([float(x["v_mean"]) for x in ly], [float(x["w_mean"]) for x in ly])
        R.append({"e1": e1, "pr": pr, "prof": prof, "lay": lay})
    return R


# ---------------------------------------------------------------------------------------------------------------- P1
def p1(data):
    out = ("## P1 Pairing efficiency (derived, post-hoc)\n\n"
           "Per amplitude and primary statistic, over the ten paired realizations (`primary.json`): `mean_a`, `mean_b`, sample sd (n-1) `sd_a`, `sd_b`, "
           "Pearson `corr(a,b)`, `sd(D)` with `D_r = P_a,r - P_b,r`, and `eff = sd(D) / sqrt(sd_a^2 + sd_b^2)` (1 = no gain from pairing, i.e. as for "
           "independent samples with equal variances; 0 = perfect pairing). Raw scale of each statistic (S_T in box units). `eff` is on the raw "
           "difference, not on the relative scale of the rule's delta-method CI.\n\n")
    rows = []
    for amp, _ in AMPS:
        for s in STATS:
            a = [r["pr"]["a"][s] for r in data[amp]]
            b = [r["pr"]["b"][s] for r in data[amp]]
            D = [x - y for x, y in zip(a, b)]
            sa, sb, sd = st.stdev(a), st.stdev(b), st.stdev(D)
            rows.append((amp, s, f4(st.mean(a)), f4(st.mean(b)), f4(sa), f4(sb), "%.3f" % pearson(a, b), f4(sd), "%.3f" % (sd / math.sqrt(sa * sa + sb * sb))))
    return out + table(("amp", "stat", "mean_a", "mean_b", "sd_a", "sd_b", "corr(a,b)", "sd(D)", "eff = sd(D)/sqrt(sd_a^2+sd_b^2)"), rows)


# ---------------------------------------------------------------------------------------------------------------- P2
def p2(raw):
    res = read(os.path.join(raw, "sf38_campaign", "ab", "results.txt"))
    S = [toks(l) for l in res.splitlines() if l.startswith("C_STAT ")]
    rows, nrows = [], []
    inc = outside = 0
    inc4 = outside4 = 0
    for t in S:
        est, mg = float(t["est"]), float(t["margin"])
        lo, hi = float(t["ci_lo"]), float(t["ci_hi"])
        out_ = abs(est) > mg
        rows.append((t["amp"], t["stat"], t["test"], f4(est), f4(mg), "yes" if out_ else "no", "[%s, %s]" % (f4(lo), f4(hi)), t["class"]))
        if t["class"] == "inconclusive":
            inc += 1
            outside += out_
            if t["amp"] == "4":
                inc4 += 1
                outside4 += out_
            if not out_:
                h10 = (hi - lo) / 2
                gap = mg - abs(est)
                nrows.append((t["amp"], t["stat"], f4(est), f4(mg), f4(h10), f4(gap), "%.0f" % (10 * (h10 / gap) ** 2)))
    out = ("## P2 Point estimates against the margins (derived from the `C_STAT` lines, post-hoc)\n\n"
           "`est` is `R = ln(Pbar_a/Pbar_b)` for `test=ratio` and the paired difference for `test=diff|floor`; `margin` on the same scale; `|est| > margin` "
           "flags a point estimate outside the equivalence window (the rule classifies by the 90 % CI, not by the point estimate).\n\n")
    out += table(("amp", "stat", "test", "est", "margin", "|est| > margin", "90 % CI", "class"), rows)
    out += ("\nPlain statement (derived): %d of the %d inconclusive statistics have their point estimate outside the margin (%d of %d at `sigma^2 = 4`; "
            "none at `sigma^2 = 1`: the single inconclusive statistic there has `|est| < margin`).\n\n" % (outside, inc, outside4, inc4))
    out += ("Rough sample size for the inconclusive statistics with `|est| < margin` (labelled rough: `N ~ 10 (h10 / (margin - |est|))^2` with `h10` the observed "
            "CI half-width at `N_R = 10`; assumes the point estimate and the sd stay constant and ignores the change of the t-quantile; it is the size at which "
            "the CI would just touch the margin, not a power calculation):\n\n")
    out += table(("amp", "stat", "est", "margin", "h10", "margin - |est|", "N ~ (rough)"), nrows)
    return out


# ---------------------------------------------------------------------------------------------------------------- P3
def p3(raw, data):
    res = read(os.path.join(raw, "sf38_campaign", "ab", "results.txt"))
    lam = {toks(l)["amp"]: float(toks(l)["lambda"]) for l in res.splitlines() if l.startswith("C_INPUT ")}
    out = ("## P3 Paired boundary-layer profile of `var_i(v1)/qbar^2` (derived, post-hoc)\n\n"
           "Per realization `r` and face plane `i` (`kind=plane` rows of `e1_{a,b}.csv`, `u_var`): `D_i,r = u_var_b,i / qbar_b^2 - u_var_a,i / qbar_a^2`, with `qbar` "
           "the `qbar` of `e1.json` of each medium (= `u_mean` of plane 0; the maximum |qbar - u_mean(plane 0)| over all runs and media is printed below), as in the "
           "note. The comparison is (b) minus (a) on the same field; (a), which has no condition at the `x` faces, is the control. Ensemble mean and SE (sd over "
           "the ten realizations / sqrt(10)) of `D_i` against the distance from the west face (`x1`) and from the east face (`L1 - x1`) in units of `lambda` "
           "(nearest plane). `M_a` = mean over the interior planes `x1 in [5 lambda, L1 - 5 lambda]` of the (a) ensemble-mean profile. Thickness readings, "
           "defined here (post-hoc): `d_1%` = smallest distance `d` such that `|mean D_i| <= 0.01 M_a` for every plane between `d` and `20 lambda` from that "
           "face; `d_2SE` = smallest `d` such that `|mean D_i| <= 2 SE_i` for every plane between `d` and `20 lambda` (`n/a` = the plane at `20 lambda` already fails). "
           "`viol 5-20` counts the planes between `5 lambda` and `20 lambda` violating the `2 SE` condition (a pure-noise reference is ~5 % of the planes). "
           "Decay ratio = geometric mean of `D(d+1)/D(d)`, `d = 1, 2, 3` (nearest planes at 1, 2, 3, 4 `lambda`; `n/a` if a mean `D` is not positive).\n\n")
    qdiff = 0.0
    summ = []
    for amp, _ in AMPS:
        L = lam[amp]
        R = data[amp]
        xs = R[0]["prof"]["a"][0]
        n = len(xs)
        L1 = xs[-1]
        Dr = []
        ua = []
        for r in R:
            qa, qb = r["e1"]["a"]["qbar"], r["e1"]["b"]["qbar"]
            qdiff = max(qdiff, abs(qa - r["prof"]["a"][2]), abs(qb - r["prof"]["b"][2]))
            va = [u / qa ** 2 for u in r["prof"]["a"][1]]
            vb = [u / qb ** 2 for u in r["prof"]["b"][1]]
            ua.append(va)
            Dr.append([b - a for a, b in zip(va, vb)])
        mD = [st.mean(d[i] for d in Dr) for i in range(n)]
        se = [st.stdev([d[i] for d in Dr]) / math.sqrt(10) for i in range(n)]
        inter = [i for i in range(n) if 5 * L <= xs[i] <= L1 - 5 * L]
        Ma = st.mean(st.mean(u[i] for u in ua) for i in inter)
        out += "### amp %s: `lambda = %s`, `L1 = %s`, planes %d, interior planes %d, `M_a = %.4f`\n\n" % (amp, L, L1, n, len(inter), Ma)
        for face in ("west", "east"):
            dist = [xs[i] if face == "west" else L1 - xs[i] for i in range(n)]
            order = [i for i in range(n) if dist[i] >= 0]
            rows = []
            for k in DIST:
                i = min(order, key=lambda j: abs(dist[j] - k * L))
                rows.append(("%g" % k, "%.3f" % (dist[i] / L), "%.4e" % mD[i], "%.3e" % se[i], "%.4f" % (mD[i] / Ma)))
            out += "%s face:\n\n" % face + table(("d/lambda (target)", "d/lambda (plane)", "mean D", "SE", "mean D / M_a"), rows) + "\n"
            near = sorted((i for i in range(n) if dist[i] <= 20 * L + 1e-12), key=lambda j: dist[j])
            pts = [(dist[i] / L, mD[i], se[i]) for i in near]
            d1 = thickness(pts, lambda p: abs(p[1]) <= 0.01 * Ma)
            d2 = thickness(pts, lambda p: abs(p[1]) <= 2 * p[2])
            v520 = [p for p in pts if p[0] >= 5 - 1e-12 and abs(p[1]) > 2 * p[2]]
            n520 = len([p for p in pts if p[0] >= 5 - 1e-12])
            dd = []
            for k in (1, 2, 3, 4):
                i = min(order, key=lambda j: abs(dist[j] - k * L))
                dd.append(mD[i])
            dec = "%.3f" % gmean([dd[k + 1] / dd[k] for k in range(3)]) if all(v > 0 for v in dd) else "n/a"
            summ.append((amp, face, "%.3f" % d1 if d1 is not None else "n/a", "%.3f" % d2 if d2 is not None else "n/a", "%d/%d" % (len(v520), n520), dec))
    out += "### Derived thickness readings and decay (post-hoc)\n\n" + table(("amp", "face", "d_1% / lambda", "d_2SE / lambda", "viol 5-20 (2SE)", "decay ratio per lambda, 1-4 lambda"), summ)
    out += "\nmax |qbar(e1.json) - u_mean(plane 0)| over all runs and media: %.3e\n" % qdiff
    return out


# ---------------------------------------------------------------------------------------------------------------- P4
P4_CODE = """\
### Imposed conditions (confirmed in code)

| medium | what is imposed | evidence |
|---|---|---|
| (a) SF-19 affine-periodic | the **mean flux**: target `qbar = (1, 0, 0)`; the mean gradient `G` is an output of the full 3x3 solve `K_eff G = qbar`, so it is free and may have transverse components | `src/physics/flow/AffinePeriodicFlowSolver.cuh:156-167` (`qbar` is the target mean Darcy flux, default `(1,0,0)`), `:212-213` (`G` = solution of `K_eff G = qbar`); `src/physics/flow/AffinePeriodicFlowSolver.cu:965` (lean path used by the stage: `solve_3x3_cramer(report.K_eff, qbar, report.G)`; non-lean `:837`); `src/physics/particles/lagrangian_labels/LongBoxStage.cu:251-253` (`fcfg.qbar = (1,0,0)`) and `:261` (`solve_affine_periodic_flow_lean`) |
| (b) `DirichletMedium` / `solve_head` | the **head drop along x**: `h(0) = G1 L1`, `h(L1) = 0` with `G1 = G[0]` of the (a) solve of the same realization, periodic `y`/`z`; hence the mean head gradient is along `x` and the mean transverse flux is free | `src/physics/particles/lagrangian_labels/DirichletMedium.cu:153-160` (`head_drop = spec.G1 * M.L1`; `xmin` Dirichlet `head_drop`, `xmax` Dirichlet `0`, `ymin/ymax/zmin/zmax` Periodic); `apps/lagrangian_medium_ab/lagrangian_medium_ab_main.cu:599` (`G1 = stage->G[0]`) and `:671` (`bs.G1 = G1`) |

Consequence (hypothesis consistent with the data, not tested here): the two media differ in the macroscopic forcing everywhere in the domain, not only near the `x` faces. In (a) the mean flux is `e1` by construction and the mean gradient carries the transverse components `G[1]`, `G[2]`; in (b) the mean gradient is `(G1, 0, 0)` and the mean flux of a finite realization is `K_eff (G1, 0, 0)` with whatever transverse part the realization's effective tensor gives (the per-realization `theta_2`, `theta_3` of (b) below).

"""


def p4(raw, data):
    out = ("## P4 Mean-flow forcing and tilt (derived, post-hoc)\n\n"
           "Per realization `theta_2`, `theta_3` (interior, `primary.json`) and `qbar` of both media; sample sd over the ten realizations per medium and amplitude. "
           "For (b): the layer-mean `<v2>/qbar` and `<v3>/qbar` over the four contiguous quarters of the interior cell layers `[layer_lo, layer_hi]` of each "
           "realization (Q1 = nearest the west face). `G` is the mean gradient of the (a) solve (`STAGE ... G=` line of `ab_s*.log`).\n\n")
    out += P4_CODE
    srows, qrows, quarters = [], [], []
    gcorr = []
    for amp, d in AMPS:
        R = data[amp]
        log = read(os.path.join(raw, "sf38_campaign", "ab", d + ".log"))
        G = {int(re.match(r"R=(\d+) ", l).group(1)): [float(v) for v in toks(l)["G"].split(",")] for l in log.splitlines() if re.match(r"R=\d+ STAGE ", l)}
        t2b = [r["pr"]["b"]["theta_2"] for r in R]
        t3b = [r["pr"]["b"]["theta_3"] for r in R]
        t2a = [r["pr"]["a"]["theta_2"] for r in R]
        t3a = [r["pr"]["a"]["theta_3"] for r in R]
        qa = [r["pr"]["a"]["qbar"] for r in R]
        qb = [r["pr"]["b"]["qbar"] for r in R]
        g2 = [G[i][1] / G[i][0] for i in range(10)]
        g3 = [G[i][2] / G[i][0] for i in range(10)]
        for nm, v in (("theta_2 (a)", t2a), ("theta_3 (a)", t3a), ("theta_2 (b)", t2b), ("theta_3 (b)", t3b),
                      ("theta_2 (b) - (a)", [x - y for x, y in zip(t2b, t2a)]), ("theta_3 (b) - (a)", [x - y for x, y in zip(t3b, t3a)]),
                      ("qbar (a)", qa), ("qbar (b)", qb), ("G2/G1 (a solve)", g2), ("G3/G1 (a solve)", g3)):
            srows.append((amp, nm, "%.4e" % st.mean(v), "%.4e" % st.stdev(v), "%.4e" % min(v), "%.4e" % max(v)))
        rms = math.sqrt(st.mean([x * x + y * y for x, y in zip(t2b, t3b)]))
        gcorr.append((amp, "%.3f" % pearson(t2b, g2), "%.3f" % pearson(t3b, g3), "%.4e" % rms, "%.3f" % (st.stdev(t2b) / 0.005), "%.3f" % (st.stdev(t3b) / 0.005),
                      "%.3f" % (rms / 0.0071)))
        for i, r in enumerate(R):
            lo, hi = r["e1"]["b"]["layer_lo"], r["e1"]["b"]["layer_hi"]
            assert (lo, hi) == (r["e1"]["a"]["layer_lo"], r["e1"]["a"]["layer_hi"])
            n = hi - lo + 1
            cuts = [lo + (n * k) // 4 for k in range(5)]
            Q = {}
            for m in "ab":
                q = r["e1"][m]["qbar"]
                Q[m] = [[st.mean(c[cuts[k]:cuts[k + 1]]) / q for k in range(4)] for c in r["lay"][m]]  # [v2 quarters, v3 quarters]
            tot = [st.mean(c[lo:hi + 1]) / r["e1"]["b"]["qbar"] for c in r["lay"]["b"]]
            assert abs(tot[0] - r["pr"]["b"]["theta_2"]) < 1e-12 and abs(tot[1] - r["pr"]["b"]["theta_3"]) < 1e-12
            quarters.append((amp, Q, r["pr"]))
            qrows.append((amp, i, "%.4e" % r["pr"]["b"]["theta_2"]) + tuple("%.4e" % x for x in Q["b"][0]) + ("%.4e" % r["pr"]["b"]["theta_3"],) + tuple("%.4e" % x for x in Q["b"][1]))
    out += "### Per-medium statistics over the ten realizations\n\n" + table(("amp", "quantity", "mean", "sd", "min", "max"), srows)
    out += ("\n### Tilt of (b) against the forcing\n\n`corr` = Pearson correlation over the ten realizations of the (b) tilt with the transverse-to-longitudinal ratio of the "
            "mean gradient of the (a) solve (exploratory; n = 10). `rms |theta|` = sqrt(mean(theta_2^2 + theta_3^2)) of (b). Ratios only (no alpha statement): "
            "sd of each component of (b) over 0.005 (the D19 per-component margin) and `rms |theta|` over 0.0071 (the D19 vector bound).\n\n")
    out += table(("amp", "corr(theta_2 b, G2/G1)", "corr(theta_3 b, G3/G1)", "rms abs theta (b)", "sd(theta_2 b)/0.005", "sd(theta_3 b)/0.005", "rms abs theta (b)/0.0071"), gcorr)
    out += ("\nGeometric consequence (no alpha statement): in fixed axes, the transverse variance of a plume whose mean flow is tilted by `theta` picks up approximately "
            "`theta^2 sigma_L^2`, with `sigma_L^2` the longitudinal variance of the plume.\n\n")
    out += ("### Layer-mean tilt of (b) along `x1` (quarters of the interior layers)\n\nThe four quarters check whether the (b) tilt is one constant along `x1` "
            "(equal Q1..Q4) or varies.\n\n")
    out += table(("amp", "r", "theta_2 (b)", "Q1 v2", "Q2 v2", "Q3 v2", "Q4 v2", "theta_3 (b)", "Q1 v3", "Q2 v3", "Q3 v3", "Q4 v3"), qrows)
    out += ("\nSummary of the quarter spread (derived): per realization `spread = max(Q) - min(Q)` over the four quarters, averaged over the ten realizations, "
            "against the mean `|overall|` of the same quantity. `(b)` = the table above; `(a)` = the same quarters of the affine-periodic medium (its window-mean "
            "fluctuation is the sampling control); `(b)-(a)` = quarter differences on the same field, which removes the field's own fluctuation (overall = "
            "`theta (b) - theta (a)`). A constant medium offset along `x1` would give `(b)-(a)` spread much smaller than its `|overall|`.\n\n")
    srows2 = []
    for amp, _ in AMPS:
        for c, comp in enumerate(("v2", "v3")):
            for nm in ("(b)", "(a)", "(b)-(a)"):
                sp, th = [], []
                for a_, Q, pr in quarters:
                    if a_ != amp:
                        continue
                    key = "theta_%d" % (c + 2)
                    if nm == "(b)":
                        q, t = Q["b"][c], pr["b"][key]
                    elif nm == "(a)":
                        q, t = Q["a"][c], pr["a"][key]
                    else:
                        q, t = [x - y for x, y in zip(Q["b"][c], Q["a"][c])], pr["b"][key] - pr["a"][key]
                    sp.append(max(q) - min(q))
                    th.append(abs(t))
                srows2.append((amp, comp, nm, "%.4e" % st.mean(sp), "%.4e" % st.mean(th), "%.3f" % (st.mean(sp) / st.mean(th))))
    out += table(("amp", "component", "quantity", "mean spread", "mean abs overall", "mean spread / mean abs overall"), srows2)
    return out


# ---------------------------------------------------------------------------------------------------------------- P5
def p5(raw):
    ab = os.path.join(raw, "sf38_campaign", "ab")
    res = read(os.path.join(ab, "results.txt"))
    rows = []
    bpc = {}
    for amp, d in AMPS:
        log = read(os.path.join(ab, d + ".log"))
        geo = toks([l for l in res.splitlines() if l.startswith("RECORDED run=%s R=0 MEM_GEOM " % d)][0])
        cells = int(geo["n"])
        pk = toks([l for l in res.splitlines() if l.startswith("RECORDED run=%s pid_peak_mib" % d)][0])
        pid = float(pk["pid_peak_mib"])
        sol = toks([l for l in log.splitlines() if l.startswith("R=0 MEM_PHASE_B phase=solve ")][0])
        ex = int(sol["exact_bytes"])
        bpc[amp] = pid * MIB / cells
        rows.append((d, "%sx%sx%s" % (geo["nx"], geo["ny"], geo["nz"]), cells, "%.0f" % pid, "%.2f" % bpc[amp], "%.2f" % (ex / cells), "%.0f" % (bpc[amp] * PROD_CELLS / MIB)))
    b2 = read(os.path.join(raw, "sf38_campaign", "prodgrid", "B2.log"))
    s2 = toks([l for l in b2.splitlines() if l.startswith("R=0 MEM_PHASE phase=S2 ")][0])
    sp = toks([l for l in read(os.path.join(raw, "sf38_campaign", "prodgrid", "results.txt")).splitlines() if l.startswith("RECORDED run=B2 pid_peak_mib")][0])
    splus_exact = int(s2["exact_bytes"]) / PROD_CELLS
    splus_pid = float(sp["pid_peak_mib"]) * MIB / PROD_CELLS
    ceil_bpc = CEILING_MIB * MIB / PROD_CELLS
    out = ("## P5 Memory of medium (b) at the production grid (derived, post-hoc)\n\n"
           "Measured peaks of the A/B runs (`RECORDED run=ab_s* pid_peak_mib` of `ab/results.txt`; the peak is the (b) `solve` phase, realization 0 `MEM_PHASE_B` of "
           "`ab_s*.log`). `B/cell (PID)` = PID peak MiB x 1048576 / cells, **including the CUDA context** and the unmodeled overhead; `B/cell (exact solve)` = "
           "`exact_bytes` of the `solve` phase / cells (allocations only). Extrapolation to 3328x384x384 (%d cells) is linear in cells (rough; the context does not scale, "
           "so it overstates slightly).\n\n" % PROD_CELLS)
    out += table(("run", "grid", "cells", "PID peak MiB", "B/cell (PID, with context)", "B/cell (exact solve)", "extrapolated PID peak at 3328x384x384, MiB (derived)"), rows)
    lean = MEM_LEAN_BPC * PROD_CELLS / MIB
    out += ("\nLean non-projected (b) flow solve: %.2f B/cell (memory audit 2026-10-22 section 10, buffer count of `solve_head` with the section 8 tricks; class "
            "**estimate from code count, not measured**) x %d cells = %.0f MiB (allocations only, no context) against the %.0f MiB ceiling. "
            "The non-periodic spline stage is not included in either figure (the extrapolated measured figure or the lean estimate).\n\n" % (MEM_LEAN_BPC, PROD_CELLS, lean, CEILING_MIB))
    out += ("Reference levels at the same box (measured, SF-38 B2): the lean periodic stage with the SF-19 flow and the spline reaches exact %.3f B/cell (`MEM_PHASE phase=S2`, "
            "`prodgrid/B2.log`) and a PID peak of %s MiB = %.2f B/cell with context; the ceiling %.0f MiB corresponds to %.2f B/cell with context.\n\n" % (
                splus_exact, sp["pid_peak_mib"], splus_pid, CEILING_MIB, ceil_bpc))
    out += ("Conclusion (derived): medium (b) as measured (%.0f MiB extrapolated from `ab_s4`, %.0f MiB from `ab_s1`) does not fit 3328x384x384 in %.0f MiB; even the lean "
            "non-projected estimate (%.0f MiB, before the spline) exceeds it. A lean (b) flow stage together with the non-periodic spline would have to come in at "
            "or below the level of the SF-40 stage (~%.0f B/cell, %.3f exact; PID peak %s MiB) to fit; otherwise stop rule R3 (domain reduction by owner decision) applies at "
            "`sigma^2 >= 2.25`.\n" % (bpc["4"] * PROD_CELLS / MIB, bpc["1"] * PROD_CELLS / MIB, CEILING_MIB, lean, round(splus_exact), splus_exact, sp["pid_peak_mib"]))
    return out


# ---------------------------------------------------------------------------------------------------------------- P6
def p6(raw):
    """E2 plane rows of both media: contributing / non_ok counts (the Pollock plane pass records no x1 < 0 excursion)."""
    rows = []
    for amp, d in AMPS:
        base = os.path.join(raw, "sf38_campaign", "ab", d, "realizations")
        for m in "ab":
            nfiles = nplanes = nfull = nnonok = 0
            for r in range(10):
                with open(os.path.join(base, "r_%04d" % r, "e2_%s.csv" % m)) as f:
                    for x in csv.DictReader(f):
                        nplanes += 1
                        nfull += int(x["contributing"]) == 10000
                        nnonok += int(x["non_ok"]) != 0
                nfiles += 1
            rows.append((d, m, nfiles, nplanes, nfull, nnonok))
    tot = (sum(r[3] for r in rows), sum(r[4] for r in rows), sum(r[5] for r in rows))
    out = ("## P6 Pollock plane pass of medium (b): recorded `non_ok` and x-exit handling (derived, post-hoc)\n\n"
           "Counts over the `e2_{a,b}.csv` plane rows of the 20 realizations (10 per amplitude) of each medium: planes with `contributing = 10000` and planes with `non_ok != 0`.\n\n")
    out += table(("run", "medium", "realizations", "plane rows", "rows with contributing = 10000", "rows with non_ok != 0"), rows)
    out += ("\nTotal (both media): %d plane rows, %d with `contributing = 10000`, %d with `non_ok != 0`. No per-particle wrap counter and no minimum `x1` is written to `raw/`, so the number "
            "of particles that reached `x1 < 0` from the injection plane `X0 = 5 lambda` is not in the imported evidence.\n" % tot)
    return out


# ---------------------------------------------------------------------------------------------------------------- driver
def main(argv):
    if argv[1:] == ["--selftest"]:
        assert abs(pearson([1, 2, 3], [2, 4, 6]) - 1) < 1e-15 and abs(pearson([1, 2, 3], [3, 2, 1]) + 1) < 1e-15
        assert abs(gmean([1, 4]) - 2) < 1e-15
        pts = [(0.0, 5, 1), (1.0, 3, 1), (2.0, 0.1, 1), (3.0, 0.5, 1)]
        assert thickness(pts, lambda p: abs(p[1]) <= 1) == 2.0
        assert thickness(pts, lambda p: abs(p[1]) <= 10) == 0.0
        assert thickness(pts, lambda p: abs(p[1]) <= 0.01) is None
        assert table(("a",), [(1,)]) == "| a |\n|---|\n| 1 |\n"
        print("selftest ok")
        return 0
    raw, outd = argv[1], argv[2]
    data = {amp: load_amp(raw, d) for amp, d in AMPS}
    head = ("# SF-38 post-hoc readings of the medium A/B evidence (N9)\n\n"
            "**Post-hoc, NOT pre-registered.** Derived by `scripts/posthoc_ab.py` from `raw/sf38_campaign/ab/` (and `raw/sf38_campaign/prodgrid/B2.log`) after FINAL_AUDIT, at the "
            "owner's request. Exploratory (forking-paths risk): none of these readings changes any pre-registered verdict, threshold or analysis file, and the "
            "medium decision stands as the rule's outcome. No `alpha_L` / `alpha_T` statement; `S_T` and `S_tau` appear only as the rule's statistics.\n\n")
    text = head + "\n".join([p1(data), p2(raw), p3(raw, data), p4(raw, data), p5(raw), p6(raw)])
    os.makedirs(outd, exist_ok=True)
    with open(os.path.join(outd, "posthoc_ab.md"), "w") as f:
        f.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
