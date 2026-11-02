#!/usr/bin/env python3
"""SF-43 pre-activation review, Gate 5 statistics: read-only, CPU-only (nothing runs on the host or on a GPU).

Reproduces the numbers of the SF-43 Gate 5 pre-activation analysis from committed inputs only: the SF-42 pilot tables
(--pilots, the SF-42 artifact directory pilots/), the R5 coverage summary (--coverage, coverage/), the SF-42 statistics
package mcstat (--sf42-scripts, the SF-42 scripts/ directory; imported with bytecode writing disabled) and, for one table
of the runtime sub-command, the SF-39 per-arm raw logs (--sf39, the SF-39 artifact root; by default inferred next to the
SF-42 artifact; skipped with a note when absent).  Python >= 3.11 with numpy and scipy.  Every random stream is seeded.

Sub-commands (markdown tables on stdout):
  moments           Section 1. G1 / G2 (bias-corrected) of per-realization alpha_LP of each code, of the paired
                    linearized contrast e_r = L_r / mean(L) - P_r / mean(P) and of log(L_r / P_r), with seed-bootstrap
                    bounds (80 / 95 % quantiles of |G1*| and G2*, B = 20000: the covdist rule); alpha_TP by basis; the
                    provenance of the stress family; what 8 pairs constrain; the estimand shift of a geometric-mean TOST.
  fieller-coverage  Section 2(i). R5 sampling cells re-judged as a Fieller-form TOST (each one-sided test at a_j, or at
                    a_j / 2 judged against a_j); R5 TOST cells restricted to tails >= 0.0125 and to 0.05; R5 host seconds
                    per cell and the CPU cost of the restricted coverage run (section 3).
  skewcorr          Section 2(ii). Student, Johnson (1978) and Hall (1992) one-sided tests of a mean at the 7 Holm tails
                    under the stress family, the normal and a heavier mixture (local Monte Carlo; --R replicates).
  sizing            Section 4. Gate 5 sample size, power at ratio 1 and 0.95, minimal detectable equivalence; halved
                    level; Holm m = 4 level; normal-quantile ratios m = 3 against m = 7; uncertainty of the pilot CVs;
                    GPU-h of Gate 5 designs from the measured pilot times.
  g7                Section 5. Minimal detectable effects of the G7 split-plot factorial.
  runtime           Section 6. Spearman correlations of per-realization runtime, PCG iterations and tracker work with alpha.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import re
import sys
import warnings
import zlib

import numpy as np
from scipy import optimize, stats

HOLM = np.array([0.05 / (7 - j) for j in range(7)])  # Holm levels a_j for m = 7 (decision G5.6), increasing
A7 = float(HOLM[0])  # Holm worst level 0.05 / 7
BOOT_SEED = 20261102  # moments bootstrap stream: SeedSequence([BOOT_SEED, crc32(label)])
B_MOMENTS = 20000
AMPS = ("0.25", "1", "2.25", "4")
GATE5 = {  # sigma^2: (consumer table, Par2 table, Par2 horizon factor); no horizon-3.6 Par2 run exists at 0.25
    "0.25": ("J3/reduced/c_s025_1040x240.csv", "J3/reduced/p_s025_1040x240.csv", "1.2"),
    "1": ("J3/reduced/c_s1_1040x240.csv", "J3B/reduced/p_s1_1040x240_h3.csv", "3.6"),
    "2.25": ("J3/reduced/c_s225_1248x288.csv", "J3B/reduced/p_s225_1248x288_h3.csv", "3.6"),
    "4": ("J3/reduced/c_s4_1664x192.csv", "J3B/reduced/p_s4_1664x192_h3.csv", "3.6"),
}
J2_TABLES = {"0.25": "J2/reduced/c_s025_M128.csv", "1": "J2/reduced/c_s1_M128.csv",
             "2.25": "J2/reduced/c_s225_M100.csv", "4": "J2/reduced/c_s4_M100.csv"}
RUN_IDS = {  # pilot runs on the Gate 5 grids: (job, run id); per-realization time as in the SF-42 cost model
    "consumer": {"0.25": ("J3", "c_s025_1040x240"), "1": ("J3", "c_s1_1040x240"), "2.25": ("J3", "c_s225_1248x288"),
                 "4": ("J3", "c_s4_1664x192")},
    "par2": {"1": ("J3B", "p_s1_1040x240_h3"), "2.25": ("J3B", "p_s225_1248x288_h3"), "4": ("J3B", "p_s4_1664x192_h3")},
}
HDR = ["sigma^2", "quantity", "n", "mean", "CV", "G1", "|G1| UCB80 / 95", "G2", "G2 UCB80 / 95", "G1 90 % pct"]
SKEW_RUNS = (  # (rng seed, ((distribution, sizes), ...)): the two local runs of the report, in their order
    (20261102, (("stress", (30, 50, 100, 200, 300)),)),
    (99, (("stress", (150, 250)), ("normal", (50, 100, 200, 300)), ("mix:2.73:7.59", (50, 100, 200, 300)))),
)
SKEW_KEYS = ("Student upper", "Johnson upper", "Hall upper", "Student lower", "Johnson lower", "Hall lower")
SF39_GROUPS = {  # SF-39 sigma^2 = 4 arms with per-realization TIMING lines: (raw logs, reduced table)
    "s4_n32": (("raw/campaign/pilot/logs/s4_n32_s5001_n20.log",), "s4_n32.csv"),
    "s4_n64": (("raw/campaign/pilot/logs/s4_n64_s5001_n20.log",), "s4_n64.csv"),
    "s4_n16": (("raw/campaign/pilot/logs/s4_n16_s5001_n20.log", "raw/campaign/extend/logs/s4_n16_s5001_n30.log"),
               "s4_n16_first30.csv"),
    "s4_tw": (("raw/campaign/pilot/logs/s4_tw_s5001_n20.log", "raw/campaign/extend/logs/s4_tw_s5001_n30.log"),
              "s4_tw_first30.csv"),
    "s4_w24": (("raw/campaign/width/logs/s4_w24_s7101_n30.log",), "s4_w24.csv"),
    "s4_w12": (("raw/campaign/width/logs/s4_w12_s7201_n30.log", "raw/campaign/width/logs/s4_w12c_s7215_n17.log"),
               "s4_w12.csv"),
}
SF39_COLS = (("alpha_TP", "tp.alpha_TP"), ("alpha_LAM", "win.alpha_e1"), ("alpha_TAM", "win.alpha_TAM"))


# ----------------------------------------------------------------------------------------------------------------------
# utilities
# ----------------------------------------------------------------------------------------------------------------------
def fail(msg):
    sys.exit("error: " + msg)


def need(args, *names):
    for name in names:
        if getattr(args, name, None) is None:
            fail("--%s is required by this sub-command" % name.replace("_", "-"))


def load_mcstat(args):
    """covdist, gate5, sizing of mcstat from --sf42-scripts (no bytecode is written into the repository)."""
    need(args, "sf42_scripts")
    sys.dont_write_bytecode = True
    root = os.path.abspath(args.sf42_scripts)
    if not os.path.isfile(os.path.join(root, "mcstat", "__init__.py")):
        fail("no mcstat package under %s" % root)
    if root not in sys.path:
        sys.path.insert(0, root)
    from mcstat import covdist, gate5, sizing
    return covdist, gate5, sizing


@contextlib.contextmanager
def quiet():
    """Silence numpy / scipy warnings of degenerate resamples (their NaN results are filtered)."""
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        yield


def fmt(x, spec="%.2f"):
    if x is None:
        return "-"
    try:
        if not math.isfinite(float(x)):
            return "-"
    except (TypeError, ValueError):
        return str(x)
    return spec % x


def table(header, rows, title=None):
    if title:
        print("\n### %s\n" % title)
    print("| " + " | ".join(str(h) for h in header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        print("| " + " | ".join(str(c) for c in r) + " |")
    print("")


def read_csv(path):
    if not os.path.isfile(path):
        fail("missing input %s" % path)
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def read_json(path):
    if not os.path.isfile(path):
        fail("missing input %s" % path)
    with open(path) as fh:
        return json.load(fh)


def plane_values(rows, code, weights, basis, col):
    """{r: value} of the plane-family perplane rows of a reduced table (finite values only), as pilots.analyze."""
    out = {}
    for r in rows:
        if (r["code"] == code and r["family"] == "plane" and r["weights"] == weights and r["basis"] == basis
                and r["set"] == "perplane"):
            s = r[col]
            v = float(s) if s not in ("", "nan", "NaN") else float("nan")
            if math.isfinite(v):
                out[int(r["r"])] = v
    return out


def vals(rows, code, weights, basis, col):
    d = plane_values(rows, code, weights, basis, col)
    return np.array([d[k] for k in sorted(d)])


def gate5_pairs(pilots, amp, col="alpha_LP"):
    """Paired per-seed values: consumer uniform weights (twin series) and Par2, position basis, matched by r."""
    cf, pf, hz = GATE5[amp]
    L = plane_values(read_csv(os.path.join(pilots, cf)), "consumer", "unif", "pos", col)
    P = plane_values(read_csv(os.path.join(pilots, pf)), "par2", "unif", "pos", col)
    rs = sorted(set(L) & set(P))
    return np.array([L[r] for r in rs]), np.array([P[r] for r in rs]), rs, hz


def spearman(x, y):
    with quiet():
        return float(stats.spearmanr(x, y)[0])


def fisher_z(rho):
    return math.atanh(min(max(rho, -0.999), 0.999))


def pooled_fisher(lst):
    """Fisher-z pooling of [(z, n)] with weights n - 3: (rho, 95 % lower, 95 % upper)."""
    z = np.array([a for a, _ in lst])
    w = np.array([n - 3.0 for _, n in lst])
    zb, se = float((w * z).sum() / w.sum()), 1.0 / math.sqrt(float(w.sum()))
    return math.tanh(zb), math.tanh(zb - 1.96 * se), math.tanh(zb + 1.96 * se)


def near(t, arr):
    return bool(np.any(np.abs(np.asarray(arr) - t) < 1e-12))


# ----------------------------------------------------------------------------------------------------------------------
# moments (section 1; GMR shift of section 2(b))
# ----------------------------------------------------------------------------------------------------------------------
def g1g2(v):
    with quiet():
        return float(stats.skew(v, bias=False)), float(stats.kurtosis(v, bias=False))


def boot_bounds(v, label, B=B_MOMENTS):
    """covdist rule: 80 / 95 % quantiles of |G1*| and of G2* over B seed resamples; also the 5 / 95 % quantiles of G1*."""
    rng = np.random.default_rng(np.random.SeedSequence([BOOT_SEED, zlib.crc32(str(label).encode())]))
    bs = v[rng.integers(0, v.size, size=(B, v.size))]
    with quiet():
        g1 = stats.skew(bs, axis=1, bias=False)
        g2 = stats.kurtosis(bs, axis=1, bias=False)
    ok = np.isfinite(g1) & np.isfinite(g2)
    g1, g2 = g1[ok], g2[ok]
    a = np.abs(g1)
    return (float(np.quantile(a, 0.8)), float(np.quantile(a, 0.95)), float(np.quantile(g2, 0.8)),
            float(np.quantile(g2, 0.95)), float(np.quantile(g1, 0.05)), float(np.quantile(g1, 0.95)))


def moment_row(group, name, v, label, show_cv):
    g1, g2 = g1g2(v)
    a80, a95, k80, k95, lo, hi = boot_bounds(v, label)
    cv = float(v.std(ddof=1) / abs(v.mean())) if show_cv else float("nan")
    return (group, name, v.size, fmt(float(v.mean()), "%.4g"), fmt(cv, "%.3f"), fmt(g1), "%s / %s" % (fmt(a80), fmt(a95)),
            fmt(g2), "%s / %s" % (fmt(k80), fmt(k95)), "[%s, %s]" % (fmt(lo), fmt(hi)))


def cmd_moments(args):
    covdist, g5, _ = load_mcstat(args)
    need(args, "pilots")
    pil = os.path.abspath(args.pilots)
    sm = read_json(os.path.join(os.path.abspath(args.sf42_scripts), "mcstat", "stress_moments.json"))
    env = sm["envelope"]
    src = [s for s in sm["series"] if s["skew_ucb"] == env["skew"] or s["ekurt_ucb"] == env["ekurt"]]
    table(["file", "column", "n", "G1", "|G1| UCB95", "G2", "G2 UCB95"],
          [(s["file"], s["column"], s["n"], fmt(s["skew"], "%.3f"), fmt(s["skew_ucb"], "%.4f"), fmt(s["ekurt"], "%.3f"),
            fmt(s["ekurt_ucb"], "%.4f")) for s in src],
          "Stress family: envelope skew %.4f, excess kurtosis %.4f = maximum over %d SF-39 series (stress_moments.json)"
          % (env["skew"], env["ekurt"], len(sm["series"])))
    table(["sigma^2", "estimator", "n", "G1", "|G1| UCB95", "G2", "G2 UCB95"],
          [(s["amplitude"], s["estimator"], s["n"], fmt(s["skew"], "%.3f"), fmt(s["skew_ucb"], "%.3f"),
            fmt(s["ekurt"], "%.3f"), fmt(s["ekurt_ucb"], "%.3f")) for s in sm["series"] if s["set"] == "prod"],
          "SF-39 production series (largest n), same file")

    rows, std_e, eg1 = [], {}, {}
    for amp in AMPS:
        x, y, rs, hz = gate5_pairs(pil, amp)
        e = x / x.mean() - y / y.mean()
        rows.append(moment_row(amp, "alpha_LP consumer (unif)", x, amp + "L", True))
        rows.append(moment_row(amp, "alpha_LP Par2 (horizon %s)" % hz, y, amp + "P", True))
        rows.append(moment_row(amp, "e_r", e, amp + "e", False))
        rows.append(moment_row(amp, "log(L/P)", np.log(x / y), amp + "lr", False))
        std_e[amp] = e / e.std(ddof=1)
        eg1[amp] = g1g2(e)[0]
    pooled = np.concatenate([std_e[a] for a in ("1", "2.25", "4")])
    rows.append(moment_row("1, 2.25, 4", "e_r / sd(e_r), pooled", pooled, "pool", False))
    rows.append(moment_row("all four", "e_r / sd(e_r), pooled", np.concatenate([pooled, std_e["0.25"]]), "pool2", False))
    table(HDR, rows, "Section 1: Gate 5 pairs (J3 consumer; J3B Par2, J3 Par2 at 0.25); bootstrap B = %d" % B_MOMENTS)

    rows = []
    for amp in AMPS:
        cf, pf, _ = GATE5[amp]
        C, P = read_csv(os.path.join(pil, cf)), read_csv(os.path.join(pil, pf))
        C2 = read_csv(os.path.join(pil, J2_TABLES[amp]))
        for b in ("pos", "disp"):
            for name, v, lab in (("alpha_TP J3 consumer flux " + b, vals(C, "consumer", "flux", b, "alpha_TP"), amp + b + "c"),
                                 ("alpha_TP J3 consumer unif " + b, vals(C, "consumer", "unif", b, "alpha_TP"), amp + b + "cu"),
                                 ("alpha_TP Par2 unif " + b, vals(P, "par2", "unif", b, "alpha_TP"), amp + b + "p"),
                                 ("alpha_TP J2 consumer flux " + b + " (6 fields)", vals(C2, "consumer", "flux", b, "alpha_TP"),
                                  amp + b + "j2")):
                rows.append(moment_row(amp, name, v, lab, False))
        rows.append(moment_row(amp, "alpha_LP J3 consumer flux pos", vals(C, "consumer", "flux", "pos", "alpha_LP"), amp + "lpf", True))
        rows.append(moment_row(amp, "alpha_LP J2 consumer flux pos (6 fields)", vals(C2, "consumer", "flux", "pos", "alpha_LP"),
                               amp + "lpj2", True))
    table(HDR, rows, "alpha_TP by basis and flux-weighted alpha_LP (J3: 8 fields; J2: 6 fields)")

    for amp in AMPS:
        cf, pf, hz = GATE5[amp]
        C, P = read_csv(os.path.join(pil, cf)), read_csv(os.path.join(pil, pf))
        Lu = plane_values(C, "consumer", "unif", "pos", "alpha_LP")
        Lf = plane_values(C, "consumer", "flux", "pos", "alpha_LP")
        Pp = plane_values(P, "par2", "unif", "pos", "alpha_LP")
        seed = {int(r["r"]): r["seed"] for r in C if r["code"] == "consumer"}
        rs = sorted(set(Lu) & set(Pp))
        x, y = np.array([Lu[r] for r in rs]), np.array([Pp[r] for r in rs])
        e = x / x.mean() - y / y.mean()
        k = int(np.argmax(np.abs(e)))
        keep = np.arange(x.size) != k
        table(["r", "seed", "L (unif)", "L (flux)", "P", "L / P", "e_r"],
              [(r, seed.get(r, "-"), "%.4f" % Lu[r], fmt(Lf.get(r), "%.4f"), "%.4f" % Pp[r], "%.3f" % (Lu[r] / Pp[r]),
                "%+.3f" % e[i]) for i, r in enumerate(rs)],
              "Pairs at sigma^2 = %s: ratio of means %.4f; %.4f without the pair of largest |e_r| (r = %d)"
              % (amp, x.mean() / y.mean(), x[keep].mean() / y[keep].mean(), rs[k]))

    d = covdist.get_dist("stress")
    table(["minority weight p", "within variance v", "x1", "x2", "skewness", "excess kurtosis"],
          [(fmt(d.p, "%.5f"), fmt(d.v, "%.5f"), fmt(d.x1, "%.4f"), fmt(d.x2, "%.4f"), fmt(d.skew, "%.3f"), fmt(d.ekurt, "%.3f"))],
          "Stress mixture (mcstat covdist)")
    table(["n"] + [str(n) for n in (8, 24, 30, 50, 100)],
          [["P(no minority draw)"] + ["%.3f" % (1 - d.p) ** n for n in (8, 24, 30, 50, 100)]])
    rng = np.random.default_rng(7)
    N = 200000
    gobs = sorted({round(v, 2) for v in list(eg1.values()) + [g1g2(pooled)[0]]})
    q = (0.05, 0.25, 0.5, 0.75, 0.95)
    for n in (8, 24):
        xs = d.draw(rng, (N, n))
        xn = rng.standard_normal((N, n))
        with quiet():
            gs, gn = stats.skew(xs, axis=1, bias=False), stats.skew(xn, axis=1, bias=False)
            ks, kn = stats.kurtosis(xs, axis=1, bias=False), stats.kurtosis(xn, axis=1, bias=False)
        table(["statistic", "distribution"] + ["q%02d" % int(round(100 * p)) for p in q],
              [("G1", "stress") + tuple("%.2f" % v for v in np.quantile(gs, q)),
               ("G1", "normal") + tuple("%.2f" % v for v in np.quantile(gn, q)),
               ("G2", "stress") + tuple("%.2f" % v for v in np.quantile(ks, q)),
               ("G2", "normal") + tuple("%.2f" % v for v in np.quantile(kn, q))],
              "Sampling distribution of G1 and G2 at n = %d (%d samples per distribution, rng seed 7)" % (n, N))
        table(["g (observed e_r G1, rounded)", "P(G1 <= g | stress)", "P(G1 <= g | normal)", "P(G1 >= g | stress)",
               "P(G1 >= g | normal)"],
              [(g, "%.3f" % (gs <= g).mean(), "%.3f" % (gn <= g).mean(), "%.3f" % (gs >= g).mean(), "%.3f" % (gn >= g).mean())
               for g in gobs])

    rows = []
    D = g5.LN_MARGIN
    for amp in AMPS:
        x, y, _, _ = gate5_pairs(pil, amp)
        cvL, cvP = x.std(ddof=1) / x.mean(), y.std(ddof=1) / y.mean()
        ccL, ccP, _ = g5.gate5_conservative_inputs(x.std(ddof=1), y.std(ddof=1), x.mean(), y.mean(), x.size, corr=None)
        dp, dc = (cvL ** 2 - cvP ** 2) / 2.0, (ccL ** 2 - ccP ** 2) / 2.0
        lrm, mlr = math.log(x.mean() / y.mean()), float(np.mean(np.log(x / y)))
        rows.append((amp, "%+.4f (%+.1f %%)" % (dp, 100 * dp / D), "%+.4f (%+.1f %%)" % (dc, 100 * dc / D),
                     "%+.4f (%+.1f %%)" % (lrm - mlr, 100 * (lrm - mlr) / D), "%.4f" % lrm, "%.4f" % mlr))
    table(["sigma^2", "(CV_L^2 - CV_P^2)/2, point CVs (% of log 1.1)", "same, 80 % CV bounds",
           "empirical log(mean L / mean P) - mean log(L/P)", "log ratio of means", "mean log ratio"], rows,
          "Section 2(b): estimand shift of a TOST on the mean of per-seed log ratios (margin log 1.1 = %.5f)" % D)


# ----------------------------------------------------------------------------------------------------------------------
# fieller-coverage (section 2(i), section 2(e)/(f2) evidence, section 3 cost)
# ----------------------------------------------------------------------------------------------------------------------
def cmd_fieller(args):
    need(args, "coverage")
    s = read_json(os.path.join(os.path.abspath(args.coverage), "summary.json"))
    half = HOLM / 2.0
    worst = {}
    for c in s["cells"]:
        if c["label"] not in ("sampling:student", "sampling:boott"):
            continue
        for row, ents in c["rows"].items():
            for e in ents:
                t = e["tail"]
                for kind, scale, ok in (("aj", 1.0, near(t, HOLM)), ("half", 0.5, near(t, half))):
                    if ok:  # half: the one-sided test runs at a_j / 2 and its miss rate is judged against a_j = 2 t
                        cand = (e["ucl_over_tail"] * scale, e["rate_over_tail"] * scale, c["dist"], row, t)
                        key = (c["label"], c["n"], kind)
                        if key not in worst or cand[0] > worst[key][0]:
                            worst[key] = cand
    rows = []
    for label in ("sampling:student", "sampling:boott"):
        for n in sorted({k[1] for k in worst if k[0] == label}):
            a, h = worst[(label, n, "aj")], worst[(label, n, "half")]
            rows.append((label.split(":")[1], n, "%.3f" % a[1], "%.3f" % a[0], "%s %s %.4g" % a[2:], "yes" if a[0] <= 1.25 else "no",
                         "%.3f" % h[1], "%.3f" % h[0], "%s %s %.4g" % h[2:], "yes" if h[0] <= 1.25 else "no"))
    table(["method", "n", "at a_j: max rate/a_j", "max UCL/a_j", "worst (dist row tail)", "qualifies",
           "at a_j/2: max rate/a_j", "max UCL/a_j", "worst (dist row tail)", "qualifies"], rows,
          "Section 2(i): Fieller-form TOST = one-sided limit of a mean (R5 sampling cells; upper row = H02, lower = H01; "
          "a_j only at the overlapping tails 8.33e-3, 0.0125, 0.025, 0.05; criterion UCL <= 1.25 a_j)")

    for n in (100, 200, 500):
        rows, tails = [], None
        for c in s["cells"]:
            if c["label"] == "sampling:boott" and c["n"] == n:
                for row, ents in sorted(c["rows"].items()):
                    tails = [e["tail"] for e in ents]
                    rows.append((c["dist"], row) + tuple("%.2f / %.2f%s" % (e["rate_over_tail"], e["ucl_over_tail"],
                                                                             "*" if e["pass"] is False else "") for e in ents))
        table(["dist", "row"] + ["%.4g" % t for t in tails], rows,
              "R5 sampling:boott at n = %d: rate/tail / UCL/tail per tail (* = fails at that tail)" % n)

    res = {}
    for c in s["cells"]:
        if c["proc"] != "tost":
            continue
        for row, ents in c["rows"].items():
            for e in ents:
                for key, sel in (("m4", e["tail"] >= 0.0125 - 1e-9), ("ci", abs(e["tail"] - 0.05) < 1e-12)):
                    if sel:
                        r = res.setdefault((c["label"], c["n"], key), [True, 0.0])
                        r[0] = r[0] and bool(e["pass"])
                        r[1] = max(r[1], e["ucl_over_tail"])
    rows = []
    for label in ("tost:gate5", "tost:gate5/boott"):
        for n in sorted({k[1] for k in res if k[0] == label}):
            m4, ci = res[(label, n, "m4")], res[(label, n, "ci")]
            rows.append((label, n, "yes" if m4[0] else "no", "%.2f" % m4[1], "yes" if ci[0] else "no", "%.2f" % ci[1]))
    table(["procedure", "n", "qualifies at tails >= 0.0125 (Holm m = 4)", "max UCL/tail", "qualifies at 0.05 (90 % CI)",
           "max UCL/tail"], rows, "R5 Gate 5 TOST cells restricted to the tails of options (f2) and (e)")

    secs = {}
    for c in s["cells"]:
        secs.setdefault((c["label"], c["n"]), []).append((float(c["seconds"]), int(c["R"]), c["params"].get("B")))
    rows = []
    for label in ("sampling:student", "sampling:boott", "tost:gate5", "tost:gate5/boott"):
        for n in sorted(k[1] for k in secs if k[0] == label):
            v = secs[(label, n)]
            sec = [a for a, _, _ in v]
            rows.append((label, n, len(v), v[0][1], v[0][2] if v[0][2] is not None else "-", "%.1f" % min(sec),
                         "%.1f" % float(np.mean(sec)), "%.1f" % max(sec)))
    table(["procedure", "n", "cells", "R", "B", "min s", "mean s", "max s"], rows,
          "R5 host seconds per cell (summary.json cells[].seconds; 72 workers on 40 physical cores, nice 10)")

    sb = {k[1]: float(np.mean([a for a, _, _ in v])) for k, v in secs.items() if k[0] == "sampling:boott"}

    def sampling_s(n):  # host seconds per sampling:boott cell (R 49000, B 3000): power law between measured sizes
        if n in sb:
            return sb[n]
        lo, hi = max(m for m in sb if m < n), min(m for m in sb if m > n)
        return sb[lo] * (n / lo) ** (math.log(sb[hi] / sb[lo]) / math.log(hi / lo))

    tn = sorted(k[1] for k in secs if k[0] == "tost:gate5/boott")
    slope, icpt = np.polyfit(tn, [float(np.mean([a for a, _, _ in secs[("tost:gate5/boott", n)]])) for n in tn], 1)

    def tost_s(n, R):  # host seconds per tost:gate5/boott cell (B 3000): least-squares line in n at R 24500, scaled in R
        return (icpt + slope * n) * R / 24500.0

    grid = (100, 150, 200, 250, 300, 400, 500)
    blocks = (
        ("A: Fieller cells, all methods, n " + "/".join(map(str, grid)) + " x normal, stress, stress-", 21, 49000,
         3 * sum(sampling_s(n) for n in grid)),
        ("S: stress+ (2.73, 7.59) and its mirror, n >= 150 (reported, not gating)", 12, 49000,
         2 * sum(sampling_s(n) for n in grid if n >= 150)),
        ("B: Wald bootstrap-t TOST backup, n 200/300 x rho 0.5/0.9 x 3 dists", 12, 49000,
         6 * (tost_s(200, 49000) + tost_s(300, 49000))),
        ("option (a): Wald bootstrap-t TOST, n 200-500 (5) x rho 0/0.5/0.9 x 3 dists", 45, 24500,
         9 * sum(tost_s(n, 24500) for n in (200, 250, 300, 400, 500))),
        ("option (a), same cells", 45, 49000, 9 * sum(tost_s(n, 49000) for n in (200, 250, 300, 400, 500))),
    )
    rows = [(name, cells, R, "%.1f" % (sec / 3600.0), "%.2f" % (sec / 3600.0 / 32), "%.2f" % (sec / 3600.0 / 40))
            for name, cells, R, sec in blocks]
    table(["block", "cells", "R", "core-h", "wall h, 32 workers", "wall h, 40 workers"], rows,
          "Section 3: restricted-run cost at R5 measured speed (sampling:boott power-law interpolated in n; TOST bootstrap-t "
          "line %.1f + %.2f n s per cell at R 24500); the analytic Student / Johnson / Hall cells and the two-stage check cost "
          "seconds; these values supersede the 27.4 / 17.5 core-h of A / S in the report (rounded hand interpolation)"
          % (icpt, slope))


# ----------------------------------------------------------------------------------------------------------------------
# skewcorr (section 2(ii))
# ----------------------------------------------------------------------------------------------------------------------
def skew_counts(dist, n, rng, nb, batch):
    """Misses of the one-sided tests of mean 0 at the Holm tails: upper limit misses iff stat < -t, lower iff stat > t."""
    tq = stats.t.isf(HOLM, n - 1)[None, :]
    h = {k: np.zeros(len(HOLM)) for k in SKEW_KEYS}
    rn = math.sqrt(n)
    for _ in range(nb):
        x = dist.draw(rng, (batch, n))  # standardized: true mean 0
        xb = x.mean(1)
        s = x.std(1, ddof=1)
        m3 = ((x - xb[:, None]) ** 3).mean(1)
        g = m3 / s ** 3
        T = rn * xb / s  # Student
        t1 = (xb + m3 / (6 * s ** 2 * n) + m3 / (3 * s ** 4) * xb ** 2) / (s / rn)  # Johnson (1978)
        c = g / (3 * rn)
        gT = T + c * T ** 2 + c ** 2 * T ** 3 / 3 + g / (6 * rn)  # Hall (1992): monotone cubic transform of T
        for name, v in (("Student", T), ("Johnson", t1), ("Hall", gT)):
            h[name + " upper"] += (v[:, None] < -tq).sum(0)
            h[name + " lower"] += (v[:, None] > tq).sum(0)
    return h


def cmd_skewcorr(args):
    covdist, _, _ = load_mcstat(args)
    batch = max(1, min(20000, int(args.R)))
    nb = max(1, int(args.R) // batch)
    R = nb * batch
    summary = []
    for seed, plan in SKEW_RUNS:
        rng = np.random.default_rng(seed)
        for name, sizes in plan:
            dist = covdist.get_dist(name)
            for n in sizes:
                h = skew_counts(dist, n, rng, nb, batch)
                table(["test, missed limit"] + ["%.4g" % t for t in HOLM],
                      [(k,) + tuple("%.2f" % r for r in h[k] / R / HOLM) for k in SKEW_KEYS],
                      "%s, n = %d: miss rate / tail (R = %d, rng seed %d; binomial sd at 7.14e-3 about %.3f)"
                      % (name, n, R, seed, math.sqrt(1.0 / (R * HOLM[0]))))
                summary.append((name, n) + tuple("%.2f-%.2f" % tuple((h[k] / R / HOLM)[[0, -1]]) for k in SKEW_KEYS))
    summary.sort(key=lambda r: (r[0], r[1]))
    table(["distribution", "n"] + list(SKEW_KEYS), summary,
          "Summary: miss rate / tail at a_0 = 7.14e-3 and at 0.05 (upper limit = skewed side for positive skew)")


# ----------------------------------------------------------------------------------------------------------------------
# sizing (section 4; section 2(d) halved level, (e) quantile ratios, (f2) Holm m = 4; GPU-h)
# ----------------------------------------------------------------------------------------------------------------------
def mde_ratio(g5, n, sig, alpha):
    """Largest true ratio > 1 whose TOST power is at least 0.8 (NaN when the power at ratio 1 is below 0.8)."""
    def f(th):
        return g5.tost_power(n, sig, g5.LN_MARGIN, th, alpha) - 0.8
    if f(0.0) < 0:
        return float("nan")
    return math.exp(optimize.brentq(f, 0.0, g5.LN_MARGIN - 1e-9))


def pair_stats(a, b):
    e = a / a.mean() - b / b.mean()
    return a.std(ddof=1) / a.mean(), b.std(ddof=1) / b.mean(), np.corrcoef(a, b)[0, 1], e.std(ddof=1)


def mean_seconds(pilots, job, rid):
    """Code own per-realization seconds: consumer seconds.total, Par2 field + solve + velocity + transport."""
    r = read_json(os.path.join(pilots, job, rid, "RUN.json"))
    if r["code"] == "consumer":
        v = [x["seconds"]["total"] for x in r["realizations"] if x.get("done")]
    else:
        v = [p["field_s"] + p["solve_s"] + p["velocity_s"] + p["transport_s"] for p in r["par2_stage_times"]]
    return float(np.mean(v))


def cmd_sizing(args):
    _, g5, sz = load_mcstat(args)
    need(args, "pilots")
    pil = os.path.abspath(args.pilots)
    j3, j3b = read_json(os.path.join(pil, "j3_pairing.json")), read_json(os.path.join(pil, "j3b_pairing.json"))
    src = {"0.25": (j3["0.25"], "J3, horizon 1.2"), "1": (j3b["1"], "J3B, horizon 3.6"), "2.25": (j3b["2.25"], "J3B, horizon 3.6"),
           "4": (j3b["4"], "J3B, horizon 3.6")}
    NS = (30, 50, 100, 200, 300)
    inp, rows = {}, []
    for amp in AMPS:
        d, lab = src[amp]
        cvL, cvP, rl = g5.gate5_conservative_inputs(d["sd_L"], d["sd_P"], d["mean_L"], d["mean_P"], d["n_pairs"], corr=d["corr"])
        inp[amp] = {"LB": (cvL, cvP, rl), "rho = 0": (cvL, cvP, 0.0), "point": (d["cv_L"], d["cv_P"], d["corr"])}
        rows.append((amp, lab, d["n_pairs"], "%.4f" % d["ratio_of_means"], "%.3f" % d["cv_L"], "%.3f" % d["cv_P"], "%.3f" % d["corr"],
                     "%.3f" % cvL, "%.3f" % cvP, "%.3f" % rl))
    table(["sigma^2", "pilot", "pairs", "ratio of means", "CV_L", "CV_P", "corr", "CV_L 80 %", "CV_P 80 %", "corr 80 % LB"], rows,
          "Gate 5 pilot inputs (pilots/j3_pairing.json, j3b_pairing.json) and the G5.5 conservative inputs")
    rows = []
    for amp in AMPS:
        for key in ("LB", "rho = 0", "point"):
            cvL, cvP, r = inp[amp][key]
            sig = g5.sd_e(cvL, cvP, r)
            cells = ["%.3f / %.3f" % (g5.tost_power(n, sig, g5.LN_MARGIN, 0.0, A7),
                                      g5.tost_power(n, sig, g5.LN_MARGIN, math.log(0.95), A7)) for n in NS]
            rows.append((amp, key, "%.3f, %.3f, %.3f" % (cvL, cvP, r), "%.4f" % sig, g5.gate5_sample_size(cvL, cvP, r, A7)[0],
                         *cells, " / ".join(fmt(mde_ratio(g5, n, sig, A7), "%.3f") for n in NS)))
    table(["sigma^2", "inputs", "CV_L, CV_P, corr", "sd(e)", "n for power 0.8"] + ["n=%d: P(1) / P(0.95)" % n for n in NS]
          + ["MDE ratio at n = " + "/".join(map(str, NS))], rows,
          "Section 4: Student-TOST power (analytic, gate5.tost_power) at the Holm worst level 0.05/7; MDE = largest true ratio "
          "with power >= 0.8 (- : power at ratio 1 below 0.8)")

    rows = []
    for amp in AMPS:
        cvL, cvP, r = inp[amp]["LB"]
        sig = g5.sd_e(cvL, cvP, r)
        rows.append((amp, g5.gate5_sample_size(cvL, cvP, r, A7)[0], g5.gate5_sample_size(cvL, cvP, r, A7 / 2)[0],
                     " / ".join("%.2f" % g5.tost_power(n, sig, g5.LN_MARGIN, 0.0, A7 / 2) for n in (100, 150, 200, 250, 300)),
                     g5.gate5_sample_size(cvL, cvP, r, 0.0125)[0], g5.gate5_sample_size(cvL, cvP, 0.0, 0.0125)[0],
                     " / ".join("%.2f/%.2f" % (g5.tost_power(n, sig, g5.LN_MARGIN, 0.0, 0.0125),
                                               g5.tost_power(n, sig, g5.LN_MARGIN, math.log(0.95), 0.0125)) for n in NS)))
    table(["sigma^2", "n at a_0 (LB)", "n at a_0/2 (LB)", "power at a_0/2, n = 100/150/200/250/300", "n at 0.0125 (LB)",
           "n at 0.0125 (rho 0)", "power at 0.0125, P(1)/P(0.95), n = " + "/".join(map(str, NS))], rows,
          "Sections 2(d) and 2(f2): halved level a_0/2 and the Holm m = 4 worst level 0.0125 (LB inputs unless stated)")

    a3 = 0.05 / 3.0
    rows = [(lab, "%.3g" % t7, "%.3f" % stats.norm.isf(t7), "%.3g" % t3, "%.3f" % stats.norm.isf(t3),
             "%.3f" % (stats.norm.isf(t3) / stats.norm.isf(t7)) ** 2)
            for lab, t7, t3 in (("sampling a/2", A7 / 2, a3 / 2), ("component K = 3, a/6", A7 / 6, a3 / 6),
                                ("component K = 4, a/8", A7 / 8, a3 / 8), ("TOST one-sided a (Holm m = 4 worst)", A7, 0.0125))]
    table(["term", "tail, m = 7", "z", "tail, m = 3", "z", "(z3/z7)^2"], rows,
          "Section 2(e): normal quantiles at the worst Holm level, m = 7 against m = 3 (m = 4 for the TOST row)")

    for amp in ("4", "2.25"):
        x, y, rs, _ = gate5_pairs(pil, amp)
        n = x.size
        cvL, cvP, r, se = pair_stats(x, y)
        ucb = float(sz.sd_upper(se, n))
        lo, hi = math.sqrt((n - 1) / stats.chi2.ppf(0.95, n - 1)), math.sqrt((n - 1) / stats.chi2.ppf(0.05, n - 1))
        rng = np.random.default_rng(11)
        idx = rng.integers(0, n, size=(20000, n))
        with quiet():
            res = np.array([pair_stats(x[i], y[i]) for i in idx])
        res = res[np.all(np.isfinite(res), axis=1)]
        qs = (0.05, 0.1, 0.2, 0.5, 0.8, 0.9, 0.95)
        rows = [(nm, "%.3f" % val) + tuple("%.3f" % v for v in np.quantile(res[:, j], qs))
                for j, (nm, val) in enumerate((("CV_L", cvL), ("CV_P", cvP), ("corr", r), ("sd(e)", se)))]
        table(["quantity", "point"] + ["boot q%02d" % int(round(100 * p)) for p in qs], rows,
              "sigma^2 = %s: uncertainty of the pilot inputs (%d pairs; seed bootstrap B = 20000, rng seed 11). Normal-theory "
              "90 %% CI of CV_L [%.3f, %.3f]; 80 %% UCB of sd(e) %.4f gives n = %d; n at bootstrap q50/q80/q90/q95 of sd(e): %s"
              % (amp, n, cvL * lo, cvL * hi, ucb, g5.gate5_sample_size(ucb, 0.0, 0.0, A7)[0],
                 " / ".join(str(g5.gate5_sample_size(float(np.quantile(res[:, 3], p)), 0.0, 0.0, A7)[0]) for p in (0.5, 0.8, 0.9, 0.95))))
        rows = []
        for k in range(n):
            m = np.arange(n) != k
            c = pair_stats(x[m], y[m])
            rows.append((rs[k], "%.3f" % c[0], "%.3f" % c[1], "%.3f" % c[2], "%.4f" % c[3], g5.gate5_sample_size(c[3], 0.0, 0.0, A7)[0]))
        table(["left-out r", "CV_L", "CV_P", "corr", "sd(e)", "n at point sd(e)"], rows, "sigma^2 = %s: leave-one-out" % amp)

    cons = {a: mean_seconds(pil, *RUN_IDS["consumer"][a]) for a in AMPS}
    par2 = {a: mean_seconds(pil, *RUN_IDS["par2"][a]) for a in ("1", "2.25", "4")}
    p1_j3 = mean_seconds(pil, "J3", "p_s1_1040x240")
    par2["0.25"] = mean_seconds(pil, "J3", "p_s025_1040x240") * par2["1"] / p1_j3  # horizon 3.6 by the sigma^2 = 1 ratio
    table(["sigma^2"] + list(AMPS), [["consumer s / realization (J3)"] + ["%.2f" % cons[a] for a in AMPS],
                                      ["Par2 s / realization (J3B; 0.25 scaled)"] + ["%.2f" % par2[a] for a in AMPS]],
          "Measured pilot seconds per realization (RUN.json); Par2 at 0.25 = J3 value x J3B/J3 at sigma^2 = 1 (%.3f)"
          % (par2["1"] / p1_j3))
    designs = (("J3B sizing, floor 30", (30, 30, 30, 205)), ("halved bootstrap-t floors", (100, 100, 100, 250)),
               ("Hall-Fieller floors", (150, 150, 150, 250)), ("bootstrap-t floors", (200, 200, 200, 250)),
               ("option (a), 300 everywhere", (300, 300, 300, 300)))
    rows = []
    for name, ns in designs:
        shared = sum(m * (par2[a] + (0.0 if a in ("0.25", "1") else cons[a])) for m, a in zip(ns, AMPS)) / 3600.0
        alone = sum(m * (par2[a] + cons[a]) for m, a in zip(ns, AMPS)) / 3600.0
        rows.append((name, " / ".join(map(str, ns)), "%.2f" % shared, "%.2f" % alone))
    table(["design", "pairs at 0.25 / 1 / 2.25 / 4", "GPU-h, consumer from the production twin series at 0.25 and 1",
           "GPU-h, separate consumer runs"], rows, "Gate 5 GPU-h from the measured times (C20)")


# ----------------------------------------------------------------------------------------------------------------------
# g7 (section 5)
# ----------------------------------------------------------------------------------------------------------------------
def mde_t(n_eff, df, alpha=0.05, power=0.8):
    """Two-sided t test of a mean with standard error 1 / sqrt(n_eff): smallest effect (in sd units) with the power."""
    tc = stats.t.isf(alpha / 2.0, df)

    def f(dd):
        nc = dd * math.sqrt(n_eff)
        return stats.nct.sf(tc, df, nc) + stats.nct.cdf(-tc, df, nc) - power
    return optimize.brentq(f, 1e-4, 5.0 / math.sqrt(n_eff))


def cmd_g7(args):
    need(args, "pilots")
    pil = os.path.abspath(args.pilots)
    j3, j3b = read_json(os.path.join(pil, "j3_pairing.json")), read_json(os.path.join(pil, "j3b_pairing.json"))
    cvs = (("Par2 CV at 0.25, point (J3)", j3["0.25"]["cv_P"]), ("Par2 CV at 1, point (J3B)", j3b["1"]["cv_P"]),
           ("Par2 CV at 0.25, 80 % bound", j3["0.25"]["plan_lower"]["cv_P"]), ("Par2 CV at 1, 80 % bound", j3b["1"]["plan_lower"]["cv_P"]))
    rows, rows2 = [], []
    for n in (60, 30):
        w, s1, gg = mde_t(2 * n, 2 * (n - 1)), mde_t(n, n - 1), mde_t(n / 2.0, 2 * (n - 1))
        rows.append((n, "%.3f" % w, "%.3f" % s1, "%.3f" % gg, " / ".join("%.4f" % (w * c) for c in (0.01, 0.03, 0.05, 0.10))))
        rows2 += [(n, lab, "%.4f" % cv, "%.3f" % (gg * cv), "%.3f" % math.exp(gg * cv)) for lab, cv in cvs]
    table(["seeds per cell", "within-seed main effect (x CV_c)", "simple effect in one generator (x CV_c)",
           "whole-plot G, Welch (x CV_a)", "within-seed MDE at CV_c = 0.01 / 0.03 / 0.05 / 0.10"], rows,
          "Section 5: G7 split-plot MDE on the log of cell means (two-sided 0.05, power 0.8, noncentral t; df 2(n-1), n-1)")
    table(["seeds per cell", "CV_a", "value", "MDE of G (log)", "as a ratio"], rows2, "Whole-plot factor G with the pilot Par2 CVs")


# ----------------------------------------------------------------------------------------------------------------------
# runtime (section 6)
# ----------------------------------------------------------------------------------------------------------------------
def parse_sf39_log(path):
    """{seed: {sec, it, nfev}} from an SF-39 per-arm raw log (FIELD, B_SOLVER, TSTATUS pass=plane, TIMING lines)."""
    d = {}
    with open(path, errors="replace") as fh:
        for line in fh:
            m = re.match(r"\s*R=(\d+) FIELD .* seed=(\d+)", line)
            if m:
                d.setdefault(int(m.group(1)), {})["seed"] = int(m.group(2))
                continue
            m = re.match(r"\s*R=(\d+) B_SOLVER .*iterations=(\d+)", line)
            if m:
                d.setdefault(int(m.group(1)), {})["it"] = int(m.group(2))
                continue
            m = re.match(r"\s*R=(\d+) TSTATUS pass=plane .*nfev=(\d+)", line)
            if m:
                d.setdefault(int(m.group(1)), {})["nfev"] = int(m.group(2))
                continue
            m = re.match(r"\s*TIMING r=(\d+) seconds=([0-9.]+)", line)
            if m:
                d.setdefault(int(m.group(1)), {})["sec"] = float(m.group(2))
    return {v["seed"]: v for v in d.values() if "seed" in v and "sec" in v}


def cmd_runtime(args):
    need(args, "pilots")
    pil = os.path.abspath(args.pilots)
    ests = (("alpha_LP", "flux", "pos"), ("alpha_LP", "unif", "pos"), ("alpha_TP", "flux", "disp"))
    zs, rows = {}, []
    for job in ("J2", "J3"):
        for rid in sorted(os.listdir(os.path.join(pil, job))):
            if not rid.startswith("c_"):
                continue
            run = read_json(os.path.join(pil, job, rid, "RUN.json"))
            red = read_csv(os.path.join(pil, job, "reduced", rid + ".csv"))
            rr = [r for r in run["realizations"] if r.get("done")]
            sec = np.array([r["seconds"]["total"] for r in rr])
            trk = np.array([r["seconds"]["track"] for r in rr])
            it = np.array([r["iterations"] for r in rr], dtype=float)
            idx = [int(r["r"].split("_")[1]) for r in rr]
            cells = []
            for est, w, b in ests:
                v = plane_values(red, "consumer", w, b, est)
                y = np.array([v[i] for i in idx])
                for nm, x in (("seconds", sec), ("track seconds", trk), ("PCG iterations", it)):
                    if np.ptp(x) > 0:
                        rho = spearman(x, y)
                        if math.isfinite(rho):
                            zs.setdefault((est, w, b, nm), []).append((fisher_z(rho), len(y)))
                cells.append("%+.2f / %+.2f" % (spearman(sec, y), spearman(trk, y)))
            rows.append((job + "/" + rid, len(rr), "%.1f-%.1f" % (sec.min(), sec.max()), ",".join(str(int(v)) for v in sorted(set(it))),
                         *cells))
    table(["run", "n", "seconds", "PCG iterations"] + ["%s %s %s: rho(seconds) / rho(track)" % e for e in ests], rows,
          "Section 6: SF-42 pilot consumer runs (RUN.json realizations[], reduced tables), Spearman per run")
    table(["estimand", "weights", "basis", "runtime quantity", "pooled rho", "95 % CI", "runs"],
          [k[:4] + ("%+.3f" % p[0], "[%+.3f, %+.3f]" % (p[1], p[2]), len(v)) for k, v in sorted(zs.items()) for p in [pooled_fisher(v)]],
          "Pooled Fisher z (weights n - 3); the three J2 lattices of an amplitude share their 6 fields, so the CIs are too narrow")

    sf39 = os.path.abspath(args.sf39) if args.sf39 else os.path.normpath(os.path.join(pil, "..", "..", "2026-10-27-sf39-macrodispersion-study"))
    missing = [p for logs, _ in SF39_GROUPS.values() for p in logs if not os.path.isfile(os.path.join(sf39, p))]
    if missing or not os.path.isdir(os.path.join(sf39, "analysis", "reduced")):
        print("SF-39 per-arm raw logs not found under %s (%d of the expected files missing): the SF-39 runtime tables are "
              "skipped." % (sf39, len(missing)))
        return
    zs, rows, slow = {}, [], []
    for g, (logs, red_name) in SF39_GROUPS.items():
        t = {}
        for p in logs:
            t.update(parse_sf39_log(os.path.join(sf39, p)))
        red = [r for r in read_csv(os.path.join(sf39, "analysis", "reduced", red_name)) if r["method"] == "flux"]
        seeds = [int(r["seed"]) for r in red if int(r["seed"]) in t]
        nf = np.array([t[s].get("nfev", np.nan) for s in seeds], dtype=float)
        cells = []
        for nm, key in (("seconds", "sec"), ("tracker nfev", "nfev"), ("PCG iterations", "it")):
            x = np.array([t[s].get(key, np.nan) for s in seeds], dtype=float)
            out = []
            for lab, col in SF39_COLS:
                y = np.array([float(r[col]) for r in red if int(r["seed"]) in t])
                rho = spearman(x, y)
                zs.setdefault((nm, lab), []).append((fisher_z(rho), len(seeds)))
                out.append("%+.2f" % rho)
            cells.append(" / ".join(out))
        secs = [t[s]["sec"] for s in seeds]
        med = float(np.median(secs))
        rows.append((g, "%d of %d" % (len(seeds), len(red)), "%.1f-%.1f" % (min(secs), max(secs)),
                     ",".join(str(v) for v in sorted({t[s].get("it") for s in seeds})), "%.3f" % (np.nanstd(nf) / np.nanmean(nf)), *cells))
        for lab, col in SF39_COLS:
            v = np.array([float(r[col]) for r in red])
            rk = stats.rankdata(v)
            for i, r in enumerate(red):
                s = int(r["seed"])
                if s in t and t[s]["sec"] > 3.0 * med:
                    slow.append((g, s, t[s].get("it"), "%.1f (median %.1f)" % (t[s]["sec"], med), lab, "%.4g" % v[i],
                                 "%d of %d" % (int(rk[i]), v.size)))
    table(["arm", "matched", "seconds", "PCG iterations", "nfev CV"] + ["rho(%s, alpha_TP / alpha_LAM / alpha_TAM)" % nm
                                                                      for nm in ("seconds", "nfev", "PCG it")], rows,
          "Section 6: SF-39 sigma^2 = 4 arms with per-realization TIMING lines (flux rows of analysis/reduced), Spearman")
    table(["runtime quantity", "estimator", "pooled rho", "95 % CI", "rows"],
          [k + ("%+.3f" % p[0], "[%+.3f, %+.3f]" % (p[1], p[2]), sum(n for _, n in v)) for k, v in sorted(zs.items())
           for p in [pooled_fisher(v)]],
          "Pooled over the 6 arms (Fisher z, weights n - 3); n16 / n32 / n64 are rungs of the same 20 fields, so the CIs are "
          "too narrow")
    table(["arm", "seed", "PCG iterations", "seconds", "estimator", "value", "rank"], slow,
          "SF-39 realizations slower than 3 x the median of their arm (fallback solves) and the rank of their alpha in the arm")


# ----------------------------------------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="SF-43 pre-activation review: Gate 5 statistics (read-only, CPU-only).")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sf42-scripts", dest="sf42_scripts", help="SF-42 artifact scripts/ directory (contains mcstat/)")
    common.add_argument("--pilots", help="SF-42 artifact pilots/ directory")
    common.add_argument("--coverage", help="SF-42 artifact coverage/ directory (summary.json)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("moments", cmd_moments), ("fieller-coverage", cmd_fieller), ("skewcorr", cmd_skewcorr),
                     ("sizing", cmd_sizing), ("g7", cmd_g7), ("runtime", cmd_runtime)):
        p = sub.add_parser(name, parents=[common])
        p.set_defaults(fn=fn)
        if name == "skewcorr":
            p.add_argument("--R", type=int, default=200000, help="replicates per (distribution, n); 200000 reproduces the report")
        if name == "runtime":
            p.add_argument("--sf39", help="SF-39 artifact root (default: next to the SF-42 artifact)")
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
