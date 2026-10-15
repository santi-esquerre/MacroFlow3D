#!/usr/bin/env python3
"""SF-32 spurious-spreading experiment: analysis of the return-map ladders.

Usage:
    python3 analyze.py <out-root> [--fields a,b] [--figures-dir <dir>]
    python3 analyze.py --selftest

<out-root> holds one directory per label field, written by `run_ladders.sh` and the
`spurious_spreading return-map` instrument:

    <out-root>/<field>/labels.json
    <out-root>/<field>/<tracker>_<level_tag>/{seeds.csv, summary.json}

Runs are identified from the CONTENT of `summary.json` (`tracker`, `level`), never from the
directory name. Every statistic is recomputed from `seeds.csv` (population variance over the
`status == 0` rows) and compared with the summary's own values (consistency check); the
recomputed values are the ones used in the fits.

Outputs (into <out-root>/analysis/, or --figures-dir):
    tables.md                              per-field tables + cross-field exponent table
    exponents.json                         fits, two-level estimates, verdicts, PS drift
    histograms/<field>_<tracker>.json      binned counts of delta_x2, delta_x3 per level
    fig3b_<field>, fig3c_<field>, fig4a_<field>, fig4b_<field>, ps_drift_<field>,
    exponents_crossfield                   PNG (150 dpi) and PDF, analogues of Lester 2023
                                           figs. 3b, 3c, 4a, 4b

The rules are pre-registered (SF-32 orchestration record, sections 3.4-3.6, 5, 6) and live in
the single constant block below; they are applied mechanically. Nothing here may be tuned to a
result. Missing numbers or malformed files are reported under "Issues" and the analysis goes on.

Python 3.8+, numpy and matplotlib only. Output is deterministic (sorted, no timestamps).
"""

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# =====================================================================================
# PRE-REGISTERED RULES -- SF-32 (orchestration record 3.6, 5 T1-T3); do not edit after the
# first result. These are the ONLY constants of the verdicts.
POLLOCK_BAND = (1.7, 2.3)          # variance ~ Delta^p, spec p = 2 +- 0.3, fit over Delta/Delta0 in {1,2,4,8}
RK_BAND = (0.35, 0.65)             # variance ~ tol^p, spec p = 0.5 +- 0.15, fit over the paper's tol in {1e-4..1e-8}
RK_PAPER_TOLS = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
PS_DRIFT_FACTOR = 10.0             # max |delta_psi_i| <= 10 tol_psi
MIN_OK_FRACTION = 0.99             # a level enters a fit only if n_ok / n_seeds >= 0.99
HIST_BINS = 101
HIST_RANGE = {"pollock": 0.75, "rk": 0.04, "pseudo_symplectic": 1e-9}   # symmetric [-r, r], the paper's axes
POLLOCK_RATIOS = (1, 2, 4, 8)      # Delta/Delta0 levels of the Pollock fit (spec: >= 3 grids)
POLLOCK_MIN_LEVELS = 3             # fewer qualifying levels -> insufficient
RK_MIN_LEVELS = 4                  # fewer qualifying levels -> insufficient (spec: >= 4 tolerances)
AUTO_RANGE_SIGMAS = 5.0            # auto PDF panel: +- 5 sigma of the finest / tightest level
# =====================================================================================

TRACKERS = ("pollock", "rk", "pseudo_symplectic")
COMPONENTS = ("delta_x2", "delta_x3")
STAT_COLUMNS = ("delta_x2", "delta_x3", "delta_psi1", "delta_psi2")
CSV_COLUMNS = ("seed_index", "x2_0", "x3_0", "status", "delta_x2", "delta_x3",
               "delta_psi1", "delta_psi2", "tau", "count", "land_err")
LEVEL_KEY = {"pollock": "delta_ratio", "rk": "tol", "pseudo_symplectic": "tol_psi"}
O3_LABEL = "protocol numbers of the surrogate, not coefficients (O3)"

PALETTE = plt.get_cmap("tab10").colors
MARKERS = ("o", "s", "^", "D", "v", "P", "X", "*")
PNG_META = {"Software": None}
PDF_META = {"CreationDate": None, "ModDate": None}


# ------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------
def fmt(x, spec="%.4g"):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, np.integer)):
        return "%d" % x
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not math.isfinite(xf):
        return "n/a" if math.isnan(xf) else ("inf" if xf > 0 else "-inf")
    return spec % xf


def finite_or_none(x):
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return xf if math.isfinite(xf) else None


def jsonable(obj):
    """Strict-JSON copy: non-finite floats -> None, numpy scalars -> python."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return finite_or_none(obj)
    return obj


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(jsonable(obj), f, indent=1, sort_keys=True, allow_nan=False)
        f.write("\n")


def level_value(tracker, level):
    try:
        return float(level[LEVEL_KEY[tracker]])
    except (KeyError, TypeError, ValueError):
        return None


def level_label(tracker, x):
    if x is None:
        return "?"
    if tracker == "pollock":
        return "Delta/Delta0=%g" % x
    if tracker == "rk":
        return "tol=%.0e" % x
    return "tol_psi=%.0e" % x


def same_value(a, b):
    return a > 0 and b > 0 and abs(math.log10(a) - math.log10(b)) < 1e-9


# ------------------------------------------------------------------------------------
# reading
# ------------------------------------------------------------------------------------
def read_seeds(path, issues, tag):
    """Return dict column -> numpy array (floats; status/seed_index/count as float too)."""
    with open(path, newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            issues.append("%s: seeds.csv is empty" % tag)
            return None
        header = [h.strip() for h in header]
        if tuple(header) != CSV_COLUMNS:
            issues.append("%s: seeds.csv header %s differs from the schema %s" % (tag, header, list(CSV_COLUMNS)))
            missing = [c for c in CSV_COLUMNS if c not in header]
            if missing:
                issues.append("%s: seeds.csv lacks columns %s; run skipped" % (tag, missing))
                return None
        idx = {c: header.index(c) for c in CSV_COLUMNS}
        cols = {c: [] for c in CSV_COLUMNS}
        bad = 0
        for lineno, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) != len(header):
                bad += 1
                continue
            try:
                vals = {c: float(row[idx[c]]) for c in CSV_COLUMNS}
            except ValueError:
                bad += 1
                continue
            for c in CSV_COLUMNS:
                cols[c].append(vals[c])
        if bad:
            issues.append("%s: %d malformed seeds.csv row(s) skipped" % (tag, bad))
    return {c: np.asarray(v, dtype=np.float64) for c, v in cols.items()}


def compute_stats(seeds):
    """Recomputed statistics over status == 0 rows (population variance)."""
    ok = seeds["status"] == 0
    out = {"n_seeds": int(seeds["status"].size), "n_ok": int(np.count_nonzero(ok)), "nonfinite_ok": {}}
    codes, counts = np.unique(seeds["status"].astype(np.int64), return_counts=True)
    out["status_counts"] = {str(int(c)): int(n) for c, n in zip(codes, counts)}
    for c in STAT_COLUMNS + ("tau", "land_err"):
        a = seeds[c][ok]
        fin = np.isfinite(a)
        if not np.all(fin):
            out["nonfinite_ok"][c] = int(np.count_nonzero(~fin))
        a = a[fin]
        if a.size == 0:
            out[c] = {"mean": None, "var": None, "rms": None, "max_abs": None, "min": None, "max": None}
            continue
        out[c] = {
            "mean": float(np.mean(a)),
            "var": float(np.var(a)),
            "rms": float(np.sqrt(np.mean(a * a))),
            "max_abs": float(np.max(np.abs(a))),
            "min": float(np.min(a)),
            "max": float(np.max(a)),
        }
    mt = out["tau"]["mean"]
    for c, key in (("delta_x2", "D22_uniform"), ("delta_x3", "D33_uniform")):
        v = out[c]["var"]
        out[key] = (v / (2.0 * mt)) if (v is not None and mt) else None
    return out


def load_root(root, fields_filter, issues_global):
    fields = {}
    for name in sorted(os.listdir(root)):
        fdir = os.path.join(root, name)
        if not os.path.isdir(fdir) or name == "analysis":
            continue
        if fields_filter and name not in fields_filter:
            continue
        issues = []
        labels = None
        lpath = os.path.join(fdir, "labels.json")
        if os.path.isfile(lpath):
            try:
                with open(lpath) as f:
                    labels = json.load(f)
            except (ValueError, OSError) as e:
                issues.append("labels.json unreadable: %s" % e)
        else:
            issues.append("labels.json missing")
        fpath = os.path.join(fdir, "failures.txt")
        if os.path.isfile(fpath):
            with open(fpath) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        issues.append("launcher failure: %s" % line)
        runs = {t: [] for t in TRACKERS}
        for sub in sorted(os.listdir(fdir)):
            rdir = os.path.join(fdir, sub)
            if not os.path.isdir(rdir):
                continue
            spath = os.path.join(rdir, "summary.json")
            cpath = os.path.join(rdir, "seeds.csv")
            if not os.path.isfile(spath):
                issues.append("%s: summary.json missing" % sub)
                continue
            try:
                with open(spath) as f:
                    summary = json.load(f)
            except (ValueError, OSError) as e:
                issues.append("%s: summary.json unreadable: %s" % (sub, e))
                continue
            tracker = summary.get("tracker")
            if tracker not in TRACKERS:
                issues.append("%s: unknown tracker %r; run skipped" % (sub, tracker))
                continue
            level = summary.get("level") or {}
            x = level_value(tracker, level)
            if x is None:
                issues.append("%s: summary.level lacks %s; run skipped" % (sub, LEVEL_KEY[tracker]))
                continue
            if not os.path.isfile(cpath):
                issues.append("%s: seeds.csv missing; run skipped" % sub)
                continue
            seeds = read_seeds(cpath, issues, sub)
            if seeds is None:
                continue
            st = compute_stats(seeds)
            for c, n in sorted(st["nonfinite_ok"].items()):
                issues.append("%s: %d status-0 row(s) with non-finite %s (excluded from that statistic)" % (sub, n, c))
            if any(same_value(x, r["x"]) for r in runs[tracker]):
                issues.append("%s: duplicate %s level %g; run skipped" % (sub, tracker, x))
                continue
            runs[tracker].append({"dir": sub, "tracker": tracker, "x": x, "level": level,
                                  "summary": summary, "seeds": seeds, "stats": st})
        for t in TRACKERS:
            runs[t].sort(key=lambda r: -r["x"])          # coarsest / loosest first
        fields[name] = {"labels": labels, "runs": runs, "issues": issues}
    if not fields:
        issues_global.append("no field directories found under %s" % root)
    return fields


# ------------------------------------------------------------------------------------
# consistency check (recomputed vs summary)
# ------------------------------------------------------------------------------------
def consistency(run, issues):
    s = run["summary"]
    st = run["stats"]
    tag = run["dir"]
    res = {}
    sst = s.get("stats") or {}

    def diff(a, b):
        a, b = finite_or_none(a), finite_or_none(b)
        if a is None or b is None:
            return None
        return abs(a - b)

    for c in STAT_COLUMNS:
        d = []
        for k in ("mean", "var", "rms", "max_abs"):
            v = diff(st[c][k], (sst.get(c) or {}).get(k))
            if v is None:
                issues.append("%s: summary stats.%s.%s missing or non-finite" % (tag, c, k))
            else:
                d.append(v)
        res[c] = max(d) if d else None
    d = []
    for k in ("mean", "var", "min", "max"):
        v = diff(st["tau"][k], (sst.get("tau") or {}).get(k))
        if v is None:
            issues.append("%s: summary stats.tau.%s missing or non-finite" % (tag, k))
        else:
            d.append(v)
    res["tau"] = max(d) if d else None
    d = []
    for k in ("D22_uniform", "D33_uniform"):
        v = diff(st[k], s.get(k))
        if v is None:
            issues.append("%s: summary %s missing or non-finite" % (tag, k))
        else:
            d.append(v)
    res["D_uniform"] = max(d) if d else None
    for k in ("n_seeds", "n_ok"):
        if s.get(k) is None:
            issues.append("%s: summary %s missing" % (tag, k))
            res[k] = None
        else:
            res[k] = abs(int(s[k]) - st[k])
            if res[k]:
                issues.append("%s: summary %s=%s but seeds.csv gives %d" % (tag, k, s[k], st[k]))
    sc = s.get("status_counts")
    if isinstance(sc, dict) and {str(k): int(v) for k, v in sc.items()} != st["status_counts"]:
        issues.append("%s: summary status_counts %s differ from seeds.csv %s" % (tag, sc, st["status_counts"]))
    return res


# ------------------------------------------------------------------------------------
# fits and verdicts
# ------------------------------------------------------------------------------------
def ok_fraction(run):
    n = run["stats"]["n_seeds"]
    return run["stats"]["n_ok"] / n if n else 0.0


def qualifies(run, comp):
    v = run["stats"][comp]["var"]
    return ok_fraction(run) >= MIN_OK_FRACTION and v is not None and v > 0


def fit_x(run):
    """Abscissa of the fit: Delta for pollock (summary level.delta), tol for rk."""
    if run["tracker"] == "pollock":
        d = finite_or_none(run["level"].get("delta"))
        return d
    return run["x"]


def ols(xs, ys):
    X = np.log(np.asarray(xs, dtype=np.float64))
    Y = np.log(np.asarray(ys, dtype=np.float64))
    xm, ym = X.mean(), Y.mean()
    sxx = float(np.sum((X - xm) ** 2))
    if sxx == 0.0:
        return None
    slope = float(np.sum((X - xm) * (Y - ym)) / sxx)
    intercept = float(ym - slope * xm)
    res = Y - (intercept + slope * X)
    sst = float(np.sum((Y - ym) ** 2))
    r2 = 1.0 - float(np.sum(res * res)) / sst if sst > 0 else None
    return {"slope": slope, "intercept": intercept, "r2": r2, "n_points": int(X.size)}


def exponent_analysis(runs, comp, band, min_levels, select, issues, what):
    """Fit log(var) on log(x) over the selected levels; verdict and monotonicity."""
    levels, pts = [], []
    for r in runs:
        if not select(r):
            continue
        q = qualifies(r, comp)
        xf = fit_x(r)
        if xf is None:
            issues.append("%s: no fit abscissa (level.delta) for %s; level excluded" % (r["dir"], what))
            q = False
        reason = None
        if not q:
            if ok_fraction(r) < MIN_OK_FRACTION:
                reason = "n_ok/n_seeds = %.4f < %g" % (ok_fraction(r), MIN_OK_FRACTION)
            elif xf is None:
                reason = "no abscissa"
            else:
                reason = "variance missing or <= 0"
        levels.append({"dir": r["dir"], "x": r["x"], "fit_x": xf, "var": r["stats"][comp]["var"],
                       "ok_fraction": ok_fraction(r), "qualifies": bool(q), "excluded_reason": reason})
        if q:
            pts.append((xf, r["stats"][comp]["var"]))
    # runs are sorted coarsest/loosest first -> pts in decreasing x
    fit = ols([p[0] for p in pts], [p[1] for p in pts]) if len(pts) >= 2 else None
    two_level = []
    for (xa, va), (xb, vb) in zip(pts[:-1], pts[1:]):
        two_level.append({"x_a": xa, "x_b": xb, "slope": math.log(va / vb) / math.log(xa / xb)})
    non_monotone = any(vb >= va for (_, va), (_, vb) in zip(pts[:-1], pts[1:]))
    if len(pts) < min_levels or fit is None:
        verdict = "insufficient"
    elif band[0] <= fit["slope"] <= band[1]:
        verdict = "in_band"
    else:
        verdict = "out_of_band"
    return {"band": list(band), "min_levels": min_levels, "n_qualifying": len(pts), "fit": fit,
            "two_level": two_level, "non_monotone": bool(non_monotone), "verdict": verdict,
            "levels": levels}


def ps_analysis(runs, issues):
    out = []
    for r in runs:
        st = r["stats"]
        tol = r["x"]
        m1, m2 = st["delta_psi1"]["max_abs"], st["delta_psi2"]["max_abs"]
        if m1 is None or m2 is None:
            drift = None
            verdict = "fail"
            issues.append("%s: no finite delta_psi on status-0 rows; PS verdict fail" % r["dir"])
        else:
            drift = max(m1, m2)
            verdict = "pass" if drift <= PS_DRIFT_FACTOR * tol else "fail"
        if st["nonfinite_ok"].get("delta_psi1") or st["nonfinite_ok"].get("delta_psi2"):
            verdict = "fail"
        e = {"dir": r["dir"], "tol_psi": tol, "n_ok": st["n_ok"], "n_seeds": st["n_seeds"],
             "all_seeds_ok": st["n_ok"] == st["n_seeds"],
             "max_abs_delta_psi": drift, "drift_over_tol": (drift / tol) if drift is not None else None,
             "bound": PS_DRIFT_FACTOR * tol, "verdict": verdict}
        for c in ("delta_psi1", "delta_psi2", "delta_x2", "delta_x3"):
            e[c + "_rms"] = st[c]["rms"]
            e[c + "_max_abs"] = st[c]["max_abs"]
        for c in ("delta_x2", "delta_x3"):
            e[c + "_rms_over_tol"] = st[c]["rms"] / tol if st[c]["rms"] is not None else None
            e[c + "_max_over_tol"] = st[c]["max_abs"] / tol if st[c]["max_abs"] is not None else None
        out.append(e)
    return out


def analyze_field(name, fd):
    issues = fd["issues"]
    runs = fd["runs"]
    for t in TRACKERS:
        if not runs[t]:
            issues.append("no %s runs" % t)
        for r in runs[t]:
            r["consistency"] = consistency(r, issues)
    res = {"pollock": {}, "rk": {"paper": {}, "all": {}}}
    for comp in COMPONENTS:
        res["pollock"][comp] = exponent_analysis(
            runs["pollock"], comp, POLLOCK_BAND, POLLOCK_MIN_LEVELS,
            lambda r: any(abs(r["x"] - m) < 1e-12 for m in POLLOCK_RATIOS), issues, "pollock")
        res["rk"]["paper"][comp] = exponent_analysis(
            runs["rk"], comp, RK_BAND, RK_MIN_LEVELS,
            lambda r: any(same_value(r["x"], t) for t in RK_PAPER_TOLS), issues, "rk")
        res["rk"]["all"][comp] = exponent_analysis(
            runs["rk"], comp, RK_BAND, RK_MIN_LEVELS, lambda r: True, issues, "rk")
    res["pseudo_symplectic"] = ps_analysis(runs["pseudo_symplectic"], issues)
    eq36 = []
    for t in TRACKERS:
        for r in runs[t]:
            s = r["summary"]
            eq36.append({"tracker": t, "x": r["x"], "dir": r["dir"],
                         "D22_uniform": r["stats"]["D22_uniform"], "D33_uniform": r["stats"]["D33_uniform"],
                         "D22_flux_summary": finite_or_none(s.get("D22_flux")),
                         "D33_flux_summary": finite_or_none(s.get("D33_flux")),
                         "mean_tau": r["stats"]["tau"]["mean"],
                         "flux_weighted_tau_mean_summary": finite_or_none(s.get("flux_weighted_tau_mean"))})
    res["eq36"] = eq36
    return res


# ------------------------------------------------------------------------------------
# histograms
# ------------------------------------------------------------------------------------
def histogram(a, r):
    edges = np.linspace(-r, r, HIST_BINS + 1)
    counts, _ = np.histogram(a, bins=edges)
    return edges, counts, int(np.count_nonzero(a < -r)), int(np.count_nonzero(a > r))


def ok_values(run, comp):
    ok = run["seeds"]["status"] == 0
    a = run["seeds"][comp][ok]
    return a[np.isfinite(a)]


def auto_range(runs):
    """+- AUTO_RANGE_SIGMAS sigma of the finest / tightest level (last run), max over components."""
    if not runs:
        return None
    st = runs[-1]["stats"]
    sig = [math.sqrt(st[c]["var"]) for c in COMPONENTS if st[c]["var"] is not None]
    sig = [s for s in sig if s > 0]
    return AUTO_RANGE_SIGMAS * max(sig) if sig else None


def histograms_json(name, tracker, runs):
    r_fixed = HIST_RANGE[tracker]
    r_auto = auto_range(runs)
    out = {"field": name, "tracker": tracker, "bins": HIST_BINS,
           "range_fixed": [-r_fixed, r_fixed], "range_auto": [-r_auto, r_auto] if r_auto else None,
           "range_auto_rule": "+-%g sigma of the finest/tightest level" % AUTO_RANGE_SIGMAS, "levels": []}
    for r in runs:
        e = {"dir": r["dir"], LEVEL_KEY[tracker]: r["x"], "n_ok": r["stats"]["n_ok"]}
        for c in COMPONENTS:
            a = ok_values(r, c)
            _, cnt, lo, hi = histogram(a, r_fixed)
            e[c] = {"fixed": {"counts": cnt.tolist(), "underflow": lo, "overflow": hi}}
            if r_auto:
                _, cnt, lo, hi = histogram(a, r_auto)
                e[c]["auto"] = {"counts": cnt.tolist(), "underflow": lo, "overflow": hi}
        out["levels"].append(e)
    return out


# ------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------
def savefig(fig, outdir, stem, written):
    png = os.path.join(outdir, stem + ".png")
    pdf = os.path.join(outdir, stem + ".pdf")
    fig.savefig(png, dpi=150, metadata=PNG_META)
    fig.savefig(pdf, metadata=PDF_META)
    plt.close(fig)
    written.extend([png, pdf])


def title_of(name, labels):
    route = (labels or {}).get("route", "?")
    return "%s (label route: %s)" % (name, route)


def plot_pdfs(ax, runs, tracker, r):
    edges = np.linspace(-r, r, HIST_BINS + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    for k, run in enumerate(runs):
        n = run["stats"]["n_ok"]
        if n == 0:
            continue
        col = PALETTE[k % len(PALETTE)]
        mk = MARKERS[k % len(MARKERS)]
        for comp, ls in (("delta_x2", "--"), ("delta_x3", "-")):
            _, cnt, _, _ = histogram(ok_values(run, comp), r)
            pdf = cnt / (n * width)
            pdf = np.where(pdf > 0, pdf, np.nan)
            ax.plot(centres, pdf, ls=ls, color=col, marker=mk, markevery=10, ms=4, lw=1.2,
                    label="%s, %s" % (level_label(tracker, run["x"]), comp))
    ax.set_yscale("log")
    ax.set_xlim(-r, r)
    ax.set_ylabel("PDF [1/L]")
    ax.grid(True, which="major", alpha=0.3)


def fig_pdf_pair(name, labels, runs, tracker, stem, xlabel, outdir, written):
    if not runs:
        return
    r_auto = auto_range(runs)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    plot_pdfs(axes[0], runs, tracker, HIST_RANGE[tracker])
    axes[0].set_title("fixed range (paper's axis)")
    if r_auto:
        plot_pdfs(axes[1], runs, tracker, r_auto)
        axes[1].set_title("auto range: +-%g sigma of the finest level" % AUTO_RANGE_SIGMAS)
    else:
        axes[1].text(0.5, 0.5, "no finite variance", ha="center", transform=axes[1].transAxes)
    for ax in axes:
        ax.set_xlabel(xlabel)
    axes[0].legend(fontsize=7, ncol=1)
    fig.suptitle("%s: %s; dashed delta_x2, solid delta_x3" % (title_of(name, labels), stem.split("_")[0]))
    fig.tight_layout()
    savefig(fig, outdir, "%s_%s" % (stem, name), written)


def fig_variance(name, labels, runs, ex, tracker, outdir, written):
    if not runs:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for comp, col, mk, ls in (("delta_x2", PALETTE[0], "o", "--"), ("delta_x3", PALETTE[3], "s", "-")):
        an = ex[comp]
        plot_x = [r["x"] for r in runs]          # Delta/Delta0 (pollock) or tol (rk)
        fx = {r["dir"]: r["x"] for r in runs}
        vv = [r["stats"][comp]["var"] for r in runs]
        if tracker == "rk":
            paper = [any(same_value(x, t) for t in RK_PAPER_TOLS) for x in plot_x]
            xs_p = [x for x, p, v in zip(plot_x, paper, vv) if p and v]
            vs_p = [v for x, p, v in zip(plot_x, paper, vv) if p and v]
            xs_e = [x for x, p, v in zip(plot_x, paper, vv) if not p and v]
            vs_e = [v for x, p, v in zip(plot_x, paper, vv) if not p and v]
            ax.plot(xs_p, vs_p, ls="none", marker=mk, color=col, label="var(%s), paper's tol" % comp)
            ax.plot(xs_e, vs_e, ls="none", marker=mk, mfc="none", color=col, label="var(%s), extra tol" % comp)
        else:
            ax.plot([x for x, v in zip(plot_x, vv) if v], [v for v in vv if v], ls="none", marker=mk,
                    color=col, label="var(%s)" % comp)
        excl = [(r["x"], v) for r, v in zip(runs, vv) if v and v > 0 and ok_fraction(r) < MIN_OK_FRACTION]
        if excl:
            ax.plot([e[0] for e in excl], [e[1] for e in excl], ls="none", marker="x", ms=11, color="0.2",
                    label="excluded from fit (n_ok/n_seeds < %g), %s" % (MIN_OK_FRACTION, comp))
        fit = an["fit"]
        if fit:
            q = [lv for lv in an["levels"] if lv["qualifies"]]
            xq = np.array([fx[lv["dir"]] for lv in q])
            fq = np.array([lv["fit_x"] for lv in q])
            ax.plot(xq, np.exp(fit["intercept"]) * fq ** fit["slope"], ls=ls, color=col,
                    label="fit %s: p = %.3f (%s)" % (comp, fit["slope"], an["verdict"]))
    # reference line
    if tracker == "pollock":
        ref = [r for r in runs if abs(r["x"] - 1.0) < 1e-12]
        slope, xlab, rlab = 2.0, "Delta/Delta0 (Delta0 = finest label-grid spacing)", "slope 2 through Delta/Delta0 = 1"
        stem = "fig3c"
    else:
        ref = [r for r in runs if same_value(r["x"], 1e-4)]
        slope, xlab, rlab = 0.5, "RK tolerance tol [-]", "slope 1/2 through tol = 1e-4"
        stem = "fig4b"
    if ref and ref[0]["stats"]["delta_x3"]["var"]:
        x0, v0 = ref[0]["x"], ref[0]["stats"]["delta_x3"]["var"]
        xr = np.array(sorted(r["x"] for r in runs))
        ax.plot(xr, v0 * (xr / x0) ** slope, ls=":", color="0.3", label=rlab + " (var(delta_x3))")
    ax.set_xscale("log")
    ax.set_yscale("log")
    if tracker == "pollock":
        ticks = sorted(r["x"] for r in runs)
        ax.set_xticks(ticks)
        ax.set_xticks([], minor=True)
        ax.set_xticklabels(["%g" % t for t in ticks])
    ax.set_xlabel(xlab)
    ax.set_ylabel("variance of the one-period displacement [L^2]")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("%s: %s" % (title_of(name, labels), stem), fontsize=9)
    fig.tight_layout()
    savefig(fig, outdir, "%s_%s" % (stem, name), written)


def fig_ps(name, labels, ps, outdir, written):
    if not ps:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    tol = np.array([e["tol_psi"] for e in ps])
    series = [("max |delta_psi_i|", "max_abs_delta_psi", PALETTE[0], "o", "-"),
              ("RMS delta_psi1", "delta_psi1_rms", PALETTE[1], "s", "--"),
              ("RMS delta_psi2", "delta_psi2_rms", PALETTE[2], "^", "--"),
              ("RMS delta_x2 [L]", "delta_x2_rms", PALETTE[3], "D", "-."),
              ("RMS delta_x3 [L]", "delta_x3_rms", PALETTE[4], "v", "-.")]
    for lab, key, col, mk, ls in series:
        y = np.array([e[key] if (e[key] is not None and e[key] > 0) else np.nan for e in ps], dtype=float)
        ax.plot(tol, y, ls=ls, marker=mk, color=col, label=lab)
    order = np.argsort(tol)
    ax.plot(tol[order], PS_DRIFT_FACTOR * tol[order], ls=":", color="0.3", label="%g tol_psi (T3 bound)" % PS_DRIFT_FACTOR)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("pseudo-symplectic label tolerance tol_psi [label units]")
    ax.set_ylabel("one-period drift [label units] / displacement [L]")
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(fontsize=7)
    ax.set_title("%s: pseudo-symplectic drift" % title_of(name, labels), fontsize=9)
    fig.tight_layout()
    savefig(fig, outdir, "ps_drift_%s" % name, written)


def fig_crossfield(results, outdir, written):
    names = sorted(results)
    if not names:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, tracker, band, title in ((axes[0], "pollock", POLLOCK_BAND, "Pollock: var ~ Delta^p"),
                                     (axes[1], "rk", RK_BAND, "RK: var ~ tol^p (paper's five tol)")):
        ax.axhspan(band[0], band[1], color="0.85", label="pre-registered band [%g, %g]" % band)
        for j, (comp, mk, col) in enumerate((("delta_x2", "o", PALETTE[0]), ("delta_x3", "s", PALETTE[3]))):
            xs, ys = [], []
            for i, n in enumerate(names):
                an = results[n]["pollock"][comp] if tracker == "pollock" else results[n]["rk"]["paper"][comp]
                if an["fit"] is not None:
                    xs.append(i + (j - 0.5) * 0.15)
                    ys.append(an["fit"]["slope"])
                if an["verdict"] == "insufficient":
                    ax.annotate("insufficient", (i, band[1]), fontsize=6, ha="center", va="bottom")
            ax.plot(xs, ys, ls="none", marker=mk, color=col, label="p of var(%s)" % comp)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=7)
        ax.set_ylabel("fitted variance exponent p [-]")
        ax.set_title(title, fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("SF-32 fitted exponents across label fields")
    fig.tight_layout()
    savefig(fig, outdir, "exponents_crossfield", written)


# ------------------------------------------------------------------------------------
# tables
# ------------------------------------------------------------------------------------
def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def nonok_text(st):
    return ", ".join("%s:%d" % (k, v) for k, v in sorted(st["status_counts"].items()) if k != "0") or "-"


def field_tables(name, fd, res):
    L = ["## Field `%s`" % name, ""]
    lab = fd["labels"] or {}
    L.append("### 1. Label metadata (`labels.json`)")
    L.append("")
    rows = [[k, "`%s`" % json.dumps(lab[k], sort_keys=True)] for k in sorted(lab)]
    L.append(md_table(["key", "value"], rows) if rows else "(none)")
    L.append("")
    L.append("### 2. Runs (recomputed from `seeds.csv`, status 0 rows, population variance)")
    L.append("")
    hdr = ["tracker", "level", "n_ok/n_seeds", "non-ok", "mean dx2", "var dx2", "rms dx2", "max abs dx2",
           "mean dx3", "var dx3", "rms dx3", "max abs dx3", "rms dpsi1", "max abs dpsi1", "rms dpsi2", "max abs dpsi2",
           "mean tau", "flux mean tau (summary)", "max land_err", "div max rel", "wall [s]"]
    rows = []
    for t in TRACKERS:
        for r in fd["runs"][t]:
            st, s = r["stats"], r["summary"]
            land = finite_or_none((s.get("landing") or {}).get("max_err"))
            if land is None:
                land = st["land_err"]["max_abs"]
            rows.append([t, level_label(t, r["x"]), "%d/%d" % (st["n_ok"], st["n_seeds"]), nonok_text(st),
                         fmt(st["delta_x2"]["mean"]), fmt(st["delta_x2"]["var"]), fmt(st["delta_x2"]["rms"]),
                         fmt(st["delta_x2"]["max_abs"]), fmt(st["delta_x3"]["mean"]), fmt(st["delta_x3"]["var"]),
                         fmt(st["delta_x3"]["rms"]), fmt(st["delta_x3"]["max_abs"]),
                         fmt(st["delta_psi1"]["rms"]), fmt(st["delta_psi1"]["max_abs"]),
                         fmt(st["delta_psi2"]["rms"]), fmt(st["delta_psi2"]["max_abs"]),
                         fmt(st["tau"]["mean"], "%.8g"), fmt(s.get("flux_weighted_tau_mean"), "%.8g"),
                         fmt(land), fmt(s.get("divergence_max_rel")) if t == "pollock" else "-",
                         fmt(s.get("wall_seconds"), "%.1f")])
    L.append(md_table(hdr, rows))
    L.append("")
    L.append("`max land_err` is the summary's `landing.max_err` (seeds.csv maximum if absent).")
    L.append("")
    L.append("### 3. Exponents (OLS of log var on log x; Pollock x = Delta, RK x = tol)")
    L.append("")
    hdr = ["tracker / fit", "component", "levels used", "slope p", "intercept", "R^2", "two-level estimates",
           "band", "verdict", "non_monotone"]
    rows = []
    for tag, an_c in (("pollock (Delta/Delta0 = 1,2,4,8)", res["pollock"]),
                      ("rk (paper's five tol) [verdict]", res["rk"]["paper"]),
                      ("rk (all levels) [reported]", res["rk"]["all"])):
        for comp in COMPONENTS:
            an = an_c[comp]
            fit = an["fit"] or {}
            tl = ", ".join("%.3f" % e["slope"] for e in an["two_level"]) or "-"
            rows.append([tag, comp, "%d (min %d)" % (an["n_qualifying"], an["min_levels"]),
                         fmt(fit.get("slope"), "%.4f"), fmt(fit.get("intercept"), "%.4f"),
                         fmt(fit.get("r2"), "%.5f"), tl, "[%g, %g]" % tuple(an["band"]), an["verdict"],
                         fmt(an["non_monotone"])])
    L.append(md_table(hdr, rows))
    excl = []
    for tag, an_c in (("pollock", res["pollock"]), ("rk", res["rk"]["all"])):
        for comp in COMPONENTS:
            for lv in an_c[comp]["levels"]:
                if not lv["qualifies"]:
                    excl.append("- %s %s %s: excluded (%s)" % (tag, comp, level_label(tag, lv["x"]), lv["excluded_reason"]))
    if excl:
        L.append("")
        L.append("Excluded levels:")
        L.extend(excl)
    L.append("")
    L.append("Two-level estimates are `log(var_a/var_b)/log(x_a/x_b)` between consecutive qualifying "
             "levels, coarsest/loosest first. `non_monotone`: the variance does not decrease strictly "
             "with decreasing Delta/tol over the qualifying levels.")
    L.append("")
    L.append("Pseudo-symplectic drift (T3: `max_i max_p |delta_psi_i| <= %g tol_psi`):" % PS_DRIFT_FACTOR)
    L.append("")
    hdr = ["tol_psi", "n_ok/n_seeds", "max abs dpsi_i", "max/tol_psi", "rms dx2", "max abs dx2", "rms dx2/tol_psi",
           "rms dx3", "max abs dx3", "rms dx3/tol_psi", "verdict"]
    rows = [[fmt(e["tol_psi"], "%.0e"), "%d/%d" % (e["n_ok"], e["n_seeds"]), fmt(e["max_abs_delta_psi"]),
             fmt(e["drift_over_tol"]), fmt(e["delta_x2_rms"]), fmt(e["delta_x2_max_abs"]),
             fmt(e["delta_x2_rms_over_tol"]), fmt(e["delta_x3_rms"]), fmt(e["delta_x3_max_abs"]),
             fmt(e["delta_x3_rms_over_tol"]), e["verdict"]] for e in res["pseudo_symplectic"]]
    L.append(md_table(hdr, rows) if rows else "(no pseudo-symplectic runs)")
    L.append("")
    L.append("### 4. Eq. (36) numbers `D_ii = var / (2 <tau>)` -- %s" % O3_LABEL)
    L.append("")
    L.append("Uniform: recomputed from `seeds.csv`. Flux-weighted: copied from `summary.json` "
             "(`D22_flux`, `D33_flux`; the flux weights `c1(x_0)` are not in `seeds.csv`).")
    L.append("")
    hdr = ["tracker", "level", "<tau> uniform", "<tau> flux (summary)", "D22 uniform", "D33 uniform",
           "D22 flux (summary)", "D33 flux (summary)"]
    rows = [[e["tracker"], level_label(e["tracker"], e["x"]), fmt(e["mean_tau"], "%.8g"),
             fmt(e["flux_weighted_tau_mean_summary"], "%.8g"), fmt(e["D22_uniform"]), fmt(e["D33_uniform"]),
             fmt(e["D22_flux_summary"]), fmt(e["D33_flux_summary"])] for e in res["eq36"]]
    L.append(md_table(hdr, rows))
    L.append("")
    L.append("### 5. Consistency: recomputed vs `summary.json` (max abs difference per statistic)")
    L.append("")
    hdr = ["tracker", "level", "delta_x2", "delta_x3", "delta_psi1", "delta_psi2", "tau", "D_uniform",
           "n_seeds", "n_ok"]
    rows = []
    for t in TRACKERS:
        for r in fd["runs"][t]:
            c = r["consistency"]
            rows.append([t, level_label(t, r["x"])] + [fmt(c[k], "%.3g") for k in
                        ("delta_x2", "delta_x3", "delta_psi1", "delta_psi2", "tau", "D_uniform", "n_seeds", "n_ok")])
    L.append(md_table(hdr, rows))
    L.append("")
    L.append("### 6. Issues")
    L.append("")
    L.extend(["- " + i for i in fd["issues"]] or ["(none)"])
    L.append("")
    return "\n".join(L)


def crossfield_table(results):
    hdr = ["field", "Pollock p var(dx2)", "Pollock p var(dx3)", "RK p var(dx2) (5 tol)", "RK p var(dx3) (5 tol)",
           "RK p var(dx2) (all)", "RK p var(dx3) (all)", "PS T3"]
    rows = []
    for n in sorted(results):
        res = results[n]

        def cell(an):
            if an["fit"] is None:
                return "n/a (%s)" % an["verdict"]
            return "%.3f (%s%s)" % (an["fit"]["slope"], an["verdict"], ", non_monotone" if an["non_monotone"] else "")
        ps = res["pseudo_symplectic"]
        ps_txt = ", ".join("%.0e:%s" % (e["tol_psi"], e["verdict"]) for e in ps) or "n/a"
        rows.append([n, cell(res["pollock"]["delta_x2"]), cell(res["pollock"]["delta_x3"]),
                     cell(res["rk"]["paper"]["delta_x2"]), cell(res["rk"]["paper"]["delta_x3"]),
                     cell(res["rk"]["all"]["delta_x2"]), cell(res["rk"]["all"]["delta_x3"]), ps_txt])
    return md_table(hdr, rows)


# ------------------------------------------------------------------------------------
# driver
# ------------------------------------------------------------------------------------
def constants_block():
    return {"POLLOCK_BAND": list(POLLOCK_BAND), "RK_BAND": list(RK_BAND), "RK_PAPER_TOLS": list(RK_PAPER_TOLS),
            "PS_DRIFT_FACTOR": PS_DRIFT_FACTOR, "MIN_OK_FRACTION": MIN_OK_FRACTION, "HIST_BINS": HIST_BINS,
            "HIST_RANGE": dict(HIST_RANGE), "POLLOCK_RATIOS": list(POLLOCK_RATIOS),
            "POLLOCK_MIN_LEVELS": POLLOCK_MIN_LEVELS, "RK_MIN_LEVELS": RK_MIN_LEVELS,
            "AUTO_RANGE_SIGMAS": AUTO_RANGE_SIGMAS}


def run_analysis(root, outdir=None, fields_filter=None, quiet=False):
    outdir = outdir or os.path.join(root, "analysis")
    os.makedirs(os.path.join(outdir, "histograms"), exist_ok=True)
    issues_global = []
    fields = load_root(root, fields_filter, issues_global)
    if fields_filter:
        for f in sorted(fields_filter - set(fields)):
            issues_global.append("requested field %s not found" % f)
    results = {n: analyze_field(n, fd) for n, fd in fields.items()}
    written = []
    for n in sorted(fields):
        fd, res = fields[n], results[n]
        for t in TRACKERS:
            if fd["runs"][t]:
                p = os.path.join(outdir, "histograms", "%s_%s.json" % (n, t))
                write_json(p, histograms_json(n, t, fd["runs"][t]))
                written.append(p)
        fig_pdf_pair(n, fd["labels"], fd["runs"]["pollock"], "pollock", "fig3b",
                     "one-period displacement delta_x2, delta_x3 [L]", outdir, written)
        fig_variance(n, fd["labels"], fd["runs"]["pollock"], res["pollock"], "pollock", outdir, written)
        fig_pdf_pair(n, fd["labels"], fd["runs"]["rk"], "rk", "fig4a",
                     "one-period displacement delta_x2, delta_x3 [L]", outdir, written)
        fig_variance(n, fd["labels"], fd["runs"]["rk"], res["rk"]["paper"], "rk", outdir, written)
        fig_ps(n, fd["labels"], res["pseudo_symplectic"], outdir, written)
    fig_crossfield(results, outdir, written)

    exp = {"constants": constants_block(), "issues": issues_global, "fields": {}}
    for n in sorted(results):
        res = results[n]
        exp["fields"][n] = {"pollock": res["pollock"], "rk": res["rk"], "pseudo_symplectic": res["pseudo_symplectic"],
                            "eq36": res["eq36"], "eq36_label": O3_LABEL,
                            "consistency": {r["dir"]: r["consistency"] for t in TRACKERS for r in fields[n]["runs"][t]},
                            "issues": fields[n]["issues"]}
    p = os.path.join(outdir, "exponents.json")
    write_json(p, exp)
    written.append(p)

    T = ["# SF-32 spurious-spreading analysis", "",
         "Generated by `apps/spurious_spreading/analyze.py` from `%s`." % os.path.basename(os.path.normpath(root)), "",
         "Pre-registered rules (constant block of the script): Pollock band %s over Delta/Delta0 in %s "
         "(>= %d qualifying levels); RK band %s over tol in %s (>= %d qualifying levels; all-level fit "
         "reported); PS drift <= %g tol_psi; a level qualifies if n_ok/n_seeds >= %g; histograms %d bins on "
         "%s." % (list(POLLOCK_BAND), list(POLLOCK_RATIOS), POLLOCK_MIN_LEVELS, list(RK_BAND), list(RK_PAPER_TOLS),
                  RK_MIN_LEVELS, PS_DRIFT_FACTOR, MIN_OK_FRACTION, HIST_BINS, dict(sorted(HIST_RANGE.items()))), ""]
    for n in sorted(fields):
        T.append(field_tables(n, fields[n], results[n]))
    T.append("## Cross-field fitted exponents (robustness reading of T1/T2)")
    T.append("")
    T.append(crossfield_table(results))
    T.append("")
    T.append("## Global issues")
    T.append("")
    T.extend(["- " + i for i in issues_global] or ["(none)"])
    T.append("")
    p = os.path.join(outdir, "tables.md")
    with open(p, "w") as f:
        f.write("\n".join(T))
    written.append(p)
    if not quiet:
        print("analyze: %d field(s), %d file(s) written to %s" % (len(fields), len(written), outdir))
        print(crossfield_table(results))
    return {"fields": fields, "results": results, "written": written, "outdir": outdir,
            "issues": issues_global}


# ------------------------------------------------------------------------------------
# self-test
# ------------------------------------------------------------------------------------
def _write_run(fdir, tracker, tag, level, n, gen, rng):
    """Write one synthetic run with the exact instrument schema. `gen` gives target variances etc."""
    rdir = os.path.join(fdir, "%s_%s" % (tracker, tag))
    os.makedirs(rdir, exist_ok=True)
    n_fail = gen.get("n_fail", 0)
    status = np.zeros(n, dtype=np.int64)
    status[n - n_fail:] = 12
    ok = status == 0
    m = int(np.count_nonzero(ok))

    def standard():
        z = rng.standard_normal(m)
        z = z - z.mean()
        return z / np.sqrt(np.mean(z * z))
    cols = {c: np.full(n, np.nan) for c in STAT_COLUMNS + ("tau",)}
    cols["delta_x2"][ok] = math.sqrt(gen["var2"]) * standard()
    cols["delta_x3"][ok] = math.sqrt(gen["var3"]) * standard()
    if "psi_max" in gen:
        for c in ("delta_psi1", "delta_psi2"):
            u = rng.uniform(-1.0, 1.0, m)
            u[0] = 1.0
            cols[c][ok] = gen["psi_max"] * u
        if "psi1_max" in gen:
            cols["delta_psi1"][np.flatnonzero(ok)[1]] = gen["psi1_max"]
    else:
        cols["delta_psi1"][ok] = 1e-3 * standard()
        cols["delta_psi2"][ok] = 1e-3 * standard()
    cols["tau"][ok] = 1.0 + 0.01 * standard()
    x20, x30 = rng.uniform(0, 1, n), rng.uniform(0, 1, n)
    with open(os.path.join(rdir, "seeds.csv"), "w") as f:
        f.write(",".join(CSV_COLUMNS) + "\n")
        for i in range(n):
            vals = [repr(float(v)) for v in (x20[i], x30[i])]
            vals.append("%d" % status[i])
            vals += [("nan" if not math.isfinite(cols[c][i]) else repr(float(cols[c][i]))) for c in
                     ("delta_x2", "delta_x3", "delta_psi1", "delta_psi2", "tau")]
            vals += ["%d" % (100 + i % 7), repr(0.0 if tracker == "pollock" else 1e-13)]
            f.write("%d,%s\n" % (i, ",".join(vals)))
    issues = []
    seeds = read_seeds(os.path.join(rdir, "seeds.csv"), issues, tag)
    assert not issues, issues
    st = compute_stats(seeds)
    summary = {
        "field": {"route": "synthetic"}, "tracker": tracker, "level": level, "n_seeds": st["n_seeds"],
        "n_ok": st["n_ok"], "status_counts": st["status_counts"],
        "stats": {c: {k: st[c][k] for k in ("mean", "var", "rms", "max_abs")} for c in STAT_COLUMNS},
        "flux_weighted_tau_mean": st["tau"]["mean"],
        "D22_uniform": st["D22_uniform"], "D33_uniform": st["D33_uniform"],
        "D22_flux": st["D22_uniform"], "D33_flux": st["D33_uniform"],
        "min_abs_c_grid": 0.8, "landing": {"max_err": 1e-13, "max_iterations": 40},
        "wall_seconds": 1.0, "config": {"selftest": True},
    }
    summary["stats"]["tau"] = {k: st["tau"][k] for k in ("mean", "var", "min", "max")}
    if tracker == "pollock":
        summary["divergence_max_rel"] = 1e-15
    write_json(os.path.join(rdir, "summary.json"), summary)


def make_synthetic_root(root):
    """Two synthetic fields with exact power laws (see the self-test assertions)."""
    rng = np.random.default_rng(20261006)
    n = 512
    spec = {
        # field A: pollock p = 2 (in band), rk p = 0.5 (in band), PS drift 0.5 tol_psi (pass)
        "synthetic_A_e1_n32": {"p_pol": 2.0, "p_rk": 0.5, "rk_plateau": False, "ps_fail": None,
                               "pol_fail": {1: 2}, "rk_fail": {}},
        # field B: pollock p = 3 (out of band), rk p = 1.2 (out of band; the 1e-4 level has 30/512
        # failures and is excluded; 1e-9, 1e-10 plateau -> non_monotone in the all-level fit only),
        # PS: the 1e-10 level drifts 20 tol_psi (fail)
        "synthetic_B_e1_n32": {"p_pol": 3.0, "p_rk": 1.2, "rk_plateau": True, "ps_fail": 1e-10,
                               "pol_fail": {}, "rk_fail": {1e-4: 30}},
    }
    for name, s in spec.items():
        fdir = os.path.join(root, name)
        os.makedirs(fdir, exist_ok=True)
        write_json(os.path.join(fdir, "labels.json"), {"route": "synthetic", "field": name, "n": 32, "h": 1 / 32})
        for m in (1, 2, 4, 8):
            delta = m / 32.0
            gen = {"var2": 1e-4 * m ** s["p_pol"], "var3": 2.5e-4 * m ** s["p_pol"], "n_fail": s["pol_fail"].get(m, 0)}
            _write_run(fdir, "pollock", "m%d" % m, {"delta_ratio": m, "n_cells": 32 // m, "delta": delta}, n, gen, rng)
        for k, tol in enumerate((1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10)):
            a2, a3 = (1e-2, 2e-2) if s["p_rk"] == 0.5 else (1.0, 2.0)
            t_eff = tol if not (s["rk_plateau"] and tol < 1e-8) else 1e-8
            bump = 1.0 if t_eff == tol else 1.1
            gen = {"var2": bump * a2 * t_eff ** s["p_rk"], "var3": bump * a3 * t_eff ** s["p_rk"],
                   "n_fail": s["rk_fail"].get(tol, 0)}
            _write_run(fdir, "rk", "tol%.0e" % tol, {"tol": tol, "dt_max": 1 / 64}, n, gen, rng)
        for tol in (1e-8, 1e-10, 1e-12):
            gen = {"var2": (0.3 * tol) ** 2, "var3": (0.4 * tol) ** 2, "psi_max": 0.5 * tol}
            if s["ps_fail"] is not None and same_value(tol, s["ps_fail"]):
                gen["psi1_max"] = 20.0 * tol
            _write_run(fdir, "pseudo_symplectic", "tolpsi%.0e" % tol, {"tol_psi": tol, "ds": 1 / 64}, n, gen, rng)


def tree_digest(d):
    h = {}
    for base, _, files in os.walk(d):
        for fn in files:
            p = os.path.join(base, fn)
            with open(p, "rb") as f:
                h[os.path.relpath(p, d)] = hashlib.sha256(f.read()).hexdigest()
    return h


def selftest():
    failures = []

    def check(cond, msg):
        print("  [%s] %s" % ("ok" if cond else "FAIL", msg))
        if not cond:
            failures.append(msg)

    tmp = tempfile.mkdtemp(prefix="sf32_analyze_selftest_")
    try:
        root = os.path.join(tmp, "root")
        make_synthetic_root(root)
        print("selftest 1: synthetic root (generator in the script) -> %s" % root)
        out1 = run_analysis(root, os.path.join(tmp, "out1"), quiet=True)
        R = out1["results"]
        A, B = R["synthetic_A_e1_n32"], R["synthetic_B_e1_n32"]
        for comp in COMPONENTS:
            for lab, an, p, verdict in (("A pollock", A["pollock"][comp], 2.0, "in_band"),
                                        ("A rk paper", A["rk"]["paper"][comp], 0.5, "in_band"),
                                        ("A rk all", A["rk"]["all"][comp], 0.5, "in_band"),
                                        ("B pollock", B["pollock"][comp], 3.0, "out_of_band"),
                                        ("B rk paper", B["rk"]["paper"][comp], 1.2, "out_of_band")):
                s = an["fit"]["slope"] if an["fit"] else float("nan")
                check(abs(s - p) <= 1e-6, "%s %s slope %.12f vs generating %g" % (lab, comp, s, p))
                check(an["verdict"] == verdict, "%s %s verdict %s (expected %s)" % (lab, comp, an["verdict"], verdict))
                check(all(abs(t["slope"] - p) <= 1e-6 for t in an["two_level"]),
                      "%s %s two-level estimates %s" % (lab, comp, ["%.9f" % t["slope"] for t in an["two_level"]]))
                check(not an["non_monotone"], "%s %s monotone" % (lab, comp))
            check(A["pollock"][comp]["n_qualifying"] == 4, "A pollock %s: m1 with 2/512 failures qualifies" % comp)
            check(B["rk"]["paper"][comp]["n_qualifying"] == 4,
                  "B rk %s: tol 1e-4 with 30/512 failures excluded (4 levels left)" % comp)
            check(B["rk"]["all"][comp]["non_monotone"], "B rk all-level %s plateau flagged non_monotone" % comp)
        for lab, res, expect in (("A", A, {1e-8: "pass", 1e-10: "pass", 1e-12: "pass"}),
                                 ("B", B, {1e-8: "pass", 1e-10: "fail", 1e-12: "pass"})):
            for e in res["pseudo_symplectic"]:
                want = [v for k, v in expect.items() if same_value(k, e["tol_psi"])][0]
                check(e["verdict"] == want, "%s PS tol_psi %.0e verdict %s (drift/tol %.3g)" %
                      (lab, e["tol_psi"], e["verdict"], e["drift_over_tol"]))
        maxdiff = 0.0
        for n in sorted(out1["fields"]):
            for t in TRACKERS:
                for r in out1["fields"][n]["runs"][t]:
                    for v in r["consistency"].values():
                        if v is not None:
                            maxdiff = max(maxdiff, v)
        check(maxdiff == 0.0, "consistency: max |recomputed - summary| = %g" % maxdiff)
        check(not any(out1["fields"][n]["issues"] for n in out1["fields"]),
              "no issues reported (%s)" % [i for n in out1["fields"] for i in out1["fields"][n]["issues"]])
        stems = ["fig3b", "fig3c", "fig4a", "fig4b", "ps_drift"]
        expected = ["%s_%s.%s" % (s, n, e) for s in stems for n in out1["fields"] for e in ("png", "pdf")]
        expected += ["exponents_crossfield.png", "exponents_crossfield.pdf", "tables.md", "exponents.json"]
        expected += ["histograms/%s_%s.json" % (n, t) for n in out1["fields"] for t in TRACKERS]
        missing = [e for e in expected if not os.path.isfile(os.path.join(out1["outdir"], e))]
        check(not missing, "%d expected outputs written (missing: %s)" % (len(expected), missing))
        with open(os.path.join(out1["outdir"], "histograms", "synthetic_A_e1_n32_pollock.json")) as f:
            hj = json.load(f)
        tot = sum(sum(lv[c]["fixed"]["counts"]) + lv[c]["fixed"]["underflow"] + lv[c]["fixed"]["overflow"]
                  for lv in hj["levels"] for c in COMPONENTS)
        check(tot == 2 * (4 * 512 - 2), "histogram counts + under/overflow = 2 x n_ok over levels (%d)" % tot)
        out2 = run_analysis(root, os.path.join(tmp, "out2"), quiet=True)
        d1, d2 = tree_digest(out1["outdir"]), tree_digest(out2["outdir"])
        check(d1 == d2, "byte-identical outputs on a second run (%d files)" % len(d1))

        fx = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_selftest_fixture")
        print("selftest 2: versioned on-disk fixture -> %s" % fx)
        out3 = run_analysis(fx, os.path.join(tmp, "out3"), quiet=True)
        F = out3["results"].get("analytic_G_a0.05_n16")
        check(F is not None, "fixture field analytic_G_a0.05_n16 loaded")
        if F is not None:
            for comp in COMPONENTS:
                an = F["pollock"][comp]
                check(an["verdict"] == "insufficient", "fixture pollock %s verdict %s (2 levels < %d)" %
                      (comp, an["verdict"], POLLOCK_MIN_LEVELS))
                tl = an["two_level"][0]["slope"] if an["two_level"] else float("nan")
                check(abs(tl - 2.0) <= 1e-12, "fixture pollock %s two-level estimate %.15f (constructed: 2)" % (comp, tl))
                fit = an["fit"]
                check(fit is not None and abs(fit["slope"] - 2.0) <= 1e-12, "fixture pollock %s 2-point fit slope" % comp)
            c = [v for r in out3["fields"]["analytic_G_a0.05_n16"]["runs"]["pollock"]
                 for v in r["consistency"].values() if v is not None]
            check(c and max(c) <= 1e-12, "fixture consistency max |diff| = %g" % (max(c) if c else float("nan")))
            fi = out3["fields"]["analytic_G_a0.05_n16"]["issues"]
            check(fi == ["no rk runs", "no pseudo_symplectic runs"], "fixture issues reported: %s" % fi)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if failures:
        print("SELFTEST FAILED: %d check(s)" % len(failures))
        return 1
    print("SELFTEST PASSED")
    return 0


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("root", nargs="?")
    ap.add_argument("--fields", default=None, help="comma-separated field names (default: all)")
    ap.add_argument("--figures-dir", default=None, help="output directory (default: <root>/analysis)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.root:
        ap.error("<out-root> is required")
    ff = set(x for x in a.fields.split(",") if x) if a.fields else None
    run_analysis(a.root, a.figures_dir, ff)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
