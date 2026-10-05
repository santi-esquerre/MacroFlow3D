#!/usr/bin/env python3
"""SF-30 streamline-closure gate: follow-up checks cited by the experiment note.

Usage:
    python3 followup_checks.py <artifacts_dir> [--out <file.md>]

<artifacts_dir> is `docs/experiments/artifacts/2026-10-05-sf30-closure-gate` (the directory that
contains `raw/` and `raw_followup/`). Prints four Markdown tables (stdout, or --out):

    F1  per-seed recomputation of the period-1 `R` from every `streamlines.csv.gz`
    F2  pointwise grid convergence of the period-1 return map over the paired seeds
    F3  tolerance-induced separation (1e-8 vs 1e-12) versus the number of periods
    F4  many-period statistics across the four integrator tolerances

These are checks of the committed raw data, not part of the pre-registered decision rule
(which lives in `analyze.py`). F3 uses the exploratory follow-up runs of job `sf30-post`
(`raw_followup/sensitivity`, NOT pre-registered). Python 3 standard library only; output is
deterministic and independent of the invocation path.
"""

import gzip
import json
import math
import os
import sys

WORKING_TOL = 1e-8
TIGHTEST_TOL = 1e-12
CASES = [("s025_l8", 0.25, 0.125), ("s025_l16", 0.25, 0.0625), ("s1_l8", 1.0, 0.125),
         ("s1_l16", 1.0, 0.0625), ("s4_l8", 4.0, 0.125), ("s4_l16", 4.0, 0.0625)]
LABEL = {(s, l): c for c, s, l in CASES}


def fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        return "%.4g" % x
    return str(x)


def table(title, caption, headers, rows):
    out = ["### " + title, "", caption, ""]
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        out.append("| " + " | ".join(fmt(v) for v in r) + " |")
    out.append("")
    return "\n".join(out)


def find_runs(base):
    """Return {relative run dir: summary dict} for every summary.json under base."""
    runs = {}
    for root, dirs, files in os.walk(base):
        dirs.sort()
        if "summary.json" in files:
            with open(os.path.join(root, "summary.json"), "r", encoding="utf-8") as f:
                runs[root] = json.load(f)
    return runs


def tol_entry(summary, tol):
    for t in summary.get("tolerances") or []:
        if float(t["tol"]) == tol:
            return t
    return None


def period_entry(entry, n):
    for p in (entry or {}).get("periods") or []:
        if p.get("n") == n:
            return p
    return None


def ladder_entry(entry, n):
    lad = (entry or {}).get("ladder_vs_tightest") or {}
    for p in lad.get("periods") or []:
        if p.get("n") == n:
            return p
    return None


def read_csv(path):
    """Return {id: row dict} of a gzip-compressed per-seed CSV."""
    rows = {}
    with gzip.open(path, "rt", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
        for line in f:
            line = line.strip()
            if not line:
                continue
            vals = line.split(",")
            rows[int(vals[0])] = dict(zip(header, vals))
    return rows


def r_from_rows(rows):
    """Population RMS of the period-1 displacement about its mean, over `ok` rows."""
    d = [(float(r["y_1"]) - float(r["y0"]), float(r["z_1"]) - float(r["z0"]))
         for r in rows.values() if r["status"] == "ok"]
    n = len(d)
    m2 = sum(a for a, _ in d) / n
    m3 = sum(b for _, b in d) / n
    var = sum((a - m2) ** 2 + (b - m3) ** 2 for a, b in d) / n
    return math.sqrt(var), n


def quantile(sorted_vals, q):
    """Linear interpolation between order statistics (position q (n - 1))."""
    n = len(sorted_vals)
    pos = q * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (pos - lo) * (sorted_vals[hi] - sorted_vals[lo])


def gauss_key(summary):
    c = summary["configuration"]
    return (LABEL.get((float(c["sigma2"]), float(c["ell"]))), c["seed"], c["n"], c.get("periods") or 1)


# -------------------------------------------------------------------------------------
# F1
# -------------------------------------------------------------------------------------

def f1(art, runs):
    rows = []
    maxdiff = 0.0
    count = 0
    per_group = {}
    for d in sorted(runs):
        csv = os.path.join(d, "streamlines.csv.gz")
        if not os.path.exists(csv):
            continue
        s = runs[d]
        wt = float(s["configuration"]["working_tol"])
        R_sum = period_entry(tol_entry(s, wt), 1)["unweighted"]["R"]
        R_csv, n_ok = r_from_rows(read_csv(csv))
        n_sum = period_entry(tol_entry(s, wt), 1)["unweighted"]["count"]
        diff = abs(R_csv - R_sum)
        group = os.path.relpath(d, art).split(os.sep)[:2]
        g = "/".join(group)
        st = per_group.setdefault(g, [0, 0.0, 0, set()])
        st[0] += 1
        st[1] = max(st[1], diff)
        st[2] += int(n_ok != n_sum)
        st[3].add(wt)
        count += 1
        maxdiff = max(maxdiff, diff)
    for g in sorted(per_group):
        n, md, cm, wts = per_group[g]
        rows.append([g, n, ", ".join("%g" % w for w in sorted(wts)), cm, md])
    rows.append(["all", count, "", sum(v[2] for v in per_group.values()), maxdiff])
    return table(
        "F1. Per-seed recomputation of `R`",
        "For every run with a `streamlines.csv.gz` (the per-seed CSV is written at the run's working tolerance "
        "`configuration.working_tol`: 1e-10 for the `_p16` analytic controls, 1e-8 otherwise): the unweighted "
        "period-1 `R` recomputed from the CSV as the square root of the population variance (about the mean) of "
        "the displacement `(y_1 - y0, z_1 - z0)` over the rows with `status == ok`, compared with "
        "`tolerances[working_tol].periods[n=1].unweighted.R` of the run's `summary.json`. Columns: runs "
        "compared, working tolerances, runs whose `ok` count differs from the summary's `count`, maximum "
        "absolute difference of `R`.",
        ["group", "runs compared", "working tol", "count mismatches", "max abs diff R"], rows), maxdiff, count


# -------------------------------------------------------------------------------------
# F2
# -------------------------------------------------------------------------------------

def gauss_csv_index(art, runs):
    idx = {}
    for d, s in runs.items():
        if s["configuration"]["field"] != "gaussian":
            continue
        rel = os.path.relpath(d, art)
        if not rel.startswith("raw" + os.sep) or not os.path.exists(os.path.join(d, "streamlines.csv.gz")):
            continue
        idx[gauss_key(s)] = d
    return idx


def pair_dist(rows_a, rows_b):
    out = []
    for i in sorted(rows_a):
        a, b = rows_a[i], rows_b.get(i)
        if b is None or a["status"] != "ok" or b["status"] != "ok":
            continue
        out.append(math.hypot(float(a["y_1"]) - float(b["y_1"]), float(a["z_1"]) - float(b["z_1"])))
    return out


def dist_stats(ds):
    s = sorted(ds)
    rms = math.sqrt(sum(x * x for x in s) / len(s))
    return rms, quantile(s, 0.5), quantile(s, 0.99), s[-1], len(s)


def f2(art, runs):
    idx = gauss_csv_index(art, runs)
    rows = []
    cache = {}

    def rows_of(key):
        if key not in cache:
            cache[key] = read_csv(os.path.join(idx[key], "streamlines.csv.gz"))
        return cache[key]

    items = [(c, 3001) for c, _, _ in CASES] + [("s4_l16", 3002), ("s4_l16", 3003)]
    for case, seed in items:
        k64, k128, k256 = (case, seed, 64, 1), (case, seed, 128, 1), (case, seed, 256, 1)
        R256 = period_entry(tol_entry(runs[idx[k256]], WORKING_TOL), 1)["unweighted"]["R"]
        b = dist_stats(pair_dist(rows_of(k128), rows_of(k256)))
        if k64 in idx:
            a = dist_stats(pair_dist(rows_of(k64), rows_of(k128)))
            ratio = a[0] / b[0]
        else:
            a = (None, None, None, None, None)
            ratio = None
        rows.append([case, seed, a[4] if a[4] is not None else None, a[0], a[1], a[2], a[3],
                     b[4], b[0], b[1], b[2], b[3], ratio, R256, b[0] / R256])
    return table(
        "F2. Pointwise grid convergence of the period-1 return map",
        "Gaussian runs at the working tolerance 1e-8 (`raw/ladder` for 64^3 and 256^3, `raw/matrix128` for "
        "128^3), the 1024 seeds paired by `id` (seeds `ok` on both grids): distance between the period-1 return "
        "points on two grids, `hypot(y_1(N) - y_1(2N), z_1(N) - z_1(2N))`; its RMS, median, 99th percentile "
        "(linear interpolation between order statistics) and maximum, for 64 vs 128 and 128 vs 256; the ratio "
        "RMS(64-128)/RMS(128-256) (4 for second order); `R(256)` from the 256^3 summary; RMS(128-256)/R(256). "
        "`(4, 0.0625)` seeds 3002 and 3003 have no 64^3 run.",
        ["case", "seed", "pairs 64-128", "RMS 64-128", "median 64-128", "p99 64-128", "max 64-128",
         "pairs 128-256", "RMS 128-256", "median 128-256", "p99 128-256", "max 128-256",
         "RMS ratio", "R(256)", "RMS(128-256)/R(256)"], rows)


# -------------------------------------------------------------------------------------
# F3
# -------------------------------------------------------------------------------------

def f3(art, runs):
    # (case, N) -> {n: [(rms, max, source)]}
    data = {}
    for d in sorted(runs):
        s = runs[d]
        c = s["configuration"]
        if c["field"] != "gaussian" or (c.get("periods") or 1) <= 1:
            continue
        rel = os.path.relpath(d, art)
        case, seed, n_grid, P = gauss_key(s)
        w = tol_entry(s, WORKING_TOL)
        if float(w["ladder_vs_tightest"]["tightest_tol"]) != TIGHTEST_TOL:
            continue
        src = "follow-up" if rel.startswith("raw_followup") else "many-period"
        for p in w["ladder_vs_tightest"]["periods"]:
            data.setdefault((case, n_grid), {}).setdefault(p["n"], []).append(
                (p["rms_distance"], p["max_distance"], src, os.path.basename(d)))
    rows = []
    consistency = []
    order = {c: i for i, (c, _, _) in enumerate(CASES)}
    for key in sorted(data, key=lambda k: (order[k[0]], k[1])):
        prev = None
        for n in sorted(data[key]):
            entries = data[key][n]
            rms, mx = entries[0][0], entries[0][1]
            if len(entries) > 1:
                same = all(e[0] == rms and e[1] == mx for e in entries)
                consistency.append((key, n, len(entries), same))
            srcs = sorted(set(e[2] for e in entries))
            src = srcs[0] if len(srcs) == 1 else "both"
            growth = rms / prev[1] if prev is not None else None
            rows.append([key[0], key[1], n, src, rms, mx, prev[0] if prev else None, growth])
            prev = (n, rms)
    t = table(
        "F3. Tolerance-induced separation versus the number of periods",
        "Gaussian seed 3001, runs with more than one period: `raw/manyperiod` (pre-registered) and "
        "`raw_followup/sensitivity` (job `sf30-post`, exploratory, NOT pre-registered). Each summary stores, for the "
        "working tolerance 1e-8, the distance to the tightest tolerance 1e-12 of the per-seed return points at "
        "period 1 and at its last period (`ladder_vs_tightest`); this table collects them per `(case, N)`: "
        "`rms_distance` and `max_distance` at period `n`, the previous available `n` and the growth factor "
        "RMS(n)/RMS(previous n). Source: `many-period`, `follow-up`, or `both` (several runs give the same `n`).",
        ["case", "N", "n", "source", "rms distance 1e-8 vs 1e-12", "max distance", "previous n",
         "RMS growth factor"], rows)
    lines = ["", "Consistency of the entries given by several runs for the same `(case, N, n)` "
             "(identical continuous integrations must give identical values):", ""]
    lines.append("| case | N | n | runs | identical |")
    lines.append("|---|---|---|---|---|")
    for key, n, cnt, same in consistency:
        lines.append("| %s | %d | %d | %d | %s |" % (key[0], key[1], n, cnt, fmt(same)))
    lines.append("")
    return t + "\n".join(lines)


# -------------------------------------------------------------------------------------
# F4
# -------------------------------------------------------------------------------------

def f4(art, runs):
    rows = []
    sel = []
    for d in sorted(runs):
        s = runs[d]
        rel = os.path.relpath(d, art)
        if rel.startswith(os.path.join("raw", "manyperiod")):
            sel.append((os.path.basename(d), s))
    for rid, s in sel:
        P = s["configuration"]["periods"]
        tols = sorted((float(t["tol"]) for t in s["tolerances"]), reverse=True)
        for n in (1, 4, 16, 64):
            if n > P:
                continue
            for q in ("R", "var_d2", "var_d3"):
                vals = [period_entry(tol_entry(s, t), n)["unweighted"][q] for t in tols]
                mean = sum(vals) / len(vals)
                rows.append([rid, n, q] + vals + [(max(vals) - min(vals)) / mean])
    return table(
        "F4. Many-period statistics across tolerances",
        "Runs of `raw/manyperiod` (seed 3001): unweighted `R`, `var(d2)`, `var(d3)` of the displacement after "
        "`n` periods (n = 1, 4, 16, 64 where available) at each integrator tolerance of the ladder, and the "
        "relative spread `(max - min)/mean` over the four tolerances.",
        ["run", "n", "quantity", "tol 1e-6", "tol 1e-8", "tol 1e-10", "tol 1e-12", "rel spread"], rows)


def main(argv):
    args = argv[1:]
    out = None
    art = None
    i = 0
    while i < len(args):
        if args[i] == "--out" and i + 1 < len(args):
            out = args[i + 1]
            i += 2
        elif args[i].startswith("--") or art is not None:
            print(__doc__, file=sys.stderr)
            return 2
        else:
            art = args[i]
            i += 1
    if art is None or not os.path.isdir(os.path.join(art, "raw")):
        print(__doc__, file=sys.stderr)
        return 2
    art = os.path.normpath(art)
    runs = find_runs(os.path.join(art, "raw"))
    runs.update(find_runs(os.path.join(art, "raw_followup")))
    t1, maxdiff, count = f1(art, runs)
    parts = ["# SF-30 closure gate: follow-up checks of `%s`" % os.path.basename(os.path.abspath(art)), "",
             "Generated by `scripts/followup_checks.py` from `raw/` and `raw_followup/`. Checks of the committed raw "
             "data cited by the experiment note; not part of the pre-registered decision rule (`scripts/analyze.py`).",
             "", "Summaries read: %d; F1 runs compared: %d; F1 maximum absolute difference: %s." %
             (len(runs), count, fmt(maxdiff)), ""]
    parts += [t1, f2(art, runs), f3(art, runs), f4(art, runs)]
    text = "\n".join(parts)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
