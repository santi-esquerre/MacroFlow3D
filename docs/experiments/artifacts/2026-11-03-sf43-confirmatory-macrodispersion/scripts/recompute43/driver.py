"""Recomputation driver: reduced tables (`python3 -m sf42red reduce`) vs values recomputed from the raw moment files.

CAMPAIGN LAYOUT (contract with the SF-43 campaign engine)
  <C>/arms/<arm_id>/<segment>/            one runner invocation
      out/.../realizations/r_XXXX/        consumer realization (pass root R, or R/hmax_<d>): track.json, moment CSVs, [arrays]
      <segment>/out/crossings/realization_XXXX_planes.csv   Par2 plane crossings (and `config.yaml`)
REDUCED TABLES  `--reduced <dir>`: every `*.csv` below it (not `*.planes.csv`) whose header starts `code,source,r,hmax_div,seed,`,
  each with `<table>.manifest.json` and `<table minus .csv>.planes.csv` written by `sf42red reduce`.  A row's `source` is relative
  to the reducer's `--root`; it is resolved against `--root` (repeatable; default: the campaign directory).

WHAT IS RECOMPUTED (every other column/row is listed under NOT_COVERED, never silently skipped)
  consumer, family plane
    moment CSVs (all realizations; weights flux = P/moments_planes.csv, unif = P/twin/moments_planes.csv; basis pos; set perplane):
        alpha_TP, alpha_LP, U_r, sigT2_first, sigT2_last, tilt, n_first, n_last, n_min, n_max, n_common, set_differs (sets are
        nested: n_common = min over the window of `contributing`), n_planes, d_first, d_last, seed, r, hmax_div, ell, x0;
        planes table: n, mean_tau, var_tau, mean_x2, mean_x3, var_x2, var_x3 of every plane.
    arrays (retained subset only): every weights (flux/unif/expI) x basis (pos/disp) x set (perplane/common) row: all of the above
        plus alpha_TAP, wfrac_min, n_traced; planes table: also cov_x2x3.
    NOT covered: family `time` rows (the window end needs status_time/times_x), split-half columns `*_A`/`*_B`, `note`.
  par2, family plane: every estimator/count column of both bases and the planes table, from the crossing CSV.
    NOT covered: family `time` rows (series + snapshots), `ell`/`x0`/`d_lo`/`d_hi` (reducer inputs, not in the raw files).
Manifest: the sha256 of every input still present is recomputed (a pruned input is counted, not a failure); a changed input or table
is a failure.
Exit code: 0 PASS, 1 FAIL (a value beyond tolerance, a changed input, an unreadable raw file), 2 nothing compared.
"""
import csv
import glob
import hashlib
import json
import math
import os
import re
from collections import OrderedDict

from . import formulas as F
from .sources import ConsumerSource, Par2Source, plane_row_values

NAN = float("nan")
RTOL, ATOL = 1e-10, 1e-15
TABLE_HEAD = "code,source,r,hmax_div,seed,"
ESTIMATORS = ("alpha_TP", "alpha_TAP", "alpha_LP", "U_r", "sigT2_first", "sigT2_last", "tilt")


def fnum(s):
    s = (s or "").strip()
    return NAN if s == "" else float(s)


class Tally:
    """Comparisons by (code, family, origin, column) with counts, realizations, max relative difference and the failures."""

    def __init__(self, rtol, atol):
        self.rtol, self.atol = rtol, atol
        self.cov = OrderedDict()
        self.fails = []
        self.n = 0
        self.max_rel = 0.0

    def add(self, key, who, label, got, want):
        ok, rel = F.close(float(got), float(want), self.rtol, self.atol)
        c = self.cov.setdefault(key, {"n": 0, "who": set(), "max": 0.0})
        c["n"] += 1
        c["who"].add(who)
        self.n += 1
        if math.isfinite(rel):
            c["max"] = max(c["max"], rel)
            self.max_rel = max(self.max_rel, rel)
        if not ok:
            self.fails.append("FAIL %s %s: recomputed %.17g reduced %.17g rel %.3e" % (who, label, float(got), float(want), rel))


def read_table(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def find_tables(reduced):
    out = []
    for p in sorted(glob.glob(os.path.join(reduced, "**", "*.csv"), recursive=True)):
        if p.endswith(".planes.csv"):
            continue
        with open(p) as f:
            if f.readline().startswith(TABLE_HEAD):
                out.append(p)
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def check_manifest(table, roots):
    """(n_inputs, n_verified, n_pruned, problems[str]) from `<table>.manifest.json`; inputs resolve against `roots`."""
    mp = table + ".manifest.json"
    if not os.path.isfile(mp):
        return 0, 0, 0, ["manifest %s is missing" % mp]
    with open(mp) as f:
        man = json.load(f)
    bad, ver, pruned = [], 0, 0
    for e in man["files"]:
        hit = next((os.path.join(r, e["path"]) for r in roots if os.path.isfile(os.path.join(r, e["path"]))), None)
        if hit is None:
            pruned += 1
        elif sha256(hit) != e["sha256"]:
            bad.append("input changed since the reduction: %s" % e["path"])
        else:
            ver += 1
    for e in man["tables"]:
        p = os.path.join(os.path.dirname(table), e["path"])
        if not os.path.isfile(p) or sha256(p) != e["sha256"]:
            bad.append("table changed or missing: %s" % e["path"])
    return len(man["files"]), ver, pruned, bad


def resolve(source, roots):
    return next((os.path.join(r, source) for r in roots if os.path.exists(os.path.join(r, source))), None)


def _vals(mom, x1w, ell, extra, cnt, origin):
    vals = {"n_first": cnt[0], "n_last": cnt[-1], "n_min": min(cnt), "n_max": max(cnt)}
    vals.update(extra)
    vals["set_differs"] = int(any(c != vals["n_common"] for c in cnt))
    try:
        vals.update(F.estimate(x1w, mom, ell))
    except F.Undefined:
        vals.update({c: NAN for c in ESTIMATORS})
    if origin == "csv":
        vals["alpha_TAP"] = None  # no cross covariance in the moment CSV
    return vals


def expected_plane_row(src, ell, idx, x1w, row):
    """([(origin, {column: value})], why_not) for one reduced plane row: every raw source that can deliver it."""
    wmode, basis, setname = row["weights"], row["basis"], row["set"]
    common = setname == "common"
    out = []
    if src.code == "par2":
        if setname != "perplane" or wmode != "unif":
            return [], "par2 has only the per-plane uniform set"
        m_all = src.moments(basis)
        mom = [m_all[k] for k in idx]
        cnt = [m["n"] if m else 0 for m in mom]
        n_inj = src.t["n_injected"][idx[0]]
        out.append(("crossings", _vals(mom, x1w, ell, {"n_traced": n_inj, "n_common": min(cnt), "wfrac_min": min(cnt) / n_inj}, cnt, "crossings")))
    else:
        if src.csv_available(wmode, basis) and not common:
            m_all = src.csv_moments(wmode)
            mom = [m_all[k] for k in idx]
            cnt = [m["n"] if m else 0 for m in mom]
            out.append(("csv", _vals(mom, x1w, ell, {"n_common": min(cnt)}, cnt, "csv")))  # nested sets: n_common = min
        if src.arrays is not None:
            mom = src.array_moments(wmode, basis, idx, common)
            cnt = [m["n"] if m else 0 for m in mom]
            n_common, wfrac, traced = src.set_stats(wmode, idx)
            out.append(("arrays", _vals(mom, x1w, ell, {"n_traced": traced, "n_common": n_common, "wfrac_min": wfrac}, cnt, "arrays")))
    return out, (None if out else "variant needs the per-particle arrays (pruned)")


def compare_plane_rows(tally, src, rows, who, not_covered):
    for row in rows:
        if src.code == "consumer":
            ell_r, x0_r = src.ell, src.x0
        else:
            ell_r, x0_r = float(row["ell"]), float(row["x0"])
        d_lo, d_hi = float(row["d_lo"]), float(row["d_hi"])
        tag = "%s/%s/%s" % (row["weights"], row["basis"], row["set"])
        try:
            idx = F.window_planes(src.planes, x0_r, ell_r, d_lo, d_hi)
        except F.Undefined as e:
            not_covered.append("%s %s: window not evaluable (%s)" % (who, tag, e))
            continue
        x1w = [src.planes[k] for k in idx]
        got, why = expected_plane_row(src, ell_r, idx, x1w, row)
        if not got:
            not_covered.append("%s %s: %s" % (who, tag, why))
            continue
        meta = {"n_planes": len(idx), "d_first": (x1w[0] - x0_r) / ell_r, "d_last": (x1w[-1] - x0_r) / ell_r}
        if src.code == "consumer":
            meta.update(seed=int(src.field["seed"]), ell=ell_r, x0=x0_r, r=int(re.search(r"r_(\d{4})", src.rdir).group(1)))
            hm = re.search(r"hmax_(\d+)$", os.path.normpath(src.root))
            meta["hmax_div"] = int(hm.group(1)) if hm else -1
        else:
            meta["r"] = int(re.search(r"realization_(\d{4})_planes", os.path.basename(src.path)).group(1))
        for origin, vals in got:
            for col, v in list(vals.items()) + list(meta.items()):
                if v is not None:
                    tally.add((src.code, "plane", origin, col), who, "%s %s" % (tag, col), v, fnum(row[col]))


def compare_planes_table(tally, src, prows, who, not_covered):
    """Per-plane table rows of one source: every plane, per-plane sets."""
    for (wmode, basis), grp in _group(prows, ("weights", "basis")).items():
        mom = []
        if src.code == "par2":
            if wmode == "unif":
                mom.append(("crossings", src.moments(basis)))
        else:
            if src.csv_available(wmode, basis):
                mom.append(("csv", src.csv_moments(wmode)))
            if src.arrays is not None:
                mom.append(("arrays", src.array_moments(wmode, basis, list(range(len(src.planes))), False)))
        if not mom:
            not_covered.append("%s planes table %s/%s: variant needs the per-particle arrays (pruned)" % (who, wmode, basis))
        for origin, m in mom:
            for r in grp:
                k = int(r["plane"])
                for col, v in plane_row_values(m[k]).items():
                    if v is not None:
                        tally.add((src.code, "planes-table", origin, col), who, "plane %d %s/%s %s" % (k, wmode, basis, col), v, fnum(r[col]))
                tally.add((src.code, "planes-table", origin, "x1"), who, "plane %d x1" % k, src.planes[k], fnum(r["x1"]))


def _group(rows, cols):
    out = OrderedDict()
    for r in rows:
        out.setdefault(tuple(r[c] for c in cols), []).append(r)
    return out


def open_source(code, path):
    return ConsumerSource(path) if code == "consumer" else Par2Source(path)


def run(campaign, reduced, roots, out, rtol=RTOL, atol=ATOL):
    """Write the report to the file object `out`; return the exit code."""
    P = lambda *a: print(*a, file=out)  # noqa: E731
    roots = list(roots) if roots else [campaign]
    tally = Tally(rtol, atol)
    P("RECOMPUTE43 campaign=%s reduced=%s roots=%s rtol=%g atol=%g" % (campaign, reduced, ",".join(roots), rtol, atol))
    problems, not_covered, nonrow = [], [], []
    realizations = {"consumer": set(), "par2": set()}
    with_arrays = set()
    tables = find_tables(reduced)
    if not tables:
        P("no reduced table under %s" % reduced)
    for t in tables:
        rows = read_table(t)
        pl = t[:-4] + ".planes.csv"
        prows = read_table(pl) if os.path.isfile(pl) else []
        if not prows:
            problems.append("%s: planes table missing or empty" % t)
        n_in, n_ver, n_pr, bad = check_manifest(t, roots)
        problems += ["%s: %s" % (os.path.basename(t), b) for b in bad]
        P("TABLE %s rows=%d planes_rows=%d manifest inputs=%d verified=%d pruned_or_absent=%d problems=%d" % (
            os.path.relpath(t, reduced), len(rows), len(prows), n_in, n_ver, n_pr, len(bad)))
        by_src = _group(rows, ("code", "source"))
        pby_src = _group(prows, ("code", "source"))
        for (code, source), srows in by_src.items():
            path = resolve(source, roots)
            who = "%s:%s" % (code, source)
            if path is None:
                problems.append("%s: raw source not found under the roots" % who)
                continue
            prow_src = pby_src.get((code, source), [])
            plane_rows = [r for r in srows if r["family"] == "plane"]
            for r in srows:
                if r["family"] != "plane":
                    nonrow.append("%s %s/%s/%s: family %s is not recomputed" % (who, r["weights"], r["basis"], r["set"], r["family"]))
            try:
                src = open_source(code, path)
            except (OSError, ValueError, KeyError) as e:
                problems.append("%s: raw source unreadable (%s: %s)" % (who, type(e).__name__, e))
                continue
            realizations[code].add(who)
            if code == "consumer" and src.arrays is not None:
                with_arrays.add(who)
            if code == "consumer" and not src.csv and src.arrays is None:
                problems.append("%s: neither moment CSVs nor arrays present" % who)
                continue
            before = tally.n
            try:
                compare_plane_rows(tally, src, plane_rows, who, not_covered)
                compare_planes_table(tally, src, prow_src, who, not_covered)
            except (ValueError, KeyError, IndexError) as e:
                problems.append("%s: recomputation failed (%s: %s)" % (who, type(e).__name__, e))
            if tally.n == before:
                not_covered.append("%s: nothing compared" % who)
    P("")
    P("COVERAGE (values compared per code / family / origin / column, realizations covered, max relative difference)")
    for (code, fam, origin, col), c in sorted(tally.cov.items()):
        P("  %-8s %-12s %-9s %-12s values=%-6d realizations=%-4d max_rel=%.3e" % (code, fam, origin, col, c["n"], len(c["who"]), c["max"]))
    P("REALIZATIONS consumer=%d (with per-particle arrays %d) par2=%d" % (len(realizations["consumer"]), len(with_arrays), len(realizations["par2"])))
    for tag, items in (("NOT_COVERED", not_covered), ("NOT_RECOMPUTED_ROWS", nonrow)):
        agg = OrderedDict()
        for s in items:
            agg.setdefault(re.sub(r"^\S+ ", "", s), []).append(s)
        for k, v in agg.items():
            P("%s %d x %s" % (tag, len(v), k))
    for s in problems:
        P("PROBLEM %s" % s)
    for s in tally.fails[:200]:
        P(s)
    if len(tally.fails) > 200:
        P("... %d further failures" % (len(tally.fails) - 200))
    P("RECOMPUTE compared=%d beyond_tolerance=%d problems=%d max_rel=%.3e" % (tally.n, len(tally.fails), len(problems), tally.max_rel))
    code = 1 if (tally.fails or problems) else (2 if tally.n == 0 else 0)
    P("RECOMPUTE %s" % ("PASS" if code == 0 else "FAIL"))
    return code
