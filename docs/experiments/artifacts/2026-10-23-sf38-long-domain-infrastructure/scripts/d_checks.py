#!/usr/bin/env python3
"""SF-38 threshold D (RR7) comparison tools for run_sf38.sh part `ensemble`.

  d_checks.py tree-cmp   --name N --ref DIR --cand DIR [--exclude BASENAME ...] [--path REL ...]
  d_checks.py csv-cmp    --name N --ref FILE --cand FILE [--rtol 1e-12] [--atol 1e-300]
  d_checks.py ttime-cmp  --name N --log RUNNER.log --stats-dir DIR [--rtol 1e-12] [--t-rtol 1e-9]
  d_checks.py walls      --name N --root DIR
  d_checks.py --selftest

tree-cmp  : every file of --ref (restricted to --path prefixes when given) must exist in --cand and be byte-identical; files
            whose basename is in --exclude are ignored on both sides (RR7: exactly `timings.json`); files only in --cand
            (inside the --path restriction) are `extra`.  Output
              D_BYTES name= files_ref= files_cand= identical= diff= missing= extra= excluded= IDENTICAL|DIFF
              D_BYTES_DIFF name= kind=diff|missing|extra path=          (at most 20)
            exit 0 iff IDENTICAL.
csv-cmp   : two CSV files (leading `#` lines skipped, one header line, numeric rows): same header, same shape, every value
            equal, or relative difference <= rtol of the reference value, or (reference 0) absolute difference <= atol; NaN
            equals NaN.  Output
              D_CSV name= ref= cand= rows= cols= max_rel= max_abs_at_zero= byte_equal=0|1 within=1|0 PASS|FAIL
ttime-cmp : the `R=<r> TTIME n= t= active= mean1..3= var1..3=` lines of the runner log (printed with %.12e; t with %.10g)
            against the rows of stats-dir/realization_%04d_timeseries.csv (read as `read_timeseries_csv` does: skip `#` lines
            and one header line).  The moments are compared at rtol (the %.12e print loses <= 5e-13 relative), `active`
            exactly, the time column at t-rtol (the log prints it with %.10g, 5e-10 relative resolution; RR7 asks 1e-12 for
            the moments).  Output
              D_TTIME name= realizations= rows= max_rel_moments= max_rel_t= active_equal=1|0 PASS|FAIL
walls     : RECORDED D_WALL name= r= total_s= stage_s= prepare_s= track_s= bridge_repack_s= bridge_track_s= write_s=
            from root/realizations/r_*/timings.json (wall-clock values live only there).
Stdlib only.
"""
import filecmp
import glob
import json
import math
import os
import re
import sys


# ---------------------------------------------------------------- tree comparison
def list_files(root, excludes, paths):
    out = set()
    for dp, _, fn in os.walk(root):
        for f in fn:
            if f in excludes:
                continue
            rel = os.path.relpath(os.path.join(dp, f), root)
            if paths and not any(rel == p or rel.startswith(p.rstrip("/") + "/") for p in paths):
                continue
            out.add(rel)
    return out


def excluded_count(root, excludes):
    n = 0
    for _, _, fn in os.walk(root):
        n += sum(1 for f in fn if f in excludes)
    return n


def tree_cmp(name, ref, cand, excludes, paths):
    lines = []
    if not os.path.isdir(ref) or not os.path.isdir(cand):
        return ["D_BYTES name=%s files_ref=- files_cand=- identical=0 diff=0 missing=1 extra=0 excluded=0 DIFF (missing directory %s)"
                % (name, ref if not os.path.isdir(ref) else cand)], False
    fr, fc = list_files(ref, excludes, paths), list_files(cand, excludes, paths)
    ident, diffs, missing, extra = 0, [], sorted(fr - fc), sorted(fc - fr)
    for rel in sorted(fr & fc):
        if filecmp.cmp(os.path.join(ref, rel), os.path.join(cand, rel), shallow=False):
            ident += 1
        else:
            diffs.append(rel)
    ok = not diffs and not missing and not extra and len(fr) > 0
    lines.append("D_BYTES name=%s files_ref=%d files_cand=%d identical=%d diff=%d missing=%d extra=%d excluded=%d/%d %s"
                 % (name, len(fr), len(fc), ident, len(diffs), len(missing), len(extra), excluded_count(ref, excludes),
                    excluded_count(cand, excludes), "IDENTICAL" if ok else "DIFF"))
    shown = 0
    for kind, lst in (("diff", diffs), ("missing", missing), ("extra", extra)):
        for rel in lst:
            if shown < 20:
                lines.append("D_BYTES_DIFF name=%s kind=%s path=%s" % (name, kind, rel))
                shown += 1
    return lines, ok


# ---------------------------------------------------------------- csv comparison
def read_csv_numeric(path):
    with open(path, errors="replace") as f:
        lines = f.read().splitlines()
    i = 0
    while i < len(lines) and lines[i].lstrip().startswith("#"):
        i += 1
    header = lines[i].strip() if i < len(lines) else ""
    rows = []
    for ln in lines[i + 1:]:
        if ln.strip():
            rows.append([float(x) for x in ln.strip().split(",")])
    return header, rows


def close(a, b, rtol, atol):
    """-> (ok, rel or None, abs_at_zero or None) for candidate a against reference b."""
    if a != a or b != b:
        return (a != a and b != b), None, None
    if a == b:
        return True, 0.0, None
    if b == 0.0:
        return abs(a) <= atol, None, abs(a)
    rel = abs(a - b) / abs(b)
    return rel <= rtol, rel, None


def csv_cmp(name, ref, cand, rtol, atol):
    try:
        hr, rr = read_csv_numeric(ref)
        hc, rc = read_csv_numeric(cand)
    except (OSError, ValueError) as e:
        return ["D_CSV name=%s ref=%s cand=%s rows=- cols=- max_rel=- max_abs_at_zero=- byte_equal=0 within=0 FAIL (%s)"
                % (name, ref, cand, e)], False
    byte_eq = filecmp.cmp(ref, cand, shallow=False)
    ok = hr == hc and len(rr) == len(rc) and len(rr) > 0 and all(len(a) == len(b) for a, b in zip(rr, rc))
    max_rel, max_abs0 = 0.0, 0.0
    if ok:
        for a, b in zip(rc, rr):
            for x, y in zip(a, b):
                g, rel, a0 = close(x, y, rtol, atol)
                ok = ok and g
                if rel is not None:
                    max_rel = max(max_rel, rel)
                if a0 is not None:
                    max_abs0 = max(max_abs0, a0)
    cols = len(rr[0]) if rr else 0
    return ["D_CSV name=%s ref=%s cand=%s rows=%d/%d cols=%d max_rel=%.3e max_abs_at_zero=%.3e byte_equal=%d within=%d %s"
            % (name, ref, cand, len(rr), len(rc), cols, max_rel, max_abs0, int(byte_eq), int(ok), "PASS" if ok else "FAIL")], ok


# ---------------------------------------------------------------- TTIME lines vs stats CSV
TT = re.compile(r"^R=(\d+) TTIME n=(\d+) t=(\S+) active=(-?\d+) mean1=(\S+) mean2=(\S+) mean3=(\S+) var1=(\S+) var2=(\S+) var3=(\S+)")


def read_ttime(log):
    d = {}
    with open(log, errors="replace") as f:
        for ln in f:
            m = TT.match(ln)
            if m:
                r, n = int(m.group(1)), int(m.group(2))
                d.setdefault(r, {})[n] = (float(m.group(3)), int(m.group(4)), [float(m.group(i)) for i in range(5, 11)])
    return d


def read_timeseries_like(path):
    """as MacrodispersionAnalysis read_timeseries_csv: skip leading '#' lines and one header line."""
    h, rows = read_csv_numeric(path)
    return rows


def ttime_cmp(name, log, stats_dir, rtol, t_rtol):
    try:
        tt = read_ttime(log)
    except OSError as e:
        return ["D_TTIME name=%s realizations=- rows=- max_rel_moments=- max_rel_t=- active_equal=0 FAIL (%s)" % (name, e)], False
    if not tt:
        return ["D_TTIME name=%s realizations=0 rows=0 max_rel_moments=- max_rel_t=- active_equal=0 FAIL (no TTIME line)" % name], False
    ok, nrows, mrel, mt, act = True, 0, 0.0, 0.0, True
    for r in sorted(tt):
        path = os.path.join(stats_dir, "realization_%04d_timeseries.csv" % r)
        try:
            rows = read_timeseries_like(path)
        except (OSError, ValueError):
            ok = False
            continue
        if len(rows) != len(tt[r]) or sorted(tt[r]) != list(range(len(tt[r]))):
            ok = False
            continue
        for n in range(len(rows)):
            t, a, mom = tt[r][n]
            row = rows[n]  # t, mean_x..z, var_x..z, active
            g, rel, _ = close(t, row[0], t_rtol, 1e-300)
            ok = ok and g
            mt = max(mt, rel or 0.0)
            for x, y in zip(mom, row[1:7]):
                g, rel, _ = close(x, y, rtol, 1e-300)
                ok = ok and g
                mrel = max(mrel, rel or 0.0)
            if a != int(row[7]):
                act = False
            nrows += 1
    ok = ok and act
    return ["D_TTIME name=%s realizations=%d rows=%d max_rel_moments=%.3e max_rel_t=%.3e active_equal=%d %s"
            % (name, len(tt), nrows, mrel, mt, int(act), "PASS" if ok else "FAIL")], ok


# ---------------------------------------------------------------- wall times
def walls(name, root):
    out = []
    for p in sorted(glob.glob(os.path.join(root, "realizations", "r_*", "timings.json"))):
        try:
            with open(p) as f:
                j = json.load(f)
        except (OSError, ValueError):
            out.append("RECORDED D_WALL name=%s path=%s unreadable" % (name, p))
            continue
        s = j.get("seconds", {})
        out.append("RECORDED D_WALL name=%s r=%s total_s=%s stage_s=%s prepare_s=%s track_s=%s bridge_repack_s=%s bridge_track_s=%s write_s=%s"
                   % (name, j.get("r"), *(("%.3f" % s[k]) if k in s else "-" for k in
                                         ("total", "stage", "prepare", "track", "bridge_repack", "bridge_track", "write"))))
    return out


# ---------------------------------------------------------------- selftest
def selftest():
    import tempfile
    fails = []

    def expect(name, cond):
        if not cond:
            fails.append(name)
            print("SELFTEST FAIL %s" % name)

    def w(path, text):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)

    with tempfile.TemporaryDirectory() as d:
        A, B, C = (os.path.join(d, x) for x in "ABC")
        for root in (A, B, C):
            w(root + "/stats/realization_0000_timeseries.csv", "x\n")
            w(root + "/realizations/r_0000/DONE", "sig\n")
            w(root + "/realizations/r_0000/timings.json", json.dumps({"r": 0, "seconds": {"total": 1.5, "stage": 1.0}}) if root == A else
              json.dumps({"r": 0, "seconds": {"total": 9.5}}))
            w(root + "/analysis_config.yaml", "run_mode: analysis_only\n")
        lines, ok = tree_cmp("t", A, B, {"timings.json"}, [])
        expect("tree_identical_excluding_timings", ok and "IDENTICAL" in lines[0] and "excluded=1/1" in lines[0])
        lines, ok = tree_cmp("t", A, B, set(), [])
        expect("tree_timings_differ_when_not_excluded", not ok and "diff=1" in lines[0])
        w(B + "/stats/realization_0000_timeseries.csv", "y\n")
        lines, ok = tree_cmp("t", A, B, {"timings.json"}, [])
        expect("tree_diff", not ok and any("kind=diff path=stats/realization_0000_timeseries.csv" in l for l in lines))
        lines, ok = tree_cmp("t", A, B, {"timings.json"}, ["realizations/r_0000", "analysis_config.yaml"])
        expect("tree_subset_ignores_other_files", ok)
        w(C + "/extra.txt", "e\n")
        os.remove(C + "/analysis_config.yaml")
        lines, ok = tree_cmp("t", A, C, {"timings.json"}, [])
        expect("tree_missing_extra", not ok and "missing=1" in lines[0] and "extra=1" in lines[0])
        lines, ok = tree_cmp("t", A, os.path.join(d, "none"), set(), [])
        expect("tree_absent", not ok)
        # csv
        w(d + "/a.csv", "# format_version=1\nt,v\n1.0,2.0\n2.0,0.0\n")
        w(d + "/b.csv", "# format_version=1\nt,v\n1.0,2.0000000000005\n2.0,0.0\n")
        lines, ok = csv_cmp("c", d + "/a.csv", d + "/b.csv", 1e-12, 1e-300)
        expect("csv_within", ok and "byte_equal=0" in lines[0] and "PASS" in lines[0])
        w(d + "/b.csv", "# format_version=1\nt,v\n1.0,2.000000000005\n2.0,0.0\n")
        lines, ok = csv_cmp("c", d + "/a.csv", d + "/b.csv", 1e-12, 1e-300)
        expect("csv_outside", not ok)
        w(d + "/b.csv", "# format_version=1\nt,v\n1.0,2.0\n2.0,1e-299\n")
        lines, ok = csv_cmp("c", d + "/a.csv", d + "/b.csv", 1e-12, 1e-300)
        expect("csv_zero_reference_abs", not ok)
        w(d + "/b.csv", "# format_version=1\nt,v\n1.0,2.0\n2.0,0.0\n")
        lines, ok = csv_cmp("c", d + "/a.csv", d + "/b.csv", 1e-12, 1e-300)
        expect("csv_identical", ok and "byte_equal=1" in lines[0])
        w(d + "/n.csv", "t,v\n1.0,nan\n")
        lines, ok = csv_cmp("c", d + "/n.csv", d + "/n.csv", 1e-12, 1e-300)
        expect("csv_nan_equal", ok)
        # ttime
        sd = d + "/stats"
        hdr = "# format_version=1\nt,mean_x,mean_y,mean_z,var_x,var_y,var_z,active\n"
        row = [0.1, 1.0, 2.0, 3.0, 0.5, 0.25, 0.125]
        w(sd + "/realization_0000_timeseries.csv", hdr + ",".join("%.15e" % v for v in row) + ",100\n")
        logl = ("R=0 TTIME n=0 t=%.10g active=100 " % row[0] + " ".join("mean%d=%.12e" % (k + 1, row[1 + k]) for k in range(3)) + " "
                + " ".join("var%d=%.12e" % (k + 1, row[4 + k]) for k in range(3)) + "\nTIMING r=0 seconds=1\n")
        w(d + "/run.log", logl)
        lines, ok = ttime_cmp("tt", d + "/run.log", sd, 1e-12, 1e-9)
        expect("ttime_pass", ok and "rows=1" in lines[0] and "active_equal=1" in lines[0])
        w(d + "/run.log", logl.replace("mean1=1.000000000000e+00", "mean1=1.000000001000e+00"))
        lines, ok = ttime_cmp("tt", d + "/run.log", sd, 1e-12, 1e-9)
        expect("ttime_fail", not ok)
        w(d + "/run.log", logl.replace("active=100", "active=99"))
        lines, ok = ttime_cmp("tt", d + "/run.log", sd, 1e-12, 1e-9)
        expect("ttime_active_fail", not ok and "active_equal=0" in lines[0])
        w(d + "/run.log", "nothing\n")
        expect("ttime_none", not ttime_cmp("tt", d + "/run.log", sd, 1e-12, 1e-9)[1])
        # walls
        ws = walls("w", A)
        expect("walls", len(ws) == 1 and "total_s=1.500" in ws[0] and "stage_s=1.000" in ws[0] and "track_s=-" in ws[0])
    if fails:
        print("SELFTEST FAILED: %s" % ", ".join(fails))
        return 1
    print("SELFTEST OK d_checks")
    return 0


def main(argv):
    if len(argv) == 2 and argv[1] == "--selftest":
        return selftest()
    if len(argv) < 2 or argv[1] not in ("tree-cmp", "csv-cmp", "ttime-cmp", "walls"):
        print(__doc__, file=sys.stderr)
        return 2
    cmd = argv[1]
    o = {"--name": "", "--ref": None, "--cand": None, "--log": None, "--stats-dir": None, "--root": None, "--rtol": "1e-12",
         "--atol": "1e-300", "--t-rtol": "1e-9"}
    excl, paths = set(), []
    i = 2
    while i < len(argv):
        a = argv[i]
        if a == "--exclude" and i + 1 < len(argv):
            excl.add(argv[i + 1])
            i += 2
        elif a == "--path" and i + 1 < len(argv):
            paths.append(argv[i + 1])
            i += 2
        elif a in o and i + 1 < len(argv):
            o[a] = argv[i + 1]
            i += 2
        else:
            print("d_checks.py: bad argument %r" % a, file=sys.stderr)
            return 2
    if cmd == "tree-cmp" and o["--ref"] and o["--cand"]:
        lines, ok = tree_cmp(o["--name"], o["--ref"], o["--cand"], excl, paths)
    elif cmd == "csv-cmp" and o["--ref"] and o["--cand"]:
        lines, ok = csv_cmp(o["--name"], o["--ref"], o["--cand"], float(o["--rtol"]), float(o["--atol"]))
    elif cmd == "ttime-cmp" and o["--log"] and o["--stats-dir"]:
        lines, ok = ttime_cmp(o["--name"], o["--log"], o["--stats-dir"], float(o["--rtol"]), float(o["--t-rtol"]))
    elif cmd == "walls" and o["--root"]:
        print("\n".join(walls(o["--name"], o["--root"])))
        return 0
    else:
        print(__doc__, file=sys.stderr)
        return 2
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
