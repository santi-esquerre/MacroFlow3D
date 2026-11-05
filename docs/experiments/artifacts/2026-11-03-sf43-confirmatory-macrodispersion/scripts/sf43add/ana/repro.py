"""Reproduction check of ``preregistration-addendum-1.md`` 3.1.1 (the gate of J1): ``j1_base10`` against the frozen ``s4_g5p``.

Every numeric column of every reduced row present in both (rows matched by ``seed, family, weights, basis, set, d_lo, d_hi, hmax_div, code``;
columns ``source``, ``r`` and path / timing columns excluded) must be equal within 1e-12 relative (1e-15 absolute where a value is 0).  Par2
rows carry ``seed = nan`` and the realization index ``r``: the seed is ``seed0 + r`` with ``seed0`` the segment directory's ``u<unit>_s<seed0>``
(10-seed units).  Strict (a tightening): a row present on one side only, a different column set or a duplicated row is a FAIL.

The Par2 crossing CSV of every seed is compared by sha256 (bitwise identity is the expectation; REPORTED, it does not gate): the campaign
side from ``<campaign>/segments/<arm>/<segment>/out/crossings/realization_%04d_planes.csv`` if present, else from the ``files`` list of the
segment's ``.csv.manifest.json``; the frozen side from ``--frozen-segments`` if given and present, else from the frozen ``.manifest.json``
(itself verified against the freeze manifest).  Only counts and the largest relative difference are reported, never a value of alpha.
"""
import csv
import glob
import hashlib
import json
import math
import os
import re
import sys

from sf43ana.loader import LoadError

from .data import Frozen

REL_TOL, ABS_TOL = 1e-12, 1e-15
KEY = ("seed", "family", "weights", "basis", "set", "d_lo", "d_hi", "hmax_div", "code")
EXCLUDED = {"source", "r"}
_TIMING = re.compile(r"(^|_)(path|wall|elapsed|seconds|timing)(_|$)")  # `U_time`, `t_end` are physical quantities and stay compared


def excluded(col):
    return col in EXCLUDED or bool(_TIMING.search(col))


def parse_seeds(spec):
    out = []
    for part in str(spec).split(","):
        lo, _, hi = part.partition("-")
        out.extend(range(int(lo), int(hi or lo) + 1))
    return sorted(set(out))


def seed0_of(segdir):
    m = re.search(r"_s(\d+)$", segdir)
    if not m:
        raise LoadError("segment directory %r is not of the form u<unit>_s<seed0>" % segdir)
    return int(m.group(1))


def _num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _bound(x):
    """Window bound of a row key; the ``time`` family rows have none (empty)."""
    return float(x) if x.strip() else ""


def tables(reduced, arm, seg="*"):
    """Paths of the window tables ``<reduced>/<arm>/<segment>/<arm>__<segment>__d<lo>-<hi>.csv`` (no planes / manifests)."""
    return sorted(p for p in glob.glob(os.path.join(reduced, arm, seg, "%s__*__d*.csv" % arm)) if not p.endswith(".planes.csv"))


def read_side(reduced, arm, seeds, verify=None):
    """-> ({key: row}, {seed: segdir}, columns) of the rows of ``seeds``; ``verify(path)`` is called before each file is parsed."""
    want, rows, seg_of, cols = set(seeds), {}, {}, None
    for p in tables(reduced, arm):
        if verify:
            verify(p)
        seg = os.path.basename(os.path.dirname(p))
        with open(p, newline="") as f:
            rd = csv.DictReader(f)
            if cols is not None and list(rd.fieldnames) != cols:
                raise LoadError("%s: different column set than the other tables of the arm" % p)
            cols = list(rd.fieldnames)
            for row in rd:
                s = _num(row["seed"])
                seed = seed0_of(seg) + int(float(row["r"])) if s is None or math.isnan(s) else int(s)
                if seed not in want:
                    continue
                k = (seed, row["family"], row["weights"], row["basis"], row["set"], _bound(row["d_lo"]), _bound(row["d_hi"]),
                     int(float(row["hmax_div"])), row["code"])
                if k in rows:
                    raise LoadError("%s: duplicated row %r (the same seed in two segments or windows)" % (p, k))
                rows[k] = row
                seg_of[seed] = seg
    return rows, seg_of, cols or []


def cell_diff(x, y):
    """-> (ok, difference) of two table cells: numeric within the tolerances, else string equality."""
    a, b = _num(x), _num(y)
    if a is None or b is None:
        return x == y, 0.0
    if a == b or (math.isnan(a) and math.isnan(b)):
        return True, 0.0
    if not (math.isfinite(a) and math.isfinite(b)):
        return False, math.inf
    d = abs(a - b)
    if a == 0.0 or b == 0.0:
        return d <= ABS_TOL, d
    r = d / max(abs(a), abs(b))
    return r <= REL_TOL, r


def _sha_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _manifest_sha(table_path, rel, verify=None):
    mp = table_path + ".manifest.json"
    if not os.path.isfile(mp):
        return None
    if verify:
        verify(mp)
    with open(mp) as f:
        return {e["path"]: e["sha256"] for e in json.load(f).get("files", [])}.get(rel)


def crossing_sha(seg_root, reduced, arm, seg, seed, verify=None):
    """sha256 of the crossing CSV of ``seed``: the file under ``seg_root`` if present, else the segment's table manifest (None: unavailable)."""
    rel = "out/crossings/realization_%04d_planes.csv" % (seed - seed0_of(seg))
    if seg_root and os.path.isfile(os.path.join(seg_root, arm, seg, rel)):
        return _sha_file(os.path.join(seg_root, arm, seg, rel))
    for tp in tables(reduced, arm, seg):
        h = _manifest_sha(tp, rel, verify)
        if h:
            return h
    return None


def compare(campaign, arm, seeds, frozen_reduced, frozen_manifest, frozen_segments=None, frozen_arm="s4_g5p", expected_manifest_sha=None, fro=None):
    """-> dict: ``passed``, ``n_seeds`` (seeds with rows on both sides), ``n_seeds_requested``, ``n_rows``, ``max_rel``, ``csv_sha_equal`` /
    ``csv_sha_differ`` / ``csv_sha_unavailable``, ``problems`` (names and keys only, no values)."""
    fro = fro or Frozen(frozen_reduced, frozen_manifest, expected_manifest_sha)
    seeds = list(seeds)
    a_rows, a_seg, a_cols = read_side(os.path.join(campaign, "reduced"), arm, seeds)
    b_rows, b_seg, b_cols = read_side(fro.dir, frozen_arm, seeds, verify=fro.verify)
    problems = []
    if not a_rows:
        problems.append("no reduced rows of arm %s for the requested seeds in the campaign" % arm)
    if not b_rows:
        problems.append("no frozen rows of arm %s for the requested seeds" % frozen_arm)
    if a_rows and b_rows and a_cols != b_cols:
        problems.append("column sets differ: only campaign %s, only frozen %s" % (sorted(set(a_cols) - set(b_cols)), sorted(set(b_cols) - set(a_cols))))
    only_a, only_b = sorted(set(a_rows) - set(b_rows), key=repr), sorted(set(b_rows) - set(a_rows), key=repr)
    if only_a or only_b:
        problems.append("rows on one side only: campaign %d, frozen %d (first keys %s, %s)" % (len(only_a), len(only_b), only_a[:1], only_b[:1]))
    max_rel, n_rows, bad = 0.0, 0, {}
    for k in sorted(set(a_rows) & set(b_rows), key=repr):
        n_rows += 1
        for c in a_cols:
            if excluded(c) or c not in b_cols:
                continue
            ok, d = cell_diff(a_rows[k][c], b_rows[k][c])
            max_rel = max(max_rel, d)
            if not ok:
                bad.setdefault(c, []).append(k[0])
    for c, ss in sorted(bad.items()):
        problems.append("column %s outside the tolerance in %d row(s), seeds %s" % (c, len(ss), sorted(set(ss))[:5]))
    both = sorted(set(a_seg) & set(b_seg))
    if len(both) != len(seeds):
        problems.append("only %d of %d requested seeds have rows on both sides" % (len(both), len(seeds)))
    eq = diff = na = 0
    for s in seeds:
        ha = crossing_sha(os.path.join(campaign, "segments"), os.path.join(campaign, "reduced"), arm, a_seg[s], s) if s in a_seg else None
        hb = crossing_sha(frozen_segments, fro.dir, frozen_arm, b_seg[s], s, fro.verify) if s in b_seg else None
        if ha is None or hb is None:
            na += 1
        elif ha == hb:
            eq += 1
        else:
            diff += 1
    return {"evaluated": True, "passed": not problems, "n_seeds": len(both), "n_seeds_requested": len(seeds), "n_rows": n_rows, "max_rel": max_rel,
            "csv_sha_equal": eq, "csv_sha_differ": diff, "csv_sha_unavailable": na, "problems": problems, "arm": arm, "frozen_arm": frozen_arm,
            "seeds": [seeds[0], seeds[-1]], "rel_tol": REL_TOL, "abs_tol_zero": ABS_TOL, "frozen_files_verified": len(fro.verified)}


def line(r):
    return "REPRO %s %d %d %.3g csv_sha_equal=%d/%d" % ("PASS" if r["passed"] else "FAIL", r["n_seeds"], r["n_rows"], r["max_rel"], r["csv_sha_equal"],
                                                       r["n_seeds_requested"])


def main(a):
    try:
        r = compare(a.campaign, a.arm, parse_seeds(a.seeds), a.frozen_reduced, a.frozen_manifest, a.frozen_segments, a.frozen_arm)
    except (LoadError, OSError, ValueError) as e:
        print("repro: %s" % e, file=sys.stderr)
        return 2
    print(line(r))
    for p in r["problems"]:  # names, counts and keys only
        print("repro problem: %s" % p, file=sys.stderr)
    print("repro crossing sha256: equal=%d differ=%d unavailable=%d" % (r["csv_sha_equal"], r["csv_sha_differ"], r["csv_sha_unavailable"]), file=sys.stderr)
    return 0 if r["passed"] else 1
