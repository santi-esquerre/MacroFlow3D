"""SF-42 N5: pruning of per-particle arrays after the reduced tables are written and verified (decision G8).

`prune(arms, retained_file, table, root, execute, log, min_retained)`:
  1. HARD GUARD: any path (arm, root, table, manifest, retained list or entry, log) whose text or realpath contains
     `sf39_campaign` is refused; there is no override in this increment.
  2. The manifest `<table>.manifest.json` (written by `sf42red reduce`) must list the table's sha256 and the table must match
     it; every recorded input file (relative to `--root`) must exist with the recorded size and sha256 (verification failure
     refuses everything).
  3. Coverage: every realization directory of every arm (`<arm>/realizations/r_%04d`, which must have `DONE`) and each of its
     pass roots must appear as a `source` of a consumer plane row of the table, and at least one such row must carry a finite
     plane estimator (a pass whose reduced rows are all NaN is NOT verified and refuses).
  4. The sha256 of EVERY `.npy` under the arms' realization directories is recorded (kept and deleted ones alike).
  5. Only the `.npy` of realizations NOT in the retained list are deletion candidates.  The retained list is a text file with
     one realization directory per line, relative to `--root` (e.g. `armA/realizations/r_0003`), '#' comments; an entry that
     is not a realization of the arms refuses.  Decision G8: at least `min_retained` (default 5, the G8 number) realizations
     of every arm are retained (an arm with fewer realizations must be retained whole).  The CLI default is the G8 number.
     SF-43 (disclosed change to this SF-42 tooling): `--min-retained N` (CLI) / `min_retained=N` (API) lowers it, down to 0, because
     the SF-43 campaign prunes per segment directory (a consecutive seed range of one arm) against an ARM-level retained list (the
     first seeds of the arm plus a human-review sample), so a single segment may legitimately retain fewer than 5 (or none); the G8
     arm-level minimum is then enforced by the SF-43 caller.  The value used is reported in the plan line.
  6. Symlinks: the `realizations/` directory, every realization directory and every directory below one must not be a symlink,
     and the realpath of every `.npy` that would be deleted must lie inside the realpath of its arm and not contain
     `sf39_campaign`; any violation refuses BEFORE anything is deleted (`os.walk` never follows links).
  SF-43 (disclosed, backward-compatible): `--keep-arrays name,...` (CLI) / `keep_arrays=(name, ...)` (API) names `.npy` arrays
     (file name without the extension, e.g. `seeds`, `weights`, `planes_x23`, `planes_tau`) that are NEVER deletion candidates in any
     realization (recorded with action `keep` in the log); the default is empty, which is the previous behaviour.  The SF-43 campaign
     uses it to keep the plane arrays of every realization of its primary arms (the independent recomputation then covers the
     displacement basis for all of them) and to delete only the others (time-pass and Pollock arrays).
  SF-43 (disclosed, backward-compatible): the Pollock bridge arrays (`<realization>/pollock/pollock_*.npy`) are deletion candidates only
     when the table holds `code=pollock` plane rows of that directory (`source` = `<realization>/pollock`) with a finite plane estimator
     (`sf42red reduce --pollock`); otherwise they are recorded with action `keep` (never deleted unreduced) and the plan line says how
     many.  Tables without Pollock arrays on disk behave exactly as before.
  Dry run by default (prints the plan, writes nothing); `execute=True` needs a log path, writes the log (path, size, sha256,
  action) BEFORE deleting, then deletes only `*.npy` files inside the arms' realization directories.  CSV, JSON and every
  other file are never touched.
"""
import csv
import json
import math
import os
import re

from .reduce import PLANE_EST, ReduceError, sha256_file

GUARD = "sf39_campaign"
MIN_RETAINED = 5  # decision G8: at least 5 retained realizations per arm


class PruneRefused(ReduceError):
    """The pruning was refused (guard, failed verification, incomplete coverage, bad retained list)."""


def _guard(*paths):
    for p in paths:
        if p is None:
            continue
        for q in (str(p), os.path.realpath(str(p))):
            if GUARD in q:
                raise PruneRefused("refusing: path contains '%s' (hard guard, no override): %s" % (GUARD, p))


def _realizations(arm):
    rd = os.path.join(arm, "realizations")
    if os.path.islink(rd):
        raise PruneRefused("%s is a symlink" % rd)
    if not os.path.isdir(rd):
        raise PruneRefused("%s has no realizations/ directory" % arm)
    out = [os.path.join(rd, n) for n in sorted(os.listdir(rd)) if re.fullmatch(r"r_\d{4}", n)]
    if not out:
        raise PruneRefused("%s has no realization directory" % arm)
    for R in out:
        if os.path.islink(R):
            raise PruneRefused("realization directory %s is a symlink" % R)
    return out


def _pass_roots(R):
    if os.path.isfile(os.path.join(R, "track.json")):
        return [R]
    roots = [os.path.join(R, d) for d in sorted(os.listdir(R)) if re.fullmatch(r"hmax_\d+", d)]
    if not roots:
        raise PruneRefused("no pass root in %s" % R)
    return roots


def verify(table, root):
    """Raise PruneRefused unless the manifest of `table` matches the table and every recorded input matches the disk."""
    man_path = table + ".manifest.json"
    if not os.path.isfile(man_path) or not os.path.isfile(table):
        raise PruneRefused("missing table or manifest (%s, %s)" % (table, man_path))
    with open(man_path) as f:
        man = json.load(f)
    if man.get("format") != "sf42red_manifest_v1":
        raise PruneRefused("unknown manifest format in %s" % man_path)
    by_name = {t["path"]: t["sha256"] for t in man["tables"]}
    if by_name.get(os.path.basename(table)) != sha256_file(table):
        raise PruneRefused("table %s is not the one the manifest recorded" % table)
    bad = []
    for e in man["files"]:
        p = os.path.join(root, e["path"])
        if not os.path.isfile(p) or os.path.getsize(p) != e["size"] or sha256_file(p) != e["sha256"]:
            bad.append(e["path"])
    if bad:
        raise PruneRefused("%d recorded input(s) differ from the disk or are missing, e.g. %s" % (len(bad), bad[:3]))
    return man


def _covered_sources(table, code="consumer"):
    """(sources with a plane row of `code`, sources with such a row that has a finite plane estimator)."""
    seen, finite = set(), set()
    with open(table) as f:
        for r in csv.DictReader(f):
            if r["code"] == code and r["family"] == "plane":
                seen.add(r["source"])
                if any(r[e] != "" and math.isfinite(float(r[e])) for e in PLANE_EST):
                    finite.add(r["source"])
    return seen, finite


def _retained(path, root):
    out = set()
    with open(path) as f:
        for ln in f:
            ln = ln.split("#", 1)[0].strip()
            if ln:
                _guard(ln)
                out.add(os.path.normpath(os.path.join(root, ln)))
    return out


def prune(arms, retained_file, table, root, execute=False, log=None, out=print, min_retained=MIN_RETAINED, keep_arrays=()):
    """Plan (and with `execute` carry out) the deletion; returns the list of deleted (or, in a dry run, planned) paths."""
    root = os.path.abspath(root)
    arms = [os.path.abspath(a) for a in arms]
    _guard(root, table, table + ".manifest.json", retained_file, log, *arms)
    if execute and not log:
        raise PruneRefused("--execute needs a --log path")
    if min_retained < 0:
        raise PruneRefused("min_retained must be >= 0")
    keep_arrays = set(keep_arrays)
    if any(not re.fullmatch(r"[A-Za-z0-9_]+", k) for k in keep_arrays):
        raise PruneRefused("keep_arrays entries must be array names without extension, got %s" % sorted(keep_arrays))
    verify(table, root)
    covered, finite = _covered_sources(table)
    pollock_ok = _covered_sources(table, "pollock")[1]
    keep = _retained(retained_file, root)
    all_R, per_arm = [], []
    real_root = os.path.realpath(root)
    for arm in arms:
        if os.path.commonpath([root, arm]) != root or os.path.commonpath([real_root, os.path.realpath(arm)]) != real_root:
            raise PruneRefused("arm %s is outside the root %s" % (arm, root))
        Rs = _realizations(arm)
        for R in Rs:
            if not os.path.isfile(os.path.join(R, "DONE")):
                raise PruneRefused("realization %s is not DONE" % R)
            for pr in _pass_roots(R):
                src = os.path.relpath(pr, root)
                if src not in covered:
                    raise PruneRefused("the reduced table has no row for %s" % src)
                if src not in finite:
                    raise PruneRefused("the reduced rows of %s are all NaN: not verified for deletion" % src)
            all_R.append(R)
        per_arm.append((arm, Rs))
    unknown = keep - set(all_R)
    if unknown:
        raise PruneRefused("retained entries that are not realizations of the arms: %s" % sorted(os.path.relpath(u, root) for u in unknown))
    for arm, Rs in per_arm:  # decision G8: a minimum retained subset per arm (an arm smaller than the minimum is retained whole)
        need, have = min(min_retained, len(Rs)), len(keep & set(Rs))
        if have < need:
            raise PruneRefused("arm %s retains %d realization(s), decision G8 requires at least %d" % (arm, have, need))
    records, n_unreduced = [], 0  # records: (path, size, sha, action)
    for R in all_R:
        real_R = os.path.realpath(R)
        for dp, dn, fn in os.walk(R, followlinks=False):
            for d in dn:
                if os.path.islink(os.path.join(dp, d)):
                    raise PruneRefused("symlinked directory %s below %s" % (os.path.join(dp, d), R))
            for n in sorted(fn):
                if n.endswith(".npy"):
                    p = os.path.join(dp, n)
                    rp = os.path.realpath(p)
                    _guard(p)
                    if os.path.commonpath([real_R, rp]) != real_R:
                        raise PruneRefused("%s resolves outside its realization directory (%s)" % (p, rp))
                    kept = R in keep or n[:-4] in keep_arrays
                    unreduced = not kept and n.startswith("pollock_") and os.path.relpath(dp, root) not in pollock_ok
                    n_unreduced += unreduced  # only the arrays this rule saves from deletion
                    records.append((p, os.path.getsize(p), sha256_file(p), "keep" if kept or unreduced else "delete"))
    todo = [r for r in records if r[3] == "delete"]
    out("%s: %d .npy recorded, %d retained realization(s) (min per arm %d)%s%s, %d file(s) / %d bytes to delete"
        % ("EXECUTE" if execute else "DRY RUN", len(records), len(keep), min_retained,
           (", kept arrays %s" % sorted(keep_arrays)) if keep_arrays else "",
           (", %d unreduced pollock array(s) kept" % n_unreduced) if n_unreduced else "", len(todo), sum(r[1] for r in todo)))
    if not execute:
        for r in todo:
            out("would delete %s" % os.path.relpath(r[0], root))
        return [r[0] for r in todo]
    with open(log, "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["path", "size", "sha256", "action"])
        for p, size, sha, act in records:
            wr.writerow([os.path.relpath(p, root), size, sha, act])
        f.flush()
        os.fsync(f.fileno())
    deleted = []
    for p, _, _, _ in todo:
        if not p.endswith(".npy") or not any(p.startswith(R + os.sep) for R in all_R):
            raise PruneRefused("internal check: refusing to delete %s" % p)
        os.remove(p)
        deleted.append(p)
    with open(log, "a") as f:
        f.write("# deleted %d files, %d bytes\n" % (len(deleted), sum(r[1] for r in todo)))
    return deleted
