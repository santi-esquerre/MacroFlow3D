"""SF-42 N5: pruning of per-particle arrays after the reduced tables are written and verified (decision G8).

`prune(arms, retained_file, table, root, execute, log)`:
  1. HARD GUARD: any path (arm, root, table, manifest, retained list or entry, log) whose text or realpath contains
     `sf39_campaign` is refused; there is no override in this increment.
  2. The manifest `<table>.manifest.json` (written by `sf42red reduce`) must list the table's sha256 and the table must match
     it; every recorded input file (relative to `--root`) must exist with the recorded size and sha256 (verification failure
     refuses everything).
  3. Coverage: every realization directory of every arm (`<arm>/realizations/r_%04d`, which must have `DONE`) and each of its
     pass roots must appear as a `source` of a consumer plane row of the table.
  4. The sha256 of EVERY `.npy` under the arms' realization directories is recorded (kept and deleted ones alike).
  5. Only the `.npy` of realizations NOT in the retained list are deletion candidates.  The retained list is a text file with
     one realization directory per line, relative to `--root` (e.g. `armA/realizations/r_0003`), '#' comments; an entry that
     is not a realization of the arms refuses.
  Dry run by default (prints the plan, writes nothing); `execute=True` needs a log path, writes the log (path, size, sha256,
  action) BEFORE deleting, then deletes only `*.npy` files inside the arms' realization directories.  CSV, JSON and every
  other file are never touched.
"""
import csv
import json
import os
import re

from .reduce import ReduceError, sha256_file

GUARD = "sf39_campaign"


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
    if not os.path.isdir(rd):
        raise PruneRefused("%s has no realizations/ directory" % arm)
    out = [os.path.join(rd, n) for n in sorted(os.listdir(rd)) if re.fullmatch(r"r_\d{4}", n)]
    if not out:
        raise PruneRefused("%s has no realization directory" % arm)
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


def _covered_sources(table):
    with open(table) as f:
        return {r["source"] for r in csv.DictReader(f) if r["code"] == "consumer" and r["family"] == "plane"}


def _retained(path, root):
    out = set()
    with open(path) as f:
        for ln in f:
            ln = ln.split("#", 1)[0].strip()
            if ln:
                _guard(ln)
                out.add(os.path.normpath(os.path.join(root, ln)))
    return out


def prune(arms, retained_file, table, root, execute=False, log=None, out=print):
    """Plan (and with `execute` carry out) the deletion; returns the list of deleted (or, in a dry run, planned) paths."""
    root = os.path.abspath(root)
    arms = [os.path.abspath(a) for a in arms]
    _guard(root, table, table + ".manifest.json", retained_file, log, *arms)
    if execute and not log:
        raise PruneRefused("--execute needs a --log path")
    verify(table, root)
    covered = _covered_sources(table)
    keep = _retained(retained_file, root)
    all_R = []
    for arm in arms:
        if os.path.commonpath([root, arm]) != root:
            raise PruneRefused("arm %s is outside the root %s" % (arm, root))
        for R in _realizations(arm):
            if not os.path.isfile(os.path.join(R, "DONE")):
                raise PruneRefused("realization %s is not DONE" % R)
            for pr in _pass_roots(R):
                if os.path.relpath(pr, root) not in covered:
                    raise PruneRefused("the reduced table has no row for %s" % os.path.relpath(pr, root))
            all_R.append(R)
    unknown = keep - set(all_R)
    if unknown:
        raise PruneRefused("retained entries that are not realizations of the arms: %s" % sorted(os.path.relpath(u, root) for u in unknown))
    records = []  # (path, size, sha, action)
    for R in all_R:
        for dp, _, fn in os.walk(R):
            for n in sorted(fn):
                if n.endswith(".npy"):
                    p = os.path.join(dp, n)
                    records.append((p, os.path.getsize(p), sha256_file(p), "keep" if R in keep else "delete"))
    todo = [r for r in records if r[3] == "delete"]
    out("%s: %d .npy recorded, %d retained realization(s), %d file(s) / %d bytes to delete"
        % ("EXECUTE" if execute else "DRY RUN", len(records), len(keep), len(todo), sum(r[1] for r in todo)))
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
