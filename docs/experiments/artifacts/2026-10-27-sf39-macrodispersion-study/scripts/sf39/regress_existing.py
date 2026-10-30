"""SF-39 regression of the existing amplitudes (addendum 2026-10-29, A6): the analysis re-run with s025 must reproduce the outputs of
`s1`, `s225` and `s4` of the reference analysis (evidence commit 4dbf31f2) identically.

    python3 -m sf39.regress_existing --ref <old analysis dir> --new <new analysis dir>

The amplitudes are fixed (`AMPS` = s1, s225, s4).  On real data `--ref` is an extraction of the committed analysis, so that nothing but the
evidence commit is compared, and `--new` is the new analysis output (README):

    git archive 4dbf31f2 docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study/analysis | tar -x -C <empty dir>

Guards against a vacuous pass (each refuses with exit 2 and compares nothing):
  * `--ref` and `--new` are the same directory (compared by realpath);
  * the reference has no `manifest.txt`, or a file it lists under `reduced/`, `series/` or as `summary.json` / `decisions.csv` is missing or
    does not match its sha256 (the bookkeeping entries completed after the manifest - `logs/`, `results.txt`, `run.env`, `exit_codes.tsv`,
    `walls.tsv`, `sha256.txt`, `PART_EXIT`, `.build_digest` - are not checked), or a file compared below is not listed;
  * the number of `reduced/*.csv` of the reference is 0 or differs from the number of `reduced/` entries of its manifest (53 for the evidence of commit 4dbf31f2, 68 for the four-amplitude evidence).
The report prints the sha256 of the reference manifest.

Checks (stdlib only):
  (a) parsed-JSON equality of `summary.json` `amplitudes.<amp>` and `budget.<amp>` (bool and number are distinct types; NaN is null);
  (b) byte equality of every `reduced/*.csv` present in the reference and of `series/alpha_t_<amp>.csv`;
  (c) byte equality of the ordered `decisions.csv` lines whose amplitude is one of `AMPS`, and the reference `all` (process) rows present
      in the new file in the same order (an ordered subsequence).  New rows (an amplitude absent from the reference, extra `all` rows)
      are listed and never fail.
Exit codes: 0 identical, 1 at least one difference (listed), 2 refused or unreadable input.  Any difference is a defect, never accepted as new.
"""
import argparse
import csv
import hashlib
import json
import os
import re
import sys

AMPS = ("s1", "s225", "s4")
_SHA_LINE = re.compile(r"^([0-9a-f]{64})  (.+)$")


class Refused(Exception):
    """The comparison refuses to run (exit 2): it could pass vacuously or the reference does not verify."""


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def json_diff(a, b, path="$"):
    """Path of the first difference between two parsed JSON values (None if equal); bool and number are different types."""
    if isinstance(a, bool) != isinstance(b, bool):
        return path
    if isinstance(a, dict) and isinstance(b, dict):
        if sorted(a) != sorted(b):
            return "%s keys %s" % (path, sorted(set(a) ^ set(b)))
        pairs = [("%s.%s" % (path, k), a[k], b[k]) for k in sorted(a)]
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return "%s length %d != %d" % (path, len(a), len(b))
        pairs = [("%s[%d]" % (path, i), x, y) for i, (x, y) in enumerate(zip(a, b))]
    elif isinstance(a, (dict, list)) or isinstance(b, (dict, list)):
        return path  # container vs something else
    else:
        return None if a == b else path
    for p, x, y in pairs:
        d = json_diff(x, y, p)
        if d:
            return d
    return None


def decision_lines(path):
    """(header line, [(amplitude, raw line)]) of a decisions.csv; a line is one CSV record (no embedded newlines)."""
    text = _read_bytes(path).decode("utf-8")
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    out = []
    for ln in lines[1:]:
        row = next(csv.reader([ln]))
        if len(row) < 2:
            raise ValueError("%s: cannot read the amplitude of line %r" % (path, ln[:80]))
        out.append((row[1], ln))
    return lines[0], out


def read_manifest(d):
    """({relative path: sha256}, sha256 of manifest.txt) of an analysis directory; `Refused` if it is absent or malformed."""
    path = os.path.join(d, "manifest.txt")
    if not os.path.isfile(path):
        raise Refused("%s has no manifest.txt: the reference cannot be verified" % d)
    raw = _read_bytes(path)
    entries = {}
    for ln in raw.decode("utf-8").split("\n"):
        if not ln.strip() or ln.startswith("#"):
            continue
        m = _SHA_LINE.match(ln)
        if not m:
            raise Refused("%s: malformed manifest line %r" % (path, ln[:80]))
        entries[m.group(2)] = m.group(1)
    return entries, hashlib.sha256(raw).hexdigest()


def verify_reference(ref):
    """Check the reference against its own manifest (see the module docstring); returns (n_reduced, manifest sha256, notes)."""
    entries, msha = read_manifest(ref)
    names = sorted(f for f in os.listdir(os.path.join(ref, "reduced")) if f.endswith(".csv")) if os.path.isdir(os.path.join(ref, "reduced")) else []
    listed = sorted(p for p in entries if p.startswith("reduced/"))
    if not names:
        raise Refused("the reference has no reduced/*.csv: nothing would be compared")
    if len(names) != len(listed):
        raise Refused("the reference holds %d reduced/*.csv but its manifest lists %d reduced/ entries" % (len(names), len(listed)))
    required = ["summary.json", "decisions.csv"] + ["reduced/" + f for f in names] + ["series/alpha_t_%s.csv" % a for a in AMPS]
    bad = []
    for p in sorted(set(required) | {p for p in entries if p.startswith(("reduced/", "series/"))}):
        if p not in entries:
            bad.append("%s is not listed in the manifest" % p)
        elif not os.path.isfile(os.path.join(ref, p)):
            bad.append("%s is listed but missing" % p)
        elif hashlib.sha256(_read_bytes(os.path.join(ref, p))).hexdigest() != entries[p]:
            bad.append("%s does not match its manifest sha256" % p)
    if bad:
        raise Refused("the reference does not verify against its manifest: %s" % "; ".join(bad[:5]) + (" ..." if len(bad) > 5 else ""))
    return len(names), msha, ["reference manifest sha256 %s: %d reduced/ entries and every compared file verified" % (msha, len(names))]


def compare(ref, new):
    """(failures, additions, notes): lists of strings; no failure = the existing outputs are reproduced.  `Refused` if it could be vacuous."""
    if os.path.realpath(ref) == os.path.realpath(new):
        raise Refused("--ref and --new are the same directory (%s): the comparison would be vacuous" % os.path.realpath(ref))
    n_red, _, notes = verify_reference(ref)
    fails, adds = [], []
    # (a) summary.json
    rs, ns = (json.loads(_read_bytes(os.path.join(d, "summary.json"))) for d in (ref, new))
    for sect in ("amplitudes", "budget"):
        for amp in AMPS:
            if amp not in rs.get(sect, {}) or amp not in ns.get(sect, {}):
                fails.append("summary.json %s.%s missing (ref %s, new %s)" % (sect, amp, amp in rs.get(sect, {}), amp in ns.get(sect, {})))
                continue
            d = json_diff(rs[sect][amp], ns[sect][amp], "%s.%s" % (sect, amp))
            if d:
                fails.append("summary.json differs at %s" % d)
    notes.append("summary.json: %d x 2 subtrees compared" % len(AMPS))
    # (b) reduced/*.csv and series/alpha_t_<amp>.csv
    names = sorted(f for f in os.listdir(os.path.join(ref, "reduced")) if f.endswith(".csv"))
    pairs = [os.path.join("reduced", f) for f in names] + [os.path.join("series", "alpha_t_%s.csv" % amp) for amp in AMPS]
    for rel in pairs:
        pr, pn = os.path.join(ref, rel), os.path.join(new, rel)
        if not os.path.isfile(pr) or not os.path.isfile(pn):
            fails.append("%s missing (ref %s, new %s)" % (rel, os.path.isfile(pr), os.path.isfile(pn)))
        elif _read_bytes(pr) != _read_bytes(pn):
            fails.append("%s differs (bytes)" % rel)
    notes.append("csv files: %d reduced + %d series compared" % (n_red, len(AMPS)))
    new_only = sorted(f for f in os.listdir(os.path.join(new, "reduced")) if f.endswith(".csv") and f not in names)
    if new_only:
        adds.append("reduced/*.csv new files: %d (%s ...)" % (len(new_only), ", ".join(new_only[:3])))
    # (c) decisions.csv
    (hr, rrows), (hn, nrows) = decision_lines(os.path.join(ref, "decisions.csv")), decision_lines(os.path.join(new, "decisions.csv"))
    if hr != hn:
        fails.append("decisions.csv header differs")
    want = [ln for a, ln in rrows if a in AMPS]
    got = [ln for a, ln in nrows if a in AMPS]
    if want != got:
        i = next((k for k, (x, y) in enumerate(zip(want, got)) if x != y), min(len(want), len(got)))
        fails.append("decisions.csv rows of %s differ (ref %d rows, new %d; first difference at row %d)" % (",".join(AMPS), len(want), len(got), i))
    notes.append("decisions.csv: %d rows of %s compared" % (len(want), ",".join(AMPS)))
    ref_all = [ln for a, ln in rrows if a == "all"]
    new_all = [ln for a, ln in nrows if a == "all"]
    k = 0
    used = set()
    for ln in ref_all:
        while k < len(new_all) and new_all[k] != ln:
            k += 1
        if k == len(new_all):
            fails.append("decisions.csv process row of the reference not found in order in the new file: %s" % ln[:120])
            break
        used.add(k)
        k += 1
    notes.append("decisions.csv: %d process (all) rows of the reference, in order" % len(ref_all))
    adds += ["decisions.csv new process row: %s" % ln[:120] for i, ln in enumerate(new_all) if i not in used]
    ref_amps = {a for a, _ in rrows} - {"all"}
    for a in sorted({a for a, _ in nrows} - ref_amps - {"all"}):
        adds.append("decisions.csv new rows of %s: %d" % (a, sum(1 for x, _ in nrows if x == a)))
    return fails, adds, notes


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m sf39.regress_existing", description=__doc__.split("\n")[0])
    ap.add_argument("--ref", required=True, help="reference analysis directory: an extraction of the committed analysis/ of evidence commit 4dbf31f2")
    ap.add_argument("--new", required=True, help="new analysis directory")
    a = ap.parse_args(argv)
    try:
        fails, adds, notes = compare(a.ref, a.new)
    except Refused as e:
        print("REGRESS_EXISTING_REFUSED: %s" % e, file=sys.stderr)
        return 2
    except (OSError, ValueError, KeyError) as e:
        print("REGRESS_EXISTING_ERROR %s: %s" % (type(e).__name__, e), file=sys.stderr)
        return 2
    for n in notes:
        print("  " + n)
    for ad in adds:
        print("  ADDED " + ad)
    for f in fails:
        print("  DIFFERENT " + f)
    print("REGRESS_EXISTING %s" % ("FAILED n=%d" % len(fails) if fails else "IDENTICAL (%s)" % ",".join(AMPS)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
