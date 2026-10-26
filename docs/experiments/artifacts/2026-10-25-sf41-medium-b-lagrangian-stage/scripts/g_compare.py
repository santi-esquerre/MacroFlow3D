#!/usr/bin/env python3
"""SF-41 threshold G, regression (pre-registration RR9): byte comparison of output trees and the part verdict.

  g_compare.py tree --name N --ref DIR --cand DIR [--exclude BASENAME ...] [--allow-extra GLOB ...]
  g_compare.py verdict RESULTS [--ctest-times FILE]
  g_compare.py --selftest

tree     : every file under --ref must exist under --cand and be byte-identical (filecmp, shallow=False); files whose basename is in
           --exclude (RR9: exactly `timings.json`) are ignored on both sides; files only in --cand are `extra`: allowed iff their
           relative path matches an --allow-extra glob (fnmatch; RR9 (iii): `realizations/r_*/exits_b.csv`), otherwise unexpected
           (a failure).  `DONE` files are compared like any other.  Output
             G_BYTES name= files_ref= files_cand= identical= diff= missing= extra_allowed= extra_unexpected= excluded=<ref>/<cand> IDENTICAL|DIFF
             G_BYTES_DIFF name= kind=diff|missing|extra path=         (at most 20)
             G_BYTES_EXTRA name= path=                                (every allowed extra)
           exit 0 iff IDENTICAL (and the reference is not empty).
verdict  : G (RR9) from the results.txt of run_sf41.sh part `final` (the shell wrote the lines below): met iff
             G_CTEST ... PASS            full ctest green
             G_TRIO IDENTICAL            SF-27 trio byte-identical to the base references
             G_RSET IDENTICAL            the 20-run R set identical to the base references
             G_BYTES name=runner_medium_a ... IDENTICAL    (ii) the medium (a) lagrangian_ensemble tree, DONE included
             G_BYTES name=medium_ab ... IDENTICAL          (iii) the A/B tree; exits_b.csv allowed
           and, with --ctest-times FILE (name<TAB>status<TAB>seconds), `interpolation_periodic_tricubic` and `interpolation_hessian`
           are listed as Passed.  Output `G_VERDICT PASS|FAIL [missing/failing items]`.  Exit 0 iff PASS.
The `git diff 2cf48c1..H1` emptiness of the SF-27/SF-28 periodic spline files (RR9 (iv)) is the orchestrator's check, not made here.
"""
import fnmatch
import filecmp
import os
import re
import sys


def list_files(root, excludes):
    out = set()
    for dp, _, fn in os.walk(root):
        for f in fn:
            if f in excludes:
                continue
            out.add(os.path.relpath(os.path.join(dp, f), root))
    return out


def count_excluded(root, excludes):
    return sum(1 for _, _, fn in os.walk(root) for f in fn if f in excludes)


def tree_cmp(name, ref, cand, excludes=("timings.json",), allow=()):
    if not os.path.isdir(ref) or not os.path.isdir(cand):
        return ["G_BYTES name=%s files_ref=- files_cand=- identical=0 diff=0 missing=1 extra_allowed=0 extra_unexpected=0 excluded=-/- DIFF "
                "(missing directory %s)" % (name, ref if not os.path.isdir(ref) else cand)], False
    fr, fc = list_files(ref, excludes), list_files(cand, excludes)
    ident, diffs = 0, []
    for rel in sorted(fr & fc):
        if filecmp.cmp(os.path.join(ref, rel), os.path.join(cand, rel), shallow=False):
            ident += 1
        else:
            diffs.append(rel)
    missing = sorted(fr - fc)
    extra = sorted(fc - fr)
    ok_extra = [e for e in extra if any(fnmatch.fnmatchcase(e, g) for g in allow)]
    bad_extra = [e for e in extra if e not in ok_extra]
    ok = not diffs and not missing and not bad_extra and len(fr) > 0
    lines = ["G_BYTES name=%s files_ref=%d files_cand=%d identical=%d diff=%d missing=%d extra_allowed=%d extra_unexpected=%d "
             "excluded=%d/%d %s" % (name, len(fr), len(fc), ident, len(diffs), len(missing), len(ok_extra), len(bad_extra),
                                    count_excluded(ref, excludes), count_excluded(cand, excludes), "IDENTICAL" if ok else "DIFF")]
    shown = 0
    for kind, lst in (("diff", diffs), ("missing", missing), ("extra", bad_extra)):
        for rel in lst:
            if shown < 20:
                lines.append("G_BYTES_DIFF name=%s kind=%s path=%s" % (name, kind, rel))
                shown += 1
    for rel in ok_extra:
        lines.append("G_BYTES_EXTRA name=%s path=%s" % (name, rel))
    return lines, ok


def read_ctest_times(path):
    out = {}
    try:
        with open(path) as f:
            for ln in f:
                p = ln.rstrip("\n").split("\t")
                if len(p) >= 2:
                    out[p[0]] = p[1].strip()
    except OSError:
        return None
    return out


def verdict(results_text, ctest_times=None):
    lines = results_text.splitlines()
    missing = []

    def has(pred, what):
        if not any(pred(x) for x in lines):
            missing.append(what)

    has(lambda x: x.startswith("G_CTEST ") and re.search(r"\bPASS\b", x) and "FAIL" not in x, "G_CTEST_PASS")
    has(lambda x: x.startswith("G_TRIO IDENTICAL"), "G_TRIO_IDENTICAL")
    has(lambda x: x.startswith("G_RSET IDENTICAL"), "G_RSET_IDENTICAL")
    has(lambda x: x.startswith("G_BYTES name=runner_medium_a ") and x.endswith(" IDENTICAL"), "G_BYTES_runner_medium_a")
    has(lambda x: x.startswith("G_BYTES name=medium_ab ") and x.endswith(" IDENTICAL"), "G_BYTES_medium_ab")
    if ctest_times is not None:
        for t in ("interpolation_periodic_tricubic", "interpolation_hessian"):
            if ctest_times.get(t) != "Passed":
                missing.append("ctest_%s=%s" % (t, ctest_times.get(t)))
    else:
        missing.append("ctest_times_unreadable")
    return not missing, missing


def selftest():
    import tempfile
    fails = []

    def expect(n, c):
        if not c:
            fails.append(n)
            print("SELFTEST FAIL %s" % n)

    def mk(root, files):
        for rel, body in files.items():
            p = os.path.join(root, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "wb") as f:
                f.write(body)

    base = {"a/r_0000/x.csv": b"1,2\n", "a/r_0000/DONE": b"sig\n", "a/r_0000/timings.json": b"{t:1}", "top.csv": b"z\n"}
    with tempfile.TemporaryDirectory() as d:
        r, c = os.path.join(d, "ref"), os.path.join(d, "cand")
        mk(r, base)
        mk(c, dict(base, **{"a/r_0000/timings.json": b"{t:2}"}))
        lines, ok = tree_cmp("t", r, c)
        expect("identical_modulo_timings", ok and lines[0].endswith("IDENTICAL") and "excluded=1/1" in lines[0])
        mk(c, {"a/r_0000/DONE": b"sig2\n"})
        lines, ok = tree_cmp("t", r, c)
        expect("done_is_compared", not ok and any("kind=diff path=a/r_0000/DONE" in x for x in lines))
        mk(c, {"a/r_0000/DONE": b"sig\n", "a/r_0000/exits_b.csv": b"x\n"})
        lines, ok = tree_cmp("t", r, c)
        expect("extra_unexpected", not ok and "extra_unexpected=1" in lines[0])
        lines, ok = tree_cmp("t", r, c, allow=("a/r_*/exits_b.csv",))
        expect("extra_allowed", ok and "extra_allowed=1" in lines[0] and any(x.startswith("G_BYTES_EXTRA") for x in lines))
        os.remove(os.path.join(c, "top.csv"))
        lines, ok = tree_cmp("t", r, c, allow=("a/r_*/exits_b.csv",))
        expect("missing", not ok and "missing=1" in lines[0])
        lines, ok = tree_cmp("t", os.path.join(d, "nope"), c)
        expect("missing_dir", not ok)
        e1, e2 = os.path.join(d, "e1"), os.path.join(d, "e2")
        os.makedirs(e1)
        os.makedirs(e2)
        expect("empty_reference_is_not_identical", not tree_cmp("t", e1, e2)[1])
    good = ("G_CTEST exit=0 tests=40 failed=0 wall_s=100 PASS\nG_TRIO IDENTICAL x\nG_RSET IDENTICAL y\n"
            "G_BYTES name=runner_medium_a files_ref=3 files_cand=3 identical=3 diff=0 missing=0 extra_allowed=0 extra_unexpected=0 excluded=1/1 IDENTICAL\n"
            "G_BYTES name=medium_ab files_ref=3 files_cand=4 identical=3 diff=0 missing=0 extra_allowed=1 extra_unexpected=0 excluded=1/1 IDENTICAL\n")
    ct = {"interpolation_periodic_tricubic": "Passed", "interpolation_hessian": "Passed"}
    expect("verdict_pass", verdict(good, ct)[0])
    expect("verdict_ctest_fail", not verdict(good.replace("failed=0 wall_s=100 PASS", "failed=2 wall_s=100 FAIL"), ct)[0])
    expect("verdict_trio_diff", not verdict(good.replace("G_TRIO IDENTICAL", "G_TRIO DIFF"), ct)[0])
    expect("verdict_runner_diff", not verdict(good.replace("IDENTICAL\nG_BYTES name=medium_ab", "DIFF\nG_BYTES name=medium_ab"), ct)[0])
    expect("verdict_hessian_not_passed", not verdict(good, dict(ct, interpolation_hessian="Failed"))[0])
    expect("verdict_no_ctest_times", not verdict(good, None)[0])
    expect("verdict_empty", not verdict("", ct)[0])
    if fails:
        print("SELFTEST FAILED: %s" % ", ".join(fails))
        return 1
    print("SELFTEST OK g_compare")
    return 0


def main(a):
    if a == ["--selftest"]:
        return selftest()
    if a and a[0] == "tree":
        o = {"--name": None, "--ref": None, "--cand": None}
        ex, al = [], []
        i = 1
        while i < len(a):
            if a[i] in o and i + 1 < len(a):
                o[a[i]] = a[i + 1]
                i += 2
            elif a[i] == "--exclude" and i + 1 < len(a):
                ex.append(a[i + 1])
                i += 2
            elif a[i] == "--allow-extra" and i + 1 < len(a):
                al.append(a[i + 1])
                i += 2
            else:
                break
        if i < len(a) or None in o.values():
            print(__doc__, file=sys.stderr)
            return 2
        lines, ok = tree_cmp(o["--name"], o["--ref"], o["--cand"], tuple(ex) or ("timings.json",), tuple(al))
        print("\n".join(lines))
        return 0 if ok else 1
    if a and a[0] == "verdict" and len(a) >= 2:
        ct = None
        if len(a) == 4 and a[2] == "--ctest-times":
            ct = read_ctest_times(a[3])
        elif len(a) != 2:
            print(__doc__, file=sys.stderr)
            return 2
        try:
            with open(a[1]) as f:
                text = f.read()
        except OSError as e:
            print("G_VERDICT FAIL [unreadable %s]" % e)
            return 1
        ok, miss = verdict(text, ct)
        print("G_VERDICT %s%s" % ("PASS" if ok else "FAIL", "" if ok else " [%s]" % ",".join(miss)))
        return 0 if ok else 1
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
