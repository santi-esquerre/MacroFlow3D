#!/usr/bin/env python3
"""SF-35 regression G: compare the --production 32^3 eps 0.25 log with the SF-33 reference log.

  regress_diff.py NEW.log REF.log [--out analysis/regress.md]

Compared lines: those whose first token is CASE, ORACLE, ORACLE_SUMMARY or EXTRA (the spec's set), in order, after
masking ONLY the timing fields (orchestrator reading rule 5): `t=<num>[s]`, `t_trace=<num>s`, `t_labels=<num>s`, and
`t_lu=`, `t_apply=`, `t_fact=`, `t_lin=` if present on those lines.  Every other character must be identical.
Informational (not part of verdict G): the same comparison for ORACLE_CONFIG, ORACLE_RUN and ORACLE_LABELDIFF lines
(ORACLE_RUN prints the thread count, so it may legitimately differ when THREADS differs from the reference run).
Exit status 0 always when both logs are readable (the verdict is in the output).
"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import md_table, rel, write_out  # noqa: E402

VERDICT_TOKENS = ("CASE", "ORACLE", "ORACLE_SUMMARY", "EXTRA")
INFO_TOKENS = ("ORACLE_CONFIG", "ORACLE_RUN", "ORACLE_LABELDIFF")
MASK = re.compile(r"\b(t|t_trace|t_labels|t_lu|t_apply|t_fact|t_lin)=[-+0-9.eE]+s?")


def mask(line):
    return MASK.sub(lambda m: m.group(1) + "=<T>", line)


def select(path, tokens):
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            tok = line.split(" ", 1)[0]
            if tok in tokens:
                out.append(mask(line))
    return out


def compare(new, ref):
    diffs = []
    for i in range(max(len(new), len(ref))):
        a = new[i] if i < len(new) else "<missing>"
        b = ref[i] if i < len(ref) else "<missing>"
        if a != b:
            diffs.append((i + 1, a, b))
    return diffs


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("new_log")
    ap.add_argument("ref_log")
    ap.add_argument("--out", help="markdown output path (also printed)")
    a = ap.parse_args(argv)
    try:
        new = select(a.new_log, VERDICT_TOKENS)
        ref = select(a.ref_log, VERDICT_TOKENS)
        inew = select(a.new_log, INFO_TOKENS)
        iref = select(a.ref_log, INFO_TOKENS)
    except OSError as ex:
        print("ERROR: %s" % ex, file=sys.stderr)
        return 2
    d = compare(new, ref)
    di = compare(inew, iref)
    L = ["# SF-35 regression G: `--production` 32^3 eps 0.25 vs SF-33 `ladder_0.25_c4/N32.log`", ""]
    L.append("- new log: `%s`" % rel(a.new_log))
    L.append("- reference log: `%s`" % rel(a.ref_log))
    L.append("- compared lines (first token in %s): new %d, reference %d" % (", ".join(VERDICT_TOKENS), len(new), len(ref)))
    L.append("- masked fields (timing only): `t=`, `t_trace=`, `t_labels=`, `t_lu=`, `t_apply=`, `t_fact=`, `t_lin=`")
    L.append("")
    if not new or not ref:
        L.append("**Verdict G (production lines): not_evaluable** (no comparable lines in one of the logs)")
    elif not d:
        L.append("**Verdict G (production lines): identical** (all %d lines equal after masking the timing fields)" % len(new))
    else:
        L.append("**Verdict G (production lines): DIFFERENT** (%d differing lines)" % len(d))
        L.append("")
        L.append("Full list of differing lines (index among compared lines):")
        L.append("")
        for i, x, y in d:
            L.append("- line %d" % i)
            L.append("  - new: `%s`" % x)
            L.append("  - ref: `%s`" % y)
    L.append("")
    L.append("Informational (not part of G): %s lines: new %d, reference %d, %s" % (
        "/".join(INFO_TOKENS), len(inew), len(iref), "identical" if not di else "%d differ" % len(di)))
    for i, x, y in di:
        L.append("- info line %d" % i)
        L.append("  - new: `%s`" % x)
        L.append("  - ref: `%s`" % y)
    L.append("")
    write_out("\n".join(L), a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
