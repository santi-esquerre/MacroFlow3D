#!/usr/bin/env python3
# Provenance: verbatim copy of docs/experiments/artifacts/2026-10-25-sf41-medium-b-lagrangian-stage/scripts/lines_cmp.py at commit 3452a7d.
"""SF-37 criterion R / preflight helper (provenance: verbatim copy of
docs/experiments/artifacts/2026-10-17-sf36-lagrangian-tracker-core/scripts/lines_cmp.py, SF-36 N6; the timing masks
already cover t=, t_trace=, t_labels=, t_lu=, t_apply=, t_fact=, t_lin= and `built in <num>s`, no change needed):
compare compare selected driver-log lines after masking ONLY the timing fields.

  lines_cmp.py NEW.log REF.log TOKEN [TOKEN ...]

Compared lines: those whose first whitespace-delimited token is one of TOKEN, in order.  Masked fields:
`t=<num>[s]`, `t_trace=`, `t_labels=`, `t_lu=`, `t_apply=`, `t_fact=`, `t_lin=` and `built in <num>s`.
Every other character must be identical (TIMING and MEMORY lines are simply not selected).
Prints `IDENTICAL <n_lines>` or `DIFF <n_differences> (n_new=<a> n_ref=<b>)` followed by up to 5 differing line
pairs; exit status 0 for IDENTICAL, 1 for DIFF, 2 for unreadable input.
"""
import re
import sys

MASK = re.compile(r"\b(t|t_trace|t_labels|t_lu|t_apply|t_fact|t_lin)=[-+0-9.eE]+s?")
BUILT = re.compile(r"built in [-+0-9.eE]+s")


def mask(line):
    line = MASK.sub(lambda m: m.group(1) + "=<T>", line)
    return BUILT.sub("built in <T>s", line)


def select(path, tokens):
    out = []
    with open(path, errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            parts = line.split(None, 1)
            if parts and parts[0] in tokens:
                out.append(mask(line))
    return out


def main(argv):
    if len(argv) < 4:
        print("usage: lines_cmp.py NEW.log REF.log TOKEN [TOKEN ...]", file=sys.stderr)
        return 2
    try:
        new = select(argv[1], set(argv[3:]))
        ref = select(argv[2], set(argv[3:]))
    except OSError as e:
        print("UNREADABLE %s" % e)
        return 2
    diffs = []
    for i in range(max(len(new), len(ref))):
        a = new[i] if i < len(new) else "<missing>"
        b = ref[i] if i < len(ref) else "<missing>"
        if a != b:
            diffs.append((i + 1, a, b))
    if not diffs and new:
        print("IDENTICAL %d" % len(new))
        return 0
    if not diffs:
        print("DIFF 0 (n_new=0 n_ref=0: no selected line in either log)")
        return 1
    print("DIFF %d (n_new=%d n_ref=%d)" % (len(diffs), len(new), len(ref)))
    for i, a, b in diffs[:5]:
        print("  line %d\n    new: %s\n    ref: %s" % (i, a[:300], b[:300]))
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
