#!/usr/bin/env python3
"""SF-37 criterion R helper (run_sf37.sh part bytecmp): structural equality of two JSON files after removing keys.

  json_strip_cmp.py A.json B.json [--drop SPEC ...]

SPEC is a dotted key path from the document root whose LAST component may carry shell-style wildcards (fnmatch), e.g.
`transport.seconds_*`, or `**.name`, which removes every key called `name` at any depth.  Only dictionary keys are
removed (lists are compared element-wise as they are).  Floats are compared EXACTLY (== on the parsed values): the
instrument for the bitwise criterion R, not a tolerance test.

Prints `IDENTICAL` or `DIFF <n> <first differing paths>`; exit status 0 for IDENTICAL, 1 for DIFF, 2 for unreadable input.
"""
import fnmatch
import json
import sys


def strip(obj, specs, path=()):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            here = path + (k,)
            drop = False
            for s in specs:
                comps = s.split(".")
                if comps[0] == "**":
                    if len(comps) == 2 and fnmatch.fnmatchcase(k, comps[1]):
                        drop = True
                elif len(comps) == len(here) and all(fnmatch.fnmatchcase(a, b) for a, b in zip(here, comps)):
                    drop = True
            if not drop:
                out[k] = strip(v, specs, here)
        return out
    if isinstance(obj, list):
        return [strip(v, specs, path + ("[]",)) for v in obj]
    return obj


def diffs(a, b, path="$"):
    if type(a) is not type(b):
        return [path]
    if isinstance(a, dict):
        d = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                d.append("%s.%s" % (path, k))
            else:
                d.extend(diffs(a[k], b[k], "%s.%s" % (path, k)))
        return d
    if isinstance(a, list):
        if len(a) != len(b):
            return [path + "[len]"]
        d = []
        for i, (x, y) in enumerate(zip(a, b)):
            d.extend(diffs(x, y, "%s[%d]" % (path, i)))
        return d
    return [] if a == b else [path]


def main(argv):
    if len(argv) < 3:
        print("usage: json_strip_cmp.py A.json B.json [--drop SPEC ...]", file=sys.stderr)
        return 2
    specs = []
    i = 3
    while i < len(argv):
        if argv[i] == "--drop" and i + 1 < len(argv):
            specs.append(argv[i + 1])
            i += 2
        else:
            print("json_strip_cmp.py: bad argument %r" % argv[i], file=sys.stderr)
            return 2
    try:
        with open(argv[1]) as f:
            a = json.load(f)
        with open(argv[2]) as f:
            b = json.load(f)
    except (OSError, ValueError) as e:
        print("UNREADABLE %s" % e)
        return 2
    d = diffs(strip(a, specs), strip(b, specs))
    if not d:
        print("IDENTICAL")
        return 0
    print("DIFF %d %s" % (len(d), " ".join(d[:5])))
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
