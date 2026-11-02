#!/usr/bin/env python3
"""SF-42 R1 byte comparison of two output trees (stdlib only).

  treecmp.py A B

File sets must be equal; every file is compared byte for byte (`cmp` semantics), except `timings.json` (listed, not compared) and
`manifest.json`, compared structurally and exactly after dropping the keys named `timestamp` and `build.git_hash`
(scripts/regress/json_strip_cmp.py semantics); the raw byte equality of the manifest is printed as well. Prints one line per
exception and `TREECMP IDENTICAL files=<n>` or `TREECMP DIFF n=<k>`; exit 0 iff identical.
"""
import filecmp
import json
import os
import sys


def files(root):
    out = set()
    for d, _, fs in os.walk(root):
        for f in fs:
            out.add(os.path.relpath(os.path.join(d, f), root))
    return out


def strip(o, path=()):
    if isinstance(o, dict):
        return {k: strip(v, path + (k,)) for k, v in o.items() if k != "timestamp" and path + (k,) != ("build", "git_hash")}
    if isinstance(o, list):
        return [strip(v, path) for v in o]
    return o


def main(a, b):
    fa, fb = files(a), files(b)
    bad = 0
    for f in sorted(fa ^ fb):
        print("DIFF fileset: %s only in %s" % (f, "A" if f in fa else "B"))
        bad += 1
    n = 0
    for f in sorted(fa & fb):
        pa, pb = os.path.join(a, f), os.path.join(b, f)
        base = os.path.basename(f)
        if base == "timings.json":
            print("SKIP timings.json: %s" % f)
        elif base == "manifest.json":
            same = strip(json.load(open(pa))) == strip(json.load(open(pb)))
            raw = filecmp.cmp(pa, pb, shallow=False)
            print("MANIFEST %s: stripped(timestamp, build.git_hash) %s; raw bytes %s" % (f, "IDENTICAL" if same else "DIFF", "equal" if raw else "differ"))
            bad += 0 if same else 1
            n += 1
        else:
            n += 1
            if not filecmp.cmp(pa, pb, shallow=False):
                print("DIFF bytes: %s" % f)
                bad += 1
    print("TREECMP IDENTICAL files=%d" % n if bad == 0 else "TREECMP DIFF n=%d (files compared %d)" % (bad, n))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
