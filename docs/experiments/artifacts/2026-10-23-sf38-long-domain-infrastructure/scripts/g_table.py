#!/usr/bin/env python3
"""SF-38 threshold G (RR11): SF-18 raw variance and applied scale per seed and geometry (recorded, not interpreted).

  g_table.py --geom NAME=DIR [--geom NAME=DIR ...] [--ab NAME=DIR ...] [--seed0 4001 --nr 10]
  g_table.py --selftest

Every `field.json` below a DIR (the runner writes realizations/r_%04d/field.json; the A/B application writes its own) is read.
NAME=DIR pairs of --geom are the field-only runs of the three production geometries; --ab pairs are the A/B output
directories (their grids' values).  Output (stdout; also appended to results.txt by the campaign script):
  G_FIELD set=geom|ab name= grid= ell= seed= raw_mean= raw_variance= applied_scale= final_variance=
  G_SUMMARY name= n= raw_variance_mean= raw_variance_sd= applied_scale_mean= applied_scale_sd= (sample sd, n-1)
  G_COMPLETE name= expected=<seed0>-<seed0+nr-1> found=<n> missing=<list> OK|INCOMPLETE
Exit 0 iff every --geom set is complete.  No interpretation, no threshold.  Stdlib only.
"""
import json
import math
import os
import sys


def find_fields(root):
    out = []
    for dp, _, fn in os.walk(root):
        if "field.json" in fn:
            p = os.path.join(dp, "field.json")
            try:
                with open(p) as f:
                    out.append((p, json.load(f)))
            except (OSError, ValueError):
                out.append((p, None))
    return sorted(out, key=lambda t: (t[1] or {}).get("seed", 0))


def fmt(x):
    return "-" if x is None else ("%.12e" % x if isinstance(x, (int, float)) else str(x))


def msd(v):
    m = sum(v) / len(v)
    s = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) if len(v) > 1 else float("nan")
    return m, s


def table(sets, kind, seed0, nr):
    lines, complete = [], True
    for name, root in sets:
        recs = [(p, j) for p, j in find_fields(root)]
        seeds, rv, sc = [], [], []
        for p, j in recs:
            if not j:
                lines.append("G_FIELD set=%s name=%s unreadable=%s" % (kind, name, p))
                continue
            g = j.get("grid")
            lines.append("G_FIELD set=%s name=%s grid=%s ell=%s seed=%s raw_mean=%s raw_variance=%s applied_scale=%s final_variance=%s"
                         % (kind, name, "x".join(str(x) for x in g) if g else "-", fmt(j.get("ell")), j.get("seed"),
                            fmt(j.get("raw_mean")), fmt(j.get("raw_variance")), fmt(j.get("applied_scale")),
                            fmt(j.get("final_variance"))))
            seeds.append(j.get("seed"))
            if isinstance(j.get("raw_variance"), (int, float)):
                rv.append(j["raw_variance"])
            if isinstance(j.get("applied_scale"), (int, float)):
                sc.append(j["applied_scale"])
        if rv and sc:
            m1, s1 = msd(rv)
            m2, s2 = msd(sc)
            lines.append("G_SUMMARY name=%s n=%d raw_variance_mean=%s raw_variance_sd=%s applied_scale_mean=%s applied_scale_sd=%s"
                         % (name, len(rv), fmt(m1), fmt(s1), fmt(m2), fmt(s2)))
        want = list(range(seed0, seed0 + nr))
        miss = [s for s in want if s not in seeds]
        ok = not miss and len(seeds) == len(want)
        if kind == "geom":
            complete = complete and ok
        lines.append("G_COMPLETE name=%s expected=%d-%d found=%d missing=%s %s"
                     % (name, seed0, seed0 + nr - 1, len(seeds), ",".join(str(s) for s in miss) or "-", "OK" if ok else "INCOMPLETE"))
    return lines, complete


def selftest():
    import tempfile
    fails = []

    def expect(name, cond):
        if not cond:
            fails.append(name)
            print("SELFTEST FAIL %s" % name)

    with tempfile.TemporaryDirectory() as d:
        for r in range(10):
            os.makedirs("%s/g/realizations/r_%04d" % (d, r))
            with open("%s/g/realizations/r_%04d/field.json" % (d, r), "w") as f:
                json.dump({"grid": [64, 16, 16], "ell": 0.25, "seed": 4001 + r, "raw_mean": 0.0, "raw_variance": 1.0 + 0.01 * r,
                           "applied_scale": 1.0 / math.sqrt(1.0 + 0.01 * r), "final_variance": 1.0}, f)
        lines, ok = table([("B1", d + "/g")], "geom", 4001, 10)
        expect("complete", ok and lines[-1].endswith("OK") and sum(1 for l in lines if l.startswith("G_FIELD")) == 10)
        expect("summary", any(l.startswith("G_SUMMARY name=B1 n=10 raw_variance_mean=1.045000000000e+00") for l in lines))
        os.remove("%s/g/realizations/r_0003/field.json" % d)
        lines, ok = table([("B1", d + "/g")], "geom", 4001, 10)
        expect("missing", not ok and "missing=4004" in lines[-1] and lines[-1].endswith("INCOMPLETE"))
        lines, ok = table([("ab", d + "/none")], "ab", 4001, 10)
        expect("ab_missing_not_fatal", ok and "INCOMPLETE" in lines[-1])
    if fails:
        print("SELFTEST FAILED: %s" % ", ".join(fails))
        return 1
    print("SELFTEST OK g_table")
    return 0


def main(argv):
    if len(argv) == 2 and argv[1] == "--selftest":
        return selftest()
    geoms, abs_, seed0, nr = [], [], 4001, 10
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in ("--geom", "--ab") and i + 1 < len(argv) and "=" in argv[i + 1]:
            n, p = argv[i + 1].split("=", 1)
            (geoms if a == "--geom" else abs_).append((n, p))
            i += 2
        elif a == "--seed0" and i + 1 < len(argv):
            seed0 = int(argv[i + 1])
            i += 2
        elif a == "--nr" and i + 1 < len(argv):
            nr = int(argv[i + 1])
            i += 2
        else:
            print(__doc__, file=sys.stderr)
            return 2
    if not geoms and not abs_:
        print(__doc__, file=sys.stderr)
        return 2
    l1, ok = table(geoms, "geom", seed0, nr)
    l2, _ = table(abs_, "ab", seed0, nr)
    print("\n".join(l1 + l2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
