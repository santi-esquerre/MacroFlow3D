"""Host-side reference numbers for the pilots (read-only on ``~/sf39_campaign``; writes one JSON given by --out).

    python3 -m pilots.hostref --sf39 ~/sf39_campaign --r3 ~/sf42_campaign/r3 --out <json>

1. ``pcg``: PCG iterations of SF-39's medium (b) solves per amplitude (``runs/<amp>_A|_B|...`` ``realizations/r_*/solver_b.json``,
   DONE realizations): production arms (A and B) and all arms; count, min, median, 90th percentile, max.  Same solver tolerance as the
   pilots (``--pcg-rtol`` default 1e-10).
2. ``planning_sd``: across-realization SD (ddof = 1) of ``alpha_TP``, ``alpha_TAP`` (positions and displacements) and ``alpha_LP`` of the
   SF-39 production sets on d in [25, 94] lambda, from the R3 reduced tables ``<r3>/noise/<amp>.csv`` (consumer, flux weights, set
   perplane, hmax_div 8 or -1): the planning denominators of the particle share.
"""
import argparse
import csv
import glob
import json
import os
import statistics
import sys

AMPS = ("s025", "s1", "s225", "s4")
EST = ("alpha_TP", "alpha_TAP", "alpha_LP")


def pcg_stats(sf39):
    out = {}
    for a in AMPS:
        prod, allv = [], []
        for d in sorted(glob.glob(os.path.join(sf39, "runs", a + "_*"))):
            arm = os.path.basename(d)[len(a) + 1:]
            for sj in sorted(glob.glob(os.path.join(d, "realizations", "r_*", "solver_b.json"))):
                if not os.path.exists(os.path.join(os.path.dirname(sj), "DONE")):
                    continue
                with open(sj) as f:
                    it = json.load(f)["iterations"]
                allv.append(it)
                if arm in ("A", "B"):
                    prod.append(it)
        rec = {}
        for name, v in (("production", prod), ("all_arms", allv)):
            if v:
                v = sorted(v)
                rec[name] = {"n": len(v), "min": v[0], "median": statistics.median(v), "p90": v[min(len(v) - 1, int(0.9 * len(v)))],
                             "max": v[-1]}
        out[a] = rec
    return out


def planning_sd(r3):
    out = {}
    for a in AMPS:
        p = os.path.join(r3, "noise", a + ".csv")
        with open(p) as f:
            rows = [r for r in csv.DictReader(f) if r["code"] == "consumer" and r["family"] == "plane" and r["weights"] == "flux"
                    and r["set"] == "perplane" and r["hmax_div"] in ("8", "-1")]
        rec = {}
        for e in EST:
            for b in ("pos", "disp"):
                v = [float(r[e]) for r in rows if r["basis"] == b]
                v = [x for x in v if x == x and abs(x) != float("inf")]
                rec["%s_%s" % (e, b)] = {"n": len(v), "mean": statistics.fmean(v), "sd": statistics.stdev(v)}
        out[a] = rec
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sf39", required=True)
    ap.add_argument("--r3", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = {"pcg": pcg_stats(os.path.expanduser(a.sf39)), "planning_sd": planning_sd(os.path.expanduser(a.r3))}
    with open(os.path.expanduser(a.out), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res["pcg"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
