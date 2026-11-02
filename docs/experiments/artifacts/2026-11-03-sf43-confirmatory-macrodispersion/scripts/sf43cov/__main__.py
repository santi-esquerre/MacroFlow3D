"""CLI of the SF-43 coverage harness: ``python3 -m sf43cov {plan,bench,run,summarize} --out DIR [options]`` (see the package docstring).

The plan (``plan.json`` in ``--out``, written once; a different plan in the same directory is refused) holds one cell per (distribution,
``n``): ``normal``, ``stress``, ``stress-`` at ``n`` in ``--ns`` (judged, G5.3), and the REPORTED (non-gating) block at the heavier stress
(skewness 2.73, excess kurtosis 7.59, the J3B upper bounds) and its mirror for ``--report-ns``.  Replicates per cell
``R = ceil(target / smallest tail)`` (target 175 expected misses at the smallest tail, G5.3): with the 8-hypothesis family the smallest tail
is the half of the smallest Holm level, 0.05 / 16 = 3.125e-3, so ``R = 56 000`` (an R of 49 000 gives 153 expected misses there, below the
175 of G5.3, and the engine would not judge that tail; pass ``--R 49000`` to run it anyway, ``--target 350`` for twice the replicates).
Streams: ``default_rng(SeedSequence([seed, sha256(cell id)[:8], batch]))`` (mcstat.coverage), independent of the number of workers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

from sf43cov import mcpath  # noqa: F401
from sf43cov import procs
from sf43cov.runner import run

from mcstat import coverage as cov
from sf43ana import gate5 as g5

DEFAULT_NS = (100, 150, 200, 250, 300, 400, 500)
REPORT_NS = (150, 200, 250, 300, 400)
JUDGED_DISTS = ("normal", "stress", "stress-")
REPORT_DISTS = ("mix:2.73:7.59", "mix:2.73:7.59-")  # skewness 2.73, excess kurtosis 7.59 (J3B upper bounds) and the mirror
DEFAULT_SEED = 20261103
DEFAULT_WORKERS = 32
VERSION = 1


def est_spr(n, B):
    """Workstation seconds per replicate (measured, one core, numpy 2.5, m = 50): 1.15e-3 + 1.85e-5 n at B = 33599, the bootstrap
    part linear in B.  Used only to size chunks (about ``seconds`` per chunk) and as the default for ``bench`` extrapolation checks."""
    return 1.15e-3 + 1.85e-5 * n * (B / 33599.0) if B > 0 else 2.0e-5 * n / 100.0 + 1e-5


def build_cells(args):
    params = {"B": args.B, "family_m": args.family_m, "fwer": args.fwer, "ref": args.ref}
    tails_min = float(procs.fieller_tails(params).min())
    cells = []
    for dists, ns, report in ((args.dists, args.ns, False), (args.report_dists if args.report_ns else (), args.report_ns, True)):
        for d in dists:
            for n in ns:
                spr = est_spr(n, args.B)
                chunk = max(args.batch, int(args.seconds / spr) // args.batch * args.batch)
                c = cov.make_cell("fieller", "fieller", "report" if report else "", d, n, params, target=args.target, m=args.batch,
                                  chunk=chunk, share=args.B > 0, report_only=report)
                if args.R:  # an explicit R replaces the target-derived one
                    c["R"] = int(math.ceil(args.R / args.batch) * args.batch)
                    c["chunk"] = min(c["chunk"], c["R"])
                cells.append(c)
    return cells, tails_min


def _plan(args):
    cells, _ = build_cells(args)
    return cov.init_plan(args.out, cells, args.seed, "sf43-gate5-fieller")


def bench(cells, batches=1, seed=1, log=None):
    """Measured seconds per replicate for each distinct (n, report) group of the plan cells (one core, ``batches`` batches of ``m``)."""
    log = log or (lambda s: print(s, file=sys.stderr, flush=True))
    out = {}
    for c in cells:
        key = (c["n"], c["dist"])
        if key in out:
            continue
        t = time.time()
        cov.run_cell_batches(c, seed, 0, batches)
        out[key] = (time.time() - t) / (batches * c["m"])
        log("bench n=%d %-16s %.5f s/replicate" % (c["n"], c["dist"], out[key]))
    return out


def project(cells, spr, host_factors=(4.0, 9.0), workers=DEFAULT_WORKERS, efficiency=0.9):
    """Core-hours of the plan on this machine and on the host (``host_factors`` times slower per core), wall hours on ``workers``."""
    core_s = sum(c["R"] * spr[(c["n"], c["dist"])] for c in cells)
    judged = sum(c["R"] * spr[(c["n"], c["dist"])] for c in cells if not c["report_only"])
    res = {"replicates": sum(c["R"] for c in cells), "cells": len(cells), "core_hours_here": core_s / 3600.0,
           "core_hours_here_judged_only": judged / 3600.0}
    for f in host_factors:
        res["core_hours_host_x%g" % f] = core_s / 3600.0 * f
        res["wall_hours_host_x%g_workers%d" % (f, workers)] = core_s / 3600.0 * f / (workers * efficiency)
    return res


def summarize(out):
    plan = cov.load_plan(out)
    summ = cov.summarize(out, write=False)
    p0 = plan["cells"][0]["params"]
    m, fwer = int(p0["family_m"]), float(p0["fwer"])
    q = g5.qualification(summ, m, fwer)
    ns = sorted(q["hall"])
    sel = {}
    for n in ns:
        first = next((p for p in g5.PROCS if q[p][n]["qualified"]), None)
        sel[str(n)] = first
    ff = {}
    for R in sorted({c["R"] for c in plan["cells"]}):
        r = cov.nominal_fail_probability(R, [float(t) for t in procs.fieller_tails(p0)])
        ff[str(R)] = {"p_any_one_row": r["p_any"], "p_any_6_rows_independent_upper": 1.0 - (1.0 - r["p_any"]) ** 6}
    summ["sf43cov"] = {"version": VERSION, "family_m": m, "fwer": fwer, "B": p0.get("B"), "ref": p0.get("ref"),
                       "tails_main": list(g5.holm_levels(m, fwer)), "qualification": {p: {str(n): v for n, v in d.items()} for p, d in q.items()},
                       "first_qualified_by_n": sel, "nominal_false_fail": ff}
    cov.write_json(os.path.join(out, "summary.json"), summ)
    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write(render(summ))
    return summ


def render(s):
    c = s["sf43cov"]
    L = ["# SF-43 Gate 5 coverage (sf43cov)", "",
         "Criterion (G5.3): one-sided miss rate with 95%% one-sided Clopper-Pearson UCL at most %.2f x tail, >= %d expected misses at the "
         "smallest tail. Cells complete: %d of %d; core-seconds %.0f; seed %s; plan hash `%s`; B = %s; family m = %d; reference %s." %
         (cov.CRITERION, cov.TARGET, s["cells_complete"], s["cells_total"], s["core_seconds"], s["seed"], s["plan_hash"][:16], c["B"],
          c["family_m"], c["ref"]), "",
         "Tails used: hall and boott at the Holm levels %s; boott_half at their halves." % ", ".join("%.4g" % t for t in c["tails_main"]), "",
         "## Qualification (normal, stress, stress-; both sides; every tail used)", "",
         "| procedure | " + " | ".join("n=%s" % n for n in sorted(c["qualification"]["hall"], key=int)) + " |",
         "|---|" + "---|" * len(c["qualification"]["hall"])]
    for p in g5.PROCS:
        row = []
        for n in sorted(c["qualification"][p], key=int):
            e = c["qualification"][p][n]
            row.append("%s (%.2f)" % ("QUALIFIED" if e["qualified"] else "no", e["max_ucl_over_tail"]))
        L.append("| %s | %s |" % (p, " | ".join(row)))
    L += ["", "Entry: verdict (largest UCL / tail over the tails used). First qualified procedure by n (the selection rule): " +
          ", ".join("%s: %s" % (n, v) for n, v in sorted(c["first_qualified_by_n"].items(), key=lambda kv: int(kv[0]))), "",
          "## False-fail probability of the criterion for an exactly nominal procedure", ""]
    for R, v in c["nominal_false_fail"].items():
        L.append("- R = %s: %.3f per (cell, distribution, row); at most %.2f over the six rows of a procedure at one n if independent." %
                 (R, v["p_any_one_row"], v["p_any_6_rows_independent_upper"]))
    L += ["", "## Failing rows (procedures hall and boott, judged cells)", ""]
    bad = [v for v in s["verdicts"] if v["row"].split("/")[0] in ("hall", "boott") and v["tails_fail"] and not v["report_only"]]
    for v in bad[:60]:
        L.append("- %s n=%d: %d failing (tail, dist): %s" % (v["row"], v["n"], len(v["tails_fail"]),
                                                              "; ".join("%.4g %s %.2f" % (f["tail"], f["dist"], f["ucl_over_tail"]) for f in v["tails_fail"][:4])))
    if not bad:
        L.append("(none)")
    L += ["", "## Reported, non-gating: heavier stress (skewness 2.73, excess kurtosis 7.59) and its mirror", ""]
    rep = [x for x in s["cells"] if x["report_only"]]
    for x in sorted(rep, key=lambda x: (x["dist"], x["n"])):
        for row in ("hall/hi", "hall/lo", "boott/hi", "boott/lo"):
            ents = x["rows"].get(row, [])
            if ents:
                L.append("- n=%d %s %s: max rate/tail %.2f, max UCL/tail %.2f" % (x["n"], x["dist"], row, max(e["rate_over_tail"] for e in ents),
                                                                                  max(e["ucl_over_tail"] for e in ents)))
    return "\n".join(L) + "\n"


def parser():
    ap = argparse.ArgumentParser(prog="sf43cov", description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("plan", "bench", "run", "summarize"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--ns", type=lambda s: tuple(int(v) for v in s.split(",")), default=DEFAULT_NS)
    ap.add_argument("--dists", type=lambda s: tuple(s.split(",")), default=JUDGED_DISTS)
    ap.add_argument("--report-ns", type=lambda s: tuple(int(v) for v in s.split(",") if v), default=REPORT_NS)
    ap.add_argument("--report-dists", type=lambda s: tuple(s.split(",")), default=REPORT_DISTS)
    ap.add_argument("--family-m", type=int, default=8)
    ap.add_argument("--fwer", type=float, default=0.05)
    ap.add_argument("--B", type=int, default=g5.DEFAULT_B)
    ap.add_argument("--ref", default="t", choices=("t", "normal"))
    ap.add_argument("--batch", type=int, default=50, help="replicates per batch (they share one bootstrap resampling matrix)")
    ap.add_argument("--target", type=float, default=cov.TARGET, help="expected misses at the smallest tail (R = target / tail)")
    ap.add_argument("--R", type=int, default=0, help="replicates per cell, overriding --target (e.g. 49000)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--seconds", type=float, default=30.0, help="chunk size in workstation seconds (about)")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--batches", type=int, default=1, help="bench: batches timed per group")
    ap.add_argument("--max-chunks", type=int, default=None)
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    if args.cmd == "plan":
        plan = _plan(args)
        print(json.dumps({"plan_hash": plan["plan_hash"], "cells": len(plan["cells"]), "replicates": sum(c["R"] for c in plan["cells"]),
                          "chunks": sum(cov.nchunks(c) for c in plan["cells"]), "tails": plan["cells"][0]["tails"]}))
    elif args.cmd == "bench":
        cells, _ = build_cells(args)
        spr = bench(cells, args.batches)
        print(json.dumps(project(cells, spr, workers=args.workers), indent=1, sort_keys=True))
    elif args.cmd == "run":
        _plan(args)
        run(args.out, args.workers, args.max_chunks)
    else:
        s = summarize(args.out)
        print("summarize: %d of %d cells complete; first qualified by n: %s" %
              (s["cells_complete"], s["cells_total"], json.dumps(s["sf43cov"]["first_qualified_by_n"], sort_keys=True)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
