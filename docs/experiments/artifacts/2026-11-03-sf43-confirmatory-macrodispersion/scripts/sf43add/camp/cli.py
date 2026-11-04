"""Command line of `python3 -m sf43add.camp` (README.md)."""
import argparse
import hashlib
import json
import os
import sys

from . import plan as P
from . import s0 as S0
from . import seeds as SD

PLAN_FILES = {"production": "plan_addendum.json", "j8": "plan_addendum_j8.json"}


def _plan_args(ap, finalize):
    ap.add_argument("--job", choices=("production", "j8"), default="production")
    ap.add_argument("--costs", required=finalize, help="costs.json of `s0-report` (est_seconds, raw_gb, min_free_mib, drops)")
    ap.add_argument("--budget-gpu-h", type=float, required=finalize, help="default 7.2 (= 8 - 0.3 - 0.5) until S0 is measured")
    ap.add_argument("--disk-min-free-gb", type=float, required=finalize, help="default 100 until finalization (host free GB - 60)")
    ap.add_argument("--drop", action="append", default=[], metavar="ARM|J1|J3|J8|J9", help="remove arms (repeatable; e.g. J1 after a failed reproduction check)")
    ap.add_argument("--allow-missing", action="store_true", help="do not fail when a file to hash (e.g. sf43add/ana) is not on disk yet; listed in _provenance")
    ap.add_argument("--out", default=None, help="output file ('-' = stdout); default <A>/plan_addendum[_j8].json")


def cmd_plan(a):
    costs = None
    if a.costs:
        with open(a.costs, "rb") as f:
            raw = f.read()
        costs = json.loads(raw)
        costs["sha256"] = hashlib.sha256(raw).hexdigest()
    plan = P.build_plan(a.job, costs, a.budget_gpu_h, a.disk_min_free_gb, a.drop, a.allow_missing)
    text = P.dumps(plan)
    if a.out == "-":
        sys.stdout.write(text)
        return 0
    out = a.out or os.path.join(P.ART, PLAN_FILES[a.job])
    with open(out, "w") as f:
        f.write(text)
    print("wrote %s (%d arms, budget_gpu_h %s, disk_min_free_gb %s, %d files hashed%s)" % (
        out, len(plan["arms"]), plan["budget_gpu_h"], plan["disk_min_free_gb"], len(plan["analysis"]["sha256_list"]),
        ", dropped " + ",".join(plan["_provenance"]["dropped"]) if plan["_provenance"]["dropped"] else ""))
    return 0


def cmd_s0(a):
    return S0.run(os.path.realpath(a.campaign), a.prereg, a.out, not a.no_du)


def cmd_seeds(a):
    reg = SD.build_registry()
    lines = SD.check(reg)
    text = SD.dumps(reg)
    for ln in lines:
        print(ln)
    if a.verify:
        with open(SD.REGISTRY) as f:
            same = f.read() == text
        print("%s %s" % ("PASS committed registry is the regenerated one:" if same else "FAIL committed registry differs from the regenerated one:", SD.REGISTRY))
        return 0 if same else 1
    with open(SD.REGISTRY, "w") as f:
        f.write(text)
    print("wrote %s" % SD.REGISTRY)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sf43add.camp")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _plan_args(sub.add_parser("plan", help="generate plan_addendum.json (+ smoke section) or plan_addendum_j8.json"), False)
    _plan_args(sub.add_parser("finalize", help="plan with --costs, --budget-gpu-h and --disk-min-free-gb all required (after S0)"), True)
    s = sub.add_parser("s0-report", help="structure and cost of the smoke campaign; writes costs.json (never reads an alpha)")
    s.add_argument("--campaign", required=True)
    s.add_argument("--prereg", default=os.path.join(P.ART, "prereg_addendum.json"))
    s.add_argument("--out", default=None, help="default <campaign>/costs.json")
    s.add_argument("--no-du", action="store_true", help="skip the raw-size measurement (a pruned campaign, e.g. the production job for J8 costs)")
    c = sub.add_parser("check-seeds", help="write seeds_registry_addendum.json and run the frozen check_seeds on it")
    c.add_argument("--verify", action="store_true", help="do not write: compare the committed file with the regenerated one")
    a = ap.parse_args(argv)
    try:
        return {"plan": cmd_plan, "finalize": cmd_plan, "s0-report": cmd_s0, "check-seeds": cmd_seeds}[a.cmd](a)
    except (P.AddendumPlanError, SD.SeedsError) as e:
        print("ERROR: %s" % e, file=sys.stderr)
        return 1
