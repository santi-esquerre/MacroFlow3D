"""``python3 -m sf43cov qualify --results <dir>``: ``<dir>/qualification.json``, the document the pre-registration quotes (H5, H15).

Reads the chunk results of a coverage run (profile ``sf43``; incomplete cells are reported as not judged) and writes, with the G5.3
criterion applied by ``mcstat.coverage`` (95 % one-sided Clopper-Pearson UCL of the miss rate at most 1.25 x the tail, at least 175
expected misses at the tail; a tail that is not judged is not qualified):

``gate5``       per Fieller-form procedure (``hall``, ``boott``, ``boott_half``) and per grid size ``n``: qualified yes/no for the
                families ``normal`` / ``stress`` / ``stress-`` (``families.base``) and for those plus the J3B-bound family (skewness 2.73,
                excess kurtosis 7.59, both signs; ``families.with_j3b``: gating at sigma^2 = 4), the largest UCL / tail and the first problems; and
                ``first_qualified`` per ``n`` (the selection rule of ``sf43ana.gate5``) for both;
``sampling``    the bootstrap-t sampling limit of the production mean (grid n = 100, 150, 200, m = 8 Holm tails and their halves): per distribution and side
                (``lower`` is the one the robust sign uses) qualified yes/no, the largest UCL / tail and the failing tails;
``pipeline``    the whole robust-sign pipeline per cell (structure K, arm sizes, bias scenario, distribution): ``RL(a_j) > 0`` at the m = 8 Holm
                levels; qualified yes/no, the largest UCL / tail, the misses;
``pipeline_profiles``  the cells grouped by profile (structure K and the four arm sizes production / ladder / width / h_max): qualified iff
                every cell of the profile passed (``sf43ana.qualification.pipeline_profiles``, the function the analysis applies);
``sampling_grid``  the bootstrap-t sampling limit per grid size n (``lower`` side, every distribution);
``overall``     the headline booleans (``null`` = not judged because cells are missing).
"""
import json
import os
import sys

from sf43cov import mcpath  # noqa: F401

from mcstat import coverage as cov
from sf43ana import gate5 as g5
from sf43ana import qualification as Q


def _status(entries):
    """True if every tail passed, False if one failed (final), None if nothing was judged / some tails were not judged."""
    if not entries:
        return None
    flags = [e["pass"] for e in entries]
    if any(f is False for f in flags):
        return False
    return True if all(f is True for f in flags) else None


def _entry(entries):
    return {"qualified": _status(entries), "max_ucl_over_tail": max((e["ucl_over_tail"] for e in entries), default=None),
            "max_rate_over_tail": max((e["rate_over_tail"] for e in entries), default=None),
            "tails_failing": [e["tail"] for e in entries if e["pass"] is False], "tails": len(entries), "R": entries[0]["R"] if entries else 0}


def _gate5(summ, m, fwer):
    out = {"families": {}, "first_qualified": {}}
    for name, dists in (("base", g5.DISTS), ("with_j3b", g5.DISTS + g5.J3B_DISTS)):
        q = g5.qualification(summ, m, fwer, dists)
        out["families"][name] = {"dists": list(dists), "procedures": {p: {str(n): {k: v for k, v in e.items() if k != "problems"} | {"problems": e["problems"][:6]}
                                                                          for n, e in d.items()} for p, d in q.items()}}
        out["first_qualified"][name] = {str(n): next((p for p in g5.PROCS if q[p][n]["qualified"]), None) for n in sorted(q[g5.PROCS[0]])}
    return out


def _cells(summ, proc):
    return [c for c in summ["cells"] if c["proc"] == proc]


def build(out):
    plan = cov.load_plan(out)
    summ = cov.summarize(out, write=False)
    p0 = plan["cells"][0]["params"]
    m, fwer = int(p0["family_m"]), float(p0["fwer"])
    planned = {}
    for c in plan["cells"]:
        planned[c["proc"]] = planned.get(c["proc"], 0) + 1
    done = {}
    for c in summ["cells"]:
        done[c["proc"]] = done.get(c["proc"], 0) + 1
    doc = {"version": 1, "plan_hash": plan["plan_hash"], "profile": plan.get("profile"), "seed": plan["seed"], "family_m": m, "fwer": fwer,
           "criterion": summ["criterion"], "cells_planned": planned, "cells_complete": done, "all_cells_complete": not summ["incomplete"],
           "incomplete": summ["incomplete"], "core_seconds": summ["core_seconds"]}
    doc["gate5"] = _gate5(summ, m, fwer)
    sampling = {}
    for c in _cells(summ, "sf43_sampling"):
        sampling.setdefault("n=%d" % c["n"], {})[c["dist"]] = {side: _entry(c["rows"][side]) for side in ("lower", "upper")}
    doc["sampling"] = sampling
    pipeline = []
    for c in _cells(summ, "sf43_pipeline"):
        pp = c["params"]
        e = _entry(c["rows"]["RL"])
        pipeline.append(dict(e, cell=c["id"], K=pp["K"], N=pp["N"], arm_sizes={"ladder": pp["n_l"], "h_max": pp["n_i"], "width": pp["n_w"]},
                             scenario=c["scen"], dist=c["dist"], misses=[x["misses"] for x in c["rows"]["RL"]], tails=[x["tail"] for x in c["rows"]["RL"]]))
    doc["pipeline"] = pipeline
    doc["pipeline_profiles"] = Q.pipeline_profiles(doc)
    doc["sampling_grid"] = {str(n): q for n, q in sorted(Q.sampling_grid(doc).items())}
    n_pl_planned = planned.get("sf43_pipeline", 0)
    n_sa_planned = planned.get("sf43_sampling", 0)

    def agg(flags, planned_n):
        if planned_n == 0:
            return None
        if any(f is False for f in flags):
            return False
        return True if len(flags) == planned_n and all(f is True for f in flags) else None

    sa_flags = [s[side]["qualified"] for byd in sampling.values() for s in byd.values() for side in ("lower",)]
    doc["overall"] = {
        "gate5_first_qualified_base": doc["gate5"]["first_qualified"]["base"],
        "gate5_first_qualified_with_j3b": doc["gate5"]["first_qualified"]["with_j3b"],
        "sampling_boott_lower_qualified": agg(sa_flags, n_sa_planned),
        "pipeline_qualified": agg([p["qualified"] for p in pipeline], n_pl_planned),
        "pipeline_by_profile": {"K%d:%d/%d/%d/%d" % ((p["K"],) + tuple(p["arm_sizes"][k] for k in Q.ARM_KEYS)): p["qualified"] for p in doc["pipeline_profiles"]},
    }
    return doc


def main(out):
    doc = build(out)
    path = os.path.join(out, "qualification.json")
    cov.write_json(path, doc)
    o = doc["overall"]
    print("qualify: wrote %s (cells complete %s of %s); sampling %s, pipeline %s; Gate 5 first qualified (base) %s" %
          (path, sum(doc["cells_complete"].values()), sum(doc["cells_planned"].values()), o["sampling_boott_lower_qualified"],
           o["pipeline_qualified"], json.dumps(o["gate5_first_qualified_base"], sort_keys=True)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
