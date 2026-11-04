"""`s0-report`: structure and cost of the smoke campaign (S0) and the pre-registered cut rules.  Stdlib only, runs on the host.

BLINDED: reads ONLY
  * `<C>/ledger.jsonl`   kinds `unit_end` (arm, cls, completed, seconds {seed: s}, wall, peak_mib, segment, segment_seed0), `seed`
                         (arm, seed, verdict, class, iterations) and `reduced` (arm, segment: only THAT a segment was reduced, never
                         the table), plus `hb` (GPU-s) when there is no REPORT;
  * `<C>/REPORT.txt`     first line (status) and `gpu_h_used`;
  * `<C>/segments/<arm>/<seg>/run.log` (last lines of a failed unit), the presence of `config.yaml` (Par2 vs consumer) and the SIZE of each realization's raw output
    (consumer `out/realizations/r_XXXX`; Par2 `out/crossings/realization_XXXX_planes.csv`, `out/stats/realization_XXXX*`,
    `out/snapshots/r_XXXX`), as `du` counts it (allocated blocks).
It never opens `<C>/reduced`, `<C>/analysis`, any table or any alpha column.
"""
import collections
import json
import os
import re
import sys

from .plan import LABELS, jload

USABLE = ("converged", "fallback")
GB = 1e9      # the engine's unit (procs.disk_free_gb)


def read_ledger(path):
    out = []
    with open(path) as f:
        for ln in f:
            try:
                out.append(json.loads(ln))
            except ValueError:      # a torn last line
                pass
    return out


def read_report(path):
    """(status or None, gpu_h_used or None) from the first line and the `gpu_h_used:` line of REPORT.txt."""
    if not os.path.isfile(path):
        return None, None
    with open(path) as f:
        lines = f.read().splitlines()
    m = re.search(r": (\w+)$", lines[0]) if lines else None
    g = next((re.match(r"gpu_h_used: ([0-9.]+) of", x) for x in lines if x.startswith("gpu_h_used:")), None)
    return (m.group(1) if m else None), (float(g.group(1)) if g else None)


def du_bytes(path):
    """Allocated bytes (st_blocks * 512, like `du`) of a file or a directory tree; 0 if absent."""
    if os.path.isfile(path):
        return os.lstat(path).st_blocks * 512
    n = 0
    for dp, _dn, fn in os.walk(path):
        n += os.lstat(dp).st_blocks * 512 + sum(os.lstat(os.path.join(dp, x)).st_blocks * 512 for x in fn)
    return n


def realization_paths(seg, r):
    """Raw-output paths of realization index r in a segment directory (consumer or Par2, told apart by config.yaml)."""
    if not os.path.isfile(os.path.join(seg, "config.yaml")):
        return [os.path.join(seg, "out", "realizations", "r_%04d" % r)]
    stats = os.path.join(seg, "out", "stats")
    st = [os.path.join(stats, x) for x in sorted(os.listdir(stats)) if x.startswith("realization_%04d" % r)] if os.path.isdir(stats) else []
    return [os.path.join(seg, "out", "crossings", "realization_%04d_planes.csv" % r), os.path.join(seg, "out", "snapshots", "r_%04d" % r)] + st


def log_hint(seg):
    try:
        with open(os.path.join(seg, "run.log"), errors="replace") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 3000))
            tail = [x.strip() for x in f.read().splitlines() if x.strip()]
    except OSError:
        return "(no run.log)"
    return " | ".join(tail[-2:])


def collect(campaign, planned, du=True):
    """{arm: stats} from the ledger and the segments; `planned` {arm: [seeds]} are the S0 units of the pre-registration (inputs, not results)."""
    led = read_ledger(os.path.join(campaign, "ledger.jsonl"))
    reduced = set(r["segment"] for r in led if r["kind"] == "reduced")
    arms = collections.OrderedDict()

    def st(a):
        return arms.setdefault(a, {"completed": set(), "classes": collections.Counter(), "secs": [], "wall": 0.0, "wall_n": 0, "peak": None, "iters": None,
                                   "last_seed": {}, "failures": [], "bytes": 0, "n_raw": 0})
    for r in led:
        k = r["kind"]
        if k == "unit_end":
            a = st(r["arm"])
            a["classes"][r["cls"]] += 1
            a["completed"].update(r["completed"])
            a["secs"] += list(r["seconds"].values())
            if r.get("wall") is not None and r["completed"]:
                a["wall"], a["wall_n"] = a["wall"] + r["wall"], a["wall_n"] + len(r["completed"])
            if r.get("peak_mib") is not None:
                a["peak"] = max(a["peak"] or 0.0, r["peak_mib"])
            seg = os.path.join(campaign, r["segment"])
            if r["cls"] != "ok":
                a["failures"].append("unit %s cls=%s failed_seed=%s: %s" % (r["segment"], r["cls"], r.get("failed_seed"), log_hint(seg)))
            if r["completed"] and r["segment"] not in reduced:       # a reduction that failed blocks the analysis as surely as a runner error
                a["failures"].append("segment %s completed but was not reduced" % r["segment"])
            if du:
                for s in r["completed"]:
                    ps = [p for p in realization_paths(seg, s - r["segment_seed0"]) if os.path.exists(p)]
                    if ps:
                        a["bytes"], a["n_raw"] = a["bytes"] + sum(du_bytes(p) for p in ps), a["n_raw"] + 1
        elif k == "seed":
            a = st(r["arm"])
            a["last_seed"][r["seed"]] = r
            if r.get("iterations") is not None:
                a["iters"] = max(a["iters"] or 0, r["iterations"])
    for i, s in planned.items():
        a = st(i)
        a["planned"] = len(s)
    for a in arms.values():
        bad = sorted(sd for sd, r in a["last_seed"].items() if r["class"] != "ok" or r["verdict"] not in USABLE)
        a["failures"] += ["seed %d: verdict %s class %s" % (sd, a["last_seed"][sd]["verdict"], a["last_seed"][sd]["class"]) for sd in bad]
        if "planned" in a and len(a["completed"]) < a["planned"]:
            a["failures"].append("%d of %d planned realizations completed" % (len(a["completed"]), a["planned"]))
    return arms


def summarize(a):
    return {"planned": a.get("planned"), "n": len(a["completed"]), "seconds_per_realization": round(sum(a["secs"]) / len(a["secs"]), 3) if a["secs"] else None,
            "seconds_max": round(max(a["secs"]), 3) if a["secs"] else None,
            "wall_per_realization": round(a["wall"] / a["wall_n"], 3) if a["wall_n"] else None, "peak_mib": a["peak"], "max_iterations": a["iters"],
            "raw_gb_per_realization": round(a["bytes"] / a["n_raw"] / GB, 5) if a["n_raw"] else None,
            "verdicts": dict(collections.Counter(r["verdict"] for r in a["last_seed"].values())), "unit_classes": dict(a["classes"]),
            "structural_failures": a["failures"]}


def decide(summ, pre):
    """({arm: [reasons]} to drop, [notes]) by the S0 cut rules of the pre-registration (section 4)."""
    thr = pre["smoke"]["j9_drop_if"]
    prod = [s["id"] for s in pre["arms"] if s["job"] == "production"]
    drop, notes = {}, []
    j9 = summ.get("j9_w48")
    if j9 is None or not j9["n"]:
        notes.append("j9_w48: NOT MEASURED (no completed realization)%s" % (": " + "; ".join(j9["structural_failures"]) if j9 else ""))
        if j9 and j9["unit_classes"].get("config_error"):
            drop["j9_w48"] = ["the runner rejects the geometry (config_error)"]
    else:
        why = []
        if j9["seconds_per_realization"] > thr["seconds_per_realization_gt"]:
            why.append("%.1f s per realization > %g" % (j9["seconds_per_realization"], thr["seconds_per_realization_gt"]))
        if j9["peak_mib"] is not None and j9["peak_mib"] > thr["peak_mib_gt"]:
            why.append("peak %.0f MiB > %g" % (j9["peak_mib"], thr["peak_mib_gt"]))
        if j9["max_iterations"] is not None and j9["max_iterations"] > thr["cg_iterations_gt"]:
            why.append("%d CG iterations > %g" % (j9["max_iterations"], thr["cg_iterations_gt"]))
        if why:
            drop["j9_w48"] = why
        elif j9["structural_failures"]:
            notes.append("j9_w48: structural failure that is not an S0 cut rule (orchestrator decides): " + "; ".join(j9["structural_failures"]))
    for lab, rule in (("J1", "J1 stopped: a J1 smoke unit failed structurally"), ("J3", "J3 dropped: its smoke unit failed structurally")):
        ids = [i for i in prod if i.startswith(LABELS[lab])]
        ran = [i for i in ids if i in pre["smoke"]["units"]]
        bad = [i for i in ran if i not in summ or summ[i]["structural_failures"] or not summ[i]["n"]]
        if bad:
            for i in ids:
                drop[i] = [rule + " (%s)" % ", ".join(bad)]
    return drop, notes


def run(campaign, prereg_path, out=None, du=True, w=None):
    w = w or sys.stdout
    pre = jload(prereg_path)
    if not os.path.isfile(os.path.join(campaign, "ledger.jsonl")):
        w.write("s0-report: no ledger.jsonl in %s\n" % campaign)
        return 1
    status, held = read_report(os.path.join(campaign, "REPORT.txt"))
    arms = collect(campaign, pre["smoke"]["units"], du)
    summ = {i: summarize(a) for i, a in arms.items()}
    if held is None:
        held = max((r["gpu_s"] for r in read_ledger(os.path.join(campaign, "ledger.jsonl")) if r["kind"] == "hb"), default=None)
        held = None if held is None else round(held / 3600.0, 3)
    drop, notes = decide(summ, pre)
    sugg = None if held is None else round(pre["budget_gpu_h_cap"] - held - pre["plan_tail_reserve_gpu_h"], 2)
    f = lambda v, spec: "-" if v is None else spec % v  # noqa: E731
    w.write("S0 REPORT campaign=%s status=%s gpu_h_held=%s\n" % (campaign, status, f(held, "%.3f")))
    w.write("%-11s %7s %4s %9s %9s %9s %8s %9s  %s\n" % ("arm", "planned", "done", "s/real", "wall/real", "peak_mib", "max_it", "GB/real", "verdicts"))
    for i, s in sorted(summ.items()):
        w.write("%-11s %7s %4d %9s %9s %9s %8s %9s  %s\n" % (i, f(s["planned"], "%d"), s["n"], f(s["seconds_per_realization"], "%.1f"),
                                                            f(s["wall_per_realization"], "%.1f"), f(s["peak_mib"], "%.0f"), f(s["max_iterations"], "%d"),
                                                            f(s["raw_gb_per_realization"], "%.4f"), json.dumps(s["verdicts"], sort_keys=True)))
        for x in s["structural_failures"]:
            w.write("    STRUCTURAL %s\n" % x)
    w.write("CUT RULES (pre-registration section 4)\n")
    for i in sorted(drop):
        w.write("  DROP %-11s %s\n" % (i, "; ".join(drop[i])))
    for n in notes:
        w.write("  NOTE %s\n" % n)
    if not drop and not notes:
        w.write("  none triggered\n")
    w.write("suggested budget_gpu_h = %s (cap %g - held - tail reserve %g)\n" % (f(sugg, "%.2f"), pre["budget_gpu_h_cap"], pre["plan_tail_reserve_gpu_h"]))
    w.write("the reproduction pre-check of J1 (sf43add.ana repro) is NOT part of this report; if it fails run `plan --drop J1`\n")
    costs = {"schema": 1, "campaign": os.path.basename(os.path.normpath(campaign)), "status": status, "gpu_h_held": held, "suggested_budget_gpu_h": sugg,
             "arms": {i: summ[i] for i in sorted(summ)}, "drop": sorted(drop), "decisions": {i: drop[i] for i in sorted(drop)}, "notes": notes}
    path = out or os.path.join(campaign, "costs.json")
    with open(path, "w") as fo:
        json.dump(costs, fo, indent=1)
        fo.write("\n")
    w.write("wrote %s\n" % path)
    return 0
