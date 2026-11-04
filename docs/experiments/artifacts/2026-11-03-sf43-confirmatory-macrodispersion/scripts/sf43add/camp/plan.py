"""Deterministic generator of the SF-43 addendum-1 campaign plans (`plan_addendum.json`, `plan_addendum_j8.json`).

Every arm is a deep copy of its ORIGIN arm of the frozen `plan.json` (runner dict AND `reduce` block) with only the `factor` of
`prereg_addendum.json` applied (J9 derives from the derived `j3_l5`, J8 from `j1_*`); `check_arm` re-derives the difference from the two
dicts and fails loudly if it is not exactly the factor.  Nothing is read from results.  Stdlib + the frozen `sf43camp.plan` loader.
"""
import copy
import glob
import hashlib
import json
import math
import os

from sf43camp.plan import utc_epoch

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(os.path.dirname(HERE))                       # <A>/scripts
ART = os.path.dirname(SCRIPTS)                                         # <A>
REPO = os.path.normpath(os.path.join(ART, "..", "..", "..", ".."))
ART_REL = os.path.relpath(ART, REPO).replace(os.sep, "/")

# arm-id prefix of each experiment label (the S0 cut rules and --drop use them)
LABELS = {"J1": "j1_", "J3": "j3_", "J8": "j8_", "J9": "j9_"}
S0_EXPECTED_GPU_H = 0.3          # the owner's estimate of S0: default budget = cap - S0 - tail = 7.2 until S0 is measured
SMOKE_BUDGET_GPU_H = 1.0
SMOKE_CUTOFF_MINUTES = 240
DEFAULT_DISK_MIN_FREE_GB = 100   # until finalization (host free GB at finalization - 60)
EST_FACTOR = 1.25                # finalization: est_seconds = 1.25 x measured seconds per realization; min_free_mib = 1.25 x measured peak
# files that must exist unless --allow-missing: the packages of both addendum nodes (the plan hashes what is on disk)
REQUIRED = ("sf43add/__init__.py", "sf43add/camp/__init__.py", "sf43add/camp/__main__.py", "sf43add/ana/__init__.py", "sf43add/ana/__main__.py")
MISSING = "<missing>"


class AddendumPlanError(ValueError):
    pass


def jload(path):
    with open(path) as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def canon(x):
    return json.dumps(x, sort_keys=True)


def expand_drop(tokens, ids):
    """Arm ids named by `tokens` (arm ids or the labels J1/J3/J8/J9, case-insensitive); an unknown token is an error."""
    out = set()
    for t in tokens:
        lab = LABELS.get(t.upper())
        hit = [i for i in ids if i == t] if lab is None else [i for i in ids if i.startswith(lab)]
        if not hit:
            raise AddendumPlanError("--drop %s: no such arm or label in this job (arms: %s)" % (t, ", ".join(ids)))
        out.update(hit)
    return out


# ---------------------------------------------------------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------------------------------------------------------
def check_arm(origin, arm, factor):
    """Raise unless `arm` differs from `origin` in the runner dict by EXACTLY `factor` (key by key, canonical JSON; a no-op factor value is
    also an error) and has an identical `reduce` block, family and amplitude."""
    r = arm["runner"]
    if origin["runner"] != r:
        raise AddendumPlanError("arm %s: runner %s != origin's %s" % (arm["id"], r, origin["runner"]))
    o, n = origin[r], arm[r]
    diff = {k: n.get(k, MISSING) for k in set(o) | set(n) if canon(o.get(k, MISSING)) != canon(n.get(k, MISSING))}
    if canon(diff) != canon(factor):
        raise AddendumPlanError("arm %s: runner-dict difference to %s is %s, the pre-registered factor is %s" % (arm["id"], origin["id"], diff, factor))
    if canon(arm["reduce"]) != canon(origin["reduce"]):
        raise AddendumPlanError("arm %s: reduce block differs from %s" % (arm["id"], origin["id"]))
    for k in ("family", "amplitude"):
        if arm[k] != origin[k]:
            raise AddendumPlanError("arm %s: %s differs from %s" % (arm["id"], k, origin["id"]))


def apply_factor(runner_dict, factor):
    runner_dict.update(copy.deepcopy(factor))


def steps_per_unit(p):
    """Par2 time steps are proportional to hz / dt_coeff, and x2 with dt_half (exactly 2 n_steps)."""
    return p.get("hz", 3.6) / p.get("dt_coeff", 0.0244140625) * (2 if p.get("dt_half", False) else 1)


def provisional(root, arm):
    """(est_seconds, raw_gb, min_free_mib) before S0, scaled from the frozen ROOT arm (s4_g5p 17.5 s / 0.065 GB / 10000 MiB; s1_dist 54 s /
    0.561 GB / 9000 MiB).  Par2: time and snapshot volume grow with the number of steps (horizon x3 / x2, dt_half x2): an upper bound, the head
    solve does not grow.  Consumer: time scales with the cells (J3 = 1/8 of s1_dist, J9 = 1/2); the raw arrays are per particle and per plane,
    independent of the grid, so raw_gb is the root's.  min_free_mib is the root's (S0 measures it)."""
    r = arm["runner"]
    if r == "par2":
        m = steps_per_unit(arm[r]) / steps_per_unit(root[r])
        est, raw = root["est_seconds"] * m, root["raw_gb"] * m
    else:
        est, raw = root["est_seconds"] * math.prod(arm[r]["grid"]) / math.prod(root[r]["grid"]), root["raw_gb"]
    return round(est, 1), round(raw, 4), root["min_free_mib"]


def build_arms(frozen, pre):
    """({id: arm dict (plan schema)}, {id: frozen root arm}) for EVERY arm of prereg_addendum.json (production and J8), provisional costs."""
    froz = {a["id"]: a for a in frozen["arms"]}
    made, root = {}, {}
    for s in pre["arms"]:
        o = made.get(s["origin"]) or froz.get(s["origin"])
        if o is None:
            raise AddendumPlanError("arm %s: origin %s is neither a frozen nor an earlier addendum arm" % (s["id"], s["origin"]))
        rt = root.get(s["origin"], o)
        if froz.get(rt["id"]) is not rt:
            raise AddendumPlanError("arm %s: root of the origin chain is not a frozen arm" % s["id"])
        if s["runner"] != o["runner"]:
            raise AddendumPlanError("arm %s: runner %s, origin %s has %s" % (s["id"], s["runner"], o["id"], o["runner"]))
        run = copy.deepcopy(o[s["runner"]])
        apply_factor(run, s["factor"])
        n = s["seeds"]["count"]
        arm = {"id": s["id"], "family": o["family"], "amplitude": o["amplitude"], "runner": s["runner"], "priority": s["priority"], "heavy": False,
               "min_free_mib": 0, "group": s["group"], "seeds": dict(s["seeds"]), "n_conf": n, "n_max": n, "unit_size": s["unit_size"],
               "est_seconds": 0.0, "raw_gb": 0.0, s["runner"]: run, "reduce": copy.deepcopy(o["reduce"]), "retained": list(s["retained"]),
               "primary": False, "no_reestimation": True}
        check_arm(o, arm, s["factor"])
        lo, hi = s["seeds"]["start"], s["seeds"]["start"] + n - 1
        if not all(lo <= x <= hi for x in arm["retained"]):
            raise AddendumPlanError("arm %s: retained seeds outside its seed range" % s["id"])
        arm["est_seconds"], arm["raw_gb"], arm["min_free_mib"] = provisional(rt, arm)
        made[s["id"]], root[s["id"]] = arm, rt
    return made, root


# ---------------------------------------------------------------------------------------------------------------------------
# costs from S0 (`s0-report`): est_seconds = 1.25 x measured, raw_gb measured, min_free_mib = max(1.25 x peak, frozen origin's)
# ---------------------------------------------------------------------------------------------------------------------------
def linear_fit(points, m):
    """Value at m of the least-squares line through [(m_i, y_i)]; with fewer than two distinct m, or a line that is not increasing from a
    non-negative intercept, the proportional bound max(y_i / m_i) * m (conservative)."""
    if len(set(p[0] for p in points)) >= 2:
        n = len(points)
        mx, my = sum(p[0] for p in points) / n, sum(p[1] for p in points) / n
        b = sum((p[0] - mx) * (p[1] - my) for p in points) / sum((p[0] - mx) ** 2 for p in points)
        a = my - b * mx
        if b > 0 and a >= 0:
            return a + b * m
    return max(y / x for x, y in points) * m


def apply_costs(arms, root, pre, costs):
    """Replace est_seconds / raw_gb / min_free_mib of `arms` (dict, in place) from `costs` (the dict of `s0-report`).  An arm without its own
    measurement uses its origin's when it is an exact copy (J8 from J1); an unmeasured Par2 arm (j1_D, j1_H2) is fitted on the measured Par2 arms."""
    ms = costs.get("arms", {})
    copy_of = {s["id"]: s["origin"] for s in pre["arms"] if not s["factor"] and s["origin"] in arms}   # exact copies (j8_X <- j1_X)

    def own(i):
        c = ms.get(i)
        return c if c and c.get("seconds_per_realization") else None

    def meas(i):
        return own(i) or own(copy_of.get(i, ""))
    steps = lambda i: steps_per_unit(arms[i]["par2"]) / steps_per_unit(root[i]["par2"])  # noqa: E731
    pts = [i for i, a in arms.items() if a["runner"] == "par2" and own(i)]
    for i, a in arms.items():
        c, rt = meas(i), root[i]
        if c:
            sec, raw, peak = c["seconds_per_realization"], c.get("raw_gb_per_realization"), c.get("peak_mib")
        elif a["runner"] == "par2" and pts:
            sec = linear_fit([(steps(j), own(j)["seconds_per_realization"]) for j in pts], steps(i))
            rp = [(steps(j), own(j)["raw_gb_per_realization"]) for j in pts if own(j).get("raw_gb_per_realization")]
            raw, peak = (linear_fit(rp, steps(i)) if rp else None), None
        else:
            continue
        a["est_seconds"] = max(0.1, round(EST_FACTOR * sec, 1))
        if raw:
            a["raw_gb"] = round(raw, 4)
        a["min_free_mib"] = max(int(math.ceil(EST_FACTOR * peak)) if peak else 0, rt["min_free_mib"])


# ---------------------------------------------------------------------------------------------------------------------------
# hashes of the analysis files
# ---------------------------------------------------------------------------------------------------------------------------
def is_test_path(rel):
    return any(p == "tests" or p.startswith("tests_") or p == "__pycache__" for p in rel.split("/"))


def analysis_hashes(frozen, allow_missing):
    """({repo-relative path: sha256}, [missing paths skipped]) = the frozen plan's own list (re-verified against its recorded hashes: a frozen
    file that changed is an error) + sf43add/**/*.py (no tests) + sf43camp/*.py (the engine) + prereg_addendum.json, hashed from the files ON DISK."""
    want = dict(frozen["analysis"]["sha256_list"])
    paths = set(want)
    paths.add("%s/prereg_addendum.json" % ART_REL)
    for pkg, pat in (("sf43add", "**/*.py"), ("sf43camp", "*.py")):
        for p in glob.glob(os.path.join(SCRIPTS, pkg, pat), recursive=True):
            if not is_test_path(os.path.relpath(p, SCRIPTS).replace(os.sep, "/")):
                paths.add(os.path.relpath(p, REPO).replace(os.sep, "/"))
    gone = sorted(p for p in paths | set("%s/scripts/%s" % (ART_REL, r) for r in REQUIRED) if not os.path.isfile(os.path.join(REPO, p)))
    if gone and not allow_missing:
        raise AddendumPlanError("files the plan must hash are missing (generate after A1-ANA is integrated, or pass --allow-missing): %s" % ", ".join(gone))
    out = {}
    for p in sorted(paths - set(gone)):
        h = sha256_file(os.path.join(REPO, p))
        if p in want and want[p] != h:
            raise AddendumPlanError("frozen file changed: %s (sha256 %s, frozen plan %s)" % (p, h, want[p]))
        out[p] = h
    return out, gone


# ---------------------------------------------------------------------------------------------------------------------------
# the plan
# ---------------------------------------------------------------------------------------------------------------------------
def host_paths(pre):
    """Frozen-campaign paths of the analysis contract, derived from the pre-registration (`~` -> the parent of the addendum campaign_dir)."""
    home = os.path.dirname(pre["campaign_dir"])
    red = pre["frozen_reduced"]["host_dir"].replace("~", home, 1)
    return {"reduced": red, "ledger": os.path.join(os.path.dirname(red), "ledger.jsonl"), "segments": os.path.join(os.path.dirname(red), "segments"),
            "manifest": pre["frozen_reduced"]["manifest"].replace("~", home, 1), "campaign": pre["campaign_dir"]}


def analysis_command(pre, job, smoke=False):
    """The A1-ANA CLI contract: `run` (production; J8 adds --j8-campaign) or `check` (the smoke structure check)."""
    pr = "%s/prereg_addendum.json" % ART_REL
    if smoke:
        return ["python3", "-W", "default", "-m", "sf43add.ana", "check", "--prereg", pr, "--campaign", "{campaign_dir}", "--out", "{campaign_dir}/analysis"]
    h = host_paths(pre)
    camp = ["--campaign", h["campaign"], "--j8-campaign", "{campaign_dir}"] if job == "j8" else ["--campaign", "{campaign_dir}"]
    return (["python3", "-W", "default", "-m", "sf43add.ana", "run", "--prereg", pr] + camp
            + ["--frozen-reduced", h["reduced"], "--frozen-manifest", h["manifest"], "--frozen-ledger", h["ledger"], "--frozen-segments", h["segments"],
               "--out", "{campaign_dir}/analysis"])


def smoke_section(arms, pre, groups_of, sha):
    """The `smoke` overlay: the S0 arms (same ids and runner dicts, seeds = the S0 units, all retained so nothing is pruned, unit = all its seeds)."""
    units = pre["smoke"]["units"]
    sm_arms, groups = [], {}
    for a in sorted((arms[i] for i in units), key=lambda x: x["priority"]):
        seeds = units[a["id"]]
        lo, n = seeds[0], len(seeds)
        if seeds != list(range(lo, lo + n)) or not (a["seeds"]["start"] <= lo and lo + n <= a["seeds"]["start"] + a["seeds"]["count"]):
            raise AddendumPlanError("smoke units of %s are not a consecutive sub-range of its seeds" % a["id"])
        s = copy.deepcopy(a)
        s.update(seeds={"start": lo, "count": n}, n_conf=n, n_max=n, unit_size=n, retained=list(seeds))
        sm_arms.append(s)
        groups[a["group"]] = copy.deepcopy(groups_of[a["group"]])
    return {"campaign_dir": pre["smoke_campaign_dir"], "cutoff_minutes": SMOKE_CUTOFF_MINUTES, "budget_gpu_h": SMOKE_BUDGET_GPU_H, "groups": groups,
            "arms": sm_arms, "analysis": {"command": analysis_command(pre, "production", smoke=True), "sha256_list": sha}}


def build_plan(job="production", costs=None, budget_gpu_h=None, disk_min_free_gb=None, drop=(), allow_missing=False):
    """The plan dict of `job` ('production' with its `smoke` section, or 'j8').  Deterministic: same inputs and files, same output.  The smoke
    section never takes the costs (S0 runs before any measurement and its record must not change when the plan is finalized)."""
    if job not in ("production", "j8"):
        raise AddendumPlanError("job must be production or j8")
    frozen, pre, bins = (jload(os.path.join(ART, n)) for n in ("plan.json", "prereg_addendum.json", "binaries.json"))
    made, root = build_arms(frozen, pre)
    prov = copy.deepcopy(made)
    if costs:
        apply_costs(made, root, pre, costs)
    specs = {s["id"]: s for s in pre["arms"]}
    want = "j8_conditional" if job == "j8" else "production"
    ids = [i for i, s in specs.items() if s["job"] == want]
    dropped = sorted(expand_drop(drop, ids) | set(i for i in (costs or {}).get("drop", []) if i in ids))
    ids = [i for i in ids if i not in dropped]
    if not ids:
        raise AddendumPlanError("every arm of the %s job is dropped" % job)
    if utc_epoch(pre["cutoff_utc"]) + 60 * pre["grace_minutes"] > utc_epoch(pre["hard_end_utc"]):
        raise AddendumPlanError("cutoff_utc + grace_minutes is after the hard end %s" % pre["hard_end_utc"])
    sha, missing = analysis_hashes(frozen, allow_missing)
    groups_of = {s["group"]: {"reserves": dict(s["reserves"])} for s in pre["arms"]}
    plan = {
        "campaign": "sf43-addendum-1" + ("-j8" if job == "j8" else ""), "schema_version": 1, "cutoff_utc": pre["cutoff_utc"],
        "grace_minutes": pre["grace_minutes"], "stage1_max_hours": 0,    # one GPU: lane A only; 0 = no stage 1 (a second granted GPU is ignored)
        "budget_gpu_h": round(pre["budget_gpu_h_cap"] - S0_EXPECTED_GPU_H - pre["plan_tail_reserve_gpu_h"], 6) if budget_gpu_h is None else budget_gpu_h,
        "disk_min_free_gb": DEFAULT_DISK_MIN_FREE_GB if disk_min_free_gb is None else disk_min_free_gb,
        "campaign_dir": pre["campaign_dir"] + ("_j8" if job == "j8" else ""), "executor": "sf43camp.runner_executor:RunnerExecutor",
        "binaries": dict(bins), "analysis": {"command": analysis_command(pre, job), "sha256_list": sha},
        # no --primary-arm (no primary arm) and no --single-pass-hmax (no hmax_div_first arm: single-pass roots carry the label -1)
        "recompute": {"command": ["python3", "-W", "error", "-m", "recompute43", "run", "--campaign", "{campaign_dir}", "--reduced",
                                  "{campaign_dir}/reduced", "--out", "{campaign_dir}.recompute.txt"]},
        "reconcile": {"command": ["python3", "-W", "error", "-m", "reconcile43", "run", "--campaign", "{campaign_dir}", "--out", "{campaign_dir}.reconcile.txt"]},
        "failure": dict(pre["failure"]), "min_retained": pre["min_retained"],
        "groups": {specs[i]["group"]: copy.deepcopy(groups_of[specs[i]["group"]]) for i in ids},
        "arms": [made[i] for i in sorted(ids, key=lambda i: (specs[i]["priority"], ids.index(i)))]}
    if job == "production":
        plan["smoke"] = smoke_section(prov, pre, groups_of, sha)
    plan["_provenance"] = {"generator": "python3 -m sf43add.camp plan", "job": job, "frozen_plan_sha256": sha256_file(os.path.join(ART, "plan.json")),
                           "prereg_addendum_sha256": sha256_file(os.path.join(ART, "prereg_addendum.json")),
                           "binaries_sha256": sha256_file(os.path.join(ART, "binaries.json")),
                           "costs_sha256": (costs or {}).get("sha256"), "dropped": dropped, "missing_allowed": missing}
    return plan


def dumps(plan):
    return json.dumps(plan, indent=1) + "\n"
