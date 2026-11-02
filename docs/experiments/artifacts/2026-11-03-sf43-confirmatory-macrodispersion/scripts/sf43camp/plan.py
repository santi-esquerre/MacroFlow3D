"""Plan loading and validation (schema: README.md).  Pure functions, no I/O beyond reading the plan file."""
import calendar
import hashlib
import json
import re
import time

ID = re.compile(r"^[A-Za-z0-9_.-]+$")


class PlanError(ValueError):
    pass


def utc_epoch(s):
    return calendar.timegm(time.strptime(s, "%Y-%m-%dT%H:%M:%SZ"))


def utc_iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def need(d, key, kinds, where):
    if not isinstance(d, dict) or key not in d:
        raise PlanError("%s: missing '%s'" % (where, key))
    v = d[key]
    ks = kinds if isinstance(kinds, tuple) else (kinds,)
    if not isinstance(v, ks) or (isinstance(v, bool) and bool not in ks):
        raise PlanError("%s.%s: wrong type %s" % (where, key, type(v).__name__))
    return v


NUM = (int, float)
DEFAULT_EXECUTOR = "sf43camp.runner_executor:RunnerExecutor"


def validate(p):
    need(p, "campaign", str, "plan")
    need(p, "schema_version", int, "plan")
    p["cutoff"] = utc_epoch(need(p, "cutoff_utc", str, "plan"))
    for k in ("grace_minutes", "stage1_max_hours", "budget_gpu_h", "disk_min_free_gb"):
        if need(p, k, NUM, "plan") < 0:
            raise PlanError("plan.%s must be >= 0" % k)
    need(p, "campaign_dir", str, "plan")
    if not isinstance(p.setdefault("executor", DEFAULT_EXECUTOR), str) or ":" not in p["executor"]:
        raise PlanError("plan.executor must be 'module:Class'")
    if p.setdefault("min_retained", 5) != int(p["min_retained"]) or p["min_retained"] < 0:
        raise PlanError("plan.min_retained must be an integer >= 0")
    if "lane_b_min_free_mib" in p:
        need(p, "lane_b_min_free_mib", int, "plan")
    for n, m in need(p, "binaries", dict, "plan").items():
        if not re.fullmatch(r"[0-9a-f]{32}", str(m)):
            raise PlanError("plan.binaries.%s: not an md5" % n)
    an = need(p, "analysis", dict, "plan")
    need(an, "command", list, "plan.analysis")
    for f, h in need(an, "sha256_list", dict, "plan.analysis").items():
        if not re.fullmatch(r"[0-9a-f]{64}", str(h)):
            raise PlanError("plan.analysis.sha256_list.%s: not a sha256" % f)
    for k in ("recompute", "reconcile"):
        if k in p:
            need(p[k], "command", list, "plan." + k)
    fl = need(p, "failure", dict, "plan")
    need(fl, "halt_rate", NUM, "plan.failure")
    need(fl, "min_for_rate", int, "plan.failure")
    groups = need(p, "groups", dict, "plan")
    for g, d in groups.items():
        r = need(d, "reserves", dict, "plan.groups." + g)
        need(r, "start", int, "plan.groups.%s.reserves" % g)
        need(r, "count", int, "plan.groups.%s.reserves" % g)
    arms = need(p, "arms", list, "plan")
    ids = set()
    gseen = {}
    for a in arms:
        i = need(a, "id", str, "arm")
        w = "arm " + i
        if not ID.match(i) or i in ids:
            raise PlanError("%s: invalid or duplicate id" % w)
        ids.add(i)
        for k, t in (("family", str), ("amplitude", NUM + (str,)), ("priority", int), ("heavy", bool), ("min_free_mib", int),
                     ("n_conf", int), ("n_max", int), ("unit_size", int), ("est_seconds", NUM), ("raw_gb", NUM),
                     ("reduce", dict), ("retained", list)):
            need(a, k, t, w)
        for flag in ("primary", "no_reestimation"):  # optional arm flags, default false
            if not isinstance(a.setdefault(flag, False), bool):
                raise PlanError("%s.%s must be true or false" % (w, flag))
        if need(a, "runner", str, w) not in ("consumer", "par2"):
            raise PlanError("%s: runner must be consumer or par2" % w)
        need(a, a["runner"], dict, w)
        if a.get("lane") not in (None, "A", "B"):
            raise PlanError("%s.lane must be 'A' or 'B'" % w)
        hf = a[a["runner"]].get("hmax_div_first")
        if hf is not None:
            if a["runner"] != "consumer" or not isinstance(hf, dict) or not isinstance(hf.get("n"), int) or hf["n"] < 1 \
                    or not isinstance(hf.get("divs"), list) or not hf["divs"] or not all(isinstance(d, int) and d > 0 for d in hf["divs"]):
                raise PlanError("%s: consumer.hmax_div_first must be {'n': int >= 1, 'divs': [int, ...]}" % w)
            if hf["n"] % a["unit_size"]:
                raise PlanError("%s: unit_size %d must divide hmax_div_first.n %d (a unit never straddles position n)" % (w, a["unit_size"], hf["n"]))
        g = need(a, "group", str, w)
        if g not in groups:
            raise PlanError("%s: unknown group %s" % (w, g))
        sd = need(a, "seeds", dict, w)
        start, count = need(sd, "start", int, w + ".seeds"), need(sd, "count", int, w + ".seeds")
        if not (1 <= a["n_conf"] <= a["n_max"] <= count) or a["unit_size"] < 1 or a["est_seconds"] <= 0:
            raise PlanError("%s: need 1 <= n_conf <= n_max <= seeds.count, unit_size >= 1, est_seconds > 0" % w)
        gs = gseen.setdefault(g, {"start": start, "prio": a["priority"], "end": 0})
        if gs["start"] != start or gs["prio"] != a["priority"]:
            raise PlanError("%s: arms of CRN group %s must share seeds.start and priority" % (w, g))
        gs["end"] = max(gs["end"], start + count)
    for g, d in groups.items():
        r = d["reserves"]
        if g in gseen and r["start"] < gseen[g]["end"] and r["start"] + r["count"] > gseen[g]["start"]:
            raise PlanError("group %s: reserve seeds overlap the arms' seed range" % g)
    for pr in p.get("determinism_pairs", []):
        if not (isinstance(pr, list) and len(pr) == 2 and all(x in ids for x in pr) and pr[0] != pr[1]):
            raise PlanError("determinism_pairs: %r is not a pair of two distinct arm ids" % (pr,))
        ax, ay = (next(a for a in arms if a["id"] == x) for x in pr)
        if ax["runner"] != ay["runner"] or ax[ax["runner"]] != ay[ay["runner"]] or ax["seeds"] != ay["seeds"]:
            raise PlanError("determinism_pairs %s: the arms need identical runner dicts and seeds" % (pr,))
    names = set()
    for b in p.get("background", []):
        if not (isinstance(b, dict) and isinstance(b.get("name"), str) and ID.match(b["name"]) and b["name"] not in names
                and isinstance(b.get("argv"), list) and b["argv"]):
            raise PlanError("background: each entry needs a unique 'name' and a non-empty 'argv'")
        names.add(b["name"])
        if not isinstance(b.get("nice", 19), int) or not isinstance(b.get("env", {}), dict):
            raise PlanError("background %s: nice must be an integer and env an object" % b["name"])
    if not isinstance(p.setdefault("background_wait_minutes", 0), NUM) or p["background_wait_minutes"] < 0:
        raise PlanError("plan.background_wait_minutes must be >= 0")
    br = p.get("blinded_reestimation")
    if br is not None:
        need(br, "fraction", NUM, "blinded_reestimation")
        need(br, "max_factor", NUM, "blinded_reestimation")
        hw = need(br, "hw_target", dict, "blinded_reestimation")  # planning half-width targets per arm: {arm: {estimator: half-width}}
        flags = {a["id"]: a["no_reestimation"] for a in arms}
        for i in need(br, "arms", list, "blinded_reestimation"):
            if i not in ids or not isinstance(hw.get(i), dict) or not hw[i] or not all(isinstance(v, NUM) and v > 0 for v in hw[i].values()):
                raise PlanError("blinded_reestimation: arm %s unknown or without positive hw_target" % i)
            if flags[i]:
                raise PlanError("blinded_reestimation: arm %s is flagged no_reestimation (Gate 5 arms and the sigma^2 = 4 (12,16) rung are never re-sized)" % i)


def load_plan(path, smoke=False, now=0.0):
    """Plan dict (validated; adds 'cutoff' epoch and 'plan_sha256').  smoke: overlay of the plan's 'smoke' section."""
    with open(path, "rb") as f:
        raw = f.read()
    try:
        p = json.loads(raw)
        if not isinstance(p, dict):
            raise PlanError("plan: not a JSON object")
        s = p.pop("smoke", None)
        if smoke:
            if not isinstance(s, dict):
                raise PlanError("plan: no smoke section")
            rel = s.pop("cutoff_minutes", None)
            p.update(s)
            if "campaign_dir" not in s:
                p["campaign_dir"] += "_smoke"
            if rel is not None:
                p["cutoff_utc"] = utc_iso(now + 60 * rel)
        validate(p)
    except (ValueError, KeyError, TypeError) as e:  # PlanError is a ValueError
        raise PlanError("%s: %s" % (path, e))
    p["plan_sha256"] = hashlib.sha256(raw).hexdigest() + ("+smoke" if smoke else "")
    return p
