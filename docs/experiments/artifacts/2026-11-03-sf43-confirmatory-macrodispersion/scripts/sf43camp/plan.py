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


def validate(p):
    need(p, "campaign", str, "plan")
    need(p, "schema_version", int, "plan")
    p["cutoff"] = utc_epoch(need(p, "cutoff_utc", str, "plan"))
    for k in ("grace_minutes", "stage1_max_hours", "budget_gpu_h", "disk_min_free_gb"):
        if need(p, k, NUM, "plan") < 0:
            raise PlanError("plan.%s must be >= 0" % k)
    need(p, "campaign_dir", str, "plan")
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
        if need(a, "runner", str, w) not in ("consumer", "par2"):
            raise PlanError("%s: runner must be consumer or par2" % w)
        need(a, a["runner"], dict, w)
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
    br = p.get("blinded_reestimation")
    if br is not None:
        need(br, "fraction", NUM, "blinded_reestimation")
        need(br, "max_factor", NUM, "blinded_reestimation")
        ub = need(br, "sd_plan_ub", dict, "blinded_reestimation")
        for i in need(br, "arms", list, "blinded_reestimation"):
            if i not in ids or not isinstance(ub.get(i), dict) or not ub[i]:
                raise PlanError("blinded_reestimation: arm %s unknown or without sd_plan_ub" % i)


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
