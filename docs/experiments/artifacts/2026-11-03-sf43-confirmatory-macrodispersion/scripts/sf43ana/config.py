"""Pre-registration loader and validator (schema documented in README.md).

Every number the analysis uses that is a pre-registered decision (levels, fractions, null values, orders, bootstrap sizes and
seeds, ladders, widths, floors) is read from the JSON; nothing is defaulted silently except the values listed in DEFAULTS, which
are the pre-activation defaults (H2-H6) and are echoed into ``summary.json`` (``process.prereg_effective``).
"""
import copy
import json

COMPONENTS = ("E_h", "E_W", "E_WxH", "E_int")
DEFAULTS = {
    "windows": {"main": [25.0, 94.0], "h1": [25.0, 59.5], "h2": [59.5, 94.0], "far": [94.0, 194.0]},
    "primary_window": "main",
    "selection": {"code": "consumer", "family": "plane", "weights": "flux", "set": "perplane"},
    "bases": {"primary": "disp", "secondary": "pos"},
    "ledger": "ledger.jsonl",
    # executor naming (sf43camp runner_executor.reduce): reduced/<arm>/<segdir>/<arm>__<segdir>__d<lo>-<hi>.csv; {segment} is the segment
    # directory name (basename of the ledger's segment path), {stem} = loader.window_stem(lo, hi) = "d%g-%g" of the window bounds (one rule,
    # whatever the window's semantic label), {lo}, {hi} the bounds, {window} the label
    "reduced_pattern": "{arm}/{segment}/{arm}__{segment}__{stem}.csv",
    # solver verdicts that make a realization usable (consumer: solver_b verdict "converged" or the 1e-8 "fallback"; Par2: "converged")
    "ok_verdicts": ["converged", "fallback"],
    "fallback_verdicts": ["fallback"],
    "levels": {"fwer": 0.05, "interval": 0.90, "ucl": 0.05},
    "tie_rtol": 1e-9,  # relative tolerance of the decisions at the Holm levels (ties are not rejections)
    # H13 floors (seeds): Gate 4 production / each ladder level (complete paired seeds) / each width cell / h_max passes; Gate 5 pairs
    "floors": {"production": 100, "ladder": 30, "width": 30, "hmax": 20, "gate5": 200},
    "null_per_sigma2": 7.992e-5,
    "numerical_fraction": 0.5,
    "eh": {"p_range": [0.5, 2.1], "p_cap": 2.0, "safety": 1.25, "fallback_factor": 3.0, "gate_level": 0.90},
    # B_sampling + 1 = 33600 is a multiple of 1/a for every Holm level a = 0.05/j (j <= 8) and every half (exact bootstrap-t ranks)
    "bootstrap": {"B_sampling": 33599, "B_budget": 4000, "B_ucl": 4000, "B_bca": 5000, "seed": 20261103},
    "sampling_method": "bootstrap-t",
}


BLOCKS = ("gate5", "g7", "secondaries", "failures", "positive_control", "pollock", "gate4_power", "basis")
# ``qualification``: path of the sf43cov qualification.json (the robust sign's qualification gate, ``sf43ana.qualification``; absent = no document)
TOP_KEYS = set(DEFAULTS) | {"amplitudes", "arms", "family", "targets", "first_order", "report_levels", "name", "variant", "note", "qualification"} | set(BLOCKS)
# n_conf = planned size; n_max = seeds the engine may run (default len(seeds)); reest: the analysis cut follows the engine's blinded
# re-estimation of the SOURCE arm (``reest`` ledger record: n_new, never below n_conf)
ARM_KEYS = {"id", "amplitude", "group", "seeds", "reserves", "n_conf", "n_max", "floor", "source", "hmax_div", "code", "role", "family", "selection",
            "reest"}
CODES = ("consumer", "par2", "pollock")
# a control amplitude (the positive control) is referenced by arms only: no Gate 4, no family
AMP_KEYS = {"sigma2", "production", "components", "ladder", "width", "hmax", "distance_arm", "primary", "null_value", "floors", "control"}
LADDER_KEYS = {"arms", "resolutions", "production_index", "triples", "gates", "fallback_levels"}
WIDTH_KEYS = {"wide_fine", "prod_fine", "wide_coarse", "prod_coarse"}
GATE_KEYS = {"type", "triple", "triples", "arm", "resolution"}
SELECTION_KEYS = {"code", "family", "weights", "set"}
NESTED = ("levels", "eh", "bootstrap", "bases", "floors", "selection")


class ConfigError(ValueError):
    pass


def _merge(default, given):
    out = copy.deepcopy(default)
    for k, v in (given or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


class Prereg:
    """Validated pre-registration.  ``raw`` is the merged dictionary (defaults filled); ``arms`` maps id -> arm dict;
    ``amps`` is the list of amplitude keys in increasing sigma^2."""

    def __init__(self, raw):
        self.raw = raw
        self.arms = {a["id"]: a for a in raw["arms"]}
        self.amps = sorted((k for k, v in raw["amplitudes"].items() if not v.get("control")),
                           key=lambda k: float(raw["amplitudes"][k].get("sigma2", k)))

    def amp(self, key):
        return self.raw["amplitudes"][key]

    def sigma2(self, key):
        return float(self.amp(key).get("sigma2", key))

    def null_value(self, key):
        a = self.amp(key)
        return float(a["null_value"]) if "null_value" in a else float(self.raw["null_per_sigma2"]) * self.sigma2(key)

    def family_ids(self):
        """The FIXED Holm family: explicit list, else robust:<amp> for every amplitude not marked primary=false, plus gate5:<amp>."""
        if self.raw.get("family"):
            return list(self.raw["family"])
        return (["robust:%s" % k for k in self.amps if self.amp(k).get("primary", True)] + ["gate5:%s" % k for k in self.amps])

    def __getitem__(self, k):
        return self.raw[k]


def _need(cond, msg):
    if not cond:
        raise ConfigError(msg)


def _strict(d, allowed, where):
    extra = sorted(set(d) - set(allowed))
    _need(not extra, "%s: unknown key(s) %s (allowed: %s)" % (where, extra, sorted(allowed)))


def _validate_blocks(raw, arms):
    """Strict keys and arm references of the optional analysis blocks."""
    from . import basis, failures, g7, gate4_power, gate5, pollock, positive_control, secondaries  # lazy: scipy-heavy

    def arm_ref(a, where):
        _need(a in arms, "%s: unknown arm %r" % (where, a))

    if raw.get("gate5"):
        g = raw["gate5"]
        _strict(g, gate5.GATE5_KEYS, "gate5")
        _need(g.get("amplitudes"), "gate5 needs 'amplitudes'")
        for amp, sp in g["amplitudes"].items():
            _need(amp in raw["amplitudes"], "gate5: unknown amplitude %r" % amp)
            _strict(sp, gate5.GATE5_AMP_KEYS, "gate5.amplitudes.%s" % amp)
            for k in ("consumer", "par2"):
                _need(k in sp, "gate5.amplitudes.%s lacks '%s'" % (amp, k))
                arm_ref(sp[k], "gate5.amplitudes.%s.%s" % (amp, k))
            for blk in ("unmatched", "pilot"):
                for k, a in sp.get(blk, {}).items():
                    _need(k in ("consumer", "par2"), "gate5.amplitudes.%s.%s: bad key %r" % (amp, blk, k))
                    if blk == "unmatched":
                        arm_ref(a, "gate5.amplitudes.%s.unmatched" % amp)
    if raw.get("g7"):
        _strict(raw["g7"], g7.KEYS, "g7")
        for amp, sp in raw["g7"].get("twins", {}).items():
            for k, a in sp.items():
                _need(k in ("consumer", "pollock", "par2"), "g7.twins.%s: bad key %r" % (amp, k))
                arm_ref(a, "g7.twins.%s.%s" % (amp, k))
        for amp, cells in raw["g7"].get("factorial", {}).items():
            for c in cells:
                _strict(c, {"g", "i", "d", "r", "arm"}, "g7.factorial.%s" % amp)
                arm_ref(c["arm"], "g7.factorial.%s" % amp)
    if raw.get("secondaries"):
        sc = raw["secondaries"]
        _strict(sc, secondaries.KEYS, "secondaries")
        for a in sc.get("time_window", {}):
            arm_ref(a, "secondaries.time_window")
        for k, v in list(sc.get("expI", {}).items()) + list(sc.get("recrossings", {}).items()):
            arm_ref(v["arm"], "secondaries.%s" % k)
        for k, v in sc.get("par2_transverse", {}).items():
            arm_ref(v["par2"], "secondaries.par2_transverse.%s" % k)
            if v.get("consumer"):
                arm_ref(v["consumer"], "secondaries.par2_transverse.%s" % k)
        for it in sc.get("bdd_fit", []):
            arm_ref(it["arm"], "secondaries.bdd_fit")
    if "failures" in raw:
        _strict(raw["failures"], failures.KEYS, "failures")
    if raw.get("pollock"):
        _strict(raw["pollock"], pollock.KEYS, "pollock")
        for blk, keys in (("contrast", ("consumer", "pollock")), ("decomposition", ("consumer", "pollock", "par2"))):
            for amp, sp in raw["pollock"].get(blk, {}).items():
                _need(amp in raw["amplitudes"], "pollock.%s: unknown amplitude %r" % (blk, amp))
                _strict(sp, keys, "pollock.%s.%s" % (blk, amp))
                for k in keys:
                    _need(k in sp, "pollock.%s.%s lacks '%s'" % (blk, amp, k))
                    arm_ref(sp[k], "pollock.%s.%s.%s" % (blk, amp, k))
    if raw.get("gate4_power"):
        gp = raw["gate4_power"]
        _strict(gp, gate4_power.KEYS, "gate4_power")
        for amp, alt in gp.get("alternatives", {}).items():
            _need(amp in raw["amplitudes"] and not raw["amplitudes"][amp].get("control"), "gate4_power.alternatives: unknown amplitude %r" % amp)
            _strict(alt, {"alpha_TP", "alpha_TAP"}, "gate4_power.alternatives.%s" % amp)
            _need(set(alt) == {"alpha_TP", "alpha_TAP"}, "gate4_power.alternatives.%s needs alpha_TP and alpha_TAP" % amp)
        _need(set(gp.get("scenarios", gate4_power.SCENARIOS)) <= set(gate4_power.SCENARIOS), "gate4_power.scenarios must be among %s" % (gate4_power.SCENARIOS,))
    if raw.get("basis"):
        _strict(raw["basis"], basis.KEYS, "basis")
        for amp, sp in raw["basis"].get("amplitudes", {}).items():
            _need(amp in raw["amplitudes"], "basis.amplitudes: unknown amplitude %r" % amp)
            _strict(sp, basis.AMP_KEYS, "basis.amplitudes.%s" % amp)
            for k in sorted(basis.AMP_KEYS):
                _need(k in sp, "basis.amplitudes.%s lacks '%s'" % (amp, k))
                arm_ref(sp[k], "basis.amplitudes.%s.%s" % (amp, k))
    if raw.get("positive_control"):
        pc = raw["positive_control"]
        _strict(pc, positive_control.KEYS, "positive_control")
        for k in ("arm", "sigma2", "width_lambda", "length_lambda"):
            _need(k in pc, "positive_control lacks '%s'" % k)
        arm_ref(pc["arm"], "positive_control.arm")


def validate(raw):
    _strict(raw, TOP_KEYS, "pre-registration")
    for key in NESTED:
        _strict(raw[key], DEFAULTS[key], key)
    for key in ("amplitudes", "arms"):
        _need(key in raw, "pre-registration lacks '%s'" % key)
    for a in raw["arms"]:
        _strict(a, ARM_KEYS, "arm %s" % a.get("id"))
        _strict(a.get("selection", {}), SELECTION_KEYS, "arm %s selection" % a.get("id"))
    for amp, c in raw["amplitudes"].items():
        _strict(c, AMP_KEYS, "amplitude %s" % amp)
        if "ladder" in c:
            _strict(c["ladder"], LADDER_KEYS, "amplitude %s ladder" % amp)
            for g in c["ladder"].get("gates", []):
                _strict(g, GATE_KEYS, "amplitude %s ladder gate" % amp)
        if "width" in c:
            _strict(c["width"], WIDTH_KEYS, "amplitude %s width" % amp)
        if "hmax" in c:
            _strict(c["hmax"], {"h16", "h8"}, "amplitude %s hmax" % amp)
        _strict(c.get("floors", {}), DEFAULTS["floors"], "amplitude %s floors" % amp)
    ids = [a.get("id") for a in raw["arms"]]
    _need(all(isinstance(i, str) and i for i in ids) and len(set(ids)) == len(ids), "arm ids must be unique non-empty strings")
    arms = {a["id"]: a for a in raw["arms"]}
    for a in raw["arms"]:
        for k in ("amplitude", "group", "seeds", "n_conf", "floor"):
            _need(k in a, "arm %s lacks '%s'" % (a["id"], k))
        a.setdefault("source", a["id"])
        a.setdefault("code", "consumer")
        a.setdefault("hmax_div", -1)
        a.setdefault("selection", {})
        a.setdefault("reserves", [])
        a.setdefault("reest", False)
        a.setdefault("role", a.get("family", ""))
        _need(a["code"] in CODES, "arm %s: code must be one of %s" % (a["id"], CODES))
        a.setdefault("n_max", len(a["seeds"]))
        _need(a["amplitude"] in raw["amplitudes"], "arm %s: unknown amplitude %r" % (a["id"], a["amplitude"]))
        s = list(a["seeds"]) + list(a["reserves"])
        _need(len(set(s)) == len(s), "arm %s: seeds/reserves are not unique" % a["id"])
        _need(2 <= a["floor"] <= a["n_conf"] <= a["n_max"] <= len(a["seeds"]), "arm %s: need 2 <= floor <= n_conf <= n_max <= len(seeds)" % a["id"])
    W = raw["windows"]
    _need(raw["primary_window"] in W, "primary_window not among windows")
    _need(raw["sampling_method"] in ("bootstrap-t", "student"), "sampling_method must be 'bootstrap-t' or 'student'")
    _need(raw.get("qualification") is None or isinstance(raw["qualification"], str), "qualification must be a path string (or absent)")
    _need(0.0 < raw["eh"].get("gate_level", 0.9) < 1.0 and raw["tie_rtol"] >= 0.0, "bad eh.gate_level / tie_rtol")
    _need(set(raw["floors"]) >= {"production", "ladder", "width", "hmax", "gate5"}, "floors must define production, ladder, width, hmax, gate5")
    lo, hi = raw["eh"]["p_range"]
    _need(0 < lo <= hi and raw["eh"]["p_cap"] > 1.0 and raw["eh"]["safety"] >= 1.0, "bad eh parameters")

    def ref(arm, amp, what):
        _need(arm in arms, "amplitude %s: %s refers to unknown arm %r" % (amp, what, arm))
        _need(arms[arm]["amplitude"] == amp, "amplitude %s: arm %s belongs to amplitude %s" % (amp, arm, arms[arm]["amplitude"]))
        return arms[arm]

    for amp, c in raw["amplitudes"].items():
        if c.get("control"):
            continue
        prod = ref(c.get("production"), amp, "production")
        comps = c.get("components", [])
        _need(set(comps) <= set(COMPONENTS) and len(set(comps)) == len(comps), "amplitude %s: components must be a subset of %s" % (amp, COMPONENTS))
        if "E_h" in comps:
            L = c.get("ladder")
            _need(L is not None, "amplitude %s: E_h needs a 'ladder' block" % amp)
            la = [ref(i, amp, "ladder") for i in L["arms"]]
            res = L["resolutions"]
            _need(len(la) == len(res) >= 2 and all(y > x for x, y in zip(res, res[1:])), "amplitude %s: ladder resolutions must increase" % amp)
            _need(len({a["group"] for a in la}) == 1, "amplitude %s: ladder arms must share one group (common seeds)" % amp)
            _need(la[L["production_index"]]["id"] == prod["id"], "amplitude %s: ladder production_index is not the production arm" % amp)
            for t in L.get("triples", []):
                _need(len(t) == 3 and t[0] < t[1] < t[2] < len(la), "amplitude %s: bad triple %r" % (amp, t))
            for g in L.get("gates", []):
                _need(g.get("type") in ("extra_level", "orders_agree"), "amplitude %s: unknown ladder gate %r" % (amp, g))
                trs = [g["triple"]] if g["type"] == "extra_level" else g.get("triples", [])
                _need(trs and all(len(t) == 3 and 0 <= t[0] < t[1] < t[2] < len(la) for t in trs), "amplitude %s: bad gate triples in %r" % (amp, g))
                if g["type"] == "extra_level":
                    _need("resolution" in g, "amplitude %s: an extra_level gate needs its 'resolution'" % amp)
                    a_g = ref(g.get("arm"), amp, "gate arm")
                    _need(a_g["group"] == la[0]["group"], "amplitude %s: the gate arm must share the ladder's seeds (group)" % amp)
            for i in L.get("fallback_levels", range(len(la))):
                _need(0 <= i < len(la), "amplitude %s: bad fallback_levels" % amp)
        if "E_W" in comps or "E_WxH" in comps:
            Wd = c.get("width")
            _need(Wd is not None, "amplitude %s: E_W/E_WxH need a 'width' block" % amp)
            need = ("wide_fine", "prod_fine") + (("wide_coarse", "prod_coarse") if "E_WxH" in comps else ())
            for k in need:
                _need(k in Wd, "amplitude %s: width block lacks %s" % (amp, k))
                ref(Wd[k], amp, "width." + k)
            if "E_WxH" in comps:
                _need(arms[Wd["wide_fine"]]["group"] == arms[Wd["wide_coarse"]]["group"], "amplitude %s: wide_fine/wide_coarse must share a group" % amp)
                _need(arms[Wd["prod_fine"]]["group"] == arms[Wd["prod_coarse"]]["group"], "amplitude %s: prod_fine/prod_coarse must share a group" % amp)
        if "E_int" in comps:
            H = c.get("hmax")
            _need(H is not None and "h16" in H and "h8" in H, "amplitude %s: E_int needs hmax {h16, h8}" % amp)
            _need(ref(H["h16"], amp, "hmax.h16")["group"] == ref(H["h8"], amp, "hmax.h8")["group"], "amplitude %s: h_max arms must share a group" % amp)
        if c.get("distance_arm"):
            ref(c["distance_arm"], amp, "distance_arm")
    _validate_blocks(raw, arms)
    fam = Prereg(raw).family_ids()
    _need(len(set(fam)) == len(fam) and fam, "the Holm family must be non-empty and without duplicates")
    for h in fam:
        kind, _, amp = h.partition(":")
        _need(kind in ("robust", "gate5") and amp in raw["amplitudes"], "unknown hypothesis id %r in the Holm family" % h)


def load_prereg(path):
    with open(path) as f:
        given = json.load(f)
    raw = _merge(DEFAULTS, given)
    raw["windows"] = given.get("windows", DEFAULTS["windows"])
    for k in ("amplitudes", "arms", "family", "targets", "first_order", "report_levels", "qualification"):
        if k in given:
            raw[k] = given[k]
    validate(raw)
    return Prereg(raw)
