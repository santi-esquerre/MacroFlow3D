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
    "reduced_pattern": "{arm}/{segment}/{window}.csv",
    "ok_verdicts": ["ok"],
    "levels": {"fwer": 0.05, "interval": 0.90, "ucl": 0.05},
    "null_per_sigma2": 7.992e-5,
    "numerical_fraction": 0.5,
    "eh": {"p_range": [0.5, 2.1], "p_cap": 2.0, "safety": 1.25, "fallback_factor": 3.0},
    "bootstrap": {"B_sampling": 4000, "B_budget": 4000, "B_ucl": 4000, "B_bca": 5000, "seed": 20261103},
    "sampling_method": "bootstrap-t",
}


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
        self.amps = sorted(raw["amplitudes"], key=lambda k: float(raw["amplitudes"][k].get("sigma2", k)))

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


def validate(raw):
    for key in ("amplitudes", "arms"):
        _need(key in raw, "pre-registration lacks '%s'" % key)
    ids = [a.get("id") for a in raw["arms"]]
    _need(all(isinstance(i, str) and i for i in ids) and len(set(ids)) == len(ids), "arm ids must be unique non-empty strings")
    arms = {a["id"]: a for a in raw["arms"]}
    for a in raw["arms"]:
        for k in ("amplitude", "group", "seeds", "n_conf", "floor"):
            _need(k in a, "arm %s lacks '%s'" % (a["id"], k))
        a.setdefault("source", a["id"])
        a.setdefault("code", "consumer")
        a.setdefault("hmax_div", -1)
        a.setdefault("reserves", [])
        a.setdefault("role", a.get("family", ""))
        _need(a["amplitude"] in raw["amplitudes"], "arm %s: unknown amplitude %r" % (a["id"], a["amplitude"]))
        s = list(a["seeds"]) + list(a["reserves"])
        _need(len(set(s)) == len(s), "arm %s: seeds/reserves are not unique" % a["id"])
        _need(2 <= a["floor"] <= a["n_conf"] <= len(a["seeds"]), "arm %s: need 2 <= floor <= n_conf <= len(seeds)" % a["id"])
    W = raw["windows"]
    _need(raw["primary_window"] in W, "primary_window not among windows")
    _need(raw["sampling_method"] in ("bootstrap-t", "student"), "sampling_method must be 'bootstrap-t' or 'student'")
    lo, hi = raw["eh"]["p_range"]
    _need(0 < lo <= hi and raw["eh"]["p_cap"] > 1.0 and raw["eh"]["safety"] >= 1.0, "bad eh parameters")

    def ref(arm, amp, what):
        _need(arm in arms, "amplitude %s: %s refers to unknown arm %r" % (amp, what, arm))
        _need(arms[arm]["amplitude"] == amp, "amplitude %s: arm %s belongs to amplitude %s" % (amp, arm, arms[arm]["amplitude"]))
        return arms[arm]

    for amp, c in raw["amplitudes"].items():
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
    for k in ("amplitudes", "arms", "family", "targets", "first_order", "report_levels"):
        if k in given:
            raw[k] = given[k]
    validate(raw)
    return Prereg(raw)
