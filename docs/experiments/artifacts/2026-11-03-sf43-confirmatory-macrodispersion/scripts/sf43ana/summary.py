"""Orchestration and ``summary.json`` (sections ``estimands``, ``gate4``, ``family``, ``budgets``, ``precision``, ``process``).

Optional blocks of the sibling modules ``gate5``, ``g7``, ``secondaries``, ``failures``, ``positive_control``: when the module
``sf43ana.<name>`` exists and defines ``analyse(ctx) -> dict | None``, the dictionary is merged under the key ``<name>`` (``ctx`` is a
``Context``: ``prereg``, ``arms``, ``campaign``, ``reduced``, ``gate5_p``, ``ledger`` and the lazily built ``data`` of ``loader``).  Absent
modules are skipped, ``None`` (block not configured) is omitted.  A failure of a DESCRIPTIVE block (all but ``gate5``) is recorded as
``{"error": ...}`` under its key and on stderr (``BLOCK_ERROR``): a descriptive analysis never costs the primary results; Gate 5
errors propagate (they decide a hypothesis of the family).

Gate 5 p-values (``--gate5``): omitted or ``auto``: computed by ``sf43ana.gate5`` from the pre-registration's ``gate5`` block (none ->
no p-values); ``none``: no p-values; a path: JSON ``{"p": {"<amplitude>": p, ...}}`` (the ``p`` wrapper optional; a value may also be
``{"p": ...}``).  A missing amplitude enters the Holm family with ``p = 1`` (untestable).
"""
import hashlib
import importlib
import importlib.util
import json
import math
import os

import numpy as np

from . import config as cfgmod
from . import figures
from .estimands import TRANSVERSE, amplitude_estimands
from .gate4 import (CLASS_PHYS, RobustTest, classify_amplitude, classify_estimator, design_from_arms)
from .holm import holm
from .loader import build_data, load_all

OPTIONAL = ("gate5", "g7", "secondaries", "failures", "positive_control")


class Context:
    def __init__(self, prereg, arms, campaign, reduced, gate5_p, ledger=None):
        self.prereg, self.arms, self.campaign, self.reduced, self.gate5_p, self.ledger = prereg, arms, campaign, reduced, gate5_p, ledger
        self._data = None

    @property
    def data(self):
        if self._data is None:
            self._data = build_data(self.prereg, self.arms, self.ledger)
        return self._data


def _analyse(name, ctx):
    """-> result dict, None (module absent or block not configured); an error of a descriptive block is returned as {"error": ...}."""
    if importlib.util.find_spec("sf43ana." + name) is None:
        return None
    fn = getattr(importlib.import_module("sf43ana." + name), "analyse", None)
    if not callable(fn):
        return None
    if name == "gate5":
        return fn(ctx)
    try:
        return fn(ctx)
    except Exception as e:  # noqa: BLE001  (descriptive block: recorded, never fatal)
        import sys
        import traceback
        print("BLOCK_ERROR %s: %s: %s" % (name, type(e).__name__, e), file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return {"error": "%s: %s" % (type(e).__name__, e)}


def read_gate5(path):
    if path in (None, "none", "", "auto"):
        return {}
    with open(path) as f:
        raw = json.load(f)
    raw = raw.get("p", raw)
    return {str(k): float(v["p"] if isinstance(v, dict) else v) for k, v in raw.items()}


def clean(o):
    """JSON-safe copy: numpy to Python, non-finite floats as the strings 'nan' / 'inf' / '-inf'."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return f if math.isfinite(f) else ("nan" if math.isnan(f) else "inf" if f > 0 else "-inf")
    return o


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def gate4_tests(prereg, arms):
    """{(amp, basis, estimator): RobustTest} for every amplitude, both bases, both transverse estimators."""
    out = {}
    for amp in prereg.amps:
        for role, b in prereg["bases"].items():
            for e in TRANSVERSE:
                label = "%s/%s/%s" % (amp, e, b)
                out[(amp, b, e)] = RobustTest(design_from_arms(prereg, amp, arms, e, b), prereg.null_value(amp), prereg.raw, label)
    return out


def _scalar(v):
    return float(np.asarray(v).reshape(-1)[0])


def build_family(prereg, tests, gate5_p):
    pb = prereg["bases"]["primary"]
    hyps = []
    for hid in prereg.family_ids():
        kind, _, amp = hid.partition(":")
        if kind == "robust":
            ts = [tests[(amp, pb, e)] for e in TRANSVERSE]
            ok = all(t.testable for t in ts)
            hyps.append({"id": hid, "testable": ok, "p": max(_scalar(t.p_value()[0]) for t in ts) if ok else 1.0,
                         "exact": (lambda lv, ts=ts: all(t.reject(lv) for t in ts)) if ok else None,
                         "note": "" if ok else "; ".join(r for t in ts for r in t.missing)})
        else:
            ok = amp in gate5_p
            hyps.append({"id": hid, "testable": ok, "p": gate5_p[amp] if ok else 1.0, "exact": None,
                         "note": "" if ok else "no Gate 5 p-value supplied"})
    return hyps


def budgets_section(prereg, tests, levels):
    out = {}
    for (amp, b, e), t in tests.items():
        entry = out.setdefault(amp, {}).setdefault(b, {})
        if not t.testable:
            entry[e] = {"testable": False, "reasons": t.missing}
            continue
        rep = t.report(levels)
        comps = {}
        for name, c in rep["components"].items():
            comps[name] = {"raw": c["raw"], "enforced": c["enforced"], "details": c.get("details")}
        entry[e] = {"testable": True, "K": rep["K"], "levels": rep["a"], "L_a_over_2": rep["L_a_over_2"], "RL": rep["RL"],
                    "null_value": t.null, "components": comps}
    return out


def precision_section(prereg, est):
    out = {}
    for amp, tg in prereg.raw.get("targets", {}).items():
        out[amp] = {}
        for name, target in tg.items():
            e = est.get(amp, {}).get("estimators", {}).get(name)
            if e is None:
                continue
            for b, s in (e.items() if name in TRANSVERSE else [("-", e)]):
                hw = s["halfwidth_bootstrap_t"]
                out[amp]["%s/%s" % (name, b)] = {"target_halfwidth": target, "achieved_halfwidth": hw,
                                                 "met": bool(math.isfinite(hw) and hw <= target), "n": s["n"]}
    return out


def run(prereg_path, campaign, reduced, gate5_path, out_dir):
    prereg = cfgmod.load_prereg(prereg_path)
    arms, proc, ledger = load_all(prereg, campaign, reduced)
    auto = gate5_path in (None, "auto", "")
    gate5_p = read_gate5(gate5_path)
    ctx = Context(prereg, arms, campaign, reduced, gate5_p, ledger)
    optional = {}
    if auto and prereg.raw.get("gate5"):
        optional["gate5"] = _analyse("gate5", ctx)
        if optional["gate5"] is not None:
            gate5_p.update({str(k): float(v) for k, v in optional["gate5"]["pvalues"].items()})
    est = {amp: amplitude_estimands(prereg, amp, arms) for amp in prereg.amps}
    tests = gate4_tests(prereg, arms)
    hyps = build_family(prereg, tests, gate5_p)
    fwer = prereg["levels"]["fwer"]
    res, info = holm(hyps, fwer, prereg["tie_rtol"])
    by_id = {r["id"]: r for r in res}
    for r, h in zip(res, hyps):
        r["note"] = h["note"]
    pb = prereg["bases"]["primary"]
    gate4 = {}
    for amp in prereg.amps:
        g = {"sigma2": prereg.sigma2(amp), "null_value": prereg.null_value(amp), "primary_basis": pb, "bases": {}}
        for role, b in prereg["bases"].items():
            per = {}
            for e in TRANSVERSE:
                t = tests[(amp, b, e)]
                p, never = t.p_value()
                cl = classify_estimator(t, prereg.raw, "%s/%s/%s" % (amp, e, b))
                per[e] = {"testable": t.testable, "reasons": t.missing, "p": _scalar(p) if t.testable else 1.0,
                          "never_positive": bool(_scalar(never)) if t.testable else True, "class_inputs": cl}
            ps = [per[e]["p"] for e in TRANSVERSE]
            g["bases"][b] = {"role": role, "estimators": per, "amplitude_p": max(ps), "in_family": role == "primary"}
        rej = by_id.get("robust:%s" % amp, {}).get("rejected", False)
        cls, note = classify_amplitude(rej, {e: g["bases"][pb]["estimators"][e]["class_inputs"] for e in TRANSVERSE})
        g["holm_rejected"], g["class"], g["class_note"] = rej, cls, note
        g["bound_unresolved_bounded"] = ({e: g["bases"][pb]["estimators"][e]["class_inputs"]["bound"] for e in TRANSVERSE}
                                         if cls == "unresolved (bounded)" else None)
        gate4[amp] = g
    levels = prereg.raw.get("report_levels") or [fwer, fwer / info["m"]]
    summary = {"estimands": est, "gate4": gate4, "family": {"info": info, "hypotheses": res, "gate5_p_supplied": gate5_p},
               "budgets": budgets_section(prereg, tests, levels), "precision": precision_section(prereg, est)}
    summary["process"] = {"prereg_sha256": sha256(prereg_path), "prereg_effective": {k: v for k, v in prereg.raw.items() if k != "arms"},
                          "loader": proc, "arms": {a: d.info for a, d in arms.items()},
                          "classes_legend": [CLASS_PHYS, "numerical", "unresolved (bounded)", "unresolved"]}
    bs, m = prereg["bootstrap"], info["m"]
    summary["process"]["bootstrap_resolution"] = {
        "note": "a tail t is resolvable by B resamples iff t (B + 1) >= 1; the Holm worst level is fwer/m; E_h uses p_L at tail a/(4K), "
                "the bootstrap-t sampling limit tail a/2 (an unresolved tail silently gives the conservative fallback / -inf)",
        "B_sampling_needed": int(math.ceil(2 * m / fwer)), "B_sampling": bs["B_sampling"],
        "per_amplitude_B_budget_needed": {a: int(math.ceil(4 * len(prereg.amp(a).get("components", [])) * m / fwer)) for a in prereg.amps},
        "B_budget": bs["B_budget"],
        "ok": bool(bs["B_sampling"] + 1 >= 2 * m / fwer and all(bs["B_budget"] + 1 >= 4 * len(prereg.amp(a).get("components", [])) * m / fwer for a in prereg.amps))}
    for name in OPTIONAL:
        out = optional[name] if name in optional else _analyse(name, ctx)
        if out is not None:
            summary[name] = out
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(clean(summary), f, indent=1, sort_keys=True)
        f.write("\n")
    figures.make(path, os.path.join(out_dir, "figures"))  # never raises; FIGURES_SKIPPED if matplotlib is absent
    return summary, path
