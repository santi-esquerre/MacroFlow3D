"""Orchestration and ``summary.json`` (sections ``estimands``, ``gate4``, ``family``, ``budgets``, ``precision``, ``process``).

Optional blocks of the sibling modules ``gate5``, ``g7``, ``secondaries``, ``failures``, ``positive_control``: when the module
``sf43ana.<name>`` exists and defines ``analyse(ctx) -> dict | None``, the dictionary is merged under the key ``<name>`` (``ctx`` is a
``Context``: ``prereg``, ``arms``, ``campaign``, ``reduced``, ``gate5_p``, ``ledger`` and the lazily built ``data`` of ``loader``).  Absent
modules are skipped, ``None`` (block not configured) is omitted.  A failure of a DESCRIPTIVE block (all but ``gate5``) is recorded as
``{"error": ...}`` under its key and on stderr (``BLOCK_ERROR``): a descriptive analysis never costs the primary results.

Per-hypothesis isolation: an exception while computing ONE hypothesis of the family (the robust sign at one amplitude, Gate 5 at one
amplitude) or a non-finite p-value makes that hypothesis untestable (``p = 1``, ``status`` ``error`` / ``error: non-finite p``); it is
recorded in ``process.errors`` (id, status, exception type, message, traceback; ``process.errors_count``) and on stderr
(``HYPOTHESIS_ERROR <id>``), and the analysis continues with the rest of the family (Holm over the FIXED family, ``m`` never shrinks).
Every warning is recorded per block under ``process.warnings`` (category, message, ``file:line``, count; ``process.warnings_count``);
under ``-W error`` (the unit tests) warnings stay exceptions.  ``run`` exits 0 whenever ``summary.json`` was written.

Qualification gate (H4, H13 vs H15; ``sf43ana.qualification``): a robust sign whose design is testable by the floors is TESTABLE only if the
qualification document named by the pre-registration (``qualification``; ``--qualification`` overrides, ``none`` = no document) qualifies the
pipeline of its structure at the largest profile that fits the REALIZED arm sizes (H15: pipeline level; the sampling-limit cells are reported
in ``process.qualification.sampling_diagnostic``, never gating).  Otherwise ``p = 1`` and the hypothesis ``status`` is ``not qualified at the realized sizes`` (``no qualification
document`` without a document).  The estimates, budgets and classes stay descriptive; ``gate4.<amp>.qualification`` and
``process.qualification`` record the profile, the reasons and the document hash.

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
import sys
import warnings
from collections import Counter
from contextlib import contextmanager

import numpy as np

from . import config as cfgmod
from . import figures
from . import qualification as qual_mod
from .estimands import TRANSVERSE, amplitude_estimands
from .gate4 import (CLASS_PHYS, FailedTest, RobustTest, classify_amplitude, classify_estimator, design_from_arms, record_error)
from .holm import holm
from .loader import build_data, load_all

OPTIONAL = ("gate5", "g7", "secondaries", "failures", "positive_control", "pollock", "gate4_power", "basis")


class Context:
    def __init__(self, prereg, arms, campaign, reduced, gate5_p, ledger=None):
        self.prereg, self.arms, self.campaign, self.reduced, self.gate5_p, self.ledger = prereg, arms, campaign, reduced, gate5_p, ledger
        self._data = None
        self.errors = []    # per-hypothesis errors (``gate4.record_error``) -> process.errors
        self.warnings = {}  # block -> [{category, message, location, count}] -> process.warnings

    @property
    def data(self):
        if self._data is None:
            self._data = build_data(self.prereg, self.arms, self.ledger)
        return self._data


@contextmanager
def watch(ctx, block):
    """Records the warnings raised inside the block into ``ctx.warnings[block]`` (counted per category / message / location).  Unless ``-W error``
    is active (unit tests: warnings stay fatal) every occurrence is recorded, not only the first per location."""
    with warnings.catch_warnings(record=True) as w:
        if "error" not in sys.warnoptions:
            warnings.simplefilter("always")
        try:
            yield
        finally:
            c = Counter((x.category.__name__, str(x.message), "%s:%d" % (x.filename, x.lineno)) for x in w)
            if c:
                ctx.warnings.setdefault(block, []).extend({"category": k[0], "message": k[1], "location": k[2], "count": n} for k, n in sorted(c.items()))


def _analyse(name, ctx):
    """-> result dict, None (module absent or block not configured); an error of a descriptive block is returned as {"error": ...}."""
    if importlib.util.find_spec("sf43ana." + name) is None:
        return None
    fn = getattr(importlib.import_module("sf43ana." + name), "analyse", None)
    if not callable(fn):
        return None
    if name == "gate5":  # per-amplitude errors are isolated inside the block (hypotheses of the family)
        with watch(ctx, name):
            return fn(ctx)
    try:
        with watch(ctx, name):
            return fn(ctx)
    except Exception as e:  # noqa: BLE001  (descriptive block: recorded, never fatal)
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


def gate4_tests(prereg, arms, errors=None):
    """{(amp, basis, estimator): RobustTest} for every amplitude, both bases, both transverse estimators.  A construction failure is recorded
    in ``errors`` (id ``robust:<amp>`` for the primary basis) and the entry is an untestable ``FailedTest``."""
    out = {}
    for amp in prereg.amps:
        for role, b in prereg["bases"].items():
            for e in TRANSVERSE:
                label = "%s/%s/%s" % (amp, e, b)
                try:
                    out[(amp, b, e)] = RobustTest(design_from_arms(prereg, amp, arms, e, b), prereg.null_value(amp), prereg.raw, label)
                except Exception as ex:  # noqa: BLE001
                    record_error(errors, "robust:%s" % amp if role == "primary" else "gate4:" + label, ex)
                    out[(amp, b, e)] = FailedTest(label, prereg.null_value(amp))
    return out


def _scalar(v):
    return float(np.asarray(v).reshape(-1)[0])


def qualification_status(prereg, tests, doc):
    """{amp: {"status": None | NO_DOC | NOT_QUALIFIED, "detail": ...}} for the robust sign of every amplitude whose primary-basis design is
    testable by the floors (otherwise ``{"evaluated": False}``)."""
    pb = prereg["bases"]["primary"]
    out = {}
    for amp in prereg.amps:
        ts = [tests[(amp, pb, e)] for e in TRANSVERSE]
        if not all(t.testable and t.design is not None for t in ts):
            out[amp] = {"status": None, "evaluated": False, "reason": "the design is untestable by the floors / missing arms (see the hypothesis note)"}
            continue
        st, detail = qual_mod.check(doc, len(ts[0].design.components), qual_mod.realized_sizes(ts[0].design))
        out[amp] = {"status": st, "evaluated": True, **detail}
    return out


def build_family(prereg, tests, gate5_p, errors=None, qual=None):
    """Hypotheses of the FIXED family in order.  Any exception of a hypothesis (or a non-finite p) makes it untestable with ``p = 1`` and ``status``
    ``error``; ``errors`` (list) receives the record (``gate4.record_error``); ``errors`` entries of ``gate5:<amp>`` already there mark Gate 5."""
    pb = prereg["bases"]["primary"]
    gate5_failed = {e["id"] for e in errors or []}
    hyps = []
    for hid in prereg.family_ids():
        kind, _, amp = hid.partition(":")
        h = {"id": hid, "testable": False, "p": 1.0, "exact": None, "note": "", "status": "ok"}
        hyps.append(h)
        try:
            if kind == "robust":
                ts = []
                for e in TRANSVERSE:
                    t = tests[(amp, pb, e)]
                    try:
                        if t.testable:
                            t.p_value()
                    except Exception as ex:  # noqa: BLE001
                        record_error(errors, hid, ex)
                        t = tests[(amp, pb, e)] = FailedTest(t.label, t.null)
                    ts.append(t)
                if not all(t.testable for t in ts):
                    h["note"] = "; ".join(r for t in ts for r in t.missing)
                    h["status"] = "error" if any(isinstance(t, FailedTest) for t in ts) else "ok"
                    continue
                qs = (qual or {}).get(amp, {}).get("status")
                if qs:  # qualification gate: p stays 1, the hypothesis stays in the family
                    h["note"], h["status"] = "; ".join([qs] + qual[amp].get("reasons", [])), qs
                    continue
                h["p"] = max(_scalar(t.p_value()[0]) for t in ts)
                h["testable"] = True

                def exact(lv, ts=ts, h=h):
                    try:
                        return all(t.reject(lv) for t in ts)
                    except Exception as ex:  # noqa: BLE001  (conservative: not rejected)
                        record_error(errors, h["id"], ex)
                        h["status"] = "error"
                        return False
                h["exact"] = exact
            elif hid in gate5_failed:
                h["status"], h["note"] = "error", "Gate 5 computation failed (process.errors)"
            elif amp in gate5_p:
                h["p"], h["testable"] = gate5_p[amp], True
            else:
                h["note"] = "no Gate 5 p-value supplied"
            if h["testable"] and not math.isfinite(h["p"]):
                record_error(errors, hid, ValueError("non-finite p-value"), status="error: non-finite p")
                h.update(p=1.0, testable=False, exact=None, status="error: non-finite p", note="non-finite p-value")
        except Exception as ex:  # noqa: BLE001
            record_error(errors, hid, ex)
            h.update(p=1.0, testable=False, exact=None, status="error", note="%s: %s" % (type(ex).__name__, ex))
    return hyps


def budgets_section(prereg, tests, levels, errors=None):
    out = {}
    for (amp, b, e), t in tests.items():
        entry = out.setdefault(amp, {}).setdefault(b, {})
        if not t.testable:
            entry[e] = {"testable": False, "reasons": t.missing}
            continue
        try:
            rep = t.report(levels)
        except Exception as ex:  # noqa: BLE001
            record_error(errors, "budgets:%s/%s/%s" % (amp, e, b), ex)
            entry[e] = {"testable": False, "reasons": ["error: %s: %s" % (type(ex).__name__, ex)]}
            continue
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


def gate5_reporting(optional, by_id):
    """Labels what decides and what only describes in the Gate 5 block: the confirmatory outcome is the Holm TOST decision of ``gate5:<amp>``;
    the G5.5 verdict (agree / disagree / inconclusive from the 90 % Fieller interval) is descriptive (acceptance D)."""
    for amp, a in optional["gate5"]["amplitudes"].items():
        h = by_id.get("gate5:%s" % amp)
        a["reporting"] = {
            "confirmatory_outcome": {
                "what": "Holm TOST decision of the family hypothesis gate5:%s (equivalence of mu_L / mu_P within [1/m, m] at the Holm level)" % amp,
                "decided": h is not None, "testable": None if h is None else h["testable"], "rejected": None if h is None else h["rejected"],
                "p_tost": None if h is None else h["p"], "holm_level": None if h is None else h["level"]},
            "descriptive_verdict": {
                "what": "G5.5 verdict from the %g %% Fieller interval of the ratio (agree | disagree | inconclusive); descriptive, never decides "
                        "the hypothesis" % (100 * optional["gate5"]["config"]["level"]),
                "verdict": a.get("verdict")}}


def _gate4_section(prereg, tests, by_id, pb, gate4, errors, qual=None):
    """``gate4`` entries per amplitude.  An exception of a test (p-value, class inputs) replaces it by an untestable ``FailedTest`` (recorded)."""
    for amp in prereg.amps:
        g = {"sigma2": prereg.sigma2(amp), "null_value": prereg.null_value(amp), "primary_basis": pb, "bases": {},
             "qualification": (qual or {}).get(amp)}
        for role, b in prereg["bases"].items():
            per = {}
            for e in TRANSVERSE:
                label = "%s/%s/%s" % (amp, e, b)
                t = tests[(amp, b, e)]
                try:
                    p, never = t.p_value()
                    cl = classify_estimator(t, prereg.raw, label)
                except Exception as ex:  # noqa: BLE001
                    record_error(errors, "gate4:" + label, ex)  # after the Holm step: a class-input failure does not change a decided hypothesis
                    t = tests[(amp, b, e)] = FailedTest(label, t.null)
                    p, never, cl = 1.0, True, classify_estimator(t, prereg.raw, label)
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


def run(prereg_path, campaign, reduced, gate5_path, out_dir, coverage=None, qualification=None):
    """``coverage``: None (the pre-registration's ``gate5.coverage_summary``), ``"none"`` (no coverage summary: Gate 5 descriptive) or a path.
    ``qualification``: None (the pre-registration's ``qualification``), ``"none"`` (no document: every robust sign has p = 1) or a path."""
    prereg = cfgmod.load_prereg(prereg_path)
    if coverage is not None and prereg.raw.get("gate5"):
        prereg.raw["gate5"]["coverage_summary"] = None if coverage == "none" else coverage
    if qualification is not None:
        prereg.raw["qualification"] = None if qualification == "none" else qualification
    qdoc, qinfo = qual_mod.load(prereg.raw.get("qualification"))
    ctx = Context(prereg, None, campaign, reduced, {}, None)
    with watch(ctx, "loader"):
        arms, proc, ledger = load_all(prereg, campaign, reduced)
    auto = gate5_path in (None, "auto", "")
    gate5_p = read_gate5(gate5_path)
    ctx.arms, ctx.gate5_p, ctx.ledger = arms, gate5_p, ledger
    errors = ctx.errors
    optional = {}
    if auto and prereg.raw.get("gate5"):
        optional["gate5"] = _analyse("gate5", ctx)
        if optional["gate5"] is not None:
            gate5_p.update({str(k): float(v) for k, v in optional["gate5"]["pvalues"].items()})
    est = {}
    with watch(ctx, "estimands"):
        for amp in prereg.amps:
            try:
                est[amp] = amplitude_estimands(prereg, amp, arms)
            except Exception as ex:  # noqa: BLE001  (not a hypothesis: recorded, the rest continues)
                record_error(errors, "estimands:%s" % amp, ex)
                est[amp] = {}
    with watch(ctx, "gate4"):
        tests = gate4_tests(prereg, arms, errors)
        qual = qualification_status(prereg, tests, qdoc)
        hyps = build_family(prereg, tests, gate5_p, errors, qual)
        fwer = prereg["levels"]["fwer"]
        res, info = holm(hyps, fwer, prereg["tie_rtol"])
    by_id = {r["id"]: r for r in res}
    for r, h in zip(res, hyps):
        r["note"], r["status"] = h["note"], h["status"]
    if optional.get("gate5") is not None:
        gate5_reporting(optional, by_id)
    pb = prereg["bases"]["primary"]
    gate4 = {}
    with watch(ctx, "gate4_classes"):
        _gate4_section(prereg, tests, by_id, pb, gate4, errors, qual)
    levels = prereg.raw.get("report_levels") or [fwer, fwer / info["m"]]
    with watch(ctx, "budgets"):
        budgets = budgets_section(prereg, tests, levels, errors)
    summary = {"estimands": est, "gate4": gate4, "family": {"info": info, "hypotheses": res, "gate5_p_supplied": gate5_p},
               "budgets": budgets, "precision": precision_section(prereg, est)}
    cov_path = (prereg.raw.get("gate5") or {}).get("coverage_summary")
    summary["process"] = {"prereg_sha256": sha256(prereg_path),
                          "coverage_summary": {"path": cov_path, "sha256": sha256(cov_path) if cov_path and os.path.isfile(cov_path) else None},
                          "qualification": qinfo, "prereg_effective": {k: v for k, v in prereg.raw.items() if k != "arms"},
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
    summary["process"].update(errors=errors, errors_count=len(errors), warnings=ctx.warnings,
                              warnings_count=sum(w["count"] for ws in ctx.warnings.values() for w in ws))
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary.json")
    with open(path, "w") as f:
        json.dump(clean(summary), f, indent=1, sort_keys=True)
        f.write("\n")
    figures.make(path, os.path.join(out_dir, "figures"))  # never raises; FIGURES_SKIPPED if matplotlib is absent
    return summary, path
