"""``python3 -m sf43ana check-finite --summary <summary.json>``: the smoke test's structural check of a ``summary.json``.

Prints ONLY structure and COUNTS (per section: finite numbers, non-finite markers, other leaves) and ``STRUCTURE_OK`` or
``STRUCTURE_FAIL <reasons>``.  It never prints a number of an estimand, a bound, a p-value or a decision, nor a class name: the smoke
test must not read alpha beyond finiteness.  Reasons are phrased with section and key NAMES only.

STRUCTURE_FAIL when: a required section is missing; a descriptive block recorded an ``error``; the Holm family does not list ``m``
hypotheses or one lacks ``p`` / ``level`` / ``rejected``; a configured amplitude has no ``gate4`` entry with both transverse estimators
for the primary basis; a required section holds no finite number; a reported MEAN of a production arm with ``N > 0`` is not finite.
Non-finite bounds and p-values are legitimate (unresolved tails) and only counted.
"""
import json
import math
import sys

REQUIRED = ("estimands", "gate4", "family", "budgets", "precision", "process")
NONFINITE = ("nan", "inf", "-inf")


def count(o):
    """(finite numbers, non-finite markers, other leaves) of a JSON value."""
    if isinstance(o, dict):
        t = [count(v) for v in o.values()]
    elif isinstance(o, list):
        t = [count(v) for v in o]
    elif isinstance(o, bool) or o is None:
        return (0, 0, 1)
    elif isinstance(o, (int, float)):
        return (1, 0, 0) if math.isfinite(o) else (0, 1, 0)
    elif isinstance(o, str):
        return (0, 1, 0) if o in NONFINITE else (0, 0, 1)
    else:
        return (0, 0, 1)
    return tuple(sum(x) for x in zip(*t)) if t else (0, 0, 0)


def _errors(o, path, out):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "error":
                out.append(path or "?")
            else:
                _errors(v, path + "." + k if path else k, out)
    elif isinstance(o, list):
        for i, v in enumerate(o):
            _errors(v, "%s[%d]" % (path, i), out)


def check(summary):
    """-> (lines to print, list of failure reasons)."""
    reasons, lines = [], []
    for sec in REQUIRED:
        if sec not in summary:
            reasons.append("missing section %s" % sec)
    for sec in sorted(summary):
        f, n, o = count(summary[sec])
        lines.append("section %s: finite=%d nonfinite=%d other=%d" % (sec, f, n, o))
        if sec in ("estimands", "gate4", "budgets") and f == 0:
            reasons.append("no finite number in section %s" % sec)
    errs = []
    for sec in ("gate5", "g7", "secondaries", "failures", "positive_control"):
        if sec in summary:
            _errors(summary[sec], sec, errs)
    for e in errs:
        reasons.append("error recorded in %s" % e)
    fam = summary.get("family", {})
    hyps = fam.get("hypotheses", [])
    m = fam.get("info", {}).get("m")
    lines.append("family: hypotheses=%s m=%s" % (len(hyps), m))
    if m is None or len(hyps) != m:
        reasons.append("family does not list m hypotheses")
    for h in hyps:
        for k in ("id", "p", "level", "rejected", "testable"):
            if k not in h:
                reasons.append("family hypothesis lacks %s" % k)
                break
    nz = 0
    for amp, g in summary.get("gate4", {}).items():
        pb = g.get("primary_basis")
        ests = g.get("bases", {}).get(pb, {}).get("estimators", {})
        if sorted(ests) != ["alpha_TAP", "alpha_TP"]:
            reasons.append("gate4 amplitude %s lacks the primary-basis estimators" % amp)
        for k in ("class", "holm_rejected"):
            if k not in g:
                reasons.append("gate4 amplitude %s lacks %s" % (amp, k))
    for amp, e in summary.get("estimands", {}).items():
        if e.get("N", 0) > 0:
            for est, bases in e.get("estimators", {}).items():
                for b, s in (bases.items() if est != "alpha_LP" else [("-", bases)]):
                    if isinstance(s, dict) and "mean" in s and not (isinstance(s["mean"], (int, float)) and math.isfinite(s["mean"])):
                        nz += 1
    lines.append("estimands: nonfinite_means=%d" % nz)
    if nz:
        reasons.append("a reported mean of a production arm is not finite")
    return lines, reasons


def run(path, out=None):
    out = out or sys.stdout
    with open(path) as f:
        summary = json.load(f)
    lines, reasons = check(summary)
    for ln in lines:
        print(ln, file=out)
    print("STRUCTURE_OK" if not reasons else "STRUCTURE_FAIL " + "; ".join(sorted(set(reasons))), file=out)
    return 0 if not reasons else 1
