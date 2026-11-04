"""``run``: the pre-registered analysis of addendum 1 -> ``<out>/addendum_summary.json`` + ``addendum_summary.md`` (+ ``reproduction.json``).

STDOUT / STDERR carry STRUCTURE ONLY (the job log is read during the blinded phase): arms and realizations per arm, common-seed counts per
contrast, finite flags, whether the reproduction check was evaluated (not its outcome), files verified, warnings / errors counts and
``STRUCTURE_OK`` | ``STRUCTURE_FAIL``.  No ratio, interval, class, alpha or rule outcome is printed; those are in the files, read after the freeze.
Exit 0 when the analysis completed (a rule that is "not evaluable" is a result), 1 if a block raised, 2 if an input was refused (a frozen file
that does not match the freeze manifest, missing ledger, ...).
"""
import json
import math
import os
import sys
import traceback
import warnings
from collections import Counter
from contextlib import contextmanager

import numpy as np

from sf43ana.config import ConfigError
from sf43ana.loader import LoadError
from sf43ana.summary import clean, sha256

from . import repro, rules
from .data import FROZEN_ARMS, LP, TP_FAR, TP_MAIN, Frozen, load_addendum, load_campaign, load_frozen, resolve

LABEL = "SF-43 addendum 1: exploratory and descriptive; outside the confirmatory Holm family (m = 8 unchanged)"


@contextmanager
def watch(sink, block):
    """Records the warnings raised in the block into ``sink[block]`` (unless ``-W error`` is active: unit tests keep them fatal)."""
    with warnings.catch_warnings(record=True) as w:
        if "error" not in sys.warnoptions:
            warnings.simplefilter("always")
        try:
            yield
        finally:
            c = Counter((x.category.__name__, str(x.message), "%s:%d" % (x.filename, x.lineno)) for x in w)
            if c:
                sink.setdefault(block, []).extend({"category": k[0], "message": k[1], "location": k[2], "count": n} for k, n in sorted(c.items()))


def _arm_info(v, keys):
    n = v.N
    return {"N": n, "n_conf": v.info.get("n_conf"), "stopped_at": v.info.get("stopped_at"), "missing_tables": v.info.get("missing_tables"),
            "seed_range": [int(v.seeds.min()), int(v.seeds.max())] if n else None,
            "finite": {"/".join(k): int(np.isfinite(v.get(k)).sum()) for k in keys}}


def _n_of(r):
    if isinstance(r, dict) and "evaluable" in r:
        return r["n"] if isinstance(r["n"], int) else sum(r["n"])
    return None


def _seed_range(seeds):
    """[first, last] seed of a ``{"start", "count"}`` block."""
    return [seeds["start"], seeds["start"] + seeds["count"] - 1]


def analyse(add, add_path, campaign, frozen_reduced, frozen_manifest, frozen_ledger, frozen_segments=None, j8_campaign=None):
    """-> (summary dict, the verified-frozen-files registry).  Raises LoadError / ConfigError / OSError when an input is refused."""
    errors, warns = [], {}
    prereg_path = resolve(add["frozen_prereg"], add_path)
    fro = Frozen(frozen_reduced, frozen_manifest, add["frozen_reduced"]["manifest_sha256"])
    with watch(warns, "loading"):  # warnings are recorded in the summary, not printed (the log is read during the blinded phase)
        P, fv, _ = load_frozen(prereg_path, fro, frozen_ledger)
        for a in FROZEN_ARMS:
            if fv[a].N < P.arms[a]["n_conf"]:
                raise LoadError("frozen arm %s has %d of %d realizations (wrong frozen directory or ledger)" % (a, fv[a].N, P.arms[a]["n_conf"]))
        cv, _ = load_campaign(P, add, campaign, "production")
        jv, _ = load_campaign(P, add, j8_campaign, "j8_conditional") if j8_campaign else ({}, None)
    S = {"version": 1, "label": LABEL, "config": {"budget_gpu_h_cap": add["budget_gpu_h_cap"], "hard_end_utc": add["hard_end_utc"],
                                                   "inference": add["inference"]["paired"]}}

    def block(name, fn):
        with watch(warns, name):
            try:
                return fn()
            except LoadError:  # a refused input (e.g. a frozen file that does not match the freeze manifest) is fatal, not a block error
                raise
            except Exception as e:  # noqa: BLE001  (a rule block that raises is recorded; the others continue)
                errors.append({"block": name, "type": type(e).__name__, "message": str(e), "traceback": traceback.format_exc()})
                print("BLOCK_ERROR %s: %s" % (name, type(e).__name__), file=sys.stderr)
                return {"error": "%s: %s" % (type(e).__name__, e)}

    rp = add["j1"]["reproduction"]
    present = os.path.isdir(os.path.join(campaign, "reduced", rp["arm"]))
    if present:
        S["reproduction"] = block("reproduction", lambda: repro.compare(
            campaign, rp["arm"], repro.parse_seeds("%d-%d" % tuple(_seed_range(rp["seeds"]))), frozen_reduced, frozen_manifest, frozen_segments, rp["reference"],
            add["frozen_reduced"]["manifest_sha256"], fro))
    else:
        S["reproduction"] = {"evaluated": False, "reason": "no reduced tables of %s in the campaign" % rp["arm"]}
    lo, hi = _seed_range(add["j1"]["seeds"])
    views60 = dict({a: fv[a].where(lo, hi) for a in ("s4_g5p", "s4_g5pol")}, **{k: v for k, v in cv.items() if k.startswith("j1_")})
    S["j1"] = {"n60": block("j1_n60", lambda: rules.evaluate_j1(add, views60, S["reproduction"], "n60"))}
    if "error" not in S["j1"]["n60"]:
        S["j8_trigger_inputs"] = rules.j8_inputs(add, S["j1"]["n60"])
    if jv:
        hi100 = _seed_range(add["j1"]["j8_trigger"]["seeds"])[1]
        v100 = {a: fv[a].where(lo, hi100) for a in ("s4_g5p", "s4_g5pol")}
        for arm in add["arms"]:
            if arm["job"] == "j8_conditional" and arm["origin"] in cv and arm["id"] in jv:
                v100[arm["origin"]] = cv[arm["origin"]].concat(jv[arm["id"]])
        S["j1"]["n100"] = block("j1_n100", lambda: rules.evaluate_j1(add, v100, S["reproduction"], "n100"))
    views = dict(fv, **{k: v for k, v in cv.items() if not k.startswith("j1_")})
    S["j3"] = block("j3", lambda: rules.evaluate_j3(add, views))
    S["j9"] = block("j9", lambda: rules.evaluate_j9(add, views, S["j3"] if "error" not in S["j3"] else {"equivalent": False}))
    allv = dict(fv, **cv, **jv)
    S["arms"] = {a: dict(_arm_info(v, [LP] if a.startswith(("j1_", "j8_", "s4_")) else [TP_MAIN, TP_FAR]),
                         source="frozen" if a in fv else ("j8 campaign" if a in jv else "campaign")) for a, v in allv.items()}
    S["inputs"] = {"prereg_addendum_sha256": sha256(add_path), "frozen_manifest_sha256": fro.manifest_sha256,
                   "frozen_ledger_sha256": sha256(frozen_ledger), "campaign_ledger_sha256": sha256(os.path.join(campaign, "ledger.jsonl")),
                   "j8_campaign": j8_campaign, "frozen_files_verified": len(fro.verified)}
    S["process"] = {"errors": errors, "errors_count": len(errors), "warnings": warns,
                    "warnings_count": sum(w["count"] for ws in warns.values() for w in ws)}
    return S, fro


def structure(S, fro):
    """Lines (counts and flags only) and the failure reasons."""
    lines, reasons = [], []
    for src in ("frozen", "campaign", "j8 campaign"):
        arms = [(a, d) for a, d in S["arms"].items() if d["source"] == src]
        if arms:
            lines.append("%s arms: %s" % (src, " ".join("%s realizations=%d finite=%s" % (a, d["N"], "/".join(str(x) for x in d["finite"].values()))
                                                         for a, d in arms)))
            for a, d in arms:
                if any(x != d["N"] for x in d["finite"].values()):
                    reasons.append("non-finite response values in arm %s" % a)
    counts = []
    for tag, blk in (("j1:n60", S["j1"].get("n60")), ("j1:n100", S["j1"].get("n100")), ("j3", S["j3"]), ("j9", S["j9"])):
        if blk and "error" not in blk:
            for name, r in blk["quantities"].items():
                counts.append("%s:%s=%s" % (tag, name, r["n"] if isinstance(r["n"], int) else "+".join(str(x) for x in r["n"])))
    lines.append("common seeds per contrast: " + " ".join(counts))
    lines.append("reproduction evaluated: %s" % ("yes" if S["reproduction"].get("evaluated") else "no"))
    lines.append("frozen files verified: %d" % len(fro.verified))
    p = S["process"]
    lines.append("warnings=%d errors=%d" % (p["warnings_count"], p["errors_count"]))
    if p["errors_count"]:
        reasons.append("%d block error(s)" % p["errors_count"])
    return lines, reasons


# ----------------------------------------------------------------------------------------------------------------------------------
# markdown tables for the note
# ----------------------------------------------------------------------------------------------------------------------------------
def f(x, d=4):
    if x is None:
        return "-"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(f(y, d) for y in x) + "]"
    if isinstance(x, float):
        return "%.*g" % (d, x) if math.isfinite(x) else str(x)
    return str(x)


def qrow(name, r):
    if not r.get("evaluable"):
        return "| %s | %s | not evaluable: %s | | | |" % (name, f(r.get("n")), r.get("reason"))
    n = r["n"] if isinstance(r["n"], int) else "%d, %d" % tuple(r["n"])
    return "| %s | %s | %s | %s | %s |" % (name, n, f(r["ratio"]), f(r["ci90"]), f(r.get("student_ci90")))


QHEAD = ["| quantity | n | ratio | IC90 bootstrap-t / Welch | IC90 Student |", "|---|---|---|---|---|"]


def md(S):
    o = ["# SF-43 addendum 1: exploratory analysis (descriptive; outside the Holm family)", "", "Generated by `sf43add.ana run`; read only after the freeze.", ""]
    r = S["reproduction"]
    o += ["## Reproduction check (3.1.1)", "", "- evaluated: %s; passed: %s" % (r.get("evaluated"), r.get("passed")),
          "- seeds %s, rows compared %s, max relative difference %s, crossing CSV sha256 equal/differ/unavailable %s/%s/%s" % (
              f(r.get("seeds")), f(r.get("n_rows")), f(r.get("max_rel")), f(r.get("csv_sha_equal")), f(r.get("csv_sha_differ")),
              f(r.get("csv_sha_unavailable"))),
          "- problems: %s" % (f(r.get("problems")) if r.get("problems") else "none"), ""]
    for lab, j in S["j1"].items():
        if "error" in j:
            o += ["## J1 (%s): ERROR %s" % (lab, j["error"]), ""]
            continue
        o += ["## J1 (%s)" % lab, "", "- status: %s; class: **%s**; %s" % (j["status"], j["class"], j["reason"]), ""] + QHEAD
        o += [qrow(k, v) for k, v in j["quantities"].items()] + ["", "Completeness (mean, fraction complete, n):", ""]
        o += ["| arm | n | mean | fraction complete | quantiles 0/5/25/50/75/95/100 |", "|---|---|---|---|---|"]
        for a, c in j["completeness"].items():
            o.append("| %s | %s | %s | %s | %s |" % (a, f(c.get("n")), f(c.get("mean")), f(c.get("fraction_complete")), f(list(c.get("quantiles", {}).values()), 4)
                                                      if c.get("evaluable") else c.get("reason")))
        m, rep = j["manipulation"], j["representativity"]
        o += ["", "- manipulation (%s): passed %s (mean %s >= %s, fraction %s >= %s)" % (m.get("arm"), m.get("passed"), f(m.get("mean")),
                                                                                     f(m.get("mean_completeness_min")), f(m.get("fraction_complete")),
                                                                                     f(m.get("fraction_complete_min"))),
              "- representativity q_base vs frozen %s %s: frozen ratio in bootstrap-t IC %s, in Student IC %s, Student ICs overlap %s" % (
                  f(rep["frozen"]["ratio"]), f(rep["frozen"]["student_ci90"]), rep.get("frozen_ratio_in_bootstrap_t_ci"),
                  rep.get("frozen_ratio_in_student_ci"), rep.get("student_ci_overlap_with_frozen")), "", "Conditions (value, truth):", ""]
        for cname, cs in j["conditions"].items():
            o.append("- %s: %s" % (cname, "; ".join("%s %s %s -> %s %s" % (c["lhs"], c["op"], f(c["rhs"]), f(c["value"]), c["truth"]) for c in cs)))
        o.append("")
    if "j8_trigger_inputs" in S:
        o += ["## J8 trigger inputs (n = 60)", ""] + ["- %s: IC90 %s, thresholds %s, contained in the interior: %s" % (k, f(v["ci90"]), f(v["thresholds"]), f(v["contained"]))
                                                      for k, v in S["j8_trigger_inputs"]["quantities"].items()]
        o += ["- any contained: %s" % S["j8_trigger_inputs"]["any_contained"], ""]
    j3 = S["j3"]
    o += ["## J3", ""]
    if "error" in j3:
        o += ["ERROR %s" % j3["error"], ""]
    else:
        o += ["- class: **%s** (%s)" % (j3["class"], j3["reason"]), ""] + QHEAD + [qrow(k, v) for k, v in j3["quantities"].items()] + [""]
    j9 = S["j9"]
    o += ["## J9", ""]
    if "error" in j9:
        o += ["ERROR %s" % j9["error"], ""]
    else:
        o += ["- class: **%s** (%s); realizations completed %d (minimum %d); J3 qualification: %s" % (j9["class"], j9["reason"], j9["n_completed"], j9["min_completed"],
                                                                                              j9["j3_qualification"] or "none (j3 equivalent)"),
              "- frozen F24 %s, percentile IC90 %s" % (f(j9["F24_frozen"]["ratio"]), f(j9["F24_frozen"]["percentile_ci90"])), ""] + QHEAD
        o += [qrow(k, v) for k, v in {**j9["quantities"], **j9["descriptive_levels"]}.items()] + [""]
    o += ["## Process", "", "- errors: %d, warnings: %d, frozen files verified: %d" % (S["process"]["errors_count"], S["process"]["warnings_count"],
                                                                                    S["inputs"]["frozen_files_verified"]), ""]
    return "\n".join(o)


def main(a):
    try:
        add = load_addendum(a.prereg)
        S, fro = analyse(add, a.prereg, a.campaign, a.frozen_reduced, a.frozen_manifest, a.frozen_ledger,
                         a.frozen_segments, a.j8_campaign)
    except (LoadError, ConfigError, OSError, KeyError, ValueError) as e:
        print("sf43add.ana run: input refused: %s" % type(e).__name__ + (": %s" % e if isinstance(e, (LoadError, ConfigError, KeyError)) else ""), file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    for name, text in (("addendum_summary.json", json.dumps(clean(S), indent=1, sort_keys=True) + "\n"), ("addendum_summary.md", md(clean(S)) + "\n"),
                       ("reproduction.json", json.dumps(clean(S["reproduction"]), indent=1, sort_keys=True) + "\n")):
        with open(os.path.join(a.out, name), "w") as fh:
            fh.write(text)
    lines, reasons = structure(S, fro)
    print("\n".join(lines))
    print("STRUCTURE_OK" if not reasons else "STRUCTURE_FAIL " + "; ".join(reasons))
    return 0 if S["process"]["errors_count"] == 0 else 1
