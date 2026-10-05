#!/usr/bin/env python3
"""SF-30 streamline-closure gate: analysis and classification of the run matrix.

Usage:
    python3 analyze.py <raw_dir> [--tables <file.md>] [--json <file.json>]
    python3 analyze.py --self-test

Reads every `summary.json` (schema `sf30-closure-gate-1`, written by `closure_gate`) found
recursively under <raw_dir>, identifies each run from the CONTENT of the file (never from
the directory name), prints twelve Markdown tables (stdout, or --tables) and applies the
decision rule pre-registered in the SF-30 bitacora (rows D-5, D-11, D-12 of 2026-10-05)
mechanically. Python 3 standard library only. Output is deterministic (sorted).

Conventions:
    R(run)      unweighted `R` at period 1 at the working tolerance 1e-8
    e_int(run)  `max_distance` at period 1 of the 1e-8 entry's `ladder_vs_tightest`
                (the tightest tolerance must be 1e-12; otherwise the run is invalid for
                classification)
    non_ok(run) `non_ok_fraction` (period 1, working tolerance)
    darcy_ok    `darcy_converged`
    E_ctrl(N; seed) = max(R of the gaussian2d run with the same sigma2, ell, seed, N;
                          R of the lester2021 eps 1, 1024-seed run at N)
                (seed 3001's matched control is used if the seed's own is not present)

Classes: does_not_close | closes | ambiguous | incomplete. A required run that is absent
gives `incomplete`; a required run that is present but lacks a quantity the rule needs
(no 1e-8 entry, no R, tightest tolerance not 1e-12) is "invalid for classification" and
also gives `incomplete`. A failed validity criterion gives `ambiguous`.
"""

import json
import math
import os
import statistics
import subprocess
import sys
import tempfile

# =====================================================================================
# Decision thresholds -- pre-registered in the SF-30 bitácora, 2026-10-05; do not edit.
# These are the ONLY numerical thresholds of the decision rule.
REL_CHANGE_MAX = 0.10       # |R(N_f) - R(N_c)| / R(N_f) < REL_CHANGE_MAX       (strict)
CONTROL_FACTOR = 10         # R > CONTROL_FACTOR * E_ctrl (strict) / R <= CONTROL_FACTOR * E_ctrl
INTEGRATOR_REL_MAX = 0.01   # e_int <= INTEGRATOR_REL_MAX * R(N_f)                 (non-strict)
NON_OK_FRACTION_MAX = 0.01  # non_ok_fraction <= NON_OK_FRACTION_MAX               (non-strict)
# =====================================================================================

# Definitions of the instrument configuration (not thresholds).
WORKING_TOL = 1e-8
TIGHTEST_TOL = 1e-12
GAUSS_TOLS = [1e-6, 1e-8, 1e-10, 1e-12]
N_FINE = 256
N_COARSE = 128
N_LADDER = (64, 128, 256)
SEEDS_128 = (3001, 3002, 3003, 3004, 3005)
SEED_MAIN = 3001
SEEDS_S4_L16 = (3001, 3002, 3003)
SEED_RNG_DEFAULT = 20261005
PROBE_SEEDS_FILE = "probe_seeds16.csv"

# Display flag of table 4 only (task specification); not part of the decision rule.
D3_DISPLAY_FLAG = 1e-9

# Reference values of the 2026-10-02 spectral probes (16 probe seeds), copied from
# docs/experiments/artifacts/2026-10-02-closure-probes/raw/closure.txt (column
# nonuniform_rms). control2d and lester2021 close in the probes.
PROBE_REFERENCE = {
    "lester_brk": 8.690e-3,   # eps 1
    "two_mode": 2.847e-2,     # eps 0.5
    "generic3d": 5.402e-2,    # eps 0.5
    "control2d": "closes",
    "lester2021": "closes",
}

SIGMA2_LABEL = {0.25: "s025", 1.0: "s1", 4.0: "s4"}
ELL_LABEL = {0.125: "l8", 0.0625: "l16"}
CASES = [(0.25, 0.125), (0.25, 0.0625), (1.0, 0.125), (1.0, 0.0625), (4.0, 0.125), (4.0, 0.0625)]
CASE_S4_L16 = (4.0, 0.0625)
ANALYTIC = [("lester2021", 1.0), ("lester_brk", 1.0), ("control2d", 1.0), ("two_mode", 0.5), ("generic3d", 0.5)]
GROUPS = ["controls", "matched", "matrix128", "ladder", "manyperiod"]

CLASSES = ("does_not_close", "closes", "ambiguous", "incomplete")


# -------------------------------------------------------------------------------------
# Expected run list (mirror of run_matrix.sh; the self-test cross-checks the two)
# -------------------------------------------------------------------------------------

def case_label(case):
    return SIGMA2_LABEL[case[0]] + "_" + ELL_LABEL[case[1]]


def gauss_id(prefix, case, seed, n, periods=1):
    rid = "%s_%s_r%d_n%d" % (prefix, case_label(case), seed, n)
    if periods > 1:
        rid += "_p%d" % periods
    return rid


def expected_runs():
    """Return the ordered list of the 103 pre-registered runs as dicts."""
    out = []

    def gauss(group, prefix, field, case, seed, n, periods=1):
        out.append({
            "group": group, "id": gauss_id(prefix, case, seed, n, periods), "field": field, "n": n,
            "sigma2": case[0], "ell": case[1], "seed": seed, "eps": None, "periods": periods,
            "seeds": "1024", "tols": GAUSS_TOLS, "working_tol": WORKING_TOL, "pcg_rtol": 1e-10,
        })

    for n in (64, 128, 256):
        for field, eps in ANALYTIC:
            for seeds, wt in (("p16", 1e-10), ("1024", WORKING_TOL)):
                out.append({
                    "group": "controls", "id": "a_%s_n%d%s" % (field, n, "_p16" if seeds == "p16" else ""),
                    "field": field, "n": n, "sigma2": None, "ell": None, "seed": None, "eps": eps,
                    "periods": 1, "seeds": seeds, "tols": GAUSS_TOLS, "working_tol": wt, "pcg_rtol": 1e-12,
                })
    for case in CASES:
        for n in (64, 128, 256):
            gauss("matched", "m", "gaussian2d", case, SEED_MAIN, n)
    for seed in (3002, 3003):
        for n in (128, 256):
            gauss("matched", "m", "gaussian2d", CASE_S4_L16, seed, n)
    for case in CASES:
        for seed in SEEDS_128:
            gauss("matrix128", "g", "gaussian", case, seed, 128)
    for case in CASES:
        for n in (64, 256):
            gauss("ladder", "g", "gaussian", case, SEED_MAIN, n)
    for seed in (3002, 3003):
        gauss("ladder", "g", "gaussian", CASE_S4_L16, seed, 256)
    for case in CASES:
        gauss("manyperiod", "p", "gaussian", case, SEED_MAIN, 128, 64)
    gauss("manyperiod", "p", "gaussian", CASE_S4_L16, SEED_MAIN, 256, 16)
    return out


EXPECTED = expected_runs()
EXPECTED_BY_ID = {e["id"]: e for e in EXPECTED}


def identity_of_expected(e):
    return (e["field"], e["n"], e["sigma2"], e["ell"], e["seed"], e["eps"], e["periods"], e["seeds"])


EXPECTED_BY_IDENTITY = {identity_of_expected(e): e for e in EXPECTED}


# -------------------------------------------------------------------------------------
# Loader
# -------------------------------------------------------------------------------------

def _get(d, *keys):
    """Nested get; returns None if any level is missing or null."""
    for k in keys:
        if d is None:
            return None
        if isinstance(k, int):
            if not isinstance(d, list) or k >= len(d):
                return None
            d = d[k]
        else:
            if not isinstance(d, dict):
                return None
            d = d.get(k)
    return d


def _float_or_none(x):
    return None if x is None else float(x)


def tol_entry(run, tol):
    for t in run["tolerances"] or []:
        if t.get("tol") is not None and float(t["tol"]) == tol:
            return t
    return None


def period_entry(entry, n):
    if entry is None:
        return None
    for p in entry.get("periods") or []:
        if p.get("n") == n:
            return p
    return None


def ladder_entry(entry, n):
    lad = _get(entry, "ladder_vs_tightest")
    if lad is None:
        return None
    for p in lad.get("periods") or []:
        if p.get("n") == n:
            return p
    return None


def load_run(path):
    with open(path, "r", encoding="utf-8") as f:
        s = json.load(f)
    c = s.get("configuration") or {}
    field = c.get("field")
    gaussian_kind = field in ("gaussian", "gaussian2d")
    seeds_source = c.get("seeds_source")
    n_seeds = c.get("n_seeds")
    if seeds_source == "file" and n_seeds == 16:
        seeds_kind = "p16"
    elif seeds_source == "generated" and n_seeds == 1024:
        seeds_kind = "1024"
    else:
        seeds_kind = "%s:%s" % (seeds_source, n_seeds)
    run = {
        "path": path,
        "schema_version": s.get("schema_version"),
        "field": field,
        "n": c.get("n"),
        "sigma2": _float_or_none(c.get("sigma2")) if gaussian_kind else None,
        "ell": _float_or_none(c.get("ell")) if gaussian_kind else None,
        "seed": c.get("seed") if gaussian_kind else None,
        "eps": None if gaussian_kind else _float_or_none(c.get("eps")),
        "periods": c.get("periods"),
        "seeds_kind": seeds_kind,
        "seeds_file": c.get("seeds_file"),
        "seed_rng": c.get("seed_rng"),
        "tols": sorted((float(t) for t in (c.get("tols") or [])), reverse=True),
        "working_tol": _float_or_none(c.get("working_tol")),
        "pcg_rtol": _float_or_none(c.get("pcg_rtol")),
        "darcy_ok": s.get("darcy_converged") is True,
        "summary": s,
        "tolerances": s.get("tolerances"),
    }
    ident = (run["field"], run["n"], run["sigma2"], run["ell"], run["seed"], run["eps"], run["periods"],
             run["seeds_kind"])
    exp = EXPECTED_BY_IDENTITY.get(ident)
    run["id"] = exp["id"] if exp else None
    run["group"] = exp["group"] if exp else None
    run["deviations"] = config_deviations(run, exp) if exp else []
    derive(run)
    return run


def config_deviations(run, exp):
    dev = []
    if run["schema_version"] != "sf30-closure-gate-1":
        dev.append("schema_version %r" % run["schema_version"])
    if run["tols"] != sorted(exp["tols"], reverse=True):
        dev.append("tols %s (expected %s)" % (run["tols"], sorted(exp["tols"], reverse=True)))
    if run["working_tol"] != exp["working_tol"]:
        dev.append("working_tol %r (expected %r)" % (run["working_tol"], exp["working_tol"]))
    if run["pcg_rtol"] != exp["pcg_rtol"]:
        dev.append("pcg_rtol %r (expected %r)" % (run["pcg_rtol"], exp["pcg_rtol"]))
    if exp["seeds"] == "1024" and run["seed_rng"] != SEED_RNG_DEFAULT:
        dev.append("seed_rng %r (expected %d)" % (run["seed_rng"], SEED_RNG_DEFAULT))
    if exp["seeds"] == "p16" and os.path.basename(str(run["seeds_file"])) != PROBE_SEEDS_FILE:
        dev.append("seeds_file %r (expected %s)" % (run["seeds_file"], PROBE_SEEDS_FILE))
    return dev


def derive(run):
    """Fill R, e_int, non_ok and the reasons a run is invalid for classification."""
    run["R"] = run["e_int"] = run["non_ok"] = None
    run["invalid_R"] = []
    run["invalid_eint"] = []
    if not run["darcy_ok"]:
        run["invalid_R"].append("darcy_converged is false")
        run["invalid_eint"].append("darcy_converged is false")
        return
    w = tol_entry(run, WORKING_TOL)
    if w is None:
        run["invalid_R"].append("no tolerance entry %g" % WORKING_TOL)
        run["invalid_eint"].append("no tolerance entry %g" % WORKING_TOL)
        return
    p1 = period_entry(w, 1)
    run["R"] = _float_or_none(_get(p1, "unweighted", "R"))
    run["non_ok"] = _float_or_none(_get(p1, "counts", "non_ok_fraction"))
    if run["R"] is None:
        run["invalid_R"].append("unweighted R at period 1 is null")
    if run["non_ok"] is None:
        run["invalid_R"].append("non_ok_fraction is missing")
    tight = _get(w, "ladder_vs_tightest", "tightest_tol")
    if tight is None:
        run["invalid_eint"].append("no ladder_vs_tightest for the %g entry" % WORKING_TOL)
    elif float(tight) != TIGHTEST_TOL:
        run["invalid_eint"].append("tightest tolerance is %g, not %g" % (float(tight), TIGHTEST_TOL))
    else:
        run["e_int"] = _float_or_none(_get(ladder_entry(w, 1), "max_distance"))
        if run["e_int"] is None:
            run["invalid_eint"].append("ladder max_distance at period 1 is null")


def find_summaries(raw_dir):
    paths = []
    for root, dirs, files in os.walk(raw_dir):
        dirs.sort()
        if "summary.json" in files:
            paths.append(os.path.join(root, "summary.json"))
    return sorted(paths)


def load_all(raw_dir):
    runs = []
    errors = []
    for p in find_summaries(raw_dir):
        try:
            runs.append(load_run(p))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append((p, "%s: %s" % (type(exc).__name__, exc)))
    by_id = {}
    duplicates = {}
    for r in runs:
        if r["id"] is None:
            continue
        if r["id"] in by_id:
            duplicates.setdefault(r["id"], [by_id[r["id"]]["path"]]).append(r["path"])
        else:
            by_id[r["id"]] = r
    for rid in duplicates:
        del by_id[rid]  # an ambiguous identity is not used for classification
    return runs, by_id, duplicates, errors


# -------------------------------------------------------------------------------------
# Decision rule
# -------------------------------------------------------------------------------------

def control_ids(case, seed, n, mode, by_id):
    """Ids of the runs whose R define E(n; seed) and a note about the seed used."""
    note = None
    if mode == "ctrl":
        mid = gauss_id("m", case, seed, n)
        if mid not in by_id and seed != SEED_MAIN:
            note = "matched control of seed %d at N=%d not present; seed %d used" % (seed, n, SEED_MAIN)
            mid = gauss_id("m", case, SEED_MAIN, n)
        return [mid, "a_lester2021_n%d" % n], note
    return ["a_control2d_n%d" % n, "a_lester2021_n%d" % n], note


def e_value(ids, by_id):
    vals = [by_id[i]["R"] if i in by_id else None for i in ids]
    if any(v is None for v in vals):
        return None
    return max(vals)


def evaluate(case, r, mode, by_id):
    """Apply the pre-registered rule to one case on one realization.

    mode 'ctrl': E = E_ctrl (pre-registered); mode 'lit': E = E_lit (information only).
    """
    f_id = gauss_id("g", case, r, N_FINE)
    c_id = gauss_id("g", case, r, N_COARSE)
    ctrl_f, note_f = control_ids(case, r, N_FINE, mode, by_id)
    ctrl_c, note_c = control_ids(case, r, N_COARSE, mode, by_id)
    ctrl_128, _ = control_ids(case, SEED_MAIN, N_COARSE, mode, by_id)
    real_ids = [gauss_id("g", case, s, N_COARSE) for s in SEEDS_128]
    validity_ids = [f_id, c_id, ctrl_f[0], ctrl_c[0]]
    required = []
    for i in [f_id, c_id] + ctrl_f + ctrl_c + ctrl_128 + real_ids:
        if i not in required:
            required.append(i)

    res = {
        "case": case_label(case), "sigma2": case[0], "ell": case[1], "realization": r, "mode": mode,
        "runs": {"fine": f_id, "coarse": c_id, "E_fine": ctrl_f, "E_coarse": ctrl_c, "E_128_seed3001": ctrl_128,
                 "realizations_128": real_ids},
        "notes": [n for n in (note_f, note_c) if n],
        "missing": [i for i in required if i not in by_id],
        "class": None, "reason": None,
    }

    def rv(i):
        return by_id[i]["R"] if i in by_id else None

    R_f, R_c = rv(f_id), rv(c_id)
    E_f, E_c, E_128 = e_value(ctrl_f, by_id), e_value(ctrl_c, by_id), e_value(ctrl_128, by_id)
    run_f = by_id.get(f_id)
    e_int = run_f["e_int"] if run_f else None
    rel = None
    if R_f is not None and R_c is not None:
        diff = abs(R_f - R_c)
        rel = diff / R_f if R_f != 0 else (math.inf if diff > 0 else math.nan)
    reals = [(i, rv(i)) for i in real_ids]
    crit = {
        "R_fine": R_f, "R_coarse": R_c, "rel_change": rel, "E_fine": E_f, "E_coarse": E_c,
        "E_128_seed3001": E_128, "e_int": e_int,
        "R_128_realizations": {i: v for i, v in reals},
        "rel_change_small": None, "far_above_control": None, "all_128_far_above_control": None,
        "within_control": None, "decreasing": None,
        "darcy_ok_all": None, "non_ok_max": None, "non_ok_ok": None, "e_int_ok": None,
    }
    if rel is not None:
        crit["rel_change_small"] = rel < REL_CHANGE_MAX
    if R_f is not None and E_f is not None:
        crit["far_above_control"] = R_f > CONTROL_FACTOR * E_f
        crit["within_control"] = R_f <= CONTROL_FACTOR * E_f
    if R_f is not None and R_c is not None:
        crit["decreasing"] = R_f < R_c
    if E_128 is not None and all(v is not None for _, v in reals):
        crit["all_128_far_above_control"] = all(v > CONTROL_FACTOR * E_128 for _, v in reals)
    vruns = [by_id[i] for i in validity_ids if i in by_id]
    if vruns:
        crit["darcy_ok_all"] = all(x["darcy_ok"] for x in vruns)
        nok = [x["non_ok"] for x in vruns if x["non_ok"] is not None]
        crit["non_ok_max"] = max(nok) if nok else None
        crit["non_ok_ok"] = all(x["non_ok"] is not None and x["non_ok"] <= NON_OK_FRACTION_MAX
                                for x in vruns if x["darcy_ok"])
    if e_int is not None and R_f is not None:
        crit["e_int_ok"] = (e_int <= INTEGRATOR_REL_MAX * R_f) or (E_f is not None and e_int <= E_f)
    res["criteria"] = crit

    # 1. any required run missing -> incomplete
    if res["missing"]:
        res["class"] = "incomplete"
        res["reason"] = "missing: " + ", ".join(res["missing"])
        return res
    # 2. validity (darcy_ok, non_ok) on the four runs -> ambiguous
    vfail = []
    for i in validity_ids:
        x = by_id[i]
        if not x["darcy_ok"]:
            vfail.append("Darcy not converged on %s" % i)
        elif x["non_ok"] is not None and not (x["non_ok"] <= NON_OK_FRACTION_MAX):
            vfail.append("non_ok_fraction %.4g on %s exceeds %g" % (x["non_ok"], i, NON_OK_FRACTION_MAX))
    if vfail:
        res["class"] = "ambiguous"
        res["reason"] = "validity: " + "; ".join(vfail)
        return res
    # 3. a required quantity unavailable -> invalid for classification -> incomplete
    invalid = []
    for i in required:
        for why in by_id[i]["invalid_R"]:
            invalid.append("%s (%s)" % (i, why))
    for why in by_id[f_id]["invalid_eint"]:
        invalid.append("%s e_int (%s)" % (f_id, why))
    if invalid:
        res["class"] = "incomplete"
        res["reason"] = "invalid for classification: " + "; ".join(invalid)
        return res
    # 4. integrator validity on the fine case run -> ambiguous
    if not crit["e_int_ok"]:
        res["class"] = "ambiguous"
        res["reason"] = ("validity: e_int %.4g on %s exceeds both %g x R(N_f) = %.4g and E(N_f) = %.4g"
                         % (e_int, f_id, INTEGRATOR_REL_MAX, INTEGRATOR_REL_MAX * R_f, E_f))
        return res
    # 5. the rule
    if crit["rel_change_small"] and crit["far_above_control"] and crit["all_128_far_above_control"]:
        res["class"] = "does_not_close"
        res["reason"] = ""
    elif crit["within_control"] and crit["decreasing"]:
        res["class"] = "closes"
        res["reason"] = ""
    else:
        failed = []
        if not crit["rel_change_small"]:
            failed.append("relative change %.4g is not < %g" % (rel, REL_CHANGE_MAX))
        if not crit["far_above_control"]:
            failed.append("R(N_f) is not > %gx E(N_f)" % CONTROL_FACTOR)
        if not crit["all_128_far_above_control"]:
            low = [i for i, v in reals if not (v > CONTROL_FACTOR * E_128)]
            failed.append("128^3 realizations not > %gx E(128; %d): %s" % (CONTROL_FACTOR, SEED_MAIN, ", ".join(low)))
        if not crit["within_control"]:
            failed.append("R(N_f) is not <= %gx E(N_f)" % CONTROL_FACTOR)
        if not crit["decreasing"]:
            failed.append("R(N_f) is not < R(N_c)")
        res["class"] = "ambiguous"
        res["reason"] = "neither rule holds: " + "; ".join(failed)
    return res


def classify_cases(by_id, mode):
    out = []
    for case in CASES:
        seeds = SEEDS_S4_L16 if case == CASE_S4_L16 else (SEED_MAIN,)
        evals = [evaluate(case, r, mode, by_id) for r in seeds]
        classes = [e["class"] for e in evals]
        if len(evals) == 1:
            cls, reason = classes[0], evals[0]["reason"]
        elif "incomplete" in classes:
            cls = "incomplete"
            reason = "; ".join("r%d: %s" % (e["realization"], e["reason"]) for e in evals if e["class"] == "incomplete")
        elif len(set(classes)) == 1:
            cls, reason = classes[0], ("" if classes[0] != "ambiguous" else
                                       "; ".join("r%d: %s" % (e["realization"], e["reason"]) for e in evals))
        else:
            cls = "ambiguous"
            reason = "realizations disagree: " + ", ".join("r%d=%s" % (e["realization"], e["class"]) for e in evals)
        out.append({"case": case_label(case), "sigma2": case[0], "ell": case[1], "class": cls, "reason": reason,
                    "realizations": evals})
    return out


# -------------------------------------------------------------------------------------
# Formatting
# -------------------------------------------------------------------------------------

def fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return "%.4g" % x
    return str(x)


def table(num, title, caption, headers, rows):
    lines = ["### %d. %s" % (num, title), "", caption, ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    if not rows:
        lines.append("| " + " | ".join(["(none)"] + [""] * (len(headers) - 1)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(fmt(v) for v in r) + " |")
    lines.append("")
    return "\n".join(lines)


def ratio(a, b):
    if a is None or b is None or b == 0:
        return None
    return a / b


def order(coarse, fine):
    """Observed order between grids N/2 and N: log2(R(N/2)/R(N))."""
    if coarse is None or fine is None or coarse <= 0 or fine <= 0:
        return None
    return math.log2(coarse / fine)


def stat(run, tol, n, *keys):
    if run is None:
        return None
    return _get(period_entry(tol_entry(run, tol), n), *keys)


def hypot2(v):
    if v is None or len(v) != 2 or v[0] is None or v[1] is None:
        return None
    return math.hypot(v[0], v[1])


# -------------------------------------------------------------------------------------
# Tables
# -------------------------------------------------------------------------------------

def t1_probes(by_id):
    rows = []
    for field, eps in ANALYTIC:
        Rs = {}
        for n in N_LADDER:
            run = by_id.get("a_%s_n%d_p16" % (field, n))
            tight = min(run["tols"]) if run and run["tols"] else None
            R = _float_or_none(stat(run, tight, 1, "unweighted", "R")) if run and run["darcy_ok"] else None
            Rs[n] = R
            mean = stat(run, tight, 1, "unweighted", "mean") if run and run["darcy_ok"] else None
            rich = ref = None
            if n == N_FINE:
                if Rs.get(N_FINE) is not None and Rs.get(N_COARSE) is not None:
                    rich = Rs[N_FINE] + (Rs[N_FINE] - Rs[N_COARSE]) / 3.0
                ref = PROBE_REFERENCE[field]
            obs = order(Rs.get(n // 2), R) if field == "control2d" and n > N_LADDER[0] else None
            rows.append([field, eps, n, tight, R, mean[0] if mean else None, mean[1] if mean else None,
                         stat(run, tight, 1, "unweighted", "max_abs_d") if run and run["darcy_ok"] else None,
                         rich, ref, obs])
    return table(1, "Analytic controls vs the spectral probes",
                 "Runs `a_<field>_n<N>_p16` (16 probe seeds), period 1 at the tightest tolerance: unweighted `R`, "
                 "mean displacement (d2, d3), `max|d|`; on the N=256 row the two-grid Richardson value "
                 "`R_256 + (R_256 - R_128)/3` and the 2026-10-02 spectral-probe value (`raw/closure.txt`); "
                 "for `control2d` the observed order `log2(R(N/2)/R(N))`.",
                 ["field", "eps", "N", "tol", "R", "mean d2", "mean d3", "max abs d", "Richardson 128/256",
                  "probe reference", "observed order (control2d)"], rows)


def t2_controls1024(by_id):
    rows = []
    for field, eps in ANALYTIC:
        for n in N_LADDER:
            run = by_id.get("a_%s_n%d" % (field, n))
            ok = run is not None and run["darcy_ok"]
            rows.append([field, eps, n, run["R"] if run else None, run["e_int"] if run else None,
                         _get(run["summary"], "round_trip", "max_distance") if ok else None,
                         stat(run, WORKING_TOL, 1, "mean_tau_flux_weighted") if ok else None,
                         hypot2(stat(run, WORKING_TOL, 1, "flux_weighted", "mean")) if ok else None])
    return table(2, "Analytic controls, 1024 seeds",
                 "Runs `a_<field>_n<N>` at the working tolerance 1e-8, period 1: unweighted `R`, `e_int` "
                 "(ladder `max_distance` 1e-8 vs 1e-12), round-trip `max_distance`, flux-weighted mean travel "
                 "time `<tau>`, magnitude of the flux-weighted mean displacement.",
                 ["field", "eps", "N", "R", "e_int", "round-trip max", "flux-weighted <tau>",
                  "abs(flux-weighted mean d)"], rows)


def gaussian_runs(by_id, field="gaussian"):
    rs = [r for r in by_id.values() if r["field"] == field]
    return sorted(rs, key=lambda r: (r["sigma2"], -r["ell"], r["seed"], r["n"], r["periods"]))


def t3_matrix(by_id):
    rows = []
    for run in gaussian_runs(by_id):
        s = run["summary"]
        ok = run["darcy_ok"]
        it = [_get(s, "darcy", "corrector_results", k, "iterations") for k in range(3)]
        p = lambda *k: stat(run, WORKING_TOL, 1, *k) if ok else None  # noqa: E731
        mean = p("unweighted", "mean")
        rows.append([case_label((run["sigma2"], run["ell"])), run["seed"], run["n"], run["periods"], ok,
                     _get(s, "field", "Y_variance"), "/".join(fmt(x) for x in it),
                     _get(s, "direction_field_diagnostics", "min_g1"),
                     _get(s, "direction_field_diagnostics", "backflow_volume_fraction"),
                     p("counts", "seed_backflow"), p("counts", "ok"), p("counts", "ok_with_backflow_encounter"),
                     p("counts", "non_ok_fraction"), run["R"], mean[0] if mean else None, mean[1] if mean else None,
                     p("unweighted", "max_abs_d"), p("unweighted", "var_d2"), p("unweighted", "var_d3"),
                     p("R_no_backflow"), p("flux_weighted", "R"), run["e_int"],
                     _get(s, "round_trip", "max_distance") if ok else None])
    return table(3, "Matrix",
                 "One row per `gaussian` run (case, seed, grid, periods), period 1 at the working tolerance 1e-8: "
                 "`Y` variance, PCG iterations of the three Darcy correctors, `min g1`, backflow volume fraction, "
                 "status counts, `non_ok_fraction`, unweighted `R`, mean displacement, `max|d|`, `var(d2)`, `var(d3)`, "
                 "`R` without backflow encounters, flux-weighted `R`, `e_int`, round-trip `max_distance`.",
                 ["case", "seed", "N", "periods", "darcy_ok", "Y var", "PCG it", "min g1", "backflow vol frac",
                  "seed_backflow", "ok", "ok with backflow", "non_ok frac", "R", "mean d2", "mean d3", "max abs d",
                  "var d2", "var d3", "R no backflow", "R flux-weighted", "e_int", "round-trip max"], rows)


def t4_matched(by_id):
    rows = []
    for run in gaussian_runs(by_id, "gaussian2d"):
        case = (run["sigma2"], run["ell"])
        d3 = stat(run, WORKING_TOL, 1, "unweighted", "max_abs_d3") if run["darcy_ok"] else None
        flag = None if d3 is None else ("ABOVE %g" % D3_DISPLAY_FLAG if d3 > D3_DISPLAY_FLAG else "")
        half = by_id.get(gauss_id("m", case, run["seed"], run["n"] // 2))
        rows.append([case_label(case), run["seed"], run["n"], run["darcy_ok"], run["R"], d3, flag,
                     order(half["R"] if half else None, run["R"])])
    return table(4, "Matched 2-D controls",
                 "Runs `gaussian2d` at the working tolerance 1e-8, period 1: unweighted `R` (the instrument level), "
                 "`max|d3|` (flagged when above %g), observed order `log2(R(N/2)/R(N))` for the same seed."
                 % D3_DISPLAY_FLAG,
                 ["case", "seed", "N", "darcy_ok", "R", "max abs d3", "flag", "observed order"], rows)


def t5_ladders(by_id):
    rows = []
    for case in CASES:
        for seed in (SEEDS_S4_L16 if case == CASE_S4_L16 else (SEED_MAIN,)):
            R = {n: (by_id[gauss_id("g", case, seed, n)]["R"] if gauss_id("g", case, seed, n) in by_id else None)
                 for n in N_LADDER}
            e128 = e_value(control_ids(case, seed, 128, "ctrl", by_id)[0], by_id)
            e256 = e_value(control_ids(case, seed, 256, "ctrl", by_id)[0], by_id)
            rel = None
            if R[256] is not None and R[128] is not None and R[256] != 0:
                rel = abs(R[256] - R[128]) / R[256]
            rows.append([case_label(case), seed, R[64], R[128], R[256], rel, e128, e256, ratio(R[256], e256)])
    return table(5, "Grid ladders",
                 "Unweighted `R` at the working tolerance for N = 64, 128, 256; relative change "
                 "`abs(R(256) - R(128))/R(256)`; `E_ctrl(128)`, `E_ctrl(256)` of the same seed; `R(256)/E_ctrl(256)`.",
                 ["case", "seed", "R(64)", "R(128)", "R(256)", "rel change 128->256", "E_ctrl(128)", "E_ctrl(256)",
                  "R(256)/E_ctrl(256)"], rows)


def classification_tables(num, title, caption, cases, ename):
    rows = []
    for c in cases:
        for e in c["realizations"]:
            k = e["criteria"]
            min_ratio = None
            vals = [v for v in k["R_128_realizations"].values() if v is not None]
            if vals and k["E_128_seed3001"]:
                min_ratio = min(vals) / k["E_128_seed3001"]
            rows.append([c["case"], e["realization"], k["R_fine"], k["R_coarse"], k["rel_change"],
                         k["rel_change_small"], k["E_fine"], ratio(k["R_fine"], k["E_fine"]),
                         k["far_above_control"], min_ratio, k["all_128_far_above_control"], k["within_control"],
                         k["decreasing"], k["darcy_ok_all"], k["non_ok_max"], k["non_ok_ok"], k["e_int"],
                         k["e_int_ok"], e["class"], e["reason"] + ("" if not e["notes"] else
                                                                     " [" + "; ".join(e["notes"]) + "]")])
    hdr = ["case", "realization", "R(256)", "R(128)", "rel change", "rel < %g" % REL_CHANGE_MAX,
           "%s(256)" % ename, "R(256)/%s(256)" % ename, "R(256) > %gx %s" % (CONTROL_FACTOR, ename),
           "min R_128/%s(128;%d)" % (ename, SEED_MAIN), "all five 128^3 > %gx" % CONTROL_FACTOR,
           "R(256) <= %gx %s" % (CONTROL_FACTOR, ename), "R(256) < R(128)", "darcy ok (4 runs)",
           "max non_ok (4 runs)", "non_ok <= %g" % NON_OK_FRACTION_MAX, "e_int", "e_int valid", "class", "reason"]
    t = table(num, title, caption, hdr, rows)
    crow = [[c["case"], c["sigma2"], c["ell"], c["class"],
             ", ".join("r%d=%s" % (e["realization"], e["class"]) for e in c["realizations"]), c["reason"]]
            for c in cases]
    t2 = "\n".join(["Per-case result (`(4, 0.0625)` receives a class only if realizations 3001, 3002, 3003 agree).",
                    "", "| case | sigma2 | ell | class | realizations | reason |", "|---|---|---|---|---|---|"]
                   + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in crow] + [""])
    return t + "\n" + t2


def t8_amplitude(by_id):
    rows = []
    for ell in (0.125, 0.0625):
        for seed in SEEDS_128:
            R = {}
            for s2 in (0.25, 1.0, 4.0):
                rid = gauss_id("g", (s2, ell), seed, 128)
                R[s2] = by_id[rid]["R"] if rid in by_id else None
            rows.append([ell, seed, R[0.25], R[1.0], R[4.0], ratio(R[1.0], R[0.25]), ratio(R[4.0], R[1.0])])
    return table(8, "Amplitude scaling",
                 "Grid 128, fixed seed and `ell`: unweighted `R` at sigma2 = 0.25, 1, 4 and the ratios "
                 "`R(1)/R(0.25)`, `R(4)/R(1)` (prediction P-B: the first ratio about 4).",
                 ["ell", "seed", "R(0.25)", "R(1)", "R(4)", "R(1)/R(0.25)", "R(4)/R(1)"], rows)


REINJ_KEYS = ["reinjection_D22", "reinjection_D33", "reinjection_D22_flux_weighted",
              "reinjection_D33_flux_weighted", "mean_tau", "mean_tau_flux_weighted"]


def t9_reinjection(by_id):
    rows = []
    for run in gaussian_runs(by_id):
        vals = [stat(run, WORKING_TOL, 1, k) if run["darcy_ok"] else None for k in REINJ_KEYS]
        rows.append([case_label((run["sigma2"], run["ell"])), run["seed"], run["n"], run["periods"]] + vals)
    t = table(9, "Re-injection estimate",
              "Lester 2023 eq. 36 applied to the one-period return map, `D_ii = var(d_i)/(2 <tau>)`, at the working "
              "tolerance: a property of the re-injection protocol applied to the return map, not an accepted "
              "macrodispersion coefficient. Columns: `D22`, `D33`, their flux-weighted versions, `<tau>`, "
              "flux-weighted `<tau>`.",
              ["case", "seed", "N", "periods", "D22", "D33", "D22 fw", "D33 fw", "<tau>", "<tau> fw"], rows)
    srows = []
    for case in CASES:
        per = []
        for seed in SEEDS_128:
            run = by_id.get(gauss_id("g", case, seed, 128))
            per.append([stat(run, WORKING_TOL, 1, k) if run and run["darcy_ok"] else None for k in REINJ_KEYS])
        row = [case_label(case)]
        for j in range(len(REINJ_KEYS)):
            v = [float(p[j]) for p in per if p[j] is not None]
            row += [statistics.fmean(v) if v else None, statistics.stdev(v) if len(v) >= 2 else None, len(v)]
        srows.append(row)
    hdr = ["case"]
    for k in ["D22", "D33", "D22 fw", "D33 fw", "<tau>", "<tau> fw"]:
        hdr += [k + " mean", k + " sd", "n"]
    t += "\n" + "\n".join(["Per case over the 128^3 realizations 3001..3005: mean, sample standard deviation, count.",
                           "", "| " + " | ".join(hdr) + " |", "|" + "|".join("---" for _ in hdr) + "|"]
                          + ["| " + " | ".join(fmt(v) for v in r) + " |" for r in srows] + [""])
    return t


def t10_manyperiod(by_id):
    rows = []
    for run in sorted((r for r in by_id.values() if r["field"] == "gaussian" and (r["periods"] or 1) > 1),
                      key=lambda r: r["id"]):
        if not run["darcy_ok"]:
            rows.append([run["id"], None] + [None] * 10)
            continue
        P = run["periods"]
        ns = []
        q = 1
        while q <= P:
            ns.append(q)
            q *= 2
        v2 = {n: stat(run, WORKING_TOL, n, "unweighted", "var_d2") for n in ns}
        v3 = {n: stat(run, WORKING_TOL, n, "unweighted", "var_d3") for n in ns}
        w = tol_entry(run, WORKING_TOL)
        e_last = _get(ladder_entry(w, P), "max_distance") if run["e_int"] is not None else None
        for n in ns:
            pe = period_entry(w, n)
            cnt = _get(pe, "count")
            cnb = _get(pe, "count_no_backflow")
            rows.append([run["id"], n, v2[n], v3[n], _get(pe, "unweighted", "R"),
                         ratio(v2[n], n * v2[1]) if v2[1] is not None else None,
                         ratio(v3[n], n * v3[1]) if v3[1] is not None else None,
                         order(v2.get(2 * n), v2[n]),  # log2(var(2n)/var(n))
                         order(v3.get(2 * n), v3[n]),
                         cnt, (cnt - cnb) if cnt is not None and cnb is not None else None,
                         e_last if n == ns[-1] else None])
    return table(10, "Many-period iteration",
                 "Runs with periods > 1, working tolerance 1e-8, for n = 1, 2, 4, ...: `var(d2)(n)`, `var(d3)(n)`, "
                 "unweighted `R(n)`, `var(n)/(n var(1))` per component, local exponent `log2(var(2n)/var(n))` per "
                 "component, number of `ok` streamlines used at n and of those with a backflow encounter, and on the "
                 "last row `e_int` at the last period (ladder `max_distance` 1e-8 vs 1e-12 at n = periods).",
                 ["run", "n", "var d2", "var d3", "R", "var d2/(n var d2(1))", "var d3/(n var d3(1))",
                  "exponent d2", "exponent d3", "ok used", "ok with backflow", "e_int at last period"], rows)


def t11_tol_ladder(by_id):
    rows = []
    sel = [r for r in by_id.values() if r["field"] == "gaussian" and
           ((r["n"] == N_FINE and r["periods"] == 1) or (r["periods"] or 1) > 1)]
    for run in sorted(sel, key=lambda r: r["id"]):
        if not run["darcy_ok"]:
            rows.append([run["id"], None, None, None, None, None, None])
            continue
        for t in sorted(run["tolerances"] or [], key=lambda t: -float(t["tol"])):
            P = run["periods"]
            rows.append([run["id"], float(t["tol"]), _get(period_entry(t, 1), "unweighted", "R"),
                         _get(t, "ladder_vs_tightest", "tightest_tol"), _get(ladder_entry(t, 1), "max_distance"),
                         _get(ladder_entry(t, 1), "rms_distance"),
                         _get(ladder_entry(t, P), "max_distance") if P > 1 else None])
    return table(11, "Tolerance ladder",
                 "Gaussian runs at 256^3 and many-period runs: unweighted `R` at period 1 for each integrator "
                 "tolerance, the tightest tolerance of the ladder, and the ladder `max_distance` / `rms_distance` "
                 "to it at period 1 (and `max_distance` at the last period for many-period runs).",
                 ["run", "tol", "R", "tightest", "ladder max (n=1)", "ladder rms (n=1)", "ladder max (n=periods)"],
                 rows)


def t12_inventory(runs, by_id, duplicates, errors):
    kinds = {}
    for r in runs:
        if r["id"] is None:
            k = "unrecognized"
        elif r["id"].startswith("a_"):
            k = "analytic, 16 probe seeds" if r["id"].endswith("_p16") else "analytic, 1024 seeds"
        elif r["id"].startswith("m_"):
            k = "gaussian2d (matched control)"
        elif r["id"].startswith("p_"):
            k = "gaussian, many-period"
        else:
            k = "gaussian, one period"
        kinds[k] = kinds.get(k, 0) + 1
    lines = ["### 12. Run inventory", "",
             "Counts of `summary.json` files found per kind; expected runs (the 103 of `run_matrix.sh`) that are "
             "missing, have `darcy_converged == false`, deviate from the expected configuration, or are invalid for "
             "classification; unrecognized, duplicated and unreadable files.", "",
             "| kind | runs found |", "|---|---|"]
    for k in sorted(kinds):
        lines.append("| %s | %d |" % (k, kinds[k]))
    lines.append("| total | %d |" % len(runs))
    lines.append("")
    lines.append("| group | expected | found |")
    lines.append("|---|---|---|")
    for g in GROUPS:
        ids = [e["id"] for e in EXPECTED if e["group"] == g]
        lines.append("| %s | %d | %d |" % (g, len(ids), sum(1 for i in ids if i in by_id)))
    lines.append("| all | %d | %d |" % (len(EXPECTED), sum(1 for e in EXPECTED if e["id"] in by_id)))
    lines.append("")
    missing = [e for e in EXPECTED if e["id"] not in by_id]
    lines.append("Missing expected runs (%d):" % len(missing))
    lines += ["- `%s` (%s)" % (e["id"], e["group"]) for e in missing] or ["- none"]
    lines.append("")
    nc = sorted(r["id"] for r in by_id.values() if not r["darcy_ok"])
    lines.append("Runs with `darcy_converged == false` (%d):" % len(nc))
    lines += ["- `%s`" % i for i in nc] or ["- none"]
    lines.append("")
    dev = sorted((r["id"], d) for r in by_id.values() for d in r["deviations"])
    lines.append("Configuration deviations from the expected command (%d):" % len(dev))
    lines += ["- `%s`: %s" % (i, d) for i, d in dev] or ["- none"]
    lines.append("")
    inv = sorted((r["id"], w) for r in by_id.values() if r["darcy_ok"] for w in r["invalid_R"] + r["invalid_eint"])
    lines.append("Runs invalid for classification (darcy converged; quantity missing) (%d):" % len(inv))
    lines += ["- `%s`: %s" % (i, w) for i, w in inv] or ["- none"]
    lines.append("")
    unrec = sorted(r["path"] for r in runs if r["id"] is None)
    lines.append("Unrecognized summaries (%d):" % len(unrec))
    lines += ["- `%s`" % p for p in unrec] or ["- none"]
    lines.append("")
    lines.append("Duplicated identities, not used (%d):" % len(duplicates))
    lines += ["- `%s`: %s" % (i, ", ".join(sorted(duplicates[i]))) for i in sorted(duplicates)] or ["- none"]
    lines.append("")
    lines.append("Unreadable files (%d):" % len(errors))
    lines += ["- `%s`: %s" % e for e in sorted(errors)] or ["- none"]
    lines.append("")
    return "\n".join(lines), [e["id"] for e in missing]


# -------------------------------------------------------------------------------------
# Driver
# -------------------------------------------------------------------------------------

def thresholds():
    return {"REL_CHANGE_MAX": REL_CHANGE_MAX, "CONTROL_FACTOR": CONTROL_FACTOR,
            "INTEGRATOR_REL_MAX": INTEGRATOR_REL_MAX, "NON_OK_FRACTION_MAX": NON_OK_FRACTION_MAX,
            "source": "SF-30 bitacora, 2026-10-05 (D-5, D-11, D-12)"}


def analyze(raw_dir):
    runs, by_id, duplicates, errors = load_all(raw_dir)
    cases = classify_cases(by_id, "ctrl")
    cases_lit = classify_cases(by_id, "lit")
    parts = ["# SF-30 closure gate: analysis of `%s`" % raw_dir, "",
             "Decision thresholds (pre-registered): rel change < %g, factor %g, integrator %g x R, non-ok <= %g."
             % (REL_CHANGE_MAX, CONTROL_FACTOR, INTEGRATOR_REL_MAX, NON_OK_FRACTION_MAX), ""]
    parts.append(t1_probes(by_id))
    parts.append(t2_controls1024(by_id))
    parts.append(t3_matrix(by_id))
    parts.append(t4_matched(by_id))
    parts.append(t5_ladders(by_id))
    parts.append(classification_tables(
        6, "Classification",
        "Pre-registered rule (bitacora D-5, D-11, D-12) per case and realization, N_f = 256, N_c = 128: every "
        "criterion's value and truth, the class, and the reason when ambiguous or incomplete. `E_ctrl(N)` = max(R of "
        "the matched `gaussian2d` run of the same seed, R of `lester2021` 1024 seeds) at N.", cases, "E_ctrl"))
    parts.append(classification_tables(
        7, "Spec-literal reading, for information only",
        "NOT the pre-registered rule: the same rule with `E_ctrl` replaced by `E_lit(N)` = max(R of analytic "
        "`control2d` 1024 seeds, R of `lester2021` 1024 seeds) at N; validity uses the case runs and the "
        "`control2d` runs at N_f and N_c.", cases_lit, "E_lit"))
    parts.append(t8_amplitude(by_id))
    parts.append(t9_reinjection(by_id))
    parts.append(t10_manyperiod(by_id))
    parts.append(t11_tol_ladder(by_id))
    inv, missing = t12_inventory(runs, by_id, duplicates, errors)
    parts.append(inv)
    tables = "\n".join(parts)

    def compact(r):
        return {"id": r["id"], "group": r["group"], "path": r["path"], "field": r["field"], "n": r["n"],
                "sigma2": r["sigma2"], "ell": r["ell"], "seed": r["seed"], "eps": r["eps"], "periods": r["periods"],
                "seeds": r["seeds_kind"], "tols": r["tols"], "working_tol": r["working_tol"],
                "darcy_ok": r["darcy_ok"], "R": r["R"], "e_int": r["e_int"], "non_ok": r["non_ok"],
                "invalid_R": r["invalid_R"], "invalid_e_int": r["invalid_eint"], "deviations": r["deviations"]}

    data = {"thresholds": thresholds(),
            "runs": [compact(r) for r in sorted(runs, key=lambda r: (r["id"] or "~", r["path"]))],
            "cases": cases, "cases_spec_literal": cases_lit, "missing": missing,
            "duplicates": {k: sorted(v) for k, v in sorted(duplicates.items())},
            "unreadable": [list(e) for e in sorted(errors)]}
    return tables, data


def _json_safe(x):
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return str(x)
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


# -------------------------------------------------------------------------------------
# Self-test
# -------------------------------------------------------------------------------------

SAMPLE = ("/home/sesquerre/Projects/MacroFlow3D/.claude/orchestration/SF-30-streamline-closure-gate/"
          "sample_summary_gaussian64.json")


def synth_summary(field, n, sigma2=None, ell=None, seed=None, eps=None, seeds="1024", periods=1, R=1e-2,
                  e_int=1e-9, non_ok=0.0, darcy=True, tols=None, working_tol=WORKING_TOL, pcg_rtol=None):
    """Minimal summary.json in the real schema's structure (only keys the loader and tables read)."""
    tols = sorted(tols or GAUSS_TOLS, reverse=True)
    gaussian_kind = field in ("gaussian", "gaussian2d")
    if pcg_rtol is None:
        pcg_rtol = 1e-10 if gaussian_kind else 1e-12
    cfg = {"field": field, "n": n, "sigma2": sigma2, "ell": ell, "seed": seed, "eps": eps,
           "seeds_source": "file" if seeds == "p16" else "generated", "n_seeds": 16 if seeds == "p16" else 1024,
           "seed_rng": None if seeds == "p16" else SEED_RNG_DEFAULT,
           "seeds_file": "scripts/" + PROBE_SEEDS_FILE if seeds == "p16" else None,
           "tols": tols, "working_tol": working_tol, "periods": periods, "pcg_rtol": pcg_rtol}
    s = {"schema_version": "sf30-closure-gate-1", "configuration": cfg, "field": {"Y_variance": sigma2},
         "darcy_converged": darcy,
         "darcy": {"converged": darcy, "corrector_results": [{"iterations": 20}, {"iterations": 21},
                                                             {"iterations": 22}]}}
    if not darcy:
        s.update({"direction_field_diagnostics": None, "tolerances": None, "round_trip": None})
        return s
    s["direction_field_diagnostics"] = {"min_g1": 0.1, "backflow_volume_fraction": 0.0}
    seeds_n = cfg["n_seeds"]
    nok = int(round(non_ok * seeds_n))
    ents = []
    for t in tols:
        pers = []
        for k in range(1, periods + 1):
            pers.append({
                "n": k,
                "counts": {"seeds": seeds_n, "ok": seeds_n - nok, "seed_backflow": 0,
                           "ok_with_backflow_encounter": 0, "domain": seeds_n, "non_ok_fraction": non_ok},
                "count": seeds_n - nok,
                "unweighted": {"mean": [0.0, 0.0], "R": R * math.sqrt(k), "max_abs_d": 3 * R, "var_d2": R * R / 2 * k,
                               "var_d3": R * R / 2 * k, "max_abs_d3": 2 * R},
                "flux_weighted": {"mean": [0.0, 0.0], "R": R},
                "count_no_backflow": seeds_n - nok, "R_no_backflow": R,
                "mean_tau": 1.0, "mean_tau_flux_weighted": 1.0,
                "reinjection_D22": R * R / 4, "reinjection_D33": R * R / 4,
                "reinjection_D22_flux_weighted": R * R / 4, "reinjection_D33_flux_weighted": R * R / 4,
            })
        lad = None
        if t != tols[-1]:
            lad = {"tightest_tol": tols[-1],
                   "periods": [{"n": 1, "max_distance": e_int, "rms_distance": e_int / 2}]
                   + ([{"n": periods, "max_distance": e_int, "rms_distance": e_int / 2}] if periods > 1 else [])}
        ents.append({"tol": t, "periods": pers, "ladder_vs_tightest": lad})
    s["tolerances"] = ents
    s["round_trip"] = {"max_distance": 1e-9}
    return s


class Tree:
    """Synthetic raw directory; directory names are deliberately uninformative."""

    def __init__(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="sf30_selftest_")
        self.count = 0

    def add(self, summary):
        d = os.path.join(self.tmp.name, "x%03d" % self.count, "deep")
        self.count += 1
        os.makedirs(d)
        with open(os.path.join(d, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f)

    def classify(self):
        _, by_id, _, _ = load_all(self.tmp.name)
        return {c["case"]: c for c in classify_cases(by_id, "ctrl")}

    def close(self):
        self.tmp.cleanup()


def build_case(tree, case, R_f=1e-2, R_c=1.02e-2, R_reals=None, E=1e-4, R_lester=1e-10, seeds=(SEED_MAIN,),
               over=None, skip=()):
    """Write the runs one case needs. `over` maps run id -> synth_summary overrides; `skip` lists ids not written."""
    over = over or {}
    R_reals = R_reals if R_reals is not None else {s: R_c for s in SEEDS_128}
    specs = {}
    for n in (128, 256):
        specs["a_lester2021_n%d" % n] = dict(field="lester2021", n=n, eps=1.0, R=R_lester)
    for seed in seeds:
        specs[gauss_id("g", case, seed, 256)] = dict(field="gaussian", n=256, sigma2=case[0], ell=case[1], seed=seed,
                                                     R=R_f)
        for n in (128, 256):
            specs[gauss_id("m", case, seed, n)] = dict(field="gaussian2d", n=n, sigma2=case[0], ell=case[1], seed=seed,
                                                       R=E)
    for seed in sorted(set(SEEDS_128) | set(seeds)):
        specs[gauss_id("g", case, seed, 128)] = dict(field="gaussian", n=128, sigma2=case[0], ell=case[1], seed=seed,
                                                     R=R_reals.get(seed, R_c))
    for rid, kw in sorted(specs.items()):
        if rid in skip:
            continue
        kw = dict(kw)
        kw.update(over.get(rid, {}))
        tree.add(synth_summary(**kw))


def self_test():
    results = []

    def check(name, cond, detail=""):
        results.append(cond)
        print("%s %s%s" % ("PASS" if cond else "FAIL", name, (" -- " + detail) if detail else ""))

    case = (1.0, 0.125)
    lab = case_label(case)
    f_id = gauss_id("g", case, SEED_MAIN, 256)

    def run_scenario(**kw):
        t = Tree()
        try:
            build_case(t, case, **kw)
            return t.classify()[lab]
        finally:
            t.close()

    c = run_scenario()
    check("1 does_not_close: stable R far above control, five realizations above", c["class"] == "does_not_close",
          "%s %s" % (c["class"], c["reason"]))
    c = run_scenario(R_f=5e-4, R_c=2e-3)
    check("2 closes: R falls with the grid, within the control factor", c["class"] == "closes",
          "%s %s" % (c["class"], c["reason"]))
    c = run_scenario(R_f=1e-2, R_c=0.85e-2)
    check("3 ambiguous: R changes by 15 percent between 128 and 256",
          c["class"] == "ambiguous" and "relative change" in c["reason"], "%s %s" % (c["class"], c["reason"]))
    c = run_scenario(R_f=5e-4, R_c=4e-4)
    check("4 ambiguous: R(256) = 5x control but increased from 128",
          c["class"] == "ambiguous" and "R(N_f) is not < R(N_c)" in c["reason"], "%s %s" % (c["class"], c["reason"]))
    c = run_scenario(over={f_id: {"darcy": False}})
    check("5 ambiguous by validity: Darcy not converged on the 256 run",
          c["class"] == "ambiguous" and "Darcy not converged on %s" % f_id in c["reason"],
          "%s %s" % (c["class"], c["reason"]))
    c = run_scenario(over={f_id: {"e_int": 0.05 * 1e-2}})
    check("6 ambiguous by validity: e_int = 0.05 R and above E_ctrl",
          c["class"] == "ambiguous" and "e_int" in c["reason"], "%s %s" % (c["class"], c["reason"]))
    c = run_scenario(over={f_id: {"non_ok": 0.02}})
    check("7 ambiguous by validity: non_ok_fraction = 0.02",
          c["class"] == "ambiguous" and "non_ok_fraction" in c["reason"], "%s %s" % (c["class"], c["reason"]))
    reals = {s: 1.02e-2 for s in SEEDS_128}
    reals[3004] = 8e-4  # 8x the control 1e-4
    c = run_scenario(R_reals=reals)
    check("8 does_not_close blocked by one 128^3 realization at 8x the control -> ambiguous",
          c["class"] == "ambiguous" and gauss_id("g", case, 3004, 128) in c["reason"],
          "%s %s" % (c["class"], c["reason"]))

    # 9: (4, 0.0625) on three realizations
    s4 = CASE_S4_L16
    t = Tree()
    try:
        build_case(t, s4, seeds=SEEDS_S4_L16)
        c = t.classify()[case_label(s4)]
    finally:
        t.close()
    check("9a (4, 0.0625): three seeds agree -> class", c["class"] == "does_not_close" and
          [e["class"] for e in c["realizations"]] == ["does_not_close"] * 3, "%s %s" % (c["class"], c["reason"]))
    t = Tree()
    try:
        build_case(t, s4, seeds=SEEDS_S4_L16,
                   over={gauss_id("g", s4, 3003, 256): {"R": 5e-4}, gauss_id("g", s4, 3003, 128): {"R": 2e-3}})
        c = t.classify()[case_label(s4)]
    finally:
        t.close()
    check("9b (4, 0.0625): one seed disagreeing -> ambiguous", c["class"] == "ambiguous" and
          [e["class"] for e in c["realizations"]] == ["does_not_close", "does_not_close", "closes"],
          "%s %s" % (c["class"], c["reason"]))

    c = run_scenario(skip=(f_id,))
    check("10a missing 256 run -> incomplete", c["class"] == "incomplete" and f_id in c["reason"],
          "%s %s" % (c["class"], c["reason"]))
    r5 = gauss_id("g", case, 3005, 128)
    c = run_scenario(skip=(r5,))
    check("10b only four 128^3 realizations -> incomplete", c["class"] == "incomplete" and r5 in c["reason"],
          "%s %s" % (c["class"], c["reason"]))
    c = run_scenario(over={f_id: {"tols": [1e-6, 1e-8, 1e-10]}})
    check("11 tightest tolerance not 1e-12 -> invalid for classification",
          c["class"] == "incomplete" and "invalid for classification" in c["reason"] and "tightest" in c["reason"],
          "%s %s" % (c["class"], c["reason"]))

    # 12: the real sample
    ok = False
    detail = "sample not found: " + SAMPLE
    if os.path.isfile(SAMPLE):
        with open(SAMPLE, "r", encoding="utf-8") as f:
            raw = json.load(f)
        ref = None
        for tentry in raw["tolerances"]:
            if tentry["tol"] == 1e-8:
                ref = tentry["periods"][0]["unweighted"]["R"]
        run = load_run(SAMPLE)
        ok = (run["id"] == "g_s1_l8_r3001_n64" and run["R"] is not None and run["R"] == ref
              and "%.6f" % run["R"] == "0.053248")
        detail = "id=%s R=%r (file %r); e_int invalid: %s" % (run["id"], run["R"], ref, run["invalid_eint"])
    check("12 loader parses the real sample and extracts R at the working tolerance", ok, detail)

    # Extra: expected list mirrors run_matrix.sh --list
    ids = [e["id"] for e in EXPECTED]
    per = {g: sum(1 for e in EXPECTED if e["group"] == g) for g in GROUPS}
    check("13a expected list: 103 unique runs, groups 30/22/30/14/7",
          len(ids) == 103 and len(set(ids)) == 103 and [per[g] for g in GROUPS] == [30, 22, 30, 14, 7], str(per))
    sh = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_matrix.sh")
    try:
        out = subprocess.run(["bash", sh, "all", "--list"], capture_output=True, text=True, check=True).stdout
        listed = [(ln.split()[0], ln.split()[1]) for ln in out.splitlines() if ln.strip()]
        check("13b run_matrix.sh all --list matches the analysis' expected list (ids, groups, order)",
              listed == [(e["group"], e["id"]) for e in EXPECTED], "%d lines" % len(listed))
    except (OSError, subprocess.CalledProcessError) as exc:
        check("13b run_matrix.sh all --list matches the analysis' expected list", False, str(exc))

    n_fail = results.count(False)
    print("self-test: %d checks, %d failed" % (len(results), n_fail))
    return 0 if n_fail == 0 else 1


def main(argv):
    if len(argv) == 2 and argv[1] == "--self-test":
        return self_test()
    args = argv[1:]
    raw_dir = tables_path = json_path = None
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("--tables", "--json"):
            if i + 1 >= len(args):
                print(__doc__, file=sys.stderr)
                return 2
            if a == "--tables":
                tables_path = args[i + 1]
            else:
                json_path = args[i + 1]
            i += 2
        elif a.startswith("--") or raw_dir is not None:
            print(__doc__, file=sys.stderr)
            return 2
        else:
            raw_dir = a
            i += 1
    if raw_dir is None or not os.path.isdir(raw_dir):
        print(__doc__, file=sys.stderr)
        return 2
    tables, data = analyze(raw_dir)
    if tables_path:
        with open(tables_path, "w", encoding="utf-8") as f:
            f.write(tables)
    else:
        sys.stdout.write(tables)
    if json_path:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(_json_safe(data), f, indent=1, sort_keys=True)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
