#!/usr/bin/env python3
"""SF-33 N5: compare `inlet_slab --proto` GPU logs with the SF-29 CPU prototype (claim (a), step 9a).

For every GPU log (one case each) it prints, next to the prototype's values:
  e_v, e_psi, e_psi1, e_psi2, e_i1, e_i2, e_div, min_c, p0.1, p1, p5, p50 of cand=i1o4 and of the ceiling
  cand=oracle_fd4, r_F, its, the continuation PATH (equality), relative differences and PASS/FAIL at 1e-6,
  the FIELDDIFF values (field-wise comparison with the saved 16^3 prototype solution), the r_F history agreement
  (correct digits while r_F > 1e-10) and the GMRES statistics per Newton stage (max / median iterations: the
  preconditioner gate report).  A Markdown table is written with --out.

Prototype values, best available source first:
  1. full precision: <exports>/solutions/<field>_<eps>_<N>_i1o4/solution.json (`metrics_json`, `hist`; the saved
     16^3 solutions) for cand=i1o4 and oracle_fd4; <exports>/<field>_<eps>_<N>/ref_metrics.json for oracle_fd4;
  2. 4 significant digits: the SF-29 sweep2 cell log `i1o4-<field>-<eps>-N<N>.txt` (CASE / PATH / HISTORY lines)
     next to summary.md, else the `i1o4` table of summary.md (e_v, e_psi, e_psi1, e_psi2, min_c, r_F, its) and its
     `- PATH N=..` lines.
GPU values: full precision from the driver's JSON summary (the `SUMMARY_JSON <path>` line of the log) when that
file exists, else the 4-digit CASE line.

PASS rule (threshold 1e-6 relative, SF-33 acceptance (a)):
  full-precision reference: |gpu - proto| / |proto| <= 1e-6;
  4-significant-digit reference (printed %.3e): the pass band is the rounding bound of the printed value,
  |gpu - proto| <= 0.5e-3 * 10**floor(log10|proto|) (+ the same bound of the GPU value when it is itself a
  printed %.3e value); a 1e-6 relative agreement cannot be resolved from 4 digits and is NOT claimed then.
Values that are roundoff-sized on both sides (oracle e_psi*, control2d e_i2 / e_psi2 / e_div below 1e-10) are
reported as `roundoff` (not gated).

numpy only (numpy 1.26 compatible); SF-29 `metrics.py` is imported read-only (parse_case_line).

Usage:
  python3 compare_proto.py LOG [LOG ...] [--summary SF29/raw/sweep2/summary.md] [--exports ../exports]
                           [--out table.md]
  python3 compare_proto.py --crosscheck LOG [LOG ...] [--out table.md]   (SF-19 cross-check table + orders)
"""
import argparse
import json
import math
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
SF29_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "2026-10-02-sf29-inlet-labels"))
SF29_SCRIPTS = os.path.join(SF29_DIR, "scripts")
if SF29_SCRIPTS not in sys.path:
    sys.path.insert(0, SF29_SCRIPTS)
import metrics as M  # noqa: E402  (read-only reuse of the SF-29 artifact)

THRESH = 1e-6
ROUNDOFF = 1e-10
KEYS = ["e_v", "e_psi", "e_psi1", "e_psi2", "e_i1", "e_i2", "e_div", "min_c", "p0.1", "p1", "p5", "p50"]


# --------------------------------------------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------------------------------------------
def case_dict(line):
    """metrics.parse_case_line + e_i tuple split into e_i1 / e_i2."""
    d = M.parse_case_line(line)
    if isinstance(d.get("e_i"), tuple):
        d["e_i1"], d["e_i2"] = d["e_i"]
    return d


def parse_gpu_log(path):
    g = {"log": path, "case": {}, "stages": [], "path": None, "status": None, "fielddiff": None,
         "hist_full": None, "json": None, "meta": {}}
    cur = None
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("CASE_DIR "):
                for tok in line.split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        g["meta"][k] = v
            elif line.startswith("CASE "):
                d = case_dict(line)
                g["case"][d["cand"]] = d
                g["meta"].setdefault("field", d["field"])
                g["meta"].setdefault("eps_case", d["eps"])
                g["meta"].setdefault("N_case", d["N"])
            elif line.startswith("STAGE field="):
                m = re.match(r"STAGE field=(\S+) eps=(\S+) N=(\d+)", line)
                cur = {"eps": float(m.group(2)), "final": "final attempt" in line, "lin_its": [],
                       "lin_status": [], "lin_rel": [], "status": None}
                g["stages"].append(cur)
            elif line.lstrip().startswith("LINEAR gmres") and cur is not None:
                m = re.search(r"its=(\d+) rel=(\S+) .* status=(\S+)", line)
                cur["lin_its"].append(int(m.group(1)))
                cur["lin_rel"].append(float(m.group(2)))
                cur["lin_status"].append(m.group(3))
            elif line.lstrip().startswith("STAGE_END") and cur is not None:
                m = re.search(r"status=(\S+) its=(\d+)", line)
                cur["status"] = m.group(1)
                cur["its"] = int(m.group(2))
                cur["accepted"] = line.rstrip().endswith("accepted")
            elif line.startswith("PATH field="):
                g["path"] = line.split("continuation path:", 1)[1].strip()
            elif line.startswith("FIELDDIFF "):
                fd = {}
                for tok in line.split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        try:
                            fd[k] = float(v)
                        except ValueError:
                            pass
                g["fielddiff"] = fd
            elif line.startswith("HISTORY_FULL ") and " r_F: " in line:
                g["hist_full"] = [float(x) for x in line.split(" r_F: ", 1)[1].split()]
            elif line.startswith("STATUS "):
                g["status"] = line.split()[1]
            elif line.startswith("SUMMARY_JSON "):
                p = line.split(" ", 1)[1].strip()
                if os.path.exists(p):
                    with open(p) as jf:
                        g["json"] = json.load(jf)
    return g


def parse_summary(path):
    """{(field, eps, N): {...table row...}} and {(field, eps, N): PATH} for candidate i1o4."""
    rows, paths = {}, {}
    if not path or not os.path.exists(path):
        return rows, paths
    field = eps = None
    in_i1o4 = False
    header = None
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            m = re.match(r"## (\S+), eps = (\S+)$", line)
            if m:
                field, eps = m.group(1), float(m.group(2))
                in_i1o4 = False
                continue
            if line.startswith("### "):
                in_i1o4 = line.startswith("### candidate `i1o4` ")
                header = None
                continue
            if not in_i1o4 or field is None:
                continue
            if line.startswith("| N |"):
                header = [c.strip() for c in line.strip("|").split("|")]
                continue
            if header and line.startswith("|") and not line.startswith("|---"):
                cells = [c.strip() for c in line.strip("|").split("|")]
                if len(cells) != len(header):
                    continue
                r = dict(zip(header, cells))
                try:
                    N = int(r["N"])
                except ValueError:
                    continue
                row = {"solver_status": r.get("solver status")}
                for k_src, k in (("r_F", "r_F"), ("its", "its"), ("e_v", "e_v"), ("ceiling", "ceiling_e_v"),
                                 ("e_psi", "e_psi"), ("e_psi1", "e_psi1"), ("e_psi2", "e_psi2"),
                                 ("min_c", "min_c")):
                    try:
                        row[k] = float(r[k_src])
                    except (KeyError, ValueError):
                        pass
                rows[(field, eps, N)] = row
            m = re.match(r"- PATH N=(\d+): `(.*)`$", line)
            if m:
                paths[(field, eps, int(m.group(1)))] = m.group(2)
    return rows, paths


def parse_cell_log(sweep_dir, field, eps, N):
    p = os.path.join(sweep_dir, "i1o4-%s-%g-N%d.txt" % (field, eps, N))
    if not os.path.exists(p):
        return None
    out = {"file": p, "case": {}, "path": None, "hist": None}
    with open(p) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("CASE "):
                d = case_dict(line)
                out["case"][d["cand"]] = d
            elif line.startswith("PATH field="):
                out["path"] = line.split("continuation path:", 1)[1].strip()
            elif line.startswith("HISTORY ") and " r_F: " in line:
                out["hist"] = [float(x) for x in line.split(" r_F: ", 1)[1].split()]
    return out


# --------------------------------------------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------------------------------------------
def print_band(v):
    """Rounding bound of a %.3e printed value."""
    if v is None or not np.isfinite(v) or v == 0.0:
        return 0.0
    return 0.5e-3 * 10.0 ** math.floor(math.log10(abs(v)))


def compare_value(g, p, g_full, p_full):
    """-> (rel, verdict).  g_full / p_full: whether the values carry full precision."""
    if g is None or p is None or not np.isfinite(g) or not np.isfinite(p):
        return float("nan"), "n/a"
    if abs(g) < ROUNDOFF and abs(p) < ROUNDOFF:
        return float("nan"), "roundoff"
    rel = abs(g - p) / abs(p) if p != 0 else abs(g - p)
    if g_full and p_full:
        return rel, "PASS" if rel <= THRESH else "FAIL"
    band = (0.0 if p_full else print_band(p)) + (0.0 if g_full else print_band(g))
    return rel, "PASS(4dig)" if abs(g - p) <= band else "FAIL(4dig)"


def gpu_metrics(g, cand):
    if g["json"] and "metrics" in g["json"] and cand in g["json"]["metrics"]:
        return g["json"]["metrics"][cand], True
    c = g["case"].get(cand)
    return (c or {}), False


def proto_metrics(case_key, cand, exports, sweep_dir, rows):
    field, eps, N = case_key
    name = "%s_%g_%d" % (field, eps, N)
    sol = os.path.join(exports, "solutions", name + "_i1o4", "solution.json") if exports else None
    if sol and os.path.exists(sol):
        with open(sol) as f:
            sj = json.load(f)
        if cand in sj.get("metrics_json", {}):
            return sj["metrics_json"][cand], True, "solution.json"
    if cand == "oracle_fd4" and exports:
        rm = os.path.join(exports, name, "ref_metrics.json")
        if os.path.exists(rm):
            with open(rm) as f:
                return json.load(f)["oracle_fd4"], True, "ref_metrics.json"
    cl = parse_cell_log(sweep_dir, field, eps, N) if sweep_dir else None
    if cl and cand in cl["case"]:
        return cl["case"][cand], False, os.path.basename(cl["file"])
    row = rows.get(case_key)
    if row:
        if cand == "i1o4":
            return {k: row[k] for k in ("e_v", "e_psi", "e_psi1", "e_psi2", "min_c", "r_F", "its") if k in row}, \
                False, "summary.md"
        if cand == "oracle_fd4" and "ceiling_e_v" in row:
            return {"e_v": row["ceiling_e_v"]}, False, "summary.md"
    return {}, False, "none"


def fmt(x, f="%.6e"):
    if x is None:
        return "-"
    try:
        if not np.isfinite(x):
            return "nan"
    except TypeError:
        return str(x)
    return f % x


def compare_case(g, summary_rows, summary_paths, exports, sweep_dir):
    field = g["meta"].get("field")
    eps = float(g["meta"].get("eps_case", g["meta"].get("eps", "nan")))
    N = int(g["meta"].get("N_case", g["meta"].get("N", 0)))
    key = (field, eps, N)
    res = {"key": key, "rows": [], "verdicts": []}
    print("=" * 110)
    print("CASE %s eps=%g N=%d   (GPU log %s, GPU STATUS %s)" % (field, eps, N, g["log"], g["status"]))
    for cand in ("i1o4", "oracle_fd4"):
        gm, gfull = gpu_metrics(g, cand)
        pm, pfull, src = proto_metrics(key, cand, exports, sweep_dir, summary_rows)
        print("  cand=%s  gpu: %s | prototype: %s (%s)" % (cand, "full precision (JSON)" if gfull else "CASE line %.3e",
                                                          src, "full precision" if pfull else "4 significant digits"))
        print("    %-7s %-24s %-24s %-11s %s" % ("key", "gpu", "prototype", "rel diff", "verdict"))
        for k in KEYS:
            gv, pv = gm.get(k), pm.get(k)
            if gv is None and pv is None:
                continue
            rel, verdict = compare_value(gv, pv, gfull, pfull)
            print("    %-7s %-24s %-24s %-11s %s" % (k, fmt(gv, "%.17g"), fmt(pv, "%.17g"), fmt(rel, "%.2e"), verdict))
            res["rows"].append((cand, k, gv, pv, rel, verdict))
            if verdict not in ("n/a", "roundoff"):
                res["verdicts"].append(verdict.startswith("PASS"))
    # r_F, its
    gi = g["case"].get("i1o4", {})
    cl = parse_cell_log(sweep_dir, field, eps, N) if sweep_dir else None
    p_rF = summary_rows.get(key, {}).get("r_F")
    p_its = summary_rows.get(key, {}).get("its")
    if cl and "i1o4" in cl["case"]:
        p_rF = cl["case"]["i1o4"].get("r_F", p_rF)
        p_its = cl["case"]["i1o4"].get("its", p_its)
    rF = gi.get("r_F")
    res["r_F"] = rF
    rF_ok = rF is not None and np.isfinite(rF) and rF <= 1e-10
    res["verdicts"].append(rF_ok)
    print("  r_F gpu=%s (gate <= 1e-10: %s) prototype=%s | its gpu=%s prototype=%s" % (
        fmt(rF, "%.3e"), "PASS" if rF_ok else "FAIL", fmt(p_rF, "%.3e"), fmt(gi.get("its"), "%d"),
        fmt(p_its, "%d")))
    # PATH
    p_path = summary_paths.get(key) or (cl["path"] if cl else None)
    same = p_path is not None and g["path"] == p_path
    res["path"] = (g["path"], p_path, same)
    print("  PATH gpu=`%s` prototype=`%s` -> %s" % (g["path"], p_path, "EQUAL" if same else
                                                  ("DIFFERENT" if p_path else "no prototype PATH")))
    # FIELDDIFF
    fd = g["fielddiff"]
    if fd:
        j = fd.get("max_rel_diff_joint")
        if j is None and "max_rel_diff_u1" in fd:   # logs without the joint value: per-field maximum
            j = max(fd.get("max_rel_diff_u1"), fd.get("max_rel_diff_u2", 0.0))
        print("  FIELDDIFF max_rel_diff_u1=%s max_rel_diff_u2=%s max_rel_diff_joint=%s max_abs_diff=(%s, %s) "
              "-> %s (expected <= 1e-8 relative; joint = max abs diff / max|u_proto| over both fields)"
              % (fmt(fd.get("max_rel_diff_u1"), "%.3e"), fmt(fd.get("max_rel_diff_u2"), "%.3e"), fmt(j, "%.3e"),
                 fmt(fd.get("max_abs_diff_u1"), "%.3e"), fmt(fd.get("max_abs_diff_u2"), "%.3e"),
                 "PASS" if j is not None and j <= 1e-8 else "FAIL"))
        res["fielddiff"] = j
    else:
        res["fielddiff"] = None
        print("  FIELDDIFF: none (no --solution for this case)")
    # r_F history
    sol = os.path.join(exports, "solutions", "%s_%g_%d_i1o4" % key, "solution.json") if exports else None
    res["hist_digits"] = None
    if sol and os.path.exists(sol) and g["hist_full"]:
        with open(sol) as f:
            ph = [h[0] for h in json.load(f)["hist"]]
        gh = g["hist_full"]
        print("  r_F history (gpu, full precision):       %s" % " ".join("%.6e" % x for x in gh))
        print("  r_F history (prototype hist, solution.json): %s" % " ".join("%.6e" % x for x in ph))
        digs = []
        for a, b in zip(gh, ph):
            if b > 1e-10:
                d = abs(a - b) / abs(b)
                digs.append(-math.log10(d) if d > 0 else 17.0)
        res["hist_digits"] = (min(digs) if digs else float("nan"), len(gh), len(ph))
        print("  r_F history agreement while r_F > 1e-10: min correct digits %.1f over %d entries; lengths gpu=%d "
              "prototype=%d" % (res["hist_digits"][0], len(digs), len(gh), len(ph)))
    elif cl and cl["hist"] and g["hist_full"]:
        print("  r_F history (gpu):       %s" % " ".join("%.2e" % x for x in g["hist_full"]))
        print("  r_F history (prototype, %%.2e cell log): %s" % " ".join("%.2e" % x for x in cl["hist"]))
    # GMRES statistics per stage
    res["gmres"] = []
    for st in g["stages"]:
        its = st["lin_its"]
        if its:
            mx, med = max(its), float(np.median(its))
        else:
            mx, med = 0, float("nan")
        nfail = sum(1 for s in st["lin_status"] if s != "converged")
        res["gmres"].append((st["eps"], st.get("status"), len(its), mx, med, nfail))
        print("  GMRES stage eps=%g%s status=%s newton_steps=%d its_max=%d its_median=%.1f linear_failures=%d"
              % (st["eps"], "(final)" if st["final"] else "", st.get("status"), len(its), mx, med, nfail))
    res["pass"] = all(res["verdicts"]) and (same or p_path is None) and \
        (res["fielddiff"] is None or res["fielddiff"] <= 1e-8)
    print("  OVERALL %s" % ("PASS" if res["pass"] else "FAIL"))
    return res


def markdown(results):
    out = ["| case | status | r_F | PATH equal | e_v gpu | e_v proto | e_v verdict | e_psi gpu | e_psi proto | "
           "e_psi verdict | ceiling e_v verdict | FIELDDIFF joint | r_F hist digits | GMRES max / median (target) "
           "| overall |", "|" + "---|" * 15]
    for r in results:
        field, eps, N = r["key"]
        def pick(cand, k):
            for row in r["rows"]:
                if row[0] == cand and row[1] == k:
                    return row
            return None
        ev, ep, ce = pick("i1o4", "e_v"), pick("i1o4", "e_psi"), pick("oracle_fd4", "e_v")
        gm = r["gmres"][-1] if r["gmres"] else None
        out.append("| %s:%g:%d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            field, eps, N, r.get("status"), fmt(r.get("r_F"), "%.2e"),
            "yes" if r["path"][2] else "no (`%s` vs `%s`)" % (r["path"][0], r["path"][1]),
            fmt(ev[2] if ev else None, "%.6e"), fmt(ev[3] if ev else None, "%.6e"), ev[5] if ev else "-",
            fmt(ep[2] if ep else None, "%.6e"), fmt(ep[3] if ep else None, "%.6e"), ep[5] if ep else "-",
            ce[5] if ce else "-", fmt(r.get("fielddiff"), "%.2e"),
            "%.1f" % r["hist_digits"][0] if r.get("hist_digits") else "-",
            "%d / %.1f" % (gm[3], gm[4]) if gm else "-", "PASS" if r["pass"] else "FAIL"))
    out.append("")
    out.append("Pass rule: full-precision prototype values (solution.json / ref_metrics.json) gated at 1e-6 relative; "
               "4-significant-digit prototype values (cell logs / summary.md) gated at the rounding bound of the "
               "printed value (`PASS(4dig)`), which cannot resolve 1e-6.  r_F gate <= 1e-10; FIELDDIFF expected "
               "<= 1e-8 (joint normalization).")
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------------------------------------------
# SF-19 cross-check
# --------------------------------------------------------------------------------------------------------------
def crosscheck(logs, out):
    rows = []
    for p in logs:
        with open(p) as f:
            for line in f:
                if line.startswith("CROSSCHECK N="):
                    m = re.search(r"N=(\d+) v1_face: rms_rel=(\S+) max_rel=(\S+) .* rms_rel_avg=(\S+) .* "
                                  r"vperp_vertex: rms=(\S+) max=(\S+)", line)
                    rows.append([int(m.group(1))] + [float(m.group(i)) for i in range(2, 7)])
    rows.sort()
    names = ["v1 rms_rel", "v1 max_rel", "v1 rms_rel_avg", "vperp rms", "vperp max"]
    lines = ["| N | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for r in rows:
        lines.append("| %d | " % r[0] + " | ".join("%.3e" % x for x in r[1:]) + " |")
    if len(rows) >= 2:
        Ns = [r[0] for r in rows]
        for i, nm in enumerate(names):
            o = M.orders([r[i + 1] for r in rows], Ns)
            lines.append("")
            lines.append("- observed order %s over N=%s: %s" % (nm, "/".join(map(str, Ns)),
                                                                 " ".join("%.2f" % x for x in o)))
    txt = "\n".join(lines) + "\n"
    print(txt)
    if out:
        with open(out, "w") as f:
            f.write(txt)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--summary", default=os.path.join(SF29_DIR, "raw", "sweep2", "summary.md"))
    ap.add_argument("--exports", default=os.path.normpath(os.path.join(_HERE, "..", "exports")))
    ap.add_argument("--out", default=None)
    ap.add_argument("--crosscheck", action="store_true")
    a = ap.parse_args(argv)
    if a.crosscheck:
        crosscheck(a.logs, a.out)
        return 0
    rows, paths = parse_summary(a.summary)
    sweep_dir = os.path.dirname(os.path.abspath(a.summary)) if a.summary else None
    results = []
    for p in a.logs:
        g = parse_gpu_log(p)
        if not g["meta"].get("field"):
            print("SKIP %s: no CASE_DIR / CASE line (status %s)" % (p, g["status"]))
            continue
        r = compare_case(g, rows, paths, a.exports, sweep_dir)
        r["status"] = g["status"]
        results.append(r)
    md = markdown(results)
    print(md)
    if a.out:
        with open(a.out, "w") as f:
            f.write("# SF-33 prototype reproduction: GPU `inlet_slab --proto` vs SF-29 prototype\n\n"
                    "Generated by `scripts/compare_proto.py` (no interpretation).\n\n" + md)
    return 0 if results and all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
