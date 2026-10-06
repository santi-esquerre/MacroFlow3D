#!/usr/bin/env python3
"""SF-33 N5: observed orders of the production refinement ladder (claim (b), step 9b) from
`inlet_slab --production` logs (one grid per log, e.g. logs/ladder_0.5/N32.log N64.log N128.log).

Per grid: solver STATUS / STATUS_DETAIL, r_F, its, PATH, cand=i1o4 metrics (e_v, e_psi, e_psi1, e_psi2, e_i1,
e_i2, e_div, min_c), ceiling cand=oracle_fd4 e_v, GMRES statistics per stage (GMRES_STATS lines), TIMING (total,
solve, stage builds, oracle), MEMORY (peak device bytes, workspace bytes), production-oracle round trips
(ORACLE_SUMMARY per (h_max, tol)), the SF-18 applied scale (must agree across grids to 1e-10 relative: same
continuum field) and the SF-19 / inlet setup of the target stage.  Then observed orders over consecutive grids
(metrics.orders of the SF-29 artifact, imported read-only) and the SF-33 acceptance (b) reading:
orders of e_v and e_psi >= 1.8 on both pairs with r_F <= 1e-10 and no continuation floor.  Values come from the
driver's JSON summary (full precision) when the `SUMMARY_JSON` file exists, else from the 4-digit CASE lines.
No interpretation beyond the stated thresholds.

numpy only.  Usage:  python3 ladder_orders.py LOG [LOG ...] [--out table.md]
"""
import argparse
import json
import os
import re
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
SF29_SCRIPTS = os.path.normpath(os.path.join(_HERE, "..", "..", "2026-10-02-sf29-inlet-labels", "scripts"))
if SF29_SCRIPTS not in sys.path:
    sys.path.insert(0, SF29_SCRIPTS)
import metrics as M  # noqa: E402  (read-only reuse)

ORDER_KEYS = ["e_v", "e_psi", "e_psi1", "e_psi2", "e_i1", "e_i2", "e_div"]


def parse(path):
    g = {"log": path, "case": {}, "gmres": [], "oracle": [], "timing": {}, "memory": {}, "setup": [],
         "status": None, "detail": None, "path": None, "json": None, "applied_scale": None, "N": None,
         "eps": None, "field": None}
    with open(path) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if line.startswith("CASE "):
                d = M.parse_case_line(line)
                if isinstance(d.get("e_i"), tuple):
                    d["e_i1"], d["e_i2"] = d["e_i"]
                g["case"][d["cand"]] = d
                g["N"], g["eps"], g["field"] = int(d["N"]), d["eps"], d["field"]
            elif line.startswith("PATH field="):
                g["path"] = line.split("continuation path:", 1)[1].strip()
            elif line.startswith("GMRES_STATS "):
                g["gmres"].append(dict(t.split("=", 1) for t in line.split()[1:] if "=" in t))
            elif line.startswith("ORACLE_SUMMARY "):
                g["oracle"].append(dict(t.split("=", 1) for t in line.split()[1:] if "=" in t))
            elif line.startswith("TIMING "):
                for t in line.split()[1:]:
                    k, v = t.split("=", 1)
                    g["timing"][k] = float(v.rstrip("s"))
            elif line.startswith("TIMING_SETUP "):
                m = re.search(r"stage_builds=(\S+)s", line)
                g["timing"]["stage_builds"] = float(m.group(1))
            elif line.startswith("MEMORY "):
                for t in line.split()[1:]:
                    if "=" in t:
                        k, v = t.split("=", 1)
                        try:
                            g["memory"][k] = int(v)
                        except ValueError:
                            pass
            elif line.startswith("FIELD gaussian"):
                m = re.search(r"applied_scale=(\S+)", line)
                g["applied_scale"] = float(m.group(1))
            elif line.startswith("SETUP "):
                g["setup"].append(line[6:])
            elif line.startswith("STATUS_DETAIL "):
                g["detail"] = line.split(" ", 1)[1]
            elif line.startswith("STATUS "):
                g["status"] = line.split()[1]
            elif line.startswith("SUMMARY_JSON "):
                p = line.split(" ", 1)[1].strip()
                if os.path.exists(p):
                    with open(p) as jf:
                        g["json"] = json.load(jf)
    return g


def metric(g, cand, key):
    if g["json"] and cand in g["json"].get("metrics", {}):
        v = g["json"]["metrics"][cand].get(key)
        if v is not None:
            return float(v)
    v = g["case"].get(cand, {}).get(key)
    return float(v) if v is not None else float("nan")


def f(x, fm="%.3e"):
    try:
        return fm % x if np.isfinite(x) else "nan"
    except TypeError:
        return str(x)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    grids = [parse(p) for p in a.logs]
    grids = [g for g in grids if g["N"] is not None] + [g for g in grids if g["N"] is None]
    done = sorted([g for g in grids if g["N"] is not None], key=lambda g: g["N"])
    for g in grids:
        if g["N"] is None:
            print("NOTE %s: no CASE line (STATUS %s)" % (g["log"], g["status"]))
    L = []
    if not done:
        print("no grid with CASE lines")
        return 1
    L.append("# SF-33 production ladder: field=%s eps=%g\n" % (done[0]["field"], done[0]["eps"]))
    L.append("Generated by `scripts/ladder_orders.py` from %s (no interpretation).\n" %
             ", ".join(os.path.basename(g["log"]) for g in done))
    L.append("| N | status | r_F | its | PATH | e_v | e_psi | e_psi1 | e_psi2 | e_i1 | e_i2 | e_div | min_c | "
             "ceiling e_v | oracle max round trip (primary) | GMRES max / median (target stage) | wall [s] | "
             "solve [s] | oracle [s] | peak device [GB] |")
    L.append("|" + "---|" * 20)
    for g in done:
        fin = g["gmres"][-1] if g["gmres"] else {}
        orc = g["oracle"][0] if g["oracle"] else {}
        L.append("| %d | %s | %s | %s | `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s / %s | %s | %s | %s "
                 "| %s |" % (
                     g["N"], g["detail"] or g["status"], f(g["case"].get("i1o4", {}).get("r_F", float("nan"))),
                     f(g["case"].get("i1o4", {}).get("its", float("nan")), "%d"), g["path"],
                     f(metric(g, "i1o4", "e_v")), f(metric(g, "i1o4", "e_psi")), f(metric(g, "i1o4", "e_psi1")),
                     f(metric(g, "i1o4", "e_psi2")), f(metric(g, "i1o4", "e_i1")), f(metric(g, "i1o4", "e_i2")),
                     f(metric(g, "i1o4", "e_div")), f(metric(g, "i1o4", "min_c")),
                     f(metric(g, "oracle_fd4", "e_v")), orc.get("max_roundtrip", "-"),
                     fin.get("its_max", "-"), fin.get("its_median", "-"), f(g["timing"].get("total", float("nan")),
                                                                           "%.1f"),
                     f(g["timing"].get("solve", float("nan")), "%.1f"), f(g["timing"].get("oracle", float("nan")),
                                                                          "%.1f"),
                     f(g["memory"].get("peak_device_bytes", float("nan")) / 1e9, "%.2f")))
    L.append("")
    Ns = [g["N"] for g in done]
    if len(done) >= 2:
        L.append("Observed orders over N = %s (`metrics.orders`):\n" % "/".join(map(str, Ns)))
        for k in ORDER_KEYS:
            errs = [metric(g, "i1o4", k) for g in done]
            o = M.orders(errs, Ns) if all(np.isfinite(errs)) and all(e > 0 for e in errs) else []
            L.append("- %s: %s | orders %s" % (k, " ".join(f(e) for e in errs), " ".join("%.2f" % x for x in o)))
        ce = [metric(g, "oracle_fd4", "e_v") for g in done]
        oc = M.orders(ce, Ns) if all(np.isfinite(ce)) and all(e > 0 for e in ce) else []
        L.append("- ceiling e_v (oracle_fd4): %s | orders %s" % (" ".join(f(e) for e in ce),
                                                                 " ".join("%.2f" % x for x in oc)))
        L.append("")
        # acceptance (b) reading (spec thresholds; no interpretation)
        def ok_orders(k):
            errs = [metric(g, "i1o4", k) for g in done]
            if not (all(np.isfinite(errs)) and all(e > 0 for e in errs)):
                return False, []
            o = M.orders(errs, Ns)
            return all(x >= 1.8 for x in o), o
        ev_ok, ev_o = ok_orders("e_v")
        ep_ok, ep_o = ok_orders("e_psi")
        rF = [g["case"].get("i1o4", {}).get("r_F", float("nan")) for g in done]
        rF_ok = all(np.isfinite(x) and x <= 1e-10 for x in rF)
        floor = any("fail" in (g["path"] or "") or (g["status"] == "continuation_floor") for g in done)
        L.append("SF-33 acceptance (b) reading on N = %s: e_v orders >= 1.8 on every pair: %s (%s); e_psi: %s (%s); "
                 "r_F <= 1e-10 on every grid: %s; continuation floor reached: %s; PATH with a failed stage: %s."
                 % ("/".join(map(str, Ns)), "yes" if ev_ok else "no", " ".join("%.2f" % x for x in ev_o),
                    "yes" if ep_ok else "no", " ".join("%.2f" % x for x in ep_o), "yes" if rF_ok else "no",
                    "yes" if any(g["status"] == "continuation_floor" for g in done) else "no",
                    "yes" if floor else "no"))
        L.append("")
    sc = [g["applied_scale"] for g in done if g["applied_scale"] is not None]
    if len(sc) >= 2:
        rel = max(abs(s - sc[0]) / abs(sc[0]) for s in sc)
        L.append("SF-18 applied_scale per grid: %s; max relative spread %.2e (%s 1e-10: same continuum field)"
                 % (" ".join("%.15e" % s for s in sc), rel, "<=" if rel <= 1e-10 else ">"))
        L.append("")
    L.append("Production-oracle round trips (ORACLE_SUMMARY, every (h_max, tol) run):\n")
    for g in done:
        for o in g["oracle"]:
            L.append("- N=%d hmax=%s tol=%s status=%s max_roundtrip=%s non_ok=%s t_trace=%s" % (
                g["N"], o.get("hmax"), o.get("tol"), o.get("status"), o.get("max_roundtrip"), o.get("non_ok"),
                o.get("t_trace")))
    L.append("")
    L.append("GMRES statistics per continuation stage (GMRES_STATS):\n")
    for g in done:
        for s in g["gmres"]:
            L.append("- N=%d stage_eps=%s status=%s newton_its=%s its_max=%s its_median=%s its_total=%s "
                     "linear_failures=%s" % (g["N"], s.get("stage_eps"), s.get("status"), s.get("newton_its"),
                                             s.get("its_max"), s.get("its_median"), s.get("its_total"),
                                             s.get("linear_failures")))
    L.append("")
    L.append("Timing / memory:\n")
    for g in done:
        L.append("- N=%d TIMING %s | MEMORY %s" % (g["N"], " ".join("%s=%.1fs" % kv for kv in sorted(g["timing"].items())),
                                                   " ".join("%s=%d" % kv for kv in sorted(g["memory"].items()))))
    L.append("")
    L.append("Target-stage setup (SF-19 / inlet / v1 consistency):\n")
    for g in done:
        tgt = [s for s in g["setup"] if s.startswith("STAGE field=") and ("eps=%g " % g["eps"]) in s]
        if tgt:
            i = g["setup"].index(tgt[-1])
            for s in g["setup"][i:i + 7]:
                L.append("- N=%d %s" % (g["N"], s.strip()))
    txt = "\n".join(L) + "\n"
    print(txt)
    if a.out:
        with open(a.out, "w") as fo:
            fo.write(txt)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
