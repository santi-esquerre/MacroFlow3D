#!/usr/bin/env python3
"""SF-29 N4: factual digest of the slow / timed-out sweep cells -> <out>/timeouts.md.

Parses only existing files (manifest.json, the cell logs <cell>.txt and the job log sf29-run-all.joblog.txt);
computes nothing numerical beyond counting, min/median/max and time-weighted concurrency.  No interpretation.

  python3 sweep_digest.py --out ../raw/sweep [--slow 3600]
  python3 sweep_digest.py --out ../raw/sweep2 --joblog sf29-corrective.joblog.txt   (corrective matrix, N4c)

The corrective layout (run_all.py --matrix corrective) adds: kind i1o4 (parsed as candidate i), LINEAR splu
factorization times, a table of elapsed / peak RSS medians per (kind, N), and the job-wide max running / max mem.
"""
import json
import os
import re
import statistics
import sys

RE_JOB = re.compile(r"^\[\s*(\d+)s\]\s+(\w+)\s+(\S+)")
RE_I_NEWTON = re.compile(r"NEWTON (\S+) it=\s*(\d+) r_F=(\S+) r_out=(\S+)(?: lambda=(\S+) \|dx\|max=\S+ lin=(\S+) "
                         r"rel=(\S+) lits=(\d+) t_jac=(\S+)s t_lin=(\S+)s)?")
RE_I_STAGE = re.compile(r"^STAGE field=\S+ eps=(\S+) N=\d+ cand=\S+ \((.*)\)")
RE_I_STAGE_END = re.compile(r"STAGE_END field=\S+ eps=(\S+) N=\d+ cand=\S+ status=(\S+) its=(\d+) r_F=(\S+)")
RE_I_FACT = re.compile(r"LINEAR splu .*t_fact=(\S+)s .*maxrss=(\d+)MB")
RE_JOB_MEM = re.compile(r"running (\d+)/\d+ mem (\S+)/(\S+) GB")
RE_II_NEWTON = re.compile(r"NEWTON s=(\S+) it=\s*(\d+) g_rel=(\S+) c_rel=(\S+) E=\S+ \|F\|=(\S+) nu=(\S+)")
RE_II_STEP = re.compile(r"step alpha=(\S+) \|d\|=\S+ nu=\S+ trials=(\d+) hess\(.*?\)=(\S+)s solve\[(.*?)\]=(\S+)s")
RE_II_HSTAGE = re.compile(r"HISTORY_STAGE .* s=(\S+) its=(\d+) status=(\S+)")
I_LIN_TOL = 1e-13      # candidate_i.py default --lin-tol (cmd lines of the sweep do not override it)
I_GMRES_CAP = 6000     # candidate_i.gmres_right maxiter
II_GMRES_CAP = 2400    # candidate_ii._gmres restart 60 x maxiter 40


def mmm(v, fmt="%d"):
    if not v:
        return "-"
    return "/".join(fmt % x for x in (min(v), statistics.median(v), max(v)))


def job_intervals(joblog):
    start, end = {}, {}
    for line in open(joblog):
        m = RE_JOB.match(line)
        if not m:
            continue
        t, ev, cid = int(m.group(1)), m.group(2), m.group(3)
        if ev == "start":
            start[cid] = t
        elif ev in ("done", "timeout", "failed"):
            end[cid] = t
    return {c: (start[c], end.get(c)) for c in start}


def running_profile(iv, tend):
    """run[t] = number of cells running during the second [t, t + 1) (1 s resolution of the job log)."""
    d = [0] * (tend + 2)
    for a, b in iv.values():
        if b is None:
            continue
        d[a] += 1; d[b] -= 1
    run, cur = [], 0
    for t in range(tend):
        cur += d[t]; run.append(cur)
    return run


def concurrency(run, t0, t1):
    """Time-weighted mean and max number of running cells over [t0, t1)."""
    seg = run[t0:t1]
    if not seg:
        return float("nan"), 0
    return sum(seg) / len(seg), max(seg)


def parse_i(lines):
    stages, cur = [], None
    path, nbis, newtons = None, 0, []
    for ln in lines:
        s = ln.strip()
        m = RE_I_STAGE.match(s)
        if m:
            cur = {"eps": m.group(1), "from": m.group(2), "status": "in progress at kill", "its": None, "r_F": None,
                   "lits": [], "rel": [], "t_lin": 0.0, "t_jac": 0.0}
            stages.append(cur)
            continue
        m = RE_I_STAGE_END.search(s)
        if m and cur is not None:
            cur["status"], cur["its"], cur["r_F"] = m.group(2), int(m.group(3)), m.group(4)
            continue
        if s.startswith("CONTINUATION"):
            nbis += 1
        if s.startswith("PATH"):
            path = s.split("continuation path:")[-1].strip()
        m = re.match(r"EXTRA .* status=(\S+) r_F=(\S+)", s)
        if m and cur is not None and cur["its"] is None:      # final attempt: no STAGE_END, status on EXTRA
            cur["status"] = "%s (EXTRA line; final attempt has no STAGE_END)" % m.group(1)
            cur["r_F"] = m.group(2)
        m = RE_I_NEWTON.search(s)
        if m:
            newtons.append(m)
            if m.group(8) is not None and cur is not None:
                cur["lits"].append(int(m.group(8))); cur["rel"].append(float(m.group(7)))
                cur["t_lin"] += float(m.group(10)); cur["t_jac"] += float(m.group(9))
                cur.setdefault("lin", set()).add(m.group(6))
    return stages, path, nbis, newtons


def fmt_i(cell, rec, lines):
    stages, path, nbis, newtons = parse_i(lines)
    out = []
    if path is None:
        path = "->".join("%s(%s)" % (st["eps"], st["status"] if st["status"] != "converged" else "ok")
                         for st in stages) + " [no PATH line]"
    out.append("- continuation path: `%s`; bisections (CONTINUATION lines): %d" % (path, nbis))
    if newtons:
        m = newtons[-1]
        out.append("- last Newton line: stage `%s` it=%s r_F=%s r_out=%s lambda=%s" % (
            m.group(1), m.group(2), m.group(3), m.group(4), m.group(5) or "-"))
    all_l = [x for st in stages for x in st["lits"]]
    all_r = [x for st in stages for x in st["rel"]]
    gm = [(l, r) for st in stages for l, r in zip(st["lits"], st["rel"]) if "gmres+lin0" in st.get("lin", ())]
    n_cap = sum(1 for l, r in gm if l >= I_GMRES_CAP)
    n_stag = sum(1 for l, r in gm if l < I_GMRES_CAP and r > I_LIN_TOL)
    n_ok = sum(1 for l, r in gm if r <= I_LIN_TOL)
    lins = sorted(set(x for st in stages for x in st.get("lin", ())))
    out.append("- linear solver(s): %s; Newton steps with a linear solve: %d; lits min/median/max %s; GMRES steps: "
               "%d converged (rel <= %.0e), %d hit the cap (lits >= %d), %d stopped by the restart-stagnation rule "
               "(lits < cap, rel > tol); max rel %s" % (
                   ", ".join(lins) or "-", len(all_l), mmm(all_l), n_ok, I_LIN_TOL, n_cap, I_GMRES_CAP, n_stag,
                   ("%.1e" % max(all_r)) if all_r else "-"))
    out.append("- time: sum t_lin %.0f s, sum t_jac %.0f s (of elapsed %.0f s)" % (
        sum(st["t_lin"] for st in stages), sum(st["t_jac"] for st in stages), rec["elapsed_s"]))
    fac = [RE_I_FACT.search(ln) for ln in lines]
    fac = [m for m in fac if m]
    if fac:
        tf = [float(m.group(1)) for m in fac]
        out.append("- splu factorizations: %d; t_fact min/median/max %s s, sum %.0f s; last t_fact %.0f s; max "
                   "maxrss %d MB" % (len(tf), mmm(tf, "%.0f"), sum(tf), tf[-1], max(int(m.group(2)) for m in fac)))
    for st in stages:
        out.append("  - stage eps=%s (%s): status %s, its %s, r_F %s; lits per step: %s; rel per step: %s" % (
            st["eps"], st["from"], st["status"], st["its"] if st["its"] is not None else len(st["lits"]),
            st["r_F"] or "-", " ".join(map(str, st["lits"])) or "-", " ".join("%.0e" % r for r in st["rel"]) or "-"))
    return out


def fmt_ii(cell, rec, lines):
    stages, cur, hst = [], None, {}
    for ln in lines:
        s = ln.strip()
        m = RE_II_NEWTON.search(s)
        if m:
            if cur is None or cur["s"] != m.group(1):
                cur = {"s": m.group(1), "newton": [], "steps": []}
                stages.append(cur)
            cur["newton"].append(m)
            continue
        m = RE_II_STEP.search(s)
        if m and cur is not None:
            cur["steps"].append(m)
            continue
        m = RE_II_HSTAGE.search(s)
        if m:
            hst["%.2f" % float(m.group(1))] = (m.group(2), m.group(3))
    out = []
    solve_line = next((ln.strip() for ln in lines if ln.strip().startswith("SOLVE ")), None)
    def ii_status(j):
        key = "%.2f" % float(stages[j]["s"])
        if key in hst:
            return hst[key][1]
        if j < len(stages) - 1:     # HISTORY_STAGE lines are printed only at the end of the solve
            return "ended, next stage started; status not logged before kill"
        return "in progress at kill"

    pth = []
    for j, st in enumerate(stages):
        pth.append("s=%s(%s)" % (st["s"], ii_status(j)))
    out.append("- continuation path (candidate_ii: fixed stages s = 0.25, 0.5, 1 from the inlet start, single stage s = 1 with --start oracle; no bisection): `%s`" %
               " -> ".join(pth))
    if solve_line:
        m = re.search(r"solver=(\S+) status=(\S+) its=(\d+) g_rel=(\S+)", solve_line)
        if m:
            out.append("- SOLVE line: solver=%s status=%s its=%s g_rel=%s" % m.groups())
    if stages and stages[-1]["newton"]:
        m = stages[-1]["newton"][-1]
        out.append("- last Newton line: s=%s it=%s g_rel=%s c_rel=%s |F|=%s nu=%s" % m.groups())
    steps = [x for st in stages for x in st["steps"]]
    kinds = sorted(set(x.group(4).split()[0] for x in steps))
    gits = [int(re.search(r"its=(\d+)", x.group(4)).group(1)) for x in steps if x.group(4).startswith("gmres")]
    ginfo = [int(re.search(r"info=(-?\d+)", x.group(4)).group(1)) for x in steps if x.group(4).startswith("gmres")]
    ts = [float(x.group(5)) for x in steps]
    th = [float(x.group(3)) for x in steps]
    tr = [int(x.group(2)) for x in steps]
    line = "- linear solver(s): %s; Newton steps: %d; solve time per step min/median/max %s s (sum %.0f s); Hessian " \
           "assembly sum %.0f s; steps with > 1 LM trial: %d (max trials %s)" % (
               ", ".join(kinds) or "-", len(steps), mmm(ts, "%.0f"), sum(ts), sum(th),
               sum(1 for t in tr if t > 1), max(tr) if tr else "-")
    if gits:
        line += "; GMRES its min/median/max %s, steps with info != 0 (not converged; cap %d = restart 60 x " \
                "maxiter 40): %d" % (mmm(gits), II_GMRES_CAP, sum(1 for i in ginfo if i != 0))
    else:
        line += "; no GMRES (direct splu)"
    out.append(line)
    for j, st in enumerate(stages):
        g = [float(n.group(3)) for n in st["newton"]]
        out.append("  - stage s=%s: status %s, Newton lines %d, g_rel first/last %s/%s; solve s per step: %s" % (
            st["s"], ii_status(j), len(st["newton"]),
            "%.1e" % g[0] if g else "-", "%.1e" % g[-1] if g else "-",
            " ".join("%.0f" % float(x.group(5)) for x in st["steps"]) or "-"))
    return out


def fmt_cons(cell, rec, lines):
    c = [ln.strip() for ln in lines if ln.strip().startswith("CONSIST ")]
    ns = [re.search(r" N=(\d+)", x).group(1) for x in c]
    return ["- consistency evaluation at the oracle labels (no Newton solve); grids completed: %s; CONSIST_ORDER "
            "lines: %d" % (",".join(ns) or "-", sum(1 for ln in lines if ln.strip().startswith("CONSIST_ORDER")))]


def main(argv):
    out, slow, joblog_name = None, 3600.0, "sf29-run-all.joblog.txt"
    i = 0
    while i < len(argv):
        if argv[i] == "--out":
            out = os.path.abspath(argv[i + 1]); i += 2
        elif argv[i] == "--slow":
            slow = float(argv[i + 1]); i += 2
        elif argv[i] == "--joblog":
            joblog_name = argv[i + 1]; i += 2
        else:
            print(__doc__); return 2
    man = json.load(open(os.path.join(out, "manifest.json")))
    cells = man["cells"]
    joblog = os.path.join(out, joblog_name)
    jl = open(joblog).read().splitlines()
    iv = job_intervals(joblog)
    tend = max(b for a, b in iv.values() if b is not None)
    run = running_profile(iv, tend)
    cm, cx = concurrency(run, 0, tend)
    sel = sorted((k for k, r in cells.items() if r["status"] == "timeout" or (r.get("elapsed_s") or 0) > slow),
                 key=lambda k: (cells[k]["status"] != "timeout", cells[k]["kind"], k))
    L = ["# SF-29 N4 sweep: timed-out and slow cells (machine-generated by `scripts/sweep_digest.py`; do not edit)", ""]
    L.append("Source: `manifest.json`, the cell logs `<cell>.txt` and the job log `%s` in this "
             "directory." % joblog_name + " Facts only (parsed log lines, counts, min/median/max); no interpretation.")
    L.append("")
    L.append("## Job settings and concurrency")
    L.append("")
    for ln in jl[:5]:
        if "command=" in ln or ln.startswith("run_all:"):
            L.append("- `%s`" % ln.strip())
    L.append("- manifest: workers %s, threads_per_worker %s, host `%s`, python %s, numpy %s" % (
        man.get("workers"), man.get("threads_per_worker"), man.get("host"), man.get("python"), man.get("numpy")))
    for r in man.get("runs", []):
        L.append("- run: start %s, end %s, wall %s s, cells run %s" % (r.get("start"), r.get("end"), r.get("wall_s"),
                                                                     r.get("cells_to_run")))
        if "max_running" in r:
            L.append("  - manifest run record: max_running %s, max_mem_gb %s (mem budget %s GB)" % (
                r.get("max_running"), r.get("max_mem_gb"), man.get("mem_budget_gb")))
    jm = [RE_JOB_MEM.search(ln) for ln in jl]
    jm = [m for m in jm if m]
    if jm:
        L.append("- job log scheduler lines: max running %d, max reserved mem %.1f GB (budget %s GB)" % (
            max(int(m.group(1)) for m in jm), max(float(m.group(2)) for m in jm), jm[0].group(3)))
    for ln in jl:
        if man.get("matrix") == "corrective" and ln.startswith("run_all") and "finished" in ln:
            L.append("- `%s`" % ln.strip())
    L.append("- concurrent running cells over the job (from the start/done/timeout events of the job log, 1 s "
             "resolution): time-weighted mean %.1f, max %d (worker pool size %s; each cell a process with 3 BLAS "
             "threads)" % (cm, cx, man.get("workers")))
    hours = []
    for h in range(0, int(tend) + 1, 3600):
        hours.append("%dh:%d" % (h // 3600, run[h] if h < len(run) else 0))
    L.append("- running cells at each hour mark: %s" % " ".join(hours))
    counts = {}
    for r in cells.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    L.append("- cell statuses: %s; this file lists %d cells: %d timeout + %d done with elapsed > %.0f s" % (
        ", ".join("%s %d" % kv for kv in sorted(counts.items())), len(sel),
        sum(1 for k in sel if cells[k]["status"] == "timeout"), sum(1 for k in sel if cells[k]["status"] != "timeout"),
        slow))
    L.append("")
    L.append("## Elapsed [s] of candidate `i1` on gauss and gauss_ch (T = timeout at the cell cap)")
    L.append("")
    i1n = sorted({r["N"] for r in cells.values() if r["kind"] == "i1" and r.get("N")}) or [16, 24, 32, 48, 64]
    L.append("| field | eps | %s |" % " | ".join("N=%d" % N for N in i1n))
    L.append("|---|---|%s" % ("---|" * len(i1n)))
    for f in ("gauss", "gauss_ch"):
        for e in ("0.25", "0.5", "1"):
            row = []
            for N in i1n:
                r = cells.get("i1-%s-%s-N%d" % (f, e, N))
                row.append("-" if r is None else "%.0f%s" % (r["elapsed_s"], " T" if r["status"] == "timeout" else ""))
            L.append("| %s | %s | %s |" % (f, e, " | ".join(row)))
    L.append("")
    if man.get("matrix") == "corrective":
        L.append("## Elapsed [s] and peak RSS [GB] vs (kind, N) (median over all cells of that kind and N; n = cells, "
                 "T = timeouts included)")
        L.append("")
        L.append("| kind | N | n | T | elapsed median / max [s] | peak RSS median / max [GB] |")
        L.append("|---|---|---|---|---|---|")
        grp = {}
        for r in cells.values():
            grp.setdefault((r["kind"], r.get("N")), []).append(r)
        for (kd, N) in sorted(grp, key=lambda x: (x[0], -1 if x[1] is None else x[1])):
            g = grp[(kd, N)]
            el = [r["elapsed_s"] for r in g if r.get("elapsed_s") is not None]
            rs = [r["peak_rss_gb"] for r in g if r.get("peak_rss_gb") is not None]
            L.append("| %s | %s | %d | %d | %s | %s |" % (
                kd, "-" if N is None else N, len(g), sum(1 for r in g if r["status"] == "timeout"),
                ("%.0f / %.0f" % (statistics.median(el), max(el))) if el else "-",
                ("%.2f / %.2f" % (statistics.median(rs), max(rs))) if rs else "-"))
        L.append("")
    L.append("## Overview of the listed cells")
    L.append("")
    L.append("| cell | status | elapsed [s] | cap [s] | mean / max concurrent cells during the cell |")
    L.append("|---|---|---|---|---|")
    for k in sel:
        r = cells[k]
        a, b = iv.get(k, (None, None))
        cc = concurrency(run, a, b) if a is not None and b is not None else (float("nan"), 0)
        L.append("| `%s` | %s | %.0f | %d | %.1f / %d |" % (k, r["status"], r["elapsed_s"], r["cap_s"], cc[0], cc[1]))
    L.append("")
    L.append("## Per-cell details")
    L.append("")
    for k in sel:
        r = cells[k]
        lines = open(os.path.join(out, r["log"])).read().splitlines()
        L.append("### `%s` (%s, elapsed %.0f s, cap %d s)" % (k, r["status"], r["elapsed_s"], r["cap_s"]))
        L.append("")
        L.append("- command: `%s`" % r["cmd"])
        if r["kind"] in ("i1", "i0", "i1o4"):
            L += fmt_i(k, r, lines)
        elif r["kind"] in ("ii", "iiorc"):
            L += fmt_ii(k, r, lines)
        elif r["kind"].startswith("cons"):
            L += fmt_cons(k, r, lines)
        else:
            L.append("- (kind %s: no parser)" % r["kind"])
        tail = [ln for ln in lines if ln.startswith("RUN_ALL TIMEOUT")]
        if tail:
            L.append("- `%s`" % tail[-1])
        L.append("")
    path = os.path.join(out, "timeouts.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("sweep_digest: wrote %s (%d cells)" % (path, len(sel)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
