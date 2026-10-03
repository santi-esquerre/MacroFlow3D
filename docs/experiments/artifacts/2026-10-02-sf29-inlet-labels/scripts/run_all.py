#!/usr/bin/env python3
"""SF-29 N4: resumable sweep driver (oracle ceilings, candidates i1/i0/ii, consistency ladders, dense spectra)
and the machine-generated summary `summary.md`.

Run from this directory.  Every matrix cell is one subprocess (the candidate scripts are used exactly as their
CLIs define them; nothing is re-implemented here except the oracle-ceiling cell, which only calls the N1/N3
helpers).  Cells run in a pool of `--workers` concurrent subprocesses (subprocess + process groups, no
multiprocessing pickling, so the driver behaves identically under any start method / Python 3.11-3.13).

  python3 run_all.py --plan [--workers 22]          print the matrix, cost estimates, CPU hours, host check
  python3 run_all.py --workers 22 --out ../raw/sweep run (resumable); writes <out>/<cell>.txt, manifest.json,
                                                    spectra/, and summary.md at the end
  python3 run_all.py --summarize --out ../raw/sweep  rebuild summary.md from the logs only
  python3 run_all.py --oracle-cell field:eps --grids 12,16,...   (internal) oracle ladder + ceiling CASE lines

Filters (smoke / partial runs): --only field:eps[,field:eps...]  --grids 16[,24,...] (N of the solve/oracle
cells; spectra keep --spectra)  --spectra 12[,16]  --kinds oracle,i1,i0,ii,iiorc,cons_i1,cons_ii,spec_i1,spec_i0,spec_ii
Resume: cells with status `done` are always skipped; statuses listed in --retry (default failed,running,pending)
are rerun; `timeout` and `unsupported` are kept unless listed in --retry.

Matrix (N4 prompt section 3.1):
  oracle   (field, eps): load_case for N in {12, 16, 24, 32, 48} (+64 for gauss, gauss_ch, control2d), one process per
           (field, eps) with the reference built once (cases.build_reference memoized in that process); prints
           cand=oracle_fd (metrics.fd_metrics of the oracle labels) and cand=oracle_mim (candidate_ii mimetic flux of
           the oracle labels vs vD_faces) CASE lines.  N = 12 is included because the 12^3 spectra need its cache;
           every other cell starts only after the oracle cache of every N it uses exists (no concurrent cache writes).
  i1       candidate_i.py field:eps:N:i1 --direct-max 16, N in {16, 24, 32, 48} (+64 for gauss, gauss_ch, control2d)
  i0       candidate_i.py field:eps:16:i0 --bisect 2 --lm 10
  ii       candidate_ii.py field:eps:N (Q1 energy default), N in {16, 24, 32}, wall cap 4 h
  iiorc    candidate_ii.py --start oracle --maxit 6 field:eps:48 for gauss_ch:0.25, gauss:0.25 (diagnostic), cap 4 h
  cons_i1  candidate_i.py --consistency field:eps --grids 16,32,48
  cons_ii  candidate_ii.py --consistency field:eps --grids 16,32,48
  spec_i1  candidate_i.py --spectrum M field:eps:M:i1, M in {12, 16}
  spec_i0  candidate_i.py --spectrum M field:eps:M:i0 --bisect 2 --lm 10, M = 12 (+16 for gauss, gauss_ch)
  spec_ii  candidate_ii.py --spectrum M field:eps:M, M = 12 (+16 for gauss, gauss_ch, control2d at eps 0.25, 1)
"""
import functools
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import metrics as M  # noqa: E402

FIELDS = ("control2d", "lester2021", "lester_brk", "two_mode", "generic3d", "gauss", "gauss_ch")
EPSS = (0.25, 0.5, 1.0)
EXT64 = ("gauss", "gauss_ch", "control2d")             # D-6 extended ladder (N = 64)
I1_GRIDS = (16, 24, 32, 48)
II_GRIDS = (16, 24, 32)
ORACLE_GRIDS = (12, 16, 24, 32, 48)
CRIT_GRIDS = (16, 32, 48)                              # the spec's three grids (criteria (1)/(2))
II_CAP = 14400                                         # (ii) and the (ii) oracle-start diagnostic: 4 h
DEFAULT_CAP = 6 * 3600                                 # every other cell (a hang must not block the sweep)
I1_64_CAP = 8 * 3600
KINDS = ("oracle", "i1", "i0", "ii", "iiorc", "cons_i1", "cons_ii", "spec_i1", "spec_i0", "spec_ii")
K1_CONTROLS = ("raw/spectrum_cand_i_i0_uniform_k1_12.txt", "raw/spectrum_cand_i_i0_uniform_k1_16.txt",
               "raw/spectrum_cand_i_i1_uniform_k1_12.txt", "raw/spectrum_cand_i_i1_uniform_k1_16.txt",
               "raw/spectrum_cand_ii_q1_uniform_0_12.txt", "raw/spectrum_cand_ii_q1_uniform_0_12_kkt.txt",
               "raw/spectrum_cand_ii_q1_ctl_whitney_uniform_0_12.txt")


def fe(field, eps):
    return "%s:%g" % (field, eps)


# ------------------------------------------------------------------------------------------------
# matrix
# ------------------------------------------------------------------------------------------------
def oracle_grids(field):
    return ORACLE_GRIDS + ((64,) if field in EXT64 else ())


def est_cost(kind, field, eps, N):
    """Rough single-cell wall-time estimate in seconds (3 BLAS threads), from the N1/N2/N3/C-ii timings; used only
    for scheduling (longest first) and for the --plan totals."""
    ef = {0.25: 1.0, 0.5: 1.5, 1.0: 2.5}[eps]
    big = 2.0 if field in ("gauss_ch", "lester_brk") or eps == 1.0 else 1.0
    if kind == "oracle":
        per = {12: 6, 16: 12, 24: 30, 32: 70, 48: 250, 64: 650}
        return 30 + big * sum(per[n] for n in oracle_grids(field))
    if kind == "i1":
        return ef * {16: 40, 24: 250, 32: 600, 48: 2400, 64: 6000}[N]
    if kind == "i0":
        return 900
    if kind == "ii":
        return min(II_CAP, ef * {16: 400, 24: 2400, 32: 9000}[N])
    if kind == "iiorc":
        return II_CAP
    if kind == "cons_i1":
        return 150
    if kind == "cons_ii":
        return 400
    if kind == "spec_i1":
        return {12: 60, 16: 450}[N] * ef
    if kind == "spec_i0":
        return {12: 400, 16: 1800}[N]
    if kind == "spec_ii":
        return {12: 250, 16: 2400}[N] * ef
    raise ValueError(kind)


def cell_id(kind, field, eps, N=None):
    base = "%s-%s-%g" % (kind, field, eps)
    if N is None:
        return base
    return base + ("-M%d" % N if kind.startswith("spec") else "-N%d" % N)


def build_matrix(spectra_dir, only=None, grids=None, spectra=None, kinds=None):
    """List of cell dicts: id, kind, field, eps, N, argv (relative to this directory), needs (oracle N set), cap."""
    py = sys.executable or "python3"
    cells = []

    def add(kind, field, eps, N, argv, needs, cap=DEFAULT_CAP):
        if kinds is not None and kind not in kinds:
            return
        cells.append({"id": cell_id(kind, field, eps, N), "kind": kind, "field": field, "eps": eps, "N": N,
                      "argv": [py, "-u"] + argv, "needs": sorted(set(needs)), "cap": cap,
                      "est": est_cost(kind, field, eps, N)})

    for field in FIELDS:
        for eps in EPSS:
            if only is not None and fe(field, eps) not in only:
                continue
            spec = fe(field, eps)
            gsel = (lambda Ns: [n for n in Ns if grids is None or n in grids])
            ssel = (lambda Ms: [m for m in Ms if spectra is None or m in spectra])
            og = sorted(set([n for n in oracle_grids(field) if grids is None or n in grids]
                            + ssel([12, 16])))
            add("oracle", field, eps, None, ["run_all.py", "--oracle-cell", spec, "--grids", ",".join(map(str, og))],
                [])
            i1g = list(I1_GRIDS) + ([64] if field in EXT64 else [])
            for N in gsel(i1g):
                add("i1", field, eps, N, ["candidate_i.py", "%s:%d:i1" % (spec, N), "--direct-max", "16"], [N],
                    I1_64_CAP if N == 64 else DEFAULT_CAP)
            for N in gsel([16]):
                add("i0", field, eps, N, ["candidate_i.py", "%s:%d:i0" % (spec, N), "--bisect", "2", "--lm", "10"],
                    [N])
            for N in gsel(II_GRIDS):
                add("ii", field, eps, N, ["candidate_ii.py", "%s:%d" % (spec, N)], [N], II_CAP)
            if (field, eps) in (("gauss_ch", 0.25), ("gauss", 0.25)):
                for N in gsel([48]):
                    add("iiorc", field, eps, N, ["candidate_ii.py", "--start", "oracle", "--maxit", "6",
                                                 "%s:%d" % (spec, N)], [N], II_CAP)
            cg = [n for n in CRIT_GRIDS if grids is None or n in grids]
            if len(cg) >= 2:
                add("cons_i1", field, eps, None, ["candidate_i.py", "--consistency", spec, "--grids",
                                                  ",".join(map(str, cg))], cg)
                add("cons_ii", field, eps, None, ["candidate_ii.py", "--consistency", spec, "--grids",
                                                  ",".join(map(str, cg))], cg)
            for Msp in ssel([12, 16]):
                add("spec_i1", field, eps, Msp, ["candidate_i.py", "--spectrum", str(Msp), "%s:%d:i1" % (spec, Msp),
                                                 "--out", spectra_dir], [Msp])
            for Msp in ssel([12] + ([16] if field in ("gauss", "gauss_ch") else [])):
                add("spec_i0", field, eps, Msp, ["candidate_i.py", "--spectrum", str(Msp), "%s:%d:i0" % (spec, Msp),
                                                 "--bisect", "2", "--lm", "10", "--out", spectra_dir], [Msp])
            ii16 = field in EXT64 and eps in (0.25, 1.0)
            for Msp in ssel([12] + ([16] if ii16 else [])):
                add("spec_ii", field, eps, Msp, ["candidate_ii.py", "--spectrum", str(Msp), "%s:%d" % (spec, Msp),
                                                 "--out", spectra_dir], [Msp])
    return cells


# ------------------------------------------------------------------------------------------------
# oracle-ceiling cell (internal mode)
# ------------------------------------------------------------------------------------------------
def oracle_cell(spec, grids):
    import cases as C
    import candidate_ii as CII
    field, eps, _ = C.parse_spec(spec)
    # one reference per process: load_case rebuilds it per call otherwise (identical object, same inputs)
    C.build_reference = functools.lru_cache(maxsize=None)(C.build_reference)
    print("run_all.py --oracle-cell %s grids=%s python=%s numpy=%s host=%s" % (spec, grids, sys.version.split()[0],
                                                                             np.__version__, socket.gethostname()),
          flush=True)
    for N in grids:
        t0 = time.time()
        case = C.load_case(field, eps, N, verbose=False)
        mt = case["meta"]
        tl = time.time() - t0
        psi_or = case["psi_or"]
        mo = M.fd_metrics(psi_or[0], psi_or[1], case["vD"], psi_or)
        M.print_case(field, eps, N, "oracle_fd", mo, t=tl)
        a2, a3 = M.affine(N)
        U1, U2 = psi_or[0] - a2, psi_or[1] - a3
        fl = CII.mimetic_fluxes(U1, U2, 1.0 / N)
        mm = CII.candidate_metrics(case, U1, U2, fl)
        mm["e_psi"] = 0.0
        M.print_case(field, eps, N, "oracle_mim", mm, t=tl)
        print("ORACLE_READY field=%s eps=%g N=%d cache_hit=%s nphi=%d max_roundtrip=%.2e t_oracle=%.1fs t_faces=%.1fs "
              "t_cell=%.1fs cache=%s" % (field, eps, N, mt["cache_hit"], mt["nphi"], float(np.max(mt["roundtrip"])),
                                         mt["t_oracle"], mt["t_faces"], time.time() - t0,
                                         os.path.relpath(mt["cache"], _HERE)), flush=True)
    return 0


# ------------------------------------------------------------------------------------------------
# manifest / scheduler
# ------------------------------------------------------------------------------------------------
def load_manifest(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {"cells": {}}


def save_manifest(man, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
    os.replace(tmp, path)


def tail(path, n=25):
    try:
        with open(path, errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


def oracle_ready(out, field, eps):
    """Set of N for which the oracle cell of (field, eps) has written its cache (ORACLE_READY lines)."""
    s = set()
    p = os.path.join(out, cell_id("oracle", field, eps) + ".txt")
    if not os.path.exists(p):
        return s
    with open(p, errors="replace") as f:
        for line in f:
            if line.startswith("ORACLE_READY "):
                m = re.search(r" N=(\d+) ", line)
                if m:
                    s.add(int(m.group(1)))
    return s


def run(cells, out, workers, retry, poll=5.0):
    os.makedirs(out, exist_ok=True)
    mpath = os.path.join(out, "manifest.json")
    man = load_manifest(mpath)
    host = socket.gethostname()
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env.setdefault(k, "3")
    env["PYTHONUNBUFFERED"] = "1"
    todo = []
    for c in cells:
        rec = man["cells"].get(c["id"])
        if rec is not None and (rec["status"] == "done" or rec["status"] not in retry):
            continue
        man["cells"][c["id"]] = {"status": "pending", "kind": c["kind"], "field": c["field"], "eps": c["eps"],
                                 "N": c["N"], "cmd": " ".join(c["argv"][2:]), "log": c["id"] + ".txt",
                                 "cap_s": c["cap"], "est_s": c["est"]}
        todo.append(c)
    man["host"] = host
    man["workers"] = workers
    man["threads_per_worker"] = env["OMP_NUM_THREADS"]
    man["python"] = sys.version.split()[0]
    man["numpy"] = np.__version__
    man.setdefault("runs", []).append({"start": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "cells_to_run": len(todo)})
    save_manifest(man, mpath)
    print("run_all: %d cells to run (%d in matrix), workers=%d, threads/worker=%s, host=%s"
          % (len(todo), len(cells), workers, env["OMP_NUM_THREADS"], host), flush=True)
    oracle_ids = {(c["field"], c["eps"]): c["id"] for c in cells if c["kind"] == "oracle"}
    # oracles first, then longest first
    todo.sort(key=lambda c: (c["kind"] != "oracle", -c["est"]))
    running = {}
    t_start = time.time()

    def finish(cid, status, code, extra=None):
        rec = man["cells"][cid]
        rec["status"] = status
        rec["exit"] = code
        rec["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        if "t0" in rec:
            rec["elapsed_s"] = round(time.time() - rec.pop("t0"), 1)
        if status != "done":
            rec["tail"] = tail(os.path.join(out, rec["log"]))
        if extra:
            rec.update(extra)
        save_manifest(man, mpath)
        print("[%7.0fs] %-8s %s (exit %s, %ss)" % (time.time() - t_start, status, cid, code, rec.get("elapsed_s")),
              flush=True)

    while todo or running:
        # reap
        for cid in list(running):
            p, fh, c = running[cid]
            code = p.poll()
            rec = man["cells"][cid]
            if code is None:
                if time.time() - rec["t0"] > c["cap"]:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    p.wait()
                    fh.write("\nRUN_ALL TIMEOUT after %d s (cell wall cap); process group killed\n" % c["cap"])
                    fh.close()
                    del running[cid]
                    finish(cid, "timeout", None)
                continue
            fh.close()
            del running[cid]
            finish(cid, "done" if code == 0 else "failed", code)
        # dependency resolution: failed oracle -> dependents fail explicitly
        for c in list(todo):
            if c["kind"] == "oracle":
                continue
            oid = oracle_ids.get((c["field"], c["eps"]))
            orec = man["cells"].get(oid) if oid else None
            if orec is not None and orec["status"] in ("failed", "timeout"):
                rdy = oracle_ready(out, c["field"], c["eps"])
                if not set(c["needs"]) <= rdy:
                    todo.remove(c)
                    with open(os.path.join(out, c["id"] + ".txt"), "w") as fh:
                        fh.write("RUN_ALL NOT STARTED: oracle cell %s ended %s before the cache of N=%s existed\n"
                                 % (oid, orec["status"], sorted(set(c["needs"]) - rdy)))
                    man["cells"][c["id"]]["t0"] = time.time()
                    finish(c["id"], "failed", None, {"reason": "oracle dependency %s" % orec["status"]})
        # launch
        launched = True
        while launched and len(running) < workers and todo:
            launched = False
            for c in todo:
                if c["kind"] != "oracle":
                    oid = oracle_ids.get((c["field"], c["eps"]))
                    odone = oid is None or man["cells"].get(oid, {}).get("status") == "done"
                    if not odone and not set(c["needs"]) <= oracle_ready(out, c["field"], c["eps"]):
                        continue
                todo.remove(c)
                log = os.path.join(out, c["id"] + ".txt")
                fh = open(log, "w")
                fh.write("RUN_ALL cell=%s host=%s cwd=%s cmd=%s\n" % (c["id"], host, _HERE, " ".join(c["argv"])))
                fh.flush()
                p = subprocess.Popen(c["argv"], cwd=_HERE, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                     start_new_session=True)
                rec = man["cells"][c["id"]]
                rec.update({"status": "running", "host": host, "t0": time.time(),
                            "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "pid": p.pid})
                save_manifest(man, mpath)
                running[c["id"]] = (p, fh, c)
                print("[%7.0fs] start    %s (est %.0fs)" % (time.time() - t_start, c["id"], c["est"]), flush=True)
                launched = True
                break
        if running or todo:
            time.sleep(poll)
    man["runs"][-1]["end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    man["runs"][-1]["wall_s"] = round(time.time() - t_start, 1)
    save_manifest(man, mpath)
    counts = {}
    for c in cells:
        st = man["cells"].get(c["id"], {}).get("status", "missing")
        counts[st] = counts.get(st, 0) + 1
    print("run_all: finished in %.0f s; statuses %s" % (time.time() - t_start, counts), flush=True)
    return counts


# ------------------------------------------------------------------------------------------------
# plan
# ------------------------------------------------------------------------------------------------
def plan(cells, workers):
    import platform
    print("run_all --plan  python=%s numpy=%s host=%s cpus=%s" % (platform.python_version(), np.__version__,
                                                                    socket.gethostname(), os.cpu_count()))
    try:
        import scipy
        print("scipy=%s" % scipy.__version__)
        with open("/proc/meminfo") as f:
            print("memory: %s" % f.readline().strip())
    except Exception as exc:  # noqa: BLE001
        print("host check: %r" % exc)
    for mod in ("cases", "candidate_i", "candidate_ii"):
        __import__(mod)
        print("import %s: ok" % mod)
    by = {}
    for c in cells:
        by.setdefault(c["kind"], []).append(c)
    tot = 0.0
    print("%-8s %6s %12s %10s" % ("kind", "cells", "est CPU h", "max est h"))
    for k in KINDS:
        if k in by:
            s = sum(c["est"] for c in by[k]); tot += s
            print("%-8s %6d %12.1f %10.2f" % (k, len(by[k]), s / 3600.0, max(c["est"] for c in by[k]) / 3600.0))
    # greedy LPT simulation (oracles first; dependencies ignored beyond that)
    order = sorted(cells, key=lambda c: (c["kind"] != "oracle", -c["est"]))
    slots = [0.0] * workers
    for c in order:
        i = int(np.argmin(slots)); slots[i] += c["est"]
    print("total cells=%d  estimated sum=%.1f cell-hours (x3 threads = %.0f core-hours)  LPT wall with %d workers ~ "
          "%.1f h (lower bound: longest single cell %.1f h)"
          % (len(cells), tot / 3600.0, 3 * tot / 3600.0, workers, max(slots) / 3600.0,
             max(c["est"] for c in cells) / 3600.0))
    for c in order:
        print("  %-34s est %7.0fs cap %6ds needs %-18s %s" % (c["id"], c["est"], c["cap"], c["needs"],
                                                               " ".join(c["argv"][2:])))


# ------------------------------------------------------------------------------------------------
# summary
# ------------------------------------------------------------------------------------------------
def read(path):
    try:
        with open(path, errors="replace") as f:
            return f.read().splitlines()
    except OSError:
        return []


def case_lines(lines):
    out = []
    for ln in lines:
        if ln.startswith("CASE "):
            try:
                out.append(M.parse_case_line(ln))
            except ValueError:
                pass
    return out


def fnum(x, fmt="%.3e"):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "-"
    return fmt % x


def safe_orders(vals, Ns):
    o = []
    for i in range(len(Ns) - 1):
        a, b = vals[i], vals[i + 1]
        if a is None or b is None or not (a > 0 and b > 0) or not (math.isfinite(a) and math.isfinite(b)):
            o.append(None)
        else:
            o.append(M.orders([a, b], [Ns[i], Ns[i + 1]])[0])
    return o


def spec_eval(path):
    """Sorted relative singular values from a saved spectrum file -> stats and the criterion-(4) verdict.
    Criterion (4) (N4 prompt): FAIL iff a gap-separated group exists below 1e-3: consecutive sorted values
    r[i] < 1e-3 with r[i+1] / r[i] >= 100 (r[i] = 0 counts as an infinite gap)."""
    try:
        r = np.sort(np.abs(np.loadtxt(path, comments="#").ravel()))
    except (OSError, ValueError):
        return None
    if r.size == 0:
        return None
    r = r / r.max()
    cnt = {t: int(np.sum(r < t)) for t in (1e-3, 1e-6, 1e-10)}
    best = (1.0, 0, float("nan"), float("nan"))
    sep = None
    for i in range(r.size - 1):
        if r[i] < 1e-2:
            g = r[i + 1] / r[i] if r[i] > 0 else float("inf")
            if g > best[0]:
                best = (g, i + 1, r[i], r[i + 1])
            if r[i] < 1e-3 and g >= 100 and sep is None:
                sep = (i + 1, g)
    # the largest qualifying group (last index i below 1e-3 with a >= 100 gap) defines the cluster size
    for i in range(r.size - 1):
        if r[i] < 1e-3:
            g = r[i + 1] / r[i] if r[i] > 0 else float("inf")
            if g >= 100:
                sep = (i + 1, g)
    return {"n": int(r.size), "small3": [float(v) for v in r[:3]], "cnt": cnt, "gap": best, "sep": sep,
            "verdict": "PASS" if sep is None else "FAIL"}


def last_history(lines, prefix_re):
    """Values of the last matching history line (list of floats)."""
    vals = None
    rx = re.compile(prefix_re)
    for ln in lines:
        m = rx.match(ln)
        if m:
            try:
                vals = [float(v) for v in ln.split(":", 1)[1].split()]
            except ValueError:
                pass
    return vals


def plateau(hist, final):
    """Floor flag: final residual above 1e-13 (tolerance not reached) with the last three history values within a
    factor 10 (no decade gained over the last two iterations)."""
    if not hist or len(hist) < 3 or final is None or not math.isfinite(final):
        return False
    h = [v for v in hist[-3:] if v > 0]
    return final > 1e-13 and len(h) == 3 and max(h) / min(h) < 10.0


def collect(out, man):
    """Parse every log into per-(field, eps) records."""
    cells = man.get("cells", {})
    data = {}
    for cid, rec in cells.items():
        kind, field, eps, N = rec["kind"], rec["field"], rec["eps"], rec["N"]
        d = data.setdefault((field, eps), {"ceil_fd": {}, "ceil_mim": {}, "cand": {}, "cons": {}, "spec": [],
                                           "status": {}})
        d["status"][cid] = rec["status"]
        lines = read(os.path.join(out, rec["log"]))
        if kind == "oracle":
            for c in case_lines(lines):
                d["ceil_fd" if c["cand"] == "oracle_fd" else "ceil_mim"][int(c["N"])] = c
        elif kind in ("i1", "i0", "ii", "iiorc"):
            name = {"i1": "i1", "i0": "i0", "ii": "ii", "iiorc": "ii_from_oracle"}[kind]
            cl = [c for c in case_lines(lines) if c["cand"] == name]
            ent = {"status": rec["status"], "cid": cid, "elapsed": rec.get("elapsed_s")}
            if cl:
                ent.update(cl[-1])
                if kind in ("i1", "i0"):
                    hist = last_history(lines, r"HISTORY field=\S+ eps=\S+ N=\d+ cand=\S+ r_F:")
                    m = [ln for ln in lines if ln.startswith("EXTRA ")]
                    ent["solver_status"] = re.search(r"status=(\S+)", m[-1]).group(1) if m else "?"
                    pl = [ln for ln in lines if ln.startswith("PATH ")]
                    ent["path"] = pl[-1].split("path:", 1)[1].strip() if pl else ""
                else:
                    hist = last_history(lines, r"HISTORY_STAGE field=\S+ eps=\S+ N=\d+ cand=\S+ s=\S+ its=\d+ "
                                               r"status=\S+ g_rel:")
                    m = [ln for ln in lines if ln.startswith("SOLVE ")]
                    ent["solver_status"] = re.search(r"status=(\S+)", m[-1]).group(1) if m else "?"
                    ent["path"] = ""
                ent["plateau"] = plateau(hist, ent.get("r_F"))
            d["cand"].setdefault(name, {})[N] = ent
        elif kind in ("cons_i1", "cons_ii"):
            key = "CONSISTENCY_ORDER" if kind == "cons_i1" else "CONSIST_ORDER"
            rows = []
            for ln in lines:
                if ln.startswith(key + " "):
                    rows.append(ln.split(" ", 3)[3] if kind == "cons_i1" else ln.split(" ", 3)[3])
            d["cons"][kind] = {"status": rec["status"], "rows": rows}
        elif kind.startswith("spec"):
            var = kind[5:]
            files = []
            sd = os.path.join(out, "spectra")
            if var in ("i1", "i0"):
                for st in ("", "_final_iterate", "_oracle"):
                    p = os.path.join(sd, "spectrum_cand_i_%s_%s_%g_%d%s.txt" % (var, field, eps, N, st))
                    if os.path.exists(p):
                        files.append((st.strip("_") or "converged", p))
            else:
                ch = field.endswith("_ch")
                for st, lab in (("", "hessian" if ch else "reduced_hessian"), ("_kkt", "kkt"), ("_hL", "hessian_L")):
                    p = os.path.join(sd, "spectrum_cand_ii_q1_%s_%g_%d%s.txt" % (field, eps, N, st))
                    if os.path.exists(p):
                        files.append((lab, p))
            cl = [c for c in case_lines(lines) if c["cand"] in ("i1", "i0", "ii")]
            conv = (cl[-1].get("r_F") is not None and cl[-1]["r_F"] <= 1e-10) if cl else None
            for st, p in files:
                ev = spec_eval(p)
                if ev is not None:
                    d["spec"].append({"cand": var, "M": N, "state": st, "file": os.path.relpath(p, out), "ev": ev,
                                      "converged": conv, "status": rec["status"]})
            if not files:
                d["spec"].append({"cand": var, "M": N, "state": "-", "file": None, "ev": None, "converged": conv,
                                  "status": rec["status"]})
    return data


def classify_ev(Ns, e, eor, cand_order_key="e_v"):
    """D-5 criterion (1) on a ladder: returns (verdict, detail)."""
    if any(v is None for v in e) or any(v is None for v in eor):
        return "INCOMPLETE", "missing grid(s)"
    o = safe_orders(e, Ns); oc = safe_orders(eor, Ns)
    if any(v is None for v in o):
        return "INCOMPLETE", "non-positive value"
    if any(v <= 0.0 for v in o):
        return "FAIL", "floor (e_v not decreasing on a pair)"
    if all(v >= 1.8 for v in o):
        return "PASS", "orders %s" % " ".join("%.2f" % v for v in o)
    ratios = [a / b for a, b in zip(e, eor)]
    if all(r <= 1.5 for r in ratios) and all(oc[i] is not None and abs(o[i] - oc[i]) <= 0.15 for i in range(len(o))):
        return "ceiling-limited", "orders %s vs ceiling %s, max ratio %.2f" % (
            " ".join("%.2f" % v for v in o), " ".join("%.2f" % v for v in oc), max(ratios))
    return "FAIL", "orders %s vs ceiling %s, max ratio %.2f" % (
        " ".join("%.2f" % v for v in o), " ".join("%.2f" % v if v is not None else "-" for v in oc), max(ratios))


def classify_epsi(Ns, ep, eor):
    """Criterion (2): PASS iff e_psi orders >= 1.8 on every pair; ceiling-limited iff the ratio e_psi/e_v^or is
    bounded in the sense |order(e_psi) - order(e_v^or)| <= 0.15 on every pair; floor or otherwise FAIL."""
    if any(v is None for v in ep) or any(v is None for v in eor):
        return "INCOMPLETE", "missing grid(s)"
    o = safe_orders(ep, Ns); oc = safe_orders(eor, Ns)
    if any(v is None for v in o):
        return "INCOMPLETE", "non-positive value"
    if any(v <= 0.0 for v in o):
        return "FAIL", "floor (e_psi not decreasing on a pair)"
    if all(v >= 1.8 for v in o):
        return "PASS", "orders %s" % " ".join("%.2f" % v for v in o)
    if all(oc[i] is not None and abs(o[i] - oc[i]) <= 0.15 for i in range(len(o))):
        return "ceiling-limited", "orders %s vs ceiling %s" % (" ".join("%.2f" % v for v in o),
                                                               " ".join("%.2f" % v for v in oc))
    return "FAIL", "orders %s vs ceiling %s" % (" ".join("%.2f" % v for v in o),
                                                " ".join("%.2f" % v if v is not None else "-" for v in oc))


def summarize(out):
    man = load_manifest(os.path.join(out, "manifest.json"))
    data = collect(out, man)
    L = []
    L.append("# SF-29 N4 sweep summary (machine-generated by `scripts/run_all.py --summarize`; do not edit)")
    L.append("")
    L.append("Source: the cell logs `<cell>.txt` and `manifest.json` in this directory, spectra in `spectra/`. "
             "Numbers are copied from the CASE / CONSISTENCY / SPECTRUM lines of the logs (`metrics.parse_case_line`); "
             "orders are `metrics.orders` over consecutive grids. No interpretation; the experiment note (N5) reads "
             "this file.")
    L.append("")
    counts = {}
    for rec in man.get("cells", {}).values():
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    L.append("Host: `%s`; workers %s x %s threads; python %s, numpy %s. Cell statuses: %s." % (
        man.get("host"), man.get("workers"), man.get("threads_per_worker"), man.get("python"), man.get("numpy"),
        ", ".join("%s %d" % kv for kv in sorted(counts.items()))))
    for r in man.get("runs", []):
        L.append("Run: start %s, end %s, wall %s s, cells run %s." % (r.get("start"), r.get("end", "-"),
                                                                     r.get("wall_s", "-"), r.get("cells_to_run")))
    L.append("")
    L.append("## Reading rules applied (D-5, N4 prompt)")
    L.append("")
    L.append("- (1) `e_v`: PASS iff observed order >= 1.8 on both pairs of 16/32/48 (candidate (ii): 16/24/32, the "
             "only grids it runs); else `ceiling-limited` iff `e_v <= 1.5 e_v^or` on every grid and "
             "`|order - ceiling order| <= 0.15` on every pair; else FAIL. A pair with non-decreasing `e_v` is a "
             "floor -> FAIL (D-5). Ceiling: `oracle_fd` for (i), `oracle_mim` for (ii) (oracle cells).")
    L.append("- (2) `e_psi`: PASS iff order >= 1.8 on both pairs; `ceiling-limited` iff "
             "`|order(e_psi) - order(e_v^or)| <= 0.15` on every pair (the ratio `e_psi/e_v^or` is bounded; "
             "e_psi and e_v have different normalizations, so no 1.5 factor is applied; operationalization of "
             "D-5 by the N4 worker); floor or otherwise FAIL.")
    L.append("- (3) PASS iff `r_F <= 1e-10` on every grid of the candidate's ladder and no plateau (plateau flag: "
             "final `r_F > 1e-13` and the last three history values within a factor 10).")
    L.append("- (4) PASS iff no spectrum of the candidate at that (field, eps) has a gap-separated group below 1e-3 "
             "(consecutive sorted relative values `r_i < 1e-3` with `r_{i+1}/r_i >= 100`). Candidate (ii): the "
             "reduced Hessian (periodic) / Hessian (`_ch`) file decides; KKT and full Lagrangian Hessian are "
             "reported. Candidate (i): the converged-state file. Only spectra at a converged state (`r_F <= 1e-10` "
             "in that cell) are classified; spectra at unconverged states (final iterate / oracle labels) are listed "
             "in the tables and named in the verdict note; no converged spectrum -> `INCOMPLETE`.")
    L.append("- (5) = (1)-(4) on `gauss_ch` (rows of the classification table with field `gauss_ch`).")
    L.append("- `INCOMPLETE`: a needed grid has no CASE line (cell failed / timeout).")
    L.append("- Extended ladders (24 and, where run, 64) are reported as orders over every consecutive pair; the "
             "classification uses the criterion grids only.")
    L.append("")
    L.append("`k = 1` controls (not rerun, produced by N2 / C-ii): " + ", ".join("`%s`" % p for p in K1_CONTROLS))
    L.append("")
    classes = []
    for field in FIELDS:
        for eps in EPSS:
            d = data.get((field, eps))
            if d is None:
                continue
            L.append("## %s, eps = %g" % (field, eps))
            L.append("")
            bad = ["%s: %s" % (cid, st) for cid, st in sorted(d["status"].items()) if st != "done"]
            L.append("Cells not `done`: %s" % (", ".join("`%s`" % b for b in bad) if bad else "none"))
            L.append("")
            for cand, ceil_key in (("i1", "ceil_fd"), ("i0", "ceil_fd"), ("ii", "ceil_mim"),
                                   ("ii_from_oracle", "ceil_mim")):
                rows = d["cand"].get(cand)
                if not rows:
                    continue
                ceil = d[ceil_key]
                Ns = sorted(rows)
                L.append("### candidate `%s` (ceiling `%s`)" % (cand, "oracle_fd" if ceil_key == "ceil_fd"
                                                                else "oracle_mim"))
                L.append("")
                L.append("| N | status | solver status | e_v | e_psi | r_F | its | plateau | ceiling e_v | e_v/ceil | "
                         "e_psi/e_v^or | min_c | ceiling min_c | t [s] |")
                L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
                for N in Ns:
                    r = rows[N]; c = ceil.get(N, {})
                    ev, ep, ec = r.get("e_v"), r.get("e_psi"), c.get("e_v")
                    L.append("| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                        N, r["status"], r.get("solver_status", "-"), fnum(ev), fnum(ep), fnum(r.get("r_F")),
                        fnum(r.get("its"), "%.0f"), ("yes" if r.get("plateau") else "no") if "r_F" in r else "-",
                        fnum(ec), fnum(ev / ec if ev and ec else None, "%.2f"),
                        fnum(ep / ec if ep is not None and ec else None, "%.2f"), fnum(r.get("min_c")),
                        fnum(c.get("min_c")), fnum(r.get("t"), "%.0f")))
                L.append("")
                if len(Ns) >= 2:
                    def ser(get):
                        return [get(N) for N in Ns]
                    ev = ser(lambda N: rows[N].get("e_v")); ep = ser(lambda N: rows[N].get("e_psi"))
                    ec = ser(lambda N: ceil.get(N, {}).get("e_v"))
                    fmt = lambda o: " ".join("%.2f" % v if v is not None else "-" for v in o)  # noqa: E731
                    L.append("Orders over N = %s: e_v %s; e_psi %s; ceiling e_v %s" % (
                        "/".join(map(str, Ns)), fmt(safe_orders(ev, Ns)), fmt(safe_orders(ep, Ns)),
                        fmt(safe_orders(ec, Ns))))
                    L.append("")
                for N in Ns:
                    if rows[N].get("path"):
                        L.append("- continuation path N=%d: `%s`" % (N, rows[N]["path"]))
                if any(r.get("path") for r in rows.values()):
                    L.append("")
                # classification
                if cand in ("i1", "ii"):
                    cg = list(CRIT_GRIDS) if cand == "i1" else list(II_GRIDS)
                    e = [rows.get(N, {}).get("e_v") for N in cg]
                    p = [rows.get(N, {}).get("e_psi") for N in cg]
                    eo = [ceil.get(N, {}).get("e_v") for N in cg]
                    c1 = classify_ev(cg, e, eo)
                    c2 = classify_epsi(cg, p, eo)
                else:
                    cg = sorted(rows)
                    c1 = c2 = ("n/a", "single grid / diagnostic")
                rf = [rows[N].get("r_F") for N in sorted(rows)]
                pl = any(rows[N].get("plateau") for N in rows)
                if any(v is None or not math.isfinite(v) for v in rf):
                    c3 = ("INCOMPLETE", "no r_F at N=%s" % ",".join(
                        "%d(%s)" % (N, rows[N]["status"]) for N in sorted(rows)
                        if rows[N].get("r_F") is None or not math.isfinite(rows[N]["r_F"])))
                elif all(v <= 1e-10 for v in rf) and not pl:
                    c3 = ("PASS", "max r_F %.1e" % max(rf))
                else:
                    c3 = ("FAIL", "r_F %s%s" % (" ".join(fnum(v, "%.1e") for v in rf), "; plateau" if pl else ""))
                specs = [s for s in d["spec"] if s["cand"] == ("ii" if cand.startswith("ii") else cand)]
                prim = []
                for s in specs:
                    if s["ev"] is None:
                        continue
                    if cand.startswith("ii") and s["state"] not in ("hessian", "reduced_hessian"):
                        continue
                    if cand in ("i1", "i0") and s["state"] == "oracle":
                        continue
                    prim.append(s)
                # criterion (4) is read at converged states only; unconverged spectra are listed, not classified
                unconv = [s for s in prim if not s["converged"]]
                prim = [s for s in prim if s["converged"]]
                note = ("; not classified (state not converged): M=%s" % ",".join(
                    "%d/%s" % (s["M"], s["state"]) for s in unconv)) if unconv else ""
                if cand == "ii_from_oracle":
                    c4 = ("n/a", "diagnostic")
                elif not prim:
                    c4 = ("INCOMPLETE", "no converged-state spectrum%s" % note)
                else:
                    f4 = [s for s in prim if s["ev"]["verdict"] == "FAIL"]
                    c4 = ("FAIL" if f4 else "PASS", "%d converged spectra%s%s" % (
                        len(prim), (", gap-separated group at M=%s" % ",".join("%d(%s: %d values)" % (
                            s["M"], s["state"], s["ev"]["sep"][0]) for s in f4)) if f4 else "", note))
                classes.append((field, eps, cand, c1, c2, c3, c4))
            # consistency
            for kind, title in (("cons_i1", "candidate (i) residual at the oracle labels"),
                                ("cons_ii", "candidate (ii)-Q1 gradient/constraints at the oracle labels")):
                cs = d["cons"].get(kind)
                if cs is None:
                    continue
                L.append("### consistency: %s (`%s`, status %s)" % (title, kind, cs["status"]))
                L.append("")
                for row in cs["rows"]:
                    L.append("- `%s`" % row)
                if not cs["rows"]:
                    L.append("- (no order lines)")
                L.append("")
            # spectra
            if d["spec"]:
                L.append("### dense spectra (sorted relative singular values)")
                L.append("")
                L.append("| cand | M | state | converged | n | smallest 3 | <1e-3 | <1e-6 | <1e-10 | largest gap "
                         "below 1e-2 (ratio @ position: r -> r') | gap-separated group < 1e-3 | file |")
                L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
                for s in sorted(d["spec"], key=lambda s: (s["cand"], s["M"], s["state"])):
                    ev = s["ev"]
                    if ev is None:
                        L.append("| %s | %d | - | %s | - | - | - | - | - | - | - | (no file; cell %s) |"
                                 % (s["cand"], s["M"], s["converged"], s["status"]))
                        continue
                    g = ev["gap"]
                    L.append("| %s | %d | %s | %s | %d | %s | %d | %d | %d | %s @ %d: %s -> %s | %s | `%s` |" % (
                        s["cand"], s["M"], s["state"], s["converged"], ev["n"],
                        " ".join("%.2e" % v for v in ev["small3"]), ev["cnt"][1e-3], ev["cnt"][1e-6], ev["cnt"][1e-10],
                        "%.3g" % g[0], g[1], fnum(g[2], "%.2e"), fnum(g[3], "%.2e"),
                        ("%d values (gap %.3g)" % ev["sep"]) if ev["sep"] else "none", s["file"]))
                L.append("")
    L.append("## D-5 classification per criterion")
    L.append("")
    L.append("| field | eps | candidate | (1) e_v | (2) e_psi | (3) r_F | (4) spectrum |")
    L.append("|---|---|---|---|---|---|---|")
    for field, eps, cand, c1, c2, c3, c4 in classes:
        L.append("| %s | %g | %s | %s | %s | %s | %s |" % (field, eps, cand, *("**%s** (%s)" % c for c in (c1, c2, c3,
                                                                                                         c4))))
    L.append("")
    L.append("### (5) constant-head case (`gauss_ch`)")
    L.append("")
    L.append("| eps | candidate | (1) | (2) | (3) | (4) |")
    L.append("|---|---|---|---|---|---|")
    for field, eps, cand, c1, c2, c3, c4 in classes:
        if field == "gauss_ch":
            L.append("| %g | %s | %s | %s | %s | %s |" % (eps, cand, c1[0], c2[0], c3[0], c4[0]))
    L.append("")
    L.append("### Counts")
    L.append("")
    for cand in ("i1", "i0", "ii", "ii_from_oracle"):
        for k, name in ((3, "(1)"), (4, "(2)"), (5, "(3)"), (6, "(4)")):
            vals = [c[k][0] for c in classes if c[2] == cand]
            if not vals:
                continue
            cnt = {}
            for v in vals:
                cnt[v] = cnt.get(v, 0) + 1
            L.append("- `%s` %s: %s" % (cand, name, ", ".join("%s %d" % kv for kv in sorted(cnt.items()))))
    L.append("")
    path = os.path.join(out, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("run_all: wrote %s (%d (field, eps) blocks, %d classification rows)" % (path, len(data), len(classes)),
          flush=True)
    return path


# ------------------------------------------------------------------------------------------------
def main(argv):
    opts = {"workers": 22, "out": os.path.normpath(os.path.join(_HERE, "..", "raw", "sweep")), "only": None,
            "grids": None, "spectra": None, "kinds": None, "retry": ("failed", "running", "pending"), "mode": "run"}
    i = 0
    ocell = None
    while i < len(argv):
        a = argv[i]
        if a == "--plan":
            opts["mode"] = "plan"; i += 1
        elif a == "--summarize":
            opts["mode"] = "summarize"; i += 1
        elif a == "--oracle-cell":
            opts["mode"] = "oracle"; ocell = argv[i + 1]; i += 2
        elif a == "--workers":
            opts["workers"] = int(argv[i + 1]); i += 2
        elif a == "--out":
            opts["out"] = os.path.abspath(argv[i + 1]); i += 2
        elif a == "--only":
            opts["only"] = set(fe(s.split(":")[0], float(s.split(":")[1])) for s in argv[i + 1].split(",")); i += 2
        elif a == "--grids":
            opts["grids"] = set(int(x) for x in argv[i + 1].split(",")); i += 2
        elif a == "--spectra":
            opts["spectra"] = set(int(x) for x in argv[i + 1].split(",")); i += 2
        elif a == "--kinds":
            opts["kinds"] = set(argv[i + 1].split(",")); i += 2
        elif a == "--retry":
            opts["retry"] = tuple(argv[i + 1].split(",")); i += 2
        else:
            print(__doc__); return 2
    if opts["mode"] == "oracle":
        grids = sorted(opts["grids"]) if opts["grids"] else list(ORACLE_GRIDS)
        return oracle_cell(ocell, grids)
    out = opts["out"]
    if opts["mode"] == "summarize":
        summarize(out)
        return 0
    spectra_dir = os.path.join(out, "spectra")
    cells = build_matrix(spectra_dir, opts["only"], opts["grids"], opts["spectra"], opts["kinds"])
    if opts["mode"] == "plan":
        plan(cells, opts["workers"])
        return 0
    os.makedirs(spectra_dir, exist_ok=True)
    counts = run(cells, out, opts["workers"], opts["retry"])
    summarize(out)
    return 0 if set(counts) <= {"done", "timeout", "unsupported"} else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
