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

Corrective matrix (N4c, `--matrix corrective`; the N4 matrix above stays the default, `--matrix n4`):
  python3 run_all.py --matrix corrective --plan                      cells, counts, GB-hours at the caps, order
  python3 run_all.py --matrix corrective --workers 26 --mem-budget 100 --threads 3 --out ../raw/sweep2
  python3 run_all.py --matrix corrective --summarize --out ../raw/sweep2
  fields gauss, gauss_ch, control2d, generic3d x eps 0.25/0.5/1; G = 16/20/24/28
  orc    run_all.py --oracle-cell field:eps --fd4 --grids <every N used by the (field, eps) cells, incl. 12 for
         spec4 and 32 where a solver uses it>: oracle caches + ceilings oracle_fd, oracle_fd4, oracle_mim
  i1o4   candidate_i.py field:eps:N:i1 --order 4 --direct-max 32 --reuse-lu 0 --save <out>/solutions, N in G
         (+32 for gauss:0.25, gauss_ch:0.25)
  i1     same without --order 4: gauss, gauss_ch, control2d x N in G + 32; generic3d x N in 20, 28
  ii     candidate_ii.py field:eps:N: gauss, gauss_ch, control2d at N = 20; control2d at 16, 24; N = 32 for
         gauss:0.25, gauss_ch:0.25 (lowest priority)
  spec4  candidate_i.py --spectrum M --order 4 field:eps:M:i1 --out <out>/spectra, M = 12, 16
  cons4  candidate_i.py --consistency --order 4 --grids 16,24,28 field:eps
  Scheduler: sum(mem_gb of running) <= --mem-budget, running <= --workers, OMP/OPENBLAS/MKL threads = --threads
  per cell, priority orc > i1o4 > i1 > spec4/cons4 > ii > ii N=32 (larger mem_gb first in a class; the first
  blocked ready cell reserves its memory), wall cap per cell -> `timeout`, non-zero exit -> `failed` + log tail,
  peak RSS per cell (os.wait4), resumable (`done` skipped; --retry default failed,running,pending,interrupted).
  Filters: --only field:eps,...  --grids N,... (REPLACES the solver ladders)  --types orc,i1o4,i1,ii,spec4,cons4
  Launcher tests: --dry-run (every cell runs `true`, or `sleep S` with --dry-sleep S) and --dry-faults (adds a
  fake cell that sleeps past a 2 s cap and one that exits 1).  --merge-sweep DIR (default <out>/../sweep): source
  of the generic3d i1 rows at N = 16, 24 in the summary.
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
def oracle_cell(spec, grids, fd4=False):
    """fd4=True (corrective matrix, `--fd4`): also print the 4th-order ceiling line cand=oracle_fd4."""
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
        if fd4:
            mo4 = M.fd_metrics(psi_or[0], psi_or[1], case["vD"], psi_or, order=4)
            M.print_case(field, eps, N, "oracle_fd4", mo4, t=tl)
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
# corrective matrix (N4c, `--matrix corrective`): cells, memory-budget scheduler, summary
# ------------------------------------------------------------------------------------------------
CFIELDS = ("gauss", "gauss_ch", "control2d", "generic3d")
CGRIDS = (16, 20, 24, 28)
CTYPES = ("orc", "i1o4", "i1", "ii", "spec4", "cons4")
C_I1_FIELDS = ("gauss", "gauss_ch", "control2d")
C_I1_GENERIC = (20, 28)                       # generic3d i1: the other grids exist in raw/sweep/
C_II_FIELDS = ("gauss", "gauss_ch", "control2d")
C_EXT32 = (("gauss", 0.25), ("gauss_ch", 0.25))
C_CONS_GRIDS = (16, 24, 28)
C_SPEC = (12, 16)
# (mem_gb, cap_s) per type and N (task N4c section 3.B.2); N absent from a table -> nearest smaller key, else the
# smallest key (used only by filtered smoke runs, e.g. --grids 12)
C_EST = {
    "i1o4": {16: (1.5, 3600), 20: (3, 7200), 24: (7, 10800), 28: (14, 21600), 32: (28, 36000)},
    "i1": {16: (1, 3600), 20: (1.5, 3600), 24: (2.5, 7200), 28: (5, 14400), 32: (9, 21600)},
    "ii": {16: (1, 7200), 20: (2, 14400), 24: (3, 14400), 32: (4, 36000)},
    "spec4": {12: (1, 1800), 16: (4, 3600)},
    "orc": {None: (2, 7200)},
    "cons4": {None: (2, 7200)},
}
# launch priority class (lower first); within a class larger mem_gb first
C_PRIO = {"orc": 0, "i1o4": 1, "i1": 2, "spec4": 3, "cons4": 3, "ii": 4}
C_PRIO_II32 = 5                                 # ii at N = 32: lowest priority


def c_est(typ, N):
    tab = C_EST[typ]
    if None in tab:
        return tab[None]
    if N in tab:
        return tab[N]
    lower = [k for k in tab if k < N]
    return tab[max(lower)] if lower else tab[min(tab)]


def c_cell_id(typ, field, eps, N=None):
    base = "%s-%s-%g" % (typ, field, eps)
    if N is None:
        return base
    return base + ("-M%d" % N if typ == "spec4" else "-N%d" % N)


def build_corrective(out, only=None, grids=None, types=None, dry_run=False, dry_sleep=0.0, dry_faults=False):
    """Corrective matrix (task N4c 3.B.1).  `grids` (from --grids) REPLACES the solver ladders (i1o4, i1, ii; the
    N = 32 extras are kept only if listed); spec4 and cons4 keep their own grids.  The orc cell of a (field, eps)
    loads every N used by the selected cells of that (field, eps) (solver N, spectrum M, consistency grids), so the
    oracle caches exist before any dependent cell starts (no concurrent cache writes).  dry_run: every cell runs
    `sleep dry_sleep` (`true` if 0) instead of its command; dry_faults adds two fake cells (one sleeps past a 2 s
    cap, one exits 1) to exercise timeout / failure recording."""
    py = sys.executable or "python3"

    def rel(x):                                   # cells run with cwd = this directory
        r = os.path.relpath(x, _HERE)
        return x if r.startswith(os.path.join("..", "..", "..")) else r
    sol = rel(os.path.join(out, "solutions"))
    spd = rel(os.path.join(out, "spectra"))
    cells = []

    def lad(base, extra=()):
        if grids is None:
            return sorted(set(base) | set(extra))
        return sorted(grids)

    def add(typ, field, eps, N, argv, needs, prio=None):
        if types is not None and typ not in types:
            return
        mem, cap = c_est(typ, N)
        cells.append({"id": c_cell_id(typ, field, eps, N), "kind": typ, "field": field, "eps": eps, "N": N,
                      "argv": [py, "-u"] + argv, "needs": sorted(set(needs)), "cap": cap, "mem_gb": float(mem),
                      "prio": C_PRIO[typ] if prio is None else prio})

    for field in CFIELDS:
        for eps in EPSS:
            spec = fe(field, eps)
            if only is not None and spec not in only:
                continue
            ext = (32,) if (field, eps) in C_EXT32 else ()
            n0 = len(cells)
            for N in lad(CGRIDS, ext):
                add("i1o4", field, eps, N, ["candidate_i.py", "%s:%d:i1" % (spec, N), "--order", "4",
                                            "--direct-max", "32", "--reuse-lu", "0", "--save", sol], [N])
            if field in C_I1_FIELDS:
                i1g = lad(CGRIDS, (32,))
            else:
                i1g = lad(C_I1_GENERIC)
            for N in i1g:
                add("i1", field, eps, N, ["candidate_i.py", "%s:%d:i1" % (spec, N), "--direct-max", "32",
                                          "--reuse-lu", "0", "--save", sol], [N])
            if field in C_II_FIELDS:
                iig = [20] + ([16, 24] if field == "control2d" else [])
                for N in lad(iig, ext):
                    add("ii", field, eps, N, ["candidate_ii.py", "%s:%d" % (spec, N)], [N],
                        C_PRIO_II32 if N >= 32 else None)
            for Msp in C_SPEC:
                add("spec4", field, eps, Msp, ["candidate_i.py", "--spectrum", str(Msp), "--order", "4",
                                               "%s:%d:i1" % (spec, Msp), "--out", spd], [Msp])
            add("cons4", field, eps, None, ["candidate_i.py", "--consistency", "--order", "4", "--grids",
                                            ",".join(map(str, C_CONS_GRIDS)), spec], list(C_CONS_GRIDS))
            og = sorted(set(n for c in cells[n0:] for n in c["needs"]))
            if types is None or "orc" in types:
                if not og:
                    og = lad(CGRIDS, ext)
                mem, cap = c_est("orc", None)
                cells.insert(n0, {"id": c_cell_id("orc", field, eps), "kind": "orc", "field": field, "eps": eps,
                                  "N": None, "argv": [py, "-u", "run_all.py", "--oracle-cell", spec, "--fd4",
                                                      "--grids", ",".join(map(str, og))],
                                  "needs": [], "cap": cap, "mem_gb": float(mem), "prio": C_PRIO["orc"]})
    for c in cells:
        c["unsupported"] = None
    if dry_run:
        for c in cells:
            c["real_cmd"] = " ".join(c["argv"][2:])
            c["argv"] = ["sleep", "%g" % dry_sleep] if dry_sleep > 0 else ["true"]
        if dry_faults:
            cells.append({"id": "dryfault-sleep-cap2", "kind": "dryfault", "field": "dry", "eps": 0.0, "N": None,
                          "argv": ["sleep", "30"], "needs": [], "cap": 2, "mem_gb": 1.0, "prio": 9,
                          "unsupported": None, "real_cmd": "(fake: sleeps 30 s, cap 2 s)"})
            cells.append({"id": "dryfault-exit1", "kind": "dryfault", "field": "dry", "eps": 0.0, "N": None,
                          "argv": ["sh", "-c", "echo fake failure; exit 1"], "needs": [], "cap": 60, "mem_gb": 1.0,
                          "prio": 9, "unsupported": None, "real_cmd": "(fake: exits 1)"})
    return cells


def c_check_supported(cells):
    """Check every cell command line against the actual argument parsers (flags present in the scripts' option
    loops).  Returns {type: reason} for unsupported types."""
    flags = {}
    for script in ("candidate_i.py", "candidate_ii.py", "run_all.py"):
        with open(os.path.join(_HERE, script)) as f:
            src = f.read()
        flags[script] = set(re.findall(r'a == "(--[a-z0-9-]+)"', src))
    bad = {}
    for c in cells:
        argv = c["argv"][2:] if c["argv"][:1] != ["sleep"] and c["argv"][:1] != ["true"] else \
            c.get("real_cmd", "").split()
        if not argv or argv[0] not in flags:
            continue
        miss = [a for a in argv[1:] if a.startswith("--") and a not in flags[argv[0]]]
        if miss:
            c["unsupported"] = "flag(s) %s not in %s argument parser" % (",".join(miss), argv[0])
            bad.setdefault(c["kind"], c["unsupported"])
    return bad


def c_order(cells):
    return sorted(cells, key=lambda c: (c["prio"], -c["mem_gb"], c["id"]))


def plan_corrective(cells, workers, budget, threads):
    import platform
    print("run_all --matrix corrective --plan  python=%s numpy=%s host=%s cpus=%s" % (
        platform.python_version(), np.__version__, socket.gethostname(), os.cpu_count()))
    bad = c_check_supported(cells)
    print("workers=%d mem-budget=%g GB threads/cell=%d" % (workers, budget, threads))
    by = {}
    for c in cells:
        by.setdefault(c["kind"], []).append(c)
    print("%-9s %6s %14s %14s %10s" % ("type", "cells", "sum mem GB", "GB-h at cap", "max cap h"))
    tot_cells = 0; tot_gbh = 0.0
    for k in list(CTYPES) + ["dryfault"]:
        if k not in by:
            continue
        cs = by[k]
        gbh = sum(c["mem_gb"] * c["cap"] / 3600.0 for c in cs)
        tot_cells += len(cs); tot_gbh += gbh
        print("%-9s %6d %14.1f %14.1f %10.2f%s" % (k, len(cs), sum(c["mem_gb"] for c in cs), gbh,
                                                   max(c["cap"] for c in cs) / 3600.0,
                                                   "  UNSUPPORTED: " + bad[k] if k in bad else ""))
    print("total cells=%d  estimated memory-hours at the wall caps (upper bound) = %.1f GB-h  -> wall lower bound "
          "at budget %g GB if every cell ran to its cap = %.1f h" % (tot_cells, tot_gbh, budget, tot_gbh / budget))
    big = [c for c in cells if c["mem_gb"] > budget]
    if big:
        print("WARNING: %d cell(s) exceed the memory budget and will run alone: %s" % (
            len(big), ", ".join(c["id"] for c in big)))
    print("unsupported types: %s" % (", ".join("%s (%s)" % kv for kv in sorted(bad.items())) if bad else "none"))
    print("dependency order: orc-<field>-<eps> (oracle caches + ceilings oracle_fd/oracle_fd4/oracle_mim) -> every "
          "i1o4 / i1 / ii / spec4 / cons4 cell of the same (field, eps) (started when the orc cell is done or has "
          "written ORACLE_READY for every N the cell uses)")
    print("launch priority: orc(0) > i1o4(1) > i1(2) > spec4, cons4(3) > ii(4) > ii N=32(5); larger mem_gb first "
          "within a class; a blocked cell reserves its memory (later cells start only if they fit beside it)")
    for c in c_order(cells):
        print("  p%d %-30s mem %5.1f GB cap %6ds needs %-14s %s" % (
            c["prio"], c["id"], c["mem_gb"], c["cap"], c["needs"], c.get("real_cmd") or " ".join(c["argv"][2:])))
    return bad


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def run_corrective(cells, out, workers, budget, threads, retry, poll=2.0):
    """Memory-budget scheduler: sum(mem_gb of running) <= budget and running <= workers; priority order with
    memory reservation for the first blocked ready cell (no starvation of large cells); per-cell wall cap (process
    group killed -> `timeout`); non-zero exit -> `failed` with log tail; peak RSS per cell from os.wait4; resumable
    (cells `done` in the manifest are skipped, statuses in `retry` rerun)."""
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "spectra"), exist_ok=True)
    os.makedirs(os.path.join(out, "solutions"), exist_ok=True)
    gi = os.path.join(out, "solutions", ".gitignore")
    if not os.path.exists(gi):                    # same rule as raw/solutions/: only the 16^3 npz are committed
        with open(gi, "w") as f:
            f.write("# written by run_all.py --matrix corrective: only the 16^3 solutions are committed\n"
                    "*.npz\n!*_16_*.npz\n")
    mpath = os.path.join(out, "manifest.json")
    man = load_manifest(mpath)
    host = socket.gethostname()
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = str(threads)
    env["PYTHONUNBUFFERED"] = "1"
    bad = c_check_supported(cells)
    todo = []
    skipped = 0
    for c in cells:
        rec = man["cells"].get(c["id"])
        if rec is not None and (rec["status"] == "done" or rec["status"] not in retry):
            skipped += 1
            continue
        rec = {"status": "pending", "kind": c["kind"], "field": c["field"], "eps": c["eps"], "N": c["N"],
               "cmd": c.get("real_cmd") or " ".join(c["argv"][2:]), "log": c["id"] + ".txt", "cap_s": c["cap"],
               "mem_gb": c["mem_gb"], "prio": c["prio"]}
        if c["argv"][:1] in (["true"], ["sleep"], ["sh"]):
            rec["dry_run_cmd"] = " ".join(c["argv"])
        man["cells"][c["id"]] = rec
        if c.get("unsupported"):
            rec.update({"status": "unsupported", "reason": c["unsupported"]})
            continue
        todo.append(c)
    man.update({"matrix": "corrective", "host": host, "workers": workers, "mem_budget_gb": budget,
                "threads_per_worker": threads, "python": sys.version.split()[0], "numpy": np.__version__})
    man.setdefault("runs", []).append({"start": utc(), "cells_to_run": len(todo), "skipped_done": skipped})
    save_manifest(man, mpath)
    print("run_all corrective: %d cells to run, %d skipped (done / kept), %d in matrix; workers=%d mem-budget=%g GB "
          "threads/cell=%d host=%s%s" % (len(todo), skipped, len(cells), workers, budget, threads, host,
                                         ("; unsupported: %s" % bad) if bad else ""), flush=True)
    orc_ids = {(c["field"], c["eps"]): c["id"] for c in cells if c["kind"] == "orc"}
    todo = c_order(todo)
    running = {}
    t_start = time.time()
    mem_used = [0.0]
    max_seen = {"running": 0, "mem": 0.0}

    def finish(cid, status, code, maxrss_kb=None, extra=None):
        rec = man["cells"][cid]
        rec["status"] = status
        rec["exit"] = code
        rec["end_utc"] = utc()
        if "t0" in rec:
            rec["elapsed_s"] = round(time.time() - rec.pop("t0"), 1)
        if maxrss_kb is not None:
            rec["peak_rss_gb"] = round(maxrss_kb / 1048576.0, 3)
        if status != "done":
            rec["tail"] = tail(os.path.join(out, rec["log"]))
        if extra:
            rec.update(extra)
        save_manifest(man, mpath)
        print("[%7.0fs] %-8s %s (exit %s, %ss, peak RSS %s GB) running %d/%d mem %.1f/%g GB" % (
            time.time() - t_start, status, cid, code, rec.get("elapsed_s"), rec.get("peak_rss_gb", "-"),
            len(running), workers, mem_used[0], budget), flush=True)

    def stop(signum, frame):  # noqa: ARG001
        for cid, (p, fh, c) in list(running.items()):
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                os.wait4(p.pid, 0)
            except ChildProcessError:
                pass
            fh.write("\nRUN_ALL INTERRUPTED (driver received signal %d); process group killed\n" % signum)
            fh.close()
            del running[cid]
            mem_used[0] -= c["need_gb"]
            finish(cid, "interrupted", None)
        man["runs"][-1]["end"] = utc(); man["runs"][-1]["interrupted"] = signum
        save_manifest(man, mpath)
        sys.exit(130)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    def deps_ok(c):
        if c["kind"] in ("orc", "dryfault"):
            return True
        oid = orc_ids.get((c["field"], c["eps"]))
        if oid is None:
            return True
        st = man["cells"].get(oid, {}).get("status")
        if st == "done":
            return True
        return set(c["needs"]) <= c_oracle_ready(out, c["field"], c["eps"])

    while todo or running:
        # reap (os.wait4 gives the per-child peak RSS)
        for cid in list(running):
            p, fh, c = running[cid]
            rec = man["cells"][cid]
            try:
                pid, wst, ru = os.wait4(p.pid, os.WNOHANG)
            except ChildProcessError:
                pid, wst, ru = p.pid, None, None     # reaped elsewhere: exit status from Popen
            if pid == 0:
                if time.time() - rec["t0"] > c["cap"]:
                    try:
                        os.killpg(p.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    try:
                        _, _, ru = os.wait4(p.pid, 0)
                    except ChildProcessError:
                        ru = None
                    p.returncode = -9
                    fh.write("\nRUN_ALL TIMEOUT after %d s (cell wall cap); process group killed\n" % c["cap"])
                    fh.close()
                    del running[cid]
                    mem_used[0] -= c["need_gb"]
                    finish(cid, "timeout", None, ru.ru_maxrss if ru else None)
                continue
            code = os.waitstatus_to_exitcode(wst) if wst is not None else p.returncode
            p.returncode = code
            fh.close()
            del running[cid]
            mem_used[0] -= c["need_gb"]
            finish(cid, "done" if code == 0 else "failed", code, ru.ru_maxrss if ru else None)
        # failed / timed-out orc -> dependents that cannot get their caches fail explicitly
        for c in list(todo):
            if c["kind"] in ("orc", "dryfault"):
                continue
            oid = orc_ids.get((c["field"], c["eps"]))
            orec = man["cells"].get(oid) if oid else None
            if orec is not None and orec["status"] in ("failed", "timeout", "unsupported", "interrupted"):
                rdy = c_oracle_ready(out, c["field"], c["eps"])
                if not set(c["needs"]) <= rdy:
                    todo.remove(c)
                    with open(os.path.join(out, c["id"] + ".txt"), "w") as fh:
                        fh.write("RUN_ALL NOT STARTED: orc cell %s ended %s before the cache of N=%s existed\n"
                                 % (oid, orec["status"], sorted(set(c["needs"]) - rdy)))
                    finish(c["id"], "failed", None, extra={"reason": "orc dependency %s" % orec["status"]})
        # launch in priority order; the first ready cell that does not fit reserves its memory
        reserved = 0.0
        for c in list(todo):
            if len(running) >= workers:
                break
            if not deps_ok(c):
                continue
            need = min(c["mem_gb"], budget)            # a cell above the budget runs alone
            if mem_used[0] + reserved + need > budget + 1e-9:
                if reserved == 0.0:
                    reserved = need
                continue
            todo.remove(c)
            log = os.path.join(out, c["id"] + ".txt")
            fh = open(log, "w")
            fh.write("RUN_ALL cell=%s host=%s cwd=%s mem_gb=%g cap_s=%d threads=%d cmd=%s\n" % (
                c["id"], host, _HERE, c["mem_gb"], c["cap"], threads, " ".join(c["argv"])))
            fh.flush()
            p = subprocess.Popen(c["argv"], cwd=_HERE, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                 start_new_session=True)
            rec = man["cells"][c["id"]]
            rec.update({"status": "running", "host": host, "t0": time.time(), "start_utc": utc(), "pid": p.pid})
            running[c["id"]] = (p, fh, c)
            c["need_gb"] = need
            mem_used[0] += need
            max_seen["running"] = max(max_seen["running"], len(running))
            max_seen["mem"] = max(max_seen["mem"], mem_used[0])
            save_manifest(man, mpath)
            print("[%7.0fs] start    %s (mem %g GB, cap %d s) running %d/%d mem %.1f/%g GB" % (
                time.time() - t_start, c["id"], c["mem_gb"], c["cap"], len(running), workers, mem_used[0], budget),
                flush=True)
        if running or todo:
            if not running and todo and not any(deps_ok(c) for c in todo):
                for c in list(todo):                  # unreachable: dependency never satisfiable
                    todo.remove(c)
                    finish(c["id"], "failed", None, extra={"reason": "dependency not satisfiable"})
                continue
            time.sleep(poll)
    man["runs"][-1].update({"end": utc(), "wall_s": round(time.time() - t_start, 1),
                            "max_running": max_seen["running"], "max_mem_gb": max_seen["mem"]})
    save_manifest(man, mpath)
    counts = {}
    for c in cells:
        st = man["cells"].get(c["id"], {}).get("status", "missing")
        counts[st] = counts.get(st, 0) + 1
    print("run_all corrective: finished in %.0f s; max running %d (cap %d), max mem %.1f GB (budget %g); statuses %s"
          % (time.time() - t_start, max_seen["running"], workers, max_seen["mem"], budget, counts), flush=True)
    return counts


def c_oracle_ready(out, field, eps):
    s = set()
    for ln in read(os.path.join(out, c_cell_id("orc", field, eps) + ".txt")):
        if ln.startswith("ORACLE_READY "):
            m = re.search(r" N=(\d+) ", ln)
            if m:
                s.add(int(m.group(1)))
    return s


def lsq_slope(Ns, vals):
    """Least-squares slope of log(e) vs log(h), h = 1/N, over the grids with a positive finite value."""
    pts = [(math.log(1.0 / n), math.log(v)) for n, v in zip(Ns, vals)
           if v is not None and math.isfinite(v) and v > 0]
    if len(pts) < 2:
        return None
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    return float(np.polyfit(x, y, 1)[0])


def gap_eval(path):
    """Spectrum statistics for the corrective summary; criterion (4): FAIL iff a consecutive pair of sorted
    relative singular values with r_i < 1e-2 has r_{i+1}/r_i >= 100 (r_i = 0 counts as an infinite gap)."""
    ev = spec_eval(path)
    if ev is None:
        return None
    r = np.sort(np.abs(np.loadtxt(path, comments="#").ravel()))
    r = r / r.max()
    gap4 = None
    for i in range(r.size - 1):
        if r[i] < 1e-2:
            g = r[i + 1] / r[i] if r[i] > 0 else float("inf")
            if g >= 100 and gap4 is None:
                gap4 = (i + 1, g)
    ev["gap4"] = gap4
    ev["verdict4"] = "PASS" if gap4 is None else "FAIL"
    return ev


def c_collect(out, man, merge_dir=None):
    data = {}

    def blk(field, eps):
        return data.setdefault((field, eps), {"ceil": {"oracle_fd": {}, "oracle_fd4": {}, "oracle_mim": {}},
                                              "cand": {}, "cons": None, "spec": [], "status": {}})
    for cid, rec in man.get("cells", {}).items():
        kind, field, eps, N = rec["kind"], rec["field"], rec["eps"], rec["N"]
        if kind == "dryfault":
            continue
        d = blk(field, eps)
        d["status"][cid] = rec["status"]
        lines = read(os.path.join(out, rec["log"]))
        cls = case_lines(lines)
        for c in cls:                                     # ceilings: orc cell first, candidate logs as fallback
            if c["cand"] in d["ceil"]:
                n = int(c["N"])
                if kind == "orc" or n not in d["ceil"][c["cand"]]:
                    d["ceil"][c["cand"]][n] = c
        if kind in ("i1o4", "i1", "ii"):
            names = {"i1o4": ("i1o4", "i1o4_fd2"), "i1": ("i1",), "ii": ("ii",)}[kind]
            for name in names:
                cl = [c for c in cls if c["cand"] == name]
                ent = {"status": rec["status"], "cid": cid, "src": "sweep2", "elapsed": rec.get("elapsed_s")}
                if cl:
                    ent.update(cl[-1])
                    if kind == "ii":
                        m = [ln for ln in lines if ln.startswith("SOLVE ")]
                    else:
                        m = [ln for ln in lines if ln.startswith("EXTRA ")]
                    ent["solver_status"] = (re.search(r"status=(\S+)", m[-1]).group(1) if m else "?")
                    pl = [ln for ln in lines if ln.startswith("PATH ")]
                    ent["path"] = pl[-1].split("path:", 1)[1].strip() if pl else ""
                d["cand"].setdefault(name, {})[N] = ent
        elif kind == "cons4":
            d["cons"] = {"status": rec["status"],
                         "rows": [ln.split(" ", 3)[3] for ln in lines if ln.startswith("CONSISTENCY_ORDER ")]}
        elif kind == "spec4":
            sd = os.path.join(out, "spectra")
            files = []
            for st in ("", "_final_iterate", "_oracle"):
                p = os.path.join(sd, "spectrum_cand_i_i1o4_%s_%g_%d%s.txt" % (field, eps, N, st))
                if os.path.exists(p):
                    files.append((st.strip("_") or "converged", p))
            for st, p in files:
                d["spec"].append({"M": N, "state": st, "file": os.path.relpath(p, out), "ev": gap_eval(p),
                                  "status": rec["status"]})
            if not files:
                d["spec"].append({"M": N, "state": "-", "file": None, "ev": None, "status": rec["status"]})
    # generic3d i1: the grids of G = 16/20/24/28 not run here (16, 24) are read from the N4 sweep (raw/sweep/),
    # marked src=sweep (same discretization; pre-C-i4 e_psi normalization, no e_psi1/e_psi2)
    if merge_dir and os.path.exists(os.path.join(merge_dir, "manifest.json")):
        mm = load_manifest(os.path.join(merge_dir, "manifest.json"))
        for cid, rec in mm.get("cells", {}).items():
            if rec.get("kind") != "i1" or rec.get("field") != "generic3d" or rec.get("N") not in CGRIDS:
                continue
            d = data.get((rec["field"], rec["eps"]))
            if d is None:
                continue
            rows = d["cand"].setdefault("i1", {})
            if rec["N"] in rows:
                continue
            lines = read(os.path.join(merge_dir, rec["log"]))
            cl = [c for c in case_lines(lines) if c["cand"] == "i1"]
            ent = {"status": rec["status"], "cid": cid, "src": "sweep", "elapsed": rec.get("elapsed_s")}
            if cl:
                ent.update(cl[-1])
                m = [ln for ln in lines if ln.startswith("EXTRA ")]
                ent["solver_status"] = re.search(r"status=(\S+)", m[-1]).group(1) if m else "?"
                pl = [ln for ln in lines if ln.startswith("PATH ")]
                ent["path"] = pl[-1].split("path:", 1)[1].strip() if pl else ""
            rows[rec["N"]] = ent
            for c in case_lines(lines):
                if c["cand"] == "oracle_fd" and int(c["N"]) not in d["ceil"]["oracle_fd"]:
                    d["ceil"]["oracle_fd"][int(c["N"])] = c
    return data


C_CEIL = {"i1o4": "oracle_fd4", "i1o4_fd2": "oracle_fd", "i1": "oracle_fd", "ii": "oracle_mim"}


def c_completed(r):
    return r.get("status") == "done" and r.get("e_v") is not None and math.isfinite(r["e_v"])


def c_classify(Ns, e, ec=None):
    """Criteria (1) (with ceiling ec) / (2) (ec None) on the three finest completed grids."""
    if len(Ns) < 3:
        return "INCOMPLETE", "%d completed grid(s)" % len(Ns)
    Ns, e = Ns[-3:], e[-3:]
    o = safe_orders(e, Ns)
    if any(v is None for v in o):
        return "INCOMPLETE", "non-positive value"
    fmt = lambda xs: " ".join("%.2f" % v if v is not None else "-" for v in xs)  # noqa: E731
    if all(v >= 1.8 for v in o):
        return "PASS", "N=%s orders %s" % ("/".join(map(str, Ns)), fmt(o))
    if ec is None:
        return "FAIL", "N=%s orders %s" % ("/".join(map(str, Ns)), fmt(o))
    ec = ec[-3:]
    if any(v is None for v in ec):
        return "INCOMPLETE", "ceiling missing on N=%s" % "/".join(map(str, Ns))
    oc = safe_orders(ec, Ns)
    ratios = [a / b for a, b in zip(e, ec)]
    if all(r <= 1.5 for r in ratios) and all(oc[i] is not None and abs(o[i] - oc[i]) <= 0.15 for i in range(2)):
        return "ceiling-limited", "N=%s orders %s vs ceiling %s, max ratio %.2f" % (
            "/".join(map(str, Ns)), fmt(o), fmt(oc), max(ratios))
    return "FAIL", "N=%s orders %s vs ceiling %s, max ratio %.2f" % ("/".join(map(str, Ns)), fmt(o), fmt(oc),
                                                                     max(ratios))


def summarize_corrective(out, merge_dir=None):
    man = load_manifest(os.path.join(out, "manifest.json"))
    data = c_collect(out, man, merge_dir)
    L = []
    L.append("# SF-29 N4c corrective sweep summary (machine-generated by `scripts/run_all.py --matrix corrective "
             "--summarize`; do not edit)")
    L.append("")
    L.append("Source: cell logs `<cell>.txt` and `manifest.json` in this directory, spectra in `spectra/`%s. Numbers are "
             "copied from the CASE / PATH / EXTRA / SOLVE / CONSISTENCY_ORDER lines (`metrics.parse_case_line`); "
             "orders are `metrics.orders` over consecutive completed grids; slope = least-squares slope of log(e) vs "
             "log(h). No interpretation." % (
                 ("; generic3d `i1` rows marked `sweep` are read from `%s`" % os.path.relpath(merge_dir, out))
                 if merge_dir and os.path.exists(os.path.join(merge_dir, "manifest.json")) else ""))
    L.append("")
    counts = {}
    for rec in man.get("cells", {}).values():
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    L.append("Host `%s`; workers %s, mem budget %s GB, %s threads/cell; python %s, numpy %s. Cell statuses: %s." % (
        man.get("host"), man.get("workers"), man.get("mem_budget_gb"), man.get("threads_per_worker"),
        man.get("python"), man.get("numpy"), ", ".join("%s %d" % kv for kv in sorted(counts.items()))))
    for r in man.get("runs", []):
        L.append("Run: start %s, end %s, wall %s s, cells run %s, skipped %s, max running %s, max mem %s GB." % (
            r.get("start"), r.get("end", "-"), r.get("wall_s", "-"), r.get("cells_to_run"), r.get("skipped_done"),
            r.get("max_running", "-"), r.get("max_mem_gb", "-")))
    L.append("")
    L.append("## Reading rules applied (D-5, task N4c 3.B.3), on the three finest COMPLETED grids of each ladder")
    L.append("")
    L.append("- completed grid: cell status `done` and a CASE line with finite `e_v`.")
    L.append("- (1) `e_v`: PASS iff order >= 1.8 on both pairs; else `ceiling-limited` iff `e_v <= 1.5 ceiling` on "
             "the three grids and `|order - ceiling order| <= 0.15` on both pairs; else FAIL. Ceiling: `oracle_fd4` "
             "(`i1o4`), `oracle_fd` (`i1`, `i1o4_fd2`), `oracle_mim` (`ii`).")
    L.append("- (2) `e_psi`: PASS iff order >= 1.8 on both pairs; else FAIL (no ceiling clause).")
    L.append("- (3) PASS iff final `r_F <= 1e-10` and solver status `converged` on every completed grid.")
    L.append("- (4) PASS iff no consecutive pair of sorted relative singular values with `r_i < 1e-2` has ratio "
             ">= 100, on every converged-state `spec4` spectrum (12^3, 16^3) of the (field, eps); `i1o4` and "
             "`i1o4_fd2` (same solves) only; `i1`/`ii`: no spectrum cell in this matrix (`n/a`).")
    L.append("- `INCOMPLETE`: fewer than three completed grids ((1), (2)), no completed grid ((3)), no converged "
             "spectrum ((4)).")
    L.append("")
    fmt = lambda xs: " ".join("%.2f" % v if v is not None else "-" for v in xs)  # noqa: E731
    classes = []
    for field in CFIELDS:
        for eps in EPSS:
            d = data.get((field, eps))
            if d is None:
                continue
            L.append("## %s, eps = %g" % (field, eps))
            L.append("")
            bad = ["%s: %s" % kv for kv in sorted(d["status"].items()) if kv[1] != "done"]
            L.append("Cells not `done`: %s" % (", ".join("`%s`" % b for b in bad) if bad else "none"))
            L.append("")
            spec_conv = [s for s in d["spec"] if s["ev"] is not None and s["state"] == "converged"]
            for cand in ("i1o4", "i1o4_fd2", "i1", "ii"):
                rows = d["cand"].get(cand)
                if not rows:
                    continue
                ck = C_CEIL[cand]
                ceil = d["ceil"][ck]
                Ns = sorted(rows)
                L.append("### candidate `%s` (ceiling `%s`)" % (cand, ck))
                L.append("")
                L.append("| N | src | status | solver status | r_F | its | e_v | ceiling | e_v/ceiling | e_psi | e_psi1 "
                         "| e_psi2 | min_c | t [s] |")
                L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
                for N in Ns:
                    r = rows[N]; ce = ceil.get(N, {}).get("e_v")
                    ev = r.get("e_v")
                    L.append("| %d | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
                        N, r["src"], r["status"], r.get("solver_status", "-"), fnum(r.get("r_F")),
                        fnum(r.get("its"), "%.0f"), fnum(ev), fnum(ce),
                        fnum(ev / ce if ev is not None and ce else None, "%.2f"), fnum(r.get("e_psi")),
                        fnum(r.get("e_psi1")), fnum(r.get("e_psi2")), fnum(r.get("min_c")), fnum(r.get("t"), "%.0f")))
                L.append("")
                comp = [N for N in Ns if c_completed(rows[N])]
                ev = [rows[N]["e_v"] for N in comp]
                ep = [rows[N].get("e_psi") for N in comp]
                ec = [ceil.get(N, {}).get("e_v") for N in comp]
                if len(comp) >= 2:
                    L.append("Orders over completed N = %s: e_v %s; e_psi %s; ceiling %s. LSQ slope log(e) vs log(h): "
                             "e_v %s; e_psi %s; ceiling %s." % (
                                 "/".join(map(str, comp)), fmt(safe_orders(ev, comp)), fmt(safe_orders(ep, comp)),
                                 fmt(safe_orders(ec, comp)), fnum(lsq_slope(comp, ev), "%.2f"),
                                 fnum(lsq_slope(comp, ep), "%.2f"), fnum(lsq_slope(comp, ec), "%.2f")))
                else:
                    L.append("Orders: fewer than two completed grids (%s)." % ("/".join(map(str, comp)) or "none"))
                L.append("")
                for N in Ns:
                    if rows[N].get("path"):
                        L.append("- PATH N=%d: `%s`" % (N, rows[N]["path"]))
                if any(r.get("path") for r in rows.values()):
                    L.append("")
                c1 = c_classify(comp, ev, ec)
                c2 = c_classify(comp, ep)
                done_rows = [rows[N] for N in Ns if rows[N].get("status") == "done" and "r_F" in rows[N]]
                if not done_rows:
                    c3 = ("INCOMPLETE", "no completed grid")
                else:
                    okc = all(r.get("r_F") is not None and math.isfinite(r["r_F"]) and r["r_F"] <= 1e-10
                              and str(r.get("solver_status", "")).startswith("converged") for r in done_rows)
                    c3 = ("PASS" if okc else "FAIL", "r_F %s; status %s" % (
                        " ".join(fnum(r.get("r_F"), "%.1e") for r in done_rows),
                        " ".join(str(r.get("solver_status")) for r in done_rows)))
                if cand in ("i1o4", "i1o4_fd2"):
                    if not spec_conv:
                        c4 = ("INCOMPLETE", "no converged-state spec4 spectrum")
                    else:
                        f4 = [s for s in spec_conv if s["ev"]["verdict4"] == "FAIL"]
                        c4 = ("FAIL" if f4 else "PASS", "M=%s%s" % (
                            ",".join(str(s["M"]) for s in spec_conv),
                            ("; gap >= 100 at M=%s" % ",".join("%d(after %d, ratio %.3g)" % (
                                s["M"], s["ev"]["gap4"][0], s["ev"]["gap4"][1]) for s in f4)) if f4 else ""))
                else:
                    c4 = ("n/a", "no spectrum cell in this matrix")
                classes.append((field, eps, cand, c1, c2, c3, c4))
            if d["cons"] is not None:
                L.append("### cons4: `i1o4` residual at the oracle labels (status %s)" % d["cons"]["status"])
                L.append("")
                for row in d["cons"]["rows"]:
                    L.append("- `%s`" % row)
                if not d["cons"]["rows"]:
                    L.append("- (no order lines)")
                L.append("")
            if d["spec"]:
                L.append("### spec4: dense spectra of `i1o4` (sorted relative singular values)")
                L.append("")
                L.append("| M | state | n | smallest 3 | <1e-3 | <1e-6 | <1e-10 | largest gap below 1e-2 (ratio @ position: "
                         "r -> r') | first gap >= 100 below 1e-2 | file |")
                L.append("|---|---|---|---|---|---|---|---|---|---|")
                for s in sorted(d["spec"], key=lambda s: (s["M"], s["state"])):
                    ev = s["ev"]
                    if ev is None:
                        L.append("| %d | - | - | - | - | - | - | - | - | (no file; cell %s) |" % (s["M"], s["status"]))
                        continue
                    g = ev["gap"]
                    L.append("| %d | %s | %d | %s | %d | %d | %d | %s @ %d: %s -> %s | %s | `%s` |" % (
                        s["M"], s["state"], ev["n"], " ".join("%.2e" % v for v in ev["small3"]), ev["cnt"][1e-3],
                        ev["cnt"][1e-6], ev["cnt"][1e-10], "%.3g" % g[0], g[1], fnum(g[2], "%.2e"),
                        fnum(g[3], "%.2e"), ("after %d (ratio %.3g)" % ev["gap4"]) if ev["gap4"] else "none",
                        s["file"]))
                L.append("")
    L.append("## D-5 classification per criterion")
    L.append("")
    L.append("| field | eps | candidate | (1) e_v | (2) e_psi | (3) r_F | (4) spectrum |")
    L.append("|---|---|---|---|---|---|---|")
    for field, eps, cand, c1, c2, c3, c4 in classes:
        L.append("| %s | %g | %s | %s | %s | %s | %s |" % (field, eps, cand,
                                                          *("**%s** (%s)" % c for c in (c1, c2, c3, c4))))
    L.append("")
    path = os.path.join(out, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")
    print("run_all: wrote %s (%d (field, eps) blocks, %d classification rows)" % (path, len(data), len(classes)),
          flush=True)
    return path


# ------------------------------------------------------------------------------------------------
def main_corrective(argv):
    opts = {"workers": 26, "budget": 100.0, "threads": 3, "out": os.path.normpath(os.path.join(_HERE, "..", "raw",
                                                                                                "sweep2")),
            "only": None, "grids": None, "types": None, "retry": ("failed", "running", "pending", "interrupted"),
            "mode": "run", "dry": False, "dry_sleep": 0.0, "dry_faults": False, "merge": None}
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--matrix":
            i += 2
        elif a == "--plan":
            opts["mode"] = "plan"; i += 1
        elif a == "--summarize":
            opts["mode"] = "summarize"; i += 1
        elif a == "--workers":
            opts["workers"] = int(argv[i + 1]); i += 2
        elif a == "--mem-budget":
            opts["budget"] = float(argv[i + 1]); i += 2
        elif a == "--threads":
            opts["threads"] = int(argv[i + 1]); i += 2
        elif a == "--out":
            opts["out"] = os.path.abspath(argv[i + 1]); i += 2
        elif a == "--only":
            opts["only"] = set(fe(s.split(":")[0], float(s.split(":")[1])) for s in argv[i + 1].split(",")); i += 2
        elif a == "--grids":
            opts["grids"] = set(int(x) for x in argv[i + 1].split(",")); i += 2
        elif a == "--types":
            opts["types"] = set(argv[i + 1].split(",")); i += 2
            unk = opts["types"] - set(CTYPES)
            if unk:
                print("unknown --types %s (known: %s)" % (",".join(sorted(unk)), ",".join(CTYPES))); return 2
        elif a == "--retry":
            opts["retry"] = tuple(argv[i + 1].split(",")); i += 2
        elif a == "--dry-run":
            opts["dry"] = True; i += 1
        elif a == "--dry-sleep":
            opts["dry_sleep"] = float(argv[i + 1]); i += 2
        elif a == "--dry-faults":
            opts["dry_faults"] = True; i += 1
        elif a == "--merge-sweep":
            opts["merge"] = os.path.abspath(argv[i + 1]); i += 2
        else:
            print(__doc__); return 2
    out = opts["out"]
    merge = opts["merge"] or os.path.normpath(os.path.join(out, "..", "sweep"))
    if opts["mode"] == "summarize":
        summarize_corrective(out, merge)
        return 0
    cells = build_corrective(out, opts["only"], opts["grids"], opts["types"], opts["dry"], opts["dry_sleep"],
                             opts["dry_faults"])
    if opts["mode"] == "plan":
        plan_corrective(cells, opts["workers"], opts["budget"], opts["threads"])
        return 0
    counts = run_corrective(cells, out, opts["workers"], opts["budget"], opts["threads"], opts["retry"])
    summarize_corrective(out, merge)
    return 0 if set(counts) <= {"done", "timeout", "unsupported"} else 1


# ------------------------------------------------------------------------------------------------
def main(argv):
    if "--matrix" in argv:
        mx = argv[argv.index("--matrix") + 1] if argv.index("--matrix") + 1 < len(argv) else ""
        if mx == "corrective":
            return main_corrective(argv)
        if mx != "n4":
            print("--matrix must be n4 (default) or corrective"); return 2
        argv = [a for j, a in enumerate(argv) if not (a == "--matrix" or (j > 0 and argv[j - 1] == "--matrix"))]
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
        elif a == "--fd4":
            opts["fd4"] = True; i += 1
        else:
            print(__doc__); return 2
    if opts["mode"] == "oracle":
        grids = sorted(opts["grids"]) if opts["grids"] else list(ORACLE_GRIDS)
        return oracle_cell(ocell, grids, fd4=opts.get("fd4", False))
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
