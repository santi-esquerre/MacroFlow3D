"""Host runner of one design-pilot job (SF-42 R4).

    python3 -m pilots.run J1|J2|J3 --build <build dir> --out <campaign dir> [--only ID ...] [--dry-run] [--reduce]

For every run of the job (``pilots.design.JOBS``) in order: runs the binary in ``<out>/<job>/<id>/`` with stdout+stderr in ``run.log``,
samples the PID's own GPU memory every second (``nvidia-smi --query-compute-apps``; never the GPU total, the GPU is shared), writes
``RUN.json`` (wall, exit, peak memory, per-realization solver and timing records) and prints one summary line.  A run whose ``RUN.json``
has exit 0 is skipped (restart); a failing run is recorded and the job goes on (its exit is the number of failed runs, capped at 100).
With ``--reduce`` each successful run is reduced with ``python3 -m sf42red reduce`` into ``<out>/<job>/reduced/<id>.csv``
(consumer: weights flux / unif / expI x bases pos / disp, window d in [25, 94]; Par2: the crossing CSVs).
No outcome (alpha) is read or printed here.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import threading
import time

from . import design


def utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gpu_mem_of(pid):
    """Used MiB of this PID on any GPU (None when nvidia-smi is unavailable or the PID is not listed)."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    best = None
    for line in out.splitlines():
        bits = [b.strip() for b in line.split(",")]
        if len(bits) == 2 and bits[0] == str(pid) and bits[1].isdigit():
            best = max(best or 0, int(bits[1]))
    return best


_KEY = re.compile(r"\[[4567]\]|Transport complete|Post-processing")


def launch(cmd, cwd, log_path):
    """Run ``cmd`` to completion with stdout+stderr in ``log_path``.

    Returns (exit code, wall seconds, peak MiB or None, number of memory samples, marks) where ``marks`` = [(seconds since start,
    line)] of the stage lines ("[4]" field .. "[7]" transport, "Transport complete") used for the Par2 stage times."""
    t0 = time.monotonic()
    peak, nsamp, marks = None, 0, []
    proc = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)

    def reader():
        with open(log_path, "wb") as log:
            for raw in iter(proc.stdout.readline, b""):
                log.write(raw)
                line = raw.decode(errors="replace")
                if _KEY.search(line):
                    marks.append((time.monotonic() - t0, line.strip()[:80]))

    th = threading.Thread(target=reader, daemon=True)
    th.start()
    while proc.poll() is None:
        m = gpu_mem_of(proc.pid)
        if m is not None:
            peak = max(peak or 0, m)
            nsamp += 1
        time.sleep(1.0)
    th.join()
    return proc.returncode, time.monotonic() - t0, peak, nsamp, marks


def par2_stage_times(marks):
    """Per realization (list of dicts): field [4]->[5], head solve [5]->[6], velocity [6]->[7], transport [7]->'Transport complete', and
    ``flow_stage`` = [4]->[7] (field + solve + velocity; the analogue of the consumer's stage).  Needs line-buffered stdout."""
    out, cur = [], {}
    for t, line in marks:
        for k, tag in (("t4", "[4]"), ("t5", "[5]"), ("t6", "[6]"), ("t7", "[7]"), ("t8", "Transport complete")):
            if tag in line:
                if k == "t4" and cur:
                    out.append(cur)
                    cur = {}
                cur[k] = t
    if cur:
        out.append(cur)
    res = []
    for c in out:
        if all(k in c for k in ("t4", "t5", "t6", "t7")):
            res.append({"field_s": c["t5"] - c["t4"], "solve_s": c["t6"] - c["t5"], "velocity_s": c["t7"] - c["t6"],
                        "transport_s": (c["t8"] - c["t7"]) if "t8" in c else None, "flow_stage_s": c["t7"] - c["t4"]})
    return res


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def summarize_consumer(out):
    """Per-realization records of a finished consumer run (solver_b.json, timings.json); no estimator values."""
    recs = []
    rdir = os.path.join(out, "realizations")
    for name in sorted(os.listdir(rdir)) if os.path.isdir(rdir) else []:
        d = os.path.join(rdir, name)
        sv, tm = read_json(os.path.join(d, "solver_b.json")), read_json(os.path.join(d, "timings.json"))
        rec = {"r": name, "done": os.path.exists(os.path.join(d, "DONE"))}
        if sv:
            rec.update({k: sv.get(k) for k in ("seed", "iterations", "verdict", "converged", "fallback", "usable", "rel", "balance", "qbar",
                                               "mg_levels", "cg_rtol")})
        if tm:
            rec["seconds"] = tm.get("seconds")
            rec["mem_lines_stage"] = tm.get("mem_lines_stage")
            rec["mem_line_bridge"] = tm.get("mem_line_bridge")
        recs.append(rec)
    return recs


_RE_ITERS = re.compile(r"Converged=(\w+)\s+iters=(\d+)\s+res=([0-9.eE+-]+)")


def summarize_par2(log_path):
    """Par2 stdout: the per-realization head-solve lines (converged flag, PCG iterations) and any timing lines."""
    recs, timing = [], []
    with open(log_path, errors="replace") as f:
        for line in f:
            m = _RE_ITERS.search(line)
            if m:
                recs.append({"converged": m.group(1), "iterations": int(m.group(2))})
            elif re.search(r"(?i)time|seconds| ms\b", line) and len(timing) < 60:
                timing.append(line.rstrip()[:200])
    return recs, timing


def run_one(run, build, root, dry=False):
    d = os.path.join(root, run.job, run.id)
    done = read_json(os.path.join(d, "RUN.json"))
    if done and done.get("exit") == 0:
        print("SKIP_DONE %s" % run.id, flush=True)
        return 0
    os.makedirs(d, exist_ok=True)
    if run.code == "consumer":
        cmd = [os.path.join(build, "lagrangian_ensemble")] + design.consumer_args(run, os.path.join(d, "out"))
    else:
        with open(os.path.join(d, "config.yaml"), "w") as f:
            f.write(design.par2_yaml(run))
        cmd = ["stdbuf", "-oL", os.path.join(build, "macroflow3d_pipeline"), "config.yaml"]   # line-buffered: stage timestamps
    with open(os.path.join(d, "cmd.txt"), "w") as f:
        f.write(" ".join(cmd) + "\n")
    print("RUN %s code=%s grid=%dx%dx%d amp=%s seed0=%d nr=%d M=%d" % (run.id, run.code, run.nx, run.ny, run.ny, run.amp, run.seed0,
                                                                  run.nr, run.M), flush=True)
    if dry:
        print("  " + " ".join(cmd)[:300])
        return 0
    start = utc()
    rc, wall, peak, nsamp, marks = launch(cmd, d, os.path.join(d, "run.log"))
    rec = {"id": run.id, "job": run.job, "code": run.code, "nx": run.nx, "ny": run.ny, "nz": run.ny, "lh": run.lh, "amp": run.amp,
           "seed0": run.seed0, "nr": run.nr, "M": run.M, "start_utc": start, "end_utc": utc(), "wall_s": wall, "exit": rc,
           "peak_gpu_mib_own_pid": peak, "mem_samples": nsamp}
    if run.code == "consumer":
        rec["realizations"] = summarize_consumer(os.path.join(d, "out"))
    else:
        rec["par2_solves"], _ = summarize_par2(os.path.join(d, "run.log"))
        rec["par2_stage_times"] = par2_stage_times(marks)
        dt, nst = design.par2_time(run.nx, run.ny, run.lh)
        rec["par2_dt"], rec["par2_n_steps"] = dt, nst
    with open(os.path.join(d, "RUN.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print("DONE %s exit=%d wall=%.1fs peak_mib=%s" % (run.id, rc, wall, peak), flush=True)
    return rc


def reduce_one(run, root):
    d = os.path.join(root, run.job, run.id)
    outdir = os.path.join(root, run.job, "reduced")
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, run.id + ".csv")
    e = design.ell(run.ny, run.lh)
    cmd = [sys.executable, "-B", "-m", "sf42red", "reduce", "--root", os.path.join(root, run.job), "--out", out,
           "--ell", design.fmt(e), "--x0", design.fmt(5.0 * e)]
    if run.code == "consumer":
        cmd += ["--consumer", os.path.join(d, "out")]
    else:
        cmd += ["--par2"] + [os.path.join(d, "out", "crossings", "realization_%04d_planes.csv" % r) for r in range(run.nr)]
    rc = subprocess.run(cmd).returncode
    print("REDUCE %s exit=%d -> %s" % (run.id, rc, out), flush=True)
    return rc


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pilots.run")
    ap.add_argument("job", choices=sorted(design.JOBS))
    ap.add_argument("--build", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--reduce", action="store_true")
    ap.add_argument("--nr", type=int, default=None, help="realizations per run (J2: default 6; J3: default 8)")
    a = ap.parse_args(argv)
    kw = {"nr": a.nr} if a.nr is not None and a.job in ("J2", "J3") else {}
    runs = design.JOBS[a.job](**kw)
    if a.only:
        runs = [r for r in runs if r.id in a.only]
    root, bad = os.path.abspath(os.path.expanduser(a.out)), 0
    print("JOB_BEGIN %s runs=%d %s" % (a.job, len(runs), utc()), flush=True)
    for r in runs:
        rc = run_one(r, os.path.abspath(os.path.expanduser(a.build)), root, a.dry_run)
        if rc != 0:
            bad += 1
        elif a.reduce and not a.dry_run:
            if reduce_one(r, root) != 0:
                bad += 1
    print("JOB_END %s failed_runs=%d %s" % (a.job, bad, utc()), flush=True)
    return min(bad, 100)


if __name__ == "__main__":
    sys.exit(main())
