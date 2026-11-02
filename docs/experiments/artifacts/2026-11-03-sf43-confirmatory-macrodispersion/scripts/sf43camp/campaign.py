"""Campaign driver: gates, resume, engine, final analysis, report and exit code.  Exit codes: see README (never 70, 75, 97, 130)."""
import argparse
import fcntl
import hashlib
import importlib
import os
import signal
import subprocess
import time

from . import procs
from .fake import FakeExecutor
from .lanes import Engine, Lane
from .ledger import Ledger
from .plan import PlanError, load_plan, utc_iso
from .scheduler import State

DONE, PARTIAL, STOPPED, ANALYSIS_FAILED, REFUSED, SIGNALLED = 0, 10, 11, 12, 13, 143
MARKERS = ("DONE", "DONE_PARTIAL", "STOPPED", "ANALYSIS_FAILED")


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def md5_gate(plan, build_dir):
    """Problems of the binaries named in the plan (empty list = gate passed)."""
    bad = []
    for name, want in plan["binaries"].items():
        p = os.path.join(build_dir, name)
        got = md5_file(p) if os.path.isfile(p) else "absent"
        if got != want:
            bad.append("%s: md5 %s != plan %s" % (name, got, want))
    return bad


def run_cmd(argv, log_path):
    with open(log_path, "ab") as f:
        return subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT).returncode


def analysis(plan, build_dir, log, runner=run_cmd):
    """Verify the sha256 of the analysis files, then run analysis, recompute and reconcile commands.  -> (ok, detail)."""
    cdir = plan["campaign_dir"]
    for path, want in plan["analysis"]["sha256_list"].items():
        got = "absent"
        if os.path.isfile(path):
            with open(path, "rb") as f:
                got = hashlib.sha256(f.read()).hexdigest()
        if got != want:
            return False, "sha256 mismatch %s (%s != %s)" % (path, got, want)
    for name in ("analysis", "recompute", "reconcile"):
        if name in plan:
            argv = [x.replace("{campaign_dir}", cdir).replace("{build_dir}", build_dir) for x in plan[name]["command"]]
            try:
                rc = runner(argv, os.path.join(cdir, "%s.log" % name))
            except OSError as e:
                rc = -1
                log("ANALYSIS_STEP %s could not start: %s" % (name, e))
            log("ANALYSIS_STEP %s rc=%d" % (name, rc))
            if rc != 0:
                return False, "%s rc=%d" % (name, rc)
    return True, "ok"


def report(st, eng, status, an, cdir):
    ev = [e for e in st.events if not e[2].startswith("PROJECTION")]
    ln = ["SF-43 campaign %s: %s" % (st.plan["campaign"], status), "analysis: %s" % an,
          "stage1_start: %s" % utc_iso(st.stage1_start), "lane_b_released: %s" % st.lane_b_released,
          "gpu_h_used: %.2f of %.2f" % (st.gpu_s / 3600.0, st.plan["budget_gpu_h"]), "stopping: %s" % eng.stopping, "lane_crashes: %d" % eng.crashed, "",
          "%-24s %8s %9s %8s %6s %5s %4s %s" % ("arm", "planned", "completed", "replaced", "failed", "oom", "run", "halted")]
    for a in st.arms.values():
        ln.append("%-24s %8d %9d %8d %6d %5d %4d %s" % (a.id, len(a.planned) - len(a.dropped), len(a.done), len(a.dropped), len(a.failed), a.oom_total,
                                                        len(a.pending()) + len(a.claimed), a.halted or ""))
    ln += ["", "events:"] + ["  %s %s %s" % e for e in ev]
    with open(os.path.join(cdir, "REPORT.txt"), "w") as f:
        f.write("\n".join(ln) + "\n")


def run_campaign(plan, build_dir, ex, *, resume, gpus, clock=time.time, sleep=time.sleep, disk_free_gb=procs.disk_free_gb,
                 gpu_free_mib=procs.nvidia_free_mib, spawn=procs.spawn, threaded=True, log=None, poll_s=2.0, runner=run_cmd, on_engine=None):
    log = log or (lambda m: print(m, flush=True))
    cdir = plan["campaign_dir"]
    os.makedirs(cdir, exist_ok=True)
    lock = open(os.path.join(cdir, "LOCK"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        log("REFUSED another launcher holds %s/LOCK" % cdir)
        lock.close()
        return REFUSED
    try:
        return _run(plan, build_dir, ex, resume, gpus, clock, sleep, disk_free_gb, gpu_free_mib, spawn, threaded, log, poll_s, runner, on_engine)
    finally:
        lock.close()


def _run(plan, build_dir, ex, resume, gpus, clock, sleep, disk_free_gb, gpu_free_mib, spawn, threaded, log, poll_s, runner, on_engine):
    cdir = plan["campaign_dir"]
    ledger = Ledger(os.path.join(cdir, "ledger.jsonl"), clock)
    recs = ledger.records()
    if recs:
        first = next((r for r in recs if r["kind"] == "campaign_start"), None)
        if not resume or first is None or first["plan_sha256"] != plan["plan_sha256"]:
            log("REFUSED stale state in %s (%s)" % (cdir, "ledger exists, use --resume" if not resume else "no campaign_start or a different plan"))
            return REFUSED
    else:
        ledger.append({"kind": "campaign_start", "campaign": plan["campaign"], "plan_sha256": plan["plan_sha256"]})
    for m in MARKERS:
        if os.path.exists(os.path.join(cdir, m)):
            os.remove(os.path.join(cdir, m))
    st = State(plan)
    for r in recs:
        st.apply(r)
    if st.stage1_start is None:
        st.apply(ledger.append({"kind": "stage1_start"}))
    ledger.append({"kind": "session_start", "granted": len(gpus), "resume": bool(recs)})
    lanes = [Lane("A", gpus[0], False)]
    if len(gpus) >= 2:
        if st.lane_b_released or clock() >= st.stage1_start + 3600.0 * plan["stage1_max_hours"]:
            log("RESUME_SINGLE_LANE stage 1 is over (released=%s); %d GPUs granted, lane A only: relaunch with one GPU" % (st.lane_b_released, len(gpus)))
        else:
            lanes.append(Lane("B", gpus[1], True))
    log("CAMPAIGN %s lanes=%s stage1_start=%s utc=%s" % (plan["campaign"], ",".join("%s:gpu%s" % (ln.name, ln.gpu) for ln in lanes), utc_iso(st.stage1_start), utc_iso(clock())))
    eng = Engine(st, ledger, ex, build_dir, cdir, lanes, clock=clock, sleep=sleep, disk_free_gb=disk_free_gb, gpu_free_mib=gpu_free_mib,
                 spawn=spawn, worker=procs.Worker(threaded), log=log, poll_s=poll_s)
    if on_engine:
        on_engine(eng)
    eng.recover()
    eng.run()
    if eng.sig:
        ledger.append({"kind": "final", "status": "SIGNAL"})
        log("SIGNALLED sig=%s: units terminated and recorded; no analysis (cancel)" % eng.sig)
        return SIGNALLED
    complete = st.complete()
    status = "STOPPED" if eng.stopping in ("budget", "disk") and not complete else "DONE" if complete and not eng.crashed else "DONE_PARTIAL"
    ok, detail = analysis(plan, build_dir, log, runner)
    ledger.append({"kind": "analysis", "ok": ok, "detail": detail})
    if not ok:
        status_file, code = "ANALYSIS_FAILED", ANALYSIS_FAILED
    else:
        status_file, code = status, {"DONE": DONE, "DONE_PARTIAL": PARTIAL, "STOPPED": STOPPED}[status]
    ledger.append({"kind": "final", "status": status, "code": code})
    report(st, eng, status, detail, cdir)
    with open(os.path.join(cdir, status_file), "w") as f:
        f.write(utc_iso(clock()) + "\n")
    log("FINAL status=%s analysis=%s exit=%d gpu_h=%.2f utc=%s" % (status, detail, code, st.gpu_s / 3600.0, utc_iso(clock())))
    return code


def dry_run(plan, ngpu, log):
    """Simulate the whole plan on a virtual clock with the plan's est_seconds (no process, no file, no ledger)."""
    t = [time.time()]

    def sleep(s):
        t[0] += s

    ex = FakeExecutor(plan, lambda: t[0])
    st = State(plan)
    led = Ledger(None, lambda: t[0])
    st.apply(led.append({"kind": "stage1_start"}))
    lanes = [Lane("A", "g0", False)] + ([Lane("B", "g1", True)] if ngpu >= 2 else [])
    eng = Engine(st, led, ex, "<build_dir>", plan["campaign_dir"], lanes, clock=lambda: t[0], sleep=sleep, disk_free_gb=lambda p: 1e9,
                 gpu_free_mib=lambda g: None, spawn=ex.spawn, worker=procs.Worker(False), log=lambda m: log("DRYRUN " + m), poll_s=60.0, dry=True)
    t0 = t[0]
    eng.run()
    rel = next((r["t"] for r in led.records() if r["kind"] == "lane_release"), None)
    log("DRYRUN_SUMMARY lanes=%d complete=%s stopping=%s stage1_h=%.1f stage2_h=%.1f total_h=%.1f gpu_h=%.1f budget_gpu_h=%s end=%s cutoff=%s"
        % (len(lanes), st.complete(), eng.stopping, ((rel or t[0]) - t0) / 3600.0, (t[0] - (rel or t[0])) / 3600.0 if rel else 0.0, (t[0] - t0) / 3600.0,
           st.gpu_s / 3600.0, plan["budget_gpu_h"], utc_iso(t[0]), utc_iso(plan["cutoff"])))
    return DONE


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sf43camp")
    ap.add_argument("build_dir")
    ap.add_argument("cmd", choices=["all"])
    ap.add_argument("--plan", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--artifact", default=".")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:  # usage error -> refused (argparse would exit 2)
        return REFUSED if e.code else 0

    def log(m):
        print(m, flush=True)

    try:
        plan = load_plan(a.plan or os.path.join(a.artifact, "plan.json"), a.smoke, time.time())
    except (PlanError, OSError) as e:
        log("REFUSED invalid plan: %s" % e)
        return REFUSED
    gpus = [g for g in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if g.strip()]
    if a.dry_run:
        return dry_run(plan, int(os.environ.get("REMOTE_GPU_COUNT_GRANTED", "2")), log)
    bad = md5_gate(plan, a.build_dir)
    if bad:
        log("REFUSED binaries do not match the plan: " + "; ".join(bad))
        return REFUSED
    if not gpus:
        log("REFUSED CUDA_VISIBLE_DEVICES is empty (the helper exports the granted GPUs)")
        return REFUSED
    mod, _, cls = plan.get("executor", "sf43exec:Executor").partition(":")
    try:
        ex = getattr(importlib.import_module(mod), cls)(plan, a.build_dir)
    except (ImportError, AttributeError) as e:
        log("REFUSED executor %s not available: %s" % (plan.get("executor", "sf43exec:Executor"), e))
        return REFUSED

    def on_engine(eng):
        def h(sig, _frm):
            eng.sig = sig
        signal.signal(signal.SIGTERM, h)
        signal.signal(signal.SIGINT, h)

    return run_campaign(plan, a.build_dir, ex, resume=a.resume, gpus=gpus, log=log, poll_s=plan.get("poll_s", 2.0), on_engine=on_engine)

