"""Campaign driver: gates, resume, engine, final analysis, report and exit code.  Exit codes: see README (never 70, 75, 97, 130)."""
import argparse
import fcntl
import hashlib
import importlib
import os
import signal
import subprocess
import time

from . import determinism, procs
from .background import Background
from .fake import FakeExecutor
from .lanes import Engine, Lane
from .ledger import Ledger
from .plan import PlanError, load_plan, utc_iso
from .scheduler import State

DONE, PARTIAL, STOPPED, ANALYSIS_FAILED, REFUSED, SIGNALLED = 0, 10, 11, 12, 13, 143
MARKERS = ("DONE", "DONE_PARTIAL", "STOPPED", "ANALYSIS_FAILED")
BLOCKED = "ANALYSIS_BLOCKED_UNREDUCED"


def lane_b_class_mib(plan, env=os.environ):
    """Launch threshold of lane B's GPU: plan `lane_b_min_free_mib`, else REMOTE_GPU_MIN_FREE_MIB_2, else REMOTE_GPU_MIN_FREE_MIB (the helper
    does not export them to the job: pass them in the job command or put the number in the plan), else None (every heavy unit)."""
    if "lane_b_min_free_mib" in plan:
        return plan["lane_b_min_free_mib"]
    for k in ("REMOTE_GPU_MIN_FREE_MIB_2", "REMOTE_GPU_MIN_FREE_MIB"):
        if env.get(k, "").strip().isdigit():
            return int(env[k])
    return None


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


def sha256_gate(plan):
    """Files of ``analysis.sha256_list`` that are absent or whose sha256 differs from the plan (empty list = gate passed)."""
    bad = []
    for path, want in plan["analysis"]["sha256_list"].items():
        got = "absent"
        if os.path.isfile(path):
            with open(path, "rb") as f:
                got = hashlib.sha256(f.read()).hexdigest()
        if got != want:
            bad.append((path, got, want))
    return bad


def run_cmd(argv, log_path):
    with open(log_path, "ab") as f:
        return subprocess.run(argv, stdout=f, stderr=subprocess.STDOUT).returncode


def analysis(plan, build_dir, log, runner=run_cmd):
    """Re-verify the sha256 of the analysis files (also checked at start, ``sha256_gate``), then run analysis, recompute and reconcile commands.  -> (ok, detail)."""
    cdir = plan["campaign_dir"]
    for path, got, want in sha256_gate(plan)[:1]:
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


def report(st, eng, status, an, cdir, extra=()):
    ev = [e for e in st.events if not e[2].startswith("PROJECTION")]
    ln = ["SF-43 campaign %s: %s" % (st.plan["campaign"], status), "analysis: %s" % an,
          "stage1_start: %s" % utc_iso(st.stage1_start), "lane_b_released: %s" % st.lane_b_released,
          "gpu_h_used: %.2f of %.2f" % (st.gpu_s / 3600.0, st.plan["budget_gpu_h"]), "stopping: %s" % eng.stopping, "lane_crashes: %d" % eng.crashed, "",
          "%-24s %8s %9s %8s %6s %5s %4s %9s %s" % ("arm", "planned", "completed", "replaced", "failed", "oom", "run", "peak_mib", "halted")]
    for a in st.arms.values():
        ln.append("%-24s %8d %9d %8d %6d %5d %4d %9s %s" % (a.id, len(a.planned) - len(a.dropped), len(a.done), len(a.dropped), len(a.failed), a.oom_total,
                                                           len(a.pending()) + len(a.claimed), "-" if a.peak_mib is None else "%.0f" % a.peak_mib,
                                                           a.halted or ""))
    ln += ["", "retention (raw-array realizations kept / required per arm):"] + ["  %-24s %d / %d" % (k, h, n) for k, (h, n) in eng.retention.items()]
    ln += ["", "UNREDUCED (completed, not reduced: MISSING for the analysis, not the end of a prefix):"]
    ln += ["  %s %s seeds %s" % u for u in eng.unreduced] or ["  none"]
    if eng.bg and eng.bg.state:
        ln += ["", "background jobs:"] + eng.bg.report()
    ln += [""] + list(extra) if extra else []
    ln += ["", "events:"] + ["  %s %s %s" % e for e in ev]
    with open(os.path.join(cdir, "REPORT.txt"), "w") as f:
        f.write("\n".join(ln) + "\n")


def run_campaign(plan, build_dir, ex, *, resume, gpus, clock=time.time, sleep=time.sleep, disk_free_gb=procs.disk_free_gb,
                 gpu_free_mib=procs.nvidia_free_mib, spawn=procs.spawn, threaded=True, log=None, poll_s=2.0, runner=run_cmd, on_engine=None,
                 third_party=procs.third_party_pids):
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
        return _run(plan, build_dir, ex, resume, gpus, clock, sleep, disk_free_gb, gpu_free_mib, spawn, threaded, log, poll_s, runner, on_engine, third_party)
    finally:
        lock.close()


def _run(plan, build_dir, ex, resume, gpus, clock, sleep, disk_free_gb, gpu_free_mib, spawn, threaded, log, poll_s, runner, on_engine, third_party):
    cdir = plan["campaign_dir"]
    build_dir = os.path.abspath(build_dir)  # units run with cwd = their segment directory: no relative path may reach a child
    ledger = Ledger(os.path.join(cdir, "ledger.jsonl"), clock)
    recs = ledger.records()
    if recs:
        first = next((r for r in recs if r["kind"] == "campaign_start"), None)
        if not resume or first is None or first["plan_sha256"] != plan["plan_sha256"]:
            log("REFUSED stale state in %s (%s)" % (cdir, "ledger exists, use --resume" if not resume else "no campaign_start or a different plan"))
            return REFUSED
    else:
        ledger.append({"kind": "campaign_start", "campaign": plan["campaign"], "plan_sha256": plan["plan_sha256"]})
    for m in MARKERS + (BLOCKED,):
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
            lanes.append(Lane("B", gpus[1], True, lane_b_class_mib(plan)))
    log("CAMPAIGN %s lanes=%s stage1_start=%s utc=%s" % (plan["campaign"], ",".join("%s:gpu%s" % (ln.name, ln.gpu) for ln in lanes), utc_iso(st.stage1_start), utc_iso(clock())))
    eng = Engine(st, ledger, ex, build_dir, cdir, lanes, clock=clock, sleep=sleep, disk_free_gb=disk_free_gb, gpu_free_mib=gpu_free_mib,
                 spawn=spawn, worker=procs.Worker(threaded), log=log, poll_s=poll_s, bg=Background(plan, build_dir, cdir, log, clock, sleep),
                 third_party=third_party)
    if on_engine:
        on_engine(eng)
    eng.recover()
    eng.bg.start(eng.rec)
    eng.run()
    if eng.sig:
        ledger.append({"kind": "final", "status": "SIGNAL"})
        log("SIGNALLED sig=%s: units terminated and recorded; no analysis (cancel)" % eng.sig)
        return SIGNALLED
    unreduced = eng.final_reductions()  # every completed segment is reduced before the analysis, or listed
    if unreduced:
        with open(os.path.join(cdir, BLOCKED), "w") as f:
            f.write("segments completed but not reduced; their seeds are MISSING for the analysis:\n")
            f.writelines("%s %s seeds %s\n" % u for u in unreduced)
        log("%s: %d segment(s) unreduced after %d rounds; the analysis runs and must treat their seeds as missing" % (BLOCKED, len(unreduced), 3))
    eng.retention = eng.retention_check()
    complete = st.complete()
    # nothing can run any more and the work is not done (every lane crashed, or no unit produced a seed and the run did not stop on purpose): a wait for the
    # background jobs would only hold the GPUs idle
    dead = not complete and not eng.sig and eng.stopping is None and ((eng.crashed and not any(ln.alive for ln in eng.lanes)) or not any(a.done for a in st.arms.values()))
    if dead:
        log("NO_LIVE_LANE crashed_lanes=%d: background jobs are not waited for" % eng.crashed)
        eng.bg.terminate(eng.rec, "no_live_lane")
    else:
        eng.bg.wait(60.0 * plan["background_wait_minutes"], eng.rec, lambda: eng.sig)
    if eng.sig:  # a signal during the final reductions or the background wait: no analysis (cancel)
        ledger.append({"kind": "final", "status": "SIGNAL"})
        log("SIGNALLED sig=%s: units and background jobs terminated; no analysis (cancel)" % eng.sig)
        return SIGNALLED
    extra = determinism.run(plan, st, cdir, log)
    status = "STOPPED" if eng.stopping in ("budget", "disk") and not complete else "DONE" if complete and not eng.crashed else "DONE_PARTIAL"
    ok, detail = analysis(plan, build_dir, log, runner)
    ledger.append({"kind": "analysis", "ok": ok, "detail": detail})
    if not ok:
        status_file, code = "ANALYSIS_FAILED", ANALYSIS_FAILED
    else:
        status_file, code = status, {"DONE": DONE, "DONE_PARTIAL": PARTIAL, "STOPPED": STOPPED}[status]
    ledger.append({"kind": "final", "status": status, "code": code})
    report(st, eng, status, detail, cdir, extra)
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
    badsha = sha256_gate(plan)  # at START (also on --resume): a frozen-file mismatch must not surface after two days
    if badsha:
        log("REFUSED analysis file sha256 mismatch: " + "; ".join(p for p, _, _ in badsha))
        return REFUSED
    if not gpus:
        log("REFUSED CUDA_VISIBLE_DEVICES is empty (the helper exports the granted GPUs)")
        return REFUSED
    mod, _, cls = plan["executor"].partition(":")  # default sf43camp.runner_executor:RunnerExecutor
    try:
        ex = getattr(importlib.import_module(mod), cls)(plan, a.build_dir)
    except (ImportError, AttributeError, TypeError) as e:
        log("REFUSED executor %s not available: %s" % (plan["executor"], e))
        return REFUSED

    def on_engine(eng):
        def h(sig, _frm):
            eng.sig = sig
        signal.signal(signal.SIGTERM, h)
        signal.signal(signal.SIGINT, h)

    return run_campaign(plan, a.build_dir, ex, resume=a.resume, gpus=gpus, log=log, poll_s=plan.get("poll_s", 2.0), on_engine=on_engine)

