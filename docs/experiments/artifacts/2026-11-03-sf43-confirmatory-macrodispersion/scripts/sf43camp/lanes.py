"""The engine: lanes pull units from the queue, run them as subprocesses, classify, and apply the failure / guard / re-sizing rules.

One thread polls the lanes' subprocesses (injectable clock and sleep make 30-hour logic instantaneous in tests); reduction and pruning
run in a background Worker.  Lane A = first granted GPU (heavy units first, then light); lane B = second GPU (heavy units only; leaves
when the heavy queue is empty or when the next heavy unit would end after the stage-1 cap).
"""
import os
import traceback

from . import guards
from .plan import utc_iso

HB_S, PROJECT_S, PRUNE_S = 300.0, 3600.0, 1800.0  # heartbeat, projection log, prune pass (raw output 25-57 GB/h: backlog stays under ~30 GB + reduction lag)
REDUCE_BACKOFF_S = (60.0, 300.0, 900.0)  # a failed reduction is retried 3 times within the session, after these waits
FINAL_ROUNDS = 3  # rounds of "re-submit and wait for every unreduced segment" before the final analysis


class Lane:
    def __init__(self, name, gpu, heavy_only, class_mib=None):
        self.name, self.gpu, self.heavy_only = name, gpu, heavy_only
        self.class_mib = class_mib  # lane B: heavy units with min_free_mib <= this are its class (None = every heavy unit)
        self.unit = self.proc = None
        self.alive = self.held = True  # alive: may claim; held: its GPU is still held by the job
        self.nothing = False  # last pick found nothing to claim
        self.retry_at = 0.0


class Engine:
    def __init__(self, st, ledger, ex, build_dir, cdir, lanes, *, clock, sleep, disk_free_gb, gpu_free_mib, spawn, worker, log, poll_s=2.0, dry=False,
                 bg=None):
        self.st, self.ledger, self.ex, self.build_dir, self.cdir = st, ledger, ex, build_dir, cdir
        self.lanes, self.clock, self.sleep, self.disk_free_gb, self.gpu_free_mib = lanes, clock, sleep, disk_free_gb, gpu_free_mib
        self.spawn, self.worker, self.log, self.poll_s, self.dry, self.bg = spawn, worker, log, poll_s, dry, bg
        self.pin_logged = set()
        self.retention = {}
        self.unreduced = []  # [(arm, segment, seeds)] still unreduced after the final rounds
        self.plan = st.plan
        self.cap = st.stage1_start + 3600.0 * self.plan["stage1_max_hours"]  # epoch end of stage 1 (st.stage1_start is set by the caller)
        self.stopping = None  # 'cutoff' | 'budget' | 'disk'
        self.sig = None
        self.disk_pressure = False
        self.crashed = 0
        self.last = self.clock()
        self.next_hb = self.next_proj = self.next_prune = 0.0

    def recover(self):
        """Units claimed by a launcher that died: close them with the seeds the executor finds complete (they are re-run from the
        first incomplete seed into a new segment)."""
        for uid, c in sorted(self.st.open.items()):
            a = self.st.arms[c["arm"]]
            seg = os.path.join(self.cdir, c["segment"])
            done = sorted(s for s in self.ex.completed(a.spec, seg) if s in c["seeds"])
            det = self.details(a, c["seeds"], seg, os.path.join(seg, "run.log"), None)
            self.rec(kind="unit_end", unit=uid, arm=a.id, lane=c["lane"], gpu=c["gpu"], cls="recovered", completed=done, failed_seed=None,
                     failed_seeds=[], next_seed=next((s for s in c["seeds"] if s not in done), None), seconds={},
                     verdicts={str(s): v for s, v in det.get("verdicts", {}).items() if s in done}, peak_mib=det.get("peak_mib"), wall=None,
                     segment=c["segment"], segment_seed0=c["seeds"][0])
            self.seed_records(a, det, c["segment"], uid, c["seeds"][0], done, [])
            self.log("RECOVERED unit=%d arm=%s completed=%d of %d" % (uid, a.id, len(done), len(c["seeds"])))

    def details(self, a, seeds, seg, log_path, rc):
        """The executor's classification of a unit that was killed (verdicts, solver numbers, peak memory of the seeds it did finish); {}
        if it cannot say.  Only these details are used: the class of a forced end is the engine's."""
        try:
            with open(log_path, errors="replace") as f:
                f.seek(max(0, os.fstat(f.fileno()).st_size - 4000000))
                text = f.read()
        except OSError:
            text = ""
        try:
            return self.ex.classify(a.spec, seeds, seg, rc, text)
        except Exception:
            return {}

    def seed_records(self, a, c, rel, uid, seed0, done, bad):
        """One ledger record per seed with a solve (kind `seed`): the contract of sf43ana's loader and of reconcile43."""
        for sd in done + bad:
            v = c.get("solver", {}).get(sd, {})
            self.rec(**{"kind": "seed", "arm": a.id, "group": a.spec["group"], "segment": os.path.basename(rel), "segment_path": rel, "seed": sd,
                        "seed0": seed0, "unit": uid, "verdict": c.get("verdicts", {}).get(sd) or v.get("verdict"),
                        "class": "ok" if sd in done else "unusable_solve", "iterations": v.get("iterations"), "rel": v.get("rel"),
                        "balance": v.get("balance"), "max_div": v.get("max_div"), "field_seed": v.get("field_seed")})

    # records ------------------------------------------------------------------------------------------------------------
    def rec(self, **r):
        r = self.ledger.append(r)
        self.st.apply(r)
        return r

    def event(self, text, kind="guard"):
        self.log(text)
        self.rec(kind=kind, text=text)

    # main loop ----------------------------------------------------------------------------------------------------------
    def run(self):
        for a in self.st.arms.values():  # reductions that a kill interrupted
            for s in a.segments:
                if not s["reduced"]:
                    self.submit_reduce(a, s["segment"], s["seeds"])
        while True:
            now = self.clock()
            self.account(now)
            self.reap(now)
            self.collect()
            if self.bg:
                self.bg.poll(self.rec)
            if self.sig:
                self.kill_all("terminated")
                return
            if now >= self.plan["cutoff"]:
                self.stop("cutoff", "cutoff reached: no new units")
                if now >= self.plan["cutoff"] + 60.0 * self.plan["grace_minutes"]:
                    self.kill_all("terminated")
            self.maintain(now)
            if not self.stopping:
                self.claims(now)
            busy = [ln for ln in self.lanes if ln.unit]
            if not busy and (self.stopping or all(ln.nothing for ln in self.lanes if ln.alive)):
                break
            self.sleep(self.poll_s)
        self.worker.flush()
        self.collect()
        self.account(self.clock())
        self.rec(kind="hb", gpu_s=self.st.gpu_s)

    def account(self, now):
        self.st.gpu_s += sum(1 for ln in self.lanes if ln.held) * (now - self.last)
        self.last = now

    def stop(self, reason, text):
        if not self.stopping:
            self.stopping = reason
            self.event("STOP reason=%s %s" % (reason, text))

    # claim ---------------------------------------------------------------------------------------------------------------
    def claims(self, now):
        self.st.lane_names = {ln.name for ln in self.lanes if ln.alive}
        for a in self.st.arms.values():  # a pin to a lane that does not exist (any more) is ignored
            pin = a.spec.get("lane")
            if pin and pin not in self.st.lane_names and a.id not in self.pin_logged and a.pending():
                self.pin_logged.add(a.id)
                self.event("LANE_PIN_IGNORED arm=%s lane=%s (lanes: %s)" % (a.id, pin, ",".join(sorted(self.st.lane_names))))
        for ln in self.lanes:
            if ln.heavy_only and ln.alive and not ln.unit and now >= self.cap:
                self.release(ln)  # the stage-1 cap: lane B leaves even if it was waiting for memory or an OOM back-off
                continue
            if ln.unit or not ln.alive or ln.retry_at > now or self.stopping:
                continue
            try:
                u, wake = self.st.pick(ln, now, lambda ln=ln: self.gpu_free_mib(ln.gpu), self.cap if ln.heavy_only else None)
                ln.nothing = u is None and wake is None
                if u is None:
                    if ln.nothing and ln.heavy_only:
                        self.release(ln)
                    elif wake:
                        ln.retry_at = wake
                elif self.guards(u, now):
                    self.launch(ln, u, now)
            except Exception:
                self.crash(ln, traceback.format_exc())

    def unit_h(self, u):
        return self.st.est(self.st.arms[u.arm], len(u.seeds)) / 3600.0

    def guards(self, u, now):
        a = self.st.arms[u.arm]
        infl = sum(max(0.0, ln.t0 + ln.est - now) for ln in self.lanes if ln.unit) / 3600.0
        used = self.st.gpu_s / 3600.0
        if not guards.budget_ok(used, infl, self.unit_h(u), self.plan["budget_gpu_h"]):
            self.stop("budget", "used=%.2f in_flight=%.2f unit=%.2f budget=%.2f GPU-h" % (used, infl, self.unit_h(u), self.plan["budget_gpu_h"]))
            return False
        raw = self.ex.expected_raw_gb(a.spec, len(u.seeds))
        mn = self.plan["disk_min_free_gb"]
        if self.disk_pressure or not guards.disk_ok(self.disk_free_gb(self.cdir), mn, raw):
            self.worker.flush()
            self.collect()
            self.prune_all()
            self.worker.flush()
            self.collect()
            free = self.disk_free_gb(self.cdir)
            self.event("DISK_GUARD free_gb=%.1f need_gb=%.1f after pruning" % (free, mn + raw))
            if not guards.disk_ok(free, mn, raw):
                self.stop("disk", "free %.1f GB < %.1f GB + %.1f GB" % (free, mn, raw))
                return False
        self.disk_pressure = False
        return True

    def launch(self, ln, u, now):
        a = self.st.arms[u.arm]
        uid = self.st.unit_n + 1
        rel = "segments/%s/u%06d_s%d" % (a.id, uid, u.seeds[0])
        seg = os.path.join(self.cdir, rel)
        cmd = self.ex.command(a.spec, u.seeds, seg, self.build_dir)
        self.rec(kind="claim", unit=uid, arm=a.id, seeds=u.seeds, lane=ln.name, gpu=ln.gpu, segment=rel)
        ln.unit, ln.uid, ln.seg, ln.rel, ln.t0, ln.est, ln.cmd = u, uid, seg, rel, now, self.st.est(a, len(u.seeds)), cmd
        if not self.dry:
            os.makedirs(seg, exist_ok=True)
            os.makedirs(os.path.dirname(cmd["log_path"]), exist_ok=True)
            for p, text in cmd.get("files", {}).items():
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w") as f:
                    f.write(text)
        self.log("SF43_UNIT lane=%s gpu=%s arm=%s seed0=%d nr=%d segment=%s utc=%s" % (ln.name, ln.gpu, a.id, u.seeds[0], len(u.seeds), rel, utc_iso(now)))
        ln.proc = self.spawn(cmd, ln.gpu)

    # unit end ------------------------------------------------------------------------------------------------------------
    def reap(self, now):
        for ln in self.lanes:
            if ln.unit and ln.proc.poll() is not None:
                try:
                    self.end(ln, now)
                except Exception:
                    self.crash(ln, traceback.format_exc())

    def kill_all(self, cls):
        if self.bg:
            self.bg.terminate(self.rec)
        for ln in self.lanes:
            if ln.unit:
                ln.proc.terminate()
                self.end(ln, self.clock(), cls)

    def end(self, ln, now, forced=None):
        u, a = ln.unit, self.st.arms[ln.unit.arm]
        seeds = u.seeds
        if forced:
            det = self.details(a, seeds, ln.seg, ln.cmd["log_path"], ln.proc.poll())
            c = {"cls": forced, "completed_seeds": self.ex.completed(a.spec, ln.seg), "verdicts": det.get("verdicts", {}), "solver": det.get("solver", {}),
                 "peak_mib": det.get("peak_mib")}
        else:
            try:
                with open(ln.cmd["log_path"], errors="replace") as f:
                    f.seek(max(0, os.fstat(f.fileno()).st_size - 4000000))
                    text = f.read()
            except OSError:
                text = ""
            c = self.ex.classify(a.spec, seeds, ln.seg, ln.proc.poll(), text)
        done = sorted(s for s in c.get("completed_seeds", []) if s in seeds)
        nxt = next((s for s in seeds if s not in done), None)
        cls = c["cls"]
        if cls == "ok" and nxt is not None:
            cls = "other_error"
        if cls == "other_error" and nxt is None:
            cls = "ok"
        fs = c.get("failed_seed")
        if cls == "unusable_solve" and fs is None:
            fs = nxt
        bad = sorted(set(s for s in c.get("failed_seeds", []) if s in seeds and s not in done) | ({fs} if cls == "unusable_solve" and fs is not None else set()))
        again = cls == "other_error" and nxt in a.other_seen
        r = self.rec(kind="unit_end", unit=ln.uid, arm=a.id, lane=ln.name, gpu=ln.gpu, cls=cls, completed=done, failed_seed=fs, failed_seeds=bad, next_seed=nxt,
                     seconds={str(s): v for s, v in c.get("seconds", {}).items() if s in done},
                     verdicts={str(s): v for s, v in c.get("verdicts", {}).items() if s in seeds}, peak_mib=c.get("peak_mib"),
                     wall=now - ln.t0, segment=ln.rel, segment_seed0=seeds[0])
        self.log("SF43_UNIT_END lane=%s gpu=%s arm=%s seed0=%d nr=%d segment=%s cls=%s completed=%d seconds=%.1f utc=%s"
                 % (ln.name, ln.gpu, a.id, seeds[0], len(seeds), ln.rel, cls, len(done), now - ln.t0, r["utc"]))
        ln.unit = ln.proc = None
        self.seed_records(a, c, ln.rel, ln.uid, seeds[0], done, bad)
        if done:
            self.submit_reduce(a, ln.rel, done)
        fl = self.plan["failure"]
        if cls == "oom":
            delay = guards.oom_backoff_s(a.oom_count.get(nxt, 1))
            if nxt is not None:
                a.not_before[nxt] = now + delay
            self.event("OOM arm=%s seed0=%d retry_in_min=%d (not a failure)" % (a.id, seeds[0], delay // 60))
        elif cls == "unusable_solve":
            for sd in bad:
                self.unusable(a, sd)
        elif cls == "config_error":
            self.halt(a, "config_error")
        elif cls == "other_error" and again:
            self.halt(a, "other_error twice at seed %s" % nxt)
        elif cls == "disk_full":
            self.disk_pressure = True
            self.event("DISK_FULL arm=%s seed0=%d" % (a.id, seeds[0]))
        if a.attempted >= fl["min_for_rate"] and a.attempted and len(a.failed) / a.attempted > fl["halt_rate"]:
            self.halt(a, "failure rate %d/%d > %s" % (len(a.failed), a.attempted, fl["halt_rate"]))

    def halt(self, a, why):
        if not a.halted:
            self.rec(kind="arm_halt", arm=a.id, reason=why)
            self.log("ARM_HALT arm=%s %s" % (a.id, why))

    def unusable(self, a, s):
        g = self.st.groups[a.spec["group"]]
        if s in g.repl.values():
            return
        if g.next_reserve >= g.res["count"]:
            self.event("RESERVES_EXHAUSTED group=%s seed=%s arm=%s" % (g.id, s, a.id))
            return
        r = g.res["start"] + g.next_reserve
        self.rec(kind="replacement", group=g.id, seed=s, replaced_by=r, arm=a.id)
        self.log("REPLACEMENT group=%s seed=%d -> %d (unusable solve in %s)" % (g.id, s, r, a.id))

    def release(self, ln):
        ln.alive = ln.held = False
        self.rec(kind="lane_release", lane=ln.name, gpu=ln.gpu)
        self.log("LANE_B_RELEASED gpu=%s utc=%s" % (ln.gpu, utc_iso(self.clock())))

    def crash(self, ln, tb):
        self.crashed += 1
        ln.alive = False
        if ln.unit and ln.proc:
            ln.proc.terminate()
        self.event("LANE_CRASH lane=%s gpu=%s %s" % (ln.name, ln.gpu, tb.strip().splitlines()[-1]), "lane_crash")
        self.log(tb)
        if ln.unit:
            try:
                self.end(ln, self.clock(), "crashed")
            except Exception:
                ln.unit = ln.proc = None

    # background work -----------------------------------------------------------------------------------------------------
    def seg_entry(self, a, rel):
        return next(x for x in a.segments if x["segment"] == rel)

    def submit_reduce(self, a, rel, seeds):
        seg = os.path.join(self.cdir, rel)
        out = os.path.join(self.cdir, "reduced", a.id, os.path.basename(rel))
        e = self.seg_entry(a, rel)
        e["reducing"], e["tries"] = True, e.get("tries", 0) + 1
        self.worker.submit(("reduce", a.id, rel, list(seeds)), lambda: self.ex.reduce(a.spec, seg, out))

    def prune_all(self):
        k = self.plan["min_retained"]
        for a in self.st.arms.values():
            for s in a.segments:
                if s["reduced"] and not s["pruned"] and not s.get("pruning"):
                    s["pruning"] = True
                    seg = os.path.join(self.cdir, s["segment"])
                    keep = sorted(x for x in self.st.retain_set(a, k) if x in s["seeds"])
                    self.worker.submit(("prune", a.id, s["segment"]), lambda a=a, s=s, seg=seg, keep=keep: self.ex.prune(a.spec, seg, s["info"], keep))

    def collect(self):
        for tag, status, val in self.worker.collect():
            kind, arm, rel = tag[:3]
            e = self.seg_entry(self.st.arms[arm], rel)
            if status == "err":
                self.event("%s_ERROR arm=%s segment=%s %s" % (kind.upper(), arm, rel, val))
                if kind == "prune":
                    e["pruning"] = False  # a failed prune is retried at the next pass
                else:
                    e["reducing"], n = False, e.get("tries", 1)
                    if n <= len(REDUCE_BACKOFF_S):  # retried within the session after a back-off (X5)
                        e["reduce_at"] = self.clock() + REDUCE_BACKOFF_S[n - 1]
                        self.log("REDUCE_RETRY arm=%s segment=%s attempt=%d in %.0f s" % (arm, rel, n + 1, REDUCE_BACKOFF_S[n - 1]))
                    else:
                        e["reduce_gave_up"] = True
                        self.event("REDUCE_GAVE_UP arm=%s segment=%s after %d attempts" % (arm, rel, n))
            elif kind == "reduce":
                e["reducing"] = False
                self.rec(kind="reduced", arm=arm, segment=rel, seeds=tag[3], reduced=val)
            else:
                self.rec(kind="pruned", arm=arm, segment=rel, info=val)

    def retry_reductions(self, now):
        for a in self.st.arms.values():
            for e in a.segments:
                if not e["reduced"] and not e.get("reducing") and not e.get("reduce_gave_up") and e.get("tries", 0) > 0 and now >= e.get("reduce_at", 0.0):
                    self.submit_reduce(a, e["segment"], e["seeds"])

    def final_reductions(self):
        """Before the analysis: wait for the worker and re-submit every completed but unreduced segment (up to FINAL_ROUNDS rounds).
        What stays unreduced is recorded (ledger kind `unreduced`, an event, `self.unreduced`): the analysis must treat those seeds
        as MISSING, not as the end of a prefix."""
        for rnd in range(FINAL_ROUNDS):
            self.worker.flush()
            self.collect()
            todo = [(a, e) for a in self.st.arms.values() for e in a.segments if not e["reduced"]]
            if not todo:
                break
            if rnd:
                self.sleep(REDUCE_BACKOFF_S[min(rnd - 1, len(REDUCE_BACKOFF_S) - 1)])
            for a, e in todo:
                self.log("FINAL_REDUCE arm=%s segment=%s round=%d" % (a.id, e["segment"], rnd + 1))
                self.submit_reduce(a, e["segment"], e["seeds"])
        self.worker.flush()
        self.collect()
        self.unreduced = [(a.id, e["segment"], list(e["seeds"])) for a in self.st.arms.values() for e in a.segments if not e["reduced"]]
        for arm, seg, seeds in self.unreduced:
            self.rec(kind="unreduced", arm=arm, segment=seg, seeds=seeds)
            self.log("UNREDUCED arm=%s segment=%s seeds=%s" % (arm, seg, seeds))
        return self.unreduced

    def retention_check(self):
        """G8: at least min_retained raw-array realizations per arm survive (or all of them if fewer completed).  -> {arm: (have, need)}."""
        k, out = self.plan["min_retained"], {}
        for a in self.st.arms.values():
            if not a.done:
                continue
            gone = set()
            for e in a.segments:
                if e["pruned"]:
                    gone |= set(e["seeds"]) - set(e.get("prune_info", {}).get("retained", []))
            have, need = len(a.done - gone), min(k, len(a.done))
            out[a.id] = (have, need)
            if have < need:
                self.event("RETENTION_SHORT arm=%s retained=%d needed=%d" % (a.id, have, need))
        return out

    def maintain(self, now):
        if now >= self.next_hb:
            self.next_hb = now + HB_S
            self.rec(kind="hb", gpu_s=self.st.gpu_s)
        self.retry_reductions(now)
        if now >= self.next_prune:
            self.next_prune = now + PRUNE_S
            self.prune_all()
        if now >= self.next_proj:
            self.next_proj = now + PROJECT_S
            used = self.st.gpu_s / 3600.0
            p = guards.projection(self.st, now, used, sum(1 for ln in self.lanes if ln.alive))
            flags = (" OVER_BUDGET" if p["projected_gpu_h"] > self.plan["budget_gpu_h"] else "") + (" PAST_CUTOFF" if p["completion"] > self.plan["cutoff"] else "")
            self.event("PROJECTION used_gpu_h=%.2f projected_gpu_h=%.2f budget_gpu_h=%.2f completion=%s cutoff=%s%s"
                       % (used, p["projected_gpu_h"], self.plan["budget_gpu_h"], utc_iso(p["completion"]), utc_iso(self.plan["cutoff"]), flags))
        if self.st.gpu_s / 3600.0 >= self.plan["budget_gpu_h"]:
            self.stop("budget", "GPU-h budget reached")
        self.reestimate()

    def reestimate(self):
        br = self.plan.get("blinded_reestimation")
        for i in (br or {}).get("arms", []):
            a = self.st.arms[i]
            if a.reest_done or a.halted or a.spec["no_reestimation"] or len(a.reduced) < br["fraction"] * a.spec["n_conf"]:
                continue
            n = len(a.reduced)
            try:
                res = self.ex.blinded_sd(a.spec, [s["info"] for s in a.segments if s["reduced"]])  # standard deviations and counts only
                sd, n = res if isinstance(res, tuple) else (res, n)
                n_req, new = guards.resize(sd, n, br["hw_target"][i], a.spec["n_conf"], br["max_factor"], min(a.spec["n_max"], a.spec["seeds"]["count"]))
            except Exception as e:  # N unchanged, once: the failure is a recorded event, not a retry loop
                text = "REESTIMATE_ERROR arm=%s %s: %s (N unchanged)" % (i, type(e).__name__, e)
                self.log(text)
                self.rec(kind="reest", arm=i, sd={}, n_obs=n, n_req=a.spec["n_conf"], n_old=a.n_target, n_new=a.n_target, text=text)
                continue
            text = "REESTIMATE arm=%s n_obs=%d sd=%s N_req=%d N %d -> %d" % (i, n, {k: float("%.6g" % v) for k, v in sd.items()}, n_req, a.n_target, new)
            self.log(text)
            self.rec(kind="reest", arm=i, sd=sd, n_obs=n, n_req=n_req, n_old=a.n_target, n_new=new, text=text)
