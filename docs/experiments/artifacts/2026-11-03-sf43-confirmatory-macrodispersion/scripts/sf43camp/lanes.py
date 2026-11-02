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


class Lane:
    def __init__(self, name, gpu, heavy_only):
        self.name, self.gpu, self.heavy_only = name, gpu, heavy_only
        self.unit = self.proc = None
        self.alive = self.held = True  # alive: may claim; held: its GPU is still held by the job
        self.nothing = False  # last pick found nothing to claim
        self.retry_at = 0.0


class Engine:
    def __init__(self, st, ledger, ex, build_dir, cdir, lanes, *, clock, sleep, disk_free_gb, gpu_free_mib, spawn, worker, log, poll_s=2.0, dry=False):
        self.st, self.ledger, self.ex, self.build_dir, self.cdir = st, ledger, ex, build_dir, cdir
        self.lanes, self.clock, self.sleep, self.disk_free_gb, self.gpu_free_mib = lanes, clock, sleep, disk_free_gb, gpu_free_mib
        self.spawn, self.worker, self.log, self.poll_s, self.dry = spawn, worker, log, poll_s, dry
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
            done = sorted(s for s in self.ex.completed(a.spec, os.path.join(self.cdir, c["segment"])) if s in c["seeds"])
            self.rec(kind="unit_end", unit=uid, arm=a.id, lane=c["lane"], gpu=c["gpu"], cls="recovered", completed=done, failed_seed=None,
                     next_seed=next((s for s in c["seeds"] if s not in done), None), seconds={}, verdicts={}, peak_mib=None, wall=None,
                     segment=c["segment"], segment_seed0=c["seeds"][0])
            self.log("RECOVERED unit=%d arm=%s completed=%d of %d" % (uid, a.id, len(done), len(c["seeds"])))

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
        for ln in self.lanes:
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
        for ln in self.lanes:
            if ln.unit:
                ln.proc.terminate()
                self.end(ln, self.clock(), cls)

    def end(self, ln, now, forced=None):
        u, a = ln.unit, self.st.arms[ln.unit.arm]
        seeds = u.seeds
        if forced:
            c = {"cls": forced, "completed_seeds": self.ex.completed(a.spec, ln.seg)}
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
        again = cls == "other_error" and nxt in a.other_seen
        r = self.rec(kind="unit_end", unit=ln.uid, arm=a.id, lane=ln.name, gpu=ln.gpu, cls=cls, completed=done, failed_seed=fs, next_seed=nxt,
                     seconds={str(s): v for s, v in c.get("seconds", {}).items() if s in done},
                     verdicts={str(s): v for s, v in c.get("verdicts", {}).items() if s in seeds}, peak_mib=c.get("peak_mib"),
                     wall=now - ln.t0, segment=ln.rel, segment_seed0=seeds[0])
        self.log("SF43_UNIT_END lane=%s gpu=%s arm=%s seed0=%d nr=%d segment=%s cls=%s completed=%d seconds=%.1f utc=%s"
                 % (ln.name, ln.gpu, a.id, seeds[0], len(seeds), ln.rel, cls, len(done), now - ln.t0, r["utc"]))
        ln.unit = ln.proc = None
        if done:
            self.submit_reduce(a, ln.rel, done)
        fl = self.plan["failure"]
        if cls == "oom":
            delay = guards.oom_backoff_s(a.oom_count.get(nxt, 1))
            if nxt is not None:
                a.not_before[nxt] = now + delay
            self.event("OOM arm=%s seed0=%d retry_in_min=%d (not a failure)" % (a.id, seeds[0], delay // 60))
        elif cls == "unusable_solve":
            self.unusable(a, fs)
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
        self.rec(kind="replacement", group=g.id, seed=s, replacement=r, arm=a.id)
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
    def submit_reduce(self, a, rel, seeds):
        seg = os.path.join(self.cdir, rel)
        out = os.path.join(self.cdir, "reduced", a.id, os.path.basename(rel))
        self.worker.submit(("reduce", a.id, rel, list(seeds)), lambda: self.ex.reduce(a.spec, seg, out))

    def prune_all(self):
        for a in self.st.arms.values():
            for s in a.segments:
                if s["reduced"] and not s["pruned"] and not s.get("pruning"):
                    s["pruning"] = True
                    seg = os.path.join(self.cdir, s["segment"])
                    keep = [x for x in a.spec["retained"] if x in s["seeds"]]
                    self.worker.submit(("prune", a.id, s["segment"]), lambda a=a, s=s, seg=seg, keep=keep: self.ex.prune(a.spec, seg, s["info"], keep))

    def collect(self):
        for tag, status, val in self.worker.collect():
            kind, arm, rel = tag[:3]
            if status == "err":
                self.event("%s_ERROR arm=%s segment=%s %s" % (kind.upper(), arm, rel, val))
                for x in self.st.arms[arm].segments:
                    if x["segment"] == rel:
                        x["pruning"] = False  # a failed prune is retried at the next pass; a failed reduction is not (the raw data stay)
            elif kind == "reduce":
                self.rec(kind="reduced", arm=arm, segment=rel, seeds=tag[3], reduced=val)
            else:
                self.rec(kind="pruned", arm=arm, segment=rel, info=val)

    def maintain(self, now):
        if now >= self.next_hb:
            self.next_hb = now + HB_S
            self.rec(kind="hb", gpu_s=self.st.gpu_s)
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
            if a.reest_done or a.halted or len(a.reduced) < br["fraction"] * a.spec["n_conf"]:
                continue
            n = len(a.reduced)
            try:
                sd = self.ex.blinded_sd(a.spec, [s["info"] for s in a.segments if s["reduced"]])  # standard deviations only
            except Exception as e:  # N unchanged, once: the failure is a recorded event, not a retry loop
                text = "REESTIMATE_ERROR arm=%s %s: %s (N unchanged)" % (i, type(e).__name__, e)
                self.log(text)
                self.rec(kind="reest", arm=i, sd={}, n_obs=n, factor=1.0, n_old=a.n_target, n_new=a.n_target, text=text)
                continue
            f, new = guards.resize(sd, n, br["sd_plan_ub"][i], a.spec["n_conf"], br["max_factor"], min(a.spec["n_max"], a.spec["seeds"]["count"]))
            text = "REESTIMATE arm=%s n_obs=%d sd=%s factor=%.3f N %d -> %d" % (i, n, {k: float("%.6g" % v) for k, v in sd.items()}, f, a.n_target, new)
            self.log(text)
            self.rec(kind="reest", arm=i, sd=sd, n_obs=n, factor=f, n_old=a.n_target, n_new=new, text=text)
