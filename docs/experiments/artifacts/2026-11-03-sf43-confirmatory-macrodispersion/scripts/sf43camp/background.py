"""Plan `background` jobs: CPU processes that run next to the units (the H15 coverage simulation inside the smoke-test job).

Each entry {name, argv, nice (default 19), env}: started once at engine start with `CUDA_VISIBLE_DEVICES=""` in the LAUNCHER'S process group
(no new session: the helper's group signal on a cancel must reach them; `terminate` still kills each job's tree)
(`argv` may contain `{campaign_dir}` / `{build_dir}`), output in `<campaign_dir>/background/<name>.log`; its end writes
`<name>.done` (the exit code) and the ledger record `background_end`.  A resumed session skips the jobs whose done file exists.
SIGTERM / SIGINT terminate them with the units.  `wait(seconds, rec, stop)` is the final analysis' wait (plan `background_wait_minutes`); what is
still running after it is terminated and reported; `stop()` (the engine's signal flag) is checked at least every 2 s and ends the wait at once.
"""
import os
import signal
import subprocess
import time

from . import procs


class Background:
    def __init__(self, plan, build_dir, cdir, log, clock=time.time, sleep=time.sleep, popen=subprocess.Popen):
        self.entries, self.cdir, self.log, self.clock, self.sleep, self.popen = plan.get("background", []), cdir, log, clock, sleep, popen
        self.build_dir, self.dir = build_dir, os.path.join(cdir, "background")
        self.procs, self.state = {}, {}  # name -> Popen; name -> 'running' | 'done rc=N' | 'skipped (done file)' | 'terminated'

    def done_file(self, name):
        return os.path.join(self.dir, name + ".done")

    def start(self, rec):
        if not self.entries:
            return
        os.makedirs(self.dir, exist_ok=True)
        for b in self.entries:
            n = b["name"]
            if os.path.exists(self.done_file(n)):
                self.state[n] = "skipped (done file)"
                self.log("BACKGROUND_SKIPPED name=%s (done file exists)" % n)
                continue
            argv = [x.replace("{campaign_dir}", self.cdir).replace("{build_dir}", self.build_dir) for x in b["argv"]]
            env = dict(os.environ, **{k: str(v) for k, v in b.get("env", {}).items()})
            env["CUDA_VISIBLE_DEVICES"] = ""
            logf = open(os.path.join(self.dir, n + ".log"), "ab")
            try:
                self.procs[n] = self.popen(["nice", "-n", str(b.get("nice", 19))] + argv, env=env, stdout=logf, stderr=subprocess.STDOUT)
            except OSError as e:
                self.state[n] = "failed to start: %s" % e
                rec(kind="background_end", name=n, rc=-1, text=self.state[n])
                self.log("BACKGROUND_FAILED name=%s %s" % (n, e))
                continue
            finally:
                logf.close()
            self.state[n] = "running"
            rec(kind="background_start", name=n, pid=self.procs[n].pid, argv=argv)
            self.log("BACKGROUND_START name=%s pid=%d" % (n, self.procs[n].pid))

    def poll(self, rec):
        for n, p in list(self.procs.items()):
            rc = p.poll()
            if rc is not None:
                del self.procs[n]
                self.state[n] = "done rc=%d" % rc
                with open(self.done_file(n), "w") as f:
                    f.write("%d\n" % rc)
                rec(kind="background_end", name=n, rc=rc)
                self.log("BACKGROUND_END name=%s rc=%d" % (n, rc))

    def terminate(self, rec, reason=None):
        for n, p in list(self.procs.items()):
            procs.kill_tree(p.pid, signal.SIGTERM)
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                procs.kill_tree(p.pid, signal.SIGKILL)
                p.wait()
            del self.procs[n]
            self.state[n] = "terminated"
            rec(kind="background_end", name=n, rc=p.returncode, text="terminated")
            self.log("BACKGROUND_TERMINATED name=%s%s" % (n, " reason=" + reason if reason else ""))

    def wait(self, seconds, rec, stop=lambda: None):
        """Wait up to `seconds` (the engine's clock) for the running jobs, then terminate the rest; a signal (`stop()` truthy) ends the wait now."""
        deadline = self.clock() + seconds
        while self.procs and self.clock() < deadline and not stop():
            for p in list(self.procs.values()):
                try:
                    p.wait(timeout=0.5)  # real time passes even on a virtual clock
                except subprocess.TimeoutExpired:
                    pass
            self.poll(rec)
            if self.procs:
                self.sleep(min(1.0, max(0.0, deadline - self.clock())))  # a signal is seen within ~2 s
        self.poll(rec)
        self.terminate(rec, "signal" if stop() else None)

    def report(self):
        return ["  %-24s %s" % (n, st) for n, st in sorted(self.state.items())]
