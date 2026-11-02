"""Operating-system side: runner subprocesses, process-tree kill, disk and GPU-memory queries, background worker."""
import os
import queue
import shutil
import signal
import subprocess
import threading
import time


def kill_tree(pid, sig):
    """Signal a process and all its descendants (children first).  The runners stay in the launcher's process group, so the
    helper's group kill on a cancel reaches them too; this reaches one lane's unit only."""
    kids = []
    try:
        for d in os.listdir("/proc"):
            if d.isdigit():
                try:
                    with open("/proc/%s/stat" % d) as f:
                        if int(f.read().rsplit(")", 1)[1].split()[1]) == pid:
                            kids.append(int(d))
                except (OSError, ValueError, IndexError):
                    pass
    except OSError:
        pass
    for k in kids:
        kill_tree(k, sig)
    try:
        os.kill(pid, sig)
    except OSError:
        pass


class Proc:
    def __init__(self, popen, logf):
        self.p, self.logf = popen, logf

    def poll(self):
        rc = self.p.poll()
        if rc is not None and not self.logf.closed:
            self.logf.close()
        return rc

    def terminate(self, grace=30.0):
        kill_tree(self.p.pid, signal.SIGTERM)
        t = time.time() + grace
        while self.p.poll() is None and time.time() < t:
            time.sleep(0.2)
        if self.p.poll() is None:
            kill_tree(self.p.pid, signal.SIGKILL)
            self.p.wait()
        self.poll()


def spawn(cmd, gpu):
    """Start the runner of a command dict (executor.command) on one GPU; stdout/stderr go to the unit's own log."""
    env = dict(os.environ, **cmd.get("env", {}))
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)  # the runners always use device 0 of what they see
    logf = open(cmd["log_path"], "ab")
    return Proc(subprocess.Popen(cmd["argv"], cwd=cmd.get("cwd") or None, env=env, stdout=logf, stderr=subprocess.STDOUT), logf)


def disk_free_gb(path):
    u = shutil.disk_usage(path)
    return u.free / 1e9


def nvidia_free_mib(gpu):
    """Free memory of one GPU (read-only query) or None if it cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)],
                             capture_output=True, text=True, timeout=30, check=True).stdout
        tot, used = (float(x) for x in out.strip().split(","))
        return tot - used
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class Worker:
    """Background executor of reduction and pruning jobs (one thread; never between units of a lane).  threaded=False runs the job
    inline at submit (deterministic tests).  Results are collected by the main thread, which alone touches the ledger."""

    def __init__(self, threaded):
        self.threaded, self.q, self.res = threaded, queue.Queue(), queue.Queue()
        if threaded:
            threading.Thread(target=self._loop, daemon=True).start()

    def _run(self, tag, fn):
        try:
            self.res.put((tag, "ok", fn()))
        except Exception as e:  # reported, never fatal
            self.res.put((tag, "err", "%s: %s" % (type(e).__name__, e)))

    def _loop(self):
        while True:
            tag, fn = self.q.get()
            self._run(tag, fn)
            self.q.task_done()

    def submit(self, tag, fn):
        if self.threaded:
            self.q.put((tag, fn))
        else:
            self._run(tag, fn)

    def flush(self):
        if self.threaded:
            self.q.join()

    def collect(self):
        out = []
        while not self.res.empty():
            out.append(self.res.get())
        return out
