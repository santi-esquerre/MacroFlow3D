"""The REAL launch environment of the host job: `build_dir` RELATIVE to the repo root, units running with cwd = their segment directory
(a relative program path would resolve against the child's cwd), a missing binary, SIGTERM / a process-group kill while the engine waits
for a background job.  Real engine (`campaign.main`), real `RunnerExecutor`, real subprocesses, the fake runner binaries of `fake_runner.py`.
Run from `scripts/`:  python3 -W error -m unittest discover -s sf43camp/tests_executor -t .
"""
import contextlib
import io
import json
import os
import signal
import subprocess
import sys
import threading
import time
import unittest
import unittest.mock

from sf43camp import campaign, lanes
from sf43camp.plan import utc_iso

from . import test_e2e as e2e

SLEEPER = [sys.executable, "-c", "import time; time.sleep(120)"]


def alive(pid):
    try:
        os.kill(pid, 0)
        with open("/proc/%d/stat" % pid) as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"  # a zombie is dead
    except OSError:
        return False


class Paths(unittest.TestCase):
    setUp = e2e.E2E.setUp
    ledger = e2e.E2E.ledger

    def plan(self, wait_min=5.0):
        prod = e2e.arm("prod", "consumer", "g1", 1000, 4, 4, 2, e2e.consumer_dict(), primary=True)
        twin = e2e.arm("twin", "par2", "g1", 1000, 4, 4, 2, e2e.par2_dict(), heavy=False, no_reestimation=True)
        return {"campaign": "paths", "schema_version": 1, "cutoff_utc": utc_iso(time.time() + 3600), "grace_minutes": 1, "stage1_max_hours": 30,
                "budget_gpu_h": 100, "disk_min_free_gb": 0, "campaign_dir": self.cdir, "binaries": {}, "poll_s": 0.2,
                "analysis": {"command": [sys.executable, self.stub, "{campaign_dir}", self.reports], "sha256_list": {}},
                "failure": {"halt_rate": 0.9, "min_for_rate": 100}, "groups": {"g1": {"reserves": {"start": 1900, "count": 3}}},
                "background": [{"name": "long", "argv": SLEEPER, "nice": 19, "env": {}}], "background_wait_minutes": wait_min, "arms": [prod, twin]}

    def write_plan(self, **kw):
        with open(self.plan_file, "w") as f:
            json.dump(self.plan(**kw), f)

    def run_main(self, build):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = campaign.main([build, "all", "--plan", self.plan_file])
        return rc, out.getvalue()

    def chdir_root(self):
        old = os.getcwd()
        os.chdir(self.root)
        self.addCleanup(os.chdir, old)

    def bg_pids(self):
        return [r["pid"] for r in self.ledger() if r["kind"] == "background_start"]

    def test_relative_build_dir_runs_both_arms(self):
        self.write_plan(wait_min=0.0)
        self.chdir_root()
        rc, out = self.run_main("build")  # the host job: `run_sf43.sh build/v100-release ...` from the repo root
        self.assertEqual(rc, campaign.DONE, out[-3000:])
        ends = [r for r in self.ledger() if r["kind"] == "unit_end"]
        self.assertEqual(sorted((r["arm"], r["cls"]) for r in ends), [("prod", "ok")] * 2 + [("twin", "ok")] * 2)
        self.assertNotIn("LANE_CRASH", out)

    def test_missing_binary_is_config_error_and_the_engine_does_not_wait_for_background(self):
        self.write_plan(wait_min=5.0)
        self.chdir_root()
        for exe in ("lagrangian_ensemble", "macroflow3d_pipeline"):
            os.remove(os.path.join(self.build, exe))
        t0 = time.time()
        rc, out = self.run_main("build")
        self.assertLess(time.time() - t0, 30.0, out[-3000:])
        self.assertEqual(rc, campaign.PARTIAL, out[-3000:])
        self.assertRegex(out, r"SPAWN_FAILED arm=prod path=\S*lagrangian_ensemble error=")
        self.assertNotIn("LANE_CRASH", out)
        halts = {r["arm"]: r["reason"] for r in self.ledger() if r["kind"] == "arm_halt"}
        self.assertEqual(halts, {"prod": "config_error", "twin": "config_error"})
        self.assertIn("BACKGROUND_TERMINATED name=long reason=no_live_lane", out)
        pids = self.bg_pids()
        self.assertEqual(len(pids), 1)
        self.assertFalse(alive(pids[0]))

    def test_sigterm_during_the_background_wait_terminates_the_job(self):
        self.write_plan(wait_min=5.0)
        orig = lanes.Engine.retention_check

        def retention_then_signal_soon(eng):  # the last step before the background wait
            threading.Timer(1.0, os.kill, (os.getpid(), signal.SIGTERM)).start()
            return orig(eng)

        self.chdir_root()
        t0 = time.time()
        with unittest.mock.patch.object(lanes.Engine, "retention_check", retention_then_signal_soon):
            rc, out = self.run_main(self.build)
        self.assertEqual(rc, campaign.SIGNALLED, out[-3000:])
        self.assertLess(time.time() - t0, 30.0)
        self.assertFalse(os.path.exists(os.path.join(self.cdir, "REPORT.txt")))
        pids = self.bg_pids()
        self.assertEqual(len(pids), 1)
        self.assertFalse(alive(pids[0]))
        self.assertIn("BACKGROUND_TERMINATED name=long reason=signal", out)

    def test_group_kill_of_the_engine_reaches_the_background_job(self):
        self.write_plan(wait_min=5.0)
        env = dict(os.environ, PYTHONPATH=e2e.SCRIPTS + os.pathsep + e2e.SF42)
        logf = open(os.path.join(self.root, "engine.out"), "wb")
        self.addCleanup(logf.close)
        eng = subprocess.Popen([sys.executable, "-W", "error", "-m", "sf43camp", self.build, "all", "--plan", self.plan_file], cwd=self.root, env=env,
                               stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)  # a session of its own, like the helper's job
        self.addCleanup(lambda: os.killpg(eng.pid, signal.SIGKILL) if eng.poll() is None else None)
        deadline, pid = time.time() + 60.0, None
        while time.time() < deadline and eng.poll() is None:  # all 4 units ended: the engine is in its final phase (reductions, then the background wait)
            if os.path.exists(os.path.join(self.cdir, "ledger.jsonl")) and sum(r["kind"] == "unit_end" for r in self.ledger()) >= 4 and self.bg_pids():
                pid = self.bg_pids()[0]
                break
            time.sleep(0.1)
        with open(os.path.join(self.root, "engine.out")) as f:
            self.assertIsNotNone(pid, f.read()[-3000:])
        time.sleep(1.5)
        self.assertTrue(alive(pid))
        os.killpg(eng.pid, signal.SIGKILL)  # SIGKILL: the engine cannot clean up; only the shared process group can reach the job
        eng.wait(timeout=10)
        t = time.time() + 5.0
        while alive(pid) and time.time() < t:
            time.sleep(0.1)
        self.assertFalse(alive(pid))


if __name__ == "__main__":
    unittest.main()
