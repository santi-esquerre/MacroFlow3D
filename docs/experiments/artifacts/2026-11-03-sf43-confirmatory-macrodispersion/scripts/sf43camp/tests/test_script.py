"""run_sf43.sh end to end with a fake runner binary (real subprocesses, real threads, nothing outside a temp directory).  ~10 s."""
import hashlib
import json
import os
import signal
import subprocess
import time
import unittest
from pathlib import Path

from .fx import arm, mk_plan, tmpdir

SH = Path(__file__).resolve().parents[2] / "run_sf43.sh"
RUNNER = """#!/usr/bin/env bash
arm=$1; s0=$2; nr=$3; seg=$4
echo $$ > "$seg/pid"
if [ "$arm" = slow ]; then sleep 30; exit 0; fi
for ((i = 0; i < nr; i++)); do
  d=$seg/r_$((s0 + i)); mkdir -p "$d"; echo "$CUDA_VISIBLE_DEVICES" > "$d/gpu"; sleep 0.2; : > "$d/DONE"
done
echo "STATUS fake nr=$nr"
"""


class Script(unittest.TestCase):
    def setUp(self):
        self.d = Path(tmpdir(self))
        self.bin = self.d / "bin"
        self.bin.mkdir()
        r = self.bin / "fake_runner"
        r.write_text(RUNNER)
        r.chmod(0o755)
        self.cdir = str(self.d / "camp")

    def plan_file(self, arms, **over):
        p = mk_plan(self.cdir, arms, binaries={"fake_runner": hashlib.md5((self.bin / "fake_runner").read_bytes()).hexdigest()},
                    executor="sf43camp.tests.shell_exec:Executor", poll_s=0.2,
                    analysis={"command": ["python3", "-c", "import sys; open(sys.argv[1] + '/analysis.out', 'w').write('x')", "{campaign_dir}"], "sha256_list": {}},
                    **over)
        p.pop("cutoff"), p.pop("plan_sha256"), p.pop("recompute"), p.pop("reconcile")
        f = self.d / "plan.json"
        f.write_text(json.dumps(p))
        return str(f)

    def sh(self, *args, gpus="1,0"):
        return subprocess.run(["bash", str(SH), str(self.bin)] + list(args), capture_output=True, text=True, timeout=120,
                              env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, REMOTE_GPU_COUNT_GRANTED="2"))

    def gpus_of(self, arm_id):
        out = {}
        for p in Path(self.cdir, "segments", arm_id).glob("*/r_*/gpu"):
            out[int(p.parent.name[2:])] = p.read_text().strip()
        return out

    def test_full_run_resume_and_exit_codes(self):
        f = self.plan_file([arm("H", n=8, unit=2, retained=[100]), arm("L", heavy=False, n=6, unit=2)])
        r = self.sh("all", "--plan", f)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stderr, "")
        self.assertRegex(r.stdout, r"SF43_UNIT lane=[AB] gpu=[01] arm=H seed0=100 nr=2 segment=segments/H/u\d+_s100 utc=")
        self.assertIn("SF43_UNIT_END", r.stdout)
        self.assertIn("cls=ok completed=2", r.stdout)
        self.assertTrue((Path(self.cdir) / "DONE").exists() and (Path(self.cdir) / "REPORT.txt").exists())
        self.assertTrue((Path(self.cdir) / "analysis.out").exists())
        self.assertEqual(set(self.gpus_of("L").values()), {"1"})  # light arm: lane A only (first granted id)
        self.assertEqual(set(self.gpus_of("H").values()), {"0", "1"})  # heavy arm: both lanes
        self.assertEqual(len(self.gpus_of("H")) + len(self.gpus_of("L")), 14)
        self.assertEqual(set(self.gpus_of("H")), set(range(100, 108)))
        n_units = r.stdout.count("SF43_UNIT ")
        self.assertEqual(self.sh("all", "--plan", f).returncode, 13)  # stale state without --resume
        r2 = self.sh("all", "--plan", f, "--resume")
        self.assertEqual(r2.returncode, 0, r2.stdout)
        self.assertEqual(r2.stdout.count("SF43_UNIT "), 0)
        self.assertEqual(n_units, 7)

    def test_single_gpu(self):
        f = self.plan_file([arm("H", n=4, unit=2), arm("L", heavy=False, n=4, unit=2)])
        r = self.sh("all", "--plan", f, gpus="5")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(set(self.gpus_of("H").values()) | set(self.gpus_of("L").values()), {"5"})

    def test_md5_mismatch_and_bad_plan_and_usage_refused(self):
        f = self.plan_file([arm("H", n=2)])
        with open(self.bin / "fake_runner", "a") as fh:
            fh.write("# changed\n")
        r = self.sh("all", "--plan", f)
        self.assertEqual(r.returncode, 13, r.stdout)
        self.assertIn("REFUSED binaries do not match", r.stdout)
        self.assertFalse(os.path.exists(self.cdir))
        bad = self.d / "bad.json"
        bad.write_text("{}")
        self.assertEqual(self.sh("all", "--plan", str(bad)).returncode, 13)
        self.assertEqual(subprocess.run(["bash", str(SH), "x"], capture_output=True).returncode, 13)

    def test_dry_run_prints_schedule_and_touches_nothing(self):
        f = self.plan_file([arm("H", n=10, unit=5, est=600.0), arm("L", heavy=False, n=10, unit=5, est=600.0)])
        r = self.sh("all", "--plan", f, "--dry-run")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertEqual(r.stdout.count("DRYRUN SF43_UNIT "), 4)
        self.assertRegex(r.stdout, r"DRYRUN_SUMMARY lanes=2 complete=True .*stage1_h=.* stage2_h=.* gpu_h=")
        self.assertFalse(os.path.exists(self.cdir))
        self.assertNotRegex(r.stdout, r"gpu=[01] ")  # no real GPU id is written by a dry run

    def test_sigterm_terminates_runner_records_and_exits_143(self):
        f = self.plan_file([arm("slow", n=2, unit=2)])
        p = subprocess.Popen(["bash", str(SH), str(self.bin), "all", "--plan", f], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES="1"))
        pidf = None
        for _ in range(100):
            g = list(Path(self.cdir, "segments", "slow").glob("*/pid"))
            if g:
                pidf = g[0]
                break
            time.sleep(0.2)
        self.assertIsNotNone(pidf, "runner never started")
        time.sleep(0.5)
        runner_pid = int(pidf.read_text())
        p.send_signal(signal.SIGTERM)
        out, err = p.communicate(timeout=60)
        self.assertEqual(p.returncode, 143, out + err)
        with self.assertRaises(OSError):
            os.kill(runner_pid, 0)
        recs = [json.loads(ln) for ln in Path(self.cdir, "ledger.jsonl").read_text().splitlines()]
        self.assertIn("terminated", [r.get("cls") for r in recs if r["kind"] == "unit_end"])
        self.assertFalse((Path(self.cdir) / "analysis.out").exists())


if __name__ == "__main__":
    unittest.main()
