"""END-TO-END mini-campaign: the real engine (`sf43camp.campaign.main`, real subprocesses and clock), the real default executor
(`sf43camp.runner_executor:RunnerExecutor`, built by the engine as `Class(plan, build_dir)`), the fake runner binaries of
`fake_runner.py`, the real SF-42 reducer/pruner, a simulated kill (SIGTERM of the launcher in the middle) and `--resume`, and the
real final-analysis hooks (`recompute43`, `reconcile43`) plus a stub analysis command.

Covers: executor wiring (X1), blinded SD contract and H9 re-sizing, CRN group adjacency, priorities, hmax_div_first layouts,
primary-arm selective pruning with retention, Par2 seed map / fallback verdict / own-generator log lines (G7), replacement, solver
records, reconcile43 and recompute43 against the engine's real layout and ledger, lane pins, determinism pair, background job,
peak memory.  Run from `scripts/`:  python3 -W error -m unittest discover -s sf43camp/tests_executor -t .
"""
import glob
import json
import os
import signal
import stat
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

from sf43camp import campaign, lanes
from sf43camp.plan import utc_iso

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.abspath(os.path.join(HERE, "..", ".."))
SF42 = os.path.abspath(os.path.join(SCRIPTS, "..", "..", "2026-10-30-sf42-confirmatory-tooling", "scripts"))

PLANES_D = [25, 30, 40, 60, 94, 120, 150]
WINDOW = "0.375,0.625,0.375,0.625"
STUB = "import sys,os\nopen(os.path.join(sys.argv[2], 'analysis_ran.txt'), 'w').write(sys.argv[1] + '\\n')\n"


def consumer_dict(**over):
    c = {"grid": [208, 4, 4], "eps": 1.5, "ell": 0.25, "window": WINDOW, "np_side": 8, "planes_d": PLANES_D, "hmax_div": [8],
         "twin": True, "pollock_bridge": True}
    c.update(over)
    return c


def par2_dict(**over):
    p = {"grid": [208, 4, 4], "eps": 1.5, "ell": 0.25, "window": WINDOW, "planes_d": PLANES_D, "n_particles": 100, "snapshots_every": 100}
    p.update(over)
    return p


def arm(i, runner, group, start, count, n_conf, unit, run, prio=1, heavy=True, mem=1000, **extra):
    a = {"id": i, "family": "t", "amplitude": 2.25, "runner": runner, "priority": prio, "heavy": heavy, "min_free_mib": mem, "group": group,
         "seeds": {"start": start, "count": count}, "n_conf": n_conf, "n_max": count, "unit_size": unit, "est_seconds": 1.0, "raw_gb": 0.001,
         runner: run, "reduce": {"code": runner, "d_windows": [[25, 94]]}, "retained": []}
    a.update(extra)
    return a


class E2E(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.root = os.path.realpath(self._t.name)
        self.build = os.path.join(self.root, "build")
        os.makedirs(self.build)
        for exe, mode in (("lagrangian_ensemble", "consumer"), ("macroflow3d_pipeline", "par2")):
            p = os.path.join(self.build, exe)
            with open(p, "w") as f:
                f.write('#!/bin/sh\nexec %s %s %s "$@"\n' % (sys.executable, os.path.join(HERE, "fake_runner.py"), mode))
            os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR)
        self.reports = os.path.join(self.root, "reports")
        os.makedirs(self.reports)
        self.stub = os.path.join(self.root, "stub_analysis.py")
        with open(self.stub, "w") as f:
            f.write(STUB)
        self.cdir = os.path.join(self.root, "camp")
        for sig in (signal.SIGTERM, signal.SIGINT):
            self.addCleanup(signal.signal, sig, signal.getsignal(sig))
        env = mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,0", "PYTHONPATH": SCRIPTS + os.pathsep + SF42, "FAKE_FALLBACK_SEED": "1001",
                                          "FAKE_NOCONV_SEED": "3003"})
        env.start()
        self.addCleanup(env.stop)
        pr = mock.patch.object(lanes, "PRUNE_S", 0.3)  # prune passes while the campaign runs
        pr.start()
        self.addCleanup(pr.stop)
        self.plan_file = os.path.join(self.root, "plan.json")
        with open(self.plan_file, "w") as f:
            json.dump(self.plan(), f)

    def plan(self):
        prod = arm("prod", "consumer", "g1", 1000, 16, 8, 4, consumer_dict(hmax_div_first={"n": 4, "divs": [8, 16]}), primary=True)
        twin = arm("twin", "par2", "g1", 1000, 8, 8, 4, par2_dict(), heavy=False, no_reestimation=True)
        g7 = arm("g7", "par2", "g3", 3000, 4, 4, 4, par2_dict(generator="random_spectral", n_modes=10), prio=3, heavy=False, mem=500)
        dx = arm("det_x", "consumer", "g2", 2000, 4, 4, 2, consumer_dict(), prio=2, lane="A")
        dy = arm("det_y", "consumer", "g2", 2000, 4, 4, 2, consumer_dict(), prio=2, lane="B")
        cmd = lambda *a: [sys.executable, "-W", "error", "-m"] + list(a)  # noqa: E731
        return {
            "campaign": "e2e", "schema_version": 1, "cutoff_utc": utc_iso(time.time() + 3600), "grace_minutes": 1, "stage1_max_hours": 30,
            "budget_gpu_h": 100, "disk_min_free_gb": 0, "campaign_dir": self.cdir, "binaries": {}, "poll_s": 0.2,
            "analysis": {"command": [sys.executable, self.stub, "{campaign_dir}", self.reports], "sha256_list": {}},
            "recompute": {"command": cmd("recompute43", "run", "--campaign", "{campaign_dir}", "--reduced", "{campaign_dir}/reduced", "--out",
                                         os.path.join(self.reports, "recompute.txt"), "--primary-arm", "prod", "--single-pass-hmax", "8")},
            "reconcile": {"command": cmd("reconcile43", "run", "--campaign", "{campaign_dir}", "--out", os.path.join(self.reports, "reconcile.txt"))},
            "blinded_reestimation": {"fraction": 0.5, "max_factor": 2.0, "arms": ["prod"], "hw_target": {"prod": {"alpha_TP": 1e-6}}},
            "failure": {"halt_rate": 0.9, "min_for_rate": 100},
            "groups": {"g1": {"reserves": {"start": 1900, "count": 3}}, "g2": {"reserves": {"start": 2900, "count": 2}},
                       "g3": {"reserves": {"start": 3900, "count": 3}}},
            "determinism_pairs": [["det_x", "det_y"]],
            "background": [{"name": "cov", "argv": [sys.executable, "-c", "print('cov done')"], "nice": 19, "env": {}}],
            "background_wait_minutes": 0.2,
            "arms": [prod, twin, g7, dx, dy]}

    def main(self, *extra):
        return campaign.main([self.build, "all", "--plan", self.plan_file] + list(extra))

    def test_kill_resume_and_the_whole_pipeline(self):
        stop = threading.Event()

        def killer():  # SIGTERM the launcher in the middle: after the third unit has ended
            ledger = os.path.join(self.cdir, "ledger.jsonl")
            while not stop.wait(0.02):
                if os.path.exists(ledger):
                    with open(ledger) as f:
                        if f.read().count('"kind": "unit_end"') >= 3:
                            os.kill(os.getpid(), signal.SIGTERM)
                            return
        timer = threading.Thread(target=killer)
        timer.start()
        rc = self.main()
        stop.set()
        timer.join()
        self.assertEqual(rc, campaign.SIGNALLED, "the campaign finished before the simulated kill; make the plan longer")
        self.assertFalse(os.path.exists(os.path.join(self.cdir, "REPORT.txt")))  # a cancel runs no analysis
        rc = self.main()  # no --resume: a stale ledger is refused
        self.assertEqual(rc, campaign.REFUSED)
        rc = self.main("--resume")
        self.assertIn(rc, (campaign.DONE, campaign.PARTIAL), self.report())
        self.check_everything()

    # ------------------------------------------------------------------------------------------------------------------------------
    def report(self):
        p = os.path.join(self.cdir, "REPORT.txt")
        if not os.path.exists(p):
            return "(no REPORT)"
        with open(p) as f:
            txt = f.read()
        for n in ("recompute.txt", "reconcile.txt"):
            if os.path.exists(os.path.join(self.reports, n)):
                with open(os.path.join(self.reports, n)) as f:
                    txt += "\n--- %s\n%s" % (n, f.read()[-3000:])
        return txt

    def ledger(self):
        with open(os.path.join(self.cdir, "ledger.jsonl")) as f:
            return [json.loads(x) for x in f if x.strip()]

    def check_everything(self):
        L, rep = self.ledger(), self.report()
        kind = lambda k: [r for r in L if r["kind"] == k]  # noqa: E731
        # executor wiring, analysis hooks
        self.assertTrue(os.path.exists(os.path.join(self.reports, "analysis_ran.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.cdir, "DONE")) or os.path.exists(os.path.join(self.cdir, "DONE_PARTIAL")), rep)
        self.assertFalse(os.path.exists(os.path.join(self.cdir, "ANALYSIS_FAILED")), rep)
        self.assertFalse(os.path.exists(os.path.join(self.cdir, "ANALYSIS_BLOCKED_UNREDUCED")))
        with open(os.path.join(self.reports, "reconcile.txt")) as f:
            recon = f.read()
        self.assertTrue(recon.rstrip().endswith("RECONCILE OK"), recon[-1500:])
        self.assertIn("par2     g7", recon)  # the own-generator (`K generation`) cell was mapped
        with open(os.path.join(self.reports, "recompute.txt")) as f:
            rec = f.read()
        self.assertTrue(rec.rstrip().endswith("RECOMPUTE PASS"), rec[-1500:])
        self.assertIn("PRIMARY_ARM prod", rec)
        # blinded re-estimation through the real executor: SDs and counts only, N doubled (H9: min(2 n_conf, max(n_conf, N_req)))
        re = kind("reest")
        self.assertEqual([(r["arm"], r["n_old"], r["n_new"]) for r in re], [("prod", 8, 16)])
        self.assertNotIn("REESTIMATE_ERROR", rep)
        self.assertEqual(set(re[0]["sd"]), {"alpha_TP", "alpha_TAP", "alpha_LP"} & set(re[0]["sd"]))
        self.assertGreaterEqual(re[0]["n_obs"], 4)
        # replacement of the unusable Par2 solve (g7: r = 3 never converges), group-wide, key replaced_by
        rp = kind("replacement")
        self.assertEqual([(r["group"], r["seed"], r["replaced_by"]) for r in rp], [("g3", 3003, 3900)])
        seeds = kind("seed")
        bad = [(r["arm"], r["seed"]) for r in seeds if r["class"] == "unusable_solve"]
        self.assertEqual(bad, [("g7", 3003)])
        self.assertTrue(any(r["verdict"] == "fallback" and r["class"] == "ok" for r in seeds))  # FAKE_FALLBACK_AT=1: usable Par2 / consumer fallbacks
        c = [r for r in seeds if r["arm"] == "prod"]
        self.assertTrue(c and all(r["iterations"] is not None and r["max_div"] == 2.0e-08 for r in c))
        self.assertEqual(sorted(r["seed"] for r in c), list(range(1000, 1016)))
        # layouts: the first 4 positions are one 2-pass segment, the rest single-pass; reduced rows carry one label for the production pass
        with open(os.path.join(self.cdir, "reduced", "prod", self.segment("prod", 1000), "prod__%s__d25-94.csv" % self.segment("prod", 1000))) as f:
            head = f.read().splitlines()
        hm = {x.split(",")[3] for x in head[1:]}
        self.assertEqual(hm, {"8", "16"})
        late = self.segment("prod", 1012)
        with open(os.path.join(self.cdir, "reduced", "prod", late, "prod__%s__d25-94.csv" % late)) as f:
            self.assertEqual({x.split(",")[3] for x in f.read().splitlines()[1:]}, {"8"})
        # Par2 reduced rows carry the seed
        tw = self.segment("twin", 1000)
        with open(os.path.join(self.cdir, "reduced", "twin", tw, "twin__%s__d25-94.csv" % tw)) as f:
            rows = [x.split(",") for x in f.read().splitlines()[1:]]
        self.assertEqual({int(float(r[4])) for r in rows if r[5] == "plane"}, {1000, 1001, 1002, 1003})
        # selective pruning: plane arrays of EVERY prod realization, all arrays of the first five positions
        npy = {}
        for d in glob.glob(os.path.join(self.cdir, "segments", "prod", "*", "out", "realizations", "r_*")):
            seg = d.split(os.sep)[-4]
            sm = json.load(open(os.path.join(self.cdir, "segments", "prod", seg, "seedmap.json")))["seed0"]
            seed = sm + int(os.path.basename(d)[2:])
            npy[seed] = {os.path.basename(x)[:-4] for x in glob.glob(os.path.join(d, "**", "*.npy"), recursive=True)}
        self.assertEqual(sorted(npy), list(range(1000, 1016)))
        for seed, names in npy.items():
            self.assertTrue({"seeds", "weights", "planes_x23", "planes_tau"} <= names, (seed, names))
        self.assertTrue(all(len(npy[s]) == 9 for s in range(1000, 1005)), {s: len(npy[s]) for s in range(1000, 1006)})  # first 5 positions: everything
        self.assertTrue(any(len(npy[s]) == 4 for s in range(1005, 1016)), "nothing was pruned")
        self.assertRegex(rep, r"retention.*\n(  \S+ +\d+ / \d+\n)+")
        # lane pins, determinism pair, background job, peak memory
        units = [r for r in kind("unit_end") if r["arm"] in ("det_x", "det_y")]
        self.assertEqual({(r["arm"], r["lane"]) for r in units}, {("det_x", "A"), ("det_y", "B")})
        with open(os.path.join(self.cdir, "DETERMINISM.txt")) as f:
            det = f.read()
        self.assertIn("PASS det_x det_y", det, det[-1500:])
        self.assertNotIn("FAIL", det)
        self.assertIn("determinism: pair det_x,det_y PASS", rep)
        self.assertEqual([r["rc"] for r in kind("background_end")], [0])
        self.assertRegex(rep, r"background jobs:\n  cov +(done rc=0|skipped)")  # the resumed session finds the done file of the first one
        peaks = {r["peak_mib"] for r in kind("unit_end") if r["arm"] == "prod" and r["cls"] == "ok"}
        self.assertTrue(peaks and min(peaks) >= 4096)  # MEM_PHASE device_used of the fake: 4 GiB + r bytes
        self.assertRegex(rep, r"\nprod +\d+ +16 .* 4096 ")

    def segment(self, arm, seed0):
        """Name of the segment directory whose seed0 is `seed0` (the segmentation depends on the kill)."""
        for d in sorted(glob.glob(os.path.join(self.cdir, "segments", arm, "*"))):
            if json.load(open(os.path.join(d, "seedmap.json")))["seed0"] <= seed0 < json.load(open(os.path.join(d, "seedmap.json")))["seed0"] + 4 \
                    and os.path.isdir(os.path.join(self.cdir, "reduced", arm, os.path.basename(d))):
                return os.path.basename(d)
        self.fail("no segment of %s contains seed %d" % (arm, seed0))


if __name__ == "__main__":
    unittest.main()
