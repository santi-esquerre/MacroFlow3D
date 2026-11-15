"""END-TO-END on CPU: the REAL generated addendum plan (scaled down), the real engine (`sf43camp.campaign.main`), the real `RunnerExecutor` and the
fake runner binaries of `sf43camp/tests_executor/fake_runner.py` (imported, not modified), the real SF-42 reducer / pruner, and the real
`recompute43` / `reconcile43` hooks on the addendum campaign layout; then `s0-report` on the smoke campaign and `finalize` with its costs.

What is scaled: seed counts (4 per arm, units of 2), the planes (a handful), `np_side` (8), the executor's GPU-memory gate; the Par2 time-window
keys `U_time` / `par2_exit_x` are dropped (the fake Par2 runner writes no time series; the frozen integration test does the same).  What is NOT
scaled: the factors (hz, dt_half, grid, ell, window), groups, priorities, retained seeds, the hooks' arguments.  The analysis command is a stub
(the sibling node's package `sf43add.ana` is not part of this node).  Statistical values of the fake data are never asserted.
"""
import glob
import io
import json
import os
import re
import signal
import stat
import sys
import tempfile
import time
import unittest
from unittest import mock

from sf43add.camp import cli
from sf43add.camp import plan as P
from sf43camp import campaign, lanes
from sf43camp.plan import load_plan, utc_iso
from sf43camp.runner_executor import RunnerExecutor
from sf43camp.tests_executor import test_e2e as base            # the frozen e2e module: its fake-runner location (module not run from here)

SCRIPTS, SF42 = base.SCRIPTS, base.SF42
PLANES_PAR2 = [25, 30, 40, 60, 94]
PLANES_CONS = [25, 40, 60, 94, 120, 150, 194]
ORDER = ["j1_base10", "j1_H3", "j1_D", "j1_DH3", "j1_H2", "j3_l5", "j9_w48"]
STUB = base.STUB


def shrink(a, counts):
    a["min_free_mib"], a["est_seconds"] = 1, 1.0                      # no GPU-memory gate, one second per realization
    if counts:                                                        # (smoke arms keep their own S0 units)
        a["seeds"]["count"] = a["n_conf"] = a["n_max"] = 4
        a["unit_size"] = 2
    run = a[a["runner"]]
    if a["runner"] == "par2":
        run["planes_d"] = PLANES_PAR2
        a["reduce"].pop("U_time", None)                               # the fake Par2 runner writes no time series
        a["reduce"].pop("par2_exit_x", None)
    else:
        run["planes_d"], run["np_side"] = PLANES_CONS, 8


def scale(plan, root, reports, stub):
    """In place: the plan as the engine runs it in the test (see the module docstring); the smoke overlay too, when there is one."""
    for a in plan["arms"]:
        shrink(a, True)
    plan.update(binaries={}, disk_min_free_gb=0, poll_s=0.2, grace_minutes=1, budget_gpu_h=100, cutoff_utc=utc_iso(time.time() + 3600),
                campaign_dir=os.path.join(root, "camp"))
    plan["analysis"] = {"command": [sys.executable, stub, "{campaign_dir}", reports], "sha256_list": {}}
    for name in ("recompute", "reconcile"):
        c = plan[name]["command"]
        c[0] = sys.executable
        c[c.index("--out") + 1] = os.path.join(reports, name + ".txt")
    if "smoke" in plan:
        sm = plan["smoke"]
        sm.update(campaign_dir=os.path.join(root, "smoke"), cutoff_minutes=60, budget_gpu_h=50, analysis=plan["analysis"])
        for a in sm["arms"]:
            shrink(a, False)
    return plan


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
                f.write('#!/bin/sh\nexec %s %s %s "$@"\n' % (sys.executable, os.path.join(base.HERE, "fake_runner.py"), mode))
            os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR)
        self.reports = os.path.join(self.root, "reports")
        os.makedirs(self.reports)
        self.stub = os.path.join(self.root, "stub_analysis.py")
        with open(self.stub, "w") as f:
            f.write(STUB)
        for sig in (signal.SIGTERM, signal.SIGINT):
            self.addCleanup(signal.signal, sig, signal.getsignal(sig))
        env = mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": SCRIPTS + os.pathsep + SF42})       # ONE GPU: lane A only
        env.start()
        self.addCleanup(env.stop)
        pr = mock.patch.object(lanes, "PRUNE_S", 0.3)
        pr.start()
        self.addCleanup(pr.stop)

    def run_plan(self, plan, *extra):
        pf = os.path.join(self.root, "plan.json")
        with open(pf, "w") as f:
            f.write(P.dumps(plan))
        self.assertIsInstance(load_plan(pf), dict)
        rc = campaign.main([self.build, "all", "--plan", pf] + list(extra))
        return rc, plan["campaign_dir"] if "--smoke" not in extra else plan["smoke"]["campaign_dir"]

    def report(self, cdir):
        txt = ""
        for p in (os.path.join(cdir, "REPORT.txt"), os.path.join(self.reports, "recompute.txt"), os.path.join(self.reports, "reconcile.txt")):
            if os.path.exists(p):
                with open(p) as f:
                    txt += "\n--- %s\n%s" % (os.path.basename(p), f.read()[-2500:])
        return txt

    @staticmethod
    def read(path):
        with open(path) as f:
            return f.read()

    @staticmethod
    def ledger(cdir):
        with open(os.path.join(cdir, "ledger.jsonl")) as f:
            return [json.loads(x) for x in f if x.strip()]

    def check_hooks(self, cdir, rc):
        rep = self.report(cdir)
        self.assertEqual(rc, campaign.DONE, rep)
        self.assertTrue(os.path.exists(os.path.join(cdir, "DONE")) and not os.path.exists(os.path.join(cdir, "ANALYSIS_FAILED")), rep)
        self.assertFalse(os.path.exists(os.path.join(cdir, "ANALYSIS_BLOCKED_UNREDUCED")))
        with open(os.path.join(self.reports, "analysis_ran.txt")) as f:
            self.assertEqual(f.read().strip(), cdir)
        with open(os.path.join(self.reports, "recompute.txt")) as f:
            rec = f.read()
        self.assertTrue(rec.rstrip().endswith("RECOMPUTE PASS"), rec[-1500:])
        with open(os.path.join(self.reports, "reconcile.txt")) as f:
            self.assertTrue(f.read().rstrip().endswith("RECONCILE OK"))

    # ------------------------------------------------------------------------------------------------------------------------------
    def test_production_plan_end_to_end(self):
        """Order, factors, geometry, reduction and the recompute / reconcile hooks.  min_retained = 4 = every realization: the fake consumer writes
        no moment CSVs, which recompute43 needs for a realization whose arrays were pruned (the frozen integration test does the same)."""
        plan = scale(P.build_plan(allow_missing=True), self.root, self.reports, self.stub)
        plan["min_retained"] = 4
        rc, cdir = self.run_plan(plan)
        self.check_hooks(cdir, rc)
        L = self.ledger(cdir)
        # the owner's order, one arm after the other
        arms = [r["arm"] for r in L if r["kind"] == "claim"]
        order = [a for i, a in enumerate(arms) if i == 0 or a != arms[i - 1]]
        self.assertEqual(order, ORDER)
        self.assertEqual({a: sum(1 for r in L if r["kind"] == "seed" and r["arm"] == a) for a in ORDER}, {a: 4 for a in ORDER})
        self.assertFalse([r for r in L if r["kind"] in ("replacement", "arm_halt", "lane_crash", "unreduced")])
        # Par2 factors reach the pipeline: steps x3 / x2 / x6 / x2 (hz), dt / 2 (dt_half) with EXACTLY twice the steps
        cfg = {a: self.par2_cfg(cdir, a) for a in ORDER[:5]}
        n = {a: c["n_steps"] for a, c in cfg.items()}
        dt = {a: c["dt"] for a, c in cfg.items()}
        self.assertEqual(n["j1_D"], 2 * n["j1_base10"])
        self.assertEqual(n["j1_DH3"], 2 * n["j1_H3"])
        self.assertEqual(dt["j1_D"], dt["j1_base10"] / 2)
        self.assertEqual((dt["j1_H3"], dt["j1_H2"], dt["j1_DH3"]), (dt["j1_base10"], dt["j1_base10"], dt["j1_D"]))
        for a, m in (("j1_H3", 3), ("j1_H2", 2)):
            self.assertAlmostEqual(n[a] / n["j1_base10"], m, delta=1e-3)
        for a in ORDER[:5]:                                                       # everything else of the pipeline config is the origin's
            self.assertEqual(cfg[a]["seed_stream"], 420800001 + 1000 * 434001)
        # consumer geometry reaches the runner: the command the engine builds, and the log of the executed segment
        ex = RunnerExecutor(plan, self.build)
        for a, grid, ell, win in (("j3_l5", "1040,120,120", "0.041666666666666664", "0.1,0.9,0.1,0.9"), ("j9_w48", "1040,240,240", "0.020833333333333332", "0.3,0.7,0.3,0.7")):
            arm = next(x for x in plan["arms"] if x["id"] == a)
            argv = ex.command(arm, [arm["seeds"]["start"]], os.path.join(self.root, "x"), self.build)["argv"]
            self.assertEqual((argv[argv.index("--grid") + 1], argv[argv.index("--ell") + 1], argv[argv.index("--window") + 1]), (grid, ell, win))
            self.assertEqual(argv[argv.index("--hmax-div") + 1], "8")
            log = self.read(glob.glob(os.path.join(cdir, "segments", a, "*", "run.log"))[0])
            self.assertIn("grid=%sx%sx%s" % tuple(grid.split(",")), log)
        # reduced tables: one per segment and window; consumer rows carry the label -1 (single pass, no hmax_div_first)
        for a, windows in (("j1_H3", 1), ("j3_l5", 2), ("j9_w48", 2)):
            tabs = glob.glob(os.path.join(cdir, "reduced", a, "*", "%s__*__d*.csv" % a))
            tabs = [t for t in tabs if not t.endswith(".planes.csv")]
            self.assertEqual(len(tabs), 2 * windows, tabs)
            if a != "j1_H3":
                self.assertEqual({x.split(",")[3] for t in tabs for x in self.read(t).splitlines()[1:]}, {"-1"})
        rep = self.report(cdir)
        for a in ORDER:
            self.assertRegex(rep, r"\n%s +4 +4 +0 +0 +0 +0 " % a)

    def test_retention_and_pruning(self):
        """`retained` (the first two seeds of each arm) keeps its raw data, the rest is pruned after the reduction.  No recompute hook (see above);
        reconcile and the analysis stub still run.  The last arm may not have been pruned yet when the engine exits, so the consumer check is 'some'."""
        plan = scale(P.build_plan(allow_missing=True), self.root, self.reports, self.stub)
        del plan["recompute"]
        rc, cdir = self.run_plan(plan)
        self.assertEqual(rc, campaign.DONE, self.report(cdir))
        some = False
        for a in ("j3_l5", "j9_w48"):
            kept = {}
            for d in glob.glob(os.path.join(cdir, "segments", a, "*", "out", "realizations", "r_*")):
                seg = d.split(os.sep)[-4]
                s0 = P.jload(os.path.join(cdir, "segments", a, seg, "seedmap.json"))["seed0"]
                kept[s0 + int(os.path.basename(d)[2:])] = len(glob.glob(os.path.join(d, "**", "*.npy"), recursive=True))
            first = plan["arms"][ORDER.index(a)]["seeds"]["start"]
            self.assertEqual(sorted(kept), list(range(first, first + 4)))
            self.assertTrue(kept[first] == kept[first + 1] > 0, kept)                  # retained: all arrays (the Pollock ones included)
            self.assertEqual(kept[first + 2], kept[first + 3])
            some = some or kept[first + 2] < kept[first]
        self.assertTrue(some, "nothing was pruned")
        snaps = glob.glob(os.path.join(cdir, "segments", "j1_H3", "*", "out", "snapshots", "r_*", "*"))
        self.assertEqual(len(snaps), 2)                                                # Par2: the retained segment's snapshots stay, the other segment's are deleted
        rep = self.report(cdir)
        held = re.findall(r"\n  (\S+) +(\d+) / (\d+)(?=\n)", rep.split("retention", 1)[1])
        self.assertEqual({h[0] for h in held} & set(ORDER), set(ORDER))
        self.assertTrue(all(int(h) >= int(n) == 2 for _, h, n in held), held)

    def par2_cfg(self, cdir, arm):
        txt = self.read(sorted(glob.glob(os.path.join(cdir, "segments", arm, "*", "config.yaml")))[0])
        return {"n_steps": int(re.search(r"n_steps: (\d+)", txt).group(1)), "dt": float(re.search(r"\n  dt: (\S+)", txt).group(1)),
                "seed_stream": int(re.search(r"\n  seed: (\d+)\n  output_every", txt).group(1))}

    # ------------------------------------------------------------------------------------------------------------------------------
    def test_smoke_then_s0_report_then_finalize(self):
        plan = scale(P.build_plan(allow_missing=True), self.root, self.reports, self.stub)
        rc, cdir = self.run_plan(plan, "--smoke")
        self.assertEqual(cdir, os.path.join(self.root, "smoke"))
        self.check_hooks(cdir, rc)
        # S0: nothing pruned (every seed retained), one unit per arm
        self.assertFalse([r for r in self.ledger(cdir) if r["kind"] == "pruned" and r["info"].get("deleted_files")])
        out = os.path.join(self.root, "costs.json")
        with mock.patch("sys.stdout", new=io.StringIO()) as so:
            self.assertEqual(cli.main(["s0-report", "--campaign", cdir, "--out", out]), 0)
        self.assertIn("none triggered", so.getvalue())
        costs = P.jload(out)
        self.assertEqual(sorted(costs["arms"]), sorted(["j1_base10", "j1_H3", "j1_DH3", "j3_l5", "j9_w48"]))
        self.assertEqual(costs["drop"], [])
        j9 = costs["arms"]["j9_w48"]
        self.assertEqual((j9["n"], j9["planned"], j9["seconds_per_realization"], j9["verdicts"]), (1, 1, 10.0, {"converged": 1}))
        self.assertGreaterEqual(j9["peak_mib"], 4096)                              # MEM_PHASE device_used of the fake: 4 GiB + r bytes
        self.assertEqual(j9["max_iterations"], 30)
        self.assertGreater(j9["raw_gb_per_realization"], 0)
        self.assertGreater(costs["arms"]["j1_H3"]["raw_gb_per_realization"], 0)    # Par2: crossings + stats + snapshots
        self.assertIsNone(costs["arms"]["j1_H3"]["peak_mib"])
        # finalize: the plan the production job would run
        fin = os.path.join(self.root, "final.json")
        with mock.patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(cli.main(["finalize", "--costs", out, "--budget-gpu-h", "6.5", "--disk-min-free-gb", "90", "--allow-missing", "--out", fin]), 0)
        p = load_plan(fin)
        self.assertEqual((p["budget_gpu_h"], p["disk_min_free_gb"]), (6.5, 90))
        got = {a["id"]: a for a in p["arms"]}
        self.assertEqual(got["j9_w48"]["est_seconds"], 12.5)                       # 1.25 x 10 s
        self.assertEqual(got["j9_w48"]["min_free_mib"], max(int(-(-1.25 * j9["peak_mib"] // 1)), 9000))
        self.assertTrue(all(a["est_seconds"] > 0 for a in p["arms"]))
        self.assertEqual(len(p["arms"]), 7)                                        # j1_D and j1_H2 are fitted, not dropped

    def test_j8_plan_end_to_end(self):
        plan = scale(P.build_plan(job="j8", allow_missing=True), self.root, self.reports, self.stub)
        rc, cdir = self.run_plan(plan)
        self.check_hooks(cdir, rc)
        L = self.ledger(cdir)
        self.assertEqual({r["arm"] for r in L if r["kind"] == "seed"}, {"j8_H3", "j8_D", "j8_DH3"})
        self.assertEqual(sorted({r["seed"] for r in L if r["kind"] == "seed"}), list(range(434061, 434065)))


if __name__ == "__main__":
    unittest.main()
