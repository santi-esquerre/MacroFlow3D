"""S0 report on synthetic smoke campaigns: costs, the pre-registered cut rules, and that no result file is ever opened."""
import builtins
import io
import json
import os
import tempfile
import unittest
from unittest import mock

from sf43add.camp import cli
from sf43add.camp import plan as P
from sf43add.camp import s0

PRE = P.jload(os.path.join(P.ART, "prereg_addendum.json"))
UNITS = PRE["smoke"]["units"]
MB = 1 << 20
TRAP = "alpha_TP TRAP"


def unit(arm, seeds, cls="ok", sec=10.0, peak=None, iters=40, verdict="converged", wall=None, completed=None, reduced=True):
    return dict(arm=arm, seeds=list(seeds), cls=cls, sec=sec, peak=peak, iters=iters, verdict=verdict, wall=wall, reduced=reduced,
                completed=seeds if completed is None else completed)


def default_units(**over):
    spec = {"j1_base10": [unit("j1_base10", UNITS["j1_base10"], sec=17.0)], "j1_H3": [unit("j1_H3", UNITS["j1_H3"], sec=50.0)],
            "j1_DH3": [unit("j1_DH3", UNITS["j1_DH3"], sec=100.0)], "j3_l5": [unit("j3_l5", UNITS["j3_l5"], sec=7.0, peak=7000.0, iters=60)],
            "j9_w48": [unit("j9_w48", UNITS["j9_w48"], sec=40.0, peak=12000.0, iters=80)]}
    spec.update(over)
    return spec


def make_campaign(root, spec, gpu_h=0.25, report=True, torn=False):
    os.makedirs(os.path.join(root, "reduced", "j3_l5"))
    with open(os.path.join(root, "reduced", "j3_l5", "t.csv"), "w") as f:
        f.write(TRAP + "\n")
    os.makedirs(os.path.join(root, "analysis"))
    with open(os.path.join(root, "analysis", "summary.json"), "w") as f:
        f.write(TRAP)
    led, n = [{"kind": "campaign_start"}, {"kind": "hb", "gpu_s": gpu_h * 3600.0}], 0
    for arm, units in spec.items():
        par2 = arm.startswith("j1_")
        for u in units:
            n += 1
            s0_ = u["seeds"][0]
            rel = "segments/%s/u%06d_s%d" % (arm, n, s0_)
            seg = os.path.join(root, rel)
            os.makedirs(seg)
            with open(os.path.join(seg, "run.log"), "w") as f:
                f.write("some line\nERROR: boom in %s\n" % arm)
            with open(os.path.join(seg, "seedmap.json"), "w") as f:
                json.dump({"seed0": s0_}, f)
            for s in u["completed"]:
                r = s - s0_
                if par2:
                    open(os.path.join(seg, "config.yaml"), "w").close()
                    for d, name, size in (("crossings", "realization_%04d_planes.csv" % r, 10), ("stats", "realization_%04d_timeseries.csv" % r, 20)):
                        os.makedirs(os.path.join(seg, "out", d), exist_ok=True)
                        with open(os.path.join(seg, "out", d, name), "wb") as f:
                            f.write(b"x" * size * 1024)
                    os.makedirs(os.path.join(seg, "out", "snapshots", "r_%04d" % r))
                    with open(os.path.join(seg, "out", "snapshots", "r_%04d" % r, "s.csv"), "wb") as f:
                        f.write(b"x" * MB)
                else:
                    os.makedirs(os.path.join(seg, "out", "realizations", "r_%04d" % r))
                    with open(os.path.join(seg, "out", "realizations", "r_%04d" % r, "a.npy"), "wb") as f:
                        f.write(b"x" * 3 * MB)
            if u.get("reduced", True) and u["completed"]:
                led.append({"kind": "reduced", "arm": arm, "segment": rel, "seeds": u["completed"], "reduced": {}})
            led.append({"kind": "unit_end", "unit": n, "arm": arm, "lane": "A", "gpu": "0", "cls": u["cls"], "completed": u["completed"], "failed_seed": None,
                        "failed_seeds": [], "next_seed": None, "seconds": {str(s): u["sec"] for s in u["completed"]}, "verdicts": {},
                        "peak_mib": u["peak"], "wall": u["wall"] if u["wall"] is not None else u["sec"] * len(u["completed"]) + 4.0,
                        "segment": rel, "segment_seed0": s0_})
            for s in u["completed"]:
                led.append({"kind": "seed", "arm": arm, "group": "g", "segment": os.path.basename(rel), "segment_path": rel, "seed": s, "seed0": s0_, "unit": n,
                            "verdict": u["verdict"], "class": "ok" if u["verdict"] in s0.USABLE else "unusable_solve", "iterations": u["iters"]})
    with open(os.path.join(root, "ledger.jsonl"), "w") as f:
        for r in led:
            f.write(json.dumps(r) + "\n")
        if torn:
            f.write('{"kind": "unit_end", "arm": "j9_w')
    if report:
        with open(os.path.join(root, "REPORT.txt"), "w") as f:
            f.write("SF-43 campaign sf43-addendum-1: DONE\nanalysis: ok\nstage1_start: x\nlane_b_released: False\ngpu_h_used: %.2f of 1.00\nstopping: None\n" % gpu_h)


class Report(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.root = os.path.join(self._t.name, "smoke")
        os.makedirs(self.root)

    def run_report(self, spec, **kw):
        make_campaign(self.root, spec, **kw)
        out, w = os.path.join(self._t.name, "costs.json"), io.StringIO()
        rc = s0.run(self.root, os.path.join(P.ART, "prereg_addendum.json"), out, True, w)
        with open(out) as f:
            return rc, json.load(f), w.getvalue()

    def test_costs_and_no_cut(self):
        rc, c, text = self.run_report(default_units())
        self.assertEqual(rc, 0)
        self.assertEqual(c["drop"], [])
        self.assertEqual((c["status"], c["gpu_h_held"], c["suggested_budget_gpu_h"]), ("DONE", 0.25, 7.25))
        a = c["arms"]
        self.assertEqual((a["j9_w48"]["n"], a["j9_w48"]["seconds_per_realization"], a["j9_w48"]["peak_mib"], a["j9_w48"]["max_iterations"]), (1, 40.0, 12000.0, 80))
        self.assertEqual((a["j1_H3"]["n"], a["j1_H3"]["seconds_per_realization"], a["j1_H3"]["peak_mib"], a["j1_H3"]["wall_per_realization"]), (2, 50.0, None, 52.0))
        self.assertEqual(a["j9_w48"]["verdicts"], {"converged": 1})
        # du: consumer realization = one 3 MiB array; Par2 = crossings 10 KiB + series 20 KiB + one 1 MiB snapshot file
        self.assertAlmostEqual(a["j3_l5"]["raw_gb_per_realization"], 3 * MB / 1e9, delta=0.03 * MB / 1e9)
        self.assertAlmostEqual(a["j1_H3"]["raw_gb_per_realization"], (MB + 30 * 1024) / 1e9, delta=0.1 * MB / 1e9)
        self.assertIn("none triggered", text)
        self.assertIn("suggested budget_gpu_h = 7.25", text)
        # usable by `plan --costs`
        plan = P.build_plan(costs=c, budget_gpu_h=7.25, disk_min_free_gb=90, allow_missing=True)
        got = {x["id"]: x for x in plan["arms"]}
        self.assertEqual((got["j9_w48"]["est_seconds"], got["j9_w48"]["min_free_mib"]), (50.0, 15000))
        self.assertEqual(got["j1_H3"]["est_seconds"], 62.5)

    def test_j9_cut_rules(self):
        for over, frag in ((dict(sec=180.5), "s per realization"), (dict(peak=20481.0), "peak"), (dict(iters=1501), "CG iterations")):
            with self.subTest(frag), tempfile.TemporaryDirectory() as t:
                self.root = os.path.join(t, "smoke")
                os.makedirs(self.root)
                kw = dict(sec=40.0, peak=12000.0, iters=80)
                kw.update(over)
                rc, c, text = self.run_report(default_units(j9_w48=[unit("j9_w48", UNITS["j9_w48"], **kw)]))
                self.assertEqual(c["drop"], ["j9_w48"])
                self.assertIn(frag, c["decisions"]["j9_w48"][0])
                plan = P.build_plan(costs=c, allow_missing=True)
                self.assertNotIn("j9_w48", [a["id"] for a in plan["arms"]])
        # the thresholds themselves are not exceeded
        with tempfile.TemporaryDirectory() as t:
            self.root = os.path.join(t, "smoke")
            os.makedirs(self.root)
            rc, c, _ = self.run_report(default_units(j9_w48=[unit("j9_w48", UNITS["j9_w48"], sec=180.0, peak=20480.0, iters=1500)]))
            self.assertEqual(c["drop"], [])

    def test_j9_rejected_geometry_and_unmeasured(self):
        rc, c, text = self.run_report(default_units(j9_w48=[unit("j9_w48", UNITS["j9_w48"], cls="config_error", completed=[])]))
        self.assertEqual(c["drop"], ["j9_w48"])
        self.assertIn("rejects the geometry", c["decisions"]["j9_w48"][0])
        self.assertIn("ERROR: boom", json.dumps(c["arms"]["j9_w48"]["structural_failures"]))          # the log tail is quoted for the orchestrator
        with tempfile.TemporaryDirectory() as t:
            self.root = os.path.join(t, "smoke")
            os.makedirs(self.root)
            rc, c, text = self.run_report(default_units(j9_w48=[unit("j9_w48", UNITS["j9_w48"], cls="oom", completed=[])]))
            self.assertEqual(c["drop"], [])                                                            # OOM is not a listed rule: reported, orchestrator decides
            self.assertTrue(any("j9_w48: NOT MEASURED" in n for n in c["notes"]))

    def test_j1_structural_failure_stops_all_of_j1(self):
        for bad in (unit("j1_H3", UNITS["j1_H3"], cls="other_error", completed=[434001]), unit("j1_H3", UNITS["j1_H3"], verdict="not_converged_beyond_fallback")):
            with tempfile.TemporaryDirectory() as t:
                self.root = os.path.join(t, "smoke")
                os.makedirs(self.root)
                rc, c, text = self.run_report(default_units(j1_H3=[bad]))
                self.assertEqual(c["drop"], ["j1_D", "j1_DH3", "j1_H2", "j1_H3", "j1_base10"])
                plan = P.build_plan(costs=c, allow_missing=True)
                self.assertEqual([a["id"] for a in plan["arms"]], ["j3_l5", "j9_w48"])
        self.assertIn("STRUCTURAL", text)

    def test_a_failed_reduction_is_structural(self):
        rc, c, text = self.run_report(default_units(j3_l5=[unit("j3_l5", UNITS["j3_l5"], reduced=False)]))
        self.assertEqual(c["drop"], ["j3_l5"])
        self.assertIn("completed but was not reduced", text)

    def test_j3_failure_and_missing_arm(self):
        rc, c, _ = self.run_report(default_units(j3_l5=[unit("j3_l5", UNITS["j3_l5"], cls="other_error", completed=[])]))
        self.assertEqual(c["drop"], ["j3_l5"])
        with tempfile.TemporaryDirectory() as t:
            self.root = os.path.join(t, "smoke")
            os.makedirs(self.root)
            spec = default_units()
            del spec["j1_DH3"]                                                                          # an arm that never ran is a failure too
            rc, c, _ = self.run_report(spec)
            self.assertEqual(c["drop"], ["j1_D", "j1_DH3", "j1_H2", "j1_H3", "j1_base10"])

    def test_torn_ledger_and_hb_fallback(self):
        rc, c, _ = self.run_report(default_units(), report=False, torn=True)
        self.assertEqual((rc, c["status"], c["gpu_h_held"], c["drop"]), (0, None, 0.25, []))

    def test_never_opens_a_result(self):
        make_campaign(self.root, default_units())
        opened, real_open, real_walk = [], builtins.open, os.walk

        def spy_open(f, *a, **k):
            opened.append(os.fspath(f))
            return real_open(f, *a, **k)

        def spy_walk(top, *a, **k):
            opened.append(os.fspath(top))
            return real_walk(top, *a, **k)
        w = io.StringIO()
        with mock.patch.object(builtins, "open", spy_open), mock.patch.object(os, "walk", spy_walk):
            s0.run(self.root, os.path.join(P.ART, "prereg_addendum.json"), os.path.join(self._t.name, "c.json"), True, w)
        self.assertTrue(opened)
        self.assertFalse([p for p in opened if os.sep + "reduced" in p or os.sep + "analysis" in p or p.endswith(".csv") and "crossings" not in p], opened)
        with open(os.path.join(self._t.name, "c.json")) as f:
            self.assertNotIn("TRAP", f.read())
        self.assertNotIn("TRAP", w.getvalue())

    def test_cli(self):
        make_campaign(self.root, default_units())
        out = os.path.join(self._t.name, "out.json")
        with mock.patch("sys.stdout", new=io.StringIO()) as so:
            self.assertEqual(cli.main(["s0-report", "--campaign", self.root, "--out", out]), 0)
        self.assertIn("S0 REPORT", so.getvalue())
        self.assertTrue(os.path.isfile(out))
        with mock.patch("sys.stdout", new=io.StringIO()):
            self.assertEqual(cli.main(["s0-report", "--campaign", self._t.name + "/nowhere"]), 1)


if __name__ == "__main__":
    unittest.main()
