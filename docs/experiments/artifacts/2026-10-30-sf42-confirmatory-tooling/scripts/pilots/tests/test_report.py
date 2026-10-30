import csv
import json
import os
import tempfile
import unittest

import numpy as np

from mcstat import costmodel as cm
from pilots import cost, report, tables
from pilots.tests.test_analyze import cons_rows


def write_table(path, rows):
    cols = sorted({k for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


class TestReportJ2(unittest.TestCase):
    def test_j2_recovers_inverse_square_and_conclusion(self):
        rng = np.random.default_rng(3)
        n = 6
        base = 0.02 + 0.01 * rng.standard_normal(n)
        Ms = (64, 128, 192)
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "J2", "reduced"))
            for M in Ms:
                # half-lattice estimates differ by +-d/M (so (aA-aB)^2/4 = d^2/M^2... exactly an inverse-square law)
                d = 1.9e-2 * 64.0 / M
                A, B = base + d, base - d
                rid = "c_s1_M%d" % M
                os.makedirs(os.path.join(td, "J2", rid))
                with open(os.path.join(td, "J2", rid, "RUN.json"), "w") as f:
                    json.dump({"id": rid, "amp": "1", "exit": 0, "M": M, "nx": 1040, "ny": 240, "nz": 240, "lh": 10}, f)
                rows = cons_rows(list(base), list(A), list(B)) + cons_rows(list(base), list(A), list(B), basis="disp")
                for x in rows:
                    x.update(n_traced="100", n_first="100", n_last="99")
                write_table(os.path.join(td, "J2", "reduced", rid + ".csv"), rows)
            sd = {k: {e + "_" + b: {"sd": 0.01, "mean": 0.02, "n": 50} for e in ("alpha_TP", "alpha_TAP", "alpha_LP") for b in ("pos", "disp")}
                  for k in ("s025", "s1", "s225", "s4")}
            with open(os.path.join(td, "hostref.json"), "w") as f:
                json.dump({"pcg": {}, "planning_sd": sd}, f)
            out = os.path.join(td, "out")
            os.makedirs(out)
            res = report.j2(td, out)
            self.assertTrue(os.path.exists(os.path.join(out, "j2_components.csv")))
        r = res["1"]["estimators"]["alpha_TP/pos"]
        self.assertAlmostEqual(r["exponent"], -2.0, places=9)
        comp64 = (1.9e-2) ** 2  # mean (A-B)^2/4 = d^2
        self.assertAlmostEqual(r["by_M"][64]["component"], comp64, places=15)
        # share(M) = 3.61 (64/M)^2: <= 0.25 needs M >= 243.2 -> 256 (admissible multiples of 64); <= 0.10 needs M >= 384.5 -> 448
        self.assertEqual(r["M_0.25_fixed"], 256)
        self.assertEqual(res["1"]["M_0.25_pos"], 256)
        self.assertEqual(r["M_0.10_fixed"], 448)
        self.assertEqual(res["1"]["consumer_crossing"][64]["min_last_over_injected"], 0.99)


class TestReportJ1Rule(unittest.TestCase):
    def test_three_times_rule(self):
        with tempfile.TemporaryDirectory() as td:
            os.makedirs(os.path.join(td, "J1", "c1"))
            os.makedirs(os.path.join(td, "J1", "c2"))
            for rid, it in (("c1", 90), ("c2", 100)):
                rec = {"id": rid, "code": "consumer", "nx": 520, "ny": 120, "nz": 120, "lh": 5, "amp": "1", "M": 96, "seed0": 1, "exit": 0,
                       "wall_s": 10.0, "peak_gpu_mib_own_pid": 5000,
                       "realizations": [{"iterations": it, "verdict": "converged", "seconds": {"stage": 1.0},
                                         "mem_lines_stage": ["MEM_PHASE phase=S2 exact_bytes=1048576 x"], "mem_line_bridge": ""}]}
                with open(os.path.join(td, "J1", rid, "RUN.json"), "w") as f:
                    json.dump(rec, f)
            ref = {k: {"production": {"median": 30}} for k in ("s025", "s1", "s225", "s4")}
            with open(os.path.join(td, "hostref.json"), "w") as f:
                json.dump({"pcg": ref, "planning_sd": {}}, f)
            out = os.path.join(td, "out")
            os.makedirs(out)
            rows = {r["id"]: r for r in report.j1(td, out)}
        self.assertFalse(rows["c1"]["different_grid_required"])      # 90 = 3.0 x 30: not above 3x
        self.assertTrue(rows["c2"]["different_grid_required"])       # 100 > 90
        self.assertAlmostEqual(rows["c1"]["runner_exact_bytes_max_mib"], 1.0)


class TestCrossing(unittest.TestCase):
    def row(self, code, r, traced, first, last, weights="unif"):
        return {"code": code, "family": "plane", "weights": weights, "basis": "pos", "set": "perplane", "r": str(r), "n_traced": str(traced),
                "n_first": str(first), "n_last": str(last)}

    def test_min_over_realizations_and_code_filter(self):
        rows = [self.row("par2", 0, 100, 100, 99), self.row("par2", 1, 100, 98, 90), self.row("consumer", 0, 50, 50, 25),
                self.row("par2", 2, 100, 100, 100, weights="flux")]
        p = report.crossing_fraction(rows)
        self.assertEqual((p["n_realizations"], p["min_last_over_injected"], p["min_first_over_injected"]), (3, 0.9, 0.98))
        c = report.crossing_fraction(rows, "consumer", "unif")
        self.assertEqual((c["n_realizations"], c["min_last_over_injected"]), (1, 0.5))
        with self.assertRaises(ValueError):
            report.crossing_fraction(rows, "consumer", "flux")


class TestScenarioTotals(unittest.TestCase):
    def setUp(self):
        self.lag = cm.lagrangian_arm(100, 10, 10, 1.0, 100.0, (8,), 70, "L")
        self.par = cm.par2_arm(100, 10, 10, 1.0, 1e4, cost.PAR2_H36_TAG, "P")
        self.prow = [{"arm": self.lag}, {"arm": self.par}]
        self.rows = [{"code": "lagrangian", "gpu_h": 10.0}, {"code": cost.PAR2_H36_TAG, "gpu_h": 4.0}]

    def test_values(self):
        rms = {"lagrangian": 0.1, cost.PAR2_H36_TAG: 0.2}
        mx = {"lagrangian": 0.3, cost.PAR2_H36_TAG: 0.5}
        anchor = {("lagrangian", self.lag.cells, self.lag.nx): 1.5}            # only the consumer geometry was measured
        t = report.scenario_totals(self.rows, self.prow, anchor, rms, mx, reserve=0.05)
        self.assertAlmostEqual(t["total_minus_1rms"], 1.05 * (10 * 0.9 + 4 * 0.8))
        self.assertAlmostEqual(t["total_plus_1rms"], 1.05 * (10 * 1.1 + 4 * 1.2))
        self.assertAlmostEqual(t["total_plus_max_residual"], 1.05 * (10 * 1.3 + 4 * 1.5))
        self.assertAlmostEqual(t["total_anchored"], 1.05 * (15.0 + 4.0))
        self.assertAlmostEqual(t["total_anchored_plus_1rms"], 1.05 * (15.0 * 1.1 + 4.0 * 1.2))


class TestStopRule(unittest.TestCase):
    def proj(self, total, plus):
        return {"total_gpu_h": total, "total_plus_1rms": plus, "total_plus_max_residual": 0.0, "total_anchored": 0.0, "total_anchored_plus_1rms": 0.0}

    def test_rule_uses_total_and_plus_one_rms_only(self):
        ok = report.stop_rule(self.proj(100.0, 150.0), {"M_plan": self.proj(100.0, 150.0)}, 150.0)
        self.assertFalse(ok["stop_budget"])                      # not above the budget: 150 is allowed
        self.assertEqual(ok["verdict"], "WITHIN_BUDGET")
        self.assertAlmostEqual(ok["margin_gpu_h"]["total_M_plan"], 50.0)
        hi = report.stop_rule(self.proj(100.0, 150.01), {"M_plan": self.proj(100.0, 150.01)}, 150.0)
        self.assertTrue(hi["stop_budget"])
        self.assertEqual(hi["verdict"], "STOP_BUDGET")
        sens = dict(self.proj(100.0, 120.0), total_anchored=400.0)    # a sensitivity far above the budget does not enter the rule
        self.assertFalse(report.stop_rule(sens, {"M_plan": sens}, 150.0)["stop_budget"])


class TestHorizonTagAndTables(unittest.TestCase):
    def test_j3b_points_get_their_own_code_and_plan_uses_it(self):
        with tempfile.TemporaryDirectory() as td:
            for job, rid, hz in (("J3", "p1", 1.2), ("J3B", "p1_h3", 3.6)):
                d = os.path.join(td, job, rid)
                os.makedirs(d)
                with open(os.path.join(d, "RUN.json"), "w") as f:
                    json.dump({"id": rid, "code": "par2", "nx": 1040, "ny": 240, "nz": 240, "amp": "1", "nr": 2, "M": 0, "wall_s": 40.0, "exit": 0,
                               "par2_horizon_factor": hz, "par2_stage_times": [{"flow_stage_s": 5.0, "transport_s": 5.0}] * 2}, f)
            pts = {p.arm.label: p.arm.code for p in cost.points_from_runs(td)}
        self.assertEqual(pts, {"J3/p1": cost.PAR2_TAG, "J3B/p1_h3": cost.PAR2_H36_TAG})
        rows = cost.plan({a: 100 for a in cost.AMPS}, {a: 10 for a in cost.AMPS})
        self.assertEqual({r["arm"].code for r in rows if r["family"] in ("gate5_par2", "g7_par2")}, {cost.PAR2_H36_TAG})
        rows = cost.plan({a: 100 for a in cost.AMPS}, {a: 10 for a in cost.AMPS}, par2_tag=cost.PAR2_TAG)
        self.assertEqual({r["arm"].code for r in rows if r["family"] in ("gate5_par2", "g7_par2")}, {cost.PAR2_TAG})

    def test_stop_table(self):
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "cost_projection.json"), "w") as f:
                json.dump({"stop_rule": {"budget_gpu_h": 150.0, "verdict": "WITHIN_BUDGET", "checked": {"total_M_plan": 100.0},
                                         "margin_gpu_h": {"total_M_plan": 50.0}}}, f)
            t = tables.stop(td)
        self.assertIn("**WITHIN_BUDGET**", t)
        self.assertIn("| total_M_plan | 100.0 | 50.0 |", t)


if __name__ == "__main__":
    unittest.main()
