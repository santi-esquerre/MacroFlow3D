import os
import re
import unittest

import numpy as np

from mcstat import costmodel as cm
from mcstat.tests._paths import A


class Sf39Calibration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points, cls.excluded = cm.sf39_points(A)
        cls.model = cm.CostModel(cls.points)
        cls.by_label = {}
        for p in cls.points:
            cls.by_label.setdefault(p.arm.label, []).append(p)

    def test_realization_counts_of_the_walls_entries(self):
        self.assertTrue(all(p.n_real > 0 and p.wall_s > 0 for p in self.points))
        # extension entries run only the realizations beyond the pilot (the id carries the target count)
        self.assertEqual([p.n_real for p in self.by_label["s225_tw"]], [20, 6])
        self.assertEqual([p.n_real for p in self.by_label["s4_B"]], [51])
        self.assertEqual([p.n_real for p in self.by_label["par2_s1_ext"]], [30])
        self.assertEqual([p.n_real for p in self.by_label["par2_s025_ext"]], [5])
        self.assertEqual([p.n_real for p in self.by_label["s025_B"]], [5])
        self.assertEqual([p.n_real for p in self.by_label["s4_w12c"]], [17])
        # the no-op re-invocations (4 s) of the extension are not calibration points
        self.assertEqual([p.n_real for p in self.by_label["s1_A"]], [20])
        self.assertEqual(sum(p.n_real for p in self.by_label["s4_n16"]), 30)
        # the two copies of the campaign walls (campaign and campaign_s025) are counted once
        self.assertEqual(len([p for p in self.points if p.arm.label == "s1_n10"]), 1)

    def test_halted_arm_is_excluded_with_a_reason(self):
        self.assertEqual([j for j, _ in self.excluded], ["s4_w12:s7201:n30"])
        self.assertNotIn("s4_w12", self.by_label)
        self.assertIn("halted", self.excluded[0][1])

    def test_gpu_seconds_per_realization_from_the_walls(self):
        p = self.by_label["s225_B"][0]
        self.assertAlmostEqual(p.gpu_s_per_real, 4288 / 42.0, places=12)
        self.assertEqual(p.arm.cells, 3328 * 384 * 384)
        self.assertEqual(p.arm.sigma2, 2.25)
        self.assertEqual(self.by_label["s1_A"][0].arm.pass_work, 7.0)  # passes h/8, h/16, h/32
        self.assertEqual(self.by_label["par2_s225"][0].arm.cells, 2496 * 288 * 288)

    def test_fit_quality_is_stated_and_acceptable(self):
        summ = self.model.summary()
        self.assertEqual(set(summ), {"lagrangian", "par2", "par2_fs"})
        self.assertLess(summ["lagrangian"]["rms_rel"], 0.15)
        self.assertLess(summ["lagrangian"]["max_abs_rel"], 0.55)  # the 2x outlier of s4_tw (42 s against 22 s)
        self.assertLess(summ["par2"]["max_abs_rel"], 0.25)
        for code, s in summ.items():
            self.assertTrue(all(v >= 0.0 for v in s["coef"].values()), msg=code)  # NNLS
        rows = self.model.residuals()
        worst = max(rows, key=lambda r: abs(r["rel"]))
        self.assertEqual(worst["label"], "s4_tw")
        self.assertEqual(len(rows), len(self.points))

    def test_hold_out_checks_are_predicted_within_the_stated_scatter(self):
        hold = cm.sf39_holdout_points(A)
        self.assertEqual(len(hold), 14)
        rel = [abs(self.model.predict(p.arm) - p.gpu_s_per_real) / p.gpu_s_per_real for p in hold]
        self.assertLess(max(rel), 0.30)
        self.assertLess(float(np.median(rel)), 0.12)

    def test_predictions_are_monotone_in_the_cost_drivers(self):
        base = cm.lagrangian_arm(2048, 256, 256, 1.0)
        bigger = cm.lagrangian_arm(4096, 256, 256, 1.0)
        passes = cm.lagrangian_arm(2048, 256, 256, 1.0, hmax_div=(8, 16, 32))
        hot = cm.lagrangian_arm(2048, 256, 256, 4.0)
        t0 = self.model.predict(base)
        for other in (bigger, passes, hot):
            self.assertGreater(self.model.predict(other), t0)
        self.assertGreater(self.model.predict(cm.par2_arm(2496, 288, 288, 2.25)), self.model.predict(cm.par2_arm(2048, 256, 256, 1.0)))

    def test_projection_and_validity_range(self):
        inside = cm.lagrangian_arm(3328, 384, 384, 4.0, label="in")
        outside = cm.lagrangian_arm(3328, 480, 480, 4.0, label="out")  # 7.7e8 cells: above the calibrated 4.9e8
        proj = self.model.project([(inside, 100), {"arm": outside, "n_real": 10}])
        self.assertEqual(proj["out_of_range"], ["out"])
        want = (100 * self.model.predict(inside) + 10 * self.model.predict(outside)) / 3600.0
        self.assertAlmostEqual(proj["gpu_h"], want, places=12)
        self.assertEqual(len(proj["rows"]), 2)
        self.assertTrue(self.model.in_range(inside))
        with self.assertRaises(ValueError):
            self.model.predict(outside, strict=True)
        with self.assertRaises(KeyError):
            self.model.predict(cm.par2_arm(2048, 256, 256, 1.0, code="par2_rtol1e-10"))

    def test_adding_calibration_points_refits(self):
        # synthetic pilot points of a new code generated from known coefficients are recovered exactly
        truth = 6.0e-7
        new = [cm.Point(cm.par2_arm(nx, ny, ny, s2, 1e4, "par2_rtol1e-10", "pilot%d" % nx), 1, truth * nx * ny * ny, "synthetic")
               for nx, ny, s2 in ((1040, 240, 0.25), (1664, 192, 4.0), (1248, 288, 2.25))]
        m2 = self.model.with_points(new)
        self.assertAlmostEqual(m2.coef["par2_rtol1e-10"]["cells"], truth, delta=1e-12 * truth + 1e-15)
        self.assertEqual(m2.coef["par2"], self.model.coef["par2"])  # other codes unchanged
        arm = cm.par2_arm(1040, 480, 480, 1.0, 1e4, "par2_rtol1e-10")
        self.assertAlmostEqual(m2.predict(arm), truth * arm.cells, places=6)
        with self.assertRaises(ValueError):
            cm.CostModel([cm.Point(cm.lagrangian_arm(100, 10, 10, 1.0), 1, 1.0)])  # fewer points than coefficients


class ArmTableAgainstThePreregistration(unittest.TestCase):
    def test_grids_and_passes_match_the_committed_tables(self):
        text = ""
        for name in ("preregistration.md", "preregistration-addendum-s025.md"):
            with open(os.path.join(A, name)) as fh:
                text += fh.read()
        lines = text.splitlines()
        checked = 0
        for aid, (nx, ny, nz, lh, eps, hmax) in cm.ARMS.items():
            if aid == "s4_w12c":  # the continuation arm is defined in the bitacora, not in the table
                continue
            tag = "`%s`" % aid
            rows = [ln for ln in lines if tag in ln and ln.lstrip().startswith("|")]
            self.assertTrue(rows, msg=aid)
            self.assertTrue(any("`%d,%d,%d`" % (nx, ny, nz) in ln for ln in rows), msg="grid of " + aid)
            self.assertTrue(any(("`%s`" % ",".join(str(d) for d in hmax)) in ln for ln in rows), msg="hmax of " + aid)
            checked += 1
        self.assertGreaterEqual(checked, 26)

    def test_par2_grids_match_the_committed_configs(self):
        for c, (nx, ny, nz, s2) in cm.PAR2_GRID.items():
            with open(os.path.join(A, "configs", "par2_%s.yaml" % c)) as fh:
                txt = fh.read()
            got = [int(re.search(r"^\s+%s:\s*(\d+)" % k, txt, re.M).group(1)) for k in ("nx", "ny", "nz")]
            self.assertEqual(got, [nx, ny, nz], msg=c)
            self.assertAlmostEqual(float(re.search(r"^\s+sigma2:\s*([0-9.]+)", txt, re.M).group(1)), s2)
            self.assertEqual(int(re.search(r"^\s+n_particles:\s*(\d+)", txt, re.M).group(1)), 10000)


if __name__ == "__main__":
    unittest.main()
