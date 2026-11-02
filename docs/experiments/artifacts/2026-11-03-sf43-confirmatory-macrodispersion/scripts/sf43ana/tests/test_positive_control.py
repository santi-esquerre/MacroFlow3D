import csv
import json
import os
import tempfile
import unittest

import numpy as np

import sf43ana  # noqa: F401
from sf42red import reduce as red
from sf43ana import config, positive_control as pcm, summary
from sf43ana.tests import fixtures as fx

L1, W, SIGMA2, ELL = 40.0, 12.0, 0.04, 1.0 / 12.0
WINDOW = (5.0, 30.0)


class FirstOrderBox(unittest.TestCase):
    def test_reproduces_the_review_values(self):
        d = np.arange(25, 94.5, 1.0)
        self.assertAlmostEqual(0.5 * pcm._ols_slope(d, pcm.x22_box(d, 104.0, 24.0)), -3.4954540627e-5, delta=2e-10)  # firstorder_box.py: -3.50e-5
        self.assertAlmostEqual(0.5 * pcm._ols_slope(d, pcm.x22_box(d, 104.0, 12.0)), -4.11947e-6, delta=2e-10)
        x = pcm.x22_box([30.0, 74.0], 104.0, 24.0)  # periodic along x: symmetric about L1 / 2
        self.assertAlmostEqual(x[0], x[1], places=12)

    def test_short_distance_limit_scales_with_distance_squared(self):
        x = pcm.x22_box([0.05, 0.1], L1, W)  # X22(d) ~ <u2^2> d^2 for small d
        self.assertAlmostEqual(x[1] / x[0], 4.0, delta=0.05)


def write_planes(root, src, seg, seed0, nseg, factor, rng, centred_share=0.0):
    """<table>.planes.csv for realizations r = 0..nseg-1 of one segment: displacement-basis moments around ``factor`` x prediction."""
    d = np.arange(1.0, 36.0)
    pred = SIGMA2 * pcm.x22_box(d, L1, W)
    path = fx.table_path(root, src, seg, 25, 94)[:-4] + ".planes.csv"
    rows = []
    for r in range(nseg):
        z = 1.0 + 0.03 * rng.standard_normal(d.size)
        zr = rng.standard_normal()
        for w in ("flux", "unif"):
            for basis in ("pos", "disp"):
                for k, dk in enumerate(d):
                    m = np.sqrt(centred_share * factor * pred[k]) * ELL * zr
                    v = factor * pred[k] * ELL ** 2 * z[k] * (1.0 - centred_share)
                    rows.append({"code": "consumer", "source": "x", "r": r, "hmax_div": -1, "weights": w, "basis": basis, "plane": k, "x1": dk * ELL,
                                 "d": dk, "in_window": int(WINDOW[0] <= dk <= WINDOW[1]), "n": 100, "mean_tau": 1.0, "var_tau": 1.0,
                                 "mean_x2": m, "mean_x3": m, "var_x2": v, "var_x3": v, "cov_x2x3": 0.0})
    red.write_table(path, rows, red.PLANE_COLUMNS)


class PositiveControl(unittest.TestCase):
    def build(self, factor, n=24, centred_share=0.0):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        root = self._t.name
        pr = fx.prereg_dict()
        pr["arms"] = [fx.arm("pc", "0.04", "gpc", range(500, 500 + n), n, 5)]
        pr["amplitudes"] = {"0.04": {"production": "pc", "components": []}}
        pr["family"] = ["gate5:0.04"]
        for k in ("targets", "first_order"):
            del pr[k]
        pr["floors"] = {"production": 5, "ladder": 5, "width": 5, "hmax": 5, "gate5": 5}
        pr["positive_control"] = {"arm": "pc", "sigma2": SIGMA2, "width_lambda": W, "length_lambda": L1, "window": list(WINDOW), "B": 400, "seed": 5}
        self.paths = fx.write_world(root, pr, lambda a, w, b, e: 0.01, segments=2, ell=ELL)
        rng = np.random.default_rng(1)
        for seg in sorted(os.listdir(os.path.join(root, "reduced", "pc"))):
            s0 = int(seg.split("_s")[1])
            nseg = n // 2
            write_planes(root, "pc", seg, s0, nseg, factor, rng, centred_share)
        self.prereg = config.load_prereg(self.paths["prereg"])
        return root

    def run_pc(self, root):
        out = os.path.join(root, "out")
        S, _ = summary.run(self.paths["prereg"], self.paths["campaign"], self.paths["reduced"], "none", out)
        return S["positive_control"]

    def test_agreement_gives_ratios_near_one_with_intervals_containing_one(self):
        r = self.run_pc(self.build(1.0))
        self.assertEqual(r["N"], 24)
        self.assertAlmostEqual(r["ell"], ELL)
        for w in ("flux", "unif"):
            sm = r["weights"][w]["second_moment"]
            self.assertTrue(np.allclose(sm["ratio"], 1.0, atol=0.05))
            self.assertGreaterEqual(sm["planes_ratio_interval_contains_1_in_window"], 0.8 * sm["planes_in_window"])
            ws = sm["window_slope"]
            self.assertAlmostEqual(ws["ratio"], 1.0, delta=0.1)
            self.assertTrue(ws["difference_interval_contains_0"])
            self.assertLess(ws["ratio_interval"][0], 1.0 < ws["ratio_interval"][1])

    def test_a_factor_two_error_is_visible_and_centring_separates_the_two_profiles(self):
        r = self.run_pc(self.build(2.0))
        ws = r["weights"]["flux"]["second_moment"]["window_slope"]
        self.assertAlmostEqual(ws["ratio"], 2.0, delta=0.5)  # the window slope of the periodic box is small: a noisy ratio
        self.assertFalse(ws["difference_interval_contains_0"])
        self.assertAlmostEqual(r["weights"]["unif"]["second_moment"]["ratio"][10], 2.0, delta=0.15)
        r = self.run_pc(self.build(1.0, centred_share=0.3))  # 30 % of the second moment in the realization mean: centred profile is 0.7 of it
        sm, ce = r["weights"]["flux"]["second_moment"], r["weights"]["flux"]["centred"]
        self.assertAlmostEqual(np.mean(sm["ratio"]), 1.0, delta=0.25)  # noisy in the mean (24 realizations), not biased
        self.assertAlmostEqual(np.mean(np.array(ce["ratio"])) / np.mean(np.array(sm["ratio"])), 0.7, delta=0.08)

    def test_alpha_equivalent_is_half_the_slope(self):
        r = self.run_pc(self.build(1.0))
        ws = r["weights"]["flux"]["centred"]["window_slope"]
        self.assertAlmostEqual(ws["alpha_T_equivalent_predicted"], 0.5 * ws["predicted"], places=15)


if __name__ == "__main__":
    unittest.main()
