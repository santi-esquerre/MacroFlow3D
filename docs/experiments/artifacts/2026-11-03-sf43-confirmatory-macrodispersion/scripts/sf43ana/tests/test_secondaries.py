import math
import unittest

import numpy as np
from scipy import integrate

from sf43ana import secondaries as sec


class Eq6(unittest.TestCase):
    def test_formula_and_table(self):
        a, tl, p = 4.35, 4.83, 0.63
        self.assertAlmostEqual(float(sec.bdd_f(7.0, a, tl, p)), a * (1 - math.exp(-7.0 / tl)) ** p, places=14)
        self.assertEqual(sec.BDD_TABLE["s4"], (12.7, 20.5, 0.49))
        self.assertEqual([sec.BDD_TABLE[k][0] for k in ("s025", "s1", "s225", "s4")], [0.24, 1.27, 4.35, 12.7])

    def test_window_average_against_quadrature(self):
        a, tl, p = sec.BDD_TABLE["s225"]
        t_end = 37.0
        want = integrate.quad(lambda t: a * (1 - math.exp(-t / tl)) ** p, 0.75 * t_end, t_end)[0] / (0.25 * t_end)
        self.assertAlmostEqual(float(sec.bdd_window_average("s225", t_end)), want, places=7)

    def test_fit_recovers_known_parameters(self):
        for truth in ((4.35, 4.83, 0.63), (12.7, 20.5, 0.49), (1.27, 2.21, 0.80)):
            t = np.geomspace(0.3, 80.0, 25)
            p, rmse = sec.fit_bdd(t, sec.bdd_f(t, *truth))
            self.assertTrue(np.allclose(p, truth, rtol=1e-4), (truth, p))
            self.assertLess(rmse, 1e-6)

    def test_noisy_fit_and_bootstrap_interval(self):
        truth = (4.35, 4.83, 0.63)
        rng = np.random.default_rng(7)
        n, K = 60, 6
        T = np.geomspace(0.5, 60.0, K)[None, :] * np.exp(0.1 * rng.standard_normal((n, K)))
        Y = sec.bdd_f(T, *truth) * (1 + 0.08 * rng.standard_normal((n, K)))
        seeds = np.arange(n)
        data = {"arms": {"A": {"seed": seeds, "cols": {"curve_t": T, "curve_alpha": Y}}}}
        r = sec.run({"secondaries": {"B": 150, "bdd_fit": [{"name": "A", "arm": "A", "amplitude": "s225"}]}}, data)["bdd_fit"]["A"]
        names = ("a_LAF", "t_LF", "a_LF")
        for nm, tv in zip(names, truth):
            self.assertAlmostEqual(r["params"][nm], tv, delta=0.15 * tv, msg=nm)
            b = r["bootstrap"][nm]  # a sane interval (one noise realization: coverage of the truth is not asserted)
            self.assertLess(b["lo"], r["params"][nm])
            self.assertGreater(b["hi"], r["params"][nm])
            self.assertLess(abs(r["params"][nm] - tv), 4 * (b["hi"] - b["lo"]))
        self.assertEqual(r["bootstrap"]["failed"], 0)
        # the gap to Table 1 of the amplitude is recorded as difference and ratio; at the truth of s225 it is small
        self.assertAlmostEqual(r["gap"]["a_LAF"]["difference"], r["params"]["a_LAF"] - 4.35, places=12)
        self.assertAlmostEqual(r["gap"]["t_LF"]["ratio"], r["params"]["t_LF"] / 4.83, places=12)

    def test_reproducible_and_few_points_refused(self):
        T = np.geomspace(0.5, 30.0, 10)[None, :].repeat(20, 0)
        Y = sec.bdd_f(T, 2.0, 3.0, 0.7) + 0.01 * np.random.default_rng(1).standard_normal(T.shape)
        data = {"arms": {"A": {"seed": np.arange(20), "cols": {"curve_t": T, "curve_alpha": Y}}}}
        pre = {"secondaries": {"B": 20, "bdd_fit": [{"arm": "A"}]}}
        self.assertEqual(sec.run(pre, data), sec.run(pre, data))
        self.assertIsNone(sec.fit_bdd([1.0, 2.0], [1.0, 1.5])[0])


class Blocks(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(2)
        n = 50
        seeds = np.arange(n)
        base = 1.0 + 0.2 * rng.standard_normal(n)
        te = 30.0 + 5 * rng.standard_normal(n)
        pred = sec.bdd_window_average("s1", te)
        c = {"alpha_LP": base, "alpha_LP:expI": 1.07 * base * (1 + 1e-3 * rng.standard_normal(n)),
             "alpha_TP": 0.01 * base, "alpha_TP:expI": 0.012 * base, "alpha_TP:disp": 0.0101 * base,
             "alpha_LAM:time": 0.9 * pred, "t_end_adv": te}
        par2 = {"alpha_TP": 0.011 * base, "alpha_TP:disp": 0.0102 * base}
        return {"arms": {"cons": {"seed": seeds, "cols": c}, "par2": {"seed": seeds[::-1].copy(), "cols": {k: v[::-1].copy() for k, v in par2.items()}}}}

    def test_expI_time_window_and_par2(self):
        pre = {"secondaries": {"expI": {"s1": {"arm": "cons"}}, "time_window": {"cons": {"amplitude": "s1"}},
                               "par2_transverse": {"s1": {"par2": "par2", "consumer": "cons"}}}}
        r = sec.run(pre, self.data())
        e = r["expI"]["s1"]
        self.assertAlmostEqual(e["alpha_LP"]["ratio"], 1.07, places=2)
        self.assertAlmostEqual(e["alpha_TP"]["ratio"], 1.2, places=10)
        tw = r["time_window"]["cons"]
        self.assertAlmostEqual(tw["bdd_window"]["ratio_of_means"], 0.9, places=10)
        self.assertEqual(tw["alpha_LAM:time"]["n"], 50)
        p = r["par2_transverse"]["s1"]
        self.assertAlmostEqual(p["position"]["par2_over_consumer"]["ratio"], 1.1, places=10)
        self.assertAlmostEqual(p["displacement"]["par2_over_consumer"]["ratio"], 0.0102 / 0.0101, places=10)
        self.assertLess(p["position"]["par2"]["lo"], p["position"]["par2"]["mean"])

    def test_absent_columns_are_noted(self):
        d = self.data()
        del d["arms"]["cons"]["cols"]["t_end_adv"]
        r = sec.run({"secondaries": {"time_window": {"cons": {"amplitude": "s1"}}}}, d)
        self.assertIn("t_end_adv", r["time_window"]["cons"]["bdd_window"]["note"])



class Recrossings(unittest.TestCase):
    def data(self, with_cols):
        rng = np.random.default_rng(2)
        n = 40
        lp = 1.0 + 0.1 * rng.standard_normal(n)
        cols = {"alpha_LP": lp}
        if with_cols:
            cols.update({"frac_recross": 0.02 + 0.005 * rng.standard_normal(n), "frac_backflow": 0.004 + 0.001 * rng.standard_normal(n),
                         "alpha_LP:nobackflow": lp * 0.95})
        return {"arms": {"a": {"seed": np.arange(n), "cols": cols}}}

    def test_fractions_and_the_sensitivity_without_backflow_encounters(self):
        r = sec.run({"secondaries": {"recrossings": {"s4": {"arm": "a"}}}}, self.data(True))["recrossings"]["s4"]
        self.assertAlmostEqual(r["frac_recross"]["mean"], 0.02, delta=0.005)
        self.assertAlmostEqual(r["frac_backflow"]["mean"], 0.004, delta=0.001)
        self.assertAlmostEqual(r["alpha_LP_without_backflow_over_all"]["ratio"], 0.95, places=10)
        self.assertNotIn("note", r)

    def test_without_the_input_the_block_says_what_is_needed(self):
        r = sec.run({"secondaries": {"recrossings": {"s4": {"arm": "a"}}}}, self.data(False))["recrossings"]["s4"]
        self.assertIn("input absent", r["note"])
        for c in ("frac_recross", "frac_backflow", "alpha_LP:nobackflow"):
            self.assertIn(c, r["note"])


if __name__ == "__main__":
    unittest.main()
