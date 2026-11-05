import copy
import unittest

import numpy as np
from scipy import stats

import sf43ana  # noqa: F401
from sf43ana import config
from sf43ana.gate4 import (CLASS_BOUNDED, CLASS_NUM, CLASS_PHYS, CLASS_UNRES, Design, RobustTest, classify_amplitude,
                           classify_estimator)

CFG = copy.deepcopy(config.DEFAULTS)
CFG["bootstrap"].update(B_sampling=400, B_budget=2000, B_ucl=500, B_bca=100)


def std(n, seed):
    z = np.random.default_rng(seed).normal(size=n)
    return (z - z.mean()) / z.std(ddof=1)


def tq(tail, dof):
    return stats.t.isf(tail, dof)


def ladder_values(f, sd, n=40, seed=3):
    """u_s + f_k with u standardized: every pair difference is constant (zero standard error), the level means are mean(u) + f_k = f_k."""
    return sd * std(n, seed)[:, None] + np.asarray(f, float)[None, :]


class RobustLimitByHand(unittest.TestCase):
    def setUp(self):
        n = 40
        self.mu, self.s = 0.0100, 0.0020
        self.prod = self.mu + self.s * std(n, 1)
        self.f = [0.004, 0.002, 0.001]  # errors at 5/10/20: order 1, production in the middle
        self.levels = ladder_values(self.f, 0.001)
        self.wide = 0.0105 + 0.003 * std(30, 4)
        self.pfine = 0.0100 + 0.002 * std(40, 5)
        d = 0.0002 + 0.0004 * std(20, 6)
        self.h8 = 0.003 + 0.002 * std(20, 7)
        self.h16 = self.h8 + d
        self.d = Design(self.prod, ["E_h", "E_W", "E_int"], levels=self.levels, res=[5, 10, 20], ip=1, wide_fine=self.wide,
                        prod_fine=self.pfine, h16=self.h16, h8=self.h8)

    def hand(self, a):
        K, t = 3, a / 6.0
        n = self.prod.size
        L = self.mu - tq(a / 2.0, n - 1) * self.s / np.sqrt(n)
        Bh = 1.25 * 0.001 * 2.0 / (2.0 - 1.0)  # finer branch, p = 1 (order exactly 1), r = 2: 1.25 |m3 - m2| r^p/(r^p - 1)
        v1, v2 = 0.003 ** 2 / 30, 0.002 ** 2 / 40
        nu = (v1 + v2) ** 2 / (v1 ** 2 / 29 + v2 ** 2 / 39)
        BW = abs(0.0005) + tq(t, nu) * np.sqrt(v1 + v2)
        Bi = 2.0 * (0.0002 + tq(t, 19) * 0.0004 / np.sqrt(20))
        return L, Bh, BW, Bi, L - Bh - BW - Bi

    def test_rl_equals_hand_value(self):
        t = RobustTest(self.d, 0.0, CFG, "hand", sampling="student")
        for a in (0.05, 0.01, 0.0125):
            L, Bh, BW, Bi, RL = self.hand(a)
            rep = t.report([a])
            self.assertAlmostEqual(float(rep["L_a_over_2"][0]), L, places=12)
            self.assertAlmostEqual(float(rep["components"]["E_h"]["enforced"][0]), Bh, places=8)
            self.assertAlmostEqual(float(rep["components"]["E_W"]["enforced"][0]), BW, places=10)
            self.assertAlmostEqual(float(rep["components"]["E_int"]["enforced"][0]), Bi, places=10)
            self.assertAlmostEqual(float(t.rl_at(a)), RL, places=8)

    def test_shifted_null_and_exact_decision(self):
        p0 = RobustTest(self.d, 0.0, CFG, "hand", sampling="student")
        pshift = RobustTest(self.d, 0.0030, CFG, "hand", sampling="student")
        self.assertLess(p0.p_value()[0], 1.0)
        self.assertGreater(float(pshift.p_value()[0]), float(p0.p_value()[0]))  # a shifted null is harder to reject
        # exact decision at a level equals the sign of RL(level) - null, on both sides of the p-value
        p = float(p0.p_value()[0])
        self.assertTrue(p0.reject(p) and p0.reject(min(1.0, p * 1.5)))
        self.assertFalse(p0.reject(p / 1.2))
        # the shift is on RL, so RL(a) > null  <=>  RL(a) - null > 0
        a = 0.05
        self.assertEqual(RobustTest(self.d, 0.0030, CFG, "hand", sampling="student").reject(a), bool(self.hand(a)[4] > 0.0030))

    def test_never_positive_gives_p_one(self):
        t = RobustTest(self.d, 10.0, CFG, "hand", sampling="student")
        self.assertEqual(float(t.p_value()[0]), 1.0)
        self.assertTrue(bool(t.p_value()[1]))

    def test_upper_bound_of_the_bounded_class(self):
        t = RobustTest(self.d, 0.0, CFG, "hand", sampling="student")
        n = self.prod.size
        U = self.mu + tq(0.05, n - 1) * self.s / np.sqrt(n)
        _, Bh, BW, Bi, _ = self.hand(0.1)  # B_k(0.05 / K) = B_k(a / (2K)) at a = 0.1
        self.assertAlmostEqual(float(t.upper_bound(0.05)), U + Bh + BW + Bi, places=8)

    def test_bootstrap_t_sampling_runs_and_is_deterministic(self):
        a = RobustTest(self.d, 0.0, CFG, "same-label")
        b = RobustTest(self.d, 0.0, CFG, "same-label")
        self.assertEqual(float(a.p_value()[0]), float(b.p_value()[0]))
        self.assertEqual(float(a.rl_at(0.05)), float(b.rl_at(0.05)))

    def test_untestable_design(self):
        d = Design(self.prod, ["E_h", "E_W"], levels=self.levels, res=[5, 10, 20], ip=1, reasons=["arm w below its floor (N=3 < 20)"])
        t = RobustTest(d, 0.0, CFG, "u")
        self.assertFalse(t.testable)
        self.assertEqual(float(t.p_value()[0]), 1.0)
        self.assertFalse(t.reject(0.05))
        d2 = Design(self.prod, ["E_h", "E_W"], levels=self.levels, res=[5, 10, 20], ip=1)  # E_W configured but no width data
        self.assertEqual(RobustTest(d2, 0.0, CFG, "u").missing, ["width"])


class Classes(unittest.TestCase):
    def numerical_design(self, mu_prod=0.010, e_prod=0.008):
        """Production biased upward by e_prod (order-1 ladder 5/10/20); the true value is mu_prod - e_prod."""
        star = mu_prod - e_prod
        f = star + e_prod * np.array([2.0, 1.0, 0.5])  # errors halve with each doubling of lambda/h: order 1
        return Design(mu_prod + 0.0005 * std(40, 1), ["E_h"], levels=ladder_values(f, 0.0005), res=[5, 10, 20], ip=1)

    def test_numerical_class_inputs(self):
        d = self.numerical_design()
        t = RobustTest(d, 7.992e-5, CFG, "num", sampling="student")
        self.assertFalse(t.reject(0.05))  # E_h = 1.25 * 0.004 * 2 = 0.01 swallows the margin
        c = classify_estimator(t, CFG, "num")
        self.assertTrue(c["numerical"])
        self.assertAlmostEqual(c["corrected"]["mean_prod"], 0.010, places=12)
        self.assertAlmostEqual(c["corrected"]["richardson"], -0.008, places=9)
        self.assertAlmostEqual(c["corrected"]["estimate"], 0.002, places=9)
        self.assertLess(c["corrected"]["ucl"], 0.5 * 0.010)
        self.assertTrue(c["bounded"] and np.isfinite(c["bound"]))

    def test_not_numerical_when_correction_is_small_or_interval_covers_null(self):
        small = self.numerical_design(0.010, 0.0005)  # corrected ~ 0.0095 > half of the mean
        self.assertFalse(classify_estimator(RobustTest(small, 7.992e-5, CFG, "n2", sampling="student"), CFG, "n2")["numerical"])
        covers = self.numerical_design(0.00005, 0.00004)  # production interval contains the null 7.992e-5
        c = classify_estimator(RobustTest(covers, 7.992e-5, CFG, "n3", sampling="student"), CFG, "n3")
        self.assertFalse(c["numerical"])

    def test_width_correction_enters_the_corrected_estimate(self):
        d = self.numerical_design()
        d.components = ["E_h", "E_W"]
        d.wide_fine = 0.0070 + 0.0005 * std(30, 8)   # the wide box is lower by 0.003 ...
        d.prod_fine = 0.0100 + 0.0005 * std(40, 9)
        d.prod_fine_paired = False
        c = classify_estimator(RobustTest(d, 7.992e-5, CFG, "w", sampling="student"), CFG, "w")
        self.assertAlmostEqual(c["corrected"]["width"], -0.003, places=9)
        self.assertAlmostEqual(c["corrected"]["estimate"], 0.010 - 0.008 - 0.003, places=9)

    def test_precedence_and_discordance(self):
        num = {"numerical": True, "bounded": True}
        bnd = {"numerical": False, "bounded": True}
        none = {"numerical": False, "bounded": False}
        self.assertEqual(classify_amplitude(True, {"alpha_TP": none, "alpha_TAP": none})[0], CLASS_PHYS)  # Holm first
        self.assertEqual(classify_amplitude(False, {"alpha_TP": num, "alpha_TAP": num})[0], CLASS_NUM)  # numerical before bounded
        self.assertEqual(classify_amplitude(False, {"alpha_TP": bnd, "alpha_TAP": bnd})[0], CLASS_BOUNDED)
        self.assertEqual(classify_amplitude(False, {"alpha_TP": none, "alpha_TAP": none})[0], CLASS_UNRES)
        cls, note = classify_amplitude(False, {"alpha_TP": num, "alpha_TAP": bnd})  # estimators disagree
        self.assertEqual(cls, CLASS_UNRES)
        self.assertIn("discordance", note)
        self.assertEqual(classify_amplitude(False, {"alpha_TP": bnd, "alpha_TAP": none})[0], CLASS_UNRES)

    def test_untestable_estimator_is_unresolved(self):
        t = RobustTest(Design(None, ["E_h"], reasons=["arm p below its floor"]), 0.0, CFG, "x")
        c = classify_estimator(t, CFG, "x")
        self.assertFalse(c["numerical"] or c["bounded"])
        self.assertEqual(classify_amplitude(False, {"alpha_TP": c, "alpha_TAP": c})[0], CLASS_UNRES)


if __name__ == "__main__":
    unittest.main()
