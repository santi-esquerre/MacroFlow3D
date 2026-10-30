import math
import unittest

import numpy as np
from scipy import stats

from mcstat import intervals as iv
from mcstat.tests._paths import HEAVY
from mcstat.tests._ref import t_ppf


class SlopeEstimators(unittest.TestCase):
    def test_ols_slope_exact_and_batched(self):
        d = np.linspace(25.0, 94.0, 15)
        y = 3.0 + 0.7 * d
        self.assertAlmostEqual(float(iv.ols_slope(d, y)), 0.7, places=13)
        rng = np.random.default_rng(3)
        Y = rng.normal(size=(6, 4, 15))
        S = iv.ols_slope(d, Y)
        self.assertEqual(S.shape, (6, 4))
        self.assertAlmostEqual(float(S[2, 3]), float(np.polyfit(d, Y[2, 3], 1)[0]), places=12)

    def test_alpha_helpers(self):
        self.assertEqual(float(iv.alpha_T_from_slope(0.2)), 0.1)
        self.assertEqual(float(iv.alpha_L_from_slope(2.0, 3.0)), 9.0)

    def test_alpha_LP_known_answer_both_speed_definitions(self):
        # crossing times t(d) = d / U and var t(d) = 2 alpha_L d / U^2 -> alpha_LP = alpha_L exactly
        d = np.linspace(25.0, 94.0, 11)
        for U in (0.4, 1.3):
            for aL in (0.05, 2.5):
                tmean, tvar = d / U, 2.0 * aL * d / U ** 2
                for method in ("ols", "endpoint"):
                    Ur = iv.mean_crossing_speed(d, tmean, method)
                    a = iv.alpha_L_from_slope(iv.ols_slope(d, tvar), Ur)
                    self.assertAlmostEqual(float(a) / aL, 1.0, places=12)
        with self.assertRaises(ValueError):
            iv.mean_crossing_speed(d, d, "bad")

    def test_alpha_TP_known_answer(self):
        d = np.linspace(25.0, 94.0, 11)
        aT = 0.03
        var_t = 1.7 + 2.0 * aT * d  # (var x2 + var x3) / 2 grows as 2 alpha_T d
        self.assertAlmostEqual(float(iv.alpha_T_from_slope(iv.ols_slope(d, var_t))), aT, places=13)

    def test_orthogonal_variance(self):
        # drift along x2: the orthogonal variance is var x3
        self.assertAlmostEqual(float(iv.orthogonal_variance(2.0, 5.0, 0.7, 1.0, 0.0)), 5.0, places=14)
        # drift at 45 degrees, var2 = var3 = 1, cov = 0.5: n = (-1, 1)/sqrt(2), n S n = 1 - 0.5
        self.assertAlmostEqual(float(iv.orthogonal_variance(1.0, 1.0, 0.5, 1.0, 1.0)), 0.5, places=14)
        self.assertTrue(np.isnan(iv.orthogonal_variance(1.0, 1.0, 0.5, 0.0, 0.0)))
        # orthogonal + parallel variance = trace
        v = iv.orthogonal_variance(2.0, 3.0, 0.4, 0.6, -0.8)
        par = iv.orthogonal_variance(2.0, 3.0, 0.4, 0.8, 0.6)  # drift rotated by 90 degrees gives the parallel part
        self.assertAlmostEqual(float(v + par), 5.0, places=12)


class StudentAndWelch(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.x = rng.normal(0.3, 1.2, size=17)
        self.y = rng.normal(0.0, 2.0, size=23)

    def test_student_matches_scipy(self):
        r = iv.student_interval(self.x, 0.90)
        half = t_ppf(0.95, len(self.x) - 1) * stats.sem(self.x)
        self.assertAlmostEqual(float(r.lo), self.x.mean() - half, places=13)
        self.assertAlmostEqual(float(r.hi), self.x.mean() + half, places=13)
        lo, hi = stats.t.interval(0.90, len(self.x) - 1, loc=self.x.mean(), scale=stats.sem(self.x))  # scipy's own, to 1e-8
        self.assertAlmostEqual(float(r.lo), lo, places=8)

    def test_one_sided_limits_are_the_two_sided_tails(self):
        two = iv.student_interval(self.x, 0.90)
        self.assertAlmostEqual(float(iv.student_limit(self.x, 0.05, "lower")), float(two.lo), places=13)
        self.assertAlmostEqual(float(iv.student_limit(self.x, 0.05, "upper")), float(two.hi), places=13)
        one = iv.student_interval(self.x, 0.95, "lower")
        self.assertAlmostEqual(float(one.lo), float(two.lo), places=13)
        self.assertTrue(np.isinf(one.hi))
        lims = iv.student_limit(self.x, [1e-4, 1e-3, 1e-2, 0.1], "lower")
        self.assertEqual(lims.shape, (4,))
        self.assertTrue(np.all(np.diff(lims) > 0))  # smaller tail = lower (more cautious) limit

    def test_welch_against_hand_formula_and_scipy_p_value(self):
        a, b = self.x, self.y
        w = iv.welch(a, b, 0.90)
        va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
        nu = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
        half = t_ppf(0.95, nu) * math.sqrt(va + vb)
        self.assertAlmostEqual(float(w.nu), nu, places=10)
        self.assertAlmostEqual(float(w.lo), a.mean() - b.mean() - half, places=12)
        self.assertAlmostEqual(float(w.hi), a.mean() - b.mean() + half, places=12)
        p_scipy = stats.ttest_ind(a, b, equal_var=False).pvalue
        t = float(w.diff / w.se)
        self.assertAlmostEqual(2 * stats.t.sf(abs(t), float(w.nu)), p_scipy, places=12)
        # summary-statistic entry point agrees with the array one
        ws = iv.welch_from_summary(a.mean(), a.std(ddof=1), a.size, b.mean(), b.std(ddof=1), b.size, 0.90)
        self.assertAlmostEqual(float(ws.lo), float(w.lo), places=13)

    def test_welch_one_sided_limit_and_contrast(self):
        lim = iv.welch_limit(self.x, self.y, 0.05, "lower")
        self.assertAlmostEqual(float(lim), float(iv.welch(self.x, self.y, 0.90).lo), places=12)
        c = iv.welch(self.x, self.y, 0.90, c1=1.0, c2=-0.9)
        self.assertAlmostEqual(float(c.diff), self.x.mean() - 0.9 * self.y.mean(), places=13)

    def test_paired_equals_one_sample_on_differences(self):
        a = self.x[:15]
        b = self.y[:15]
        r = iv.paired_interval(a, b, 0.90)
        s = iv.student_interval(a - b, 0.90)
        self.assertEqual((float(r.lo), float(r.hi)), (float(s.lo), float(s.hi)))
        with self.assertRaises(ValueError):
            iv.paired_interval(a, b[:-1])

    def test_margin_multiple(self):
        self.assertAlmostEqual(float(iv.margin_multiple(-0.2, 0.5, 0.25)), 2.0, places=14)


class BootstrapAndBCa(unittest.TestCase):
    def test_bootstrap_t_reproducible_and_close_to_student_for_normal(self):
        x = np.random.default_rng(5).normal(1.0, 1.0, size=60)
        a = iv.bootstrap_t_interval(x, 0.90, seed=7, B=3999)
        b = iv.bootstrap_t_interval(x, 0.90, seed=7, B=3999)
        self.assertEqual((float(a.lo), float(a.hi)), (float(b.lo), float(b.hi)))
        s = iv.student_interval(x, 0.90)
        self.assertLess(abs(float(a.lo) - float(s.lo)), 0.05)
        self.assertLess(abs(float(a.hi) - float(s.hi)), 0.05)

    def test_bootstrap_work_requires_an_explicit_stream(self):
        x = np.arange(10.0)
        with self.assertRaises(ValueError):
            iv.bootstrap_t(x, B=99)
        with self.assertRaises(ValueError):
            iv.bootstrap_t_interval(x, B=99)

    def test_bootstrap_t_unresolvable_tail_is_trivial_never_optimistic(self):
        x = np.random.default_rng(5).normal(size=30)
        bt = iv.bootstrap_t(x, B=99, seed=1)
        self.assertEqual(float(iv.bootstrap_t_limit(bt, 1e-4, "lower")), -np.inf)
        self.assertEqual(float(iv.bootstrap_t_limit(bt, 1e-4, "upper")), np.inf)
        self.assertTrue(np.isfinite(float(iv.bootstrap_t_limit(bt, 0.05, "lower"))))

    def test_bootstrap_t_batch_shapes(self):
        x = np.random.default_rng(2).normal(size=(5, 4, 20))
        bt = iv.bootstrap_t(x, B=199, seed=3, share_draws=True)
        lim = iv.bootstrap_t_limit(bt, [0.05, 0.1], "lower")
        self.assertEqual(lim.shape, (5, 4, 2))
        bt2 = iv.bootstrap_t(x[:2], B=199, seed=3)  # independent draws per replicate row
        self.assertEqual(iv.bootstrap_t_limit(bt2, 0.05, "lower").shape, (2, 4))

    def test_bca_for_reporting(self):
        x = np.random.default_rng(9).lognormal(0, 0.6, size=40)
        r = iv.bca_interval(x, 0.90, seed=1, B=4000)
        self.assertFalse(r.degenerate)
        self.assertLess(r.lo, x.mean())
        self.assertGreater(r.hi, x.mean())
        again = iv.bca_interval(x, 0.90, seed=1, B=4000)
        self.assertEqual((r.lo, r.hi), (again.lo, again.hi))
        self.assertTrue(iv.bca_interval(np.ones(10)).degenerate)
        one = iv.bca_interval(x, 0.95, "lower", seed=1, B=4000)
        self.assertTrue(np.isinf(one.hi))


class Coverage(unittest.TestCase):
    """Monte Carlo coverage with fixed seeds: deterministic, tolerance = 4 Monte Carlo standard errors."""

    @staticmethod
    def se(p, R):
        return math.sqrt(p * (1 - p) / R)

    def test_student_two_sided_and_per_tail_on_normal(self):
        R, n = (200000 if HEAVY else 40000), 20
        x = np.random.default_rng(101).standard_normal((R, n))
        two = iv.student_interval(x, 0.90)
        miss = np.mean((two.lo > 0) | (two.hi < 0))
        self.assertLess(abs(miss - 0.10), 4 * self.se(0.10, R))
        for tail in (0.05, 0.01, 0.001):
            lo_miss = np.mean(iv.student_limit(x, tail, "lower") > 0)
            up_miss = np.mean(iv.student_limit(x, tail, "upper") < 0)
            for m in (lo_miss, up_miss):
                self.assertLess(abs(m - tail), 4 * self.se(tail, R), msg="tail %g" % tail)

    def test_skewed_samples_student_fails_where_bootstrap_t_holds(self):
        # exponential (mean 1), n = 100: the Student upper limit misses ~1.5x its nominal 5 % (right skew); the bootstrap-t
        # limits stay within the second-order allowance.  This is the discordance the SF-42 coverage simulation guards against.
        R, n, B = (6000 if HEAVY else 1500), 100, 399
        rng = np.random.default_rng(202)
        x = rng.exponential(1.0, size=(R, n))
        st_up = np.mean(iv.student_limit(x, 0.05, "upper") < 1.0)
        self.assertGreater(st_up, 1.25 * 0.05)  # fails the G5.3 criterion (1.25 x nominal)
        bt = iv.bootstrap_t(x, B=B, rng=rng)
        tol = 4 * self.se(0.05, R) + 0.012
        for side, miss in (("lower", np.mean(iv.bootstrap_t_limit(bt, 0.05, "lower") > 1.0)),
                           ("upper", np.mean(iv.bootstrap_t_limit(bt, 0.05, "upper") < 1.0))):
            self.assertLess(abs(miss - 0.05), tol, msg=side)

    def test_bootstrap_t_normal_per_tail(self):
        R, n, B = (4000 if HEAVY else 1500), 30, 999
        rng = np.random.default_rng(303)
        x = rng.standard_normal((R, n))
        bt = iv.bootstrap_t(x, B=B, rng=rng)
        for tail in (0.05, 0.01):
            lo_m = np.mean(iv.bootstrap_t_limit(bt, tail, "lower") > 0)
            up_m = np.mean(iv.bootstrap_t_limit(bt, tail, "upper") < 0)
            for m in (lo_m, up_m):
                self.assertLess(abs(m - tail), 4 * self.se(tail, R) + 0.004, msg="tail %g" % tail)


if __name__ == "__main__":
    unittest.main()
