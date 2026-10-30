import math
import unittest

import numpy as np
from scipy import stats

from mcstat import gate5 as g5
from mcstat.tests._paths import HEAVY
from mcstat.tests._ref import t_ppf


def bivariate(rng, R, n, ratio, cv_L, cv_P, corr):
    """R replicates of n paired values: mean_P = 1, mean_L = ratio, given CVs and correlation."""
    z = rng.standard_normal((R, n, 2))
    zl = z[..., 0]
    zp = corr * z[..., 0] + math.sqrt(1 - corr ** 2) * z[..., 1]
    return ratio * (1 + cv_L * zl), 1 + cv_P * zp


class Gate5Estimate(unittest.TestCase):
    def test_hand_computed_example(self):
        L = np.array([1.0, 1.2, 0.8, 1.1, 0.9])
        P = np.array([1.0, 1.0, 1.1, 1.0, 1.0])
        r = g5.gate5_paired(L, P, 0.90)
        mL, mP = sum(L) / 5, sum(P) / 5
        e = [l / mL - p / mP for l, p in zip(L, P)]
        sd = math.sqrt(sum(v * v for v in e) / 4)  # the mean of e is exactly 0
        se = sd / math.sqrt(5)
        theta = math.log(mL / mP)
        half = t_ppf(0.95, 4) * se
        self.assertAlmostEqual(float(r.theta), theta, places=14)
        self.assertAlmostEqual(float(r.se), se, places=14)
        self.assertAlmostEqual(float(r.lo), theta - half, places=13)
        self.assertAlmostEqual(float(r.hi), theta + half, places=13)
        self.assertEqual(r.dof, 4)
        self.assertAlmostEqual(abs(sum(e)), 0.0, places=14)
        self.assertAlmostEqual(float(r.p_tost), max(stats.t.sf((theta + math.log(1.1)) / se, 4),
                                                    stats.t.sf((math.log(1.1) - theta) / se, 4)), places=14)

    def test_verdicts(self):
        rng = np.random.default_rng(1)
        n = 60
        base = rng.normal(0, 1, n)
        P = 5.0 + 0.5 * base
        tiny = 0.001 * rng.standard_normal(n)
        self.assertEqual(g5.gate5_paired(P * 1.00 * (1 + tiny), P)["verdict"], "agree")
        self.assertEqual(g5.gate5_paired(P * 1.30 * (1 + tiny), P)["verdict"], "disagree")
        self.assertEqual(g5.gate5_paired(P * 0.70 * (1 + tiny), P)["verdict"], "disagree")
        L, P2 = bivariate(rng, 1, 30, 1.08, 0.25, 0.25, 0.0)  # theta = 0.077, se ~ 0.065: CI straddles the margin
        self.assertEqual(g5.gate5_paired(L[0], P2[0])["verdict"], "inconclusive")

    def test_tost_p_is_consistent_with_the_ci_and_batches(self):
        rng = np.random.default_rng(7)
        L, P = bivariate(rng, 400, 25, 1.02, 0.2, 0.2, 0.3)
        r = g5.gate5_paired(L, P, 0.90)
        self.assertEqual(r.theta.shape, (400,))
        delta = math.log(1.1)
        inside = (r.lo > -delta) & (r.hi < delta)
        self.assertTrue(np.array_equal(r.p_tost <= 0.05, inside))
        # a 99.9 % level interval is inside exactly when the TOST p-value is below 0.0005
        r2 = g5.gate5_paired(L, P, 1 - 2 * 0.0005)
        self.assertTrue(np.array_equal(r2.p_tost <= 0.0005, (r2.lo > -delta) & (r2.hi < delta)))
        np.testing.assert_allclose(r.ratio, np.exp(r.theta))

    def test_validation(self):
        with self.assertRaises(ValueError):
            g5.gate5_paired(np.ones(5), np.ones(4))
        with self.assertRaises(ValueError):
            g5.gate5_paired(np.ones(2), np.ones(2))
        with self.assertRaises(ValueError):
            g5.gate5_paired(-np.ones(5), np.ones(5))


class Gate5Power(unittest.TestCase):
    def test_analytic_power_against_monte_carlo_of_the_full_pipeline(self):
        R = 60000 if HEAVY else 20000
        rng = np.random.default_rng(2024)
        cases = [  # n, ratio, cv_L, cv_P, corr, alpha
            (40, 1.00, 0.25, 0.25, 0.5, 0.05),
            (60, 0.95, 0.20, 0.30, 0.3, 0.05),
            (100, 1.00, 0.30, 0.30, 0.5, 0.05 / 7),  # Holm level of seven tests
        ]
        for n, ratio, cl, cp, rho, alpha in cases:
            L, P = bivariate(rng, R, n, ratio, cl, cp, rho)
            res = g5.gate5_paired(L, P, 1 - 2 * alpha)
            mc = float(np.mean(res.p_tost <= alpha))
            ana = g5.gate5_power(n, cl, cp, rho, alpha, ratio)
            se = math.sqrt(max(mc * (1 - mc), 1e-4) / R)
            self.assertLess(abs(mc - ana), 4 * se + 0.012, msg="n=%d ratio=%g: MC %.4f analytic %.4f" % (n, ratio, mc, ana))

    def test_quadrature_against_adaptive_quadrature(self):
        from scipy import integrate

        def ref(n, sigma, delta, theta, alpha):  # independent adaptive-quadrature evaluation of the same integral
            df, c, psi = n - 1, stats.t.isf(alpha, n - 1) / math.sqrt(n), sigma / math.sqrt(n)
            hi = min(df * (delta / (c * sigma)) ** 2, stats.chi2.isf(1e-13, df))
            f = lambda w: max(stats.norm.cdf((delta - c * sigma * math.sqrt(w / df) - theta) / psi)
                              - stats.norm.cdf((-delta + c * sigma * math.sqrt(w / df) - theta) / psi), 0.0) * stats.chi2.pdf(w, df)
            return integrate.quad(f, 0.0, hi, limit=400, epsabs=1e-14, epsrel=1e-12)[0]
        for n, sigma, theta, alpha in ((4, 0.1, 0.0, 0.05), (12, 0.2, 0.03, 0.05), (60, 0.25, 0.0, 0.007), (400, 0.3, 0.05, 0.05)):
            self.assertAlmostEqual(g5.tost_power(n, sigma, g5.LN_MARGIN, theta, alpha), ref(n, sigma, g5.LN_MARGIN, theta, alpha), places=8)

    def test_power_properties(self):
        s = g5.sd_e(0.25, 0.25, 0.5)
        self.assertAlmostEqual(s, 0.25, places=14)
        self.assertAlmostEqual(g5.sd_e(0.3, 0.2, 1.0), 0.1, places=14)
        pw = [g5.tost_power(n, s) for n in range(5, 200, 7)]
        self.assertTrue(all(b >= a - 1e-12 for a, b in zip(pw, pw[1:])))
        # size of the test: at the boundary of the null the rejection probability is at most alpha
        self.assertLessEqual(g5.tost_power(60, s, theta=math.log(1.1)), 0.05 + 1e-9)
        self.assertGreater(g5.tost_power(60, s, theta=0.0), g5.tost_power(60, s, theta=0.06))
        self.assertAlmostEqual(g5.tost_power(60, s, theta=0.03), g5.tost_power(60, s, theta=-0.03), places=10)

    def test_sample_size_is_the_smallest_n_with_the_target_power(self):
        for cl, cp, rho, alpha in ((0.25, 0.25, 0.5, 0.05), (0.3, 0.3, 0.0, 0.05 / 7), (0.2, 0.35, 0.4, 0.01)):
            n, ok = g5.gate5_sample_size(cl, cp, rho, alpha)
            self.assertTrue(ok)
            self.assertGreaterEqual(g5.gate5_power(n, cl, cp, rho, alpha), 0.8)
            self.assertLess(g5.gate5_power(n - 1, cl, cp, rho, alpha), 0.8)
        n_hi, _ = g5.gate5_sample_size(0.3, 0.3, 0.0, 0.05 / 7)
        n_lo, _ = g5.gate5_sample_size(0.3, 0.3, 0.0, 0.05)
        self.assertGreater(n_hi, n_lo)  # a stricter Holm level needs more seeds
        n_c, _ = g5.gate5_sample_size(0.3, 0.3, 0.6, 0.05 / 7)
        self.assertLess(n_c, n_hi)  # pairing correlation helps

    def test_conservative_inputs(self):
        cvl, cvp, rho = g5.gate5_conservative_inputs(0.4, 0.5, 2.0, 2.2, n_pilot=25, corr=0.6)
        self.assertGreater(cvl, 0.4 / 2.0)
        self.assertGreater(cvp, 0.5 / 2.2)
        self.assertLess(rho, 0.6)
        self.assertEqual(g5.gate5_conservative_inputs(0.4, 0.5, 2.0, 2.2, 25)[2], 0.0)  # uncalibrated: 0


if __name__ == "__main__":
    unittest.main()
