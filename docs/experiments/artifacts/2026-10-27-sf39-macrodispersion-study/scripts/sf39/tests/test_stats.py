import math
import unittest
import warnings

import numpy as np
from scipy import stats as st

from sf39 import stats as S


def arr(mean, sd, n=20):
    """Deterministic sample with exactly this mean and (ddof=1) sd."""
    z = np.linspace(-1.0, 1.0, n)
    z = (z - z.mean()) / z.std(ddof=1)
    return mean + sd * z


class TestIntervals(unittest.TestCase):
    def test_t_two_sided_vs_scipy(self):
        x = np.random.default_rng(1).normal(2.0, 3.0, 17)
        mean, lo, hi, n, sd = S.t_interval(x, 0.90)
        slo, shi = st.t.interval(0.90, 16, loc=x.mean(), scale=st.sem(x))
        self.assertAlmostEqual(lo, slo, 12)
        self.assertAlmostEqual(hi, shi, 12)
        self.assertEqual(n, 17)
        self.assertAlmostEqual(sd, x.std(ddof=1), 12)
        self.assertAlmostEqual(mean, x.mean(), 12)

    def test_t_one_sided(self):
        x = np.random.default_rng(2).normal(0.0, 1.0, 12)
        tc = st.t.ppf(0.95, 11)
        _, lo, hi, _, sd = S.t_interval(x, 0.95, "upper")
        self.assertEqual(lo, -np.inf)
        self.assertAlmostEqual(hi, x.mean() + tc * sd / math.sqrt(12), 12)
        _, lo, hi, _, _ = S.t_interval(x, 0.95, "lower")
        self.assertEqual(hi, np.inf)
        self.assertAlmostEqual(lo, x.mean() - tc * sd / math.sqrt(12), 12)
        with self.assertRaises(ValueError):
            S.t_interval(x, 0.9, "both")

    def test_t_n1_nan(self):
        mean, lo, hi, n, sd = S.t_interval([3.0])
        self.assertTrue(np.isnan(lo) and np.isnan(hi))

    def test_welch_closed_form(self):
        a = np.array([1.0, 2.0, 4.0, 7.0, 3.0])
        b = np.array([0.5, 1.5, 2.5, 0.0, 1.0, 2.0, 3.0, 1.2])
        va, vb = a.var(ddof=1) / 5, b.var(ddof=1) / 8
        nu = (va + vb) ** 2 / (va**2 / 4 + vb**2 / 7)
        half = st.t.ppf(0.95, nu) * math.sqrt(va + vb)
        d, lo, hi, nu_o, se = S.welch_interval(a, b, 0.90)
        self.assertAlmostEqual(d, a.mean() - b.mean(), 12)
        self.assertAlmostEqual(nu_o, nu, 10)
        self.assertAlmostEqual(lo, d - half, 10)
        self.assertAlmostEqual(hi, d + half, 10)
        ref = st.ttest_ind(a, b, equal_var=False).confidence_interval(0.90)
        self.assertAlmostEqual(lo, ref.low, 8)
        self.assertAlmostEqual(hi, ref.high, 8)

    def test_paired_and_align(self):
        x, y = arr(1.0, 0.2), arr(0.5, 0.1)
        m, lo, hi, n, sd = S.paired_interval(x, y)
        self.assertAlmostEqual(m, 0.5, 12)
        with self.assertRaises(ValueError):
            S.paired_interval(x, y[:-1])
        s, a, b = S.align_by_seed([5003, 5001, 5002], [3.0, 1.0, 2.0], [5002, 5003, 5009], [20.0, 30.0, 90.0])
        self.assertEqual(list(s), [5002, 5003])
        self.assertEqual(list(a), [2.0, 3.0])
        self.assertEqual(list(b), [20.0, 30.0])


class TestAncova(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.nom = 1.0
        self.s2 = 1.0 + rng.normal(0, 0.1, 25)
        self.y = 0.30 + 0.4 * (self.s2 - self.nom) + rng.normal(0, 0.02, 25)

    def test_vs_matrix_form(self):
        c = self.s2 - self.nom
        X = np.column_stack([np.ones(25), c])
        beta = np.linalg.lstsq(X, self.y, rcond=None)[0]
        res = self.y - X @ beta
        s2e = (res**2).sum() / 23
        cov = s2e * np.linalg.inv(X.T @ X)
        r = S.ancova_adjusted(self.y, self.s2, self.nom)
        self.assertAlmostEqual(r.a, beta[0], 12)
        self.assertAlmostEqual(r.b, beta[1], 12)
        self.assertAlmostEqual(r.se_a, math.sqrt(cov[0, 0]), 12)
        self.assertEqual(r.dof, 23)
        self.assertEqual(r.n, 25)

    def test_vs_welch_satterthwaite(self):
        rng = np.random.default_rng(8)
        L = rng.normal(0.31, 0.05, 14)
        anc = S.ancova_adjusted(self.y, self.s2, self.nom)
        vL = L.var(ddof=1) / 14
        va = anc.se_a**2
        nu = (va + vL) ** 2 / (va**2 / 23 + vL**2 / 13)
        half = st.t.ppf(0.95, nu) * math.sqrt(va + vL)
        d, lo, hi, nu_o, _ = S.ancova_vs(L, self.y, self.s2, self.nom)
        self.assertAlmostEqual(d, L.mean() - anc.a, 12)
        self.assertAlmostEqual(nu_o, nu, 10)
        self.assertAlmostEqual(lo, d - half, 10)
        self.assertAlmostEqual(hi, d + half, 10)

    def test_degenerate_covariate_raises(self):
        with self.assertRaises(ValueError):
            S.ancova_adjusted([1.0, 2.0, 3.0, 4.0], [1.0] * 4, 1.0)


class TestBCa(unittest.TestCase):
    def setUp(self):
        warnings.simplefilter("error")  # no warning may escape a BCa call
        self.addCleanup(warnings.resetwarnings)

    def test_one_sample_paired_difference(self):
        d = np.random.default_rng(3).normal(0.5, 1.0, 30)
        lo, hi, flag = S.bca_interval(d, 0.90)
        self.assertFalse(flag)
        self.assertLess(lo, d.mean())
        self.assertGreater(hi, d.mean())
        _, tlo, thi, _, _ = S.t_interval(d, 0.90)
        self.assertLess(abs(lo - tlo), 0.25)
        self.assertEqual((lo, hi, flag), S.bca_interval(d, 0.90))  # fresh generator: deterministic

    def test_one_sided(self):
        d = np.random.default_rng(4).normal(0.5, 1.0, 30)
        lo, hi, flag = S.bca_interval(d, 0.95, "upper")
        self.assertEqual(lo, -np.inf)
        self.assertTrue(np.isfinite(hi) and not flag)
        lo, hi, flag = S.bca_interval(d, 0.95, "lower")
        self.assertEqual(hi, np.inf)
        self.assertTrue(np.isfinite(lo) and not flag)

    def test_welch_independent_different_sizes(self):
        rng = np.random.default_rng(5)
        a, b = rng.normal(1.0, 1.0, 25), rng.normal(0.0, 1.0, 40)
        lo, hi, flag = S.bca_welch(a, b)
        self.assertFalse(flag)
        self.assertLess(lo, a.mean() - b.mean())
        self.assertGreater(hi, a.mean() - b.mean())

    def test_ancova_index_sample(self):
        rng = np.random.default_rng(6)
        s2 = 1.0 + rng.normal(0, 0.1, 30)
        y = 0.3 + 0.5 * (s2 - 1.0) + rng.normal(0, 0.03, 30)
        a = S.ancova_adjusted(y, s2, 1.0).a
        lo, hi, flag = S.bca_ancova_index(y, s2, 1.0)
        self.assertFalse(flag)
        self.assertLess(lo, a)
        self.assertGreater(hi, a)
        L = rng.normal(0.31, 0.04, 20)
        lo, hi, flag = S.bca_ancova_vs(L, y, s2, 1.0)
        self.assertFalse(flag)
        self.assertLess(lo, L.mean() - a)
        self.assertGreater(hi, L.mean() - a)

    def test_degenerate_returns_nan_and_flag(self):
        for out in (S.bca_interval(np.full(10, 0.25)),
                    S.bca_interval([1.0]),
                    S.bca_welch(np.full(8, 1.0), np.full(9, 1.0)),
                    S.bca_ancova_index(np.full(6, 0.3), np.linspace(0.9, 1.1, 6), 1.0),
                    S.bca_ancova_index([0.1, 0.2, 0.3, 0.4], [1.0] * 4, 1.0)):
            lo, hi, flag = out
            self.assertTrue(np.isnan(lo) and np.isnan(hi) and flag)


class TestEquivalence(unittest.TestCase):
    def test_labels_and_edges(self):
        m = 0.01
        self.assertEqual(S.equivalence(-0.01, 0.01, m), "converged")      # touching the margin is inside
        self.assertEqual(S.equivalence(-0.0100001, 0.0, m), "inconclusive")
        self.assertEqual(S.equivalence(0.0, 0.0100001, m), "inconclusive")
        self.assertEqual(S.equivalence(0.0100001, 0.02, m), "not converged")
        self.assertEqual(S.equivalence(0.01, 0.02, m), "inconclusive")     # lo == m is not entirely outside
        self.assertEqual(S.equivalence(-0.02, -0.0100001, m), "not converged")
        self.assertEqual(S.equivalence(-0.02, -0.01, m), "inconclusive")
        self.assertEqual(S.equivalence(-0.02, 0.02, m), "inconclusive")
        self.assertEqual(S.equivalence(np.nan, np.nan, m), "inconclusive")

    def test_gate5_labels(self):
        self.assertEqual(S.equivalence(-0.01, 0.01, 0.01, "gate5"), "agree")
        self.assertEqual(S.equivalence(0.02, 0.03, 0.01, "gate5"), "disagree")
        self.assertEqual(S.equivalence(-0.02, 0.03, 0.01, "gate5"), "inconclusive")

    def test_margins(self):
        self.assertAlmostEqual(S.margin_transverse(0.1), 0.005)
        self.assertAlmostEqual(S.margin_transverse(-0.1), 0.005)
        self.assertAlmostEqual(S.margin_transverse(0.01), 1.25e-3)  # floor
        self.assertAlmostEqual(S.margin_lam(0.4), 0.01)
        self.assertAlmostEqual(S.m_h("TP", 0.10, 0.02), 1.25e-3)
        self.assertAlmostEqual(S.m_h("TP", 0.10, 0.20), 0.005)
        self.assertAlmostEqual(S.m_h("LAM", 0.4, 0.2), 0.005)
        with self.assertRaises(ValueError):
            S.margin("X", 1.0)

    def test_halfwidth(self):
        x = arr(0.1, 0.05, 20)
        mean, hw, target, met = S.halfwidth_check(x, S.halfwidth_target("TP", 0.1))
        self.assertAlmostEqual(hw, st.t.ppf(0.975, 19) * 0.05 / math.sqrt(20), 12)
        self.assertAlmostEqual(target, 0.01, 12)
        self.assertFalse(met)
        self.assertTrue(S.halfwidth_check(arr(0.1, 0.01, 20), 0.01)[3])
        self.assertEqual(S.halfwidth_target("TAM", 0.001), 2.5e-3)
        self.assertAlmostEqual(S.halfwidth_target("LAM", 0.4), 0.02)


class TestReporting(unittest.TestCase):
    def test_richardson(self):
        r = S.richardson_guard(4.0, 1.0, 0.25)  # e = h^2 exactly, Q = 0
        self.assertTrue(r["applied"])
        self.assertAlmostEqual(r["p_obs"], 2.0)
        self.assertAlmostEqual(r["prod_error"], 1.0)
        self.assertAlmostEqual(r["bound"], 0.75)
        r = S.richardson_guard(2.0, 1.0, 0.5)  # order 1: outside the guard
        self.assertFalse(r["applied"])
        self.assertIsNone(r["prod_error"])
        self.assertAlmostEqual(r["bound"], 0.5)
        self.assertFalse(S.richardson_guard(8.0, 1.0, 0.125)["applied"])  # order 3
        r = S.richardson_guard(1.0, 2.0, 1.5)  # not monotone
        self.assertFalse(r["monotone"] or r["applied"])
        self.assertTrue(np.isnan(r["p_obs"]))
        self.assertAlmostEqual(r["bound"], 0.5)
        self.assertTrue(S.richardson_guard(-4.0, -1.0, -0.25)["applied"])  # decreasing sequence

    def test_splithalf(self):
        a = arr(0.3, 0.1)
        r = S.splithalf_fraction(a + 0.05, a - 0.05, a)
        self.assertAlmostEqual(r["component"], 0.01 / 4)
        self.assertAlmostEqual(r["fraction"], (0.01 / 4) / 0.01)
        self.assertTrue(S.splithalf_fraction(a + 0.0501, a - 0.0501, a)["flag"])
        self.assertFalse(S.splithalf_fraction(a, a, a)["flag"])
        r = S.splithalf_fraction(a + 1, a, np.full(20, 0.25))
        self.assertTrue(r["flag"] and np.isinf(r["fraction"]))

    def test_delta_half(self):
        r = S.delta_half(arr(0.01, 0.005))
        self.assertTrue(r["decreasing"])
        self.assertFalse(S.delta_half(arr(0.001, 0.005))["decreasing"])
        self.assertFalse(S.delta_half(arr(-0.01, 0.001))["decreasing"])

    def test_rho_readings(self):
        m, Sv = 0.005, 0.5  # 0.1 m/S = 1e-3 ; sqrt(0.2 m/S) = sqrt(2e-3) = 0.0447
        r = S.rho_readings(np.full(50, 5e-4), m, Sv)
        self.assertTrue(r["pathwise_resolved"])
        self.assertFalse(r["pathwise_limited"] or r["decorrelated"])
        rho = np.full(21, 0.05)
        rho[:2] = 1e-4
        r = S.rho_readings(rho, m, Sv)
        self.assertFalse(r["pathwise_resolved"])
        self.assertTrue(r["pathwise_limited"])
        self.assertFalse(r["decorrelated"])
        r = S.rho_readings(np.full(10, 0.5), m, Sv)
        self.assertTrue(r["decorrelated"])
        v = np.append(np.full(19, 1e-4), 10.0)
        self.assertAlmostEqual(S.rho_readings(v, m, Sv)["p95"], np.percentile(v, 95))  # linear interpolation

    def test_rho_growth_fits(self):
        x0, ell = 0.2, 0.04
        x1 = x0 + np.array([1.0, 2.0, 4.0, 8.0, 16.0]) * ell
        x1 = np.concatenate([[x0 + 0.5 * ell], x1])  # first plane is before x0 + ell: excluded
        med = 0.01 * np.exp(3.0 * x1)
        mat = np.vstack([med * 0.9, med, med * 1.1])
        f = S.rho_growth_fits(x1, mat, x0, ell)
        self.assertEqual(f["n"], 5)
        self.assertAlmostEqual(f["exp"]["slope"], 3.0, 10)
        self.assertAlmostEqual(f["exp"]["r2"], 1.0, 10)
        med = 0.02 * (x1 - x0) ** 1.5
        f = S.rho_growth_fits(x1, np.vstack([med, med]), x0, ell)
        self.assertAlmostEqual(f["power"]["slope"], 1.5, 10)
        self.assertIsNone(S.rho_growth_fits(x1[:3], np.ones((2, 3)), x0, ell)["exp"])

    def test_width_check(self):
        wide, narrow = arr(0.10, 0.002, 30), arr(0.101, 0.002, 30)
        r = S.width_check(wide, narrow)
        self.assertAlmostEqual(r["margin"], 0.01)
        self.assertEqual(r["label"], "agree")
        self.assertEqual(S.width_check(arr(0.2, 0.002, 30), arr(0.1, 0.002, 30))["label"], "disagree")
        self.assertEqual(S.width_check(arr(0.10, 0.05, 30), arr(0.08, 0.05, 30))["label"], "inconclusive")

    def test_pollock_and_dd(self):
        cons = arr(0.05, 0.01)
        poll = cons + arr(0.004, 0.001)
        m, lo, hi, n, _ = S.delta_pollock(poll, cons)
        self.assertAlmostEqual(m, 0.004, 12)
        self.assertGreater(lo, 0)
        dd = S.diff_in_diff(arr(0.004, 0.001), arr(0.001, 0.001))
        self.assertAlmostEqual(dd[0], 0.003, 12)


if __name__ == "__main__":
    unittest.main()
