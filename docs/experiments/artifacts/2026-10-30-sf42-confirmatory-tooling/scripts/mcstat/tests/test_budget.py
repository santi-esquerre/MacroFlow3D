import math
import unittest

import numpy as np

from mcstat import budget as bd
from mcstat.tests._paths import HEAVY


def ladder(f, n=24, seed=0, sd=1.0):
    """Per-seed values u_s + f_k: the level differences are exactly f_k - f_j in every bootstrap resample."""
    u = np.random.default_rng(seed).normal(0.0, sd, size=n)
    return u[:, None] + np.asarray(f, dtype=float)[None, :]


class LadderKnownAnswers(unittest.TestCase):
    P = 2.0
    RES = [5.0, 10.0, 20.0]

    def errors(self):  # e(h) = C (1/res)^p with e = 0.01 at the middle level
        return 0.01 * (np.array(self.RES) / 10.0) ** (-self.P)

    def test_finer_level_formula(self):
        x = ladder(self.errors())  # m_k = alpha* + e_k, production level 10 (index 1)
        lb = bd.LadderBound(x, self.RES, ip=1, B=499, seed=1)
        d = lb.details(0.05)
        delta = abs(self.errors()[2] - self.errors()[1])
        want = 1.25 * delta * 4.0 / 3.0  # 1.25 |alpha(h/2) - alpha(h)| r^p / (r^p - 1)
        self.assertAlmostEqual(float(d["order_hat"]), 2.0, places=12)
        self.assertTrue(bool(d["order_branch_used"]))
        self.assertAlmostEqual(float(d["bound"]), want, places=12)
        self.assertAlmostEqual(float(lb.bound(0.05)), want, places=12)
        self.assertGreaterEqual(want, 0.01)  # at least the true bias of the production level (safety factor 1.25)

    def test_finest_level_formula(self):
        x = ladder(self.errors())
        lb = bd.LadderBound(x, self.RES, ip=2, B=499, seed=1)
        delta = abs(self.errors()[2] - self.errors()[1])
        want = 1.25 * delta / (4.0 - 1.0)  # 1.25 |alpha(h_p) - alpha(r h_p)| / (r^p - 1)
        self.assertAlmostEqual(float(lb.bound(0.05)), want, places=12)
        self.assertGreaterEqual(want, self.errors()[2])

    def test_coarsest_production_level_uses_the_next_finer_difference(self):
        x = ladder(self.errors())
        lb = bd.LadderBound(x, self.RES, ip=0, B=499, seed=1)
        delta = abs(self.errors()[1] - self.errors()[0])
        self.assertAlmostEqual(float(lb.bound(0.05)), 1.25 * delta * 4.0 / 3.0, places=12)

    def test_non_monotone_uses_the_fallback(self):
        x = ladder(0.01 * np.array([0.0, 1.0, 0.5]))
        lb = bd.LadderBound(x, self.RES, ip=1, B=499, seed=1)
        d = lb.details(0.05)
        self.assertFalse(bool(d["order_branch_used"]))
        self.assertTrue(np.isnan(d["order_hat"]))
        self.assertAlmostEqual(float(d["bound"]), 3.0 * 0.01 / (2.0 - 1.0), places=12)  # 3 range / (r_min - 1)

    def test_oscillating_four_level_sequence_uses_the_fallback(self):
        x = ladder(0.01 * np.array([0.0, 1.0, 0.0, 1.0]))
        lb = bd.LadderBound(x, [4.0, 8.0, 16.0, 32.0], ip=2, B=499, seed=1)
        d = lb.details(0.05)
        self.assertFalse(bool(d["order_branch_used"]))
        self.assertAlmostEqual(float(d["bound"]), 0.03, places=12)

    def test_order_outside_the_preregistered_range_uses_the_fallback(self):
        for p, kwargs in ((0.2, {}), (6.0, {}), (2.0, {"p_max": 1.5})):
            e = 0.01 * (np.array(self.RES) / 10.0) ** (-p)
            lb = bd.LadderBound(ladder(e), self.RES, ip=1, B=499, seed=1, **kwargs)
            d = lb.details(0.05)
            self.assertFalse(bool(d["order_branch_used"]), msg="p=%g" % p)
            self.assertAlmostEqual(float(d["order_hat"]), p, places=10)
            self.assertAlmostEqual(float(d["bound"]), 3.0 * float(np.ptp(e)), places=12)

    def test_two_levels_and_unequal_ratios_have_no_order(self):
        x2 = ladder([0.0, 0.01])
        lb2 = bd.LadderBound(x2, [10.0, 20.0], ip=0, B=299, seed=1)
        self.assertFalse(bool(lb2.details(0.05)["order_branch_used"]))
        self.assertAlmostEqual(float(lb2.bound(0.05)), 3.0 * 0.01, places=12)
        x3 = ladder(0.01 * np.array([0.0, 0.5, 0.75]))
        lb3 = bd.LadderBound(x3, [4.0, 8.0, 10.0], ip=1, B=299, seed=1)  # ratios 2 and 1.25
        self.assertFalse(bool(lb3.details(0.05)["order_branch_used"]))
        self.assertAlmostEqual(float(lb3.bound(0.05)), 3.0 * 0.0075 / 0.25, places=12)

    def test_noisy_ladder_recovers_the_order_and_is_batched(self):
        rng = np.random.default_rng(4)
        R, n = 6, 40
        base = self.errors()
        x = (rng.normal(0, 1, size=(R, n, 1)) + base[None, None, :] + rng.normal(0, 0.002, size=(R, n, 3)))
        lb = bd.LadderBound(x, self.RES, ip=1, B=499, seed=2)
        self.assertEqual(lb.bound([0.01, 0.05]).shape, (R, 2))
        self.assertTrue(np.all(np.abs(lb.order_hat - 2.0) < 0.7))

    def test_validation(self):
        x = ladder([0.0, 0.1, 0.2])
        with self.assertRaises(ValueError):
            bd.LadderBound(x, [10.0, 5.0, 20.0], ip=1)
        with self.assertRaises(ValueError):
            bd.LadderBound(x, [5.0, 10.0, 20.0], ip=3)
        with self.assertRaises(ValueError):
            bd.LadderBound(x, [5.0, 10.0, 20.0], ip=1, p_min=0.0)


class ContrastBounds(unittest.TestCase):
    def test_e_int_paired_known_answer(self):
        u = np.random.default_rng(1).normal(size=30)
        a8 = u + 1.0
        a16 = u + 1.0 - 0.003  # constant paired difference: every resample sees exactly 0.003
        cb = bd.e_int(a16, a8, B=299, seed=1)
        self.assertAlmostEqual(float(cb.estimate), 0.006, places=14)
        self.assertAlmostEqual(float(cb.bound(0.01)), 0.006, places=12)

    def test_e_w_unpaired_bound_exceeds_the_estimate_by_the_sampling_noise(self):
        rng = np.random.default_rng(2)
        wide = rng.normal(1.20, 0.10, size=30)
        prod = rng.normal(1.00, 0.10, size=30)
        cb = bd.e_w(wide, prod, B=3999, seed=3)
        est = abs(wide.mean() - prod.mean())
        self.assertAlmostEqual(float(cb.estimate), est, places=14)
        se = math.sqrt(wide.var(ddof=1) / 30 + prod.var(ddof=1) / 30)
        self.assertAlmostEqual(float(cb.bound(0.05)), est + 1.645 * se, delta=0.35 * se)
        self.assertGreater(float(cb.bound(0.01)), float(cb.bound(0.05)))

    def test_interaction_contrast_is_the_difference_of_resolution_effects(self):
        rng = np.random.default_rng(3)
        uw, up = rng.normal(size=20), rng.normal(size=25)
        # resolution effect +0.30 in the wide arm and +0.10 in the production-width arm: interaction 0.20
        cb = bd.e_interaction(uw + 0.30, uw, up + 0.10, up, B=299, seed=1)
        self.assertAlmostEqual(float(cb.estimate), 0.20, places=13)
        cb0 = bd.e_interaction(uw + 0.10, uw, up + 0.10, up, B=299, seed=1)
        self.assertAlmostEqual(float(cb0.estimate), 0.0, places=13)

    def test_independent_draws_per_replicate_unless_shared(self):
        rng = np.random.default_rng(1)
        row = rng.normal(size=(1, 20))
        a, b = np.repeat(row, 4, axis=0), np.repeat(row + 0.5 + rng.normal(0, 0.2, size=(1, 20)), 4, axis=0)  # identical rows
        ind = bd.e_int(a, b, B=199, seed=1)
        shared = bd.e_int(a, b, B=199, seed=1, share_draws=True)
        bi, bs = ind.bound(0.1), shared.bound(0.1)
        self.assertEqual(len(set(np.round(bs, 15))), 1)  # one index matrix for the batch: identical replicates, identical bounds
        self.assertGreater(len(set(np.round(bi, 15))), 1)  # independent resamples per replicate
        self.assertTrue(np.allclose(ind.estimate, shared.estimate))
        lad = ladder([0.0, 0.1, 0.2], n=20)
        lb = bd.LadderBound(np.repeat(lad[None], 3, axis=0), [5.0, 10.0, 20.0], ip=1, B=199, seed=2)
        self.assertEqual(lb.bound([0.05, 0.1]).shape, (3, 2))

    def test_bootstrap_components_require_an_explicit_stream(self):
        with self.assertRaises(ValueError):
            bd.e_int(np.zeros(5), np.ones(5), B=99)
        with self.assertRaises(ValueError):
            bd.LadderBound(ladder([0.0, 0.1, 0.2]), [5.0, 10.0, 20.0], ip=1, B=99)

    def test_group_size_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            bd.ContrastBound("x", [(np.zeros(5), "g", 1.0), (np.zeros(6), "g", -1.0)])


class MonotoneAndPValue(unittest.TestCase):
    def build(self, seed=0, alpha=0.6, n=50):
        rng = np.random.default_rng(seed)
        prod = rng.normal(alpha, 0.5, size=n)
        lad = ladder(0.01 * np.array([4.0, 1.0, 0.25]) + 0.0, n=24, seed=seed + 1, sd=0.4) + rng.normal(0, 0.03, size=(24, 3))
        comps = [bd.LadderBound(lad, [5.0, 10.0, 20.0], ip=1, B=1999, rng=rng),
                 bd.e_int(rng.normal(0, 0.01, 24), rng.normal(0, 0.01, 24), B=1999, rng=rng)]
        return bd.RobustLimit(bd.StudentMean(prod), comps)

    def test_enforced_bounds_are_non_increasing_in_the_tail_and_dominate_the_raw_ones(self):
        rl = self.build()
        a = np.geomspace(1e-4, 1.0, 120)
        enforced = rl.component_bounds(a)
        with np.errstate(invalid="ignore"):  # inf - inf at tails the B resamples cannot resolve
            d = np.diff(enforced, axis=-1)
        self.assertTrue(np.all(d[~np.isnan(d)] <= 0.0))
        self.assertTrue(np.isfinite(enforced[:, -1]).all())
        for k, c in enumerate(rl.components):
            raw = c.bound(a / (2.0 * rl.K))
            self.assertTrue(np.all(enforced[k] >= raw - 1e-15))
        self.assertTrue(bd.is_non_increasing_in_tail(lambda t: rl.component_bounds(t * 2.0 * rl.K)[0], a / (2.0 * rl.K)))

    def test_robust_limit_is_non_decreasing_in_a_and_p_value_is_its_inf(self):
        rl = self.build()
        a = np.geomspace(1e-4, 1.0, 200)
        RL = rl.lower(a)
        self.assertTrue(np.all(np.diff(RL[np.isfinite(RL)]) >= -1e-12))
        pv = rl.p_value()
        self.assertFalse(bool(pv.never_positive))
        lat = rl.lattice
        i = int(np.searchsorted(lat, float(pv.p)))
        self.assertAlmostEqual(lat[i], float(pv.p), places=15)
        self.assertGreater(float(rl.lower(lat[i])), 0.0)
        self.assertLessEqual(float(rl.lower(lat[i - 1])), 0.0)
        self.assertLess(lat[i] / lat[i - 1], 1.012)  # stated precision of the monotone search

    def test_never_positive_when_the_bias_dominates(self):
        rng = np.random.default_rng(1)
        prod = rng.normal(0.05, 0.5, size=40)
        big = bd.ContrastBound("big", [(np.full(20, 5.0), "g", 1.0), (np.zeros(20), "g", -1.0)], B=299, seed=1)
        pv = bd.RobustLimit(bd.StudentMean(prod), [big]).p_value()
        self.assertTrue(bool(pv.never_positive))
        self.assertEqual(float(pv.p), 1.0)

    def test_report_returns_intermediates_and_intersection_union(self):
        rl = self.build()
        rep = rl.report([0.01, 0.1])
        self.assertEqual(set(rep["components"]), {"E_h", "E_int"})
        self.assertIn("details", rep["components"]["E_h"])
        self.assertEqual(rep["RL"].shape, (2,))
        self.assertAlmostEqual(float(bd.intersection_union(0.02, 0.2, 0.05)), 0.2)
        np.testing.assert_allclose(bd.intersection_union(np.array([0.1, 0.5]), np.array([0.3, 0.2])), [0.3, 0.5])

    def test_needs_a_component(self):
        with self.assertRaises(ValueError):
            bd.RobustLimit(bd.StudentMean(np.arange(5.0)), [])

    def test_bootstrap_t_sampling_object_plugs_in(self):
        prod = np.random.default_rng(5).normal(1.0, 0.3, size=60)
        comp = bd.e_int(np.zeros(20), np.zeros(20) + 1e-3 * np.arange(20) / 20, B=999, seed=1)
        rl = bd.RobustLimit(bd.BootstrapTMean(prod, B=999, seed=2), [comp])
        self.assertTrue(np.isfinite(float(rl.lower(0.2))))
        self.assertEqual(float(bd.BootstrapTMean(prod, B=99, seed=2).lower(1e-4)), -np.inf)  # unresolvable tail -> trivial


class UnionBoundMonteCarlo(unittest.TestCase):
    """Synthetic world with known biases: P(RL_e(a) > alpha) <= a, and each component bound covers its true bias."""
    ALPHA = 0.5
    BIAS_H, BIAS_W, BIAS_INT = 0.06, 0.04, 0.02

    def simulate(self, R, n_prod=40, n_lad=25, n_w=30, n_int=25, seed=11, safety=1.25):
        rng = np.random.default_rng(seed)
        res = np.array([5.0, 10.0, 20.0])
        e_h = self.BIAS_H * (res / 10.0) ** (-2.0)  # e(h) = C h^2, e(h_p) = BIAS_H
        prod = self.ALPHA + self.BIAS_H + self.BIAS_W + self.BIAS_INT + rng.normal(0, 0.3, size=(R, n_prod))
        lad = (self.ALPHA + self.BIAS_W + e_h)[None, None, :] + rng.normal(0, 0.3, size=(R, n_lad, 1)) \
            + rng.normal(0, 0.04, size=(R, n_lad, 3))
        wide = self.ALPHA + self.BIAS_H + rng.normal(0, 0.3, size=(R, n_w))  # no width bias at W*
        narrow = self.ALPHA + self.BIAS_H + self.BIAS_W + rng.normal(0, 0.3, size=(R, n_w))
        u = rng.normal(0, 0.3, size=(R, n_int))
        h8 = u + self.BIAS_INT  # bias of the production h/8 pass
        h16 = u + 0.5 * self.BIAS_INT + rng.normal(0, 0.005, size=(R, n_int))  # order 1: the bias halves, E_int = 2 |diff| = BIAS_INT
        comps = {"E_h": bd.LadderBound(lad, res, ip=1, safety=safety, B=499, rng=rng),
                 "E_W": bd.e_w(wide, narrow, B=499, rng=rng),
                 "E_int": bd.e_int(h16, h8, B=499, rng=rng)}
        return bd.StudentMean(prod), comps

    def test_each_component_bound_covers_its_bias(self):
        R = 3000 if HEAVY else 600
        samp, comps = self.simulate(R, safety=1.0)
        truth = {"E_h": self.BIAS_H, "E_W": self.BIAS_W, "E_int": self.BIAS_INT}
        for t in (0.2, 0.05):
            for name, c in comps.items():
                miss = np.mean(c.bound(t) < truth[name])
                se = math.sqrt(t * (1 - t) / R)
                self.assertLessEqual(miss, t + 3 * se, msg="%s at tail %g: miss %.3f" % (name, t, miss))

    def test_robust_limit_union_bound(self):
        R = 3000 if HEAVY else 600
        samp, comps = self.simulate(R)
        rl = bd.RobustLimit(samp, list(comps.values()))
        for a in (0.4, 0.2):
            RL = rl.lower(a)
            self.assertEqual(RL.shape, (R,))
            miss = np.mean(RL > self.ALPHA)
            self.assertLessEqual(miss, a + 3 * math.sqrt(a * (1 - a) / R), msg="a=%g miss=%.3f" % (a, miss))
        # with the sampling noise removed and a large true effect the test does reject (the budget is not vacuous)
        strong = bd.RobustLimit(bd.StudentMean(self.ALPHA + 3.0 + np.random.default_rng(1).normal(0, 0.1, size=(R, 40))),
                                list(comps.values()))
        self.assertTrue(np.all(strong.lower(0.2) > 0.0))
        pv = rl.p_value()
        self.assertEqual(pv.p.shape, (R,))
        # p-values of a null world are super-uniform: P(p <= a) <= a (checked at the null alpha_e = 0)
        null_samp, null_comps = self.simulate(R, seed=12)
        null = bd.RobustLimit(bd.StudentMean(null_samp.x - self.ALPHA), list(null_comps.values()))
        for a in (0.3, 0.1):
            self.assertLessEqual(np.mean(null.p_value().p <= a), a + 3 * math.sqrt(a * (1 - a) / R))


if __name__ == "__main__":
    unittest.main()
