import contextlib
import io
import math
import unittest

import numpy as np
from scipy import stats

from mcstat import blinded as bl
from mcstat import gate5 as g5
from mcstat import multiplicity as mt
from mcstat import sizing as sz
from mcstat.tests._ref import t_ppf


class Holm(unittest.TestCase):
    def test_hand_example(self):
        # p = (0.01, 0.04, 0.03, 0.005), m = 4: sorted 0.005 -> x4 = 0.02; 0.01 -> x3 = 0.03; 0.03 -> x2 = 0.06; 0.04 -> x1 = 0.04 -> 0.06
        adj, rej = mt.holm([0.01, 0.04, 0.03, 0.005], 0.05)
        np.testing.assert_allclose(adj, [0.03, 0.06, 0.06, 0.02], atol=1e-15)
        self.assertEqual(list(rej), [True, False, False, True])

    def test_step_down_stops_at_the_first_failure(self):
        adj, rej = mt.holm([0.001, 0.011, 0.012], 0.05)  # 0.003, 0.022, max(0.012, 0.022) = 0.022 -> all rejected
        np.testing.assert_allclose(adj, [0.003, 0.022, 0.022], atol=1e-15)
        self.assertTrue(rej.all())
        adj, rej = mt.holm([0.001, 0.30, 0.0001], 0.05)  # 0.0001 x3, 0.001 x2, 0.30 x1 -> only the two small ones
        np.testing.assert_allclose(adj, [0.002, 0.30, 0.0003], atol=1e-15)
        self.assertEqual(list(rej), [True, False, True])
        # the SF-43 family of seven p-values: the worst level is alpha / 7
        adj, rej = mt.holm([0.007] + [0.5] * 6, 0.05)
        self.assertAlmostEqual(adj[0], 0.049, places=12)
        self.assertTrue(rej[0])
        adj, rej = mt.holm([0.0072] + [0.5] * 6, 0.05)
        self.assertFalse(rej[0])

    def test_monotone_capped_and_nan(self):
        adj, rej = mt.holm([0.4, 0.5, float("nan")], 0.05)
        self.assertTrue(np.all(adj <= 1.0))
        self.assertFalse(rej.any())
        self.assertEqual(mt.holm([], 0.05)[0].size, 0)


class Sizing(unittest.TestCase):
    def test_sd_upper_matches_the_sf39_chi_square_rule(self):
        for s, n in ((0.5, 20), (2.0, 71)):
            self.assertAlmostEqual(float(sz.sd_upper(s, n)), s * math.sqrt((n - 1) / stats.chi2.ppf(0.20, n - 1)), places=13)
        self.assertGreater(float(sz.sd_upper(1.0, 10)), float(sz.sd_upper(1.0, 100)))  # tighter with more data
        self.assertLess(float(sz.mean_lower(5.0, 2.0, 25)), 5.0)
        self.assertLess(float(sz.corr_lower(0.6, 25)), 0.6)
        self.assertGreater(float(sz.corr_lower(0.6, 25)), 0.0)
        with self.assertRaises(ValueError):
            sz.corr_lower(0.5, 3)

    def test_n_for_halfwidth_is_the_smallest_meeting_the_target(self):
        for sd, half, lvl in ((1.0, 0.5, 0.95), (0.3, 0.02, 0.90), (2.0, 0.05, 0.95)):
            n = sz.n_for_halfwidth(sd, half, lvl)
            hw = lambda m: t_ppf(0.5 * (1 + lvl), m - 1) * sd / math.sqrt(m)
            self.assertLessEqual(hw(n), half)
            self.assertGreater(hw(n - 1), half)
        self.assertEqual(sz.n_for_halfwidth(0.0, 1.0), 2)

    def test_absolute_and_relative_targets_use_the_conservative_inputs(self):
        plain = sz.n_for_halfwidth(0.5, 0.05)
        self.assertGreater(sz.n_absolute_target(0.5, 20, 0.05), plain)  # sd inflated to its 80 % upper bound
        n_rel = sz.n_relative_target(1.0, 0.5, 20, 0.10)
        n_nominal = sz.n_for_halfwidth(0.5, 0.10 * 1.0)
        self.assertGreater(n_rel, n_nominal)  # mean deflated AND sd inflated
        with self.assertRaises(ValueError):
            sz.n_relative_target(0.01, 0.5, 20, 0.1)

    def test_clamp_resize_upward_only_at_most_double(self):
        self.assertEqual(sz.clamp_resize(30, 40), 40)
        self.assertEqual(sz.clamp_resize(55, 40), 55)
        self.assertEqual(sz.clamp_resize(500, 40), 80)
        self.assertEqual(sz.clamp_resize(81, 41), 81)
        self.assertEqual(sz.clamp_resize(1000, 41), 82)


class Gate4PowerAndAllocation(unittest.TestCase):
    COMPS = [{"sd": 0.10, "cost": 3.0, "slack": 0.004}, {"sd": 0.05, "cost": 1.0, "slack": 0.002}, {"sd": 0.20, "cost": 0.5}]

    def test_power_model_properties(self):
        comps = [dict(c, n=30) for c in self.COMPS]
        p0 = sz.gate4_power(0.05, 0.01, 100, 0.4, comps)
        self.assertTrue(0.0 < p0 < 1.0)
        # more production seeds -> more power; larger effect -> more power; zero effect and zero slack -> below one half
        self.assertGreater(sz.gate4_power(0.05, 0.01, 400, 0.4, comps), p0)
        self.assertGreater(sz.gate4_power(0.08, 0.01, 100, 0.4, comps), p0)
        self.assertLess(sz.gate4_power(0.0, 0.01, 10 ** 6, 0.4, [{"n": 10 ** 6, "sd": 0.1}]), 0.5)
        big = [dict(c, n=10 ** 7) for c in self.COMPS]
        self.assertGreater(sz.gate4_power(0.05, 0.01, 10 ** 7, 0.4, big), 0.99)
        # normal quantiles (use_t=False) agree with Student for large n
        self.assertAlmostEqual(sz.gate4_power(0.05, 0.01, 10 ** 5, 0.4, [dict(c, n=10 ** 5) for c in self.COMPS], use_t=False),
                               sz.gate4_power(0.05, 0.01, 10 ** 5, 0.4, [dict(c, n=10 ** 5) for c in self.COMPS], use_t=True), places=3)

    def test_allocation_reaches_the_power_and_beats_the_equal_split(self):
        n_p, ns, pw, cost = sz.allocate_gate4(0.05, 0.01, 0.4, 2.0, self.COMPS, power=0.8)
        self.assertGreaterEqual(pw, 0.8)
        comps = [dict(c, n=n) for c, n in zip(self.COMPS, ns)]
        self.assertAlmostEqual(sz.gate4_power(0.05, 0.01, n_p, 0.4, comps), pw, places=14)
        costs = [2.0] + [c["cost"] for c in self.COMPS]
        self.assertAlmostEqual(cost, sum(c * n for c, n in zip(costs, [n_p] + ns)), places=9)
        # the equal-n allocation that reaches the same power is at least as expensive
        ok = lambda n: sz.gate4_power(0.05, 0.01, n, 0.4, [dict(c, n=n) for c in self.COMPS]) >= 0.8
        lo, hi = 2, 4
        while not ok(hi):
            lo, hi = hi, 2 * hi
        while hi - lo > 1:  # power is increasing in the common n
            mid = (lo + hi) // 2
            lo, hi = (lo, mid) if ok(mid) else (mid, hi)
        n_eq = hi
        self.assertLessEqual(cost, sum(costs) * n_eq)
        self.assertTrue(all(n >= 2 for n in [n_p] + ns))
        with self.assertRaises(ValueError):
            sz.allocate_gate4(0.005, 0.01, 0.4, 2.0, self.COMPS)  # slack 0.006 exceeds the planning alpha


class BlindedReestimation(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(10)
        self.arms = {"prod": rng.normal(3.0, 0.8, 25), "pair": rng.normal(0.4, 0.3, 25)}

    def test_output_is_standard_deviations_only_and_location_free(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            r = bl.blinded_reestimate(self.arms, 60, {"prod": 0.05, "pair": 0.03})
            shifted = bl.blinded_reestimate({k: v + 1.0e6 for k, v in self.arms.items()}, 60, {"prod": 0.05, "pair": 0.03})
            sds = bl.blinded_sds(self.arms)
        self.assertEqual(buf.getvalue(), "")  # nothing is printed
        self.assertEqual(set(r.__dataclass_fields__), {"sd", "n_obs", "n_required", "n_new"})
        # shifting every value by a constant (changing every mean and quantile) changes no returned number
        for f in r.__dataclass_fields__:
            for k in r.sd:
                self.assertAlmostEqual(getattr(r, f)[k], getattr(shifted, f)[k], places=5, msg="%s/%s" % (f, k))
        self.assertAlmostEqual(r.sd["prod"], sds["prod"], places=14)
        # no returned number equals a mean or a quantile of the data
        leaked = {float(v.mean()) for v in self.arms.values()} | {float(np.median(v)) for v in self.arms.values()}
        values = [x for d in (r.sd, r.n_obs, r.n_required, r.n_new) for x in d.values()]
        for x in values:
            self.assertNotIn(float(x), leaked)
        self.assertNotIn("mean", repr(r).lower())

    def test_resize_rule(self):
        r = bl.blinded_reestimate(self.arms, 60, {"prod": 0.05, "pair": 0.03})
        for k in ("prod", "pair"):
            self.assertGreaterEqual(r.n_new[k], 60)  # upward only
            self.assertLessEqual(r.n_new[k], 120)  # at most 2x
        self.assertEqual(r.n_new["prod"], 120)  # sd 0.8 against 0.05: the requirement is far above the cap
        easy = bl.blinded_reestimate(self.arms, {"prod": 60, "pair": 60}, {"prod": 5.0, "pair": 5.0})
        self.assertEqual(easy.n_new, {"prod": 60, "pair": 60})  # a loose target never shrinks N
        self.assertEqual(r.n_obs["prod"], 25)
        with self.assertRaises(ValueError):
            bl.blinded_reestimate({"x": np.array([1.0, 2.0])}, 10, {"x": 0.1})

    def test_gate5_variant_uses_planning_means_not_data_means(self):
        rng = np.random.default_rng(12)
        P = rng.normal(2.0, 0.5, 30)
        L = 0.9 * P + rng.normal(0.0, 0.2, 30)
        r = bl.blinded_gate5(L, P, 1.8, 2.0, 0.05 / 7, n_orig=50)
        r_shift = bl.blinded_gate5(L + 100.0, P - 50.0, 1.8, 2.0, 0.05 / 7, n_orig=50)
        self.assertAlmostEqual(r.sd_L, r_shift.sd_L, places=10)
        self.assertAlmostEqual(r.corr, r_shift.corr, places=12)
        self.assertEqual((r.n_required, r.n_new), (r_shift.n_required, r_shift.n_new))
        self.assertTrue(50 <= r.n_new <= 100)
        # equals the direct sizing from the same conservative inputs
        cv_l = float(sz.sd_upper(r.sd_L, 30)) / 1.8
        cv_p = float(sz.sd_upper(r.sd_P, 30)) / 2.0
        n, _ = g5.gate5_sample_size(cv_l, cv_p, float(sz.corr_lower(r.corr, 30)), 0.05 / 7)
        self.assertEqual(r.n_required, n)


if __name__ == "__main__":
    unittest.main()
