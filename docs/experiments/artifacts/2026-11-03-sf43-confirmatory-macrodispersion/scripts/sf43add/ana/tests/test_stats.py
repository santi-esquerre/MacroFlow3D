import math
import unittest

import numpy as np

import sf43ana  # noqa: F401
from mcstat.gate5 import gate5_paired

from sf43add.ana import stats


def sample(n=60, seed=1, shift=1.1, cv=0.3):
    rng = np.random.default_rng(seed)
    common = np.exp(cv * rng.standard_normal(n))
    return common * (1 + 0.05 * rng.standard_normal(n)), shift * common * (1 + 0.05 * rng.standard_normal(n))


class PairedRatio(unittest.TestCase):
    def test_is_gate5_paired_with_the_pre_registered_settings(self):
        y, x = sample()
        r = stats.paired_ratio(x, y)
        g = gate5_paired(x, y, level=0.90, method="bootstrap-t", B=2000, seed=20261104)
        s = gate5_paired(x, y, level=0.90, method="student")
        self.assertEqual([r["ratio"], *r["ci90"]], [float(g.ratio), float(g.ratio_lo), float(g.ratio_hi)])
        self.assertEqual(r["student_ci90"], [float(s.ratio_lo), float(s.ratio_hi)])
        self.assertAlmostEqual(r["ratio"], x.mean() / y.mean(), places=13)
        self.assertEqual(r["n"], 60)

    def test_not_evaluable_never_raises(self):
        self.assertFalse(stats.paired_ratio([1.0, 2.0], [1.0, 2.0])["evaluable"])
        r = stats.paired_ratio([1.0, -2.0, -1.0, 0.5], [1.0, 2.0, 3.0, 4.0])
        self.assertFalse(r["evaluable"])
        self.assertIn("non-positive", r["reason"])

    def test_a_resample_with_a_non_positive_mean_is_reported_not_raised(self):
        x = [-1.0, -1.0, 10.0, 10.0]  # positive mean, but a resample of four -1 has a negative one
        for r in (stats.paired_ratio(x, [1.0, 2.0, 3.0, 4.0]), stats.log_contrast([x, [1.0, 2.0, 3.0, 4.0]], [1, -1])):
            self.assertFalse(r["evaluable"])
            self.assertIn("non-positive mean", r["reason"])

    def test_within_is_closed(self):
        self.assertTrue(stats.within([0.97, 1.03], (0.97, 1.03)))
        self.assertFalse(stats.within([0.9699, 1.03], (0.97, 1.03)))
        self.assertFalse(stats.within([0.97, 1.0301], (0.97, 1.03)))


class RatioOfRatios(unittest.TestCase):
    def test_two_series_reproduce_gate5_paired_exactly(self):
        """The unit test of the pre-registration: the linearized studentized seed bootstrap with two series IS gate5_paired (bootstrap-t)."""
        for seed, n in ((1, 60), (2, 25), (3, 100)):
            y, x = sample(n, seed)
            r = stats.log_contrast([x, y], [1, -1])
            g = gate5_paired(x, y, level=0.90, method="bootstrap-t", B=2000, seed=20261104)
            self.assertEqual(r["theta"], float(g.theta))
            self.assertEqual(r["se"], float(g.se))
            self.assertEqual(r["ci90"], [float(g.ratio_lo), float(g.ratio_hi)])

    def test_four_series_known_structure(self):
        jf, jm = sample(80, 5, 1.0)
        df, dm = sample(80, 6, 1.0)
        r = stats.log_contrast([jf, df, jm, dm], [1, -1, -1, 1])
        want = (jf.mean() / df.mean()) / (jm.mean() / dm.mean())
        self.assertAlmostEqual(r["ratio"], want, places=12)
        e = jf / jf.mean() - df / df.mean() - jm / jm.mean() + dm / dm.mean()
        self.assertAlmostEqual(r["se"], e.std(ddof=1) / math.sqrt(80), places=14)
        self.assertLess(r["ci90"][0], r["ratio"])
        self.assertGreater(r["ci90"][1], r["ratio"])

    def test_equal_series_is_degenerate_at_one(self):
        x, _ = sample(30, 7)
        r = stats.log_contrast([x, x, x, x], [1, -1, -1, 1])
        self.assertEqual((r["ratio"], r["ci90"]), (1.0, [1.0, 1.0]))

    def test_not_evaluable(self):
        self.assertFalse(stats.log_contrast([[1.0, 2.0]] * 4, [1, -1, -1, 1])["evaluable"])


class Unpaired(unittest.TestCase):
    def test_welch_on_the_log_matches_a_direct_computation(self):
        a, b = sample(70, 11, 1.0)
        c, d = sample(50, 12, 1.0)
        r = stats.unpaired_ratio(([a, b], [1, -1]), ([c, d], [1, -1]))
        s1 = np.std(a / a.mean() - b / b.mean(), ddof=1) / math.sqrt(70)
        s2 = np.std(c / c.mean() - d / d.mean(), ddof=1) / math.sqrt(50)
        th = math.log(a.mean() / b.mean()) - math.log(c.mean() / d.mean())
        se = math.hypot(s1, s2)
        nu = se ** 4 / (s1 ** 4 / 69 + s2 ** 4 / 49)
        from scipy import stats as st
        q = st.t.isf(0.05, nu)
        self.assertAlmostEqual(r["theta"], th, places=13)
        self.assertAlmostEqual(r["se"], se, places=14)
        self.assertAlmostEqual(r["nu"], nu, places=9)
        self.assertAlmostEqual(r["ci90"][0], math.exp(th - q * se), places=9)
        self.assertAlmostEqual(r["ci90"][1], math.exp(th + q * se), places=9)
        self.assertEqual(r["n"], [70, 50])

    def test_levels_of_single_series_and_symmetry(self):
        a, _ = sample(40, 13)
        c, _ = sample(55, 14)
        r = stats.unpaired_ratio(([a], [1]), ([c], [1]))
        r2 = stats.unpaired_ratio(([c], [1]), ([a], [1]))
        self.assertAlmostEqual(r["ratio"] * r2["ratio"], 1.0, places=12)
        self.assertAlmostEqual(r["ci90"][0] * r2["ci90"][1], 1.0, places=9)

    def test_degenerate_groups(self):
        r = stats.unpaired_ratio(([np.full(5, 2.0)], [1]), ([np.full(5, 1.0)], [1]))
        self.assertEqual((r["ratio"], r["ci90"]), (2.0, [2.0, 2.0]))
        self.assertTrue(math.isnan(r["nu"]))
        self.assertFalse(stats.unpaired_ratio(([[1.0, 2.0]], [1]), ([[1.0, 2.0, 3.0]], [1]))["evaluable"])


if __name__ == "__main__":
    unittest.main()
