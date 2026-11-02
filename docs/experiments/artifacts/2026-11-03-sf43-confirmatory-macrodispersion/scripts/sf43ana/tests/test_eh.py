import unittest

import numpy as np

import sf43ana  # noqa: F401
from mcstat.budget import LadderBound
from sf43ana.eh import EhBound


def ladder(f, n=24, seed=0, sd=1.0):
    """Per-seed values u_s + f_k: every level difference is exactly f_k - f_j in every resample (zero standard error)."""
    u = np.random.default_rng(seed).normal(0.0, sd, size=n)
    return u[:, None] + np.asarray(f, dtype=float)[None, :]


def power_errors(res, p, c=1.0):
    return c * np.asarray(res, float) ** (-p)


class KnownAnswers(unittest.TestCase):
    def test_finer_branch(self):
        res = [5, 10, 20]
        e = power_errors(res, 2.0, 100.0)  # e(10) = 1
        eh = EhBound(ladder(e), res, 1, B=499, seed=1)
        d = abs(e[2] - e[1])
        self.assertAlmostEqual(float(eh.bound(0.05)), 1.25 * d * 4.0 / 3.0, places=11)
        self.assertAlmostEqual(float(eh.correction(ladder(e).mean(0))), -e[1], places=11)  # alpha* - alpha(h_p) = -e(h_p)

    def test_finest_two_differences_and_cap(self):
        res = [4, 8, 16]
        p = 2.05  # inside [0.5, 2.1], above the cap 2: p_use = 2
        e = power_errors(res, p, 100.0)
        eh = EhBound(ladder(e), res, 2, B=499, seed=1)
        d1, d0 = abs(e[2] - e[1]), abs(e[1] - e[0])
        want = 1.25 * max(d1, d0 / 2.0 ** 2) / (2.0 ** 2 - 1.0)
        self.assertAlmostEqual(float(eh.bound(0.05)), want, places=10)
        self.assertTrue(bool(eh.details(0.05)["order_branch_used"]))
        # correction uses the point order capped at 2: (a(h_p) - a(r h_p)) / (r^2 - 1)
        self.assertAlmostEqual(float(eh.correction(ladder(e).mean(0))), (e[2] - e[1]) / 3.0, places=11)

    def test_second_difference_can_dominate(self):
        res = [4, 8, 16]
        f = np.array([0.0, 1.0, 1.1])  # decelerating: d0 = 1, d1 = 0.1, order log2(10)
        eh = EhBound(ladder(f), res, 2, p_max=4.0, B=499, seed=1)
        d = eh.details(0.05)
        self.assertGreater(d["p_L"], 2.9)  # above the cap 2: p_use = 2
        want = 1.25 * max(0.1, 1.0 / 4.0) / 3.0
        self.assertAlmostEqual(float(eh.bound(0.05)), want, places=10)

    def test_fallback_when_order_out_of_range_or_oscillating(self):
        res = [5, 10, 20]
        for f, why in ((power_errors(res, 3.5, 100.0), "order above the range"), ([0.0, 1.0, 0.5], "non-monotone"), ([0.0, 1.0, 1.0], "zero difference")):
            f = np.asarray(f, float)
            eh = EhBound(ladder(f), res, 1, B=499, seed=2)
            d = eh.details(0.05)
            self.assertFalse(bool(d["order_branch_used"]), why)
            rng_ = f.max() - f.min()
            # exact studentization: se of every pair difference is 0, so U_ij = |m_i - m_j|; fallback = 3 max / (r_min - 1)
            self.assertAlmostEqual(float(eh.bound(0.05)), 3.0 * rng_ / 1.0, places=10, msg=why)

    def test_four_levels_minimum_over_triples(self):
        res = [4, 8, 16, 32]
        # triple (4, 8, 16) has order 1; triple (8, 16, 32) has order 2: p_L = min = 1 (production finest)
        e = np.array([1.0, 0.5, 0.25, 0.1875])  # differences 0.5, 0.25, 0.0625: orders 1 and 2
        f = e  # m_k = alpha* + e_k
        eh = EhBound(ladder(f), res, 3, B=499, seed=1)
        self.assertEqual(eh.triples, [(0, 1, 2), (1, 2, 3)])
        d = eh.details(0.05)
        self.assertAlmostEqual(float(d["p_L"]), 1.0, places=9)
        d1, d0 = abs(f[3] - f[2]), abs(f[2] - f[1])
        want = 1.25 * max(d1, d0 / 2.0) / (2.0 - 1.0)
        self.assertAlmostEqual(float(eh.bound(0.05)), want, places=10)

    def test_k2_has_no_order(self):
        eh = EhBound(ladder([0.0, 1.0]), [10, 20], 1, B=99, seed=1)
        self.assertFalse(bool(eh.details(0.05)["order_branch_used"]))
        self.assertAlmostEqual(float(eh.bound(0.05)), 3.0 * 1.0 / 1.0, places=10)


class AtLeastLadderBound(unittest.TestCase):
    """E_h >= mcstat.LadderBound (same range, same bootstrap draws, method 'student') in every branch."""

    def test_random_ladders(self):
        rng = np.random.default_rng(5)
        res = [5.0, 10.0, 20.0]
        seen = {(ip, br): 0 for ip in (0, 1, 2) for br in (True, False)}
        for trial in range(120):
            p = rng.uniform(0.2, 3.0)
            c = rng.uniform(0.5, 3.0)
            sgn = rng.choice([-1.0, 1.0])
            f = sgn * power_errors(res, p, c * 10.0 ** p)
            if rng.random() < 0.25:
                f = f + rng.normal(0, 0.5 * abs(f).max(), 3)  # may make it non-monotone
            x = ladder(f, n=int(rng.integers(8, 40)), seed=trial, sd=rng.uniform(0.2, 2.0)) + rng.normal(0, 0.05, (1, 3)) * rng.uniform(0, 1)
            x = x + rng.normal(0.0, 0.1, x.shape) * rng.uniform(0.0, 1.0)  # level-specific noise: positive standard errors
            for ip in (0, 1, 2):
                a = EhBound(x, res, ip, p_min=0.5, p_max=2.1, B=299, seed=trial)
                b = LadderBound(x, res, ip, p_min=0.5, p_max=2.1, safety=1.25, fallback_factor=3.0, B=299, seed=trial, method="student")
                for tail in (0.2, 0.05, 0.004):
                    ba, bb = float(a.bound(tail)), float(b.bound(tail))
                    self.assertGreaterEqual(ba, bb * (1 - 1e-12), (trial, ip, tail))
                seen[(ip, bool(a.details(0.05)["order_branch_used"]))] += 1
        for k, v in seen.items():  # every branch (finer ip=0,1; finest ip=2; fallback) was exercised
            self.assertGreater(v, 0, k)

    def test_capped_branch_strictly_larger(self):
        res = [4.0, 8.0, 16.0]
        x = ladder(power_errors(res, 2.05, 100.0))
        a = EhBound(x, res, 1, B=299, seed=1)
        b = LadderBound(x, res, 1, p_max=2.1, B=299, seed=1, method="student")
        self.assertGreater(float(a.bound(0.05)), float(b.bound(0.05)))


if __name__ == "__main__":
    unittest.main()
