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
        self.assertAlmostEqual(float(eh.correction(ladder(e).mean(0), 2.0)), -e[1], places=11)  # alpha* - alpha(h_p) = -e(h_p)

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
        self.assertAlmostEqual(float(eh.correction(ladder(e).mean(0), eh.details(0.05)['p_use'])), (e[2] - e[1]) / 3.0, places=11)

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


def noisy_ladder(law, res, extra_res=None, n=100, seed=0, sd_level=0.003):
    """u_s + law(res_k) + level noise on common seeds; ``extra`` the same for ``extra_res``."""
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, 1.0, n)
    lev = np.stack([u + law(r) + rng.normal(0.0, sd_level, n) for r in res], axis=-1)
    ex = None if extra_res is None else u + law(extra_res) + rng.normal(0.0, sd_level, n)
    return lev, ex


class PerTripleRange(unittest.TestCase):
    def test_every_triple_must_be_in_range_the_upper_limit_acts_per_triple(self):
        res = [4, 8, 16, 32]
        # differences 0.5, 0.25, 0.03125: orders 1 and 3 -> the second triple is above 2.1: fallback although the minimum (1) is in range
        f = np.array([1.0, 0.5, 0.25, 0.21875])
        eh = EhBound(ladder(f), res, 3, B=499, seed=1)
        d = eh.details(0.05)
        self.assertFalse(bool(d["order_branch_used"]))
        self.assertAlmostEqual(float(eh.bound(0.05)), 3.0 * (f.max() - f.min()) / 1.0, places=10)
        # orders 1 and 2: both in range, p_use = the minimum
        f = np.array([1.0, 0.5, 0.25, 0.1875])
        d = EhBound(ladder(f), res, 3, B=499, seed=1).details(0.05)
        self.assertTrue(bool(d["order_branch_used"]))
        self.assertAlmostEqual(float(d["p_use"]), 1.0, places=9)

    def test_fallback_levels_restrict_the_range(self):
        res = [4, 8, 16, 32]
        f = np.array([0.0, 1.0, 0.0, 0.1])  # non-monotone: fallback; the range over all levels is 1.0, over levels 2, 3 only 0.1
        a = EhBound(ladder(f), res, 3, B=199, seed=1)
        b = EhBound(ladder(f), res, 3, B=199, seed=1, fallback_levels=[2, 3])
        self.assertAlmostEqual(float(a.bound(0.05)), 3.0 * 1.0, places=10)
        self.assertAlmostEqual(float(b.bound(0.05)), 3.0 * 0.1, places=10)
        with self.assertRaises(ValueError):
            EhBound(ladder(f), res, 3, B=19, seed=1, fallback_levels=[3])


class AsymptoticRangeGate(unittest.TestCase):
    GATE = [{"type": "extra_level", "triple": [0, 1, 2], "resolution": 12}]

    def bound(self, law, seed, sd_level, gates=GATE):
        lev, ex = noisy_ladder(law, [4, 8, 16], 12, seed=seed, sd_level=sd_level)
        return EhBound(lev, [4, 8, 16], 2, B=1999, seed=1, gates=gates, extra=ex), lev

    def test_passes_on_an_exact_power_law(self):
        eh, _ = self.bound(lambda r: 100.0 * r ** -2.0, 0, 0.3)  # fixed data: the gate is a 90 % test, it fails 10 % of exact laws by design
        d = eh.details(0.01)
        self.assertTrue(bool(d["gate_ok"]))
        self.assertTrue(bool(d["order_branch_used"]))
        self.assertAlmostEqual(float(d["bound"]), float(d["order_branch_value"]))
        self.assertLess(float(d["bound"]), float(d["fallback_value"]) / 10.0)  # the order branch is much smaller than the fallback

    def test_fails_on_a_pre_asymptotic_law_and_takes_the_maximum_with_the_fallback(self):
        law = lambda r: 2.0 * r ** -2.0 + 0.9 * r ** -0.3  # apparent order 0.8 on 4 / 8 / 16 (in range), true error decays like h^0.3
        for seed in range(3):
            eh, lev = self.bound(law, seed, 0.003)
            d = eh.details(0.01)
            self.assertTrue(bool(d["order_branch_used"]), seed)
            self.assertFalse(bool(d["gate_ok"]), seed)
            self.assertAlmostEqual(float(d["bound"]), max(float(d["order_branch_value"]), float(d["fallback_value"])), places=12)
            self.assertGreater(float(d["bound"]), 3.0 * float(d["order_branch_value"]))
            self.assertGreater(float(d["bound"]), law(16))  # covers the true error |a(16) - a_inf| (a_inf = 0)
            self.assertLess(float(d["order_branch_value"]), law(16))  # the ungated bound would NOT cover it

    def test_gate_levels_are_configurable(self):
        law = lambda r: 100.0 * r ** -2.0
        lev, ex = noisy_ladder(law, [4, 8, 16], 12, seed=1, sd_level=0.3)  # seed 1: a 10 %-level false fail
        strict = EhBound(lev, [4, 8, 16], 2, B=1999, seed=1, gates=self.GATE, extra=ex, gate_level=0.90)
        loose = EhBound(lev, [4, 8, 16], 2, B=1999, seed=1, gates=self.GATE, extra=ex, gate_level=0.99)
        self.assertFalse(bool(strict.gate_ok))
        self.assertTrue(bool(loose.gate_ok))
        lev2, ex2 = noisy_ladder(law, [4, 8, 16, 32], 12, seed=1, sd_level=0.3)
        other = EhBound(lev2, [4, 8, 16, 32], 3, B=499, seed=1, gates=[{"type": "extra_level", "triple": [1, 2, 3], "resolution": 12}], extra=ex2)
        self.assertEqual(len(other.gate_info), 1)  # any triple of the ladder can carry the fit
        with self.assertRaises(ValueError):
            EhBound(lev, [4, 8, 16], 2, B=99, seed=1, gates=self.GATE)  # the extra level is missing

    def test_orders_agree_gate_at_the_high_amplitude(self):
        res = [4, 8, 16, 32]
        gate = [{"type": "orders_agree", "triples": [[0, 1, 2], [1, 2, 3]]}]
        ok, _ = noisy_ladder(lambda r: 100.0 * r ** -2.0, res, n=100, seed=0, sd_level=0.3)
        bad, _ = noisy_ladder(lambda r: 2.0 * r ** -2.0 + 0.9 * r ** -0.3, res, n=100, seed=0, sd_level=0.003)
        a = EhBound(ok, res, 3, B=1999, seed=1, gates=gate)
        b = EhBound(bad, res, 3, B=1999, seed=1, gates=gate)
        self.assertTrue(bool(a.gate_ok))
        self.assertFalse(bool(b.gate_ok))
        self.assertTrue(np.isfinite(b.gate_info[0]["diff"]))  # the orders of the two triples exist and differ
        self.assertGreaterEqual(float(b.bound(0.01)), float(b.details(0.01)["fallback_value"]))

    def test_gated_bound_is_at_least_the_plain_ladder_bound_on_random_inputs(self):
        rng = np.random.default_rng(11)
        res = [4.0, 8.0, 16.0]
        fired = 0
        for trial in range(60):
            p, c = rng.uniform(0.3, 2.5), rng.uniform(0.5, 3.0)
            law = lambda r, p=p, c=c: c * 10.0 ** p * r ** -p + (0.5 * c * r ** -0.3 if trial % 2 else 0.0)
            lev, ex = noisy_ladder(law, res, 12, n=int(rng.integers(20, 60)), seed=trial, sd_level=rng.uniform(0.01, 0.4))
            for ip in (1, 2):
                a = EhBound(lev, res, ip, p_min=0.5, p_max=2.1, B=299, seed=trial, gates=self.GATE, extra=ex)
                b = LadderBound(lev, res, ip, p_min=0.5, p_max=2.1, safety=1.25, fallback_factor=3.0, B=299, seed=trial, method="student")
                for tail in (0.2, 0.05, 0.004):
                    self.assertGreaterEqual(float(a.bound(tail)), float(b.bound(tail)) * (1 - 1e-12), (trial, ip, tail))
                fired += int(not bool(a.gate_ok))
        self.assertGreater(fired, 0)  # the gate failed somewhere: the property was exercised with active gates


if __name__ == "__main__":
    unittest.main()
