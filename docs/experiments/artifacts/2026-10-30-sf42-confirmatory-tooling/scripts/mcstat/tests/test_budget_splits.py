"""Known-answer tests of the tail splits of decision G5.4 (audit finding AUD-F-4), by brute force.

* ``LadderBound``: the order lower limit ``p_L`` and the adjacent-difference bound ``dU`` are taken at tail ``t / 2``, the
  fallback range bound at the full tail ``t`` (percentile) or ``t / P`` (studentized).  The reference below re-draws the SAME
  resample indices (``rng.integers(0, n, (B, n))``, the one draw of ``_util.boot_counts``) and evaluates every resample in an
  explicit Python loop with its own rank rule.
* ``RobustLimit``: ``RL(a) = L(a / 2) - sum_k B_k(a / (2 K))`` for K = 3 and 4 against the explicit formula built from scipy.

Every test also asserts that the wrong split (tail ``t`` instead of ``t / 2``, ``a`` instead of ``a / 2``) gives a visibly different
number on the same data, so the test cannot pass by insensitivity.
"""
import math
import unittest

import numpy as np
from scipy import stats

from mcstat import budget as bd

RES = [5.0, 10.0, 20.0]
R = 2.0
EPS = 1e-9


def _rank_lower(vals, tau):  # lower tau-quantile: order statistic floor(tau (B + 1)); -inf when unresolved
    k = math.floor(tau * (len(vals) + 1) + EPS)
    return -math.inf if k < 1 else sorted(vals)[k - 1]


def _rank_upper(vals, tau):  # upper tau-quantile: order statistic ceil((1 - tau)(B + 1)); +inf when unresolved
    k = math.ceil((1.0 - tau) * (len(vals) + 1) - EPS)
    return math.inf if k > len(vals) else sorted(vals)[k - 1]


def _order(m):  # three levels, ratio 2: log(|d0| / |d1|) / ln r for a monotone ladder, else no order (-inf: outside the range)
    d0, d1 = m[1] - m[0], m[2] - m[1]
    if d0 * d1 <= 0.0:
        return -math.inf
    return math.log(abs(d0) / abs(d1)) / math.log(R)


def brute_ladder(x, ip, t, split, method, B, seed, safety=1.25, factor=3.0, p_min=0.5, p_max=4.0):
    """Reference E_h for a 3-level ladder; ``split`` is the tail divisor of p_L and dU (2 in the specification)."""
    n, K = x.shape
    idx = np.random.default_rng(seed).integers(0, n, size=(B, n))
    mstar = [x[idx[b]].mean(axis=0) for b in range(B)]
    j = ip + 1 if ip < K - 1 else ip - 1
    pL = _rank_lower([_order(m) for m in mstar], t / split)
    pairs = [(a, b) for a in range(K) for b in range(a + 1, K)]  # i < j, as the library orders them

    def U(i, k, tail):  # upper bound of |m_k - m_i| at tail `tail`
        u = x[:, k] - x[:, i]
        d, se = u.mean(), u.std(ddof=1) / math.sqrt(n)
        if method == "percentile":
            return _rank_upper([abs(m[k] - m[i]) for m in mstar], tail)
        if method == "student":
            return abs(d) + stats.t.isf(tail, n - 1) * se
        ts = []  # bootstrap-t
        for b in range(B):
            ub = u[idx[b]]
            ts.append((ub.mean() - d) / math.sqrt(ub.var(ddof=1) / n))
        ucl, lcl = d - se * _rank_lower(ts, tail), d - se * _rank_upper(ts, tail)
        return max(ucl, -lcl)

    in_range = p_min <= pL <= p_max
    r = R ** (pL if in_range else p_min)
    if in_range:
        dU = U(min(ip, j), max(ip, j), t / split)
        return pL, True, (safety * dU * r / (r - 1.0) if ip < K - 1 else safety * dU / (r - 1.0))
    if method == "percentile":
        rng_u = _rank_upper([float(m.max() - m.min()) for m in mstar], t)
    else:
        rng_u = max(U(a, b, t / len(pairs)) for a, b in pairs)
    return pL, False, factor * rng_u / (R - 1.0)


def noisy_ladder(f, noise, seed, n=30):
    g = np.random.default_rng(seed)
    return g.normal(0.0, 1.0, size=(n, 1)) + np.asarray(f, float)[None, :] + g.normal(0.0, noise, size=(n, len(f)))


ORDER_LADDER = noisy_ladder(0.01 * (np.array(RES) / 10.0) ** -2.0, noise=0.004, seed=3)  # true order 2, noisy
NO_ORDER_LADDER = noisy_ladder([0.0, 0.01, 0.005], noise=0.004, seed=4)  # non-monotone: fallback branch


class LadderSplits(unittest.TestCase):
    B = 199

    def check(self, x, ip, method, t, want_order_branch):
        lb = bd.LadderBound(x, RES, ip=ip, B=self.B, seed=11, method=method)
        pL, branch, want = brute_ladder(x, ip, t, 2.0, method, self.B, 11)
        got = lb.details(t)
        self.assertEqual(bool(got["order_branch_used"]), branch)
        self.assertEqual(branch, want_order_branch)
        np.testing.assert_allclose(float(lb.bound(t)), want, rtol=1e-8, err_msg="%s ip=%d t=%g" % (method, ip, t))
        if branch:
            np.testing.assert_allclose(float(got["p_L"]), pL, rtol=1e-12)  # log-slope, two summation orders
            _, _, wrong = brute_ladder(x, ip, t, 1.0, method, self.B, 11)  # tails t instead of t / 2
            self.assertGreater(abs(wrong - want), 1e-3 * abs(want))  # the test discriminates the split
        return pL, want

    def test_order_branch_all_methods_and_production_levels(self):
        for method in ("percentile", "student", "bootstrap-t"):
            for ip in (0, 1, 2):
                for t in (0.05, 0.2):
                    self.check(ORDER_LADDER, ip, method, t, True)

    def test_p_L_is_the_half_tail_rank(self):
        lb = bd.LadderBound(ORDER_LADDER, RES, ip=1, B=self.B, seed=11)
        a = float(lb.details(0.05)["p_L"])
        b = brute_ladder(ORDER_LADDER, 1, 0.05, 1.0, "percentile", self.B, 11)[0]  # rank floor(0.05 * 200) = 10
        self.assertNotEqual(a, b)
        np.testing.assert_allclose(a, brute_ladder(ORDER_LADDER, 1, 0.05, 2.0, "percentile", self.B, 11)[0], rtol=1e-12)  # rank floor(0.025 * 200) = 5

    def test_fallback_range_uses_the_full_tail_or_the_pair_split(self):
        for method in ("percentile", "student", "bootstrap-t"):
            self.check(NO_ORDER_LADDER, 1, method, 0.1, False)
        # the fallback bound is NOT halved: it differs from a bound taken at t / 2
        lb = bd.LadderBound(NO_ORDER_LADDER, RES, ip=1, B=self.B, seed=11)
        self.assertGreater(abs(float(lb.bound(0.1)) - float(lb.bound(0.05))), 1e-6)


class RobustLimitSplits(unittest.TestCase):
    def build(self, K):
        g = np.random.default_rng(100 + K)
        n = 24
        prod = g.normal(1.0, 0.3, size=n)
        a, b, c, d = (g.normal(m, 0.2, size=n) for m in (0.5, 0.45, 0.4, 0.38))
        comps = [bd.e_int(a, b, method="student"), bd.e_w(c, d, method="student"),
                 bd.e_interaction(a, b, c, d, method="student"),
                 bd.ContrastBound("E_x", [(b, "g", 1.0), (d, "h", -1.0)], 1.5, method="student")][:K]
        # reference data of each component: (independent groups of per-seed values, factor)
        refs = [([a - b], 2.0), ([c, -d], 1.0), ([a - b, -(c - d)], 1.0), ([b, -d], 1.5)][:K]
        return prod, comps, refs

    @staticmethod
    def welch_bound(groups, factor, tail):  # factor * (|delta| + t_{1-tail, nu} se), Welch-Satterthwaite, written out
        n = [len(u) for u in groups]
        v = [u.var(ddof=1) / m for u, m in zip(groups, n)]
        se = math.sqrt(sum(v))
        nu = sum(v) ** 2 / sum(vi * vi / (m - 1) for vi, m in zip(v, n))
        return factor * (abs(sum(u.mean() for u in groups)) + stats.t.isf(tail, nu) * se)

    def test_RL_is_L_at_a_over_2_minus_the_bounds_at_a_over_2K(self):
        for K in (3, 4):
            prod, comps, refs = self.build(K)
            rl = bd.RobustLimit(bd.StudentMean(prod), comps)
            n, mean, se = prod.size, prod.mean(), prod.std(ddof=1) / math.sqrt(prod.size)
            for a in (0.01, 0.05, 0.2, 0.9):
                L = mean - stats.t.isf(a / 2.0, n - 1) * se
                Bs = [self.welch_bound(g, f, a / (2.0 * K)) for g, f in refs]
                want = L - sum(Bs)
                np.testing.assert_allclose(float(rl.lower(a)), want, rtol=1e-8, atol=1e-10, err_msg="K=%d a=%g" % (K, a))
                wrong_L = mean - stats.t.isf(a, n - 1) * se  # sampling limit at a instead of a / 2
                wrong_B = sum(self.welch_bound(g, f, a / K) for g, f in refs)  # tail a / K instead of a / (2K)
                self.assertGreater(abs(wrong_L - L), 1e-3 * abs(L))
                self.assertGreater(abs(wrong_B - sum(Bs)), 1e-3 * sum(Bs))

    def test_report_exposes_the_same_split(self):
        prod, comps, refs = self.build(3)
        a = 0.05
        rep = bd.RobustLimit(bd.StudentMean(prod), comps).report(a)
        want_L = prod.mean() - stats.t.isf(a / 2.0, prod.size - 1) * prod.std(ddof=1) / math.sqrt(prod.size)
        np.testing.assert_allclose(float(np.ravel(rep["L_a_over_2"])[0]), want_L, rtol=1e-8)
        for entry, (g, f) in zip(rep["components"].values(), refs):
            np.testing.assert_allclose(float(np.ravel(entry["raw"])[0]), self.welch_bound(g, f, a / 6.0), rtol=1e-8)


if __name__ == "__main__":
    unittest.main()
