"""SF-42 N6c: studentized bounds of |Delta| (ContrastBound / LadderBound), bootstrap-t TOST of Gate 5, and the coverage cells that
measure them.  Defaults stay bit-for-bit what N6b shipped (golden values recorded before the change)."""
import json
import math
import os
import unittest

import numpy as np

from mcstat import budget as bd
from mcstat import coverage as cv
from mcstat import covproc, gate5
from mcstat import intervals as iv
from mcstat.tests import _golden
from mcstat.tests._ref import t_ppf

HERE = os.path.dirname(os.path.abspath(__file__))
N6B_PLAN_HASH = "e9cc6c01aaf3eceea57545cc31c8f161330ea218052a9ddaddd2c7544765a1b2"  # plan_hash(profile_n6b(), DEFAULT_SEED) of N6b
N6B_CELLS = 573


def student_q(t, nu):
    return t_ppf(1.0 - t, nu)


def assert_nested_close(tc, a, b, path=""):
    if isinstance(a, dict):
        tc.assertEqual(sorted(a), sorted(b), path)
        for k in a:
            assert_nested_close(tc, a[k], b[k], path + "/" + k)
    else:
        if a and isinstance(a[0], str):
            tc.assertEqual(a, b, path)
        else:
            np.testing.assert_allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=1e-12, atol=0, err_msg=path)


class DefaultsUnchanged(unittest.TestCase):
    def test_default_methods_reproduce_the_n6b_values(self):
        with open(os.path.join(HERE, "golden_default_methods.json")) as f:
            golden = json.load(f)
        assert_nested_close(self, json.loads(json.dumps(_golden.cases())), golden)

    def test_explicit_default_method_is_the_default(self):
        u = np.random.default_rng(1).normal(0.3, 1.0, size=(3, 25))
        a = bd.e_int(u, 0.0 * u, B=499, seed=2).bound([0.01, 0.1])
        b = bd.e_int(u, 0.0 * u, B=499, seed=2, method="percentile").bound([0.01, 0.1])
        np.testing.assert_array_equal(a, b)

    def test_the_n6b_plan_is_untouched(self):
        n6b = cv.profile_n6b()
        self.assertEqual(len(n6b), N6B_CELLS)
        self.assertEqual(cv.plan_hash(n6b, cv.DEFAULT_SEED), N6B_PLAN_HASH)  # ids, parameters, replicate counts, tails
        full = cv.profile_full()
        self.assertEqual(full[:N6B_CELLS], n6b)  # same cells, same order, so the same streams and chunk files
        self.assertGreater(len(full), N6B_CELLS)
        self.assertEqual(len({c["id"] for c in full}), len(full))


class StudentBoundsAnalytic(unittest.TestCase):
    TAILS = [1e-3, 8.9e-4, 0.0125, 0.05]

    def test_validation(self):
        u = np.zeros((2, 10))
        with self.assertRaises(ValueError):
            bd.e_int(u, u, method="percentile-t", seed=1)
        with self.assertRaises(ValueError):
            bd.e_int(u, u, B=100, method="bootstrap-t")  # neither rng nor seed: bootstrap work must be reproducible
        bd.e_int(u + 1.0, u, method="student")  # the Student bound needs no random numbers
        with self.assertRaises(ValueError):
            bd.e_int(u[:, :1], u[:, :1], method="student")  # one seed: no standard error
        with self.assertRaises(ValueError):
            bd.LadderBound(np.zeros((10, 3)), [5.0, 10.0, 20.0], 1, B=99, seed=1, method="t")

    def test_paired_contrast_matches_the_one_sample_formula(self):
        rng = np.random.default_rng(3)
        a16, a8 = rng.normal(0.2, 1.0, size=(4, 30)), rng.normal(0.9, 1.0, size=(4, 30))
        cb = bd.e_int(a16, a8, method="student")  # factor 2, one pairing group
        d = a16 - a8
        se = d.std(axis=1, ddof=1) / math.sqrt(30)
        for t in self.TAILS:
            want = 2.0 * (np.abs(d.mean(axis=1)) + student_q(t, 29) * se)
            np.testing.assert_allclose(cb.bound(t), want, rtol=1e-12)
        self.assertEqual(cb.bound(self.TAILS).shape, (4, 4))
        np.testing.assert_allclose(cb.estimate, 2.0 * np.abs(d.mean(axis=1)), rtol=1e-14)

    def test_unpaired_contrast_matches_welch(self):
        rng = np.random.default_rng(4)
        wide, prod = rng.normal(1.2, 0.5, size=30), rng.normal(1.0, 1.5, size=40)
        cb = bd.e_w(wide, prod, method="student")
        v1, v2 = wide.var(ddof=1) / 30, prod.var(ddof=1) / 40
        nu = (v1 + v2) ** 2 / (v1 ** 2 / 29 + v2 ** 2 / 39)
        for t in self.TAILS:
            want = abs(wide.mean() - prod.mean()) + student_q(t, nu) * math.sqrt(v1 + v2)
            self.assertAlmostEqual(float(cb.bound(t)), want, delta=1e-10 * want)
        self.assertTrue(29 < nu < 68)  # Welch-Satterthwaite lies between the smaller group and the pooled degrees of freedom

    def test_interaction_contrast_has_two_groups_with_their_own_pairing(self):
        rng = np.random.default_rng(5)
        wf, wc, pf, pc = (rng.normal(m, 1.0, size=n) for m, n in ((0.5, 20), (0.1, 20), (0.2, 25), (0.1, 25)))
        cb = bd.e_interaction(wf, wc, pf, pc, method="student")
        uw, up = wf - wc, pc - pf  # signs as in the definition: [wf - wc] - [pf - pc]
        v1, v2 = uw.var(ddof=1) / 20, up.var(ddof=1) / 25
        nu = (v1 + v2) ** 2 / (v1 ** 2 / 19 + v2 ** 2 / 24)
        want = abs(uw.mean() + up.mean()) + student_q(0.01, nu) * math.sqrt(v1 + v2)
        self.assertAlmostEqual(float(cb.bound(0.01)), want, delta=1e-10 * want)

    def test_the_bound_is_max_of_the_two_one_sided_limits_and_covers_both_signs(self):
        rng = np.random.default_rng(6)
        d = rng.normal(0.0, 1.0, size=(50, 20))
        d[::2] *= -1.0  # estimates of both signs
        d += 0.3 * np.where(np.arange(50) % 2 == 0, -1.0, 1.0)[:, None]
        zero = np.zeros_like(d)
        cb = bd.e_int(d, zero, factor=1.0, method="student")
        mean, se = d.mean(axis=1), d.std(axis=1, ddof=1) / math.sqrt(20)
        q = student_q(0.02, 19)
        ucl, lcl = mean + q * se, mean - q * se
        np.testing.assert_allclose(cb.bound(0.02), np.maximum(ucl, -lcl), rtol=1e-12)  # U = max(UCL, -LCL)
        self.assertTrue(np.all(cb.bound(0.02) >= np.abs(mean)))

    def test_constant_difference_has_zero_width_for_every_method(self):
        u = np.random.default_rng(1).normal(size=30)
        for method in ("student", "bootstrap-t"):
            cb = bd.e_int(u + 1.0 - 0.003, u + 1.0, B=299, seed=1, method=method)
            self.assertAlmostEqual(float(cb.bound(0.01)), 0.006, places=12, msg=method)  # factor 2 x 0.003

    def test_smaller_tail_gives_larger_bound_and_batches_are_independent_rows(self):
        rng = np.random.default_rng(7)
        a, b = rng.normal(size=(6, 25)), rng.normal(size=(6, 25))
        for method in ("student", "bootstrap-t"):
            cb = bd.e_w(a, b, B=999, seed=2, method=method)
            self.assertTrue(bd.is_non_increasing_in_tail(cb.bound, [1e-3, 3e-3, 1e-2, 0.05, 0.2]), method)
            row = bd.e_w(a[2], b[2], B=999, seed=2, method=method)
            if method == "student":
                np.testing.assert_allclose(row.bound(0.01), cb.bound(0.01)[2], rtol=1e-12)


class BootstrapTBounds(unittest.TestCase):
    def test_single_group_equals_the_one_sided_bootstrap_t_limits(self):
        u = np.random.default_rng(8).normal(0.4, 1.0, size=(3, 30))
        cb = bd.e_int(u, 0.0 * u, factor=1.0, B=1999, seed=9, method="bootstrap-t")
        bt = iv.bootstrap_t(u, B=1999, seed=9)  # the same seed-resampling stream and the same moments
        for t in (0.005, 0.02, 0.1):
            ucl, lcl = iv.bootstrap_t_limit(bt, t, "upper"), iv.bootstrap_t_limit(bt, t, "lower")
            np.testing.assert_allclose(cb.bound(t), np.maximum(ucl, -lcl), rtol=1e-11)

    def test_resample_statistics_match_a_brute_force_resampling(self):
        rng = np.random.default_rng(10)
        wide, prod = rng.lognormal(0.0, 0.5, size=18), rng.lognormal(0.2, 0.7, size=26)
        B = 400
        cb = bd.e_w(wide, prod, B=B, seed=11, method="bootstrap-t")
        g = np.random.default_rng(11)
        i1, i2 = g.integers(0, 18, size=(B, 18)), g.integers(0, 26, size=(B, 26))  # one index matrix per group, in group order
        u1, u2 = wide[i1], -prod[i2]
        delta_star = u1.mean(axis=1) + u2.mean(axis=1)
        se_star = np.sqrt(u1.var(axis=1, ddof=1) / 18 + u2.var(axis=1, ddof=1) / 26)
        delta = wide.mean() - prod.mean()
        tstar = np.sort((delta_star - delta) / se_star)
        np.testing.assert_allclose(cb._tstar, tstar, rtol=1e-10, atol=1e-12)
        se = math.sqrt(wide.var(ddof=1) / 18 + prod.var(ddof=1) / 26)
        k_lo, k_hi = int(math.floor(0.05 * (B + 1))), int(math.ceil(0.95 * (B + 1)))
        ucl, lcl = delta - se * tstar[k_lo - 1], delta - se * tstar[k_hi - 1]
        self.assertAlmostEqual(float(cb.bound(0.05)), max(ucl, -lcl), delta=1e-10)

    def test_unresolved_tail_is_infinite_never_optimistic(self):
        u = np.random.default_rng(12).normal(0.4, 1.0, size=(2, 30))
        cb = bd.e_int(u, 0.0 * u, B=99, seed=1, method="bootstrap-t")
        self.assertTrue(np.all(np.isinf(cb.bound(0.005))))  # 0.005 < 1 / (B + 1)
        self.assertTrue(np.all(np.isfinite(cb.bound(0.05))))

    def test_close_to_student_on_symmetric_data(self):
        rng = np.random.default_rng(13)
        u = rng.normal(0.3, 1.0, size=(300, 50))
        z = np.zeros_like(u)
        st = bd.e_int(u, z, factor=1.0, method="student")
        bt = bd.e_int(u, z, factor=1.0, B=2000, rng=np.random.default_rng(14), share_draws=False, method="bootstrap-t")
        se = u.std(axis=1, ddof=1) / math.sqrt(50)
        for t in (0.01, 0.05):
            diff = (bt.bound(t) - st.bound(t)) / se
            self.assertLess(abs(float(diff.mean())), 0.05, msg="tail %g" % t)  # no systematic offset
            self.assertLess(float(np.abs(diff).mean()), 0.2, msg="tail %g" % t)  # replicate-wise Monte Carlo + second-order noise


class LadderStudentized(unittest.TestCase):
    RES = [5.0, 10.0, 20.0]

    @staticmethod
    def levels(shift, n=40, R=3, seed=0, noise=0.02):
        rng = np.random.default_rng(seed)
        return rng.normal(0, 1, size=(R, n, 1)) + np.asarray(shift)[None, None, :] + rng.normal(0, noise, size=(R, n, 3))

    def test_order_branch_uses_the_studentized_adjacent_difference(self):
        x = self.levels(0.05 * (np.array(self.RES) / 10.0) ** -2.0, seed=1)
        for ip, (i, j) in ((1, (1, 2)), (2, (1, 2)), (0, (0, 1))):
            lb = bd.LadderBound(x, self.RES, ip, B=999, seed=2, method="student")
            d = lb.details([0.01, 0.05])
            self.assertTrue(np.all(d["order_branch_used"]))
            u = x[..., j] - x[..., i]
            se = u.std(axis=1, ddof=1) / math.sqrt(40)
            for k, t in enumerate((0.01, 0.05)):
                dU = np.abs(u.mean(axis=1)) + student_q(t / 2.0, 39) * se  # each input at tail t / 2
                rp = 2.0 ** d["p_L"][:, k]
                want = 1.25 * dU * rp / (rp - 1.0) if ip < 2 else 1.25 * dU / (rp - 1.0)
                np.testing.assert_allclose(d["order_branch_value"][:, k], want, rtol=1e-10, err_msg="ip=%d" % ip)

    def test_bonferroni_fallback_formula_and_dominance(self):
        x = self.levels(0.05 * np.array([0.0, 1.0, 0.5]), seed=3)  # non-monotone: fallback
        for method in ("student", "bootstrap-t"):
            lb = bd.LadderBound(x, self.RES, 1, B=1999, seed=4, method=method)
            d = lb.details([0.01, 0.05])
            self.assertFalse(np.any(d["order_branch_used"]))
            np.testing.assert_allclose(d["bound"], d["fallback_value"])
        lb = bd.LadderBound(x, self.RES, 1, B=999, seed=4, method="student")
        iu, ju = np.triu_indices(3, 1)
        u = x[..., ju] - x[..., iu]  # (R, n, 3): the three pair differences
        se = u.std(axis=1, ddof=1) / math.sqrt(40)
        for t in (0.01, 0.05):
            pair_t3 = np.abs(u.mean(axis=1)) + student_q(t / 3.0, 39) * se
            pair_t1 = np.abs(u.mean(axis=1)) + student_q(t, 39) * se
            fb = lb.details(t)["fallback_value"]
            np.testing.assert_allclose(fb, 3.0 * pair_t3.max(axis=1) / (2.0 - 1.0), rtol=1e-10)  # fallback_factor * max_ij U_ij(t/P) / (r_min-1)
            self.assertTrue(np.all(fb >= 3.0 * pair_t1.max(axis=1) * (1 - 1e-12)))  # at least the uncorrected pairwise maximum
            self.assertTrue(np.all(fb >= 3.0 * np.abs(u.mean(axis=1)).max(axis=1)))  # and the observed range

    def test_two_levels_one_pair(self):
        x = self.levels([0.0, 0.05, 0.0], seed=5)[..., :2]
        lb = bd.LadderBound(x, [10.0, 20.0], 0, B=499, seed=1, method="student")
        u = x[..., 1] - x[..., 0]
        want = 3.0 * (np.abs(u.mean(axis=1)) + student_q(0.01, 39) * u.std(axis=1, ddof=1) / math.sqrt(40)) / (2.0 - 1.0)  # P = 1: t / P = t
        np.testing.assert_allclose(lb.bound(0.01), want, rtol=1e-10)

    def test_bootstrap_t_pair_statistics_match_a_brute_force_resampling(self):
        x = self.levels(0.05 * np.array([0.0, 1.0, 0.5]), n=24, R=1, seed=6, noise=0.1)[0]  # (n, K)
        B = 300
        lb = bd.LadderBound(x, self.RES, 1, B=B, seed=7, method="bootstrap-t")
        idx = np.random.default_rng(7).integers(0, 24, size=(B, 24))
        iu, ju = np.triu_indices(3, 1)
        u = x[:, ju] - x[:, iu]  # (n, P)
        us = u[idx]  # (B, n, P)
        tstar = (us.mean(axis=1) - u.mean(axis=0)) / (us.std(axis=1, ddof=1) / math.sqrt(24))
        np.testing.assert_allclose(lb._pt, np.sort(tstar, axis=0).T, rtol=1e-9, atol=1e-12)  # sorted per pair
        self.assertEqual(lb._p_sorted.shape, (B,))  # the order bootstrap uses the same resamples
        self.assertTrue(np.all(np.isfinite(lb.bound([0.05, 0.2]))))

    def test_bootstrap_t_ladder_is_infinite_when_b_cannot_resolve_the_tail(self):
        x = self.levels(0.05 * (np.array(self.RES) / 10.0) ** -2.0, R=2, seed=8)
        lb = bd.LadderBound(x, self.RES, 1, B=99, seed=1, method="bootstrap-t")
        self.assertTrue(np.all(np.isinf(lb.bound(0.005))))  # t / 2 = 0.0025 < 1 / (B + 1)
        self.assertTrue(np.all(np.isfinite(lb.bound(0.2))))

    def test_robust_limit_accepts_studentized_components(self):
        rng = np.random.default_rng(9)
        x = rng.normal(2.0, 1.0, size=(3, 80))
        u, v, w = rng.normal(0.1, 0.3, size=(3, 30)), rng.normal(0.0, 0.3, size=(3, 30)), rng.normal(0.0, 1.0, size=(3, 30))
        lev = self.levels(0.01 * (np.array(self.RES) / 10.0) ** -1.0, R=3, seed=10)
        for method in ("student", "bootstrap-t"):
            comps = [bd.LadderBound(lev, self.RES, 1, B=2999, seed=1, method=method), bd.e_int(u, v, B=2999, seed=2, method=method),
                     bd.e_w(u, w, B=2999, seed=3, method=method)]
            rl = bd.RobustLimit(bd.StudentMean(x), comps)
            lower = rl.lower([0.01, 0.05, 0.3])
            self.assertEqual(lower.shape, (3, 3))
            self.assertTrue(np.all(np.diff(lower, axis=-1) >= -1e-12))  # RL(a) non-decreasing in a
            pv = rl.p_value()
            self.assertTrue(np.all((pv.p > 0.0) & (pv.p <= 1.0)))
            rep = rl.report(0.05)
            self.assertEqual(set(rep["components"]), {"E_h", "E_int", "E_W"})


class Gate5BootstrapT(unittest.TestCase):
    @staticmethod
    def data(theta, n, R, cv_=0.3, rho=0.5, seed=0):
        rng = np.random.default_rng(seed)
        A, Bc = math.sqrt((1.0 + rho) / 2.0), math.sqrt((1.0 - rho) / 2.0)
        c, u = rng.standard_normal((R, n)), rng.standard_normal((R, n))
        theta = np.broadcast_to(np.asarray(theta, dtype=float), (R,))
        return np.exp(theta)[:, None] * (1.0 + cv_ * (A * c + Bc * u)), 1.0 + cv_ * (A * c - Bc * u)

    def test_resample_statistics_match_a_brute_force_resampling(self):
        L, P = self.data(0.03, 25, 1, seed=1)
        L, P = L[0], P[0]
        B = 500
        got = gate5._boot_tstar(L, P, B, np.random.default_rng(5), False)
        idx = np.random.default_rng(5).integers(0, 25, size=(B, 25))
        Ls, Ps = L[idx], P[idx]
        ml, mp = Ls.mean(axis=1), Ps.mean(axis=1)
        e = Ls / ml[:, None] - Ps / mp[:, None]
        se_star = e.std(axis=1, ddof=1) / math.sqrt(25)
        theta = math.log(L.mean() / P.mean())
        want = np.sort((np.log(ml / mp) - theta) / se_star)
        np.testing.assert_allclose(got, want, rtol=1e-9, atol=1e-12)

    def test_result_fields_and_reproducibility(self):
        L, P = self.data(0.0, 40, 4, seed=2)
        a = gate5.gate5_paired(L, P, method="bootstrap-t", B=999, seed=3)
        b = gate5.gate5_paired(L, P, method="bootstrap-t", B=999, seed=3)
        c = gate5.gate5_paired(L, P, method="bootstrap-t", B=999, seed=4)
        np.testing.assert_array_equal(a.p_tost, b.p_tost)
        np.testing.assert_array_equal(a.lo, b.lo)
        self.assertFalse(np.array_equal(a.lo, c.lo))
        self.assertIsNone(a.dof)
        self.assertEqual(a.B, 999)
        s = gate5.gate5_paired(L, P)
        np.testing.assert_array_equal(a.theta, s.theta)  # same point estimate and linearized se
        np.testing.assert_array_equal(a.se, s.se)
        self.assertTrue(np.all(a.p_tost >= 1.0 / 1000.0 - 1e-15))  # rank-form p-value: never below 1 / (B + 1)
        one = gate5.gate5_paired(L[0], P[0], method="bootstrap-t", B=999, seed=3)  # a single replicate
        self.assertEqual(np.shape(one.p_tost), ())

    def test_validation(self):
        L, P = self.data(0.0, 20, 2, seed=3)
        with self.assertRaises(ValueError):
            gate5.gate5_paired(L, P, method="percentile")
        with self.assertRaises(ValueError):
            gate5.gate5_paired(L, P, method="bootstrap-t", B=99)  # neither rng nor seed
        with self.assertRaises(ValueError):
            gate5.gate5_paired(L - 5.0 * P.mean(), P, method="bootstrap-t", B=99, seed=1)  # non-positive mean
        L2 = np.ones((1, 20))
        L2[0, 0] = -10.0  # positive mean (0.45) but resamples that draw the outlier twice have a negative mean
        gate5.gate5_paired(L2, P[:1], method="student")  # the Student TOST does not look at resamples
        with self.assertRaises(ValueError):
            gate5.gate5_paired(L2, P[:1], method="bootstrap-t", B=99, seed=1)

    def test_tost_rejection_is_equivalent_to_the_ci_lying_inside_the_margin(self):
        theta = np.linspace(-0.13, 0.13, 120)
        L, P = self.data(theta, 30, 120, cv_=0.2, rho=0.6, seed=4)
        B = 1500
        delta = math.log(gate5.MARGIN_RATIO)
        rejected = False
        for alpha in (0.1, 0.05, 0.025):
            res = gate5.gate5_paired(L, P, level=1.0 - 2.0 * alpha, method="bootstrap-t", B=B, seed=5)
            inside = (res.lo > -delta) & (res.hi < delta)
            np.testing.assert_array_equal(res.p_tost <= alpha, inside, err_msg="alpha=%g" % alpha)
            self.assertTrue(0 < inside.sum() < inside.size)  # both outcomes occur
            rejected |= bool(inside.any())
        self.assertTrue(rejected)

    def test_unresolved_tail_never_agrees(self):
        L, P = self.data(0.0, 30, 3, cv_=0.05, rho=0.9, seed=6)
        res = gate5.gate5_paired(L, P, method="bootstrap-t", B=15, seed=1)  # 0.05 < 1 / 16: both CI ends infinite
        self.assertTrue(np.all(np.isneginf(res.lo)) and np.all(np.isposinf(res.hi)))
        self.assertTrue(np.all(res.verdict == "inconclusive"))
        self.assertTrue(np.all(res.p_tost >= 1.0 / 16.0 - 1e-15))

    def test_agrees_with_the_student_tost_on_large_normal_samples(self):
        L, P = self.data(np.linspace(-0.08, 0.08, 150), 400, 150, cv_=0.2, rho=0.5, seed=7)
        st = gate5.gate5_paired(L, P)
        bt = gate5.gate5_paired(L, P, method="bootstrap-t", B=1999, seed=8)
        np.testing.assert_allclose(bt.lo, st.lo, atol=0.15 * float(st.se.mean()) + 3e-3)  # CI ends: second-order differences + Monte Carlo
        np.testing.assert_allclose(bt.hi, st.hi, atol=0.15 * float(st.se.mean()) + 3e-3)
        self.assertLess(abs(float(((bt.hi - bt.lo) / (st.hi - st.lo)).mean() - 1.0)), 0.03)  # same average width
        agree = (bt.p_tost <= 0.05) == (st.p_tost <= 0.05)
        self.assertGreaterEqual(float(agree.mean()), 0.95)  # same decision except near the boundary

    def test_type_one_error_at_the_boundary_on_normal_data(self):
        cell = cv.make_cell("tost", "t", "", "normal", 60, {"rho": 0.5, "cv": 0.3, "method": "bootstrap-t", "B": 1000}, m=100)
        R = 1500
        rows, _ = cv.run_cell_batches(dict(cell, R=R), 3, 0, R // 100)
        tails = covproc.holm_levels()
        for r in ("ratio_hi", "ratio_lo"):
            misses = rows[r].sum(axis=0)
            for i in (5, 6):  # tails 0.025 and 0.05: 38 and 75 expected misses; second-order accuracy, so a little slack
                self.assertLessEqual(abs(misses[i] / R - tails[i]), 4.5 * math.sqrt(tails[i] * (1 - tails[i]) / R) + 0.004, "%s tail %g" % (r, tails[i]))


class CoverageCells(unittest.TestCase):
    def small(self, **kw):
        base = dict(dists=("normal",), ladder_ns=(20,), component_ns=(20,), gate5_ns=(20,),
                    ladder_cases=(("power", 1.0, 12.0), ("nonmono", 1.0, 12.0)), pipeline_configs=((3, 1, 100), (4, 2, 30)),
                    pipeline_biases=(cv.PIPELINE_BIASES[1],), target=0.1)
        base.update(kw)
        return cv.profile_studentized(**base)

    def test_labels_cover_every_new_procedure_and_method(self):
        cells = cv.profile_studentized()
        labels = {c["label"] for c in cells}
        for kind in ("paired", "unpaired", "interaction"):
            for tag in ("student", "boott"):
                self.assertIn("contrast:%s/%s" % (kind, tag), labels)
        for tag in ("student", "boott"):
            self.assertIn("ladder:E_h/" + tag, labels)
            self.assertIn("pipeline:K3,ip=1/" + tag, labels)
            self.assertIn("pipeline:K4,ip=2/" + tag, labels)
        self.assertIn("tost:gate5/boott", labels)
        by = {(c["label"], c["n"]): c for c in cells}
        self.assertEqual({c["n"] for c in cells if c["label"] == "tost:gate5/boott"}, set(cv.GATE5_NS))
        self.assertEqual({c["scen"] for c in cells if c["label"] == "tost:gate5/boott"}, {"rho=0", "rho=0.5", "rho=0.9"})
        self.assertEqual({c["n"] for c in cells if c["label"] == "contrast:paired/student"}, set(cv.COMPONENT_NS))
        self.assertEqual({c["n"] for c in cells if c["label"] == "ladder:E_h/boott"}, set(cv.LADDER_NS))
        self.assertEqual(by[("contrast:paired/student", 100)]["R"], 196000)  # the same 175 / 8.929e-4 rule
        for c in cells:
            self.assertGreaterEqual(c["R"] * min(c["tails"]), cv.TARGET - 1e-9, c["id"])
            self.assertEqual(c["R"] % c["m"], 0)

    def test_bootstrap_sizes_resolve_the_tails_they_use(self):
        t_c = float(covproc.component_tails().min())
        for c in self.small():
            if c["label"] == "contrast:paired/boott":
                self.assertGreaterEqual(c["params"]["B"], 10.0 / t_c)
            if c["label"] == "ladder:E_h/boott":  # the Bonferroni range needs t / 3, the order limit t / 2
                self.assertGreaterEqual(c["params"]["B"], 10.0 * 3.0 / t_c)
            if c["label"] == "ladder:E_h/student":
                self.assertGreaterEqual(c["params"]["B"], 10.0 * 2.0 / t_c)
            if c["label"] == "tost:gate5/boott":
                self.assertGreaterEqual(c["params"]["B"], 20.0 / float(covproc.holm_levels().min()))

    def test_new_cells_run_and_report_per_tail_rates(self):
        cells = self.small()
        procs = {}
        for c in cells:
            procs.setdefault(c["label"], c)
        self.assertGreaterEqual(len(procs), 9)
        for label, c in sorted(procs.items()):
            c = dict(c, R=c["m"] * 2)
            rows, extra = cv.run_cell_batches(c, 3, 0, 2)
            for name, v in rows.items():
                self.assertEqual(v.shape, (2, len(c["tails"])), "%s %s" % (label, name))
            if c["proc"] in ("contrast", "ladder"):
                self.assertEqual(int(extra["unresolved"].sum()), 0, label)  # B resolves every tail by construction

    def test_engine_end_to_end_on_the_new_cells(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            first = {}
            for c in self.small():
                first.setdefault(c["label"], c)
            cells = [dict(c, R=c["m"] * 2, chunk=c["m"]) for c in first.values()]
            cv.init_plan(tmp, cells, 5, "test")
            cv.run(tmp, 1, log=lambda s: None)
            s = cv.summarize(tmp)
            self.assertEqual(s["cells_complete"], len(cells))
            self.assertEqual(s["incomplete"], [])

    def test_smoke_profile_contains_the_studentized_cells(self):
        labels = {c["label"] for c in cv.profile_smoke()}
        for needed in ("contrast:paired/student", "contrast:unpaired/boott", "ladder:E_h/student", "ladder:E_h/boott",
                       "tost:gate5/boott", "pipeline:K3/student", "pipeline:K3/boott"):
            self.assertIn(needed, labels)

    def test_percentile_contrast_under_covers_and_student_does_not_at_n30(self):
        """The N6b finding that motivated this node, reproduced at small scale (Delta = 5 standard errors, normal data, n = 30,
        tail 0.0125): percentile misses about 1.3-1.6 times the tail, the Student bound the tail itself."""
        R = 12000
        out = {}
        for meth in ("percentile", "student"):
            params = {"kind": "paired", "kappa": 5.0, "method": meth}
            if meth == "percentile":
                params["B"] = 2000
            cell = cv.make_cell("contrast", "c", "", "normal", 30, params, m=10, share=meth == "percentile")
            rows, _ = cv.run_cell_batches(dict(cell, R=R), 3, 0, R // 10)
            tails = np.array(cell["tails"])
            i = int(np.argmin(np.abs(tails - 0.0125)))
            out[meth] = rows["bound"][:, i].sum() / R / tails[i]
        self.assertGreater(out["percentile"], 1.15)
        self.assertLess(abs(out["student"] - 1.0), 4.5 * math.sqrt(1.0 / (R * 0.0125)))


if __name__ == "__main__":
    unittest.main()
