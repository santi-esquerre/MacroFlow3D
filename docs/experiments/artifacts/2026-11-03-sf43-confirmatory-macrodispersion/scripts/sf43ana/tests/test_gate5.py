import contextlib
import io
import math
import unittest

import numpy as np
from scipy import stats

from sf43cov import mcpath  # noqa: F401
from mcstat import intervals as iv

from sf43ana import gate5 as g5

M = 1.10


def paired(rng, n, ratio, cv_L=0.15, cv_P=0.2, rho=0.5):
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    zl, zp = z1, rho * z1 + math.sqrt(1 - rho ** 2) * z2
    return ratio * (1 + cv_L * zl), 1 + cv_P * zp


class HallStatistic(unittest.TestCase):
    def test_against_a_direct_formula(self):
        x = np.array([0.3, -0.8, 1.9, 0.1, -0.2, 2.5, -1.1, 0.7, 0.0, 4.0, -0.5, 0.2])
        n = len(x)
        mean = sum(x) / n
        sd = math.sqrt(sum((v - mean) ** 2 for v in x) / (n - 1))
        m3 = sum((v - mean) ** 3 for v in x) / n
        gamma = m3 / sd ** 3
        T = mean / (sd / math.sqrt(n))
        c = gamma / (3 * math.sqrt(n))
        h = T + c * T ** 2 + c ** 2 * T ** 3 / 3 + gamma / (6 * math.sqrt(n))
        t = g5._studentized(x)[0]
        self.assertAlmostEqual(float(t), T, places=13)
        self.assertAlmostEqual(float(g5.sample_skewness(x)), gamma, places=13)
        self.assertAlmostEqual(float(g5.hall_statistic(t, g5.sample_skewness(x), n)), h, places=13)
        ph, pl = g5.onesample_p(x, "hall")
        self.assertAlmostEqual(float(ph), float(stats.t.cdf(h, n - 1)), places=13)
        self.assertAlmostEqual(float(pl), float(stats.t.sf(h, n - 1)), places=13)
        # the cubic form ((1 + cT)^3 - 1) / (3c) + gamma / (6 sqrt n)
        self.assertAlmostEqual(h, ((1 + c * T) ** 3 - 1) / (3 * c) + gamma / (6 * math.sqrt(n)), places=12)

    def test_reduces_to_student_without_skewness_and_is_monotone(self):
        t = np.linspace(-6, 6, 241)
        self.assertTrue(np.allclose(g5.hall_statistic(t, 0.0, 100), t))
        for gamma in (-2.5, -0.5, 0.7, 3.0):
            h = g5.hall_statistic(t, gamma, 50)
            self.assertTrue(np.all(np.diff(h) > 0), gamma)  # strictly increasing: monotonically invertible

    def test_vectorized_over_replicates(self):
        x = np.random.default_rng(3).standard_normal((5, 40))
        ph, pl = g5.onesample_p(x, "hall")
        for i in range(5):
            p1, p2 = g5.onesample_p(x[i], "hall")
            self.assertAlmostEqual(float(ph[i]), float(p1), places=14)
            self.assertAlmostEqual(float(pl[i]), float(p2), places=14)
        self.assertTrue(np.allclose(ph + pl, 1.0))


class FiellerDecisions(unittest.TestCase):
    def test_boundary_ratios_give_p_one_half_for_student(self):
        rng = np.random.default_rng(1)
        _, P = paired(rng, 60, 1.0)
        for ratio, side in ((M, 0), (1.0 / M, 1)):
            L = ratio * P * (1 + 0.1 * rng.standard_normal(60))
            L = L - (L.mean() - ratio * P.mean())  # mean(L) / mean(P) is exactly the margin
            F = g5.Fieller(L, P)
            self.assertAlmostEqual(F.theta_hat, ratio, places=12)
            ph, pl = F.p("student", ratio)
            self.assertAlmostEqual((ph, pl)[side], 0.5, places=10)
            ph2, pl2 = F.p("hall", ratio)
            self.assertTrue(0.3 < (ph2, pl2)[side] < 0.7)

    def test_decisions(self):
        rng = np.random.default_rng(2)
        _, P = paired(rng, 150, 1.0)
        tiny = 1 + 0.01 * rng.standard_normal(150)
        for ratio, want, tost_small in ((1.00, "agree", True), (1.30, "disagree", False), (0.75, "disagree", False)):
            L = ratio * P * tiny
            F = g5.Fieller(L, P)
            for proc in ("student", "hall"):
                ph, pl, pt = g5.p_tost(F, proc)
                self.assertEqual(pt <= 0.05 / 8, tost_small, (ratio, proc, pt))
                lo, hi = F.limits(proc, 0.05)
                self.assertEqual(g5.verdict(lo, hi), want, (ratio, proc, lo, hi))
        L = 1.10 * P * tiny  # on the margin: not equivalent, not outside
        F = g5.Fieller(L, P)
        lo, hi = F.limits("student", 0.05)
        self.assertEqual(g5.verdict(lo, hi), "inconclusive")
        self.assertGreater(g5.p_tost(F, "student")[2], 0.4)

    def test_student_limits_are_the_classical_fieller_roots(self):
        rng = np.random.default_rng(5)
        L, P = paired(rng, 40, 1.03)
        n, a, b = len(L), L.mean(), P.mean()
        sll, slp, spp = (np.cov(L, P, ddof=1)[0, 0], np.cov(L, P, ddof=1)[0, 1], np.cov(L, P, ddof=1)[1, 1])
        q = stats.t.isf(0.05, n - 1)
        A, Bq, C = n * b * b - q * q * spp, -2 * n * a * b + 2 * q * q * slp, n * a * a - q * q * sll
        roots = sorted(np.roots([A, Bq, C]).real)
        lo, hi = g5.Fieller(L, P).limits("student", 0.05)
        self.assertAlmostEqual(lo, roots[0], places=8)
        self.assertAlmostEqual(hi, roots[1], places=8)

    def test_tost_p_is_consistent_with_the_interval(self):
        rng = np.random.default_rng(6)
        for ratio in (0.98, 1.04, 1.07):
            L, P = paired(rng, 80, ratio, 0.08, 0.1)
            F = g5.Fieller(L, P)
            for proc in ("student", "hall"):
                pt = g5.p_tost(F, proc)[2]
                for a in (0.05, 0.05 / 8):
                    lo, hi = F.limits(proc, a)
                    self.assertEqual(pt <= a, lo > 1 / M and hi < M, (ratio, proc, a, pt, lo, hi))

    def test_inversion_monotone_and_limits_hit_the_tail(self):
        rng = np.random.default_rng(8)
        x = rng.lognormal(0.0, 0.8, 90)
        P = 1 + 0.2 * rng.standard_normal(90)
        L = 1.02 * P * (x / x.mean())
        F = g5.Fieller(L, P)
        grid = np.linspace(0.85, 1.25, 161)
        for proc in ("student", "hall"):
            ph = np.array([F.p(proc, th)[0] for th in grid])
            pl = np.array([F.p(proc, th)[1] for th in grid])
            self.assertTrue(np.all(np.diff(ph) <= 1e-12), proc)  # ratio >= theta is rejected more strongly for larger theta
            self.assertTrue(np.all(np.diff(pl) >= -1e-12), proc)
            lo, hi = F.limits(proc, 0.05)
            self.assertAlmostEqual(F.p(proc, hi)[0], 0.05, places=8)
            self.assertAlmostEqual(F.p(proc, lo)[1], 0.05, places=8)
            self.assertLess(lo, F.theta_hat)
            self.assertGreater(hi, F.theta_hat)

    def test_unbounded_when_the_denominator_is_not_resolved(self):
        L = np.array([1.0, 1.2, 0.9, 1.1, 1.05])
        P = np.array([0.001, -0.2, 0.15, -0.1, 0.16]) + 0.0001  # mean(P) barely positive and noisy
        lo, hi = g5.Fieller(L, P).limits("student", 0.05)
        self.assertEqual(hi, math.inf)


class Bootstrap(unittest.TestCase):
    def test_unsorted_tstar_is_the_sf42_bootstrap_t(self):
        x = np.random.default_rng(1).lognormal(size=(3, 30))
        ts = g5.boot_tstar(x, 500, np.random.default_rng(9), share_draws=True)
        bt = iv.bootstrap_t(x, 500, np.random.default_rng(9), share_draws=True)
        self.assertTrue(np.allclose(np.sort(ts, axis=-1), bt.tstar, rtol=1e-12, atol=1e-12))

    def test_rank_p_is_the_rank_rule_of_the_sf42_limits(self):
        rng = np.random.default_rng(4)
        for i in range(20):
            x = 0.3 + rng.lognormal(size=25) - 1.0
            B, tail = 1999, 0.05
            bt = iv.bootstrap_t(x, B, np.random.default_rng(i), share_draws=True)
            ph, pl = g5.onesample_p(x, "boott", B=B, rng=np.random.default_rng(i))
            # p_hi <= tail  iff  the upper bootstrap-t limit of the mean is below 0 (reject H0: mean >= 0)
            self.assertEqual(float(ph) <= tail, bool(iv.bootstrap_t_limit(bt, tail, "upper") < 0), i)
            self.assertEqual(float(pl) <= tail, bool(iv.bootstrap_t_limit(bt, tail, "lower") > 0), i)

    def test_fieller_bootstrap_equals_the_one_sample_test_on_d(self):
        rng = np.random.default_rng(11)
        L, P = paired(rng, 70, 1.05)
        for theta in (M, 1 / M, 1.0):
            F = g5.Fieller(L, P, B=999, rng=np.random.default_rng(21))
            got = F.p("boott", theta)
            want = g5.onesample_p(L - theta * P, "boott", B=999, rng=np.random.default_rng(21))
            self.assertEqual(got, (float(want[0]), float(want[1])))
            half = F.p("boott_half", theta)
            self.assertEqual(half, (min(1.0, 2 * got[0]), min(1.0, 2 * got[1])))

    def test_bootstrap_decisions_and_limits(self):
        rng = np.random.default_rng(12)
        _, P = paired(rng, 120, 1.0)
        L = 1.0 * P * (1 + 0.02 * rng.standard_normal(120))
        F = g5.Fieller(L, P, B=33599, rng=np.random.default_rng(1))
        ph, pl, pt = g5.p_tost(F, "boott")
        self.assertAlmostEqual(pt, 1 / 33600)  # the smallest attainable p-value: the rank granularity
        self.assertAlmostEqual(g5.p_tost(F, "boott_half")[2], 2 / 33600)
        lo, hi = F.limits("boott", 0.05)
        self.assertEqual(g5.verdict(lo, hi), "agree")
        self.assertTrue(0.95 < lo < 1.0 < hi < 1.05)

    def test_bootstrap_needs_a_stream(self):
        L, P = paired(np.random.default_rng(1), 20, 1.0)
        with self.assertRaises(ValueError):
            g5.Fieller(L, P).p("boott", 1.0)


def fake_summary(rows_ok, ns=(100, 150, 200), m=8, fwer=0.05, drop=None, override=None):
    """A coverage summary in the mcstat format: every (n, dist) cell has rows <proc>/<side> with entries at the 16 tails.

    ``rows_ok[proc][n]`` is True/False for the whole procedure; ``override`` = {(proc, n, dist, side, tail_index): entry fields}."""
    main = g5.holm_levels(m, fwer)
    tails = sorted(set(main) | set(main / 2))
    cells = []
    for n in ns:
        for d in g5.DISTS:
            if drop == (n, d):
                continue
            rows = {}
            for proc in ("student", "hall", "boott"):
                for side in ("hi", "lo"):
                    ok = rows_ok[proc][n]
                    rows["%s/%s" % (proc, side)] = [{"tail": t, "pass": ok, "ucl_over_tail": 1.1 if ok else 1.6} for t in tails]
            cells.append({"label": "fieller", "dist": d, "n": n, "rows": rows, "report_only": False})
    for key, fields in (override or {}).items():
        proc, n, d, side, tv = key  # tv: the tail value
        ent = next(c for c in cells if c["n"] == n and c["dist"] == d)["rows"]["%s/%s" % (proc, side)]
        next(e for e in ent if abs(e["tail"] - tv) < 1e-12).update(fields)
    return {"cells": cells, "plan_hash": "fake"}


class Selection(unittest.TestCase):
    def ok(self, hall, boott, ns=(100, 150, 200)):
        return {"student": {n: False for n in ns}, "hall": dict(zip(ns, hall)), "boott": dict(zip(ns, boott))}

    def test_first_qualified_in_the_preregistered_order(self):
        s = fake_summary(self.ok((False, True, True), (True, True, True)))
        self.assertEqual(g5.select_procedure(s, 150)["procedure"], "hall")
        sel = g5.select_procedure(s, 120)  # largest grid n <= 120 is 100: hall fails there, boott qualifies
        self.assertEqual((sel["grid_n"], sel["procedure"]), (100, "boott"))
        self.assertEqual(g5.select_procedure(s, 1000)["grid_n"], 200)
        self.assertIsNone(g5.select_procedure(s, 99)["procedure"])  # below the smallest grid size: nothing is qualified

    def test_half_level_is_third_and_uses_the_half_tails(self):
        # the Holm levels 0.05 / j and their halves coincide for j = 1, 2, 3, 4 (0.025, 0.0125, 0.00833, 0.00625); 0.05 is not a half level
        ov = {("boott", n, "normal", "hi", 0.05): {"pass": False, "ucl_over_tail": 1.4} for n in (100, 150, 200)}
        s = fake_summary(self.ok((False,) * 3, (True,) * 3), override=ov)
        sel = g5.select_procedure(s, 150)
        self.assertEqual(sel["procedure"], "boott_half")
        self.assertEqual(sel["table"], {"hall": False, "boott": False, "boott_half": True})
        # a failure at a half level that is not a Holm level (0.003125) leaves plain boott qualified and boott_half not
        ov = {("boott", 150, "stress", "lo", 0.003125): {"pass": False, "ucl_over_tail": 1.4}}
        sel = g5.select_procedure(fake_summary(self.ok((False,) * 3, (True,) * 3), override=ov), 150)
        self.assertEqual(sel["procedure"], "boott")
        self.assertFalse(sel["table"]["boott_half"])

    def test_a_single_failing_tail_dist_or_side_disqualifies(self):
        base = self.ok((True, True, True), (True, True, True))
        for key in (("hall", 150, "stress-", "lo", 0.05 / 8), ("hall", 150, "normal", "hi", 0.05), ("hall", 150, "stress", "hi", 0.0125)):
            s = fake_summary(base, override={key: {"pass": False, "ucl_over_tail": 1.3}})
            sel = g5.select_procedure(s, 150)
            self.assertEqual(sel["procedure"], "boott", key)
            self.assertTrue(sel["problems"]["hall"], key)
        # a failure at a tail hall does not use (a half level that is not a Holm level) does not matter to it
        s = fake_summary(base, override={("hall", 150, "stress", "hi", 0.003125): {"pass": False, "ucl_over_tail": 1.3}})
        self.assertEqual(g5.select_procedure(s, 150)["procedure"], "hall")

    def test_unjudged_missing_and_incomplete_do_not_qualify(self):
        base = self.ok((True, True, True), (True, True, True))
        s = fake_summary(base, override={("hall", 150, "normal", "hi", 0.05 / 3): {"pass": None}})
        self.assertEqual(g5.select_procedure(s, 150)["procedure"], "boott")  # an unjudged tail is not a pass
        s = fake_summary(base, drop=(150, "stress"))  # a planned distribution is missing for n = 150
        sel = g5.select_procedure(s, 150)
        self.assertIsNone(sel["procedure"])
        self.assertTrue(any("missing" in p for p in sel["problems"]["hall"]))
        s = fake_summary(base, m=6)  # simulated for the 6-family: the Holm level 0.05 / 7 of the 8-family is not there
        self.assertIsNone(g5.select_procedure(s, 150, m=8)["procedure"])
        self.assertEqual(g5.select_procedure(s, 150, m=6)["procedure"], "hall")


def arms_for(n=150, ratio=1.0, seed0=1000):
    rng = np.random.default_rng(31)
    L, P = paired(rng, n, ratio, 0.1, 0.12)
    seeds = np.arange(seed0, seed0 + n)
    return {"arms": {"L": {"seed": seeds, "cols": {"alpha_LP": L}}, "P": {"seed": seeds.copy(), "cols": {"alpha_LP": P}}}}


class Run(unittest.TestCase):
    prereg = {"gate5": {"family_m": 8, "B": 3999, "seed": 7, "floor": 100, "amplitudes": {"s1": {"consumer": "L", "par2": "P"}}}}

    def test_selected_procedure_p_ci_and_descriptives(self):
        data = arms_for()
        data["coverage_summary"] = fake_summary({"student": {n: False for n in (100, 150, 200)}, "hall": {100: True, 150: True, 200: True},
                                                 "boott": {100: True, 150: True, 200: True}})
        r = g5.run(self.prereg, data)
        a = r["amplitudes"]["s1"]
        self.assertEqual(a["procedure"], "hall")
        self.assertEqual(a["n"], 150)
        self.assertEqual(r["pvalues"], {"s1": a["p_tost"]})
        self.assertEqual(a["p_tost"], a["p_by_procedure"]["hall"]["p_tost"])
        self.assertLess(a["p_tost"], 1e-6)
        self.assertEqual(a["verdict"], "agree")
        self.assertEqual(a["delta_method"]["verdict"], "agree")
        self.assertEqual(a["student_fieller"]["verdict"], "agree")
        self.assertGreater(a["power"]["delta_method"]["ratio_1"], 0.99)
        self.assertGreater(a["power"]["fieller"]["ratio_1"], 0.99)
        self.assertEqual(a["unmatched"]["verdict"], "agree")
        self.assertTrue(0.95 < a["ci"]["lo"] < a["ratio"] < a["ci"]["hi"] < 1.06)

    def test_descriptive_when_nothing_is_qualified(self):
        data = arms_for()
        data["coverage_summary"] = fake_summary({"student": {n: False for n in (100, 150, 200)}, "hall": {n: False for n in (100, 150, 200)},
                                                 "boott": {n: False for n in (100, 150, 200)}})
        a = g5.run(self.prereg, data)["amplitudes"]["s1"]
        self.assertIsNone(a["procedure"])
        self.assertEqual(a["p_tost"], 1.0)
        self.assertTrue(a["ci"]["descriptive"])
        self.assertIn("descriptive", a["note"])
        self.assertLess(a["p_by_procedure"]["student"]["p_tost"], 1e-6)  # still reported

    def test_pairs_are_the_prefix_in_arm_order_and_stop_at_a_nonfinite_pair(self):
        data = arms_for(n=120)
        data["arms"]["P"]["cols"]["alpha_LP"][110] = np.nan
        data["arms"]["P"]["seed"] = data["arms"]["P"]["seed"][:115]
        data["arms"]["P"]["cols"]["alpha_LP"] = data["arms"]["P"]["cols"]["alpha_LP"][:115]
        r = g5.run(self.prereg, dict(data, coverage_summary=None))["amplitudes"]["s1"]
        self.assertEqual(r["n"], 110)  # the first 110 positions; the NaN at position 110 ends the prefix, later pairs are not used
        self.assertTrue(r["pairing"]["stopped_at_nonfinite"])
        self.assertIsNone(r["procedure"])  # no coverage summary: descriptive
        data["arms"]["P"]["seed"] = data["arms"]["P"]["seed"][::-1].copy()
        errors = []
        with contextlib.redirect_stderr(io.StringIO()) as err:  # seed orders differ: not a paired prefix -> the hypothesis is untestable (p = 1)
            a = g5.run(self.prereg, dict(data, coverage_summary=None), errors)["amplitudes"]["s1"]
        self.assertEqual((a["status"], a["p_tost"]), ("error", 1.0))
        self.assertEqual([(e["id"], e["type"]) for e in errors], [("gate5:s1", "ValueError")])
        self.assertIn("HYPOTHESIS_ERROR gate5:s1: ValueError", err.getvalue())
        self.assertIn("seed orders", errors[0]["message"])
        self.assertIn("Traceback", errors[0]["traceback"])

    def test_floor_makes_gate5_untestable(self):
        data = arms_for(n=150)
        data["coverage_summary"] = fake_summary({"student": {n: False for n in (100, 150, 200)}, "hall": {100: True, 150: True, 200: True},
                                                 "boott": {100: True, 150: True, 200: True}})
        pr = {"gate5": dict(self.prereg["gate5"], floor=200)}
        a = g5.run(pr, data)["amplitudes"]["s1"]
        self.assertEqual((a["procedure"], a["p_tost"]), (None, 1.0))
        self.assertIn("floor 200", a["note"])
        self.assertLess(a["p_by_procedure"]["hall"]["p_tost"], 1e-6)  # the estimates are still reported (descriptive)
        pr = {"gate5": dict(self.prereg["gate5"], floor=200, amplitudes={"s1": {"consumer": "L", "par2": "P", "floor": 150}})}
        self.assertEqual(g5.run(pr, data)["amplitudes"]["s1"]["procedure"], "hall")  # per-amplitude floor

    def test_j3b_family_gates_only_where_configured(self):
        data = arms_for(n=150)
        base = {"student": {n: False for n in (100, 150, 200)}, "hall": {n: True for n in (100, 150, 200)}, "boott": {n: True for n in (100, 150, 200)}}
        summ = fake_summary(base)
        j3b = [dict(c, dist=d) for c in summ["cells"] if c["n"] == 150 and c["dist"] == "normal" for d in g5.J3B_DISTS]
        for c in j3b:
            c["rows"] = {k: [dict(e, pass_=None) for e in v] for k, v in c["rows"].items()}
            for v in c["rows"].values():
                for e in v:
                    e.update(**{"pass": False, "ucl_over_tail": 1.9})
        summ["cells"] += j3b  # the J3B-bound family fails at n = 150
        data["coverage_summary"] = summ
        plain = g5.run(self.prereg, data)["amplitudes"]["s1"]
        self.assertEqual(plain["procedure"], "hall")  # not gating where not configured
        gated = g5.run({"gate5": dict(self.prereg["gate5"], amplitudes={"s1": {"consumer": "L", "par2": "P", "j3b": True}})}, data)["amplitudes"]["s1"]
        self.assertIsNone(gated["procedure"])
        self.assertEqual(gated["p_tost"], 1.0)
        self.assertIn("mix:2.73:7.59", " ".join(gated["selection"]["problems"]["hall"]))

    def test_fieller_power_and_moments_and_exclusion_sensitivity(self):
        data = arms_for(n=150)
        data["coverage_summary"] = fake_summary({"student": {n: False for n in (100, 150, 200)}, "hall": {n: True for n in (100, 150, 200)},
                                                 "boott": {n: True for n in (100, 150, 200)}})
        verd = np.array(["converged"] * 150, dtype=object)
        verd[[3, 40]] = "fallback"
        data["arms"]["L"]["verdict"] = verd
        data["arms"]["P"]["verdict"] = np.array(["converged"] * 150, dtype=object)
        data["arms"]["pilot_L"] = {"seed": np.arange(50), "cols": {"alpha_LP": 1.0 + 0.1 * np.sin(np.arange(50))}}
        data["arms"]["pilot_P"] = {"seed": np.arange(50), "cols": {"alpha_LP": 1.0 + 0.1 * np.cos(np.arange(50))}}
        pr = {"gate5": dict(self.prereg["gate5"], amplitudes={"s1": {"consumer": "L", "par2": "P", "pilot": {"consumer": "pilot_L", "par2": "pilot_P"}}})}
        a = g5.run(pr, data)["amplitudes"]["s1"]
        self.assertEqual(a["pair_exclusion_sensitivity"]["n_excluded"], 2)
        self.assertEqual(a["pair_exclusion_sensitivity"]["n"], 148)
        self.assertEqual(a["pair_exclusion_sensitivity"]["procedure"], "hall")
        self.assertEqual(a["pilot"]["n"], 50)
        self.assertEqual(sorted(a["pilot"]["moments"]), ["corr_hi_lo", "hi", "lo"])
        self.assertEqual(sorted(a["d_moments"]["hi"]), ["ekurt", "mean", "sd", "skew"])
        # Fieller-form power: ~1 at ratio 1 for this tight data; low at a ratio beyond the margin; matches a direct simulation of the Student procedure
        pw = a["power"]["fieller"]
        self.assertGreater(pw["ratio_1"], 0.99)
        L, P = data["arms"]["L"]["cols"]["alpha_LP"], data["arms"]["P"]["cols"]["alpha_LP"]
        far = g5.fieller_power(L, P, 0.05 / 8, ratios=(1.10, 1.14))
        self.assertLess(far["ratio_1.1"], 0.06)  # at the margin: at most about the level
        self.assertLess(far["ratio_1.14"], far["ratio_1.1"] + 1e-9)

    def test_fieller_power_agrees_with_simulation(self):
        rng = np.random.default_rng(3)
        n, cvl, cvp, rho, r_true, alpha = 80, 0.30, 0.15, 0.6, 1.04, 0.05 / 8
        z1, z2 = rng.standard_normal((2, 20000, n))
        L = r_true * (1 + cvl * z1)
        P = 1 + cvp * (rho * z1 + math.sqrt(1 - rho ** 2) * z2)
        Dh, Dl = L - M * P, L - P / M
        t = lambda D: D.mean(1) / (D.std(1, ddof=1) / math.sqrt(n))  # noqa: E731
        c = stats.t.isf(alpha, n - 1)
        sim = float(np.mean((t(Dh) < -c) & (t(Dl) > c)))  # the Student Fieller TOST at the true ratio
        big = np.random.default_rng(4).standard_normal((2, 400000))
        Lb = r_true * (1 + cvl * big[0])
        Pb = 1 + cvp * (rho * big[0] + math.sqrt(1 - rho ** 2) * big[1])
        formula = g5.fieller_power(Lb, Pb, alpha, ratios=(r_true,), n=n)["ratio_%g" % r_true]  # population moments, n = 80
        self.assertAlmostEqual(sim, formula, delta=0.03)
        self.assertTrue(0.05 < sim < 0.95)

    def test_power_and_ratio_of_means_not_mean_of_ratios(self):
        data = arms_for(ratio=1.0)
        a = g5.run(self.prereg, dict(data, coverage_summary=None))["amplitudes"]["s1"]
        L = data["arms"]["L"]["cols"]["alpha_LP"]
        P = data["arms"]["P"]["cols"]["alpha_LP"]
        self.assertAlmostEqual(a["ratio"], L.mean() / P.mean(), places=12)


if __name__ == "__main__":
    unittest.main()
