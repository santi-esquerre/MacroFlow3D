import unittest

import numpy as np

from sf39 import decide as D
from sf39 import stats as S

T = 2.0**1.5
UP = "unresolved (grid-limited, lower bound)"
DOWN = "unresolved (grid-limited, upper bound)"


def arr(mean, sd, n=20):
    """Deterministic sample with exactly this mean and (ddof=1) sd."""
    z = np.linspace(-1.0, 1.0, n)
    z = (z - z.mean()) / z.std(ddof=1)
    return mean + sd * z


# Named samples; n = 20 so the 90 % half-width is 0.3866 sd.
ABOVE = lambda: arr(0.010, 0.001)    # CI entirely above 0
BELOW = lambda: arr(-0.010, 0.001)   # CI entirely below 0
ZERO = lambda: (np.arange(20) - 9.5) * 2.0**-8  # CI contains 0, mean exactly 0 (dyadic values)
POS_NS = lambda: arr(0.0005, 0.02)   # mean slightly above 0, CI contains 0
NEG_NS = lambda: arr(-0.0005, 0.02)  # mean slightly below 0, CI contains 0
PROD0 = lambda: arr(0.05, 0.002)     # positive production values


def h(verdict, d_f, d_tw, f_f=None, f_tw=None):
    return {"verdict": verdict, "diffs": {"D_f": d_f, "D_tw": d_tw},
            "finer": {"D_f": f_f if f_f is not None else arr(0.05, 0.002),
                      "D_tw": f_tw if f_tw is not None else arr(0.05, 0.002)}}


def hm(verdict, d, fin=None):
    return {"verdict": verdict, "diffs": {"d16_8": d},
            "finer": {"d16_8": fin if fin is not None else arr(0.05, 0.002)}}


GOOD_H = lambda: h("converged", ZERO(), ZERO())
GOOD_HM = lambda: hm("converged", ZERO())


class TestF4(unittest.TestCase):
    m = 0.005

    def run_f4(self, dc, df, dtw):
        return D.production_h_verdict(dc, df, dtw, self.m)

    def test_converged(self):
        r = self.run_f4(arr(0.010, 0.001), arr(0.0, 0.001), arr(0.0, 0.001))
        self.assertEqual(r["verdict"], "converged")
        self.assertEqual(r["narrow"], "converged")
        self.assertTrue(r["tw_inside"] and r["order_ok"])
        self.assertEqual(r["s"], 1.0)

    def test_narrow_not_converged_dominates(self):
        r = self.run_f4(arr(0.01, 0.001), arr(0.02, 0.001), arr(0.0, 0.001))
        self.assertEqual(r["narrow"], "not converged")
        self.assertEqual(r["verdict"], "not converged")
        # also when the twin is outside and the order is rejected
        r = self.run_f4(arr(0.001, 0.001), arr(0.02, 0.001), arr(0.1, 0.001))
        self.assertEqual(r["verdict"], "not converged")

    def test_narrow_inconclusive(self):
        r = self.run_f4(arr(0.0, 0.05), arr(0.0, 0.05), arr(0.0, 0.001))
        self.assertEqual(r["narrow"], "inconclusive")
        self.assertEqual(r["verdict"], "inconclusive")

    def test_twin_bound_is_2p15_m(self):
        # twin CI half-width 0.3866*sd: mean 0.0141 -> inside 2^1.5 m = 0.014142 only with tiny sd
        inside = self.run_f4(arr(0.01, 0.001), arr(0.0, 0.001), arr(0.0140, 1e-5))
        self.assertTrue(inside["tw_inside"])
        self.assertEqual(inside["verdict"], "converged")
        out = self.run_f4(arr(0.01, 0.001), arr(0.0, 0.001), arr(0.0142, 1e-5))
        self.assertFalse(out["tw_inside"])
        self.assertEqual(out["verdict"], "inconclusive")  # narrow converged but twin not bounded
        wide = self.run_f4(arr(0.01, 0.001), arr(0.0, 0.001), arr(0.0, 0.05))
        self.assertEqual(wide["verdict"], "inconclusive")

    def test_order_contrast_rejects_monotone_order(self):
        # D_c small, D_f large: 2^1.5 D_f - D_c clearly > 0 -> order < 1.5 -> not converged-able
        r = self.run_f4(arr(0.001, 0.0002), arr(0.004, 0.0002), arr(0.0, 0.001))
        self.assertEqual(r["narrow"], "converged")
        self.assertFalse(r["order_ok"])
        self.assertEqual(r["verdict"], "inconclusive")
        # negative D_c: s = -1 flips the contrast so the same geometry is rejected as well
        r = self.run_f4(arr(-0.001, 0.0002), arr(-0.004, 0.0002), arr(0.0, 0.001))
        self.assertEqual(r["s"], -1.0)
        self.assertFalse(r["order_ok"])
        # D_c large, D_f small (order >= 1.5): accepted
        r = self.run_f4(arr(0.02, 0.0002), arr(0.002, 0.0002), arr(0.0, 0.001))
        self.assertTrue(r["order_ok"])
        # opposite-signed D_f with s = -1 and D_c < 0: 2^1.5 D_f - D_c < 0 times s -> accepted
        r = self.run_f4(arr(-0.02, 0.0002), arr(-0.002, 0.0002), arr(0.0, 0.001))
        self.assertTrue(r["order_ok"])

    def test_order_contrast_is_the_ci_of_a_contrast_not_a_ratio(self):
        # noisy but not rejected: contrast CI contains 0
        r = self.run_f4(arr(0.003, 0.02), arr(0.001, 0.0002), arr(0.0, 0.001))
        self.assertTrue(r["order_ok"])
        self.assertLessEqual(r["order_ci"]["lo"], 0)

    def test_zero_mean_Dc_uses_plus_sign(self):
        r = self.run_f4(np.zeros(20), arr(0.0, 0.001), arr(0.0, 0.001))
        self.assertEqual(r["s"], 1.0)

    def test_pairing_required(self):
        with self.assertRaises(ValueError):
            self.run_f4(np.zeros(20), np.zeros(19), np.zeros(20))

    def test_twin_length_may_differ(self):
        r = self.run_f4(arr(0.01, 0.001), arr(0.0, 0.001), arr(0.0, 0.001, 30))
        self.assertEqual(r["tw_ci"]["n"], 30)

    def test_hmax_verdict(self):
        self.assertEqual(D.hmax_verdict(arr(0.0, 0.001), 0.005)["verdict"], "converged")
        self.assertEqual(D.hmax_verdict(arr(0.02, 0.001), 0.005)["verdict"], "not converged")
        self.assertEqual(D.hmax_verdict(arr(0.0, 0.05), 0.005)["verdict"], "inconclusive")


class TestRuleDBranches(unittest.TestCase):
    def label(self, ladders, prod=None):
        return D.rule_d(ladders, prod if prod is not None else PROD0())

    # ---- branch 1 ----
    def test_branch1_h_ladder_Df_above(self):
        r = self.label({"h": h("inconclusive", ABOVE(), POS_NS()), "hmax": GOOD_HM()})
        self.assertEqual((r["label"], r["branch"], r["ladder"], r["difference"]), (UP, 1, "h", "D_f"))

    def test_branch1_Dtw_above_and_not_converged_verdict(self):
        r = self.label({"h": h("not converged", ZERO(), ABOVE()), "hmax": GOOD_HM()})
        self.assertEqual((r["label"], r["difference"]), (UP, "D_tw"))

    def test_branch1_hmax_ladder(self):
        r = self.label({"h": GOOD_H(), "hmax": hm("inconclusive", ABOVE())})
        self.assertEqual((r["label"], r["ladder"], r["difference"]), (UP, "hmax", "d16_8"))

    def test_branch1_blocked_by_other_point_estimate_below_zero(self):
        # other's point estimate below 0 (CI contains 0): branch 1 does not fire, nothing below-0 for 3
        r = self.label({"h": h("inconclusive", ABOVE(), NEG_NS()), "hmax": GOOD_HM()})
        self.assertEqual((r["label"], r["branch"]), ("unresolved", 5))

    def test_branch1_not_blocked_by_other_exactly_zero(self):
        r = self.label({"h": h("inconclusive", ABOVE(), ZERO()), "hmax": GOOD_HM()})
        self.assertEqual(r["branch"], 1)

    def test_branch1_requires_verdict_not_converged(self):
        # converged verdict with an 'above' difference does not fire branch 1 (goes to branch 4)
        r = self.label({"h": h("converged", ABOVE(), POS_NS()), "hmax": GOOD_HM()})
        self.assertEqual(r["branch"], 4)

    def test_branch1_requires_CI_entirely_above(self):
        r = self.label({"h": h("inconclusive", arr(0.002, 0.02), POS_NS()), "hmax": GOOD_HM()})
        self.assertEqual(r["branch"], 5)

    # ---- branch 2 ----
    def test_branch2_finer_ci_contains_zero(self):
        r = self.label({"h": h("not converged", BELOW(), NEG_NS(), f_f=arr(0.0, 0.02)), "hmax": GOOD_HM()})
        self.assertEqual((r["label"], r["branch"], r["ladder"], r["difference"]), ("numerical", 2, "h", "D_f"))

    def test_branch2_finer_upper_limit_small(self):
        # finer-setting CI excludes 0 (positive) but its upper limit <= 2.5e-3
        fin = arr(0.0015, 0.0001)
        r = self.label({"h": h("not converged", BELOW(), ZERO(), f_f=fin), "hmax": GOOD_HM()})
        self.assertEqual(r["label"], "numerical")

    def test_branch2_finer_upper_limit_edge(self):
        fin = arr(0.0026, 1e-6)  # lo, hi both above 2.5e-3 and above 0: does not vanish -> branch 3
        r = self.label({"h": h("not converged", BELOW(), ZERO(), f_f=fin), "hmax": GOOD_HM()})
        self.assertEqual((r["label"], r["branch"]), (DOWN, 3))
        fin = arr(0.00249, 1e-6)
        r = self.label({"h": h("not converged", BELOW(), ZERO(), f_f=fin), "hmax": GOOD_HM()})
        self.assertEqual(r["label"], "numerical")

    def test_branch2_finer_clearly_positive_falls_to_branch3(self):
        r = self.label({"h": h("not converged", BELOW(), NEG_NS(), f_f=arr(0.05, 0.001)), "hmax": GOOD_HM()})
        self.assertEqual((r["label"], r["branch"]), (DOWN, 3))

    def test_branch2_requires_verdict_exactly_not_converged(self):
        r = self.label({"h": h("inconclusive", BELOW(), NEG_NS(), f_f=arr(0.0, 0.02)), "hmax": GOOD_HM()})
        self.assertEqual((r["label"], r["branch"]), (DOWN, 3))

    def test_branch2_blocked_by_other_point_estimate_above_zero(self):
        r = self.label({"h": h("not converged", BELOW(), POS_NS(), f_f=arr(0.0, 0.02)), "hmax": GOOD_HM()})
        self.assertEqual((r["label"], r["branch"]), (DOWN, 3))

    def test_branch2_existential_over_differences(self):
        # D_f qualifies but its finer CI does not vanish; D_tw also below and its finer CI contains 0
        r = self.label({"h": h("not converged", BELOW(), BELOW(), f_f=arr(0.05, 0.001), f_tw=arr(0.0, 0.02)),
                        "hmax": GOOD_HM()})
        self.assertEqual((r["label"], r["branch"], r["difference"]), ("numerical", 2, "D_tw"))

    def test_branch2_hmax_ladder(self):
        r = self.label({"h": GOOD_H(), "hmax": hm("not converged", BELOW(), arr(0.0, 0.02))})
        self.assertEqual((r["label"], r["ladder"], r["difference"]), ("numerical", "hmax", "d16_8"))

    # ---- branch 3 ----
    def test_branch3_has_no_other_estimate_clause(self):
        # D_f above and D_tw below (inconclusive verdict): branch 1 blocked by D_tw < 0, branch 3 fires on D_tw
        r = self.label({"h": h("inconclusive", ABOVE(), BELOW()), "hmax": GOOD_HM()})
        self.assertEqual((r["label"], r["branch"], r["difference"]), (DOWN, 3, "D_tw"))
        # D_tw below with D_f above-zero point estimate and verdict not converged: branch 2 blocked, branch 3 fires
        r = self.label({"h": h("not converged", arr(0.003, 0.02), BELOW(), f_tw=arr(0.0, 0.02)), "hmax": GOOD_HM()})
        self.assertEqual((r["label"], r["branch"]), (DOWN, 3))

    def test_branch3_hmax(self):
        r = self.label({"h": GOOD_H(), "hmax": hm("inconclusive", BELOW())})
        self.assertEqual((r["label"], r["branch"], r["ladder"]), (DOWN, 3, "hmax"))

    def test_branch3_requires_CI_entirely_below(self):
        r = self.label({"h": h("inconclusive", NEG_NS(), NEG_NS()), "hmax": GOOD_HM()})
        self.assertEqual(r["branch"], 5)

    # ---- branch 4 ----
    def test_branch4_positive(self):
        r = self.label({"h": GOOD_H(), "hmax": GOOD_HM()}, arr(0.05, 0.002))
        self.assertEqual((r["label"], r["branch"]), ("physical (positive)", 4))

    def test_branch4_negative(self):
        r = self.label({"h": GOOD_H(), "hmax": GOOD_HM()}, arr(-0.05, 0.002))
        self.assertEqual(r["label"], "physical (negative)")

    def test_branch4_bounded_reports_U(self):
        prod = arr(0.0, 0.002)
        r = self.label({"h": GOOD_H(), "hmax": GOOD_HM()}, prod)
        self.assertEqual(r["label"], "physical (bounded)")
        self.assertAlmostEqual(r["U"], r["prod_ci"]["hi"])
        self.assertLessEqual(r["U"], 2.5e-3)

    def test_branch4_imprecise(self):
        r = self.label({"h": GOOD_H(), "hmax": GOOD_HM()}, arr(0.0, 0.05))
        self.assertEqual((r["label"], r["branch"]), ("unresolved (imprecise)", 4))

    def test_branch4_upper_exactly_at_cut_is_bounded(self):
        prod = arr(0.0, 0.002)
        _, lo, hi, _, _ = S.t_interval(prod)
        prod = prod * (2.5e-3 / hi)  # CI upper limit scaled to the cut (within rounding)
        r = self.label({"h": GOOD_H(), "hmax": GOOD_HM()}, prod - 1e-12)
        self.assertEqual(r["label"], "physical (bounded)")

    # ---- branch 5 ----
    def test_branch5_one_verdict_inconclusive_no_direction(self):
        r = self.label({"h": h("inconclusive", ZERO(), ZERO()), "hmax": GOOD_HM()})
        self.assertEqual((r["label"], r["branch"]), ("unresolved", 5))
        r = self.label({"h": GOOD_H(), "hmax": hm("inconclusive", ZERO())})
        self.assertEqual(r["label"], "unresolved")

    def test_branch5_not_converged_but_no_direction(self):
        r = self.label({"h": h("not converged", ZERO(), ZERO()), "hmax": GOOD_HM()})
        self.assertEqual(r["label"], "unresolved")


class TestRuleDPrecedenceAndBinding(unittest.TestCase):
    def label(self, ladders):
        return D.rule_d(ladders, PROD0())

    def test_branch1_beats_branch2_across_ladders(self):
        r = self.label({"h": h("inconclusive", ABOVE(), POS_NS()),
                        "hmax": hm("not converged", BELOW(), arr(0.0, 0.02))})
        self.assertEqual(r["branch"], 1)
        r = self.label({"h": h("not converged", BELOW(), NEG_NS(), f_f=arr(0.0, 0.02)),
                        "hmax": hm("inconclusive", ABOVE())})
        self.assertEqual(r["branch"], 1)  # ladder order does not matter

    def test_branch2_beats_branch3_across_ladders(self):
        r = self.label({"h": h("inconclusive", BELOW(), NEG_NS()),
                        "hmax": hm("not converged", BELOW(), arr(0.0, 0.02))})
        self.assertEqual((r["label"], r["branch"], r["ladder"]), ("numerical", 2, "hmax"))

    def test_branch3_beats_branch4_when_a_ladder_is_not_converged(self):
        r = self.label({"h": GOOD_H(), "hmax": hm("inconclusive", BELOW())})
        self.assertEqual(r["branch"], 3)

    def test_below_hmax_difference_does_not_bind_to_inconclusive_h_verdict(self):
        # h verdict inconclusive with no direction; hmax verdict converged but its difference is below 0:
        # per-ladder binding -> no branch 3 (hmax is converged); h has no below-0 difference.
        r = self.label({"h": h("inconclusive", ZERO(), ZERO()), "hmax": hm("converged", BELOW())})
        self.assertEqual((r["label"], r["branch"]), ("unresolved", 5))

    def test_above_hmax_difference_does_not_bind_to_inconclusive_h_verdict(self):
        r = self.label({"h": h("inconclusive", ZERO(), ZERO()), "hmax": hm("converged", ABOVE())})
        self.assertEqual((r["label"], r["branch"]), ("unresolved", 5))

    def test_below_h_difference_does_not_bind_to_hmax_verdict(self):
        # h verdict converged with a below-0 D_tw, hmax inconclusive without direction -> unresolved
        r = self.label({"h": h("converged", ZERO(), BELOW()), "hmax": hm("inconclusive", ZERO())})
        self.assertEqual((r["label"], r["branch"]), ("unresolved", 5))

    def test_branch1_other_difference_is_same_ladder_only(self):
        # hmax diff below 0 (point estimate) must not block branch 1 of the h ladder
        r = self.label({"h": h("inconclusive", ABOVE(), POS_NS()), "hmax": hm("converged", BELOW())})
        self.assertEqual((r["label"], r["branch"], r["ladder"]), (UP, 1, "h"))

    def test_trace_records_firing_entry(self):
        r = self.label({"h": h("inconclusive", ABOVE(), POS_NS()), "hmax": GOOD_HM()})
        fired = [t for t in r["trace"] if t.get("holds")]
        self.assertEqual(len(fired), 1)
        self.assertEqual((fired[0]["branch"], fired[0]["ladder"], fired[0]["difference"]), (1, "h", "D_f"))
        r = self.label({"h": GOOD_H(), "hmax": GOOD_HM()})
        self.assertTrue(any(t["branch"] == 4 and t["holds"] for t in r["trace"]))

    def test_degenerate_short_samples_never_fire_direction_branches(self):
        short = {"verdict": "inconclusive", "diffs": {"D_f": np.array([1.0]), "D_tw": np.array([1.0])},
                 "finer": {"D_f": np.array([1.0]), "D_tw": np.array([1.0])}}
        r = D.rule_d({"h": short, "hmax": GOOD_HM()}, PROD0())
        self.assertEqual(r["label"], "unresolved")


class TestAmplitude(unittest.TestCase):
    def test_label_agreement(self):
        self.assertEqual(D.amplitude_label("numerical", "numerical"), "numerical")
        self.assertEqual(D.amplitude_label("physical (positive)", "physical (positive)"), "physical (positive)")

    def test_discordant(self):
        d = "unresolved (estimator-discordant)"
        self.assertEqual(D.amplitude_label("physical (positive)", "physical (bounded)"), d)
        self.assertEqual(D.amplitude_label("physical (positive)", "unresolved"), d)
        self.assertEqual(D.amplitude_label("numerical", UP), d)
        self.assertEqual(D.amplitude_label(UP, DOWN), d)

    def test_unresolved_pair_is_common_label(self):
        self.assertEqual(D.amplitude_label("unresolved", "unresolved"), "unresolved")


class TestGate5(unittest.TestCase):
    def test_agree_disagree_inconclusive(self):
        P = arr(0.30, 0.01, 40)
        r = D.gate5_verdict(arr(0.301, 0.01, 40), P)
        self.assertEqual(r["primary"]["verdict"], "agree")
        self.assertAlmostEqual(r["primary"]["margin"], 0.03, 12)  # fixed number 0.10 mean(Par2)
        self.assertEqual(D.gate5_verdict(arr(0.40, 0.01, 40), P)["primary"]["verdict"], "disagree")
        self.assertEqual(D.gate5_verdict(arr(0.31, 0.10, 20), arr(0.30, 0.10, 20))["primary"]["verdict"],
                         "inconclusive")

    def test_matches_welch_and_margin(self):
        L, P = arr(0.32, 0.02, 25), arr(0.30, 0.03, 35)
        r = D.gate5_verdict(L, P)["primary"]
        d, lo, hi, nu, _ = S.welch_interval(L, P)
        self.assertEqual((r["lo"], r["hi"], r["nu"]), (lo, hi, nu))

    def test_sensitivities(self):
        rng = np.random.default_rng(11)
        s2 = 1.0 + rng.normal(0, 0.08, 30)
        P = 0.30 + 0.3 * (s2 - 1.0) + rng.normal(0, 0.01, 30)
        L = arr(0.30, 0.01, 30)
        r = D.gate5_verdict(L, P, L_alpha_fv=arr(0.42, 0.01, 30), par2_sigma2=s2, sigma2_nominal=1.0)
        self.assertEqual(r["primary"]["verdict"], "agree")
        self.assertEqual(r["fv_qbar"]["verdict"], "disagree")
        anc = S.ancova_adjusted(P, s2, 1.0)
        self.assertAlmostEqual(r["ancova"]["margin"], 0.1 * anc.a, 12)
        self.assertAlmostEqual(r["ancova"]["diff"], L.mean() - anc.a, 12)
        self.assertIn(r["ancova"]["verdict"], ("agree", "inconclusive", "disagree"))
        self.assertNotIn("fv_qbar", D.gate5_verdict(L, P))
        self.assertNotIn("ancova", D.gate5_verdict(L, P))


class TestRows(unittest.TestCase):
    def test_rows(self):
        rows = D.decision_table_rows([
            {"amplitude": 1, "estimator": "alpha_TP", "decision_point": "Gate 4", "verdict": "numerical",
             "ci": (-0.1, 0.2), "n": 20},
            {"amplitude": 4, "estimator": "alpha_TAM", "decision_point": "A narrow", "verdict": "inconclusive",
             "ci_lo": -1.0, "ci_hi": 1.0, "n": 20, "note": "x"},
            {"amplitude": 4, "estimator": "-", "decision_point": "RR3", "verdict": "pass"}])
        self.assertEqual(rows[0]["ci_lo"], -0.1)
        self.assertEqual(rows[1]["ci_hi"], 1.0)
        self.assertEqual(rows[1]["note"], "x")
        self.assertIsNone(rows[2]["ci_lo"])
        self.assertEqual(set(rows[0]), {"amplitude", "estimator", "decision_point", "verdict", "ci_lo", "ci_hi",
                                        "n", "note"})


if __name__ == "__main__":
    unittest.main()
