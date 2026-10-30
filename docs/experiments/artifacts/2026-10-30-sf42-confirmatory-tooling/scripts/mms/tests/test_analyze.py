"""Known-answer tests of the MMS order computation and of the acceptance-D verdict logic (synthetic errors)."""
import math
import unittest

from mms import analyze as an

NORM_KEYS = ["head", "flux_interior", "flux_far", "flux_dirichlet", "flux_first_layer"]


def run(ny, eps, errs, rtol=1e-10, conv=True):
    """errs: dict norm-key -> abs error (used for l2_abs and max_abs)."""
    d = {"schema": "sf42-mms-dirichlet-1", "nx": 4 * ny, "ny": ny, "eps": eps, "mg_levels": 3, "pcg_rtol": rtol,
         "pcg_iterations": 10, "pcg_converged": conv, "residual_rel": 1e-11}
    for k in NORM_KEYS:
        e = errs[k]
        d[k] = {"l2_abs": e, "max_abs": e, "l2_rel": e, "max_rel": e}
    return d


def ladder(eps, order_by_key, nys=(16, 32, 64, 128), c=1.0, rtol=1e-10):
    runs = []
    for ny in nys:
        runs.append(run(ny, eps, {k: c * (16.0 / ny) ** order_by_key[k] for k in NORM_KEYS}, rtol))
    return runs


GOOD = {"head": 2.0, "flux_interior": 2.0, "flux_far": 2.0, "flux_dirichlet": 1.0, "flux_first_layer": 1.0}


class OrderTests(unittest.TestCase):
    def test_exact_orders(self):
        self.assertAlmostEqual(an.observed_order(1.0, 0.25, 16, 32), 2.0, places=12)
        self.assertAlmostEqual(an.observed_order(1.0, 0.5, 16, 32), 1.0, places=12)
        self.assertAlmostEqual(an.observed_order(1.0, 1.0 / 8.0, 16, 32), 3.0, places=12)

    def test_non_unit_ratio(self):
        # ny 16 -> 48 (ratio 3), error ratio 9 -> order 2
        self.assertAlmostEqual(an.observed_order(9.0, 1.0, 16, 48), 2.0, places=12)

    def test_degenerate_errors_are_nan(self):
        self.assertTrue(math.isnan(an.observed_order(0.0, 1.0, 16, 32)))
        self.assertTrue(math.isnan(an.observed_order(1.0, float("inf"), 16, 32)))

    def test_orders_series(self):
        ps = an.orders([(16, 1.0), (32, 0.25), (64, 0.25 / 8)])
        self.assertEqual([(a, b) for a, b, _ in ps], [(16, 32), (32, 64)])
        self.assertAlmostEqual(ps[0][2], 2.0, places=12)
        self.assertAlmostEqual(ps[1][2], 3.0, places=12)

    def test_gate_uses_two_finest_pairs_only(self):
        # first pair order 0.5 (pre-asymptotic) must not matter; last two pairs 2.0
        s = [(16, 1.0), (32, 2 ** -0.5), (64, 2 ** -2.5), (128, 2 ** -4.5)]
        g = an.gate_metric(s, 1.8)
        self.assertTrue(g["pass"])
        self.assertAlmostEqual(g["last_two"][0], 2.0, places=12)
        # one of the last two below threshold -> fail
        s2 = [(16, 1.0), (32, 0.25), (64, 0.25 * 2 ** -1.5), (128, 0.25 * 2 ** -3.5)]
        self.assertFalse(an.gate_metric(s2, 1.8)["pass"])

    def test_gate_threshold_is_inclusive(self):
        s = [(16, 1.0), (32, 2 ** -1.8), (64, 2 ** -3.6)]
        self.assertTrue(an.gate_metric(s, 1.8 - 1e-12)["pass"])
        self.assertFalse(an.gate_metric(s, 1.8 + 1e-9)["pass"])

    def test_gate_needs_three_grids(self):
        self.assertIsNone(an.gate_metric([(16, 1.0), (32, 0.25)], 1.8)["pass"])


class VerdictTests(unittest.TestCase):
    EPS = (0.5, 1.0, 1.5, 2.0)

    def runs(self, **override):
        out = []
        for eps in self.EPS:
            out += ladder(eps, GOOD)
        return out

    def test_pass(self):
        v = an.verdict(self.runs(), self.EPS)
        self.assertEqual(v["verdict"], "PASS")
        self.assertEqual(v["problems"], [])

    def test_dirichlet_order_one_does_not_affect_verdict(self):
        v = an.verdict(self.runs(), self.EPS)
        m = v["per_eps"]["1.0"]["metrics"]["Dirichlet faces L2"]
        self.assertFalse(m["gated"])
        self.assertAlmostEqual(m["last_two"][0], 1.0, places=12)
        self.assertEqual(v["verdict"], "PASS")

    def test_fail_when_one_eps_and_one_gated_metric_is_first_order(self):
        runs = []
        for eps in self.EPS:
            order = dict(GOOD)
            if eps == 1.5:
                order["flux_far"] = 1.5
            runs += ladder(eps, order)
        v = an.verdict(runs, self.EPS)
        self.assertEqual(v["verdict"], "FAIL")
        self.assertFalse(v["per_eps"]["1.5"]["metrics"]["flux max (>=2 cells from Dirichlet faces)"]["pass"])
        self.assertTrue(v["per_eps"]["1.0"]["metrics"]["flux max (>=2 cells from Dirichlet faces)"]["pass"])

    def test_incomplete_with_too_few_grids(self):
        runs = []
        for eps in self.EPS:
            runs += ladder(eps, GOOD, nys=(16, 32, 64))
        self.assertEqual(an.verdict(runs, self.EPS)["verdict"], "INCOMPLETE")
        self.assertEqual(an.verdict(runs, self.EPS, min_grids=3)["verdict"], "PASS")

    def test_incomplete_with_missing_eps(self):
        runs = []
        for eps in self.EPS[:3]:
            runs += ladder(eps, GOOD)
        v = an.verdict(runs, self.EPS)
        self.assertEqual(v["verdict"], "INCOMPLETE")
        self.assertTrue(any("eps=2" in p for p in v["problems"]))

    def test_incomplete_when_a_solve_did_not_converge(self):
        runs = self.runs()
        runs[3]["pcg_converged"] = False
        v = an.verdict(runs, self.EPS)
        self.assertEqual(v["verdict"], "INCOMPLETE")

    def test_fail_not_masked_by_second_order_elsewhere(self):
        # head max only first order at one eps: FAIL even if all else passes
        runs = []
        for eps in self.EPS:
            order = dict(GOOD)
            if eps == 0.5:
                order["head"] = 1.0
            runs += ladder(eps, order)
        self.assertEqual(an.verdict(runs, self.EPS)["verdict"], "FAIL")

    def test_tight_rtol_runs_are_not_mixed_into_the_ladder(self):
        runs = self.runs()
        runs += [run(128, eps, {k: 1e-30 for k in NORM_KEYS}, rtol=1e-12) for eps in self.EPS]
        v = an.verdict(runs, self.EPS)
        self.assertEqual(v["verdict"], "PASS")
        self.assertEqual(v["per_eps"]["1.0"]["grids"], [16, 32, 64, 128])


class AlgebraicControlTests(unittest.TestCase):
    def test_negligible_and_not(self):
        base = run(128, 1.0, {k: 1.0e-3 for k in NORM_KEYS})
        tight = run(128, 1.0, {k: 1.0e-3 * (1 + 1e-6) for k in NORM_KEYS}, rtol=1e-12)
        c = an.algebraic_control([base, tight])
        (d,) = c.values()
        self.assertTrue(d["negligible"])
        self.assertAlmostEqual(d["max_rel_change"], 1e-6, delta=1e-9)
        tight2 = run(128, 1.0, {k: 1.5e-3 for k in NORM_KEYS}, rtol=1e-12)
        (d2,) = an.algebraic_control([base, tight2]).values()
        self.assertFalse(d2["negligible"])
        self.assertAlmostEqual(d2["max_rel_change"], 0.5, places=9)

    def test_no_matching_main_run(self):
        tight = run(128, 1.0, {k: 1.0e-3 for k in NORM_KEYS}, rtol=1e-12)
        self.assertEqual(an.algebraic_control([tight]), {})


class MarkdownSmokeTests(unittest.TestCase):
    def test_markdown_contains_orders_and_verdict(self):
        runs = []
        for eps in (0.5, 1.0, 1.5, 2.0):
            runs += ladder(eps, GOOD)
        v = an.verdict(runs)
        md = an.markdown(runs, v)
        self.assertIn("Verdict D: **PASS**", md)
        self.assertIn("2.00", md)
        self.assertIn("## eps = 1", md)


if __name__ == "__main__":
    unittest.main()
