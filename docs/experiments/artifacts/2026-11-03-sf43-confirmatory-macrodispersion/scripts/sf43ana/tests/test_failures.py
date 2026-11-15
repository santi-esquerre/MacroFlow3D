import math
import unittest

import numpy as np

from sf43ana import failures as fl


def rec(arm, seed, cls="ok", verdict="converged", segment=0, **kw):
    return dict(arm=arm, segment=segment, seed=seed, verdict=verdict, **{"class": cls}, **kw)


class ClopperPearson(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(fl.clopper_pearson(0, 100)[0], 0.0)
        self.assertAlmostEqual(fl.clopper_pearson(0, 100)[1], 1 - 0.025 ** 0.01, places=12)  # closed form for k = 0
        self.assertEqual(fl.clopper_pearson(100, 100)[1], 1.0)
        self.assertAlmostEqual(fl.clopper_pearson(100, 100)[0], 0.025 ** 0.01, places=12)
        lo, hi = fl.clopper_pearson(1, 100)  # the roots of P(X <= 1; p) = 0.025 and P(X >= 1; p) = 0.025 (independent of the beta form)
        from scipy import optimize
        self.assertAlmostEqual(hi, optimize.brentq(lambda p: (1 - p) ** 100 + 100 * p * (1 - p) ** 99 - 0.025, 1e-6, 0.5), places=10)
        self.assertAlmostEqual(lo, optimize.brentq(lambda p: 1 - (1 - p) ** 100 - 0.025, 1e-9, 0.5), places=10)
        self.assertAlmostEqual(hi, 0.0545, places=4)  # tabulated
        lo, hi = fl.clopper_pearson(10, 50)  # tabulated: 0.1003 .. 0.3372
        self.assertAlmostEqual(lo, 0.1003, places=4)
        self.assertAlmostEqual(hi, 0.3372, places=4)
        self.assertTrue(all(math.isnan(v) for v in fl.clopper_pearson(0, 0)))


class Imputation(unittest.TestCase):
    def test_worst_case_is_the_5th_and_95th_percentile(self):
        v = np.arange(1.0, 101.0)
        r = fl.impute_worst_case(v, 5)
        q05, q95 = np.quantile(v, 0.05), np.quantile(v, 0.95)
        self.assertAlmostEqual(r["low"]["fill"], q05)
        self.assertAlmostEqual(r["high"]["fill"], q95)
        self.assertAlmostEqual(r["low"]["mean"], (v.sum() + 5 * q05) / 105, places=12)
        self.assertAlmostEqual(r["high"]["mean"], (v.sum() + 5 * q95) / 105, places=12)
        aug = np.concatenate([v, np.full(5, q05)])
        from scipy import stats
        self.assertAlmostEqual(r["low"]["bound95"], aug.mean() - stats.t.isf(0.05, 104) * aug.std(ddof=1) / math.sqrt(105), places=12)
        self.assertLess(r["low"]["mean"], r["mean_obs"])
        self.assertGreater(r["high"]["mean"], r["mean_obs"])


class Run(unittest.TestCase):
    def build(self):
        ledger = []
        for s in range(100):  # arm A: seeds 0..99 attempted, 3 failed (3 %), replaced by 100..102
            ledger.append(rec("A", s, "unusable_solve" if s in (5, 17, 60) else "ok", "fallback" if s in (7, 8) else "converged"))
        for s, f in ((100, 5), (101, 17), (102, 60)):
            ledger.append(rec("A", s, "ok", replaces=f))
        for s in range(200):  # arm B: 200 attempted, 1 failed (0.5 %)
            ledger.append(rec("B", s, "unusable_solve" if s == 3 else "ok"))
        ledger.append(rec("H", 0, "halted"))
        ledger.append(rec("A", 5, "unusable_solve", segment=1))  # a second failed segment of the same seed counts once
        rng = np.random.default_rng(3)
        usable_A = [s for s in range(103) if s not in (5, 17, 60)]
        usable_B = [s for s in range(200) if s != 3]
        arms = {"A": {"seed": np.array(usable_A), "cols": {"alpha_TP": 0.05 + 0.01 * rng.standard_normal(len(usable_A))}},
                "B": {"seed": np.array(usable_B), "cols": {"alpha_TP": 0.05 + 0.01 * rng.standard_normal(len(usable_B))}}}
        return {"ledger": ledger, "arms": arms}

    def test_rates_replacements_imputation_and_fallback_sensitivity(self):
        data = self.build()
        r = fl.run({"failures": {"estimands": ["alpha_TP"]}}, data)
        A, B = r["arms"]["A"], r["arms"]["B"]
        self.assertEqual((A["attempted"], A["failed"]), (103, 3))
        self.assertAlmostEqual(A["rate"], 3 / 103)
        self.assertEqual(A["cp95"], list(fl.clopper_pearson(3, 103)))
        self.assertTrue(A["exceeds_max_rate"])
        self.assertEqual(A["failed_seeds"], [5, 17, 60])
        self.assertEqual(A["replacements"], [{"failed_seed": 5, "replacement_seed": 100}, {"failed_seed": 17, "replacement_seed": 101},
                                             {"failed_seed": 60, "replacement_seed": 102}])
        self.assertEqual(A["failed_not_replaced"], [])
        self.assertEqual(A["fallback_accepted_seeds"], [7, 8])
        imp = A["estimands"]["alpha_TP"]["worst_case_imputation"]
        self.assertEqual(imp["n_imputed"], 3)
        self.assertLess(imp["low"]["mean"], imp["mean_obs"])
        ex = A["estimands"]["alpha_TP"]["excluding_fallback"]
        self.assertEqual(ex["n_excluded"], 2)
        self.assertEqual(ex["n"], 98)
        self.assertEqual(A["estimands"]["alpha_TP"]["all"]["n"], 100)
        v = data["arms"]["A"]["cols"]["alpha_TP"][[i for i, s in enumerate(data["arms"]["A"]["seed"]) if s not in (7, 8)]]
        self.assertAlmostEqual(ex["mean"], v.mean(), places=14)
        # arm B: 1 / 200 = 0.5 % <= 1 %: no imputation, still reported
        self.assertFalse(B["exceeds_max_rate"])
        self.assertNotIn("worst_case_imputation", B["estimands"]["alpha_TP"])
        self.assertEqual(r["halted"], ["H"])
        self.assertEqual(r["total"]["attempted"], 303)
        self.assertEqual(r["total"]["failed"], 4)

    def test_unreplaced_failures_listed_and_threshold_configurable(self):
        data = self.build()
        data["ledger"] = [x for x in data["ledger"] if x.get("replaces") != 60]
        r = fl.run({"failures": {"arms": ["A"], "max_rate": 0.5}}, data)
        self.assertEqual(r["arms"]["A"]["failed_not_replaced"], [60])
        self.assertFalse(r["arms"]["A"]["exceeds_max_rate"])
        self.assertEqual(list(r["arms"]), ["A"])


if __name__ == "__main__":
    unittest.main()
