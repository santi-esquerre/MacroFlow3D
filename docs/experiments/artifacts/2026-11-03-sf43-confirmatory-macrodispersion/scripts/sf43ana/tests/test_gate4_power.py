"""Achieved Gate 4 power at the realized sizes (``sf43ana.gate4_power``): known answer, monotonicity, determinism, schema."""
import json
import os
import unittest

import numpy as np
from scipy import stats

import sf43ana  # noqa: F401
from sf43ana import config as cfgmod
from sf43ana import gate4_power as gp
from sf43ana.gate4 import Design
from sf43ana.tests import fixtures as fx
from sf43ana.tests.test_e2e import World, mean_of


class StubPrereg:
    """What ``power_cell`` reads of a ``Prereg``."""
    raw = {"bootstrap": {"B_budget": 200, "B_sampling": 200, "seed": 1}, "eh": {"p_range": [0.5, 2.1], "p_cap": 2.0, "safety": 1.25, "fallback_factor": 3.0},
           "tie_rtol": 1e-9, "sampling_method": "student", "levels": {"fwer": 0.05, "interval": 0.9, "ucl": 0.05}}

    @staticmethod
    def null_value(amp):
        return 0.0


class KnownAnswer(unittest.TestCase):
    def test_single_component_power_is_the_noncentral_t_power_of_the_student_limit(self):
        """One negligible budget component (E_int with a 1e-9 spread): RL(a) = the Student lower limit at tail a / 2, so the power of
        'RL > 0' is P(T > t_{1 - a/2, N-1}) with T noncentral t, ncp = alt sqrt(N) / sd."""
        N, sd, alt = 30, 1.0, 0.5
        rng = np.random.default_rng(0)
        x = sd * (rng.standard_normal(N) - rng.standard_normal(N).mean())
        x = (x - x.mean()) / x.std(ddof=1) * sd + alt  # observed production SD = sd exactly
        d = Design(x, ["E_int"], h16=1e-9 * rng.standard_normal(20), h8=np.zeros(20))
        cfg = {"R": 4000, "B_budget": 200, "seed": 11, "chunk": 1000}
        levels = [0.05 / 8, 0.05]
        rej = gp.power_cell(StubPrereg, "a", {"alpha_TP": d, "alpha_TAP": d}, {"alpha_TP": alt, "alpha_TAP": alt}, "no_bias", cfg, levels)["alpha_TP"]
        for j, a in enumerate(levels):
            want = float(stats.nct.sf(stats.t.isf(a / 2, N - 1), N - 1, alt * np.sqrt(N) / sd))
            got = rej[:, j].mean()
            self.assertAlmostEqual(got, want, delta=4.5 * np.sqrt(want * (1 - want) / 4000) + 1e-3, msg=(a, got, want))

    def test_the_two_estimators_share_their_draws_and_the_run_is_deterministic(self):
        N = 40
        x = np.random.default_rng(1).standard_normal(N)
        d = Design(x, ["E_int"], h16=np.random.default_rng(2).standard_normal(20) * 0.1, h8=np.zeros(20))
        cfg = {"R": 300, "B_budget": 200, "seed": 3, "chunk": 100}
        args = ("a", {"alpha_TP": d, "alpha_TAP": d}, {"alpha_TP": 0.4, "alpha_TAP": 0.4}, "observed", cfg, [0.00625, 0.05])
        r1, r2 = gp.power_cell(StubPrereg, *args), gp.power_cell(StubPrereg, *args)
        self.assertTrue(np.array_equal(r1["alpha_TP"], r2["alpha_TP"]))
        self.assertTrue(np.array_equal(r1["alpha_TP"], r1["alpha_TAP"]))  # identical inputs, identical draws


class InTheAnalysis(World):
    GP = {"R": 120, "B_budget": 600, "seed": 5, "chunk": 60,
          "alternatives": {"1": {"alpha_TP": 0.006, "alpha_TAP": 0.0055}, "2.25": {"alpha_TP": 0.028, "alpha_TAP": 0.027}}}

    def run_with(self, gpcfg, **kw):
        self.build(**kw)
        with open(self.paths["prereg"]) as f:
            raw = json.load(f)
        raw["gate4_power"] = gpcfg
        with open(self.paths["prereg"], "w") as f:
            json.dump(raw, f)
        return self.run_()[0]

    def test_block_structure_sizes_and_monotonicity_in_the_alternative(self):
        J = self.run_with(self.GP)
        P = J["gate4_power"]
        self.assertNotIn("error", P)
        self.assertEqual(P["config"]["levels"], {"fwer_over_m": 0.05 / 4, "fwer": 0.05})
        a = P["amplitudes"]["1"]
        self.assertEqual(a["status"], "computed")
        self.assertEqual(a["realized_sizes"], {"production": 40, "ladder": 40, "width": 40, "h_max": 20})
        self.assertEqual(sorted(a["scenarios"]), ["no_bias", "observed"])
        sd_obs = J["estimands"]["1"]["estimators"]["alpha_TP"]["disp"]["sd"]
        self.assertAlmostEqual(a["observed"]["alpha_TP"]["sd_prod"], sd_obs, places=12)  # the OBSERVED production SD enters
        for sc in a["scenarios"].values():
            for e in ("alpha_TP", "alpha_TAP"):
                lo, hi = sc["estimators"][e]["fwer_over_m"]["p"], sc["estimators"][e]["fwer"]["p"]
                self.assertLessEqual(lo, hi + 1e-12)                  # the larger level rejects more
            self.assertLessEqual(sc["both"]["fwer"]["p"], min(sc["estimators"][e]["fwer"]["p"] for e in sc["estimators"]) + 1e-12)
            self.assertAlmostEqual(sc["both_frechet_upper"]["fwer"], min(sc["estimators"][e]["fwer"]["p"] for e in sc["estimators"]), places=12)
        # a clearly larger alternative gives power 1; a null alternative gives (conservatively) at most about the level
        big = self.run_with(dict(self.GP, alternatives={"1": {"alpha_TP": 0.05, "alpha_TAP": 0.05}}))["gate4_power"]["amplitudes"]["1"]
        self.assertEqual(big["scenarios"]["no_bias"]["both"]["fwer_over_m"]["p"], 1.0)
        zero = self.run_with(dict(self.GP, alternatives={"1": {"alpha_TP": 7.992e-05, "alpha_TAP": 7.992e-05}}))["gate4_power"]["amplitudes"]["1"]
        self.assertLessEqual(zero["scenarios"]["no_bias"]["both"]["fwer"]["p"], 0.05 + 0.04)
        self.assertEqual(set(J["gate4_power"]["amplitudes"]), {"1", "2.25"})
        # deterministic
        J2 = self.run_with(self.GP)
        self.assertEqual(J2["gate4_power"], P)

    def test_an_untestable_design_is_not_computed_with_its_reasons(self):
        J = self.run_with(self.GP, fail=[("l5", 1005)])   # the ladder arm l5 stops at 5 seeds < floor
        a = J["gate4_power"]["amplitudes"]["1"]
        self.assertTrue(a["status"].startswith("not computed"))
        self.assertTrue(any("below its floor" in r for r in a["reasons"]))
        self.assertEqual(J["gate4_power"]["amplitudes"]["2.25"]["status"], "computed")

    def test_absent_block_and_strict_schema(self):
        self.build()
        J, _ = self.run_()
        self.assertNotIn("gate4_power", J)
        with open(self.paths["prereg"]) as f:
            raw = json.load(f)
        for gpcfg, text in ((dict(self.GP, bogus=1), "bogus"),
                            (dict(self.GP, alternatives={"1": {"alpha_TP": 1.0}}), "needs alpha_TP and alpha_TAP"),
                            (dict(self.GP, alternatives={"9": {"alpha_TP": 1.0, "alpha_TAP": 1.0}}), "unknown amplitude"),
                            (dict(self.GP, scenarios=["nope"]), "scenarios")):
            r2 = json.loads(json.dumps(raw))
            r2["gate4_power"] = gpcfg
            path = os.path.join(self.root, "bad.json")
            with open(path, "w") as f:
                json.dump(r2, f)
            with self.assertRaises(cfgmod.ConfigError) as cm:
                cfgmod.load_prereg(path)
            self.assertIn(text, str(cm.exception))


if __name__ == "__main__":
    unittest.main()
