"""Basis contrasts (``sf43ana.basis``, decision H2): known answers on synthetic arms, the uniform-weight control and its flag, schema."""
import json
import os
import unittest

import numpy as np

import sf43ana  # noqa: F401
from sf43ana import basis
from sf43ana import config as cfgmod
from sf43ana.tests import fixtures as fx
from sf43ana.tests.test_e2e import World

BASES = {"primary": "disp", "secondary": "pos"}


def arm(seeds, pos_tp, disp_tp, pos_tap=None, disp_tap=None):
    return {"seed": np.asarray(seeds), "cols": {"alpha_TP": np.asarray(pos_tp, float), "alpha_TP:disp": np.asarray(disp_tp, float),
                                                "alpha_TAP": np.asarray(pos_tap if pos_tap is not None else pos_tp, float),
                                                "alpha_TAP:disp": np.asarray(disp_tap if disp_tap is not None else disp_tp, float)}}


class Known(unittest.TestCase):
    PRE = {"bases": BASES, "primary_window": "main", "basis": {"B": 300, "seed": 4, "amplitudes": {"s1": {"flux": "prod", "unif": "produnif"}}}}

    def data(self, shift_unif):
        rng = np.random.default_rng(5)
        n = 60
        disp = 0.01 + 0.002 * rng.standard_normal(n)
        pos = disp * 1.2 + 0.0005 * rng.standard_normal(n)           # flux: the position basis is 20 % larger
        du = 0.01 + 0.002 * rng.standard_normal(n)
        pu = du + shift_unif + 0.0002 * rng.standard_normal(n)
        return {"arms": {"prod": arm(np.arange(n), pos, disp), "produnif": arm(np.arange(n), pu, du)}}, (pos, disp, pu, du)

    def test_flux_difference_and_ratio_are_the_paired_mean_difference_and_ratio_of_means(self):
        d, (pos, disp, _, _) = self.data(0.0)
        r = basis.run(self.PRE, d)["amplitudes"]["s1"]["flux"]["alpha_TP"]
        self.assertAlmostEqual(r["difference"], float((pos - disp).mean()), places=15)
        self.assertAlmostEqual(r["ratio"], float(pos.mean() / disp.mean()), places=15)
        self.assertEqual(r["n"], 60)
        self.assertEqual((r["mean_consumer"], r["mean_pollock"]), (float(pos.mean()), float(disp.mean())))  # (paired_contrast's field names: first, second)
        lo, hi = r["student"]
        sd = float((pos - disp).std(ddof=1))
        from scipy import stats
        h = stats.t.isf(0.05, 59) * sd / np.sqrt(60)
        self.assertAlmostEqual(lo, float((pos - disp).mean()) - h, places=15)
        self.assertAlmostEqual(hi, float((pos - disp).mean()) + h, places=15)
        self.assertLess(r["bootstrap_t"][0], r["difference"])
        self.assertLess(r["difference"], r["bootstrap_t"][1])
        self.assertLess(r["ratio_student"][0], r["ratio"])
        self.assertLess(r["ratio"], r["ratio_student"][1])
        self.assertLess(r["ratio_percentile"][0], r["ratio"])
        self.assertLess(r["ratio"], r["ratio_percentile"][1])
        self.assertTrue(lo > 0)  # the flux contrast is not a control: no flag attached
        self.assertNotIn("flag", r)

    def test_control_flag_is_raised_only_if_the_interval_excludes_zero(self):
        d, _ = self.data(0.0)
        c = basis.run(self.PRE, d)["amplitudes"]["s1"]["control_uniform"]["alpha_TP"]
        self.assertFalse(c["flag"])
        self.assertEqual(c["interval_excludes_zero"], {"student": False, "bootstrap_t": False})
        d, _ = self.data(0.001)  # a position-basis offset of half a standard deviation: far outside
        c = basis.run(self.PRE, d)["amplitudes"]["s1"]["control_uniform"]["alpha_TP"]
        self.assertTrue(c["flag"])
        self.assertTrue(c["interval_excludes_zero"]["student"])
        self.assertGreater(c["student"][0], 0.0)

    def test_estimators_arms_and_a_missing_column(self):
        d, _ = self.data(0.0)
        out = basis.run(self.PRE, d)
        r = out["amplitudes"]["s1"]
        self.assertEqual(sorted(r["flux"]), ["alpha_TAP", "alpha_TP"])
        self.assertEqual(sorted(r["control_uniform"]), ["alpha_TAP", "alpha_TP"])
        self.assertEqual(r["arms"], {"flux": "prod", "unif": "produnif"})
        self.assertEqual(out["config"]["difference"], "pos - disp")
        del d["arms"]["produnif"]["cols"]["alpha_TAP"]
        self.assertIn("unavailable", basis.run(self.PRE, d)["amplitudes"]["s1"]["control_uniform"]["alpha_TAP"]["note"])


class InTheAnalysis(World):
    def test_block_from_the_campaign_with_the_uniform_weight_arm(self):
        def rows(src, seg, s, wname, base):
            if src != "p1" or wname != "main":
                return []
            return [dict(base, weights="unif", alpha_TP=0.01 + 0.0001 * (s % 5), alpha_TAP=0.01 + 0.0001 * (s % 5))]
        self._tmp = __import__("tempfile").TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        self.pr = fx.prereg_dict()
        self.pr["arms"].append(fx.arm("p1u", "1", "g1", range(1000, 1040), 40, 20, range(1900, 1910), source="p1", role="basis_control_uniform"))
        self.pr["arms"][-1]["selection"] = {"weights": "unif"}
        self.pr["basis"] = {"B": 100, "amplitudes": {"1": {"flux": "p1", "unif": "p1u"}}}
        self.paths = fx.write_world(self.root, self.pr, __import__("sf43ana.tests.test_e2e", fromlist=["mean_of"]).mean_of, extra_rows=rows)
        self.out = os.path.join(self.root, "out")
        J, _ = self.run_()
        b = J["basis"]["amplitudes"]["1"]
        self.assertNotIn("error", J["basis"])
        c = b["control_uniform"]["alpha_TP"]
        self.assertEqual(c["n"], 40)
        # the unif rows carry the same value in both bases (the fixture's extra rows copy the base row with weights unif): difference exactly 0
        self.assertEqual(c["difference"], 0.0)
        self.assertFalse(c["flag"])
        self.assertEqual(b["flux"]["alpha_TP"]["n"], 40)
        self.assertGreater(b["flux"]["alpha_TP"]["ratio"], 1.0)  # the fixture's position basis is 2 % larger


class Schema(World):
    def test_strict_keys_and_arm_references(self):
        self.build()
        with open(self.paths["prereg"]) as f:
            raw = json.load(f)
        for blk, text in (({"bogus": 1, "amplitudes": {}}, "bogus"),
                          ({"amplitudes": {"1": {"flux": "p1"}}}, "lacks 'unif'"),
                          ({"amplitudes": {"1": {"flux": "p1", "unif": "nope"}}}, "nope"),
                          ({"amplitudes": {"7": {"flux": "p1", "unif": "p1"}}}, "unknown amplitude"),
                          ({"amplitudes": {"1": {"flux": "p1", "unif": "p1", "x": 1}}}, "unknown key")):
            r2 = json.loads(json.dumps(raw))
            r2["basis"] = blk
            path = os.path.join(self.root, "bad.json")
            with open(path, "w") as f:
                json.dump(r2, f)
            with self.assertRaises(cfgmod.ConfigError) as cm:
                cfgmod.load_prereg(path)
            self.assertIn(text, str(cm.exception))


if __name__ == "__main__":
    unittest.main()
