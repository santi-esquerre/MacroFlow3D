"""Pollock contrast and decomposition (sf43ana.pollock) on a synthetic campaign whose Pollock rows are the consumer's times a known factor."""
import json
import os
import tempfile
import unittest

import numpy as np

import sf43ana  # noqa: F401
from sf43ana import config, pollock, summary
from sf43ana.tests import fixtures as fx
from sf43ana.tests.test_e2e import mean_of

FACTOR = {"alpha_TP": 0.9, "alpha_TAP": 0.8, "alpha_LP": 1.25, "U_r": 1.0}


def pollock_rows(src, seg, seed, wname, base):
    """Pollock rows next to the consumer's: same family / weights / basis / set / seed / hmax_div, ``code = pollock``."""
    if base["code"] != "consumer":
        return []
    return [dict(base, code="pollock", **{k: base[k] * f for k, f in FACTOR.items()})]


def build(root, amp_key="1"):
    pr = fx.prereg_dict()
    pr["arms"] += [fx.arm("p1pol", "1", "g1", range(1000, 1040), 40, 20, range(1900, 1910), source="p1", code="pollock", role="pollock_production"),
                   fx.arm("g5pol", "1", "gg", range(8000, 8040), 40, 20, range(8900, 8905), source="g5c", code="pollock", role="pollock_twin")]
    pr["gate5"] = {"B": 199, "seed": 3, "floor": 20, "amplitudes": {"1": {"consumer": "g5c", "par2": "g5p"}}}
    pr["g7"] = {"twins": {"1": {"consumer": "g5c", "pollock": "g5pol", "par2": "g5p"}}}
    pr["pollock"] = {"B": 300, "seed": 5, "contrast": {"1": {"consumer": "p1", "pollock": "p1pol"}},
                     "decomposition": {"1": {"consumer": "g5c", "pollock": "g5pol", "par2": "g5p"}}}
    paths = fx.write_world(root, pr, mean_of, extra_rows=pollock_rows)
    return pr, paths


class PollockBlocks(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.root = self._t.name
        self.pr, self.paths = build(self.root)

    def run_(self):
        S, _ = summary.run(self.paths["prereg"], self.paths["campaign"], self.paths["reduced"], "none", os.path.join(self.root, "out"))
        return S

    def test_contrast_recovers_the_known_factors_in_both_bases(self):
        C = self.run_()["pollock"]["contrast"]["1"]
        self.assertEqual(sorted(C), sorted("%s/%s" % (e, b) for e in ("alpha_TP", "alpha_TAP", "alpha_LP", "U_r") for b in ("disp", "pos")))
        for key, f in (("alpha_TP/disp", 0.9), ("alpha_TAP/pos", 0.8), ("alpha_LP/disp", 1.25), ("U_r/pos", 1.0)):
            c = C[key]
            self.assertEqual(c["n"], 40, key)
            self.assertAlmostEqual(c["ratio"], 1.0 / f, places=9)  # consumer / Pollock
            self.assertAlmostEqual(c["difference"], c["mean_consumer"] - c["mean_pollock"], places=12)
            self.assertLess(c["ratio_percentile"][0], c["ratio"] + 1e-12)
        d = C["alpha_TP/disp"]  # consumer is larger than Pollock by a factor 1/0.9: positive difference, intervals around it
        self.assertGreater(d["difference"], 0)
        self.assertTrue(d["student"][0] < d["difference"] < d["student"][1])
        self.assertTrue(d["bootstrap_t"][0] < d["difference"] < d["bootstrap_t"][1])
        self.assertEqual(C["U_r/pos"]["difference"], 0.0)
        self.assertIn("note", C["U_r/pos"])  # identical values: degenerate intervals, no NaN / warning
        self.assertEqual(self.run_()["pollock"]["label"], pollock.LABEL)

    def test_decomposition_pairs_the_gate5_arms_by_seed(self):
        D = self.run_()["pollock"]["decomposition"]["1"]
        r = D["alpha_LP"]
        self.assertEqual(sorted(r), ["consumer/par2", "consumer/pollock", "pollock/par2"])
        self.assertEqual(r["consumer/pollock"]["n"], 40)
        self.assertEqual(r["consumer/pollock"]["pairing"]["n_common"], 40)
        self.assertAlmostEqual(r["consumer/pollock"]["ratio"], 1.0 / 1.25, places=9)
        self.assertAlmostEqual(r["consumer/pollock"]["ratio"] * r["pollock/par2"]["ratio"], r["consumer/par2"]["ratio"], places=9)
        q = r["consumer/pollock"]  # the synthetic factor is exact, so the interval is degenerate (up to rounding)
        self.assertTrue(q["ratio_lo"] - 1e-9 <= q["ratio"] <= q["ratio_hi"] + 1e-9)
        self.assertIn("alpha_TP:disp", D)

    def test_g7_twin_points_at_the_real_pollock_entry(self):
        J = self.run_()
        self.assertEqual(J["g7"]["twins"]["1"]["alpha_LP"]["consumer/pollock"]["n"], 40)
        self.assertNotIn("note", J["g7"]["twins"]["1"]["alpha_LP"]["consumer/pollock"])

    def test_pollock_arm_without_rows_is_reported_not_fatal(self):
        root2 = os.path.join(self.root, "w2")
        pr = json.loads(json.dumps(self.pr))
        paths = fx.write_world(root2, pr, mean_of)  # no Pollock rows at all
        S, _ = summary.run(paths["prereg"], paths["campaign"], paths["reduced"], "none", os.path.join(root2, "out"))
        self.assertEqual(S["process"]["arms"]["p1pol"]["status"], "below_floor")
        self.assertEqual(S["process"]["arms"]["p1pol"]["N"], 0)
        c = S["pollock"]["contrast"]["1"]["alpha_TP/disp"]
        self.assertIn("note", c)

    def test_strict_validation_of_the_block(self):
        raw = json.loads(json.dumps(self.pr))
        for mutate in (lambda r: r["pollock"].update(bogus=1), lambda r: r["pollock"]["contrast"]["1"].update(consumer="nope"),
                       lambda r: r["pollock"]["decomposition"]["1"].pop("par2"), lambda r: r["pollock"]["contrast"].update({"9": {}})):
            r2 = json.loads(json.dumps(raw))
            mutate(r2)
            p = os.path.join(self.root, "bad.json")
            with open(p, "w") as f:
                json.dump(dict(r2, windows=fx.WINDOWS), f)
            with self.assertRaises(config.ConfigError):
                config.load_prereg(p)


if __name__ == "__main__":
    unittest.main()
