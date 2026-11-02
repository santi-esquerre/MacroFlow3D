import contextlib
import importlib.machinery
import io
import json
import os
import sys
import tempfile
import types
import unittest

import sf43ana  # noqa: F401
from sf43ana import __main__ as cli
from sf43ana import summary
from sf43ana.tests import fixtures as fx

TRUE = {"1": {"alpha_TP": 0.006, "alpha_TAP": 0.0055, "alpha_LP": 0.5}, "2.25": {"alpha_TP": 0.028, "alpha_TAP": 0.027, "alpha_LP": 0.9}}
BIAS = {"l5": 0.0004, "p1": 0.00014, "l20": 0.00005, "l4": 0.003, "l8": 0.001, "p2": 0.0004, "p2h10": 0.0007}


def mean_of(a, w, basis, est):
    m = TRUE[a["amplitude"]][est]
    if est != "alpha_LP":
        m += BIAS.get(a["id"], 0.0)
    return m * (1.3 if w == "far" else 1.0) * (1.02 if basis == "pos" else 1.0)


def strict_json(path):
    def bad(c):
        raise ValueError("non-finite number " + c)
    with open(path) as f:
        return json.load(f, parse_constant=bad)


class World(unittest.TestCase):
    def build(self, **kw):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        self.pr = fx.prereg_dict()
        self.paths = fx.write_world(self.root, self.pr, mean_of, **kw)
        self.out = os.path.join(self.root, "out")

    def run_(self, gate5=None):
        g = "none"
        if gate5 is not None:
            g = os.path.join(self.root, "gate5.json")
            with open(g, "w") as f:
                json.dump({"p": gate5}, f)
        s, path = summary.run(self.paths["prereg"], self.paths["campaign"], self.paths["reduced"], g, self.out)
        return strict_json(path), path


class EndToEnd(World):
    def test_sections_values_and_decisions(self):
        self.build()
        J, _ = self.run_({"1": 0.001, "2.25": 0.002})
        for k in ("estimands", "gate4", "family", "budgets", "precision", "process"):
            self.assertIn(k, J)
        e1 = J["estimands"]["1"]["estimators"]
        self.assertAlmostEqual(e1["alpha_TP"]["disp"]["mean"], 0.006 + 0.00014, delta=4 * 0.0021 / 40 ** 0.5)
        self.assertAlmostEqual(e1["alpha_TP"]["pos"]["mean"] / e1["alpha_TP"]["disp"]["mean"], 1.02, places=2)
        self.assertAlmostEqual(e1["alpha_TP"]["disp"]["ratio_to_first_order"], e1["alpha_TP"]["disp"]["mean"] / 0.006, places=12)
        self.assertGreater(e1["alpha_LP"]["U_r"]["mean"], 0.9)
        self.assertIn("Ubar2_over2_x_mean_slope", e1["alpha_LP"])
        self.assertEqual(sorted(J["estimands"]["1"]["halves_no_claim"]), ["h1", "h2"])
        d = J["estimands"]["2.25"]["distance"]
        self.assertAlmostEqual(d["ratio_far_to_main"]["alpha_LP/disp"]["ratio"], 1.3, delta=0.02)
        fam = {h["id"]: h for h in J["family"]["hypotheses"]}
        self.assertEqual(J["family"]["info"]["m"], 4)
        self.assertTrue(fam["robust:2.25"]["rejected"])  # alpha_T 0.028 against a small budget
        self.assertTrue(fam["gate5:2.25"]["rejected"])  # supplied p = 0.002 passes its Holm level once robust:2.25 is rejected
        self.assertEqual(J["gate4"]["2.25"]["class"], "physical (restricted)")
        self.assertEqual(J["precision"]["1"]["alpha_TP/disp"]["target_halfwidth"], 1e-3)
        self.assertIn("p_holm_adjusted", fam["robust:1"])
        bud = J["budgets"]["2.25"]["disp"]["alpha_TP"]
        self.assertEqual(bud["K"], 3)
        self.assertEqual(sorted(bud["components"]), ["E_W", "E_WxH", "E_h"])
        self.assertFalse(J["process"]["bootstrap_resolution"]["ok"])  # fixture B (300) cannot resolve the Holm worst level
        self.assertEqual(J["process"]["arms"]["w2far"]["status"], "ok")

    def test_gate5_missing_enters_with_p_one(self):
        self.build()
        J, _ = self.run_(None)
        fam = {h["id"]: h for h in J["family"]["hypotheses"]}
        self.assertEqual((fam["gate5:1"]["p"], fam["gate5:1"]["testable"]), (1.0, False))
        self.assertEqual(J["family"]["info"]["m"], 4)

    def test_untestable_arm_below_floor(self):
        self.build(fail=[("l5", 1005)])  # the ladder arm l5 keeps 5 seeds < floor 20, no replacement
        J, _ = self.run_({"1": 0.001, "2.25": 0.002})
        fam = {h["id"]: h for h in J["family"]["hypotheses"]}
        self.assertFalse(fam["robust:1"]["testable"])
        self.assertEqual(fam["robust:1"]["p"], 1.0)
        self.assertIn("below its floor", fam["robust:1"]["note"])
        self.assertEqual(J["gate4"]["1"]["class"], "unresolved")
        self.assertEqual(J["process"]["arms"]["l5"]["status"], "below_floor")
        self.assertTrue(fam["robust:2.25"]["rejected"])  # the other amplitude is unaffected
        self.assertEqual(J["family"]["info"]["m"], 4)
        self.assertEqual(J["estimands"]["1"]["status"], "ok")  # estimates of the production arm are still reported

    def test_replacement_is_disclosed(self):
        self.build(fail=[("p1", 1005), ("l5", 1005), ("l20", 1005)], replace=[("g1", 1005, 1900)])
        J, _ = self.run_(None)
        self.assertEqual(J["process"]["arms"]["p1"]["replacements_applied"], [{"replaced_by": 1900, "seed": 1005}])
        self.assertEqual(J["process"]["arms"]["p1"]["N"], 40)

    def test_deterministic_and_cli(self):
        self.build()
        _, p1 = self.run_({"1": 0.5})
        with open(p1, "rb") as f:
            a = f.read()
        self.out = os.path.join(self.root, "out2")
        _, p2 = self.run_({"1": 0.5})
        with open(p2, "rb") as f:
            self.assertEqual(a, f.read())
        out3 = os.path.join(self.root, "out3")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli.main(["run", "--prereg", self.paths["prereg"], "--campaign", self.paths["campaign"], "--reduced", self.paths["reduced"],
                           "--gate5", "none", "--out", out3])
        self.assertEqual(rc, 0)
        self.assertTrue(os.path.isfile(os.path.join(out3, "summary.json")))

    def test_optional_sibling_blocks_are_merged(self):
        self.build()
        name = "sf43ana.gate5"
        fake = types.ModuleType(name)
        fake.__spec__ = importlib.machinery.ModuleSpec(name, None)
        fake.analyse = lambda ctx: {"seen_arms": sorted(ctx.arms)[:2], "gate5_p": ctx.gate5_p}
        old = sys.modules.get(name)
        sys.modules[name] = fake
        try:
            J, _ = self.run_({"1": 0.5})
        finally:
            if old is None:
                del sys.modules[name]
            else:
                sys.modules[name] = old
        self.assertEqual(J["gate5"], {"seen_arms": ["h16", "h8"], "gate5_p": {"1": 0.5}})
        self.assertNotIn("g7", J)

    def test_bad_prereg_is_rejected(self):
        self.build()
        with open(self.paths["prereg"]) as f:
            raw = json.load(f)
        raw["amplitudes"]["1"]["ladder"]["production_index"] = 0
        with open(self.paths["prereg"], "w") as f:
            json.dump(raw, f)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            rc = cli.main(["run", "--prereg", self.paths["prereg"], "--campaign", self.paths["campaign"], "--reduced", self.paths["reduced"],
                           "--gate5", "none", "--out", self.out])
        self.assertEqual(rc, 2)
        self.assertIn("production_index", err.getvalue())


if __name__ == "__main__":
    unittest.main()
