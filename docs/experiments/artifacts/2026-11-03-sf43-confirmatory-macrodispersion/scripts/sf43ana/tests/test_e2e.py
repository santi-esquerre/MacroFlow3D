import contextlib
import importlib.machinery
import io
import json
import os
import sys
import tempfile
import types
import unittest
import warnings
from unittest import mock

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
        g = "auto"
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
                           "--out", out3])
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
        self.assertEqual(J["gate5"], {"seen_arms": ["g5c", "g5p"], "gate5_p": {"1": 0.5}})
        self.assertNotIn("g7", J)

    def test_gate5_from_the_campaign_in_prefix_order_with_the_qualified_procedure(self):
        from sf43ana.tests.test_gate5 import fake_summary
        self.build(fail=[("g5c", 8030)])  # the consumer arm stops after 30 complete positions, the Par2 arm has 40
        cov = os.path.join(self.root, "cov.json")
        ok = {"student": {20: False, 40: False}, "hall": {20: True, 40: True}, "boott": {20: True, 40: True}}
        with open(cov, "w") as f:
            json.dump(fake_summary(ok, ns=(20, 40)), f)
        with open(self.paths["prereg"]) as f:
            raw = json.load(f)
        raw["gate5"] = {"B": 999, "seed": 3, "coverage_summary": cov, "floor": 20,
                        "amplitudes": {"1": {"consumer": "g5c", "par2": "g5p"}}}
        with open(self.paths["prereg"], "w") as f:
            json.dump(raw, f)
        J, _ = self.run_()  # --gate5 auto: p-values from the block
        a = J["gate5"]["amplitudes"]["1"]
        self.assertEqual(a["n"], 30)  # the paired prefix
        self.assertEqual(a["selection"]["grid_n"], 20)  # largest coverage grid point <= 30
        self.assertEqual(a["procedure"], "hall")
        fam = {h["id"]: h for h in J["family"]["hypotheses"]}
        self.assertTrue(fam["gate5:1"]["testable"])
        self.assertEqual(fam["gate5:1"]["p"], a["p_tost"])
        self.assertFalse(fam["gate5:2.25"]["testable"])  # no Gate 5 entry for this amplitude: p = 1
        self.assertEqual(J["failures"]["arms"]["g5c"]["failed_seeds"], [8030])
        self.assertEqual(J["failures"]["arms"]["g5c"]["failed"], 1)
        self.assertIn("fieller", a["power"])
        self.assertIn("pair_exclusion_sensitivity", a)
        # floor: 30 pairs < 40 -> untestable
        raw["gate5"]["floor"] = 40
        with open(self.paths["prereg"], "w") as f:
            json.dump(raw, f)
        self.out = os.path.join(self.root, "out_floor")
        J2, _ = self.run_()
        a2 = J2["gate5"]["amplitudes"]["1"]
        self.assertIsNone(a2["procedure"])
        self.assertEqual(a2["p_tost"], 1.0)
        self.assertIn("floor", a2["note"])

    def test_strict_prereg_rejects_unknown_keys_and_unknown_arms(self):
        self.build()
        with open(self.paths["prereg"]) as f:
            raw = json.load(f)
        for mutate, text in ((lambda r: r.update(bogus=1), "bogus"),
                             (lambda r: r["amplitudes"]["1"].update(bogus=1), "bogus"),
                             (lambda r: r["arms"][0].update(bogus=1), "bogus"),
                             (lambda r: r["bootstrap"].update(bogus=1), "bogus"),
                             (lambda r: r.update(gate5={"amplitudes": {"1": {"consumer": "nope", "par2": "g5p"}}}), "nope"),
                             (lambda r: r.update(positive_control={"arm": "nope", "sigma2": 0.04, "width_lambda": 24, "length_lambda": 104}), "nope"),
                             (lambda r: r.update(secondaries={"expI": {"x": {"arm": "nope"}}}), "nope"),
                             (lambda r: r.update(failures={"bogus": 1}), "bogus")):
            r2 = json.loads(json.dumps(raw))
            mutate(r2)
            path = os.path.join(self.root, "bad.json")
            with open(path, "w") as f:
                json.dump(r2, f)
            with contextlib.redirect_stderr(io.StringIO()) as err:
                rc = cli.main(["run", "--prereg", path, "--campaign", self.paths["campaign"], "--reduced", self.paths["reduced"], "--out", self.out])
            self.assertEqual(rc, 2, text)
            self.assertIn(text, err.getvalue())

    def test_descriptive_blocks_from_the_campaign(self):
        def rows(src, seg, s, wname, base):
            if src != "p1" or wname != "main" or base["basis"] != "pos":
                return []
            e = dict(base, weights="expI", alpha_TP=base["alpha_TP"] * 1.1, alpha_LP=base["alpha_LP"] * 0.9)
            t = dict(base, family="time", ell=0.1, t_end=20.0, U_time=0.5, alpha_LAM=0.3, alpha_Tfixed=0.01, alpha_TAM=0.02)
            return [e, t]
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = self._tmp.name
        self.pr = fx.prereg_dict()
        self.pr["secondaries"] = {"B": 50, "expI": {"w": {"arm": "p1"}}, "time_window": {"p1": {"amplitude": "s1"}},
                                  "recrossings": {"r": {"arm": "p1"}}, "bdd_fit": [{"name": "f", "arm": "p1", "amplitude": "s1"}]}
        self.pr["g7"] = {"twins": {"1": {"consumer": "p1", "pollock": "l5", "par2": "par2_1"}}}
        self.paths = fx.write_world(self.root, self.pr, mean_of, extra_rows=rows)
        self.out = os.path.join(self.root, "out")
        J, _ = self.run_()
        sec = J["secondaries"]
        self.assertAlmostEqual(sec["expI"]["w"]["alpha_TP"]["ratio"], 1.1, places=1)
        self.assertAlmostEqual(sec["expI"]["w"]["alpha_LP"]["ratio"], 0.9, places=1)
        self.assertAlmostEqual(sec["time_window"]["p1"]["t_end_adv"]["mean"], 20.0 * 0.5 / 0.1, places=9)
        self.assertAlmostEqual(sec["time_window"]["p1"]["alpha_LAM:time"]["mean"], 0.3, places=9)
        self.assertIn("input absent", sec["recrossings"]["r"]["note"])
        self.assertIn("input absent", sec["bdd_fit"]["f"]["note"])
        self.assertIn("consumer/pollock", J["g7"]["twins"]["1"]["alpha_LP"])  # twins ran on the loaded data

    def test_a_failing_descriptive_block_does_not_cost_the_primary_results(self):
        self.build()
        with open(self.paths["prereg"]) as f:
            raw = json.load(f)
        raw["positive_control"] = {"arm": "p1", "sigma2": 0.04, "width_lambda": 24, "length_lambda": 104, "B": 20, "weights": ["flux"]}
        with open(self.paths["prereg"], "w") as f:
            json.dump(raw, f)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            J, _ = self.run_({"1": 0.001, "2.25": 0.002})  # the fixture has no planes tables
        self.assertIn("error", J["positive_control"])
        self.assertIn("BLOCK_ERROR positive_control", err.getvalue())
        self.assertEqual(J["gate4"]["2.25"]["class"], "physical (restricted)")

    def test_check_finite_prints_only_counts(self):
        import re
        from sf43ana import finite
        self.build()
        J, path = self.run_({"1": 0.001, "2.25": 0.002})
        buf = io.StringIO()
        rc = finite.run(path, buf)
        text = buf.getvalue()
        self.assertEqual(rc, 0)
        self.assertIn("STRUCTURE_OK", text)
        self.assertIn("section gate4: finite=", text)
        self.assertIn("process: warnings_count=0 errors_count=0", text)
        for tok in re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", text):  # every number printed is an integer count
            self.fail("a non-integer number was printed: " + tok)
        for word in ("physical", "rejected", "numerical"):
            self.assertNotIn(word, text)
        bad = json.loads(json.dumps(J))  # a damaged summary fails
        del bad["family"]["hypotheses"][0]
        bad["estimands"]["1"]["estimators"]["alpha_TP"]["disp"]["mean"] = "nan"
        bad["secondaries"] = {"x": {"error": "boom"}}
        bp = os.path.join(self.root, "bad_summary.json")
        with open(bp, "w") as f:
            json.dump(bad, f)
        buf = io.StringIO()
        self.assertEqual(finite.run(bp, buf), 1)
        self.assertIn("STRUCTURE_FAIL", buf.getvalue())
        self.assertIn("error recorded in secondaries.x", buf.getvalue())
        self.assertIn("family does not list m hypotheses", buf.getvalue())
        self.assertIn("a reported mean of a production arm is not finite", buf.getvalue())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(cli.main(["check-finite", "--summary", path]), 0)
        self.assertIn("STRUCTURE_OK", buf.getvalue())

    def test_figures_are_drawn_from_the_summary_or_skipped_without_failing(self):
        from sf43ana import figures
        self.build()
        _, path = self.run_({"1": 0.001, "2.25": 0.002})
        try:
            import matplotlib  # noqa: F401
            have = True
        except ImportError:
            have = False
        self.assertEqual(os.path.isdir(os.path.join(self.out, "figures")), have)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            written = figures.make(path, os.path.join(self.root, "figs"))
            self.assertEqual(bool(written), have)
            self.assertEqual(figures.make(os.path.join(self.root, "missing.json"), os.path.join(self.root, "figs2")), [])
        self.assertIn("FIGURES_ERROR" if have else "FIGURES_SKIPPED", err.getvalue())

    def test_two_entries_sharing_the_production_arm_cover_the_hmax_passes(self):
        self.build()
        arms = {a["id"]: a for a in self.pr["arms"]}
        self.assertEqual((arms["h16"]["source"], arms["h8"]["source"], arms["h16"]["hmax_div"], arms["h8"]["hmax_div"]), ("p1", "p1", 16, 8))
        J, _ = self.run_()
        self.assertEqual(J["process"]["arms"]["h16"]["selection"]["code"], "consumer")
        bud = J["budgets"]["1"]["disp"]["alpha_TP"]
        self.assertIn("E_int", bud["components"])  # E_int built from the two passes of the one engine arm

    def test_ladder_gate_arm_and_floors_enter_the_design(self):
        self.build()
        J, _ = self.run_()
        det = J["budgets"]["2.25"]["disp"]["alpha_TP"]["components"]["E_h"]["details"]
        self.assertIn("gate_ok", det)
        self.assertEqual(det["gates"][0]["gate"]["arm"], "l12")
        self.assertIn("l12", J["process"]["arms"])
        # H13 floors act on the PAIRED prefix, besides each arm's own floor: the gate arm stops after 10 complete seeds
        self.build(fail=[("l12", 3010)])
        with open(self.paths["prereg"]) as f:
            raw = json.load(f)
        for a in raw["arms"]:
            a["floor"] = 5  # arm floors out of the way
        raw["floors"] = {"production": 5, "ladder": 20, "width": 5, "hmax": 5, "gate5": 5}
        with open(self.paths["prereg"], "w") as f:
            json.dump(raw, f)
        J, _ = self.run_({"1": 0.001, "2.25": 0.002})
        fam = {h["id"]: h for h in J["family"]["hypotheses"]}
        self.assertFalse(fam["robust:2.25"]["testable"])
        self.assertIn("ladder (paired prefix", fam["robust:2.25"]["note"])
        self.assertIn("10 complete seeds < floor 20", fam["robust:2.25"]["note"])
        self.assertEqual(fam["robust:2.25"]["p"], 1.0)
        self.assertTrue(fam["robust:1"]["testable"])
        raw["amplitudes"]["2.25"]["floors"] = {"ladder": 10}  # an amplitude-level override
        with open(self.paths["prereg"], "w") as f:
            json.dump(raw, f)
        self.out = os.path.join(self.root, "out_override")
        J, _ = self.run_({"1": 0.001, "2.25": 0.002})
        fam = {h["id"]: h for h in J["family"]["hypotheses"]}
        self.assertTrue(fam["robust:2.25"]["testable"])
        raw["floors"]["production"] = 100  # production floor 100 > the 40 seeds of the production arm
        with open(self.paths["prereg"], "w") as f:
            json.dump(raw, f)
        self.out = os.path.join(self.root, "out_prod")
        J, _ = self.run_({"1": 0.001, "2.25": 0.002})
        fam = {h["id"]: h for h in J["family"]["hypotheses"]}
        self.assertFalse(fam["robust:1"]["testable"])
        self.assertIn("production arm p1: 40 complete seeds < floor 100", fam["robust:1"]["note"])

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


class Isolation(World):
    """Per-hypothesis isolation (an exception or a non-finite p makes ONE hypothesis untestable, p = 1; the family stays fixed) and warnings."""

    def fam(self, J):
        return {h["id"]: h for h in J["family"]["hypotheses"]}

    def test_robust_sign_failure_at_one_amplitude(self):
        self.build()
        base, _ = self.run_({"1": 0.001, "2.25": 0.002})
        bad = "2.25/alpha_TP/" + "disp"

        class Boom(summary.RobustTest):
            def p_value(self):
                if self.label == bad:
                    raise RuntimeError("injected failure")
                return super().p_value()

        self.out = os.path.join(self.root, "out_boom")
        with mock.patch.object(summary, "RobustTest", Boom), contextlib.redirect_stderr(io.StringIO()) as err:
            J, _ = self.run_({"1": 0.001, "2.25": 0.002})
        f, f0 = self.fam(J), self.fam(base)
        self.assertEqual((f["robust:2.25"]["p"], f["robust:2.25"]["testable"], f["robust:2.25"]["rejected"], f["robust:2.25"]["status"]),
                         (1.0, False, False, "error"))
        self.assertEqual((J["family"]["info"]["m"], len(f)), (4, 4))  # the fixed family never shrinks
        self.assertEqual(f["robust:1"]["p"], f0["robust:1"]["p"])  # the other amplitude is unaffected
        self.assertEqual(f["robust:1"]["status"], "ok")  # (its Holm decision may differ: the step-down runs over the whole fixed family)
        self.assertEqual(f["gate5:2.25"]["p"], f0["gate5:2.25"]["p"])
        self.assertEqual(J["process"]["errors_count"], 1)
        e = J["process"]["errors"][0]
        self.assertEqual((e["id"], e["type"], e["message"]), ("robust:2.25", "RuntimeError", "injected failure"))
        self.assertIn("Traceback", e["traceback"])
        self.assertIn("HYPOTHESIS_ERROR robust:2.25: RuntimeError: injected failure", err.getvalue())
        self.assertEqual(J["gate4"]["2.25"]["bases"]["disp"]["estimators"]["alpha_TP"]["testable"], False)
        self.assertEqual(J["gate4"]["2.25"]["class"], "unresolved")  # untestable amplitude

    def test_gate5_failure_at_one_amplitude_and_nonfinite_p(self):
        from sf43ana import gate5
        from sf43ana.tests.test_gate5 import fake_summary
        self.build()
        cov = os.path.join(self.root, "cov.json")
        ok = {"student": {20: False, 40: False}, "hall": {20: True, 40: True}, "boott": {20: True, 40: True}}
        with open(cov, "w") as f:
            json.dump(fake_summary(ok, ns=(20, 40)), f)
        with open(self.paths["prereg"]) as f:
            raw = json.load(f)
        raw["gate5"] = {"B": 999, "seed": 3, "coverage_summary": cov, "floor": 20,
                        "amplitudes": {"1": {"consumer": "g5c", "par2": "g5p"}, "2.25": {"consumer": "g5c", "par2": "g5p"}}}
        with open(self.paths["prereg"], "w") as f:
            json.dump(raw, f)
        J0, _ = self.run_()
        orig = gate5._amplitude

        def amp(cfg, a, *args):
            if a == "1":
                raise ArithmeticError("injected gate5 failure")
            return orig(cfg, a, *args)

        self.out = os.path.join(self.root, "out_g5")
        with mock.patch.object(gate5, "_amplitude", amp), contextlib.redirect_stderr(io.StringIO()) as err:
            J, _ = self.run_()
        f, f0 = self.fam(J), self.fam(J0)
        self.assertEqual((f["gate5:1"]["p"], f["gate5:1"]["testable"], f["gate5:1"]["rejected"], f["gate5:1"]["status"]), (1.0, False, False, "error"))
        self.assertEqual((f["gate5:2.25"]["p"], f["gate5:2.25"]["testable"], f["gate5:2.25"]["status"]), (f0["gate5:2.25"]["p"], True, "ok"))
        self.assertEqual(J["family"]["info"]["m"], 4)
        self.assertEqual(J["gate5"]["amplitudes"]["1"]["status"], "error")
        self.assertEqual(J["process"]["errors_count"], 1)
        self.assertEqual((J["process"]["errors"][0]["id"], J["process"]["errors"][0]["type"]), ("gate5:1", "ArithmeticError"))
        self.assertIn("HYPOTHESIS_ERROR gate5:1: ArithmeticError", err.getvalue())
        self.assertEqual(f["robust:1"]["p"], f0["robust:1"]["p"])
        # a non-finite supplied p is the same: p = 1, "error: non-finite p"
        self.out = os.path.join(self.root, "out_nan")
        with contextlib.redirect_stderr(io.StringIO()):
            J2, _ = self.run_({"1": float("nan"), "2.25": 0.002})
        h = self.fam(J2)["gate5:1"]
        self.assertEqual((h["p"], h["testable"], h["status"]), (1.0, False, "error: non-finite p"))
        self.assertEqual(J2["process"]["errors"][0]["status"], "error: non-finite p")
        self.assertEqual(J2["family"]["info"]["m"], 4)

    def test_warnings_are_recorded_per_block_and_counted(self):
        from sf43ana import finite
        self.build()
        name = "sf43ana.g7"
        fake = types.ModuleType(name)
        fake.__spec__ = importlib.machinery.ModuleSpec(name, None)

        def analyse(ctx):
            for _ in range(3):
                warnings.warn("w one", RuntimeWarning)
            warnings.warn("w two", UserWarning)
            return {"ok": 1}
        fake.analyse = analyse
        old = sys.modules.get(name)
        sys.modules[name] = fake
        try:
            with mock.patch.object(sys, "warnoptions", []):  # as the production wrapper (-W default): every occurrence is recorded
                J, path = self.run_({"1": 0.001, "2.25": 0.002})
        finally:
            if old is None:
                del sys.modules[name]
            else:
                sys.modules[name] = old
        w = J["process"]["warnings"]["g7"]
        self.assertEqual([(x["category"], x["message"], x["count"]) for x in w], [("RuntimeWarning", "w one", 3), ("UserWarning", "w two", 1)])
        self.assertRegex(w[0]["location"], r"test_e2e\.py:\d+$")
        self.assertEqual(J["process"]["warnings_count"], 4)
        self.assertEqual(J["g7"], {"ok": 1})
        buf = io.StringIO()
        finite.run(path, buf)
        self.assertIn("process: warnings_count=4 errors_count=0", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
