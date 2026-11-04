"""``run`` / ``check`` / ``j8-trigger`` end to end on synthetic worlds (frozen world with the REAL frozen entries; campaign of the REAL addendum arms)."""
import contextlib
import io
import json
import os
import re
import shutil
import tempfile
import unittest
from unittest import mock

import sf43ana  # noqa: F401

from sf43add.ana import __main__ as cli
from sf43add.ana import rules
from sf43add.ana.tests import synth

J1_ARMS = ["j1_base10", "j1_H3", "j1_D", "j1_DH3", "j1_H2"]
SAME = {"label": {"j1_base10": "s4_g5p"}}  # j1_base10 reproduces s4_g5p value for value
CLASS_WORDS = ("horizonte", "paso dominante", "mixto", "abierto", "meseta", "decae", "inconcluso", "equivalente", "PASS", "FAIL")


def call(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = cli.main(argv)
    return rc, out.getvalue(), err.getvalue()


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = cls.tmp.name
        cls.fw = synth.frozen_world(os.path.join(cls.root, "frozen"))
        cls.add_path = os.path.join(cls.root, "add.json")
        cls.add = synth.write_addendum(cls.add_path, cls.fw["frozen_manifest"])
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def world(self, spec=None, **kw):
        type(self).n += 1
        d = os.path.join(self.root, "c%d" % self.n)
        synth.campaign_world(d, dict(SAME, **(spec or {})), add=self.add, **kw)
        return d

    def run_cli(self, campaign, j8=None, frozen=None):
        fw = frozen or self.fw
        out = os.path.join(campaign, "analysis")
        argv = ["run", "--prereg", self.add_path, "--campaign", campaign, "--frozen-reduced", fw["frozen_reduced"], "--frozen-manifest",
                fw["frozen_manifest"], "--frozen-ledger", fw["frozen_ledger"], "--frozen-segments", fw["frozen_segments"], "--out", out]
        if j8:
            argv += ["--j8-campaign", j8]
        rc, so, se = call(argv)
        S = None
        if os.path.isfile(os.path.join(out, "addendum_summary.json")):
            with open(os.path.join(out, "addendum_summary.json")) as f:
                S = json.load(f)
        return rc, so, se, S, out

    def assertStructureOnly(self, so, se):
        for text in (so, se):
            self.assertIsNone(re.search(r"\d\.\d", text), "a decimal number reached the log: %r" % text)
            for w in CLASS_WORDS:
                self.assertNotIn(w, text)


class J1EndToEnd(Base):
    def check_class(self, spec, want):
        c = self.world(dict(lp=spec), only=J1_ARMS)
        rc, so, se, S, out = self.run_cli(c)
        self.assertEqual(rc, 0, se)
        self.assertEqual(S["j1"]["n60"]["class"], want)
        self.assertTrue(S["reproduction"]["passed"])
        self.assertEqual(S["reproduction"]["csv_sha_equal"], 10)
        self.assertIn("STRUCTURE_OK", so)
        self.assertStructureOnly(so, se)
        return S

    def test_the_four_classes(self):
        # the frozen Pollock twin is 1.16 x the frozen Par2 arm, so q_X = 1.16 / factor(X)
        self.check_class({"j1_H3": 1.12, "j1_D": 1.0, "j1_DH3": 1.13, "j1_H2": 1.06}, "horizonte dominante")
        self.check_class({"j1_H3": 1.0, "j1_D": 1.08, "j1_DH3": 1.10, "j1_H2": 1.0}, "paso dominante")
        self.check_class({"j1_H3": 1.06, "j1_D": 1.06, "j1_DH3": 1.12, "j1_H2": 1.03}, "mixto")
        S = self.check_class({"j1_H3": 1.0, "j1_D": 1.0, "j1_DH3": 1.0, "j1_H2": 1.0}, "abierto")
        self.assertTrue(S["j1"]["n60"]["override"]["holds"])  # q_DH3 lower limit above 1.10, but nothing to override: `abierto` was the fall-through

    def test_amendment_a1_override_at_n60_and_n100(self):
        # class 1 would match (r_H 1.12, r_D 1.0, q_H3 1.036) but q_DH3 = 1.16 / 1.0 has its lower limit above 1.10 -> `abierto`
        c = self.world(dict(lp={"j1_H3": 1.12, "j1_D": 1.0, "j1_DH3": 1.0}), only=J1_ARMS)
        j8 = self.world(dict(lp={"j8_H3": 1.12, "j8_D": 1.0, "j8_DH3": 1.0}), job="j8_conditional")
        rc, so, se, S, out = self.run_cli(c, j8=j8)
        self.assertEqual(rc, 0, se)
        for lab in ("n60", "n100"):
            j = S["j1"][lab]
            self.assertEqual((j["class"], j["class_before_override"], j["override"]["applied"]), ("abierto", "horizonte dominante", True), lab)
        with open(os.path.join(out, "addendum_summary.md")) as f:
            self.assertIn("Amendment A1 override", f.read())
        self.assertStructureOnly(so, se)

    def test_manipulation_failure_class(self):
        c = self.world(dict(lp={"j1_H3": 1.12, "j1_DH3": 1.13}, lost={"j1_H3": 40}), only=J1_ARMS)  # 40-120 particles of 10000 never cross: mean completeness < 0.999
        rc, so, se, S, out = self.run_cli(c)
        self.assertEqual(rc, 0, se)
        self.assertEqual(S["j1"]["n60"]["class"], "abierto (horizonte no saturado)")
        self.assertFalse(S["j1"]["n60"]["manipulation"]["passed"])

    def test_files_and_trigger_inputs(self):
        c = self.world(dict(lp={"j1_H3": 1.0, "j1_D": 1.0, "j1_DH3": 1.0}), only=J1_ARMS)
        rc, so, se, S, out = self.run_cli(c)
        self.assertEqual(rc, 0)
        for f in ("addendum_summary.json", "addendum_summary.md", "reproduction.json"):
            self.assertTrue(os.path.isfile(os.path.join(out, f)))
        self.assertEqual(S["j1"]["n60"]["quantities"]["r_H"]["n"], 60)
        self.assertEqual(S["j1"]["n60"]["quantities"]["q_base"]["n"], 60)
        self.assertEqual(sorted(S["j8_trigger_inputs"]["quantities"]), ["q_DH3", "q_H3", "r_D", "r_H"])
        with open(os.path.join(out, "addendum_summary.md")) as f:
            md = f.read()
        self.assertIn("## J1 (n60)", md)
        self.assertIn("## J3", md)

    def test_reproduction_failure_blocks_the_class(self):
        c = self.world(dict(label={"j1_base10": "other"}), only=J1_ARMS)  # j1_base10 is not value for value the frozen s4_g5p
        rc, so, se, S, out = self.run_cli(c)
        self.assertEqual(rc, 0)
        self.assertFalse(S["reproduction"]["passed"])
        self.assertEqual((S["j1"]["n60"]["status"], S["j1"]["n60"]["class"]), ("not classified", None))
        self.assertIn("reproduction evaluated: yes", so)

    def test_no_reproduction_arm_means_not_evaluated(self):
        c = self.world({}, only=["j1_H3", "j1_D", "j1_DH3", "j1_H2"])
        rc, so, se, S, out = self.run_cli(c)
        self.assertEqual(rc, 0)
        self.assertFalse(S["reproduction"]["evaluated"])
        self.assertIn("reproduction evaluated: no", so)
        self.assertEqual(S["j1"]["n60"]["status"], "not classified")

    def test_replaced_seed_drops_out_of_the_paired_contrasts(self):
        c = self.world({}, only=J1_ARMS, fail=[("j1_H3", 434005)], replace=[("a1_j1_H3", 434005, 438211)])
        rc, so, se, S, out = self.run_cli(c)
        self.assertEqual(rc, 0, se)
        q = S["j1"]["n60"]["quantities"]
        self.assertEqual((q["r_H"]["n"], q["r_D"]["n"]), (59, 60))
        self.assertEqual(S["arms"]["j1_H3"]["N"], 60)  # the reserve takes the position
        self.assertIn(438211, range(S["arms"]["j1_H3"]["seed_range"][0], S["arms"]["j1_H3"]["seed_range"][1] + 1))

    def test_j8_extension_reports_n100_and_the_trigger_cli_reads_the_summary(self):
        c = self.world(dict(lp={"j1_H3": 1.05, "j1_D": 1.0, "j1_DH3": 1.0}), only=J1_ARMS)
        j8 = self.world(dict(lp={"j8_H3": 1.05, "j8_D": 1.0, "j8_DH3": 1.0}), job="j8_conditional")
        rc, so, se, S, out = self.run_cli(c, j8=j8)
        self.assertEqual(rc, 0, se)
        n100 = S["j1"]["n100"]
        self.assertEqual((n100["quantities"]["r_H"]["n"], n100["quantities"]["q_H3"]["n"], n100["quantities"]["q_base"]["n"]), (100, 100, 100))
        self.assertFalse(n100["quantities"]["r_H2"]["evaluable"])  # no j8 arm for H2
        self.assertEqual(S["j1"]["n60"]["quantities"]["r_H"]["n"], 60)  # n = 60 stays the primary
        self.assertIn("j1:n100:r_H=100", so)
        rc, so, se = call(["j8-trigger", "--summary", os.path.join(out, "addendum_summary.json"), "--gpu-h-used", "5", "--j8-gpu-h-projected", "0.5",
                           "--j8-end-utc", "2026-11-06T20:00:00Z"])
        self.assertEqual(rc, 0)
        self.assertIn(so.split("\n")[0], ("J8 RUN", "J8 SKIP"))
        self.assertEqual(len([ln for ln in so.splitlines() if ln.startswith("(")]), 4)


class FullCampaign(Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.camp = os.path.join(cls.root, "full")
        synth.campaign_world(cls.camp, dict(SAME, lp={"j1_H3": 1.12, "j1_D": 1.0, "j1_DH3": 1.12, "j1_H2": 1.06}, far={"j3_l5": 1.02, "j9_w48": 1.02}),
                             add=cls.add)
        cls.rc, cls.so, cls.se, cls.S, cls.out = cls.run_cli(cls, cls.camp)

    def test_all_rules_have_a_class(self):
        self.assertEqual(self.rc, 0, self.se)
        self.assertEqual(self.S["j3"]["class"], "lambda/h = 5 equivalente")
        self.assertEqual(self.S["j9"]["class"], "meseta robusta al ancho")
        self.assertEqual(self.S["j9"]["n_completed"], 150)
        self.assertEqual((self.S["j1"]["n60"]["status"], self.S["j1"]["n60"]["class"]), ("classified", "horizonte dominante"))

    def test_stdout_is_structure_only(self):
        self.assertIn("STRUCTURE_OK", self.so)
        self.assertStructureOnly(self.so, self.se)
        self.assertIn("j9_w48 realizations=150", self.so)
        self.assertIn("frozen files verified:", self.so)
        self.assertIn("warnings=0 errors=0", self.so)

    def test_summary_is_complete_json(self):
        for k in ("arms", "inputs", "reproduction", "j1", "j3", "j9", "j8_trigger_inputs", "process", "config"):
            self.assertIn(k, self.S)
        self.assertEqual(self.S["process"]["errors_count"], 0)
        self.assertGreater(self.S["inputs"]["frozen_files_verified"], 0)
        self.assertEqual(self.S["arms"]["s4_g5p"]["source"], "frozen")


class Failures(Base):
    def test_a_tampered_frozen_file_is_refused(self):
        fw = synth.frozen_world(os.path.join(self.root, "tampered"))
        victim = os.path.join(fw["frozen_reduced"], "s4_g5p")
        victim = os.path.join(victim, sorted(os.listdir(victim))[0])
        victim = os.path.join(victim, [f for f in sorted(os.listdir(victim)) if f.endswith("d25-94.csv")][0])
        with open(victim, "a") as f:
            f.write("\n")
        c = self.world({}, only=["j1_H3"])
        rc, so, se, S, out = self.run_cli(c, frozen=fw)
        self.assertEqual(rc, 2)
        self.assertIn("input refused", se)
        self.assertIsNone(S)

    def test_a_manifest_that_is_not_the_pre_registered_one_is_refused(self):
        fw = dict(self.fw, frozen_manifest=os.path.join(self.root, "other.sha256"))
        shutil.copy(self.fw["frozen_manifest"], fw["frozen_manifest"])
        with open(fw["frozen_manifest"], "a") as f:
            f.write("%s  ./x/y/z\n" % ("0" * 64))
        rc, so, se, S, out = self.run_cli(self.world({}, only=["j1_H3"]), frozen=fw)
        self.assertEqual(rc, 2)

    def test_a_block_error_is_recorded_and_fails_the_structure(self):
        c = self.world({}, only=["j1_H3"])
        with mock.patch.object(rules, "evaluate_j3", side_effect=RuntimeError("boom 3.14159")):
            rc, so, se, S, out = self.run_cli(c)
        self.assertEqual(rc, 1)
        self.assertIn("STRUCTURE_FAIL", so)
        self.assertEqual(S["process"]["errors_count"], 1)
        self.assertIn("error", S["j3"])
        self.assertNotIn("3.14159", so + se)  # the log carries the block name and the exception type, not the message
        self.assertIn("j1", S)  # the other blocks still ran

    def test_a_campaign_without_ledger_is_refused(self):
        c = os.path.join(self.root, "empty")
        os.makedirs(os.path.join(c, "reduced"))
        rc, so, se, S, out = self.run_cli(c)
        self.assertEqual(rc, 2)

    def test_the_sentinel_value_never_reaches_the_log(self):
        old = dict(synth.MEAN)
        synth.MEAN.update(alpha_LP=0.31415926, alpha_TP=0.27182818)
        try:
            with tempfile.TemporaryDirectory() as d:
                fw = synth.frozen_world(os.path.join(d, "f"))
                add = synth.write_addendum(os.path.join(d, "a.json"), fw["frozen_manifest"])
                c = os.path.join(d, "c")
                synth.campaign_world(c, SAME, add=add, only=J1_ARMS + ["j3_l5"])
                out = os.path.join(d, "o")
                rc, so, se = call(["run", "--prereg", os.path.join(d, "a.json"), "--campaign", c, "--frozen-reduced", fw["frozen_reduced"],
                                   "--frozen-manifest", fw["frozen_manifest"], "--frozen-ledger", fw["frozen_ledger"], "--out", out])
                with open(os.path.join(out, "addendum_summary.json")) as f:
                    text = f.read()
                S = json.loads(text)
                self.assertAlmostEqual(S["j1"]["n60"]["quantities"]["r_H"]["mean_den"], 0.31415926, delta=0.01)  # the sentinel level is in the analysed data ...
                self.assertEqual(rc, 0, se)
                for s in (so, se):
                    self.assertNotIn("3141", s)  # ... and not in the log
                    self.assertNotIn("2718", s)
                self.assertStructureOnly(so, se)
        finally:
            synth.MEAN.clear()
            synth.MEAN.update(old)


class SmokeCheck(Base):
    def smoke(self, **kw):
        units = self.add["smoke"]["units"]
        d = os.path.join(self.root, "smoke%d" % len(os.listdir(self.root)))
        synth.campaign_world(d, SAME, add=self.add, seeds=units, only=list(units), **kw)
        rc, so, se = call(["check", "--prereg", self.add_path, "--campaign", d, "--out", os.path.join(d, "analysis")])
        return rc, so, se, d

    def test_structure_ok(self):
        rc, so, se, d = self.smoke()
        self.assertEqual((rc, se), (0, ""), so)
        self.assertIn("STRUCTURE_OK", so)
        self.assertIn("arm j9_w48: planned=1 ledger_completed=1 loader_prefix=1 finite_rows=1", so)
        self.assertIn("arm j1_H3: planned=2 ledger_completed=2 loader_prefix=2 finite_rows=2 crossing_counts=2", so)
        self.assertIsNone(re.search(r"\d\.\d", so))
        with open(os.path.join(d, "analysis", "smoke_check.json")) as f:
            self.assertEqual({a["arm"] for a in json.load(f)["arms"]}, set(self.add["smoke"]["units"]))

    def test_an_unusable_solve_fails_the_structure(self):
        rc, so, se, d = self.smoke(fail=[("j1_H3", 434002)])
        self.assertEqual(rc, 1)
        self.assertIn("STRUCTURE_FAIL", so)
        self.assertIn("j1_H3", so.splitlines()[-1])

    def test_a_missing_smoke_arm_fails_the_structure(self):
        rc, so, se, d = self.smoke(counts={"j9_w48": 0})
        self.assertEqual(rc, 1)
        self.assertIn("arm j9_w48: planned=1 ledger_completed=0", so)


if __name__ == "__main__":
    unittest.main()
