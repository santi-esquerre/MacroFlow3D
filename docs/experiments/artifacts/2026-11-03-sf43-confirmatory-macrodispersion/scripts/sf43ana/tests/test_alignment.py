"""Reporting labels of Gate 5, the coverage-summary override of the CLI, and the unreduced block of the failures section."""
import contextlib
import io
import json
import os
import tempfile
import unittest

import sf43ana  # noqa: F401
from sf43ana import __main__ as cli
from sf43ana import summary
from sf43ana.tests import fixtures as fx
from sf43ana.tests.test_e2e import mean_of
from sf43ana.tests.test_gate5 import fake_summary


class GateFiveReporting(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.root = self._t.name
        self.pr = fx.prereg_dict()
        self.pr["gate5"] = {"B": 199, "seed": 3, "floor": 20, "coverage_summary": os.path.join(self.root, "missing.json"),
                            "amplitudes": {"1": {"consumer": "g5c", "par2": "g5p"}}}
        self.paths = fx.write_world(self.root, self.pr, mean_of)
        self.cov = os.path.join(self.root, "cov.json")
        with open(self.cov, "w") as f:
            json.dump(fake_summary({"student": {20: False, 40: False}, "hall": {20: True, 40: True}, "boott": {20: True, 40: True}}, ns=(20, 40)), f)

    def run_(self, coverage):
        S, _ = summary.run(self.paths["prereg"], self.paths["campaign"], self.paths["reduced"], "auto", os.path.join(self.root, "out"), coverage)
        return S

    def test_confirmatory_outcome_and_descriptive_verdict_are_labelled(self):
        S = self.run_(self.cov)
        a = S["gate5"]["amplitudes"]["1"]
        rep = a["reporting"]
        h = {x["id"]: x for x in S["family"]["hypotheses"]}["gate5:1"]
        self.assertIn("Holm TOST decision", rep["confirmatory_outcome"]["what"])
        self.assertEqual((rep["confirmatory_outcome"]["rejected"], rep["confirmatory_outcome"]["p_tost"]), (h["rejected"], h["p"]))
        self.assertIn("descriptive", rep["descriptive_verdict"]["what"])
        self.assertEqual(rep["descriptive_verdict"]["verdict"], a["verdict"])
        self.assertIn(a["verdict"], ("agree", "disagree", "inconclusive"))
        self.assertEqual(S["process"]["coverage_summary"]["path"], self.cov)
        self.assertEqual(len(S["process"]["coverage_summary"]["sha256"]), 64)

    def test_override_none_makes_gate5_descriptive_and_the_prereg_path_is_not_read(self):
        S = self.run_("none")
        a = S["gate5"]["amplitudes"]["1"]
        self.assertIsNone(a["procedure"])
        self.assertEqual(a["p_tost"], 1.0)
        self.assertIsNone(S["process"]["coverage_summary"]["path"])

    def test_without_override_the_prereg_path_is_used_and_must_exist(self):
        with self.assertRaises(FileNotFoundError):
            self.run_(None)

    def test_cli_flag(self):
        out = os.path.join(self.root, "out_cli")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            rc = cli.main(["run", "--prereg", self.paths["prereg"], "--campaign", self.paths["campaign"], "--reduced", self.paths["reduced"],
                           "--coverage-summary", self.cov, "--out", out])
        self.assertEqual(rc, 0)
        with open(os.path.join(out, "summary.json")) as f:
            self.assertEqual(json.load(f)["gate5"]["amplitudes"]["1"]["procedure"], "hall")


class UnreducedAndReestInTheSummary(unittest.TestCase):
    def test_unreduced_seeds_are_disclosed_in_process_and_failures(self):
        with tempfile.TemporaryDirectory() as root:
            pr = fx.prereg_dict()
            paths = fx.write_world(root, pr, mean_of, unreduced=[("p1", 1030)])
            S, _ = summary.run(paths["prereg"], paths["campaign"], paths["reduced"], "none", os.path.join(root, "out"))
        self.assertEqual(list(S["process"]["loader"]["unreduced"]), list(S["failures"]["unreduced_segments"]))
        self.assertTrue(S["process"]["loader"]["unreduced"])
        self.assertIn("unreduced segment", S["process"]["arms"]["p1"]["stopped_at"]["reason"])
        self.assertIn("MISSING", S["failures"]["unreduced_note"])


if __name__ == "__main__":
    unittest.main()
