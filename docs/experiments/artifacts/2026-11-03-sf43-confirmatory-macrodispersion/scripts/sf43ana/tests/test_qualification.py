"""The qualification gate of the robust sign at the realized sizes (decision H4 with H13 against H15): ``sf43ana.qualification`` and its use in ``summary``."""
import contextlib
import hashlib
import io
import json
import os
import unittest

import sf43ana  # noqa: F401
from sf43ana import __main__ as cli
from sf43ana import qualification as Q
from sf43ana.tests import fixtures as fx
from sf43ana.tests.test_e2e import World


class Rule(unittest.TestCase):
    SIZES = {"production": 100, "ladder": 75, "width": 75, "h_max": 20}

    def doc(self, **kw):
        return fx.qualification_doc(profiles=((3, 100, 50, 75, 20), (4, 100, 75, 75, 20), (4, 200, 150, 150, 30)), sampling=(100, 150, 200), **kw)

    def test_profiles_and_grid_aggregate_the_cells(self):
        d = self.doc(qualified={(4, 100, 75, 75, 20): False, 150: False})
        prof = {(p["K"], p["arm_sizes"]["production"]): p for p in Q.pipeline_profiles(d)}
        self.assertEqual({k: p["qualified"] for k, p in prof.items()}, {(3, 100): True, (4, 100): False, (4, 200): True})
        self.assertEqual(prof[(4, 100)]["cells"], 2)
        diag = {(c["n"], c["dist"], c["side"]): c for c in Q.sampling_diagnostic(d)}
        self.assertEqual({n: diag[(n, "normal", "lower")]["qualified"] for n in (100, 150, 200)}, {100: True, 150: False, 200: True})
        self.assertEqual(diag[(150, "normal", "lower")]["max_ucl_over_tail"], 1.3)
        d["pipeline"][0]["qualified"] = None  # a not-judged cell: the profile is not qualified
        self.assertIsNone(Q.pipeline_profiles(d)[0]["qualified"])

    def test_testable_at_the_largest_eligible_profile(self):
        st, det = Q.check(self.doc(), 4, self.SIZES)
        self.assertIsNone(st)
        self.assertEqual(det["pipeline_profile"]["arm_sizes"], {"production": 100, "ladder": 75, "width": 75, "h_max": 20})
        st, det = Q.check(self.doc(), 4, dict(self.SIZES, production=250, ladder=200, width=200, h_max=40))
        self.assertIsNone(st)
        self.assertEqual(det["pipeline_profile"]["arm_sizes"]["production"], 200)  # the largest that fits
        st, det = Q.check(self.doc(), 3, dict(self.SIZES, ladder=60))
        self.assertIsNone(st)
        self.assertEqual(det["pipeline_profile"]["K"], 3)

    def test_not_qualified_cases(self):
        for why, K, sizes, kw in (
                ("a smaller arm than every profile", 4, dict(self.SIZES, width=74), {}),
                ("production below the smallest profile", 4, dict(self.SIZES, production=99), {}),
                ("another structure", 5, self.SIZES, {}),
                ("the largest eligible profile fails", 4, self.SIZES, {"qualified": {(4, 100, 75, 75, 20): False}}),
                ("an incomplete document", 4, self.SIZES, {"complete": False})):
            st, det = Q.check(self.doc(**kw), K, sizes)
            self.assertEqual(st, Q.NOT_QUALIFIED, why)
            self.assertTrue(det["reasons"], why)

    def test_the_sampling_limit_is_reported_but_never_gates(self):
        # H15: pipeline true + sampling false (every grid point) -> TESTABLE; pipeline false + sampling true -> not qualified
        d = self.doc(qualified={100: False, 150: False, 200: False})
        self.assertTrue(all(p["qualified"] for p in Q.pipeline_profiles(d)))
        self.assertIsNone(Q.check(d, 4, self.SIZES)[0])
        self.assertEqual(Q.check(d, 3, self.SIZES)[0], None)
        self.assertEqual(Q.check(d, 4, dict(self.SIZES, production=50, ladder=50, width=50, h_max=10))[0], Q.NOT_QUALIFIED)  # no eligible profile
        d = self.doc(qualified={(4, 100, 75, 75, 20): False})
        st, det = Q.check(d, 4, self.SIZES)
        self.assertEqual(st, Q.NOT_QUALIFIED)
        self.assertNotIn("sampling", " ".join(det["reasons"]))
        self.assertTrue(all(not c["qualified"] for c in Q.sampling_diagnostic(self.doc(qualified={100: False, 150: False, 200: False})) if c["side"] == "lower"))

    def test_a_smaller_qualified_profile_does_not_rescue_a_failing_larger_eligible_one(self):
        d = self.doc(qualified={(4, 100, 75, 75, 20): False}, extra=((4, 60, 40, 40, 10),))
        self.assertEqual(Q.check(d, 4, self.SIZES)[0], Q.NOT_QUALIFIED)

    def test_a_component_the_structure_lacks_constrains_no_profile(self):
        st, _ = Q.check(self.doc(), 4, dict(self.SIZES, h_max=None))
        self.assertIsNone(st)

    def test_no_document(self):
        self.assertEqual(Q.check(None, 3, self.SIZES)[0], Q.NO_DOC)
        doc, info = Q.load(None)
        self.assertEqual((doc, info["found"]), (None, False))
        doc, info = Q.load("/no/such/qualification.json")
        self.assertEqual((doc, info["found"], info["sha256"]), (None, False, None))


class InTheAnalysis(World):
    def fam(self, J):
        return {h["id"]: h for h in J["family"]["hypotheses"]}

    def set_doc(self, doc):
        with open(os.path.join(self.root, "qualification.json"), "w") as f:
            json.dump(doc, f)

    def test_qualified_sizes_leave_the_robust_sign_testable_and_the_document_is_hashed(self):
        self.build()
        J, _ = self.run_({"1": 0.001, "2.25": 0.002})
        f = self.fam(J)
        self.assertTrue(f["robust:2.25"]["rejected"])
        self.assertEqual((f["robust:1"]["status"], f["robust:1"]["testable"]), ("ok", True))
        q = J["process"]["qualification"]
        path = os.path.join(self.root, "qualification.json")
        with open(path, "rb") as fh:
            self.assertEqual((q["found"], q["sha256"]), (True, hashlib.sha256(fh.read()).hexdigest()))
        g = J["gate4"]["2.25"]["qualification"]
        self.assertEqual((g["status"], g["pipeline_profile"]["arm_sizes"]["production"]), (None, 20))
        self.assertEqual({(c["n"], c["side"]) for c in q["sampling_diagnostic"] if c["dist"] == "normal"}, {(20, "lower"), (20, "upper")})
        self.assertEqual(g["realized_sizes"]["production"], 40)

    def test_unqualified_sizes_give_p_one_and_the_status_but_the_estimates_stay(self):
        self.build()
        self.set_doc(fx.qualification_doc(qualified={(3, 20, 20, 20, 10): False}))
        J, _ = self.run_({"1": 0.001, "2.25": 0.002})
        f = self.fam(J)
        for amp in ("1", "2.25"):
            h = f["robust:" + amp]
            self.assertEqual((h["p"], h["testable"], h["rejected"], h["status"]), (1.0, False, False, "not qualified at the realized sizes"), amp)
            self.assertIn("is not qualified", h["note"])
            self.assertEqual(J["gate4"][amp]["qualification"]["status"], "not qualified at the realized sizes")
        self.assertEqual(J["family"]["info"]["m"], 4)                             # the fixed family never shrinks
        self.assertTrue(f["gate5:2.25"]["rejected"])                               # the Gate 5 hypotheses (p 0.001, 0.002) come first in the step-down
        self.assertEqual(J["estimands"]["1"]["status"], "ok")
        self.assertTrue(J["budgets"]["2.25"]["disp"]["alpha_TP"]["testable"])      # the budgets stay descriptive
        self.assertTrue(J["gate4"]["2.25"]["bases"]["disp"]["estimators"]["alpha_TP"]["testable"])

    def test_sampling_limit_not_qualified_is_reported_but_the_robust_sign_stays_testable(self):
        self.build()
        self.set_doc(fx.qualification_doc(sampling=(20, 30), qualified={30: False}))  # pipeline qualified, sampling limit fails at n = 30
        J, _ = self.run_({"1": 0.001, "2.25": 0.002})
        h = self.fam(J)["robust:1"]
        self.assertEqual((h["status"], h["testable"]), ("ok", True))
        bad = [c for c in J["process"]["qualification"]["sampling_diagnostic"] if not c["qualified"]]
        self.assertEqual([(c["n"], c["side"], c["max_ucl_over_tail"]) for c in bad if c["dist"] == "normal"], [(30, "lower", 1.3)])

    def test_no_document_is_p_one_with_its_own_status(self):
        self.build(qualification=None)
        J, _ = self.run_({"1": 0.001, "2.25": 0.002})
        f = self.fam(J)
        for amp in ("1", "2.25"):
            self.assertEqual((f["robust:" + amp]["p"], f["robust:" + amp]["status"], f["robust:" + amp]["testable"]), (1.0, "no qualification document", False))
        self.assertEqual(J["process"]["qualification"], {"path": None, "found": False, "sha256": None, "plan_hash": None, "all_cells_complete": None})
        # a path that does not exist is the same: no document
        with open(self.paths["prereg"]) as fh:
            raw = json.load(fh)
        raw["qualification"] = os.path.join(self.root, "missing.json")
        with open(self.paths["prereg"], "w") as fh:
            json.dump(raw, fh)
        self.out = os.path.join(self.root, "o2")
        J, _ = self.run_({"1": 0.001, "2.25": 0.002})
        self.assertEqual(self.fam(J)["robust:1"]["status"], "no qualification document")

    def test_cli_override_and_strict_schema(self):
        self.build()
        for arg, status in (("none", "no qualification document"), (os.path.join(self.root, "qualification.json"), "ok")):
            out = os.path.join(self.root, "cli_" + status[:2])
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(cli.main(["run", "--prereg", self.paths["prereg"], "--campaign", self.paths["campaign"], "--reduced", self.paths["reduced"],
                                           "--gate5", "none", "--qualification", arg, "--out", out]), 0)
            with open(os.path.join(out, "summary.json")) as fh:
                S = json.load(fh)
            self.assertEqual({h["id"]: h for h in S["family"]["hypotheses"]}["robust:1"]["status"], status)
        with open(self.paths["prereg"]) as fh:
            raw = json.load(fh)
        raw["qualification"] = 3
        with open(self.paths["prereg"], "w") as fh:
            json.dump(raw, fh)
        with contextlib.redirect_stderr(io.StringIO()) as err:
            rc = cli.main(["run", "--prereg", self.paths["prereg"], "--campaign", self.paths["campaign"], "--reduced", self.paths["reduced"],
                           "--gate5", "none", "--out", os.path.join(self.root, "bad")])
        self.assertEqual(rc, 2)
        self.assertIn("qualification must be a path", err.getvalue())


if __name__ == "__main__":
    unittest.main()
