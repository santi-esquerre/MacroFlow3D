"""Tests of sf39.regress_existing (addendum A6) on small fixtures and on the committed evidence analysis: identical -> 0, additions listed and
accepted, every kind of change -> 1, every guard against a vacuous pass -> 2 (nothing compared)."""
import contextlib
import copy
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from sf39 import regress_existing as rg

AMPS = ("s1", "s225", "s4")
HEAD = "tier,amplitude,estimator,decision_point,verdict,ci_lo,ci_hi,n,note"
REF_ROWS = [
    "primary,s1,alpha_TAM,Gate 4 label (D),physical (positive),0.0074,0.0115,50,branch 4; mean=0.0095",
    "primary,s225,alpha_TAM,Gate 4 label (D),physical (positive),0.02,0.03,62,\"branch 4; a, b\"",
    "primary,s4,alpha_LAM,Gate 5 (C),inconclusive,-0.1,-0.05,71/71,diff=-0.1",
    "estimation,s1,alpha_TP,F7 Delta_half,decreasing,0.1,0.2,50,mean=0.15",
    "process,all,,CHECK_unittest,ok,,,,rc=0 (from checks/results.txt)",
    "process,all,,CHECK_geometry,ok,,,,rc=0 (from checks/results.txt)",
    "process,all,,CHECK_bytecheck,ok,,,,rc=0 (from checks/results.txt)",
]
ANALYSIS = Path(__file__).resolve().parents[3] / "analysis"  # the committed evidence analysis (evidence commit 4dbf31f2)


def summary(amps=AMPS):
    return {"schema": "sf39-analysis-v1", "partial": False, "n_pilot": 20,
            "budget": {a: {"N_R": 20 + i, "N_A": 20} for i, a in enumerate(amps)},
            "amplitudes": {a: {"A": {"mean": 0.01 * (i + 1), "lo": None, "hi": "inf", "ok": True, "n": 20 + i}, "seeds": [5001, 5002]}
                           for i, a in enumerate(amps)}}


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write(text)


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_manifest(d, drop=(), extra=()):
    """manifest.txt in the format of `analyze` (sha256sum): every file of d but the manifest, plus bookkeeping entries completed after it
    (listed with a hash that does not match, one of them for a file that does not exist)."""
    lines = []
    for r, _, fs in os.walk(d):
        for f in fs:
            rel = os.path.relpath(os.path.join(r, f), d)
            if rel != "manifest.txt" and rel not in drop:
                lines.append("%s  %s" % (sha(os.path.join(d, rel)), rel))
    lines += ["%s  logs/analysis.log" % ("0" * 64), "%s  results.txt" % ("1" * 64), "%s  PART_EXIT" % ("2" * 64)] + list(extra)
    write(os.path.join(d, "manifest.txt"), "\n".join(sorted(lines)) + "\n")
    write(os.path.join(d, "results.txt"), "bookkeeping completed after the manifest\n")


def make(d, rows=REF_ROWS, summ=None, amps=AMPS):
    write(os.path.join(d, "summary.json"), json.dumps(summ or summary(amps), indent=1, sort_keys=True) + "\n")
    for a in amps:
        write(os.path.join(d, "reduced", "%s_prod.csv" % a), "set,seed,v\n%s_prod,5001,1.5\n%s_prod,5002,2.5\n" % (a, a))
        write(os.path.join(d, "series", "alpha_t_%s.csv" % a), "t,alpha\n1,0.1\n2,0.2\n")
    write(os.path.join(d, "reduced", "rho_s4.csv"), "pair,rho\n8-16,0.1\n")
    write(os.path.join(d, "decisions.csv"), "\n".join([HEAD] + rows) + "\n")
    write_manifest(d)


def run(ref, new, *extra):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = rg.main(["--ref", ref, "--new", new, *extra])
    return rc, out.getvalue(), err.getvalue()


class Fixture(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ref = os.path.join(self.tmp.name, "ref")
        self.new = os.path.join(self.tmp.name, "new")
        make(self.ref)
        shutil.copytree(self.ref, self.new)

    def edit(self, rel, fn, root=None):
        p = os.path.join(root or self.new, rel)
        with open(p, newline="") as f:
            t = f.read()
        with open(p, "w", newline="") as f:
            f.write(fn(t))

    def edit_summary(self, fn):
        p = os.path.join(self.new, "summary.json")
        with open(p) as f:
            d = json.load(f)
        fn(d)
        with open(p, "w") as f:
            f.write(json.dumps(d, indent=1, sort_keys=True) + "\n")


class Identical(Fixture):
    def test_identical_is_zero_and_prints_the_manifest_sha(self):
        rc, out, _ = run(self.ref, self.new)
        self.assertEqual(rc, 0, out)
        self.assertIn("REGRESS_EXISTING IDENTICAL (s1,s225,s4)", out)
        self.assertIn("csv files: 4 reduced + 3 series compared", out)
        self.assertIn("reference manifest sha256 %s: 4 reduced/ entries" % sha(os.path.join(self.ref, "manifest.txt")), out)
        # the bookkeeping entries of the manifest do not verify (by construction) and are ignored
        self.assertIn("0" * 64 + "  logs/analysis.log", Path(self.ref, "manifest.txt").read_text())

    def test_additions_are_listed_and_never_fail(self):
        # what the s025 run adds: an amplitude in summary / budget / decisions, new csv files, new process rows (also between old ones)
        self.edit_summary(lambda d: (d["amplitudes"].update(s025={"A": {"mean": 1.0}}), d["budget"].update(s025={"N_R": 20})))
        write(os.path.join(self.new, "reduced", "s025_prod.csv"), "set,seed,v\ns025_prod,5001,1\n")
        write(os.path.join(self.new, "series", "alpha_t_s025.csv"), "t,alpha\n1,1\n")
        rows = REF_ROWS[:3] + ["primary,s025,alpha_TAM,Gate 4 label (D),physical (bounded),,,20,x"] + REF_ROWS[3:5] + [
            "process,all,,CHECK_solve_chk_s025_n05,converged,,,,rc=0 (from checks/results.txt)"] + REF_ROWS[5:]
        write(os.path.join(self.new, "decisions.csv"), "\n".join([HEAD] + rows) + "\n")
        rc, out, _ = run(self.ref, self.new)
        self.assertEqual(rc, 0, out)
        self.assertIn("ADDED decisions.csv new rows of s025: 1", out)
        self.assertIn("ADDED decisions.csv new process row: process,all,,CHECK_solve_chk_s025_n05", out)
        self.assertIn("ADDED reduced/*.csv new files: 1", out)
        self.assertNotIn("DIFFERENT", out)

    def test_the_amplitudes_are_fixed(self):
        for extra in (("--amps", "s1"), ("--amps", "s1,s225,s4")):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as cm:
                run(self.ref, self.new, *extra)
            self.assertEqual(cm.exception.code, 2)


class Different(Fixture):
    def assertFails(self, needle):
        rc, out, _ = run(self.ref, self.new)
        self.assertEqual(rc, 1, out)
        self.assertIn("DIFFERENT", out)
        self.assertIn(needle, out)
        self.assertIn("REGRESS_EXISTING FAILED", out)

    def test_a_one_changed_value_in_summary(self):
        self.edit_summary(lambda d: d["amplitudes"]["s225"]["A"].update(mean=0.02000000000000001))
        self.assertFails("summary.json differs at amplitudes.s225.A.mean")

    def test_a_budget_changed(self):
        self.edit_summary(lambda d: d["budget"]["s4"].update(N_R=999))
        self.assertFails("budget.s4.N_R")

    def test_a_types_and_structure(self):
        self.edit_summary(lambda d: d["amplitudes"]["s1"]["A"].update(ok=1))  # true -> 1: bool is not a number
        self.assertFails("amplitudes.s1.A.ok")
        self.setUp()
        self.edit_summary(lambda d: d["amplitudes"]["s1"].update(extra=1))
        self.assertFails("amplitudes.s1 keys ['extra']")
        self.setUp()
        self.edit_summary(lambda d: d["amplitudes"]["s4"]["seeds"].append(5003))
        self.assertFails("amplitudes.s4.seeds length 2 != 3")
        self.setUp()
        self.edit_summary(lambda d: d["amplitudes"].pop("s4"))
        self.assertFails("summary.json amplitudes.s4 missing")

    def test_b_reduced_csv_byte(self):
        self.edit("reduced/s1_prod.csv", lambda t: t.replace("1.5", "1.5000000001"))
        self.assertFails("reduced/s1_prod.csv differs (bytes)")

    def test_b_reduced_csv_missing_and_series(self):
        os.remove(os.path.join(self.new, "reduced", "rho_s4.csv"))
        self.assertFails("reduced/rho_s4.csv missing")
        self.setUp()
        self.edit("series/alpha_t_s225.csv", lambda t: t + "3,0.3\n")
        self.assertFails("series/alpha_t_s225.csv differs")

    def test_c_decisions_row_changed_or_removed_or_reordered(self):
        self.edit("decisions.csv", lambda t: t.replace("0.0074", "0.0075"))
        self.assertFails("decisions.csv rows of s1,s225,s4 differ")
        self.setUp()
        self.edit("decisions.csv", lambda t: t.replace("primary,s4,alpha_LAM,Gate 5 (C),inconclusive,-0.1,-0.05,71/71,diff=-0.1\n", ""))
        self.assertFails("decisions.csv rows of s1,s225,s4 differ")
        self.setUp()
        rows = [REF_ROWS[1], REF_ROWS[0]] + REF_ROWS[2:]
        write(os.path.join(self.new, "decisions.csv"), "\n".join([HEAD] + rows) + "\n")
        self.assertFails("decisions.csv rows of s1,s225,s4 differ")

    def test_c_process_rows_must_be_present_in_order(self):
        rows = REF_ROWS[:4] + [REF_ROWS[5], REF_ROWS[4], REF_ROWS[6]]  # CHECK_geometry before CHECK_unittest
        write(os.path.join(self.new, "decisions.csv"), "\n".join([HEAD] + rows) + "\n")
        self.assertFails("process row of the reference not found in order")
        self.setUp()
        write(os.path.join(self.new, "decisions.csv"), "\n".join([HEAD] + REF_ROWS[:6]) + "\n")  # CHECK_bytecheck dropped
        self.assertFails("CHECK_bytecheck")
        self.setUp()
        self.edit("decisions.csv", lambda t: t.replace("CHECK_geometry,ok", "CHECK_geometry,FAILED"))
        self.assertFails("CHECK_geometry")

    def test_c_header_changed(self):
        self.edit("decisions.csv", lambda t: t.replace("decision_point", "decision"))
        self.assertFails("header differs")


class Refusals(Fixture):
    """Every guard exits 2, says why, and compares nothing."""

    def assertRefused(self, needle, ref=None, new=None):
        rc, out, err = run(ref or self.ref, new or self.new)
        self.assertEqual(rc, 2, out + err)
        self.assertIn("REGRESS_EXISTING_REFUSED", err)
        self.assertIn(needle, err)
        self.assertNotIn("IDENTICAL", out)
        self.assertNotIn("DIFFERENT", out)

    def test_same_directory_even_through_a_link(self):
        self.assertRefused("same directory", new=self.ref)
        link = os.path.join(self.tmp.name, "link")
        os.symlink(self.ref, link)
        self.assertRefused("same directory", new=link)
        self.assertRefused("same directory", ref=link, new=self.ref)

    def test_reference_without_manifest(self):
        os.remove(os.path.join(self.ref, "manifest.txt"))
        self.assertRefused("no manifest.txt")

    def test_malformed_manifest(self):
        self.edit("manifest.txt", lambda t: t + "not a sha256sum line\n", root=self.ref)
        self.assertRefused("malformed manifest line")

    def test_reference_file_not_matching_the_manifest(self):
        for rel, fn in (("reduced/s1_prod.csv", lambda t: t.replace("1.5", "1.6")), ("series/alpha_t_s225.csv", lambda t: t + "9,9\n"),
                        ("summary.json", lambda t: t.replace("0.01", "0.02")), ("decisions.csv", lambda t: t.replace("0.0074", "0.0075"))):
            self.setUp()
            self.edit(rel, fn, root=self.ref)
            self.edit(rel, fn, root=self.new)  # an identical change on both sides must not hide it
            self.assertRefused("%s does not match its manifest sha256" % rel)

    def test_listed_but_missing_and_not_listed(self):
        os.remove(os.path.join(self.ref, "series", "alpha_t_s4.csv"))
        self.assertRefused("series/alpha_t_s4.csv is listed but missing")
        self.setUp()
        write_manifest(self.ref, drop=("series/alpha_t_s4.csv",))
        self.assertRefused("series/alpha_t_s4.csv is not listed in the manifest")
        self.setUp()
        write_manifest(self.ref, drop=("summary.json",))
        self.assertRefused("summary.json is not listed in the manifest")

    def test_reduced_count_must_match_the_manifest(self):
        write(os.path.join(self.ref, "reduced", "extra.csv"), "a,b\n1,2\n")  # a reduced file the manifest does not list
        self.assertRefused("holds 5 reduced/*.csv but its manifest lists 4")
        self.setUp()
        write_manifest(self.ref, extra=["%s  reduced/ghost.csv" % ("3" * 64)])  # a listed entry with no file
        self.assertRefused("holds 4 reduced/*.csv but its manifest lists 5")

    def test_no_reduced_csv(self):
        shutil.rmtree(os.path.join(self.ref, "reduced"))
        write_manifest(self.ref)
        self.assertRefused("no reduced/*.csv")
        os.makedirs(os.path.join(self.ref, "reduced"))
        self.assertRefused("no reduced/*.csv")

    def test_unreadable_input_is_2(self):
        rc, _, err = run(self.ref, os.path.join(self.tmp.name, "absent"))
        self.assertEqual(rc, 2)
        self.assertIn("REGRESS_EXISTING_ERROR", err)


class CommittedEvidence(unittest.TestCase):
    """The real committed analysis: it verifies against its own manifest (53 reduced/ entries) and an exact copy is reproduced."""

    def test_committed_analysis_against_a_copy(self):
        with tempfile.TemporaryDirectory() as t:
            new = os.path.join(t, "new")
            shutil.copytree(ANALYSIS, new)
            rc, out, err = run(str(ANALYSIS), new)
            self.assertEqual((rc, err), (0, ""), out)
            self.assertIn("53 reduced/ entries", out)
            self.assertIn("csv files: 53 reduced + 3 series compared", out)
            self.assertIn("reference manifest sha256 %s" % sha(str(ANALYSIS / "manifest.txt")), out)
            with open(os.path.join(new, "reduced", "s1_prod.csv"), "a") as f:
                f.write("x")
            rc, out, err = run(str(ANALYSIS), new)
            self.assertEqual(rc, 1, out)
            self.assertIn("reduced/s1_prod.csv differs (bytes)", out)


class JsonDiff(unittest.TestCase):
    def test_json_diff(self):
        a = {"x": [1, {"y": None, "z": "inf"}], "b": True}
        self.assertIsNone(rg.json_diff(a, copy.deepcopy(a)))
        self.assertEqual(rg.json_diff(a, {"x": [1, {"y": None, "z": "-inf"}], "b": True}), "$.x[1].z")
        self.assertEqual(rg.json_diff(a, {"x": [1, {"y": 0, "z": "inf"}], "b": True}), "$.x[1].y")
        self.assertEqual(rg.json_diff(a, {"x": [1, {"y": None, "z": "inf"}], "b": 1}), "$.b")
        self.assertIsNone(rg.json_diff({"v": 1}, {"v": 1.0}))  # parsed-JSON equality: 1 == 1.0


if __name__ == "__main__":
    unittest.main()
