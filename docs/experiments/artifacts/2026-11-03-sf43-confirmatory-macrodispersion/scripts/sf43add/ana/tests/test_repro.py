"""The reproduction check (3.1.1) on synthetic worlds: tolerances, strictness, the crossing-CSV sha256, the CLI line."""
import contextlib
import csv
import io
import os
import re
import shutil
import tempfile
import unittest

import sf43ana  # noqa: F401
from sf43ana.loader import LoadError

from sf43add.ana import __main__ as cli
from sf43add.ana import repro
from sf43add.ana.tests import synth

SAME = {"label": {"j1_base10": "s4_g5p"}}
LINE = re.compile(r"^REPRO (PASS|FAIL) \d+ \d+ \S+ csv_sha_equal=\d+/\d+$")


def edit_table(path, fn):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    fn(rows)
    with open(path, "w", newline="") as f:
        csv.writer(f, lineterminator="\n").writerows(rows)


class Repro(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.fw = synth.frozen_world(os.path.join(cls.tmp.name, "frozen"))
        cls.base = os.path.join(cls.tmp.name, "base")
        synth.campaign_world(cls.base, SAME, only=["j1_base10"])
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def camp(self):
        type(self).n += 1
        d = os.path.join(self.tmp.name, "k%d" % self.n)
        shutil.copytree(self.base, d)
        return d

    def tables(self, camp, suffix="d25-94.csv"):
        out = []
        for seg in sorted(os.listdir(os.path.join(camp, "reduced", "j1_base10"))):
            out += [os.path.join(camp, "reduced", "j1_base10", seg, f) for f in sorted(os.listdir(os.path.join(camp, "reduced", "j1_base10", seg)))
                    if f.endswith(suffix) and not f.endswith(".planes.csv")]
        return out

    def compare(self, camp, seeds="434001-434010", segments=True, **kw):
        return repro.compare(camp, "j1_base10", repro.parse_seeds(seeds), self.fw["frozen_reduced"], self.fw["frozen_manifest"],
                             self.fw["frozen_segments"] if segments else None, **kw)

    def test_identical_campaign_passes_with_bitwise_crossings(self):
        r = self.compare(self.camp())
        self.assertTrue(r["passed"], r["problems"])
        self.assertEqual((r["n_seeds"], r["n_rows"], r["max_rel"]), (10, 80, 0.0))
        self.assertEqual((r["csv_sha_equal"], r["csv_sha_differ"], r["csv_sha_unavailable"]), (10, 0, 0))

    def test_without_the_frozen_segments_the_frozen_manifest_gives_the_hashes(self):
        r = self.compare(self.camp(), segments=False)
        self.assertTrue(r["passed"])
        self.assertEqual(r["csv_sha_equal"], 10)

    def test_the_smoke_precheck_one_seed(self):
        r = self.compare(self.camp(), "434001")
        self.assertTrue(r["passed"])
        self.assertEqual((r["n_seeds"], r["n_rows"], r["csv_sha_equal"]), (1, 8, 1))

    def test_relative_tolerance_1e_12(self):
        for rel, ok in ((1e-13, True), (1e-9, False)):
            c = self.camp()
            t = self.tables(c)[0]

            def bump(rows):
                i = rows[0].index("alpha_TP")
                rows[1][i] = repr(float(rows[1][i]) * (1 + rel))
            edit_table(t, bump)
            r = self.compare(c)
            self.assertEqual(r["passed"], ok, (rel, r["problems"]))
            if not ok:
                self.assertAlmostEqual(r["max_rel"], 1e-9, delta=1e-10)
                self.assertIn("alpha_TP", r["problems"][0])

    def test_excluded_columns_are_not_compared(self):
        """`source` is scribbled over; `r` differs by construction (the campaign's units are not the frozen ones) and the identical-campaign test passes."""
        c = self.camp()
        for t in self.tables(c, ".csv"):
            edit_table(t, lambda rows: [row.__setitem__(rows[0].index("source"), "elsewhere/realization_9999_planes.csv") for row in rows[1:]])
        r = self.compare(c)
        self.assertTrue(r["passed"], r["problems"])

    def test_cell_diff_zero_uses_the_absolute_tolerance(self):
        self.assertTrue(repro.cell_diff("0", "1e-16")[0])
        self.assertFalse(repro.cell_diff("0", "1e-14")[0])
        self.assertTrue(repro.cell_diff("", "")[0])
        self.assertTrue(repro.cell_diff("nan", "nan")[0])
        self.assertFalse(repro.cell_diff("nan", "1")[0])
        self.assertTrue(repro.cell_diff("a", "a")[0])
        self.assertFalse(repro.cell_diff("a", "b")[0])
        self.assertTrue(repro.cell_diff("1", "1.0000000000001")[0])
        self.assertFalse(repro.cell_diff("1", "1.00000000001")[0])

    def test_timing_and_path_columns_are_excluded_but_physical_times_are_not(self):
        for c in ("source", "r", "wall_s", "elapsed_seconds", "run_path"):
            self.assertTrue(repro.excluded(c), c)
        for c in ("U_time", "t_end", "t_end_next", "alpha_LP", "n_last"):
            self.assertFalse(repro.excluded(c), c)

    def test_tables_of_one_arm_with_different_columns_are_refused(self):
        c = self.camp()
        edit_table(self.tables(c)[0], lambda rows: [row.append("x" if k else "extra") for k, row in enumerate(rows)])
        with self.assertRaises(LoadError):
            self.compare(c)

    def test_a_missing_row_fails(self):
        c = self.camp()
        edit_table(self.tables(c)[0], lambda rows: rows.pop(3))
        r = self.compare(c)
        self.assertFalse(r["passed"])
        self.assertTrue(any("one side only" in p for p in r["problems"]))

    def test_a_missing_seed_fails(self):
        r = self.compare(self.camp(), "434001-434011")  # 434011 is not in the 10-seed campaign
        self.assertFalse(r["passed"])
        self.assertEqual(r["n_seeds"], 10)
        self.assertTrue(any("10 of 11" in p for p in r["problems"]))

    def test_a_different_column_set_fails(self):
        c = self.camp()
        for t in self.tables(c, ".csv"):
            edit_table(t, lambda rows: [row.append("x" if k else "extra") for k, row in enumerate(rows)])
        r = self.compare(c)
        self.assertFalse(r["passed"])
        self.assertTrue(any("column sets differ" in p for p in r["problems"]))

    def test_a_crossing_csv_that_differs_is_reported_not_gating(self):
        c = self.camp()
        seg = sorted(os.listdir(os.path.join(c, "segments", "j1_base10")))[0]
        p = os.path.join(c, "segments", "j1_base10", seg, "out", "crossings", "realization_0000_planes.csv")
        with open(p, "a") as f:
            f.write("changed\n")
        r = self.compare(c)
        self.assertTrue(r["passed"])
        self.assertEqual((r["csv_sha_equal"], r["csv_sha_differ"]), (9, 1))

    def test_a_duplicated_seed_is_an_error(self):
        c = self.camp()
        segs = sorted(os.listdir(os.path.join(c, "reduced", "j1_base10")))
        a, b = segs[0], segs[1]
        src = os.path.join(c, "reduced", "j1_base10", a, "j1_base10__%s__d25-94.csv" % a)
        dst = os.path.join(c, "reduced", "j1_base10", b, "j1_base10__%s__d25-94.csv" % b)
        s0b = int(b.rsplit("_s", 1)[1])
        shutil.copy(src, dst)  # segment b now repeats the seeds of segment a, whose seed0 differs: the same (seed, ...) keys appear twice
        edit_table(dst, lambda rows: [row.__setitem__(rows[0].index("r"), str(int(row[rows[0].index("r")]) + int(a.rsplit("_s", 1)[1]) - s0b))
                                      for row in rows[1:]])
        with self.assertRaises(LoadError):
            self.compare(c)

    def test_a_frozen_file_that_changed_since_the_freeze_is_refused(self):
        fw = synth.frozen_world(os.path.join(self.tmp.name, "frozen2"))
        t = os.path.join(fw["frozen_reduced"], "s4_g5p")
        t = os.path.join(t, sorted(os.listdir(t))[0])
        t = os.path.join(t, [f for f in sorted(os.listdir(t)) if f.endswith("d25-94.csv")][0])
        with open(t, "a") as f:
            f.write("\n")
        with self.assertRaises(LoadError):
            repro.compare(self.camp(), "j1_base10", repro.parse_seeds("434001"), fw["frozen_reduced"], fw["frozen_manifest"])

    def test_parse_seeds(self):
        self.assertEqual(repro.parse_seeds("434001"), [434001])
        self.assertEqual(repro.parse_seeds("434001-434003"), [434001, 434002, 434003])
        self.assertEqual(repro.parse_seeds("5,7-8"), [5, 7, 8])

    def run_cli(self, camp, seeds="434001-434010"):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = cli.main(["repro", "--campaign", camp, "--arm", "j1_base10", "--seeds", seeds, "--frozen-reduced", self.fw["frozen_reduced"],
                           "--frozen-manifest", self.fw["frozen_manifest"], "--frozen-segments", self.fw["frozen_segments"]])
        return rc, out.getvalue(), err.getvalue()

    def test_cli_prints_one_line_and_nothing_else_on_stdout(self):
        rc, out, err = self.run_cli(self.camp())
        self.assertEqual(rc, 0)
        self.assertEqual(len(out.strip().splitlines()), 1)
        self.assertRegex(out.strip(), LINE)
        self.assertEqual(out.strip(), "REPRO PASS 10 80 0 csv_sha_equal=10/10")
        c = self.camp()
        edit_table(self.tables(c)[0], lambda rows: rows[1].__setitem__(rows[0].index("alpha_LP"), repr(float(rows[1][rows[0].index("alpha_LP")]) * 1.001)))
        rc, out, err = self.run_cli(c)
        self.assertEqual(rc, 1)
        self.assertRegex(out.strip(), LINE)
        self.assertTrue(out.startswith("REPRO FAIL"))
        self.assertEqual(len(out.strip().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
