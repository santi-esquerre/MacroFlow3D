"""SF-43 `--tracker-counts` option of the reducer: opt-in copy of the plane pass's tracker totals (track.json) onto the consumer plane rows."""
import csv
import math
import os
import tempfile
import unittest

import numpy as np

from sf42red import __main__ as cli
from sf42red import reduce as red
from sf42red.tests import synth

ELL, X0 = 0.25, 1.25


def slurp(path):
    with open(path, "rb") as f:
        return f.read()


def table(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class TrackerCounts(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.root = self._t.name
        self.addCleanup(self._t.cleanup)

    def arm(self, name, tracker, hmax=None, r=0, flux_zero=()):
        w = np.ones(64)
        for p in flux_zero:
            w[p] = 0.0
        return synth.write_pass(os.path.join(self.root, name), ell=ELL, x0=X0, hmax=hmax, r=r, tracker=tracker, flux_weights=w)

    def reduce(self, arm, out, **kw):
        red.reduce_all([os.path.join(self.root, arm)], [], [], None, None, 25.0, 94.0, None, os.path.join(self.root, out), self.root, **kw)
        return os.path.join(self.root, out)

    def test_default_off_tables_are_byte_identical_and_unchanged(self):
        T = {"recrossings": 7, "backflow_encounters": 3, "seed_backflow": 2}
        self.arm("a", None)
        a = self.reduce("a", "ta.csv")
        self.arm("a", T)  # the same realization (same paths), now with the tracker totals in track.json
        b = self.reduce("a", "tb.csv")
        for sfx in (".csv", ".planes.csv"):
            self.assertEqual(slurp(a[:-4] + sfx), slurp(b[:-4] + sfx))
        with open(a) as f:
            self.assertEqual(f.readline().rstrip("\n").split(","), red.COLUMNS)

    def test_on_appends_columns_and_values_per_pass(self):
        arm = "m"
        self.arm(arm, {"recrossings": 12, "backflow_encounters": 4, "seed_backflow": 1}, hmax=8, flux_zero=(5,))  # 63 traced particles
        self.arm(arm, {"recrossings": 30, "backflow_encounters": 9, "seed_backflow": 1}, hmax=16, flux_zero=(5,))
        on = table(self.reduce(arm, "on.csv", tracker_counts=True))
        off = table(self.reduce(arm, "off.csv"))
        self.assertEqual(list(on[0]), red.COLUMNS + red.TRACKER_COLUMNS)
        for h, rec, bf in ((8, 12, 4), (16, 30, 9)):
            rows = [r for r in on if r["hmax_div"] == str(h) and r["family"] == "plane"]
            self.assertEqual(len(rows), 12)
            for r in rows:
                self.assertEqual(r["n_traced"], "63")
                self.assertAlmostEqual(float(r["recrossings_per_traced"]), rec / 63.0, delta=1e-15)
                self.assertAlmostEqual(float(r["frac_backflow"]), bf / 63.0, delta=1e-15)
                self.assertEqual(r["seed_backflow"], "1")
        for r in on:  # time rows: empty
            if r["family"] == "time":
                self.assertEqual((r["recrossings_per_traced"], r["frac_backflow"], r["seed_backflow"]), ("", "", ""))
        strip = lambda rows: [{k: v for k, v in r.items() if k not in red.TRACKER_COLUMNS} for r in rows]  # noqa: E731
        self.assertEqual(strip(on), off)  # nothing else changes

    def test_pollock_rows_leave_the_columns_empty(self):
        proot = self.arm("p", {"recrossings": 5, "backflow_encounters": 2, "seed_backflow": 0})
        synth.write_pollock(proot, proot)
        rows = table(self.reduce("p", "pol.csv", pollock=True, tracker_counts=True))
        self.assertEqual({r["code"] for r in rows}, {"consumer", "pollock"})
        for r in rows:
            if r["code"] == "pollock":
                self.assertEqual((r["recrossings_per_traced"], r["frac_backflow"], r["seed_backflow"]), ("", "", ""))
            elif r["family"] == "plane":
                self.assertEqual(float(r["recrossings_per_traced"]), 5 / 64.0)

    def test_missing_totals_are_an_error_when_requested(self):
        self.arm("n", None)
        with self.assertRaises(red.MissingData):
            self.reduce("n", "bad.csv", tracker_counts=True)

    def test_zero_traced_gives_nan(self):
        tr = {"plane_pass": {"recrossings": 1, "backflow_encounters": 1}, "seed_backflow": 4}
        self.assertTrue(all(math.isnan(v) for v in red.tracker_counts_of(tr, 0, "x")[:2]))

    def test_cli_flag(self):
        self.arm("c", {"recrossings": 1, "backflow_encounters": 1, "seed_backflow": 0})
        out = os.path.join(self.root, "cli.csv")
        for flag, cols in ((None, red.COLUMNS), ("--tracker-counts", red.COLUMNS + red.TRACKER_COLUMNS)):
            argv = ["reduce", "--consumer", os.path.join(self.root, "c"), "--d-window", "25,94", "--root", self.root, "--out", out]
            self.assertEqual(cli.main(argv + ([flag] if flag else [])), 0)
            self.assertEqual(list(table(out)[0]), cols)


if __name__ == "__main__":
    unittest.main()
