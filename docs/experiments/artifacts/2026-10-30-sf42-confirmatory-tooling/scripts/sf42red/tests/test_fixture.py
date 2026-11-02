"""Integration test on a tiny REAL consumer output (fixture/README.txt): the reducer's per-plane moments and time-window ends are
compared with the runner's own CSVs (moments_planes.csv, twin/moments_planes.csv, exits_time.csv, stats_twin series)."""
import csv
import math
import os
import tempfile
import unittest

import numpy as np

from sf42red import reduce as red

NAN = float("nan")

HERE = os.path.dirname(os.path.abspath(__file__))
FIX = os.path.join(HERE, "fixture")
ARM = os.path.join(FIX, "arm")


def table(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def runner_csv(path):
    with open(path) as f:
        return list(csv.DictReader(ln for ln in f if not ln.startswith("#")))


def near(a, b, rtol=1e-9, atol=1e-15):
    return abs(a - b) <= rtol * max(abs(a), abs(b)) + atol


class TestFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.out = os.path.join(cls._tmp.name, "t.csv")
        red.reduce_all([ARM], [], [], 0.25, 1.25, 4.0, 20.0, None, cls.out, FIX)
        cls.rows = table(cls.out)
        cls.planes = table(cls.out[:-4] + ".planes.csv")

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_row_inventory(self):
        plane = [r for r in self.rows if r["family"] == "plane"]
        time_ = [r for r in self.rows if r["family"] == "time"]
        self.assertEqual((len(plane), len(time_)), (24, 6))  # 2 realizations x 12 / 2 x 3
        self.assertTrue(all(r["note"] == "" for r in plane), [r["note"] for r in plane if r["note"]])
        self.assertTrue(all(r["n_planes"] == "5" and r["n_traced"] == "100" for r in plane))
        for r in plane:
            self.assertTrue(math.isfinite(float(r["alpha_TP"])) and math.isfinite(float(r["alpha_TAP"])) and math.isfinite(float(r["alpha_LP"])))
            self.assertTrue(math.isfinite(float(r["alpha_TP_A"])) and math.isfinite(float(r["alpha_LP_B"])))
        # sources are relative to the root and the seeds are the runner's seed0 + r
        self.assertEqual(sorted({(r["r"], r["seed"], r["source"]) for r in plane}),
                         [("0", "420501", "arm/realizations/r_0000"), ("1", "420502", "arm/realizations/r_0001")])

    def test_plane_moments_equal_the_runners_csv(self):
        for r in (0, 1):
            R = os.path.join(ARM, "realizations", "r_%04d" % r)
            for weights, ref in (("flux", "moments_planes.csv"), ("unif", os.path.join("twin", "moments_planes.csv"))):
                ref_rows = runner_csv(os.path.join(R, ref))
                mine = [p for p in self.planes if p["r"] == str(r) and p["weights"] == weights and p["basis"] == "pos"]
                self.assertEqual(len(mine), len(ref_rows))
                for m, c in zip(mine, ref_rows):
                    self.assertTrue(near(float(m["x1"]), float(c["x1"]), 1e-12))
                    self.assertEqual(int(m["n"]), int(float(c["contributing"])))
                    for a, b in (("mean_x2", "mean_x2"), ("var_x2", "var_x2"), ("mean_x3", "mean_x3"), ("var_x3", "var_x3"),
                                 ("mean_tau", "mean_tau"), ("var_tau", "var_tau")):
                        self.assertTrue(near(float(m[a]), float(c[b])), (r, weights, a, m[a], c[b]))

    def test_displacement_moments_differ_from_position_by_the_seed_offset(self):
        for p in (p for p in self.planes if p["basis"] == "disp" and p["weights"] == "unif" and p["r"] == "0"):
            q = [x for x in self.planes if x["basis"] == "pos" and x["weights"] == "unif" and x["r"] == "0" and x["plane"] == p["plane"]][0]
            self.assertTrue(math.isfinite(float(p["cov_x2x3"])))
            self.assertNotEqual(p["var_x2"], q["var_x2"])

    def test_time_window_end_matches_the_runners_exit_counts_and_series(self):
        for r in (0, 1):
            R = os.path.join(ARM, "realizations", "r_%04d" % r)
            ex = runner_csv(os.path.join(R, "exits_time.csv"))
            first = next((i for i, e in enumerate(ex) if int(e["exit_inlet"]) + int(e["exit_outlet"]) > 0), None)
            row = [x for x in self.rows if x["family"] == "time" and x["r"] == str(r) and x["weights"] == "unif"][0]
            if first is None:
                self.assertTrue(math.isnan(float(row["t_end_next"])))
                self.assertEqual(int(row["j_end"]), len(ex) - 1)
            else:
                self.assertEqual(int(row["j_end"]), first - 1)
                self.assertTrue(near(float(row["t_end_next"]), float(ex[first]["t"]), 1e-12))
                self.assertTrue(near(float(row["t_end"]), float(ex[first - 1]["t"]), 1e-12))
            # the same window from the runner's own uniform-weight series (Par2 format), an independent path (CSV vs arrays)
            # (the twin series has a decreasing `active`, so the old count rule gives the window end independently here; the
            # production Par2 rule uses snapshot positions, see test_reduce)
            ser = os.path.join(ARM, "stats_twin", "realization_%04d_timeseries.csv" % r)
            s = red.read_par2_series(red.Recorder(), ser)
            j_act = int(np.flatnonzero(s["active"] == s["active"][0])[-1])
            self.assertEqual(j_act, int(row["j_end"]))
            prow = red.par2_time_row_at(red.Recorder(), ser, 0.25, 1.25, float(row["U_time"]), float(s["t"][j_act]), NAN, j_act)
            self.assertTrue(near(prow["alpha_LAM"], float(row["alpha_LAM"]), 1e-8), (prow["alpha_LAM"], row["alpha_LAM"]))
            self.assertTrue(near(prow["alpha_Tfixed"], float(row["alpha_Tfixed"]), 1e-8))

    def test_no_particle_lost_inside_the_window(self):
        row = [x for x in self.rows if x["family"] == "time" and x["r"] == "1" and x["weights"] == "unif"][0]
        self.assertEqual(row["n_set"], "100")  # all 100 have a record at t_end although 7 exit later


class TestCli(unittest.TestCase):
    def test_reduce_splitshare_prune_dry_run_and_guard(self):
        import contextlib
        import io
        from sf42red.__main__ import main
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "t.csv")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(main(["reduce", "--consumer", ARM, "--ell", "0.25", "--x0", "1.25", "--d-window", "4,20", "--out", out,
                                       "--root", FIX]), 0)
                self.assertEqual(main(["splitshare", out]), 0)
            self.assertIn("wrote 30 rows", buf.getvalue())
            self.assertIn("code,family,weights,basis,set,hmax_div,estimator,n,component,variance,fraction,flag", buf.getvalue())
            ret = os.path.join(td, "ret.txt")
            with open(ret, "w") as f:
                f.write("arm/realizations/r_0000\n")
            err = io.StringIO()
            with contextlib.redirect_stderr(err):  # the CLI always applies decision G8's minimum: the arm has 2 realizations, 1 retained
                self.assertEqual(main(["prune", "--arms", ARM, "--retained", ret, "--table", out, "--root", FIX]), 2)
            self.assertIn("G8", err.getvalue())
            with open(ret, "w") as f:
                f.write("arm/realizations/r_0000\narm/realizations/r_0001\n")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(main(["prune", "--arms", ARM, "--retained", ret, "--table", out, "--root", FIX]), 0)
            self.assertIn("DRY RUN: 18 .npy recorded, 2 retained realization(s) (min per arm 5), 0 file(s)", buf.getvalue())
            self.assertEqual(len([f for f in os.listdir(os.path.join(ARM, "realizations", "r_0001")) if f.endswith(".npy")]), 9)
            err = io.StringIO()
            with contextlib.redirect_stderr(err):
                self.assertEqual(main(["prune", "--arms", ARM, "--retained", ret, "--table", out, "--root", FIX, "--execute"]), 2)
            self.assertIn("needs a --log", err.getvalue())

    def test_prune_min_retained_option(self):
        """SF-43: `--min-retained N` (default stays decision G8's 5); a segment may retain fewer, down to none."""
        import contextlib
        import io
        from sf42red.__main__ import main
        with tempfile.TemporaryDirectory() as td:
            out = os.path.join(td, "t.csv")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["reduce", "--consumer", ARM, "--ell", "0.25", "--x0", "1.25", "--d-window", "4,20", "--out", out,
                                       "--root", FIX]), 0)
            ret = os.path.join(td, "ret.txt")
            with open(ret, "w") as f:
                f.write("arm/realizations/r_0000\n")
            base = ["prune", "--arms", ARM, "--retained", ret, "--table", out, "--root", FIX]

            def run(*extra):
                o, e = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(o), contextlib.redirect_stderr(e):
                    rc = main(base + list(extra))
                return rc, o.getvalue(), e.getvalue()
            rc, _, err = run()
            self.assertEqual((rc, "G8" in err), (2, True))                    # default unchanged
            rc, _, err = run("--min-retained", "2")
            self.assertEqual((rc, "requires at least 2" in err), (2, True))   # explicit value is enforced
            rc, o, _ = run("--min-retained", "1")
            self.assertEqual(rc, 0)
            self.assertIn("(min per arm 1)", o)
            with open(ret, "w") as f:
                f.write("# nothing retained in this segment\n")
            rc, o, _ = run("--min-retained", "0")
            self.assertEqual(rc, 0)
            self.assertIn("(min per arm 0)", o)
            rc, _, err = run("--min-retained", "-1")
            self.assertEqual((rc, "must be >= 0" in err), (2, True))


if __name__ == "__main__":
    unittest.main()
