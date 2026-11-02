"""recompute43 against tables produced by the SF-42 reducer on synthetic passes with exactly known moments.

The tool under test is independent of `sf42red`; the TESTS use `sf42red` (from the SF-42 artifact) as the producer of the reduced
tables, so a pass means two separately written implementations agree.  Raw moment CSVs are written here the way the tracker does
(`finalize_track_result`): biased moments over the particles that landed on the plane, weights renormalised over that set.
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SF42 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "2026-10-30-sf42-confirmatory-tooling", "scripts"))
sys.path.insert(0, SF42)
from sf42red import reduce as red  # noqa: E402
from sf42red.tests import synth  # noqa: E402

from recompute43 import driver  # noqa: E402
from recompute43.__main__ import main  # noqa: E402

ELL, X0 = 0.25, 1.25


def tracker_planes_csv(path, proot, wmode):
    """moments_planes.csv of a synthetic pass: the tracker's header, `%.15e`, own set per plane (finite record, weight > 0)."""
    X = np.load(os.path.join(proot, "planes_x23.npy"))
    tau = np.load(os.path.join(proot, "planes_tau.npy"))
    w0 = np.load(os.path.join(proot, "weights.npy"))
    with open(os.path.join(proot, "track.json")) as f:
        planes = json.load(f)["planes"]
    w = w0 if wmode == "flux" else np.full(len(w0), 1.0 / len(w0))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("x1,contributing,non_ok,mean_x2,var_x2,mean_x3,var_x3,mean_tau,var_tau,rms_ev,max_ev\n")
        for m, x1 in enumerate(planes):
            sel = np.isfinite(X[:, m, :]).all(axis=1) & np.isfinite(tau[:, m])
            ww = w[sel]
            sw = ww.sum()

            def mv(v):
                mu = (ww * v[sel]).sum() / sw
                return mu, (ww * (v[sel] - mu) ** 2).sum() / sw
            (m2, v2), (m3, v3), (mt, vt) = mv(X[:, m, 0]), mv(X[:, m, 1]), mv(tau[:, m])
            f.write("%.15e,%d,0,%.15e,%.15e,%.15e,%.15e,%.15e,%.15e,0.0,0.0\n" % (x1, sel.sum(), m2, v2, m3, v3, mt, vt))


class Base(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.C = os.path.join(self._t.name, "camp")
        self.R = os.path.join(self._t.name, "reduced")
        os.makedirs(self.R)
        self.addCleanup(self._t.cleanup)

    def consumer(self, seg="seg0", keep_arrays=True, **kw):
        out = os.path.join(self.C, "arms", "a1", seg, "out")
        proot = synth.write_pass(out, ell=ELL, x0=X0, **kw)
        tracker_planes_csv(os.path.join(proot, "moments_planes.csv"), proot, "flux")
        tracker_planes_csv(os.path.join(proot, "twin", "moments_planes.csv"), proot, "unif")
        return out, proot

    def reduce(self, name="t.csv", consumer=(), par2=(), **kw):
        red.reduce_all(list(consumer), list(par2), [], ELL if par2 else None, X0 if par2 else None, 25.0, 94.0, None,
                       os.path.join(self.R, name), self.C, **kw)

    def prune(self, proot):
        for n in ("seeds", "weights", "planes_x23", "planes_tau", "planes_I", "times_x", "times_I", "status_plane", "status_time"):
            os.remove(os.path.join(proot, n + ".npy"))

    def run_tool(self, **kw):
        buf = io.StringIO()
        code = driver.run(self.C, self.R, [self.C], buf, **kw)
        return code, buf.getvalue()

    def edit_table(self, name, fn):
        import csv
        p = os.path.join(self.R, name)
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
            cols = list(rows[0].keys())
        for r in rows:
            fn(r)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, cols, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        mp = p + ".manifest.json"  # a deliberate edit refreshes the table hash so that only the VALUE comparison is exercised
        with open(mp) as f:
            man = json.load(f)
        for e in man["tables"]:
            if e["path"] == name:
                e["sha256"] = driver.sha256(p)
        with open(mp, "w") as f:
            json.dump(man, f)


class Consumer(Base):
    def test_pass_with_arrays_and_csv(self):
        out, _ = self.consumer(drift=(0.3, -0.1), miss={3: 5, 10: 6})
        self.reduce(consumer=[out])
        code, rep = self.run_tool()
        self.assertEqual(code, 0, rep)
        self.assertIn("RECOMPUTE PASS", rep)
        for col in ("alpha_TP", "alpha_TAP", "alpha_LP", "U_r", "tilt", "n_first", "wfrac_min"):
            self.assertIn(" arrays    %-12s" % col, rep)
        self.assertIn(" csv       alpha_TP", rep)
        self.assertNotIn(" csv       alpha_TAP", rep)  # no cross covariance in the moment CSV
        self.assertIn("realizations=1", rep)
        self.assertIn("REALIZATIONS consumer=1 (with plane arrays 1)", rep)

    def test_pruned_pass_is_csv_only(self):
        out, proot = self.consumer(drift=(0.2, 0.0), miss={4: 6})
        self.reduce(consumer=[out])
        self.prune(proot)
        code, rep = self.run_tool()
        self.assertEqual(code, 0, rep)
        self.assertIn("REALIZATIONS consumer=1 (with plane arrays 0)", rep)
        self.assertNotRegex(rep, r"consumer +plane +arrays")
        self.assertIn(" csv       alpha_LP", rep)
        # disp, common and expI rows, and the TAP column, are listed as not covered
        self.assertIn("NOT_COVERED", rep)
        self.assertIn("variant needs the per-particle arrays", rep)
        self.assertIn("NOT_RECOMPUTED_ROWS", rep)  # the time-family rows

    def test_perturbed_reduced_value_is_detected_beyond_1e10_not_below(self):
        out, _ = self.consumer()
        self.reduce(consumer=[out])

        def bump(f):
            def fn(r):
                if r["family"] == "plane" and r["weights"] == "flux" and r["basis"] == "pos" and r["set"] == "perplane":
                    r["alpha_TP"] = "%.17g" % (float(r["alpha_TP"]) * (1.0 + f))
            return fn
        self.edit_table("t.csv", bump(1e-11))
        code, rep = self.run_tool()
        self.assertEqual(code, 0, rep)
        self.edit_table("t.csv", bump(1e-9))
        code, rep = self.run_tool()
        self.assertEqual(code, 1)
        self.assertIn("RECOMPUTE FAIL", rep)
        self.assertEqual(rep.count("FAIL consumer"), 2, rep)  # the csv and the arrays origin
        self.assertIn("alpha_TP: recomputed", rep)

    def test_perturbed_raw_moment_is_detected(self):
        out, proot = self.consumer(drift=(0.1, 0.1))
        self.reduce(consumer=[out])
        self.prune(proot)
        p = os.path.join(proot, "moments_planes.csv")
        with open(p) as f:
            lines = f.read().splitlines()
        cells = lines[5].split(",")
        cells[4] = "%.15e" % (float(cells[4]) * (1 + 1e-6))  # var_x2 of plane 4 (a window plane)
        lines[5] = ",".join(cells)
        with open(p, "w") as f:
            f.write("\n".join(lines) + "\n")
        code, rep = self.run_tool()
        self.assertEqual(code, 1)
        self.assertIn("FAIL consumer", rep)  # (the moment CSVs are not reducer inputs: no manifest entry)

    def test_contributor_count_mismatch_is_a_failure(self):
        out, proot = self.consumer(miss={2: 5})
        self.reduce(consumer=[out])
        self.prune(proot)
        p = os.path.join(proot, "moments_planes.csv")
        with open(p) as f:
            lines = f.read().splitlines()
        cells = lines[5].split(",")
        cells[1] = str(int(cells[1]) + 1)
        lines[5] = ",".join(cells)
        with open(p, "w") as f:
            f.write("\n".join(lines) + "\n")
        code, rep = self.run_tool()
        self.assertEqual(code, 1)
        self.assertIn("n_", rep)

    def test_missing_raw_source_is_a_problem_and_nothing_compared_is_code_2(self):
        out, _ = self.consumer()
        self.reduce(consumer=[out])
        shutil.rmtree(os.path.join(self.C, "arms"))
        code, rep = self.run_tool()
        self.assertEqual(code, 1)
        self.assertIn("raw source not found", rep)
        os.remove(os.path.join(self.R, "t.csv"))
        os.remove(os.path.join(self.R, "t.csv.manifest.json"))
        os.remove(os.path.join(self.R, "t.planes.csv"))
        code, rep = self.run_tool()
        self.assertEqual(code, 2)

    def test_cli_pass_writes_report(self):
        out, _ = self.consumer()
        self.reduce(consumer=[out])
        rep = os.path.join(self._t.name, "report.txt")
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            code = main(["run", "--campaign", self.C, "--reduced", self.R, "--out", rep])
        self.assertEqual((code, buf.getvalue().strip()), (0, "RECOMPUTE PASS"))

    def test_cli_refuses_report_inside_campaign(self):
        os.makedirs(self.C)
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            main(["run", "--campaign", self.C, "--reduced", self.R, "--out", os.path.join(self.C, "report.txt")])


class Pollock(Base):
    """`code=pollock` rows (sf42red reduce --pollock) against the Pollock bridge arrays (`R/pollock/`), recomputed without sf42red."""

    def build(self, hmax=(), **kw):
        out = os.path.join(self.C, "arms", "a1", "seg0", "out")
        if hmax:
            proots = [synth.write_pass(out, ell=ELL, x0=X0, hmax=h, **kw) for h in hmax]
            rdir = os.path.dirname(proots[0])
        else:
            proots = [synth.write_pass(out, ell=ELL, x0=X0, **kw)]
            rdir = proots[0]
        twin = synth.write_pass(os.path.join(self.C, "cloud"), ell=ELL, x0=X0, a_T=0.05, U0=3.0, c_tau=0.01, drift=(0.1, 0.2), miss={7: 4})
        pd = synth.write_pollock(rdir, proots[0], planes_x23=np.load(os.path.join(twin, "planes_x23.npy")),
                                 planes_t=np.load(os.path.join(twin, "planes_tau.npy")), weights=np.arange(1.0, 65.0) % 7)  # includes weight-0 seeds
        return out, rdir, pd

    def test_pollock_rows_are_recomputed_from_the_bridge_arrays(self):
        out, _, _ = self.build(miss={3: 5})
        self.reduce(consumer=[out], pollock=True)
        code, rep = self.run_tool()
        self.assertEqual(code, 0, rep)
        for col in ("alpha_TP", "alpha_TAP", "alpha_LP", "U_r", "wfrac_min", "n_common", "seed"):
            self.assertIn(" pollock  plane        arrays    %-12s" % col, rep)
        self.assertIn(" pollock  planes-table arrays    cov_x2x3", rep)
        self.assertIn("POLLOCK realizations=1 (arrays on disk 1)", rep)

    def test_wrong_pollock_value_is_a_failure_and_a_consumer_value_is_not_confused_with_it(self):
        out, _, _ = self.build()
        self.reduce(consumer=[out], pollock=True)

        def bump(code):
            def fn(r):
                if r["code"] == code and r["family"] == "plane" and r["weights"] == "unif" and r["basis"] == "disp" and r["set"] == "common":
                    r["alpha_TAP"] = "%.17g" % (float(r["alpha_TAP"]) * (1.0 + 1e-9))
            return fn
        self.edit_table("t.csv", bump("pollock"))
        code, rep = self.run_tool()
        self.assertEqual(code, 1)
        self.assertEqual(rep.count("FAIL pollock"), 1, rep)
        self.assertNotIn("FAIL consumer", rep)

    def test_pruned_pollock_arrays_are_not_covered_not_failed(self):
        out, _, pd = self.build()
        self.reduce(consumer=[out], pollock=True)
        for n in os.listdir(pd):
            if n.endswith(".npy"):
                os.remove(os.path.join(pd, n))
        code, rep = self.run_tool()
        self.assertEqual(code, 0, rep)
        self.assertIn("POLLOCK realizations=1 (arrays on disk 0)", rep)
        self.assertIn("NOT_COVERED 1 x Pollock arrays (or seeds) pruned", rep)
        self.assertNotRegex(rep, r"pollock +plane")

    def test_hmax_label_must_be_a_pass_of_the_realization(self):
        out, _, _ = self.build(hmax=(8, 16))
        self.reduce(consumer=[out], pollock=True)
        code, rep = self.run_tool()
        self.assertEqual(code, 0, rep)
        self.edit_table("t.csv", lambda r: r.update(hmax_div="32") if r["code"] == "pollock" and r["hmax_div"] == "16" else None)
        code, rep = self.run_tool()
        self.assertEqual(code, 1)
        self.assertIn("hmax_div label 32 is not a pass", rep)

    def test_single_pass_label(self):
        out, _, _ = self.build()
        self.reduce(consumer=[out], pollock=True, single_pass_hmax=8)
        code, rep = self.run_tool()
        self.assertEqual(code, 1)  # the reducer labelled the single pass 8: the tool must be told
        code, rep = self.run_tool(single_pass_hmax=8)
        self.assertEqual(code, 0, rep)


class CampaignLayout(Base):
    """The engine's layout: segments/<arm>/<segment>, reduced/<arm>/<segment>/*.csv with `source` relative to the segment directory;
    a primary arm keeps the plane arrays of every realization (sf42red prune --keep-arrays)."""

    def build(self, arm="prim", seg="u000001_s100", drop=()):
        sdir = os.path.join(self.C, "segments", arm, seg)
        proot = synth.write_pass(os.path.join(sdir, "out"), ell=ELL, x0=X0, drift=(0.3, -0.1), miss={3: 5})
        tracker_planes_csv(os.path.join(proot, "moments_planes.csv"), proot, "flux")
        tracker_planes_csv(os.path.join(proot, "twin", "moments_planes.csv"), proot, "unif")
        rdir = os.path.join(self.C, "reduced", arm, seg)
        os.makedirs(rdir)
        red.reduce_all([os.path.join(sdir, "out")], [], [], None, None, 25.0, 94.0, None, os.path.join(rdir, "%s__%s__d25-94.csv" % (arm, seg)), sdir)
        for n in ("planes_I", "times_x", "times_I", "status_plane", "status_time") + tuple(drop):  # the selective prune
            os.remove(os.path.join(proot, n + ".npy"))
        return proot

    def run_tool(self, **kw):
        buf = io.StringIO()
        code = driver.run(self.C, os.path.join(self.C, "reduced"), [self.C], buf, **kw)
        return code, buf.getvalue()

    def test_displacement_basis_and_alpha_TAP_are_recomputed_from_the_kept_plane_arrays(self):
        self.build()
        code, rep = self.run_tool(primary_arms=("prim",))
        self.assertEqual(code, 0, rep[-600:])
        self.assertIn("PRIMARY_ARM prim realization passes=1 with_arrays=1", rep)
        for col in ("alpha_TAP", "alpha_TP", "wfrac_min"):
            self.assertIn(" arrays    %-12s" % col, rep)
        self.assertIn("NOT_COVERED 1 x expI/pos/perplane", rep)  # planes_I was pruned: expI is listed, never skipped silently
        self.assertNotIn("PROBLEM", rep)

    def test_a_primary_realization_without_the_plane_arrays_is_a_problem(self):
        self.build(drop=("seeds",))
        code, rep = self.run_tool(primary_arms=("prim",))
        self.assertEqual(code, 1)
        self.assertIn("PROBLEM", rep)
        self.assertIn("primary arm prim without the plane arrays", rep)
        code, _ = self.run_tool()  # not declared primary: only the moment CSVs are recomputed, as before
        self.assertEqual(code, 0)

    def test_a_declared_primary_arm_with_no_reduced_realization_is_a_problem(self):
        self.build()
        code, rep = self.run_tool(primary_arms=("other",))
        self.assertEqual(code, 1)
        self.assertIn("primary arm other: no reduced consumer realization", rep)


class Par2(Base):
    def rows(self, x1, a=0.02, v0=0.1, c=0.03, U=1.5, drift=0.3, nan_from=None):
        rows = []
        for k, x in enumerate(x1):
            off = x - X0
            if nan_from is not None and k >= nan_from:
                rows.append([x, 1000, 0] + [float("nan")] * 12)
                continue
            v = v0 + 2 * a * ELL * off
            g2 = 2 * a * ELL * off
            rows.append([x, 1000, 990 - k, off / U, c * off + 1e-3 * k, drift * off, 0.1 * off, v, v * 1.1, 0.01 * off, drift * off,
                         0.0, g2, g2 * 1.2, 0.02 * off])
        return rows

    def test_par2_all_columns_and_perturbation(self):
        x1 = X0 + np.array([5, 25, 40, 70, 94, 100]) * ELL
        p = os.path.join(self.C, "arms", "p1", "seg0", "out", "crossings", "realization_0002_planes.csv")
        synth.write_par2(p, self.rows(x1))
        self.reduce(par2=[p])
        code, rep = self.run_tool()
        self.assertEqual(code, 0, rep)
        for col in ("alpha_TP", "alpha_TAP", "alpha_LP", "U_r", "tilt", "wfrac_min", "n_traced"):
            self.assertIn(" crossings %-12s" % col, rep)
        self.assertIn("par2", rep)
        # raw file changed after the reduction: manifest problem and value failure
        synth.write_par2(p, self.rows(x1, a=0.021))
        code, rep = self.run_tool()
        self.assertEqual(code, 1)
        self.assertIn("input changed since the reduction", rep)
        self.assertIn("alpha_TP", rep)

    def test_par2_nan_planes_outside_window_and_empty_window_plane(self):
        x1 = X0 + np.array([5, 25, 40, 70, 94, 100]) * ELL
        p = os.path.join(self.C, "arms", "p1", "seg0", "out", "crossings", "realization_0000_planes.csv")
        synth.write_par2(p, self.rows(x1, nan_from=5))
        self.reduce(par2=[p])
        code, rep = self.run_tool()
        self.assertEqual(code, 0, rep)
        synth.write_par2(p, self.rows(x1, nan_from=3))  # empty plane inside the window: the reducer's NaN row must be reproduced
        self.reduce(par2=[p])
        code, rep = self.run_tool()
        self.assertEqual(code, 0, rep)


    def test_par2_time_series_rows_are_listed_not_recomputed_and_not_read_as_crossings(self):
        x1 = X0 + np.array([5, 25, 40, 70, 94, 100]) * ELL
        out = os.path.join(self.C, "arms", "p1", "seg0", "out")
        p = os.path.join(out, "crossings", "realization_0002_planes.csv")
        synth.write_par2(p, self.rows(x1))
        ser = os.path.join(out, "stats", "realization_0002_timeseries.csv")  # the Par2 time-series format, not the plane-crossings one
        os.makedirs(os.path.dirname(ser))
        with open(ser, "w") as f:
            f.write("# format_version=1\nt,mean_x,mean_y,mean_z,var_x,var_y,var_z,active\n")
            for k in range(1, 13):
                f.write("%.17g,%.17g,0,0,%.17g,%.17g,%.17g,100\n" % (0.5 * k, 1.0 * k, 0.01 * k, 0.008 * k, 0.008 * k))
        sd = os.path.join(out, "snapshots", "r_0002")
        os.makedirs(sd)
        for k in range(1, 13):
            with open(os.path.join(sd, "step_%08d.csv" % k), "w") as f:
                f.write("id,x coord,y coord,z coord\n0,1.0,0.5,0.5\n1,%.17g,0.5,0.5\n" % (9.0 if k - 1 >= 8 else 4.0))
        red.reduce_all([], [p], [ser], ELL, X0, 25.0, 94.0, 2.0, os.path.join(self.R, "t.csv"), self.C, par2_exit_x=8.75, par2_dt=0.5)
        code, rep = self.run_tool()
        self.assertEqual(code, 0, rep)
        self.assertIn("RECOMPUTE PASS", rep)
        self.assertNotIn("PROBLEM", rep)
        self.assertIn("NOT_RECOMPUTED_ROWS 1 x unif/pos/perplane: family time is not recomputed", rep)
        self.assertIn(" crossings alpha_TP", rep)  # plane rows still compared


class Formulas(unittest.TestCase):
    def test_close(self):
        from recompute43 import formulas as F
        self.assertTrue(F.close(float("nan"), float("nan"), 1e-10, 1e-15)[0])
        self.assertFalse(F.close(1.0, float("nan"), 1e-10, 1e-15)[0])
        self.assertTrue(F.close(1.0, 1.0 + 5e-11, 1e-10, 1e-15)[0])
        self.assertFalse(F.close(1.0, 1.0 + 5e-10, 1e-10, 1e-15)[0])
        self.assertTrue(F.close(0.0, 1e-16, 1e-10, 1e-15)[0])  # below atol
        self.assertFalse(F.close(0.0, 1e-14, 1e-10, 1e-15)[0])


if __name__ == "__main__":
    unittest.main()
