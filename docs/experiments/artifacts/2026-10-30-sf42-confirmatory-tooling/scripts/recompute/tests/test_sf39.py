"""Tests of the independent SF-39 recomputation.

Known-answer tests use hand-built moment files with analytic values (no analysis package).  The cross-check class builds a
complete fake runner tree with the SF-39 test fixtures, reduces it with the SF-39 reducer (the "committed" values) and checks that
`recompute.sf39` (which imports none of it) reproduces them; it is skipped if the SF-39 artifact is not importable.
"""
import contextlib
import csv
import io
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from recompute import sf39 as rc

HERE = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_SF39 = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "2026-10-27-sf39-macrodispersion-study"))

ELL, L1, X0, U = 0.5, 20.0, 2.5, 1.0
PLANES = [6.25, 10.0, 17.5]  # X_q, X_a, X_b
TIMES = [0.25 * n for n in range(1, 9)]


def wjson(path, d):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f)


def write_pass(P, planes_rows, time_rows, twin=False):
    """planes_rows: (x1, contributing, var_x2, var_x3); time_rows: (t, var_x, var_y, var_z, active)."""
    d = os.path.join(P, "twin") if twin else P
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "moments_planes.csv"), "w") as f:
        f.write("x1,contributing,non_ok,mean_x2,var_x2,mean_x3,var_x3,mean_tau,var_tau,rms_ev,max_ev\n")
        for x1, n, v2, v3 in planes_rows:
            f.write("%.15e,%d,0,0,%.15e,0,%.15e,0,0,0,0\n" % (x1, n, v2, v3))
    with open(os.path.join(d, "moments_time.csv"), "w") as f:
        f.write("# format_version=1\nt,mean_x,mean_y,mean_z,var_x,var_y,var_z,active\n")
        for t, vx, vy, vz, a in time_rows:
            f.write("%.15e,0,0,0,%.15e,%.15e,%.15e,%d\n" % (t, vx, vy, vz, a))


def build(root, n_planes=(100, 100, 100), actives=None, tau_planes=None):
    """A hand-built realization: sigma_T^2 = 0.01, 0.03, 0.06 at X_q, X_a, X_b; var_x = 3t+1, var_y = 0.02+0.04t, var_z = 0.1+0.08t."""
    R = root
    wjson(R + "/field.json", {"seed": 5001, "ell": ELL, "applied_scale": 1.0, "raw_variance": 1.1, "final_variance": 1.0})
    wjson(R + "/flow_stats.json", {"qbar": 1.0, "lambda": ELL, "L1": L1, "x_nonpositive_fraction": 0.0,
                                   "speed_over_qbar_percentiles": {"p0": 0.1, "p0.1": 0.2, "p1": 0.3, "p50": 0.9},
                                   "interior": {"mean_u": 1.0, "mean_v": 0.0, "mean_w": 0.0}})
    wjson(R + "/track.json", {"planes": PLANES, "times": TIMES, "X0": X0})
    with open(R + "/exits_time.csv", "w") as f:
        f.write("t,exit_inlet,exit_outlet\n")
        for j, t in enumerate(TIMES):
            f.write("%.15e,%d,%d\n" % (t, j, 2 * j))
    sig = [(0.008, 0.012), (0.02, 0.04), (0.05, 0.07)]
    prow = [(x, n, v2, v3) for x, n, (v2, v3) in zip(PLANES, n_planes, sig)]
    acts = actives or [100] * len(TIMES)
    trow = [(t, 3.0 * t + 1.0, 0.02 + 0.04 * t, 0.1 + 0.08 * t, a) for t, a in zip(TIMES, acts)]
    write_pass(R, prow, trow)
    write_pass(R, prow, trow, twin=True)
    return R


class KnownAnswer(unittest.TestCase):
    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.R = build(self._t.name)

    def tearDown(self):
        self._t.cleanup()

    def test_plane_and_window_values(self):
        # tau = 3: t_hi = 1.5 (snapshot j = 5), t_lo = 1.125 (between j = 3 and 4)
        v, un = rc.recompute_pass(self.R, self.R, 3.0, U)
        self.assertEqual(un, {})
        self.assertEqual((v["j_hi"], v["t_jhi"], v["t_hi"], v["t_lo"]), (5, 1.5, 1.5, 1.125))
        self.assertEqual((v["Xa"], v["Xb"], v["Xq"]), (10.0, 17.5, 6.25))
        for m in ("flux", "unif"):
            self.assertEqual(v[m + "|tp.n"], 100)
            self.assertAlmostEqual(v[m + "|tp.sigT2_Xa"], 0.03, places=15)
            self.assertAlmostEqual(v[m + "|tp.sigT2_Xb"], 0.06, places=15)
            self.assertAlmostEqual(v[m + "|tp.alpha_TP"], 0.03 / (2 * 0.5 * 7.5), places=15)  # 0.004
            self.assertEqual(v[m + "|win.n"], 100)
            self.assertAlmostEqual(v[m + "|win.alpha_e1"], 3.0 / (2 * ELL * U), places=14)  # slope / (2 ell U)
            self.assertAlmostEqual(v[m + "|win.alpha_e2"], 0.04 / (2 * ELL * U), places=15)
            self.assertAlmostEqual(v[m + "|win.alpha_e3"], 0.08 / (2 * ELL * U), places=15)
            self.assertAlmostEqual(v[m + "|win.alpha_Tfixed"], 0.06 / (2 * ELL * U), places=15)
        self.assertAlmostEqual(v["flux|tp.alpha_TP_q"], 0.02 / (2 * 0.5 * 3.75), places=15)
        self.assertAlmostEqual(v["flux|tp.delta_half"], 0.02 / 3.75 - 0.004, places=15)
        self.assertNotIn("unif|tp.alpha_TP_q", v)
        self.assertAlmostEqual(v["S_r"], 0.06 / (0.5 * 7.5), places=15)
        self.assertEqual((v["exit_inlet_jhi"], v["exit_outlet_jhi"]), (5, 10))
        self.assertEqual(v["tilt"], 0.0)

    def test_tilt(self):
        self.assertAlmostEqual(rc.tilt_angle((1.0, 1.0, 0.0)), np.pi / 4, places=15)
        self.assertAlmostEqual(rc.tilt_angle((0.0, 0.0, 2.0)), np.pi / 2, places=15)

    def test_window_between_snapshots_and_exact_snapshot(self):
        # t_hi = 1.4 between j = 4 (1.25) and j = 5 (1.5): tau = 2.8; t_lo = 1.05; linear var => same alpha
        v, un = rc.recompute_pass(self.R, self.R, 2.8, U)
        self.assertEqual((v["j_hi"], un), (5, {}))
        self.assertAlmostEqual(v["flux|win.alpha_e1"], 3.0, places=13)

    def test_window_errors(self):
        with self.assertRaises(ValueError):
            rc.window_edges(TIMES, 100.0, ELL, U)  # no snapshot >= t_hi
        with self.assertRaises(ValueError):
            rc.window_edges(TIMES, 0.1, ELL, U)  # t_lo < t_1

    def test_active_set_change_inside_window_needs_arrays(self):
        # a particle left between the bracket snapshots and j_hi (active 101 at j = 3, 100 from j = 4): not reproducible
        acts = [101, 101, 101, 101, 100, 100, 100, 100]
        build(self.R, actives=acts)
        v, un = rc.recompute_pass(self.R, self.R, 3.0, U)
        self.assertEqual(v["flux|win.n"], 100)
        self.assertNotIn("flux|win.alpha_e1", v)
        self.assertIn("flux|win.alpha_e1", un)
        self.assertIn("needs per-particle arrays", un["unif|win.alpha_Tfixed"])
        self.assertIn("flux|tp.alpha_TP", v)  # plane columns are unaffected

    def test_active_set_change_before_window_is_irrelevant(self):
        acts = [120, 110, 100, 100, 100, 100, 100, 100]  # exits happened before the first window snapshot (j = 3)
        build(self.R, actives=acts)
        v, un = rc.recompute_pass(self.R, self.R, 3.0, U)
        self.assertEqual(un, {})

    def test_contributor_mismatch_needs_arrays(self):
        build(self.R, n_planes=(100, 100, 97))
        v, un = rc.recompute_pass(self.R, self.R, 3.0, U)
        for c in ("tp.n", "tp.sigT2_Xa", "tp.sigT2_Xb", "tp.alpha_TP", "tp.alpha_TP_q", "tp.delta_half"):
            self.assertIn("flux|" + c, un)
            self.assertNotIn("flux|" + c, v)
        self.assertNotIn("S_r", v)
        self.assertIn("flux|win.alpha_e1", v)
        build(self.R, n_planes=(103, 100, 100))  # only the X_q -> X_a pair differs
        v, un = rc.recompute_pass(self.R, self.R, 3.0, U)
        self.assertIn("flux|tp.alpha_TP", v)
        self.assertEqual(sorted(k for k in un if k.startswith("flux|")), ["flux|tp.alpha_TP_q", "flux|tp.delta_half"])

    def test_inconsistent_inputs_are_refused(self):
        wjson(self.R + "/flow_stats.json", {"qbar": 1.0, "lambda": 0.6, "L1": L1, "x_nonpositive_fraction": 0.0,
                                            "speed_over_qbar_percentiles": {"p0": 0, "p0.1": 0, "p1": 0, "p50": 0},
                                            "interior": {"mean_u": 1.0, "mean_v": 0.0, "mean_w": 0.0}})
        with self.assertRaises(ValueError):
            rc.recompute_pass(self.R, self.R, 3.0, U)

    def test_compare_tolerances(self):
        self.assertEqual(rc.rel_diff(1.0, 1.0), 0.0)
        self.assertEqual(rc.rel_diff(float("nan"), float("nan")), 0.0)
        self.assertEqual(rc.rel_diff(1.0, float("nan")), float("inf"))
        self.assertAlmostEqual(rc.rel_diff(1.0, 1.0 + 1e-9), 1e-9 / (1 + 1e-9), places=15)
        row = {"flux": {"tp.alpha_TP": "0.0040000000001", "qbar": "1"}}
        res = rc.compare_realization({"flux|tp.alpha_TP": 0.004, "qbar": 1.0, "flux|tp.n": 5}, {}, row, 1e-10, 1e-15)
        by = {r[0]: r for r in res}
        self.assertTrue(by["flux.tp.alpha_TP"][4])  # 2.5e-11 relative is inside 1e-10
        self.assertEqual(sorted(by), ["flux.tp.alpha_TP", "qbar"])  # tp.n absent from the committed row: not compared

    def test_compare_tolerances_pass_and_fail(self):
        row = {"flux": {"tp.alpha_TP": "0.0040000000001"}}
        res = rc.compare_realization({"flux|tp.alpha_TP": 0.004}, {}, row, 1e-10, 1e-15)
        self.assertTrue(res[0][4])
        res = rc.compare_realization({"flux|tp.alpha_TP": 0.004}, {}, row, 1e-12, 1e-15)
        self.assertFalse(res[0][4])

    def test_set_to_arms(self):
        self.assertEqual(rc.arms_and_div("s4_prod"), (["s4_A", "s4_B"], 8))
        self.assertEqual(rc.arms_and_div("s4_prod_twinpairs"), (["s4_A", "s4_B"], 8))
        self.assertEqual(rc.arms_and_div("s225_A_hmax16"), (["s225_A"], 16))
        self.assertEqual(rc.arms_and_div("s4_n16_first30"), (["s4_n16"], 8))
        self.assertEqual(rc.arms_and_div("s4_w12"), (["s4_w12", "s4_w12c"], 8))
        self.assertEqual(rc.arms_and_div("s1_tw"), (["s1_tw"], 8))

    def test_positions(self):
        self.assertEqual(rc.parse_positions("0,2-4"), [0, 2, 3, 4])


class Par2Values(unittest.TestCase):
    def test_known_answer(self):
        # mean_x = 5 + 2 t, inject_x = 5 => U_c = 2 for every g; var_k = slope_k(g) t => alpha_k = slope_k / (2 lam U_c)
        lam, tau_c, t_b = 1.0, 10.0, 8.0
        t = list(np.linspace(0.0, 20.0, 81))
        series = []
        for g in range(5):
            series.append({"t": t, "mean_x": [5 + 2 * s for s in t], "var_x": [(3 + g) * s for s in t],
                           "var_y": [(0.4 + 0.1 * g) * s for s in t], "var_z": [(0.2 + 0.05 * g) * s for s in t]})
        v = rc.par2_values(series, lam, 5.0, t_b, tau_c)
        self.assertAlmostEqual(v["U_c"], 2.0, places=14)
        self.assertAlmostEqual(v["t_hi"], 5.0, places=14)
        self.assertAlmostEqual(v["t_lo"], 3.75, places=14)
        np.testing.assert_allclose(v["alpha"]["x"], [(3 + g) / 4.0 for g in range(5)], rtol=1e-13)
        np.testing.assert_allclose(v["alpha"]["T"], [(0.3 + 0.075 * g) / 4.0 for g in range(5)], rtol=1e-13)

    def test_interp_outside_series_is_an_error(self):
        s = {"t": [0.0, 1.0], "var_x": [0.0, 1.0]}
        with self.assertRaises(ValueError):
            rc.interp_series(s, "var_x", 2.0)
        self.assertEqual(rc.interp_series(s, "var_x", 0.25), 0.25)


class Par2Gate5Driver(unittest.TestCase):
    """par2_gate5: pilot (20) + extension series pooled by global index, against a summary.json built from the same values."""

    def setUp(self):
        self._t = tempfile.TemporaryDirectory()
        self.addCleanup(self._t.cleanup)
        self.out = os.path.join(self._t.name, "host")
        n_pilot, n = 20, 22
        self.series = []
        for g in range(n):
            d = os.path.join(self.out, "par2", "raw", "par2_s4" if g < n_pilot else "par2_s4_ext", "stats")
            os.makedirs(d, exist_ok=True)
            t = np.linspace(0.0, 20.0, 41)
            cols = {"t": t, "mean_x": 5 + 2 * t, "var_x": (3 + 0.1 * g) * t, "var_y": (0.4 + 0.01 * g) * t, "var_z": (0.2 + 0.02 * g * (g % 3)) * t}
            r = g if g < n_pilot else g - n_pilot
            with open(os.path.join(d, "realization_%04d_timeseries.csv" % r), "w") as f:
                f.write("# format_version=1\nt,mean_x,mean_y,mean_z,var_x,var_y,var_z,active\n")
                for k in range(len(t)):
                    f.write("%.17g,%.17g,0,0,%.17g,%.17g,%.17g,100\n" % (t[k], cols["mean_x"][k], cols["var_x"][k], cols["var_y"][k], cols["var_z"][k]))
            self.series.append({k: list(v) for k, v in cols.items()})
        self.taus = {"s4": {"gate5": {"tau_c": 10.0, "par2": {"corr_length": 1.0, "inject_x": 5.0, "t_b_min": 8.0}}}}
        v = rc.par2_values(self.series, 1.0, 5.0, 8.0, 10.0)
        agg = lambda a: {"mean": float(a.mean()), "sd": float(a.std(ddof=1))}
        self.summary = {"amplitudes": {"s4": {"C": {"par2": {"n": n, "U_c": v["U_c"], "window": [v["t_lo"], v["t_hi"]],
                                                              "alpha_x": agg(v["alpha"]["x"]), "alpha_y": agg(v["alpha"]["y"]),
                                                              "alpha_z": agg(v["alpha"]["z"]), "alpha_T_fixed": agg(v["alpha"]["T"])}}}}}

    def test_reproduces_summary_and_detects_a_change(self):
        res, v = rc.par2_gate5(self.out, "unused", "s4", self.taus, self.summary)
        self.assertEqual(len(res), 11)
        self.assertTrue(all(x[4] for x in res), res)
        self.assertAlmostEqual(v["U_c"], 2.0, places=13)
        self.summary["amplitudes"]["s4"]["C"]["par2"]["alpha_y"]["sd"] *= 1 + 1e-6
        res, _ = rc.par2_gate5(self.out, "unused", "s4", self.taus, self.summary)
        self.assertEqual([x[0] for x in res if not x[4]], ["par2.alpha_y.sd"])

    def test_missing_series_is_a_failure_in_run(self):
        os.remove(os.path.join(self.out, "par2", "raw", "par2_s4_ext", "stats", "realization_0001_timeseries.csv"))
        with self.assertRaises(OSError):
            rc.par2_gate5(self.out, "unused", "s4", self.taus, self.summary)


# ------------------------------------------------------------------------------------------------
# cross-check against the SF-39 reducer on a full fake runner tree
# ------------------------------------------------------------------------------------------------
def _import_sf39():
    if not os.path.isdir(ARTIFACT_SF39):
        return None
    sys.path.insert(0, os.path.join(ARTIFACT_SF39, "scripts"))
    try:
        import sf39.reduce as red
        from sf39.tests import fixtures_runner as fx
        return red, fx
    except Exception:  # pragma: no cover - artifact not importable
        return None
    finally:
        sys.path.pop(0)


SF39 = _import_sf39()


@unittest.skipIf(SF39 is None, "SF-39 artifact (reducer and fixtures) not importable")
class AgainstReducer(unittest.TestCase):
    def build_tree(self, tau, plane_reach):
        red, fx = SF39
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        out = os.path.join(self.tmp.name, "host")
        art = os.path.join(self.tmp.name, "art")
        with mock.patch.object(fx, "PLANE_REACH", plane_reach):
            fx.build_run(os.path.join(out, "runs", "s4_A"), multi=True, nr=3, divs=(8, 16, 32), pollock=True)
            recs = []
            for r in range(3):
                real_dir = os.path.join(out, "runs", "s4_A", "realizations", "r_%04d" % r)
                recs.append(real_dir)
            qbar = np.array([rc.read_json(d + "/flow_stats.json")["qbar"] for d in recs])
            Uset = float(qbar.mean())
            rows = []
            for r, d in enumerate(recs):
                real = red.Realization(d, 8)
                common, methods = red.reduce_methods(real, tau, Uset, r=r)
                for row in red.long_rows(common, methods, keep=tuple(common)):
                    rows.append(dict(set="s4_prod", hmax_div=8, **row))
        os.makedirs(os.path.join(art, "analysis", "reduced"))
        red.write_csv(os.path.join(art, "analysis", "reduced", "s4_prod.csv"), rows)
        wjson(os.path.join(art, "raw", "campaign", "checkpoint", "tau_b_star.json"),
              {"amplitudes": {"s4": {"tau_b_star": tau}}, "n_pilot": 20})
        wjson(os.path.join(art, "analysis", "summary.json"), {"amplitudes": {}})
        return out, art

    def run_rc(self, out, art, positions=(0, 1, 2)):
        buf = io.StringIO()
        code = rc.run(out, art, ["s4_prod"], list(positions), buf, par2=False)
        return code, buf.getvalue()

    def test_window_columns_reproduce_reducer_and_plane_columns_flagged(self):
        out, art = self.build_tree(2.0, plane_reach=dict(SF39[1].PLANE_REACH))
        code, text = self.run_rc(out, art)
        self.assertEqual(code, 0, text)
        self.assertIn("RECOMPUTE PASS", text)
        self.assertIn("OK   flux.win.alpha_e2", text)
        self.assertIn("OK   unif.win.alpha_Tfixed", text)
        self.assertIn("ARRAYS flux.tp.alpha_TP", text)  # contributors at X_a != X_b in the fixture
        self.assertNotIn("FAIL", text)

    def test_plane_columns_reproduce_reducer_when_contributors_agree(self):
        out, art = self.build_tree(2.0, plane_reach={5: 1})
        code, text = self.run_rc(out, art)
        self.assertEqual(code, 0, text)
        for lab in ("flux.tp.alpha_TP", "flux.tp.sigT2_Xa", "flux.tp.alpha_TP_q", "flux.tp.delta_half", "unif.tp.alpha_TP",
                    "S_r", "tilt", "exit_outlet_jhi", "t_jhi", "flux.win.n"):
            self.assertIn("OK   %-26s" % lab, text)
        self.assertNotIn("ARRAYS flux.tp", text)
        # tolerance is real: the max relative difference printed is far below 1e-10 and not an artefact of skipping
        line = [ln for ln in text.splitlines() if ln.startswith("RECOMPUTE compared")][0]
        self.assertIn("beyond_tolerance=0", line)

    def test_late_window_reports_needs_arrays_not_a_mismatch(self):
        out, art = self.build_tree(10.0, plane_reach={5: 1})  # window after the first exits: S_r != active set
        code, text = self.run_rc(out, art)
        self.assertEqual(code, 0, text)
        self.assertIn("ARRAYS flux.win.alpha_e1", text)
        self.assertIn("OK   flux.tp.alpha_TP", text)

    def test_corrupted_committed_value_is_detected(self):
        out, art = self.build_tree(2.0, plane_reach={5: 1})
        p = os.path.join(art, "analysis", "reduced", "s4_prod.csv")
        with open(p, newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            if r["seed"] == "5001" and r["method"] == "flux":
                r["win.alpha_e2"] = repr(float(r["win.alpha_e2"]) * (1 + 1e-6))
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        code, text = self.run_rc(out, art)
        self.assertEqual(code, 1, text)
        self.assertIn("FAIL flux.win.alpha_e2", text)

    def test_never_writes_under_out_root(self):
        out, art = self.build_tree(2.0, plane_reach={5: 1})
        before = sorted((os.path.join(d, n), os.stat(os.path.join(d, n)).st_mtime_ns) for d, _, fs in os.walk(out) for n in fs)
        dirs_before = sorted(d for d, _, _ in os.walk(out))
        self.run_rc(out, art)
        after = sorted((os.path.join(d, n), os.stat(os.path.join(d, n)).st_mtime_ns) for d, _, fs in os.walk(out) for n in fs)
        self.assertEqual(before, after)
        self.assertEqual(dirs_before, sorted(d for d, _, _ in os.walk(out)))

    def test_cli_refuses_report_inside_out_root(self):
        out, art = self.build_tree(2.0, plane_reach={5: 1})
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            rc.main(["--out-root", out, "--artifact", art, "--sets", "s4_prod", "--report", os.path.join(out, "report.txt")])
        rep = os.path.join(self.tmp.name, "report.txt")
        with contextlib.redirect_stdout(io.StringIO()):
            code = rc.main(["--out-root", out, "--artifact", art, "--sets", "s4_prod", "--no-par2", "--report", rep])
        self.assertEqual(code, 0)
        with open(rep) as f:
            self.assertIn("RECOMPUTE PASS", f.read())


class NoAnalysisImports(unittest.TestCase):
    def test_module_imports_only_numpy_and_stdlib(self):
        with open(rc.__file__) as f:
            src = f.read()
        import re
        mods = set(re.findall(r"^(?:import|from)\s+([A-Za-z_][\w]*)", src, re.M))
        self.assertEqual(mods - {"argparse", "csv", "json", "math", "os", "re", "sys", "numpy"}, set())


if __name__ == "__main__":
    unittest.main()
