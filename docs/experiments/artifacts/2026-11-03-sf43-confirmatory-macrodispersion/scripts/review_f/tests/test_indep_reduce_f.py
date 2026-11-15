"""Tests of the independent reduction script of acceptance F.  The tests may import project modules (sf42red, as the cross-check
reference); the script under test may not (test_imports).  Run from the repository root:
  python3 -W error -m unittest discover -s <A>/scripts/review_f/tests -t <A>/scripts/review_f/tests
"""
import ast
import contextlib
import csv
import io
import json
import math
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR = os.path.dirname(HERE)
SF42 = os.path.join(SCRIPT_DIR, *([os.pardir] * 3), "2026-10-30-sf42-confirmatory-tooling", "scripts")
FIX = os.path.join(SF42, "sf42red", "tests", "fixture")
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.normpath(SF42))

import indep_reduce_f as F  # noqa: E402
from sf42red import reduce as red  # noqa: E402

TABLE_COLS = ["code", "source", "r", "hmax_div", "seed", "family", "weights", "basis", "set", "alpha_TP", "alpha_TAP", "alpha_LP",
              "U_r", "n_traced", "n_first", "n_last", "n_min", "n_max"]


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def run_main(args):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = F.main(args)
    return rc, out.getvalue(), err.getvalue()


def make_campaign(root, arm, seg, seed0, nr, seeds_override=None):
    """campaign tree skeleton; returns (segment dir, realizations dir)."""
    sd = os.path.join(root, "segments", arm, seg)
    rd = os.path.join(sd, "out", "realizations")
    os.makedirs(rd)
    write_json(os.path.join(sd, "seedmap.json"), {"seed0": seed0, "nr": nr, "seeds": seeds_override or {str(r): seed0 + r for r in range(nr)}})
    return sd, rd


def write_table(root, arm, seg, name, rows):
    d = os.path.join(root, "reduced", arm, seg)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "%s__%s__%s.csv" % (arm, seg, name)), "w", newline="") as f:
        wr = csv.DictWriter(f, TABLE_COLS)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in TABLE_COLS})


def standardize(z, w):
    """Weighted-centre z (Np,k) and whiten to unit weighted covariance (w sums to 1)."""
    z = z - w @ z
    L = np.linalg.cholesky((z * w[:, None]).T @ z)
    return z @ np.linalg.inv(L).T


# ---------------------------------------------------------------------------------------------- synthetic with a known answer
ELL, X0, D_LO, D_HI = 0.25, 1.25, 25, 94
G, S0 = 0.37, 0.2          # var x2 = var x3 = S0 + G (x1 - x0)  -> alpha_TP = G / (2 ell)
CT, U = 0.011, 1.3         # mean tau = (x1 - x0)/U, var tau = 0.5 + CT (x1 - x0) -> alpha_LP = U^2 CT / (2 ell)
MTILT = 0.4                # mean displacement x2 = MTILT (x1 - x0), x3 = 0 -> alpha_TAP = alpha_TP (1 + 1/(1+m^2))/2


def synthetic_arrays(Np=400, seed=7):
    rng = np.random.default_rng(seed)
    planes = X0 + ELL * np.arange(20, 100)               # d = 20..99: the window selects 25..94
    w = rng.uniform(0.1, 2.0, Np)
    w[::37] = 0.0                                        # untraced particles
    pos = w > 0
    wn = w[pos] / w[pos].sum()
    z = standardize(rng.standard_normal((pos.sum(), 3)), wn)   # columns: x2, x3, tau; unit weighted covariance
    seeds = np.column_stack([np.full(Np, X0), rng.uniform(0, 5, Np), rng.uniform(0, 5, Np)])
    X = np.full((Np, len(planes), 2), np.nan)
    tau = np.full((Np, len(planes)), np.nan)
    s = planes - X0
    for j in range(len(planes)):
        sd = math.sqrt(S0 + G * s[j])
        X[pos, j, 0] = seeds[pos, 1] + MTILT * s[j] + sd * z[:, 0]
        X[pos, j, 1] = seeds[pos, 2] + sd * z[:, 1]
        tau[pos, j] = s[j] / U + math.sqrt(0.5 + CT * s[j]) * z[:, 2]
    # beyond-window records: some particles do not reach the last planes (NaN); the window planes stay complete
    X[pos.nonzero()[0][::5], -3:, :] = np.nan
    tau[pos.nonzero()[0][::5], -3:] = np.nan
    return planes, w, seeds, X, tau


def write_pass(pdir, planes, w, seeds, X, tau):
    os.makedirs(pdir, exist_ok=True)
    write_json(os.path.join(pdir, "track.json"), {"planes": planes.tolist(), "X0": X0, "M": 20, "times": [0.0]})
    for n, a in (("weights", w), ("seeds", seeds), ("planes_x23", X), ("planes_tau", tau)):
        np.save(os.path.join(pdir, n + ".npy"), a)


def write_realization(rdir, seed, multipass, arrays):
    os.makedirs(rdir)
    open(os.path.join(rdir, "DONE"), "w").close()
    write_json(os.path.join(rdir, "field.json"), {"ell": ELL, "seed": seed})
    write_json(os.path.join(rdir, "flow_stats.json"), {"lambda": ELL, "qbar": U})
    if multipass:
        for d in (8, 16):
            write_pass(os.path.join(rdir, "hmax_%d" % d), *arrays)
    else:
        write_pass(rdir, *arrays)


class TestSyntheticAnalytic(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.arrays = synthetic_arrays()
        a_tp = G / (2 * ELL)
        self.exact = {"alpha_TP": a_tp, "alpha_TAP": a_tp * (1 + 1 / (1 + MTILT ** 2)) / 2, "alpha_LP": U ** 2 * CT / (2 * ELL), "U_r": U}

    def tearDown(self):
        self._tmp.cleanup()

    def build(self, multipass, perturb=0.0):
        arm, seg = "s1_prod", "u000001_s432071"
        sd, rd = make_campaign(self.root, arm, seg, 432071, 12)
        write_realization(os.path.join(rd, "r_0009"), 432080, multipass, self.arrays)
        row = dict(self.exact, code="consumer", r=9, hmax_div=8, seed=432080, family="plane", weights="flux", basis="disp", set="perplane")
        mine = F.reduce_realization(os.path.join(rd, "r_0009"), os.path.join(rd, "r_0009", "hmax_8") if multipass else os.path.join(rd, "r_0009"),
                                    D_LO, D_HI)
        row.update({k: mine[k] for k in ("n_traced", "n_first", "n_last", "n_min", "n_max")})
        row["alpha_TP"] = self.exact["alpha_TP"] * (1 + perturb)
        write_table(self.root, arm, seg, "d25-94", [row])
        return arm, mine

    def test_reproduces_the_analytic_values_to_1e12(self):
        for multipass in (False, True):
            with self.subTest(multipass=multipass):
                shutil.rmtree(self.root)
                os.makedirs(self.root)
                _, mine = self.build(multipass)
                for k, v in self.exact.items():
                    self.assertAlmostEqual(mine[k] / v, 1.0, delta=1e-12, msg=k)
                self.assertEqual(mine["n_traced"], 400 - len(range(0, 400, 37)))
                self.assertEqual(mine["n_min"], mine["n_max"])           # window planes complete by construction
                self.assertEqual(mine["_info"]["n_planes"], 70)

    def test_cli_pass_and_fail(self):
        arm, _ = self.build(multipass=True)
        rc, out, err = run_main(["--campaign", self.root, "--arm", arm, "--seed", "432080"])
        self.assertEqual((rc, err), (0, ""), out)
        self.assertIn("F PASS", out)
        self.assertIn("layout multi-pass hmax_8", out)
        shutil.rmtree(self.root)
        os.makedirs(self.root)
        arm, _ = self.build(multipass=False, perturb=1e-8)
        rc, out, err = run_main(["--campaign", self.root, "--arm", arm, "--seed", "432080"])
        self.assertEqual(rc, 1, out)
        self.assertIn("F FAIL", out)
        self.assertIn("BEYOND TOLERANCE", out)

    def test_tolerance_edge(self):
        arm, _ = self.build(multipass=False, perturb=5e-11)       # inside 1e-10
        self.assertEqual(run_main(["--campaign", self.root, "--arm", arm, "--seed", "432080"])[0], 0)

    def test_ambiguity_and_missing_are_errors(self):
        arm, _ = self.build(multipass=False)
        # same seed in a second segment
        sd2, rd2 = make_campaign(self.root, arm, "u000002_s432071", 432071, 12)
        rc, _, err = run_main(["--campaign", self.root, "--arm", arm, "--seed", "432080"])
        self.assertEqual(rc, 2)
        self.assertIn("ambiguous", err)
        shutil.rmtree(sd2)
        # both layouts at once
        rdir = os.path.join(self.root, "segments", arm, "u000001_s432071", "out", "realizations", "r_0009")
        write_pass(os.path.join(rdir, "hmax_8"), *self.arrays)
        rc, _, err = run_main(["--campaign", self.root, "--arm", arm, "--seed", "432080"])
        self.assertEqual(rc, 2)
        self.assertIn("ambiguous layout", err)
        # unknown seed
        rc, _, err = run_main(["--campaign", self.root, "--arm", arm, "--seed", "1"])
        self.assertEqual(rc, 2)
        self.assertIn("not found", err)

    def test_missing_pass_lists_available(self):
        arm, _ = self.build(multipass=True)
        rc, _, err = run_main(["--campaign", self.root, "--arm", arm, "--seed", "432080", "--hmax-div", "32"])
        self.assertEqual(rc, 2)
        self.assertIn("hmax_8", err)


# ---------------------------------------------------------------------------------------------- real tiny runner output vs sf42red
class TestAgainstSf42redOnRealFixture(unittest.TestCase):
    """The fixture is a tiny REAL consumer output (sf42red/tests/fixture/README.txt): windows d in [4, 20] of 5 planes; realization 1 has
    outlet exits, i.e. different particle sets per plane.  Reference values: sf42red reduce_all (the project reducer), tolerance 1e-10."""

    def check(self, multipass):
        with tempfile.TemporaryDirectory() as root:
            arm, seg = "arm", "u000001_s420501"
            _, rd = make_campaign(root, arm, seg, 420501, 2)
            for r in (0, 1):
                src = os.path.join(FIX, "arm", "realizations", "r_%04d" % r)
                dst = os.path.join(rd, "r_%04d" % r)
                if multipass:
                    os.makedirs(os.path.join(dst, "hmax_8"))
                    for n in ("DONE", "field.json", "flow_stats.json"):
                        shutil.copy(os.path.join(src, n), dst)
                    for n in ("track.json", "seeds.npy", "weights.npy", "planes_x23.npy", "planes_tau.npy"):
                        shutil.copy(os.path.join(src, n), os.path.join(dst, "hmax_8"))
                else:
                    shutil.copytree(src, dst)
            tab = os.path.join(root, "ref.csv")
            if multipass:   # sf42red reads the full pass files: give it a copy with every array
                full = os.path.join(root, "full")
                shutil.copytree(os.path.join(FIX, "arm"), full)
                for r in (0, 1):
                    dst = os.path.join(full, "realizations", "r_%04d" % r)
                    os.makedirs(os.path.join(dst, "hmax_8"))
                    for n in os.listdir(dst):
                        if n not in ("DONE", "field.json", "flow_stats.json", "hmax_8"):
                            shutil.move(os.path.join(dst, n), os.path.join(dst, "hmax_8", n))
                red.reduce_all([full], [], [], 0.25, 1.25, 4.0, 20.0, None, tab, root)
            else:
                red.reduce_all([os.path.join(FIX, "arm")], [], [], 0.25, 1.25, 4.0, 20.0, None, tab, FIX, single_pass_hmax=8)
            with open(tab) as f:
                ref = [r for r in csv.DictReader(f) if r["code"] == "consumer" and r["family"] == "plane" and r["weights"] == "flux"
                       and r["set"] == "perplane" and r["basis"] == "disp"]
            self.assertEqual(len(ref), 2)
            os.makedirs(os.path.join(root, "reduced", arm, seg))
            shutil.copy(tab, os.path.join(root, "reduced", arm, seg, "%s__%s__d4-20.csv" % (arm, seg)))
            for r, seed in ((0, 420501), (1, 420502)):
                rc, out, err = run_main(["--campaign", root, "--arm", arm, "--seed", str(seed), "--d-lo", "4", "--d-hi", "20"])
                self.assertEqual((rc, err), (0, ""), out)
                self.assertIn("F PASS", out)
                row = next(x for x in ref if int(x["seed"]) == seed)
                mine = F.reduce_realization(os.path.join(rd, "r_%04d" % r), os.path.join(rd, "r_%04d" % r, "hmax_8") if multipass else
                                            os.path.join(rd, "r_%04d" % r), 4, 20)
                for k in ("alpha_TP", "alpha_TAP", "alpha_LP", "U_r"):
                    self.assertTrue(abs(mine[k] - float(row[k])) <= 1e-10 * abs(float(row[k])), (seed, k, mine[k], row[k]))
                    self.assertTrue(math.isfinite(mine[k]))
                for k in ("n_traced", "n_first", "n_last", "n_min", "n_max"):
                    self.assertEqual(mine[k], int(row[k]), k)

    def test_messy_sets_in_the_production_window(self):
        """Random (non-analytic) arrays with NaN records INSIDE the window, zero weights: per-plane sets differ."""
        planes, w, seeds, X, tau = synthetic_arrays(seed=11)
        rng = np.random.default_rng(5)
        X = X + rng.normal(0, 0.05, X.shape)                    # destroy the closed forms
        tau = tau + rng.normal(0, 0.01, tau.shape)
        for j in range(30, 80):                                 # records lost at random inside the window, prefix-complete per particle
            lost = rng.random(len(w)) < 0.0004 * (j - 29)
            for jj in range(j, len(planes)):
                X[lost, jj, :] = np.nan
                tau[lost, jj] = np.nan
        with tempfile.TemporaryDirectory() as root:
            rdir = os.path.join(root, "r_0000")
            write_realization(rdir, 1, False, (planes, w, seeds, X, tau))
            np.save(os.path.join(rdir, "planes_I.npy"), np.zeros_like(tau))
            ref = [r for r in red.consumer_plane_rows(red.Recorder(), rdir, None, None, 25, 94, 8)[0]
                   if r["weights"] == "flux" and r["set"] == "perplane" and r["basis"] == "disp"][0]
            mine = F.reduce_realization(rdir, rdir, 25, 94)
        self.assertGreater(mine["n_max"], mine["n_min"])
        for k in ("alpha_TP", "alpha_TAP", "alpha_LP", "U_r"):
            self.assertLessEqual(abs(mine[k] - ref[k]), 1e-10 * abs(ref[k]), k)
        for k in ("n_traced", "n_first", "n_last", "n_min", "n_max"):
            self.assertEqual(mine[k], ref[k], k)

    def test_single_pass_layout(self):
        self.check(False)

    def test_multi_pass_layout(self):
        self.check(True)


class TestIndependence(unittest.TestCase):
    def test_imports_only_stdlib_and_numpy(self):
        with open(os.path.join(SCRIPT_DIR, "indep_reduce_f.py")) as f:
            tree = ast.parse(f.read())
        mods = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                mods |= {a.name.split(".")[0] for a in n.names}
            elif isinstance(n, ast.ImportFrom):
                self.assertEqual(n.level, 0, "relative import")
                mods.add(n.module.split(".")[0])
        self.assertEqual(mods - set(sys.stdlib_module_names), {"numpy"}, mods)


if __name__ == "__main__":
    unittest.main()
