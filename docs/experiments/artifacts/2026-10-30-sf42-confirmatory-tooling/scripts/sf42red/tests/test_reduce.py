"""Known-answer tests of the SF-42 N5 reducer (synthetic clouds with exactly known moments; see synth.py)."""
import csv
import hashlib
import math
import os
import sys
import tempfile
import types
import unittest

import numpy as np

from sf42red import reduce as red
from sf42red.tests import synth

ELL, X0 = 0.25, 1.25
A_T, C_TAU, U0 = 0.03, 0.02, 2.0
ALPHA_LP = 0.5 * U0 ** 2 * C_TAU / ELL
TOL = 1e-12


def slurp(path):
    with open(path, "rb") as f:
        return f.read()


def read_table(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def pick(rows, **kw):
    out = [r for r in rows if all(r[k] == str(v) for k, v in kw.items())]
    assert len(out) == 1, (kw, len(out))
    return out[0]


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = self._tmp.name
        self.addCleanup(self._tmp.cleanup)

    def reduce(self, consumer=(), par2=(), series=(), d=(25.0, 94.0), **kw):
        out = os.path.join(self.tmp, "table.csv")
        red.reduce_all(list(consumer), list(par2), list(series), kw.pop("ell", None), kw.pop("x0", None), d[0], d[1],
                       kw.pop("U_time", None), out, kw.pop("root", self.tmp), **kw)
        return read_table(out)


class TestEstimators(Base):
    def test_alpha_exact_linear_growth(self):
        synth.write_pass(os.path.join(self.tmp, "arm"), ell=ELL, x0=X0, a_T=A_T, c_tau=C_TAU, U0=U0)
        rows = [r for r in self.reduce([os.path.join(self.tmp, "arm")]) if r["family"] == "plane"]
        self.assertEqual(len(rows), 12)  # 3 weights x 2 bases x 2 sets
        for w in ("flux", "unif"):
            for b in ("pos", "disp"):
                for s in ("perplane", "common"):
                    r = pick(rows, weights=w, basis=b, set=s)
                    self.assertAlmostEqual(float(r["alpha_TP"]), A_T, delta=TOL, msg=(w, b, s))
                    self.assertAlmostEqual(float(r["alpha_LP"]), ALPHA_LP, delta=TOL)
                    self.assertAlmostEqual(float(r["U_r"]), U0, delta=TOL * U0)
                    self.assertEqual(r["note"], "")
                    self.assertEqual((r["n_planes"], r["d_first"], r["d_last"]), ("5", "25", "90"))

    def test_expI_differs_and_is_finite(self):
        synth.write_pass(os.path.join(self.tmp, "arm"), ell=ELL, x0=X0)
        rows = self.reduce([os.path.join(self.tmp, "arm")])
        e = pick(rows, family="plane", weights="expI", basis="pos", set="perplane")
        f = pick(rows, family="plane", weights="flux", basis="pos", set="perplane")
        self.assertTrue(math.isfinite(float(e["alpha_LP"])))
        self.assertGreater(abs(float(e["alpha_LP"]) - float(f["alpha_LP"])), 1e-6)

    def test_tilted_cloud_alpha_TAP_vs_TP(self):
        drift = 0.5  # mean displacement x2 per unit x1: e_L = (1, 0.5, 0)/|.|
        synth.write_pass(os.path.join(self.tmp, "arm"), ell=ELL, x0=X0, drift=(drift, 0.0))
        rows = self.reduce([os.path.join(self.tmp, "arm")])
        r = pick(rows, family="plane", weights="unif", basis="disp", set="perplane")
        eLx2 = 1.0 / (1.0 + drift ** 2)
        # isotropic displacement covariance g^2 I: var along each unit-free u is g^2 (1 - u_x^2); mean over u1,u2 = g^2 (1 + eLx^2)/2
        self.assertAlmostEqual(float(r["alpha_TP"]), A_T, delta=TOL)
        self.assertAlmostEqual(float(r["alpha_TAP"]), A_T * (1.0 + eLx2) / 2.0, delta=TOL)
        self.assertAlmostEqual(float(r["tilt"]), math.atan(drift), delta=1e-12)
        self.assertLess(float(r["alpha_TAP"]), float(r["alpha_TP"]))
        # no drift: TAP = (1 + 1)/2 TP
        synth.write_pass(os.path.join(self.tmp, "arm0"), ell=ELL, x0=X0)
        r0 = pick(self.reduce([os.path.join(self.tmp, "arm0")]), family="plane", weights="unif", basis="disp", set="perplane")
        self.assertAlmostEqual(float(r0["alpha_TAP"]), A_T, delta=TOL)

    def test_weighted_moments_by_hand(self):
        m = red.plane_moments(np.array([[-1.0, 0.0], [1.0, 0.0]]), np.array([1.0, 3.0]), np.array([0.75, 0.25]))
        self.assertAlmostEqual(m["mu"][0], -0.5, delta=1e-15)
        self.assertAlmostEqual(m["C"][0, 0], 0.75, delta=1e-15)
        self.assertAlmostEqual(m["mt"], 1.5, delta=1e-15)
        self.assertAlmostEqual(m["vt"], 0.75 * 0.25 + 0.25 * 2.25, delta=1e-15)
        # weights are renormalized: scaling them changes nothing
        m2 = red.plane_moments(np.array([[-1.0, 0.0], [1.0, 0.0]]), np.array([1.0, 3.0]), np.array([7.5, 2.5]))
        self.assertAlmostEqual(m2["C"][0, 0], 0.75, delta=1e-15)
        with self.assertRaises(red.ReduceError):
            red.plane_moments(np.zeros((0, 2)), np.zeros(0), np.zeros(0))

    def test_uniform_vs_flux_planes_table_against_numpy(self):
        M = 8
        w = 1.0 + 0.5 * ((np.arange(M * M) // M) % 2 == 0)  # depends on m2 only; breaks the symmetry of Z3 = (-1)^m2
        synth.write_pass(os.path.join(self.tmp, "arm"), ell=ELL, x0=X0, flux_weights=w)
        rows = self.reduce([os.path.join(self.tmp, "arm")])
        pl = read_table(os.path.join(self.tmp, "table.planes.csv"))
        X = np.load(os.path.join(self.tmp, "arm", "realizations", "r_0000", "planes_x23.npy"))
        wn = w / w.sum()
        for k in (0, 3, 6):
            C = np.cov(X[:, k, :].T, aweights=wn, bias=True)
            r = pick(pl, weights="flux", basis="pos", plane=k)
            self.assertAlmostEqual(float(r["var_x2"]), C[0, 0], delta=1e-14)
            self.assertAlmostEqual(float(r["cov_x2x3"]), C[0, 1], delta=1e-14)
            Cu = np.cov(X[:, k, :].T, bias=True)
            ru = pick(pl, weights="unif", basis="pos", plane=k)
            self.assertAlmostEqual(float(ru["var_x3"]), Cu[1, 1], delta=1e-14)
        f = pick(rows, family="plane", weights="flux", basis="pos", set="perplane")
        u = pick(rows, family="plane", weights="unif", basis="pos", set="perplane")
        self.assertGreater(abs(float(f["alpha_TP"]) - float(u["alpha_TP"])), 1e-9)
        self.assertEqual(sum(1 for r in pl if r["weights"] == "flux" and r["basis"] == "pos"), 8)  # every plane of the run

    def test_displacement_vs_position(self):
        synth.write_pass(os.path.join(self.tmp, "arm"), ell=ELL, x0=X0)
        pl = None
        self.reduce([os.path.join(self.tmp, "arm")])
        pl = read_table(os.path.join(self.tmp, "table.planes.csv"))
        for k in range(8):
            p = pick(pl, weights="unif", basis="pos", plane=k)
            d = pick(pl, weights="unif", basis="disp", plane=k)
            off = (float(p["x1"]) - X0)
            self.assertAlmostEqual(float(d["var_x2"]), 2 * A_T * ELL * off, delta=1e-13)  # g^2 exactly
            self.assertAlmostEqual(float(d["cov_x2x3"]), 0.0, delta=1e-14)
            # position variance = var(seed) + g^2 (seed lattice, M = 8: var of (m+.5)/8)
            self.assertAlmostEqual(float(p["var_x2"]) - float(d["var_x2"]), (8 ** 2 - 1) / 12.0 / 8 ** 2, delta=1e-13)


class TestSets(Base):
    def test_set_difference_flags_and_common_variant(self):
        synth.write_pass(os.path.join(self.tmp, "arm"), ell=ELL, x0=X0, miss={3: 6, 10: 5})  # window planes are indices 2..6
        rows = self.reduce([os.path.join(self.tmp, "arm")])
        r = pick(rows, family="plane", weights="unif", basis="pos", set="perplane")
        self.assertEqual((r["n_first"], r["n_last"], r["n_min"], r["n_max"], r["n_common"], r["set_differs"]), ("64", "62", "62", "64", "62", "1"))
        c = pick(rows, family="plane", weights="unif", basis="pos", set="common")
        self.assertEqual((c["n_min"], c["n_max"], c["set_differs"]), ("62", "62", "0"))
        # counts on the window planes (indices 2..6): 64, 64, 64, 63 (particle 10 lost), 62 (particles 3 and 10 lost)
        self.assertAlmostEqual(float(r["wfrac_min"]), 62 / 64, delta=1e-15)
        # the estimators are still defined on the nested per-plane sets
        self.assertEqual(r["note"], "")
        self.assertTrue(math.isfinite(float(r["alpha_TP"])))

    def test_empty_plane_gives_nan_and_note(self):
        synth.write_pass(os.path.join(self.tmp, "arm"), ell=ELL, x0=X0, miss={p: 6 for p in range(64)})
        rows = self.reduce([os.path.join(self.tmp, "arm")])
        r = pick(rows, family="plane", weights="unif", basis="pos", set="perplane")
        self.assertTrue(math.isnan(float(r["alpha_TP"])))
        self.assertIn("empty particle set", r["note"])

    def test_window_selection_and_errors(self):
        synth.write_pass(os.path.join(self.tmp, "arm"), ell=ELL, x0=X0)
        rows = self.reduce([os.path.join(self.tmp, "arm")], d=(25.0, 70.0))
        self.assertEqual(pick(rows, family="plane", weights="unif", basis="pos", set="perplane")["n_planes"], "4")
        with self.assertRaises(red.ReduceError):
            self.reduce([os.path.join(self.tmp, "arm")], d=(60.0, 94.0))  # two planes only
        with self.assertRaises(red.ReduceError):
            self.reduce([os.path.join(self.tmp, "arm")], ell=0.3)  # --ell differs from field.json
        # relative tolerance on the bounds
        idx = red.window_planes([1.0 + 25.0 * 0.1 * (1 - 1e-12), 1.0 + 30.0 * 0.1, 1.0 + 94.0 * 0.1 * (1 + 1e-12)], 1.0, 0.1, 25.0, 94.0)
        self.assertEqual(list(idx), [0, 1, 2])
        with self.assertRaises(red.ReduceError):
            red.window_planes([3.0, 2.0, 4.0, 5.0], 1.0, 0.1, 0.0, 1e9)  # not increasing


class TestTimeFamily(Base):
    def _row(self, rows, **kw):
        return pick(rows, family="time", **kw)

    def test_window_ends_before_first_exit_and_alphas_exact(self):
        synth.write_pass(os.path.join(self.tmp, "arm"), ell=ELL, x0=X0, exit_after={0: 6}, a_L=0.05, a_Tt=0.04)
        rows = self.reduce([os.path.join(self.tmp, "arm")])
        r = self._row(rows, weights="unif")
        self.assertEqual((r["j_end"], float(r["t_end"]), float(r["t_end_next"])), ("5", 3.0, 3.5))
        self.assertAlmostEqual(float(r["t_lo"]), 2.25, delta=1e-15)
        self.assertEqual(r["n_set"], "64")  # nobody lost inside the window
        self.assertAlmostEqual(float(r["alpha_LAM"]), 0.05, delta=1e-12)
        self.assertAlmostEqual(float(r["alpha_Tfixed"]), 0.04, delta=1e-12)
        self.assertAlmostEqual(float(r["alpha_TAM"]), 0.04, delta=1e-12)  # e_L = e1: aligned frame = fixed axes

    def test_aligned_frame_with_tilted_mean_flux(self):
        synth.write_pass(os.path.join(self.tmp, "arm"), ell=ELL, x0=X0, a_L=0.05, a_Tt=0.04, interior=(1.0, 1.0, 0.0))
        r = self._row(self.reduce([os.path.join(self.tmp, "arm")]), weights="unif")
        # u1 = (-1,1,0)/sqrt2 -> (aL + aT)/2; u2 = e3 -> aT
        self.assertAlmostEqual(float(r["alpha_TAM"]), ((0.05 + 0.04) / 2 + 0.04) / 2, delta=1e-12)
        self.assertEqual(r["j_end"], "11")  # no exit: the last snapshot
        self.assertTrue(math.isnan(float(r["t_end_next"])))

    def test_non_exit_status_does_not_end_the_window(self):
        synth.write_pass(os.path.join(self.tmp, "arm"), ell=ELL, x0=X0, exit_after={0: 6}, status_exit={0: 3})
        r = self._row(self.reduce([os.path.join(self.tmp, "arm")]), weights="unif")
        self.assertEqual((r["j_end"], r["n_set"], r["n_traced"]), ("11", "63", "64"))

    def test_exit_before_first_snapshot_is_a_window_error_row(self):
        synth.write_pass(os.path.join(self.tmp, "arm"), ell=ELL, x0=X0, exit_after={0: 0})
        r = self._row(self.reduce([os.path.join(self.tmp, "arm")]), weights="flux")
        self.assertIn("no complete snapshot", r["note"])
        self.assertTrue(math.isnan(float(r["alpha_LAM"])))

    def test_t_lo_before_first_snapshot(self):
        synth.write_pass(os.path.join(self.tmp, "arm"), ell=ELL, x0=X0, times=[1.0, 1.1, 1.2])
        r = self._row(self.reduce([os.path.join(self.tmp, "arm")]), weights="flux")
        self.assertIn("outside the recorded snapshots", r["note"])

    def test_U_time_override(self):
        synth.write_pass(os.path.join(self.tmp, "arm"), ell=ELL, x0=X0, a_L=0.05)
        r = self._row(self.reduce([os.path.join(self.tmp, "arm")], U_time=4.0), weights="unif")
        self.assertAlmostEqual(float(r["alpha_LAM"]), 0.05 * 2.0 / 4.0, delta=1e-12)


class TestSplitHalf(Base):
    def test_halves_against_brute_force_and_share(self):
        synth.write_pass(os.path.join(self.tmp, "arm"), ell=ELL, x0=X0)
        rows = self.reduce([os.path.join(self.tmp, "arm")])
        r = pick(rows, family="plane", weights="unif", basis="pos", set="perplane")
        X = np.load(os.path.join(self.tmp, "arm", "realizations", "r_0000", "planes_x23.npy"))
        x1 = X0 + np.array([25, 40, 55, 70, 90]) * ELL
        p = np.arange(64)
        A = ((p // 8 + p % 8) % 2) == 0
        for name, mask in (("A", A), ("B", ~A)):
            sig = np.array([(np.var(X[mask, k, 0]) + np.var(X[mask, k, 1])) / 2 for k in range(2, 7)])
            expect = 0.5 * np.polyfit(x1, sig, 1)[0] / ELL
            self.assertAlmostEqual(float(r["alpha_TP_" + name]), expect, delta=1e-12)
        self.assertEqual(int(A.sum()), 32)
        # two-half average of the (A, B) estimators is close to the full estimate but not required to be equal
        self.assertTrue(abs(0.5 * (float(r["alpha_TP_A"]) + float(r["alpha_TP_B"])) - float(r["alpha_TP"])) < 0.01)

    def test_splithalf_fraction_definition(self):
        a, A, B = np.array([1.0, 2.0, 4.0, 3.0]), np.array([1.1, 1.8, 4.3, 2.7]), np.array([0.9, 2.2, 3.7, 3.3])
        out = red.splithalf_fraction(A, B, a)
        self.assertAlmostEqual(out["component"], np.mean((A - B) ** 2) / 4.0, delta=1e-15)
        self.assertAlmostEqual(out["variance"], a.var(ddof=1), delta=1e-15)
        self.assertAlmostEqual(out["fraction"], out["component"] / out["variance"], delta=1e-15)
        self.assertEqual(out["flag"], out["fraction"] > 0.25)

    def test_split_share_from_table(self):
        for r in range(3):
            synth.write_pass(os.path.join(self.tmp, "arm"), ell=ELL, x0=X0, r=r, a_T=A_T * (1 + 0.1 * r), seed=700 + r)
        self.reduce([os.path.join(self.tmp, "arm")])
        out = red.split_share(os.path.join(self.tmp, "table.csv"))
        g = [o for o in out if (o["family"], o["weights"], o["basis"], o["set"], o["estimator"]) == ("plane", "unif", "pos", "perplane", "alpha_TP")]
        self.assertEqual(len(g), 1)
        self.assertEqual(g[0]["n"], 3)
        self.assertTrue(math.isfinite(g[0]["fraction"]))

    def test_odd_lattice_halves_unavailable(self):
        with self.assertRaises(red.ReduceError):
            red.half_masks(63, 8)


class TestPar2(Base):
    def _rows(self, x1, a=0.02, v0=0.1, c=0.03, U=1.5, drift=0.0, nan_from=None):
        rows = []
        for k, x in enumerate(x1):
            off = x - X0
            if nan_from is not None and k >= nan_from:
                rows.append([x, 1000, 0] + [float("nan")] * 12)
                continue
            v = v0 + 2 * a * ELL * off
            g2 = 2 * a * ELL * off
            rows.append([x, 1000, 990 - k, off / U, c * off, drift * off, 0.0, v, v, 0.0, drift * off, 0.0, g2, g2, 0.0])
        return rows

    def test_known_slopes_and_nan_rows(self):
        x1 = X0 + np.array([5, 25, 40, 70, 94, 100]) * ELL
        synth.write_par2(os.path.join(self.tmp, "crossings", "realization_0003_planes.csv"), self._rows(x1, nan_from=5))
        rows = self.reduce(par2=[os.path.join(self.tmp, "crossings", "realization_0003_planes.csv")], ell=ELL, x0=X0)
        self.assertEqual(len(rows), 2)
        for b in ("pos", "disp"):
            r = pick(rows, basis=b)
            self.assertAlmostEqual(float(r["alpha_TP"]), 0.02, delta=TOL)
            self.assertAlmostEqual(float(r["alpha_LP"]), 0.5 * 1.5 ** 2 * 0.03 / ELL, delta=1e-11)
            self.assertAlmostEqual(float(r["U_r"]), 1.5, delta=1e-11)
            self.assertEqual((r["code"], r["r"], r["weights"], r["set_differs"], r["n_traced"]), ("par2", "3", "unif", "1", "1000"))
            self.assertEqual(r["n_planes"], "4")
        pl = read_table(os.path.join(self.tmp, "table.planes.csv"))
        self.assertEqual(len([p for p in pl if p["basis"] == "pos"]), 6)
        self.assertTrue(math.isnan(float(pick(pl, basis="pos", plane=5)["var_x2"])))

    def test_tilted_displacement_alpha_TAP(self):
        x1 = X0 + np.array([25, 40, 70, 94]) * ELL
        synth.write_par2(os.path.join(self.tmp, "c", "realization_0000_planes.csv"), self._rows(x1, drift=0.5))
        r = pick(self.reduce(par2=[os.path.join(self.tmp, "c", "realization_0000_planes.csv")], ell=ELL, x0=X0), basis="disp")
        self.assertAlmostEqual(float(r["alpha_TAP"]), 0.02 * (1 + 1 / 1.25) / 2, delta=TOL)

    def test_window_plane_without_crossings_gives_nan_with_note(self):
        x1 = X0 + np.array([25, 40, 70, 94]) * ELL
        synth.write_par2(os.path.join(self.tmp, "c", "realization_0000_planes.csv"), self._rows(x1, nan_from=3))
        r = pick(self.reduce(par2=[os.path.join(self.tmp, "c", "realization_0000_planes.csv")], ell=ELL, x0=X0), basis="pos")
        self.assertTrue(math.isnan(float(r["alpha_TP"])))
        self.assertIn("no particle crossed", r["note"])

    def test_format_errors(self):
        p = os.path.join(self.tmp, "c", "realization_0000_planes.csv")
        os.makedirs(os.path.dirname(p))
        with open(p, "w") as f:
            f.write("# format_version=2 sf42_plane_crossings\n" + synth.PAR2_HEADER + "\n")
        with self.assertRaises(red.ReduceError):
            red.read_par2_crossings(red.Recorder(), p)
        with open(p, "w") as f:
            f.write(synth.PAR2_FORMAT + "\nx1,n_injected\n1,2\n")
        with self.assertRaises(red.ReduceError):
            red.read_par2_crossings(red.Recorder(), p)
        with self.assertRaises(red.ReduceError):
            self.reduce(par2=[p])  # --ell / --x0 missing

    def test_series_time_window(self):
        t = np.arange(0.0, 6.0, 0.5)[1:]
        active = np.array([100] * 8 + [99] * 3)
        ser = os.path.join(self.tmp, "stats", "realization_0002_timeseries.csv")
        os.makedirs(os.path.dirname(ser))
        U = 2.0
        with open(ser, "w") as f:
            f.write("# format_version=1\nt,mean_x,mean_y,mean_z,var_x,var_y,var_z,active\n")
            for ti, a in zip(t, active):
                f.write("%.17g,%.17g,0,0,%.17g,%.17g,%.17g,%d\n" % (ti, U * ti, 2 * 0.05 * ELL * U * ti, 2 * 0.04 * ELL * U * ti, 2 * 0.04 * ELL * U * ti, a))
        rows = self.reduce(series=[ser], ell=ELL, x0=X0, U_time=U)
        r = rows[0]
        self.assertEqual((r["family"], r["code"], r["j_end"], r["n_set"]), ("time", "par2", "7", "100"))
        self.assertAlmostEqual(float(r["alpha_LAM"]), 0.05, delta=1e-12)
        self.assertAlmostEqual(float(r["alpha_Tfixed"]), 0.04, delta=1e-12)
        with self.assertRaises(red.ReduceError):
            self.reduce(series=[ser], ell=ELL, x0=X0)  # --U-time required


class TestSF39Equivalence(Base):
    """The two-plane common-set estimator is SF-39's alpha_TP = (sigma_T^2(X_b) - sigma_T^2(X_a))/(2 ell (X_b - X_a))."""

    def test_matches_sf39_tp_value(self):
        art = os.path.abspath(os.path.join(os.path.dirname(__file__), *([".."] * 4), "2026-10-27-sf39-macrodispersion-study", "scripts"))
        if not os.path.isdir(os.path.join(art, "sf39")):
            self.skipTest("SF-39 artifact not present")
        sys.path.insert(0, art)
        try:
            from sf39 import reduce as sref
        finally:
            sys.path.remove(art)
        rng = np.random.default_rng(5)
        Np, ell = 400, 0.25
        w = rng.uniform(0.2, 2.0, Np)
        w /= w.sum()
        X = rng.normal(size=(Np, 2, 2)) * np.array([0.3, 0.5])[None, :, None]
        X[rng.random(Np) < 0.1, 1, :] = np.nan  # not reaching the second plane
        I = np.where(np.isfinite(X[:, :, 0]), rng.uniform(0, 0.3, (Np, 2)), np.nan)
        real = types.SimpleNamespace(ell=ell, pollock=None, rdir="synthetic",
                                     pass_=types.SimpleNamespace(weights=w, planes_x23=X, planes_I=I))
        xa, xb = 6.0, 6.0 + 20 * ell
        fin = np.isfinite(X).all(axis=2)
        both = fin.all(axis=1)
        tau = np.where(fin, np.arange(2)[None, :] * 1.0 + rng.uniform(0, 0.1, (Np, 2)), np.nan)
        for mode in ("flux", "unif", "expI"):
            ref = sref.tp_value(real, mode, 0, 1, xa, xb)["tp.alpha_TP"]
            Wk = red._weights_matrix(w, mode, np.where(fin, I, 0.0))
            mom, _ = red.consumer_plane_variant(np.array([xa, xb]), fin, Wk, X, tau, np.ones(Np, bool), True)
            mine = red.estimators(np.array([xa, xb]), mom, ell)["alpha_TP"]
            self.assertAlmostEqual(mine, ref, delta=1e-14 * max(1.0, abs(ref)), msg=mode)
        self.assertGreater(int(both.sum()), 300)


class TestManifest(Base):
    def test_manifest_lists_every_input_with_sha256(self):
        synth.write_pass(os.path.join(self.tmp, "arm"), ell=ELL, x0=X0)
        self.reduce([os.path.join(self.tmp, "arm")])
        import json
        with open(os.path.join(self.tmp, "table.csv.manifest.json")) as f:
            man = json.load(f)
        names = {e["path"] for e in man["files"]}
        for n in ("track.json", "field.json", "flow_stats.json", "weights.npy", "planes_x23.npy", "times_x.npy", "seeds.npy"):
            self.assertTrue(any(p.endswith("/" + n) for p in names), n)
        for e in man["files"]:
            p = os.path.join(self.tmp, e["path"])
            self.assertEqual(e["sha256"], hashlib.sha256(slurp(p)).hexdigest())
            self.assertEqual(e["size"], os.path.getsize(p))
            self.assertFalse(os.path.isabs(e["path"]))
        self.assertEqual({t["path"] for t in man["tables"]}, {"table.csv", "table.planes.csv"})
        for t in man["tables"]:
            self.assertEqual(t["sha256"], hashlib.sha256(slurp(os.path.join(self.tmp, t["path"]))).hexdigest())

    def test_output_is_deterministic(self):
        synth.write_pass(os.path.join(self.tmp, "arm"), ell=ELL, x0=X0)
        self.reduce([os.path.join(self.tmp, "arm")])
        a = slurp(os.path.join(self.tmp, "table.csv"))
        self.reduce([os.path.join(self.tmp, "arm")])
        self.assertEqual(a, slurp(os.path.join(self.tmp, "table.csv")))


if __name__ == "__main__":
    unittest.main()
