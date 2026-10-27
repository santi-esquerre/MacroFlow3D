"""Known-answer tests of the SF-39 reducer (C-REDUCE) on analytic fixtures.  Run (seconds):
`cd A/scripts && python3 -m unittest discover -s . -t . -p 'test_reduce*.py'`."""
import contextlib
import io as stdio
import math
import os
import shutil
import tempfile
import unittest

import numpy as np

from sf39 import io as sio
from sf39 import reduce as rd
from sf39.tests import fixtures_runner as fx

TAU, U, ELL = 12.0, 0.97, fx.ELL
RTOL = 1e-11


def close(tc, a, b, rtol=RTOL, atol=1e-13):
    tc.assertTrue(abs(a - b) <= atol + rtol * abs(b), "%.17g != %.17g" % (a, b))


def wv(x, w):  # independent biased weighted variance
    w = np.asarray(w, float)
    return float(np.cov(x, aweights=w, ddof=0))


def wcov(X, w):
    return np.cov(np.asarray(X).T, aweights=np.asarray(w, float), ddof=0)


def j_hi_of(tau=TAU, u=U):
    return int(np.flatnonzero(fx.TIMES >= tau * ELL / u)[0])


def ref_alpha(posfn, wfn, tau, u, e, nj=None):
    """Window alpha of direction e from explicit per-snapshot variances and numpy interpolation."""
    t_hi = tau * ELL / u
    t_lo = 0.75 * t_hi
    jh = j_hi_of(tau, u)
    js = np.arange(jh + 1)
    V = np.array([float(e @ np.atleast_2d(wcov(posfn(j), wfn(j))) @ e) for j in js])
    return (np.interp(t_hi, fx.TIMES[js], V) - np.interp(t_lo, fx.TIMES[js], V)) / (2 * ELL * u * (t_hi - t_lo))


class Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="sf39_fx_")
        cls.info = fx.build_run(cls.tmp + "/single", multi=False)
        cls.R0 = cls.tmp + "/single/realizations/r_0000"
        cls.g = cls.info["gens"][0]
        cls.real = sio.Realization(cls.R0)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


class TestStats(unittest.TestCase):
    def test_wvar_biased_renormalized(self):
        x, w = np.array([1.0, 2.0, 4.0]), np.array([1.0, 1.0, 2.0])
        m = (1 + 2 + 8) / 4
        close(self, rd.wvar(x, w), ((1 - m) ** 2 + (2 - m) ** 2 + 2 * (4 - m) ** 2) / 4)
        close(self, rd.wvar(x, 5.0 * w), rd.wvar(x, w))  # renormalization

    def test_empty_and_zero_weight_raise(self):
        with self.assertRaises(sio.ReduceError):
            rd.wvar(np.array([]), np.array([]))
        with self.assertRaises(sio.ReduceError):
            rd.wvar(np.array([1.0, 2.0]), np.array([0.0, 0.0]))

    def test_window_edges_and_bracket(self):
        t = np.array([1.0, 2.0, 3.0])
        tl, th, j = rd.window_edges(t, 4.0, 1.0, 2.0)  # t_hi = 2 exactly on a snapshot
        self.assertEqual((tl, th, j), (1.5, 2.0, 1))
        with self.assertRaises(sio.WindowError):
            rd.window_edges(t, 8.0, 1.0, 2.0)  # t_hi = 4 > t_N
        with self.assertRaises(sio.WindowError):
            rd.window_edges(t, 1.0, 1.0, 2.0)  # t_lo = 0.375 < t_1
        self.assertEqual(rd._bracket(t, 1.0), (0, 0, 0.0))
        ja, jb, f = rd._bracket(t, 2.5)
        self.assertEqual((ja, jb), (1, 2))
        close(self, f, 0.5)

    def test_variance_interpolated_not_positions(self):
        times = np.array([1.0, 2.0, 4.0])
        c = np.array([-1.0, 0.0, 2.0])
        w = np.array([1.0, 1.0, 1.0])
        pos = lambda j: np.stack([c * times[j], 0 * c, 0 * c], axis=1)  # variance t^2 var(c)
        _, C_lo, _, C_hi = rd.window_moments(times, 1.5, 3.0, pos, lambda j: w)
        v = np.var(c)
        V = lambda t: v * t * t
        lin = lambda t, a, b: V(a) + (t - a) / (b - a) * (V(b) - V(a))
        close(self, C_lo[0, 0], lin(1.5, 1.0, 2.0))
        close(self, C_hi[0, 0], lin(3.0, 2.0, 4.0))
        self.assertGreater(abs(C_hi[0, 0] - V(3.0)) / V(3.0), 0.05)  # differs from the exact t^2 law


class TestWindow(Base):
    def members(self, tau=TAU, u=U):
        return self.g["t_end"] > fx.TIMES[j_hi_of(tau, u)]

    def test_flux_and_uniform_closed_form(self):
        S = self.members()
        for mode, w in (("flux", self.g["w"]), ("unif", np.ones(fx.NP))):
            out = rd.window_method(self.real, mode, TAU, U)
            self.assertEqual(out["win.n"], int(S.sum()))
            for k, name in enumerate(("alpha_e1", "alpha_e2", "alpha_e3")):
                c = (self.g["c1"], self.g["c2"], self.g["c3"])[k]
                close(self, out["win." + name], wv(c[S], w[S]) / (2 * ELL * U))
            close(self, out["win.alpha_Tfixed"], (out["win.alpha_e2"] + out["win.alpha_e3"]) / 2)
        out = rd.window_method(self.real, "flux", TAU, U)
        close(self, out["win.excl"], 1 - self.g["w"][S].sum())

    def test_other_velocity_shifts_set(self):  # U = 1.3 -> t_hi = 4.62, j_hi = 18: a different set S
        u = 1.3
        S = self.members(u=u)
        self.assertNotEqual(int(S.sum()), int(self.members().sum()))
        out = rd.window_method(self.real, "flux", TAU, u)
        close(self, out["win.alpha_e1"], wv(self.g["c1"][S], self.g["w"][S]) / (2 * ELL * u))

    def test_errors_no_snapshot_and_t1(self):
        with self.assertRaises(sio.WindowError):
            rd.window_method(self.real, "flux", 40.0, U)  # t_hi = 20.6 > 16
        with self.assertRaises(sio.WindowError):
            rd.window_method(self.real, "flux", 0.5, U)  # t_lo < t_1

    def test_prefix_violation_raises(self):
        d = tempfile.mkdtemp(prefix="sf39_bad_")
        try:
            fx.build_run(d, nr=1)
            p = d + "/realizations/r_0000/times_x.npy"
            a = np.load(p)
            a[0, 18, :] = np.nan  # hole at a used snapshot (bracket of t_lo) of a particle finite at j_hi
            np.save(p, a)
            with self.assertRaises(sio.ReduceError):
                rd.window_method(sio.Realization(d + "/realizations/r_0000"), "flux", TAU, U)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_expI_and_full_against_direct_reference(self):
        g = self.g
        S = self.members()
        idx = np.flatnonzero(S)
        tx, tI = fx.times_x(g), fx.times_I(g)
        e = np.eye(3)
        out = rd.window_method(self.real, "expI", TAU, U)
        for k in range(3):
            ref = ref_alpha(lambda j: tx[idx, j, :], lambda j: g["w"][idx] * np.exp(-tI[idx, j]), TAU, U, e[k])
            close(self, out["win.alpha_e%d" % (k + 1)], ref)
        # full population: exits frozen; seeds as fallback for the exit before t_1
        st = g["status_time"]
        placed = np.flatnonzero((g["w"] > 0) & ((st == 0) | (st == 7) | (st == 8) | (st == 2)))  # status 3 (id 55) unplaced
        placed = placed[(placed != 55)]
        seeds = np.load(self.R0 + "/seeds.npy")

        def pos(j):
            X = tx[placed, j, :].copy()
            for n, i in enumerate(placed):
                if np.isnan(X[n, 0]):
                    ok = np.flatnonzero(np.isfinite(tx[i, :, 0]))
                    x23 = tx[i, ok[-1], 1:3] if len(ok) else seeds[i, 1:3]
                    X[n] = (fx.L1 if st[i] == 8 else 0.0, x23[0], x23[1])
            return X

        out = rd.window_method(self.real, "full", TAU, U)
        for k in range(3):
            close(self, out["win.alpha_e%d" % (k + 1)], ref_alpha(pos, lambda j: g["w"][placed], TAU, U, e[k]))
        self.assertEqual((out["win.n_exit_out"], out["win.n_exit_in"], out["win.n_seed_fallback"], out["win.n_unplaced"]),
                         (3, 4, 1, 1))
        self.assertEqual(out["win.n"], len(placed))

    def test_pollock_and_intersection_sets(self):
        g, jh = self.g, j_hi_of()
        Sc = self.members()
        Sp = g["p_end"] > fx.TIMES[jh]
        out = rd.window_method(self.real, "pollock", TAU, U)
        self.assertEqual(out["win.n"], int(Sp.sum()))
        close(self, out["win.alpha_e2"], wv(g["pc2"][Sp], g["pw"][Sp]) / (2 * ELL * U))
        close(self, out["win.alpha_e1"], wv(g["c1"][Sp], g["pw"][Sp]) / (2 * ELL * U))
        I = Sc & Sp
        self.assertLess(int(I.sum()), min(int(Sc.sum()), int(Sp.sum())))
        w = np.ones(fx.NP)
        oc = rd.window_method(self.real, "ixc", TAU, U)
        op = rd.window_method(self.real, "ixp", TAU, U)
        self.assertEqual((oc["win.n"], op["win.n"]), (int(I.sum()),) * 2)
        close(self, oc["win.alpha_e2"], wv(g["c2"][I], w[I]) / (2 * ELL * U))
        close(self, op["win.alpha_e2"], wv(g["pc2"][I], w[I]) / (2 * ELL * U))
        close(self, oc["win.excl"], 1 - I.sum() / (g["w"] > 0).sum())

    def test_aligned_and_drift_frames_with_tilt(self):
        th = 0.4
        d = tempfile.mkdtemp(prefix="sf39_tilt_")
        try:
            run = fx.build_run(d, nr=1, theta=th)
            real = sio.Realization(d + "/realizations/r_0000")
            g = run["gens"][0]
            S = g["t_end"] > fx.TIMES[j_hi_of()]
            w = g["w"][S]
            Cc = wcov(np.stack([g["c1"][S], g["c2"][S], g["c3"][S]], axis=1), w)
            Q = np.array([[math.cos(th), -math.sin(th), 0], [math.sin(th), math.cos(th), 0], [0, 0, 1.0]])
            K = Q @ Cc @ Q.T  # V(t) = t K
            den = 2 * ELL * U
            out = rd.window_method(real, "flux", TAU, U)
            eL, eT1, eT2 = Q[:, 0], Q[:, 1], Q[:, 2]
            close(self, out["win.alpha_eL"], eL @ K @ eL / den)
            close(self, out["win.alpha_eT1"], eT1 @ K @ eT1 / den)
            close(self, out["win.alpha_eT2"], eT2 @ K @ eT2 / den)
            close(self, out["win.alpha_TAM"], (eT1 @ K @ eT1 + eT2 @ K @ eT2) / (2 * den))
            e2 = np.array([0, 1.0, 0])
            close(self, out["win.alpha_e2"], e2 @ K @ e2 / den)
            close(self, out["win.dTfixed_minus_TAM"], out["win.alpha_Tfixed"] - out["win.alpha_TAM"])
            # plume-drift frame from the interpolated means
            tx = fx.times_x(g)
            idx = np.flatnonzero(S)
            t_hi = TAU * ELL / U
            js = np.arange(j_hi_of() + 1)
            mean = lambda t: np.array([np.interp(t, fx.TIMES[js], [np.average(tx[idx, j, k], weights=w) for j in js]) for k in range(3)])
            eD = mean(t_hi) - mean(0.75 * t_hi)
            eD /= np.linalg.norm(eD)
            close(self, out["win.alpha_eD"], eD @ K @ eD / den)
            close(self, out["win.alpha_TD"], (np.trace(K) - eD @ K @ eD) / (2 * den))
            c, _ = rd.reduce_methods(real, TAU, U, r=0)
            close(self, c["tilt"], th)
            close(self, rd.tilt_angle(eD), math.atan2(np.linalg.norm(np.cross(eD, [1, 0, 0])), eD[0]))
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_gram_schmidt_degenerate_raises(self):
        with self.assertRaises(sio.ReduceError):
            rd.frame_from_longitudinal(np.array([0.0, 1.0, 0.0]))


class TestPlane(Base):
    def sigc2(self, T, c2, c3, w):
        return (wv(c2[T], w[T]) + wv(c3[T], w[T])) / 2

    def test_breakthrough_filters_status(self):
        P = self.real.pass_
        tau = np.load(self.R0 + "/planes_tau.npy")[:, fx.IB]
        ok = (self.g["status_plane"] == 0) & np.isfinite(tau)
        self.assertLess(np.nanmin(tau), np.min(tau[ok]))  # id 25 (status 3) is faster and must be ignored
        close(self, rd.breakthrough_time(P, fx.IB), np.min(tau[ok]))

    def test_alpha_tp_closed_form_all_methods(self):
        g = self.g
        kp = self.real.key_planes()
        T = g["reach"] == 8
        Tp = fx._pollock_reach(g) == 8
        close(self, rd.tp_method(self.real, "flux", kp)["tp.alpha_TP"], self.sigc2(T, g["c2"], g["c3"], g["w"]) / (2 * ELL))
        close(self, rd.tp_method(self.real, "unif", kp)["tp.alpha_TP"], self.sigc2(T, g["c2"], g["c3"], np.ones(fx.NP)) / (2 * ELL))
        po = rd.tp_method(self.real, "pollock", kp)
        close(self, po["tp.alpha_TP"], self.sigc2(Tp, g["pc2"], g["pc3"], g["pw"]) / (2 * ELL))
        I = T & Tp
        self.assertLess(int(I.sum()), min(int(T.sum()), int(Tp.sum())))
        one = np.ones(fx.NP)
        close(self, rd.tp_method(self.real, "ixc", kp)["tp.alpha_TP"], self.sigc2(I, g["c2"], g["c3"], one) / (2 * ELL))
        close(self, rd.tp_method(self.real, "ixp", kp)["tp.alpha_TP"], self.sigc2(I, g["pc2"], g["pc3"], one) / (2 * ELL))
        # exp(-I) at each plane: direct
        Ipl = np.load(self.R0 + "/planes_I.npy")
        sig = lambda ip, X: self.sigc2(T, g["c2"], g["c3"], g["w"] * np.exp(-Ipl[:, ip])) * (X - fx.X0)
        ref = (sig(fx.IB, fx.XB) - sig(fx.IA, fx.XA)) / (2 * ELL * (fx.XB - fx.XA))
        close(self, rd.tp_method(self.real, "expI", kp)["tp.alpha_TP"], ref)

    def test_excluded_counts_and_delta_half(self):
        g = self.g
        kp = self.real.key_planes()
        out = rd.tp_method(self.real, "flux", kp)
        T = g["reach"] == 8
        self.assertEqual(out["tp.n"], int(T.sum()))
        close(self, out["tp.excl"], 1 - g["w"][T].sum())
        traced = g["w"] > 0
        self.assertEqual(out["tp.n_noncross_Xa"], int((traced & (g["reach"] <= fx.IA)).sum()))
        self.assertEqual(out["tp.n_noncross_Xb"], int((traced & (g["reach"] <= fx.IB)).sum()))
        Tq = g["reach"] >= fx.IA + 1  # records at Xq (index 2) and Xa (index 4)
        first = self.sigc2(Tq, g["c2"], g["c3"], g["w"]) / (2 * ELL)
        close(self, out["tp.alpha_TP_q"], first)
        close(self, out["tp.delta_half"], first - out["tp.alpha_TP"])

    def test_sigma_series_own_contributors(self):
        g = self.g
        rows = rd.sigma_series_rows(self.real.pass_)
        self.assertEqual(len(rows), len(fx.PLANES))
        for p, row in enumerate(rows):
            T = g["reach"] > p
            self.assertEqual(row["n_contrib"], int(T.sum()))
            self.assertEqual(row["n_noncross"], int(((g["w"] > 0) & ~T).sum()))
            close(self, row["sigma_T2"], (fx.PLANES[p] - fx.X0) * self.sigc2(T, g["c2"], g["c3"], g["w"]))

    def test_ev_percentiles(self):
        P = self.real.pass_
        Ipl = np.load(self.R0 + "/planes_I.npy")[:, fx.IB]
        e = np.abs(1 - np.exp(-Ipl[np.isfinite(Ipl)]))
        got = rd.ev_percentiles(P, fx.IB)
        for k, q in zip(rd.EV_NAMES, rd.EV_PCT):
            close(self, got[k], float(np.percentile(e, q)))


class TestSplitAndBookkeeping(Base):
    def test_split_half_parity(self):
        g = self.g
        p = np.arange(fx.NP)
        A = ((p // fx.M + p % fx.M) % 2) == 0
        a, b = rd.half_masks(self.real.pass_)
        self.assertTrue(np.array_equal(a, A) and np.array_equal(b, ~A))
        S = g["t_end"] > fx.TIMES[j_hi_of()]
        c, _ = rd.reduce_methods(self.real, TAU, U, r=0)
        for name, m in (("A", A), ("B", ~A)):
            s = S & m
            close(self, c["split.alpha_LAM.%s" % name], wv(g["c1"][s], g["w"][s]) / (2 * ELL * U))
            T = (g["reach"] == 8) & m
            close(self, c["split.alpha_TP.%s" % name], (wv(g["c2"][T], g["w"][T]) + wv(g["c3"][T], g["w"][T])) / 2 / (2 * ELL))
        self.assertNotEqual(c["split.alpha_LAM.A"], c["split.alpha_LAM.B"])

    def test_common_columns(self):
        c, methods = rd.reduce_methods(self.real, TAU, U, r=0)
        self.assertEqual((c["seed"], c["j_hi"]), (5001, j_hi_of()))
        self.assertEqual((c["exit_inlet_jhi"], c["exit_outlet_jhi"]), (4, 3))
        self.assertEqual(sum(c["plane_hist_%d" % k] for k in range(9)), fx.NP)
        self.assertEqual((c["plane_hist_1"], c["time_hist_1"], c["time_hist_8"]), (3, 3, 4))
        close(self, c["qbar"], 0.97)
        close(self, c["applied_scale"], 0.9)
        self.assertEqual(c["skipped"], "")
        self.assertEqual(set(methods), set(rd.WIN_MODES))
        close(self, c["S_r"], methods["flux"]["tp.sigT2_Xb"] / (ELL * (fx.XB - fx.XA)))
        flat = rd.flatten(c, methods)
        self.assertIn("flux.win.alpha_TAM", flat)
        self.assertIn("pollock.tp.alpha_TP", flat)
        self.assertNotIn("full.tp.alpha_TP", flat)  # window-only method
        for r in rd.long_rows(c, methods):
            self.assertIn(r["method"], rd.WIN_MODES)


class TestMissing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sf39_miss_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_find_plane_tolerance(self):
        pl = np.array([1.0, 2.0, 3.0 + 1e-13])
        self.assertEqual(sio.find_plane(pl, 3.0), 2)
        self.assertIsNone(sio.find_plane(pl, 3.0 + 1e-10))
        with self.assertRaises(sio.MissingData):
            sio.plane_index(pl, 4.0, "Xa")
        self.assertIsNone(sio.plane_index(pl, 4.0, "Xa", tolerate_missing=True))

    def test_missing_key_planes(self):
        fx.build_run(self.tmp, nr=1, key_planes=False)
        real = sio.Realization(self.tmp + "/realizations/r_0000")
        with self.assertRaises(sio.MissingData):
            rd.reduce_methods(real, TAU, U)
        c, m = rd.reduce_methods(real, TAU, U, tolerate_missing=True, r=0)
        self.assertEqual(c["skipped"], "plane_Xa;plane_Xq")
        self.assertNotIn("tp.alpha_TP", m["flux"])
        self.assertIn("win.alpha_e1", m["flux"])
        self.assertTrue(math.isnan(c["S_r"]))
        self.assertFalse(math.isnan(c["tb"]))

    def test_missing_pollock(self):
        fx.build_run(self.tmp, nr=1, pollock=False)
        real = sio.Realization(self.tmp + "/realizations/r_0000")
        with self.assertRaises(sio.MissingData):
            rd.reduce_methods(real, TAU, U)
        c, m = rd.reduce_methods(real, TAU, U, tolerate_missing=True, r=0)
        self.assertEqual(c["skipped"], "pollock")
        self.assertEqual(set(m), {"flux", "unif", "expI", "full"})

    def test_pass_root_selection(self):
        fx.build_run(self.tmp, nr=1, multi=True)
        R = self.tmp + "/realizations/r_0000"
        with self.assertRaises(sio.MissingData):
            sio.pass_root(R, None)  # three passes: ambiguous
        self.assertTrue(sio.pass_root(R, 16).endswith("hmax_16"))
        with self.assertRaises(sio.MissingData):
            sio.pass_root(R, 64)
        self.assertEqual(sio.parse_range("0-2,5,7-8"), {0, 1, 2, 5, 7, 8})


class TestRho(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="sf39_rho_")
        cls.info = fx.build_run(cls.tmp, multi=True, nr=1)
        cls.R0 = cls.tmp + "/realizations/r_0000"

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_rho_known_displacements(self):
        g = self.info["gens"][0]
        k = (np.arange(fx.NP) % 3).astype(float)
        rows = rd.rho_rows(self.R0)
        self.assertEqual(len(rows), 3 * len(fx.PLANES))
        S_r = None
        for (da, db) in rd.RHO_PAIRS:
            d2 = fx.DISP[db][0] - fx.DISP[da][0]
            d3 = (fx.DISP[db][1] - fx.DISP[da][1]) * k
            for row in [r for r in rows if r["pair"] == "%d-%d" % (da, db)]:
                p = row["plane"]
                T = g["reach"] > p
                w = g["w"][T]
                sT2 = (fx.PLANES[p] - fx.X0) * (wv(g["c2"][T], w) + wv(g["c3"][T], w)) / 2
                self.assertEqual(row["n"], int(T.sum()))
                close(self, row["sigma_T"], math.sqrt(sT2))
                msq = np.average(d2 ** 2 + d3[T] ** 2, weights=w)
                close(self, row["rho"], math.sqrt(msq) / math.sqrt(sT2))
                dbar3 = np.average(d3[T], weights=w)
                cen = (np.average((d3[T] - dbar3) ** 2, weights=w)) / 2  # d2 constant -> its centred part is 0
                close(self, row["rho_c"], math.sqrt(cen) / math.sqrt(sT2))
                close(self, row["rho_S"], row["rho"] * row["S_r"])
                S_r = row["S_r"]
        T = g["reach"] == 8
        sXb = (fx.XB - fx.X0) * (wv(g["c2"][T], g["w"][T]) + wv(g["c3"][T], g["w"][T])) / 2
        close(self, S_r, sXb / (ELL * (fx.XB - fx.XA)))

    def test_no_displacement_gives_zero_rho(self):
        # passes that coincide
        d = tempfile.mkdtemp(prefix="sf39_rho0_")
        try:
            saved = dict(fx.DISP)
            fx.DISP.update({16: (0.0, 0.0), 32: (0.0, 0.0)})
            try:
                fx.build_run(d, multi=True, nr=1)
            finally:
                fx.DISP.update(saved)
            for r in rd.rho_rows(d + "/realizations/r_0000"):
                if r["n"]:
                    self.assertEqual((r["rho"], r["rho_c"]), (0.0, 0.0))
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestCrosscheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sf39_xc_")
        self.out = self.tmp + "/run"
        fx.build_run(self.out, nr=2)
        self.R = [self.out + "/realizations/r_%04d" % r for r in range(2)]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def edit(self, path, fn):
        with open(path) as f:
            lines = f.read().split("\n")
        lines = fn(lines)
        with open(path, "w") as f:
            f.write("\n".join(lines))

    def test_matching_moments_pass(self):
        agg = rd.crosscheck_run(self.R)
        self.assertTrue(rd.crosscheck_ok(agg), agg)
        self.assertEqual(agg["planes"][0], 0)
        self.assertLess(agg["planes"][1], 1e-12)
        self.assertLess(agg["times"][1], 1e-12)
        self.assertLess(agg["twin"][1], 1e-12)
        line = rd.crosscheck_line("run", agg)
        self.assertTrue(line.startswith("CROSSCHECK run=run planes=ok times=ok twin=ok max_rel="), line)

    def test_multipass_and_twin_absent(self):
        d = self.tmp + "/multi"
        fx.build_run(d, multi=True, nr=1, twin=False)
        agg = rd.crosscheck_run([d + "/realizations/r_0000"])
        self.assertTrue(rd.crosscheck_ok(agg))
        self.assertIsNone(agg["twin"])
        self.assertIn("twin=absent", rd.crosscheck_line("m", agg))

    def test_perturbed_planes_times_twin_fail(self):
        def bump_col(col, rel, skip=0):
            def fn(lines):
                head = [i for i, l in enumerate(lines) if l.startswith("x1,") or l.startswith("t,")][0]
                cols = lines[head].split(",")
                row = lines[head + 1 + 3].split(",")
                row[cols.index(col)] = "%.15e" % (float(row[cols.index(col)]) * (1 + rel))
                lines[head + 1 + 3] = ",".join(row)
                return lines
            return fn
        cases = (("moments_planes.csv", "var_x3", "planes"), ("moments_time.csv", "mean_y", "times"),
                 ("twin/moments_planes.csv", "mean_x2", "twin"), ("twin/moments_time.csv", "var_z", "twin"),
                 ("moments_planes.csv", "contributing", "planes"))
        for name, col, key in cases:
            shutil.rmtree(self.out)
            fx.build_run(self.out, nr=2)
            rel = 1e-6 if col != "contributing" else 1.0 / 90  # 91 -> 92 style change of the count
            self.edit(self.R[1] + "/" + name, bump_col(col, rel))
            agg = rd.crosscheck_run(self.R)
            self.assertFalse(rd.crosscheck_ok(agg), name + col)
            self.assertGreaterEqual(agg[key][0], 1)
            self.assertIn("FAIL", rd.crosscheck_line("run", agg))
        # a perturbation below the tolerance is accepted
        shutil.rmtree(self.out)
        fx.build_run(self.out, nr=2)
        self.edit(self.R[0] + "/moments_planes.csv", bump_col("var_x2", 1e-11))
        self.assertTrue(rd.crosscheck_ok(rd.crosscheck_run(self.R)))

    def test_missing_csv_and_plane_x1_mismatch_fail(self):
        os.remove(self.R[0] + "/moments_time.csv")
        agg = rd.crosscheck_run(self.R)
        self.assertFalse(rd.crosscheck_ok(agg))
        fx.build_run(self.out, nr=2)
        self.edit(self.R[0] + "/moments_planes.csv", lambda L: [l.replace("3.000000000000000e+00", "3.100000000000000e+00") for l in L])
        self.assertFalse(rd.crosscheck_ok(rd.crosscheck_run(self.R)))

    def test_smoke_reports_and_fails_on_perturbation(self):
        camp = self.tmp + "/camp"
        fx.build_run(camp + "/prodgrid/E1", nr=1)
        code, out, _ = run_cli("smoke", camp)
        self.assertEqual(code, 0, out)
        self.assertIn("CROSSCHECK run=prodgrid/E1 planes=ok times=ok twin=ok", out)
        p = camp + "/prodgrid/E1/realizations/r_0000/moments_time.csv"
        self.edit(p, lambda L: [l.replace(l.split(",")[4], "%.15e" % (float(l.split(",")[4]) * 1.001), 1) if l[:1].isdigit() and l.startswith("5.0") else l for l in L])
        code, out, _ = run_cli("smoke", camp)
        self.assertEqual(code, 1, out)
        self.assertIn("times=FAIL", out)


def run_cli(*argv):
    out, err = stdio.StringIO(), stdio.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = rd.main(list(argv))
    return code, out.getvalue(), err.getvalue()


def read_csv(path):
    import csv
    with open(path) as f:
        return list(csv.DictReader(f))


class TestCLI(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="sf39_cli_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_tb_and_alpha(self):
        out = self.tmp + "/run"
        run = fx.build_run(out, multi=True, nr=3)
        dest = self.tmp + "/tb.csv"
        self.assertEqual(run_cli("tb", out, "--pass-div", "8", "--dest", dest)[0], 0)
        rows = read_csv(dest)
        self.assertEqual([r["r"] for r in rows], ["0", "1", "2"])
        for r, g in zip(rows, run["gens"]):
            tau = np.load(out + "/realizations/r_%04d/hmax_8/planes_tau.npy" % g["r"])[:, fx.IB]
            ok = (g["status_plane"] == 0) & np.isfinite(tau)
            close(self, float(r["t_b"]), float(np.min(tau[ok])), rtol=1e-15)  # %.17g round trip
            close(self, float(r["qbar"]), g["qbar"], rtol=1e-15)
        dest = self.tmp + "/alpha.csv"
        code, _, _ = run_cli("alpha", out, "--tau-b-star", "12", "--ell-run", "0.5", "--U", "0.97", "--dest", dest,
                             "--realizations", "0-1")
        self.assertEqual(code, 0)
        rows = read_csv(dest)
        self.assertEqual(len(rows), 2 * len(rd.WIN_MODES))
        flux0 = [r for r in rows if r["r"] == "0" and r["method"] == "flux"][0]
        g = run["gens"][0]
        S = g["t_end"] > fx.TIMES[j_hi_of()]
        close(self, float(flux0["win.alpha_e1"]), wv(g["c1"][S], g["w"][S]) / (2 * ELL * U), rtol=1e-14)
        self.assertEqual(flux0["seed"], "5001")
        with self.assertRaises(sio.ReduceError):  # --ell-run must match field.json
            run_cli("alpha", out, "--tau-b-star", "12", "--ell-run", "0.25", "--U", "0.97", "--dest", dest)

    def test_alpha_not_done_reported(self):
        out = self.tmp + "/run"
        fx.build_run(out, nr=2)
        os.remove(out + "/realizations/r_0001/DONE")
        code, _, err = run_cli("tb", out, "--dest", self.tmp + "/tb.csv")
        self.assertEqual(code, 0)
        self.assertIn("NOT_DONE", err)
        self.assertEqual(len(read_csv(self.tmp + "/tb.csv")), 1)

    def test_signtable(self):
        ev = self.tmp + "/evladder"
        fx.build_sign_campaign(ev, rd.SIGN_EXPECTED)
        code, out, _ = run_cli("signtable", ev)
        self.assertEqual(code, 0, out)
        self.assertIn("SIGNTABLE EQUAL", out)
        self.assertEqual(out.count("OK"), 18)
        bad = dict(rd.SIGN_EXPECTED)
        bad[(2, 6101)] = 4
        shutil.rmtree(ev)
        fx.build_sign_campaign(ev, bad)
        code, out, _ = run_cli("signtable", ev)
        self.assertEqual(code, 1)
        self.assertIn("MISMATCH", out)

    def test_smoke(self):
        camp = self.tmp + "/camp"
        fx.build_run(camp + "/prodgrid/E1", key_planes=False, pollock=False, twin_series=False, nr=2)
        fx.build_run(camp + "/prodgrid/E2", key_planes=True, pollock=True, nr=1)
        fx.build_sign_campaign(camp + "/evladder", {(1, 6101): 1})
        code, out, _ = run_cli("smoke", camp)
        self.assertEqual(code, 0, out)
        self.assertIn("NOT EVIDENCE", out)
        self.assertIn("reduced 2/2", out)  # E1 with tolerated missing planes and no Pollock
        self.assertIn("ABSENT", out)  # E3, E4
        # an empty campaign ran nothing
        self.assertEqual(run_cli("smoke", self.tmp + "/nothing")[0], 1)

    def test_smoke_defect_fails(self):
        for what in ("field.json", "weights.npy"):
            camp = self.tmp + "/camp_" + what
            fx.build_run(camp + "/prodgrid/E1", nr=1)
            R = camp + "/prodgrid/E1/realizations/r_0000"
            if what == "field.json":
                os.remove(R + "/field.json")  # a required file is not a tolerated gap
            else:
                np.save(R + "/weights.npy", np.zeros(5))  # shape mismatch -> a real failure
            code, out, _ = run_cli("smoke", camp)
            self.assertEqual(code, 1, out)
            self.assertIn("FAILED", out)


if __name__ == "__main__":
    unittest.main()
