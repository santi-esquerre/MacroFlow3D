"""SF-42 N3: the pseudo-spectral Darcy reference on exact controls."""
import unittest

import numpy as np
from scipy.special import i0

from spectral import darcy, sf18


def grid(M):
    x = np.arange(M) / M
    return np.meshgrid(x, x, x, indexing="ij")  # (Z, Y, X): arrays are (z, y, x)


class ExactControls(unittest.TestCase):
    def test_constant_k(self):
        Z, Y, X = grid(8)
        R = darcy.Reference(2.5 + 0 * X, tol=1e-13, workers=2, band=1.0)
        np.testing.assert_allclose(R.G, [1 / 2.5, 0, 0], atol=1e-14)
        np.testing.assert_allclose(R.q[0], 1.0, atol=1e-14)
        np.testing.assert_allclose(R.q[1], 0.0, atol=1e-14)
        self.assertEqual([c["iterations"] for c in R.correctors], [0, 0, 0])

    def test_layers_parallel_to_the_flow_give_the_arithmetic_mean(self):
        # K = 1 + 0.5 cos(2 pi z): q_x = K / <K> exactly, G = (1/<K>, 0, 0), K_zz = harmonic mean = sqrt(0.75)
        Z, Y, X = grid(32)
        R = darcy.Reference(1 + 0.5 * np.cos(2 * np.pi * Z), tol=1e-13, workers=2, band=1.0)
        np.testing.assert_allclose(R.G, [1.0, 0, 0], atol=1e-13)
        np.testing.assert_allclose(R.K_eff[2, 2], np.sqrt(0.75), rtol=1e-12)
        np.testing.assert_allclose(R.K_eff[0, 0], 1.0, rtol=1e-13)
        for N in (4, 8, 12):  # N does not divide M = 32 for N = 12
            U, V, W = darcy.face_fluxes(R.q, N)
            self.assertEqual((U.shape, V.shape, W.shape), ((N, N, N + 1), (N, N + 1, N), (N + 1, N, N)))
            zc = (np.arange(N) + 0.5) / N
            exact = 1 + 0.5 * np.cos(2 * np.pi * zc) * np.sinc(1.0 / N)  # cell average of K / <K>
            np.testing.assert_allclose(U[:, :, :N], exact[:, None, None] * np.ones((N, N, N)), atol=1e-13)
            np.testing.assert_array_equal(U[:, :, N], U[:, :, 0])  # periodic duplicate plane
            np.testing.assert_allclose(V, 0.0, atol=1e-13)
            np.testing.assert_allclose(W, 0.0, atol=1e-13)

    def test_layers_across_the_flow_give_the_harmonic_mean(self):
        # K = exp(a cos(2 pi x)): q = (1, 0, 0) uniform, G_x = <1/K> = I0(a)
        a = 0.8
        Z, Y, X = grid(32)
        for band, tol in ((1.0, 1e-12), (0.5, 1e-7)):  # the study band keeps modes |m| <= 8 of 32: truncation ~ I_8(a)
            R = darcy.Reference(np.exp(a * np.cos(2 * np.pi * X)), tol=1e-13, workers=2, band=band)
            self.assertLess(abs(R.G[0] / i0(a) - 1.0), tol)
            np.testing.assert_allclose(R.q[0], 1.0, atol=tol)
            np.testing.assert_allclose(R.q[1], 0.0, atol=tol)
            np.testing.assert_allclose(R.q[2], 0.0, atol=tol)
            # sign convention of the FV code: v = K (G + grad h~) points along +qbar with G_x > 0
            self.assertGreater(R.G[0], 0.0)

    def test_random_field_invariants(self):
        f = sf18.generate(16, seed=420001)
        Y = sf18.field_on_grid(f, 24, offset=0.0)
        R = darcy.Reference(np.exp(1.0 * Y), tol=1e-12, workers=2, band=1.0)
        np.testing.assert_allclose(R.mean_flux, [1.0, 0.0, 0.0], atol=1e-10)
        self.assertLess(R.symmetry_defect_rel, 1e-9)
        self.assertGreater(np.min(np.linalg.eigvalsh(0.5 * (R.K_eff + R.K_eff.T))), 0.0)
        # conservation (Galerkin): the retained-band part of the spectral divergence of the flux is at the PCG tolerance
        op = R.op
        div = op.irfft(op._band_limit(sum(1j * k * op.rfft(q) for k, q in zip(op.kd, R.q))))
        self.assertLess(np.linalg.norm(div) / np.linalg.norm(op.K), 1e-9)

    def test_preconditioned_iterations_stay_moderate_with_the_study_band(self):
        # K^{-1/2} (-Delta)^{-1} K^{-1/2} with the retained band 1/2: iterations to 1e-10 grow with the contrast
        # (near-kernel modes of isolated high-K clusters) but stay in the hundreds for eps = 2 (K spans e^{+-8})
        f = sf18.generate(16, seed=420002)
        Y = sf18.field_on_grid(f, 32, offset=0.0)
        its = {}
        for eps in (0.5, 2.0):
            op = darcy.SpectralDarcy(np.exp(eps * Y), workers=2)
            u, info = op.solve(op.rhs(0), tol=1e-10, maxit=1500)
            its[eps] = info["iterations"]
            self.assertLess(info["true_relative_residual"], 1e-10)
        self.assertLess(its[0.5], 80)
        self.assertLess(its[2.0], 1000)


class FailurePath(unittest.TestCase):
    def setUp(self):
        f = sf18.generate(16, seed=420001)
        self.K = np.exp(1.5 * sf18.field_on_grid(f, 16, offset=0.0))

    def test_raises_when_the_iteration_cap_is_reached(self):
        op = darcy.SpectralDarcy(self.K, workers=2)
        with self.assertRaises(darcy.ConvergenceError) as cm:
            op.solve(op.rhs(0), tol=1e-12, maxit=3)
        self.assertIn("not converged", str(cm.exception))

    def test_raises_when_the_tolerance_is_unattainable(self):
        op = darcy.SpectralDarcy(self.K, workers=2)
        with self.assertRaises(darcy.ConvergenceError):
            op.solve(op.rhs(1), tol=1e-30, maxit=60)

    def test_rejects_nonpositive_k(self):
        with self.assertRaises(ValueError):
            darcy.SpectralDarcy(self.K - 2 * self.K.max())


class FaceAverages(unittest.TestCase):
    def test_exact_for_trigonometric_polynomials(self):
        rng = np.random.default_rng(0)
        M, N = 16, 6  # N does not divide M
        m = np.arange(0, 7)
        amp, phi = rng.standard_normal(m.size), rng.uniform(0, 2 * np.pi, m.size)
        x = np.arange(M) / M
        f = sum(a * np.cos(2 * np.pi * k * x + p) for a, k, p in zip(amp, m, phi))
        T_pt, T_avg = darcy.face_operators(N, M)
        xi = np.arange(N) / N
        np.testing.assert_allclose(T_pt @ f, sum(a * np.cos(2 * np.pi * k * xi + p) for a, k, p in zip(amp, m, phi)), atol=1e-13)
        x0 = np.arange(N) / N
        avg = amp[0] * np.cos(phi[0]) * np.ones(N)
        for a, k, p in zip(amp[1:], m[1:], phi[1:]):  # independent closed form of the cell integral
            avg = avg + a * (np.sin(2 * np.pi * k * (x0 + 1.0 / N) + p) - np.sin(2 * np.pi * k * x0 + p)) / (2 * np.pi * k) * N
        np.testing.assert_allclose(T_avg @ f, avg, atol=1e-13)

    def test_three_dimensional_separable_flux(self):
        M, N = 16, 4
        Z, Y, X = grid(M)
        qx = np.cos(2 * np.pi * X + 0.3) * (1 + 0.5 * np.sin(2 * np.pi * Y)) * (2 + np.cos(4 * np.pi * Z))
        zero = 0 * qx
        U, V, W = darcy.face_fluxes((qx, zero, zero), N)
        xi = np.arange(N + 1) / N  # face planes (the last duplicates the first)
        yc = (np.arange(N) + 0.5) / N
        ya = 1 + 0.5 * np.sin(2 * np.pi * yc) * np.sinc(1.0 / N)
        za = 2 + np.cos(4 * np.pi * yc) * np.sinc(2.0 / N)
        exact = np.cos(2 * np.pi * xi + 0.3)[None, None, :] * ya[None, :, None] * za[:, None, None]
        np.testing.assert_allclose(U, exact, atol=1e-13)
        np.testing.assert_allclose(V, 0.0, atol=1e-13)

    def test_error_measures_and_orders(self):
        rng = np.random.default_rng(1)
        ref = tuple(rng.standard_normal(s) for s in ((4, 4, 5), (4, 5, 4), (5, 4, 4)))
        ref = (ref[0], ref[1], ref[2])
        fv = tuple(a.copy() for a in ref)
        fv[0][:, :, :-1] += 0.01  # perturb the unique x faces only
        e = darcy.flux_errors(fv, ref)
        self.assertGreater(e["rel_l2_x"], 0.0)
        self.assertLess(e["rel_l2_all"], e["rel_l2_x"])
        self.assertEqual(darcy.flux_errors(ref, ref)["rel_l2_all"], 0.0)
        ns = [16, 32, 40, 64, 128]
        errs = [3.0 / n ** 2 for n in ns]
        np.testing.assert_allclose(darcy.observed_orders(ns, errs), 2.0, rtol=1e-12)
        np.testing.assert_allclose(darcy.observed_orders([16, 32], [1.0, 0.5]), [1.0], rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
