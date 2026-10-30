"""Pseudo-spectral Darcy reference of the SF-19 affine-periodic problem (SF-42 N3).

Extends (by copy, not edit) the closure-probe solver `darcy_spectral` of
`docs/experiments/artifacts/2026-10-02-closure-probes/scripts/closure_probe.py` (PCG on the periodic
cell problem) and its box version `darcy_spectral_box` (SF-29 reference.py), with

  * the preconditioner  M = K^{-1/2} (-Delta)^{-1} K^{-1/2}.  With a = K^{1/2},
    K^{-1/2} (-div K grad) K^{-1/2} w = (-Delta + V) w,  V = Delta a / a = (eps/2) Delta Y + (eps/2)^2 |grad Y|^2,
    so M A is the identity plus a term governed by |grad ln K|; it still degrades with the barrier contrast (isolated
    high-K clusters give near-kernel modes): PCG iterations to a true residual of 1e-12 (M = 48, band 1/2, seed 420001)
    are 22 / 52 / 172 / 661 for sigma^2 = 0.25 / 1 / 2.25 / 4 (eps = 0.5 / 1 / 1.5 / 2); a finite-difference multigrid
    preconditioner was tried and was no better (a hybrid with it diverged), so it was dropped;
  * a Galerkin band: the iterates live in the modes |m_d| <= mcut = floor(band M / 2) (default band 1/2) and the
    operator is A' = P A P.  With the full grid band the symbol of the derivative is zero on the Nyquist planes and the
    pseudo-spectral Leibniz rule fails at the band edge for the product with K, which inflated lambda_max of M A to
    ~3e3 (and the iteration counts by 3-10x); the projection BEFORE (-Delta)^{-1} and a retained band of half the
    Nyquist wavenumber (no aliasing of the K spectrum tail into the retained modes once M/2 exceeds the tail) cure it;
  * a hard failure: `ConvergenceError` when the TRUE relative residual ||b - A' u|| / ||b|| exceeds the tolerance at
    `maxit` (a recursive residual that drifts below the tolerance is re-checked against the true one and the iteration
    restarted from it), never an unconverged return;
  * the three corrector problems and their combination into exactly the problem the finite-volume
    comparator solves (`src/physics/flow/AffinePeriodicFlowSolver.cuh`): zero-mean periodic correctors
    div(K (grad w_d + e_d)) = 0, K_eff[i][d] = < K (grad w_d + e_d)_i >, K_eff G = qbar (qbar = (1,0,0)),
    h~ = sum_d G_d w_d, flux q = K (G + grad h~) (so the FV face flux is F = K_f (G.n + dh~/dn));
  * the exact face average of the trigonometric interpolant of q over the FV faces (sinc factors in the
    two transverse directions, exact point evaluation in the normal one), as a real N x M matrix.

Conventions.  Unit periodic box; arrays are (z, y, x) (x fastest), the layout of `Y.npy` and of the
CompactMAC dumps of `apps/sf42_periodic_flow`; direction d = 0, 1, 2 is x, y, z.  Collocation points are the
VERTICES x_a = a / M (no half-cell shift): the continuous solution does not depend on the collocation grid.
The derivative symbol is i k with the Nyquist wavenumber set to zero (D is exactly antisymmetric, so
A = -D K D is symmetric positive semi-definite on the collocation grid).
"""
import time

import numpy as np
import scipy.fft as sfft

TWO_PI = 2.0 * np.pi
DEFAULT_BAND = 0.5  # fraction of the Nyquist wavenumber retained in the Galerkin space


class ConvergenceError(RuntimeError):
    """PCG did not reach the requested TRUE relative residual."""


def _freqs(M, half):
    k = sfft.rfftfreq(M, 1.0 / M) if half else sfft.fftfreq(M, 1.0 / M)
    k = np.array(k, dtype=np.float64)
    if M % 2 == 0:
        k[M // 2] = 0.0  # Nyquist zeroed in the derivative symbol
    return k


class SpectralDarcy:
    """-div(K grad .) on an M^3 collocation grid, K > 0 sampled at the grid points (z, y, x)."""

    def __init__(self, K, workers=None, band=DEFAULT_BAND):
        K = np.ascontiguousarray(K, dtype=np.float64)
        if K.ndim != 3 or not (K.shape[0] == K.shape[1] == K.shape[2]):
            raise ValueError("K must be a cubic 3-d array")
        if not np.all(K > 0.0) or not np.all(np.isfinite(K)):
            raise ValueError("K must be finite and positive")
        self.M = K.shape[0]
        self.K = K
        self.workers = workers
        M = self.M
        kz = TWO_PI * _freqs(M, False)[:, None, None]
        ky = TWO_PI * _freqs(M, False)[None, :, None]
        kx = TWO_PI * _freqs(M, True)[None, None, :]
        self.kd = (kx, ky, kz)  # direction d = 0, 1, 2
        k2 = kz * kz + ky * ky + kx * kx
        with np.errstate(divide="ignore"):
            self.inv_k2 = np.where(k2 > 0.0, 1.0 / np.where(k2 > 0.0, k2, 1.0), 0.0)
        # retained modes |m_d| <= mcut = floor(band * M / 2) (< M/2): the product K (grad u) of a band-limited u with
        # the K spectrum tail stays inside the grid band when mcut + tail <= M - mcut (band 1/2: tail <= M/2)
        self.mcut = min(int(np.floor(band * M / 2.0)), M // 2 - 1)
        self.Kmh = K ** -0.5
        self.matvecs = 0

    # --- transforms -------------------------------------------------------------------------
    def rfft(self, a):
        return sfft.rfftn(a, workers=self.workers)

    def irfft(self, ah):
        return sfft.irfftn(ah, s=(self.M,) * 3, workers=self.workers)

    def _band_limit(self, ah):
        """Zero all modes with |m_d| > mcut in a half-spectrum array, in place (the projector P of the Galerkin space)."""
        c, M = self.mcut, self.M
        ah[c + 1:M - c, :, :] = 0.0
        ah[:, c + 1:M - c, :] = 0.0
        ah[:, :, c + 1:] = 0.0
        return ah

    def grad(self, u, uh=None):
        uh = self.rfft(u) if uh is None else uh
        return [self.irfft(1j * k * uh) for k in self.kd]

    # --- operators --------------------------------------------------------------------------
    def apply_A(self, u):
        """A u = -div_s(K grad_s u); exactly zero mean (the k = 0 symbol multiplies the sum)."""
        self.matvecs += 1
        uh = self.rfft(u)
        acc = None
        for k in self.kd:
            g = self.irfft(1j * k * uh)
            g *= self.K
            t = self.rfft(g)
            t *= 1j * k
            acc = t if acc is None else acc + t
        return -self.irfft(self._band_limit(acc))  # A' = P A P: Galerkin in the band-limited space

    def precondition(self, r):
        """M r = K^{-1/2} (-Delta)^{-1} (K^{-1/2} r), projected on the retained band, zero mode and all-Nyquist
        corners dropped (inv_k2 = 0 there).  The retained band is imposed BEFORE (-Delta)^{-1}: the derivative symbol
        is zero on the Nyquist planes, so aliasing noise kept there would be amplified by ~(pi M / 2 pi m)^2
        (lambda_max ~ 3e3 measured at M = 48 against ~1e2 with the projection)."""
        s = self._band_limit(self.rfft(r * self.Kmh))
        s *= self.inv_k2
        z = self.irfft(s)
        z *= self.Kmh
        return self.irfft(self._band_limit(self.rfft(z)))

    def rhs(self, d):
        """b_d = D_d K (the corrector problem A w_d = D_d K, i.e. div(K (grad w_d + e_d)) = 0)."""
        return self.irfft(self._band_limit(1j * self.kd[d] * self.rfft(self.K)))

    # --- PCG with a true-residual guarantee ---------------------------------------------------
    def solve(self, b, tol=1e-12, maxit=2000, max_restarts=6):
        """Solve A u = b (b of zero mean) to a TRUE relative residual <= tol; returns (u, info).
        Raises ConvergenceError otherwise (at maxit, after max_restarts true-residual restarts, or on
        a non-positive curvature)."""
        nb = float(np.linalg.norm(b))
        info = {"iterations": 0, "restarts": 0, "true_relative_residual": 0.0, "history": []}
        if nb == 0.0:  # K independent of x_d: the corrector is zero
            return np.zeros_like(b), info
        u = np.zeros_like(b)
        r = b.copy()
        z = self.precondition(r)
        p = z.copy()
        rz = float(np.vdot(r, z))
        if not rz > 0.0:
            raise ConvergenceError("PCG breakdown: r.M.r = %.3e <= 0 at the start" % rz)
        while info["iterations"] < maxit:
            Ap = self.apply_A(p)
            pAp = float(np.vdot(p, Ap))
            if not pAp > 0.0:
                raise ConvergenceError("PCG breakdown: p.A.p = %.3e <= 0 at iteration %d" % (pAp, info["iterations"]))
            alpha = rz / pAp
            u += alpha * p
            r -= alpha * Ap
            info["iterations"] += 1
            rel = float(np.linalg.norm(r)) / nb
            info["history"].append(rel)
            if rel <= tol:
                rt = b - self.apply_A(u)
                true = float(np.linalg.norm(rt)) / nb
                info["true_relative_residual"] = true
                if true <= tol:
                    u -= u.mean()
                    return u, info
                info["restarts"] += 1
                if info["restarts"] > max_restarts:
                    break
                r = rt  # restart from the true residual
                z = self.precondition(r)
                p = z.copy()
                rz = float(np.vdot(r, z))
                continue
            z = self.precondition(r)
            rz_new = float(np.vdot(r, z))
            if not rz_new > 0.0:
                raise ConvergenceError("PCG breakdown: r.M.r = %.3e <= 0 at iteration %d" % (rz_new, info["iterations"]))
            p = z + (rz_new / rz) * p
            rz = rz_new
        true = float(np.linalg.norm(b - self.apply_A(u))) / nb
        raise ConvergenceError("PCG not converged: true relative residual %.3e > tol %.3e after %d iterations "
                               "(%d restarts)" % (true, tol, info["iterations"], info["restarts"]))


class Reference:
    """Solution of the affine-periodic problem (qbar) on one collocation grid."""

    def __init__(self, K, qbar=(1.0, 0.0, 0.0), tol=1e-12, maxit=2000, workers=None, band=DEFAULT_BAND):
        t0 = time.time()
        op = SpectralDarcy(K, workers, band)
        self.op = op
        self.M = op.M
        self.tol = tol
        self.correctors = []
        w = []
        for d in range(3):
            wd, inf = op.solve(op.rhs(d), tol=tol, maxit=maxit)
            w.append(wd)
            inf.pop("history")
            self.correctors.append(inf)
        # K_eff[i][d] = < K (D_i w_d + delta_id) >
        Keff = np.zeros((3, 3))
        for d in range(3):
            g = op.grad(w[d])
            for i in range(3):
                Keff[i, d] = float(np.mean(op.K * g[i]) + (op.K.mean() if i == d else 0.0))
        self.K_eff = Keff
        self.symmetry_defect_rel = float(np.max(np.abs(Keff - Keff.T)) / np.max(np.abs(Keff)))
        self.G = np.linalg.solve(Keff, np.asarray(qbar, dtype=np.float64))
        ht = self.G[0] * w[0] + self.G[1] * w[1] + self.G[2] * w[2]
        del w
        gh = op.grad(ht)
        self.q = [op.K * (self.G[i] + gh[i]) for i in range(3)]  # flux components, (z, y, x) arrays
        self.mean_flux = [float(np.mean(qi)) for qi in self.q]
        self.seconds = time.time() - t0
        self.matvecs = op.matvecs

    def summary(self):
        return {"M": self.M, "tol": self.tol, "mcut": self.op.mcut, "K_eff": self.K_eff.tolist(),
                "symmetry_defect_rel": self.symmetry_defect_rel, "G": self.G.tolist(),
                "mean_flux": self.mean_flux, "correctors": self.correctors,
                "matvecs": self.matvecs, "seconds": self.seconds}


# --- exact face averages of the trigonometric interpolant -------------------------------------------
def face_operators(N, M):
    """Real (N, M) matrices T_pt, T_avg mapping the M vertex samples q(a / M) of a periodic function (Nyquist
    mode dropped) to (T_pt) its exact value at the FV face planes x = i / N and (T_avg) its exact average
    over the FV cells [j/N, (j+1)/N]: T[i, a] = (1/M) sum_m w_m cos(2 pi m (x_i - a/M)),  |m| < M/2,
    w_m = 1 (point) or sinc(pi m / N) (average).  Phases are formed from exact integers (no 1e3-rad
    arguments)."""
    m = np.arange(-(M // 2) + 1, M // 2)
    a = np.arange(M)
    ang_a = TWO_PI * (np.outer(m, a) % M) / M  # (m, a): 2 pi m a / M
    ca, sa = np.cos(ang_a), np.sin(ang_a)

    def build(num, den, w):  # x_i = num_i / den  (num integers)
        ang_i = TWO_PI * (np.outer(num, m) % den) / den  # (i, m)
        return ((np.cos(ang_i) * w) @ ca + (np.sin(ang_i) * w) @ sa) / M

    i = np.arange(N)
    T_pt = build(i, N, np.ones(m.size))
    # cell centres (2 j + 1) / (2 N); 2 pi m (2j+1) / (2N): integer numerator m (2j+1) mod 2N
    ang_c = np.pi * (np.outer(2 * i + 1, m) % (2 * N)) / N
    w_avg = np.sinc(m / N)
    T_avg = ((np.cos(ang_c) * w_avg) @ ca + (np.sin(ang_c) * w_avg) @ sa) / M
    return T_pt, T_avg


def _along(T, a, axis):
    return np.moveaxis(np.tensordot(T, a, axes=(1, axis)), 0, axis)


def face_fluxes(q, N, ops=None):
    """Exact FV face averages (U, V, W) of the flux components q = (q_x, q_y, q_z) given as (z,y,x) arrays on the
    M^3 vertex grid.  Shapes (N, N, N+1), (N, N+1, N), (N+1, N, N); the periodic duplicate plane is plane 0."""
    M = q[0].shape[0]
    T_pt, T_avg = ops if ops is not None else face_operators(N, M)
    out = []
    for d in range(3):
        a = q[d]
        for ax in (0, 1, 2):  # axis 0 = z, 1 = y, 2 = x; direction d lives on axis 2 - d
            a = _along(T_pt if ax == 2 - d else T_avg, a, ax)
        ax = 2 - d
        out.append(np.concatenate([a, np.take(a, [0], axis=ax)], axis=ax))
    return tuple(out)


# --- error measures and orders ------------------------------------------------------------------
def unique_faces(U, V, W):
    """Drop the periodic duplicate planes."""
    return U[:, :, :-1], V[:, :-1, :], W[:-1, :, :]


def rel_l2(a, b):
    """||a - b||_2 / ||b||_2 over the arrays."""
    return float(np.linalg.norm((a - b).ravel()) / np.linalg.norm(b.ravel()))


def flux_errors(fv, ref):
    """Relative L2 (x-faces, all faces) and max-norm (relative to the reference rms) of fv - ref over the unique
    faces; fv, ref are (U, V, W) triples with the duplicate planes."""
    f, r = unique_faces(*fv), unique_faces(*ref)
    num = sum(float(np.sum((fi - ri) ** 2)) for fi, ri in zip(f, r))
    den = sum(float(np.sum(ri ** 2)) for ri in r)
    cnt = sum(ri.size for ri in r)
    linf = max(float(np.max(np.abs(fi - ri))) for fi, ri in zip(f, r))
    return {"rel_l2_x": rel_l2(f[0], r[0]), "rel_l2_all": float(np.sqrt(num / den)),
            "rel_linf_all": linf / float(np.sqrt(den / cnt))}


def observed_orders(ns, errs):
    """p_i = ln(e_i / e_{i+1}) / ln(N_{i+1} / N_i) between consecutive levels (h = 1/N)."""
    return [float(np.log(errs[i] / errs[i + 1]) / np.log(ns[i + 1] / ns[i])) for i in range(len(ns) - 1)]


def estimate_cost(M, iterations=150):
    """Rough device-free estimate for one `Reference` at collocation size M: bytes (about 20 real arrays of M^3 plus
    transforms) and the FFT count (10 per PCG iteration, 3 correctors)."""
    arrays = 22
    return {"M": M, "bytes": arrays * 8 * M ** 3, "ffts": 3 * iterations * 10 + 60}
