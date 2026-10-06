"""SF-29 N1 -- independent Darcy reference (pseudo-spectral) on a periodic box.

Provenance: `darcy_spectral_box` is `darcy_spectral` of
docs/experiments/artifacts/2026-10-02-closure-probes/scripts/closure_probe.py generalized to a
non-cubic box [0,L1]x[0,L2]x[0,L3] with an (N1,N2,N3) grid.  For L = (1,1,1) and a cubic grid
it performs the same floating-point operations in the same order as the original and is
bit-identical to it (checked by `oracle.py --selftest`).  The 2026-10-02 artifact is imported
read-only (`FIELDS`, `darcy_spectral`, `TrigField`, `wavenumbers`) and never modified.

Two Darcy flows (SF-29 UNDERSTAND record, sections 1 and 2.4):
  (a) periodic fields `<name>` of closure_probe.FIELDS: triply periodic cell flow of
      k = exp(eps f), mean flux exactly e1 (three cell problems, mean gradient Keff^-1 e1);
  (b) constant-head fields `<name>_ch`: mirror trick k_b(x) = exp(eps f(g(x1), x2, x3)),
      g(x1) = (1 - cos(pi x1))/2, periodic cell [0,2]x[0,1]^2, grid (2 Nphi, Nphi, Nphi), mean
      gradient along x1 only.  The fluctuation potential is odd about x1 = 0 and x1 = 1, so phi is
      constant on both faces; the slab [0,1] carries the constant-head flow.  Mean flux through
      every x1-plane normalized to 1.
`uniform` (f = 0, k = 1) is a positive control (affine labels).

grad(phi) is evaluated EXACTLY from the trigonometric interpolant of the fluctuation potential
(direct Fourier sum, Nyquist planes dropped as in closure_probe.TrigField, no small-coefficient
cut), in separable form: streamlines are parametrized by x1, so every vectorized evaluation
shares one x1; the x1 sum is contracted first, then the 2-D sum over (m2, m3 >= 0) is one
matrix product (real field: Re[c(m3=0) + 2 sum_{m3>0}]).
"""
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_CLOSURE = os.path.normpath(os.path.join(_HERE, "..", "..", "2026-10-02-closure-probes", "scripts"))
if _CLOSURE not in sys.path:
    sys.path.insert(0, _CLOSURE)
import closure_probe as cp  # noqa: E402  (read-only reuse of the 2026-10-02 artifact)

TWO_PI = 2.0 * np.pi
PERIODIC_FIELDS = tuple(cp.FIELDS.keys())   # generic3d, two_mode, control2d, lester2021, lester_brk, gauss
CH_FIELDS = ("gauss_ch",)
ALL_FIELDS = PERIODIC_FIELDS + CH_FIELDS


def base_field_fn(name):
    """f(X, Y, Z) of a field name (periodic, or the base of a mirrored `<name>_ch`); 'uniform' is f = 0."""
    base = name[:-3] if name.endswith("_ch") else name
    if base == "uniform":
        return lambda X, Y, Z: 0.0 * (np.asarray(X) + np.asarray(Y) + np.asarray(Z))
    if base not in cp.FIELDS:
        raise KeyError("unknown field %r (known: %s, <field>_ch, uniform)" % (name, ", ".join(cp.FIELDS)))
    return cp.FIELDS[base]


def is_constant_head(name):
    return name.endswith("_ch")


def mirror_g(x1):
    return 0.5 * (1.0 - np.cos(np.pi * x1))


def mirror_dg(x1):
    return 0.5 * np.pi * np.sin(np.pi * x1)


class LogConductivity:
    """Analytic ln k = eps f (mirrored for `_ch`) and its gradient.

    The gradient uses the complex-step derivative Im f(x + i h e_l)/h, h = 1e-30, which is exact
    to roundoff for the analytic (trigonometric) fields: no finite differencing of ln k.
    """

    H_CS = 1e-30

    def __init__(self, name, eps):
        self.name, self.eps = name, float(eps)
        self.f = base_field_fn(name)
        self.ch = is_constant_head(name)

    @staticmethod
    def _args(X1, X2, X3):
        return np.broadcast_arrays(np.asarray(X1, float), np.asarray(X2, float), np.asarray(X3, float))

    def lnk(self, X1, X2, X3):
        X1, X2, X3 = self._args(X1, X2, X3)
        a = mirror_g(X1) if self.ch else X1
        return self.eps * np.real(self.f(a, X2, X3))

    def k(self, X1, X2, X3):
        return np.exp(self.lnk(X1, X2, X3))

    def grad_lnk(self, X1, X2, X3):
        X1, X2, X3 = self._args(X1, X2, X3)
        a = mirror_g(X1) if self.ch else X1
        h = self.H_CS
        a, X2, X3 = a.astype(complex), X2.astype(complex), X3.astype(complex)   # fields allocate like X
        d1 = np.imag(self.f(a + 1j * h, X2, X3)) / h
        d2 = np.imag(self.f(a, X2 + 1j * h, X3)) / h
        d3 = np.imag(self.f(a, X2, X3 + 1j * h)) / h
        if self.ch:
            d1 = d1 * mirror_dg(X1)
        return self.eps * d1, self.eps * d2, self.eps * d3


def wavenumbers(N):
    return cp.wavenumbers(N)


def darcy_spectral_box(f, j=0, L=(1.0, 1.0, 1.0), tol=1e-13, maxit=4000):
    """closure_probe.darcy_spectral on [0,L1]x[0,L2]x[0,L3]; f = ln k sampled on an (N1, N2, N3) grid.

    Solves -div(k grad u) = -d_j k (phi = -x_j + u) by PCG preconditioned by the constant-coefficient
    Laplacian scaled by mean(k).  Returns (u, relative residual, iterations, mean Darcy flux for the
    mean gradient -e_j).  The only change from the original is the wavenumber scaling 1/L_i, which is
    skipped when L = (1,1,1) so that the unit-box path is bit-identical (maxit differs: 4000 vs 2000;
    it is a safety cap only, never reached in the recorded runs).
    """
    N1, N2, N3 = f.shape
    md = [wavenumbers(n)[1] for n in (N1, N2, N3)]
    if all(float(Li) == 1.0 for Li in L):
        kv = md
    else:
        kv = [md[i] / float(L[i]) for i in range(3)]
    K1, K2, K3 = np.meshgrid(kv[0], kv[1], kv[2], indexing="ij")
    k2 = K1**2 + K2**2 + K3**2
    k2p = k2.copy(); k2p[k2p == 0] = 1.0
    k = np.exp(f)
    kbar = k.mean()

    def grad(uh):
        return [np.fft.ifftn(1j * TWO_PI * K * uh).real for K in (K1, K2, K3)]

    def div(vx, vy, vz):
        return 1j * TWO_PI * (K1 * np.fft.fftn(vx) + K2 * np.fft.fftn(vy) + K3 * np.fft.fftn(vz))

    def A(u):
        g = grad(np.fft.fftn(u))
        r = -np.fft.ifftn(div(k * g[0], k * g[1], k * g[2])).real
        return r - r.mean()

    def M(r):
        rh = np.fft.fftn(r) / (TWO_PI**2 * k2p * kbar)
        rh[k2 == 0] = 0.0
        return np.fft.ifftn(rh).real

    e = [0 * k, 0 * k, 0 * k]; e[j] = -k
    b = np.fft.ifftn(div(*e)).real
    b -= b.mean()
    u = np.zeros_like(f); r = b.copy(); z = M(r); p = z.copy(); rz = np.vdot(r, z)
    b0 = np.linalg.norm(b)
    if b0 < 1e-13 * np.sqrt(b.size):
        flux = np.zeros(3); flux[j] = kbar
        return np.zeros_like(f), 0.0, 0, flux
    for it in range(maxit):
        Ap = A(p); a = rz / np.vdot(p, Ap)
        u += a * p; r -= a * Ap
        if np.linalg.norm(r) <= tol * b0:
            break
        z = M(r); rzn = np.vdot(r, z); p = z + (rzn / rz) * p; rz = rzn
    res = np.linalg.norm(b - A(u)) / b0
    g = grad(np.fft.fftn(u)); g[j] = g[j] - 1.0
    flux = np.array([-(k * gi).mean() for gi in g])
    return u, res, it + 1, flux


def phase_table(x, m, L=1.0):
    """exp(i 2 pi m x / L) for points x (P,) and integer modes m (M,) -> (P, M).

    Powers of exp(i 2 pi x / L) by cumulative product, negative modes by conjugation: ~3x cheaper
    than one complex exp per entry; differs from the direct exp by ~2e-16 in grad(phi) (measured at
    N_phi = 48, x in [0, 1)); the error bound grows like max|m| ulp.
    """
    x = np.asarray(x, float).ravel()
    mi = np.rint(np.asarray(m, float)).astype(int)
    mmax = int(np.abs(mi).max()) if mi.size else 0
    pw = np.empty((x.size, mmax + 1), dtype=complex)
    pw[:, 0] = 1.0
    if mmax > 0:
        pw[:, 1:] = np.exp(1j * (TWO_PI / L) * x)[:, None]
        np.cumprod(pw[:, 1:], axis=1, out=pw[:, 1:])
    out = pw[:, np.abs(mi)]
    neg = mi < 0
    out[:, neg] = np.conj(out[:, neg])
    return out


class SepTrigField:
    """grad(phi) for phi = G.x + u_trig(x), u_trig the trigonometric interpolant of u on [0,L).

    Nyquist planes dropped (as closure_probe.TrigField); no small-coefficient cut.
    `grad(x1, y, z)`: x1 a scalar shared by all points, y, z arrays (P,) -> (P, 3).
    """

    def __init__(self, u, G, L=(1.0, 1.0, 1.0)):
        N1, N2, N3 = u.shape
        self.G = np.asarray(G, float)
        self.L = tuple(float(x) for x in L)
        ch = np.fft.rfftn(u) / u.size                         # (N1, N2, N3//2+1)
        m1 = wavenumbers(N1)[1]; m2 = wavenumbers(N2)[1]
        m3 = np.arange(N3 // 2 + 1, dtype=float)
        if N1 % 2 == 0:
            ch[N1 // 2, :, :] = 0
        if N2 % 2 == 0:
            ch[:, N2 // 2, :] = 0
        if N3 % 2 == 0:
            ch[:, :, N3 // 2] = 0
            m3[N3 // 2] = 0.0
        w3 = np.where(np.arange(N3 // 2 + 1) == 0, 1.0, 2.0)
        ch = ch * w3[None, None, :]
        self.m2, self.m3 = m2, m3
        self.k1 = TWO_PI * m1 / self.L[0]
        self.k2 = TWO_PI * m2 / self.L[1]
        self.k3 = TWO_PI * m3 / self.L[2]
        self.n3h = ch.shape[2]
        d1 = 1j * self.k1[:, None, None] * ch
        d2 = 1j * self.k2[None, :, None] * ch
        d3 = 1j * self.k3[None, None, :] * ch
        self.n2 = N2
        # layout (N1, N2, q, N3h) flattened to (N1, N2*q*N3h): the x1 contraction is one gemv and
        # yields directly the (N2, q*N3h) matrix of the 2-D sum
        self.cd = np.ascontiguousarray(np.stack([d1, d2, d3], axis=2).reshape(N1, -1))
        self.c0 = np.ascontiguousarray(ch.reshape(N1, -1))
        self.lap = np.ascontiguousarray((-(self.k1[:, None, None]**2 + self.k2[None, :, None]**2
                                           + self.k3[None, None, :]**2) * ch).reshape(N1, -1))
        self.nmodes = int(np.count_nonzero(ch))

    def _sum2d(self, B, q, y, z):
        """B: (N2, q*N3h) -> (P, q)."""
        E2 = phase_table(y, self.m2, self.L[1])
        E3 = phase_table(z, self.m3, self.L[2])
        T = (E2 @ B).reshape(y.size, q, self.n3h)
        return np.einsum("pqm,pm->pq", T, E3).real

    def _collapse(self, M, x1):
        e1 = np.exp(1j * self.k1 * float(x1))
        return (e1 @ M).reshape(self.n2, -1)

    def grad(self, x1, y, z):
        y = np.asarray(y, float).ravel(); z = np.asarray(z, float).ravel()
        return self._sum2d(self._collapse(self.cd, x1), 3, y, z) + self.G[None, :]

    def laplacian(self, x1, y, z):
        """Laplacian of u_trig (the affine part has none)."""
        y = np.asarray(y, float).ravel(); z = np.asarray(z, float).ravel()
        return self._sum2d(self._collapse(self.lap, x1), 1, y, z)[:, 0]

    def value(self, x1, y, z):
        """u_trig at (x1, y, z) (without the affine part)."""
        y = np.asarray(y, float).ravel(); z = np.asarray(z, float).ravel()
        return self._sum2d(self._collapse(self.c0, x1), 1, y, z)[:, 0]


class DarcyReference:
    """Darcy flow of field `name` at amplitude `eps`, reference resolution `nphi`."""

    def __init__(self, name, eps, nphi, tol=1e-13, verbose=False):
        t0 = time.time()
        self.name, self.eps, self.nphi = name, float(eps), int(nphi)
        self.ch = is_constant_head(name)
        self.lk = LogConductivity(name, eps)
        n = self.nphi
        if self.ch:
            self.L = (2.0, 1.0, 1.0); shape = (2 * n, n, n)
        else:
            self.L = (1.0, 1.0, 1.0); shape = (n, n, n)
        x = [np.arange(shape[i]) / float(n) for i in range(3)]   # spacing 1/nphi in every direction
        X1, X2, X3 = np.meshgrid(x[0], x[1], x[2], indexing="ij")
        if self.ch:
            f = self.lk.lnk(X1, X2, X3)                      # eps f(g(x1), x2, x3)
        else:
            f = self.eps * base_field_fn(name)(X1, X2, X3)    # same sampling as closure_probe.run
        del X1, X2, X3
        if self.ch:
            u0, res, its, flux = darcy_spectral_box(f, 0, self.L, tol=tol)
            self.transverse_flux = flux[1:].copy()           # vanishes by the mirror symmetry
            a = np.array([1.0 / flux[0], 0.0, 0.0])          # mean flux = e1
            u = a[0] * u0
            self.pcg = (res, its)
            self.Keff = None
        else:
            sols = [darcy_spectral_box(f, jj, self.L, tol=tol) for jj in range(3)]
            Keff = np.stack([s[3] for s in sols], axis=1)
            a = np.linalg.solve(Keff, np.array([1.0, 0.0, 0.0]))
            u = sum(a[jj] * sols[jj][0] for jj in range(3))
            self.pcg = (max(s[1] for s in sols), max(s[2] for s in sols))
            self.Keff = Keff
            self.transverse_flux = None
        self.a = a
        self.u = u
        self.field = SepTrigField(u, -a, self.L)
        md1 = wavenumbers(shape[0])[1] / self.L[0]
        d1 = -a[0] + np.fft.ifftn(1j * TWO_PI * md1[:, None, None] * np.fft.fftn(u)).real
        self.max_d1phi = float(d1.max())                     # v1 > 0  <=>  d1 phi < 0
        self.shape = shape
        self.t_solve = time.time() - t0
        if verbose:
            print("REF field=%s eps=%g nphi=%d shape=%s pcg_res=%.1e its=%d max(d1phi)=%+.4f a=(%+.6f,%+.2e,%+.2e) "
                  "modes=%d t=%.1fs" % (name, eps, n, shape, self.pcg[0], self.pcg[1], self.max_d1phi,
                                        a[0], a[1], a[2], self.field.nmodes, self.t_solve), flush=True)

    def check_positive_v1(self):
        if not self.max_d1phi < 0.0:
            raise RuntimeError("ABORT case %s eps=%g nphi=%d: v1 > 0 violated (max d1phi = %+.3e >= 0); "
                               "streamlines may not all reach the inlet" % (self.name, self.eps, self.nphi,
                                                                             self.max_d1phi))

    def continuity_residual(self, npts=4096, seed=5):
        """Pointwise div(k grad phi_trig)/k = lap u + grad ln k . grad phi at random slab points.

        Returns RMS(residual) / (RMS|grad ln k| RMS|grad phi|): the continuum mass-balance error of
        the reference (aliasing of the collocation solve); 0 for an exact Darcy flow.
        """
        rng = np.random.default_rng(seed)
        nx = 16
        res = []; gl2 = []; gp2 = []
        for x1 in (np.arange(nx) + 0.5) / nx:            # slab 0 <= x1 <= 1 in both cases
            y = rng.random(npts // nx); z = rng.random(npts // nx)
            g = self.field.grad(x1, y, z)
            lap = self.field.laplacian(x1, y, z)
            gl = np.stack(self.lk.grad_lnk(np.full(y.shape, x1), y, z), axis=1)
            res.append(lap + (gl * g).sum(axis=1)); gl2.append((gl**2).sum(axis=1)); gp2.append((g**2).sum(axis=1))
        res = np.concatenate(res); gl2 = np.concatenate(gl2); gp2 = np.concatenate(gp2)
        den = np.sqrt(gl2.mean() * gp2.mean())
        return float(np.sqrt((res**2).mean()) / den) if den > 0 else float(np.sqrt((res**2).mean()))

    def grad_phi(self, x1, y, z):
        return self.field.grad(x1, y, z)

    def velocity(self, x1, y, z):
        """Darcy velocity -k grad(phi) at (shared x1, y, z) -> (P, 3)."""
        g = self.field.grad(x1, y, z)
        y = np.asarray(y, float).ravel(); z = np.asarray(z, float).ravel()
        kk = self.lk.k(np.full(y.shape, float(x1)), y, z)
        return -kk[:, None] * g
