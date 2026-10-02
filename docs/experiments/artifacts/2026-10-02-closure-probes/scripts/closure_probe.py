"""Streamline-closure probe (independent of any streamfunction solver).

Question: for a smooth, scalar (locally isotropic), triply periodic conductivity
k = exp(f) with mean gradient along x1, are Darcy streamlines closed on T^3
(return map of the face x1 = 0 equal to the identity)?  This is a NECESSARY
condition for the existence of nondegenerate invariants psi_i = affine + periodic.

Method: pseudo-spectral Darcy solve (PCG, spectral accuracy), streamlines of
grad(phi) parametrized by x1 (dx_perp/dx1 = d_perp phi / d_1 phi; k cancels),
exact trigonometric evaluation of grad(phi) (direct Fourier sum), DOP853.
Controls: 2-D field (must close), resolution convergence, amplitude scaling,
forward/backward integrator round trip.
"""
import sys, time
import numpy as np
from scipy.integrate import solve_ivp

TWO_PI = 2.0 * np.pi

def wavenumbers(N):
    m = np.fft.fftfreq(N, d=1.0 / N)
    md = m.copy()
    if N % 2 == 0:
        md[N // 2] = 0.0            # zero the Nyquist mode in derivatives
    return m, md

def darcy_spectral(f, j=0, tol=1e-13, maxit=2000):
    N = f.shape[0]
    m, md = wavenumbers(N)
    K1, K2, K3 = np.meshgrid(md, md, md, indexing="ij")
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
    b = np.fft.ifftn(div(*e)).real                    # div(k grad(phibar)), phibar = -x_j
    b -= b.mean()
    u = np.zeros_like(f); r = b.copy(); z = M(r); p = z.copy(); rz = np.vdot(r, z)
    b0 = np.linalg.norm(b)
    if b0 < 1e-13 * np.sqrt(b.size):                  # k independent of x_j: trivial cell problem
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
    flux = np.array([-(k * gi).mean() for gi in g])   # mean Darcy flux for mean gradient -e_j
    return u, res, it + 1, flux

class TrigField:
    """grad(phi) for phi = -x1 + sum_m c_m exp(2 pi i m.x), exact evaluation."""
    def __init__(self, u, G, cut=1e-15):
        self.G = np.asarray(G, float)
        N = u.shape[0]
        c = np.fft.fftn(u) / u.size
        m, md = wavenumbers(N)
        M1, M2, M3 = np.meshgrid(md, md, md, indexing="ij")
        if N % 2 == 0:                       # drop Nyquist planes entirely
            c[N // 2, :, :] = 0; c[:, N // 2, :] = 0; c[:, :, N // 2] = 0
        keep = np.abs(c) > cut * np.abs(c).max()
        self.c = c[keep]; self.m = np.stack([M1[keep], M2[keep], M3[keep]], axis=1)
        self.tail = np.abs(c[~keep]).sum()
    def grad(self, x):                       # x: (P,3) -> (P,3)
        ph = np.exp(1j * TWO_PI * (x @ self.m.T)) * self.c       # (P,M)
        g = (1j * TWO_PI * (ph @ self.m)).real
        return g + self.G

def return_map(field, y0, z0, rtol=1e-12, atol=1e-14, x1_end=1.0):
    P = y0.size
    def rhs(x1, s):
        x = np.empty((P, 3)); x[:, 0] = x1; x[:, 1] = s[:P]; x[:, 2] = s[P:]
        g = field.grad(x)
        return np.concatenate([g[:, 1] / g[:, 0], g[:, 2] / g[:, 0]])
    s0 = np.concatenate([y0, z0])
    sol = solve_ivp(rhs, (0.0, x1_end), s0, method="DOP853", rtol=rtol, atol=atol)
    s1 = sol.y[:, -1]
    back = solve_ivp(rhs, (x1_end, 0.0), s1, method="DOP853", rtol=rtol, atol=atol)
    rt = np.abs(back.y[:, -1] - s0).max()
    return s1[:P] - y0, s1[P:] - z0, rt, sol.nfev

def grid(N):
    x = np.arange(N) / N
    return np.meshgrid(x, x, x, indexing="ij")

FIELDS = {}
def field(name):
    def deco(fn): FIELDS[name] = fn; return fn
    return deco

@field("generic3d")      # four oblique modes, no symmetry
def _(X, Y, Z):
    return (np.cos(TWO_PI * (X + Y)) + np.cos(TWO_PI * (X + Z) + 0.7)
            + 0.8 * np.cos(TWO_PI * (X - Y + Z) + 1.3) + 0.6 * np.sin(TWO_PI * (2 * X + Y - Z)))
@field("two_mode")       # f(theta1, theta2): one continuous symmetry (along n = p x q)
def _(X, Y, Z):
    return np.cos(TWO_PI * (X + Y)) + np.cos(TWO_PI * (X + Z))
@field("control2d")      # independent of x3: 2-D flow, streamlines must close
def _(X, Y, Z):
    return np.cos(TWO_PI * (X + Y)) + 0.8 * np.sin(TWO_PI * (2 * X - Y) + 0.4) + 0.5 * np.cos(TWO_PI * Y)
@field("lester2021")     # Lester et al. (2021) eq. (3.1), coefficient 2/5
def _(X, Y, Z):
    return (np.sin(TWO_PI * X) * np.cos(TWO_PI * Y) * np.sin(TWO_PI * Z)
            + 0.4 * np.sin(TWO_PI * X) * np.sin(4 * TWO_PI * Z))

@field("lester_brk")     # Lester (2021) field with the x1 -> 1/2 - x1 mirror symmetry broken by a phase
def _(X, Y, Z):
    return (np.sin(TWO_PI * X) * np.cos(TWO_PI * Y) * np.sin(TWO_PI * Z)
            + 0.4 * np.sin(TWO_PI * X + 0.9) * np.sin(4 * TWO_PI * Z))
@field("gauss")          # band-limited Gaussian random field, Gaussian covariance ell = 1/4, unit variance, seed 7
def _(X, Y, Z):
    rng = np.random.default_rng(7); ell = 0.25; out = np.zeros_like(X); var = 0.0
    for m1 in range(-3, 4):
        for m2 in range(-3, 4):
            for m3 in range(-3, 4):
                if (m1, m2, m3) <= (0, 0, 0): continue          # half space, no zero mode
                amp = rng.standard_normal() * np.exp(-(TWO_PI**2) * (m1*m1 + m2*m2 + m3*m3) * ell**2 / 8.0)
                th = rng.uniform(0, TWO_PI)
                out += amp * np.cos(TWO_PI * (m1 * X + m2 * Y + m3 * Z) + th); var += 0.5 * amp * amp
    return out / np.sqrt(var)

def run(name, eps, N, npts=4, seed=3):
    X, Y, Z = grid(N)
    f = eps * FIELDS[name](X, Y, Z)
    t0 = time.time()
    sols = [darcy_spectral(f, j) for j in range(3)]
    Keff = np.stack([s[3] for s in sols], axis=1)          # flux = Keff @ a for mean gradient -a
    a = np.linalg.solve(Keff, np.array([1.0, 0.0, 0.0]))   # mean flux = e1 exactly
    u = sum(a[j] * sols[j][0] for j in range(3)); res = max(s[1] for s in sols); its = max(s[2] for s in sols)
    fld = TrigField(u, -a)
    m, md = wavenumbers(N)
    K1 = md[:, None, None]
    d1 = -a[0] + np.fft.ifftn(1j * TWO_PI * K1 * np.fft.fftn(u)).real
    tilt = np.hypot(a[1], a[2]) / abs(a[0])
    rng = np.random.default_rng(seed)
    y0 = rng.random(npts * npts); z0 = rng.random(npts * npts)
    dy, dz, rt, nfev = return_map(fld, y0, z0)
    d = np.hypot(dy, dz)
    my, mz = dy.mean(), dz.mean()
    nonuni = np.sqrt(((dy - my)**2 + (dz - mz)**2).mean())          # drift-independent measure
    par = (dy + dz) / np.sqrt(2); perp = (dy - dz) / np.sqrt(2)      # two_mode: shear along (1,1)
    extra = f" mean=({my:+.2e},{mz:+.2e}) nonuniform_rms={nonuni:.3e} |d.(1,-1)|max={np.abs(perp).max():.1e}"
    print(f"{name:10s} eps={eps:5.3f} N={N:3d} tilt={tilt:.1e} pcg_res={res:.1e}({its:3d}) modes={fld.c.size:6d} "
          f"tail={fld.tail:.1e} max(d1phi)={d1.max():+.3f} | return-map |d|: max={d.max():.3e} "
          f"rms={np.sqrt((d**2).mean()):.3e} | roundtrip={rt:.1e} nfev={nfev} t={time.time()-t0:.0f}s" + extra, flush=True)
    return d.max()

if __name__ == "__main__":
    for spec in sys.argv[1:]:
        name, eps, N = spec.split(":")
        run(name, float(eps), int(N))
