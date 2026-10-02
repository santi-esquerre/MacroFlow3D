"""Equation (14) as the Euler-Lagrange system of the dissipation functional
    E[psi1, psi2] = 1/2 < |grad psi1 x grad psi2|^2 / k >,   psi_i = affine + periodic.
Discretize E (not the equations) and minimize with its EXACT discrete gradient
    dE/du1 = -div_h(g2 x w),  dE/du2 = -div_h(w x g1),  w = c/k,
(L-BFGS in H^1-preconditioned variables).  Question: does the variational discretization have a
residual floor?  What is the minimizer's distance to the Darcy flow?  Identity used:
    1/2 <|c - v_D|^2 / k> = E[c] - E_Darcy   (same mean flux), so (E_min - E_D)/E_D is gauge-free."""
import sys, time, numpy as np
from scipy.optimize import minimize
from closure_probe import FIELDS, grid, wavenumbers, darcy_spectral, TWO_PI

def run(name, eps, N, disc, maxit=4000):
    X, Y, Z = grid(N); f = eps * FIELDS[name](X, Y, Z); k = np.exp(f); h = 1.0 / N; n3 = N**3
    mm, md = wavenumbers(N)
    KO = np.meshgrid(TWO_PI * md, TWO_PI * md, TWO_PI * md, indexing="ij")
    if disc == "sp":
        D = lambda u: [np.fft.ifftn(1j * KO[a] * np.fft.fftn(u)).real for a in range(3)]
        sym = [KO[a]**2 for a in range(3)]
    else:                                   # 2nd-order centered differences (skew-adjoint, collocated)
        D = lambda u: [(np.roll(u, -1, a) - np.roll(u, 1, a)) / (2 * h) for a in range(3)]
        sym = [(np.sin(TWO_PI * np.meshgrid(mm, mm, mm, indexing="ij")[a] * h) / h)**2 for a in range(3)]
    lap = sym[0] + sym[1] + sym[2]; T = np.where(lap > 1e-12, 1.0 / np.sqrt(np.where(lap > 1e-12, lap, 1.0)), 0.0)
    Top = lambda r: np.fft.ifftn(T * np.fft.fftn(r)).real          # (-lap_h)^(-1/2), kills the null modes of D
    div = lambda v: D(v[0])[0] + D(v[1])[1] + D(v[2])[2]
    cross = lambda a, b: [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    dot = lambda a, b: a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    def state(y):
        u1 = Top(y[:n3].reshape(f.shape)); u2 = Top(y[n3:].reshape(f.shape))
        d1 = D(u1); d2 = D(u2); g1 = [d1[0], 1.0 + d1[1], d1[2]]; g2 = [d2[0], d2[1], 1.0 + d2[2]]
        return g1, g2, cross(g1, g2)
    def fun(y):
        g1, g2, c = state(y); w = [ci / k for ci in c]
        E = 0.5 * np.mean(dot(c, w))
        G1 = -div(cross(g2, w)) / n3; G2 = -div(cross(w, g1)) / n3
        return E, np.concatenate([Top(G1).ravel(), Top(G2).ravel()])
    t0 = time.time(); y0 = np.zeros(2 * n3); E0, gr0 = fun(y0)
    res = minimize(fun, y0, jac=True, method="L-BFGS-B", options=dict(maxiter=maxit, maxfun=3 * maxit, maxcor=30, ftol=1e-30, gtol=1e-30))
    E, gr = fun(res.x); g1, g2, c = state(res.x)
    sols = [darcy_spectral(f, j) for j in range(3)]
    Keff = np.stack([s[3] for s in sols], axis=1); a = np.linalg.solve(Keff, np.array([1.0, 0.0, 0.0])); ED = 0.5 * a[0]
    Ds = lambda u: [np.fft.ifftn(1j * KO[i] * np.fft.fftn(u)).real for i in range(3)]
    gp = Ds(sum(a[j] * sols[j][0] for j in range(3))); vD = [-k * (gp[i] - a[i]) for i in range(3)]
    rms = lambda v: np.sqrt(np.mean(dot(v, v)))
    ev = rms([c[i] - vD[i] for i in range(3)]) / rms(vD)
    print(f"{name} eps={eps:5.3f} N={N:3d} disc={disc:3s} its={res.nit:5d} |gradE|/|gradE0|={np.linalg.norm(gr)/np.linalg.norm(gr0):.2e} "
          f"E_min={E:.10f} E_Darcy={ED:.10f} sqrt((E_min-E_D)/E_D)={np.sqrt(max(E-ED,0)/ED):.3e} e_v={ev:.3e} min|c|={np.sqrt(dot(c,c)).min():.3f} t={time.time()-t0:.0f}s", flush=True)

for spec in sys.argv[1:]:
    name, eps, N, disc = spec.split(":"); run(name, float(eps), int(N), disc)
