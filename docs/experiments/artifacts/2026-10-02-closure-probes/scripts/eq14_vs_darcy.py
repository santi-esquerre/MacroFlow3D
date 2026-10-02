"""Does a converged solution of the (same-index) equation (14) with affine + periodic
streamfunctions reproduce the Darcy flow?  Pseudo-spectral eq. (14) (Anderson, Laplacian
preconditioner; same residual as SF-26 probe_spectral.py 'sp') vs pseudo-spectral Darcy with
mean flux e1, on smooth analytic fields.  Reports r_F, e_v, invariance, and the decomposition
of R = k curl(c/k) into its part along c (NOT constrained by eq. 14) and across c (eq. 14)."""
import sys, time, numpy as np
from closure_probe import FIELDS, grid, wavenumbers, darcy_spectral, TWO_PI

def run(name, eps, N, iters=3000, m=10, omega=0.5, tol=1e-11):
    X, Y, Z = grid(N); f = eps * FIELDS[name](X, Y, Z); k = np.exp(f); q = 1.0 / k
    mm, md = wavenumbers(N)
    KO = np.meshgrid(TWO_PI * md, TWO_PI * md, TWO_PI * md, indexing="ij")
    KE = np.meshgrid(TWO_PI * mm, TWO_PI * mm, TWO_PI * mm, indexing="ij")
    k2 = KE[0]**2 + KE[1]**2 + KE[2]**2; k2inv = np.where(k2 > 0, 1.0 / np.where(k2 > 0, k2, 1.0), 0.0)
    D = lambda u: [np.fft.ifftn(1j * KO[a] * np.fft.fftn(u)).real for a in range(3)]
    def H(u):
        uh = np.fft.fftn(u); Hm = {}
        for a in range(3):
            Hm[(a, a)] = np.fft.ifftn(-KE[a] * KE[a] * uh).real
            for b in range(a + 1, 3):
                Hm[(a, b)] = Hm[(b, a)] = np.fft.ifftn(-KO[a] * KO[b] * uh).real
        return Hm
    cross = lambda a, b: [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]
    dot = lambda a, b: a[0]*b[0] + a[1]*b[1] + a[2]*b[2]
    curl = lambda w: [D(w[2])[1]-D(w[1])[2], D(w[0])[2]-D(w[2])[0], D(w[1])[0]-D(w[0])[1]]
    P = lambda w: w - w.mean()
    gf = D(f); g1b = [0.0, 1.0, 0.0]; g2b = [0.0, 0.0, 1.0]; qr = np.sqrt((q**2).mean())
    def fields(u1, u2):
        d1 = D(u1); d2 = D(u2); G1 = [g1b[i] + d1[i] for i in range(3)]; G2 = [g2b[i] + d2[i] for i in range(3)]
        return G1, G2, cross(G1, G2)
    def resid(u1, u2):
        G1, G2, c = fields(u1, u2); Ha = H(u1); Hb = H(u2)
        B = [sum(Hb[(i, j)] * G1[j] for j in range(3)) - sum(Ha[(i, j)] * G2[j] for j in range(3)) for i in range(3)]
        d = dot(c, c); S1 = dot(cross(B, G1), c) / d; S2 = dot(cross(B, G2), c) / d
        L1 = Ha[(0,0)] + Ha[(1,1)] + Ha[(2,2)] - dot(gf, G1); L2 = Hb[(0,0)] + Hb[(1,1)] + Hb[(2,2)] - dot(gf, G2)
        return P(-q * (L1 - S1)), P(-q * (L2 - S2))
    rF = lambda F1, F2: np.sqrt(((F1**2).mean() + (F2**2).mean()) / (2 * qr**2))
    lapinv = lambda r: np.fft.ifftn(-k2inv * np.fft.fftn(r)).real
    n3 = N**3; x = np.zeros(2 * n3); dX = []; dG = []; xp = fp = None; best = (np.inf, x.copy()); t0 = time.time()
    for it in range(iters):
        u1 = x[:n3].reshape(f.shape); u2 = x[n3:].reshape(f.shape)
        F1, F2 = resid(u1, u2); r = rF(F1, F2)
        if not np.isfinite(r): break
        if r < best[0]: best = (r, x.copy())
        if r < tol: break
        g = np.concatenate([P(u1 - omega * lapinv(-F1 / q)).ravel(), P(u2 - omega * lapinv(-F2 / q)).ravel()]); fx = g - x
        if xp is not None:
            dX.append(x - xp); dG.append(fx - fp)
            if len(dX) > m: dX.pop(0); dG.pop(0)
        xp = x.copy(); fp = fx.copy()
        if dG:
            DG = np.array(dG).T; gam = np.linalg.lstsq(DG, fx, rcond=None)[0]
            x = g - (np.array(dX).T + DG) @ gam
        else: x = g
    r, x = best; u1 = x[:n3].reshape(f.shape); u2 = x[n3:].reshape(f.shape)
    G1, G2, c = fields(u1, u2)
    # independent Darcy flow with mean flux e1
    sols = [darcy_spectral(f, j) for j in range(3)]
    Keff = np.stack([s[3] for s in sols], axis=1); a = np.linalg.solve(Keff, np.array([1.0, 0.0, 0.0]))
    ut = sum(a[j] * sols[j][0] for j in range(3)); gp = D(ut); vD = [-k * (gp[i] - a[i]) for i in range(3)]
    rms = lambda v: np.sqrt(np.mean(dot(v, v))) if isinstance(v, list) else np.sqrt(np.mean(v**2))
    ev = rms([c[i] - vD[i] for i in range(3)]) / rms(vD)
    inv1 = rms(dot(vD, G1)) / (rms(vD) * rms(G1)); inv2 = rms(dot(vD, G2)) / (rms(vD) * rms(G2))
    w = [c[i] / k for i in range(3)]; R = curl(w); cc = dot(c, c)
    Rpar = dot(R, c) / np.sqrt(cc)                              # along c: helicity-type, free in eq. (14)
    Rperp = [R[i] - dot(R, c) * c[i] / cc for i in range(3)]    # across c: what eq. (14) annihilates
    wD = [vD[i] / k for i in range(3)]
    print(f"{name:10s} eps={eps:5.3f} N={N:3d} its={it+1:4d} r_F={r:.2e} | e_v={ev:.3e} inv=({inv1:.2e},{inv2:.2e}) "
          f"min|c|={np.sqrt(cc).min():.2f} | curl(c/k): along_c={rms(Rpar)/rms(w):.3e} across_c={rms(Rperp)/rms(w):.2e} "
          f"[Darcy curl(v/k)={rms(curl(wD))/rms(wD):.1e}] meanflux_c=({c[0].mean():.6f},{c[1].mean():+.1e},{c[2].mean():+.1e}) t={time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    for spec in sys.argv[1:]:
        name, eps, N = spec.split(":"); run(name, float(eps), int(N))
