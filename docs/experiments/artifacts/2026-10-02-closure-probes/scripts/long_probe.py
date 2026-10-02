"""Iterate the return map: transverse displacement after n periods (same machinery as closure_probe)."""
import sys, numpy as np
from scipy.integrate import solve_ivp
from closure_probe import FIELDS, grid, darcy_spectral, TrigField
def run(name, eps, N, nper, npts=16, seed=3):
    X, Y, Z = grid(N); f = eps * FIELDS[name](X, Y, Z)
    sols = [darcy_spectral(f, j) for j in range(3)]
    Keff = np.stack([s[3] for s in sols], axis=1); a = np.linalg.solve(Keff, np.array([1.0, 0.0, 0.0]))
    fld = TrigField(sum(a[j] * sols[j][0] for j in range(3)), -a)
    rng = np.random.default_rng(seed); y0 = rng.random(npts); z0 = rng.random(npts); P = npts
    def rhs(x1, s):
        x = np.empty((P, 3)); x[:, 0] = x1; x[:, 1] = s[:P]; x[:, 2] = s[P:]
        g = fld.grad(x); return np.concatenate([g[:, 1] / g[:, 0], g[:, 2] / g[:, 0]])
    te = [n for n in (1, 2, 5, 10, 20, 40, 80) if n <= nper]
    sol = solve_ivp(rhs, (0.0, float(nper)), np.concatenate([y0, z0]), method="DOP853", rtol=1e-11, atol=1e-13, t_eval=te)
    print(f"{name} eps={eps} N={N}: transverse displacement after n periods (mean flux exactly along x1)")
    for n, s in zip(te, sol.y.T):
        dy = s[:P] - y0; dz = s[P:] - z0; d = np.hypot(dy, dz)
        print(f"  n={n:3d}  rms|d|={np.sqrt((d**2).mean()):.4f}  max|d|={d.max():.4f}  var(dy)={dy.var():.3e} var(dz)={dz.var():.3e}", flush=True)
for spec in sys.argv[1:]:
    name, eps, N, nper = spec.split(":"); run(name, float(eps), int(N), int(nper))
