"""SF-29 N1 -- streamline tracing of the reference Darcy flow, parametrized by x1.

dx_perp/dx1 = grad_perp(phi) / d1(phi)  (k cancels; requires v1 > 0), vectorized over all
points that share the same starting x1 (as closure_probe.return_map), DOP853.
"""
import numpy as np
from scipy.integrate import solve_ivp

RTOL = 1e-12
ATOL = 1e-14


def _rhs_factory(ref, P):
    def rhs(x1, s):
        g = ref.grad_phi(x1, s[:P], s[P:])
        return np.concatenate([g[:, 1] / g[:, 0], g[:, 2] / g[:, 0]])
    return rhs


def trace(ref, x1_from, x1_to, y0, z0, rtol=RTOL, atol=ATOL):
    """Integrate the points (y0, z0) at x1_from to x1_to.  Returns y, z, nfev."""
    y0 = np.asarray(y0, float).ravel(); z0 = np.asarray(z0, float).ravel()
    P = y0.size
    if x1_from == x1_to:
        return y0.copy(), z0.copy(), 0
    sol = solve_ivp(_rhs_factory(ref, P), (float(x1_from), float(x1_to)), np.concatenate([y0, z0]),
                    method="DOP853", rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError("streamline integration failed: %s" % sol.message)
    s = sol.y[:, -1]
    return s[:P], s[P:], sol.nfev


def trace_to_inlet(ref, x1, y0, z0, roundtrip=True, rtol=RTOL, atol=ATOL):
    """Backward trace from plane x1 to the inlet x1 = 0.

    Returns (foot_y, foot_z, info) with info = {nfev, nfev_rt, roundtrip}; the round trip is the
    max-norm difference between the start points and the forward re-integration of the feet.
    """
    fy, fz, nfev = trace(ref, x1, 0.0, y0, z0, rtol, atol)
    info = {"nfev": nfev, "nfev_rt": 0, "roundtrip": 0.0}
    if roundtrip and x1 != 0.0:
        by, bz, nf2 = trace(ref, 0.0, x1, fy, fz, rtol, atol)
        info["nfev_rt"] = nf2
        info["roundtrip"] = float(max(np.abs(by - np.ravel(y0)).max(), np.abs(bz - np.ravel(z0)).max()))
    return fy, fz, info


def return_map_nonuniform(ref, npts=4, seed=3, rtol=RTOL, atol=ATOL):
    """Return map of the inlet face after the slab (x1: 0 -> 1) on the closure-note point set.

    Same starting points as closure_probe.run (default_rng(3), 16 points); returns the
    drift-independent `nonuniform_rms` (RMS of the displacement minus its mean), max |d|, round trip.
    """
    rng = np.random.default_rng(seed)
    y0 = rng.random(npts * npts); z0 = rng.random(npts * npts)
    y1, z1, nfev = trace(ref, 0.0, 1.0, y0, z0, rtol, atol)
    by, bz, _ = trace(ref, 1.0, 0.0, y1, z1, rtol, atol)
    rt = float(max(np.abs(by - y0).max(), np.abs(bz - z0).max()))
    dy, dz = y1 - y0, z1 - z0
    nonuni = float(np.sqrt(((dy - dy.mean())**2 + (dz - dz.mean())**2).mean()))
    return {"nonuniform_rms": nonuni, "dmax": float(np.hypot(dy, dz).max()),
            "mean": (float(dy.mean()), float(dz.mean())), "roundtrip": rt, "nfev": nfev}
