"""SF-29 shared case loader: reference Darcy flow, inlet labels and oracle labels on a vertex grid.

    load_case(field, eps, N) -> dict

Vertex grid of the slab 0 <= x1 <= 1 (periodic in x2, x3): x1 = j/N (j = 0..N, axis 0),
x2 = m2/N, x3 = m3/N (m = 0..N-1).  Keys of the returned dict:

  k, lnk                    (N+1, N, N)   analytic, at vertices
  grad_lnk                  3 x (N+1, N, N) analytic (complex step), at vertices
  vD                        3 x (N+1, N, N) reference Darcy velocity -k grad(phi) at vertices
  vD_faces                  {'f1': (N+1, N, N), 'f2': (N, N, N), 'f3': (N, N, N)} face-averaged
                            normal Darcy flux on the cell faces of the vertex grid (Whitney/MAC):
                            f1[j, m2, m3]: x1-face in plane x1 = j/N, [m2, m2+1]/N x [m3, m3+1]/N;
                            f2[j, m2, m3]: x2-face in plane x2 = m2/N, [j, j+1]/N x [m3, m3+1]/N;
                            f3[j, m2, m3]: x3-face in plane x3 = m3/N, [j, j+1]/N x [m2, m2+1]/N.
                            Gauss-Legendre NG x NG quadrature (NG = 3, error O(h^6)).
  psi0                      2 x (N, N)     inlet labels at the inlet-face vertices
  psi_or                    2 x (N+1, N, N) oracle labels at every vertex (psi_or[i][0] == psi0[i])
  vperp_in                  2 x (N, N)     tangential Darcy velocity (v2, v3) at the inlet-face vertices
  meta                      dict: field, eps, N, nphi, nf, L, a, pcg, max_d1phi, Q0, roundtrip/nfev per
                            plane, timings, cache file, ...
  ref, inlet                the live DarcyReference / InletLabels objects (not cached; rebuilt on
                            load, cost of one reference solve)

Expensive arrays (vD, vD_faces, psi0, psi_or, vperp_in) are cached in
../raw/cache/<field>_eps<eps>_N<N>_nphi<nphi>_v<CACHE_VERSION>.npz; k, lnk, grad_lnk are recomputed
(analytic, cheap).
"""
import os
import time

import numpy as np

import reference as R
import inlet as I
import tracing as T

CACHE_VERSION = 1
NG = 3
_HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.normpath(os.path.join(_HERE, "..", "raw", "cache"))

# Reference resolution N_phi(field, eps), chosen INDEPENDENTLY of the candidate grid N and shown
# resolution-independent by `oracle.py --nphi` (inlet and outlet labels at N_phi vs 1.5 N_phi differ
# by < 1e-8 relative to the RMS of their periodic part).  Measured values: raw/oracle_nphi*.txt.
# For `_ch` fields N_phi is the transverse resolution; the length-2 cell uses 2 N_phi in x1.
NPHI = {
    ("control2d", 0.25): 24, ("control2d", 0.5): 32, ("control2d", 1.0): 32,
    ("lester2021", 0.25): 48, ("lester2021", 0.5): 48, ("lester2021", 1.0): 48,
    ("lester_brk", 0.25): 48, ("lester_brk", 0.5): 48, ("lester_brk", 1.0): 64,
    ("two_mode", 0.25): 16, ("two_mode", 0.5): 24, ("two_mode", 1.0): 24,
    ("generic3d", 0.25): 24, ("generic3d", 0.5): 32, ("generic3d", 1.0): 32,
    ("gauss", 0.25): 32, ("gauss", 0.5): 32, ("gauss", 1.0): 48,
    ("gauss_ch", 0.25): 48, ("gauss_ch", 0.5): 48, ("gauss_ch", 1.0): 64,
}
NPHI_LADDER = (16, 24, 32, 48, 64)
NPHI_TOL = 1e-8


def nf_for(nphi):
    """Inlet-face sampling resolution for the trigonometric inlet labels."""
    return 2 * int(nphi)


def nphi_for(field, eps):
    key = (field, float(eps))
    if field == "uniform":
        return 8
    if key not in NPHI:
        raise KeyError("no reference resolution N_phi recorded for %s eps=%g; run `oracle.py --nphi %s:%g` "
                       "and add it to cases.NPHI (or pass nphi=...)" % (field, eps, field, eps))
    return NPHI[key]


def parse_spec(spec):
    """CLI case spec 'field:eps:N' (N optional) -> (field, eps, N or None)."""
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError("case spec must be field:eps[:N], got %r" % spec)
    return parts[0], float(parts[1]), (int(parts[2]) if len(parts) == 3 else None)


def vertex_coords(N):
    x = np.arange(N) / float(N)
    Y, Z = np.meshgrid(x, x, indexing="ij")
    return x, Y.ravel(), Z.ravel()


def build_reference(field, eps, nphi=None, verbose=True):
    nphi = nphi_for(field, eps) if nphi is None else int(nphi)
    ref = R.DarcyReference(field, eps, nphi, verbose=verbose)
    ref.check_positive_v1()
    inl = I.InletLabels(ref, nf_for(nphi))
    return ref, inl


def compute_oracle(ref, inl, N, verbose=True):
    """Oracle labels at all vertices: backward trace of every plane j >= 1 to the inlet."""
    _, Y, Z = vertex_coords(N)
    psi1 = np.empty((N + 1, N, N)); psi2 = np.empty((N + 1, N, N))
    p1, p2 = inl.labels(Y, Z)
    psi1[0] = p1.reshape(N, N); psi2[0] = p2.reshape(N, N)
    rts = np.zeros(N + 1); nfev = np.zeros(N + 1, dtype=int); nfev_rt = np.zeros(N + 1, dtype=int)
    tpl = np.zeros(N + 1)
    for j in range(1, N + 1):
        t0 = time.time()
        fy, fz, info = T.trace_to_inlet(ref, j / float(N), Y, Z, roundtrip=True)
        p1, p2 = inl.labels(fy, fz)
        psi1[j] = p1.reshape(N, N); psi2[j] = p2.reshape(N, N)
        rts[j] = info["roundtrip"]; nfev[j] = info["nfev"]; nfev_rt[j] = info["nfev_rt"]
        tpl[j] = time.time() - t0
        if verbose:
            print("  plane j=%3d x1=%.4f nfev=%5d nfev_rt=%5d roundtrip=%.1e t=%.2fs"
                  % (j, j / float(N), nfev[j], nfev_rt[j], rts[j], tpl[j]), flush=True)
    return (psi1, psi2), {"roundtrip": rts, "nfev": nfev, "nfev_rt": nfev_rt, "t_plane": tpl}


def vertex_velocity(ref, N):
    _, Y, Z = vertex_coords(N)
    v = np.empty((3, N + 1, N, N))
    for j in range(N + 1):
        vv = ref.velocity(j / float(N), Y, Z)
        for i in range(3):
            v[i, j] = vv[:, i].reshape(N, N)
    return v


def face_fluxes(ref, N, ng=NG):
    """Face-averaged normal Darcy flux on the cell faces of the vertex grid (GL ng x ng)."""
    xg, wg = np.polynomial.legendre.leggauss(ng)
    xg = 0.5 * (xg + 1.0); wg = 0.5 * wg
    h = 1.0 / N
    base = np.arange(N) * h
    # x1-faces: points (base_m2 + h xg_a, base_m3 + h xg_b)
    yy = (base[:, None] + h * xg[None, :])               # (N, ng)
    Yq = np.broadcast_to(yy[:, :, None, None], (N, ng, N, ng))
    Zq = np.broadcast_to(yy[None, None, :, :], (N, ng, N, ng))
    W2 = (wg[:, None] * wg[None, :])                     # (ng a, ng b)
    f1 = np.empty((N + 1, N, N))
    for j in range(N + 1):
        v = ref.velocity(j * h, Yq.ravel(), Zq.ravel())[:, 0].reshape(N, ng, N, ng)
        f1[j] = np.einsum("aibj,ij->ab", v, W2)
    f2 = np.zeros((N, N, N)); f3 = np.zeros((N, N, N))
    grid = np.arange(N) * h
    # x2-faces: x2 = m2 h, x3 = base_m3 + h xg_b ; x3-faces: x3 = m3 h, x2 = base_m2 + h xg_a
    Y2 = np.broadcast_to(grid[:, None, None], (N, N, ng)); Z2 = np.broadcast_to(yy[None, :, :], (N, N, ng))
    Y3 = np.broadcast_to(yy[:, :, None], (N, ng, N)); Z3 = np.broadcast_to(grid[None, None, :], (N, ng, N))
    for j in range(N):
        for a in range(ng):
            x1 = (j + xg[a]) * h
            v2 = ref.velocity(x1, Y2.ravel(), Z2.ravel())[:, 1].reshape(N, N, ng)
            f2[j] += wg[a] * (v2 @ wg)
            v3 = ref.velocity(x1, Y3.ravel(), Z3.ravel())[:, 2].reshape(N, ng, N)
            f3[j] += wg[a] * np.einsum("aib,i->ab", v3, wg)
    return {"f1": f1, "f2": f2, "f3": f3}


def face_divergence(fl, N):
    """Discrete divergence of face fluxes per cell (flux balance / cell volume)."""
    h = 1.0 / N
    f1, f2, f3 = fl["f1"], fl["f2"], fl["f3"]
    return ((f1[1:] - f1[:-1]) + (np.roll(f2, -1, axis=1) - f2) + (np.roll(f3, -1, axis=2) - f3)) / h


def cache_path(field, eps, N, nphi):
    return os.path.join(CACHE_DIR, "%s_eps%g_N%d_nphi%d_v%d.npz" % (field, eps, N, nphi, CACHE_VERSION))


def load_case(field, eps, N, nphi=None, use_cache=True, write_cache=True, verbose=True):
    t0 = time.time()
    nphi = nphi_for(field, eps) if nphi is None else int(nphi)
    ref, inl = build_reference(field, eps, nphi, verbose=verbose)
    path = cache_path(field, eps, N, nphi)
    x, Y, Z = vertex_coords(N)
    X1v = np.broadcast_to((np.arange(N + 1) / float(N))[:, None, None], (N + 1, N, N))
    X2v = np.broadcast_to(x[None, :, None], (N + 1, N, N))
    X3v = np.broadcast_to(x[None, None, :], (N + 1, N, N))
    case = {"lnk": ref.lk.lnk(X1v, X2v, X3v)}
    case["k"] = np.exp(case["lnk"])
    case["grad_lnk"] = tuple(ref.lk.grad_lnk(X1v, X2v, X3v))
    meta = {"field": field, "eps": float(eps), "N": int(N), "nphi": nphi, "nf": inl.nf, "L": ref.L,
            "a": ref.a.tolist(), "pcg_res": float(ref.pcg[0]), "pcg_its": int(ref.pcg[1]),
            "max_d1phi": ref.max_d1phi, "Q0": inl.Q0, "inlet_vmin": inl.vmin, "t_ref": ref.t_solve,
            "cache": path, "cache_version": CACHE_VERSION}
    hit = False
    if use_cache and os.path.exists(path):
        d = np.load(path)
        if int(d["cache_version"]) == CACHE_VERSION and int(d["nphi"]) == nphi:
            hit = True
            case["vD"] = tuple(d["vD"]); case["psi_or"] = (d["psi1"], d["psi2"])
            case["vD_faces"] = {"f1": d["f1"], "f2": d["f2"], "f3": d["f3"]}
            case["psi0"] = (d["psi1"][0].copy(), d["psi2"][0].copy())
            case["vperp_in"] = (d["vperp2"], d["vperp3"])
            meta["roundtrip"] = d["roundtrip"]; meta["nfev"] = d["nfev"]; meta["nfev_rt"] = d["nfev_rt"]
            meta["t_oracle"] = float(d["t_oracle"]); meta["t_faces"] = float(d["t_faces"])
    if not hit:
        t1 = time.time()
        psi_or, info = compute_oracle(ref, inl, N, verbose=verbose)
        meta["t_oracle"] = time.time() - t1
        vD = vertex_velocity(ref, N)
        t2 = time.time()
        fl = face_fluxes(ref, N)
        meta["t_faces"] = time.time() - t2
        case["vD"] = (vD[0], vD[1], vD[2]); case["psi_or"] = psi_or
        case["vD_faces"] = fl
        case["psi0"] = (psi_or[0][0].copy(), psi_or[1][0].copy())
        case["vperp_in"] = (vD[1, 0].copy(), vD[2, 0].copy())
        meta.update(info)
        if write_cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.savez(path, vD=vD, psi1=psi_or[0], psi2=psi_or[1], f1=fl["f1"], f2=fl["f2"], f3=fl["f3"],
                     vperp2=vD[1, 0], vperp3=vD[2, 0], roundtrip=info["roundtrip"], nfev=info["nfev"],
                     nfev_rt=info["nfev_rt"], t_oracle=meta["t_oracle"], t_faces=meta["t_faces"],
                     nphi=nphi, cache_version=CACHE_VERSION, field=field, eps=float(eps), N=int(N))
    meta["cache_hit"] = hit
    meta["t_total"] = time.time() - t0
    case["meta"] = meta
    case["ref"] = ref
    case["inlet"] = inl
    return case
