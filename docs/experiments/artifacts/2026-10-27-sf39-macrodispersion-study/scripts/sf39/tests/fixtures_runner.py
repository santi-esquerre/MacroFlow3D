"""Fake `lagrangian_ensemble` output trees with analytically designed particles (test fixtures; seconds, Np = 100).

Local coordinates of particle i at time t: `s = (U0 t + c1_i sqrt t, c2_i sqrt t, c3_i sqrt t)`, position
`x = x0 e1 + Rz(theta) s` (theta = 0: x1 = x0 + s1).  Weighted variances of any fixed set are then exactly linear in t, so the
window alpha has a closed form (`Var_w(c)/(2 ell U)` per direction).  Plane records: `x23 = (c2, c3) sqrt(X - x0)`, so
`sigma_T^2(X) = (X - x0) sigma_c^2` and `alpha_TP = sigma_c^2/(2 ell)`.

Geometry: ell = 0.5, L1 = 20, x0 = 2.5 (= 5 ell), L = 15: X_a = 10, X_b = 17.5, X_q = 6.25; times 0.25 n (n = 1..64).
"""
import json
import os

import numpy as np

ELL, L1, M, U0 = 0.5, 20.0, 10, 0.97
X0 = 5.0 * ELL
TIMES = 0.25 * np.arange(1, 65)
XA, XB, XQ = X0 + (L1 - 10 * ELL) / 2.0, L1 - 5 * ELL, X0 + (L1 - 10 * ELL) / 4.0
PLANES = np.array([3.0, 4.5, XQ, 8.0, XA, 13.0, 15.5, XB])
PLANES_NO_KEY = np.array([3.0, 4.5, 8.0, 13.0, 15.5, XB])  # no X_a, no X_q (SF-41-like auto planes)
IA, IB, IQ = 4, 7, 2
NP = M * M
BF = [7, 53, 91]  # seed backflow (status 1, weight 0)
OUT_IDS, OUT_T = [3, 20, 44, 66], [2.1, 4.9, 5.3, 8.0]  # outlet exits (exit time, off-grid)
IN_IDS, IN_T = [11, 30, 77, 13], [1.3, 4.2, 5.55, 0.1]  # id 13 exits before the first snapshot
STAG = {55: 3.3, 56: 7.0}  # non-exit failures: NaN records from that time on (status 3, 2)
PLANE_REACH = {5: 3, 18: 5, 40: 7, 63: 7, 80: 6}  # planes reached (prefix); others reach all 8
POL_EXIT = {9: 3.0, 21: 5.5}  # Pollock exits (NaN records after)
POL_REACH = {5: 8, 22: 4, 40: 8}
DISP = {8: (0.0, 0.0), 16: (0.02, 0.01), 32: (0.05, -0.03)}  # pass displacement (const dx2, per-particle dx3 scale)


def gen(r, theta=0.0):
    """Generating arrays of realization r."""
    rs = np.random.RandomState(1000 + r)
    n1, n2, n3, n4, n5 = (rs.standard_normal(NP) for _ in range(5))
    c1 = 0.30 * n1
    c2 = 0.50 * n2 + 0.2 * c1
    c3 = 0.40 * n3 + 0.1 * c1
    w = 1.0 + 0.5 * rs.uniform(size=NP)
    w[BF] = 0.0
    w /= w.sum()
    pc2, pc3 = c2 + 0.03 * n4, c3 + 0.03 * n5
    pw = 1.0 + 0.5 * rs.uniform(size=NP)
    pw[BF] = 0.0
    pw /= pw.sum()
    t_end = np.full(NP, np.inf)  # time from which the consumer record is NaN
    status_time = np.zeros(NP, dtype=np.uint8)
    status_time[BF] = 1
    t_end[BF] = 0.0
    for ids, ts, code in ((OUT_IDS, OUT_T, 8), (IN_IDS, IN_T, 7)):
        for i, t in zip(ids, ts):
            t_end[i], status_time[i] = t, code
    for i, t in STAG.items():
        t_end[i], status_time[i] = t, 3 if i == 55 else 2
    p_end = np.full(NP, np.inf)
    for i, t in POL_EXIT.items():
        p_end[i] = t
    reach = np.full(NP, len(PLANES), dtype=int)
    for i, k in PLANE_REACH.items():
        reach[i] = k
    status_plane = np.zeros(NP, dtype=np.uint8)
    status_plane[BF] = 1
    reach[BF] = 0
    for i in PLANE_REACH:
        status_plane[i] = 8
    status_plane[25] = 3  # finite at every plane but not ok, and the fastest (t_b filter)
    return dict(r=r, theta=theta, c1=c1, c2=c2, c3=c3, w=w, pc2=pc2, pc3=pc3, pw=pw, t_end=t_end, p_end=p_end,
                status_time=status_time, reach=reach, status_plane=status_plane, qbar=0.97 + 0.01 * r, planes=PLANES)


def local_to_x(g, t, c1, c2, c3):
    th = g["theta"]
    s1, s2, s3 = U0 * t + c1 * np.sqrt(t), c2 * np.sqrt(t), c3 * np.sqrt(t)
    return np.stack([X0 + np.cos(th) * s1 - np.sin(th) * s2, np.sin(th) * s1 + np.cos(th) * s2, s3], axis=-1)


def times_x(g, pollock=False):
    c1, c2, c3 = (g["c1"], g["pc2"], g["pc3"]) if pollock else (g["c1"], g["c2"], g["c3"])
    end = g["p_end"] if pollock else g["t_end"]
    out = np.stack([local_to_x(g, t, c1, c2, c3) for t in TIMES], axis=1)  # (Np,T,3)
    out[end[:, None] <= TIMES[None, :], :] = np.nan
    return out


def times_I(g):
    k = (np.arange(NP) % 7) * 0.01
    out = k[:, None] * TIMES[None, :]
    out[np.isnan(times_x(g)[:, :, 0])] = np.nan
    return out


def planes_arrays(g, planes, shift=(0.0, 0.0), pollock=False):
    c2, c3 = (g["pc2"], g["pc3"]) if pollock else (g["c2"], g["c3"])
    reach = _pollock_reach(g) if pollock else g["reach"]
    sq = np.sqrt(planes - X0)
    x23 = np.stack([c2[:, None] * sq[None, :], c3[:, None] * sq[None, :]], axis=-1)
    x23[:, :, 0] += shift[0]
    x23[:, :, 1] += shift[1] * (np.arange(NP) % 3)[:, None]
    tau = (planes - X0)[None, :] / U0 * (1.0 + 0.05 * ((np.arange(NP) * 7) % 11) / 11.0)[:, None]
    tau[25] *= 0.5
    Ipl = 0.003 * (np.arange(NP) % 5)[:, None] * (planes - X0)[None, :]
    miss = np.arange(len(planes))[None, :] >= reach[:, None]
    x23[miss] = np.nan
    tau[miss] = np.nan
    Ipl[miss] = np.nan
    return x23, tau, Ipl


def _pollock_reach(g):
    reach = np.full(NP, len(PLANES), dtype=int)  # Pollock tracks the backflow seeds too
    for i, k in POL_REACH.items():
        reach[i] = k
    return reach


def _save(path, a):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.ascontiguousarray(a))


def _wjson(path, d):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f)


def _exits_csv(path, head, axis, inl, out):
    with open(path, "w") as f:
        f.write("%s,exit_inlet,exit_outlet\n" % head)
        for a, i, o in zip(axis, inl, out):
            f.write("%.15e,%d,%d\n" % (a, i, o))


def write_series(path, X):
    """Uniform-weight series of the particles with a record at each time (the runner's stats_twin format)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("# format_version=1\nt,mean_x,mean_y,mean_z,var_x,var_y,var_z,active\n")
        for j, t in enumerate(TIMES):
            a = np.isfinite(X[:, j, :]).all(axis=1)
            Y = X[a, j, :]
            f.write("%.17g,%s,%s,%d\n" % (t, ",".join("%.17g" % v for v in Y.mean(axis=0)),
                                         ",".join("%.17g" % v for v in Y.var(axis=0)), a.sum()))


def write_pass(root, g, planes, shift=(0.0, 0.0)):
    P = len(planes)
    tx = times_x(g)
    x23, tau, Ipl = _planes_subset(g, planes, shift)
    seeds = np.zeros((NP, 3))
    seeds[:, 0] = X0
    seeds[:, 1] = ((np.arange(NP) // M) + 0.5) / M
    seeds[:, 2] = ((np.arange(NP) % M) + 0.5) / M
    _save(root + "/seeds.npy", seeds)
    _save(root + "/weights.npy", g["w"])
    _save(root + "/status_plane.npy", g["status_plane"])
    _save(root + "/status_time.npy", g["status_time"])
    _save(root + "/planes_x23.npy", x23)
    _save(root + "/planes_tau.npy", tau)
    _save(root + "/planes_I.npy", Ipl)
    _save(root + "/times_x.npy", tx)
    _save(root + "/times_I.npy", times_I(g))
    st = g["t_end"]
    _exits_csv(root + "/exits_time.csv", "t", TIMES,
               [int(((st < t) & (g["status_time"] == 7)).sum()) for t in TIMES],
               [int(((st < t) & (g["status_time"] == 8)).sum()) for t in TIMES])
    _exits_csv(root + "/exits_planes.csv", "x1", planes, [0] * P, [0] * P)
    _wjson(root + "/track.json", {"planes": [float(x) for x in planes], "times": [float(t) for t in TIMES], "X0": X0, "M": M,
                                  "window": [0.1, 0.9, 0.1, 0.9], "Np": NP, "h": 1.0 / 256, "domain_L1": L1,
                                  "plane_pass": {"status_hist": []}, "time_pass": {"status_hist": []}})
    return tx


def _planes_subset(g, planes, shift):
    full = planes_arrays(g, PLANES, shift)
    keep = [int(np.argmin(np.abs(PLANES - x))) for x in planes]
    return tuple(a[:, keep] if a.ndim == 2 else a[:, keep, :] for a in full)


def write_pollock(root, g, planes=PLANES):
    keep = [int(np.argmin(np.abs(PLANES - x))) for x in planes]
    x23 = planes_arrays(g, PLANES, pollock=True)[0][:, keep, :]
    tx = times_x(g, pollock=True)
    st_t = np.where(np.isfinite(tx[:, -1, 0]), 0, 9).astype(np.uint8)
    _save(root + "/pollock_weights.npy", g["pw"])
    _save(root + "/pollock_status_plane.npy", np.zeros(NP, dtype=np.uint8))
    _save(root + "/pollock_status_time.npy", st_t)
    _save(root + "/pollock_planes_x23.npy", x23)
    _save(root + "/pollock_planes_t.npy", np.zeros((NP, len(planes))))
    _save(root + "/pollock_times_x.npy", tx)
    _wjson(root + "/pollock.json", {"comparator": "pollock", "Np": NP, "planes": [float(x) for x in planes],
                                    "times": [float(t) for t in TIMES]})


def build_run(out, *, multi=False, nr=2, divs=(8, 16, 32), pollock=True, key_planes=True, theta=0.0, done=True, twin_series=True):
    """Write `out/realizations/r_0000..` (+ stats_twin series); returns {'gens': [...], 'out': out}."""
    gens = []
    for r in range(nr):
        g = gen(r, theta)
        gens.append(g)
        R = "%s/realizations/r_%04d" % (out, r)
        planes = PLANES if key_planes else PLANES_NO_KEY
        if multi:
            for d in divs:
                tx = write_pass("%s/hmax_%d" % (R, d), g, planes, DISP[d])
        else:
            tx = write_pass(R, g, planes)
        _wjson(R + "/field.json", {"kind": "gaussian", "sigma2": 1.0, "ell": ELL, "seed": 5001 + r, "applied_scale": 0.9 + 0.01 * r,
                                   "raw_variance": 1.2 + 0.01 * r, "final_variance": 1.0})
        _wjson(R + "/flow_stats.json", {"qbar": g["qbar"], "h": 1.0 / 256, "L1": L1, "lambda": ELL,
                                        "x_faces_nonpositive": 0, "x_nonpositive_fraction": 0.0,
                                        "speed_over_qbar_percentiles": {"p0": 0.2, "p0.1": 0.25, "p1": 0.3, "p50": 0.9},
                                        "interior": {"x1_lo": X0, "x1_hi": L1 - X0,
                                                     "mean_u": g["qbar"] * np.cos(theta), "mean_v": g["qbar"] * np.sin(theta),
                                                     "mean_w": 0.0, "theta_2": 0.0, "theta_3": 0.0}})
        if pollock:
            write_pollock(R + "/pollock", g, planes)
        if twin_series:
            base = out + "/hmax_8" if multi else out
            write_series("%s/stats_twin/realization_%04d_timeseries.csv" % (base, r), tx)
        if done:
            open(R + "/DONE", "w").close()
    return {"gens": gens, "out": out}


def write_min_pass(root, planes, x23, weights, status_plane=None):
    """Minimal single-pass root (track.json + the three arrays the sigma series needs)."""
    n = len(weights)
    _save(root + "/weights.npy", weights)
    _save(root + "/status_plane.npy", np.zeros(n, dtype=np.uint8) if status_plane is None else status_plane)
    _save(root + "/planes_x23.npy", x23)
    _wjson(root + "/track.json", {"planes": [float(x) for x in planes], "times": [], "X0": X0, "M": 4})


def build_sign_campaign(evladder, counts):
    """24 minimal runs F_e{eps}_s{seed}_g{g}: coarse sigma_T^2 below the reference on the first counts[(eps,seed)] of 14 planes."""
    planes = np.linspace(3.0, 9.5, 14)
    n = 16
    base = np.random.RandomState(5).standard_normal((n, 2))
    w = np.full(n, 1.0 / n)
    for (eps, seed), neg in counts.items():
        for g in (32, 64, 128, 256):
            x23 = np.empty((n, 14, 2))
            for p in range(14):
                s = 1.0 + 0.1 * p  # reference sigma_T scale
                if g != 256:
                    f = {32: 4.0, 64: 2.0, 128: 1.0}[g]
                    s *= (1.0 - 0.05 / f) if p < neg else (1.0 + 0.05 / f)
                x23[:, p, :] = base * s
            write_min_pass("%s/F_e%d_s%d_g%d/realizations/r_0000" % (evladder, eps, seed, g), planes, x23, w)
