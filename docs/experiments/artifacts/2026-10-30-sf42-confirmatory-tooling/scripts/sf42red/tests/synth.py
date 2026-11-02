"""Synthetic consumer pass directories and Par2 crossing CSVs with exactly known moments (test helpers).

Cloud on the M x M lattice (M even, p = m2*M + m3): the unit patterns Z2 = (-1)^m3, Z3 = (-1)^m2, Zt = (-1)^(m2+m3) have mean 0,
biased variance 1 and are pairwise uncorrelated, and uncorrelated with the seed coordinates s2 = (m2+.5)/M, s3 = (m3+.5)/M of the
same index (Z2 depends on m3 only, s2 on m2 only, ...), so
  displacement      = drift + g Z                 -> var = g^2 exactly, cov x2x3 = 0,
  position          = seed + displacement          -> var x2 = var(s2) + g^2 exactly,
  crossing time     = x1_offset/U0 + sqrt(c * x1_offset) Zt, with x1_offset = x1 - x0.
"""
import json
import os

import numpy as np

SEED_X1 = 0.0


def patterns(M):
    p = np.arange(M * M)
    m2, m3 = p // M, p % M
    return (-1.0) ** m3, (-1.0) ** m2, (-1.0) ** (m2 + m3), (m2 + 0.5) / M, (m3 + 0.5) / M


def write_pass(root, *, M=8, ell=0.25, x0=1.25, d_planes=(5, 10, 25, 40, 55, 70, 90, 100), a_T=0.03, c_tau=0.02, U0=2.0,
               drift=(0.0, 0.0), flux_weights=None, miss=None, times=None, a_L=0.05, a_Tt=0.04, status_exit=None, exit_after=None,
               r=0, hmax=None, seed=777, qbar=2.0, interior=(2.0, 0.0, 0.0), tracker=None):
    """Write a pass (`root/realizations/r_%04d[/hmax_<d>]`) and return the pass root.

    miss: dict {particle: first plane index NOT reached} (NaN from there on, status 7 for the time pass unless status_exit given);
    drift: displacement drift per unit (x1 - x0) in (x2, x3); times: snapshot times (default 1..12 * 0.5);
    status_exit/exit_after: {particle: status} / {particle: number of finite snapshots} for the time pass.
    tracker (SF-43): {"recrossings": n, "backflow_encounters": n, "seed_backflow": n} written as the tracker totals of `track.json`
    (`plane_pass`, `time_pass`, `seed_backflow` of `write_track_outputs`); default: not written (the SF-42 format).
    """
    rdir = os.path.join(root, "realizations", "r_%04d" % r)
    proot = os.path.join(rdir, "hmax_%d" % hmax) if hmax else rdir
    os.makedirs(proot, exist_ok=True)
    Np = M * M
    Z2, Z3, Zt, s2, s3 = patterns(M)
    d = np.asarray(d_planes, float)
    x1 = x0 + d * ell
    off = x1 - x0
    P = len(d)
    seeds = np.stack([np.full(Np, x0), s2, s3], axis=1)
    g = np.sqrt(2.0 * a_T * ell * off)  # var g^2 = 2 a_T ell (x1 - x0): alpha_TP = a_T exactly (units of lambda)
    planes_x23 = np.empty((Np, P, 2))
    for k in range(P):
        planes_x23[:, k, 0] = s2 + drift[0] * off[k] + g[k] * Z2
        planes_x23[:, k, 1] = s3 + drift[1] * off[k] + g[k] * Z3
    planes_tau = off[None, :] / U0 + np.sqrt(c_tau * off)[None, :] * Zt[:, None]
    planes_I = 0.1 * Zt[:, None] * np.ones((1, P)) + 0.05 * np.arange(P)[None, :]
    for pidx, k0 in (miss or {}).items():
        planes_x23[pidx, k0:, :] = np.nan
        planes_tau[pidx, k0:] = np.nan
        planes_I[pidx, k0:] = np.nan
    w = np.full(Np, 1.0 / Np) if flux_weights is None else np.asarray(flux_weights, float) / np.sum(flux_weights)
    times = np.arange(1, 13) * 0.5 if times is None else np.asarray(times, float)
    T = len(times)
    gt = np.sqrt(2.0 * a_Tt * ell * qbar * times)  # var = 2 a ell U t: alpha = a
    gl = np.sqrt(2.0 * a_L * ell * qbar * times)
    times_x = np.empty((Np, T, 3))
    times_x[:, :, 0] = x0 + qbar * times[None, :] + gl[None, :] * Zt[:, None]
    times_x[:, :, 1] = s2[:, None] + gt[None, :] * Z2[:, None]
    times_x[:, :, 2] = s3[:, None] + gt[None, :] * Z3[:, None]
    times_I = 0.01 * np.arange(T)[None, :] * np.ones((Np, 1))
    status_time = np.zeros(Np, np.uint8)
    for pidx, n in (exit_after or {}).items():
        times_x[pidx, n:, :] = np.nan
        times_I[pidx, n:] = np.nan
        status_time[pidx] = (status_exit or {}).get(pidx, 8)
    arrays = {"seeds": seeds, "weights": w, "status_plane": np.zeros(Np, np.uint8), "status_time": status_time,
              "planes_x23": planes_x23, "planes_tau": planes_tau, "planes_I": planes_I, "times_x": times_x, "times_I": times_I}
    for name, a in arrays.items():
        np.save(os.path.join(proot, name + ".npy"), a)
    with open(os.path.join(proot, "track.json"), "w") as f:
        tj = {"X0": x0, "M": M, "planes": x1.tolist(), "times": times.tolist(), "Np": Np}
        if tracker is not None:
            tot = {"status_hist": {"ok": Np}, "recrossings": tracker["recrossings"], "backflow_encounters": tracker["backflow_encounters"]}
            tj.update(seed_backflow=tracker["seed_backflow"], plane_pass=tot, time_pass=dict(tot, recrossings=0, backflow_encounters=0))
        json.dump(tj, f)
    with open(os.path.join(rdir, "field.json"), "w") as f:
        json.dump({"seed": seed, "ell": ell}, f)
    with open(os.path.join(rdir, "flow_stats.json"), "w") as f:
        json.dump({"lambda": ell, "qbar": qbar, "L1": 26.0, "interior": {"mean_u": interior[0], "mean_v": interior[1], "mean_w": interior[2]}}, f)
    with open(os.path.join(proot, "exits_time.csv"), "w") as f:
        f.write("t,exit_inlet,exit_outlet\n")
    with open(os.path.join(rdir, "DONE"), "w") as f:
        f.write("sig\n")
    return proot


PAR2_HEADER = "x1,n_injected,n_crossed,mean_t,var_t,mean_x2,mean_x3,var_x2,var_x3,cov_x2x3,mean_dx2,mean_dx3,var_dx2,var_dx3,cov_dx2dx3"
PAR2_FORMAT = "# format_version=1 sf42_plane_crossings weights=uniform time_origin=injection"


def write_par2(path, rows):
    """rows: list of 15-tuples/lists in header order (floats; nan allowed)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(PAR2_FORMAT + "\n" + PAR2_HEADER + "\n")
        for r in rows:
            f.write(",".join("nan" if v != v else "%.17g" % v for v in r) + "\n")


def write_pollock(rdir, proot, *, weights=None, planes_x23=None, planes_t=None, seeds_equal=True, planes=None, status_plane=None):
    """SF-43: the Pollock bridge directory `rdir/pollock/` in the format of `EnsembleRunner.cu` (pollock_{weights,status_plane,planes_x23,
    planes_t}.npy, pollock.json), by default a copy of the pass `proot` arrays (weights, planes_x23, planes_tau; Pollock clock = tau)."""
    d = os.path.join(rdir, "pollock")
    os.makedirs(d, exist_ok=True)
    A = lambda n: np.load(os.path.join(proot, n + ".npy"))  # noqa: E731
    w = A("weights") if weights is None else np.asarray(weights, float)
    x = A("planes_x23") if planes_x23 is None else planes_x23
    t = A("planes_tau") if planes_t is None else planes_t
    np.save(os.path.join(d, "pollock_weights.npy"), w)
    np.save(os.path.join(d, "pollock_planes_x23.npy"), x)
    np.save(os.path.join(d, "pollock_planes_t.npy"), t)
    np.save(os.path.join(d, "pollock_status_plane.npy"), np.zeros(len(w), np.uint8) if status_plane is None else status_plane)
    np.save(os.path.join(d, "pollock_status_time.npy"), np.zeros(len(w), np.uint8))
    np.save(os.path.join(d, "pollock_times_x.npy"), np.zeros((len(w), 2, 3)))
    np.save(os.path.join(d, "pollock_cells.npy"), np.zeros((len(w), 2), np.uint64))
    with open(os.path.join(proot, "track.json")) as f:
        tr = json.load(f)
    with open(os.path.join(d, "pollock.json"), "w") as f:
        json.dump({"comparator": "pollock", "weights": "rt0_v1_at_seed", "Np": len(w), "seeds_equal_tracker": seeds_equal,
                   "planes": tr["planes"] if planes is None else planes}, f)
    return d
