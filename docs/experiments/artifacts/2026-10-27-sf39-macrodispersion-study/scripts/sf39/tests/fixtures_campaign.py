"""Fake SF-39 campaign output tree for the checkpoint tests (tiny Np = 100, small pilot count; seconds).

Built from `fixtures_runner` (runner arms) and `fixtures_par2` (Par2 pilot).  Every runner arm is the base fixture with its consumer
transverse components c2, c3 scaled by a factor `tf` and its flow `qbar` scaled by `qf` (directory A uses tf = 1 x the amplitude factor): the estimators then have closed forms (`expected_*`), computed here by plain loops independent of `sf39.reduce`:
  window:  alpha_TAM = tf^2 (Var_w(c2) + Var_w(c3)) / 2 / (2 ell U)       (theta = 0: e_T1 = e2, e_T2 = e3; set S_r = records at j_hi)
  plane:   alpha_TP  = [sigma_T^2(X_b) - sigma_T^2(X_a)] / (2 ell (X_b - X_a)) from the plane records of the fixture generator.
"""
import json
import os
import shutil

import numpy as np

from sf39 import campaign as cp
from sf39.tests import fixtures_par2 as fp
from sf39.tests import fixtures_runner as fx

AMP_F = {"s1": 1.0, "s225": 1.2, "s4": 1.5}  # transverse scale of every arm of the amplitude
# (tf, qf) per arm role; the three narrow rungs in the order (1/2, 1, 2); A is the unscaled base
ROLE_F = {"tw": (0.9, 1.01), "n0": (0.8, 0.98), "n1": (1.0, 0.99), "n2": (1.1, 1.02)}
A_TOTAL0, P8, P16, P32 = 200.0, 30.0, 33.0, 36.0  # timings: A total_r = 200 + r; passes prepare+track
TB_R, TB_FACTOR = 2, 0.9  # realization of the earliest breakthrough in directory A
PAR2_STEP_S, FS_WALL, WALL_FACTOR = 10.0, 40.0, 1.25
PAR2_TB_STEPS = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220]  # t_b = step dt
PAR2_U = [0.5 + 0.01 * g for g in range(20)]


def arm_factors(arm, amp):
    """(tf, qf) of an arm."""
    a = cp.arm_ids(amp)
    role = "tw" if arm == a["tw"] else ("n%d" % list(a["narrow"]).index(arm) if arm in a["narrow"] else None)
    tf, qf = ROLE_F[role] if role else (1.0, 1.0)
    return tf * AMP_F[amp], qf


def _rewrite_flow(R, qf):
    p = R + "/flow_stats.json"
    with open(p) as f:
        d = json.load(f)
    d["qbar"] *= qf
    for k in ("mean_u", "mean_v", "mean_w"):
        d["interior"][k] *= qf
    with open(p, "w") as f:
        json.dump(d, f)


def _timings(R, total, passes):
    d = {"r": 0, "seconds": {"stage": 1.0, "prepare": 2.0, "track": 3.0, "bridge_repack": 0.1, "bridge_track": 0.2, "write": 0.3,
                              "total": total}}
    if passes:
        d["passes"] = [{"hmax_div": dv, "prepare": 0.4 * pt, "track": 0.6 * pt} for dv, pt in passes]
    with open(R + "/timings.json", "w") as f:
        json.dump(d, f)


def _single_total(arm, r):
    pilot = list(cp.all_pilot_arms())
    return 50.0 + 3.0 * (pilot.index(arm) if arm in pilot else 99) + 0.5 * r


def build_arm(out_root, arm, amp, nr):
    """`runs/<arm>` with nr DONE realizations: directory A multi-pass (unscaled base) or a scaled single-pass arm; timings.json."""
    out = cp.arm_dir(out_root, arm)
    tf, qf = arm_factors(arm, amp)
    multi = cp.is_multipass(arm)
    # directory A also scales amplitude: the three passes share the (scaled) consumer transverse components
    fx.build_run(out, multi=multi, nr=nr, pollock=True, twin_series=multi)
    for r in range(nr):
        R = "%s/realizations/r_%04d" % (out, r)
        g = fx.gen(r)
        g = dict(g, c2=g["c2"] * tf, c3=g["c3"] * tf)
        if multi:
            for d in cp.HMAX_DIVS:
                fx.write_pass("%s/hmax_%d" % (R, d), g, fx.PLANES, fx.DISP[d])
                if r == TB_R:  # the earliest arrival of the pilot: t_b,min = TB_FACTOR x the common t_b
                    pt = "%s/hmax_%d/planes_tau.npy" % (R, d)
                    np.save(pt, np.load(pt) * TB_FACTOR)
            fx.write_series("%s/hmax_8/stats_twin/realization_%04d_timeseries.csv" % (out, r), fx.times_x(g))
        else:
            fx.write_pass(R, g, fx.PLANES)
        _rewrite_flow(R, qf)
        if multi:
            _timings(R, A_TOTAL0 + r, [(8, P8), (16, P16), (32, P32)])
        else:
            _timings(R, _single_total(arm, r), None)


def build_par2(out_root, artifact, amp, n_pilot, utime0=1.0e9):
    """Par2 pilot of an amplitude: series/snapshots (fixtures_par2.make_tree), FIELDSTAT file, config; file times spaced PAR2_STEP_S."""
    c = cp.PAR2_ID[amp]
    tmp = os.path.join(out_root, "_tmp_par2_" + c)
    fp.make_tree(tmp, n_pilot, 0, PAR2_TB_STEPS[:n_pilot], lambda g: PAR2_U[g] * (1.0 + 0.1 * AMP_F[amp]))
    raw = os.path.join(out_root, "par2", "raw", "par2_s" + c)
    os.makedirs(os.path.dirname(raw), exist_ok=True)
    shutil.move(os.path.join(tmp, "raw", "par2_s1"), raw)
    shutil.rmtree(tmp)
    for r in range(n_pilot):
        t = utime0 + PAR2_STEP_S * r
        os.utime(os.path.join(raw, "stats", "realization_%04d_timeseries.csv" % r), (t, t))
    with open(os.path.join(out_root, "par2", "field_stats_par2_s%s.txt" % c), "w") as f:
        for r in range(n_pilot):
            f.write("FIELDSTAT r=%d seed=%d n=1024 mean_lnK=1.0e-03 var_lnK=%.6e Kmin=0.1 Kmax=9.0 Kmean=1.2\n" % (r, 9001 + r, 1.0 + 0.01 * r))
    os.makedirs(os.path.join(artifact, "configs"), exist_ok=True)
    with open(os.path.join(artifact, "configs", "par2_s%s.yaml" % c), "w") as f:
        f.write(fp.SYN_CFG)


def build_campaign(out_root, artifact, n_pilot, real_scripts):
    """Complete fake pilot of all 15 runner arms, three Par2 pilots, walls.tsv; the artifact gets configs/ and scripts/nr_rule.py."""
    for amp in cp.AMPS:
        for arm in cp.pilot_arms(amp):
            build_arm(out_root, arm, amp, n_pilot)
        build_par2(out_root, artifact, amp, n_pilot)
    os.makedirs(os.path.join(artifact, "scripts"), exist_ok=True)
    shutil.copy(os.path.join(real_scripts, "nr_rule.py"), os.path.join(artifact, "scripts", "nr_rule.py"))
    write_walls(out_root, n_pilot)


def arm_total_s(arm, n_pilot):
    """Sum of timings totals of an arm's pilot realizations (known by construction)."""
    if cp.is_multipass(arm):
        return sum(A_TOTAL0 + r for r in range(n_pilot))
    return sum(_single_total(arm, r) for r in range(n_pilot))


def write_walls(out_root, n_pilot, skip=()):
    """pilot/walls.tsv: runner wall = WALL_FACTOR x the arm's timings total (ratio known), Par2 walls fixed; a repeated id (last wins)."""
    os.makedirs(os.path.join(out_root, "pilot"), exist_ok=True)
    with open(os.path.join(out_root, "pilot", "walls.tsv"), "w") as f:
        for arm in cp.all_pilot_arms():
            if arm in skip:
                continue
            f.write("%s\t%d\t2026-10-28T00:00:00Z\n" % (cp.pilot_walls_id(arm, n_pilot), 7))  # superseded record
            f.write("%s\t%.6f\t2026-10-28T00:00:01Z\n" % (cp.pilot_walls_id(arm, n_pilot), WALL_FACTOR * arm_total_s(arm, n_pilot)))
        for amp in cp.AMPS:
            c = cp.PAR2_ID[amp]
            f.write("par2_s%s\t%d\t2026-10-28T00:00:02Z\n" % (c, 900))
            f.write("fs_par2_s%s\t%d\t2026-10-28T00:00:03Z\n" % (c, FS_WALL * n_pilot))


# ------------------------------------------------------------------------------------------------
# independent closed forms (plain loops)
# ------------------------------------------------------------------------------------------------
def _wvar(vals, ws):
    W = sum(ws)
    m = sum(w * v for v, w in zip(vals, ws)) / W
    return sum(w * (v - m) ** 2 for v, w in zip(vals, ws)) / W


def expected_alpha_TAM(r, tf, U, t_hi):
    """alpha_TAM of realization r of a (tf, U) arm for a window ending at t_hi (the set is the records at the first snapshot >= t_hi)."""
    g = fx.gen(r)
    j = next(k for k, t in enumerate(fx.TIMES) if t >= t_hi)
    idx = [i for i in range(fx.NP) if g["t_end"][i] > fx.TIMES[j]]
    w = [float(g["w"][i]) for i in idx]
    v2 = _wvar([tf * float(g["c2"][i]) for i in idx], w)
    v3 = _wvar([tf * float(g["c3"][i]) for i in idx], w)
    return (v2 + v3) / 2.0 / (2.0 * fx.ELL * U)


def expected_alpha_TP(r, tf, shift=(0.0, 0.0)):
    """alpha_TP of realization r, scale tf, plane displacement `shift` (the fixture's pass displacement)."""
    g = fx.gen(r)
    g = dict(g, c2=g["c2"] * tf, c3=g["c3"] * tf)
    x23 = fx.planes_arrays(g, fx.PLANES, shift)[0]
    idx = [i for i in range(fx.NP) if np.isfinite(x23[i, fx.IA, 0]) and np.isfinite(x23[i, fx.IB, 0])]
    w = [float(g["w"][i]) for i in idx]

    def sig(ip):
        return (_wvar([float(x23[i, ip, 0]) for i in idx], w) + _wvar([float(x23[i, ip, 1]) for i in idx], w)) / 2.0

    return (sig(fx.IB) - sig(fx.IA)) / (2.0 * fx.ELL * (fx.XB - fx.XA))


def tree_bytes(path):
    return sum(os.path.getsize(os.path.join(a, f)) for a, _, fs in os.walk(path) for f in fs)


