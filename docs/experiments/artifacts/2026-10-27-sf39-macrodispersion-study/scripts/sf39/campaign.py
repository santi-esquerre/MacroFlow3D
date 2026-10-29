"""SF-39 campaign helpers (contract C-GLUE, shared by `sf39.checkpoint` and the analysis part).

What is here: the arm/amplitude tables (taken from `sf39.geometry` / `sf39.power`), discovery of the complete (DONE) realizations of an
arm under `$OUT/runs/<arm>/`, the pass-root and series-path rules, the velocity-`U` rules of preregistration section 3, per-arm
per-realization reduction loops (window and plane estimators through `sf39.reduce`), pairing by seed, the pilot breakthrough times
and the Gate 5 frozen times (section 4 C).  No statistics beyond means of `qbar`; no decisions.  numpy + stdlib only.

Layout read (README "Outputs layout"; `$OUT` = `SF39_OUT_ROOT`):
  runner arms   `$OUT/runs/<arm>/realizations/r_%04d/` (single pass) or `.../r_%04d/hmax_<d>/` (directory A, `s{c}_A`, d in 8,16,32);
                run-level uniform-weight series `runs/<arm>/stats_twin/` (single pass) or `runs/s{c}_A/hmax_8/stats_twin/`
  Par2          `$OUT/par2/raw/par2_s{025,1,225,4}/`, configs `<A>/configs/par2_s{c}.yaml`, `$OUT/par2/field_stats_par2_s{c}.txt`
Amplitude ids are `s025`, `s1`, `s225`, `s4`; Par2 ids are `025`, `1`, `225`, `4`.  Seeds are `5001 + g` (production, narrow, twin).

U rules (section 3), in this module:
  standalone value of an arm        U = mean qbar_r over the realizations in that value                  (`u_standalone`)
  paired difference A - B           each side its own mean qbar_r over the seeds of the pair set         (`u_paired`)
  every pass of a realization and its Pollock bridge use the production-side U of the pair set (the first element of `u_paired`
  when the production arm is passed first), so that passes share their window exactly.
At the checkpoint every set is the pilot's (the common seeds of the arms).
"""
import os
from typing import NamedTuple

import numpy as np

from . import geometry
from . import io as sio
from . import par2
from . import power
from . import reduce as red
from . import stats

AMPS = power.AMPLITUDES                                   # ("s025", "s1", "s225", "s4")
PAR2_ID = {"s025": "025", "s1": "1", "s225": "225", "s4": "4"}
PILOT_SEED0 = 5001
PROD_DIV = 8                                              # production pass: h_max = h/8
HMAX_DIVS = (8, 16, 32)                                   # directory A passes
ESTIMATORS = ("alpha_TAM", "alpha_TP", "alpha_LAM")


# ------------------------------------------------------------------------------------------------
# arm tables
# ------------------------------------------------------------------------------------------------
def arm_ids(amp):
    """{'A','B','tw','narrow'} arm ids of an amplitude (preregistration section 1); narrow = rungs (1/2, 1, 2 x lambda/h)."""
    return {"A": amp + "_A", "B": amp + "_B", "tw": amp + "_tw", "narrow": power.NARROW_ARMS[amp]}


def pilot_arms(amp):
    """The pilot runner arms of an amplitude, in a fixed order: A, the three narrow rungs, the coarse twin."""
    a = arm_ids(amp)
    return (a["A"],) + tuple(a["narrow"]) + (a["tw"],)


def all_pilot_arms(amps=None):
    """The pilot runner arms (5 per amplitude) over `amps` (default: every amplitude of `AMPS`, 20 arms)."""
    return tuple(arm for amp in (AMPS if amps is None else amps) for arm in pilot_arms(amp))


def check_tables():
    """Every arm id used here exists in `geometry.ARMS` (raises AssertionError otherwise)."""
    for amp in AMPS:
        for arm in pilot_arms(amp) + (amp + "_B",):
            assert arm in geometry.ARMS, arm
    for arm in power.WIDTH_ARMS:
        assert arm in geometry.ARMS, arm


def cell_count(arm):
    """nx ny nz of an arm (geometry table)."""
    nx, ny, nz = geometry.ARMS[arm].grid
    return nx * ny * nz


def is_multipass(arm):
    """Directory A arms (`s{c}_A`) hold the three h_max passes; every other arm is single pass."""
    return arm in tuple(arm_ids(a)["A"] for a in AMPS)


def pilot_walls_id(arm, n_pilot):
    """Id of a runner arm in `pilot/walls.tsv` (run_sf39.sh: `<arm>:s<seed0>:n<nr>`)."""
    return "%s:s%d:n%d" % (arm, PILOT_SEED0, n_pilot)


# ------------------------------------------------------------------------------------------------
# realizations
# ------------------------------------------------------------------------------------------------
class Rec(NamedTuple):
    """A complete realization: arm, runner index r, field seed, realization directory, flow qbar."""
    arm: str
    r: int
    seed: int
    rdir: str
    qbar: float


def arm_dir(out_root, arm):
    """`$OUT/runs/<arm>`."""
    return os.path.join(out_root, "runs", arm)


def done_records(out_root, arm):
    """Realizations of `arm` with a DONE file, sorted by r ([] if the arm directory is absent); seed from field.json, qbar from
    flow_stats.json (the realization root, both passes share them)."""
    try:
        items = sio.list_realizations(arm_dir(out_root, arm))
    except sio.MissingData:
        return []
    recs = []
    for r, path, done in items:
        if done:
            seed = int(sio.read_json(os.path.join(path, "field.json"))["seed"])
            qbar = float(sio.read_json(os.path.join(path, "flow_stats.json"))["qbar"])
            recs.append(Rec(arm, r, seed, path, qbar))
    return recs


def pilot_records(out_root, arm, n_pilot):
    """The DONE realizations r = 0 .. n_pilot-1 of an arm; `MissingData` if any is absent or has the wrong seed."""
    recs = [x for x in done_records(out_root, arm) if x.r < n_pilot]
    if [x.r for x in recs] != list(range(n_pilot)):
        raise sio.MissingData("%s: realizations 0..%d are not all DONE" % (arm, n_pilot - 1))
    bad = [x.r for x in recs if x.seed != PILOT_SEED0 + x.r]
    if bad:
        raise sio.ReduceError("%s: seed != %d + r for r = %s" % (arm, PILOT_SEED0, bad))
    return recs


def production_records(out_root, amp, n_r=None):
    """Production realizations of an amplitude pooled over directory A and directory B by seed (`g = seed - 5001`); `n_r` keeps
    g < n_r (production estimate), None keeps all.  Each record keeps its own arm (pass roots differ: A multi-pass, B single)."""
    a = arm_ids(amp)
    recs = done_records(out_root, a["A"]) + done_records(out_root, a["B"])
    recs = [x for x in recs if n_r is None or x.seed - PILOT_SEED0 < n_r]
    seeds = [x.seed for x in recs]
    if len(set(seeds)) != len(seeds):
        raise sio.ReduceError("%s: a seed appears in both directory A and directory B" % amp)
    return sorted(recs, key=lambda x: x.seed)


def pass_root_of(rec, div=PROD_DIV):
    """Pass root of a realization: `R/hmax_<div>` in directory A, `R` for a single-pass arm (only the production pass exists there)."""
    if not is_multipass(rec.arm) and div != PROD_DIV:
        raise sio.ReduceError("%s is single pass: only hmax_div %d exists (asked %d)" % (rec.arm, PROD_DIV, div))
    return sio.pass_root(rec.rdir, div)


def realization(rec, div=PROD_DIV):
    """`sf39.io.Realization` of a record at a pass."""
    pass_root_of(rec, div)  # enforce the single-pass rule
    return sio.Realization(rec.rdir, div)


def twin_series_path(out_root, rec):
    """Uniform-weight run-level series of the production pass of a record's arm (`stats_twin/realization_%04d_timeseries.csv`)."""
    return sio.series_path(arm_dir(out_root, rec.arm), rec.r, PROD_DIV if is_multipass(rec.arm) else None, twin=True)


# ------------------------------------------------------------------------------------------------
# U rules
# ------------------------------------------------------------------------------------------------
def u_standalone(recs):
    """U of a standalone value: mean qbar_r over the realizations in that value."""
    if not recs:
        raise sio.ReduceError("no realizations")
    return float(np.mean([x.qbar for x in recs]))


def u_paired(recs_a, recs_b):
    """(U_a, U_b, seeds) of a paired difference: each side its own mean qbar over the common seeds.  Pass the production side first:
    its U is the one every pass of a realization and its Pollock bridge use."""
    seeds = sorted({x.seed for x in recs_a} & {x.seed for x in recs_b})
    if not seeds:
        raise sio.ReduceError("the two arms share no seed")
    ua = float(np.mean([x.qbar for x in recs_a if x.seed in seeds]))
    ub = float(np.mean([x.qbar for x in recs_b if x.seed in seeds]))
    return ua, ub, seeds


# ------------------------------------------------------------------------------------------------
# per-realization estimators
# ------------------------------------------------------------------------------------------------
def estimates(real, tau_b_star, U, mode="flux"):
    """{alpha_TAM, alpha_TP, alpha_LAM} of one realization/pass for a reducer method (default flux weights): window estimator on
    W = [0.75, 1] tau_b* ell / U (aligned transverse frame; fixed e1 for alpha_LAM) and plane estimator X_a -> X_b.  alpha_TP is NaN
    for a method without a plane estimator (`full`).  Raises WindowError/MissingData as the reducer does (nothing is clipped)."""
    win = red.window_method(real, mode, tau_b_star, U)
    out = {"alpha_TAM": win["win.alpha_TAM"], "alpha_LAM": win["win.alpha_e1"], "alpha_TP": float("nan")}
    if mode in red.TP_MODES:
        kp = real.key_planes()
        (xa, ia), (xb, ib) = kp["Xa"], kp["Xb"]
        out["alpha_TP"] = red.tp_value(real, mode, ia, ib, xa, xb)["tp.alpha_TP"]
    return out


def reduce_arm(recs, div, tau_b_star, U, mode="flux"):
    """{seed: estimates} over the records at one pass with one common U (loop over realizations; arrays are mmapped lazily)."""
    return {x.seed: estimates(realization(x, div), tau_b_star, U, mode) for x in recs}


def paired(est_a, est_b, key):
    """(seeds, a, b): `key` of two {seed: estimates} dicts restricted to their common seeds, ordered by seed (`stats.align_by_seed`)."""
    sa, sb = list(est_a), list(est_b)
    return stats.align_by_seed(sa, [est_a[s][key] for s in sa], sb, [est_b[s][key] for s in sb])


def breakthrough_times(recs):
    """t_b,r (box time) of each record, production pass: min planes_tau at X_b over status_plane == 0 (`reduce.breakthrough_time`)."""
    out = []
    for x in recs:
        real = realization(x, PROD_DIV)
        out.append(red.breakthrough_time(real.pass_, real.key_planes()["Xb"][1]))
    return out


# ------------------------------------------------------------------------------------------------
# Gate 5 frozen times (preregistration section 4 C)
# ------------------------------------------------------------------------------------------------
def twin_uc(out_root, recs, t_b_min):
    """(U_c, per-realization list, x0) of the Lagrangian uniform-weight series (production `stats_twin`) at t_b,min with anchor x0:
    U_c = mean_r [mean_x(t_b,min) - x0] / t_b,min, mean_x interpolated linearly (outside the series is an error, never clipped)."""
    x0s = {realization(x, PROD_DIV).x0 for x in recs}
    if len(x0s) != 1:
        raise sio.ReduceError("realizations of %s disagree on x0: %s" % (recs[0].arm, sorted(x0s)))
    x0 = x0s.pop()
    series = [par2.read_series(twin_series_path(out_root, x)) for x in recs]
    uc, per = par2.uc_mean(series, t_b_min, x0)
    return uc, per, x0


def par2_paths(out_root, artifact, amp):
    """(raw run directory, pilot config, FIELDSTAT file) of the Par2 pilot of an amplitude."""
    c = PAR2_ID[amp]
    return (os.path.join(out_root, "par2", "raw", "par2_s" + c), os.path.join(artifact, "configs", "par2_s%s.yaml" % c),
            os.path.join(out_root, "par2", "field_stats_par2_s%s.txt" % c))


def par2_frozen(out_root, artifact, amp, n_pilot):
    """Par2 pilot freeze of an amplitude: t_b,min over its first n_pilot realizations (last snapshot with no particle at
    x >= Lx - 5 lambda), U_c at t_b,min with anchor inject_x, tau_b*(Par2) = U_c t_b,min / corr_length (all through `sf39.par2`)."""
    raw, cfg, _ = par2_paths(out_root, artifact, amp)
    with open(cfg) as f:
        v = par2.cfg_values(f.read())
    entries = par2.pooled(raw)[:n_pilot]
    table = par2.t_b_table(entries, v["Lx"], v["lam"], v["dt"])
    tbm = par2.t_b_min(table, n_pilot)
    uc, _ = par2.uc_mean([par2.series_of(e) for e in entries], tbm, v["inject_x"])
    return {"t_b_min": float(tbm), "U_c": float(uc), "corr_length": float(v["lam"]), "inject_x": float(v["inject_x"]),
            "tau_b_star": float(par2.tau_b_star(uc, tbm, v["lam"]))}


def gate5_frozen(out_root, artifact, amp, recs, t_b_min, ell, n_pilot):
    """Frozen Gate 5 normalized times of an amplitude: twin = Lagrangian uniform-weight series (U_c, anchor x0,
    tau_b*(twin) = U_c t_b,min / ell), par2 = `par2_frozen`, tau_c = min of the two."""
    uc, _, x0 = twin_uc(out_root, recs, t_b_min)
    tw = {"t_b_min": float(t_b_min), "U_c": float(uc), "anchor_x0": float(x0), "ell": float(ell),
          "tau_b_star": float(uc * t_b_min / ell)}
    p2 = par2_frozen(out_root, artifact, amp, n_pilot)
    return {"twin": tw, "par2": p2, "tau_c": min(tw["tau_b_star"], p2["tau_b_star"])}
