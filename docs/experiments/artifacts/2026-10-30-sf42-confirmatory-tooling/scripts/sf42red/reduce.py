"""SF-42 N5: reader and host-side per-realization reduction on control planes, both codes (decision G4).

Self-standing (stdlib + numpy; nothing is imported from the SF-39 package: the readers and the covariance/window logic
below are copies, adapted).  Runs unchanged on the host stack (Python 3.11 / numpy 1.26) and the local one.

INPUTS
  consumer  a pass root (directory with `track.json`) or an arm directory with `realizations/r_%04d/` (every DONE
            realization, every pass `R` or `R/hmax_<d>`).  Files read: `track.json` (planes, times, X0, M), `field.json`
            (seed, ell), `flow_stats.json` (lambda, qbar, interior mean velocity), the per-particle `.npy` of the pass
            (`seeds`, `weights`, `status_time`, `planes_x23`, `planes_tau`, `planes_I`, `times_x`, `times_I`).
  par2      `crossings/realization_%04d_planes.csv` (first line `# format_version=1 sf42_plane_crossings weights=uniform
            time_origin=injection`; header `x1,n_injected,n_crossed,mean_t,var_t,mean_x2,mean_x3,var_x2,var_x3,cov_x2x3,
            mean_dx2,mean_dx3,var_dx2,var_dx3,cov_dx2dx3`; biased moments over the particles that crossed, `nan` if none;
            transverse unwrapped; dx = crossing minus injection position); `--ell` and `--x0` (config length units) required.
  par2 series (optional, secondary) `stats/realization_%04d_timeseries.csv` (`t,mean_x,mean_y,mean_z,var_x,var_y,var_z,active`)
            plus the snapshots `<arm>/snapshots/r_%04d/step_%08d.csv` (header `id,x coord,y coord,z coord`; `<arm>` = the parent
            of the series' `stats/` directory) that define the window end; needs `--U-time`, `--par2-exit-x` and `--par2-dt`.

ESTIMANDS (decision G4; d = (x1 - x0)/ell in lambda; the travel window is d in [d_lo, d_hi], default [25, 94])
  Window planes: the run's planes with d_lo(1-tol) <= d <= d_hi(1+tol), tol = 1e-9 relative (absolute 1e-9 for |bound| < 1);
  fewer than 3 -> ReduceError.  OLS slopes are against x1 (equivalently d) of the window planes; alpha in units of lambda
  (divide by ell).  Moments are the BIASED weighted form with the weights renormalized over the particle set.
  alpha_TP   = (1/2) slope[(var x2 + var x3)/2] / ell                       (fixed axes)
  alpha_TAP  = (1/2) slope[(var_u1 + var_u2)/2] / ell, with q = (x1_last - x1_first, mu2_last - mu2_first, mu3_last - mu3_first)
               (first and last window planes; mu = mean crossing position, or mean crossing displacement for the displacement
               basis), e_L = q/|q|, u1/u2 = Gram-Schmidt of e2, e3 against e_L (u1 from e2, u2 from e3 minus its u1 part), and
               var_u = (u2,u3) C (u2,u3)^T with C the 2x2 transverse covariance at the plane (x1 is constant on a plane).
  alpha_LP   = (U_r^2/2) slope[var(tau)] / ell,  U_r = 1/slope[mean(tau)]  (tau = time since injection).  Independent of the
               basis (the same value is reported in the position and displacement rows).
  SF-43 `--pollock` (`pollock=True`; default off, which is the SF-42 behaviour; disclosed addition, like prune's `--keep-arrays`): for every
  consumer realization the same plane estimators on the arrays of the Pollock bridge (`<R>/pollock/`, one directory per realization,
  written once whatever the number of h_max passes; `lagrangian_ensemble --pollock-bridge on`, `EnsembleRunner.cu`), rows `code=pollock`,
  family `plane`, the consumer's columns, `source` = `<R>/pollock`.  Arrays: `pollock_weights` (Np, flux weights of the RT0 interpolant at the
  seed, 0 for non-positive seeds), `pollock_planes_x23` (Np,P,2 unwrapped, NaN once the particle stopped or exited) and `pollock_planes_t`
  (Np,P clock since injection) play the roles of `weights`, `planes_x23`, `planes_tau`; the displacement basis uses the pass's `seeds`
  (the Pollock lattice is the tracker's: `pollock.json seeds_equal_tracker` and the plane list are checked).  Variants: weights flux
  (= pollock_weights) and unif, bases pos/disp, sets perplane/common, and the split-half columns (lattice index p = m2 M + m3 is the same);
  no `expI` (the bridge has no `planes_I`) and no `time` family.  A realization without `pollock/` is an error (`MissingData`).  Under a
  multi-pass realization the identical Pollock rows are written under each pass's `hmax_div` label (the labels of the consumer rows).  The
  per-plane covariance table gets `code=pollock` rows too.  `prune` deletes a Pollock array only when these rows exist (finite).
  Particle sets: per plane = particles that crossed that plane (finite x23 and tau; seeds with weight 0 never; weights
  renormalized over the set); `set=common` = particles that crossed EVERY plane of the window (check variant).  Columns
  n_first, n_last, n_min, n_max, n_common (counts), set_differs (1 if any plane count differs from n_common; the sets are
  nested because records are prefix-complete), wfrac_min (smallest share of the variant's reference weight inside a plane set).
  Variants: weights `flux` (consumer weights.npy; primary), `unif` (uniform over the set; twin-equivalent, used against Par2),
  `expI` (flux * exp(-planes_I) at each plane; weight-sensitivity only); basis `pos` (x23) and `disp` (x23 - seed x23, seeds.npy);
  Par2: weights `unif`, bases `pos`/`disp` from the CSV, `set=perplane` only (the CSV has no common-set information).
  Split-half (consumer): lattice checkerboard, half A = (m2 + m3) even, half B = odd, with lattice index p = m2*M + m3 and
  Np = M^2 (as SF-39 half_masks); every estimator is recomputed on each half with the same weights rule (renormalized over the
  half's own sets) and reported as `<est>_A`, `<est>_B`; the particle share of the per-realization variance is
  `splitshare` (below, SF-39's definition: mean_r[(a_A - a_B)^2]/4 over the sample variance, ddof=1, flag > 0.25).
  Par2 has no per-particle data: split columns are NaN.

SECONDARY, family `time` (SF-39 style, descriptive; consumer all weights/bases pos; Par2 from the series, fixed axes only)
  t_end,r: the exact exit times are NOT recorded (`exits_time.csv` holds cumulative counts per snapshot).  Best available
  measure: j_first = the smallest snapshot index at which a particle with weight > 0 and status_time 7/8 has no record (the
  count of its finite `times_x` records); t_end,r = times[j_first - 1], the last snapshot with no exit before it, so the first
  exit lies in (t_end, times[j_first]] (resolution = that interval, column t_end_next; NaN if no particle exited, then
  t_end = the last snapshot).  The window is [0.75, 1] t_end,r; the set = particles with a record at t_end (no exit particle
  is lost inside the window: W13; particles that stopped for a non-exit reason are excluded and counted via n_set vs n_traced).
  Variances (full 3x3 covariance, projections e^T C e) are linearly interpolated between snapshots; alpha_e = [V_e(t_end) -
  V_e(t_lo)] / (2 ell U (t_end - t_lo)) with U = the realization's qbar (flow_stats.json) unless `--U-time` is given.
  alpha_LAM = fixed-axis e1; alpha_Tfixed = (e2 + e3)/2; alpha_TAM = aligned frame (e_L = unit interior mean flux vector of
  flow_stats.json, Gram-Schmidt transverse pair).  Par2: t_end = the last series row with active == its first row's active,
  alpha_LAM from var_x, alpha_Tfixed from (var_y + var_z)/2 (needs `--U-time`), alpha_TAM NaN.
  Par2 window end (positions, NOT the `active` count): in the pipeline the Dirichlet/Neumann x faces map to Par2 `Closed`
  (src/physics/particles/par2_adapter/par2_mapping.cuh:28-37; nothing maps to `Open`), and `Closed` zeroes a move that would leave the
  domain (Par2_Core types.hpp:36-44), so the particle stays and piles up near the outlet; only `Open` sets ParticleStatus::Exited
  (types.hpp:59-67; kernels/move_particles.cu:528-531), and `active` counts Status::Active particles (stats.hpp:87,
  Par2StatsAdapter.cu:57).  Hence `active` never decreases in the pipeline (it is the injected count at every sample) and the
  old rule "last row with the initial active count" would end the window at the end of the run, with particles frozen at the outlet
  inside it.  Rule (copy of SF-39 `par2.t_b_last_clean`): t_end,r = step*`--par2-dt` of the LAST snapshot (step order, stopping at the
  first offender) in which no particle has x >= `--par2-exit-x` (config length units; SF-39 used Lx - 5 lambda, the consumer's first exit
  is at the outlet face x = L1; the choice is the caller's and is logged in the table by t_end_next); t_end_next = time of the first
  snapshot with a particle beyond X_exit (resolution; NaN if none, then t_end = the last snapshot); j_end = index of the t_end snapshot;
  no snapshot, or a particle beyond X_exit already in the first snapshot -> WindowError (no fallback to `active`).  The window is
  [0.75, 1] t_end,r on the series (t_end must lie inside the series' times, within 1e-9 relative); n_set = the series' `active`
  at the last row with t <= t_end (diagnostic only).

TABLES (`%.17g`, `nan` for undefined; ints and 0/1 flags as integers)
  <out>              one row per realization x pass x code x family x weights x basis x set; columns COLUMNS
  <out minus .csv>.planes.csv   per realization/pass x weights x basis x plane (every plane of the run, per-plane sets):
                     n, x1, d, in_window, mean_tau, var_tau, mean_x2, mean_x3, var_x2, var_x3, cov_x2x3 (positions or
                     displacements per basis) -- the full transverse covariances the moment CSVs lack
  <out>.manifest.json  sha256 of every input file read (path relative to the root, size) and of the tables.
  A row that cannot be evaluated (empty set at a window plane, degenerate frame, ...) carries NaN estimates and the reason in
  `note`; nothing is clipped or dropped silently.

CLI: `python3 -m sf42red reduce|splitshare|prune ...` (see `__main__`).
"""
import csv
import glob
import hashlib
import json
import math
import os
import re
import sys

import numpy as np

STATUS_EXIT_INLET, STATUS_EXIT_OUTLET = 7, 8
REL_TOL = 1e-9
WINDOW_FRACTION = 0.75
NAN = float("nan")
PAR2_HEADER = ["x1", "n_injected", "n_crossed", "mean_t", "var_t", "mean_x2", "mean_x3", "var_x2", "var_x3", "cov_x2x3",
               "mean_dx2", "mean_dx3", "var_dx2", "var_dx3", "cov_dx2dx3"]
PAR2_FORMAT_LINE = "# format_version=1 sf42_plane_crossings weights=uniform time_origin=injection"
PLANE_EST = ("alpha_TP", "alpha_TAP", "alpha_LP")
TIME_EST = ("alpha_LAM", "alpha_TAM", "alpha_Tfixed")
ALL_EST = PLANE_EST + TIME_EST
COLUMNS = (["code", "source", "r", "hmax_div", "seed", "family", "weights", "basis", "set", "ell", "x0", "d_lo", "d_hi",
            "n_planes", "d_first", "d_last", "n_traced", "n_first", "n_last", "n_min", "n_max", "n_common", "set_differs",
            "wfrac_min", "U_r", "sigT2_first", "sigT2_last", "tilt",
            "t_end", "t_end_next", "j_end", "t_lo", "U_time", "n_set"]
           + list(ALL_EST) + [e + s for e in ALL_EST for s in ("_A", "_B")] + ["note"])
PLANE_COLUMNS = ["code", "source", "r", "hmax_div", "weights", "basis", "plane", "x1", "d", "in_window", "n", "mean_tau",
                 "var_tau", "mean_x2", "mean_x3", "var_x2", "var_x3", "cov_x2x3"]


class ReduceError(Exception):
    """Base class of every reducer error."""


class MissingData(ReduceError):
    """A required file is absent."""


class WindowError(ReduceError):
    """A time window does not fit the recorded snapshots."""


# ------------------------------------------------------------------------------------------------
# readers (every file read is recorded for the manifest)
# ------------------------------------------------------------------------------------------------
class Recorder:
    """Opens input files and remembers every path read (first-read order)."""

    def __init__(self):
        self.files = []
        self._seen = set()

    def _note(self, path):
        ap = os.path.abspath(path)
        if not os.path.isfile(ap):
            raise MissingData("missing file %s" % path)
        if ap not in self._seen:
            self._seen.add(ap)
            self.files.append(ap)
        return ap

    def json(self, path):
        with open(self._note(path)) as f:
            return json.load(f)

    def lines(self, path):
        with open(self._note(path)) as f:
            return f.read().splitlines()

    def csv_rows(self, path):
        """Rows (dicts of float) of a CSV; '#' lines skipped (`nan` parses)."""
        lines = [ln for ln in self.lines(path) if ln.strip() and not ln.startswith("#")]
        return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(lines)]

    def npy(self, path):
        return np.load(self._note(path), mmap_mode="r")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_par2_crossings(rec, path):
    """dict column -> array of a Par2 plane-crossing CSV (format line and header are checked)."""
    lines = rec.lines(path)
    if not lines or lines[0].strip() != PAR2_FORMAT_LINE:
        raise ReduceError("%s: first line is not '%s'" % (path, PAR2_FORMAT_LINE))
    body = [ln for ln in lines[1:] if ln.strip() and not ln.startswith("#")]
    if not body or body[0].split(",") != PAR2_HEADER:
        raise ReduceError("%s: header differs from the fixed sf42_plane_crossings header" % path)
    rows = [[float(v) for v in ln.split(",")] for ln in body[1:]]
    if any(len(r) != len(PAR2_HEADER) for r in rows) or not rows:
        raise ReduceError("%s: empty or ragged crossing table" % path)
    a = np.array(rows)
    return {k: a[:, i] for i, k in enumerate(PAR2_HEADER)}


def read_par2_series(rec, path):
    rows = rec.csv_rows(path)
    if not rows:
        raise ReduceError("%s: empty series" % path)
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


# ------------------------------------------------------------------------------------------------
# estimators on per-plane moments
# ------------------------------------------------------------------------------------------------
def ols_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    xc = x - x.mean()
    den = float(xc @ xc)
    if not den > 0.0:
        raise ReduceError("degenerate abscissa in the OLS slope")
    return float(xc @ (y - y.mean())) / den


def _unit(v, what):
    n = float(np.linalg.norm(v))
    if not (n > 0.0) or not math.isfinite(n):
        raise ReduceError("degenerate %s (zero or non-finite norm)" % what)
    return np.asarray(v, dtype=float) / n


def transverse_frame(eL):
    """(u1, u2): Gram-Schmidt of e2, e3 against the unit vector eL (SF-39 frame_from_longitudinal)."""
    e2, e3 = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    u1 = _unit(e2 - (e2 @ eL) * eL, "transverse frame (e_L parallel to e2)")
    u2 = _unit(e3 - (e3 @ eL) * eL - (e3 @ u1) * u1, "transverse frame (e_L in span(e2,e3))")
    return u1, u2


def plane_moments(Y, tau, w):
    """Biased weighted moments of one plane: Y (n,2) transverse, tau (n,), w (n,) renormalized.  dict n, mt, vt, mu(2), C(2,2)."""
    w = np.asarray(w, float)
    sw = float(w.sum())
    if w.size == 0 or not np.all(np.isfinite(w)) or not sw > 0.0:
        raise ReduceError("empty particle set or non-positive total weight")
    Y, tau = np.asarray(Y, float), np.asarray(tau, float)
    if not (np.all(np.isfinite(Y)) and np.all(np.isfinite(tau))):
        raise ReduceError("non-finite record inside the evaluated set")
    mu = (w[:, None] * Y).sum(axis=0) / sw
    D = Y - mu
    mt = float((w * tau).sum() / sw)
    return {"n": int(w.size), "mt": mt, "vt": float((w * (tau - mt) ** 2).sum() / sw), "mu": mu, "C": (D * w[:, None]).T @ D / sw}


def estimators(x1, mom, ell):
    """alpha_TP, alpha_TAP, alpha_LP (+ U_r, sigT2_first/last, tilt) from per-plane moments `mom` (list of plane_moments dicts
    or None for an empty plane) on planes x1 (increasing).  Raises ReduceError when a plane is empty or a frame degenerates."""
    if any(m is None for m in mom):
        raise ReduceError("empty particle set at a window plane")
    x1 = np.asarray(x1, float)
    C = np.array([m["C"] for m in mom])
    mu = np.array([m["mu"] for m in mom])
    sig = (C[:, 0, 0] + C[:, 1, 1]) / 2.0
    out = {"alpha_TP": 0.5 * ols_slope(x1, sig) / ell, "sigT2_first": float(sig[0]), "sigT2_last": float(sig[-1])}
    q = np.array([x1[-1] - x1[0], mu[-1, 0] - mu[0, 0], mu[-1, 1] - mu[0, 1]])
    eL = _unit(q, "mean crossing displacement over the window")
    u1, u2 = transverse_frame(eL)
    var_u = [np.array([float(u[1:] @ Ck @ u[1:]) for Ck in C]) for u in (u1, u2)]
    out["alpha_TAP"] = 0.5 * ols_slope(x1, (var_u[0] + var_u[1]) / 2.0) / ell
    out["tilt"] = math.atan2(float(np.linalg.norm(np.cross(eL, [1.0, 0.0, 0.0]))), float(eL[0]))
    s_mt = ols_slope(x1, [m["mt"] for m in mom])
    if not s_mt > 0.0:
        raise ReduceError("non-positive slope of the mean crossing time (U_r undefined)")
    out["U_r"] = 1.0 / s_mt
    out["alpha_LP"] = 0.5 * out["U_r"] ** 2 * ols_slope(x1, [m["vt"] for m in mom]) / ell
    return out


def window_planes(x1, x0, ell, d_lo, d_hi):
    """Indices of the planes with d = (x1 - x0)/ell in [d_lo, d_hi] (relative tolerance 1e-9); >= 3 required."""
    x1 = np.asarray(x1, float)
    if x1.ndim != 1 or np.any(np.diff(x1) <= 0.0):
        raise ReduceError("the plane list is not strictly increasing")
    d = (x1 - x0) / ell
    tl, th = REL_TOL * max(1.0, abs(d_lo)), REL_TOL * max(1.0, abs(d_hi))
    idx = np.flatnonzero((d >= d_lo - tl) & (d <= d_hi + th))
    if len(idx) < 3:
        raise ReduceError("only %d planes in the travel window d in [%.17g, %.17g] lambda (>= 3 required)" % (len(idx), d_lo, d_hi))
    return idx


# ------------------------------------------------------------------------------------------------
# consumer: plane family
# ------------------------------------------------------------------------------------------------
def _nan_est(names, with_halves):
    """NaN placeholders of the estimator columns (an estimate that cannot be evaluated is NaN with a note, never blank)."""
    out = {e: NAN for e in names}
    if with_halves:
        out.update({e + s: NAN for e in names for s in ("_A", "_B")})
    return out


def half_masks(Np, M):
    """(A, B): (m2+m3) even / odd with lattice index p = m2*M + m3 (SF-39 half_masks)."""
    if Np != M * M:
        raise ReduceError("Np=%d is not M^2 (M=%d): cannot form the lattice halves" % (Np, M))
    p = np.arange(Np)
    even = ((p // M + p % M) % 2) == 0
    return even, ~even


class ConsumerPass:
    """Lazy arrays of one tracker pass plus the realization metadata (all reads go through the recorder)."""

    def __init__(self, rec, pass_root, pollock=False):
        self.rec, self.root = rec, pass_root
        self.rdir = pass_root if os.path.isfile(os.path.join(pass_root, "field.json")) else os.path.dirname(os.path.abspath(pass_root))
        self.track = rec.json(os.path.join(pass_root, "track.json"))
        self.field = rec.json(os.path.join(self.rdir, "field.json"))
        self.flow = rec.json(os.path.join(self.rdir, "flow_stats.json"))
        self.planes = np.asarray(self.track["planes"], float)
        self.times = np.asarray(self.track["times"], float)
        self.x0 = float(self.track["X0"])
        self.M = int(self.track["M"])
        e, lam = float(self.field["ell"]), float(self.flow["lambda"])
        if abs(e - lam) > 1e-9 * abs(lam):
            raise ReduceError("field.json ell=%.17g differs from flow_stats.json lambda=%.17g in %s" % (e, lam, self.rdir))
        self.ell = e
        self.pollock_dir = None
        if pollock:
            self._open_pollock()
        else:
            self.weights = np.asarray(self.npy("weights"), float)
        self.traced = self.weights > 0.0

    def npy(self, name):
        return self.rec.npy(os.path.join(self.root, name + ".npy"))

    def _open_pollock(self):
        """SF-43 (`pollock=True`): the Pollock bridge arrays of this realization (`R/pollock/`, written once per realization, the same
        for every h_max pass).  Same seed lattice (p = m2 M + m3), planes and Np as the pass: checked, never assumed."""
        d = os.path.join(self.rdir, "pollock")
        if not os.path.isdir(d):
            raise MissingData("pollock reduction requested but %s does not exist (the realization ran without the Pollock bridge?)" % d)
        self.pollock_dir = d
        pj = self.rec.json(os.path.join(d, "pollock.json"))
        if pj.get("seeds_equal_tracker") is not True:
            raise ReduceError("%s: pollock.json seeds_equal_tracker is not true (the displacement basis needs the tracker's seeds)" % d)
        pl = np.asarray(pj["planes"], float)
        if pl.shape != self.planes.shape or np.any(np.abs(pl - self.planes) > 1e-12 * np.maximum(1.0, np.abs(self.planes))):
            raise ReduceError("%s: pollock.json planes differ from track.json planes of %s" % (d, self.root))
        self.weights = np.asarray(self.pollock_npy("weights"), float)
        Np = len(self.npy("seeds"))
        if self.weights.shape != (Np,):
            raise ReduceError("%s: pollock_weights has shape %s for %d seeds" % (d, self.weights.shape, Np))

    def pollock_npy(self, name):
        return self.rec.npy(os.path.join(self.pollock_dir, "pollock_" + name + ".npy"))


def _weights_matrix(w, mode, I):
    """(Np,K) weights of a mode: flux (w), unif (1), expI (w exp(-I), finite I required where used)."""
    K = I.shape[1] if I is not None else 1
    if mode == "flux":
        return np.repeat(w[:, None], K, axis=1)
    if mode == "unif":
        return np.ones((len(w), K))
    return w[:, None] * np.exp(-I)


def consumer_plane_variant(x1, fin, Wk, Yk, tauk, member, common):
    """(moments list, counts) for one variant.  fin (Np,K) crossed; Wk weights; Yk (Np,K,2); tauk (Np,K)."""
    S = fin & member[:, None]
    if common:
        S = S & S.all(axis=1, keepdims=True)
    mom, counts = [], []
    for k in range(S.shape[1]):
        sel = np.flatnonzero(S[:, k])
        counts.append(len(sel))
        mom.append(plane_moments(Yk[sel, k, :], tauk[sel, k], Wk[sel, k]) if len(sel) else None)
    return mom, counts


def consumer_plane_rows(rec, pass_root, ell_arg, x0_arg, d_lo, d_hi, single_pass_hmax=None, pollock=False):
    """(rows, plane_rows) of the plane family for one tracker pass.  `pollock=True` (SF-43): the same estimators on the Pollock bridge
    arrays of the realization (`pollock_{weights,planes_x23,planes_t}.npy` for the consumer's `weights,planes_x23,planes_tau`; seeds
    from the pass), rows `code=pollock`, `source` = the realization's `pollock/` directory, weights flux (= pollock_weights) and unif
    only (the bridge has no `planes_I`: no `expI`).  Records are NaN from a particle's first non-active plane on and the finite mask
    below is the only status handling, exactly as for the consumer."""
    P = ConsumerPass(rec, pass_root, pollock)
    code, source = ("pollock", P.pollock_dir) if pollock else ("consumer", pass_root)
    if ell_arg is not None and abs(P.ell - ell_arg) > 1e-9 * abs(P.ell):
        raise ReduceError("--ell %.17g differs from field.json ell %.17g in %s" % (ell_arg, P.ell, pass_root))
    if x0_arg is not None and abs(P.x0 - x0_arg) > 1e-9 * max(1.0, abs(P.x0)):
        raise ReduceError("--x0 %.17g differs from track.json X0 %.17g in %s" % (x0_arg, P.x0, pass_root))
    idx = window_planes(P.planes, P.x0, P.ell, d_lo, d_hi)
    x1w = P.planes[idx]
    seeds = np.asarray(P.npy("seeds"), float)
    Np = len(P.weights)
    if pollock:
        X = np.asarray(P.pollock_npy("planes_x23")[:, :, :], float)
        tau = np.asarray(P.pollock_npy("planes_t"), float)
        I = np.zeros(tau.shape)
        modes = ("flux", "unif")
        if X.shape != (Np, len(P.planes), 2) or tau.shape != (Np, len(P.planes)):
            raise ReduceError("%s: pollock array shapes %s %s differ from (Np=%d, P=%d)" % (P.pollock_dir, X.shape, tau.shape, Np, len(P.planes)))
    else:
        X = np.asarray(P.npy("planes_x23")[:, :, :], float)  # all planes (covariance table), (Np,P,2)
        tau = np.asarray(P.npy("planes_tau"), float)
        I = np.asarray(P.npy("planes_I"), float)
        modes = ("flux", "unif", "expI")
    fin_all = np.isfinite(X).all(axis=2) & np.isfinite(tau) & P.traced[:, None]
    if not pollock and np.any(fin_all & ~np.isfinite(I)):
        raise ReduceError("non-finite planes_I at a recorded crossing in %s" % pass_root)
    I0 = np.where(fin_all, I, 0.0)
    disp = X - seeds[:, None, 1:3]
    try:
        A, B = half_masks(Np, P.M)
        halves = (("_A", A), ("_B", B))
        half_note = ""
    except ReduceError as e:
        halves, half_note = (), str(e)
    r = int(re.search(r"r_(\d{4})", P.rdir).group(1))
    hm = re.search(r"hmax_(\d+)$", os.path.normpath(pass_root))
    base = {"code": code, "source": source, "r": r, "hmax_div": int(hm.group(1)) if hm else -1 if single_pass_hmax is None else int(single_pass_hmax),
            "seed": int(P.field["seed"]), "family": "plane", "ell": P.ell, "x0": P.x0, "d_lo": d_lo, "d_hi": d_hi,
            "n_planes": len(idx), "d_first": (x1w[0] - P.x0) / P.ell, "d_last": (x1w[-1] - P.x0) / P.ell,
            "n_traced": int(P.traced.sum())}
    rows, prow = [], []
    all_mem = np.ones(Np, bool)
    for wmode in modes:
        Wk_all = _weights_matrix(P.weights, wmode, I0)
        for basis, Y_all in (("pos", X), ("disp", disp)):
            for sname, common in (("perplane", False), ("common", True)):
                row = dict(base, weights=wmode, basis=basis, set=sname, note="", **_nan_est(PLANE_EST, bool(halves)))
                fin = fin_all[:, idx]
                try:
                    mom, cnt = consumer_plane_variant(x1w, fin, Wk_all[:, idx], Y_all[:, idx, :], tau[:, idx], all_mem, common)
                    n_common = int(fin.all(axis=1).sum())
                    if wmode == "unif":
                        fr = [fin[:, k].sum() / float(P.traced.sum()) for k in range(len(idx))]
                    else:
                        fr = [P.weights[fin[:, k]].sum() / P.weights.sum() for k in range(len(idx))]
                    row.update(n_first=cnt[0], n_last=cnt[-1], n_min=min(cnt), n_max=max(cnt), n_common=n_common,
                               set_differs=int(any(c != n_common for c in cnt)), wfrac_min=float(min(fr)))
                    row.update(estimators(x1w, mom, P.ell))
                except ReduceError as e:
                    row["note"] = str(e)
                for suffix, mem in halves:
                    try:
                        mh, _ = consumer_plane_variant(x1w, fin, Wk_all[:, idx], Y_all[:, idx, :], tau[:, idx], mem, common)
                        eh = estimators(x1w, mh, P.ell)
                        for e in PLANE_EST:
                            row[e + suffix] = eh[e]
                    except ReduceError as e:
                        row["note"] = (row["note"] + "; " if row["note"] else "") + "half%s: %s" % (suffix, e)
                if half_note:
                    row["note"] = (row["note"] + "; " if row["note"] else "") + "split-half unavailable: " + half_note
                rows.append(row)
            # per-plane covariance table (every plane of the run, per-plane sets)
            for k in range(len(P.planes)):
                sel = np.flatnonzero(fin_all[:, k])
                pr = {"code": code, "source": source, "r": r, "hmax_div": base["hmax_div"], "weights": wmode,
                      "basis": basis, "plane": k, "x1": float(P.planes[k]), "d": (P.planes[k] - P.x0) / P.ell,
                      "in_window": int(k in set(idx.tolist())), "n": len(sel)}
                if len(sel):
                    m = plane_moments(Y_all[sel, k, :], tau[sel, k], Wk_all[sel, k])
                    pr.update(mean_tau=m["mt"], var_tau=m["vt"], mean_x2=m["mu"][0], mean_x3=m["mu"][1], var_x2=m["C"][0, 0],
                              var_x3=m["C"][1, 1], cov_x2x3=m["C"][0, 1])
                prow.append(pr)
    return rows, prow


# ------------------------------------------------------------------------------------------------
# consumer: time family (secondary)
# ------------------------------------------------------------------------------------------------
def end_snapshot(times_x, status_time, w):
    """(j_end, j_next): t_end = times[j_end] is the last snapshot before the first exit; j_next the first snapshot at which an
    exited particle has no record (None if no particle with weight > 0 exited inside the snapshots; then j_end = T-1)."""
    T = times_x.shape[1]
    st = np.asarray(status_time)
    ex = np.flatnonzero(((st == STATUS_EXIT_INLET) | (st == STATUS_EXIT_OUTLET)) & (np.asarray(w) > 0.0))
    if len(ex) == 0:
        return T - 1, None
    n_fin = np.isfinite(np.asarray(times_x[ex, :, :])).all(axis=2).sum(axis=1)
    j_next = int(n_fin.min())
    if j_next >= T:
        return T - 1, None
    if j_next == 0:
        raise WindowError("a particle exits before the first snapshot: no complete snapshot")
    return j_next - 1, j_next


def _bracket(times, t):
    jb = int(np.searchsorted(times, t, side="left"))
    if jb >= len(times) or (jb == 0 and times[0] != t):
        raise WindowError("time %.17g outside the recorded snapshots" % t)
    if times[jb] == t:
        return jb, jb, 0.0
    return jb - 1, jb, (t - times[jb - 1]) / (times[jb] - times[jb - 1])


def cov3(X, w):
    w = np.asarray(w, float)
    if w.size == 0 or not np.all(np.isfinite(X)) or not w.sum() > 0.0:
        raise ReduceError("empty set, non-finite record or zero total weight in a snapshot")
    sw = w.sum()
    m = (w[:, None] * X).sum(axis=0) / sw
    D = X - m
    return (D * w[:, None]).T @ D / sw


def time_alphas(times, cov_at, t_lo, t_hi, q, ell, U):
    """alpha_LAM (e1), alpha_Tfixed ((e2+e3)/2), alpha_TAM (aligned frame, q=None -> NaN) from covariances cov_at(j)."""
    def at(t):
        ja, jb, f = _bracket(times, t)
        Ca = cov_at(ja)
        return Ca if f == 0.0 else Ca + f * (cov_at(jb) - Ca)
    dC = at(t_hi) - at(t_lo)
    den = 2.0 * ell * U * (t_hi - t_lo)
    a = lambda e: float(e @ dC @ e) / den  # noqa: E731
    E = np.eye(3)
    out = {"alpha_LAM": a(E[0]), "alpha_Tfixed": (a(E[1]) + a(E[2])) / 2.0, "alpha_TAM": NAN}
    if q is not None:
        t1, t2 = transverse_frame(_unit(np.asarray(q, float), "interior mean velocity"))
        out["alpha_TAM"] = (a(t1) + a(t2)) / 2.0
    return out


def consumer_time_rows(rec, pass_root, U_arg, single_pass_hmax=None):
    P = ConsumerPass(rec, pass_root)
    tx = P.npy("times_x")
    tI = P.npy("times_I")
    try:
        j_end, j_next = end_snapshot(tx, P.npy("status_time"), P.weights)
    except WindowError as e:
        return [{"code": "consumer", "source": pass_root, "r": int(re.search(r"r_(\d{4})", P.rdir).group(1)), "family": "time",
                 "weights": w, "basis": "pos", "set": "perplane", "note": str(e), **_nan_est(TIME_EST, True)}
                for w in ("flux", "unif", "expI")]
    t_end = float(P.times[j_end])
    t_lo = WINDOW_FRACTION * t_end
    U = float(P.flow["qbar"]) if U_arg is None else U_arg
    q = (P.flow["interior"]["mean_u"], P.flow["interior"]["mean_v"], P.flow["interior"]["mean_w"])
    r = int(re.search(r"r_(\d{4})", P.rdir).group(1))
    hm = re.search(r"hmax_(\d+)$", os.path.normpath(pass_root))
    Np = len(P.weights)
    fin = np.isfinite(np.asarray(tx[:, j_end, :])).all(axis=1) & P.traced
    try:
        A, B = half_masks(Np, P.M)
        halves, hnote = (("_A", A), ("_B", B)), ""
    except ReduceError as e:
        halves, hnote = (), str(e)
    rows = []
    for wmode in ("flux", "unif", "expI"):
        row = {"code": "consumer", "source": pass_root, "r": r, "hmax_div": int(hm.group(1)) if hm else -1 if single_pass_hmax is None else int(single_pass_hmax),
               "seed": int(P.field["seed"]), "family": "time", "weights": wmode, "basis": "pos", "set": "perplane",
               "ell": P.ell, "x0": P.x0, "n_traced": int(P.traced.sum()), "t_end": t_end,
               "t_end_next": NAN if j_next is None else float(P.times[j_next]), "j_end": j_end, "t_lo": t_lo, "U_time": U,
               "note": ""}
        row.update(_nan_est(TIME_EST, True))
        for suffix, mem in (("", np.ones(Np, bool)),) + tuple(halves):
            try:
                idx = np.flatnonzero(fin & mem)
                if suffix == "":
                    row["n_set"] = len(idx)
                wt = lambda j: (P.weights[idx] if wmode == "flux" else np.ones(len(idx)) if wmode == "unif"  # noqa: E731
                                else P.weights[idx] * np.exp(-np.asarray(tI[idx, j], float)))
                cov_at = lambda j: cov3(np.asarray(tx[idx, j, :], float), wt(j))  # noqa: E731
                al = time_alphas(P.times, cov_at, t_lo, t_end, q, P.ell, U)
                for e in TIME_EST:
                    row[e + suffix] = al[e]
            except ReduceError as e:
                row["note"] = (row["note"] + "; " if row["note"] else "") + "%s: %s" % (suffix or "all", e)
        if hnote:
            row["note"] = (row["note"] + "; " if row["note"] else "") + "split-half unavailable: " + hnote
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------------------------
# Par2
# ------------------------------------------------------------------------------------------------
def par2_rows(rec, path, ell, x0, d_lo, d_hi, seed0=None):
    """(rows, plane_rows) from one crossing CSV.  seed0 (SF-43): seed of realization 0 of the run, the rows then carry seed0 + r
    (default None: NaN, the SF-42 behaviour)."""
    t = read_par2_crossings(rec, path)
    r = int(re.search(r"realization_(\d{4})_planes", os.path.basename(path)).group(1))
    idx = window_planes(t["x1"], x0, ell, d_lo, d_hi)
    x1w = t["x1"][idx]
    ncr = t["n_crossed"]
    base = {"code": "par2", "source": path, "r": r, "hmax_div": -1, "seed": NAN if seed0 is None else int(seed0) + r, "family": "plane",
            "weights": "unif", "set": "perplane", "ell": ell, "x0": x0, "d_lo": d_lo, "d_hi": d_hi, "n_planes": len(idx),
            "d_first": (x1w[0] - x0) / ell, "d_last": (x1w[-1] - x0) / ell, "n_traced": int(t["n_injected"][idx[0]]),
            "n_first": int(ncr[idx[0]]), "n_last": int(ncr[idx[-1]]), "n_min": int(ncr[idx].min()), "n_max": int(ncr[idx].max()),
            "n_common": int(ncr[idx].min()), "set_differs": int(ncr[idx].min() != ncr[idx].max()),
            "wfrac_min": float(ncr[idx].min() / t["n_injected"][idx[0]])}
    rows, prow = [], []
    for basis, (m2, m3, v2, v3, c23) in (("pos", ("mean_x2", "mean_x3", "var_x2", "var_x3", "cov_x2x3")),
                                         ("disp", ("mean_dx2", "mean_dx3", "var_dx2", "var_dx3", "cov_dx2dx3"))):
        row = dict(base, basis=basis, note="", **_nan_est(PLANE_EST, False))
        try:
            if np.any(ncr[idx] <= 0):
                raise ReduceError("no particle crossed a window plane")
            mom = [{"n": int(ncr[k]), "mt": t["mean_t"][k], "vt": t["var_t"][k], "mu": np.array([t[m2][k], t[m3][k]]),
                    "C": np.array([[t[v2][k], t[c23][k]], [t[c23][k], t[v3][k]]])} for k in idx]
            row.update(estimators(x1w, mom, ell))
        except ReduceError as e:
            row["note"] = str(e)
        rows.append(row)
        for k in range(len(t["x1"])):
            prow.append({"code": "par2", "source": path, "r": r, "hmax_div": -1, "weights": "unif", "basis": basis, "plane": k,
                         "x1": t["x1"][k], "d": (t["x1"][k] - x0) / ell, "in_window": int(k in set(idx.tolist())),
                         "n": int(ncr[k]), "mean_tau": t["mean_t"][k], "var_tau": t["var_t"][k], "mean_x2": t[m2][k],
                         "mean_x3": t[m3][k], "var_x2": t[v2][k], "var_x3": t[v3][k], "cov_x2x3": t[c23][k]})
    return rows, prow


def par2_snapshot_end(rec, snap_dir, x_exit, dt):
    """(j_end, t_end, t_next, n_snapshots): the LAST snapshot (step order, stop at the first offender) with no particle at
    x >= x_exit; t = step*dt (copy of SF-39 `par2.t_b_last_clean`).  t_next = time of the first offender (NaN if none)."""
    files = sorted((int(m.group(1)), p) for p in glob.glob(os.path.join(snap_dir, "step_*.csv"))
                   for m in [re.search(r"step_(\d+)\.csv$", p)] if m)
    if not files:
        raise WindowError("no Par2 snapshots in %s (the window end is defined by particle positions)" % snap_dir)
    j_end, t_next = None, NAN
    for j, (step, p) in enumerate(files):
        lines = rec.lines(p)
        head = [h.strip().lower() for h in lines[0].split(",")]
        if "x coord" not in head:
            raise ReduceError("%s: no `x coord` column (header %s)" % (p, head))
        k = head.index("x coord")
        xs = [float(ln.split(",")[k]) for ln in lines[1:] if ln.strip()]
        if xs and max(xs) >= x_exit:
            t_next = step * dt
            break
        j_end = j
    if j_end is None:
        raise WindowError("a particle is at x >= %.17g already in the first snapshot of %s: no clean snapshot" % (x_exit, snap_dir))
    return j_end, files[j_end][0] * dt, t_next, len(files)


def par2_time_row(rec, path, ell, x0, U, x_exit, dt, snap_dir=None, seed0=None):
    """Secondary time-window row of one Par2 series (fixed axes); the window end comes from the snapshots (`par2_snapshot_end`)."""
    r = int(re.search(r"realization_(\d{4})_timeseries", os.path.basename(path)).group(1))
    if snap_dir is None:
        snap_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(path))), "snapshots", "r_%04d" % r)
    j_end, t_end, t_next, _ = par2_snapshot_end(rec, snap_dir, x_exit, dt)
    return par2_time_row_at(rec, path, ell, x0, U, t_end, t_next, j_end, seed0)


def par2_time_row_at(rec, path, ell, x0, U, t_end, t_next, j_end, seed0=None):
    """Window [0.75, 1] t_end of one Par2 series for a given t_end (series covariances interpolated)."""
    s = read_par2_series(rec, path)
    r = int(re.search(r"realization_(\d{4})_timeseries", os.path.basename(path)).group(1))
    t, active = s["t"], s["active"]
    near = np.flatnonzero(np.abs(t - t_end) <= 1e-9 * abs(t_end))
    if len(near):
        t_end = float(t[near[0]])  # series and snapshot times are both step*dt: absorb the text round-off
    t_lo = WINDOW_FRACTION * t_end
    kact = np.flatnonzero(t <= t_end)
    row = {"code": "par2", "source": path, "r": r, "hmax_div": -1, "seed": NAN if seed0 is None else int(seed0) + r, "family": "time", "weights": "unif",
           "basis": "pos", "set": "perplane", "ell": ell, "x0": x0, "n_traced": int(active[0]),
           "n_set": int(active[kact[-1]]) if len(kact) else 0, "t_end": t_end, "t_end_next": t_next, "j_end": j_end,
           "t_lo": t_lo, "U_time": U, "note": ""}
    row.update(_nan_est(TIME_EST, False))
    try:
        if np.any(np.diff(t) <= 0.0):
            raise ReduceError("series times are not increasing")
        cov_at = lambda j: np.diag([s["var_x"][j], s["var_y"][j], s["var_z"][j]])  # noqa: E731
        al = time_alphas(t, cov_at, t_lo, t_end, None, ell, U)
        row.update(alpha_LAM=al["alpha_LAM"], alpha_Tfixed=al["alpha_Tfixed"])
    except ReduceError as e:
        row["note"] = str(e)
    return row


# ------------------------------------------------------------------------------------------------
# drivers, tables, manifest
# ------------------------------------------------------------------------------------------------
def expand_consumer(paths, warn=sys.stderr):
    """Pass roots from pass directories or arm directories (`realizations/r_%04d` with DONE; every pass)."""
    roots = []
    for p in paths:
        if os.path.isdir(os.path.join(p, "realizations")):
            rd = os.path.join(p, "realizations")
            for n in sorted(os.listdir(rd)):
                if not re.fullmatch(r"r_\d{4}", n):
                    continue
                R = os.path.join(rd, n)
                if not os.path.exists(os.path.join(R, "DONE")):
                    print("NOT_DONE (skipped, reported): %s" % R, file=warn)
                    continue
                if os.path.isfile(os.path.join(R, "track.json")):
                    roots.append(R)
                else:
                    roots += [os.path.join(R, d) for d in sorted(os.listdir(R)) if re.fullmatch(r"hmax_\d+", d)]
        elif os.path.isfile(os.path.join(p, "track.json")):
            roots.append(p)
        else:
            raise MissingData("%s is neither a pass root nor an arm directory" % p)
    return roots


def _fmt(v):
    if v is None or v == "":
        return ""
    if isinstance(v, (bool, np.bool_)):
        return str(int(v))
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return "nan" if math.isnan(v) else "%.17g" % v
    return str(v)


def write_table(path, rows, columns):
    unknown = {k for row in rows for k in row} - set(columns)
    if unknown:
        raise ReduceError("columns not declared: %s" % sorted(unknown))
    with open(path, "w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(columns)
        for row in rows:
            wr.writerow([_fmt(row.get(k)) for k in columns])


def _relpath(p, root):
    return os.path.relpath(os.path.abspath(p), root)


def write_manifest(path, rec, root, tables):
    """JSON manifest: sha256 of every input read (path relative to `root`) and of each table (relative to the manifest dir)."""
    root = os.path.abspath(root)
    for f in rec.files:
        if os.path.commonpath([root, f]) != root:
            raise ReduceError("input %s is outside the manifest root %s" % (f, root))
    man = {"format": "sf42red_manifest_v1", "root_note": "input paths are relative to the --root given to the tool",
           "files": [{"path": _relpath(f, root), "size": os.path.getsize(f), "sha256": sha256_file(f)} for f in rec.files],
           "tables": [{"path": os.path.basename(t), "sha256": sha256_file(t)} for t in tables]}
    with open(path, "w") as f:
        json.dump(man, f, indent=1, sort_keys=True)
        f.write("\n")


def reduce_all(consumer, par2, par2_series, ell, x0, d_lo, d_hi, U_time, out, root=None, warn=sys.stderr, par2_exit_x=None,
               par2_dt=None, par2_seed0=None, single_pass_hmax=None, pollock=False):
    """Run every reduction, write <out>, <out minus .csv>.planes.csv and <out>.manifest.json; returns the table rows.
    SF-43 options (default None = the SF-42 behaviour): par2_seed0 (Par2 rows carry seed0 + r); single_pass_hmax (a single-pass
    realization root, which has no `hmax_<d>` directory, is labelled hmax_div = this value instead of -1, so an arm that mixes
    single- and multi-pass segments has one label for its production pass); pollock=True (SF-43): per consumer pass root additionally the
    plane rows of the realization's Pollock bridge arrays (`code=pollock`, see `consumer_plane_rows`; a realization without them is an
    error).  The Pollock arrays do not depend on h_max, so a multi-pass realization gets identical Pollock rows under each of its pass labels."""
    if par2 or par2_series:
        if ell is None or x0 is None:
            raise ReduceError("--ell and --x0 are required for Par2 inputs")
    if par2_series and U_time is None:
        raise ReduceError("--U-time is required for --par2-series")
    if par2_series and (par2_exit_x is None or par2_dt is None):
        raise ReduceError("--par2-exit-x and --par2-dt are required for --par2-series (the window end is defined by snapshots)")
    if root is None:
        root = os.path.commonpath([os.path.abspath(p) if os.path.isdir(p) else os.path.dirname(os.path.abspath(p))
                                   for p in list(consumer) + list(par2)]
                                  + [os.path.dirname(os.path.dirname(os.path.abspath(p))) for p in par2_series])  # arm dir: snapshots/
    rec = Recorder()
    rows, prows = [], []
    for pr in expand_consumer(consumer, warn):
        a, b = consumer_plane_rows(rec, pr, ell, x0, d_lo, d_hi, single_pass_hmax)
        rows += a + consumer_time_rows(rec, pr, U_time, single_pass_hmax)
        prows += b
        if pollock:
            a, b = consumer_plane_rows(rec, pr, ell, x0, d_lo, d_hi, single_pass_hmax, pollock=True)
            rows += a
            prows += b
    for p in par2:
        a, b = par2_rows(rec, p, ell, x0, d_lo, d_hi, par2_seed0)
        rows += a
        prows += b
    for p in par2_series:
        rows.append(par2_time_row(rec, p, ell, x0, U_time, par2_exit_x, par2_dt, seed0=par2_seed0))
    if not rows:
        raise ReduceError("no input")
    for row in rows + prows:  # `source` relative to the root, the key prune uses to check coverage
        row["source"] = _relpath(row["source"], root)
    base = out[:-4] if out.endswith(".csv") else out
    planes_path = base + ".planes.csv"
    write_table(out, rows, COLUMNS)
    write_table(planes_path, prows, PLANE_COLUMNS)
    write_manifest(out + ".manifest.json", rec, root, [out, planes_path])
    return rows


# ------------------------------------------------------------------------------------------------
# split-half share (SF-39 definition)
# ------------------------------------------------------------------------------------------------
def splithalf_fraction(a_A, a_B, a_r):
    """mean_r[(aA - aB)^2]/4 over the sample variance (ddof=1) of the per-realization estimator; flag if > 0.25."""
    aA, aB, a = (np.asarray(v, float) for v in (a_A, a_B, a_r))
    comp = float(np.mean((aA - aB) ** 2) / 4.0)
    var = float(a.var(ddof=1))
    frac = comp / var if var > 0 else (math.inf if comp > 0 else NAN)
    return {"component": comp, "variance": var, "fraction": float(frac), "flag": bool(frac > 0.25), "n": int(len(a))}


def split_share(table_path):
    """Rows (group, estimator, n, component, variance, fraction, flag) of the particle share, grouped by code, family, weights,
    basis, set and hmax_div; realizations with a non-finite value or half are dropped (counted in n)."""
    with open(table_path) as f:
        rows = list(csv.DictReader(f))
    groups = {}
    for row in rows:
        groups.setdefault((row["code"], row["family"], row["weights"], row["basis"], row["set"], row["hmax_div"]), []).append(row)
    out = []
    for g, rs in sorted(groups.items()):
        for e in ALL_EST:
            vals = [(float(r[e]), float(r[e + "_A"]), float(r[e + "_B"])) for r in rs if r.get(e, "") != ""]
            vals = [v for v in vals if all(math.isfinite(x) for x in v)]
            if len(vals) < 2:
                continue
            a, A, B = zip(*vals)
            out.append(dict(zip(("code", "family", "weights", "basis", "set", "hmax_div"), g), estimator=e,
                            **splithalf_fraction(A, B, a)))
    return out
