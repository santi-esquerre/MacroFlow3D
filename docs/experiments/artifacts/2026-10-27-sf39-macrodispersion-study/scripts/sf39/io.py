"""SF-39 readers of the `lagrangian_ensemble` (medium b) outputs (contract C-REDUCE, input side).

Layout (preregistration section 3): realization directory `R = <out>/realizations/r_%04d/` (`R/DONE` marks completion); pass
root `P = R/hmax_<d>/` when the run used several `--hmax-div` values (directory A: 8,16,32), `P = R` for a single pass. Per-pass
files live in `P`; `field.json`, `flow_stats.json`, `status.json`, `pollock/` live in `R`.  Arrays are loaded lazily (mmap).
A record is NaN until reached and prefix-complete (finite for indices < n_reached).  Times and planes are always read from
`track.json` / `pollock.json`, never from stdout.

Tracker status codes: 0 ok, 1 seed_backflow, 2 cap_exceeded, 3 stagnation, 4 step_underflow, 5 landing_failed,
6 invalid_conductivity, 7 exit_inlet, 8 exit_outlet.

Errors: `ReduceError` (base), `MissingData` (a file, plane or pass that is absent), `WindowError` (the window does not fit the
recorded snapshots).  Nothing is clipped or dropped silently; the `tolerate_missing` flag of the callers only turns
`MissingData` into a reported `None`.
"""
import csv
import json
import os
import re
from functools import cached_property

import numpy as np

STATUS_OK, STATUS_BACKFLOW, STATUS_EXIT_INLET, STATUS_EXIT_OUTLET = 0, 1, 7, 8
N_STATUS = 9


class ReduceError(Exception):
    """Base class of every reducer error."""


class MissingData(ReduceError):
    """A required file, plane or pass is absent."""


class WindowError(ReduceError):
    """The averaging window does not fit the recorded snapshots (no snapshot >= t_hi, or t_lo < t_1)."""


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        raise MissingData("missing file %s" % path)


def read_csv_rows(path):
    """Rows (dicts of float) of a runner CSV; lines starting with '#' (format_version) are skipped."""
    try:
        with open(path) as f:
            lines = [ln for ln in f if not ln.startswith("#") and ln.strip()]
    except FileNotFoundError:
        raise MissingData("missing file %s" % path)
    return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(lines)]


def read_series_csv(path):
    """Run-level uniform-weight series (`stats_twin/realization_%04d_timeseries.csv`): dict column -> array."""
    rows = read_csv_rows(path)
    return {k: np.array([r[k] for r in rows]) for k in rows[0]} if rows else {}


def series_path(out_dir, r, hmax_div=None, twin=True):
    """`<out>/hmax_<d>/stats_twin/realization_%04d_timeseries.csv` (d given) or `<out>/stats_twin/...` (single pass)."""
    base = os.path.join(out_dir, "hmax_%d" % hmax_div) if hmax_div is not None else out_dir
    return os.path.join(base, "stats_twin" if twin else "stats", "realization_%04d_timeseries.csv" % r)


def _load(path):
    try:
        return np.load(path, mmap_mode="r")
    except FileNotFoundError:
        raise MissingData("missing file %s" % path)


def find_plane(planes, x):
    """Index of the plane nearest to x if within 1e-12*max(1,|x|), else None."""
    planes = np.asarray(planes, dtype=float)
    i = int(np.argmin(np.abs(planes - x)))
    return i if abs(planes[i] - x) <= 1e-12 * max(1.0, abs(x)) else None


def plane_index(planes, x, name, tolerate_missing=False):
    """`find_plane` that raises `MissingData` (or returns None with `tolerate_missing`)."""
    i = find_plane(planes, x)
    if i is None and not tolerate_missing:
        raise MissingData("plane %s = %.17g not in the recorded planes" % (name, x))
    return i


def key_planes(x0, L1, ell):
    """X_a = x0 + L/2, X_b = L1 - 5 ell, X_q = x0 + L/4 with L = L1 - 10 ell (preregistration sections 2-3)."""
    L = L1 - 10.0 * ell
    return {"Xa": x0 + L / 2.0, "Xb": L1 - 5.0 * ell, "Xq": x0 + L / 4.0}


def list_realizations(out_dir, subset=None):
    """(r, path, has_done) of every `realizations/r_%04d` directory (sorted); `subset` = iterable of r to keep."""
    root = os.path.join(out_dir, "realizations")
    out = []
    try:
        names = os.listdir(root)
    except FileNotFoundError:
        raise MissingData("missing directory %s" % root)
    for n in names:
        m = re.fullmatch(r"r_(\d{4})", n)
        if m and (subset is None or int(m.group(1)) in subset):
            out.append((int(m.group(1)), os.path.join(root, n), os.path.exists(os.path.join(root, n, "DONE"))))
    return sorted(out)


def parse_range(s):
    """'0-19', '3', '0-4,7,9-10' -> sorted set of ints."""
    out = set()
    for part in s.split(","):
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b if b else a) + 1))
    return out


def pass_root(rdir, div=None):
    """Pass root of a realization: `R` for a single-pass layout (`R/track.json`), else `R/hmax_<div>`."""
    if os.path.exists(os.path.join(rdir, "track.json")):
        return rdir  # single pass: the one --hmax-div cannot be verified from the files (documented assumption)
    if div is not None:
        p = os.path.join(rdir, "hmax_%d" % div)
        if os.path.isdir(p):
            return p
        raise MissingData("no pass root hmax_%d in %s" % (div, rdir))
    cands = sorted(d for d in os.listdir(rdir) if re.fullmatch(r"hmax_\d+", d)) if os.path.isdir(rdir) else []
    if len(cands) == 1:
        return os.path.join(rdir, cands[0])
    raise MissingData("cannot choose a pass root in %s (candidates %s); pass the divisor" % (rdir, cands))


def pass_roots(rdir):
    """Every pass root of a realization: [R] (single pass) or the sorted R/hmax_<d> directories."""
    if os.path.exists(os.path.join(rdir, "track.json")):
        return [rdir]
    roots = sorted(os.path.join(rdir, d) for d in os.listdir(rdir) if re.fullmatch(r"hmax_\d+", d)) if os.path.isdir(rdir) else []
    if not roots:
        raise MissingData("no pass root in %s" % rdir)
    return roots


class Pass:
    """Per-pass tracker files (lazy)."""

    def __init__(self, root):
        self.root = root

    def _a(self, name):
        return _load(os.path.join(self.root, name))

    @cached_property
    def track(self):
        return read_json(os.path.join(self.root, "track.json"))

    @cached_property
    def planes(self):
        return np.asarray(self.track["planes"], dtype=float)

    @cached_property
    def times(self):
        return np.asarray(self.track["times"], dtype=float)

    @cached_property
    def M(self):
        return int(self.track["M"])

    @cached_property
    def x0(self):
        return float(self.track["X0"])

    @cached_property
    def seeds(self):
        return self._a("seeds.npy")

    @cached_property
    def weights(self):
        return self._a("weights.npy")

    @cached_property
    def status_plane(self):
        return self._a("status_plane.npy")

    @cached_property
    def status_time(self):
        return self._a("status_time.npy")

    @cached_property
    def planes_x23(self):
        return self._a("planes_x23.npy")

    @cached_property
    def planes_tau(self):
        return self._a("planes_tau.npy")

    @cached_property
    def planes_I(self):
        return self._a("planes_I.npy")

    @cached_property
    def times_x(self):
        return self._a("times_x.npy")

    @cached_property
    def times_I(self):
        return self._a("times_I.npy")

    def exits_time_row(self, j):
        """(inlet, outlet) cumulative exit counts before reaching snapshot j (row j of exits_time.csv; t is checked)."""
        rows = read_csv_rows(os.path.join(self.root, "exits_time.csv"))
        if len(rows) != len(self.times) or abs(rows[j]["t"] - self.times[j]) > 1e-9 * max(1.0, abs(self.times[j])):
            raise ReduceError("exits_time.csv does not match track.json times in %s" % self.root)
        return int(rows[j]["exit_inlet"]), int(rows[j]["exit_outlet"])

    def status_hist(self, which):
        """Histogram (length 9) of status_plane / status_time computed from the array."""
        return np.bincount(np.asarray(getattr(self, "status_" + which), dtype=np.int64), minlength=N_STATUS)[:N_STATUS]


class Pollock:
    """Pollock bridge files of a realization (`R/pollock/`)."""

    def __init__(self, root):
        self.root = root

    def _a(self, name):
        return _load(os.path.join(self.root, name))

    @cached_property
    def meta(self):
        return read_json(os.path.join(self.root, "pollock.json"))

    @cached_property
    def planes(self):
        return np.asarray(self.meta["planes"], dtype=float)

    @cached_property
    def times(self):
        return np.asarray(self.meta["times"], dtype=float)

    @cached_property
    def weights(self):
        return self._a("pollock_weights.npy")

    @cached_property
    def status_plane(self):
        return self._a("pollock_status_plane.npy")

    @cached_property
    def status_time(self):
        return self._a("pollock_status_time.npy")

    @cached_property
    def planes_x23(self):
        return self._a("pollock_planes_x23.npy")

    @cached_property
    def times_x(self):
        return self._a("pollock_times_x.npy")


class Realization:
    """One realization at one pass: `Realization(rdir, div)`; `pollock` is None when `R/pollock/` is absent."""

    def __init__(self, rdir, div=None):
        self.rdir = rdir
        self.div = div
        self.pass_ = Pass(pass_root(rdir, div))

    @cached_property
    def field(self):
        return read_json(os.path.join(self.rdir, "field.json"))

    @cached_property
    def flow(self):
        return read_json(os.path.join(self.rdir, "flow_stats.json"))

    @cached_property
    def pollock(self):
        p = os.path.join(self.rdir, "pollock")
        if not os.path.isdir(p):
            return None
        q = Pollock(p)
        if q.planes.shape != self.pass_.planes.shape or np.any(np.abs(q.planes - self.pass_.planes) > 1e-12 * np.maximum(1.0, np.abs(self.pass_.planes))):
            raise ReduceError("Pollock planes differ from the tracker planes in %s" % self.rdir)
        if q.times.shape != self.pass_.times.shape or np.any(np.abs(q.times - self.pass_.times) > 1e-12 * np.maximum(1.0, np.abs(self.pass_.times))):
            raise ReduceError("Pollock times differ from the tracker times in %s" % self.rdir)
        return q

    @cached_property
    def seed(self):
        return int(self.field["seed"])

    @cached_property
    def ell(self):
        e, lam = float(self.field["ell"]), float(self.flow["lambda"])
        if abs(e - lam) > 1e-9 * abs(lam):
            raise ReduceError("field.json ell=%.17g differs from flow_stats.json lambda=%.17g in %s" % (e, lam, self.rdir))
        return e

    @property
    def L1(self):
        return float(self.flow["L1"])

    @property
    def x0(self):
        return self.pass_.x0

    @property
    def qbar(self):
        return float(self.flow["qbar"])

    def key_planes(self, tolerate_missing=False):
        """{name: (x, index or None)} for Xa, Xb, Xq in this pass's planes."""
        out = {}
        for k, x in key_planes(self.x0, self.L1, self.ell).items():
            out[k] = (x, plane_index(self.pass_.planes, x, k, tolerate_missing))
        return out
