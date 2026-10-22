#!/usr/bin/env python3
"""SF-37 analysis library (python3 + numpy + scipy only; numpy 1.26 .. 2.x, scipy 1.11 .. 1.18).

Shared by digest_v1v2.py, digest_v4v5.py and selftest_sf37.py.  Everything here implements the reading rules
RR1-RR15 of the SF-37 activation row literally; no threshold lives here except where a function takes it as a
documented argument whose default is the pre-registered value.

Contents
  * input handling   : Manifest (path + sha256 of the files actually read), RunData (lazy run directory),
                       Campaign (C-LAYOUT: ROOT/<part>/<run>/, <run>.log, <run>.json, exit_codes.tsv), Design
  * moments          : tracker-identical sequential weighted moments (RR1, RR2), pair separation
  * paired bootstrap : block partition, committed index matrix, shifted per-block sums (RR3)
  * orders / rules   : observed orders (RR4), V1 / V2 / production lambda/h cell rules, weighted quantiles
  * output           : OutWriter (csv / markdown stamped with versions, bootstrap sha256 and the manifest)
"""
import collections
import csv
import hashlib
import json
import math
import os
import platform
import re

import numpy as np

# ------------------------------------------------------------------------------------------------------------
# constants (pre-registered)
# ------------------------------------------------------------------------------------------------------------
BOOT_SEED = 20261017
BOOT_B = 2000
BOOT_NBLOCKS = 16
BOOT_SHA256 = "16c1a5b9d9fe062576d5f18fc9d54127db76472c7fbe64493201d8cffc66af39"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACT_DIR = os.path.dirname(SCRIPT_DIR)
BOOT_PATH = os.path.join(ARTIFACT_DIR, "analysis", "bootstrap_blocks.npy")

EPS_CODES = (10, 15, 20)  # eps 1, 1.5, 2 -> sigma^2 1, 2.25, 4
SEEDS = (3101, 3102, 3103)
V2_NS = (32, 40, 64, 80, 128, 256, 512)
V1_NS = (256, 512)
V1_TOLS = (1e-6, 1e-7, 1e-8, 1e-9, 1e-10)
TOL_PRIMARY = 1e-8
TOL_REF = 1e-10
PLANES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
TIMES = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
N_REF = 512
N_TRIPLE = (128, 256, 512)
LAMBDA_CANDIDATES = ((32, 8), (40, 10), (64, 16), (80, 20), (128, 32), (256, 64))  # (N, lambda/h)

V1_REL = 1e-3
V2_REL = 2e-2
V2_ORDER = 1.5
LAMH_REL = 5e-2
NSE = 2.0
FAR_FRACTION = 0.1
V4_ORDER = 1.8

REQUIRED_FILES = ("track.json", "weights.npy", "status_plane.npy", "status_time.npy", "planes_x23.npy",
                  "planes_tau.npy", "planes_I.npy", "times_x.npy", "times_I.npy", "moments_planes.csv",
                  "moments_time.csv")


def eps_of(code):
    return code / 10.0


def sigma2_of(code):
    return (code / 10.0) ** 2


def fmt_tol(t):
    """1e-06 -> '1e-6' (run-name spelling)."""
    s = "%.0e" % t
    m, e = s.split("e")
    return "%se%d" % (m, int(e))


def v2_name(e, s, n):
    return "v2_e%d_s%d_N%d" % (e, s, n)


def v1_name(e, s, n, tol, hdiv):
    return "v1_e%d_s%d_N%d_tol%s_h%d" % (e, s, n, fmt_tol(tol), hdiv)


_RE_V2 = re.compile(r"^v2_e(\d+)_s(\d+)_N(\d+)$")
_RE_V1 = re.compile(r"^v1_e(\d+)_s(\d+)_N(\d+)_tol([0-9.]+e[-+]?\d+)_h(\d+)$")


def parse_run_name(name):
    """-> ('v2', e, s, n) or ('v1', e, s, n, tol, hdiv) or None."""
    m = _RE_V2.match(name)
    if m:
        return ("v2", int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = _RE_V1.match(name)
    if m:
        return ("v1", int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4)), int(m.group(5)))
    return None


# ------------------------------------------------------------------------------------------------------------
# versions, hashing, manifest
# ------------------------------------------------------------------------------------------------------------
def versions():
    d = {"python": platform.python_version(), "numpy": np.__version__}
    try:
        import scipy
        d["scipy"] = scipy.__version__
    except Exception:  # pragma: no cover
        d["scipy"] = "unavailable"
    return d


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class Manifest:
    """Records every input file actually read; hashes lazily (streaming) in entries()."""

    def __init__(self, root=None):
        self.root = root
        self._paths = collections.OrderedDict()
        self._entries = None

    def add(self, path):
        p = os.path.abspath(str(path))
        if p not in self._paths:
            self._paths[p] = None
            self._entries = None

    def entries(self):
        """list of (display path, sha256, bytes), sorted by path."""
        if self._entries is None:
            out = []
            for p in sorted(self._paths):
                shown = os.path.relpath(p, self.root) if self.root else p
                out.append((shown, sha256_file(p), os.path.getsize(p)))
            self._entries = out
        return self._entries

    def digest_of_digests(self):
        h = hashlib.sha256()
        for shown, sha, size in self.entries():
            h.update(("%s %s %d\n" % (shown, sha, size)).encode())
        return h.hexdigest()

    def __len__(self):
        return len(self._paths)


# ------------------------------------------------------------------------------------------------------------
# bootstrap blocks
# ------------------------------------------------------------------------------------------------------------
def load_bootstrap(path=None, manifest=None):
    """Load the COMMITTED index matrix and verify the pre-registered sha256 (never regenerates it)."""
    path = path or BOOT_PATH
    m = np.load(path, allow_pickle=False)
    if manifest is not None:
        manifest.add(path)
    if m.shape != (BOOT_B, BOOT_NBLOCKS):
        raise ValueError("bootstrap matrix has shape %s, expected (%d, %d)" % (m.shape, BOOT_B, BOOT_NBLOCKS))
    sha = hashlib.sha256(m.astype("<i8").tobytes()).hexdigest()
    if sha != BOOT_SHA256:
        raise ValueError("bootstrap matrix sha256 %s != pre-registered %s" % (sha, BOOT_SHA256))
    return m.astype(np.int64)


def runtime_bootstrap(nb):
    """Sensitivity matrix generated at runtime (4 or 64 blocks); nb=16 reproduces the committed matrix."""
    return np.random.default_rng(BOOT_SEED).integers(0, nb, size=(BOOT_B, nb)).astype(np.int64)


def counts_from_idx(idx, nb):
    """(R, nb) float multiplicities of the blocks in each resample."""
    idx = np.asarray(idx)
    return (idx[:, :, None] == np.arange(nb)[None, None, :]).sum(axis=1).astype(np.float64)


_BLOCK_CACHE = {}


def block_ids(Np, g=4):
    """Block of particle p = m2 M + m3: (m2 // side) * g + m3 // side, side = M/g (g = 4: 64x64 for M = 256)."""
    key = (int(Np), int(g))
    if key not in _BLOCK_CACHE:
        M = int(round(math.sqrt(Np)))
        if M * M != Np:
            raise ValueError("Np = %d is not a square" % Np)
        if M % g != 0:
            raise ValueError("M = %d is not divisible by g = %d" % (M, g))
        side = M // g
        p = np.arange(Np)
        _BLOCK_CACHE[key] = ((p // M) // side) * g + (p % M) // side
    return _BLOCK_CACHE[key]


# ------------------------------------------------------------------------------------------------------------
# run directory
# ------------------------------------------------------------------------------------------------------------
class RunData:
    """One particle directory of an `inlet_slab --track forward --save-particles` run (lazy)."""

    def __init__(self, path, manifest=None, name=None):
        self.path = str(path)
        self.name = name or os.path.basename(self.path.rstrip("/"))
        self.manifest = manifest
        self._mem = {}
        self._track = None
        self._pmask = None
        self._tmask = None

    def fpath(self, fname):
        return os.path.join(self.path, fname)

    def exists(self, fname):
        return os.path.isfile(self.fpath(fname))

    @property
    def complete(self):
        return all(self.exists(f) for f in REQUIRED_FILES)

    def missing_files(self):
        return [f for f in REQUIRED_FILES if not self.exists(f)]

    def _note(self, fname):
        if self.manifest is not None:
            self.manifest.add(self.fpath(fname))

    def track(self):
        if self._track is None:
            self._note("track.json")
            with open(self.fpath("track.json")) as f:
                self._track = json.load(f)
        return self._track

    def mem(self, fname):
        """array of the run directory, memory-mapped read-only (lazy; cached until release())."""
        if fname not in self._mem:
            self._note(fname)
            self._mem[fname] = np.load(self.fpath(fname), mmap_mode="r", allow_pickle=False)
        return self._mem[fname]

    def mmap(self, fname):
        """memory-mapped array (not cached here)."""
        self._note(fname)
        return np.load(self.fpath(fname), mmap_mode="r", allow_pickle=False)

    def read_array(self, fname):
        """array read without caching (covectors, seed_v)."""
        self._note(fname)
        return np.load(self.fpath(fname), allow_pickle=False)

    def release(self):
        self._mem.clear()
        self._pmask = None
        self._tmask = None

    @property
    def planes(self):
        return np.asarray(self.track()["planes"], dtype=np.float64)

    @property
    def times(self):
        return np.asarray(self.track()["times"], dtype=np.float64)

    @property
    def P(self):
        return len(self.track()["planes"])

    @property
    def T(self):
        return len(self.track()["times"])

    @property
    def w(self):
        return np.asarray(self.mem("weights.npy"), dtype=np.float64)

    @property
    def Np(self):
        return int(self.w.shape[0])

    @property
    def M(self):
        M = int(round(math.sqrt(self.Np)))
        if M * M != self.Np:
            raise ValueError("%s: Np = %d is not a square" % (self.name, self.Np))
        return M

    def plane_mask(self):
        """RR1: contributors = finite record at the plane (the particle landed there in this run)."""
        if self._pmask is None:
            X = self.mem("planes_x23.npy")
            self._pmask = np.isfinite(X[:, :, 0]) & np.isfinite(X[:, :, 1])
        return self._pmask

    def time_mask(self):
        if self._tmask is None:
            X = self.mem("times_x.npy")
            self._tmask = np.isfinite(X).all(axis=2)
        return self._tmask


# ------------------------------------------------------------------------------------------------------------
# tracker-identical moments (RR1, RR2)
# ------------------------------------------------------------------------------------------------------------
def seq_sum(a):
    """Sequential (index-order) sum, as the tracker's accumulation loop."""
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float(np.cumsum(a)[-1])


def seq_wmean_var(w, v):
    """Weighted mean and BIASED weighted variance over contributors (already selected, index order):
    mean = sum w v / sum w ; var = sum w (v - mean)^2 / sum w, sequentially accumulated.  NaN if empty or W <= 0."""
    w = np.asarray(w, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if w.size == 0:
        return float("nan"), float("nan")
    W = seq_sum(w)
    if not (W > 0.0):
        return float("nan"), float("nan")
    mean = seq_sum(w * v) / W
    d = v - mean
    return mean, seq_sum(w * d * d) / W


def _plane_core(run, mask=None):
    """mean/var of x2, x3, tau per plane over the contributors (or over `mask`)."""
    mp = run.plane_mask() if mask is None else mask
    X = run.mem("planes_x23.npy")
    tau = run.mem("planes_tau.npy")
    w = run.w
    P = run.P
    out = {k: np.full(P, np.nan) for k in ("mean_x2", "var_x2", "mean_x3", "var_x3", "mean_tau", "var_tau")}
    n = np.zeros(P, dtype=np.int64)
    for m in range(P):
        sel = mp[:, m]
        n[m] = int(sel.sum())
        ws = w[sel]
        out["mean_x2"][m], out["var_x2"][m] = seq_wmean_var(ws, X[sel, m, 0])
        out["mean_x3"][m], out["var_x3"][m] = seq_wmean_var(ws, X[sel, m, 1])
        out["mean_tau"][m], out["var_tau"][m] = seq_wmean_var(ws, tau[sel, m])
    out["n"] = n
    return out


def _time_core(run, mask=None):
    mt = run.time_mask() if mask is None else mask
    X = run.mem("times_x.npy")
    w = run.w
    T = run.T
    out = {}
    for k in range(3):
        out["mean_x%d" % (k + 1)] = np.full(T, np.nan)
        out["var_x%d" % (k + 1)] = np.full(T, np.nan)
    n = np.zeros(T, dtype=np.int64)
    for j in range(T):
        sel = mt[:, j]
        n[j] = int(sel.sum())
        ws = w[sel]
        for k in range(3):
            out["mean_x%d" % (k + 1)][j], out["var_x%d" % (k + 1)][j] = seq_wmean_var(ws, X[sel, j, k])
    out["n"] = n
    return out


def plane_moments(run):
    """Reproduces moments_planes.csv: x1, contributing, non_ok, mean/var x2, x3, tau, rms_ev, max_ev."""
    core = _plane_core(run)
    mp = run.plane_mask()
    I = run.mem("planes_I.npy")
    w = run.w
    st = run.mem("status_plane.npy")
    P = run.P
    rms = np.full(P, np.nan)
    mx = np.full(P, np.nan)
    for m in range(P):
        sel = mp[:, m]
        if sel.sum() == 0:
            continue
        e = 1.0 - np.exp(-I[sel, m])
        ws = w[sel]
        W = seq_sum(ws)
        S = seq_sum(ws * e * e)
        rms[m] = math.sqrt(S / W) if W > 0.0 else float("nan")
        mx[m] = float(np.max(np.abs(e)))
    out = dict(core)
    out["x1"] = run.planes
    out["contributing"] = core["n"]
    out["non_ok"] = np.full(P, int((st != 0).sum()), dtype=np.int64)
    out["rms_ev"] = rms
    out["max_ev"] = mx
    return out


def time_moments(run):
    """Reproduces moments_time.csv: t, mean_x/y/z, var_x/y/z, active."""
    core = _time_core(run)
    out = {"t": run.times, "active": core["n"]}
    for k in range(3):
        out["mean_x%d" % (k + 1)] = core["mean_x%d" % (k + 1)]
        out["var_x%d" % (k + 1)] = core["var_x%d" % (k + 1)]
    return out


def read_moments_planes_csv(path):
    with open(path) as f:
        head = f.readline().strip().split(",")
        rows = [ln.strip().split(",") for ln in f if ln.strip()]
    arr = np.array([[float(x) for x in r] for r in rows], dtype=np.float64).reshape(len(rows), len(head))
    return {h: arr[:, i] for i, h in enumerate(head)}


def read_moments_time_csv(path):
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    lines = [ln for ln in lines if not ln.startswith("#")]
    head = lines[0].split(",")
    arr = np.array([[float(x) for x in ln.split(",")] for ln in lines[1:]], dtype=np.float64).reshape(len(lines) - 1, len(head))
    return {h: arr[:, i] for i, h in enumerate(head)}


def max_rel_diff(a, b):
    """max over entries of |a - b| / |b| (|a - b| where b == 0); NaN must coincide."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        return float("inf")
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    if (nan_a != nan_b).any():
        return float("inf")
    ok = ~nan_a
    if not ok.any():
        return 0.0
    d = np.abs(a[ok] - b[ok])
    den = np.abs(b[ok])
    r = np.where(den > 0, d / np.where(den > 0, den, 1.0), d)
    return float(r.max())


def check_moments_reproduction(run, tol=1e-12):
    """Compare the library moments with moments_planes.csv / moments_time.csv of the run.
    -> (ok, worst relative difference, per-column dict)."""
    run._note("moments_planes.csv")
    run._note("moments_time.csv")
    pm = plane_moments(run)
    pc = read_moments_planes_csv(run.fpath("moments_planes.csv"))
    tm = time_moments(run)
    tc = read_moments_time_csv(run.fpath("moments_time.csv"))
    cols = {}
    for k in ("x1", "contributing", "non_ok", "mean_x2", "var_x2", "mean_x3", "var_x3", "mean_tau", "var_tau",
              "rms_ev", "max_ev"):
        cols["planes." + k] = max_rel_diff(pm[k], pc[k])
    pairs = (("t", "t"), ("mean_x1", "mean_x"), ("mean_x2", "mean_y"), ("mean_x3", "mean_z"),
             ("var_x1", "var_x"), ("var_x2", "var_y"), ("var_x3", "var_z"), ("active", "active"))
    for mine, theirs in pairs:
        cols["time." + mine] = max_rel_diff(tm[mine], tc[theirs])
    worst = max(cols.values()) if cols else 0.0
    return worst <= tol, worst, cols


# ------------------------------------------------------------------------------------------------------------
# series (quantities of RR2) and derived series
# ------------------------------------------------------------------------------------------------------------
PLANE_SERIES = ("sigma_T2", "var_tau")
TIME_SERIES = ("var_x1", "var_x2", "var_x3")
DPLANE_SERIES = ("dsigma_T2_dx1",)
DTIME_SERIES = ("dvar_x1_dt", "dvar_x2_dt", "dvar_x3_dt")
ALL_SERIES = PLANE_SERIES + TIME_SERIES + DPLANE_SERIES + DTIME_SERIES
SERIES_KIND = {}
for _n in PLANE_SERIES:
    SERIES_KIND[_n] = "plane"
for _n in TIME_SERIES:
    SERIES_KIND[_n] = "time"
for _n in DPLANE_SERIES:
    SERIES_KIND[_n] = "dplane"
for _n in DTIME_SERIES:
    SERIES_KIND[_n] = "dtime"
DERIVED_FROM = {"dsigma_T2_dx1": "sigma_T2", "dvar_x1_dt": "var_x1", "dvar_x2_dt": "var_x2", "dvar_x3_dt": "var_x3"}


def add_derived(series, planes, times):
    """Adds the finite-difference series (along the last axis) to a dict of arrays."""
    out = dict(series)
    dx = np.diff(np.asarray(planes, dtype=np.float64))
    dt = np.diff(np.asarray(times, dtype=np.float64))
    for name, base in DERIVED_FROM.items():
        if base in series:
            s = np.asarray(series[base])
            step = dx if SERIES_KIND[name] == "dplane" else dt
            if s.shape[-1] == step.shape[0] + 1:
                out[name] = (s[..., 1:] - s[..., :-1]) / step
    return out


def series_axis_labels(name, planes, times):
    """Human-readable axis labels of a series (x1 values, t values, or the interval)."""
    kind = SERIES_KIND[name]
    g = lambda v: "%g" % v
    if kind == "plane":
        return [g(v) for v in planes]
    if kind == "time":
        return [g(v) for v in times]
    if kind == "dplane":
        return ["%s->%s" % (g(planes[i]), g(planes[i + 1])) for i in range(len(planes) - 1)]
    return ["%s->%s" % (g(times[i]), g(times[i + 1])) for i in range(len(times) - 1)]


def observed_series(run, mask_p=None, mask_t=None):
    """Observed (full-sample) series of a run over its own contributors (or the given masks)."""
    pc = _plane_core(run, mask_p)
    tc = _time_core(run, mask_t)
    s = {"sigma_T2": 0.5 * (pc["var_x2"] + pc["var_x3"]), "var_tau": pc["var_tau"]}
    for k in (1, 2, 3):
        s["var_x%d" % k] = tc["var_x%d" % k]
    return add_derived(s, run.planes, run.times)


def full_sample_means(run):
    """Per-(position, column) weighted full-sample means used as bootstrap shifts: plane (P,3) = (x2, x3, tau),
    time (T,3) = (x1, x2, x3)."""
    pc = _plane_core(run)
    tc = _time_core(run)
    mp = np.stack([pc["mean_x2"], pc["mean_x3"], pc["mean_tau"]], axis=1)
    mt = np.stack([tc["mean_x1"], tc["mean_x2"], tc["mean_x3"]], axis=1)
    return np.nan_to_num(mp, nan=0.0), np.nan_to_num(mt, nan=0.0)


def _block_sums(w, mask, X, shift, ids, nb):
    """Per-block shifted weighted sums for X (Np, K, C): W (nb, K), A (nb, K, C) = sum w y, B (nb, K, C) =
    sum w y^2 with y = X - shift.  Masked-out entries are excluded; a NaN among contributors propagates."""
    Np, K, C = X.shape
    W = np.zeros((nb, K))
    A = np.zeros((nb, K, C))
    B = np.zeros((nb, K, C))
    for k in range(K):
        sel = mask[:, k]
        wm = np.where(sel, w, 0.0)
        W[:, k] = np.bincount(ids, weights=wm, minlength=nb)
        for c in range(C):
            y = np.where(sel, X[:, k, c] - shift[k, c], 0.0)
            wy = wm * y
            A[:, k, c] = np.bincount(ids, weights=wy, minlength=nb)
            B[:, k, c] = np.bincount(ids, weights=wy * y, minlength=nb)
    return W, A, B


def _resampled_var(W, A, B, counts):
    """Variance on each resample: counts (R, nb) multiplicities.  Returns var (R, K, C)."""
    Ws = counts @ W
    As = np.einsum("rb,bkc->rkc", counts, A)
    Bs = np.einsum("rb,bkc->rkc", counts, B)
    with np.errstate(divide="ignore", invalid="ignore"):
        Wd = Ws[:, :, None]
        m = As / Wd
        var = Bs / Wd - m * m
    var = np.where(Wd > 0, var, np.nan)
    return var


def resampled_series(run, shifts, counts, g=4, mask_p=None, mask_t=None):
    """Series on every bootstrap resample (R, len): each resample recomputes the quantity over the run's OWN
    contributors in the resampled multiset with the weights times the multiplicity, renormalized (RR3).
    `shifts` = (plane (P,3), time (T,3)) full-sample means of the reference."""
    mp = run.plane_mask() if mask_p is None else mask_p
    mt = run.time_mask() if mask_t is None else mask_t
    Np = run.Np
    ids = block_ids(Np, g)
    nb = g * g
    w = run.w
    Xp = np.concatenate([run.mem("planes_x23.npy"), run.mem("planes_tau.npy")[:, :, None]], axis=2)
    Wp, Ap, Bp = _block_sums(w, mp, Xp, shifts[0], ids, nb)
    vp = _resampled_var(Wp, Ap, Bp, counts)  # (R, P, 3)
    Xt = run.mem("times_x.npy")
    Wt, At, Bt = _block_sums(w, mt, Xt, shifts[1], ids, nb)
    vt = _resampled_var(Wt, At, Bt, counts)  # (R, T, 3)
    s = {"sigma_T2": 0.5 * (vp[:, :, 0] + vp[:, :, 1]), "var_tau": vp[:, :, 2]}
    for k in (1, 2, 3):
        s["var_x%d" % k] = vt[:, :, k - 1]
    return add_derived(s, run.planes, run.times)


def block_observed_series(run, g=4):
    """The observed series recomputed from the per-block sums with all multiplicities 1 (consistency check of the
    shifted-sum machinery against the sequential moments)."""
    counts = np.ones((1, g * g))
    sh = full_sample_means(run)
    r = resampled_series(run, sh, counts, g)
    return {k: v[0] for k, v in r.items()}


def paired_stats(a, b, counts, g=4, mask_p=None, mask_t=None):
    """Paired block-bootstrap comparison of test run `a` against reference run `b` (same realization, same seed
    lattice).  Shift = full-sample weighted mean of the reference.  Returns {series: dict(a, b, D, se)}."""
    if a.Np != b.Np:
        raise ValueError("paired runs %s / %s have different Np" % (a.name, b.name))
    if not (np.array_equal(a.planes, b.planes) and np.array_equal(a.times, b.times)):
        raise ValueError("paired runs %s / %s have different planes/times" % (a.name, b.name))
    sh = full_sample_means(b)
    oa = observed_series(a, mask_p, mask_t)
    ob = observed_series(b, mask_p, mask_t)
    ra = resampled_series(a, sh, counts, g, mask_p, mask_t)
    rb = resampled_series(b, sh, counts, g, mask_p, mask_t)
    out = {}
    for k in oa:
        Ds = ra[k] - rb[k]
        with np.errstate(invalid="ignore"):
            se = np.std(Ds, axis=0, ddof=1)
        out[k] = {"a": oa[k], "b": ob[k], "D": oa[k] - ob[k], "se": se}
    return out


def pair_separation(a, b, kind="plane"):
    """RR2 paired separation of test `a` vs reference `b` over the common contributors.
    dx_T^2 = ((x2a - x2b)^2 + (x3a - x3b)^2)/2, MS weighted with b's weights renormalized over the common set.
    -> dict(ms, rms, sigma_T2_b, rel = rms/sigma_T(b), ratio = ms/sigma_T2(b), n_common, w_common)."""
    wb = b.w
    if kind == "plane":
        Xa, Xb = a.mem("planes_x23.npy"), b.mem("planes_x23.npy")
        com = a.plane_mask() & b.plane_mask()
        d2 = 0.5 * ((Xa[:, :, 0] - Xb[:, :, 0]) ** 2 + (Xa[:, :, 1] - Xb[:, :, 1]) ** 2)
        pc = _plane_core(b)
        s2b = 0.5 * (pc["var_x2"] + pc["var_x3"])
    else:
        Xa, Xb = a.mem("times_x.npy"), b.mem("times_x.npy")
        com = a.time_mask() & b.time_mask()
        d2 = 0.5 * ((Xa[:, :, 1] - Xb[:, :, 1]) ** 2 + (Xa[:, :, 2] - Xb[:, :, 2]) ** 2)
        tc = _time_core(b)
        s2b = 0.5 * (tc["var_x2"] + tc["var_x3"])
    K = d2.shape[1]
    ms = np.full(K, np.nan)
    nc = np.zeros(K, dtype=np.int64)
    wc = np.zeros(K)
    for k in range(K):
        sel = com[:, k]
        nc[k] = int(sel.sum())
        if nc[k] == 0:
            continue
        W = seq_sum(wb[sel])
        wc[k] = W
        if W > 0.0:
            ms[k] = seq_sum(wb[sel] * d2[sel, k]) / W
    with np.errstate(invalid="ignore", divide="ignore"):
        rms = np.sqrt(ms)
        rel = rms / np.sqrt(s2b)
        ratio = ms / s2b
    return {"ms": ms, "rms": rms, "sigma_T2_b": s2b, "rel": rel, "ratio": ratio, "n_common": nc, "w_common": wc}


def far_index(rel, frac=FAR_FRACTION):
    """RR2 'far plane' of a pair: first index with RMS(dx_T) > frac * sigma_T(b) (and every later one); None if none."""
    rel = np.asarray(rel, dtype=np.float64)
    for i in range(rel.shape[0]):
        if np.isfinite(rel[i]) and rel[i] > frac:
            return i
    return None


def bitwise_identical_particles(a, b, names=("planes_x23.npy", "planes_tau.npy", "planes_I.npy", "times_x.npy",
                                              "times_I.npy")):
    """Per-particle boolean: every named record bitwise identical (NaN payloads included)."""
    Np = a.Np
    ident = np.ones(Np, dtype=bool)
    for n in names:
        xa = np.ascontiguousarray(a.mem(n), dtype="<f8")
        xb = np.ascontiguousarray(b.mem(n), dtype="<f8")
        if xa.shape != xb.shape:
            return np.zeros(Np, dtype=bool)
        eq = (xa.view("<i8") == xb.view("<i8")).reshape(Np, -1).all(axis=1)
        ident &= eq
    return ident


def statuses_identical(a, b):
    return bool(np.array_equal(a.mem("status_plane.npy"), b.mem("status_plane.npy")) and
                np.array_equal(a.mem("status_time.npy"), b.mem("status_time.npy")))


def effective_sample_size(w):
    w = np.asarray(w, dtype=np.float64)
    s2 = float((w * w).sum())
    return float(w.sum()) ** 2 / s2 if s2 > 0 else float("nan")


# ------------------------------------------------------------------------------------------------------------
# orders, quantiles, rules (RR4, RR5, V1, V2)
# ------------------------------------------------------------------------------------------------------------
def order_triple(qa, qb, qc):
    """Observed order on a ratio-2 triple (coarse a, b, fine c): log2(|qa - qb| / |qb - qc|); nan iff a
    difference is exactly 0, the two differences have opposite signs, or an input is not finite."""
    if not (math.isfinite(qa) and math.isfinite(qb) and math.isfinite(qc)):
        return float("nan")
    d1 = qa - qb
    d2 = qb - qc
    if d1 == 0.0 or d2 == 0.0 or (d1 > 0.0) != (d2 > 0.0):
        return float("nan")
    return math.log2(abs(d1) / abs(d2))


def order_pair(e_coarse, e_fine):
    """Order of an error quantity vanishing in the limit on a pair (N, 2N): log2(e_N / e_2N)."""
    if not (math.isfinite(e_coarse) and math.isfinite(e_fine)):
        return float("nan")
    if e_coarse <= 0.0 or e_fine < 0.0:
        return float("nan")
    if e_fine == 0.0:
        return float("inf")
    return math.log2(e_coarse / e_fine)


def wquantile(values, weights, q):
    """Weighted quantile: the smallest value whose cumulative weight (ascending values) is >= q W; entries with
    non-finite value or non-positive weight are ignored."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    v = v[ok]
    w = w[ok]
    o = np.argsort(v, kind="stable")
    vs = v[o]
    cw = np.cumsum(w[o])
    i = int(np.searchsorted(cw, q * cw[-1], side="left"))
    return float(vs[min(i, vs.shape[0] - 1)])


def _fin(*xs):
    return all(math.isfinite(x) for x in xs)


def zscore(D, se):
    if not _fin(D, se):
        return float("nan")
    if se == 0.0:
        return 0.0 if D == 0.0 else math.copysign(float("inf"), D)
    return D / se


def v1_cell(D, qref, se, rel=V1_REL, nse=NSE):
    """V1: pass iff |D| <= max(rel |Qref|, nse SE_D).  branch: det (relative branch), stat (nse SE_D branch), fail.
    binding = the larger threshold ('rel' / 'stat')."""
    if not _fin(D, qref, se):
        return {"verdict": "not_evaluable", "branch": "not_evaluable", "binding": "", "thr_rel": float("nan"),
                "thr_stat": float("nan"), "z": float("nan")}
    thr_rel = rel * abs(qref)
    thr_stat = nse * se
    aD = abs(D)
    if aD <= thr_rel:
        branch = "det"
    elif aD <= thr_stat:
        branch = "stat"
    else:
        branch = "fail"
    return {"verdict": "fail" if branch == "fail" else "pass", "branch": branch,
            "binding": "rel" if thr_rel >= thr_stat else "stat", "thr_rel": thr_rel, "thr_stat": thr_stat,
            "z": zscore(D, se)}


def v2_cell(q_c, q_m, q_f, se_mf, se_cm=float("nan"), rel=V2_REL, order=V2_ORDER, nse=NSE):
    """V2 on a ratio-2 triple (coarse 128, middle 256, fine 512): pass iff observed order >= `order` or
    |Q_m - Q_f| <= max(rel |Q_f|, nse SE_D(m, f)).  Not evaluable if any of the three values or SE_D is missing.
    branch (first satisfied): det (relative), order, stat (nse SE_D), fail.  All three booleans are reported."""
    if not _fin(q_c, q_m, q_f, se_mf):
        return {"verdict": "not_evaluable", "branch": "not_evaluable", "p": float("nan"), "order_ok": False,
                "det_ok": False, "stat_ok": False, "thr_rel": float("nan"), "thr_stat": float("nan"),
                "z": float("nan"), "richardson": float("nan"), "noise_flag": False, "D": float("nan")}
    p = order_triple(q_c, q_m, q_f)
    D = q_m - q_f
    aD = abs(D)
    thr_rel = rel * abs(q_f)
    thr_stat = nse * se_mf
    order_ok = math.isfinite(p) and p >= order
    det_ok = aD <= thr_rel
    stat_ok = aD <= thr_stat
    branch = "det" if det_ok else ("order" if order_ok else ("stat" if stat_ok else "fail"))
    rich = float("nan")
    if math.isfinite(p) and p > 0 and q_f != 0.0:
        rich = aD / ((2.0 ** p - 1.0) * abs(q_f))
    noise = False
    if math.isfinite(p):
        noise = (abs(q_c - q_m) < nse * se_cm) if math.isfinite(se_cm) else False
        noise = noise or (aD < thr_stat)
    return {"verdict": "pass" if branch != "fail" else "fail", "branch": branch, "p": p, "order_ok": order_ok,
            "det_ok": det_ok, "stat_ok": stat_ok, "thr_rel": thr_rel, "thr_stat": thr_stat,
            "z": zscore(D, se_mf), "richardson": rich, "noise_flag": noise, "D": D}


def lambda_h_cell(qN, qref, se, rel=LAMH_REL, nse=NSE):
    """Production lambda/h inequality |Q(N) - Q(512)| <= max(rel |Q(512)|, nse SE_D(N, 512))."""
    if not _fin(qN, qref, se):
        return {"verdict": "not_evaluable", "branch": "not_evaluable", "thr_rel": float("nan"),
                "thr_stat": float("nan"), "z": float("nan"), "D": float("nan"), "binding": ""}
    D = qN - qref
    thr_rel = rel * abs(qref)
    thr_stat = nse * se
    aD = abs(D)
    branch = "det" if aD <= thr_rel else ("stat" if aD <= thr_stat else "fail")
    return {"verdict": "fail" if branch == "fail" else "pass", "branch": branch, "thr_rel": thr_rel,
            "thr_stat": thr_stat, "z": zscore(D, se), "D": D, "binding": "rel" if thr_rel >= thr_stat else "stat"}


def production_lambda_h(candidates, status):
    """RR5.  candidates: ascending list of lambda/h.  status[c] in {'pass', 'fail', 'missing'} (a candidate is
    'missing' ONLY if a needed run has no data).  Literal primary = smallest candidate with status 'pass';
    monotone variant = smallest candidate such that it and every larger candidate pass.  None = 'none'."""
    primary = None
    for c in candidates:
        if status.get(c) == "pass":
            primary = c
            break
    monotone = None
    for i, c in enumerate(candidates):
        if all(status.get(x) == "pass" for x in candidates[i:]):
            monotone = c
            break
    return primary, monotone


def ladder_of_ns(ns):
    return sorted(ns)


# ------------------------------------------------------------------------------------------------------------
# V5 decorrelation fits (RR8)
# ------------------------------------------------------------------------------------------------------------
def fit_decorrelation(x1, rms, rel, lo=1e-12, frac=FAR_FRACTION, min_planes=3):
    """Fits of log RMS(dx_T) on the eligible planes (lo < RMS < frac sigma_T, i.e. rel = RMS/sigma_T < frac):
    exponential = least squares on x1, algebraic = least squares on log x1.  < min_planes eligible -> 'insufficient'.
    -> dict(name, n, rss_exp, rss_alg, slope_exp, slope_alg)."""
    x1 = np.asarray(x1, dtype=np.float64)
    rms = np.asarray(rms, dtype=np.float64)
    rel = np.asarray(rel, dtype=np.float64)
    ok = np.isfinite(rms) & np.isfinite(rel) & (rms > lo) & (rel < frac) & (x1 > 0)
    n = int(ok.sum())
    out = {"name": "insufficient", "n": n, "rss_exp": float("nan"), "rss_alg": float("nan"),
           "slope_exp": float("nan"), "slope_alg": float("nan")}
    if n < min_planes:
        return out
    y = np.log(rms[ok])
    res = {}
    for key, xv in (("exp", x1[ok]), ("alg", np.log(x1[ok]))):
        A = np.stack([np.ones(n), xv], axis=1)
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        r = y - A @ coef
        res[key] = (float(r @ r), float(coef[1]))
    out["rss_exp"], out["slope_exp"] = res["exp"]
    out["rss_alg"], out["slope_alg"] = res["alg"]
    if out["rss_exp"] < out["rss_alg"]:
        out["name"] = "exponential"
    elif out["rss_alg"] < out["rss_exp"]:
        out["name"] = "algebraic"
    else:
        out["name"] = "tie"
    return out


def amplitude_name(case_names):
    """Majority over the evaluable (seed, N) cases ('exponential' / 'algebraic'); 'mixed' on a tie; 'insufficient'
    if fewer than 2 evaluable."""
    ev = [c for c in case_names if c in ("exponential", "algebraic")]
    if len(ev) < 2:
        return "insufficient"
    ne, na = ev.count("exponential"), ev.count("algebraic")
    if ne > na:
        return "exponential"
    if na > ne:
        return "algebraic"
    return "mixed"


# ------------------------------------------------------------------------------------------------------------
# log / json parsing
# ------------------------------------------------------------------------------------------------------------
def parse_field_line(text):
    """The first `FIELD gaussian (SF-18) ...` line of a driver log -> dict of key=value tokens (floats where
    numeric), or None."""
    for ln in text.splitlines():
        if ln.startswith("FIELD gaussian"):
            head = ln.split("|")[0]
            d = {}
            for k, v in re.findall(r"(\w+)=(\S+)", head):
                try:
                    d[k] = float(v)
                except ValueError:
                    d[k] = v
            return d
    return None


PASS_COUNTERS = ("nfev", "accepted_steps", "rejected_steps", "landing_steps", "landing_rejected",
                 "landings_discarded", "recrossings", "backflow_encounters")


def pass_counters(track, which):
    """track.json plane_pass / time_pass -> dict of the integer counters + status_hist."""
    p = track.get(which, {})
    d = {k: p.get(k) for k in PASS_COUNTERS}
    d["status_hist"] = p.get("status_hist", {})
    return d


# ------------------------------------------------------------------------------------------------------------
# campaign layout (C-LAYOUT) and design
# ------------------------------------------------------------------------------------------------------------
class Campaign:
    """ROOT/<part>/<run>/ (particle dir), <run>.log, <run>.json, <run>.time, exit_codes.tsv."""

    def __init__(self, root, manifest, cache=12):
        self.root = os.path.abspath(root)
        self.manifest = manifest
        self._cache = collections.OrderedDict()
        self._cache_n = cache
        self._exit = {}

    def part_dir(self, part):
        return os.path.join(self.root, part)

    def discover(self):
        """{part: sorted run names that are directories with a recognized name}."""
        out = {}
        for part in ("v2", "v1"):
            d = self.part_dir(part)
            names = []
            if os.path.isdir(d):
                for e in sorted(os.listdir(d)):
                    if os.path.isdir(os.path.join(d, e)) and parse_run_name(e) is not None:
                        names.append(e)
            out[part] = names
        return out

    def exit_codes(self, part):
        if part not in self._exit:
            codes = {}
            p = os.path.join(self.part_dir(part), "exit_codes.tsv")
            if os.path.isfile(p):
                self.manifest.add(p)
                with open(p) as f:
                    for ln in f:
                        t = ln.rstrip("\n").split("\t")
                        if len(t) >= 2:
                            try:
                                codes[t[0].strip()] = int(t[-1].strip())
                            except ValueError:
                                pass
            self._exit[part] = codes
        return self._exit[part]

    def run(self, part, name):
        """RunData, or None if the directory is absent or incomplete (missing data never passes)."""
        key = (part, name)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        d = os.path.join(self.part_dir(part), name)
        r = RunData(d, self.manifest, name) if os.path.isdir(d) else None
        if r is not None and not r.complete:
            r = None
        self._cache[key] = r
        while len(self._cache) > self._cache_n:
            _, old = self._cache.popitem(last=False)
            if old is not None:
                old.release()
        return r

    def log_text(self, part, name):
        p = os.path.join(self.part_dir(part), name + ".log")
        if not os.path.isfile(p):
            return None
        self.manifest.add(p)
        with open(p, errors="replace") as f:
            return f.read()

    def summary_json(self, part, name):
        p = os.path.join(self.part_dir(part), name + ".json")
        if not os.path.isfile(p):
            return None
        self.manifest.add(p)
        with open(p) as f:
            return json.load(f)


class Design:
    """Pre-registered design ('full') or the design implied by the runs found ('discovered')."""

    def __init__(self, kind, eps_codes, seeds, v2_ns, v1_ns, v1_tols, planes, times, n_ref, n_triple, candidates):
        self.kind = kind
        self.eps_codes = tuple(eps_codes)
        self.seeds = tuple(seeds)
        self.v2_ns = tuple(v2_ns)
        self.v1_ns = tuple(v1_ns)
        self.v1_tols = tuple(v1_tols)
        self.planes = tuple(planes)
        self.times = tuple(times)
        self.n_ref = n_ref
        self.n_triple = tuple(n_triple)
        self.candidates = tuple(candidates)  # ((N, lambda/h), ...) ascending

    @staticmethod
    def full():
        return Design("full", EPS_CODES, SEEDS, V2_NS, V1_NS, V1_TOLS, PLANES, TIMES, N_REF, N_TRIPLE,
                      LAMBDA_CANDIDATES)

    @staticmethod
    def discovered(campaign):
        disc = campaign.discover()
        eps, seeds, v2n, v1n, tols = set(), set(), set(), set(), set()
        for name in disc["v2"]:
            r = parse_run_name(name)
            eps.add(r[1]); seeds.add(r[2]); v2n.add(r[3])
        for name in disc["v1"]:
            r = parse_run_name(name)
            eps.add(r[1]); seeds.add(r[2]); v1n.add(r[3]); tols.add(r[4])
        nref = max(v2n) if v2n else N_REF
        trip = (nref // 4, nref // 2, nref)
        cands = tuple((n, n / 4.0) for n in sorted(v2n) if n < nref)
        return Design("discovered", sorted(eps), sorted(seeds), sorted(v2n), sorted(v1n),
                      sorted(tols) if tols else V1_TOLS, PLANES, TIMES, nref, trip, cands)

    def expected_runs(self):
        """[(part, name)] of every run the design needs."""
        out = []
        for e in self.eps_codes:
            for s in self.seeds:
                for n in self.v2_ns:
                    out.append(("v2", v2_name(e, s, n)))
                for n in self.v1_ns:
                    for t in self.v1_tols:
                        out.append(("v1", v1_name(e, s, n, t, 8)))
                    out.append(("v1", v1_name(e, s, n, TOL_REF, 32)))
        return out


class PairEngine:
    """Paired bootstrap statistics between two runs of a campaign, cached per (a, b, partition)."""

    def __init__(self, campaign, idx16):
        self.c = campaign
        self.idx16 = idx16
        self._counts = {}
        self._cache = {}

    def counts(self, g):
        if g not in self._counts:
            nb = g * g
            idx = self.idx16 if nb == BOOT_NBLOCKS else runtime_bootstrap(nb)
            self._counts[g] = counts_from_idx(idx, nb)
        return self._counts[g]

    def pair(self, ka, kb, g=4):
        """ka, kb = (part, name).  -> paired_stats dict or None if a run is missing."""
        key = (ka, kb, g)
        if key not in self._cache:
            a = self.c.run(*ka)
            b = self.c.run(*kb)
            if a is None or b is None:
                self._cache[key] = None
            else:
                self._cache[key] = paired_stats(a, b, self.counts(g), g)
        return self._cache[key]


# ------------------------------------------------------------------------------------------------------------
# output
# ------------------------------------------------------------------------------------------------------------
def fnum(x, digits=6):
    """compact display number."""
    if x is None:
        return ""
    if isinstance(x, (bool, np.bool_)):
        return "true" if x else "false"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return "%.*g" % (digits, x)
    return str(x)


def cnum(x):
    """full-precision csv number."""
    if isinstance(x, (float, np.floating)):
        if math.isnan(x):
            return "nan"
        if math.isinf(x):
            return "inf" if x > 0 else "-inf"
        return repr(float(x))
    if isinstance(x, (bool, np.bool_)):
        return "true" if x else "false"
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    return "" if x is None else str(x)


def md_table(header, rows):
    esc = lambda t: str(t).replace("|", "\\|")
    out = ["| " + " | ".join(esc(h) for h in header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(esc(fnum(c)) for c in r) + " |")
    return "\n".join(out)


class OutWriter:
    """Writes csv / markdown outputs, each stamped with versions and the bootstrap-matrix sha256."""

    def __init__(self, outdir, manifest, digest_name, argv_text=""):
        self.outdir = outdir
        self.manifest = manifest
        self.digest = digest_name
        self.argv = argv_text
        self.written = []
        os.makedirs(outdir, exist_ok=True)

    def stamp(self):
        v = versions()
        return ("python=%s numpy=%s scipy=%s bootstrap_sha256=%s digest=%s" %
                (v["python"], v["numpy"], v["scipy"], BOOT_SHA256, self.digest))

    def csv(self, name, header, rows):
        path = os.path.join(self.outdir, name)
        with open(path, "w", newline="") as f:
            f.write("# %s\n" % self.stamp())
            wr = csv.writer(f, lineterminator="\n")
            wr.writerow(header)
            for r in rows:
                wr.writerow([cnum(c) for c in r])
        self.written.append(path)
        return path

    def md(self, name, text):
        path = os.path.join(self.outdir, name)
        with open(path, "w") as f:
            f.write(text.rstrip("\n") + "\n")
        self.written.append(path)
        return path

    def md_header(self, title, extra=()):
        v = versions()
        lines = ["# " + title, "",
                 "- python %s, numpy %s, scipy %s" % (v["python"], v["numpy"], v["scipy"]),
                 "- bootstrap matrix: `analysis/bootstrap_blocks.npy`, sha256 `%s` (verified against the "
                 "pre-registered value)" % BOOT_SHA256,
                 "- command: `%s`" % self.argv]
        lines += ["- " + e for e in extra]
        return lines

    def write_manifest(self, name):
        ents = self.manifest.entries()
        self.csv(name, ["path", "sha256", "bytes"], ents)
        return len(ents), self.manifest.digest_of_digests()
