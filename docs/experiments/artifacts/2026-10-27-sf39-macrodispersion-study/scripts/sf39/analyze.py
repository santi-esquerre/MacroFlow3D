"""SF-39 analysis part (contract C-GLUE part 2; preregistration sections 3-5, 10): all reductions, criteria A-E, F7, F8, F10, F11
and the decision-point table.

    python3 -m sf39.analyze --out-root $OUT --artifact A --dest $OUT/analysis [--allow-partial]

Inputs: `$OUT/checkpoint/tau_b_star.json` (frozen) merged with `$OUT/checkpoint/s025/tau_b_star_s025.json`, `A/configs/budget.json` (final N per amplitude), runner arms `$OUT/runs/<arm>`, Par2
`$OUT/par2/raw/par2_s{c}[_ext]` + `field_stats_par2_s{c}[_ext].txt`, Par2 configs `A/configs/par2_s{c}.yaml`.  Everything is read through
`sf39.campaign` / `sf39.reduce` / `sf39.par2`; the statistics through `sf39.stats` / `sf39.decide` (nothing is re-implemented here).

Exit codes: 0 done, 2 refused (no frozen file, no/invalid budget, realizations missing relative to the budget: they are listed and nothing
is analysed or written), 1 failure (window/data error: the reducer never clips).  `--allow-partial` analyses what exists (effective N =
the complete prefix of every arm, capped by the budget) and labels every output PARTIAL (`PARTIAL.txt`, `summary.json` "partial",
decision notes, `decisions.md` title, manifest comment).

Sets and U (section 3; every value below is per amplitude, alpha in units of lambda, Student t primary, BCa beside each Student CI of A-D,
F7, F10, F11 and Gate 5; the recorded sensitivities of E carry Student intervals only):
  production   realizations g < N_R pooled over directory A and B by seed (`campaign.production_records`), flux weights, h/8 pass, U = mean qbar of the set
  narrow       the three rungs, first N_narrow seeds, each rung its own U over those seeds
  twin         first N_twin seeds paired with the production realizations of the same seeds; production side U = mean qbar of the production
               realizations g < N_twin, twin side U = mean qbar of the twin over the same seeds (`campaign.u_paired` through `pair_u`)
  h_max        directory A, first N_hmax seeds; passes 8/16/32 share one U = mean qbar of that set
  F10          s4_w24 / s4_w12 and the first 30 of s4_tw / s4_n16, each arm its own standalone U; the s4_w12 wide set is `W12_PARTS`
               (s4_w12 seeds 7201-7213 + continuation arm s4_w12c seeds 7215-7231 = 30, seed 7214 excluded: owner decision 2026-10-28)
A reduction (`reduce.reduce_methods`: all methods flux, unif, expI, full, pollock, ixc, ixp) is cached per (set, pass, U).

Outputs under `dest/`: `reduced/<set>.csv` (one row per realization and method, every documented reducer column; sets `<amp>_prod`
(pooled A+B, g < N_R), `<amp>_prod_twinpairs` (g < N_twin at the pair-set U), narrow rungs `<arm>`, `<amp>_tw`, `<amp>_A_hmax<d>`, `s4_w24`,
`s4_w12` (the combined 30 of `W12_PARTS`), `s4_tw_first30`, `s4_n16_first30`), `reduced/sigma_series_<set>.csv` (sigma_T^2(x1), per plane over its contributors),
`reduced/rho_<amp>.csv` (F8 rows, directory A), `series/alpha_t_<amp>.csv` (threshold E: alpha(t) and sigma_T^2(t) series of the
production estimate set, B&dD eq. 4 convention, see `alpha_series_rows`; recorded, non-gating), `summary.json` (every number with CI, n, BCa, verdict/label; sorted keys, %.17g; NaN -> null,
infinite one-sided bounds -> "inf"/"-inf"), `decisions.csv`, `decisions.md` (preregistration section 10; tiers primary / key secondary /
estimation / exploratory / process), `manifest.txt` (sha256sum format of every other file).
"""
import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys

import numpy as np

from . import campaign as cp
from . import decide
from . import io as sio
from . import par2, power
from . import reduce as red
from . import stats

N_PILOT = 20
EST = cp.ESTIMATORS                      # alpha_TAM, alpha_TP, alpha_LAM
TRANSVERSE = ("alpha_TAM", "alpha_TP")
KIND = {"alpha_TAM": "TAM", "alpha_TP": "TP", "alpha_LAM": "LAM"}
KEY = {"alpha_TAM": "win.alpha_TAM", "alpha_LAM": "win.alpha_e1", "alpha_TP": "tp.alpha_TP", "alpha_Tfixed": "win.alpha_Tfixed"}
SIGMA2_NOMINAL = {"s025": 0.25, "s1": 1.0, "s225": 2.25, "s4": 4.0}   # eps^2 (preregistration section 1: --eps 1, 1.5, 2 with --sigma2 1; s025: --eps 0.5)
BD_BOUND = {"s025": 1e-3, "s1": 0.01, "s225": 0.044}    # B&dD fixed-axis alpha_T (F11, descriptive; s025: their Table 2 upper bound alpha_TA < 1e-3)
BUDGET_KEYS = ("N_narrow", "N_twin", "N_hmax", "N_R", "N_A", "N_run", "par2_NR")
TIERS = ("primary", "key secondary", "estimation", "exploratory", "process")
L90, L95 = 0.90, 0.95
# F10 (ii) wide set (owner decision 2026-10-28; preregistration unchanged, `power.WIDTH_ARMS` keeps the preregistered projection of 30 for
# s4_w12): s4_w12 was halted at r = 13 (runner exit 4, seed 7214 never completed), so the wide set is (arm, first seed, realizations) =
# s4_w12 r = 0..12 + the continuation arm s4_w12c r = 0..16.  The ONE Python source of truth for the split (tests patch it with small counts).
W12_PARTS = (("s4_w12", 7201, 13), ("s4_w12c", 7215, 17))
W12_EXCLUDED_SEEDS = [7214]
W12_EXCLUDED_REASON = ("runner exit 4 (medium (b) solve not converged) at s4_w12 r=13; owner decision 2026-10-28: continuation arm "
                       "s4_w12c seeds 7215-7231")
TWO_P15 = 2.0 ** 1.5


class Refused(Exception):
    """The analysis refuses to proceed (exit 2)."""


# ------------------------------------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------------------------------------
def pair_u(prod_side, other_side):
    """(U_production_side, U_other_side, seeds) of a paired difference: each side its own mean qbar over the common seeds."""
    return cp.u_paired(prod_side, other_side)


def ci(x, level=L90, sided="two", bca=True):
    """Student CI of a mean (+ BCa beside it): dict mean, lo, hi, n, sd, level, sided, bca_lo, bca_hi, bca_degenerate."""
    x = np.asarray(x, dtype=float)
    mean, lo, hi, n, sd = stats.t_interval(x, level, sided)
    out = {"mean": mean, "lo": lo, "hi": hi, "n": n, "sd": sd, "level": level, "sided": sided}
    if bca:
        bl, bh, deg = stats.bca_interval(x, level, sided)
        out.update(bca_lo=bl, bca_hi=bh, bca_degenerate=deg)
    return out


def welch(a, b, level=L90):
    """Welch interval of mean(a) - mean(b) + BCa."""
    diff, lo, hi, nu, se = stats.welch_interval(a, b, level)
    bl, bh, deg = stats.bca_welch(a, b, level)
    return {"diff": diff, "lo": lo, "hi": hi, "nu": nu, "se": se, "n_a": len(a), "n_b": len(b),
            "bca_lo": bl, "bca_hi": bh, "bca_degenerate": deg}


def dist_block(seeds, d):
    """Distribution of a paired difference and the realization (seed) with the largest |difference| (threshold E)."""
    d = np.asarray(d, dtype=float)
    i = int(np.nanargmax(np.abs(d)))
    q = np.nanpercentile(d, [0, 25, 50, 75, 100])
    return {"n": int(d.size), "min": q[0], "p25": q[1], "median": q[2], "p75": q[3], "max": q[4],
            "argmax_seed": int(seeds[i]), "argmax_value": float(d[i])}


def diff_block(seeds, d):
    """Trend (point estimate + 90 % CI) and distribution of one paired difference."""
    return {"ci": ci(d), "dist": dist_block(seeds, d)}


def values(res, est, mode="flux"):
    """Per-realization values (sorted by seed) of an estimator for a reducer method; NaN where the method lacks it."""
    k = KEY[est]
    return np.array([res[s][1].get(mode, {}).get(k, np.nan) for s in sorted(res)], dtype=float)


def column(res, key, mode=None):
    """Per-realization values (sorted by seed) of a common column (mode None) or of a method column."""
    if mode is None:
        return np.array([res[s][0].get(key, np.nan) for s in sorted(res)], dtype=float)
    return np.array([res[s][1].get(mode, {}).get(key, np.nan) for s in sorted(res)], dtype=float)


def pdiff(ra, rb, est, mode="flux"):
    """(seeds, a - b) of an estimator over the common seeds of two reductions."""
    s, a, b = stats.align_by_seed(sorted(ra), values(ra, est, mode), sorted(rb), values(rb, est, mode))
    return s, a - b


def read_text(path):
    with open(path) as f:
        return f.read()


def read_json(path):
    return sio.read_json(path)


# ------------------------------------------------------------------------------------------------
# context, completeness, effective counts
# ------------------------------------------------------------------------------------------------
class Ctx:
    """Run context: paths, frozen file, effective counts, reduction cache, CSV tables, decision rows."""

    def __init__(self, out, art, dest, frz, eff, n_pilot, partial):
        self.out, self.art, self.dest, self.frz, self.eff = out, art, dest, frz, eff
        self.n_pilot, self.partial = n_pilot, partial
        self.cache, self.csv, self.rows, self.series = {}, {}, [], {}

    def tau(self, amp):
        return float(self.frz["amplitudes"][amp]["tau_b_star"])


def load_budget(art):
    """The validated `A/configs/budget.json` ({amp: {key: int}}); `Refused` if absent or invalid."""
    path = os.path.join(art, "configs", "budget.json")
    if not os.path.isfile(path):
        raise Refused("no budget file %s (the final N per amplitude are fixed after the owner's budget)" % path)
    with open(path) as f:
        bud = json.load(f)
    bad = []
    for amp in cp.AMPS:
        for k in BUDGET_KEYS:
            v = bud.get(amp, {}).get(k)
            if not (isinstance(v, int) and not isinstance(v, bool) and v > 0):
                bad.append("%s.%s=%r" % (amp, k, v))
    if bad:
        raise Refused("invalid budget file %s: %s" % (path, ", ".join(bad)))
    return bud


def _read_frozen(path):
    frz = read_json(path)
    if frz.get("schema") != "sf39-tau_b_star-v1":
        raise Refused("%s: unexpected schema %r" % (path, frz.get("schema")))
    return frz


def load_frozen(out):
    """`checkpoint/tau_b_star.json` merged with every separate-case file `checkpoint/<amp>/tau_b_star_<amp>.json` (s025, addendum
    2026-10-29).  `Refused` on a schema mismatch, on an amplitude present in two files, or on an amplitude of `cp.AMPS` with no entry."""
    path = os.path.join(out, "checkpoint", "tau_b_star.json")
    if not os.path.isfile(path):
        raise Refused("no frozen file %s (run the checkpoint part first)" % path)
    frz = _read_frozen(path)
    where = {amp: path for amp in frz["amplitudes"]}
    for amp in cp.AMPS:
        extra_path = os.path.join(out, "checkpoint", amp, "tau_b_star_%s.json" % amp)
        if not os.path.isfile(extra_path):
            continue
        extra = _read_frozen(extra_path)
        dup = sorted(set(extra["amplitudes"]) & set(where))
        if dup:
            raise Refused("amplitude(s) %s present in both %s and %s" % (",".join(dup), where[dup[0]], extra_path))
        frz["amplitudes"].update(extra["amplitudes"])
        where.update({a: extra_path for a in extra["amplitudes"]})
    absent = [amp for amp in cp.AMPS if amp not in frz["amplitudes"]]
    if absent:
        raise Refused("no frozen entry for %s (run the checkpoint part%s first)" % (",".join(absent), " (selector s025)" if "s025" in absent else ""))
    return frz


def expected_seed0(arm, b):
    """First field seed of an arm (preregistration section 1)."""
    if arm.endswith("_B"):
        return cp.PILOT_SEED0 + b["N_A"]
    return {"s4_w24": 7101, **{a: s0 for a, s0, _ in W12_PARTS}}.get(arm, cp.PILOT_SEED0)


def par2_files(out, amp):
    """(pilot raw dir, extension raw dir, FIELDSTAT files) of an amplitude."""
    raw, _, fsf = cp.par2_paths(out, "", amp)
    return raw, raw + "_ext", (fsf, fsf[:-len(".txt")] + "_ext.txt")


def par2_entries(C, amp):
    """Pooled Par2 realizations (pilot + extension when its directory exists)."""
    raw, ext, _ = par2_files(C.out, amp)
    return par2.pooled(raw, ext if os.path.isdir(ext) else None, n_pilot=C.n_pilot)


def par2_fieldstat(C, amp):
    """{field seed: FIELDSTAT dict} of the pilot and extension files."""
    fs = {}
    for p in par2_files(C.out, amp)[2]:
        if os.path.isfile(p):
            fs.update(par2.parse_fieldstat(read_text(p)))
    return fs


def available(out, arm):
    """Number of complete realizations r = 0 .. n-1 of an arm (the contiguous DONE prefix)."""
    rs = {x.r for x in cp.done_records(out, arm)}
    n = 0
    while n in rs:
        n += 1
    return n


def w12_counts(C):
    """[(arm, n)] realizations of the F10 (ii) wide set: `W12_PARTS` (full mode) or their complete prefixes (--allow-partial)."""
    return [(arm, min(cnt, available(C.out, arm)) if C.partial else cnt) for arm, _, cnt in W12_PARTS]


def w12_records(C):
    """Realizations of the F10 (ii) wide set: r < n of each part, in part order (seeds ascending)."""
    return [x for arm, n in w12_counts(C) for x in cp.done_records(C.out, arm)[:n]]


def missing_items(C, bud):
    """Strings describing every realization/file the budget asks for and cannot be found (empty = complete)."""
    miss = []
    for amp in cp.AMPS:
        b = bud[amp]
        counts = power.arm_counts(b, amp)
        if amp == "s4":  # the preregistered s4_w12 count is replaced by its two parts
            counts.pop("s4_w12", None)
            counts.update({arm: cnt for arm, _, cnt in W12_PARTS})
        for arm, cnt in counts.items():
            if arm.endswith("_par2") or cnt <= 0:
                continue
            recs = {x.r: x for x in cp.done_records(C.out, arm)}
            absent = [r for r in range(cnt) if r not in recs]
            if absent:
                miss.append("%s: realizations not DONE: %s" % (arm, ",".join(str(r) for r in absent)))
            s0 = expected_seed0(arm, b)
            wrong = [r for r in range(cnt) if r in recs and recs[r].seed != s0 + r]
            if wrong:
                miss.append("%s: seed != %d + r for r = %s" % (arm, s0, ",".join(str(r) for r in wrong)))
            if cp.is_multipass(arm):
                for r in range(cnt):
                    for d in cp.HMAX_DIVS:
                        if r in recs and not os.path.isdir(os.path.join(recs[r].rdir, "hmax_%d" % d)):
                            miss.append("%s: r_%04d has no hmax_%d pass" % (arm, r, d))
        n2 = b["par2_NR"]
        try:
            ents = par2_entries(C, amp)
        except (ValueError, OSError) as e:
            miss.append("par2 s%s: %s" % (cp.PAR2_ID[amp], e))
            continue
        if len(ents) < n2:
            miss.append("par2 s%s: %d realizations, budget asks for %d" % (cp.PAR2_ID[amp], len(ents), n2))
        fs = par2_fieldstat(C, amp)
        for e in ents[:n2]:
            if e["field_seed"] not in fs:
                miss.append("par2 s%s: no FIELDSTAT line for field seed %d" % (cp.PAR2_ID[amp], e["field_seed"]))
            if not os.path.isdir(os.path.join(e["dir"], "snapshots", "r_%04d" % e["r"])):
                miss.append("par2 s%s: no snapshots for g=%d" % (cp.PAR2_ID[amp], e["g"]))
    return miss


def effective_counts(C, bud):
    """Per amplitude the counts the analysis uses: the budget (full mode) or the complete prefixes capped by it (--allow-partial)."""
    eff = {}
    for amp in cp.AMPS:
        b, a = bud[amp], cp.arm_ids(amp)
        e = {k: b[k] for k in ("N_narrow", "N_twin", "N_hmax", "N_R")}
        e["n_w24"], e["n_w12"] = power.WIDTH_ARMS["s4_w24"], sum(cnt for _, _, cnt in W12_PARTS)
        e["n_ref"] = power.REF_FLOOR
        if C.partial:
            prod = cp.production_records(C.out, amp)
            g = [x.seed - cp.PILOT_SEED0 for x in prod]
            n_prod = 0
            while n_prod in g:
                n_prod += 1
            try:
                n_p2 = min(len(par2_entries(C, amp)), len(par2_fieldstat(C, amp)))
            except (ValueError, OSError):
                n_p2 = 0
            e["N_R"] = min(b["N_R"], n_prod, n_p2, b["par2_NR"])
            e["N_narrow"] = min([b["N_narrow"]] + [available(C.out, r) for r in a["narrow"]])
            e["N_twin"] = min(b["N_twin"], available(C.out, a["tw"]), n_prod)
            e["N_hmax"] = min(b["N_hmax"], available(C.out, a["A"]))
            if amp == "s4":
                e["n_w24"], e["n_w12"] = min(e["n_w24"], available(C.out, "s4_w24")), sum(n for _, n in w12_counts(C))
                e["n_ref"] = min(e["n_ref"], available(C.out, "s4_tw"), available(C.out, "s4_n16"))
        eff[amp] = e
    return eff


# ------------------------------------------------------------------------------------------------
# reductions
# ------------------------------------------------------------------------------------------------
def reduce_set(C, amp, recs, div, U):
    """{seed: (common, methods)} of `reduce.reduce_methods` at one pass with one U (cached per set, pass, U)."""
    key = (tuple((x.arm, x.seed) for x in recs), div, float(U))
    if key not in C.cache:
        res, tau = {}, C.tau(amp)
        for x in recs:
            real = cp.realization(x, div)
            if real.seed != x.seed:
                raise sio.ReduceError("%s r=%d: field.json seed %d != %d" % (x.arm, x.r, real.seed, x.seed))
            res[x.seed] = red.reduce_methods(real, tau, U, r=x.r)
        C.cache[key] = res
    return C.cache[key]


def add_csv(C, name, recs, res, div=cp.PROD_DIV):
    """Register `reduced/<name>.csv` (one row per realization and method, all common columns) and the sigma_T^2(x1) series."""
    rows = []
    for s in sorted(res):
        common, methods = res[s]
        for row in red.long_rows(common, methods, keep=tuple(common)):
            rows.append(dict(set=name, hmax_div=div, **row))
    C.csv[name] = rows
    if div == cp.PROD_DIV and "_hmax" not in name and not name.endswith("twinpairs"):
        srows = []
        for x in sorted(recs, key=lambda x: x.seed):
            for row in red.sigma_series_rows(cp.realization(x, div).pass_):
                srows.append(dict(set=name, seed=x.seed, **row))
        C.csv["sigma_series_" + name] = srows


def reduce_named(C, amp, name, recs, div, U):
    res = reduce_set(C, amp, recs, div, U)
    add_csv(C, name, recs, res, div)
    return res


# ------------------------------------------------------------------------------------------------
# alpha(t) and sigma_T^2(t) series (threshold E; B&dD eq. 4 as in src/analysis/macrodispersion/MacrodispersionAnalysis.hpp)
# ------------------------------------------------------------------------------------------------
SERIES_MODES = ("flux", "unif", "pollock", "pollock_unif")
SERIES_COLS = ("alpha_L_fixed", "alpha_T_fixed", "alpha_L_aligned", "alpha_TAM", "sigma_T2_fixed")


def variance_series(real, mode):
    """(times, V) of one realization: V[j] = projected variances (e1, e2, e3, e_L, e_T1, e_T2) at snapshot j of the biased weighted
    covariance of the particles with a record at j (flux: weights.npy, unif: uniform, pollock: pollock_weights.npy over Pollock
    records, pollock_unif: uniform over Pollock records).  An empty set or a zero total weight raises (`reduce.wmoments3`)."""
    P = real.pass_
    if mode.startswith("pollock"):
        Q = real.pollock
        if Q is None:
            raise sio.MissingData("no Pollock bridge in %s" % real.rdir)
        tx, w = Q.times_x, np.asarray(Q.weights, dtype=float)
    else:
        tx, w = P.times_x, np.asarray(P.weights, dtype=float)
    unif = mode in ("unif", "pollock_unif")
    q = (real.flow["interior"]["mean_u"], real.flow["interior"]["mean_v"], real.flow["interior"]["mean_w"])
    E = np.vstack([np.eye(3), np.vstack(red.aligned_frame(q))])
    times = np.asarray(P.times, dtype=float)
    V = np.empty((len(times), 6))
    for j in range(len(times)):
        X = np.asarray(tx[:, j, :], dtype=float)
        idx = np.flatnonzero(np.isfinite(X).all(axis=1))
        _, Cv = red.wmoments3(X[idx], np.ones(len(idx)) if unif else w[idx])
        V[j] = np.einsum("ik,kl,il->i", E, Cv, E)
    return times, V


def time_derivative(t, V):
    """dV/dt per snapshot: central differences, one-sided at the first and last snapshot (the runner's convention)."""
    if len(t) < 2:
        raise sio.ReduceError("a series needs at least two snapshots")
    d = np.empty_like(V)
    d[0] = (V[1] - V[0]) / (t[1] - t[0])
    d[-1] = (V[-1] - V[-2]) / (t[-1] - t[-2])
    d[1:-1] = (V[2:] - V[:-2]) / (t[2:] - t[:-2])[:, None]
    return d


def alpha_series_rows(recs, U):
    """Long-format rows (mode, j, t, tau, n_real, <col>_mean/_se) of the ensemble series of a set of realizations (h/8 pass) with one U.

    Per realization: alpha_e(t_j) = dV_e/dt / (2 ell U) with the derivative taken over that realization's own snapshots; the ensemble is
    taken at the common snapshot index over the common prefix of the snapshot counts, whose times must agree within 1e-12 relative
    (`ReduceError` otherwise).  Columns: alpha_L_fixed (e1), alpha_T_fixed ((e2+e3)/2), alpha_L_aligned (e_L), alpha_TAM ((e_T1+e_T2)/2),
    sigma_T2_fixed = (V_e2 + V_e3)/2; tau = U t / ell.  Returns (rows, n_real, n_snapshots).
    """
    per = {m: [] for m in SERIES_MODES}
    times, ell = [], None
    for x in recs:  # one realization at a time (arrays are mmapped)
        real = cp.realization(x, cp.PROD_DIV)
        ell = real.ell if ell is None else ell
        for m in SERIES_MODES:
            t, V = variance_series(real, m)
            a = time_derivative(t, V) / (2.0 * real.ell * U)
            cols = np.column_stack([a[:, 0], 0.5 * (a[:, 1] + a[:, 2]), a[:, 3], 0.5 * (a[:, 4] + a[:, 5]), 0.5 * (V[:, 1] + V[:, 2])])
            per[m].append(cols)
            if m == SERIES_MODES[0]:
                times.append(t)
    n = min(len(t) for t in times)
    for x, t in zip(recs, times):
        if np.any(np.abs(t[:n] - times[0][:n]) > 1e-12 * np.maximum(1.0, np.abs(times[0][:n]))):
            raise sio.ReduceError("%s r=%d: snapshot times differ from the first realization within the common prefix" % (x.arm, x.r))
    rows = []
    for m in SERIES_MODES:
        A = np.stack([c[:n] for c in per[m]])  # (n_real, n, 5)
        for j in range(n):
            row = {"mode": m, "j": j, "t": float(times[0][j]), "tau": float(U * times[0][j] / ell), "n_real": len(recs)}
            for k, name in enumerate(SERIES_COLS):
                v = A[:, j, k]
                row[name + "_mean"] = float(v.mean())
                row[name + "_se"] = float(v.std(ddof=1) / np.sqrt(v.size)) if v.size > 1 else float("nan")
            rows.append(row)
    return rows, len(recs), n


# ------------------------------------------------------------------------------------------------
# criterion A and D plumbing (pure functions of reductions; unit-tested on synthetic arrays)
# ------------------------------------------------------------------------------------------------
def ladder_block(est, vals_prod, vals_half, vals_1x, vals_2x, Dc, Df, Dtw, vals_prod_tw, d16_8, d32_16, vals_h16):
    """Criteria A and D of one estimator: margins, narrow / F4 production / h_max verdicts, trends, Richardson guard, rule D.

    vals_prod production standalone values; vals_half/1x/2x the narrow rungs (first N_narrow seeds); Dc = 1x - 1/2, Df = 2x - 1x (paired by
    seed); Dtw = production - twin over the pair seeds; vals_prod_tw the production values over those seeds; d16_8 = h/16 - h/8,
    d32_16 = h/32 - h/16 over directory A; vals_h16 the h/16 values.  Returns (block dict, rule-D result or None for alpha_LAM).
    """
    kind = KIND[est]
    mean_prod, mean_2x = float(np.mean(vals_prod)), float(np.mean(vals_2x))
    m = stats.m_h(kind, mean_2x, mean_prod)
    f4 = decide.production_h_verdict(Dc, Df, Dtw, m)
    mean_h16 = float(np.mean(vals_h16))
    m_hm = stats.margin(kind, mean_h16)
    hv = decide.hmax_verdict(d16_8, m_hm)
    order_sample = f4["s"] * (TWO_P15 * np.asarray(Df, float) - np.asarray(Dc, float))
    blk = {"margins": {"m_h": m, "margin_at_2x_mean": stats.margin(kind, mean_2x), "margin_at_production_mean": stats.margin(kind, mean_prod),
                       "m_hmax": m_hm, "mean_2x": mean_2x, "mean_production": mean_prod, "mean_h16": mean_h16},
           "narrow_verdict": f4["narrow"], "F4": f4, "F4_verdict": f4["verdict"],
           "F4_order_ci_bca": ci(order_sample), "F4_Dtw_ci_bca": ci(Dtw), "F4_Df_ci_bca": ci(Df),
           "hmax": hv, "hmax_verdict": hv["verdict"], "hmax_ci_bca": ci(d16_8),
           "richardson": stats.richardson_guard(float(np.mean(vals_half)), float(np.mean(vals_1x)), mean_2x)}
    rd = None
    if est in TRANSVERSE:
        rd = decide.rule_d(
            {"h": {"verdict": f4["verdict"], "diffs": {"D_f": Df, "D_tw": Dtw}, "finer": {"D_f": vals_2x, "D_tw": vals_prod_tw}},
             "hmax": {"verdict": hv["verdict"], "diffs": {"d16_8": d16_8}, "finer": {"d16_8": vals_h16}}},
            vals_prod)
    return blk, rd


# ------------------------------------------------------------------------------------------------
# decision rows
# ------------------------------------------------------------------------------------------------
def row(C, tier, amp, estimator, point, verdict, cis=None, n=None, note=""):
    C.rows.append((tier, dict(amplitude=amp, estimator=estimator, decision_point=point, verdict=verdict, ci=cis, n=n,
                              note=("PARTIAL: " if C.partial else "") + note)))


def sign_word(lo, hi):
    return "above 0" if lo > 0 else ("below 0" if hi < 0 else "includes 0")


# ------------------------------------------------------------------------------------------------
# Gate 5 (criterion C)
# ------------------------------------------------------------------------------------------------
def lagrangian_alpha_x(recs, tau, U):
    """alpha_LAM per realization (sorted by seed), uniform weights on the window [0.75, 1] tau ell / U of the production pass."""
    return np.array([red.window_method(cp.realization(x, cp.PROD_DIV), "unif", tau, U)["win.alpha_e1"]
                     for x in sorted(recs, key=lambda x: x.seed)], dtype=float)


def gate5_block(C, amp, prod):
    g5 = C.frz["amplitudes"][amp]["gate5"]
    tw, p2, tau_c = g5["twin"], g5["par2"], float(g5["tau_c"])
    n2 = C.eff[amp]["N_R"]
    uc_tw, _, x0 = cp.twin_uc(C.out, prod, tw["t_b_min"])
    L = lagrangian_alpha_x(prod, tau_c, uc_tw)
    # Par2
    _, cfg_path, _ = cp.par2_paths(C.out, C.art, amp)
    v = par2.cfg_values(read_text(cfg_path))
    if abs(v["lam"] - p2["corr_length"]) > 1e-12 * v["lam"] or abs(v["inject_x"] - p2["inject_x"]) > 1e-12 * abs(v["inject_x"]):
        raise sio.ReduceError("Par2 config of s%s disagrees with the frozen corr_length / inject_x" % amp)
    ents = par2_entries(C, amp)[:n2]
    if len(ents) < n2:
        raise sio.MissingData("Par2 s%s: %d realizations, need %d" % (amp, len(ents), n2))
    series = [par2.series_of(e) for e in ents]
    uc_p, _ = par2.uc_mean(series, p2["t_b_min"], p2["inject_x"])
    t_lo, t_hi = par2.window(tau_c, v["lam"], uc_p)
    al = [par2.alpha_window(s, v["lam"], uc_p, t_lo, t_hi) for s in series]
    P = np.array([a["x"] for a in al])
    Ty = np.array([a["y"] for a in al])
    Tz = np.array([a["z"] for a in al])
    table = par2.t_b_table(ents, v["Lx"], v["lam"], v["dt"])
    late = par2.late_count(table, t_hi, C.n_pilot)
    fs = par2_fieldstat(C, amp)
    sig2 = np.array([fs[e["field_seed"]]["var_lnK"] for e in ents])
    # sensitivity (i): FV qbar everywhere in the Lagrangian arm
    u_fv = cp.u_standalone(prod)
    tau_tw_fv = u_fv * tw["t_b_min"] / tw["ell"]
    tau_c_fv = min(tau_tw_fv, float(p2["tau_b_star"]))
    L_fv = lagrangian_alpha_x(prod, tau_c_fv, u_fv)
    nominal = SIGMA2_NOMINAL[amp]
    v5 = decide.gate5_verdict(L, P, L_fv, sig2, nominal)
    for name, (a_, b_) in (("primary", (L, P)), ("fv_qbar", (L_fv, P))):
        bl, bh, deg = stats.bca_welch(a_, b_, L90)
        v5[name].update(bca_lo=bl, bca_hi=bh, bca_degenerate=deg, n_L=len(a_), n_P=len(b_))
    bl, bh, deg = stats.bca_ancova_vs(L, P, sig2, nominal, L90)
    v5["ancova"].update(bca_lo=bl, bca_hi=bh, bca_degenerate=deg, n_L=len(L), n_P=len(P))
    blk = {"verdict": v5["primary"]["verdict"], "gate5": v5,
           "frozen": {"tau_c": tau_c, "t_b_min_twin": tw["t_b_min"], "t_b_min_par2": p2["t_b_min"],
                      "tau_b_star_twin": tw["tau_b_star"], "tau_b_star_par2": p2["tau_b_star"]},
           "lagrangian": {"U_c": uc_tw, "anchor_x0": x0, "window": [0.75 * tau_c * tw["ell"] / uc_tw, tau_c * tw["ell"] / uc_tw],
                          "alpha_LAM": ci(L), "seeds": [x.seed for x in sorted(prod, key=lambda x: x.seed)]},
           "lagrangian_fv_qbar": {"U": u_fv, "tau_b_star_twin": tau_tw_fv, "tau_c": tau_c_fv,
                                  "window": [0.75 * tau_c_fv * tw["ell"] / u_fv, tau_c_fv * tw["ell"] / u_fv], "alpha_LAM": ci(L_fv)},
           "par2": {"U_c": uc_p, "window": [t_lo, t_hi], "n": len(ents), "alpha_x": ci(P), "alpha_y": ci(Ty), "alpha_z": ci(Tz),
                    "alpha_T_fixed": ci(0.5 * (Ty + Tz)), "late_beyond_pilot": late[0], "n_beyond_pilot": late[1], "late_pilot": late[2],
                    "var_lnK": {"mean": float(sig2.mean()), "sd": float(sig2.std(ddof=1)) if sig2.size > 1 else float("nan"),
                                "nominal": nominal, "config_sigma2": v["sigma2"]}}}
    row(C, "primary", amp, "alpha_LAM", "Gate 5 (C) Lagrangian twin vs Par2", v5["primary"]["verdict"],
        (v5["primary"]["lo"], v5["primary"]["hi"]), "%d/%d" % (len(L), len(P)),
        "diff=%.6g margin=%.6g; late Par2 beyond pilot %d/%d" % (v5["primary"]["diff"], v5["primary"]["margin"], late[0], late[1]))
    row(C, "primary", amp, "alpha_LAM", "Gate 5 sensitivity FV qbar", v5["fv_qbar"]["verdict"], (v5["fv_qbar"]["lo"], v5["fv_qbar"]["hi"]),
        "%d/%d" % (len(L_fv), len(P)), "diff=%.6g margin=%.6g" % (v5["fv_qbar"]["diff"], v5["fv_qbar"]["margin"]))
    row(C, "primary", amp, "alpha_LAM", "Gate 5 sensitivity ANCOVA", v5["ancova"]["verdict"], (v5["ancova"]["lo"], v5["ancova"]["hi"]),
        "%d/%d" % (len(L), len(P)), "diff=%.6g margin=%.6g adjusted Par2 mean=%.6g" % (v5["ancova"]["diff"], v5["ancova"]["margin"], v5["ancova"]["a"]))
    return blk


# ------------------------------------------------------------------------------------------------
# F8 (rho)
# ------------------------------------------------------------------------------------------------
def f8_block(C, amp, hm, m_tp_hmax, S):
    """rho(X) rows of directory A (g < N_hmax), readings per pass pair (8-16 is the gating reading), growth fits."""
    rows = []
    for x in hm:
        for r_ in red.rho_rows(x.rdir, cp.HMAX_DIVS, red.RHO_PAIRS):
            rows.append(dict(r_, seed=x.seed, r=x.r))
    C.csv["rho_" + amp] = rows
    real0 = cp.realization(hm[0], cp.PROD_DIV)
    planes, x0, ell = real0.pass_.planes, real0.x0, real0.ell
    ib = real0.key_planes()["Xb"][1]
    out = {"m_alpha_TP_hmax": m_tp_hmax, "S": S, "pairs": {}}
    for pair in ("%d-%d" % p for p in red.RHO_PAIRS):
        M = np.full((len(hm), len(planes)), np.nan)
        for k, x in enumerate(hm):
            for r_ in rows:
                if r_["seed"] == x.seed and r_["pair"] == pair:
                    M[k, r_["plane"]] = r_["rho"]
        xb = M[:, ib]
        fin = xb[np.isfinite(xb)]
        rd_ = stats.rho_readings(fin, m_tp_hmax, S) if fin.size else None
        cen = np.array([r_["rho_c"] for r_ in rows if r_["pair"] == pair and r_["plane"] == ib], dtype=float)
        rs = np.array([r_["rho_S"] for r_ in rows if r_["pair"] == pair and r_["plane"] == ib], dtype=float)
        out["pairs"][pair] = {"readings": rd_, "n_finite": int(fin.size), "growth_fits": stats.rho_growth_fits(planes, M, x0, ell),
                              "rho_c_median_Xb": float(np.nanmedian(cen)) if cen.size else float("nan"),
                              "rho_S_median_Xb": float(np.nanmedian(rs)) if rs.size else float("nan"),
                              "rho_S_max_Xb": float(np.nanmax(rs)) if rs.size else float("nan")}
    g = out["pairs"]["8-16"]["readings"]
    if g is None:
        label = "no finite rho(X_b)"
    else:
        flags = [n.replace("_", " ") for n in ("pathwise_resolved", "pathwise_limited", "decorrelated") if g[n]]
        label = " + ".join(flags) if flags else "intermediate (none of the three readings)"
        label = label.replace("pathwise limited", "pathwise-limited")
    out["gating_label"] = label
    return out


# ------------------------------------------------------------------------------------------------
# one amplitude
# ------------------------------------------------------------------------------------------------
def analyze_amp(C, amp):
    e, a, out = C.eff[amp], cp.arm_ids(amp), C.out
    S = {"partial": C.partial, "n": {k: int(v) for k, v in e.items()}}
    # ---- sets and U (section 3)
    prod = cp.production_records(out, amp, e["N_R"])
    allprod = cp.production_records(out, amp)
    prod_tw = [x for x in allprod if x.seed - cp.PILOT_SEED0 < e["N_twin"]]
    hm = cp.done_records(out, a["A"])[:e["N_hmax"]]
    rungs = [cp.done_records(out, arm)[:e["N_narrow"]] for arm in a["narrow"]]
    tw = cp.done_records(out, a["tw"])[:e["N_twin"]]
    for nm, rr, n in [("production", prod, e["N_R"]), ("production over twin pairs", prod_tw, e["N_twin"]), ("h_max set", hm, e["N_hmax"]),
                      ("twin", tw, e["N_twin"])] + [("rung " + arm, r_, e["N_narrow"]) for arm, r_ in zip(a["narrow"], rungs)]:
        if len(rr) != n:
            raise sio.MissingData("%s %s: %d realizations, expected %d" % (amp, nm, len(rr), n))
    U_prod = cp.u_standalone(prod)
    U_ptw, U_tw, _ = pair_u(prod_tw, tw)
    U_rung = [cp.u_standalone(r_) for r_ in rungs]
    U_hm = cp.u_standalone(hm)
    S["U"] = {"production": U_prod, "production_twin_pairs": U_ptw, "twin": U_tw, "rungs": dict(zip(a["narrow"], U_rung)), "hmax": U_hm}
    S["frozen"] = {"tau_b_star": C.tau(amp), "t_b_min": C.frz["amplitudes"][amp]["t_b_min"], "U_prod_pilot": C.frz["amplitudes"][amp]["U_prod"]}
    # ---- reductions
    R_prod = reduce_named(C, amp, amp + "_prod", prod, cp.PROD_DIV, U_prod)
    srows, n_real, n_snap = alpha_series_rows(sorted(prod, key=lambda x: x.seed), U_prod)
    C.series["alpha_t_" + amp] = srows
    S["alpha_t"] = {"file": "series/alpha_t_%s.csv" % amp, "n_real": n_real, "n_snapshots": n_snap, "modes": list(SERIES_MODES),
                    "recorded_non_gating": True}
    R_ptw = reduce_named(C, amp, amp + "_prod_twinpairs", prod_tw, cp.PROD_DIV, U_ptw)
    R_tw = reduce_named(C, amp, amp + "_tw", tw, cp.PROD_DIV, U_tw)
    R_n = [reduce_named(C, amp, arm, r_, cp.PROD_DIV, u) for arm, r_, u in zip(a["narrow"], rungs, U_rung)]
    R_h = {d: reduce_named(C, amp, "%s_A_hmax%d" % (amp, d), hm, d, U_hm) for d in cp.HMAX_DIVS}
    n_half, n_1x, n_2x = R_n
    # ---- production values (B, means) and recorded tb
    S["production"] = {est: ci(values(R_prod, est)) for est in EST}
    S["production"]["alpha_Tfixed"] = ci(values(R_prod, "alpha_Tfixed"))
    S["production"]["seeds"] = sorted(R_prod)
    S["production"]["t_b_r"] = [float(t) for t in column(R_prod, "tb")]
    # ---- A, B, D
    A, B, D = {}, {}, {}
    for est in EST:
        vp = values(R_prod, est)
        sD, Dc = pdiff(n_1x, n_half, est)
        sF, Df = pdiff(n_2x, n_1x, est)
        sT, Dtw = pdiff(R_ptw, R_tw, est)
        sH, d16_8 = pdiff(R_h[16], R_h[8], est)
        _, d32_16 = pdiff(R_h[32], R_h[16], est)
        blk, rd = ladder_block(est, vp, values(n_half, est), values(n_1x, est), values(n_2x, est), Dc, Df, Dtw, values(R_ptw, est),
                               d16_8, d32_16, values(R_h[16], est))
        blk["trend"] = {"D_c": diff_block(sD, Dc), "D_f": diff_block(sF, Df), "D_tw": diff_block(sT, Dtw),
                        "d16_8": diff_block(sH, d16_8), "d32_16": diff_block(sH, d32_16)}
        blk["narrow_rung_vs_production"] = welch(values(n_1x, est), vp)
        A[est] = blk
        mean, hw, target, met = stats.halfwidth_check(vp, stats.halfwidth_target(KIND[est], float(np.mean(vp))), L95)
        B[est] = {"mean": mean, "halfwidth_95": hw, "target": target, "met": met, "n": int(vp.size)}
        row(C, "key secondary", amp, est, "B half-width (95 %)", "met" if met else "not met", (mean - hw, mean + hw), int(vp.size),
            "halfwidth=%.6g target=%.6g" % (hw, target))
        f4 = blk["F4"]
        row(C, "key secondary", amp, est, "A h narrow verdict (D_f)", f4["narrow"], (f4["narrow_ci"]["lo"], f4["narrow_ci"]["hi"]),
            f4["narrow_ci"]["n"], "m=%.6g mean D_f=%.6g" % (f4["m"], f4["narrow_ci"]["mean"]))
        row(C, "key secondary", amp, est, "A h production verdict (F4)", f4["verdict"], (f4["tw_ci"]["lo"], f4["tw_ci"]["hi"]),
            f4["tw_ci"]["n"], "CI of D_tw vs +-2^1.5 m=%.6g (inside=%s)" % (TWO_P15 * f4["m"], f4["tw_inside"]))
        row(C, "key secondary", amp, est, "A h order contrast", "monotone order >= 1.5 not rejected" if f4["order_ok"] else "rejected",
            (f4["order_ci"]["lo"], f4["order_ci"]["hi"]), f4["order_ci"]["n"], "CI of s(2^1.5 D_f - D_c), s=%+g" % f4["s"])
        row(C, "key secondary", amp, est, "A h_max verdict (h/16 - h/8)", blk["hmax_verdict"], (blk["hmax"]["ci"]["lo"], blk["hmax"]["ci"]["hi"]),
            blk["hmax"]["ci"]["n"], "m=%.6g mean=%.6g" % (blk["hmax"]["m"], blk["hmax"]["ci"]["mean"]))
        if rd is not None:
            D[est] = rd
    S["A"], S["B"] = A, B
    lab = decide.amplitude_label(D["alpha_TAM"]["label"], D["alpha_TP"]["label"])
    S["D"] = {"alpha_TAM": D["alpha_TAM"], "alpha_TP": D["alpha_TP"], "amplitude_label": lab}
    for est in TRANSVERSE:
        d = D[est]
        row(C, "primary", amp, est, "Gate 4 label (D)", d["label"], (d["prod_ci"]["lo"], d["prod_ci"]["hi"]), d["prod_ci"]["n"],
            "branch %d; ladder=%s difference=%s; h=%s hmax=%s; production mean=%.6g" % (
                d["branch"], d["ladder"], d["difference"], A[est]["F4_verdict"], A[est]["hmax_verdict"], d["prod_ci"]["mean"]))
    row(C, "primary", amp, "alpha_TAM+alpha_TP", "Gate 4 amplitude label (D)", lab, None, e["N_R"],
        "alpha_TAM: %s; alpha_TP: %s" % (D["alpha_TAM"]["label"], D["alpha_TP"]["label"]))
    # ---- C (Gate 5)
    S["C"] = gate5_block(C, amp, prod)
    # ---- F7
    dh = column(R_prod, "tp.delta_half", "flux")
    if np.isfinite(dh).all():
        d7 = stats.delta_half(dh)
        d7["bca"] = ci(dh)
        d7["second_half_alpha_TP"] = ci(values(R_prod, "alpha_TP"))
        d7["statement"] = ("decreasing at the measured distance; the second-half alpha_TP is an upper bound on the asymptote under monotone decay"
                           if d7["decreasing"] else "no statement beyond 'at the measured distance'")
        S["F7"] = d7
        row(C, "estimation", amp, "alpha_TP", "F7 Delta_half", "decreasing" if d7["decreasing"] else "not shown decreasing", (d7["lo"], d7["hi"]),
            d7["n"], "mean Delta_half=%.6g" % d7["mean"])
    else:
        S["F7"] = {"available": False}
    # ---- F8
    S["F8"] = f8_block(C, amp, hm, A["alpha_TP"]["margins"]["m_hmax"], float(np.mean(column(R_h[8], "S_r"))))
    row(C, "estimation", amp, "rho(X_b), pair 8-16", "F8 readings", S["F8"]["gating_label"], None, len(hm),
        "median=%.6g p95=%.6g (m=%.6g, S=%.6g)" % (
            (S["F8"]["pairs"]["8-16"]["readings"] or {}).get("median", float("nan")),
            (S["F8"]["pairs"]["8-16"]["readings"] or {}).get("p95", float("nan")), S["F8"]["m_alpha_TP_hmax"], S["F8"]["S"]))
    # ---- F11
    S["F11"] = f11_block(C, amp, R_prod, R_n, R_tw, R_ptw)
    # ---- E / recorded
    S["E"] = recorded_block(C, amp, R_prod)
    # ---- F10 (s4 only)
    if amp == "s4":
        S["F10"] = f10_block(C, amp)
    return S


# ------------------------------------------------------------------------------------------------
# F11, E, F10
# ------------------------------------------------------------------------------------------------
F11_EST = ("alpha_TAM", "alpha_TP", "alpha_Tfixed")


def f11_block(C, amp, R_prod, R_n, R_tw, R_ptw):
    arms = {"production": R_prod, "narrow_half": R_n[0], "narrow_1x": R_n[1], "narrow_2x": R_n[2], "twin": R_tw, "production_twin_pairs": R_ptw}
    dp = {}
    for nm, res in arms.items():
        for est in F11_EST:
            dp[(nm, est)] = (sorted(res), values(res, est, "ixp") - values(res, est, "ixc"))
    out = {"delta_pollock": {}, "diff_in_diff": {}}
    for nm in ("production", "narrow_half", "narrow_1x", "narrow_2x", "twin"):
        for est in F11_EST:
            s, d = dp[(nm, est)]
            mean, lo, hi, n, sd = stats.delta_pollock(values(arms[nm], est, "ixp"), values(arms[nm], est, "ixc"), L90)
            blk = {"ci": ci(d), "mean_consumer": float(np.mean(values(arms[nm], est, "ixc"))),
                   "mean_pollock": float(np.mean(values(arms[nm], est, "ixp"))), "dist": dist_block(s, d)}
            out["delta_pollock"].setdefault(nm, {})[est] = blk
            row(C, "estimation", amp, est, "F11 Delta_P %s" % nm, "Delta_P " + sign_word(lo, hi), (lo, hi), n, "mean Delta_P=%.6g" % mean)
    for label, (na, nb) in {"narrow_half_minus_1x": ("narrow_half", "narrow_1x"), "narrow_1x_minus_2x": ("narrow_1x", "narrow_2x"),
                            "twin_minus_production": ("twin", "production_twin_pairs")}.items():
        for est in F11_EST:
            (sa, da), (sb, db) = dp[(na, est)], dp[(nb, est)]
            s, a_, b_ = stats.align_by_seed(sa, da, sb, db)
            mean, lo, hi, n, sd = stats.diff_in_diff(a_, b_, L90)
            out["diff_in_diff"].setdefault(label, {})[est] = {"ci": ci(a_ - b_), "dist": dist_block(s, a_ - b_)}
            row(C, "estimation", amp, est, "F11 difference in differences %s" % label, "difference " + sign_word(lo, hi), (lo, hi), n,
                "mean=%.6g" % mean)
    fx = values(R_prod, "alpha_Tfixed")
    bound = BD_BOUND.get(amp)
    out["upper_bound_fixed_axis_alpha_T"] = {"alpha_Tfixed": ci(fx, L95, "upper"), "BdD_value": bound,
                                              "below_BdD": None if bound is None else bool(stats.t_interval(fx, L95, "upper")[2] < bound),
                                              "conditions": "descriptive; B&dD: fixed axes, 5 lambda injection buffer, N_R = 500, Euler stepping"}
    if amp == "s025":  # addendum A4.3: a disagreement with B&dD's bound is stated only if the one-sided 95 % LOWER limit exceeds it
        lower = ci(fx, L95, "lower")
        out["upper_bound_fixed_axis_alpha_T"].update(alpha_Tfixed_lower=lower, exceeds_BdD=bool(lower["lo"] > bound))
    return out


def recorded_block(C, amp, R):
    """Threshold E: ensemble summaries of the sensitivities and recorded context (production set)."""
    E = {"sensitivities": {}, "split_half": {}, "context": {}}
    for est in EST:
        for mode in red.WIN_MODES:
            v = values(R, est, mode)
            if np.isfinite(v).any():
                E["sensitivities"].setdefault(est, {})[mode] = ci(v, bca=False)
    for key in ("win.alpha_TD", "win.alpha_eD", "win.alpha_Tfixed", "win.dTfixed_minus_TAM", "win.alpha_eL"):
        E["sensitivities"].setdefault("flux_frames", {})[key] = ci(column(R, key, "flux"), bca=False)
    E["sensitivities"]["tilt"] = ci(column(R, "tilt"), bca=False)
    E["sensitivities"]["drift_tilt"] = ci(column(R, "win.drift_tilt", "flux"), bca=False)
    for est in EST:
        a_ = column(R, "split.%s.A" % est)
        b_ = column(R, "split.%s.B" % est)
        E["split_half"][est] = stats.splithalf_fraction(a_, b_, values(R, est))
    summ = lambda x: {"mean": float(np.nanmean(x)), "min": float(np.nanmin(x)), "max": float(np.nanmax(x))}  # noqa: E731
    E["context"]["exits_jhi"] = {k: summ(column(R, k)) for k in ("exit_inlet_jhi", "exit_outlet_jhi")}
    E["context"]["excluded_fraction"] = {m: summ(column(R, "win.excl", m)) for m in ("flux", "unif")}
    E["context"]["plane_excluded_fraction_flux"] = summ(column(R, "tp.excl", "flux"))
    E["context"]["noncrossing"] = {k: summ(column(R, k, "flux")) for k in ("tp.n_noncross_Xa", "tp.n_noncross_Xb")}
    E["context"]["set_size"] = {"win.n": summ(column(R, "win.n", "flux")), "tp.n": summ(column(R, "tp.n", "flux"))}
    E["context"]["plane_hist_mean"] = [float(np.mean(column(R, "plane_hist_%d" % k))) for k in range(sio.N_STATUS)]
    E["context"]["time_hist_mean"] = [float(np.mean(column(R, "time_hist_%d" % k))) for k in range(sio.N_STATUS)]
    E["context"]["full_exits"] = {k: summ(column(R, k, "full")) for k in ("win.n_exit_out", "win.n_exit_in", "win.n_seed_fallback", "win.n_unplaced")}
    E["context"]["ev"] = {k: {"median": float(np.nanmedian(column(R, k))), "max": float(np.nanmax(column(R, k)))} for k in red.EV_NAMES}
    E["context"]["flow_stats"] = {k: summ(column(R, k)) for k in ("qbar", "x_nonpositive_fraction", "speed_p0", "speed_p0_1", "speed_p1", "speed_p50")}
    E["context"]["t_b"] = summ(column(R, "tb"))
    E["context"]["field"] = {k: summ(column(R, k)) for k in ("applied_scale", "raw_variance", "final_variance")}
    srows = [r_ for r_ in C.csv["sigma_series_" + amp + "_prod"]]
    xs = sorted({r_["x1"] for r_ in srows})
    by = {x: [r_["sigma_T2"] for r_ in srows if r_["x1"] == x and np.isfinite(r_["sigma_T2"])] for x in xs}
    E["sigma_T2_mean"] = {"x1": xs, "mean": [float(np.mean(by[x])) if by[x] else float("nan") for x in xs], "n": [len(by[x]) for x in xs]}
    return E


def f10_block(C, amp):
    e, out = C.eff[amp], C.out
    a = cp.arm_ids(amp)
    pairs = {"w24_vs_tw": ("s4_w24", a["tw"], e["n_w24"], e["n_ref"], "s4_tw_first30"),
             "w12_vs_n16": ("s4_w12", a["narrow"][0], e["n_w12"], e["n_ref"], "s4_n16_first30")}
    res_out = {}
    for label, (wide_arm, ref_arm, nw, nref, ref_name) in pairs.items():
        wide = w12_records(C) if wide_arm == "s4_w12" else cp.done_records(out, wide_arm)[:nw]
        ref = cp.done_records(out, ref_arm)[:nref]
        Rw = reduce_named(C, amp, wide_arm, wide, cp.PROD_DIV, cp.u_standalone(wide))
        Rr = reduce_named(C, amp, ref_name, ref, cp.PROD_DIV, cp.u_standalone(ref))
        res_out[label] = {"U_wide": cp.u_standalone(wide), "U_reference": cp.u_standalone(ref), "n_wide": len(wide), "n_reference": len(ref)}
        note = ""
        if wide_arm == "s4_w12":  # disclosure of the owner-decided split
            res_out[label].update(wide_arms=[arm for arm, _, _ in W12_PARTS], wide_seeds=[x.seed for x in wide],
                                  excluded_seeds=W12_EXCLUDED_SEEDS, excluded_reason=W12_EXCLUDED_REASON)
            note = "; wide = %s; seed %s excluded (s4_w12 halted, owner decision 2026-10-28)" % (
                "+".join(arm for arm, _, _ in W12_PARTS), ",".join(str(q) for q in W12_EXCLUDED_SEEDS))
        for est in TRANSVERSE:
            vw, vr = values(Rw, est), values(Rr, est)
            wc = stats.width_check(vw, vr, L90)
            bl, bh, deg = stats.bca_welch(vw, vr, L90)
            wc.update(bca_lo=bl, bca_hi=bh, bca_degenerate=deg, mean_wide=float(vw.mean()), mean_reference=float(vr.mean()))
            res_out[label][est] = wc
            row(C, "exploratory", amp, est, "F10 width check %s" % label, wc["label"], (wc["lo"], wc["hi"]), "%d/%d" % (len(vw), len(vr)),
                "diff=%.6g margin=%.6g%s" % (wc["diff"], wc["margin"], note))
    return res_out


# ------------------------------------------------------------------------------------------------
# outputs
# ------------------------------------------------------------------------------------------------
def jdump(o, ind=0):
    """Deterministic JSON: sorted keys, floats as %.17g, NaN -> null, +-inf -> "inf" / "-inf"."""
    pad = " " * (ind + 1)
    if isinstance(o, dict):
        if not o:
            return "{}"
        items = ",\n".join("%s%s: %s" % (pad, json.dumps(str(k)), jdump(o[k], ind + 1)) for k in sorted(o, key=str))
        return "{\n%s\n%s}" % (items, " " * ind)
    if isinstance(o, (list, tuple, np.ndarray)):
        if len(o) == 0:
            return "[]"
        return "[\n%s\n%s]" % (",\n".join(pad + jdump(v, ind + 1) for v in o), " " * ind)
    if isinstance(o, (bool, np.bool_)):
        return "true" if o else "false"
    if o is None:
        return "null"
    if isinstance(o, (int, np.integer)):
        return str(int(o))
    if isinstance(o, (float, np.floating)):
        o = float(o)
        if math.isnan(o):
            return "null"
        if math.isinf(o):
            return '"inf"' if o > 0 else '"-inf"'
        return "%.17g" % o
    return json.dumps(str(o))


def fmt(v):
    if v is None:
        return ""
    if isinstance(v, (float, np.floating)):
        return "%.6g" % v
    return str(v)


def decisions_md(rows, partial):
    t = ["# SF-39 decision points (preregistration section 10)%s" % (" - PARTIAL (missing realizations; not a final analysis)" if partial else ""), ""]
    for tier in TIERS:
        sel = [r_ for tr, r_ in rows if tr == tier]
        if not sel:
            continue
        t += ["## %s" % tier, "", "| amplitude | estimator | decision point | verdict | CI | n | note |", "|---|---|---|---|---|---|---|"]
        for r_ in sel:
            ci_ = "" if r_["ci_lo"] is None else "[%s, %s]" % (fmt(r_["ci_lo"]), fmt(r_["ci_hi"]))
            t.append("| %s |" % " | ".join(str(x).replace("|", "\\|") for x in (r_["amplitude"], r_["estimator"], r_["decision_point"],
                                                                                  r_["verdict"], ci_, fmt(r_["n"]), r_["note"])))
        t.append("")
    return "\n".join(t)


def process_rows(C):
    """Process tier: the RR3 / RR2 check lines of `$OUT/checks/results.txt` when the checks part ran on this output root."""
    path = os.path.join(C.out, "checks", "results.txt")
    pre = "PARTIAL: " if C.partial else ""
    rows = []
    if os.path.isfile(path):
        for ln in read_text(path).splitlines():
            m = re.match(r"^(CHECK_\S+) rc=(\S+) verdict=(\S*)", ln)
            if m:
                rows.append(("process", dict(amplitude="all", estimator="", decision_point=m.group(1), verdict=m.group(3) or "rc=" + m.group(2),
                                             ci=None, n=None, note=pre + "rc=%s (from checks/results.txt)" % m.group(2))))
    if not rows:
        rows.append(("process", dict(amplitude="all", estimator="", decision_point="RR3 checks / RR2 regression / byte check",
                                     verdict="not available to the analysis part", ci=None, n=None, note=pre + "no checks/results.txt in the output root")))
    return rows


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def write_outputs(C, summary):
    dest = C.dest
    shutil.rmtree(os.path.join(dest, "reduced"), ignore_errors=True)
    os.makedirs(os.path.join(dest, "reduced"), exist_ok=True)
    for name in sorted(C.csv):
        red.write_csv(os.path.join(dest, "reduced", name + ".csv"), C.csv[name])
    shutil.rmtree(os.path.join(dest, "series"), ignore_errors=True)
    os.makedirs(os.path.join(dest, "series"), exist_ok=True)
    for name in sorted(C.series):
        red.write_csv(os.path.join(dest, "series", name + ".csv"), C.series[name])
    rows = C.rows + process_rows(C)
    rows.sort(key=lambda tr: TIERS.index(tr[0]))  # stable: keeps the generation order inside a tier
    table = decide.decision_table_rows([r_ for _, r_ in rows])
    for (tier, _), t_ in zip(rows, table):
        t_["tier"] = tier
    cols = ["tier", "amplitude", "estimator", "decision_point", "verdict", "ci_lo", "ci_hi", "n", "note"]
    red.write_csv(os.path.join(dest, "decisions.csv"), table, cols)
    with open(os.path.join(dest, "decisions.md"), "w") as f:
        f.write(decisions_md([(tr, t_) for (tr, _), t_ in zip(rows, table)], C.partial))
    summary["decision_rows"] = len(table)
    with open(os.path.join(dest, "summary.json"), "w") as f:
        f.write(jdump(summary) + "\n")
    if C.partial:
        with open(os.path.join(dest, "PARTIAL.txt"), "w") as f:
            f.write("PARTIAL analysis (--allow-partial): effective counts below; not a final analysis.\n" + jdump(C.eff) + "\n")
    elif os.path.exists(os.path.join(dest, "PARTIAL.txt")):
        os.remove(os.path.join(dest, "PARTIAL.txt"))
    files = sorted(os.path.relpath(os.path.join(r_, fn), dest) for r_, _, fs in os.walk(dest) for fn in fs)
    files = [p for p in files if p not in ("manifest.txt", "manifest.txt.tmp")]
    with open(os.path.join(dest, "manifest.txt"), "w") as f:
        if C.partial:
            f.write("# PARTIAL analysis (--allow-partial)\n")
        for p in files:
            f.write("%s  %s\n" % (sha256_file(os.path.join(dest, p)), p))


# ------------------------------------------------------------------------------------------------
# driver
# ------------------------------------------------------------------------------------------------
def run(out_root, artifact, dest, allow_partial=False, n_pilot=N_PILOT, err=None):
    """Run the analysis; returns the exit code.  `n_pilot` (Par2 pooling offset) is a test hook: the CLI always uses 20."""
    err = err or sys.stderr
    try:
        frz = load_frozen(out_root)
        bud = load_budget(artifact)
        C = Ctx(out_root, artifact, dest, frz, None, n_pilot, allow_partial)
        miss = missing_items(C, bud)
        if miss and not allow_partial:
            print("ANALYSIS_INCOMPLETE: %d item(s) missing; nothing is analysed" % len(miss), file=err)
            for m in miss:
                print("  MISSING " + m, file=err)
            return 2
        C.eff = effective_counts(C, bud)
        summary = {"schema": "sf39-analysis-v1", "partial": allow_partial, "missing": miss if allow_partial else [], "n_pilot": n_pilot,
                   "budget": {amp: {k: bud[amp][k] for k in BUDGET_KEYS} for amp in cp.AMPS}, "amplitudes": {}}
        for amp in cp.AMPS:
            summary["amplitudes"][amp] = analyze_amp(C, amp)
    except Refused as e:
        print("ANALYSIS_REFUSED: %s" % e, file=err)
        return 2
    except (sio.ReduceError, ValueError, RuntimeError, OSError, KeyError) as e:
        print("ANALYSIS_FAILED %s: %s" % (type(e).__name__, e), file=err)
        return 1
    os.makedirs(dest, exist_ok=True)
    write_outputs(C, summary)
    print("ANALYSIS_DONE%s" % (" PARTIAL" if allow_partial else ""), file=err)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python3 -m sf39.analyze", description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", required=True, help="$SF39_OUT_ROOT")
    ap.add_argument("--artifact", required=True, help="the artifact root A (configs/budget.json, configs/par2_s*.yaml)")
    ap.add_argument("--dest", required=True, help="output directory (rewritten on every run)")
    ap.add_argument("--allow-partial", action="store_true", help="analyse the complete prefixes and label every output PARTIAL")
    a = ap.parse_args(argv)
    return run(a.out_root, a.artifact, a.dest, a.allow_partial)


if __name__ == "__main__":
    sys.exit(main())
