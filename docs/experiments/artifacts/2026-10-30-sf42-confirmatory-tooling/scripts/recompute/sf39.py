"""Independent recomputation of SF-39 per-realization reduced values from the runner's raw moment files.

Independence: numpy and the standard library only.  This module imports NO analysis package (not `sf39`, `sf42red`, `mcstat`);
every formula below is written from the definitions of preregistration sections 2-4 (restated in each function).  Nothing is ever
written under `--out-root` (the SF-39 host tree is read only; `--report` inside it is refused).

Host command (the recomputation is a CPU-only, seconds-long job; it still goes through `scripts/remote run`):

    python3 -m recompute.sf39 --out-root $HOME/sf39_campaign --artifact <A> \\
        --sets s025_prod,s1_prod,s225_prod,s4_prod --realizations 0,1,2 --report <file>

`A` = docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study (committed `analysis/reduced/<set>.csv`,
`analysis/summary.json`, `raw/campaign*/checkpoint/*tau_b_star*.json`).  `--realizations` are 0-based positions in the seed-sorted
list of the committed set (production seeds are 5001 + g, so 0,1,2 = seeds 5001-5003).  The report prints, per value, the
recomputed and the committed number and the relative difference; the last line is `RECOMPUTE PASS|FAIL`.

Inputs read per realization R (pass root P = R if R/track.json exists, else R/hmax_<d>, d = the committed `hmax_div`):
  R/field.json, R/flow_stats.json, P/track.json, P/exits_time.csv,
  P/moments_planes.csv, P/moments_time.csv, P/twin/moments_planes.csv, P/twin/moments_time.csv   (flux and uniform weights)
  Par2: par2/raw/par2_s{c}[_ext]/stats/realization_%04d_timeseries.csv  (global index g = r pilot, n_pilot + r extension)
Set-level inputs taken from the committed files, never from the host: `U` (mean qbar of the set, from the committed rows, whose own
qbar column is also checked) and `tau_b_star` (checkpoint json, checked against the committed rows).

Coverage (which committed columns of `reduced/<set>.csv` this reproduces):
  A. metadata copies/derivations (any row): seed, qbar, applied_scale, raw_variance, final_variance, ell, L1, x0, Xa, Xb, Xq, U,
     tau_b_star, t_lo, t_hi, j_hi, t_jhi, tilt, x_nonpositive_fraction, speed_p0, speed_p0_1, speed_p1, speed_p50,
     exit_inlet_jhi, exit_outlet_jhi.
  B. plane estimator, methods `flux` (moments_planes.csv) and `unif` (twin/moments_planes.csv): tp.n, tp.sigT2_Xa, tp.sigT2_Xb,
     tp.alpha_TP; common column S_r (= flux tp.sigT2_Xb/(ell (Xb-Xa))); flux only: tp.alpha_TP_q, tp.delta_half.
     Reproducible ONLY when the contributor counts at the planes involved are equal (the reducer's set T_r is the intersection of
     the contributors; the moment files give each plane over its own contributors).  Otherwise the column is reported
     `needs per-particle arrays`.
  C. window estimator, methods `flux` (moments_time.csv) and `unif` (twin/moments_time.csv): win.n, win.alpha_e1 (= alpha_LAM),
     win.alpha_e2, win.alpha_e3, win.alpha_Tfixed.  The reducer fixes the set S_r = particles with a record at snapshot j_hi at
     both window ends; the moment files give each snapshot over its own active particles.  Reproducible ONLY when `active` at
     every snapshot bracketing t_lo and t_hi equals `active` at j_hi (no particle left or entered between them); else
     `needs per-particle arrays`.  `win.n` is always reproducible (= active at j_hi).
  NOT reproducible from moment files (need per-particle arrays or weights): tb, plane_hist_*/time_hist_*, speed.. (copied above),
     ev_p*, split.* (lattice halves), win.excl/tp.excl, tp.n_noncross_*, aligned/plume-drift frames (win.alpha_eL, _eT1, _eT2,
     _TAM, dTfixed_minus_TAM, _eD, _TD, drift_tilt: need the full covariance, the moment files hold only the diagonal), and the
     methods expI, full, pollock, ixc, ixp.
  Gate 5: Par2 common-window values from the per-realization series (U_c, window, mean and sd of alpha_x, alpha_y, alpha_z,
     alpha_T_fixed over all N_R realizations against `summary.json` `amplitudes.<amp>.C.par2`).  The per-realization Par2 values
     are not committed, so they are compared through these aggregates.
"""
import argparse
import csv
import json
import math
import os
import re
import sys

import numpy as np

NAN = float("nan")
RTOL_DEFAULT = 1e-10
ATOL_DEFAULT = 1e-15  # moment files carry 16 significant digits (%.15e): values below atol/rtol are compared absolutely at atol
AMPS = ("s025", "s1", "s225", "s4")
PAR2_ID = {"s025": "025", "s1": "1", "s225": "225", "s4": "4"}

COVERED_COMMON = ("seed", "qbar", "applied_scale", "raw_variance", "final_variance", "ell", "L1", "x0", "Xa", "Xb", "Xq", "U",
                  "tau_b_star", "t_lo", "t_hi", "j_hi", "t_jhi", "tilt", "x_nonpositive_fraction", "speed_p0", "speed_p0_1",
                  "speed_p1", "speed_p50", "exit_inlet_jhi", "exit_outlet_jhi", "S_r")
COVERED_METHOD = {"flux": ("tp.n", "tp.sigT2_Xa", "tp.sigT2_Xb", "tp.alpha_TP", "tp.alpha_TP_q", "tp.delta_half", "win.n",
                           "win.alpha_e1", "win.alpha_e2", "win.alpha_e3", "win.alpha_Tfixed"),
                  "unif": ("tp.n", "tp.sigT2_Xa", "tp.sigT2_Xb", "tp.alpha_TP", "win.n", "win.alpha_e1", "win.alpha_e2",
                           "win.alpha_e3", "win.alpha_Tfixed")}


# ------------------------------------------------------------------------------------------------
# readers (read only)
# ------------------------------------------------------------------------------------------------
def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_rows(path):
    """Rows (dicts of float) of a CSV whose first non-comment line is the header; '#' lines skipped."""
    with open(path) as f:
        lines = [ln for ln in f if ln.strip() and not ln.startswith("#")]
    return [{k: float(v) for k, v in row.items()} for row in csv.DictReader(lines)]


def tofloat(s):
    s = (s or "").strip()
    return NAN if s == "" else float(s)


# ------------------------------------------------------------------------------------------------
# definitions
# ------------------------------------------------------------------------------------------------
def key_planes(x0, L1, ell):
    """X_a = x0 + L/2, X_b = L1 - 5 ell, X_q = x0 + L/4, L = L1 - 10 ell."""
    L = L1 - 10.0 * ell
    return x0 + L / 2.0, L1 - 5.0 * ell, x0 + L / 4.0


def find_plane(planes, x):
    """Index of the recorded plane within 1e-12 max(1,|x|) of x, else None."""
    i = min(range(len(planes)), key=lambda k: abs(planes[k] - x))
    return i if abs(planes[i] - x) <= 1e-12 * max(1.0, abs(x)) else None


def window_edges(times, tau, ell, U):
    """t_lo = 0.75 tau ell/U, t_hi = tau ell/U, j_hi = first snapshot with t_j >= t_hi (error if none or t_lo < t_1)."""
    t_lo, t_hi = 0.75 * tau * ell / U, 1.0 * tau * ell / U
    j_hi = next((j for j, t in enumerate(times) if t >= t_hi), None)
    if j_hi is None:
        raise ValueError("no snapshot >= t_hi=%.17g" % t_hi)
    if t_lo < times[0]:
        raise ValueError("t_lo=%.17g < first snapshot %.17g" % (t_lo, times[0]))
    return t_lo, t_hi, j_hi


def bracket(times, t):
    """(ja, jb, f) with t = times[ja] + f (times[jb] - times[ja]); ja == jb, f = 0 on an exact snapshot."""
    jb = next((j for j, s in enumerate(times) if s >= t), None)
    if jb is None or (jb == 0 and times[0] != t):
        raise ValueError("time %.17g outside the recorded snapshots" % t)
    if times[jb] == t:
        return jb, jb, 0.0
    return jb - 1, jb, (t - times[jb - 1]) / (times[jb] - times[jb - 1])


def lerp(va, vb, f):
    return va + f * (vb - va) if f != 0.0 else va


def tilt_angle(q):
    """atan2(|e_L x e_1|, e_L . e_1) with e_L = q/|q|."""
    n = math.sqrt(q[0] ** 2 + q[1] ** 2 + q[2] ** 2)
    e = [c / n for c in q]
    return math.atan2(math.hypot(e[1], e[2]), e[0])  # |e_L x e_1| = sqrt(e_y^2 + e_z^2)


# ------------------------------------------------------------------------------------------------
# estimators from the moment files
# ------------------------------------------------------------------------------------------------
def tp_from_planes(rows, ia, ib, iq, ell, xa, xb, xq, with_q):
    """Plane estimator from moments_planes rows.  Returns (values, unavailable{column: reason}).
    sigma_T^2(X) = (var x2 + var x3)/2; alpha_TP = [sigma_T^2(X_b) - sigma_T^2(X_a)] / (2 ell (X_b - X_a))."""
    vals, un = {}, {}
    if ia is None or ib is None:
        un["tp.*"] = "plane X_a or X_b not recorded"
        return vals, un
    sig = lambda i: (rows[i]["var_x2"] + rows[i]["var_x3"]) / 2.0
    na, nb = int(rows[ia]["contributing"]), int(rows[ib]["contributing"])
    if na != nb:
        for c in ("tp.n", "tp.sigT2_Xa", "tp.sigT2_Xb", "tp.alpha_TP", "tp.alpha_TP_q", "tp.delta_half"):
            un[c] = "needs per-particle arrays: contributors at X_a (%d) != at X_b (%d); the reducer uses their intersection" % (na, nb)
        return vals, un
    vals["tp.n"] = nb
    vals["tp.sigT2_Xa"], vals["tp.sigT2_Xb"] = sig(ia), sig(ib)
    vals["tp.alpha_TP"] = (sig(ib) - sig(ia)) / (2.0 * ell * (xb - xa))
    if with_q and iq is not None:
        nq = int(rows[iq]["contributing"])
        if nq != na:
            for c in ("tp.alpha_TP_q", "tp.delta_half"):
                un[c] = "needs per-particle arrays: contributors at X_q (%d) != at X_a (%d)" % (nq, na)
        else:
            first = (sig(ia) - sig(iq)) / (2.0 * ell * (xa - xq))
            vals["tp.alpha_TP_q"] = first
            vals["tp.delta_half"] = first - vals["tp.alpha_TP"]
    return vals, un


def win_from_time(rows, times, t_lo, t_hi, j_hi, ell, U):
    """Window estimator, fixed axes, from moments_time rows: V_e(t) = linear interpolation of var_e between the bracketing
    snapshots; alpha_e = [V_e(t_hi) - V_e(t_lo)] / (2 ell U (t_hi - t_lo)); alpha_Tfixed = (alpha_e2 + alpha_e3)/2."""
    vals, un = {}, {}
    act = [int(r["active"]) for r in rows]
    vals["win.n"] = act[j_hi]
    lo, hi = bracket(times, t_lo), bracket(times, t_hi)
    js = sorted({lo[0], lo[1], hi[0], hi[1]})
    off = [j for j in js if act[j] != act[j_hi]]
    if off:
        for c in ("win.alpha_e1", "win.alpha_e2", "win.alpha_e3", "win.alpha_Tfixed"):
            un[c] = ("needs per-particle arrays: active particles differ between window snapshots (j=%s: %s vs %d at j_hi=%d); "
                     "the reducer fixes the set at j_hi" % (off, [act[j] for j in off], act[j_hi], j_hi))
        return vals, un
    den = 2.0 * ell * U * (t_hi - t_lo)

    def V(col, br):
        ja, jb, f = br
        return lerp(rows[ja][col], rows[jb][col], f)

    a = {k: (V("var_" + c, hi) - V("var_" + c, lo)) / den for k, c in (("e1", "x"), ("e2", "y"), ("e3", "z"))}
    vals["win.alpha_e1"], vals["win.alpha_e2"], vals["win.alpha_e3"] = a["e1"], a["e2"], a["e3"]
    vals["win.alpha_Tfixed"] = (a["e2"] + a["e3"]) / 2.0
    return vals, un


def recompute_pass(R, P, tau, U):
    """Recompute everything covered for one realization/pass.  R realization dir, P pass root, tau = tau_b_star, U set velocity.
    Returns (values{key: number}, unavailable{key: reason}); method keys are '<method>|<column>'."""
    field, flow, track = read_json(os.path.join(R, "field.json")), read_json(os.path.join(R, "flow_stats.json")), read_json(os.path.join(P, "track.json"))
    ell, lam = float(field["ell"]), float(flow["lambda"])
    if abs(ell - lam) > 1e-9 * abs(lam):
        raise ValueError("field.json ell %.17g != flow_stats.json lambda %.17g in %s" % (ell, lam, R))
    L1, x0 = float(flow["L1"]), float(track["X0"])
    planes, times = [float(x) for x in track["planes"]], [float(t) for t in track["times"]]
    xa, xb, xq = key_planes(x0, L1, ell)
    ia, ib, iq = find_plane(planes, xa), find_plane(planes, xb), find_plane(planes, xq)
    t_lo, t_hi, j_hi = window_edges(times, tau, ell, U)
    q = (flow["interior"]["mean_u"], flow["interior"]["mean_v"], flow["interior"]["mean_w"])
    sp = flow["speed_over_qbar_percentiles"]
    ex = read_rows(os.path.join(P, "exits_time.csv"))
    vals = {"seed": int(field["seed"]), "qbar": float(flow["qbar"]), "applied_scale": field["applied_scale"],
            "raw_variance": field["raw_variance"], "final_variance": field["final_variance"], "ell": ell, "L1": L1, "x0": x0,
            "Xa": xa, "Xb": xb, "Xq": xq, "U": U, "tau_b_star": tau, "t_lo": t_lo, "t_hi": t_hi, "j_hi": j_hi, "t_jhi": times[j_hi],
            "tilt": tilt_angle(q), "x_nonpositive_fraction": flow["x_nonpositive_fraction"], "speed_p0": sp["p0"],
            "speed_p0_1": sp["p0.1"], "speed_p1": sp["p1"], "speed_p50": sp["p50"],
            "exit_inlet_jhi": int(ex[j_hi]["exit_inlet"]), "exit_outlet_jhi": int(ex[j_hi]["exit_outlet"])}
    un = {}
    for method, sub in (("flux", ""), ("unif", "twin")):
        d = os.path.join(P, sub) if sub else P
        pf, tf = os.path.join(d, "moments_planes.csv"), os.path.join(d, "moments_time.csv")
        if not (os.path.exists(pf) and os.path.exists(tf)):
            un[method + "|*"] = "moment files absent in %s" % d
            continue
        prow, trow = read_rows(pf), read_rows(tf)
        if len(prow) != len(planes) or len(trow) != len(times):
            raise ValueError("moment files in %s do not match track.json (planes %d/%d, times %d/%d)" % (d, len(prow), len(planes), len(trow), len(times)))
        for v, c in list(zip((r["x1"] for r in prow), planes)) + list(zip((r["t"] for r in trow), times)):
            if abs(v - c) > 1e-12 * max(1.0, abs(c)):
                raise ValueError("moment file coordinate %.17g != track.json %.17g in %s" % (v, c, d))
        tv, tu = tp_from_planes(prow, ia, ib, iq, ell, xa, xb, xq, with_q=(method == "flux"))
        wv, wu = win_from_time(trow, times, t_lo, t_hi, j_hi, ell, U)
        for k, v in {**tv, **wv}.items():
            vals[method + "|" + k] = v
        for k, v in {**tu, **wu}.items():
            un[method + "|" + k] = v
        if method == "flux" and "tp.sigT2_Xb" in tv:
            vals["S_r"] = tv["tp.sigT2_Xb"] / (ell * (xb - xa))
    return vals, un


# ------------------------------------------------------------------------------------------------
# host layout
# ------------------------------------------------------------------------------------------------
def arms_and_div(set_name):
    """(runner arms that may hold the set's seeds, default pass divisor) from the reduced-set name."""
    m = re.fullmatch(r"(s\d+)_prod(?:_twinpairs)?", set_name)
    if m:
        return [m.group(1) + "_A", m.group(1) + "_B"], 8
    m = re.fullmatch(r"(s\d+_A)_hmax(\d+)", set_name)
    if m:
        return [m.group(1)], int(m.group(2))
    arm = re.sub(r"_first\d+$", "", set_name)
    return [arm] + ([arm + "c"] if arm == "s4_w12" else []), 8


def index_arm(out_root, arm):
    """{seed: realization dir} of `runs/<arm>/realizations/r_*` (seed from field.json)."""
    root = os.path.join(out_root, "runs", arm, "realizations")
    out = {}
    if not os.path.isdir(root):
        return out
    for n in sorted(os.listdir(root)):
        p = os.path.join(root, n, "field.json")
        if re.fullmatch(r"r_\d{4}", n) and os.path.exists(p) and os.path.exists(os.path.join(root, n, "DONE")):
            out[int(read_json(p)["seed"])] = os.path.join(root, n)
    return out


def pass_root(R, div):
    return R if os.path.exists(os.path.join(R, "track.json")) else os.path.join(R, "hmax_%d" % div)


def load_taus(artifact):
    """{amp: tau_b_star} from the checkpoint json files committed in the SF-39 artifact (original three + s025)."""
    taus = {}
    for rel in ("raw/campaign/checkpoint/tau_b_star.json", "raw/campaign_s025/checkpoint/s025/tau_b_star_s025.json"):
        p = os.path.join(artifact, rel)
        if os.path.exists(p):
            for a, v in read_json(p)["amplitudes"].items():
                taus[a] = v
    return taus


def committed_set(artifact, set_name):
    """{seed: {method: row dict (strings)}} of analysis/reduced/<set>.csv."""
    out = {}
    with open(os.path.join(artifact, "analysis", "reduced", set_name + ".csv"), newline="") as f:
        for row in csv.DictReader(f):
            out.setdefault(int(row["seed"]), {})[row["method"]] = row
    return out


# ------------------------------------------------------------------------------------------------
# comparison
# ------------------------------------------------------------------------------------------------
def rel_diff(a, b, floor=0.0):
    """|a-b| / max(|a|, |b|, floor); floor = atol/rtol makes values below it compare at the absolute tolerance atol."""
    if math.isnan(a) and math.isnan(b):
        return 0.0
    if math.isnan(a) or math.isnan(b):
        return math.inf
    d = abs(a - b)
    return 0.0 if d == 0.0 else d / max(abs(a), abs(b), floor)


def compare_realization(vals, unav, committed, rtol, atol):
    """List of (label, recomputed, committed, rel, ok) for the covered columns present in `committed` ({method: row})."""
    out = []
    any_row = next(iter(committed.values()))
    for key, v in vals.items():
        if "|" in key:
            method, col = key.split("|", 1)
            row = committed.get(method)
            label = "%s.%s" % (method, col)
        else:
            row, col, label = any_row, key, key
        if row is None or col not in row:
            continue
        c = tofloat(row[col])
        r = rel_diff(float(v), c, atol / rtol)
        ok = r <= rtol
        out.append((label, float(v), c, r, ok))
    return out


# ------------------------------------------------------------------------------------------------
# Par2 Gate 5 window values
# ------------------------------------------------------------------------------------------------
def read_par2_series(path):
    """Par2 per-realization series: dict column -> list (needs t, mean_x, var_x, var_y, var_z)."""
    with open(path) as f:
        rows = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    head = [h.strip() for h in rows[0].split(",")]
    data = [[float(x) for x in ln.split(",")] for ln in rows[1:]]
    s = {h: [r[k] for r in data] for k, h in enumerate(head)}
    if any(b <= a for a, b in zip(s["t"], s["t"][1:])):
        raise ValueError("%s: t not strictly increasing" % path)
    return s


def interp_series(s, col, t):
    ts = s["t"]
    tol = 1e-12 * max(1.0, abs(ts[-1]))
    if t < ts[0] - tol or t > ts[-1] + tol:
        raise ValueError("t=%.15g outside the series [%.15g, %.15g]" % (t, ts[0], ts[-1]))
    return float(np.interp(t, ts, s[col]))


def par2_values(series_list, lam, inject_x, t_b_min, tau_c):
    """Common-window Par2 values.  U_c = mean_g [mean_x(t_b_min) - inject_x]/t_b_min; window [0.75, 1] tau_c lam/U_c;
    alpha_k(g) = [var_k(t_hi) - var_k(t_lo)]/(2 lam U_c (t_hi - t_lo)), k = x, y, z; alpha_T_fixed = (alpha_y + alpha_z)/2."""
    U_c = float(np.mean([(interp_series(s, "mean_x", t_b_min) - inject_x) / t_b_min for s in series_list]))
    t_hi = tau_c * lam / U_c
    t_lo = 0.75 * t_hi
    den = 2.0 * lam * U_c * (t_hi - t_lo)
    al = {k: np.array([(interp_series(s, "var_" + k, t_hi) - interp_series(s, "var_" + k, t_lo)) / den for s in series_list]) for k in "xyz"}
    al["T"] = 0.5 * (al["y"] + al["z"])
    return {"U_c": U_c, "t_lo": t_lo, "t_hi": t_hi, "alpha": al}


def par2_gate5(out_root, artifact, amp, taus, summary):
    """Compare recomputed Par2 aggregates with summary.json; returns list of (label, recomputed, committed, rel, ok)."""
    c = PAR2_ID[amp]
    g5 = taus[amp]["gate5"]
    p2 = summary["amplitudes"][amp]["C"]["par2"]
    n, n_pilot = int(p2["n"]), 20
    pilot = os.path.join(out_root, "par2", "raw", "par2_s" + c)
    ext = os.path.join(out_root, "par2", "raw", "par2_s%s_ext" % c)
    series = []
    for g in range(n):
        d, r = (pilot, g) if g < n_pilot else (ext, g - n_pilot)
        series.append(read_par2_series(os.path.join(d, "stats", "realization_%04d_timeseries.csv" % r)))
    v = par2_values(series, float(g5["par2"]["corr_length"]), float(g5["par2"]["inject_x"]), float(g5["par2"]["t_b_min"]), float(g5["tau_c"]))
    rows = [("par2.U_c", v["U_c"], p2["U_c"]), ("par2.t_lo", v["t_lo"], p2["window"][0]), ("par2.t_hi", v["t_hi"], p2["window"][1])]
    for k, name in (("x", "alpha_x"), ("y", "alpha_y"), ("z", "alpha_z"), ("T", "alpha_T_fixed")):
        rows.append(("par2.%s.mean" % name, float(v["alpha"][k].mean()), p2[name]["mean"]))
        rows.append(("par2.%s.sd" % name, float(v["alpha"][k].std(ddof=1)), p2[name]["sd"]))
    return [(lab, a, b, rel_diff(a, b), rel_diff(a, b) <= RTOL_DEFAULT) for lab, a, b in rows], v


# ------------------------------------------------------------------------------------------------
# driver
# ------------------------------------------------------------------------------------------------
def run(out_root, artifact, sets, positions, report, rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT, par2=True):
    """Write the report to the file object `report`; return the exit code (0 pass, 1 mismatch, 2 nothing reproduced/unreadable)."""
    taus = load_taus(artifact)
    with open(os.path.join(artifact, "analysis", "summary.json")) as f:
        summary = json.load(f)
    P = lambda *a: print(*a, file=report)
    P("RECOMPUTE out_root=%s artifact=%s rtol=%g atol=%g" % (out_root, artifact, rtol, atol))
    n_cmp = n_bad = 0
    max_rel = 0.0
    empty_sets = []
    amps_seen = []
    for sname in sets:
        com = committed_set(artifact, sname)
        seeds = sorted(com)
        arms, div0 = arms_and_div(sname)
        idx = {}
        for arm in arms:
            idx.update(index_arm(out_root, arm))
        amp = re.match(r"(s\d+)", sname).group(1)
        amps_seen.append(amp)
        if sname.endswith("_prod"):  # U of a standalone value = mean qbar over the set (preregistration section 3)
            qs = [tofloat(next(iter(com[s_].values()))["qbar"]) for s_ in seeds]
            u0 = tofloat(next(iter(com[seeds[0]].values()))["U"])
            ok_u = rel_diff(sum(qs) / len(qs), u0) <= 1e-12
            P("SET %s: mean of the %d committed qbar = %.17g, committed U = %.17g %s" % (sname, len(qs), sum(qs) / len(qs), u0, "OK" if ok_u else "FAIL"))
            n_bad += not ok_u
        set_cmp = 0
        for pos in positions:
            if pos >= len(seeds):
                P("SET %s position %d: only %d seeds in the committed set" % (sname, pos, len(seeds)))
                n_bad += 1
                continue
            seed = seeds[pos]
            rows = com[seed]
            any_row = next(iter(rows.values()))
            if seed not in idx:
                P("SET %s seed %d: realization directory not found under %s/runs/%s" % (sname, seed, out_root, "|".join(arms)))
                n_bad += 1
                continue
            div = int(float(any_row["hmax_div"]))
            R = idx[seed]
            Pr = pass_root(R, div)
            U, tau = float(any_row["U"]), float(any_row["tau_b_star"])
            if abs(tau - taus[amp]["tau_b_star"]) > 1e-12 * tau:
                P("SET %s seed %d: committed tau_b_star %.17g != checkpoint json %.17g" % (sname, seed, tau, taus[amp]["tau_b_star"]))
                n_bad += 1
                continue
            vals, unav = recompute_pass(R, Pr, tau, U)
            res = compare_realization(vals, unav, rows, rtol, atol)
            P("SET %s seed %d dir %s pass %s: compared %d values, max_rel %.3e, %d beyond tolerance, %d columns need per-particle arrays"
              % (sname, seed, R, os.path.relpath(Pr, R) if Pr != R else ".", len(res), max([x[3] for x in res] + [0.0]),
                 sum(not x[4] for x in res), len(unav)))
            for lab, a, b, r_, ok in res:
                P("  %-4s %-26s recomputed %.17g committed %.17g rel %.3e" % ("OK" if ok else "FAIL", lab, a, b, r_))
            for k, why in sorted(unav.items()):
                P("  ARRAYS %-26s %s" % (k.replace("|", "."), why))
            n_cmp += len(res)
            set_cmp += len(res)
            n_bad += sum(not x[4] for x in res)
            max_rel = max([max_rel] + [x[3] for x in res if math.isfinite(x[3])])
        if set_cmp == 0:
            empty_sets.append(sname)
    if par2:
        for amp in sorted(set(amps_seen)):
            try:
                res, _ = par2_gate5(out_root, artifact, amp, taus, summary)
            except (OSError, ValueError, KeyError) as e:
                P("PAR2 %s: not recomputed (%s: %s)" % (amp, type(e).__name__, e))
                n_bad += 1
                continue
            P("PAR2 %s Gate 5 window values vs summary.json (all N_R realizations): %d values, max_rel %.3e"
              % (amp, len(res), max([x[3] for x in res] + [0.0])))
            for lab, a, b, r_, ok in res:
                P("  %-4s %-26s recomputed %.17g committed %.17g rel %.3e" % ("OK" if ok else "FAIL", lab, a, b, r_))
            n_cmp += len(res)
            n_bad += sum(not x[4] for x in res)
            max_rel = max([max_rel] + [x[3] for x in res if math.isfinite(x[3])])
    P("RECOMPUTE compared=%d beyond_tolerance=%d max_rel=%.3e empty_sets=%s" % (n_cmp, n_bad, max_rel, ",".join(empty_sets) or "none"))
    code = 1 if n_bad else (2 if empty_sets or n_cmp == 0 else 0)
    P("RECOMPUTE %s" % ("PASS" if code == 0 else "FAIL"))
    return code


def parse_positions(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b if b else a) + 1))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="recompute.sf39", description=__doc__.split("\n")[0])
    ap.add_argument("--out-root", required=True, help="SF-39 host output root ($HOME/sf39_campaign); read only")
    ap.add_argument("--artifact", required=True, help="docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study")
    ap.add_argument("--sets", required=True, help="comma-separated reduced-set names, e.g. s025_prod,s1_prod,s225_prod,s4_prod")
    ap.add_argument("--realizations", default="0,1,2", help="positions in the seed-sorted committed set (default 0,1,2)")
    ap.add_argument("--report", default=None, help="report file (default stdout); must not be inside --out-root")
    ap.add_argument("--rtol", type=float, default=RTOL_DEFAULT)
    ap.add_argument("--atol", type=float, default=ATOL_DEFAULT)
    ap.add_argument("--no-par2", action="store_true", help="skip the Par2 Gate 5 aggregate comparison")
    a = ap.parse_args(argv)
    out_root = os.path.realpath(a.out_root)
    if a.report and os.path.realpath(a.report).startswith(out_root + os.sep):
        ap.error("--report must not be inside --out-root (the SF-39 host tree is read only)")
    f = open(a.report, "w") if a.report else sys.stdout
    try:
        code = run(out_root, os.path.realpath(a.artifact), a.sets.split(","), parse_positions(a.realizations), f,
                   a.rtol, a.atol, par2=not a.no_par2)
    finally:
        if a.report:
            f.close()
    if a.report:
        with open(a.report) as g:
            print(g.read().splitlines()[-1])
    return code


if __name__ == "__main__":
    sys.exit(main())
