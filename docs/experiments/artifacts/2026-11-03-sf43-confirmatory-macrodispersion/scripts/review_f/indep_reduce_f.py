#!/usr/bin/env python3
"""SF-43 acceptance F (addendum 1): independent reduction of ONE retained production realization, compared with the frozen reduced table.

Prepared by an agent, executed and compared by the owner (decision of 2026-11-04).  python3 + numpy only: this file imports nothing from
the repository (sf43ana, sf43camp, sf42red, mcstat, ... are NOT used; a unit test parses the imports).  Nothing is written anywhere:
the raw arrays and the reduced table are opened read-only, the result goes to stdout.

ESTIMATORS (definitions of A/preregistration.md section 4 only, re-implemented here from the text):
  window planes   the run's planes with d = (x1 - x0)/ell in [d_lo, d_hi] (default [25, 94]; 70 planes), x0 = 5 ell (read from the pass)
  moments         per window plane, over the particles of the set `perplane` (those that crossed that plane), flux weights renormalized
                  over that set; BIASED weighted central moments (divide by the weight sum, no n/(n-1)); centred per realization
  basis           displacement (primary): x23(plane) - x23(seed), i.e. planes_x23 minus seeds[:, 1:3]
  alpha_TP        (1/2) slope[(var x2 + var x3)/2] / ell                       (fixed axes; slope = OLS against x1 over the window planes)
  alpha_TAP       the same on the transverse variance orthogonal to the realization's mean displacement q = (x1_last - x1_first,
                  mu2_last - mu2_first, mu3_last - mu3_first) (first/last window plane; mu = mean displacement); e_L = q/|q|;
                  u1 = e2 - (e2.e_L) e_L normalized, u2 = e3 - (e3.e_L) e_L - (e3.u1) u1 normalized (Gram-Schmidt); variance along u
                  = (u_y, u_z) C (u_y, u_z)^T with C the 2x2 transverse covariance; alpha_TAP = (1/2) slope[(var_u1 + var_u2)/2] / ell
  alpha_LP        (U_r^2/2) slope[var tau] / ell, U_r = 1/slope[mean tau]       (tau = time since injection; basis independent)
  PASS rule       every compared quantity has relative difference <= 1e-10 (absolute <= 1e-15 when the table value is 0), else FAIL.

CONVENTIONS TAKEN FROM FORMAT DOCUMENTS (file format only, never formulas; sources named; all re-checked by the shape/consistency tests below)
  [S1] sf42red/reduce.py module docstring + ConsumerPass, A/scripts/sf43camp/README.md ("campaign_dir receives", "Seed map", "Retention"):
       - campaign layout: <campaign>/segments/<arm>/<segment>/{seedmap.json, out/realizations/r_XXXX/} and
         <campaign>/reduced/<arm>/<segment>/<arm>__<segment>__d<lo>-<hi>.csv ;  seedmap.json = {seed0, nr, seeds: {"r": seed}}
       - a realization directory holds either a single pass (track.json in it) or one sub-directory hmax_<d> per h_max pass (track.json in
         each); field.json and flow_stats.json live in the realization directory (not in the pass directory)
       - arrays (numpy .npy, float64): seeds (Np, 3) = (x1, x2, x3) of the injection point; weights (Np,) flux weights, 0 = not traced;
         planes_x23 (Np, P, 2) unwrapped transverse positions at the P planes, NaN from the first plane not reached; planes_tau (Np, P)
         time since injection, NaN likewise
       - track.json: "planes" (P strictly increasing x1 values), "X0" (injection plane x1), "M" (lattice side); field.json: "ell", "seed";
         flow_stats.json: "lambda" (= ell; checked)
  [S2] reduced table (sf42red COLUMNS; A/scripts/sf43camp/README.md): row key (code, seed, family, weights, basis, set, hmax_div), seed = the
       FIELD seed; consumer rows code = consumer, family = plane; for an arm that mixes single- and multi-pass segments the production pass
       carries hmax_div = 8 in both layouts (`--single-pass-hmax`); columns used: alpha_TP alpha_TAP alpha_LP U_r n_traced n_first n_last n_min n_max
  [S3] a particle belongs to the set of a plane iff weight > 0 and its x23 and tau records at that plane are finite (the runner writes NaN
       from the first plane not reached on); this is the file-level meaning of "particles that crossed that plane".
  [S4] window membership uses a relative tolerance of 1e-9 on d_lo and d_hi (floating-point round-off of planes = x0 + d ell); >= 3 planes.

CLI:  indep_reduce_f.py --campaign DIR --arm ARM --seed FIELD_SEED [--hmax-div 8] [--d-lo 25 --d-hi 94]
Exit status: 0 on `F PASS`, 1 on `F FAIL`, 2 on a usage/data error (message on stderr, nothing guessed).
"""
import argparse
import csv
import glob
import json
import math
import os
import sys

import numpy as np

REL_TOL = 1e-10
ABS_TOL = 1e-15
WINDOW_TOL = 1e-9


class DataError(Exception):
    pass


def read_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as e:
        raise DataError("cannot read %s: %s" % (path, e))


def read_npy(path):
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError) as e:
        raise DataError("cannot read %s: %s" % (path, e))


# ---------------------------------------------------------------------------------------------- locating the realization
def find_segment(campaign, arm, seed):
    """(segment_dir, r) of the unique segment of `arm` whose seedmap lists `seed`."""
    hits = []
    for sm in sorted(glob.glob(os.path.join(campaign, "segments", arm, "*", "seedmap.json"))):
        seeds = read_json(sm).get("seeds", {})
        for r, s in seeds.items():
            if int(s) == seed:
                hits.append((os.path.dirname(sm), int(r)))
    if not hits:
        raise DataError("seed %d not found in any %s/segments/%s/*/seedmap.json" % (seed, campaign, arm))
    if len(hits) > 1:
        raise DataError("ambiguous: seed %d appears in several segments: %s" % (seed, [h[0] for h in hits]))
    return hits[0]


def find_pass(rdir, hmax_div):
    """(pass_root, layout) of a realization directory: single pass (track.json in it) or hmax_<d> sub-directories."""
    single = os.path.isfile(os.path.join(rdir, "track.json"))
    multi = sorted(d for d in glob.glob(os.path.join(rdir, "hmax_*")) if os.path.isdir(d))
    if single and multi:
        raise DataError("ambiguous layout in %s: track.json and hmax_* directories both exist" % rdir)
    if single:
        return rdir, "single-pass"
    if not multi:
        raise DataError("no track.json and no hmax_* directory in %s" % rdir)
    want = os.path.join(rdir, "hmax_%d" % hmax_div)
    if want not in multi:
        raise DataError("no pass hmax_%d in %s (found: %s)" % (hmax_div, rdir, [os.path.basename(d) for d in multi]))
    return want, "multi-pass hmax_%d" % hmax_div


# ---------------------------------------------------------------------------------------------- estimators
def ols_slope(x, y):
    xc = x - x.mean()
    return float(xc @ (y - y.mean()) / (xc @ xc))


def reduce_realization(rdir, pass_root, d_lo, d_hi):
    """dict of the primary estimands and diagnostics from the raw arrays (flux weights, set perplane, displacement basis)."""
    track = read_json(os.path.join(pass_root, "track.json"))
    ell = float(read_json(os.path.join(rdir, "field.json"))["ell"])
    lam = float(read_json(os.path.join(rdir, "flow_stats.json"))["lambda"])
    if abs(ell - lam) > 1e-9 * abs(lam):
        raise DataError("field.json ell=%r differs from flow_stats.json lambda=%r" % (ell, lam))
    planes = np.asarray(track["planes"], float)
    x0 = float(track["X0"])
    seeds = read_npy(os.path.join(pass_root, "seeds.npy")).astype(float)
    w = read_npy(os.path.join(pass_root, "weights.npy")).astype(float)
    X = read_npy(os.path.join(pass_root, "planes_x23.npy")).astype(float)
    tau = read_npy(os.path.join(pass_root, "planes_tau.npy")).astype(float)
    Np, P = len(w), len(planes)
    if seeds.shape != (Np, 3) or X.shape != (Np, P, 2) or tau.shape != (Np, P) or np.any(np.diff(planes) <= 0):
        raise DataError("array shapes inconsistent: seeds %s weights %s planes_x23 %s planes_tau %s planes %s"
                        % (seeds.shape, w.shape, X.shape, tau.shape, planes.shape))
    d = (planes - x0) / ell
    k = np.flatnonzero((d >= d_lo - WINDOW_TOL * max(1.0, abs(d_lo))) & (d <= d_hi + WINDOW_TOL * max(1.0, abs(d_hi))))
    if len(k) < 3:
        raise DataError("only %d planes in the window [%g, %g]" % (len(k), d_lo, d_hi))
    disp = X - seeds[:, None, 1:3]
    var_tr = np.empty(len(k)); mean_t = np.empty(len(k)); var_t = np.empty(len(k))
    mu = np.empty((len(k), 2)); C = np.empty((len(k), 2, 2)); counts = []
    for i, j in enumerate(k):
        sel = (w > 0) & np.isfinite(disp[:, j, 0]) & np.isfinite(disp[:, j, 1]) & np.isfinite(tau[:, j])
        counts.append(int(sel.sum()))
        if counts[-1] == 0:
            raise DataError("empty particle set at window plane %d" % j)
        ws, Y, t = w[sel] / w[sel].sum(), disp[sel, j, :], tau[sel, j]
        mu[i] = ws @ Y
        D = Y - mu[i]
        C[i] = (D * ws[:, None]).T @ D
        mean_t[i] = ws @ t
        var_t[i] = ws @ (t - mean_t[i]) ** 2
    x1 = planes[k]
    out = {"alpha_TP": 0.5 * ols_slope(x1, 0.5 * (C[:, 0, 0] + C[:, 1, 1])) / ell}
    q = np.array([x1[-1] - x1[0], mu[-1, 0] - mu[0, 0], mu[-1, 1] - mu[0, 1]])
    eL = q / np.linalg.norm(q)
    e2, e3 = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    u1 = e2 - (e2 @ eL) * eL
    u1 /= np.linalg.norm(u1)
    u2 = e3 - (e3 @ eL) * eL - (e3 @ u1) * u1
    u2 /= np.linalg.norm(u2)
    vu = [np.array([u[1:] @ Ck @ u[1:] for Ck in C]) for u in (u1, u2)]
    out["alpha_TAP"] = 0.5 * ols_slope(x1, 0.5 * (vu[0] + vu[1])) / ell
    out["U_r"] = 1.0 / ols_slope(x1, mean_t)
    out["alpha_LP"] = 0.5 * out["U_r"] ** 2 * ols_slope(x1, var_t) / ell
    out.update(n_traced=int((w > 0).sum()), n_first=counts[0], n_last=counts[-1], n_min=min(counts), n_max=max(counts))
    out["_info"] = {"ell": ell, "x0": x0, "n_planes": len(k), "d_first": float(d[k[0]]), "d_last": float(d[k[-1]]), "Np": Np}
    return out


# ---------------------------------------------------------------------------------------------- table
def table_row(path, seed, hmax_div, basis="disp"):
    """The unique consumer / plane / flux / perplane row of the realization in the reduced table."""
    try:
        with open(path, newline="") as f:
            rows = list(csv.DictReader(f))
    except OSError as e:
        raise DataError("cannot read %s: %s" % (path, e))
    hit = [r for r in rows if r["code"] == "consumer" and r["family"] == "plane" and r["weights"] == "flux" and r["set"] == "perplane"
           and r["basis"] == basis and int(float(r["seed"])) == seed and int(float(r["hmax_div"])) == hmax_div]
    if len(hit) != 1:
        raise DataError("expected exactly one table row (consumer/plane/flux/perplane/%s, seed %d, hmax_div %d) in %s, found %d"
                        % (basis, seed, hmax_div, path, len(hit)))
    return hit[0]


def compare(mine, tab):
    """List of (name, mine, table, rel, ok)."""
    res = []
    for name in ("alpha_TP", "alpha_TAP", "alpha_LP", "U_r", "n_traced", "n_first", "n_last", "n_min", "n_max"):
        a, b = float(mine[name]), float(tab[name])
        if not (math.isfinite(a) and math.isfinite(b)):
            res.append((name, a, b, float("nan"), False))
            continue
        diff = abs(a - b)
        rel = diff / abs(b) if b != 0.0 else diff
        res.append((name, a, b, rel, rel <= REL_TOL if b != 0.0 else diff <= ABS_TOL))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--campaign", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", required=True, type=int, help="field seed of the realization")
    ap.add_argument("--hmax-div", type=int, default=8, help="production pass / table label (default 8)")
    ap.add_argument("--d-lo", type=float, default=25.0)
    ap.add_argument("--d-hi", type=float, default=94.0)
    a = ap.parse_args(argv)
    try:
        campaign = os.path.expanduser(a.campaign)
        seg, r = find_segment(campaign, a.arm, a.seed)
        sname = os.path.basename(seg)
        rdir = os.path.join(seg, "out", "realizations", "r_%04d" % r)
        if not os.path.isfile(os.path.join(rdir, "DONE")):
            raise DataError("%s has no DONE marker" % rdir)
        root, layout = find_pass(rdir, a.hmax_div)
        mine = reduce_realization(rdir, root, a.d_lo, a.d_hi)
        tpath = os.path.join(campaign, "reduced", a.arm, sname, "%s__%s__d%g-%g.csv" % (a.arm, sname, a.d_lo, a.d_hi))
        res = compare(mine, table_row(tpath, a.seed, a.hmax_div))
    except (DataError, KeyError, ValueError) as e:
        print("ERROR: %s" % (e,), file=sys.stderr)
        return 2
    i = mine["_info"]
    print("arm %s  seed %d  segment %s  r %d  layout %s" % (a.arm, a.seed, sname, r, layout))
    print("pass  %s\ntable %s" % (root, tpath))
    print("ell %.17g  x0 %.17g  Np %d  window d in [%g, %g]: %d planes (d %.6g .. %.6g)  weights flux, set perplane, basis disp"
          % (i["ell"], i["x0"], i["Np"], a.d_lo, a.d_hi, i["n_planes"], i["d_first"], i["d_last"]))
    print("%-10s %24s %24s %12s" % ("quantity", "independent", "table", "rel.diff"))
    for name, x, y, rel, ok in res:
        print("%-10s %24.17g %24.17g %12.3e %s" % (name, x, y, rel, "" if ok else "  <-- BEYOND TOLERANCE"))
    good = all(ok for *_, ok in res)
    print("F %s  (max rel.diff %.3e; tolerance %.0e relative, %.0e absolute at 0)"
          % ("PASS" if good else "FAIL", max(x[3] for x in res), REL_TOL, ABS_TOL))
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
