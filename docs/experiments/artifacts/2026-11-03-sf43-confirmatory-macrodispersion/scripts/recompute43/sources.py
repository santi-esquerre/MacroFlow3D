"""Raw-data sources of one realization pass: what each can deliver to the recomputation.

Consumer pass root `P` (a realization directory, or `R/hmax_<d>`), files as written by `lagrangian_ensemble`:
  P/track.json (planes, X0), R/field.json (seed, ell), R/flow_stats.json (lambda),
  P/moments_planes.csv (flux weights) and P/twin/moments_planes.csv (uniform weights): header
      x1,contributing,non_ok,mean_x2,var_x2,mean_x3,var_x3,mean_tau,var_tau,rms_ev,max_ev   (`%.15e`),
  per-particle P/{seeds,weights,planes_x23,planes_tau}.npy and P/planes_I.npy.  The campaign keeps the first four for EVERY realization
  of its primary arms (`sf42red prune --keep-arrays`; `planes_I`, time-pass and Pollock arrays are pruned except for the retained
  subset), which is all the displacement basis, the cross covariance (alpha_TAP) and the common set need; `planes_I` only serves the
  `expI` weights, which are NOT covered when it is absent.
Par2: `crossings/realization_%04d_planes.csv` (format line, then x1,n_injected,n_crossed,mean_t,var_t,mean_x2,mean_x3,var_x2,var_x3,
  cov_x2x3,mean_dx2,mean_dx3,var_dx2,var_dx3,cov_dx2dx3).

Set identity (decides whether 1e-10 is reachable from the moment CSVs): `PlaneMoments` of the tracker are taken over the particles
with `stats_plane.n_reached > m`, i.e. those whose plane pass LANDED on plane m (`LagrangianTracker.cu` `finalize_track_result`), and a
landing is the only thing that writes a finite `planes_x23` / `planes_tau` record (`LagrangianTrackerDevice.cuh`); seed-backflow
particles (weight 0) are never traced (`n_reached = 0`).  The reducer's per-plane set is "finite x23 and tau, weight > 0": the same
set.  Both normalise the weights over the set and use the biased form.  Hence per-plane moments from the CSVs equal the reducer's
(`set=perplane`) to the 16 printed digits.  The `set=common` set (particles that landed on every window plane) is NOT recoverable
from the CSVs (the sets are nested, but a plane's moments cannot be restricted to a subset), nor are the cross covariance
(alpha_TAP) and the displacement basis (needs the seeds): those use the plane arrays, which the campaign keeps for every realization
of the primary arms (the `expI` weights also need `planes_I`, kept for the retained subset only).
The check `n(plane) = contributing` is made on every plane of every realization, so a set difference is reported, never assumed.
"""
import csv
import json
import os

import numpy as np

from .formulas import NAN


def read_json(path):
    with open(path) as f:
        return json.load(f)


def read_csv_rows(path):
    """Rows (dicts of float; blank and `nan`/`-nan` -> NaN) of a CSV whose first non-comment line is the header."""
    with open(path, newline="") as f:
        lines = [ln for ln in f if ln.strip() and not ln.startswith("#")]
    return [{k: (NAN if v.strip() == "" else float(v)) for k, v in row.items()} for row in csv.DictReader(lines)]


def _mom(n, mt, vt, mu, c22, c33, c23):
    return {"n": int(n), "mt": mt, "vt": vt, "mu": mu, "C": (c22, c33, c23)}


class ConsumerSource:
    """Moments of a consumer pass from the moment CSVs (flux, unif; position basis; no cross covariance) and/or the arrays."""

    code = "consumer"
    ARRAYS = ("seeds", "weights", "planes_x23", "planes_tau")  # what the plane estimators read; planes_I (expI weights) is optional

    def __init__(self, pass_root):
        self.root = pass_root
        rdir = pass_root if os.path.isfile(os.path.join(pass_root, "field.json")) else os.path.dirname(os.path.abspath(pass_root))
        self.rdir = rdir
        self.track = read_json(os.path.join(pass_root, "track.json"))
        self.field = read_json(os.path.join(rdir, "field.json"))
        self.ell = float(self.field["ell"])
        lam = float(read_json(os.path.join(rdir, "flow_stats.json"))["lambda"])
        if abs(self.ell - lam) > 1e-9 * abs(lam):
            raise ValueError("field.json ell %.17g != flow_stats.json lambda %.17g" % (self.ell, lam))
        self.x0 = float(self.track["X0"])
        self.planes = [float(x) for x in self.track["planes"]]
        self.csv = {}
        for wmode, sub in (("flux", ""), ("unif", "twin")):
            p = os.path.join(pass_root, sub, "moments_planes.csv")
            if os.path.isfile(p):
                rows = read_csv_rows(p)
                if len(rows) != len(self.planes):
                    raise ValueError("%s: %d rows for %d planes of track.json" % (p, len(rows), len(self.planes)))
                for r, x in zip(rows, self.planes):
                    if abs(r["x1"] - x) > 1e-12 * max(1.0, abs(x)):
                        raise ValueError("%s: plane %.17g != track.json %.17g" % (p, r["x1"], x))
                self.csv[wmode] = rows
        self.arrays = None
        if all(os.path.isfile(os.path.join(pass_root, a + ".npy")) for a in self.ARRAYS):
            a = {n: np.load(os.path.join(pass_root, n + ".npy")) for n in self.ARRAYS}
            if os.path.isfile(os.path.join(pass_root, "planes_I.npy")):
                a["planes_I"] = np.load(os.path.join(pass_root, "planes_I.npy"))
            a["fin"] = np.isfinite(a["planes_x23"]).all(axis=2) & np.isfinite(a["planes_tau"]) & (a["weights"] > 0.0)[:, None]
            self.arrays = a

    # -- moment CSV ---------------------------------------------------------------------------------
    def csv_available(self, wmode, basis):
        return basis == "pos" and wmode in self.csv

    def csv_moments(self, wmode):
        """Per-plane moments (all planes, own sets) from the CSV; empty planes are None."""
        out = []
        for r in self.csv[wmode]:
            n = int(r["contributing"])
            out.append(None if n == 0 else _mom(n, r["mean_tau"], r["var_tau"], (r["mean_x2"], r["mean_x3"]), r["var_x2"], r["var_x3"], None))
        return out

    # -- arrays -------------------------------------------------------------------------------------
    def array_moments(self, wmode, basis, ks, common):
        """Moments at planes ks (indices) with the set of each plane (common: landed on every plane of ks); None for an empty set."""
        a = self.arrays
        fin = a["fin"][:, ks]
        S = fin & fin.all(axis=1, keepdims=True) if common else fin
        out = []
        for j, k in enumerate(ks):
            sel = np.flatnonzero(S[:, j])
            if len(sel) == 0:
                out.append(None)
                continue
            Y = np.asarray(a["planes_x23"][sel, k, :], float)
            if basis == "disp":
                Y = Y - a["seeds"][sel, 1:3]
            tau = np.asarray(a["planes_tau"][sel, k], float)
            w = {"flux": lambda: a["weights"][sel], "unif": lambda: np.ones(len(sel)),
                 "expI": lambda: a["weights"][sel] * np.exp(-np.asarray(a["planes_I"][sel, k], float))}[wmode]()
            sw = float(w.sum())
            if not sw > 0.0:
                out.append(None)
                continue
            mu = (w[:, None] * Y).sum(axis=0) / sw
            D = Y - mu
            mt = float((w * tau).sum() / sw)
            C = (D * w[:, None]).T @ D / sw
            out.append(_mom(len(sel), mt, float((w * (tau - mt) ** 2).sum() / sw), (float(mu[0]), float(mu[1])),
                            float(C[0, 0]), float(C[1, 1]), float(C[0, 1])))
        return out

    def has_weights(self, wmode):
        """Whether the arrays can deliver the weight variant (expI needs planes_I)."""
        return self.arrays is not None and (wmode != "expI" or "planes_I" in self.arrays)

    def set_stats(self, wmode, ks):
        """(n_common, wfrac_min, n_traced) over planes ks: n_common = landed on every plane; wfrac_min = smallest share of the
        reference weight (uniform: of the traced particles; flux and expI: of the flux weight) inside a plane's set."""
        a = self.arrays
        fin = a["fin"][:, ks]
        traced = int((a["weights"] > 0.0).sum())
        if wmode == "unif":
            fr = [fin[:, j].sum() / float(traced) for j in range(len(ks))]
        else:
            fr = [a["weights"][fin[:, j]].sum() / a["weights"].sum() for j in range(len(ks))]
        return int(fin.all(axis=1).sum()), float(min(fr)), traced


class Par2Source:
    """Moments of a Par2 plane-crossing CSV (uniform weights, both bases, own sets, full 2x2 covariance)."""

    code = "par2"
    HEADER = ["x1", "n_injected", "n_crossed", "mean_t", "var_t", "mean_x2", "mean_x3", "var_x2", "var_x3", "cov_x2x3",
              "mean_dx2", "mean_dx3", "var_dx2", "var_dx3", "cov_dx2dx3"]
    FORMAT = "# format_version=1 sf42_plane_crossings weights=uniform time_origin=injection"

    def __init__(self, path):
        self.path = path
        with open(path) as f:
            lines = f.read().splitlines()
        if not lines or lines[0].strip() != self.FORMAT:
            raise ValueError("%s: first line is not the sf42_plane_crossings format line" % path)
        body = [ln for ln in lines[1:] if ln.strip() and not ln.startswith("#")]
        if body[0].split(",") != self.HEADER:
            raise ValueError("%s: header differs from the fixed crossing header" % path)
        self.t = {h: [float(ln.split(",")[i]) for ln in body[1:]] for i, h in enumerate(self.HEADER)}
        self.planes = self.t["x1"]

    def moments(self, basis):
        m2, m3, v2, v3, c23 = (("mean_x2", "mean_x3", "var_x2", "var_x3", "cov_x2x3") if basis == "pos"
                               else ("mean_dx2", "mean_dx3", "var_dx2", "var_dx3", "cov_dx2dx3"))
        t = self.t
        return [None if t["n_crossed"][k] <= 0 else
                _mom(t["n_crossed"][k], t["mean_t"][k], t["var_t"][k], (t[m2][k], t[m3][k]), t[v2][k], t[v3][k], t[c23][k])
                for k in range(len(self.planes))]


def plane_row_values(mom_k):
    """Columns of the per-plane table (`*.planes.csv`) from one plane's moments (NaN for an empty plane)."""
    if mom_k is None:
        return {"n": 0, "mean_tau": NAN, "var_tau": NAN, "mean_x2": NAN, "mean_x3": NAN, "var_x2": NAN, "var_x3": NAN, "cov_x2x3": NAN}
    c = mom_k["C"]
    return {"n": mom_k["n"], "mean_tau": mom_k["mt"], "var_tau": mom_k["vt"], "mean_x2": mom_k["mu"][0], "mean_x3": mom_k["mu"][1],
            "var_x2": c[0], "var_x3": c[1], "cov_x2x3": c[2]}  # None: no cross covariance in this source

