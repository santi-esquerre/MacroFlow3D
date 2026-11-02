"""GPU-second cost model of one realization, calibrated on the SF-39 campaign walls (decision G5.2 budgeting, G10 budget).

Data (committed): ``A/raw/campaign*/**/walls.tsv`` (``id<TAB>wall_s<TAB>end_utc``), ``A/configs/budget.json`` (Par2 realization counts),
arm geometry transcribed from ``A/preregistration.md`` section 1 (``ARMS`` below; a test cross-checks it against that table).
``A`` = ``docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study``.

Cost per realization = wall seconds of the job / realizations it ran (one GPU; the host GPU is shared with colleagues, so the walls
carry load noise: SF-39 itself shows a 2x outlier, ``s4_tw`` extension 42 s against 22 s in the pilot).

Functional form (non-negative least squares, relative-error weights ``1 / t``)::

    lagrangian :  t = c0 + c_cells * cells + c_sig * cells * sigma^2 + c_trace * (M^2 * nx * pass_work)
    par2       :  t = c_cells * cells                 (also codes 'par2_fs', 'par2_<tag>': each code fitted separately)

``cells`` = ``nx ny nz``: the flow stage (head solve, velocity, splines) is linear in the number of cells; the ``cells * sigma^2``
term lets the PCG iteration count grow with the contrast (it moves the ``sigma^2 = 4`` large-grid arms, which dominate a study's
cost, from 23 % to 15 % under-prediction; the other amplitudes barely change).  ``trace`` = particles
``M^2`` x cells crossed along ``x`` (``nx``, the path length in cells: the step is bounded by ``h_max = h / div``) x ``pass_work``
= ``sum over passes of div / 8`` (relative to SF-39's production bound ``h/8``; a three-pass A directory ``8,16,32`` has pass_work 7):
tracing cost per particle is proportional to the number of steps.  Not resolvable in SF-39 data (documented, not fitted): the
dependence on the number of recorded planes (95-199, a minor cost), on ``M^2`` (always 1e4, so the proportionality to ``M^2`` is a
prior) and a ``sigma^2`` dependence of the tracing term.

Validity range (``CostModel.ranges``): the span of the calibration points of each code; outside it ``predict(strict=True)`` raises and
``project`` lists the arm.  New calibration points (the grid pilot of a later node, Par2 at ``cg_rtol 1e-10``) are added with
``with_points`` and the model is refitted; a code with its own tag (e.g. ``par2_rtol1e-10``) is fitted from its own points only.
This is a planning tool (about +-25 % per arm), not a billing record.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np
from scipy import optimize as _opt

A_RELATIVE = "docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study"
PILOT_NR = 20  # Par2 pilot realizations (run_sf39.sh PILOT_NR)

# arm id -> (nx, ny, nz, lambda/h, eps, hmax_div tuple); preregistration section 1 and the 2026-10-29 addendum (s025)
ARMS = {}


def _add(aid, nx, ny, lh, eps, hmax=(8,)):
    ARMS[aid] = (nx, ny, ny, lh, eps, tuple(hmax))


for _c, _eps in (("s1", 1.0), ("s025", 0.5)):
    _add(_c + "_A", 2048, 256, 10, _eps, (8, 16, 32))
    _add(_c + "_B", 2048, 256, 10, _eps)
    _add(_c + "_n05", 1024, 64, 5, _eps)
    _add(_c + "_n10", 2048, 128, 10, _eps)
    _add(_c + "_n20", 4096, 256, 20, _eps)
    _add(_c + "_tw", 1024, 128, 5, _eps)
for _c, _eps, _l in (("s225", 1.5, 8), ("s4", 2.0, 16)):  # _l: lambda/h of the coarse narrow rung (production is 2 _l)
    _add(_c + "_A", 3328, 384, 2 * _l, _eps, (8, 16, 32))
    _add(_c + "_B", 3328, 384, 2 * _l, _eps)
    _add(_c + "_n%02d" % _l, 1664, 64, _l, _eps)
    _add(_c + "_n%02d" % (2 * _l), 3328, 128, 2 * _l, _eps)
    _add(_c + "_n%02d" % (4 * _l), 6656, 256, 4 * _l, _eps)
    _add(_c + "_tw", 1664, 192, _l, _eps)
_add("s4_w24", 1664, 384, 16, 2.0)
_add("s4_w12", 1664, 192, 16, 2.0)
_add("s4_w12c", 1664, 192, 16, 2.0)

PLANE_COUNT = {"s025": 197, "s1": 197, "s225": 199, "s4": 95}  # recorded, not fitted
PAR2_GRID = {"s025": (2048, 256, 256, 0.25), "s1": (2048, 256, 256, 1.0), "s225": (2496, 288, 288, 2.25),
             "s4": (2496, 288, 288, 4.0)}  # A/configs/par2_*.yaml; n_particles 1e4 in all
EXCLUDED = {"s4_w12": "halted arm (runner exit, owner decision 2026-10-28): its wall covers an unrecorded number of realizations, "
                      "including non-converged solves; the continuation s4_w12c is used instead"}


@dataclass(frozen=True)
class Arm:
    """What is costed: one realization of a code on a geometry."""
    code: str            # 'lagrangian' | 'par2' | 'par2_fs' | 'par2_<tag>' | ...
    cells: float
    sigma2: float
    particles: float     # M^2 (Lagrangian) or n_particles (Par2)
    nx: float            # cells along x (path length of the tracing)
    pass_work: float     # sum over passes of hmax_div / 8
    planes: int = 0      # recorded plane count (documented, not a fitted feature)
    label: str = ""

    @property
    def family(self):
        return "par2" if self.code.startswith("par2") else "lagrangian"


def lagrangian_arm(nx, ny, nz, sigma2, particles=1e4, hmax_div=(8,), planes=0, label=""):
    """A consumer (``lagrangian_ensemble``) arm; ``hmax_div`` lists the passes (SF-39 production: ``(8,)``, A directory ``(8, 16, 32)``)."""
    return Arm("lagrangian", float(nx) * ny * nz, sigma2, particles, float(nx), sum(d / 8.0 for d in hmax_div), planes, label)


def par2_arm(nx, ny, nz, sigma2, particles=1e4, code="par2", label=""):
    """A Par2 (``macroflow3d_pipeline``) realization; ``code`` may carry a tag, e.g. ``'par2_rtol1e-10'`` (fitted separately)."""
    return Arm(code, float(nx) * ny * nz, sigma2, particles, float(nx), 0.0, 0, label)


@dataclass(frozen=True)
class Point:
    """A calibration point: ``n_real`` realizations of ``arm`` took ``wall_s`` seconds on one GPU."""
    arm: Arm
    n_real: int
    wall_s: float
    source: str = ""

    @property
    def gpu_s_per_real(self):
        return self.wall_s / self.n_real


def _features(arm):
    if arm.family == "par2":
        return {"cells": arm.cells}
    return {"const": 1.0, "cells": arm.cells, "cells_sigma2": arm.cells * arm.sigma2,
            "trace": arm.particles * arm.nx * arm.pass_work}


def _arm_of(aid):
    nx, ny, nz, lh, eps, hmax = ARMS[aid]
    c = aid.split("_")[0]
    return lagrangian_arm(nx, ny, nz, eps ** 2, 1e4, hmax, PLANE_COUNT[c], aid)


# ----------------------------------------------------------------------------------------------------------------------
# SF-39 calibration data
# ----------------------------------------------------------------------------------------------------------------------
def _read_walls(a_root):
    ents, seen = [], set()
    for dirpath, _, files in sorted(os.walk(os.path.join(a_root, "raw"))):
        if "walls.tsv" not in files:
            continue
        part = os.path.basename(dirpath)
        if part in ("checks", "checkpoint", "analysis"):
            continue
        with open(os.path.join(dirpath, "walls.tsv")) as fh:
            for line in fh:
                bits = line.rstrip("\n").split("\t")
                if len(bits) != 3:
                    continue
                key = (bits[2], bits[0])
                if key in seen:
                    continue
                seen.add(key)
                ents.append((bits[2], bits[0], float(bits[1]), os.path.relpath(dirpath, a_root)))
    return sorted(ents)


def sf39_points(a_root):
    """SF-39 calibration points from the committed walls: ``(points, excluded)``.

    Lagrangian jobs ``arm:s<seed0>:n<count>``: ``count`` is the TARGET number of realizations of the arm, already-done ones are
    skipped by the runner, so the realizations run by an entry are ``count`` minus the largest count of the arm's earlier entries
    (entries of 4 s or less are the no-op re-invocations of ``extend``).  Par2 jobs ``par2_<c>`` / ``fs_par2_<c>`` (pilot, 20
    realizations) and ``..._ext`` (``par2_NR - 20``, from ``budget.json``).  ``excluded`` lists ``(id, reason)``.
    """
    with open(os.path.join(a_root, "configs", "budget.json")) as fh:
        budget = json.load(fh)
    points, excluded, done = [], [], {}
    for ts, jid, wall, src in _read_walls(a_root):
        if ":" in jid:
            arm, _, n = jid.split(":")
            n = int(n[1:])
            new = n - done.get(arm, 0)
            done[arm] = max(done.get(arm, 0), n)
            if arm in EXCLUDED:
                excluded.append((jid, EXCLUDED[arm]))
                continue
            if new <= 0:
                continue
            points.append(Point(_arm_of(arm), new, wall, "%s %s" % (src, jid)))
            continue
        base = jid[3:] if jid.startswith("fs_") else jid
        if not base.startswith("par2_"):
            continue
        c = base[5:].replace("_ext", "")
        nx, ny, nz, s2 = PAR2_GRID[c]
        nr = budget[c]["par2_NR"]
        n = nr - PILOT_NR if base.endswith("_ext") else PILOT_NR
        code = "par2_fs" if jid.startswith("fs_") else "par2"
        points.append(Point(par2_arm(nx, ny, nz, s2, 1e4, code, jid), n, wall, "%s %s" % (src, jid)))
    return points, excluded


def sf39_holdout_points(a_root):
    """The single-realization ``chk_*`` check runs (seed 4901) of the SF-39 ``checks`` part: NOT used in the fit, a hold-out.

    Each wall includes process start-up for one realization, so they are expected a little above the per-realization cost.
    """
    pts = []
    path = os.path.join(a_root, "raw", "campaign", "checks", "walls.tsv")
    with open(path) as fh:
        for line in fh:
            jid, wall, _ = line.rstrip("\n").split("\t")
            if not jid.startswith("chk_") or ":" not in jid:
                continue
            aid = jid.split(":")[0][4:]
            if aid in ARMS and aid != "s4_w12c":
                pts.append(Point(_arm_of(aid), 1, float(wall), "checks " + jid))
    return pts


# ----------------------------------------------------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------------------------------------------------
class CostModel:
    """Per-code NNLS fit of GPU-seconds per realization; see the module docstring for the form and its validity."""

    def __init__(self, points):
        self.points = list(points)
        self.coef = {}
        self.ranges = {}
        by_code = {}
        for p in self.points:
            by_code.setdefault(p.arm.code, []).append(p)
        for code, pts in by_code.items():
            names = list(_features(pts[0].arm))
            if len(pts) < len(names):
                raise ValueError("code %r: %d calibration points for %d coefficients" % (code, len(pts), len(names)))
            X = np.array([[_features(p.arm)[k] for k in names] for p in pts])
            y = np.array([p.gpu_s_per_real for p in pts])
            w = 1.0 / y
            c, _ = _opt.nnls(X * w[:, None], y * w)
            self.coef[code] = dict(zip(names, (float(v) for v in c)))
            self.ranges[code] = {"cells": (min(p.arm.cells for p in pts), max(p.arm.cells for p in pts)),
                                 "sigma2": (min(p.arm.sigma2 for p in pts), max(p.arm.sigma2 for p in pts)),
                                 "particles": (min(p.arm.particles for p in pts), max(p.arm.particles for p in pts)),
                                 "n_points": len(pts)}

    def predict(self, arm, strict=False):
        """Predicted GPU-seconds of one realization of ``arm`` (``strict=True``: raise outside the calibrated range)."""
        if arm.code not in self.coef:
            raise KeyError("no calibration for code %r: add points with with_points()" % arm.code)
        if strict and not self.in_range(arm):
            raise ValueError("arm %r is outside the calibrated range %r" % (arm.label, self.ranges[arm.code]))
        f = _features(arm)
        return float(sum(self.coef[arm.code][k] * f[k] for k in self.coef[arm.code]))

    def in_range(self, arm):
        r = self.ranges[arm.code]
        return all(r[k][0] <= getattr(arm, k) <= r[k][1] for k in ("cells", "sigma2", "particles"))

    def residuals(self):
        """One row per calibration point: observed and predicted GPU-s per realization and the relative residual."""
        rows = []
        for p in self.points:
            pred = self.predict(p.arm)
            rows.append({"label": p.arm.label, "code": p.arm.code, "observed": p.gpu_s_per_real, "predicted": pred,
                         "rel": (pred - p.gpu_s_per_real) / p.gpu_s_per_real, "n_real": p.n_real, "source": p.source})
        return rows

    def summary(self):
        """Per code: coefficients, number of points, relative rms and maximum absolute residual, validity ranges."""
        out = {}
        for code in self.coef:
            rel = np.array([r["rel"] for r in self.residuals() if r["code"] == code])
            out[code] = {"coef": self.coef[code], "n_points": int(rel.size), "rms_rel": float(np.sqrt(np.mean(rel ** 2))),
                         "max_abs_rel": float(np.max(np.abs(rel))), "ranges": self.ranges[code]}
        return out

    def with_points(self, new_points):
        """A refitted model with additional calibration points (e.g. the grid pilot of a later node)."""
        return CostModel(self.points + list(new_points))

    def project(self, plan):
        """GPU-hours of a study plan: ``plan`` is a list of ``(arm, n_realizations)`` (or dicts with those keys).

        Returns ``{'gpu_h', 'rows', 'out_of_range'}``; ``rows`` has the per-arm GPU-s per realization and GPU-h.
        """
        rows, bad, tot = [], [], 0.0
        for item in plan:
            arm, n = (item["arm"], item["n_real"]) if isinstance(item, dict) else item
            t = self.predict(arm)
            rows.append({"label": arm.label, "code": arm.code, "n_real": n, "gpu_s_per_real": t, "gpu_h": n * t / 3600.0})
            tot += n * t
            if not self.in_range(arm):
                bad.append(arm.label or arm.code)
        return {"gpu_h": tot / 3600.0, "rows": rows, "out_of_range": bad}


def fit_sf39(repo_root):
    """Convenience: fit on the SF-39 data under ``repo_root`` (returns ``(model, excluded)``)."""
    pts, excl = sf39_points(os.path.join(repo_root, A_RELATIVE))
    return CostModel(pts), excl
