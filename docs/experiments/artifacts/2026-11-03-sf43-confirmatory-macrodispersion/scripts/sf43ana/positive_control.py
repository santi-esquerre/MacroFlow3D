"""Positive control of SF-43 (decision H9): the end-to-end check of the estimand pipeline against the first-order theory of the box.

The ``sigma^2 = 0.04`` arm on 1040x240x240 (width 24 lambda, box 104 lambda): at that amplitude the transverse displacement variance of a
particle is, to first order in ``sigma^2``, ``X22(d)`` of the periodic box (``firstorder_box.py`` of the pre-activation review, ported
here: the SF-18 field is torus-periodic on the flow grid, so the first-order transverse velocity of mode ``k`` is
``u2 = -U (k1 k2 / k^2) Y_hat(k)`` on the discrete lattice ``k = 2 pi (m1 / L1, m2 / W, m3 / W)`` and

    X22(d) = 2 sum_k U^2 (k1 k2 / k^2)^2 S_d(k) (1 - cos(k1 d)) / k1^2       (k1 != 0),   units lambda = U = sigma^2 = 1,

times ``sigma^2 lambda^2`` here; ``S(k) = sigma^2 lambda^3 exp(-k^2 lambda^2 / 4) / (8 pi^1.5)`` for ``C_Y(r) = sigma^2 exp(-(r/lambda)^2)``).
One-particle, Dirichlet faces of medium (b) not modelled (they perturb the last 5-10 lambda; the window ends 5 lambda before the outlet).

MEASURED: from the per-plane tables (``<table>.planes.csv`` of ``sf42red``: per realization, per plane, the weighted moments of the
particles that crossed it) in the DISPLACEMENT basis (``x23(d) - x23(seed)``), for each weights variant (``flux``, ``unif``):
    ``second moment``  ``E_r[(var_x2 + var_x3)/2 + (mean_x2^2 + mean_x3^2)/2]``: the one-particle ensemble second moment (what ``X22`` is);
    ``centred``        ``E_r[(var_x2 + var_x3)/2]``: per-realization centred, the quantity the estimand ``alpha_TP`` is the slope of.
Lengths of the tables are in box units; ``/ ell^2`` converts to lambda^2 (``ell`` from the ``ell`` column of the reduced table).

REPORTED (descriptive; no verdict, outside the family): per plane the ratio measured / predicted with a seed-bootstrap percentile interval
(realizations resampled; ``level``, ``B``), the OLS slope of the measured and of the predicted profile over the window planes (the
estimand is half the slope of the centred profile), their difference and ratio with seed-bootstrap intervals, and a count of the planes
whose interval contains 1.  ``prereg["positive_control"]``::

    {"arm": <arm id of the sigma^2 = 0.04 arm>, "sigma2": 0.04, "width_lambda": 24, "length_lambda": 104, "window": [25, 94],
     "weights": ["flux", "unif"], "level": 0.90, "B": 2000, "seed": 20261103, "kmax": 12.0}
"""
from __future__ import annotations

import csv
import zlib

import numpy as np

from .loader import TableCache

KEYS = {"arm", "sigma2", "width_lambda", "length_lambda", "window", "weights", "level", "B", "seed", "kmax"}


def x22_box(d, L1, W, kmax=12.0):
    """First-order transverse displacement variance (one axis) of the periodic box, units lambda = U = sigma^2 = 1 (module docstring)."""
    m1 = np.arange(1, int(np.ceil(kmax * L1 / (2 * np.pi))) + 1)
    mt = int(np.ceil(kmax * W / (2 * np.pi)))
    m2 = np.arange(-mt, mt + 1)
    k1 = 2 * np.pi * m1 / L1
    k2 = 2 * np.pi * m2 / W
    K1, K2, K3 = np.meshgrid(k1, k2, k2, indexing="ij")
    ksq = K1 ** 2 + K2 ** 2 + K3 ** 2
    S = (1.0 / (8 * np.pi ** 1.5)) * np.exp(-ksq / 4.0) * (2 * np.pi) ** 3 / (L1 * W * W)
    w = 2.0 * ((K1 * K2 / ksq) ** 2 * S).sum(axis=(1, 2))  # factor 2: +-k1
    d = np.atleast_1d(np.asarray(d, dtype=float))
    return np.array([2.0 * np.sum(w * (1 - np.cos(k1 * dd)) / k1 ** 2) for dd in d])


def _ols_slope(x, y):
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())


def planes_path(tables, arm, segment, window):
    p = tables.path(arm, segment, window)
    return (p[:-4] if p.endswith(".csv") else p) + ".planes.csv"


def read_planes(path, code, weights, basis, hmax_div):
    """{r: (d array, in_window array, second-moment array, centred-variance array)} of one planes table, planes in file order."""
    acc = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["code"] != code or row["weights"] != weights or row["basis"] != basis or int(float(row["hmax_div"])) != hmax_div:
                continue
            v2, v3, m2, m3 = (float(row[c]) for c in ("var_x2", "var_x3", "mean_x2", "mean_x3"))
            acc.setdefault(int(float(row["r"])), []).append((int(float(row["plane"])), float(row["d"]), int(float(row["in_window"])),
                                                              0.5 * (v2 + v3) + 0.5 * (m2 * m2 + m3 * m3), 0.5 * (v2 + v3)))
    out = {}
    for r, rows in acc.items():
        rows.sort()
        a = np.array(rows)
        out[r] = (a[:, 1], a[:, 2].astype(bool), a[:, 3], a[:, 4])
    return out


def analyse_arm(pc, arm, tables, cfg_defaults):
    """Report of one weights variant list for the arm (``ArmData``); see the module docstring."""
    sigma2, W, L1 = float(pc["sigma2"]), float(pc["width_lambda"]), float(pc["length_lambda"])
    lo, hi = pc.get("window", [25, 94])
    level, B = float(pc.get("level", 0.90)), int(pc.get("B", 2000))
    prim = cfg_defaults["primary_window"]
    cfg = arm.cfg
    out = {"arm": arm.id, "N": arm.N, "sigma2": sigma2, "width_lambda": W, "length_lambda": L1, "window": [lo, hi], "weights": {}}
    if arm.N < 2:
        out["note"] = "fewer than 2 complete realizations: no positive control"
        return out
    ell = None
    for i, s in enumerate(arm.seeds):  # ell of the tables (box units per lambda)
        tab = tables.table(cfg["source"], arm.segments[i], prim)
        if tab:
            ell = float(next(iter(tab.values()))["ell"])
            break
    if ell is None or not ell > 0:
        out["note"] = "no reduced table with an ell column"
        return out
    out["ell"] = ell
    code = cfg["code"]
    for w in pc.get("weights", ["flux", "unif"]):
        cache, V2, VC, dgrid, win = {}, [], [], None, None
        for i, s in enumerate(arm.seeds):
            seg = arm.segments[i]
            if seg not in cache:
                cache[seg] = read_planes(planes_path(tables, cfg["source"], seg, prim), code, w, "disp", cfg["hmax_div"])
            s0 = tables.ledger.seg_seed0[(cfg["source"], seg)]
            rec = cache[seg].get(int(s) - s0)
            if rec is None:
                continue
            if dgrid is None:
                dgrid, win = rec[0], rec[1]
            elif rec[0].shape != dgrid.shape or not np.allclose(rec[0], dgrid):
                raise ValueError("planes differ between realizations of arm %s" % arm.id)
            V2.append(rec[2] / ell ** 2)
            VC.append(rec[3] / ell ** 2)
        if len(V2) < 2:
            out["weights"][w] = {"note": "fewer than 2 realizations with plane rows"}
            continue
        V2, VC = np.array(V2), np.array(VC)
        pred = sigma2 * x22_box(dgrid, L1, W, float(pc.get("kmax", 12.0)))
        inw = win & (dgrid >= lo * (1 - 1e-9)) & (dgrid <= hi * (1 + 1e-9))
        rng = np.random.default_rng(np.random.SeedSequence([int(pc.get("seed", 20261103)), zlib.crc32(w.encode()), zlib.crc32(arm.id.encode())]))
        n = V2.shape[0]
        idx = rng.integers(0, n, size=(B, n))
        a = 0.5 * (1.0 - level)

        def prof(M):
            return M.mean(axis=0), M[idx].mean(axis=1)  # (planes,), (B, planes)

        res = {"n": int(n), "d": dgrid.tolist(), "predicted": pred.tolist()}
        for name, M in (("second_moment", V2), ("centred", VC)):
            m, ms = prof(M)
            rl, rh = np.quantile(ms / pred, [a, 1.0 - a], axis=0)
            slope = _ols_slope(dgrid[inw], m[inw])
            slope_b = np.array([_ols_slope(dgrid[inw], ms[b][inw]) for b in range(B)])
            ps = _ols_slope(dgrid[inw], pred[inw])
            dlo, dhi = np.quantile(slope_b - ps, [a, 1.0 - a])
            rlo, rhi = (np.quantile(slope_b / ps, [a, 1.0 - a]) if ps != 0 else (float("nan"), float("nan")))
            res[name] = {"measured": m.tolist(), "ratio": (m / pred).tolist(), "ratio_lo": rl.tolist(), "ratio_hi": rh.tolist(),
                         "planes_ratio_interval_contains_1_in_window": int(((rl <= 1.0) & (rh >= 1.0))[inw].sum()), "planes_in_window": int(inw.sum()),
                         "window_slope": {"measured": slope, "predicted": ps, "difference": slope - ps, "difference_interval": [float(dlo), float(dhi)],
                                          "ratio": slope / ps if ps else float("nan"), "ratio_interval": [float(rlo), float(rhi)],
                                          "alpha_T_equivalent_measured": 0.5 * slope, "alpha_T_equivalent_predicted": 0.5 * ps,
                                          "difference_interval_contains_0": bool(dlo <= 0.0 <= dhi)}}
        res["note"] = "descriptive; first order in sigma^2, one-particle, Dirichlet faces not modelled"
        out["weights"][w] = res
    return out


def analyse(ctx):
    pc = ctx.prereg.raw.get("positive_control")
    if not pc:
        return None
    arm = ctx.arms[pc["arm"]]
    tables = TableCache(ctx.reduced, ctx.prereg, ctx.ledger)
    return analyse_arm(pc, arm, tables, ctx.prereg.raw)
