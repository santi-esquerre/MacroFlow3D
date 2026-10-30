"""Gate 5 attribution factorial (decision G7): a ``2^(4-1)`` resolution-IV fraction analysed as a split-plot.

Factors (coded -1 / +1)
    G  generator          whole-plot, UNPAIRED between levels: Par2 random-spectral (-1) against the SF-18 periodic generator (+1)
    I  injection          within seed: full plane (-1) against the transverse window (+1)
    D  Par2 time step     within seed: ``dt`` (-1) against ``dt / 2`` (+1)
    R  Par2 solver tol.   within seed: ``cg_rtol`` 1e-6 (-1) against 1e-10 (+1)

Fraction.  The defining relation is ``I = GIDR`` (here the identity is written ``1`` to avoid the clash with the injection factor
``I``): the 8 cells are those with ``g * i * d * r = +1``.  Aliasing (word times the defining word, letters cancelling in pairs):
main effects with three-factor interactions (``G = IDR``, ``I = GDR``, ``D = GIR``, ``R = GID``) and the two-factor interactions in
pairs (``GI = DR``, ``GD = IR``, ``GR = ID``).  The aliased two-factor pairs cannot be separated; they are reported as such.

Split-plot structure.  Within a generator level ``g`` the 4 cells are those with ``i d r = g``; every whole plot (a seed = one
field of that generator) runs all 4 cells (within-seed, paired).  Generators have different seed sets (the field depends on the
generator): the whole-plot factor is compared BETWEEN independent seed groups.

Analysis (per-seed quantities, then means over seeds per generator; ``n_g`` seeds, sample sd ``s``)
* within-seed (sub-plot) contrasts: for ``f`` in I, D, R and generator ``g``: ``c_s^{(g,f)} = mean(y_s at f = +1) - mean(y_s at f = -1)``
  over the 4 cells of seed ``s`` (a difference of two cell means); ``chat_{g,f}`` its mean over seeds.  A ONE-SAMPLE ``t`` on
  the per-seed contrasts (n_g - 1 d.o.f.) tests each *simple* effect (aliased with the 2fi of the other two factors inside the
  generator);
* effects of the 8-cell design (difference of mean cell values at the high and the low level of the contrast, ``E_w = (1/4) sum_c w_c ybar_c``):
  ``E_f = (chat_{+,f} + chat_{-,f}) / 2`` (I, D, R; sub-plot stratum),  ``E_{Gf} = (chat_{+,f} - chat_{-,f}) / 2`` (GI = DR, GD = IR,
  GR = ID; sub-plot stratum compared between generators),  ``E_G = abar_+ - abar_-`` with ``a_s`` the mean of the seed's 4 cells (G = IDR;
  whole-plot stratum).  Each is a linear combination of two independent per-generator means, tested by Welch-Satterthwaite (variance
  ``sum c^2 s^2 / n``: the variance of the *stratum the effect lives in* -- the per-seed average, which carries the field-to-field variance,
  for G; the per-seed contrast, from which the field effect cancels, for the others: the two-stratum variance);
* effects on the log of cell means: ``E_w^{log} = (1/4) sum_c w_c log(ybar_c)`` with ``ybar_c`` the mean over seeds of cell ``c`` in its
  generator; ``exp(E^{log})`` is a ratio of geometric means of cell means.  CI by a percentile bootstrap over seeds (seeds
  resampled with replacement INSIDE each generator, the same indices for the 4 cells of that generator; ``np.random.default_rng``).
  Requires positive cell means.
"""
from __future__ import annotations

import itertools
from typing import NamedTuple

import numpy as np
from scipy import stats as _st

from .intervals import welch_contrast

FACTORS = ("G", "I", "D", "R")
LEVELS = {"G": ("par2_random_spectral", "sf18_periodic"), "I": ("full_plane", "window"),
          "D": ("dt", "dt_half"), "R": ("cg_rtol_1e-6", "cg_rtol_1e-10")}  # (-1, +1)
DEFINING_WORD = "GIDR"  # defining relation 1 = GIDR (written I = GIDR in decision G7)
SUBPLOT = ("I", "D", "R")


def aliases(word, defining=DEFINING_WORD):
    """Alias of an effect word under the defining relation: letters of ``word`` and ``defining`` cancel in pairs.

    ``aliases('GI') == 'DR'``; ``aliases('G') == 'IDR'``; the empty word is the identity ``'1'``.
    """
    s = set(word) ^ set(defining)
    out = "".join(f for f in FACTORS if f in s)
    return out if out else "1"


def cells_of(g):
    """The 4 sub-plot cells ``(i, d, r)`` run within a seed of generator level ``g`` (those with ``i d r = g``)."""
    return [c for c in itertools.product((-1, 1), repeat=3) if c[0] * c[1] * c[2] == g]


def design_cells():
    """The 8 cells ``(g, i, d, r)`` of the fraction (``g i d r = +1``)."""
    return [(g,) + c for g in (-1, 1) for c in cells_of(g)]


def design(seeds_plus, seeds_minus):
    """Design rows: one per (generator, seed, cell).  ``seeds_plus`` are the seeds of the SF-18 generator (``g = +1``),
    ``seeds_minus`` those of Par2's random-spectral generator (``g = -1``); every seed runs the 4 cells of its generator.

    Total runs: ``4 (n_plus + n_minus)``.  Returns a list of dicts with keys generator, seed, cell (i, d, r), g, i, d, r.
    """
    rows = []
    for g, seeds in ((1, seeds_plus), (-1, seeds_minus)):
        for s in seeds:
            for (i, d, r) in cells_of(g):
                rows.append({"generator": LEVELS["G"][(g + 1) // 2], "seed": s, "cell": (i, d, r), "g": g, "i": i, "d": d, "r": r})
    return rows


def effect_table():
    """The 7 estimable effects: ``(label, alias, stratum, contrast coefficients over the 8 cells)``; labels list the alias class
    by its lowest-order member (``G``, ``I``, ``D``, ``R``, ``GI``, ``GD``, ``GR``)."""
    cells = design_cells()
    out = []
    for word in ("G", "I", "D", "R", "GI", "GD", "GR"):
        w = np.array([np.prod([dict(zip(FACTORS, c))[f] for f in word]) for c in cells], dtype=float)
        out.append((word, aliases(word), "whole-plot" if word == "G" else "sub-plot", w))
    return out


_CELLS8 = design_cells()
_EFFECTS = effect_table()


class SimpleEffect(NamedTuple):
    mean: float
    sd: float
    n: int
    t: float
    dof: int
    p: float


def _group_arrays(data, g):
    """-> (cells list, array (n_g, 4)) in the canonical order of ``cells_of(g)``."""
    cells = cells_of(g)
    cols = []
    for c in cells:
        if c not in data[g]:
            raise ValueError("missing cell %r for generator %+d" % (c, g))
        cols.append(np.asarray(data[g][c], dtype=float))
    n = {len(v) for v in cols}
    if len(n) != 1:
        raise ValueError("the 4 cells of generator %+d must have the same seeds (aligned arrays)" % g)
    return cells, np.column_stack(cols)


def _contrast_per_seed(cells, Y, f):
    """Per-seed difference of the two cell means at ``f = +1`` against ``f = -1`` (``f`` in I, D, R)."""
    k = SUBPLOT.index(f)
    x = np.array([c[k] for c in cells], dtype=float)
    return Y[:, x > 0].mean(axis=1) - Y[:, x < 0].mean(axis=1)


def analyze(data, level=0.90, log_bootstrap=2000, seed=20261030):
    """Analyse a split-plot realization of the fraction.

    ``data``: ``{+1: {(i, d, r): y (n_plus,)}, -1: {(i, d, r): y (n_minus,)}}`` -- for each generator the 4 cells of
    ``cells_of(g)``, each with one value per seed (same seed order across the 4 cells of a generator; the two generators have
    different seeds, so ``n_plus != n_minus`` is fine).  Returns a dict with keys ``cells`` (cell means), ``simple`` (one-sample t
    of each within-seed contrast per generator), ``effects`` (the 7 effects: estimate, se, Welch nu, ``level`` CI, t, p, alias,
    stratum, and the log-scale estimate with its bootstrap CI), ``aliases`` and ``design``.  ``log_bootstrap=0`` skips the log part.
    """
    arrs = {g: _group_arrays(data, g) for g in (1, -1)}
    ns = {g: arrs[g][1].shape[0] for g in arrs}
    if min(ns.values()) < 2:
        raise ValueError("at least 2 seeds per generator are needed")
    # within-generator per-seed contrasts and averages
    per_seed = {g: {f: _contrast_per_seed(arrs[g][0], arrs[g][1], f) for f in SUBPLOT} for g in arrs}
    avg = {g: arrs[g][1].mean(axis=1) for g in arrs}
    simple = {}
    for g in arrs:
        for f in SUBPLOT:
            v = per_seed[g][f]
            n = v.size
            m, s = float(v.mean()), float(v.std(ddof=1))
            with np.errstate(divide="ignore", invalid="ignore"):
                t = m / (s / np.sqrt(n)) if s > 0 else (np.inf if m != 0 else 0.0)
            p = float(2.0 * _st.t.sf(abs(t), n - 1))
            simple[(g, f)] = SimpleEffect(m, s, n, float(t), n - 1, p)
    effects = {}

    def combine(label, v_plus, v_minus, c_plus, c_minus, stratum):
        w = welch_contrast(v_plus.mean(), v_plus.std(ddof=1), v_plus.size, v_minus.mean(), v_minus.std(ddof=1), v_minus.size,
                           c_plus, c_minus, level, "two")
        with np.errstate(divide="ignore", invalid="ignore"):
            t = float(w.diff / w.se) if w.se > 0 else (np.inf if w.diff != 0 else 0.0)
        p = float(2.0 * _st.t.sf(abs(t), w.nu)) if np.isfinite(w.nu) else float("nan")
        effects[label] = {"estimate": float(w.diff), "se": float(w.se), "nu": float(w.nu), "lo": float(w.lo), "hi": float(w.hi),
                          "t": t, "p": p, "alias": aliases(label), "stratum": stratum}

    combine("G", avg[1], avg[-1], 1.0, -1.0, "whole-plot")
    for f in SUBPLOT:
        combine(f, per_seed[1][f], per_seed[-1][f], 0.5, 0.5, "sub-plot")
        combine("G" + f, per_seed[1][f], per_seed[-1][f], 0.5, -0.5, "sub-plot")
    out = {"effects": effects, "simple": simple,
           "cells": {(g,) + c: float(arrs[g][1][:, k].mean()) for g in arrs for k, c in enumerate(arrs[g][0])},
           "aliases": {w: aliases(w) for w in effects}, "design": design_cells(), "n": ns}
    if log_bootstrap:
        out["log_effects"] = _log_effects(arrs, level, log_bootstrap, seed)
    return out


def _log_contrasts(arrs, idx=None):
    """Log-scale effects (dict word -> value) from per-generator arrays (optionally resampled by seed indices ``idx[g]``)."""
    ybar = {}
    for g, (cells, Y) in arrs.items():
        Yg = Y if idx is None else Y[idx[g]]
        for k, c in enumerate(cells):
            ybar[(g,) + c] = Yg[:, k].mean()
    if any(ybar[c] <= 0 for c in _CELLS8):
        raise ValueError("cell means must be positive for effects on the log scale")
    ly = np.array([np.log(ybar[c]) for c in _CELLS8])
    return {label: float(w @ ly) / 4.0 for label, _, _, w in _EFFECTS}


def _log_effects(arrs, level, B, seed):
    rng = np.random.default_rng(seed)
    est = _log_contrasts(arrs)
    boots = {k: np.empty(B) for k in est}
    ns = {g: arrs[g][1].shape[0] for g in arrs}
    for b in range(B):
        idx = {g: rng.integers(0, ns[g], size=ns[g]) for g in arrs}
        e = _log_contrasts(arrs, idx)
        for k in est:
            boots[k][b] = e[k]
    lo_q, hi_q = 0.5 * (1.0 - level), 0.5 * (1.0 + level)
    return {k: {"estimate": est[k], "lo": float(np.quantile(boots[k], lo_q)), "hi": float(np.quantile(boots[k], hi_q)),
                "ratio": float(np.exp(est[k])), "alias": aliases(k)} for k in est}
