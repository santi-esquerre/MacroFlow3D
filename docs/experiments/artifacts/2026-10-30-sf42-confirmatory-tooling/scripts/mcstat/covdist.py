"""Data distributions of the coverage simulator (SF-42, decision G5.3): the normal and the stress family.

Every distribution is STANDARDIZED (mean 0, variance 1) and exposes its exact skewness and excess kurtosis; the simulator builds
its arms as ``mean + sd * dist.draw(...)``.

Stress family (G5.3 "also under a stress family at the upper bootstrap bounds of skewness and kurtosis")
--------------------------------------------------------------------------------------------------------
* Moments.  For every per-realization series of the SF-39 reduced tables (``analysis/reduced/s<amp>_<set>.csv``, rows of the flux
  method, columns ``tp.alpha_TP`` (``alpha_TP``), ``win.alpha_TAM`` (``alpha_TAM``) and ``win.alpha_e1`` (``alpha_LAM``), the columns
  the SF-39 analysis uses) the sample skewness ``G1`` and excess kurtosis ``G2`` (bias-corrected, ``scipy.stats.skew/kurtosis`` with
  ``bias=False``) are bootstrapped over seeds (``B`` resamples); the stress skewness is the ``conf`` quantile (default 95 %) of
  ``|G1*|`` (both signs are simulated by the sides of the procedures, so the magnitude is what matters) and the stress excess
  kurtosis the ``conf`` quantile of ``G2*``.  Per amplitude the pair is the maximum over series and estimators ("amplitude pairs");
  the ``envelope`` is the maximum over amplitudes.  Series with identical values (the SF-39 ``h_max`` sets repeat the same
  per-realization numbers) count once; series with fewer than 10 seeds are skipped.  The pairs are small-sample (n = 20-71) bootstrap
  bounds, so they are deliberately severe; they are written to ``stress_moments.json`` (provenance: file sha256, B, seed, quantile)
  by ``python3 -m mcstat.coverage stress`` and read from there at run time.
* Family.  A two-component Gaussian mixture with weights ``(p, 1 - p)``, a common within-component variance ``v`` and between-component
  spread ``1 - v``, standardized to mean 0 and variance 1.  Its four moments are closed-form, and for ``0 < p < 1/2`` it reaches
  every (skewness, excess kurtosis) pair above the impossible boundary ``kurt = skew^2 - 2`` (checked numerically, tested), which the
  Johnson SU system does not (SU needs the kurtosis above the lognormal line, ``7.3`` at skewness 1.94, while the bound found is
  ``4.7``).  ``mixture_params`` solves the two moment equations by nested Brent root finding (inner: ``v`` from the skewness at
  fixed ``p``; outer: ``p`` from the kurtosis) and the solution is verified; the sample moments of large draws are tested.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import zlib

import numpy as np
from scipy import optimize
from scipy import stats as _st

HERE = os.path.dirname(os.path.abspath(__file__))
STRESS_JSON = os.path.join(HERE, "stress_moments.json")

# the SF-39 columns of the three estimators (flux method), as in sf39.analyze.KEY
SF39_COLUMNS = {"alpha_TP": "tp.alpha_TP", "alpha_TAM": "win.alpha_TAM", "alpha_LAM": "win.alpha_e1"}
AMPLITUDES = ("s025", "s1", "s225", "s4")


# ----------------------------------------------------------------------------------------------------------------------
# Distributions
# ----------------------------------------------------------------------------------------------------------------------
class Dist:
    """Standardized distribution: ``draw(rng, shape)`` -> float array of mean 0, variance 1."""

    name = "dist"
    skew = 0.0
    ekurt = 0.0

    def draw(self, rng, shape):
        raise NotImplementedError

    def describe(self):
        return {"name": self.name, "skew": self.skew, "ekurt": self.ekurt}


class Normal(Dist):
    name = "normal"

    def draw(self, rng, shape):
        return rng.standard_normal(shape)


class Mirror(Dist):
    """The reflection ``-x`` of a distribution (negative skewness): the lower limit of a mean is the anti-conservative one for it."""

    def __init__(self, base):
        self.base = base
        self.name = base.name + "-"
        self.skew, self.ekurt = -base.skew, base.ekurt

    def draw(self, rng, shape):
        z = self.base.draw(rng, shape)
        np.negative(z, out=z)
        return z

    def describe(self):
        return dict(self.base.describe(), name=self.name, skew=self.skew, mirrored=True)


def mixture_moments(p, v):
    """(skewness, excess kurtosis) of the standardized two-component mixture ``(p, v)`` (see the module docstring).

    Components: mean ``x1 = sqrt((1 - v) q / p)`` with weight ``p`` and ``x2 = -sqrt((1 - v) p / q)`` with weight ``q = 1 - p``, both
    with variance ``v`` (so the mixture has mean 0 and variance 1).  Raw moments of ``N(x, v)``: ``x^3 + 3 x v`` and
    ``x^4 + 6 x^2 v + 3 v^2``.
    """
    p, v = np.asarray(p, dtype=float), np.asarray(v, dtype=float)
    q = 1.0 - p
    b = 1.0 - v
    x1, x2 = np.sqrt(b * q / p), -np.sqrt(b * p / q)
    m3 = p * (x1 ** 3 + 3.0 * x1 * v) + q * (x2 ** 3 + 3.0 * x2 * v)
    m4 = p * (x1 ** 4 + 6.0 * x1 ** 2 * v + 3.0 * v * v) + q * (x2 ** 4 + 6.0 * x2 ** 2 * v + 3.0 * v * v)
    return m3, m4 - 3.0  # the variance is 1 by construction


def mixture_params(skew, ekurt):
    """``(p, v)`` of the standardized two-component mixture with exactly this skewness (> 0) and excess kurtosis.

    Feasible iff ``ekurt > skew^2 - 2`` (the boundary of every distribution); ``ValueError`` otherwise.  Inner problem at fixed ``p``:
    the skewness decreases from the two-point value ``(1 - 2p) / sqrt(p q)`` (``v = 0``) to 0 (``v -> 1``), so ``v`` is a Brent root;
    outer problem: the kurtosis along that curve runs from the boundary value at ``p_hi = (1 - s / sqrt(4 + s^2)) / 2`` to infinity
    as ``p -> 0``, a Brent root again (bracketed on a log grid; the solution is verified to 1e-10 at the end).
    """
    skew, ekurt = float(skew), float(ekurt)
    if not skew > 0.0:
        raise ValueError("the stress family is positively skewed (mirror the data for a negative skewness)")
    if not ekurt > skew * skew - 2.0 + 1e-9:
        raise ValueError("no distribution has skewness %g and excess kurtosis %g (kurtosis must exceed skew^2 - 2)" % (skew, ekurt))
    p_hi = 0.5 * (1.0 - skew / math.sqrt(4.0 + skew * skew))

    def v_of(p):
        f = lambda v: float(mixture_moments(p, v)[0]) - skew
        if f(0.0) <= 0.0:
            return 0.0
        return optimize.brentq(f, 0.0, 1.0 - 1e-12, xtol=1e-15, rtol=1e-14)

    def g(p):
        return float(mixture_moments(p, v_of(p))[1]) - ekurt

    grid = p_hi * np.logspace(-8.0, 0.0, 161)
    gv = [g(p) for p in grid]
    for i in range(1, len(grid)):  # ascending p: the kurtosis falls from +large to the boundary value, g from > 0 to <= 0
        if gv[i - 1] > 0.0 >= gv[i]:
            p = optimize.brentq(g, grid[i - 1], grid[i], xtol=1e-16, rtol=1e-14)
            break
    else:
        raise ValueError("could not bracket the mixture weight for skewness %g, excess kurtosis %g" % (skew, ekurt))
    v = v_of(p)
    s_chk, k_chk = mixture_moments(p, v)
    if abs(float(s_chk) - skew) > 1e-9 or abs(float(k_chk) - ekurt) > 1e-9:
        raise ValueError("mixture moment matching failed: got (%g, %g) for (%g, %g)" % (s_chk, k_chk, skew, ekurt))
    return float(p), float(v)


class Mixture(Dist):
    """Standardized two-component Gaussian mixture with the exact skewness and excess kurtosis of the arguments."""

    def __init__(self, skew, ekurt, name="stress"):
        self.name = name
        self.skew, self.ekurt = float(skew), float(ekurt)
        self.p, self.v = mixture_params(skew, ekurt)
        q = 1.0 - self.p
        self.x1 = math.sqrt((1.0 - self.v) * q / self.p)
        self.x2 = -math.sqrt((1.0 - self.v) * self.p / q)
        self.sw = math.sqrt(self.v)

    def draw(self, rng, shape):
        z = rng.standard_normal(shape)
        z *= self.sw
        z += np.where(rng.random(shape) < self.p, self.x1, self.x2)
        return z

    def describe(self):
        d = Dist.describe(self)
        d.update({"p": self.p, "within_variance": self.v, "family": "two-component Gaussian mixture"})
        return d


# ----------------------------------------------------------------------------------------------------------------------
# Stress moments from the committed SF-39 per-realization tables
# ----------------------------------------------------------------------------------------------------------------------
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _flux_series(path):
    """{estimator: float array of the flux-method values} of one reduced table (empty cells and NaN dropped)."""
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("method") == "flux"]
    out = {}
    for est, col in SF39_COLUMNS.items():
        vals = []
        for r in rows:
            s = r.get(col, "")
            if s not in ("", "nan", "NaN"):
                vals.append(float(s))
        v = np.array(vals, dtype=float)
        out[est] = v[np.isfinite(v)]
    return out


def stress_from_sf39(sf39_root, conf=0.95, B=10000, seed=20261030):
    """Skewness / excess-kurtosis upper bootstrap bounds of the SF-39 per-realization series (see the module docstring).

    ``sf39_root``: the SF-39 artifact root (the directory that contains ``analysis/reduced``).  Returns the dict written to
    ``stress_moments.json``.
    """
    red = os.path.join(sf39_root, "analysis", "reduced")
    pat = re.compile(r"^(s025|s1|s225|s4)_(.*)\.csv$")
    series, seen, files = [], set(), {}
    for fn in sorted(os.listdir(red)):
        m = pat.match(fn)
        if not m or "twinpairs" in fn:
            continue
        path = os.path.join(red, fn)
        files[fn] = _sha256(path)
        for est, v in _flux_series(path).items():
            if v.size < 10:
                continue
            key = (m.group(1), est, tuple(np.round(v, 14)))
            if key in seen:
                continue
            seen.add(key)
            rng = np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(("%s|%s" % (fn, est)).encode())]))
            bs = v[rng.integers(0, v.size, size=(B, v.size))]
            g1 = _st.skew(bs, axis=1, bias=False)
            g2 = _st.kurtosis(bs, axis=1, bias=False)
            series.append({"amplitude": m.group(1), "set": m.group(2), "file": fn, "estimator": est, "column": SF39_COLUMNS[est],
                           "n": int(v.size), "skew": float(_st.skew(v, bias=False)), "ekurt": float(_st.kurtosis(v, bias=False)),
                           "skew_ucb": float(np.quantile(np.abs(g1), conf)), "ekurt_ucb": float(np.quantile(g2, conf))})
    amps = {}
    for a in AMPLITUDES:
        rows = [r for r in series if r["amplitude"] == a]
        if not rows:
            raise ValueError("no SF-39 series found for amplitude %s under %s" % (a, red))
        ks = max(rows, key=lambda r: r["skew_ucb"])
        kk = max(rows, key=lambda r: r["ekurt_ucb"])
        amps[a] = {"skew": ks["skew_ucb"], "ekurt": kk["ekurt_ucb"], "skew_from": [ks["file"], ks["estimator"]],
                   "ekurt_from": [kk["file"], kk["estimator"]]}
    env = {"skew": max(v["skew"] for v in amps.values()), "ekurt": max(v["ekurt"] for v in amps.values())}
    for name, d in list(amps.items()) + [("envelope", env)]:
        Mixture(d["skew"], d["ekurt"])  # raises when the pair is not realizable
    return {"description": "upper bootstrap bounds of the skewness and excess kurtosis of the SF-39 per-realization values (flux method)",
            "confidence": conf, "B": B, "seed": seed,
            "skew_rule": "conf-quantile of |G1*| (bias-corrected sample skewness of the seed bootstrap)",
            "ekurt_rule": "conf-quantile of G2* (bias-corrected sample excess kurtosis of the seed bootstrap)",
            "columns": SF39_COLUMNS, "amplitudes": amps, "envelope": env, "series": series, "files_sha256": files,
            "computed_with": {"numpy": np.__version__, "scipy": __import__("scipy").__version__}}


_STRESS_CACHE = {}


def load_stress(path=STRESS_JSON):
    if path not in _STRESS_CACHE:
        with open(path) as f:
            _STRESS_CACHE[path] = json.load(f)
    return _STRESS_CACHE[path]


def get_dist(name, path=STRESS_JSON):
    """``normal``; ``stress`` (the envelope); ``stress_<amp>`` (amplitude pair, ``<amp>`` in s025, s1, s225, s4);
    ``mix:<skew>:<ekurt>`` (explicit pair, used by the tests); a trailing ``-`` is the reflection (negative skewness)."""
    if name.endswith("-"):
        return Mirror(get_dist(name[:-1], path))
    if name == "normal":
        return Normal()
    if name.startswith("mix:"):
        _, s, k = name.split(":")
        return Mixture(float(s), float(k), name)
    st = load_stress(path)
    if name == "stress":
        return Mixture(st["envelope"]["skew"], st["envelope"]["ekurt"], "stress")
    if name.startswith("stress_") and name[7:] in st["amplitudes"]:
        d = st["amplitudes"][name[7:]]
        return Mixture(d["skew"], d["ekurt"], name)
    raise ValueError("unknown distribution %r" % (name,))
