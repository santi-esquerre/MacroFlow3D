"""post hoc (2026-11-04), SF-43 addendum 1, node A1-POSTHOC: shared helpers.  Imports the FROZEN packages unmodified (sf43ana, sf43cov,
SF-42 mcstat / sf42red); never writes into the frozen artifact.

Every script is run from anywhere as ``python3 <script>.py SCRATCH`` with SCRATCH made by ``build_scratch.py``; the outputs go to
``<artifact>/posthoc/<analysis>/``.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ART = os.path.dirname(SCRIPTS)
SF42 = os.path.join(ART, "..", "2026-10-30-sf42-confirmatory-tooling", "scripts")
for _p in (SCRIPTS, SF42):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from sf43ana import config as cfgmod  # noqa: E402
from sf43ana.loader import load_all  # noqa: E402

PREREG = os.path.join(ART, "prereg.json")
OUT = os.path.join(ART, "posthoc")
LABEL = "post hoc (2026-11-04)"


def load(scratch):
    """-> (prereg, {arm id: ArmData}, ledger) from the frozen pre-registration and the scratch campaign layout."""
    pr = cfgmod.load_prereg(PREREG)
    camp = os.path.join(scratch, "camp")
    arms, _, ledger = load_all(pr, camp, os.path.join(camp, "reduced"))
    return pr, arms, ledger


def outdir(name):
    d = os.path.join(OUT, name)
    os.makedirs(d, exist_ok=True)
    return d


def dump(path, obj):
    def conv(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        raise TypeError(type(o))

    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=conv, allow_nan=True)
        f.write("\n")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def ci(samples, level=0.90, axis=0):
    a = 0.5 * (1.0 - level)
    return np.quantile(samples, [a, 1.0 - a], axis=axis)


def ols_slope(x, y):
    xc = x - x.mean()
    return float((xc * (y - y.mean())).sum() / (xc * xc).sum())
