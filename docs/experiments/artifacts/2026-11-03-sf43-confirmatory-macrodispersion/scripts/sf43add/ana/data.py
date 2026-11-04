"""Data of addendum 1 through the FROZEN ``sf43ana.loader`` (imported, not copied).

* frozen arms (``s4_g5p``, ``s4_g5pol``, ``s1_dist``): the entries of the frozen ``prereg.json`` (validated by ``sf43ana.config``), tables
  read from the frozen reduced directory, every file verified against the freeze manifest before it is parsed (``Frozen``);
* addendum arms: entries built from ``prereg_addendum.json["arms"]`` (``arm_cfgs``: Par2 arms like ``s4_g5p`` with uniform weights, consumer
  arms like ``s1_dist``), tables of the addendum campaign, its ledger;
* ``View``: per-seed arrays of an arm's analysis set (longest complete prefix after the group-wide replacements: the loader's rule), with
  the seed restriction and concatenation the paired contrasts and J8 need.
"""
import json
import os

import numpy as np

import sf43ana  # noqa: F401
from sf43ana import config as cfgmod
from sf43ana.gate5 import pair_by_seed
from sf43ana.loader import LoadError, TableCache, load_arm, read_ledger
from sf43ana.summary import sha256

FROZEN_ARMS = ("s4_g5p", "s4_g5pol", "s1_dist")
LP, TP_MAIN, TP_FAR = ("main", "pos", "alpha_LP"), ("main", "disp", "alpha_TP"), ("far", "disp", "alpha_TP")


def resolve(path, near):
    """``path`` as given (relative to the working directory), else its basename next to the file ``near``."""
    return path if os.path.exists(path) else os.path.join(os.path.dirname(os.path.abspath(near)), os.path.basename(path))


def load_addendum(path):
    with open(path) as f:
        return json.load(f)


def read_manifest(path):
    """``{"./<arm>/<segment>/<file>": sha256}`` of the freeze manifest (``<sha256>  ./<path>`` lines)."""
    out = {}
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if line.strip():
                h, sep, rel = line.partition("  ")
                if not sep or len(h) != 64:
                    raise LoadError("%s:%d: not a '<sha256>  ./<path>' line" % (path, n))
                out[rel] = h
    return out


class Frozen:
    """Read-only frozen reduced directory; ``verify`` checks a file against the freeze manifest (a missing entry or a mismatch is an error)."""

    def __init__(self, reduced_dir, manifest_path, expected_manifest_sha=None):
        self.dir, self.manifest, self.verified = os.path.abspath(reduced_dir), read_manifest(manifest_path), {}
        self.manifest_sha256 = sha256(manifest_path)
        if expected_manifest_sha and self.manifest_sha256 != expected_manifest_sha:
            raise LoadError("the freeze manifest does not have the pre-registered sha256")

    def verify(self, path):
        rel = "./" + os.path.relpath(os.path.abspath(path), self.dir).replace(os.sep, "/")
        if rel not in self.manifest:
            raise LoadError("frozen file not listed in the freeze manifest: %s" % rel)
        if rel not in self.verified:
            if sha256(path) != self.manifest[rel]:
                raise LoadError("sha256 mismatch against the freeze manifest: %s" % rel)
            self.verified[rel] = self.manifest[rel]
        return path


class FrozenTables(TableCache):
    """``TableCache`` that verifies each frozen table before parsing it."""

    def __init__(self, frozen, prereg, ledger):
        super().__init__(frozen.dir, prereg, ledger)
        self.frozen = frozen

    def table(self, arm, segment, window):
        p = self.path(arm, segment, window)
        if p not in self._c and os.path.isfile(p):
            self.frozen.verify(p)
        return super().table(arm, segment, window)


def arm_cfgs(add):
    """Loader entries of the addendum arms (like ``s4_g5p`` for Par2, like ``s1_dist`` for the consumer)."""
    out = {}
    for a in add["arms"]:
        s, r, par2 = a["seeds"], a.get("reserves") or {}, a["runner"] == "par2"
        out[a["id"]] = {"id": a["id"], "amplitude": "s4" if par2 else "s1", "group": a["group"], "source": a["id"], "hmax_div": -1,
                        "seeds": list(range(s["start"], s["start"] + s["count"])), "n_conf": s["count"], "n_max": s["count"], "floor": 2,
                        "reserves": list(range(r["start"], r["start"] + r["count"])) if r else [], "role": "addendum", "reest": False,
                        "code": "par2" if par2 else "consumer", "selection": {"weights": "unif"} if par2 else {}}
    return out


def load_ledger(campaign):
    path = os.path.join(campaign, "ledger.jsonl")
    if not os.path.isfile(path):
        raise LoadError("no ledger.jsonl in the campaign directory %s" % campaign)
    return read_ledger(path)


class View:
    """Arrays of one arm's analysis set: ``seeds`` (int), ``vals`` {(window, basis, column): array}, ``extra`` {column: array}."""

    def __init__(self, id, seeds, vals, extra, info=None):
        self.id, self.seeds, self.vals, self.extra, self.info = id, np.asarray(seeds, dtype=int), vals, extra, info or {}

    @classmethod
    def of(cls, a):
        return cls(a.id, a.seeds, dict(a.values), dict(a.extra), {"N": a.N, "n_conf": a.n_conf, "stopped_at": a.info.get("stopped_at"),
                                                                    "missing_tables": len(a.info.get("missing_tables", []))})

    @property
    def N(self):
        return int(self.seeds.size)

    def get(self, key):
        return self.vals[key]

    def where(self, lo, hi):
        m = (self.seeds >= lo) & (self.seeds <= hi)
        return View(self.id, self.seeds[m], {k: v[m] for k, v in self.vals.items()}, {k: v[m] for k, v in self.extra.items()}, self.info)

    def concat(self, other):
        keys = self.vals.keys() & other.vals.keys()
        ex = self.extra.keys() & other.extra.keys()
        return View(self.id, np.concatenate([self.seeds, other.seeds]), {k: np.concatenate([self.vals[k], other.vals[k]]) for k in keys},
                    {k: np.concatenate([self.extra[k], other.extra[k]]) for k in ex}, self.info)

    def completeness(self):
        """``n_last / n_traced`` per realization (nan where undefined), or None if the tables carry no crossing counts (H12)."""
        if "n_last" not in self.extra or "n_traced" not in self.extra:
            return None
        nl, nt = self.extra["n_last"], self.extra["n_traced"]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(nt > 0, nl / nt, np.nan)

    def complete(self):
        """Boolean per realization: ``n_last == n_traced`` (None without counts)."""
        c = self.completeness()
        return None if c is None else (self.extra["n_last"] == self.extra["n_traced"]) & np.isfinite(c)


def pair(va, ka, vb, kb):
    """Values of two views on their common seeds, finite in both, sorted by seed: ``(seeds, a, b, info)`` (frozen ``pair_by_seed``)."""
    return pair_by_seed(va.seeds, va.get(ka), vb.seeds, vb.get(kb))


def common_all(views_keys):
    """Arrays of several (view, key) on the seeds common to all and finite in all: ``(seeds, [arrays])``."""
    seeds = None
    for v, k in views_keys:
        ok = {int(s) for s, x in zip(v.seeds, v.get(k)) if np.isfinite(x)}
        seeds = ok if seeds is None else seeds & ok
    seeds = sorted(seeds)
    out = []
    for v, k in views_keys:
        idx = {int(s): i for i, s in enumerate(v.seeds)}
        out.append(np.array([v.get(k)[idx[s]] for s in seeds], dtype=float))
    return np.array(seeds, dtype=np.int64), out


def load_frozen(frozen_prereg, frozen, ledger_path):
    """-> (Prereg, {frozen arm id: View}, Ledger) reading the three frozen arms through the frozen loader, files verified."""
    P = cfgmod.load_prereg(frozen_prereg)
    ledger = read_ledger(ledger_path)
    tables = FrozenTables(frozen, P, ledger)
    return P, {a: View.of(load_arm(P.arms[a], P, ledger, tables)) for a in FROZEN_ARMS}, ledger


def load_campaign(P, add, campaign, job):
    """{arm id: View} of the addendum arms of ``job`` ('production' | 'j8_conditional'), from ``campaign`` (ledger + ``reduced/``)."""
    ledger = load_ledger(campaign)
    tables = TableCache(os.path.join(campaign, "reduced"), P, ledger)
    cfgs = arm_cfgs(add)
    return {a["id"]: View.of(load_arm(cfgs[a["id"]], P, ledger, tables)) for a in add["arms"] if a["job"] == job}, ledger
