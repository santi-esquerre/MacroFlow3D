"""Synthetic worlds for the addendum tests: a FROZEN world (the three frozen arms with their REAL entries of ``prereg.json``, tables in the
executor's format, a freeze manifest) and an ADDENDUM campaign (arms built by ``data.arm_cfgs`` from the REAL ``prereg_addendum.json``).
Values are deterministic functions of (label, seed, window): a common component per seed (common random numbers) times a factor per arm times
a small own noise.  Built with the frozen test fixture ``sf43ana.tests.fixtures.write_world`` (the real ``sf42red`` table writer)."""
import copy
import csv
import hashlib
import json
import os
import zlib

import numpy as np

import sf43ana  # noqa: F401
from sf43ana.tests import fixtures as fx

from sf43add.ana import data

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.path.normpath(os.path.join(HERE, "..", "..", "..", ".."))
ADD_PATH = os.path.join(A, "prereg_addendum.json")
PREREG_PATH = os.path.join(A, "prereg.json")
RESULTS = os.path.join(A, "results")
MEAN = {"alpha_LP": 8.0, "alpha_TP": 0.006, "alpha_TAP": 0.0055}
NOISE = 0.004


def z(*key):
    return float(np.random.default_rng(zlib.crc32("|".join(map(str, key)).encode())).standard_normal())


def value(label, seed, window, est, factor=1.0, noise=NOISE):
    return MEAN[est] * (1.0 + 0.1 * z("common", seed, est)) * factor * (1.0 + noise * z(label, seed, window, est))


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def frozen_arms():
    """The REAL entries of the frozen pre-registration for the world writer (``selection`` dropped: the fixture skips arms with their own selection)."""
    with open(PREREG_PATH) as f:
        by = {a["id"]: a for a in json.load(f)["arms"]}
    out = [copy.deepcopy(by[i]) for i in ("s4_g5p", "s4_l16", "s4_g5pol", "s1_dist")]
    for a in out:
        a.pop("selection", None)
    return out


def _hook(spec, label_of):
    """``extra_rows`` of ``write_world``: overwrite the estimators of each row by the world's model, set the crossing counts, add the Pollock rows."""
    def hook(src, seg, seed, window, row):
        lab = label_of.get(src, src)
        f = spec["far"].get(lab, 1.0) if window == "far" else 1.0
        for est in ("alpha_TP", "alpha_TAP"):
            row[est] = value(lab, seed, window, est, spec["tp"].get(lab, 1.0) * f)
        row["alpha_LP"] = value(lab, seed, window, "alpha_LP", spec["lp"].get(lab, 1.0))
        row.update(n_traced=10000.0, n_last=10000.0 - spec["lost"].get(lab, 0) * (1 + seed % 3), U_r=1.0 + 0.01 * z(lab, seed, "U"))
        out = []
        if src == "s4_l16":
            out.append(dict(row, code="pollock", weights="unif", alpha_LP=value("s4_pollock", seed, window, "alpha_LP", spec["lp"].get("s4_pollock", 1.0)),
                            n_last=10000.0))
        return out
    return hook


def merge(spec):
    """Per-key merge of a scenario into ``DEFAULT`` (a scenario changes single arms, the other factors stay)."""
    spec = spec or {}
    return {k: {**DEFAULT.get(k, {}), **spec.get(k, {})} for k in set(DEFAULT) | set(spec)}


DEFAULT = {"lp": {"s4_g5p": 1.0, "s4_pollock": 1.16}, "tp": {}, "far": {"s1_dist": 1.02}, "lost": {"s4_g5p": 30}}


def freeze_manifest(reduced, path):
    lines = []
    for root, _, files in sorted(os.walk(reduced)):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            lines.append("%s  ./%s" % (sha(p), os.path.relpath(p, reduced).replace(os.sep, "/")))
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def crossings(root, arm, nreal_of=None):
    """Writes ``segments/<arm>/<seg>/out/crossings/realization_%04d_planes.csv`` per Par2 realization and the ``files`` lists of the table manifests."""
    for tp in sorted(os.listdir(os.path.join(root, "reduced", arm))):
        seg = tp
        d = os.path.join(root, "reduced", arm, seg)
        main = os.path.join(d, "%s__%s__d25-94.csv" % (arm, seg))
        s0 = int(seg.rsplit("_s", 1)[1])
        with open(main) as fh:
            seeds = sorted({s0 + int(float(r["r"])) for r in csv.DictReader(fh)})
        files = []
        for s in seeds:
            rel = "out/crossings/realization_%04d_planes.csv" % (s - s0)
            p = os.path.join(root, "segments", arm, seg, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as f:
                f.write("# crossings of seed %d\n" % s)
            files.append({"path": rel, "sha256": sha(p), "size": os.path.getsize(p)})
        for fn in os.listdir(d):
            if fn.endswith(".csv") and not fn.endswith(".planes.csv"):
                with open(os.path.join(d, fn + ".manifest.json"), "w") as f:
                    json.dump({"files": files}, f)


class World:
    """Paths of a frozen world and (optionally) an addendum campaign; ``addendum`` is a written copy of ``prereg_addendum.json``."""


def frozen_world(root, spec=None, with_crossings=True):
    spec = merge(spec)
    arms = frozen_arms()
    fx.write_world(root, {"amplitudes": {}, "arms": arms}, lambda *a: 0.0, segments=29, ell=0.1, qualification=None,
                       extra_rows=_hook(spec, {"s4_l16": "s4_l16"}))
    if with_crossings:
        crossings(root, "s4_g5p")
    freeze_manifest(os.path.join(root, "reduced"), os.path.join(root, "freeze.sha256"))
    return {"frozen_reduced": os.path.join(root, "reduced"), "frozen_manifest": os.path.join(root, "freeze.sha256"),
            "frozen_ledger": os.path.join(root, "campaign", "ledger.jsonl"), "frozen_segments": os.path.join(root, "segments")}


def campaign_world(root, spec, add=None, job="production", counts=None, seeds=None, fail=(), replace=(), segments=3, with_crossings=True, only=None):
    """Writes ``<root>/ledger.jsonl`` and ``<root>/reduced/`` for the addendum arms of ``job``.  ``counts``: {arm: n seeds written} (fewer than
    planned); ``seeds``: {arm: explicit seed list} (the smoke); ``fail``: [(arm, seed)] unusable solves; ``None`` for an arm in counts skips it."""
    add = add or data.load_addendum(ADD_PATH)
    cfgs = data.arm_cfgs(add)
    arms = []
    for a in add["arms"]:
        if a["job"] != job or (counts and counts.get(a["id"], 1) == 0) or (only and a["id"] not in only):
            continue
        c = dict(cfgs[a["id"]])
        c.pop("selection")
        if seeds and a["id"] in seeds:
            c.update(seeds=list(seeds[a["id"]]), reserves=[], n_conf=len(seeds[a["id"]]), n_max=len(seeds[a["id"]]))
        elif counts and a["id"] in counts:
            c.update(seeds=c["seeds"][:counts[a["id"]]], n_conf=counts[a["id"]], n_max=counts[a["id"]])
        arms.append(c)
    spec = merge(spec)
    fx.write_world(root, {"amplitudes": {}, "arms": arms}, lambda *a: 0.0, segments=segments, ell=0.1, qualification=None, fail=fail, replace=replace,
                   extra_rows=_hook(spec, {c["id"]: spec.get("label", {}).get(c["id"], c["id"]) for c in arms}))
    os.replace(os.path.join(root, "campaign", "ledger.jsonl"), os.path.join(root, "ledger.jsonl"))
    if with_crossings:
        for c in arms:
            if c["code"] == "par2":
                crossings(root, c["id"])
    return root


def write_addendum(path, frozen_manifest):
    """A copy of the real ``prereg_addendum.json`` whose pre-registered manifest sha256 is that of the synthetic freeze manifest."""
    add = data.load_addendum(ADD_PATH)
    add["frozen_prereg"] = PREREG_PATH
    add["frozen_reduced"] = dict(add["frozen_reduced"], manifest_sha256=sha(frozen_manifest))
    with open(path, "w") as f:
        json.dump(add, f)
    return add
