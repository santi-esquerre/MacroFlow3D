"""Campaign ledger + reduced tables -> per-arm, per-seed arrays (prefix rule, group-wide replacements): THE data contract of ``sf43ana``.

Inputs are exactly what the campaign engine (``sf43camp``) writes:

* the LEDGER ``<campaign>/ledger.jsonl`` (JSON lines; a torn last line is ignored).  Used records (``sf43camp/lanes.py``):
  ``seed`` (one per solve: ``arm``, ``segment`` = segment directory name, ``seed``, ``seed0`` = first seed of the unit, ``verdict``, ``class``
  ``ok`` | ``unusable_solve``, ``iterations``, ``rel``, ``max_div``), ``replacement`` (``group``, ``seed``, ``replaced_by``),
  ``arm_halt`` (``arm``, ``reason``), ``reduced`` / ``unreduced`` (``arm``, ``segment``, ``seeds``: a segment listed ``unreduced`` and not
  reduced afterwards has its seeds MISSING, explicitly) and ``reest`` (``arm``, ``n_new``: the blinded re-estimation).  Per (arm, seed) the
  LATEST ``seed`` record counts.
* the REDUCED TABLES written by ``sf43camp.runner_executor.reduce``: ``<reduced>/<arm>/<segdir>/<arm>__<segdir>__d<lo>-<hi>.csv``.
  THE window rule: the file stem is ``d%g-%g`` of the window bounds (``window_stem``), whatever the semantic label of the window in the
  pre-registration (``main``, ``h1``, ``h2``, ``far``); ``reduced_pattern`` has ``{arm}``, ``{segment}`` = ``<segdir>``, ``{stem}``,
  ``{window}``, ``{lo}``, ``{hi}``.  A consumer row carries its ``seed``; a Par2 row has ``seed = nan`` and the realization index ``r`` of its
  segment, its seed is ``seed0 + r`` (seed0 from the ``seed`` records).  Rows are keyed by ``(code, family, weights, basis, set, hmax_div,
  seed)`` with ``code`` in ``consumer``, ``par2``, ``pollock``.

Analysis set of an arm = the longest COMPLETE prefix of its pre-registered seed order, after the group-wide replacements of the ledger
(a replaced seed's position is taken by its reserve; chains are followed), cut at the arm's ``n_conf`` (planned) or, for an entry with
``reest: true`` whose engine arm has a ``reest`` record, at ``n_new`` (never below ``n_conf``, never beyond the seeds listed).  A position is complete when the
ledger's latest record for (source arm, seed) is completed with a usable verdict (``ok_verdicts``) AND the reduced table of that
record's segment has the selected rows (both bases) with finite ``alpha_TP``, ``alpha_TAP``, ``alpha_LP``, ``U_r`` in the primary
window.  The first incomplete position ends the prefix (nothing later is used, whatever its state): no outcome-dependent selection.
An arm whose prefix is shorter than its ``floor`` has ``status = 'below_floor'``.

Row selection per arm: ``prereg.selection`` (code, family, weights, set), the arm's ``code``, then the arm's own ``selection`` override;
Par2 rows exist with ``weights = unif`` only (applied unless overridden), ``hmax_div`` = the arm's.

``data`` (``build_data``): the one structure every sibling module (``gate5``, ``g7``, ``secondaries``, ``failures``,
``positive_control``) reads::

    {"arms":   {arm id: {"seed": int array (N,) in PREFIX ORDER, "verdict": str array (N,), "N": int, "status": "ok"|"below_floor",
                         "cols": {name: float array (N,)}, "segment": [segdir per seed]}},
     "ledger": [{"arm","segment","seed","verdict","class",["replaces"]}, ...],      # one per (arm, seed), latest record
     "halted": [arm, ...], "sources": {ledger arm: [arm ids reading its tables]}, "coverage_summary": dict|None}

Column names: ``<est>`` is the POSITION basis, ``<est>:<basis>`` any other (``alpha_TP:disp``) for the estimators ``alpha_TP``,
``alpha_TAP``, ``alpha_LP``, ``U_r`` at the primary window; ``alpha_LAM:time``, ``alpha_Tfixed:time``, ``alpha_TAM:time`` and
``t_end_adv`` (= ``t_end U_time / ell``, advection times) from the ``family=time`` rows (weights flux for the consumer, unif for Par2);
``alpha_TP:expI``, ``alpha_LP:expI`` (consumer weights ``expI``); ``recrossings_per_traced``, ``frac_backflow``, ``seed_backflow`` (per-realization
tracker totals of the plane pass, ``sf42red reduce --tracker-counts``), ``n_last`` and ``n_traced`` (the counts of the row: for Par2 the
particles that crossed the last window plane and the particles injected at the first one, H12) and the legacy ``frac_recross``,
``alpha_LP:nobackflow``, each if (and only if) the table carries them.
"""
import csv
import json
import math
import os

import numpy as np

COLS = ("alpha_TP", "alpha_TAP", "alpha_LP", "U_r")
TIME_COLS = ("alpha_LAM", "alpha_Tfixed", "alpha_TAM")
# columns of the plane row of the primary window that are copied when the table carries them: H17 (``sf42red reduce --tracker-counts``:
# recrossings_per_traced, frac_backflow, seed_backflow; the legacy names frac_recross / alpha_LP:nobackflow are not written by any tool today)
# and H12 (n_last / n_traced: the Par2 crossing completeness n_last / n_traced of the main window)
OPTIONAL_COLS = ("frac_recross", "frac_backflow", "alpha_LP:nobackflow", "recrossings_per_traced", "seed_backflow", "n_last", "n_traced")


class LoadError(Exception):
    pass


class Ledger:
    """Parsed campaign ledger (see the module docstring)."""

    def __init__(self):
        self.seeds = {}      # (arm, seed) -> normalized latest ``seed`` record
        self.repl = {}       # group -> {seed: replacement}
        self.halted = {}     # arm -> reason
        self.seg_seed0 = {}  # (arm, segdir) -> seed0
        self.unreduced = {}  # (arm, segdir) -> [seeds] listed ``unreduced`` and not reduced afterwards
        self.reest = {}      # arm -> n_new of the (last) blinded re-estimation record
        self.n = 0
        self.verdict_unknown = 0


def segdir(path):
    return os.path.basename(os.path.normpath(path))


def window_stem(lo, hi):
    """The executor's file stem of a window (``sf43camp.runner_executor.reduce``: ``d%g-%g``): THE rule, whatever the window's label."""
    return "d%g-%g" % (lo, hi)


def read_ledger(path):
    L = Ledger()
    with open(path) as f:
        lines = f.read().split("\n")
    last = max((i for i, ln in enumerate(lines) if ln.strip()), default=-1)
    for ln, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError as e:
            if ln == last:  # a torn last line (the writer was killed mid-append) is ignored, as the engine does
                continue
            raise LoadError("%s:%d: %s" % (path, ln + 1, e))
        L.n += 1
        kind = r.get("kind")
        if kind == "replacement":
            g = L.repl.setdefault(r["group"], {})
            s, t = int(r["seed"]), int(r["replaced_by"])
            if s in g and g[s] != t:
                raise LoadError("%s:%d: seed %s replaced twice in group %s" % (path, ln + 1, s, r["group"]))
            g[s] = t
        elif kind == "arm_halt":
            L.halted[r["arm"]] = r.get("reason", "")
        elif kind == "seed":
            seg = segdir(r["segment"])
            L.seg_seed0[(r["arm"], seg)] = int(r["seed0"])
            L.seeds[(r["arm"], int(r["seed"]))] = {
                "arm": r["arm"], "segment": seg, "seed": int(r["seed"]), "verdict": r.get("verdict"), "class": r.get("class"),
                "iterations": r.get("iterations"), "rel": r.get("rel"), "max_div": r.get("max_div"), "seed0": int(r["seed0"])}
        elif kind == "unreduced":
            L.unreduced[(r["arm"], segdir(r["segment"]))] = [int(x) for x in r.get("seeds", [])]
        elif kind == "reduced":
            L.unreduced.pop((r["arm"], segdir(r["segment"])), None)  # reduced in a later session
        elif kind == "reest":
            L.reest[r["arm"]] = int(r["n_new"])
    L.verdict_unknown = sum(1 for r in L.seeds.values() if r["class"] == "ok" and r["verdict"] is None)
    return L


def effective_order(seeds, replacements):
    """Seed order after replacements: each replaced seed's position holds its (transitive) replacement."""
    out, applied = [], []
    for s in seeds:
        t, seen = int(s), {int(s)}
        while t in replacements:
            t = replacements[t]
            if t in seen:
                raise LoadError("replacement cycle at seed %d" % s)
            seen.add(t)
        out.append(t)
        if t != int(s):
            applied.append((int(s), t))
    if len(set(out)) != len(out):
        raise LoadError("a replacement reserve is already in the seed order")
    return out, applied


def selection_of(prereg, cfg):
    """Row selection of an arm: pre-registered default, the arm's code (Par2: uniform weights only), the arm's own override."""
    sel = dict(prereg["selection"])
    sel["code"] = cfg["code"]
    if cfg["code"] == "par2":
        sel["weights"] = "unif"
    sel.update(cfg.get("selection", {}))
    return sel


class TableCache:
    """Reads the reduced tables once: ``table(source arm, segdir, window)`` -> {(code, family, weights, basis, set, hmax_div, seed): row}
    (``row``: dict of the raw strings), or ``None`` if the file is absent.  Par2 seeds come from ``segment_seed0 + r``."""

    def __init__(self, reduced_dir, prereg, ledger):
        self.dir, self.p, self.ledger, self._c = reduced_dir, prereg, ledger, {}

    def path(self, arm, segment, window):
        lo, hi = self.p["windows"][window]
        return os.path.join(self.dir, self.p["reduced_pattern"].format(arm=arm, segment=segment, window=window, lo=lo, hi=hi,
                                                                       stem=window_stem(lo, hi)))

    def table(self, arm, segment, window):
        p = self.path(arm, segment, window)
        if p in self._c:
            return self._c[p]
        if not os.path.isfile(p):
            self._c[p] = None
            return None
        out = {}
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                seed = float(row["seed"]) if row["seed"] not in ("", "nan") else float("nan")
                if math.isnan(seed):
                    s0 = self.ledger.seg_seed0.get((arm, segment))
                    if s0 is None:
                        raise LoadError("%s: a row without a seed and no ledger seed0 for segment %s of arm %s" % (p, segment, arm))
                    seed = s0 + int(float(row["r"]))
                k = (row["code"], row["family"], row["weights"], row["basis"], row["set"], int(float(row["hmax_div"])), int(seed))
                if k in out:
                    raise LoadError("%s: duplicate row %r" % (p, k))
                out[k] = row
        self._c[p] = out
        return out


def _num(row, col):
    v = row.get(col, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


class ArmData:
    def __init__(self, cfg):
        self.id, self.cfg = cfg["id"], cfg
        self.amplitude, self.group = cfg["amplitude"], cfg["group"]
        self.n_conf, self.floor = cfg["n_conf"], cfg["floor"]
        self.n_cut = cfg["n_conf"]  # analysis cut: n_conf (planned), or n_new of the blinded re-estimation (load_arm)
        self.seeds = np.zeros(0, dtype=int)
        self.verdicts, self.segments = [], []
        self.values = {}  # (window, basis, column) -> array (N,)
        self.extra = {}   # extra column name -> array (N,)
        self.info = {}

    @property
    def N(self):
        return int(self.seeds.size)

    @property
    def usable(self):
        return self.N >= self.floor

    def get(self, window, basis, col, n=None):
        v = self.values[(window, basis, col)]
        return v if n is None else v[:n]

    def as_data(self, prereg):
        """The ``data["arms"][id]`` view (module docstring)."""
        prim = prereg["primary_window"]
        cols = {}
        for b in prereg["bases"].values():
            for c in COLS:
                cols[c if b == "pos" else "%s:%s" % (c, b)] = self.values[(prim, b, c)]
        cols.update(self.extra)
        return {"seed": self.seeds, "verdict": np.array(self.verdicts, dtype=object), "N": self.N, "status": self.info.get("status"),
                "cols": cols, "segment": list(self.segments)}


def _extra_columns(cfg, prereg, sel, seg_of, seeds, tables):
    """Columns of the sibling modules from other rows of the primary-window tables (see the module docstring)."""
    prim = prereg["primary_window"]
    n = len(seeds)
    consumer = cfg["code"] == "consumer"
    wt = sel["weights"] if consumer else "unif"
    variants = {"%s:time" % c: ("time", wt, "pos", c) for c in TIME_COLS}
    if consumer:
        variants.update({"alpha_TP:expI": ("plane", "expI", "pos", "alpha_TP"), "alpha_LP:expI": ("plane", "expI", "pos", "alpha_LP")})
    out = {k: np.full(n, np.nan) for k in list(variants) + ["t_end_adv"] + list(OPTIONAL_COLS)}
    seen = {k: False for k in out}
    for i, s in enumerate(seeds):
        tab = tables.table(cfg["source"], seg_of[int(s)], prim)
        if tab is None:
            continue
        for name, (fam, w, basis, col) in variants.items():
            row = tab.get((cfg["code"], fam, w, basis, sel["set"], cfg["hmax_div"], int(s)))
            if row is not None:
                out[name][i] = _num(row, col)
        row = tab.get((cfg["code"], "time", wt, "pos", sel["set"], cfg["hmax_div"], int(s)))
        if row is not None:
            te, ut, ell = _num(row, "t_end"), _num(row, "U_time"), _num(row, "ell")
            if ell > 0:
                out["t_end_adv"][i] = te * ut / ell
        prow = tab.get((cfg["code"], sel["family"], sel["weights"], "pos", sel["set"], cfg["hmax_div"], int(s)))
        if prow is not None:
            for c in OPTIONAL_COLS:
                if c in prow:
                    out[c][i] = _num(prow, c)
                    seen[c] = True
    return {k: v for k, v in out.items() if not np.all(np.isnan(v)) or seen.get(k)}


def load_arm(cfg, prereg, ledger, tables):
    a = ArmData(cfg)
    prim, bases = prereg["primary_window"], (prereg["bases"]["primary"], prereg["bases"]["secondary"])
    order, applied = effective_order(cfg["seeds"], ledger.repl.get(cfg["group"], {}))
    ok = set(prereg["ok_verdicts"])
    sel = selection_of(prereg, cfg)
    n_new = ledger.reest.get(cfg["source"]) if cfg.get("reest") else None
    if n_new is not None:
        a.n_cut = min(max(n_new, a.n_conf), len(order))
    info = {"n_conf": a.n_conf, "n_cut": a.n_cut, "reest_n_new": n_new, "floor": a.floor,
            "replacements_applied": [{"seed": s, "replaced_by": t} for s, t in applied], "stopped_at": None, "missing_tables": [],
            "selection": sel}
    seg_of, verd, n = {}, [], 0
    for pos in range(a.n_cut):
        s = order[pos]
        rec = ledger.seeds.get((cfg["source"], s))
        why = None
        if rec is None:
            why = "no ledger record"
        elif rec["class"] != "ok":
            why = "class %r (verdict %r)" % (rec["class"], rec["verdict"])
        elif (cfg["source"], rec["segment"]) in ledger.unreduced:
            why = "unreduced segment %s (ledger 'unreduced': completed, never reduced)" % rec["segment"]
        elif rec["verdict"] is not None and rec["verdict"] not in ok:
            why = "verdict %r" % (rec["verdict"],)
        else:
            tab = tables.table(cfg["source"], rec["segment"], prim)
            if tab is None:
                why = "missing reduced table %s" % tables.path(cfg["source"], rec["segment"], prim)
                info["missing_tables"].append(tables.path(cfg["source"], rec["segment"], prim))
            else:
                for b in bases:
                    row = tab.get((sel["code"], sel["family"], sel["weights"], b, sel["set"], cfg["hmax_div"], s))
                    if row is None or not all(math.isfinite(_num(row, c)) for c in COLS):
                        why = "no complete finite row in the reduced table"
                        break
        if why:
            info["stopped_at"] = {"position": pos, "seed": s, "reason": why}
            break
        seg_of[s] = rec["segment"]
        verd.append(rec["verdict"] if rec["verdict"] is not None else "unknown")
        n += 1
    a.seeds = np.array(order[:n], dtype=int)
    a.verdicts, a.segments = verd, [seg_of[int(s)] for s in a.seeds]
    for w in prereg["windows"]:
        for b in bases:
            arr = {c: np.full(n, np.nan) for c in COLS}
            for i, s in enumerate(a.seeds):
                tab = tables.table(cfg["source"], seg_of[int(s)], w)
                row = None if tab is None else tab.get((sel["code"], sel["family"], sel["weights"], b, sel["set"], cfg["hmax_div"], int(s)))
                if row is not None:
                    for c in COLS:
                        arr[c][i] = _num(row, c)
            for c in COLS:
                a.values[(w, b, c)] = arr[c]
    a.extra = _extra_columns(cfg, prereg, sel, seg_of, a.seeds, tables)
    info["N"] = n
    info["status"] = "ok" if n >= a.floor else "below_floor"
    info["complete_prefix_equals_n_cut"] = n == a.n_cut
    a.info = info
    return a


def load_all(prereg, campaign_dir, reduced_dir):
    """-> ({arm id: ArmData} for EVERY pre-registered arm (both codes), process info, Ledger)."""
    ledger = read_ledger(os.path.join(campaign_dir, prereg["ledger"]))
    tables = TableCache(reduced_dir, prereg, ledger)
    arms = {aid: load_arm(cfg, prereg, ledger, tables) for aid, cfg in prereg.arms.items()}
    proc = {"ledger_records": ledger.n, "replacement_groups": {g: len(m) for g, m in ledger.repl.items()},
            "halted_arms": dict(ledger.halted), "verdict_unknown": ledger.verdict_unknown,
            "unreduced": {"%s/%s" % k: v for k, v in sorted(ledger.unreduced.items())}, "reest_n_new": dict(ledger.reest)}
    return arms, proc, ledger


def common_prefix(arms):
    """Common prefix length of arms that share seeds (paired use); the seed lists must agree on it."""
    n = min(a.N for a in arms)
    for a in arms[1:]:
        if not np.array_equal(a.seeds[:n], arms[0].seeds[:n]):
            raise LoadError("arms %s do not share their seed order on the common prefix" % [x.id for x in arms])
    return n


def build_data(prereg, arms, ledger, coverage_summary=None):
    """The ``data`` structure of the module docstring."""
    repl_of = {}
    for g, m in ledger.repl.items():
        for s, t in m.items():
            repl_of[(g, t)] = s
    groups_of = {}
    for a in prereg.arms.values():
        groups_of.setdefault(a["source"], set()).add(a["group"])
    recs = []
    for (arm, seed), r in sorted(ledger.seeds.items()):
        rec = {"arm": arm, "segment": r["segment"], "seed": seed, "verdict": r["verdict"], "class": r["class"]}
        for g in sorted(groups_of.get(arm, ())):
            if (g, seed) in repl_of:
                rec["replaces"] = repl_of[(g, seed)]
        recs.append(rec)
    for arm, why in sorted(ledger.halted.items()):
        recs.append({"arm": arm, "segment": None, "seed": -1, "verdict": None, "class": "halted", "reason": why})
    sources = {}
    for aid, cfg in prereg.arms.items():
        sources.setdefault(cfg["source"], []).append(aid)
    return {"arms": {aid: a.as_data(prereg) for aid, a in arms.items()}, "ledger": recs, "halted": sorted(ledger.halted),
            "sources": sources, "coverage_summary": coverage_summary,
            "unreduced": {"%s/%s" % k: v for k, v in sorted(ledger.unreduced.items())}}
