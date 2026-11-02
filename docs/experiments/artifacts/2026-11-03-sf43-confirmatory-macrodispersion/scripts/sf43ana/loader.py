"""Reduced tables + campaign ledger -> per-arm, per-seed arrays (prefix rule and group-wide replacements).

Analysis set of an arm = the longest COMPLETE prefix of its pre-registered seed order, after the group-wide replacements of the
ledger (a replaced seed's position is taken by its reserve; chains are followed), cut at ``n_conf``.  A position is complete when
the ledger's latest record for (source arm, seed) has an accepted verdict AND the reduced table of that record's segment has the
row (selection: code, family, weights, set, hmax_div; both bases) with finite ``alpha_TP``, ``alpha_TAP``, ``alpha_LP``, ``U_r``
in the primary window.  The first incomplete position ends the prefix (nothing later is used, whatever its state): no
outcome-dependent selection.  An arm whose prefix is shorter than its ``floor`` has ``status = 'below_floor'``: its hypotheses are
untestable and its estimates descriptive.

Ledger: JSON lines.  Seed records ``{"arm","segment","seed","verdict","class"}`` (no ``kind``, or ``"kind": "seed"``); replacement
records ``{"kind": "replacement", "seed", "replaced_by", "group"}``.  Reduced tables: ``reduced_pattern.format(arm=<source arm>,
segment=<segment of the ledger record>, window=<window label>)`` under the ``--reduced`` directory, in the ``sf42red`` format.
"""
import csv
import json
import math
import os

import numpy as np

COLS = ("alpha_TP", "alpha_TAP", "alpha_LP", "U_r")
TABLE_NUM = COLS


class LoadError(Exception):
    pass


def read_ledger(path):
    """-> (latest seed record per (arm, seed), {group: {seed: replaced_by}}, number of records)."""
    seeds, repl, n = {}, {}, 0
    with open(path) as f:
        for ln, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except ValueError as e:
                raise LoadError("%s:%d: %s" % (path, ln, e))
            n += 1
            if r.get("kind") == "replacement":
                g = repl.setdefault(r["group"], {})
                if r["seed"] in g and g[r["seed"]] != r["replaced_by"]:
                    raise LoadError("%s:%d: seed %s replaced twice in group %s" % (path, ln, r["seed"], r["group"]))
                g[int(r["seed"])] = int(r["replaced_by"])
            elif r.get("kind", "seed") == "seed":
                seeds[(r["arm"], int(r["seed"]))] = r
    return seeds, repl, n


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


class TableCache:
    def __init__(self, reduced_dir, prereg):
        self.dir, self.p, self._c = reduced_dir, prereg, {}

    def path(self, arm, segment, window):
        return os.path.join(self.dir, self.p["reduced_pattern"].format(arm=arm, segment=segment, window=window))

    def rows(self, arm_cfg, segment, window):
        """dict (seed, basis) -> {column: float} of the selected rows of one table, or None if the file is absent."""
        p = self.path(arm_cfg["source"], segment, window)
        key = (p, arm_cfg["hmax_div"], arm_cfg["code"])
        if key in self._c:
            return self._c[key]
        if not os.path.isfile(p):
            self._c[key] = None
            return None
        sel = dict(self.p["selection"], code=arm_cfg["code"])
        out = {}
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                if any(row[k] != v for k, v in sel.items()) or int(float(row["hmax_div"])) != arm_cfg["hmax_div"]:
                    continue
                k = (int(float(row["seed"])), row["basis"])
                if k in out:
                    raise LoadError("%s: duplicate row for seed %d basis %s" % (p, k[0], k[1]))
                out[k] = {c: float(row[c]) for c in TABLE_NUM}
        self._c[key] = out
        return out


class ArmData:
    def __init__(self, cfg):
        self.id, self.cfg = cfg["id"], cfg
        self.amplitude, self.group = cfg["amplitude"], cfg["group"]
        self.n_conf, self.floor = cfg["n_conf"], cfg["floor"]
        self.seeds = np.zeros(0, dtype=int)
        self.values = {}  # (window, basis, column) -> array (N,)
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


def load_arm(cfg, prereg, ledger, replacements, tables):
    a = ArmData(cfg)
    prim, bases = prereg["primary_window"], (prereg["bases"]["primary"], prereg["bases"]["secondary"])
    order, applied = effective_order(cfg["seeds"], replacements.get(cfg["group"], {}))
    ok = set(prereg["ok_verdicts"])
    info = {"n_conf": a.n_conf, "floor": a.floor, "replacements_applied": [{"seed": s, "replaced_by": t} for s, t in applied],
            "stopped_at": None, "missing_tables": []}
    seg_of, n = {}, 0
    for pos in range(a.n_conf):
        s = order[pos]
        rec = ledger.get((cfg["source"], s))
        why = None
        if rec is None:
            why = "no ledger record"
        elif rec["verdict"] not in ok:
            why = "verdict %r (class %r)" % (rec["verdict"], rec.get("class"))
        else:
            rows = tables.rows(cfg, rec["segment"], prim)
            if rows is None:
                why = "missing reduced table %s" % tables.path(cfg["source"], rec["segment"], prim)
                info["missing_tables"].append(tables.path(cfg["source"], rec["segment"], prim))
            elif any((s, b) not in rows or not all(math.isfinite(rows[(s, b)][c]) for c in COLS) for b in bases):
                why = "no complete finite row in the reduced table"
        if why:
            info["stopped_at"] = {"position": pos, "seed": s, "reason": why}
            break
        seg_of[s] = rec["segment"]
        n += 1
    a.seeds = np.array(order[:n], dtype=int)
    for w in prereg["windows"]:
        for b in bases:
            arr = {c: np.full(n, np.nan) for c in COLS}
            for i, s in enumerate(a.seeds):
                rows = tables.rows(cfg, seg_of[int(s)], w)
                if rows is not None and (int(s), b) in rows:
                    for c in COLS:
                        arr[c][i] = rows[(int(s), b)][c]
            for c in COLS:
                a.values[(w, b, c)] = arr[c]
    info["N"] = n
    info["status"] = "ok" if n >= a.floor else "below_floor"
    info["complete_prefix_equals_n_conf"] = n == a.n_conf
    a.info = info
    return a


def load_all(prereg, campaign_dir, reduced_dir):
    """-> ({arm id: ArmData} for the arms of the selected code, process info)."""
    ledger, repl, nrec = read_ledger(os.path.join(campaign_dir, prereg["ledger"]))
    tables = TableCache(reduced_dir, prereg)
    arms, skipped = {}, []
    for aid, cfg in prereg.arms.items():
        if cfg["code"] != prereg["selection"]["code"]:
            skipped.append(aid)
            continue
        arms[aid] = load_arm(cfg, prereg, ledger, repl, tables)
    return arms, {"ledger_records": nrec, "replacement_groups": {g: len(m) for g, m in repl.items()}, "skipped_arms_other_code": skipped}


def common_prefix(arms):
    """Common prefix length of arms that share seeds (paired use); the seed lists must agree on it."""
    n = min(a.N for a in arms)
    for a in arms[1:]:
        if not np.array_equal(a.seeds[:n], arms[0].seeds[:n]):
            raise LoadError("arms %s do not share their seed order on the common prefix" % [x.id for x in arms])
    return n
