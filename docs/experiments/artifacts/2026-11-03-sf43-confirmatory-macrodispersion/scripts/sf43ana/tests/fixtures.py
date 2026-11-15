"""Synthetic campaign (pre-registration, campaign ledger, reduced tables in the executor's naming and the real `sf42red` format)
for the loader / end-to-end tests."""
import json
import os
import zlib

import numpy as np

import sf43ana  # noqa: F401  (puts the SF-42 scripts on sys.path)
from sf42red import reduce as red

WINDOWS = {"main": [25.0, 94.0], "h1": [25.0, 59.5], "h2": [59.5, 94.0], "far": [94.0, 194.0]}


def arm(aid, amp, group, seeds, n_conf, floor, reserves=(), source=None, hmax_div=-1, role="", code="consumer"):
    d = {"id": aid, "amplitude": amp, "group": group, "seeds": list(seeds), "n_conf": n_conf, "floor": floor,
         "reserves": list(reserves), "hmax_div": hmax_div, "role": role, "code": code}
    if source:
        d["source"] = source
    return d


def prereg_dict(n=40, nh=20):
    s1, s2, w1, w2, wc2 = range(1000, 1000 + n), range(3000, 3000 + n), range(2000, 2000 + n), range(4000, 4000 + n), range(5000, 5000 + n)
    r1, r2 = range(1900, 1910), range(3900, 3910)
    arms = [
        arm("p1", "1", "g1", s1, n, 20, r1, role="production"),
        arm("l5", "1", "g1", s1, n, 20, r1), arm("l20", "1", "g1", s1, n, 20, r1),
        arm("w48", "1", "gw1", w1, n, 20, range(2900, 2910), role="width"),
        arm("h16", "1", "gh1", list(s1)[:nh], nh, 10, source="p1", hmax_div=16),
        arm("h8", "1", "gh1", list(s1)[:nh], nh, 10, source="p1", hmax_div=8),
        arm("l4", "2.25", "g2", s2, n, 20, r2), arm("l8", "2.25", "g2", s2, n, 20, r2), arm("l12", "2.25", "g2", s2, n, 20, r2),
        arm("p2", "2.25", "g2", s2, n, 20, r2, role="production"), arm("p2h10", "2.25", "g2", s2, n, 20, r2),
        arm("w2f", "2.25", "gw2", w2, n, 20, range(4900, 4910)), arm("w2c", "2.25", "gw2", w2, n, 20, range(4900, 4910)),
        arm("par2_1", "1", "gp", range(7000, 7010), 10, 5, code="par2"),
        arm("w2far", "2.25", "gd", wc2, 30, 15),
        arm("g5c", "1", "gg", range(8000, 8040), 40, 20, range(8900, 8905)),
        arm("g5p", "1", "gg", range(8000, 8040), 40, 20, range(8900, 8905), code="par2"),
    ]
    amps = {"1": {"production": "p1", "components": ["E_h", "E_W", "E_int"],
                  "ladder": {"arms": ["l5", "p1", "l20"], "resolutions": [5, 10, 20], "production_index": 1},
                  "width": {"wide_fine": "w48", "prod_fine": "p1"}, "hmax": {"h16": "h16", "h8": "h8"}},
            "2.25": {"production": "p2", "components": ["E_h", "E_W", "E_WxH"],
                     "ladder": {"arms": ["l4", "l8", "p2"], "resolutions": [4, 8, 16], "production_index": 2,
                                "gates": [{"type": "extra_level", "triple": [0, 1, 2], "arm": "l12", "resolution": 12}]},
                     "width": {"wide_fine": "w2f", "prod_fine": "p2h10", "wide_coarse": "w2c", "prod_coarse": "l8"},
                     "distance_arm": "w2far"}}
    return {"amplitudes": amps, "arms": arms, "family": ["robust:1", "robust:2.25", "gate5:1", "gate5:2.25"],
            "bootstrap": {"B_sampling": 400, "B_budget": 300, "B_ucl": 400, "B_bca": 200, "seed": 11},
            "floors": {"production": 20, "ladder": 20, "width": 20, "hmax": 10, "gate5": 20},
            "targets": {"1": {"alpha_TP": 1e-3}},
            "first_order": {"1": {"alpha_TP": 0.006}}}


def qualification_doc(profiles=((3, 20, 20, 20, 10), (4, 20, 20, 20, 10)), sampling=(20,), qualified=True, complete=True, extra=()):
    """A ``qualification.json`` of ``sf43cov qualify`` (the fields ``sf43ana.qualification`` reads): one pipeline cell per profile
    ``(K, production N, ladder, width, h_max)`` (two cells, so that the profile aggregates) and one sampling grid point per ``n`` in ``sampling``.
    ``qualified``: True / False / a dict {profile or n: bool}; ``extra``: more profiles of the same form."""
    def q(key):
        return qualified.get(key, True) if isinstance(qualified, dict) else qualified

    doc = {"version": 1, "all_cells_complete": complete, "incomplete": [], "pipeline": [], "sampling": {}, "plan_hash": "fixture"}
    for prof in tuple(profiles) + tuple(extra):
        K, N, nl, nw, ni = prof
        for scen in ("zero", "bias"):
            doc["pipeline"].append({"cell": "K%d_%s" % (K, scen), "K": K, "N": N, "arm_sizes": {"ladder": nl, "h_max": ni, "width": nw},
                                    "scenario": scen, "dist": "normal", "qualified": q(prof)})
    for n in sampling:
        doc["sampling"]["n=%d" % n] = {d: {"lower": {"qualified": q(n), "max_ucl_over_tail": 1.3 if q(n) is False else 0.9}, "upper": {"qualified": True, "max_ucl_over_tail": 0.9}} for d in ("normal", "stress")}
    return doc


def _val(aid, seed, mu, common_sd, own_sd, group):
    z = np.random.default_rng([zlib.crc32(group.encode()), seed]).standard_normal()
    e = np.random.default_rng([zlib.crc32(aid.encode()), seed, 7]).standard_normal()
    return mu + common_sd * z + own_sd * e


def table_path(root, src, seg, lo, hi):
    return os.path.join(root, "reduced", src, seg, "%s__%s__d%g-%g.csv" % (src, seg, lo, hi))


def seed_record(arm_id, group, segment_path, seed, seed0, unit, verdict, cls):
    """The engine's per-solve ledger record (``sf43camp/lanes.py`` ``seed_records``)."""
    return {"kind": "seed", "arm": arm_id, "group": group, "segment": os.path.basename(segment_path), "segment_path": segment_path, "seed": seed,
            "seed0": seed0, "unit": unit, "verdict": verdict, "class": cls, "iterations": 30, "rel": 1e-12, "balance": 1e-10, "max_div": 2e-8,
            "field_seed": seed}


def write_world(root, prereg, mean_of, fail=(), replace=(), sd=(0.002, 0.0005), segments=2, extra_rows=None, ell=0.1, unreduced=(), reest=None,
                qualification="permissive"):
    """Writes ``prereg.json``, ``campaign/ledger.jsonl`` (``seed`` / ``replacement`` / ``unreduced`` / ``reest`` records of the engine; ``unit_end``
    too, which the loader ignores) and the reduced tables
    ``reduced/<arm>/<segdir>/<arm>__<segdir>__d<lo>-<hi>.csv`` of every arm.

    ``mean_of(arm_cfg, window, basis, est)`` -> mean of the estimator; ``fail``: iterable of (arm id, seed) that end their unit as an
    ``unusable_solve`` (no table row); ``replace``: (group, seed, replacement) replacement records.  Par2 arms write rows with ``seed =
    nan`` and the realization index ``r = seed - seed0`` (consecutive seeds per segment).  ``extra_rows(src, seg, seed, window, base_row)``
    may return more rows (other family / weights).  Returns the paths.
    """
    os.makedirs(os.path.join(root, "campaign"), exist_ok=True)
    pp = os.path.join(root, "prereg.json")
    extra_keys = {}
    if qualification is not None:  # the robust sign needs a qualification document (sf43ana.qualification); "permissive" qualifies every fixture size
        qdoc = qualification_doc() if qualification == "permissive" else qualification
        qpath = os.path.join(root, "qualification.json")
        with open(qpath, "w") as f:
            json.dump(qdoc, f)
        extra_keys["qualification"] = qpath
    with open(pp, "w") as f:
        json.dump(dict(prereg, windows=WINDOWS, **extra_keys), f)
    fail, unreduced = set(fail), set(unreduced)
    ledger, rows, sources = [], {}, {}
    for a in prereg["arms"]:
        sources.setdefault(a.get("source", a["id"]), []).append(a)
    unit = 0
    for src, group_arms in sources.items():
        code = group_arms[0]["code"]
        seeds = sorted({s for a in group_arms for s in list(a["seeds"]) + list(a.get("reserves", []))})
        for k0 in range(segments):
            seg_seeds = [s for k, s in enumerate(seeds) if (k * segments) // len(seeds) == k0]
            if not seg_seeds:
                continue
            unit += 1
            seg = "u%06d_s%d" % (unit, seg_seeds[0])
            ok = [s for s in seg_seeds if (src, s) not in fail]
            ledger.append({"kind": "unit_end", "unit": unit, "arm": src, "cls": "ok", "completed": ok, "failed_seed": None,
                           "segment": "segments/%s/%s" % (src, seg), "segment_seed0": seg_seeds[0],
                           "verdicts": {str(s): "converged" for s in ok}})
            for s in ok:
                ledger.append(seed_record(src, group_arms[0]["group"], "segments/%s/%s" % (src, seg), s, seg_seeds[0], unit, "converged", "ok"))
            skip_seg = any((src, s) in unreduced for s in seg_seeds)
            if skip_seg:
                ledger.append({"kind": "unreduced", "arm": src, "segment": "segments/%s/%s" % (src, seg), "seeds": ok})
            for s in seg_seeds:
                if (src, s) in fail:
                    unit += 1
                    ledger.append({"kind": "unit_end", "unit": unit, "arm": src, "cls": "unusable_solve", "completed": [], "failed_seed": s,
                                   "segment": "segments/%s/u%06d_s%d" % (src, unit, s), "segment_seed0": s, "verdicts": {str(s): "stagnated"}})
                    ledger.append(seed_record(src, group_arms[0]["group"], "segments/%s/u%06d_s%d" % (src, unit, s), s, s, unit, "stagnated",
                                              "unusable_solve"))
                    continue
                if skip_seg:
                    continue
                for a in group_arms:
                    if a["code"] == "pollock" or a.get("selection") or s not in set(a["seeds"]) | set(a.get("reserves", [])):
                        continue  # a Pollock entry or an entry with its own row selection (uniform weights) reads rows of the table that ``extra_rows`` adds
                    for wname, (lo, hi) in WINDOWS.items():
                        for basis in ("pos", "disp"):
                            mu = lambda est: mean_of(a, wname, basis, est)  # noqa: E731
                            row = {"code": code, "family": "plane", "weights": "flux" if code == "consumer" else "unif", "set": "perplane",
                                   "basis": basis, "hmax_div": a["hmax_div"], "seed": s if code == "consumer" else float("nan"),
                                   "r": s - seg_seeds[0], "d_lo": lo, "d_hi": hi, "ell": ell,
                                   "alpha_TP": _val(a["id"] + "TP", s, mu("alpha_TP"), sd[0], sd[1], a["group"]),
                                   "alpha_TAP": _val(a["id"] + "TAP", s, mu("alpha_TAP"), sd[0], sd[1], a["group"] + "x"),
                                   "alpha_LP": _val(a["id"] + "LP", s, mu("alpha_LP"), 0.01, 0.002, a["group"] + "y"),
                                   "U_r": 1.0 + 0.01 * _val(a["id"] + "U", s, 0.0, 1.0, 0.0, a["group"] + "u"), "note": ""}
                            rows.setdefault((src, seg, wname), []).append(row)
                            for extra in (extra_rows(src, seg, s, wname, row) if extra_rows else []):
                                rows[(src, seg, wname)].append(extra)
    for (src, seg, wname), rs in rows.items():
        lo, hi = WINDOWS[wname]
        path = table_path(root, src, seg, lo, hi)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        extra = [c for c in red.TRACKER_COLUMNS if any(c in r for r in rs)]  # the opt-in tracker columns (sf42red reduce --tracker-counts)
        red.write_table(path, rs, red.COLUMNS + extra)
    for g, s, t in replace:
        ledger.append({"kind": "replacement", "group": g, "seed": s, "replaced_by": t, "arm": "?"})
    for arm_id, n_new in (reest or {}).items():
        ledger.append({"kind": "reest", "arm": arm_id, "sd": {}, "n_obs": 0, "n_req": n_new, "n_old": 0, "n_new": n_new})
    with open(os.path.join(root, "campaign", "ledger.jsonl"), "w") as f:
        for r in ledger:
            f.write(json.dumps(r) + "\n")
    return {"prereg": pp, "campaign": os.path.join(root, "campaign"), "reduced": os.path.join(root, "reduced")}
