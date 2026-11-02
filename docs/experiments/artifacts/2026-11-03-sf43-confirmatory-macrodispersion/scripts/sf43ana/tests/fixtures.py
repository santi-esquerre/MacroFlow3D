"""Synthetic campaign (pre-registration, ledger, reduced tables in the real `sf42red` format) for the loader / end-to-end tests."""
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
        arm("l4", "2.25", "g2", s2, n, 20, r2), arm("l8", "2.25", "g2", s2, n, 20, r2),
        arm("p2", "2.25", "g2", s2, n, 20, r2, role="production"), arm("p2h10", "2.25", "g2", s2, n, 20, r2),
        arm("w2f", "2.25", "gw2", w2, n, 20, range(4900, 4910)), arm("w2c", "2.25", "gw2", w2, n, 20, range(4900, 4910)),
        arm("par2_1", "1", "gp", range(7000, 7010), 10, 5, code="par2"),
        arm("w2far", "2.25", "gd", wc2, 30, 15),
    ]
    amps = {"1": {"production": "p1", "components": ["E_h", "E_W", "E_int"],
                  "ladder": {"arms": ["l5", "p1", "l20"], "resolutions": [5, 10, 20], "production_index": 1},
                  "width": {"wide_fine": "w48", "prod_fine": "p1"}, "hmax": {"h16": "h16", "h8": "h8"}},
            "2.25": {"production": "p2", "components": ["E_h", "E_W", "E_WxH"],
                     "ladder": {"arms": ["l4", "l8", "p2"], "resolutions": [4, 8, 16], "production_index": 2},
                     "width": {"wide_fine": "w2f", "prod_fine": "p2h10", "wide_coarse": "w2c", "prod_coarse": "l8"},
                     "distance_arm": "w2far"}}
    return {"amplitudes": amps, "arms": arms, "family": ["robust:1", "robust:2.25", "gate5:1", "gate5:2.25"],
            "bootstrap": {"B_sampling": 400, "B_budget": 300, "B_ucl": 400, "B_bca": 200, "seed": 11},
            "targets": {"1": {"alpha_TP": 1e-3}},
            "first_order": {"1": {"alpha_TP": 0.006}}}


def _val(aid, seed, mu, common_sd, own_sd, group):
    z = np.random.default_rng([zlib.crc32(group.encode()), seed]).standard_normal()
    e = np.random.default_rng([zlib.crc32(aid.encode()), seed, 7]).standard_normal()
    return mu + common_sd * z + own_sd * e


def write_world(root, prereg, mean_of, fail=(), replace=(), sd=(0.002, 0.0005), segments=2):
    """Writes ``prereg.json``, ``campaign/ledger.jsonl`` and ``reduced/<arm source>/<segment>/<window>.csv``.

    ``mean_of(arm_cfg, window, basis, est)`` -> mean of the estimator; ``fail``: iterable of (arm id, seed) given verdict 'bad';
    ``replace``: iterable of (group, seed, replaced_by) ledger replacement records.  Returns the paths.
    """
    os.makedirs(os.path.join(root, "campaign"), exist_ok=True)
    pp = os.path.join(root, "prereg.json")
    with open(pp, "w") as f:
        json.dump(dict(prereg, windows=WINDOWS), f)
    fail = set(fail)
    ledger, rows = [], {}
    sources = {}
    for a in prereg["arms"]:
        if a["code"] != "consumer":
            continue
        src = a.get("source", a["id"])
        sources.setdefault(src, []).append(a)
    for src, group_arms in sources.items():
        seeds = sorted({s for a in group_arms for s in list(a["seeds"]) + list(a.get("reserves", []))})
        base = next(a for a in group_arms if a["id"] == src) if any(a["id"] == src for a in group_arms) else group_arms[0]
        for k, s in enumerate(seeds):
            seg = "s%d" % (1 + (k * segments) // len(seeds))
            ledger.append({"arm": src, "segment": seg, "seed": s, "verdict": "bad" if (src, s) in fail else "ok", "class": "x"})
            for a in group_arms:
                if s not in set(a["seeds"]) | set(a.get("reserves", [])):
                    continue
                for wname, (lo, hi) in WINDOWS.items():
                    for basis in ("pos", "disp"):
                        mu = lambda est: mean_of(a, wname, basis, est)  # noqa: E731
                        row = {"code": "consumer", "family": "plane", "weights": "flux", "set": "perplane", "basis": basis,
                               "hmax_div": a["hmax_div"], "seed": s, "r": k, "d_lo": lo, "d_hi": hi,
                               "alpha_TP": _val(a["id"] + "TP", s, mu("alpha_TP"), sd[0], sd[1], a["group"]),
                               "alpha_TAP": _val(a["id"] + "TAP", s, mu("alpha_TAP"), sd[0], sd[1], a["group"] + "x"),
                               "alpha_LP": _val(a["id"] + "LP", s, mu("alpha_LP"), 0.01, 0.002, a["group"] + "y"),
                               "U_r": 1.0 + 0.01 * _val(a["id"] + "U", s, 0.0, 1.0, 0.0, a["group"] + "u"), "note": ""}
                        rows.setdefault((src, seg, wname), []).append(row)
    for (src, seg, wname), rs in rows.items():
        d = os.path.join(root, "reduced", src, seg)
        os.makedirs(d, exist_ok=True)
        red.write_table(os.path.join(d, wname + ".csv"), rs, red.COLUMNS)
    for g, s, t in replace:
        ledger.append({"kind": "replacement", "seed": s, "replaced_by": t, "group": g})
    lp = os.path.join(root, "campaign", "ledger.jsonl")
    with open(lp, "w") as f:
        for r in ledger:
            f.write(json.dumps(r) + "\n")
    return {"prereg": pp, "campaign": os.path.join(root, "campaign"), "reduced": os.path.join(root, "reduced")}
