"""``check``: structure check of the smoke (S0) campaign against ``prereg_addendum.json["smoke"]``.  COUNTS and flags only, never an alpha.

Per smoke arm: the seeds the smoke planned, how many are completed in the ledger (class ok, usable verdict), how many form the loader's complete
prefix, how many have finite estimators in every window the arm reduces (main; also far for the consumer arms j3_l5, j9_w48) in both bases, and for
Par2 arms how many carry finite crossing counts (``n_last``, ``n_traced > 0``).  ``STRUCTURE_OK`` iff every count equals the planned one.
"""
import json
import os
import sys

import numpy as np

from sf43ana import config as cfgmod
from sf43ana.config import ConfigError
from sf43ana.loader import COLS, LoadError, TableCache, load_arm

from .data import arm_cfgs, load_addendum, load_ledger, resolve


def check_arm(P, cfg, seeds, ledger, tables):
    cfg = dict(cfg, seeds=list(seeds), reserves=[], n_conf=len(seeds), n_max=len(seeds), floor=1)
    a = load_arm(cfg, P, ledger, tables)
    ok = set(P["ok_verdicts"])
    in_ledger = sum(1 for s in seeds if (cfg["source"], s) in ledger.seeds and ledger.seeds[(cfg["source"], s)]["class"] == "ok"
                    and ledger.seeds[(cfg["source"], s)]["verdict"] in ok | {None})
    windows = ["main"] + (["far"] if cfg["code"] == "consumer" else [])
    keys = [(w, b, c) for w in windows for b in ("pos", "disp") for c in COLS]
    fin = np.ones(a.N, dtype=bool)
    for k in keys:
        fin &= np.isfinite(a.values[k])
    out = {"arm": cfg["id"], "planned": len(seeds), "ledger_completed": in_ledger, "loader_prefix": a.N, "finite_rows": int(fin.sum()),
           "windows": windows, "stopped_at": a.info["stopped_at"]}
    if cfg["code"] == "par2":
        nl, nt = a.extra.get("n_last"), a.extra.get("n_traced")
        out["crossing_counts"] = 0 if nl is None or nt is None else int((np.isfinite(nl) & np.isfinite(nt) & (nt > 0)).sum())
    ok_all = all(out[k] == len(seeds) for k in ("ledger_completed", "loader_prefix", "finite_rows")) and out.get("crossing_counts", len(seeds)) == len(seeds)
    return dict(out, structure_ok=bool(ok_all))


def main(a):
    try:
        add = load_addendum(a.prereg)
        P = cfgmod.load_prereg(resolve(add["frozen_prereg"], a.prereg))
        ledger = load_ledger(a.campaign)
        tables = TableCache(os.path.join(a.campaign, "reduced"), P, ledger)
        cfgs = arm_cfgs(add)
        res = [check_arm(P, cfgs[arm], seeds, ledger, tables) for arm, seeds in add["smoke"]["units"].items()]
    except (LoadError, ConfigError, OSError, KeyError, ValueError) as e:
        print("sf43add.ana check: input refused: %s" % type(e).__name__, file=sys.stderr)
        return 2
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "smoke_check.json"), "w") as fh:
        json.dump({"label": "structure and counts only", "arms": res}, fh, indent=1, sort_keys=True)
        fh.write("\n")
    for r in res:
        print("arm %s: planned=%d ledger_completed=%d loader_prefix=%d finite_rows=%d%s" % (
            r["arm"], r["planned"], r["ledger_completed"], r["loader_prefix"], r["finite_rows"],
            " crossing_counts=%d" % r["crossing_counts"] if "crossing_counts" in r else ""))
    bad = [r["arm"] for r in res if not r["structure_ok"]]
    print("STRUCTURE_OK" if not bad else "STRUCTURE_FAIL arms: %s" % ", ".join(bad))
    return 0 if not bad else 1
