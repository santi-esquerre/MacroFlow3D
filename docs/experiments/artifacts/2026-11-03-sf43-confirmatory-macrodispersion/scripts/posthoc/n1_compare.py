"""post hoc (2026-11-04) N1: compare the summary.json of the post hoc G7 run (frozen sf43ana, prereg_posthoc_g7.json) with the committed one,
block by block, and extract the G, I, GI effects of the factorial.

    python3 n1_compare.py SCRATCH

Writes posthoc/n1/{summary.json (copy of the new run), compare.json, g7_effects.json}.  Numeric leaves are compared with a relative
tolerance (|a-b| / max(|a|,|b|,tiny)); the reported number is the max over the leaves of a block; Gate 5 Monte-Carlo power entries are reported
separately (tolerance 1e-4).  ``process`` and the ``g7`` block are excluded from the equality verdict (g7 gets its own check: the
position-basis ``alpha_TP`` entries must equal the committed ones).
"""
import json
import math
import os
import shutil
import sys

from common import ART, LABEL, dump, outdir


def leaves(o, path=()):
    if isinstance(o, dict):
        for k, v in o.items():
            yield from leaves(v, path + (k,))
    elif isinstance(o, list):
        for i, v in enumerate(o):
            yield from leaves(v, path + (i,))
    else:
        yield path, o


def num(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if x in ("nan", "inf", "-inf"):
        return float(x)
    return None


def compare(a, b):
    """-> (max rel dev, worst path, n numeric leaves, n structural mismatches (type/str/keys), examples)"""
    la, lb = dict(leaves(a)), dict(leaves(b))
    worst, wp, n, mism, ex = 0.0, None, 0, 0, []
    for k in set(la) | set(lb):
        if k not in la or k not in lb:
            mism += 1
            ex.append(("missing", k))
            continue
        x, y = la[k], lb[k]
        nx, ny = num(x), num(y)
        if nx is not None and ny is not None:
            n += 1
            if math.isnan(nx) and math.isnan(ny) or nx == ny:
                continue
            d = abs(nx - ny) / max(abs(nx), abs(ny), 1e-300)
            if d > worst:
                worst, wp = d, k
        elif x != y:
            mism += 1
            if len(ex) < 8:
                ex.append((k, x, y))
    return worst, wp, n, mism, ex


def main():
    scratch = sys.argv[1]
    new_p = os.path.join(scratch, "camp", "analysis", "summary.json")
    d = outdir("n1")
    shutil.copy(new_p, os.path.join(d, "summary.json"))
    with open(new_p) as f:
        new = json.load(f)
    with open(os.path.join(ART, "results", "analysis", "summary.json")) as f:
        old = json.load(f)
    res = {"label": LABEL, "blocks": {}}
    for k in sorted(set(new) | set(old)):
        if k in ("process", "g7"):
            continue
        w, wp, n, mism, ex = compare(new.get(k), old.get(k))
        entry = {"max_rel_dev": w, "worst_path": list(map(str, wp)) if wp else None, "numeric_leaves": n, "non_numeric_mismatches": mism,
                 "examples": [list(map(str, e)) for e in ex[:5]]}
        if k == "gate5":  # Monte-Carlo power entries separately
            pw = {p: v for p, v in dict(leaves(new[k])).items() if any(isinstance(s, str) and "power" in s for s in p)}
            pwo = {p: v for p, v in dict(leaves(old[k])).items() if any(isinstance(s, str) and "power" in s for s in p)}
            dev_p = max((abs(pw[p] - pwo[p]) / max(abs(pw[p]), abs(pwo[p]), 1e-300) for p in pw if p in pwo and num(pw[p]) is not None and num(pwo[p]) is not None), default=0.0)
            non = {p: v for p, v in dict(leaves(new[k])).items() if p not in pw}
            nono = {p: v for p, v in dict(leaves(old[k])).items() if p not in pwo}
            w2, wp2, n2, m2, ex2 = compare({str(p): v for p, v in non.items()}, {str(p): v for p, v in nono.items()})
            entry["power_entries"] = {"n": len(pw), "max_rel_dev": dev_p}
            entry["excluding_power_entries"] = {"max_rel_dev": w2, "numeric_leaves": n2, "non_numeric_mismatches": m2}
        res["blocks"][k] = entry
    # process: only provenance may differ; report the differing top-level keys
    pn, po = new["process"], old["process"]
    res["process_keys_different"] = sorted(k for k in set(pn) | set(po) if pn.get(k) != po.get(k))
    res["process_prereg_sha256"] = {"new": pn.get("prereg_sha256"), "committed": po.get("prereg_sha256")}
    # g7: position-basis alpha_TP entries equal the committed ones
    g = {"twins": {}, "factorial": {}}
    for part in ("twins", "factorial"):
        for amp in old["g7"][part]:
            for resp in old["g7"][part][amp]:
                if resp not in new["g7"][part][amp]:
                    g[part]["%s/%s" % (amp, resp)] = "missing in new"
                    continue
                w, wp, n, mism, ex = compare(new["g7"][part][amp][resp], old["g7"][part][amp][resp])
                g[part]["%s/%s" % (amp, resp)] = {"max_rel_dev": w, "numeric_leaves": n, "non_numeric_mismatches": mism}
    res["g7_vs_committed_for_responses_present_in_committed"] = g
    dump(os.path.join(d, "compare.json"), res)
    for k, e in res["blocks"].items():
        print("%-18s max rel dev %.3e  leaves %d  non-numeric mismatches %d" % (k, e["max_rel_dev"], e["numeric_leaves"], e["non_numeric_mismatches"]),
              e.get("power_entries", ""), e.get("excluding_power_entries", ""))
    print("process keys different:", res["process_keys_different"])
    print(json.dumps(g, indent=0)[:1500])

    # G, I, GI effects
    eff = {"label": LABEL, "config": new["g7"]["config"], "factorial": {}}
    for amp in new["g7"]["factorial"]:
        for resp, r in new["g7"]["factorial"][amp].items():
            eff["factorial"].setdefault(amp, {})[resp] = r
    dump(os.path.join(d, "g7_effects_full.json"), eff)
    print(json.dumps(sorted(next(iter(next(iter(eff["factorial"].values())).values())).keys())))


if __name__ == "__main__":
    main()
