"""post hoc (2026-11-04) N1: tables of the G7 factorial effects (ratio = exp(log effect), 90 % interval, unadjusted p) and of the twin
decomposition from posthoc/n1/summary.json (output of the frozen sf43ana with prereg_posthoc_g7.json).  Writes posthoc/n1/g7_effects.csv
and g7_twins.csv.  Basis: responses ``X:disp`` are the displacement basis (primary of the pre-registration), ``X`` without suffix the
position basis (the basis the frozen run used for G7, deviation D8)."""
import csv
import json
import os

from common import OUT


def main():
    with open(os.path.join(OUT, "n1", "summary.json")) as f:
        g7 = json.load(f)["g7"]
    rows = []
    for amp, resps in g7["factorial"].items():
        for resp, r in resps.items():
            if "log_effects" not in r:
                rows.append([amp, resp, "", "", "", "", "", r.get("note", "")])
                continue
            for eff in ("G", "I", "GI", "D", "R", "GD", "GR"):
                le, e = r["log_effects"][eff], r["effects"][eff]
                import math
                rows.append([amp, resp, eff, "%.6g" % le["ratio"], "%.6g" % math.exp(le["lo"]), "%.6g" % math.exp(le["hi"]), "%.6g" % e["p"],
                             "alias %s; %s; nu %.1f; n_seeds %s/%s" % (le["alias"], e["stratum"], e["nu"], r["seeds"]["1"]["n_seeds"], r["seeds"]["-1"]["n_seeds"])])
    with open(os.path.join(OUT, "n1", "g7_effects.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma2_arm", "response", "effect", "ratio", "ci90_lo", "ci90_hi", "p_unadjusted", "note"])
        w.writerows(rows)
    tw = []
    for amp, resps in g7["twins"].items():
        for resp, pairs in resps.items():
            for name, c in pairs.items():
                if "ratio" in c:
                    tw.append([amp, resp, name, "%.6g" % c["ratio"], "%.6g" % c["ratio_lo"], "%.6g" % c["ratio_hi"], c["n"]])
    with open(os.path.join(OUT, "n1", "g7_twins.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sigma2_arm", "response", "contrast", "ratio", "ci90_lo", "ci90_hi", "n_pairs"])
        w.writerows(tw)
    for r in rows:
        if r[2] in ("G", "I", "GI") or not r[2]:
            print(*r[:7])
    print("-- twins")
    for r in tw:
        print(*r)


if __name__ == "__main__":
    main()
