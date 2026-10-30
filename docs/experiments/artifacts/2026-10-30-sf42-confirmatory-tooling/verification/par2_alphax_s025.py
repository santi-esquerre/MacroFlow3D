"""Per-realization Par2 alpha_x of SF-39's s025 Gate 5 set, read-only from the SF-39 host tree with the SF-39 package's own functions.

Mirrors `sf39.analyze.gate5_block` (`ents = par2_entries(C, amp)[:n2]` ... `P = np.array([a["x"] for a in al])`): pooled pilot + extension
realizations (`sf39.par2.pooled`), U_c at the frozen t_b,min with anchor inject_x (`uc_mean`), window [0.75, 1] tau_c lam / U_c (`window`),
alpha_x per realization (`alpha_window`).  The frozen values are those of `tau_b_star_s025.json`.  Usage:
    python3 par2_alphax_s025.py <sf39 out root> <sf39 artifact> <output txt>
"""
import json
import os
import sys

sys.dont_write_bytecode = True
out, art, dest = sys.argv[1:4]
sys.path.insert(0, os.path.join(art, "scripts"))
from sf39 import par2  # noqa: E402

with open(os.path.join(art, "raw/campaign_s025/checkpoint/s025/tau_b_star_s025.json")) as f:
    g5 = json.load(f)["amplitudes"]["s025"]["gate5"]
with open(os.path.join(art, "analysis/summary.json")) as f:
    n2 = int(json.load(f)["amplitudes"]["s025"]["C"]["par2"]["alpha_x"]["n"])
if os.path.realpath(dest).startswith(os.path.realpath(out) + os.sep):
    raise SystemExit("output inside the read-only SF-39 tree")
raw = os.path.join(out, "par2", "raw", "par2_s025")
ext = raw + "_ext"
ents = par2.pooled(raw, ext if os.path.isdir(ext) else None, n_pilot=20)[:n2]
if len(ents) < n2:
    raise SystemExit("only %d realizations, need %d" % (len(ents), n2))
series = [par2.series_of(e) for e in ents]
p2 = g5["par2"]
uc, _ = par2.uc_mean(series, p2["t_b_min"], p2["inject_x"])
t_lo, t_hi = par2.window(float(g5["tau_c"]), p2["corr_length"], uc)
P = [par2.alpha_window(s, p2["corr_length"], uc, t_lo, t_hi)["x"] for s in series]
with open(dest, "w") as f:
    f.write("\n".join("%.17g" % v for v in P) + "\n")
print("PAR2_ALPHAX_S025 n=%d U_c=%.17g window=[%.10g, %.10g] mean=%.17g" % (len(P), uc, t_lo, t_hi, sum(P) / len(P)))
