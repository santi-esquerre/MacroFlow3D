"""post hoc (2026-11-04) N8: Par2 crossing-completeness strata at sigma^2 = 4 (SELECTION, NOT CAUSAL).

Pairs = the Gate 5 pairs of the frozen pre-registration at sigma^2 = 4 on ``alpha_LP`` (column ``alpha_LP``, uniform weights, main window
[25, 94]): Pollock twin ``s4_g5pol`` (source s4_l16) against Par2 ``s4_g5p``, paired by seed (``sf43ana.gate5.pair_by_seed``).  Each pair is
assigned to a stratum by the number of Par2 particles that did not cross the last window plane, ``n_traced - n_last`` (columns of the
reduced Par2 plane row): 0; 1-2; 3-9; >= 10.  Per stratum: n, ratio of means Pollock/Par2 (paired delta-method Student 90 % interval of
``mcstat.gate5.gate5_paired`` and a bootstrap-t one, B 2000, seed 20261104), mean completeness c = n_last / n_traced; over all pairs corr(log(Q_r/P_r),
1 - c_r) (Pearson and Spearman).  The all-pairs ratio reproduces the frozen decomposition (1.164 [1.131, 1.198]).
"""
import json
import os
import sys

import numpy as np
from scipy import stats

from common import ART, LABEL, dump, load, outdir
from mcstat import gate5 as mg5
from sf43ana.gate5 import arm_series, pair_by_seed
from sf43ana.loader import build_data

STRATA = (("0", 0, 0), ("1-2", 1, 2), ("3-9", 3, 9), (">=10", 10, 10 ** 9))


def main():
    pr, arms, ledger = load(sys.argv[1])
    data = build_data(pr, arms, ledger)
    sq, vq = arm_series(data, "s4_g5pol", "alpha_LP")
    sp, vp = arm_series(data, "s4_g5p", "alpha_LP")
    seeds, Q, P, info = pair_by_seed(sq, vq, sp, vp)
    _, nl = arm_series(data, "s4_g5p", "n_last")
    _, nt = arm_series(data, "s4_g5p", "n_traced")
    idx = {int(s): i for i, s in enumerate(data["arms"]["s4_g5p"]["seed"])}
    nl = np.array([nl[idx[int(s)]] for s in seeds])
    nt = np.array([nt[idx[int(s)]] for s in seeds])
    miss = nt - nl
    c = nl / nt
    out = {"label": LABEL, "note": "selection by an outcome-correlated Par2 property, not causal", "pairing": info, "n_pairs": int(Q.size),
           "interval_methods": "gate5_paired Student (delta method) 90 %, bootstrap-t B 2000 seed 20261104", "strata": {},
           "missing_particles_summary": {"min": float(miss.min()), "median": float(np.median(miss)), "max": float(miss.max()),
                                         "n_traced_min": float(nt.min()), "n_traced_max": float(nt.max())}}
    allr = mg5.gate5_paired(Q, P, level=0.90)
    out["all_pairs"] = {"ratio": float(allr.ratio), "lo": float(allr.ratio_lo), "hi": float(allr.ratio_hi), "n": int(Q.size)}
    for k, (name, lo, hi) in enumerate(STRATA):
        m = (miss >= lo) & (miss <= hi)
        n = int(m.sum())
        e = {"n": n, "mean_completeness": float(c[m].mean()) if n else None, "mean_missing": float(miss[m].mean()) if n else None}
        if n >= 3:
            r = mg5.gate5_paired(Q[m], P[m], level=0.90)
            b = mg5.gate5_paired(Q[m], P[m], level=0.90, method="bootstrap-t", B=2000, rng=np.random.default_rng(np.random.SeedSequence([20261104, 8, k])))
            e.update(ratio=float(r.ratio), student=[float(r.ratio_lo), float(r.ratio_hi)], bootstrap_t=[float(b.ratio_lo), float(b.ratio_hi)],
                     mean_pollock=float(Q[m].mean()), mean_par2=float(P[m].mean()))
        out["strata"][name] = e
    lr = np.log(Q / P)
    rp, pp = stats.pearsonr(lr, 1.0 - c)
    rs, ps = stats.spearmanr(lr, 1.0 - c)
    out["corr_log_ratio_vs_1_minus_c"] = {"pearson": float(rp), "pearson_p": float(pp), "spearman": float(rs), "spearman_p": float(ps), "n": int(lr.size)}
    with open(os.path.join(ART, "results", "analysis", "summary.json")) as f:
        fz = json.load(f)["g7"]["twins"]["s4"]["alpha_LP"]["pollock/par2"]
    out["reproduction"] = {"frozen_pollock_over_par2": [fz["ratio"], fz["ratio_lo"], fz["ratio_hi"]],
                           "here": [out["all_pairs"]["ratio"], out["all_pairs"]["lo"], out["all_pairs"]["hi"]]}
    dump(os.path.join(outdir("n8"), "n8_par2_completeness.json"), out)
    print("pairs", out["n_pairs"], "all", out["all_pairs"], "frozen", out["reproduction"]["frozen_pollock_over_par2"])
    for k, e in out["strata"].items():
        print(k, e)
    print(out["corr_log_ratio_vs_1_minus_c"], out["missing_particles_summary"])


if __name__ == "__main__":
    main()
