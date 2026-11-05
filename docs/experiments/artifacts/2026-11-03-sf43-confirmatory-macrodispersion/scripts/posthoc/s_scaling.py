"""post hoc (2026-11-04) S: scaling of ``alpha_TP`` (displacement basis, flux weights, main window [25, 94], production arms) with sigma^2.

``alpha_TP / sigma^4`` at 0.04 (positive control ``pc_s004``, N 30), 0.25, 1, 2.25, 4 (``s*_prod``, N 200 = analysis set of the frozen run), with
90 % Student intervals (t on N-1 dof) and the local log-log exponents ``d ln alpha_TP / d ln sigma^2`` between consecutive amplitudes
with delta-method standard errors from the arm standard errors (independent arms): se(exponent) = sqrt((se1/m1)^2 + (se0/m0)^2) / ln(s2_1/s2_0);
90 % interval = exponent -/+ 1.645 se (normal).  A realization-bootstrap percentile interval (B 20000, seed 20261104) is given alongside.
Descriptive; sigma^2 in the lognormal field is the pre-registered amplitude (sigma_Y^2).
"""
import os
import sys

import numpy as np
from scipy import stats

from common import LABEL, dump, load, outdir

ARMS = (("pc_s004", 0.04), ("s025_prod", 0.25), ("s1_prod", 1.0), ("s225_prod", 2.25), ("s4_prod", 4.0))


def main():
    pr, arms, ledger = load(sys.argv[1])
    rng = np.random.default_rng(np.random.SeedSequence([20261104, 99]))
    rows, boots = [], {}
    for aid, s2 in ARMS:
        x = arms[aid].get("main", "disp", "alpha_TP")
        n = x.size
        m, se = float(x.mean()), float(x.std(ddof=1) / np.sqrt(n))
        q = stats.t.ppf(0.95, n - 1)
        boots[aid] = x[rng.integers(0, n, size=(20000, n))].mean(1)
        rows.append({"arm": aid, "sigma2": s2, "N": int(n), "alpha_TP": m, "se": se, "student_ci90": [m - q * se, m + q * se],
                     "alpha_TP_over_sigma4": m / s2 ** 2, "over_sigma4_ci90": [(m - q * se) / s2 ** 2, (m + q * se) / s2 ** 2],
                     "alpha_TP_over_sigma2": m / s2})
    ex = []
    for a, b in zip(rows[:-1], rows[1:]):
        L = np.log(b["sigma2"] / a["sigma2"])
        e = np.log(b["alpha_TP"] / a["alpha_TP"]) / L
        se = np.sqrt((b["se"] / b["alpha_TP"]) ** 2 + (a["se"] / a["alpha_TP"]) ** 2) / L
        eb = np.log(boots[b["arm"]] / boots[a["arm"]]) / L
        ex.append({"from": a["arm"], "to": b["arm"], "sigma2": [a["sigma2"], b["sigma2"]], "exponent": float(e), "se_delta": float(se),
                   "ci90_delta": [float(e - 1.645 * se), float(e + 1.645 * se)], "ci90_bootstrap": [float(x) for x in np.quantile(eb, [0.05, 0.95])]})
    dump(os.path.join(outdir("scaling"), "scaling.json"), {"label": LABEL, "rows": rows, "exponents": ex})
    for r in rows:
        print("%-10s s2=%-5g N=%3d alpha_TP %.4e  /s4: %.3e  [%.3e, %.3e]" % (r["arm"], r["sigma2"], r["N"], r["alpha_TP"], r["alpha_TP_over_sigma4"], *r["over_sigma4_ci90"]))
    for e in ex:
        print("exponent %s->%s %.3f  se %.3f  delta [%.3f, %.3f]  boot [%.3f, %.3f]" % (e["from"], e["to"], e["exponent"], e["se_delta"], *e["ci90_delta"], *e["ci90_bootstrap"]))


if __name__ == "__main__":
    main()
