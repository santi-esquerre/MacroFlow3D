"""Local analysis of the SF-42 R4 pilots (design inputs; no interpretation of the sign or size of any alpha).

Functions (each pure, tested with synthetic tables):

* ``component_rows``       particle component / split-half share per (amplitude, M, estimator, basis) from reduced J2 tables.
* ``project_M``            smallest admissible M whose projected particle share is at most a threshold (fixed M^-2 law anchored on
                           the pilot's M values; the fitted-exponent law is reported beside it).
* ``pairing``              Gate 5 pairing statistics from reduced J3 tables (consumer ``unif`` against Par2, ``alpha_LP``).
* ``cost_points``          calibration points of the cost model from ``RUN.json`` records.
* ``scale_n``              variant-B realization counts.
"""
import csv
import math

import numpy as np

from mcstat import gate5 as g5
from mcstat import sizing
from sf42red import reduce as red


def read_table(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def select(rows, **kw):
    """Rows whose columns equal the given strings (values compared as str)."""
    return [r for r in rows if all(str(r[k]) == str(v) for k, v in kw.items())]


# ----------------------------------------------------------------------------------------------------------------------
# J2: particle component
# ----------------------------------------------------------------------------------------------------------------------
EST = ("alpha_TP", "alpha_TAP", "alpha_LP")


def component_rows(rows, est, basis):
    """Split-half statistics of the consumer production rows (weights flux, set perplane, plane family) of one reduced table.

    -> dict(n, component, variance, fraction, flag, mean) ; finite values only."""
    rs = [r for r in rows if r["code"] == "consumer" and r["family"] == "plane" and r["weights"] == "flux" and r["set"] == "perplane"
          and r["basis"] == basis]
    vals = [(float(r[est]), float(r[est + "_A"]), float(r[est + "_B"])) for r in rs]
    vals = [v for v in vals if all(math.isfinite(x) for x in v)]
    if len(vals) < 2:
        raise ValueError("fewer than two finite realizations for %s/%s" % (est, basis))
    a, A, B = zip(*vals)
    out = red.splithalf_fraction(A, B, a)
    out["mean"] = float(np.mean(a))
    return out


def fit_loglog(M, comp):
    """Least-squares line log c = a + b log M -> (b, se_b, a); se_b from the residuals (nan with fewer than 3 points)."""
    x, y = np.log(np.asarray(M, float)), np.log(np.asarray(comp, float))
    b, a = np.polyfit(x, y, 1)
    if len(x) < 3:
        return float(b), float("nan"), float(a)
    res = y - (a + b * x)
    se = math.sqrt(float(np.sum(res ** 2)) / (len(x) - 2) / float(np.sum((x - x.mean()) ** 2)))
    return float(b), se, float(a)


def project_M(Ms, comps, V, adm, thresholds=(0.25, 0.10)):
    """Smallest admissible M with projected component / V <= threshold.

    ``Ms``/``comps``: the pilot's M values and measured components; ``V``: planning across-realization variance; ``adm``: sorted
    admissible M of the SF-43 grid and window.  Law A (primary): comp = k M^-2, ``log k`` = mean(log c_i + 2 log M_i).  Law B (reported):
    the fitted line.  Returns {threshold: {"M_fixed": ..., "M_fit": ..., "share_at": ...}} and the exponent fit."""
    Ms, comps = np.asarray(Ms, float), np.asarray(comps, float)
    k = math.exp(float(np.mean(np.log(comps) + 2.0 * np.log(Ms))))
    b, se, a = fit_loglog(Ms, comps)
    out = {}
    for thr in thresholds:
        rec = {}
        for name, f in (("M_fixed", lambda m: k / (m * m)), ("M_fit", lambda m: math.exp(a) * m ** b)):
            rec[name] = next((int(m) for m in adm if f(float(m)) / V <= thr), None)
        out[thr] = rec
    return out, {"exponent": b, "se": se, "k_fixed": k}


# ----------------------------------------------------------------------------------------------------------------------
# J3: Gate 5 pairing
# ----------------------------------------------------------------------------------------------------------------------
def alpha_by_r(rows, code, weights, est="alpha_LP"):
    """{r: value} of the plane-family rows of a code (basis pos, set perplane); finite values only."""
    out = {}
    for r in rows:
        if r["code"] == code and r["family"] == "plane" and r["weights"] == weights and r["basis"] == "pos" and r["set"] == "perplane":
            v = float(r[est])
            if math.isfinite(v):
                out[int(r["r"])] = v
    return out


def pairing(cons_rows, par2_rows, alpha_holm=0.05 / 7.0, power=0.8):
    """Gate 5 pilot statistics of the paired alpha_LP (consumer ``unif`` against Par2) and the planning n (G5.5)."""
    L, P = alpha_by_r(cons_rows, "consumer", "unif"), alpha_by_r(par2_rows, "par2", "unif")
    rs = sorted(set(L) & set(P))
    x, y = np.array([L[r] for r in rs]), np.array([P[r] for r in rs])
    n = len(rs)
    out = {"n_pairs": n, "n_consumer": len(L), "n_par2": len(P), "mean_L": float(x.mean()), "mean_P": float(y.mean()),
           "sd_L": float(x.std(ddof=1)), "sd_P": float(y.std(ddof=1))}
    out["cv_L"], out["cv_P"] = out["sd_L"] / out["mean_L"], out["sd_P"] / out["mean_P"]
    if n <= 3:
        raise ValueError("a pairing pilot needs more than 3 pairs (got %d)" % n)
    r = float(np.corrcoef(x, y)[0, 1])
    out["corr"] = r
    out["corr_lower80"] = float(sizing.corr_lower(r, n))
    out["ratio_of_means"] = out["mean_L"] / out["mean_P"]
    for label, corr in (("lower", r), ("zero", None)):
        try:
            cvL, cvP, rl = g5.gate5_conservative_inputs(out["sd_L"], out["sd_P"], out["mean_L"], out["mean_P"], n, corr=corr)
            nn, ok = g5.gate5_sample_size(cvL, cvP, rl, alpha_holm, power=power, ratio=1.0)
            out["plan_" + label] = {"cv_L": cvL, "cv_P": cvP, "corr_used": rl, "n": int(nn), "found": bool(ok)}
        except ValueError as e:  # deflated mean not positive
            out["plan_" + label] = {"error": str(e)}
    return out


# ----------------------------------------------------------------------------------------------------------------------
# cost
# ----------------------------------------------------------------------------------------------------------------------
def scale_n(n, sd_pos, sd_disp, floor=30):
    """Variant B: n (SD_disp / SD_pos)^2, at least ``floor`` (never above the original count)."""
    return min(n, max(floor, int(math.ceil(n * (sd_disp / sd_pos) ** 2))))
