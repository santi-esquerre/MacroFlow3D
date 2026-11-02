"""Failure reporting of SF-43 (specification, Failure rules): the failure criteria are fixed in advance and applied blind to ``alpha``.

Per arm, from the campaign ledger (JSON lines, one record per segment of a seed's solve):

* failure rate = failed seeds / attempted seeds, with the Clopper-Pearson 95 % two-sided interval (exact beta limits; 0 / 1 at the ends);
* replacements disclosed: a record with ``replaces`` = the failed seed it replaces (the next reserved seed); failed seeds never replaced
  are listed too;
* if the failure rate of an arm exceeds ``max_rate`` (1 %): worst-case imputation.  The failed realizations (all of them, replaced or
  not: a failure that depends on the field would otherwise be invisible) are added as ``n_failed`` extra values equal to the 5th (low
  case) or the 95th (high case) percentile of the arm's usable per-seed values (linear-interpolation quantiles), for every estimand
  column; the mean of the augmented sample, and its one-sided 95 % Student lower (low case) / upper (high case) bound, are reported;
* sensitivity excluding fallback-accepted solves (ledger ``verdict`` in ``fallback_verdicts``, default ``fallback``; solver_b.json
  verdict): the mean and the 90 % Student interval of every estimand with those seeds removed, against all usable seeds;
* halted arms (ledger ``class == "halted"`` or ``data["halted"]``) listed: a halted arm stops and goes to the owner.

``run(prereg, data)`` -- ``prereg["failures"]`` (all optional)::

    max_rate (0.01), failure_classes (["failed"]), fallback_verdicts (["fallback"]), estimands (["alpha_TP", "alpha_TAP", "alpha_LP"]),
    q_low (0.05), q_high (0.95), arms (default: the arms of the ledger)

``data``::

    {"ledger": [{"arm", "segment", "seed", "verdict", "class", ["replaces"]}, ...],
     "arms": {arm: {"seed": int array, "cols": {name: float array}}},  "halted": [arm, ...]}

A record counts as failed iff its ``class`` is in ``failure_classes``.  The usable per-seed values of an arm are the finite entries of
``data["arms"][arm]`` (the loader has already excluded failed solves); an arm absent from ``data["arms"]`` is reported without
estimand blocks.
"""
from __future__ import annotations

import numpy as np
from scipy import stats as _st

from .gate5 import jsonable


def clopper_pearson(k, n, conf=0.95):
    """Exact two-sided Clopper-Pearson interval of a binomial proportion ``k / n`` (``(nan, nan)`` for ``n = 0``)."""
    if n <= 0:
        return float("nan"), float("nan")
    a = 0.5 * (1.0 - conf)
    lo = 0.0 if k == 0 else float(_st.beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(_st.beta.ppf(1.0 - a, k + 1, n - k))
    return lo, hi


def _student(x, level):
    n = x.size
    if n < 2:
        return {"n": int(n), "mean": float(x.mean()) if n else float("nan"), "lo": float("nan"), "hi": float("nan")}
    se = x.std(ddof=1) / np.sqrt(n)
    q = float(_st.t.isf(0.5 * (1.0 - level), n - 1))
    return {"n": int(n), "mean": float(x.mean()), "lo": float(x.mean() - q * se), "hi": float(x.mean() + q * se)}


def impute_worst_case(values, n_failed, q_low=0.05, q_high=0.95):
    """Mean of ``values`` augmented by ``n_failed`` copies of its ``q_low`` (low case) or ``q_high`` quantile, with the one-sided 95 %
    Student lower (low case) / upper (high case) bound of the augmented mean."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return {"note": "fewer than 2 usable values"}
    out = {"n_obs": int(v.size), "n_imputed": int(n_failed), "mean_obs": float(v.mean())}
    for tag, q in (("low", q_low), ("high", q_high)):
        fill = float(np.quantile(v, q))
        aug = np.concatenate([v, np.full(n_failed, fill)])
        se = aug.std(ddof=1) / np.sqrt(aug.size)
        t = float(_st.t.isf(0.05, aug.size - 1))
        out[tag] = {"quantile": q, "fill": fill, "mean": float(aug.mean()),
                    "bound95": float(aug.mean() - t * se) if tag == "low" else float(aug.mean() + t * se)}
    return out


def run(prereg, data):
    cfg = dict(prereg.get("failures", {}))
    max_rate = float(cfg.get("max_rate", 0.01))
    fclasses = set(cfg.get("failure_classes", ["failed"]))
    fbverd = set(cfg.get("fallback_verdicts", ["fallback"]))
    ests = list(cfg.get("estimands", ["alpha_TP", "alpha_TAP", "alpha_LP"]))
    ql, qh = float(cfg.get("q_low", 0.05)), float(cfg.get("q_high", 0.95))
    ledger = list(data.get("ledger", []))
    arms = list(cfg.get("arms", sorted({r["arm"] for r in ledger})))
    halted = sorted(set(data.get("halted", [])) | {r["arm"] for r in ledger if r.get("class") == "halted"})
    out = {"config": {"max_rate": max_rate, "failure_classes": sorted(fclasses), "fallback_verdicts": sorted(fbverd), "estimands": ests,
                      "q_low": ql, "q_high": qh},
           "arms": {}, "halted": halted, "total": {"attempted": 0, "failed": 0}}
    for arm in arms:
        recs = [r for r in ledger if r["arm"] == arm and r.get("class") != "halted"]
        attempted = sorted({int(r["seed"]) for r in recs})
        failed = sorted({int(r["seed"]) for r in recs if r.get("class") in fclasses})
        fallback = sorted({int(r["seed"]) for r in recs if r.get("verdict") in fbverd and r.get("class") not in fclasses} - set(failed))
        repl = sorted((int(r["replaces"]), int(r["seed"])) for r in recs if r.get("replaces") is not None)
        replaced = {a for a, _ in repl}
        n, k = len(attempted), len(failed)
        lo, hi = clopper_pearson(k, n)
        res = {"attempted": n, "failed": k, "rate": k / n if n else float("nan"), "cp95": [lo, hi], "failed_seeds": failed,
               "replacements": [{"failed_seed": a, "replacement_seed": b} for a, b in repl],
               "failed_not_replaced": [s for s in failed if s not in replaced],
               "replacements_without_failure": [a for a in sorted(replaced) if a not in failed],
               "exceeds_max_rate": bool(n and k / n > max_rate), "fallback_accepted_seeds": fallback, "halted": arm in halted}
        out["total"]["attempted"] += n
        out["total"]["failed"] += k
        if arm in data.get("arms", {}):
            a = data["arms"][arm]
            seed = np.asarray(a["seed"])
            res["estimands"] = {}
            for e in ests:
                if e not in a["cols"]:
                    continue
                v = np.asarray(a["cols"][e], dtype=float)
                use = np.isfinite(v) & ~np.isin(seed, failed)
                block = {"all": _student(v[use], 0.90)}
                keep = use & ~np.isin(seed, fallback)
                block["excluding_fallback"] = dict(_student(v[keep], 0.90), n_excluded=int((use & ~keep).sum()),
                                                   mean_shift=float(v[keep].mean() - v[use].mean()) if keep.any() and use.any() else float("nan"))
                if res["exceeds_max_rate"] and k > 0:
                    block["worst_case_imputation"] = impute_worst_case(v[use], k, ql, qh)
                res["estimands"][e] = block
        out["arms"][arm] = res
    n, k = out["total"]["attempted"], out["total"]["failed"]
    out["total"].update(rate=k / n if n else float("nan"), cp95=list(clopper_pearson(k, n)))
    return jsonable(out)
