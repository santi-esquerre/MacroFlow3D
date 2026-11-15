"""Standard figures of SF-43 (H18), drawn from ``summary.json`` ONLY and only if ``matplotlib`` imports; otherwise ``FIGURES_SKIPPED``
is logged.  ``make`` never raises: any failure is logged as ``FIGURES_ERROR`` and the analysis is unaffected.

Figures (PNG, ``Agg`` backend), one file each:
  ``estimates.png``   production mean of ``alpha_TP`` and ``alpha_TAP`` (displacement basis) per amplitude with the 90 % bootstrap-t interval
                      and the pre-registered threshold ``c``;
  ``budget.png``      the error-budget components at the smallest Holm level next to the production mean (``alpha_TP``, primary basis);
  ``gate5.png``       the Fieller ratio interval per amplitude with the margins ``1/m`` and ``m``;
  ``positive_control.png``  measured / predicted second-moment profile of the ``sigma^2 = 0.04`` arm with its seed-bootstrap band.
"""
import json
import os
import sys


def _log(msg):
    print(msg, file=sys.stderr, flush=True)


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def make(summary_path, out_dir):
    """-> list of written files ([] if skipped).  Never raises."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        _log("FIGURES_SKIPPED matplotlib is not importable (%s)" % type(e).__name__)
        return []
    try:
        with open(summary_path) as f:
            S = json.load(f)
        os.makedirs(out_dir, exist_ok=True)
        written = []

        def save(fig, name):
            p = os.path.join(out_dir, name)
            fig.savefig(p, dpi=130, bbox_inches="tight")
            plt.close(fig)
            written.append(p)

        amps = sorted(S.get("estimands", {}), key=_f)
        fig, ax = plt.subplots(figsize=(6, 4))
        for j, e in enumerate(("alpha_TP", "alpha_TAP")):
            xs, ys, lo, hi = [], [], [], []
            for i, a in enumerate(amps):
                s = S["estimands"][a].get("estimators", {}).get(e, {})
                d = next((v for v in s.values() if v.get("role") == "primary"), None) if s else None
                if d and d.get("bootstrap_t"):
                    xs.append(i + 0.1 * (j - 0.5)), ys.append(_f(d["mean"])), lo.append(_f(d["mean"]) - _f(d["bootstrap_t"][0])), hi.append(_f(d["bootstrap_t"][1]) - _f(d["mean"]))
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o", label=e, capsize=3)
        for i, a in enumerate(amps):
            c = S.get("gate4", {}).get(a, {}).get("null_value")
            if c is not None:
                ax.plot([i - 0.25, i + 0.25], [_f(c)] * 2, "k--", lw=1, label="threshold c" if i == 0 else None)
        ax.set_xticks(range(len(amps)))
        ax.set_xticklabels(["sigma^2 = %s" % a for a in amps])
        ax.set_ylabel("alpha (lambda)")
        ax.set_yscale("symlog", linthresh=1e-5)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()
        save(fig, "estimates.png")

        fig, ax = plt.subplots(figsize=(6, 4))
        for i, a in enumerate(amps):
            ent = S.get("budgets", {}).get(a, {})
            pb = next(iter(ent), None)
            t = ent.get(pb, {}).get("alpha_TP") if pb else None
            if not t or not t.get("testable"):
                continue
            k = 0
            for name, c in t["components"].items():
                v = c["enforced"]
                ax.bar(i + 0.15 * k - 0.2, _f(v[0] if isinstance(v, list) else v), width=0.14, label=name if i == 0 else None)
                k += 1
        ax.set_xticks(range(len(amps)))
        ax.set_xticklabels(["sigma^2 = %s" % a for a in amps])
        ax.set_ylabel("enforced bound at the first reported level (lambda)")
        if ax.get_legend_handles_labels()[0]:
            ax.set_yscale("log")
            ax.legend()
        save(fig, "budget.png")

        g5 = S.get("gate5", {}).get("amplitudes", {})
        if g5:
            fig, ax = plt.subplots(figsize=(6, 4))
            m = _f(S["gate5"].get("config", {}).get("margin_ratio", 1.10))
            for i, (a, r) in enumerate(sorted(g5.items(), key=lambda kv: _f(kv[0]))):
                ci = r.get("ci")
                if ci:
                    ax.errorbar([i], [_f(r.get("ratio"))], yerr=[[_f(r["ratio"]) - _f(ci["lo"])], [_f(ci["hi"]) - _f(r["ratio"])]], fmt="o", capsize=3)
            ax.axhline(m, color="k", ls="--", lw=1)
            ax.axhline(1.0 / m, color="k", ls="--", lw=1)
            ax.set_xticks(range(len(g5)))
            ax.set_xticklabels(["sigma^2 = %s" % a for a in sorted(g5, key=_f)])
            ax.set_ylabel("alpha_LP ratio (consumer / Par2)")
            save(fig, "gate5.png")

        pc = S.get("positive_control", {}).get("weights", {})
        if pc:
            fig, ax = plt.subplots(figsize=(6, 4))
            for w, r in pc.items():
                sm = r.get("second_moment")
                if sm:
                    ax.plot(r["d"], sm["ratio"], label="%s second moment" % w)
                    ax.fill_between(r["d"], sm["ratio_lo"], sm["ratio_hi"], alpha=0.25)
            ax.axhline(1.0, color="k", lw=1)
            ax.set_xlabel("d (lambda)")
            ax.set_ylabel("measured / first-order box prediction")
            ax.legend()
            save(fig, "positive_control.png")
        _log("FIGURES_WRITTEN %d" % len(written))
        return written
    except Exception as e:  # noqa: BLE001
        _log("FIGURES_ERROR %s: %s" % (type(e).__name__, e))
        return []
