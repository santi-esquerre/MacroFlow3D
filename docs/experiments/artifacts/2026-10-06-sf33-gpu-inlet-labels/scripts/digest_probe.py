#!/usr/bin/env python3
"""SF-33 N7c: table of the linear-probe JSON summaries (inlet_slab --linear-probe --summary).

Usage: digest_probe.py <json>... [--curves]
One row per (case, stage, k, mu, preconditioner): GMRES iterations, status, final true relative
residual, coarse K / rcond estimate / assembly + LU time / mean application time; with --curves the
per-restart true relative residuals follow each row.
"""
import json
import sys


def main(argv):
    curves = "--curves" in argv
    files = [a for a in argv if not a.startswith("--")]
    hdr = ("case", "stage", "from", "k", "mu_kind", "mu", "prec", "its", "status", "rel",
           "K", "rcond_est", "t_asm", "t_lu", "t_app_ms", "t_s")
    print(" ".join(f"{h:>10}" if i else f"{h:<18}" for i, h in enumerate(hdr)))
    for fn in files:
        with open(fn) as f:
            J = json.load(f)
        if "results" not in J:
            print(f"{fn}: no results ({J.get('probe_abort', J.get('status'))})")
            continue
        it = J.get("iterate", {})
        print(f"# {fn.rsplit('/', 1)[-1]}: {J['case']} stage={J['stage']} from={J['from']} k={J['k']} "
              f"r_F={it.get('r_F')} merit={it.get('merit')} mu_SER={it.get('mu_ser')}")
        for r in J["results"]:
            c = r.get("coarse", {})
            row = (J["case"], f"{J['stage']:g}", f"{J['from']:g}", str(J["k"]), r["mu_kind"],
                   f"{r['mu']:.2e}", r["prec"], str(r["its"]), r["status"], f"{r['rel']:.2e}",
                   str(c.get("K", "-")),
                   f"{c['rcond_est']:.2e}" if c else "-",
                   f"{c['t_assembly']:.2f}" if c else "-",
                   f"{c['t_lu']:.2f}" if c else "-",
                   f"{1e3 * r['t_apply_avg']:.2f}" if "t_apply_avg" in r else "-",
                   f"{r['seconds']:.1f}")
            print(" ".join(f"{v:>10}" if i else f"{v:<18}" for i, v in enumerate(row)))
            if curves:
                print("      curve: " + " ".join(f"{v:.2e}" for v in r["cycle_true"]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
