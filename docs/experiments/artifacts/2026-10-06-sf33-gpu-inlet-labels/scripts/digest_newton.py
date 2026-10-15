#!/usr/bin/env python3
"""SF-33 N7b: per-stage Newton / GMRES digest of `inlet_slab --summary` JSON files (no interpretation).

For every JSON: the case status and PATH; per continuation stage: Newton status, steps, the merit (r_F) history,
the forcing terms eta_k, the Psi-tc shifts mu_k (`^` marks a line-search retry of the same step), GMRES total /
max / median iterations per linear solve, and for every failed linear solve the per-restart true relative
residuals.  Works on pre-N7b JSON (no mu fields: printed as `-`).

Usage: python3 digest_newton.py JSON [JSON ...] > digest.txt
"""
import json
import statistics
import sys


def fmt_list(v, f="%.2e"):
    return "[" + ",".join(f % x for x in v) + "]" if v else "[]"


def stage_line(st):
    n = st["newton"]
    steps = n.get("steps", [])
    solves = []
    for s in steps:
        solves.extend(s.get("lin_its_solves") or [s.get("lin_its", 0)])
    total = sum(solves)
    mx = max(solves) if solves else 0
    med = statistics.median(solves) if solves else float("nan")
    mus = []
    for s in steps:
        if "mu_ser" not in s:
            break
        t = "%.2e" % s["mu_ser"]
        for r in s.get("mu_retries") or []:
            t += "^%.2e" % r
        mus.append(t)
    head = "  stage %-9g%-8s from %-9g %-16s newton_steps=%2d solves=%2d gmres_total=%6d max=%5d median=%7.1f" % (
        st["eps"], "(final)" if st.get("final_attempt") else "", st.get("from_eps", 0.0), n["status"], len(steps),
        len(solves), total, mx, med)
    out = [head]
    out.append("      r_F  " + fmt_list(n.get("hist_r_F", []), "%.2e"))
    out.append("      eta  " + fmt_list([s["eta"] for s in steps]))
    out.append("      mu   [" + ",".join(mus) + "]" if mus else "      mu   -")
    for k, s in enumerate(steps):
        if s.get("lin_status") != "converged":
            out.append("      FAILED linear solve step %d: status=%s its=%d cycles=%d eta=%.2e mu=%s rel=%.2e; "
                       "per-restart true rel: %s" % (
                           k + 1, s.get("lin_status"), s.get("lin_its", 0), s.get("lin_cycles", 0), s["eta"],
                           "%.2e" % s["mu"] if "mu" in s else "-", s.get("lin_rel", float("nan")),
                           " ".join("%.2e" % x for x in s.get("lin_cycle_true", []))))
    return out, len(solves), total


def main(paths):
    for p in paths:
        with open(p) as f:
            j = json.load(f)
        c = j.get("continuation", {})
        cfg = j.get("solver_config", {})
        print("== %s  status=%s path=%s" % (p.split("/")[-1], j.get("status"), c.get("path")))
        print("   solver: forcing=%s psitc=%s mu0=%s mu_max=%s max_newton=%s" % (
            cfg.get("forcing"), cfg.get("psitc", "-"), cfg.get("psitc_mu0", "-"), cfg.get("psitc_mu_max", "-"),
            cfg.get("max_newton")))
        ns = nt = nsteps = 0
        for st in c.get("stages", []):
            lines, s, t = stage_line(st)
            ns += s
            nt += t
            nsteps += len(st["newton"].get("steps", []))
            print("\n".join(lines))
        print("  TOTAL newton steps %d, linear solves %d, GMRES its %d" % (nsteps, ns, nt))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
