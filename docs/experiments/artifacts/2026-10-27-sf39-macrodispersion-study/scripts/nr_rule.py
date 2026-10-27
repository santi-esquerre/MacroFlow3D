#!/usr/bin/env python3
"""SF-39 ``nr_rule.py`` (contract C-NR; preregistration section 6).  Prints ``N`` values only: never a mean, sd, margin or verdict.

Usage::

    python3 nr_rule.py pilot.json --out nr.json
    python3 nr_rule.py pilot.json --out nr.json --budget-gpu-h X --costs costs.json [--done done.json]

``pilot.json`` (produced by the checkpoint node)::

    {"s1": {"alpha_TAM": {...}, "alpha_TP": {...}}, "s225": {...}, "s4": {...}}

each estimator object holding the pilot arrays (length n = 20, one entry per pilot realization, units of lambda) and scalars:

    D_f      narrow 2x rung minus production rung, paired by seed
    D_tw     production minus coarse twin, paired by seed
    d16_8    h/16 pass minus h/8 pass (production geometry, directory A)
    prod     production-geometry (h/8) pilot values
    mean_2x  mean of the 2x-rung pilot values (scalar)
    mean_h16 mean of the h/16-pass pilot values (scalar)

Output: per amplitude N_narrow, N_twin, N_hmax, N_R, N_A, N_run, par2_NR (``>500`` where the cap of 500 was reached without meeting
the target; 500 is then used).  ``--out`` JSON: {amp: {N_*: int, ..., "capped": [names]}}.  With ``--budget-gpu-h`` the budget rule
(section 6) is applied: ``costs.json`` = {arm: GPU-s per realization} for every arm of preregistration section 1 and each amplitude
(``<c>_A``, ``<c>_B`` (distinct), narrow rungs ``s1_n05`` ..., ``<c>_tw``, ``<c>_par2``, ``s4_w24``, ``s4_w12``); ``done.json`` =
{arm: realizations already run} (default: the pilot, 20 for every A, narrow, twin and Par2 arm; 0 for B and width arms).  Prints the
final N, the common factor f, whether the s4 ladders went to 20, projected GPU-h, and ``owner=yes`` if even f=0.01 does not fit.
"""
import argparse
import json
import sys

from sf39 import power

KEYS = ("N_narrow", "N_twin", "N_hmax", "N_R", "N_A", "N_run", "par2_NR")


def default_done():
    done = {}
    for amp in power.AMPLITUDES:
        for arm in (f"{amp}_A", f"{amp}_tw", f"{amp}_par2", *power.NARROW_ARMS[amp]):
            done[arm] = power.N_MIN
    return done


def _emit(amp, n, capped):
    for k in KEYS:
        print(f"{amp} {k}={'>500' if k in capped else n[k]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("pilot")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget-gpu-h", type=float)
    ap.add_argument("--costs")
    ap.add_argument("--done")
    a = ap.parse_args(argv)
    if (a.budget_gpu_h is None) != (a.costs is None):
        ap.error("--budget-gpu-h and --costs go together")
    with open(a.pilot) as f:
        pilot = json.load(f)
    ns, capped = {}, {}
    for amp in power.AMPLITUDES:
        ns[amp], capped[amp] = power.amplitude_ns(pilot[amp])
    out = {amp: {**ns[amp], "capped": capped[amp]} for amp in ns}
    if a.budget_gpu_h is None:
        for amp in ns:
            _emit(amp, ns[amp], capped[amp])
    else:
        with open(a.costs) as f:
            costs = json.load(f)
        done = default_done()
        if a.done:
            with open(a.done) as f:
                done = json.load(f)
        r = power.allocate(ns, costs, done, a.budget_gpu_h)
        print(f"budget_gpu_h={a.budget_gpu_h:.2f} projected_gpu_h_unscaled={r['projected_gpu_h_unscaled']:.2f} "
              f"s4_ladders_to_20={'yes' if r['s4_ladders_to_20'] else 'no'} f={r['f']:.2f} "
              f"projected_gpu_h={r['projected_gpu_h']:.2f} owner={'yes' if r['owner'] else 'no'}")
        for amp in r["N"]:
            _emit(amp, r["N"][amp], ())
        out = {"N": {amp: {**r["N"][amp], "capped_before_budget": capped[amp]} for amp in r["N"]},
               **{k: v for k, v in r.items() if k != "N"}}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
