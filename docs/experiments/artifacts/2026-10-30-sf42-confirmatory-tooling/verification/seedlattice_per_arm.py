#!/usr/bin/env python3
"""SF-42 R1, acceptance C: admissible lattice sizes M of every SF-43 planning arm taken ALONE (RR12 by seedlattice.rule).

  cd <R>/scripts && PYTHONPATH=. python3 ../verification/seedlattice_per_arm.py

Columns: sigma^2, family, nx, ny, window, number of admissible M in [1, 4096], number in [47, 250] (the search range around
SF-39's seed density). The common-M-per-amplitude result is in scripts/seedlattice/data/seedlattice_plan.md.
"""
import json
import os

from seedlattice import rule

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, "..", "scripts", "seedlattice", "data", "seedlattice_plan.json")) as f:
    plan = json.load(f)
none = 0
print("sigma2 family nx ny window n_admissible[1,4096] n_admissible[47,250]")
for a in plan["arms"]:
    adm = rule.admissible(a["ny"], a["window"], range(1, rule.M_MAX + 1))
    near = [m for m in adm if 47 <= m <= 250]
    none += 0 if near else 1
    print(a["amp"], a["family"], a["nx"], a["ny"], a["window"], len(adm), len(near))
print("arms with no admissible M in [47, 250]: %d of %d" % (none, len(plan["arms"])))
