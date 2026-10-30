#!/usr/bin/env bash
# R5 detached host job (CPU only): host-stack unit tests of the statistics tree, then the full coverage simulation.
#   usage (from the mirror root): bash <this> <workers>
# Same command as the R5 brief, wrapped in a file because the agent sandbox rejects `cd` in `scripts/remote run` text.
set -u
W="${1:?workers}"
R=docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling
mkdir -p ~/sf42_campaign
(cd $R/scripts && OMP_NUM_THREADS=1 MCSTAT_PAR2_ALPHAX_S025=~/sf42_campaign/r3/par2_alphax_s025.txt python3 -W error -m unittest discover -s . -t . > ~/sf42_campaign/r5_hosttests.log 2>&1)
rc=$?
tail -4 ~/sf42_campaign/r5_hosttests.log
test $rc -eq 0 && bash $R/scripts/mcstat/run_coverage.sh ~/sf42_campaign/coverage --profile full --workers "$W"
