export SF37_COMMIT=d866c40875cc52b3bdef7c7c9921b00b504a2350
A=docs/experiments/artifacts/2026-10-19-sf37-lagrangian-cube-verification/scripts
mkdir -p $HOME/sf37_campaign/jobs
for p in v3a v3b v1 digest; do
    echo "=== PART_BEGIN $p $(date -u +%FT%TZ)"
    bash $A/run_sf37.sh build/v100-release $p 2>&1 | tee $HOME/sf37_campaign/jobs/sf37-science-$p.log
    rc=${PIPESTATUS[0]}
    echo "=== PART_EXIT $p $rc $(date -u +%FT%TZ)"
done
