#!/usr/bin/env bash
# SF-42 R1 job sf42-base: copy the base binaries aside, then the regress reference of the base 3164e4fb.
# Run from the mirror root (job cwd).
date -u +%FT%TZ
mkdir -p "$HOME/sf42_campaign/base_bin"
cp -p build/v100-release/macroflow3d_pipeline build/v100-release/lagrangian_ensemble "$HOME/sf42_campaign/base_bin/"
(cd "$HOME/sf42_campaign/base_bin" && md5sum macroflow3d_pipeline lagrangian_ensemble | tee md5.txt)
if [ -f "$HOME/regress/refs/3164e4f/out/REGRESS_DONE" ]; then
    echo "REFERENCE_REUSED $HOME/regress/refs/3164e4f/out"
else
    REGRESS_SOURCE_TREE=3164e4fb bash scripts/regress/run.sh build/v100-release "$HOME/regress/refs/3164e4f/out"
    echo "run_rc=$?"
fi
cat "$HOME/regress/refs/3164e4f/out/REGRESS_DONE"
date -u +%FT%TZ
