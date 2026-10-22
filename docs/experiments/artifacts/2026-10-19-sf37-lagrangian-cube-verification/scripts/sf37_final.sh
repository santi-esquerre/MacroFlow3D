#!/usr/bin/env bash
# SF-37 FINAL_AUDIT job sf37-final: full V100 ctest + SF-27 trio byte-compare vs the base build (~/sf37_base_refs/trio).
# bash 4.2-safe. Passed verbatim as the command string of `scripts/remote --increment SF-37 run sf37-final` (preceded by
# `export SF37_FINAL_COMMIT=<hash>`), so the host keeps the exact text in commands/sf37-final.command. Runs on the SF-37
# mirror synced from a clean worktree of the source-bearing tree it reports (H1; later commits are docs/evidence only).
set -o pipefail
M=~/MacroFlow3D-SF-37
cd "$M" || exit 2
echo "FINAL_TREE_COMMIT=${SF37_FINAL_COMMIT:-unknown}"
echo "GPU=${CUDA_VISIBLE_DEVICES:-unset} $(nvidia-smi --query-gpu=name --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES:-0}")"
cmake --preset v100-release > /dev/null && cmake --build build/v100-release -j 2>&1 | tail -n 5
echo "BUILD_EXIT=$?"
md5sum build/v100-release/inlet_slab build/v100-release/macroflow3d_pipeline
T0=$(date +%s)
ctest --test-dir build/v100-release --output-on-failure
echo "CTEST_EXIT=$? CTEST_WALL_S=$(( $(date +%s) - T0 ))"
R=~/sf37_head_refs
rm -rf "$R"; mkdir -p "$R"; cd "$R" || exit 2
for c in config_pspta_small config_streamfunctions_homogeneous config_streamfunctions_continuation; do
    "$M/build/v100-release/macroflow3d_pipeline" "$M/apps/$c.yaml" > "$c.log" 2>&1
    echo "$c exit=$?"
done
for o in output_pspta_small output_streamfunctions_homogeneous output_streamfunctions_continuation; do
    echo "== $o"
    diff -rq -x manifest.json ~/sf37_base_refs/trio/$o "$R/$o" && echo "BYTECMP_IDENTICAL $o (excluding manifest.json)"
    diff <(grep -v -E 'time|date|timestamp|elapsed|duration|_s"|seconds' ~/sf37_base_refs/trio/$o/manifest.json) <(grep -v -E 'time|date|timestamp|elapsed|duration|_s"|seconds' "$R/$o/manifest.json") && echo "MANIFEST_IDENTICAL_MODULO_TIMES $o"
done
echo FINAL_JOB_END
