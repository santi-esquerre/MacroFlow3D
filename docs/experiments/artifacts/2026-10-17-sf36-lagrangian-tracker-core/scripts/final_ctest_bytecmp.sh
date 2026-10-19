#!/usr/bin/env bash
# SF-36 FINAL_AUDIT job: full ctest + SF-27 trio byte-compare vs the base build (~/sf36_base_refs/trio).
set -o pipefail
M=~/MacroFlow3D-SF-36
cd "$M" || exit 2
echo "FINAL_TREE_COMMIT=$(cat "$M/.sf36_final_commit")"
cmake --preset v100-release > /dev/null && cmake --build build/v100-release -j 2>&1 | tail -n 5
echo "BUILD_EXIT=$?"
ctest --test-dir build/v100-release --output-on-failure
echo "CTEST_EXIT=$?"
R=~/sf36_head_refs
rm -rf "$R"; mkdir -p "$R"; cd "$R" || exit 2
for c in config_pspta_small config_streamfunctions_homogeneous config_streamfunctions_continuation; do
    "$M/build/v100-release/macroflow3d_pipeline" "$M/apps/$c.yaml" > "$c.log" 2>&1
    echo "$c exit=$?"
done
for o in output_pspta_small output_streamfunctions_homogeneous output_streamfunctions_continuation; do
    echo "== $o"
    diff -rq -x manifest.json ~/sf36_base_refs/trio/$o "$R/$o" && echo "BYTECMP_IDENTICAL $o (excluding manifest.json)"
    diff <(grep -v -E 'time|date|timestamp|elapsed|duration|_s"|seconds' ~/sf36_base_refs/trio/$o/manifest.json) <(grep -v -E 'time|date|timestamp|elapsed|duration|_s"|seconds' "$R/$o/manifest.json") && echo "MANIFEST_IDENTICAL_MODULO_TIMES $o"
done
echo FINAL_JOB_END
