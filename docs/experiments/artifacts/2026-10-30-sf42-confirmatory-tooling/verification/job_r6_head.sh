#!/usr/bin/env bash
# SF-42 R6 job sf42-r6-head (acceptance A re-verification of the corrected tree): scripts/regress set v2 on the R6 build,
# compared with the reference of the increment base 3164e4fb (~/regress/refs/3164e4f/out, reused, never rebuilt here).
# Same as job_head.sh except the heads directory (~/regress/heads/SF-42-r6-<rev>) and the copies (~/sf42_campaign/r6/regress),
# so R1's outputs stay intact.  Run from the mirror root.
set -u
REV=$(cat .mf3d_source_rev)
H=$HOME/regress/heads/SF-42-r6-$REV
D=$HOME/sf42_campaign/r6/regress
REF=$HOME/regress/refs/3164e4f/out
date -u +START_UTC=%FT%TZ
echo "HEAD_TREE=$REV"
[ -f "$REF/REGRESS_DONE" ] || { echo "REFERENCE_MISSING $REF"; exit 1; }
mkdir -p "$H" "$D"
REGRESS_SOURCE_TREE=$REV bash scripts/regress/run.sh build/v100-release "$H/out"
echo "run_rc=$?"
bash scripts/regress/compare.sh "$REF" "$H/out" > "$H/compare.txt" 2>&1
echo "compare_rc=$?"
tail -n 60 "$H/compare.txt"
cp "$H/compare.txt" "$D/compare.txt"
cp "$H/out/binaries.md5" "$D/head_binaries.md5"
cp "$REF/binaries.md5" "$D/base_binaries.md5"
cp "$H/out/exit_codes.tsv" "$D/head_exit_codes.tsv"
cp "$REF/exit_codes.tsv" "$D/base_exit_codes.tsv"
cp "$H/out/env.txt" "$D/head_env.txt"
cp "$REF/env.txt" "$D/base_env.txt"
cat "$H/out/binaries.md5"
date -u +END_UTC=%FT%TZ
