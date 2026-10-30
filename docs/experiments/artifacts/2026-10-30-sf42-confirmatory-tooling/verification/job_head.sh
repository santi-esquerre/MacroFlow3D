#!/usr/bin/env bash
# SF-42 R1 job sf42-head (acceptance A): scripts/regress set v2 on the head build, compared with the reference built from the
# increment base 3164e4fb (job sf42-base, ~/regress/refs/3164e4f/out). Run from the mirror root.
set -u
REV=$(cat .mf3d_source_rev)
H=$HOME/regress/heads/SF-42-$REV
date -u +START_UTC=%FT%TZ
echo "HEAD_TREE=$REV"
mkdir -p "$H"
REGRESS_SOURCE_TREE=$REV bash scripts/regress/run.sh build/v100-release "$H/out"
echo "run_rc=$?"
bash scripts/regress/compare.sh "$HOME/regress/refs/3164e4f/out" "$H/out" > "$H/compare.txt" 2>&1
echo "compare_rc=$?"
tail -n 60 "$H/compare.txt"
mkdir -p "$HOME/sf42_campaign/regress"
cp "$H/compare.txt" "$HOME/sf42_campaign/regress/compare.txt"
cp "$H/out/binaries.md5" "$HOME/sf42_campaign/regress/head_binaries.md5"
cp "$HOME/regress/refs/3164e4f/out/binaries.md5" "$HOME/sf42_campaign/regress/base_binaries.md5"
cp "$H/out/exit_codes.tsv" "$HOME/sf42_campaign/regress/head_exit_codes.tsv"
cp "$HOME/regress/refs/3164e4f/out/exit_codes.tsv" "$HOME/sf42_campaign/regress/base_exit_codes.tsv"
cp "$H/out/env.txt" "$HOME/sf42_campaign/regress/head_env.txt"
cp "$HOME/regress/refs/3164e4f/out/env.txt" "$HOME/sf42_campaign/regress/base_env.txt"
cat "$H/out/binaries.md5"
date -u +END_UTC=%FT%TZ
