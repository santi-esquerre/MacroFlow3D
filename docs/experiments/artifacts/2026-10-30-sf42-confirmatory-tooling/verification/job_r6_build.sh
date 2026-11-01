#!/usr/bin/env bash
# SF-42 R6 job sf42-r6-build (CPU): clean release build of the corrected tree, then the md5 of the two binaries the
# regress / bytes / identity jobs use.  Run from the mirror root (job cwd).  Same recipe as R1's sf42-head-build; the temp
# directory and the parallelism are those of R4's build (the default /tmp of the host is small).
set -u
date -u +%FT%TZ
echo "TREE=$(cat .mf3d_source_rev)"
export TMPDIR=$HOME/.cache/mf3d-tmp CMAKE_BUILD_PARALLEL_LEVEL=16
mkdir -p "$TMPDIR"
rm -rf build/v100-release
cmake --preset v100-release > /dev/null || { echo "CONFIGURE_FAILED"; exit 1; }
cmake --build build/v100-release -j || { echo "BUILD_FAILED"; exit 1; }
md5sum build/v100-release/macroflow3d_pipeline build/v100-release/lagrangian_ensemble
date -u +%FT%TZ
