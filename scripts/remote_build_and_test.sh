#!/usr/bin/env bash
set -euo pipefail

# Configure, build, and test on the remote V100 server.
#
# Defaults:
#   REMOTE_HOST=v100
#   REMOTE_DIR=~/MacroFlow3D
#   BUILD_DIR=build/v100-release
#
# Example:
#   scripts/remote_build_and_test.sh
#   BUILD_DIR=build/v100-petsc ENABLE_PETSC=ON scripts/remote_build_and_test.sh

REMOTE_HOST="${REMOTE_HOST:-v100}"
REMOTE_DIR="${REMOTE_DIR:-~/MacroFlow3D}"
BUILD_DIR="${BUILD_DIR:-build/v100-release}"
ENABLE_PETSC="${ENABLE_PETSC:-OFF}"

if [[ "$ENABLE_PETSC" == "ON" ]]; then
  EXTRA_CMAKE='
    -DMACROFLOW3D_ENABLE_PETSC=ON \
    -DPETSC_DIR=$HOME/MacroFlow3D/src/external/petsc \
    -DPETSC_ARCH=arch-cuda \
    -DSLEPC_DIR=$HOME/MacroFlow3D/src/external/slepc'
else
  EXTRA_CMAKE='-DMACROFLOW3D_ENABLE_PETSC=OFF'
fi

ssh "$REMOTE_HOST" "
  set -euo pipefail
  cd $REMOTE_DIR

  cmake -S . -B $BUILD_DIR -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=70 \
    -DMACROFLOW3D_ENABLE_DIAGNOSTICS=OFF \
    -DMACROFLOW3D_ENABLE_PROFILING=ON \
    -DMACROFLOW3D_ENABLE_NVTX=ON \
    $EXTRA_CMAKE

  cmake --build $BUILD_DIR -j
  ctest --test-dir $BUILD_DIR --output-on-failure
"
