#!/usr/bin/env bash
# Setup sccache for MacroFlow3D builds.
#
# Source this script before running cmake:
#   source scripts/setup-sccache.sh
#   cmake --preset wsl-debug
#
# Or just run it to check status:
#   bash scripts/setup-sccache.sh
set -euo pipefail

if command -v sccache &>/dev/null; then
    SCCACHE_PATH="$(command -v sccache)"
    echo "sccache found: $SCCACHE_PATH"

    # Export for CMake (also works if cmake doesn't auto-detect)
    export CMAKE_CXX_COMPILER_LAUNCHER="$SCCACHE_PATH"
    export CMAKE_CUDA_COMPILER_LAUNCHER="$SCCACHE_PATH"

    # Show cache stats
    sccache --show-stats 2>/dev/null || true
    echo ""
    echo "sccache is ready. CMake will auto-detect it via find_program()."
    echo "You can also pass -DCMAKE_CXX_COMPILER_LAUNCHER=$SCCACHE_PATH manually."
else
    echo "sccache not found in PATH."
    echo ""
    echo "To install (no root required):"
    echo "  cargo install sccache --locked"
    echo ""
    echo "Or with conda:"
    echo "  conda install -c conda-forge sccache"
    echo ""
    echo "Or download a binary:"
    echo "  https://github.com/mozilla/sccache/releases"
    echo ""
    echo "The build will work fine without sccache — it just won't cache."
fi
