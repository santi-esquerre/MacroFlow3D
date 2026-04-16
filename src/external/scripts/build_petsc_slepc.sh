#!/usr/bin/env bash
# ============================================================================
# build_petsc_slepc.sh — Reproducible static build of PETSc + SLEPc with CUDA
# ============================================================================
#
# Pinned versions:
#   PETSc  3.24.5   (src/external/petsc,  git tag v3.24.5)
#   SLEPc  3.24.2   (src/external/slepc,  git tag v3.24.2)
#
# CUDA architecture strategy:
#   PETSc only supports a single -arch=sm_XX per build. We default to sm_70
#   (Tesla V100) which produces both SASS and PTX. Higher-capability GPUs
#   (RTX 3050 = sm_86) run via forward-compatible JIT from the embedded PTX.
#   This is fine for PETSc's memory-bound kernels.
#
#   MacroFlow3D's own code uses CMAKE_CUDA_ARCHITECTURES = "70;75;80;86" for
#   full fat-binary coverage — that is independent of this script.
#
# Override the CUDA arch via environment variable:
#   MACROFLOW3D_CUDA_ARCH=86 ./build_petsc_slepc.sh
#
# Prerequisites (system packages):
#   mpicc, mpicxx   (OpenMPI or MPICH)
#   nvcc            (CUDA toolkit ≥ 11.0)
#   python3         (for PETSc/SLEPc configure)
#   make, cmake
#   LAPACK, BLAS    (liblapack-dev, libblas-dev on Debian/Ubuntu)
#
# Usage:
#   cd <project_root>
#   bash src/external/scripts/build_petsc_slepc.sh        # full build
#   bash src/external/scripts/build_petsc_slepc.sh clean   # remove build artifacts
#
# ============================================================================
set -euo pipefail

# ── Locate project root (one level up from src/external/scripts) ──────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

PETSC_SRC="${PROJECT_ROOT}/src/external/petsc"
SLEPC_SRC="${PROJECT_ROOT}/src/external/slepc"

# ── Configurable parameters ───────────────────────────────────────────────
CUDA_ARCH="${MACROFLOW3D_CUDA_ARCH:-70}"   # default sm_70 (V100)
MAKE_NP="${MACROFLOW3D_MAKE_NP:-2}"        # parallel jobs (keep low for CUDA)
PETSC_ARCH_NAME="arch-cuda"

# ── Clean mode ────────────────────────────────────────────────────────────
if [[ "${1:-}" == "clean" ]]; then
    echo "=== Cleaning PETSc + SLEPc build artifacts ==="
    rm -rf "${PETSC_SRC}/${PETSC_ARCH_NAME}"
    rm -f  "${PETSC_SRC}/configure.log" "${PETSC_SRC}/RDict.log"
    rm -f  "${PETSC_SRC}/configtest.mod" "${PETSC_SRC}/.nagged"
    rm -rf "${SLEPC_SRC}/${PETSC_ARCH_NAME}"
    echo "Done."
    exit 0
fi

# ── Preflight checks ─────────────────────────────────────────────────────
echo "=== PETSc + SLEPc Build Script ==="
echo "  Project root : ${PROJECT_ROOT}"
echo "  CUDA arch    : sm_${CUDA_ARCH}"
echo "  Make jobs    : ${MAKE_NP}"
echo ""

for cmd in mpicc mpicxx nvcc python3 make; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "ERROR: Required command '$cmd' not found in PATH." >&2
        exit 1
    fi
done

if [[ ! -d "${PETSC_SRC}/config" ]]; then
    echo "ERROR: PETSc source not found at ${PETSC_SRC}" >&2
    echo "       Clone with: git clone --depth 1 -b v3.24.5 https://gitlab.com/petsc/petsc.git src/external/petsc" >&2
    exit 1
fi

if [[ ! -d "${SLEPC_SRC}/config" ]]; then
    echo "ERROR: SLEPc source not found at ${SLEPC_SRC}" >&2
    echo "       Clone with: git clone --depth 1 -b v3.24.2 https://gitlab.com/slepc/slepc.git src/external/slepc" >&2
    exit 1
fi

# ── Phase 1: Configure PETSc ─────────────────────────────────────────────
echo "============================================================"
echo "  Phase 1: Configuring PETSc (sm_${CUDA_ARCH}, static)"
echo "============================================================"

cd "${PETSC_SRC}"

# Clean previous build if present
rm -rf "${PETSC_ARCH_NAME}" configure.log RDict.log configtest.mod .nagged 2>/dev/null || true

python3 ./configure \
    PETSC_ARCH="${PETSC_ARCH_NAME}" \
    --with-cc=mpicc \
    --with-cxx=mpicxx \
    --with-fc=0 \
    --with-debugging=0 \
    --with-cuda=1 \
    --with-cudac=nvcc \
    --with-cuda-arch="${CUDA_ARCH}" \
    --with-shared-libraries=0 \
    --with-precision=double \
    --with-scalar-type=real \
    --with-make-np="${MAKE_NP}" \
    COPTFLAGS="-O2" \
    CXXOPTFLAGS="-O2" \
    CUDAOPTFLAGS="-O2"

echo ""
echo "  PETSc configure complete."

# ── Phase 2: Build PETSc ─────────────────────────────────────────────────
echo "============================================================"
echo "  Phase 2: Building PETSc"
echo "============================================================"

make PETSC_DIR="${PETSC_SRC}" PETSC_ARCH="${PETSC_ARCH_NAME}" all -j"${MAKE_NP}"

# Verify
if [[ ! -f "${PETSC_SRC}/${PETSC_ARCH_NAME}/lib/libpetsc.a" ]]; then
    echo "ERROR: libpetsc.a not found after build." >&2
    exit 1
fi
echo ""
echo "  PETSc build complete: $(ls -lh "${PETSC_SRC}/${PETSC_ARCH_NAME}/lib/libpetsc.a" | awk '{print $5}')"

# ── Phase 3: Configure SLEPc ─────────────────────────────────────────────
echo "============================================================"
echo "  Phase 3: Configuring SLEPc"
echo "============================================================"

cd "${SLEPC_SRC}"

# Clean previous build if present
rm -rf "${PETSC_ARCH_NAME}" 2>/dev/null || true

export PETSC_DIR="${PETSC_SRC}"
export PETSC_ARCH="${PETSC_ARCH_NAME}"

python3 ./configure

echo ""
echo "  SLEPc configure complete."

# ── Phase 4: Build SLEPc ─────────────────────────────────────────────────
echo "============================================================"
echo "  Phase 4: Building SLEPc"
echo "============================================================"

make SLEPC_DIR="${SLEPC_SRC}" PETSC_DIR="${PETSC_SRC}" PETSC_ARCH="${PETSC_ARCH_NAME}" -j"${MAKE_NP}"

# Verify
if [[ ! -f "${SLEPC_SRC}/${PETSC_ARCH_NAME}/lib/libslepc.a" ]]; then
    echo "ERROR: libslepc.a not found after build." >&2
    exit 1
fi
echo ""
echo "  SLEPc build complete: $(ls -lh "${SLEPC_SRC}/${PETSC_ARCH_NAME}/lib/libslepc.a" | awk '{print $5}')"

# ── Summary ───────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Build Summary"
echo "============================================================"
echo "  PETSc version : $(grep PETSC_VERSION_SUBMINOR "${PETSC_SRC}/include/petscversion.h" | head -1 | awk '{print "3.24."$3}')"
echo "  SLEPc version : $(grep SLEPC_VERSION_SUBMINOR "${SLEPC_SRC}/include/slepcversion.h" | head -1 | awk '{print "3.24."$3}')"
echo "  CUDA arch     : sm_${CUDA_ARCH}"
echo "  PETSC_DIR     : ${PETSC_SRC}"
echo "  PETSC_ARCH    : ${PETSC_ARCH_NAME}"
echo "  SLEPC_DIR     : ${SLEPC_SRC}"
echo "  libpetsc.a    : ${PETSC_SRC}/${PETSC_ARCH_NAME}/lib/libpetsc.a"
echo "  libslepc.a    : ${SLEPC_SRC}/${PETSC_ARCH_NAME}/lib/libslepc.a"
echo ""
echo "  To use in MacroFlow3D:"
echo "    cmake -DMACROFLOW3D_ENABLE_PETSC=ON .."
echo ""
