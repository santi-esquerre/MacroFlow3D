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
DEFAULT_CUDA_HOME="${MACROFLOW3D_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda-11.4}}"

prepend_path_if_dir() {
    local dir="$1"
    if [[ -d "${dir}" ]]; then
        PATH="${dir}${PATH:+:${PATH}}"
    fi
}

pick_python() {
    if [[ -n "${MACROFLOW3D_PYTHON:-}" ]]; then
        printf '%s' "${MACROFLOW3D_PYTHON}"
        return 0
    fi

    if [[ -x /usr/bin/python3 ]]; then
        printf '%s' "/usr/bin/python3"
        return 0
    fi

    command -v python3
}

pick_cpp() {
    if [[ -n "${MACROFLOW3D_CPP:-}" ]]; then
        printf '%s' "${MACROFLOW3D_CPP}"
        return 0
    fi

    if [[ -x /usr/bin/cpp ]]; then
        printf '%s' "/usr/bin/cpp"
        return 0
    fi

    command -v cpp
}

pick_host_cc() {
    if [[ -n "${MACROFLOW3D_HOST_CC:-}" ]]; then
        printf '%s' "${MACROFLOW3D_HOST_CC}"
        return 0
    fi

    if [[ -x /opt/rh/devtoolset-9/root/usr/bin/gcc ]]; then
        printf '%s' "/opt/rh/devtoolset-9/root/usr/bin/gcc"
        return 0
    fi

    command -v gcc
}

pick_host_cxx() {
    if [[ -n "${MACROFLOW3D_HOST_CXX:-}" ]]; then
        printf '%s' "${MACROFLOW3D_HOST_CXX}"
        return 0
    fi

    if [[ -x /opt/rh/devtoolset-9/root/usr/bin/g++ ]]; then
        printf '%s' "/opt/rh/devtoolset-9/root/usr/bin/g++"
        return 0
    fi

    command -v g++
}

detect_mpi_flag_path() {
    local flag="$1"
    local compiler="$2"
    local show_output

    show_output="$("${compiler}" -show 2>/dev/null || true)"
    printf '%s\n' "${show_output}" | tr ' ' '\n' | sed -n "s/^-${flag}//p" | head -1
}

pick_existing_lib() {
    local dir="$1"
    shift
    local candidate
    for candidate in "$@"; do
        if [[ -f "${dir}/${candidate}" ]]; then
            printf '%s' "${dir}/${candidate}"
            return 0
        fi
    done
    return 1
}

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

# ── Deterministic toolchain environment ───────────────────────────────────
unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PREFIX CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE
unset PYTHONPATH PYTHONHOME _CE_CONDA _CE_M
export LC_ALL=C
export LANG=C

PATH=""
prepend_path_if_dir "/opt/rh/devtoolset-9/root/usr/bin"
prepend_path_if_dir "${DEFAULT_CUDA_HOME}/bin"
prepend_path_if_dir "${MPI_BIN:-}"
prepend_path_if_dir "/usr/lib64/mpich-3.2/bin"
prepend_path_if_dir "/usr/local/bin"
prepend_path_if_dir "/usr/bin"
prepend_path_if_dir "/usr/local/sbin"
prepend_path_if_dir "/usr/sbin"
prepend_path_if_dir "${HOME}/.local/bin"
prepend_path_if_dir "${HOME}/bin"
export PATH

MPI_CC="${MACROFLOW3D_MPI_CC:-$(command -v mpicc)}"
MPI_CXX="${MACROFLOW3D_MPI_CXX:-$(command -v mpicxx)}"
HOST_CC="$(pick_host_cc)"
HOST_CXX="$(pick_host_cxx)"
PYTHON_BIN="$(pick_python)"
CPP_BIN="$(pick_cpp)"
NVCC_BIN="${MACROFLOW3D_NVCC:-${DEFAULT_CUDA_HOME}/bin/nvcc}"
MPI_INCLUDE_DIR="${MACROFLOW3D_MPI_INCLUDE:-${MPI_INCLUDE:-}}"
MPI_LIB_DIR="${MACROFLOW3D_MPI_LIBDIR:-${MPI_LIB:-}}"

# Force MPI wrapper compilers onto a modern host toolchain. On the V100 host the
# MPICH wrappers otherwise fall back to GCC 4.8-era C++ headers, which PETSc
# correctly rejects during C++11/14/17 capability checks.
export MPICH_CC="${MACROFLOW3D_MPICH_CC:-${HOST_CC}}"
export MPICH_CXX="${MACROFLOW3D_MPICH_CXX:-${HOST_CXX}}"
export OMPI_CC="${MACROFLOW3D_OMPI_CC:-${HOST_CC}}"
export OMPI_CXX="${MACROFLOW3D_OMPI_CXX:-${HOST_CXX}}"
export CUDAHOSTCXX="${MACROFLOW3D_CUDAHOSTCXX:-${HOST_CXX}}"

if [[ -z "${MPI_INCLUDE_DIR}" ]]; then
    MPI_INCLUDE_DIR="$(detect_mpi_flag_path "I" "${MPI_CC}")"
fi

if [[ -z "${MPI_LIB_DIR}" ]]; then
    MPI_LIB_DIR="$(detect_mpi_flag_path "L" "${MPI_CC}")"
fi

MPI_CXX_LIB="$(pick_existing_lib "${MPI_LIB_DIR}" libmpicxx.so libmpicxx.a)"
MPI_C_LIB="$(pick_existing_lib "${MPI_LIB_DIR}" libmpi.so libmpi.a)"
PETSC_MPI_LIBS="[${MPI_CXX_LIB},${MPI_C_LIB}]"

# ── Preflight checks ─────────────────────────────────────────────────────
echo "=== PETSc + SLEPc Build Script ==="
echo "  Project root : ${PROJECT_ROOT}"
echo "  CUDA arch    : sm_${CUDA_ARCH}"
echo "  Make jobs    : ${MAKE_NP}"
echo "  Python       : ${PYTHON_BIN}"
echo "  CPP          : ${CPP_BIN}"
echo "  MPI C        : ${MPI_CC}"
echo "  MPI C++      : ${MPI_CXX}"
echo "  Host CC      : ${HOST_CC}"
echo "  Host C++     : ${HOST_CXX}"
echo "  MPI include  : ${MPI_INCLUDE_DIR}"
echo "  MPI lib dir  : ${MPI_LIB_DIR}"
echo "  NVCC         : ${NVCC_BIN}"
echo ""

for cmd in "${MPI_CC}" "${MPI_CXX}" "${NVCC_BIN}" "${PYTHON_BIN}" "${CPP_BIN}" make; do
    if [[ ! -x "${cmd}" ]] && ! command -v "${cmd}" &>/dev/null; then
        echo "ERROR: Required command '${cmd}' not found." >&2
        exit 1
    fi
done

if [[ -z "${MPI_INCLUDE_DIR}" ]]; then
    echo "ERROR: Could not determine MPI include directory." >&2
    exit 1
fi

if [[ -z "${MPI_LIB_DIR}" ]]; then
    echo "ERROR: Could not determine MPI library directory." >&2
    exit 1
fi

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

"${PYTHON_BIN}" ./configure \
    PETSC_ARCH="${PETSC_ARCH_NAME}" \
    --with-cc="${MPI_CC}" \
    --with-cxx="${MPI_CXX}" \
    --with-fc=0 \
    --with-mpi-include="${MPI_INCLUDE_DIR}" \
    --with-mpi-lib="${PETSC_MPI_LIBS}" \
    --with-debugging=0 \
    --with-cuda=1 \
    --with-cudac="${NVCC_BIN}" \
    --with-cuda-arch="${CUDA_ARCH}" \
    --with-shared-libraries=0 \
    --with-precision=double \
    --with-scalar-type=real \
    --with-make-np="${MAKE_NP}" \
    CPP="${CPP_BIN}" \
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

"${PYTHON_BIN}" ./configure

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
