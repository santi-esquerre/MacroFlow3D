cmake_minimum_required(VERSION 3.18)
project(MacroFlow3D LANGUAGES CUDA CXX)
enable_testing()

# ============================================================================
# Standards
# ============================================================================
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# Optional: sccache compiler launcher (auto-detected, no hard dependency)
# ============================================================================
if(NOT DEFINED CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(SCCACHE_PROGRAM sccache)
    if(SCCACHE_PROGRAM)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}" CACHE STRING "")
        set(CMAKE_CUDA_COMPILER_LAUNCHER "${SCCACHE_PROGRAM}" CACHE STRING "")
        message(STATUS "sccache found: ${SCCACHE_PROGRAM} — enabled as compiler launcher")
    endif()
endif()

# ============================================================================
# Project options
# ============================================================================
option(MACROFLOW3D_BUILD_APPS          "Build application executables"                  ON)
option(MACROFLOW3D_ENABLE_DIAGNOSTICS  "Enable diagnostic prints and sync checks"      OFF)
option(MACROFLOW3D_ENABLE_PROFILING    "Enable stage profiling with CUDA events"       OFF)
option(MACROFLOW3D_ENABLE_NVTX         "Enable NVTX markers for Nsight Systems"       OFF)
option(MACROFLOW3D_ENABLE_PETSC        "Enable PETSc/SLEPc eigensolver backend"       OFF)

# CUDA architectures: honour CMAKE_CUDA_ARCHITECTURES if set externally,
# otherwise provide a sensible default covering Volta → Ampere.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)
endif()

# In Debug builds, enable diagnostics by default
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(MACROFLOW3D_ENABLE_DIAGNOSTICS ON CACHE BOOL "" FORCE)
endif()

# ============================================================================
# Find CUDA toolkit
# ============================================================================
find_package(CUDAToolkit REQUIRED)

# clangd
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES OFF)


# ============================================================================
# External: yaml-cpp (vendorized, static build, tests/tools off)
# ============================================================================
set(YAML_CPP_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_TOOLS   OFF CACHE BOOL "" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
set(YAML_CPP_INSTALL       OFF CACHE BOOL "" FORCE)
add_subdirectory(src/external/yaml-cpp-0.8.0 EXCLUDE_FROM_ALL)

# ============================================================================
# External: nlohmann/json (header-only, vendorized)
# ============================================================================
# Included via src/external/nlohmann/json.hpp — no add_subdirectory needed.

# ============================================================================
# Build metadata: git hash, build type, CUDA architectures (Etapa 8)
# ============================================================================
execute_process(
    COMMAND git rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE MACROFLOW3D_GIT_HASH_RAW
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE GIT_RESULT
)
if(NOT GIT_RESULT EQUAL 0)
    set(MACROFLOW3D_GIT_HASH_RAW "unknown")
endif()

# Stringify CUDA architectures list for embedding
string(REPLACE ";" " " MACROFLOW3D_CUDA_ARCH_STR "${CMAKE_CUDA_ARCHITECTURES}")

# ============================================================================
# External: Par2_Core (GPU RWPT engine, static build, extras off)
# ============================================================================
if(NOT EXISTS "${CMAKE_SOURCE_DIR}/src/external/Par2_Core/CMakeLists.txt")
    message(FATAL_ERROR
        "Par2_Core is required but src/external/Par2_Core is not populated.\n"
        "Initialize the submodule in this worktree before configuring:\n"
        "  git submodule update --init --recursive src/external/Par2_Core")
endif()
set(PAR2_BUILD_EXAMPLES       OFF CACHE BOOL "" FORCE)
set(PAR2_BUILD_LEGACY_RUNNER  OFF CACHE BOOL "" FORCE)
set(PAR2_BUILD_ORIGINAL       OFF CACHE BOOL "" FORCE)
add_subdirectory(src/external/Par2_Core EXCLUDE_FROM_ALL)

# ============================================================================
# External: PETSc + SLEPc + hypre   (optional — full-GPU eigensolver backend)
# ============================================================================
if(MACROFLOW3D_ENABLE_PETSC)
    # --- Locate PETSc / SLEPc installations ---
    if(NOT DEFINED PETSC_DIR)
        set(PETSC_DIR "${CMAKE_SOURCE_DIR}/src/external/petsc")
    endif()
    if(NOT DEFINED PETSC_ARCH)
        set(PETSC_ARCH "arch-cuda")
    endif()
    if(NOT DEFINED SLEPC_DIR)
        set(SLEPC_DIR "${CMAKE_SOURCE_DIR}/src/external/slepc")
    endif()

    set(_PETSC_LIB_DIR "${PETSC_DIR}/${PETSC_ARCH}/lib")
    set(_SLEPC_LIB_DIR "${SLEPC_DIR}/${PETSC_ARCH}/lib")
    set(_PETSC_CONF_VARS "${_PETSC_LIB_DIR}/petsc/conf/petscvariables")

    # --- Pre-flight: verify libraries exist ---
    if(NOT EXISTS "${_PETSC_LIB_DIR}/libpetsc.a")
        message(FATAL_ERROR
            "MACROFLOW3D_ENABLE_PETSC=ON but libpetsc.a not found at:\n"
            "  ${_PETSC_LIB_DIR}/libpetsc.a\n"
            "Build PETSc first:\n"
            "  bash src/external/scripts/build_petsc_slepc.sh")
    endif()
    if(NOT EXISTS "${_SLEPC_LIB_DIR}/libslepc.a")
        message(FATAL_ERROR
            "MACROFLOW3D_ENABLE_PETSC=ON but libslepc.a not found at:\n"
            "  ${_SLEPC_LIB_DIR}/libslepc.a\n"
            "Build SLEPc first:\n"
            "  bash src/external/scripts/build_petsc_slepc.sh")
    endif()
    if(NOT EXISTS "${_PETSC_CONF_VARS}")
        message(FATAL_ERROR
            "MACROFLOW3D_ENABLE_PETSC=ON but PETSc link metadata not found at:\n"
            "  ${_PETSC_CONF_VARS}\n"
            "Rebuild PETSc through the repo workflow:\n"
            "  bash src/external/scripts/build_petsc_slepc.sh")
    endif()

    file(STRINGS "${_PETSC_CONF_VARS}" _petsc_external_lib_line
         REGEX "^PETSC_EXTERNAL_LIB_BASIC = ")
    if(NOT _petsc_external_lib_line)
        message(FATAL_ERROR
            "Could not read PETSC_EXTERNAL_LIB_BASIC from:\n"
            "  ${_PETSC_CONF_VARS}")
    endif()
    string(REPLACE "PETSC_EXTERNAL_LIB_BASIC = " "" _petsc_external_lib_raw
           "${_petsc_external_lib_line}")
    separate_arguments(_petsc_external_lib_args UNIX_COMMAND "${_petsc_external_lib_raw}")
    set(_petsc_interface_link_libs "")
    foreach(_petsc_arg IN LISTS _petsc_external_lib_args)
        if(_petsc_arg STREQUAL "-lcudart")
            list(APPEND _petsc_interface_link_libs CUDA::cudart)
        elseif(_petsc_arg STREQUAL "-lcublas")
            list(APPEND _petsc_interface_link_libs CUDA::cublas)
        elseif(_petsc_arg STREQUAL "-lcusparse")
            list(APPEND _petsc_interface_link_libs CUDA::cusparse)
        elseif(_petsc_arg STREQUAL "-lcusolver")
            list(APPEND _petsc_interface_link_libs CUDA::cusolver)
        elseif(_petsc_arg STREQUAL "-lcurand")
            list(APPEND _petsc_interface_link_libs CUDA::curand)
        elseif(_petsc_arg STREQUAL "-lcufft")
            list(APPEND _petsc_interface_link_libs CUDA::cufft)
        elseif(_petsc_arg STREQUAL "-lnvToolsExt")
            list(APPEND _petsc_interface_link_libs CUDA::nvToolsExt)
        elseif(_petsc_arg STREQUAL "-lcuda")
            list(APPEND _petsc_interface_link_libs CUDA::cuda_driver)
        elseif(_petsc_arg MATCHES "^-l")
            string(SUBSTRING "${_petsc_arg}" 2 -1 _petsc_lib_name)
            if(NOT _petsc_lib_name STREQUAL "mpi" AND
               NOT _petsc_lib_name STREQUAL "mpicxx")
                list(APPEND _petsc_interface_link_libs "${_petsc_lib_name}")
            endif()
        endif()
    endforeach()
    list(REMOVE_DUPLICATES _petsc_interface_link_libs)

    # --- Import PETSc (static) ---
    # PETSc is a static import here, so its full external link surface must
    # appear after libpetsc.a. Read that surface from PETSc's generated
    # petscvariables metadata instead of maintaining a fragile handwritten list.
    add_library(petsc STATIC IMPORTED)
    set_target_properties(petsc PROPERTIES
        IMPORTED_LOCATION "${_PETSC_LIB_DIR}/libpetsc.a"
        INTERFACE_INCLUDE_DIRECTORIES
            "${PETSC_DIR}/include;${PETSC_DIR}/${PETSC_ARCH}/include"
        INTERFACE_LINK_LIBRARIES
            "${_petsc_interface_link_libs}"
    )

    # --- Import SLEPc (static) ---
    add_library(slepc STATIC IMPORTED)
    set_target_properties(slepc PROPERTIES
        IMPORTED_LOCATION "${_SLEPC_LIB_DIR}/libslepc.a"
        INTERFACE_INCLUDE_DIRECTORIES
            "${SLEPC_DIR}/include;${SLEPC_DIR}/${PETSC_ARCH}/include"
        INTERFACE_LINK_LIBRARIES
            "petsc"
    )

    # --- MPI (needed by PETSc even in single-rank mode) ---
    find_package(MPI REQUIRED COMPONENTS CXX)

    message(STATUS "PETSc/SLEPc eigensolver backend ENABLED")
    message(STATUS "  PETSC_DIR  = ${PETSC_DIR}")
    message(STATUS "  PETSC_ARCH = ${PETSC_ARCH}")
    message(STATUS "  SLEPC_DIR  = ${SLEPC_DIR}")
endif()

# ============================================================================
# Source files
# ============================================================================
set(RUNTIME_SOURCES
    src/runtime/CudaContext.cu
    src/runtime/pipeline/PipelineRunner.cu
    src/runtime/ensemble/EnsembleRunner.cu
)

set(BLAS_SOURCES
    src/numerics/blas/fill.cu
    src/numerics/blas/copy.cu
    src/numerics/blas/scal.cu
    src/numerics/blas/axpy.cu
    src/numerics/blas/axpby.cu
    src/numerics/blas/dot.cu
    src/numerics/blas/nrm2.cu
    src/numerics/blas/cg_kernels.cu
)

set(SOLVER_SOURCES
    src/numerics/solvers/preconditioner.cu
    src/numerics/solvers/mg_preconditioner.cu
)

set(OPERATOR_SOURCES
    src/numerics/operators/poisson3d_operator.cu
    src/numerics/operators/varcoeff_laplacian.cu
)

set(MULTIGRID_SOURCES
    src/multigrid/transfer/restrict_3d.cu
    src/multigrid/transfer/prolong_3d.cu
    src/multigrid/smoothers/residual_3d.cu
    src/multigrid/smoothers/gsrb_3d.cu
    src/multigrid/cycle/v_cycle.cu
)

set(PHYSICS_SOURCES
    src/physics/flow/coarsen_K.cu
    src/physics/flow/rhs_head.cu
    src/physics/flow/solve_head.cu
    src/physics/flow/velocity_from_head.cu
    src/physics/stochastic/stochastic.cu
    src/physics/flow/velocity_diagnostics.cu
    src/physics/particles/par2_adapter/Par2TransportAdapter.cu
    src/physics/particles/par2_adapter/Par2StatsAdapter.cu
    src/physics/particles/par2_adapter/Par2SnapshotAdapter.cu
    src/physics/particles/pspta/PsptaPsiField.cu
    src/physics/particles/pspta/PsptaEngine.cu
    # PSPTA refactor: invariants module (Stages 1-3)
    src/physics/particles/pspta/invariants/PsptaInvariantField.cu
    src/physics/particles/pspta/invariants/TransportOperator3D.cu
    src/physics/particles/pspta/invariants/EigensolverBackend.cu
    src/physics/particles/pspta/invariants/GaugeFixer.cu
    src/physics/particles/pspta/invariants/InvariantPairSearch.cpp
    src/physics/particles/pspta/invariants/RefinementAC.cu
    src/physics/particles/pspta/invariants/OperatorTestHarness.cu
    src/physics/particles/pspta/legacy/LegacyMarchingInvariantBuilder.cu
)

# PETSc/SLEPc sources (conditional)
if(MACROFLOW3D_ENABLE_PETSC)
    list(APPEND RUNTIME_SOURCES src/runtime/PetscSlepcInit.cpp)
    list(APPEND PHYSICS_SOURCES
        src/physics/particles/pspta/invariants/SLEPcBackend.cu)
endif()

set(IO_SOURCES
    src/io/config/ConfigYaml.cpp
    src/runtime/analysis/AnalysisRunner.cpp
)

# ============================================================================
# Library target: macroflow3d_lib
# ============================================================================
add_library(macroflow3d_lib STATIC
    ${RUNTIME_SOURCES}
    ${BLAS_SOURCES}
    ${SOLVER_SOURCES}
    ${OPERATOR_SOURCES}
    ${MULTIGRID_SOURCES}
    ${PHYSICS_SOURCES}
    ${IO_SOURCES}
)

target_include_directories(macroflow3d_lib PUBLIC ${CMAKE_SOURCE_DIR})

# Link dependencies — curand is needed by stochastic.cu (in the lib)
target_link_libraries(macroflow3d_lib
    PUBLIC  CUDA::cudart CUDA::cublas CUDA::curand
            yaml-cpp par2_core
)

# PETSc/SLEPc link (conditional)
if(MACROFLOW3D_ENABLE_PETSC)
    # Dependencies (curand, cusparse, cusolver, cufft, lapack, blas, etc.)
    # are declared as INTERFACE_LINK_LIBRARIES on the imported targets above,
    # so CMake resolves them in the correct static-link order automatically.
    target_link_libraries(macroflow3d_lib
        PUBLIC  slepc petsc MPI::MPI_CXX
    )
    target_compile_definitions(macroflow3d_lib PUBLIC MACROFLOW3D_HAS_PETSC=1)
endif()

# Separable compilation (required for cross-TU __device__ function references)
set_target_properties(macroflow3d_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# NVCC flags
target_compile_options(macroflow3d_lib
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
)

# ============================================================================
# Build metadata compile definitions (Etapa 8)
#   Injected so BuildInfo.hpp can read them at compile time.
# ============================================================================
target_compile_definitions(macroflow3d_lib PUBLIC
    MACROFLOW3D_GIT_HASH="${MACROFLOW3D_GIT_HASH_RAW}"
    MACROFLOW3D_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
    MACROFLOW3D_CUDA_ARCHITECTURES="${MACROFLOW3D_CUDA_ARCH_STR}"
)

# ============================================================================
# Diagnostic and profiling compile definitions
#
#   cmake -DCMAKE_BUILD_TYPE=Release ..                      # All flags OFF
#   cmake -DCMAKE_BUILD_TYPE=Debug ..                        # DIAGNOSTICS ON
#   cmake -DMACROFLOW3D_ENABLE_PROFILING=ON ..               # Force profiling ON
# ============================================================================
if(MACROFLOW3D_ENABLE_DIAGNOSTICS)
    target_compile_definitions(macroflow3d_lib PUBLIC MACROFLOW3D_ENABLE_DIAGNOSTICS=1)
else()
    target_compile_definitions(macroflow3d_lib PUBLIC MACROFLOW3D_ENABLE_DIAGNOSTICS=0)
endif()

if(MACROFLOW3D_ENABLE_PROFILING)
    target_compile_definitions(macroflow3d_lib PUBLIC MACROFLOW3D_ENABLE_PROFILING=1)
else()
    target_compile_definitions(macroflow3d_lib PUBLIC MACROFLOW3D_ENABLE_PROFILING=0)
endif()

if(MACROFLOW3D_ENABLE_NVTX)
    target_compile_definitions(macroflow3d_lib PUBLIC MACROFLOW3D_ENABLE_NVTX=1 USE_NVTX=1)
    target_link_libraries(macroflow3d_lib PUBLIC CUDA::nvToolsExt)
else()
    target_compile_definitions(macroflow3d_lib PUBLIC MACROFLOW3D_ENABLE_NVTX=0)
endif()

# ============================================================================
# Apps
# ============================================================================
if(MACROFLOW3D_BUILD_APPS)
    add_executable(macroflow3d_pipeline apps/macroflow3d_pipeline.cu)
    target_link_libraries(macroflow3d_pipeline PRIVATE macroflow3d_lib)

    # Operator test runner (no PETSc dependency)
    add_executable(run_operator_tests apps/run_operator_tests.cu)
    target_link_libraries(run_operator_tests PRIVATE macroflow3d_lib)
    target_compile_options(run_operator_tests
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
    )
    add_test(NAME operator_tests COMMAND run_operator_tests)

    add_executable(test_invariant_pair_search apps/test_invariant_pair_search.cpp)
    target_link_libraries(test_invariant_pair_search PRIVATE macroflow3d_lib)
    add_test(NAME invariant_pair_search COMMAND test_invariant_pair_search)

    add_executable(test_refinement_ac apps/test_refinement_ac.cpp)
    target_link_libraries(test_refinement_ac PRIVATE macroflow3d_lib)
    add_test(NAME refinement_ac COMMAND test_refinement_ac)

    add_executable(test_refinement_ac_subspace apps/test_refinement_ac_subspace.cpp)
    target_link_libraries(test_refinement_ac_subspace PRIVATE macroflow3d_lib)
    add_test(NAME refinement_ac_subspace COMMAND test_refinement_ac_subspace)

    add_executable(test_refinement_ac_subspace_gn apps/test_refinement_ac_subspace_gn.cpp)
    target_link_libraries(test_refinement_ac_subspace_gn PRIVATE macroflow3d_lib)
    add_test(NAME refinement_ac_subspace_gn COMMAND test_refinement_ac_subspace_gn)

    add_executable(test_refinement_ac_transport_proxy apps/test_refinement_ac_transport_proxy.cpp)
    target_link_libraries(test_refinement_ac_transport_proxy PRIVATE macroflow3d_lib)
    add_test(NAME refinement_ac_transport_proxy COMMAND test_refinement_ac_transport_proxy)

    add_executable(test_refinement_ac_engine_proxy apps/test_refinement_ac_engine_proxy.cpp)
    target_link_libraries(test_refinement_ac_engine_proxy PRIVATE macroflow3d_lib)
    add_test(NAME refinement_ac_engine_proxy COMMAND test_refinement_ac_engine_proxy)

    # PETSc/SLEPc apps (only when backend is enabled)
    if(MACROFLOW3D_ENABLE_PETSC)
        # Minimal smoke test — verifies init, VECCUDA creation, link
        add_executable(smoke_test_petsc apps/smoke_test_petsc.cu)
        target_link_libraries(smoke_test_petsc PRIVATE macroflow3d_lib)
        target_compile_options(smoke_test_petsc
            PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
        )

        add_test(NAME smoke_test_petsc COMMAND smoke_test_petsc)

        # Full eigensolver validation (for future stages)
        add_executable(validate_slepc_eigensolver apps/validate_slepc_eigensolver.cu)
        target_link_libraries(validate_slepc_eigensolver PRIVATE macroflow3d_lib)
        target_compile_options(validate_slepc_eigensolver
            PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
        )
        add_test(NAME validate_slepc_eigensolver COMMAND validate_slepc_eigensolver)

        # Stage 4C: grid-ladder benchmark (not a ctest — run manually)
        add_executable(benchmark_eigensolver apps/benchmark_eigensolver.cu)
        target_link_libraries(benchmark_eigensolver PRIVATE macroflow3d_lib)
        target_compile_options(benchmark_eigensolver
            PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
        )

        # Gate 3/5/6 invariant-quality and transport-consumption analysis
        add_executable(analyze_invariant_quality apps/analyze_invariant_quality.cu)
        target_link_libraries(analyze_invariant_quality PRIVATE macroflow3d_lib)
        target_compile_options(analyze_invariant_quality
            PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
        )
    endif()

    # add_executable(verify_stochastic apps/verify_stochastic.cu)
    # target_link_libraries(verify_stochastic PRIVATE macroflow3d_lib)
endif()
