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

    # --- Import PETSc (static) ---
    # PETSc internally uses curand, cufft, cusparse, cusolver.
    # Declaring them as INTERFACE_LINK_LIBRARIES ensures they appear
    # AFTER libpetsc.a in the link line (critical for static linking).
    add_library(petsc STATIC IMPORTED)
    set_target_properties(petsc PROPERTIES
        IMPORTED_LOCATION "${_PETSC_LIB_DIR}/libpetsc.a"
        INTERFACE_INCLUDE_DIRECTORIES
            "${PETSC_DIR}/include;${PETSC_DIR}/${PETSC_ARCH}/include"
        INTERFACE_LINK_LIBRARIES
            "CUDA::cusparse;CUDA::cusolver;CUDA::curand;CUDA::cufft;lapack;blas;m;stdc++;dl"
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
    src/numerics/blas/sum.cu
    src/numerics/blas/cg_kernels.cu
)

set(SOLVER_SOURCES
    src/numerics/solvers/preconditioner.cu
    src/numerics/solvers/mg_preconditioner.cu
    src/numerics/solvers/projected_positive_mg_preconditioner.cu
)

set(CONSTRAINT_SOURCES
    src/numerics/constraints/MeanZeroProjector.cu
)

set(OPERATOR_SOURCES
    src/numerics/operators/poisson3d_operator.cu
    src/numerics/operators/varcoeff_laplacian.cu
)

# SF-28: periodic tricubic B-spline interpolation (value + gradient, GPU + CPU mirror)
set(INTERPOLATION_SOURCES
    src/numerics/interpolation/PeriodicTricubicBSpline.cu
    src/numerics/interpolation/NonPeriodicXTricubicBSpline.cu
)

# SF-31: streamline-tracker shared label evaluator and particle bookkeeping
set(STREAMLINE_TRACKER_SOURCES
    src/physics/particles/streamline_tracker/StreamlineTrackerCommon.cu
    src/physics/particles/streamline_tracker/PseudoSymplecticTracker.cu
    src/physics/particles/streamline_tracker/ReferenceRkTracker.cu
    src/physics/particles/streamline_tracker/StokesFaceVelocity.cu
    src/physics/particles/streamline_tracker/PollockTracker.cu
)

# SF-33: inlet-label equation (14) on the x1-non-periodic slab (grid/types, 4th-order stencils, residual, metrics)
set(INLET_SLAB_SOURCES
    src/physics/streamfunctions/inlet_slab/SlabStencils4.cu
    src/physics/streamfunctions/inlet_slab/SlabResidual.cu
    src/physics/streamfunctions/inlet_slab/SlabMetrics.cu
    src/physics/streamfunctions/inlet_slab/SlabJacobianVectorProduct.cu
    src/physics/streamfunctions/inlet_slab/SlabGmres.cu
    src/physics/streamfunctions/inlet_slab/SlabModePreconditioner.cu
    src/physics/streamfunctions/inlet_slab/SlabNewtonKrylov.cu
    src/physics/streamfunctions/inlet_slab/SlabCoarseCorrection.cu
    # SF-33 N3: D-1 inlet labels, production inputs (SF-18/SF-19/SF-28), production oracle
    src/physics/streamfunctions/inlet_slab/InletLabels.cu
    src/physics/streamfunctions/inlet_slab/SlabProductionSetup.cu
    src/physics/streamfunctions/inlet_slab/SlabOracle.cu
    src/physics/streamfunctions/inlet_slab/SlabLabelTransport.cu
    # SF-35 N6: GPU tracer (device copy of the SF-30 integrator; compiled with --fmad=false below)
    src/physics/streamfunctions/inlet_slab/SlabTraceGpu.cu
    src/physics/streamfunctions/inlet_slab/ProtoCase.cu
)

# SF-35 N6: IEEE mult/add in the GPU tracer so that it reproduces the host (no-FMA) tracer; this file only.
set_source_files_properties(src/physics/streamfunctions/inlet_slab/SlabTraceGpu.cu
    PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")

# SF-36 N4: forward Lagrangian-label tracker (device integrator templates; --fmad=false like SlabTraceGpu.cu)
set(LAGRANGIAN_LABELS_SOURCES
    src/physics/particles/lagrangian_labels/LagrangianTracker.cu
    src/physics/particles/lagrangian_labels/FaceFluxRepack.cu
    src/physics/particles/lagrangian_labels/PollockComparator.cu
    # SF-40 L3: lean parametric stage (host orchestration; includes only LagrangianTracker.cuh)
    src/physics/particles/lagrangian_labels/LongBoxStage.cu
    # SF-38 N2: medium (b) (solve_head, Dirichlet in x) and the Eulerian face statistics E1
    src/physics/particles/lagrangian_labels/DirichletMedium.cu
    # SF-41 N2: lean medium (b) stage (flow part; default flags like DirichletMedium.cu / LongBoxStage.cu)
    src/physics/particles/lagrangian_labels/DirichletLongBoxStage.cu
    src/physics/particles/lagrangian_labels/EulerianFaceStats.cpp
    # SF-38 N3: Lagrangian ensemble runner (host orchestration over the libraries above)
    src/physics/particles/lagrangian_labels/EnsembleRunner.cu
)
set_source_files_properties(src/physics/particles/lagrangian_labels/LagrangianTracker.cu
    PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")
# SF-37 N3: CompactMAC -> periodic face repack and the Pollock lattice comparator (V3b)
set_source_files_properties(src/physics/particles/lagrangian_labels/FaceFluxRepack.cu
    src/physics/particles/lagrangian_labels/PollockComparator.cu
    src/physics/particles/lagrangian_labels/EnsembleRunner.cu
    PROPERTIES COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>")

set(MULTIGRID_SOURCES
    src/multigrid/coefficient_hierarchy.cu
    src/multigrid/transfer/restrict_3d.cu
    src/multigrid/transfer/prolong_3d.cu
    src/multigrid/smoothers/residual_3d.cu
    src/multigrid/smoothers/gsrb_3d.cu
    src/multigrid/cycle/v_cycle.cu
    src/multigrid/cycle/projected_positive_v_cycle.cu
)

set(PHYSICS_SOURCES
    src/physics/streamfunctions/affine_periodic_rhs.cu
    src/physics/streamfunctions/DifferentialOperators.cu
    src/physics/streamfunctions/NonlinearSources.cu
    src/physics/streamfunctions/ResidualEvaluator.cu
    # SF-22: matrix-free Jacobian-vector product, reusing the accepted
    # residual evaluator verbatim (no second PDE discretization).
    src/physics/streamfunctions/JacobianVectorProduct.cu
    # SF-23: block diagonal MG preconditioner adapter and restarted,
    # right-preconditioned coupled GMRES built on the SF-22 matrix-free JVP.
    src/physics/streamfunctions/BlockDiagonalMGPreconditioner.cu
    src/physics/streamfunctions/CoupledGmres.cu
    # SF-25 T01: shifted-system matrix-free operator (mu*A_blk + J), a D-gate
    # enabler for the manifold-robust terminal solver diagnostic; see
    # docs/decisions/2026-08-14-manifold-robust-terminal-solver.md.
    src/physics/streamfunctions/ShiftedJacobianOperator.cu
    src/physics/streamfunctions/Diagnostics.cu
    src/physics/streamfunctions/AndersonAccelerator.cu
    # SF-24: globalized Newton-Krylov phase (inexact-Newton GMRES step +
    # Armijo line search) consuming the SF-22/SF-23 JVP/GMRES/preconditioner
    # surface; opt-in and bitwise-preserving when disabled.
    src/physics/streamfunctions/NewtonKrylovSolver.cu
    src/physics/streamfunctions/StreamfunctionWorkspace.cu
    src/physics/streamfunctions/StreamfunctionSolver.cu
    src/physics/streamfunctions/ContinuationController.cu
    src/physics/flow/rhs_head.cu
    src/physics/flow/solve_head.cu
    src/physics/flow/velocity_from_head.cu
    # SF-19: standalone affine-periodic Darcy solve for the triply periodic
    # benchmark (does not touch the Dirichlet head/velocity path above).
    src/physics/flow/AffinePeriodicFlowSolver.cu
    src/physics/stochastic/stochastic.cu
    # SF-18: standalone truly-periodic spectral Gaussian generator (cuFFT).
    src/physics/stochastic/PeriodicGaussianField.cu
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
    ${CONSTRAINT_SOURCES}
    ${OPERATOR_SOURCES}
    ${INTERPOLATION_SOURCES}
    ${STREAMLINE_TRACKER_SOURCES}
    ${INLET_SLAB_SOURCES}
    ${LAGRANGIAN_LABELS_SOURCES}
    ${MULTIGRID_SOURCES}
    ${PHYSICS_SOURCES}
    ${IO_SOURCES}
)

target_include_directories(macroflow3d_lib PUBLIC ${CMAKE_SOURCE_DIR})

# Link dependencies — curand is needed by stochastic.cu (in the lib)
# CUDA::cufft is needed by PeriodicGaussianField.cu (SF-18 periodic Gaussian
# generator); scoped here because macroflow3d_lib is the only target that
# owns this implementation.
target_link_libraries(macroflow3d_lib
    PUBLIC  CUDA::cudart CUDA::cublas CUDA::curand CUDA::cufft
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

    # SF-30 N3: e_v(h) driver on the frozen periodic streamfunction stack (documented experiment tool; not a ctest entry)
    add_executable(streamfunction_ev_ladder apps/closure_gate/ev_ladder_main.cu)
    target_link_libraries(streamfunction_ev_ladder PRIVATE macroflow3d_lib)
    target_compile_options(streamfunction_ev_ladder PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)

    add_executable(streamfunction_operator_tests
        tests/streamfunctions/reference_operators.cpp
        tests/streamfunctions/streamfunction_operator_tests.cpp
        tests/streamfunctions/streamfunction_operator_gpu_cases.cu
        tests/streamfunctions/mean_zero_projector_gpu_cases.cu
        tests/streamfunctions/projected_pcg_gpu_cases.cu
        tests/streamfunctions/multigrid_reuse_gpu_cases.cu
        tests/streamfunctions/affine_periodic_rhs_gpu_cases.cu
        tests/streamfunctions/streamfunction_gradient_gpu_cases.cu
        tests/streamfunctions/hessian_vector_b_gpu_cases.cu
        tests/streamfunctions/nonlinear_sources_gpu_cases.cu
        tests/streamfunctions/coupled_residual_gpu_cases.cu
        tests/streamfunctions/physical_diagnostics_gpu_cases.cu
        tests/streamfunctions/streamfunction_api_workspace_gpu_cases.cu
        tests/streamfunctions/streamfunction_homogeneous_gpu_cases.cu
        tests/streamfunctions/streamfunction_picard_fixed_gpu_cases.cu
        tests/streamfunctions/streamfunction_picard_adaptive_gpu_cases.cu
        tests/streamfunctions/streamfunction_continuation_gpu_cases.cu
        tests/streamfunctions/streamfunction_anderson_gpu_cases.cu
        tests/streamfunctions/heterogeneity_continuation_gpu_cases.cu
        tests/streamfunctions/jvp_gpu_cases.cu
        tests/streamfunctions/gmres_gpu_cases.cu
        tests/streamfunctions/newton_gpu_cases.cu
        tests/streamfunctions/terminal_solver_gpu_cases.cu
    )
    target_link_libraries(streamfunction_operator_tests PRIVATE macroflow3d_lib)
    target_compile_options(streamfunction_operator_tests
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
    )
    add_test(NAME streamfunction_operator_tests COMMAND streamfunction_operator_tests)
    # AUD-02: the umbrella entry above runs every case in the fast registry
    # (cases(): the no-argument default). The eleven per-group `--case` subset
    # entries that used to follow (streamfunction_exact_pair, _homogeneous,
    # _picard_fixed, _picard_adaptive, _continuation, _anderson,
    # _heterogeneity_unit, _jvp, _gmres, _newton, _terminal_unit) re-ran strict
    # subsets of it and were removed; run one group with
    # `streamfunction_operator_tests --case <name>`.
    # SF-27 (process rule: ctest holds fast contract tests only): the heavy
    # tier is deliberately NOT registered in ctest. The eight heavy cases
    #   anderson_stall_fixture_a, anderson_stall_fixture_b,
    #   heterogeneity_smoke_sigma025, heterogeneity_smoke_sigma1,
    #   newton_difficult_case, terminal_dgate_diagnostic,
    #   terminal_resolution_probe,
    #   coupled_residual_gauge_recombination_sigma025
    # remain reachable through `streamfunction_operator_tests --case <name>`
    # (heavy_cases() in tests/streamfunctions/streamfunction_operator_tests.cpp)
    # and are documented, with their detached-V100 commands and last recorded
    # outcomes, in docs/experiments/2026-10-02-heavy-streamfunction-cases-index.md.

    # SF-18 T02: periodic Gaussian generator acceptance tests + 256^3
    # memory/runtime record. Standalone (mirrors run_operator_tests.cu),
    # not part of the streamfunctions case registry.
    add_executable(periodic_gaussian_tests tests/stochastic/periodic_gaussian_tests.cu)
    target_link_libraries(periodic_gaussian_tests PRIVATE macroflow3d_lib)
    target_compile_options(periodic_gaussian_tests
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
    )
    add_test(NAME periodic_gaussian COMMAND periodic_gaussian_tests)

    # Baseline direct-sum generator (stochastic.cu): Gaussian-only contract
    # tests (decision 2026-10-07, docs/decisions/2026-10-07-gaussian-covariance-only.md).
    add_executable(stochastic_baseline_tests tests/stochastic/stochastic_baseline_tests.cu)
    target_link_libraries(stochastic_baseline_tests PRIVATE macroflow3d_lib)
    target_compile_options(stochastic_baseline_tests
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
    )
    add_test(NAME stochastic_baseline_gaussian COMMAND stochastic_baseline_tests)

    # SF-19 T02: affine-periodic Darcy solve acceptance tests
    # (AffinePeriodicFlowSolver.cuh). Standalone (mirrors
    # periodic_gaussian_tests.cu), not part of the streamfunctions case
    # registry.
    add_executable(affine_periodic_flow_tests tests/flow/affine_periodic_flow_tests.cu)
    target_link_libraries(affine_periodic_flow_tests PRIVATE macroflow3d_lib)
    target_compile_options(affine_periodic_flow_tests
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
    )
    add_test(NAME affine_periodic_flow COMMAND affine_periodic_flow_tests)

    # SF-40 L2: solve_affine_periodic_flow_lean is bit-for-bit identical to
    # solve_affine_periodic_flow (b_hat mapped host and device) with an exact S2 byte report;
    # fast mode only under ctest, `--full` (256^3, 512x64x64) is a V100 manual run.
    add_executable(affine_periodic_flow_lean_tests tests/flow/affine_periodic_flow_lean_tests.cu)
    target_link_libraries(affine_periodic_flow_lean_tests PRIVATE macroflow3d_lib)
    target_compile_options(affine_periodic_flow_lean_tests
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
    )
    add_test(NAME affine_periodic_flow_lean COMMAND affine_periodic_flow_lean_tests)

    # SF-40 L1: projected_pcg_solve_external is bit-for-bit identical to
    # projected_pcg_solve (z == Ap == MG b, b == MG x, b_hat mapped host); fast mode only
    # under ctest, `--full` (256^3, 512x64x64) is a V100 manual run.
    add_executable(projected_pcg_external_tests tests/streamfunctions/projected_pcg_external_tests.cu)
    target_link_libraries(projected_pcg_external_tests PRIVATE macroflow3d_lib)
    target_compile_options(projected_pcg_external_tests
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
    )
    add_test(NAME projected_pcg_external COMMAND projected_pcg_external_tests)

    # SF-40 L7: FlowWorkspace byte-count contract (dead residual/aux1/aux2 removed)
    add_executable(flow_workspace_bytes_tests tests/flow/flow_workspace_bytes_tests.cu)
    target_link_libraries(flow_workspace_bytes_tests PRIVATE macroflow3d_lib)
    target_compile_options(flow_workspace_bytes_tests
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
    )
    add_test(NAME flow_workspace_bytes COMMAND flow_workspace_bytes_tests)

    # SF-28 T02: periodic tricubic B-spline contract tests (fast tier)
    add_executable(interpolation_tests tests/interpolation/periodic_tricubic_bspline_tests.cu)
    target_link_libraries(interpolation_tests PRIVATE macroflow3d_lib)
    target_compile_options(interpolation_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
    add_test(NAME interpolation_periodic_tricubic COMMAND interpolation_tests)

    # SF-36 N3: spline Hessian (evaluate_point_hessian) contract tests (fast tier)
    add_executable(interpolation_hessian_tests tests/interpolation/spline_hessian_tests.cu)
    target_link_libraries(interpolation_hessian_tests PRIVATE macroflow3d_lib)
    target_compile_options(interpolation_hessian_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
    add_test(NAME interpolation_hessian COMMAND interpolation_hessian_tests)

    # SF-41 N1: non-periodic-in-x1 tricubic B-spline (threshold A program; ctest runs the fast tier,
    # the full RR3 tier is `interpolation_nonperiodic_x_tests --full` on the V100)
    add_executable(interpolation_nonperiodic_x_tests tests/interpolation/nonperiodic_x_bspline_tests.cu)
    target_link_libraries(interpolation_nonperiodic_x_tests PRIVATE macroflow3d_lib)
    target_compile_options(interpolation_nonperiodic_x_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
    add_test(NAME interpolation_nonperiodic_x COMMAND interpolation_nonperiodic_x_tests)

    # SF-31 N3: pseudo-symplectic streamline tracker contract tests (fast tier)
    add_executable(streamline_tracker_ps_tests tests/streamline_tracker/pseudo_symplectic_tracker_tests.cu)
    target_link_libraries(streamline_tracker_ps_tests PRIVATE macroflow3d_lib)
    target_compile_options(streamline_tracker_ps_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
    add_test(NAME streamline_tracker_pseudo_symplectic COMMAND streamline_tracker_ps_tests)

    # SF-31 N4: adaptive RK reference tracker contract tests + PS/RK cross-check (fast tier)
    add_executable(streamline_tracker_rk_tests tests/streamline_tracker/reference_rk_tracker_tests.cu)
    target_link_libraries(streamline_tracker_rk_tests PRIVATE macroflow3d_lib)
    target_compile_options(streamline_tracker_rk_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
    add_test(NAME streamline_tracker_reference_rk COMMAND streamline_tracker_rk_tests)

    # SF-32 N3a: Stokes face-flux (StokesFaceVelocity) contract tests S1-S7 (fast tier)
    add_executable(streamline_tracker_face_flux_tests tests/streamline_tracker/face_flux_tests.cu)
    target_link_libraries(streamline_tracker_face_flux_tests PRIVATE macroflow3d_lib)
    target_compile_options(streamline_tracker_face_flux_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
    add_test(NAME streamline_tracker_face_flux COMMAND streamline_tracker_face_flux_tests)

    # SF-32 N3b: Pollock-type (RT0) tracker contract tests P1-P6 (fast tier)
    add_executable(streamline_tracker_pollock_tests tests/streamline_tracker/pollock_tracker_tests.cu)
    target_link_libraries(streamline_tracker_pollock_tests PRIVATE macroflow3d_lib)
    target_compile_options(streamline_tracker_pollock_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
    add_test(NAME streamline_tracker_pollock COMMAND streamline_tracker_pollock_tests)

    # SF-30 N1: SF-19 potential accessor + oracle streamline integrator contracts (fast tier)
    add_executable(streamline_closure_integrator_tests tests/closure/streamline_closure_integrator_tests.cu)
    target_link_libraries(streamline_closure_integrator_tests PRIVATE macroflow3d_lib)
    target_compile_options(streamline_closure_integrator_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
    add_test(NAME streamline_closure_integrator COMMAND streamline_closure_integrator_tests)

    # SF-30 N2: streamline-closure gate executable (documented-experiment instrument)
    add_executable(closure_gate apps/closure_gate/closure_gate_main.cu)
    target_link_libraries(closure_gate PRIVATE macroflow3d_lib)
    target_compile_options(closure_gate PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)

    # SF-32 N2a/N2b: spurious-spreading return-map instrument (documented-experiment instrument; not a ctest entry).
    # The non-main translation units form a small static library shared with the controls ctest below,
    # so the test drives exactly the instrument's code (in memory, no subprocess).
    add_library(spurious_spreading_core STATIC
        apps/spurious_spreading/label_routes.cu
        apps/spurious_spreading/return_map_run.cu)
    target_link_libraries(spurious_spreading_core PUBLIC macroflow3d_lib)
    # Same device-code model as macroflow3d_lib (relocatable device code, resolved at the executables' device link).
    set_target_properties(spurious_spreading_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(spurious_spreading_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
    add_executable(spurious_spreading apps/spurious_spreading/spurious_spreading_main.cu)
    target_link_libraries(spurious_spreading PRIVATE spurious_spreading_core)
    target_compile_options(spurious_spreading PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)

    # SF-32 N2b: 16^3 controls of the spurious-spreading instrument (fast tier)
    add_executable(spurious_spreading_controls_tests tests/spurious_spreading/spurious_spreading_controls_tests.cu)
    target_link_libraries(spurious_spreading_controls_tests PRIVATE spurious_spreading_core)
    target_compile_options(spurious_spreading_controls_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
    add_test(NAME spurious_spreading_controls16 COMMAND spurious_spreading_controls_tests)

    # SF-30 N2: 16^3 controls of the closure gate (fast tier)
    add_executable(streamline_closure_controls_tests tests/closure/streamline_closure_controls_tests.cu)
    target_link_libraries(streamline_closure_controls_tests PRIVATE macroflow3d_lib)
    target_compile_options(streamline_closure_controls_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
    add_test(NAME streamline_closure_controls16 COMMAND streamline_closure_controls_tests)

    # --- SF-33 inlet slab tests: begin ---
    # SF-33 N0: inlet-slab foundation contract tests (fast tier; grids <= 32^3)
    add_executable(inlet_slab_contract_tests tests/inlet_slab/slab_contract_tests.cu)
    target_link_libraries(inlet_slab_contract_tests PRIVATE macroflow3d_lib)
    target_compile_options(inlet_slab_contract_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
    add_test(NAME inlet_slab_contracts COMMAND inlet_slab_contract_tests)
    # SF-33 N1: analytic slab Jacobian-vector product vs central FD ladder, k = 1 symbol (fast tier; N = 12)
    add_executable(inlet_slab_jvp_tests tests/inlet_slab/slab_jvp_tests.cu)
    target_link_libraries(inlet_slab_jvp_tests PRIVATE macroflow3d_lib)
    target_compile_options(inlet_slab_jvp_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
    add_test(NAME inlet_slab_jvp COMMAND inlet_slab_jvp_tests)
    # SF-33 N2: slab GMRES, per-mode preconditioner P-A, Newton line search, continuation (fast tier; N <= 16)
    add_executable(inlet_slab_newton_tests tests/inlet_slab/slab_newton_tests.cu)
    target_link_libraries(inlet_slab_newton_tests PRIVATE macroflow3d_lib)
    target_compile_options(inlet_slab_newton_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
    add_test(NAME inlet_slab_newton COMMAND inlet_slab_newton_tests)
    # SF-33 N7c (probe): Galerkin coarse-space correction (N = 12 contract checks)
    add_executable(inlet_slab_coarse_tests tests/inlet_slab/slab_coarse_tests.cu)
    target_link_libraries(inlet_slab_coarse_tests PRIVATE macroflow3d_lib)
    target_compile_options(inlet_slab_coarse_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
    add_test(NAME inlet_slab_coarse COMMAND inlet_slab_coarse_tests)
    # SF-33 N3: inlet labels, production inputs and production oracle (fast tier; grids <= 16^3)
    add_executable(inlet_slab_production_tests tests/inlet_slab/slab_production_tests.cu)
    target_link_libraries(inlet_slab_production_tests PRIVATE macroflow3d_lib)
    target_compile_options(inlet_slab_production_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
    add_test(NAME inlet_slab_production16 COMMAND inlet_slab_production_tests)
    # SF-35 N1: label-transport constructor (wrapper of the oracle; N = 16, fast tier)
    add_executable(inlet_slab_trace_tests tests/inlet_slab/slab_trace_tests.cu)
    target_link_libraries(inlet_slab_trace_tests PRIVATE macroflow3d_lib)
    target_compile_options(inlet_slab_trace_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
    add_test(NAME inlet_slab_trace16 COMMAND inlet_slab_trace_tests)
    # SF-35 N2: verification-without-tracing controls and per-plane Gate 3A metrics (fast tier; 16^3)
    add_executable(inlet_slab_controls_tests tests/inlet_slab/slab_controls_tests.cu)
    target_link_libraries(inlet_slab_controls_tests PRIVATE macroflow3d_lib)
    target_compile_options(inlet_slab_controls_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
    add_test(NAME inlet_slab_controls COMMAND inlet_slab_controls_tests)
    # SF-35 N6: GPU tracer vs host tracer (N = 16/32; <= 60 s on V100)
    add_executable(inlet_slab_trace_gpu_tests tests/inlet_slab/slab_trace_gpu_tests.cu)
    target_link_libraries(inlet_slab_trace_gpu_tests PRIVATE macroflow3d_lib)
    target_compile_options(inlet_slab_trace_gpu_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
    add_test(NAME inlet_slab_trace_gpu COMMAND inlet_slab_trace_gpu_tests)

    # SF-36 N4: forward Lagrangian tracker vs host trace_to_plane + determinism (the test TU
    # instantiates the device templates, hence --fmad=false like the library TUs that do)
    add_executable(lagrangian_gpu_host_tests tests/lagrangian_labels/lagrangian_gpu_host_tests.cu)
    target_link_libraries(lagrangian_gpu_host_tests PRIVATE macroflow3d_lib)
    target_compile_options(lagrangian_gpu_host_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda --fmad=false>)
    add_test(NAME lagrangian_tracker_gpu_host COMMAND lagrangian_gpu_host_tests)
    # SF-36 N6: exact-control tests of the forward tracker (A1-A3, B, D, E + diagnostics; fast tier
    # <= 60 s; --full is the V100 mode). Instantiates the device templates: --fmad=false.
    add_executable(lagrangian_controls_tests tests/lagrangian_labels/lagrangian_controls_tests.cu)
    target_link_libraries(lagrangian_controls_tests PRIVATE macroflow3d_lib)
    target_compile_options(lagrangian_controls_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda --fmad=false>)
    add_test(NAME lagrangian_tracker_controls COMMAND lagrangian_controls_tests)
    # SF-40 L6: opt-in tracker speedups (fused two-spline evaluation, single launch) bitwise vs the
    # default path (fast tier <= 60 s). Includes the fused header -> --fmad=false.
    add_executable(tracker_l6_tests tests/lagrangian_labels/tracker_l6_tests.cu)
    target_link_libraries(tracker_l6_tests PRIVATE macroflow3d_lib)
    target_compile_options(tracker_l6_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda --fmad=false>)
    add_test(NAME lagrangian_tracker_l6 COMMAND tracker_l6_tests)
    # SF-40 L3: LongBoxStage + tracker device-coefficient API: bitwise vs build_production_stage on cubes,
    # memory records (C-MEMPHASE), F1-F4 controls on non-cubic boxes (fast subset <= 60 s; --full and
    # --cufft-size are the V100 modes). Includes the tracker API only -> --fmad=false like its siblings.
    add_executable(long_box_tests tests/lagrangian_labels/long_box_tests.cu)
    target_link_libraries(long_box_tests PRIVATE macroflow3d_lib)
    target_compile_options(long_box_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda --fmad=false>)
    add_test(NAME lagrangian_long_box COMMAND long_box_tests)

    # SF-41 N1b: P-CROSS (threshold B, RR4): non-periodic-x1 vs periodic spline on the SF-19 h_tilde and
    # ln K samples (fast tier = smoke at 128x16x16; `--only B` is the V100 RR4 configuration).
    add_executable(sf41_cross_representation_tests tests/lagrangian_labels/sf41_cross_representation_tests.cu)
    target_link_libraries(sf41_cross_representation_tests PRIVATE macroflow3d_lib)
    target_compile_options(sf41_cross_representation_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda --fmad=false>)
    add_test(NAME lagrangian_cross_representation COMMAND sf41_cross_representation_tests)

    # SF-38 N2: DirichletMedium (medium (b), C-DIRICHLET) + EulerianFaceStats (E1): k = 1, layered exact
    # 1-D flux, periodic repack, workspace release, E1 exact (fast <= 60 s; --full is the V100 mode).
    add_executable(dirichlet_medium_tests tests/lagrangian_labels/dirichlet_medium_tests.cu)
    target_link_libraries(dirichlet_medium_tests PRIVATE macroflow3d_lib)
    target_compile_options(dirichlet_medium_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda --fmad=false>)
    add_test(NAME lagrangian_dirichlet_medium COMMAND dirichlet_medium_tests)

    # SF-41 N2: lean medium (b) stage (C-STAGE flow part): bitwise vs DirichletMedium, C5 solver controls,
    # exact-bytes closed form, external-vector PCG (fast tier <= 32^3 cells; --only D / --only C5 are the V100 modes).
    add_executable(dirichlet_long_box_stage_tests tests/lagrangian_labels/dirichlet_long_box_stage_tests.cu)
    target_link_libraries(dirichlet_long_box_stage_tests PRIVATE macroflow3d_lib)
    target_compile_options(dirichlet_long_box_stage_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda --fmad=false>)
    add_test(NAME lagrangian_dirichlet_long_box_stage COMMAND dirichlet_long_box_stage_tests)

    # SF-37 N3: CompactMAC -> periodic face repack (bitwise), Pollock lattice tracker on k = 1 and a smooth K
    add_executable(face_flux_repack_tests tests/lagrangian_labels/face_flux_repack_tests.cu)
    target_link_libraries(face_flux_repack_tests PRIVATE macroflow3d_lib)
    target_compile_options(face_flux_repack_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda --fmad=false>)
    add_test(NAME lagrangian_face_flux_repack COMMAND face_flux_repack_tests)
    # SF-41 N3a: Pollock core and comparator non-periodic in x (exit_inlet / exit_outlet), controls C4a/C4b (FAST
    # tier; `--only C4` is the V100 form of RR5, not a ctest entry)
    add_executable(pollock_nonperiodic_tests tests/lagrangian_labels/pollock_nonperiodic_tests.cu)
    target_link_libraries(pollock_nonperiodic_tests PRIVATE macroflow3d_lib)
    target_compile_options(pollock_nonperiodic_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda --fmad=false>)
    add_test(NAME lagrangian_pollock_nonperiodic COMMAND pollock_nonperiodic_tests)
    # SF-41 N3b: exact controls C1-C4 of the Lagrangian consumer on medium (b) (non-periodic-in-x1 spline, x-exit
    # handling of the tracker and of the Pollock comparator), heads written into the sample array, no solver
    # (FAST tier <= 32^3, well under 60 s; `--full` is the RR5 form for the V100, not a ctest entry)
    add_executable(dirichlet_controls_tests tests/lagrangian_labels/dirichlet_controls_tests.cu)
    target_link_libraries(dirichlet_controls_tests PRIVATE macroflow3d_lib)
    target_compile_options(dirichlet_controls_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda --fmad=false>)
    add_test(NAME lagrangian_dirichlet_controls COMMAND dirichlet_controls_tests)
    # SF-38 N3: ensemble runner (determinism, restart after a simulated crash, CSV round trip, twin, seed-face
    # assertion, field-only) on a 64x16x16 box; drives the lagrangian_ensemble and macroflow3d_pipeline binaries
    # (build both targets first). Fast tier <= 60 s.
    add_executable(lagrangian_ensemble_tests tests/lagrangian_labels/lagrangian_ensemble_tests.cu)
    target_link_libraries(lagrangian_ensemble_tests PRIVATE macroflow3d_lib)
    target_compile_options(lagrangian_ensemble_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda --fmad=false>)
    add_test(NAME lagrangian_ensemble
        COMMAND lagrangian_ensemble_tests $<TARGET_FILE:lagrangian_ensemble> $<TARGET_FILE:macroflow3d_pipeline>)
    # SF-41 N5: the same runner with --medium b (lean Dirichlet stage, exit-aware tracker and Pollock bridge,
    # multi-pass h_max, flow_stats.json) on a 128x16x16 box; also unit-tests the medium (b) auto rules and the
    # flow statistics. Links the library (in-process stage rebuild for the independent flow-stats check).
    add_executable(lagrangian_ensemble_b_tests tests/lagrangian_labels/lagrangian_ensemble_b_tests.cu)
    target_link_libraries(lagrangian_ensemble_b_tests PRIVATE macroflow3d_lib)
    target_compile_options(lagrangian_ensemble_b_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda --fmad=false>)
    add_test(NAME lagrangian_ensemble_medium_b
        COMMAND lagrangian_ensemble_b_tests $<TARGET_FILE:lagrangian_ensemble>)
    # SF-38 N4: medium A/B application (determinism, crash/restart, k = 1 equivalence, primary statistics
    # recomputed independently, ab_primary.csv, usage/seed-face exits) on a 128x16x16 box. Fast tier <= 60 s.
    add_executable(lagrangian_medium_ab_tests tests/lagrangian_labels/lagrangian_medium_ab_tests.cu)
    target_link_libraries(lagrangian_medium_ab_tests PRIVATE macroflow3d_lib)
    target_compile_options(lagrangian_medium_ab_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda --fmad=false>)
    add_test(NAME lagrangian_medium_ab
        COMMAND lagrangian_medium_ab_tests $<TARGET_FILE:lagrangian_medium_ab>)
    # SF-36 N2: ProductionStageOptions::require_inlet_labels / flow_ready (fast tier; N = 16)
    add_executable(inlet_slab_stage_options_tests tests/inlet_slab/slab_stage_options_tests.cu)
    target_link_libraries(inlet_slab_stage_options_tests PRIVATE macroflow3d_lib)
    target_compile_options(inlet_slab_stage_options_tests PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)
    add_test(NAME inlet_slab_stage_options COMMAND inlet_slab_stage_options_tests)
    # SF-33 N4: .npy reader/writer and prototype-case loader (fast tier; synthetic N = 8 case)
    add_executable(inlet_slab_npy_tests tests/inlet_slab/slab_npy_tests.cu)
    target_link_libraries(inlet_slab_npy_tests PRIVATE macroflow3d_lib)
    target_compile_definitions(inlet_slab_npy_tests PRIVATE
        INLET_SLAB_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/inlet_slab/fixtures")
    add_test(NAME inlet_slab_npy COMMAND inlet_slab_npy_tests)
    # SF-33 N4: prototype cross-check on a Python export (built, NOT a ctest entry: needs
    # docs/experiments/artifacts/2026-10-06-sf33-gpu-inlet-labels/scripts/export_proto.py output)
    add_executable(inlet_slab_proto_check tests/inlet_slab/slab_proto_check.cu)
    target_link_libraries(inlet_slab_proto_check PRIVATE macroflow3d_lib)
    # --- SF-33 inlet slab tests: end ---

    # SF-33 N5: inlet-slab experiment driver (documented-experiment instrument, NOT a ctest entry):
    # --proto (SF-29 prototype reproduction), --production (SF-18/SF-19 ladder + oracle),
    # --sf19-crosscheck (step 8). See docs/experiments/artifacts/2026-10-06-sf33-gpu-inlet-labels/.
    add_executable(inlet_slab apps/inlet_slab/inlet_slab_main.cu)
    target_link_libraries(inlet_slab PRIVATE macroflow3d_lib)
    target_compile_options(inlet_slab PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)

    # SF-38 N3: Lagrangian ensemble runner on the long box (medium (a), contract C-RUNNER); production
    # instrument (V100 jobs), NOT a ctest entry itself. Its fast ctest drives it as a subprocess.
    add_executable(lagrangian_ensemble apps/lagrangian_ensemble/lagrangian_ensemble_main.cu)
    target_link_libraries(lagrangian_ensemble PRIVATE macroflow3d_lib)
    target_compile_options(lagrangian_ensemble PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda --fmad=false>)
    # SF-38 D17: realized ln K / K statistics of the Par2 baseline fields (documented-experiment tool; not a ctest entry)
    add_executable(par2_field_stats apps/par2_field_stats/par2_field_stats_main.cu)
    target_link_libraries(par2_field_stats PRIVATE macroflow3d_lib)
    target_compile_options(par2_field_stats PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)

    # SF-38 N4: medium A/B application (contract C-AB): affine-periodic (a) vs Dirichlet-in-x (b) on the same K;
    # production instrument (V100 jobs); its fast ctest drives it as a subprocess.
    add_executable(lagrangian_medium_ab apps/lagrangian_medium_ab/lagrangian_medium_ab_main.cu)
    target_link_libraries(lagrangian_medium_ab PRIVATE macroflow3d_lib)
    target_compile_options(lagrangian_medium_ab PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda --fmad=false>)

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
    endif()
endif()
