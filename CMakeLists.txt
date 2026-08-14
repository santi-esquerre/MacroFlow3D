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
    )
    target_link_libraries(streamfunction_operator_tests PRIVATE macroflow3d_lib)
    target_compile_options(streamfunction_operator_tests
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
    )
    add_test(NAME streamfunction_operator_tests COMMAND streamfunction_operator_tests)
    add_test(NAME streamfunction_homogeneous COMMAND streamfunction_operator_tests
        --case homogeneous_solver_exact_16
        --case homogeneous_solver_exact_32
        --case homogeneous_solver_exact_64
        --case homogeneous_solver_repeated_reuse
    )
    add_test(NAME streamfunction_picard_fixed COMMAND streamfunction_operator_tests
        --case picard_fixed_manufactured_16
        --case picard_fixed_manufactured_32
        --case picard_fixed_research_amplitude_05
        --case picard_fixed_homogeneous_zero_iterations
        --case picard_fixed_config_error_contract
    )
    add_test(NAME streamfunction_picard_adaptive COMMAND streamfunction_operator_tests
        --case picard_adaptive_accept_growth
        --case picard_adaptive_reject_to_floor
        --case picard_adaptive_stagnation
        --case picard_adaptive_degeneracy_rejection
        --case picard_adaptive_fixed_mode_equivalence
        --case picard_adaptive_config_error_contract
    )
    add_test(NAME streamfunction_continuation COMMAND streamfunction_operator_tests
        --case continuation_stepper_eta_all_accept
        --case continuation_stepper_epsilon_all_accept
        --case continuation_stepper_reject_halve
        --case continuation_stepper_floor
        --case continuation_stepper_growth_streak_reset
        --case continuation_config_validation_contract
        --case continuation_warm_start_semantics
        --case continuation_injected_failure_bitwise_restore
        --case continuation_floor_exhaustion
        --case continuation_invalid_problem_propagation
        --case continuation_trig_control
        --case continuation_homogeneous_control
    )
    add_test(NAME streamfunction_anderson COMMAND streamfunction_operator_tests
        --case anderson_disabled_equivalence
        --case anderson_non_regression_trig32
        --case anderson_non_regression_homogeneous16
        --case anderson_memory_bytes
        --case anderson_config_validation
        --case anderson_condition_reset_control
        --case anderson_form_candidate_unit
    )
    # SF-20 T02: expensive STALL fixtures (a full 500-iteration 32^3 Picard
    # budget for both a control and a gate run, per fixture). Kept as a
    # separate ctest entry (not part of the default `streamfunction_anderson`
    # or bare `streamfunction_operator_tests` runs); see
    # streamfunction_anderson_gpu_cases.cu / streamfunction_operator_tests.cpp
    # (heavy_cases()) for the isolation mechanism.
    add_test(NAME streamfunction_anderson_stall COMMAND streamfunction_operator_tests
        --case anderson_stall_fixture_a
        --case anderson_stall_fixture_b
    )
    # SF-21 (first built pre-re-sequencing as SF-20 T02): CoefficientState +
    # lambda/eta-rescue contract cases, plus the SF-21 re-activation Anderson
    # entry-clear proof (fast, 16^3, host+small-GPU). Kept in a separate
    # ctest entry (rather than extended into streamfunction_continuation or
    # streamfunction_anderson) so the two very different runtime profiles
    # (fast unit cases vs. the tens-of-minutes 32^3 physical smokes below)
    # are independently attributable.
    add_test(NAME streamfunction_heterogeneity_unit COMMAND streamfunction_operator_tests
        --case heterogeneity_coefficient_state_contract
        --case heterogeneity_rescue_ordering
        --case heterogeneity_rescue_failure_lambda_halving
        --case heterogeneity_lambda_floor_exhaustion
        --case heterogeneity_anderson_entry_clear
    )
    # SF-21 (first built pre-re-sequencing as SF-20 T02, activation bitácora
    # decision 5(b); SF-21 re-activation decision R5 enables Anderson):
    # PRESPECIFIED 32^3 physical Gaussian gating smokes. May take tens of
    # minutes combined; deliberately NOT budget-capped below the solver's
    # documented defaults. Heavy tier: see streamfunction_operator_tests.cpp
    # (heavy_cases()) for the isolation mechanism.
    add_test(NAME streamfunction_heterogeneity_smoke COMMAND streamfunction_operator_tests
        --case heterogeneity_smoke_sigma025
        --case heterogeneity_smoke_sigma1
    )
    # SF-22 T02: matrix-free Jacobian-vector product (JvpWorkspace, SF-22 T01)
    # acceptance cases. Cheap tier: see jvp_gpu_cases.cu / streamfunction_
    # operator_tests.cpp (cases()) for the exact fixtures.
    add_test(NAME streamfunction_jvp COMMAND streamfunction_operator_tests
        --case jvp_delta_sweep_u_shape
        --case jvp_fd_convergence_order
        --case jvp_eta0_linearity_oracle
        --case jvp_converged_picard_base
        --case jvp_contract_fail_fast
        --case jvp_cache_and_determinism
        --case jvp_memory_accounting
        --case jvp_repeated_apply_stress
    )
    # SF-23 T02: restarted, right-preconditioned coupled GMRES (CoupledGmres,
    # SF-23 T01) and BlockDiagonalMGPreconditioner acceptance cases. Cheap
    # tier: see gmres_gpu_cases.cu / streamfunction_operator_tests.cpp
    # (cases()) for the exact fixtures.
    add_test(NAME streamfunction_gmres COMMAND streamfunction_operator_tests
        --case gmres_preconditioner_fixed_linear
        --case gmres_eta0_pcg_cross_check
        --case gmres_dense_lu_oracle
        --case gmres_iteration_reduction
        --case gmres_residual_agreement
        --case gmres_determinism
        --case gmres_memory_accounting
        --case gmres_restart15_path
        --case gmres_fail_fast_contracts
    )
    # SF-24 T03: globalized Newton-Krylov (NewtonKrylovSolver, SF-24 T01)
    # PRESPECIFIED-gate acceptance cases (G1/G2/G3/G6/G7). Cheap tier: see
    # newton_gpu_cases.cu / streamfunction_operator_tests.cpp (cases()) for
    # the exact fixtures.
    add_test(NAME streamfunction_newton COMMAND streamfunction_operator_tests
        --case newton_equivalence
        --case newton_activation_semantics
        --case newton_forced_failure_preservation
        --case newton_memory_accounting
        --case newton_fail_fast
    )
    # SF-24 T03: globalized Newton-Krylov "difficult case" acceptance control
    # (G4/G5: two 32^3 solves of the SF-21 heterogeneity-continuation stall
    # fixture, plus a determinism rerun). Kept as a separate ctest entry (not
    # part of the default `streamfunction_newton` or bare
    # `streamfunction_operator_tests` runs); see newton_gpu_cases.cu /
    # streamfunction_operator_tests.cpp (heavy_cases()) for the isolation
    # mechanism.
    add_test(NAME streamfunction_newton_difficult COMMAND streamfunction_operator_tests
        --case newton_difficult_case
    )

    # SF-18 T02: periodic Gaussian generator acceptance tests + 256^3
    # memory/runtime record. Standalone (mirrors run_operator_tests.cu),
    # not part of the streamfunctions case registry.
    add_executable(periodic_gaussian_tests tests/stochastic/periodic_gaussian_tests.cu)
    target_link_libraries(periodic_gaussian_tests PRIVATE macroflow3d_lib)
    target_compile_options(periodic_gaussian_tests
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>
    )
    add_test(NAME periodic_gaussian COMMAND periodic_gaussian_tests)

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

    # add_executable(verify_stochastic apps/verify_stochastic.cu)
    # target_link_libraries(verify_stochastic PRIVATE macroflow3d_lib)
endif()
