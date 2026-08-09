#pragma once

#include <map>
#include <string>

namespace macroflow3d::streamfunctions::test {

struct CaseResult {
    bool pass{};
    std::string name;
    std::string kind;
    std::string grid;
    double coarse_norm{};
    double fine_norm{};
    std::string expected_order;
    std::string observed_order{"n/a"};
    std::string threshold;
};

using CaseFunction = CaseResult (*)();
using CaseRegistry = std::map<std::string, CaseFunction>;

// GPU production-operator cases. The CPU-only runner owns the combined CLI.
[[nodiscard]] CaseRegistry gpu_case_registry();

// GPU mean-zero projector cases.  Kept separate from the SF-02 operator
// cases so the projector's workspace and stream contract remains explicit.
[[nodiscard]] CaseRegistry mean_zero_projector_case_registry();

// GPU projected-PCG manufactured controls.  Their RHS and residual oracle are
// explicit CPU long-double stencils, kept separate from the GPU operator.
[[nodiscard]] CaseRegistry projected_pcg_case_registry();

// SF-05 quantitative acceptance controls for the reused, projected positive
// multigrid hierarchy.  Kept separate so their CPU long-double oracle remains
// visibly independent from the production kernels.
[[nodiscard]] CaseRegistry multigrid_reuse_case_registry();

// SF-06 affine-periodic RHS acceptance controls.  Kept separate so their
// independent long-double oracle cannot accidentally become a production API.
[[nodiscard]] CaseRegistry affine_periodic_rhs_case_registry();

// SF-07 positive total-gradient acceptance controls.  These retain an
// independent long-double CPU stencil and deliberately unequal spacing.
[[nodiscard]] CaseRegistry streamfunction_gradient_case_registry();

// SF-08 positive Hessian-vector/B controls.  They deliberately run the SF-07
// total-gradient kernel immediately before the register-only HVP/B kernel.
[[nodiscard]] CaseRegistry hessian_vector_b_case_registry();

// SF-09 positive nonlinear-source (`c`, `S1`, `S2`, explicit denominator
// regularization) acceptance controls. They deliberately run the full SF-07
// gradient -> SF-08 Hessian-vector/B -> SF-09 source GPU chain before
// comparing against an independent long-double CPU oracle.
[[nodiscard]] CaseRegistry nonlinear_sources_case_registry();

// SF-10 coupled residual and dimensionless-reduction acceptance controls.
// These run the production `ResidualEvaluator` (which itself composes the
// accepted SF-02/06/07/08/09 modules) and compare against the independent
// `coupled_residual_reference` CPU oracle, including reductions, the |c|
// histogram, and its percentile helper.
[[nodiscard]] CaseRegistry coupled_residual_case_registry();

// SF-11 CompactMAC velocity reconstruction and physical-diagnostics
// acceptance controls. These run the production
// `enqueue_streamfunction_physical_diagnostics` /
// `synchronize_streamfunction_physical_diagnostics_report` chain and compare
// against the independent `physical_diagnostics_mirror` CPU oracle,
// including exact-count angle/degeneracy agreement and convergence order.
[[nodiscard]] CaseRegistry physical_diagnostics_case_registry();

// SF-12 T02 public API/workspace acceptance controls. These run the
// production StreamfunctionTypes.hpp/StreamfunctionWorkspace.cuh surface
// (validation, the owned StreamfunctionFields/StreamfunctionWorkspace pair,
// the exact-byte memory estimator, allocation-freedom across repeated use,
// and re-preparation semantics) against an independent closed-form host
// reconstruction. `solve_streamfunctions` is declaration-only in SF-12 and is
// never referenced here.
[[nodiscard]] CaseRegistry api_workspace_case_registry();

// SF-13 T02 homogeneous (K=1) exact-control acceptance cases. These run the
// production `solve_streamfunctions` entry point (SF-13 T01) on the triply
// periodic unit torus with constant conductivity and the benchmark gauge,
// where the zero-source linear path, S1/S2, and every SF-11 physical
// diagnostic are exact in continuous arithmetic, and assert hard,
// non-relaxed floating-point-roundoff thresholds.
[[nodiscard]] CaseRegistry homogeneous_solver_case_registry();

// SF-14 T02 fixed-relaxation Picard manufactured-nonlinear acceptance cases.
// These run the production `solve_streamfunctions` entry point (SF-14 T01
// fixed-relaxation Picard loop) on the triply periodic unit torus with a
// manufactured heterogeneous conductivity field, asserting the documented
// `PicardIterationRecord`/history-layout semantics and (for the two gating
// cases) convergence within the configured iteration budget.
[[nodiscard]] CaseRegistry picard_fixed_case_registry();

// SF-15 T02 deterministic adaptive-branch acceptance cases. These run the
// production `solve_streamfunctions` entry point (SF-15 T01 adaptive-Picard
// globalization) on the triply periodic unit torus with the same
// manufactured heterogeneous conductivity field, forcing each documented
// adaptive branch (accept+growth, floor-rejected backtracking, stagnation,
// degeneracy-guard rejection, disabled-mode SF-14 equivalence, and the
// `config.adaptive` validation contract) with extreme-but-valid
// configuration values.
[[nodiscard]] CaseRegistry picard_adaptive_case_registry();

// SF-17 T02 adaptive eta/epsilon continuation-with-rollback acceptance
// cases. These exercise the host-only `ContinuationStepper`/
// `validate_streamfunction_continuation_config` contract (bitwise/exactly
// against ContinuationController.hpp's documented policy) and the
// production `run_streamfunction_continuation` driver on GPU fixtures
// mirrored from the SF-14/15 Picard acceptance cases (homogeneous K=1
// controls, rollback/floor/invalid_problem injection via `StageSolveFn`,
// and the a=0.5 trigonometric headline control).
[[nodiscard]] CaseRegistry streamfunction_continuation_case_registry();

}  // namespace macroflow3d::streamfunctions::test
