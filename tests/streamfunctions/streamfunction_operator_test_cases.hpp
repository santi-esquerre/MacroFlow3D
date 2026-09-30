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

// SF-20 T02 Anderson-acceleration safeguard/equivalence/non-regression/
// memory acceptance cases (cheap tier). These exercise
// `AndersonAccelerator` (SF-20 T01, `AndersonAccelerator.cuh/.cu`) and its
// `solve_streamfunctions` insertion point directly, including the disabled-
// path bitwise-equivalence contract, depth-5 non-regression on convergent
// fixtures, the exact-byte memory contract across every validated depth, the
// `config.anderson` validation contract, a hostile-`condition_limit`
// safeguard control, and two hand-checkable `form_candidate` unit tests. See
// `streamfunction_anderson_gpu_cases.cu` for the exact case list; folded into
// `streamfunction_operator_tests.cpp`'s aggregated `cases()` map exactly like
// every other registry above.
[[nodiscard]] CaseRegistry anderson_case_registry();

// SF-20 T02 Anderson-acceleration STALL-fixture acceptance cases (expensive
// tier: a full 500-iteration 32^3 Picard budget for both a control and a
// gate run, per fixture). Deliberately NOT folded into
// `streamfunction_operator_tests.cpp`'s aggregated `cases()` map (the map
// backing `--list` and the no-argument "run everything" default): reachable
// only via an explicit `--case anderson_stall_fixture_a` / `--case
// anderson_stall_fixture_b`, matching the CMakeLists.txt
// `streamfunction_anderson_stall` ctest entry. See
// `streamfunction_operator_tests.cpp`'s `heavy_cases()` lookup and
// `streamfunction_anderson_gpu_cases.cu` for the exact fixtures.
[[nodiscard]] CaseRegistry anderson_stall_case_registry();

// SF-21 (first built pre-re-sequencing as SF-20 T02) heterogeneity-
// continuation acceptance cases (cheap tier). These exercise the host+GPU
// `CoefficientState`/lambda/eta-rescue contract (bitwise/exactly against
// `ContinuationController.hpp`'s documented policy) via `StageSolveFn`
// injection, plus the SF-21 re-activation addition (decision R2a): a
// deterministic proof that `solve_streamfunctions` clears the Anderson
// history at solve entry. See `heterogeneity_continuation_gpu_cases.cu` for
// the exact case list; folded into `streamfunction_operator_tests.cpp`'s
// aggregated `cases()` map exactly like every other cheap-tier registry
// above.
[[nodiscard]] CaseRegistry heterogeneity_continuation_case_registry();

// SF-21 (first built pre-re-sequencing as SF-20 T02, decision 5(b))
// heterogeneity-continuation PHYSICAL Gaussian gating smokes (heavy tier:
// two 32^3 runs of the production `run_streamfunction_heterogeneity_
// continuation` driver, each potentially many lambda/eta-rescue stages).
// Deliberately NOT folded into `streamfunction_operator_tests.cpp`'s
// aggregated `cases()` map (the map backing `--list` and the no-argument
// "run everything" default): reachable only via an explicit `--case
// heterogeneity_smoke_sigma025` / `--case heterogeneity_smoke_sigma1`,
// matching the CMakeLists.txt `streamfunction_heterogeneity_smoke` ctest
// entry. See `streamfunction_operator_tests.cpp`'s `heavy_cases()` lookup
// and `heterogeneity_continuation_gpu_cases.cu` for the exact fixtures.
[[nodiscard]] CaseRegistry heterogeneity_continuation_smoke_case_registry();

// SF-22 T02 matrix-free Jacobian-vector product (`JvpWorkspace`, SF-22 T01
// `JacobianVectorProduct.cuh/.cu`) acceptance cases (cheap tier): the D6
// U-shaped forward/central delta-sweep study plus its 1e-5 policy gate, the
// D6-adjacent O(delta) forward-truncation evidence, the D7 eta=0 exact-
// linearity oracle against `operators::LesterPositiveDiffusionOperator`, an
// extra converged-adaptive-Picard base-state fixture, the D2/D3/D4
// fail-fast/clamp/cache/determinism contract, and the exact-byte memory
// contract. See `jvp_gpu_cases.cu` for the exact case list; folded into
// `streamfunction_operator_tests.cpp`'s aggregated `cases()` map exactly
// like every other cheap-tier registry above.
[[nodiscard]] CaseRegistry jvp_case_registry();

// SF-23 T02 restarted, right-preconditioned coupled GMRES (`CoupledGmres`,
// SF-23 T01 `CoupledGmres.cuh/.cu`) and block-diagonal MG preconditioner
// (`BlockDiagonalMGPreconditioner.cuh/.cu`) acceptance cases (cheap tier),
// incorporating C01's zero-direction/checkpoint-pairing fix. Covers the
// PRESPECIFIED E2/E3/E5/E6/E7 activation gates as amended by E9: the
// preconditioner fixedness/linearity STOP-semantics gate, an eta=0 cross-
// check against the accepted `projected_pcg_solve`, an 8^3 dense-LU column-
// by-column oracle at eta in {0,1}, a preconditioned-vs-unpreconditioned
// iteration-reduction gate, reported-vs-true residual checkpoint agreement,
// repeated-solve determinism through one reused workspace stack, the exact-
// byte memory contract, the restart=15/16 boundary, and the ordering/size/
// zero-rhs fail-fast contract. See `gmres_gpu_cases.cu` for the exact case
// list; folded into `streamfunction_operator_tests.cpp`'s aggregated
// `cases()` map exactly like every other cheap-tier registry above.
[[nodiscard]] CaseRegistry gmres_case_registry();

// SF-24 T03 globalized Newton-Krylov (`NewtonKrylovSolver`, SF-24 T01
// `NewtonKrylovSolver.cuh/.cu`) PRESPECIFIED-gate acceptance cases (cheap
// tier): G1 Newton/Picard equivalence on convergent 16^3/32^3 fixtures, G2
// activation-semantics exactness, G3 forced-failure/rescue/retry state-
// machine bitwise preservation (three variants), G6 exact-byte memory
// accounting, and G7 the C01/E10 validation fail-fast contract plus a
// never-activated homogeneous control. See `newton_gpu_cases.cu` for the
// exact case list; folded into `streamfunction_operator_tests.cpp`'s
// aggregated `cases()` map exactly like every other cheap-tier registry
// above.
[[nodiscard]] CaseRegistry newton_case_registry();

// SF-24 T03 globalized Newton-Krylov "difficult case" acceptance control
// (G4/G5, heavy tier: two 32^3 solves of the SF-21 heterogeneity-continuation
// stall fixture, one SF-20 Anderson-gate control and one +Newton run, plus a
// determinism rerun). Deliberately NOT folded into
// `streamfunction_operator_tests.cpp`'s aggregated `cases()` map (the map
// backing `--list` and the no-argument "run everything" default): reachable
// only via an explicit `--case newton_difficult_case`, matching the
// CMakeLists.txt `streamfunction_newton_difficult` ctest entry. See
// `streamfunction_operator_tests.cpp`'s `heavy_cases()` lookup and
// `newton_gpu_cases.cu` for the exact fixture.
[[nodiscard]] CaseRegistry newton_difficult_case_registry();

// SF-25 T02 D-gate diagnostic (`ShiftedJacobianOperator`, SF-25 T01
// `ShiftedJacobianOperator.cuh/.cu`) acceptance cases. `terminal_solver_
// case_registry()` (cheap tier) has one enabler unit case
// (`terminal_shifted_apply_unit`): shifted-apply exactness against an
// independently recomputed `Jp+mu*Ap`, the `mu=0` bitwise passthrough, and
// the `set_mu`/unprepared-apply fail-fast contract. See
// `terminal_solver_gpu_cases.cu` for the exact case list; folded into
// `streamfunction_operator_tests.cpp`'s aggregated `cases()` map exactly
// like every other cheap-tier registry above.
[[nodiscard]] CaseRegistry terminal_solver_case_registry();

// SF-25 T02 D-gate diagnostic PROTOCOL case (`terminal_dgate_diagnostic`,
// HEAVY tier: freezes a genuine sigma_Y^2=1, 32^3 Picard/Anderson plateau
// state, then runs the PRESPECIFIED E2-E5 mu-sweep/spectral-probe/LM
// mini-solve protocol from the SF-25 activation bitácora), plus the SF-25
// corrective C05 owner-approved experiment case
// (`terminal_resolution_probe`, HEAVY tier: two DIRECT 64^3 zero-source
// solves at the critical amplitude 0.5125*Y_unit, sigma_Y^2=1 -- R1a at
// ell/h=16 vs R1b control at ell/h=8 -- discriminating the resolution-per-
// correlation-length hypothesis; ALWAYS-PASS print-only evidence recorder,
// bitácora 2026-08-14T15:20Z). Deliberately NOT folded into
// `streamfunction_operator_tests.cpp`'s aggregated `cases()` map (the map
// backing `--list` and the no-argument "run everything" default): reachable
// only via an explicit `--case terminal_dgate_diagnostic` / `--case
// terminal_resolution_probe`, matching the CMakeLists.txt
// `streamfunction_terminal_dgate` / `streamfunction_terminal_resolution`
// ctest entries. See `streamfunction_operator_tests.cpp`'s `heavy_cases()`
// lookup and `terminal_solver_gpu_cases.cu` for the exact fixtures.
[[nodiscard]] CaseRegistry terminal_solver_dgate_case_registry();

}  // namespace macroflow3d::streamfunctions::test
