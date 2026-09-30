# SF-27 — Paper-faithful explicit pseudo-time solver

- State: `pending`
- Goal: `Implementar el solver pseudo-tiempo explícito de paso variable del paper (flujo k-escalado, inicialización armónica) sobre el residuo corregido y demostrar que reproduce las soluciones del stack implícito.`
- Depends on: `SF-26`
- Unlocks: `SF-28`
- Branch: `science/lester-sf27-pseudo-time-solver`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Lester et al. (2023) §5.1 solve eq. (14) by a homogeneous (`S = 0`) Krylov
initial estimate followed by explicit variable-step pseudo-time stepping to
residual `1e-16`. This increment adds that method class as a second,
independent solver on the corrected residual, first to cross-validate the
implicit stack (same solutions), later (SF-28) to evaluate cost at the
reference resolution. The implicit stack remains the accepted solver.

## Preconditions

- SF-26 `done` (corrected pairing, contract tests, recorded gate outcome).

## In scope

- `PseudoTimeSolver.{cu,cuh}` in `src/physics/streamfunctions/`, driven by a
  `PseudoTimeConfig` (in `StreamfunctionTypes.hpp`) selected through
  `StreamfunctionSolverConfig::method = {picard, pseudo_time}` with default
  `picard` (behavior of every existing caller unchanged).
- Flow on the mean-zero fluctuations: `d u_i / d tau = -k .* F_i(u)`, where
  `F` is the SF-10 residual (`A u_i - G_i`), so the linear part is the paper's
  unit-coefficient operator `k div((1/k) grad)` and the zero set is identical
  to `F = 0`. `k` is the per-cell conductivity of the problem view.
- Integrator: explicit RK2 (Heun) with embedded Euler error estimate,
  relative tolerance `tol_step` (default `1e-3`) and a PI controller, AND a
  hard cap `dtau <= safety * 2 / lambda_max_G`, `lambda_max_G` the Gershgorin
  bound of the scaled linear operator (`max_C k_C * sum_f q_f / h^2` times 2,
  computed once per coefficient state). Rejection on nonfinite or
  `r_F_trial > growth_factor * r_F` (default 2). No servo may exceed the cap
  (the P2-A confound is impossible by construction).
- Initialization: the SF-13 `zero_source` harmonic solve (PCG+MG) — the
  paper's Krylov initial estimate; `warm_start` also allowed.
- `epsilon` default `0` for this method (paper: no regularization); the
  degeneracy counters stay diagnostic-only; `epsilon > 0` remains available.
- Stopping: `r_F <= tol` (default `1e-10`), plus the absolute `Linf(F)`
  printed for comparison with the paper's `1e-16`; step/wall budgets.
- Rolling residual reuse (one residual evaluation per accepted step, buffer
  swap) and zero allocations inside the loop; exact memory accounting.
- Report: steps, rejections, `dtau` history summary, final `r_F`, Gate 3A
  diagnostics via the SF-11 evaluator; CSV/JSON export through the SF-16
  surface with the new fields additive.

## Out of scope

- Nested grid iteration and the 256^3 case (SF-28); kernel fusion; mixed
  precision; any change to the implicit stack.

## Files and symbols

- New: `src/physics/streamfunctions/PseudoTimeSolver.cu/.cuh`,
  `tests/streamfunctions/pseudo_time_gpu_cases.cu`.
- Extend: `StreamfunctionTypes.hpp` (`PseudoTimeConfig`, `method`),
  `StreamfunctionSolver.cu` (dispatch), `StreamfunctionWorkspace.*` (buffers),
  `src/io/` config loader/validation/effective-config, `CMakeLists.txt`.
- Reuse: `enqueue_streamfunction_residual`, `synchronize_streamfunction_residual_report`,
  `operators::LesterPositiveDiffusionOperator`, `constraints::MeanZeroProjector`,
  `blas::axpy`, `DeviceBuffer::swap`; the structure of `run_explicit_flow_arm`
  in `terminal_solver_gpu_cases.cu` is the reference pattern.

## Implementation specification

1. Add `PseudoTimeConfig{tol, tol_step, safety, growth_factor, dtau0, dtau_min,
   max_steps, max_wall_s, epsilon}` with validation; `method` enum with
   `picard` default; strict YAML parsing with no silent defaults change.
2. Compute the Gershgorin cap once per `CoefficientState::rebuild`.
3. Heun step: `k1 = -k.*F(u)`, `u* = P(u + dtau k1)`, `k2 = -k.*F(u*)`,
   `u_new = P(u + dtau (k1+k2)/2)`, error `= dtau ||k2-k1||/2` (RMS,
   normalized like `r_F`); accept/reject per tolerance and growth guard; PI
   update clipped to the cap. `P` = mean-zero projection.
4. The residual at `u_new` seeds the next step (one extra evaluation per
   Heun step relative to Euler; document the count).
5. Structured exits: converged, budget, dtau floor, nonfinite; never a silent
   fallback to Picard.

## Expected numerical effect

- On the SF-26 fixtures (`sigma^2 in {0.25, 1}`, 32^3, seed 12345, `eta = 1`),
  the pseudo-time solution equals the implicit-stack solution: `||u_i^pt -
  u_i^picard||_RMS / ||u_i||_RMS <= 1e-5` at `tol = 1e-10` (both converge to the
  same isolated solution of the corrected system), and Gate 3A metrics agree to
  4 significant figures.
- Step count at 32^3: `O(1e3-1e4)` after harmonic init (cap `~h^2/12`;
  slowest-mode rate `~(2 pi / L)^2`).

## Validation commands

```bash
cmake --preset wsl-debug && cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R pseudo_time
./build/wsl-debug/streamfunction_operator_tests --case pseudo_time_homogeneous_exact
./build/wsl-debug/streamfunction_operator_tests --case pseudo_time_exact_pair
scripts/remote exec -- "ctest --test-dir build/v100-release --output-on-failure"
scripts/remote run sf27-agreement -- "./build/v100-release/streamfunction_operator_tests --case pseudo_time_vs_picard_sigma1_32"
```

## Acceptance thresholds

- Homogeneous exact control: `u = 0` is a fixed point (residual stays `<= 1e-13`).
- Exact pair (SF-26 fixture) from the harmonic init converges to `r_F` at
  truncation level with `dtau` never exceeding the cap (assert).
- Agreement with the implicit stack per "Expected numerical effect".
- Determinism (bitwise across two runs), zero allocations in the loop
  (cudaMemGetInfo unchanged across steps), exact memory accounting.
- Default pipeline byte-compares unchanged (`method = picard`).

## Regression surface

- Config parsing (new fields; strict parser), workspace accounting, SF-16
  exports (additive columns only).

## Failure and rollback policy

- Disagreement with the implicit stack beyond threshold is a recorded finding
  (both solutions and diagnostics kept), not a tolerance change.
- If the explicit cap makes 32^3 runs exceed 1 h on V100, record the cost and
  keep the method opt-in; SF-28 decides on nested iteration.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] `PseudoTimeSolver` implemented behind `method = pseudo_time`; defaults preserve every existing path bitwise.
- [ ] Gershgorin cap, embedded-pair control, and structured exits unit-tested.
- [ ] Homogeneous and exact-pair controls pass.
- [ ] Agreement with the implicit stack recorded for sigma^2 = 0.25 and 1 (32^3).
- [ ] Gate 3A metrics, memory accounting, and determinism evidence recorded; human review recorded.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-27 complete and selects SF-28.
<!-- completion-checklist:end -->

## Advancement rule

SF-28 may use both solvers on the reference case once SF-27 is merged.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-09-30T00:00Z | created by the owner-directed re-sequencing (decision 2026-09-30) | Replaces the former "SF-27 — Grid continuation" (absorbed by SF-28). Owner direction: keep the implicit stack; implement pseudo-time to evaluate correctness first, performance later. | `docs/decisions/2026-09-30-eq14-source-pairing-root-cause.md` | Activate only when named by `NEXT`. |
