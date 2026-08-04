# SF-02 — Discrete operator contract

- State: `validating`
- Goal: `Fijar y demostrar el contrato discreto del operador periódico A=-div(q grad).`
- Depends on: `SF-01`
- Unlocks: `SF-03`
- Branch: `science/lester-sf02-operator-contract`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf02-operator-contract`
- Acceptance gate: `Gate 1 + Gate 2`
- Human review: `required`
- Owner: `Codex (orchestrator)`
- Started: `2026-08-04T22:01Z`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Resolve the current sign/comment mismatch and prove that the reused matrix-free
operator represents the positive semidefinite periodic diffusion required by
the Lester formulation.

## Preconditions

- SF-01 reference operators and manufactured fixtures are accepted.

## In scope

- Tests and the smallest wrapper/comment corrections needed for `A(q)`.
- Harmonic face interpolation of the cell-centered coefficient `q`.

## Out of scope

- Nullspace projection, iterative solves, multigrid, and nonlinear terms.

## Files and symbols

- Inspect/extend `src/numerics/operators/VarCoeffLaplacian.*` and its wrappers.
- Add operator-contract cases to `streamfunction_operator_tests`.
- Update `src/numerics/AGENTS.md` only if a durable sign convention changes.

## Implementation specification

1. Preserve existing flow callers by adding an explicitly named positive
   wrapper if the underlying kernel remains `div(q grad)`.
2. Compute every face coefficient as the harmonic mean of `q`, not as the
   inverse of the harmonic mean of `K`.
3. Exercise triply periodic boundaries and verify the constant null mode.
4. Test constant and smooth positive `q` against the independent CPU reference.

## Expected numerical effect

No flow behavior change.  New Lester callers gain an unambiguous positive
operator contract.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_operator_tests
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- `RMS(A*1) <= 1e-13` after scale normalization.
- Symmetry defect `|x.Ay-y.Ax|/(|x.Ay|+|y.Ax|) < 1e-12`.
- Discrete energy is nonnegative to roundoff.
- Manufactured L2 convergence order is at least 1.8.

## Regression surface

- Existing flow CG/PCG sign wrappers and multigrid residual conventions.

## Failure and rollback policy

- Do not alter flow signs to satisfy a Lester test.
- If harmonic `q` conflicts with a current generic operator assumption, add a
  named coefficient policy and record the decision rather than branching
  silently on caller identity.

## Completion checklist

<!-- completion-checklist:start -->
- [x] Actual legacy sign is documented with a regression test.
- [x] Positive Lester wrapper applies `A=-div(q grad)`.
- [x] Harmonic-q face tests pass.
- [x] Nullspace, symmetry, energy, and convergence thresholds pass.
- [x] Existing flow tests and smoke pass unchanged.
- [ ] Human review and evidence are recorded.
- [ ] Dashboard marks SF-02 complete and selects SF-03.
<!-- completion-checklist:end -->

## Advancement rule

SF-03 may use this accepted operator contract to define its gauge projector.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-04T22:01Z | `0fefaa9`, active | Verified SF-01 closure on default, created the exact SF-02 runtime Goal, and created the canonical worktree. | `master=origin/master=0fefaa9`; checker passed with `next=SF-02`; SF-01 is `done`; SF-02 depends only on SF-01. | Inspect the actual operator/sign/coefficient contracts and construct the SF-02 task DAG. |
| 2026-08-04T22:08Z | `2daddc3`, active | Accepted read-only audits SF-02-T01/T02/T03 and the explicit task DAG; established the baseline before implementation. | The production kernel applies the legacy negative-semidefinite `L(c)=div_h(c grad_h)` with harmonic cell coefficient and periodic wrapping in x/y/z; `-L(q)` is the required Lester operator. Flow CG/PCG, MG, pin behavior, and PSPTA remain frozen. Baseline `cmake --preset wsl-debug`, build, 2/2 CTest, and `macroflow3d_pipeline apps/config_pspta_small.yaml` passed on RTX 3050 Laptop GPU (CC 8.6). | Implement the periodic, pin-free, explicitly positive Lester wrapper and correct only contradictory operator comments. |
| 2026-08-04T22:13Z | `6230f8c` (T04) | Added `LesterPositiveDiffusionOperator`, an explicit periodic, pin-free `A(q)=-div_h(q grad_h)` wrapper over the frozen legacy `VarCoeffLaplacian`; corrected only the contradictory sign/lifetime comments in the operator and flow solve path. | The wrapper owns no coefficient storage, delegates on the supplied CUDA stream, and has no allocation or host synchronization in `apply()`. Existing CG/PCG/MG/RHS call paths and PSPTA behavior were not changed. | Add GPU production-vs-independent-CPU contract cases. |
| 2026-08-04T22:25Z | `47f98cc` (T05) | Added the GPU contract harness, its 9 production cases, and CTest linkage to the existing SF-01 runner. | The initial implementation attempt used three overlapping builds and yielded inconclusive artifacts; recovery was a single serialized configure/build followed by the target checks. Cases cover legacy sign, constant/smooth CPU oracle including boundaries, null mode, symmetry, energy, manufactured constant/smooth convergence, and harmonic `q_f(1,4)=1.6`. | Independently integrate and validate from the canonical worktree. |
| 2026-08-04T22:30Z | T06 read-only | Reviewed CMake/CTest registration and runner CLI without adding a source change. | An initial target-only build/CTest observation was inconclusive (`CTest` reported Not Run before the target was built); the recovered complete build registered exactly 2 tests and both passed. The combined runner lists exactly 19 cases; its negative CLI contracts return rc=2. | Integrate T04 then T05 and rerun all required validation from clean HEAD. |
| 2026-08-04T22:37Z | `47f98cc`, T07 integration | Cherry-picked accepted T04 then T05 in topological order (`e5e583f -> 6230f8c -> 47f98cc`) and validated Gate 1 + Gate 2 from the canonical worktree. | `cmake --preset wsl-debug`; one serialized `cmake --build build/wsl-debug -j`; selected 9 GPU cases; all 19 runner cases; four negative CLI checks (each rc=2); targeted CTests (1/1 and 2/2) and full CTest (2/2) all passed; smoke `apps/config_pspta_small.yaml` passed: `mg_cg` 10 iterations, residual `1.02e+01 -> 1.77e-13`, PSPTA `active=387 exited=113 newton_stalls=0 nonzero_fail=0 max_fail=0`. Gate 2: `RMS(A1)` normalized `0`; symmetry `1.1721108e-16`; energy `250802.35`, relative face match `3.8529867e-18`; manufactured orders constant/smooth `1.953998/1.952429`; CPU/GPU global/boundary errors constant `4.342104e-17/5.0932152e-17`, smooth `2.5274923e-16/2.4915802e-16`; legacy/positive sign error `2.5274923e-16`; harmonic face `1.6` (relative `2.7755576e-16`). Local hardware: RTX 3050 Laptop GPU, driver 610.43.03, CC 8.6, 4096 MiB total/3749 MiB free after suite; host 15 GiB/10 GiB available; Debug, GCC 16.1.1, CUDA 13.3. Gate 3A, Gate 4, and V100 are not applicable because this increment adds only the linear operator contract and does not alter runtime science. Residual deferred risk: the pre-existing generic `PinSpec` identity-row comment/convention remains outside this pin-free wrapper and SF-02 scope. | Master audit; do not change state/checklist or advance NEXT before human review. |
| 2026-08-04T22:39Z | `5ed8712`, validating | Master audit classified SF-02 `PASS` and froze implementation changes pending human review. | Independently inspected every commit and the complete diff against `master`; re-derived the sign, `q/h^2` stencil, harmonic face placement, periodic x/y/z wraps, constant gauge mode, symmetry and face-energy identity; checked buffer ownership and absence of allocations/synchronization inside `apply`; confirmed Flow executable tokens and the CPU oracle are unchanged and no SF-03+ work is present. Re-ran checker, incremental build, all 19 cases, full CTest 2/2, and the 500-step PSPTA smoke with head residual `1.77e-13` and zero stalls/failures. Nonlinear residual, velocity reconstruction, invariance, divergence, cross-product percentiles, epsilon dependence, continuation, Gate 3A/4, and V100 are not applicable to this linear operator-contract increment. The inaccurate pre-existing generic pin comment is recorded as deferred and cannot affect the pin-free Lester wrapper. | Commit the validating state, publish the scientific PR, then mark `awaiting_review` and request mandatory human review. |
