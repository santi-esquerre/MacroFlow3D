# SF-30 — Mixed-precision preconditioner study

- State: `pending`
- Goal: `Evaluar precisión mixta únicamente dentro del precondicionador.`
- Depends on: `SF-29`
- Unlocks: `none`
- Branch: `science/lester-sf-30-mixed-precision`
- Worktree: `~/src/MacroFlow3D/.agents/worktrees/lester-sf28-mixed-precision`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A + Gate 4 + performance evidence`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Determine whether a float multigrid preconditioner reduces V100 cost while all
authoritative residuals, Krylov algebra, states, and physical diagnostics remain
double precision.

## Preconditions

- SF-28 accepts the full-stack double-precision V100 baseline (Newton-Krylov included since SF-24).

## In scope

- Float copies of preconditioner coefficients/vectors, double outer iteration,
  iterative refinement as needed, FGMRES for variable preconditioning, and
  measured comparison.

## Out of scope

- Float streamfunctions, float nonlinear sources, relaxed scientific
  tolerances, or replacing the double default without evidence.

## Files and symbols

- Add a precision policy to MG/preconditioner adapters without changing current
  double callers.
- Add FGMRES storage only when the preconditioner application is variable.

## Implementation specification

1. Cast into/out of preconditioner storage at explicit boundaries.
2. Keep `F`, Jv differencing, orthogonalization, line search, and all acceptance
   metrics in double.
3. Use FGMRES when inner preconditioner work or refinement varies by iteration.
4. Compare memory, preconditioner time, total time, iterations, final fields,
   and physical metrics against the SF-28 double baseline.

## Expected numerical effect

Accepted mixed precision yields equivalent double-level outer convergence with
a measured memory or runtime benefit; otherwise it remains disabled.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction
scripts/remote run lester-mixed -- "<fixed-double-vs-mixed-suite>"
scripts/remote wait lester-mixed
```

## Acceptance thresholds

- Double and mixed final `r_F` and Gate 3A metrics agree within predeclared
  solver/spatial tolerances.
- No increase in failed realizations or continuation reductions.
- A meaningful V100 peak-memory or wall-time reduction is measured; otherwise
  mixed precision remains experimental and disabled.

## Regression surface

- MG templates/storage, conversion cost, FGMRES memory, loss of preconditioner
  quality, and hardware-dependent results.

## Failure and rollback policy

- Preserve the double path as default and correctness oracle.
- Do not loosen outer tolerances to claim a mixed-precision speedup.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Precision boundaries and FGMRES requirement are explicit.
- [ ] Double/mixed correctness suite passes.
- [ ] V100 memory, runtime, iterations, and robustness are compared.
- [ ] Default enablement decision is evidence-backed and documented.
- [ ] Gate 3A/4 review passes.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-30 complete; the execution sequence ends.
<!-- completion-checklist:end -->

## Advancement rule

After SF-29 is merged, the planned Lester solver sequence is complete.  Any
consumer integration or new scientific regime requires a new decision and
increment series.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
