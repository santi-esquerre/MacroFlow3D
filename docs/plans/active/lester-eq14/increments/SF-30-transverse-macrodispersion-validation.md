# SF-30 — Transverse macrodispersion validation

- State: `pending`
- Goal: `Demostrar en el campo de referencia que el tracker pseudo-simpléctico da macrodispersión transversal nula en advección pura mientras los trackers convencionales producen dispersión espuria con los escalamientos del paper.`
- Depends on: `SF-29`
- Unlocks: `none`
- Branch: `science/lester-sf30-transverse-macrodispersion`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 3A + Gate 4 + Gate 5`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

The project's central claim (theory note §4-6): in smooth, locally isotropic,
stagnation-free Darcy flow, purely advective transverse macrodispersion is
zero and conventional trackers manufacture it numerically. This increment
produces the controlled comparison on the reproduced reference field and
interprets every observed transverse growth explicitly as physical,
numerical, or unresolved (Gate 4).

## Preconditions

- SF-29 `done`.

## In scope

- Ensemble runs (random re-injection protocol over multiple domain lengths,
  flux-weighted and uniform injection) with the three trackers; time series
  `sigma^2_x2(t)`, `sigma^2_x3(t)`, `alpha_L(t)`, `alpha_T(t)` through the
  existing analysis path.
- Pseudo-symplectic: bounded transverse variance (no linear growth);
  references: growth rates vs tolerance / grid per paper eq. (36) and Figs. 3-4.
- Optional (recorded separately if attempted): local dispersion via Langevin
  noise and `D_T(Pe)` vs paper eqs. (66)-(69), Fig. 5.
- Experiment note with the Gate 5 comparison protocol (covariance, `sigma^2`,
  BCs, injection, tracking method, asymptotic estimation stated explicitly).

## Out of scope

- Exponential covariance, tensor conductivity, block-scale upscaling.

## Files and symbols

- `src/runtime/ensemble/EnsembleRunner.*`, `src/runtime/analysis/AnalysisRunner.*`
  (adapter usage; additive outputs), `apps/config_macrodispersion_reference_*.yaml`,
  `docs/experiments/`.

## Implementation specification

1. Wire the SF-29 adapter into the ensemble/analysis runners without changing
   the Par2 baseline outputs (byte-compare).
2. Run the prespecified matrix (trackers x injection x tolerance/grid levels).
3. Write the experiment note with the Gate 4 statement per observed growth.

## Expected numerical effect

- Pseudo-symplectic `alpha_T -> 0` (bounded `sigma^2_T`); RK45 `sigma^2_T`
  growth `~ sqrt(tol) * t`; Pollock-like `~ Delta^2 * t`.

## Validation commands

```bash
scripts/remote run sf30-ensemble -- "./build/v100-release/macroflow3d_pipeline apps/config_macrodispersion_reference_pseudosymplectic.yaml"
scripts/remote run sf30-rk -- "./build/v100-release/macroflow3d_pipeline apps/config_macrodispersion_reference_rk45.yaml"
```

## Acceptance thresholds

- Pseudo-symplectic: asymptotic slope of `sigma^2_T(t)` consistent with zero
  within the ensemble standard error; references: slopes scale with the
  expected exponents within a factor 2.
- Par2 baseline outputs byte-identical to pre-increment references.

## Regression surface

- Ensemble/analysis outputs for existing configs.

## Failure and rollback policy

- Any positive transverse growth from the pseudo-symplectic tracker is a
  recorded finding requiring its numerical origin to be identified before any
  physical interpretation (Gate 4 rule).

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Ensemble/analysis wiring with baseline byte-compares clean.
- [ ] Prespecified comparison matrix run and recorded.
- [ ] Gate 4 interpretation written for every observed transverse growth.
- [ ] Gate 5 comparison protocol recorded in the experiment note.
- [ ] Human review recorded.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-30 complete and sets `NEXT` to `COMPLETE` or the next planned increment.
<!-- completion-checklist:end -->

## Advancement rule

Performance work (kernel fusion, V100 benchmark, mixed precision — the former
SF-28..SF-30 specifications at `9a71963`) may be re-planned as new increments
after this one is merged.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-09-30T00:00Z | created by the owner-directed re-sequencing (decision 2026-09-30) | Replaces the former "SF-30 — Mixed-precision preconditioner study" (deferred). | `docs/decisions/2026-09-30-eq14-source-pairing-root-cause.md` | Activate only when named by `NEXT`. |
