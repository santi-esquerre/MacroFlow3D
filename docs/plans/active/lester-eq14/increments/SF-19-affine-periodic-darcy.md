# SF-19 — Affine-periodic Darcy solve

- State: `awaiting_review`
- Goal: `Resolver el flujo Darcy affine-periodic necesario para el benchmark triplemente periódico.`
- Depends on: `SF-18`
- Unlocks: `SF-20`
- Branch: `science/lester-sf19-affine-periodic-darcy`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A prerequisite + Gate 4`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-10T15:33Z`
- Completed: `not completed`
- PR: [#31 — SF-19: affine-periodic Darcy solve — homogenization cell problems, effective tensor, prescribed mean flux, mass-conservative CompactMAC velocity](https://github.com/santi-esquerre/MacroFlow3D/pull/31)
- Commit: `cfcc2e13bdb0650bc3ed6a55378914a4e98d1b04 (frozen audited source head)`

## Scientific or engineering intent

Produce a mass-conservative reference Darcy field with triply periodic
fluctuation and a controlled mean flux, rather than applying incompatible
Dirichlet head boundaries.

## Preconditions

- SF-18 supplies accepted periodic scalar conductivity fields.

## In scope

- Periodic affine-head cell problems, effective conductivity tensor, prescribed
  mean-flux solve, and affine-aware CompactMAC velocity reconstruction.

## Out of scope

- Changing the existing Dirichlet flow path or invoking the streamfunction
  nonlinear solver.

## Files and symbols

- Add `src/physics/flow/AffinePeriodicFlowSolver.cuh/.cu` and an affine velocity
  reconstruction overload.
- Reuse projected `A(K)`, PCG, and MG through explicit sign/coefficient adapters.

## Implementation specification

1. Solve three zero-mean periodic head-corrector cell problems for unit mean
   pressure gradients.
2. Integrate their face fluxes to construct the 3x3 effective conductivity.
3. Solve the small host 3x3 system for a pressure gradient producing target
   mean Darcy flux `(1,0,0)`.
4. Recombine correctors and reconstruct final CompactMAC velocity with affine
   pressure contribution and harmonic `K` faces.

## Expected numerical effect

The reference velocity is periodic, discretely mass conservative, and has the
specified mean flux even when a finite realization has transverse effective
coupling.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R affine_periodic_flow
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- `K=1` gives exact identity effective conductivity and uniform target flux.
- Effective tensor symmetry defect is below `1e-10` on test fields and all
  eigenvalues are positive.
- Mean flux error is below `1e-10` relative; mass residual meets flow tolerance.

## Regression surface

- Flow operator signs, harmonic-K velocity faces, periodic gauges, and existing
  head-solver APIs.

## Failure and rollback policy

- Do not force transverse mean flux to zero by discarding tensor coupling.
- Do not alter the current boundary-driven flow solver; keep this a separate
  explicit entry point.

## Completion checklist

<!-- completion-checklist:start -->
- [x] Three cell problems and effective tensor are implemented.
- [x] Target mean-flux solve and affine velocity reconstruction are implemented.
- [x] Homogeneous, symmetry/SPD, flux, and mass tests pass.
- [x] Existing flow cases remain unchanged.
- [ ] Gate 4 interpretation and human review are recorded.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-19 complete and selects SF-20.
<!-- completion-checklist:end -->

## Advancement rule

SF-20 may combine accepted periodic fields and Darcy flow with lambda
continuation in the streamfunction solver.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-10T15:33Z | activation on `master=3d98691` (SF-18 closure merged via PR #30) | SF-19 activated after verifying `NEXT: SF-19`, SF-18 `done`, and checker `OK (29 increments, next=SF-19)` on the default branch. Interpretive decisions recorded for the human reviewer: (1) **Mathematical convention (homogenization cell problems):** head `h = -G.x + (periodic corrector)`; for each unit mean pressure-gradient direction `e_d` solve the zero-mean periodic corrector `div(K(grad w_d + e_d)) = 0`, i.e. `A_K w_d = div(K e_d)` with `A_K u = -div(K grad u)` — EXACTLY the accepted SF-02..05 operator/projected-PCG/MG stack with the coefficient array filled with `K` itself (the Lester path fills `q = 1/K`; this coefficient swap IS the spec's "explicit sign/coefficient adapters", and the sign identity `A w = div(K e_d)` matches the accepted SF-06 affine-RHS form `A u = div(coeff*gbar)`). The RHS uses the SAME harmonic face-coefficient convention as the operator (`2 K_C K_N/(K_C+K_N)`) — REQUIRED so discrete mass conservation reduces to the linear-solve residual; the SF-06 pairwise assembler may be adapted per-direction (small dedicated kernel or gauge-pair calls), worker's choice, same face convention mandatory. (2) **Effective conductivity:** `K_eff` column `d` = domain average of the face fluxes `K_f (grad w_d + e_d)` over the unique periodic faces; by discrete summation-by-parts this equals the symmetric energy form up to the corrector residuals, so the spec's `1e-10` symmetry-defect threshold is REL to `max|K_eff|` and is attainable iff the correctors are solved tightly — the acceptance tests therefore PRESPECIFY corrector `rtol = 1e-11` (library default stays configurable). (3) **Mean-flux solve:** solve the full 3x3 host system `K_eff G = qbar` with `qbar = (1,0,0)` (Cramer/LU with SPD/positivity checks); NEVER discard transverse coupling (spec rollback rule) — a finite realization's transverse mean flux is honored through the full tensor. (4) **Velocity reconstruction:** affine-aware CompactMAC overload; face flux `F_f = K_f (G + sum_d G_d grad w_d).n` with the same harmonic `K_f`; discrete `div F` per cell equals the recombined corrector residual — reported as the mass diagnostic. (5) **Scope:** new `src/physics/flow/AffinePeriodicFlowSolver.cuh/.cu` standalone entry point; the existing Dirichlet head solver, velocity path, and every existing config remain byte-untouched (final audit re-verifies at artifact level); no pipeline/YAML wiring in SF-19 (consumption is SF-20+). (6) **PRESPECIFIED acceptance fixtures and thresholds (fixed NOW, before any implementation):** (a) `K = 1` (16^3 and 32^3): correctors exactly zero (0 PCG iterations on zero RHS), `K_eff = I` exact to fp roundoff (<= 1e-14 abs), `G = (1,0,0)` exact, uniform velocity `(1,0,0)` exact, `div F = 0` exact. (b) Deterministic trig `K = exp(0.5 sin(2pi x/L) sin(2pi y/L) sin(2pi z/L))` at 32^3: symmetry defect `max|K_eff - K_eff^T| / max|K_eff| <= 1e-10`; all eigenvalues `> 0`; achieved mean flux `max_i |<F>_i - qbar_i| <= 1e-10` (qbar=(1,0,0)); mass `max_cells |div F| <= 1e-8` (velocity scale 1). (c) SF-18 periodic Gaussian field (64^3, dx=1, l=8, sigma2=1, seed 12345, normalize_variance=true, K = exp(Y)): same four thresholds as (b). (d) Reproducibility: repeated solve on the same field is bitwise-identical (deterministic stack). Corrector solves in tests use `rtol=1e-11`, `max_iter=2000`, MG defaults. (7) **Gate 4 interpretation (recorded, no transport claims):** this increment only CONSTRUCTS the smooth, locally isotropic, triply periodic reference Darcy field with controlled mean flux — the regime where Lester theory forbids purely advective transverse macrodispersion; no transport or dispersion claims are made or implied here, and nothing in this increment may later be cited as evidence about transverse dispersion without the Gate-4 controls of the consuming increments. | Base commit is this activation commit on `master=3d98691`. Gate 1 + Gate 2 + Gate 3A prerequisite + Gate 4 (interpretation recorded) apply; human review required, so the PR will stop at `awaiting_review` with `NEXT` unchanged. | Build intra-increment DAG; delegate implementation to isolated worker worktrees. |
| 2026-08-10T14:45Z | `cfcc2e1`, integration validation | Two-node DAG completed and orchestrator-audited node by node, zero corrective cycles. T01 `0802b7b`: `AffinePeriodicFlowSolver.cuh/.cu` — three zero-mean periodic corrector cell problems `A_K w_d = P(div(K e_d))` on the accepted operator/projected-PCG/MG stack with the coefficient array = K (decision-1 adapter, explicitly commented), RHS via the UNMODIFIED SF-06 assembler with a degenerate `(e_d,e_d)` gauge pair (harmonic-face identity by construction), face-flux kernels using the operator's own `harmonic_mean_positive_cell_coefficient`, RAW unique-face `K_eff` (excludes the duplicated wrap plane; symmetry check is a genuine measurement, not energy-form-symmetrized), symmetry/SPD evidence computed BEFORE the full-3x3 Cramer solve on the UNSYMMETRIZED tensor (transverse coupling never discarded), recombined `h_tilde = sum G_d w_d`, affine CompactMAC velocity + per-cell divergence diagnostics, exact-byte memory report, Gate-4 note in the header; existing flow files byte-untouched. T02 `cfcc2e1`: `affine_periodic_flow` ctest entry implementing the PRESPECIFIED decision-6 fixtures/thresholds verbatim (rtol=1e-11 correctors), 37/37 checks green with no adjustment. Single integrator: trivial fast-forward (both approved commits preserved verbatim), 4 files +1722, zero conflicts, full validation green. | Acceptance evidence: K=1 (16^3, 32^3) all EXACT — 0-iteration correctors, K_eff=I (max dev 0.0), G=(1,0,0), uniform velocity, div=0.0. Trig 32^3: K_eff=1.004972*I, symmetry defect 3.7e-20 (<=1e-10), eigenvalues 1.004972 (>0), flux error 0.0 (<=1e-10), div_max 3.5e-15 (<=1e-8), 10 iters/direction — **effective-medium cross-check: exp(sigma^2/6)=1.0052 predicted, matches**. SF-18 periodic Gaussian 64^3 (sigma2=1, l=8, seed 12345): full tensor with off-diagonals up to 0.057 honored, symmetry defect 5.2e-17, eigenvalues 1.105/1.178/1.248 — **3D lognormal cross-check: e^{1/6}=1.181 predicted vs 1.178 observed**, G=(0.867,-0.003,-0.042), flux error 0.0, div_max 2.6e-13, 20 iters/direction, 3.32 s. Bitwise reproducibility across repeated solves. Four distinct validation messages. Full ctest 8/8. Hardware: RTX 3050 4 GiB, Debug sm_86, sccache launchers disabled. | Orchestrator FINAL_AUDIT on the control checkout. |
| 2026-08-10T14:50Z | `cfcc2e1`, final audit PASS | Orchestrator personally re-audited the integrated head on the control checkout: fresh configure/build; full ctest 8/8 (793 s); pipeline invariance vs the orchestrator's OWN base build (exact `751071a` refs): pspta_small and the SF-16 homogeneous fixture have IDENTICAL stdout and byte-identical artifacts except manifest git_hash/timestamp — "existing flow cases remain unchanged" proven at the byte level; checker OK with `NEXT: SF-19` unchanged. Gate 1 + Gate 2 + Gate 3A prerequisite PASS; Gate 4 interpretation recorded (activation decision 7: this increment only constructs the reference flow; no transport/dispersion claims). | Flagged for the human reviewer: (1) the seven activation decisions — esp. the coefficient-K adapter reuse, the degenerate gauge-pair RHS trick, the RAW-tensor Cramer solve with SPD check, and the prespecified rtol=1e-11; (2) the Gate-4 statement; (3) the new host 3x3 helpers; (4) mandatory-review paths src/physics/flow/ + tests/. Frozen audited source head: `cfcc2e1`. | Publish PR as `awaiting_review`; do not advance `NEXT`; await explicit human approval. |
| 2026-08-10T14:58Z | `2a016ff` published, PR #31 open | Delivery branch pushed and [PR #31](https://github.com/santi-esquerre/MacroFlow3D/pull/31) opened as `awaiting_review` with the frozen audited source head `cfcc2e1` (later commits are increment-state documentation only). | PR description carries the DAG (zero corrective cycles), the construction and its by-construction convention-identity arguments, all prespecified acceptance evidence including the two effective-medium physics cross-checks, the Gate-4 interpretation, and the reviewer flags. No agent merges; `NEXT` remains `SF-19`. | Await explicit human review/approval of PR #31; on approval, add only the closure metadata commit (`done`, checklist, `NEXT: SF-20`) on this same PR. |
