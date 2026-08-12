# SF-24 — Globalized Newton-Krylov

- State: `active`
- Goal: `Implementar Newton-Krylov globalizado y su fallback reproducible a Picard.`
- Depends on: `SF-23`
- Unlocks: `SF-25`
- Branch: `science/lester-sf-24-newton-krylov`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A + Gate 4`
- Human review: `required`
- Owner: `Claude Code orchestrator (Fable) + delegated workers`
- Started: `2026-08-12`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Accelerate locally converged difficult states without sacrificing the accepted
Picard continuation path as the authoritative robustness fallback.

## Preconditions

- SF-23 validates matrix-free GMRES and the block preconditioner.

## In scope

- Newton activation, inexact linear forcing, Armijo line search, state rollback,
  Picard rescue, continuation fallback, and comparative histories.

## Out of scope

- Mixed precision, replacing Picard defaults, and scientific benchmarks beyond
  the SF-24 parameter regime.

## Files and symbols

- Add `NewtonKrylovSolver.cuh/.cu` or a method implementation behind the stable
  streamfunction solver API.
- Extend config/report only when the method is functional.

## Implementation specification

1. Activate Newton after Picard reaches `r_F<1e-2`, or after documented
   stagnation below `1e-1` with acceptable nondegeneracy.
2. Solve `J delta=-F` with an inexact tolerance tied to current `r_F` and capped
   by configured minimum/maximum forcing terms.
3. Project corrections and use Armijo decrease of `0.5*r_F^2`, backtracking
   without overwriting the accepted state.
4. On Newton failure, restore state, run five accepted Picard steps, retry once,
   then reduce the current continuation step.
5. Log all activations, Jv/GMRES counts, line-search trials, fallbacks, and final
   physical metrics.

## Expected numerical effect

Near a valid solution Newton reduces nonlinear iterations while failed Newton
attempts return deterministically to the same safe Picard/continuation state.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_newton
scripts/remote run lester-newton -- "<fixed-picard-vs-newton-suite>"
scripts/remote wait lester-newton
```

## Acceptance thresholds

- Small cases reach the same final fields/metrics as Picard within nonlinear
  tolerance.
- Forced line-search, GMRES, and retry failures preserve accepted state.
- At least one fixed difficult case shows fewer nonlinear residual evaluations
  or lower wall time than Picard without reducing robustness across the suite.

## Regression surface

- Coupled scaling, line-search state ownership, continuation interaction,
  memory, and nonlinear stopping criteria.

## Failure and rollback policy

- Newton remains opt-in until it passes the full fixed suite.
- After the documented retry, return to Picard/reduced continuation; do not add
  silent fallback variants.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Activation, inexact solve, Armijo, rollback, and fallback are implemented.
- [ ] Forced failure paths preserve accepted state.
- [ ] Picard/Newton final-solution equivalence is demonstrated.
- [ ] Runtime/residual-evaluation comparison is recorded.
- [ ] Gate 3A/4 regressions and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-24 complete and selects SF-25.
<!-- completion-checklist:end -->

## Advancement rule

SF-25 may re-impose the full heterogeneity gates with Newton terminal convergence once the Newton path is
accepted and merged.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-12T21:55Z | activation on `master=f4dcde8` (SF-23 merged; checker OK next=SF-24) | UNDERSTAND completed with re-read of the Lester paper §5.1/Table 1 (homogenized-Krylov + explicit variable-step pseudo-time to 1e-16 at 256^3 — the paper does NOT use Newton; our Newton-Krylov is a deliberate, recorded methodological divergence chosen for stagewise robustness control), the SF-21 failing-case evidence (sigma^2=1, 32^3: non-contractive damped Picard/Anderson map exactly at eta=1 beyond lambda~0.5, r_F plateau 0.9e-3..4.8e-3 via stagnation detector, zero degeneracy — a contraction boundary; the plateau satisfies the r_F<1e-2 Newton activation threshold with margin), and full code reads of the SF-15/20/21 solver loop and the SF-22/23 Jv/GMRES/preconditioner surfaces. Recorded structural risk: Picard eigenvalue ~1 at the plateau implies a small Jacobian eigenvalue in the same mode (possible GMRES budget pressure; mitigated by MG preconditioning, forcing floor, restart-15, line search, rescue protocol, structured failure). PRESPECIFIED activation decisions E1-E12 and test gates G1-G8 (recorded verbatim in `.claude/orchestration/SF-24-newton-krylov/understanding.md` BEFORE any implementation): E1 Newton phase behind `config.newton` (default OFF, bitwise-preserving disabled path), NewtonKrylovSolver module + solve_streamfunctions integration, ContinuationController UNTOUCHED (SF-25 wires it); E2 activation at head r_F<=1e-2 or stagnation-with-r_F<=1e-1-and-clean-degeneracy (Anderson history cleared on Newton entry); E3 iteration = head eval + prepare_jvp_base + GMRES(J M^-1) with the SF-23 block preconditioner + projected line search through the IDENTICAL trial guard chain with Armijo on phi=0.5*r_F^2 (`phi_t <= (1-c*alpha)*phi_k`, c=1e-4, alpha floor 2^-5); E4 forcing `rel_tol_k = clamp(sqrt(r_F_k), 1e-8, 1e-1)`; E5 linear-accept: converged -> line search; budget/breakdown with strict true-residual reduction -> line search; nonfinite/no-reduction -> step failure; E6 failure protocol: state intact by construction -> 5 accepted Picard/Anderson rescue steps (config >=0, 0 exists only for the bitwise-preservation test) -> retry once per activation -> `newton_exhausted` structured exit consumed by the existing continuation failure handling; budget stop `newton_budget_exhausted`; E7 convergence identical to Picard (r_F <= picard.tolerance); E8 full comparative histories + `newton_jv_evaluations` from the monotone SF-22 counters; E9 optional workspace (Jvp + CoupledGmres(n,restart) + preconditioner + delta pair) with exact additive bytes, zero when disabled; E10 validation regardless of enabled; E11 bitwise determinism; E12 YAML `newton:` subsection per the SF-21 anderson precedent. Gates: G1 Picard/Newton equivalence on the accepted converging fixtures (both r_F<=1e-6, weighted field diff <=1e-3, e_v and \|c\| p0.1% <=1e-3 relative); G2 activation/forcing semantics from histories; G3 self-calibrating forced-failure BITWISE state preservation + rescue-count assertion; G4 difficult case = anderson_stall_fixture_a (sigma^2=1, lambda*=0.1125, 32^3) Newton-vs-Anderson: both converge, Newton strictly fewer total residual evaluations OR lower wall (raw numbers retained); G5 determinism; G6 memory exactness; G7 fail-fast; G8 disabled-path byte preservation (existing suites + pipeline byte-compares vs f4dcde8 base refs). Venue: workers compile-gate ONLY (permanent policy `c1433ce`); all runs orchestrator-driven on the V100. | DAG: T01 library -> {T02 io/pipeline surface, T03 tests} (parallel) -> V01 remote validation -> I01 -> F01. Out of scope reaffirmed: continuation wiring (SF-25), mixed precision, Picard-default changes, sigma^2=1 lambda=1 gate (SF-25 verbatim). | Delegate T01. |
| 2026-08-13T00:20Z | T01 candidate `6e113ce` audited (worker compile-gate only, per policy) | Personal audit of the full +1076-line diff: E1-E11 all verified correct by reading (guard chain byte-for-byte vs SF-15, Armijo on phi exact, forcing/linear-accept rules exact, every rescue/retry state-machine path traced, memory estimators cross-verified against the module estimators incl. the levels[0]==finest convention, disabled path statement-identical). ONE MAJOR finding T01-F1: `newton.enabled` with `adaptive.enabled==false` validated silently while the Newton phase lives only in the adaptive loop — a silent config no-op violating the AGENTS.md hard rule. Corrective C01 delegated: validation rejects the combination with a distinct message (precedent: reuse-requires-warm_start). Verified-behavior note recorded: during a rescue window the ordinary stagnation detector still governs; a zero-progress Newton failure on a stagnant plateau exits `stagnated` honestly instead of running a provably-futile deterministic retry against a bitwise-unchanged state. | Audit record: `.claude/orchestration/SF-24-newton-krylov/audits/T01-audit.md`. SF-24 base byte-compare refs regenerated on the V100 at `f4dcde8` (`~/sf24_base_refs/`, checksum rsync + md5 spot checks). | C01 -> then T02 (io/pipeline surface) and T03 (tests) in parallel on the corrected chain. |
