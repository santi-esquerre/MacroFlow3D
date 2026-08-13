# SF-24 — Globalized Newton-Krylov

- State: `awaiting_review`
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
- PR: `pending publication (opened by the orchestrator after this commit)`
- Commit: `5407c22` (frozen audited source-bearing head)

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
- [x] Activation, inexact solve, Armijo, rollback, and fallback are implemented.
- [x] Forced failure paths preserve accepted state.
- [x] Picard/Newton final-solution equivalence is demonstrated.
- [x] Runtime/residual-evaluation comparison is recorded.
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
| 2026-08-13T01:40Z | C01 `31206ac` ACCEPTED; T02 `abb554e` ACCEPTED; T03 `54f59ac` ACCEPTED (all compile-gate-only workers; full diffs personally audited) | C01: exact required guard (first check, distinct message, both call sites, docs) — T01-F1 discharged. T02: newton YAML subsection + validator + serializer + EnsembleRunner wiring + summary.json/stdout export, defaults byte-equal to the library struct; exports CONDITIONAL on `newton.enabled` so default outputs stay byte-identical; PRE-DECLARED byte-compare exception (recorded here BEFORE any validation run): the effective-config serializer writes the `newton:` block unconditionally — the exact `anderson:` precedent — so default-config `effective_config.yaml` will differ from the f4dcde8 base refs by exactly that appended block with default values; every other artifact must stay byte-identical. T02 also surfaced a genuine pre-existing gap: the SF-20 Anderson counters have NO export path anywhere (recorded as backlog; candidate for the SF-25 stage-attribution diagnostics extension). T03: G1-G7 implemented verbatim (constants untouched), fixtures copied verbatim with cited provenance (SF-20 equivalence trig-K 16^3/32^3; anderson_stall_fixture_a for the difficult case; homogeneous K=1 for the never-activated control), G3 self-calibrating with the retry/rescue offsets I re-traced against the state machine, heavy case isolated from the aggregated run via the anderson_stall registry mechanism. Honest-outcome risks recorded: the G4 improvement gate (fewer total residual evaluations OR lower wall vs the Anderson control) is a real algorithmic claim that may legitimately fail on the V100. | Audits: `.claude/orchestration/SF-24-newton-krylov/audits/{T02,T03}-audit.md`. Approved integration order: `6e113ce -> 31206ac -> abb554e -> 54f59ac`. V01 folded into F01 (T02/T03 are parallel siblings; the integrated tree is the first containing both — one comprehensive V100 validation on the exact integrated head). | I01 integrator (cherry-pick + patch-equality + compile gate + ctest -N only), then the combined remote validation/final audit. |
| 2026-08-13T01:45Z | I01 head `8ac6bf5` (4 cherry-picks byte-identical to originals: 1349/94/381/1343 patch lines, zero conflicts, zero integration edits; 16 ctest entries); first comprehensive V100 run on that head (job `sf24-final`, raw logs `/tmp/sf24_newton_cheap.log`, `/tmp/sf24_newton_difficult.log`, `/tmp/sf24_full_suite.log` + artifact diffs) | RESULTS. G1 equivalence PASS with margin: weighted field diff 4.29e-9 (16^3) / 7.54e-9 (32^3) vs gate 1e-3; e_v relative agreement ~2e-8; \|c\| p0.1% EXACTLY equal; Newton runs converged in 1 and 7 outer iterations vs 20 and 23 Picard-only. G2 PASS (single activation at k*=1; forcing values exact to the clamp formula: 0.0893/0.01266/0.001199 at r_F 7.98e-3/1.60e-4/1.44e-6 — textbook superlinear trajectory). G6 PASS (all byte identities exact, both grids). G7 PASS (14/14 incl. C01 rule; never-activated homogeneous control clean). **G4/G5 PASS — the increment's scientific threshold**: on the sigma^2=1-class stall fixture (lambda*=0.1125, 32^3), Anderson CONTROL converged in 64 outer iterations / 136 residual evaluations / 39.37 s; NEWTON converged in 2 outer iterations + 1 activation + 3 accepted Newton steps / 164 evaluations / **20.08 s (1.96x faster)**, final r_F 1.61e-7; the gate passed on the WALL branch (Newton honestly LOSES the raw evaluation count 164 vs 136 — Jv evaluations are individually far cheaper than Picard block solves); bitwise determinism rerun PASS. **G3 FAIL — prespecification defect in the forced-failure MECHANISM (mine, not the library's and not the gate targets')**: `armijo_c=0.999999` forces rejection only at alpha=1 (merit factor 1e-6 -> r_F factor 1e-3); at alpha=0.5 the threshold (1-c*alpha)~0.5 on phi is met by ANY (1-alpha)-damped Newton step. Measured: every variant's Newton iterations accepted at alpha=0.5 after one alpha=1 rejection (2-trial pattern, 13-16 iterations, all three variants CONVERGED healthily; variant A record trail retained). The solver behaved CORRECTLY; the fixture could not force failure. **AMENDMENT E13 (measurement-methodology correction, recorded HERE before the corrective run; G3 assertion targets unchanged — preservation, counters, taxonomy, rescue offset):** forced-failure mechanism becomes `alpha_min=1` (single-trial ladder), `armijo_c=1-1e-8` (demands r_F factor 1e-4 in one step), `forcing_min=forcing_max=1e-1` (inexactness floor); measured single-inexact-step reduction at these fixtures is ~2e-2, a ~200x margin above the rejection threshold; ladder assertion updated from 6 trials to exactly 1 `rejected_armijo` trial per failing record. Full suite on this head: 13/16 (failures = the G3 case propagated to `streamfunction_newton` + aggregated entry, plus the KNOWN `streamfunction_heterogeneity_smoke` moved to SF-25). Byte-compares vs `~/sf24_base_refs`: stdout IDENTICAL x3; artifacts differ by EXACTLY the pre-declared `newton:` effective-config block (18 lines, default values) + the manifest timestamp — the declared expectation holds precisely. | Raw numbers retained above and in the remote logs. Corrective C02 delegated: test-only change to newton_gpu_cases.cu G3 variants (mechanism + ladder assertion + doc), no gate target touched, no src/** change. | C02 -> audit -> fresh integration of the corrected chain -> full FINAL_AUDIT rerun on the exact new head. |
| 2026-08-13T14:05Z | PUBLICATION. Frozen audited source-bearing head `5407c22` = integrated chain (T01 `2fa12fc` ≡ `6e113ce`, C01 `ceed6bb` ≡ `31206ac`, T02 `e4dbb26` ≡ `abb554e`, T03 `8ac6bf5` ≡ `54f59ac`; all patch-equal, zero integration edits) + C02 `5407c22` (E13 test-mechanism corrective, recovered verbatim from the interrupted worker's completed edits, orchestrator-audited and compile-gated). FINAL_AUDIT PASS on the exact head (V100 job `sf24-final2`): full suite 15/16 — the ONLY failure is the KNOWN `streamfunction_heterogeneity_smoke` (sigma^2=1 gate moved verbatim to SF-25); G3 after E13 19/19 (bitwise state preservation across both failed activations, exact activation/rescue/failure counters, single-trial ladder, taxonomy, rescue offset, 5 accepted rescue advances); G1/G2/G4/G5/G6/G7 green on this head via the suite entries (first-pass raw numbers in the 2026-08-13T01:45Z row); byte-compares vs the f4dcde8 base refs: stdout identical x3, artifacts differ by exactly the pre-declared `newton:` effective-config block + manifest timestamp. State set to `awaiting_review` (human review REQUIRED); implementation checklist items 1-4 checked; items 5-7 (Gate 3A/4 + human review, evidence/PR record, dashboard advance) remain for closure after explicit owner approval. `NEXT` remains `SF-24`; no advancement before merge visibility on `master`. | Final audit record: `.claude/orchestration/SF-24-newton-krylov/audits/final-audit.md`. Flagged for the reviewer: E13 amendment (test mechanism only), G4's honest metric split (wall 1.96x win, raw evaluation count loss 164-vs-136 — threshold is OR by prespecification), stagnation-during-rescue semantics, the pre-existing Anderson-counter export gap (backlog, candidate SF-25 extension). | Orchestrator pushes the branch and opens the PR; stop at AWAIT_HUMAN_REVIEW. |
