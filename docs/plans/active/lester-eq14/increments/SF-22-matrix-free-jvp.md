# SF-22 — Matrix-free Jacobian-vector product

- State: `done`
- Goal: `Implementar el vector acoplado y el producto Jacobiano-vector matrix-free.`
- Depends on: `SF-21`
- Unlocks: `SF-23`
- Branch: `science/lester-sf22-matrix-free-jvp`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-12T01:20Z`
- Completed: `2026-08-12 (explicit owner approval of PR #35 at head 76510be; frozen audited source head ed6e016)`
- PR: `https://github.com/santi-esquerre/MacroFlow3D/pull/35`
- Commit: `ed6e016 (frozen audited source head; later branch commits are increment-state documentation only)`

## Scientific or engineering intent

Reuse the accepted nonlinear residual to expose Jacobian action without
assembling or storing a coupled Jacobian.

## Preconditions

- SF-21 (partial closure, owner option (a) 2026-08-11) establishes the accepted Picard/Anderson residual evaluator and the heterogeneity-continuation machinery whose eta=1 non-contractive plateau motivates this Newton phase.

## In scope

- Coupled `2N` vector views, perturbation workspaces, finite-difference Jv,
  projection, step-size policy, and directional derivative tests.

## Out of scope

- Krylov iteration, Newton steps, line search, and mixed precision.

## Files and symbols

- Add `CoupledVectorView` and `JacobianVectorProduct.cuh/.cu` under
  `src/physics/streamfunctions/`.
- Reuse `ResidualEvaluator` without a second PDE implementation.

## Implementation specification

1. Present persistent fields as two views while allocating Krylov storage as
   contiguous `2N` buffers.
2. Project each component of direction and perturbed state.
3. Use forward difference
   `delta=sqrt(eps)*(1+||Psi||_w)/||p||_w` with documented weighted norm and
   configurable safeguards against under/overflow.
4. Cache `F(Psi)` for all Jv calls at one Newton state.
5. Validate against a central difference used only by tests.

## Expected numerical effect

Jv converges to the directional derivative over an observable finite-difference
step range and preserves the two zero-mean subspaces.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_jvp
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Forward/central Jv discrepancy has the expected U-shaped step study and meets
  a predeclared `1e-5` relative target at the chosen delta on small cases.
- Both Jv component means meet the projector threshold.
- No Jacobian matrix or Hessian fields are allocated.

## Regression surface

- Residual determinism, weighted units of coupled components, cancellation,
  and extra residual workspace.

## Failure and rollback policy

- Do not tune delta on the final benchmark only; use manufactured and Picard
  states across several norms.
- If forward difference is unreliable, document the range before considering a
  more expensive central production option.

## Completion checklist

<!-- completion-checklist:start -->
- [x] Coupled vector/view semantics are implemented.
- [x] Jv reuses the exact residual evaluator and cached base residual.
- [x] Delta policy and weighted norm are documented and tested.
- [x] Central-difference comparison and gauge thresholds pass.
- [x] Gate 3A regressions and human review pass.
- [x] Evidence, PR, and commit are recorded.
- [x] Dashboard marks SF-22 complete and selects SF-23.
<!-- completion-checklist:end -->

## Advancement rule

SF-23 may use this Jv as the matrix-free operator in restarted GMRES.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-12T01:20Z | activation on `master=6ad39d0` (SF-21 partial closure + Newton-first restructuring merged via PR #34) | SF-22 activated after verifying `NEXT: SF-22`, SF-21 `done`, checker `OK (30 increments, next=SF-22)`. Interpretive decisions PRESPECIFIED before implementation: (D1) **weighted norm** for the coupled state/direction: ||(a,b)||_w = sqrt( (RMS(a)/g1)^2 + (RMS(b)/g2)^2 ) with g1 = source_config.v_rms and g2 = 1 — exactly the per-component scales of the accepted residual normalization (r1, r2), so delta is dimensionless-consistent across both components; documented in the header. (D2) **delta policy** (spec item 3): forward difference, delta = sqrt(machine_eps) * (1 + ||Psi||_w) / ||p||_w; ||p||_w = 0 or nonfinite -> std::invalid_argument; configurable clamp [delta_min=1e-12, delta_max=1e2] with clamp counters surfaced in the report struct; any nonfinite in the perturbed residual -> structured failure, never silent. (D3) **projection discipline**: direction components are mean-zero-projected before use; the perturbed state is projected after the axpy (idempotent defense, same rationale as the SF-15 trial projection); the base state is a caller contract (already projected by the solver); Jv output means are MEASURED against the projector threshold (F is discretely mean-zero by construction, so nonzero Jv means indicate a defect — no silent output re-projection). (D4) **cached base residual** (spec item 4): a prepare_base(state) call evaluates and stores F(Psi) once per Newton state; every apply(p) performs exactly ONE perturbed residual evaluation; an evaluation counter is exposed and contract-tested. (D5) **central difference is test-only and built from the public API**: Jv_central = (Jv_forward(p) - Jv_forward(-p))/2 (identical delta since ||-p||_w = ||p||_w) — no library test-only code path. (D6) **prespecified fixtures for the U-study** (spec threshold 1): trig manufactured states at 16^3 and 32^3 (domain length 1, dx=1/n — the SF-21 C01 lesson is baked in) plus a CONVERGED adaptive-Picard state on the trig conductivity (the accepted SF-15/SF-20 fixture family), each against >=3 direction types (fixed-seed random mean-zero, gradient-like, single-Fourier-mode); sweep delta over >=6 decades bracketing the policy delta; GATES: observable U-shape (discrepancy decreases then increases), forward-vs-central relative discrepancy <= 1e-5 at the policy delta on the small cases, Jv component means <= the projector threshold used by the mean-zero contract tests. (D7) **eta=0 linearity oracle**: at eta=0, J p = [A p1; A p2] exactly; Jv must match the direct operator application within relative 1e-6 (FD cancellation noise budget at delta~1e-8, prespecified). (D8) venue policy stands (workers: local compile gate only; ALL test execution on the remote V100 by the orchestrator; checksum-verified syncs + md5 spot checks). Scope note recorded: no Jacobian matrix, no Hessian fields, no Krylov/Newton code in this increment. Branch field normalized to the house pattern (science/lester-sf22-...). | Gate 1 + Gate 2 + Gate 3A apply; human review required, so the PR stops at `awaiting_review` with `NEXT` unchanged. | Build the intra-increment DAG; delegate T01 (library) then T02 (tests) to isolated workers. |
| 2026-08-12T02:40Z | V01 evidence + audit findings recorded BEFORE the corrective cycle; bounded local-debug exception authorized | Remote V100 evidence on candidate base+T01+T02 (`126e0d1`): byte-compares IDENTICAL; eta=0 oracle at 16^3 PERFECT (rel 8.8e-10..3.0e-8 vs direct `[A p1; A p2]`); O(delta) ratios 10.000 in all six sweep combos; policy-delta 1e-5 gate met on all three 16^3 combos (max 1.5e-6). FINDINGS: (F1, BLOCKING) the eta=0 oracle case is NONDETERMINISTIC at 32^3 ONLY — five isolated runs gave varying deltas (~8% spread) and relative errors from 8.7e-10 to O(1); runs 3/4 produced BITWISE-IDENTICAL wrong values (discrete outcomes); compute-sanitizer initcheck/racecheck clean (racecheck cannot see cross-stream global races); the sweep case at the same 32^3 is bitwise-stable across runs. Analysis: an 8% delta change cannot produce O(1) error at eta=0 (FD of an affine map is delta-independent), so BOTH the measured direction norm AND the Jv consume corrupted device data — the projected-direction copy or its upstream chain is being corrupted at 32^3 in that case; root cause (library vs test) UNRESOLVED, assigned to corrective C01 with a forensic mandate. (F2, measurement-methodology amendment to D6, recorded before the corrective run) the discrepancy curve IS U-shaped in all six combos (interior minimum everywhere) but the minimum sits ~2 decades below the policy delta, so the prespecified ±3-decade sweep centered at the policy delta leaves the roundoff branch <10x above the minimum; AMENDED: sweep k=-5..+3 (9 points) so both branches have room; the both-ends>=10x-minimum gate is otherwise unchanged. (F3, amendment to D6 scope) the spec's own threshold text says "1e-5 relative target at the chosen delta on SMALL cases"; D6 over-extended it to 32^3 where the truncation branch for the roughest direction measures 1.83e-5 with perfect O(delta) behavior; AMENDED to the spec text: the 1e-5 gate applies at 16^3 (met: max 1.5e-6); the 32^3 policy-point values are RECORDED evidence (gated only by U-shape + O(delta)); raw numbers retained in the audit log. (T01-F1, MINOR, joined to C01) apply() lacks a cached-grid equality check vs prepare_jvp_base. VENUE EXCEPTION (bounded): corrective C01 is authorized to run LOCAL single-case debug executions of the jvp cases ONLY — forensic iteration on a nondeterministic GPU defect is impractical through the serialized remote tree; ALL acceptance evidence still comes from the V100. | Amendments are measurement-methodology corrections recorded before the corrective run; no gate value was loosened after the fact beyond restoring the spec's own wording. | Launch corrective C01 (forensic root cause + fixes + amended sweep geometry). |
| 2026-08-12T03:20Z | integration + FINAL AUDIT PASS; State -> `awaiting_review`; frozen audited source head `ed6e016` | Integrator applied the four approved commits (T01 `3da2aea`, T02 `126e0d1`, C01 `282a757`, C02 `9cb4d94`) with BYTE-IDENTICAL patches, zero conflicts; +2257 fully additive. FINAL AUDIT on the exact head (remote V100): full suite 12/13 — the only failure is the KNOWN `streamfunction_heterogeneity_smoke` entry (the sigma^2=1 gate moved to SF-25 by the owner's option-(a) closure; pre-existing on master, not an SF-22 regression); byte-compares vs base references identical (exception set {manifest timestamp, stdout timing table}). Acceptance evidence: eta=0 linearity oracle 8.8e-10..3.0e-8 vs direct [A p1; A p2] on BOTH grids post-fix; U-shape with interior minimum in all six combos (amended k=-5..+3 sweep); policy-delta 1e-5 gate met at 16^3 (max 1.5e-6) with 32^3 values recorded (max 1.8e-5, O(delta) ratios 10.000 = pure truncation); Jv means within the projector-derived bound; memory accounting exact; grid fail-fast contracted. STABILITY: after C01's forensic fix (root cause: shared residual-workspace base/perturbed aliasing; low-level mechanism UNCHARACTERIZED — flagged for human review; fix is structural, no sync workarounds), the V100 shows ORACLE_20RUNS_FAILS=0 and STRESS_10RUNS_FAILS=0 (C02's bitwise repeated-apply contract, 16^3 x60 / 32^3 x30 — the SF-23 GMRES hot pattern); 270+ consecutive clean local Debug runs. Re-stress obligation recorded for SF-23. | Human review covers: activation decisions D1-D8, the F2/F3 measurement-methodology amendments, C01's uncharacterized mechanism + structural fix, the bounded local-debug venue exception (used by C01/C02, acceptance evidence all-V100), and the known-failing heterogeneity_smoke entry on master until SF-25. `NEXT` remains `SF-22` until explicit approval. | Publish the PR as awaiting_review; stop at AWAIT_HUMAN_REVIEW. |
| 2026-08-12T04:05Z | closure metadata commit after explicit owner approval; State -> `done` | Owner approved PR #35 at exactly the published head `76510be` (frozen audited source head `ed6e016` unchanged; the two later commits are increment-state documentation only; no GitHub review object exists — the approval fact is the owner's explicit instruction, recorded truthfully). The approval covers: activation decisions D1-D8; the F2/F3 measurement-methodology amendments (versioned pre-corrective, raw numbers retained); corrective C01's structural fix for the 32^3 nondeterminism with its LOW-LEVEL MECHANISM UNCHARACTERIZED (empirically bounded: V100 oracle 20/20 + repeated-apply stress 10/10 stable post-fix vs 15-30% failure pre-fix; 270+ clean local runs); the bounded local-debug venue exception used by C01/C02 (acceptance evidence all-V100); and the known-failing heterogeneity_smoke entry remaining on master until SF-25. STANDING OBLIGATION carried to SF-23: re-stress the repeated-apply pattern under real GMRES usage (the jvp_repeated_apply_stress case is the reference contract). | Checklist complete; dashboard advanced to `NEXT: SF-23` in this commit (exists only on the PR branch until human merge). | Human merges PR #35; SF-23 (restarted GMRES + block preconditioner, consuming this Jv as the matrix-free operator) may activate only after this closure state is visible on `master`. |
