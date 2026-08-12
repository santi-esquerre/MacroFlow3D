# SF-23 — Restarted GMRES and block preconditioner

- State: `active`
- Goal: `Implementar GMRES reiniciado con precondicionador bloque diagonal.`
- Depends on: `SF-22`
- Unlocks: `SF-24`
- Branch: `science/lester-sf23-gmres-preconditioner`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-12T05:10Z`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Solve the coupled Newton linearization with bounded GPU memory and the already
validated diffusion blocks as a physics-informed preconditioner.

## Preconditions

- SF-22 provides an accepted matrix-free Jv.

## In scope

- Restarted double-precision GMRES, per-component projection, block diagonal
  `P=diag(A,A)`, fixed V-cycle application, and memory/iteration reports.

## Out of scope

- Newton globalization, adaptive/mixed preconditioning, and FGMRES production.

## Files and symbols

- Add a coupled restarted GMRES module or a narrowly reusable numerical solver.
- Add a block preconditioner adapter applying two projected MG V-cycles.

## Implementation specification

1. Default restart to 10; allow 15 only through explicit config and measured
   memory.  Store `2*(m+1)` scalar fields for the basis plus work vectors.
2. Use modified Gram-Schmidt with reorthogonalization trigger and Givens updates.
3. Project each coupled basis vector and true residual component.
4. Keep the preconditioner fixed and linear in this increment, so standard
   GMRES is valid; introduce FGMRES only when SF-29 makes it necessary.
5. Recompute the true residual at each restart and termination.

## Expected numerical effect

The block preconditioner reduces matrix-free Krylov iterations without changing
the converged linearized correction.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_gmres
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Small explicitly assembled/reference linearizations match GMRES corrections
  within `1e-8` relative.
- Reported and true residuals agree at restart/termination.
- Block preconditioning reduces iterations on the fixed suite.
- Restart-10 basis cost at `256^3` is approximately 2.75 GiB and agrees with
  actual allocation accounting.

## Regression surface

- Coupled scaling, orthogonality loss, MG linearity, memory, and restart logic.

## Failure and rollback policy

- If a supposedly fixed V-cycle is observably nonlinear, stop and either make
  it fixed or revise this increment to FGMRES through an explicit decision.
- Do not raise restart beyond the V100 memory budget to force convergence.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Restarted GMRES and true-residual checks are implemented.
- [ ] Block diagonal projected preconditioner is implemented.
- [ ] Reference correction and iteration-reduction tests pass.
- [ ] Restart memory accounting is measured and documented.
- [ ] Gate 3A regressions and human review pass.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-23 complete and selects SF-24.
<!-- completion-checklist:end -->

## Advancement rule

SF-24 may use the accepted linear solver inside a globalized Newton iteration.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-12T05:10Z | activation on `master=21dd32e` (SF-22 closure merged via PR #35) | SF-23 activated after verifying `NEXT: SF-23`, SF-22 `done`, checker `OK (30 increments, next=SF-23)`. Interpretive decisions PRESPECIFIED before implementation: (E1) **RIGHT preconditioning**: GMRES solves J M^-1 u = b with delta = M^-1 u, so the Givens-recurrence residual estimates the TRUE unpreconditioned residual ||b - J delta|| and the spec's reported-vs-true agreement check is meaningful in one norm (left preconditioning would change the residual norm and weaken that check; standard inexact-Newton practice). (E2) **block preconditioner adapter**: M^-1 = diag(M_A, M_A) with M_A = TWO successive projected positive V-cycles per block (zero initial guess, dashboard-locked smoothing counts, existing `projected_positive_v_cycle` machinery and hierarchy; per-component mean-zero projection at input/output as in the accepted SF-05 adapter); FIXEDNESS/LINEARITY is a prespecified TEST, not an assumption: bitwise repeatability of M v across calls AND ||M(ax+by) - aMx - bMy|| <= 1e-12 * scale on fixture vectors — if the cycle is observably nonlinear the increment STOPS per the spec's failure policy. (E3) **GMRES core**: restart m=10 default (15 only via explicit config + measured memory per the spec); storage (m+1) coupled basis vectors as contiguous 2N buffers (SF-22 CoupledVectorView layout) + work vectors, exact closed-form byte accounting; modified Gram-Schmidt with ONE-pass reorthogonalization triggered by the classical norm-drop criterion kappa = 1/sqrt(2) (reorthogonalization events counted in the report); Givens rotations for the Hessenberg least squares; TRUE residual recomputed at every restart and at termination — GATE: |true - reported| <= 1e-8 * max(true, ||b||) at each such point. (E4) **projection discipline**: rhs components projected on entry; every Krylov basis vector projected after Jv and after M application, before orthogonalization; the returned correction projected. (E5) **reference-correction oracles (prespecified)**: (i) eta=0 exactness: J = diag(A,A) exactly, so the GMRES correction must match the accepted projected-PCG per-block solutions within 1e-8 relative L2 (spec threshold 1); (ii) eta=1 dense oracle on 8^3 (2N=1024): assemble J column-by-column from the SF-22-validated Jv on unit directions, dense partial-pivot LU on host, GMRES correction within 1e-8 relative (validates GMRES+preconditioner treating Jv as the operator definition); (iii) iteration-reduction gate (spec threshold 3): preconditioned GMRES must use STRICTLY fewer total inner iterations than unpreconditioned on the fixed eta=1 trig suite (16^3 and 32^3, fixed seeds/states), ratios recorded — the spec fixes no numeric factor and none is invented. (E6) **memory**: restart-10 basis cost formula 2*(m+1)*n*8 B documented and checked: exact allocated==estimate equality at test sizes plus the arithmetic 256^3 prediction ~2.75 GiB (spec threshold 4) — no 256^3 allocation in tests. (E7) **SF-22 re-stress obligation discharged here**: a GMRES bitwise-determinism case at 32^3 eta=1 (two identical solves -> bitwise-identical corrections), which exercises the repeated-Jv-apply pattern under real Krylov load, PLUS the SF-22 jvp_repeated_apply_stress case remaining green in the suite. (E8) venue policy stands (workers: local compile gate only; all test execution on the remote V100; checksum-verified syncs). Out-of-scope confirmed: Newton globalization, line search, FGMRES, mixed precision. Branch field normalized to the house slug. | Gate 1 + Gate 2 + Gate 3A apply; human review required, so the PR stops at `awaiting_review` with `NEXT` unchanged. | Build the intra-increment DAG; delegate T01 (GMRES + preconditioner library) then T02 (tests) to isolated workers. |
| 2026-08-12T06:40Z | pre-run gate refinement E9 (recorded BEFORE any GMRES numerical run; no result has been seen) | Orchestrator audit of the T01 design surfaced a prespecification defect in E3/E5's uniform 1e-8 gates: the operator GMRES applies is the SF-22 forward-difference Jv with a DIRECTION-DEPENDENT delta, which is exactly linear only at eta=0 (affine map, delta-independent); at eta=1 successive applications are mutually consistent only to the FD truncation level (SF-22 measured ~1.8e-5 relative at 32^3 policy delta, ~1.5e-6 at 16^3), so reported-vs-true agreement and dense-oracle matching CANNOT beat that floor regardless of GMRES correctness. REFINED GATES (two-tier, replacing the uniform numbers in E3/E5): (a) E3 residual agreement per checkpoint: eta=0 <= 1e-8 * max(true, ||b||); eta=1 <= 1e-4 * max(true, ||b||) with all values recorded (headroom factor over the truncation floor for m-fold accumulation and conditioning). (b) E5(ii) dense-LU oracle at 8^3: assembled at eta=0 gate <= 1e-8 relative (delta-independent assembly, meaningful machine-level check of the GMRES/preconditioner algebra); assembled at eta=1 gate <= 1e-4 relative with values recorded. (c) E5(i) eta=0 PCG cross-check unchanged at 1e-8 (spec threshold, achievable). The spec's own 1e-8 threshold text ('small explicitly assembled/reference linearizations') is honored at eta=0 where an exact reference linearization exists; at eta=1 no exact reference exists through an FD operator and the recorded two-tier gate is the honest quantitative rendering. | Rationale: prespecification refinement grounded in the SF-22 measured truncation numbers, made before any GMRES run; not a post-hoc adjustment. | Launch T02 with the E9-refined gates. |
