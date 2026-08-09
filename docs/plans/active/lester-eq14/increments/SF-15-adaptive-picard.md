# SF-15 — Adaptive Picard

- State: `done`
- Goal: `Añadir relajación adaptativa, rechazo y detección de estancamiento a Picard.`
- Depends on: `SF-14`
- Unlocks: `SF-16`
- Branch: `science/lester-sf15-adaptive-picard`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-09T03:16Z`
- Completed: `2026-08-09 (explicit owner approval of PR #27; closure metadata commit on the same PR)`
- PR: [#27 — SF-15: adaptive Picard — trial safeguard, omega backtracking/growth, rejection guards, stagnation](https://github.com/santi-esquerre/MacroFlow3D/pull/27)
- Commit: `a98117c2405024041320fd36df2c0f21973bede7 (frozen audited source head; later branch commits are increment-state documentation only)`

## Scientific or engineering intent

Globalize Picard without changing its fixed-point map and make every rejected
or stalled update observable and recoverable.

## Preconditions

- SF-14 fixed-relaxation map and histories are accepted.

## In scope

- Trial buffers, Armijo-like residual safeguard, adaptive `omega`, rollback,
  degeneracy rejection, maximum iterations, and stagnation status.

## Out of scope

- Continuation parameter changes, Anderson, and Newton.

## Files and symbols

- Extend nonlinear control/report code in `StreamfunctionSolver`.
- Add deterministic tests that force accept, reject, minimum-step, and
  stagnation branches.

## Implementation specification

1. Start `omega=0.25`; halve a rejected trial to a minimum `0.01`.
2. Grow by `1.2` after three easy accepted trials, capped at one.
3. Reject nonfinite trials and dashboard-defined unexplained degeneracy growth.
4. Keep the last accepted pair immutable during backtracking; recompute only
   trial residual/diagnostics, not the expensive Picard map.
5. Flag stagnation after less than 1% residual reduction in ten accepted steps.

## Expected numerical effect

Residual growth is globally controlled, failed trials do not corrupt state, and
failure reason is deterministic.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R streamfunction_picard_adaptive
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Forced bad trials leave accepted fields bitwise unchanged.
- `omega` stays in `[0.01,1]` and follows the specified transition sequence.
- All exit statuses include iteration, residual, omega, and reason.

## Regression surface

- Trial/current buffer aliasing, extra residual evaluations, and degeneracy
  classification.

## Failure and rollback policy

- Do not accept a residual increase by relabeling it stagnation.
- If `omega_min` fails, return a structured failure for continuation to handle
  later.

## Completion checklist

<!-- completion-checklist:start -->
- [x] Adaptive omega and rollback are implemented.
- [x] Accept/reject/minimum/stagnation branches have deterministic tests.
- [x] Degeneracy and nonfinite policies match the dashboard.
- [x] Histories identify every trial and accepted state.
- [x] Gate 3A regressions and human review pass.
- [x] Evidence, PR, and commit are recorded.
- [x] Dashboard marks SF-15 complete and selects SF-16.
<!-- completion-checklist:end -->

## Advancement rule

SF-16 may expose this validated solver through configuration and pipeline I/O.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-09T03:16Z | activation on `master=b7d0c86` (SF-14 closure merged via PR #26) | SF-15 activated after verifying `NEXT: SF-15`, SF-14 `done`, and checker `OK (29 increments, next=SF-15)` on the default branch. Interpretive decisions recorded for the human reviewer: (1) globalization parameters live in a new `AdaptivePicardConfig` composed as `config.adaptive` with `enabled` defaulting to **true** (the plan's progression makes adaptive Picard the operative solver); `enabled=false` reproduces the SF-14 fixed path exactly, and the accepted `picard_fixed_*` test cases are pinned to `enabled=false` so they keep testing the fixed map they were written for. (2) The "Armijo-like residual safeguard" is operationalized as accept iff `r_F(trial) <= (1 - c*omega_try) * r_F(accepted)` with `c` default `1e-4` (configurable, validated); a residual increase can therefore never be accepted, and is never relabeled stagnation. (3) Trial state lives in **two new device fields** (`u_trial1/2`, +2 fine-grid-equivalent fields — memory option (a) tolerates; the SF-12 estimator, memory-report categories, and the api-workspace closed-form test are amended coherently); the block solutions `u_hat` stay immutable in `f1`/`f2` during backtracking (the expensive Picard map is NOT recomputed — only the relaxed candidate and its residual/diagnostics are), and trial residual outputs go to the otherwise-idle `rhs1`/`rhs2` buffers. (4) Dashboard degeneracy guards operationalized per trial: reject on nonfinite trial residual/sources; reject if the trial's unexplained degenerate fraction (SF-11 split at the FIRST configured diagnostics threshold) exceeds `max_unexplained_fraction` (default 0.01) OR exceeds `growth_factor*f_prev + growth_offset` (defaults 2, 1e-4) where `f_prev` is the last ACCEPTED state's fraction; reject if the |c| 0.1% percentile (SF-10 residual histogram, `residual_histogram_percentile`) collapses by more than one decade vs the last accepted state while the unexplained fraction did not stay <= `f_prev` (the "without matching Darcy low-speed population" reading); trial SF-11 diagnostics are evaluated only when diagnostics degeneracy thresholds are configured (guards vacuous otherwise, saving per-trial cost). (5) Omega policy locked to the dashboard: start `config.picard.omega` (0.25), halve on rejection clamped to `omega_min=0.01` with exactly one final trial at the floor (a rejected floor trial is the structured failure), grow ×1.2 after three consecutive zero-backtrack ("easy") acceptances, capped at `omega_max=1`; omega persists across iterations. (6) Stagnation: after at least `window=10` accepted steps, exit when `r_F_now > (1 - 0.01) * r_F(window ago)`; forced-branch tests may set extreme window/reduction values to reach the branch deterministically, never to hide a residual increase. (7) `StreamfunctionSolveStatus` stays 4-valued; the report gains `PicardExitReason { none, converged, budget_exhausted, linear_block_failure, stagnated, omega_floor_rejected }`, `final_omega`, and a per-trial `PicardTrialRecord` history (iteration, omega, trial r_F, outcome) alongside the unchanged SF-14 accepted-state history, so every exit carries iteration, residual, omega, and reason and every trial is identifiable. (8) "Forced bad trials leave accepted fields bitwise unchanged" is verified by re-evaluating the residual on the returned fields after an omega-floor failure and matching the last accepted r_F bitwise. | Base commit is this activation commit on `master=b7d0c86`. Gate 1 + Gate 2 + Gate 3A apply; human review required, so the PR will stop at `awaiting_review` with `NEXT` unchanged. Memory: +2 device fields (trial pair) recorded explicitly against option (a). | Build intra-increment DAG; delegate implementation to isolated worker worktrees. |
| 2026-08-09T04:36Z | `a98117c`, integration validation | Two-node DAG completed and orchestrator-audited node by node, zero blocking/major/minor findings, no corrective cycle. T01 `1532849`: `AdaptivePicardConfig` (12 fields, dashboard-locked defaults, unconditional validation with distinct messages) composed as `config.adaptive` with `enabled=true` default; two new trial device fields threaded coherently through prepare/bytes/report/estimator and the SF-12 closed-form test (+2 ffe exactly); adaptive loop: head evaluation captures `r_F_k`, |c| 0.1% percentile, and unexplained fraction on host BEFORE trials; map (2 block solves) once per outer iteration; backtracking forms trials in the new buffers (accepted state never touched), guards in order nonfinite → degeneracy (`f>0.01 ∨ f>2·f_prev+1e-4`) → percentile collapse (`p<p_prev/10 ∧ f>f_prev`) → Armijo (`r_F ≤ (1−1e-4·ω)·r_F_prev`), ω persistente = ω aceptado, growth ×1.2 tras 3 aceptaciones sin backtrack (tope 1), halving clamped a 0.01 con exactamente un trial en el piso (rechazo ⇒ fallo estructurado `omega_floor_rejected`); estancamiento en cabeza sobre estados aceptados; `PicardExitReason`/`final_omega`/`trial_history` en el reporte; `enabled=false` = camino SF-14 verbatim. T02 `a98117c`: seis casos deterministas + CTest `streamfunction_picard_adaptive` + pin de una línea (`adaptive.enabled=false`) en el `valid_config()` de los casos `picard_fixed_*` preservando su significado de mapa fijo. Single integrator verified the linear chain (merge-base == base, 11 files +1723/−168, `diff --check` clean, `src/**` limited to five streamfunctions files) and reran the full suite green; final commit `a98117c`, no integration commit. | Acceptance evidence (integrator + orchestrator reruns agreeing): **accept+growth**: converged in 20 iterations at 16³/a=0.25, ω trajectory BITWISE-exact vs the host-recomputed rule (0.25×3→0.3×3→0.36×3→0.432×3→0.5184×3→0.62208×3→0.746496×2), r_F estrictamente decreciente a 4.57e-7; **reject→floor**: secuencia de trials exacta {0.25,0.125,0.0625,0.03125,0.015625,0.01} toda `rejected_armijo` (armijo_c=0.999999 extremo-pero-válido), `omega_floor_rejected`, `picard_iterations=0`, **integridad bitwise** probada re-evaluando el residuo test-side sobre los campos devueltos (= `0.010360807248276992` exacto tras 6 trials rechazados); **stagnation**: dispara exactamente en k=10 (min_reduction=0.99 extremo-pero-válido) con r_F aún decreciendo en la ventana (nunca re-etiqueta un aumento); **degeneracy**: todos los trials `rejected_degeneracy` con `unexplained[0]=4096=n`, `low_speed=0`; **fixed-mode**: `enabled=false` reproduce SF-14 exacto (40 iteraciones, r_F=9.9099533174336207e-07, trial_history vacía) y los casos pineados vuelven a su perfil SF-14; **error contract**: 20/20; estimador de memoria con igualdad exacta de tres vías en 16³/32³/64³ (+2 ffe: presupuesto 256³ ahora 69.582 ffe, over_budget registrado honestamente). Full suite: ctest 5/5, runner 104/104 PASS, `run_operator_tests` 8/8, smoke OK, checker OK. Hardware: RTX 3050 Laptop 4 GiB, Debug sm_86, sccache launchers disabled. | Orchestrator FINAL_AUDIT on the control checkout, then publish PR as `awaiting_review`. |
| 2026-08-09T04:40Z | `a98117c`, final audit PASS | Orchestrator personally re-audited the integrated head against the original spec on the control checkout: fresh reconfigure/build, ctest 5/5, 104/104 case verdicts, 8/8 operator tests, smoke, checker all green; all three spec acceptance thresholds have explicit evidence (bitwise-unchanged accepted fields under forced bad trials; ω ∈ [0.01,1] con transiciones regla-exactas bitwise en ambas direcciones; todo exit con iteración/residuo/ω/razón + historia por trial); la política de fallo se respeta (un aumento de residuo es inaceptable por construcción; el rechazo en el piso devuelve el fallo estructurado para continuación). Gate 1 + Gate 2 + Gate 3A PASS; Gate 4/5, V100 N/A. | Flagged for the human reviewer: (1) las ocho decisiones interpretativas de activación (default adaptativo con suite fija pineada; constante Armijo 1e-4 y su rechazo de mesetas exactas; lectura operacional del guard de colapso de percentil; +2 campos de device → 69.58 ffe); (2) frontera de cobertura conocida: `rejected_nonfinite`/`rejected_percentile` sin rama forzada dedicada (verificados por inspección); (3) ruta de revisión obligatoria `src/physics/streamfunctions/`. Frozen audited source head: `a98117c`. | Publish PR as `awaiting_review`; do not advance `NEXT`; await explicit human approval. |
| 2026-08-09T04:44Z | `01462b5` published, PR #27 open | Delivery branch pushed and [PR #27](https://github.com/santi-esquerre/MacroFlow3D/pull/27) opened as `awaiting_review` with the frozen audited source head `a98117c` (later commits on the branch are increment-state documentation only). | PR description carries the DAG, audit summaries, the deterministic branch-evidence table (bitwise ω trajectories, bitwise field integrity, stagnation/degeneracy branches, SF-14 equivalence), the +2-field memory record (69.58 ffe), interpretive decisions, and the known coverage boundary. No agent merges; `NEXT` remains `SF-15`. | Await explicit human review/approval of PR #27; on approval, add only the closure metadata commit (`done`, checklist, `NEXT: SF-16`) on this same PR. |
| 2026-08-09T08:56Z | PR #27 head `0b26ab7`, human approval | The repository owner explicitly approved PR #27 with the instruction "Apruebo la PR #27, hacé el cierre". No GitHub review object exists (`reviews=0`); the approval fact is this recorded instruction. Verified before closure: PR #27 `OPEN` at head `0b26ab7` — exactly the published state; frozen audited source head `a98117c` unchanged (later commits are increment-state documentation only), so the approval applies to the audited content. | The approval covers the items flagged at publication: the eight activation interpretive decisions (adaptive default with pinned fixed suite, Armijo operationalization and plateau rejection, percentile-collapse guard reading, +2 trial device fields → 69.58 ffe), the known coverage boundary (`rejected_nonfinite`/`rejected_percentile` without dedicated forced-branch tests), and the mandatory-review path `src/physics/streamfunctions/`. | Closure metadata commit on this PR: set `done`, complete checklist, advance `NEXT` to `SF-16`. |
| 2026-08-09T08:56Z | closure metadata commit | SF-15 set `done`; checklist completed 7/7; dashboard updated (`SF-15` checked, `Last completed increment: SF-15`, `NEXT: SF-16`, active goal `none`); checker rerun. The new `NEXT: SF-16` exists only on this PR branch until a human merges it and does not authorize work ahead of the default branch. | Metadata/documentation-only diff (increment spec + dashboard); frozen audited source head remains `a98117c`. | Human merges PR #27; SF-16 (pipeline, configuration, and output) may activate only after this closure state is visible on `master`. |
