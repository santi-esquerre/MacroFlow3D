# SF-32 — Face-flux reference trackers and the paper's scalings

- State: `done`
- Goal: `Reproducir las figuras 3 y 4 de Lester 2023, mostrando la dispersión transversal espuria de los trackers convencionales frente al pseudo-simpléctico en un campo de invariantes exactos.`
- Depends on: `SF-31`
- Unlocks: `none`
- Branch: `science/lester-sf32-reference-trackers-scalings`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 4`
- Human review: `required`
- Owner: `Claude Fable 5.1 orchestrator session (2026-10-06)`
- Started: `2026-10-06T14:59Z on master=37bfb25`
- Completed: `2026-10-07T13:13Z (owner approval; closure metadata commit on PR #50)`
- PR: `#50` (https://github.com/santi-esquerre/MacroFlow3D/pull/50), delivery branch `science/lester-sf32-reference-trackers-scalings`
- Commit: `0f5b916` (audited source-bearing head: integration of N0 `d1bafa9`, N1 `5a4db8f`, N3a `9d0b580`, N4 `9e6ad13`, N2a `ef56372`, N2b `ac602f3`, C1 `cb445fb`, N3b `0bcbafd` on the activation head `9db86f0`, base `37bfb25`; later commits are artifacts, the experiment note and metadata only)

## Scientific or engineering intent

Reproduce Lester 2023 figures 3-4 (spurious transverse spreading of conventional trackers vs the pseudo-symplectic one) on a field whose invariants are exact.

Context: `docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md` and `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`.

## Preconditions

- `SF-31` is `done` on the default branch.

## In scope

- `StokesFaceVelocity`: face fluxes from line integrals of `psi1 grad psi2` around faces (Lester 2023 eqs. 32-33), exactly divergence-free per cell.
- A Pollock-type cellwise-linear tracker in the same module.
- Streamfunctions of the Lester (2021) field obtained with the frozen stack; if it does not converge, reduce the amplitude, then fall back to an analytic pair, never touching the solver.
- Diagnostics: `delta psi_i`, `delta x2, delta x3` after one period vs the pseudo-symplectic trajectory (eqs. 34-35).

## Out of scope

- Runner wiring, ensembles, macrodispersion interpretation.

## Files and symbols

- `src/physics/particles/streamline_tracker/*` (extended)
- New tests under `tests/` and an experiment note `docs/experiments/<date>-sf32-spurious-spreading.md`

## Implementation specification

1. Implement `StokesFaceVelocity` and the Pollock-type tracker.
2. Obtain the invariants (frozen stack, reduced amplitude, or analytic pair) and record which route was used.
3. Run grid and tolerance ladders as detached V100 jobs and fit the exponents.
4. Record exponents outside the bands without tuning.

## Expected numerical effect

New capability only; no existing result changes.

## Validation commands

```bash
cmake --preset wsl-debug && cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R tracker
scripts/remote sync
scripts/remote run sf32-scalings -- "<scaling ladders command>"
scripts/remote wait sf32-scalings
bash scripts/hooks/check-lester-increments.sh
```

## Acceptance thresholds

- Pollock variance proportional to `Delta^p` with `p = 2 +- 0.3` over >= 3 grids.
- RK with `p = 0.5 +- 0.15` over >= 4 tolerances.
- Pseudo-symplectic drift <= 10x the Newton tolerance.
- An exponent outside the band is recorded, not tuned.

## Regression surface

- None existing; consumes SF-31 and the frozen stack.

## Failure and rollback policy

- Out-of-band exponents are reported as findings; the increment stays active until the human review decides.

## Completion checklist

<!-- completion-checklist:start -->
- [x] Implementation matches the scope and contains no unrelated changes.
- [x] Targeted validation passes and its evidence is recorded.
- [x] Required regression tests pass.
- [x] Scientific or engineering findings are appended to the bitácora.
- [x] Required human review is recorded.
- [x] PR and commit identifiers are recorded.
- [x] The master checklist entry is checked in this branch and `check-lester-increments.sh` passes.
<!-- completion-checklist:end -->

## Advancement rule

No increment is unlocked directly; its outcome feeds the later phases described in the dashboard.

## Bitácora

Append entries; do not rewrite prior observations.  Store large raw outputs as
artifacts or experiment notes and link them here.

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-10-02T00:00Z | not started | Specification created by the 2026-10-02 re-sequencing (decision record `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`). | Replaces the cancelled SF-27..SF-30 specifications (git history at `4670fb5`). | Activate only when the checker reports it READY. |
| 2026-10-06T14:59Z | activation on `master=37bfb25` (PR #49 merged; checker OK ready=SF-32 SF-33, nonterminal=none); delivery branch `science/lester-sf32-reference-trackers-scalings` | UNDERSTAND: SF-31 `done` on the default branch; no other increment nonterminal (SF-33 stays READY for a concurrent session; remote isolated by `--increment SF-32`). Human-review increment (new tracker consuming `psi1`/`psi2`, scientific comparison with the paper). Scientific-rigor skill invoked (categories B and C; interpretation bounded: numerical spreading on the paper's surrogate, nothing about Darcy or `alpha_T`, O3). Source check (Lester 2023 §5.1-5.4, eqs. 13, 31-38, figs. 3-4): face velocities by the Stokes line integral `Delta^-2 oint psi1 grad psi2 . dl` (eq. 32), exactly divergence-free (eq. 33); coarse grids by averaging face velocities; errors after traversing `Omega` are the one-period displacements (eq. 34) because the surrogate's streamlines close, and the label errors (eq. 35); Pollock variances `~ (Delta/Delta0)^2`, RK variances `~ sqrt(tol)` with an unspecified 4th-order adaptive pair; the paper does not run its own pseudo-symplectic method in the advective case (`delta = 0` trivially). Numerical contract fixed before any code (orchestration record `understanding.md` §3): one edge integral per grid edge with composite 4-point Gauss-Legendre on the spline knot intervals (exact for the degree-6 integrand, so the fluxes are exact surface integrals of the spline velocity and the only Pollock error is the RT0 interpolation); Pollock semi-analytical exits with the exit coordinate set exactly, stagnation status 15, no fallback; return map = first return of unwrapped `x1` to `x1_0 + 1` for the three trackers (Pollock lands on the face; pseudo-symplectic and RK land by bisection on the last panel / chunk from a saved state, SF-31 cores reused unmodified), `N_p = 8192` hash seeds on `x1 = 0` shared by all runs. Label routes (D-1): primary = frozen stack on `lester2021`, `eps = 0.25`, at 256^3 (paper grid) and 128^3, final accepted state used as the pair whatever the exit reason (`r_F`, `e_v`, `min abs(c)` recorded; a stack state at `r_F ~ 1e-6` is accepted as exact invariants by definition of the surrogate — deviation from the letter of the spec, for the reviewer); control = analytic SF-31 pair G at amplitude 0.05 splined at 128^3. Ladders: Pollock `Delta/Delta0 = 1, 2, 4, 8`; RK `tol = 1e-4 .. 1e-8` (paper) plus `1e-9, 1e-10`; pseudo-symplectic `tol_psi = 1e-8, 1e-10, 1e-12`. | Pre-registered readings (T1-T4) and predictions (P1-P6) in the record. P2 predicts an RK variance exponent in `[0.9, 1.7]`, OUTSIDE the paper's band `[0.35, 0.65]`: on the C^2 B-spline the velocity is C^1, a knot crossing costs `O(h^3)` and `h ~ tol^(1/5)` gives `std ~ tol^0.6` (SF-31 measured 0.55-0.82); recorded as a finding if observed, nothing tuned (spec policy). P1: Pollock variance slope near 2 (coherent accumulation of the zero-mean `O(Delta)` RT0 error) with slope 3 (random-walk) as the admissible alternative. DAG: N0 Stokes face fluxes, N2a return-map instrument (pseudo-symplectic + RK + label routes) and N4 analysis script in parallel -> N1 Pollock tracker and N3a face-flux tests -> N2b Pollock in the instrument and N3b Pollock tests -> one integrator -> orchestrator-owned detached V100 jobs on `--increment SF-32` -> N5 experiment note -> FINAL_AUDIT. | Commit activation; start the base build on the SF-32 mirror; launch N0, N2a, N4. |
| 2026-10-06T16:00Z | integrated source head `0f5b916` (N0 `d1bafa9`, N1 `5a4db8f`, N3a `9d0b580`, N4 `9e6ad13`, N2a `ef56372`, N2b `ac602f3`, C1 `cb445fb`, N3b `0bcbafd` on the activation head `9db86f0`, base `37bfb25`) | EXECUTE/AUDIT/INTEGRATE. Three corrections of the ORCHESTRATOR's own contract, each from evidence before the affected node ran or before any experiment run (none a worker deviation): (1) face fluxes — a periodic edge array of `oint psi1 grad psi2 . dl` is wrong where `psi1` is affine (numpy prototype: `u - 1 = -(n-1)` on the last row); replaced by the exact periodic decomposition `flux/area = (gbar1 x gbar2).n + area^-1 oint [s1 (grad s2 + gbar2) - s2 gbar1] . dl` (prototype then at 1e-16); (2) Pollock exits — the textbook `ln(v_f/v_p)/A`, `(v_p e^{At} - u_-)/A` lose every digit when two face values agree to roundoff (prototype: garbage exit time on pair A); replaced by `log1p(A d/v_p)/A`, `v_p expm1(A t)/A`, exit iff `1 + z > 0`, no threshold (prototype then exact to 4e-16); (3) RK `dt_max = h/2` caps the DP5(4) step so that the controller is inactive over the whole tolerance ladder (measured on G 32^3: bitwise identical results at `tol = 1e-4, 1e-6, 1e-8`); replaced by `dt_max = 0.25` absolute, chunk = `dt_max` (D-2). Two pre-registered checks were degenerate and restated with disclosure: P3 (`x1 = 0.5` is an exact return point of pair B) -> targets 0.25/0.30; the controls' pair-B Pollock variance (pair B is exact for Pollock over a period: separable velocity) -> pair G (C1). Nodes: N0 `StokesFaceVelocity` (one exact GL4 edge integral per edge on the spline knot intervals; S1-S6 at 1e-15; orchestrator driver: divergence 0..1e-15, host==device 3e-15, pair A exact 1e-15, additivity 4e-16, 0.61 s at 128^3); N1 `PollockTracker` (host/device core + engine; INDEPENDENT numpy replay on the same N0 fluxes and seeds: positions to 1.1e-16, tau to 4.4e-16, cell counts identical, 256/256 on G 32^3 and B 16^3; D-3 rounding guards for the reviewer); N3a/N3b contract tests by independent authors (96 and 32 checks, gates as pre-registered; P3 orders 1.4-4.0); N2a instrument (`solve-labels` via the frozen stack's public API, analytic labels, PS/RK return maps by bisection landing on the SF-31 cores unmodified, schema of `dag.json`; V100 digits = local digits); N2b Pollock in the instrument + `--dt-max` + static library + controls ctest; C1; N4 launcher + `analyze.py` (pre-registered constants in one block, OLS fits, two-level estimates, band verdicts, figures 3b/3c/4a/4b analogues, self-test recovers synthetic slopes to 1e-12). All seven worker trees compiled on V100 (nvcc 11.4) before integration. One integrator, eight cherry-picks, one textual CMake conflict (both test blocks kept), no integration-only change; every node file blob-identical to its approved version; frozen paths empty vs `37bfb25`; `ctest -N` 20 -> 23. | Labels (job `sf32-labels`, 15:42-15:47Z, mirror `SF-32-labels`, N2a binary `ef56372`; the integrated binary regenerates them bytewise, job `sf32-labels-equiv`): frozen stack on `lester2021` `eps = 0.25` at 128^3 reproduces SF-30 digit for digit (176 it., stagnated, `r_F` 1.088e-6, `e_v` 2.720e-4, `min|c|` 0.775, 147 s); 256^3: 61 it., stagnated, `r_F` 2.970e-5, `e_v` 6.776e-5 (ratio 4.01: order 2), `min|c|` 0.7746, 99 s; analytic G 0.05 at 128^3 `min|c|` 0.7008. Stack states accepted as the pair under the D-1 guard (`tolerance_met = false`; for the reviewer). Prototype predictions before any run: P1' Pollock variance slope ~3 (random-walk law) outside the spec band; P2 RK ~1.2 outside the paper's band (C^1 spline velocity). | V100 on `0f5b916`: `sf32-int-build`, `sf32-ctest-full`, `sf32-ladders`; then the experiment note and FINAL_AUDIT. |
| 2026-10-06T18:48Z | FINAL_AUDIT positive on `0f5b916`; State -> `awaiting_review` (human-review increment: new tracker consuming `psi1`/`psi2`, scientific comparison with the paper, out-of-band exponents left to the review per the spec) | V100 (`scripts/remote --increment SF-32`, mirror `~/MacroFlow3D-SF-32`, Tesla V100, nvcc 11.4, `v100-release`, detached jobs): `sf32-int-build` exit 0, `ctest -N` = 23, `ctest -R 'tracker|spurious'` 5/5; **`sf32-ctest-full`: 23/23 passed, `Total Test time (real) = 2752.88 sec`**; `config_pspta_small` smoke exit 0; `sf32-ladders` (16:08-16:10Z) + `sf32-brk` (16:14-16:21Z): five label fields x 14 runs (Pollock `Delta/Delta0 = 1,2,4,8`; RK `tol = 1e-4..1e-10`, `dt_max = 0.25`; PS `tol_psi = 1e-8, 1e-10, 1e-12`, `ds = h/2`), 8192 seeds each, 70 runs exit 0, 8192/8192 active everywhere, landing <= 1e-12, every statistic recomputed from the per-seed data by `analyze.py` (differences 0). **F-SYM (not predicted):** on both `lester2021` stack pairs the Pollock return map is exact to roundoff at every `Delta` (variances 2.8e-34..3.1e-31, max |delta| <= 3.9e-15): the field is mirror-symmetric under `x1 -> 1/2 - x1` (label fluctuations even to 3e-16 / 5e-15), the Stokes fluxes on an even Pollock grid inherit it and the RT0 trajectory mirrors back to its start — the spec's primary field is degenerate for the Pollock question (the same symmetry made it the closure control of SF-30). Post-hoc field, disclosed with its prediction before any `lester_brk` result existed: frozen stack on `lester_brk` (`eps = 0.25`, 128^3 and 256^3; `r_F` 3.8e-5 / 1.3e-5, `e_v` 4.6e-4 / 3.8e-4, `min|c|` 0.794). Exponents (variance vs `Delta` / `tol`, OLS, verdicts as printed): analytic G 128^3 Pollock **3.030 / 3.040** (`x2`/`x3`, R^2 0.99998/0.99947, two-level 2.83-3.15); `lester_brk` 1.627/1.528 (128^3), 2.358/1.776 (256^3; `x3` in band; `x2` two-level 1.07 -> 3.23); RK on the paper's five tolerances 1.90/1.85 (G), 1.40/1.39 and 1.71/1.67 (`lester2021`), 1.22/1.42 and 1.48/1.71 (`lester_brk`), all seven 0.90-1.45 with a plateau below `tol ~ 1e-7` whose level falls with the label grid; PS `max|delta_psi_i| <= tol_psi` in 15/15 runs (ratio 0.0085-1.0), transverse RMS 0.02-0.4 `tol_psi`. Experiment note `docs/experiments/2026-10-06-sf32-spurious-spreading.md` (N5, independent author; it corrected five imprecisions of the orchestrator's prompt from the artifacts and found that the orchestration record's clock labels after 15:00Z were estimates — erratum appended to the record with the git/log times; orderings that matter unaffected). Artifacts (24 MB) under `docs/experiments/artifacts/2026-10-06-sf32-spurious-spreading/` incl. the pre-registration record and the orchestrator prototype. | Acceptance thresholds: (T1) Pollock `p = 2 +- 0.3` over >= 3 grids: NOT MET on the exponent-bearing fields (3.03/3.04 on G: the random-walk law, pre-registered P1'; `lester_brk` pre-asymptotic 1.5-2.4) — RECORDED, not tuned; (T2) RK `p = 0.5 +- 0.15`: NOT MET anywhere (1.2-1.9; C^1 knot plateau, pre-registered P2) — RECORDED; (T3) PS drift <= 10 x `tol_psi`: MET with a 10x margin; (T4) out-of-band recorded, not tuned: MET (no ladder point, seed set, norm, controller or band changed after a result; the brk field and the restated controls are disclosed additions). Gate 1 + Gate 2 + Gate 4; Gate 4 statement: every transverse displacement measured is numerical by construction and classified as such; the pseudo-symplectic tracker shows none beyond `tol_psi`; nothing is inferred about physical transverse macrodispersion, `alpha_T` neither presupposed nor measured, eq. 36 numbers are protocol numbers (O3). For the human reviewer: the spec bands (the paper's empirical values) vs the measured laws; D-1 (stack states at `r_F` 1e-5..1e-6 accepted as the pair), D-2, D-3; the limited comparability of the RK exponent (unspecified RK4 pair/controller/spline in the paper); unexplained: the RK tight-tolerance floor and a 1.5e-3 offset of the flux-weighted `<tau>` on G. | Push the delivery branch and open the PR; stop at AWAIT_HUMAN_REVIEW; do not set `done` or check the master entry. |
| 2026-10-06T18:49Z | PR #50 opened at head `8551d61` (artifacts, experiment note and metadata above the audited source head `0f5b916`) | PUBLISH_PR: delivery branch pushed over SSH and PR opened against `master`. The PR body lists Goal, module/instrument/test layout, the measurement, the three disclosed contract corrections, the acceptance table (T1/T2 not met and recorded; T3/T4 met), finding F-SYM, the pre-registered predictions and their outcomes, the detached V100 jobs, the Gate 4 statement and the decisions for the reviewer (spec bands vs measured laws; D-1, D-2, D-3; RK comparability). Human-review increment: `done` and the master-checklist entry are deferred to the closure-only metadata commit after explicit approval of the source-bearing head `0f5b916`. | Checklist items 1-4 and 6 checked; items 5 and 7 open. `check-lester-increments.sh` OK (nonterminal=SF-32, ready=SF-33). | AWAIT_HUMAN_REVIEW. On approval: closure metadata commit on this PR; after merge: `scripts/remote remove-mirror SF-32` and `SF-32-labels` (copy the raw `.bin` labels out first if wanted). |
| 2026-10-07T13:13Z | closure metadata commit on PR #50; State -> `done` | Human review recorded: explicit owner approval ("Muy bien, apruebo la PR, hacé el cierre formal") given in the orchestrator session on 2026-10-07, after the review summary of PR #50 and an explanation of the out-of-band exponents had been presented. It refers to PR #50 at head `bba5ebe`, whose audited source-bearing head is `0f5b916` (verified before closure: `git diff 0f5b916 origin/science/lester-sf32-reference-trackers-scalings -- . ':!docs'` empty; PR OPEN, MERGEABLE, not merged, no review or comment objects; `origin/master` still `37bfb25`). No GitHub review object is claimed. The approval accepts the recorded findings as findings: the out-of-band Pollock (3.03/3.04 on G; pre-asymptotic on `lester_brk`) and RK (1.2-1.9, C^1 plateau) exponents stay recorded against the spec bands, with no tuning; F-SYM (the Lester 2021 field is degenerate for the Pollock question) stands as recorded. | Checklist complete (7/7); master-checklist entry checked in this branch; `check-lester-increments.sh` OK. No source, test, config, or numerical change in this commit. Items the approval leaves open as recorded in the experiment note (not conditions of closure): the RK tight-tolerance floor and the 1.5e-3 flux-weighted `<tau>` offset on G (unexplained); whether the spec bands or the measured laws govern future tracker benchmarks (a later decision record); the Lester 2021 field must not be used as a Pollock benchmark. | READY_FOR_HUMAN_MERGE. After merge: `scripts/remote remove-mirror SF-32` and `remove-mirror SF-32-labels` (raw `.bin` labels live there; copy out first if wanted). |
