# SF-42 — Tooling and verification for the confirmatory macrodispersion study

- State: `awaiting_review`
- Goal: `Dotar al estudio confirmatorio de macrodispersión de las herramientas y la verificación que SF-39 no tuvo: Par2 con el campo SF-18, inyección en ventana, tolerancia del solver igualada y salida de cruces de plano; regla de red de semillas sin caras para cualquier malla y ventana; lector y reducción por planos de control en ambos códigos; orden continuo del solver de Darcy Dirichlet por solución manufacturada; contraste de la velocidad de volúmenes finitos con el solver pseudo-espectral a sigma^2 de 0.25 a 4; biblioteca estadística con presupuesto de error y simulación de cobertura; pilotos de diseño, sin cambiar ningún resultado de los caminos existentes.`
- Depends on: `SF-38, SF-39, SF-41`
- Unlocks: `SF-43`
- Branch: `science/lester-sf42-confirmatory-tooling`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 (new options verified on exact controls; existing paths byte-identical) + verification records (MMS order, spectral cross-check, coverage simulation, design pilots)`
- Human review: `required`
- Owner: `Claude Opus 5.5 orchestrator (session 2026-10-30)`
- Started: `2026-10-30T03:40Z on base 3164e4f`
- Completed: `not completed`
- PR: `#85 https://github.com/santi-esquerre/MacroFlow3D/pull/85`
- Commit: `source-bearing audited head fd703ec118db32f07956689d6bf2cf9c2f5d6f13`

## Scientific or engineering intent

Decision record `docs/decisions/2026-10-29-sf39-remediation-sf42-sf43.md` (G1, G4-G8). SF-39 (the exploratory pilot) could not validate
`alpha_L` against Par2 (arms unmatched, W1), had no continuum order for the Dirichlet solve and no FV-independent comparator (W4), used
time-window estimators that select surviving particles (W13), and lacked a tested statistics toolchain and reproducible evidence (W2, W9,
W10). SF-42 builds and verifies what SF-43 needs. It produces no macrodispersion estimate; its pilots are design inputs with seed ranges
disjoint from SF-43's.

## Preconditions

- SF-38, SF-39 and SF-41 `done` on the default branch (SF-39's evidence, `reduced/` and `summary.json`, is read from there).
- The SF-39 host outputs (`$HOME/sf39_campaign`, per-particle arrays and Par2 per-realization series) are kept until the SF-43
  pre-registration commit.
- One GPU, the least loaded, through `scripts/remote run` (decision `2026-10-27-remote-single-gpu.md`).

## In scope

1. **Par2 on the SF-18 field.** An opt-in configuration of `macroflow3d_pipeline` (`transport.method: par2`) that takes its K from the SF-18
   periodic normalized generator without the streamfunction stage, in box units (`dx = 1/ny`, `corr_length = ell`, west head `L1`,
   `inject_x = 5 ell`, the consumer's `%.17g` strings), with `Y` normalized to variance 1 and `ln K = eps Y`, `K = exp(ln K)` computed as the
   consumer does (host `std::exp`), so that `Y` and `K` are bitwise equal to `lagrangian_ensemble --medium b` for the same seed. It must reuse
   the existing log-K workspace (Par2's memory ceiling is about 2e8 cells).
2. **Par2 injection window.** An opt-in transverse window `[w2a, w2b] x [w3a, w3b]` for `inject_box` (`src/runtime/ensemble/EnsembleRunner.cu`).
3. **Par2 solver tolerance and plane crossings.** Par2 runs at `cg_rtol` 1e-10, `cg_max_iter` 2000 when configured (an existing key); a new
   opt-in per-realization output of plane crossings at a list of planes: per plane, the crossing-time mean and variance and the transverse
   crossing moments (means, variances, covariance) of the particles, uniform weights, written per realization.
4. **Seed-lattice rule (no runner change).** A script that, for any grid, injection window (box fractions, including the full plane
   `[0, 1]^2`, which the runner already accepts) and lattice `M`, predicts the runner's RR12 seed-on-face check, and lists admissible `M`.
   Known cases: with the full plane, `v2(M) >= v2(ny)` (`v2` the power of 2 in the factorization); with the window `[0.1, 0.9]^2`, powers of
   two fail whenever 3 divides `ny` (for example `ny = 384`, `M = 128`), while SF-39's `M = 100` passes on its grids. Seeds per `lambda` of
   source are kept constant across arms of an amplitude (the source has a fixed size in `lambda`, decision G4).
5. **Reader and host-side reduction (both codes).** Per realization: the plane-based estimators of decision G4 (`alpha_TP`, `alpha_TAP`,
   `alpha_LP`) on a travel window; position-based and displacement-based (seed-subtracted) transverse variants; the time-window estimators of
   SF-39 (secondary) ending at the first exit; full transverse covariances read from the per-particle arrays where the moment CSVs lack them.
   Output: per-realization reduced tables. A pruning tool that records the sha256 of every per-particle array, keeps a pre-registered subset,
   and deletes the rest only after the reduced tables are written and verified.
6. **Manufactured-solution order of the Dirichlet solve.** A new driver executable (new source file and CMake target; no change of
   production code) that forms the manufactured right-hand side plus the Dirichlet terms of the production RHS builder, solves with the
   production operator, multigrid preconditioner and PCG settings, and computes face fluxes with the production velocity routine, in
   medium (b) geometry. Manufactured `K = exp(eps Y_a)` with `dK/dx != 0` at the Dirichlet faces; head affine in `x` plus a smooth
   perturbation; at least four grids; `eps` 0.5, 1, 1.5, 2; errors in L2 and max norms for the head and the interior face fluxes, the max norm
   restricted to faces at least two cells from the Dirichlet faces, and, separately, for the Dirichlet faces and the first interior layer.
7. **Pseudo-spectral cross-check of the FV velocity.** The closure-probe spectral solver
   (`docs/experiments/artifacts/2026-10-02-closure-probes/scripts/`) extended to import the SF-18 field (the generator's modes, or a sample
   at the reference resolution `2N` with phase and scale handled; never the FV sample), preconditioned by `K^{-1/2} (-Delta)^{-1} K^{-1/2}`,
   failing when the residual exceeds tolerance, with a self-convergence check (`N` against `2N`). Compared with the SF-19
   `AffinePeriodicFlowSolver` (same interior scheme as `solve_head`) at `sigma^2` 0.25, 1, 2.25, 4, at least two seeds, box at least 4
   `lambda`, FV `lambda/h` 4, 8, 10, 16, 32 (64 where it fits).
8. **Statistics and error-budget library** (stdlib + numpy + scipy; host stack Python 3.11 / numpy 1.26 / scipy 1.11 and the current stack):
   per-realization slope estimators; Student and bootstrap-t intervals, BCa for reporting; the robust lower limit of decision G5.4
   (Bonferroni-split one-sided upper bounds of the bias components, joint seed bootstrap across arms sharing seeds, the p-value `p_e`, the
   intersection-union amplitude p-value); the `E_h` rules (observed order with its lower confidence limit; conservative fallback for
   non-monotone and oscillating sequences); the `E_W` bound; the paired log-ratio TOST and its power at a Holm level; the split-plot
   factorial analysis of G7 (two-stratum variance, aliases stated); Holm; a coverage simulator (per-tail miss rates at the tail levels SF-43
   will use, at least 175 / tail replicates, stress family, whole robust-sign pipeline, over a range of `n`; host compute budgeted); the
   blinded variance re-estimation tool (outputs standard deviations only); a cost model calibrated on SF-39 and the grid pilot.
9. **Design pilots (seed ranges disjoint from SF-43's; design inputs, not evidence).**
   - Grid feasibility: one realization at the highest planned `sigma^2` on every planning grid of SF-43 not run by SF-39 (at least
     520x120x120, 832x96x96, 832x192x192, 1040x240x240, 1040x480x480, 2080x480x480, 2496x288x288, and 1248x288x288 and 1664x192x192 in both
     codes), and Par2 at `cg_rtol` 1e-10 on the Gate 5 grids: PCG iterations, time and peak memory.
   - Particle count: three lattice sizes admissible by the seed-lattice rule (about 0.6x, 1x and 2x SF-39's seeds per `lambda`) on one
     configuration per `sigma^2` (a few realizations): the split-half particle share of each estimator's variance.
   - Estimator noise: position-based against displacement-based transverse variance, from SF-39 host arrays (no new runs).
   - Gate 5 pairing: the correlation between matched Par2 and consumer `alpha_LP` on the same SF-18 fields (a few realizations).
10. **Reproducibility tooling.** sha256 lists of every reducer input; a recomputation script independent of the analysis package, from raw
    per-realization moment files; a scripted reconciliation of solver verdicts against job logs; `scripts/remote exec` logging during a
    declared blinded phase; runbook text for the pre-registration commit and the blinded phase.

## Out of scope

- Any macrodispersion estimate (SF-43); the pilots are design inputs only.
- Changes of defaults or of existing outputs (Par2 baseline, `lagrangian_ensemble` defaults, SF-39 paths, `scripts/regress` set v2).
- Memory engineering (G3); the open questions G9a and G9b; finite Péclet; other covariance models.

## Files and symbols

- Par2: `src/io/config/Config.hpp`, `src/io/config/ConfigYaml.cpp` (new opt-in keys and their validation),
  `src/runtime/ensemble/EnsembleRunner.cu` (K source, `inject_box`, plane-crossing output), `src/physics/stochastic/` (SF-18 generator entry
  points), `src/physics/particles/par2_adapter/` and `src/runtime/stats/` (crossing statistics), `src/io/writers/` (per-realization output).
- Consumer (read only): `src/physics/particles/lagrangian_labels/EnsembleRunner.cu` (window parsing, `find_seed_on_face`),
  `LagrangianTracker.cu` (seed lattice), for the seed-lattice rule script.
- Flow (read-only production code, reused): `src/physics/flow/solve_head.cu`, `rhs_head.cu`, `velocity_from_head.cu`,
  `src/physics/particles/lagrangian_labels/DirichletLongBoxStage.cu`, SF-19 `AffinePeriodicFlowSolver`; new MMS driver source and CMake target.
- New artifact root `docs/experiments/artifacts/<date>-sf42-confirmatory-tooling/` (`scripts/`: reader and reduction, pruning tool, MMS and
  spectral drivers, `mcstat/` with tests, recomputation and reconciliation scripts, field export), experiment note
  `docs/experiments/<date>-sf42-confirmatory-tooling.md`.
- `scripts/remote` (exec logging); `docs/runbooks/remote-v100.md`, `docs/runbooks/lester-increment-workflow.md`.

## Implementation specification

1. New keys and flags are opt-in, validated (unknown or inconsistent combinations rejected with a message) and recorded in the effective
   config and manifest; without them the code path and outputs are byte-identical to the base.
2. Field identity: dump `Y` and `K` from Par2 (new option) and from the consumer for one realization on a small grid and compare bitwise.
3. Plane crossings in Par2: verify on a small case against crossing statistics computed by brute force from dense snapshots.
4. The MMS and spectral studies, the pilots and the coverage simulation run as detached `scripts/remote run` jobs; drivers, configs and
   outputs are artifacts; results and exact commands go to the experiment note.
5. Library known-answer tests: synthetic sequences with known order and limit (monotone, non-monotone, oscillating); synthetic normal and
   skewed samples for per-tail coverage within Monte Carlo error; analytic TOST power; the union-bound robust limit on a synthetic case with
   known biases. Reproduction tests against deterministic SF-39 numbers computed from committed data (`summary.json`, `reduced/`): the Gate 5
   Welch interval at 0.25, the ratio-consistent contrast [-0.0466, -0.0014], the F4 ratios, at tolerances that allow the host scipy quantile
   differences (1e-9 relative). Tests that need Par2 per-realization values (on the host only) are host-only jobs.
6. The independent recomputation does not import the analysis package and reproduces, from raw per-realization moments on the host, the
   reduced values of at least three SF-39 realizations per amplitude.

## Expected numerical effect

None on existing paths. Par2 on the SF-18 field: `var Y = 1` and `var ln K = eps^2` to rounding. MMS: order near 2 for the head and the
interior fluxes, near 1 for the Dirichlet face fluxes. Spectral cross-check: FV velocity error order near 2 in the asymptotic range (SF-33
measured 1.92-2.04 at small amplitude).

## Validation commands

```bash
# local fixed loop (docs/runbooks/lester-increment-workflow.md section 8)
cmake --preset wsl-debug && cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R '<new config-validation and option tests>'
cd docs/experiments/artifacts/<date>-sf42-confirmatory-tooling/scripts && python3 -W error -m unittest discover -s . -t .
# remote, detached, one job at a time (never --gpu / CUDA_VISIBLE_DEVICES; REMOTE_GPU_MIN_FREE_MIB=30000 for CUDA parts)
REMOTE_GPU_WAIT=7200 scripts/remote --increment SF-42 run sf42-base -- "<clean v100-release build of the increment base + scripts/regress refs>"
REMOTE_GPU_WAIT=7200 scripts/remote --increment SF-42 run sf42-head -- "<v100-release build of the head; scripts/regress/run.sh then compare.sh>"
REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=30000 scripts/remote --increment SF-42 run sf42-ctest -- "ctest --test-dir build/v100-release --output-on-failure"
REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=30000 scripts/remote --increment SF-42 run sf42-identity -- "<field identity, injection, crossings checks>"
REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=30000 scripts/remote --increment SF-42 run sf42-mms -- "<MMS driver>"
REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=30000 scripts/remote --increment SF-42 run sf42-spectral -- "<spectral cross-check>"
REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=30000 scripts/remote --increment SF-42 run sf42-pilots -- "<grid, particle, pairing pilots>"
REMOTE_GPU_WAIT=7200 scripts/remote --increment SF-42 run sf42-coverage -- "<coverage simulation on the host stack>"
bash scripts/hooks/check-lester-increments.sh
```

## Acceptance thresholds

- **A. Byte identity.** `scripts/regress` set v2 `REGRESS IDENTICAL` against references built from the increment base; one default
  `lagrangian_ensemble` medium (b) realization byte-identical to the base binary's; Par2 configs without the new keys byte-identical.
- **B. Field identity.** `Y` and `K` bitwise equal between Par2 (new option) and the consumer for the same seed (one realization, small
  grid); `var Y = 1` to 1e-12 relative.
- **C. Injection and crossings.** Par2 window: every injected particle inside the window. Seed-lattice rule: its predictions match the
  runner's RR12 outcome on a sample of grids and windows (including the failing cases above), and it lists an admissible `M` for every SF-43
  planning grid and source. Par2 plane crossings equal the brute-force reference to 1e-12 relative on the test case.
- **D. MMS.** Head and interior face fluxes: observed order at least 1.8 over the two finest grid pairs at every `eps`, in the L2 norm and in
  the max norm restricted to faces at least two cells from the Dirichlet faces; the Dirichlet faces and the first interior layer reported
  (order near 1 expected). Otherwise `blocked`, with the measured orders recorded.
- **E. Spectral cross-check.** The reference converges (`N` against `2N` error at least 10 times below the finest FV error) at every
  `sigma^2`; the FV velocity error decreases monotonically along the ladder with observed order at least 1.5 over the finest pair; the
  relative L2 error at `lambda/h` 10, 16 and 32 recorded per `sigma^2`. Otherwise recorded as a finding; Gate 2 waits for the owner's
  decision on budgeting it in SF-43.
- **F. Library.** All known-answer and reproduction tests pass on both stacks; the coverage simulation is reported over a range of `n` for
  every planned SF-43 procedure at the planned tail levels with at least 175 / tail replicates (SF-43 re-runs it at its final `n` and
  pre-registers the procedures that meet G5.3).
- **G. Pilots.** Recorded: PCG iterations, time and memory per new grid (a pre-stated rule: more than 3 times SF-39's iterations requires a
  different grid in SF-43); particle shares per `M`; position against displacement variance; Gate 5 pairing correlation.
- **H. Reproducibility tooling.** The independent recomputation reproduces at least three SF-39 realizations per amplitude to 1e-10
  relative; the reconciliation script reproduces SF-39's four fallback or non-converged solves from the job logs; `exec` logging and the
  runbook text in place.

## Regression surface

- Par2 config parsing, the ensemble runner and the stats writers: every existing Par2 config (byte identity, A).
- The consumer: no change (byte identity of one default realization, A).
- `scripts/remote`: every remote workflow (logging must not change command semantics or exit codes).
- CMake: the new driver target must not change existing targets.

## Failure and rollback policy

- A byte-identity failure (A) is a defect, fixed before anything else.
- D or E failing leaves SF-42 `blocked` with the measurements recorded; the owner decides whether SF-43 proceeds with a budgeted error.
- New options that cannot be byte-neutral by default are reverted.

## Completion checklist

<!-- completion-checklist:start -->
- [x] Implementation matches the scope and contains no unrelated changes.
- [x] Targeted validation passes and its evidence is recorded.
- [x] Required regression tests pass.
- [x] Scientific or engineering findings are appended to the bitácora.
- [ ] Required human review is recorded.
- [x] PR and commit identifiers are recorded.
- [ ] The master checklist entry is checked in this branch and `check-lester-increments.sh` passes.
<!-- completion-checklist:end -->

## Advancement rule

Merging SF-42 `done` (with SF-39 `done`) makes SF-43 eligible (READY).

## Bitácora

Append entries; do not rewrite prior observations. Store large raw outputs as artifacts or experiment notes and link them here.

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-10-29T21:00Z | `pending`; governance PR (docs only) | Specification created on the governance PR of decision `2026-10-29-sf39-remediation-sf42-sf43.md` (G1, G4-G8), after the SF-39 post-review evaluation and three adversarial reviews of the draft. | Owner decisions G1-G4, G10, G11 of 2026-10-29. | Merge of the governance PR after PR #83; activation when SF-38, SF-39 and SF-41 are `done` on the default branch. |
| 2026-10-30T03:40Z | `active`; base `3164e4fb5c2e3bebf4fdc9ae5647bc5b1d27384a` | Activated by the owner's request in the orchestrator session. Preconditions verified: local `master` fast-forwarded over HTTPS to `3164e4fb` (merge of the governance PR #84, merged 2026-10-30T03:00:37Z); `check-lester-increments.sh` on that tree: `OK (44 increments, ready=SF-34 SF-42, nonterminal=none)`, so activating SF-42 leaves one nonterminal increment. Owner constraints for this run: one remote job at a time, `scripts/remote --increment SF-42` (own mirror), `REMOTE_GPU_WAIT=7200`, `REMOTE_GPU_MIN_FREE_MIB=30000` for CUDA parts, never `--gpu` / `CUDA_VISIBLE_DEVICES` / GPU work under `exec`; SF-39 host outputs (`$HOME/sf39_campaign`, 169 GB) not to be modified or deleted; GPU-h of every job recorded here; stop and consult the owner if the pilots project SF-43 above the G10 budget (about 150 GPU-h); D or E failing leaves SF-42 `blocked`. | Owner request 2026-10-30; checker output above. | UNDERSTAND, intra-increment DAG, delegated nodes in isolated worktrees. |
| 2026-11-01T21:40Z | `active`; owner consultation (owner run constraint: pilots projecting SF-43 above the G10 budget of about 150 GPU-h mean stop and consult) | Independent read-only audit AUD-GH-1 (major): the R4 SF-43 projection (105.0 GPU-h, +1 rms 121.2; pre-stated STOP rule not triggered) used `M` = 64 at `sigma^2` 0.25 and 1, which is not admissible on every arm of those amplitudes (only `M` = 128 is; spec item 4 keeps seeds per `lambda` constant across arms). Admissible design `M_common_128_128_86_68`: 118.2 (model) / 136.5 (+1 rms) / 147.4 (anchored to J2) / 170.1 (anchored +1 rms) GPU-h, simulation only; excluded: SF-43's coverage re-run, reductions and job overhead (+10.1 % measured in J2; 147.4 x 1.101 = 162.3), builds and test jobs. Verified by the orchestrator against `R/pilots/cost_projection.json` and `seedlattice_plan.md`. | Owner decisions 2026-11-01 (session): (1) continue SF-42 to its PR; SF-43's budget and design are an open owner decision at its pre-registration; G10 unchanged; (2) CPU-only jobs that hold the single project slot do not count toward G10 and are reported separately as CPU slot-hours; (3) V100 re-verification of the corrected tree plus the three missing grid pilots (832x384x384, 2080x240x240, 416x96x96) authorized. Recorded in the note (Commands preamble, Findings (i), Deviation 11). | Corrective nodes C1, C2, R6; integration; final audit. |
| 2026-11-02T01:15Z | `awaiting_review`; GPU-h ledger of every remote job (R1 to R6; mirror `~/MacroFlow3D-SF-42`, one job at a time, `REMOTE_GPU_WAIT=7200`, CUDA jobs `REMOTE_GPU_MIN_FREE_MIB=30000`, no `--gpu`) | Slot-hours per job (first to last helper-log stamp; G = CUDA, C = CPU-only). R1: `sf42-base-build` 0.087 C, `sf42-base` 0.058 G, `sf42-head-build` 0.086 C, `sf42-head` 0.059 G, `sf42-bytes` 0.006 G, `sf42-ctest` 0.511 G, `sf42-identity` 0.014 G. R2: `sf42-r2-build` 0.087 C, `sf42-mms` 0.424 G, `sf42-spectral-fv` 0.082 G, `sf42-spectral-ref` 2.496 C. R3: `sf42-hosttests` 0.011 C, `sf42-recompute` 0.004 C, `sf42-reconcile` 0.001 C, `sf42-noise` 0.236 C. R4: `sf42-r4-build` 0.092 C, `sf42-pilot-grid` 0.154 G, `sf42-pilot-particles` 1.491 G, `sf42-pilot-pairing` 0.432 G, `sf42-pilot-horizon` 0.118 G. R5: `sf42-coverage` 41.693 C (72 workers, 2995 host core-hours, about 6x the docstring projection). R6: `sf42-r6-build` 0.085 C, `sf42-r6-head` 0.053 G, `sf42-r6-bytes` 0.003 G, `sf42-r6-identity` 0.007 G, `sf42-r6-ctest` 0.511 G, `sf42-r6-grid` 0.030 G, `sf42-r6-hosttests` 0.013 C (exit 1, see next row), `sf42-r6-hosttests2` 0.013 C. **Totals: GPU-h (CUDA jobs) 3.951; CPU-only slot-hours 44.904; slot-hours held 48.856.** All exits 0 except `sf42-r6-hosttests`. `~/sf39_campaign` untouched (every `find -newer` check empty). | `R/verification/jobs*.tsv`, `R/verification/logs/`, note Commands "Ledger of slot time (R1 to R6)"; owner decision (2) of 2026-11-01 (CPU-only slot-hours outside G10). | Human review. |
| 2026-11-02T01:20Z | `awaiting_review`; source-bearing audited head `fd703ec118db32f07956689d6bf2cf9c2f5d6f13` (integration: ff-only of the 37 accepted commits on `5f48caf9`) | **Nodes accepted after orchestrator audit** (runtime records `.claude/orchestration/SF-42/audits/`): N1, N1b, N2/N2c, N3, N4, N5/N5c, N6a, N6b, N6c, N7, R1 (A, B, C, Gate 1), R2 (D, E), R3 (F host, G noise, H), R4/R4b/R4c (G pilots, cost model), R5 (coverage job; worker stopped by the owner while polling, job kept running) and R5b (report corrections: the nominal false-fail probability of G5.3 is about 0.08 per cell-distribution and 0.22 per three-distribution group, not about 1e-3; results, section F). **Five independent read-only audits** (A/B/C, D/E, F, G/H, scope/docs) supported A to E and found: a validator gap (`sf18_medium_b` + `periodic_gaussian` streamfunction stage read an uninitialized `Y_field`; `dx = 1/ny` not enforced), an erasable blinded-phase log, no host-stack run at the head, the cost-margin issue (previous row) and note inaccuracies; corrective nodes **C1** (validator, recorder doc, independent crossing reference r0/r1, on/off byte control, blinded-log protections, runbooks), **C2** (G5.4 split known-answer tests, 1e-12 SF-39 contrast, prune hardening) and **R6** (V100 re-verification of the corrected production tree: regress v2 27 cases REGRESS IDENTICAL vs the base, consumer 41 and Par2 112 files byte-identical to the base binaries, FIELD_IDENTITY x4 with abs(var Y - 1) <= 2.2e-16, window 2000/2000, `par2_sf42_options` 101/0, seed-lattice probe 1575/0, full ctest 43/43; grid pilots J1B; host suite 369 OK with 0 skipped after fixing one C2 test reference that used a raw scipy 1.11 quantile; ledger; note corrections). **Acceptance:** A, B, C, D (min order 1.997; Dirichlet faces 2.00), E (8 ladders, order 2.00), H met; F: library tests pass on both stacks at the final tree, coverage simulation reported in full (1134 cells; 310 groups qualify, 101 clear and 9 marginal failures); G recorded (every SF-43 planning grid not run by SF-39 piloted). No macrodispersion estimate. | Note `docs/experiments/2026-10-30-sf42-confirmatory-tooling.md` (sections A to H, Findings (a)-(j), Conclusion, Caveats, Deviations 1-11); final audit `.claude/orchestration/SF-42/audits/FINAL.md`; checker `OK (44 increments, ready=SF-34, nonterminal=SF-42)`. **Open for the owner / SF-43 pre-registration:** SF-43 budget and design (admissible lattice; no common `M` at 2.25), the 3x PCG rule tripped on recorded counts by 1040x240x240 at 0.25 and 2080x240x240 at 1, the Par2 horizon, the coverage qualification (no method for the paired contrast or Gate 5 TOST under `stress` at the simulated `n`; bootstrap-t unpaired/interaction fail under `stress`; G5.3 false-fail rate), the CPU-only coverage re-run cost. | Push the delivery branch and open the PR for human review; do not set `done`, do not check the dashboard entry. |
| 2026-11-02T01:35Z | `awaiting_review`; PR #85 opened from `science/lester-sf42-confirmatory-tooling` (source-bearing audited head `fd703ec1`; metadata commit `c1466c51` on top) | PR and commit identifiers recorded (metadata only; `git diff --quiet fd703ec1 HEAD -- . ':(exclude)docs/plans'` holds). Review checklist item and dashboard entry left open. | PR body: Goal, delegated DAG, evidence A-H, audit loop, ledger, open owner decisions for SF-43 (budget and design on the admissible lattice, 3x PCG rule at two arms, coverage qualification under stress, coverage re-run cost, Par2 horizon), process disclosures, files left untouched. | Human review of the source-bearing head `fd703ec1`. On explicit owner approval: closure-metadata commit only (record the approval, the human-review item, State `done`, master-checklist entry, checker). The owner merges. |
