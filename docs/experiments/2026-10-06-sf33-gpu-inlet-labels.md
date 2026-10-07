# SF-33: GPU inlet-label streamfunctions (equation (14) on the `x1`-non-periodic slab)

- Date: 2026-10-06 (runs 2026-10-06T18:59Z .. 2026-10-07T13:37Z)
- Status: `partial — claim (a) complete; claim (b) not established at the spec's amplitudes (preconditioner gate) and reached 32^3-64^3 at eps 0.25; owner decision pending`
- Increment: [`SF-33-gpu-inlet-label-streamfunctions.md`](../plans/active/lester-eq14/increments/SF-33-gpu-inlet-label-streamfunctions.md)
- Theory: [`docs/theory/lester-2023-key-claims.md`](../theory/lester-2023-key-claims.md) §3A (equation (14),
  same-index pairing, the inlet-label target)
- Decision record: [`2026-10-06-eq14-inlet-label-formulation.md`](../decisions/2026-10-06-eq14-inlet-label-formulation.md)
  (items 1-7 locked; item 6 = the open `sigma_Y = 1` question)
- Predecessors: [`2026-10-02-sf29-inlet-labels.md`](2026-10-02-sf29-inlet-labels.md) (CPU prototype, R1-R8),
  [`2026-10-05-sf30-streamline-closure-gate.md`](2026-10-05-sf30-streamline-closure-gate.md) (production-stack
  streamline instrument reused by the oracle)
- Artifacts: [`artifacts/2026-10-06-sf33-gpu-inlet-labels/`](artifacts/2026-10-06-sf33-gpu-inlet-labels/README.md)
  (`scripts/`, `logs/`, `raw/`, `analysis/`)

Every number below is read from a committed file named next to it. Abbreviations for the sources (all relative to
the artifact directory `A = artifacts/2026-10-06-sf33-gpu-inlet-labels/`):
`PC` = [`analysis/proto_comparison.md`](artifacts/2026-10-06-sf33-gpu-inlet-labels/analysis/proto_comparison.md) (N6),
`PCB` = [`analysis/proto_comparison_b.md`](artifacts/2026-10-06-sf33-gpu-inlet-labels/analysis/proto_comparison_b.md) (N6b),
`PG` = [`analysis/preconditioner_gate.md`](artifacts/2026-10-06-sf33-gpu-inlet-labels/analysis/preconditioner_gate.md) (N6),
`PGB` = [`analysis/preconditioner_gate_b.md`](artifacts/2026-10-06-sf33-gpu-inlet-labels/analysis/preconditioner_gate_b.md) (N6b),
`XC` = [`analysis/sf19_crosscheck.md`](artifacts/2026-10-06-sf33-gpu-inlet-labels/analysis/sf19_crosscheck.md) (N6),
`L1` = [`analysis/ladder_orders_025.md`](artifacts/2026-10-06-sf33-gpu-inlet-labels/analysis/ladder_orders_025.md) (N8'),
`L2` = [`analysis/ladder_orders_025_c4.md`](artifacts/2026-10-06-sf33-gpu-inlet-labels/analysis/ladder_orders_025_c4.md) (N8''),
`P05` = [`analysis/prod32_05.md`](artifacts/2026-10-06-sf33-gpu-inlet-labels/analysis/prod32_05.md) (N8'),
`OS` = [`analysis/oracle32_vs_sf29.md`](artifacts/2026-10-06-sf33-gpu-inlet-labels/analysis/oracle32_vs_sf29.md) (N8'),
`IDX` = [`analysis/README.md`](artifacts/2026-10-06-sf33-gpu-inlet-labels/analysis/README.md) (job tables of
N6, N6b, N8', N8''). Numbers are rounded to 3-4 significant digits unless a comparison needs more.

## Question

The increment's exact Goal: `Implementar en GPU, dentro de src/physics/streamfunctions/, la formulación de etiquetas
de entrada decidida por SF-29 (slab no periódico en x1, condición de salida, esténciles de cuarto orden y
Newton-Krylov con continuación) y verificar que reproduce el prototipo CPU y converge bajo refinamiento hasta
128^3.`

The four claims of the specification, verbatim (intent paragraph for (a)-(b), acceptance thresholds for (c)-(d)):

- (a) "the GPU code solves the same discrete problem as the SF-29 CPU prototype (identical inputs give identical
  metrics)";
- (b) "on the production stack (SF-18 field, SF-19 Darcy) the labels converge to the Darcy labels under refinement
  beyond the prototype's 32^3, and the open item `sigma_Y = 1` (decision item 6) is measured at `ell/h = 16, 32`.
  It does not claim anything about transport or `alpha_T`.";
- (c) "Gate 3A metrics, V100 wall time and peak device memory reported at 128^3.";
- (d) "Production oracle: round trip <= 1e-8 relative on every plane; the oracle labels at 32^3 agree with the
  SF-29 spectral oracle on the step-8 field to the order of the SF-19/spline discretization (recorded)."

## Hypothesis

### Acceptance thresholds (spec, verbatim; fixed before any run)

- (a) Prototype reproduction (same discrete problem): for every case of step 9a, `e_v` and `e_psi` equal to the
  prototype's `i1o4` CASE values within 1e-6 relative, `r_F <= 1e-10`, and the saved 16^3 solutions reproduced
  field-wise (max relative difference recorded; expected at the level of the solver tolerance).
- (b) Production refinement on the same continuum field: at `eps = 0.5`, `e_v` and `e_psi` observed orders
  >= 1.8 on both pairs of 32/64/128 with `r_F <= 1e-10` and no floor. At `eps = 1` the measured orders are
  recorded: PASS if >= 1.8 on both pairs; otherwise the open item of the decision record stays open with the
  measured orders. No criterion presupposes the answer.
- (c) Gate 3A metrics, V100 wall time and peak device memory reported at 128^3.
- (d) Production oracle: round trip <= 1e-8 relative on every plane; the oracle labels at 32^3 agree with the
  SF-29 spectral oracle on the step-8 field to the order of the SF-19/spline discretization (recorded).
- SF-19 cross-check (step 8) recorded with its observed order.
- Fast contract tests pass; full `ctest` on V100 green; byte-compare of the default configs clean.

Spec "Expected numerical effect", verbatim: "(a) GPU metrics equal to the prototype's to roundoff amplification
of the same discrete problem; (b) at `eps = 0.5` orders near the prototype's (1.8-2.1 on 20-28) or higher on
32/64/128, limited by the 2nd-order SF-19 inlet velocity and the oracle's spline accuracy; at `eps = 1` the orders
are not predicted (open item of the decision record)."

Spec failure policy, verbatim (it governs the reduced campaign below): "(b) not met at `eps = 0.5`: stop and return
to the owner with the ladders (no tolerance or threshold tuned); the decision record's validity claim is then
re-examined." and "Preconditioner not validated against direct solves: the 64^3-128^3 runs do not start."

### Readings and decisions of the orchestrator, made before the corresponding runs (deviations for the reviewer)

None of these moves a spec threshold; each restates an orchestrator sub-criterion, fixes an instrument setting,
or records a scope consequence. Dates are UTC days of the decision.

| date | node | deviation / reading | reason and evidence |
|---|---|---|---|
| 2026-10-06 | UNDERSTAND | `src/physics/streamfunctions/Diagnostics.cuh` not applicable; Gate 3A quantities computed by new slab kernels to the prototype's definitions | `Diagnostics.cuh` is periodic, cell-centred, 2nd-order; the slab is non-periodic in `x1`, vertex-based, 4th order. Spec: "through `Diagnostics.cuh` where applicable". |
| 2026-10-06 | UNDERSTAND | `CoupledGmres.cuh` not reused; a new GMRES on raw device vectors (`SlabGmres`) | bound to the periodic coupled layout and mean-zero projection. Spec: "if its interface fits". |
| 2026-10-06 | UNDERSTAND / O1 | byte-compare on the SF-27/SF-26 precedent trio (`config_pspta_small`, `config_streamfunctions_homogeneous`, `config_streamfunctions_continuation`) | the literal `config_pipeline_par2/pspta` are multi-day production runs (same deviation as SF-27). Base references: job `sf33-base-build-refs` (exit 0, [`logs/jobs/sf33-base-build-refs.log`](artifacts/2026-10-06-sf33-gpu-inlet-labels/logs/jobs/sf33-base-build-refs.log)); the after-change comparison is integration validation, recorded in the increment file, not in this artifact. |
| 2026-10-06 | UNDERSTAND / N3 | inlet `v1` = SF-19 U-face flux (face-averaged, 2nd order); `v_perp,in` and the reference `vD` = the SF-28 spline flow that the oracle traces | the two differ at O(h^2), the inconsistency the oracle labels inherit at the inlet; measured order 1.98-2.02 (rms_rel 6.13e-3 / 2.73e-3 / 1.54e-3 at 16/24/32, `XC` "Additional SETUP line"). |
| 2026-10-06 | UNDERSTAND / N4 | stage inputs exported for the whole reachable dyadic amplitude set (ladder `0.25 -> 0.5 -> 1`, <= 4 bisections): 16 / 32 / 48 amplitudes for eps 0.25 / 0.5 / 1 | the driver cannot call Python; replay of all 151 PATH lines of SF-29 `raw/sweep2/summary.md` exact ([`scripts/README.md`](artifacts/2026-10-06-sf33-gpu-inlet-labels/scripts/README.md), "Reachable continuation amplitudes"); missing amplitude = distinct `missing_stage_input` exit. |
| 2026-10-06 | N0 | orchestrator criterion 2 (stencil orders >= 3.8) restated from 16 -> 32 to the prototype's 32 -> 64 gate, plus identity with the prototype's errors at 16 and 32 to 1e-6 | the one-sided plane-`N` rows are pre-asymptotic at 16 identically in the prototype (`d1` order 3.719 on 16 -> 32); 32 -> 64: `d1` 3.934, `d2`/`d3` 3.995, `d11` 4.960, `d23` 3.995 (all >= 3.8); identity with the prototype's errors, e.g. rel diff 1.5e-11 (`d1`), 3.6e-12 (`d23`) ([`logs/jobs/sf33-n0-build.log`](artifacts/2026-10-06-sf33-gpu-inlet-labels/logs/jobs/sf33-n0-build.log)). |
| 2026-10-06 | N1 | JVP-vs-FD check at the EXACT pair read against the measured FD roundoff floor (smallest step gated at/below the floor) | at the exact pair the base residual is at roundoff, so the FD error at the smallest step is roundoff-limited (~2e-10 relative): rule documented in [`tests/inlet_slab/slab_jvp_tests.cu`](../../tests/inlet_slab/slab_jvp_tests.cu) (header item and the floor estimate before `floor_t`). |
| 2026-10-06 | N2 | orchestrator criterion "noisy inlet from x = 0 converges" restated to "noisy inlet converges from the exact pair" | the noisy inlet sits outside the `x = 0` Newton basin (real cases use continuation): from the exact pair 4 Newton steps, `r_F` 7.571e-15; from `x = 0` `linear_failure`, not gated ([`raw/n7b_local/inlet_slab_newton_tests.log`](artifacts/2026-10-06-sf33-gpu-inlet-labels/raw/n7b_local/inlet_slab_newton_tests.log)). |
| 2026-10-06 | N2 | driver default GMRES restart 100 (inner cap 6000) instead of 50 | restart 50 stagnates at true rel 1.507e-2 after 200 its on the perturbed-exact-pair fixture; restart 200 / 400 converge (596 / 287 its, rel 8.5e-14 / 9.8e-14) (same log). |
| 2026-10-06 | N3 | production oracle step bound `h_max = h/8` (SF-30 default `h`); `h/16` and tol 1e-10 reported as a ladder; the orchestrator's own sub-criterion "<= 1e-10 at tol 1e-10" dropped | the round trip is step-limited, ~`h_max^3`, not tolerance-limited: 1.3e-6 / 8.8e-8 / 6.6e-9 / 1.2e-9 / 1.6e-10 at `h_max = h, h/4, h/8, h/16, h/32` (generic3d, eps 0.25, 16^3, tol 1e-8), tol 1e-8 and 1e-10 bitwise identical at `h/16` ([`SlabOracle.cuh`](../../src/physics/streamfunctions/inlet_slab/SlabOracle.cuh) header, lines 29-37). Acceptance (d) kept as written at the production `h_max`. |
| 2026-10-06 | N5 | the 1e-6 criterion of (a) evaluated against full-precision prototype references only: the saved 16^3 solutions and five 24^3 references regenerated with the prototype itself (job `sf33-proto-ref24`); at 32^3 (and `control2d:0.25:24`) bounded by the 4-significant-digit printed CASE values (`PASS(4dig)`) plus `r_F <= 1e-10` | the printed `%.3e` values cannot resolve 1e-6 (rounding bound ~5e-5); a 32^3 prototype reference costs 16 663 s median and 26 GB RSS (SF-29 R7). Reading rule printed in `PC` and `PCB`. |
| 2026-10-06 | N6 -> N7a | after the N6 gate FAIL: inexact Newton (Eisenstat-Walker) as driver default; the converged labels (metrics + FIELDDIFF), not the iterate history, carry claim (a); PATH equality informational from N7b on | an exact linear solve is not needed for the same converged discrete solution; `--forcing fixed` reproduces the N2-N6 behaviour bitwise ([`raw/n7a_local/fixed_vs_pre.txt`](artifacts/2026-10-06-sf33-gpu-inlet-labels/raw/n7a_local/fixed_vs_pre.txt): 0 differing lines). |
| 2026-10-07 | N7b | pseudo-transient continuation `(J + mu_k D) p = -F`, `D = q_v/h^2` on equation rows, SER `mu_k = mu_0 m_k/m_0`, P-A factored for the shifted operator | N7a: failed solves still stagnate at true rel 0.38-0.71 right after the first full step of a stage ([`raw/n7a_local/digest_ew.txt`](artifacts/2026-10-06-sf33-gpu-inlet-labels/raw/n7a_local/digest_ew.txt)). |
| 2026-10-07 | N7c | discriminating probe of a Galerkin coarse-space correction on the `x1`-constant / `x1`-linear column space, combined multiplicatively with P-A; productize iff it works | N7b: stalls persist once `mu` is small ([`raw/n7b_local/digest_psitc_on.txt`](artifacts/2026-10-06-sf33-gpu-inlet-labels/raw/n7b_local/digest_psitc_on.txt), e.g. 0.525, 0.519, 0.518 at eta 0.1). |
| 2026-10-07 | N6b -> N8' | gate-reduced campaign B: the eps 0.5 / 1.0 ladders at 64/128 NOT run; eps 0.25 (the gate-validated amplitude) substituted as the refinement study, plus the production 32^3 eps 0.5 point (<= 32^3 is allowed) and the oracle32-vs-SF-29 comparison | spec rule quoted above ("Preconditioner not validated against direct solves: the 64^3-128^3 runs do not start"); the N6b gate failed at 32^3 eps 0.5 (`PGB`). |
| 2026-10-07 | C4 / N8'' | Psi-tc initial shift scaled with the grid: `mu_0 (h_ref/h)^2`, `h_ref = 1/16` (bitwise identical at N = 16); eps 0.25 ladder re-run | N8': at 128^3 the bisected stages 0.125 .. 0.015625 each stopped on the Newton stagnation rule after 20 steps with 1-3 GMRES its per step (damped linear contraction while `mu D` dominates the weak family) (`L1`, "The 128^3 failure"). |
| 2026-10-07 | O5 (cancelled) | job `sf33e-n128-sf098` (128^3 eps 0.25, `--gmres-stagnation-factor 0.98 --max-inner 12000`) cancelled before it ran | the failing 128^3 solves have flat per-restart curves (0.348, 0.347, 0.347; 4.85e-3, 4.76e-3, 4.74e-3, `L2` "every failed stage"): ~0.4 % per restart cannot reach eta within a 12 000 cap; the run could not change the outcome. No evidence file exists for it. |

## Build / environment

- Remote: V100 host (two V100, 80 cores; `IDX` "Execution facts"), per-increment mirror `~/MacroFlow3D-SF-33`
  (`scripts/remote --increment SF-33`), state root `~/.macroflow3d-remote/macroflow3d-SF-33`; preset
  `v100-release`, CUDA 11.4.152 / GNU 9.3.1 ([`logs/jobs/sf33-base-build-refs.log`](artifacts/2026-10-06-sf33-gpu-inlet-labels/logs/jobs/sf33-base-build-refs.log)),
  Python 3.11.7 / numpy 1.26.4 / scipy 1.11.4 for the exporter
  ([`scripts/README.md`](artifacts/2026-10-06-sf33-gpu-inlet-labels/scripts/README.md)).
- Local: WSL, RTX 3050 (`OS`, last bullet of the reading), sm_86 / CUDA 13.4
  ([`tests/inlet_slab/slab_newton_tests.cu`](../../tests/inlet_slab/slab_newton_tests.cu), line 41), presets
  `wsl-debug` / `wsl-release`: fast `ctest -R inlet_slab_` contract tests and the N7a/N7b/N7c development runs at
  16^3-24^3 (`logs/n7*_local/`, `raw/n7*_local/`).
- Code: `src/physics/streamfunctions/inlet_slab/` (new), `apps/inlet_slab/` (driver `inlet_slab`),
  `tests/inlet_slab/` (6 `ctest` entries `inlet_slab_*`). Chain `37bfb25` (base) -> N0 `2766beb` -> N1 `0d6755c`
  -> N2 `4366127` -> N3 `2be35a8` -> N4 `f198975` -> C1 `0681c19` -> N5 `8dee959` -> C2 `82a4af9` -> C2b `681b039`
  -> N6 `f5b050f` -> N7a `a198369`, `5edb0fb` -> N7b `30347aa`, `c869e0a` -> N7c `d093c7a`, `ff9864f`, `b92385a`,
  `ea53d09` -> C3 `bfe4efb` -> N6b `3676bb0` -> N8' `94c5d53`, `13c6bf8` -> C4 `964c43a` -> N8'' `e113ff9`.

V100 jobs (all `scripts/remote --increment SF-33 run`, all exit 0; build jobs from the job-log headers, the others
transcribed from `IDX`):

| job | node | kind | GPU | start (UTC) | end (UTC) | exit | outputs |
|---|---|---|---|---|---|---|---|
| `sf33-base-build-refs` | O1 | base build (`37bfb25`) + trio reference runs into `~/sf33_base_refs` | 0 | 2026-10-06T18:59:49Z | 19:00:29Z | 0 | `logs/jobs/sf33-base-build-refs.log` |
| `sf33-n0-build` | N0 | build + `inlet_slab_contracts` + `--with-64` | 0 | 19:18:56Z | 19:19:18Z | 0 | `logs/jobs/sf33-n0-build.log` |
| `sf33-n134-build` | N1/N3/N4 | build + 4 `inlet_slab_` tests | 0 | 19:55:49Z | 19:56:34Z | 0 | `logs/jobs/sf33-n134-build.log` |
| `sf33-n01234-build` | N2 | build + 5 `inlet_slab_` tests | 0 | 20:03:27Z | 20:04:14Z | 0 | `logs/jobs/sf33-n01234-build.log` |
| `sf33-proto-ref24` | N6 | CPU: five 24^3 prototype references | 0 (lock) | 21:00:02Z | 22:01:28Z | 0 | `logs/jobs/candidate_i_24*.log` |
| `sf33-build` | N6 | build + 5/5 tests (`681b039`) | 1 | 21:00:54Z | 21:02:01Z | 0 | `logs/jobs/sf33-build.log` |
| `sf33-export` | N6 | CPU: 19 case exports, 12 solution conversions, 3 cross-check exports | 1 (lock) | 21:02:57Z | 21:25:57Z | 0 | `logs/jobs/sf33-export.log` |
| `sf33-proto` | N6 | 9a matrix + generic3d, P-A defaults | 1 | 21:26:53Z | 21:36:58Z | 0 | `logs/proto/`, `raw/proto/` |
| `sf33-proto-r200` | N6 | same, `--restart 200 --max-inner 12000` | 1 | 21:38:27Z | 22:00:59Z | 0 | `logs/proto_r200/`, `raw/proto_r200/` |
| `sf33-crosscheck` | N6 | SF-19 cross-check 16/24/32 | 0 | 22:01:32Z | 22:01:46Z | 0 | `logs/crosscheck/`, `raw/crosscheck/` |
| `sf33-proto-24ref` | N6 | five 24^3 cases with `--solution` | 0 | 22:03:12Z | 22:06:52Z | 0 | `logs/proto_24ref/`, `raw/proto_24ref/` |
| `sf33-compare` | N6 | CPU: `compare_proto.py` | 0 (lock) | 22:07:50Z | 22:07:52Z | 0 | `raw/proto*/compare_proto*.md` |
| `sf33b-build` | N6b | build + 6/6 tests (`bfe4efb`) | 0 | 2026-10-07T00:34:21Z | 00:35:32Z | 0 | `logs/jobs/sf33b-build.log` |
| `sf33b-export` | N6b | CPU: `gauss:0.5:32`, `gauss_ch:0.5:32` exports | 1 (lock) | 00:34:46Z | 00:40:43Z | 0 | `logs/jobs/sf33b-export.log` |
| `sf33b-proto` | N6b | 21 cases, C3 defaults | 0 | 00:36:36Z | 00:52:18Z | 0 | `logs/proto_b/`, `raw/proto_b/` |
| `sf33b-compare` | N6b | CPU: `compare_proto.py`, `digest_newton.py` | 0 (lock) | 00:53:58Z | 00:53:59Z | 0 | `raw/proto_b/compare_proto_b.md`, `digest_b.txt` |
| `sf33c-build` | N8' | build + 6/6 tests (`94c5d53`) | 0 | 01:08:51Z | 01:10:01Z | 0 | `logs/jobs/sf33c-build.log` |
| `sf33c-ladder-025` | N8' | eps 0.25 ladder 32/64/128 | 0 | 01:12:14Z | 10:05:48Z | 0 | `logs/ladder_0.25/`, `raw/ladder_0.25/` |
| `sf33c-oracle32-vs-sf29` | N8' | oracle on the `gauss` field 16/24/32 | 1 | 01:12:36Z | 01:13:49Z | 0 | `logs/oracle32/`, `raw/oracle32/` |
| `sf33c-prod32-05` | N8' | production 32^3 eps 0.5 | 1 | 01:14:32Z | 01:15:51Z | 0 | `logs/prod32_05/`, `raw/prod32_05/` |
| `sf33d-build` | N8'' | build + 6/6 tests (`964c43a`) | 0 | 10:23:53Z | 10:25:04Z | 0 | `logs/jobs/sf33d-build.log` |
| `sf33d-ladder-025` | N8'' | eps 0.25 ladder 32/64/128, C4 shift | 0 | 10:26:02Z | 13:36:57Z | 0 | `logs/ladder_0.25_c4/`, `raw/ladder_0.25_c4/` |

Driver exit codes inside the jobs are per case (0 converged, 15 `continuation_floor`, ...; table in
[`scripts/README.md`](artifacts/2026-10-06-sf33-gpu-inlet-labels/scripts/README.md)); a job exit 0 means the job
script completed. No job waited for a GPU lock (`IDX`).

## Config(s)

Fields.

- Prototype exports (step 9a; `cases.load_case` of the SF-29 artifact, read-only): `gauss` (band-limited Gaussian
  covariance `|m_i| <= 3`, `ell = 1/4`, unit variance, seed 7; triply periodic), `gauss_ch` (the same with
  constant-head faces), `control2d` (planar control), `generic3d` (analytic asymmetric field); `k = exp(eps f)`,
  analytic `ln k` and `grad ln k` at the vertices, inlet labels `psi0`, `v_perp,in`, DOP853 oracle labels, plus
  stage inputs for every reachable amplitude. Matrix (`PC`): `gauss`, `gauss_ch`, `control2d` x eps {0.25, 0.5} x
  N {16, 24} plus N 32 for `gauss:0.25`, `gauss_ch:0.25` (14 cases, the spec's 9a); added by the orchestrator:
  `generic3d` eps 1 at 16/20/24 and eps 0.25/0.5 at 16 (informative, outside 9a); added in N6b:
  `gauss:0.5:32`, `gauss_ch:0.5:32` (gate point; no prototype reference, SF-29 sweep2 timed out there).
- Production field (step 9b): SF-18 `generate_periodic_gaussian_field`, `sigma2 = 1`, `corr_length ell = 1/4`,
  seed 3001, `normalize_variance = true`; `applied_scale` 1.602888711251 and `raw_variance` 0.3892183071637 at 32,
  64 and 128 (spread 0.0; `L1`, `L2`); stage field `k = exp(eps Y)`; `ell/h` = 8 / 16 / 32 at N = 32 / 64 / 128.
  SF-19 Darcy per stage (`qbar = e1`), inlet `min v1` 0.5095 / 0.5078 / 0.5071 at eps 0.25 (`L1`), 0.2445 at
  32^3 eps 0.5 (`P05`): no backflow anywhere.
- Step 8 / oracle comparison: SF-19 on the `gauss` field at eps 0.25, `Y_cells` = `ln k` at cell centres, 16/24/32
  (`XC`, `OS`).

Solver settings per phase (driver `inlet_slab`; each log's `SOLVER` line records them):

| phase | runs | linear solver | forcing | shift | other |
|---|---|---|---|---|---|
| P-A only (N2-N6) | `sf33-proto`, `sf33-proto-r200`, `sf33-proto-24ref` | GMRES(100) + P-A (per-transverse-mode banded solve of the plane-averaged frozen linearization), cap 6000 (r200: restart 200, cap 12 000) | fixed, `lin_tol` 1e-12 | none | `newton_tol` 1e-13, max-newton 40, bisect 4 |
| + EW (N7a) | `logs/n7a_local/ew` (local) | as above | Eisenstat-Walker choice 2, eta0 = eta_max 0.1, eta_min 1e-12 | none | `--forcing fixed` = N6 bitwise |
| + Psi-tc (N7b) | `logs/n7b_local/` (local) | as above | EW | `mu_k D`, `D = q_v/h^2`, SER, `mu_0 = 1`, `mu_max = 100` | max-newton 120 |
| + coarse correction (N7c) | `logs/n7c_local/` (local) | GMRES(100) + P-A + Galerkin coarse correction (`mult`, 1 or 2 `x1` profiles, K = 2 P N^2) | EW | Psi-tc | dense or banded host LU |
| C3 defaults (N6b, N8') | `sf33b-proto`, `sf33c-*` | `--coarse mult --coarse-profiles 2 --coarse-assembly colored --coarse-factor banded`, restart 100, cap 6000, `--gmres-stagnation-factor 0.9` | EW | Psi-tc, `mu_0 = 1` | max-newton 120, bisect 4, `lin_tol` 1e-12, `newton_tol` 1e-13 |
| C4 scaling (N8'') | `sf33d-ladder-025` | as C3 | EW | `mu_0 (h_ref/h)^2`, `h_ref = 1/16` (effective 1/4, 1/16, 1/64 at 32/64/128) | as C3 |

Oracle settings (production): backward tracing of the SF-19 potential on the SF-28 spline with the SF-30 DP5(4)
integrator (thin `trace_to_plane` wrapper), `h_max = h/8`, tol 1e-8, gate 1e-8 per plane; `--oracle-ladder` adds
`(h/16, 1e-8)` and `(h/16, 1e-10)`; 32 host threads (`L1` "Command").

## Commands

From the repository root on the mirror (exact strings in the job logs and in `IDX`):

```bash
# exports (N4 exporter, CPU; host-only outputs under <A>/exports, gitignored)
cd docs/experiments/artifacts/2026-10-06-sf33-gpu-inlet-labels/scripts
python3 export_proto.py gauss:0.25:16 [...] --out ../exports
python3 export_proto.py --solutions <SF-29>/raw/sweep2/solutions/<case>_16_i1o4.npz --out ../exports/solutions
python3 export_proto.py --crosscheck gauss:0.25:16 gauss:0.25:24 gauss:0.25:32 --out ../exports
# 24^3 full-precision prototype references (job sf33-proto-ref24, SF-29 scripts)
python3 candidate_i.py <field>:<eps>:24:i1o4 --direct-max 32 --reuse-lu 0 --save ~/sf33_ref24
# step 9a (driver, prototype mode)
inlet_slab --proto <exports>/<case> [--solution <exports>/solutions/<case>_i1o4] --summary raw/proto_b/<case>.json
# step 8
bash scripts/run_crosscheck.sh build/v100-release
# step 9b and oracle comparison (campaign B)
inlet_slab --production --n N --eps 0.25 --sigma2 1 --ell 0.25 --seed 3001 --oracle-ladder --threads 32 \
           --summary raw/ladder_0.25/N<N>.json
inlet_slab --production --n 32 --eps 0.5 --sigma2 1 --ell 0.25 --seed 3001 --oracle-ladder --threads 32
inlet_slab --production --n N --eps 1 --cells exports/crosscheck_gauss_0.25_N/Y_cells.npy --oracle-ladder \
           --threads 32 --save-oracle exports/oracle_gpu/N<N>
# analysis (no computation; as called by the jobs / run_campaign_b.sh)
python3 compare_proto.py ../logs/proto_b/*.log --exports ../exports --out ../raw/proto_b/compare_proto_b.md
python3 digest_newton.py ../raw/proto_b/*.json > ../raw/proto_b/digest_b.txt
python3 ladder_orders.py <logs>/N32.log <logs>/N64.log <logs>/N128.log --out <raw>/ladder_orders.md
python3 compare_oracle_sf29.py --case 16 ../exports/oracle_gpu/N16 ../exports/gauss_0.25_16 \
        --case 24 ... --case 32 ... --suffix '<ladder suffix>' --out ../raw/oracle32/oracle_vs_sf29<suffix>.md
python3 digest_coarse.py <logs> --out ../raw/digest_coarse.md
```

Jobs (detached):

```bash
scripts/remote --increment SF-33 run sf33-proto -- "bash docs/experiments/artifacts/2026-10-06-sf33-gpu-inlet-labels/scripts/run_proto.sh build/v100-release"
scripts/remote --increment SF-33 run sf33c-ladder-025 -- "bash docs/experiments/artifacts/2026-10-06-sf33-gpu-inlet-labels/scripts/run_campaign_b.sh build/v100-release ladder"
scripts/remote --increment SF-33 run sf33c-prod32-05 -- "bash .../scripts/run_campaign_b.sh build/v100-release prod32-05"
scripts/remote --increment SF-33 run sf33c-oracle32-vs-sf29 -- "bash .../scripts/run_campaign_b.sh build/v100-release oracle32"
scripts/remote --increment SF-33 run sf33d-ladder-025 -- "env LADDER_DIR=ladder_0.25_c4 bash .../scripts/run_campaign_b.sh build/v100-release ladder"
```

The N6b matrix and the N6 additional runs were inline driver loops recorded in their job logs
(`logs/jobs/sf33b-proto.log`, `sf33-proto-r200.log`, `sf33-proto-24ref.log`). The local N7a/N7b/N7c runs used
`run_proto.sh <build> <out> -- <extra args>` and `run_linear_probe.sh` (scripts README).

## Outputs inspected

### O-1. Claim (a): prototype reproduction, final table (N6b, C3 solver; `PCB`)

Cell = GPU / prototype (relative difference). `PASS` = full-precision reference (16^3 saved solutions, 24^3
regenerated references) within 1e-6; `PASS(4dig)` = prototype value printed with 4 significant digits, gated at
its rounding bound (1e-6 not resolvable, not claimed). `e_psi2` of `control2d` is roundoff on both sides
(~5e-12 / ~2e-13), not gated. FIELDDIFF = joint max relative field difference vs the saved solution.

| case | status | r_F | PATH GPU / prototype | e_v | e_psi | e_psi1 | e_psi2 | FIELDDIFF | verdict |
|---|---|---|---|---|---|---|---|---|---|
| control2d:0.25:16 | converged | 5.03e-14 | `0.25` / same | 3.0673e-3 / 3.0673e-3 (4.8e-12) | 3.2204e-2 / 3.2204e-2 (3.8e-11) | 3.2204e-2 / same (3.8e-11) | roundoff | 8.58e-13 | PASS |
| control2d:0.25:24 | converged | 2.86e-15 | `0.25` / same | 5.2468e-4 / 5.247e-4 (4.3e-5) | 2.6669e-3 / 2.667e-3 (3.9e-5) | 2.6669e-3 / 2.667e-3 (3.9e-5) | roundoff | - | PASS(4dig) |
| control2d:0.5:16 | converged | 2.65e-15 | `0.25->0.5` / same | 7.6868e-3 / same (3.8e-15) | 3.1833e-2 / same (2.8e-14) | same (2.8e-14) | roundoff | 8.90e-15 | PASS |
| control2d:0.5:24 | converged | 5.40e-15 | `0.25->0.5` / same | 1.6180e-3 / same (1.4e-14) | 1.0650e-2 / same (3.3e-14) | same (3.3e-14) | roundoff | 7.91e-15 | PASS |
| gauss:0.25:16 | converged | 9.64e-14 | `0.25` / same | 5.5059e-3 / same (7.4e-13) | 7.7266e-2 / same (1.8e-12) | same (1.8e-12) | 5.8595e-2 / same (6.7e-14) | 7.05e-13 | PASS |
| gauss:0.25:24 | converged | 6.79e-15 | `0.25` / `0.25(fail)->0.125->0.25` | 1.8081e-3 / same (6.7e-14) | 2.8861e-2 / same (8.0e-14) | same (8.0e-14) | 2.1060e-2 / same (4.3e-14) | 2.14e-14 | PASS |
| gauss:0.25:32 | converged | 9.85e-15 | `0.25` / `0.25(fail)->0.125->0.25` | 8.0143e-4 / 8.014e-4 (3.5e-5) | 1.3027e-2 / 1.303e-2 (2.2e-4) | same as e_psi | 9.9517e-3 / 9.952e-3 (3.4e-5) | - | PASS(4dig) |
| gauss:0.5:16 | converged | 6.31e-15 | `0.25->0.5` / `0.25->0.5(fail)->0.375->0.5` | 1.9084e-2 / same (1.8e-16) | 1.4026e-1 / same (1.8e-15) | same (1.8e-15) | 9.9536e-2 / same (1.3e-14) | 6.14e-15 | PASS |
| gauss:0.5:24 | converged | 4.74e-14 | `0.25->0.5` / `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5` | 8.4740e-3 / same (9.9e-14) | 6.8315e-2 / same (2.7e-14) | same (2.7e-14) | 4.4045e-2 / same (9.3e-14) | 7.58e-14 | PASS |
| gauss_ch:0.25:16 | converged | 4.20e-15 | `0.25` / `0.25(fail)->0.125->0.25` | 5.2874e-3 / same (6.2e-15) | 6.7804e-2 / same (1.1e-14) | same (1.1e-14) | 4.1388e-2 / same (1.1e-14) | 4.57e-15 | PASS |
| gauss_ch:0.25:24 | converged | 7.32e-15 | `0.25` / `0.25(fail)->0.125->0.25` | 1.7830e-3 / same (1.2e-14) | 2.4827e-2 / same (5.7e-15) | same (5.7e-15) | 1.5027e-2 / same (3.1e-14) | 2.09e-14 | PASS |
| gauss_ch:0.25:32 | converged | 1.22e-14 | `0.25` / `0.25(fail)->0.125->0.25` | 7.6572e-4 / 7.657e-4 (2.5e-5) | 1.0861e-2 / 1.086e-2 (5.2e-5) | same as e_psi | 6.7772e-3 / 6.777e-3 (3.5e-5) | - | PASS(4dig) |
| gauss_ch:0.5:16 | converged | 7.62e-15 | `0.25->0.5` / `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5` | 1.7846e-2 / same (3.1e-15) | 1.2316e-1 / same (9.0e-15) | same (9.0e-15) | 6.0806e-2 / same (2.0e-14) | 1.27e-14 | PASS |
| gauss_ch:0.5:24 | converged | 4.00e-14 | `0.25->0.5` / `0.25(fail)->0.125->0.25->0.5(fail)->0.375->0.5` | 8.4193e-3 / same (7.2e-14) | 6.2214e-2 / same (8.6e-15) | same (8.6e-15) | 3.2592e-2 / same (6.1e-14) | 9.32e-14 | PASS |
| gauss:0.5:32 (gate point) | continuation_floor | 1.41e-2 | `0.25->0.5(fail)->0.375->0.5(fail)->0.4375(fail)->0.40625->0.4375(fail)->0.421875->0.4375(fail)->0.5(final)` / none | 1.387e-2 (unconverged) / - | 1.151e-1 / - | 1.097e-1 / - | 1.151e-1 / - | - | FAIL (no reference) |
| gauss_ch:0.5:32 (gate point) | continuation_floor | 9.66e-3 | `0.25->0.5(fail)->0.375->0.5(fail)->0.4375->0.5(fail)->0.46875(fail)->0.453125(fail)->0.5(final)` / none | 1.074e-2 (unconverged) / - | 7.023e-2 / - | 7.023e-2 / - | 6.796e-2 / - | - | FAIL (no reference) |

Over the 11 full-precision rows the largest relative difference of any gated key is 3.84e-11
(`control2d:0.25:16`, `e_psi`; `PCB` per-case table). The reading paragraphs of `PCB` and the criterion table of
`PGB` state "1.75e-12"; that is the `gauss:0.25:16` value and is contradicted by the `control2d:0.25:16` row of
the same table (transcription slip in those files; both values are far inside 1e-6). The 4-digit rows agree
within their rounding bound (largest 2.2e-4 for `gauss:0.25:32` `e_psi`, printed `1.303e-02`). The oracle-ceiling
metrics of the two 32^3 eps 0.5 exports agree with the exporter's `ref_metrics.json` to 4.7e-15 / 2.8e-15
(`PCB`), so their inputs are sane and the failure is the solver's.

The same 14 cases with the P-A-only solver (N6, `PC`): 12 PASS with metrics rel diff <= 3.2e-14 and FIELDDIFF
<= 9.7e-15, PATHs equal and Newton iteration counts equal to the prototype's; `gauss:0.5:24` and `gauss_ch:0.5:24`
FAIL (`continuation_floor`, `r_F` 5.2e-2 / 1.0e-1), also with restart 200 / cap 12 000. r_F history agreement
5.6-7.1 digits (control2d 15.7-17.0) (`PC`).

`generic3d` (informative): eps 0.25 / 0.5 at 16^3 PASS at full precision (max rel diff 9.0e-13 / 4.4e-13); eps 1
at 16/20/24 `continuation_floor` in N6, N7a, N7b, N7c and N6b (last accepted 0.875 / 0.75 / 0.65625 in N6b,
`PCB`), where the prototype converged with direct solves.

### O-2. The preconditioner gate story (spec item 5)

1. **N6, P-A only — FAIL** (`PG`). Linear failures in 11/19 cases (36 failures) with restart 100 / cap 6000,
   7/19 (28) with restart 200 / cap 12 000. The decisive ones are stages the prototype accepts with direct
   solves: `gauss:0.5:24` 0.375 -> 0.5 (cap 6000 at rel 4.0e-5; r200 cap 12 000 at 6.3e-6), `gauss_ch:0.5:24`
   (12 000 its at 1.2e-5), `generic3d` eps 1 0.75 stages. Accepted-step GMRES its max / median: 5600 / 166
   (16^3), 5900 / 300 (24^3), 1900 / 357 (32^3, eps 0.25 only). Every accepted step reached `lin_tol` (max
   9.9e-13).
2. **N7a, + Eisenstat-Walker (local)** ([`raw/n7a_local/digest_ew.txt`](artifacts/2026-10-06-sf33-gpu-inlet-labels/raw/n7a_local/digest_ew.txt),
   [`compare_all_ew.md`](artifacts/2026-10-06-sf33-gpu-inlet-labels/raw/n7a_local/compare_all_ew.md)): controls
   converge with the same metrics (FIELDDIFF <= 6.6e-13); the five hard cases still end `continuation_floor`. The
   failed solves stall at the start of a new stage, e.g. `gauss_ch_0.5_24` stage 0.5 from 0.375, step 2,
   per-restart true rel 0.545, 0.466, 0.431, 0.406, 0.394 at eta 2.2e-3; `gauss_0.5_24` 0.658, 0.646, 0.644.
3. **N7b, + Psi-tc (local)** ([`raw/n7b_local/digest_psitc_on.txt`](artifacts/2026-10-06-sf33-gpu-inlet-labels/raw/n7b_local/digest_psitc_on.txt),
   [`compare_psitc_on.md`](artifacts/2026-10-06-sf33-gpu-inlet-labels/raw/n7b_local/compare_psitc_on.md)): controls
   converge (`gauss_0.25_16` 12 Newton steps / 233 GMRES its; `gauss_ch_0.25_16` 257; `control2d_0.5_16` 97); the
   hard cases still `continuation_floor`, stalls once `mu` is small (0.525, 0.519, 0.518 at eta 0.1, mu 7.3e-3).
4. **N7c, + Galerkin coarse correction (local probe and continuation)**
   ([`raw/n7c_local/digest_probe_batch1.txt`](artifacts/2026-10-06-sf33-gpu-inlet-labels/raw/n7c_local/digest_probe_batch1.txt),
   [`digest_probe_batch2.txt`](artifacts/2026-10-06-sf33-gpu-inlet-labels/raw/n7c_local/digest_probe_batch2.txt)):
   on the frozen Jacobians of the N7b stall iterates (GMRES(100) to 1e-8, cap 6000):

   | iterate | mu | P-A | P-A + CC(mult, 2 profiles) |
   |---|---|---|---|
   | `gauss_ch_0.5_24` stage 0.5 from 0.375, k = 8 | SER 1.02e-2 | max_iterations, rel 7.45e-2 | 287 its |
   | same | 0 | stagnation 8.15e-1 | 492 its |
   | `gauss_0.5_24` stage 0.5 from 0.4375, k = 8 | SER 7.26e-3 | max_iterations, rel 3.28e-2 | 368 its |
   | same | 0 | max_iterations, 9.09e-1 | 496 its |
   | `generic3d_1_16` stage 0.75 from 0.625, k = 7 | SER 1.70e-2 | max_iterations, 1.14e-2 | 250 its |
   | same | 0 | stagnation 7.32e-1 | 385 its |
   | `generic3d_1_16` stage 1 from 0.875, k = 6 | SER 5.51e-2 | max_iterations, 4.14e-1 | max_iterations, **1.15e-1** |

   With the coarse correction the 24^3 eps 0.5 continuations converge (`gauss:0.5:24` / `gauss_ch:0.5:24`, `r_F`
   4.73e-14 / 4.01e-14, FIELDDIFF 7.21e-14 / 9.60e-14 vs the full-precision references;
   [`compare_coarse_mult2_colored_banded.md`](artifacts/2026-10-06-sf33-gpu-inlet-labels/raw/n7c_local/compare_coarse_mult2_colored_banded.md));
   `generic3d` eps 1 does not (last row: a genuine plateau, slow modes outside the `x1`-constant / linear column
   space; [`compare_coarse_mult2.md`](artifacts/2026-10-06-sf33-gpu-inlet-labels/raw/n7c_local/compare_coarse_mult2.md)).
5. **N6b, C3 defaults on V100 — 9a PASS, gate FAIL at 32^3 eps 0.5** (`PGB`). The 14 9a cases: zero linear
   failures, zero bisections. The two 32^3 eps 0.5 points: `continuation_floor` after four
   `linear_failure` bisections (6 and 7 linear failures; last accepted 0.421875 / 0.4375). Accepted-step GMRES its
   max / median grow with N at eps 0.5: **74 / 10** (`gauss_0.5_16`) -> **400 / 16** and **500 / 17**
   (`gauss_0.5_24`, `gauss_ch_0.5_24`) -> **2800 / 51.5** and **5800 / 40** (`gauss_0.5_32`, `gauss_ch_0.5_32`).
   Failed-solve curves at 32^3 (restart 100, eta <= 0.1): e.g. `gauss_0.5_32` stage 0.5 from 0.25 step 12:
   0.603, 0.557, 0.552 (mu 3.2e-3); stage 0.4375 from 0.40625 step 15: 0.832, 0.810, 0.808 (mu 2.0e-4);
   `gauss_ch_0.5_32` stage 0.4375 from 0.375 step 19: monotone over 60 restarts to 1.59e-4 at the 6000 cap
   against eta 6.47e-5. Coarse `rcond_est` minimum 3.4e-9 in the failing `gauss_0.5_32` stage (1.6e-5 for the
   converged 32^3 eps 0.25 stage). Not tuned (rule printed in `PGB`). `generic3d` eps 1: still
   `continuation_floor` (informative).

Verdict recorded in `PGB`: FAIL for the 64^3-128^3 ladders at eps 0.5 / 1.0.

### O-3. SF-19 cross-check (step 8; `XC`)

SF-19 inlet U-face `v1` and the spline-flow vertex `v_perp` vs the prototype's spectral reference on `gauss` eps
0.25:

| N | v1 rms_rel (point) | v1 max_rel | v1 rms_rel (face average) | vperp rms | vperp max |
|---|---|---|---|---|---|
| 16 | 3.993e-3 | 9.292e-3 | 3.557e-3 | 2.678e-3 | 5.806e-3 |
| 24 | 1.778e-3 | 4.142e-3 | 1.592e-3 | 1.169e-3 | 2.571e-3 |
| 32 | 1.001e-3 | 2.387e-3 | 8.977e-4 | 6.538e-4 | 1.436e-3 |

Observed orders 16 -> 24 / 24 -> 32: 1.99 / 2.00, 1.99 / 1.92, 1.98 / 1.99, 2.04 / 2.02, 2.01 / 2.02 (range
1.92-2.04). Inlet `min v1` 0.650 / 0.639 / 0.635.

### O-4. Production ladder at eps 0.25 (claim (b) at the gate-validated amplitude; `L1` = N8', `L2` = N8'')

Per grid (C4 run `L2`; the N8' run `L1` reached the same converged solutions at 32 and 64):

| N | status | r_F | PATH | Newton steps | GMRES total / max / median | linear failures | wall |
|---|---|---|---|---|---|---|---|
| 32 | converged | 1.097e-14 | `0.25` | 11 | 218 / 58 / 13 | 0 | 17 s |
| 64 | converged | 5.102e-14 | `0.25(fail)->0.125->0.25(fail)->0.1875->0.25` | 34 | 3914 / 1100 / 36 | 2 | 11 min 30 s |
| 128 | **continuation_floor** | 9.158e-1 (final attempt) | `0.25(fail)->0.125->0.25(fail)->0.1875(fail)->0.15625(fail)->0.140625(fail)->0.25(final)` | 25 | 4449 / 900 / 100 | 6 | 2 h 59 min |

N8' (`L1`, unscaled shift): 32^3 converged (18 steps, 22 s), 64^3 converged with PATH `0.25(fail)->0.125->0.25`
(60 steps, 944 s), 128^3 `continuation_floor` (`r_F` 0.156, 106 steps, 8 h 37 min).

Gate 3A metrics (`cand=i1o4`) and the ceiling (`cand=oracle_fd4`, the oracle labels' own 4th-order FD metrics,
reading aid only). The 128^3 `i1o4` values belong to the non-converged final iterate and are shown in parentheses
only for completeness:

| N | e_v | e_psi | e_psi1 | e_psi2 | e_i1 | e_i2 | e_div | min_c | ceiling e_v | ceiling e_i1 / e_i2 | ceiling e_div |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 1.711e-3 | 1.250e-2 | 1.250e-2 | 1.140e-2 | 8.425e-4 | 5.905e-4 | 7.440e-4 | 0.4950 | 2.247e-3 | 8.389e-5 / 7.095e-5 | 8.281e-4 |
| 64 | 3.904e-4 | 1.555e-3 | 1.483e-3 | 1.555e-3 | 1.449e-4 | 1.155e-4 | 5.528e-5 | 0.4946 | 5.684e-4 | 9.110e-6 / 7.396e-6 | 5.742e-5 |
| 128 | (8.219e-2) | (4.500e-1) | (4.429e-1) | (4.500e-1) | (4.818e-2) | (3.457e-2) | (5.993e-4) | (0.5810) | 1.427e-4 | 1.368e-6 / 1.198e-6 | 3.750e-6 |

Observed orders 32 -> 64: `e_v` **2.13**, `e_psi` **3.01** (`e_psi1` 3.08, `e_psi2` 2.87), `e_i1` / `e_i2`
2.54 / 2.35, `e_div` 3.75; ceiling `e_v` 1.98 (32 -> 64) and 1.99 (64 -> 128). No 64 -> 128 pair exists (no
converged 128^3 solution; the script's negative values are meaningless). The reading in `L1` and `L2`: the
acceptance-(b) threshold applied to eps 0.25 is FAIL (three converged grids required).

128^3 failure characterization (`L2`, "every failed stage"): C4 removed the N8' Newton-stagnation mode (the 0.125
stage now converges: 9 steps, `r_F` 7.94e-14, GMRES max 900), but every stage >= 0.140625 ends in
`linear_failure` = GMRES restart stagnation at step 2-3:

| stage (from) | eta | mu | per-restart true rel | rel / eta | coarse rcond_est |
|---|---|---|---|---|---|
| 0.25 (0) | 6.24e-2 | 1.85e-4 | 0.348, 0.347, 0.347 | 5.6 | 6.8e-9 |
| 0.25 (0.125) | 1.80e-3 | 7.00e-4 | 4.85e-3, 4.76e-3, 4.74e-3 | 2.6 | 2.4e-6 |
| 0.1875 (0.125) | 5.89e-3 | 5.68e-5 | 0.109, 0.108, 0.107 | 18 | 3.5e-9 |
| 0.15625 (0.125) | 5.11e-3 | 5.31e-5 | 4.17e-2, 3.93e-2, 3.89e-2 | 7.6 | 1.1e-6 |
| 0.140625 (0.125) | 5.02e-3 | 5.29e-5 | 1.40e-2 .. 8.37e-3 (6 restarts) | 1.7 | 1.2e-6 |

The curves are flat (a fraction of a percent per restart), so a relaxed stagnation rule could not reach eta within
any affordable cap (O5 cancelled, see Hypothesis). Coarse `rcond_est` spans 3.5e-9 .. 2.4e-6 over the failing
solves: near-singularity of the coarse operator is not a uniform explanation (`L2`, observation (i)).

Cost at 128^3 (`L2`; claim (c)): total 10 743 s (wall 2 h 59 min 07 s); solve 9846 s, of which coarse banded host
LU 7818 s (**79 %**; 312.7 s mean per build, K = 65 536, kl = ku = 2559), GMRES 1905 s, oracle (3 runs) 894 s;
peak device **6.506 GB** (workspaces 5.960 GB); host max RSS **4.79 GiB** (`L2` cost table). 64^3: 686 s total, peak device 1.324 GB.
N8' at 128^3: 31 042 s total, coarse LU 29 116 s (97 % of the solve), peak device 6.506 GB, RSS 4.79 GiB (`L1`).

### O-5. Production 32^3 at eps 0.5 (`P05`)

Converged, PATH `0.25->0.5`, no linear failure. Stage 0.5: 19 Newton steps, GMRES 6031 total / 2000 max / 100
median (the last four steps need 400 / 600 / 1400 / 2000 its as eta tightens to 5.6e-5); `r_F` 4.714e-14, `r_out`
1.29e-15. CASE: `e_v` 5.315e-3, `e_psi` 3.837e-2 (`e_psi1` 3.837e-2, `e_psi2` 2.756e-2), `e_i` (3.941e-3,
2.242e-3), `e_div` 4.860e-3, `min_c` 0.2291; ceiling `e_v` 5.984e-3. Oracle round trip 2.740e-9 (h/8). Solve
71.9 s, peak device 621.8 MB. One field, one grid: it does not validate the preconditioner at eps 0.5 (the N6b
failure on the analytic fields stands).

### O-6. Production oracle (claim (d))

Round trips (max over planes and vertices; `L1`, `L2`, `P05`, `OS`):

| field / N | (h/8, 1e-8) | (h/16, 1e-8) | (h/16, 1e-10) | non-ok streamlines |
|---|---|---|---|---|
| production eps 0.25, 32 | 1.227e-9 | 1.053e-10 | 1.053e-10 | 0 |
| production eps 0.25, 64 | 1.071e-10 | 1.065e-11 | 1.065e-11 | 0 |
| production eps 0.25, 128 | 1.127e-11 | 9.331e-13 | 9.331e-13 | 0 |
| production eps 0.5, 32 | 2.740e-9 | 4.168e-10 | 4.168e-10 | 0 |
| `gauss` eps 0.25, 16 | 9.698e-9 | 8.218e-10 | 8.218e-10 | - |
| `gauss` eps 0.25, 24 | 2.576e-9 | 2.023e-10 | 2.023e-10 | - |
| `gauss` eps 0.25, 32 | 9.557e-10 | 7.723e-11 | 7.723e-11 | - |

Every plane <= 1e-8 at the production `h_max = h/8` on every grid; the production-field round trip falls by 11.5x
and 9.5x per refinement (`h_max^3` scaling predicts 8x). The ladder runs change the labels by <= 7.1e-10 (production 32^3) and <= 5.4e-9 (`gauss`
16^3). SF-18 `applied_scale` identical across 32/64/128.

Oracle vs the SF-29 spectral oracle on the step-8 `gauss` field (`OS`, RMS of the label difference relative to the
fluctuating part of the SF-29 labels):

| N | psi1 rms | psi1 max | psi2 rms | psi2 max |
|---|---|---|---|---|
| 16 | 2.326e-2 | 7.358e-2 | 2.533e-2 | 6.438e-2 |
| 24 | 1.035e-2 | 3.363e-2 | 1.132e-2 | 3.060e-2 |
| 32 | **5.829e-3** | 1.900e-2 | **6.380e-3** | 1.702e-2 |

Observed RMS orders 2.00 / 2.00 (psi1) and 1.99 / 1.99 (psi2) on 16 -> 24 / 24 -> 32; the inlet plane alone
carries a difference of the same size and order (rms_inlet 5.37e-3 / 4.77e-3 at 32^3, orders 1.99-2.00), i.e. the
difference is the 2nd-order SF-19 inlet input plus the same-order difference of the traced flow.

## Result

Classification per `docs/AGENTS.md` ("confirmed in runs" = established by the runs cited, within the caveats).

**(a) Same discrete problem — CONFIRMED IN RUNS on the spec's full 9a matrix.** All 14 cases converge with the
final solver (`r_F` <= 9.64e-14); `e_v`, `e_psi`, `e_psi1`, `e_psi2` agree with the full-precision prototype
references at 16^3 and 24^3 within 3.84e-11 (11 cases) and with the 4-digit values at 32^3 and
`control2d:0.25:24` within their rounding bound (3 cases); the saved 16^3 solutions (and the 24^3 references) are
reproduced field-wise with FIELDDIFF <= 8.6e-13 (C3 solver) and <= 9.7e-15 (exact-solve P-A runs of N6). PATHs
differ from the prototype's only by stage success (the GPU needs fewer bisections); PATH equality is not part of
the criterion. Source: O-1.

**(b) Production refinement — NOT ESTABLISHED at eps 0.5 / 1.0; at eps 0.25 confirmed on 32 -> 64 only.**
- eps 0.5 and 1.0 at 64^3 / 128^3: **not run** (gate rule; O-2, N6b verdict). The open item of the decision record
  (`sigma_Y = 1` at `ell/h = 16, 32`) remains open and unmeasured. At 32^3, eps 0.5 converges on the production
  field (O-5) but fails on the two analytic 9a fields (O-2 step 5).
- eps 0.25 (substituted, recorded deviation): `e_v` order 2.13 and `e_psi` order 3.01 on 32 -> 64 with `r_F` <=
  5.1e-14 on both grids (O-4); the 64^3 run needed bisections (no floor). At 128^3 the solver reaches no solution:
  a genuine linear-solver plateau of P-A + coarse-corrected GMRES for every stage with eps >= 0.140625 (flat
  restart curves, O-4). The threshold "orders >= 1.8 on both pairs of 32/64/128" is therefore not met at any
  amplitude. This is a **linear-solver limit**, not a floor of the formulation: the 128^3 oracle labels exist
  (round trip 1.1e-11) and their FD ceiling converges at 1.99 (O-4).

**(c) Gate 3A metrics, V100 wall time and peak device memory at 128^3 — recorded, for a non-converged run.**
128^3 eps 0.25: wall 2 h 59 min (C4) / 8 h 37 min (N8'), peak device 6.506 GB, host RSS 4.79 GiB, coarse LU 79 %
of the solve; the Gate 3A metrics printed at 128^3 are those of the non-converged final iterate (`r_F` 0.916) and
of the oracle ceiling. Converged Gate 3A metrics exist at 32^3 and 64^3 (O-4) and at 32^3 eps 0.5 (O-5).

**(d) Production oracle — CONFIRMED IN RUNS at every grid run.** Round trip <= 1e-8 on every plane at `h_max =
h/8` on all seven field/grid combinations (max 9.70e-9, `gauss` 16^3; production fields <= 2.74e-9); non-ok
streamlines 0; the 32^3 oracle agrees with the SF-29 spectral oracle to 0.58 % / 0.64 % (RMS) at observed order
2.00, the order of the SF-19 discretization, matching the step-8 cross-check orders 1.92-2.04 (O-3, O-6).

**Diagnosis of the solver limit (orchestrator analysis; consistent with the evidence, not a theorem proved
here).** The linearized operator's principal symbol restricted to the `x1`-independent block
(`(d22) du1 + d23 du2`, `(d33) du2 + d23 du1`) has identically zero determinant, so that family (shear /
potential) is controlled only by lower-order terms and the boundary rows; its singular values scale like `h^2`
(SF-29 R2: smallest relative singular value 4.36e-4 -> 2.54e-4 from 12^3 to 16^3). P-A inverts that family
exactly only for transverse-constant (plane-averaged) coefficients; with transverse coefficient variation of
order eps, `P^-1`'s ~`h^-2` response on the weak family amplifies the mismatch, so the preconditioned spectrum
degrades with N and eps (O-2 steps 1-3). The Galerkin coarse correction on the `x1`-constant / `x1`-linear column
space represents the weak family with the true local coefficients and fixes it at 24^3 (O-2 step 4), but not at
32^3 eps 0.5 nor at 128^3 eps >= 0.14 (O-2 step 5, O-4): slow modes outside that column space remain (already
visible on `generic3d` eps 1 at 16^3, plateau 0.115).

**Implementation findings (recorded for future GPU work).**
- Stream-ordering race: pageable `cudaMemcpy` on the legacy default stream followed by kernels on the
  non-blocking `CudaContext` stream may read stale data; `inlet_slab_jvp` failed 11/15 runs under a concurrent
  GPU process before the fix, 0/15 after ([`tests/inlet_slab/slab_jvp_tests.cu`](../../tests/inlet_slab/slab_jvp_tests.cu),
  line 72; commit `0681c19` message). The same pattern in the library preparation paths was fixed by
  synchronization only (C2 `82a4af9`, C2b `681b039`); driver outputs bitwise unchanged.
- Oracle round trip is step-limited, ~`h_max^3`, on the gradient of a C^2 spline (the DP5(4) error estimate does
  not see the knots): numbers in the Hypothesis table (N3 row) and in O-6.

**`generic3d` eps 1 limitation.** Outside the 9a matrix (orchestrator's addition): `continuation_floor` at
16/20/24 with every solver variant (O-1), where the SF-29 prototype converged with direct solves. A limit of the
iterative solver, not of the formulation.

**Options for the owner (stated neutrally; no recommendation implied by the order).**
1. A dedicated increment for a scalable preconditioner of the slab Jacobian: a GPU sparse coarse solve (removing
   the host banded LU, 79-97 % of the 128^3 solve), richer or adapted `x1` profiles (or the P-A near-null profiles)
   for the coarse space, a different splitting, or a multigrid that treats the weak `x1`-independent family; then
   re-run 9b as specified.
2. Accept the increment as partial (claim (a) and (d) established; (b) at eps 0.25 on 32 -> 64) and re-scope
   SF-34 to <= 64^3 at eps 0.25.
3. Re-examine the decision record's resolution claim: item 6 expects `sigma_Y = 1` to need `ell/h >= 24-32`; with
   this solver that resolution is unreachable on the production field at `ell = 1/4` (no converged 128^3 solution
   even at eps 0.25).

No statement about `alpha_T` or transport is made or implied (theory note §4).

## Caveats

- One production realization (SF-18 seed 3001), one correlation length `ell = 1/4` (`L/ell = 4`); the analytic
  9a fields are single fixed fields.
- The inlet `v1` comes from the 2nd-order SF-19 flux; the production labels and oracle are limited to 2nd order by
  it (measured 1.92-2.04, O-3; oracle difference order 2.00, O-6). The eps 0.25 `e_v` order 2.13 is consistent
  with that limit.
- The coarse correction's banded LU runs on the host; its cost (312.7 s mean per build at 128^3) dominates the
  128^3 runs and was measured, not optimized.
- Not run: eps 0.5 and 1.0 at 64^3 / 128^3 on the production field (gate rule); eps 1 anywhere on the production
  field; the relaxed-stagnation 128^3 job (O5, cancelled). Not measured: robustness over realizations.
- The 4-digit comparison at 32^3 cannot resolve the 1e-6 criterion; it is bounded by the printed values (reading
  decided before the runs, Hypothesis).
- The full `ctest` on V100 and the byte-compare of the default configs belong to integration validation and are
  recorded in the increment file, not in this artifact.
- The diagnosis of the solver limit is an analysis consistent with the measured curves; the singular-value scaling
  is cited from SF-29 at 12^3-16^3, not re-measured at 32^3-128^3.

## Next step

Owner decision among the options above. SF-34 is blocked until then (it depends on SF-33 `done`).
