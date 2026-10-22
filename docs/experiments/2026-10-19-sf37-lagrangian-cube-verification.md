# SF-37: Verification of the Lagrangian consumer on the periodic cube at sigma^2 in {1, 2.25, 4}, with blind O3

- Date: 2026-10-19 (activation commit `b27ab81`, first job 13:33:50Z) to 2026-10-22 (digests 07:56Z, `sf37-final` 08:46Z, `sf37-audit` 08:50Z)
- Status: `measurement complete; all pre-registered parts ran to completion; the classification per criterion is in "Result"; the increment State is set by the orchestrator after FINAL_AUDIT and by human review, not by this note`
- Increment: [`SF-37-lagrangian-cube-verification.md`](../plans/active/lester-eq14/increments/SF-37-lagrangian-cube-verification.md)
- Decision record: [`2026-10-17-lagrangian-consumer-sigma4.md`](../decisions/2026-10-17-lagrangian-consumer-sigma4.md) (items 2-4 and 8, stop rules R1, R2, R5)
- Theory: [`lester-2023-key-claims.md`](../theory/lester-2023-key-claims.md) §5 (sources A and B), §6, Rule 3
- Predecessor note (structure and tracker under test): [`2026-10-17-sf36-lagrangian-tracker-core.md`](2026-10-17-sf36-lagrangian-tracker-core.md)
- Artifacts: [`artifacts/2026-10-19-sf37-lagrangian-cube-verification/`](artifacts/2026-10-19-sf37-lagrangian-cube-verification/README.md)
  (`scripts/`, `logs/`, `raw/`, `analysis/`)
- Commits: base `master 44d4697`; activation `b27ab81`; **measured source tree H1 `d866c40`** (every head-binary job); delivery head
  before the evidence import `530bc76` and the import `edadc52` are docs/evidence only (no source change).
- Binaries (V100, `build/v100-release`, GPU 0 `Tesla V100-PCIE-32GB`): head `inlet_slab` md5 `a2ecce53bdeda469b41593f706dd7028`,
  `macroflow3d_pipeline` md5 `d0112ce8497b751303543c7de60a9e94`; base `inlet_slab` (built from `44d4697`, used for the gridded labels and
  as the criterion-R reference) md5 `fef042097f08dc725831f3df7ab975b5`. Analysis toolchain python 3.11.7, numpy 1.26.4, scipy 1.11.4.
  The CUDA and GCC versions are not recorded in the SF-37 job logs (the SF-36 note lists CUDA 11.4.152, GCC 9.3.1 for the same preset).

Abbreviations (`A` = `artifacts/2026-10-19-sf37-lagrangian-cube-verification/`): `V12` = `A/analysis/v1v2/v1v2_summary.md`, `V45` =
`A/analysis/v4v5/v4v5_summary.md`, `V3` = `A/analysis/v3/v3_summary.md`, `O3` = `A/analysis/o3/o3_summary.md`, `AUD` =
`A/analysis/audit/audit_summary.txt` (the **independent recomputation (orchestrator, `scripts/audit_recompute.py`)**, written from the spec
without the digest code; its CSVs are in the same directory), `SPEC` = the increment spec, `ACT` = its activation bitacora row
2026-10-19T13:35Z (reading rules RR1-RR15 and diagnostics D1-D9), `BIT` = its later bitacora rows. Every number below is copied from a cited
committed file; values marked `~` or a "rounded" column header are truncated for readability; counts marked "(counted)" were obtained by
counting rows of the cited CSV.

## Question

The increment's exact Goal: `Verificar en el cubo periódico, con semillas nuevas y umbrales pre-registrados, que las estadísticas de la pluma del tracker lagrangiano convergen en tolerancia y en h para sigma^2 en {1, 2.25, 4}, contrastarlas con las etiquetas en grilla verificadas de SF-35 (sigma_Y <= 1) y con el comparador de Pollock sobre las caras de SF-19, fijar con evidencia el lambda/h de producción y aplicar a ciegas la métrica robusta O3 de las etiquetas en grilla.`

Claims allowed (SPEC "Scientific or engineering intent"): the per-amplitude classification of tolerance-induced (V1) and grid-induced (V2)
changes of the plume statistics and the production `lambda/h`; the agreement with the gridded labels (V3a) and with Pollock (V3b) and the
Lagrangian `e_v` (V4); the stretching and decorrelation behaviour (V5); a blind verdict of O3. **No claim about `alpha_L` or `alpha_T`**
(periodic-cube statistics; the study is SF-39), and none is presupposed (`AGENTS.md`; theory note Rule 1).

## Hypothesis

Pre-registered thresholds (SPEC "Acceptance thresholds" and "Implementation specification"; `Q` ranges over `sigma_T^2(x1)` and `var tau(x1)`
at every plane, per seed):

| criterion | pre-registered rule |
|---|---|
| V1 (tolerance, source B) | `N` in {256, 512}, every amplitude, seed, plane: `\|Q(1e-8) - Q(1e-10)\| <= max(1e-3 \|Q(1e-10)\|, 2 SE_D)` for `Q = sigma_T^2`, at `h/8` |
| V2 (grid, source A) | every amplitude, seed, plane, `Q`: observed order `>= 1.5` on 128 -> 256 -> 512, or `\|Q(256) - Q(512)\| <= max(0.02 \|Q(512)\|, 2 SE_D)` |
| production `lambda/h` | per amplitude the smallest of {8, 10, 16, 20, 32, 64} with `\|Q(N) - Q(512)\| <= max(0.05 \|Q(512)\|, 2 SE_D)` at every plane, `Q`, seed; the study uses the maximum over amplitudes |
| V3a (recorded, classified) | per seed at `eps = 1`: SF-35 criterion C `accept` on 64/128/256; then unweighted RMS of `\|psi_grid(x_p) - psi_p\|` per plane `concordant` iff it decreases on both refinements |
| V3b (recorded, classified, not blocking) | `D_P(N) = sigma_T^2(Pollock) - sigma_T^2(O6)`: `concordant` iff `\|D_P\|` decreases on 64 -> 128 -> 256 -> 512 and `\|D_P(512)\| <= max(0.05 sigma_T^2(O6, 512), 2 SE_D)`, else `discordant`; a `discordant` is reported to the owner |
| V4 (Lagrangian `e_v`) | every amplitude, seed, plane `x1 <= 4`: RMS `e_v` decreases on every refinement 64 -> 128 -> 256 -> 512 with observed order `>= 1.8` on 256 -> 512 |
| V5 (recorded, no threshold) | covector percentiles and rates; decorrelation fits (exponential vs algebraic) and the named better fit per amplitude |
| O3 (blind, recorded, not blocking) | per `(eps, seed)` at `eps` 1.5, 2: O3a `accept` if `e_v^(q)` decreases on every refinement with order `>= 1.8` on 256 -> 512, `reject` if it changes `< 10 %` between 256 and 512 while above 1e-12, else `unresolved`; O3b `shrinking` iff `f_half` decreases on every refinement; O3c recorded; `q = 1e-3` |
| R (regression) | `lagrangian_face_flux_repack` green; SF-35/SF-36 outputs byte-identical with `keep_face_fluxes = false`; full V100 `ctest` green |

Expected numerical effect (SPEC, "not presupposed"): V1 deterministic agreement at early planes and a transition to statistical agreement at
`sigma^2 = 4`; V4 order ~2; V3b Pollock excess decreasing with `N`; O3 not predicted.

**Pre-registration.** The spec blob `e6e0acb98753dbb0a32a1c268acf7cb4f8a9b063` (identical at `040e3ce` and `44d4697`) and the reading rules
RR1-RR15 and diagnostics D1-D9 (`ACT`) were committed in the activation commit `b27ab81` before any run; the first run (job `sf37-base-refs`)
started at 13:33:50Z (`BIT` row 2026-10-19T15:50Z). The bootstrap index matrix `A/analysis/bootstrap_blocks.npy`
(`default_rng(20261017).integers(0, 16, size=(2000, 16))`, int64-data sha256 `16c1a5b9d9fe062576d5f18fc9d54127db76472c7fbe64493201d8cffc66af39`)
was committed before the campaign and verified by every digest (`V12` header). `o3.py` and RR9 were frozen at `60c4471` before any
`eps` 1.5/2 export of seeds 3101-3103 was read (`BIT`). No threshold, design value, Goal or failure-policy text was changed afterwards.

## Build / environment

H1 `d866c40` built with `cmake --preset v100-release` (job `sf37-build`, exit 0); the head `inlet_slab` differs from the base binary and its
usage text lists `--comparator` (`A/raw/bytecmp/results.txt`, last lines). Jobs (all `STATE=succeeded`, `EXIT_CODE=0`,
`A/logs/status/*.env`; start-end from `A/README.md`, strictly sequential, GPU 0, one SF-37 mirror `~/MacroFlow3D-SF-37`):

| job | what | UTC window |
|---|---|---|
| `sf37-base-refs` | base build of `44d4697`, criterion-R references | 2026-10-19T13:33:50Z-13:48:37Z |
| `sf37-labels-eps10` | SF-35 gridded labels `eps` 1, seeds 3101-3103, `N` 64/128/256 (9 constructions, base binary) | 2026-10-19T14:03:33Z-14:31:46Z |
| `sf37-labels-o3` | gridded labels `eps` 1.5 and 2, `N` 64-512 (24 constructions, base binary; blind O3 input) | 2026-10-19T14:32:43Z-2026-10-20T15:13:29Z |
| `sf37-build`, `sf37-dryrun` | head build; `SF37_DRYRUN=1` smoke (NOT evidence, tree not imported) | 2026-10-20T19:37:32Z-19:47:02Z |
| `sf37-bytecmp` | criterion R, head vs base | 2026-10-20T19:48:46Z-19:51:26Z |
| `sf37-v2` | grid ladder (63 runs) | 2026-10-20T19:51:56Z-21:52:25Z |
| `sf37-science` | parts `v3a` (9 runs), `v3b` (36), `v1` (108), `digest` | 2026-10-21T13:57:58Z-2026-10-22T07:56:16Z |
| `sf37-final` | head build, full V100 `ctest`, SF-27 trio vs base | 2026-10-22T07:58:19Z-08:46:56Z |
| `sf37-audit` | independent recomputation | 2026-10-22T08:49:15Z-08:50:36Z |

## Config(s)

| item | value |
|---|---|
| field | SF-18 Gaussian, `--sigma2 1 --ell 0.25` (`normalize_variance`, raw variance recorded), `k = exp(eps Y)`, periodic cube `[0,1]^3`, mean flux `e1`, double precision |
| seeds (fresh; 3001 excluded) | 3101, 3102, 3103 (`R5`) |
| `eps` -> amplitude | 1 -> `sigma^2` 1; 1.5 -> 2.25; 2 -> 4 (codes `10`, `15`, `20` in the artifact) |
| grids `N` (`lambda/h = N/4`, `lambda = ell = 1/4`) | V2/V4: 32, 40, 64, 80, 128, 256, 512 (`lambda/h` 8, 10, 16, 20, 32, 64, 128); V1: 256, 512; V3b: 64-512; V3a labels: 64, 128, 256; O3: 128, 256, 512 with pairs (N/2, N) |
| injection | whole face `x1 = 0`, `M = 256` (65536 particles), flux weights |
| planes / times | `x1` in {0.25, 0.5, 1, 2, 4, 8, 16} (unwrapped; 1 = one period `4 ell`); `t` in {0.25, 0.5, 1, 2, 4, 8, 16}; V3a planes 0.25, 0.5, 0.75, 1 |
| primary setting | `--tol 1e-8 --hmax-div 8` (`h_max = h/8`), `--max-arclength 1000` (RR11) |
| V1 ladder | `--tol` in {1e-6, 1e-7, 1e-8, 1e-9, 1e-10} at `h/8` and the reference `(h/32, 1e-10)`, all `--covectors --save-seed-velocity` |
| bootstrap | 16 blocks (`ell x ell` squares of 64 x 64 seeds), `B = 2000`, seed 20261017 |
| runs | 171 forward runs for V1/V2/V4/V5 (63 + 108), 9 V3a, 36 V3b (Pollock), 33 gridded-label constructions |

## Commands

Exact strings are in `A/logs/commands/*.command` (`sf37-base-refs`, `sf37-labels-eps10`, `sf37-labels-o3`, `sf37-build`, `sf37-dryrun`,
`sf37-bytecmp`, `sf37-v2`, `sf37-science`, `sf37-final`, `sf37-audit`); job logs in `A/logs/jobs/`, per-part logs of `sf37-science` in
`A/logs/parts/`. Runner `A/scripts/run_sf37.sh <bin_dir> <part>` with parts `labels-eps10`, `labels-o3` (base binary `~/sf37_base_refs/bin`),
`bytecmp`, `v2`, `v3a`, `v3b`, `v1`, `digest` (head binary), each launched with `SF37_COMMIT=<hash>`; `sf37-science` is the loop
`for p in v3a v3b v1 digest; do bash $A/run_sf37.sh build/v100-release $p; done` (`A/logs/commands/sf37-science.command`). Digests (part
`digest`): `digest_v1v2.py`, `digest_v4v5.py`, `digest_v3.py`, `o3.py` as `<script> $HOME/sf37_campaign <out> --design full`
(`V12`, `V45`, `V3`, `O3` headers); exit codes all 0 (`A/analysis/exit_codes.tsv`). Independent recomputation:
`audit_recompute.py ~/sf37_campaign analysis/bootstrap_blocks.npy ~/sf37_audit --o3 --v3a`
(`AUDIT_SCRIPT_SHA256=ebafc94f330d7bd99e01320a08aa0859102d9910e504487c3b41967fcdd3856c`, `A/logs/jobs/sf37-audit.log`).

## Outputs inspected

### 0. Completeness and job outcomes

Expected runs 171, present 171, missing 0 (`V12` "Completeness", `V45` "Completeness"); every part exit 0 with `blocking_failures=0`
(`A/logs/jobs/*.log`, `A/raw/{v2,v3a,v3b,v1}/exit_codes.tsv`: 63, 9, 36, 108 runs, all rc 0). Gridded-label constructions: 9/9 rc 0 at `eps` 1;
at `eps` 1.5/2 rc 0 for all 18 at `N` 128/256/512 and for `eps` 1.5 seed 3101 at `N` 64, rc 21 (`roundtrip_exceeded`, labels saved, accepted by
RR11) for the other five at `N` 64 (`A/raw/labels/exit_codes_labels-o3.tsv`). Analysis consistency (RR1): recomputed moments agree with
`moments_planes.csv` to 3.214e-14 relative over 693 (run, plane) (`AUD`). Per RR14 all parts run to completion regardless of intermediate
verdicts; nothing was re-run or tuned.

### 1. V1 - tolerance, `N` in {256, 512} (rule RR15-labelled)

| `sigma^2` | cells | fail | not evaluable | binding branch | RR15 label (`V12`) |
|---|---|---|---|---|---|
| 1 | 42 | 0 | 0 | det (42) | met, vacuous (tol-independent step sequence) |
| 2.25 | 42 | 0 | 0 | det (42) | met, vacuous (tol-independent step sequence) |
| 4 | 42 | 0 | 0 | det (42) | met, vacuous (tol-independent step sequence) |

All 126 cells pass with `D = 0`, `SE_D = 0` (`V12` "V1 cells", e.g. `sigma^2` 1, seed 3101, `N` 256, plane 0.25: `Q(1e-8) = Q(1e-10) =
0.0846614`). Independent recomputation: 126 cells, 0 failing, per amplitude 42/42 met, det 42, bitwise-identical planes 42 (`AUD`).

**V1 is not evidence that the integration error is controlled (RR15).** The step is capped by `h_max = h/8` (SF-36 note, J), and the records of
the runs are bitwise identical for every `tol` from 1e-6 to 1e-10 at `h/8`: identical-weight fraction 1.0, identical statuses, first plane with
`D != 0` = `none` and identical pass counters in all 90 `h/8` rows of `A/analysis/v1v2/rr15_vacuity.csv` (e.g. eps 1, seed 3101, `N` 256: `plane_nfev`
13528003860 for `tol` 1e-6, 1e-7, 1e-8); the V1 run at `(1e-8, h/8)` is also bitwise identical to the V2 primary run in 18 of 18 `(eps, seed, N)`
(`V12` D9). The tolerance was never the binding step limiter, so the V1 inequality compared two identical computations.

**RR15 shadow (the integration-error evidence that exists; recorded, not blocking).** `(h/8, 1e-10)` vs `(h/32, 1e-10)` (a pure `h_max` change; the records
differ: identical-weight fraction 0.0, first plane with `D != 0` = 0.25 in the 18 `h/32` rows of `rr15_vacuity.csv`) and `(h/8, 1e-8)` vs `(h/32,
1e-10)` (`V12` "RR15 shadow classifications"): the V1 inequality passes in all 36 rows, no failing plane, and "first plane RMS > 0.1 sigma_T" is
`none` in all of them. Largest `MS(dx_T)/sigma_T^2` per amplitude: `sigma^2` 1: 8.08972e-21 (seed 3102, `N` 256); 2.25: 2.41913e-17 (seed 3102, `N` 256);
4: 1.27272e-13 (seed 3102, `N` 256) (the `N` 512 values are lower). The `h_max` change moves the plume statistic far below the V1 threshold;
it does not exercise the tolerance.

### 2. V2 - grid, `N` 128 -> 256 -> 512

| `sigma^2` | cells | fail | not evaluable | binding branch | (seed, Q) passing only via 2 SE_D at every far plane |
|---|---|---|---|---|---|
| 1 | 42 | 0 | 0 | det (all) | 0 / 6 |
| 2.25 | 42 | 0 | 0 | det (all) | 0 / 6 |
| 4 | 42 | 0 | 0 | det (all) | 0 / 6 |

`V12` "V2": 126/126 cells pass on the relative branch (`det_ok = true`); the order branch (`>= 1.5`) is also satisfied in 117/126 (counted,
`v2_cells.csv`), the 9 cells where it is not (2 at `sigma^2` 2.25, 7 at 4) pass by `det`. In 71 cells the order is computed from differences
below 2 SE_D (`order_from_noise` = true, counted), so it is not informative there; typical orders are 1.86-2.16 at `sigma^2` 1 (42 of 42 `>= 1.5`).
Only one (seed, plane) has `RMS(dx_T) > 0.1 sigma_T` for the 256-vs-512 pair (`sigma^2` 4, seed 3102, plane 16: the 2 cells with `far` = true, counted). Independent
recomputation: 126 cells, 0 failing (`AUD`).

### 3. Production `lambda/h` (RR5) and D1 multiplicity

Candidate table (`V12` "Production lambda/h"); each candidate is 42 cells (3 seeds x 2 `Q` x 7 planes); pass, or fail with the number of failing
cells; worst `|D| / max(threshold)` in parentheses:

| `lambda/h` (`N`) | `sigma^2` 1 | `sigma^2` 2.25 | `sigma^2` 4 |
|---|---|---|---|
| 8 (32) | pass (0.58314) | fail (2) | fail (4) |
| 10 (40) | pass (0.375943) | fail (1) | fail (2) |
| 16 (64) | pass (0.150157) | pass (0.751971) | fail (1) |
| 20 (80) | pass (0.0972129) | pass (0.482681) | fail (1) |
| 32 (128) | pass (0.0365294) | pass (0.218644) | pass (0.56124) |
| 64 (256) | pass (0.00741868) | pass (0.0598699) | pass (0.201683) |

Primary (literal) = monotone variant: **8 / 16 / 32** at `sigma^2` 1 / 2.25 / 4 (no difference to report to the owner, RR5); the **study value
(maximum over amplitudes) is `lambda/h = 32`, i.e. `N = 128` at `lambda = ell = 1/4`**. Independent recomputation: passing candidates {8, 10, 16, 20, 32, 64},
{16, 20, 32, 64}, {32, 64}; literal and monotone 8 / 16 / 32 (`AUD`). Branches over the 756 cells (counted, `lambda_h_cells.csv`): det 740, stat 5, fail 11; at `lambda/h` 32 and 64 every cell passes on
the relative (det) branch at all three amplitudes.

**D1 multiplicity line** (`V12` D1): for `lambda_h`, 756 cells, 32 in the statistical regime (2 SE_D above the relative threshold), expected chance
exceedances 1.60 / 2.05 / 2.08 (5 % / t15 / 6.5 %), **failing cells 11, of which 8 in the statistical regime**. The 11 failing cells have
`z` between -4.01 and +2.28 (`d1_cells.csv`); sign agreement across seeds 2 of 2 groups, across adjacent planes 2 of 2 pairs. Implication: 3 of the
11 failures sit where the relative branch dominates (not attributable to bootstrap noise), and the 8 statistical-regime failures are about four
times the expected chance count; the failures of the smaller candidates are not attributable to chance alone. V1 and V2 have 0 statistical-regime cells (0.00 expected).
SE_D with 4 / 64 blocks relative to 16: medians 1.06 / 0.732 (`V12`).

### 4. V3a - agreement with the verified SF-35 gridded labels (`eps = 1`)

SF-35 criterion C (invoked in place, `orders_digest.py`) on the labels at 64/128/256: `accept` for seeds 3101, 3102, 3103
(`V3` V3a table, `A/analysis/v3/v3a_criterionC.csv`, `criterionC_s31xx.md`); constructions rc 0, max round trip 2.54909e-09 (`v3a_constructor.csv`);
stage identity trace vs forward identical for G, `inlet_vmin`, `v_rms` in 9/9 (`V3`). Classification: **12 concordant of 12 (seed, plane)
cells** (`V3`, `v3a_verdicts.csv`); unweighted RMS of `|psi_grid(x_p) - psi_p|` decreases 64 -> 128 -> 256 in every cell, e.g. seed 3101,
`x1` 1: 6.34511e-06, 2.84229e-07, 1.90358e-08; largest at `N` 64 is 1.93747e-05 (seed 3102, `x1` 1), at `N` 256 3.17125e-08 (same cell).
Independent spot check: 36 rows recomputed with SciPy `map_coordinates` (`A/analysis/audit/v3a_spot.csv`; `BIT` row 2026-10-22T09:15Z).

### 5. V3b - Pollock (RT0 on the SF-19 faces) against the Lagrangian O6 (not blocking)

Pollock and O6 are paired in the same process (`--comparator pollock`); O6 files are bitwise identical to the `v2` runs and the Pollock moments reproduce
the CSV to <= 5.5e-16 (`V3` "Consistency", 36/36). No Pollock non-active particle in any of the 72 (run, pass) records
(`v3b_pollock_status.csv`, counted).

| `sigma^2` | seed 3101 | seed 3102 | seed 3103 | concordant / 21 |
|---|---|---|---|---|
| 1 | 0 / 7 | 2 / 7 (planes 8, 16) | 0 / 7 | 2 |
| 2.25 | 0 / 7 | 3 / 7 (planes 4, 8, 16) | 0 / 7 | 3 |
| 4 | 1 / 7 (plane 16) | 3 / 7 (planes 1, 8, 16) | 0 / 7 | 4 |

Totals: **54 discordant, 9 concordant, 0 not evaluable of 63** (`V3` "Cells"); the audit gives the same per-amplitude counts 2 / 3 / 4 of 21
(`AUD`). Binding clause: the magnitude clause holds in 63/63 (counted; the largest `|D_P(512)|` is 8.99604e-4 against a threshold of 0.00897601,
`sigma^2` 4, seed 3101, plane 8), and the failing clause is "`|D_P|` decreases on every refinement": in all 54 discordant cells `|D_P(512)| >
|D_P(256)|` (counted), in 49 of them `|D_P|` decreases 64 -> 128 -> 256, and `D_P(512)` is negative in 59 of 63 cells (counted, `v3b_verdicts.csv`).
`D_P` is small in absolute terms (`|D_P(256)| <= 1.325e-3`, `|D_P(512)| <= 8.996e-4`, against `sigma_T^2` of order 0.04-0.3) and 152 of the 189
refinement steps are flagged noise-limited (`|D_P(N) - D_P(2N)| < 2` x the larger SE_D; 45 of the 63 steps 256 -> 512; `v3b_noise_limited.csv`,
counted). **Verdict (RR7): `discordant`, reported to the owner, non-blocking; it stays the verdict.**

**D6 decomposition (recorded observation).** The digest splits `D_P` as `D_weight_part = sigma^2(P-traj, P-w) - sigma^2(P-traj, O6-w)` and
`D_trajectory_part = sigma^2(P-traj, O6-w) - sigma^2(O6)` (`v3b_cells.csv`; `digest_v3.py` docstring); the audit uses the other ordering of the two-factor
split, `WP = S(X_O6, w_P) - S(X_O6, w_O6)` and `TP = D_P - WP` (`A/analysis/audit/v3b_cells.csv`); the two differ by the interaction term only, `D_P`
agrees to 7.5e-14 (`BIT` row 2026-10-22T09:15Z; e.g. 0.0013551701718036296 vs 0.0013551701718056974). Example (`sigma^2` 4, seed 3101, `x1` 4):

| `N` | `D_P` | weight part (digest) | trajectory part (digest) | `WP` (audit) | `TP` (audit) |
|---|---|---|---|---|---|
| 64 | 1.35517e-3 | -2.68474e-5 | 1.38202e-3 | -2.98906e-5 | 1.38506e-3 |
| 128 | 3.22463e-4 | -9.65550e-6 | 3.32119e-4 | -9.68109e-6 | 3.32144e-4 |
| 256 | 7.81281e-5 | -4.84945e-6 | 8.29776e-5 | -4.85754e-6 | 8.29857e-5 |
| 512 | -5.92532e-4 | -6.16176e-4 | 2.36436e-5 | -6.16326e-4 | 2.37938e-5 |

Counts over the 63 (eps, seed, plane) cells (counted from the committed CSVs; they reproduce the orchestrator's counts): digest ordering -
`|D_weight_part|` grows from 256 to 512 in 63/63 cells (ratio median 41.82, range 1.74-614.1), `|D_trajectory_part|` is strictly decreasing
64 -> 128 -> 256 -> 512 in 48/63 (and smaller at 512 than at 256 in 56/63); audit ordering - `|WP|` grows in 63/63 (ratio median 41.81, range 1.737-613.8),
`|TP|` strictly decreasing in 48/63. The weight part decreases strictly 64 -> 128 -> 256 in 62 of 63 cells (digest), then jumps at `N` 512. In this decomposition the
non-monotone step at `N` 512 sits in the weight part (recorded observation).

**Mechanism: an orchestrator hypothesis from reading the code, not a measured result and not a re-verdict.** The seeds sit at `(m + 0.5)/M`
(`make_seed_lattice`, `src/physics/particles/lagrangian_labels/LagrangianTracker.cu`), so at `N = 2M = 512` every seed lies exactly on a transverse cell
boundary; the Pollock weight is the RT0 `x1`-velocity in the cell chosen by the core (`pollock_seed_weight_kernel`, `PollockComparator.cu`; cell choice
in `pollock_init_state`, `src/physics/particles/streamline_tracker/PollockTracker.cuh`), i.e. a face average offset by `h/2` in `x2` and `x3`: a one-sided
`O(h)` weight bias, which cancels to `O(h^2)` at `N <= 256` (face centres at 256, symmetric positions at 64/128). No run isolating this (e.g. shifted
seeds) was made. Verdict unchanged: `discordant`. Time-snapshot analogues, `D_P` slopes in `x1` and Pollock non-active weight: `v3b_time_snapshots.csv`,
`v3b_slopes.csv`, `v3b_pollock_status.csv` (the slopes are of mixed sign at `N` 512).

### 6. V4 - Lagrangian `e_v = 1 - exp(-I)`

| `sigma^2` | cells | fail | not evaluable | verdict |
|---|---|---|---|---|
| 1 | 15 | 0 | 0 | pass |
| 2.25 | 15 | 0 | 0 | pass |
| 4 | 15 | 0 | 0 | pass |

RMS `e_v` decreases strictly on every refinement in 45/45 cells and the observed order on 256 -> 512 is 1.99582-2.00679 (`v4_cells.csv`,
`V45`; audit range 1.996 .. 2.007, 45 cells, 0 failing). Example, plane 4, `N` 64 -> 512 (rounded, `V45`):

| `sigma^2` | seed 3101 | seed 3102 | seed 3103 |
|---|---|---|---|
| 1 | 0.0153831 -> 2.38616e-4 | 0.0126624 -> 1.96442e-4 | 0.010786 -> 1.67282e-4 |
| 2.25 | 0.0324242 -> 5.05974e-4 | 0.0239276 -> 3.7387e-4 | 0.0245867 -> 3.80486e-4 |
| 4 | 0.0580933 -> 9.17482e-4 | 0.039723 -> 6.27084e-4 | 0.0418192 -> 6.48457e-4 |

D4 (recorded): the V4 rule re-run on RMS `I` passes in 45/45 and on the 99.9 %-trimmed RMS `e_v` in 45/45 (`V45`, `d4_robustness.csv`). D3: the passive-`I`
change between `(h/8, 1e-8)` and `(h/32, 1e-10)` at `x1 <= 4` has median relative 1.75224e-06 over 90 cells (max 3.57642e-05), and the V4 order on the
reference runs is `>= 1.8` in 45/45 (`V45`, `d3_passive_I.csv`). The measured `e_v` is the divergence of the spline representation `v_s`
(source A reduced to divergence, theory note §6), not the integration error.

### 7. V5 - covector stretching and pair decorrelation (recorded, no threshold)

| `sigma^2` | named fit (amplitude, majority over evaluable `(seed, N)`) | evaluable named cases | FALLBACK `(h/8, 1e-10)` vs `(h/32, 1e-10)` (reported only as fallback) |
|---|---|---|---|
| 1 | `insufficient` | 0 / 6 | exponential (2 evaluable cases) |
| 2.25 | `insufficient` | 0 / 6 | mixed (4 evaluable: 2 exponential, 2 algebraic) |
| 4 | `insufficient` | 0 / 6 | exponential (6 evaluable: 4 exponential, 2 algebraic) |

The named pair `(1e-8, 1e-10)` at `h/8` has no eligible plane (`1e-12 < RMS(dx_T) < 0.1 sigma_T` is never met because the pair is identical, RR15), so no fit is named
(`V45` "V5", `v5_fits.csv`). The fallback pairs separate trajectories by an `h_max` change, not a tolerance change; the pair `(1e-6, 1e-10)` at `h/8` is
`insufficient` at every amplitude (also identical). Fallback exponential slopes of `log RMS(dx_T)` vs `x1` are 0.15-0.76 per unit `x1` where evaluable
(`v5_fits.csv`). Covector norms (`V45`, unweighted over contributors), `|a|` at plane 16, `N` 512, p50 / max: `sigma^2` 1: 2.24525 / 2093.18, 2.52323 / 2987.27,
1.54709 / 12.1649; 2.25: 4.29157 / 7750.91, 7.3034 / 92082.5, 2.25491 / 60.0177; 4: 10.5196 / 6.94141e+06, 36.3868 / 5.07964e+07, 3.65994 / 663.217
(seeds 3101, 3102, 3103). The tails are not grid-converged (`sigma^2` 1, seed 3101, plane 16 `|a|` max: 10062.9 at `N` 256 vs 2093.18 at 512). Rates:
`v5_rates.csv`, `v5_local_slopes.csv`.

**Consequence per the spec's failure policy.** The policy item applies when V5 "names the exponential fit" (then SF-39 long-range criteria are statistical only, R2).
Here no fit is named at any amplitude (`insufficient`), so that clause is not triggered by the named result; the fallback fits are not a named result and are not
used as one. The statistical-only reading of SF-39's long-range criteria therefore remains an open owner/SF-39 decision, not a finding of this note.

### 8. O3 - blind robust metric of the gridded labels (not blocking)

Metric as pre-registered: `c` by 2nd-order central differences, `d = |c_N - c_{N/2}| / RMS|c_N|`, `M_q` = the top `q = 1e-3` fraction of the common vertices by `d`
(`n_marked` = 259 / 2081 / 16712 at `N` 128 / 256 / 512, `o3_cells.csv`), O3a over the unmarked vertices (`O3`, `A/analysis/o3/o3_verdicts.csv`):

| `sigma^2` | seed | O3a | order 256 -> 512 | `e_v^(q)` 128 / 256 / 512 | O3b | `f_half` 128 / 256 / 512 |
|---|---|---|---|---|---|---|
| 2.25 | 3101 | accept | 1.98629 | 0.00923625 / 0.00237677 / 0.000599867 | shrinking | 0.00457279 / 0.00375006 / 0.00336794 |
| 2.25 | 3102 | accept | 2.12746 | 0.00933969 / 0.00234119 / 0.000535804 | shrinking | 0.0020035 / 0.000141294 / 2.68674e-05 |
| 2.25 | 3103 | accept | 1.99345 | 0.00867114 / 0.0020787 / 0.000522038 | shrinking | 0.00179036 / 0.000970795 / 0.00076276 |
| 4 | 3101 | accept | 2.20155 | 0.0486779 / 0.0171976 / 0.00373881 | shrinking | 0.00178649 / 0.00164507 / 0.000276812 |
| 4 | 3102 | **unresolved** | 1.65159 | 0.122639 / 0.0737958 / 0.0234884 | shrinking | 0.000193762 / 2.54714e-05 / 8.2577e-06 |
| 4 | 3103 | accept | 2.59793 | 0.23737 / 0.150569 / 0.0248703 | shrinking | 0.00023639 / 0.000102847 / 4.45198e-05 |

(`sigma^2` 4 seed 3102: relative change 256 -> 512 = 0.68171, so not `reject`; the other seeds 0.7476-0.8348.) Independent recomputation reproduces the six verdicts, orders
(1.986, 2.127, 1.993, 2.202, 1.652, 2.598) and `f_half` (`AUD`). O3b reads as concentration of the defect, not convergence (`ACT` D7).

**`q = 1e-3` disclosure.** The metric excludes the top 0.1 % of the common vertices ranked by `d` (259 / 2081 / 16712 vertices); `q` and the ranking were chosen after the
SF-35 post-hoc inspection of seed 3001 (SPEC item 6), the application to 3101-3103 is blind. The exclusion matters at `sigma^2` 4: the unmasked
`e_v` over all common vertices (`o3_cells.csv`, `e_v_all`, rounded) is 0.0546 / 0.0235 / 0.00980 (seed 3101), **0.234 / 0.437 / 0.917 (seed 3102, growing with `N`)**
and 0.470 / 0.395 / 0.209 (seed 3103) at `N` 128 / 256 / 512, against the masked values above; for `sigma^2` 2.25 it is 0.0104 / 0.00275 / 0.000700,
0.0163 / 0.00820 / 0.00278 and 0.0112 / 0.00312 / 0.000804. `d_max` for seed 3102 at `sigma^2` 4 is 20.5596 / 76.8302 / 232.682 (`N` 128 / 256 / 512), located near
`x1` 0.83-0.93, `x2` 0.03-0.06, `x3` 0.78-0.80 (`o3_cells.csv`). The O3a verdicts are therefore verdicts of the masked metric.

O3c (weighted fraction of O6 landings within one coarse cell of a marked vertex, planes `x1` 0.25 and 0.5, `o3c.csv`): 0 at `x1` = 0.25 in all 18 `(sigma^2, seed, N)`; at
`x1` = 0.5 nonzero in 11 of 18 (counted), decreasing with `N` where nonzero: `sigma^2` 2.25 seed 3101 0 / 0.000599844 / 0.000446769, seed 3103 0.0033348 / 0.00203459 /
0.00140361; `sigma^2` 4 seed 3101 0.00213087 / 0.00192127 / 0.00149105, seed 3103 0.00250004 / 0.00166711 / 0.0011395 (`N` 128 / 256 / 512); seed 3102 is 0 at both
amplitudes; `x1` = 1 is the outlet coarse plane, structurally empty by construction.

### 9. R - regression

- `sf37-bytecmp` (head vs base, same arguments, RR13): `BYTECMP_ALL_IDENTICAL`, 214/214 comparisons IDENTICAL (45 `--construct trace` items over 9 cases, 2 `--production 32` entries
  (the ladder, 102 lines, and its exit code), 118 `--track forward` items over 7 cases, 47 items of the neutral `--comparator pollock` / `--save-vd` / `--save-seed-velocity` re-runs of 3 cases, and the 2 binary checks (md5 differs, usage lists
  `--comparator`); `A/raw/bytecmp/results.txt`, `A/logs/jobs/sf37-bytecmp.log`): SF-35 and SF-36 outputs are byte-identical with `keep_face_fluxes = false`.
- `sf37-final` (H1 tree): `BUILD_EXIT=0`, same `inlet_slab` md5; full V100 `ctest` **38/38 passed**, `Total Test time (real) = 2901.55 sec` (`CTEST_WALL_S=2902`), against the SF-36 37/37 in 2881.9 s;
  `lagrangian_face_flux_repack` Passed (4.18 s); SF-27 trio `config_pspta_small`, `config_streamfunctions_homogeneous`, `config_streamfunctions_continuation` exit 0 with
  `BYTECMP_IDENTICAL` (excluding `manifest.json`) and `MANIFEST_IDENTICAL_MODULO_TIMES` against the base build (`A/logs/jobs/sf37-final.log`).
- `sf37-final` ran on the H1 tree before the evidence import (DAG amendment of 2026-10-20, `BIT`); later commits are docs/evidence only (to be verified by the FINAL_AUDIT).

### 10. Recorded diagnostics (no thresholds)

- D2: `applied_scale` and `raw_variance` equal to <= 1e-12 across all runs of a seed (relative spread 0); plane-pass field evaluations per particle and unit length double with `N`
  (1669.5 at 32 ... 26640.7 at 512, `V12`), i.e. the cost is set by the `h_max` cap (SF-36 note, J).
- D9: largest non-contributing weight fraction 0; V1 `(1e-8, h/8)` vs V2 primary bitwise identical 18/18.
- D8 (shadow of the SF-39 estimator quantities, recorded, not blocking, `V12`): V1 and V2 shadow verdicts all pass for `var x_k(t)`, `Delta var_k / Delta t` and `Delta sigma_T^2 / Delta x1`
  (V2 63/63 or 54/54 per series). Their shadow production `lambda/h` (primary = monotone) is larger than the study value for some series: `var x1` 16 / 32 / **64** and `dvar_x1_dt` 16 / 32 / **64**
  at `sigma^2` 1 / 2.25 / 4, `dvar_x3_dt` 16 / 64 / 64, against `dsigma_T2_dx1` 16 / 20 / 32 and `var x2` 8 / 16 / 32. This is not a pre-registered criterion; it is flagged for SF-39.

## Result

Classification per `docs/AGENTS.md`.

- **Met as pre-registered (blocking):**
  - V1: 126/126 cells, det branch, at every amplitude, but **"met, vacuous" (RR15)**: not evidence of integration-error control. The only integration-error evidence is the RR15 shadow
    (`h/8` vs `h/32`, all pass, `MS(dx_T)/sigma_T^2 <= 1.27272e-13`, no 0.1 `sigma_T` plane).
  - V2: 126/126, det branch; 0 / 6 `(seed, Q)` per amplitude pass only via 2 SE_D.
  - V4: 45/45, order 1.99582-2.00679 on 256 -> 512.
  - Production `lambda/h`: 8 / 16 / 32 at `sigma^2` 1 / 2.25 / 4 (primary = monotone); **study value 32 (`N = 128`)**.
  - R: 214/214 byte-identical, `ctest` 38/38 (2901.55 s), `lagrangian_face_flux_repack` green, SF-27 trio identical.
- **Recorded with classification (non-blocking):** V3a 12/12 concordant (C `accept` for the three seeds); V3b **discordant 54 / concordant 9 / not evaluable 0 of 63** (RR7: stays `discordant`; the
  weight/trajectory decomposition is a recorded observation and the `h/2` seed-offset mechanism is an orchestrator hypothesis); O3a `accept` in 5 of 6 `(sigma^2, seed)` and `unresolved` in
  `sigma^2` 4 seed 3102, O3b `shrinking` in 6 of 6, O3c as above, all under the `q = 1e-3` disclosure; V5 named fit `insufficient` at every amplitude (fallbacks only).
- **RR14 reading.** Stop conditions checked: V1 or V2 not met at any amplitude - no (R1 at `sigma^2` 4 does not fire); V4 not met - no; production `lambda/h` "none <= 64" - no (32);
  a blocking cell not evaluable - no (0). Per RR14 no stop rule fires. The "otherwise" clause requires listing every non-blocking outcome: (1) V1 vacuous (RR15) - **yes**, all 126 cells;
  (2) V3b discordant - **yes**, 54 of 63 (to the owner); (3) O3 reject / unresolved / not shrinking - **unresolved: 1 of 6** (`sigma^2` 4 seed 3102), reject 0, not shrinking 0; (4) V5 exponential -
  no (named fit `insufficient`; fallback fits only); (5) V1/V2 passing only via 2 SE_D at every far plane - no; (6) V3a not concordant or reference not verified - no (12/12, C `accept` x3);
  (7) a shadow classification not met - no (RR15 shadows 36/36 pass). Also recorded for the owner: the D8 shadow `lambda/h` of 64 for some SF-39 estimator series (above).
- **Independent recomputation agreement:** V1, V2, `lambda/h`, V3b classes, V4, O3a/O3b and the V3a spot check are reproduced by the orchestrator recomputation (`AUD`; `BIT` row 2026-10-22T09:15Z reports
  agreement cell by cell, to <= 1e-12 relative, the only difference being the D6 ordering). V5, D1-D4 and D8 were not recomputed.
- **Production `lambda/h`:** 8 / 16 / 32 at `sigma^2` 1 / 2.25 / 4; the study value is 32 (`N = 128` for `lambda = ell = 1/4`). The smaller candidates fail in 11 cells, 3 of them where chance cannot explain it.

## Caveats

- One field realization per seed, three seeds; periodic cube, `sigma^2`-normalized SF-18 field, `ell = 1/4`, `N` up to 512; no ensemble statistics; cube statistics are periodic-medium statistics.
- **G8.** The decision record (item 4) calls the SF-19 faces "exactly solenoidal". They are solenoidal to the PCG residual (~3e-10), not to roundoff (`BIT` row 2026-10-19T15:50Z); this bounds what the
  Pollock comparator (V3b) can show. The decision record is not edited here.
- **RR15 vacuity.** V1 compares identical computations (step capped by `h_max = h/8` at every tested `tol`); the tolerance dependence of the integrator is not exercised and V5's tolerance fits are
  unavailable for the same reason.
- **Statistical branch and multiplicity (D1).** No V1/V2 cell uses the statistical branch; the `lambda/h` candidate comparisons do (32 cells), with 11 failures against ~2 expected by chance. SE_D rests on 16
  blocks of one realization per seed.
- **V3b.** The decomposition counts are observations; the mechanism is a hypothesis from reading the code and no confirming run exists. Pollock's own convergence is not established (decision item 4).
- **O3.** The metric excludes the top 0.1 % of vertices (`q = 1e-3`), chosen after inspecting seed 3001; at `sigma^2` 4 the unmasked quantity differs strongly from the masked one (seed 3102 grows with `N`).
  Gridded labels remain a diagnostic inside the O1 envelope (decision item 3).
- **No `alpha_L` / `alpha_T` claim.** This note measures plume-statistic convergence of the consumer only; whether any transverse dispersion is physical is not assessed, and `alpha_T` is not presupposed
  (`AGENTS.md`; theory note, Rule 1 and "Do not overclaim"). The classification of transverse behaviour is SF-39.
- **Process deviations.** (i) The local chain of remote jobs died with the Claude session after `sf37-v2` (GPU idle 2026-10-20T21:52Z-2026-10-21T13:58Z, no data affected); the remaining parts
  `v3a`, `v3b`, `v1`, `digest` ran unchanged as one detached job `sf37-science` (same script, options, mirror tree and binary; `A/logs/commands/sf37-science.command`, `BIT` row
  2026-10-22T09:15Z). (ii) `sf37-final` was run before the evidence import (DAG amendment of 2026-10-20) on the H1 tree; later commits are docs/evidence only. (iii) V100 jobs were launched by the orchestrator from
  clean worktrees (departure from the SF-35/SF-36 worker-campaign precedent, `ACT`). Other process findings: `BIT` row 2026-10-19T15:50Z.
- CUDA/GCC versions were not recorded in the SF-37 logs.

## Next step

SF-38 (long domain) and SF-39 (macrodispersion study) consume `lambda/h = 32` (the study value; per-amplitude 8 / 16 / 32 are available) and the V4 `e_v` as the representation-divergence measure; SF-39
should also decide, on the owner's reading, whether long-range criteria are statistical only (V5 named no fit) and whether the D8 shadow `lambda/h` of 64 for some estimator series changes the grid used. Human review
of SF-37 is required (SPEC "Human review"), in particular the V3b `discordant` outcome and the O3 `q` disclosure.
