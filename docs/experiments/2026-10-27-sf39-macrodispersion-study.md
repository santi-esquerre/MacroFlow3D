# SF-39: macrodispersion study at `sigma_Y^2` in {1, 2.25, 4} with the Lagrangian consumer

- What this is: the run record and result of increment SF-39
  (`docs/plans/active/lester-eq14/increments/SF-39-macrodispersion-study.md`): `alpha_L`, `alpha_T` on the long domain with
  confidence intervals, refinement ladders, the Gate 5 comparison with Par2 RWPT and the Gate 4 classification of the transverse growth.
- What it is not: an asymptotic statement, a test of the Lester et al. (2023) zero-transverse claim at `t -> infinity`, or a
  statement about other covariance models, media or domains.
- Normative contract: `docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study/preregistration.md` (sha256
  `88fe1fa67eab28a209778cf5a240e270ab4ed1e5f44777c3d0f7cecc89ad7cb7`, fixed in the activation row before any run). Campaign mechanics:
  the artifact `README.md`. Theory: `docs/theory/lester-2023-key-claims.md` (helicity-free regime, the zero-transverse claim and its
  assumption of bounded label fluctuations), `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md` (classical 3D baseline, context only).
- Artifact root `A = docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study/`: `analysis/` (the analysis part output:
  `summary.json`, `decisions.md`/`.csv`, `reduced/`, `series/`, `manifest.txt`), `raw/` (part bookkeeping, Par2 aggregate outputs,
  field statistics, sha256 lists of what stayed on the host), `logs/` (job logs and status files).

## Question

Per `sigma_Y^2` in {1, 2.25, 4}, in smooth Gaussian-covariance log-conductivity fields (`C(r) = sigma^2 exp(-(r/lambda)^2)`) and
purely advective transport traced by the Lagrangian consumer on the long domain:

1. What are `alpha_L` and `alpha_T` (window estimator over `[0.75, 1] tau_b*`; for `alpha_T` also the plane estimator `alpha_TP`), with
   intervals and convergence evidence in `lambda/h`, `h_max` and `N_R`?
2. Does `alpha_L` agree with Par2 RWPT in the same configuration (Gate 5)?
3. Is the transverse growth physical, numerical or unresolved (Gate 4, rule D), without presupposing `alpha_T`?

## Hypothesis

No result was predicted (specification, "Expected numerical effect"), except the pre-registered anticipated outcome of F9: at
`sigma^2 = 4` the verdicts were expected `inconclusive` for power reasons. That outcome was **not** observed: the paired ladder
differences were small enough that every transverse ladder verdict is `converged` at every amplitude.

Hypotheses recorded, not presupposed:

- Lester et al. (2023): `alpha_T = 0` asymptotically in the helicity-free regime, under bounded label fluctuations (an assumption this
  project found to fail in the periodic cell; `docs/theory/lester-2023-key-claims.md`). It concerns `t -> infinity`; this study measures
  a finite travel distance.
- Lester et al. (2023) §5.2, F11: the face-flux representation (Pollock bridge) shows more transverse spreading than the consumer
  (`Delta_P > 0`), shrinking under refinement.
- B&dD (2013): positive `alpha_TA` (0.01 / 0.044 at `sigma^2` 1 / 2.25, Table 2), with Euler stepping on linearly interpolated
  velocities and fixed axes; context only.

## Build / environment

- Increment base `1e40cfe9cb5132174bd167d7f21ba0c77910623e` (merge of PR #82 over PR #81). Every SF-39 run used the one V100 build of
  that base (job `sf39-build`, preset `v100-release`, tree `1e40cfe9cb51`); the five binaries' md5 list (sha256 `b1985f91d49d51f9...`,
  `A/raw/campaign/build/binaries.md5`) is gated at the start of every part (`binaries=ok` in every `PART_BEGIN`).
- Later commits touched scripts and docs only (`git diff --quiet 1e40cfe9 HEAD -- src apps tests cmake CMakeLists.txt CMakePresets.json
  scripts/regress scripts/remote` is empty on the delivery branch). Scripts tree per part (`.mf3d_source_rev` of the mirror):
  checks `dd5799c11454`; pilot and checkpoint `b0be775db0f4`; extend and the first width invocation `6630cc27d3f3`; the second width
  invocation (`s4_w12c`) and analysis `40f5996c2a1e`.
- Host analysis stack: Python 3.11.7, numpy 1.26.4, scipy 1.11.4 (`A/raw/campaign/*/run.env`); the 232 unit tests of the analysis
  package pass on that stack and on Python 3.13 / numpy 2.5 / scipy 1.18.
- One GPU at a time, the least loaded, chosen by `scripts/remote run` (decision `docs/decisions/2026-10-27-remote-single-gpu.md`);
  `REMOTE_GPU_WAIT=7200`, `REMOTE_GPU_MIN_FREE_MIB=30000` for CUDA parts; never `--gpu`, never `CUDA_VISIBLE_DEVICES`.

## Config(s)

Fixed in `preregistration.md` section 1 (arm table) and section 2 (plane lists P1-P7 with sha256); generated per arm by
`python3 -m sf39.geometry flags <arm>`. Common runner flags: `lagrangian_ensemble --medium b --sigma2 1 --eps <sigma> --np-side 100
--times auto --tol 1e-8 --fused-splines --single-launch --twin on --pollock-bridge on` (`--sigma2 1 --eps sigma` = variance `sigma^2`,
`normalize_variance` on), window `[0.1, 0.9]^2`, injection `x0 = 5 lambda`, `N_p = 1e4`, flux weights.

| `sigma^2` | production (`lambda/h`, box) | narrow rungs 1/2, 1, 2 x (section) | coarse twin | Par2 (SF-38 copies) |
|---|---|---|---|---|
| 1 | 2048x256x256 (10; 204.8 x 25.6^2 `lambda`) | 1024x64^2, 2048x128^2, 4096x256^2 (12.8 `lambda`) | 1024x128^2 (5) | `configs/par2_s1[_ext].yaml` |
| 2.25 | 3328x384x384 (16; 208 x 24^2) | 1664x64^2, 3328x128^2, 6656x256^2 (8 `lambda`) | 1664x192^2 (8) | `configs/par2_s225[_ext].yaml` |
| 4 | 3328x384x384 (32; 104 x 12^2) | 1664x64^2, 3328x128^2, 6656x256^2 (4 `lambda`) | 1664x192^2 (16) | `configs/par2_s4[_ext].yaml` |

Seeds: production, rungs and twins from 5001 (directory A `r < 20` with `--hmax-div 8,16,32`, directory B from `5001 + N_A` with
`--hmax-div 8`); width arms 7101 (`s4_w24`) and 7201 (`s4_w12`) plus the continuation `s4_w12c` 7215 (below); Par2 field seeds 9001
(pilot) and 9021 (extension, `transport.seed` 123476789, R15). Final counts (`A/configs/budget.json`, bitácora 2026-10-28):

| `sigma^2` | `N_R` = `N_run` = Par2 `NR` | `N_narrow` | `N_twin` | `N_hmax` = `N_A` |
|---|---|---|---|---|
| 1 | 50 | 20 | 20 | 20 |
| 2.25 | 62 | 20 | 26 | 20 |
| 4 | 71 | 20 (`s4_n16` 30) | 20 (`s4_tw` 30) | 20 |

## Commands

From the remote mirror root, `A=docs/experiments/artifacts/2026-10-27-sf39-macrodispersion-study`:

```bash
scripts/remote --increment SF-39 run sf39-build -- "<clean v100-release build of 1e40cfe, md5 list>"          # RR1, R30
env REMOTE_GPU_WAIT=7200 REMOTE_GPU_MIN_FREE_MIB=30000 scripts/remote --increment SF-39 run sf39-checks -- "bash $A/scripts/run_sf39.sh build/v100-release checks"
env ... run sf39-pilot-{s1,s225,s4,par2} -- "bash $A/scripts/run_sf39.sh build/v100-release pilot <sel>"
REMOTE_GPU_WAIT=7200 scripts/remote --increment SF-39 run sf39-checkpoint -- "bash $A/scripts/run_sf39.sh build/v100-release checkpoint"
# STOP: owner budget (2026-10-28: free use, several days, one GPU at a time) -> f = 1; every N in the bitácora (commit 6630cc27)
env ... run sf39-extend -- "bash $A/scripts/run_sf39.sh build/v100-release extend; bash $A/scripts/run_sf39.sh build/v100-release width"
env ... run sf39-width  -- "bash $A/scripts/run_sf39.sh build/v100-release width"     # s4_w12c after the owner decision
REMOTE_GPU_WAIT=7200 scripts/remote --increment SF-39 run sf39-analysis -- "bash $A/scripts/run_sf39.sh build/v100-release analysis"
```

| part (job) | scripts tree | part seconds | outcome |
|---|---|---|---|
| checks (`sf39-checks`) | `dd5799c1` | 1138 | 173 unit tests OK; GEOMETRY_CHECK OK; regression set v2 REGRESS IDENTICAL; 14/14 solve checks converged; BYTECHECK IDENTICAL (41 items); smoke 28/28 (reducer vs runner moments max rel. 1.02e-13); SIGNTABLE EQUAL |
| pilot (`sf39-pilot-s1/-s225/-s4/-par2`) | `b0be775d` | 4149 / 8661 / 9044 / 4690 | 15 Lagrangian arms 20/20 rc 0; Par2 20/20 series and field-stat lines per amplitude |
| checkpoint (`sf39-checkpoint`) | `b0be775d` | 89 | blinded report (sha256 `2fc104dd...9f71`): projected 6.84 GPU-h, 133.5 GB |
| extend (`sf39-extend`, first half) | `6630cc27` | 23225 | every arm at its count, rc 0; Par2 extension 30/42/51 |
| width (`sf39-extend`, second half) | `6630cc27` | 2201 | `s4_w24` 30/30; `s4_w12` runner exit 4 at seed 7214 after 13 (RR6 halt) |
| width (`sf39-width`) | `40f5996c` | 348 | `s4_w12` ARM_HALTED (by design); `s4_w12c` 17/17 rc 0; part rc 1 (the halted arm) |
| analysis (`sf39-analysis`) | `40f5996c` | 909 | rc 0; `A/analysis/` |

Total recorded part time about 54 400 s (15.1 h) on one shared GPU, plus the base build and the regression references.

## Outputs inspected

### 0. Completeness, statuses and exclusions

Every arm reached its budget count; no realization was skipped except seed 7214 of `s4_w12` (below). Medium (b) solves: all converged
except one `s4_w12` realization accepted through the SF-38 C0 fallback (rel 9.88e-10, balance met) and seed 7214 (not usable). Tracker
statuses at the planes (mean per realization of `1e4` particles): `sigma^2` 1: all `ok`; 2.25: 0.19 `seed_backflow` (status 1: the `x` velocity at
the seed is `<= 0`, so the particle is not integrated); 4: 6.5 `seed_backflow` (0.065 %). No exits through the inlet, no seed fallbacks, no unplaced
particles in any production window. Fraction of `x`-faces with `u <= 0`: 0 / 8.5e-6 / 6.7e-4.

### 1. Production estimates (90 % Student intervals, units of `lambda`; `tau_b*` in `ell / U`)

| `sigma^2` | geometry | `N_R` | `tau_b*` | `U` | `alpha_LAM` | `alpha_TAM` | `alpha_TP` | `alpha_T` fixed axes |
|---|---|---|---|---|---|---|---|---|
| 1 | 2048x256x256, `lambda/h` 10, 204.8 x 25.6 x 25.6 `lambda` | 50 | 125.6 | 1.179 | 1.038 [1.008, 1.068] | 0.009515 [0.007447, 0.01158] | 0.008241 [0.006754, 0.009727] | 0.009502 [0.007435, 0.01157] |
| 2.25 | 3328x384x384, `lambda/h` 16, 208 x 24 x 24 `lambda` | 62 | 93.33 | 1.443 | 3.141 [3.083, 3.198] | 0.02872 [0.02618, 0.03126] | 0.03078 [0.02866, 0.0329] | 0.02873 [0.02618, 0.03127] |
| 4 | 3328x384x384, `lambda/h` 32, 104 x 12 x 12 `lambda` | 71 | 23.22 | 1.873 | 4.654 [4.53, 4.779] | 0.07016 [0.06497, 0.07535] | 0.06665 [0.06219, 0.07111] | 0.07096 [0.06561, 0.0763] |

BCa intervals (`B = 5000`, seed 20261027) beside each Student interval are in `summary.json`; they are close to the Student intervals
(largest shift: `sigma^2 = 1` `alpha_TAM`, BCa [0.00764, 0.01166]).

**B (ensemble size, 95 % half-widths).** Met for every estimator and amplitude except `alpha_TAM` at `sigma^2 = 2.25`: achieved 0.003037
against the target 0.002872 (6 % over). `N_R = 62` came from the pre-registered rule (target met with probability 0.91-0.997 at the
chosen `N`); it is reported, not extended after the fact.

| `sigma^2` | estimator | 95 % half-width | target | met |
|---|---|---|---|---|
| 1 | `alpha_TAM` | 0.002478 | 0.0025 | yes |
| 1 | `alpha_TP` | 0.001782 | 0.0025 | yes |
| 1 | `alpha_LAM` | 0.03591 | 0.05189 | yes |
| 2.25 | `alpha_TAM` | 0.003037 | 0.002872 | **no** |
| 2.25 | `alpha_TP` | 0.002541 | 0.003078 | yes |
| 2.25 | `alpha_LAM` | 0.0693 | 0.157 | yes |
| 4 | `alpha_TAM` | 0.006209 | 0.007016 | yes |
| 4 | `alpha_TP` | 0.005336 | 0.006665 | yes |
| 4 | `alpha_LAM` | 0.1486 | 0.2327 | yes |

### 2. Ladders (A) and the Gate 4 classification (D)

| `sigma^2` | estimator | narrow `D_f` 90 % CI (margin `m`) | narrow | twin `D_tw` 90 % CI (band `2^1.5 m`) | order contrast CI | F4 `h` | `h/16 - h/8` 90 % CI | `h_max` | narrow 1x rung minus production (Welch 90 %) |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `alpha_TAM` | [-0.000168, 3.921e-05] (0.00125) | converged | [0.0001025, 0.00104] (0.003536) | [-0.0005614, 0.000184] | converged | [-1.56e-10, 5.399e-10] | converged | -0.000923 [-0.003431, 0.001585] |
| 1 | `alpha_TP` | [-3.204e-05, 4.022e-05] (0.00125) | converged | [-0.0001701, 0.0001358] (0.003536) | [-0.0001386, 6.623e-05] | converged | [-7.582e-11, 4.057e-11] | converged | -0.002019 [-0.004331, 0.0002933] |
| 1 | `alpha_LAM` | [-0.005712, 0.003959] (0.02435) | converged | [-0.0185, 0.008365] (0.06888) | [-0.02971, -0.002364] | converged | [-5.672e-09, 4.757e-09] | converged | -0.06294 [-0.1421, 0.01618] |
| 2.25 | `alpha_TAM` | [-0.0003983, 0.0007367] (0.00125) | converged | [-0.002636, 0.0005919] (0.003536) | [-0.003115, 0.001204] | converged | [-1.539e-08, 8.207e-09] | converged | -0.006675 [-0.01114, -0.00221] |
| 2.25 | `alpha_TP` | [-0.0003317, 0.0002256] (0.001292) | converged | [-0.0003073, 0.0009325] (0.003655) | [-0.001306, 0.0007084] | converged | [-5.141e-07, 4.754e-07] | converged | -0.00488 [-0.009079, -0.0006816] |
| 2.25 | `alpha_LAM` | [-0.08213, 0.006287] (0.0739) | inconclusive | [-0.1078, 0.03889] (0.209) | [-0.2753, 0.03465] | inconclusive | [-3.597e-07, 6.993e-08] | converged | -0.1465 [-0.3494, 0.05643] |
| 4 | `alpha_TAM` | [-0.000318, 0.0001182] (0.001467) | converged | [0.0004761, 0.003701] (0.004149) | [-0.001507, -0.0001742] | converged | [-3.386e-10, 2.332e-10] | converged | -0.04072 [-0.05114, -0.0303] |
| 4 | `alpha_TP` | [-0.0003326, 4.594e-05] (0.00125) | converged | [-0.001058, 0.001787] (0.003536) | [-0.001327, 4.062e-05] | converged | [-7.273e-08, 2.65e-09] | converged | -0.04487 [-0.05048, -0.03927] |
| 4 | `alpha_LAM` | [-0.03014, 0.0007384] (0.09601) | converged | [-0.0199, 0.08741] (0.2715) | [-0.01456, 0.05784] | converged | [1.822e-09, 1.182e-08] | converged | -0.7995 [-1.232, -0.3672] |

`D_f` = 2x rung minus production rung (narrow section), `D_tw` = production minus coarse twin (production domain), both paired by seed,
finer minus coarser. No Richardson correction is applied (`|Delta Q|` is the bound where the order condition fails; `summary.json`).

**Rule D result (both transverse estimators, every amplitude): `physical (positive)`, branch 4** — both ladder verdicts (`h` by F4 and
`h_max`) are `converged`, so branches 1-3 cannot fire, and the production 90 % lower limit is above 0. Amplitude label: `physical
(positive)` at `sigma^2` 1, 2.25 and 4 (identical label of `alpha_TAM` and `alpha_TP`).

The orchestrator recomputed the production intervals, `D_f`, `D_tw`, the order contrast, the `h_max` differences, the margins, the
verdicts and the branch from `A/analysis/reduced/*.csv` with an independent script (no `sf39` import): every value above is reproduced.

What the label means and does not mean:

- It is a statement **within this model, medium and domain, at the measured distance**: Gaussian-covariance `ln K` from the SF-18
  normalized periodic generator, medium (b) (`solve_head`, Dirichlet in `x`, periodic `y`/`z`), the consumer's DP5(4) traces on the C^2
  spline flow, the transverse widths of the table above, the window `[0.75, 1] tau_b*` before the first `ok` arrival at `L1 - 5 lambda`.
- The transverse growth does not vanish or decrease under a 2x refinement of `lambda/h` (narrow sections and the production-domain twin)
  or of `h_max`; at `sigma^2` 1 and 4 the twin difference `D_tw` of `alpha_TAM` is even positive (finer above coarser: +0.0006 = 6 %
  and +0.0021 = 3 %).
- The `h_max` ladder is the integration-error ladder (decision 2026-10-23 D15: the `tol` ladder is vacuous at the `h/8` cap, SF-37
  RR15, where `h_max` is the binding step limiter). Its differences are 1e-10..1e-7, i.e. the integration error is negligible at `h/8`.
- Checks that do not share the consumer's representation give the same transverse growth: the Pollock bridge on the FV face fluxes
  (F11: `Delta_P` includes 0 at production for every estimator and amplitude) and the `exp(-I)` weights that compensate the
  representation's divergence (change below 0.3 %; section 5). The growth is therefore a property of the discrete Darcy flow that both
  representations share, and it is stable under the flow's own refinement.
- It is not asymptotic: F7 (`Delta_half`, second minus first half of the window) is not shown decreasing at any amplitude, so the
  approach to an asymptote is not characterized. It does not test Lester et al.'s `t -> infinity` claim.

**Transverse-domain effect (stated with the classification, specification D).** The narrow sections are 12.8 / 8 / 4 `lambda` wide. The
narrow 1x rung is below the production domain by 0.0009 (not resolved) / 0.0067 (23 %) / 0.041 (58 %) in `alpha_TAM`, and at
`sigma^2 = 4` the narrow section is a different flow regime (`U` 1.45 against 1.87). F4's stated assumption ("the production-domain error
is represented by the narrow ladder or by `|D_tw| / 2^1.5`") therefore rests, at `sigma^2` 2.25 and especially 4, on the coarse twin,
which is the only `lambda/h` evidence on the production domain: `D_tw` inside `+-2^1.5 m` for both estimators (at `sigma^2 = 4`:
`alpha_TAM` +3.0 %, `alpha_TP` +0.5 % from `lambda/h` 16 to 32). The width check (F10, section 4) shows `alpha_T` at `sigma^2 = 4` still
depends on the width between 4 and 12 `lambda`; between 12 and 24 `lambda` it is unresolved at `+-10 %`.

### 3. Gate 5 (C): `alpha_LAM` of the Lagrangian twin against Par2 RWPT

| `sigma^2` | `tau_c` | window `[0.75, 1] tau_c` (box time) | Lagrangian twin `alpha_LAM` (`U_c`) | Par2 `alpha_x` (`U_c`) | difference 90 % CI (Welch) | margin | verdict | FV `qbar` | ANCOVA | Par2 `alpha_T` fixed (descriptive) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 124.5 | [3.086, 4.114] | 1.038 [1.006, 1.07] (1.182) | 1.138 [1.105, 1.171] (0.01157) | -0.1001 [-0.146, -0.05427] | 0.1138 | inconclusive | inconclusive | inconclusive | 0.006752 [0.004572, 0.008932] |
| 2.25 | 92.95 | [2.01, 2.68] | 3.165 [3.103, 3.226] (1.445) | 3.208 [3.15, 3.266] (0.01418) | -0.04282 [-0.1268, 0.04112] | 0.3208 | agree | agree | agree | 0.02541 [0.02322, 0.0276] |
| 4 | 23.75 | [0.7847, 1.046] | 5.614 [5.409, 5.82] (1.892) | 5.681 [5.525, 5.836] (0.0188) | -0.06639 [-0.3226, 0.1898] | 0.5681 | agree | agree | agree | 0.05721 [0.05198, 0.06243] |

(`U_c` in each arm's own length and time units; both arms use the same normalized window in `tau = U_c t / lambda`. Par2's in-pipeline
`alpha` is not used.)

- `agree` at `sigma^2` 2.25 and 4 with both sensitivities. At `sigma^2 = 4` the comparison carries the pre-asymptotic caveat (D3: 104
  `lambda`; B&dD's longitudinal fit reaches its asymptote at about 20.5 advection times).
- `inconclusive` at `sigma^2 = 1`: the Lagrangian `alpha_LAM` is 8.8 % below Par2, the CI `[-0.146, -0.054]` straddles the lower margin
  `-0.114`; the ANCOVA on Par2's realized `sigma^2` (adjusted Par2 mean 1.139) and the FV-`qbar` normalization do not change it. An
  inconclusive verdict needs no attribution under C (only `disagree` does); the stated differences that could contribute are listed
  below.
- Comparison statement (Gate 5): covariance model Gaussian `C(r) = sigma^2 exp(-(r/lambda)^2)` in both arms; `sigma^2` 1 / 2.25 / 4
  (Lagrangian: SF-18 periodic generator, `normalize_variance` on, realized variance exactly `sigma^2`; Par2: random-spectral,
  non-periodic, not normalized, realized `sigma^2` recorded per realization in `raw/field_stats_par2_*.txt` and used in the ANCOVA);
  boundary conditions: head-driven, Dirichlet in `x`, periodic `y`/`z` in both (Par2's non-periodic field makes a seam at the periodic
  faces); same physical domains (`lambda/h` 10 / 12 / 24 for Par2); injection: Lagrangian window `[0.1, 0.9]^2` of the plane `x = 5
  lambda`, 100 x 100 lattice, uniform weights for the twin series used here; Par2 uniform over the full plane at `x = 5 lambda`; tracking:
  DP5(4) on the C^2 spline flow (`tol` 1e-8, `h_max = h/8`) against Par2's explicit Euler stepping (`dt/2` sensitivity of SF-38 E),
  purely advective in both; estimation: `alpha = [Var(t_hi) - Var(t_lo)] / (2 lambda U_c (t_hi - t_lo))` in the common normalized window
  `[0.75, 1] tau_c`, `tau = U_c t / lambda`, `U_c` the centroid slope of each arm, `tau_c = min(tau_b*(twin), tau_b*(Par2))`, Welch
  interval, margin `+-10 %` of the Par2 mean.

### 4. Estimation and exploratory items

- **F7 `Delta_half`** (`alpha_TP`): not shown decreasing at any amplitude (90 % CIs `[-0.0064, 0.0012]`, `[-0.0067, 0.0034]`,
  `[-0.0086, 0.0068]`).
- **F8 `rho(X_b)`** (pathwise drift between the `h/8` and `h/16` passes): `pathwise resolved` at `sigma^2` 1 and 4; at 2.25 none of the
  three readings applies (median 1.4e-4, p95 7.3e-4 against the margin 1.5e-3).
- **F11 method contrast** (`Delta_P = alpha(Pollock) - alpha(consumer)`, uniform-weight series): at production `Delta_P` includes 0 for
  `alpha_TAM`, `alpha_TP` and the fixed-axis `alpha_T` at every amplitude (largest magnitude 2.3e-4). On the coarse settings at `sigma^2 =
  4`, `Delta_P > 0` for `alpha_TAM` (narrow 1/2 rung +5.7e-4, twin +1.7e-3) and it shrinks under refinement (difference in differences
  above 0), as Lester et al. §5.2 hypothesize, at 1-2 % of `alpha_TAM`; `alpha_TP` does not show it, and at `sigma^2 = 1` `Delta_P` of
  `alpha_TP` is slightly below 0 on the narrow 1/2, 1x and twin settings. Estimation only.
- **F10 width check** (`sigma^2 = 4`, `lambda/h = 16`, 30 realizations per arm, `+-10 %` of the wider arm):

  | comparison | estimator | wide | reference | difference 90 % CI (Welch) | margin | reading |
  |---|---|---|---|---|---|---|
  | 24 vs 12 `lambda` (`s4_w24` vs first 30 of `s4_tw`; `U` 1.884 / 1.875) | `alpha_TAM` | 0.06866 | 0.07532 | [-0.01908, 0.00576] | 0.00687 | inconclusive |
  | | `alpha_TP` | 0.06767 | 0.06417 | [-0.00698, 0.01398] | 0.00677 | inconclusive |
  | 12 vs 4 `lambda` (`s4_w12`+`s4_w12c` vs first 30 of `s4_n16`; `U` 1.874 / 1.457) | `alpha_TAM` | 0.06352 | 0.02914 | [0.02155, 0.04720] | 0.00635 | disagree |
  | | `alpha_TP` | 0.06780 | 0.02142 | [0.03866, 0.05410] | 0.00678 | disagree |

  Exploratory, non-gating. The 12-`lambda` arm (ii) uses seeds 7201-7213 and 7215-7231: seed 7214 is excluded because its medium (b)
  solve did not converge (runner exit 4, 2000 iterations, rel 3.4e-7); owner decision of 2026-10-28 (bitácora), disclosed in
  `summary.json` (`F10.w12_vs_n16.excluded_seeds`). This is a selection of one field out of 31.
- **Fixed-axis `alpha_T` against B&dD** (one-sided 95 % upper limits, hypothesis-generating): `sigma^2 = 1`: 0.01157, not below B&dD's
  0.01; `sigma^2 = 2.25`: 0.03127, below B&dD's 0.044. Conditions differ: B&dD use fixed axes, a 5 `lambda` injection buffer, `N_R = 500`,
  first-order Euler stepping on linearly interpolated velocities (`docs/theory/beaudoin-de-dreuzy-2013-key-claims.md`), and domains,
  generator and boundary conditions of their own. Context: B&dD Table 1 longitudinal fit `alpha_LAF` 1.27 / 4.35 / 12.7 at `sigma^2` 1 /
  2.25 / 4 (asymptotic, `t_LF` 2.21 / 4.83 / 20.5); the window values here (1.04 / 3.14 / 4.65) are pre-asymptotic, increasingly so with
  `sigma^2`. B&dD eq. 7 (`alpha_TAM ~ sigma^4`) would give ratios 5.1 and 16 between the amplitudes; the measured `alpha_TAM` ratios are
  3.0 and 7.4.

### 5. Recorded context (E)

| `sigma^2` | plane statuses (mean per realization of 1e4) | `e_v` median of p50 / p99 / p100 | `x`-faces `u <= 0` (mean) | tilt mean (rad) | drift tilt mean | split-half fraction TAM / TP / LAM | `alpha_TAM` flux / `exp(-I)` / Pollock / uniform | `alpha_TP` flux / `exp(-I)` / Pollock / uniform |
|---|---|---|---|---|---|---|---|---|
| 1 | ok 1e+04, status 1 0 | 0.05524 / 0.2194 / 0.3769 | 0 | 0.001735 | 0.003432 | 0.01166 / 0.003746 / 0.1095 | 0.009515 / 0.009539 / 0.009577 / 0.00919 | 0.008241 / 0.008256 / 0.008244 / 0.007938 |
| 2.25 | ok 1e+04, status 1 0.1935 | 0.05019 / 0.1988 / 0.3463 | 8.539e-06 | 0.003745 | 0.006738 | 0.08309 / 0.02366 / 0.3356 | 0.02872 / 0.02873 / 0.02835 / 0.02806 | 0.03078 / 0.03075 / 0.03006 / 0.03048 |
| 4 | ok 9994, status 1 6.465 | 0.01631 / 0.06499 / 0.107 | 0.0006681 | 0.01607 | 0.0236 | 0.0896 / 0.0229 / 0.1497 | 0.07016 / 0.0701 / 0.0694 / 0.06449 | 0.06665 / 0.06665 / 0.06694 / 0.06717 |

- The Lagrangian `e_v = 1 - exp(-int div v_s dt)` is not small at `lambda/h` 10 and 16 (median per-realization p50 5.5 % and 5.0 %) and
  drops at `lambda/h = 32` (1.6 %); `alpha_T` with the compensating `exp(-I)` weights differs from the flux-weighted value by less than
  0.3 % at every amplitude, and the `lambda/h` ladders do not move `alpha_T` while `e_v` changes, so the divergence of `v_s` is not what
  produces the transverse growth.
- Recorded per realization in `A/analysis/reduced/<set>.csv`: statuses and exits by type, excluded fractions, `active` counts, `t_b,r`,
  `e_v` percentiles, `flow_stats.json` fields, realized/applied variance, tilt, the fixed-minus-aligned difference, `alpha_TD` in the
  plume-drift frame, the split-half components; per ladder the distribution of the paired differences and the realization with the
  largest one (`summary.json`, `trend`/`dist`); `sigma_T^2(x1)` (`reduced/sigma_series_*.csv`); `rho` rows (`reduced/rho_*.csv`);
  `alpha_L(t)`, `alpha_T(t)` series (`series/alpha_t_*.csv`, flux, uniform, Pollock, Pollock-uniform; B&dD eq. 4 convention).

## Conclusion

Within this model, medium and domain, at the measured travel distance:

- `alpha_L` (fixed axes, window `[0.75, 1] tau_b*`): 1.038 [1.008, 1.068], 3.141 [3.083, 3.198], 4.654 [4.530, 4.779] `lambda` at
  `sigma^2` 1, 2.25, 4 (90 %); converged in `lambda/h` and `h_max` at 1 and 4, inconclusive in `lambda/h` at 2.25 (`h_max` converged).
- `alpha_T`: `alpha_TAM` 0.0095 [0.0074, 0.0116], 0.0287 [0.0262, 0.0313], 0.0702 [0.0650, 0.0754]; `alpha_TP` 0.0082 [0.0068, 0.0097],
  0.0308 [0.0287, 0.0329], 0.0666 [0.0622, 0.0711] `lambda`. Gate 4 classification by rule D: **`physical (positive)` at every
  amplitude, both estimators**, with the transverse-domain qualification above (at `sigma^2 = 4` the value depends on the transverse
  width up to at least 12 `lambda`; whether 12 `lambda` is width-converged is unresolved).
- Gate 5: `agree` at `sigma^2` 2.25 and 4, `inconclusive` at 1 (Lagrangian 8.8 % below Par2).
- The pre-registered anticipated outcome at `sigma^2 = 4` (inconclusive) did not occur.

The classification follows the pre-registered refinement protocol and independent comparators (Pollock bridge, `exp(-I)` weights,
`h_max` passes); it is not an automatic reading of a positive `alpha_T` (Gate 4, AGENTS.md hard rule). It is a finite-distance result: it
neither confirms nor refutes the asymptotic zero-transverse claim of Lester et al. (2023), whose premise (bounded label fluctuations) is
an assumption the project does not presuppose.

## Caveats

- Finite distance and pre-asymptotic longitudinal regime, increasingly with `sigma^2` (D3); F7 cannot show a decrease.
- At `sigma^2 = 4` (and less so at 2.25) the narrow-section ladder measures a domain-restricted object; the production-domain `lambda/h`
  evidence is the coarse twin (one factor of 2, band `+-2^1.5 m`). The width dependence up to 12 `lambda` and the unresolved 12-to-24
  comparison qualify the `sigma^2 = 4` value.
- B not met for `alpha_TAM` at `sigma^2 = 2.25` (6 % over the half-width target).
- One field (seed 7214) of the exploratory width arm (ii) was excluded after a non-converged solve (owner decision, disclosed); the
  primary arms had no failure.
- Gate 5 at `sigma^2 = 1` is inconclusive; the arms differ in generator (periodic normalized vs random-spectral not normalized), in
  stepping (DP5(4) on the spline vs Euler) and in injection (window vs full plane).
- Single model: Gaussian covariance, scalar isotropic `K`, medium (b), purely advective; nothing here extends to exponential covariance,
  anisotropic tensors, finite Péclet or `sigma^2 > 4` (6.25 deferred, F12).

## Deviations recorded for the human review

- R15: the Par2 extension advances `transport.seed` by `1000 x 20` with the field seed (otherwise the injection positions of the pilot
  would repeat).
- R27: Par2 outputs live outside the mirror (`$SF39_OUT_ROOT/par2`); aggregate CSVs imported to `A/raw/par2_s{c}[_ext]`.
- R30: the base build ran as a detached `run` job with the regress `TMPDIR` step, not the specification's `exec` line.
- Rule D branch 3 is applied as written (no "other's point estimate" clause); it did not fire.
- `s4_w12c`: continuation arm of the halted `s4_w12` (RR6), owner decision 2026-10-28; the preregistration is unchanged.
- `extend` and the first `width` ran in one detached job (`sf39-extend`), with the commands and order of the specification's two jobs.
- `A/analysis/manifest.txt` also lists three bookkeeping files that `run_sf39.sh` completes after the manifest is written
  (`logs/analysis.log`, `logs/analysis.time`, `results.txt`); their sha256 there are of the empty/partial files, so `sha256sum -c` reports
  them FAILED by construction. Every analysis output (`summary.json`, `decisions.*`, `reduced/`, `series/`) verifies.

## Decision points (preregistration section 10; `A/analysis/decisions.md` verbatim, headings one level down)


### primary

| amplitude | estimator | decision point | verdict | CI | n | note |
|---|---|---|---|---|---|---|
| s1 | alpha_TAM | Gate 4 label (D) | physical (positive) | [0.00744747, 0.0115829] | 50 | branch 4; ladder=None difference=None; h=converged hmax=converged; production mean=0.00951516 |
| s1 | alpha_TP | Gate 4 label (D) | physical (positive) | [0.0067542, 0.00972685] | 50 | branch 4; ladder=None difference=None; h=converged hmax=converged; production mean=0.00824052 |
| s1 | alpha_TAM+alpha_TP | Gate 4 amplitude label (D) | physical (positive) |  | 50 | alpha_TAM: physical (positive); alpha_TP: physical (positive) |
| s1 | alpha_LAM | Gate 5 (C) Lagrangian twin vs Par2 | inconclusive | [-0.145973, -0.0542721] | 50/50 | diff=-0.100123 margin=0.113816; late Par2 beyond pilot 2/30 |
| s1 | alpha_LAM | Gate 5 sensitivity FV qbar | inconclusive | [-0.144256, -0.0522624] | 50/50 | diff=-0.098259 margin=0.113816 |
| s1 | alpha_LAM | Gate 5 sensitivity ANCOVA | inconclusive | [-0.14673, -0.0546057] | 50/50 | diff=-0.100668 margin=0.11387 adjusted Par2 mean=1.1387 |
| s225 | alpha_TAM | Gate 4 label (D) | physical (positive) | [0.0261837, 0.0312567] | 62 | branch 4; ladder=None difference=None; h=converged hmax=converged; production mean=0.0287202 |
| s225 | alpha_TP | Gate 4 label (D) | physical (positive) | [0.0286589, 0.0329041] | 62 | branch 4; ladder=None difference=None; h=converged hmax=converged; production mean=0.0307815 |
| s225 | alpha_TAM+alpha_TP | Gate 4 amplitude label (D) | physical (positive) |  | 62 | alpha_TAM: physical (positive); alpha_TP: physical (positive) |
| s225 | alpha_LAM | Gate 5 (C) Lagrangian twin vs Par2 | agree | [-0.126763, 0.0411191] | 62/62 | diff=-0.0428221 margin=0.320774; late Par2 beyond pilot 1/42 |
| s225 | alpha_LAM | Gate 5 sensitivity FV qbar | agree | [-0.121554, 0.0464275] | 62/62 | diff=-0.0375634 margin=0.320774 |
| s225 | alpha_LAM | Gate 5 sensitivity ANCOVA | agree | [-0.122403, 0.042396] | 62/62 | diff=-0.0400036 margin=0.320492 adjusted Par2 mean=3.20492 |
| s4 | alpha_TAM | Gate 4 label (D) | physical (positive) | [0.0649707, 0.0753498] | 71 | branch 4; ladder=None difference=None; h=converged hmax=converged; production mean=0.0701602 |
| s4 | alpha_TP | Gate 4 label (D) | physical (positive) | [0.0621877, 0.0711073] | 71 | branch 4; ladder=None difference=None; h=converged hmax=converged; production mean=0.0666475 |
| s4 | alpha_TAM+alpha_TP | Gate 4 amplitude label (D) | physical (positive) |  | 71 | alpha_TAM: physical (positive); alpha_TP: physical (positive) |
| s4 | alpha_LAM | Gate 5 (C) Lagrangian twin vs Par2 | agree | [-0.322621, 0.189846] | 71/71 | diff=-0.0663872 margin=0.568072; late Par2 beyond pilot 1/51 |
| s4 | alpha_LAM | Gate 5 sensitivity FV qbar | agree | [-0.288881, 0.223931] | 71/71 | diff=-0.0324749 margin=0.568072 |
| s4 | alpha_LAM | Gate 5 sensitivity ANCOVA | agree | [-0.323369, 0.188555] | 71/71 | diff=-0.067407 margin=0.568174 adjusted Par2 mean=5.68174 |

### key secondary

| amplitude | estimator | decision point | verdict | CI | n | note |
|---|---|---|---|---|---|---|
| s1 | alpha_TAM | B half-width (95 %) | met | [0.00703675, 0.0119936] | 50 | halfwidth=0.00247841 target=0.0025 |
| s1 | alpha_TAM | A h narrow verdict (D_f) | converged | [-0.000167966, 3.92091e-05] | 20 | m=0.00125 mean D_f=-6.43787e-05 |
| s1 | alpha_TAM | A h production verdict (F4) | converged | [0.000102543, 0.00103975] | 20 | CI of D_tw vs +-2^1.5 m=0.00353553 (inside=True) |
| s1 | alpha_TAM | A h order contrast | monotone order >= 1.5 not rejected | [-0.000561402, 0.000183979] | 20 | CI of s(2^1.5 D_f - D_c), s=+1 |
| s1 | alpha_TAM | A h_max verdict (h/16 - h/8) | converged | [-1.56015e-10, 5.39868e-10] | 20 | m=0.00125 mean=1.91926e-10 |
| s1 | alpha_TP | B half-width (95 %) | met | [0.00645896, 0.0100221] | 50 | halfwidth=0.00178156 target=0.0025 |
| s1 | alpha_TP | A h narrow verdict (D_f) | converged | [-3.20374e-05, 4.02191e-05] | 20 | m=0.00125 mean D_f=4.09086e-06 |
| s1 | alpha_TP | A h production verdict (F4) | converged | [-0.000170128, 0.00013577] | 20 | CI of D_tw vs +-2^1.5 m=0.00353553 (inside=True) |
| s1 | alpha_TP | A h order contrast | monotone order >= 1.5 not rejected | [-0.000138618, 6.62284e-05] | 20 | CI of s(2^1.5 D_f - D_c), s=-1 |
| s1 | alpha_TP | A h_max verdict (h/16 - h/8) | converged | [-7.58195e-11, 4.05719e-11] | 20 | m=0.00125 mean=-1.76238e-11 |
| s1 | alpha_LAM | B half-width (95 %) | met | [1.00195, 1.07378] | 50 | halfwidth=0.0359121 target=0.0518932 |
| s1 | alpha_LAM | A h narrow verdict (D_f) | converged | [-0.00571214, 0.00395925] | 20 | m=0.0243512 mean D_f=-0.000876446 |
| s1 | alpha_LAM | A h production verdict (F4) | converged | [-0.0185043, 0.00836465] | 20 | CI of D_tw vs +-2^1.5 m=0.0688756 (inside=True) |
| s1 | alpha_LAM | A h order contrast | monotone order >= 1.5 not rejected | [-0.0297057, -0.00236421] | 20 | CI of s(2^1.5 D_f - D_c), s=-1 |
| s1 | alpha_LAM | A h_max verdict (h/16 - h/8) | converged | [-5.67193e-09, 4.75714e-09] | 20 | m=0.0263506 mean=-4.57394e-10 |
| s225 | alpha_TAM | B half-width (95 %) | not met | [0.0256835, 0.0317569] | 62 | halfwidth=0.00303673 target=0.00287202 |
| s225 | alpha_TAM | A h narrow verdict (D_f) | converged | [-0.000398307, 0.000736688] | 20 | m=0.00125 mean D_f=0.00016919 |
| s225 | alpha_TAM | A h production verdict (F4) | converged | [-0.00263635, 0.000591854] | 26 | CI of D_tw vs +-2^1.5 m=0.00353553 (inside=True) |
| s225 | alpha_TAM | A h order contrast | monotone order >= 1.5 not rejected | [-0.00311542, 0.00120373] | 20 | CI of s(2^1.5 D_f - D_c), s=-1 |
| s225 | alpha_TAM | A h_max verdict (h/16 - h/8) | converged | [-1.53901e-08, 8.20671e-09] | 20 | m=0.00142575 mean=-3.59169e-09 |
| s225 | alpha_TP | B half-width (95 %) | met | [0.0282403, 0.0333227] | 62 | halfwidth=0.00254121 target=0.00307815 |
| s225 | alpha_TP | A h narrow verdict (D_f) | converged | [-0.000331679, 0.000225646] | 20 | m=0.00129241 mean D_f=-5.30166e-05 |
| s225 | alpha_TP | A h production verdict (F4) | converged | [-0.000307273, 0.000932522] | 26 | CI of D_tw vs +-2^1.5 m=0.00365547 (inside=True) |
| s225 | alpha_TP | A h order contrast | monotone order >= 1.5 not rejected | [-0.00130572, 0.000708356] | 20 | CI of s(2^1.5 D_f - D_c), s=-1 |
| s225 | alpha_TP | A h_max verdict (h/16 - h/8) | converged | [-5.14142e-07, 4.75373e-07] | 20 | m=0.00147528 mean=-1.93845e-08 |
| s225 | alpha_LAM | B half-width (95 %) | met | [3.07129, 3.2099] | 62 | halfwidth=0.0693025 target=0.15703 |
| s225 | alpha_LAM | A h narrow verdict (D_f) | inconclusive | [-0.0821292, 0.0062874] | 20 | m=0.0739047 mean D_f=-0.0379209 |
| s225 | alpha_LAM | A h production verdict (F4) | inconclusive | [-0.107786, 0.0388941] | 26 | CI of D_tw vs +-2^1.5 m=0.209034 (inside=True) |
| s225 | alpha_LAM | A h order contrast | monotone order >= 1.5 not rejected | [-0.275268, 0.0346545] | 20 | CI of s(2^1.5 D_f - D_c), s=+1 |
| s225 | alpha_LAM | A h_max verdict (h/16 - h/8) | converged | [-3.5973e-07, 6.99308e-08] | 20 | m=0.0785595 mean=-1.449e-07 |
| s4 | alpha_TAM | B half-width (95 %) | met | [0.063951, 0.0763695] | 71 | halfwidth=0.00620926 target=0.00701602 |
| s4 | alpha_TAM | A h narrow verdict (D_f) | converged | [-0.000318004, 0.000118186] | 20 | m=0.00146704 mean D_f=-9.99087e-05 |
| s4 | alpha_TAM | A h production verdict (F4) | converged | [0.000476139, 0.00370095] | 20 | CI of D_tw vs +-2^1.5 m=0.00414942 (inside=True) |
| s4 | alpha_TAM | A h order contrast | monotone order >= 1.5 not rejected | [-0.00150686, -0.00017419] | 20 | CI of s(2^1.5 D_f - D_c), s=+1 |
| s4 | alpha_TAM | A h_max verdict (h/16 - h/8) | converged | [-3.38615e-10, 2.33162e-10] | 20 | m=0.00374365 mean=-5.27268e-11 |
| s4 | alpha_TP | B half-width (95 %) | met | [0.0613115, 0.0719836] | 71 | halfwidth=0.00533605 target=0.00666475 |
| s4 | alpha_TP | A h narrow verdict (D_f) | converged | [-0.00033262, 4.5937e-05] | 20 | m=0.00125 mean D_f=-0.000143342 |
| s4 | alpha_TP | A h production verdict (F4) | converged | [-0.00105794, 0.00178742] | 20 | CI of D_tw vs +-2^1.5 m=0.00353553 (inside=True) |
| s4 | alpha_TP | A h order contrast | monotone order >= 1.5 not rejected | [-0.00132743, 4.06242e-05] | 20 | CI of s(2^1.5 D_f - D_c), s=+1 |
| s4 | alpha_TP | A h_max verdict (h/16 - h/8) | converged | [-7.27348e-08, 2.65007e-09] | 20 | m=0.00320762 mean=-3.50424e-08 |
| s4 | alpha_LAM | B half-width (95 %) | met | [4.50578, 4.80307] | 71 | halfwidth=0.148649 target=0.232721 |
| s4 | alpha_LAM | A h narrow verdict (D_f) | converged | [-0.0301356, 0.000738439] | 20 | m=0.0960066 mean D_f=-0.0146986 |
| s4 | alpha_LAM | A h production verdict (F4) | converged | [-0.0199022, 0.0874097] | 20 | CI of D_tw vs +-2^1.5 m=0.271548 (inside=True) |
| s4 | alpha_LAM | A h order contrast | monotone order >= 1.5 not rejected | [-0.0145633, 0.0578378] | 20 | CI of s(2^1.5 D_f - D_c), s=-1 |
| s4 | alpha_LAM | A h_max verdict (h/16 - h/8) | converged | [1.82155e-09, 1.18161e-08] | 20 | m=0.120875 mean=6.81882e-09 |

### estimation

| amplitude | estimator | decision point | verdict | CI | n | note |
|---|---|---|---|---|---|---|
| s1 | alpha_TP | F7 Delta_half | not shown decreasing | [-0.00643519, 0.00115292] | 50 | mean Delta_half=-0.00264114 |
| s1 | rho(X_b), pair 8-16 | F8 readings | pathwise resolved |  | 20 | median=5.37004e-08 p95=8.31056e-08 (m=0.00125, S=0.387342) |
| s1 | alpha_TAM | F11 Delta_P production | Delta_P includes 0 | [-0.000186375, 0.000102516] | 50 | mean Delta_P=-4.19296e-05 |
| s1 | alpha_TP | F11 Delta_P production | Delta_P includes 0 | [-7.83217e-05, 2.6016e-05] | 50 | mean Delta_P=-2.61528e-05 |
| s1 | alpha_Tfixed | F11 Delta_P production | Delta_P includes 0 | [-0.000187466, 0.000101399] | 50 | mean Delta_P=-4.30339e-05 |
| s1 | alpha_TAM | F11 Delta_P narrow_half | Delta_P includes 0 | [-0.000435164, 0.000157469] | 20 | mean Delta_P=-0.000138847 |
| s1 | alpha_TP | F11 Delta_P narrow_half | Delta_P below 0 | [-0.000426843, -3.18243e-05] | 20 | mean Delta_P=-0.000229334 |
| s1 | alpha_Tfixed | F11 Delta_P narrow_half | Delta_P includes 0 | [-0.000447737, 0.000152104] | 20 | mean Delta_P=-0.000147817 |
| s1 | alpha_TAM | F11 Delta_P narrow_1x | Delta_P includes 0 | [-0.000185368, 6.27616e-05] | 20 | mean Delta_P=-6.1303e-05 |
| s1 | alpha_TP | F11 Delta_P narrow_1x | Delta_P below 0 | [-0.000127339, -1.88507e-05] | 20 | mean Delta_P=-7.3095e-05 |
| s1 | alpha_Tfixed | F11 Delta_P narrow_1x | Delta_P includes 0 | [-0.00018534, 6.20825e-05] | 20 | mean Delta_P=-6.16285e-05 |
| s1 | alpha_TAM | F11 Delta_P narrow_2x | Delta_P includes 0 | [-7.80215e-05, 8.67154e-05] | 20 | mean Delta_P=4.34694e-06 |
| s1 | alpha_TP | F11 Delta_P narrow_2x | Delta_P includes 0 | [-3.04902e-05, 2.60078e-06] | 20 | mean Delta_P=-1.39447e-05 |
| s1 | alpha_Tfixed | F11 Delta_P narrow_2x | Delta_P includes 0 | [-7.82868e-05, 8.68736e-05] | 20 | mean Delta_P=4.29338e-06 |
| s1 | alpha_TAM | F11 Delta_P twin | Delta_P includes 0 | [-0.00035312, 0.000504434] | 20 | mean Delta_P=7.56568e-05 |
| s1 | alpha_TP | F11 Delta_P twin | Delta_P below 0 | [-0.00039058, -9.42999e-08] | 20 | mean Delta_P=-0.000195337 |
| s1 | alpha_Tfixed | F11 Delta_P twin | Delta_P includes 0 | [-0.00035364, 0.000504917] | 20 | mean Delta_P=7.56389e-05 |
| s1 | alpha_TAM | F11 difference in differences narrow_half_minus_1x | difference includes 0 | [-0.000398082, 0.000242993] | 20 | mean=-7.75444e-05 |
| s1 | alpha_TP | F11 difference in differences narrow_half_minus_1x | difference includes 0 | [-0.000314335, 1.85778e-06] | 20 | mean=-0.000156239 |
| s1 | alpha_Tfixed | F11 difference in differences narrow_half_minus_1x | difference includes 0 | [-0.000409852, 0.000237476] | 20 | mean=-8.6188e-05 |
| s1 | alpha_TAM | F11 difference in differences narrow_1x_minus_2x | difference includes 0 | [-0.000202396, 7.10961e-05] | 20 | mean=-6.565e-05 |
| s1 | alpha_TP | F11 difference in differences narrow_1x_minus_2x | difference below 0 | [-0.000107962, -1.03384e-05] | 20 | mean=-5.91503e-05 |
| s1 | alpha_Tfixed | F11 difference in differences narrow_1x_minus_2x | difference includes 0 | [-0.000201836, 6.99926e-05] | 20 | mean=-6.59219e-05 |
| s1 | alpha_TAM | F11 difference in differences twin_minus_production | difference includes 0 | [-0.000273447, 0.000652481] | 20 | mean=0.000189517 |
| s1 | alpha_TP | F11 difference in differences twin_minus_production | difference below 0 | [-0.000385762, -5.22942e-05] | 20 | mean=-0.000219028 |
| s1 | alpha_Tfixed | F11 difference in differences twin_minus_production | difference includes 0 | [-0.000272033, 0.00065633] | 20 | mean=0.000192149 |
| s225 | alpha_TP | F7 Delta_half | not shown decreasing | [-0.00671303, 0.00336172] | 62 | mean Delta_half=-0.00167565 |
| s225 | rho(X_b), pair 8-16 | F8 readings | intermediate (none of the three readings) |  | 20 | median=0.000141543 p95=0.000725131 (m=0.00147528, S=0.428558) |
| s225 | alpha_TAM | F11 Delta_P production | Delta_P includes 0 | [-0.000424675, 0.000340994] | 62 | mean Delta_P=-4.18404e-05 |
| s225 | alpha_TP | F11 Delta_P production | Delta_P includes 0 | [-0.000332789, 9.53276e-05] | 62 | mean Delta_P=-0.000118731 |
| s225 | alpha_Tfixed | F11 Delta_P production | Delta_P includes 0 | [-0.000434931, 0.000335817] | 62 | mean Delta_P=-4.95568e-05 |
| s225 | alpha_TAM | F11 Delta_P narrow_half | Delta_P includes 0 | [-0.00128453, 0.000303805] | 20 | mean Delta_P=-0.000490363 |
| s225 | alpha_TP | F11 Delta_P narrow_half | Delta_P includes 0 | [-0.000646863, 8.91488e-05] | 20 | mean Delta_P=-0.000278857 |
| s225 | alpha_Tfixed | F11 Delta_P narrow_half | Delta_P includes 0 | [-0.0014023, 0.000200612] | 20 | mean Delta_P=-0.000600842 |
| s225 | alpha_TAM | F11 Delta_P narrow_1x | Delta_P includes 0 | [-0.000436835, 0.000383136] | 20 | mean Delta_P=-2.68496e-05 |
| s225 | alpha_TP | F11 Delta_P narrow_1x | Delta_P includes 0 | [-0.000134061, 0.000327665] | 20 | mean Delta_P=9.6802e-05 |
| s225 | alpha_Tfixed | F11 Delta_P narrow_1x | Delta_P includes 0 | [-0.000417479, 0.000412666] | 20 | mean Delta_P=-2.40657e-06 |
| s225 | alpha_TAM | F11 Delta_P narrow_2x | Delta_P includes 0 | [-0.000277933, 0.000166297] | 20 | mean Delta_P=-5.58175e-05 |
| s225 | alpha_TP | F11 Delta_P narrow_2x | Delta_P includes 0 | [-0.000191414, 0.000145695] | 20 | mean Delta_P=-2.28595e-05 |
| s225 | alpha_Tfixed | F11 Delta_P narrow_2x | Delta_P includes 0 | [-0.000306044, 0.000143381] | 20 | mean Delta_P=-8.13314e-05 |
| s225 | alpha_TAM | F11 Delta_P twin | Delta_P includes 0 | [-0.0014632, 0.000645925] | 26 | mean Delta_P=-0.000408638 |
| s225 | alpha_TP | F11 Delta_P twin | Delta_P includes 0 | [-0.000652842, 0.000583532] | 26 | mean Delta_P=-3.4655e-05 |
| s225 | alpha_Tfixed | F11 Delta_P twin | Delta_P includes 0 | [-0.00146032, 0.000659679] | 26 | mean Delta_P=-0.000400322 |
| s225 | alpha_TAM | F11 difference in differences narrow_half_minus_1x | difference includes 0 | [-0.00127641, 0.000349387] | 20 | mean=-0.000463514 |
| s225 | alpha_TP | F11 difference in differences narrow_half_minus_1x | difference includes 0 | [-0.000802666, 5.13481e-05] | 20 | mean=-0.000375659 |
| s225 | alpha_Tfixed | F11 difference in differences narrow_half_minus_1x | difference includes 0 | [-0.00141559, 0.000218718] | 20 | mean=-0.000598435 |
| s225 | alpha_TAM | F11 difference in differences narrow_1x_minus_2x | difference includes 0 | [-0.000415263, 0.000473199] | 20 | mean=2.8968e-05 |
| s225 | alpha_TP | F11 difference in differences narrow_1x_minus_2x | difference includes 0 | [-0.000127858, 0.000367181] | 20 | mean=0.000119662 |
| s225 | alpha_Tfixed | F11 difference in differences narrow_1x_minus_2x | difference includes 0 | [-0.000363163, 0.000521012] | 20 | mean=7.89249e-05 |
| s225 | alpha_TAM | F11 difference in differences twin_minus_production | difference includes 0 | [-0.00175333, 0.000691443] | 26 | mean=-0.000530944 |
| s225 | alpha_TP | F11 difference in differences twin_minus_production | difference includes 0 | [-0.000790772, 0.000616907] | 26 | mean=-8.69328e-05 |
| s225 | alpha_Tfixed | F11 difference in differences twin_minus_production | difference includes 0 | [-0.00173026, 0.000726109] | 26 | mean=-0.000502073 |
| s4 | alpha_TP | F7 Delta_half | not shown decreasing | [-0.00858636, 0.00676019] | 71 | mean Delta_half=-0.000913087 |
| s4 | rho(X_b), pair 8-16 | F8 readings | pathwise resolved |  | 20 | median=1.3892e-05 p95=0.000143805 (m=0.00320762, S=0.435013) |
| s4 | alpha_TAM | F11 Delta_P production | Delta_P includes 0 | [-0.000470114, 6.87079e-05] | 71 | mean Delta_P=-0.000200703 |
| s4 | alpha_TP | F11 Delta_P production | Delta_P includes 0 | [-0.00015048, 0.00037593] | 71 | mean Delta_P=0.000112725 |
| s4 | alpha_Tfixed | F11 Delta_P production | Delta_P includes 0 | [-0.00050369, 4.33945e-05] | 71 | mean Delta_P=-0.000230148 |
| s4 | alpha_TAM | F11 Delta_P narrow_half | Delta_P above 0 | [8.90982e-05, 0.00104871] | 20 | mean Delta_P=0.000568906 |
| s4 | alpha_TP | F11 Delta_P narrow_half | Delta_P includes 0 | [-0.000396899, 7.72057e-05] | 20 | mean Delta_P=-0.000159846 |
| s4 | alpha_Tfixed | F11 Delta_P narrow_half | Delta_P above 0 | [0.000163591, 0.00108213] | 20 | mean Delta_P=0.00062286 |
| s4 | alpha_TAM | F11 Delta_P narrow_1x | Delta_P includes 0 | [-0.000145173, 0.000178299] | 20 | mean Delta_P=1.6563e-05 |
| s4 | alpha_TP | F11 Delta_P narrow_1x | Delta_P includes 0 | [-0.000287204, 3.85062e-05] | 20 | mean Delta_P=-0.000124349 |
| s4 | alpha_Tfixed | F11 Delta_P narrow_1x | Delta_P includes 0 | [-0.000120184, 0.00016716] | 20 | mean Delta_P=2.3488e-05 |
| s4 | alpha_TAM | F11 Delta_P narrow_2x | Delta_P includes 0 | [-5.4731e-05, 0.000119642] | 20 | mean Delta_P=3.24555e-05 |
| s4 | alpha_TP | F11 Delta_P narrow_2x | Delta_P includes 0 | [-7.03452e-05, 0.000168325] | 20 | mean Delta_P=4.89898e-05 |
| s4 | alpha_Tfixed | F11 Delta_P narrow_2x | Delta_P includes 0 | [-7.18347e-05, 0.0001017] | 20 | mean Delta_P=1.49328e-05 |
| s4 | alpha_TAM | F11 Delta_P twin | Delta_P above 0 | [0.00042044, 0.00299174] | 20 | mean Delta_P=0.00170609 |
| s4 | alpha_TP | F11 Delta_P twin | Delta_P below 0 | [-0.0011444, -3.47944e-05] | 20 | mean Delta_P=-0.000589598 |
| s4 | alpha_Tfixed | F11 Delta_P twin | Delta_P above 0 | [0.000406437, 0.0030188] | 20 | mean Delta_P=0.00171262 |
| s4 | alpha_TAM | F11 difference in differences narrow_half_minus_1x | difference above 0 | [0.000163689, 0.000940997] | 20 | mean=0.000552343 |
| s4 | alpha_TP | F11 difference in differences narrow_half_minus_1x | difference includes 0 | [-0.000328952, 0.000257957] | 20 | mean=-3.54978e-05 |
| s4 | alpha_Tfixed | F11 difference in differences narrow_half_minus_1x | difference above 0 | [0.000220871, 0.000977873] | 20 | mean=0.000599372 |
| s4 | alpha_TAM | F11 difference in differences narrow_1x_minus_2x | difference includes 0 | [-0.000161679, 0.000129894] | 20 | mean=-1.58924e-05 |
| s4 | alpha_TP | F11 difference in differences narrow_1x_minus_2x | difference includes 0 | [-0.000376106, 2.94288e-05] | 20 | mean=-0.000173338 |
| s4 | alpha_Tfixed | F11 difference in differences narrow_1x_minus_2x | difference includes 0 | [-0.000125471, 0.000142581] | 20 | mean=8.55524e-06 |
| s4 | alpha_TAM | F11 difference in differences twin_minus_production | difference above 0 | [0.000415771, 0.00287489] | 20 | mean=0.00164533 |
| s4 | alpha_TP | F11 difference in differences twin_minus_production | difference includes 0 | [-0.000918347, 0.000428867] | 20 | mean=-0.00024474 |
| s4 | alpha_Tfixed | F11 difference in differences twin_minus_production | difference above 0 | [0.000423462, 0.00293935] | 20 | mean=0.00168141 |

### exploratory

| amplitude | estimator | decision point | verdict | CI | n | note |
|---|---|---|---|---|---|---|
| s4 | alpha_TAM | F10 width check w24_vs_tw | inconclusive | [-0.0190798, 0.00576286] | 30/30 | diff=-0.00665847 margin=0.0068662 |
| s4 | alpha_TP | F10 width check w24_vs_tw | inconclusive | [-0.00697705, 0.0139799] | 30/30 | diff=0.00350142 margin=0.0067672 |
| s4 | alpha_TAM | F10 width check w12_vs_n16 | disagree | [0.0215526, 0.0472036] | 30/30 | diff=0.0343781 margin=0.00635197; wide = s4_w12+s4_w12c; seed 7214 excluded (s4_w12 halted, owner decision 2026-10-28) |
| s4 | alpha_TP | F10 width check w12_vs_n16 | disagree | [0.0386602, 0.0541048] | 30/30 | diff=0.0463825 margin=0.00678013; wide = s4_w12+s4_w12c; seed 7214 excluded (s4_w12 halted, owner decision 2026-10-28) |

### process

| amplitude | estimator | decision point | verdict | CI | n | note |
|---|---|---|---|---|---|---|
| all |  | CHECK_unittest | ok |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_geometry | ok |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_regress_refs | reused(/home/sesquerre/regress/refs/1e40cfe/out/REGRESS_DONE) |  |  | rc=NA (from checks/results.txt) |
| all |  | CHECK_regress_head | REGRESS_DONE |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_regress | REGRESS |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s1_n05 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s1_n10 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s1_n20 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s1_tw | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s225_n08 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s225_n16 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s225_n32 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s225_tw | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s4_n16 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s4_n32 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s4_n64 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s4_tw | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s4_w24 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_solve_chk_s4_w12 | converged |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_bytecheck | BYTECHECK |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_smoke_reduce | SMOKE |  |  | rc=0 (from checks/results.txt) |
| all |  | CHECK_signtable | SIGNTABLE |  |  | rc=0 (from checks/results.txt) |
