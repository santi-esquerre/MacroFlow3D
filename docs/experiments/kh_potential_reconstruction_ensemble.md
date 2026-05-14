# KH potential reconstruction ensemble

Date: 2026-05-14

Status: completed 100-seed experimental comparison.

Branch: `feature/kh-potential-flow-interpolation`

MacroFlow3D code under test: `6a6357ad81b3c1f8818d9c69b910756b2db73232`

Par2_Core code under test: `6611c681dfe5aeeada345a94825496c850b8289b`

Related records:

- PSPTA standby: `docs/experiments/pspta_standby_2026-05-13.md`
- KH interface contract: `docs/experiments/kh_potential_reconstruction_interface_2026-05-13.md`
- Theory basis: `docs/theory/lester-2023-key-claims.md`
- Monte Carlo baseline discipline: `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md`
- Result bundle: `docs/experiments/kh_potential_reconstruction_results/`

## Question

Does reconstructing particle velocities from interpolated scalar fields,

```math
q(x) = -K(x) \nabla h(x),
```

reduce numerical helicity and transverse macrodispersion relative to the
existing Par2 face-velocity interpolation backend?

This experiment tests KH reconstruction only. It does not construct the two
streamline invariants `psi1, psi2`, does not enforce invariant preservation, and
does not claim to replace PSPTA.

## Backends

`FACE_TRILINEAR` is the current MacroFlow3D-Par2 velocity path. The label follows
the experiment vocabulary; in the current Par2 implementation this maps to the
legacy face-field linear interpolation path.

`KH_POTENTIAL_RECONSTRUCTION` is the new experimental pure-advection path. It
binds cell-centered `K` and hydraulic head `h` directly from device memory,
samples `K`, estimates `grad(h)` with local finite differences plus trilinear
interpolation, and returns `q=-K grad(h)`.

No velocity projection, divergence repair, or forced matching to the face field
is applied.

## Configuration

The full 100-seed run used the reproducible driver:

```bash
scripts/remote run kh-ensemble-full-20260513-1415 -- \
  "bash scripts/kh_reconstruction/remote_kh_ensemble_driver.sh 100 full"
```

Per seed, both backends used:

- grid: `64 x 32 x 32`, `dx=5`
- `sigma2=1.0`, `corr_length=50`, `n_modes=1000`, exponential covariance
- seeds: `0..99`
- flow: `mg_cg`, west/east Dirichlet head `100 -> 0`, periodic y/z
- transport: `1000` particles, `dt=1`, `n_steps=500`
- pure advection: `diffusion=0`, `alpha_l=0`, `alpha_t=0`
- same particle seed: `123456789`
- same macrodispersion estimator and sampling cadence

This is a controlled standard-size experimental harness, not the production
`2048 x 256 x 256` macrodispersion configuration.

## Commands Run

Local build:

```bash
cmake --build build/wsl-debug --target macroflow3d_pipeline kh_potential_reconstruction_tests -j
```

Remote V100 build/test:

```bash
./scripts/remote exec -- \
  "cmake --build build/v100-release --target macroflow3d_pipeline kh_potential_reconstruction_tests -j && \
   ctest --test-dir build/v100-release --output-on-failure -R kh_potential_reconstruction_tests"
```

Remote smoke:

```bash
./scripts/remote exec -- \
  "bash scripts/kh_reconstruction/remote_kh_ensemble_driver.sh 2 smoke"
```

Remote full ensemble:

```bash
./scripts/remote run kh-ensemble-full-20260513-1415 -- \
  "bash scripts/kh_reconstruction/remote_kh_ensemble_driver.sh 100 full"
```

Local analysis from fetched lightweight artifacts:

```bash
scripts/kh_reconstruction/analyze_kh_ensemble.py artifacts/kh_reconstruction
```

## Validation

The KH unit/microtest validates:

- linear `h`, constant `K`: constant reconstructed `q`
- linear `h`, variable `K`: velocity scales with `K`
- smooth manufactured periodic y/z field: numerical `q` against analytic `q`
- y/z periodic samples near transverse boundaries
- finite velocities, no NaNs/Infs in the test samples

Results:

- local build passed
- remote V100 KH microtest passed
- 2-seed remote smoke passed
- 100-seed remote ensemble completed with `state=succeeded`, `exit_code=0`

Remote status timestamp:

```text
updated_at: 2026-05-13T17:29:53Z
```

All current full-run expected artifacts exist:

| File type | Count |
| --- | ---: |
| `alpha_timeseries.csv` | 200 |
| `field_diagnostics.csv` | 200 |
| `transport_diagnostics.csv` | 200 |
| `runtime_diagnostics.csv` | 200 |
| `log.txt` | 200 |
| `config_used.yaml` | 200 |

Some `.previous.*` directories also exist because earlier smoke outputs were
archived before rerunning the full ensemble.

## Result Artifacts

Remote canonical artifacts:

```text
artifacts/kh_reconstruction/
  config/
  logs/
  raw/seed_000/{face,kh}/...
  ...
  raw/seed_099/{face,kh}/...
  summary/
  plots/
```

Durable checked-in summaries and figures:

```text
docs/experiments/kh_potential_reconstruction_results/
  summary/
    ensemble_summary.csv
    kh_statistical_analysis.csv
    alpha_timeseries_mean_ci.csv
    alphaT_comparison_face_vs_kh.csv
    helicity_comparison_face_vs_kh.csv
    runtime_comparison.csv
  plots/
    alpha_L_mean_ci.svg
    alpha_T1_mean_ci.svg
    alpha_T2_mean_ci.svg
    alphaT_final_boxplots.svg
    helicity_norm_boxplot.svg
    helicity_vs_alphaT.svg
    velocity_diff_vs_delta_alphaT.svg
```

## Summary Table

Values are seed means. `ci95` is the paired or ensemble 95% confidence interval
half-width written by the analysis scripts.

| Metric | FACE mean | KH mean | KH - FACE | 95% CI half-width |
| --- | ---: | ---: | ---: | ---: |
| final `alpha_L` | 0.016356 | -0.001239 | -0.017594 | 0.006809 |
| final `alpha_T1` | 0.005938 | 0.006548 | 0.000610 | 0.001407 |
| final `alpha_T2` | 0.003390 | 0.005292 | 0.001901 | 0.001224 |
| final mean transverse alpha | 0.004664 | 0.005920 | 0.001256 | 0.000912 |
| mean normalized helicity | 0.079423 | 0.042684 | -0.036739 | 0.001169 |
| mean absolute divergence diagnostic | 0.003764 | 0.008792 | 0.005028 | 0.000347 |
| transport runtime, seconds | 0.302057 | 0.313929 | 0.011872 | 0.004786 |

Particle status:

- FACE: active mean/min/max = `1000 / 1000 / 1000`
- KH: active mean/min/max = `1000 / 1000 / 1000`
- problematic particles: `0` for all current runs

Velocity comparison:

- mean relative velocity L2 difference `||q_KH-q_FACE||/||q_FACE||`: `0.189532 +/- 0.004349`
- min/max relative velocity L2 difference: `0.148685 / 0.267030`
- mean p95 pointwise velocity difference: `0.171869 +/- 0.011982`
- mean vector correlation: `0.985675 +/- 0.000615`

## Required Figures

Average curves with uncertainty bands:

- `docs/experiments/kh_potential_reconstruction_results/plots/alpha_L_mean_ci.svg`
- `docs/experiments/kh_potential_reconstruction_results/plots/alpha_T1_mean_ci.svg`
- `docs/experiments/kh_potential_reconstruction_results/plots/alpha_T2_mean_ci.svg`

Boxplots:

- `docs/experiments/kh_potential_reconstruction_results/plots/alphaT_final_boxplots.svg`
- `docs/experiments/kh_potential_reconstruction_results/plots/helicity_norm_boxplot.svg`

Scatter plots:

- `docs/experiments/kh_potential_reconstruction_results/plots/helicity_vs_alphaT.svg`
- `docs/experiments/kh_potential_reconstruction_results/plots/velocity_diff_vs_delta_alphaT.svg`

## Analysis Questions

### 1. Does KH reduce numerical helicity?

Yes, strongly in this harness.

Mean normalized helicity drops from `0.079423` to `0.042684`. The paired
difference is `-0.036739 +/- 0.001169`, so this effect is much larger than the
sampling uncertainty of the 100-seed ensemble.

### 2. Does reduced helicity translate into lower alpha_T?

No.

`alpha_T1` is statistically consistent with no meaningful change:

```text
KH - FACE = +0.000610 +/- 0.001407
```

`alpha_T2` increases in this run:

```text
KH - FACE = +0.001901 +/- 0.001224
```

The mean transverse value also increases:

```text
KH - FACE = +0.001256 +/- 0.000912
```

Therefore, the measured helicity reduction did not produce the desired
transverse macrodispersion reduction.

### 3. Are alpha_T1 and alpha_T2 symmetric?

Within uncertainty, yes.

FACE:

```text
alpha_T1 - alpha_T2 = +0.002548 +/- 0.006867
```

KH:

```text
alpha_T1 - alpha_T2 = +0.001257 +/- 0.007295
```

Both intervals include zero. There is no strong evidence of transverse
directional asymmetry in this 100-seed reduced ensemble.

### 4. Is the FACE/KH difference larger than statistical uncertainty?

For helicity, yes. For transverse alpha, not in the desired direction.

The helicity decrease is decisive. The final mean transverse alpha increase is
small but larger than its estimated paired CI half-width in this harness. That
does not support KH as an alpha_T reducer.

### 5. Does the difference persist in time?

The averaged time curves show that the KH-FACE transverse alpha difference is
not just an isolated last-sample spike:

| Metric | first sampled delta | mid-run delta | final delta | mean delta over sampled times |
| --- | ---: | ---: | ---: | ---: |
| `alpha_L` | +0.002083 | -0.004245 | -0.017594 | -0.005554 |
| `alpha_T1` | +0.001405 | +0.000891 | +0.000610 | +0.001357 |
| `alpha_T2` | +0.002211 | +0.000243 | +0.001901 | +0.001683 |

The transverse KH increase is small but present over much of the sampled time
window.

### 6. Is KH computationally acceptable?

For this small harness, yes.

Runtime increases from `0.302057 s` to `0.313929 s` per run, about `3.9%`.
That overhead is acceptable for further experiments, but this result should not
be extrapolated to production grids without profiling.

### 7. Does this support higher-order KH interpolation?

Not as the next priority.

KH reduces normalized helicity, so the scalar-potential idea is not meaningless.
However, this baseline also increases the approximate divergence diagnostic and
does not reduce alpha_T. A higher-order KH variant may be worth testing later,
but the current evidence does not justify making tricubic/B-spline KH the main
path ahead of invariant-preserving transport work.

### 8. What does the result suggest is the main issue?

This experiment separates two effects:

- scalar KH reconstruction better aligns with the helicity-free continuous
  structure as measured by normalized helicity;
- trajectory-level transverse spreading still remains and is not reduced.

That points away from normalized helicity alone as the controlling error metric.
The likely remaining sources are:

- trajectory integration / lack of invariant preservation;
- divergence or mass-consistency error introduced by the KH finite-difference
  reconstruction;
- finite-time macrodispersion estimator noise in this reduced harness;
- insufficient smoothness/order of the baseline KH reconstruction.

The observed correlation diagnostics are weak:

- helicity vs final mean transverse alpha, all runs: `r=-0.010`
- FACE only: `r=+0.105`
- KH only: `r=+0.044`
- relative velocity difference vs KH-FACE alpha_T change: `r=-0.006`

So neither lower normalized helicity nor larger FACE/KH velocity mismatch
explains the alpha_T outcome by itself.

## Limitations

- The ensemble uses a reduced grid and runtime (`64 x 32 x 32`, `500` steps,
  `1000` particles), not the full production macrodispersion case.
- The KH backend is first-order/second-order local finite-difference gradient
  reconstruction plus trilinear interpolation, not tricubic or B-spline.
- The divergence diagnostic is an approximate sampled finite-difference measure
  of the evaluated backend field. In this grid the diagnostic samples all cell
  centers, but it is still not a conservative finite-volume projection.
- The final `alpha_L` estimate is noisy and even changes sign for KH in the
  final derivative estimate, which reinforces that this harness is not a
  production longitudinal macrodispersion measurement.
- Local WSL GPU execution remains limited by the local CUDA driver/runtime
  mismatch observed earlier; GPU execution evidence comes from V100.
- Remote sync emits non-fatal warnings about stale `petsc.incomplete-*` and
  `slepc.incomplete-*` directories. The KH run itself completed successfully.

## Recommendation

Recommendation: **B. KH does not reduce alpha_T significantly in the desired
direction; prioritize integrators / invariants / PSPTA.**

The precise conclusion is:

> `KH_POTENTIAL_RECONSTRUCTION` preserves the helicity-free signature better
> than the face-velocity backend under the normalized helicity diagnostic, but
> this does not translate into lower transverse macrodispersion in the 100-seed
> reduced ensemble. KH is not equivalent to PSPTA and does not guarantee
> conservation of the invariants `psi1, psi2`.

If KH is revisited, the next KH-specific experiment should first address the
increased divergence diagnostic, then run an `h/dt/seeds` convergence study
before investing in tricubic or B-spline interpolation.
