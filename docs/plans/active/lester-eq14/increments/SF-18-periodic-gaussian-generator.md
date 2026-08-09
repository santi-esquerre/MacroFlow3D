# SF-18 — Periodic Gaussian generator

- State: `active`
- Goal: `Generar campos gaussianos suaves verdaderamente periódicos y reproducibles.`
- Depends on: `SF-17`
- Unlocks: `SF-19`
- Branch: `science/lester-sf18-periodic-gaussian-generator`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 1 + Gate 2 + Gate 3A prerequisite`
- Human review: `required`
- Owner: `Claude Fable (orchestrator)`
- Started: `2026-08-09T20:50Z`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

Remove the current finite-box nonperiodicity and direct-summation scaling from
the physical validation input so boundary roughness cannot masquerade as a
streamfunction failure.

## Preconditions

- SF-17 completes the solver-side homotopy infrastructure.

## In scope

- A discrete spectral generator for periodic Gaussian-covariance log
  conductivity, reproducible seeds, variance/log-mean normalization, and cuFFT
  integration where justified.

## Out of scope

- Exponential covariance, Darcy flow, lambda continuation, and modifying the
  existing generator's behavior for current callers.

## Files and symbols

- Add `src/physics/stochastic/PeriodicGaussianField.cuh/.cu` and config/types
  scoped to the new generator.
- Extend CMake with cuFFT only for this implementation.

## Implementation specification

1. Sample reciprocal-lattice wavevectors and enforce Hermitian symmetry.
2. Use the periodicized Gaussian spectrum corresponding to the documented
   covariance convention; set/control the zero mode separately.
3. Normalize realized variance and log mean explicitly and record both.
4. Define a seed-to-mode mapping that represents the same continuous
   realization under grid refinement, through spectral truncation or generation
   at the finest requested grid followed by controlled restriction.
5. Retain the old stochastic generator API and results for existing configs.

## Expected numerical effect

Generated `Y` and its derivatives join smoothly at domain boundaries and their
statistics approach the requested Gaussian covariance.

## Validation commands

```bash
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure -R periodic_gaussian
ctest --test-dir build/wsl-debug --output-on-failure
```

## Acceptance thresholds

- Same config/seed produces identical device output on repeated runs.
- Boundary wrap value/derivative discrepancies match interior discretization
  error, not an O(1) jump.
- Mean and variance meet documented finite-sample tolerances.
- Radially binned spectrum/covariance matches the requested Gaussian shape over
  resolved modes; exact tolerances are fixed in the test before implementation.

## Regression surface

- cuFFT linking, seed semantics, covariance convention, zero mode, and memory
  at `256^3`.

## Failure and rollback policy

- Do not relabel the current continuous-wavevector direct sum as periodic.
- A disputed spectral normalization blocks physical benchmarks but not the
  previously accepted synthetic solver tests.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Periodic spectral construction and Hermitian symmetry are implemented.
- [ ] Seed, mean, variance, and refinement semantics are documented.
- [ ] Reproducibility, wrap, covariance, and spectrum tests pass.
- [ ] Existing stochastic configs are unchanged.
- [ ] Memory/runtime and human review are recorded.
- [ ] Evidence, PR, and commit are recorded.
- [ ] Dashboard marks SF-18 complete and selects SF-19.
<!-- completion-checklist:end -->

## Advancement rule

SF-19 may compute affine-periodic Darcy flow on these accepted periodic fields.

## Bitácora

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-08-09T20:50Z | activation on `master=bb432c8` (SF-17 closure merged via PR #29) | SF-18 activated after verifying `NEXT: SF-18`, SF-17 `done`, and checker `OK (29 increments, next=SF-18)` on the default branch. Interpretive decisions recorded for the human reviewer: (1) **Covariance/spectrum convention:** the target is the repository's documented smooth Gaussian covariance `C_Y(r)=sigma^2 exp(-(r/l)^2)` (overview + existing generator's Gaussian branch). The periodic generator realizes the PERIODICIZED covariance by sampling the continuous spectral density on the reciprocal lattice: with DFT convention `Y(x_j)=sum_m Yhat_m exp(i k_m.x_j)`, `k_m=2*pi*(m1/L1,m2/L2,m3/L3)`, the mode variances are `E|Yhat_m|^2 = sigma^2 * pi^{3/2} * l^3 * exp(-|k_m|^2 l^2/4) / (L1*L2*L3)` for `m != 0`; this equals wrapping `C` around the torus and is the documented meaning of "periodicized Gaussian spectrum". (2) **Zero mode and log mean:** the zero mode is excluded from the random sum (so the discrete sample mean of the fluctuation is exactly zero up to FFT roundoff); the log mean enters only as the existing `ln(K_g)` shift convention at exponentiation, and both the realized mean and the zero-mode handling are recorded in the generator report. (3) **Normalization:** a `normalize_variance` flag (default true) rescales the realization so its SAMPLE variance equals `sigma^2` exactly; the RAW realized variance and the applied scale are both recorded (spec item 3 "normalize ... explicitly and record both"); the scale is uniform so the spectrum SHAPE is preserved; refinement-consistency testing uses `normalize_variance=false`. (4) **Seed-to-mode mapping / refinement semantics (spectral truncation):** each integer mode `m` gets its complex coefficient from a stateless counter-based hash of `(seed, m)` (deterministic, order-independent, grid-independent); Hermitian symmetry is enforced by assigning the canonical representative of each +/- pair and conjugating the mirror; ALL Nyquist planes (`m_i = -N_i/2`) are zeroed, so a grid resolves exactly the modes `|m_i| <= N_i/2 - 1` and a finer grid uses a SUPERSET with the same coefficients — the same continuous realization under refinement, per the spec's spectral-truncation option (chosen over generate-finest+restrict). Truncation error is negligible for the benchmark family (spectrum ~ exp(-153) at the 128^3 cutoff for l=L/16). (5) **Sample locations:** cell centers `x=h(i+1/2)` (project convention), implemented as an exact half-cell phase twist of the spectrum before the inverse FFT; cross-grid comparisons must untwist this known phase. (6) **cuFFT:** double-precision 3D complex-to-real inverse transform; CMake links `CUDA::cufft` only for the targets owning this implementation; ~135 MiB half-spectrum + 128 MiB field at 256^3 (fits the local 4 GiB budget); cuFFT determinism for identical plan/input/GPU is an assumption VERIFIED by the bitwise reproducibility test. (7) **API scope:** new standalone `PeriodicGaussianField.cuh/.cu` with its own config/report types; the existing direct-sum generator, its API, and every existing config path remain byte-untouched; no pipeline/YAML wiring in SF-18 (consumption is SF-19+). (8) **PRESPECIFIED acceptance tolerances (fixed NOW, before any implementation or run; the test node implements them verbatim and never adjusts them after a run):** primary statistical fixture `64^3, dx=1 (L=64), l=8, sigma^2=1, seed 12345`. (a) Reproducibility: two same-seed generations bitwise identical (device memcmp == 0); a different seed produces a non-identical field. (b) Wrap: for each axis, RMS of first differences across the periodic seam / RMS of interior first differences in `[0.7, 1.4]`, and the same band for second differences; additionally max seam jump `<= 5x` max interior neighbor difference (an O(1) seam jump would give ratios ~8 at this l/h). (c) Mean/variance: |sample mean| <= 1e-10*sigma; with normalization |sample var/sigma^2 - 1| <= 1e-12; RAW realized variance within `5*sqrt(2/N_eff)` relative of the truncated-spectrum expectation, with `N_eff = (sum S_m)^2 / sum S_m^2` computed in the test from the same spectrum formula. (d) Spectrum: forward-FFT the UNNORMALIZED field, radially bin `|Yhat_m|^2*V` in |k| shells; for every bin with `n_bin >= 50` modes and theoretical `S >= 1e-6*S_max`: |P_bin/<S>_bin - 1| <= 5*sqrt(2/n_bin), `<S>_bin` averaged over the same modes. (e) Refinement: seeds equal, `normalize_variance=false`, 32^3 vs 64^3: untwisted spectral coefficients on the common active-mode set (|m_i| <= 15) equal within 1e-12 relative. (f) 256^3: generation must complete on the local 4 GiB GPU; peak memory and runtime are RECORDED (no numeric gate). | Base commit is this activation commit on `master=bb432c8`. Gate 1 + Gate 2 + Gate 3A-prerequisite apply; human review required, so the PR will stop at `awaiting_review` with `NEXT` unchanged. | Build intra-increment DAG; delegate implementation to isolated worker worktrees. |
| 2026-08-09T22:30Z | T02 audit finding T02-F1: measurement-methodology amendment to decision 8(e) | The T02 worker implemented every prespecified tolerance VERBATIM and honestly reported one failure without touching anything: refinement case (e) fails with max_rel_error=10.19 at mode (15,-14,-12). Orchestrator root cause (verified analytically and against the worker's magnitude-bucketed diagnostics): that mode's TRUE coefficient magnitude is ~1e-20 (Gaussian spectral decay exp(-k^2 l^2/8) at l=8, L=64), while a forward-FFT round trip of an O(sigma) field carries an ABSOLUTE coefficient noise floor of order eps*sigma (~1e-16..1e-17 observed). A RELATIVE 1e-12 bound on such modes is unmeasurable for ANY correct implementation — the original operationalization (relative 1e-12 with only a 1e-300 exact-zero floor) was a measurement-methodology defect in the prespecification, not an implementation defect. Where the comparison IS measurable the contract holds at machine precision: max relative error 1.15e-15 over the 333 common modes with |coef| >= 1e-2, degrading monotonically with signal magnitude exactly as an fp noise floor predicts (1.38e-14 at >=1e-3, 1.47e-13 at >=1e-4, ...). | **Amended criterion (decision 8(e) revision, recorded BEFORE the corrective run):** per common active mode, `|c_fine - c_coarse| <= max(1e-12 * |c_coarse|, 1e-15 * sqrt(sigma2))`. The absolute arm is an a-priori fp bound (~5x machine epsilon times the field scale sigma; ~100x above the OBSERVED ~1e-17 noise floor, i.e. deliberately not fitted to the data); the relative arm and the compared mode set are unchanged. The underlying scientific contract — identical grid-independent coefficients under spectral-truncation refinement — is unchanged and is already demonstrated at machine precision where resolvable. All other 25/26 prespecified checks passed unchanged: wrap ratios 0.90-1.05 in [0.7,1.4]; |mean|=7.2e-17; normalized variance exact to 3e-14; raw variance dev 0.141 vs tol 0.436 (N_eff=262.5, E_var=0.9891 — both computed in-test from the exact S_m sum); worst spectrum bin dev 0.110 within 5*sqrt(2/n_bin); 256^3 record: ~0.07-0.21 s, 135,266,304 B spectrum + 2,056 B scratch (cufft work area 0), ~258 MiB device delta, 8,323,199 active modes. | Corrective node C01 amends ONLY the case-(e) gating comparison to the criterion above (diagnostics and printed evidence retained); no other tolerance, fixture, seed, or binning changes; re-audit follows. |
