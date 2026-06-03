# KH Higher-Order Reconstruction Analysis

Date: 2026-05-15

Status: complete on `feature/kh-higher-order-reconstruction`.

This report records the higher-order KH reconstruction experiment series:

- `FACE_TRILINEAR`
- `KH_LINEAR`
- `KH_CUBIC_POTENTIAL_RECONSTRUCTION`
- `KH_LOGK_CUBIC_POTENTIAL_RECONSTRUCTION`

PSPTA, Strategy A, and Strategy C remain on standby and are intentionally out
of scope for this document.

## 1. Resumen ejecutivo

Se implementaron y evaluaron cuatro backends:

- `FACE_TRILINEAR`
- `KH_LINEAR`
- `KH_CUBIC_POTENTIAL_RECONSTRUCTION`
- `KH_LOGK_CUBIC_POTENTIAL_RECONSTRUCTION`

La validación manufacturada respondió bien a la pregunta numérica de base:
las dos variantes cúbicas mejoran claramente el error de `q` respecto a
`KH_LINEAR`, con la derivada de `h` tomada de manera analíticamente
consistente desde el mismo interpolante cúbico.

En transporte, el resultado principal es más matizado:

- en `gaussian_smooth` de 100 seeds, `KH_LOGK_CUBIC` mejora
  `alpha_T_mean` frente a `KH_LINEAR` y evita la deriva de `alpha_L` que sí
  aparece en `KH_CUBIC`;
- en `exponential_previous` de 100 seeds, ambas variantes cúbicas mejoran
  `alpha_T_mean` frente a `KH_LINEAR`, con una ventaja pequeña de
  `KH_LOGK_CUBIC` sobre `KH_CUBIC`;
- ninguna variante KH supera de forma estadísticamente concluyente a `FACE`
  en `alpha_T_mean` en los dos harnesses reducidos corridos aquí.

Las métricas de campo no siguen una narrativa simple:

- la divergencia diagnóstica baja de `KH_LINEAR` a las variantes cúbicas en
  ambos casos, pero sigue por encima de `FACE`;
- la helicidad normalizada muestreada no predice bien `alpha_T_mean`;
- no hubo partículas problemáticas ni NaN/Inf, y `KH_LOGK_CUBIC` mantuvo
  positividad de `K` por construcción.

Conclusión principal:

> El fallo del KH anterior no era una refutación limpia de la idea KH, porque
> el KH de mayor orden sí mejora de forma reproducible al backend
> `KH_LINEAR`. Sin embargo, la línea KH no entrega aquí una victoria clara
> sobre `FACE`. Si se sigue invirtiendo en KH, la variante a priorizar es
> `KH_LOGK_CUBIC`.

## 2. Motivación científica

`KH_LINEAR` was not a definitive test of the KH idea because it did not use a
single smooth interpolant for `h`. It interpolated `h`-derived gradients built
by separate finite differences, so the discrete evaluation was not a faithful
realization of the continuous Darcy structure

```math
q = -K \nabla h.
```

This phase therefore upgrades the KH backend family to evaluate:

- `h_interp(x)` from a local cubic tensor-product interpolant;
- `grad(h_interp)(x)` from the analytic derivative of that same interpolant;
- `K_interp(x)` either from direct cubic interpolation of `K`, or from cubic
  interpolation of `Y = ln K` followed by `K = exp(Y)`.

KH remains distinct from PSPTA: it does not construct `psi1, psi2`, does not
enforce invariant preservation, and does not by itself guarantee confinement to
streamsurfaces.

## 3. Implementación

Implemented hot-path changes are concentrated in:

- `src/external/Par2_Core/src/internal/fields/potential_flow_accessor.cuh`
- `src/external/Par2_Core/src/kernels/move_particles.cu`
- `src/external/Par2_Core/src/transport_engine.cu`
- MacroFlow3D enum/config/diagnostic plumbing around `velocity_eval_mode`

Current higher-order details:

- 4x4x4 local tensor-product cubic reconstruction using local Lagrange basis;
- double-precision cubic weights, stencil accumulation, derivatives, and
  `q = -K grad(h)`;
- periodic wrap in `y/z`;
- non-periodic `x` uses a shifted one-sided interior 4-point stencil near the
  boundary rather than a ghost-cell Dirichlet reconstruction;
- `KH_LOGK_CUBIC` evaluates `Y = ln K` cubically and reconstructs `K = exp(Y)`;
- `KH_CUBIC` evaluates `K` directly and reports sampled non-positive events in
  diagnostics instead of silently clamping them.

Diagnostics now report backend-specific:

- velocity/helicity/divergence summaries;
- `K_interp` min/max/mean;
- `count(K_interp <= 0)`;
- `count(clamped_K)` (currently zero because no clamp is applied);
- `logK` min/max for `KH_LOGK_CUBIC`;
- relative and pointwise FACE-vs-backend velocity differences.

## 4. Validación manufacturada

Completed manufactured validation:

- linear `h`, constant `K`;
- linear `h`, smooth positive `K = exp(Y)`;
- cubic manufactured `h` with analytic gradient;
- periodicity checks near `y/z` boundaries;
- direct-cubic positivity stress test;
- resolution sweep `N = 16, 32, 64`.

Artifacts already generated:

- `artifacts/kh_higher_order/summary/manufactured_order_tests.csv`

Current measured order-study errors:

| N | KH_LINEAR | KH_CUBIC | KH_LOGK_CUBIC |
| --- | ---: | ---: | ---: |
| 16 | 3.94549e-02 | 4.02994e-03 | 4.02994e-03 |
| 32 | 9.33264e-03 | 5.04344e-04 | 5.04344e-04 |
| 64 | 2.37782e-03 | 7.75639e-05 | 7.75639e-05 |

So the cubic variants improve clearly over `KH_LINEAR` before any ensemble
claim is made.

## 5. Setup físico

### `gaussian_smooth`

- covariance: Gaussian (`covariance_type = 1`)
- `sigma2 = 1.0`
- `corr_length = 50`
- grid: `64 x 32 x 32`, `dx = 5`
- pure advection, stationary Darcy flow
- west/east Dirichlet, periodic `y/z`

Completed so far:

- local smoke
- remote smoke
- remote 10-seed reduced ensemble
- remote 100-seed reduced ensemble

### `dreuzy_gaussian_reduced`

Defined as a desirable intermediate Monte Carlo baseline, but not executed in
this branch. The available validation budget was spent on:

- the smooth Gaussian case required to answer the Lester-compatible question;
- the exact `exponential_previous` repeat required to compare against the
  earlier KH failure mode.

### `exponential_previous`

Completed 100-seed repetition of the earlier exponential harness for direct
comparison against the environment where `KH_LINEAR` did not reduce `alpha_T`.

Operational note:

- the remote job reached all 100 seeds and wrote complete artifacts;
- the remote status ended as failed only because the final collector script
  assumed `.git` metadata on the remote sync tree;
- that harness bug was fixed in
  `scripts/kh_higher_order/collect_kh_higher_order_results.sh`;
- local fetch and local postprocessing completed successfully.

## 6. Resultados de campo

`gaussian_smooth` 100-seed summary:

- `FACE`: `helicity_norm_mean = 7.555050e-03`, `div_abs_mean = 4.482696e-04`
- `KH_LINEAR`: `helicity_norm_mean = 7.224158e-03`, `div_abs_mean = 1.115656e-03`
- `KH_CUBIC`: `helicity_norm_mean = 9.294009e-03`, `div_abs_mean = 1.016795e-03`
- `KH_LOGK_CUBIC`: `helicity_norm_mean = 9.294009e-03`, `div_abs_mean = 1.016795e-03`

Additional field notes from the 100-seed smooth run:

- both cubic variants lowered sampled `|div q|` relative to `KH_LINEAR` by
  about `9.89e-05` in paired mean delta, but remained above `FACE` by about
  `5.69e-04`;
- both cubic variants raised sampled normalized helicity relative to
  `KH_LINEAR` by about `2.07e-03` and relative to `FACE` by about `1.74e-03`;
- `KH_CUBIC` and `KH_LOGK_CUBIC` are identical under the current cell-center
  field diagnostics in the smooth case because both reconstructions recover the
  same nodal value at sampled cell centers; their transport difference comes
  from off-center particle sampling.

`exponential_previous` 100-seed summary:

- `FACE`: `helicity_norm_mean = 7.942350e-02`, `div_abs_mean = 3.763916e-03`
- `KH_LINEAR`: `helicity_norm_mean = 4.268448e-02`, `div_abs_mean = 8.791655e-03`
- `KH_CUBIC`: `helicity_norm_mean = 7.431659e-02`, `div_abs_mean = 7.689654e-03`
- `KH_LOGK_CUBIC`: `helicity_norm_mean = 7.431659e-02`, `div_abs_mean = 7.689654e-03`

Additional field notes from the 100-seed exponential run:

- both cubic variants lowered sampled `|div q|` relative to `KH_LINEAR` by
  about `1.102001e-03`, with extremely small paired `p`-values, but still
  remained above `FACE` by about `3.925738e-03`;
- both cubic variants raised sampled normalized helicity relative to
  `KH_LINEAR` by about `3.163211e-02`, while still remaining below `FACE`
  by about `5.106908e-03`;
- FACE-relative velocity differences were much larger in the exponential case
  than in the smooth case:
  mean relative L2 difference about `0.1807` versus `0.0587` in the smooth
  run, with mean vector correlation about `0.9865` versus `0.9980`.

Conductivity sampling diagnostics across both reduced ensembles showed:

- `count(K_interp <= 0) = 0` on the sampled field points for `KH_CUBIC`;
- `count(clamped_K) = 0` because no clamp was used;
- `KH_LOGK_CUBIC` preserved positivity by construction, with sampled
  `logK` ranges `[-4.30873, 4.21338]` in `gaussian_smooth` and
  `[-5.11754, 4.71355]` in `exponential_previous`.

## 7. Resultados de transporte

`gaussian_smooth` 100-seed means:

- `FACE`: `alpha_T_mean = 2.023468e-03 ± 4.743900e-03`
- `KH_LINEAR`: `alpha_T_mean = 3.098438e-03 ± 4.828270e-03`
- `KH_CUBIC`: `alpha_T_mean = 2.265209e-03 ± 4.371329e-03`
- `KH_LOGK_CUBIC`: `alpha_T_mean = 1.904823e-03 ± 4.466867e-03`

All current runs kept:

- active particles ≈ `1000`
- problematic particles = `0`
- no observed NaN/Inf transport failures

Paired smooth-case deltas versus `KH_LINEAR`:

- `KH_CUBIC - KH_LINEAR`:
  `delta alpha_T_mean = -8.332285e-04` with approximate 95% CI
  `[-2.097690e-03, 4.312327e-04]`, paired `t`-test `p = 1.94e-01`,
  Wilcoxon `p = 1.05e-01`
- `KH_LOGK_CUBIC - KH_LINEAR`:
  `delta alpha_T_mean = -1.193615e-03` with approximate 95% CI
  `[-2.357513e-03, -2.971579e-05]`, paired `t`-test `p = 4.45e-02`,
  Wilcoxon `p = 1.94e-02`

Transport caveat from the smooth case:

- direct `KH_CUBIC` shows a noticeable upward shift in `alpha_L_mean`
  (`5.248711e-02` versus `2.461461e-02` for `KH_LINEAR` and
  `2.683687e-02` for `KH_LOGK_CUBIC`);
- `KH_LOGK_CUBIC` avoids that `alpha_L` drift while slightly outperforming
  `FACE` and `KH_LINEAR` on `alpha_T_mean`.

The `alpha_L` shift is not subtle in the paired smooth data:

- `KH_CUBIC - KH_LINEAR`: `delta alpha_L_mean = 2.787250e-02`,
  paired `t`-test `p = 6.12e-09`, Wilcoxon `p = 1.32e-09`
- `KH_LOGK_CUBIC - KH_LINEAR`: `delta alpha_L_mean = 2.222257e-03`,
  paired `t`-test `p = 3.58e-01`, Wilcoxon `p = 2.54e-01`

`exponential_previous` 100-seed means:

- `FACE`: `alpha_T_mean = 4.664254e-03 ± 3.621228e-03`
- `KH_LINEAR`: `alpha_T_mean = 5.919932e-03 ± 3.678031e-03`
- `KH_CUBIC`: `alpha_T_mean = 4.128373e-03 ± 3.247612e-03`
- `KH_LOGK_CUBIC`: `alpha_T_mean = 4.043108e-03 ± 3.429940e-03`

Paired exponential-case deltas versus `KH_LINEAR`:

- `KH_CUBIC - KH_LINEAR`:
  `delta alpha_T_mean = -1.791559e-03` with approximate 95% CI
  `[-3.231871e-03, -3.512471e-04]`, paired `t`-test `p = 1.53e-02`,
  Wilcoxon `p = 6.95e-03`
- `KH_LOGK_CUBIC - KH_LINEAR`:
  `delta alpha_T_mean = -1.876824e-03` with approximate 95% CI
  `[-3.252764e-03, -5.008837e-04]`, paired `t`-test `p = 8.01e-03`,
  Wilcoxon `p = 6.60e-03`

Against `FACE`, the exponential means moved in the favorable direction but did
not become statistically decisive in this reduced campaign:

- `KH_CUBIC - FACE`: `delta alpha_T_mean = -5.358811e-04`,
  paired `t`-test `p = 4.68e-01`, Wilcoxon `p = 6.67e-01`
- `KH_LOGK_CUBIC - FACE`: `delta alpha_T_mean = -6.211458e-04`,
  paired `t`-test `p = 3.26e-01`, Wilcoxon `p = 4.31e-01`

Longitudinal transport stayed informative:

- in `gaussian_smooth`, `KH_CUBIC` showed a statistically strong upward
  `alpha_L` drift while `KH_LOGK_CUBIC` stayed close to `KH_LINEAR` and `FACE`;
- in `exponential_previous`, both cubic variants moved `alpha_L` upward
  relative to `KH_LINEAR`, but `KH_CUBIC` ended closer to `FACE` than
  `KH_LOGK_CUBIC`.

Across all reduced ensembles:

- active particles stayed at `1000` on average;
- problematic particles stayed at `0`;
- no NaN/Inf transport failures were observed.

## 8. Relación métricas-causa

The smooth 100-seed data already warns against a naive field-to-transport
story:

- lower sampled helicity did not imply lower `alpha_T_mean`, because
  `KH_LINEAR` had the lowest smooth-case helicity of the KH family but not the
  lowest `alpha_T_mean`;
- lower sampled divergence relative to `KH_LINEAR` did align with lower
  `alpha_T_mean` for both cubic variants, but the cubic variants still had
  larger sampled divergence than `FACE`, so divergence alone does not explain
  the ordering;
- the smooth paired tests already favor `KH_LOGK_CUBIC` over `KH_LINEAR` on
  `alpha_T_mean`, while `KH_CUBIC` remains statistically weaker on that metric
  and clearly worse on `alpha_L`.

The exponential 100-seed data points in the same general direction:

- lower sampled helicity again failed as a predictor of lower `alpha_T_mean`,
  because `KH_LINEAR` had by far the lowest sampled helicity but also the worst
  `alpha_T_mean` of the four backends;
- lower sampled divergence relative to `KH_LINEAR` again aligned with lower
  `alpha_T_mean` for the cubic variants, but not enough to beat `FACE`
  decisively;
- the large jump in FACE-relative velocity difference from the smooth case
  (`~0.059`) to the exponential case (`~0.181`) is consistent with the KH
  family becoming more intrusive as the conductivity field gets rougher.

So the best causal reading from this phase is:

- consistent higher-order KH reconstruction does improve over the old
  `KH_LINEAR` baseline;
- divergence reduction seems more informative than helicity reduction inside the
  KH family, but neither metric alone predicts `alpha_T_mean`;
- the main benefit comes from replacing the inconsistent low-order KH
  reconstruction, not from any simple monotone helicity story;
- `KH_LOGK_CUBIC` is the safest KH continuation because the log-conductivity
  path preserves positivity and avoids the most obvious transport pathology of
  direct cubic `K`.

## 9. Costo computacional

`gaussian_smooth` 100-seed runtime means:

- `FACE`: `0.301411 s`
- `KH_LINEAR`: `0.309551 s`
- `KH_CUBIC`: `0.312429 s`
- `KH_LOGK_CUBIC`: `0.323713 s`

Relative to `FACE`, the smooth 100-seed mean overheads are modest:

- `KH_LINEAR`: about `+2.7%`
- `KH_CUBIC`: about `+3.7%`
- `KH_LOGK_CUBIC`: about `+7.4%`

`exponential_previous` 100-seed runtime means:

- `FACE`: `0.301227 s`
- `KH_LINEAR`: `0.311142 s`
- `KH_CUBIC`: `0.314989 s`
- `KH_LOGK_CUBIC`: `0.327990 s`

Relative to `FACE`, the exponential mean overheads are also modest:

- `KH_LINEAR`: about `+3.3%`
- `KH_CUBIC`: about `+4.6%`
- `KH_LOGK_CUBIC`: about `+8.9%`

This keeps the higher-order KH line viable as a reduced-harness research path.
It does not by itself justify production preference over `FACE` or over an
invariant-preserving method.

## 10. Limitaciones

- reduced grid and particle count;
- finite-time macrodispersion estimate;
- KH does not preserve the invariant structure by construction;
- no global B-spline prefiltering in this phase;
- no divergence projection in this phase;
- no separate `dreuzy_gaussian_reduced` ensemble was executed in this branch;
- current field diagnostics are sampled at cell centers, which is informative
  but does not fully resolve off-center differences seen by particles.
- direct-cubic `K` positivity was only sampled on the field-diagnostic points
  during ensembles; off-center overshoots remain possible even though the
  manufactured tests already demonstrate the issue in principle.
- the remote exponential job initially reported failure because of a collector
  harness bug, though the underlying simulation outputs were complete and the
  harness was fixed afterward.

## 11. Recomendación

Recommendation: `B. KH_LOGK mejora más que KH_CUBIC: priorizar LOGK.`

Reasoning:

- the higher-order KH line does improve materially over `KH_LINEAR`, so the
  previous KH failure cannot be read as a clean rejection of the KH idea;
- `KH_LOGK_CUBIC` is the most stable KH variant across the reduced campaigns:
  it preserves positivity by construction, improves `alpha_T_mean` versus
  `KH_LINEAR` in both the smooth and exponential harnesses, and avoids the
  strongest smooth-case `alpha_L` pathology seen in direct cubic `K`;
- direct `KH_CUBIC` remains scientifically useful as a comparison backend, but
  its smooth-case `alpha_L` drift makes it a weaker candidate for follow-on
  investment;
- neither KH cubic variant yet delivers a decisive `alpha_T_mean` win over
  `FACE`, so KH should remain an experimental side line rather than replacing
  invariant-focused work.

Recommended next step if the KH branch is continued at all:

- keep `KH_LOGK_CUBIC` as the reference KH variant;
- add off-center conductivity diagnostics or subcell sampling for direct-cubic
  `K`;
- only then consider whether additional KH effort is warranted before returning
  focus to invariant-preserving transport.
