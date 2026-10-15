# Gaussian log-conductivity covariance only; exponential option retired

- Status: accepted (owner decision, 2026-10-07; human review of the PR pending)
- Date: 2026-10-07
- Deciders: owner
- Scope: baseline direct-sum generator `src/physics/stochastic/stochastic.cu`,
  `stochastic.covariance_type` in the config layer, the `apps/` reference configs,
  and the convention shared with the SF-18 periodic generator
  `src/physics/stochastic/PeriodicGaussianField.cuh` (unchanged).

## Context

- The long-domain reference configs (`apps/config_pipeline_par2.yaml`,
  `apps/config_pipeline_pspta.yaml`, `apps/config_pspta_small.yaml`) carried
  `covariance_type: 0` (exponential), and the par2 header claimed to "replicate the legacy
  problem". The claim was false: `legacy/main_transport_JSON_input.cu` (field generation,
  ~line 879) only ever launched `random_kernel_3D_gauss`. The legacy reference problem
  (`legacy/input/parameters_sigma1.json`) was Gaussian.
- The legacy Gaussian kernel used a different correlation-length convention:
  `k = k/(2 lambda/sqrt(pi)) * sqrt(2)` (`legacy/random_field_generation.cu:114`), i.e.
  `C(r) = sigma2 * exp(-pi r^2 / (4 lambda^2))`.
- Two conventions coexisted in the code: the exponential branch (`covariance_type == 0`,
  the default) and the Gaussian branch (any other integer, silently) of the baseline
  generator. The baseline Gaussian path had no test.
- The SF-18 periodic generator already uses `C(r) = sigma2 * exp(-(r/lambda)^2)`
  (`PeriodicGaussianField.cuh`, section 1).
- AGENTS.md already forbids treating exponential-covariance fields as equivalent to smooth
  Gaussian fields for invariant-existence validation; the project scope is smooth Gaussian
  fields.

## Decision

1. Convention: both generators produce `C(r) = sigma2 * exp(-(r/lambda)^2)`.
2. `stochastic.covariance_type` accepts only `1` (Gaussian). The default is `1` in
   `Config.hpp`, `ConfigDefaults.hpp` and `physics_config.hpp`. `validate_config` reports
   `[stochastic.covariance_type] ...` for any other value, so old configs fail loudly at
   load time (no fallback, no silent switch). `generate_gaussian_field` throws
   `std::invalid_argument` for `covariance_type != 1`. The exponential mode sampler
   (`kernel_random_modes_exp`) is deleted. The Gaussian sampler arithmetic and the
   direct-sum evaluation are unchanged.
3. The reference configs switch to `covariance_type: 1`; their headers state exactly what is
   and is not replicated from the legacy problem. The streamfunction configs switch too
   (their `stochastic` block is inert: `sigma2 = 0` or `field_source: periodic_gaussian`).
4. "`alpha_L` must match RWPT" (roadmap record
   `2026-10-02-roadmap-audit-and-foundational-redesign.md`, later phases; plan "Later
   phases" item 3) means an RWPT ensemble re-run with the same YAML in this code, not the
   legacy figures.
5. Legacy lambda mapping, recorded and not used: a legacy covariance would be obtained with
   `corr_length = 2 lambda_legacy / sqrt(pi)`.

## Consequences

- Outputs of `apps/config_pspta_small.yaml` (and of par2/pspta reference runs) change by
  design: the field is now Gaussian. Any byte-compare reference of those outputs must be
  re-baselined.
- `apps/config_streamfunctions_homogeneous.yaml` and
  `apps/config_streamfunctions_continuation.yaml` (`sigma2 = 0`): the K field is unchanged
  (`logK = 0` exactly, `K = K_mean`); only the `covariance_type` line of the effective config
  and manifest changes.
- New fast ctest entry `stochastic_baseline_gaussian`
  (`tests/stochastic/stochastic_baseline_tests.cu`): mode-sampler moments, CPU
  re-evaluation, mean/variance/covariance against `exp(-(r/lambda)^2)`, reproducibility,
  lognormal transform, and the retirement contract (generator guard + validator).
- Reintroducing exponential fields would require a new generator and a new decision record.
- Open: which generator (periodic SF-18 or the non-periodic baseline) feeds the long-domain
  study (plan "Later phases" item 3) remains open.

## Classification

| Item | Status |
|---|---|
| Legacy reference problem used only the Gaussian kernel | confirmed in code (`legacy/main_transport_JSON_input.cu`) |
| Legacy lambda convention `exp(-pi r^2/(4 lambda^2))` | confirmed in code (`legacy/random_field_generation.cu:114`) |
| SF-18 convention `exp(-(r/lambda)^2)` | confirmed in code (`PeriodicGaussianField.cuh`) |
| `covariance_type` accepts only 1; default 1; exponential kernel deleted | confirmed in code (this change) |
| Baseline sampler/field reproduce `exp(-(r/lambda)^2)` | confirmed in code (`stochastic_baseline_gaussian`) |
| `alpha_L` RWPT comparison = re-run with the same YAML in this code | accepted scope |
| Legacy mapping `corr_length = 2 lambda_legacy/sqrt(pi)` | recorded, not used |
| Generator for the long-domain study (SF-18 periodic vs baseline) | open question |
