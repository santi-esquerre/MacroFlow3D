# SF-21 — 32^3 heterogeneity-continuation smokes with Anderson (V100)

## Question

Does the SF-21 lambda/eta-rescue continuation, with the SF-20-validated
Anderson acceleration wired into the stage solver (depth 5, start 5,
condition limit 1e12), reach the full lognormal target `lambda=1` for the
PRESPECIFIED fixed-seed 32^3 physical Gaussian smokes (`sigma_Y^2=0.25` and
`1`)?

## Hypothesis (recorded at re-activation, before any run)

Anderson converges the recorded eta=1 Picard-stall fixtures (SF-20 evidence:
64/88 iterations vs 500-exhaustion), so the lambda legs should progress far
beyond the pre-Anderson floor points (`lambda=0.373` at sigma^2=0.25,
`lambda=0.10` at sigma^2=1); `lambda->1` at `sigma^2>=1` was explicitly
flagged as beyond tested territory with honest failure possible.

## Configuration (fixtures verbatim; zero tuning)

- Grid 32^3, dx=1, triply periodic; SF-18 spectral Gaussian `Y` with
  seed 12345, `ell=8` (`ell/h=8`, `L/ell=4`), normalize_variance=true.
- `K_lambda = exp(lambda*Y)`; per-lambda SF-19 affine-periodic Darcy solve,
  prescribed mean flux (1,0,0) => exact `vbar=1` benchmark gauge.
- Lambda axis: start 0, target 1, initial step 0.1, min 0.0125, max 0.2,
  halve on failure, grow 1.5 after 2 easy stages (spec-locked).
- Eta rescue: restore -> eta=0 -> SF-17 eta axis ramp (spec-locked ordering).
- Epsilon fixed at 1e-2 (degenerate epsilon leg). Picard: tol 1e-6, max 500,
  adaptive omega (dashboard-locked). Anderson: enabled, depth 5, start 5,
  limit 1e12; solve-entry history clear (R2a).
- Gate (prespecified 2026-08-10, decision 5(b)): `reached_target`,
  `final_lambda==1`, every accepted stage `r_F <= 1e-6`.

## Build / commands

- Tree: SF-21 integrated head `463753d` (and the source-equivalent candidate
  for the first run); remote V100 `v100-release` preset (sm_70, CUDA 11.4),
  checksum-verified rsync + md5 spot checks.
- `./build/v100-release/streamfunction_operator_tests
   --case heterogeneity_smoke_sigma025 --case heterogeneity_smoke_sigma1`
  (ctest entry `streamfunction_heterogeneity_smoke`, heavy tier).
- End-to-end pipeline confirmation:
  `./build/v100-release/macroflow3d_pipeline
   apps/config_streamfunctions_gaussian_smoke32.yaml`.

## Observed outputs

| case | status | final lambda | stages (acc/total) | rescue stages | MG rebuilds | wall |
|---|---|---|---|---|---|---|
| sigma^2=0.25 | reached_target | 1.0 | 8/8 | 0 | 8 | 488.1 s |
| sigma^2=1 | lambda_floor_exhausted | 0.5 | 48/83 | 73 | 10 | 1580.1 s |

- sigma^2=0.25: every accepted stage `r_F <= 1e-6`; no degeneracy
  (unexplained=0); |c| p0.1% healthy (0.85 -> ~0.7 across the leg); e_v
  1.6e-3 -> ~2.5e-2 at lambda=1; pipeline run reproduced `reached_target`
  identically end-to-end with full Gate-3A exports (achieved mean flux
  (1,0,0) exact, div ~1e-13, K_eff diag 1.02-1.05 vs effective-medium
  exp(sigma^2/6)=1.04, final r_F=8.0e-7).
- sigma^2=1: per-stage integrity intact everywhere (accepted-stage
  `r_F <= 1e-6` sub-check PASS; zero unexplained degeneracy; smooth
  metrics). Failure mechanism: eta-rescue ramps accept eta=0.98125 and
  0.996875 (59-136 iterations), but attempts AT `eta=1` exit via the
  stagnation detector (<1% reduction over 10 iterations) after 21-72
  iterations with `r_F` plateauing at 0.9e-3..4.8e-3. Deterministic
  bitwise-identical repeats of identical attempts confirm solver
  determinism. Raw stage tables (91 records) preserved in the orchestration
  audit log (`vsmoke-gate-run1.log`) and reproduced on the final audited
  head in `sf21-final-suite`.

## Conclusion

Anderson qualitatively transformed the reachable regime (5x deeper lambda at
~1/10 the wall cost; sigma^2=0.25 fully passed) but the damped
Picard/Anderson fixed-point map is NON-CONTRACTIVE exactly at `eta=1` for
`sigma^2=1` beyond `lambda~0.5` at this resolution/regularization: a rate/
contraction boundary, not an instability or a physics failure. This is the
regime the plan reserves for Newton-Krylov terminal convergence. Owner
decision (option (a)): partial closure of SF-21 and Newton pull-forward;
the unmet gates move verbatim to SF-25.

## Caveats

- Single realization (seed 12345) at one resolution; the boundary
  `lambda~0.5` is not a universal constant.
- `epsilon=1e-2` fixed throughout (degenerate leg by design); smaller
  epsilon would be harder still.
- The stage records do not yet attribute per-stage Anderson accept/reject
  counters (candidate diagnostics extension for SF-25).
- Cite `docs/theory/lester-2023-key-claims.md` for regime interpretation:
  all claims here are about invariant CONSTRUCTION; no transport or
  transverse-macrodispersion claim is made.
