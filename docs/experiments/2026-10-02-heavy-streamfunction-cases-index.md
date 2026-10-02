# Heavy streamfunction cases — index of documented experiments (not ctest)

- Date: 2026-10-02
- Status: active index (SF-27, validation-tier hygiene)
- Policy implemented: `docs/validation/acceptance-gates.md`, Gate 1,
  "Validation tiers in ctest"
- Execution rules: `docs/runbooks/remote-v100.md` Section 0 (detached jobs,
  per-increment mirrors, one job per GPU) and Section 3 (interface)
- Last recorded outcomes: `docs/experiments/2026-10-01-sf26-pairing-correction-gates.md`
  (V100 full suite, head `58898bf`, 2026-10-01); raw extract
  `docs/experiments/artifacts/2026-10-01-sf26-probes/raw/v100_ctest_full_extract.txt`
- Classification of the red outcomes:
  `docs/decisions/2026-10-01-eta1-residual-floor-gauge-degeneracy.md`, item D6
- Theory: `docs/theory/lester-2023-key-claims.md`

## What this note is for

SF-27 removed six `ctest` registrations that ran multi-case solver sweeps and
science smokes of the periodic-fluctuation streamfunction stack. Together they
took 63 979 s of the 66 700 s V100 full-suite run at `58898bf`. This note is
the one place that lists their eight cases: what each tests, how to run it,
how long it takes, and what it last produced.

Use it when:

- an increment or decision needs fresh evidence from one of these cases;
- you need the exact command or the expected wall time before launching a job;
- you need the last recorded outcome of a case to compare against.

Do not use it for:

- **acceptance by ctest substitution.** These cases are not part of Tier A
  "all registered tests pass". Running one is a scientific experiment whose
  result is recorded and interpreted, not a pass/fail build gate.
- **silent re-baselining.** Record a changed outcome in a new experiment note
  or the increment bitácora, with head, build, job name, and log. Do not edit
  the "last recorded outcome" lines below to match a new run without that
  record.
- **the fast tier.** Fast contract tests stay registered in `ctest`.

## Mechanism

- All eight cases are compiled into the existing `streamfunction_operator_tests`
  target. They live in `heavy_cases()` in
  `tests/streamfunctions/streamfunction_operator_tests.cpp`, deliberately
  outside `cases()`, so `--list` and the no-argument run stay on the fast tier.
- `--case <name>` is resolved first in `cases()`, then in `heavy_cases()`.
  An unknown name prints `unknown case: <name>` and exits 2.
- Removing the `ctest` registrations changed no compiled target and no
  numerical code path. Each case runs exactly as it did under `ctest`.
- Fixture purposes are documented in the comments of
  `tests/streamfunctions/streamfunction_operator_test_cases.hpp`.

## How to run a case (detached V100 job)

Always use an increment-scoped mirror (`--increment SF-NN`). Build first if
the mirror has no `build/v100-release`. If the build may take longer than
about a minute, run it as a `run` job instead of `exec`.

```bash
scripts/remote --increment SF-NN sync
scripts/remote --increment SF-NN exec -- "cmake --preset v100-release && cmake --build build/v100-release -j"
scripts/remote --increment SF-NN run <job> -- "./build/v100-release/streamfunction_operator_tests --case <name>"
scripts/remote --increment SF-NN wait <job>
```

Rules:

- Never run these cases locally or through `scripts/remote exec`.
- One job runs per GPU, and at most two run on the host at once (host-wide
  locks). Use `REMOTE_GPU_WAIT=<seconds>` to queue for a GPU.
- Do not `sync` the same increment mirror while one of its jobs is `running`
  or `waiting_gpu`.
- Two cases need multiple hours each and should be separate jobs:
  - `heterogeneity_smoke_sigma025` and `heterogeneity_smoke_sigma1`: about
    5.3 h each (18 996 s and 19 056 s);
  - `coupled_residual_gauge_recombination_sigma025`: about 5.4 h (19 530 s).
- Record every result (head, build preset, job name, log path, outcome) in an
  experiment note or the increment bitácora. A red outcome is a recorded
  science result, not a `ctest` failure.

Wall times below are from the V100 run at `58898bf`. When a former entry had
two cases, the time covers both.

## Cases

### `anderson_stall_fixture_a`, `anderson_stall_fixture_b`

- Former ctest entry: `streamfunction_anderson_stall`
- Purpose: SF-20 Anderson stall fixtures. Each fixture runs a full
  500-iteration 32^3 Picard budget twice: a control with Anderson disabled and
  a gate run with Anderson (depth 5). The control is meant to stall and the
  Anderson run to converge. Fixture a: sigma^2 = 1, lambda* = 0.1125.
  Fixture b: sigma^2 = 0.25, lambda* = 0.386.
- Commands:

  ```bash
  scripts/remote --increment SF-NN run anderson-stall-a -- "./build/v100-release/streamfunction_operator_tests --case anderson_stall_fixture_a"
  scripts/remote --increment SF-NN wait anderson-stall-a
  scripts/remote --increment SF-NN run anderson-stall-b -- "./build/v100-release/streamfunction_operator_tests --case anderson_stall_fixture_b"
  scripts/remote --increment SF-NN wait anderson-stall-b
  ```

- Wall time: 744.13 s for both fixtures.
- Last recorded outcome (SF-26, `58898bf`, 2026-10-01): **FAILED — recorded
  science gate (D6).** On the corrected same-index system both arms stagnate.
  Fixture a: control r_F 2.51e-4, Anderson 2.48e-4. Fixture b: control
  7.51e-4, Anderson 7.41e-4. Not re-baselined: the fixture assumes a stall
  that Anderson cures, and on the corrected system both arms stall.

### `heterogeneity_smoke_sigma025`, `heterogeneity_smoke_sigma1`

- Former ctest entry: `streamfunction_heterogeneity_smoke`
- Purpose: SF-21 prespecified 32^3 physical Gaussian gating smokes. Each runs
  the production heterogeneity-continuation driver
  (`run_streamfunction_heterogeneity_continuation`) through many lambda and
  eta-rescue stages. Seed 12345, ell = 8, epsilon 1e-2, Anderson and Newton
  enabled. Variants: sigma^2 = 0.25 and sigma^2 = 1.
- Commands:

  ```bash
  scripts/remote --increment SF-NN run hetero-smoke-s025 -- "./build/v100-release/streamfunction_operator_tests --case heterogeneity_smoke_sigma025"
  scripts/remote --increment SF-NN wait hetero-smoke-s025
  scripts/remote --increment SF-NN run hetero-smoke-s1 -- "./build/v100-release/streamfunction_operator_tests --case heterogeneity_smoke_sigma1"
  scripts/remote --increment SF-NN wait hetero-smoke-s1
  ```

- Wall time: 38 055.82 s for both variants. Multi-hour.
- Last recorded outcome (SF-26, `58898bf`, 2026-10-01): **FAILED — recorded
  science gate (D6).** Both end with `lambda_floor_exhausted`: at
  lambda = 0.0125 for sigma^2 = 0.25 (38/66 stages accepted) and at
  lambda = 0 for sigma^2 = 1 (37/65 accepted). Every eta = 1 stage stops at
  r_F = 1.50e-6, 1.5x the 1e-6 stage tolerance.

### `newton_difficult_case`

- Former ctest entry: `streamfunction_newton_difficult`
- Purpose: SF-24 G4/G5 Newton-Krylov difficult-case control. Two 32^3 solves
  of the SF-21 heterogeneity-continuation stall fixture (an SF-20
  Anderson-gate control and a run with Newton added), plus a determinism
  rerun.
- Commands:

  ```bash
  scripts/remote --increment SF-NN run newton-difficult -- "./build/v100-release/streamfunction_operator_tests --case newton_difficult_case"
  scripts/remote --increment SF-NN wait newton-difficult
  ```

- Wall time: 1 401.58 s.
- Last recorded outcome (SF-26, `58898bf`, 2026-10-01): **FAILED — recorded
  science gate (D6).** 50 accepted Newton steps, r_F 1.77e-4 -> 1.73e-4,
  5351 Jv, not converged. Not re-baselined, for the same reason as the
  Anderson stall fixtures.

### `terminal_dgate_diagnostic`

- Former ctest entry: `streamfunction_terminal_dgate`
- Purpose: SF-25 D-gate protocol E2-E5. Freezes a sigma^2 = 1, 32^3
  Picard/Anderson plateau state, then runs the prespecified mu-sweep,
  spectral probe, and LM mini-solve. **Always-pass evidence recorder** since
  SF-26 T03: read its printed output, not its exit code.
- Commands:

  ```bash
  scripts/remote --increment SF-NN run terminal-dgate -- "./build/v100-release/streamfunction_operator_tests --case terminal_dgate_diagnostic"
  scripts/remote --increment SF-NN wait terminal-dgate
  ```

- Wall time: 2 538.33 s.
- Last recorded outcome (SF-26, `58898bf`, 2026-10-01): passed (evidence
  recorder). The Newton-disabled sigma^2 = 1 continuation also dies at
  lambda = 0; the direct lambda = 0.5125 stage stagnates at r_F 5.4e-3.

### `terminal_resolution_probe`

- Former ctest entry: `streamfunction_terminal_resolution`
- Purpose: SF-25 C05 resolution-discriminating experiment. Two direct 64^3
  zero-source solves at amplitude 0.5125 (sigma^2 = 1): ell/h = 16 against an
  ell/h = 8 control. **Always-pass print-only evidence recorder.**
- Commands:

  ```bash
  scripts/remote --increment SF-NN run terminal-resolution -- "./build/v100-release/streamfunction_operator_tests --case terminal_resolution_probe"
  scripts/remote --increment SF-NN wait terminal-resolution
  ```

- Wall time: 1 708.82 s.
- Last recorded outcome (SF-26, `58898bf`, 2026-10-01): passed (evidence
  recorder). r_F 1.22e-3 at ell/h = 16; the other reported values are 4.69e-3
  and 6.88e-4. Not converged.

### `coupled_residual_gauge_recombination_sigma025`

- Former ctest entry: `streamfunction_gauge_recombination_heavy`
- Purpose: SF-26 T02 heavy gauge-recombination case. Converges the verbatim
  sigma^2 = 0.25, 32^3 heterogeneity-smoke fixture through the production
  continuation driver, then compares the same-index production residual with
  a test-local crossed recomposition on a gauge-recombined state of the
  converged fields.
- Commands:

  ```bash
  scripts/remote --increment SF-NN run gauge-recomb-s025 -- "./build/v100-release/streamfunction_operator_tests --case coupled_residual_gauge_recombination_sigma025"
  scripts/remote --increment SF-NN wait gauge-recomb-s025
  ```

- Wall time: 19 530.09 s. Multi-hour.
- Last recorded outcome (SF-26, `58898bf`, 2026-10-01): **FAILED — recorded
  science gate (D6).** Precondition not met: the sigma^2 = 0.25 continuation
  no longer reaches lambda = 1, so there is no converged state to recombine.

## Summary table (V100, `58898bf`, 2026-10-01)

| former ctest entry | `--case` names | last outcome | wall (s) |
|---|---|---|---|
| `streamfunction_anderson_stall` | `anderson_stall_fixture_a`, `anderson_stall_fixture_b` | FAILED (D6 science gate) | 744.13 |
| `streamfunction_heterogeneity_smoke` | `heterogeneity_smoke_sigma025`, `heterogeneity_smoke_sigma1` | FAILED (D6 science gate) | 38 055.82 |
| `streamfunction_newton_difficult` | `newton_difficult_case` | FAILED (D6 science gate) | 1 401.58 |
| `streamfunction_terminal_dgate` | `terminal_dgate_diagnostic` | passed (always-pass recorder) | 2 538.33 |
| `streamfunction_terminal_resolution` | `terminal_resolution_probe` | passed (always-pass recorder) | 1 708.82 |
| `streamfunction_gauge_recombination_heavy` | `coupled_residual_gauge_recombination_sigma025` | FAILED (D6 science gate) | 19 530.09 |

Total for the six: 63 979 s of 66 700 s. The 15 entries that stay in `ctest`
all passed and took about 2 721 s on V100. That is still long-duration under
`remote-v100.md` Section 0, so the authoritative full-suite pass stays a
detached V100 job.

## Caveats

- Outcomes are transcribed from the SF-26 record. This note did not re-run any
  case, and nothing was re-baselined or reinterpreted.
- All eight cases exercise the frozen periodic-fluctuation stack. Its periodic
  solution on generic Gaussian fields is not the Darcy flow
  (`docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md`), so
  these results characterize that stack, not the Darcy-labels target.
- A new per-increment mirror has no build directory. Configure and build
  before the first job.
