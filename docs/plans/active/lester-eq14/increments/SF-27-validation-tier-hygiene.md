# SF-27 — Validation-tier hygiene

- State: `pending`
- Goal: `Dejar en ctest solo contratos rápidos, moviendo los barridos multi-caso y smokes científicos a experimentos documentados sin cambiar ningún resultado numérico.`
- Depends on: `SF-26`
- Unlocks: `SF-28`
- Branch: `chore/lester-sf27-validation-tier-hygiene`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 0 + Gate 1`
- Human review: `required`
- Owner: `unassigned`
- Started: `not started`
- Completed: `not completed`
- PR: `not opened`
- Commit: `not recorded`

## Scientific or engineering intent

`ctest` must hold only fast contract tests; multi-case solver sweeps and science smokes become documented experiments run as detached V100 jobs (decision D6 disposition and the 2026-10-02 process rule). The increment establishes no scientific claim.

Context: `docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md` and `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`.

## Preconditions

- `SF-26` is `done` on the default branch.
- The six heavy `add_test` entries exist in `CMakeLists.txt` (lines about 464-579) and their `--case` executables in `streamfunction_operator_tests`.

## In scope

- Remove from `CMakeLists.txt` the six heavy entries `streamfunction_anderson_stall`, `streamfunction_heterogeneity_smoke`, `streamfunction_newton_difficult`, `streamfunction_terminal_dgate`, `streamfunction_terminal_resolution`, `streamfunction_gauge_recombination_heavy`.
- Keep their `--case` executables runnable from `streamfunction_operator_tests`.
- Document them in a short `docs/experiments/` index note: each case, its purpose, the detached-V100 command, and the last recorded outcome/red status (reference: the SF-26 experiment note).
- Update `docs/validation/eval-tiers.md` and the CMake comments accordingly.

## Out of scope

- Any change to test-code semantics, to the solver, or to configs.

## Files and symbols

- `CMakeLists.txt`
- `docs/validation/eval-tiers.md`
- `docs/experiments/<new index note>`
- `tests/streamfunctions/streamfunction_operator_tests.cpp` only if the heavy-case isolation mechanism needs a comment update.

## Implementation specification

1. Record the baseline outputs of the three default configs (`apps/config_pspta_small.yaml`, `apps/config_pipeline_par2.yaml`, `apps/config_pipeline_pspta.yaml`) with a detached V100 job before the change.
2. Remove the six `add_test` entries and update comments; verify with `ctest -N`.
3. Write the experiments index note with, for each case, the command `scripts/remote run <job> -- "./build/v100-release/streamfunction_operator_tests --case <name>"`.
4. Update `docs/validation/eval-tiers.md` so that ctest is described as fast contract tests only.
5. Re-run the three configs after the change and byte-compare outputs; run the full ctest on V100 and record wall time in the bitácora.

## Expected numerical effect

No numerical change anywhere. The ctest registry shrinks by six entries; every remaining test behaves identically.

## Validation commands

```bash
cmake --preset wsl-debug && cmake --build build/wsl-debug -j && ctest --test-dir build/wsl-debug -N
scripts/remote sync
scripts/remote exec -- "cmake --preset v100-release && cmake --build build/v100-release -j"
scripts/remote run sf27-ctest-full -- "ctest --test-dir build/v100-release --output-on-failure"
scripts/remote wait sf27-ctest-full
# byte-compare of the three default configs before/after as a detached job (scripts/remote run ... + wait)
bash scripts/hooks/check-lester-increments.sh
```

## Acceptance thresholds

- `ctest -N` no longer lists the six entries.
- Full suite green on V100; wall time recorded in the bitácora (baseline 66 700 s with 4 red; target all green, duration recorded, no threshold).
- The six cases still run by `--case`.
- Byte-compare of the three default configs' outputs is clean; any manifest/CSV difference beyond timestamps is a failure.

## Regression surface

- Every remaining ctest entry; the documented experiment commands; validation tier documentation.

## Failure and rollback policy

- A red remaining test or a byte-compare difference leaves the increment active; revert the CMake change and investigate.

## Completion checklist

<!-- completion-checklist:start -->
- [ ] Implementation matches the scope and contains no unrelated changes.
- [ ] Targeted validation passes and its evidence is recorded.
- [ ] Required regression tests pass.
- [ ] Scientific or engineering findings are appended to the bitácora.
- [ ] Required human review is recorded.
- [ ] PR and commit identifiers are recorded.
- [ ] The master checklist entry is checked in this branch and `check-lester-increments.sh` passes.
<!-- completion-checklist:end -->

## Advancement rule

`SF-28` becomes eligible after this increment is merged and marked `done` on the default branch.

## Bitácora

Append entries; do not rewrite prior observations.  Store large raw outputs as
artifacts or experiment notes and link them here.

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-10-02T00:00Z | not started | Specification created by the 2026-10-02 re-sequencing (decision record `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`). | Replaces the cancelled SF-27..SF-30 specifications (git history at `4670fb5`). | Activate only when the checker reports it READY. |
