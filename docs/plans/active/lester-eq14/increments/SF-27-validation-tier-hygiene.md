# SF-27 — Validation-tier hygiene

- State: `done`
- Goal: `Dejar en ctest solo contratos rápidos, moviendo los barridos multi-caso y smokes científicos a experimentos documentados sin cambiar ningún resultado numérico.`
- Depends on: `SF-26`
- Unlocks: `SF-28`
- Branch: `chore/lester-sf27-validation-tier-hygiene`
- Worktree: `Claude-managed per-node isolated worktrees`
- Acceptance gate: `Gate 0 + Gate 1`
- Human review: `required`
- Owner: `Claude Fable 5.1 orchestrator session (2026-10-02)`
- Started: `2026-10-02T23:10Z on master=81cd612`
- Completed: `2026-10-03T01:42Z (owner approval; closure metadata commit on PR #44)`
- PR: `#44` (https://github.com/santi-esquerre/MacroFlow3D/pull/44), delivery branch `chore/lester-sf27-validation-tier-hygiene`
- Commit: `1446084` (audited source-bearing head: integration `f3e9878` = N1 `a4177d8` + C1 `0a89f09` + N2 `f3e9878` on harness base `21be8a0`, plus harness follow-up `1446084`; later commits are docs/metadata only)

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
- [x] Implementation matches the scope and contains no unrelated changes.
- [x] Targeted validation passes and its evidence is recorded.
- [x] Required regression tests pass.
- [x] Scientific or engineering findings are appended to the bitácora.
- [x] Required human review is recorded.
- [x] PR and commit identifiers are recorded.
- [x] The master checklist entry is checked in this branch and `check-lester-increments.sh` passes.
<!-- completion-checklist:end -->

## Advancement rule

`SF-28` becomes eligible after this increment is merged and marked `done` on the default branch.

## Bitácora

Append entries; do not rewrite prior observations.  Store large raw outputs as
artifacts or experiment notes and link them here.

| UTC | Commit/state | Observation or action | Evidence/decision | Next action |
|---|---|---|---|---|
| 2026-10-02T00:00Z | not started | Specification created by the 2026-10-02 re-sequencing (decision record `docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`). | Replaces the cancelled SF-27..SF-30 specifications (git history at `4670fb5`). | Activate only when the checker reports it READY. |
| 2026-10-02T23:10Z | activation on `master=81cd612` (PR #43 merged; checker OK ready=SF-27 SF-29, nonterminal=none); delivery branch `chore/lester-sf27-validation-tier-hygiene` | UNDERSTAND: SF-27 chosen over the co-READY SF-29 (first in master-checklist order and on the critical path SF-27->SF-28->SF-30/31->SF-32; SF-29 is CPU-only and can run in a second session). Source is byte-identical to the SF-26 audited head `58898bf` (sha256 of `src apps tests CMakeLists.txt CMakePresets.json` equal locally and on the legacy V100 mirror). The six entries sum to 63 979 s of the 66 700 s SF-26 suite (`anderson_stall` 744, `heterogeneity_smoke` 38 056, `newton_difficult` 1 402, `terminal_dgate` 2 538, `terminal_resolution` 1 709, `gauge_recombination_heavy` 19 530); the 15 remaining entries (~2 721 s) were all green. Dispatch mechanism: `heavy_cases()` fallback in `streamfunction_operator_tests.cpp::main()`, untouched by any `add_test` removal. DAG: N1 (CMake registry + comment-only test-source updates) in parallel with N2 (experiment index note + `eval-tiers.md`); one integrator; orchestrator-owned detached V100 jobs on the per-increment mirror (`scripts/remote --increment SF-27`, PR #43): `sf27-base-build-refs` before the change, then `sf27-ctest-full`, `sf27-bytecompare`, `sf27-case-*` after it. | Recorded deviations: (a) the spec's byte-compare names `config_pipeline_par2.yaml`/`config_pipeline_pspta.yaml`, which are 2048x256x256, 409 600-step production runs (multi-day on one V100); the increment uses the SF-25/SF-26 precedent trio `config_pspta_small`, `config_streamfunctions_homogeneous`, `config_streamfunctions_continuation` (the CMake change touches no compiled target, so no config output can change by construction) — for the human reviewer. (b) Comment-only edits in `tests/streamfunctions/*` that name the removed ctest entries are treated as in-scope stale-comment fixes (no test-code semantics change). (c) The scientific-rigor skill is not invoked: no scientific claim or numerical change is involved. Human-review increment (spec). | Launch N1 and N2 workers; run `sf27-base-build-refs` on V100. |
| 2026-10-03T00:00Z | harness detour on the delivery branch: `21be8a0`, `1446084` (`scripts/remote.env` only) | EXECUTE: the first `scripts/remote --increment SF-27 sync` produced a mirror without `src/io/output_layout.hpp` (job `sf27-base-build-refs` #1 failed: `fatal error: ../output_layout.hpp`): the unanchored rsync exclude `output_*` matched that tracked file in any path component; the legacy mirror only built because it kept a copy synced before the exclusion existed (sha256 identical to HEAD, so SF-25/SF-26 V100 evidence is unaffected). Fix 1 (`21be8a0`) anchored the patterns, which then let rsync copy the `build/` trees of the local agent worktrees (`.claude/worktrees`, 13 GB); fix 2 (`1446084`) uses directory-only patterns (`build/`, `build-*/`, `output/`, `output_*/`) and excludes `.claude`/`.agents`. Operational error recorded: one post-change sync + build (`sf27-build`, 23:26Z) ran against the LEGACY mirror `~/MacroFlow3D` because `REMOTE_INCREMENT` was not exported after a failed step in a chained command; no other increment was using it (SF-29 is CPU-only), it compiled the post-change tree with 0 errors and listed 15 tests; every subsequent call passes `--increment SF-27` explicitly. | N1 `c0e3016` (CMake: six add_test blocks removed, comment-only edits in 3 test sources; `ctest -N` = 15) ACCEPTED; corrective C1 `f4fc753` (comment-only, `heterogeneity_continuation_gpu_cases.cu` header) ACCEPTED; N2 `8e687f9` (index note + `eval-tiers.md` + README pointer) ACCEPTED. Integrator: three cherry-picks, patch-ids identical, no conflicts, `f3e9878`. Audits under `.claude/orchestration/SF-27-validation-tier-hygiene/audits/`. | V100 evidence on the per-increment mirror. |
| 2026-10-03T01:30Z | FINAL_AUDIT positive on `1446084`; State -> `awaiting_review` (human-review increment) | V100 (`scripts/remote --increment SF-27`, mirror `~/MacroFlow3D-SF-27`, preset `v100-release`, detached jobs): `sf27-base-build-refs` on the PRE-change tree `21be8a0` (synced from a detached worktree; 123 objects, `ctest -N` = 21, three configs exit 0 into `~/sf27_base_refs`); `sf27-build` + forced `sf27-rebuild` of the 5 touched files on the post-change tree (22 objects, 0 errors, `ctest -N` = 15); **`sf27-ctest-full`: 15/15 passed, `Total Test time (real) = 2709.82 s`** (baseline 66 700.16 s with 4 red; the six removed entries accounted for 63 979 s); `sf27-bytecompare`: `config_pspta_small`, `config_streamfunctions_homogeneous`, `config_streamfunctions_continuation` IDENTICAL excluding manifest, manifests identical modulo timestamp lines (spec deviation (a) of the activation row: the literal `config_pipeline_par2/pspta` are 2048x256x256 x 409 600-step production runs, not run); `--case` evidence on the post-change binary: `anderson_stall_fixture_a/b` FAILED with r_F bit-identical to SF-26 (2.5131715e-4/2.4841275e-4; 7.5080154e-4/7.4111595e-4), `newton_difficult_case` FAILED identical (1.7346282e-4, 5351 Jv, determinism PASS), `terminal_dgate_diagnostic` PASS (recorder), `terminal_resolution_probe` PASS (recorder; R1a 1.2178e-3, R1b 4.6934e-3, R2a 6.8847e-4 as in SF-26), `sf27-case-dispatch`: `heterogeneity_smoke_sigma025/sigma1` and `coupled_residual_gauge_recombination_sigma025` accepted and running at 180 s (exit 124), negative control exit 2, `--list` contains none of the eight heavy names (158 fast cases). Local: `cmake --preset wsl-debug` + `ctest -N` = 15 in the control checkout and in the integrator worktree; checker OK. Full diff from `81cd612`: 10 files; zero changes under `src/`/`apps/`; non-comment diff in `tests/` + `CMakeLists.txt` is exactly the six removed `add_test` blocks. Logs archived under `.claude/orchestration/SF-27-validation-tier-hygiene/integration/`. | Acceptance thresholds: (1) `ctest -N` without the six: PASS; (2) full suite green on V100, wall time recorded: PASS (2 709.82 s); (3) six cases still run by `--case`: PASS (four re-executed in full, two by dispatch check + unchanged test code); (4) byte-compare clean: PASS on the precedent trio. Gate 0 + Gate 1 satisfied. Review items left open for the human: the byte-compare trio deviation, the `scripts/remote.env` harness fix carried on this branch (`21be8a0`, `1446084`), and the comment-only edits in `tests/`. | Push the delivery branch and open the PR; stop at AWAIT_HUMAN_REVIEW; do not set `done` or check the master entry. |
| 2026-10-03T01:45Z | PR #44 opened at head `906105c` (docs/metadata above the audited source head `1446084`) | PUBLISH_PR: delivery branch pushed; PR body lists Goal, DAG, jobs, acceptance evidence and the three recorded deviations for the reviewer. Human-review increment: `done` and the master-checklist entry are deferred to the closure-only metadata commit after explicit approval of head `1446084`. | Checklist items 1-4 and 6 checked; items 5 and 7 open. `check-lester-increments.sh` OK (nonterminal=SF-27, ready=SF-29). | AWAIT_HUMAN_REVIEW. On approval: closure metadata commit on this PR; after merge, `scripts/remote remove-mirror SF-27`. |
| 2026-10-03T01:42Z | closure metadata commit on PR #44; State -> `done` | Human review recorded: explicit owner approval ("Aprobado") given in the orchestrator session on 2026-10-03 for PR #44 at head `1efd310`, whose audited source-bearing head is `1446084` (verified before closure: `git diff 1446084 origin/<branch> -- . ':!docs'` empty; PR OPEN, not merged). No GitHub review object is claimed. | Checklist complete (7/7); master-checklist entry checked in this branch; `check-lester-increments.sh` OK. No source, test, config, or numerical change in this commit. | READY_FOR_HUMAN_MERGE. After merge: SF-28 becomes READY on the default branch; `scripts/remote remove-mirror SF-27`. |
