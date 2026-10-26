# Codebase audit and cleanup before SF-39 (owner decisions A1-A10)

- Status: accepted
- Date: 2026-10-26
- Plan: [`docs/plans/active/audit-cleanup-2026-10.md`](../plans/active/audit-cleanup-2026-10.md)

## Context

After SF-41 (merged 2026-10-26, #65/#66) the repository carries code from alternatives the project no longer
follows: the PETSc/SLEPc eigensolver backend and the legacy PSPTA invariant/transport engine (retired from the
consumer path by decision 2026-10-17), the pre-refactor `legacy/` tree, vendored yaml-cpp tests, dead headers and
functions. A read-only audit (2026-10-26, at `3452a7d`) also found integrity defects on the path of the coming SF-39
campaign: unchecked `fclose`/`ferror` in the realization writers before `DONE` is written; kernels in
`src/physics/flow/velocity_from_head.cu` launched on the legacy default stream while the context stream is
`cudaStreamNonBlocking`; a Release-only silent `int` truncation in `dot.cu`/`nrm2.cu`; the git revision captured at
configure time and recorded as `unknown` on the V100 (the mirror has no `.git`); a CI workflow that never ran; and
11 ctest entries that re-run cases of `streamfunction_operator_tests` (~1359 s of a 3182 s V100 ctest).

The owner wants the project to stay a complete pipeline suite (generate the field, solve the flow, transport), to
keep every experiment note and scientifically relevant datum, and to prioritize memory, then time.

Classification of each item below: **accepted scope** unless marked otherwise.

## Decision

- **A1. PETSc/SLEPc retired.** Code, apps, presets (`wsl-petsc-debug`, `v100-petsc`), build script and
  `docs/runbooks/petsc-slepc.md` are removed; the required-docs and presets hooks stop requiring them.
- **A2. Legacy PSPTA engine retired** in two steps: (A) invariants/eigensolver stack and PSPTA-only apps/tests with
  no output change; (B) engine, pipeline `transport.method: pspta`, its config fields and configs. `docs/theory/**`
  is untouched (there "PSPTA" names Lester's method). Kept: the periodic streamfunction stack SF-02..SF-26, the
  SF-33 slab Newton-Krylov, `closure_gate`, `spurious_spreading`, `streamline_tracker`, Par2.
- **A3. `legacy/` archived.** Annotated tag `archive/pre-cleanup-2026-10` (commit `3452a7d`, the full pre-cleanup
  tree) and branch `archive/legacy` (`git subtree split --prefix=legacy`, history kept). On `master` only
  `legacy/README.md` and the cited input data `legacy/input*/` remain. Citations `legacy/<path>:<line>` in frozen
  records resolve with `git show archive/pre-cleanup-2026-10:legacy/<path>`; no frozen record is edited.
- **A4. Regression set v2.** The byte-compare set of `scripts/regress` (pipeline configs `config_par2_small`,
  `config_streamfunctions_homogeneous`, `config_streamfunctions_continuation`; the 20-run `inlet_slab` R set; runner
  medium (a), A/B, runner medium (b)) replaces the "SF-27 trio" for every increment or PR activated after step A2(B)
  is merged. Records of `done` increments are unchanged. `config_par2_small` (heterogeneous, Par2) replaces
  `config_pspta_small`, the only heterogeneous member of the trio; both run side by side in A2(A) as the overlap.
- **A5. Gates.** Gate 1 smoke -> `apps/config_par2_small.yaml`; Gate 2 -> `ctest -R '^streamfunction_operator_tests$'`
  (the PSPTA `operator_tests` is removed); Gate 3 (PSPTA) retired. No renumbering: frozen specs cite "Gate 3A".
- **A6. Expected diff of A2(B).** In every `effective_config.yaml` the `pspta_*` block disappears; `config_pspta_small`
  leaves the set. Any other difference is a defect. Later comparisons across this boundary apply the same filter.
- **A7. Bitwise rule.** Performance changes in `src/physics`, `src/numerics`, `src/multigrid` must leave every output
  byte unchanged (base vs head byte-compare plus a production-grid realization). Non-bitwise ideas are recorded as
  proposals in the plan and not implemented.
- **A8. ctest.** The 11 subset entries are removed; the umbrella `streamfunction_operator_tests` runs every fast case;
  one group runs with `streamfunction_operator_tests --case <name>`.
- **A9. Register.** The plan keeps a register of rejected and deferred items with their reasons, so they are not
  re-proposed without new evidence.
- **A10. Base references.** Base outputs are produced by the base commit's binaries driven by the head tree's
  harness and configs, on a per-PR V100 mirror, and reused while the base commit is unchanged.

Sequence: the cleanup and the must-have fixes (AUD-00..AUD-06, AUD-10..AUD-14, AUD-20) merge before SF-39 activates;
remaining quality work and bitwise performance (including the medium (b) buffer placement that lowers the solve peak
from 60.57 to 52.57 B/cell) follow SF-39.

## Consequences

- SF-39 activates on a smaller tree with checked writers, ordered streams and V100 build provenance.
- The V100 full ctest drops to ~30 minutes; the CI runs the harness hooks on every PR to `master`.
- Pipeline outputs change only in `effective_config.yaml` (A6) and in the excluded provenance fields
  (`manifest.json` `build.git_hash`, `timings.json` `commit`).
- PETSc/SLEPc and PSPTA results remain reproducible only from the archive tag.
