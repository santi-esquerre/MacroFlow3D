# PSPTA Standby Before KH Reconstruction

## Purpose

This record freezes the current PSPTA / Strategy A / Strategy C line before starting
the KH potential-flow reconstruction experiment. It is an experiment-control note,
not a PSPTA implementation plan.

The new KH line must compare velocity evaluation backends without continuing,
optimizing, or partially reworking the current invariant-recovery and
pseudo-symplectic tracking path.

## Standby date

2026-05-13

## Repository state

- MacroFlow3D commit: `4c9c8ca9f6b03b0448603a941aa123a3072330e4`
- Commit summary: `chore: standardize remote execution workflow`
- Experimental KH worktree: `.agents/worktrees/feature-kh-potential-flow-interpolation`
- Experimental KH branch: `feature/kh-potential-flow-interpolation`
- PSPTA safeguard branch: `pspta-standby-before-kh-reconstruction`
- Par2_Core submodule commit: `ccc7f2e26fe92b449b3c048701bdc82b475af66f`

Relevant local branches at standby time:

- `master`
- `science/pspta-pipeline-map`
- `science/pspta-gate3-gauge`
- `wip/pspta-gate3-checkpoint`
- `codex_pspta_phase1`
- `chore/remote-runner-standardization`
- `pspta-standby-before-kh-reconstruction`
- `feature/kh-potential-flow-interpolation`

No local git tags were present at standby time.

## PSPTA scope frozen

The following PSPTA-related line is frozen for the KH phase:

- Strategy A invariant recovery through the transport-operator eigenproblem
  `A = D^T W D + mu L`.
- PETSc/SLEPc eigensolver integration for invariant recovery.
- `PsptaInvariantField` metadata and Strategy A eigenvector ingestion.
- Strategy C alternating fit plus Poisson projection refinement design.
- `RefinementAC` and `GaugeFixer` implementation work.
- PSPTA advance-plus-project particle transport.
- PSPTA invariant quality diagnostics and Gate 3 acceptance work.
- The active PSPTA operational plan in
  `docs/plans/active/pspta-execution-plan.md`.

Representative files in the frozen line:

- `src/physics/particles/pspta/PsptaEngine.*`
- `src/physics/particles/pspta/PsptaPsiField.*`
- `src/physics/particles/pspta/invariants/PsptaInvariantField.*`
- `src/physics/particles/pspta/invariants/TransportOperator3D.*`
- `src/physics/particles/pspta/invariants/EigensolverBackend.*`
- `src/physics/particles/pspta/invariants/SLEPcBackend.*`
- `src/physics/particles/pspta/invariants/RefinementAC.*`
- `src/physics/particles/pspta/invariants/GaugeFixer.*`
- `src/physics/particles/pspta/invariants/OperatorTestHarness.*`
- `apps/analyze_invariant_quality.cu`
- `apps/benchmark_eigensolver.cu`
- `apps/compare_eigensolver_backends.cu`
- `apps/validate_slepc_eigensolver.cu`
- `apps/config_pipeline_pspta.yaml`
- `apps/config_pspta_small.yaml`

## Existing artifacts

The clean experimental worktree did not contain committed run artifacts under
`artifacts/` or `output*/`.

Known generated artifact path from the current code:

- `artifacts/gate3/invariant_quality_summary.csv`
- `artifacts/gate3/invariant_quality_rotation_scan.csv`
- `artifacts/gate3/invariant_quality_localization.csv`

Those files are produced by `apps/analyze_invariant_quality.cu` when the
appropriate PETSc/SLEPc-enabled path is built and run. They are not present in
this standby worktree.

## What must not change during the KH phase

Do not modify the frozen PSPTA line while testing KH reconstruction unless a new
task explicitly reopens PSPTA work:

- Do not change invariant semantics, gauge conventions, refinement metadata, or
  projection behavior.
- Do not continue Strategy A eigensolver tuning.
- Do not continue Strategy C refinement implementation.
- Do not alter PSPTA Newton projection, status codes, or invariant buffers.
- Do not wire KH reconstruction through PSPTA as a hidden dependency.
- Do not compare KH as if it were equivalent to PSPTA or to exact invariant
  preservation.

The KH experiment may read PSPTA theory notes for scientific interpretation, but
its implementation should live in the Par2/MacroFlow3D baseline-transport path
and in explicit KH experiment scripts/docs.

## KH phase boundary

The KH hypothesis is narrower than PSPTA:

```math
q(x) = -K(x) \nabla h(x)
```

KH reconstruction tests whether evaluating smooth scalar fields `K` and `h`
before reconstructing velocity reduces numerical helicity and transverse
macrodispersion relative to direct face-velocity interpolation. It does not
recover the two invariants `psi1`, `psi2`, does not enforce streamsurface
confinement, and does not replace pseudo-symplectic tracking.

Correct scientific wording during this phase:

> KH_POTENTIAL_RECONSTRUCTION may preserve the helicity-free structure associated
> with continuous isotropic Darcy flow better than direct face-velocity
> interpolation, but it is not PSPTA and does not guarantee exact preservation of
> the invariants.

## How to resume PSPTA later

1. Check out or create a worktree from
   `pspta-standby-before-kh-reconstruction`, `science/pspta-gate3-gauge`, or the
   latest agreed PSPTA branch.
2. Re-read:
   - `AGENTS.md`
   - `src/physics/particles/pspta/AGENTS.md`
   - `docs/plans/active/pspta-execution-plan.md`
   - `docs/theory/lester-2023-key-claims.md`
   - `docs/validation/acceptance-gates.md`
3. Confirm the resumed task belongs to the active PSPTA phase before editing.
4. Rebuild the relevant baseline:

   ```bash
   cmake -S . -B build/wsl-debug -G Ninja \
     -DCMAKE_BUILD_TYPE=Debug \
     -DCMAKE_CUDA_ARCHITECTURES=86 \
     -DMACROFLOW3D_ENABLE_PETSC=OFF
   cmake --build build/wsl-debug -j
   ctest --test-dir build/wsl-debug --output-on-failure -R operator_tests
   ```

5. For PETSc/SLEPc or production-like validation, use the remote workflow:

   ```bash
   scripts/remote sync
   scripts/remote exec -- "cmake --preset v100-petsc && cmake --build build/v100-petsc -j && ctest --test-dir build/v100-petsc --output-on-failure -R validate_slepc_eigensolver"
   ```

6. Resume with Gate 3 / Gate 4 evidence before making any scientific claim about
   invariant preservation or transverse macrodispersion.
