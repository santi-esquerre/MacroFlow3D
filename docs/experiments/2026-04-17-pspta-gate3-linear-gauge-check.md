# Gate 3 experiment — linear gauge readiness check

- Date: 2026-04-17
- Branch: `science/pspta-gate3-gauge`
- Parent plan: `docs/plans/active/2026-04-16-pspta-gate3-gauge-iteration.md`
- Scope: Gate 3 diagnostics only. No transport, solver, or refinement semantics changed.

## Question

Can a diagnostic-only gauge stage consisting of:

- mean removal,
- in-subspace orthogonal rotation,
- deterministic sign/orientation,
- and a global determinant/scaling fit,

make `||v - ∇ψ1 × ∇ψ2||` scientifically interpretable on the controlled cases
`uniform_x` and `layered_x`, and therefore on `darcy_small`?

## Hypothesis

If the mismatch problem is mostly a linear gauge/scale artifact, then:

- `uniform_x` and `layered_x` should collapse from `rel_residual ≈ 1` to near zero after gauge,
- the residual floor should become small on the controls,
- and `darcy_small` should become interpretable without touching refinement.

If this does **not** happen on the controls, then the remaining blocker is not just linear gauge.

## Files changed for the experiment

- `apps/analyze_invariant_quality.cu`
- `CMakeLists.txt`

## What was implemented

In `apps/analyze_invariant_quality.cu`:

- added `GaugeReadyMetrics` and `GaugeReadyEvaluation`,
- added a linear gauge-ready stage:
  - rotate within the recovered 2D subspace,
  - subtract means,
  - fit `alpha_opt = <v,c> / ||c||^2` with `c = ∇ψ1 × ∇ψ2`,
  - apply a symmetric scale/sign transform so `det(M) = alpha_opt`,
- added per-mode transport vs regularization energy decomposition for the **actual current operator** `A = D†D + μL`,
- added refined localization output:
  - `x_interior` vs `x_boundary_halo`,
  - `x_slice_i`,
  - low/high `|∇ψ1 × ∇ψ2|`,
  - degenerate vs nondegenerate regions,
- added new CSV artifacts:
  - `artifacts/gate3/invariant_quality_gauge.csv`
  - `artifacts/gate3/invariant_quality_energy.csv`
  - `artifacts/gate3/invariant_quality_localization_v2.csv`

In `CMakeLists.txt`:

- added the `analyze_invariant_quality` PETSc app target,
- added optional `X11` / `Xau` link dependencies to the imported `petsc` target to match the remote static PETSc installation.

## Commands run

Local attempt:

```bash
cmake -S . -B /tmp/mf3d-pspta-gate3-wsl-debug -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DMACROFLOW3D_ENABLE_DIAGNOSTICS=ON \
  -DMACROFLOW3D_ENABLE_PROFILING=OFF \
  -DMACROFLOW3D_ENABLE_NVTX=OFF \
  -DMACROFLOW3D_ENABLE_PETSC=OFF
```

Observed:

- local configure from the worktree failed before project configuration completed because the worktree does not contain a usable `src/external/Par2_Core/CMakeLists.txt`.
- This blocked a local build/test pass from the worktree itself.

Remote configure/build on `v100`:

```bash
cmake -S . -B build/v100-petsc -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DMACROFLOW3D_ENABLE_DIAGNOSTICS=OFF \
  -DMACROFLOW3D_ENABLE_PROFILING=ON \
  -DMACROFLOW3D_ENABLE_NVTX=ON \
  -DMACROFLOW3D_ENABLE_PETSC=ON \
  -DPETSC_DIR=$HOME/MacroFlow3D/src/external/petsc \
  -DPETSC_ARCH=arch-cuda \
  -DSLEPC_DIR=$HOME/MacroFlow3D/src/external/slepc

cmake --build build/v100-petsc -j2 --target analyze_invariant_quality
```

Remote analysis run:

```bash
./build/v100-petsc/analyze_invariant_quality
```

Remote reference check:

```bash
cmake --build build/v100-petsc -j2 --target validate_slepc_eigensolver
./build/v100-petsc/validate_slepc_eigensolver
```

## Artifacts produced

Remote run produced:

- `artifacts/gate3/invariant_quality_summary.csv`
- `artifacts/gate3/invariant_quality_rotation_scan.csv`
- `artifacts/gate3/invariant_quality_localization.csv`
- `artifacts/gate3/invariant_quality_gauge.csv`
- `artifacts/gate3/invariant_quality_energy.csv`
- `artifacts/gate3/invariant_quality_localization_v2.csv`

Important caveat:

- `uniform_x` and `layered_x` rows are scientifically usable.
- `darcy_small` rows were emitted **while PETSc was reporting matrix-assembly errors** in the production backend and therefore are **not trustworthy acceptance evidence**.

## Quantitative results

### 1. Controlled cases: linear gauge did not fix reconstruction

From `invariant_quality_gauge.csv`, best-rotation rows:

- `uniform_x`
  - `rel_residual_after_gauge = 1.0` for all `μ ∈ {1e-5, 3e-5, 1e-4, 3e-4, 1e-3}`
  - `alpha_opt` magnitude stayed between `0` and `6.03e-16`
  - `cos(v, ∇ψ1×∇ψ2)` stayed effectively zero
  - `residual_floor_rel = 1.0`
- `layered_x`
  - `rel_residual_after_gauge = 1.0` for all swept `μ`
  - `alpha_opt` magnitude stayed between `4.63e-14` and `2.40e-9`
  - `cos(v, ∇ψ1×∇ψ2)` stayed effectively zero
  - `residual_floor_rel = 1.0`

Interpretation:

- the controlled-case mismatch did **not** become sensible after the linear gauge stage,
- therefore `v - ∇ψ1 × ∇ψ2` is still not a scientifically interpretable metric under this gauge model,
- and the missing freedom is larger than offset/sign/scale/orthogonal rotation.

### 2. Controlled cases: recovered modes are transport nulls selected by regularization

From `invariant_quality_energy.csv`:

- `uniform_x`
  - `f_transport ≈ 0`
  - `f_regularization ≈ 1`
- `layered_x`
  - `f_transport ≈ 0`
  - `f_regularization = 1`

Interpretation:

- on the controls, the recovered pair sits in a highly degenerate transport near-nullspace,
- the eigensolver is choosing smooth Fourier-like representatives from that nullspace,
- so a linear 2×2 transform inside the chosen pair cannot convert those modal coordinates into the physical `(y,z)`-like streamsurface coordinates needed for reconstruction.

### 3. Rotation helps independence, but not reconstruction

From `invariant_quality_summary.csv`:

- `uniform_x` best-rotation combined score dropped from about `1.069` to `1.001–1.061`
- `layered_x` best-rotation combined score dropped to about `1.001`
- degeneracy fraction on best-rotation rows dropped to `0` on `layered_x` and to `0–0.15625` on `uniform_x`

Interpretation:

- basis rotation still helps independence / degeneracy bookkeeping,
- but it does not address the reconstruction problem because the pair still carries the wrong parametrization.

### 4. `darcy_small`: production backend emitted contaminated output

During the `darcy_small` phase the run printed repeated PETSc errors of the form:

```text
Argument out of range
New nonzero at (...) caused a malloc
```

originating from `MatSetValues_SeqAIJ()` inside the production backend path.

Additional evidence:

- `SLEPcBackend.cu` in the current repo still reports `Assembly (45-color probing)` in the production backend.
- The emitted `darcy_small` energy rows show placeholder-like solver eigenvalues (`-1e-08`) and should not be trusted as scientific evidence.

The separate validation executable still runs on `uniform_x`, but that does **not** rescue the `darcy_small` production-backend output from contamination.

## Localization

`invariant_quality_localization_v2.csv` is informative for the controlled cases only:

- `uniform_x` gauge-ready best-rotation rows gave
  - `x_boundary_halo rel_residual_after_gauge = 1.0`
  - `x_interior rel_residual_after_gauge = 1.0`
- `layered_x` showed the same `1.0` / `1.0` pattern

Interpretation:

- the failure is global even on the controls,
- it is **not** a localized boundary artifact,
- which is consistent with a missing nonlinear coordinate gauge rather than a local stencil defect in those controls.

## Conclusion

This iteration resolved one ambiguity decisively:

- **A linear gauge/scale fix is not enough.**

The controlled cases show that:

- invariance residuals can be excellent,
- the transport term can be essentially zero,
- the recovered subspace can be the expected one,
- and yet `∇ψ1 × ∇ψ2` can remain globally orthogonal to `v` because the chosen pair still uses the wrong coordinates on the invariant manifold.

Therefore the current blocker is **not refinement** and **not μ tuning**.

The immediate blocker is:

- a missing **nonlinear section calibration / coordinate gauge / pair-selection stage** that can turn the degenerate transport-nullspace basis into reconstruction-ready streamsurface coordinates.

In parallel, the current production backend remains unsafe on `darcy_small` because the probing assembly still triggers PETSc sparsity-preallocation errors on the heterogeneous case.

## Next recommended move

Recommendation: **D — a narrower blocker has emerged and is now explicit.**

Do next:

1. Add a **nonlinear section-calibration gauge** on controlled cases.
   - Goal: map the recovered invariant pair to monotone `(y,z)`-like coordinates on a reference section instead of trying to fix everything with a linear 2×2 transform.
2. Revisit **pair selection inside the degenerate near-nullspace**.
   - The first two eigenvectors are smooth modal representatives, not automatically the right physical coordinates.
3. Reopen the production backend assembly path for heterogeneous cases.
   - The current `45-color probing` path is still capable of PETSc preallocation failure on `darcy_small`.
4. Do **not** start refinement yet.
   - Refinement would be acting on a pair that is not reconstruction-ready even on the controls.

## Gate impact

- Gate 3 is **closer to a credible reopening** only in the limited sense that the ambiguity is sharper:
  - refinement is now clearly premature,
  - μ tuning is not the right first move on the controlled cases,
  - and the remaining scientific blocker is more precise than “mismatch still mediocre.”

- Gate 3 is **not** ready to reopen yet because:
  - reconstruction mismatch is still not interpretable after the linear gauge stage,
  - and the heterogeneous-case production backend still emits PETSc assembly errors.
