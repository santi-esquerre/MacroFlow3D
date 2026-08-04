# Lester equation (14) streamfunction solver — execution dashboard

## Purpose

This is the authoritative execution dashboard for the Lester equation (14)
streamfunction solver.  It records the decisions shared by all increments and
selects the only increment that may be worked on next.

For a concise explanation of the scientific problem, the Lester reference
case, and the Picard/Anderson/Newton-Krylov strategy, read the
[solver overview](lester-eq14-streamfunction-solver-overview.md).

Read, in order:

1. `AGENTS.md` and the closest local `AGENTS.md`;
2. `docs/plans/active/lester-eq14-streamfunction-solver-overview.md`;
3. `docs/theory/lester-2023-key-claims.md`;
4. `docs/validation/acceptance-gates.md`;
5. this dashboard;
6. the specification linked by `NEXT` below;
7. `docs/runbooks/lester-increment-workflow.md`.

The legacy PSPTA invariant-construction route is not part of this plan.

## Execution state

- NEXT: `SF-02`
- Active runtime goal: `SF-02`
- Execution model: strictly sequential
- Canonical state: the state visible on the repository default branch
- Last completed increment: `SF-01`

An increment may start only when it is named by `NEXT` and every dependency in
its specification is `done`.  The next increment is not enabled until the
current increment's completion commit has been merged.

## Master checklist

- [x] [SF-00 — Increment harness](lester-eq14/increments/SF-00-harness.md)
- [x] [SF-01 — Reference test harness](lester-eq14/increments/SF-01-reference-tests.md)
- [ ] [SF-02 — Discrete operator contract](lester-eq14/increments/SF-02-operator-contract.md)
- [ ] [SF-03 — Mean-zero projector](lester-eq14/increments/SF-03-mean-zero-projector.md)
- [ ] [SF-04 — Projected PCG](lester-eq14/increments/SF-04-projected-pcg.md)
- [ ] [SF-05 — Multigrid reuse](lester-eq14/increments/SF-05-multigrid-reuse.md)
- [ ] [SF-06 — Affine-periodic right-hand sides](lester-eq14/increments/SF-06-affine-periodic-rhs.md)
- [ ] [SF-07 — Streamfunction gradients](lester-eq14/increments/SF-07-gradients.md)
- [ ] [SF-08 — Hessian-vector products and B](lester-eq14/increments/SF-08-hessian-vector-b.md)
- [ ] [SF-09 — Nonlinear sources](lester-eq14/increments/SF-09-nonlinear-sources.md)
- [ ] [SF-10 — Coupled residual](lester-eq14/increments/SF-10-coupled-residual.md)
- [ ] [SF-11 — Physical diagnostics](lester-eq14/increments/SF-11-physical-diagnostics.md)
- [ ] [SF-12 — Public API and workspace](lester-eq14/increments/SF-12-public-api-workspace.md)
- [ ] [SF-13 — Homogeneous solver](lester-eq14/increments/SF-13-homogeneous-solver.md)
- [ ] [SF-14 — Fixed-relaxation Picard](lester-eq14/increments/SF-14-fixed-picard.md)
- [ ] [SF-15 — Adaptive Picard](lester-eq14/increments/SF-15-adaptive-picard.md)
- [ ] [SF-16 — Pipeline, configuration, and output](lester-eq14/increments/SF-16-pipeline-io-config.md)
- [ ] [SF-17 — Eta and epsilon continuation](lester-eq14/increments/SF-17-eta-epsilon-continuation.md)
- [ ] [SF-18 — Periodic Gaussian generator](lester-eq14/increments/SF-18-periodic-gaussian-generator.md)
- [ ] [SF-19 — Affine-periodic Darcy solve](lester-eq14/increments/SF-19-affine-periodic-darcy.md)
- [ ] [SF-20 — Heterogeneity continuation](lester-eq14/increments/SF-20-heterogeneity-continuation.md)
- [ ] [SF-21 — Grid continuation](lester-eq14/increments/SF-21-grid-continuation.md)
- [ ] [SF-22 — Anderson acceleration](lester-eq14/increments/SF-22-anderson.md)
- [ ] [SF-23 — GPU optimization](lester-eq14/increments/SF-23-gpu-optimization.md)
- [ ] [SF-24 — V100 benchmark](lester-eq14/increments/SF-24-v100-benchmark.md)
- [ ] [SF-25 — Matrix-free Jacobian-vector product](lester-eq14/increments/SF-25-matrix-free-jvp.md)
- [ ] [SF-26 — Restarted GMRES and block preconditioner](lester-eq14/increments/SF-26-gmres-preconditioner.md)
- [ ] [SF-27 — Globalized Newton-Krylov](lester-eq14/increments/SF-27-newton-krylov.md)
- [ ] [SF-28 — Mixed-precision preconditioner study](lester-eq14/increments/SF-28-mixed-precision.md)

## Locked mathematical and discrete decisions

For a smooth scalar conductivity `K > 0`, define `q = 1/K` and

```math
A u = -\nabla\cdot(q\nabla u).
```

Store only periodic, cell-centered fluctuations:

```math
\psi_1 = \bar v x_2 + \widetilde\psi_1,
\qquad
\psi_2 = x_3 + \widetilde\psi_2,
```

with

```math
\langle\widetilde\psi_1\rangle=
\langle\widetilde\psi_2\rangle=0.
```

The benchmark affine gradients are

```math
\bar g_1=(0,\bar v,0),\qquad \bar g_2=(0,0,1).
```

For the fluctuations, solve

```math
A\widetilde\psi_1 = \nabla\cdot(q\bar g_1)-\eta qS_2,
\qquad
A\widetilde\psi_2 = \nabla\cdot(q\bar g_2)-\eta qS_1.
```

Locked discretization rules:

- use the actual positive operator `A`; wrap the current legacy-sign operator
  rather than silently changing all existing flow callers;
- use the harmonic mean of `q` at faces,
  `q_f = 2 q_C q_N / (q_C + q_N) = 2 / (K_C + K_N)`;
- do not obtain `q_f` by inverting the harmonic mean of `K`;
- assemble affine right-hand sides with the same face coefficients used by `A`;
- use periodic wrapping in all three directions for the target benchmark;
- use a mean-zero projection, not `PinSpec`, to remove the periodic null mode;
- project right-hand sides, iterates, PCG vectors, multigrid corrections,
  Picard trials, accelerated states, and grid-prolongated states;
- begin with second-order centered derivatives;
- compute Hessian-vector products directly; do not store full Hessians;
- use the radius-one, 19-point union stencil for each streamfunction;
- evaluate the nonlinear residual with exactly the same discrete `A` and
  source construction used by the solver;
- retain double precision through the accepted Picard and V100 benchmark
  phases.

## Locked nonlinear and continuation policy

Picard starts sequentially: evaluate one authoritative nonlinear state, solve
the two blocks consecutively with the same operator and multigrid hierarchy,
then relax both fields as a pair.

Defaults:

- nonlinear tolerance: `1e-6` initially, `1e-8` after mesh convergence;
- linear relative tolerance: `1e-10`;
- maximum Picard iterations: `500`;
- initial relaxation: `0.25`;
- minimum relaxation: `0.01`;
- rejected step reduction: `0.5`;
- accepted-step growth: `1.2` after three easy accepted steps;
- stagnation: less than 1% residual reduction over ten iterations.

Regularize

```math
d_\epsilon=|\nabla\psi_1\times\nabla\psi_2|^2
             +(\epsilon v_{\mathrm{rms}})^2.
```

Start at `epsilon=1e-2`, reduce by decades after converged stages, require
`1e-6` for the first accepted solver, and study `1e-8` later.  Reject a trial
on NaN/Inf, on more than 1% unexplained degenerate cells, when the unexplained
fraction exceeds `2*f_previous + 1e-4`, or when the 0.1% percentile collapses
by more than one decade without a matching Darcy low-speed population.

Continuation order:

1. solve each grid from the previous accepted grid when available;
2. continue `lambda` in `K_lambda=exp(lambda Y)` with initial step `0.1`,
   minimum `0.0125`, maximum `0.2`, halving on failure and growing by `1.5`
   after two easy stages;
3. keep `eta=1` normally; on a failed lambda stage solve the harmonic-coordinate
   problem at `eta=0` and ramp `eta` back to one;
4. reduce `epsilon` only after `lambda=eta=1` is accepted.

## Required physical diagnostics

Use RMS-based, dimensionless normalizations.  With
`L_ref=(Lx*Ly*Lz)^(1/3)`, report

```math
r_i = \frac{\mathrm{RMS}(F_i)}{q_{\mathrm{rms}}g_i/L_{ref}},
\qquad
r_F=\sqrt{(r_1^2+r_2^2)/2},
```

where `g_1=v_rms` and `g_2=1` in the benchmark units.  Also report:

- L2, Linf, component, magnitude, correlation, and angular errors between
  reconstructed CompactMAC velocity and Darcy velocity;
- `RMS(v_D dot grad(psi_i))/(v_D,rms * grad(psi_i),rms)`;
- `L_ref*RMS(div(v_psi))/v_rms`;
- min, max, mean, 0.1%, 1%, 5%, and 50% percentiles of `|cross-grad|`;
- counts below configured relative thresholds, split between cells with
  genuinely low Darcy speed and unexplained degeneracy;
- gauge means, raw right-hand-side compatibility defects, linear histories,
  nonlinear histories, continuation histories, and rejected-step reasons.

The initial CompactMAC reconstruction is expected to be approximately, not
algebraically, divergence-free.  Its divergence must converge under grid
refinement.

## Architecture and memory constraints

New invariant construction belongs under `src/physics/streamfunctions/`.  Do
not extend `src/physics/particles/pspta/`.  Reuse is conditional on the tests
in SF-02 through SF-05.

The intended public surface is:

```cpp
struct AffineGauge;
struct StreamfunctionProblemView;
struct StreamfunctionFields;
struct StreamfunctionWorkspace;
struct StreamfunctionSolveReport;

StreamfunctionSolveReport solve_streamfunctions(
    CudaContext& context,
    const StreamfunctionProblemView& problem,
    const StreamfunctionSolverConfig& config,
    StreamfunctionFields& fields,
    StreamfunctionWorkspace& workspace);
```

At `256^3`, one double scalar field is 128 MiB.  The sequential Picard design
must target about 24.6 fine-grid-equivalent scalar fields (about 3.1 GiB), or
about 3.6–4 GiB including `Y` and Darcy velocity.  Anderson stores four scalar
fields per history level.  Restarted GMRES stores two scalar fields per coupled
basis vector; FGMRES adds the preconditioned basis.  No allocation or CPU/GPU
field transfer is permitted inside hot nonlinear loops.

## Benchmark progression

- operator controls: periodic trigonometric manufactured functions;
- homogeneous controls: `16^3`, `32^3`, `64^3`;
- smooth Gaussian smoke: `32^3`, then `64^3`, with
  `sigma_Y^2={0.25,1,2.25,4}` and fixed seeds;
- physical mesh study: same continuous periodic realization, physical
  `ell=1/16`, then `128^3 -> 256^3`;
- reference run: `[0,1]^3`, `256^3`, `ell=1/16`, `sigma_Y^2=4`,
  normalized mean velocity one;
- robustness: 5–10 selected realizations at a validated smaller grid;
- `sigma_Y^2=6.25` only after the `sigma_Y^2=4` suite is accepted.

Exponential covariance, tensor conductivity, a PSPTA consumer, and scientific
macrodispersion production are outside this execution sequence.

## Recording and advancement

Every increment uses the specification template in
`docs/plans/increment-template.md` and the runbook in
`docs/runbooks/lester-increment-workflow.md`.  Real scientific runs also create
or update a note in `docs/experiments/`.

Run before marking an increment complete:

```bash
bash scripts/hooks/check-lester-increments.sh
cmake --preset wsl-debug
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure
```

Add the targeted Gate 2/3A/4 and V100 commands required by the increment.  A
low algebraic residual alone never completes a scientific increment.
