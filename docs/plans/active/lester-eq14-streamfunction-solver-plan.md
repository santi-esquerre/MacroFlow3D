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
6. the specification of the READY increment being activated;
7. `docs/runbooks/lester-increment-workflow.md`.

The legacy PSPTA invariant-construction route is not part of this plan.

## Execution state

- Increment ordering: dependency DAG; an increment is READY when pending with every dependency done on the default branch
- Concurrency: at most two nonterminal increments, one orchestrator session and one delivery PR each
- Intra-increment execution: orchestrated DAG; independent nodes may run in parallel
- Delivery: final audited GitHub pull request; no automatic merge
- Canonical state: the state visible on the repository default branch

The ready set is computed by `bash scripts/hooks/check-lester-increments.sh`: an
increment may start only when it is `pending` and every dependency in its
specification is `done` on the default branch. Within each active increment, the
orchestrator may decompose the Goal into a DAG and execute independent nodes
concurrently. A dependent increment is not enabled until the audited PR that
marks its dependencies `done` has been merged and that state is visible on the
default branch.

## Claude Code intra-increment orchestration contract

This section governs **how each active increment is executed**. It does
not relax the dependency ordering of increments.

1. **UNDERSTAND — Fable 5 / xhigh**
   - Read the project foundations, architecture, solver overview, theory note,
     acceptance gates, this dashboard, the active increment specification, the
     increment workflow, and the relevant code/tests.
   - Establish the increment base commit and exact Goal before delegating work.

2. **PLAN — Fable 5 / xhigh**
   - Build a concrete, acyclic DAG whose complete accepted result implies the
     increment Goal.
   - Each node must define objective, dependencies, required context, expected
     write scope, forbidden scope, deliverables, acceptance criteria, validation
     commands, and required predecessor commits.
   - Persist runtime planning/audit state under
     `.claude/orchestration/<increment-id>/` when useful.

3. **EXECUTE — Sonnet 5 / medium**
   - Every implementation or corrective node is delegated to
     `increment-worker`.
   - Every worker runs with native `isolation: worktree`.
   - Independent nodes with non-conflicting write scopes may run concurrently.
   - A dependent node must explicitly incorporate the exact approved predecessor
     commit hashes supplied by the orchestrator.
   - Workers commit their own result but do not push or open PRs.

4. **AUDIT / CORRECT — Fable 5 / xhigh**
   - Worker `success` is only a report, never acceptance.
   - The orchestrator independently reads the diff, checks the scientific and
     discrete contracts, re-runs appropriate validation, and evaluates every
     acceptance criterion.
   - Failed audits generate a corrective DAG and repeat until no blocking or
     major findings remain.

5. **INTEGRATE — Sonnet 5 / medium**
   - After all required nodes are accepted, launch exactly one
     `increment-integrator` in a fresh isolated worktree.
   - The integrator starts from the increment base and incorporates only
     orchestrator-approved commits in dependency-compatible order.
   - It resolves and documents integration conflicts and runs the full
     increment validation, but it does not push or open a PR.

6. **FINAL_AUDIT / PUBLISH_PR — Fable 5 / xhigh**
   - The orchestrator independently audits the integrated commit against the
     original increment Goal, checklist, acceptance gates, and full diff from
     the increment base.
   - Any failure returns to a corrective DAG and fresh integration/audit.
   - High-autonomy increments may finalize `done` metadata before opening the
     PR; the default branch still blocks dependents until human merge.
   - Human-review increments publish the audited source result as
     `awaiting_review` (not `done`).
   - After explicit human approval, resume the **same PR** and add only the
     closure metadata commit: complete the checklist, set `done`, check the
     master-checklist entry, and run `check-lester-increments.sh`.
   - No agent merges the PR.

The autonomous implementation run ends when the audited PR is opened. Formal
closure of a human-review increment adds the closure-only metadata commit after
human approval and before human merge. Dependent increments remain blocked
until that closure state is merged and visible on the default branch.

## Master checklist

- [x] [SF-00 — Increment harness](lester-eq14/increments/SF-00-harness.md)
- [x] [SF-01 — Reference test harness](lester-eq14/increments/SF-01-reference-tests.md)
- [x] [SF-02 — Discrete operator contract](lester-eq14/increments/SF-02-operator-contract.md)
- [x] [SF-03 — Mean-zero projector](lester-eq14/increments/SF-03-mean-zero-projector.md)
- [x] [SF-04 — Projected PCG](lester-eq14/increments/SF-04-projected-pcg.md)
- [x] [SF-05 — Multigrid reuse](lester-eq14/increments/SF-05-multigrid-reuse.md)
- [x] [SF-06 — Affine-periodic right-hand sides](lester-eq14/increments/SF-06-affine-periodic-rhs.md)
- [x] [SF-07 — Streamfunction gradients](lester-eq14/increments/SF-07-gradients.md)
- [x] [SF-08 — Hessian-vector products and B](lester-eq14/increments/SF-08-hessian-vector-b.md)
- [x] [SF-09 — Nonlinear sources](lester-eq14/increments/SF-09-nonlinear-sources.md)
- [x] [SF-10 — Coupled residual](lester-eq14/increments/SF-10-coupled-residual.md)
- [x] [SF-11 — Physical diagnostics](lester-eq14/increments/SF-11-physical-diagnostics.md)
- [x] [SF-12 — Public API and workspace](lester-eq14/increments/SF-12-public-api-workspace.md)
- [x] [SF-13 — Homogeneous solver](lester-eq14/increments/SF-13-homogeneous-solver.md)
- [x] [SF-14 — Fixed-relaxation Picard](lester-eq14/increments/SF-14-fixed-picard.md)
- [x] [SF-15 — Adaptive Picard](lester-eq14/increments/SF-15-adaptive-picard.md)
- [x] [SF-16 — Pipeline, configuration, and output](lester-eq14/increments/SF-16-pipeline-io-config.md)
- [x] [SF-17 — Eta and epsilon continuation](lester-eq14/increments/SF-17-eta-epsilon-continuation.md)
- [x] [SF-18 — Periodic Gaussian generator](lester-eq14/increments/SF-18-periodic-gaussian-generator.md)
- [x] [SF-19 — Affine-periodic Darcy solve](lester-eq14/increments/SF-19-affine-periodic-darcy.md)
- [x] [SF-20 — Anderson acceleration](lester-eq14/increments/SF-20-anderson.md)
- [x] [SF-21 — Heterogeneity continuation](lester-eq14/increments/SF-21-heterogeneity-continuation.md)
- [x] [SF-22 — Matrix-free Jacobian-vector product](lester-eq14/increments/SF-22-matrix-free-jvp.md)
- [x] [SF-23 — Restarted GMRES and block preconditioner](lester-eq14/increments/SF-23-gmres-preconditioner.md)
- [x] [SF-24 — Globalized Newton-Krylov](lester-eq14/increments/SF-24-newton-krylov.md)
- [x] [SF-25 — Manifold-robust terminal solver](lester-eq14/increments/SF-25-terminal-manifold-solver.md)
- [x] [SF-26 — Equation (14) source pairing correction](lester-eq14/increments/SF-26-source-pairing-correction.md)
- [ ] [SF-27 — Validation-tier hygiene](lester-eq14/increments/SF-27-validation-tier-hygiene.md)
- [ ] [SF-28 — Periodic tricubic B-spline interpolation](lester-eq14/increments/SF-28-periodic-tricubic-spline.md)
- [ ] [SF-29 — CPU prototype: equation (14) with `x1` non-periodic and inlet labels](lester-eq14/increments/SF-29-eq14-inlet-labels-cpu-prototype.md)
- [ ] [SF-30 — Streamline-closure gate on the production stack](lester-eq14/increments/SF-30-streamline-closure-gate.md)
- [ ] [SF-31 — Pseudo-symplectic tracker core and RK reference](lester-eq14/increments/SF-31-pseudo-symplectic-tracker-core.md)
- [ ] [SF-32 — Face-flux reference trackers and the paper's scalings](lester-eq14/increments/SF-32-reference-trackers-and-scalings.md)

Re-sequencing 2026-10-02: CPU probes showed that for a generic smooth triply
periodic scalar `k` (including a Gaussian-covariance field) Darcy streamlines do
not close on the torus, so no affine plus triply periodic invariant pair exists
for them and the periodic solution of eq. (14) is a different flow (see
`docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md` and
`docs/decisions/2026-10-02-roadmap-audit-and-foundational-redesign.md`). The
former SF-27..SF-30 are cancelled (their specifications remain in git history at
`4670fb5`) and replaced by SF-27..SF-32 above. Dependency graph: SF-27 and SF-29
start in parallel; SF-28 follows SF-27; SF-30 and SF-31 follow SF-28 (in
parallel); SF-32 follows SF-31. Disposition of the open decisions D1-D7 of
`docs/decisions/2026-10-01-eta1-residual-floor-gauge-degeneracy.md`: D1, D3, and
D5 are moot (they tune a system whose solution is not the target); D2 is retired
with the 2026-10-02 record as the reason; D4 is resolved by cancelling the
pseudo-time increment; D6 is resolved by SF-27; D7 is resolved by adopting the
Lester (2021) field as positive control, the paper's `1e-16` remaining
unreconciled and non-blocking.

## Locked mathematical and discrete decisions

*Contract of the frozen periodic-fluctuation stack (SF-02..SF-26). These rules do not govern the `x1`-non-periodic inlet-label formulation; SF-29 selects that formulation and its closure PR writes the corresponding locked decisions.*

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
A\widetilde\psi_1 = \nabla\cdot(q\bar g_1)-\eta qS_1,
\qquad
A\widetilde\psi_2 = \nabla\cdot(q\bar g_2)-\eta qS_2.
```

Source pairing (corrected 2026-09-30): each block receives its OWN source,
`S_i = ((B x grad psi_i) . c)/|c|^2`, as derived in
`docs/decisions/2026-09-30-eq14-source-pairing-root-cause.md`. The paper's
printed equation (14) crosses the indices relative to its own definitions;
the code implemented that crossed form through SF-25 and SF-26 corrects it.

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

*Contract of the frozen periodic-fluctuation stack (SF-02..SF-26). These rules do not govern the `x1`-non-periodic inlet-label formulation; SF-29 selects that formulation and its closure PR writes the corresponding locked decisions.*

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

Applies to the frozen periodic stack; the closure-gate matrix lives in SF-30.

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

## Process rules (2026-10-02)

1. Run an independent existence or positive-control check before building any
   solver for a new target.
2. Accept on a physics metric under refinement, never on a residual alone.
3. Build a CPU prototype before a GPU increment.
4. `ctest` holds fast contract tests only; science runs are experiment notes run
   as detached V100 jobs.

## Later phases (prose; specifications are created by the SF-29 closure PR)

1. GPU generalization of `src/physics/streamfunctions/` to `x1` non-periodic per
   the SF-29 choice, reusing PCG/MG, SF-18, SF-19, and `Diagnostics.cuh`.
2. Acceptance in the periodic medium: `e_v(h)` and invariance convergent and the
   labels' return map agreeing with SF-30 at `sigma^2 = 0.25, 1, 2.25`;
   `(4, 1/16, 256^3)` is non-blocking.
3. Long domain (Dirichlet in `x`, periodic in `y`, `z`; 2048x256x256,
   `lambda/h = 10`): runner wiring and the study. `alpha_L` must match RWPT;
   `alpha_T` is reported with grid and tolerance convergence without
   presupposing its value; `sigma^2` in `{1, 2.25}` then `{4, 6.25}`. Open items:
   the generator (periodic SF-18 vs baseline) and memory at 1.3e8 cells on one
   V100.
4. Local dispersion and `D_T(Pe)` (Lester 2023 figure 5).

The periodic-cell study (`D_T` by re-injection, deterministic many-period
iteration) has no increment of its own: it is read off the return map that SF-30
measures and that the labels must reproduce. The move to the long domain
requires `e_v(h)` convergent and the return map concordant up to
`sigma^2 = 2.25`.

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
ctest --test-dir build/wsl-debug --output-on-failure -R <fast-targeted-case>
```

The full `ctest` suite is long-duration computation: run the authoritative
full pass on V100 as a detached job, never locally and never as a blocking
`scripts/remote exec` call (`docs/runbooks/remote-v100.md` Section 0):

```bash
scripts/remote sync
scripts/remote exec -- "cmake --preset v100-release && cmake --build build/v100-release -j"
scripts/remote run ctest-full -- "ctest --test-dir build/v100-release --output-on-failure"
scripts/remote wait ctest-full
```

Add the targeted Gate 2/3A/4 and V100 commands required by the increment.  A
low algebraic residual alone never completes a scientific increment.
