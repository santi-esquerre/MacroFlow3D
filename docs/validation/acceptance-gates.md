# Acceptance gates

This file defines the minimum validation gates for changes that affect scientific or numerical behavior.

A change is not accepted because it is elegant, fast, or plausible.
It is accepted only if it passes the right gate.

---

## Gate taxonomy

### Gate 0 — Repo / tooling hygiene

Use for:

- docs
- scripts
- AGENTS
- runbooks
- config-only changes
- CI / workflow changes

Required:

- file correctness
- command correctness
- no broken documented workflow

### Gate 1 — Build / smoke

Use for:

- general code changes
- refactors not intended to change science
- low-risk runtime/I/O work

Required:

- configure succeeds
- build succeeds
- relevant tests run
- relevant smoke run

Minimum commands:

```bash
cmake --build <build-dir> -j
ctest --test-dir <build-dir> --output-on-failure
./<build-dir>/macroflow3d_pipeline apps/config_pspta_small.yaml
```

The full `ctest --test-dir <build-dir> --output-on-failure` pass is
long-duration computation: local WSL runs only a fast targeted `-R` subset for
iteration, and the authoritative full pass runs on V100 as a detached
`scripts/remote run <job>` job, not locally and not via blocking
`scripts/remote exec` — see `docs/runbooks/remote-v100.md` Section 0.

#### Validation tiers in ctest

`ctest` holds fast contract tests only. Multi-case solver sweeps and science
smokes are not registered as `ctest` entries: they are documented experiments
(`docs/experiments/`) run as detached V100 jobs (`scripts/remote run <job>`),
consistent with `docs/runbooks/remote-v100.md` Section 0. A heavy entry that
stays in `ctest` is a defect of the validation tier, not a gate.

### Gate 2 — Algebra / operator integrity

Use for changes touching:

- operators
- eigensolver backend
- invariant construction algebra
- Lester equation (14) linear operator `A psi = -div(q grad psi)`
- adjoint/symmetry assumptions
- refinement logic

Required:

- operator tests pass
- no new unexplained residual growth
- numerical properties remain within expected tolerances

Minimum commands:

```bash
ctest --test-dir <build-dir> --output-on-failure -R operator_tests
ctest --test-dir <build-dir> --output-on-failure -R validate_slepc_eigensolver
```

`validate_slepc_eigensolver` requires PETSc/SLEPc and runs on V100 as a
detached job (`scripts/remote run` + `scripts/remote wait`), never locally —
see `docs/runbooks/remote-v100.md` Section 0.

Required evidence:

- pass/fail output
- residual norms
- operator sign, boundary, gauge, and coefficient convention if changed or newly introduced
- if changed: before/after comparison

For equation (14) operator work, required evidence also includes manufactured-solution checks for `k=1` and smooth variable `k` before any nonlinear solver claim.

### Gate 3 — Legacy PSPTA compatibility integrity

Use for changes touching:

- `src/physics/particles/pspta/`
- interpolation used by PSPTA
- Newton projection
- invariant sampling
- transport/invariant coupling

**Status:** legacy PSPTA compatibility gate. Use `docs/plans/archive/pspta-execution-plan.md` only as historical context. New invariant construction must use Gate 3A instead.

Required:

- Gate 1 and Gate 2
- quality metrics for invariants and projection
- no unexplained increase in failure modes

Required metrics to inspect:

- `v·∇ψ1` residual summary
- `v·∇ψ2` residual summary
- `||v - ∇ψ1 × ∇ψ2||` or equivalent reconstruction mismatch if available
- independence / degeneracy signal
- Newton failures:
  - nonzero fail count
  - max fail count
  - histogram / summary
- particle status summary:
  - active
  - exited
  - other

Reject if:

- residuals materially worsen without a reasoned tradeoff,
- independence collapses,
- Newton failures explode,
- particles cross behavior boundaries unexpectedly.

### Gate 3A — Lester equation (14) streamfunction solver integrity

Use for changes touching:

- `psi1`/`psi2` construction through Lester equation (14),
- `S1`, `S2`, `B`, gradient, Hessian-vector, or denominator kernels,
- Picard, Anderson, or Newton-Krylov solver loops,
- gauge restoration for streamfunctions,
- velocity reconstruction from `grad psi1 x grad psi2`.

**Operational plan:** `docs/plans/active/lester-eq14-streamfunction-solver-plan.md`.

Required:

- Gate 1 and Gate 2;
- start on `16^3`/`32^3` controls before larger grids;
- explicit smoothness regime: Gaussian covariance, uniform `k`, or documented regularized field;
- coupled residual `r_F`;
- velocity reconstruction error `e_v`;
- Darcy invariance errors `e_i`;
- reconstructed-flow divergence `e_div`;
- minimum and 0.1%, 1%, 5% percentiles of `|grad psi1 x grad psi2|`;
- denominator regularization value and convergence plan;
- gauge definition and gauge restoration evidence;
- grid-convergence plan or result;
- velocity-reconstruction error `e_v` measured under grid refinement on the same continuum field, with the observed order reported; a result is never accepted on `r_F` alone;
- for any new target object, an independent existence / positive-control check (not using the solver under test) precedes solver acceptance.

Reject if:

- multigrid reuse is assumed without operator compatibility evidence,
- exponential covariance is treated as a smooth Gaussian-equivalent benchmark,
- denominator regularization is hidden or arbitrary,
- a result is accepted solely because the linear or nonlinear residual decreased,
- tensor/local anisotropy is treated as in-scope without a new theory decision.

### Gate 4 — Helicity-free regime correctness

Use for changes that can affect the central scientific claim in the smooth, locally isotropic, purely advective regime.

This is the most important gate.

**Scientific basis:** `docs/theory/lester-2023-key-claims.md`
**Operational plan:** use `docs/plans/active/lester-eq14-streamfunction-solver-plan.md` for new invariant construction and `docs/plans/archive/pspta-execution-plan.md` for existing PSPTA transport validation.

The target regime is:

- smooth, locally isotropic Darcy flow,
- helicity-free (proven for scalar isotropic conductivity — Lester 2023 §1),
- two invariants / streamfunctions exist locally (Lester 2023 §2); a global affine + periodic pair does not exist for generic periodic `k` (`docs/experiments/2026-10-02-streamline-closure-and-eq14-vs-darcy.md`),
- the project does not presuppose the value of `alpha_T`; every observed transverse growth must be classified physical / numerical / unresolved with grid and tolerance refinement evidence, and compared against the streamline-closure oracle (return map of the independently integrated Darcy streamlines),
- the tracker should preserve the labels of the actual Darcy flow (non-periodic in `x1`, anchored at the inlet).

Required:

- Gate 1-3, or Gate 3A for the Lester equation (14) solver path
- controlled pure-advection test case
- careful interpretation of transverse spreading
- explicit statement whether any observed transverse growth is:
  - physical,
  - numerical,
  - or unresolved

Reject if:

- positive transverse spreading is treated as physical without control tests,
- a method degrades confinement to streamsurfaces,
- interpolation or stepping changes introduce leakage across invariant surfaces.

Expected qualitative behavior:

- trajectories remain consistent with the invariant labels of the flow being tracked (no leakage across label surfaces beyond the stated tolerance),
- observed transverse growth is reported with its classification (physical / numerical / unresolved), grid and tolerance refinement evidence, and the closure-oracle comparison; neither positive nor zero transverse spreading is accepted or rejected a priori.

### Gate 5 — Ensemble / macrodispersion behavior

Use for changes touching:

- ensemble statistics,
- macrodispersion analysis,
- transport outputs used for scientific interpretation,
- solver/interpolation choices likely to change reported `α_L`, `α_T`

**Scientific basis:**

- `docs/theory/lester-2023-key-claims.md` — the paper's claim `α_T = 0` under bounded label fluctuations; not presupposed by the project
- `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md` — classical 3D baseline for `α_L`, `α_T`

When comparing to historical 3D macrodispersion literature (e.g. Beaudoin & de Dreuzy 2013), comparisons must state explicitly: covariance model, `σ_Y²`, boundary conditions, injection protocol, tracking method, and asymptotic-estimation procedure. Without that, agreement or disagreement is scientifically weak.

Required:

- baseline comparison against prior known-good run
- exact config(s) recorded
- output artifacts preserved
- before/after comparison of:
  - `α_L(t)`
  - `α_T1(t)`, `α_T2(t)` if applicable
  - selected raw moment trends
  - relevant diagnostics

Reject if:

- output changes are not explained,
- the comparison is missing,
- the run is not reproducible from the reported config/build.

---

## Required evidence by change type

### A. Docs / workflow only

Need:

- list of files changed
- commands checked

### B. Refactor with no intended numerical change

Need:

- Gate 1
- statement of invariance target
- comparison proving no material change on smoke case

### C. Solver / operator change

Need:

- Gate 1
- Gate 2
- if runtime behavior changed: Gate 3 or Gate 3A, depending on path

### D. Legacy PSPTA / invariant / tracking compatibility change

Need:

- Gate 1
- Gate 2
- Gate 3 for legacy PSPTA compatibility, or Gate 3A for Lester equation (14) invariant construction
- likely Gate 4

### E. Macrodispersion / scientific output change

Need:

- Gate 1
- Gate 3 or 4 as relevant
- Gate 5

---

## Minimum report template for scientific changes

Paste this into the final summary or PR description:

```md
## Scientific change report

### Scope
- files changed:
- intended effect:

### Commands run
- configure:
- build:
- tests:
- smoke:
- remote runs:

### Metrics checked
- operator residuals:
- invariant residuals:
- equation (14) residuals:
- velocity reconstruction error:
- denominator percentiles:
- Newton failure summary:
- macrodispersion outputs:

### Result
- passed gates:
- known risks:
- unresolved questions:
```

---

## Practical rule of thumb

If a change can affect:

- the velocity field,
- interpolation,
- invariant quality,
- tracking,
- or macrodispersion outputs,

assume Gate 4 or Gate 5 until proven otherwise.

---

## Scientific reference notes

The following theory notes underpin the gate definitions:

| Note | Gates it informs | Core claim |
|------|------------------|------------|
| `docs/theory/lester-2023-key-claims.md` | Gate 3, 4, 5 | Smooth isotropic Darcy flow is helicity-free; the paper's zero-transverse claim holds under bounded label fluctuations and is not presupposed here (periodic-cell limits in the experiment note of 2026-10-02); methods must preserve invariant geometry |
| `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md` | Gate 5 | Classical 3D numerical baseline for `α_L`, `α_T`; valuable for domain design, Monte Carlo discipline, and longitudinal validation; must be interpreted with regime awareness after Lester |

Read the relevant note before authoring or reviewing changes that touch Gate 3+.
