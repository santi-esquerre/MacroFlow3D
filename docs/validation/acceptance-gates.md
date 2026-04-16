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

### Gate 2 — Algebra / operator integrity

Use for changes touching:

- operators
- eigensolver backend
- invariant construction algebra
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

Required evidence:

- pass/fail output
- residual norms
- if changed: before/after comparison

### Gate 3 — PSPTA local scientific integrity

Use for changes touching:

- `src/physics/particles/pspta/`
- interpolation used by PSPTA
- Newton projection
- invariant sampling
- transport/invariant coupling

**Operational plan:** `docs/plans/active/pspta-execution-plan.md` — verify that the change aligns with the current execution phase before accepting.

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

### Gate 4 — Helicity-free regime correctness

Use for changes that can affect the central scientific claim in the smooth, locally isotropic, purely advective regime.

This is the most important gate.

**Scientific basis:** `docs/theory/lester-2023-key-claims.md`
**Operational plan:** `docs/plans/active/pspta-execution-plan.md` — Phase 6 validation criteria apply.

The target regime is:

- smooth, locally isotropic Darcy flow,
- helicity-free (proven for scalar isotropic conductivity — Lester 2023 §1),
- two invariants / streamfunctions exist (Lester 2023 §2),
- no intended physical transverse macrodispersion in pure advection (Lester 2023 §4),
- PSPTA should preserve the relevant kinematic constraints.

Required:

- Gate 1–3
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

- trajectories remain consistent with invariant confinement,
- purely advective transverse spreading should not be accepted as evidence of physical macrodispersion in this regime.

### Gate 5 — Ensemble / macrodispersion behavior

Use for changes touching:

- ensemble statistics,
- macrodispersion analysis,
- transport outputs used for scientific interpretation,
- solver/interpolation choices likely to change reported `α_L`, `α_T`

**Scientific basis:**

- `docs/theory/lester-2023-key-claims.md` — regime where `α_T = 0` is expected
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
- if runtime behavior changed: Gate 3

### D. PSPTA / invariant / tracking change

Need:

- Gate 1
- Gate 2
- Gate 3
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
| `docs/theory/lester-2023-key-claims.md` | Gate 3, 4, 5 | Smooth isotropic Darcy flow is helicity-free; purely advective transverse macrodispersion is zero; methods must preserve invariant geometry |
| `docs/theory/beaudoin-de-dreuzy-2013-key-claims.md` | Gate 5 | Classical 3D numerical baseline for `α_L`, `α_T`; valuable for domain design, Monte Carlo discipline, and longitudinal validation; must be interpreted with regime awareness after Lester |

Read the relevant note before authoring or reviewing changes that touch Gate 3+.
