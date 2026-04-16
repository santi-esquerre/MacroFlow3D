# Numerics AGENTS.md

## Scope

This file applies to `src/numerics/` and related low-level numerical work.

This layer provides numerical mechanisms used by the science code.
It should stay precise, predictable, and easy to reason about.

---

## Responsibilities of this layer

This layer should own:
- operators
- BLAS-like kernels
- solver machinery
- preconditioners
- linear algebra helper infrastructure

This layer should not own:
- scientific interpretation
- experiment policy
- output policy
- orchestration

---

## Hard rules

- No hidden allocations in repeated numerical kernels.
- No silent host-device synchronization in hot paths.
- No mixed semantic changes hidden inside “cleanup.”
- Keep interfaces explicit.
- Preserve numerical intent when refactoring.

---

## Refactor policy

Safe refactors in this layer:
- naming cleanup
- local structure cleanup
- workspace reuse improvements
- clearer operator boundaries

High-risk refactors:
- operator sign conventions
- boundary handling
- stencil changes
- adjoint/symmetry changes
- solver stopping criteria
- reduction / synchronization behavior

Treat high-risk refactors as scientific changes, not cosmetic ones.

---

## Validation expectations

For operator or solver changes, run:

```bash
ctest --test-dir <build-dir> --output-on-failure -R operator_tests
```

If PETSc/SLEPc-related code is touched, also run:

```bash
ctest --test-dir <build-dir> --output-on-failure -R validate_slepc_eigensolver
```

If runtime behavior is affected downstream, run a smoke pipeline case too.

---

## Code style for numerics

- prefer small, explicit kernels
- make indexing conventions obvious
- document boundary assumptions
- document precision assumptions
- comment only where the numerical contract is not obvious

Avoid comments that merely restate code.

---

## Done criteria

A numerics task is done when:
1. the numerical contract is still clear,
2. validation was run,
3. no new hidden synchronization/allocation was introduced,
4. any boundary/sign/precision change is documented explicitly.
