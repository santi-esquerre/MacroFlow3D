# legacy/

The pre-refactor CUDA code (multigrid head solve, TVD advection, Par2 random-walk
transport, `Makefile`, plot scripts) used in earlier papers (e.g. MECOM 2025) was
removed from `master`. It is preserved, unmodified:

- branch `archive/legacy`: the former `legacy/` tree as a standalone repository root
  (with its history); includes its own `Makefile`;
- tag `archive/pre-cleanup-2026-10`: the complete repository tree before the cleanup
  (commit `3452a7d`).

Frozen records (experiments, decisions, code comments) cite paths such as
`legacy/random_field_generation.cu:113-114`; they resolve at the same line numbers in the tag.

```bash
# read one file as it was
git show archive/pre-cleanup-2026-10:legacy/random_field_generation.cu
# check out the standalone, buildable tree
git worktree add ../macroflow3d-legacy archive/legacy
```

The input-parameter directories `input/`, `input_advección_pura/` and
`input_advección_difusión/` remain here because they are cited scientific data
(`apps/config_pipeline_par2.yaml`, decision 2026-10-07). Note
`input_advección_difusión/parameters_sigma2.25 copy.json` differs from its sibling
and is kept as-is.
