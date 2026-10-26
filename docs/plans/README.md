# Plans

Use this directory for execution plans that are too detailed to keep in chat.

Recommended convention:
- one plan per file
- date-prefixed filenames
- short status marker in the title

Suggested filename pattern:
```text
YYYY-MM-DD-short-topic.md
```

Each plan should include:
- objective
- scope
- files/subsystems
- commands
- validation
- risks
- done criteria

## Active authority

- `active/lester-eq14-streamfunction-solver-plan.md` is authoritative for new invariant-construction work.
- `active/lester-eq14/increments/` contains the decision-complete,
  dependency-ordered increment specifications and their append-only work logs.
  At most two increments may be nonterminal at once; the checker
  (`scripts/hooks/check-lester-increments.sh`) reports the READY set.
- `active/audit-cleanup-2026-10.md` is the maintenance plan of the 2026-10 codebase audit and cleanup
  (decision `docs/decisions/2026-10-26-audit-cleanup.md`); it is not a Lester increment.
- `archive/pspta-execution-plan.md` and `archive/deep-research-report.md` are historical context only. They are not active implementation plans.
