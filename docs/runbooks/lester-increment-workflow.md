# Lester increment workflow

## Purpose

Use this runbook for every increment in
`docs/plans/active/lester-eq14-streamfunction-solver-plan.md`.

It keeps the versioned dashboard/specification, Claude runtime orchestration,
validation evidence, human-review lifecycle, and default-branch advancement
consistent.

It does **not** authorize working ahead of the dashboard's `NEXT` pointer.

The fundamental scheduling rule is:

- increments are strictly sequential;
- the one active increment may be decomposed into an internal DAG;
- independent DAG nodes may execute concurrently in isolated worktrees.

---

## 1. Select the increment

From the up-to-date default branch:

1. read the dashboard and the complete specification linked by `NEXT`;
2. verify every `Depends on` increment is `done`;
3. read the solver overview, relevant theory note, acceptance gates, and local
   `AGENTS.md` files;
4. run:

```bash
bash scripts/hooks/check-lester-increments.sh
```

Stop if the checker or dependency test fails.

Record the exact default-branch commit as the **increment base**.

There may be only one nonterminal Lester increment at a time.

---

## 2. Establish the runtime goal and orchestration record

Use the exact `Goal` sentence from the increment specification as the runtime
objective.

The Fable orchestrator creates/maintains local runtime evidence under:

```text
.claude/orchestration/<increment-id>/
  understanding.md
  dag.json
  audits/
  integration/
```

This runtime record may be gitignored. It exists to support concurrency,
auditing, recovery, and later durable reporting.

Do not create a single shared source worktree for the entire increment when
using the Claude orchestrator. Source-writing subagents use native
`isolation: worktree`:

- one isolated worktree per worker/corrective node;
- one fresh isolated worktree for the final integrator.

The increment specification's `Branch` identifies the delivery/PR branch.
Temporary worker branches/worktrees are execution details recorded by the
orchestrator.

A runtime that supports persistent goals should keep the exact Goal active until
the increment is merged and visible as `done` on the default branch. If no
persistent-goal feature is available, record that fact in the orchestration
record rather than inventing one.

---

## 3. UNDERSTAND before implementation

Fable must understand enough of the project to independently judge the result.
For scientific/numerical increments this includes, as applicable:

- theoretical and physical assumptions;
- equations and sign conventions;
- discrete operator/stencil contracts;
- boundary conditions, gauges, nullspaces, and projections;
- solver tolerances/convergence criteria;
- precision and memory constraints;
- CPU/GPU ownership, synchronization, and allocation rules;
- acceptance gates and regression surface;
- explicit out-of-scope behavior.

Inspect both documentation and actual code/tests. Record contradictions instead
of silently choosing one.

---

## 4. Build the intra-increment DAG

Fable constructs an acyclic, decision-complete DAG whose accepted result is
sufficient for the increment Goal.

Each node records at least:

- id and objective;
- dependencies;
- required context;
- required predecessor commits;
- expected write scope;
- forbidden scope;
- deliverables;
- acceptance criteria;
- validation commands.

Create a coverage map from every increment acceptance/checklist item to one or
more DAG nodes or final integration checks.

### Parallelism

A node is ready only after every dependency is accepted.

Independent ready nodes may execute concurrently only when their write scopes
and external state do not conflict.

Do not parallelize dependent nodes. Do not serialize independent work without a
reason.

---

## 5. Execute workers in isolated worktrees

Every implementation/corrective node is delegated to `increment-worker`
(Sonnet 5 / medium).

The delegation must be self-contained because workers do not inherit the parent
conversation. Supply:

- increment id and exact Goal;
- task id/objective;
- relevant theory/physics/mathematics/numerics/computation context;
- exact accepted predecessor commits;
- allowed and forbidden scope;
- deliverables;
- acceptance criteria;
- validation commands;
- known risks.

A dependent worker explicitly incorporates only the approved predecessor commits
it needs.

Each worker commits its result and reports the exact commit. Workers never push,
open PRs, or integrate unrelated workers.

---

## 6. Audit and corrective cycles

Worker `success` is not acceptance.

Fable independently:

1. inspects every candidate commit and full diff;
2. checks scope and surrounding interfaces;
3. verifies scientific/numerical/computational contracts;
4. reruns required validation;
5. evaluates every node acceptance criterion.

Findings are classified `BLOCKING`, `MAJOR`, `MINOR`, or `INFORMATIONAL`.

A node is accepted only when no blocking/major findings remain and all required
criteria have evidence.

Failures produce a corrective DAG. Corrective Sonnet workers run in isolated
worktrees and are audited exactly like original nodes. Repeat until accepted or
explicitly `blocked`.

---

## 7. Integrate accepted work

After all required nodes are accepted, launch exactly one
`increment-integrator` (Sonnet 5 / medium) in a fresh isolated worktree.

Give it:

- increment base;
- exact Goal/specification;
- approved task -> commit mapping;
- dependency-compatible integration order;
- relevant audit results;
- complete increment acceptance criteria;
- complete validation commands.

The integrator incorporates **only** approved commits, resolves semantic/textual
conflicts, runs full validation, and returns an exact integrated commit.

It does not push or create a PR.

---

## 8. Final audit and durable evidence

Fable audits the integrated result against the original increment Goal, not only
worker summaries.

Verify:

- full diff from the increment base;
- all expected accepted work is represented;
- no unapproved/out-of-scope functionality exists;
- complete build/tests/gates pass;
- numerical/scientific metrics satisfy the increment thresholds;
- memory/performance evidence is present when required;
- versioned documentation accurately records the accepted result.

Durable increment evidence is reconstructed/summarized from worker reports and
runtime audit records into the append-only bitacora before publication. Do not
have concurrent workers append to the same versioned bitacora file.

Run the fixed baseline plus exact increment commands:

```bash
bash scripts/hooks/check-lester-increments.sh
cmake --preset wsl-debug
cmake --build build/wsl-debug -j
ctest --test-dir build/wsl-debug --output-on-failure
```

Add the increment's Gate 2/3A/4/V100/experiment commands as required.

Never check a validation item for a command that was not run. Invalid,
interrupted, or superseded attempts may remain in the bitacora but must be
explicitly excluded from acceptance evidence.

---

## 9. Publish according to review class

The orchestrator is the only agent allowed to push/update the delivery branch
and PR.

### 9A. High-autonomy increment

After positive final audit:

1. complete every checklist item that is actually satisfied;
2. set the increment `State` to `done`;
3. record the final audited commit/PR metadata available at publication time;
4. append the closure bitacora row;
5. check its dashboard entry;
6. set `Last completed increment` to this increment;
7. set `NEXT` to the first remaining pending increment (or `COMPLETE`);
8. set `Active runtime goal` to `none` until another increment is actually
   activated;
9. run the harness checker;
10. push and open the PR.

The new `NEXT` exists only on the PR branch until a human merges it, so the
default branch still prevents premature advancement.

### 9B. Human-review increment

After positive Fable final audit:

1. freeze the accepted source result;
2. set/keep `State: awaiting_review`;
3. record the PR and exact source-bearing audited head;
4. append review-ready evidence to the bitacora;
5. **do not** mark the final human-review checklist item complete;
6. **do not** advance the dashboard `NEXT` yet;
7. push/update the PR for human review.

If review requests source/test/config changes, return to corrective DAG -> audit
-> integration -> final audit and publish a new source-bearing head for review.

---

## 10. Finalize after human approval — same PR

When a human explicitly approves the exact audited source-bearing PR head,
resume the same PR branch.

The orchestrator may now make **only a closure metadata commit**. It must:

1. record the human approval fact accurately;
2. complete all remaining checklist items;
3. set `State: done`;
4. record the PR and final audited source head/closure metadata as defined by the
   increment;
5. append a final bitacora row;
6. check the increment in the dashboard master checklist;
7. set `Last completed increment` to the increment;
8. set `NEXT` to the first remaining pending increment;
9. set `Active runtime goal` to `none` until the next increment actually starts;
10. run:

```bash
bash scripts/hooks/check-lester-increments.sh
```

11. push the closure-only commit to the same PR.

Do not change scientific source, tests, physics configs, or numerical behavior in
this step. Any such change invalidates the existing human approval and returns
the increment to audit/review.

A human then merges the PR.

---

## 11. Default-branch advancement

The default branch remains authoritative.

Do not start the next increment merely because the delivery branch says `done`.
After human merge:

1. update/fetch the default branch;
2. verify the closure state is actually visible there;
3. run the harness checker;
4. verify the new `NEXT` and predecessor `done` state;
5. only then complete/clear the previous persistent runtime goal and begin the
   next increment.

---

## 12. Exceptional closure repair

Use this only when an implementation PR was already merged while the increment
remained nonterminal (`awaiting_review`, `validating`, etc.).

Do **not** infer the next increment is enabled from the source code alone.
Create a metadata-only closure-repair PR:

1. verify on GitHub/default branch that the implementation PR is merged;
2. identify the canonical default-branch commit containing it;
3. verify the recorded final audit/evidence still corresponds to that accepted
   implementation;
4. record human approval truthfully — a manual owner merge may establish human
   approval even if no separate GitHub review event exists; do not claim a review
   object that does not exist;
5. set the increment `done`;
6. complete its checklist;
7. append a closure-repair bitacora row explaining why repair was necessary;
8. check its dashboard entry and advance `NEXT`;
9. update `Last completed increment` and clear the active runtime goal;
10. run the harness checker;
11. publish the metadata-only repair PR;
12. human merges it.

No scientific source changes are allowed in a closure repair.

### Historical repair: SF-08

PR #18 was merged into `master` while SF-08 was still `awaiting_review`.
The canonical squash/default-branch commit is:

`855dcf14458d3ac92ef31a7a30e373d5d4b16a1b`

Formal closure therefore consists only of recording that merged/approved result,
marking SF-08 `done`, and advancing the dashboard to SF-09.

---

## 13. Resume interrupted work

On resume:

1. read the increment metadata and entire durable bitacora;
2. inspect `.claude/orchestration/<increment-id>/` when it exists;
3. inspect the live PR and refs rather than recreating work;
4. recover worker/integrator commit hashes;
5. rerun only inconclusive validation whose artifacts are unavailable;
6. continue from the last proven state.

Never infer completion from code presence alone. Completion comes from the
versioned checklist, evidence, required review, merge, and default-branch state.
