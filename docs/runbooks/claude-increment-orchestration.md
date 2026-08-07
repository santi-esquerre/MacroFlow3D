# Claude Code increment orchestration

## Purpose

This runbook describes the Claude Code execution harness used to deliver one
MacroFlow3D increment as an independently audited GitHub pull request.

It supplements, rather than replaces, the scientific increment workflow. The
Lester dashboard and increment specification remain authoritative for the Goal,
dependencies, checklist, acceptance gates, and advancement state.

## Roles

| Role | Model | Effort | Writes source? | Worktree | Publishes |
|---|---|---:|---|---|---|
| `orchestrator` | Claude Fable 5 | xhigh | only orchestration/durable record changes | control checkout | final PR only |
| `increment-worker` | Claude Sonnet 5 | medium | yes, one DAG node | native isolated worktree | never |
| `increment-integrator` | Claude Sonnet 5 | medium | integration/conflict changes only | native isolated worktree | never |

## State machine

```text
UNDERSTAND
  -> PLAN
  -> EXECUTE_DAG
  -> AUDIT
  -> (CORRECT -> AUDIT)*
  -> INTEGRATE
  -> FINAL_AUDIT
  -> (CORRECT -> REINTEGRATE -> FINAL_AUDIT)*
  -> PUBLISH_PR
  -> [AWAIT_HUMAN_REVIEW -> FINALIZE_CLOSURE]*
  -> READY_FOR_HUMAN_MERGE
```

Opening the audited PR finishes the autonomous **implementation** run. If human
review is required, the PR remains `awaiting_review`. After explicit approval,
Fable resumes the same PR and adds only the closure metadata commit; the result
is `READY_FOR_HUMAN_MERGE`. No agent merges. Advancement remains blocked until
the closure state is merged and visible on the default branch.

## DAG rules

A node must be self-contained and verifiable. Each node records:

- id/title/objective;
- dependencies;
- required context;
- predecessor commits;
- expected and forbidden write scopes;
- deliverables;
- acceptance criteria;
- validation commands.

Two nodes may run concurrently only if their dependencies are satisfied and
their write/external-state scopes are compatible.

## Commit flow

```text
increment base
   |-- worker T01 -> C01 -- accepted
   |-- worker T02 -> C02 -- accepted
   |       depends on C01? then T02 explicitly incorporates C01 first
   |-- corrective -> C03 -- accepted
   |
   `-- integrator worktree
          + approved commits only
          + integration fixes
          -> FINAL
```

Fable audits each candidate commit before it can enter the approved set.

## Runtime records

Ephemeral detailed records may be stored under:

```text
.claude/orchestration/<increment-id>/
  understanding.md
  dag.json
  audits/
  integration/
```

These records may be gitignored. Durable evidence must still be represented in
the versioned increment checklist/bitácora, experiment notes when required, and
PR description.

## Permission posture

The project requests `bypassPermissions`.

For a truly non-interactive local experience, the user-level Claude Code
settings must also enable `skipDangerousModePermissionPrompt`; repositories are
not allowed to suppress that warning themselves.

## Worktree posture

Both source-writing agent definitions use:

```yaml
isolation: worktree
```

Project settings use:

```json
{
  "worktree": {
    "baseRef": "head"
  }
}
```

Therefore each new worker/integrator worktree begins from the local HEAD of the
control session. Dependency commits are still passed explicitly: a dependent
worker must incorporate the approved predecessor hashes before implementing its
node.

## Publish posture

Only the orchestrator may:

```bash
git push -u origin <final-branch>
gh pr create ...
```

No agent auto-merges the pull request.

For human-review increments, publication is two-stage on the same PR: audited
source head -> human approval -> metadata-only closure head. The PR remains the
increment deliverable; a human performs merge.


## Human-review closure protocol

When `Human review: required`:

1. FINAL_AUDIT freezes the accepted source-bearing commit.
2. Publish/update the PR with increment state `awaiting_review` and `NEXT`
   unchanged.
3. Human approves that exact source-bearing head.
4. Resume the same PR.
5. Add only closure metadata: complete checklist, `State: done`, final bitacora,
   dashboard checkmark/`Last completed`/`NEXT`, active goal cleanup.
6. Run `bash scripts/hooks/check-lester-increments.sh`.
7. Push the metadata-only commit.
8. Human merges.

Any post-approval source/test/config change invalidates approval and returns to
corrective audit/review.

## Stale merged-state repair

If a PR was merged while its increment was still `awaiting_review`, use a
metadata-only closure-repair PR. Verify the actual merged commit, record the
human merge fact without inventing a review object, repair the increment and
dashboard state, run the checker, and leave merge of the repair PR to a human.
