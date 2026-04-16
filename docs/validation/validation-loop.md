# Validation loop

The fixed validation loop for every MacroFlow3D change.

---

## The loop

```
configure → build → test → smoke → evals → PR
```

Every step must pass before proceeding to the next.

---

## Step 1: Configure

```bash
cmake --preset wsl-debug
```

**Pass:** CMake generation completes without error.

## Step 2: Build

```bash
cmake --build build/wsl-debug -j
```

**Pass:** compilation completes. No new errors in changed files.

## Step 3: Test

```bash
ctest --test-dir build/wsl-debug --output-on-failure
```

**Pass:** all registered tests pass.

## Step 4: Smoke

```bash
./build/wsl-debug/macroflow3d_pipeline apps/config_pspta_small.yaml
```

**Pass:** pipeline completes without crash or assertion failure.

## Step 5: Evals

Run the eval tier appropriate to the change:

| Change type | Tier |
|-------------|------|
| Docs / tooling | A (steps 1–4 are sufficient) |
| Operators / numerics | A + B |
| Physics / PSPTA | A + B + C |
| Macrodispersion output | A + B + C (with before/after) |

See `docs/validation/eval-tiers.md` for exact commands per tier.

## Step 6: PR

Create the PR only after steps 1–5 pass:

```bash
git push -u origin <branch>
gh pr create --fill
```

PR description must include evidence from the relevant tiers.

---

## Remote extension

For changes requiring V100 validation, insert after step 4:

```bash
scripts/rsync_to_v100.sh
scripts/remote_build_and_test.sh
scripts/remote_run_pipeline.sh apps/config_pspta_small.yaml
```

---

## Checklist form

- [ ] `cmake --preset wsl-debug` — configure OK
- [ ] `cmake --build build/wsl-debug -j` — build OK
- [ ] `ctest --test-dir build/wsl-debug --output-on-failure` — tests OK
- [ ] `./build/wsl-debug/macroflow3d_pipeline apps/config_pspta_small.yaml` — smoke OK
- [ ] Eval tier (A/B/C) commands run and passed
- [ ] PR created with evidence

---

## Related

- `docs/validation/eval-tiers.md`
- `docs/validation/acceptance-gates.md`
- `docs/validation/local-remote-split.md`
- `skills/macroflow-build/SKILL.md`
