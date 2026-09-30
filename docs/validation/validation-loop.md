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

Local WSL is for a fast, targeted subset only — used to iterate while editing:

```bash
ctest --test-dir build/wsl-debug --output-on-failure -R <fast-targeted-case>
```

**The full local suite is long-duration computation** (several registered
`ctest` entries are multi-case solver/continuation sweeps) and must not be run
as acceptance evidence. The authoritative full-suite pass is the remote
detached job in the "Remote extension" section below.

**Pass (local, fast subset):** the targeted case(s) pass.
**Pass (authoritative, full suite):** the remote detached `ctest` job
(Section "Remote extension") completes with all registered tests passing.

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

For changes requiring V100 validation — and for the authoritative full-suite
`ctest` pass in every case, since it is long-duration computation — insert
after step 4. Configure/build are short and bounded (`exec` is fine); the full
test pass and any pipeline run are long-duration and must be detached
(`run` + `wait`), per `docs/runbooks/remote-v100.md` Section 0:

```bash
scripts/remote sync
scripts/remote exec -- "cmake --preset v100-release && cmake --build build/v100-release -j"
scripts/remote run ctest-full -- "ctest --test-dir build/v100-release --output-on-failure"
scripts/remote wait ctest-full
scripts/remote run pspta-small -- "./build/v100-release/macroflow3d_pipeline apps/config_pspta_small.yaml"
scripts/remote wait pspta-small
```

---

## Checklist form

- [ ] `cmake --preset wsl-debug` — configure OK
- [ ] `cmake --build build/wsl-debug -j` — build OK
- [ ] `ctest --test-dir build/wsl-debug --output-on-failure -R <fast-targeted-case>` — local fast-subset OK
- [ ] `./build/wsl-debug/macroflow3d_pipeline apps/config_pspta_small.yaml` — smoke OK
- [ ] remote detached full `ctest` job (`scripts/remote run <job> -- "ctest ..."` + `scripts/remote wait <job>`) — full suite OK
- [ ] Eval tier (A/B/C) commands run and passed
- [ ] PR created with evidence

---

## Related

- `docs/validation/eval-tiers.md`
- `docs/validation/acceptance-gates.md`
- `docs/validation/local-remote-split.md`
- `skills/macroflow-build/SKILL.md`
