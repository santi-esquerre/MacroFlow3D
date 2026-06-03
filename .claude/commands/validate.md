Determine the correct validation tier for my current changes and execute it.

Steps:

1. Run `git diff --stat` to identify changed files.
2. Classify the change using acceptance gates from `docs/validation/acceptance-gates.md`.
3. Determine the eval tier from `docs/validation/eval-tiers.md`.
4. Execute the corresponding commands from the `macroflow-evals` skill.
5. Report: gate classification, tier, commands run, pass/fail for each.

If changes span multiple gates, use the highest applicable gate.

$ARGUMENTS
