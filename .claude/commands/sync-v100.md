Sync the current worktree to V100 and run a remote build.

Follow the `macroflow-remote-v100` skill. Steps:

1. Sync: `scripts/rsync_to_v100.sh`
2. Build and test: `scripts/remote_build_and_test.sh`
3. Report sync status, build result, and test results.

If PETSc/SLEPc build is needed, use:
`BUILD_DIR=build/v100-petsc ENABLE_PETSC=ON scripts/remote_build_and_test.sh`

$ARGUMENTS
