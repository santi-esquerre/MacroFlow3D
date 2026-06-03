Run the local WSL build cycle for MacroFlow3D.

Follow the `macroflow-build` skill. Steps:

1. Configure: `cmake --preset wsl-debug`
2. Build: `cmake --build build/wsl-debug -j`
3. Test: `ctest --test-dir build/wsl-debug --output-on-failure`
4. Smoke: `./build/wsl-debug/macroflow3d_pipeline apps/config_pspta_small.yaml`

Report the result at each step. Stop on first failure.

If any step fails, diagnose and report — do not silently retry.
