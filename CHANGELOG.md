# Changelog

All notable changes to this project will be documented in this file.

## [0.1.0] — 2025-06-12

### Added
- **Core pipeline**: K generation → head solve → velocity → RWPT transport.
- **GPU solver stack**: CG + CCMG V-cycle preconditioner for 3D Poisson/variable-coefficient Laplacian.
- **Par2_Core integration**: vendorized RWPT engine accessed exclusively through adapter layer.
- **YAML configuration**: strict parser with validation, known-key enforcement, effective_config.yaml serialization.
- **Output layout**: stable directory structure (`stats/`, `snapshots/`, `ensemble/`, `analysis/`).
- **I/O scheduler**: zero-alloc hot-loop with pinned staging buffers.
- **Particle moments collector**: async stats via Par2StatsAdapter.
- **Macrodispersion analysis**: ensemble mean, variance time-series, α(t) computation.
- **Manifest writer**: enriched JSON manifest with GPU info, build metadata, git hash.
- **Config validator**: pre-flight checks before any GPU allocation.
- **Stage profiler**: CUDA event timing with compile-time ON/OFF (`RWPT_ENABLE_PROFILING`).
- **NVTX markers**: Nsight Systems annotations with compile-time ON/OFF (`RWPT_ENABLE_NVTX`).
- **Run counters**: lightweight event counters (steps, stats, snapshots) with diagnostic-only report.
- **Performance contract**: `PERFORMANCE_CONTRACT.md` documenting all sync points, D2H/H2D, allocations.
- **Run modes**: `ensemble` (default), `single_run`, `analysis_only` via `run_mode` config key.
- **EnsembleRunner**: NR loop isolated in `src/runtime/ensemble/EnsembleRunner.cu`.
- **AnalysisRunner**: offline CPU-only macrodispersion recomputation from existing CSVs.
- **CMake build**: separable compilation, vendorized deps (yaml-cpp, nlohmann/json, Par2_Core).
