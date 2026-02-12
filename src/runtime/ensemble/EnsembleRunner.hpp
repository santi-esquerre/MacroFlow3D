#pragma once

/**
 * @file EnsembleRunner.hpp
 * @brief Orchestrates NR realizations, accumulates series, writes ensemble.
 *
 * Etapa 10: the NR loop lives ONLY here.
 *
 * Responsibilities:
 *   - Derive per-realization seeds (base_seed + r)
 *   - Set up OutputLayout + one-shot workspace allocation
 *   - Invoke the single-realization transport loop NR times
 *   - Accumulate time-series for ensemble mean + macrodispersion
 *   - Write manifest, effective_config, ensemble_mean, macrodispersion
 *
 * HPC contract:
 *   - GPU workspaces allocated once, reused across realizations.
 *   - No allocations inside the hot loop (see PERFORMANCE_CONTRACT.md).
 */

#include "../../io/config/Config.hpp"
#include "../CudaContext.cuh"
#include "../StageProfiler.cuh"
#include "../RunCounters.hpp"

namespace macroflow3d {
namespace ensemble {

/**
 * @brief Run the full ensemble pipeline (NR realizations).
 *
 * This is the main entry point for RunMode::Ensemble and RunMode::SingleRun.
 * Allocates all GPU resources once, runs NR realizations, computes analysis.
 *
 * @param cfg       Fully-validated config.
 * @param ctx       CUDA context (stream + cublas).
 * @param profiler  Stage profiler (no-op if MACROFLOW3D_ENABLE_PROFILING=0).
 * @return EXIT_SUCCESS or EXIT_FAILURE.
 */
int run_ensemble(const io::AppConfig& cfg,
                 CudaContext& ctx,
                 StageProfiler& profiler);

} // namespace ensemble
} // namespace macroflow3d
