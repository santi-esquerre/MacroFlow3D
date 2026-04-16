#pragma once

/**
 * @file PipelineRunner.hpp
 * @brief Orchestrator for the K → head → velocity → RWPT transport pipeline
 * @ingroup runtime_pipeline
 *
 * Owns all GPU buffers, workspaces, and engine state.  The app layer
 * constructs one PipelineRunner and calls run().
 *
 * HPC contract:
 *   - All device memory allocated in the constructor (one-shot).
 *   - Hot loop (engine.step) is allocation-free and sync-free.
 *   - Stream sync happens ONLY at sample/snapshot events.
 */

#include "../../io/config/Config.hpp"
#include "../CudaContext.cuh"
#include "../StageProfiler.cuh"

namespace macroflow3d {
namespace pipeline {

/**
 * @brief Print a human-readable summary of the configuration to stdout.
 */
void print_config_summary(const io::AppConfig& cfg);

/**
 * @brief Run the full MacroFlow3D macrodispersion pipeline.
 *
 * Performs NR realizations of:
 *   (1) generate lognormal K,
 *   (2) solve steady-state head,
 *   (3) compute Darcy velocity,
 *   (4) RWPT transport with Par2_Core,
 *   (5) stats sampling + optional CSV snapshots.
 * Then computes macrodispersivity α(t) from variance time-series.
 *
 * @param cfg       Fully-populated application config (from YAML).
 * @param ctx       Initialised CUDA context (stream + cublas).
 * @param profiler  Stage profiler (may be no-op in Release).
 * @return EXIT_SUCCESS on completion, EXIT_FAILURE on fatal error.
 */
int run_pipeline(const io::AppConfig& cfg, CudaContext& ctx, StageProfiler& profiler);

} // namespace pipeline
} // namespace macroflow3d
