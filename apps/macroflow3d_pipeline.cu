/**
 * @file macroflow3d_pipeline.cu
 * @brief Thin entry-point for the MacroFlow3D macrodispersion pipeline
 *
 * Dispatches by RunMode (Etapa 10):
 *   - Ensemble / SingleRun → init CUDA → run_pipeline()
 *   - AnalysisOnly         → no GPU    → run_analysis()
 *
 * All orchestration logic lives in EnsembleRunner / AnalysisRunner.
 * This file only parses CLI args, loads YAML config, and delegates.
 *
 * Usage:
 *   ./macroflow3d_pipeline config.yaml
 */

#include <cstdio>
#include <cstdlib>
#include <string>

#include "src/io/config/Config.hpp"
#include "src/runtime/analysis/AnalysisRunner.hpp"
#include "src/runtime/pipeline/PipelineRunner.hpp"

using namespace macroflow3d;

int main(int argc, char** argv) {
    // ── 1. Parse arguments ─────────────────────────────────────────────
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <config.yaml>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const std::string config_path = argv[1];
    std::printf("[1] Loading config: %s\n", config_path.c_str());
    io::AppConfig cfg = io::load_config_yaml(config_path);

    // ── 2. Dispatch by RunMode ─────────────────────────────────────────
    if (cfg.run_mode == io::RunMode::AnalysisOnly) {
        // Pure-CPU path — no CUDA initialization needed
        std::printf("[mode] analysis_only — skipping GPU init\n");
        pipeline::print_config_summary(cfg);
        return analysis_runner::run_analysis(cfg);
    }

    // ── 3. Init CUDA (Ensemble / SingleRun) ────────────────────────────
    std::printf("[2] Initializing CUDA context\n");
    CudaContext ctx(0);
    StageProfiler profiler(ctx);

    // ── 4. Print summary & run ─────────────────────────────────────────
    pipeline::print_config_summary(cfg);
    return pipeline::run_pipeline(cfg, ctx, profiler);
}
