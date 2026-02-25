/**
 * @file PipelineRunner.cu
 * @brief Thin orchestration shim — dispatches to EnsembleRunner.
 *
 * Etapa 10: all heavy work (NR loop, workspace allocation, analysis)
 * now lives in EnsembleRunner.cu.  PipelineRunner keeps ONLY:
 *   - print_config_summary()        — human-readable summary
 *   - run_pipeline()                — thin delegate to run_ensemble()
 */

#include "PipelineRunner.hpp"
#include "OutputPaths.hpp"
#include "../ensemble/EnsembleRunner.hpp"

#include <cstdio>
#include <cstdlib>

using namespace macroflow3d::io;

namespace macroflow3d {
namespace pipeline {

// ============================================================================
// print_config_summary
// ============================================================================

void print_config_summary(const AppConfig& cfg) {
    print_separator();
    std::printf("  MacroFlow3D Pipeline — Configuration Summary\n");
    print_separator();
    std::printf("  Grid:        %d × %d × %d  (dx=%.6f)\n",
                cfg.grid.nx, cfg.grid.ny, cfg.grid.nz, cfg.grid.dx);
    std::printf("  Domain:      [0, %.4f] × [0, %.4f] × [0, %.4f]\n",
                cfg.grid.Lx(), cfg.grid.Ly(), cfg.grid.Lz());
    std::printf("  Stochastic:  σ²=%.2f  λ=%.4f  modes=%d  seed=%lu\n",
                cfg.stochastic.sigma2, cfg.stochastic.corr_length,
                cfg.stochastic.n_modes, (unsigned long)cfg.stochastic.seed);
    std::printf("  Solver:      %s  rtol=%.1e\n",
                cfg.flow.solver.c_str(), cfg.flow.rtol);
    std::printf("  Transport:   N=%d  dt=%.4e  steps=%d  Dm=%.2e  αL=%.2e  αT=%.2e\n",
                cfg.transport.n_particles, cfg.transport.dt, cfg.transport.n_steps,
                cfg.transport.diffusion, cfg.transport.alpha_l, cfg.transport.alpha_t);

    const auto& mac = cfg.analysis.macrodispersion;
    if (mac.enabled) {
        std::printf("  Analysis:    NR=%d  λ_macro=%.4f  ||<v>||=%.4f  sample_every=%d  estimator=%s\n",
                    mac.NR, mac.lambda, mac.vmean_norm, mac.sample_every,
                    mac.var_estimator.c_str());
    } else {
        std::printf("  Analysis:    macrodispersion OFF (single realization)\n");
    }
    const auto& snap = cfg.analysis.snapshots;
    if (snap.enabled) {
        std::printf("  Snapshots:   every=%d  stride=%d  unwrapped=%s\n",
                    snap.every, snap.stride, snap.include_unwrapped ? "yes" : "no");
    }
    std::printf("  Output:      %s\n", cfg.output.output_dir.c_str());
    if (cfg.diagnostics.velocity_field) {
        std::printf("  Diagnostics: velocity_field ON (div/|ω|/helicity per realization)\n");
    }
    print_separator();
    print_separator();
}

// ============================================================================
// run_pipeline  —  thin delegate (Etapa 10)
// ============================================================================

int run_pipeline(const AppConfig& cfg,
                 CudaContext& ctx,
                 StageProfiler& profiler)
{
    return ensemble::run_ensemble(cfg, ctx, profiler);
}

} // namespace pipeline
} // namespace macroflow3d
