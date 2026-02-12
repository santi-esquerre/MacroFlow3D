#pragma once

/**
 * @file ConfigSerializer.hpp
 * @brief Serialize AppConfig → YAML text (effective_config.yaml).
 *
 * Writes every field with its resolved value so runs are 100 %
 * reproducible without guessing which defaults were applied.
 *
 * No yaml-cpp dependency — plain fprintf for minimal coupling.
 */

#include "../config/Config.hpp"
#include "../config/ConfigDefaults.hpp"
#include "../output_layout.hpp"
#include <cstdio>
#include <string>

namespace macroflow3d {
namespace io {

struct ConfigSerializer {

    /**
     * @brief Serialize full config to effective_config.yaml.
     *
     * @param layout  OutputLayout (provides path).
     * @param cfg     Resolved config.
     * @return true on success.
     */
    static bool write(const OutputLayout& layout, const AppConfig& cfg) {
        std::string path = layout.effective_config();
        FILE* f = std::fopen(path.c_str(), "w");
        if (!f) return false;

        std::fprintf(f, "# Effective configuration (auto-generated, schema_version=%d)\n",
                     kConfigSchemaVersion);
        std::fprintf(f, "# All values are resolved (user + defaults).\n\n");

        // Run mode
        const char* mode_str =
            cfg.run_mode == RunMode::SingleRun    ? "single_run" :
            cfg.run_mode == RunMode::AnalysisOnly ? "analysis_only" : "ensemble";
        std::fprintf(f, "run_mode: %s\n\n", mode_str);

        // Grid
        std::fprintf(f, "grid:\n");
        std::fprintf(f, "  nx: %d\n", cfg.grid.nx);
        std::fprintf(f, "  ny: %d\n", cfg.grid.ny);
        std::fprintf(f, "  nz: %d\n", cfg.grid.nz);
        std::fprintf(f, "  dx: %.15e\n\n", (double)cfg.grid.dx);

        // Stochastic
        std::fprintf(f, "stochastic:\n");
        std::fprintf(f, "  sigma2: %.15e\n", (double)cfg.stochastic.sigma2);
        std::fprintf(f, "  corr_length: %.15e\n", (double)cfg.stochastic.corr_length);
        std::fprintf(f, "  n_modes: %d\n", cfg.stochastic.n_modes);
        std::fprintf(f, "  covariance_type: %d\n", cfg.stochastic.covariance_type);
        std::fprintf(f, "  seed: %llu\n", (unsigned long long)cfg.stochastic.seed);
        std::fprintf(f, "  K_mean: %.15e\n\n", (double)cfg.stochastic.K_mean);

        // Flow
        std::fprintf(f, "flow:\n");
        std::fprintf(f, "  solver: %s\n", cfg.flow.solver.c_str());
        std::fprintf(f, "  mg_levels: %d\n", cfg.flow.mg_levels);
        std::fprintf(f, "  mg_pre_smooth: %d\n", cfg.flow.mg_pre_smooth);
        std::fprintf(f, "  mg_post_smooth: %d\n", cfg.flow.mg_post_smooth);
        std::fprintf(f, "  mg_coarse_iters: %d\n", cfg.flow.mg_coarse_iters);
        std::fprintf(f, "  mg_max_cycles: %d\n", cfg.flow.mg_max_cycles);
        std::fprintf(f, "  cg_max_iter: %d\n", cfg.flow.cg_max_iter);
        std::fprintf(f, "  cg_rtol: %.15e\n", (double)cfg.flow.cg_rtol);
        std::fprintf(f, "  cg_check_every: %d\n", cfg.flow.cg_check_every);
        std::fprintf(f, "  rtol: %.15e\n", (double)cfg.flow.rtol);
        std::fprintf(f, "  verify_velocity: %s\n",
                     cfg.flow.verify_velocity ? "true" : "false");

        // Pin
        const char* pin_mode =
            cfg.flow.pin.mode == PinMode::On ? "on" :
            cfg.flow.pin.mode == PinMode::Off ? "off" : "auto";
        std::fprintf(f, "  pin:\n");
        std::fprintf(f, "    mode: %s\n", pin_mode);

        // BCs
        auto bc_name = [](BCType t) -> const char* {
            switch (t) {
                case BCType::Dirichlet: return "dirichlet";
                case BCType::Neumann:   return "neumann";
                case BCType::Periodic:  return "periodic";
                default: return "dirichlet";
            }
        };
        auto write_face = [&](const char* name, const BCFace& face) {
            std::fprintf(f, "    %s: { type: %s, value: %.15e }\n",
                         name, bc_name(face.type), (double)face.value);
        };
        std::fprintf(f, "  bc:\n");
        write_face("xmin", cfg.flow.bc.xmin);
        write_face("xmax", cfg.flow.bc.xmax);
        write_face("ymin", cfg.flow.bc.ymin);
        write_face("ymax", cfg.flow.bc.ymax);
        write_face("zmin", cfg.flow.bc.zmin);
        write_face("zmax", cfg.flow.bc.zmax);
        std::fprintf(f, "\n");

        // Transport
        std::fprintf(f, "transport:\n");
        std::fprintf(f, "  n_particles: %d\n", cfg.transport.n_particles);
        std::fprintf(f, "  dt: %.15e\n", (double)cfg.transport.dt);
        std::fprintf(f, "  n_steps: %d\n", cfg.transport.n_steps);
        std::fprintf(f, "  porosity: %.15e\n", (double)cfg.transport.porosity);
        std::fprintf(f, "  diffusion: %.15e\n", (double)cfg.transport.diffusion);
        std::fprintf(f, "  alpha_l: %.15e\n", (double)cfg.transport.alpha_l);
        std::fprintf(f, "  alpha_t: %.15e\n", (double)cfg.transport.alpha_t);
        std::fprintf(f, "  seed: %llu\n", (unsigned long long)cfg.transport.seed);
        std::fprintf(f, "  output_every: %d\n", cfg.transport.output_every);
        std::fprintf(f, "  snapshot_every: %d\n", cfg.transport.snapshot_every);
        std::fprintf(f, "  inject_x: %.15e\n", (double)cfg.transport.inject_x);
        std::fprintf(f, "  velocity_layout: %s\n\n", cfg.transport.velocity_layout.c_str());

        // Analysis
        std::fprintf(f, "analysis:\n");
        std::fprintf(f, "  macrodispersion:\n");
        std::fprintf(f, "    enabled: %s\n", cfg.analysis.macrodispersion.enabled ? "true" : "false");
        std::fprintf(f, "    NR: %d\n", cfg.analysis.macrodispersion.NR);
        std::fprintf(f, "    lambda: %.15e\n", (double)cfg.analysis.macrodispersion.lambda);
        std::fprintf(f, "    vmean_norm: %.15e\n", (double)cfg.analysis.macrodispersion.vmean_norm);
        std::fprintf(f, "    sample_every: %d\n", cfg.analysis.macrodispersion.sample_every);
        std::fprintf(f, "    var_estimator: %s\n", cfg.analysis.macrodispersion.var_estimator.c_str());
        std::fprintf(f, "  snapshots:\n");
        std::fprintf(f, "    enabled: %s\n", cfg.analysis.snapshots.enabled ? "true" : "false");
        std::fprintf(f, "    every: %d\n", cfg.analysis.snapshots.every);
        std::fprintf(f, "    legacy_format: %s\n", cfg.analysis.snapshots.legacy_format ? "true" : "false");
        std::fprintf(f, "    include_time: %s\n", cfg.analysis.snapshots.include_time ? "true" : "false");
        std::fprintf(f, "    include_status: %s\n", cfg.analysis.snapshots.include_status ? "true" : "false");
        std::fprintf(f, "    include_wrap_counts: %s\n", cfg.analysis.snapshots.include_wrap_counts ? "true" : "false");
        std::fprintf(f, "    include_unwrapped: %s\n", cfg.analysis.snapshots.include_unwrapped ? "true" : "false");
        std::fprintf(f, "    stride: %d\n", cfg.analysis.snapshots.stride);
        std::fprintf(f, "    max_particles: %d\n", cfg.analysis.snapshots.max_particles);
        std::fprintf(f, "    precision: %d\n\n", cfg.analysis.snapshots.precision);

        // Output
        std::fprintf(f, "output:\n");
        std::fprintf(f, "  output_dir: %s\n", cfg.output.output_dir.c_str());
        std::fprintf(f, "  save_K: %s\n", cfg.output.save_K ? "true" : "false");
        std::fprintf(f, "  save_head: %s\n", cfg.output.save_head ? "true" : "false");
        std::fprintf(f, "  save_velocity: %s\n", cfg.output.save_velocity ? "true" : "false");
        std::fprintf(f, "  save_particles: %s\n", cfg.output.save_particles ? "true" : "false");
        std::fprintf(f, "  format: %s\n", cfg.output.format.c_str());

        std::fclose(f);
        return true;
    }
};

} // namespace io
} // namespace macroflow3d
