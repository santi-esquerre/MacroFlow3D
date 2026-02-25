#pragma once

/**
 * @file ConfigDefaults.hpp
 * @brief Centralized default values for AppConfig.
 *
 * ALL defaults live here — no magic numbers scattered in the pipeline.
 * ConfigReaderYaml merges user YAML over these defaults.
 */

#include "Config.hpp"

namespace macroflow3d {
namespace io {

/// Schema version for the config format. Bump when adding/removing fields.
inline constexpr int kConfigSchemaVersion = 1;

/**
 * @brief Create a fully-populated default config.
 *
 * Every field has a sensible default. The YAML reader starts from this
 * and overwrites only the fields present in the user's file.
 */
inline AppConfig make_default_config() {
    AppConfig cfg;

    // Run mode
    cfg.run_mode = RunMode::Ensemble;

    // Grid
    cfg.grid.nx = 64;
    cfg.grid.ny = 64;
    cfg.grid.nz = 64;
    cfg.grid.dx = 1.0;

    // Stochastic
    cfg.stochastic.sigma2         = 1.0;
    cfg.stochastic.corr_length    = 1.0;
    cfg.stochastic.n_modes        = 1000;
    cfg.stochastic.covariance_type = 0;
    cfg.stochastic.seed           = 12345;
    cfg.stochastic.K_mean         = 1.0;

    // Flow
    cfg.flow.solver          = "mg";
    cfg.flow.mg_levels       = 4;
    cfg.flow.mg_pre_smooth   = 2;
    cfg.flow.mg_post_smooth  = 2;
    cfg.flow.mg_coarse_iters = 50;
    cfg.flow.mg_max_cycles   = 20;
    cfg.flow.cg_max_iter     = 1000;
    cfg.flow.cg_rtol         = 1e-8;
    cfg.flow.cg_check_every  = 10;
    cfg.flow.rtol            = 1e-6;
    cfg.flow.pin.mode        = PinMode::Auto;
    cfg.flow.verify_velocity = false;
    // BCs default to Dirichlet(0)

    // Transport
    cfg.transport.n_particles    = 10000;
    cfg.transport.dt             = 0.01;
    cfg.transport.n_steps        = 1000;
    cfg.transport.porosity       = 1.0;
    cfg.transport.diffusion      = 0.0;
    cfg.transport.alpha_l        = 0.0;
    cfg.transport.alpha_t        = 0.0;
    cfg.transport.seed           = 54321;
    cfg.transport.output_every   = 100;
    cfg.transport.snapshot_every = 0;
    cfg.transport.inject_x       = 0.0;
    cfg.transport.velocity_layout = "padded";

    // Analysis — macrodispersion
    cfg.analysis.macrodispersion.enabled       = false;
    cfg.analysis.macrodispersion.NR            = 1;
    cfg.analysis.macrodispersion.lambda        = 1.0;
    cfg.analysis.macrodispersion.vmean_norm    = 1.0;
    cfg.analysis.macrodispersion.sample_every  = 10;
    cfg.analysis.macrodispersion.var_estimator = "biased";

    // Analysis — snapshots
    cfg.analysis.snapshots.enabled             = false;
    cfg.analysis.snapshots.every               = 200;
    cfg.analysis.snapshots.legacy_format       = true;
    cfg.analysis.snapshots.include_time        = false;
    cfg.analysis.snapshots.include_status      = false;
    cfg.analysis.snapshots.include_wrap_counts = false;
    cfg.analysis.snapshots.include_unwrapped   = false;
    cfg.analysis.snapshots.stride              = 1;
    cfg.analysis.snapshots.max_particles       = -1;
    cfg.analysis.snapshots.precision           = 15;

    // Diagnostics
    cfg.diagnostics.velocity_field = false;

    // Output
    cfg.output.output_dir    = "./output";
    cfg.output.save_K        = true;
    cfg.output.save_head     = true;
    cfg.output.save_velocity = false;
    cfg.output.save_particles = true;
    cfg.output.format        = "binary";

    return cfg;
}

} // namespace io
} // namespace macroflow3d
