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
    cfg.stochastic.sigma2 = 1.0;
    cfg.stochastic.corr_length = 1.0;
    cfg.stochastic.n_modes = 1000;
    cfg.stochastic.covariance_type = 0;
    cfg.stochastic.seed = 12345;
    cfg.stochastic.K_mean = 1.0;

    // Flow
    cfg.flow.solver = "mg";
    cfg.flow.mg_levels = 4;
    cfg.flow.mg_pre_smooth = 2;
    cfg.flow.mg_post_smooth = 2;
    cfg.flow.mg_coarse_iters = 50;
    cfg.flow.mg_max_cycles = 20;
    cfg.flow.cg_max_iter = 1000;
    cfg.flow.cg_rtol = 1e-8;
    cfg.flow.cg_check_every = 10;
    cfg.flow.rtol = 1e-6;
    cfg.flow.pin.mode = PinMode::Auto;
    cfg.flow.verify_velocity = false;
    // BCs default to Dirichlet(0)

    // Transport
    cfg.transport.n_particles = 10000;
    cfg.transport.dt = 0.01;
    cfg.transport.n_steps = 1000;
    cfg.transport.porosity = 1.0;
    cfg.transport.diffusion = 0.0;
    cfg.transport.alpha_l = 0.0;
    cfg.transport.alpha_t = 0.0;
    cfg.transport.seed = 54321;
    cfg.transport.output_every = 100;
    cfg.transport.snapshot_every = 0;
    cfg.transport.inject_x = 0.0;
    cfg.transport.method = "par2"; // supported: "par2" | "pspta"
    cfg.transport.pspta_diagnostics = false;
    cfg.transport.pspta_refine.enabled = false;
    cfg.transport.pspta_refine.outer_iters = 5;
    cfg.transport.pspta_refine.omega = 0.5;
    cfg.transport.pspta_refine.omega_min = 1.0e-6;
    cfg.transport.pspta_refine.max_backtracks = 18;
    cfg.transport.pspta_refine.eps_vx = 1.0e-10;
    cfg.transport.pspta_refine.source_clip_cells = 0.1;
    cfg.transport.pspta_refine.no_descent_patience = 4;
    cfg.transport.pspta_refine.stop_rel_rms = 0.25;
    cfg.transport.pspta_refine.stop_abs_rms = 1.0e-6;
    cfg.transport.pspta_refine.print_every_iter = true;
    cfg.transport.pspta_refine.save_history_csv = true;
    cfg.transport.pspta_refine.eq13_diagnostics = false;
    // velocity_layout is derived from method — not set here

    // Analysis — macrodispersion
    cfg.analysis.macrodispersion.enabled = false;
    cfg.analysis.macrodispersion.NR = 1;
    cfg.analysis.macrodispersion.lambda = 1.0;
    cfg.analysis.macrodispersion.vmean_norm = 1.0;
    cfg.analysis.macrodispersion.sample_every = 10;
    cfg.analysis.macrodispersion.var_estimator = "biased";

    // Analysis — snapshots
    cfg.analysis.snapshots.enabled = false;
    cfg.analysis.snapshots.every = 200;
    cfg.analysis.snapshots.legacy_format = true;
    cfg.analysis.snapshots.include_time = false;
    cfg.analysis.snapshots.include_status = false;
    cfg.analysis.snapshots.include_wrap_counts = false;
    cfg.analysis.snapshots.include_unwrapped = false;
    cfg.analysis.snapshots.stride = 1;
    cfg.analysis.snapshots.max_particles = -1;
    cfg.analysis.snapshots.precision = 15;

    // Diagnostics
    cfg.diagnostics.velocity_field = false;

    // Output
    cfg.output.output_dir = "./output";
    cfg.output.save_K = true;
    cfg.output.save_head = true;
    cfg.output.save_velocity = false;
    cfg.output.save_particles = true;
    cfg.output.format = "binary";

    // Streamfunction solver (SF-16 T01) — disabled by default, no behavior
    // change for existing configs.
    cfg.streamfunction_solver.enabled = false;
    cfg.streamfunction_solver.affine_mean_velocity.mode = "fixed";
    cfg.streamfunction_solver.affine_mean_velocity.value = 1.0;
    cfg.streamfunction_solver.epsilon = 1.0e-2;
    cfg.streamfunction_solver.eta = 1.0;
    cfg.streamfunction_solver.picard.max_iter = 500;
    cfg.streamfunction_solver.picard.tolerance = 1.0e-6;
    cfg.streamfunction_solver.picard.omega = 0.25;
    cfg.streamfunction_solver.adaptive.enabled = true;
    cfg.streamfunction_solver.linear.rtol = 1.0e-10;
    cfg.streamfunction_solver.linear.max_iter = 1000;
    cfg.streamfunction_solver.linear.check_every = 10;
    cfg.streamfunction_solver.mg.num_levels = 4;
    cfg.streamfunction_solver.exports.iteration_history = true;
    cfg.streamfunction_solver.exports.summary = true;
    cfg.streamfunction_solver.exports.fields = false;

    // Streamfunction solver continuation (SF-17 T03) — disabled by default,
    // no behavior change: the SF-16 single-solve path stays byte-identical.
    cfg.streamfunction_solver.continuation.enabled = false;
    cfg.streamfunction_solver.continuation.eta.start = 0.0;
    cfg.streamfunction_solver.continuation.eta.initial_step = 0.1;
    cfg.streamfunction_solver.continuation.eta.min_step = 0.0125;
    cfg.streamfunction_solver.continuation.eta.max_step = 0.25;
    cfg.streamfunction_solver.continuation.eta.backtrack_factor = 0.5;
    cfg.streamfunction_solver.continuation.eta.growth_factor = 1.5;
    cfg.streamfunction_solver.continuation.eta.easy_streak = 2;
    cfg.streamfunction_solver.continuation.epsilon.target = 1.0e-6;
    cfg.streamfunction_solver.continuation.epsilon.initial_step_log10 = 1.0;
    cfg.streamfunction_solver.continuation.epsilon.min_step_log10 = 0.125;
    cfg.streamfunction_solver.continuation.epsilon.max_step_log10 = 1.0;
    cfg.streamfunction_solver.continuation.epsilon.backtrack_factor = 0.5;
    cfg.streamfunction_solver.continuation.epsilon.growth_factor = 1.5;
    cfg.streamfunction_solver.continuation.epsilon.easy_streak = 2;

    return cfg;
}

} // namespace io
} // namespace macroflow3d
