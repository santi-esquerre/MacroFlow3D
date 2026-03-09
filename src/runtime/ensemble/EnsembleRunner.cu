/**
 * @file EnsembleRunner.cu
 * @brief Orchestrates NR realizations and ensemble analysis.
 *
 * Etapa 10: the realization loop lives ONLY here.
 *
 * Allocates all GPU workspaces ONCE, runs NR realizations of
 * K → head → velocity → RWPT, accumulates stats, and writes
 * ensemble results + macrodispersion.
 */

#include "EnsembleRunner.hpp"

// I/O — config validation + serialization (Etapa 7)
#include "../../io/config/ConfigValidator.hpp"
#include "../../io/writers/ConfigSerializer.hpp"

// I/O — output layout + writers + scheduler (Etapa 5 + 8)
#include "../../io/output_layout.hpp"
#include "../../io/writers/ManifestWriter.hpp"
#include "../../io/writers/BuildInfo.hpp"
#include "../../io/writers/CsvTimeSeriesWriter.hpp"
#include "../../runtime/io/IOScheduler.hpp"

// Runtime stats collector (Etapa 6)
#include "../../runtime/stats/ParticleMomentsCollector.hpp"

// Analysis — macrodispersion (Etapa 6, CPU-only)
#include "../../analysis/macrodispersion/MacrodispersionAnalysis.hpp"

// Physics — fields & workspaces
#include "../../physics/common/fields.cuh"
#include "../../physics/common/workspaces.cuh"
#include "../../physics/common/physics_config.hpp"

// Physics — stages
#include "../../physics/stochastic/stochastic.cuh"
#include "../../physics/flow/solve_head.cuh"
#include "../../physics/flow/velocity_from_head.cuh"
#include "../../physics/flow/velocity_diagnostics.cuh"

// Par2 adapters (no <par2_core/...> included here)
#include "../../physics/particles/par2_adapter/Par2TransportAdapter.hpp"
#include "../../physics/particles/par2_adapter/par2_views.hpp"

// PSPTA engine + precomputed ψ fields
#include "../../physics/particles/pspta/PsptaEngine.hpp"
#include "../../physics/particles/pspta/PsptaPsiField.cuh"

// PSPTA runtime diagnostics writer
#include "../../runtime/io/CsvDiagnosticsWriter.hpp"

// Runtime — counters + NVTX + printing helpers
#include "../../runtime/RunCounters.hpp"
#include "../../runtime/nvtx_range.cuh"
#include "../pipeline/OutputPaths.hpp"  // print_separator()

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

using namespace macroflow3d;
using namespace macroflow3d::io;
using namespace macroflow3d::physics;

namespace macroflow3d {
namespace ensemble {

// ============================================================================
// Internal helpers
// ============================================================================

static StochasticConfig make_stochastic_cfg(const StochasticYamlConfig& y) {
    StochasticConfig c;
    c.sigma2           = y.sigma2;
    c.corr_length      = y.corr_length;
    c.n_modes          = y.n_modes;
    c.covariance_type  = y.covariance_type;
    c.seed             = y.seed;
    c.K_geometric_mean = y.K_mean;
    return c;
}

static bool bc_has_any_periodic(const BCSpec& bc) {
    return bc.xmin.type == BCType::Periodic
        || bc.ymin.type == BCType::Periodic
        || bc.zmin.type == BCType::Periodic;
}

// ============================================================================
// run_ensemble
// ============================================================================

int run_ensemble(const AppConfig& cfg,
                 CudaContext& ctx,
                 StageProfiler& profiler)
{
    // ── Validate config BEFORE any allocation (Etapa 7) ──────────────
    require_valid_config(cfg);

    const auto& mac  = cfg.analysis.macrodispersion;
    const auto& snap = cfg.analysis.snapshots;
    const int NR = mac.enabled ? mac.NR : 1;

    // ── Output layout ────────────────────────────────────────────────
    OutputLayout layout(cfg.output.output_dir);
    layout.ensure_all_dirs();

    // Write effective config (Etapa 7)
    ConfigSerializer::write(layout, cfg);

    // Write enriched manifest (Etapa 8)
    GPUInfo gpu = GPUInfo::query();
    ManifestWriter::write(layout, cfg, gpu);

    // ── Event counters (Etapa 9) ─────────────────────────────────────
    runtime::RunCounters counters;

    // ── Allocate fields & workspaces (ONE-SHOT) ──────────────────────
    MACROFLOW3D_NVTX_PUSH("allocate_fields");
    std::printf("[3] Allocating fields and workspaces\n");
    const int nx = cfg.grid.nx, ny = cfg.grid.ny, nz = cfg.grid.nz;
    const real dx = cfg.grid.dx;
    const Grid3D grid(nx, ny, nz, dx, dx, dx);
    const size_t N = grid.num_cells();

    StochasticConfig stoch_cfg = make_stochastic_cfg(cfg.stochastic);
    HeadSolveConfig  head_cfg  = HeadSolveConfig::from_yaml(cfg.flow);

    // Cell-centered fields (reused across realizations)
    ScalarField K_field(grid);
    ScalarField head_field(grid);

    // Workspaces
    StochasticWorkspace stoch_ws;
    stoch_ws.allocate(grid, stoch_cfg);

    FlowWorkspace flow_ws;
    flow_ws.allocate(grid, head_cfg.mg_levels);

    // Velocity (padded for Par2_Core; compact/CompactMAC for PSPTA)
    PaddedVelocityField vel(grid);
    VelocityField vel_compact;  // default-constructed (no alloc); resized below if pspta
    if (cfg.transport.method == "pspta") {
        vel_compact = VelocityField(grid);
    }

    // Velocity diagnostics workspace (allocated once, reused across realizations)
    VelocityDiagnostics vel_diag;
    if (cfg.diagnostics.velocity_field) {
        vel_diag.resize(grid);
    }

    // Particles (device SoA — reused across realizations)
    const int NP = cfg.transport.n_particles;
    DeviceBuffer<real> px(NP), py(NP), pz(NP);

    // Wrap counters — needed for unwrapped stats when periodic
    const bool any_periodic = bc_has_any_periodic(cfg.flow.bc);
    DeviceBuffer<int32_t> wrapX, wrapY, wrapZ;
    if (any_periodic) {
        wrapX.resize(NP); wrapY.resize(NP); wrapZ.resize(NP);
    }

    // Status array
    DeviceBuffer<uint8_t> status_buf(NP);

    // Unwrapped positions buffer (device)
    DeviceBuffer<real> ux, uy, uz;
    const bool need_unwrap = any_periodic || snap.include_unwrapped;
    if (need_unwrap) {
        ux.resize(NP); uy.resize(NP); uz.resize(NP);
    }

    std::printf("       Fields: K=%zu  head=%zu  vel=3×%zu  particles=%d\n",
                N, N, vel.field_size(), NP);

    // ── Transport adapter config ─────────────────────────────────────
    using namespace macroflow3d::physics::particles;

    TransportAdapterConfig ta_cfg;
    ta_cfg.molecular_diffusion  = cfg.transport.diffusion;
    ta_cfg.alpha_l              = cfg.transport.alpha_l;
    ta_cfg.alpha_t              = cfg.transport.alpha_t;
    ta_cfg.linear_interpolation = true;
    ta_cfg.rng_seed             = cfg.transport.seed;

    // ── Stats collector (Etapa 6 — par2-neutral wrapper) ─────────────
    const bool do_stats = mac.enabled;
    const bool use_biased = (mac.var_estimator == "biased");
    runtime::ParticleMomentsCollector collector(NP, grid, any_periodic, use_biased);

    // Per-realization time-series for post-processing
    std::vector<std::vector<io::TimeSeriesPoint<real>>> all_series;
    if (do_stats) all_series.reserve(NR);

    // ── I/O Scheduler (Etapa 5 — single decision point) ──────────────
    const int n_steps = cfg.transport.n_steps;
    const int r0_snap_every = (n_steps > 100) ? (n_steps / 100) : 1;

    runtime::SchedulerConfig sched_cfg;
    sched_cfg.stats_every            = do_stats ? mac.sample_every : 0;
    sched_cfg.snapshot_every         = snap.enabled ? snap.every : 0;
    sched_cfg.r0_snapshot_every      = r0_snap_every;
    sched_cfg.n_particles            = NP;
    sched_cfg.n_steps                = n_steps;
    sched_cfg.has_periodic           = any_periodic;
    // Mirror all Par2-compatible snapshot options from SnapshotConfig (Task 1)
    sched_cfg.snap_writer.legacy_format       = snap.legacy_format;
    sched_cfg.snap_writer.include_time        = snap.include_time;
    sched_cfg.snap_writer.include_status      = snap.include_status;
    sched_cfg.snap_writer.include_wrap_counts = snap.include_wrap_counts;
    sched_cfg.snap_writer.include_unwrapped   = snap.include_unwrapped;
    sched_cfg.snap_writer.stride              = snap.stride;
    sched_cfg.snap_writer.max_particles       = snap.max_particles;
    sched_cfg.snap_writer.precision           = snap.precision;

    runtime::IOScheduler scheduler(sched_cfg, layout);

    // ── PSPTA objects — hoisted for reuse across realizations (Task 4) ─────
    // PsptaPsiField and PsptaEngine are constructed once here and rebind each
    // realization; only precompute_levelA() and bind*() are called per-r.
    // This eliminates per-realization cudaMalloc/cudaFree of ψ buffers and
    // engine workspace, keeping the GPU memory footprint stable.
    std::unique_ptr<pspta::PsptaEngine> pspta_eng;
    pspta::PsptaPsiField pspta_psi_field;
    if (cfg.transport.method == "pspta") {
        pspta_psi_field.resize(grid);
        pspta_eng = std::make_unique<pspta::PsptaEngine>(
            grid, ctx.cuda_stream(), cfg.transport.seed);
    }

    // ================================================================
    // REALIZATION LOOP (NR loop lives ONLY here — Etapa 10)
    // ================================================================
    for (int r = 0; r < NR; ++r) {
        if (NR > 1) {
            pipeline::print_separator();
            std::printf("  Realization %d / %d\n", r + 1, NR);
            pipeline::print_separator();
        }

        // ── Generate K (per-realization seed) ──────────────────────────
        StochasticConfig r_stoch = stoch_cfg;
        r_stoch.seed = stoch_cfg.seed + static_cast<uint64_t>(r);

        std::printf("  [4] K generation (seed=%lu)\n", (unsigned long)r_stoch.seed);
        MACROFLOW3D_NVTX_PUSH("K_generation");
        profiler.start("K_generation");
        generate_K_field(K_field.span(), stoch_ws, grid, r_stoch, ctx);
        profiler.stop();

        if (MACROFLOW3D_DIAGNOSTICS_ENABLED) {
            real kmin, kmax, kmean;
            compute_field_stats(K_field.span(), kmin, kmax, kmean, ctx);
            std::printf("       K stats: min=%.4e  max=%.4e  mean=%.4e\n", kmin, kmax, kmean);
        }

        // ── Solve head ─────────────────────────────────────────────────
        std::printf("  [5] Solving head (%s)\n", cfg.flow.solver.c_str());
        MACROFLOW3D_NVTX_PUSH("solve_head");
        profiler.start("solve_head");
        init_head_guess(head_field.span(), grid, cfg.flow.bc, ctx);
        const ScalarField& K_const = K_field;
        HeadSolveResult result = solve_head(
            head_field.span(), K_const.span(),
            grid, cfg.flow.bc, head_cfg, ctx, flow_ws);
        profiler.stop();
        std::printf("       Converged=%s  iters=%d  res=%.2e → %.2e\n",
                    result.converged ? "YES" : "NO", result.num_iterations,
                    result.initial_residual, result.final_residual);
        if (!result.converged)
            std::fprintf(stderr, "       WARNING: Head solve did NOT converge (r=%d)!\n", r);

        // ── Compute velocity ───────────────────────────────────────────
        MACROFLOW3D_NVTX_PUSH("velocity");
        profiler.start("velocity");
        if (cfg.transport.method == "pspta") {
            std::printf("  [6] Computing velocity (compact)\n");
            compute_velocity_from_head(vel_compact, head_field, K_field, grid, cfg.flow.bc, ctx);
            // Compute padded only if needed for diagnostics
            if (cfg.diagnostics.velocity_field) {
                compute_velocity_from_head(vel, head_field, K_field, grid, cfg.flow.bc, ctx);
            }
        } else {
            std::printf("  [6] Computing velocity (padded)\n");
            compute_velocity_from_head(vel, head_field, K_field, grid, cfg.flow.bc, ctx);
        }
        profiler.stop();

        // ── Velocity diagnostics (optional, once per realization) ──────
        if (cfg.diagnostics.velocity_field) {
            compute_velocity_diagnostics(vel, vel_diag, grid, ctx);
            print_velocity_diagnostics(vel_diag, r, ctx);
        }

        // ── Transport ─────────────────────────────────────────────────
        const real dt          = cfg.transport.dt;
        const int sample_every = mac.sample_every;
        const real Ly = cfg.grid.Ly();
        const real Lz = cfg.grid.Lz();

        // Common particle view (same for both engines)
        ParticlesSoA<real> pv;
        pv.x = px.data(); pv.y = py.data(); pv.z = pz.data();
        pv.n = NP;
        pv.status = status_buf.data();
        if (any_periodic) {
            pv.wrapX = wrapX.data();
            pv.wrapY = wrapY.data();
            pv.wrapZ = wrapZ.data();
            cudaMemsetAsync(wrapX.data(), 0, NP * sizeof(int32_t), ctx.cuda_stream());
            cudaMemsetAsync(wrapY.data(), 0, NP * sizeof(int32_t), ctx.cuda_stream());
            cudaMemsetAsync(wrapZ.data(), 0, NP * sizeof(int32_t), ctx.cuda_stream());
        }
        cudaMemsetAsync(status_buf.data(), 0, NP * sizeof(uint8_t), ctx.cuda_stream());

        // Common unwrapped view
        UnwrappedSoA<real> unwrap_view;
        if (need_unwrap) {
            unwrap_view.x_u      = ux.data();
            unwrap_view.y_u      = uy.data();
            unwrap_view.z_u      = uz.data();
            unwrap_view.capacity = NP;
        }

        // Generic hot loop — templated on engine type; same interface for Par2 and PSPTA.
        // Engine must expose: bind_particles, inject_box, ensure_tracking, prepare,
        //                     step, particles(), compute_unwrapped, synchronize.
        auto run_hot_loop = [&](auto& eng) {
            eng.bind_particles(pv);
            eng.inject_box(cfg.transport.inject_x, static_cast<real>(0.0),
                           static_cast<real>(0.0),
                           cfg.transport.inject_x, Ly, Lz, 0, NP);
            eng.ensure_tracking();
            eng.prepare();

            if (do_stats) collector.reset();
            scheduler.begin_realization(r);

            MACROFLOW3D_NVTX_PUSH("transport");
            profiler.start("transport");
            std::printf("       Stepping %d × %d (dt=%.4e) ...\n", NP, n_steps, dt);

            for (int step = 1; step <= n_steps; ++step) {
                eng.step(dt);
                counters.add_step();

                // Progress printout (for both Par2 and PSPTA)
                const int out_every = cfg.transport.output_every;
                if (out_every > 0 && step % out_every == 0) {
                    std::printf("           step %5d/%d  t=%.3e\n",
                                step, n_steps, static_cast<double>(step) * dt);
                }
                const bool sample_now = do_stats && (step % sample_every == 0);
                const bool snap_now   = scheduler.snapshot_due(step);
                const bool r0_snap    = scheduler.r0_snapshot_due(step);
                const bool any_event  = sample_now || snap_now || r0_snap;

                if (any_event) {
                    ConstParticlesSoA<real> cpv = eng.particles();

                    // [Task 2] Unwrapped only for regular snapshots that request it
                    if (snap_now && sched_cfg.snap_writer.include_unwrapped &&
                        unwrap_view.valid())
                    {
                        eng.compute_unwrapped(unwrap_view, ctx.cuda_stream());
                    }

                    // [Task 3] Stage D2H before the single sync
                    if (snap_now || r0_snap) {
                        scheduler.stage_snapshot_async(step, cpv, unwrap_view,
                                                       ctx.cuda_stream());
                    }
                    if (sample_now) {
                        collector.sample_async(cpv, ctx.cuda_stream());
                    }

                    // Single sync — covers snapshot D2H and stats reduction
                    cudaStreamSynchronize(ctx.cuda_stream());

                    io::TimeSeriesPoint<real> ts_point;
                    bool have_stats = false;
                    if (sample_now) {
                        have_stats = collector.store_sample(step, dt, ts_point);
                        counters.add_stats();
                    }

                    // Data already synced: scheduler writes without re-staging/re-syncing
                    scheduler.on_step(step, dt, cpv, unwrap_view,
                                      have_stats ? &ts_point : nullptr,
                                      ctx.cuda_stream(), /*pre_synced=*/true);
                }
            }

            eng.synchronize();

            // [Task 5] Write final snapshot if last step wasn't covered by cadence
            {
                ConstParticlesSoA<real> cpv_fin = eng.particles();
                if (sched_cfg.snap_writer.include_unwrapped && unwrap_view.valid() &&
                    sched_cfg.snapshot_every > 0 &&
                    n_steps % sched_cfg.snapshot_every != 0)
                {
                    eng.compute_unwrapped(unwrap_view, ctx.cuda_stream());
                }
                scheduler.maybe_write_final(dt, cpv_fin, unwrap_view,
                                            ctx.cuda_stream());
            }

            profiler.stop();

            std::printf("       Transport complete (r=%d).\n", r);
            scheduler.end_realization();
            counters.add_realization();
        };  // end run_hot_loop

        if (cfg.transport.method == "pspta") {
            std::printf("  [7] Transport (PSPTA)\n");
            // Per-realization: precompute ψ fields for this velocity field
            // (overwrites existing buffers — no cudaMalloc/cudaFree per Task 4)
            pspta::PsptaPrecomputeReport psi_rep =
                pspta_psi_field.precompute_levelA(vel_compact, grid, ctx.cuda_stream());
            if (psi_rep.n_vx_clamped > 0) {
                std::printf("       [psi] vx_clamped=%lld / %lld\n",
                            (long long)psi_rep.n_vx_clamped,
                            (long long)psi_rep.n_total);
            }
            if (cfg.transport.pspta_diagnostics) {
                auto qual_rep = pspta_psi_field.compute_psi_quality(
                    vel_compact, grid, ctx.cuda_stream());
                std::printf("       [psi_quality] rms_r1=%.3e max_r1=%.3e "
                            "rms_r2=%.3e max_r2=%.3e  ncells=%lld\n",
                            qual_rep.rms_r1, qual_rep.max_r1,
                            qual_rep.rms_r2, qual_rep.max_r2,
                            (long long)qual_rep.n_cells);
                runtime::CsvDiagnosticsWriter::write_psi_quality_row(
                    layout.base + "/psi_quality.csv", r, psi_rep, qual_rep);
            }
            // Rebind per-realization (engine workspace already allocated)
            pspta_eng->set_inject_seed(
                cfg.transport.seed + static_cast<uint64_t>(r) * 100ULL);
            pspta_eng->bind_velocity(&vel_compact);
            pspta_eng->bind_psifield(&pspta_psi_field);
            run_hot_loop(*pspta_eng);
            // Post-transport PSPTA diagnostics
            auto ts = pspta_eng->compute_transport_stats();
            std::printf("       [pspta] active=%d  exited=%d  newton_stalls=%lld  "
                        "nonzero_fail=%u  max_fail=%u  (NP=%d)\n",
                        ts.n_active, ts.n_exited, (long long)ts.total_fail,
                        ts.n_nonzero_fail, ts.max_fail_count, NP);
            if (cfg.transport.pspta_diagnostics) {
                runtime::CsvDiagnosticsWriter::write_newton_fail_row(
                    layout.base + "/newton_fail_summary.csv", r, NP, ts);
            }
        } else {
            std::printf("  [7] Transport (Par2_Core)\n");
            TransportAdapterConfig r_cfg = ta_cfg;
            r_cfg.rng_seed = cfg.transport.seed + static_cast<uint64_t>(r) * 1000ULL;
            Par2TransportAdapter eng(grid, cfg.flow.bc, r_cfg, ctx.cuda_stream());
            eng.bind_velocity(vel);
            run_hot_loop(eng);
        }

        // Store this realization's series for post-processing
        if (do_stats) {
            all_series.push_back(scheduler.stats_series());
        }
    }
    // ── END REALIZATION LOOP ───────────────────────────────────────────

    // ── Post-processing: macrodispersion (Etapa 6 — analysis) ──────────
    if (do_stats && NR > 0) {
        pipeline::print_separator();
        std::printf("  Post-processing: macrodispersion (NR=%d)\n", NR);

        // Ensemble mean time-series
        std::string mean_path = layout.ensemble_timeseries();
        io::CsvTimeSeriesWriter::write_ensemble_mean(mean_path, all_series);
        std::printf("       Wrote %s\n", mean_path.c_str());

        // Macrodispersivity α(t)
        auto alpha_rows = analysis::compute_macrodispersion(
            all_series, mac.lambda, mac.vmean_norm);

        std::string alpha_path = layout.macrodispersion_csv();
        analysis::write_macrodispersion_csv(alpha_path, alpha_rows);
        std::printf("       Wrote %s\n", alpha_path.c_str());
    }

    // ── Summary ────────────────────────────────────────────────────────
    pipeline::print_separator();
    profiler.report();
    counters.report();
    pipeline::print_separator();
    std::printf("Ensemble finished successfully.\n");

    return EXIT_SUCCESS;
}

} // namespace ensemble
} // namespace macroflow3d
