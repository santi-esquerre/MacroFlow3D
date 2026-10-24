/**
 * @file par2_field_stats_main.cu
 * @brief SF-38 D17 tool: realized ln K statistics of the Par2 baseline fields.
 *
 * Usage: par2_field_stats <config.yaml> [--nr N]
 *
 * Loads a pipeline config with the existing loader, builds the same Grid3D and
 * StochasticConfig as `ensemble::run_ensemble`, and for r = 0..NR-1 regenerates
 * the realization field exactly as EnsembleRunner's default (Par2) branch does:
 * `r_stoch.seed = stoch_cfg.seed + r; generate_K_field(K, ws, grid, r_stoch, ctx)`.
 * Nothing in the Par2 path writes K to disk, hence this tool (decision
 * 2026-10-23 D17; spec SF-38 "Par2 RWPT baseline"; C-PAR2).
 *
 * One line per realization:
 *   FIELDSTAT r= seed= n= mean_lnK= var_lnK= Kmin= Kmax= Kmean=
 * mean/var of ln K are population statistics (divide by n) of the Gaussian
 * field `logK` (the field before the K_g shift; ln K = logK when K_mean = 1) by a
 * deterministic index-order double reduction on the host (two pass: mean, then
 * sum of squared deviations). Kmin/Kmax/Kmean come from `compute_field_stats`,
 * the function behind the pipeline's `K stats` line (diagnostics builds).
 *
 * Device memory: K (8 B/cell) + stochastic workspace (logK 8 B/cell + O(n_modes)
 * RNG/mode buffers) = 16 B/cell, ~2.7 GB at 2496x288x288 (no head, velocity,
 * flow workspace or particles). Host: one downloaded logK copy (8 B/cell,
 * ~1.7 GB at 2496x288x288).
 */

#include "src/io/config/Config.hpp"
#include "src/io/config/ConfigValidator.hpp"
#include "src/physics/common/fields.cuh"
#include "src/physics/common/physics_config.hpp"
#include "src/physics/common/workspaces.cuh"
#include "src/physics/stochastic/stochastic.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

using namespace macroflow3d;
using namespace macroflow3d::io;
using namespace macroflow3d::physics;

namespace {

// Same mapping as EnsembleRunner.cu (static there, so repeated here).
StochasticConfig make_stochastic_cfg(const StochasticYamlConfig& y) {
    StochasticConfig c;
    c.sigma2 = y.sigma2;
    c.corr_length = y.corr_length;
    c.n_modes = y.n_modes;
    c.covariance_type = y.covariance_type;
    c.seed = y.seed;
    c.K_geometric_mean = y.K_mean;
    return c;
}

} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::fprintf(stderr, "Usage: %s <config.yaml> [--nr N]\n", argv[0]);
            return 2;
        }
        const std::string path = argv[1];
        int nr_override = -1;
        for (int i = 2; i < argc; ++i) {
            if (std::string(argv[i]) == "--nr" && i + 1 < argc) {
                nr_override = std::atoi(argv[++i]);
                if (nr_override < 1) {
                    std::fprintf(stderr, "--nr must be >= 1\n");
                    return 2;
                }
            } else {
                std::fprintf(stderr, "Usage: %s <config.yaml> [--nr N]\n", argv[0]);
                return 2;
            }
        }

        AppConfig cfg = load_config_yaml(path);
        require_valid_config(cfg); // same gate as run_ensemble
        if (cfg.run_mode == RunMode::AnalysisOnly) {
            std::fprintf(stderr, "par2_field_stats: run_mode analysis_only has no fields\n");
            return 2;
        }
        if (cfg.streamfunction_solver.enabled) {
            // Only the default Par2 branch of EnsembleRunner is reproduced.
            std::fprintf(stderr, "par2_field_stats: streamfunction_solver.enabled is not supported "
                                 "(only the default stochastic field branch is reproduced)\n");
            return 2;
        }

        const auto& mac = cfg.analysis.macrodispersion;
        const int NR = nr_override > 0 ? nr_override : (mac.enabled ? mac.NR : 1);

        CudaContext ctx(0);
        const Grid3D grid(cfg.grid.nx, cfg.grid.ny, cfg.grid.nz, cfg.grid.dx, cfg.grid.dx,
                          cfg.grid.dx);
        const size_t N = grid.num_cells();
        const StochasticConfig stoch_cfg = make_stochastic_cfg(cfg.stochastic);

        ScalarField K_field(grid);
        StochasticWorkspace stoch_ws;
        stoch_ws.allocate(grid, stoch_cfg);

        std::vector<real> h_logK(N);
        for (int r = 0; r < NR; ++r) {
            StochasticConfig r_stoch = stoch_cfg;
            r_stoch.seed = stoch_cfg.seed + static_cast<uint64_t>(r);
            generate_K_field(K_field.span(), stoch_ws, grid, r_stoch, ctx);

            real kmin, kmax, kmean;
            compute_field_stats(K_field.span(), kmin, kmax, kmean, ctx);

            MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(h_logK.data(), stoch_ws.logK.data(),
                                                   N * sizeof(real), cudaMemcpyDeviceToHost,
                                                   ctx.cuda_stream()));
            ctx.synchronize();

            double s = 0.0;
            for (size_t i = 0; i < N; ++i) s += h_logK[i];
            const double mean = s / static_cast<double>(N);
            double q = 0.0;
            for (size_t i = 0; i < N; ++i) {
                const double d = h_logK[i] - mean;
                q += d * d;
            }
            const double var = q / static_cast<double>(N);

            std::printf("FIELDSTAT r=%d seed=%lu n=%zu mean_lnK=%.12e var_lnK=%.12e "
                        "Kmin=%.4e Kmax=%.4e Kmean=%.4e\n",
                        r, (unsigned long)r_stoch.seed, N, mean, var, (double)kmin, (double)kmax,
                        (double)kmean);
            std::fflush(stdout);
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "par2_field_stats: %s\n", e.what());
        return 1;
    }
}
