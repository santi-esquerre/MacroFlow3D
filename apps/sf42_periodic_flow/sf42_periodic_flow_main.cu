/**
 * @file sf42_periodic_flow_main.cu
 * @brief SF-42 N3: finite-volume comparator of the pseudo-spectral cross-check. Solves the SF-19
 *        affine-periodic Darcy problem (qbar = (1,0,0)) on the SF-18 field and dumps Y, (K,) and
 *        the three CompactMAC face-flux arrays at full precision.
 *
 * Documented-experiment tool (NOT a ctest entry, no production change). It calls the production
 * public API only:
 *   - `physics::generate_periodic_gaussian_field` (SF-18) on the unit box (`h = 1/N`),
 *     `normalize_variance = true`, `corr_length = ell`, `sigma2` (default 1: Y has variance 1);
 *   - host `lnK = eps * Y; K = std::exp(lnK)` exactly as `build_long_box_stage` phase S1 does;
 *   - `physics::solve_affine_periodic_flow` (SF-19; the lean variant is bitwise identical) with
 *     the linear tolerance of the study (default 1e-12, tighter than the production 1e-10 so that
 *     the algebraic error is far below the discretization error; recorded in the report).
 *
 * Output directory (all float64, little-endian, C order; "(a,b,c)" = numpy shape):
 *   Y.npy  (nz,ny,nx)    cell-centred SF-18 field, element [k,j,i] = Y(i,j,k), flat index i+nx(j+ny k)
 *   K.npy  (nz,ny,nx)    only with --dump-k
 *   U.npy  (nz,ny,nx+1)  x-face fluxes F = K_f (G.n + dh~/dn); plane i = nx duplicates plane 0
 *   V.npy  (nz,ny+1,nx)  y-faces; plane j = ny duplicates plane 0
 *   W.npy  (nz+1,ny,nx)  z-faces; plane k = nz duplicates plane 0
 *   report.json          K_eff, G, achieved mean flux, divergence, per-corrector PCG results,
 *                        SF-18 report (raw variance, applied scale, ...), tolerances, timings
 * Exit codes: 0 done and all three correctors converged; 3 files written but a corrector did not
 * converge; 2 usage error; 1 exception.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/external/nlohmann/json.hpp"
#include "src/physics/flow/AffinePeriodicFlowSolver.cuh"
#include "src/physics/stochastic/PeriodicGaussianField.cuh"
#include "src/physics/streamfunctions/inlet_slab/NpyIo.hpp"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace macroflow3d;
using json = nlohmann::json;
namespace npy = macroflow3d::streamfunctions::inlet_slab;

struct UsageError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Options {
    int n = 0;
    double eps = -1.0;
    double sigma2 = 1.0;
    double ell = 0.25;
    unsigned long long seed = 0ULL;
    double rtol = 1e-12;
    int max_iter = 4000;
    bool dump_k = false;
    std::string out;
};

void print_usage() {
    std::fprintf(stderr,
                 "usage: sf42_periodic_flow --n N --eps E --seed S --out DIR [--sigma2 V] [--ell L] "
                 "[--rtol R] [--max-iter M] [--dump-k]\n"
                 "  --n N        cubic grid N^3 on the unit periodic box (h = 1/N); N even\n"
                 "  --eps E      ln K = E * Y (E = sqrt(target ln K variance) when --sigma2 1)\n"
                 "  --sigma2 V   SF-18 variance of Y before ln K = E Y (default 1)\n"
                 "  --ell L      SF-18 correlation length in box units (default 0.25: box = 4 lambda)\n"
                 "  --rtol R     PCG relative tolerance of the three correctors (default 1e-12)\n"
                 "  --max-iter M PCG iteration cap (default 4000)\n"
                 "  --dump-k     also write K.npy\n");
}

Options parse(int argc, char** argv) {
    Options o;
    bool have_n = false, have_eps = false, have_seed = false;
    for (int a = 1; a < argc; ++a) {
        const std::string k = argv[a];
        auto val = [&]() -> std::string {
            if (a + 1 >= argc)
                throw UsageError("missing value for " + k);
            return argv[++a];
        };
        if (k == "--n") {
            o.n = std::atoi(val().c_str());
            have_n = true;
        } else if (k == "--eps") {
            o.eps = std::atof(val().c_str());
            have_eps = true;
        } else if (k == "--sigma2") {
            o.sigma2 = std::atof(val().c_str());
        } else if (k == "--ell") {
            o.ell = std::atof(val().c_str());
        } else if (k == "--seed") {
            o.seed = std::strtoull(val().c_str(), nullptr, 10);
            have_seed = true;
        } else if (k == "--rtol") {
            o.rtol = std::atof(val().c_str());
        } else if (k == "--max-iter") {
            o.max_iter = std::atoi(val().c_str());
        } else if (k == "--out") {
            o.out = val();
        } else if (k == "--dump-k") {
            o.dump_k = true;
        } else {
            throw UsageError("unknown option " + k);
        }
    }
    if (!have_n || !have_eps || !have_seed || o.out.empty())
        throw UsageError("--n, --eps, --seed and --out are required");
    if (o.n < 8 || o.n % 2 != 0)
        throw UsageError("--n must be even and >= 8");
    if (!(o.eps >= 0.0) || !std::isfinite(o.eps))
        throw UsageError("--eps must be finite and >= 0");
    if (!(o.sigma2 > 0.0) || !std::isfinite(o.sigma2))
        throw UsageError("--sigma2 must be finite and > 0");
    if (!(o.ell > 0.0) || !std::isfinite(o.ell))
        throw UsageError("--ell must be finite and > 0");
    if (!(o.rtol > 0.0) || !std::isfinite(o.rtol))
        throw UsageError("--rtol must be finite and > 0");
    if (o.max_iter < 1)
        throw UsageError("--max-iter must be >= 1");
    return o;
}

// Same rule as streamfunction_ev_ladder / box_auto_mg_levels on a cube.
int auto_mg_levels(int n) {
    int levels = 1;
    int m = n;
    while (m % 2 == 0 && (m / 2) >= 4 && ((m / 2) % 2) == 0) {
        m /= 2;
        ++levels;
    }
    return levels;
}

double since(const std::chrono::steady_clock::time_point& t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::vector<real> download(const DeviceBuffer<real>& b, std::size_t n) {
    std::vector<real> h(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), b.data(), n * sizeof(real), cudaMemcpyDeviceToHost));
    return h;
}

json corrector_json(const solvers::ProjectedPCGResult& r) {
    return {{"iterations", r.iterations},
            {"converged", r.converged},
            {"status", static_cast<int>(r.status)},
            {"relative_projected_residual", static_cast<double>(r.relative_projected_residual)},
            {"initial_projected_residual", static_cast<double>(r.initial_projected_residual)},
            {"final_projected_residual", static_cast<double>(r.final_projected_residual)}};
}

int run(const Options& o) {
    const auto t_all = std::chrono::steady_clock::now();
    const int N = o.n;
    const real h = real{1} / static_cast<real>(N);
    const Grid3D grid(N, N, N, h, h, h);
    const std::size_t n = grid.num_cells();
    const int mg_levels = auto_mg_levels(N);
    std::filesystem::create_directories(o.out);

    CudaContext ctx(0);

    // ---- SF-18 field ----
    physics::PeriodicGaussianFieldConfig gcfg;
    gcfg.sigma2 = static_cast<real>(o.sigma2);
    gcfg.corr_length = static_cast<real>(o.ell);
    gcfg.seed = o.seed;
    gcfg.normalize_variance = true;
    std::vector<real> Y;
    physics::PeriodicGaussianFieldReport grep;
    {
        DeviceBuffer<real> y_dev(n);
        physics::PeriodicGaussianFieldWorkspace gws;
        grep = physics::generate_periodic_gaussian_field(ctx, grid, gcfg, y_dev.span(), gws);
        ctx.synchronize();
        Y = download(y_dev, n);
    }
    double ymean = 0.0, yvar = 0.0;
    for (std::size_t c = 0; c < n; ++c)
        ymean += Y[c];
    ymean /= static_cast<double>(n);
    for (std::size_t c = 0; c < n; ++c)
        yvar += (Y[c] - ymean) * (Y[c] - ymean);
    yvar /= static_cast<double>(n);

    // ---- K exactly as build_long_box_stage phase S1: lnK = eps Y; K = exp(lnK) on the host ----
    std::vector<real> K(n);
    for (std::size_t c = 0; c < n; ++c) {
        const real lnK = static_cast<real>(o.eps) * Y[c];
        K[c] = std::exp(lnK);
    }

    // ---- SF-19 flow ----
    physics::AffinePeriodicFlowConfig fcfg{}; // qbar = (1, 0, 0)
    fcfg.linear.rtol = static_cast<real>(o.rtol);
    fcfg.linear.max_iter = o.max_iter;
    fcfg.mg.num_levels = mg_levels;
    const std::size_t nu = static_cast<std::size_t>(N + 1) * N * N;
    const std::size_t nv = nu, nw = nu; // cubic: all three have (N+1) N^2 elements
    DeviceBuffer<real> k_dev(n), u_dev(nu), v_dev(nv), w_dev(nw);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(k_dev.data(), K.data(), n * sizeof(real), cudaMemcpyHostToDevice));
    physics::AffinePeriodicFlowWorkspace flow_ws;
    const auto t_flow = std::chrono::steady_clock::now();
    const physics::AffinePeriodicFlowReport fr = physics::solve_affine_periodic_flow(
        ctx, grid, DeviceSpan<const real>(k_dev.span()), fcfg,
        physics::AffinePeriodicVelocityView{u_dev.span(), v_dev.span(), w_dev.span()}, flow_ws);
    ctx.synchronize();
    const double flow_s = since(t_flow);
    const bool all_converged = fr.corrector_results[0].converged && fr.corrector_results[1].converged &&
                               fr.corrector_results[2].converged;

    // ---- dumps (float64, C order, documented shapes) ----
    const std::size_t Nz = static_cast<std::size_t>(N);
    npy::write_npy(o.out + "/Y.npy", {Nz, Nz, Nz}, Y.data(), n);
    if (o.dump_k)
        npy::write_npy(o.out + "/K.npy", {Nz, Nz, Nz}, K.data(), n);
    {
        const std::vector<real> U = download(u_dev, nu), V = download(v_dev, nv), W = download(w_dev, nw);
        npy::write_npy(o.out + "/U.npy", {Nz, Nz, Nz + 1}, U.data(), nu);
        npy::write_npy(o.out + "/V.npy", {Nz, Nz + 1, Nz}, V.data(), nv);
        npy::write_npy(o.out + "/W.npy", {Nz + 1, Nz, Nz}, W.data(), nw);
    }

    json rep;
    rep["tool"] = "sf42_periodic_flow (SF-42 N3)";
    rep["n"] = N;
    rep["h"] = static_cast<double>(h);
    rep["ell"] = o.ell;
    rep["sigma2_gen"] = o.sigma2;
    rep["eps"] = o.eps;
    rep["seed"] = o.seed;
    rep["pcg_rtol"] = o.rtol;
    rep["pcg_max_iter"] = o.max_iter;
    rep["mg_levels"] = mg_levels;
    rep["sf18"] = {{"raw_mean", static_cast<double>(grep.raw_mean)},
                   {"raw_variance", static_cast<double>(grep.raw_variance)},
                   {"applied_scale", static_cast<double>(grep.applied_scale)},
                   {"final_variance", static_cast<double>(grep.final_variance)},
                   {"active_mode_count", grep.active_mode_count}};
    rep["Y_mean_host"] = ymean;
    rep["Y_variance_host"] = yvar;
    json keff = json::array();
    for (int i = 0; i < 3; ++i)
        keff.push_back({static_cast<double>(fr.K_eff[i][0]), static_cast<double>(fr.K_eff[i][1]),
                        static_cast<double>(fr.K_eff[i][2])});
    rep["K_eff"] = keff; // K_eff[i][d]: component i of the mean flux for corrector direction d
    rep["symmetry_defect_rel"] = static_cast<double>(fr.symmetry_defect_rel);
    rep["G"] = {static_cast<double>(fr.G[0]), static_cast<double>(fr.G[1]), static_cast<double>(fr.G[2])};
    rep["achieved_mean_flux"] = {static_cast<double>(fr.achieved_mean_flux[0]),
                                 static_cast<double>(fr.achieved_mean_flux[1]),
                                 static_cast<double>(fr.achieved_mean_flux[2])};
    rep["div_max_abs"] = static_cast<double>(fr.div_max_abs);
    rep["div_rms"] = static_cast<double>(fr.div_rms);
    rep["correctors"] = {corrector_json(fr.corrector_results[0]), corrector_json(fr.corrector_results[1]),
                         corrector_json(fr.corrector_results[2])};
    rep["all_correctors_converged"] = all_converged;
    rep["seconds"] = {{"flow", flow_s}, {"total", since(t_all)}};
    rep["layout"] = {{"Y", "(nz,ny,nx)"}, {"U", "(nz,ny,nx+1)"}, {"V", "(nz,ny+1,nx)"}, {"W", "(nz+1,ny,nx)"},
                     {"flux", "F = K_f (G.n + d h~/dn), K_f harmonic; periodic duplicate plane = plane 0"}};
    std::ofstream os(o.out + "/report.json", std::ios::trunc);
    os << rep.dump(2) << '\n';
    os.close();
    if (!os)
        throw std::runtime_error("failed writing report.json in " + o.out);

    std::printf("sf42_periodic_flow N=%d eps=%g seed=%llu rtol=%g: iterations=(%d,%d,%d) converged=%d "
                "G=(%.12g,%.12g,%.12g) flow=%.2fs\n",
                N, o.eps, o.seed, o.rtol, fr.corrector_results[0].iterations,
                fr.corrector_results[1].iterations, fr.corrector_results[2].iterations,
                all_converged ? 1 : 0, static_cast<double>(fr.G[0]), static_cast<double>(fr.G[1]),
                static_cast<double>(fr.G[2]), flow_s);
    return all_converged ? 0 : 3;
}

} // namespace

int main(int argc, char** argv) {
    Options o;
    try {
        o = parse(argc, argv);
    } catch (const UsageError& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        print_usage();
        return 2;
    }
    try {
        return run(o);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "sf42_periodic_flow: exception: %s\n", e.what());
        return 1;
    }
}
