/**
 * @file affine_periodic_flow_tests.cu
 * @brief SF-19 T02: acceptance tests for `solve_affine_periodic_flow`
 *        (`AffinePeriodicFlowSolver.cuh`).
 *
 * Standalone ctest-friendly runner (mirrors
 * `tests/stochastic/periodic_gaussian_tests.cu`'s single-executable,
 * printed-checks style) rather than the heavier `tests/streamfunctions/*`
 * case-registry framework: `AffinePeriodicFlowSolver` is not part of the
 * streamfunctions namespace/registry and this is the least-new-infrastructure
 * option for a single new solver entry point.
 *
 * All acceptance tolerances, fixtures, and seeds below were PRESPECIFIED in
 * the SF-19 activation bitácora (decision 6) BEFORE this file existed and are
 * implemented VERBATIM. No tolerance, fixture, or seed here may be adjusted
 * to make a failing case pass; comparison-METHOD fixes are allowed only if
 * the contract is unchanged, and must be reported.
 *
 * Corrector solves in ALL cases use config.linear.rtol = 1e-11,
 * config.linear.max_iter = 2000, and MG defaults (multigrid::MGConfig{}: 4
 * levels, 2 pre/post smooths, 50 coarse iterations) as the reused MG
 * stack/library defines them.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/numerics/blas/fill.cuh"
#include "src/physics/flow/AffinePeriodicFlowSolver.cuh"
#include "src/physics/stochastic/PeriodicGaussianField.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>
#include <string>
#include <vector>

using namespace macroflow3d;
using namespace macroflow3d::physics;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

// ============================================================================
// Bookkeeping (mirrors tests/stochastic/periodic_gaussian_tests.cu)
// ============================================================================

struct TestReport {
    bool overall_pass = true;
    int checks = 0;

    void check(bool cond, const std::string& name, const std::string& detail = "") {
        ++checks;
        std::printf("[%s] %s%s%s\n", cond ? "PASS" : "FAIL", name.c_str(),
                    detail.empty() ? "" : "  ", detail.c_str());
        overall_pass = overall_pass && cond;
    }
};

std::vector<real> download(const DeviceBuffer<real>& buf, size_t n) {
    std::vector<real> host(n);
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(host.data(), buf.data(), n * sizeof(real), cudaMemcpyDeviceToHost));
    return host;
}

std::size_t u_size(const Grid3D& g) {
    return static_cast<std::size_t>(g.nx + 1) * static_cast<std::size_t>(g.ny) *
           static_cast<std::size_t>(g.nz);
}
std::size_t v_size(const Grid3D& g) {
    return static_cast<std::size_t>(g.nx) * static_cast<std::size_t>(g.ny + 1) *
           static_cast<std::size_t>(g.nz);
}
std::size_t w_size(const Grid3D& g) {
    return static_cast<std::size_t>(g.nx) * static_cast<std::size_t>(g.ny) *
           static_cast<std::size_t>(g.nz + 1);
}

// Trivial elementwise exp kernel (device->device), used to build K = exp(Y)
// from a device-resident log-conductivity field; documented in the file
// header as an explicitly allowed small kernel local to this test file.
__global__ void exp_kernel(const real* __restrict__ in, real* __restrict__ out, std::size_t n) {
    const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = exp(in[idx]);
    }
}

void device_exp(CudaContext& ctx, DeviceSpan<const real> in, DeviceSpan<real> out) {
    const std::size_t n = in.size();
    constexpr int kBlock = 256;
    const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
    exp_kernel<<<blocks, kBlock, 0, ctx.cuda_stream()>>>(in.data(), out.data(), n);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    ctx.synchronize();
}

AffinePeriodicFlowConfig default_config() {
    AffinePeriodicFlowConfig cfg;
    cfg.qbar[0] = real{1};
    cfg.qbar[1] = real{0};
    cfg.qbar[2] = real{0};
    cfg.linear.rtol = real{1e-11};
    cfg.linear.max_iter = 2000;
    cfg.linear.check_every = 10; // ProjectedPCGConfig default
    cfg.mg = multigrid::MGConfig{}; // library/reused-stack defaults
    return cfg;
}

// ============================================================================
// Case: config/validation contract (distinct std::invalid_argument messages)
// ============================================================================

bool case_config_validation(TestReport& rep) {
    std::set<std::string> messages;

    auto expect_invalid = [&](const std::string& name, auto&& thrower) {
        try {
            thrower();
            rep.check(false, name, "no exception thrown");
        } catch (const std::invalid_argument& e) {
            const std::string msg = e.what();
            const bool distinct = messages.insert(msg).second;
            rep.check(distinct, name, "msg=\"" + msg + "\"");
        } catch (const std::exception& e) {
            rep.check(false, name, std::string("wrong exception type: ") + e.what());
        }
    };

    // Good base grid (16^3, dx=1): compatible with default MG (4 levels:
    // 16,8,4,2).
    const Grid3D good_grid(16, 16, 16, real{1}, real{1}, real{1});

    expect_invalid("config_validation_wrong_K_size", [&] {
        CudaContext ctx(0);
        AffinePeriodicFlowConfig cfg = default_config();
        const std::size_t n = good_grid.num_cells();
        DeviceBuffer<real> K(n - 1);
        blas::fill(ctx, K.span(), real{1});
        DeviceBuffer<real> u(u_size(good_grid)), v(v_size(good_grid)), w(w_size(good_grid));
        AffinePeriodicFlowWorkspace ws;
        (void)solve_affine_periodic_flow(ctx, good_grid, DeviceSpan<const real>(K.span()), cfg,
                                   AffinePeriodicVelocityView{u.span(), v.span(), w.span()}, ws);
    });

    expect_invalid("config_validation_wrong_velocity_span_size", [&] {
        CudaContext ctx(0);
        AffinePeriodicFlowConfig cfg = default_config();
        const std::size_t n = good_grid.num_cells();
        DeviceBuffer<real> K(n);
        blas::fill(ctx, K.span(), real{1});
        DeviceBuffer<real> u(u_size(good_grid) - 1), v(v_size(good_grid)), w(w_size(good_grid));
        AffinePeriodicFlowWorkspace ws;
        (void)solve_affine_periodic_flow(ctx, good_grid, DeviceSpan<const real>(K.span()), cfg,
                                   AffinePeriodicVelocityView{u.span(), v.span(), w.span()}, ws);
    });

    expect_invalid("config_validation_anisotropic_spacing", [&] {
        CudaContext ctx(0);
        const Grid3D bad_grid(16, 16, 16, real{1}, real{1}, real{2});
        AffinePeriodicFlowConfig cfg = default_config();
        const std::size_t n = bad_grid.num_cells();
        DeviceBuffer<real> K(n);
        blas::fill(ctx, K.span(), real{1});
        DeviceBuffer<real> u(u_size(bad_grid)), v(v_size(bad_grid)), w(w_size(bad_grid));
        AffinePeriodicFlowWorkspace ws;
        (void)solve_affine_periodic_flow(ctx, bad_grid, DeviceSpan<const real>(K.span()), cfg,
                                   AffinePeriodicVelocityView{u.span(), v.span(), w.span()}, ws);
    });

    expect_invalid("config_validation_odd_extent", [&] {
        CudaContext ctx(0);
        // Odd nx: isotropic spacing (dx==dy==dz), but not MG-coarsenable ->
        // distinct message from the anisotropic-spacing case above.
        const Grid3D odd_grid(15, 16, 16, real{1}, real{1}, real{1});
        AffinePeriodicFlowConfig cfg = default_config();
        const std::size_t n = odd_grid.num_cells();
        DeviceBuffer<real> K(n);
        blas::fill(ctx, K.span(), real{1});
        DeviceBuffer<real> u(u_size(odd_grid)), v(v_size(odd_grid)), w(w_size(odd_grid));
        AffinePeriodicFlowWorkspace ws;
        (void)solve_affine_periodic_flow(ctx, odd_grid, DeviceSpan<const real>(K.span()), cfg,
                                   AffinePeriodicVelocityView{u.span(), v.span(), w.span()}, ws);
    });

    rep.check(messages.size() == 4, "config_validation_four_distinct_messages",
              "count=" + std::to_string(messages.size()));
    return rep.overall_pass;
}

// ============================================================================
// Case (a): K=1 exactness at 16^3 and 32^3 (dx=1)
// ============================================================================

void case_k_equals_one(TestReport& rep, int N) {
    CudaContext ctx(0);
    const Grid3D grid(N, N, N, real{1}, real{1}, real{1});
    const std::size_t n = grid.num_cells();

    DeviceBuffer<real> K(n);
    blas::fill(ctx, K.span(), real{1});

    DeviceBuffer<real> u(u_size(grid)), v(v_size(grid)), w(w_size(grid));
    AffinePeriodicFlowWorkspace ws;
    AffinePeriodicFlowConfig cfg = default_config();

    const auto report = solve_affine_periodic_flow(
        ctx, grid, DeviceSpan<const real>(K.span()), cfg,
        AffinePeriodicVelocityView{u.span(), v.span(), w.span()}, ws);

    const std::string tag = "N" + std::to_string(N);

    for (int d = 0; d < 3; ++d) {
        rep.check(report.corrector_results[d].iterations == 0,
                  "k1_" + tag + "_corrector_" + std::to_string(d) + "_zero_iterations",
                  "iterations=" + std::to_string(report.corrector_results[d].iterations));
    }

    real max_abs_diff = real{0};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            const real expect = (i == j) ? real{1} : real{0};
            max_abs_diff = std::max(max_abs_diff, std::abs(report.K_eff[i][j] - expect));
        }
    }
    rep.check(max_abs_diff <= real{1e-14}, "k1_" + tag + "_K_eff_identity",
              "max_abs_diff=" + std::to_string(static_cast<double>(max_abs_diff)));

    const bool G_exact =
        report.G[0] == real{1} && report.G[1] == real{0} && report.G[2] == real{0};
    rep.check(G_exact, "k1_" + tag + "_G_exact",
              "G=(" + std::to_string(static_cast<double>(report.G[0])) + "," +
                  std::to_string(static_cast<double>(report.G[1])) + "," +
                  std::to_string(static_cast<double>(report.G[2])) + ")");

    const auto hu = download(u, u_size(grid));
    const auto hv = download(v, v_size(grid));
    const auto hw = download(w, w_size(grid));

    real max_u_dev = real{0}, max_v_dev = real{0}, max_w_dev = real{0};
    for (real val : hu) max_u_dev = std::max(max_u_dev, std::abs(val - real{1}));
    for (real val : hv) max_v_dev = std::max(max_v_dev, std::abs(val));
    for (real val : hw) max_w_dev = std::max(max_w_dev, std::abs(val));

    rep.check(max_u_dev == real{0}, "k1_" + tag + "_velocity_U_exact",
              "max|U-1|=" + std::to_string(static_cast<double>(max_u_dev)));
    rep.check(max_v_dev == real{0}, "k1_" + tag + "_velocity_V_exact",
              "max|V|=" + std::to_string(static_cast<double>(max_v_dev)));
    rep.check(max_w_dev == real{0}, "k1_" + tag + "_velocity_W_exact",
              "max|W|=" + std::to_string(static_cast<double>(max_w_dev)));

    rep.check(report.div_max_abs == real{0}, "k1_" + tag + "_div_max_abs_exact",
              "div_max_abs=" + std::to_string(static_cast<double>(report.div_max_abs)));

    std::printf("  K=1 %s: K_eff=[[%.6e,%.6e,%.6e],[%.6e,%.6e,%.6e],[%.6e,%.6e,%.6e]] "
                "G=(%.6e,%.6e,%.6e) iters=(%d,%d,%d) div_max_abs=%.6e\n",
                tag.c_str(), static_cast<double>(report.K_eff[0][0]),
                static_cast<double>(report.K_eff[0][1]), static_cast<double>(report.K_eff[0][2]),
                static_cast<double>(report.K_eff[1][0]), static_cast<double>(report.K_eff[1][1]),
                static_cast<double>(report.K_eff[1][2]), static_cast<double>(report.K_eff[2][0]),
                static_cast<double>(report.K_eff[2][1]), static_cast<double>(report.K_eff[2][2]),
                static_cast<double>(report.G[0]), static_cast<double>(report.G[1]),
                static_cast<double>(report.G[2]), report.corrector_results[0].iterations,
                report.corrector_results[1].iterations, report.corrector_results[2].iterations,
                static_cast<double>(report.div_max_abs));
}

// ============================================================================
// Shared: the four gating checks used by cases (b) and (c)
// ============================================================================

void check_symmetry_eigen_flux_div(TestReport& rep, const std::string& tag,
                                   const AffinePeriodicFlowReport& report, const real qbar[3]) {
    rep.check(report.symmetry_defect_rel <= real{1e-10}, tag + "_symmetry_defect_rel",
              "symmetry_defect_rel=" + std::to_string(static_cast<double>(report.symmetry_defect_rel)));

    const bool all_positive = report.eigenvalues_symmetric_part[0] > real{0} &&
                              report.eigenvalues_symmetric_part[1] > real{0} &&
                              report.eigenvalues_symmetric_part[2] > real{0};
    rep.check(all_positive, tag + "_eigenvalues_positive",
              "eig=(" +
                  std::to_string(static_cast<double>(report.eigenvalues_symmetric_part[0])) + "," +
                  std::to_string(static_cast<double>(report.eigenvalues_symmetric_part[1])) + "," +
                  std::to_string(static_cast<double>(report.eigenvalues_symmetric_part[2])) + ")");

    real max_flux_err = real{0};
    for (int i = 0; i < 3; ++i) {
        max_flux_err = std::max(max_flux_err, std::abs(report.achieved_mean_flux[i] - qbar[i]));
    }
    rep.check(max_flux_err <= real{1e-10}, tag + "_achieved_mean_flux",
              "max_flux_err=" + std::to_string(static_cast<double>(max_flux_err)));

    rep.check(report.div_max_abs <= real{1e-8}, tag + "_div_max_abs",
              "div_max_abs=" + std::to_string(static_cast<double>(report.div_max_abs)));
}

// ============================================================================
// Case (b): deterministic trig conductivity, 32^3 (dx=1, L=32)
// ============================================================================

std::vector<real> trig_conductivity_field(const Grid3D& grid) {
    const double L = static_cast<double>(grid.Lx());
    const double h = static_cast<double>(grid.dx);
    std::vector<real> K(grid.num_cells());
    for (int k = 0; k < grid.nz; ++k) {
        const double z = h * (k + 0.5);
        for (int j = 0; j < grid.ny; ++j) {
            const double y = h * (j + 0.5);
            for (int i = 0; i < grid.nx; ++i) {
                const double x = h * (i + 0.5);
                const double arg = 0.5 * std::sin(2.0 * kPi * x / L) * std::sin(2.0 * kPi * y / L) *
                                   std::sin(2.0 * kPi * z / L);
                K[grid.idx(i, j, k)] = static_cast<real>(std::exp(arg));
            }
        }
    }
    return K;
}

AffinePeriodicFlowReport solve_with_host_field(CudaContext& ctx, const Grid3D& grid,
                                               const std::vector<real>& host_K,
                                               const AffinePeriodicFlowConfig& cfg,
                                               DeviceBuffer<real>& K, DeviceBuffer<real>& u,
                                               DeviceBuffer<real>& v, DeviceBuffer<real>& w,
                                               AffinePeriodicFlowWorkspace& ws) {
    const std::size_t n = grid.num_cells();
    K.resize(n);
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(K.data(), host_K.data(), n * sizeof(real), cudaMemcpyHostToDevice));
    u.resize(u_size(grid));
    v.resize(v_size(grid));
    w.resize(w_size(grid));
    return solve_affine_periodic_flow(ctx, grid, DeviceSpan<const real>(K.span()), cfg,
                                      AffinePeriodicVelocityView{u.span(), v.span(), w.span()}, ws);
}

void case_trig_conductivity(TestReport& rep) {
    CudaContext ctx(0);
    const Grid3D grid(32, 32, 32, real{1}, real{1}, real{1});
    const auto host_K = trig_conductivity_field(grid);

    AffinePeriodicFlowConfig cfg = default_config();
    DeviceBuffer<real> K, u, v, w;
    AffinePeriodicFlowWorkspace ws;
    const auto report = solve_with_host_field(ctx, grid, host_K, cfg, K, u, v, w, ws);

    std::printf("  trig 32^3: K_eff=[[%.9e,%.9e,%.9e],[%.9e,%.9e,%.9e],[%.9e,%.9e,%.9e]]\n",
                static_cast<double>(report.K_eff[0][0]), static_cast<double>(report.K_eff[0][1]),
                static_cast<double>(report.K_eff[0][2]), static_cast<double>(report.K_eff[1][0]),
                static_cast<double>(report.K_eff[1][1]), static_cast<double>(report.K_eff[1][2]),
                static_cast<double>(report.K_eff[2][0]), static_cast<double>(report.K_eff[2][1]),
                static_cast<double>(report.K_eff[2][2]));
    std::printf("  trig 32^3: symmetry_defect_rel=%.6e eig=(%.9e,%.9e,%.9e) G=(%.9e,%.9e,%.9e) "
                "div_max_abs=%.6e div_rms=%.6e iters=(%d,%d,%d)\n",
                static_cast<double>(report.symmetry_defect_rel),
                static_cast<double>(report.eigenvalues_symmetric_part[0]),
                static_cast<double>(report.eigenvalues_symmetric_part[1]),
                static_cast<double>(report.eigenvalues_symmetric_part[2]),
                static_cast<double>(report.G[0]), static_cast<double>(report.G[1]),
                static_cast<double>(report.G[2]), static_cast<double>(report.div_max_abs),
                static_cast<double>(report.div_rms), report.corrector_results[0].iterations,
                report.corrector_results[1].iterations, report.corrector_results[2].iterations);

    check_symmetry_eigen_flux_div(rep, "trig32", report, cfg.qbar);
}

// ============================================================================
// Case (c): SF-18 periodic Gaussian field, K=exp(Y), 64^3 (dx=1)
// ============================================================================

void case_periodic_gaussian_conductivity(TestReport& rep) {
    CudaContext ctx(0);
    const Grid3D grid(64, 64, 64, real{1}, real{1}, real{1});
    const std::size_t n = grid.num_cells();

    PeriodicGaussianFieldConfig gcfg;
    gcfg.sigma2 = real{1};
    gcfg.corr_length = real{8};
    gcfg.seed = 12345ULL;
    gcfg.normalize_variance = true;

    DeviceBuffer<real> Y(n);
    PeriodicGaussianFieldWorkspace gws;
    generate_periodic_gaussian_field(ctx, grid, gcfg, Y.span(), gws);

    DeviceBuffer<real> K(n);
    device_exp(ctx, DeviceSpan<const real>(Y.span()), K.span());

    AffinePeriodicFlowConfig cfg = default_config();
    DeviceBuffer<real> u(u_size(grid)), v(v_size(grid)), w(w_size(grid));
    AffinePeriodicFlowWorkspace ws;

    const auto t0 = std::chrono::steady_clock::now();
    const auto report = solve_affine_periodic_flow(
        ctx, grid, DeviceSpan<const real>(K.span()), cfg,
        AffinePeriodicVelocityView{u.span(), v.span(), w.span()}, ws);
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(t1 - t0).count();

    std::printf("  gaussian 64^3: K_eff=[[%.9e,%.9e,%.9e],[%.9e,%.9e,%.9e],[%.9e,%.9e,%.9e]]\n",
                static_cast<double>(report.K_eff[0][0]), static_cast<double>(report.K_eff[0][1]),
                static_cast<double>(report.K_eff[0][2]), static_cast<double>(report.K_eff[1][0]),
                static_cast<double>(report.K_eff[1][1]), static_cast<double>(report.K_eff[1][2]),
                static_cast<double>(report.K_eff[2][0]), static_cast<double>(report.K_eff[2][1]),
                static_cast<double>(report.K_eff[2][2]));
    std::printf("  gaussian 64^3: symmetry_defect_rel=%.6e eig=(%.9e,%.9e,%.9e) "
                "G=(%.9e,%.9e,%.9e)\n",
                static_cast<double>(report.symmetry_defect_rel),
                static_cast<double>(report.eigenvalues_symmetric_part[0]),
                static_cast<double>(report.eigenvalues_symmetric_part[1]),
                static_cast<double>(report.eigenvalues_symmetric_part[2]),
                static_cast<double>(report.G[0]), static_cast<double>(report.G[1]),
                static_cast<double>(report.G[2]));
    std::printf("  gaussian 64^3: achieved_mean_flux=(%.9e,%.9e,%.9e) div_max_abs=%.6e "
                "div_rms=%.6e\n",
                static_cast<double>(report.achieved_mean_flux[0]),
                static_cast<double>(report.achieved_mean_flux[1]),
                static_cast<double>(report.achieved_mean_flux[2]),
                static_cast<double>(report.div_max_abs), static_cast<double>(report.div_rms));
    std::printf("  gaussian 64^3: corrector iterations=(%d,%d,%d) wall_time_s=%.6f\n",
                report.corrector_results[0].iterations, report.corrector_results[1].iterations,
                report.corrector_results[2].iterations, wall_seconds);

    check_symmetry_eigen_flux_div(rep, "gaussian64", report, cfg.qbar);
}

// ============================================================================
// Case (d): bitwise reproducibility
//
// Uses the deterministic trig 32^3 field (case (b)'s fixture) rather than
// re-generating the SF-18 Gaussian field: the spec requires "the same
// field" solved twice; the trig field is bitwise-deterministic to build
// from its closed-form host formula and cheaper to re-solve twice than the
// 64^3 Gaussian fixture, without changing what is under test (the SOLVER's
// determinism, not the field generator's, which SF-18 T02 already covers).
// Fresh output buffers and a fresh workspace are used for each solve, per
// the task's "fresh workspace or same; document" option.
// ============================================================================

void case_bitwise_reproducibility(TestReport& rep) {
    CudaContext ctx(0);
    const Grid3D grid(32, 32, 32, real{1}, real{1}, real{1});
    const auto host_K = trig_conductivity_field(grid);
    const AffinePeriodicFlowConfig cfg = default_config();

    DeviceBuffer<real> K1, u1, v1, w1;
    AffinePeriodicFlowWorkspace ws1;
    const auto report1 = solve_with_host_field(ctx, grid, host_K, cfg, K1, u1, v1, w1, ws1);

    DeviceBuffer<real> K2, u2, v2, w2;
    AffinePeriodicFlowWorkspace ws2;
    const auto report2 = solve_with_host_field(ctx, grid, host_K, cfg, K2, u2, v2, w2, ws2);

    const auto hu1 = download(u1, u_size(grid));
    const auto hu2 = download(u2, u_size(grid));
    const auto hv1 = download(v1, v_size(grid));
    const auto hv2 = download(v2, v_size(grid));
    const auto hw1 = download(w1, w_size(grid));
    const auto hw2 = download(w2, w_size(grid));

    const bool u_bitwise = std::memcmp(hu1.data(), hu2.data(), hu1.size() * sizeof(real)) == 0;
    const bool v_bitwise = std::memcmp(hv1.data(), hv2.data(), hv1.size() * sizeof(real)) == 0;
    const bool w_bitwise = std::memcmp(hw1.data(), hw2.data(), hw1.size() * sizeof(real)) == 0;

    rep.check(u_bitwise, "repro_velocity_U_bitwise_identical");
    rep.check(v_bitwise, "repro_velocity_V_bitwise_identical");
    rep.check(w_bitwise, "repro_velocity_W_bitwise_identical");

    bool K_eff_bitwise = true;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            K_eff_bitwise = K_eff_bitwise && (report1.K_eff[i][j] == report2.K_eff[i][j]);
        }
    }
    rep.check(K_eff_bitwise, "repro_K_eff_bitwise_identical");

    const bool G_bitwise = report1.G[0] == report2.G[0] && report1.G[1] == report2.G[1] &&
                           report1.G[2] == report2.G[2];
    rep.check(G_bitwise, "repro_G_bitwise_identical");

    const bool flux_bitwise = report1.achieved_mean_flux[0] == report2.achieved_mean_flux[0] &&
                              report1.achieved_mean_flux[1] == report2.achieved_mean_flux[1] &&
                              report1.achieved_mean_flux[2] == report2.achieved_mean_flux[2];
    rep.check(flux_bitwise, "repro_achieved_mean_flux_bitwise_identical");
}

} // namespace

int main() {
    TestReport rep;

    std::printf("=== SF-19 T02: config/validation contract ===\n");
    case_config_validation(rep);

    std::printf("=== SF-19 T02: (a) K=1 exactness, 16^3 ===\n");
    case_k_equals_one(rep, 16);

    std::printf("=== SF-19 T02: (a) K=1 exactness, 32^3 ===\n");
    case_k_equals_one(rep, 32);

    std::printf("=== SF-19 T02: (b) deterministic trig conductivity, 32^3 ===\n");
    case_trig_conductivity(rep);

    std::printf("=== SF-19 T02: (c) SF-18 periodic Gaussian conductivity, 64^3 ===\n");
    case_periodic_gaussian_conductivity(rep);

    std::printf("=== SF-19 T02: (d) bitwise reproducibility ===\n");
    case_bitwise_reproducibility(rep);

    std::printf("\n=== affine_periodic_flow: %d checks, %s ===\n", rep.checks,
                rep.overall_pass ? "PASS" : "FAIL");
    return rep.overall_pass ? 0 : 1;
}
