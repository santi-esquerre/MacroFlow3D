/**
 * @file streamline_closure_integrator_tests.cu
 * @brief SF-30 N1 fast contract tests: SF-19 potential accessor and the
 *        oracle streamline integrator on analytic direction fields.
 *
 * Standalone ctest runner in the `TestReport` / `[PASS]/[FAIL]` style of
 * `tests/flow/affine_periodic_flow_tests.cu`. Every gate below was fixed in
 * the SF-30 N1 task specification before this file existed; none may be
 * loosened to make a case pass.
 *
 * Cases: accessor_contract (GPU, 16^3), fields_reference_values, seed_points,
 * uniform_tilt, shear_closed_form, round_trip, backflow_first_return,
 * thread_independence, statistics_helper, timing_record.
 */

#include "apps/closure_gate/closure_fields.hpp"
#include "apps/closure_gate/closure_statistics.hpp"
#include "apps/closure_gate/streamline_integrator.hpp"

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/physics/flow/AffinePeriodicFlowSolver.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace macroflow3d;
using namespace macroflow3d::closure_gate;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr std::uint64_t kSeedRng = 20261005ULL;

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

std::string fmt(const char* f, double a) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), f, a);
    return buf;
}

// ============================================================================
// 1. accessor_contract (GPU)
// ============================================================================

std::vector<real> download(const real* dptr, std::size_t n) {
    std::vector<real> host(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(host.data(), dptr, n * sizeof(real), cudaMemcpyDeviceToHost));
    return host;
}

struct FaceDiff {
    double rel[3];
};

// Recomputes every face flux K_f (G_d + sign * (h[b] - h[a]) / h) with
// K_f = 2 K_a K_b / (K_a + K_b) and compares with the solver's U, V, W.
FaceDiff recompute_faces(const Grid3D& g, const std::vector<real>& K, const std::vector<real>& hh,
                         const real G[3], const std::vector<real>& U, const std::vector<real>& V,
                         const std::vector<real>& W, double sign) {
    const int nx = g.nx, ny = g.ny, nz = g.nz;
    const double h = g.dx;
    auto cell = [&](int i, int j, int k) {
        return static_cast<std::size_t>(i) +
               static_cast<std::size_t>(nx) *
                   (static_cast<std::size_t>(j) + static_cast<std::size_t>(ny) * k);
    };
    auto flux = [&](std::size_t a, std::size_t b, double Gd) {
        const double Kf = 2.0 * K[a] * K[b] / (K[a] + K[b]);
        return Kf * (Gd + sign * (hh[b] - hh[a]) / h);
    };
    double md[3] = {0, 0, 0}, mx[3] = {0, 0, 0};
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i <= nx; ++i) {
                const std::size_t a = cell(i == 0 ? nx - 1 : i - 1, j, k), b = cell(i == nx ? 0 : i, j, k);
                const std::size_t f = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * (nx + 1) +
                                      static_cast<std::size_t>(k) * (nx + 1) * ny;
                md[0] = std::max(md[0], std::abs(flux(a, b, G[0]) - U[f]));
                mx[0] = std::max(mx[0], std::abs(static_cast<double>(U[f])));
            }
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j <= ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t a = cell(i, j == 0 ? ny - 1 : j - 1, k), b = cell(i, j == ny ? 0 : j, k);
                const std::size_t f = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * nx +
                                      static_cast<std::size_t>(k) * nx * (ny + 1);
                md[1] = std::max(md[1], std::abs(flux(a, b, G[1]) - V[f]));
                mx[1] = std::max(mx[1], std::abs(static_cast<double>(V[f])));
            }
    for (int k = 0; k <= nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t a = cell(i, j, k == 0 ? nz - 1 : k - 1), b = cell(i, j, k == nz ? 0 : k);
                const std::size_t f = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * nx +
                                      static_cast<std::size_t>(k) * nx * ny;
                md[2] = std::max(md[2], std::abs(flux(a, b, G[2]) - W[f]));
                mx[2] = std::max(mx[2], std::abs(static_cast<double>(W[f])));
            }
    FaceDiff d{};
    for (int c = 0; c < 3; ++c) d.rel[c] = md[c] / mx[c];
    return d;
}

struct SolveOut {
    physics::AffinePeriodicFlowReport report;
    std::vector<real> K, h, U, V, W;
};

SolveOut solve_field(CudaContext& ctx, const Grid3D& grid, AnalyticField f, double eps,
                     physics::AffinePeriodicFlowWorkspace& ws) {
    SolveOut out;
    std::vector<double> Y;
    fill_analytic_log_conductivity(grid, f, eps, Y);
    out.K.resize(Y.size());
    for (std::size_t c = 0; c < Y.size(); ++c) out.K[c] = std::exp(Y[c]);
    const std::size_t n = grid.num_cells();
    DeviceBuffer<real> K(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(K.data(), out.K.data(), n * sizeof(real), cudaMemcpyHostToDevice));
    DeviceBuffer<real> u(static_cast<std::size_t>(grid.nx + 1) * grid.ny * grid.nz);
    DeviceBuffer<real> v(static_cast<std::size_t>(grid.nx) * (grid.ny + 1) * grid.nz);
    DeviceBuffer<real> w(static_cast<std::size_t>(grid.nx) * grid.ny * (grid.nz + 1));
    physics::AffinePeriodicFlowConfig cfg; // default: qbar = (1,0,0), rtol 1e-10
    out.report = physics::solve_affine_periodic_flow(
        ctx, grid, DeviceSpan<const real>(K.span()), cfg,
        physics::AffinePeriodicVelocityView{u.span(), v.span(), w.span()}, ws);
    const auto span = ws.potential_fluctuation();
    out.h = download(span.data(), span.size());
    out.U = download(u.data(), u.size());
    out.V = download(v.data(), v.size());
    out.W = download(w.data(), w.size());
    return out;
}

void check_faces(TestReport& rep, const Grid3D& grid, const SolveOut& s, const std::string& tag) {
    const FaceDiff d = recompute_faces(grid, s.K, s.h, s.report.G, s.U, s.V, s.W, +1.0);
    char buf[256];
    std::snprintf(buf, sizeof(buf), "rel U=%.3e V=%.3e W=%.3e (gate <= 1e-13)", d.rel[0], d.rel[1],
                  d.rel[2]);
    rep.check(d.rel[0] <= 1e-13 && d.rel[1] <= 1e-13 && d.rel[2] <= 1e-13,
              "accessor_contract_" + tag + "_face_flux_recompute", buf);
    const FaceDiff f = recompute_faces(grid, s.K, s.h, s.report.G, s.U, s.V, s.W, -1.0);
    std::snprintf(buf, sizeof(buf), "rel U=%.3e V=%.3e W=%.3e (gate each > 1e-3)", f.rel[0],
                  f.rel[1], f.rel[2]);
    rep.check(f.rel[0] > 1e-3 && f.rel[1] > 1e-3 && f.rel[2] > 1e-3,
              "accessor_contract_" + tag + "_opposite_sign_disagrees", buf);
    double mean = 0.0, mxh = 0.0;
    for (double v : s.h) {
        mean += v;
        mxh = std::max(mxh, std::abs(v));
    }
    mean /= static_cast<double>(s.h.size());
    std::snprintf(buf, sizeof(buf), "|mean|=%.3e max|h|=%.3e (gate |mean| <= 1e-12 max|h|)",
                  std::abs(mean), mxh);
    rep.check(std::abs(mean) <= 1e-12 * mxh, "accessor_contract_" + tag + "_zero_mean", buf);
}

void case_accessor_contract(TestReport& rep) {
    CudaContext ctx(0);
    const Grid3D grid(16, 16, 16, 1.0 / 16, 1.0 / 16, 1.0 / 16);
    physics::AffinePeriodicFlowWorkspace ws;

    // (a) throws before any solve
    bool threw = false;
    try {
        (void)ws.potential_fluctuation();
    } catch (const std::logic_error& e) {
        threw = true;
        std::printf("  (a) message: %s\n", e.what());
    }
    rep.check(threw, "accessor_contract_a_throws_before_solve");

    // (b)-(d), (f): Y = 0.5 generic3d
    const SolveOut s1 = solve_field(ctx, grid, AnalyticField::generic3d, 0.5, ws);
    std::printf("  solve 1: G=(%.15e, %.15e, %.15e) pcg iters=(%d,%d,%d)\n", s1.report.G[0],
                s1.report.G[1], s1.report.G[2], s1.report.corrector_results[0].iterations,
                s1.report.corrector_results[1].iterations, s1.report.corrector_results[2].iterations);
    check_faces(rep, grid, s1, "b_solve1");
    rep.check(ws.potential_fluctuation().size() == grid.num_cells(), "accessor_contract_f_size",
              "size=" + std::to_string(ws.potential_fluctuation().size()));

    // (e) second solve with a different K on the same workspace
    const SolveOut s2 = solve_field(ctx, grid, AnalyticField::lester_brk, 0.5, ws);
    check_faces(rep, grid, s2, "e_solve2");
    double dmax = 0.0, h1max = 0.0;
    for (std::size_t c = 0; c < s1.h.size(); ++c) {
        dmax = std::max(dmax, std::abs(s2.h[c] - s1.h[c]));
        h1max = std::max(h1max, std::abs(s1.h[c]));
    }
    rep.check(dmax > 1e-3 * h1max, "accessor_contract_e_contents_changed",
              fmt("max|h2-h1|/max|h1|=%.3e", dmax / h1max));

    // (g) a failed solve (wrong K size) leaves the accessor unavailable
    {
        DeviceBuffer<real> Kbad(grid.num_cells() - 1);
        DeviceBuffer<real> u(17 * 16 * 16), v(16 * 17 * 16), w(16 * 16 * 17);
        physics::AffinePeriodicFlowConfig cfg;
        bool solve_threw = false;
        try {
            (void)physics::solve_affine_periodic_flow(
                ctx, grid, DeviceSpan<const real>(Kbad.span()), cfg,
                physics::AffinePeriodicVelocityView{u.span(), v.span(), w.span()}, ws);
        } catch (const std::invalid_argument&) {
            solve_threw = true;
        }
        bool acc_threw = false;
        try {
            (void)ws.potential_fluctuation();
        } catch (const std::logic_error&) {
            acc_threw = true;
        }
        rep.check(solve_threw && acc_threw, "accessor_contract_g_failed_solve_invalidates");
    }
    // (h) re-prepare for a different grid invalidates a successful solve
    {
        (void)solve_field(ctx, grid, AnalyticField::generic3d, 0.5, ws);
        const Grid3D g32(32, 32, 32, 1.0 / 32, 1.0 / 32, 1.0 / 32);
        ws.prepare(g32, physics::AffinePeriodicFlowConfig{});
        bool acc_threw = false;
        try {
            (void)ws.potential_fluctuation();
        } catch (const std::logic_error&) {
            acc_threw = true;
        }
        rep.check(acc_threw, "accessor_contract_h_reprepare_other_grid_invalidates");
    }
}

// ============================================================================
// 2. fields_reference_values
// ============================================================================

struct HostGrid {
    int nx, ny, nz;
    double dx, dy, dz;
};

void case_fields_reference_values(TestReport& rep) {
    const double pts[3][3] = {{0.1, 0.2, 0.3}, {0.37, 0.81, 0.55}, {0.9375, 0.03125, 0.6875}};
    const AnalyticField order[5] = {AnalyticField::control2d, AnalyticField::lester2021,
                                    AnalyticField::lester_brk, AnalyticField::two_mode,
                                    AnalyticField::generic3d};
    const double ref[3][5] = {
        {0.15702617665944663, 0.3963525491562422, 0.5528251992594198, -1.118033988749895,
         -1.6207671119919336},
        {0.5779916105734668, 0.19439104863557138, -0.1145344124306242, 1.302085971608936,
         1.0799028767927934},
        {1.031589929875204, 0.4998333342765735, 0.15243201620254626, 0.2736784992166832,
         1.7342877253026843}};
    double worst = 0.0;
    for (int p = 0; p < 3; ++p)
        for (int f = 0; f < 5; ++f) {
            const double v = analytic_log_conductivity(order[f], pts[p][0], pts[p][1], pts[p][2]);
            worst = std::max(worst, std::abs(v - ref[p][f]));
        }
    rep.check(worst <= 1e-14, "fields_reference_values_15", fmt("max abs err=%.3e", worst));

    const HostGrid g{8, 8, 8, 1.0 / 8, 1.0 / 8, 1.0 / 8};
    std::vector<double> Y;
    fill_analytic_log_conductivity(g, AnalyticField::generic3d, 0.7, Y);
    const int cells[3][3] = {{0, 0, 0}, {3, 5, 7}, {7, 2, 4}};
    double wf = 0.0;
    for (const auto& c : cells) {
        const double v = 0.7 * analytic_log_conductivity(AnalyticField::generic3d, (c[0] + 0.5) / 8,
                                                         (c[1] + 0.5) / 8, (c[2] + 0.5) / 8);
        wf = std::max(wf, std::abs(Y[c[0] + 8 * (c[1] + 8 * c[2])] - v));
    }
    rep.check(wf == 0.0, "fields_fill_matches_pointwise", fmt("max diff=%.3e", wf));

    std::mt19937_64 rng(12345);
    std::normal_distribution<double> nd(0.3, 1.7);
    std::vector<double> R(512);
    for (auto& v : R) v = nd(rng);
    const double sigma2 = 4.0;
    const X3AveragedControlReport cr = make_x3_averaged_control(g, sigma2, R);
    double kdev = 0.0, mean = 0.0;
    for (int k = 0; k < 8; ++k)
        for (int c = 0; c < 64; ++c) kdev = std::max(kdev, std::abs(R[c + 64 * k] - R[c]));
    for (double v : R) mean += v;
    mean /= 512.0;
    double var = 0.0;
    for (double v : R) var += (v - mean) * (v - mean);
    var /= 512.0;
    std::printf("  x3 control: raw_mean=%.6e raw_variance=%.6e scale=%.6e mean=%.3e var=%.15e\n",
                cr.raw_mean, cr.raw_variance, cr.applied_scale, mean, var);
    rep.check(kdev == 0.0, "x3_control_independent_of_k", fmt("max k-dev=%.3e", kdev));
    rep.check(std::abs(mean) <= 1e-14, "x3_control_zero_mean", fmt("|mean|=%.3e", std::abs(mean)));
    rep.check(std::abs(var - sigma2) / sigma2 <= 1e-13, "x3_control_variance_sigma2",
              fmt("rel err=%.3e", std::abs(var - sigma2) / sigma2));
    std::vector<double> Z(512);
    for (int k = 0; k < 8; ++k)
        for (int c = 0; c < 64; ++c) Z[c + 64 * k] = 0.25 * k; // independent of (i, j)
    bool threw = false;
    try {
        (void)make_x3_averaged_control(g, sigma2, Z);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    rep.check(threw, "x3_control_throws_on_ij_independent_field");
}

// ============================================================================
// 3. seed_points
// ============================================================================

void case_seed_points(TestReport& rep) {
    bool in_range = true, repeat = true;
    std::vector<std::array<double, 2>> s;
    for (std::uint64_t i = 0; i < 5; ++i) {
        const auto p = seed_point(kSeedRng, i);
        const auto q = seed_point(kSeedRng, i);
        std::printf("  seed_point(%llu, %llu) = (%.17g, %.17g)\n",
                    static_cast<unsigned long long>(kSeedRng), static_cast<unsigned long long>(i),
                    p[0], p[1]);
        in_range = in_range && p[0] >= 0.0 && p[0] < 1.0 && p[1] >= 0.0 && p[1] < 1.0;
        repeat = repeat && std::memcmp(&p, &q, sizeof(p)) == 0;
        s.push_back(p);
    }
    bool distinct = true;
    for (std::size_t a = 0; a < s.size(); ++a)
        for (std::size_t b = a + 1; b < s.size(); ++b)
            distinct = distinct && s[a][0] != s[b][0] && s[a][1] != s[b][1];
    rep.check(in_range, "seed_points_in_unit_square");
    rep.check(repeat, "seed_points_stateless_repeatable");
    rep.check(distinct, "seed_points_distinct");
}

// ============================================================================
// Analytic direction fields
// ============================================================================

struct TiltField {
    void operator()(const double*, double g[3], double& k) const {
        g[0] = 1.0;
        g[1] = 0.3;
        g[2] = -0.2;
        k = 2.0;
    }
};

struct ShearField {
    double b = 0.35;
    void operator()(const double x[3], double g[3], double& k) const {
        g[0] = 1.0;
        g[1] = 0.0;
        g[2] = b * std::cos(2.0 * kPi * x[2]);
        k = 1.0;
    }
};

inline double backflow_phi(double y) { return 1.0 + 0.9 * std::cos(2.0 * kPi * y); }

struct BackflowField {
    void operator()(const double x[3], double g[3], double& k) const {
        const double p = backflow_phi(x[1]);
        g[0] = std::cos(p);
        g[1] = std::sin(p);
        g[2] = 0.0;
        k = 1.0;
    }
};

struct StagnationField {
    void operator()(const double x[3], double g[3], double& k) const {
        if (x[0] > 0.5) {
            g[0] = g[1] = g[2] = 0.0;
        } else {
            g[0] = 1.0;
            g[1] = 0.3;
            g[2] = -0.2;
        }
        k = 1.0;
    }
};

double shear_exact_x3(double b, double x3_0, double x1) {
    return std::atan(std::exp(2.0 * kPi * b * x1) * std::tan(kPi * x3_0 + kPi / 4.0)) / kPi - 0.25;
}

IntegratorOptions opts(double tol, double h_max, int n_periods, int sigma = 1) {
    IntegratorOptions o;
    o.tol = tol;
    o.h_max = h_max;
    o.n_periods = n_periods;
    o.sigma = sigma;
    return o;
}

// ============================================================================
// 4. uniform_tilt
// ============================================================================

void case_uniform_tilt(TestReport& rep) {
    const TiltField f;
    const double sq = std::sqrt(1.13);
    double ed = 0.0, es = 0.0, et = 0.0;
    bool all_ok = true, nobf = true;
    for (std::uint64_t i = 0; i < 8; ++i) {
        const auto p = seed_point(kSeedRng, i);
        const StreamlineResult r = integrate_streamline(f, {0.0, p[0], p[1]}, opts(1e-8, 1.0 / 32, 3));
        all_ok = all_ok && r.status == StreamlineStatus::ok && r.periods_completed == 3;
        nobf = nobf && !r.backflow_encounter;
        for (int n = 1; n <= r.periods_completed; ++n) {
            const PlaneRecord& rc = r.records[static_cast<std::size_t>(n - 1)];
            ed = std::max(ed, std::max(std::abs(rc.x2 - p[0] - 0.3 * n), std::abs(rc.x3 - p[1] + 0.2 * n)));
            es = std::max(es, std::abs(rc.s - n * sq));
            et = std::max(et, std::abs(rc.tau - rc.s / (2.0 * sq)));
        }
        if (i == 0)
            std::printf("  seed 0: accepted=%lld rejected=%lld landing=%lld nfev=%lld\n",
                        r.accepted_steps, r.rejected_steps, r.landing_steps, r.field_evaluations);
    }
    rep.check(all_ok, "uniform_tilt_all_ok_3_periods");
    rep.check(ed <= 1e-13, "uniform_tilt_displacement", fmt("max err=%.3e", ed));
    rep.check(es <= 1e-13, "uniform_tilt_arclength", fmt("max err=%.3e", es));
    rep.check(et <= 1e-13, "uniform_tilt_travel_time", fmt("max err=%.3e", et));
    rep.check(nobf, "uniform_tilt_no_backflow");
}

// ============================================================================
// 5. shear_closed_form
// ============================================================================

void case_shear_closed_form(TestReport& rep) {
    const ShearField f;
    const double seeds[5] = {-0.2, -0.1, 0.0, 0.05, 0.15};
    const double tols[4] = {1e-4, 1e-6, 1e-8, 1e-10};
    double errs[4];
    double x2max = 0.0;
    bool landed_exact = true, all_ok = true;
    for (int t = 0; t < 4; ++t) {
        double e = 0.0;
        long long acc = 0, rej = 0, land = 0;
        for (double z0 : seeds) {
            const double x1_0 = 0.0;
            const StreamlineResult r = integrate_streamline(f, {x1_0, 0.4, z0}, opts(tols[t], 1.0 / 32, 1));
            all_ok = all_ok && r.status == StreamlineStatus::ok && r.periods_completed == 1;
            if (r.periods_completed < 1) continue;
            const PlaneRecord& rc = r.records[0];
            e = std::max(e, std::abs(rc.x3 - shear_exact_x3(f.b, z0, 1.0)));
            x2max = std::max(x2max, std::abs(rc.x2 - 0.4));
            const double expect = x1_0 + 1.0;
            landed_exact = landed_exact && std::memcmp(&rc.x1, &expect, sizeof(double)) == 0;
            acc += r.accepted_steps;
            rej += r.rejected_steps;
            land += r.landing_steps;
        }
        errs[t] = e;
        std::printf("  tol=%.0e  max|x3-x3_exact|=%.3e  accepted=%lld rejected=%lld landing=%lld\n",
                    tols[t], e, acc, rej, land);
    }
    bool mono = true;
    for (int t = 1; t < 4; ++t) mono = mono && errs[t] <= errs[t - 1] + 1e-13;
    rep.check(all_ok, "shear_all_ok");
    rep.check(errs[3] <= 1e-8, "shear_error_at_tol_1e-10", fmt("err=%.3e (gate <= 1e-8)", errs[3]));
    rep.check(mono, "shear_error_non_increasing_down_ladder");
    rep.check(x2max <= 1e-13, "shear_x2_displacement", fmt("max=%.3e", x2max));
    rep.check(landed_exact, "shear_landed_x1_bitwise_x1_0_plus_1");
    // Off-zero seed x1_0 also lands bitwise on x1_0 + 1.
    const StreamlineResult r = integrate_streamline(f, {0.3125, 0.4, 0.05}, opts(1e-8, 1.0 / 32, 2));
    const double e1 = 0.3125 + 1.0, e2 = 0.3125 + 2.0;
    rep.check(r.status == StreamlineStatus::ok && std::memcmp(&r.records[0].x1, &e1, 8) == 0 &&
                  std::memcmp(&r.records[1].x1, &e2, 8) == 0,
              "shear_landed_x1_bitwise_offset_seed_2_periods");
}

// ============================================================================
// 7. backflow_first_return (independent quadrature reference)
// ============================================================================

struct GaussLegendre {
    std::vector<double> x, w;
    explicit GaussLegendre(int n) : x(n), w(n) {
        for (int i = 0; i < n; ++i) {
            double z = std::cos(kPi * (i + 0.75) / (n + 0.5));
            for (int it = 0; it < 100; ++it) {
                double p0 = 1.0, p1 = 0.0;
                for (int j = 1; j <= n; ++j) {
                    const double p2 = p1;
                    p1 = p0;
                    p0 = ((2.0 * j - 1.0) * z * p1 - (j - 1.0) * p2) / j;
                }
                const double dp = n * (z * p0 - p1) / (z * z - 1.0);
                const double dz = p0 / dp;
                z -= dz;
                if (std::abs(dz) < 1e-16) {
                    double q0 = 1.0, q1 = 0.0;
                    for (int j = 1; j <= n; ++j) {
                        const double q2 = q1;
                        q1 = q0;
                        q0 = ((2.0 * j - 1.0) * z * q1 - (j - 1.0) * q2) / j;
                    }
                    const double dq = n * (z * q0 - q1) / (z * z - 1.0);
                    x[i] = z;
                    w[i] = 2.0 / ((1.0 - z * z) * dq * dq);
                    break;
                }
            }
        }
    }
    template <class F> double integrate(F f, double a, double b) const {
        const double c = 0.5 * (a + b), r = 0.5 * (b - a);
        double s = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i) s += w[i] * f(c + r * x[i]);
        return r * s;
    }
};

// Along a streamline of the backflow field, x2 is monotone (sin(phi) > 0)
// and x1(x2) = x1(start) + integral_{start}^{x2} cot(phi(y)) dy. Marching
// from `start` in the direction `dir` (+1: increasing x2, the forward
// curve; -1: decreasing x2, the backward curve), returns the FIRST x2 where
// the signed integral reaches `target` (+1 forward, -1 backward), refined by
// bisection. Composite Gauss-Legendre on cells of width dy.
double reference_crossing(double start, double dir, double target, double dy,
                          const GaussLegendre& gl) {
    auto cotphi = [](double y) { const double p = backflow_phi(y); return std::cos(p) / std::sin(p); };
    auto reached = [&](double v) { return target > 0.0 ? v >= target : v <= target; };
    double a = start, F = 0.0;
    for (int it = 0; it < 10000000; ++it) {
        const double b = a + dir * dy;
        const double Fb = F + gl.integrate(cotphi, a, b);
        if (reached(Fb)) {
            double lo = a, hi = b; // lo: not reached, hi: reached
            for (int k = 0; k < 200; ++k) {
                const double mid = 0.5 * (lo + hi);
                if (mid == lo || mid == hi) break;
                if (reached(F + gl.integrate(cotphi, a, mid))) hi = mid; else lo = mid;
            }
            return 0.5 * (lo + hi);
        }
        a = b;
        F = Fb;
    }
    return std::nan("");
}

// First x2* > x2_0 with integral_{x2_0}^{x2*} cot(phi(y)) dy = 1 (x1 from 0 to 1).
double reference_first_return(double x2_0, double dy, const GaussLegendre& gl) {
    return reference_crossing(x2_0, +1.0, 1.0, dy, gl);
}

void case_backflow_first_return(TestReport& rep) {
    const BackflowField f;
    const GaussLegendre gl(12);
    struct Seed { double x2_0, sanity; bool backflow; };
    const Seed seeds[4] = {{0.60, 1.4795, true}, {0.70, 1.5010, true}, {0.80, 1.5077, true},
                           {0.30, 0.5099, false}};
    for (const Seed& sd : seeds) {
        const double ref = reference_first_return(sd.x2_0, 1.0 / 1024, gl);
        const double ref2 = reference_first_return(sd.x2_0, 1.0 / 2048, gl);
        double g[3], k;
        const double xs[3] = {0.0, sd.x2_0, 0.1};
        f(xs, g, k);
        const StreamlineResult r = integrate_streamline(f, {0.0, sd.x2_0, 0.1}, opts(1e-10, 1.0 / 64, 1));
        const std::string tag = "backflow_seed_" + fmt("%.2f", sd.x2_0);
        std::printf("  x2_0=%.2f g1(seed)=%.4f ref x2*=%.15f (dy/2: %.15f) integrator=%.15f status=%s "
                    "bf=%d min_g1_hat=%.4f bf_arclength=%.4f accepted=%lld rejected=%lld "
                    "discarded_landings=%lld nfev=%lld\n",
                    sd.x2_0, g[0], ref, ref2,
                    r.periods_completed ? r.records[0].x2 : std::nan(""), to_string(r.status),
                    static_cast<int>(r.backflow_encounter), r.min_g1_hat, r.backflow_arclength,
                    r.accepted_steps, r.rejected_steps, r.landings_discarded, r.field_evaluations);
        rep.check(g[0] > 0.0, tag + "_g1_positive_at_seed");
        rep.check(std::abs(ref - ref2) <= 1e-12 && std::abs(ref - sd.sanity) <= 1e-3,
                  tag + "_reference_self_consistent_and_sane",
                  fmt("|ref(dy)-ref(dy/2)|=%.3e", std::abs(ref - ref2)));
        rep.check(r.status == StreamlineStatus::ok && r.periods_completed == 1, tag + "_status_ok");
        if (r.periods_completed == 1) {
            const double e = std::abs(r.records[0].x2 - ref);
            rep.check(e <= 1e-8, tag + "_first_return_vs_quadrature", fmt("err=%.3e (gate <= 1e-8)", e));
            rep.check(std::abs(r.records[0].x3 - 0.1) <= 1e-13, tag + "_x3_unchanged");
        }
        if (sd.backflow) {
            rep.check(r.backflow_encounter && r.min_g1_hat < 0.0 && r.backflow_arclength > 0.0,
                      tag + "_backflow_counted");
        } else {
            rep.check(!r.backflow_encounter && r.min_g1_hat > 0.0 && r.backflow_arclength == 0.0,
                      tag + "_no_backflow_control");
        }
    }

    // seed inside the backflow band
    const StreamlineResult rb = integrate_streamline(f, {0.0, 0.02, 0.1}, opts(1e-10, 1.0 / 64, 1));
    rep.check(rb.status == StreamlineStatus::seed_backflow && rb.accepted_steps == 0 &&
                  rb.rejected_steps == 0 && rb.periods_completed == 0,
              "backflow_seed_inside_band_seed_backflow",
              std::string("status=") + to_string(rb.status) + fmt(" min_g1_hat=%.4f", rb.min_g1_hat));
    // cap_exceeded
    bool cap_ok = true;
    for (double x2_0 : {0.60, 0.70, 0.80}) {
        IntegratorOptions o = opts(1e-10, 1.0 / 64, 1);
        o.max_arclength_per_period = 0.5;
        const StreamlineResult rc = integrate_streamline(f, {0.0, x2_0, 0.1}, o);
        cap_ok = cap_ok && rc.status == StreamlineStatus::cap_exceeded && rc.periods_completed == 0;
    }
    rep.check(cap_ok, "backflow_cap_exceeded_zero_periods");
    // stagnation
    const StreamlineResult rs = integrate_streamline(StagnationField{}, {0.0, 0.3, 0.3}, opts(1e-8, 1.0 / 32, 1));
    rep.check(rs.status == StreamlineStatus::stagnation && rs.periods_completed == 0,
              "stagnation_status", std::string("status=") + to_string(rs.status));
}

// ============================================================================
// 6. round_trip
// ============================================================================

template <class Field>
double round_trip(const Field& f, const std::array<double, 3>& seed, double h_max, bool& ok) {
    const StreamlineResult fw = integrate_streamline(f, seed, opts(1e-8, h_max, 1, +1));
    if (fw.status != StreamlineStatus::ok) {
        ok = false;
        return std::nan("");
    }
    const PlaneRecord& p = fw.records[0];
    const StreamlineResult bw = integrate_streamline(f, {p.x1, p.x2, p.x3}, opts(1e-8, h_max, 1, -1));
    if (bw.status != StreamlineStatus::ok) {
        ok = false;
        return std::nan("");
    }
    const PlaneRecord& q = bw.records[0];
    ok = ok && q.x1 == seed[0];
    return std::max({std::abs(q.x1 - seed[0]), std::abs(q.x2 - seed[1]), std::abs(q.x3 - seed[2])});
}

void case_round_trip(TestReport& rep) {
    const double gate = 100 * 1e-8;
    bool ok = true;
    double e4 = 0.0, e5 = 0.0, e7 = 0.0;
    for (std::uint64_t i = 0; i < 8; ++i) {
        const auto p = seed_point(kSeedRng, i);
        e4 = std::max(e4, round_trip(TiltField{}, {0.0, p[0], p[1]}, 1.0 / 32, ok));
    }
    for (double z0 : {-0.2, -0.1, 0.0, 0.05, 0.15})
        e5 = std::max(e5, round_trip(ShearField{}, {0.0, 0.4, z0}, 1.0 / 32, ok));
    // Case-7 seeds whose forward curve stays strictly above the seed plane
    // x1 = 0 between the seed and the return (so the backward first return
    // IS the seed): 0.60, 0.70 (backflow) and 0.30 (control).
    for (double x2_0 : {0.60, 0.70, 0.30})
        e7 = std::max(e7, round_trip(BackflowField{}, {0.0, x2_0, 0.1}, 1.0 / 64, ok));
    rep.check(ok, "round_trip_all_ok");
    rep.check(e4 <= gate, "round_trip_uniform_tilt", fmt("dist=%.3e (gate <= 1e-6)", e4));
    rep.check(e5 <= gate, "round_trip_shear", fmt("dist=%.3e (gate <= 1e-6)", e5));
    rep.check(e7 <= gate, "round_trip_backflow", fmt("dist=%.3e (gate <= 1e-6)", e7));

    // Seed 0.80: its forward curve dips BELOW x1 = 0 inside the backflow band
    // (x1 min ~ -0.054 near x2 ~ 1.14), so the backward first return from the
    // landed point is NOT the seed: it is the last x2 < x2* where x1 = 0. The
    // backward result is checked against that independent quadrature
    // reference with the same 100*tol gate, and must differ from the seed.
    {
        const GaussLegendre gl(12);
        const BackflowField f;
        const StreamlineResult fw = integrate_streamline(f, {0.0, 0.80, 0.1}, opts(1e-8, 1.0 / 64, 1, +1));
        const double x2s = reference_first_return(0.80, 1.0 / 1024, gl);
        const double back_ref = reference_crossing(x2s, -1.0, -1.0, 1.0 / 1024, gl);
        bool okb = fw.status == StreamlineStatus::ok;
        double e = std::nan("");
        double got = std::nan("");
        if (okb) {
            const PlaneRecord& p = fw.records[0];
            const StreamlineResult bw = integrate_streamline(f, {p.x1, p.x2, p.x3}, opts(1e-8, 1.0 / 64, 1, -1));
            okb = bw.status == StreamlineStatus::ok && bw.records[0].x1 == 0.0;
            if (okb) {
                got = bw.records[0].x2;
                e = std::abs(got - back_ref);
            }
        }
        std::printf("  seed 0.80 backward first return: integrator x2=%.15f reference=%.15f seed=0.80\n",
                    got, back_ref);
        rep.check(okb && e <= gate && std::abs(back_ref - 0.80) > 0.1,
                  "round_trip_backflow_0.80_backward_return_vs_quadrature",
                  fmt("err=%.3e (gate <= 1e-6)", e));
    }
}

// ============================================================================
// 8. thread_independence
// ============================================================================

std::vector<unsigned char> serialize(const std::vector<StreamlineResult>& rs) {
    std::vector<unsigned char> out;
    auto put = [&](const void* p, std::size_t n) {
        const auto* c = static_cast<const unsigned char*>(p);
        out.insert(out.end(), c, c + n);
    };
    for (const auto& r : rs) {
        const int st = static_cast<int>(r.status);
        put(&st, sizeof(st));
        put(&r.periods_completed, sizeof(int));
        for (const auto& rc : r.records) {
            put(&rc.x1, 8); put(&rc.x2, 8); put(&rc.x3, 8); put(&rc.s, 8); put(&rc.tau, 8);
        }
        const unsigned char bf = r.backflow_encounter ? 1 : 0;
        put(&bf, 1);
        put(&r.min_g1_hat, 8); put(&r.backflow_arclength, 8);
        put(&r.accepted_steps, 8); put(&r.rejected_steps, 8); put(&r.landing_steps, 8);
        put(&r.landing_rejected, 8); put(&r.landings_discarded, 8); put(&r.field_evaluations, 8);
        put(r.final_x, 24); put(&r.final_s, 8); put(&r.final_tau, 8);
    }
    return out;
}

void case_thread_independence(TestReport& rep) {
    std::vector<std::array<double, 3>> seeds;
    for (std::uint64_t i = 0; i < 64; ++i) {
        const auto p = seed_point(kSeedRng, i);
        seeds.push_back({0.0, p[0], p[1]});
    }
    const IntegratorOptions o = opts(1e-8, 1.0 / 32, 2);
    const auto r1 = integrate_streamlines(ShearField{}, seeds, o, 1);
    const auto r4 = integrate_streamlines(ShearField{}, seeds, o, 4);
    const auto b1 = serialize(r1), b4 = serialize(r4);
    // Per-seed result equals the single-call result too.
    const auto single = serialize({integrate_streamline(ShearField{}, seeds[17], o)});
    const auto from_batch = serialize({r4[17]});
    rep.check(b1.size() == b4.size() && std::memcmp(b1.data(), b4.data(), b1.size()) == 0 &&
                  single == from_batch,
              "thread_independence_bitwise_1_vs_4", "bytes=" + std::to_string(b1.size()));
}

// ============================================================================
// 9. statistics_helper
// ============================================================================

void case_statistics_helper(TestReport& rep) {
    const std::vector<std::array<double, 2>> d = {{1, 2}, {3, 4}, {5, 6}, {7, 8}};
    const DisplacementStatistics u = displacement_statistics(d);
    bool okU = std::abs(u.mean[0] - 4) <= 1e-15 && std::abs(u.mean[1] - 5) <= 1e-15 &&
               std::abs(u.R - std::sqrt(10.0)) <= 1e-15 && std::abs(u.var_d2 - 5) <= 1e-15 &&
               std::abs(u.var_d3 - 5) <= 1e-15 && std::abs(u.rms_abs - std::sqrt(51.0)) <= 1e-15 &&
               std::abs(u.max_abs - std::sqrt(113.0)) <= 1e-15;
    std::printf("  unweighted: mean=(%.17g,%.17g) R=%.17g var=(%.17g,%.17g) rms=%.17g max=%.17g\n",
                u.mean[0], u.mean[1], u.R, u.var_d2, u.var_d3, u.rms_abs, u.max_abs);
    rep.check(okU, "statistics_unweighted");
    const DisplacementStatistics w = displacement_statistics(d, {1, 1, 1, 3});
    bool okW = std::abs(w.mean[0] - 5) <= 1e-15 && std::abs(w.mean[1] - 6) <= 1e-15 &&
               std::abs(w.R - std::sqrt(64.0 / 6.0)) <= 1e-15 &&
               std::abs(w.var_d2 - 32.0 / 6.0) <= 1e-15 && std::abs(w.var_d3 - 32.0 / 6.0) <= 1e-15 &&
               std::abs(w.rms_abs - std::sqrt(430.0 / 6.0)) <= 1e-15;
    std::printf("  weighted (1,1,1,3): mean=(%.17g,%.17g) R=%.17g var=(%.17g,%.17g) rms=%.17g\n",
                w.mean[0], w.mean[1], w.R, w.var_d2, w.var_d3, w.rms_abs);
    rep.check(okW, "statistics_weighted");
}

} // namespace

int main() {
    const auto t0 = std::chrono::steady_clock::now();
    TestReport rep;
    try {
        std::printf("=== SF-30 N1: 1 accessor_contract (GPU, 16^3) ===\n");
        case_accessor_contract(rep);
        std::printf("=== SF-30 N1: 2 fields_reference_values ===\n");
        case_fields_reference_values(rep);
        std::printf("=== SF-30 N1: 3 seed_points ===\n");
        case_seed_points(rep);
        std::printf("=== SF-30 N1: 4 uniform_tilt ===\n");
        case_uniform_tilt(rep);
        std::printf("=== SF-30 N1: 5 shear_closed_form ===\n");
        case_shear_closed_form(rep);
        std::printf("=== SF-30 N1: 6 round_trip ===\n");
        case_round_trip(rep);
        std::printf("=== SF-30 N1: 7 backflow_first_return ===\n");
        case_backflow_first_return(rep);
        std::printf("=== SF-30 N1: 8 thread_independence ===\n");
        case_thread_independence(rep);
        std::printf("=== SF-30 N1: 9 statistics_helper ===\n");
        case_statistics_helper(rep);
    } catch (const std::exception& e) {
        rep.check(false, "unexpected_exception", e.what());
    }
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("=== SF-30 N1: 10 timing_record ===\n  wall time %.3f s (no gate; expected well under 20 s)\n",
                secs);
    std::printf("\n=== streamline_closure_integrator: %d checks, %s ===\n", rep.checks,
                rep.overall_pass ? "PASS" : "FAIL");
    return rep.overall_pass ? 0 : 1;
}
