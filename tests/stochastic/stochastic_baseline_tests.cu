/**
 * @file stochastic_baseline_tests.cu
 * @brief Gaussian-only contract tests for the baseline direct-sum (randomized
 *        spectral method) log-conductivity generator `stochastic.cu`.
 *
 * Decision: docs/decisions/2026-10-07-gaussian-covariance-only.md.
 * Target covariance: C(r) = sigma2 * exp(-(r/lambda)^2).
 *
 * Standalone ctest-friendly runner (mirrors tests/stochastic/
 * periodic_gaussian_tests.cu: printed checks, non-zero exit on failure).
 *
 * ALL tolerances below are a priori (derivation stated next to each check)
 * and were fixed BEFORE the first run. They must not be relaxed after seeing
 * results: a failing statistical check means the generator is suspect.
 *
 * Every statistical check is a 5-sigma band; with ~40 such checks the
 * family-wise false-failure probability is < 40 * 5.7e-7 ~ 2e-5. The seeds
 * are fixed, so the outcome is deterministic for a given build/GPU.
 *
 * Cases:
 *   A. Mode-sampler contract (8^3, n_modes = 65536, lambda = 2, sigma2 = 1).
 *   B. Field fixture (64^3, h = 1, lambda = 2, sigma2 = 1, n_modes = 4096).
 *   C. Retirement contract (generator guard + config validator).
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/io/config/ConfigDefaults.hpp"
#include "src/io/config/ConfigValidator.hpp"
#include "src/physics/common/physics_config.hpp"
#include "src/physics/stochastic/stochastic.cuh"
#include "src/runtime/cuda_check.cuh"
#include "src/runtime/CudaContext.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace macroflow3d;
using namespace macroflow3d::physics;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr unsigned long long kSeed = 12345ULL;
constexpr unsigned long long kOtherSeed = 54321ULL;

struct TestReport {
    bool overall_pass = true;
    int checks = 0;

    void check(bool cond, const std::string& name, const std::string& detail = "") {
        ++checks;
        std::printf("[%s] %s%s%s\n", cond ? "PASS" : "FAIL", name.c_str(),
                    detail.empty() ? "" : "  ", detail.c_str());
        overall_pass = overall_pass && cond;
    }

    // |observed - expected| <= tol, printing all three numbers.
    void band(const std::string& name, double observed, double expected, double tol) {
        char buf[256];
        const double dev = std::fabs(observed - expected);
        std::snprintf(buf, sizeof(buf), "observed=%.6g expected=%.6g |dev|=%.4g tol=%.4g", observed,
                      expected, dev, tol);
        check(std::isfinite(observed) && dev <= tol, name, buf);
    }
};

std::vector<real> download(const DeviceBuffer<real>& buf, size_t n) {
    std::vector<real> host(n);
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(host.data(), buf.data(), n * sizeof(real), cudaMemcpyDeviceToHost));
    return host;
}

StochasticConfig make_cfg(real sigma2, real lambda, int n_modes, unsigned long long seed,
                          real K_mean = 1.0) {
    StochasticConfig c;
    c.sigma2 = sigma2;
    c.corr_length = lambda;
    c.n_modes = n_modes;
    c.covariance_type = 1;
    c.seed = seed;
    c.K_geometric_mean = K_mean;
    return c;
}

// init RNG with cfg.seed + generate logK (same sequence as generate_K_field).
void generate(StochasticWorkspace& ws, const Grid3D& grid, const StochasticConfig& cfg,
              const CudaContext& ctx) {
    init_stochastic_rng(ws, cfg.seed, ctx);
    generate_gaussian_field(ws, grid, cfg, ctx);
    ctx.synchronize();
}

// ============================================================================
// A. Mode-sampler contract
// ============================================================================
//
// kernel_random_modes_gauss draws kappa from the chi_3 density
// kappa^2 exp(-kappa^2/2) (rejection on [0, k_max]), an isotropic direction,
// and sets k = kappa * sqrt(2)/lambda * dir. Hence u_d = k_d lambda / sqrt(2)
// = kappa * dir_d is N(0, 1) per axis, independent across axes, and
// kappa^2 = sum u_d^2 is chi^2 with 3 dof. This pins the convention:
// E[cos(k.r)] = exp(-|r|^2/lambda^2). (Legacy convention would give
// Var[u_d] = pi/4 = 0.785; an exponential covariance has infinite Var[u_d].)
void case_mode_sampler(TestReport& rep, const CudaContext& ctx) {
    const int N = 65536;
    const real lambda = 2.0;
    const Grid3D grid(8, 8, 8, 1.0, 1.0, 1.0);
    const StochasticConfig cfg = make_cfg(1.0, lambda, N, kSeed);

    StochasticWorkspace ws;
    ws.allocate(grid, cfg);
    generate(ws, grid, cfg, ctx);

    const auto k1 = download(ws.k1, N);
    const auto k2 = download(ws.k2, N);
    const auto k3 = download(ws.k3, N);
    const auto a = download(ws.coef_a, N);
    const auto b = download(ws.coef_b, N);

    const double s = static_cast<double>(lambda) / std::sqrt(2.0);
    double mu[3] = {0, 0, 0}, m2[3] = {0, 0, 0};
    double cxy = 0, cyz = 0, cxz = 0, mk2 = 0, mk4 = 0, max_kappa = 0;
    double ma = 0, mb = 0, va = 0, vb = 0;
    bool finite = true;
    for (int i = 0; i < N; ++i) {
        const double u[3] = {k1[i] * s, k2[i] * s, k3[i] * s};
        for (int d = 0; d < 3; ++d) {
            mu[d] += u[d];
            m2[d] += u[d] * u[d];
            finite = finite && std::isfinite(u[d]);
        }
        cxy += u[0] * u[1];
        cyz += u[1] * u[2];
        cxz += u[0] * u[2];
        const double kap2 = u[0] * u[0] + u[1] * u[1] + u[2] * u[2];
        mk2 += kap2;
        mk4 += kap2 * kap2;
        max_kappa = std::max(max_kappa, std::sqrt(kap2));
        ma += a[i];
        mb += b[i];
        va += static_cast<double>(a[i]) * a[i];
        vb += static_cast<double>(b[i]) * b[i];
        finite = finite && std::isfinite(a[i]) && std::isfinite(b[i]);
    }
    const double Nd = N;
    rep.check(finite, "A_modes_finite");

    // Sample variance of N iid N(0,1): sd = sqrt(2/N) -> tol 5 sqrt(2/N) = 0.0276.
    const double tol_var = 5.0 * std::sqrt(2.0 / Nd);
    const char* ax = "xyz";
    for (int d = 0; d < 3; ++d) {
        const double mean = mu[d] / Nd;
        const double var = m2[d] / Nd - mean * mean;
        rep.band(std::string("A_var_u_") + ax[d], var, 1.0, tol_var);
    }
    // Cross moment of independent N(0,1): Var[u_x u_y] = 1 -> tol 5/sqrt(N) = 0.0195.
    const double tol_cross = 5.0 / std::sqrt(Nd);
    rep.band("A_cross_u_xy", cxy / Nd, 0.0, tol_cross);
    rep.band("A_cross_u_yz", cyz / Nd, 0.0, tol_cross);
    rep.band("A_cross_u_xz", cxz / Nd, 0.0, tol_cross);
    // chi^2_3: E = 3, Var = 6 -> tol 5 sqrt(6/N) = 0.0478.
    rep.band("A_mean_kappa2", mk2 / Nd, 3.0, 5.0 * std::sqrt(6.0 / Nd));
    // E[X^2] = 15, E[X^4] = 3*5*7*9 = 945 -> Var[X^2] = 720 -> tol 5 sqrt(720/N) = 0.524.
    rep.band("A_mean_kappa4", mk4 / Nd, 15.0, 5.0 * std::sqrt(720.0 / Nd));
    // a, b ~ N(0,1): mean tol 5/sqrt(N), variance tol 5 sqrt(2/N).
    const double mean_a = ma / Nd, mean_b = mb / Nd;
    rep.band("A_mean_a", mean_a, 0.0, tol_cross);
    rep.band("A_mean_b", mean_b, 0.0, tol_cross);
    rep.band("A_var_a", va / Nd - mean_a * mean_a, 1.0, tol_var);
    rep.band("A_var_b", vb / Nd - mean_b * mean_b, 1.0, tol_var);
    // P(chi_3 > 8) ~ 8e-14 per sample -> ~5e-9 over N samples.
    char buf[160];
    std::snprintf(buf, sizeof(buf), "max_kappa=%.6g bound=8 (P(chi_3>8)~8e-14/sample)", max_kappa);
    rep.check(max_kappa <= 8.0, "A_max_kappa_le_8", buf);
    std::snprintf(buf, sizeof(buf), "max_kappa=%.6g k_max=100", max_kappa);
    rep.check(max_kappa <= 100.0, "A_max_kappa_le_kmax", buf);
}

// ============================================================================
// B. Field fixture
// ============================================================================
void case_field(TestReport& rep, const CudaContext& ctx) {
    const int n = 64;
    const real h = 1.0;
    const real lambda = 2.0;
    const int N = 4096;
    const Grid3D grid(n, n, n, h, h, h);
    const size_t nc = grid.num_cells();
    const StochasticConfig cfg = make_cfg(1.0, lambda, N, kSeed);

    StochasticWorkspace ws;
    ws.allocate(grid, cfg);

    // --- Reproducibility ---------------------------------------------------
    generate(ws, grid, cfg, ctx);
    const auto Y = download(ws.logK, nc);
    const auto k1 = download(ws.k1, N);
    const auto k2 = download(ws.k2, N);
    const auto k3 = download(ws.k3, N);
    const auto a = download(ws.coef_a, N);
    const auto b = download(ws.coef_b, N);

    generate(ws, grid, cfg, ctx);
    const auto Y_again = download(ws.logK, nc);
    rep.check(std::memcmp(Y.data(), Y_again.data(), nc * sizeof(real)) == 0,
              "B_same_seed_bitwise_identical");

    generate(ws, grid, make_cfg(1.0, lambda, N, kOtherSeed), ctx);
    const auto Y_other = download(ws.logK, nc);
    size_t ndiff = 0;
    for (size_t i = 0; i < nc; ++i)
        ndiff += (Y[i] != Y_other[i]) ? 1 : 0;
    rep.check(ndiff > 0, "B_other_seed_differs",
              "differing_cells=" + std::to_string(ndiff) + "/" + std::to_string(nc));

    // --- Independent CPU double re-evaluation -------------------------------
    // logK = (sigma/sqrt(N)) sum_i (a_i sin phi_i + b_i cos phi_i),
    // phi_i = k_i . h (ix+1/2, iy+1/2, iz+1/2). Expected discrepancy: per-mode
    // phase/sin differences of a few ulp(|phi| <~ 400) ~ 1e-13, random-signed
    // over 4096 modes and scaled by 1/64 -> ~1e-13; 1e-10 is a wide margin
    // that still rejects any convention change (vertex vs cell centre,
    // sigma vs sigma2, 1/N vs 1/sqrt(N)).
    {
        const double sigma_f = 1.0;
        double max_err = 0.0;
        for (int s = 0; s < 256; ++s) {
            const size_t idx = (static_cast<size_t>(s) * 1031u + 17u) % nc;
            const int ix = static_cast<int>(idx % n);
            const int iy = static_cast<int>((idx / n) % n);
            const int iz = static_cast<int>(idx / (static_cast<size_t>(n) * n));
            const double x = h * (ix + 0.5), y = h * (iy + 0.5), z = h * (iz + 0.5);
            double sum = 0.0;
            for (int i = 0; i < N; ++i) {
                const double ph = k1[i] * x + k2[i] * y + k3[i] * z;
                sum += a[i] * std::sin(ph) + b[i] * std::cos(ph);
            }
            const double cpu = sigma_f / std::sqrt(static_cast<double>(N)) * sum;
            max_err = std::max(max_err, std::fabs(cpu - Y[idx]));
        }
        char buf[128];
        std::snprintf(buf, sizeof(buf), "max|gpu-cpu|=%.3g over 256 cells tol=1e-10", max_err);
        rep.check(max_err <= 1e-10, "B_cpu_reevaluation", buf);
    }

    // --- Mean / variance ------------------------------------------------------
    const double V = static_cast<double>(nc) * h * h * h;
    const double lam3 = static_cast<double>(lambda) * lambda * lambda;
    double mean = 0.0;
    for (size_t i = 0; i < nc; ++i)
        mean += Y[i];
    mean /= static_cast<double>(nc);
    double s2 = 0.0;
    for (size_t i = 0; i < nc; ++i)
        s2 += (Y[i] - mean) * (Y[i] - mean);
    s2 /= static_cast<double>(nc);
    // Var[spatial mean] ~ (1/V) int C = sigma2 pi^{3/2} lambda^3 / V -> 5 sd ~ 0.065.
    rep.band("B_spatial_mean", mean, 0.0, 5.0 * std::sqrt(std::pow(kPi, 1.5) * lam3 / V));
    // s2/sigma2 fluctuation: finite-mode term (1/N) Var[(a^2+b^2)/2] = 1/N plus
    // the ergodic term 2 I2/V with I2 = int rho^2 = (pi/2)^{3/2} lambda^3 -> ~0.095.
    const double I2 = std::pow(kPi / 2.0, 1.5) * lam3;
    rep.band("B_spatial_variance", s2 / 1.0, 1.0, 5.0 * std::sqrt(1.0 / N + 2.0 * I2 / V));

    // --- Empirical covariance per axis --------------------------------------
    // c_d(m) = mean over interior pairs of Y(x) Y(x + m e_d) (no periodic wrap).
    // Per-mode contribution (a^2+b^2)/2 cos(k.r) has variance
    // 1 + rho(2r) - rho(r)^2 -> finite-mode term /N; ergodic term
    // (1/V_p) int [rho(s)^2 + rho(s+r) rho(s-r)] ds <= 2 I2 / V_p, with
    // V_p = (n-m) n^2 h^3 the pair-count volume (slightly conservative vs V).
    auto at = [&](int ix, int iy, int iz) {
        return static_cast<double>(
            Y[ix + static_cast<size_t>(n) * (iy + static_cast<size_t>(n) * iz)]);
    };
    for (int m = 1; m <= 4; ++m) {
        const double r = m * h;
        const double rho = std::exp(-(r / lambda) * (r / lambda));
        const double rho2r = std::exp(-(2.0 * r / lambda) * (2.0 * r / lambda));
        const double Vp = static_cast<double>(n - m) * n * n * h * h * h;
        const double tol = 5.0 * std::sqrt((1.0 + rho2r - rho * rho) / N + 2.0 * I2 / Vp);
        double c[3] = {0, 0, 0};
        for (int iz = 0; iz < n; ++iz)
            for (int iy = 0; iy < n; ++iy)
                for (int ix = 0; ix < n; ++ix) {
                    const double y0 = at(ix, iy, iz);
                    if (ix + m < n)
                        c[0] += y0 * at(ix + m, iy, iz);
                    if (iy + m < n)
                        c[1] += y0 * at(ix, iy + m, iz);
                    if (iz + m < n)
                        c[2] += y0 * at(ix, iy, iz + m);
                }
        const double npairs = static_cast<double>(n - m) * n * n;
        for (int d = 0; d < 3; ++d)
            c[d] /= npairs;
        const char* ax = "xyz";
        for (int d = 0; d < 3; ++d) {
            char nm[64];
            std::snprintf(nm, sizeof(nm), "B_cov_%c_lag%d(r/lambda=%.1f)", ax[d], m, r / lambda);
            rep.band(nm, c[d], rho, tol);
        }
        char nm[64];
        std::snprintf(nm, sizeof(nm), "B_isotropy_xy_lag%d", m);
        rep.band(nm, c[0] - c[1], 0.0, std::sqrt(2.0) * tol);
        std::snprintf(nm, sizeof(nm), "B_isotropy_yz_lag%d", m);
        rep.band(nm, c[1] - c[2], 0.0, std::sqrt(2.0) * tol);
    }

    // --- Lognormal transform with K_mean = 2.5 -------------------------------
    {
        generate(ws, grid, cfg, ctx); // logK back to the seed-12345 field
        DeviceBuffer<real> K(nc);
        const StochasticConfig cfg_k = make_cfg(1.0, lambda, N, kSeed, 2.5);
        generate_K_lognormal(DeviceSpan<real>(K.data(), nc),
                             DeviceSpan<const real>(ws.logK.data(), nc), grid, cfg_k, ctx);
        ctx.synchronize();
        const auto Kh = download(K, nc);
        double max_rel = 0.0;
        for (int s = 0; s < 256; ++s) {
            const size_t idx = (static_cast<size_t>(s) * 1031u + 17u) % nc;
            const double ref = 2.5 * std::exp(static_cast<double>(Y[idx]));
            max_rel = std::max(max_rel, std::fabs(Kh[idx] - ref) / ref);
        }
        char buf[128];
        std::snprintf(buf, sizeof(buf), "max_rel_err=%.3g over 256 cells tol=1e-12", max_rel);
        rep.check(max_rel <= 1e-12, "B_K_mean_2p5_lognormal", buf);
    }

    // --- sigma2 = 0 -> homogeneous field --------------------------------------
    {
        const StochasticConfig cfg0 = make_cfg(0.0, lambda, N, kSeed, 1.0);
        generate(ws, grid, cfg0, ctx);
        const auto Y0 = download(ws.logK, nc);
        size_t nonzero = 0, negzero = 0;
        for (size_t i = 0; i < nc; ++i) {
            nonzero += (Y0[i] != 0.0) ? 1 : 0;
            negzero += (Y0[i] == 0.0 && std::signbit(Y0[i])) ? 1 : 0;
        }
        // 0.0 * sum carries the sign of sum, so -0.0 is expected in ~half the cells;
        // the contract is value equality with 0 (both zeros), and exp(+-0) = 1 exactly.
        rep.check(nonzero == 0, "B_sigma2_zero_logK_is_zero",
                  "nonzero=" + std::to_string(nonzero) +
                      " (cells holding -0.0: " + std::to_string(negzero) + ")");
        DeviceBuffer<real> K(nc);
        generate_K_lognormal(DeviceSpan<real>(K.data(), nc),
                             DeviceSpan<const real>(ws.logK.data(), nc), grid, cfg0, ctx);
        ctx.synchronize();
        const auto Kh = download(K, nc);
        const real one = 1.0;
        size_t notbitwise = 0;
        for (size_t i = 0; i < nc; ++i)
            notbitwise += (std::memcmp(&Kh[i], &one, sizeof(real)) != 0) ? 1 : 0;
        rep.check(notbitwise == 0, "B_sigma2_zero_K_bitwise_K_mean(1.0)",
                  "cells_not_bitwise_1.0=" + std::to_string(notbitwise));
    }
}

// ============================================================================
// C. Retirement contract
// ============================================================================
void case_retirement(TestReport& rep, const CudaContext& ctx) {
    {
        StochasticConfig def;
        io::AppConfig app_def = io::make_default_config();
        rep.check(def.covariance_type == 1, "C_StochasticConfig_default_is_1",
                  "value=" + std::to_string(def.covariance_type));
        rep.check(app_def.stochastic.covariance_type == 1, "C_AppConfig_default_is_1",
                  "value=" + std::to_string(app_def.stochastic.covariance_type));
    }

    const Grid3D grid(8, 8, 8, 1.0, 1.0, 1.0);
    for (int ct : {0, 2}) {
        StochasticConfig cfg = make_cfg(1.0, 2.0, 64, kSeed);
        cfg.covariance_type = ct;
        StochasticWorkspace ws;
        ws.allocate(grid, cfg);
        init_stochastic_rng(ws, cfg.seed, ctx);
        const std::string name = "C_generator_throws_covariance_type_" + std::to_string(ct);
        try {
            generate_gaussian_field(ws, grid, cfg, ctx);
            rep.check(false, name, "no exception thrown");
        } catch (const std::invalid_argument& e) {
            rep.check(true, name, std::string("msg=\"") + e.what() + "\"");
        } catch (const std::exception& e) {
            rep.check(false, name, std::string("wrong exception type: ") + e.what());
        }
    }

    auto key_errors = [](const io::ValidationResult& r, std::string& first) {
        int count = 0;
        for (const auto& e : r.errors)
            if (e.find("stochastic.covariance_type") != std::string::npos) {
                if (count == 0)
                    first = e;
                ++count;
            }
        return count;
    };
    for (int ct : {0, 2}) {
        io::AppConfig app = io::make_default_config();
        app.stochastic.covariance_type = ct;
        std::string first;
        const int c = key_errors(io::validate_config(app), first);
        rep.check(c >= 1, "C_validator_rejects_covariance_type_" + std::to_string(ct),
                  "error=\"" + first + "\"");
    }
    {
        io::AppConfig app = io::make_default_config();
        app.stochastic.covariance_type = 1;
        std::string first;
        const int c = key_errors(io::validate_config(app), first);
        rep.check(c == 0, "C_validator_accepts_covariance_type_1",
                  "covariance_type_errors=" + std::to_string(c));
    }
}

} // namespace

int main() {
    const auto t0 = std::chrono::steady_clock::now();
    TestReport rep;
    try {
        CudaContext ctx(0);

        std::printf("=== stochastic_baseline: A. mode-sampler contract ===\n");
        case_mode_sampler(rep, ctx);

        std::printf("=== stochastic_baseline: B. field fixture (64^3, lambda=2) ===\n");
        case_field(rep, ctx);

        std::printf("=== stochastic_baseline: C. retirement contract ===\n");
        case_retirement(rep, ctx);
    } catch (const std::exception& e) {
        std::printf("[FAIL] unexpected exception: %s\n", e.what());
        rep.overall_pass = false;
    }
    const double secs =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n=== stochastic_baseline_gaussian: %d checks, %s (%.2f s) ===\n", rep.checks,
                rep.overall_pass ? "PASS" : "FAIL", secs);
    return rep.overall_pass ? 0 : 1;
}
