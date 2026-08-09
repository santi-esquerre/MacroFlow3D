/**
 * @file periodic_gaussian_tests.cu
 * @brief SF-18 T02: acceptance tests for the truly-periodic spectral Gaussian
 *        generator (`PeriodicGaussianField.cuh`) plus a 256^3 memory/runtime
 *        record.
 *
 * Standalone ctest-friendly runner (mirrors `apps/run_operator_tests.cu`'s
 * single-executable, printed-checks style) rather than the heavier
 * `tests/streamfunctions/*` case-registry framework: `PeriodicGaussianField`
 * is not part of the streamfunctions namespace/registry and does not share
 * its CPU long-double reference-operator infrastructure, so reusing that
 * registry would import unrelated machinery for no benefit. This file is the
 * sole owner of the SF-18 T02 acceptance record.
 *
 * All acceptance tolerances below were PRESPECIFIED (SF-18 activation
 * bitácora, decision 8) BEFORE this file existed and are implemented
 * VERBATIM; see the increment task description for the authoritative
 * wording. No tolerance, fixture, or seed here may be adjusted to make a
 * failing case pass.
 *
 * Primary statistical fixture (decision 8): 64^3, dx=dy=dz=1 (L=64),
 * corr_length l=8, sigma2=1, seed=12345.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/physics/stochastic/PeriodicGaussianField.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cufft.h>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using namespace macroflow3d;
using namespace macroflow3d::physics;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

// Primary statistical fixture (SF-18 activation decision 8, verbatim).
constexpr int kFixtureN = 64;
constexpr real kFixtureDx = 1.0;
constexpr real kFixtureL = 64.0;
constexpr real kFixtureCorrLength = 8.0;
constexpr real kFixtureSigma2 = 1.0;
constexpr unsigned long long kFixtureSeed = 12345ULL;
constexpr unsigned long long kOtherSeed = 54321ULL;

// ============================================================================
// Bookkeeping
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

std::vector<real> download(DeviceBuffer<real>& buf, size_t n) {
    std::vector<real> host(n);
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(host.data(), buf.data(), n * sizeof(real), cudaMemcpyDeviceToHost));
    return host;
}

double theoretical_S(double sigma2, double l, double Lx, double Ly, double Lz, int mx, int my,
                     int mz) {
    const double kx = 2.0 * kPi * static_cast<double>(mx) / Lx;
    const double ky = 2.0 * kPi * static_cast<double>(my) / Ly;
    const double kz = 2.0 * kPi * static_cast<double>(mz) / Lz;
    const double k2 = kx * kx + ky * ky + kz * kz;
    return sigma2 * std::pow(kPi, 1.5) * l * l * l * std::exp(-k2 * l * l * 0.25) / (Lx * Ly * Lz);
}

inline int wrap_mode(int g, int N) { return (g < N / 2) ? g : g - N; }

// ============================================================================
// Forward (D2Z) FFT helper, mirroring PeriodicGaussianField.cu's Z2D
// argument order/layout so recovered coefficients live at the SAME half-
// spectrum index convention documented in the .cuh file section 6.
// ============================================================================

struct ForwardSpectrum {
    int nx = 0, ny = 0, nz = 0, hx = 0;
    std::vector<double> re, im;  // size hx*ny*nz, index gx + hx*(gy + ny*gz)
};

ForwardSpectrum forward_transform(const CudaContext& ctx, const Grid3D& grid,
                                  DeviceBuffer<real>& y_dev) {
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const int hx = nx / 2 + 1;
    const size_t spec_n = static_cast<size_t>(hx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);

    DeviceBuffer<cufftDoubleComplex> spec_dev(spec_n);

    cufftHandle plan;
    if (cufftPlan3d(&plan, nz, ny, nx, CUFFT_D2Z) != CUFFT_SUCCESS) {
        throw std::runtime_error("cufftPlan3d(D2Z) failed");
    }
    if (cufftSetStream(plan, ctx.cuda_stream()) != CUFFT_SUCCESS) {
        throw std::runtime_error("cufftSetStream failed");
    }
    if (cufftExecD2Z(plan, y_dev.data(), spec_dev.data()) != CUFFT_SUCCESS) {
        throw std::runtime_error("cufftExecD2Z failed");
    }
    ctx.synchronize();
    cufftDestroy(plan);

    std::vector<cufftDoubleComplex> host(spec_n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(host.data(), spec_dev.data(),
                                      spec_n * sizeof(cufftDoubleComplex),
                                      cudaMemcpyDeviceToHost));

    ForwardSpectrum out;
    out.nx = nx;
    out.ny = ny;
    out.nz = nz;
    out.hx = hx;
    out.re.resize(spec_n);
    out.im.resize(spec_n);
    for (size_t i = 0; i < spec_n; ++i) {
        out.re[i] = host[i].x;
        out.im[i] = host[i].y;
    }
    return out;
}

// ============================================================================
// Case: config validation contract (distinct std::invalid_argument messages)
// ============================================================================

bool case_config_validation(TestReport& rep) {
    const Grid3D good_grid(8, 8, 8, 1.0, 1.0, 1.0);
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

    expect_invalid("config_validation_sigma2_negative", [&] {
        PeriodicGaussianFieldConfig cfg;
        cfg.sigma2 = -1.0;
        cfg.corr_length = 1.0;
        cfg.validate(good_grid);
    });
    expect_invalid("config_validation_corr_length_zero", [&] {
        PeriodicGaussianFieldConfig cfg;
        cfg.sigma2 = 1.0;
        cfg.corr_length = 0.0;
        cfg.validate(good_grid);
    });
    expect_invalid("config_validation_odd_extent", [&] {
        PeriodicGaussianFieldConfig cfg;
        cfg.sigma2 = 1.0;
        cfg.corr_length = 1.0;
        const Grid3D odd_grid(9, 8, 8, 1.0, 1.0, 1.0);
        cfg.validate(odd_grid);
    });
    expect_invalid("config_validation_bad_spacing", [&] {
        PeriodicGaussianFieldConfig cfg;
        cfg.sigma2 = 1.0;
        cfg.corr_length = 1.0;
        const Grid3D bad_grid(8, 8, 8, 0.0, 1.0, 1.0);
        cfg.validate(bad_grid);
    });
    expect_invalid("config_validation_wrong_y_out_size", [&] {
        CudaContext ctx(0);
        PeriodicGaussianFieldConfig cfg;
        cfg.sigma2 = 1.0;
        cfg.corr_length = 1.0;
        cfg.seed = 1ULL;
        DeviceBuffer<real> y(good_grid.num_cells() - 1);
        PeriodicGaussianFieldWorkspace ws;
        generate_periodic_gaussian_field(ctx, good_grid, cfg, y.span(), ws);
    });

    rep.check(messages.size() == 5, "config_validation_five_distinct_messages",
              "count=" + std::to_string(messages.size()));
    return rep.overall_pass;
}

// ============================================================================
// Case (a): bitwise reproducibility
// ============================================================================

void case_reproducibility(TestReport& rep) {
    CudaContext ctx(0);
    const Grid3D grid(kFixtureN, kFixtureN, kFixtureN, kFixtureDx, kFixtureDx, kFixtureDx);
    const size_t n = grid.num_cells();

    PeriodicGaussianFieldConfig cfg;
    cfg.sigma2 = kFixtureSigma2;
    cfg.corr_length = kFixtureCorrLength;
    cfg.seed = kFixtureSeed;
    cfg.normalize_variance = true;

    DeviceBuffer<real> y1(n), y2(n), y3(n);
    PeriodicGaussianFieldWorkspace ws1, ws2, ws3;

    generate_periodic_gaussian_field(ctx, grid, cfg, y1.span(), ws1);
    generate_periodic_gaussian_field(ctx, grid, cfg, y2.span(), ws2);

    PeriodicGaussianFieldConfig cfg_other = cfg;
    cfg_other.seed = kOtherSeed;
    generate_periodic_gaussian_field(ctx, grid, cfg_other, y3.span(), ws3);

    const auto h1 = download(y1, n);
    const auto h2 = download(y2, n);
    const auto h3 = download(y3, n);

    const bool bitwise_same = std::memcmp(h1.data(), h2.data(), n * sizeof(real)) == 0;
    const bool bitwise_diff = std::memcmp(h1.data(), h3.data(), n * sizeof(real)) != 0;

    rep.check(bitwise_same, "reproducibility_same_seed_bitwise_identical");
    rep.check(bitwise_diff, "reproducibility_different_seed_not_bitwise_identical");
}

// ============================================================================
// Case (b): torus periodicity (seam vs. interior differences)
// ============================================================================

struct SeamStats {
    double interior_d1_rms = 0.0, seam_d1_rms = 0.0;
    double interior_d2_rms = 0.0, seam_d2_rms = 0.0;
    double max_interior_d1 = 0.0, max_seam_d1 = 0.0;
};

std::vector<real> extract_line(const Grid3D& grid, const std::vector<real>& y, int axis, int a,
                               int b) {
    std::vector<real> line;
    if (axis == 0) {
        line.resize(static_cast<size_t>(grid.nx));
        for (int i = 0; i < grid.nx; ++i) line[static_cast<size_t>(i)] = y[grid.idx(i, a, b)];
    } else if (axis == 1) {
        line.resize(static_cast<size_t>(grid.ny));
        for (int j = 0; j < grid.ny; ++j) line[static_cast<size_t>(j)] = y[grid.idx(a, j, b)];
    } else {
        line.resize(static_cast<size_t>(grid.nz));
        for (int k = 0; k < grid.nz; ++k) line[static_cast<size_t>(k)] = y[grid.idx(a, b, k)];
    }
    return line;
}

SeamStats axis_seam_stats(const Grid3D& grid, const std::vector<real>& y, int axis) {
    const int N = axis == 0 ? grid.nx : axis == 1 ? grid.ny : grid.nz;
    const int outer_a = axis == 0 ? grid.ny : grid.nx;
    const int outer_b = axis == 2 ? grid.ny : grid.nz;

    double interior_d1_sq = 0.0, seam_d1_sq = 0.0;
    double interior_d2_sq = 0.0, seam_d2_sq = 0.0;
    size_t interior_d1_n = 0, seam_d1_n = 0, interior_d2_n = 0, seam_d2_n = 0;
    double max_interior_d1 = 0.0, max_seam_d1 = 0.0;

    for (int a = 0; a < outer_a; ++a) {
        for (int b = 0; b < outer_b; ++b) {
            const auto line = extract_line(grid, y, axis, a, b);

            // First differences: interior i=0..N-2; seam is the wrap i=N-1.
            for (int i = 0; i + 1 < N; ++i) {
                const double d = static_cast<double>(line[static_cast<size_t>(i + 1)]) -
                                 static_cast<double>(line[static_cast<size_t>(i)]);
                interior_d1_sq += d * d;
                max_interior_d1 = std::max(max_interior_d1, std::fabs(d));
                ++interior_d1_n;
            }
            {
                const double seam = static_cast<double>(line[0]) -
                                    static_cast<double>(line[static_cast<size_t>(N - 1)]);
                seam_d1_sq += seam * seam;
                max_seam_d1 = std::max(max_seam_d1, std::fabs(seam));
                ++seam_d1_n;
            }

            // Second differences: interior i=1..N-2; seam is the two stencils
            // that read across the wrap boundary.
            for (int i = 1; i + 1 < N; ++i) {
                const double d = static_cast<double>(line[static_cast<size_t>(i + 1)]) -
                                 2.0 * static_cast<double>(line[static_cast<size_t>(i)]) +
                                 static_cast<double>(line[static_cast<size_t>(i - 1)]);
                interior_d2_sq += d * d;
                ++interior_d2_n;
            }
            {
                const double d0 = static_cast<double>(line[1]) -
                                  2.0 * static_cast<double>(line[0]) +
                                  static_cast<double>(line[static_cast<size_t>(N - 1)]);
                const double dN = static_cast<double>(line[0]) -
                                  2.0 * static_cast<double>(line[static_cast<size_t>(N - 1)]) +
                                  static_cast<double>(line[static_cast<size_t>(N - 2)]);
                seam_d2_sq += d0 * d0 + dN * dN;
                seam_d2_n += 2;
            }
        }
    }

    SeamStats stats;
    stats.interior_d1_rms = std::sqrt(interior_d1_sq / static_cast<double>(interior_d1_n));
    stats.seam_d1_rms = std::sqrt(seam_d1_sq / static_cast<double>(seam_d1_n));
    stats.interior_d2_rms = std::sqrt(interior_d2_sq / static_cast<double>(interior_d2_n));
    stats.seam_d2_rms = std::sqrt(seam_d2_sq / static_cast<double>(seam_d2_n));
    stats.max_interior_d1 = max_interior_d1;
    stats.max_seam_d1 = max_seam_d1;
    return stats;
}

void case_wrap_periodicity(TestReport& rep) {
    CudaContext ctx(0);
    const Grid3D grid(kFixtureN, kFixtureN, kFixtureN, kFixtureDx, kFixtureDx, kFixtureDx);
    const size_t n = grid.num_cells();

    PeriodicGaussianFieldConfig cfg;
    cfg.sigma2 = kFixtureSigma2;
    cfg.corr_length = kFixtureCorrLength;
    cfg.seed = kFixtureSeed;
    cfg.normalize_variance = true;

    DeviceBuffer<real> y(n);
    PeriodicGaussianFieldWorkspace ws;
    generate_periodic_gaussian_field(ctx, grid, cfg, y.span(), ws);
    const auto h = download(y, n);

    const char* axis_name[3] = {"x", "y", "z"};
    for (int axis = 0; axis < 3; ++axis) {
        const auto stats = axis_seam_stats(grid, h, axis);
        const double d1_ratio = stats.seam_d1_rms / stats.interior_d1_rms;
        const double d2_ratio = stats.seam_d2_rms / stats.interior_d2_rms;
        const double max_ratio = stats.max_seam_d1 / stats.max_interior_d1;

        std::printf("  wrap axis=%s d1_ratio=%.6f d2_ratio=%.6f max_seam=%.6g max_interior=%.6g "
                    "max_ratio=%.6f\n",
                    axis_name[axis], d1_ratio, d2_ratio, stats.max_seam_d1, stats.max_interior_d1,
                    max_ratio);

        rep.check(d1_ratio >= 0.7 && d1_ratio <= 1.4,
                  std::string("wrap_d1_ratio_axis_") + axis_name[axis],
                  "ratio=" + std::to_string(d1_ratio));
        rep.check(d2_ratio >= 0.7 && d2_ratio <= 1.4,
                  std::string("wrap_d2_ratio_axis_") + axis_name[axis],
                  "ratio=" + std::to_string(d2_ratio));
        rep.check(stats.max_seam_d1 <= 5.0 * stats.max_interior_d1,
                  std::string("wrap_max_seam_bound_axis_") + axis_name[axis],
                  "max_seam=" + std::to_string(stats.max_seam_d1) +
                      " 5*max_interior=" + std::to_string(5.0 * stats.max_interior_d1));
    }
}

// ============================================================================
// Case (c): mean / variance
// ============================================================================

void case_mean_variance(TestReport& rep) {
    CudaContext ctx(0);
    const Grid3D grid(kFixtureN, kFixtureN, kFixtureN, kFixtureDx, kFixtureDx, kFixtureDx);
    const size_t n = grid.num_cells();
    const double sigma = std::sqrt(kFixtureSigma2);

    // --- normalized run: sample mean/variance computed independently on host ---
    {
        PeriodicGaussianFieldConfig cfg;
        cfg.sigma2 = kFixtureSigma2;
        cfg.corr_length = kFixtureCorrLength;
        cfg.seed = kFixtureSeed;
        cfg.normalize_variance = true;

        DeviceBuffer<real> y(n);
        PeriodicGaussianFieldWorkspace ws;
        generate_periodic_gaussian_field(ctx, grid, cfg, y.span(), ws);
        const auto h = download(y, n);

        double sum = 0.0;
        for (real v : h) sum += static_cast<double>(v);
        const double mean = sum / static_cast<double>(n);
        double sumsq = 0.0;
        for (real v : h) {
            const double d = static_cast<double>(v) - mean;
            sumsq += d * d;
        }
        const double variance = sumsq / static_cast<double>(n);

        std::printf("  mean_variance normalized: host_mean=%.6e host_variance=%.15e\n", mean,
                    variance);

        rep.check(std::fabs(mean) <= 1.0e-10 * sigma, "mean_variance_normalized_mean_bound",
                  "|mean|=" + std::to_string(std::fabs(mean)));
        rep.check(std::fabs(variance / kFixtureSigma2 - 1.0) <= 1.0e-12,
                  "mean_variance_normalized_variance_bound",
                  "|var/sigma2-1|=" + std::to_string(std::fabs(variance / kFixtureSigma2 - 1.0)));
    }

    // --- raw (unnormalized) run: report vs. host-computed raw variance, and
    //     comparison against the exact theoretical E_var / N_eff ---
    {
        PeriodicGaussianFieldConfig cfg;
        cfg.sigma2 = kFixtureSigma2;
        cfg.corr_length = kFixtureCorrLength;
        cfg.seed = kFixtureSeed;
        cfg.normalize_variance = false;

        DeviceBuffer<real> y(n);
        PeriodicGaussianFieldWorkspace ws;
        const auto report = generate_periodic_gaussian_field(ctx, grid, cfg, y.span(), ws);
        const auto h = download(y, n);

        double sum = 0.0;
        for (real v : h) sum += static_cast<double>(v);
        const double host_mean = sum / static_cast<double>(n);
        double sumsq = 0.0;
        for (real v : h) {
            const double d = static_cast<double>(v) - host_mean;
            sumsq += d * d;
        }
        const double host_raw_variance = sumsq / static_cast<double>(n);

        const double agree =
            std::fabs(host_raw_variance / static_cast<double>(report.raw_variance) - 1.0);
        std::printf("  mean_variance raw: report.raw_variance=%.15e host_raw_variance=%.15e "
                    "agree=%.3e\n",
                    static_cast<double>(report.raw_variance), host_raw_variance, agree);
        rep.check(agree <= 1.0e-9, "mean_variance_raw_report_matches_host",
                  "relative_diff=" + std::to_string(agree));

        // Exact theoretical E_var = sum_m S_m and N_eff = (sum S_m)^2 / sum
        // S_m^2 over the FULL reciprocal-lattice active-mode set
        // (|m_d| <= N_d/2 - 1 on every axis, m != 0). See file header /
        // increment task for the derivation of why Var[Y(x)] equals this
        // full-lattice sum (not the half-spectrum storage count).
        const int half_extent = kFixtureN / 2 - 1;  // 31 for N=64
        const double Lx = grid.Lx(), Ly = grid.Ly(), Lz = grid.Lz();

        double sum_S = 0.0, sum_S2 = 0.0;
        for (int mx = -half_extent; mx <= half_extent; ++mx) {
            for (int my = -half_extent; my <= half_extent; ++my) {
                for (int mz = -half_extent; mz <= half_extent; ++mz) {
                    if (mx == 0 && my == 0 && mz == 0) continue;
                    const double S = theoretical_S(kFixtureSigma2, kFixtureCorrLength, Lx, Ly, Lz,
                                                   mx, my, mz);
                    sum_S += S;
                    sum_S2 += S * S;
                }
            }
        }
        const double E_var = sum_S;
        const double N_eff = (sum_S * sum_S) / sum_S2;
        const double tol = 5.0 * std::sqrt(2.0 / N_eff);
        const double dev = std::fabs(static_cast<double>(report.raw_variance) / E_var - 1.0);

        std::printf("  mean_variance raw: E_var=%.15e N_eff=%.6f tol=%.6e observed_dev=%.6e\n",
                    E_var, N_eff, tol, dev);
        rep.check(dev <= tol, "mean_variance_raw_variance_matches_E_var",
                  "dev=" + std::to_string(dev) + " tol=" + std::to_string(tol));
    }
}

// ============================================================================
// Case (d): spectrum shape (radially binned mode powers vs. theoretical S_m)
// ============================================================================

void case_spectrum_shape(TestReport& rep) {
    CudaContext ctx(0);
    const Grid3D grid(kFixtureN, kFixtureN, kFixtureN, kFixtureDx, kFixtureDx, kFixtureDx);
    const size_t n = grid.num_cells();
    const double Lx = grid.Lx(), Ly = grid.Ly(), Lz = grid.Lz();
    const double N_total = static_cast<double>(n);

    PeriodicGaussianFieldConfig cfg;
    cfg.sigma2 = kFixtureSigma2;
    cfg.corr_length = kFixtureCorrLength;
    cfg.seed = kFixtureSeed;
    cfg.normalize_variance = false;  // undoing the truncation-dependent rescale (decision 8)

    DeviceBuffer<real> y(n);
    PeriodicGaussianFieldWorkspace ws;
    generate_periodic_gaussian_field(ctx, grid, cfg, y.span(), ws);

    const auto spec = forward_transform(ctx, grid, y);

    // Fixed, documented binning: shell width = one fundamental wavenumber
    // 2*pi/L, over active modes only.
    const double dk = 2.0 * kPi / kFixtureL;

    struct Bin {
        double sum_power = 0.0;
        double sum_S = 0.0;
        size_t count = 0;
    };
    std::map<int, Bin> bins;
    double S_max = 0.0;

    for (int gz = 0; gz < spec.nz; ++gz) {
        const int mz = wrap_mode(gz, spec.nz);
        for (int gy = 0; gy < spec.ny; ++gy) {
            const int my = wrap_mode(gy, spec.ny);
            for (int gx = 0; gx < spec.hx; ++gx) {
                const int mx = gx;
                const bool nyquist = (mx == spec.nx / 2) || (gy == spec.ny / 2) ||
                                     (gz == spec.nz / 2);
                const bool zero_mode = (mx == 0 && my == 0 && mz == 0);
                if (nyquist || zero_mode) continue;

                const size_t idx = static_cast<size_t>(gx) +
                                   static_cast<size_t>(spec.hx) *
                                       (static_cast<size_t>(gy) +
                                        static_cast<size_t>(spec.ny) * static_cast<size_t>(gz));
                const double re = spec.re[idx] / N_total;
                const double im = spec.im[idx] / N_total;
                const double power = re * re + im * im;

                const double S = theoretical_S(kFixtureSigma2, kFixtureCorrLength, Lx, Ly, Lz, mx,
                                               my, mz);
                S_max = std::max(S_max, S);

                const double kx = 2.0 * kPi * mx / Lx;
                const double ky = 2.0 * kPi * my / Ly;
                const double kz = 2.0 * kPi * mz / Lz;
                const double kmag = std::sqrt(kx * kx + ky * ky + kz * kz);
                const int bin_index = static_cast<int>(std::floor(kmag / dk));

                auto& b = bins[bin_index];
                b.sum_power += power;
                b.sum_S += S;
                ++b.count;
            }
        }
    }

    size_t evaluated_bins = 0;
    double worst_dev = 0.0;
    int worst_bin = -1;
    bool all_pass = true;
    for (const auto& [bin_index, b] : bins) {
        if (b.count < 50) continue;
        const double S_bin_avg = b.sum_S / static_cast<double>(b.count);
        if (S_bin_avg < 1.0e-6 * S_max) continue;
        const double P_bin = b.sum_power / static_cast<double>(b.count);
        const double tol = 5.0 * std::sqrt(2.0 / static_cast<double>(b.count));
        const double dev = std::fabs(P_bin / S_bin_avg - 1.0);
        ++evaluated_bins;
        if (dev > worst_dev) {
            worst_dev = dev;
            worst_bin = bin_index;
        }
        if (dev > tol) {
            all_pass = false;
            std::printf("  spectrum_shape FAIL bin=%d n=%zu P_bin=%.6e S_bin=%.6e dev=%.6e "
                        "tol=%.6e\n",
                        bin_index, b.count, P_bin, S_bin_avg, dev, tol);
        }
    }

    std::printf("  spectrum_shape evaluated_bins=%zu worst_bin=%d worst_dev=%.6e S_max=%.6e\n",
                evaluated_bins, worst_bin, worst_dev, S_max);
    rep.check(evaluated_bins > 0, "spectrum_shape_has_evaluated_bins",
              "count=" + std::to_string(evaluated_bins));
    rep.check(all_pass, "spectrum_shape_all_bins_within_tolerance",
              "worst_dev=" + std::to_string(worst_dev));
}

// ============================================================================
// Case (e): refinement (spectral-truncation coefficient identity)
// ============================================================================

void case_refinement(TestReport& rep) {
    CudaContext ctx(0);

    const Grid3D coarse_grid(32, 32, 32, 2.0, 2.0, 2.0);  // L=64, h=2
    const Grid3D fine_grid(64, 64, 64, 1.0, 1.0, 1.0);    // L=64, h=1

    PeriodicGaussianFieldConfig cfg;
    cfg.sigma2 = kFixtureSigma2;
    cfg.corr_length = kFixtureCorrLength;
    cfg.seed = kFixtureSeed;
    cfg.normalize_variance = false;

    DeviceBuffer<real> y_coarse(coarse_grid.num_cells());
    DeviceBuffer<real> y_fine(fine_grid.num_cells());
    PeriodicGaussianFieldWorkspace ws_coarse, ws_fine;
    generate_periodic_gaussian_field(ctx, coarse_grid, cfg, y_coarse.span(), ws_coarse);
    generate_periodic_gaussian_field(ctx, fine_grid, cfg, y_fine.span(), ws_fine);

    const auto spec_coarse = forward_transform(ctx, coarse_grid, y_coarse);
    const auto spec_fine = forward_transform(ctx, fine_grid, y_fine);

    auto untwisted = [&](const ForwardSpectrum& spec, const Grid3D& grid, int mx, int my, int mz) {
        const int gy = my >= 0 ? my : my + spec.ny;
        const int gz = mz >= 0 ? mz : mz + spec.nz;
        const size_t idx = static_cast<size_t>(mx) +
                           static_cast<size_t>(spec.hx) *
                               (static_cast<size_t>(gy) +
                                static_cast<size_t>(spec.ny) * static_cast<size_t>(gz));
        const double N_total = static_cast<double>(grid.num_cells());
        const double re = spec.re[idx] / N_total;
        const double im = spec.im[idx] / N_total;

        const double kx = 2.0 * kPi * mx / static_cast<double>(grid.Lx());
        const double ky = 2.0 * kPi * my / static_cast<double>(grid.Ly());
        const double kz = 2.0 * kPi * mz / static_cast<double>(grid.Lz());
        const double phase =
            0.5 * (kx * static_cast<double>(grid.dx) + ky * static_cast<double>(grid.dy) +
                  kz * static_cast<double>(grid.dz));
        // Undo the half-cell twist with THIS grid's own h (untwist = multiply
        // by exp(-i*phase)).
        const double cw = std::cos(phase);
        const double sw = std::sin(phase);
        const double ure = re * cw + im * sw;
        const double uim = im * cw - re * sw;
        return std::pair<double, double>{ure, uim};
    };

    constexpr int kCommon = 15;  // 32/2 - 1
    constexpr double kFloor = 1.0e-300;
    double max_rel_error = 0.0;
    int worst_mx = 0, worst_my = 0, worst_mz = 0;
    size_t compared = 0, floor_excluded = 0;

    // Non-gating diagnostic: max relative error restricted to modes whose
    // canonical (untwisted) magnitude is resolvable above the double-
    // precision noise floor of an O(1)-magnitude field's cuFFT round trip
    // (~1e-8 is a generous margin above the ~1e-13..1e-14 floor observed
    // empirically). This does NOT alter the prespecified 1e-12/1e-300
    // gating check below; it exists purely to distinguish "generator bug"
    // from "true coefficient buried in floating-point roundoff at this
    // fixture's extreme spectral-decay corner" for the increment report.
    constexpr double kResolvedMagnitude = 1.0e-8;
    double max_rel_error_resolved = 0.0;
    size_t resolved_count = 0, unresolved_count = 0;
    constexpr int kBucketCount = 6;
    constexpr double kBucketThresholds[kBucketCount] = {1.0e-2,  1.0e-3,  1.0e-4,
                                                        1.0e-6, 1.0e-8, 1.0e-10};
    double bucket_max_rel[kBucketCount] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    size_t bucket_count[kBucketCount] = {0, 0, 0, 0, 0, 0};

    for (int mx = 0; mx <= kCommon; ++mx) {
        for (int my = -kCommon; my <= kCommon; ++my) {
            for (int mz = -kCommon; mz <= kCommon; ++mz) {
                if (mx == 0 && my == 0 && mz == 0) continue;
                const auto c = untwisted(spec_coarse, coarse_grid, mx, my, mz);
                const auto f = untwisted(spec_fine, fine_grid, mx, my, mz);
                const double c_mag = std::sqrt(c.first * c.first + c.second * c.second);
                if (c_mag < kFloor) {
                    ++floor_excluded;
                    continue;
                }
                const double diff_re = c.first - f.first;
                const double diff_im = c.second - f.second;
                const double diff_mag = std::sqrt(diff_re * diff_re + diff_im * diff_im);
                const double rel_error = diff_mag / c_mag;
                ++compared;
                if (rel_error > max_rel_error) {
                    max_rel_error = rel_error;
                    worst_mx = mx;
                    worst_my = my;
                    worst_mz = mz;
                }
                if (c_mag >= kResolvedMagnitude) {
                    ++resolved_count;
                    max_rel_error_resolved = std::max(max_rel_error_resolved, rel_error);
                } else {
                    ++unresolved_count;
                }
                for (int b = 0; b < kBucketCount; ++b) {
                    if (c_mag >= kBucketThresholds[b]) {
                        bucket_max_rel[b] = std::max(bucket_max_rel[b], rel_error);
                        ++bucket_count[b];
                    }
                }
            }
        }
    }

    std::printf("  refinement compared=%zu floor_excluded=%zu max_rel_error=%.6e worst_mode=(%d,"
                "%d,%d)\n",
                compared, floor_excluded, max_rel_error, worst_mx, worst_my, worst_mz);
    std::printf("  refinement DIAGNOSTIC (non-gating): resolved(|coef|>=%.1e)=%zu "
                "unresolved=%zu max_rel_error_resolved=%.6e\n",
                kResolvedMagnitude, resolved_count, unresolved_count, max_rel_error_resolved);
    for (int b = 0; b < kBucketCount; ++b) {
        std::printf("  refinement DIAGNOSTIC (non-gating): |coef|>=%.1e count=%zu "
                    "max_rel_error=%.6e\n",
                    kBucketThresholds[b], bucket_count[b], bucket_max_rel[b]);
    }
    rep.check(compared > 0, "refinement_has_compared_modes", "count=" + std::to_string(compared));
    rep.check(max_rel_error <= 1.0e-12, "refinement_common_mode_coefficients_match",
              "max_rel_error=" + std::to_string(max_rel_error));
}

// ============================================================================
// Case (f): 256^3 memory / runtime record (no numeric gate beyond completion)
// ============================================================================

void case_256_cubed_record(TestReport& rep) {
    CudaContext ctx(0);
    const Grid3D grid(256, 256, 256, 1.0, 1.0, 1.0);  // L=256
    const size_t n = grid.num_cells();

    PeriodicGaussianFieldConfig cfg;
    cfg.sigma2 = 1.0;
    cfg.corr_length = 16.0;
    cfg.seed = kFixtureSeed;
    cfg.normalize_variance = true;

    size_t free_before = 0, total_before = 0;
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free_before, &total_before));

    DeviceBuffer<real> y(n);
    PeriodicGaussianFieldWorkspace ws;

    const auto t0 = std::chrono::steady_clock::now();
    const auto report = generate_periodic_gaussian_field(ctx, grid, cfg, y.span(), ws);
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(t1 - t0).count();

    size_t free_after = 0, total_after = 0;
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free_after, &total_after));

    std::printf("  256^3 record: wall_time_s=%.6f\n", wall_seconds);
    std::printf("  256^3 record: raw_mean=%.6e raw_variance=%.6e applied_scale=%.6e "
                "final_variance=%.6e active_mode_count=%zu\n",
                static_cast<double>(report.raw_mean), static_cast<double>(report.raw_variance),
                static_cast<double>(report.applied_scale),
                static_cast<double>(report.final_variance), report.active_mode_count);
    std::printf("  256^3 record: spectrum_bytes=%zu cufft_work_area_bytes=%zu "
                "reduction_scratch_bytes=%zu total_device_bytes=%zu\n",
                report.spectrum_bytes, report.cufft_work_area_bytes,
                report.reduction_scratch_bytes, report.total_device_bytes);
    std::printf("  256^3 record: cudaMemGetInfo before: free=%zu total=%zu ; after: free=%zu "
                "total=%zu ; delta_used=%zd bytes\n",
                free_before, total_before, free_after, total_after,
                static_cast<std::ptrdiff_t>(free_before) - static_cast<std::ptrdiff_t>(free_after));

    const bool finite = std::isfinite(static_cast<double>(report.raw_mean)) &&
                        std::isfinite(static_cast<double>(report.raw_variance)) &&
                        std::isfinite(static_cast<double>(report.final_variance));
    rep.check(finite, "record_256_cubed_completed_with_finite_stats");
}

}  // namespace

int main() {
    TestReport rep;

    std::printf("=== SF-18 T02: config validation contract ===\n");
    case_config_validation(rep);

    std::printf("=== SF-18 T02: (a) bitwise reproducibility ===\n");
    case_reproducibility(rep);

    std::printf("=== SF-18 T02: (b) torus periodicity (wrap) ===\n");
    case_wrap_periodicity(rep);

    std::printf("=== SF-18 T02: (c) mean / variance ===\n");
    case_mean_variance(rep);

    std::printf("=== SF-18 T02: (d) spectrum shape ===\n");
    case_spectrum_shape(rep);

    std::printf("=== SF-18 T02: (e) refinement ===\n");
    case_refinement(rep);

    std::printf("=== SF-18 T02: (f) 256^3 memory/runtime record ===\n");
    case_256_cubed_record(rep);

    std::printf("\n=== periodic_gaussian: %d checks, %s ===\n", rep.checks,
                rep.overall_pass ? "PASS" : "FAIL");
    return rep.overall_pass ? 0 : 1;
}
