#include "streamfunction_operator_test_cases.hpp"

#include "src/core/BCSpec.hpp"
#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/numerics/constraints/MeanZeroProjector.cuh"
#include "src/numerics/operators/lester_positive_diffusion_operator.cuh"
#include "src/physics/streamfunctions/JacobianVectorProduct.cuh"
#include "src/physics/streamfunctions/NonlinearSources.cuh"
#include "src/physics/streamfunctions/ResidualEvaluator.cuh"
#include "src/physics/streamfunctions/StreamfunctionSolver.cuh"
#include "src/physics/streamfunctions/StreamfunctionTypes.hpp"
#include "src/physics/streamfunctions/StreamfunctionWorkspace.cuh"
#include "src/physics/streamfunctions/affine_gauge.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// SF-22 T02: activation-decision (D6, D7) GPU acceptance cases for the SF-22
// T01 matrix-free Jacobian-vector product (`JvpWorkspace`,
// `JacobianVectorProduct.cuh/.cu`). Every fixture, threshold, and seed below
// was PRESPECIFIED in the SF-22 activation bitácora before this file was
// written and is implemented VERBATIM; no value is adjusted after seeing a
// result. This file never touches `src/**`.
//
// Domain-length lesson (SF-21 audit finding T02r-F1, also documented in
// heterogeneity_continuation_gpu_cases.cu): the trig conductivity fixture
// below is sampled on a grid of DOMAIN LENGTH 1 (`Grid3D(n,n,n, 1.0/n, ...)`),
// never `dx=1`, so the sine factors do not degenerate to zero at every cell
// center.

namespace macroflow3d::streamfunctions::test {
namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;

// ---------------------------------------------------------------------------
// Grid / problem helpers (mirrors streamfunction_anderson_gpu_cases.cu).
// ---------------------------------------------------------------------------

[[nodiscard]] std::size_t compact_mac_u_size(const Grid3D& grid) {
    return static_cast<std::size_t>(grid.nx + 1) * static_cast<std::size_t>(grid.ny) *
           static_cast<std::size_t>(grid.nz);
}
[[nodiscard]] std::size_t compact_mac_v_size(const Grid3D& grid) {
    return static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny + 1) *
           static_cast<std::size_t>(grid.nz);
}
[[nodiscard]] std::size_t compact_mac_w_size(const Grid3D& grid) {
    return static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny) *
           static_cast<std::size_t>(grid.nz + 1);
}

[[nodiscard]] Grid3D isotropic_grid(int n) {
    const real h = real{1} / static_cast<real>(n);
    return Grid3D{n, n, n, h, h, h};
}

[[nodiscard]] BCSpec triply_periodic() {
    BCSpec bc;
    bc.xmin = BCFace(BCType::Periodic, real{0});
    bc.xmax = BCFace(BCType::Periodic, real{0});
    bc.ymin = BCFace(BCType::Periodic, real{0});
    bc.ymax = BCFace(BCType::Periodic, real{0});
    bc.zmin = BCFace(BCType::Periodic, real{0});
    bc.zmax = BCFace(BCType::Periodic, real{0});
    return bc;
}

[[nodiscard]] std::vector<real> download(const DeviceSpan<const real>& span) {
    std::vector<real> host(span.size());
    if (!host.empty()) {
        MACROFLOW3D_CUDA_CHECK(
            cudaMemcpy(host.data(), span.data(), host.size() * sizeof(real), cudaMemcpyDeviceToHost));
    }
    return host;
}

[[nodiscard]] DeviceBuffer<real> upload(const std::vector<real>& host) {
    DeviceBuffer<real> buffer(host.size());
    if (!host.empty()) {
        MACROFLOW3D_CUDA_CHECK(
            cudaMemcpy(buffer.data(), host.data(), host.size() * sizeof(real), cudaMemcpyHostToDevice));
    }
    return buffer;
}

[[nodiscard]] bool bitwise_equal(const std::vector<real>& a, const std::vector<real>& b) {
    if (a.size() != b.size()) return false;
    if (a.empty()) return true;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(real)) == 0;
}

// Two nrm2-derived RMS terms combined exactly as JacobianVectorProduct.cuh's
// D1 weighted coupled norm: ||(a,b)||_w = sqrt((RMS(a)/g1)^2 + (RMS(b)/g2)^2).
[[nodiscard]] double weighted_norm_host(const std::vector<real>& a, const std::vector<real>& b,
                                       double g1, double g2) {
    long double sum_a = 0.0L, sum_b = 0.0L;
    for (const real v : a) sum_a += static_cast<long double>(v) * v;
    for (const real v : b) sum_b += static_cast<long double>(v) * v;
    const double rms_a = std::sqrt(static_cast<double>(sum_a / static_cast<long double>(a.size())));
    const double rms_b = std::sqrt(static_cast<double>(sum_b / static_cast<long double>(b.size())));
    const double ta = rms_a / g1;
    const double tb = rms_b / g2;
    return std::sqrt(ta * ta + tb * tb);
}

[[nodiscard]] double mean_abs_host(const std::vector<real>& values) {
    long double sum = 0.0L;
    for (const real v : values) sum += static_cast<long double>(v);
    return std::abs(static_cast<double>(sum / static_cast<long double>(values.size())));
}

// (d) gauge/mean-zero bound. F is discretely mean-zero by construction
// (ResidualEvaluator.cuh), so mean(F(base)) and mean(F(perturbed)) are both
// O(reduction-roundoff), not exact zero on device. Jv = (F(pert)-F(base))/
// delta AMPLIFIES that roundoff by 1/delta, so the mean-zero contract for Jv
// must scale like the projector's own absolute-vs-scale bound
// (mean_zero_projector_gpu_cases.cu: 100*eps*max(RMS(x),1)) divided by delta
// -- documented explicitly here rather than reusing an arbitrary constant.
constexpr double kJvMeanFactor = 1.0e4;
[[nodiscard]] double jv_mean_limit(const std::vector<real>& jv_component, double delta) {
    const double scale = std::max(std::sqrt([&] {
                                       long double sum = 0.0L;
                                       for (const real v : jv_component)
                                           sum += static_cast<long double>(v) * v;
                                       return static_cast<double>(
                                           sum / static_cast<long double>(jv_component.size()));
                                   }()),
                                   1.0);
    return kJvMeanFactor * std::numeric_limits<double>::epsilon() * scale / delta;
}

// splitmix64, fixed seed = 0x5EED2026081200ULL (documented, never re-seeded
// after the first observed result). Produces uniform doubles in [-1, 1].
[[nodiscard]] std::uint64_t splitmix64_next(std::uint64_t& state) {
    std::uint64_t z = (state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
[[nodiscard]] double splitmix64_uniform_pm1(std::uint64_t& state) {
    const std::uint64_t bits = splitmix64_next(state) >> 11; // top 53 bits
    const double u01 = static_cast<double>(bits) / static_cast<double>(1ULL << 53);
    return 2.0 * u01 - 1.0;
}

// ---------------------------------------------------------------------------
// Trig conductivity / q field, uniform Darcy velocity (v_rms=1), gauge.
// ---------------------------------------------------------------------------

[[nodiscard]] std::vector<real> trig_q_host(const Grid3D& grid, double amplitude) {
    std::vector<real> q(grid.num_cells());
    for (int iz = 0; iz < grid.nz; ++iz) {
        const double z = (iz + 0.5) * static_cast<double>(grid.dz);
        for (int iy = 0; iy < grid.ny; ++iy) {
            const double y = (iy + 0.5) * static_cast<double>(grid.dy);
            for (int ix = 0; ix < grid.nx; ++ix) {
                const double x = (ix + 0.5) * static_cast<double>(grid.dx);
                const double log_k =
                    amplitude * std::sin(2.0 * kPi * x) * std::sin(2.0 * kPi * y) * std::sin(2.0 * kPi * z);
                // q = 1/K = exp(-log_k).
                q[grid.idx(ix, iy, iz)] = static_cast<real>(std::exp(-log_k));
            }
        }
    }
    return q;
}

[[nodiscard]] std::vector<real> trig_k_host(const Grid3D& grid, double amplitude) {
    std::vector<real> k(grid.num_cells());
    for (int iz = 0; iz < grid.nz; ++iz) {
        const double z = (iz + 0.5) * static_cast<double>(grid.dz);
        for (int iy = 0; iy < grid.ny; ++iy) {
            const double y = (iy + 0.5) * static_cast<double>(grid.dy);
            for (int ix = 0; ix < grid.nx; ++ix) {
                const double x = (ix + 0.5) * static_cast<double>(grid.dx);
                const double log_k =
                    amplitude * std::sin(2.0 * kPi * x) * std::sin(2.0 * kPi * y) * std::sin(2.0 * kPi * z);
                k[grid.idx(ix, iy, iz)] = static_cast<real>(std::exp(log_k));
            }
        }
    }
    return k;
}

struct DarcyBuffers {
    DeviceBuffer<real> u, v, w;
    explicit DarcyBuffers(const Grid3D& grid)
        : u(upload(std::vector<real>(compact_mac_u_size(grid), real{1}))),
          v(upload(std::vector<real>(compact_mac_v_size(grid), real{0}))),
          w(upload(std::vector<real>(compact_mac_w_size(grid), real{0}))) {}
};

[[nodiscard]] NonlinearSourceConfig source_config_v_rms_1() {
    NonlinearSourceConfig cfg;
    cfg.v_rms = real{1}; // uniform Darcy v=(1,0,0) -> v_rms=1 exactly.
    return cfg;
}

// ---------------------------------------------------------------------------
// Fluctuation-pair host fixtures (base states and directions).
// ---------------------------------------------------------------------------

struct FluctuationPair {
    std::vector<real> c1, c2;
};

// Base (i): manufactured trig fluctuation pair, smooth low-mode periodic
// combinations, discretely mean-zero by periodic symmetry (still explicitly
// mean-zero projected through the public MeanZeroProjector before use, per
// prepare_jvp_base's caller contract).
[[nodiscard]] FluctuationPair manufactured_base(const Grid3D& grid) {
    FluctuationPair pair;
    pair.c1.resize(grid.num_cells());
    pair.c2.resize(grid.num_cells());
    for (int iz = 0; iz < grid.nz; ++iz) {
        const double z = (iz + 0.5) * static_cast<double>(grid.dz);
        for (int iy = 0; iy < grid.ny; ++iy) {
            const double y = (iy + 0.5) * static_cast<double>(grid.dy);
            for (int ix = 0; ix < grid.nx; ++ix) {
                const double x = (ix + 0.5) * static_cast<double>(grid.dx);
                const std::size_t id = grid.idx(ix, iy, iz);
                pair.c1[id] = static_cast<real>(0.30 * std::sin(2.0 * kPi * x) +
                                                0.20 * std::cos(4.0 * kPi * y) -
                                                0.15 * std::sin(2.0 * kPi * z));
                pair.c2[id] = static_cast<real>(0.25 * std::cos(2.0 * kPi * x) -
                                                0.10 * std::sin(4.0 * kPi * z) +
                                                0.20 * std::cos(2.0 * kPi * y));
            }
        }
    }
    return pair;
}

enum class DirectionType { random_seeded, gradient_like, single_fourier_mode };

[[nodiscard]] const char* direction_type_name(DirectionType type) {
    switch (type) {
        case DirectionType::random_seeded: return "random_seeded";
        case DirectionType::gradient_like: return "gradient_like";
        case DirectionType::single_fourier_mode: return "single_fourier_mode";
    }
    return "unknown";
}

[[nodiscard]] FluctuationPair build_direction(const Grid3D& grid, DirectionType type,
                                              const FluctuationPair& base) {
    if (type == DirectionType::gradient_like) return base; // smooth low-mode field: the base itself.

    FluctuationPair pair;
    pair.c1.resize(grid.num_cells());
    pair.c2.resize(grid.num_cells());

    if (type == DirectionType::random_seeded) {
        std::uint64_t state = 0x5EED2026081200ULL; // documented fixed seed.
        for (std::size_t i = 0; i < pair.c1.size(); ++i)
            pair.c1[i] = static_cast<real>(0.5 * splitmix64_uniform_pm1(state));
        for (std::size_t i = 0; i < pair.c2.size(); ++i)
            pair.c2[i] = static_cast<real>(0.5 * splitmix64_uniform_pm1(state));
        return pair;
    }

    // single_fourier_mode: c1 = sin(2*pi*x), c2 = cos(2*pi*y).
    for (int iz = 0; iz < grid.nz; ++iz) {
        for (int iy = 0; iy < grid.ny; ++iy) {
            const double y = (iy + 0.5) * static_cast<double>(grid.dy);
            for (int ix = 0; ix < grid.nx; ++ix) {
                const double x = (ix + 0.5) * static_cast<double>(grid.dx);
                const std::size_t id = grid.idx(ix, iy, iz);
                pair.c1[id] = static_cast<real>(std::sin(2.0 * kPi * x));
                pair.c2[id] = static_cast<real>(std::cos(2.0 * kPi * y));
            }
        }
    }
    return pair;
}

[[nodiscard]] FluctuationPair negate(const FluctuationPair& pair) {
    FluctuationPair out;
    out.c1.resize(pair.c1.size());
    out.c2.resize(pair.c2.size());
    for (std::size_t i = 0; i < pair.c1.size(); ++i) out.c1[i] = -pair.c1[i];
    for (std::size_t i = 0; i < pair.c2.size(); ++i) out.c2[i] = -pair.c2[i];
    return out;
}

// Mean-zero projects a device coupled state in place via the public
// MeanZeroProjector API (used to satisfy prepare_jvp_base's "base already
// projected" caller contract, and to build the eta=0 oracle's reference
// direction).
void project_inplace(CudaContext& ctx, DeviceSpan<real> c1, DeviceSpan<real> c2) {
    constraints::MeanZeroWorkspace workspace;
    workspace.prepare(c1.size());
    constraints::MeanZeroProjector projector;
    projector.project(ctx, c1, workspace);
    projector.project(ctx, c2, workspace);
    ctx.synchronize();
}

// ---------------------------------------------------------------------------
// (a)/(c) shared delta-sweep fixture computation.
// ---------------------------------------------------------------------------

struct SweepPoint {
    double delta{};
    double discrepancy{};
    bool delta_min_clamped{};
    bool delta_max_clamped{};
    double gauge_ratio{}; // max(|mean(Jv_forward)|, |mean(Jv_central)|) / jv_mean_limit, both components.
};

struct SweepFixtureResult {
    std::string label;
    int n{};
    DirectionType direction_type{};
    std::vector<SweepPoint> points; // forced-delta sweep, ascending multiplier order.
    double policy_delta{};
    double policy_discrepancy{};
    bool policy_unclamped{};
    double max_mean_abs_over_limit_ratio{}; // max over every Jv mean check of |mean|/limit.
    bool pass{};
};

[[nodiscard]] JvpDeltaConfig forced_delta_config(real delta_fixed) {
    JvpDeltaConfig cfg;
    cfg.delta_max = delta_fixed;
    cfg.delta_min = std::nextafter(delta_fixed, real{0});
    return cfg;
}

// One combo: prepares the workspace/base once, runs an 7-point >=6-decade
// forced-delta sweep bracketing the unclamped policy delta, plus one
// unclamped policy-delta evaluation. Every Jv (forward and central-combined)
// mean is checked against jv_mean_limit (D... gauge check (d)).
[[nodiscard]] SweepFixtureResult run_delta_sweep(const std::string& label, int n,
                                                 DirectionType direction_type, double amplitude) {
    const Grid3D grid = isotropic_grid(n);
    const std::size_t cells = grid.num_cells();

    CudaContext ctx(0);
    DeviceBuffer<real> q = upload(trig_q_host(grid, amplitude));
    const AffineGauge gauge = AffineGauge::benchmark(real{1});
    const NonlinearSourceConfig source_config = source_config_v_rms_1();
    const ResidualHistogramConfig histogram_config{};

    const FluctuationPair base_host = manufactured_base(grid);
    DeviceBuffer<real> base_c1 = upload(base_host.c1);
    DeviceBuffer<real> base_c2 = upload(base_host.c2);
    project_inplace(ctx, base_c1.span(), base_c2.span());

    JvpWorkspace workspace;
    workspace.prepare(cells);
    workspace.prepare_jvp_base(ctx, grid, DeviceSpan<const real>(q.span()),
                               CoupledVectorView{base_c1.span(), base_c2.span()}, gauge, real{1},
                               source_config, histogram_config);

    const FluctuationPair dir_pos_host = build_direction(grid, direction_type, base_host);
    const FluctuationPair dir_neg_host = negate(dir_pos_host);
    DeviceBuffer<real> dir_pos_c1 = upload(dir_pos_host.c1);
    DeviceBuffer<real> dir_pos_c2 = upload(dir_pos_host.c2);
    DeviceBuffer<real> dir_neg_c1 = upload(dir_neg_host.c1);
    DeviceBuffer<real> dir_neg_c2 = upload(dir_neg_host.c2);

    DeviceBuffer<real> jv_pos_c1(cells), jv_pos_c2(cells);
    DeviceBuffer<real> jv_neg_c1(cells), jv_neg_c2(cells);

    const auto evaluate_at = [&](const JvpDeltaConfig& delta_config) -> SweepPoint {
        const JvpApplyReport report_pos = workspace.apply(
            ctx, grid, ConstCoupledVectorView(DeviceSpan<const real>(dir_pos_c1.span()),
                                              DeviceSpan<const real>(dir_pos_c2.span())),
            delta_config, CoupledVectorView{jv_pos_c1.span(), jv_pos_c2.span()});
        const JvpApplyReport report_neg = workspace.apply(
            ctx, grid, ConstCoupledVectorView(DeviceSpan<const real>(dir_neg_c1.span()),
                                              DeviceSpan<const real>(dir_neg_c2.span())),
            delta_config, CoupledVectorView{jv_neg_c1.span(), jv_neg_c2.span()});
        ctx.synchronize();

        if (report_pos.status != JvpApplyStatus::ok || report_neg.status != JvpApplyStatus::ok) {
            return SweepPoint{static_cast<double>(report_pos.delta),
                              std::numeric_limits<double>::infinity(), report_pos.delta_min_clamped,
                              report_pos.delta_max_clamped, 0.0};
        }

        const std::vector<real> jv_p1 = download(DeviceSpan<const real>(jv_pos_c1.span()));
        const std::vector<real> jv_p2 = download(DeviceSpan<const real>(jv_pos_c2.span()));
        const std::vector<real> jv_n1 = download(DeviceSpan<const real>(jv_neg_c1.span()));
        const std::vector<real> jv_n2 = download(DeviceSpan<const real>(jv_neg_c2.span()));

        std::vector<real> jv_c1(cells), jv_c2(cells);
        for (std::size_t i = 0; i < cells; ++i)
            jv_c1[i] = static_cast<real>(0.5 * (static_cast<double>(jv_p1[i]) - static_cast<double>(jv_n1[i])));
        for (std::size_t i = 0; i < cells; ++i)
            jv_c2[i] = static_cast<real>(0.5 * (static_cast<double>(jv_p2[i]) - static_cast<double>(jv_n2[i])));

        std::vector<real> diff1(cells), diff2(cells);
        for (std::size_t i = 0; i < cells; ++i) diff1[i] = jv_p1[i] - jv_c1[i];
        for (std::size_t i = 0; i < cells; ++i) diff2[i] = jv_p2[i] - jv_c2[i];

        const double num = weighted_norm_host(diff1, diff2, 1.0, 1.0);
        const double den = weighted_norm_host(jv_c1, jv_c2, 1.0, 1.0);
        const double discrepancy = den > 0.0 ? num / den : std::numeric_limits<double>::infinity();

        // (d) gauge check: every Jv mean (forward and central) against the
        // documented delta-scaled bound (jv_mean_limit).
        const double limit = jv_mean_limit(jv_p1, static_cast<double>(report_pos.delta));
        const double m1 = mean_abs_host(jv_p1);
        const double m2 = mean_abs_host(jv_p2);
        const double mc1 = mean_abs_host(jv_c1);
        const double mc2 = mean_abs_host(jv_c2);
        const double gauge_ratio = std::max({m1, m2, mc1, mc2}) / std::max(limit, 1e-300);

        return SweepPoint{static_cast<double>(report_pos.delta), discrepancy, report_pos.delta_min_clamped,
                          report_pos.delta_max_clamped, gauge_ratio};
    };

    // Probe: unclamped policy delta (default JvpDeltaConfig).
    const JvpDeltaConfig default_config{};
    double max_ratio = 0.0;
    const auto evaluate_and_track_ratio = [&](const JvpDeltaConfig& cfg) {
        const SweepPoint pt = evaluate_at(cfg);
        max_ratio = std::max(max_ratio, pt.gauge_ratio);
        return pt;
    };

    const SweepPoint policy_point = evaluate_and_track_ratio(default_config);

    SweepFixtureResult result;
    result.label = label;
    result.n = n;
    result.direction_type = direction_type;
    result.policy_delta = policy_point.delta;
    result.policy_discrepancy = policy_point.discrepancy;
    result.policy_unclamped = !policy_point.delta_min_clamped && !policy_point.delta_max_clamped;

    for (int k = -3; k <= 3; ++k) {
        const real delta_fixed = static_cast<real>(policy_point.delta * std::pow(10.0, k));
        const JvpDeltaConfig cfg = forced_delta_config(delta_fixed);
        const SweepPoint point = evaluate_and_track_ratio(cfg);
        result.points.push_back(point);
    }

    result.max_mean_abs_over_limit_ratio = max_ratio;
    return result;
}

[[nodiscard]] std::vector<SweepFixtureResult>& sweep_fixture_cache() {
    static std::vector<SweepFixtureResult> cache = [] {
        std::vector<SweepFixtureResult> results;
        for (const int n : {16, 32}) {
            for (const DirectionType dir :
                {DirectionType::random_seeded, DirectionType::gradient_like,
                 DirectionType::single_fourier_mode}) {
                std::ostringstream label;
                label << "n" << n << "_" << direction_type_name(dir);
                results.push_back(run_delta_sweep(label.str(), n, dir, 0.5));
            }
        }
        return results;
    }();
    return cache;
}

// ===========================================================================
// (a) jvp_delta_sweep_u_shape: U-shaped delta study + 1e-5 policy gate + (d).
// ===========================================================================

[[nodiscard]] CaseResult case_jvp_delta_sweep_u_shape() {
    auto& fixtures = sweep_fixture_cache();

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const std::string& name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    for (const auto& fixture : fixtures) {
        std::cout << "jvp_delta_sweep label=" << fixture.label << " n=" << fixture.n
                  << " direction=" << direction_type_name(fixture.direction_type)
                  << " policy_delta=" << std::setprecision(10) << fixture.policy_delta
                  << " policy_discrepancy=" << fixture.policy_discrepancy
                  << " policy_unclamped=" << (fixture.policy_unclamped ? "true" : "false") << '\n';
        std::size_t min_index = 0;
        double min_value = std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < fixture.points.size(); ++i) {
            const auto& p = fixture.points[i];
            std::cout << "  sweep[" << i << "] delta=" << p.delta << " discrepancy=" << p.discrepancy
                      << " min_clamped=" << (p.delta_min_clamped ? "true" : "false")
                      << " max_clamped=" << (p.delta_max_clamped ? "true" : "false") << '\n';
            if (p.discrepancy < min_value) {
                min_value = p.discrepancy;
                min_index = i;
            }
        }
        const bool interior_minimum = min_index > 0 && min_index + 1 < fixture.points.size();
        const bool ends_exceed_10x = fixture.points.front().discrepancy >= 10.0 * min_value &&
                                     fixture.points.back().discrepancy >= 10.0 * min_value;
        add_check(fixture.label + "_u_shape_interior_minimum", interior_minimum);
        add_check(fixture.label + "_u_shape_ends_ge_10x_min", ends_exceed_10x);
        add_check(fixture.label + "_policy_unclamped", fixture.policy_unclamped);
        add_check(fixture.label + "_policy_discrepancy_le_1e-5", fixture.policy_discrepancy <= 1e-5);
        add_check(fixture.label + "_gauge_mean_within_limit", fixture.max_mean_abs_over_limit_ratio <= 1.0);
    }

    for (const auto& [name, ok] : checks) std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';

    return {pass,
            "jvp_delta_sweep_u_shape",
            "gpu-jvp-fd-sweep",
            "16^3, 32^3, K=exp(0.5*sin sin sin), 3 direction types, 7-point forced-delta sweep",
            0.0,
            0.0,
            "U-shaped curve (interior minimum, both ends >=10x min); policy discrepancy<=1e-5; "
            "gauge mean within jv_mean_limit",
            pass ? "all pass" : "some failed",
            "SF-22 activation bitácora decision D6: prespecified U-study + 1e-5 policy gate on the "
            "small trig fixtures, plus the D3/(d) gauge mean-zero check on every Jv computed"};
}

// ===========================================================================
// (c) jvp_fd_convergence_order: O(delta) evidence on the large-delta side.
// ===========================================================================

[[nodiscard]] CaseResult case_jvp_fd_convergence_order() {
    auto& fixtures = sweep_fixture_cache(); // reuses the cached sweep, no recompute.

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const std::string& name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    for (const auto& fixture : fixtures) {
        // points[] index 4,5,6 correspond to multipliers k=+1,+2,+3 (large-
        // delta / truncation-dominated side); use k=+1 vs k=+2 (10x apart).
        const double d1 = fixture.points[4].discrepancy; // k=+1
        const double d2 = fixture.points[5].discrepancy; // k=+2
        const bool finite = std::isfinite(d1) && std::isfinite(d2) && d1 > 0.0;
        const double ratio = finite ? d2 / d1 : std::numeric_limits<double>::quiet_NaN();
        const bool in_range = finite && ratio >= 5.0 && ratio <= 20.0;
        std::cout << "jvp_fd_convergence_order label=" << fixture.label << " d(k=1)=" << d1
                  << " d(k=2)=" << d2 << " ratio=" << ratio << " expected_range=[5,20]\n";
        add_check(fixture.label + "_ratio_in_range", in_range);
    }

    for (const auto& [name, ok] : checks) std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';

    return {pass,
            "jvp_fd_convergence_order",
            "gpu-jvp-evidence",
            "reuses jvp_delta_sweep_u_shape's cached sweep table",
            0.0,
            0.0,
            "discrepancy(k=+2)/discrepancy(k=+1) in [5,20] for a 10x delta ratio (O(delta) evidence)",
            pass ? "all pass" : "some failed",
            "SF-22 activation bitácora decision D6/spec item (c): loose evidence check, not the hard "
            "gate (that is jvp_delta_sweep_u_shape); reported honestly even if occasionally marginal"};
}

// ===========================================================================
// (b) jvp_eta0_linearity_oracle: eta=0 exact-linearity oracle (D7) + (d).
// ===========================================================================

[[nodiscard]] CaseResult case_jvp_eta0_linearity_oracle() {
    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const std::string& name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    for (const int n : {16, 32}) {
        for (const DirectionType dir :
            {DirectionType::random_seeded, DirectionType::gradient_like,
             DirectionType::single_fourier_mode}) {
            const Grid3D grid = isotropic_grid(n);
            const std::size_t cells = grid.num_cells();
            std::ostringstream label_stream;
            label_stream << "n" << n << "_" << direction_type_name(dir);
            const std::string label = label_stream.str();

            CudaContext ctx(0);
            DeviceBuffer<real> q = upload(trig_q_host(grid, 0.5));
            const AffineGauge gauge = AffineGauge::benchmark(real{1});
            const NonlinearSourceConfig source_config = source_config_v_rms_1();
            const ResidualHistogramConfig histogram_config{};

            const FluctuationPair base_host = manufactured_base(grid);
            DeviceBuffer<real> base_c1 = upload(base_host.c1);
            DeviceBuffer<real> base_c2 = upload(base_host.c2);
            project_inplace(ctx, base_c1.span(), base_c2.span());

            JvpWorkspace workspace;
            workspace.prepare(cells);
            workspace.prepare_jvp_base(ctx, grid, DeviceSpan<const real>(q.span()),
                                       CoupledVectorView{base_c1.span(), base_c2.span()}, gauge,
                                       real{0} /* eta=0: F is affine in Psi */, source_config,
                                       histogram_config);

            const FluctuationPair dir_host = build_direction(grid, dir, base_host);
            DeviceBuffer<real> dir_c1 = upload(dir_host.c1);
            DeviceBuffer<real> dir_c2 = upload(dir_host.c2);

            DeviceBuffer<real> jv_c1(cells), jv_c2(cells);
            const JvpDeltaConfig default_config{};
            const JvpApplyReport report = workspace.apply(
                ctx, grid,
                ConstCoupledVectorView(DeviceSpan<const real>(dir_c1.span()),
                                      DeviceSpan<const real>(dir_c2.span())),
                default_config, CoupledVectorView{jv_c1.span(), jv_c2.span()});
            ctx.synchronize();

            add_check(label + "_apply_ok", report.status == JvpApplyStatus::ok);
            add_check(label + "_delta_unclamped", !report.delta_min_clamped && !report.delta_max_clamped);
            if (report.status != JvpApplyStatus::ok) continue;

            // Reference: A applied to the SAME mean-zero-projected direction
            // (public MeanZeroProjector API), independent of JvpWorkspace's
            // own internal projected copy.
            DeviceBuffer<real> ref_dir_c1 = upload(dir_host.c1);
            DeviceBuffer<real> ref_dir_c2 = upload(dir_host.c2);
            project_inplace(ctx, ref_dir_c1.span(), ref_dir_c2.span());

            const operators::LesterPositiveDiffusionOperator diffusion_operator(
                grid, DeviceSpan<const real>(q.span()));
            DeviceBuffer<real> ref1(cells), ref2(cells);
            diffusion_operator.apply(ctx, DeviceSpan<const real>(ref_dir_c1.span()), ref1.span());
            diffusion_operator.apply(ctx, DeviceSpan<const real>(ref_dir_c2.span()), ref2.span());
            ctx.synchronize();

            const std::vector<real> jv1 = download(DeviceSpan<const real>(jv_c1.span()));
            const std::vector<real> jv2 = download(DeviceSpan<const real>(jv_c2.span()));
            const std::vector<real> a1 = download(DeviceSpan<const real>(ref1.span()));
            const std::vector<real> a2 = download(DeviceSpan<const real>(ref2.span()));

            std::vector<real> diff1(cells), diff2(cells);
            for (std::size_t i = 0; i < cells; ++i) diff1[i] = jv1[i] - a1[i];
            for (std::size_t i = 0; i < cells; ++i) diff2[i] = jv2[i] - a2[i];
            const double relative_error =
                weighted_norm_host(diff1, diff2, 1.0, 1.0) / weighted_norm_host(a1, a2, 1.0, 1.0);

            add_check(label + "_relative_error_le_1e-6", relative_error <= 1e-6);

            const double limit = jv_mean_limit(jv1, static_cast<double>(report.delta));
            const double gauge_ratio = std::max(mean_abs_host(jv1), mean_abs_host(jv2)) / std::max(limit, 1e-300);
            add_check(label + "_gauge_mean_within_limit", gauge_ratio <= 1.0);

            std::cout << std::setprecision(10) << "jvp_eta0_linearity label=" << label
                      << " delta=" << report.delta << " relative_error=" << relative_error
                      << " gauge_ratio=" << gauge_ratio << '\n';
        }
    }

    for (const auto& [name, ok] : checks) std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';

    return {pass,
            "jvp_eta0_linearity_oracle",
            "gpu-jvp-oracle",
            "16^3, 32^3, K=exp(0.5*sin sin sin), 3 direction types, eta=0 exact-affine oracle",
            0.0,
            0.0,
            "relative error vs A*project(p) <= 1e-6 at the unclamped policy delta; gauge mean within limit",
            pass ? "all pass" : "some failed",
            "SF-22 activation bitácora decision D7: at eta=0, F is affine in Psi, so J p == [A p1; A "
            "p2] exactly; operators::LesterPositiveDiffusionOperator is reused directly, never a "
            "second discretization"};
}

// ===========================================================================
// Extra fixture-family (ii) check: converged adaptive-Picard base state.
// ===========================================================================

[[nodiscard]] CaseResult case_jvp_converged_picard_base() {
    constexpr int n = 16;
    constexpr double amplitude = 0.5;
    const Grid3D grid = isotropic_grid(n);
    const std::size_t cells = grid.num_cells();

    CudaContext ctx(0);
    DeviceBuffer<real> k_field = upload(trig_k_host(grid, amplitude));
    DarcyBuffers darcy(grid);

    StreamfunctionProblemView problem;
    problem.grid = grid;
    problem.conductivity = DeviceSpan<const real>(k_field.span());
    problem.conductivity_representation = ConductivityRepresentation::conductivity_k;
    problem.darcy_velocity = CompactMacVelocityConstView{DeviceSpan<const real>(darcy.u.span()),
                                                         DeviceSpan<const real>(darcy.v.span()),
                                                         DeviceSpan<const real>(darcy.w.span())};
    problem.bc = triply_periodic();
    problem.gauge = AffineGauge::benchmark(real{1});

    StreamfunctionSolverConfig config{}; // eta=1, epsilon=1e-2 defaults (accepted SF-15/SF-20 fixture family).
    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;
    const StreamfunctionSolveReport solve_report = solve_streamfunctions(ctx, problem, config, fields, workspace);
    ctx.synchronize();

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };
    add_check("picard_converged", solve_report.status == StreamfunctionSolveStatus::converged);

    // q must match the same K field the solver used (representation
    // conductivity_k -> q = 1/K).
    DeviceBuffer<real> q = upload(trig_q_host(grid, amplitude));
    const AffineGauge gauge = AffineGauge::benchmark(real{1});
    const NonlinearSourceConfig source_config = source_config_v_rms_1();
    const ResidualHistogramConfig histogram_config{};

    JvpWorkspace jvp_workspace;
    jvp_workspace.prepare(cells);
    // Caller contract: base already mean-zero projected. solve_streamfunctions
    // leaves fields.u1_span()/u2_span() mean-zero projected by construction
    // (every accepted-state update is followed by a projection); no
    // additional projection is applied here.
    jvp_workspace.prepare_jvp_base(ctx, grid, DeviceSpan<const real>(q.span()),
                                   CoupledVectorView{fields.u1_span(), fields.u2_span()}, gauge, real{1},
                                   source_config, histogram_config);

    // Direction: the converged state itself (a smooth, mean-zero, low-mode-
    // dominated field), matching this file's "gradient_like" convention.
    const std::vector<real> dir1_host = download(fields.u1_span());
    const std::vector<real> dir2_host = download(fields.u2_span());
    DeviceBuffer<real> dir1 = upload(dir1_host);
    DeviceBuffer<real> dir2 = upload(dir2_host);
    DeviceBuffer<real> dir1_neg = upload([&] {
        auto v = dir1_host;
        for (auto& x : v) x = -x;
        return v;
    }());
    DeviceBuffer<real> dir2_neg = upload([&] {
        auto v = dir2_host;
        for (auto& x : v) x = -x;
        return v;
    }());

    DeviceBuffer<real> jv1(cells), jv2(cells);
    DeviceBuffer<real> jv1_neg(cells), jv2_neg(cells);
    const JvpDeltaConfig default_config{};
    const JvpApplyReport report_pos = jvp_workspace.apply(
        ctx, grid,
        ConstCoupledVectorView(DeviceSpan<const real>(dir1.span()), DeviceSpan<const real>(dir2.span())),
        default_config, CoupledVectorView{jv1.span(), jv2.span()});
    const JvpApplyReport report_neg = jvp_workspace.apply(
        ctx, grid,
        ConstCoupledVectorView(DeviceSpan<const real>(dir1_neg.span()), DeviceSpan<const real>(dir2_neg.span())),
        default_config, CoupledVectorView{jv1_neg.span(), jv2_neg.span()});
    ctx.synchronize();

    add_check("apply_pos_ok", report_pos.status == JvpApplyStatus::ok);
    add_check("apply_neg_ok", report_neg.status == JvpApplyStatus::ok);

    if (report_pos.status == JvpApplyStatus::ok && report_neg.status == JvpApplyStatus::ok) {
        const std::vector<real> jvp1 = download(DeviceSpan<const real>(jv1.span()));
        const std::vector<real> jvp2 = download(DeviceSpan<const real>(jv2.span()));
        const std::vector<real> jvn1 = download(DeviceSpan<const real>(jv1_neg.span()));
        const std::vector<real> jvn2 = download(DeviceSpan<const real>(jv2_neg.span()));

        std::vector<real> jvc1(cells), jvc2(cells);
        for (std::size_t i = 0; i < cells; ++i)
            jvc1[i] = static_cast<real>(0.5 * (static_cast<double>(jvp1[i]) - static_cast<double>(jvn1[i])));
        for (std::size_t i = 0; i < cells; ++i)
            jvc2[i] = static_cast<real>(0.5 * (static_cast<double>(jvp2[i]) - static_cast<double>(jvn2[i])));

        std::vector<real> diff1(cells), diff2(cells);
        for (std::size_t i = 0; i < cells; ++i) diff1[i] = jvp1[i] - jvc1[i];
        for (std::size_t i = 0; i < cells; ++i) diff2[i] = jvp2[i] - jvc2[i];
        const double discrepancy =
            weighted_norm_host(diff1, diff2, 1.0, 1.0) / weighted_norm_host(jvc1, jvc2, 1.0, 1.0);
        add_check("policy_discrepancy_le_1e-5", discrepancy <= 1e-5);

        const double limit = jv_mean_limit(jvp1, static_cast<double>(report_pos.delta));
        const double gauge_ratio = std::max(mean_abs_host(jvp1), mean_abs_host(jvp2)) / std::max(limit, 1e-300);
        add_check("gauge_mean_within_limit", gauge_ratio <= 1.0);

        std::cout << std::setprecision(10) << "jvp_converged_picard_base delta=" << report_pos.delta
                  << " discrepancy=" << discrepancy << " gauge_ratio=" << gauge_ratio
                  << " picard_iterations=" << solve_report.picard_iterations
                  << " r_F=" << solve_report.residual.r_F << '\n';
    }

    for (const auto& [name, ok] : checks) std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';

    return {pass,
            "jvp_converged_picard_base",
            "gpu-jvp-fixture",
            "16^3, K=exp(0.5*sin sin sin), base state = converged adaptive-Picard solve_streamfunctions "
            "result (eta=1, epsilon=1e-2)",
            0.0,
            0.0,
            "policy discrepancy<=1e-5; gauge mean within limit, at a genuine converged nonlinear state",
            pass ? "all pass" : "some failed",
            "SF-22 fixture family item (ii): the U-study/eta=0 oracle above use manufactured trig "
            "fluctuation bases; this case exercises the SAME JvpWorkspace contract at an actually "
            "converged Picard fixed point"};
}

// ===========================================================================
// (e) jvp_contract_fail_fast: fail-fast preconditions and observed clamps.
// ===========================================================================

[[nodiscard]] CaseResult case_jvp_contract_fail_fast() {
    constexpr int n = 16;
    const Grid3D grid = isotropic_grid(n);
    const std::size_t cells = grid.num_cells();

    CudaContext ctx(0);
    DeviceBuffer<real> q = upload(trig_q_host(grid, 0.5));
    const AffineGauge gauge = AffineGauge::benchmark(real{1});
    const NonlinearSourceConfig source_config = source_config_v_rms_1();
    const ResidualHistogramConfig histogram_config{};

    const FluctuationPair base_host = manufactured_base(grid);
    DeviceBuffer<real> base_c1 = upload(base_host.c1);
    DeviceBuffer<real> base_c2 = upload(base_host.c2);
    project_inplace(ctx, base_c1.span(), base_c2.span());

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const std::string& name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    // apply() before prepare_jvp_base() -> std::logic_error.
    {
        JvpWorkspace ws;
        ws.prepare(cells);
        DeviceBuffer<real> d1 = upload(base_host.c1);
        DeviceBuffer<real> d2 = upload(base_host.c2);
        DeviceBuffer<real> out1(cells), out2(cells);
        bool threw_logic_error = false;
        try {
            const JvpApplyReport unused_report = ws.apply(
                ctx, grid,
                ConstCoupledVectorView(DeviceSpan<const real>(d1.span()), DeviceSpan<const real>(d2.span())),
                JvpDeltaConfig{}, CoupledVectorView{out1.span(), out2.span()});
            (void)unused_report;
        } catch (const std::logic_error&) {
            threw_logic_error = true;
        } catch (const std::exception&) {
        }
        add_check("apply_before_prepare_jvp_base_throws_logic_error", threw_logic_error);
    }

    // prepare_jvp_base() before prepare() -> std::logic_error.
    {
        JvpWorkspace ws;
        bool threw_logic_error = false;
        try {
            ws.prepare_jvp_base(ctx, grid, DeviceSpan<const real>(q.span()),
                                CoupledVectorView{base_c1.span(), base_c2.span()}, gauge, real{1},
                                source_config, histogram_config);
        } catch (const std::logic_error&) {
            threw_logic_error = true;
        } catch (const std::exception&) {
        }
        add_check("prepare_jvp_base_before_prepare_throws_logic_error", threw_logic_error);
    }

    // Zero direction -> std::invalid_argument, message fragment documented.
    {
        JvpWorkspace ws;
        ws.prepare(cells);
        ws.prepare_jvp_base(ctx, grid, DeviceSpan<const real>(q.span()),
                            CoupledVectorView{base_c1.span(), base_c2.span()}, gauge, real{1}, source_config,
                            histogram_config);
        DeviceBuffer<real> zero1 = upload(std::vector<real>(cells, real{0}));
        DeviceBuffer<real> zero2 = upload(std::vector<real>(cells, real{0}));
        DeviceBuffer<real> out1(cells), out2(cells);
        bool threw_invalid_argument = false;
        std::string message;
        try {
            const JvpApplyReport unused_report = ws.apply(
                ctx, grid,
                ConstCoupledVectorView(DeviceSpan<const real>(zero1.span()), DeviceSpan<const real>(zero2.span())),
                JvpDeltaConfig{}, CoupledVectorView{out1.span(), out2.span()});
            (void)unused_report;
        } catch (const std::invalid_argument& error) {
            threw_invalid_argument = true;
            message = error.what();
        } catch (const std::exception&) {
        }
        const bool has_fragment = message.find("nonzero weighted direction norm") != std::string::npos;
        add_check("zero_direction_throws_invalid_argument", threw_invalid_argument);
        add_check("zero_direction_message_fragment", has_fragment);
        std::cout << "jvp_contract_fail_fast zero_direction message=\"" << message << "\"\n";
    }

    // Direction scaled so the (default, unclamped-normally) policy delta
    // exceeds delta_max -> delta_max_clamped flag + counter increments.
    {
        JvpWorkspace ws;
        ws.prepare(cells);
        ws.prepare_jvp_base(ctx, grid, DeviceSpan<const real>(q.span()),
                            CoupledVectorView{base_c1.span(), base_c2.span()}, gauge, real{1}, source_config,
                            histogram_config);
        const FluctuationPair dir = build_direction(grid, DirectionType::random_seeded, base_host);
        std::vector<real> tiny1(cells), tiny2(cells);
        for (std::size_t i = 0; i < cells; ++i) tiny1[i] = dir.c1[i] * real{1e-12};
        for (std::size_t i = 0; i < cells; ++i) tiny2[i] = dir.c2[i] * real{1e-12};
        DeviceBuffer<real> d1 = upload(tiny1);
        DeviceBuffer<real> d2 = upload(tiny2);
        DeviceBuffer<real> out1(cells), out2(cells);

        const unsigned long long before = ws.delta_max_clamp_count();
        const JvpApplyReport report = ws.apply(
            ctx, grid, ConstCoupledVectorView(DeviceSpan<const real>(d1.span()), DeviceSpan<const real>(d2.span())),
            JvpDeltaConfig{}, CoupledVectorView{out1.span(), out2.span()});
        ctx.synchronize();
        const unsigned long long after = ws.delta_max_clamp_count();

        add_check("tiny_direction_forces_delta_max_clamp", report.delta_max_clamped);
        add_check("tiny_direction_delta_equals_delta_max", report.delta == JvpDeltaConfig{}.delta_max);
        add_check("delta_max_clamp_counter_incremented", after == before + 1);
        std::cout << std::setprecision(10) << "jvp_contract_fail_fast tiny_direction delta=" << report.delta
                  << " weighted_p_norm=" << report.weighted_p_norm << " clamp_count_before=" << before
                  << " clamp_count_after=" << after << '\n';
    }

    for (const auto& [name, ok] : checks) std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';

    return {pass,
            "jvp_contract_fail_fast",
            "contract",
            "16^3 trig fixture, JvpWorkspace ordering/precondition/clamp contract",
            static_cast<double>(checks.size()),
            0.0,
            "all checks pass",
            pass ? "all pass" : "some failed",
            "SF-22 T01 D2/ordering contract: apply()-before-prepare_jvp_base() and prepare_jvp_base()"
            "-before-prepare() throw std::logic_error; a zero direction throws std::invalid_argument; "
            "a direction scaled small enough to push the policy delta past delta_max clamps and "
            "increments the cumulative counter"};
}

// ===========================================================================
// (f) jvp_cache_and_determinism: cached base + bitwise-deterministic apply().
// ===========================================================================

[[nodiscard]] CaseResult case_jvp_cache_and_determinism() {
    constexpr int n = 16;
    const Grid3D grid = isotropic_grid(n);
    const std::size_t cells = grid.num_cells();

    CudaContext ctx(0);
    DeviceBuffer<real> q = upload(trig_q_host(grid, 0.5));
    const AffineGauge gauge = AffineGauge::benchmark(real{1});
    const NonlinearSourceConfig source_config = source_config_v_rms_1();
    const ResidualHistogramConfig histogram_config{};

    const FluctuationPair base_host = manufactured_base(grid);
    DeviceBuffer<real> base_c1 = upload(base_host.c1);
    DeviceBuffer<real> base_c2 = upload(base_host.c2);
    project_inplace(ctx, base_c1.span(), base_c2.span());

    const FluctuationPair dir_host = build_direction(grid, DirectionType::single_fourier_mode, base_host);
    DeviceBuffer<real> dir_c1 = upload(dir_host.c1);
    DeviceBuffer<real> dir_c2 = upload(dir_host.c2);

    JvpWorkspace ws;
    ws.prepare(cells);
    ws.prepare_jvp_base(ctx, grid, DeviceSpan<const real>(q.span()),
                        CoupledVectorView{base_c1.span(), base_c2.span()}, gauge, real{1}, source_config,
                        histogram_config);

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const std::string& name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    add_check("base_evaluations_eq_1_after_one_prepare_jvp_base", ws.base_evaluations() == 1);

    DeviceBuffer<real> out1_a(cells), out2_a(cells);
    const JvpApplyReport report_a =
        ws.apply(ctx, grid,
                ConstCoupledVectorView(DeviceSpan<const real>(dir_c1.span()), DeviceSpan<const real>(dir_c2.span())),
                JvpDeltaConfig{}, CoupledVectorView{out1_a.span(), out2_a.span()});
    ctx.synchronize();
    add_check("apply_a_ok", report_a.status == JvpApplyStatus::ok);
    add_check("apply_evaluations_eq_1_after_one_apply", ws.apply_evaluations() == 1);

    DeviceBuffer<real> out1_b(cells), out2_b(cells);
    const JvpApplyReport report_b =
        ws.apply(ctx, grid,
                ConstCoupledVectorView(DeviceSpan<const real>(dir_c1.span()), DeviceSpan<const real>(dir_c2.span())),
                JvpDeltaConfig{}, CoupledVectorView{out1_b.span(), out2_b.span()});
    ctx.synchronize();
    add_check("apply_b_ok", report_b.status == JvpApplyStatus::ok);
    add_check("apply_evaluations_eq_2_after_two_apply", ws.apply_evaluations() == 2);

    const std::vector<real> a1 = download(DeviceSpan<const real>(out1_a.span()));
    const std::vector<real> a2 = download(DeviceSpan<const real>(out2_a.span()));
    const std::vector<real> b1 = download(DeviceSpan<const real>(out1_b.span()));
    const std::vector<real> b2 = download(DeviceSpan<const real>(out2_b.span()));
    add_check("two_identical_apply_calls_bitwise_identical_c1", bitwise_equal(a1, b1));
    add_check("two_identical_apply_calls_bitwise_identical_c2", bitwise_equal(a2, b2));
    add_check("two_identical_apply_calls_bitwise_identical_delta", report_a.delta == report_b.delta);

    // Second prepare_jvp_base() (same base state) -> base_evaluations()==2;
    // apply() results still bitwise identical to before.
    ws.prepare_jvp_base(ctx, grid, DeviceSpan<const real>(q.span()),
                        CoupledVectorView{base_c1.span(), base_c2.span()}, gauge, real{1}, source_config,
                        histogram_config);
    add_check("base_evaluations_eq_2_after_second_prepare_jvp_base", ws.base_evaluations() == 2);

    DeviceBuffer<real> out1_c(cells), out2_c(cells);
    const JvpApplyReport report_c =
        ws.apply(ctx, grid,
                ConstCoupledVectorView(DeviceSpan<const real>(dir_c1.span()), DeviceSpan<const real>(dir_c2.span())),
                JvpDeltaConfig{}, CoupledVectorView{out1_c.span(), out2_c.span()});
    ctx.synchronize();
    add_check("apply_c_ok", report_c.status == JvpApplyStatus::ok);
    add_check("apply_evaluations_eq_3_after_third_apply", ws.apply_evaluations() == 3);

    const std::vector<real> c1v = download(DeviceSpan<const real>(out1_c.span()));
    const std::vector<real> c2v = download(DeviceSpan<const real>(out2_c.span()));
    add_check("post_second_prepare_apply_bitwise_identical_c1", bitwise_equal(a1, c1v));
    add_check("post_second_prepare_apply_bitwise_identical_c2", bitwise_equal(a2, c2v));

    for (const auto& [name, ok] : checks) std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';

    return {pass,
            "jvp_cache_and_determinism",
            "contract",
            "16^3 trig fixture, JvpWorkspace cached-base / evaluation-counter / determinism contract",
            static_cast<double>(checks.size()),
            0.0,
            "all checks pass",
            pass ? "all pass" : "some failed",
            "SF-22 T01 D4: base_evaluations()/apply_evaluations() are exact monotone counters; two "
            "apply() calls against the SAME cached base and direction are bitwise identical; a "
            "second prepare_jvp_base() on the SAME state refreshes the counter but not the result"};
}

// ===========================================================================
// (g) jvp_memory_accounting: exact-byte memory contract, no growth on apply.
// ===========================================================================

[[nodiscard]] CaseResult case_jvp_memory_accounting() {
    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const std::string& name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    for (const int n : {16, 32}) {
        const Grid3D grid = isotropic_grid(n);
        const std::size_t cells = grid.num_cells();

        CudaContext ctx(0);
        DeviceBuffer<real> q = upload(trig_q_host(grid, 0.5));
        const AffineGauge gauge = AffineGauge::benchmark(real{1});
        const NonlinearSourceConfig source_config = source_config_v_rms_1();
        const ResidualHistogramConfig histogram_config{};

        const FluctuationPair base_host = manufactured_base(grid);
        DeviceBuffer<real> base_c1 = upload(base_host.c1);
        DeviceBuffer<real> base_c2 = upload(base_host.c2);
        project_inplace(ctx, base_c1.span(), base_c2.span());
        const FluctuationPair dir_host = build_direction(grid, DirectionType::random_seeded, base_host);
        DeviceBuffer<real> dir_c1 = upload(dir_host.c1);
        DeviceBuffer<real> dir_c2 = upload(dir_host.c2);

        JvpWorkspace ws;
        ws.prepare(cells);
        const std::size_t actual_after_prepare = ws.allocated_device_bytes();
        const std::size_t estimate = JvpWorkspace::estimate_device_bytes(cells);

        std::ostringstream tag;
        tag << "n" << n;
        add_check(tag.str() + "_allocated_bytes_eq_estimate_after_prepare", actual_after_prepare == estimate);
        std::cout << "jvp_memory_accounting n=" << n << " allocated_bytes=" << actual_after_prepare
                  << " estimate_bytes=" << estimate << '\n';

        ws.prepare_jvp_base(ctx, grid, DeviceSpan<const real>(q.span()),
                            CoupledVectorView{base_c1.span(), base_c2.span()}, gauge, real{1}, source_config,
                            histogram_config);
        const std::size_t after_prepare_base = ws.allocated_device_bytes();
        add_check(tag.str() + "_bytes_unchanged_after_prepare_jvp_base", after_prepare_base == actual_after_prepare);

        DeviceBuffer<real> out1(cells), out2(cells);
        for (int call = 0; call < 3; ++call) {
            const JvpApplyReport report = ws.apply(
                ctx, grid,
                ConstCoupledVectorView(DeviceSpan<const real>(dir_c1.span()), DeviceSpan<const real>(dir_c2.span())),
                JvpDeltaConfig{}, CoupledVectorView{out1.span(), out2.span()});
            ctx.synchronize();
            add_check(tag.str() + "_apply_call" + std::to_string(call) + "_ok", report.status == JvpApplyStatus::ok);
            const std::size_t after_apply = ws.allocated_device_bytes();
            add_check(tag.str() + "_bytes_unchanged_after_apply_call" + std::to_string(call),
                      after_apply == actual_after_prepare);
        }
    }

    for (const auto& [name, ok] : checks) std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';

    return {pass,
            "jvp_memory_accounting",
            "contract",
            "16^3, 32^3, JvpWorkspace exact-byte memory contract",
            static_cast<double>(checks.size()),
            0.0,
            "all checks pass",
            pass ? "all pass" : "some failed",
            "allocated_device_bytes() == estimate_device_bytes(n) exactly after prepare(n); no "
            "additional allocation across prepare_jvp_base() or repeated apply() calls"};
}

} // namespace

CaseRegistry jvp_case_registry() {
    return {
        {"jvp_delta_sweep_u_shape", case_jvp_delta_sweep_u_shape},
        {"jvp_fd_convergence_order", case_jvp_fd_convergence_order},
        {"jvp_eta0_linearity_oracle", case_jvp_eta0_linearity_oracle},
        {"jvp_converged_picard_base", case_jvp_converged_picard_base},
        {"jvp_contract_fail_fast", case_jvp_contract_fail_fast},
        {"jvp_cache_and_determinism", case_jvp_cache_and_determinism},
        {"jvp_memory_accounting", case_jvp_memory_accounting},
    };
}

}  // namespace macroflow3d::streamfunctions::test
