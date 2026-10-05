#pragma once

/**
 * @file closure_fields.hpp
 * @brief SF-30 closure gate: analytic log-conductivity controls, the matched
 *        x3-averaged 2-D control, and stateless seed points (header-only,
 *        host-only C++17; no CUDA include).
 *
 * Analytic fields are ported VERBATIM from
 * `docs/experiments/artifacts/2026-10-02-closure-probes/scripts/closure_probe.py`
 * (`FIELDS`; unit amplitude, the caller multiplies by `eps`), with
 * `(X, Y, Z) = (x1, x2, x3)` on the unit period. The probe's band-limited
 * `gauss` field is not ported (the production gate uses the SF-18 generator).
 *
 * Grid arguments are any type with members `nx, ny, nz, dx, dy, dz` (e.g.
 * `macroflow3d::Grid3D`); taking it generically keeps this header free of
 * CUDA includes. Layout of every cell-centred array: `i + nx*(j + ny*k)`.
 */

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace macroflow3d {
namespace closure_gate {

enum class AnalyticField { control2d, lester2021, lester_brk, two_mode, generic3d };

inline const char* analytic_field_name(AnalyticField f) {
    switch (f) {
    case AnalyticField::control2d: return "control2d";
    case AnalyticField::lester2021: return "lester2021";
    case AnalyticField::lester_brk: return "lester_brk";
    case AnalyticField::two_mode: return "two_mode";
    case AnalyticField::generic3d: return "generic3d";
    }
    return "unknown";
}

inline AnalyticField analytic_field_from_name(const std::string& name) {
    if (name == "control2d") return AnalyticField::control2d;
    if (name == "lester2021") return AnalyticField::lester2021;
    if (name == "lester_brk") return AnalyticField::lester_brk;
    if (name == "two_mode") return AnalyticField::two_mode;
    if (name == "generic3d") return AnalyticField::generic3d;
    throw std::invalid_argument("closure_gate: unknown analytic field '" + name + "'");
}

/// Unit-amplitude analytic log-conductivity f(X, Y, Z) of closure_probe.py.
inline double analytic_log_conductivity(AnalyticField f, double X, double Y, double Z) {
    const double TWO_PI = 2.0 * 3.141592653589793; // = 2.0 * np.pi
    switch (f) {
    case AnalyticField::generic3d: // four oblique modes, no symmetry
        return (std::cos(TWO_PI * (X + Y)) + std::cos(TWO_PI * (X + Z) + 0.7) +
                0.8 * std::cos(TWO_PI * (X - Y + Z) + 1.3) +
                0.6 * std::sin(TWO_PI * (2 * X + Y - Z)));
    case AnalyticField::two_mode: // f(theta1, theta2): one continuous symmetry
        return std::cos(TWO_PI * (X + Y)) + std::cos(TWO_PI * (X + Z));
    case AnalyticField::control2d: // independent of x3: 2-D flow, streamlines must close
        return std::cos(TWO_PI * (X + Y)) + 0.8 * std::sin(TWO_PI * (2 * X - Y) + 0.4) +
               0.5 * std::cos(TWO_PI * Y);
    case AnalyticField::lester2021: // Lester et al. (2021) eq. (3.1), coefficient 2/5
        return (std::sin(TWO_PI * X) * std::cos(TWO_PI * Y) * std::sin(TWO_PI * Z) +
                0.4 * std::sin(TWO_PI * X) * std::sin(4 * TWO_PI * Z));
    case AnalyticField::lester_brk: // x1 -> 1/2 - x1 mirror symmetry broken by a phase
        return (std::sin(TWO_PI * X) * std::cos(TWO_PI * Y) * std::sin(TWO_PI * Z) +
                0.4 * std::sin(TWO_PI * X + 0.9) * std::sin(4 * TWO_PI * Z));
    }
    throw std::invalid_argument("closure_gate: invalid AnalyticField");
}

/// Y = eps * f at the cell centres ((i+1/2)dx, (j+1/2)dy, (k+1/2)dz).
template <class GridLike>
void fill_analytic_log_conductivity(const GridLike& grid, AnalyticField f, double eps,
                                    std::vector<double>& Y) {
    if (grid.nx < 1 || grid.ny < 1 || grid.nz < 1) {
        throw std::invalid_argument("closure_gate: fill_analytic_log_conductivity: empty grid");
    }
    const std::size_t nx = static_cast<std::size_t>(grid.nx);
    const std::size_t ny = static_cast<std::size_t>(grid.ny);
    const std::size_t nz = static_cast<std::size_t>(grid.nz);
    Y.assign(nx * ny * nz, 0.0);
    for (std::size_t k = 0; k < nz; ++k) {
        const double z = (static_cast<double>(k) + 0.5) * grid.dz;
        for (std::size_t j = 0; j < ny; ++j) {
            const double y = (static_cast<double>(j) + 0.5) * grid.dy;
            for (std::size_t i = 0; i < nx; ++i) {
                const double x = (static_cast<double>(i) + 0.5) * grid.dx;
                Y[i + nx * (j + ny * k)] = eps * analytic_log_conductivity(f, x, y, z);
            }
        }
    }
}

struct X3AveragedControlReport {
    double raw_mean = 0.0;       ///< mean of the x3-averaged 2-D field before centring
    double raw_variance = 0.0;   ///< population variance (sum/n) of the x3-averaged field
    double applied_scale = 0.0;  ///< sqrt(sigma2 / raw_variance)
};

/**
 * Matched 2-D control (in place): Y <- extrude_k( scale * (Ybar - mean(Ybar)) )
 * with Ybar(i, j) = mean_k Y(i, j, k) and scale such that the population
 * variance (sum/n, SF-18 convention) equals sigma2. The x3-average of an
 * SF-18 realization is exactly its m3 = 0 Fourier plane: a 2-D
 * Gaussian-covariance field with the same correlation length; its Darcy
 * flow is 2-D and its streamlines must close. Throws std::invalid_argument
 * if the averaged field has zero variance.
 */
template <class GridLike>
X3AveragedControlReport make_x3_averaged_control(const GridLike& grid, double sigma2,
                                                 std::vector<double>& Y) {
    if (grid.nx < 1 || grid.ny < 1 || grid.nz < 1) {
        throw std::invalid_argument("closure_gate: make_x3_averaged_control: empty grid");
    }
    if (!(sigma2 > 0.0) || !std::isfinite(sigma2)) {
        throw std::invalid_argument(
            "closure_gate: make_x3_averaged_control: sigma2 must be finite and > 0");
    }
    const std::size_t nx = static_cast<std::size_t>(grid.nx);
    const std::size_t ny = static_cast<std::size_t>(grid.ny);
    const std::size_t nz = static_cast<std::size_t>(grid.nz);
    const std::size_t n2 = nx * ny;
    if (Y.size() != n2 * nz) {
        throw std::invalid_argument(
            "closure_gate: make_x3_averaged_control: Y size != nx*ny*nz");
    }
    std::vector<double> avg(n2, 0.0);
    for (std::size_t j = 0; j < ny; ++j) {
        for (std::size_t i = 0; i < nx; ++i) {
            double sum = 0.0;
            for (std::size_t k = 0; k < nz; ++k) {
                sum += Y[i + nx * (j + ny * k)];
            }
            avg[i + nx * j] = sum / static_cast<double>(nz);
        }
    }
    double mean = 0.0;
    for (std::size_t c = 0; c < n2; ++c) mean += avg[c];
    mean /= static_cast<double>(n2);
    double var = 0.0;
    for (std::size_t c = 0; c < n2; ++c) {
        const double d = avg[c] - mean;
        var += d * d;
    }
    var /= static_cast<double>(n2);
    if (!(var > 0.0) || !std::isfinite(var)) {
        throw std::invalid_argument(
            "closure_gate: make_x3_averaged_control: the x3-averaged field has zero variance");
    }
    X3AveragedControlReport rep;
    rep.raw_mean = mean;
    rep.raw_variance = var;
    rep.applied_scale = std::sqrt(sigma2 / var);
    for (std::size_t c = 0; c < n2; ++c) {
        avg[c] = (avg[c] - mean) * rep.applied_scale;
    }
    for (std::size_t k = 0; k < nz; ++k) {
        for (std::size_t c = 0; c < n2; ++c) {
            Y[c + n2 * k] = avg[c];
        }
    }
    return rep;
}

/// splitmix64 finalizer (Steele, Lea, Flood 2014), exactly as specified.
inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/// u(n) = double(splitmix64(seed_rng + 0x9E3779B97F4A7C15 * (n + 1)) >> 11) * 2^-53, in [0, 1).
inline double seed_uniform(std::uint64_t seed_rng, std::uint64_t n) {
    const std::uint64_t z = splitmix64(seed_rng + 0x9E3779B97F4A7C15ULL * (n + 1ULL));
    return static_cast<double>(z >> 11) * 0x1.0p-53;
}

/// Stateless seed point i: (y0, z0) = (u(2i), u(2i+1)) in [0, 1)^2.
inline std::array<double, 2> seed_point(std::uint64_t seed_rng, std::uint64_t i) {
    return {seed_uniform(seed_rng, 2ULL * i), seed_uniform(seed_rng, 2ULL * i + 1ULL)};
}

} // namespace closure_gate
} // namespace macroflow3d
