#include "reference_operators.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace macroflow3d::streamfunctions::reference {
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

[[nodiscard]] bool finite_positive(double value) {
    return std::isfinite(value) && value > 0.0;
}

void validate_position_and_lengths(const Vec3& position, const Vec3& lengths) {
    if (!std::isfinite(position.x) || !std::isfinite(position.y) || !std::isfinite(position.z)) {
        throw std::invalid_argument("fixture position must be finite");
    }
    if (!finite_positive(lengths.x) || !finite_positive(lengths.y) || !finite_positive(lengths.z)) {
        throw std::invalid_argument("fixture lengths must be finite and positive");
    }
}

[[nodiscard]] double axis_spacing(const Grid& grid, Axis axis) {
    switch (axis) {
        case Axis::x: return grid.spacing.x;
        case Axis::y: return grid.spacing.y;
        case Axis::z: return grid.spacing.z;
    }
    throw std::invalid_argument("unknown axis");
}

[[nodiscard]] std::size_t offset_index(const Grid& grid, std::size_t ix, std::size_t iy,
                                        std::size_t iz, Axis axis, std::ptrdiff_t offset) {
    switch (axis) {
        case Axis::x:
            return grid.index(wrap_index(static_cast<std::ptrdiff_t>(ix) + offset, grid.nx), iy, iz);
        case Axis::y:
            return grid.index(ix, wrap_index(static_cast<std::ptrdiff_t>(iy) + offset, grid.ny), iz);
        case Axis::z:
            return grid.index(ix, iy, wrap_index(static_cast<std::ptrdiff_t>(iz) + offset, grid.nz));
    }
    throw std::invalid_argument("unknown axis");
}

void validate_field(const Grid& grid, const std::vector<double>& field, const char* name,
                    bool require_positive) {
    grid.validate();
    if (field.size() != grid.cell_count()) {
        throw std::invalid_argument(std::string(name) + " size does not match grid cell count");
    }
    for (double value : field) {
        if (!std::isfinite(value) || (require_positive && value <= 0.0)) {
            throw std::invalid_argument(std::string(name) + " contains invalid value");
        }
    }
}

[[nodiscard]] double trigonometric_q(const Vec3& position, const Vec3& lengths) {
    validate_position_and_lengths(position, lengths);
    const double phase = 2.0 * kPi * (position.x / lengths.x + position.y / lengths.y + position.z / lengths.z);
    return 1.25 + 0.25 * std::cos(phase);  // [1, 1.5], strictly positive by construction.
}

}  // namespace

std::size_t Grid::cell_count() const {
    validate();
    if (nx > std::numeric_limits<std::size_t>::max() / ny || nx * ny > std::numeric_limits<std::size_t>::max() / nz) {
        throw std::overflow_error("grid cell count overflows size_t");
    }
    return nx * ny * nz;
}

std::size_t Grid::index(std::size_t ix, std::size_t iy, std::size_t iz) const {
    validate();
    if (ix >= nx || iy >= ny || iz >= nz) {
        throw std::out_of_range("grid index outside extent");
    }
    return ix + nx * (iy + ny * iz);
}

Vec3 Grid::cell_center(std::size_t ix, std::size_t iy, std::size_t iz) const {
    (void)index(ix, iy, iz);
    return {(static_cast<double>(ix) + 0.5) * spacing.x,
            (static_cast<double>(iy) + 0.5) * spacing.y,
            (static_cast<double>(iz) + 0.5) * spacing.z};
}

void Grid::validate() const {
    if (nx == 0 || ny == 0 || nz == 0) {
        throw std::invalid_argument("grid extents must be nonzero");
    }
    if (!finite_positive(spacing.x) || !finite_positive(spacing.y) || !finite_positive(spacing.z)) {
        throw std::invalid_argument("grid spacing must be finite and positive");
    }
}

std::size_t wrap_index(std::ptrdiff_t index, std::size_t extent) {
    if (extent == 0 || extent > static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max())) {
        throw std::invalid_argument("periodic extent is invalid");
    }
    const auto signed_extent = static_cast<std::ptrdiff_t>(extent);
    const auto wrapped = index % signed_extent;
    return static_cast<std::size_t>(wrapped < 0 ? wrapped + signed_extent : wrapped);
}

double harmonic_mean_q(double q_cell, double q_neighbor) {
    if (!finite_positive(q_cell) || !finite_positive(q_neighbor)) {
        throw std::invalid_argument("harmonic mean requires finite positive q values");
    }
    return 2.0 * q_cell * q_neighbor / (q_cell + q_neighbor);
}

std::vector<double> centered_first(const Grid& grid, const std::vector<double>& field, Axis axis) {
    validate_field(grid, field, "field", false);
    std::vector<double> result(grid.cell_count());
    const double inverse_two_h = 1.0 / (2.0 * axis_spacing(grid, axis));
    for (std::size_t iz = 0; iz < grid.nz; ++iz) for (std::size_t iy = 0; iy < grid.ny; ++iy) for (std::size_t ix = 0; ix < grid.nx; ++ix) {
        const auto center = grid.index(ix, iy, iz);
        result[center] = (field[offset_index(grid, ix, iy, iz, axis, 1)] - field[offset_index(grid, ix, iy, iz, axis, -1)]) * inverse_two_h;
    }
    return result;
}

std::vector<double> centered_second(const Grid& grid, const std::vector<double>& field, Axis axis) {
    validate_field(grid, field, "field", false);
    std::vector<double> result(grid.cell_count());
    const double inverse_h_squared = 1.0 / std::pow(axis_spacing(grid, axis), 2.0);
    for (std::size_t iz = 0; iz < grid.nz; ++iz) for (std::size_t iy = 0; iy < grid.ny; ++iy) for (std::size_t ix = 0; ix < grid.nx; ++ix) {
        const auto center = grid.index(ix, iy, iz);
        result[center] = (field[offset_index(grid, ix, iy, iz, axis, 1)] - 2.0 * field[center] + field[offset_index(grid, ix, iy, iz, axis, -1)]) * inverse_h_squared;
    }
    return result;
}

std::vector<double> centered_mixed(const Grid& grid, const std::vector<double>& field, Axis first, Axis second) {
    validate_field(grid, field, "field", false);
    if (first == second) {
        throw std::invalid_argument("mixed derivative requires two distinct axes");
    }
    std::vector<double> result(grid.cell_count());
    const double denominator = 4.0 * axis_spacing(grid, first) * axis_spacing(grid, second);
    for (std::size_t iz = 0; iz < grid.nz; ++iz) for (std::size_t iy = 0; iy < grid.ny; ++iy) for (std::size_t ix = 0; ix < grid.nx; ++ix) {
        const auto sample = [&](std::ptrdiff_t first_offset, std::ptrdiff_t second_offset) {
            std::size_t sx = ix, sy = iy, sz = iz;
            const auto apply = [&](Axis axis, std::ptrdiff_t delta) {
                switch (axis) {
                    case Axis::x: sx = wrap_index(static_cast<std::ptrdiff_t>(sx) + delta, grid.nx); break;
                    case Axis::y: sy = wrap_index(static_cast<std::ptrdiff_t>(sy) + delta, grid.ny); break;
                    case Axis::z: sz = wrap_index(static_cast<std::ptrdiff_t>(sz) + delta, grid.nz); break;
                }
            };
            apply(first, first_offset);
            apply(second, second_offset);
            return field[grid.index(sx, sy, sz)];
        };
        result[grid.index(ix, iy, iz)] = (sample(1, 1) - sample(1, -1) - sample(-1, 1) + sample(-1, -1)) / denominator;
    }
    return result;
}

std::vector<double> divergence_form_diffusion(const Grid& grid, const std::vector<double>& q,
                                              const std::vector<double>& u) {
    validate_field(grid, q, "q", true);
    validate_field(grid, u, "u", false);
    std::vector<double> result(grid.cell_count());
    for (std::size_t iz = 0; iz < grid.nz; ++iz) for (std::size_t iy = 0; iy < grid.ny; ++iy) for (std::size_t ix = 0; ix < grid.nx; ++ix) {
        const auto center = grid.index(ix, iy, iz);
        double value = 0.0;
        for (Axis axis : {Axis::x, Axis::y, Axis::z}) {
            const auto plus = offset_index(grid, ix, iy, iz, axis, 1);
            const auto minus = offset_index(grid, ix, iy, iz, axis, -1);
            const double h2 = std::pow(axis_spacing(grid, axis), 2.0);
            value += (harmonic_mean_q(q[center], q[plus]) * (u[center] - u[plus]) +
                      harmonic_mean_q(q[center], q[minus]) * (u[center] - u[minus])) / h2;
        }
        result[center] = value;
    }
    return result;
}

Vec3 cross(const Vec3& left, const Vec3& right) {
    if (!std::isfinite(left.x) || !std::isfinite(left.y) || !std::isfinite(left.z) || !std::isfinite(right.x) || !std::isfinite(right.y) || !std::isfinite(right.z)) {
        throw std::invalid_argument("cross product requires finite vectors");
    }
    const Vec3 result{left.y * right.z - left.z * right.y, left.z * right.x - left.x * right.z,
                      left.x * right.y - left.y * right.x};
    if (!std::isfinite(result.x) || !std::isfinite(result.y) || !std::isfinite(result.z)) {
        throw std::invalid_argument("cross product result is not finite");
    }
    return result;
}

double rms_norm(const std::vector<double>& values) {
    if (values.empty()) throw std::invalid_argument("RMS norm requires nonempty values");
    double sum = 0.0;
    for (double value : values) {
        if (!std::isfinite(value)) throw std::invalid_argument("RMS norm requires finite values");
        sum += value * value;
    }
    return std::sqrt(sum / static_cast<double>(values.size()));
}

double linf_norm(const std::vector<double>& values) {
    if (values.empty()) throw std::invalid_argument("Linf norm requires nonempty values");
    double maximum = 0.0;
    for (double value : values) {
        if (!std::isfinite(value)) throw std::invalid_argument("Linf norm requires finite values");
        maximum = std::max(maximum, std::abs(value));
    }
    return maximum;
}

ObservedOrder observed_order(double coarse_error, double fine_error, double coarse_spacing, double fine_spacing) {
    if (!std::isfinite(coarse_error) || !std::isfinite(fine_error) || !std::isfinite(coarse_spacing) || !std::isfinite(fine_spacing)) {
        return {ObservedOrderStatus::non_finite_error, 0.0, "errors and spacings must be finite"};
    }
    if (coarse_error <= 0.0 || fine_error <= 0.0) {
        return {ObservedOrderStatus::non_positive_error, 0.0, "errors must be strictly positive"};
    }
    if (coarse_spacing <= 0.0 || fine_spacing <= 0.0 || coarse_spacing == fine_spacing) {
        return {ObservedOrderStatus::invalid_refinement_ratio, 0.0, "spacings must be positive and distinct"};
    }
    return {ObservedOrderStatus::valid, std::log(coarse_error / fine_error) / std::log(coarse_spacing / fine_spacing), ""};
}

double trigonometric_scalar(const Vec3& position, const Vec3& lengths) {
    validate_position_and_lengths(position, lengths);
    const double ax = 2.0 * kPi * position.x / lengths.x;
    const double ay = 2.0 * kPi * position.y / lengths.y;
    const double az = 2.0 * kPi * position.z / lengths.z;
    return std::sin(ax) + 0.5 * std::cos(2.0 * ay) - 0.25 * std::sin(3.0 * az) + 0.125 * std::sin(ax + ay - az);
}

Vec3 trigonometric_gradient(const Vec3& position, const Vec3& lengths) {
    validate_position_and_lengths(position, lengths);
    const double ax = 2.0 * kPi * position.x / lengths.x;
    const double ay = 2.0 * kPi * position.y / lengths.y;
    const double az = 2.0 * kPi * position.z / lengths.z;
    const double mixed = std::cos(ax + ay - az);
    return {(2.0 * kPi / lengths.x) * (std::cos(ax) + 0.125 * mixed),
            (2.0 * kPi / lengths.y) * (-std::sin(2.0 * ay) + 0.125 * mixed),
            (2.0 * kPi / lengths.z) * (-0.75 * std::cos(3.0 * az) - 0.125 * mixed)};
}

double trigonometric_laplacian(const Vec3& position, const Vec3& lengths) {
    validate_position_and_lengths(position, lengths);
    const double ax = 2.0 * kPi * position.x / lengths.x;
    const double ay = 2.0 * kPi * position.y / lengths.y;
    const double az = 2.0 * kPi * position.z / lengths.z;
    const double kx = 2.0 * kPi / lengths.x, ky = 2.0 * kPi / lengths.y, kz = 2.0 * kPi / lengths.z;
    return -kx * kx * std::sin(ax) - 0.5 * (2.0 * ky) * (2.0 * ky) * std::cos(2.0 * ay) + 0.25 * (3.0 * kz) * (3.0 * kz) * std::sin(3.0 * az) - 0.125 * (kx * kx + ky * ky + kz * kz) * std::sin(ax + ay - az);
}

TrigonometricFixture make_cubic_trigonometric_fixture(std::size_t cells_per_axis) {
    if (cells_per_axis != 16 && cells_per_axis != 32) throw std::invalid_argument("cubic fixture supports 16^3 or 32^3");
    const double h = 1.0 / static_cast<double>(cells_per_axis);
    TrigonometricFixture fixture{{cells_per_axis, cells_per_axis, cells_per_axis, {h, h, h}}, {1.0, 1.0, 1.0}, {}, {}};
    fixture.scalar.resize(fixture.grid.cell_count()); fixture.q.resize(fixture.grid.cell_count());
    for (std::size_t iz = 0; iz < fixture.grid.nz; ++iz) for (std::size_t iy = 0; iy < fixture.grid.ny; ++iy) for (std::size_t ix = 0; ix < fixture.grid.nx; ++ix) {
        const auto id = fixture.grid.index(ix, iy, iz); const auto position = fixture.grid.cell_center(ix, iy, iz);
        fixture.scalar[id] = trigonometric_scalar(position, fixture.lengths); fixture.q[id] = trigonometric_q(position, fixture.lengths);
    }
    return fixture;
}

TrigonometricFixture make_anisotropic_trigonometric_fixture(bool fine) {
    const double h = fine ? 1.0 / 32.0 : 1.0 / 16.0;
    TrigonometricFixture fixture{{fine ? 32U : 16U, fine ? 48U : 24U, fine ? 64U : 32U, {h, h, h}}, {1.0, 1.5, 2.0}, {}, {}};
    fixture.scalar.resize(fixture.grid.cell_count()); fixture.q.resize(fixture.grid.cell_count());
    for (std::size_t iz = 0; iz < fixture.grid.nz; ++iz) for (std::size_t iy = 0; iy < fixture.grid.ny; ++iy) for (std::size_t ix = 0; ix < fixture.grid.nx; ++ix) {
        const auto id = fixture.grid.index(ix, iy, iz); const auto position = fixture.grid.cell_center(ix, iy, iz);
        fixture.scalar[id] = trigonometric_scalar(position, fixture.lengths); fixture.q[id] = trigonometric_q(position, fixture.lengths);
    }
    return fixture;
}

}  // namespace macroflow3d::streamfunctions::reference
