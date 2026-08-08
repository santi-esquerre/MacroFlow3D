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

void validate_vector(const Vec3& value, const char* name) {
    if (!std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z)) {
        throw std::invalid_argument(std::string(name) + " must be finite");
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

void validate_vector_field(const Grid& grid, const VectorField& field, const char* name) {
    validate_field(grid, field.x, (std::string(name) + ".x").c_str(), false);
    validate_field(grid, field.y, (std::string(name) + ".y").c_str(), false);
    validate_field(grid, field.z, (std::string(name) + ".z").c_str(), false);
}

void validate_total_gradient_fixture(const TotalGradientFixture& fixture) {
    fixture.grid.validate();
    validate_position_and_lengths({0.0, 0.0, 0.0}, fixture.lengths);
    validate_vector(fixture.psi1_affine_gradient, "psi1 affine gradient");
    validate_vector(fixture.psi2_affine_gradient, "psi2 affine gradient");
    validate_field(fixture.grid, fixture.psi1_fluctuation, "psi1 fluctuation", false);
    validate_field(fixture.grid, fixture.psi2_fluctuation, "psi2 fluctuation", false);
}

[[nodiscard]] VectorField make_vector_field(std::size_t size) {
    return {std::vector<double>(size, 0.0), std::vector<double>(size, 0.0),
            std::vector<double>(size, 0.0)};
}

void set_vector(VectorField& field, std::size_t index, const Vec3& value) {
    field.x[index] = value.x;
    field.y[index] = value.y;
    field.z[index] = value.z;
}

[[nodiscard]] Vec3 subtract(const Vec3& left, const Vec3& right) {
    const Vec3 result{left.x - right.x, left.y - right.y, left.z - right.z};
    validate_vector(result, "vector difference");
    return result;
}

[[nodiscard]] double dot(const Vec3& left, const Vec3& right) {
    if (!std::isfinite(left.x) || !std::isfinite(left.y) || !std::isfinite(left.z) ||
        !std::isfinite(right.x) || !std::isfinite(right.y) || !std::isfinite(right.z)) {
        throw std::invalid_argument("dot product requires finite vectors");
    }
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

// Long-double vector used only by the independent SF-09 discrete oracle so it
// stays decoupled from the double-precision Vec3 used everywhere else.
struct Vec3Ld {
    long double x{};
    long double y{};
    long double z{};
};

[[nodiscard]] Vec3Ld cross_ld(const Vec3Ld& left, const Vec3Ld& right) {
    return {left.y * right.z - left.z * right.y, left.z * right.x - left.x * right.z,
            left.x * right.y - left.y * right.x};
}

[[nodiscard]] long double dot_ld(const Vec3Ld& left, const Vec3Ld& right) {
    return left.x * right.x + left.y * right.y + left.z * right.z;
}

void validate_nonlinear_source_config(const NonlinearSourceReferenceConfig& config) {
    if (!std::isfinite(config.epsilon) || config.epsilon < 0.0) {
        throw std::invalid_argument("nonlinear source epsilon must be finite and nonnegative");
    }
    if (!finite_positive(config.v_rms)) {
        throw std::invalid_argument("nonlinear source v_rms must be finite and strictly positive");
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

AffineRhsFixture make_affine_rhs_fixture(std::size_t cells_per_axis, bool constant_q) {
    if (cells_per_axis != 16 && cells_per_axis != 32) {
        throw std::invalid_argument("affine RHS fixture supports 16^3 or 32^3");
    }
    const double h = 1.0 / static_cast<double>(cells_per_axis);
    AffineRhsFixture fixture{{cells_per_axis, cells_per_axis, cells_per_axis, {h, h, h}}, {}};
    fixture.q.resize(fixture.grid.cell_count());
    for (std::size_t iz = 0; iz < cells_per_axis; ++iz) for (std::size_t iy = 0; iy < cells_per_axis; ++iy) for (std::size_t ix = 0; ix < cells_per_axis; ++ix) {
        const Vec3 p = fixture.grid.cell_center(ix, iy, iz);
        fixture.q[fixture.grid.index(ix, iy, iz)] = constant_q ? 1.7 :
            1.40 + 0.15 * std::sin(2.0 * kPi * p.x) + 0.10 * std::cos(2.0 * kPi * p.y) +
            0.05 * std::sin(2.0 * kPi * p.z);
    }
    return fixture;
}

std::vector<double> affine_rhs_discrete(const Grid& grid, const std::vector<double>& q,
                                        const Vec3& gradient) {
    validate_field(grid, q, "q", true);
    std::vector<double> result(grid.cell_count());
    for (std::size_t iz = 0; iz < grid.nz; ++iz) for (std::size_t iy = 0; iy < grid.ny; ++iy) for (std::size_t ix = 0; ix < grid.nx; ++ix) {
        const auto c = grid.index(ix, iy, iz);
        const auto xp = grid.index(wrap_index(static_cast<std::ptrdiff_t>(ix) + 1, grid.nx), iy, iz);
        const auto xm = grid.index(wrap_index(static_cast<std::ptrdiff_t>(ix) - 1, grid.nx), iy, iz);
        const auto yp = grid.index(ix, wrap_index(static_cast<std::ptrdiff_t>(iy) + 1, grid.ny), iz);
        const auto ym = grid.index(ix, wrap_index(static_cast<std::ptrdiff_t>(iy) - 1, grid.ny), iz);
        const auto zp = grid.index(ix, iy, wrap_index(static_cast<std::ptrdiff_t>(iz) + 1, grid.nz));
        const auto zm = grid.index(ix, iy, wrap_index(static_cast<std::ptrdiff_t>(iz) - 1, grid.nz));
        const long double qc = q[c];
        const auto harmonic = [qc](double neighbor) {
            const long double qn = neighbor;
            return 2.0L * qc * qn / (qc + qn);
        };
        const long double dx = harmonic(q[xp]) - harmonic(q[xm]);
        const long double dy = harmonic(q[yp]) - harmonic(q[ym]);
        const long double dz = harmonic(q[zp]) - harmonic(q[zm]);
        result[c] = static_cast<double>((dx * gradient.x + dy * gradient.y + dz * gradient.z) /
                                        static_cast<long double>(grid.spacing.x));
    }
    return result;
}

std::vector<double> affine_rhs_continuous(const Grid& grid, const Vec3& gradient) {
    grid.validate();
    std::vector<double> result(grid.cell_count());
    for (std::size_t iz = 0; iz < grid.nz; ++iz) for (std::size_t iy = 0; iy < grid.ny; ++iy) for (std::size_t ix = 0; ix < grid.nx; ++ix) {
        const Vec3 p = grid.cell_center(ix, iy, iz);
        const Vec3 grad_q{0.30 * kPi * std::cos(2.0 * kPi * p.x),
                          -0.20 * kPi * std::sin(2.0 * kPi * p.y),
                          0.10 * kPi * std::cos(2.0 * kPi * p.z)};
        result[grid.index(ix, iy, iz)] = gradient.x * grad_q.x + gradient.y * grad_q.y + gradient.z * grad_q.z;
    }
    return result;
}

long double long_double_mean(const std::vector<double>& values) {
    if (values.empty()) throw std::invalid_argument("mean requires nonempty values");
    long double sum = 0.0L;
    for (double value : values) {
        if (!std::isfinite(value)) throw std::invalid_argument("mean requires finite values");
        sum += static_cast<long double>(value);
    }
    return sum / static_cast<long double>(values.size());
}

std::vector<double> mean_zero_projected(const std::vector<double>& values) {
    const long double mean = long_double_mean(values);
    std::vector<double> result(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) result[i] = static_cast<double>(static_cast<long double>(values[i]) - mean);
    return result;
}

TotalGradientFixture make_total_gradient_fixture(std::size_t cells_per_axis) {
    if (cells_per_axis != 16 && cells_per_axis != 32 && cells_per_axis != 64) {
        throw std::invalid_argument("total-gradient fixture supports 16^3, 32^3, or 64^3");
    }
    const Vec3 lengths{1.0, 1.5, 2.25};
    const Grid grid{cells_per_axis, cells_per_axis, cells_per_axis,
                    {lengths.x / static_cast<double>(cells_per_axis),
                     lengths.y / static_cast<double>(cells_per_axis),
                     lengths.z / static_cast<double>(cells_per_axis)}};
    TotalGradientFixture fixture{grid, lengths, {0.7, -1.1, 0.35},
                                 {-0.45, 0.6, 1.3}, {}, {}};
    fixture.psi1_fluctuation.resize(grid.cell_count());
    fixture.psi2_fluctuation.resize(grid.cell_count());
    for (std::size_t iz = 0; iz < grid.nz; ++iz) for (std::size_t iy = 0; iy < grid.ny; ++iy) for (std::size_t ix = 0; ix < grid.nx; ++ix) {
        const std::size_t id = grid.index(ix, iy, iz);
        const Vec3 position = grid.cell_center(ix, iy, iz);
        fixture.psi1_fluctuation[id] = total_gradient_periodic_scalar(
            GradientFixtureField::psi1, position, lengths);
        fixture.psi2_fluctuation[id] = total_gradient_periodic_scalar(
            GradientFixtureField::psi2, position, lengths);
    }
    return fixture;
}

TotalGradientFixture make_pure_affine_total_gradient_fixture(std::size_t cells_per_axis) {
    TotalGradientFixture fixture = make_total_gradient_fixture(cells_per_axis);
    std::fill(fixture.psi1_fluctuation.begin(), fixture.psi1_fluctuation.end(), 0.0);
    std::fill(fixture.psi2_fluctuation.begin(), fixture.psi2_fluctuation.end(), 0.0);
    return fixture;
}

double total_gradient_periodic_scalar(GradientFixtureField field, const Vec3& position,
                                      const Vec3& lengths) {
    validate_position_and_lengths(position, lengths);
    const double ax = 2.0 * kPi * position.x / lengths.x;
    const double ay = 2.0 * kPi * position.y / lengths.y;
    const double az = 2.0 * kPi * position.z / lengths.z;
    switch (field) {
        case GradientFixtureField::psi1:
            return std::sin(ax) + 0.35 * std::cos(2.0 * ay) - 0.20 * std::sin(3.0 * az) +
                   0.15 * std::sin(ax + ay - az);
        case GradientFixtureField::psi2:
            return 0.60 * std::cos(2.0 * ax) + 0.25 * std::sin(ay) + 0.30 * std::cos(2.0 * az) -
                   0.10 * std::cos(ax - 2.0 * ay + az);
    }
    throw std::invalid_argument("unknown total-gradient fixture field");
}

Vec3 total_gradient_periodic_analytic(GradientFixtureField field, const Vec3& position,
                                      const Vec3& lengths) {
    validate_position_and_lengths(position, lengths);
    const double ax = 2.0 * kPi * position.x / lengths.x;
    const double ay = 2.0 * kPi * position.y / lengths.y;
    const double az = 2.0 * kPi * position.z / lengths.z;
    const double kx = 2.0 * kPi / lengths.x;
    const double ky = 2.0 * kPi / lengths.y;
    const double kz = 2.0 * kPi / lengths.z;
    switch (field) {
        case GradientFixtureField::psi1: {
            const double mixed = std::cos(ax + ay - az);
            return {kx * (std::cos(ax) + 0.15 * mixed),
                    ky * (-0.70 * std::sin(2.0 * ay) + 0.15 * mixed),
                    kz * (-0.60 * std::cos(3.0 * az) - 0.15 * mixed)};
        }
        case GradientFixtureField::psi2: {
            const double mixed = std::sin(ax - 2.0 * ay + az);
            return {kx * (-1.20 * std::sin(2.0 * ax) + 0.10 * mixed),
                    ky * (0.25 * std::cos(ay) - 0.20 * mixed),
                    kz * (-0.60 * std::sin(2.0 * az) + 0.10 * mixed)};
        }
    }
    throw std::invalid_argument("unknown total-gradient fixture field");
}

Vec3 total_gradient_analytic(GradientFixtureField field, const Vec3& position,
                             const Vec3& lengths, const Vec3& affine_gradient) {
    validate_vector(affine_gradient, "affine gradient");
    const Vec3 periodic = total_gradient_periodic_analytic(field, position, lengths);
    return {periodic.x + affine_gradient.x, periodic.y + affine_gradient.y,
            periodic.z + affine_gradient.z};
}

VectorField centered_total_gradient_oracle(const Grid& grid,
                                           const std::vector<double>& fluctuation,
                                           const Vec3& affine_gradient) {
    validate_field(grid, fluctuation, "fluctuation", false);
    validate_vector(affine_gradient, "affine gradient");
    VectorField result{{}, {}, {}};
    result.x.resize(grid.cell_count());
    result.y.resize(grid.cell_count());
    result.z.resize(grid.cell_count());
    const long double inverse_two_dx = 1.0L / (2.0L * static_cast<long double>(grid.spacing.x));
    const long double inverse_two_dy = 1.0L / (2.0L * static_cast<long double>(grid.spacing.y));
    const long double inverse_two_dz = 1.0L / (2.0L * static_cast<long double>(grid.spacing.z));
    for (std::size_t iz = 0; iz < grid.nz; ++iz) for (std::size_t iy = 0; iy < grid.ny; ++iy) for (std::size_t ix = 0; ix < grid.nx; ++ix) {
        const std::size_t id = grid.index(ix, iy, iz);
        const std::size_t xp = grid.index(wrap_index(static_cast<std::ptrdiff_t>(ix) + 1, grid.nx), iy, iz);
        const std::size_t xm = grid.index(wrap_index(static_cast<std::ptrdiff_t>(ix) - 1, grid.nx), iy, iz);
        const std::size_t yp = grid.index(ix, wrap_index(static_cast<std::ptrdiff_t>(iy) + 1, grid.ny), iz);
        const std::size_t ym = grid.index(ix, wrap_index(static_cast<std::ptrdiff_t>(iy) - 1, grid.ny), iz);
        const std::size_t zp = grid.index(ix, iy, wrap_index(static_cast<std::ptrdiff_t>(iz) + 1, grid.nz));
        const std::size_t zm = grid.index(ix, iy, wrap_index(static_cast<std::ptrdiff_t>(iz) - 1, grid.nz));
        result.x[id] = static_cast<double>((static_cast<long double>(fluctuation[xp]) -
                                            static_cast<long double>(fluctuation[xm])) * inverse_two_dx +
                                           static_cast<long double>(affine_gradient.x));
        result.y[id] = static_cast<double>((static_cast<long double>(fluctuation[yp]) -
                                            static_cast<long double>(fluctuation[ym])) * inverse_two_dy +
                                           static_cast<long double>(affine_gradient.y));
        result.z[id] = static_cast<double>((static_cast<long double>(fluctuation[zp]) -
                                            static_cast<long double>(fluctuation[zm])) * inverse_two_dz +
                                           static_cast<long double>(affine_gradient.z));
    }
    return result;
}

SymmetricHessian total_gradient_periodic_hessian_analytic(GradientFixtureField field,
                                                           const Vec3& position,
                                                           const Vec3& lengths) {
    validate_position_and_lengths(position, lengths);
    const double ax = 2.0 * kPi * position.x / lengths.x;
    const double ay = 2.0 * kPi * position.y / lengths.y;
    const double az = 2.0 * kPi * position.z / lengths.z;
    const double kx = 2.0 * kPi / lengths.x;
    const double ky = 2.0 * kPi / lengths.y;
    const double kz = 2.0 * kPi / lengths.z;

    SymmetricHessian result{};
    switch (field) {
        case GradientFixtureField::psi1: {
            const double mixed_sine = std::sin(ax + ay - az);
            result = {-kx * kx * std::sin(ax) - 0.15 * kx * kx * mixed_sine,
                      -0.15 * kx * ky * mixed_sine,
                      0.15 * kx * kz * mixed_sine,
                      -0.35 * (2.0 * ky) * (2.0 * ky) * std::cos(2.0 * ay) -
                          0.15 * ky * ky * mixed_sine,
                      0.15 * ky * kz * mixed_sine,
                      0.20 * (3.0 * kz) * (3.0 * kz) * std::sin(3.0 * az) -
                          0.15 * kz * kz * mixed_sine};
            break;
        }
        case GradientFixtureField::psi2: {
            const double mixed_cosine = std::cos(ax - 2.0 * ay + az);
            result = {-0.60 * (2.0 * kx) * (2.0 * kx) * std::cos(2.0 * ax) +
                          0.10 * kx * kx * mixed_cosine,
                      -0.20 * kx * ky * mixed_cosine,
                      0.10 * kx * kz * mixed_cosine,
                      -0.25 * ky * ky * std::sin(ay) + 0.40 * ky * ky * mixed_cosine,
                      -0.20 * ky * kz * mixed_cosine,
                      -0.30 * (2.0 * kz) * (2.0 * kz) * std::cos(2.0 * az) +
                          0.10 * kz * kz * mixed_cosine};
            break;
        }
    }
    if (!std::isfinite(result.xx) || !std::isfinite(result.xy) || !std::isfinite(result.xz) ||
        !std::isfinite(result.yy) || !std::isfinite(result.yz) || !std::isfinite(result.zz)) {
        throw std::invalid_argument("analytic periodic Hessian is not finite");
    }
    return result;
}

Vec3 symmetric_hessian_vector_product(const SymmetricHessian& hessian,
                                      const Vec3& gradient) {
    if (!std::isfinite(hessian.xx) || !std::isfinite(hessian.xy) ||
        !std::isfinite(hessian.xz) || !std::isfinite(hessian.yy) ||
        !std::isfinite(hessian.yz) || !std::isfinite(hessian.zz)) {
        throw std::invalid_argument("symmetric Hessian must be finite");
    }
    validate_vector(gradient, "Hessian-vector gradient");
    const Vec3 result{hessian.xx * gradient.x + hessian.xy * gradient.y + hessian.xz * gradient.z,
                      hessian.xy * gradient.x + hessian.yy * gradient.y + hessian.yz * gradient.z,
                      hessian.xz * gradient.x + hessian.yz * gradient.y + hessian.zz * gradient.z};
    validate_vector(result, "Hessian-vector product");
    return result;
}

HessianVectorBFields analytic_hessian_vector_b(const TotalGradientFixture& fixture) {
    validate_total_gradient_fixture(fixture);
    const std::size_t cells = fixture.grid.cell_count();
    HessianVectorBFields result{make_vector_field(cells), make_vector_field(cells),
                                make_vector_field(cells)};
    for (std::size_t iz = 0; iz < fixture.grid.nz; ++iz) for (std::size_t iy = 0;
         iy < fixture.grid.ny; ++iy) for (std::size_t ix = 0; ix < fixture.grid.nx; ++ix) {
        const std::size_t id = fixture.grid.index(ix, iy, iz);
        const Vec3 position = fixture.grid.cell_center(ix, iy, iz);
        const Vec3 g1 = total_gradient_analytic(GradientFixtureField::psi1, position,
                                                 fixture.lengths, fixture.psi1_affine_gradient);
        const Vec3 g2 = total_gradient_analytic(GradientFixtureField::psi2, position,
                                                 fixture.lengths, fixture.psi2_affine_gradient);
        const Vec3 h2g1 = symmetric_hessian_vector_product(
            total_gradient_periodic_hessian_analytic(GradientFixtureField::psi2, position,
                                                      fixture.lengths),
            g1);
        const Vec3 h1g2 = symmetric_hessian_vector_product(
            total_gradient_periodic_hessian_analytic(GradientFixtureField::psi1, position,
                                                      fixture.lengths),
            g2);
        set_vector(result.hessian_psi2_times_gradient_psi1, id, h2g1);
        set_vector(result.hessian_psi1_times_gradient_psi2, id, h1g2);
        set_vector(result.b, id, subtract(h2g1, h1g2));
    }
    return result;
}

HessianVectorBFields centered_hessian_vector_b_oracle(
    const Grid& grid, const std::vector<double>& psi1_fluctuation,
    const std::vector<double>& psi2_fluctuation, const VectorField& psi1_total_gradient,
    const VectorField& psi2_total_gradient) {
    validate_field(grid, psi1_fluctuation, "psi1 fluctuation", false);
    validate_field(grid, psi2_fluctuation, "psi2 fluctuation", false);
    validate_vector_field(grid, psi1_total_gradient, "psi1 total gradient");
    validate_vector_field(grid, psi2_total_gradient, "psi2 total gradient");

    const std::size_t cells = grid.cell_count();
    HessianVectorBFields result{make_vector_field(cells), make_vector_field(cells),
                                make_vector_field(cells)};
    const long double inverse_dx2 = 1.0L / (static_cast<long double>(grid.spacing.x) * grid.spacing.x);
    const long double inverse_dy2 = 1.0L / (static_cast<long double>(grid.spacing.y) * grid.spacing.y);
    const long double inverse_dz2 = 1.0L / (static_cast<long double>(grid.spacing.z) * grid.spacing.z);
    const long double inverse_4dxdy = 1.0L / (4.0L * grid.spacing.x * grid.spacing.y);
    const long double inverse_4dxdz = 1.0L / (4.0L * grid.spacing.x * grid.spacing.z);
    const long double inverse_4dydz = 1.0L / (4.0L * grid.spacing.y * grid.spacing.z);

    for (std::size_t iz = 0; iz < grid.nz; ++iz) for (std::size_t iy = 0;
         iy < grid.ny; ++iy) for (std::size_t ix = 0; ix < grid.nx; ++ix) {
        const std::size_t id = grid.index(ix, iy, iz);
        const auto cell_at = [&](std::ptrdiff_t dx, std::ptrdiff_t dy, std::ptrdiff_t dz) {
            return grid.index(wrap_index(static_cast<std::ptrdiff_t>(ix) + dx, grid.nx),
                              wrap_index(static_cast<std::ptrdiff_t>(iy) + dy, grid.ny),
                              wrap_index(static_cast<std::ptrdiff_t>(iz) + dz, grid.nz));
        };

        const auto direct_hvp = [&](const std::vector<double>& fluctuation, const Vec3& gradient) {
            const auto value = [&](std::ptrdiff_t dx, std::ptrdiff_t dy, std::ptrdiff_t dz) {
                return static_cast<long double>(fluctuation[cell_at(dx, dy, dz)]);
            };
            const long double center = value(0, 0, 0);
            const long double xx = (value(1, 0, 0) - 2.0L * center + value(-1, 0, 0)) * inverse_dx2;
            const long double yy = (value(0, 1, 0) - 2.0L * center + value(0, -1, 0)) * inverse_dy2;
            const long double zz = (value(0, 0, 1) - 2.0L * center + value(0, 0, -1)) * inverse_dz2;
            const long double xy = (value(1, 1, 0) - value(1, -1, 0) - value(-1, 1, 0) +
                                    value(-1, -1, 0)) * inverse_4dxdy;
            const long double xz = (value(1, 0, 1) - value(1, 0, -1) - value(-1, 0, 1) +
                                    value(-1, 0, -1)) * inverse_4dxdz;
            const long double yz = (value(0, 1, 1) - value(0, 1, -1) - value(0, -1, 1) +
                                    value(0, -1, -1)) * inverse_4dydz;
            return Vec3{static_cast<double>(xx * gradient.x + xy * gradient.y + xz * gradient.z),
                        static_cast<double>(xy * gradient.x + yy * gradient.y + yz * gradient.z),
                        static_cast<double>(xz * gradient.x + yz * gradient.y + zz * gradient.z)};
        };

        const Vec3 g1{psi1_total_gradient.x[id], psi1_total_gradient.y[id],
                      psi1_total_gradient.z[id]};
        const Vec3 g2{psi2_total_gradient.x[id], psi2_total_gradient.y[id],
                      psi2_total_gradient.z[id]};
        const Vec3 h2g1 = direct_hvp(psi2_fluctuation, g1);
        const Vec3 h1g2 = direct_hvp(psi1_fluctuation, g2);
        set_vector(result.hessian_psi2_times_gradient_psi1, id, h2g1);
        set_vector(result.hessian_psi1_times_gradient_psi2, id, h1g2);
        set_vector(result.b, id, subtract(h2g1, h1g2));
    }
    return result;
}

TotalGradientFixture make_parallel_total_gradient_fixture(std::size_t cells_per_axis,
                                                           double scale) {
    if (!std::isfinite(scale)) {
        throw std::invalid_argument("parallel total-gradient scale must be finite");
    }
    TotalGradientFixture fixture = make_total_gradient_fixture(cells_per_axis);
    fixture.psi2_affine_gradient = {scale * fixture.psi1_affine_gradient.x,
                                    scale * fixture.psi1_affine_gradient.y,
                                    scale * fixture.psi1_affine_gradient.z};
    validate_vector(fixture.psi2_affine_gradient, "parallel psi2 affine gradient");
    for (std::size_t id = 0; id < fixture.grid.cell_count(); ++id) {
        fixture.psi2_fluctuation[id] = scale * fixture.psi1_fluctuation[id];
    }
    return fixture;
}

NonlinearSourceFields analytic_nonlinear_source_reference(
    const TotalGradientFixture& fixture, const NonlinearSourceReferenceConfig& config) {
    validate_total_gradient_fixture(fixture);
    validate_nonlinear_source_config(config);
    const HessianVectorBFields hvb = analytic_hessian_vector_b(fixture);
    const std::size_t cells = fixture.grid.cell_count();
    NonlinearSourceFields result{make_vector_field(cells), std::vector<double>(cells),
                                 std::vector<double>(cells), std::vector<double>(cells)};
    const double regularization = config.epsilon * config.v_rms;
    const double regularization_sq = regularization * regularization;
    for (std::size_t iz = 0; iz < fixture.grid.nz; ++iz) for (std::size_t iy = 0;
         iy < fixture.grid.ny; ++iy) for (std::size_t ix = 0; ix < fixture.grid.nx; ++ix) {
        const std::size_t id = fixture.grid.index(ix, iy, iz);
        const Vec3 position = fixture.grid.cell_center(ix, iy, iz);
        const Vec3 g1 = total_gradient_analytic(GradientFixtureField::psi1, position,
                                                 fixture.lengths, fixture.psi1_affine_gradient);
        const Vec3 g2 = total_gradient_analytic(GradientFixtureField::psi2, position,
                                                 fixture.lengths, fixture.psi2_affine_gradient);
        const Vec3 b{hvb.b.x[id], hvb.b.y[id], hvb.b.z[id]};
        const Vec3 c = cross(g1, g2);
        const double d = c.x * c.x + c.y * c.y + c.z * c.z + regularization_sq;
        const double s1 = dot(cross(b, g1), c) / d;
        const double s2 = dot(cross(b, g2), c) / d;
        set_vector(result.c, id, c);
        result.denominator[id] = d;
        result.s1[id] = s1;
        result.s2[id] = s2;
    }
    return result;
}

NonlinearSourceFields centered_nonlinear_source_oracle(
    const Grid& grid, const VectorField& g1_total_gradient, const VectorField& g2_total_gradient,
    const VectorField& b, const NonlinearSourceReferenceConfig& config) {
    validate_vector_field(grid, g1_total_gradient, "g1 total gradient");
    validate_vector_field(grid, g2_total_gradient, "g2 total gradient");
    validate_vector_field(grid, b, "B");
    validate_nonlinear_source_config(config);

    const std::size_t cells = grid.cell_count();
    NonlinearSourceFields result{make_vector_field(cells), std::vector<double>(cells),
                                 std::vector<double>(cells), std::vector<double>(cells)};
    const long double regularization =
        static_cast<long double>(config.epsilon) * static_cast<long double>(config.v_rms);
    const long double regularization_sq = regularization * regularization;

    for (std::size_t id = 0; id < cells; ++id) {
        const Vec3Ld g1{g1_total_gradient.x[id], g1_total_gradient.y[id], g1_total_gradient.z[id]};
        const Vec3Ld g2{g2_total_gradient.x[id], g2_total_gradient.y[id], g2_total_gradient.z[id]};
        const Vec3Ld bv{b.x[id], b.y[id], b.z[id]};
        const Vec3Ld c = cross_ld(g1, g2);
        const long double d = dot_ld(c, c) + regularization_sq;
        const long double s1 = dot_ld(cross_ld(bv, g1), c) / d;
        const long double s2 = dot_ld(cross_ld(bv, g2), c) / d;

        result.c.x[id] = static_cast<double>(c.x);
        result.c.y[id] = static_cast<double>(c.y);
        result.c.z[id] = static_cast<double>(c.z);
        result.denominator[id] = static_cast<double>(d);
        result.s1[id] = static_cast<double>(s1);
        result.s2[id] = static_cast<double>(s2);
    }
    return result;
}

NonlinearSourceMirrorDiagnostics double_precision_nonlinear_source_mirror(
    const VectorField& g1_total_gradient, const VectorField& g2_total_gradient,
    const VectorField& b, double epsilon, double v_rms,
    const std::vector<double>& degeneracy_thresholds) {
    const std::size_t cells = g1_total_gradient.x.size();
    const auto matches_cells = [cells](const std::vector<double>& values) {
        return values.size() == cells;
    };
    if (!matches_cells(g1_total_gradient.y) || !matches_cells(g1_total_gradient.z) ||
        !matches_cells(g2_total_gradient.x) || !matches_cells(g2_total_gradient.y) ||
        !matches_cells(g2_total_gradient.z) || !matches_cells(b.x) || !matches_cells(b.y) ||
        !matches_cells(b.z)) {
        throw std::invalid_argument("nonlinear source mirror requires matching field sizes");
    }
    if (!std::isfinite(epsilon) || epsilon < 0.0) {
        throw std::invalid_argument("nonlinear source mirror epsilon must be finite and nonnegative");
    }
    if (!finite_positive(v_rms)) {
        throw std::invalid_argument("nonlinear source mirror v_rms must be finite and strictly positive");
    }
    for (double tau : degeneracy_thresholds) {
        if (!std::isfinite(tau) || tau < 0.0) {
            throw std::invalid_argument("degeneracy thresholds must be finite and nonnegative");
        }
    }

    NonlinearSourceMirrorDiagnostics result;
    result.fields = NonlinearSourceFields{make_vector_field(cells), std::vector<double>(cells),
                                          std::vector<double>(cells), std::vector<double>(cells)};
    result.degenerate_counts.assign(degeneracy_thresholds.size(), 0);
    result.degenerate_separation.assign(degeneracy_thresholds.size(),
                                        std::numeric_limits<double>::infinity());

    std::vector<double> threshold_sq(degeneracy_thresholds.size());
    for (std::size_t t = 0; t < degeneracy_thresholds.size(); ++t) {
        threshold_sq[t] = std::pow(degeneracy_thresholds[t] * v_rms, 2.0);
    }

    const double regularization = epsilon * v_rms;
    const double regularization_sq = regularization * regularization;

    for (std::size_t id = 0; id < cells; ++id) {
        const double g1x = g1_total_gradient.x[id], g1y = g1_total_gradient.y[id],
                    g1z = g1_total_gradient.z[id];
        const double g2x = g2_total_gradient.x[id], g2y = g2_total_gradient.y[id],
                    g2z = g2_total_gradient.z[id];
        const double bx = b.x[id], by = b.y[id], bz = b.z[id];

        const double cx = g1y * g2z - g1z * g2y;
        const double cy = g1z * g2x - g1x * g2z;
        const double cz = g1x * g2y - g1y * g2x;
        const double c_sq = cx * cx + cy * cy + cz * cz;
        const double d = c_sq + regularization_sq;

        const double bxg1x = by * g1z - bz * g1y;
        const double bxg1y = bz * g1x - bx * g1z;
        const double bxg1z = bx * g1y - by * g1x;
        const double bxg2x = by * g2z - bz * g2y;
        const double bxg2y = bz * g2x - bx * g2z;
        const double bxg2z = bx * g2y - by * g2x;

        const double s1 = (bxg1x * cx + bxg1y * cy + bxg1z * cz) / d;
        const double s2 = (bxg2x * cx + bxg2y * cy + bxg2z * cz) / d;

        result.fields.c.x[id] = cx;
        result.fields.c.y[id] = cy;
        result.fields.c.z[id] = cz;
        result.fields.denominator[id] = d;
        result.fields.s1[id] = s1;
        result.fields.s2[id] = s2;

        if (!std::isfinite(s1)) {
            ++result.nonfinite_s1_count;
        }
        if (!std::isfinite(s2)) {
            ++result.nonfinite_s2_count;
        }

        for (std::size_t t = 0; t < degeneracy_thresholds.size(); ++t) {
            if (std::isfinite(c_sq) && c_sq < threshold_sq[t]) {
                ++result.degenerate_counts[t];
            }
            if (std::isfinite(c_sq)) {
                const double denominator_reference = std::max(threshold_sq[t], 1.0);
                const double separation = std::abs(c_sq - threshold_sq[t]) / denominator_reference;
                result.degenerate_separation[t] = std::min(result.degenerate_separation[t], separation);
            }
        }
    }
    return result;
}

TotalGradientFixture make_near_degenerate_total_gradient_fixture(std::size_t cells_per_axis,
                                                                  double parallel_scale,
                                                                  double perturbation_scale) {
    if (!std::isfinite(parallel_scale) || !std::isfinite(perturbation_scale)) {
        throw std::invalid_argument("near-degenerate fixture scales must be finite");
    }
    TotalGradientFixture fixture = make_total_gradient_fixture(cells_per_axis);
    const std::vector<double> original_psi2_fluctuation = fixture.psi2_fluctuation;
    const Vec3 original_psi2_affine_gradient = fixture.psi2_affine_gradient;

    fixture.psi2_affine_gradient = {
        parallel_scale * fixture.psi1_affine_gradient.x +
            perturbation_scale * original_psi2_affine_gradient.x,
        parallel_scale * fixture.psi1_affine_gradient.y +
            perturbation_scale * original_psi2_affine_gradient.y,
        parallel_scale * fixture.psi1_affine_gradient.z +
            perturbation_scale * original_psi2_affine_gradient.z};
    validate_vector(fixture.psi2_affine_gradient, "near-degenerate psi2 affine gradient");

    for (std::size_t id = 0; id < fixture.grid.cell_count(); ++id) {
        fixture.psi2_fluctuation[id] = parallel_scale * fixture.psi1_fluctuation[id] +
                                       perturbation_scale * original_psi2_fluctuation[id];
    }
    return fixture;
}

VectorField inject_nonfinite_values(const VectorField& field,
                                    const std::vector<NonfiniteInjection>& injections) {
    if (field.x.size() != field.y.size() || field.y.size() != field.z.size()) {
        throw std::invalid_argument("VectorField components must have matching sizes");
    }
    VectorField result = field;
    for (const NonfiniteInjection& injection : injections) {
        if (injection.cell_index >= result.x.size()) {
            throw std::out_of_range("nonfinite injection cell index out of range");
        }
        result.x[injection.cell_index] = injection.replacement.x;
        result.y[injection.cell_index] = injection.replacement.y;
        result.z[injection.cell_index] = injection.replacement.z;
    }
    return result;
}

namespace {

[[nodiscard]] CoupledResidualFields coupled_residual_impl(
    const Grid& grid, const std::vector<double>& q, const std::vector<double>& psi1_fluctuation,
    const std::vector<double>& psi2_fluctuation, const Vec3& psi1_affine_gradient,
    const Vec3& psi2_affine_gradient, double eta, const NonlinearSourceReferenceConfig& config) {
    validate_field(grid, q, "q", true);
    validate_field(grid, psi1_fluctuation, "psi1 fluctuation", false);
    validate_field(grid, psi2_fluctuation, "psi2 fluctuation", false);
    validate_vector(psi1_affine_gradient, "psi1 affine gradient");
    validate_vector(psi2_affine_gradient, "psi2 affine gradient");
    if (!std::isfinite(eta)) {
        throw std::invalid_argument("coupled residual eta must be finite");
    }
    validate_nonlinear_source_config(config);

    // g1, g2, B, S1, S2: exclusively composed from the existing SF-07/08/09
    // CPU oracles, never re-derived.
    const VectorField g1 =
        centered_total_gradient_oracle(grid, psi1_fluctuation, psi1_affine_gradient);
    const VectorField g2 =
        centered_total_gradient_oracle(grid, psi2_fluctuation, psi2_affine_gradient);
    const HessianVectorBFields hvb =
        centered_hessian_vector_b_oracle(grid, psi1_fluctuation, psi2_fluctuation, g1, g2);
    const NonlinearSourceFields sources =
        centered_nonlinear_source_oracle(grid, g1, g2, hvb.b, config);

    // A u1, A u2 via the SF-02 divergence-form diffusion oracle.
    const std::vector<double> a_u1 = divergence_form_diffusion(grid, q, psi1_fluctuation);
    const std::vector<double> a_u2 = divergence_form_diffusion(grid, q, psi2_fluctuation);
    // div_h(q*gbar1), div_h(q*gbar2) via the SF-06 affine RHS oracle.
    const std::vector<double> affine1 = affine_rhs_discrete(grid, q, psi1_affine_gradient);
    const std::vector<double> affine2 = affine_rhs_discrete(grid, q, psi2_affine_gradient);

    const std::size_t cells = grid.cell_count();
    std::vector<double> raw_rhs1(cells);
    std::vector<double> raw_rhs2(cells);
    for (std::size_t id = 0; id < cells; ++id) {
        // Pairing: F1<->S2, F2<->S1.
        raw_rhs1[id] = affine1[id] - eta * q[id] * sources.s2[id];
        raw_rhs2[id] = affine2[id] - eta * q[id] * sources.s1[id];
    }

    CoupledResidualFields result;
    result.raw_rhs1_mean = static_cast<double>(long_double_mean(raw_rhs1));
    result.raw_rhs2_mean = static_cast<double>(long_double_mean(raw_rhs2));
    result.projected_rhs1 = mean_zero_projected(raw_rhs1);
    result.projected_rhs2 = mean_zero_projected(raw_rhs2);

    result.f1.resize(cells);
    result.f2.resize(cells);
    for (std::size_t id = 0; id < cells; ++id) {
        result.f1[id] = a_u1[id] - result.projected_rhs1[id];
        result.f2[id] = a_u2[id] - result.projected_rhs2[id];
    }
    result.s1 = sources.s1;
    result.s2 = sources.s2;
    return result;
}

}  // namespace

std::vector<double> make_positive_q_field(const Grid& grid, const Vec3& lengths) {
    grid.validate();
    validate_position_and_lengths({0.0, 0.0, 0.0}, lengths);
    std::vector<double> result(grid.cell_count());
    for (std::size_t iz = 0; iz < grid.nz; ++iz) for (std::size_t iy = 0; iy < grid.ny; ++iy) for (std::size_t ix = 0; ix < grid.nx; ++ix) {
        const Vec3 position = grid.cell_center(ix, iy, iz);
        result[grid.index(ix, iy, iz)] = trigonometric_q(position, lengths);
    }
    return result;
}

double dimensionless_length_reference(const Vec3& lengths) {
    validate_position_and_lengths({0.0, 0.0, 0.0}, lengths);
    return std::cbrt(lengths.x * lengths.y * lengths.z);
}

CoupledResidualFields coupled_residual_reference(
    const Grid& grid, const std::vector<double>& q, const std::vector<double>& psi1_fluctuation,
    const std::vector<double>& psi2_fluctuation, const Vec3& psi1_affine_gradient,
    const Vec3& psi2_affine_gradient, double eta, const NonlinearSourceReferenceConfig& config) {
    return coupled_residual_impl(grid, q, psi1_fluctuation, psi2_fluctuation,
                                 psi1_affine_gradient, psi2_affine_gradient, eta, config);
}

CoupledResidualFields coupled_residual_reference(const std::vector<double>& q,
                                                 const TotalGradientFixture& fixture, double eta,
                                                 const NonlinearSourceReferenceConfig& config) {
    validate_total_gradient_fixture(fixture);
    return coupled_residual_impl(fixture.grid, q, fixture.psi1_fluctuation,
                                 fixture.psi2_fluctuation, fixture.psi1_affine_gradient,
                                 fixture.psi2_affine_gradient, eta, config);
}

ResidualNormalizationReference residual_normalization_reference(double rms_f1, double rms_f2,
                                                                 double q_rms, double v_rms,
                                                                 double l_ref) {
    if (!std::isfinite(rms_f1) || rms_f1 < 0.0 || !std::isfinite(rms_f2) || rms_f2 < 0.0) {
        throw std::invalid_argument("residual normalization RMS values must be finite and nonnegative");
    }
    if (!finite_positive(q_rms) || !finite_positive(v_rms) || !finite_positive(l_ref)) {
        throw std::invalid_argument(
            "residual normalization q_rms, v_rms, and l_ref must be finite and strictly positive");
    }
    ResidualNormalizationReference result;
    result.r1 = rms_f1 * l_ref / (q_rms * v_rms);
    result.r2 = rms_f2 * l_ref / q_rms;
    result.r_f = std::sqrt((result.r1 * result.r1 + result.r2 * result.r2) / 2.0);
    return result;
}

LogHistogramReference log_histogram_reference(const VectorField& c, double c_min, double c_max) {
    if (c.x.size() != c.y.size() || c.y.size() != c.z.size()) {
        throw std::invalid_argument("histogram VectorField components must have matching sizes");
    }
    if (!finite_positive(c_min) || !finite_positive(c_max) || c_max <= c_min) {
        throw std::invalid_argument("histogram range requires 0 < c_min < c_max, both finite");
    }

    const double log_min = std::log10(c_min);
    const double log_max = std::log10(c_max);
    const double inv_bin_width = static_cast<double>(kHistogramBins) / (log_max - log_min);

    LogHistogramReference result;
    for (std::size_t id = 0; id < c.x.size(); ++id) {
        const double cx = c.x[id];
        const double cy = c.y[id];
        const double cz = c.z[id];
        const double c_sq = cx * cx + cy * cy + cz * cz;
        const double v = std::sqrt(c_sq);
        if (!std::isfinite(v)) {
            ++result.overflow;
        } else if (v < c_min) {
            ++result.underflow;
        } else if (v >= c_max) {
            ++result.overflow;
        } else {
            const double t = (std::log10(v) - log_min) * inv_bin_width;
            auto idx = static_cast<std::ptrdiff_t>(std::floor(t));
            idx = std::max<std::ptrdiff_t>(0, std::min<std::ptrdiff_t>(
                                                  idx, static_cast<std::ptrdiff_t>(kHistogramBins) - 1));
            ++result.counts[static_cast<std::size_t>(idx)];
            const double separation = std::abs(t - std::round(t));
            result.min_edge_separation = std::min(result.min_edge_separation, separation);
        }
    }
    return result;
}

double histogram_percentile(const std::vector<std::size_t>& counts, std::size_t underflow,
                            std::size_t overflow, double c_min, double c_max, double p) {
    if (counts.size() != kHistogramBins) {
        throw std::invalid_argument("histogram percentile requires exactly kHistogramBins counts");
    }
    if (!finite_positive(c_min) || !finite_positive(c_max) || c_max <= c_min) {
        throw std::invalid_argument("histogram range requires 0 < c_min < c_max, both finite");
    }
    if (!std::isfinite(p) || p < 0.0 || p > 1.0) {
        throw std::invalid_argument("histogram percentile p must be finite and in [0, 1]");
    }

    long double total = static_cast<long double>(underflow) + static_cast<long double>(overflow);
    for (std::size_t count : counts) total += static_cast<long double>(count);
    if (total <= 0.0L) {
        throw std::invalid_argument("histogram percentile requires a nonempty population");
    }
    const long double target = static_cast<long double>(p) * total;

    const double log_min = std::log10(c_min);
    const double log_max = std::log10(c_max);
    const double bin_width = (log_max - log_min) / static_cast<double>(kHistogramBins);
    const auto edge = [&](std::size_t bin_boundary) {
        return std::pow(10.0, log_min + static_cast<double>(bin_boundary) * bin_width);
    };

    long double cumulative = static_cast<long double>(underflow);
    if (cumulative >= target) {
        return c_min;
    }
    for (std::size_t bin = 0; bin < kHistogramBins; ++bin) {
        cumulative += static_cast<long double>(counts[bin]);
        if (cumulative >= target) {
            return edge(bin + 1);
        }
    }
    // Target only reached inside the open-ended overflow bucket. Matches the
    // production `residual_histogram_percentile` open-ended overflow
    // convention.
    return std::numeric_limits<double>::infinity();
}

double exact_sorted_percentile(std::vector<double> values, double p) {
    if (values.empty()) {
        throw std::invalid_argument("exact sorted percentile requires nonempty values");
    }
    if (!std::isfinite(p) || p < 0.0 || p > 1.0) {
        throw std::invalid_argument("exact sorted percentile p must be finite and in [0, 1]");
    }
    for (double value : values) {
        if (!std::isfinite(value)) {
            throw std::invalid_argument("exact sorted percentile requires finite values");
        }
    }
    std::sort(values.begin(), values.end());
    const auto n = static_cast<std::ptrdiff_t>(values.size());
    auto rank = static_cast<std::ptrdiff_t>(std::ceil(p * static_cast<double>(n))) - 1;
    rank = std::max<std::ptrdiff_t>(0, std::min<std::ptrdiff_t>(rank, n - 1));
    return values[static_cast<std::size_t>(rank)];
}

// ============================================================================
// SF-11 CompactMAC reconstruction and physical diagnostics mirrors.
// ============================================================================

namespace {

void require_compact_mac_sizes(const Grid& grid, const CompactMacField& field, const char* name) {
    grid.validate();
    if (field.u.size() != compact_mac_u_size(grid) || field.v.size() != compact_mac_v_size(grid) ||
        field.w.size() != compact_mac_w_size(grid)) {
        throw std::invalid_argument(std::string(name) + " has inconsistent CompactMAC face sizes");
    }
}

}  // namespace

std::size_t compact_mac_u_size(const Grid& grid) {
    grid.validate();
    return (grid.nx + 1) * grid.ny * grid.nz;
}

std::size_t compact_mac_v_size(const Grid& grid) {
    grid.validate();
    return grid.nx * (grid.ny + 1) * grid.nz;
}

std::size_t compact_mac_w_size(const Grid& grid) {
    grid.validate();
    return grid.nx * grid.ny * (grid.nz + 1);
}

std::size_t compact_mac_u_index(const Grid& grid, std::size_t i, std::size_t j, std::size_t k) {
    grid.validate();
    if (i > grid.nx || j >= grid.ny || k >= grid.nz) {
        throw std::out_of_range("CompactMAC U-face index outside extent");
    }
    return i + (grid.nx + 1) * (j + grid.ny * k);
}

std::size_t compact_mac_v_index(const Grid& grid, std::size_t i, std::size_t j, std::size_t k) {
    grid.validate();
    if (i >= grid.nx || j > grid.ny || k >= grid.nz) {
        throw std::out_of_range("CompactMAC V-face index outside extent");
    }
    return i + grid.nx * (j + (grid.ny + 1) * k);
}

std::size_t compact_mac_w_index(const Grid& grid, std::size_t i, std::size_t j, std::size_t k) {
    grid.validate();
    if (i >= grid.nx || j >= grid.ny || k > grid.nz) {
        throw std::out_of_range("CompactMAC W-face index outside extent");
    }
    return i + grid.nx * (j + grid.ny * k);
}

VectorField total_gradient_double_mirror(const Grid& grid, const std::vector<double>& fluctuation,
                                         const Vec3& affine_gradient) {
    validate_field(grid, fluctuation, "fluctuation", false);
    validate_vector(affine_gradient, "affine gradient");
    VectorField result{{}, {}, {}};
    result.x.resize(grid.cell_count());
    result.y.resize(grid.cell_count());
    result.z.resize(grid.cell_count());
    for (std::size_t iz = 0; iz < grid.nz; ++iz) for (std::size_t iy = 0; iy < grid.ny; ++iy) for (std::size_t ix = 0; ix < grid.nx; ++ix) {
        const std::size_t id = grid.index(ix, iy, iz);
        const std::size_t xp = grid.index(wrap_index(static_cast<std::ptrdiff_t>(ix) + 1, grid.nx), iy, iz);
        const std::size_t xm = grid.index(wrap_index(static_cast<std::ptrdiff_t>(ix) - 1, grid.nx), iy, iz);
        const std::size_t yp = grid.index(ix, wrap_index(static_cast<std::ptrdiff_t>(iy) + 1, grid.ny), iz);
        const std::size_t ym = grid.index(ix, wrap_index(static_cast<std::ptrdiff_t>(iy) - 1, grid.ny), iz);
        const std::size_t zp = grid.index(ix, iy, wrap_index(static_cast<std::ptrdiff_t>(iz) + 1, grid.nz));
        const std::size_t zm = grid.index(ix, iy, wrap_index(static_cast<std::ptrdiff_t>(iz) - 1, grid.nz));
        result.x[id] = (fluctuation[xp] - fluctuation[xm]) / (2.0 * grid.spacing.x) + affine_gradient.x;
        result.y[id] = (fluctuation[yp] - fluctuation[ym]) / (2.0 * grid.spacing.y) + affine_gradient.y;
        result.z[id] = (fluctuation[zp] - fluctuation[zm]) / (2.0 * grid.spacing.z) + affine_gradient.z;
    }
    return result;
}

CompactMacField reconstruct_velocity_compact_mac(const Grid& grid, const VectorField& g1_total_gradient,
                                                  const VectorField& g2_total_gradient) {
    validate_vector_field(grid, g1_total_gradient, "g1 total gradient");
    validate_vector_field(grid, g2_total_gradient, "g2 total gradient");
    CompactMacField result;
    result.u.resize(compact_mac_u_size(grid));
    result.v.resize(compact_mac_v_size(grid));
    result.w.resize(compact_mac_w_size(grid));

    for (std::size_t k = 0; k < grid.nz; ++k) for (std::size_t j = 0; j < grid.ny; ++j) for (std::size_t i = 0; i <= grid.nx; ++i) {
        const std::size_t a = grid.index(wrap_index(static_cast<std::ptrdiff_t>(i) - 1, grid.nx), j, k);
        const std::size_t b = grid.index(wrap_index(static_cast<std::ptrdiff_t>(i), grid.nx), j, k);
        const double t1y = 0.5 * (g1_total_gradient.y[a] + g1_total_gradient.y[b]);
        const double t1z = 0.5 * (g1_total_gradient.z[a] + g1_total_gradient.z[b]);
        const double t2y = 0.5 * (g2_total_gradient.y[a] + g2_total_gradient.y[b]);
        const double t2z = 0.5 * (g2_total_gradient.z[a] + g2_total_gradient.z[b]);
        result.u[compact_mac_u_index(grid, i, j, k)] = t1y * t2z - t1z * t2y;
    }
    for (std::size_t k = 0; k < grid.nz; ++k) for (std::size_t j = 0; j <= grid.ny; ++j) for (std::size_t i = 0; i < grid.nx; ++i) {
        const std::size_t a = grid.index(i, wrap_index(static_cast<std::ptrdiff_t>(j) - 1, grid.ny), k);
        const std::size_t b = grid.index(i, wrap_index(static_cast<std::ptrdiff_t>(j), grid.ny), k);
        const double t1z = 0.5 * (g1_total_gradient.z[a] + g1_total_gradient.z[b]);
        const double t1x = 0.5 * (g1_total_gradient.x[a] + g1_total_gradient.x[b]);
        const double t2z = 0.5 * (g2_total_gradient.z[a] + g2_total_gradient.z[b]);
        const double t2x = 0.5 * (g2_total_gradient.x[a] + g2_total_gradient.x[b]);
        result.v[compact_mac_v_index(grid, i, j, k)] = t1z * t2x - t1x * t2z;
    }
    for (std::size_t k = 0; k <= grid.nz; ++k) for (std::size_t j = 0; j < grid.ny; ++j) for (std::size_t i = 0; i < grid.nx; ++i) {
        const std::size_t a = grid.index(i, j, wrap_index(static_cast<std::ptrdiff_t>(k) - 1, grid.nz));
        const std::size_t b = grid.index(i, j, wrap_index(static_cast<std::ptrdiff_t>(k), grid.nz));
        const double t1x = 0.5 * (g1_total_gradient.x[a] + g1_total_gradient.x[b]);
        const double t1y = 0.5 * (g1_total_gradient.y[a] + g1_total_gradient.y[b]);
        const double t2x = 0.5 * (g2_total_gradient.x[a] + g2_total_gradient.x[b]);
        const double t2y = 0.5 * (g2_total_gradient.y[a] + g2_total_gradient.y[b]);
        result.w[compact_mac_w_index(grid, i, j, k)] = t1x * t2y - t1y * t2x;
    }
    return result;
}

CompactMacField analytic_face_velocity(const TotalGradientFixture& fixture) {
    validate_total_gradient_fixture(fixture);
    const Grid& grid = fixture.grid;
    CompactMacField result;
    result.u.resize(compact_mac_u_size(grid));
    result.v.resize(compact_mac_v_size(grid));
    result.w.resize(compact_mac_w_size(grid));

    for (std::size_t k = 0; k < grid.nz; ++k) for (std::size_t j = 0; j < grid.ny; ++j) for (std::size_t i = 0; i <= grid.nx; ++i) {
        const Vec3 position{static_cast<double>(i) * grid.spacing.x,
                            (static_cast<double>(j) + 0.5) * grid.spacing.y,
                            (static_cast<double>(k) + 0.5) * grid.spacing.z};
        const Vec3 g1 = total_gradient_analytic(GradientFixtureField::psi1, position, fixture.lengths,
                                                fixture.psi1_affine_gradient);
        const Vec3 g2 = total_gradient_analytic(GradientFixtureField::psi2, position, fixture.lengths,
                                                fixture.psi2_affine_gradient);
        result.u[compact_mac_u_index(grid, i, j, k)] = cross(g1, g2).x;
    }
    for (std::size_t k = 0; k < grid.nz; ++k) for (std::size_t j = 0; j <= grid.ny; ++j) for (std::size_t i = 0; i < grid.nx; ++i) {
        const Vec3 position{(static_cast<double>(i) + 0.5) * grid.spacing.x,
                            static_cast<double>(j) * grid.spacing.y,
                            (static_cast<double>(k) + 0.5) * grid.spacing.z};
        const Vec3 g1 = total_gradient_analytic(GradientFixtureField::psi1, position, fixture.lengths,
                                                fixture.psi1_affine_gradient);
        const Vec3 g2 = total_gradient_analytic(GradientFixtureField::psi2, position, fixture.lengths,
                                                fixture.psi2_affine_gradient);
        result.v[compact_mac_v_index(grid, i, j, k)] = cross(g1, g2).y;
    }
    for (std::size_t k = 0; k <= grid.nz; ++k) for (std::size_t j = 0; j < grid.ny; ++j) for (std::size_t i = 0; i < grid.nx; ++i) {
        const Vec3 position{(static_cast<double>(i) + 0.5) * grid.spacing.x,
                            (static_cast<double>(j) + 0.5) * grid.spacing.y,
                            static_cast<double>(k) * grid.spacing.z};
        const Vec3 g1 = total_gradient_analytic(GradientFixtureField::psi1, position, fixture.lengths,
                                                fixture.psi1_affine_gradient);
        const Vec3 g2 = total_gradient_analytic(GradientFixtureField::psi2, position, fixture.lengths,
                                                fixture.psi2_affine_gradient);
        result.w[compact_mac_w_index(grid, i, j, k)] = cross(g1, g2).z;
    }
    return result;
}

std::vector<double> natural_mac_divergence(const Grid& grid, const CompactMacField& velocity) {
    require_compact_mac_sizes(grid, velocity, "velocity");
    std::vector<double> result(grid.cell_count());
    const double inverse_dx = 1.0 / grid.spacing.x;
    const double inverse_dy = 1.0 / grid.spacing.y;
    const double inverse_dz = 1.0 / grid.spacing.z;
    for (std::size_t k = 0; k < grid.nz; ++k) for (std::size_t j = 0; j < grid.ny; ++j) for (std::size_t i = 0; i < grid.nx; ++i) {
        const double du = velocity.u[compact_mac_u_index(grid, i + 1, j, k)] -
                          velocity.u[compact_mac_u_index(grid, i, j, k)];
        const double dv = velocity.v[compact_mac_v_index(grid, i, j + 1, k)] -
                          velocity.v[compact_mac_v_index(grid, i, j, k)];
        const double dw = velocity.w[compact_mac_w_index(grid, i, j, k + 1)] -
                          velocity.w[compact_mac_w_index(grid, i, j, k)];
        result[grid.index(i, j, k)] = du * inverse_dx + dv * inverse_dy + dw * inverse_dz;
    }
    return result;
}

VectorField compact_mac_face_to_center(const Grid& grid, const CompactMacField& velocity) {
    require_compact_mac_sizes(grid, velocity, "velocity");
    VectorField result{{}, {}, {}};
    result.x.resize(grid.cell_count());
    result.y.resize(grid.cell_count());
    result.z.resize(grid.cell_count());
    for (std::size_t k = 0; k < grid.nz; ++k) for (std::size_t j = 0; j < grid.ny; ++j) for (std::size_t i = 0; i < grid.nx; ++i) {
        const std::size_t id = grid.index(i, j, k);
        result.x[id] = 0.5 * (velocity.u[compact_mac_u_index(grid, i, j, k)] +
                              velocity.u[compact_mac_u_index(grid, i + 1, j, k)]);
        result.y[id] = 0.5 * (velocity.v[compact_mac_v_index(grid, i, j, k)] +
                              velocity.v[compact_mac_v_index(grid, i, j + 1, k)]);
        result.z[id] = 0.5 * (velocity.w[compact_mac_w_index(grid, i, j, k)] +
                              velocity.w[compact_mac_w_index(grid, i, j, k + 1)]);
    }
    return result;
}

double compact_mac_v_rms(const Grid& grid, const CompactMacField& velocity) {
    const VectorField center = compact_mac_face_to_center(grid, velocity);
    const std::size_t n = grid.cell_count();
    double nu_sq = 0.0, nv_sq = 0.0, nw_sq = 0.0;
    for (std::size_t id = 0; id < n; ++id) {
        nu_sq += center.x[id] * center.x[id];
        nv_sq += center.y[id] * center.y[id];
        nw_sq += center.z[id] * center.z[id];
    }
    return std::sqrt((nu_sq + nv_sq + nw_sq) / static_cast<double>(n));
}

double mac_grid_length_reference(const Grid& grid) {
    grid.validate();
    const double lx = static_cast<double>(grid.nx) * grid.spacing.x;
    const double ly = static_cast<double>(grid.ny) * grid.spacing.y;
    const double lz = static_cast<double>(grid.nz) * grid.spacing.z;
    return std::cbrt(lx * ly * lz);
}

namespace {

void require_v_rms(double v_rms) {
    if (!finite_positive(v_rms)) {
        throw std::invalid_argument("v_rms must be finite and strictly positive");
    }
}

}  // namespace

VelocityFaceErrorReport velocity_face_error_reference(const Grid& grid, const CompactMacField& v_psi,
                                                       const CompactMacField& v_darcy, double v_rms) {
    require_compact_mac_sizes(grid, v_psi, "v_psi");
    require_compact_mac_sizes(grid, v_darcy, "v_darcy");
    require_v_rms(v_rms);

    std::vector<double> diff_u(grid.cell_count()), diff_v(grid.cell_count()), diff_w(grid.cell_count());
    for (std::size_t k = 0; k < grid.nz; ++k) for (std::size_t j = 0; j < grid.ny; ++j) for (std::size_t i = 0; i < grid.nx; ++i) {
        const std::size_t id = grid.index(i, j, k);
        diff_u[id] = v_psi.u[compact_mac_u_index(grid, i, j, k)] - v_darcy.u[compact_mac_u_index(grid, i, j, k)];
        diff_v[id] = v_psi.v[compact_mac_v_index(grid, i, j, k)] - v_darcy.v[compact_mac_v_index(grid, i, j, k)];
        diff_w[id] = v_psi.w[compact_mac_w_index(grid, i, j, k)] - v_darcy.w[compact_mac_w_index(grid, i, j, k)];
    }

    VelocityFaceErrorReport report;
    report.u = {rms_norm(diff_u), linf_norm(diff_u)};
    report.v = {rms_norm(diff_v), linf_norm(diff_v)};
    report.w = {rms_norm(diff_w), linf_norm(diff_w)};
    report.e_v = std::sqrt((report.u.rms * report.u.rms + report.v.rms * report.v.rms +
                            report.w.rms * report.w.rms) / 3.0) / v_rms;
    return report;
}

namespace {

double pearson_correlation(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size() || a.empty()) {
        throw std::invalid_argument("Pearson correlation requires matching nonempty samples");
    }
    const double n = static_cast<double>(a.size());
    double sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (std::size_t idx = 0; idx < a.size(); ++idx) {
        sx += a[idx];
        sy += b[idx];
        sxx += a[idx] * a[idx];
        syy += b[idx] * b[idx];
        sxy += a[idx] * b[idx];
    }
    const double numerator = n * sxy - sx * sy;
    const double denominator = std::sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
    // For exactly-degenerate (zero-variance) inputs this single-loop
    // accumulation is less prone to the ±1 collapse seen on the two-kernel
    // GPU path (see physical_diagnostics_gpu_cases.cu, case 1), but the
    // raw-moment Pearson formula's degenerate-input instability is inherent
    // to the algebra, not an artifact of a particular reduction strategy;
    // the value is meaningful only for non-degenerate inputs.
    return numerator / denominator;  // 0/0 or nonzero/0 yields NaN/Inf by IEEE semantics; no hidden floor.
}

}  // namespace

FaceCorrelationReport velocity_face_correlation_reference(const Grid& grid, const CompactMacField& v_psi,
                                                           const CompactMacField& v_darcy) {
    require_compact_mac_sizes(grid, v_psi, "v_psi");
    require_compact_mac_sizes(grid, v_darcy, "v_darcy");

    const std::size_t n = grid.cell_count();
    std::vector<double> psi_u(n), psi_v(n), psi_w(n), darcy_u(n), darcy_v(n), darcy_w(n);
    for (std::size_t k = 0; k < grid.nz; ++k) for (std::size_t j = 0; j < grid.ny; ++j) for (std::size_t i = 0; i < grid.nx; ++i) {
        const std::size_t id = grid.index(i, j, k);
        psi_u[id] = v_psi.u[compact_mac_u_index(grid, i, j, k)];
        psi_v[id] = v_psi.v[compact_mac_v_index(grid, i, j, k)];
        psi_w[id] = v_psi.w[compact_mac_w_index(grid, i, j, k)];
        darcy_u[id] = v_darcy.u[compact_mac_u_index(grid, i, j, k)];
        darcy_v[id] = v_darcy.v[compact_mac_v_index(grid, i, j, k)];
        darcy_w[id] = v_darcy.w[compact_mac_w_index(grid, i, j, k)];
    }
    return {pearson_correlation(psi_u, darcy_u), pearson_correlation(psi_v, darcy_v),
           pearson_correlation(psi_w, darcy_w)};
}

MagnitudeErrorReport velocity_magnitude_error_reference(const Grid& grid, const CompactMacField& v_psi,
                                                         const CompactMacField& v_darcy, double v_rms) {
    require_v_rms(v_rms);
    const VectorField p = compact_mac_face_to_center(grid, v_psi);
    const VectorField d = compact_mac_face_to_center(grid, v_darcy);
    std::vector<double> m(grid.cell_count());
    for (std::size_t id = 0; id < grid.cell_count(); ++id) {
        const double p_mag = std::sqrt(p.x[id] * p.x[id] + p.y[id] * p.y[id] + p.z[id] * p.z[id]);
        const double d_mag = std::sqrt(d.x[id] * d.x[id] + d.y[id] * d.y[id] + d.z[id] * d.z[id]);
        m[id] = p_mag - d_mag;
    }
    MagnitudeErrorReport report;
    report.rms = rms_norm(m);
    report.linf = linf_norm(m);
    report.rms_relative = report.rms / v_rms;
    report.linf_relative = report.linf / v_rms;
    return report;
}

AngleErrorReport velocity_angle_error_reference(const Grid& grid, const CompactMacField& v_psi,
                                                const CompactMacField& v_darcy, double v_rms,
                                                double angle_exclusion_rel) {
    require_v_rms(v_rms);
    if (!std::isfinite(angle_exclusion_rel) || angle_exclusion_rel < 0.0) {
        throw std::invalid_argument("angle exclusion relative threshold must be finite and nonnegative");
    }
    const VectorField p = compact_mac_face_to_center(grid, v_psi);
    const VectorField d = compact_mac_face_to_center(grid, v_darcy);
    const double floor = angle_exclusion_rel * v_rms;

    AngleErrorReport report;
    double sum_theta_sq = 0.0;
    for (std::size_t id = 0; id < grid.cell_count(); ++id) {
        const double px = p.x[id], py = p.y[id], pz = p.z[id];
        const double dxc = d.x[id], dyc = d.y[id], dzc = d.z[id];
        const double p_mag = std::sqrt(px * px + py * py + pz * pz);
        const double d_mag = std::sqrt(dxc * dxc + dyc * dyc + dzc * dzc);
        if (p_mag < floor || d_mag < floor) {
            ++report.excluded_count;
            continue;
        }
        ++report.included_count;
        const double dot = px * dxc + py * dyc + pz * dzc;
        const double cx = py * dzc - pz * dyc;
        const double cy = pz * dxc - px * dzc;
        const double cz = px * dyc - py * dxc;
        const double cross_mag = std::sqrt(cx * cx + cy * cy + cz * cz);
        const double theta = std::atan2(cross_mag, dot);
        sum_theta_sq += theta * theta;
        report.max_theta = std::max(report.max_theta, theta);
    }
    report.rms_theta = report.included_count > 0
                           ? std::sqrt(sum_theta_sq / static_cast<double>(report.included_count))
                           : std::numeric_limits<double>::quiet_NaN();
    if (report.included_count == 0) {
        report.max_theta = std::numeric_limits<double>::quiet_NaN();
    }
    return report;
}

InvarianceReport darcy_invariance_reference(const Grid& grid, const VectorField& darcy_center,
                                            const VectorField& total_gradient, double v_rms) {
    validate_vector_field(grid, darcy_center, "Darcy center velocity");
    validate_vector_field(grid, total_gradient, "total gradient");
    require_v_rms(v_rms);

    const std::size_t n = grid.cell_count();
    std::vector<double> dot(n);
    double gx_sq = 0.0, gy_sq = 0.0, gz_sq = 0.0;
    for (std::size_t id = 0; id < n; ++id) {
        dot[id] = darcy_center.x[id] * total_gradient.x[id] + darcy_center.y[id] * total_gradient.y[id] +
                  darcy_center.z[id] * total_gradient.z[id];
        gx_sq += total_gradient.x[id] * total_gradient.x[id];
        gy_sq += total_gradient.y[id] * total_gradient.y[id];
        gz_sq += total_gradient.z[id] * total_gradient.z[id];
    }
    InvarianceReport report;
    report.raw_rms = rms_norm(dot);
    report.grad_rms = std::sqrt((gx_sq + gy_sq + gz_sq) / static_cast<double>(n));
    report.e = report.raw_rms / (v_rms * report.grad_rms);
    return report;
}

DivergenceReport divergence_report_reference(const Grid& grid, const std::vector<double>& divergence,
                                             double v_rms) {
    validate_field(grid, divergence, "divergence", false);
    require_v_rms(v_rms);
    DivergenceReport report;
    report.rms_div = rms_norm(divergence);
    report.linf_div = linf_norm(divergence);
    report.e_div = mac_grid_length_reference(grid) * report.rms_div / v_rms;
    return report;
}

CrossGradientDegeneracyReport cross_gradient_degeneracy_reference(
    const Grid& grid, const VectorField& g1_total_gradient, const VectorField& g2_total_gradient,
    const VectorField& darcy_center, double v_rms, const std::vector<double>& degeneracy_thresholds,
    double low_speed_rel) {
    validate_vector_field(grid, g1_total_gradient, "g1 total gradient");
    validate_vector_field(grid, g2_total_gradient, "g2 total gradient");
    validate_vector_field(grid, darcy_center, "Darcy center velocity");
    require_v_rms(v_rms);
    if (!std::isfinite(low_speed_rel) || low_speed_rel < 0.0) {
        throw std::invalid_argument("low-speed relative threshold must be finite and nonnegative");
    }
    for (double tau : degeneracy_thresholds) {
        if (!std::isfinite(tau) || tau < 0.0) {
            throw std::invalid_argument("degeneracy thresholds must be finite and nonnegative");
        }
    }

    const std::size_t n = grid.cell_count();
    const std::size_t t_count = degeneracy_thresholds.size();
    CrossGradientDegeneracyReport report;
    report.total_degenerate.assign(t_count, 0);
    report.low_speed_degenerate.assign(t_count, 0);
    report.unexplained_degenerate.assign(t_count, 0);
    report.c_min = std::numeric_limits<double>::infinity();
    report.c_max = -std::numeric_limits<double>::infinity();
    double c_sum = 0.0;

    std::vector<double> thresholds_scaled(t_count);
    for (std::size_t t = 0; t < t_count; ++t) {
        thresholds_scaled[t] = degeneracy_thresholds[t] * v_rms;
    }

    for (std::size_t id = 0; id < n; ++id) {
        const double g1x = g1_total_gradient.x[id], g1y = g1_total_gradient.y[id],
                    g1z = g1_total_gradient.z[id];
        const double g2x = g2_total_gradient.x[id], g2y = g2_total_gradient.y[id],
                    g2z = g2_total_gradient.z[id];
        const double cx = g1y * g2z - g1z * g2y;
        const double cy = g1z * g2x - g1x * g2z;
        const double cz = g1x * g2y - g1y * g2x;
        const double c_mag = std::sqrt(cx * cx + cy * cy + cz * cz);

        report.c_min = std::min(report.c_min, c_mag);
        report.c_max = std::max(report.c_max, c_mag);
        c_sum += c_mag;

        const double dxc = darcy_center.x[id], dyc = darcy_center.y[id], dzc = darcy_center.z[id];
        const double d_mag = std::sqrt(dxc * dxc + dyc * dyc + dzc * dzc);

        for (std::size_t t = 0; t < t_count; ++t) {
            const bool degenerate = c_mag < thresholds_scaled[t];
            if (degenerate) {
                ++report.total_degenerate[t];
                if (d_mag < low_speed_rel * v_rms) {
                    ++report.low_speed_degenerate[t];
                }
            }
        }
    }
    for (std::size_t t = 0; t < t_count; ++t) {
        report.unexplained_degenerate[t] = report.total_degenerate[t] - report.low_speed_degenerate[t];
    }
    report.c_mean = c_sum / static_cast<double>(n);
    return report;
}

PhysicalDiagnosticsMirror physical_diagnostics_mirror(
    const Grid& grid, const std::vector<double>& psi1_fluctuation,
    const std::vector<double>& psi2_fluctuation, const Vec3& psi1_affine_gradient,
    const Vec3& psi2_affine_gradient, const CompactMacField& darcy_velocity,
    const PhysicalDiagnosticsConfig& config) {
    if (config.degeneracy_thresholds.size() > 4) {
        throw std::invalid_argument("physical diagnostics support at most 4 degeneracy thresholds");
    }

    PhysicalDiagnosticsMirror result;
    result.g1_total_gradient = total_gradient_double_mirror(grid, psi1_fluctuation, psi1_affine_gradient);
    result.g2_total_gradient = total_gradient_double_mirror(grid, psi2_fluctuation, psi2_affine_gradient);
    result.v_psi = reconstruct_velocity_compact_mac(grid, result.g1_total_gradient, result.g2_total_gradient);
    result.darcy_center = compact_mac_face_to_center(grid, darcy_velocity);
    result.v_rms = compact_mac_v_rms(grid, darcy_velocity);

    result.face_error = velocity_face_error_reference(grid, result.v_psi, darcy_velocity, result.v_rms);
    result.face_correlation = velocity_face_correlation_reference(grid, result.v_psi, darcy_velocity);
    result.magnitude_error =
        velocity_magnitude_error_reference(grid, result.v_psi, darcy_velocity, result.v_rms);
    result.angle_error = velocity_angle_error_reference(grid, result.v_psi, darcy_velocity, result.v_rms,
                                                         config.angle_exclusion_rel);
    result.invariance_psi1 =
        darcy_invariance_reference(grid, result.darcy_center, result.g1_total_gradient, result.v_rms);
    result.invariance_psi2 =
        darcy_invariance_reference(grid, result.darcy_center, result.g2_total_gradient, result.v_rms);
    const std::vector<double> divergence_field = natural_mac_divergence(grid, result.v_psi);
    result.divergence = divergence_report_reference(grid, divergence_field, result.v_rms);
    result.cross_gradient = cross_gradient_degeneracy_reference(
        grid, result.g1_total_gradient, result.g2_total_gradient, result.darcy_center, result.v_rms,
        config.degeneracy_thresholds, config.low_speed_rel);
    return result;
}

}  // namespace macroflow3d::streamfunctions::reference
