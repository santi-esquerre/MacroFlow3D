#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace macroflow3d::streamfunctions::reference {

// CPU-only, test-local oracle conventions: cell-centered, x-fastest storage.
struct Vec3 {
    double x{};
    double y{};
    double z{};
};

enum class Axis { x = 0, y = 1, z = 2 };

struct Grid {
    std::size_t nx{};
    std::size_t ny{};
    std::size_t nz{};
    Vec3 spacing{};

    [[nodiscard]] std::size_t cell_count() const;
    [[nodiscard]] std::size_t index(std::size_t ix, std::size_t iy, std::size_t iz) const;
    [[nodiscard]] Vec3 cell_center(std::size_t ix, std::size_t iy, std::size_t iz) const;
    void validate() const;
};

// Wraps arbitrary signed indices, including values below -extent.
[[nodiscard]] std::size_t wrap_index(std::ptrdiff_t index, std::size_t extent);
[[nodiscard]] double harmonic_mean_q(double q_cell, double q_neighbor);

[[nodiscard]] std::vector<double> centered_first(const Grid& grid,
                                                  const std::vector<double>& field, Axis axis);
[[nodiscard]] std::vector<double> centered_second(const Grid& grid,
                                                   const std::vector<double>& field, Axis axis);
[[nodiscard]] std::vector<double> centered_mixed(const Grid& grid,
                                                  const std::vector<double>& field,
                                                  Axis first, Axis second);
[[nodiscard]] std::vector<double> divergence_form_diffusion(const Grid& grid,
                                                             const std::vector<double>& q,
                                                             const std::vector<double>& u);

[[nodiscard]] Vec3 cross(const Vec3& left, const Vec3& right);
[[nodiscard]] double rms_norm(const std::vector<double>& values);
[[nodiscard]] double linf_norm(const std::vector<double>& values);

enum class ObservedOrderStatus {
    valid,
    non_finite_error,
    non_positive_error,
    invalid_refinement_ratio,
};

struct ObservedOrder {
    ObservedOrderStatus status{ObservedOrderStatus::invalid_refinement_ratio};
    double value{};
    std::string message;

    [[nodiscard]] bool valid() const { return status == ObservedOrderStatus::valid; }
};

// Returns an explicit invalid status rather than concealing an unusable rate.
[[nodiscard]] ObservedOrder observed_order(double coarse_error, double fine_error,
                                           double coarse_spacing, double fine_spacing);

struct TrigonometricFixture {
    Grid grid;
    Vec3 lengths;
    std::vector<double> scalar;
    std::vector<double> q;
};

[[nodiscard]] double trigonometric_scalar(const Vec3& position, const Vec3& lengths);
[[nodiscard]] Vec3 trigonometric_gradient(const Vec3& position, const Vec3& lengths);
[[nodiscard]] double trigonometric_laplacian(const Vec3& position, const Vec3& lengths);

// Unit-cube controls used for 16^3/32^3 convergence comparisons.
[[nodiscard]] TrigonometricFixture make_cubic_trigonometric_fixture(std::size_t cells_per_axis);
// L=(1, 1.5, 2), isotropic h=1/16 and h=1/32 respectively.
[[nodiscard]] TrigonometricFixture make_anisotropic_trigonometric_fixture(bool fine);

// Independent SF-06 oracle on [0,1]^3.  The stencil deliberately lives in
// test code and uses long-double arithmetic; it must not call production face
// coefficient helpers or CUDA kernels.
struct AffineRhsFixture {
    Grid grid;
    std::vector<double> q;
};

[[nodiscard]] AffineRhsFixture make_affine_rhs_fixture(std::size_t cells_per_axis,
                                                        bool constant_q = false);
[[nodiscard]] std::vector<double> affine_rhs_discrete(const Grid& grid,
                                                       const std::vector<double>& q,
                                                       const Vec3& gradient);
[[nodiscard]] std::vector<double> affine_rhs_continuous(const Grid& grid,
                                                         const Vec3& gradient);
[[nodiscard]] std::vector<double> mean_zero_projected(const std::vector<double>& values);
[[nodiscard]] long double long_double_mean(const std::vector<double>& values);

}  // namespace macroflow3d::streamfunctions::reference
