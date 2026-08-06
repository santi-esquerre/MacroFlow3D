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

// Component-wise vector field in the same x-fastest layout as Grid scalar fields.
struct VectorField {
    std::vector<double> x;
    std::vector<double> y;
    std::vector<double> z;
};

// Symmetric Hessian stored as its six independent components.  This is a
// point value for analytic controls only; SF-08 deliberately does not create
// grid-sized Hessian fields.
struct SymmetricHessian {
    double xx{};
    double xy{};
    double xz{};
    double yy{};
    double yz{};
    double zz{};
};

// The three vector fields exposed by the SF-08 test oracle.  The two direct
// products remain visible for component-wise convergence checks, while B is
// their signed difference required by Lester equation (14).
struct HessianVectorBFields {
    VectorField hessian_psi2_times_gradient_psi1;
    VectorField hessian_psi1_times_gradient_psi2;
    VectorField b;
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

// SF-07 total-gradient controls. All levels use the same rectangular physical
// domain; its deliberately unequal lengths make dx, dy, and dz distinct.
enum class GradientFixtureField { psi1, psi2 };

struct TotalGradientFixture {
    Grid grid;
    Vec3 lengths;
    Vec3 psi1_affine_gradient;
    Vec3 psi2_affine_gradient;
    std::vector<double> psi1_fluctuation;
    std::vector<double> psi2_fluctuation;
};

// Construct the 16^3, 32^3, or 64^3 smooth periodic control with two
// distinct fluctuations and nontrivial affine gradients.
[[nodiscard]] TotalGradientFixture make_total_gradient_fixture(std::size_t cells_per_axis);
// Same grid and affine gradients as the smooth control, but identically zero
// periodic fluctuations for pure-affine exactness tests.
[[nodiscard]] TotalGradientFixture make_pure_affine_total_gradient_fixture(
    std::size_t cells_per_axis);

[[nodiscard]] double total_gradient_periodic_scalar(GradientFixtureField field,
                                                     const Vec3& position,
                                                     const Vec3& lengths);
[[nodiscard]] Vec3 total_gradient_periodic_analytic(GradientFixtureField field,
                                                     const Vec3& position,
                                                     const Vec3& lengths);
// Analytic gradient of affine dot x plus the selected periodic fluctuation.
[[nodiscard]] Vec3 total_gradient_analytic(GradientFixtureField field,
                                           const Vec3& position,
                                           const Vec3& lengths,
                                           const Vec3& affine_gradient);

// Independent long-double, centered second-order total-gradient oracle. It
// wraps x, y, and z separately via Grid/index/wrap_index and adds the affine
// contribution after differencing the stored periodic fluctuation.
[[nodiscard]] VectorField centered_total_gradient_oracle(const Grid& grid,
                                                          const std::vector<double>& fluctuation,
                                                          const Vec3& affine_gradient);

// Hessians differentiate the periodic SF-07 fluctuations only.  The affine
// contribution is intentionally absent because its Hessian is identically
// zero.
[[nodiscard]] SymmetricHessian total_gradient_periodic_hessian_analytic(
    GradientFixtureField field, const Vec3& position, const Vec3& lengths);
[[nodiscard]] Vec3 symmetric_hessian_vector_product(const SymmetricHessian& hessian,
                                                     const Vec3& gradient);

// Analytic pointwise controls on the ordinary SF-07 smooth fixture.  The
// gradients are total (periodic plus affine), whereas Hessians are periodic.
[[nodiscard]] HessianVectorBFields analytic_hessian_vector_b(
    const TotalGradientFixture& fixture);

// Independent long-double discrete oracle.  It samples center, six axial
// neighbours, and twelve edge-diagonal neighbours of each fluctuation;
// Hessian fields are never assembled.  Input gradients must be the total
// gradients for the corresponding psi fields at the same cell centers.
[[nodiscard]] HessianVectorBFields centered_hessian_vector_b_oracle(
    const Grid& grid, const std::vector<double>& psi1_fluctuation,
    const std::vector<double>& psi2_fluctuation, const VectorField& psi1_total_gradient,
    const VectorField& psi2_total_gradient);

// Exact discrete parallel-gradient control: psi2_tilde = scale*psi1_tilde
// and g2 = scale*g1, hence H(psi2)g1 - H(psi1)g2 is zero up to roundoff.
[[nodiscard]] TotalGradientFixture make_parallel_total_gradient_fixture(
    std::size_t cells_per_axis, double scale);

}  // namespace macroflow3d::streamfunctions::reference
