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

// SF-09 nonlinear-source controls.
//
// Conventions (must match production `src/physics/streamfunctions` exactly):
//   c  = g1 cross g2                     (right-handed cross product)
//   d  = |c|^2 + (epsilon * v_rms)^2     (explicit, dimensionless regularization)
//   S1 = ((B cross g1) . c) / d
//   S2 = ((B cross g2) . c) / d
// `epsilon` is dimensionless; `v_rms` must be finite and strictly positive so
// the regularization term carries physical units. There is no hidden floor:
// the only regularization is `(epsilon*v_rms)^2`.
struct NonlinearSourceReferenceConfig {
    double epsilon{};
    double v_rms{};
};

// c, the regularized denominator d, and both sources, all in the same
// x-fastest grid layout as VectorField/scalar fields elsewhere in this file.
struct NonlinearSourceFields {
    VectorField c;
    std::vector<double> denominator;
    std::vector<double> s1;
    std::vector<double> s2;
};

// Exact-analytic regularized source reference on a TotalGradientFixture: uses
// the analytic total gradients (SF-07) and analytic periodic Hessians (SF-08)
// to build B, c, d, S1, and S2 pointwise in double precision. Suitable as the
// continuum reference for discretization-error / convergence-order studies.
//
// analytic_hessian_vector_b (and therefore this function) always evaluates
// the fixed smooth-fixture analytic formulas for GradientFixtureField::psi1
// and psi2, independent of the fixture's stored fluctuation arrays; it is
// only a valid continuum reference for make_total_gradient_fixture and
// make_parallel_total_gradient_fixture (which share that analytic form), NOT
// for make_pure_affine_total_gradient_fixture. The pure-affine/B=0 identity
// is instead verified by construction on the discrete path: on
// make_parallel_total_gradient_fixture (psi2_tilde = scale*psi1_tilde, g2 =
// scale*g1), centered_hessian_vector_b_oracle gives B = 0 up to roundoff, so
// centered_nonlinear_source_oracle below gives S1 = S2 = 0 up to roundoff
// regardless of c and d.
[[nodiscard]] NonlinearSourceFields analytic_nonlinear_source_reference(
    const TotalGradientFixture& fixture, const NonlinearSourceReferenceConfig& config);

// Independent long-double discrete source oracle. Takes discrete total
// gradients g1, g2 (e.g. from centered_total_gradient_oracle) and discrete B
// (e.g. from centered_hessian_vector_b_oracle) and applies exactly the
// production formulas above in long double. It never calls any production
// (src/) API. All three input VectorFields are validated to be finite; use
// double_precision_nonlinear_source_mirror below for nonfinite-input
// diagnostics.
[[nodiscard]] NonlinearSourceFields centered_nonlinear_source_oracle(
    const Grid& grid, const VectorField& g1_total_gradient, const VectorField& g2_total_gradient,
    const VectorField& b, const NonlinearSourceReferenceConfig& config);

// Diagnostics bundle returned by double_precision_nonlinear_source_mirror:
// the plain-double fields plus exact-count degeneracy/nonfinite diagnostics.
struct NonlinearSourceMirrorDiagnostics {
    NonlinearSourceFields fields;
    std::size_t nonfinite_s1_count{};
    std::size_t nonfinite_s2_count{};
    // Per threshold tau_t: count of cells with |c|^2 < (tau_t*v_rms)^2
    // (strict less-than, plain double comparison; documented production
    // degeneracy semantics).
    std::vector<std::size_t> degenerate_counts;
    // Per threshold tau_t: min over cells of
    // | |c|^2 - (tau_t*v_rms)^2 | / max((tau_t*v_rms)^2, 1), used to assert
    // that the degenerate_counts are robust to roundoff near the boundary.
    std::vector<double> degenerate_separation;
};

// Double-precision CPU mirror of the exact production comparison semantics.
// Unlike centered_nonlinear_source_oracle and analytic_nonlinear_source_reference,
// this function does NOT validate that g1, g2, or b are finite: it is the
// authority for counting nonfinite S1/S2 cells and degeneracy-threshold cells
// in plain double, matching bit-for-bit what a GPU mirror must reproduce.
// Only field-size consistency, epsilon, v_rms, and threshold finiteness are
// validated.
[[nodiscard]] NonlinearSourceMirrorDiagnostics double_precision_nonlinear_source_mirror(
    const VectorField& g1_total_gradient, const VectorField& g2_total_gradient,
    const VectorField& b, double epsilon, double v_rms,
    const std::vector<double>& degeneracy_thresholds);

// Near-degenerate fixture: psi2 := parallel_scale*psi1 + perturbation_scale*psi2,
// applied to both the affine gradient and the periodic fluctuation of the
// smooth SF-07 control. As perturbation_scale -> 0 this drives g2 toward the
// exactly-parallel case (make_parallel_total_gradient_fixture), so |c|
// shrinks in a controlled, non-exactly-zero way suitable for degeneracy-count
// tests.
[[nodiscard]] TotalGradientFixture make_near_degenerate_total_gradient_fixture(
    std::size_t cells_per_axis, double parallel_scale, double perturbation_scale);

// A single deterministic nonfinite-value injection: replace the VectorField
// value at cell_index with replacement (which may contain Inf/NaN).
struct NonfiniteInjection {
    std::size_t cell_index{};
    Vec3 replacement{};
};

// Returns a copy of field with the listed injections applied. Injections are
// applied in order (later entries win on repeated indices) at explicit,
// caller-chosen cell indices so CPU and GPU test cases flag exactly the same
// cells.
[[nodiscard]] VectorField inject_nonfinite_values(const VectorField& field,
                                                   const std::vector<NonfiniteInjection>& injections);

}  // namespace macroflow3d::streamfunctions::reference
