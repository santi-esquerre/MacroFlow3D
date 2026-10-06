#pragma once

/**
 * @file SlabProductionSetup.cuh
 * @brief SF-33 N3: production inputs of the inlet-slab solver from the production stack:
 *        SF-18 periodic Gaussian field (or an analytic SF-30 control field), spectral vertex
 *        ln k / grad ln k, SF-19 affine-periodic Darcy solve, D-1 inlet labels from the SF-19
 *        inlet-face flux, SF-28 splines of the potential and of ln k, reference vD at every vertex.
 *
 * Authority: SF-33 specification item 7, understanding record section 5 ("Production path"),
 * SF-30 instrument (`apps/closure_gate/closure_gate.cuh`, steps 1-3 and SplineDirectionField).
 *
 * Field. A SlabFieldSource holds a UNIT-amplitude log-conductivity Y (cell-centred samples on the
 * N^3 grid of the unit cube, h = 1/N, cell centres (i + 1/2) h, layout i + N (j + N k), x1 fastest)
 * and its values at the slab vertices. The stage field of amplitude eps is k = exp(eps Y) (no
 * geometric-mean factor). Sources:
 *   - gaussian: SF-18 generate_periodic_gaussian_field (sigma2, ell, seed, normalize_variance;
 *     applied_scale / raw_variance / final_variance recorded). Vertex Y and grad Y by SPECTRAL
 *     evaluation of the band-limited samples: forward FFT, per axis the phase exp(-i k_m h/2) that
 *     moves (i + 1/2) h to i h, i k_m for the gradient, Nyquist zeroed explicitly, inverse FFT
 *     (cuFFT Z2Z, exact for the band-limited field to roundoff). Vertex plane N = plane 0
 *     (periodicity). The inlet-face values Y(0, (m2 + 1/2) h, (m3 + 1/2) h) by the same evaluation
 *     with the shift in x1 only.
 *   - analytic: a closure_fields.hpp field f (unit amplitude): cell samples f((i + 1/2) h, ...),
 *     vertex values and gradients analytically (analytic_log_conductivity_gradient, derived by hand
 *     from analytic_log_conductivity).
 *   - spectral: arbitrary user-provided band-limited cell samples evaluated as the gaussian source
 *     (tests; e.g. Y = 0 for the k = 1 control).
 *
 * Stage (build_production_stage, amplitude eps):
 *   (i)   lnk = eps Y_v, grad_lnk = eps grad Y_v on planes 0..N (slab layout), q = 1 / exp(lnk)
 *         (fill_q_from_lnk);
 *   (ii)  SF-19 solve_affine_periodic_flow on K = exp(eps Y_cell), qbar = e1, PCG rtol 1e-10, MG
 *         levels slab_auto_mg_levels(N) (= closure_gate::auto_mg_levels); all three correctors must
 *         converge, else status darcy_failed (nothing further is built);
 *   (iii) inlet v1 samples = U-face plane i = 0, U[j (N+1) + k (N+1) N] at ((j + 1/2) h, (k + 1/2)
 * h)
 *         -> InletLabels with offset (1/2, 1/2); min v1 <= 0 -> status inlet_backflow; u0_i at the
 *         inlet vertices = psi0_i - coord (GPU evaluation of the labels);
 *   (iv)  SF-28 GPU prefilter of h_tilde = potential_fluctuation() and of eps Y_cell; host copies
 * of the coefficients for the host direction field (SlabSplineDirectionField, = SF-30
 *         SplineDirectionField: g = G + grad s_h, k = exp(s_lnK));
 *   (v)   vD = exp(lnk_v) (G + grad s_h) at EVERY vertex (j h, m2 h, m3 h), planes 0..N (GPU
 * batched spline evaluation); vperp_in = (vD_2, vD_3) on plane 0; v_rms = RMS |vD|; (vi)
 * diagnostic: U-face v1 samples vs the spline flow v1 = k g1 at the same face centres (k = exp(eps
 * Y) from the source's face values): rms_rel = RMS(d) / RMS(U), max_rel = max |d_i| / |U_i|
 * (expected O(h^2)). The SlabStageInputs and SlabReferenceData (vD only; psi_or is filled by the
 * oracle) are filled. Setup cost is one-time per stage: host <-> device copies and synchronizations
 * are explicit here (this is not a hot path).
 */

#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/Grid3D.hpp"
#include "../../../core/Scalar.hpp"
#include "../../../numerics/interpolation/PeriodicTricubicBSpline.cuh"
#include "../../../runtime/CudaContext.cuh"
#include "../../flow/AffinePeriodicFlowSolver.cuh"
#include "../../stochastic/PeriodicGaussianField.cuh"
#include "InletLabels.cuh"
#include "InletSlabGrid.cuh"

#include <array>
#include <cmath>
#include <string>
#include <vector>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

// ================================================================================================
// Field sources
// ================================================================================================

/// SF-18 field specification (unit amplitude by default; the stage multiplies by eps).
struct ProductionFieldSpec {
    int N = 0;
    real sigma2 = 1.0;
    real ell = 0.0;
    unsigned long long seed = 0ULL;
    bool normalize_variance = true;
};

enum class SlabFieldKind { gaussian, analytic, spectral };

/// Analytic gradient of closure_gate::analytic_log_conductivity (unit amplitude) at (X, Y, Z).
/// Throws std::invalid_argument for an unknown field name.
void analytic_log_conductivity_gradient(const std::string& field, real X, real Y, real Z,
                                        real grad[3]);
/// Value of closure_gate::analytic_log_conductivity (re-exported for the tests and the source).
real analytic_log_conductivity_value(const std::string& field, real X, real Y, real Z);

/**
 * Spectral evaluation of band-limited periodic cell samples (unit cube, N^3, layout i + N(j + N k),
 * samples at (i + 1/2) h). Per axis d, to_vertex[d] selects the evaluation lattice i h (true) or
 * (i + 1/2) h (false). Outputs value and the three gradient components on that lattice, same
 * layout. Nyquist modes (any |m_d| = N/2) are set to zero. cuFFT Z2Z on ctx.cuda_stream(); plans
 * are created and destroyed per call; synchronizes. N must be even and >= 4.
 */
void spectral_periodic_evaluate(const CudaContext& ctx, int N,
                                const std::vector<real>& cell_samples,
                                const std::array<bool, 3>& to_vertex, std::vector<real>& value,
                                std::array<std::vector<real>, 3>& grad);

/// Vertex array on the cell-index lattice (i h, j h, k h) -> slab full array planes 0..N
/// (full_index(i, j, k); plane N = plane 0 by periodicity).
void cell_lattice_to_slab_full(int N, const std::vector<real>& lattice, std::vector<real>& full);

/// Deepest MG hierarchy with even levels and coarsest extent >= 4 (identical to
/// closure_gate::auto_mg_levels; equality checked in the tests).
int slab_auto_mg_levels(int N);

class SlabFieldSource {
  public:
    /// SF-18 generation (unit amplitude unless spec.sigma2 != 1) + spectral vertex values.
    static SlabFieldSource gaussian(CudaContext& ctx, const ProductionFieldSpec& spec);
    /// Analytic closure_fields.hpp field (control2d, lester2021, lester_brk, two_mode, generic3d).
    static SlabFieldSource analytic(const std::string& field, int N);
    /// User-provided band-limited cell samples (spectral vertex values), e.g. Y = 0.
    static SlabFieldSource spectral(CudaContext& ctx, const std::string& name, int N,
                                    std::vector<real> cell_samples);

    SlabFieldKind kind() const { return kind_; }
    const std::string& name() const { return name_; }
    int N() const { return n_; }
    bool has_sf18() const { return has_sf18_; }
    const physics::PeriodicGaussianFieldReport& sf18() const { return sf18_; }

    /// Unit-amplitude Y at the cell centres (layout i + N (j + N k)).
    const std::vector<real>& cell_samples() const { return cell_; }
    /// Unit-amplitude Y and grad Y at the slab vertices, full layout planes 0..N.
    const std::vector<real>& vertex_value() const { return vtx_; }
    const std::vector<real>& vertex_grad(int d) const { return vgrad_[d]; }
    /// Unit-amplitude Y at the inlet face centres (0, (m2 + 1/2) h, (m3 + 1/2) h), layout
    /// m3 + N m2.
    const std::vector<real>& inlet_face_value() const { return face_; }

  private:
    void fill_spectral(CudaContext& ctx);
    SlabFieldKind kind_ = SlabFieldKind::spectral;
    std::string name_;
    int n_ = 0;
    bool has_sf18_ = false;
    physics::PeriodicGaussianFieldReport sf18_{};
    std::vector<real> cell_, vtx_, face_;
    std::array<std::vector<real>, 3> vgrad_;
};

// ================================================================================================
// Direction field (SF-30 SplineDirectionField contract)
// ================================================================================================

/// g = G + grad s_h, k = exp(s_lnK) at an unwrapped point (host views; evaluate_point owns the
/// periodic reduction). Functor contract of closure_gate::integrate_streamline. Const, thread-safe.
struct SlabSplineDirectionField {
    interpolation::PeriodicTricubicBSplineView potential{};
    interpolation::PeriodicTricubicBSplineView log_conductivity{};
    double G[3] = {0.0, 0.0, 0.0};

    void operator()(const double x[3], double g[3], double& k) const {
        real v = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
        interpolation::evaluate_point(potential, x[0], x[1], x[2], v, gx, gy, gz);
        g[0] = G[0] + gx;
        g[1] = G[1] + gy;
        g[2] = G[2] + gz;
        real y = 0.0, yx = 0.0, yy = 0.0, yz = 0.0;
        interpolation::evaluate_point(log_conductivity, x[0], x[1], x[2], y, yx, yy, yz);
        k = std::exp(y);
    }
};

// ================================================================================================
// Production stage
// ================================================================================================

enum class SlabProductionStatus { ok = 0, darcy_failed = 1, inlet_backflow = 2 };
const char* to_string(SlabProductionStatus s);

struct ProductionStageOptions {
    real pcg_rtol = 1e-10;
    int pcg_max_iter = -1; ///< -1: library default (solvers::ProjectedPCGConfig)
    int mg_levels = 0;     ///< 0: slab_auto_mg_levels(N)
    int label_chunk = 16384;
};

struct ProductionStageReport {
    SlabProductionStatus status = SlabProductionStatus::ok;
    std::string field;
    real eps = 0.0;
    int N = 0;
    bool has_sf18 = false;
    physics::PeriodicGaussianFieldReport sf18{};
    int mg_levels = 0;
    bool darcy_converged = false;
    physics::AffinePeriodicFlowReport darcy{};
    real inlet_vmin = std::nan("");
    real inlet_mean = std::nan(""); ///< mean of the U-face plane-0 samples
    real Q0 = std::nan("");
    real v1_diff_rms_rel = std::nan(""); ///< RMS(U - k g1) / RMS(U) at the inlet face centres
    real v1_diff_max_rel = std::nan(""); ///< max |U - k g1| / |U|
    real v_rms = std::nan("");
    real spline_potential_gpu_vs_host_rel = std::nan("");
    /// Multi-line human-readable stage report (SF-19 G, PCG iterations / residuals, SF-18 scale,
    /// inlet statistics, v1 difference).
    std::string summary() const;
};

class ProductionStage {
  public:
    ProductionStageReport report;
    InletSlabGrid grid;
    Grid3D cell_grid;
    SlabStageInputs inputs;             ///< filled iff ok()
    SlabReferenceData reference;        ///< vD filled iff ok(); psi_or left to the oracle
    InletLabels labels;                 ///< built and device-prepared iff ok()
    std::vector<real> inlet_v1_samples; ///< U-face plane 0, layout m3 + N m2
    std::vector<real>
        potential_coefficients;          ///< host SF-28 coefficients of h_tilde (GPU prefilter)
    std::vector<real> logk_coefficients; ///< host SF-28 coefficients of eps Y_cell
    real G[3] = {0.0, 0.0, 0.0};

    bool ok() const { return report.status == SlabProductionStatus::ok; }
    /// Host direction field over the coefficient vectors (valid while this stage lives).
    SlabSplineDirectionField direction_field() const;
};

/// Builds one stage (see the file header). Never throws for darcy_failed / inlet_backflow (status);
/// throws for invalid arguments and CUDA errors.
ProductionStage
build_production_stage(CudaContext& ctx, const InletSlabGrid& grid, const SlabFieldSource& source,
                       real eps, const ProductionStageOptions& opt = ProductionStageOptions{});

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
