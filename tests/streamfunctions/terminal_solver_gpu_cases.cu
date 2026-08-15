#include "streamfunction_operator_test_cases.hpp"

#include "src/core/BCSpec.hpp"
#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/multigrid/mg_types.hpp"
#include "src/numerics/blas/axpy.cuh"
#include "src/numerics/blas/copy.cuh"
#include "src/numerics/blas/dot.cuh"
#include "src/numerics/blas/fill.cuh"
#include "src/numerics/blas/scal.cuh"
#include "src/numerics/constraints/MeanZeroProjector.cuh"
#include "src/numerics/operators/lester_positive_diffusion_operator.cuh"
#include "src/physics/flow/AffinePeriodicFlowSolver.cuh"
#include "src/physics/stochastic/PeriodicGaussianField.cuh"
#include "src/physics/streamfunctions/BlockDiagonalMGPreconditioner.cuh"
#include "src/physics/streamfunctions/ContinuationController.hpp"
#include "src/physics/streamfunctions/CoupledGmres.cuh"
#include "src/physics/streamfunctions/Diagnostics.cuh"
#include "src/physics/streamfunctions/DifferentialOperators.cuh"
#include "src/physics/streamfunctions/JacobianVectorProduct.cuh"
#include "src/physics/streamfunctions/NonlinearSources.cuh"
#include "src/physics/streamfunctions/ResidualEvaluator.cuh"
#include "src/physics/streamfunctions/ShiftedJacobianOperator.cuh"
#include "src/physics/streamfunctions/StreamfunctionSolver.cuh"
#include "src/physics/streamfunctions/StreamfunctionTypes.hpp"
#include "src/physics/streamfunctions/StreamfunctionWorkspace.cuh"
#include "src/physics/streamfunctions/affine_gauge.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <deque>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// SF-25 T02: the D-gate diagnostic (activation decisions E2-E5,
// `docs/plans/active/lester-eq14/increments/SF-25-terminal-manifold-solver.md`,
// implemented VERBATIM per the protocol prespecified BEFORE this file was
// written) plus a cheap unit case for the T01 `ShiftedJacobianOperator`
// enabler. No fixture constant, seed, tolerance, or threshold below is
// adjusted after seeing a result (STOP semantics, matching every other
// PRESPECIFIED-gate file in this test suite, e.g. `gmres_gpu_cases.cu`,
// `newton_gpu_cases.cu`). This file never touches `src/**`.
//
// Case 1 (`terminal_shifted_apply_unit`, cheap, in the aggregated `cases()`
// registry): exactness of `ShiftedJacobianOperator::apply` against an
// independently recomputed `J p + mu*A p`, the `mu=0` bitwise passthrough,
// and the `set_mu`/unprepared-`apply` fail-fast contract, on the 16^3
// manufactured trig-K fixture copied verbatim from `newton_gpu_cases.cu`'s
// `ManufacturedProblemBuffers` (amplitude 0.25, uniform Darcy v=(1,0,0),
// benchmark(1) gauge -- the SAME provenance the SF-24 equivalence case (G1)
// uses).
//
// Case 2 (`terminal_dgate_diagnostic`, HEAVY, separate registry like
// `anderson_stall_fixture_*`/`newton_difficult_case`): the D-gate itself.
// E2 freezes a genuine Picard/Anderson plateau state (sigma_Y^2=1, 32^3,
// seed 12345, corr_length 8, lambda=0.5125 warm-started from the accepted
// (lambda=0.5, eta=1) SF-21 smoke state, NEWTON DISABLED per the SF-25
// activation bitácora's E2 decision -- the freeze characterizes the
// SF-21 plateau regime, not the SF-24 Newton budget-exhaustion regime).
// The per-lambda conductivity/flow setup (Y_att = lambda*Y, K_att =
// exp(Y_att), the SF-19 affine flow solve, the log_conductivity_y problem
// view) is mirrored EXACTLY from `ContinuationController.cu`'s
// `build_attempt`/`make_problem_view` lambdas (cited by line number at each
// mirrored step below). E3 sweeps the shift `mu` on the frozen system with
// the accepted GMRES; E4 is a non-blocking spectral probe; E5 is a bounded
// Levenberg-Marquardt mini-solve using the E3-calibrated `theta`. See the
// D-gate protocol in the increment specification and
// `docs/decisions/2026-08-14-manifold-robust-terminal-solver.md` for the
// full scientific rationale this case tests against.

namespace macroflow3d::streamfunctions::test {
namespace {

constexpr double kPi = 3.14159265358979323846264338327950288;

// ---------------------------------------------------------------------------
// Shared grid/BC/download/label helpers, mirrored from newton_gpu_cases.cu /
// heterogeneity_continuation_gpu_cases.cu / gmres_gpu_cases.cu.
// ---------------------------------------------------------------------------

[[nodiscard]] Grid3D isotropic_grid(int n, real domain_length = real{1}) {
    const real h = domain_length / static_cast<real>(n);
    return Grid3D{n, n, n, h, h, h};
}

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

[[nodiscard]] bool bitwise_equal(const std::vector<real>& a, const std::vector<real>& b) {
    if (a.size() != b.size()) return false;
    if (a.empty()) return true;
    return std::memcmp(a.data(), b.data(), a.size() * sizeof(real)) == 0;
}

[[nodiscard]] const char* solve_status_label(StreamfunctionSolveStatus status) {
    switch (status) {
        case StreamfunctionSolveStatus::not_run: return "not_run";
        case StreamfunctionSolveStatus::converged: return "converged";
        case StreamfunctionSolveStatus::not_converged: return "not_converged";
        case StreamfunctionSolveStatus::invalid_problem: return "invalid_problem";
        default: return "unknown";
    }
}

[[nodiscard]] const char* exit_reason_label(PicardExitReason reason) {
    switch (reason) {
        case PicardExitReason::none: return "none";
        case PicardExitReason::converged: return "converged";
        case PicardExitReason::budget_exhausted: return "budget_exhausted";
        case PicardExitReason::linear_block_failure: return "linear_block_failure";
        case PicardExitReason::stagnated: return "stagnated";
        case PicardExitReason::omega_floor_rejected: return "omega_floor_rejected";
        case PicardExitReason::newton_exhausted: return "newton_exhausted";
        case PicardExitReason::newton_budget_exhausted: return "newton_budget_exhausted";
        default: return "unknown";
    }
}

[[nodiscard]] const char* gmres_status_label(CoupledGmresStatus status) {
    switch (status) {
        case CoupledGmresStatus::converged: return "converged";
        case CoupledGmresStatus::max_iterations: return "max_iterations";
        case CoupledGmresStatus::breakdown: return "breakdown";
        case CoupledGmresStatus::nonfinite: return "nonfinite";
        default: return "unknown";
    }
}

[[nodiscard]] const char* heterogeneity_status_label(HeterogeneityStatus status) {
    switch (status) {
        case HeterogeneityStatus::reached_target: return "reached_target";
        case HeterogeneityStatus::baseline_failed: return "baseline_failed";
        case HeterogeneityStatus::lambda_floor_exhausted: return "lambda_floor_exhausted";
        case HeterogeneityStatus::epsilon_floor_exhausted: return "epsilon_floor_exhausted";
        case HeterogeneityStatus::invalid_problem: return "invalid_problem";
        default: return "unknown";
    }
}

// ---------------------------------------------------------------------------
// SF-25 Phase-1 (P1-I) shared helpers: host-side percentile/concentration
// reducers and the SF-11-consistent pointwise defect-field replication used
// by `case_terminal_shelf_probe_phase1`'s S6 battery. Route taken (recorded
// in the P1-I worker report): `PhysicalDiagnosticsWorkspace`'s per-cell
// scratch (`dot1_field_`, `dot2_field_`, `vpsi_c*_`, `vd_c*_`, `abs_c_field_`)
// is PRIVATE with no public accessor, so the aggregate `PhysicalDiagnosticsReport`
// is reused verbatim (via the public enqueue/synchronize pair,
// `Diagnostics.cuh`) for the reduction-level SF-11 metrics, while the
// PER-CELL fields needed for percentiles/concentration are replicated on the
// HOST, reusing the PUBLIC `enqueue_total_streamfunction_gradients` (SF-07,
// `DifferentialOperators.cuh`) for `g1`/`g2` and then reproducing, exactly,
// the interpolate-then-cross face reconstruction, face-to-center averaging,
// Darcy-invariance dot products, and `|c| = |grad(psi1) x grad(psi2)|`
// formulas from `Diagnostics.cu`'s `reconstruct_and_face_diagnostics_kernel`,
// `face_to_center_kernel`, and `cell_metrics_kernel` (cited inline below).
// 32^3 = 32768 cells makes this trivially cheap host-side.
// ---------------------------------------------------------------------------

// Linear interpolation percentile (0..100) over a copy of `values` (sorted in
// place on the copy; caller's vector is untouched).
[[nodiscard]] double host_percentile(std::vector<double> values, double p) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::sort(values.begin(), values.end());
    const double idx = p / 100.0 * static_cast<double>(values.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(std::floor(idx));
    const std::size_t hi = static_cast<std::size_t>(std::ceil(idx));
    if (lo == hi) return values[lo];
    const double frac = idx - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

// Fraction of the total SQUARED value carried by the top `fraction` (e.g.
// 0.01 for the top 1%) of cells, by |value|.
[[nodiscard]] double host_top_fraction_energy(const std::vector<double>& values, double fraction) {
    if (values.empty()) return std::numeric_limits<double>::quiet_NaN();
    std::vector<double> sq(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) sq[i] = values[i] * values[i];
    const double total = std::accumulate(sq.begin(), sq.end(), 0.0);
    std::sort(sq.begin(), sq.end());
    const std::size_t top_k =
        std::max<std::size_t>(1, static_cast<std::size_t>(std::ceil(fraction * static_cast<double>(sq.size()))));
    const double top_sum = std::accumulate(sq.end() - static_cast<std::ptrdiff_t>(top_k), sq.end(), 0.0);
    return total > 0.0 ? top_sum / total : std::numeric_limits<double>::quiet_NaN();
}

struct PointwiseDiagnosticsHost {
    std::vector<double> invariance_defect1; // |v_D . grad(psi1)| at cell centers.
    std::vector<double> invariance_defect2; // |v_D . grad(psi2)| at cell centers.
    std::vector<double> reconstruction_error; // |v_psi_c - v_D_c| at cell centers.
    std::vector<double> abs_c;                // |grad(psi1) x grad(psi2)| at cell centers.
};

// Host replication of Diagnostics.cu's per-cell defect fields at ONE state.
// `u1`/`u2` are the periodic fluctuations (already-accepted state);
// `darcy` is the SAME CompactMAC Darcy field the state was solved against.
[[nodiscard]] PointwiseDiagnosticsHost compute_pointwise_diagnostics_host(
    CudaContext& ctx, const Grid3D& grid, DeviceSpan<real> u1, DeviceSpan<real> u2, const AffineGauge& gauge,
    const CompactMacVelocityConstView& darcy) {
    const int nx = grid.nx, ny = grid.ny, nz = grid.nz;
    const std::size_t n = grid.num_cells();

    // SF-07 total gradients, reused via the public enqueue call
    // (DifferentialOperators.cuh).
    DeviceBuffer<real> g1x(n), g1y(n), g1z(n), g2x(n), g2y(n), g2z(n);
    const TotalStreamfunctionGradientOutput grad_out{g1x.span(), g1y.span(), g1z.span(),
                                                      g2x.span(), g2y.span(), g2z.span()};
    enqueue_total_streamfunction_gradients(ctx, grid, PeriodicStreamfunctionFluctuations{u1, u2}, gauge, grad_out);
    ctx.synchronize();

    const std::vector<real> g1x_h = download(DeviceSpan<const real>(g1x.span()));
    const std::vector<real> g1y_h = download(DeviceSpan<const real>(g1y.span()));
    const std::vector<real> g1z_h = download(DeviceSpan<const real>(g1z.span()));
    const std::vector<real> g2x_h = download(DeviceSpan<const real>(g2x.span()));
    const std::vector<real> g2y_h = download(DeviceSpan<const real>(g2y.span()));
    const std::vector<real> g2z_h = download(DeviceSpan<const real>(g2z.span()));

    const std::vector<real> darcy_u_h = download(darcy.u);
    const std::vector<real> darcy_v_h = download(darcy.v);
    const std::vector<real> darcy_w_h = download(darcy.w);

    const std::size_t y_stride = static_cast<std::size_t>(nx);
    const std::size_t face_u_j_stride = static_cast<std::size_t>(nx + 1);
    const std::size_t face_u_k_stride = face_u_j_stride * static_cast<std::size_t>(ny);
    const std::size_t face_v_k_stride = y_stride * static_cast<std::size_t>(ny + 1);
    const std::size_t face_w_k_stride = y_stride * static_cast<std::size_t>(ny);

    const std::size_t size_u = face_u_k_stride * static_cast<std::size_t>(nz);
    const std::size_t size_v = face_v_k_stride * static_cast<std::size_t>(nz);
    const std::size_t size_w = face_w_k_stride * static_cast<std::size_t>(nz + 1);

    std::vector<double> vpsi_u(size_u), vpsi_v(size_v), vpsi_w(size_w);

    const auto cell_index = [&](int i, int j, int k) -> std::size_t {
        return static_cast<std::size_t>(i) +
               y_stride * (static_cast<std::size_t>(j) + static_cast<std::size_t>(ny) * static_cast<std::size_t>(k));
    };

    // Interpolate-then-cross face reconstruction, EXACTLY mirroring
    // Diagnostics.cu's reconstruct_and_face_diagnostics_kernel (lines
    // ~112-245), including the periodic duplicate planes.
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const std::size_t b = cell_index(i, j, k);
                {
                    const int a_i = (i == 0) ? nx - 1 : i - 1;
                    const std::size_t a = cell_index(a_i, j, k);
                    const double t1y = 0.5 * (static_cast<double>(g1y_h[a]) + static_cast<double>(g1y_h[b]));
                    const double t1z = 0.5 * (static_cast<double>(g1z_h[a]) + static_cast<double>(g1z_h[b]));
                    const double t2y = 0.5 * (static_cast<double>(g2y_h[a]) + static_cast<double>(g2y_h[b]));
                    const double t2z = 0.5 * (static_cast<double>(g2z_h[a]) + static_cast<double>(g2z_h[b]));
                    const double uval = t1y * t2z - t1z * t2y;
                    const std::size_t face = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * face_u_j_stride +
                                             static_cast<std::size_t>(k) * face_u_k_stride;
                    vpsi_u[face] = uval;
                    if (i == 0) {
                        vpsi_u[static_cast<std::size_t>(nx) + static_cast<std::size_t>(j) * face_u_j_stride +
                              static_cast<std::size_t>(k) * face_u_k_stride] = uval;
                    }
                }
                {
                    const int b_j = (j == 0) ? ny - 1 : j - 1;
                    const std::size_t a = cell_index(i, b_j, k);
                    const double t1z = 0.5 * (static_cast<double>(g1z_h[a]) + static_cast<double>(g1z_h[b]));
                    const double t1x = 0.5 * (static_cast<double>(g1x_h[a]) + static_cast<double>(g1x_h[b]));
                    const double t2z = 0.5 * (static_cast<double>(g2z_h[a]) + static_cast<double>(g2z_h[b]));
                    const double t2x = 0.5 * (static_cast<double>(g2x_h[a]) + static_cast<double>(g2x_h[b]));
                    const double vval = t1z * t2x - t1x * t2z;
                    const std::size_t face = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * y_stride +
                                             static_cast<std::size_t>(k) * face_v_k_stride;
                    vpsi_v[face] = vval;
                    if (j == 0) {
                        vpsi_v[static_cast<std::size_t>(i) + static_cast<std::size_t>(ny) * y_stride +
                              static_cast<std::size_t>(k) * face_v_k_stride] = vval;
                    }
                }
                {
                    const int c_k = (k == 0) ? nz - 1 : k - 1;
                    const std::size_t a = cell_index(i, j, c_k);
                    const double t1x = 0.5 * (static_cast<double>(g1x_h[a]) + static_cast<double>(g1x_h[b]));
                    const double t1y = 0.5 * (static_cast<double>(g1y_h[a]) + static_cast<double>(g1y_h[b]));
                    const double t2x = 0.5 * (static_cast<double>(g2x_h[a]) + static_cast<double>(g2x_h[b]));
                    const double t2y = 0.5 * (static_cast<double>(g2y_h[a]) + static_cast<double>(g2y_h[b]));
                    const double wval = t1x * t2y - t1y * t2x;
                    const std::size_t face = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * y_stride +
                                             static_cast<std::size_t>(k) * face_w_k_stride;
                    vpsi_w[face] = wval;
                    if (k == 0) {
                        vpsi_w[static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * y_stride +
                              static_cast<std::size_t>(nz) * face_w_k_stride] = wval;
                    }
                }
            }
        }
    }

    PointwiseDiagnosticsHost out;
    out.invariance_defect1.resize(n);
    out.invariance_defect2.resize(n);
    out.reconstruction_error.resize(n);
    out.abs_c.resize(n);

    // Face-to-center averaging + cell-centered metrics, EXACTLY mirroring
    // Diagnostics.cu's face_to_center_kernel and cell_metrics_kernel's dot1/
    // dot2/|c| formulas (lines ~247-419).
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const std::size_t cell = cell_index(i, j, k);

                const std::size_t face_u0 = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * face_u_j_stride +
                                            static_cast<std::size_t>(k) * face_u_k_stride;
                const std::size_t face_u1 = face_u0 + 1;
                const double vpsi_cx = 0.5 * (vpsi_u[face_u0] + vpsi_u[face_u1]);
                const double vd_cx = 0.5 * (static_cast<double>(darcy_u_h[face_u0]) + static_cast<double>(darcy_u_h[face_u1]));

                const std::size_t face_v0 = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * y_stride +
                                            static_cast<std::size_t>(k) * face_v_k_stride;
                const std::size_t face_v1 = face_v0 + y_stride;
                const double vpsi_cy = 0.5 * (vpsi_v[face_v0] + vpsi_v[face_v1]);
                const double vd_cy = 0.5 * (static_cast<double>(darcy_v_h[face_v0]) + static_cast<double>(darcy_v_h[face_v1]));

                const std::size_t face_w0 = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * y_stride +
                                            static_cast<std::size_t>(k) * face_w_k_stride;
                const std::size_t face_w1 = face_w0 + face_w_k_stride;
                const double vpsi_cz = 0.5 * (vpsi_w[face_w0] + vpsi_w[face_w1]);
                const double vd_cz = 0.5 * (static_cast<double>(darcy_w_h[face_w0]) + static_cast<double>(darcy_w_h[face_w1]));

                const double g1xv = static_cast<double>(g1x_h[cell]), g1yv = static_cast<double>(g1y_h[cell]),
                             g1zv = static_cast<double>(g1z_h[cell]);
                const double g2xv = static_cast<double>(g2x_h[cell]), g2yv = static_cast<double>(g2y_h[cell]),
                             g2zv = static_cast<double>(g2z_h[cell]);

                const double dot1 = vd_cx * g1xv + vd_cy * g1yv + vd_cz * g1zv;
                const double dot2 = vd_cx * g2xv + vd_cy * g2yv + vd_cz * g2zv;
                out.invariance_defect1[cell] = std::abs(dot1);
                out.invariance_defect2[cell] = std::abs(dot2);

                const double cx = g1yv * g2zv - g1zv * g2yv;
                const double cy = g1zv * g2xv - g1xv * g2zv;
                const double cz = g1xv * g2yv - g1yv * g2xv;
                out.abs_c[cell] = std::sqrt(cx * cx + cy * cy + cz * cz);

                const double ddx = vpsi_cx - vd_cx, ddy = vpsi_cy - vd_cy, ddz = vpsi_cz - vd_cz;
                out.reconstruction_error[cell] = std::sqrt(ddx * ddx + ddy * ddy + ddz * ddz);
            }
        }
    }
    return out;
}

struct S6StateEvidence {
    PhysicalDiagnosticsReport sf11;
    PointwiseDiagnosticsHost pointwise;
};

// The SF-11 aggregate report reused verbatim (public enqueue/synchronize
// pair) plus the host-replicated pointwise fields, both at the SAME
// `u1`/`u2`/`darcy` state.
[[nodiscard]] S6StateEvidence evaluate_s6_state(CudaContext& ctx, const Grid3D& grid, DeviceSpan<real> u1,
                                                DeviceSpan<real> u2, const AffineGauge& gauge,
                                                const CompactMacVelocityConstView& darcy,
                                                const PhysicalDiagnosticsConfig& diag_config) {
    S6StateEvidence out;
    StreamfunctionDiagnosticsWorkspace diag_ws;
    diag_ws.prepare(grid);
    DeviceBuffer<real> vpsi_u(compact_mac_u_size(grid)), vpsi_v(compact_mac_v_size(grid)),
        vpsi_w(compact_mac_w_size(grid));
    enqueue_streamfunction_physical_diagnostics(
        ctx, grid, PeriodicStreamfunctionFluctuations{u1, u2}, gauge, darcy, diag_config,
        CompactMacVelocityView{vpsi_u.span(), vpsi_v.span(), vpsi_w.span()}, diag_ws);
    out.sf11 = synchronize_streamfunction_physical_diagnostics_report(ctx, grid, diag_config, diag_ws);
    out.pointwise = compute_pointwise_diagnostics_host(ctx, grid, u1, u2, gauge, darcy);
    return out;
}

void print_s6_state_report(const char* label, const S6StateEvidence& ev, double epsilon_scale) {
    std::cout << label << " SF11 e_v=" << ev.sf11.e_v << " invariance_e_psi1=" << ev.sf11.invariance_e_psi1
              << " invariance_e_psi2=" << ev.sf11.invariance_e_psi2 << " e_div=" << ev.sf11.e_div
              << " c_min=" << ev.sf11.c_min << " c_max=" << ev.sf11.c_max << " c_mean=" << ev.sf11.c_mean
              << " v_d_rms=" << ev.sf11.v_d_rms << '\n';

    const double p[5] = {50.0, 90.0, 99.0, 99.9, 100.0};
    const char* pname[5] = {"p50", "p90", "p99", "p99.9", "max"};
    for (int t = 0; t < 5; ++t) {
        std::cout << label << " invariance_defect1 " << pname[t] << "="
                  << host_percentile(ev.pointwise.invariance_defect1, p[t]) << '\n';
    }
    for (int t = 0; t < 5; ++t) {
        std::cout << label << " invariance_defect2 " << pname[t] << "="
                  << host_percentile(ev.pointwise.invariance_defect2, p[t]) << '\n';
    }
    for (int t = 0; t < 5; ++t) {
        std::cout << label << " reconstruction_error " << pname[t] << "="
                  << host_percentile(ev.pointwise.reconstruction_error, p[t]) << '\n';
    }

    const double conc1_1 = host_top_fraction_energy(ev.pointwise.invariance_defect1, 0.01);
    const double conc1_01 = host_top_fraction_energy(ev.pointwise.invariance_defect1, 0.001);
    const double conc2_1 = host_top_fraction_energy(ev.pointwise.invariance_defect2, 0.01);
    const double conc2_01 = host_top_fraction_energy(ev.pointwise.invariance_defect2, 0.001);
    const double concr_1 = host_top_fraction_energy(ev.pointwise.reconstruction_error, 0.01);
    const double concr_01 = host_top_fraction_energy(ev.pointwise.reconstruction_error, 0.001);
    std::cout << label << " concentration invariance_defect1 top1%=" << conc1_1 << " top0.1%=" << conc1_01 << '\n';
    std::cout << label << " concentration invariance_defect2 top1%=" << conc2_1 << " top0.1%=" << conc2_01 << '\n';
    std::cout << label << " concentration reconstruction_error top1%=" << concr_1 << " top0.1%=" << concr_01 << '\n';

    std::cout << label
              << " abs_c min=" << *std::min_element(ev.pointwise.abs_c.begin(), ev.pointwise.abs_c.end());
    const double cp[4] = {0.1, 1.0, 10.0, 50.0};
    for (double q : cp) std::cout << " p" << q << "=" << host_percentile(ev.pointwise.abs_c, q);
    std::cout << " epsilon_times_scale=" << epsilon_scale << '\n';
}

// ---------------------------------------------------------------------------
// Case 1: terminal_shifted_apply_unit (cheap). Manufactured trig-K fixture
// copied VERBATIM from newton_gpu_cases.cu's ManufacturedProblemBuffers
// (amplitude 0.25, uniform Darcy v=(1,0,0), benchmark(1) gauge).
// ---------------------------------------------------------------------------

struct ManufacturedProblemBuffers {
    DeviceBuffer<real> conductivity;
    DeviceBuffer<real> darcy_u;
    DeviceBuffer<real> darcy_v;
    DeviceBuffer<real> darcy_w;

    explicit ManufacturedProblemBuffers(const Grid3D& grid, double amplitude)
        : conductivity(grid.num_cells()), darcy_u(compact_mac_u_size(grid)),
          darcy_v(compact_mac_v_size(grid)), darcy_w(compact_mac_w_size(grid)) {
        std::vector<real> k_host(grid.num_cells());
        for (int iz = 0; iz < grid.nz; ++iz) {
            const double z = (iz + 0.5) * static_cast<double>(grid.dz);
            for (int iy = 0; iy < grid.ny; ++iy) {
                const double y = (iy + 0.5) * static_cast<double>(grid.dy);
                for (int ix = 0; ix < grid.nx; ++ix) {
                    const double x = (ix + 0.5) * static_cast<double>(grid.dx);
                    const double log_k = amplitude * std::sin(2.0 * kPi * x) *
                                          std::sin(2.0 * kPi * y) * std::sin(2.0 * kPi * z);
                    k_host[grid.idx(ix, iy, iz)] = static_cast<real>(std::exp(log_k));
                }
            }
        }
        const std::vector<real> ones_u(compact_mac_u_size(grid), real{1});
        const std::vector<real> zeros_v(compact_mac_v_size(grid), real{0});
        const std::vector<real> zeros_w(compact_mac_w_size(grid), real{0});
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(conductivity.data(), k_host.data(),
                                          k_host.size() * sizeof(real), cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(darcy_u.data(), ones_u.data(), ones_u.size() * sizeof(real),
                                          cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(darcy_v.data(), zeros_v.data(), zeros_v.size() * sizeof(real),
                                          cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(darcy_w.data(), zeros_w.data(), zeros_w.size() * sizeof(real),
                                          cudaMemcpyHostToDevice));
    }
};

[[nodiscard]] StreamfunctionProblemView manufactured_problem_view(
    const Grid3D& grid, const ManufacturedProblemBuffers& buffers) {
    StreamfunctionProblemView problem;
    problem.grid = grid;
    problem.conductivity = DeviceSpan<const real>(buffers.conductivity.span());
    problem.conductivity_representation = ConductivityRepresentation::conductivity_k;
    problem.darcy_velocity = CompactMacVelocityConstView{DeviceSpan<const real>(buffers.darcy_u.span()),
                                                         DeviceSpan<const real>(buffers.darcy_v.span()),
                                                         DeviceSpan<const real>(buffers.darcy_w.span())};
    problem.bc = triply_periodic();
    problem.gauge = AffineGauge::benchmark(real{1});
    return problem;
}

[[nodiscard]] CaseResult case_terminal_shifted_apply_unit() {
    std::cout << std::setprecision(17);
    constexpr int n = 16;
    constexpr double amplitude = 0.25;
    const Grid3D grid = isotropic_grid(n);
    const std::size_t cells = grid.num_cells();

    CudaContext ctx(0);
    ManufacturedProblemBuffers buffers(grid, amplitude);
    const StreamfunctionProblemView problem = manufactured_problem_view(grid, buffers);

    // Base state: a genuine converged adaptive-Picard solve at the solver's
    // (eta=1, epsilon=1e-2) defaults -- the SAME provenance G1's 16^3 branch
    // uses in newton_gpu_cases.cu -- so JvpWorkspace's "already mean-zero
    // projected" base-state caller contract is satisfied by construction.
    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;
    const StreamfunctionSolverConfig solver_config{};
    const StreamfunctionSolveReport solve_report =
        solve_streamfunctions(ctx, problem, solver_config, fields, workspace);
    ctx.synchronize();
    if (solve_report.status != StreamfunctionSolveStatus::converged) {
        throw std::runtime_error(
            "terminal_shifted_apply_unit fixture: base adaptive-Picard solve did not converge");
    }

    const NonlinearSourceConfig source_config{real{1e-2}, real{1}, 0, {}}; // v=(1,0,0) -> v_rms=1 exactly.
    const ResidualHistogramConfig histogram_config{};
    const AffineGauge gauge = AffineGauge::benchmark(real{1});

    JvpWorkspace jvp;
    jvp.prepare(cells);
    jvp.prepare_jvp_base(ctx, grid, DeviceSpan<const real>(workspace.q()),
                         CoupledVectorView{fields.u1_span(), fields.u2_span()}, gauge, real{1},
                         source_config, histogram_config);

    // Deterministic nonzero mean-zero direction: P(sin-based field pair).
    std::vector<real> dir1_host(cells), dir2_host(cells);
    for (int iz = 0; iz < grid.nz; ++iz) {
        const double z = (iz + 0.5) * static_cast<double>(grid.dz);
        for (int iy = 0; iy < grid.ny; ++iy) {
            const double y = (iy + 0.5) * static_cast<double>(grid.dy);
            for (int ix = 0; ix < grid.nx; ++ix) {
                const double x = (ix + 0.5) * static_cast<double>(grid.dx);
                const std::size_t idx = grid.idx(ix, iy, iz);
                dir1_host[idx] = static_cast<real>(std::sin(2.0 * kPi * x) * std::cos(2.0 * kPi * y) *
                                                    std::sin(2.0 * kPi * z));
                dir2_host[idx] = static_cast<real>(std::cos(2.0 * kPi * x) * std::sin(2.0 * kPi * y) *
                                                    std::cos(2.0 * kPi * z));
            }
        }
    }
    DeviceBuffer<real> dir1(cells), dir2(cells);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(dir1.data(), dir1_host.data(), cells * sizeof(real),
                                      cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(dir2.data(), dir2_host.data(), cells * sizeof(real),
                                      cudaMemcpyHostToDevice));
    constraints::MeanZeroWorkspace mean_zero_ws;
    mean_zero_ws.prepare(cells);
    constraints::MeanZeroProjector projector;
    projector.project(ctx, dir1.span(), mean_zero_ws);
    projector.project(ctx, dir2.span(), mean_zero_ws);
    ctx.synchronize();

    const ConstCoupledVectorView dir_view(DeviceSpan<const real>(dir1.span()),
                                          DeviceSpan<const real>(dir2.span()));

    bool pass = true;
    const auto check = [&](const char* name, bool ok) {
        pass = pass && ok;
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    };

    // (a) Exactness: shifted apply == fresh Jp + independently recomputed mu*Ap.
    const real mu_values[] = {real{1e-2}, real{1}};
    for (real mu : mu_values) {
        ShiftedJacobianOperator op(jvp, grid, DeviceSpan<const real>(workspace.q()), mu);
        op.prepare(cells);
        DeviceBuffer<real> shifted1(cells), shifted2(cells);
        const JvpApplyReport shifted_report =
            op.apply(ctx, grid, dir_view, JvpDeltaConfig{}, CoupledVectorView{shifted1.span(), shifted2.span()});

        DeviceBuffer<real> jv1(cells), jv2(cells);
        const JvpApplyReport jv_report =
            jvp.apply(ctx, grid, dir_view, JvpDeltaConfig{}, CoupledVectorView{jv1.span(), jv2.span()});

        const operators::LesterPositiveDiffusionOperator A(grid, DeviceSpan<const real>(workspace.q()));
        DeviceBuffer<real> ap1(cells), ap2(cells);
        A.apply(ctx, dir_view.c1, ap1.span());
        A.apply(ctx, dir_view.c2, ap2.span());
        blas::axpy(ctx, mu, DeviceSpan<const real>(ap1.span()), jv1.span());
        blas::axpy(ctx, mu, DeviceSpan<const real>(ap2.span()), jv2.span());
        ctx.synchronize();

        const std::vector<real> shifted1_h = download(DeviceSpan<const real>(shifted1.span()));
        const std::vector<real> shifted2_h = download(DeviceSpan<const real>(shifted2.span()));
        const std::vector<real> ref1_h = download(DeviceSpan<const real>(jv1.span()));
        const std::vector<real> ref2_h = download(DeviceSpan<const real>(jv2.span()));

        double max_diff = 0.0;
        double scale = 1.0;
        for (std::size_t i = 0; i < cells; ++i) {
            max_diff = std::max(max_diff, std::abs(static_cast<double>(shifted1_h[i]) - static_cast<double>(ref1_h[i])));
            max_diff = std::max(max_diff, std::abs(static_cast<double>(shifted2_h[i]) - static_cast<double>(ref2_h[i])));
            scale = std::max(scale, std::abs(static_cast<double>(ref1_h[i])));
            scale = std::max(scale, std::abs(static_cast<double>(ref2_h[i])));
        }
        const bool ok = shifted_report.status == JvpApplyStatus::ok &&
                        jv_report.status == JvpApplyStatus::ok && max_diff <= 1e-13 * scale;
        std::cout << "  exactness mu=" << static_cast<double>(mu) << " max_diff=" << max_diff
                  << " scale=" << scale << " threshold=" << (1e-13 * scale) << '\n';
        std::ostringstream name;
        name << "exactness_mu_" << static_cast<double>(mu);
        check(name.str().c_str(), ok);
    }

    // (b) mu=0 passthrough: bitwise-equal to a fresh plain jvp.apply.
    {
        ShiftedJacobianOperator op0(jvp, grid, DeviceSpan<const real>(workspace.q()), real{0});
        op0.prepare(cells);
        DeviceBuffer<real> shifted1(cells), shifted2(cells);
        const JvpApplyReport op0_report =
            op0.apply(ctx, grid, dir_view, JvpDeltaConfig{}, CoupledVectorView{shifted1.span(), shifted2.span()});

        DeviceBuffer<real> jv1(cells), jv2(cells);
        const JvpApplyReport jv0_report =
            jvp.apply(ctx, grid, dir_view, JvpDeltaConfig{}, CoupledVectorView{jv1.span(), jv2.span()});
        ctx.synchronize();
        (void)op0_report;
        (void)jv0_report;

        const bool bitwise_ok =
            bitwise_equal(download(DeviceSpan<const real>(shifted1.span())), download(DeviceSpan<const real>(jv1.span()))) &&
            bitwise_equal(download(DeviceSpan<const real>(shifted2.span())), download(DeviceSpan<const real>(jv2.span())));
        check("mu_zero_bitwise_passthrough", bitwise_ok);
    }

    // (c) set_mu fail-fasts; unprepared apply throws std::logic_error.
    {
        ShiftedJacobianOperator op(jvp, grid, DeviceSpan<const real>(workspace.q()), real{1e-2});
        bool threw_negative = false;
        try {
            op.set_mu(real{-1});
        } catch (const std::invalid_argument&) {
            threw_negative = true;
        }
        check("set_mu_negative_throws", threw_negative);

        bool threw_nan = false;
        try {
            op.set_mu(std::numeric_limits<real>::quiet_NaN());
        } catch (const std::invalid_argument&) {
            threw_nan = true;
        }
        check("set_mu_nan_throws", threw_nan);

        ShiftedJacobianOperator unprepared_op(jvp, grid, DeviceSpan<const real>(workspace.q()), real{1e-2});
        bool threw_unprepared = false;
        try {
            DeviceBuffer<real> out1(cells), out2(cells);
            const JvpApplyReport unprepared_report =
                unprepared_op.apply(ctx, grid, dir_view, JvpDeltaConfig{}, CoupledVectorView{out1.span(), out2.span()});
            (void)unprepared_report;
        } catch (const std::logic_error&) {
            threw_unprepared = true;
        }
        check("unprepared_apply_throws_logic_error", threw_unprepared);
    }

    std::cout << "case=terminal_shifted_apply_unit verdict=" << (pass ? "PASS" : "FAIL") << '\n';

    return {pass,
            "terminal_shifted_apply_unit",
            "gpu-terminal-shifted-apply-unit",
            "16^3 (a=0.25) manufactured trig-K fixture (copied verbatim from newton_gpu_cases.cu)",
            1.0,
            0.0,
            "SF-25 T02 enabler unit",
            pass ? "all pass" : "some failed",
            "ShiftedJacobianOperator::apply exactness (<=1e-13*scale) against an independently "
            "recomputed Jp+mu*Ap, bitwise mu=0 passthrough, and the set_mu/unprepared-apply "
            "fail-fast contract"};
}

// ---------------------------------------------------------------------------
// Case 2: terminal_dgate_diagnostic (HEAVY, separate registry). Implements
// D-gate protocol E2-E5 exactly, per the SF-25 activation bitácora and
// docs/decisions/2026-08-14-manifold-robust-terminal-solver.md.
// ---------------------------------------------------------------------------

// K_att = exp(Y_att), elementwise -- the SAME small kernel
// ContinuationController.cu's heterogeneity_exp_kernel implements (renamed
// here to avoid an ODR clash across translation units).
__global__ void terminal_dgate_exp_kernel(const real* __restrict__ y_att, real* __restrict__ k_att,
                                          std::size_t n) {
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    for (std::size_t index = start; index < n; index += stride) {
        k_att[index] = exp(y_att[index]);
    }
}

void terminal_dgate_enqueue_exp(CudaContext& ctx, DeviceSpan<const real> y_att, DeviceSpan<real> k_att) {
    const std::size_t n = k_att.size();
    if (n == 0) return;
    constexpr int kBlock = 256;
    constexpr int kMaxBlocks = 65535;
    const std::size_t requested_blocks = (n + kBlock - 1) / kBlock;
    const int blocks =
        static_cast<int>(requested_blocks < static_cast<std::size_t>(kMaxBlocks) ? requested_blocks : kMaxBlocks);
    terminal_dgate_exp_kernel<<<blocks, kBlock, 0, ctx.cuda_stream()>>>(y_att.data(), k_att.data(), n);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

[[nodiscard]] CaseResult case_terminal_dgate_diagnostic() {
    std::cout << std::setprecision(17);
    bool pass = true;
    const auto check = [&](const char* name, bool ok) {
        pass = pass && ok;
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    };

    constexpr int n32 = 32;
    const Grid3D grid(n32, n32, n32, real{1}, real{1}, real{1});
    const std::size_t n = grid.num_cells();
    CudaContext ctx(0);

    // =========================================================================
    // E2 freeze, step 1: the sigma_Y^2=1, 32^3 smoke, VERBATIM from
    // heterogeneity_continuation_gpu_cases.cu's run_heterogeneity_smoke
    // (seed 12345, ell=8, normalize_variance; anderson R5 defaults enabled;
    // degenerate epsilon leg; lambda axis defaults; flow_config default
    // qbar=(1,0,0)) EXCEPT newton stays at its default (disabled): E2
    // freezes the SF-21-characterized plateau regime, not the SF-24 Newton
    // budget-exhaustion regime the accepted smoke fixture now also exercises.
    // =========================================================================
    physics::PeriodicGaussianFieldConfig field_config;
    field_config.sigma2 = real{1};
    field_config.corr_length = real{8};
    field_config.seed = 12345ULL;
    field_config.normalize_variance = true;

    DeviceBuffer<real> y(n);
    physics::PeriodicGaussianFieldWorkspace field_workspace;
    const physics::PeriodicGaussianFieldReport field_report =
        physics::generate_periodic_gaussian_field(ctx, grid, field_config, y.span(), field_workspace);
    ctx.synchronize();

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;
    StreamfunctionSolverConfig base_config; // full defaults (adaptive Picard, newton DISABLED).
    base_config.anderson.enabled = true;
    base_config.anderson.depth = 5;
    base_config.anderson.start_iteration = 5;
    base_config.anderson.condition_limit = real{1e12};

    HeterogeneityContinuationConfig continuation_config{}; // lambda axis defaults.
    continuation_config.inner.epsilon_log10.target = continuation_config.inner.epsilon_log10.start;
    const physics::AffinePeriodicFlowConfig flow_config{}; // qbar=(1,0,0) default.

    const HeterogeneityContinuationReport freeze_report = run_streamfunction_heterogeneity_continuation(
        ctx, grid, DeviceSpan<const real>(y.span()), continuation_config, flow_config, base_config, fields,
        workspace);
    ctx.synchronize();

    std::cout << "E2 freeze: field_raw_mean=" << field_report.raw_mean
              << " field_final_variance=" << field_report.final_variance
              << " status=" << heterogeneity_status_label(freeze_report.status)
              << " final_lambda=" << freeze_report.final_lambda
              << " final_eta=" << freeze_report.final_eta
              << " stage_history_size=" << freeze_report.stage_history.size() << '\n';

    const bool e2_freeze_ok = freeze_report.status == HeterogeneityStatus::lambda_floor_exhausted &&
                              freeze_report.final_lambda == real{0.5} && freeze_report.final_eta == real{1};
    check("E2_freeze_status_lambda_floor_exhausted",
          freeze_report.status == HeterogeneityStatus::lambda_floor_exhausted);
    check("E2_freeze_final_lambda_eq_0_5", freeze_report.final_lambda == real{0.5});
    check("E2_freeze_final_eta_eq_1", freeze_report.final_eta == real{1});

    // =========================================================================
    // E2 freeze, steps 2-3: mirror ContinuationController.cu's per-lambda
    // setup EXACTLY for lambda_attempt=0.5125:
    //   - build_attempt (ContinuationController.cu lines 542-553): Y_att =
    //     lambda_attempt*Y (D2D copy + blas::scal), K_att = exp(Y_att) (the
    //     elementwise kernel above), then the SF-19 affine-periodic flow
    //     solve on K_att with THE SAME flow_config the freeze run used
    //     (default AffinePeriodicFlowConfig{}, qbar=(1,0,0));
    //   - make_problem_view (ContinuationController.cu lines 555-567):
    //     conductivity = Y_att as log_conductivity_y, darcy_velocity = that
    //     flow, triply periodic, benchmark(1) gauge.
    // Then ONE warm-started stage solve at (eta=1, epsilon=1e-2) with the
    // SAME base_config (anderson on, newton off), initial_state=warm_start,
    // coefficient_state=rebuild (first solve at this lambda on this
    // workspace) -- mirroring ContinuationController.cu lines 629-637.
    // =========================================================================
    constexpr real kLambdaAttempt = real{0.5125};

    DeviceBuffer<real> y_att(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(y_att.data(), y.data(), n * sizeof(real),
                                           cudaMemcpyDeviceToDevice, ctx.cuda_stream()));
    blas::scal(ctx, y_att.span(), kLambdaAttempt);

    DeviceBuffer<real> k_att(n);
    terminal_dgate_enqueue_exp(ctx, DeviceSpan<const real>(y_att.span()), k_att.span());

    const std::size_t u_size = compact_mac_u_size(grid);
    const std::size_t v_size = compact_mac_v_size(grid);
    const std::size_t w_size = compact_mac_w_size(grid);
    DeviceBuffer<real> flow_u(u_size), flow_v(v_size), flow_w(w_size);
    physics::AffinePeriodicFlowWorkspace flow_workspace;
    physics::AffinePeriodicVelocityView velocity{flow_u.span(), flow_v.span(), flow_w.span()};
    const physics::AffinePeriodicFlowReport attempt_flow = physics::solve_affine_periodic_flow(
        ctx, grid, DeviceSpan<const real>(k_att.span()), flow_config, velocity, flow_workspace);
    ctx.synchronize();

    StreamfunctionProblemView problem_view;
    problem_view.grid = grid;
    problem_view.conductivity = DeviceSpan<const real>(y_att.span());
    problem_view.conductivity_representation = ConductivityRepresentation::log_conductivity_y;
    problem_view.darcy_velocity = CompactMacVelocityConstView{DeviceSpan<const real>(flow_u.span()),
                                                               DeviceSpan<const real>(flow_v.span()),
                                                               DeviceSpan<const real>(flow_w.span())};
    problem_view.bc = triply_periodic();
    problem_view.gauge = AffineGauge::benchmark(real{1});

    StreamfunctionSolverConfig stage_config = base_config;
    stage_config.eta = real{1};
    stage_config.epsilon = real{1e-2};
    stage_config.initial_state = PicardInitialState::warm_start;
    stage_config.coefficient_state = CoefficientState::rebuild;

    const StreamfunctionSolveReport stage_report =
        solve_streamfunctions(ctx, problem_view, stage_config, fields, workspace);
    ctx.synchronize();

    const double r_F_frozen = static_cast<double>(stage_report.residual.r_F);
    std::cout << "E2 stage: attempt_flow_achieved_flux=(" << attempt_flow.achieved_mean_flux[0] << ","
              << attempt_flow.achieved_mean_flux[1] << "," << attempt_flow.achieved_mean_flux[2]
              << ") status=" << solve_status_label(stage_report.status)
              << " exit_reason=" << exit_reason_label(stage_report.exit_reason)
              << " picard_iterations=" << stage_report.picard_iterations << " r_F_frozen=" << r_F_frozen
              << '\n';

    check("E2_stage_not_converged", stage_report.status == StreamfunctionSolveStatus::not_converged);
    // AMENDMENT E6a (SF-25 bitácora 2026-08-14T12:10Z): both `stagnated` and
    // `omega_floor_rejected` are documented SF-21 plateau signatures (ramp
    // continuation attempts stagnate; a direct-lambda attempt can instead
    // hit the omega floor first, rejecting every Picard trial) -- accept
    // either as the E2 freeze's plateau signature.
    check("E2_stage_exit_reason_plateau_signature",
          stage_report.exit_reason == PicardExitReason::stagnated ||
              stage_report.exit_reason == PicardExitReason::omega_floor_rejected);
    const bool r_f_band_ok = r_F_frozen >= 1e-4 && r_F_frozen <= 1e-2;
    check("E2_stage_r_F_in_prespecified_band_1e-4_1e-2", r_f_band_ok);

    // =========================================================================
    // Shared setup for E3-E5.
    // =========================================================================
    const DeviceSpan<const real> q_att = workspace.q(); // = exp(-Y_att), rebuilt by the E2 stage above.
    const double v_rms = static_cast<double>(stage_report.diagnostics.v_d_rms);
    NonlinearSourceConfig source_config;
    source_config.epsilon = real{1e-2};
    source_config.v_rms = static_cast<real>(v_rms);
    const ResidualHistogramConfig histogram_config{};
    const AffineGauge gauge = AffineGauge::benchmark(real{1});

    JvpWorkspace jvp;
    jvp.prepare(n);
    jvp.prepare_jvp_base(ctx, grid, q_att, CoupledVectorView{fields.u1_span(), fields.u2_span()}, gauge,
                         real{1}, source_config, histogram_config);

    StreamfunctionResidualWorkspace frozen_residual_workspace;
    frozen_residual_workspace.prepare(n);
    DeviceBuffer<real> f1(n), f2(n);
    enqueue_streamfunction_residual(ctx, grid, q_att,
                                    PeriodicStreamfunctionFluctuations{fields.u1_span(), fields.u2_span()},
                                    gauge, real{1}, source_config, histogram_config, f1.span(), f2.span(),
                                    frozen_residual_workspace);
    const StreamfunctionResidualReport frozen_residual = synchronize_streamfunction_residual_report(
        ctx, grid, real{1}, source_config, histogram_config, frozen_residual_workspace);
    std::cout << "E3-E5 setup: v_rms=" << v_rms << " frozen_residual_recompute_r_F=" << frozen_residual.r_F
              << " (stage r_F=" << r_F_frozen << ")\n";

    DeviceBuffer<real> b1(n), b2(n);
    blas::copy(ctx, DeviceSpan<const real>(f1.span()), b1.span());
    blas::copy(ctx, DeviceSpan<const real>(f2.span()), b2.span());
    blas::scal(ctx, b1.span(), real{-1});
    blas::scal(ctx, b2.span(), real{-1});
    ctx.synchronize();

    multigrid::MGConfig mg_config = base_config.mg;
    BlockDiagonalMGPreconditioner precond(workspace.hierarchy(), mg_config);

    // =========================================================================
    // E3: mu-sweep, the MECHANISM gate.
    // =========================================================================
    struct MuResult {
        double mu;
        CoupledGmresStatus status;
        int total_inner_iterations;
        int outer_cycles;
        double first_true_residual;
        double final_true_residual;
    };
    std::vector<MuResult> mu_results;
    const double mu_list[] = {1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 0.0};

    const real e3_rel_tol =
        std::clamp(static_cast<real>(std::sqrt(r_F_frozen)), real{1e-8}, real{1e-1});

    for (double mu_d : mu_list) {
        const real mu = static_cast<real>(mu_d);
        ShiftedJacobianOperator op(jvp, grid, q_att, mu);
        op.prepare(n);
        CoupledGmres gmres;
        gmres.prepare(n, 10);

        DeviceBuffer<real> corr1(n), corr2(n);
        blas::fill(ctx, corr1.span(), real{0});
        blas::fill(ctx, corr2.span(), real{0});

        CoupledGmresConfig cfg;
        cfg.restart = 10;
        cfg.max_iterations = 100;
        cfg.rel_tol = e3_rel_tol;

        const CoupledGmresReport report = gmres.solve(
            ctx, grid, op, precond,
            ConstCoupledVectorView(DeviceSpan<const real>(b1.span()), DeviceSpan<const real>(b2.span())), cfg,
            JvpDeltaConfig{}, CoupledVectorView{corr1.span(), corr2.span()});
        ctx.synchronize();

        const double first_true =
            report.checkpoints.empty() ? 0.0 : static_cast<double>(report.checkpoints.front().true_residual);
        const double final_true =
            report.checkpoints.empty() ? 0.0 : static_cast<double>(report.checkpoints.back().true_residual);
        mu_results.push_back(
            {mu_d, report.status, report.total_inner_iterations, report.outer_cycles, first_true, final_true});

        std::cout << "E3 mu=" << mu_d << " status=" << gmres_status_label(report.status)
                  << " total_inner_iterations=" << report.total_inner_iterations
                  << " outer_cycles=" << report.outer_cycles << " first_true_residual=" << first_true
                  << " final_true_residual=" << final_true
                  << " reduction_ratio=" << (first_true > 0.0 ? final_true / first_true : 0.0) << '\n';
    }

    int it0 = 0;
    for (const auto& r : mu_results) {
        if (r.mu == 0.0) it0 = r.total_inner_iterations;
    }

    // E3 gate: mechanism_confirmed requires a mu>0 that CONVERGED with a
    // >=10x reduction in inner iterations vs the mu=0 baseline (it0).
    // E5 decision: mu_star is a SEPARATE selection rule, the smallest
    // converged-within-budget mu (mu>0, status==converged), independent of
    // the 10x iteration-reduction filter used for mechanism_confirmed.
    bool mechanism_confirmed = false;
    for (const auto& r : mu_results) {
        if (r.mu > 0.0 && r.status == CoupledGmresStatus::converged && r.total_inner_iterations * 10 <= it0) {
            mechanism_confirmed = true;
        }
    }
    double mu_star = -1.0;
    int mu_star_iterations = 0;
    for (const auto& r : mu_results) {
        if (r.mu > 0.0 && r.status == CoupledGmresStatus::converged) {
            if (mu_star < 0.0 || r.mu < mu_star) {
                mu_star = r.mu;
                mu_star_iterations = r.total_inner_iterations;
            }
        }
    }
    std::cout << "E3 gate (>=10x reduction exists): it0(mu=0)=" << it0
              << " mechanism_confirmed=" << (mechanism_confirmed ? "true" : "false")
              << " | E5 decision (smallest converged-within-budget mu): mu_star=" << mu_star
              << " mu_star_iterations=" << mu_star_iterations << '\n';
    check("E3_mechanism_confirmed_ge_10x_reduction", mechanism_confirmed);

    // =========================================================================
    // E4: generalized inverse iteration toward the smallest (J, A_blk)
    // eigenpair. Recorded evidence, non-blocking for the D-gate verdict.
    // =========================================================================
    blas::ReductionWorkspace dot_ws;
    constraints::MeanZeroWorkspace mean_zero_ws;
    mean_zero_ws.prepare(n);
    constraints::MeanZeroProjector projector;

    const auto pair_dot = [&](DeviceSpan<const real> a1, DeviceSpan<const real> a2, DeviceSpan<const real> c1,
                              DeviceSpan<const real> c2) {
        return blas::dot_host(ctx, a1, c1, dot_ws) + blas::dot_host(ctx, a2, c2, dot_ws);
    };
    const auto pair_norm = [&](DeviceSpan<const real> a1, DeviceSpan<const real> a2) {
        return std::sqrt(std::max(0.0, static_cast<double>(pair_dot(a1, a2, a1, a2))));
    };

    DeviceBuffer<real> v1(n), v2(n);
    {
        const double b_norm = pair_norm(DeviceSpan<const real>(b1.span()), DeviceSpan<const real>(b2.span()));
        blas::copy(ctx, DeviceSpan<const real>(b1.span()), v1.span());
        blas::copy(ctx, DeviceSpan<const real>(b2.span()), v2.span());
        if (b_norm > 0.0 && std::isfinite(b_norm)) {
            blas::scal(ctx, v1.span(), real{1} / static_cast<real>(b_norm));
            blas::scal(ctx, v2.span(), real{1} / static_cast<real>(b_norm));
        }
        projector.project(ctx, v1.span(), mean_zero_ws);
        projector.project(ctx, v2.span(), mean_zero_ws);
        ctx.synchronize();
    }

    const operators::LesterPositiveDiffusionOperator A_op(grid, q_att);

    ShiftedJacobianOperator e4_op(jvp, grid, q_att, real{1e-3});
    e4_op.prepare(n);
    CoupledGmres e4_gmres;
    e4_gmres.prepare(n, 10);
    CoupledGmresConfig e4_config;
    e4_config.restart = 10;
    e4_config.max_iterations = 100;
    e4_config.rel_tol = real{1e-6}; // measurement infrastructure only, per the E4 decision.

    std::vector<double> rayleigh_trajectory;
    for (int k = 1; k <= 20; ++k) {
        DeviceBuffer<real> w1(n), w2(n);
        A_op.apply(ctx, DeviceSpan<const real>(v1.span()), w1.span());
        A_op.apply(ctx, DeviceSpan<const real>(v2.span()), w2.span());
        ctx.synchronize();

        DeviceBuffer<real> z1(n), z2(n);
        blas::fill(ctx, z1.span(), real{0});
        blas::fill(ctx, z2.span(), real{0});
        const CoupledGmresReport e4_report = e4_gmres.solve(
            ctx, grid, e4_op, precond,
            ConstCoupledVectorView(DeviceSpan<const real>(w1.span()), DeviceSpan<const real>(w2.span())),
            e4_config, JvpDeltaConfig{}, CoupledVectorView{z1.span(), z2.span()});
        ctx.synchronize();

        if (e4_report.status == CoupledGmresStatus::nonfinite) {
            std::cout << "E4 k=" << k << " inverse-iteration solve nonfinite; stopping with "
                      << rayleigh_trajectory.size() << " recorded values\n";
            break;
        }

        const double z_norm = pair_norm(DeviceSpan<const real>(z1.span()), DeviceSpan<const real>(z2.span()));
        if (!(z_norm > 0.0) || !std::isfinite(z_norm)) {
            std::cout << "E4 k=" << k << " non-finite/zero z norm; stopping\n";
            break;
        }
        blas::scal(ctx, z1.span(), real{1} / static_cast<real>(z_norm));
        blas::scal(ctx, z2.span(), real{1} / static_cast<real>(z_norm));
        projector.project(ctx, z1.span(), mean_zero_ws);
        projector.project(ctx, z2.span(), mean_zero_ws);
        blas::copy(ctx, DeviceSpan<const real>(z1.span()), v1.span());
        blas::copy(ctx, DeviceSpan<const real>(z2.span()), v2.span());
        ctx.synchronize();

        DeviceBuffer<real> jv1(n), jv2(n);
        const JvpApplyReport jv_report = jvp.apply(
            ctx, grid, ConstCoupledVectorView(DeviceSpan<const real>(v1.span()), DeviceSpan<const real>(v2.span())),
            JvpDeltaConfig{}, CoupledVectorView{jv1.span(), jv2.span()});
        if (jv_report.status != JvpApplyStatus::ok) {
            std::cout << "E4 k=" << k << " Jv nonfinite; stopping\n";
            break;
        }
        DeviceBuffer<real> av1(n), av2(n);
        A_op.apply(ctx, DeviceSpan<const real>(v1.span()), av1.span());
        A_op.apply(ctx, DeviceSpan<const real>(v2.span()), av2.span());
        ctx.synchronize();

        const double numerator = pair_dot(DeviceSpan<const real>(v1.span()), DeviceSpan<const real>(v2.span()),
                                          DeviceSpan<const real>(jv1.span()), DeviceSpan<const real>(jv2.span()));
        const double denominator = pair_dot(DeviceSpan<const real>(v1.span()), DeviceSpan<const real>(v2.span()),
                                            DeviceSpan<const real>(av1.span()), DeviceSpan<const real>(av2.span()));
        const double rayleigh = denominator != 0.0 ? numerator / denominator : std::numeric_limits<double>::quiet_NaN();
        rayleigh_trajectory.push_back(rayleigh);
        std::cout << "E4 k=" << k << " rayleigh=" << rayleigh << '\n';
    }
    const double rayleigh_final =
        rayleigh_trajectory.empty() ? std::numeric_limits<double>::quiet_NaN() : rayleigh_trajectory.back();
    const bool e4_claim = std::isfinite(rayleigh_final) && std::abs(rayleigh_final) <= 1e-2;
    std::cout << "E4 claim check (non-blocking): |rayleigh_last|<=1e-2 -> " << (e4_claim ? "PASS" : "FAIL")
              << " (rayleigh_last=" << rayleigh_final << ", trajectory_length=" << rayleigh_trajectory.size()
              << ")\n";

    // =========================================================================
    // E5: LM mini-solve. theta = mu_star / r_F_frozen. SKIP if E3 found no
    // converged mu>0 -- the honest falsification path.
    // =========================================================================
    bool e5_pass = false;
    if (!mechanism_confirmed) {
        std::cout << "E5_SKIPPED_MECHANISM_FAILED\n";
        std::cout << "E5 result: skipped (mechanism not confirmed)\n";
    } else {
        const double theta = mu_star / r_F_frozen;
        std::cout << "E5 theta=" << theta << " (mu_star=" << mu_star << ", r_F_frozen=" << r_F_frozen << ")\n";

        DeviceBuffer<real> state1(n), state2(n);
        blas::copy(ctx, DeviceSpan<const real>(fields.u1_span()), state1.span());
        blas::copy(ctx, DeviceSpan<const real>(fields.u2_span()), state2.span());
        ctx.synchronize();

        ShiftedJacobianOperator e5_op(jvp, grid, q_att, real{0});
        e5_op.prepare(n);
        CoupledGmres e5_gmres;
        e5_gmres.prepare(n, 10);

        StreamfunctionResidualWorkspace e5_residual_ws;
        e5_residual_ws.prepare(n);
        DeviceBuffer<real> e5_f1(n), e5_f2(n);
        DeviceBuffer<real> trial1(n), trial2(n);
        DeviceBuffer<real> trial_f1(n), trial_f2(n);
        DeviceBuffer<real> delta1(n), delta2(n);
        DeviceBuffer<real> rhs1(n), rhs2(n);

        constexpr real kAlphaMin = real{0.03125}; // 2^-5, matches NewtonKrylovConfig::alpha_min default.
        constexpr real kArmijoC = real{1e-4};      // matches NewtonKrylovConfig::armijo_c default.

        bool reached = false;
        bool step_failure = false;
        for (int k = 0; k < 30 && !reached && !step_failure; ++k) {
            enqueue_streamfunction_residual(ctx, grid, q_att,
                                            PeriodicStreamfunctionFluctuations{state1.span(), state2.span()},
                                            gauge, real{1}, source_config, histogram_config, e5_f1.span(),
                                            e5_f2.span(), e5_residual_ws);
            const StreamfunctionResidualReport res_k = synchronize_streamfunction_residual_report(
                ctx, grid, real{1}, source_config, histogram_config, e5_residual_ws);
            const double r_F_k = static_cast<double>(res_k.r_F);

            if (r_F_k <= 1e-4) {
                reached = true;
                std::cout << "E5 k=" << k << " r_F=" << r_F_k << " REACHED\n";
                break;
            }

            jvp.prepare_jvp_base(ctx, grid, q_att, CoupledVectorView{state1.span(), state2.span()}, gauge,
                                 real{1}, source_config, histogram_config);

            blas::copy(ctx, DeviceSpan<const real>(e5_f1.span()), rhs1.span());
            blas::copy(ctx, DeviceSpan<const real>(e5_f2.span()), rhs2.span());
            blas::scal(ctx, rhs1.span(), real{-1});
            blas::scal(ctx, rhs2.span(), real{-1});

            const real mu_k = static_cast<real>(theta * r_F_k);
            e5_op.set_mu(mu_k);

            CoupledGmresConfig e5_config;
            e5_config.restart = 10;
            e5_config.max_iterations = 100;
            e5_config.rel_tol = std::clamp(static_cast<real>(std::sqrt(r_F_k)), real{1e-8}, real{1e-1});

            blas::fill(ctx, delta1.span(), real{0});
            blas::fill(ctx, delta2.span(), real{0});
            const CoupledGmresReport gmres_report = e5_gmres.solve(
                ctx, grid, e5_op, precond,
                ConstCoupledVectorView(DeviceSpan<const real>(rhs1.span()), DeviceSpan<const real>(rhs2.span())),
                e5_config, JvpDeltaConfig{}, CoupledVectorView{delta1.span(), delta2.span()});
            ctx.synchronize();

            bool accepted = false;
            real accepted_alpha = real{0};
            real alpha = real{1};
            const double phi_k = 0.5 * r_F_k * r_F_k;
            while (alpha >= kAlphaMin) {
                blas::copy(ctx, DeviceSpan<const real>(state1.span()), trial1.span());
                blas::copy(ctx, DeviceSpan<const real>(state2.span()), trial2.span());
                blas::axpy(ctx, alpha, DeviceSpan<const real>(delta1.span()), trial1.span());
                blas::axpy(ctx, alpha, DeviceSpan<const real>(delta2.span()), trial2.span());
                projector.project(ctx, trial1.span(), mean_zero_ws);
                projector.project(ctx, trial2.span(), mean_zero_ws);

                enqueue_streamfunction_residual(ctx, grid, q_att,
                                                PeriodicStreamfunctionFluctuations{trial1.span(), trial2.span()},
                                                gauge, real{1}, source_config, histogram_config, trial_f1.span(),
                                                trial_f2.span(), e5_residual_ws);
                const StreamfunctionResidualReport trial_res = synchronize_streamfunction_residual_report(
                    ctx, grid, real{1}, source_config, histogram_config, e5_residual_ws);
                const double r_F_trial = static_cast<double>(trial_res.r_F);
                const bool finite_ok = std::isfinite(r_F_trial);
                const double phi_trial = 0.5 * r_F_trial * r_F_trial;
                const bool armijo_ok = finite_ok && phi_trial <= (1.0 - static_cast<double>(kArmijoC) *
                                                                          static_cast<double>(alpha)) *
                                                                        phi_k;

                std::cout << "  E5 k=" << k << " alpha=" << alpha << " r_F_trial=" << r_F_trial
                          << " finite=" << (finite_ok ? "true" : "false")
                          << " armijo=" << (armijo_ok ? "accept" : "reject") << '\n';

                if (armijo_ok) {
                    accepted = true;
                    accepted_alpha = alpha;
                    blas::copy(ctx, DeviceSpan<const real>(trial1.span()), state1.span());
                    blas::copy(ctx, DeviceSpan<const real>(trial2.span()), state2.span());
                    ctx.synchronize();
                    break;
                }
                alpha *= real{0.5};
            }

            std::cout << "E5 k=" << k << " r_F=" << r_F_k << " mu_k=" << mu_k
                      << " gmres_status=" << gmres_status_label(gmres_report.status)
                      << " gmres_inner=" << gmres_report.total_inner_iterations
                      << " accepted_alpha=" << (accepted ? static_cast<double>(accepted_alpha) : 0.0) << '\n';

            if (!accepted) {
                std::cout << "E5_STEP_FAILURE k=" << k << '\n';
                step_failure = true;
            }
        }

        e5_pass = reached;
        std::cout << "E5 result: " << (e5_pass ? "PASS" : "FAIL") << '\n';
    }

    // =========================================================================
    // E6 (DECISION E6, bitácora 2026-08-14T12:10Z): Psi-tc / backward-Euler
    // probe with bounded-non-monotone safeguards -- the PRESPECIFIED
    // contingency that runs precisely when the mechanism is confirmed (E3)
    // but the monotone-Armijo LM mini-solve (E5) failed at a spurious local
    // minimum of the merit. Same frozen state, same shifted operator/GMRES
    // configuration pattern as E5, but full (alpha=1) projected steps are
    // ACCEPTED WITHOUT Armijo under a strict bounded-non-monotone safeguard
    // (reject only if the candidate is non-finite or more than doubles
    // r_F), with dtau evolved by switched-evolution-relaxation (SER) after
    // each accepted step.
    // =========================================================================
    bool e6_pass = false;
    if (mechanism_confirmed && !e5_pass) {
        DeviceBuffer<real> e6_state1(n), e6_state2(n);
        blas::copy(ctx, DeviceSpan<const real>(fields.u1_span()), e6_state1.span());
        blas::copy(ctx, DeviceSpan<const real>(fields.u2_span()), e6_state2.span());
        ctx.synchronize();

        ShiftedJacobianOperator e6_op(jvp, grid, q_att, real{0});
        e6_op.prepare(n);
        CoupledGmres e6_gmres;
        e6_gmres.prepare(n, 10);

        StreamfunctionResidualWorkspace e6_residual_ws;
        e6_residual_ws.prepare(n);
        DeviceBuffer<real> e6_f1(n), e6_f2(n);
        DeviceBuffer<real> e6_candidate1(n), e6_candidate2(n);
        DeviceBuffer<real> e6_candidate_f1(n), e6_candidate_f2(n);
        DeviceBuffer<real> e6_delta1(n), e6_delta2(n);
        DeviceBuffer<real> e6_rhs1(n), e6_rhs2(n);

        double dtau = 1.0 / mu_star; // E3-calibrated: dtau_0 = 1/mu_star.
        double r_F_prev = r_F_frozen;
        std::cout << "E6 dtau_0=" << dtau << " (=1/mu_star, mu_star=" << mu_star << ")\n";

        bool reached = false;
        bool step_failure = false;
        for (int k = 0; k < 60 && !reached && !step_failure; ++k) {
            enqueue_streamfunction_residual(ctx, grid, q_att,
                                            PeriodicStreamfunctionFluctuations{e6_state1.span(), e6_state2.span()},
                                            gauge, real{1}, source_config, histogram_config, e6_f1.span(),
                                            e6_f2.span(), e6_residual_ws);
            const StreamfunctionResidualReport res_k = synchronize_streamfunction_residual_report(
                ctx, grid, real{1}, source_config, histogram_config, e6_residual_ws);
            const double r_F_k = static_cast<double>(res_k.r_F);

            if (r_F_k <= 1e-4) {
                reached = true;
                std::cout << "E6 k=" << k << " r_F=" << r_F_k << " REACHED\n";
                break;
            }
            if (r_F_k > 10.0 * r_F_frozen) {
                std::cout << "E6_DIVERGENCE_STOP k=" << k << " r_F=" << r_F_k
                          << " r_F_frozen=" << r_F_frozen << '\n';
                step_failure = true;
                break;
            }

            jvp.prepare_jvp_base(ctx, grid, q_att, CoupledVectorView{e6_state1.span(), e6_state2.span()}, gauge,
                                 real{1}, source_config, histogram_config);

            blas::copy(ctx, DeviceSpan<const real>(e6_f1.span()), e6_rhs1.span());
            blas::copy(ctx, DeviceSpan<const real>(e6_f2.span()), e6_rhs2.span());
            blas::scal(ctx, e6_rhs1.span(), real{-1});
            blas::scal(ctx, e6_rhs2.span(), real{-1});

            bool accepted = false;
            double accepted_r_F = 0.0;
            for (int attempt = 0; attempt < 3 && !accepted; ++attempt) {
                const real mu_k = static_cast<real>(1.0 / dtau);
                e6_op.set_mu(mu_k);

                CoupledGmresConfig e6_config;
                e6_config.restart = 10;
                e6_config.max_iterations = 100;
                e6_config.rel_tol = std::clamp(static_cast<real>(std::sqrt(r_F_k)), real{1e-8}, real{1e-1});

                blas::fill(ctx, e6_delta1.span(), real{0});
                blas::fill(ctx, e6_delta2.span(), real{0});
                const CoupledGmresReport gmres_report = e6_gmres.solve(
                    ctx, grid, e6_op, precond,
                    ConstCoupledVectorView(DeviceSpan<const real>(e6_rhs1.span()),
                                          DeviceSpan<const real>(e6_rhs2.span())),
                    e6_config, JvpDeltaConfig{}, CoupledVectorView{e6_delta1.span(), e6_delta2.span()});
                ctx.synchronize();

                // Full step (alpha=1), projected per component.
                blas::copy(ctx, DeviceSpan<const real>(e6_state1.span()), e6_candidate1.span());
                blas::copy(ctx, DeviceSpan<const real>(e6_state2.span()), e6_candidate2.span());
                blas::axpy(ctx, real{1}, DeviceSpan<const real>(e6_delta1.span()), e6_candidate1.span());
                blas::axpy(ctx, real{1}, DeviceSpan<const real>(e6_delta2.span()), e6_candidate2.span());
                projector.project(ctx, e6_candidate1.span(), mean_zero_ws);
                projector.project(ctx, e6_candidate2.span(), mean_zero_ws);

                enqueue_streamfunction_residual(
                    ctx, grid, q_att, PeriodicStreamfunctionFluctuations{e6_candidate1.span(), e6_candidate2.span()},
                    gauge, real{1}, source_config, histogram_config, e6_candidate_f1.span(), e6_candidate_f2.span(),
                    e6_residual_ws);
                const StreamfunctionResidualReport cand_res = synchronize_streamfunction_residual_report(
                    ctx, grid, real{1}, source_config, histogram_config, e6_residual_ws);
                const double r_F_candidate = static_cast<double>(cand_res.r_F);
                const bool finite_ok = std::isfinite(r_F_candidate);
                const bool reject = !finite_ok || r_F_candidate > 2.0 * r_F_k;
                const bool nonmonotone = finite_ok && r_F_candidate > r_F_k && r_F_candidate <= 2.0 * r_F_k;

                std::cout << "E6 k=" << k << " attempt=" << attempt << " r_F_k=" << r_F_k << " dtau=" << dtau
                          << " mu_k=" << mu_k << " gmres_status=" << gmres_status_label(gmres_report.status)
                          << " gmres_inner=" << gmres_report.total_inner_iterations
                          << " r_F_candidate=" << r_F_candidate << " finite=" << (finite_ok ? "true" : "false")
                          << (reject ? " REJECTED" : " ACCEPTED") << '\n';

                if (reject) {
                    dtau /= 2.0;
                    continue;
                }
                if (nonmonotone) {
                    std::cout << "E6_NONMONOTONE_ACCEPTED k=" << k << " r_F_k=" << r_F_k
                              << " r_F_candidate=" << r_F_candidate << '\n';
                }
                blas::copy(ctx, DeviceSpan<const real>(e6_candidate1.span()), e6_state1.span());
                blas::copy(ctx, DeviceSpan<const real>(e6_candidate2.span()), e6_state2.span());
                ctx.synchronize();
                accepted = true;
                accepted_r_F = r_F_candidate;
            }

            if (!accepted) {
                std::cout << "E6_STEP_FAILURE k=" << k << '\n';
                step_failure = true;
                break;
            }

            // SER update AFTER an accepted step (uses the accepted candidate's r_F).
            dtau = std::clamp(dtau * (r_F_prev / accepted_r_F), 1.0, 1e6);
            r_F_prev = accepted_r_F;
        }

        e6_pass = reached;
        check("E6_psitc_reaches_1e-4_within_60_steps", e6_pass);
    } else if (e5_pass) {
        std::cout << "E6_SKIPPED_E5_PASSED\n";
    }

    // =========================================================================
    // E6b (PRESPECIFIED, bitácora 2026-08-14T13:05Z, corrective C03):
    // micro-step scan from the FROZEN state -- print-only evidence, no state
    // update, no check. Closes the step-size loophole: if even near-
    // infinitesimal flow steps (large mu, i.e. small dtau=1/mu) ascend r_F,
    // F^T J A^-1 F < 0 is established directly at the frozen state,
    // independent of the SER dtau schedule E6 explored.
    // =========================================================================
    jvp.prepare_jvp_base(ctx, grid, q_att, CoupledVectorView{fields.u1_span(), fields.u2_span()}, gauge,
                         real{1}, source_config, histogram_config);

    const real e6b_rel_tol = std::clamp(static_cast<real>(std::sqrt(r_F_frozen)), real{1e-8}, real{1e-1});
    const double e6b_mu_list[] = {1.0, 10.0};
    for (double e6b_mu_d : e6b_mu_list) {
        const real e6b_mu = static_cast<real>(e6b_mu_d);
        ShiftedJacobianOperator e6b_op(jvp, grid, q_att, e6b_mu);
        e6b_op.prepare(n);
        CoupledGmres e6b_gmres;
        e6b_gmres.prepare(n, 10);

        DeviceBuffer<real> e6b_delta1(n), e6b_delta2(n);
        blas::fill(ctx, e6b_delta1.span(), real{0});
        blas::fill(ctx, e6b_delta2.span(), real{0});

        CoupledGmresConfig e6b_config;
        e6b_config.restart = 10;
        e6b_config.max_iterations = 100;
        e6b_config.rel_tol = e6b_rel_tol;

        const CoupledGmresReport e6b_report = e6b_gmres.solve(
            ctx, grid, e6b_op, precond,
            ConstCoupledVectorView(DeviceSpan<const real>(b1.span()), DeviceSpan<const real>(b2.span())),
            e6b_config, JvpDeltaConfig{}, CoupledVectorView{e6b_delta1.span(), e6b_delta2.span()});
        ctx.synchronize();

        DeviceBuffer<real> e6b_candidate1(n), e6b_candidate2(n);
        blas::copy(ctx, DeviceSpan<const real>(fields.u1_span()), e6b_candidate1.span());
        blas::copy(ctx, DeviceSpan<const real>(fields.u2_span()), e6b_candidate2.span());
        blas::axpy(ctx, real{1}, DeviceSpan<const real>(e6b_delta1.span()), e6b_candidate1.span());
        blas::axpy(ctx, real{1}, DeviceSpan<const real>(e6b_delta2.span()), e6b_candidate2.span());
        projector.project(ctx, e6b_candidate1.span(), mean_zero_ws);
        projector.project(ctx, e6b_candidate2.span(), mean_zero_ws);

        StreamfunctionResidualWorkspace e6b_residual_ws;
        e6b_residual_ws.prepare(n);
        DeviceBuffer<real> e6b_cand_f1(n), e6b_cand_f2(n);
        enqueue_streamfunction_residual(
            ctx, grid, q_att, PeriodicStreamfunctionFluctuations{e6b_candidate1.span(), e6b_candidate2.span()},
            gauge, real{1}, source_config, histogram_config, e6b_cand_f1.span(), e6b_cand_f2.span(),
            e6b_residual_ws);
        const StreamfunctionResidualReport e6b_cand_res = synchronize_streamfunction_residual_report(
            ctx, grid, real{1}, source_config, histogram_config, e6b_residual_ws);
        const double r_F_candidate = static_cast<double>(e6b_cand_res.r_F);

        std::cout << "E6b mu=" << e6b_mu_d << " gmres_status=" << gmres_status_label(e6b_report.status)
                  << " inner=" << e6b_report.total_inner_iterations << " r_F_frozen=" << r_F_frozen
                  << " r_F_candidate=" << r_F_candidate << " delta_rF=" << (r_F_candidate - r_F_frozen)
                  << '\n';
    }

    // =========================================================================
    // E7 (PRESPECIFIED, bitácora 2026-08-14T13:05Z, corrective C03):
    // epsilon-fold probe at the SAME frozen state -- print-only evidence, no
    // verdict change. Tests whether the eta=1 plateau is an epsilon=1e-2
    // regularization artifact of a fold in the solution branch (eta_fold
    // crossing eta=1 near lambda~0.5 for sigma^2=1) rather than an intrinsic
    // obstruction at eta=1.
    // =========================================================================
    NonlinearSourceConfig e7_source_config;
    e7_source_config.epsilon = real{1e-3};
    e7_source_config.v_rms = static_cast<real>(v_rms);

    // (i) frozen-state residual under epsilon=1e-3.
    StreamfunctionResidualWorkspace e7_frozen_residual_ws;
    e7_frozen_residual_ws.prepare(n);
    DeviceBuffer<real> e7_frozen_f1(n), e7_frozen_f2(n);
    enqueue_streamfunction_residual(ctx, grid, q_att,
                                    PeriodicStreamfunctionFluctuations{fields.u1_span(), fields.u2_span()},
                                    gauge, real{1}, e7_source_config, histogram_config, e7_frozen_f1.span(),
                                    e7_frozen_f2.span(), e7_frozen_residual_ws);
    const StreamfunctionResidualReport e7_frozen_residual = synchronize_streamfunction_residual_report(
        ctx, grid, real{1}, e7_source_config, histogram_config, e7_frozen_residual_ws);
    std::cout << "E7 frozen_state_r_F_at_eps1e-3 = " << e7_frozen_residual.r_F << '\n';

    // Save the frozen fields before the warm-started probe mutates them.
    DeviceBuffer<real> e7_saved_u1(n), e7_saved_u2(n);
    blas::copy(ctx, DeviceSpan<const real>(fields.u1_span()), e7_saved_u1.span());
    blas::copy(ctx, DeviceSpan<const real>(fields.u2_span()), e7_saved_u2.span());
    ctx.synchronize();

    // (ii) one warm-started stage solve at eta=1 with epsilon=1e-3, reusing
    // the E2 stage's lambda/q/hierarchy (coefficient_state=reuse); epsilon
    // does not touch q/hierarchy/the affine RHS, so this is exactly the
    // reuse contract.
    StreamfunctionSolverConfig e7_stage_config = stage_config;
    e7_stage_config.epsilon = real{1e-3};
    e7_stage_config.picard.max_iter = 200;
    e7_stage_config.coefficient_state = CoefficientState::reuse;

    const StreamfunctionSolveReport e7_stage_report =
        solve_streamfunctions(ctx, problem_view, e7_stage_config, fields, workspace);
    ctx.synchronize();

    double e7_best_r_F = static_cast<double>(e7_stage_report.residual.r_F);
    std::cout << "E7 stage_eps1e-3: status=" << solve_status_label(e7_stage_report.status)
              << " exit_reason=" << exit_reason_label(e7_stage_report.exit_reason)
              << " picard_iterations=" << e7_stage_report.picard_iterations
              << " r_F=" << e7_stage_report.residual.r_F
              << " anderson_acc=" << e7_stage_report.anderson_accepted << '\n';

    // (iii) if (ii) did not converge, the E5-style LM mini-solve at
    // epsilon=1e-3, continuing from (ii)'s resulting fields state.
    if (e7_stage_report.status != StreamfunctionSolveStatus::converged) {
        DeviceBuffer<real> e7lm_state1(n), e7lm_state2(n);
        blas::copy(ctx, DeviceSpan<const real>(fields.u1_span()), e7lm_state1.span());
        blas::copy(ctx, DeviceSpan<const real>(fields.u2_span()), e7lm_state2.span());
        ctx.synchronize();

        ShiftedJacobianOperator e7lm_op(jvp, grid, q_att, real{0});
        e7lm_op.prepare(n);
        CoupledGmres e7lm_gmres;
        e7lm_gmres.prepare(n, 10);

        StreamfunctionResidualWorkspace e7lm_residual_ws;
        e7lm_residual_ws.prepare(n);
        DeviceBuffer<real> e7lm_f1(n), e7lm_f2(n);
        DeviceBuffer<real> e7lm_trial1(n), e7lm_trial2(n);
        DeviceBuffer<real> e7lm_trial_f1(n), e7lm_trial_f2(n);
        DeviceBuffer<real> e7lm_delta1(n), e7lm_delta2(n);
        DeviceBuffer<real> e7lm_rhs1(n), e7lm_rhs2(n);

        const double theta = mu_star / r_F_frozen; // theta rule unchanged (E5).
        constexpr real kE7LmAlphaMin = real{0.03125}; // 2^-5, matches E5's kAlphaMin.
        constexpr real kE7LmArmijoC = real{1e-4};      // matches E5's kArmijoC.

        bool e7lm_reached = false;
        bool e7lm_step_failure = false;
        double e7lm_final_r_F = std::numeric_limits<double>::quiet_NaN();
        for (int k = 0; k < 30 && !e7lm_reached && !e7lm_step_failure; ++k) {
            enqueue_streamfunction_residual(
                ctx, grid, q_att, PeriodicStreamfunctionFluctuations{e7lm_state1.span(), e7lm_state2.span()},
                gauge, real{1}, e7_source_config, histogram_config, e7lm_f1.span(), e7lm_f2.span(),
                e7lm_residual_ws);
            const StreamfunctionResidualReport res_k = synchronize_streamfunction_residual_report(
                ctx, grid, real{1}, e7_source_config, histogram_config, e7lm_residual_ws);
            const double r_F_k = static_cast<double>(res_k.r_F);
            e7lm_final_r_F = r_F_k;

            if (r_F_k <= 1e-4) {
                e7lm_reached = true;
                std::cout << "E7-LM k=" << k << " r_F=" << r_F_k << " REACHED\n";
                break;
            }

            jvp.prepare_jvp_base(ctx, grid, q_att, CoupledVectorView{e7lm_state1.span(), e7lm_state2.span()},
                                 gauge, real{1}, e7_source_config, histogram_config);

            blas::copy(ctx, DeviceSpan<const real>(e7lm_f1.span()), e7lm_rhs1.span());
            blas::copy(ctx, DeviceSpan<const real>(e7lm_f2.span()), e7lm_rhs2.span());
            blas::scal(ctx, e7lm_rhs1.span(), real{-1});
            blas::scal(ctx, e7lm_rhs2.span(), real{-1});

            const real mu_k = static_cast<real>(theta * r_F_k);
            e7lm_op.set_mu(mu_k);

            CoupledGmresConfig e7lm_config;
            e7lm_config.restart = 10;
            e7lm_config.max_iterations = 100;
            e7lm_config.rel_tol = std::clamp(static_cast<real>(std::sqrt(r_F_k)), real{1e-8}, real{1e-1});

            blas::fill(ctx, e7lm_delta1.span(), real{0});
            blas::fill(ctx, e7lm_delta2.span(), real{0});
            const CoupledGmresReport gmres_report = e7lm_gmres.solve(
                ctx, grid, e7lm_op, precond,
                ConstCoupledVectorView(DeviceSpan<const real>(e7lm_rhs1.span()),
                                      DeviceSpan<const real>(e7lm_rhs2.span())),
                e7lm_config, JvpDeltaConfig{}, CoupledVectorView{e7lm_delta1.span(), e7lm_delta2.span()});
            ctx.synchronize();

            bool accepted = false;
            real accepted_alpha = real{0};
            real alpha = real{1};
            const double phi_k = 0.5 * r_F_k * r_F_k;
            while (alpha >= kE7LmAlphaMin) {
                blas::copy(ctx, DeviceSpan<const real>(e7lm_state1.span()), e7lm_trial1.span());
                blas::copy(ctx, DeviceSpan<const real>(e7lm_state2.span()), e7lm_trial2.span());
                blas::axpy(ctx, alpha, DeviceSpan<const real>(e7lm_delta1.span()), e7lm_trial1.span());
                blas::axpy(ctx, alpha, DeviceSpan<const real>(e7lm_delta2.span()), e7lm_trial2.span());
                projector.project(ctx, e7lm_trial1.span(), mean_zero_ws);
                projector.project(ctx, e7lm_trial2.span(), mean_zero_ws);

                enqueue_streamfunction_residual(
                    ctx, grid, q_att, PeriodicStreamfunctionFluctuations{e7lm_trial1.span(), e7lm_trial2.span()},
                    gauge, real{1}, e7_source_config, histogram_config, e7lm_trial_f1.span(),
                    e7lm_trial_f2.span(), e7lm_residual_ws);
                const StreamfunctionResidualReport trial_res = synchronize_streamfunction_residual_report(
                    ctx, grid, real{1}, e7_source_config, histogram_config, e7lm_residual_ws);
                const double r_F_trial = static_cast<double>(trial_res.r_F);
                const bool finite_ok = std::isfinite(r_F_trial);
                const double phi_trial = 0.5 * r_F_trial * r_F_trial;
                const bool armijo_ok = finite_ok && phi_trial <= (1.0 - static_cast<double>(kE7LmArmijoC) *
                                                                          static_cast<double>(alpha)) *
                                                                        phi_k;

                std::cout << "  E7-LM k=" << k << " alpha=" << alpha << " r_F_trial=" << r_F_trial
                          << " finite=" << (finite_ok ? "true" : "false")
                          << " armijo=" << (armijo_ok ? "accept" : "reject") << '\n';

                if (armijo_ok) {
                    accepted = true;
                    accepted_alpha = alpha;
                    blas::copy(ctx, DeviceSpan<const real>(e7lm_trial1.span()), e7lm_state1.span());
                    blas::copy(ctx, DeviceSpan<const real>(e7lm_trial2.span()), e7lm_state2.span());
                    ctx.synchronize();
                    break;
                }
                alpha *= real{0.5};
            }

            std::cout << "E7-LM k=" << k << " r_F=" << r_F_k << " mu_k=" << mu_k
                      << " gmres_status=" << gmres_status_label(gmres_report.status)
                      << " gmres_inner=" << gmres_report.total_inner_iterations
                      << " accepted_alpha=" << (accepted ? static_cast<double>(accepted_alpha) : 0.0) << '\n';

            if (!accepted) {
                std::cout << "E7-LM_STEP_FAILURE k=" << k << '\n';
                e7lm_step_failure = true;
            }
        }

        std::cout << "E7_LM_result r_F=" << e7lm_final_r_F << '\n';
        if (std::isfinite(e7lm_final_r_F)) {
            e7_best_r_F = std::min(e7_best_r_F, e7lm_final_r_F);
        }
    }

    const char* e7_verdict_evidence =
        e7_best_r_F < 1e-5 ? "decisive" : (e7_best_r_F <= 1e-4 ? "supportive" : "refuting");
    std::cout << "E7 verdict-evidence: " << e7_verdict_evidence << " (best_r_F=" << e7_best_r_F << ")\n";

    // Restore fields to the frozen state so any later code sees it unchanged.
    blas::copy(ctx, DeviceSpan<const real>(e7_saved_u1.span()), fields.u1_span());
    blas::copy(ctx, DeviceSpan<const real>(e7_saved_u2.span()), fields.u2_span());
    ctx.synchronize();

    // =========================================================================
    // E8 (PRESPECIFIED, bitácora 2026-08-14T14:00Z, corrective C04): the LAST
    // probe before escalation, print-only, no verdict change. Every stalled
    // path above is warm-started along the lambda-continuation branch; the
    // paper instead initializes with the harmonic (zero-source) solve at
    // FULL heterogeneity. E8 tests a DIRECT zero-source solve at the frozen
    // attempt's parameters (lambda=0.5125, sigma_Y^2=1) using the SAME
    // `problem_view` (Y_att/log_conductivity_y, the attempt flow velocity,
    // benchmark(1) gauge) already in scope. A NEW fields/workspace pair is
    // used so the E2 workspace's coefficient/hierarchy state and the frozen
    // `fields` used by any later code stay untouched for auditability.
    // =========================================================================
    StreamfunctionFields e8_fields;
    StreamfunctionWorkspace e8_workspace;

    StreamfunctionSolverConfig e8_config = base_config; // E2 freeze config: adaptive
                                                         // defaults, anderson R5 enabled,
                                                         // newton disabled.
    e8_config.eta = real{1};                            // default; explicit per the E8 spec.
    e8_config.epsilon = real{1e-2};                     // default; explicit per the E8 spec.
    e8_config.initial_state = PicardInitialState::zero_source; // default; explicit -- the
                                                                 // harmonic-init probe itself.
    e8_config.coefficient_state = CoefficientState::rebuild;   // default; explicit.
    e8_config.picard.max_iter = 500;

    const StreamfunctionSolveReport e8_report =
        solve_streamfunctions(ctx, problem_view, e8_config, e8_fields, e8_workspace);
    ctx.synchronize();

    std::cout << "E8 harmonic_init: status=" << solve_status_label(e8_report.status)
              << " exit_reason=" << exit_reason_label(e8_report.exit_reason)
              << " picard_iterations=" << e8_report.picard_iterations << " r_F=" << e8_report.residual.r_F
              << " anderson_acc=" << e8_report.anderson_accepted
              << " anderson_rej=" << e8_report.anderson_rejected << '\n';

    if (!e8_report.picard_history.empty()) {
        std::cout << "E8 r_F_history first=" << e8_report.picard_history.front().r_F
                  << " last=" << e8_report.picard_history.back().r_F << '\n';
    } else {
        std::cout << "E8 r_F_history first=n/a last=n/a\n";
    }

    const double e8_r_F = static_cast<double>(e8_report.residual.r_F);
    const bool e8_converged = e8_report.status == StreamfunctionSolveStatus::converged;
    const char* e8_verdict_evidence =
        (e8_converged && e8_r_F <= 1e-6)
            ? "decisive_branch_fold(r_F<=1e-6)"
            : ((!e8_converged && e8_r_F >= 1e-4 && e8_r_F <= 1e-2) ? "refuting_intrinsic(~1e-3)"
                                                                    : "inconclusive");
    std::cout << "E8 verdict-evidence: " << e8_verdict_evidence << '\n';

    check("terminal_method_demonstrated_E5_or_E6", e5_pass || e6_pass);

    (void)e2_freeze_ok;

    std::cout << "case=terminal_dgate_diagnostic verdict=" << (pass ? "PASS" : "FAIL") << '\n';

    std::ostringstream detail;
    detail << "sigma_Y^2=1, 32^3, dx=1, seed=12345, corr_length=8, lambda_attempt=0.5125, newton "
              "disabled for the E2 freeze; E3 mu-sweep {1e-1,3e-2,1e-2,3e-3,1e-3,3e-4,1e-4,0}; E4 "
              "generalized inverse iteration (recorded, non-blocking); E5 bounded LM mini-solve "
              "(theta=mu_star/r_F_frozen, <=30 steps); E6 Psi-tc/backward-Euler probe with "
              "bounded-non-monotone safeguards (dtau_0=1/mu_star, SER schedule, <=60 steps) run "
              "iff E3 confirmed the mechanism and E5 failed (amendment E6a + decision E6, "
              "bitácora 2026-08-14T12:10Z); E6b micro-step scan + E7 epsilon-fold probe (print-only "
              "evidence, bitácora 2026-08-14T13:05Z); E8 harmonic-init probe (print-only, "
              "bitácora 2026-08-14T14:00Z)";

    return {pass,
            "terminal_dgate_diagnostic",
            "gpu-terminal-dgate-diagnostic",
            detail.str(),
            mu_star,
            r_F_frozen,
            "E2+E3+(E5 or E6) (E4 recorded, non-blocking) -- SF-25 D-gate",
            pass ? "all pass" : "some failed",
            "PRESPECIFIED SF-25 activation-bitácora D-gate protocol (E2-E5) plus the recorded "
            "corrective amendment E6a (E2 exit-reason assert accepts the {stagnated, "
            "omega_floor_rejected} plateau signatures) and decision E6 (Psi-tc/backward-Euler "
            "contingency probe): verdict PASS iff the amended E2 freeze asserts hold, the E3 "
            "mu-sweep confirms the shift mechanism (>=10x inner-iteration reduction over the mu=0 "
            "budget-exhaustion baseline), and EITHER the E5 monotone-Armijo LM mini-solve reaches "
            "r_F<=1e-4 within 30 steps OR the E6 Psi-tc probe reaches r_F<=1e-4 within 60 steps "
            "(the prespecified contingency when E5 stalls at a spurious merit local minimum); a "
            "failed E3 sweep is the honest falsification path (E5/E6 skipped, case FAILs with the "
            "E3/E4 evidence printed), citing amendment E6a + decision E6 "
            "(bitácora 2026-08-14T12:10Z)"};
}

// ---------------------------------------------------------------------------
// Case 3: terminal_resolution_probe (HEAVY, print-only evidence recorder).
// Implements the owner-approved experiment R1a/R1b PRESPECIFIED in the SF-25
// bitácora at 2026-08-14T15:20Z: two DIRECT (no continuation) 64^3 solves at
// the critical amplitude 0.5125*Y_unit (sigma_Y^2=1, seed 12345,
// normalize_variance) that shelves at ~1e-3 on the 32^3 D-gate fixture,
// discriminating whether the eta=1 attainability boundary is a
// resolution-per-correlation-length (ell/h) effect: R1a doubles ell/h to 16
// (the paper's own resolution) at the 32^3 fixture's domain ratio (L/ell=4);
// R1b (control) keeps ell/h=8 at the new grid size (L/ell=8), isolating
// ell/h from grid size and domain ratio. SF-25 C06 adds the PRESPECIFIED
// deconfounding run R2 (bitácora 2026-08-14T15:55Z): the R1 run showed
// BOTH R1a/R1b dying of premature adaptive-omega collapse mid-descent
// (never reaching the 32^3 shelf phenomenology), confounding the readout;
// R2a/R2b rerun the SAME two problems with `newton.enabled = true` (SF-24
// defaults) -- the accepted stack as designed, whose activation threshold
// and rescue machinery are the designed answer to globalizer stalls. This
// case is an ALWAYS-PASS evidence recorder -- the printed "R1 joint-verdict"
// and "R2 joint-verdict" lines are evidence for the orchestrator/owner's
// resolution-surface decision, NOT a pass/fail test gate (per the
// owner-approved experiment design, bitácora 2026-08-14T15:20Z, extended by
// the R2 decision at 2026-08-14T15:55Z).
// ---------------------------------------------------------------------------

struct ResolutionProbeResult {
    StreamfunctionSolveStatus status{StreamfunctionSolveStatus::not_run};
    double r_F{std::numeric_limits<double>::quiet_NaN()};
};

// Shared helper for R1a/R1b/R2a/R2b: a single DIRECT zero-source 64^3 solve
// at the critical amplitude 0.5125*Y_unit, sigma_Y^2=1, seed 12345, the
// requested correlation length `ell`. Mirrors the E2/E8 per-lambda setup (Y
// scale, K=exp(Y), SF-19 affine flow solve, log_conductivity_y problem
// view, benchmark(1) gauge) but with NO continuation: a fresh
// field/flow/fields/workspace pair per call, exactly as R1 specifies. SF-25
// C06 (bitácora 2026-08-14T15:55Z, deconfounding run R2): `enable_newton`
// threads `config.newton.enabled` -- SF-24 defaults otherwise (newton
// requires `adaptive.enabled`, which is the solver default `true`); `label`
// is printed verbatim as the line prefix (e.g. "R1a"/"R2a") so the R1 and
// R2 sub-runs are distinguishable in the raw log.
[[nodiscard]] ResolutionProbeResult run_resolution_probe_solve(const char* label, int n, real ell,
                                                                bool enable_newton) {
    std::cout << std::setprecision(17);

    const Grid3D grid(n, n, n, real{1}, real{1}, real{1}); // dx=1, per the R1 spec.
    const std::size_t cells = grid.num_cells();
    CudaContext ctx(0);

    physics::PeriodicGaussianFieldConfig field_config;
    field_config.sigma2 = real{1};
    field_config.corr_length = ell;
    field_config.seed = 12345ULL;
    field_config.normalize_variance = true;

    DeviceBuffer<real> y(cells);
    physics::PeriodicGaussianFieldWorkspace field_workspace;
    const physics::PeriodicGaussianFieldReport field_report =
        physics::generate_periodic_gaussian_field(ctx, grid, field_config, y.span(), field_workspace);
    ctx.synchronize();

    // Scale by the critical amplitude 0.5125 (the SF-25 D-gate frozen
    // amplitude that shelves at 32^3, ell/h=8).
    blas::scal(ctx, y.span(), real{0.5125});

    DeviceBuffer<real> k(cells);
    terminal_dgate_enqueue_exp(ctx, DeviceSpan<const real>(y.span()), k.span());

    const std::size_t u_size = compact_mac_u_size(grid);
    const std::size_t v_size = compact_mac_v_size(grid);
    const std::size_t w_size = compact_mac_w_size(grid);
    DeviceBuffer<real> flow_u(u_size), flow_v(v_size), flow_w(w_size);
    physics::AffinePeriodicFlowWorkspace flow_workspace;
    physics::AffinePeriodicVelocityView velocity{flow_u.span(), flow_v.span(), flow_w.span()};
    const physics::AffinePeriodicFlowConfig flow_config{}; // qbar=(1,0,0) default.
    (void)physics::solve_affine_periodic_flow(ctx, grid, DeviceSpan<const real>(k.span()), flow_config,
                                               velocity, flow_workspace);
    ctx.synchronize();

    StreamfunctionProblemView problem_view;
    problem_view.grid = grid;
    problem_view.conductivity = DeviceSpan<const real>(y.span());
    problem_view.conductivity_representation = ConductivityRepresentation::log_conductivity_y;
    problem_view.darcy_velocity = CompactMacVelocityConstView{DeviceSpan<const real>(flow_u.span()),
                                                               DeviceSpan<const real>(flow_v.span()),
                                                               DeviceSpan<const real>(flow_w.span())};
    problem_view.bc = triply_periodic();
    problem_view.gauge = AffineGauge::benchmark(real{1});

    StreamfunctionSolverConfig config{}; // full defaults (eta=1, epsilon=1e-2, newton disabled).
    config.anderson.enabled = true;
    config.anderson.depth = 5;
    config.anderson.start_iteration = 5;
    config.anderson.condition_limit = real{1e12};
    config.initial_state = PicardInitialState::zero_source; // default; explicit -- the R1 spec.
    config.coefficient_state = CoefficientState::rebuild;   // default; explicit.
    config.picard.max_iter = 500;
    config.newton.enabled = enable_newton; // SF-25 C06 (R2): SF-24 defaults otherwise.

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;

    const auto t0 = std::chrono::steady_clock::now();
    const StreamfunctionSolveReport report =
        solve_streamfunctions(ctx, problem_view, config, fields, workspace);
    ctx.synchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const double wall_seconds = std::chrono::duration<double>(t1 - t0).count();

    const double ell_over_h = static_cast<double>(ell) / static_cast<double>(grid.dx);
    const double domain_length = static_cast<double>(n) * static_cast<double>(grid.dx);
    const double l_over_ell = domain_length / static_cast<double>(ell);

    std::cout << label << " n=" << n << " ell=" << static_cast<double>(ell)
              << " ell_over_h=" << ell_over_h << " L_over_ell=" << l_over_ell
              << " field_final_variance=" << field_report.final_variance
              << " status=" << solve_status_label(report.status)
              << " exit_reason=" << exit_reason_label(report.exit_reason)
              << " picard_iterations=" << report.picard_iterations << " r_F=" << report.residual.r_F
              << " anderson_acc=" << report.anderson_accepted << " anderson_rej=" << report.anderson_rejected
              << " wall_seconds=" << wall_seconds << " newton=" << (enable_newton ? 1 : 0)
              << " newton_act=" << report.newton_activations
              << " newton_acc=" << report.newton_steps_accepted
              << " newton_fail=" << report.newton_step_failures
              << " newton_rescue=" << report.newton_rescue_events << '\n';

    if (!report.picard_history.empty()) {
        std::cout << label << " r_F_history first=" << report.picard_history.front().r_F
                  << " last=" << report.picard_history.back().r_F << '\n';
    } else {
        std::cout << label << " r_F_history first=n/a last=n/a\n";
    }

    return {report.status, static_cast<double>(report.residual.r_F)};
}

// ===========================================================================
// SF-25 Phase-1 (S2-b) ON-path evidence: terminal_floor_guard_continuation.
//
// Reuses the SAME R1a fixture (bitácora 2026-08-14T15:55Z, D-gate run
// `sf25-resprobe`): 64^3, dx=1, ell=16 (ell/h=16, L/ell=4, the paper's own
// resolution-per-correlation-length), sigma_Y^2=1, seed=12345,
// normalize_variance, amplitude 0.5125*Y_unit, eta=1, epsilon=1e-2,
// zero_source init, anderson R5 (depth 5, start 5, limit 1e12), newton
// disabled, adaptive defaults, picard.max_iter=500 -- the EXACT, bitwise-
// reproducible (four independent D-gate reruns) configuration documented to
// die `omega_floor_rejected` at k=21, r_F=9.075e-3, with the trajectory
// "STILL DESCENDING healthily" (0.263 -> 9.1e-3 over the 21 accepted
// iterations, an ~29x reduction -- far exceeding any plausible
// drop_factor=0.9-over-5-iterations floor). This is real, previously
// measured phenomenology (premature adaptive-omega collapse mid-descent),
// not a forced/synthetic mechanism, making it the PRESPECIFIED-fallback
// exemplar for the floor_guard ON-path condition ("a small problem that is
// steadily descending when the first rejection occurs").
// ===========================================================================

[[nodiscard]] CaseResult case_terminal_floor_guard_continuation() {
    std::cout << std::setprecision(17);

    constexpr int n = 64;
    const Grid3D grid(n, n, n, real{1}, real{1}, real{1}); // dx=1, per the R1 spec.
    const std::size_t cells = grid.num_cells();
    CudaContext ctx(0);

    physics::PeriodicGaussianFieldConfig field_config;
    field_config.sigma2 = real{1};
    field_config.corr_length = real{16}; // ell/h=16, the R1a resolution.
    field_config.seed = 12345ULL;
    field_config.normalize_variance = true;

    DeviceBuffer<real> y(cells);
    physics::PeriodicGaussianFieldWorkspace field_workspace;
    (void)physics::generate_periodic_gaussian_field(ctx, grid, field_config, y.span(),
                                                     field_workspace);
    ctx.synchronize();

    // Scale by the critical amplitude 0.5125 (the SF-25 D-gate frozen
    // amplitude that shelves at 32^3, ell/h=8, and premature-omega-collapses
    // at 64^3, ell/h=16 -- R1a, bitácora 2026-08-14T15:55Z).
    blas::scal(ctx, y.span(), real{0.5125});

    DeviceBuffer<real> k(cells);
    terminal_dgate_enqueue_exp(ctx, DeviceSpan<const real>(y.span()), k.span());

    const std::size_t u_size = compact_mac_u_size(grid);
    const std::size_t v_size = compact_mac_v_size(grid);
    const std::size_t w_size = compact_mac_w_size(grid);
    DeviceBuffer<real> flow_u(u_size), flow_v(v_size), flow_w(w_size);
    physics::AffinePeriodicFlowWorkspace flow_workspace;
    physics::AffinePeriodicVelocityView velocity{flow_u.span(), flow_v.span(), flow_w.span()};
    const physics::AffinePeriodicFlowConfig flow_config{}; // qbar=(1,0,0) default.
    (void)physics::solve_affine_periodic_flow(ctx, grid, DeviceSpan<const real>(k.span()),
                                               flow_config, velocity, flow_workspace);
    ctx.synchronize();

    StreamfunctionProblemView problem_view;
    problem_view.grid = grid;
    problem_view.conductivity = DeviceSpan<const real>(y.span());
    problem_view.conductivity_representation = ConductivityRepresentation::log_conductivity_y;
    problem_view.darcy_velocity = CompactMacVelocityConstView{DeviceSpan<const real>(flow_u.span()),
                                                               DeviceSpan<const real>(flow_v.span()),
                                                               DeviceSpan<const real>(flow_w.span())};
    problem_view.bc = triply_periodic();
    problem_view.gauge = AffineGauge::benchmark(real{1});

    StreamfunctionSolverConfig config{}; // full defaults (eta=1, epsilon=1e-2, newton disabled).
    config.anderson.enabled = true;
    config.anderson.depth = 5;
    config.anderson.start_iteration = 5;
    config.anderson.condition_limit = real{1e12};
    config.initial_state = PicardInitialState::zero_source; // default; explicit -- the R1 spec.
    config.coefficient_state = CoefficientState::rebuild;   // default; explicit.
    config.picard.max_iter = 500;
    // SF-25 Phase-1 (S2-b): the floor guard under test. window=5,
    // drop_factor=0.9 are the documented defaults; max_resets=1 is enough to
    // demonstrate continuation past the first (R1a-documented) floor hit
    // without masking a genuine terminal state indefinitely.
    config.adaptive.floor_guard.enabled = true;
    config.adaptive.floor_guard.window = 5;
    config.adaptive.floor_guard.drop_factor = real{0.9};
    config.adaptive.floor_guard.max_resets = 1;

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;
    const StreamfunctionSolveReport report =
        solve_streamfunctions(ctx, problem_view, config, fields, workspace);
    ctx.synchronize();

    bool pass = true;
    std::vector<std::pair<std::string, bool>> checks;
    const auto add_check = [&](const char* name, bool ok) {
        checks.emplace_back(name, ok);
        pass = pass && ok;
    };

    // The R1a-documented first floor hit is at k=21; the guard must have
    // intercepted at least once, and the solve must have continued strictly
    // past that point (more outer iterations than the un-guarded R1a run).
    add_check("omega_floor_guard_resets_ge_1", report.omega_floor_guard_resets >= 1);
    add_check("omega_floor_guard_resets_le_max_resets",
              report.omega_floor_guard_resets <= config.adaptive.floor_guard.max_resets);
    add_check("continued_past_r1a_first_floor_hit", report.picard_iterations > 21);

    std::cout << "terminal_floor_guard_continuation n=" << n << " ell=16 status="
              << solve_status_label(report.status)
              << " exit_reason=" << exit_reason_label(report.exit_reason)
              << " picard_iterations=" << report.picard_iterations << " r_F=" << report.residual.r_F
              << " omega_floor_guard_resets=" << report.omega_floor_guard_resets
              << " anderson_acc=" << report.anderson_accepted
              << " anderson_rej=" << report.anderson_rejected << '\n';
    for (const auto& [name, ok] : checks) {
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    }

    std::ostringstream detail;
    detail << "64^3 ell=16 (ell/h=16, L/ell=4), sigma_Y^2=1, seed=12345, normalize_variance, "
              "amplitude 0.5125*Y_unit, eta=1, epsilon=1e-2, zero_source init, coefficient "
              "rebuild, anderson R5 (depth 5, start 5, limit 1e12), newton disabled, "
              "picard.max_iter=500, SF-19 affine flow qbar=(1,0,0) -- the R1a fixture (bitácora "
              "2026-08-14T15:55Z), plus adaptive.floor_guard (window=5, drop_factor=0.9, "
              "max_resets=1)";

    return {pass,
            "terminal_floor_guard_continuation",
            "gpu-terminal-floor-guard-continuation",
            detail.str(),
            static_cast<double>(report.omega_floor_guard_resets),
            static_cast<double>(report.picard_iterations),
            "omega_floor_guard_resets>=1, picard_iterations>21 (past R1a's documented first floor "
            "hit at k=21)",
            pass ? "all pass" : "some failed",
            "SF-25 Phase-1 (S2-b): the R1a fixture is documented (bitácora 2026-08-14T15:55Z, four "
            "bitwise-reproduced D-gate reruns) to die omega_floor_rejected at k=21 while the "
            "residual is still descending healthily (0.263->9.1e-3, ~29x over 21 iterations, far "
            "exceeding a 10%-per-5-iterations requirement); with the floor guard enabled, that "
            "same rejection is intercepted, omega is reset, and the solve continues past k=21"};
}

[[nodiscard]] CaseResult case_terminal_resolution_probe() {
    std::cout << std::setprecision(17);

    // R1a: 64^3, ell=16 (ell/h=16, L/ell=4 -- the paper's own ell/h at the
    // 32^3 fixture's domain ratio). newton disabled -- the R1 spec.
    const ResolutionProbeResult r1a = run_resolution_probe_solve("R1a", 64, real{16}, /*enable_newton=*/false);
    // R1b (control): 64^3, ell=8 (ell/h=8, L/ell=8 -- the OLD ell/h at the
    // new grid size, isolating ell/h from grid size/domain ratio). newton
    // disabled -- the R1 spec.
    const ResolutionProbeResult r1b = run_resolution_probe_solve("R1b", 64, real{8}, /*enable_newton=*/false);

    // SF-25 C06 (bitácora 2026-08-14T15:55Z): PRESPECIFIED deconfounding run
    // R2 -- the SAME two problems with `newton.enabled = true` (SF-24
    // defaults), the accepted stack as designed, whose threshold activation
    // (r_F<=1e-2) fires exactly where R1a's omega-floor collapse died
    // (9.1e-3) and whose stagnation-redirect/rescue machinery is the
    // designed answer to globalizer stalls.
    const ResolutionProbeResult r2a = run_resolution_probe_solve("R2a", 64, real{16}, /*enable_newton=*/true);
    const ResolutionProbeResult r2b = run_resolution_probe_solve("R2b", 64, real{8}, /*enable_newton=*/true);

    const bool r1a_confirming =
        r1a.status == StreamfunctionSolveStatus::converged && r1a.r_F <= 1e-6;
    const bool r1a_shelved = r1a.status == StreamfunctionSolveStatus::not_converged;
    const bool r1b_shelf =
        r1b.status == StreamfunctionSolveStatus::not_converged && r1b.r_F >= 1e-4 && r1b.r_F <= 1e-2;
    const bool r1b_converged = r1b.status == StreamfunctionSolveStatus::converged;

    const char* verdict;
    if (r1a_confirming && r1b_shelf) {
        verdict = "RESOLUTION_HYPOTHESIS_CONFIRMED";
    } else if (!r1a_confirming && r1a_shelved) {
        verdict = "HYPOTHESIS_REFUTED_R1A_SHELVED";
    } else if (r1a_confirming && !r1b_shelf && r1b_converged) {
        verdict = "GRID_SIZE_EFFECT_R1B_CONVERGED";
    } else {
        verdict = "INCONCLUSIVE";
    }

    std::cout << "R1 joint-verdict: " << verdict << " (raw: r1a_status=" << solve_status_label(r1a.status)
              << " r1a_r_F=" << r1a.r_F << " r1b_status=" << solve_status_label(r1b.status)
              << " r1b_r_F=" << r1b.r_F << ")\n";

    // SF-25 C06 (bitácora 2026-08-14T15:55Z): PRESPECIFIED R2 joint-verdict,
    // deconfounding the R1 readout with the full accepted stack
    // (newton.enabled = true, SF-24 defaults).
    const bool r2a_confirming = r2a.status == StreamfunctionSolveStatus::converged && r2a.r_F <= 1e-6;
    const bool r2a_wall = r2a.status != StreamfunctionSolveStatus::converged && r2a.r_F >= 1e-4 && r2a.r_F <= 1e-2;
    const char* r2_verdict;
    if (r2a_confirming) {
        r2_verdict = "RESOLUTION_CONFIRMED_VIA_STACK";
    } else if (r2a_wall) {
        r2_verdict = "WALL_AT_ELLH16";
    } else {
        r2_verdict = "INCONCLUSIVE";
    }
    std::cout << "R2 joint-verdict: " << r2_verdict << " (raw: r2a_status=" << solve_status_label(r2a.status)
              << " r2a_r_F=" << r2a.r_F << ") | R2b: " << solve_status_label(r2b.status) << "/" << r2b.r_F
              << '\n';

    std::cout << "case=terminal_resolution_probe verdict=PASS (always-pass evidence recorder; the "
                 "printed R1/R2 joint-verdict lines above are evidence for the owner/orchestrator "
                 "resolution-surface decision, not a test gate)\n";

    std::ostringstream detail;
    detail << "R1a n=64 ell=16 (ell/h=16, L/ell=4) vs R1b n=64 ell=8 (ell/h=8, L/ell=8) control, "
              "both sigma_Y^2=1, seed=12345, normalize_variance, amplitude 0.5125*Y_unit, eta=1, "
              "epsilon=1e-2, zero_source init, coefficient rebuild, anderson R5 (depth 5, start 5, "
              "limit 1e12), newton disabled, picard.max_iter=500, SF-19 affine flow qbar=(1,0,0); "
              "owner-approved experiment R1 (bitácora 2026-08-14T15:20Z); PLUS the PRESPECIFIED "
              "deconfounding run R2 (bitácora 2026-08-14T15:55Z): the SAME two problems with "
              "newton.enabled=true (SF-24 defaults, the accepted stack as designed), rerun because "
              "R1a/R1b both died of premature adaptive-omega collapse mid-descent "
              "(omega_floor_rejected at r_F 9.1e-3 / 2.6e-2) rather than exhibiting the 32^3 shelf "
              "phenomenology; always-pass evidence recorder -- the reported verdicts are evidence "
              "for the orchestrator/owner, not a test gate";

    return {true,
            "terminal_resolution_probe",
            "gpu-terminal-resolution-probe",
            detail.str(),
            r1a.r_F,
            r1b.r_F,
            verdict,
            "always pass (evidence recorder; see the printed R1/R2 joint-verdict lines)",
            "owner-approved experiment R1 (bitácora 2026-08-14T15:20Z) PLUS the PRESPECIFIED "
            "deconfounding run R2 (bitácora 2026-08-14T15:55Z): R1a/R1b (newton disabled) then "
            "R2a/R2b (newton enabled, SF-24 defaults, the accepted stack as designed) at the SAME "
            "two problems, print-only, no pass/fail assertion -- the case ALWAYS returns pass=true "
            "because it is an evidence recorder for the owner's resolution-surface decision, not a "
            "D-gate-style test gate"};
}

// ===========================================================================
// SF-25 Phase-1 (P1-I) Case: terminal_eta_endgame (S1). Host-loop eta walk
// (NO ContinuationController changes): a FIXED prespecified eta ladder,
// each stage warm-started from the previous stage's ACCEPTED fields (first
// stage from zero_source init), on the SAME field/flow/gauge recipe the
// campaign uses (32^3, dx=1, seed 12345, corr_length 8, sigma^2=1
// normalize_variance, amplitude 0.5125*Y_unit, K=exp, log_conductivity_y,
// SF-19 affine flow qbar=(1,0,0), benchmark gauge). eta=1 is
// `StreamfunctionSolverConfig::eta`, the SAME Lester-source coupling weight
// being walked from 0 to 1 -- config.eta is set per stage exactly as the
// bitácora prescribes ("Set config.eta per stage").
// ===========================================================================

// The 19-value FIXED ladder: the coarse leg then the endgame halvings of
// `1-eta`, bound BEFORE any run (bitácora 2026-08-14T17:40Z).
constexpr double kEtaEndgameLadder[] = {
    0.0,       0.25,        0.5,          0.75,          0.9,
    0.95,      0.975,       0.9875,       0.99375,       0.996875,
    0.9984375, 0.99921875,  0.999609375,  0.9998046875,  0.99990234375,
    0.999951171875, 0.9999755859375, 0.99998779296875, 0.999993896484375};
constexpr int kEtaEndgameLadderSize = static_cast<int>(sizeof(kEtaEndgameLadder) / sizeof(kEtaEndgameLadder[0]));

struct EtaEndgameArm {
    // Verdict inputs.
    bool last_valid{false};
    double last_eta{std::numeric_limits<double>::quiet_NaN()};
    double last_r_F{std::numeric_limits<double>::quiet_NaN()};
    bool walk_stopped{false};
    double eta_fail{std::numeric_limits<double>::quiet_NaN()};
    bool eta_fail_plateau_signature{false}; // exit_reason in {stagnated, omega_floor_rejected}.
    double eta_fail_r_F{std::numeric_limits<double>::quiet_NaN()};
    bool coda_last_ran{false};
    StreamfunctionSolveStatus coda_last_status{StreamfunctionSolveStatus::not_run};
    double coda_last_r_F{std::numeric_limits<double>::quiet_NaN()};
    bool coda_extrap_ran{false};
    StreamfunctionSolveStatus coda_extrap_status{StreamfunctionSolveStatus::not_run};
    double coda_extrap_r_F{std::numeric_limits<double>::quiet_NaN()};
    double coda_extrap_init_r_F{std::numeric_limits<double>::quiet_NaN()};
    bool basin_entered{false};
};

// Runs ONE arm of the eta walk (fresh StreamfunctionFields/Workspace pair,
// starting from zero_source). `hygiene_on` applies the S1b hygiene settings
// (adaptive.floor_guard + anderson.restart_on_stagnation) verbatim per the
// bitácora; `hygiene_on == false` is S1a (pure P1-H-default behavior).
[[nodiscard]] EtaEndgameArm run_eta_endgame_arm(CudaContext& ctx, const StreamfunctionProblemView& problem_view,
                                                const char* arm_label, bool hygiene_on) {
    const Grid3D& grid = problem_view.grid;
    const std::size_t n = grid.num_cells();

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;

    // Fixed-size state-history staging, allocated ONCE (no allocations
    // inside the ladder loop): `last` (the most recently accepted stage's
    // state) and `prev` (the accepted stage immediately before `last`),
    // both sized n each (2*n per pair, per the memory constraint).
    DeviceBuffer<real> last_u1(n), last_u2(n), prev_u1(n), prev_u2(n);
    bool last_valid = false, prev_valid = false;
    double last_eta_val = 0.0, prev_eta_val = 0.0;

    EtaEndgameArm result;

    for (int i = 0; i < kEtaEndgameLadderSize; ++i) {
        const double eta_i = kEtaEndgameLadder[i];

        StreamfunctionSolverConfig config; // full defaults.
        config.eta = static_cast<real>(eta_i);
        config.epsilon = real{1e-2};
        config.anderson.enabled = true;
        config.anderson.depth = 5;
        config.anderson.start_iteration = 5;
        config.anderson.condition_limit = real{1e12};
        config.picard.max_iter = 500;
        if (hygiene_on) {
            config.adaptive.floor_guard.enabled = true;
            config.adaptive.floor_guard.window = 5;
            config.adaptive.floor_guard.drop_factor = real{0.9};
            config.adaptive.floor_guard.max_resets = 3;
            config.anderson.restart_on_stagnation = true;
            config.anderson.max_restarts = 2;
        }
        if (i == 0) {
            config.initial_state = PicardInitialState::zero_source;
            config.coefficient_state = CoefficientState::rebuild;
        } else {
            config.initial_state = PicardInitialState::warm_start;
            config.coefficient_state = CoefficientState::reuse;
        }

        const StreamfunctionSolveReport report = solve_streamfunctions(ctx, problem_view, config, fields, workspace);
        ctx.synchronize();

        const double r_F = static_cast<double>(report.residual.r_F);
        const bool pass = report.status == StreamfunctionSolveStatus::converged && r_F <= 1e-6;

        std::cout << arm_label << " eta=" << eta_i << " status=" << solve_status_label(report.status)
                  << " exit_reason=" << exit_reason_label(report.exit_reason)
                  << " iterations=" << report.picard_iterations << " r_F=" << r_F
                  << " anderson_acc=" << report.anderson_accepted << " anderson_rej=" << report.anderson_rejected
                  << " omega_floor_guard_resets=" << report.omega_floor_guard_resets
                  << " anderson_stagnation_restarts=" << report.anderson_stagnation_restarts
                  << " pass=" << (pass ? "true" : "false") << '\n';

        if (!pass) {
            result.walk_stopped = true;
            result.eta_fail = eta_i;
            result.eta_fail_r_F = r_F;
            result.eta_fail_plateau_signature = report.exit_reason == PicardExitReason::stagnated ||
                                                report.exit_reason == PicardExitReason::omega_floor_rejected;
            break;
        }

        if (last_valid) {
            blas::copy(ctx, DeviceSpan<const real>(last_u1.span()), prev_u1.span());
            blas::copy(ctx, DeviceSpan<const real>(last_u2.span()), prev_u2.span());
            prev_eta_val = last_eta_val;
            prev_valid = true;
        }
        blas::copy(ctx, DeviceSpan<const real>(fields.u1_span()), last_u1.span());
        blas::copy(ctx, DeviceSpan<const real>(fields.u2_span()), last_u2.span());
        ctx.synchronize();
        last_eta_val = eta_i;
        last_valid = true;
        result.last_r_F = r_F; // the LAST ACCEPTED ladder stage's own r_F (frontier_r_F).
    }

    result.last_valid = last_valid;
    result.last_eta = last_eta_val;

    // Coda (i): eta=1 attempt from the last accepted state.
    if (last_valid) {
        blas::copy(ctx, DeviceSpan<const real>(last_u1.span()), fields.u1_span());
        blas::copy(ctx, DeviceSpan<const real>(last_u2.span()), fields.u2_span());
        ctx.synchronize();

        StreamfunctionSolverConfig coda_config;
        coda_config.eta = real{1};
        coda_config.epsilon = real{1e-2};
        coda_config.anderson.enabled = true;
        coda_config.anderson.depth = 5;
        coda_config.anderson.start_iteration = 5;
        coda_config.anderson.condition_limit = real{1e12};
        coda_config.picard.max_iter = 500;
        if (hygiene_on) {
            coda_config.adaptive.floor_guard.enabled = true;
            coda_config.adaptive.floor_guard.window = 5;
            coda_config.adaptive.floor_guard.drop_factor = real{0.9};
            coda_config.adaptive.floor_guard.max_resets = 3;
            coda_config.anderson.restart_on_stagnation = true;
            coda_config.anderson.max_restarts = 2;
        }
        coda_config.initial_state = PicardInitialState::warm_start;
        coda_config.coefficient_state = CoefficientState::reuse;

        const StreamfunctionSolveReport coda_report =
            solve_streamfunctions(ctx, problem_view, coda_config, fields, workspace);
        ctx.synchronize();

        result.coda_last_ran = true;
        result.coda_last_status = coda_report.status;
        result.coda_last_r_F = static_cast<double>(coda_report.residual.r_F);
        if (coda_report.status == StreamfunctionSolveStatus::converged && result.coda_last_r_F <= 1e-6) {
            result.basin_entered = true;
        }

        std::cout << arm_label << " coda_last eta=1 status=" << solve_status_label(coda_report.status)
                  << " exit_reason=" << exit_reason_label(coda_report.exit_reason)
                  << " iterations=" << coda_report.picard_iterations << " r_F=" << result.coda_last_r_F << '\n';
    } else {
        std::cout << arm_label << " coda_last: skipped (no accepted stage)\n";
    }

    // Coda (ii): two-point linear extrapolation to eta=1, only when >=2
    // accepted stages had eta>=0.99 (last_eta and prev_eta both >=0.99).
    if (last_valid && prev_valid && last_eta_val >= 0.99 && prev_eta_val >= 0.99) {
        const double factor = (1.0 - last_eta_val) / (last_eta_val - prev_eta_val);

        DeviceBuffer<real> diff1(n), diff2(n), extrap1(n), extrap2(n);
        blas::copy(ctx, DeviceSpan<const real>(last_u1.span()), diff1.span());
        blas::copy(ctx, DeviceSpan<const real>(last_u2.span()), diff2.span());
        blas::axpy(ctx, real{-1}, DeviceSpan<const real>(prev_u1.span()), diff1.span());
        blas::axpy(ctx, real{-1}, DeviceSpan<const real>(prev_u2.span()), diff2.span());
        blas::copy(ctx, DeviceSpan<const real>(last_u1.span()), extrap1.span());
        blas::copy(ctx, DeviceSpan<const real>(last_u2.span()), extrap2.span());
        blas::axpy(ctx, static_cast<real>(factor), DeviceSpan<const real>(diff1.span()), extrap1.span());
        blas::axpy(ctx, static_cast<real>(factor), DeviceSpan<const real>(diff2.span()), extrap2.span());
        ctx.synchronize();

        // r_F AT u* under eta=1, BEFORE iterating: reuse the residual
        // evaluation path (ResidualEvaluator.cuh). v_rms is the measured
        // Darcy speed, constant for this fixed problem_view across every
        // stage; measure it once here via a standalone SF-11 physical-
        // diagnostics evaluation at the last accepted state (the same
        // measurement `solve_streamfunctions` performs internally).
        NonlinearSourceConfig source_config;
        source_config.epsilon = real{1e-2};
        StreamfunctionDiagnosticsWorkspace vrms_ws;
        vrms_ws.prepare(grid);
        DeviceBuffer<real> vrms_vpsi_u(compact_mac_u_size(grid)), vrms_vpsi_v(compact_mac_v_size(grid)),
            vrms_vpsi_w(compact_mac_w_size(grid));
        const PhysicalDiagnosticsConfig vrms_diag_config{};
        enqueue_streamfunction_physical_diagnostics(
            ctx, grid, PeriodicStreamfunctionFluctuations{last_u1.span(), last_u2.span()}, problem_view.gauge,
            problem_view.darcy_velocity, vrms_diag_config,
            CompactMacVelocityView{vrms_vpsi_u.span(), vrms_vpsi_v.span(), vrms_vpsi_w.span()}, vrms_ws);
        const PhysicalDiagnosticsReport vrms_report =
            synchronize_streamfunction_physical_diagnostics_report(ctx, grid, vrms_diag_config, vrms_ws);
        source_config.v_rms = vrms_report.v_d_rms;

        const ResidualHistogramConfig histogram_config{};
        const DeviceSpan<const real> q_att = workspace.q();

        StreamfunctionResidualWorkspace init_residual_ws;
        init_residual_ws.prepare(n);
        DeviceBuffer<real> init_f1(n), init_f2(n);
        enqueue_streamfunction_residual(ctx, grid, q_att,
                                        PeriodicStreamfunctionFluctuations{extrap1.span(), extrap2.span()},
                                        problem_view.gauge, real{1}, source_config, histogram_config,
                                        init_f1.span(), init_f2.span(), init_residual_ws);
        const StreamfunctionResidualReport init_res =
            synchronize_streamfunction_residual_report(ctx, grid, real{1}, source_config, histogram_config,
                                                        init_residual_ws);
        result.coda_extrap_init_r_F = static_cast<double>(init_res.r_F);
        std::cout << arm_label << " coda_extrap init_r_F=" << result.coda_extrap_init_r_F
                  << " factor=" << factor << " last_eta=" << last_eta_val << " prev_eta=" << prev_eta_val << '\n';

        blas::copy(ctx, DeviceSpan<const real>(extrap1.span()), fields.u1_span());
        blas::copy(ctx, DeviceSpan<const real>(extrap2.span()), fields.u2_span());
        ctx.synchronize();

        StreamfunctionSolverConfig extrap_config;
        extrap_config.eta = real{1};
        extrap_config.epsilon = real{1e-2};
        extrap_config.anderson.enabled = true;
        extrap_config.anderson.depth = 5;
        extrap_config.anderson.start_iteration = 5;
        extrap_config.anderson.condition_limit = real{1e12};
        extrap_config.picard.max_iter = 500;
        if (hygiene_on) {
            extrap_config.adaptive.floor_guard.enabled = true;
            extrap_config.adaptive.floor_guard.window = 5;
            extrap_config.adaptive.floor_guard.drop_factor = real{0.9};
            extrap_config.adaptive.floor_guard.max_resets = 3;
            extrap_config.anderson.restart_on_stagnation = true;
            extrap_config.anderson.max_restarts = 2;
        }
        extrap_config.initial_state = PicardInitialState::warm_start;
        extrap_config.coefficient_state = CoefficientState::reuse;

        const StreamfunctionSolveReport extrap_report =
            solve_streamfunctions(ctx, problem_view, extrap_config, fields, workspace);
        ctx.synchronize();

        result.coda_extrap_ran = true;
        result.coda_extrap_status = extrap_report.status;
        result.coda_extrap_r_F = static_cast<double>(extrap_report.residual.r_F);
        if (extrap_report.status == StreamfunctionSolveStatus::converged && result.coda_extrap_r_F <= 1e-6) {
            result.basin_entered = true;
        }

        std::cout << arm_label << " coda_extrap eta=1 status=" << solve_status_label(extrap_report.status)
                  << " exit_reason=" << exit_reason_label(extrap_report.exit_reason)
                  << " iterations=" << extrap_report.picard_iterations << " r_F=" << result.coda_extrap_r_F << '\n';
    } else {
        std::cout << arm_label
                  << " coda_extrap: skipped (fewer than 2 accepted stages with eta>=0.99)\n";
    }

    // Readout verdicts, mechanical application of the prespecified rules.
    const bool frontier_extended = last_valid && last_eta_val > 0.996875;
    const bool cliff_climbable = last_valid && last_eta_val >= (1.0 - 1e-4);
    const bool basin_entered = result.basin_entered;
    const bool eta_fold_suggested =
        result.walk_stopped && result.eta_fail < (1.0 - 1e-4) && result.eta_fail_r_F > 1e-5;

    std::cout << arm_label << " rule FRONTIER_EXTENDED: last accepted eta > 0.996875 -> "
              << (frontier_extended ? "true" : "false") << '\n';
    std::cout << arm_label << " rule CLIFF_CLIMBABLE: last accepted eta >= 1-1e-4 -> "
              << (cliff_climbable ? "true" : "false") << '\n';
    std::cout << arm_label << " rule BASIN_ENTERED: any eta=1 coda attempt converges to <=1e-6 -> "
              << (basin_entered ? "true" : "false") << '\n';
    std::cout << arm_label
              << " rule ETA_FOLD_SUGGESTED: walk stopped at eta_fail<1-1e-4 with r_F>1e-5 -> "
              << (eta_fold_suggested ? "true" : "false") << '\n';

    std::ostringstream verdicts;
    if (frontier_extended) verdicts << "FRONTIER_EXTENDED ";
    if (cliff_climbable) verdicts << "CLIFF_CLIMBABLE ";
    if (basin_entered) verdicts << "BASIN_ENTERED ";
    if (eta_fold_suggested) verdicts << "ETA_FOLD_SUGGESTED ";
    if (verdicts.str().empty()) verdicts << "NONE";
    std::cout << arm_label << " verdicts: " << verdicts.str() << '\n';

    std::cout << arm_label << " summary frontier_eta=" << (last_valid ? last_eta_val : -1.0)
              << " frontier_r_F=" << result.last_r_F
              << " walk_stopped_at=" << (result.walk_stopped ? std::to_string(result.eta_fail) : std::string("none"))
              << " coda_last=" << (result.coda_last_ran ? solve_status_label(result.coda_last_status) : "skipped")
              << "/" << result.coda_last_r_F
              << " coda_extrap=" << (result.coda_extrap_ran ? solve_status_label(result.coda_extrap_status) : "skipped")
              << "/" << result.coda_extrap_r_F << '\n';

    return result;
}

// ---------------------------------------------------------------------------
// SF-25 Phase-2 (P2-I) shared helper: `EtaEndgameProblem`/
// `build_eta_endgame_problem`. Caller-owned buffers backing a
// `StreamfunctionProblemView` built by the E2/E8/R1-style per-amplitude
// recipe (field * 0.5125 -> K=exp -> SF-19 affine flow -> log_conductivity_y
// view, benchmark gauge), parameterized by grid resolution `n` and
// correlation length `ell` (dx=1 fixed, matching every other 32^3/64^3
// fixture in this file). `case_terminal_eta_endgame`'s 32^3 call site below
// is refactored to call this helper with EXACTLY the previous constants
// (n=32, ell=8) -- identical construction order/values, identical prints
// (this construction path prints nothing of its own, so behavior is
// unchanged bit-for-bit). Movable (DeviceBuffer is move-only); the views
// inside `view` remain valid across the move because they hold device
// pointer VALUES, not addresses into this struct.
// ---------------------------------------------------------------------------

struct EtaEndgameProblem {
    DeviceBuffer<real> y;
    DeviceBuffer<real> k;
    DeviceBuffer<real> flow_u;
    DeviceBuffer<real> flow_v;
    DeviceBuffer<real> flow_w;
    StreamfunctionProblemView view;
};

[[nodiscard]] EtaEndgameProblem build_eta_endgame_problem(CudaContext& ctx, int n, real ell) {
    EtaEndgameProblem result;
    const Grid3D grid(n, n, n, real{1}, real{1}, real{1});
    const std::size_t cells = grid.num_cells();

    physics::PeriodicGaussianFieldConfig field_config;
    field_config.sigma2 = real{1};
    field_config.corr_length = ell;
    field_config.seed = 12345ULL;
    field_config.normalize_variance = true;

    result.y = DeviceBuffer<real>(cells);
    physics::PeriodicGaussianFieldWorkspace field_workspace;
    (void)physics::generate_periodic_gaussian_field(ctx, grid, field_config, result.y.span(), field_workspace);
    ctx.synchronize();

    blas::scal(ctx, result.y.span(), real{0.5125});

    result.k = DeviceBuffer<real>(cells);
    terminal_dgate_enqueue_exp(ctx, DeviceSpan<const real>(result.y.span()), result.k.span());

    result.flow_u = DeviceBuffer<real>(compact_mac_u_size(grid));
    result.flow_v = DeviceBuffer<real>(compact_mac_v_size(grid));
    result.flow_w = DeviceBuffer<real>(compact_mac_w_size(grid));
    physics::AffinePeriodicFlowWorkspace flow_workspace;
    physics::AffinePeriodicVelocityView velocity{result.flow_u.span(), result.flow_v.span(), result.flow_w.span()};
    const physics::AffinePeriodicFlowConfig flow_config{}; // qbar=(1,0,0) default.
    (void)physics::solve_affine_periodic_flow(ctx, grid, DeviceSpan<const real>(result.k.span()), flow_config,
                                              velocity, flow_workspace);
    ctx.synchronize();

    result.view.grid = grid;
    result.view.conductivity = DeviceSpan<const real>(result.y.span());
    result.view.conductivity_representation = ConductivityRepresentation::log_conductivity_y;
    result.view.darcy_velocity = CompactMacVelocityConstView{DeviceSpan<const real>(result.flow_u.span()),
                                                              DeviceSpan<const real>(result.flow_v.span()),
                                                              DeviceSpan<const real>(result.flow_w.span())};
    result.view.bc = triply_periodic();
    result.view.gauge = AffineGauge::benchmark(real{1});
    return result;
}

[[nodiscard]] CaseResult case_terminal_eta_endgame() {
    std::cout << std::setprecision(17);

    CudaContext ctx(0);

    // The SAME field/flow/gauge recipe the campaign uses (identical
    // construction to terminal_resolution_probe's 32^3-equivalent path):
    // sigma_Y^2=1, seed=12345, corr_length=8, normalize_variance, amplitude
    // 0.5125*Y_unit, K=exp(Y_att), SF-19 affine flow qbar=(1,0,0),
    // log_conductivity_y problem view, benchmark(1) gauge. This is a DIRECT
    // build (no continuation), reusing the E2/R1 per-lambda construction
    // pattern at a single fixed amplitude -- via `build_eta_endgame_problem`
    // (SF-25 P2-I), EXACTLY the previous constants (n=32, ell=8).
    EtaEndgameProblem problem32 = build_eta_endgame_problem(ctx, 32, real{8});
    const StreamfunctionProblemView& problem_view = problem32.view;

    const EtaEndgameArm s1a = run_eta_endgame_arm(ctx, problem_view, "S1a", /*hygiene_on=*/false);
    const EtaEndgameArm s1b = run_eta_endgame_arm(ctx, problem_view, "S1b", /*hygiene_on=*/true);

    std::cout << "case=terminal_eta_endgame verdict=PASS (always-pass evidence recorder; see the "
                 "printed per-stage table and per-arm rule/verdict lines above)\n";

    std::ostringstream detail;
    detail << "32^3 dx=1 seed=12345 corr_length=8 sigma_Y^2=1 normalize_variance amplitude=0.5125*Y_unit "
              "K=exp log_conductivity_y SF-19 affine flow qbar=(1,0,0) benchmark gauge epsilon=1e-2 fixed "
              "anderson R5 newton disabled picard.max_iter=500; FIXED prespecified eta ladder ("
           << kEtaEndgameLadderSize << " stages) warm-started stage-to-stage; two arms S1a (hygiene OFF) / "
              "S1b (floor_guard+anderson-restart ON); coda: eta=1 from last accepted state, and (when "
              ">=2 accepted stages had eta>=0.99) a two-point linear extrapolation to eta=1";

    return {true,
            "terminal_eta_endgame",
            "gpu-terminal-eta-endgame",
            detail.str(),
            s1a.last_valid ? s1a.last_eta : -1.0,
            s1b.last_valid ? s1b.last_eta : -1.0,
            "S1a/S1b frontier etas (see printed per-stage table)",
            "always pass (evidence recorder; see the printed per-stage table, per-arm summary, and "
            "verdict lines above)",
            "SF-25 Phase-1 (P1-I) S1 eta-endgame probe (bitácora 2026-08-14T17:40Z): host-loop eta "
            "walk over the FIXED prespecified ladder, two hygiene arms, coda attempts at eta=1 (last "
            "accepted state and two-point linear extrapolation), mechanical FRONTIER_EXTENDED/"
            "CLIFF_CLIMBABLE/BASIN_ENTERED/ETA_FOLD_SUGGESTED readouts; always-pass evidence recorder"};
}

// ===========================================================================
// SF-25 Phase-1 (P1-I) Case: terminal_shelf_probe_phase1 (S3micro + S6, ONE
// shared E2 freeze). Reuses the dgate case's E2 freeze protocol VERBATIM
// (hygiene flags all OFF) so the frozen state MUST reproduce the recorded
// `r_F_frozen = 1.1204722529922055e-3` bitwise. S3micro Part A (single-step
// mu-scan) and Part B (sustained descent) reuse the E6b/E5 shifted-Newton
// machinery pattern (`ShiftedJacobianOperator` + `CoupledGmres`). S6 reuses
// the SF-11 `PhysicalDiagnosticsReport` verbatim (public enqueue/synchronize
// pair) plus the host-replicated pointwise defect fields (see the shared
// helpers above) on BOTH the frozen shelf state and the lambda=0.5 accepted
// predecessor baseline (captured BEFORE the 0.5125 stage mutates `fields`).
// ===========================================================================

[[nodiscard]] CaseResult case_terminal_shelf_probe_phase1() {
    std::cout << std::setprecision(17);
    bool pass = true;
    const auto check = [&](const char* name, bool ok) {
        pass = pass && ok;
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    };

    constexpr int n32 = 32;
    const Grid3D grid(n32, n32, n32, real{1}, real{1}, real{1});
    const std::size_t n = grid.num_cells();
    CudaContext ctx(0);

    // =========================================================================
    // E2 freeze, VERBATIM from case_terminal_dgate_diagnostic (hygiene flags
    // all OFF): the sigma_Y^2=1, 32^3 smoke (seed 12345, ell=8,
    // normalize_variance, anderson R5, newton disabled) to its lambda floor,
    // then the lambda=0.5125 warm-started stage to its plateau exit.
    // =========================================================================
    physics::PeriodicGaussianFieldConfig field_config;
    field_config.sigma2 = real{1};
    field_config.corr_length = real{8};
    field_config.seed = 12345ULL;
    field_config.normalize_variance = true;

    DeviceBuffer<real> y(n);
    physics::PeriodicGaussianFieldWorkspace field_workspace;
    (void)physics::generate_periodic_gaussian_field(ctx, grid, field_config, y.span(), field_workspace);
    ctx.synchronize();

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;
    StreamfunctionSolverConfig base_config; // full defaults (adaptive Picard, newton DISABLED).
    base_config.anderson.enabled = true;
    base_config.anderson.depth = 5;
    base_config.anderson.start_iteration = 5;
    base_config.anderson.condition_limit = real{1e12};

    HeterogeneityContinuationConfig continuation_config{}; // lambda axis defaults.
    continuation_config.inner.epsilon_log10.target = continuation_config.inner.epsilon_log10.start;
    const physics::AffinePeriodicFlowConfig flow_config{}; // qbar=(1,0,0) default.

    const HeterogeneityContinuationReport freeze_report = run_streamfunction_heterogeneity_continuation(
        ctx, grid, DeviceSpan<const real>(y.span()), continuation_config, flow_config, base_config, fields,
        workspace);
    ctx.synchronize();

    std::cout << "E2 freeze: status=" << heterogeneity_status_label(freeze_report.status)
              << " final_lambda=" << freeze_report.final_lambda << " final_eta=" << freeze_report.final_eta
              << '\n';

    check("E2_freeze_status_lambda_floor_exhausted",
          freeze_report.status == HeterogeneityStatus::lambda_floor_exhausted);
    check("E2_freeze_final_lambda_eq_0_5", freeze_report.final_lambda == real{0.5});
    check("E2_freeze_final_eta_eq_1", freeze_report.final_eta == real{1});

    // S6 baseline column: the accepted (lambda=0.5, eta=1) predecessor state,
    // captured BEFORE the 0.5125 warm-started stage mutates `fields`.
    DeviceBuffer<real> baseline_u1(n), baseline_u2(n);
    blas::copy(ctx, DeviceSpan<const real>(fields.u1_span()), baseline_u1.span());
    blas::copy(ctx, DeviceSpan<const real>(fields.u2_span()), baseline_u2.span());
    ctx.synchronize();

    // Rebuild the lambda=0.5 attempt's Y_att/K_att/flow -- the SAME
    // deterministic per-lambda construction ContinuationController.cu's
    // build_attempt uses (mirrored below for lambda=0.5125 exactly as
    // case_terminal_dgate_diagnostic does) -- so the S6 baseline diagnostics
    // compare against the CORRECT Darcy field that produced baseline_u1/u2.
    const std::size_t u_size = compact_mac_u_size(grid);
    const std::size_t v_size = compact_mac_v_size(grid);
    const std::size_t w_size = compact_mac_w_size(grid);

    DeviceBuffer<real> y_att0(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(y_att0.data(), y.data(), n * sizeof(real), cudaMemcpyDeviceToDevice,
                                           ctx.cuda_stream()));
    blas::scal(ctx, y_att0.span(), real{0.5});
    DeviceBuffer<real> k_att0(n);
    terminal_dgate_enqueue_exp(ctx, DeviceSpan<const real>(y_att0.span()), k_att0.span());
    DeviceBuffer<real> flow_u0(u_size), flow_v0(v_size), flow_w0(w_size);
    physics::AffinePeriodicFlowWorkspace flow_workspace0;
    physics::AffinePeriodicVelocityView velocity0{flow_u0.span(), flow_v0.span(), flow_w0.span()};
    (void)physics::solve_affine_periodic_flow(ctx, grid, DeviceSpan<const real>(k_att0.span()), flow_config,
                                              velocity0, flow_workspace0);
    ctx.synchronize();

    StreamfunctionProblemView problem_view0;
    problem_view0.grid = grid;
    problem_view0.conductivity = DeviceSpan<const real>(y_att0.span());
    problem_view0.conductivity_representation = ConductivityRepresentation::log_conductivity_y;
    problem_view0.darcy_velocity = CompactMacVelocityConstView{DeviceSpan<const real>(flow_u0.span()),
                                                                DeviceSpan<const real>(flow_v0.span()),
                                                                DeviceSpan<const real>(flow_w0.span())};
    problem_view0.bc = triply_periodic();
    problem_view0.gauge = AffineGauge::benchmark(real{1});

    // lambda=0.5125 attempt (identical to case_terminal_dgate_diagnostic).
    constexpr real kLambdaAttempt = real{0.5125};
    DeviceBuffer<real> y_att(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(y_att.data(), y.data(), n * sizeof(real), cudaMemcpyDeviceToDevice,
                                           ctx.cuda_stream()));
    blas::scal(ctx, y_att.span(), kLambdaAttempt);
    DeviceBuffer<real> k_att(n);
    terminal_dgate_enqueue_exp(ctx, DeviceSpan<const real>(y_att.span()), k_att.span());
    DeviceBuffer<real> flow_u(u_size), flow_v(v_size), flow_w(w_size);
    physics::AffinePeriodicFlowWorkspace flow_workspace;
    physics::AffinePeriodicVelocityView velocity{flow_u.span(), flow_v.span(), flow_w.span()};
    (void)physics::solve_affine_periodic_flow(ctx, grid, DeviceSpan<const real>(k_att.span()), flow_config,
                                              velocity, flow_workspace);
    ctx.synchronize();

    StreamfunctionProblemView problem_view;
    problem_view.grid = grid;
    problem_view.conductivity = DeviceSpan<const real>(y_att.span());
    problem_view.conductivity_representation = ConductivityRepresentation::log_conductivity_y;
    problem_view.darcy_velocity = CompactMacVelocityConstView{DeviceSpan<const real>(flow_u.span()),
                                                               DeviceSpan<const real>(flow_v.span()),
                                                               DeviceSpan<const real>(flow_w.span())};
    problem_view.bc = triply_periodic();
    problem_view.gauge = AffineGauge::benchmark(real{1});

    StreamfunctionSolverConfig stage_config = base_config;
    stage_config.eta = real{1};
    stage_config.epsilon = real{1e-2};
    stage_config.initial_state = PicardInitialState::warm_start;
    stage_config.coefficient_state = CoefficientState::rebuild;

    const StreamfunctionSolveReport stage_report =
        solve_streamfunctions(ctx, problem_view, stage_config, fields, workspace);
    ctx.synchronize();

    const double r_F_frozen = static_cast<double>(stage_report.residual.r_F);
    std::cout << "E2 stage: status=" << solve_status_label(stage_report.status)
              << " exit_reason=" << exit_reason_label(stage_report.exit_reason)
              << " picard_iterations=" << stage_report.picard_iterations << " r_F_frozen=" << r_F_frozen << '\n';

    check("E2_stage_not_converged", stage_report.status == StreamfunctionSolveStatus::not_converged);
    check("E2_stage_exit_reason_plateau_signature",
          stage_report.exit_reason == PicardExitReason::stagnated ||
              stage_report.exit_reason == PicardExitReason::omega_floor_rejected);
    const bool r_f_band_ok = r_F_frozen >= 1e-4 && r_F_frozen <= 1e-2;
    check("E2_stage_r_F_in_prespecified_band_1e-4_1e-2", r_f_band_ok);

    constexpr double kRecordedRFFrozen = 1.1204722529922055e-3;
    std::cout << "E2 warn-only exact compare: r_F_frozen=" << r_F_frozen << " recorded=" << kRecordedRFFrozen
              << " " << (r_F_frozen == kRecordedRFFrozen ? "MATCH" : "DIFFERS") << '\n';

    // S6 shelf column / S3micro base state: the frozen plateau iterate,
    // captured now (this IS the last mutation of `fields` before S3micro).
    DeviceBuffer<real> frozen_u1(n), frozen_u2(n);
    blas::copy(ctx, DeviceSpan<const real>(fields.u1_span()), frozen_u1.span());
    blas::copy(ctx, DeviceSpan<const real>(fields.u2_span()), frozen_u2.span());
    ctx.synchronize();

    const DeviceSpan<const real> q_att = workspace.q();
    const double v_rms = static_cast<double>(stage_report.diagnostics.v_d_rms);
    NonlinearSourceConfig source_config;
    source_config.epsilon = real{1e-2};
    source_config.v_rms = static_cast<real>(v_rms);
    const ResidualHistogramConfig histogram_config{};
    const AffineGauge gauge = AffineGauge::benchmark(real{1});

    JvpWorkspace jvp;
    jvp.prepare(n);
    jvp.prepare_jvp_base(ctx, grid, q_att, CoupledVectorView{fields.u1_span(), fields.u2_span()}, gauge, real{1},
                         source_config, histogram_config);

    StreamfunctionResidualWorkspace frozen_residual_ws;
    frozen_residual_ws.prepare(n);
    DeviceBuffer<real> f1(n), f2(n);
    enqueue_streamfunction_residual(ctx, grid, q_att,
                                    PeriodicStreamfunctionFluctuations{fields.u1_span(), fields.u2_span()}, gauge,
                                    real{1}, source_config, histogram_config, f1.span(), f2.span(),
                                    frozen_residual_ws);
    (void)synchronize_streamfunction_residual_report(ctx, grid, real{1}, source_config, histogram_config,
                                                      frozen_residual_ws);

    DeviceBuffer<real> b1(n), b2(n);
    blas::copy(ctx, DeviceSpan<const real>(f1.span()), b1.span());
    blas::copy(ctx, DeviceSpan<const real>(f2.span()), b2.span());
    blas::scal(ctx, b1.span(), real{-1});
    blas::scal(ctx, b2.span(), real{-1});
    ctx.synchronize();

    multigrid::MGConfig mg_config = base_config.mg;
    BlockDiagonalMGPreconditioner precond(workspace.hierarchy(), mg_config);

    // =========================================================================
    // S3micro Part A: single-step mu-scan, fresh from the frozen state each
    // time, exactly in the E6b pattern (full step, no line search).
    // =========================================================================
    struct MuScanResult {
        double mu;
        CoupledGmresStatus status;
        int inner;
        double r_F_candidate;
        double delta_rF;
    };
    std::vector<MuScanResult> mu_scan;
    const double mu_scan_list[] = {0.3, 0.4, 0.5, 0.7, 1.0, 2.0};
    const real s3_rel_tol = std::clamp(static_cast<real>(std::sqrt(r_F_frozen)), real{1e-8}, real{1e-1});

    StreamfunctionResidualWorkspace scan_residual_ws;
    scan_residual_ws.prepare(n);

    for (double mu_d : mu_scan_list) {
        const real mu = static_cast<real>(mu_d);
        ShiftedJacobianOperator op(jvp, grid, q_att, mu);
        op.prepare(n);
        CoupledGmres gmres;
        gmres.prepare(n, 10);

        DeviceBuffer<real> s1(n), s2(n);
        blas::fill(ctx, s1.span(), real{0});
        blas::fill(ctx, s2.span(), real{0});

        CoupledGmresConfig cfg;
        cfg.restart = 10;
        cfg.max_iterations = 100;
        cfg.rel_tol = s3_rel_tol;

        const CoupledGmresReport report = gmres.solve(
            ctx, grid, op, precond,
            ConstCoupledVectorView(DeviceSpan<const real>(b1.span()), DeviceSpan<const real>(b2.span())), cfg,
            JvpDeltaConfig{}, CoupledVectorView{s1.span(), s2.span()});
        ctx.synchronize();

        DeviceBuffer<real> cand1(n), cand2(n);
        blas::copy(ctx, DeviceSpan<const real>(frozen_u1.span()), cand1.span());
        blas::copy(ctx, DeviceSpan<const real>(frozen_u2.span()), cand2.span());
        blas::axpy(ctx, real{1}, DeviceSpan<const real>(s1.span()), cand1.span());
        blas::axpy(ctx, real{1}, DeviceSpan<const real>(s2.span()), cand2.span());

        DeviceBuffer<real> scan_f1(n), scan_f2(n);
        enqueue_streamfunction_residual(ctx, grid, q_att,
                                        PeriodicStreamfunctionFluctuations{cand1.span(), cand2.span()}, gauge,
                                        real{1}, source_config, histogram_config, scan_f1.span(), scan_f2.span(),
                                        scan_residual_ws);
        const StreamfunctionResidualReport cand_res = synchronize_streamfunction_residual_report(
            ctx, grid, real{1}, source_config, histogram_config, scan_residual_ws);
        const double r_F_candidate = static_cast<double>(cand_res.r_F);
        const double delta_rF = r_F_candidate - r_F_frozen;
        mu_scan.push_back({mu_d, report.status, report.total_inner_iterations, r_F_candidate, delta_rF});

        std::cout << "S3micro mu=" << mu_d << " gmres_status=" << gmres_status_label(report.status)
                  << " inner=" << report.total_inner_iterations << " r_F_candidate=" << r_F_candidate
                  << " delta_rF=" << delta_rF << '\n';
    }

    bool any_descent = false;
    double mu_best = 0.0, best_delta = 0.0;
    for (const auto& r : mu_scan) {
        if (r.delta_rF >= 0.0) continue;
        if (!any_descent) {
            any_descent = true;
            mu_best = r.mu;
            best_delta = r.delta_rF;
            continue;
        }
        if (r.delta_rF < best_delta || (r.delta_rF == best_delta && r.mu < mu_best)) {
            mu_best = r.mu;
            best_delta = r.delta_rF;
        }
    }
    std::cout << "S3micro selection: any_descent=" << (any_descent ? "true" : "false") << " mu_best=" << mu_best
              << " best_single_step_delta_rF=" << best_delta << '\n';

    // =========================================================================
    // S3micro Part B: N=100 consecutive full steps at fixed mu_best, from the
    // frozen state, unconditional acceptance, abort on nonfinite or
    // r_F>2*r_F_frozen. Every device buffer used inside the 100-step loop is
    // allocated ONCE, before the loop.
    // =========================================================================
    std::vector<double> sustained_rF;
    std::vector<double> sustained_delta;
    std::vector<int> sustained_inner;
    bool aborted = false;
    int abort_step = -1;
    double wall_seconds_total = 0.0;

    if (any_descent) {
        ShiftedJacobianOperator sustained_op(jvp, grid, q_att, static_cast<real>(mu_best));
        sustained_op.prepare(n);
        CoupledGmres sustained_gmres;
        sustained_gmres.prepare(n, 10);

        DeviceBuffer<real> state1(n), state2(n);
        blas::copy(ctx, DeviceSpan<const real>(frozen_u1.span()), state1.span());
        blas::copy(ctx, DeviceSpan<const real>(frozen_u2.span()), state2.span());
        ctx.synchronize();

        DeviceBuffer<real> step_s1(n), step_s2(n);
        DeviceBuffer<real> step_rhs1(n), step_rhs2(n);
        DeviceBuffer<real> step_f1(n), step_f2(n);
        blas::copy(ctx, DeviceSpan<const real>(f1.span()), step_f1.span());
        blas::copy(ctx, DeviceSpan<const real>(f2.span()), step_f2.span());
        StreamfunctionResidualWorkspace step_residual_ws;
        step_residual_ws.prepare(n);

        double r_F_k = r_F_frozen;
        sustained_rF.push_back(r_F_frozen);

        const auto t0 = std::chrono::steady_clock::now();
        for (int step = 0; step < 100; ++step) {
            jvp.prepare_jvp_base(ctx, grid, q_att, CoupledVectorView{state1.span(), state2.span()}, gauge, real{1},
                                 source_config, histogram_config);

            blas::copy(ctx, DeviceSpan<const real>(step_f1.span()), step_rhs1.span());
            blas::copy(ctx, DeviceSpan<const real>(step_f2.span()), step_rhs2.span());
            blas::scal(ctx, step_rhs1.span(), real{-1});
            blas::scal(ctx, step_rhs2.span(), real{-1});

            const real rel_tol = std::clamp(static_cast<real>(std::sqrt(r_F_k)), real{1e-8}, real{1e-1});
            CoupledGmresConfig step_cfg;
            step_cfg.restart = 10;
            step_cfg.max_iterations = 100;
            step_cfg.rel_tol = rel_tol;

            blas::fill(ctx, step_s1.span(), real{0});
            blas::fill(ctx, step_s2.span(), real{0});
            const CoupledGmresReport step_report = sustained_gmres.solve(
                ctx, grid, sustained_op, precond,
                ConstCoupledVectorView(DeviceSpan<const real>(step_rhs1.span()), DeviceSpan<const real>(step_rhs2.span())),
                step_cfg, JvpDeltaConfig{}, CoupledVectorView{step_s1.span(), step_s2.span()});
            ctx.synchronize();

            blas::axpy(ctx, real{1}, DeviceSpan<const real>(step_s1.span()), state1.span());
            blas::axpy(ctx, real{1}, DeviceSpan<const real>(step_s2.span()), state2.span());

            enqueue_streamfunction_residual(ctx, grid, q_att,
                                            PeriodicStreamfunctionFluctuations{state1.span(), state2.span()}, gauge,
                                            real{1}, source_config, histogram_config, step_f1.span(), step_f2.span(),
                                            step_residual_ws);
            const StreamfunctionResidualReport new_res = synchronize_streamfunction_residual_report(
                ctx, grid, real{1}, source_config, histogram_config, step_residual_ws);
            const double r_F_new = static_cast<double>(new_res.r_F);
            const bool finite_ok = std::isfinite(r_F_new);

            std::cout << "S3B k=" << step << " gmres_status=" << gmres_status_label(step_report.status)
                      << " inner=" << step_report.total_inner_iterations << " r_F=" << r_F_new
                      << " delta=" << (r_F_new - r_F_k) << '\n';

            if (!finite_ok || r_F_new > 2.0 * r_F_frozen) {
                std::cout << "S3B_ABORT step=" << step << " r_F=" << r_F_new << '\n';
                aborted = true;
                abort_step = step;
                break;
            }
            sustained_rF.push_back(r_F_new);
            sustained_delta.push_back(r_F_new - r_F_k);
            sustained_inner.push_back(step_report.total_inner_iterations);
            r_F_k = r_F_new;
        }
        const auto t1 = std::chrono::steady_clock::now();
        wall_seconds_total = std::chrono::duration<double>(t1 - t0).count();
    } else {
        std::cout << "PARTB_SKIPPED_NO_DESCENT\n";
    }
    if (aborted) {
        std::cout << "S3B abort_step=" << abort_step << '\n';
    }

    // Readouts (mechanical, print-only).
    if (any_descent && !sustained_rF.empty()) {
        const int completed = static_cast<int>(sustained_rF.size()) - 1;
        if (completed > 0) {
            const double rF0 = sustained_rF.front();
            const double rFN = sustained_rF.back();
            const double mean_sec_per_step = wall_seconds_total / static_cast<double>(completed);
            double rho = std::numeric_limits<double>::quiet_NaN();
            double steps_to_1e6 = std::numeric_limits<double>::quiet_NaN();
            double projected_wall_hours = std::numeric_limits<double>::quiet_NaN();
            if (rF0 > 0.0 && rFN > 0.0) {
                rho = (std::log(rF0) - std::log(rFN)) / static_cast<double>(completed);
            }
            bool projected_feasible = false;
            if (rho > 0.0) {
                steps_to_1e6 = std::log(r_F_frozen / 1e-6) / rho;
                projected_wall_hours = steps_to_1e6 * mean_sec_per_step / 3600.0;
                projected_feasible = steps_to_1e6 <= 1e6 && projected_wall_hours <= 4.0;
            }

            double mean_abs_delta = 0.0;
            for (double d : sustained_delta) mean_abs_delta += std::abs(d);
            mean_abs_delta /= static_cast<double>(sustained_delta.size());
            const bool curvature_limited = std::abs(best_delta) >= 3.0 * mean_abs_delta;

            bool flow_stalls = rho <= 0.0;
            if (!flow_stalls && static_cast<int>(sustained_delta.size()) >= 20) {
                double first20 = 0.0, last20 = 0.0;
                for (int i = 0; i < 20; ++i) first20 += std::abs(sustained_delta[i]);
                for (int i = static_cast<int>(sustained_delta.size()) - 20;
                     i < static_cast<int>(sustained_delta.size()); ++i)
                    last20 += std::abs(sustained_delta[i]);
                first20 /= 20.0;
                last20 /= 20.0;
                flow_stalls = last20 <= 0.1 * first20;
            } else if (!flow_stalls) {
                std::cout << "FLOW_STALLS_WINDOW_INSUFFICIENT completed=" << completed << " (<20)\n";
            }

            std::cout << "S3B wall_seconds_total=" << wall_seconds_total
                      << " mean_sec_per_step=" << mean_sec_per_step << " completed_steps=" << completed
                      << (aborted ? " ABORTED" : "") << '\n';
            std::cout << "S3B readout: rho=" << rho << " steps_to_1e-6=" << steps_to_1e6
                      << " projected_wall_hours=" << projected_wall_hours << '\n';
            std::cout << "S3B readout: best_single_step_delta_rF=" << best_delta
                      << " sustained_mean_abs_delta=" << mean_abs_delta << '\n';
            if (projected_feasible) std::cout << "PROJECTED_FEASIBLE\n";
            if (curvature_limited) std::cout << "CURVATURE_LIMITED\n";
            if (flow_stalls) std::cout << "FLOW_STALLS\n";
        }
    }

    // =========================================================================
    // S6 battery on BOTH the frozen shelf state (A) and the lambda=0.5
    // accepted predecessor baseline (B).
    // =========================================================================
    const PhysicalDiagnosticsConfig diag_config{}; // default thresholds.

    const S6StateEvidence shelf_evidence =
        evaluate_s6_state(ctx, grid, frozen_u1.span(), frozen_u2.span(), gauge, problem_view.darcy_velocity,
                          diag_config);
    const S6StateEvidence baseline_evidence =
        evaluate_s6_state(ctx, grid, baseline_u1.span(), baseline_u2.span(), gauge, problem_view0.darcy_velocity,
                          diag_config);

    const double epsilon_scale = static_cast<double>(source_config.epsilon) * v_rms;
    std::cout << "S6 shelf (lambda=0.5125 frozen plateau iterate):\n";
    print_s6_state_report("S6_shelf", shelf_evidence, epsilon_scale);
    std::cout << "S6 baseline (lambda=0.5 accepted predecessor):\n";
    print_s6_state_report("S6_baseline", baseline_evidence, epsilon_scale);

    const double shelf_top1_defect1 = host_top_fraction_energy(shelf_evidence.pointwise.invariance_defect1, 0.01);
    const double baseline_top1_defect1 =
        host_top_fraction_energy(baseline_evidence.pointwise.invariance_defect1, 0.01);
    const double shelf_top1_defect2 = host_top_fraction_energy(shelf_evidence.pointwise.invariance_defect2, 0.01);
    const double baseline_top1_defect2 =
        host_top_fraction_energy(baseline_evidence.pointwise.invariance_defect2, 0.01);

    std::cout << "S6 concentration_ratio invariance_defect1 shelf/baseline="
              << (baseline_top1_defect1 > 0.0 ? shelf_top1_defect1 / baseline_top1_defect1
                                              : std::numeric_limits<double>::quiet_NaN())
              << '\n';
    std::cout << "S6 concentration_ratio invariance_defect2 shelf/baseline="
              << (baseline_top1_defect2 > 0.0 ? shelf_top1_defect2 / baseline_top1_defect2
                                              : std::numeric_limits<double>::quiet_NaN())
              << '\n';

    const auto rubric = [&](const char* name, double shelf_top1, double baseline_top1) {
        const bool localized =
            shelf_top1 >= 0.50 && (baseline_top1 > 0.0 ? shelf_top1 >= 5.0 * baseline_top1 : shelf_top1 > 0.0);
        std::cout << "S6 rubric " << name << ": " << (localized ? "LOCALIZED_OBSTRUCTION" : "DIFFUSE")
                  << " (shelf_top1%=" << shelf_top1 << " baseline_top1%=" << baseline_top1 << ")\n";
    };
    rubric("invariance_defect1", shelf_top1_defect1, baseline_top1_defect1);
    rubric("invariance_defect2", shelf_top1_defect2, baseline_top1_defect2);

    std::cout << "case=terminal_shelf_probe_phase1 verdict=" << (pass ? "PASS" : "FAIL")
              << " (E2 freeze hard asserts gate `pass`; S3micro/S6 are print-only evidence)\n";

    std::ostringstream detail;
    detail << "ONE shared E2 freeze (VERBATIM from case_terminal_dgate_diagnostic, hygiene OFF): "
              "sigma_Y^2=1, 32^3, seed=12345, corr_length=8, lambda_attempt=0.5125, r_F_frozen="
           << r_F_frozen << "; S3micro Part A single-step mu-scan {0.3,0.4,0.5,0.7,1.0,2.0} "
              "(E6b pattern); Part B N<=100 sustained full steps at mu_best; S6 SF-11 report + "
              "host-replicated pointwise defect fields (percentiles, concentration) on the frozen "
              "shelf state and the lambda=0.5 accepted baseline";

    return {pass,
            "terminal_shelf_probe_phase1",
            "gpu-terminal-shelf-probe-phase1",
            detail.str(),
            r_F_frozen,
            mu_best,
            "E2_freeze_status/lambda/eta + E2_stage_not_converged/exit_reason/r_F band [1e-4,1e-2]",
            pass ? "all pass" : "some failed",
            "SF-25 Phase-1 (P1-I) S3micro+S6 probe (bitácora 2026-08-14T17:40Z): the E2 freeze hard "
            "asserts gate `pass`; the S3micro mu-scan/sustained-descent readouts (PROJECTED_FEASIBLE/"
            "CURVATURE_LIMITED/FLOW_STALLS) and the S6 LOCALIZED_OBSTRUCTION/DIFFUSE rubric are "
            "print-only evidence for the owner/orchestrator, not test gates"};
}

// ===========================================================================
// SF-25 Phase-2 (P2-I) Case: terminal_explicit_flow_probe (P2-A). PRESPECIFIED
// protocol recorded VERBATIM in the SF-25 bitácora at 2026-08-15T00:20Z:
// explicit pseudo-time flow u <- u - dtau*F(u) (forward Euler on the coupled
// projected residual) on the SAME 32^3, lambda=0.5125, eta=1 problem the E2
// freeze characterizes, discriminating "no flow-stable 32^3 solution" from
// "saddle-between-basins with a stable solution elsewhere". Two arms sharing
// ONE E2 freeze (armA1 = frozen shelf state; armA2 = zero fluctuations on the
// same coefficient state). Always-pass evidence recorder except the E2
// freeze hard asserts (identical gating discipline to
// `case_terminal_shelf_probe_phase1`).
//
// Rolling-residual cost structure (bitácora-mandated halving): a naive
// implementation would evaluate F twice per accepted step (once at u, once
// at the trial). Instead, `f_cur1_`/`f_cur2_` always hold F(u) for the
// CURRENT accepted state; each step's trial evaluation produces F(u_trial)
// into `f_trial1_`/`f_trial2_`, and on ACCEPT the two buffer pairs are
// exchanged via `DeviceBuffer::swap` (a pointer swap, not a device-to-device
// copy) so the just-computed F(u_trial) becomes F(u) for the NEXT step with
// zero extra evaluations. Net cost: ONE residual evaluation per accepted
// step, plus ONE per rejected retry (the seed evaluation of F(u0) before the
// loop starts is the only "extra" evaluation of the whole run).
// ===========================================================================

struct ExplicitFlowArmResult {
    long final_k{0};
    double final_tau{0.0};
    double final_dtau{0.0};
    double final_r_F{std::numeric_limits<double>::quiet_NaN()};
    long rejections{0};
    double min_r_F{std::numeric_limits<double>::infinity()};
    long min_at_k{0};
    double min_at_tau{0.0};
    double wall_s{0.0};
    std::string bound{"n/a"};
    bool attractor_found{false};
    bool attractor_verified{false};
    bool exceeded_2x_ref{false}; // r_F ever exceeded 2*saddle_reference_r_F (armA1 only, meaningful).
    double main_segment_slope{std::numeric_limits<double>::quiet_NaN()};   // final-<=10k-step mean d(ln r_F)/dtau.
    double coda_slope{std::numeric_limits<double>::quiet_NaN()};          // control-coda mean d(ln r_F)/dtau.
    bool growth_suspect{false};
    long band_samples{0};
    long band_hits{0};
    bool band_wandering{false};
};

// Runs ONE arm of the P2-A explicit pseudo-time flow, starting from
// `init_u1`/`init_u2` on the SHARED (q_att, gauge, source_config,
// histogram_config, darcy) coefficient state. Every device buffer used
// inside the step loop is allocated ONCE, before the loop (preallocation
// discipline). `saddle_reference_r_F` is `r_F_frozen` for armA1 (the
// SADDLE_ESCAPE_CONFIRMED reference) and NaN for armA2 (rule not
// applicable).
[[nodiscard]] ExplicitFlowArmResult run_explicit_flow_arm(
    CudaContext& ctx, const Grid3D& grid, DeviceSpan<const real> q_att, const AffineGauge& gauge,
    const NonlinearSourceConfig& source_config, const ResidualHistogramConfig& histogram_config,
    const CompactMacVelocityConstView& darcy, DeviceSpan<const real> init_u1, DeviceSpan<const real> init_u2,
    const char* arm_label, double saddle_reference_r_F) {
    const std::size_t n = grid.num_cells();
    ExplicitFlowArmResult result;

    DeviceBuffer<real> u1(n), u2(n), trial_u1(n), trial_u2(n);
    DeviceBuffer<real> f_cur1(n), f_cur2(n), f_trial1(n), f_trial2(n);
    DeviceBuffer<real> min_u1(n), min_u2(n);
    blas::copy(ctx, init_u1, u1.span());
    blas::copy(ctx, init_u2, u2.span());

    StreamfunctionResidualWorkspace residual_ws;
    residual_ws.prepare(n);

    // Seed F(u0), r_F(u0): the ONE evaluation that is NOT reused from a
    // trial (there is no prior trial yet).
    enqueue_streamfunction_residual(ctx, grid, q_att, PeriodicStreamfunctionFluctuations{u1.span(), u2.span()},
                                    gauge, real{1}, source_config, histogram_config, f_cur1.span(), f_cur2.span(),
                                    residual_ws);
    const StreamfunctionResidualReport init_report =
        synchronize_streamfunction_residual_report(ctx, grid, real{1}, source_config, histogram_config, residual_ws);

    double r_F_current = static_cast<double>(init_report.r_F);
    double max_r_F_seen = r_F_current;
    double tau = 0.0;
    double dtau = 1e-3;
    constexpr double kDtauMin = 1e-8;
    constexpr double kDtauMax = 1.0;
    long accept_streak = 0;
    long accepted = 0;
    long rejections = 0;

    blas::copy(ctx, DeviceSpan<const real>(u1.span()), min_u1.span());
    blas::copy(ctx, DeviceSpan<const real>(u2.span()), min_u2.span());
    double min_r_F = r_F_current;
    long min_at_k = 0;
    double min_at_tau = 0.0;

    std::cout << arm_label << " init r_F=" << r_F_current << '\n';

    int current_decade = (std::isfinite(r_F_current) && r_F_current > 0.0)
                              ? static_cast<int>(std::floor(std::log10(r_F_current)))
                              : 0;

    std::deque<std::pair<double, double>> history; // (tau, r_F), most recent <=10001 accepted steps.
    history.emplace_back(0.0, r_F_current);

    long band_samples = 0, band_hits = 0;

    // One step attempt: trial = u - dtau*F(u); evaluate F(trial); accept per
    // the PRESPECIFIED rule (reject iff nonfinite or > 2x growth). Returns
    // true iff accepted (state/F/tau/counters mutated in place); false iff
    // dtau underflowed `kDtauMin` (caller must treat as ABORT_DTAU_FLOOR).
    // `allow_growth` gates the streak-based dtau growth (disabled for the
    // fixed-step CONTROL coda, which must isolate the accuracy/stability
    // servo from the growth policy).
    const auto attempt_step = [&](bool allow_growth) -> bool {
        for (;;) {
            blas::copy(ctx, DeviceSpan<const real>(u1.span()), trial_u1.span());
            blas::copy(ctx, DeviceSpan<const real>(u2.span()), trial_u2.span());
            blas::axpy(ctx, static_cast<real>(-dtau), DeviceSpan<const real>(f_cur1.span()), trial_u1.span());
            blas::axpy(ctx, static_cast<real>(-dtau), DeviceSpan<const real>(f_cur2.span()), trial_u2.span());

            enqueue_streamfunction_residual(
                ctx, grid, q_att, PeriodicStreamfunctionFluctuations{trial_u1.span(), trial_u2.span()}, gauge,
                real{1}, source_config, histogram_config, f_trial1.span(), f_trial2.span(), residual_ws);
            const StreamfunctionResidualReport trial_report = synchronize_streamfunction_residual_report(
                ctx, grid, real{1}, source_config, histogram_config, residual_ws);
            const double r_F_trial = static_cast<double>(trial_report.r_F);

            const bool reject = !std::isfinite(r_F_trial) || r_F_trial > 2.0 * r_F_current;
            if (reject) {
                ++rejections;
                dtau /= 2.0;
                accept_streak = 0;
                if (dtau < kDtauMin) return false;
                continue;
            }

            u1.swap(trial_u1);
            u2.swap(trial_u2);
            f_cur1.swap(f_trial1);
            f_cur2.swap(f_trial2);
            r_F_current = r_F_trial;
            tau += dtau;
            ++accept_streak;
            if (allow_growth && accept_streak >= 50) {
                dtau = std::min(1.5 * dtau, kDtauMax);
                accept_streak = 0;
            }
            return true;
        }
    };

    constexpr long kNMax = 1000000;
    constexpr double kWallCapSeconds = 2.5 * 3600.0;

    const auto t0 = std::chrono::steady_clock::now();
    bool dtau_floor_hit = false;
    bool wall_hit = false;

    while (accepted < kNMax) {
        if (!attempt_step(/*allow_growth=*/true)) {
            dtau_floor_hit = true;
            std::cout << arm_label << " ABORT_DTAU_FLOOR k=" << accepted << " tau=" << tau << " dtau=" << dtau
                      << '\n';
            break;
        }
        ++accepted;
        max_r_F_seen = std::max(max_r_F_seen, r_F_current);

        const int new_decade = (std::isfinite(r_F_current) && r_F_current > 0.0)
                                    ? static_cast<int>(std::floor(std::log10(r_F_current)))
                                    : current_decade;
        if (new_decade != current_decade) {
            std::cout << arm_label << " DECADE r_F=" << r_F_current << " k=" << accepted << " tau=" << tau << '\n';
            current_decade = new_decade;
        }

        if (r_F_current < min_r_F) {
            min_r_F = r_F_current;
            min_at_k = accepted;
            min_at_tau = tau;
            blas::copy(ctx, DeviceSpan<const real>(u1.span()), min_u1.span());
            blas::copy(ctx, DeviceSpan<const real>(u2.span()), min_u2.span());
        }

        history.emplace_back(tau, r_F_current);
        if (history.size() > 10001) history.pop_front();

        if (accepted % 1000 == 0) {
            std::cout << arm_label << " k=" << accepted << " tau=" << tau << " dtau=" << dtau
                      << " r_F=" << r_F_current << " rejections=" << rejections << '\n';
            ++band_samples;
            if (r_F_current >= 1e-4 && r_F_current <= 1e-2) ++band_hits;

            const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (elapsed > kWallCapSeconds) {
                wall_hit = true;
                break;
            }
        }

        if (r_F_current <= 1e-6) {
            std::cout << arm_label << " ATTRACTOR_CANDIDATE k=" << accepted << " tau=" << tau << '\n';
            bool verified = true;
            for (int v = 0; v < 100; ++v) {
                if (!attempt_step(/*allow_growth=*/true)) {
                    verified = false;
                    break;
                }
                if (r_F_current > 2e-6) verified = false;
            }
            result.attractor_found = true;
            result.attractor_verified = verified;
            std::cout << arm_label << (verified ? " ATTRACTOR_FOUND VERIFIED" : " ATTRACTOR_FOUND")
                      << " k=" << accepted << " tau=" << tau << " r_F=" << r_F_current << '\n';
            break;
        }
    }

    const double main_final_dtau = dtau;
    const double main_final_tau = tau;
    const double main_final_r_F = r_F_current;

    // Main-run final-<=10k-accepted-step segment slope: mean d(ln r_F)/dtau
    // (per unit tau), an endpoint estimate over the retained history window
    // (matching the S3B `rho` endpoint-slope pattern used elsewhere in this
    // file for the same kind of "is it still descending" readout).
    double main_slope = std::numeric_limits<double>::quiet_NaN();
    if (history.size() >= 2) {
        const auto& seg_start = history.front();
        const auto& seg_end = history.back();
        const double dtau_seg = seg_end.first - seg_start.first;
        if (dtau_seg > 0.0 && seg_start.second > 0.0 && seg_end.second > 0.0) {
            main_slope = (std::log(seg_end.second) - std::log(seg_start.second)) / dtau_seg;
        }
    }

    // Step-refinement CONTROL coda: 1000 steps at dtau_final/10, FIXED (no
    // growth), same accept/reject rule, isolating whether the observed
    // trend is a numerical-stepsize artifact.
    double coda_slope = std::numeric_limits<double>::quiet_NaN();
    if (!dtau_floor_hit) {
        const double coda_dtau0 = std::max(main_final_dtau / 10.0, kDtauMin);
        dtau = coda_dtau0;
        const double coda_tau0 = tau;
        const double coda_rF0 = r_F_current;
        int coda_completed = 0;
        for (int c = 0; c < 1000; ++c) {
            if (!attempt_step(/*allow_growth=*/false)) {
                std::cout << arm_label << " CODA_ABORT_DTAU_FLOOR c=" << c << '\n';
                break;
            }
            ++coda_completed;
        }
        if (coda_completed > 0) {
            const double dtau_seg = tau - coda_tau0;
            if (dtau_seg > 0.0 && coda_rF0 > 0.0 && r_F_current > 0.0) {
                coda_slope = (std::log(r_F_current) - std::log(coda_rF0)) / dtau_seg;
            }
        }
        std::cout << arm_label << " coda completed=" << coda_completed << " dtau0=" << coda_dtau0
                  << " r_F_start=" << coda_rF0 << " r_F_end=" << r_F_current << '\n';
    } else {
        std::cout << arm_label << " coda: skipped (main run hit the dtau floor)\n";
    }

    bool growth_suspect = false;
    if (std::isfinite(main_slope) && std::isfinite(coda_slope)) {
        const bool sign_flip =
            ((main_slope > 0.0) != (coda_slope > 0.0)) && main_slope != 0.0 && coda_slope != 0.0;
        const double main_abs = std::abs(main_slope);
        const double coda_abs = std::abs(coda_slope);
        const bool magnitude_2x = main_abs > 0.0 && (coda_abs > 2.0 * main_abs || coda_abs < 0.5 * main_abs);
        growth_suspect = sign_flip || magnitude_2x;
    }
    std::cout << arm_label << " NUMERICAL_GROWTH_SUSPECT=" << (growth_suspect ? "true" : "false")
              << " main_segment_slope=" << main_slope << " coda_slope=" << coda_slope << '\n';

    // SF-11 PhysicalDiagnosticsReport AT THE ARGMIN STATE (one diagnostics
    // call on the preallocated snapshot pair, the shared enqueue/synchronize
    // pair used throughout this file).
    {
        StreamfunctionDiagnosticsWorkspace diag_ws;
        diag_ws.prepare(grid);
        DeviceBuffer<real> vpsi_u(compact_mac_u_size(grid)), vpsi_v(compact_mac_v_size(grid)),
            vpsi_w(compact_mac_w_size(grid));
        const PhysicalDiagnosticsConfig diag_config{};
        enqueue_streamfunction_physical_diagnostics(
            ctx, grid, PeriodicStreamfunctionFluctuations{min_u1.span(), min_u2.span()}, gauge, darcy, diag_config,
            CompactMacVelocityView{vpsi_u.span(), vpsi_v.span(), vpsi_w.span()}, diag_ws);
        const PhysicalDiagnosticsReport diag_report =
            synchronize_streamfunction_physical_diagnostics_report(ctx, grid, diag_config, diag_ws);
        std::cout << arm_label << " argmin SF11 e_v=" << diag_report.e_v
                  << " invariance_e_psi1=" << diag_report.invariance_e_psi1
                  << " invariance_e_psi2=" << diag_report.invariance_e_psi2 << " e_div=" << diag_report.e_div
                  << " c_min=" << diag_report.c_min << " c_max=" << diag_report.c_max
                  << " c_mean=" << diag_report.c_mean << " v_d_rms=" << diag_report.v_d_rms << '\n';
    }

    const char* bound_reason = "N_max";
    if (dtau_floor_hit) {
        bound_reason = "dtau_floor";
    } else if (wall_hit) {
        bound_reason = "wall";
    } else if (result.attractor_found) {
        // Not one of the three budget-exhaustion reasons in the PRESPECIFIED
        // enum: mission-accomplished early exit, recorded explicitly.
        bound_reason = "attractor";
    }

    result.final_k = accepted;
    result.final_tau = main_final_tau;
    result.final_dtau = main_final_dtau;
    result.final_r_F = main_final_r_F;
    result.rejections = rejections;
    result.min_r_F = min_r_F;
    result.min_at_k = min_at_k;
    result.min_at_tau = min_at_tau;
    result.wall_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    result.bound = bound_reason;
    result.exceeded_2x_ref = std::isfinite(saddle_reference_r_F) && saddle_reference_r_F > 0.0 &&
                             max_r_F_seen > 2.0 * saddle_reference_r_F;
    result.main_segment_slope = main_slope;
    result.coda_slope = coda_slope;
    result.growth_suspect = growth_suspect;
    result.band_samples = band_samples;
    result.band_hits = band_hits;
    result.band_wandering = band_samples > 0 && !result.attractor_found &&
                            static_cast<double>(band_hits) >= 0.5 * static_cast<double>(band_samples);

    std::cout << arm_label << " final k=" << result.final_k << " tau=" << result.final_tau
              << " r_F=" << result.final_r_F << " min_r_F=" << result.min_r_F << " min_at_k=" << result.min_at_k
              << " min_at_tau=" << result.min_at_tau << " rejections=" << result.rejections
              << " wall_s=" << result.wall_s << " bound=" << result.bound << '\n';

    return result;
}

[[nodiscard]] CaseResult case_terminal_explicit_flow_probe() {
    std::cout << std::setprecision(17);
    bool pass = true;
    const auto check = [&](const char* name, bool ok) {
        pass = pass && ok;
        std::cout << "  check " << name << "=" << (ok ? "PASS" : "FAIL") << '\n';
    };

    constexpr int n32 = 32;
    const Grid3D grid(n32, n32, n32, real{1}, real{1}, real{1});
    const std::size_t n = grid.num_cells();
    CudaContext ctx(0);

    // =========================================================================
    // E2 freeze, VERBATIM from case_terminal_shelf_probe_phase1 (hygiene OFF):
    // the sigma_Y^2=1, 32^3 smoke to its lambda floor, then the lambda=0.5125
    // warm-started stage to its plateau exit.
    // =========================================================================
    physics::PeriodicGaussianFieldConfig field_config;
    field_config.sigma2 = real{1};
    field_config.corr_length = real{8};
    field_config.seed = 12345ULL;
    field_config.normalize_variance = true;

    DeviceBuffer<real> y(n);
    physics::PeriodicGaussianFieldWorkspace field_workspace;
    (void)physics::generate_periodic_gaussian_field(ctx, grid, field_config, y.span(), field_workspace);
    ctx.synchronize();

    StreamfunctionFields fields;
    StreamfunctionWorkspace workspace;
    StreamfunctionSolverConfig base_config; // full defaults (adaptive Picard, newton DISABLED).
    base_config.anderson.enabled = true;
    base_config.anderson.depth = 5;
    base_config.anderson.start_iteration = 5;
    base_config.anderson.condition_limit = real{1e12};

    HeterogeneityContinuationConfig continuation_config{}; // lambda axis defaults.
    continuation_config.inner.epsilon_log10.target = continuation_config.inner.epsilon_log10.start;
    const physics::AffinePeriodicFlowConfig flow_config{}; // qbar=(1,0,0) default.

    const HeterogeneityContinuationReport freeze_report = run_streamfunction_heterogeneity_continuation(
        ctx, grid, DeviceSpan<const real>(y.span()), continuation_config, flow_config, base_config, fields,
        workspace);
    ctx.synchronize();

    std::cout << "E2 freeze: status=" << heterogeneity_status_label(freeze_report.status)
              << " final_lambda=" << freeze_report.final_lambda << " final_eta=" << freeze_report.final_eta
              << '\n';

    check("E2_freeze_status_lambda_floor_exhausted",
          freeze_report.status == HeterogeneityStatus::lambda_floor_exhausted);
    check("E2_freeze_final_lambda_eq_0_5", freeze_report.final_lambda == real{0.5});
    check("E2_freeze_final_eta_eq_1", freeze_report.final_eta == real{1});

    const std::size_t u_size = compact_mac_u_size(grid);
    const std::size_t v_size = compact_mac_v_size(grid);
    const std::size_t w_size = compact_mac_w_size(grid);

    constexpr real kLambdaAttempt = real{0.5125};
    DeviceBuffer<real> y_att(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(y_att.data(), y.data(), n * sizeof(real), cudaMemcpyDeviceToDevice,
                                           ctx.cuda_stream()));
    blas::scal(ctx, y_att.span(), kLambdaAttempt);
    DeviceBuffer<real> k_att(n);
    terminal_dgate_enqueue_exp(ctx, DeviceSpan<const real>(y_att.span()), k_att.span());
    DeviceBuffer<real> flow_u(u_size), flow_v(v_size), flow_w(w_size);
    physics::AffinePeriodicFlowWorkspace flow_workspace;
    physics::AffinePeriodicVelocityView velocity{flow_u.span(), flow_v.span(), flow_w.span()};
    (void)physics::solve_affine_periodic_flow(ctx, grid, DeviceSpan<const real>(k_att.span()), flow_config,
                                              velocity, flow_workspace);
    ctx.synchronize();

    StreamfunctionProblemView problem_view;
    problem_view.grid = grid;
    problem_view.conductivity = DeviceSpan<const real>(y_att.span());
    problem_view.conductivity_representation = ConductivityRepresentation::log_conductivity_y;
    problem_view.darcy_velocity = CompactMacVelocityConstView{DeviceSpan<const real>(flow_u.span()),
                                                               DeviceSpan<const real>(flow_v.span()),
                                                               DeviceSpan<const real>(flow_w.span())};
    problem_view.bc = triply_periodic();
    problem_view.gauge = AffineGauge::benchmark(real{1});

    StreamfunctionSolverConfig stage_config = base_config;
    stage_config.eta = real{1};
    stage_config.epsilon = real{1e-2};
    stage_config.initial_state = PicardInitialState::warm_start;
    stage_config.coefficient_state = CoefficientState::rebuild;

    const StreamfunctionSolveReport stage_report =
        solve_streamfunctions(ctx, problem_view, stage_config, fields, workspace);
    ctx.synchronize();

    const double r_F_frozen = static_cast<double>(stage_report.residual.r_F);
    std::cout << "E2 stage: status=" << solve_status_label(stage_report.status)
              << " exit_reason=" << exit_reason_label(stage_report.exit_reason)
              << " picard_iterations=" << stage_report.picard_iterations << " r_F_frozen=" << r_F_frozen << '\n';

    check("E2_stage_not_converged", stage_report.status == StreamfunctionSolveStatus::not_converged);
    check("E2_stage_exit_reason_plateau_signature",
          stage_report.exit_reason == PicardExitReason::stagnated ||
              stage_report.exit_reason == PicardExitReason::omega_floor_rejected);
    const bool r_f_band_ok = r_F_frozen >= 1e-4 && r_F_frozen <= 1e-2;
    check("E2_stage_r_F_in_prespecified_band_1e-4_1e-2", r_f_band_ok);

    constexpr double kRecordedRFFrozen = 1.1204722529922055e-3;
    std::cout << "E2 warn-only exact compare: r_F_frozen=" << r_F_frozen << " recorded=" << kRecordedRFFrozen
              << " " << (r_F_frozen == kRecordedRFFrozen ? "MATCH" : "DIFFERS") << '\n';

    // armA1 init: the frozen shelf state, deep-copied NOW (this IS the last
    // mutation of `fields` before the arms run).
    DeviceBuffer<real> frozen_u1(n), frozen_u2(n);
    blas::copy(ctx, DeviceSpan<const real>(fields.u1_span()), frozen_u1.span());
    blas::copy(ctx, DeviceSpan<const real>(fields.u2_span()), frozen_u2.span());
    ctx.synchronize();

    // Shared coefficient state for both arms (the lambda=0.5125, eta=1
    // problem the E2 freeze characterizes).
    const DeviceSpan<const real> q_att = workspace.q();
    const double v_rms = static_cast<double>(stage_report.diagnostics.v_d_rms);
    NonlinearSourceConfig source_config;
    source_config.epsilon = real{1e-2};
    source_config.v_rms = static_cast<real>(v_rms);
    const ResidualHistogramConfig histogram_config{};
    const AffineGauge gauge = AffineGauge::benchmark(real{1});

    // armA2 init: zero fluctuations on the SAME (q_att/gauge/source_config)
    // coefficient state (E8-style start).
    DeviceBuffer<real> zero_u1(n), zero_u2(n);
    blas::fill(ctx, zero_u1.span(), real{0});
    blas::fill(ctx, zero_u2.span(), real{0});
    ctx.synchronize();

    std::cout << "P2A shared coefficient state: v_rms=" << v_rms << " r_F_frozen=" << r_F_frozen << '\n';

    const ExplicitFlowArmResult armA1 = run_explicit_flow_arm(
        ctx, grid, q_att, gauge, source_config, histogram_config, problem_view.darcy_velocity,
        DeviceSpan<const real>(frozen_u1.span()), DeviceSpan<const real>(frozen_u2.span()), "P2A A1", r_F_frozen);

    const ExplicitFlowArmResult armA2 = run_explicit_flow_arm(
        ctx, grid, q_att, gauge, source_config, histogram_config, problem_view.darcy_velocity,
        DeviceSpan<const real>(zero_u1.span()), DeviceSpan<const real>(zero_u2.span()), "P2A A2",
        std::numeric_limits<double>::quiet_NaN());

    // Mechanical readouts, jointly and per-arm, per the PRESPECIFIED rules
    // (bitácora 2026-08-15T00:20Z). Raw operands printed alongside each.
    const bool saddle_escape_confirmed = armA1.exceeded_2x_ref && armA1.min_r_F < 1e-4;
    std::cout << "P2A rule SADDLE_ESCAPE_CONFIRMED: " << (saddle_escape_confirmed ? "true" : "false")
              << " (armA1_exceeded_2x_frozen=" << (armA1.exceeded_2x_ref ? "true" : "false")
              << " armA1_min_r_F=" << armA1.min_r_F << " r_F_frozen=" << r_F_frozen << ")\n";

    const bool armA1_no_stable =
        armA1.min_r_F > 1e-5 && std::isfinite(armA1.main_segment_slope) && armA1.main_segment_slope >= 0.0;
    const bool armA2_no_stable =
        armA2.min_r_F > 1e-5 && std::isfinite(armA2.main_segment_slope) && armA2.main_segment_slope >= 0.0;
    const bool no_stable_solution_at_horizon = armA1_no_stable && armA2_no_stable;
    std::cout << "P2A rule NO_STABLE_SOLUTION_AT_HORIZON: " << (no_stable_solution_at_horizon ? "true" : "false")
              << " (armA1_min_r_F=" << armA1.min_r_F << " armA1_final_segment_slope=" << armA1.main_segment_slope
              << " armA2_min_r_F=" << armA2.min_r_F << " armA2_final_segment_slope=" << armA2.main_segment_slope
              << ")\n";

    const bool armA1_horizon_descending =
        armA1.min_r_F > 1e-5 && std::isfinite(armA1.main_segment_slope) && armA1.main_segment_slope < 0.0;
    const bool armA2_horizon_descending =
        armA2.min_r_F > 1e-5 && std::isfinite(armA2.main_segment_slope) && armA2.main_segment_slope < 0.0;
    std::cout << "P2A rule HORIZON_BOUND_DESCENDING[A1]: " << (armA1_horizon_descending ? "true" : "false")
              << " (min_r_F=" << armA1.min_r_F << " final_segment_slope=" << armA1.main_segment_slope << ")\n";
    std::cout << "P2A rule HORIZON_BOUND_DESCENDING[A2]: " << (armA2_horizon_descending ? "true" : "false")
              << " (min_r_F=" << armA2.min_r_F << " final_segment_slope=" << armA2.main_segment_slope << ")\n";

    std::cout << "P2A rule ATTRACTOR_FOUND[A1]: "
              << (armA1.attractor_found ? (armA1.attractor_verified ? "true_VERIFIED" : "true") : "false")
              << " (min_r_F=" << armA1.min_r_F << ")\n";
    std::cout << "P2A rule ATTRACTOR_FOUND[A2]: "
              << (armA2.attractor_found ? (armA2.attractor_verified ? "true_VERIFIED" : "true") : "false")
              << " (min_r_F=" << armA2.min_r_F << ")\n";

    std::cout << "P2A rule BAND_WANDERING[A1]: " << (armA1.band_wandering ? "true" : "false")
              << " (band_hits=" << armA1.band_hits << "/" << armA1.band_samples
              << " attractor_found=" << (armA1.attractor_found ? "true" : "false") << ")\n";
    std::cout << "P2A rule BAND_WANDERING[A2]: " << (armA2.band_wandering ? "true" : "false")
              << " (band_hits=" << armA2.band_hits << "/" << armA2.band_samples
              << " attractor_found=" << (armA2.attractor_found ? "true" : "false") << ")\n";

    std::cout << "case=terminal_explicit_flow_probe verdict=" << (pass ? "PASS" : "FAIL")
              << " (E2 freeze hard asserts gate `pass`; the explicit-flow readouts above are "
                 "print-only evidence)\n";

    std::ostringstream detail;
    detail << "ONE shared E2 freeze (VERBATIM from case_terminal_shelf_probe_phase1, hygiene OFF): "
              "sigma_Y^2=1, 32^3, seed=12345, corr_length=8, lambda_attempt=0.5125, eta=1, r_F_frozen="
           << r_F_frozen << "; P2-A explicit pseudo-time flow u<-u-dtau*F(u), two arms (armA1=frozen "
              "shelf state, armA2=zero fluctuations), dtau_0=1e-3/dtau_min=1e-8/dtau_max=1.0, "
              "N_max=1e6 accepted steps / wall<=2.5h, rolling one-residual-eval-per-accepted-step "
              "structure, argmin snapshot + SF-11 diagnostics, step-refinement control coda, "
              "mechanical ATTRACTOR_FOUND/SADDLE_ESCAPE_CONFIRMED/NO_STABLE_SOLUTION_AT_HORIZON/"
              "HORIZON_BOUND_DESCENDING/BAND_WANDERING readouts";

    return {pass,
            "terminal_explicit_flow_probe",
            "gpu-terminal-explicit-flow-probe",
            detail.str(),
            armA1.min_r_F,
            armA2.min_r_F,
            "E2_freeze_status/lambda/eta + E2_stage_not_converged/exit_reason/r_F band [1e-4,1e-2]",
            pass ? "all pass" : "some failed",
            "SF-25 Phase-2 (P2-I) case P2-A (bitácora 2026-08-15T00:20Z): the E2 freeze hard asserts "
            "gate `pass`; the explicit pseudo-time flow readouts (ATTRACTOR_FOUND[_VERIFIED]/"
            "SADDLE_ESCAPE_CONFIRMED/NO_STABLE_SOLUTION_AT_HORIZON/HORIZON_BOUND_DESCENDING/"
            "BAND_WANDERING) are print-only evidence for the owner/orchestrator, not test gates"};
}

// ===========================================================================
// SF-25 Phase-2 (P2-I) Case: terminal_eta_endgame_64 (P2-C). The S1
// eta-endgame instrument (`run_eta_endgame_arm`, already generically
// parameterized by `problem_view`/`arm_label`/`hygiene_on`) at n=64, ell=16,
// dx=1 -- the SAME dimensionless problem as the 32^3/ell=8 fixture with h
// halved (the R1a fixture from `run_resolution_probe_solve`). Only the
// PROBLEM CONSTRUCTION needed extraction into a resolution-parameterized
// helper (`EtaEndgameProblem`/`build_eta_endgame_problem`, defined just
// above `case_terminal_eta_endgame` below, next to `run_eta_endgame_arm`,
// since that case's 32^3 call sites are refactored to call it with EXACTLY
// the previous constants, n=32/ell=8, so its behavior is unchanged: same
// construction order/values, same prints).
// ===========================================================================

[[nodiscard]] CaseResult case_terminal_eta_endgame_64() {
    std::cout << std::setprecision(17);

    std::cout << "terminal_eta_endgame_64 SF-18 caveat: statistically equivalent field, not the same "
                 "continuum realization\n";

    CudaContext ctx(0);
    EtaEndgameProblem problem64 = build_eta_endgame_problem(ctx, 64, real{16});
    const StreamfunctionProblemView& problem_view = problem64.view;

    const EtaEndgameArm s1c64 = run_eta_endgame_arm(ctx, problem_view, "S1c64", /*hygiene_on=*/true);

    // Bracket comparison vs the 32^3 S1b frontier eta (bracket [7.8125e-4,
    // 1.5625e-3), last accepted eta = 1 - 1.5625e-3 = 0.9984375).
    constexpr double kBracket32LastEta = 0.9984375;
    const char* bracket_verdict = "BRACKET_UNAVAILABLE";
    if (s1c64.last_valid) {
        if (s1c64.last_eta > kBracket32LastEta) {
            bracket_verdict = "BRACKET_MOVED_TOWARD_1";
        } else if (s1c64.last_eta == kBracket32LastEta) {
            bracket_verdict = "BRACKET_SAME";
        } else {
            bracket_verdict = "BRACKET_MOVED_AWAY";
        }
    }
    std::cout << "S1c64 rule bracket comparison vs 32^3 [7.8125e-4, 1.5625e-3): " << bracket_verdict
              << " (last_accepted_eta_64=" << (s1c64.last_valid ? s1c64.last_eta : -1.0)
              << " reference_last_accepted_eta_32=" << kBracket32LastEta << ")\n";

    std::cout << "case=terminal_eta_endgame_64 verdict=PASS (always-pass evidence recorder; see the "
                 "printed per-stage table, per-arm summary, verdict lines, and bracket comparison "
                 "above)\n";

    std::ostringstream detail;
    detail << "64^3 dx=1 seed=12345 corr_length=16 (ell/h=16, L/ell=4, the R1a fixture) sigma_Y^2=1 "
              "normalize_variance amplitude=0.5125*Y_unit K=exp log_conductivity_y SF-19 affine flow "
              "qbar=(1,0,0) benchmark gauge epsilon=1e-2 fixed anderson R5 newton disabled "
              "picard.max_iter=500; the SAME FIXED "
           << kEtaEndgameLadderSize
           << "-stage prespecified eta ladder, warm-started stage-to-stage; ONE arm (S1c64), hygiene "
              "ON with the S1b config (floor_guard{5,0.9,3} + anderson restart{2}); codas: eta=1 "
              "from last accepted state, and (when >=2 accepted stages had eta>=0.99) a two-point "
              "linear extrapolation to eta=1; SF-18 hash-mode realization caveat: statistically "
              "equivalent field, not the same continuum realization";

    return {true,
            "terminal_eta_endgame_64",
            "gpu-terminal-eta-endgame-64",
            detail.str(),
            s1c64.last_valid ? s1c64.last_eta : -1.0,
            kBracket32LastEta,
            bracket_verdict,
            "always pass (evidence recorder; see the printed per-stage table, per-arm summary, "
            "verdict lines, and bracket comparison above)",
            "SF-25 Phase-2 (P2-I) case P2-C (bitácora 2026-08-15T00:20Z): the S1 eta-endgame "
            "instrument at n=64, ell=16, dx=1 (the R1a fixture), ONE arm (hygiene ON, S1b config), "
            "identical ladder/stage-acceptance/per-stage budget/codas to the 32^3 case (identical "
            "iteration-budget-vs-floor exit_reason distinction, reused verbatim from "
            "`run_eta_endgame_arm`), plus a bracket comparison against the 32^3 S1b frontier eta; "
            "always-pass evidence recorder"};
}

} // namespace

CaseRegistry terminal_solver_case_registry() {
    return {{"terminal_shifted_apply_unit", case_terminal_shifted_apply_unit}};
}

CaseRegistry terminal_solver_dgate_case_registry() {
    return {{"terminal_dgate_diagnostic", case_terminal_dgate_diagnostic},
            {"terminal_resolution_probe", case_terminal_resolution_probe},
            {"terminal_floor_guard_continuation", case_terminal_floor_guard_continuation},
            {"terminal_eta_endgame", case_terminal_eta_endgame},
            {"terminal_shelf_probe_phase1", case_terminal_shelf_probe_phase1},
            {"terminal_explicit_flow_probe", case_terminal_explicit_flow_probe},
            {"terminal_eta_endgame_64", case_terminal_eta_endgame_64}};
}

} // namespace macroflow3d::streamfunctions::test
