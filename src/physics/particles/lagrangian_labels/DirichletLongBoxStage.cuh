#pragma once

/**
 * @file DirichletLongBoxStage.cuh
 * @brief SF-41 N2 (contract C-STAGE, FLOW PART): lean medium (b) stage of the Lagrangian-label path.
 *        The same medium as SF-38 `DirichletMedium` (K = exp(eps Y) built on the host, `solve_head`'s operator
 *        and PCG + MG preconditioner, Dirichlet G1 L1 / 0 in x, periodic y/z, velocities by
 *        `compute_velocity_from_head`), with the buffer placement of the SF-40 lean stage so that the
 *        production grids fit:
 *
 *   F  (gaussian only) SF-18 call identical to LongBoxStage phase F; Y downloaded, device freed.
 *   S1 host lnK = eps Y (kept for N4) and K = exp(lnK) (the expression of DirichletMedium / LongBoxStage S1).
 *   S2 lean solve. K is uploaded straight into the MG level-0 coefficient (then the geometric coarsening loop
 *      of `populate_coefficient_hierarchy`); `VarCoeffLaplacian` is built on that buffer; the right-hand side
 *      (`build_rhs_head`) is assembled into the dead level-0 residual buffer; `z`, `Ap` and the level-0 `b`
 *      are one buffer (`solvers::pcg_solve_external`, `pcg_solve`'s arithmetic and order); the head (PCG `x`)
 *      lives on the device or, by option, in pinned mapped host memory. No `b_hat`, no `rhs` copy.
 *      Exact bytes (closed form, `dirichlet_s2_closed_form_bytes`), n_l = cells of MG level l:
 *        8 [ (6 + [head on device]) n + 4 sum_{l>=1} n_l ] + scratch,
 *      scratch = reduction scalar + CUB temporary (60.571 B/cell at 7 levels with the head on the device).
 *      Afterwards everything but the level-0 coefficient (K) is freed BEFORE the velocity faces exist.
 *   V  velocity faces (`compute_velocity_from_head`, reads K and the head) and the DirichletMedium
 *      diagnostics verbatim (Q_in, Q_out, qbar, balance, max_div); host faces by option; faces and K freed.
 *      With the head in mapped host memory it is copied here into a device array (the velocity API takes a
 *      device `HeadField`) and the mapped host memory is released.
 *   H  the head is turned IN PLACE into the potential fluctuation
 *        f_c = ((-H_c) - G1 x_i) + G1 L1,   x_i = (i + 1/2) h,   every operation rounded (no FMA),
 *      kept on the device (`f`).
 *
 *   S3 (SF-41 N4, `build_splines`, default on) the two samples are represented by the non-periodic-in-x1 spline
 *      (`NonPeriodicXTricubicBSpline`, contract C-SPLINE; same geometry: stage grid, h = 1/ny, L1 = nx h):
 *        S3a prefilter of f (the device samples already there) -> coefficients `pot` (the workspace
 *            coefficient buffer is MOVED out); f freed.
 *        S3b upload of host_lnk, prefilter -> coefficients `logk` (second buffer of the same workspace, which
 *            keeps its spectrum and pivot table); the samples are freed, then the workspace.
 *        S3c (fused only) device interleave of {pot, logk} into one `double2` array (pure data movement, bitwise
 *            exact); pot and logk freed afterwards.
 *      Coefficient layout: x-extended `(nx+4) ny nz`, `coef[(a + 2) + (nx+4)(j + ny k)]`, a = -2..nx+1 (N1).
 *      With ne = (nx+4) ny nz, sp = (nx+4)(ny/2+1) nz, W = cuFFT work of the two batched plans of N1:
 *        S3a = 8 n + 8 ne + 16 sp + 8 (nx-4) + W          (f, coefficients, spectrum, pivot table, work)
 *        S3b = 8 ne + 8 n + 8 ne + 16 sp + 8 (nx-4) + W   (pot, ln k samples, logk, spectrum, table, work)
 *        S3c = 8 ne + 8 ne + 16 ne = 32 ne                (pot, logk, interleaved; fused only)
 *        S3  = max(S3a, S3b [, S3c])  (`dirichlet_s3_closed_form_bytes`); the record holds the items of the
 *        peak sub-step and `categories` lists the sub-step values and the peak.
 *      f and host_lnk are released at the end of S3 (the tracker needs only the coefficients).
 *      `dirichlet_long_box_stage_memory_model` predicts every phase for any grid without allocating them.
 *
 * Bitwise rule: for the same host Y, eps, G1, grid, mg_levels, cg_rtol and cg_max_iter, the solved head (device
 * or mapped host) equals `DirichletMedium`'s head byte for byte and the `B_SOLVER` fields are identical: this
 * stage re-arranges lifetimes and buffer placement, never an arithmetic operation.
 *
 * Geometry: box [0, nx h] x [0, 1]^2 (ny == nz), h = 1/ny, cells x fastest (i + nx (j + ny k)).
 */

#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/Grid3D.hpp"
#include "../../../core/Scalar.hpp"
#include "../../../runtime/CudaContext.cuh"
#include "../../../runtime/cuda_check.cuh"
#include "../../stochastic/PeriodicGaussianField.cuh"
#include "DirichletMedium.cuh"
#include "LongBoxStage.cuh"

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace macroflow3d {
namespace lagrangian_labels {

/// Debug-only solve trace (SF-41 N2c): FNV-1a hashes of device buffers and of the output of every operator /
/// preconditioner application, to localize the first divergence between two solves. Host syncs and D2H copies at
/// every entry: never in production.
namespace debug {
using Trace = std::vector<std::pair<std::string, unsigned long long>>;

inline unsigned long long fnv1a_device(const real* d, std::size_t n) {
    std::vector<real> h(n);
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(real), cudaMemcpyDeviceToHost));
    unsigned long long x = 1469598103934665603ULL;
    const unsigned char* b = reinterpret_cast<const unsigned char*>(h.data());
    for (std::size_t i = 0; i < n * sizeof(real); ++i)
        x = (x ^ b[i]) * 1099511628211ULL;
    return x;
}

/// Wraps an operator / preconditioner (`apply(ctx, in, out)`): the arithmetic is the wrapped one's.
template <class Op> struct TracedOp {
    const Op& op;
    Trace* log;
    const char* tag;
    mutable int count;
    void apply(CudaContext& ctx, DeviceSpan<const real> x, DeviceSpan<real> y) const {
        op.apply(ctx, x, y);
        log->emplace_back(std::string(tag) + std::to_string(count++), fnv1a_device(y.data(), y.size()));
    }
};
} // namespace debug

struct DirichletLongBoxSpec {
    int nx = 0, ny = 0, nz = 0; ///< ny == nz; divisible by 2^(mg_levels-1), coarsest extents >= 2
    LongBoxSourceKind source = LongBoxSourceKind::gaussian;
    // gaussian (SF-18, box units), identical call to LongBoxStage phase F
    real sigma2 = 1.0;
    real ell = 0.0;
    unsigned long long seed = 0ULL;
    bool normalize_variance = true;
    // cells: host unit-amplitude Y, x1 fastest, nx*ny*nz values (read in place, not copied)
    std::vector<real> cells;
    real eps = 1.0;                 ///< K = exp(eps * Y)
    real G1 = 1.0;                  ///< west head = G1 L1, east head = 0 (G1 > 0)
    int mg_levels = 0;              ///< 0: box_auto_mg_levels
    real cg_rtol = 1e-10;
    int cg_max_iter = 2000;         ///< as the A/B application
    bool head_mapped_host = false;  ///< pre-authorized fallback: PCG x in pinned mapped host memory (bitwise)
    bool keep_host_faces = false;   ///< host CompactMAC U V W (Pollock bridge / flow statistics)
    bool keep_host_head = false;    ///< host copy of the solved head H before it becomes f (tests)
    debug::Trace* debug_trace = nullptr; ///< non-owning; debug hashes of the solve (host syncs), see debug::
    bool build_splines = true;      ///< phase S3; false: f (device) and host_lnk are kept as the N2 outputs
    bool fused_coefficients = false; ///< S3: one interleaved double2 {pot, logk} array instead of two arrays
};

/// The `B_SOLVER` quantities of the A/B application (lagrangian_medium_ab_main.cu) and its verdict rule.
struct DirichletSolverRecord {
    bool converged = false;
    int iterations = 0;
    real initial = 0.0, final = 0.0, rel = 0.0; ///< rel = initial > 0 ? final / initial : 0
    real qbar = 0.0, Q_in = 0.0, Q_out = 0.0, balance = 0.0, max_div = 0.0;
    bool fallback = false; ///< not converged but rel <= 1e-8 and balance <= 1e-8 (usable)
    bool usable = false;
    std::string verdict;   ///< converged | converged_balance_above_1e-8 | fallback | non_finite |
                           ///< not_converged_beyond_fallback
};

/// Applies the A/B application's rule (finite_ok; converged -> balance <= 1e-8; else the 1e-8 fallback).
DirichletSolverRecord make_dirichlet_solver_record(bool converged, int iterations, real initial,
                                                   real final_res, real qbar, real Q_in, real Q_out,
                                                   real balance, real max_div);
DirichletSolverRecord make_dirichlet_solver_record(const DirichletMediumReport& r);

/// `B_SOLVER r=<r> converged=.. iterations=.. initial=%.6e final=%.6e rel=%.6e qbar=%.12e Q_in=%.12e
/// Q_out=%.12e balance=%.6e max_div=%.6e fallback=..`: the A/B application's line, same printf formats.
std::string format_b_solver(int r, const DirichletSolverRecord& s);

class DirichletLongBoxStage {
  public:
    DirichletSolverRecord solver;
    Grid3D grid;
    real h = 0.0, L1 = 0.0, G1 = 1.0, head_drop = 0.0;
    int mg_levels = 0;
    bool head_mapped_host = false;

    /// SF-18 report of phase F (source gaussian only; value-initialized otherwise). SF-41 N5 (runner field.json).
    physics::PeriodicGaussianFieldReport sf18{};

    /// Potential fluctuation f (nx*ny*nz, device, layout i + nx (j + ny k)) after phase H; freed by S3.
    DeviceBuffer<real> f;
    /// eps * Y on the host (the ln k samples); released by S3.
    std::vector<real> host_lnk;
    /// Mean potential gradient of the stage, G = {G1, 0, 0} (for the tracker).
    real G[3] = {0.0, 0.0, 0.0};
    /// S3 outputs, x-extended non-periodic layout ((nx+4) ny nz entries, see above): coef_pot / coef_logk, or
    /// coef_interleaved (double2 {pot, logk}) iff fused_coefficients. coefficients_nonperiodic_x tells the
    /// tracker side that they are NOT the SF-28 periodic layout.
    DeviceBuffer<real> coef_pot, coef_logk;
    DeviceBuffer<double2> coef_interleaved;
    bool fused_coefficients = false;
    bool coefficients_nonperiodic_x = false;
    std::size_t coefficient_count() const { return static_cast<std::size_t>(grid.nx + 4) * grid.ny * grid.nz; }
    /// Solved head H (host) iff keep_host_head.
    std::vector<real> host_head;
    /// CompactMAC faces (U (nx+1) ny nz, V nx (ny+1) nz, W nx ny (nz+1)) iff keep_host_faces.
    std::vector<real> face_u, face_v, face_w;
    /// Inlet face fluxes U[0, j, k], index j + ny k (host, always).
    std::vector<real> inlet_u_plane;

    LongBoxPhaseRecord phase_F, phase_S1, phase_S2, phase_V, phase_H, phase_S3;
    std::size_t device_used_start = 0;

    bool ready() const {
        return solver.usable &&
               (!coefficients_nonperiodic_x
                    ? f.size() > 0
                    : (fused_coefficients ? coef_interleaved.size() > 0
                                          : coef_pot.size() > 0 && coef_logk.size() > 0));
    }

    /// `MEM_PHASE phase=<p> exact_bytes=<B> cufft_work=<W> [cufft_*=<w>] device_used=<U> seconds=<t>
    /// [categories=..] items=name:bytes,...` (the LongBoxStage format) for every recorded phase, in order.
    std::vector<std::string> mem_lines() const;
    /// SF-41 N5: `mem_lines()` plus, if `tracker` is a prepared tracker and the stage is ready, the T line of
    /// LongBoxStage's `mem_lines` (coefficient arrays + `tracker_<name>` buffers; device_used sampled now).
    std::vector<std::string> mem_lines(const LagrangianTracker* tracker, double tracker_prepare_seconds) const;
};

/// Exact bytes of phase S2 by the closed form of the file header. `scratch_bytes` = reduction scalar + CUB
/// temporary (the items `scratch_*` of the S2 record). `head_on_device` selects (6 + 1) n or 6 n.
std::size_t dirichlet_s2_closed_form_bytes(int nx, int ny, int nz, int mg_levels, bool head_on_device,
                                           std::size_t scratch_bytes);

/// Exact bytes of phase S3 by the closed form of the file header; `cufft_work` = W (the report's
/// `cufft_work_area_bytes` of one N1 prefilter call). `sub` (optional) receives {S3a, S3b, S3c}.
std::size_t dirichlet_s3_closed_form_bytes(int nx, int ny, int nz, std::size_t cufft_work, bool fused,
                                           std::size_t sub[3] = nullptr);

/// Predicted exact device bytes of every phase for any grid (nothing of that size is allocated; cuFFT work
/// sizes by `cufftGetSizeMany` / `cufftGetSize3d` on handles that never make a plan). Needs a CUDA device.
struct DirichletStageMemoryModel {
    int nx = 0, ny = 0, nz = 0, mg_levels = 0;
    bool head_on_device = true;
    std::size_t F = 0, S1 = 0, S2 = 0, V = 0, H = 0, S3a = 0, S3b = 0, S3c = 0;
    std::size_t cufft_f = 0;         ///< SF-18 3-D Z2D work (phase F)
    std::size_t cufft_prefilter = 0; ///< W: the two batched N1 plans (phase S3)
    std::size_t S3(bool fused) const { return std::max(std::max(S3a, S3b), fused ? S3c : std::size_t(0)); }
    /// max over F, S2, V, H, S3 (device bytes of arrays and cuFFT work; excludes the CUDA context)
    std::size_t max_phase(bool fused) const {
        return std::max(std::max(std::max(F, S2), std::max(V, H)), S3(fused));
    }
};
DirichletStageMemoryModel dirichlet_long_box_stage_memory_model(int nx, int ny, int nz, int mg_levels,
                                                                bool head_on_device);

/// Builds the stage. Solver non-convergence is reported (`solver`), not thrown; throws
/// std::invalid_argument for invalid arguments and std::runtime_error naming `phase=<F|S1|S2|V|H>` for
/// every failure inside a phase.
DirichletLongBoxStage build_dirichlet_long_box_stage(CudaContext& ctx, const DirichletLongBoxSpec& spec);

} // namespace lagrangian_labels
} // namespace macroflow3d
