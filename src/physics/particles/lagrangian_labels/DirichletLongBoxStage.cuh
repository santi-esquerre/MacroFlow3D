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
 * SEAM FOR N4 (spline phase): after `build_dirichlet_long_box_stage` returns, `host_lnk` (eps Y, host) and `f`
 * (device) are the two samples to be prefiltered by the non-periodic spline; N4 adds its own phase records
 * (S3/T) in its own code and does not touch the fields below.
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
#include "DirichletMedium.cuh"
#include "LongBoxStage.cuh"

#include <cstddef>
#include <string>
#include <vector>

namespace macroflow3d {
namespace lagrangian_labels {

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

    /// Potential fluctuation f (nx*ny*nz, device, layout i + nx (j + ny k)) after phase H.
    DeviceBuffer<real> f;
    /// eps * Y on the host (the N4 ln k samples).
    std::vector<real> host_lnk;
    /// Solved head H (host) iff keep_host_head.
    std::vector<real> host_head;
    /// CompactMAC faces (U (nx+1) ny nz, V nx (ny+1) nz, W nx ny (nz+1)) iff keep_host_faces.
    std::vector<real> face_u, face_v, face_w;
    /// Inlet face fluxes U[0, j, k], index j + ny k (host, always).
    std::vector<real> inlet_u_plane;

    LongBoxPhaseRecord phase_F, phase_S1, phase_S2, phase_V, phase_H;
    std::size_t device_used_start = 0;

    bool ready() const { return solver.usable && f.size() > 0; }

    /// `MEM_PHASE phase=<p> exact_bytes=<B> cufft_work=<W> [cufft_*=<w>] device_used=<U> seconds=<t>
    /// [categories=..] items=name:bytes,...` (the LongBoxStage format) for every recorded phase, in order.
    std::vector<std::string> mem_lines() const;
};

/// Exact bytes of phase S2 by the closed form of the file header. `scratch_bytes` = reduction scalar + CUB
/// temporary (the items `scratch_*` of the S2 record). `head_on_device` selects (6 + 1) n or 6 n.
std::size_t dirichlet_s2_closed_form_bytes(int nx, int ny, int nz, int mg_levels, bool head_on_device,
                                           std::size_t scratch_bytes);

/// Builds the stage. Solver non-convergence is reported (`solver`), not thrown; throws
/// std::invalid_argument for invalid arguments and std::runtime_error naming `phase=<F|S1|S2|V|H>` for
/// every failure inside a phase.
DirichletLongBoxStage build_dirichlet_long_box_stage(CudaContext& ctx, const DirichletLongBoxSpec& spec);

} // namespace lagrangian_labels
} // namespace macroflow3d
