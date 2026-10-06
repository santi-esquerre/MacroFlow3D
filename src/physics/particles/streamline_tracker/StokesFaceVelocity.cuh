#pragma once

/**
 * @file StokesFaceVelocity.cuh
 * @brief SF-32 Stokes face fluxes of the label velocity c = grad psi1 x grad psi2
 *        on a periodic MAC (Pollock) grid: exact edge line integrals, exactly
 *        divergence-free cell fluxes. GPU kernels with a host mirror.
 * @ingroup physics_particles
 *
 * SF-32 (Lester eq.(14) roadmap), DAG node N0. Authoritative contract: the
 * increment specification
 * `docs/plans/active/lester-eq14/increments/SF-32-reference-trackers-and-scalings.md`
 * and its orchestration record (section 3.2, periodic decomposition). This
 * module builds the velocity object of the conventional (Pollock / RT0)
 * tracker. It decides no physical claim.
 *
 * ---------------------------------------------------------------------------
 * 1) Equations (Lester et al. 2023, eqs. 32-33)
 * ---------------------------------------------------------------------------
 *
 *  Labels as in StreamlineTrackerCommon.cuh:
 *
 *    psi_i(x) = gbar_i . x_u + s_i(x),   c = grad psi1 x grad psi2,
 *
 *  s_i the SF-28 periodic tricubic B-splines. The flux of c through a face S
 *  is, by Stokes, the loop integral of psi1 grad psi2 . dl around dS (eq. 32).
 *  Each grid edge is integrated ONCE and shared, with opposite signs, by the
 *  faces that contain it, so the discrete divergence of every cell is an
 *  algebraic sum that cancels identically (eq. 33), whatever the quadrature.
 *
 *  Unwrapped-psi1 treatment (periodic decomposition). psi1, psi2 are NOT
 *  periodic (affine part), so a periodic edge array of
 *  int psi1 grad psi2 . dl would represent the edge at x_d = L by the edge at
 *  x_d = 0, which differ by a non-zero amount (not by a closed-loop zero).
 *  Instead c is expanded exactly as
 *
 *    c = gbar1 x gbar2 + grad s1 x (grad s2 + gbar2) + gbar1 x grad s2
 *      = gbar1 x gbar2 + curl[ s1 (grad s2 + gbar2) - s2 gbar1 ]
 *
 *  (curl(s1 (grad s2 + gbar2)) = grad s1 x (grad s2 + gbar2) because
 *  curl grad s2 = 0; curl(s2 gbar1) = grad s2 x gbar1), and Stokes is applied
 *  only to the PERIODIC vector field F = s1 (grad s2 + gbar2) - s2 gbar1:
 *
 *    face average = (gbar1 x gbar2) . n + area^-1 oint_dS F . dl.
 *
 *  Along an edge in direction d the integrand is
 *
 *    f_d(x) = s1(x) (d s2/d x_d (x) + gbar2[d]) - s2(x) gbar1[d],
 *
 *  a periodic function of position, so one periodic edge array per direction
 *  is exact and consistent across the periodic boundary.
 *
 * ---------------------------------------------------------------------------
 * 2) Grid and layout
 * ---------------------------------------------------------------------------
 *
 *  Pollock grid G_Delta: n_d cells per axis, spacing Delta_d, period
 *  L_d = n_d Delta_d (must equal the label periods), cell faces at i Delta_d.
 *  All six arrays have n_x n_y n_z entries in the project layout
 *  i + nx (j + ny k); all indices are periodic (mod n_d).
 *
 *    ex[i,j,k] = int f_x dx along the edge from (i Dx, j Dy, k Dz) in +x (length Dx)
 *    ey[i,j,k] = int f_y dy along the edge from (i Dx, j Dy, k Dz) in +y (length Dy)
 *    ez[i,j,k] = int f_z dz along the edge from (i Dx, j Dy, k Dz) in +z (length Dz)
 *
 *    u[i,j,k] = average of c . e_x over the x-face at x = i Dx of cell (i,j,k)
 *    v[i,j,k] = average of c . e_y over the y-face at y = j Dy of cell (i,j,k)
 *    w[i,j,k] = average of c . e_z over the z-face at z = k Dz of cell (i,j,k)
 *
 *  Face values are AVERAGES (flux / face area): the uniform pair
 *  (psi1 = x2, psi2 = x3) gives u = 1, v = w = 0 exactly.
 *
 * ---------------------------------------------------------------------------
 * 3) Orientation (right-handed about the face normal)
 * ---------------------------------------------------------------------------
 *
 *    u = cbar_x + ( ey[i,j,k] + ez[i,j+1,k] - ey[i,j,k+1] - ez[i,j,k] ) / (Dy Dz)
 *    v = cbar_y + ( ez[i,j,k] + ex[i,j,k+1] - ez[i+1,j,k] - ex[i,j,k] ) / (Dz Dx)
 *    w = cbar_z + ( ex[i,j,k] + ey[i+1,j,k] - ex[i,j+1,k] - ey[i,j,k] ) / (Dx Dy)
 *
 *  with cbar = gbar1 x gbar2. Per cell, every one of its 12 edges appears in
 *  exactly two of its faces with opposite signs in the flux form
 *  (u[i+1]-u[i]) Dy Dz + (v[j+1]-v[j]) Dz Dx + (w[k+1]-w[k]) Dx Dy, and the
 *  constant cbar cancels: the discrete divergence is zero up to roundoff.
 *  Orientation checks (SF-32 N0 self-checks): uniform pair u = 1, v = w = 0;
 *  s1 = a sin(2 pi x1), s2 = 0 gives v[i] = -(s1(x_{i+1}) - s1(x_i)) / Dx,
 *  the exact cell average of c2 = -d s1/d x1 for that spline.
 *
 * ---------------------------------------------------------------------------
 * 4) Quadrature exactness
 * ---------------------------------------------------------------------------
 *
 *  Restricted to an axis-parallel line, each s_i and d s2/d x_d is a
 *  piecewise polynomial of degree <= 3 whose breakpoints are the label knots
 *  (j + 1/2) h_d of that axis (h the LABEL spacing; Delta may differ:
 *  Delta = m h in the experiment, but nothing assumes alignment). Hence f_d is
 *  piecewise polynomial of degree <= 6 on the edge, with breaks at the knots
 *  strictly inside the edge. Each edge is split at those knots (computed from
 *  h generically) and every sub-interval is integrated with 4-point
 *  Gauss-Legendre (exact for degree <= 7; the nodes are interior, so no node
 *  ever sits on a breakpoint). Every edge integral is therefore the exact
 *  integral of the spline integrand up to roundoff, and every face value the
 *  exact face average of the spline velocity c.
 *
 * ---------------------------------------------------------------------------
 * 5) Determinism and memory
 * ---------------------------------------------------------------------------
 *
 *  GPU: one kernel launch with one thread per edge (direction from
 *  blockIdx.y), then one launch with one thread per cell writing its three
 *  faces. No atomics, no reductions, no printf, no synchronization: results
 *  are bitwise deterministic for fixed inputs on one platform. The host mirror
 *  calls the very same inline edge and face routines (host == device up to FMA
 *  contraction). The workspace owns six double arrays of n_x n_y n_z entries
 *  (three edge, three face), grown on demand and never shrunk; the report
 *  gives their exact capacity bytes. The kernels allocate nothing.
 *  No epsilon, clamp or regularization appears anywhere in this module (the
 *  only tolerance is the 1e-12 relative period-consistency check of the
 *  input validation).
 */

#include "../../../core/DeviceBuffer.cuh"
#include "../../../core/Grid3D.hpp"
#include "../../../core/Scalar.hpp"
#include "../../../numerics/interpolation/PeriodicTricubicBSpline.cuh"
#include "StreamlineTrackerCommon.cuh"

#include <cmath>
#include <cstddef>
#include <cuda_runtime.h>
#include <vector>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace streamline_tracker {

// ============================================================================
// Views and workspace
// ============================================================================

/// POD view of the three periodic MAC face-average arrays (device or host pointers).
struct PeriodicFaceFluxView {
    const real* u;
    const real* v;
    const real* w;
    int nx, ny, nz;
    real dx, dy, dz; ///< Delta per axis
    real Lx, Ly, Lz; ///< periods (= n * Delta)
};

/// Owned device buffers: edge integrals and face averages. Grown, never shrunk.
struct StokesFaceFluxWorkspace {
    DeviceBuffer<real> ex, ey, ez; ///< edge integrals (file header section 2)
    DeviceBuffer<real> u, v, w;    ///< face averages
    Grid3D grid;                   ///< Pollock grid of the last call

    StokesFaceFluxWorkspace() = default;

    /// Device view of the last computed faces. Throws std::logic_error if no
    /// compute_stokes_face_fluxes call has completed on this workspace.
    PeriodicFaceFluxView view() const;
};

struct StokesFaceFluxReport {
    size_t device_bytes;               ///< exact capacity bytes of the six workspace buffers
    int quadrature_nodes_per_interval; ///< Gauss-Legendre nodes per knot interval (4)
};

// ============================================================================
// Inline edge / face routines (host and device; file header sections 1-4)
// ============================================================================

namespace stokes_detail {

inline constexpr int kGaussNodes = 4;

/// 4-point Gauss-Legendre on [-1, 1]: nodes ascending, matching weights.
__host__ __device__ inline void gauss4(real xn[4], real wn[4]) {
    const real x0 = static_cast<real>(0.33998104358485626480);
    const real x1 = static_cast<real>(0.86113631159405257522);
    const real w0 = static_cast<real>(0.65214515486254614263);
    const real w1 = static_cast<real>(0.34785484513745385737);
    xn[0] = -x1;
    xn[1] = -x0;
    xn[2] = x0;
    xn[3] = x1;
    wn[0] = w1;
    wn[1] = w0;
    wn[2] = w0;
    wn[3] = w1;
}

/// Spacing of the label knots along axis d of a spline view.
__host__ __device__ inline real knot_spacing(const interpolation::PeriodicTricubicBSplineView& s,
                                             int d) {
    return d == 0 ? s.hx : (d == 1 ? s.hy : s.hz);
}

/// Edge integrand f_d at point p (file header section 1).
template <class LabelPairT>
__host__ __device__ inline real edge_integrand(const LabelPairT& labels, int d, const real p[3]) {
    real v1, g1x, g1y, g1z;
    real v2, g2x, g2y, g2z;
    interpolation::evaluate_point(labels.s1, p[0], p[1], p[2], v1, g1x, g1y, g1z);
    interpolation::evaluate_point(labels.s2, p[0], p[1], p[2], v2, g2x, g2y, g2z);
    const real ds2 = d == 0 ? g2x : (d == 1 ? g2y : g2z);
    return v1 * (ds2 + labels.gbar2[d]) - v2 * labels.gbar1[d];
}

/// 4-point Gauss integral of f_d over [a, b] along axis d (other coordinates of p fixed).
template <class LabelPairT>
__host__ __device__ inline real gauss_segment(const LabelPairT& labels, int d, real p[3], real a,
                                              real b) {
    real xn[4], wn[4];
    gauss4(xn, wn);
    const real mid = static_cast<real>(0.5) * (a + b);
    const real half = static_cast<real>(0.5) * (b - a);
    real acc = 0.0;
    for (int q = 0; q < kGaussNodes; ++q) {
        p[d] = mid + half * xn[q];
        acc += wn[q] * edge_integrand(labels, d, p);
    }
    return half * acc;
}

/**
 * @brief Exact integral of f_d along the edge starting at corner (x0, y0, z0)
 *        in direction +e_d with length len (file header section 4).
 *
 * LabelPairT must expose spline views s1, s2 and arrays gbar1[3], gbar2[3]
 * (SplineLabelPair; coefficients resident on the calling side). The edge is
 * split at the label knots (j + 1/2) h_d strictly inside (a, a + len).
 */
template <class LabelPairT>
__host__ __device__ inline real edge_integral(const LabelPairT& labels, int d, real x0, real y0,
                                              real z0, real len) {
    real p[3] = {x0, y0, z0};
    const real a = p[d];
    const real b = a + len;
    const real h = knot_spacing(labels.s1, d);
    // First knot index j with (j + 1/2) h > a (a knot equal to a is skipped below).
    real j = floor(a / h - static_cast<real>(0.5)) + static_cast<real>(1.0);
    real lo = a;
    real acc = 0.0;
    for (;;) {
        const real t = (j + static_cast<real>(0.5)) * h;
        if (!(t < b))
            break;
        if (t > lo) {
            acc += gauss_segment(labels, d, p, lo, t);
            lo = t;
        }
        j += static_cast<real>(1.0);
    }
    acc += gauss_segment(labels, d, p, lo, b);
    return acc;
}

/// Edge integral e_d[i,j,k] of direction d on the Pollock grid with spacings (dx, dy, dz).
template <class LabelPairT>
__host__ __device__ inline real grid_edge(const LabelPairT& labels, int d, int i, int j, int k,
                                          real dx, real dy, real dz) {
    const real len = d == 0 ? dx : (d == 1 ? dy : dz);
    return edge_integral(labels, d, static_cast<real>(i) * dx, static_cast<real>(j) * dy,
                         static_cast<real>(k) * dz, len);
}

/// Linear index i + nx (j + ny k).
__host__ __device__ inline size_t lin(int i, int j, int k, int nx, int ny) {
    return static_cast<size_t>(i) +
           static_cast<size_t>(nx) *
               (static_cast<size_t>(j) + static_cast<size_t>(ny) * static_cast<size_t>(k));
}

/// The three face averages of cell (i, j, k) from the periodic edge arrays
/// (file header section 3). cbar = gbar1 x gbar2.
__host__ __device__ inline void cell_faces(const real* ex, const real* ey, const real* ez, int nx,
                                           int ny, int nz, real dx, real dy, real dz,
                                           const real cbar[3], int i, int j, int k, real& u,
                                           real& v, real& w) {
    const int ip = (i + 1 == nx) ? 0 : i + 1;
    const int jp = (j + 1 == ny) ? 0 : j + 1;
    const int kp = (k + 1 == nz) ? 0 : k + 1;
    const size_t c0 = lin(i, j, k, nx, ny);
    u = cbar[0] +
        (ey[c0] + ez[lin(i, jp, k, nx, ny)] - ey[lin(i, j, kp, nx, ny)] - ez[c0]) / (dy * dz);
    v = cbar[1] +
        (ez[c0] + ex[lin(i, j, kp, nx, ny)] - ez[lin(ip, j, k, nx, ny)] - ex[c0]) / (dz * dx);
    w = cbar[2] +
        (ex[c0] + ey[lin(ip, j, k, nx, ny)] - ex[lin(i, jp, k, nx, ny)] - ey[c0]) / (dx * dy);
}

} // namespace stokes_detail

// ============================================================================
// Host API (StokesFaceVelocity.cu)
// ============================================================================

/**
 * @brief GPU: labels (device coefficients) -> workspace face averages on the
 *        Pollock grid `grid`.
 *
 * Throws std::invalid_argument (distinct messages) if a cell count is < 1, a
 * spacing is not finite or not > 0, a label period is not finite or not > 0,
 * a spline coefficient pointer is null, or a grid period differs from the
 * label period by more than 1e-12 relative. Launches on `stream`; synchronizes
 * nothing; allocates only through the workspace (grow-only).
 */
StokesFaceFluxReport compute_stokes_face_fluxes(cudaStream_t stream, const SplineLabelPair& labels,
                                                const Grid3D& grid, StokesFaceFluxWorkspace& ws);

/**
 * @brief Host mirror on host coefficients (labels built from make_host_view):
 *        same inline edge and face routines. u, v, w are resized to
 *        grid.num_cells(). Same validation as the GPU entry point.
 */
void compute_stokes_face_fluxes_host(const SplineLabelPair& labels, const Grid3D& grid,
                                     std::vector<real>& u, std::vector<real>& v,
                                     std::vector<real>& w);

/**
 * @brief Diagnostic on HOST arrays: max over cells of the flux-form discrete
 *        divergence |(u[i+1]-u[i]) Dy Dz + (v[j+1]-v[j]) Dz Dx + (w[k+1]-w[k]) Dx Dy|
 *        divided by (max |u|) Dy Dz; returns 0 if the latter is 0. Not a hot
 *        path. Throws std::invalid_argument for null pointers or a cell count < 1.
 */
real max_relative_divergence(const PeriodicFaceFluxView& host_view);

} // namespace streamline_tracker
} // namespace particles
} // namespace physics
} // namespace macroflow3d
