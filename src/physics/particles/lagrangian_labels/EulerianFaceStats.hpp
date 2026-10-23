#pragma once

/**
 * @file EulerianFaceStats.hpp
 * @brief SF-38 E1 (RR5): Eulerian face statistics of CompactMAC velocity arrays, shared by medium (a)
 *        (SF-19 affine-periodic, with the periodic duplicate planes) and medium (b) (DirichletMedium).
 *
 * Arrays (FaceFluxRepack.cuh layout, x fastest, h = L1/nx = 1/ny):
 *   U (nx+1) ny nz : U[i + (nx+1)(j + ny k)]   x-face plane i at x = i h
 *   V nx (ny+1) nz : V[i + nx j + nx (ny+1) k] y-faces of cell layer i
 *   W nx ny (nz+1) : W[i + nx j + nx ny k]     z-faces of cell layer i
 *
 * Face sets (population statistics, divisor = face count, double, summed in index order, two-pass):
 *   x-face plane i (0..nx): the ny*nz faces U[i, j<ny, k<nz]  (the outlet plane U[nx] included).
 *   cell layer i (0..nx-1): V[i, j<ny, k<nz] and W[i, j<ny, k<nz]; the periodic duplicate planes
 *   V[., ny, .] and W[., ., nz] are NOT counted, so every face is counted once (ny*nz faces each).
 *   qbar = mean of U over plane 0.
 * Interior (closed intervals, 1e-9 tolerance in units of h to resolve exact-grid ties): x-face planes
 * with i h in [5 lambda, L1 - 5 lambda]; cell layers with (i + 1/2) h in the same interval.
 *   V1 = <var_i(v1)>_i / qbar^2, theta_2 = <mean_i(v2)>_i / qbar, theta_3 = <mean_i(v3)>_i / qbar,
 *   var_v2 = <var_i(v2)>_i / qbar^2, var_v3 = <var_i(v3)>_i / qbar^2 (<.>_i = unweighted average over
 *   the interior planes / layers, in index order).
 */

#include "../../../core/Scalar.hpp"

#include <vector>

namespace macroflow3d {
namespace lagrangian_labels {

struct EulerianFaceStatsResult {
    real qbar = 0.0;
    /// Raw (not normalized) profiles: u_* indexed by plane 0..nx, v_*/w_* by cell layer 0..nx-1.
    std::vector<real> u_mean, u_var, v_mean, v_var, w_mean, w_var;
    int plane_lo = 0, plane_hi = -1; ///< interior x-face planes (inclusive)
    int layer_lo = 0, layer_hi = -1; ///< interior cell layers (inclusive)
    real V1 = 0.0, theta_2 = 0.0, theta_3 = 0.0, var_v2 = 0.0, var_v3 = 0.0;
};

/// Throws std::invalid_argument on a wrong array size, non-positive extents, non-finite / non-positive
/// L1 or lambda, qbar == 0, or an empty interior (planes or layers).
EulerianFaceStatsResult eulerian_face_stats(const std::vector<real>& U, const std::vector<real>& V,
                                            const std::vector<real>& W, int nx, int ny, int nz,
                                            real L1, real lambda);

} // namespace lagrangian_labels
} // namespace macroflow3d
