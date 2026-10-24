/**
 * @file EulerianFaceStats.cpp
 * @brief SF-38 E1 face statistics. See EulerianFaceStats.hpp.
 */

#include "EulerianFaceStats.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace macroflow3d {
namespace lagrangian_labels {

namespace {

/// mean and population variance of f(m), m = 0..cnt-1 (index order, two-pass).
template <class F> void mean_var(std::size_t cnt, F f, real& mean, real& var) {
    real s = 0.0;
    for (std::size_t m = 0; m < cnt; ++m)
        s += f(m);
    mean = s / static_cast<real>(cnt);
    real q = 0.0;
    for (std::size_t m = 0; m < cnt; ++m) {
        const real d = f(m) - mean;
        q += d * d;
    }
    var = q / static_cast<real>(cnt);
}

} // namespace

EulerianFaceStatsResult eulerian_face_stats(const std::vector<real>& U, const std::vector<real>& V,
                                            const std::vector<real>& W, int nx, int ny, int nz,
                                            real L1, real lambda) {
    if (nx < 1 || ny < 1 || nz < 1)
        throw std::invalid_argument("eulerian_face_stats: nx, ny, nz must be >= 1");
    const std::size_t sx = nx, sy = ny, sz = nz;
    if (U.size() != (sx + 1) * sy * sz || V.size() != sx * (sy + 1) * sz ||
        W.size() != sx * sy * (sz + 1))
        throw std::invalid_argument(
            "eulerian_face_stats: array sizes are not CompactMAC (nx, ny, nz)");
    if (!(std::isfinite(L1) && L1 > 0.0 && std::isfinite(lambda) && lambda > 0.0))
        throw std::invalid_argument("eulerian_face_stats: L1 and lambda must be finite and > 0");

    EulerianFaceStatsResult R;
    const std::size_t nf = sy * sz; // faces per plane / layer, face index m = j + ny k
    R.u_mean.resize(sx + 1);
    R.u_var.resize(sx + 1);
    R.v_mean.resize(sx);
    R.v_var.resize(sx);
    R.w_mean.resize(sx);
    R.w_var.resize(sx);
    for (std::size_t i = 0; i <= sx; ++i)
        mean_var(nf, [&](std::size_t m) { return U[i + (sx + 1) * m]; }, R.u_mean[i], R.u_var[i]);
    for (std::size_t i = 0; i < sx; ++i) {
        mean_var(nf,
                 [&](std::size_t m) {
                     const std::size_t j = m % sy, k = m / sy;
                     return V[i + sx * (j + (sy + 1) * k)];
                 },
                 R.v_mean[i], R.v_var[i]);
        mean_var(nf, [&](std::size_t m) { return W[i + sx * m]; }, R.w_mean[i], R.w_var[i]);
    }
    R.qbar = R.u_mean[0];
    if (!(R.qbar != 0.0) || !std::isfinite(R.qbar))
        throw std::invalid_argument(
            "eulerian_face_stats: qbar (mean of U over plane 0) is 0 or non-finite");

    const real h = L1 / static_cast<real>(nx);
    const real a = 5.0 * lambda / h, b = (L1 - 5.0 * lambda) / h; // interval in units of h
    const real tol = 1e-9;
    long plo = static_cast<long>(std::ceil(a - tol)), phi = static_cast<long>(std::floor(b + tol));
    long llo = static_cast<long>(std::ceil(a - 0.5 - tol));
    long lhi = static_cast<long>(std::floor(b - 0.5 + tol));
    if (plo < 0)
        plo = 0;
    if (phi > nx)
        phi = nx;
    if (llo < 0)
        llo = 0;
    if (lhi > nx - 1)
        lhi = nx - 1;
    if (plo > phi || llo > lhi)
        throw std::invalid_argument("eulerian_face_stats: empty interior [5 lambda, L1 - 5 lambda]");
    R.plane_lo = static_cast<int>(plo);
    R.plane_hi = static_cast<int>(phi);
    R.layer_lo = static_cast<int>(llo);
    R.layer_hi = static_cast<int>(lhi);

    real su = 0.0;
    for (long i = plo; i <= phi; ++i)
        su += R.u_var[i];
    R.V1 = su / static_cast<real>(phi - plo + 1) / (R.qbar * R.qbar);
    real sv = 0.0, sw = 0.0, sv2 = 0.0, sw2 = 0.0;
    for (long i = llo; i <= lhi; ++i) {
        sv += R.v_mean[i];
        sw += R.w_mean[i];
        sv2 += R.v_var[i];
        sw2 += R.w_var[i];
    }
    const real nl = static_cast<real>(lhi - llo + 1);
    R.theta_2 = sv / nl / R.qbar;
    R.theta_3 = sw / nl / R.qbar;
    R.var_v2 = sv2 / nl / (R.qbar * R.qbar);
    R.var_v3 = sw2 / nl / (R.qbar * R.qbar);
    return R;
}

} // namespace lagrangian_labels
} // namespace macroflow3d
