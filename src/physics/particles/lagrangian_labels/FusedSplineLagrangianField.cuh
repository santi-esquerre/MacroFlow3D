#pragma once

/**
 * @file FusedSplineLagrangianField.cuh
 * @brief SF-40 L6: fused evaluation of the two splines of the Lagrangian tracker (potential and
 *        ln k) over ONE interleaved `double2 {pot, logk}` coefficient array.
 *
 * Contract (C-TRACKER-L6, bitwise): `eval` / `eval_ext` return exactly the bits of
 * `SplineLagrangianField::eval` / `eval_ext`. Per spline the loop nesting (c over z, b over y, a over
 * x), the operand order and the accumulation order are those of `interpolation::evaluate_point`
 * (ln k; potential in `eval`) and `interpolation::evaluate_point_hessian` (potential in `eval_ext`);
 * every accumulator starts at 0.0 and is accumulated term by term (nothing peeled, split or
 * reassociated). Only the stencil computation (idx/w/dw, bitwise what `axis_stencil` gives; the
 * ln k spline has the same geometry) and the coefficient load (one 16-byte load gives both values)
 * are shared. The two splines must have identical geometry (checked by `make_fused_field`).
 *
 * Include only from translation units compiled with --fmad=false (LagrangianTrackerDevice.cuh).
 */

#include "LagrangianTrackerDevice.cuh"

#include <cstddef>
#include <stdexcept>
#include <vector>

namespace macroflow3d {
namespace lagrangian_labels {

struct FusedSplineLagrangianField {
    const double2* coeff = nullptr; ///< n_cells, {potential, ln k} at i + nx (j + ny k)
    int nx = 0, ny = 0, nz = 0;
    real hx = 0.0, hy = 0.0, hz = 0.0;
    real Lx = 0.0, Ly = 0.0, Lz = 0.0;
    double G[3] = {0.0, 0.0, 0.0};

    __device__ __forceinline__ void eval(const double x[3], double g[3], double& k) const {
        interpolation::detail::AxisStencil sx, sy, sz;
        interpolation::detail::axis_stencil(x[0], nx, hx, Lx, sx);
        interpolation::detail::axis_stencil(x[1], ny, hy, Ly, sy);
        interpolation::detail::axis_stencil(x[2], nz, hz, Lz, sz);

        const size_t nxs = static_cast<size_t>(nx);
        const size_t nxy = nxs * static_cast<size_t>(ny);

        // pf* : potential (evaluate_point), qf* : ln k (evaluate_point)
        real pf = 0.0, pfx = 0.0, pfy = 0.0, pfz = 0.0;
        real qf = 0.0, qfx = 0.0, qfy = 0.0, qfz = 0.0;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const size_t off_z = static_cast<size_t>(sz.idx[c]) * nxy;
            real pf_c = 0.0, pfx_c = 0.0, pfy_c = 0.0;
            real qf_c = 0.0, qfx_c = 0.0, qfy_c = 0.0;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const double2* row = coeff + off_z + static_cast<size_t>(sy.idx[b]) * nxs;
                real pr = 0.0, prx = 0.0;
                real qr = 0.0, qrx = 0.0;
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    const double2 cv = row[sx.idx[a]];
                    pr += sx.w[a] * cv.x;
                    prx += sx.dw[a] * cv.x;
                    qr += sx.w[a] * cv.y;
                    qrx += sx.dw[a] * cv.y;
                }
                pf_c += sy.w[b] * pr;
                pfx_c += sy.w[b] * prx;
                pfy_c += sy.dw[b] * pr;
                qf_c += sy.w[b] * qr;
                qfx_c += sy.w[b] * qrx;
                qfy_c += sy.dw[b] * qr;
            }
            pf += sz.w[c] * pf_c;
            pfx += sz.w[c] * pfx_c;
            pfy += sz.w[c] * pfy_c;
            pfz += sz.dw[c] * pf_c;
            qf += sz.w[c] * qf_c;
            qfx += sz.w[c] * qfx_c;
            qfy += sz.w[c] * qfy_c;
            qfz += sz.dw[c] * qf_c;
        }
        g[0] = G[0] + pfx;
        g[1] = G[1] + pfy;
        g[2] = G[2] + pfz;
        k = exp(qf);
    }

    __device__ __forceinline__ void eval_ext(const double x[3], double g[3], double& k,
                                             double grad_lnk[3], double H[6]) const {
        interpolation::detail::AxisStencil2 sx, sy, sz; // potential: idx/w/dw/d2w
        interpolation::detail::axis_stencil2(x[0], nx, hx, Lx, sx);
        interpolation::detail::axis_stencil2(x[1], ny, hy, Ly, sy);
        interpolation::detail::axis_stencil2(x[2], nz, hz, Lz, sz);

        const size_t nxs = static_cast<size_t>(nx);
        const size_t nxy = nxs * static_cast<size_t>(ny);

        real f = 0.0, fx = 0.0, fy = 0.0, fz = 0.0;
        real fxx = 0.0, fyy = 0.0, fzz = 0.0, fxy = 0.0, fxz = 0.0, fyz = 0.0;
        real qf = 0.0, qfx = 0.0, qfy = 0.0, qfz = 0.0;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const size_t off_z = static_cast<size_t>(sz.idx[c]) * nxy;
            real f_c = 0.0, fx_c = 0.0, fy_c = 0.0;
            real fxx_c = 0.0, fyy_c = 0.0, fxy_c = 0.0;
            real qf_c = 0.0, qfx_c = 0.0, qfy_c = 0.0;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const double2* row = coeff + off_z + static_cast<size_t>(sy.idx[b]) * nxs;
                real r = 0.0, rx = 0.0, rxx = 0.0;
                real qr = 0.0, qrx = 0.0;
#pragma unroll
                for (int a = 0; a < 4; ++a) {
                    const double2 cv = row[sx.idx[a]];
                    r += sx.w[a] * cv.x;
                    rx += sx.dw[a] * cv.x;
                    rxx += sx.d2w[a] * cv.x;
                    qr += sx.w[a] * cv.y;
                    qrx += sx.dw[a] * cv.y;
                }
                f_c += sy.w[b] * r;
                fx_c += sy.w[b] * rx;
                fy_c += sy.dw[b] * r;
                fxx_c += sy.w[b] * rxx;
                fxy_c += sy.dw[b] * rx;
                fyy_c += sy.d2w[b] * r;
                qf_c += sy.w[b] * qr;
                qfx_c += sy.w[b] * qrx;
                qfy_c += sy.dw[b] * qr;
            }
            f += sz.w[c] * f_c;
            fx += sz.w[c] * fx_c;
            fy += sz.w[c] * fy_c;
            fz += sz.dw[c] * f_c;
            fxx += sz.w[c] * fxx_c;
            fyy += sz.w[c] * fyy_c;
            fxy += sz.w[c] * fxy_c;
            fxz += sz.dw[c] * fx_c;
            fyz += sz.dw[c] * fy_c;
            fzz += sz.d2w[c] * f_c;
            qf += sz.w[c] * qf_c;
            qfx += sz.w[c] * qfx_c;
            qfy += sz.w[c] * qfy_c;
            qfz += sz.dw[c] * qf_c;
        }
        (void)f;
        H[0] = fxx;
        H[1] = fyy;
        H[2] = fzz;
        H[3] = fxy;
        H[4] = fxz;
        H[5] = fyz;
        g[0] = G[0] + fx;
        g[1] = G[1] + fy;
        g[2] = G[2] + fz;
        k = exp(qf);
        grad_lnk[0] = qfx;
        grad_lnk[1] = qfy;
        grad_lnk[2] = qfz;
    }
};

/// Host: interleaves two coefficient arrays of n cells into {pot, logk} pairs.
inline std::vector<double2> interleave_coefficients(const real* pot, const real* logk,
                                                    std::size_t n) {
    std::vector<double2> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = make_double2(pot[i], logk[i]);
    return v;
}

/// Host: fused field over `coeff` (interleaved, device) with the geometry and G of `f` (the two
/// splines of `f` must share nx, ny, nz, spacings and periods).
inline FusedSplineLagrangianField make_fused_field(const SplineLagrangianField& f,
                                                   const double2* coeff) {
    const auto &p = f.potential, &q = f.log_conductivity;
    if (p.nx != q.nx || p.ny != q.ny || p.nz != q.nz || p.hx != q.hx || p.hy != q.hy ||
        p.hz != q.hz || p.Lx != q.Lx || p.Ly != q.Ly || p.Lz != q.Lz) {
        throw std::logic_error("make_fused_field: potential and ln k splines differ in geometry");
    }
    FusedSplineLagrangianField r;
    r.coeff = coeff;
    r.nx = p.nx;
    r.ny = p.ny;
    r.nz = p.nz;
    r.hx = p.hx;
    r.hy = p.hy;
    r.hz = p.hz;
    r.Lx = p.Lx;
    r.Ly = p.Ly;
    r.Lz = p.Lz;
    for (int d = 0; d < 3; ++d)
        r.G[d] = f.G[d];
    return r;
}

} // namespace lagrangian_labels
} // namespace macroflow3d
