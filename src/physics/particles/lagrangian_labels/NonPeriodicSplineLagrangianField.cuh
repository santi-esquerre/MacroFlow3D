#pragma once

/**
 * @file NonPeriodicSplineLagrangianField.cuh
 * @brief SF-41 N3b: device field functors of the Lagrangian tracker over the NON-periodic-in-x1 tricubic
 *        spline (NonPeriodicXTricubicBSpline.cuh) of the potential fluctuation f and of ln k: unfused (two
 *        views) and fused (one interleaved `double2 {pot, logk}` array in the x-extended layout).
 *
 * Contract (C-EXIT, with the exit rules of LagrangianTrackerDevice.cuh):
 *  - `kNonPeriodicX = true` selects the exit handling of the tracker passes at compile time; the periodic
 *    functors (SplineLagrangianField, FusedSplineLagrangianField) are not touched.
 *  - Evaluation domain x1 in [-hx/2, L1 + hx/2) (half-open), y, z finite; `in_domain(x)` decides it with the
 *    expression of the spline (t = x1/hx - 1/2, t >= -1 and t < nx). `eval` / `eval_ext` return false outside
 *    it: g, k (and grad_lnk, H) are NaN and no coefficient is read. Inside, they return true.
 *  - g = G + grad s_f, k = exp(s_lnk), both splines on one geometry (the fused functor reads both from one
 *    16-byte load and mirrors FusedSplineLagrangianField term by term: per spline the loop nesting c (z),
 *    b (y), a (x), operand order and accumulation order are those of evaluate_point (ln k; the potential
 *    in `eval`) and evaluate_point_hessian (potential in `eval_ext`), so fused and unfused return the same
 *    bits).
 *  - `length()` = L1 = nx hx and `margin()` = hx/2 (the half-cell continuation of the spline).
 *
 * Include only from translation units compiled with --fmad=false (LagrangianTrackerDevice.cuh). The
 * unfused functor is also callable on the host (host views), used by the controls for the C2 quadrature and C3.
 */

#include "../../../numerics/interpolation/NonPeriodicXTricubicBSpline.cuh"
#include "LagrangianTrackerDevice.cuh"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace macroflow3d {
namespace lagrangian_labels {

namespace detail_np {
__host__ __device__ inline bool in_domain_x(double x1, double x2, double x3, int nx, double hx) {
    const double t = x1 / hx - 0.5;
    return t >= -1.0 && t < static_cast<double>(nx) && std::isfinite(x2) && std::isfinite(x3);
}
} // namespace detail_np

struct NonPeriodicSplineLagrangianField {
    static constexpr bool kNonPeriodicX = true;
    interpolation::NonPeriodicXTricubicBSplineView potential{};
    interpolation::NonPeriodicXTricubicBSplineView log_conductivity{};
    double G[3] = {0.0, 0.0, 0.0};

    __host__ __device__ double length() const { return potential.Lx; }
    __host__ __device__ double margin() const { return 0.5 * potential.hx; }
    __host__ __device__ bool in_domain(const double x[3]) const {
        return detail_np::in_domain_x(x[0], x[1], x[2], potential.nx, potential.hx);
    }

    __host__ __device__ bool eval(const double x[3], double g[3], double& k) const {
        real v = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
        const bool a = interpolation::evaluate_point(potential, x[0], x[1], x[2], v, gx, gy, gz);
        g[0] = G[0] + gx;
        g[1] = G[1] + gy;
        g[2] = G[2] + gz;
        real y = 0.0, yx = 0.0, yy = 0.0, yz = 0.0;
        const bool b =
            interpolation::evaluate_point(log_conductivity, x[0], x[1], x[2], y, yx, yy, yz);
        k = exp(y);
        return a && b;
    }

    __host__ __device__ bool eval_ext(const double x[3], double g[3], double& k, double grad_lnk[3],
                                      double H[6]) const {
        real v = 0.0, gx = 0.0, gy = 0.0, gz = 0.0;
        const bool a =
            interpolation::evaluate_point_hessian(potential, x[0], x[1], x[2], v, gx, gy, gz, H);
        g[0] = G[0] + gx;
        g[1] = G[1] + gy;
        g[2] = G[2] + gz;
        real y = 0.0, yx = 0.0, yy = 0.0, yz = 0.0;
        const bool b =
            interpolation::evaluate_point(log_conductivity, x[0], x[1], x[2], y, yx, yy, yz);
        k = exp(y);
        grad_lnk[0] = yx;
        grad_lnk[1] = yy;
        grad_lnk[2] = yz;
        return a && b;
    }
};

struct FusedNonPeriodicSplineLagrangianField {
    static constexpr bool kNonPeriodicX = true;
    const double2* coeff = nullptr; ///< (nx+4) ny nz, {potential, ln k} at (a+2) + (nx+4)(j + ny k)
    int nx = 0, ny = 0, nz = 0;
    real hx = 0.0, hy = 0.0, hz = 0.0;
    real Lx = 0.0, Ly = 0.0, Lz = 0.0;
    double G[3] = {0.0, 0.0, 0.0};

    __host__ __device__ double length() const { return Lx; }
    __host__ __device__ double margin() const { return 0.5 * hx; }
    __host__ __device__ bool in_domain(const double x[3]) const {
        return detail_np::in_domain_x(x[0], x[1], x[2], nx, hx);
    }

    __device__ __forceinline__ bool eval(const double x[3], double g[3], double& k) const {
        interpolation::detail::AxisStencil sx, sy, sz;
        real ux;
        if (!in_domain(x) || !interpolation::detail::axis_stencil_x(x[0], nx, hx, sx, ux)) {
            g[0] = g[1] = g[2] = k = dnan_();
            return false;
        }
        interpolation::detail::axis_stencil(x[1], ny, hy, Ly, sy);
        interpolation::detail::axis_stencil(x[2], nz, hz, Lz, sz);

        const size_t nxe = static_cast<size_t>(nx) + 4;
        const size_t nxy = nxe * static_cast<size_t>(ny);

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
                const double2* row = coeff + off_z + static_cast<size_t>(sy.idx[b]) * nxe;
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
        return true;
    }

    __device__ __forceinline__ bool eval_ext(const double x[3], double g[3], double& k,
                                             double grad_lnk[3], double H[6]) const {
        interpolation::detail::AxisStencil2 sx, sy, sz; // potential: idx/w/dw/d2w
        if (!in_domain(x) || !interpolation::detail::axis_stencil_x2(x[0], nx, hx, sx)) {
            g[0] = g[1] = g[2] = k = dnan_();
            grad_lnk[0] = grad_lnk[1] = grad_lnk[2] = dnan_();
            for (int i = 0; i < 6; ++i)
                H[i] = dnan_();
            return false;
        }
        interpolation::detail::axis_stencil2(x[1], ny, hy, Ly, sy);
        interpolation::detail::axis_stencil2(x[2], nz, hz, Lz, sz);

        const size_t nxe = static_cast<size_t>(nx) + 4;
        const size_t nxy = nxe * static_cast<size_t>(ny);

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
                const double2* row = coeff + off_z + static_cast<size_t>(sy.idx[b]) * nxe;
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
        return true;
    }

  private:
    __host__ __device__ static double dnan_() { return interpolation::detail::np_nan(); }
};

/// Host: fused non-periodic field over `coeff` (interleaved, device, (nx+4) ny nz entries) with the geometry
/// and G of `f` (both views must share nx, ny, nz, spacings and periods).
inline FusedNonPeriodicSplineLagrangianField make_fused_np_field(const NonPeriodicSplineLagrangianField& f,
                                                                 const double2* coeff) {
    const auto &p = f.potential, &q = f.log_conductivity;
    if (p.nx != q.nx || p.ny != q.ny || p.nz != q.nz || p.hx != q.hx || p.hy != q.hy ||
        p.hz != q.hz || p.Lx != q.Lx || p.Ly != q.Ly || p.Lz != q.Lz) {
        throw std::logic_error(
            "make_fused_np_field: potential and ln k splines differ in geometry");
    }
    FusedNonPeriodicSplineLagrangianField r;
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
