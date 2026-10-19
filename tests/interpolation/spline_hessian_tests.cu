/**
 * @file spline_hessian_tests.cu
 * @brief SF-36 N3 (threshold H): contract tests of
 *        interpolation::evaluate_point_hessian (value, gradient, six second
 *        derivatives of the SF-28 periodic tricubic B-spline).
 *
 * Standalone ctest runner (printed [PASS]/[FAIL], main returns 0/1), fast tier.
 *
 *  T1 exactness: coefficient lattice = cubic polynomial in the indices
 *     c(i,j,k) = sum a_pqr i^p j^q k^r (p,q,r <= 3). With coefficient j at
 *     x_j = (j + 1/2) h and t = x/h - 1/2, the centered cubic B-spline obeys
 *       M0 = sum beta3(t-j) = 1,          M1 = sum j   beta3(t-j) = t,
 *       M2 = sum j^2 beta3(t-j) = t^2+1/3, M3 = sum j^3 beta3(t-j) = t^3+t
 *     (verified here by direct summation, check "moment_identities"), so on
 *     interior points (whole 4-point stencil inside [0,N-1], no wrap)
 *       s = sum a_pqr M_p(tx) M_q(ty) M_r(tz),
 *     whose derivatives are analytic (d/dx = (1/h) d/dt). Non-cubic grids and
 *     unequal spacings are used. Gate: normwise relative error <= 1e-12.
 *  T2 (test_fd_gradient, test_hessian_refinement_order) FD of the analytic
 *     gradient, delta in {1e-3, 5e-4} h on a random smooth
 *     lattice. The gradient is piecewise quadratic, so a central difference
 *     inside one cell is exact; the observable O(delta^2) defect arises when
 *     the stencil straddles a knot (jump of the third derivative). The probe
 *     point is therefore placed at knot + theta*delta along the differentiated
 *     axis (theta = 0.25), giving defect ~ J delta^2 f(theta): order 2.
 *     Gate: observed order >= 1.8 for every probe.
 *  T3 host vs device H: <= 1e-14 relative (normwise).
 *  T4 value/gradient of evaluate_point_hessian BITWISE equal to evaluate_point
 *     on host and on device (knots, wrap boundaries, far-outside coordinates).
 *  T5 H finite everywhere; symmetry is by construction (6 components only).
 */

#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/numerics/interpolation/PeriodicTricubicBSpline.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace macroflow3d;
using namespace macroflow3d::interpolation;

namespace {

int g_checks = 0;
bool g_pass = true;

void check(bool ok, const char* name, const char* detail = "") {
    ++g_checks;
    std::printf("[%s] %s %s\n", ok ? "PASS" : "FAIL", name, detail);
    if (!ok)
        g_pass = false;
}

uint64_t splitmix(uint64_t& s) {
    uint64_t z = (s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}
double urand(uint64_t& s) {
    return static_cast<double>(splitmix(s) >> 11) * (1.0 / 9007199254740992.0);
}

struct Lattice {
    int n[3];
    double h[3];
    std::vector<real> c;
    PeriodicTricubicBSplineView view() const {
        PeriodicTricubicBSplineView v;
        v.coeff = c.data();
        v.nx = n[0];
        v.ny = n[1];
        v.nz = n[2];
        v.hx = h[0];
        v.hy = h[1];
        v.hz = h[2];
        v.Lx = n[0] * h[0];
        v.Ly = n[1] * h[1];
        v.Lz = n[2] * h[2];
        return v;
    }
};

// B-spline moment polynomials M_p(t) and their t-derivatives (order 0..2).
void moments(double t, double M[3][4]) {
    M[0][0] = 1;
    M[0][1] = t;
    M[0][2] = t * t + 1.0 / 3.0;
    M[0][3] = t * t * t + t;
    M[1][0] = 0;
    M[1][1] = 1;
    M[1][2] = 2 * t;
    M[1][3] = 3 * t * t + 1;
    M[2][0] = 0;
    M[2][1] = 0;
    M[2][2] = 2;
    M[2][3] = 6 * t;
}

double beta3(double x) {
    x = std::fabs(x);
    if (x >= 2)
        return 0;
    if (x >= 1)
        return (2 - x) * (2 - x) * (2 - x) / 6.0;
    return (4 - 6 * x * x + 3 * x * x * x) / 6.0;
}

// Device kernel: both evaluators per point. Output per point: 4 (evaluate_point)
// + 4 + 6 (evaluate_point_hessian) doubles.
__global__ void k_eval(PeriodicTricubicBSplineView v, const real* px, const real* py,
                       const real* pz, size_t m, real* out) {
    size_t p = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (p >= m)
        return;
    real a[4], b[4], H[6];
    evaluate_point(v, px[p], py[p], pz[p], a[0], a[1], a[2], a[3]);
    evaluate_point_hessian(v, px[p], py[p], pz[p], b[0], b[1], b[2], b[3], H);
    real* o = out + p * 14;
    for (int i = 0; i < 4; ++i)
        o[i] = a[i];
    for (int i = 0; i < 4; ++i)
        o[4 + i] = b[i];
    for (int i = 0; i < 6; ++i)
        o[8 + i] = H[i];
}

void device_eval(const Lattice& L, const std::vector<real>& P, std::vector<real>& out) {
    const size_t m = P.size() / 3;
    std::vector<real> hx(m), hy(m), hz(m);
    for (size_t i = 0; i < m; ++i) {
        hx[i] = P[3 * i];
        hy[i] = P[3 * i + 1];
        hz[i] = P[3 * i + 2];
    }
    real *dc, *dx, *dy, *dz, *dout;
    MACROFLOW3D_CUDA_CHECK(cudaMalloc(&dc, L.c.size() * sizeof(real)));
    MACROFLOW3D_CUDA_CHECK(cudaMalloc(&dx, m * sizeof(real)));
    MACROFLOW3D_CUDA_CHECK(cudaMalloc(&dy, m * sizeof(real)));
    MACROFLOW3D_CUDA_CHECK(cudaMalloc(&dz, m * sizeof(real)));
    MACROFLOW3D_CUDA_CHECK(cudaMalloc(&dout, m * 14 * sizeof(real)));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(dc, L.c.data(), L.c.size() * sizeof(real), cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(dx, hx.data(), m * sizeof(real), cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(dy, hy.data(), m * sizeof(real), cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(dz, hz.data(), m * sizeof(real), cudaMemcpyHostToDevice));
    PeriodicTricubicBSplineView v = L.view();
    v.coeff = dc;
    k_eval<<<static_cast<unsigned>((m + 127) / 128), 128>>>(v, dx, dy, dz, m, dout);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    out.resize(m * 14);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(out.data(), dout, m * 14 * sizeof(real), cudaMemcpyDeviceToHost));
    cudaFree(dc);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    cudaFree(dout);
}

bool bits_equal(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

void test_moment_identities() {
    double worst = 0;
    for (double t : {0.0, 0.123, 0.5, 0.777, 1.0, 3.3}) {
        double M[3][4];
        moments(t, M);
        for (int p = 0; p < 4; ++p) {
            double s = 0;
            for (int j = -10; j <= 20; ++j)
                s += std::pow(double(j), p) * beta3(t - j);
            worst = std::max(worst, std::fabs(s - M[0][p]) / std::max(1.0, std::fabs(M[0][p])));
        }
    }
    char buf[96];
    std::snprintf(buf, sizeof buf, "(max deviation %.3e)", worst);
    check(worst < 1e-13, "moment_identities", buf);
}

struct G {
    int n[3];
    double h[3];
};

void test_exactness() {
    const G grids[] = {{{16, 16, 16}, {1.0 / 16, 1.0 / 16, 1.0 / 16}},
                       {{8, 12, 10}, {0.7, 0.45, 1.1}},
                       {{13, 9, 20}, {0.31, 1.7, 0.05}}};
    uint64_t seed = 12345;
    for (const G& g : grids) {
        double a[4][4][4];
        for (auto& A : a)
            for (auto& B : A)
                for (double& x : B)
                    x = 2 * urand(seed) - 1;
        Lattice L;
        for (int d = 0; d < 3; ++d) {
            L.n[d] = g.n[d];
            L.h[d] = g.h[d];
        }
        L.c.assign(size_t(L.n[0]) * L.n[1] * L.n[2], 0.0);
        for (int k = 0; k < L.n[2]; ++k)
            for (int j = 0; j < L.n[1]; ++j)
                for (int i = 0; i < L.n[0]; ++i) {
                    double s = 0;
                    for (int p = 0; p < 4; ++p)
                        for (int q = 0; q < 4; ++q)
                            for (int r = 0; r < 4; ++r)
                                s += a[p][q][r] * std::pow(double(i), p) * std::pow(double(j), q) *
                                     std::pow(double(k), r);
                    L.c[i + size_t(L.n[0]) * (j + size_t(L.n[1]) * k)] = s;
                }
        const auto v = L.view();
        double eH = 0, mH = 0, eg = 0, mg = 0, ev = 0, mv = 0;
        for (int it = 0; it < 400; ++it) {
            double t[3], x[3];
            for (int d = 0; d < 3; ++d) {
                t[d] = 1.0 + urand(seed) * (L.n[d] - 3.0 - 1e-9); // i0 in [1, N-3]
                x[d] = (t[d] + 0.5) * L.h[d];
            }
            double M[3][3][4]; // [axis][derivative order][power p]
            for (int d = 0; d < 3; ++d)
                moments(t[d], M[d]);
            auto D = [&](int e0, int e1, int e2) {
                double s = 0;
                for (int p = 0; p < 4; ++p)
                    for (int q = 0; q < 4; ++q)
                        for (int r = 0; r < 4; ++r)
                            s += a[p][q][r] * M[0][e0][p] * M[1][e1][q] * M[2][e2][r];
                return s / (std::pow(L.h[0], e0) * std::pow(L.h[1], e1) * std::pow(L.h[2], e2));
            };
            const double rv = D(0, 0, 0);
            const double rg[3] = {D(1, 0, 0), D(0, 1, 0), D(0, 0, 1)};
            const double rH[6] = {D(2, 0, 0), D(0, 2, 0), D(0, 0, 2),
                                  D(1, 1, 0), D(1, 0, 1), D(0, 1, 1)};
            real val, gx, gy, gz, H[6];
            evaluate_point_hessian(v, x[0], x[1], x[2], val, gx, gy, gz, H);
            const double gg[3] = {gx, gy, gz};
            ev = std::max(ev, std::fabs(val - rv));
            mv = std::max(mv, std::fabs(rv));
            for (int i = 0; i < 3; ++i) {
                eg = std::max(eg, std::fabs(gg[i] - rg[i]));
                mg = std::max(mg, std::fabs(rg[i]));
            }
            for (int i = 0; i < 6; ++i) {
                eH = std::max(eH, std::fabs(H[i] - rH[i]));
                mH = std::max(mH, std::fabs(rH[i]));
            }
        }
        char buf[200];
        std::snprintf(buf, sizeof buf, "(%dx%dx%d: H %.3e, grad %.3e, value %.3e relative)",
                      g.n[0], g.n[1], g.n[2], eH / mH, eg / mg, ev / mv);
        check(eH / mH <= 1e-12 && eg / mg <= 1e-12 && ev / mv <= 1e-12,
              "exactness_cubic_polynomial", buf);
    }
}

// FD of the analytic gradient vs H. The gradient of a cubic spline is piecewise
// QUADRATIC, so a central difference inside one cell is exact (defect = roundoff
// only) and a stencil straddling a knot has a defect J ((1-theta) delta)^2/(4 delta)
// = O(delta) (jump J of the third derivative of s, i.e. of g''): observed order 1,
// the signature of C^2 (H continuous, g'' discontinuous). An order-2 statement is
// meaningful only for the interpolation error against a smooth field (next test).
void test_fd_gradient() {
    const int N[3] = {12, 10, 14};
    const double hh[3] = {0.6, 0.9, 0.4};
    Lattice L;
    for (int d = 0; d < 3; ++d) {
        L.n[d] = N[d];
        L.h[d] = hh[d];
    }
    uint64_t seed = 777;
    L.c.resize(size_t(N[0]) * N[1] * N[2]);
    for (auto& c : L.c)
        c = 2 * urand(seed) - 1;
    const auto v = L.view();
    static const int col[3][3] = {{0, 3, 4}, {3, 1, 5}, {4, 5, 2}};
    const double deltas[2] = {1e-3, 5e-4};
    // mode 0: in-cell, knot + 0.5 h +- delta stays inside one cell.
    // mode 1: straddling, knot + 0.25 delta.
    for (int mode = 0; mode < 2; ++mode) {
        double worst_order = 1e9, best_order = -1e9, max_defect[2] = {0, 0}, max_scale = 0, max_pure = 0;
        int probes = 0;
        for (int axis = 0; axis < 3; ++axis) {
            for (int it = 0; it < 20; ++it) {
                double base[3];
                for (int d = 0; d < 3; ++d)
                    base[d] = urand(seed) * N[d] * hh[d];
                const int m = 1 + int(urand(seed) * (N[axis] - 2));
                double defect[2];
                for (int e = 0; e < 2; ++e) {
                    const double dl = deltas[e] * hh[axis];
                    double x[3] = {base[0], base[1], base[2]};
                    x[axis] = (m + 0.5) * hh[axis] + (mode == 0 ? 0.5 * hh[axis] : 0.25 * dl);
                    double xp[3] = {x[0], x[1], x[2]}, xm[3] = {x[0], x[1], x[2]};
                    xp[axis] += dl;
                    xm[axis] -= dl;
                    real vv, gp[3], gm[3], g0[3], H[6], Hs[6];
                    evaluate_point_hessian(v, xp[0], xp[1], xp[2], vv, gp[0], gp[1], gp[2], Hs);
                    evaluate_point_hessian(v, xm[0], xm[1], xm[2], vv, gm[0], gm[1], gm[2], Hs);
                    evaluate_point_hessian(v, x[0], x[1], x[2], vv, g0[0], g0[1], g0[2], H);
                    double err = 0, err_pure = 0;
                    for (int i = 0; i < 3; ++i) {
                        const double fd = (gp[i] - gm[i]) / (2 * dl);
                        const double dd = std::fabs(fd - H[col[axis][i]]);
                        if (i == axis)
                            err_pure = std::max(err_pure, dd); // g_axis along axis: quadratic
                        else if (mode == 0)
                            err = std::max(err, dd); // g_other along axis: cubic, O(delta^2)
                        if (mode == 1)
                            err = std::max(err, dd);
                        max_scale = std::max(max_scale, std::fabs(H[col[axis][i]]));
                    }
                    max_pure = std::max(max_pure, err_pure);
                    defect[e] = err;
                    max_defect[e] = std::max(max_defect[e], err);
                }
                {
                    const double o = std::log(defect[0] / defect[1]) / std::log(2.0);
                    worst_order = std::min(worst_order, o);
                    best_order = std::max(best_order, o);
                }
                ++probes;
            }
        }
        char buf[260];
        if (mode == 0) {
            std::snprintf(buf, sizeof buf,
                          "(%d probes, cross components defect %.3e (delta=1e-3h) / %.3e (5e-4h), observed order in [%.4f, %.4f]; along-axis component (quadratic, exact) defect %.3e; max|H| %.3e)",
                          probes, max_defect[0], max_defect[1], worst_order, best_order, max_pure,
                          max_scale);
            check(worst_order >= 1.8 && max_pure <= 1e-8 * max_scale,
                  "fd_gradient_in_cell_order2", buf);
        } else {
            std::snprintf(buf, sizeof buf,
                          "(%d probes, observed order in [%.4f, %.4f] (expected 1: C2 not C3), max defect %.3e / %.3e)",
                          probes, worst_order, best_order, max_defect[0], max_defect[1]);
            check(worst_order >= 0.95 && best_order <= 1.05 && max_defect[1] < max_defect[0],
                  "fd_gradient_straddle_c2_signature", buf);
        }
    }
}

// Interpolation-error order of H against the analytic Hessian of a smooth
// periodic field (SF-28 D-5 test field) under grid refinement 16^3 -> 32^3 -> 64^3.
void test_hessian_refinement_order() {
    const double a = 2 * M_PI, b = 4 * M_PI;
    auto exactH = [&](double x, double y, double z, double H[6]) {
        const double A = std::sin(a * x) * std::cos(b * y) * std::sin(a * z);
        const double cxy = std::cos(a * (x + y));
        H[0] = -a * a * A - 0.5 * a * a * cxy;
        H[1] = -b * b * A - 0.5 * a * a * cxy;
        H[2] = -a * a * A - 0.25 * 9 * a * a * std::sin(3 * a * z);
        H[3] = -a * b * std::cos(a * x) * std::sin(b * y) * std::sin(a * z) - 0.5 * a * a * cxy;
        H[4] = a * a * std::cos(a * x) * std::cos(b * y) * std::cos(a * z);
        H[5] = -a * b * std::sin(a * x) * std::sin(b * y) * std::cos(a * z);
    };
    auto field = [&](double x, double y, double z) {
        return std::sin(a * x) * std::cos(b * y) * std::sin(a * z) + 0.5 * std::cos(a * (x + y)) +
               0.25 * std::sin(3 * a * z);
    };
    double err[3];
    const int ns[3] = {16, 32, 64};
    for (int gi = 0; gi < 3; ++gi) {
        const int n = ns[gi];
        Grid3D grid(n, n, n, 1.0 / n, 1.0 / n, 1.0 / n);
        std::vector<real> f(size_t(n) * n * n), c(size_t(n) * n * n);
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    f[i + size_t(n) * (j + size_t(n) * k)] =
                        field((i + 0.5) / n, (j + 0.5) / n, (k + 0.5) / n);
        prefilter_periodic_tricubic_bspline_host(grid, f.data(), c.data());
        const auto v = make_host_view(grid, c.data());
        uint64_t seed = 4242; // same points on every grid
        double e = 0, scale = 0;
        for (int p = 0; p < 500; ++p) {
            const double x = urand(seed), y = urand(seed), z = urand(seed);
            real val, g[3], H[6];
            double R[6];
            evaluate_point_hessian(v, x, y, z, val, g[0], g[1], g[2], H);
            exactH(x, y, z, R);
            for (int i = 0; i < 6; ++i) {
                e = std::max(e, std::fabs(H[i] - R[i]));
                scale = std::max(scale, std::fabs(R[i]));
            }
        }
        err[gi] = e / scale;
    }
    const double o1 = std::log(err[0] / err[1]) / std::log(2.0);
    const double o2 = std::log(err[1] / err[2]) / std::log(2.0);
    char buf[200];
    std::snprintf(buf, sizeof buf,
                  "(normwise error %.3e, %.3e, %.3e at 16^3,32^3,64^3; orders %.3f, %.3f)",
                  err[0], err[1], err[2], o1, o2);
    check(o1 >= 1.8 && o2 >= 1.8, "hessian_refinement_order", buf);
}

std::vector<real> make_points(const Lattice& L, uint64_t seed) {
    std::vector<real> P;
    auto push = [&](double x, double y, double z) {
        P.push_back(x);
        P.push_back(y);
        P.push_back(z);
    };
    const double Ls[3] = {L.n[0] * L.h[0], L.n[1] * L.h[1], L.n[2] * L.h[2]};
    for (int i = 0; i < 3000; ++i)
        push(urand(seed) * Ls[0], urand(seed) * Ls[1], urand(seed) * Ls[2]);
    for (int i = 0; i < 500; ++i) // exactly on knots (cell centres)
        push((int(urand(seed) * L.n[0]) + 0.5) * L.h[0], (int(urand(seed) * L.n[1]) + 0.5) * L.h[1],
             (int(urand(seed) * L.n[2]) + 0.5) * L.h[2]);
    for (int i = 0; i < 300; ++i) { // wrap boundaries
        const double e[5] = {0.0, 1e-300, 1.0 - 1e-16, 1.0, -1.0};
        push(e[int(urand(seed) * 5)] * Ls[0], e[int(urand(seed) * 5)] * Ls[1],
             e[int(urand(seed) * 5)] * Ls[2]);
    }
    for (int i = 0; i < 700; ++i) // far outside [0, L)
        push((urand(seed) - 0.5) * 2000 * Ls[0], (urand(seed) - 0.5) * 2000 * Ls[1],
             (urand(seed) - 0.5) * 2000 * Ls[2]);
    return P;
}

void test_host_device() {
    const G grids[] = {{{16, 16, 16}, {1.0 / 16, 1.0 / 16, 1.0 / 16}},
                       {{8, 12, 10}, {0.7, 0.45, 1.1}}};
    uint64_t seed = 99;
    for (const G& g : grids) {
        Lattice L;
        for (int d = 0; d < 3; ++d) {
            L.n[d] = g.n[d];
            L.h[d] = g.h[d];
        }
        L.c.resize(size_t(L.n[0]) * L.n[1] * L.n[2]);
        for (auto& c : L.c)
            c = 2 * urand(seed) - 1;
        const auto v = L.view();
        const auto P = make_points(L, seed + 1);
        const size_t m = P.size() / 3;
        std::vector<real> dev;
        device_eval(L, P, dev);

        size_t bit_host = 0, bit_dev = 0, finite = 0;
        double dmax = 0, dscale = 0, dmax_far = 0, dscale_far = 0;
        for (size_t p = 0; p < m; ++p) {
            real a[4], b[4], H[6];
            evaluate_point(v, P[3 * p], P[3 * p + 1], P[3 * p + 2], a[0], a[1], a[2], a[3]);
            evaluate_point_hessian(v, P[3 * p], P[3 * p + 1], P[3 * p + 2], b[0], b[1], b[2], b[3],
                                   H);
            bool eq = true;
            for (int i = 0; i < 4; ++i)
                eq = eq && bits_equal(a[i], b[i]);
            bit_host += eq;
            const real* o = &dev[p * 14];
            bool eqd = true;
            for (int i = 0; i < 4; ++i)
                eqd = eqd && bits_equal(o[i], o[4 + i]);
            bit_dev += eqd;
            bool fin = true;
            const bool far = p >= m - 700; // far-outside block (see make_points)
            for (int i = 0; i < 6; ++i) {
                fin = fin && std::isfinite(H[i]) && std::isfinite(o[8 + i]);
                double& dm = far ? dmax_far : dmax;
                double& ds = far ? dscale_far : dscale;
                dm = std::max(dm, std::fabs(H[i] - o[8 + i]));
                ds = std::max(ds, std::fabs(H[i]));
            }
            finite += fin;
        }
        char buf[200];
        std::snprintf(buf, sizeof buf, "(%dx%dx%d: %zu/%zu points)", g.n[0], g.n[1], g.n[2],
                      bit_host, m);
        check(bit_host == m, "bitwise_value_gradient_host", buf);
        std::snprintf(buf, sizeof buf, "(%dx%dx%d: %zu/%zu points)", g.n[0], g.n[1], g.n[2],
                      bit_dev, m);
        check(bit_dev == m, "bitwise_value_gradient_device", buf);
        std::snprintf(buf, sizeof buf,
                      "(%dx%dx%d: max|H_host-H_dev| %.3e, max|H| %.3e, rel %.3e)", g.n[0], g.n[1],
                      g.n[2], dmax, dscale, dmax / dscale);
        check(dmax / dscale <= 1e-14, "host_device_hessian", buf);
        std::snprintf(buf, sizeof buf,
                      "(%dx%dx%d: far-outside |x|<=1e3 L, rel %.3e; host/device differ in the FMA-contracted coordinate reduction, so only a loose bound is gated)",
                      g.n[0], g.n[1], g.n[2], dmax_far / dscale_far);
        check(dmax_far / dscale_far <= 1e-9, "host_device_hessian_far_outside", buf);
        std::snprintf(buf, sizeof buf, "(%zu/%zu points)", finite, m);
        check(finite == m, "hessian_finite", buf);
    }
}

} // namespace

int main() {
    const auto t0 = std::chrono::steady_clock::now();
    test_moment_identities();
    test_exactness();
    test_fd_gradient();
    test_hessian_refinement_order();
    test_host_device();
    const double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("%d checks, wall %.2f s: %s\n", g_checks, sec, g_pass ? "ALL PASS" : "FAILURES");
    return g_pass ? 0 : 1;
}
