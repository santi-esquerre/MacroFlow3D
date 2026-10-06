/**
 * @file face_flux_tests.cu
 * @brief SF-32 N3a: fast contract tests of the Stokes face fluxes
 *        (`src/physics/particles/streamline_tracker/StokesFaceVelocity.cuh`).
 *
 * Standalone ctest-friendly runner (style of
 * `tests/streamline_tracker/pseudo_symplectic_tracker_tests.cu`: printed
 * `[PASS]/[FAIL]` checks, `main` returns 0/1). Fast tier.
 *
 * Every case, grid, tolerance and threshold is PRE-REGISTERED by the SF-32
 * orchestrator (node N3a specification; contract section 3.2, checks S1-S7)
 * and implemented verbatim; none may be adjusted to make a failing line pass.
 * The module under test is not modified here: defects are reported, not
 * patched.
 *
 * Fixtures: pairs U, A, B, G of analytic_pairs.hpp, fluctuations sampled at
 * the cell centres of N^3 grids of the unit cube (N = 16, 32). Device labels:
 * SF-28 GPU prefilter. Host labels: SF-28 host prefilter
 * (prefilter_periodic_tricubic_bspline_host) + make_host_view. Pollock grids
 * Delta = h (n = N) and Delta = 2h (n = N/2). Device results are downloaded
 * to the host for every check.
 *
 * Cases:
 *   S1 divergence        (module diagnostic AND an independent recomputation)
 *   S2 exactness         (direct 2-D composite Gauss-Legendre over the face)
 *   S3 additivity        (Delta = 2h face == mean of its four Delta = h faces)
 *   S4 uniform pair      (u = 1, v = w = 0)
 *   S5 pair A            (u = 1, w = 0, v(i) only, v vs exact cell mean)
 *   S6 host == device
 *   S7 no allocation     (cudaMemGetInfo unchanged on a second call)
 *   validation_errors, determinism, timing_record (information only).
 */

#include "analytic_pairs.hpp"

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/numerics/interpolation/PeriodicTricubicBSpline.cuh"
#include "src/physics/particles/streamline_tracker/StokesFaceVelocity.cuh"
#include "src/physics/particles/streamline_tracker/StreamlineTrackerCommon.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace macroflow3d;
using namespace macroflow3d::interpolation;
using namespace macroflow3d::physics::particles::streamline_tracker;
using namespace sf31_tests;

namespace {

// ============================================================================
// Bookkeeping
// ============================================================================

struct TestReport {
    bool overall_pass = true;
    int checks = 0;
    int fails = 0;

    void check(bool cond, const std::string& name, const std::string& detail = "") {
        ++checks;
        if (!cond)
            ++fails;
        std::printf("[%s] %s%s%s\n", cond ? "PASS" : "FAIL", name.c_str(),
                    detail.empty() ? "" : "  ", detail.c_str());
        overall_pass = overall_pass && cond;
    }
};

std::string strf(const char* f, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return buf;
}

template <class T> std::vector<T> d2h(const T* d, size_t n) {
    std::vector<T> h(n);
    if (n > 0)
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

template <class T> void h2d(T* d, const std::vector<T>& h) {
    if (!h.empty())
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}

size_t free_bytes() {
    size_t f = 0, t = 0;
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&f, &t));
    return f;
}

Grid3D cube_grid(int n) {
    const real h = 1.0 / static_cast<real>(n);
    return Grid3D(n, n, n, h, h, h);
}

/// Periodic linear index.
inline size_t pidx(int i, int j, int k, int n) {
    const int ii = ((i % n) + n) % n;
    const int jj = ((j % n) + n) % n;
    const int kk = ((k % n) + n) % n;
    return static_cast<size_t>(ii) + static_cast<size_t>(n) * (static_cast<size_t>(jj) +
                                                               static_cast<size_t>(n) * static_cast<size_t>(kk));
}

// ============================================================================
// Splined labels
// ============================================================================

/// Splined labels of one pair on an N^3 grid: SF-28 GPU prefilter (device pair)
/// and SF-28 host prefilter (host pair). Not movable (views point into members).
struct SplineSet {
    int pair = 0;
    int N = 0;
    Grid3D g;
    PeriodicTricubicBSplineWorkspace ws1, ws2;
    std::vector<real> hc1, hc2;
    SplineLabelPair dev{}, host{};
    double coeff_gpu_host_diff = 0.0; ///< max |c_gpu - c_host| over both labels (information)

    SplineSet() = default;
    SplineSet(const SplineSet&) = delete;
    SplineSet& operator=(const SplineSet&) = delete;
};

std::unique_ptr<SplineSet> build_spline(const CudaContext& ctx, int pair, int N) {
    std::unique_ptr<SplineSet> S(new SplineSet());
    S->pair = pair;
    S->N = N;
    S->g = cube_grid(N);
    const Grid3D& g = S->g;
    std::vector<real> s1, s2;
    sample_fluctuations(pair, g, s1, s2);
    DeviceBuffer<real> d1(s1.size()), d2(s2.size());
    h2d(d1.data(), s1);
    h2d(d2.data(), s2);
    prefilter_periodic_tricubic_bspline(ctx, g, DeviceSpan<const real>(d1.data(), d1.size()), S->ws1);
    prefilter_periodic_tricubic_bspline(ctx, g, DeviceSpan<const real>(d2.data(), d2.size()), S->ws2);
    ctx.synchronize();
    real gb1[3], gb2[3];
    pair_gbar(pair, gb1, gb2);
    S->dev = make_spline_label_pair(S->ws1.view(), S->ws2.view(), gb1, gb2);
    S->hc1.assign(g.num_cells(), 0.0);
    S->hc2.assign(g.num_cells(), 0.0);
    prefilter_periodic_tricubic_bspline_host(g, s1.data(), S->hc1.data());
    prefilter_periodic_tricubic_bspline_host(g, s2.data(), S->hc2.data());
    S->host = make_spline_label_pair(make_host_view(g, S->hc1.data()), make_host_view(g, S->hc2.data()),
                                     gb1, gb2);
    const std::vector<real> gc1 = d2h(S->ws1.coefficients.data(), g.num_cells());
    const std::vector<real> gc2 = d2h(S->ws2.coefficients.data(), g.num_cells());
    double e = 0.0;
    for (size_t c = 0; c < g.num_cells(); ++c) {
        e = std::max(e, std::fabs(gc1[c] - S->hc1[c]));
        e = std::max(e, std::fabs(gc2[c] - S->hc2[c]));
    }
    S->coeff_gpu_host_diff = e;
    std::printf("  [spline] pair %s on %d^3: max |coeff GPU - coeff host| = %.3e\n", pair_name(pair), N, e);
    return S;
}

/// Face arrays downloaded from the device (or computed by the host mirror).
struct Faces {
    int n = 0;
    real D = 0.0;
    std::vector<real> u, v, w;
};

/// GPU face fluxes of S on the Pollock grid with n cells per axis; downloaded.
Faces gpu_faces(const CudaContext& ctx, const SplineSet& S, int n, StokesFaceFluxWorkspace& ws) {
    const Grid3D pg = cube_grid(n);
    compute_stokes_face_fluxes(ctx.cuda_stream(), S.dev, pg, ws);
    ctx.synchronize();
    Faces F;
    F.n = n;
    F.D = pg.dx;
    const PeriodicFaceFluxView fv = ws.view();
    F.u = d2h(fv.u, pg.num_cells());
    F.v = d2h(fv.v, pg.num_cells());
    F.w = d2h(fv.w, pg.num_cells());
    return F;
}

Faces gpu_faces(const CudaContext& ctx, const SplineSet& S, int n) {
    StokesFaceFluxWorkspace ws;
    return gpu_faces(ctx, S, n, ws);
}

Faces host_faces(const SplineSet& S, int n) {
    const Grid3D pg = cube_grid(n);
    Faces F;
    F.n = n;
    F.D = pg.dx;
    compute_stokes_face_fluxes_host(S.host, pg, F.u, F.v, F.w);
    return F;
}

PeriodicFaceFluxView host_view(const Faces& F) {
    PeriodicFaceFluxView fv{};
    fv.u = F.u.data();
    fv.v = F.v.data();
    fv.w = F.w.data();
    fv.nx = fv.ny = fv.nz = F.n;
    fv.dx = fv.dy = fv.dz = F.D;
    fv.Lx = fv.Ly = fv.Lz = F.n * F.D;
    return fv;
}

/// Independent flux-form divergence: max_cells |sum_d (face_d[+1] - face_d) * area| / (max|u| * area).
double own_relative_divergence(const Faces& F) {
    const int n = F.n;
    const double area = F.D * F.D;
    double md = 0.0, mu = 0.0;
    for (int k = 0; k < n; ++k)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                const size_t c = pidx(i, j, k, n);
                const double div = (F.u[pidx(i + 1, j, k, n)] - F.u[c]) * area +
                                   (F.v[pidx(i, j + 1, k, n)] - F.v[c]) * area +
                                   (F.w[pidx(i, j, k + 1, n)] - F.w[c]) * area;
                md = std::max(md, std::fabs(div));
                mu = std::max(mu, std::fabs(F.u[c]));
            }
    if (mu * area == 0.0)
        return md == 0.0 ? 0.0 : INFINITY;
    return md / (mu * area);
}

/// c = grad psi1 x grad psi2 of a (host) spline pair at x.
void spline_velocity(const SplineLabelPair& L, const double x[3], double c[3]) {
    const real xi[3] = {x[0], x[1], x[2]};
    const int32_t w[3] = {0, 0, 0};
    LabelSample s;
    L(xi, w, s);
    cross3(s.g1, s.g2, c);
}

/// Breakpoints of [a, b] at the label knots (j + 1/2) h strictly inside.
std::vector<double> knot_partition(double a, double b, double h) {
    std::vector<double> p;
    p.push_back(a);
    for (long j = static_cast<long>(std::floor(a / h - 0.5)) - 1; ; ++j) {
        const double t = (static_cast<double>(j) + 0.5) * h;
        if (t >= b)
            break;
        if (t > a)
            p.push_back(t);
    }
    p.push_back(b);
    return p;
}

/// Direct 2-D composite Gauss-Legendre (4x4 per label-knot cell) average of
/// c . e_comp over the face at the lower comp-coordinate of cell (i, j, k).
double face_quadrature(const SplineLabelPair& L, double h, double D, int i, int j, int k, int comp) {
    static const double gx[4] = {-0.86113631159405257522, -0.33998104358485626480, 0.33998104358485626480,
                                 0.86113631159405257522};
    static const double gw[4] = {0.34785484513745385737, 0.65214515486254614263, 0.65214515486254614263,
                                 0.34785484513745385737};
    const int ax0 = comp == 0 ? 1 : 0;
    const int ax1 = comp == 2 ? 1 : 2;
    const double base[3] = {i * D, j * D, k * D};
    const std::vector<double> pa = knot_partition(base[ax0], base[ax0] + D, h);
    const std::vector<double> pb = knot_partition(base[ax1], base[ax1] + D, h);
    double total = 0.0;
    for (size_t s = 0; s + 1 < pa.size(); ++s) {
        const double ma = 0.5 * (pa[s] + pa[s + 1]), ha = 0.5 * (pa[s + 1] - pa[s]);
        for (size_t r = 0; r + 1 < pb.size(); ++r) {
            const double mb = 0.5 * (pb[r] + pb[r + 1]), hb = 0.5 * (pb[r + 1] - pb[r]);
            double acc = 0.0;
            for (int qa = 0; qa < 4; ++qa)
                for (int qb = 0; qb < 4; ++qb) {
                    double x[3] = {base[0], base[1], base[2]};
                    x[ax0] = ma + ha * gx[qa];
                    x[ax1] = mb + hb * gx[qb];
                    double c[3];
                    spline_velocity(L, x, c);
                    acc += gw[qa] * gw[qb] * c[comp];
                }
            total += ha * hb * acc;
        }
    }
    return total / (D * D);
}

// ============================================================================
// Fixture cache
// ============================================================================

struct Cache {
    // [pair index 0..3 = U, A, B, G][N index 0..1 = 16, 32]
    std::unique_ptr<SplineSet> S[4][2];
    const SplineSet& get(int pair, int N) const {
        const int p = pair == kPairU ? 0 : pair == kPairA ? 1 : pair == kPairB ? 2 : 3;
        return *S[p][N == 16 ? 0 : 1];
    }
};

const int kPairs[4] = {kPairU, kPairA, kPairB, kPairG};
const int kNs[2] = {16, 32};

// ============================================================================
// S1 divergence (every pair, N, Delta) + S4 uniform + S3 additivity
// ============================================================================

void case_s1_s3_s4(const CudaContext& ctx, const Cache& C, TestReport& rep) {
    std::printf("\n=== S1 divergence, S3 additivity, S4 uniform pair ===\n");
    for (int p : kPairs) {
        for (int N : kNs) {
            const SplineSet& S = C.get(p, N);
            const Faces fine = gpu_faces(ctx, S, N);
            const Faces coarse = gpu_faces(ctx, S, N / 2);
            const Faces* both[2] = {&fine, &coarse};
            for (int m = 0; m < 2; ++m) {
                const Faces& F = *both[m];
                const char* dl = m == 0 ? "h" : "2h";
                const double dmod = max_relative_divergence(host_view(F));
                const double down = own_relative_divergence(F);
                rep.check(dmod <= 1e-13, strf("S1/module_divergence/%s/N%d/Delta=%s", pair_name(p), N, dl),
                          strf("max_relative_divergence = %.3e (gate 1e-13)", dmod));
                rep.check(down <= 1e-13, strf("S1/own_divergence/%s/N%d/Delta=%s", pair_name(p), N, dl),
                          strf("recomputed flux-form relative divergence = %.3e (gate 1e-13)", down));
                if (p == kPairU) {
                    double eu = 0.0, ev = 0.0, ew = 0.0;
                    for (size_t c = 0; c < F.u.size(); ++c) {
                        eu = std::max(eu, std::fabs(F.u[c] - 1.0));
                        ev = std::max(ev, std::fabs(F.v[c]));
                        ew = std::max(ew, std::fabs(F.w[c]));
                    }
                    rep.check(eu <= 1e-15 && ev <= 1e-15 && ew <= 1e-15,
                              strf("S4/uniform/N%d/Delta=%s", N, dl),
                              strf("max|u-1| = %.3e max|v| = %.3e max|w| = %.3e (gate 1e-15)", eu, ev, ew));
                }
            }
            if (p == kPairU)
                continue;
            // S3: coarse face == mean of its four fine faces (absolute 1e-13, every face).
            const int nc = N / 2;
            double e3[3] = {0.0, 0.0, 0.0};
            for (int K = 0; K < nc; ++K)
                for (int J = 0; J < nc; ++J)
                    for (int I = 0; I < nc; ++I) {
                        const size_t cc = pidx(I, J, K, nc);
                        double su = 0.0, sv = 0.0, sw = 0.0;
                        for (int a = 0; a < 2; ++a)
                            for (int b = 0; b < 2; ++b) {
                                su += fine.u[pidx(2 * I, 2 * J + a, 2 * K + b, N)];
                                sv += fine.v[pidx(2 * I + a, 2 * J, 2 * K + b, N)];
                                sw += fine.w[pidx(2 * I + a, 2 * J + b, 2 * K, N)];
                            }
                        e3[0] = std::max(e3[0], std::fabs(coarse.u[cc] - 0.25 * su));
                        e3[1] = std::max(e3[1], std::fabs(coarse.v[cc] - 0.25 * sv));
                        e3[2] = std::max(e3[2], std::fabs(coarse.w[cc] - 0.25 * sw));
                    }
            const double e3m = std::max(e3[0], std::max(e3[1], e3[2]));
            rep.check(e3m <= 1e-13, strf("S3/additivity/%s/N%d", pair_name(p), N),
                      strf("max|coarse - mean(4 fine)| u %.3e v %.3e w %.3e (gate 1e-13)", e3[0], e3[1], e3[2]));
        }
    }
}

// ============================================================================
// S2 exactness (pairs G, B; N = 16; Delta = h, 2h; 12 faces per orientation)
// ============================================================================

// Fixed face indices (taken mod n): corners, periodic boundaries and interior faces.
const int kFaces12[12][3] = {{0, 0, 0},  {15, 15, 15}, {1, 2, 3},  {5, 0, 7},  {7, 11, 2}, {3, 15, 9},
                             {12, 4, 14}, {8, 8, 8},   {0, 9, 13}, {14, 1, 6}, {10, 13, 0}, {6, 7, 11}};

void case_s2(const Cache& C, const CudaContext& ctx, TestReport& rep) {
    std::printf("\n=== S2 exactness (direct 2-D Gauss-Legendre, 4x4 per knot cell) ===\n");
    const int pairs[2] = {kPairG, kPairB};
    for (int p : pairs) {
        const SplineSet& S = C.get(p, 16);
        const double h = S.g.dx;
        for (int m = 1; m <= 2; ++m) {
            const int n = 16 / m;
            const Faces F = gpu_faces(ctx, S, n);
            const std::vector<real>* arr[3] = {&F.u, &F.v, &F.w};
            for (int comp = 0; comp < 3; ++comp) {
                double worst = 0.0, worst_val = 0.0;
                bool ok = true;
                for (int f = 0; f < 12; ++f) {
                    const int i = kFaces12[f][0] % n, j = kFaces12[f][1] % n, k = kFaces12[f][2] % n;
                    const double q = face_quadrature(S.host, h, F.D, i, j, k, comp);
                    const double val = (*arr[comp])[pidx(i, j, k, n)];
                    const double ad = std::fabs(val - q);
                    // relative 1e-12; absolute 1e-12 when the face value is below 1e-3.
                    const double err = std::fabs(q) < 1e-3 ? ad : ad / std::fabs(q);
                    if (!(err <= 1e-12))
                        ok = false;
                    if (err > worst || f == 0) {
                        worst = err;
                        worst_val = q;
                    }
                }
                const char* cn = comp == 0 ? "u" : comp == 1 ? "v" : "w";
                rep.check(ok, strf("S2/exactness/%s/N16/Delta=%s/%s", pair_name(p), m == 1 ? "h" : "2h", cn),
                          strf("worst err over 12 faces = %.3e at quadrature value %.6e (gate 1e-12 rel; abs if |val|<1e-3)",
                               worst, worst_val));
            }
        }
    }
}

// ============================================================================
// S5 pair A
// ============================================================================

void case_s5(const CudaContext& ctx, const Cache& C, TestReport& rep) {
    std::printf("\n=== S5 pair A (s1 = a sin 2 pi x1, s2 = 0) ===\n");
    for (int N : kNs) {
        const SplineSet& S = C.get(kPairA, N);
        // E_s: max over 64 fixed hash points (seed 20261006) of |s1_spline - a sin 2 pi x1|.
        double Es = 0.0;
        for (uint64_t q = 0; q < 64; ++q) {
            const real x = inject_uniform01(20261006ULL, q, 0);
            const real y = inject_uniform01(20261006ULL, q, 1);
            const real z = inject_uniform01(20261006ULL, q, 2);
            real val, gxv, gyv, gzv;
            evaluate_point(S.host.s1, x, y, z, val, gxv, gyv, gzv);
            Es = std::max(Es, std::fabs(val - kA * std::sin(kTwoPi * x)));
        }
        std::printf("  N = %d: E_s = %.3e\n", N, Es);
        for (int m = 1; m <= 2; ++m) {
            const int n = N / m;
            const Faces F = gpu_faces(ctx, S, n);
            const double D = F.D;
            double eu = 0.0, ew = 0.0, ejk = 0.0, ev = 0.0, evs = 0.0;
            for (int i = 0; i < n; ++i) {
                const double v0 = F.v[pidx(i, 0, 0, n)];
                const double xi = i * D, xi1 = (i + 1) * D;
                const double vex = -kA * (std::sin(kTwoPi * xi1) - std::sin(kTwoPi * xi)) / D;
                ev = std::max(ev, std::fabs(v0 - vex));
                // Supplementary (not pre-registered): v against the cell mean of the SPLINE,
                // -(s1(x_{i+1}) - s1(x_i)) / Delta, evaluated on the host view (module claim).
                real s_a, s_b, g0, g1v, g2v;
                evaluate_point(S.host.s1, xi1, 0.0, 0.0, s_b, g0, g1v, g2v);
                evaluate_point(S.host.s1, xi, 0.0, 0.0, s_a, g0, g1v, g2v);
                evs = std::max(evs, std::fabs(v0 + (s_b - s_a) / D));
                for (int k = 0; k < n; ++k)
                    for (int j = 0; j < n; ++j) {
                        const size_t c = pidx(i, j, k, n);
                        eu = std::max(eu, std::fabs(F.u[c] - 1.0));
                        ew = std::max(ew, std::fabs(F.w[c]));
                        ejk = std::max(ejk, std::fabs(F.v[c] - v0));
                    }
            }
            const char* dl = m == 1 ? "h" : "2h";
            rep.check(eu <= 1e-14, strf("S5/u_eq_1/N%d/Delta=%s", N, dl), strf("max|u-1| = %.3e (gate 1e-14)", eu));
            rep.check(ew <= 1e-14, strf("S5/w_eq_0/N%d/Delta=%s", N, dl), strf("max|w| = %.3e (gate 1e-14)", ew));
            rep.check(ejk <= 1e-14, strf("S5/v_indep_jk/N%d/Delta=%s", N, dl),
                      strf("max|v[i,j,k] - v[i,0,0]| = %.3e (gate 1e-14)", ejk));
            // The face average of c2 = -d s1/dx1 over a cell of width Delta is
            // -(s1(x_{i+1}) - s1(x_i)) / Delta, so its error against the exact sin is
            // at most 2 E_s / Delta (E_s sampled at 64 points, not the true max);
            // the factor 10 is the pre-registered slack.
            const double gate = 10.0 * Es / D;
            rep.check(ev <= gate, strf("S5/v_vs_exact_cell_mean/N%d/Delta=%s", N, dl),
                      strf("max|v - v_exact| = %.3e, E_s = %.3e, gate 10 E_s/Delta = %.3e", ev, Es, gate));
            rep.check(evs <= 1e-13, strf("S5/supplementary_v_vs_spline_cell_mean/N%d/Delta=%s", N, dl),
                      strf("max|v + (s1(x_{i+1}) - s1(x_i))/Delta| = %.3e (gate 1e-13, not pre-registered)", evs));
        }
    }
}

// ============================================================================
// S6 host == device (pair G)
// ============================================================================

void case_s6(const CudaContext& ctx, const Cache& C, TestReport& rep) {
    std::printf("\n=== S6 host mirror == device (pair G) ===\n");
    for (int N : kNs) {
        const SplineSet& S = C.get(kPairG, N);
        for (int m = 1; m <= 2; ++m) {
            const int n = N / m;
            const Faces Fd = gpu_faces(ctx, S, n);
            const Faces Fh = host_faces(S, n);
            const std::vector<real>* d[3] = {&Fd.u, &Fd.v, &Fd.w};
            const std::vector<real>* hh[3] = {&Fh.u, &Fh.v, &Fh.w};
            double rel[3];
            for (int a = 0; a < 3; ++a) {
                double md = 0.0, mx = 0.0;
                for (size_t c = 0; c < d[a]->size(); ++c) {
                    md = std::max(md, std::fabs((*d[a])[c] - (*hh[a])[c]));
                    mx = std::max(mx, std::fabs((*d[a])[c]));
                }
                rel[a] = mx > 0.0 ? md / mx : md;
            }
            const double rm = std::max(rel[0], std::max(rel[1], rel[2]));
            rep.check(rm <= 1e-13, strf("S6/host_eq_device/G/N%d/Delta=%s", N, m == 1 ? "h" : "2h"),
                      strf("normwise rel u %.3e v %.3e w %.3e (gate 1e-13)", rel[0], rel[1], rel[2]));
            const double dh = own_relative_divergence(Fh);
            rep.check(dh <= 1e-13, strf("S1/host_mirror_divergence/G/N%d/Delta=%s", N, m == 1 ? "h" : "2h"),
                      strf("recomputed relative divergence of host arrays = %.3e (gate 1e-13)", dh));
        }
    }
}

// ============================================================================
// S7 no allocation + determinism
// ============================================================================

void case_s7_determinism(const CudaContext& ctx, const Cache& C, TestReport& rep) {
    std::printf("\n=== S7 no allocation, determinism ===\n");
    const SplineSet& S = C.get(kPairG, 32);
    const Grid3D pg = cube_grid(32);
    StokesFaceFluxWorkspace ws;
    const StokesFaceFluxReport r1 = compute_stokes_face_fluxes(ctx.cuda_stream(), S.dev, pg, ws);
    ctx.synchronize();
    const size_t f0 = free_bytes();
    const StokesFaceFluxReport r2 = compute_stokes_face_fluxes(ctx.cuda_stream(), S.dev, pg, ws);
    ctx.synchronize();
    const size_t f1 = free_bytes();
    rep.check(f0 == f1, "S7/cudaMemGetInfo_unchanged_second_call",
              strf("free before %zu after %zu (report bytes %zu, %zu; nodes %d)", f0, f1, r1.device_bytes,
                   r2.device_bytes, r2.quadrature_nodes_per_interval));
    rep.check(r1.device_bytes == r2.device_bytes && r2.device_bytes == 6 * pg.num_cells() * sizeof(real),
              "S7/report_bytes", strf("%zu (expected 6 n^3 * 8 = %zu)", r2.device_bytes,
                                      6 * pg.num_cells() * sizeof(real)));

    // Determinism: two calls (fresh workspaces) memcmp-identical.
    const Faces A = gpu_faces(ctx, S, 32);
    const Faces B = gpu_faces(ctx, S, 32);
    const size_t nb = A.u.size() * sizeof(real);
    const bool same = std::memcmp(A.u.data(), B.u.data(), nb) == 0 &&
                      std::memcmp(A.v.data(), B.v.data(), nb) == 0 &&
                      std::memcmp(A.w.data(), B.w.data(), nb) == 0;
    rep.check(same, "determinism/memcmp_two_calls/G/N32/Delta=h", "u, v, w bitwise identical");
}

// ============================================================================
// Validation errors
// ============================================================================

template <class Ex, class Fn> bool throws_msg(Fn&& fn, std::string& msg) {
    try {
        fn();
    } catch (const Ex& e) {
        msg = e.what();
        return true;
    } catch (const std::exception& e) {
        msg = std::string("WRONG TYPE: ") + e.what();
        return false;
    }
    msg = "no exception";
    return false;
}

void case_validation(const CudaContext& ctx, const Cache& C, TestReport& rep) {
    std::printf("\n=== validation_errors ===\n");
    const SplineSet& S = C.get(kPairG, 16);
    const Grid3D bad_period(16, 16, 16, 1.0 / 15, 1.0 / 16, 1.0 / 16);
    const Grid3D bad_spacing(16, 16, 16, 1.0 / 16, 0.0, 1.0 / 16);
    const Grid3D bad_spacing_neg(16, 16, 16, 1.0 / 16, 1.0 / 16, -1.0 / 16);
    const Grid3D bad_n(0, 16, 16, 1.0 / 16, 1.0 / 16, 1.0 / 16);
    const Grid3D* grids[4] = {&bad_period, &bad_spacing, &bad_spacing_neg, &bad_n};
    const char* names[4] = {"periods_mismatch", "zero_spacing", "negative_spacing", "nx_lt_1"};
    for (int side = 0; side < 2; ++side) {
        std::string msg[4];
        bool all = true;
        for (int t = 0; t < 4; ++t) {
            bool ok;
            if (side == 0) {
                StokesFaceFluxWorkspace ws;
                ok = throws_msg<std::invalid_argument>(
                    [&] { compute_stokes_face_fluxes(ctx.cuda_stream(), S.dev, *grids[t], ws); }, msg[t]);
            } else {
                std::vector<real> u, v, w;
                ok = throws_msg<std::invalid_argument>(
                    [&] { compute_stokes_face_fluxes_host(S.host, *grids[t], u, v, w); }, msg[t]);
            }
            all = all && ok;
            rep.check(ok, strf("validation_errors/%s/%s", side == 0 ? "device" : "host", names[t]), msg[t]);
        }
        // Distinct messages: periods mismatch vs non-positive spacing vs nx < 1.
        const bool distinct = all && msg[0] != msg[1] && msg[0] != msg[3] && msg[1] != msg[3] && msg[2] != msg[0] &&
                              msg[2] != msg[3];
        rep.check(distinct, strf("validation_errors/%s/distinct_messages", side == 0 ? "device" : "host"));
    }
    std::string m;
    StokesFaceFluxWorkspace fresh;
    const bool ok = throws_msg<std::logic_error>([&] { (void)fresh.view(); }, m);
    rep.check(ok, "validation_errors/view_before_call_logic_error", m);
}

// ============================================================================
// Timing record (information only)
// ============================================================================

void case_timing(const CudaContext& ctx, TestReport& rep) {
    std::printf("\n=== timing_record (N = 128, Delta = h, pair G; information only) ===\n");
    std::unique_ptr<SplineSet> S = build_spline(ctx, kPairG, 128);
    const Grid3D pg = cube_grid(128);
    StokesFaceFluxWorkspace ws;
    compute_stokes_face_fluxes(ctx.cuda_stream(), S->dev, pg, ws); // warm-up / workspace growth
    ctx.synchronize();
    const auto a = std::chrono::steady_clock::now();
    const StokesFaceFluxReport r = compute_stokes_face_fluxes(ctx.cuda_stream(), S->dev, pg, ws);
    ctx.synchronize();
    const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count();
    std::printf("  compute_stokes_face_fluxes 128^3: %.3f ms (workspace %zu bytes)\n", 1e3 * t, r.device_bytes);
    const PeriodicFaceFluxView fv = ws.view();
    Faces F;
    F.n = 128;
    F.D = pg.dx;
    F.u = d2h(fv.u, pg.num_cells());
    F.v = d2h(fv.v, pg.num_cells());
    F.w = d2h(fv.w, pg.num_cells());
    std::printf("  (divergence at 128^3, information: %.3e)\n", own_relative_divergence(F));
    (void)rep;
}

} // namespace

int main() {
    const auto t0 = std::chrono::steady_clock::now();
    TestReport rep;
    std::vector<std::pair<std::string, double>> times;
    auto timed = [&](const char* name, auto&& fn) {
        const auto a = std::chrono::steady_clock::now();
        try {
            fn();
        } catch (const std::exception& e) {
            rep.check(false, std::string(name) + "/unexpected_exception", e.what());
        }
        times.emplace_back(name, std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count());
    };
    try {
        CudaContext ctx(0);
        Cache C;
        bool built = false;
        timed("spline_setup", [&] {
            std::printf("=== spline setup (SF-28 GPU + host prefilter) ===\n");
            for (int pi = 0; pi < 4; ++pi)
                for (int ni = 0; ni < 2; ++ni)
                    C.S[pi][ni] = build_spline(ctx, kPairs[pi], kNs[ni]);
            built = true;
        });
        if (built) {
            timed("S1_S3_S4", [&] { case_s1_s3_s4(ctx, C, rep); });
            timed("S2", [&] { case_s2(C, ctx, rep); });
            timed("S5", [&] { case_s5(ctx, C, rep); });
            timed("S6", [&] { case_s6(ctx, C, rep); });
            timed("S7_determinism", [&] { case_s7_determinism(ctx, C, rep); });
            timed("validation_errors", [&] { case_validation(ctx, C, rep); });
            timed("timing_record", [&] { case_timing(ctx, rep); });
        } else {
            rep.check(false, "spline_setup/complete");
        }
    } catch (const std::exception& e) {
        rep.check(false, "unexpected_exception", e.what());
    }
    const double wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n=== timing_record (information only) ===\n");
    for (const auto& t : times)
        std::printf("  %-40s %8.3f s\n", t.first.c_str(), t.second);
    std::printf("  %-40s %8.3f s\n", "whole executable", wall);
    std::printf("\n%d checks, %d failed, overall %s\n", rep.checks, rep.fails, rep.overall_pass ? "PASS" : "FAIL");
    return rep.overall_pass ? 0 : 1;
}
