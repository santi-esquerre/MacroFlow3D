/**
 * @file periodic_tricubic_bspline_tests.cu
 * @brief SF-28 T02: fast contract tests for the periodic tricubic B-spline
 *        interpolation module (`src/numerics/interpolation/PeriodicTricubicBSpline.cuh`).
 *
 * Standalone ctest-friendly runner (mirrors `tests/stochastic/periodic_gaussian_tests.cu`:
 * printed `[PASS]/[FAIL]` checks, `main` returns 0/1). Fast tier: the whole
 * executable must finish well under a minute.
 *
 * Fixtures and thresholds are PRE-REGISTERED (SF-28 spec "Acceptance
 * thresholds" + orchestrator decisions D-1..D-5) and implemented verbatim; none
 * may be adjusted to make a failing case pass.
 *
 *  - Test field (D-5): f = sin(2 pi x) cos(4 pi y) sin(2 pi z)
 *                          + 0.5 cos(2 pi (x + y)) + 0.25 sin(6 pi z)
 *    on [0,1)^3; grids 16^3, 32^3, 64^3 and 16x32x24, L = 1 per axis.
 *  - Off-node points (D-2): M = 10000, the same points on every grid, from a
 *    stateless splitmix64 hash of the point index (reproducible).
 *  - Wrap points (D-1): dyadic coordinates k / 2^16, shifts q in
 *    {-2,-1,+1,+2,+3} applied to one axis at a time; bitwise comparison.
 *
 * Cases: validation_contract, node_interpolation_condition,
 * constant_partition_of_unity, order_ladder (T1), wrap_bitwise (T2),
 * cpu_gpu_agreement (T3), memory_accounting (T4), timing_record.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/numerics/interpolation/PeriodicTricubicBSpline.cuh"
#include "src/runtime/cuda_check.cuh"
#include "src/runtime/CudaContext.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

using namespace macroflow3d;
using namespace macroflow3d::interpolation;

namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr size_t kM = 10000; // D-2 off-node point count

// ============================================================================
// Bookkeeping
// ============================================================================

struct TestReport {
    bool overall_pass = true;
    int checks = 0;

    void check(bool cond, const std::string& name, const std::string& detail = "") {
        ++checks;
        std::printf("[%s] %s%s%s\n", cond ? "PASS" : "FAIL", name.c_str(),
                    detail.empty() ? "" : "  ", detail.c_str());
        overall_pass = overall_pass && cond;
    }
};

std::string fmt(const char* f, double a) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), f, a);
    return buf;
}

// ============================================================================
// Fixtures
// ============================================================================

// D-5 test field and its analytic gradient.
void field(double x, double y, double z, double& f, double& fx, double& fy, double& fz) {
    const double a = 2.0 * kPi;
    const double sx = std::sin(a * x), cx = std::cos(a * x);
    const double s2y = std::sin(2.0 * a * y), c2y = std::cos(2.0 * a * y);
    const double sz = std::sin(a * z), cz = std::cos(a * z);
    const double sxy = std::sin(a * (x + y)), cxy = std::cos(a * (x + y));
    const double s3z = std::sin(3.0 * a * z), c3z = std::cos(3.0 * a * z);
    f = sx * c2y * sz + 0.5 * cxy + 0.25 * s3z;
    fx = a * cx * c2y * sz - 0.5 * a * sxy;
    fy = -2.0 * a * sx * s2y * sz - 0.5 * a * sxy;
    fz = a * sx * c2y * cz + 0.75 * a * c3z;
}

// L = 1 per axis (h = 1/N; N * (1/N) == 1 exactly for N in {16, 24, 32, 64}).
Grid3D unit_grid(int nx, int ny, int nz) {
    return Grid3D(nx, ny, nz, 1.0 / nx, 1.0 / ny, 1.0 / nz);
}

double centre(int i, double h) {
    return (static_cast<double>(i) + 0.5) * h;
}

// Cell-centred samples of the D-5 field (host).
std::vector<real> sample_field(const Grid3D& g) {
    std::vector<real> s(g.num_cells());
    for (int k = 0; k < g.nz; ++k)
        for (int j = 0; j < g.ny; ++j)
            for (int i = 0; i < g.nx; ++i) {
                double f, fx, fy, fz;
                field(centre(i, g.dx), centre(j, g.dy), centre(k, g.dz), f, fx, fy, fz);
                s[g.idx(i, j, k)] = f;
            }
    return s;
}

// Stateless splitmix64 -> uniform double in [0,1) (53 bits).
uint64_t splitmix64(uint64_t x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
double hash_unit(uint64_t index, uint64_t salt) {
    return static_cast<double>(splitmix64(index ^ (salt * 0xD1B54A32D192ED03ULL)) >> 11) *
           (1.0 / 9007199254740992.0);
}

struct Points {
    std::vector<real> x, y, z;
    size_t size() const { return x.size(); }
    void push(real a, real b, real c) {
        x.push_back(a);
        y.push_back(b);
        z.push_back(c);
    }
};

// D-2: M points, same on every grid (three salts, one per coordinate).
Points offnode_points(size_t m) {
    Points p;
    for (size_t n = 0; n < m; ++n)
        p.push(hash_unit(n, 0x5F28A1ULL), hash_unit(n, 0x5F28B2ULL), hash_unit(n, 0x5F28C3ULL));
    return p;
}

// ============================================================================
// Device helpers
// ============================================================================

void upload(DeviceBuffer<real>& d, const std::vector<real>& h) {
    d.resize(h.size());
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(d.data(), h.data(), h.size() * sizeof(real), cudaMemcpyHostToDevice));
}

std::vector<real> download(const real* d, size_t n) {
    std::vector<real> h(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(real), cudaMemcpyDeviceToHost));
    return h;
}

struct Evaluation {
    std::vector<real> v, gx, gy, gz;
};

// Device buffers for one batched evaluation (allocated before, and outside of,
// the measured evaluations of memory_accounting).
struct EvalBuffers {
    DeviceBuffer<real> px, py, pz, v, gx, gy, gz;
    void set_points(const Points& p) {
        upload(px, p.x);
        upload(py, p.y);
        upload(pz, p.z);
        const size_t n = p.size();
        v.resize(n);
        gx.resize(n);
        gy.resize(n);
        gz.resize(n);
    }
    void launch(const CudaContext& ctx, const PeriodicTricubicBSplineView& view) {
        evaluate_periodic_tricubic_bspline(ctx, view, px.span(), py.span(), pz.span(), v.span(),
                                           gx.span(), gy.span(), gz.span());
    }
};

Evaluation eval_gpu(const CudaContext& ctx, const PeriodicTricubicBSplineView& view,
                    const Points& p) {
    EvalBuffers b;
    b.set_points(p);
    b.launch(ctx, view);
    ctx.synchronize();
    const size_t n = p.size();
    return {download(b.v.data(), n), download(b.gx.data(), n), download(b.gy.data(), n),
            download(b.gz.data(), n)};
}

Evaluation eval_host(const PeriodicTricubicBSplineView& view, const Points& p) {
    const size_t n = p.size();
    Evaluation e{std::vector<real>(n), std::vector<real>(n), std::vector<real>(n),
                 std::vector<real>(n)};
    for (size_t i = 0; i < n; ++i)
        evaluate_point(view, p.x[i], p.y[i], p.z[i], e.v[i], e.gx[i], e.gy[i], e.gz[i]);
    return e;
}

struct GpuSpline {
    DeviceBuffer<real> samples;
    PeriodicTricubicBSplineWorkspace ws;
    PeriodicTricubicBSplineReport report;
};

void build_gpu(const CudaContext& ctx, const Grid3D& g, const std::vector<real>& s,
               GpuSpline& out) {
    upload(out.samples, s);
    out.report = prefilter_periodic_tricubic_bspline(
        ctx, g, DeviceSpan<const real>(out.samples.data(), out.samples.size()), out.ws);
}

std::vector<real> host_coefficients(const Grid3D& g, const std::vector<real>& s) {
    std::vector<real> c(s.size());
    prefilter_periodic_tricubic_bspline_host(g, s.data(), c.data());
    return c;
}

std::string grid_name(const Grid3D& g) {
    return std::to_string(g.nx) + "x" + std::to_string(g.ny) + "x" + std::to_string(g.nz);
}

template <typename F> bool throws_invalid(F&& fn) {
    try {
        fn();
    } catch (const std::invalid_argument& e) {
        std::printf("      invalid_argument: %s\n", e.what());
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// ============================================================================
// Case 1: validation_contract
// ============================================================================

void case_validation_contract(const CudaContext& ctx, TestReport& rep) {
    std::printf("\n=== validation_contract ===\n");
    const Grid3D g = unit_grid(16, 16, 16);
    const std::vector<real> s = sample_field(g);
    DeviceBuffer<real> d;
    upload(d, s);
    PeriodicTricubicBSplineWorkspace ws;

    rep.check(throws_invalid([&] {
                  prefilter_periodic_tricubic_bspline(
                      ctx, g, DeviceSpan<const real>(d.data(), d.size() - 1), ws);
              }),
              "validation_contract/prefilter_wrong_sample_size");

    for (int axis = 0; axis < 3; ++axis) {
        Grid3D bad = unit_grid(16, 16, 16);
        (axis == 0 ? bad.nx : axis == 1 ? bad.ny : bad.nz) = 3;
        const std::string tag = std::string("axis") + static_cast<char>('x' + axis);
        rep.check(throws_invalid([&] {
                      prefilter_periodic_tricubic_bspline(
                          ctx, bad, DeviceSpan<const real>(d.data(), bad.num_cells()), ws);
                  }),
                  "validation_contract/prefilter_N_lt_4_" + tag);
        std::vector<real> out(s.size());
        rep.check(throws_invalid(
                      [&] { prefilter_periodic_tricubic_bspline_host(bad, s.data(), out.data()); }),
                  "validation_contract/host_prefilter_N_lt_4_" + tag);
        rep.check(throws_invalid([&] { make_host_view(bad, s.data()); }),
                  "validation_contract/host_view_N_lt_4_" + tag);
    }

    const real bad_spacings[3] = {0.0, -1.0 / 16.0, std::nan("")};
    const char* bad_names[3] = {"zero", "negative", "nan"};
    for (int axis = 0; axis < 3; ++axis)
        for (int b = 0; b < 3; ++b) {
            Grid3D bad = unit_grid(16, 16, 16);
            (axis == 0 ? bad.dx : axis == 1 ? bad.dy : bad.dz) = bad_spacings[b];
            const std::string tag =
                std::string(bad_names[b]) + "_spacing_axis" + static_cast<char>('x' + axis);
            rep.check(throws_invalid([&] {
                          prefilter_periodic_tricubic_bspline(
                              ctx, bad, DeviceSpan<const real>(d.data(), d.size()), ws);
                      }),
                      "validation_contract/prefilter_" + tag);
            std::vector<real> out(s.size());
            rep.check(throws_invalid([&] {
                          prefilter_periodic_tricubic_bspline_host(bad, s.data(), out.data());
                      }),
                      "validation_contract/host_prefilter_" + tag);
        }

    // Evaluation size rules (needs a valid view).
    GpuSpline sp;
    build_gpu(ctx, g, s, sp);
    const PeriodicTricubicBSplineView view = sp.ws.view();
    DeviceBuffer<real> p10(10), p9(9), o10a(10), o10b(10), o10c(10), o10d(10), o9(9);
    rep.check(throws_invalid([&] {
                  evaluate_periodic_tricubic_bspline(ctx, view, p10.span(), p10.span(), p9.span(),
                                                     o10a.span(), o10b.span(), o10c.span(),
                                                     o10d.span());
              }),
              "validation_contract/evaluate_mismatched_point_spans");
    rep.check(throws_invalid([&] {
                  evaluate_periodic_tricubic_bspline(ctx, view, p10.span(), p10.span(), p10.span(),
                                                     o9.span(), o10b.span(), o10c.span(),
                                                     o10d.span());
              }),
              "validation_contract/evaluate_wrong_value_size");
    rep.check(throws_invalid([&] {
                  evaluate_periodic_tricubic_bspline(ctx, view, p10.span(), p10.span(), p10.span(),
                                                     o10a.span(), o10b.span(), o10c.span(),
                                                     o9.span());
              }),
              "validation_contract/evaluate_wrong_gradient_size");

    bool logic = false;
    try {
        PeriodicTricubicBSplineWorkspace fresh;
        (void)fresh.view();
    } catch (const std::logic_error&) {
        logic = true;
    }
    rep.check(logic, "validation_contract/view_before_prefilter_throws_logic_error");
}

// ============================================================================
// Case 2: node_interpolation_condition
// ============================================================================

void case_node_interpolation(const CudaContext& ctx, TestReport& rep) {
    std::printf("\n=== node_interpolation_condition ===\n");
    for (const Grid3D& g : {unit_grid(16, 16, 16), unit_grid(16, 32, 24)}) {
        const std::vector<real> s = sample_field(g);
        double fmax = 0.0;
        for (real v : s)
            fmax = std::max(fmax, std::fabs(v));
        Points nodes;
        for (int k = 0; k < g.nz; ++k)
            for (int j = 0; j < g.ny; ++j)
                for (int i = 0; i < g.nx; ++i)
                    nodes.push(centre(i, g.dx), centre(j, g.dy), centre(k, g.dz));

        GpuSpline sp;
        build_gpu(ctx, g, s, sp);
        const Evaluation eg = eval_gpu(ctx, sp.ws.view(), nodes);
        const std::vector<real> ch = host_coefficients(g, s);
        const Evaluation ec = eval_host(make_host_view(g, ch.data()), nodes);

        double eg_max = 0.0, ec_max = 0.0;
        for (size_t n = 0; n < s.size(); ++n) {
            eg_max = std::max(eg_max, std::fabs(eg.v[n] - s[n]));
            ec_max = std::max(ec_max, std::fabs(ec.v[n] - s[n]));
        }
        const double rg = eg_max / fmax, rc = ec_max / fmax;
        rep.check(rg <= 1e-12, "node_interpolation_condition/gpu_" + grid_name(g),
                  fmt("max|s-f|/max|f| = %.3e (gate 1e-12)", rg));
        rep.check(rc <= 1e-12, "node_interpolation_condition/cpu_" + grid_name(g),
                  fmt("max|s-f|/max|f| = %.3e (gate 1e-12)", rc));
    }
}

// ============================================================================
// Case 3: constant_partition_of_unity
// ============================================================================

void case_constant(const CudaContext& ctx, TestReport& rep, const Points& pts) {
    std::printf("\n=== constant_partition_of_unity ===\n");
    const Grid3D g = unit_grid(16, 32, 24);
    const std::vector<real> s(g.num_cells(), 3.0);
    Points p;
    for (size_t n = 0; n < 1000; ++n)
        p.push(pts.x[n], pts.y[n], pts.z[n]);

    GpuSpline sp;
    build_gpu(ctx, g, s, sp);
    const Evaluation eg = eval_gpu(ctx, sp.ws.view(), p);
    const std::vector<real> ch = host_coefficients(g, s);
    const Evaluation ec = eval_host(make_host_view(g, ch.data()), p);

    auto measure = [&](const Evaluation& e, double& vmax, double& gmax) {
        vmax = gmax = 0.0;
        for (size_t n = 0; n < p.size(); ++n) {
            vmax = std::max(vmax, std::fabs(e.v[n] - 3.0));
            gmax = std::max(gmax,
                            std::sqrt(e.gx[n] * e.gx[n] + e.gy[n] * e.gy[n] + e.gz[n] * e.gz[n]));
        }
    };
    double gv, gg, cv, cg;
    measure(eg, gv, gg);
    measure(ec, cv, cg);
    rep.check(gv <= 1e-14, "constant_partition_of_unity/gpu_value",
              fmt("max|s-3| = %.3e (gate 1e-14)", gv));
    rep.check(gg <= 1e-13, "constant_partition_of_unity/gpu_gradient",
              fmt("max|grad s| = %.3e (gate 1e-13)", gg));
    rep.check(cv <= 1e-14, "constant_partition_of_unity/cpu_value",
              fmt("max|s-3| = %.3e (gate 1e-14)", cv));
    rep.check(cg <= 1e-13, "constant_partition_of_unity/cpu_gradient",
              fmt("max|grad s| = %.3e (gate 1e-13)", cg));
}

// ============================================================================
// Case 4: order_ladder (T1)
// ============================================================================

void case_order_ladder(const CudaContext& ctx, TestReport& rep, const Points& pts) {
    std::printf("\n=== order_ladder (T1) ===\n");
    const int Ns[3] = {16, 32, 64};
    double E[4][3]; // [measure][level]: val_max, val_rms, grad_max, grad_rms
    const char* names[4] = {"E_val_max", "E_val_rms", "E_grad_max", "E_grad_rms"};
    for (int l = 0; l < 3; ++l) {
        const Grid3D g = unit_grid(Ns[l], Ns[l], Ns[l]);
        GpuSpline sp;
        build_gpu(ctx, g, sample_field(g), sp);
        const Evaluation e = eval_gpu(ctx, sp.ws.view(), pts);
        double vmax = 0, vss = 0, gmax = 0, gss = 0;
        for (size_t n = 0; n < pts.size(); ++n) {
            double f, fx, fy, fz;
            field(pts.x[n], pts.y[n], pts.z[n], f, fx, fy, fz);
            const double ev = std::fabs(e.v[n] - f);
            const double dx = e.gx[n] - fx, dy = e.gy[n] - fy, dz = e.gz[n] - fz;
            const double eg2 = dx * dx + dy * dy + dz * dz; // Euclidean per point
            vmax = std::max(vmax, ev);
            vss += ev * ev;
            gmax = std::max(gmax, std::sqrt(eg2));
            gss += eg2;
        }
        E[0][l] = vmax;
        E[1][l] = std::sqrt(vss / static_cast<double>(pts.size()));
        E[2][l] = gmax;
        E[3][l] = std::sqrt(gss / static_cast<double>(pts.size()));
        std::printf("  N=%2d: E_val_max=%.6e E_val_rms=%.6e E_grad_max=%.6e E_grad_rms=%.6e\n",
                    Ns[l], E[0][l], E[1][l], E[2][l], E[3][l]);
    }
    for (int m = 0; m < 4; ++m) {
        const double p1 = std::log2(E[m][0] / E[m][1]);
        const double p2 = std::log2(E[m][1] / E[m][2]);
        const double gate = (m < 2) ? 3.8 : 2.8;
        const double pmin = std::min(p1, p2);
        char buf[200];
        std::snprintf(buf, sizeof(buf), "p_16_32=%.4f p_32_64=%.4f min=%.4f (gate >= %.1f)", p1, p2,
                      pmin, gate);
        rep.check(pmin >= gate, std::string("order_ladder/") + names[m], buf);
    }
}

// ============================================================================
// Case 5: wrap_bitwise (T2)
// ============================================================================

void case_wrap_bitwise(const CudaContext& ctx, TestReport& rep) {
    std::printf("\n=== wrap_bitwise (T2) ===\n");
    // Dyadic coordinates k / 2^16 (D-1): interior, first half-cell and last
    // half-cell of every fixture grid (the half-cell of N = 16/24/32/64 is
    // 2048/1365.33/1024/512 units of 2^-16; 65536 - 1365 = 64171).
    const int ks[] = {0,    1,     37,    511,   512,   1023,  1024,  1365, 2047,
                      2048, 32768, 63488, 64171, 64512, 65024, 65499, 65535};
    // Fixed values of the two other coordinates (dyadic, exact).
    const double others[2][2] = {{0.3125, 0.71875}, {1.0 / 65536.0, 65535.0 / 65536.0}};
    const int qs[] = {-2, -1, 1, 2, 3};

    // Base/shift pairs: entries 2n (base) and 2n+1 (shifted).
    Points pairs;
    for (int axis = 0; axis < 3; ++axis)
        for (int k : ks)
            for (const auto& o : others)
                for (int q : qs) {
                    const double a = static_cast<double>(k) / 65536.0;
                    double base[3], shifted[3];
                    for (int d = 0, oi = 0; d < 3; ++d) {
                        if (d == axis) {
                            base[d] = a;
                            shifted[d] = a + static_cast<double>(q); // exact (dyadic, L = 1)
                        } else {
                            base[d] = shifted[d] = o[oi++];
                        }
                    }
                    pairs.push(base[0], base[1], base[2]);
                    pairs.push(shifted[0], shifted[1], shifted[2]);
                }
    const size_t npairs = pairs.size() / 2;

    auto mismatches = [&](const Evaluation& e) {
        size_t bad = 0;
        for (size_t n = 0; n < npairs; ++n) {
            const size_t b = 2 * n, s = b + 1;
            if (std::memcmp(&e.v[b], &e.v[s], sizeof(real)) != 0 ||
                std::memcmp(&e.gx[b], &e.gx[s], sizeof(real)) != 0 ||
                std::memcmp(&e.gy[b], &e.gy[s], sizeof(real)) != 0 ||
                std::memcmp(&e.gz[b], &e.gz[s], sizeof(real)) != 0)
                ++bad;
        }
        return bad;
    };

    for (const Grid3D& g : {unit_grid(16, 16, 16), unit_grid(32, 32, 32), unit_grid(64, 64, 64),
                            unit_grid(16, 32, 24)}) {
        const std::vector<real> s = sample_field(g);
        GpuSpline sp;
        build_gpu(ctx, g, s, sp);
        const size_t bg = mismatches(eval_gpu(ctx, sp.ws.view(), pairs));
        const std::vector<real> ch = host_coefficients(g, s);
        const size_t bc = mismatches(eval_host(make_host_view(g, ch.data()), pairs));
        rep.check(bg == 0, "wrap_bitwise/gpu_" + grid_name(g),
                  std::to_string(bg) + " mismatches / " + std::to_string(npairs) +
                      " pairs (gate 0)");
        rep.check(bc == 0, "wrap_bitwise/cpu_" + grid_name(g),
                  std::to_string(bc) + " mismatches / " + std::to_string(npairs) +
                      " pairs (gate 0)");
    }
}

// ============================================================================
// Case 6: cpu_gpu_agreement (T3, D-3 normwise)
// ============================================================================

void normwise(const Evaluation& a, const Evaluation& ref, double& rv, double& rg) {
    double dv = 0, sv = 0, dg = 0, sg = 0;
    for (size_t n = 0; n < ref.v.size(); ++n) {
        dv = std::max(dv, std::fabs(a.v[n] - ref.v[n]));
        sv = std::max(sv, std::fabs(ref.v[n]));
        const double ex = a.gx[n] - ref.gx[n], ey = a.gy[n] - ref.gy[n], ez = a.gz[n] - ref.gz[n];
        dg = std::max(dg, std::sqrt(ex * ex + ey * ey + ez * ez));
        sg = std::max(
            sg, std::sqrt(ref.gx[n] * ref.gx[n] + ref.gy[n] * ref.gy[n] + ref.gz[n] * ref.gz[n]));
    }
    rv = dv / sv;
    rg = dg / sg;
}

void case_cpu_gpu_agreement(const CudaContext& ctx, TestReport& rep, const Points& pts) {
    std::printf("\n=== cpu_gpu_agreement (T3) ===\n");
    for (const Grid3D& g : {unit_grid(32, 32, 32), unit_grid(16, 32, 24)}) {
        const std::string gn = grid_name(g);
        const std::vector<real> s = sample_field(g);
        GpuSpline sp;
        build_gpu(ctx, g, s, sp);
        const Evaluation eg = eval_gpu(ctx, sp.ws.view(), pts);
        const std::vector<real> ch = host_coefficients(g, s);
        const Evaluation ec = eval_host(make_host_view(g, ch.data()), pts);
        const std::vector<real> cg = download(sp.ws.coefficients.data(), g.num_cells());
        const Evaluation egh = eval_host(make_host_view(g, cg.data()), pts);

        double dc = 0, sc = 0;
        for (size_t n = 0; n < cg.size(); ++n) {
            dc = std::max(dc, std::fabs(cg[n] - ch[n]));
            sc = std::max(sc, std::fabs(ch[n]));
        }
        std::printf("  %s: max|c_gpu - c_cpu| / max|c_cpu| = %.3e (reported, no gate)\n",
                    gn.c_str(), dc / sc);

        double av, ag, bv, bg;
        normwise(eg, ec, av, ag);  // (a) GPU pipeline vs CPU pipeline (reference CPU)
        normwise(eg, egh, bv, bg); // (b) GPU eval vs host eval on GPU coefficients
        rep.check(av <= 1e-13, "cpu_gpu_agreement/a_value_" + gn,
                  fmt("GPU pipeline vs CPU pipeline: %.3e (gate 1e-13)", av));
        rep.check(ag <= 1e-13, "cpu_gpu_agreement/a_gradient_" + gn,
                  fmt("GPU pipeline vs CPU pipeline: %.3e (gate 1e-13)", ag));
        rep.check(bv <= 1e-13, "cpu_gpu_agreement/b_value_" + gn,
                  fmt("GPU eval vs host eval on GPU coeffs: %.3e (gate 1e-13)", bv));
        rep.check(bg <= 1e-13, "cpu_gpu_agreement/b_gradient_" + gn,
                  fmt("GPU eval vs host eval on GPU coeffs: %.3e (gate 1e-13)", bg));
    }
}

// ============================================================================
// Case 7: memory_accounting (T4)
// ============================================================================

size_t free_bytes() {
    size_t f = 0, t = 0;
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&f, &t));
    return f;
}

void case_memory_accounting(const CudaContext& ctx, TestReport& rep, const Points& pts) {
    std::printf("\n=== memory_accounting (T4) ===\n");
    const Grid3D g = unit_grid(64, 64, 64);
    const std::vector<real> s = sample_field(g);
    DeviceBuffer<real> samples;
    upload(samples, s);
    EvalBuffers b;
    b.set_points(pts);
    PeriodicTricubicBSplineWorkspace ws;
    ctx.synchronize();

    const size_t before_prefilter = free_bytes();
    const PeriodicTricubicBSplineReport r = prefilter_periodic_tricubic_bspline(
        ctx, g, DeviceSpan<const real>(samples.data(), samples.size()), ws);
    ctx.synchronize();
    const size_t after_prefilter = free_bytes();
    const long long delta_prefilter =
        static_cast<long long>(before_prefilter) - static_cast<long long>(after_prefilter);

    std::printf("  report: coefficient_bytes=%zu spectrum_bytes=%zu cufft_work_area_bytes=%zu "
                "total_device_bytes=%zu\n",
                r.coefficient_bytes, r.spectrum_bytes, r.cufft_work_area_bytes,
                r.total_device_bytes);
    std::printf("  capacities: coefficients=%zu spectrum=%zu\n", ws.coefficients.capacity(),
                ws.spectrum.capacity());
    std::printf("  cudaMemGetInfo delta across first prefilter = %lld bytes (owned %zu bytes; "
                "allocation granularity, no gate)\n",
                delta_prefilter, r.coefficient_bytes + r.spectrum_bytes);

    rep.check(r.total_device_bytes ==
                  r.coefficient_bytes + r.spectrum_bytes + r.cufft_work_area_bytes,
              "memory_accounting/total_is_sum_of_fields");
    rep.check(r.coefficient_bytes == ws.coefficients.capacity() * 8,
              "memory_accounting/coefficient_bytes_eq_capacity_x8");
    rep.check(r.spectrum_bytes == ws.spectrum.capacity() * 16,
              "memory_accounting/spectrum_bytes_eq_capacity_x16");
    rep.check(ws.coefficients.capacity() == g.num_cells() &&
                  ws.spectrum.capacity() == static_cast<size_t>(g.nx / 2 + 1) *
                                                static_cast<size_t>(g.ny) *
                                                static_cast<size_t>(g.nz),
              "memory_accounting/capacities_match_grid");

    // The evaluation kernel has already been launched by earlier cases, so
    // lazy module loading cannot perturb the measurement below.
    const PeriodicTricubicBSplineView view = ws.view();
    bool unchanged = true;
    constexpr int kCalls = 4;
    for (int c = 0; c < kCalls; ++c) {
        ctx.synchronize();
        const size_t f0 = free_bytes();
        b.launch(ctx, view);
        ctx.synchronize();
        const size_t f1 = free_bytes();
        std::printf("  evaluation call %d: free before=%zu after=%zu delta=%lld\n", c + 1, f0, f1,
                    static_cast<long long>(f0) - static_cast<long long>(f1));
        unchanged = unchanged && (f0 == f1);
    }
    rep.check(unchanged, "memory_accounting/cudaMemGetInfo_unchanged_across_evaluations",
              std::to_string(kCalls) + " consecutive batched evaluations of " +
                  std::to_string(pts.size()) + " points");
}

} // namespace

int main() {
    const auto t0 = std::chrono::steady_clock::now();
    TestReport rep;
    try {
        CudaContext ctx(0);
        const Points pts = offnode_points(kM);

        case_validation_contract(ctx, rep);
        case_node_interpolation(ctx, rep);
        case_constant(ctx, rep, pts);
        case_order_ladder(ctx, rep, pts);
        case_wrap_bitwise(ctx, rep);
        case_cpu_gpu_agreement(ctx, rep, pts);
        case_memory_accounting(ctx, rep, pts);
    } catch (const std::exception& e) {
        rep.check(false, "unexpected_exception", e.what());
    }
    const double wall =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("\n=== timing_record ===\n  wall time = %.3f s (no gate; fast tier)\n", wall);
    std::printf("\n%d checks, overall %s\n", rep.checks, rep.overall_pass ? "PASS" : "FAIL");
    return rep.overall_pass ? 0 : 1;
}
