/**
 * @file nonperiodic_x_bspline_tests.cu
 * @brief SF-41 N1: threshold A program (P-SPLINE, pre-registration RR3) for the
 *        non-periodic-in-x1 tricubic B-spline (NonPeriodicXTricubicBSpline).
 *
 * Usage:
 *   interpolation_nonperiodic_x_tests                (fast tier, registered in ctest)
 *   interpolation_nonperiodic_x_tests --full         (complete RR3 list, V100 only)
 *   interpolation_nonperiodic_x_tests --only <group> (one group, in the selected tier)
 * Groups: node pou repro ladder cpugpu bitwise noalloc shift domain memory
 *
 * Output: one line per check,
 *   A_CHECK name=<name> box=<nx>x<ny>x<nz> side=<host|device|both> value=<%.6e> gate=<%.3e> verdict=<PASS|FAIL|RECORDED>
 * (gate=nan for RECORDED), per-level ladder errors (A_LADDER_ERR, 7 significant digits),
 * orders (A_LADDER_ORDER), measured/emulated ratios (A_EMUL), and a final
 *   A_VERDICT all_pass=<0|1> checks_pass=<n> checks_fail=<n>
 * Exit code 0 iff every gated check of the selected tier/groups passes.
 * Sense of a gate: names ending `_order_ge_*` pass iff value >= gate; all other
 * gated checks pass iff value <= gate.
 *
 * Reading choices (documented, see the report): MARGIN uses a fresh
 * std::mt19937_64(20261025) per box for its transverse coordinates (one point
 * per x value, 64 points); PS draws x1, x2, x3 per point in that order;
 * "side=host" = CPU-mirror prefilter + host evaluation, "side=device" = GPU
 * prefilter + device evaluation.
 *
 * Fast tier: every grid has <= 32768 cells (32^3); the checks that the RR3 list
 * runs on larger boxes are reduced (reported in the A_TIER lines): partition of
 * unity 32x16x16 (RR3: + 64x32x32, 128x64x64); CPU vs GPU 32x16x16 and 64x16x16
 * (RR3: 32x16x16, 64x32x32); Hessian bitwise 32x16x16 (RR3: 64x32x32); shift
 * 32x16x16 (RR3: + 64x32x32); no-allocation 32x16x16 (RR3: 64x32x32); the 3-D
 * order ladder runs on two levels 16x8x8, 32x16x16 and is RECORDED only (the
 * pair is pre-asymptotic); instead the x-only sub-ladder (boxes nx x 4 x 4,
 * <= 4096 cells) is GATED in the fast tier with the A gates (it is RECORDED in
 * the full tier, where the 3-D ladder is gated).
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/numerics/interpolation/NonPeriodicXTricubicBSpline.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace macroflow3d;
using namespace macroflow3d::interpolation;

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------------
// Reporting
// ---------------------------------------------------------------------------

enum class Kind { LE, GE, REC };

int g_pass = 0, g_fail = 0;

std::string box_str(int nx, int ny, int nz) {
    return std::to_string(nx) + "x" + std::to_string(ny) + "x" + std::to_string(nz);
}

void emit(const std::string& name, const std::string& box, const char* side, double value,
          double gate, Kind kind) {
    const char* verdict = "RECORDED";
    if (kind == Kind::LE) {
        verdict = (value <= gate) ? "PASS" : "FAIL";
    } else if (kind == Kind::GE) {
        verdict = (value >= gate) ? "PASS" : "FAIL";
    }
    if (kind != Kind::REC) {
        if (verdict[0] == 'P')
            ++g_pass;
        else
            ++g_fail;
    }
    std::printf("A_CHECK name=%s box=%s side=%s value=%.6e gate=%.3e verdict=%s\n", name.c_str(),
                box.c_str(), side, value, kind == Kind::REC ? kNaN : gate, verdict);
}

// ---------------------------------------------------------------------------
// Boxes, points, evaluation containers
// ---------------------------------------------------------------------------

struct Box {
    int nx, ny, nz;
    double h() const { return 1.0 / ny; }
    double L1() const { return nx * h(); }
    Grid3D grid() const { return Grid3D(nx, ny, nz, h(), h(), h()); }
    std::string str() const { return box_str(nx, ny, nz); }
    size_t cells() const { return static_cast<size_t>(nx) * ny * nz; }
};

struct Pts {
    std::vector<real> x, y, z;
    size_t n() const { return x.size(); }
    void push(double a, double b, double c) {
        x.push_back(a);
        y.push_back(b);
        z.push_back(c);
    }
    void append(const Pts& o) {
        x.insert(x.end(), o.x.begin(), o.x.end());
        y.insert(y.end(), o.y.begin(), o.y.end());
        z.insert(z.end(), o.z.begin(), o.z.end());
    }
};

// PS(L1): 1e4 points uniform in [0,L1]x[0,1)^2, then 2000 per face uniform in
// [0,1/16]x[0,1)^2 and [L1-1/16,L1]x[0,1)^2, one std::mt19937_64(20261025).
Pts make_ps(double L1) {
    std::mt19937_64 gen(20261025);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    auto draw = [&](double a, double b) {
        std::uniform_real_distribution<double> ux(a, b);
        return ux(gen);
    };
    Pts p;
    for (int i = 0; i < 10000; ++i) {
        const double x = draw(0.0, L1), y = u01(gen), z = u01(gen);
        p.push(x, y, z);
    }
    for (int i = 0; i < 2000; ++i) {
        const double x = draw(0.0, 1.0 / 16.0), y = u01(gen), z = u01(gen);
        p.push(x, y, z);
    }
    for (int i = 0; i < 2000; ++i) {
        const double x = draw(L1 - 1.0 / 16.0, L1), y = u01(gen), z = u01(gen);
        p.push(x, y, z);
    }
    return p;
}

// MARGIN(box): x1 = -h r and L1 + h r, r = (j+1/2)/64, j = 0..31; transverse
// coordinates from a fresh std::mt19937_64(20261025).
Pts make_margin(const Box& b) {
    std::mt19937_64 gen(20261025);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    Pts p;
    for (int side = 0; side < 2; ++side)
        for (int j = 0; j < 32; ++j) {
            const double r = (j + 0.5) / 64.0;
            const double x = side == 0 ? -b.h() * r : b.L1() + b.h() * r;
            const double y = u01(gen), z = u01(gen);
            p.push(x, y, z);
        }
    return p;
}

struct Ev {
    std::vector<real> v, gx, gy, gz, H; // H has 6 entries per point when present
    std::vector<int> inside;
    size_t n() const { return v.size(); }
    bool has_h() const { return !H.empty(); }
};

// ---------------------------------------------------------------------------
// GPU helpers
// ---------------------------------------------------------------------------

template <typename T> void upload(DeviceBuffer<T>& d, const std::vector<T>& h) {
    d.resize(h.size());
    if (!h.empty())
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(d.data(), h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
}

template <typename T> std::vector<T> download(const CudaContext& ctx, const DeviceBuffer<T>& d, size_t n) {
    ctx.synchronize();
    std::vector<T> h(n);
    if (n)
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), d.data(), n * sizeof(T), cudaMemcpyDeviceToHost));
    return h;
}

struct GpuSpline {
    NonPeriodicXTricubicBSplineWorkspace ws;
    NonPeriodicXTricubicBSplineReport report;
    Grid3D grid;
    DeviceBuffer<real> samples;
};

void gpu_build(const CudaContext& ctx, const Grid3D& g, const std::vector<real>& f, GpuSpline& s) {
    s.grid = g;
    upload(s.samples, f);
    s.report = prefilter_nonperiodic_x_tricubic_bspline(ctx, g, s.samples.span(), s.ws);
}

std::vector<real> gpu_coefficients(const CudaContext& ctx, const GpuSpline& s) {
    const size_t n = (static_cast<size_t>(s.grid.nx) + 4) * s.grid.ny * s.grid.nz;
    return download(ctx, s.ws.coefficients, n);
}

std::vector<real> host_coefficients(const Grid3D& g, const std::vector<real>& f) {
    std::vector<real> c((static_cast<size_t>(g.nx) + 4) * g.ny * g.nz);
    prefilter_nonperiodic_x_tricubic_bspline_host(g, f.data(), c.data());
    return c;
}

Ev eval_host(const NonPeriodicXTricubicBSplineView& v, const Pts& p, bool hess) {
    Ev e;
    const size_t n = p.n();
    e.v.resize(n);
    e.gx.resize(n);
    e.gy.resize(n);
    e.gz.resize(n);
    e.inside.resize(n);
    if (hess)
        e.H.resize(6 * n);
    for (size_t i = 0; i < n; ++i) {
        bool in;
        if (hess)
            in = evaluate_point_hessian(v, p.x[i], p.y[i], p.z[i], e.v[i], e.gx[i], e.gy[i], e.gz[i],
                                        &e.H[6 * i]);
        else
            in = evaluate_point(v, p.x[i], p.y[i], p.z[i], e.v[i], e.gx[i], e.gy[i], e.gz[i]);
        e.inside[i] = in ? 1 : 0;
    }
    return e;
}

Ev eval_device(const CudaContext& ctx, const NonPeriodicXTricubicBSplineView& v, const Pts& p,
               bool hess) {
    const size_t n = p.n();
    DeviceBuffer<real> px, py, pz, ov, ogx, ogy, ogz, oh;
    DeviceBuffer<int> in;
    upload(px, p.x);
    upload(py, p.y);
    upload(pz, p.z);
    ov.resize(n);
    ogx.resize(n);
    ogy.resize(n);
    ogz.resize(n);
    in.resize(n);
    oh.resize(6 * n);
    if (hess)
        evaluate_nonperiodic_x_tricubic_bspline_hessian(
            ctx, v, px.span(), py.span(), pz.span(), ov.span(), ogx.span(), ogy.span(), ogz.span(),
            oh.span(), in.span());
    else
        evaluate_nonperiodic_x_tricubic_bspline(ctx, v, px.span(), py.span(), pz.span(), ov.span(),
                                                ogx.span(), ogy.span(), ogz.span(), in.span());
    Ev e;
    e.v = download(ctx, ov, n);
    e.gx = download(ctx, ogx, n);
    e.gy = download(ctx, ogy, n);
    e.gz = download(ctx, ogz, n);
    e.inside = download(ctx, in, n);
    if (hess)
        e.H = download(ctx, oh, 6 * n);
    return e;
}

// ---------------------------------------------------------------------------
// Analytic fields: v, gradient, Hessian (xx yy zz xy xz yz)
// ---------------------------------------------------------------------------

using Field = std::function<void(double, double, double, double&, double[3], double[6])>;

Field ladder_field(double amp) {
    return [amp](double x, double y, double z, double& v, double g[3], double H[6]) {
        const double e = std::exp(x), s = std::sin(3 * x), c = std::cos(3 * x);
        const double g0 = e * s, g1 = e * (s + 3 * c), g2 = e * (-8 * s + 6 * c);
        const double sy = std::sin(kTwoPi * y), cy = std::cos(kTwoPi * y);
        const double sz = std::sin(kTwoPi * z), cz = std::cos(kTwoPi * z);
        const double q = 1 + amp * sy * cz;
        const double qy = amp * kTwoPi * cy * cz, qz = -amp * kTwoPi * sy * sz;
        const double qyy = -amp * kTwoPi * kTwoPi * sy * cz, qzz = qyy;
        const double qyz = -amp * kTwoPi * kTwoPi * cy * sz;
        v = g0 * q;
        g[0] = g1 * q;
        g[1] = g0 * qy;
        g[2] = g0 * qz;
        H[0] = g2 * q;
        H[1] = g0 * qyy;
        H[2] = g0 * qzz;
        H[3] = g1 * qy;
        H[4] = g1 * qz;
        H[5] = g0 * qyz;
    };
}

Field xonly_field() { return ladder_field(0.0); }

Field poly_field(bool cubic, double L1) {
    return [cubic, L1](double x, double, double, double& v, double g[3], double H[6]) {
        const double s = x - 0.5 * L1;
        if (cubic) {
            v = 1 + 0.5 * s - 0.7 * s * s + 0.3 * s * s * s;
            g[0] = 0.5 - 1.4 * s + 0.9 * s * s;
            H[0] = -1.4 + 1.8 * s;
        } else {
            v = s * s;
            g[0] = 2 * s;
            H[0] = 2.0;
        }
        g[1] = g[2] = 0;
        for (int i = 1; i < 6; ++i)
            H[i] = 0;
    };
}

Field const_field(double c) {
    return [c](double, double, double, double& v, double g[3], double H[6]) {
        v = c;
        g[0] = g[1] = g[2] = 0;
        for (int i = 0; i < 6; ++i)
            H[i] = 0;
    };
}

std::vector<real> sample_field(const Box& b, const Field& f) {
    std::vector<real> s(b.cells());
    const double h = b.h();
    for (int k = 0; k < b.nz; ++k)
        for (int j = 0; j < b.ny; ++j)
            for (int i = 0; i < b.nx; ++i) {
                double v, g[3], H[6];
                f((i + 0.5) * h, (j + 0.5) * h, (k + 0.5) * h, v, g, H);
                s[i + static_cast<size_t>(b.nx) * (j + static_cast<size_t>(b.ny) * k)] = v;
            }
    return s;
}

Ev exact_ev(const Pts& p, const Field& f) {
    Ev e;
    const size_t n = p.n();
    e.v.resize(n);
    e.gx.resize(n);
    e.gy.resize(n);
    e.gz.resize(n);
    e.H.resize(6 * n);
    for (size_t i = 0; i < n; ++i) {
        double g[3];
        double v;
        f(p.x[i], p.y[i], p.z[i], v, g, &e.H[6 * i]);
        e.v[i] = v;
        e.gx[i] = g[0];
        e.gy[i] = g[1];
        e.gz[i] = g[2];
    }
    return e;
}

// ---------------------------------------------------------------------------
// Error metrics
// ---------------------------------------------------------------------------

inline double gnorm(const Ev& e, size_t i) {
    return std::sqrt(e.gx[i] * e.gx[i] + e.gy[i] * e.gy[i] + e.gz[i] * e.gz[i]);
}
inline double gdiff(const Ev& a, const Ev& r, size_t i) {
    const double dx = a.gx[i] - r.gx[i], dy = a.gy[i] - r.gy[i], dz = a.gz[i] - r.gz[i];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}
inline double hdiff(const Ev& a, const Ev& r, size_t i) {
    double m = 0;
    for (int c = 0; c < 6; ++c)
        m = std::max(m, std::fabs(a.H[6 * i + c] - r.H[6 * i + c]));
    return m;
}
inline double hmag(const Ev& r, size_t i) {
    double m = 0;
    for (int c = 0; c < 6; ++c)
        m = std::max(m, std::fabs(r.H[6 * i + c]));
    return m;
}

struct Rel {
    double v = 0, g = 0, H = 0;
};

// error over the points selected by `mask` (all when empty), divided by the max
// magnitude of the reference over ALL points of the set.
Rel rel_errors(const Ev& a, const Ev& ref, const std::vector<char>& mask = {}) {
    double dv = 0, dg = 0, dh = 0, sv = 0, sg = 0, sh = 0;
    for (size_t i = 0; i < ref.n(); ++i) {
        sv = std::max(sv, std::fabs(ref.v[i]));
        sg = std::max(sg, gnorm(ref, i));
        if (ref.has_h())
            sh = std::max(sh, hmag(ref, i));
        if (!mask.empty() && !mask[i])
            continue;
        dv = std::max(dv, std::fabs(a.v[i] - ref.v[i]));
        dg = std::max(dg, gdiff(a, ref, i));
        if (ref.has_h() && a.has_h())
            dh = std::max(dh, hdiff(a, ref, i));
    }
    Rel r;
    r.v = dv / sv;
    r.g = dg / sg;
    r.H = (ref.has_h() && a.has_h()) ? dh / sh : kNaN;
    return r;
}

struct AbsErr {
    double val_max = 0, val_rms = 0, grad_max = 0, grad_rms = 0, hess_max = 0;
};

AbsErr abs_errors(const Ev& a, const Ev& ref) {
    AbsErr e;
    double sv = 0, sg = 0;
    for (size_t i = 0; i < ref.n(); ++i) {
        const double dv = std::fabs(a.v[i] - ref.v[i]);
        const double dg = gdiff(a, ref, i);
        e.val_max = std::max(e.val_max, dv);
        e.grad_max = std::max(e.grad_max, dg);
        e.hess_max = std::max(e.hess_max, hdiff(a, ref, i));
        sv += dv * dv;
        sg += dg * dg;
    }
    e.val_rms = std::sqrt(sv / ref.n());
    e.grad_rms = std::sqrt(sg / ref.n());
    return e;
}

size_t memcmp_mismatches(const Ev& a, const Ev& b) {
    size_t bad = 0;
    for (size_t i = 0; i < a.n(); ++i)
        if (std::memcmp(&a.v[i], &b.v[i], 8) || std::memcmp(&a.gx[i], &b.gx[i], 8) ||
            std::memcmp(&a.gy[i], &b.gy[i], 8) || std::memcmp(&a.gz[i], &b.gz[i], 8))
            ++bad;
    return bad;
}

// Both sides for a box/field: host (CPU mirror + host eval) and device.
struct Both {
    std::vector<real> chost;
    GpuSpline gpu;
    NonPeriodicXTricubicBSplineView hview, dview;
};

void build_both(const CudaContext& ctx, const Box& b, const std::vector<real>& f, Both& s) {
    s.chost = host_coefficients(b.grid(), f);
    s.hview = make_nonperiodic_x_view(b.grid(), s.chost.data());
    gpu_build(ctx, b.grid(), f, s.gpu);
    s.dview = s.gpu.ws.view();
}

// ---------------------------------------------------------------------------
// Group: memory report formula
// ---------------------------------------------------------------------------

void group_memory(const CudaContext& ctx, bool) {
    std::printf("# group memory\n");
    const Box b{32, 16, 16};
    const auto f = sample_field(b, ladder_field(0.05));
    GpuSpline s;
    gpu_build(ctx, b.grid(), f, s);
    const NonPeriodicXTricubicBSplineReport& r = s.report;
    const size_t nxe = b.nx + 4;
    const size_t coef = 8 * nxe * b.ny * b.nz, spec = 16 * nxe * (b.ny / 2 + 1) * b.nz,
                 tab = 8 * (b.nx - 4);
    const bool ok = r.coefficient_bytes == coef && r.spectrum_bytes == spec && r.line_table_bytes == tab &&
                    r.total_device_bytes == coef + spec + tab + r.cufft_work_area_bytes;
    std::printf("A_MEMORY box=%s coefficient_bytes=%zu spectrum_bytes=%zu line_table_bytes=%zu "
                "cufft_work_area_bytes=%zu total_device_bytes=%zu\n",
                b.str().c_str(), r.coefficient_bytes, r.spectrum_bytes, r.line_table_bytes,
                r.cufft_work_area_bytes, r.total_device_bytes);
    emit("memory_report_formula_mismatch", b.str(), "device", ok ? 0.0 : 1.0, 0.0, Kind::LE);
}

// ---------------------------------------------------------------------------
// Group: node interpolation (every cell centre), random samples
// ---------------------------------------------------------------------------

void group_node(const CudaContext& ctx, bool) {
    std::printf("# group node\n");
    for (const Box& b : {Box{32, 16, 16}, Box{64, 16, 16}}) {
        std::mt19937_64 gen(4101);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        std::vector<real> f(b.cells());
        for (auto& v : f)
            v = u(gen);
        Pts p;
        const double h = b.h();
        for (int k = 0; k < b.nz; ++k)
            for (int j = 0; j < b.ny; ++j)
                for (int i = 0; i < b.nx; ++i)
                    p.push((i + 0.5) * h, (j + 0.5) * h, (k + 0.5) * h);
        double fmax = 0;
        for (double v : f)
            fmax = std::max(fmax, std::fabs(v));
        Both s;
        build_both(ctx, b, f, s);
        const Ev eh = eval_host(s.hview, p, false), ed = eval_device(ctx, s.dview, p, false);
        double mh = 0, md = 0;
        for (size_t i = 0; i < f.size(); ++i) {
            mh = std::max(mh, std::fabs(eh.v[i] - f[i]));
            md = std::max(md, std::fabs(ed.v[i] - f[i]));
        }
        emit("node_interp_rel", b.str(), "host", mh / fmax, 1e-12, Kind::LE);
        emit("node_interp_rel", b.str(), "device", md / fmax, 1e-12, Kind::LE);
    }
}

// ---------------------------------------------------------------------------
// Group: partition of unity (constant 3.0)
// ---------------------------------------------------------------------------

void group_pou(const CudaContext& ctx, bool full) {
    std::printf("# group pou\n");
    std::vector<Box> boxes = {{32, 16, 16}};
    if (full)
        boxes = {{32, 16, 16}, {64, 32, 32}, {128, 64, 64}};
    for (const Box& b : boxes) {
        const auto f = sample_field(b, const_field(3.0));
        const Pts ps = make_ps(b.L1()), mg = make_margin(b);
        Both s;
        build_both(ctx, b, f, s);
        for (int side = 0; side < 2; ++side) {
            const char* nm = side == 0 ? "host" : "device";
            auto ev = [&](const Pts& p) {
                return side == 0 ? eval_host(s.hview, p, false) : eval_device(ctx, s.dview, p, false);
            };
            const Ev e = ev(ps), m = ev(mg);
            double dv = 0, dg = 0, mv = 0, mgr = 0;
            for (size_t i = 0; i < e.n(); ++i) {
                dv = std::max(dv, std::fabs(e.v[i] - 3.0));
                dg = std::max(dg, gnorm(e, i));
            }
            for (size_t i = 0; i < m.n(); ++i) {
                mv = std::max(mv, std::fabs(m.v[i] - 3.0));
                mgr = std::max(mgr, gnorm(m, i));
            }
            emit("pou_value_abs", b.str(), nm, dv, 1e-14, Kind::LE);
            emit("pou_gradient_abs_ps", b.str(), nm, dg, kNaN, Kind::REC);
            emit("pou_value_abs_margin", b.str(), nm, mv, kNaN, Kind::REC);
            emit("pou_gradient_abs_margin", b.str(), nm, mgr, kNaN, Kind::REC);
        }
    }
}

// ---------------------------------------------------------------------------
// Group: exact reproduction of cubic polynomials in x1
// ---------------------------------------------------------------------------

void group_repro(const CudaContext& ctx, bool) {
    std::printf("# group repro\n");
    for (const Box& b : {Box{64, 16, 16}, Box{128, 16, 16}}) {
        Pts p = make_ps(b.L1());
        p.append(make_margin(b));
        Both s;
        for (int cubic = 1; cubic >= 0; --cubic) {
            const Field fld = poly_field(cubic, b.L1());
            const auto f = sample_field(b, fld);
            build_both(ctx, b, f, s);
            const Ev ref = exact_ev(p, fld);
            const std::string tag = cubic ? "repro_cubic" : "repro_quad";
            for (int side = 0; side < 2; ++side) {
                const Ev e = side == 0 ? eval_host(s.hview, p, true) : eval_device(ctx, s.dview, p, true);
                const Rel r = rel_errors(e, ref);
                const char* nm = side == 0 ? "host" : "device";
                emit(tag + "_value_rel", b.str(), nm, r.v, 1e-12, Kind::LE);
                emit(tag + "_gradient_rel", b.str(), nm, r.g, 1e-12, Kind::LE);
                emit(tag + "_hessian_rel", b.str(), nm, r.H, kNaN, Kind::REC);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Group: order ladder (3-D) and x-only sub-ladder
// ---------------------------------------------------------------------------

// Emulated (pre-activation design emulation a_ladder.txt, numpy) errors at
// h = 1/16, 1/32, 1/64 for the field of threshold A (different point draw:
// recorded only).
const double kEmul[5][3] = {{2.17569e-03, 1.45501e-04, 9.36022e-06},
                            {2.95180e-04, 1.43162e-05, 6.47728e-07},
                            {1.20161e-01, 1.60518e-02, 2.06018e-03},
                            {2.04793e-02, 1.95918e-03, 1.79773e-04},
                            {4.02263e+00, 1.06638e+00, 2.73016e-01}};

const char* kQuantity[5] = {"val_max", "val_rms", "grad_max", "grad_rms", "hess_max"};
const double kOrderGate[5] = {3.8, 3.8, 2.8, 2.8, 1.8};

double qget(const AbsErr& e, int q) {
    switch (q) {
    case 0: return e.val_max;
    case 1: return e.val_rms;
    case 2: return e.grad_max;
    case 3: return e.grad_rms;
    default: return e.hess_max;
    }
}

void print_ladder(const char* kind, const char* side, const std::vector<int>& cells,
                  const std::vector<AbsErr>& errs) {
    for (size_t l = 0; l < errs.size(); ++l)
        std::printf("A_LADDER_ERR kind=%s side=%s cells_per_unit=%d val_max=%.6e val_rms=%.6e "
                    "grad_max=%.6e grad_rms=%.6e hess_max=%.6e\n",
                    kind, side, cells[l], errs[l].val_max, errs[l].val_rms, errs[l].grad_max,
                    errs[l].grad_rms, errs[l].hess_max);
    for (size_t l = 0; l + 1 < errs.size(); ++l) {
        std::printf("A_LADDER_ORDER kind=%s side=%s pair=%d->%d", kind, side, cells[l], cells[l + 1]);
        for (int q = 0; q < 5; ++q)
            std::printf(" %s=%.7g", kQuantity[q], std::log2(qget(errs[l], q) / qget(errs[l + 1], q)));
        std::printf("\n");
    }
}

// min over the first two pairs (16->32, 32->64) of the order of quantity q.
double min_order(const std::vector<AbsErr>& errs, int q) {
    double m = 1e300;
    for (size_t l = 0; l < 2 && l + 1 < errs.size(); ++l)
        m = std::min(m, std::log2(qget(errs[l], q) / qget(errs[l + 1], q)));
    return m;
}

void group_ladder(const CudaContext& ctx, bool full) {
    std::printf("# group ladder\n");
    const Field fld = ladder_field(0.05);
    // --- 3-D ladder ---
    {
        std::vector<int> cells = full ? std::vector<int>{16, 32, 64} : std::vector<int>{8, 16};
        const Pts ps = make_ps(2.0); // fixed physical point set (L1 = 2)
        const Ev ref = exact_ev(ps, fld);
        std::vector<AbsErr> eh, ed;
        std::string boxes;
        for (int c : cells) {
            const Box b{2 * c, c, c};
            boxes += (boxes.empty() ? "" : "..") + b.str();
            const auto f = sample_field(b, fld);
            Both s;
            build_both(ctx, b, f, s);
            eh.push_back(abs_errors(eval_host(s.hview, ps, true), ref));
            ed.push_back(abs_errors(eval_device(ctx, s.dview, ps, true), ref));
        }
        print_ladder("3d", "host", cells, eh);
        print_ladder("3d", "device", cells, ed);
        for (int side = 0; side < 2; ++side) {
            const char* nm = side == 0 ? "host" : "device";
            const auto& errs = side == 0 ? eh : ed;
            if (full) {
                for (int q = 0; q < 5; ++q) {
                    emit(std::string("ladder_min_order_ge_") + kQuantity[q], boxes, nm, min_order(errs, q),
                         kOrderGate[q], Kind::GE);
                    for (size_t l = 0; l < 3; ++l)
                        std::printf("A_EMUL side=%s level=1/%d quantity=%s measured=%.6e emulated=%.6e "
                                    "measured_over_emulated=%.6f\n",
                                    nm, cells[l], kQuantity[q], qget(errs[l], q), kEmul[q][l],
                                    qget(errs[l], q) / kEmul[q][l]);
                }
            } else {
                for (int q = 0; q < 5; ++q)
                    emit(std::string("ladder_reduced_order_") + kQuantity[q], boxes, nm,
                         std::log2(qget(errs[0], q) / qget(errs[1], q)), kNaN, Kind::REC);
            }
        }
    }
    // --- x-only sub-ladder: g(x1) = exp(x1) sin(3 x1), boxes nx x 4 x 4 ---
    {
        const Field gx = xonly_field();
        std::vector<int> cells = {16, 32, 64, 128};
        std::mt19937_64 gen(20261025);
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        Pts p;
        for (int side = 0; side < 2; ++side)
            for (int i = 0; i < 2000; ++i) {
                std::uniform_real_distribution<double> ux(side == 0 ? 0.0 : 2.0 - 1.0 / 16.0,
                                                          side == 0 ? 1.0 / 16.0 : 2.0);
                const double x = ux(gen), y = u01(gen), z = u01(gen);
                p.push(x, y, z);
            }
        const Ev ref = exact_ev(p, gx);
        std::vector<AbsErr> eh, ed;
        for (int c : cells) {
            const Box b{2 * c, 4, 4};
            Grid3D g(b.nx, 4, 4, 1.0 / c, 1.0 / c, 1.0 / c);
            // (hx = hy = hz = 1/c; the periodic y/z extent 4/c is irrelevant for a field constant there)
            std::vector<real> f(b.cells());
            for (int k = 0; k < 4; ++k)
                for (int j = 0; j < 4; ++j)
                    for (int i = 0; i < b.nx; ++i) {
                        double v, gg[3], H[6];
                        gx((i + 0.5) / c, 0, 0, v, gg, H);
                        f[i + static_cast<size_t>(b.nx) * (j + 4 * k)] = v;
                    }
            const auto ch = host_coefficients(g, f);
            const auto hv = make_nonperiodic_x_view(g, ch.data());
            GpuSpline s;
            gpu_build(ctx, g, f, s);
            eh.push_back(abs_errors(eval_host(hv, p, true), ref));
            ed.push_back(abs_errors(eval_device(ctx, s.ws.view(), p, true), ref));
        }
        print_ladder("xonly", "host", cells, eh);
        print_ladder("xonly", "device", cells, ed);
        for (int side = 0; side < 2; ++side) {
            const char* nm = side == 0 ? "host" : "device";
            const auto& errs = side == 0 ? eh : ed;
            for (int q = 0; q < 5; ++q)
                emit(std::string("xonly_min_order_ge_") + kQuantity[q], "32x4x4..256x4x4", nm,
                     min_order(errs, q), kOrderGate[q], full ? Kind::REC : Kind::GE);
        }
    }
}

// ---------------------------------------------------------------------------
// Group: CPU vs GPU
// ---------------------------------------------------------------------------

void group_cpugpu(const CudaContext& ctx, bool full) {
    std::printf("# group cpugpu\n");
    std::vector<Box> boxes = full ? std::vector<Box>{{32, 16, 16}, {64, 32, 32}}
                                  : std::vector<Box>{{32, 16, 16}, {64, 16, 16}};
    const Field fld = ladder_field(0.05);
    for (const Box& b : boxes) {
        const auto f = sample_field(b, fld);
        Pts p = make_ps(b.L1());
        const size_t nps = p.n();
        p.append(make_margin(b));
        Both s;
        build_both(ctx, b, f, s);
        // full pipeline: GPU prefilter + device evaluation vs CPU prefilter + host evaluation
        const Ev cpu = eval_host(s.hview, p, true);
        const Ev gpu = eval_device(ctx, s.dview, p, true);
        const Rel full_r = rel_errors(gpu, cpu);
        emit("cpugpu_full_value_rel", b.str(), "both", full_r.v, 1e-13, Kind::LE);
        // evaluation only: GPU coefficients, host vs device evaluation
        const std::vector<real> cg = gpu_coefficients(ctx, s.gpu);
        const auto hv_g = make_nonperiodic_x_view(b.grid(), cg.data());
        const Ev ref_e = eval_host(hv_g, p, true);
        const Rel ev_r = rel_errors(gpu, ref_e);
        emit("cpugpu_evalonly_value_rel", b.str(), "both", ev_r.v, 1e-13, Kind::LE);
        emit("cpugpu_evalonly_gradient_rel", b.str(), "both", ev_r.g, 1e-13, Kind::LE);
        emit("cpugpu_evalonly_hessian_rel", b.str(), "both", ev_r.H, 1e-13, Kind::LE);
        // recorded
        double dc = 0, cm = 0;
        for (size_t i = 0; i < cg.size(); ++i) {
            dc = std::max(dc, std::fabs(cg[i] - s.chost[i]));
            cm = std::max(cm, std::fabs(s.chost[i]));
        }
        emit("cpugpu_coeff_diff_rel", b.str(), "both", dc / cm, kNaN, Kind::REC);
        std::vector<char> m_in(p.n()), m_nf(p.n()), m_mg(p.n());
        const double L1 = b.L1();
        for (size_t i = 0; i < p.n(); ++i) {
            const double x = p.x[i];
            m_mg[i] = (x < 0 || x > L1);
            m_nf[i] = !m_mg[i] && (x <= 1.0 / 16.0 || x >= L1 - 1.0 / 16.0);
            m_in[i] = (x >= 0.5 && x <= L1 - 0.5);
        }
        (void)nps;
        const struct {
            const char* n;
            const std::vector<char>* m;
        } regions[3] = {{"interior", &m_in}, {"nearface", &m_nf}, {"margin", &m_mg}};
        for (const auto& r : regions) {
            const Rel rr = rel_errors(gpu, cpu, *r.m);
            emit(std::string("cpugpu_full_gradient_rel_") + r.n, b.str(), "both", rr.g, kNaN, Kind::REC);
            emit(std::string("cpugpu_full_hessian_rel_") + r.n, b.str(), "both", rr.H, kNaN, Kind::REC);
        }
    }
}

// ---------------------------------------------------------------------------
// Group: Hessian path bitwise equal to the value path
// ---------------------------------------------------------------------------

void group_bitwise(const CudaContext& ctx, bool full) {
    std::printf("# group bitwise\n");
    const Box b = full ? Box{64, 32, 32} : Box{32, 16, 16};
    const auto f = sample_field(b, ladder_field(0.05));
    Pts p = make_ps(b.L1());
    p.append(make_margin(b));
    Both s;
    build_both(ctx, b, f, s);
    const Ev hv = eval_host(s.hview, p, false), hh = eval_host(s.hview, p, true);
    const Ev dv = eval_device(ctx, s.dview, p, false), dh = eval_device(ctx, s.dview, p, true);
    emit("hessian_path_bitwise_mismatches", b.str(), "host", static_cast<double>(memcmp_mismatches(hv, hh)),
         0.0, Kind::LE);
    emit("hessian_path_bitwise_mismatches", b.str(), "device", static_cast<double>(memcmp_mismatches(dv, dh)),
         0.0, Kind::LE);
}

// ---------------------------------------------------------------------------
// Group: no device allocation during evaluation (and no workspace regrowth)
// ---------------------------------------------------------------------------

void group_noalloc(const CudaContext& ctx, bool full) {
    std::printf("# group noalloc\n");
    const Box b = full ? Box{64, 32, 32} : Box{32, 16, 16};
    const auto f = sample_field(b, ladder_field(0.05));
    GpuSpline s;
    gpu_build(ctx, b.grid(), f, s);
    const NonPeriodicXTricubicBSplineView v = s.ws.view();

    std::mt19937_64 gen(20261025);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    std::vector<real> hx(10000), hy(10000), hz(10000);
    for (int i = 0; i < 10000; ++i) {
        hx[i] = u01(gen) * b.L1();
        hy[i] = u01(gen);
        hz[i] = u01(gen);
    }
    DeviceBuffer<real> px, py, pz, ov, ogx, ogy, ogz, oh;
    DeviceBuffer<int> in;
    upload(px, hx);
    upload(py, hy);
    upload(pz, hz);
    ov.resize(10000);
    ogx.resize(10000);
    ogy.resize(10000);
    ogz.resize(10000);
    oh.resize(60000);
    in.resize(10000);
    auto run = [&](bool hess) {
        if (hess)
            evaluate_nonperiodic_x_tricubic_bspline_hessian(ctx, v, px.span(), py.span(), pz.span(),
                                                            ov.span(), ogx.span(), ogy.span(),
                                                            ogz.span(), oh.span(), in.span());
        else
            evaluate_nonperiodic_x_tricubic_bspline(ctx, v, px.span(), py.span(), pz.span(), ov.span(),
                                                    ogx.span(), ogy.span(), ogz.span(), in.span());
    };
    run(false); // warm-up: lazy module loading is not an allocation of the evaluation
    run(true);
    ctx.synchronize();
    size_t free_before = 0, total = 0, free_after = 0;
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
    run(false);
    run(false);
    run(true);
    run(true);
    ctx.synchronize();
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free_after, &total));
    std::printf("A_NOALLOC free_before=%zu free_after=%zu\n", free_before, free_after);
    emit("noalloc_eval_free_bytes_delta", b.str(), "device",
         std::fabs(static_cast<double>(free_after) - static_cast<double>(free_before)), 0.0, Kind::LE);

    // A second prefilter of the same size must not regrow the workspace.
    const real* c0 = s.ws.coefficients.data();
    const cufftDoubleComplex* s0 = s.ws.spectrum.data();
    const real* t0 = s.ws.line_table.data();
    gpu_build(ctx, b.grid(), f, s);
    const bool same = c0 == s.ws.coefficients.data() && s0 == s.ws.spectrum.data() &&
                      t0 == s.ws.line_table.data();
    emit("prefilter_repeat_workspace_reallocations", b.str(), "device", same ? 0.0 : 1.0, 0.0, Kind::LE);
}

// ---------------------------------------------------------------------------
// Group: shift by whole periods in y and z (dyadic coordinates)
// ---------------------------------------------------------------------------

void group_shift(const CudaContext& ctx, bool full) {
    std::printf("# group shift\n");
    std::vector<Box> boxes = full ? std::vector<Box>{{32, 16, 16}, {64, 32, 32}}
                                  : std::vector<Box>{{32, 16, 16}};
    const int ks[] = {0,    1,     37,    511,   512,   1023,  1024,  1365, 2047,
                      2048, 32768, 63488, 64171, 64512, 65024, 65499, 65535};
    const int qs[] = {-2, -1, 1, 2, 3};
    for (const Box& b : boxes) {
        const auto f = sample_field(b, ladder_field(0.05));
        Both s;
        build_both(ctx, b, f, s);
        // Two fixed settings of the two other coordinates (dyadic, exact; the x1
        // value of setting 2 is the last dyadic point below L1).
        const double L1 = b.L1();
        const double x1s[2] = {0.3125, L1 - 1.0 / 65536.0};
        const double ots[2] = {0.71875, 65535.0 / 65536.0};
        Pts pairs; // entries 2n (base) and 2n+1 (shifted)
        for (int axis = 1; axis <= 2; ++axis)
            for (int k : ks)
                for (int setting = 0; setting < 2; ++setting)
                    for (int q : qs) {
                        const double a = static_cast<double>(k) / 65536.0;
                        double base[3] = {x1s[setting], 0, 0}, sh[3] = {x1s[setting], 0, 0};
                        const int other = axis == 1 ? 2 : 1;
                        base[axis] = a;
                        sh[axis] = a + static_cast<double>(q); // exact: Ly = Lz = 1
                        base[other] = sh[other] = ots[setting];
                        pairs.push(base[0], base[1], base[2]);
                        pairs.push(sh[0], sh[1], sh[2]);
                    }
        const size_t np = pairs.n() / 2;
        for (int side = 0; side < 2; ++side) {
            const Ev e = side == 0 ? eval_host(s.hview, pairs, true) : eval_device(ctx, s.dview, pairs, true);
            double dv = 0, dg = 0, dh = 0;
            size_t bad = 0;
            for (size_t n = 0; n < np; ++n) {
                const size_t i = 2 * n, j = i + 1;
                dv = std::max(dv, std::fabs(e.v[i] - e.v[j]));
                dg = std::max({dg, std::fabs(e.gx[i] - e.gx[j]), std::fabs(e.gy[i] - e.gy[j]),
                               std::fabs(e.gz[i] - e.gz[j])});
                for (int c = 0; c < 6; ++c)
                    dh = std::max(dh, std::fabs(e.H[6 * i + c] - e.H[6 * j + c]));
                bool eq = !std::memcmp(&e.v[i], &e.v[j], 8) && !std::memcmp(&e.gx[i], &e.gx[j], 8) &&
                          !std::memcmp(&e.gy[i], &e.gy[j], 8) && !std::memcmp(&e.gz[i], &e.gz[j], 8) &&
                          !std::memcmp(&e.H[6 * i], &e.H[6 * j], 6 * 8);
                if (!eq)
                    ++bad;
            }
            const char* nm = side == 0 ? "host" : "device";
            emit("shift_value_maxabs", b.str(), nm, dv, 1e-14, Kind::LE);
            emit("shift_gradient_maxabs", b.str(), nm, dg, 1e-14, Kind::LE);
            emit("shift_hessian_maxabs", b.str(), nm, dh, 1e-14, Kind::LE);
            emit("shift_memcmp_mismatched_pairs", b.str(), nm, static_cast<double>(bad), kNaN, Kind::REC);
        }
    }
}

// ---------------------------------------------------------------------------
// Group: evaluation domain
// ---------------------------------------------------------------------------

double beta3(double x) {
    x = std::fabs(x);
    if (x >= 2)
        return 0;
    if (x >= 1)
        return (2 - x) * (2 - x) * (2 - x) / 6.0;
    return (4 - 6 * x * x + 3 * x * x * x) / 6.0;
}
double dbeta3(double x) {
    const double s = x < 0 ? -1.0 : 1.0, a = std::fabs(x);
    if (a >= 2)
        return 0;
    if (a >= 1)
        return -s * (2 - a) * (2 - a) / 2.0;
    return s * (-12 * a + 9 * a * a) / 6.0;
}

// Independent evaluation: direct beta3 sums (no weight polynomials, no stencil
// code); x over the whole extended range a = -2..nx+1, y/z over the 4 wrapped
// nearest coefficients.
void independent_eval(const std::vector<real>& c, const Box& b, double x, double y, double z, double& v,
                      double g[3]) {
    const double h = b.h();
    const int nxe = b.nx + 4;
    auto axis = [&](double xx, double L, int n, int idx[4], double w[4], double d[4]) {
        const double xr = xx - L * std::floor(xx / L);
        const double t = xr / h - 0.5;
        const int i0 = static_cast<int>(std::floor(t));
        for (int k = 0; k < 4; ++k) {
            const int a = i0 - 1 + k;
            idx[k] = ((a % n) + n) % n;
            w[k] = beta3(t - a);
            d[k] = dbeta3(t - a) / h;
        }
    };
    int iy[4], iz[4];
    double wy[4], dy[4], wz[4], dz[4];
    axis(y, b.ny * h, b.ny, iy, wy, dy);
    axis(z, b.nz * h, b.nz, iz, wz, dz);
    const double tx = x / h - 0.5;
    v = 0;
    g[0] = g[1] = g[2] = 0;
    for (int a = -2; a <= b.nx + 1; ++a) {
        const double wx = beta3(tx - a), dx = dbeta3(tx - a) / h;
        if (wx == 0 && dx == 0)
            continue;
        for (int q = 0; q < 4; ++q)
            for (int r = 0; r < 4; ++r) {
                const double cv = c[(a + 2) + static_cast<size_t>(nxe) * (iy[q] + static_cast<size_t>(b.ny) * iz[r])];
                v += cv * wx * wy[q] * wz[r];
                g[0] += cv * dx * wy[q] * wz[r];
                g[1] += cv * wx * dy[q] * wz[r];
                g[2] += cv * wx * wy[q] * dz[r];
            }
    }
}

__global__ void k_null_view_eval(NonPeriodicXTricubicBSplineView v, const real* px, const real* py,
                                 const real* pz, size_t n, int* flags) {
    const size_t p = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
    if (p >= n)
        return;
    real a, b, c, d, H[6];
    const bool i1 = evaluate_point(v, px[p], py[p], pz[p], a, b, c, d);
    const bool i2 = evaluate_point_hessian(v, px[p], py[p], pz[p], a, b, c, d, H);
    // NaN outputs expected on false
    flags[p] = (i1 ? 1 : 0) + (i2 ? 2 : 0) + ((std::isnan(a) && std::isnan(H[5])) ? 4 : 0);
}

void group_domain(const CudaContext& ctx, bool full) {
    std::printf("# group domain\n");
    std::vector<Box> boxes = full ? std::vector<Box>{{32, 16, 16}, {64, 32, 32}}
                                  : std::vector<Box>{{32, 16, 16}};
    for (const Box& b : boxes) {
        const auto f = sample_field(b, ladder_field(0.05));
        Both s;
        build_both(ctx, b, f, s);
        const Pts mg = make_margin(b);
        const std::vector<real> cg = gpu_coefficients(ctx, s.gpu);
        const double h = b.h(), L1 = b.L1();
        // margin points: finite, inside, equal to the independently evaluated end cubic pieces
        for (int side = 0; side < 2; ++side) {
            const char* nm = side == 0 ? "host" : "device";
            const Ev e = side == 0 ? eval_host(s.hview, mg, true) : eval_device(ctx, s.dview, mg, true);
            const std::vector<real>& c = side == 0 ? s.chost : cg;
            size_t notfinite = 0;
            double dv = 0, dg = 0, sv = 0, sg = 0;
            for (size_t i = 0; i < mg.n(); ++i) {
                bool fin = e.inside[i] == 1 && std::isfinite(e.v[i]) && std::isfinite(e.gx[i]) &&
                           std::isfinite(e.gy[i]) && std::isfinite(e.gz[i]);
                for (int k = 0; k < 6; ++k)
                    fin = fin && std::isfinite(e.H[6 * i + k]);
                if (!fin)
                    ++notfinite;
                double v, g[3];
                independent_eval(c, b, mg.x[i], mg.y[i], mg.z[i], v, g);
                dv = std::max(dv, std::fabs(e.v[i] - v));
                sv = std::max(sv, std::fabs(v));
                const double dd = std::sqrt((e.gx[i] - g[0]) * (e.gx[i] - g[0]) + (e.gy[i] - g[1]) * (e.gy[i] - g[1]) +
                                            (e.gz[i] - g[2]) * (e.gz[i] - g[2]));
                dg = std::max(dg, dd);
                sg = std::max(sg, std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]));
            }
            emit("domain_margin_nonfinite_or_outside", b.str(), nm, static_cast<double>(notfinite), 0.0, Kind::LE);
            emit("domain_margin_vs_independent_value_rel", b.str(), nm, dv / sv, 1e-12, Kind::LE);
            emit("domain_margin_vs_independent_gradient_rel", b.str(), nm, dg / sg, 1e-12, Kind::LE);
        }
        // Outside points: reported outside, no coefficient read (null coefficient pointer).
        Pts out;
        const double inf = std::numeric_limits<double>::infinity();
        out.push(-h / 2 - h / 8, 0.3, 0.4);
        out.push(L1 + h / 2 + h / 8, 0.3, 0.4);
        out.push(L1 + h / 2, 0.3, 0.4); // half-open: the upper end is outside
        out.push(-1e30, 0.3, 0.4);
        out.push(1e30, 0.3, 0.4);
        out.push(-inf, 0.3, 0.4);
        out.push(inf, 0.3, 0.4);
        out.push(kNaN, 0.3, 0.4);
        out.push(0.5, kNaN, 0.4);
        out.push(0.5, 0.3, inf);
        NonPeriodicXTricubicBSplineView nv = s.hview;
        nv.coeff = nullptr;
        size_t wrong = 0;
        for (size_t i = 0; i < out.n(); ++i) {
            real a, b2, c, d, H[6];
            const bool i1 = evaluate_point(nv, out.x[i], out.y[i], out.z[i], a, b2, c, d);
            const bool i2 = evaluate_point_hessian(nv, out.x[i], out.y[i], out.z[i], a, b2, c, d, H);
            if (i1 || i2 || !std::isnan(a) || !std::isnan(H[5]))
                ++wrong;
        }
        emit("domain_outside_reported_without_read", b.str(), "host", static_cast<double>(wrong), 0.0, Kind::LE);
        {
            DeviceBuffer<real> px, py, pz;
            DeviceBuffer<int> fl;
            upload(px, out.x);
            upload(py, out.y);
            upload(pz, out.z);
            fl.resize(out.n());
            k_null_view_eval<<<1, 64, 0, ctx.cuda_stream()>>>(nv, px.data(), py.data(), pz.data(), out.n(),
                                                               fl.data());
            MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
            const auto flags = download(ctx, fl, out.n());
            size_t bad = 0;
            for (int v : flags)
                if (v != 4)
                    ++bad;
            emit("domain_outside_reported_without_read", b.str(), "device", static_cast<double>(bad), 0.0, Kind::LE);
        }
        // Inclusive lower end, exclusive upper end of the half-open domain.
        Pts edge;
        edge.push(-h / 2, 0.3, 0.4);
        edge.push(L1 + h / 2, 0.3, 0.4);
        const Ev ee = eval_host(s.hview, edge, false);
        emit("domain_half_open_violations", b.str(), "host",
             static_cast<double>((ee.inside[0] == 1 ? 0 : 1) + (ee.inside[1] == 0 ? 0 : 1)), 0.0, Kind::LE);
    }
}

} // namespace

int main(int argc, char** argv) {
    bool full = false;
    std::string only;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--full")
            full = true;
        else if (a == "--only" && i + 1 < argc)
            only = argv[++i];
        else {
            std::fprintf(stderr, "usage: %s [--full] [--only node|pou|repro|ladder|cpugpu|bitwise|noalloc|shift|domain|memory]\n",
                         argv[0]);
            return 2;
        }
    }
    std::printf("A_TIER tier=%s only=%s\n", full ? "full" : "fast", only.empty() ? "all" : only.c_str());
    if (!full)
        std::printf("A_TIER reduced: pou=32x16x16; cpugpu=32x16x16,64x16x16; bitwise/noalloc/shift/domain=32x16x16; "
                    "ladder3d=16x8x8,32x16x16 (recorded only); xonly ladder gated (A gates); every grid <= 32768 cells\n");
    try {
        CudaContext ctx(0);
        struct Group {
            const char* name;
            void (*fn)(const CudaContext&, bool);
        };
        const Group groups[] = {{"memory", group_memory},   {"node", group_node},     {"pou", group_pou},
                                {"repro", group_repro},     {"ladder", group_ladder}, {"cpugpu", group_cpugpu},
                                {"bitwise", group_bitwise}, {"noalloc", group_noalloc}, {"shift", group_shift},
                                {"domain", group_domain}};
        bool any = false;
        for (const Group& g : groups)
            if (only.empty() || only == g.name) {
                g.fn(ctx, full);
                any = true;
            }
        if (!any) {
            std::fprintf(stderr, "unknown group '%s'\n", only.c_str());
            return 2;
        }
    } catch (const std::exception& e) {
        std::printf("A_ERROR %s\n", e.what());
        std::printf("A_VERDICT all_pass=0 checks_pass=%d checks_fail=%d\n", g_pass, g_fail + 1);
        return 1;
    }
    const bool ok = g_fail == 0;
    std::printf("A_VERDICT all_pass=%d checks_pass=%d checks_fail=%d\n", ok ? 1 : 0, g_pass, g_fail);
    return ok ? 0 : 1;
}
