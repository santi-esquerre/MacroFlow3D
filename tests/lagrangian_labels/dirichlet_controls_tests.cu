/**
 * @file dirichlet_controls_tests.cu
 * @brief SF-41 N3b: exact controls C1-C4 of the Lagrangian consumer on medium (b) (non-periodic-in-x1 spline,
 *        x-exit handling of the tracker and of the Pollock comparator), pre-registration RR5; ctest
 *        `lagrangian_dirichlet_controls` (FAST tier: grids <= 32^3, well under 60 s).
 *
 * The splines are built from heads WRITTEN into the sample array (no solver; f = -H - G1 x1 + G1 L1 at the
 * cell centres, ln k = eps Y samples) with the N1 prefilter (GPU), so the controls test the spline, the
 * field functors and the tracker free of the solver's iteration error.
 *
 *   C1  k = 1, H = G1 (L1 - x1): f = 0 exactly; straight line, tau, I, covectors.
 *   C2  layered ln k = 0.5 cos(2 pi 0.6 x1 + 0.3), exact discrete 1-D head (harmonic face K, half-cell Dirichlet
 *       faces, constant face flux): transverse positions, covectors, tau vs the G7K15 quadrature of 1/v1 on the
 *       host functor of the same spline; I recorded.
 *   C3  written transverse field: v_s of the host and device functors (unfused and fused) at p, p + e2, p + e3;
 *       plus the domain report of the functors (outside the domain: false, NaN, no coefficient read).
 *   C4  exits: C4a (k = 1, every particle leaves through the outlet; tracker fused/unfused, Pollock
 *       non-periodic), C4b (harmonic lobe field: inlet exits, reference classification of lobe_field_reference.hpp),
 *       plus the straddle and stage-outside-the-domain (rules E1 / E2) cases and the recorded reversed head.
 *
 * Modes: default = fast tier; `--full` = the RR5 sizes (V100); `--only C1|C2|C3|C4`.
 * Machine-readable lines: `C_CHECK name=... grid=... path=<fused|unfused|pollock|host|device> value=... gate=...
 * verdict=PASS|FAIL`, `C_RECORD ...` (no gate), `C_VERDICT all_pass=<0|1>`; exit 0 iff all gated checks pass.
 * Instantiates the device templates: compiled with --fmad=false (CMakeLists.txt).
 */

#include "lobe_field_reference.hpp"

#include "src/core/DeviceBuffer.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/external/nlohmann/json.hpp"
#include "src/numerics/interpolation/NonPeriodicXTricubicBSpline.cuh"
#include "src/physics/particles/lagrangian_labels/FaceFluxRepack.cuh"
#include "src/physics/particles/lagrangian_labels/FusedSplineLagrangianField.cuh"
#include "src/physics/particles/lagrangian_labels/LagrangianTracker.cuh"
#include "src/physics/particles/lagrangian_labels/NonPeriodicSplineLagrangianField.cuh"
#include "src/physics/particles/lagrangian_labels/PollockComparator.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using namespace macroflow3d;
namespace ll = macroflow3d::lagrangian_labels;
namespace ip = macroflow3d::interpolation;
namespace stt = macroflow3d::physics::particles::streamline_tracker;
namespace lr = macroflow3d::lobe_ref;

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

// ------------------------------------------------------------------------------------------------
// reporting
// ------------------------------------------------------------------------------------------------
bool g_all_pass = true;
bool g_full = false;

std::string strf(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
std::string strf(const char* fmt, ...) {
    char buf[512];
    va_list ap;
    va_start(ap, fmt);
    std::vsnprintf(buf, sizeof(buf), fmt, ap);
    va_end(ap);
    return std::string(buf);
}

void emit(const std::string& name, const std::string& grid, const char* path, double value,
          const std::string& gate, bool pass) {
    std::printf("C_CHECK name=%s grid=%s path=%s value=%.6e gate=%s verdict=%s\n", name.c_str(), grid.c_str(),
                path, value, gate.c_str(), pass ? "PASS" : "FAIL");
    std::fflush(stdout);
    g_all_pass = g_all_pass && pass;
}
/// value <= bound (NaN fails).
void check_le(const std::string& name, const std::string& grid, const char* path, double value, double bound) {
    emit(name, grid, path, value, strf("<=%.0e", bound), std::isfinite(value) && value <= bound);
}
/// integer count that must be 0.
void check_zero(const std::string& name, const std::string& grid, const char* path, long long count) {
    emit(name, grid, path, static_cast<double>(count), "==0", count == 0);
}
void record(const std::string& name, const std::string& grid, const char* path, const std::string& text) {
    std::printf("C_RECORD name=%s grid=%s path=%s %s\n", name.c_str(), grid.c_str(), path, text.c_str());
    std::fflush(stdout);
}

std::string grid_str(int nx, int ny, int nz) { return strf("%dx%dx%d", nx, ny, nz); }

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }
template <class T> bool same_vec(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

// ------------------------------------------------------------------------------------------------
// spline build from written samples
// ------------------------------------------------------------------------------------------------
struct Spline {
    int nx = 0, ny = 0, nz = 0;
    double h = 0.0, L1 = 0.0;
    Grid3D grid;
    ip::NonPeriodicXTricubicBSplineWorkspace wp, wk;
    std::vector<real> cp, ck; ///< host copies of the GPU coefficients
    DeviceBuffer<double2> fused;
    double G[3] = {1.0, 0.0, 0.0};
    std::vector<real> f, lnk; ///< the written samples
};

using SampleFn = std::function<double(double, double, double)>;

/// Cell-centred sample at (x1, x2, x3) = ((i+1/2) h, (j+1/2) h, (k+1/2) h).
std::vector<real> sample_field(int nx, int ny, int nz, double h, const SampleFn& fn) {
    std::vector<real> s(static_cast<std::size_t>(nx) * ny * nz);
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i)
                s[static_cast<std::size_t>(i) + static_cast<std::size_t>(nx) * (j + static_cast<std::size_t>(ny) * k)] =
                    fn((i + 0.5) * h, (j + 0.5) * h, (k + 0.5) * h);
    return s;
}

std::vector<real> download_real(const CudaContext& ctx, const DeviceBuffer<real>& d, std::size_t n) {
    ctx.synchronize();
    std::vector<real> h(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), d.data(), n * sizeof(real), cudaMemcpyDeviceToHost));
    return h;
}

void prefilter_one(CudaContext& ctx, const Grid3D& grid, const std::vector<real>& samples,
                   ip::NonPeriodicXTricubicBSplineWorkspace& ws, std::vector<real>& host_coeff) {
    DeviceBuffer<real> d(samples.size());
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(d.data(), samples.data(), samples.size() * sizeof(real), cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    ip::prefilter_nonperiodic_x_tricubic_bspline(ctx, grid, DeviceSpan<const real>(d.data(), samples.size()), ws);
    host_coeff = download_real(ctx, ws.coefficients,
                               (static_cast<std::size_t>(grid.nx) + 4) * grid.ny * grid.nz);
}

/// Builds the splines of f and ln k (nz == ny, h = 1/ny) and the interleaved fused array.
void build(CudaContext& ctx, Spline& S, int nx, int ny, int nz, std::vector<real> f, std::vector<real> lnk,
           double G1) {
    S.nx = nx;
    S.ny = ny;
    S.nz = nz;
    S.h = 1.0 / ny;
    S.L1 = nx * S.h;
    S.grid = Grid3D(nx, ny, nz, S.h, S.h, S.h);
    S.G[0] = G1;
    S.G[1] = S.G[2] = 0.0;
    S.f = std::move(f);
    S.lnk = std::move(lnk);
    prefilter_one(ctx, S.grid, S.f, S.wp, S.cp);
    prefilter_one(ctx, S.grid, S.lnk, S.wk, S.ck);
    const std::size_t n = S.cp.size();
    const std::vector<double2> inter = ll::interleave_coefficients(S.cp.data(), S.ck.data(), n);
    S.fused = DeviceBuffer<double2>(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(S.fused.data(), inter.data(), n * sizeof(double2), cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
}

ll::NonPeriodicSplineLagrangianField host_field(const Spline& S) {
    ll::NonPeriodicSplineLagrangianField f;
    f.potential = ip::make_nonperiodic_x_view(S.grid, S.cp.data());
    f.log_conductivity = ip::make_nonperiodic_x_view(S.grid, S.ck.data());
    for (int d = 0; d < 3; ++d)
        f.G[d] = S.G[d];
    return f;
}
ll::NonPeriodicSplineLagrangianField device_field(const Spline& S) {
    ll::NonPeriodicSplineLagrangianField f;
    f.potential = ip::make_nonperiodic_x_view(S.grid, S.wp.coefficients.data());
    f.log_conductivity = ip::make_nonperiodic_x_view(S.grid, S.wk.coefficients.data());
    for (int d = 0; d < 3; ++d)
        f.G[d] = S.G[d];
    return f;
}
ll::FusedNonPeriodicSplineLagrangianField fused_field(const Spline& S) {
    return ll::make_fused_np_field(device_field(S), S.fused.data());
}

ll::LagrangianTrackResult run_tracker(CudaContext& ctx, const Spline& S, bool fused, const ll::InjectionSpec& inj,
                                      ll::TrackOptions to) {
    to.fused_splines = fused;
    ll::DeviceSplineFlowNonPeriodicX flow;
    flow.grid = S.grid;
    flow.coef_pot = S.wp.coefficients.data();
    flow.coef_logk = S.wk.coefficients.data();
    flow.coef_interleaved = S.fused.data();
    for (int d = 0; d < 3; ++d)
        flow.G[d] = S.G[d];
    ll::LagrangianTracker tr;
    tr.prepare(ctx, flow, inj, to);
    ll::LagrangianTrackResult r = tr.track(ctx);
    return r;
}

const char* path_name(bool fused) { return fused ? "fused" : "unfused"; }

/// Bitwise comparison of the per-particle outputs of two tracker results (mismatching arrays).
long long result_mismatches(const ll::LagrangianTrackResult& a, const ll::LagrangianTrackResult& b) {
    long long bad = 0;
    bad += !same_vec(a.planes_x23, b.planes_x23);
    bad += !same_vec(a.planes_tau, b.planes_tau);
    bad += !same_vec(a.planes_I, b.planes_I);
    bad += !same_vec(a.planes_cov, b.planes_cov);
    bad += !same_vec(a.times_x, b.times_x);
    bad += !same_vec(a.times_I, b.times_I);
    bad += !same_vec(a.times_cov, b.times_cov);
    bad += !(a.status_plane == b.status_plane);
    bad += !(a.status_time == b.status_time);
    bad += !same_vec(a.weights, b.weights);
    bad += !(a.exit_inlet_plane == b.exit_inlet_plane && a.exit_outlet_plane == b.exit_outlet_plane &&
             a.exit_inlet_time == b.exit_inlet_time && a.exit_outlet_time == b.exit_outlet_time);
    for (std::size_t p = 0; p < a.stats_plane.size(); ++p)
        bad += a.stats_plane[p].nfev != b.stats_plane[p].nfev || a.stats_time[p].nfev != b.stats_time[p].nfev;
    return bad;
}

// ------------------------------------------------------------------------------------------------
// records
// ------------------------------------------------------------------------------------------------
using VFun = std::function<void(const double x[3], double v[3])>;

VFun velocity_of(const ll::NonPeriodicSplineLagrangianField& f) {
    return [f](const double x[3], double v[3]) {
        double g[3], k;
        f.eval(x, g, k);
        v[0] = k * g[0];
        v[1] = k * g[1];
        v[2] = k * g[2];
    };
}

void initial_covectors(const VFun& vf, const double x[3], double a[3], double b[3]) {
    double v[3];
    vf(x, v);
    a[0] = -v[1];
    a[1] = v[0];
    a[2] = 0.0;
    b[0] = -v[2] / v[0];
    b[1] = 0.0;
    b[2] = 1.0;
}

/// kind 0 = plane record m of particle p, kind 1 = time record n.
template <class F> void for_each_record(const ll::LagrangianTrackResult& R, F f) {
    const std::size_t Np = static_cast<std::size_t>(R.Np), P = R.planes.size(), T = R.times.size();
    for (std::size_t p = 0; p < Np; ++p) {
        if (R.status_plane[p] == 0) {
            for (std::size_t m = 0; m < P; ++m) {
                const double x[3] = {R.planes[m], R.planes_x23[(p * P + m) * 2], R.planes_x23[(p * P + m) * 2 + 1]};
                f(0, m, p, x, R.planes_I[p * P + m], R.planes_tau[p * P + m],
                  R.covectors ? &R.planes_cov[(p * P + m) * 6] : nullptr);
            }
        }
        if (R.status_time[p] == 0) {
            for (std::size_t n = 0; n < T; ++n) {
                f(1, n, p, &R.times_x[(p * T + n) * 3], R.times_I[p * T + n], R.times[n],
                  R.covectors ? &R.times_cov[(p * T + n) * 6] : nullptr);
            }
        }
    }
}

long long count_not_ok(const ll::LagrangianTrackResult& R) {
    long long n = 0;
    for (int p = 0; p < R.Np; ++p)
        n += (R.status_plane[p] != 0) + (R.status_time[p] != 0);
    return n;
}

// ------------------------------------------------------------------------------------------------
// quadrature (Gauss-Kronrod G7K15, copied from long_box_tests.cu / lagrangian_controls_tests.cu)
// ------------------------------------------------------------------------------------------------
template <class F> double gk15(const F& f, double a, double b, double& err) {
    static const double xgk[8] = {0.991455371120812639206854697526329, 0.949107912342758524526189684047851,
                                  0.864864423359769072789712788640926, 0.741531185599394439863864773280788,
                                  0.586087235467691130294144838258730, 0.405845151377397166906606412076961,
                                  0.207784955007898467600689403773245, 0.0};
    static const double wgk[8] = {0.022935322010529224963732008058970, 0.063092092629978553290700663189204,
                                  0.104790010322250183839876322541518, 0.140653259715525918745189590510238,
                                  0.169004726639267902826583426598550, 0.190350578064785409913256402421014,
                                  0.204432940075298892414161999234649, 0.209482141084727828012999174891714};
    static const double wg[4] = {0.129484966168869693270611432679082, 0.279705391489276667901467771423780,
                                 0.381830050505118944950369775488975, 0.417959183673469387755102040816327};
    const double c = 0.5 * (a + b), hl = 0.5 * (b - a);
    const double fc = f(c);
    double rk = wgk[7] * fc, rg = wg[3] * fc;
    for (int j = 0; j < 7; ++j) {
        const double dx = hl * xgk[j];
        const double s = f(c - dx) + f(c + dx);
        rk += wgk[j] * s;
        if (j % 2 == 1)
            rg += wg[j / 2] * s;
    }
    err = std::abs((rk - rg) * hl);
    return rk * hl;
}

template <class F> double adaptive_gk(const F& f, double a, double b, double atol, int depth) {
    double err;
    const double r = gk15(f, a, b, err);
    if (err <= atol || depth >= 40)
        return r;
    const double m = 0.5 * (a + b);
    return adaptive_gk(f, a, m, 0.5 * atol, depth + 1) + adaptive_gk(f, m, b, 0.5 * atol, depth + 1);
}

/// Integral over [a, b] split at the half-cell lattice (the spline piece boundaries lie on it).
template <class F> double quad_split(const F& f, double a, double b, double h) {
    double total = 0.0;
    double x = a;
    const double step = 0.5 * h;
    long long j = static_cast<long long>(std::floor(a / step)) + 1;
    while (x < b) {
        const double nxt = std::min(b, static_cast<double>(j) * step);
        if (nxt > x)
            total += adaptive_gk(f, x, nxt, 1e-15, 0);
        x = nxt;
        ++j;
    }
    return total;
}

// ------------------------------------------------------------------------------------------------
// C1 / C2: exact controls on x1-only fields
// ------------------------------------------------------------------------------------------------
std::vector<double> c_planes(double L1, bool c2) {
    if (L1 < 3.0)
        return {0.25, 0.5, 1.0, 1.5, 1.75};
    if (c2)
        return L1 < 6.0 ? std::vector<double>{0.25, 0.5, 1.0, 2.0, 3.0, 3.75}
                        : std::vector<double>{0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5};
    return {0.25, 0.5, 1.0, 2.0, 3.0};
}
std::vector<double> c_times(double L1) {
    return L1 < 3.0 ? std::vector<double>{0.5, 1.0, 1.5} : std::vector<double>{0.5, 1.0, 2.0};
}

ll::TrackOptions control_options(double h, double L1, bool c2) {
    ll::TrackOptions to;
    to.tol = 1e-8;
    to.h_max = h / 8.0;
    to.planes = c_planes(L1, c2);
    to.times = c_times(L1);
    to.covectors = true;
    return to;
}

void case_C1(CudaContext& ctx, int nx, int ny, int nz) {
    const std::string grid = grid_str(nx, ny, nz);
    const double h = 1.0 / ny, L1 = nx * h, G1 = 1.0;
    Spline S;
    // H = G1 (L1 - x1) at the cell centres; f = -H - G1 x1 + G1 L1 (exactly 0 for dyadic h)
    build(ctx, S, nx, ny, nz,
          sample_field(nx, ny, nz, h, [&](double x, double, double) { return -(G1 * (L1 - x)) - G1 * x + G1 * L1; }),
          sample_field(nx, ny, nz, h, [&](double, double, double) { return 0.0; }), G1);
    double fmax = 0.0, cmax = 0.0;
    for (real v : S.f)
        fmax = std::max(fmax, std::abs(v));
    for (real v : S.cp)
        cmax = std::max(cmax, std::abs(v));
    for (real v : S.ck)
        cmax = std::max(cmax, std::abs(v));
    check_le("C1_f_samples_zero_exactly", grid, "host", fmax, 0.0);
    check_le("C1_coefficients_zero_exactly", grid, "host", cmax, 0.0);
    const VFun vf = velocity_of(host_field(S));
    double vdev = 0.0;
    for (double x1 : {0.0, 0.3, 0.5 * L1, L1}) {
        double v[3];
        const double x[3] = {x1, 0.37, 0.61};
        vf(x, v);
        vdev = std::max({vdev, std::abs(v[0] - G1), std::abs(v[1]), std::abs(v[2])});
    }
    check_le("C1_v_equals_G1_e1", grid, "host", vdev, 0.0);

    ll::InjectionSpec inj;
    inj.X0 = 0.0;
    inj.M = 16;
    const ll::TrackOptions to = control_options(h, L1, false);
    ll::LagrangianTrackResult R[2];
    for (int fz = 0; fz < 2; ++fz) {
        R[fz] = run_tracker(ctx, S, fz == 1, inj, to);
        const ll::LagrangianTrackResult& r = R[fz];
        const char* path = path_name(fz == 1);
        double dx = 0.0, dtau = 0.0, dcov = 0.0, mI = 0.0, dxt = 0.0;
        for_each_record(r, [&](int kind, std::size_t idx, std::size_t p, const double* x, double I, double tau,
                               const double* cov) {
            const double* sd = &r.seeds[3 * p];
            dx = std::max({dx, std::abs(x[1] - sd[1]), std::abs(x[2] - sd[2])});
            double a0[3], b0[3];
            initial_covectors(vf, sd, a0, b0);
            for (int i = 0; i < 3; ++i) {
                dcov = std::max(dcov, std::abs(cov[i] - a0[i]));
                dcov = std::max(dcov, std::abs(cov[3 + i] - b0[i]));
            }
            mI = std::max(mI, std::abs(I));
            if (kind == 0) {
                const double D = (r.planes[idx] - inj.X0) / G1;
                dtau = std::max(dtau, std::abs(tau - D) / D);
            } else {
                dxt = std::max(dxt, std::abs(x[0] - (inj.X0 + G1 * r.times[idx])) / (G1 * r.times[idx]));
            }
        });
        check_zero("C1_particles_not_ok", grid, path, count_not_ok(r));
        check_le("C1_transverse_displacement", grid, path, dx, 1e-12);
        check_le("C1_tau_vs_dx1_over_G1_rel", grid, path, dtau, 1e-9);
        check_le("C1_x1_at_times_vs_X0_plus_G1_t_rel", grid, path, dxt, 1e-9);
        check_le("C1_abs_I", grid, path, mI, 1e-14);
        check_le("C1_covectors_constant", grid, path, dcov, 1e-12);
        check_zero("C1_exit_statuses", grid, path,
                   r.totals_plane.exit_inlet + r.totals_plane.exit_outlet + r.totals_time.exit_inlet +
                       r.totals_time.exit_outlet);
    }
    check_zero("C1_fused_equals_unfused_bitwise", grid, "fused", result_mismatches(R[0], R[1]));
}

/// Exact discrete layered head (constant face flux q = G1 L1 / R); returns the cell-centre heads.
std::vector<double> layered_head(int nx, double h, double G1, const std::vector<double>& K, double& q_out) {
    // resistances per unit area: half cell at each Dirichlet face (conductance 2 K / h), harmonic face K inside
    double R = h / (2.0 * K[0]) + h / (2.0 * K[static_cast<std::size_t>(nx) - 1]);
    for (int i = 0; i + 1 < nx; ++i) {
        const double Kf = 2.0 * K[i] * K[i + 1] / (K[i] + K[i + 1]);
        R += h / Kf;
    }
    const double L1 = nx * h;
    const double q = G1 * L1 / R;
    q_out = q;
    std::vector<double> H(static_cast<std::size_t>(nx));
    H[0] = G1 * L1 - q * h / (2.0 * K[0]);
    for (int i = 0; i + 1 < nx; ++i) {
        const double Kf = 2.0 * K[i] * K[i + 1] / (K[i] + K[i + 1]);
        H[static_cast<std::size_t>(i) + 1] = H[static_cast<std::size_t>(i)] - q * h / Kf;
    }
    return H;
}

void case_C2(CudaContext& ctx, int nx, int ny, int nz) {
    const std::string grid = grid_str(nx, ny, nz);
    const double h = 1.0 / ny, L1 = nx * h, G1 = 1.0;
    std::vector<double> lnk1(static_cast<std::size_t>(nx)), K(static_cast<std::size_t>(nx));
    for (int i = 0; i < nx; ++i) {
        lnk1[static_cast<std::size_t>(i)] = 0.5 * std::cos(kTwoPi * 0.6 * ((i + 0.5) * h) + 0.3);
        K[static_cast<std::size_t>(i)] = std::exp(lnk1[static_cast<std::size_t>(i)]);
    }
    double q = 0.0;
    const std::vector<double> H = layered_head(nx, h, G1, K, q);
    std::vector<real> f(static_cast<std::size_t>(nx) * ny * nz), lnk(f.size());
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const std::size_t id = static_cast<std::size_t>(i) + static_cast<std::size_t>(nx) * (j + static_cast<std::size_t>(ny) * k);
                f[id] = -H[static_cast<std::size_t>(i)] - G1 * ((i + 0.5) * h) + G1 * L1;
                lnk[id] = lnk1[static_cast<std::size_t>(i)];
            }
    Spline S;
    build(ctx, S, nx, ny, nz, std::move(f), std::move(lnk), G1);
    const ll::NonPeriodicSplineLagrangianField hf = host_field(S);
    const VFun vf = velocity_of(hf);
    auto v1_at = [&](double x1, double x2, double x3) {
        double v[3];
        const double x[3] = {x1, x2, x3};
        vf(x, v);
        return v[0];
    };
    auto dI_at = [&](double x1, double x2, double x3) {
        double g[3], k, gl[3], Hs[6];
        const double x[3] = {x1, x2, x3};
        hf.eval_ext(x, g, k, gl, Hs);
        return (gl[0] * g[0] + gl[1] * g[1] + gl[2] * g[2] + Hs[0] + Hs[1] + Hs[2]) / g[0];
    };
    ll::InjectionSpec inj;
    inj.X0 = 0.0;
    inj.M = 16;
    const ll::TrackOptions to = control_options(h, L1, true);
    ll::LagrangianTrackResult R[2];
    for (int fz = 0; fz < 2; ++fz) {
        R[fz] = run_tracker(ctx, S, fz == 1, inj, to);
        const ll::LagrangianTrackResult& r = R[fz];
        const char* path = path_name(fz == 1);
        double dx = 0.0, dcov = 0.0, rel_tau = 0.0, abs_I = 0.0, max_Iref = 0.0;
        for_each_record(r, [&](int kind, std::size_t, std::size_t p, const double* x, double I, double tau,
                               const double* cov) {
            const double* sd = &r.seeds[3 * p];
            dx = std::max({dx, std::abs(x[1] - sd[1]), std::abs(x[2] - sd[2])});
            double a0[3], b0[3];
            initial_covectors(vf, sd, a0, b0);
            for (int i = 0; i < 3; ++i) {
                dcov = std::max(dcov, std::abs(cov[i] - a0[i]));
                dcov = std::max(dcov, std::abs(cov[3 + i] - b0[i]));
            }
            if (kind == 0) { // plane records only for tau
                const double tau_ref =
                    quad_split([&](double s) { return 1.0 / v1_at(s, sd[1], sd[2]); }, sd[0], x[0], h);
                rel_tau = std::max(rel_tau, std::abs(tau - tau_ref) / std::abs(tau_ref));
            }
            const double I_ref = quad_split([&](double s) { return dI_at(s, sd[1], sd[2]); }, sd[0], x[0], h);
            abs_I = std::max(abs_I, std::abs(I - I_ref));
            max_Iref = std::max(max_Iref, std::abs(I_ref));
        });
        check_zero("C2_particles_not_ok", grid, path, count_not_ok(r));
        check_le("C2_transverse_positions_constant", grid, path, dx, 1e-12);
        check_le("C2_covectors_constant", grid, path, dcov, 1e-12);
        check_le("C2_tau_vs_quadrature_rel", grid, path, rel_tau, 1e-9);
        record("C2_I_vs_quadrature", grid, path,
               strf("max_abs_err=%.3e max_abs_Iref=%.3e q_exact=%.15g", abs_I, max_Iref, q));
    }
    check_zero("C2_fused_equals_unfused_bitwise", grid, "fused", result_mismatches(R[0], R[1]));
}

// ------------------------------------------------------------------------------------------------
// C3: transverse periodicity of v_s on a written field with transverse structure
// ------------------------------------------------------------------------------------------------
constexpr int kEvalDoubles = 19; ///< v(3), g(3), k, grad_lnk(3), H(6), flag_eval, flag_ext, pad

template <class F> __global__ void eval_kernel(const F f, const double* pts, int n, double* out) {
    const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (p >= n)
        return;
    const double x[3] = {pts[3 * p], pts[3 * p + 1], pts[3 * p + 2]};
    double* o = out + static_cast<std::size_t>(p) * kEvalDoubles;
    double g[3], k;
    const bool a = f.eval(x, g, k);
    o[0] = k * g[0];
    o[1] = k * g[1];
    o[2] = k * g[2];
    double g2[3], k2, gl[3], H[6];
    const bool b = f.eval_ext(x, g2, k2, gl, H);
    o[3] = g2[0];
    o[4] = g2[1];
    o[5] = g2[2];
    o[6] = k2;
    o[7] = gl[0];
    o[8] = gl[1];
    o[9] = gl[2];
    for (int i = 0; i < 6; ++i)
        o[10 + i] = H[i];
    o[16] = a ? 1.0 : 0.0;
    o[17] = b ? 1.0 : 0.0;
    o[18] = 0.0;
}

template <class F>
std::vector<double> eval_device(CudaContext& ctx, const F& f, const std::vector<double>& pts) {
    const int n = static_cast<int>(pts.size() / 3);
    DeviceBuffer<double> dp(pts.size()), dout(static_cast<std::size_t>(n) * kEvalDoubles);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(dp.data(), pts.data(), pts.size() * sizeof(double), cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    eval_kernel<F><<<(n + 63) / 64, 64, 0, ctx.cuda_stream()>>>(f, dp.data(), n, dout.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    ctx.synchronize();
    std::vector<double> out(static_cast<std::size_t>(n) * kEvalDoubles);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(out.data(), dout.data(), out.size() * sizeof(double), cudaMemcpyDeviceToHost));
    return out;
}

std::vector<double> eval_host(const ll::NonPeriodicSplineLagrangianField& f, const std::vector<double>& pts) {
    const std::size_t n = pts.size() / 3;
    std::vector<double> out(n * kEvalDoubles);
    for (std::size_t p = 0; p < n; ++p) {
        const double x[3] = {pts[3 * p], pts[3 * p + 1], pts[3 * p + 2]};
        double* o = &out[p * kEvalDoubles];
        double g[3], k;
        const bool a = f.eval(x, g, k);
        o[0] = k * g[0];
        o[1] = k * g[1];
        o[2] = k * g[2];
        double g2[3], k2, gl[3], H[6];
        const bool b = f.eval_ext(x, g2, k2, gl, H);
        o[3] = g2[0];
        o[4] = g2[1];
        o[5] = g2[2];
        o[6] = k2;
        o[7] = gl[0];
        o[8] = gl[1];
        o[9] = gl[2];
        for (int i = 0; i < 6; ++i)
            o[10 + i] = H[i];
        o[16] = a ? 1.0 : 0.0;
        o[17] = b ? 1.0 : 0.0;
        o[18] = 0.0;
    }
    return out;
}

struct ShiftStats {
    double worst_abs = 0.0, worst_rel = 0.0;
    long long violating = 0, bitwise_equal_points = 0;
    long long n = 0;
};

/// v(p) vs v(p + e2), v(p + e3) from a flat eval result: base, shifted-in-x2, shifted-in-x3.
ShiftStats shift_stats(const std::vector<double>& base, const std::vector<double>& s2, const std::vector<double>& s3) {
    ShiftStats st;
    const std::size_t n = base.size() / kEvalDoubles;
    for (std::size_t p = 0; p < n; ++p) {
        const double* v0 = &base[p * kEvalDoubles];
        const double vn = std::sqrt(v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2]);
        bool bw = true;
        for (const std::vector<double>* s : {&s2, &s3}) {
            const double* v1 = &(*s)[p * kEvalDoubles];
            for (int c = 0; c < 3; ++c) {
                const double d = std::abs(v1[c] - v0[c]);
                st.worst_abs = std::max(st.worst_abs, d);
                st.worst_rel = std::max(st.worst_rel, d / vn);
                if (!(d <= 1e-14 && d <= 1e-14 * vn))
                    ++st.violating;
                bw = bw && same_bits(v1[c], v0[c]);
            }
        }
        st.bitwise_equal_points += bw ? 1 : 0;
        ++st.n;
    }
    return st;
}

void case_C3(CudaContext& ctx, int nx, int ny, int nz) {
    const std::string grid = grid_str(nx, ny, nz);
    const double h = 1.0 / ny, L1 = nx * h, G1 = 1.0;
    Spline S;
    build(ctx, S, nx, ny, nz,
          sample_field(nx, ny, nz, h,
                       [&](double x, double y, double z) {
                           return 0.1 * std::cos(kTwoPi * y) * std::cos(kTwoPi * z) * std::exp(-kTwoPi * std::sqrt(2.0) * x);
                       }),
          sample_field(nx, ny, nz, h,
                       [&](double x, double y, double z) {
                           return 0.1 * std::sin(kTwoPi * y) * std::cos(kTwoPi * z) +
                                  0.2 * std::cos(kTwoPi * 0.6 * x + 0.3);
                       }),
          G1);
    // points: 4096 random in [0, L1] x [0,1)^2, every fourth with its transverse coordinate(s) on a cell face
    std::mt19937_64 rng(20261022ULL);
    std::uniform_real_distribution<double> U(0.0, 1.0);
    std::vector<double> rp;
    for (int i = 0; i < 4096; ++i) {
        double x1 = L1 * U(rng), x2 = U(rng), x3 = U(rng);
        if (i % 4 == 3) {
            const int cyc = (i / 4) % 3;
            const double face2 = h * static_cast<double>(rng() % static_cast<unsigned>(ny));
            const double face3 = h * static_cast<double>(rng() % static_cast<unsigned>(nz));
            if (cyc == 0 || cyc == 2)
                x2 = face2;
            if (cyc == 1 || cyc == 2)
                x3 = face3;
        }
        rp.insert(rp.end(), {x1, x2, x3});
    }
    // dyadic points k / 2^16
    std::vector<double> dp;
    const std::uint64_t D = 65536;
    for (int i = 0; i < 2048; ++i) {
        const double x1 = static_cast<double>(rng() % static_cast<std::uint64_t>(L1 * D)) / D;
        const double x2 = static_cast<double>(rng() % D) / D;
        const double x3 = static_cast<double>(rng() % D) / D;
        dp.insert(dp.end(), {x1, x2, x3});
    }
    auto shifted = [](const std::vector<double>& pts, int comp) {
        std::vector<double> s = pts;
        for (std::size_t p = 0; p < pts.size() / 3; ++p)
            s[3 * p + static_cast<std::size_t>(comp)] += 1.0;
        return s;
    };
    const auto hf = host_field(S);
    const auto df = device_field(S);
    const auto ff = fused_field(S);
    struct Set {
        const char* name;
        const std::vector<double>* pts;
    } sets[2] = {{"random", &rp}, {"dyadic", &dp}};
    for (const Set& st : sets) {
        const std::vector<double> p2 = shifted(*st.pts, 1), p3 = shifted(*st.pts, 2);
        const ShiftStats sh = shift_stats(eval_host(hf, *st.pts), eval_host(hf, p2), eval_host(hf, p3));
        const auto bd = eval_device(ctx, df, *st.pts);
        const ShiftStats sd = shift_stats(bd, eval_device(ctx, df, p2), eval_device(ctx, df, p3));
        const auto bf = eval_device(ctx, ff, *st.pts);
        const ShiftStats sf = shift_stats(bf, eval_device(ctx, ff, p2), eval_device(ctx, ff, p3));
        const ShiftStats* all[3] = {&sh, &sd, &sf};
        const char* paths[3] = {"host", "device", "fused"};
        for (int i = 0; i < 3; ++i) {
            check_zero(strf("C3_%s_shift_plus_1_violations_1e-14", st.name), grid, paths[i], all[i]->violating);
            record(strf("C3_%s_shift", st.name), grid, paths[i],
                   strf("max_abs=%.3e max_rel=%.3e bitwise_equal_points=%lld of %lld", all[i]->worst_abs,
                        all[i]->worst_rel, all[i]->bitwise_equal_points, all[i]->n));
        }
        // fused == unfused on the device (eval and eval_ext, bitwise)
        long long diff = 0;
        for (std::size_t i = 0; i < bd.size(); ++i)
            diff += !same_bits(bd[i], bf[i]);
        check_zero(strf("C3_%s_fused_equals_unfused_eval_bitwise", st.name), grid, "fused", diff);
        // host vs device (recorded)
        const auto bh = eval_host(hf, *st.pts);
        double hd = 0.0;
        for (std::size_t p = 0; p < st.pts->size() / 3; ++p)
            for (int c = 0; c < 3; ++c)
                hd = std::max(hd, std::abs(bh[p * kEvalDoubles + static_cast<std::size_t>(c)] -
                                           bd[p * kEvalDoubles + static_cast<std::size_t>(c)]));
        record(strf("C3_%s_host_vs_device_v", st.name), grid, "device", strf("max_abs=%.3e", hd));
    }
    // domain report: outside [-h/2, L1 + h/2) false + NaN, no coefficient read (null coefficient pointers)
    {
        ll::NonPeriodicSplineLagrangianField nf = device_field(S);
        nf.potential.coeff = nullptr;
        nf.log_conductivity.coeff = nullptr;
        ll::FusedNonPeriodicSplineLagrangianField nff = fused_field(S);
        nff.coeff = nullptr;
        const double e = 1e-9;
        std::vector<double> out_pts = {-0.5 * h - e, 0.3, 0.3, L1 + 0.5 * h, 0.3, 0.3, -5.0, 0.3, 0.3,
                                       L1 + 3.0,      0.3, 0.3, 0.5 * L1, std::nan(""), 0.3,
                                       0.5 * L1,      0.3, std::numeric_limits<double>::infinity()};
        const auto oh = eval_host(nf, out_pts);
        const auto of = eval_device(ctx, nff, out_pts);
        long long bad = 0;
        const std::size_t n = out_pts.size() / 3;
        for (std::size_t p = 0; p < n; ++p) {
            for (const std::vector<double>* o : {&oh, &of}) {
                const double* r = &(*o)[p * kEvalDoubles];
                bad += !(r[16] == 0.0 && r[17] == 0.0 && std::isnan(r[0]) && std::isnan(r[3]) && std::isnan(r[6]) &&
                         std::isnan(r[10]));
            }
            const double x[3] = {out_pts[3 * p], out_pts[3 * p + 1], out_pts[3 * p + 2]};
            bad += nf.in_domain(x) || nff.in_domain(x);
        }
        // the margin is inside: [-h/2, 0) and [L1, L1 + h/2) evaluate (real coefficients)
        const double mx[3] = {-0.25 * h, 0.3, 0.3};
        const double my[3] = {L1 + 0.25 * h, 0.3, 0.3};
        const auto rf = host_field(S);
        double g[3], k;
        bad += !(rf.eval(mx, g, k) && std::isfinite(g[0]) && rf.eval(my, g, k) && std::isfinite(g[0]));
        check_zero("C3_domain_report_outside_false_NaN_no_read", grid, "host+fused", bad);
    }
}

// ------------------------------------------------------------------------------------------------
// C4: exits
// ------------------------------------------------------------------------------------------------
constexpr int kExitIn = ll::kStatusExitInlet, kExitOut = ll::kStatusExitOutlet;
constexpr std::uint8_t kPIn = stt::kStatusPollockExitInlet, kPOut = stt::kStatusPollockExitOutlet;

struct Faces {
    std::vector<real> U, V, W;
};
Faces uniform_faces(int nx, int ny, int nz, double u0) {
    Faces f;
    f.U.assign(ll::compact_mac_u_size(nx, ny, nz), u0);
    f.V.assign(ll::compact_mac_v_size(nx, ny, nz), 0.0);
    f.W.assign(ll::compact_mac_w_size(nx, ny, nz), 0.0);
    return f;
}

bool finite_range(const double* p, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i)
        if (!std::isfinite(p[i]))
            return false;
    return true;
}

/// k = 1 build with f = 0 (H = G1 (L1 - x1)).
void build_uniform(CudaContext& ctx, Spline& S, int nx, int ny, int nz) {
    const double h = 1.0 / ny, L1 = nx * h, G1 = 1.0;
    build(ctx, S, nx, ny, nz,
          sample_field(nx, ny, nz, h, [&](double x, double, double) { return -(G1 * (L1 - x)) - G1 * x + G1 * L1; }),
          sample_field(nx, ny, nz, h, [&](double, double, double) { return 0.0; }), G1);
}

void case_C4a(CudaContext& ctx, int nx, int ny, int nz) {
    const std::string grid = grid_str(nx, ny, nz);
    const double h = 1.0 / ny, L1 = nx * h;
    Spline S;
    build_uniform(ctx, S, nx, ny, nz);
    const ll::InjectionSpec inj{0.5, 16, 0.1, 0.9, 0.1, 0.9};
    const long long Np = static_cast<long long>(inj.M) * inj.M;
    ll::TrackOptions to;
    to.tol = 1e-8;
    to.h_max = h / 8.0;
    to.planes = {1.0, L1 + 0.5};
    to.times = {1.0, 2.0 * L1};
    ll::LagrangianTrackResult R[2];
    for (int fz = 0; fz < 2; ++fz) {
        R[fz] = run_tracker(ctx, S, fz == 1, inj, to);
        const ll::LagrangianTrackResult& r = R[fz];
        const char* path = path_name(fz == 1);
        long long not_out = 0, bad_reach = 0, bad_records = 0, bad_counts = 0;
        for (long long p = 0; p < Np; ++p) {
            const std::size_t q = static_cast<std::size_t>(p);
            not_out += (r.status_plane[q] != kExitOut) + (r.status_time[q] != kExitOut);
            bad_reach += (r.stats_plane[q].n_reached != 1) + (r.stats_time[q].n_reached != 1);
            bad_records += !(finite_range(&r.planes_x23[q * 2 * 2], 2) && std::isfinite(r.planes_tau[q * 2]) &&
                             std::isfinite(r.planes_I[q * 2]) && finite_range(&r.times_x[q * 2 * 3], 3) &&
                             std::isfinite(r.times_I[q * 2]) && std::isfinite(r.stats_plane[q].final_s) &&
                             std::isfinite(r.stats_time[q].final_s));
            bad_records += !(std::isnan(r.planes_x23[(q * 2 + 1) * 2]) && std::isnan(r.planes_tau[q * 2 + 1]) &&
                             std::isnan(r.times_x[(q * 2 + 1) * 3]) && std::isnan(r.times_I[q * 2 + 1]));
            // straight line: x2, x3 unchanged, tau = X - X0 at the plane, x1 = X0 + t at the time
            bad_records += !(std::abs(r.planes_x23[q * 2 * 2] - r.seeds[3 * q + 1]) <= 1e-12 &&
                             std::abs(r.planes_x23[q * 2 * 2 + 1] - r.seeds[3 * q + 2]) <= 1e-12 &&
                             std::abs(r.planes_tau[q * 2] - (to.planes[0] - inj.X0)) <= 1e-9 * (to.planes[0] - inj.X0) &&
                             std::abs(r.times_x[q * 2 * 3] - (inj.X0 + to.times[0])) <= 1e-9);
        }
        bad_counts += r.totals_plane.exit_outlet != Np || r.totals_plane.exit_inlet != 0 ||
                      r.totals_time.exit_outlet != Np || r.totals_time.exit_inlet != 0;
        bad_counts += !(r.exit_outlet_plane == std::vector<long long>({0, Np}) &&
                        r.exit_outlet_time == std::vector<long long>({0, Np}) &&
                        r.exit_inlet_plane == std::vector<long long>({0, 0}) &&
                        r.exit_inlet_time == std::vector<long long>({0, 0}));
        bad_counts += !(r.moments_planes[0].contributing == Np && r.moments_planes[1].contributing == 0 &&
                        r.moments_time[0].active == Np && r.moments_time[1].active == 0);
        // the 7-slot histogram keeps its contents: nothing counted for the exits
        long long hist = 0;
        for (long long v : r.totals_plane.status_hist)
            hist += v;
        for (long long v : r.totals_time.status_hist)
            hist += v;
        bad_counts += hist != 0;
        check_zero("C4a_particles_not_exit_outlet", grid, path, not_out);
        check_zero("C4a_first_plane_time_not_reached_by_all", grid, path, bad_reach);
        check_zero("C4a_reached_records_nonfinite_or_after_exit_not_NaN", grid, path, bad_records);
        check_zero("C4a_exit_counts_inexact", grid, path, bad_counts);
    }
    check_zero("C4a_fused_equals_unfused_bitwise", grid, "fused", result_mismatches(R[0], R[1]));

    // Pollock non-periodic on the k = 1 faces (U = G1, V = W = 0)
    {
        const Faces f = uniform_faces(nx, ny, nz, 1.0);
        auto dev = ll::NonPeriodicXFaceFluxes::upload(ctx, f.U, f.V, f.W, nx, ny, nz, h);
        const auto r = ll::track_pollock_lattice(ctx, dev.view(), inj, to.planes, to.times, stt::PollockParams{10000000});
        long long not_out = 0, bad_reach = 0, bad_records = 0, bad_counts = 0;
        for (long long p = 0; p < Np; ++p) {
            const std::size_t q = static_cast<std::size_t>(p);
            not_out += (r.status_plane[q] != kPOut) + (r.status_time[q] != kPOut);
            bad_reach += (r.reached_plane[q] != 1) + (r.reached_time[q] != 1);
            bad_records += !(finite_range(&r.planes_x23[q * 2 * 2], 2) && std::isfinite(r.planes_t[q * 2]) &&
                             finite_range(&r.times_x[q * 2 * 3], 3));
            bad_records += !(std::isnan(r.planes_x23[(q * 2 + 1) * 2]) && std::isnan(r.planes_t[q * 2 + 1]) &&
                             std::isnan(r.times_x[(q * 2 + 1) * 3]));
        }
        bad_counts += !(r.exit_outlet_plane == std::vector<long long>({0, Np}) &&
                        r.exit_outlet_time == std::vector<long long>({0, Np}) &&
                        r.exit_inlet_plane == std::vector<long long>({0, 0}) &&
                        r.exit_inlet_time == std::vector<long long>({0, 0}) && r.exit_outlet_plane_total == Np &&
                        r.exit_outlet_time_total == Np && r.exit_inlet_plane_total == 0 && r.exit_inlet_time_total == 0);
        bad_counts += !(r.moments_planes[0].contributing == Np && r.moments_planes[1].contributing == 0 &&
                        r.moments_time[0].active == Np && r.moments_time[1].active == 0);
        check_zero("C4a_particles_not_exit_outlet", grid, "pollock", not_out);
        check_zero("C4a_first_plane_time_not_reached_by_all", grid, "pollock", bad_reach);
        check_zero("C4a_reached_records_nonfinite_or_after_exit_not_NaN", grid, "pollock", bad_records);
        check_zero("C4a_exit_counts_inexact", grid, "pollock", bad_counts);
    }

    // straddle (rule E1) and exact-face cases, tracker only, k = 1: the exit tau = L1 - X0 is the same for all
    {
        const double te = L1 - inj.X0;
        struct Case {
            const char* name;
            std::vector<double> planes, times;
        };
        // plane L1 - 1e-3 lands before the face, plane L1 + 1e-3 is beyond it (outlet); times te -+ 1e-3 straddle
        // the exit inside one step; plane exactly L1 is landed on.
        const Case cases[3] = {{"straddle", {L1 - 1e-3, L1 + 1e-3}, {te - 1e-3, te + 1e-3}},
                               {"plane_on_L1", {L1}, {te - 1e-3}},
                               {"plane_just_beyond", {L1 + 1e-9}, {te + 1e-9}}};
        for (int c = 0; c < 3; ++c) {
            ll::TrackOptions t2 = to;
            t2.planes = cases[c].planes;
            t2.times = cases[c].times;
            ll::LagrangianTrackResult r2[2];
            for (int fz = 0; fz < 2; ++fz) {
                r2[fz] = run_tracker(ctx, S, fz == 1, inj, t2);
                const ll::LagrangianTrackResult& r = r2[fz];
                long long bad = 0;
                for (long long p = 0; p < Np; ++p) {
                    const std::size_t q = static_cast<std::size_t>(p);
                    int ep, nrp, et, nrt;
                    if (c == 0) {
                        ep = kExitOut; nrp = 1; et = kExitOut; nrt = 1; // time te-1e-3 reached, te+1e-3 not
                    } else if (c == 1) {
                        ep = 0; nrp = 1; et = 0; nrt = 1; // plane at L1 is reached; the time te-1e-3 is reached
                    } else {
                        ep = kExitOut; nrp = 0; et = kExitOut; nrt = 0; // beyond the face: exit at once
                    }
                    bad += (r.status_plane[q] != ep) + (r.stats_plane[q].n_reached != nrp) +
                           (r.status_time[q] != et) + (r.stats_time[q].n_reached != nrt);
                }
                check_zero(strf("C4a_%s_statuses", cases[c].name), grid, path_name(fz == 1), bad);
            }
        }
    }

    // rule E2: h_max = 2 h (> margin h/2): trial stages leave the evaluation domain near the faces
    {
        ll::TrackOptions t3 = to;
        t3.h_max = 2.0 * h;
        long long rejected = 0;
        for (int fz = 0; fz < 2; ++fz) {
            const ll::LagrangianTrackResult r = run_tracker(ctx, S, fz == 1, inj, t3);
            long long bad = 0;
            for (long long p = 0; p < Np; ++p) {
                const std::size_t q = static_cast<std::size_t>(p);
                bad += (r.status_plane[q] != kExitOut) + (r.status_time[q] != kExitOut) +
                       (r.stats_plane[q].n_reached != 1) + (r.stats_time[q].n_reached != 1) +
                       !finite_range(&r.planes_x23[q * 2 * 2], 2) + !finite_range(&r.times_x[q * 2 * 3], 3);
                rejected += r.stats_plane[q].rejected_steps + r.stats_time[q].rejected_steps;
            }
            check_zero("C4a_stage_outside_domain_hmax_2h_statuses", grid, path_name(fz == 1), bad);
        }
        emit("C4a_stage_outside_domain_exercised_rejected_steps", grid, "fused+unfused",
             static_cast<double>(rejected), ">0", rejected > 0);
    }

    // outputs: new keys / files only for a non-periodic result
    {
        const std::filesystem::path dir = std::filesystem::temp_directory_path() / strf("sf41_c4a_%d_%d", nx, ny);
        std::filesystem::remove_all(dir);
        ll::TrackMeta meta;
        meta.field = "c4a";
        ll::write_track_outputs(dir.string(), R[0], meta);
        std::ifstream jf(dir / "track.json");
        const nlohmann::json j = nlohmann::json::parse(jf);
        std::ifstream ep(dir / "exits_planes.csv"), et(dir / "exits_time.csv");
        std::string l0, l1, l2;
        std::getline(ep, l0);
        std::getline(ep, l1);
        std::getline(ep, l2);
        bool ok = j.value("nonperiodic_x", false) && j["plane_pass"]["exit_outlet"].get<long long>() == Np &&
                  j["time_pass"]["exit_outlet"].get<long long>() == Np && j["plane_pass"]["exit_inlet"].get<long long>() == 0 &&
                  j["exit_status_codes"]["exit_outlet"].get<int>() == kExitOut &&
                  j["plane_pass"]["status_hist"]["ok"].get<long long>() == 0 && l0 == "x1,exit_inlet,exit_outlet" &&
                  l2.find(strf(",0,%lld", Np)) != std::string::npos && et.good();
        std::filesystem::remove_all(dir);
        check_zero("C4a_track_json_and_exit_files", grid, "unfused", ok ? 0 : 1);
    }
}

void case_C4b(CudaContext& ctx, int nx, int ny, int nz) {
    const std::string grid = grid_str(nx, ny, nz);
    const double h = 1.0 / ny, L1 = nx * h, X0 = 0.1, Xp = 0.5 * L1, a = 0.5, G1 = 1.0;
    const int M = 64;
    const lr::LobeField field; // G1 = 1, a = 0.5
    const lr::LobeLattice lat = lr::make_lobe_lattice(field, L1, X0, M, Xp);
    const double T = lr::lobe_time_T(lat);
    record("C4b_reference", grid, "reference",
           strf("included=%d inlet=%d outlet=%d plane_reaching=%d T=%g psi_drift=%.2e", lat.n_included, lat.n_inlet,
                lat.n_outlet, lat.n_plane, T, lat.max_psi_drift));
    // f = a cos(2 pi x2) exp(-2 pi x1) at the cell centres, ln k = 0
    Spline S;
    build(ctx, S, nx, ny, nz,
          sample_field(nx, ny, nz, h, [&](double x, double y, double) { return a * std::cos(kTwoPi * y) * std::exp(-kTwoPi * x); }),
          sample_field(nx, ny, nz, h, [&](double, double, double) { return 0.0; }), G1);
    const ll::InjectionSpec inj{X0, M, 0.0, 1.0, 0.0, 1.0};
    ll::TrackOptions to;
    to.tol = 1e-8;
    to.h_max = h / 8.0;
    to.planes = {Xp};
    to.times = {T};
    ll::LagrangianTrackResult R[2];
    for (int fz = 0; fz < 2; ++fz) {
        R[fz] = run_tracker(ctx, S, fz == 1, inj, to);
        const ll::LagrangianTrackResult& r = R[fz];
        const char* path = path_name(fz == 1);
        long long mism_p = 0, mism_t = 0, in_p = 0, out_p = 0, reach_p = 0, in_t = 0, out_t = 0, reach_t = 0;
        for (std::size_t p = 0; p < lat.seeds.size(); ++p) {
            const lr::LobeSeed& s = lat.seeds[p];
            if (!s.included)
                continue;
            const int sp = r.status_plane[p], st = r.status_time[p];
            in_p += sp == kExitIn;
            out_p += sp == kExitOut;
            reach_p += sp == 0 && r.stats_plane[p].n_reached == 1;
            in_t += st == kExitIn;
            out_t += st == kExitOut;
            reach_t += st == 0;
            const bool ref_in = s.ref.exit == lr::ExitKind::inlet;
            const bool pl_ok = s.ref.reached_plane ? (sp == 0 && r.stats_plane[p].n_reached == 1) : (sp == kExitIn);
            const bool tm_ok = ref_in ? (st == kExitIn) : (st == kExitOut);
            mism_p += pl_ok ? 0 : 1;
            mism_t += tm_ok ? 0 : 1;
        }
        check_zero("C4b_plane_pass_misclassified_seeds", grid, path, mism_p);
        check_zero("C4b_time_pass_misclassified_seeds", grid, path, mism_t);
        check_zero("C4b_plane_counts_inexact", grid, path,
                   std::llabs(in_p - lat.n_inlet) + std::llabs(reach_p - lat.n_plane) + out_p);
        check_zero("C4b_time_counts_inexact", grid, path,
                   std::llabs(in_t - lat.n_inlet) + std::llabs(out_t - lat.n_outlet) + reach_t);
        record("C4b_counts", grid, path,
               strf("included=%d plane:inlet=%lld reached=%lld outlet=%lld time:inlet=%lld outlet=%lld", lat.n_included,
                    in_p, reach_p, out_p, in_t, out_t));
    }
    check_zero("C4b_fused_equals_unfused_bitwise", grid, "fused", result_mismatches(R[0], R[1]));

    // Pollock non-periodic on the analytically integrated face fluxes
    {
        const lr::CompactMacHost c = lr::lobe_face_fluxes(field, nx, ny, nz, h);
        auto dev = ll::NonPeriodicXFaceFluxes::upload(ctx, c.U, c.V, c.W, nx, ny, nz, h);
        const auto r = ll::track_pollock_lattice(ctx, dev.view(), inj, to.planes, to.times, stt::PollockParams{10000000});
        long long mism_p = 0, mism_t = 0, in_p = 0, reach_p = 0, in_t = 0, out_t = 0, reach_t = 0, out_p = 0;
        for (std::size_t p = 0; p < lat.seeds.size(); ++p) {
            const lr::LobeSeed& s = lat.seeds[p];
            if (!s.included)
                continue;
            const int sp = r.status_plane[p], st = r.status_time[p];
            in_p += sp == kPIn;
            out_p += sp == kPOut;
            reach_p += r.reached_plane[p] >= 1;
            in_t += st == kPIn;
            out_t += st == kPOut;
            reach_t += r.reached_time[p] >= 1;
            const bool ref_in = s.ref.exit == lr::ExitKind::inlet;
            mism_p += (s.ref.reached_plane ? (sp == 0 && r.reached_plane[p] == 1) : (sp == kPIn)) ? 0 : 1;
            mism_t += (ref_in ? (st == kPIn) : (st == kPOut)) ? 0 : 1;
        }
        check_zero("C4b_plane_pass_misclassified_seeds", grid, "pollock", mism_p);
        check_zero("C4b_time_pass_misclassified_seeds", grid, "pollock", mism_t);
        check_zero("C4b_plane_counts_inexact", grid, "pollock",
                   std::llabs(in_p - lat.n_inlet) + std::llabs(reach_p - lat.n_plane) + out_p);
        check_zero("C4b_time_counts_inexact", grid, "pollock",
                   std::llabs(in_t - lat.n_inlet) + std::llabs(out_t - lat.n_outlet) + reach_t);
    }
}

/// Recorded, no gate: under a reversed head drop every seed has non-positive flux.
void case_reversed(CudaContext& ctx, int nx, int ny, int nz) {
    const std::string grid = grid_str(nx, ny, nz);
    const double h = 1.0 / ny, L1 = nx * h, G1 = 1.0;
    Spline S;
    // H = G1 x1 (flow towards -x): f = -H - G1 x + G1 L1 = L1 - 2 x
    build(ctx, S, nx, ny, nz, sample_field(nx, ny, nz, h, [&](double x, double, double) { return -(G1 * x) - G1 * x + G1 * L1; }),
          sample_field(nx, ny, nz, h, [&](double, double, double) { return 0.0; }), G1);
    ll::TrackOptions to;
    to.tol = 1e-8;
    to.h_max = h / 8.0;
    to.planes = {1.0};
    to.times = {1.0};
    const ll::InjectionSpec inj{0.5, 16, 0.1, 0.9, 0.1, 0.9};
    for (int fz = 0; fz < 2; ++fz) {
        std::string outcome = "no error";
        try {
            run_tracker(ctx, S, fz == 1, inj, to);
        } catch (const std::exception& e) {
            outcome = std::string("exception: ") + e.what();
        }
        record("C4_reversed_head", grid, path_name(fz == 1), "outcome=\"" + outcome + "\"");
    }
}

} // namespace

int main(int argc, char** argv) {
    const auto t0 = std::chrono::steady_clock::now();
    std::string only;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--full") {
            g_full = true;
        } else if (a == "--only" && i + 1 < argc) {
            only = argv[++i];
        } else {
            std::fprintf(stderr, "usage: %s [--full] [--only C1|C2|C3|C4]\n", argv[0]);
            return 2;
        }
    }
    if (!only.empty() && only != "C1" && only != "C2" && only != "C3" && only != "C4") {
        std::fprintf(stderr, "unknown --only value %s\n", only.c_str());
        return 2;
    }
    auto want = [&](const char* c) { return only.empty() || only == c; };
    CudaContext ctx(0);
    std::printf("dirichlet_controls_tests (%s tier%s)\n", g_full ? "full" : "fast", only.empty() ? "" : (", only " + only).c_str());
    try {
        if (want("C1")) {
            if (g_full) {
                case_C1(ctx, 64, 16, 16);
                case_C1(ctx, 128, 16, 16);
            } else {
                case_C1(ctx, 32, 16, 16);
                case_C1(ctx, 64, 8, 8);
            }
        }
        if (want("C2")) {
            if (g_full) {
                case_C2(ctx, 64, 32, 32);
                case_C2(ctx, 128, 32, 32);
            } else {
                case_C2(ctx, 32, 16, 16);
            }
        }
        if (want("C3")) {
            if (g_full) {
                case_C3(ctx, 64, 16, 16);
                case_C3(ctx, 128, 16, 16);
            } else {
                case_C3(ctx, 32, 16, 16);
                case_C3(ctx, 64, 8, 8);
            }
        }
        if (want("C4")) {
            if (g_full) {
                case_C4a(ctx, 64, 16, 16);
                case_C4a(ctx, 128, 16, 16);
                case_C4b(ctx, 64, 32, 32);
                case_C4b(ctx, 128, 32, 32);
                case_reversed(ctx, 64, 16, 16);
            } else {
                case_C4a(ctx, 32, 16, 16);
                case_C4a(ctx, 64, 8, 8);
                case_C4b(ctx, 32, 32, 32);
                case_reversed(ctx, 32, 16, 16);
            }
        }
    } catch (const std::exception& e) {
        std::printf("C_ERROR exception: %s\n", e.what());
        g_all_pass = false;
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("C_TIME seconds=%.1f\n", s);
    std::printf("C_VERDICT all_pass=%d\n", g_all_pass ? 1 : 0);
    return g_all_pass ? 0 : 1;
}
