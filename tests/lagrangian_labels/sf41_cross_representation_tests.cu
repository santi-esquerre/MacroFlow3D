/**
 * @file sf41_cross_representation_tests.cu
 * @brief SF-41 N1b, program P-CROSS (threshold B, reading rule RR4): the non-periodic-in-x1 spline
 *        (C-SPLINE) against the periodic SF-28 spline on the SAME samples of the SF-19 potential
 *        fluctuation h_tilde (and of ln k), away from the x1 faces.
 *
 * Field/flow: SF-18 Gaussian exactly as LongBoxStage phase F (unit variance, normalize_variance),
 * lnK = eps Y, K = exp(lnK) on the host, SF-19 lean solve exactly as phase S2 (qbar = e1, rtol 1e-10,
 * box_auto_mg_levels, mapped b_hat). h_tilde is the `h_tilde` device buffer of the lean result,
 * copied to the host (public API) before it is released. Both splines are prefiltered on the GPU
 * and evaluated on the host with the shared inline evaluate_point_hessian (the same code the
 * batched kernels run).
 *
 * Evaluation set: 1e5 points uniform in [2 lambda, L1 - 2 lambda] x [0,1)^2 (std::mt19937_64(6001),
 * draws x, y, z per point) plus the cell centres of the x1 lines (j,k) = (0,0), (64,32), (37,101)
 * (indices mod ny, nz) at distance >= 2 lambda from both faces. Metrics: value max|dS|/max|S_p|,
 * gradient max|dg|_2/max|g_p|_2, Hessian max over points and 6 components |dH|/max|H_p|.
 * NOTE: at the cell centres both interpolants reproduce the samples, so the value difference there
 * is round-off only; the end effect shows in gradient/Hessian (and in values between knots).
 *
 * Recorded (full length of the three lines, all cell centres): relative differences (same
 * normalisers as the metrics) against the knot distance to the nearer face, and the per-knot decay
 * factor (least squares of the log difference, on the contiguous knots from the face exceeding
 * 1e3 x the interior floor = median over knots [lo, nx-1-lo], lo = min(64, nx/4)) for gradient and
 * Hessian.
 *
 * Modes: default fast tier (128x16x16, ell 0.25, lambda/h = 4, seed 6001, eps 2): smoke; gates
 * finiteness and the decay factor in [0.25, 0.29] (the 1e-11 gate is NOT applied: 2 lambda is only
 * 8 knots). `--only B`: the RR4 configuration 1664x128x128 (V100), gates 1e-11 for h_tilde.
 *
 * Output: B_METRIC / B_PROFILE / B_DECAY / B_SOLVE lines and `B_VERDICT all_pass=<0|1>`; exit 0 iff
 * the gated checks pass. Compiled with --fmad=false (CMakeLists.txt) like the other lagrangian tests.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/numerics/interpolation/NonPeriodicXTricubicBSpline.cuh"
#include "src/numerics/interpolation/PeriodicTricubicBSpline.cuh"
#include "src/physics/flow/AffinePeriodicFlowSolver.cuh"
#include "src/physics/particles/lagrangian_labels/LongBoxStage.cuh"
#include "src/physics/stochastic/PeriodicGaussianField.cuh"
#include "src/runtime/CudaContext.cuh"
#include "src/runtime/cuda_check.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using namespace macroflow3d;
namespace ip = macroflow3d::interpolation;

namespace {

struct Cfg {
    int nx = 128, ny = 16, nz = 16;
    double ell = 0.25, eps = 2.0;
    unsigned long long seed = 6001;
    bool full_b = false;
};

struct Sample {
    double v, g[3], H[6];
};

struct Pt {
    double x, y, z;
};

constexpr double kGate = 1e-11;
bool g_ok = true;

void gate(bool ok, const char* what) {
    if (!ok) {
        std::printf("B_GATE_FAIL %s\n", what);
        g_ok = false;
    }
}

double l2(const double* g) {
    return std::sqrt(g[0] * g[0] + g[1] * g[1] + g[2] * g[2]);
}

std::vector<real> to_host(const real* d, std::size_t n) {
    std::vector<real> h(n);
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(real), cudaMemcpyDeviceToHost));
    return h;
}

double dv(const Sample& a, const Sample& b) {
    return std::fabs(a.v - b.v);
}
double dg(const Sample& a, const Sample& b) {
    const double t[3] = {a.g[0] - b.g[0], a.g[1] - b.g[1], a.g[2] - b.g[2]};
    return l2(t);
}
double dH(const Sample& a, const Sample& b) {
    double m = 0;
    for (int i = 0; i < 6; ++i)
        m = std::max(m, std::fabs(a.H[i] - b.H[i]));
    return m;
}

/// Compare the two splines of `samples` (nx*ny*nz, project layout); prints the B_* lines.
void compare(const CudaContext& ctx, const Cfg& c, const Grid3D& grid, const char* field,
             const std::vector<real>& samples, const std::vector<Pt>& rnd, bool gated) {
    const std::size_t n = samples.size();
    const int nx = c.nx;
    const double h = 1.0 / c.ny, L1 = nx * h, d2 = 2.0 * c.ell;

    // ---- coefficients of both representations (GPU prefilters, downloaded) ----------------------
    std::vector<real> cp, cn;
    {
        DeviceBuffer<real> d(n);
        MACROFLOW3D_CUDA_CHECK(
            cudaMemcpy(d.data(), samples.data(), n * sizeof(real), cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
        ip::PeriodicTricubicBSplineWorkspace wp;
        ip::prefilter_periodic_tricubic_bspline(ctx, grid, DeviceSpan<const real>(d.data(), n), wp);
        ip::NonPeriodicXTricubicBSplineWorkspace wn;
        ip::prefilter_nonperiodic_x_tricubic_bspline(ctx, grid, DeviceSpan<const real>(d.data(), n),
                                                     wn);
        ctx.synchronize();
        cp = to_host(wp.coefficients.data(), n);
        cn = to_host(wn.coefficients.data(), n / nx * (nx + 4));
    }
    const auto vp = ip::make_host_view(grid, cp.data());
    const auto vn = ip::make_nonperiodic_x_view(grid, cn.data());

    // ---- point sets -------------------------------------------------------------------------------
    const int lines[3][2] = {{0, 0}, {64, 32}, {37, 101}};
    std::vector<Pt> gp = rnd; // gated set: random + interior line cell centres
    std::vector<Pt> lp;       // full lines, all cell centres
    for (const auto& l : lines) {
        const double y = ((l[0] % c.ny) + 0.5) * h, z = ((l[1] % c.nz) + 0.5) * h;
        for (int i = 0; i < nx; ++i) {
            const Pt p{(i + 0.5) * h, y, z};
            lp.push_back(p);
            if (p.x >= d2 && p.x <= L1 - d2)
                gp.push_back(p);
        }
    }

    bool finite = true;
    auto eval = [&](const std::vector<Pt>& P, std::vector<Sample>& sp, std::vector<Sample>& sn) {
        sp.resize(P.size());
        sn.resize(P.size());
        for (std::size_t q = 0; q < P.size(); ++q) {
            Sample &a = sp[q], &b = sn[q];
            ip::evaluate_point_hessian(vp, P[q].x, P[q].y, P[q].z, a.v, a.g[0], a.g[1], a.g[2], a.H);
            const bool in = ip::evaluate_point_hessian(vn, P[q].x, P[q].y, P[q].z, b.v, b.g[0],
                                                       b.g[1], b.g[2], b.H);
            finite = finite && in;
        }
    };
    std::vector<Sample> gps, gns, lps, lns;
    eval(gp, gps, gns);
    eval(lp, lps, lns);

    // ---- gated metrics ----------------------------------------------------------------------------
    double nv = 0, ng = 0, nH = 0, mv = 0, mg = 0, mH = 0;
    for (std::size_t q = 0; q < gp.size(); ++q) {
        nv = std::max(nv, std::fabs(gps[q].v));
        ng = std::max(ng, l2(gps[q].g));
        for (int i = 0; i < 6; ++i)
            nH = std::max(nH, std::fabs(gps[q].H[i]));
        mv = std::max(mv, dv(gns[q], gps[q]));
        mg = std::max(mg, dg(gns[q], gps[q]));
        mH = std::max(mH, dH(gns[q], gps[q]));
    }
    finite = finite && std::isfinite(mv) && std::isfinite(mg) && std::isfinite(mH) && nv > 0 &&
             ng > 0 && nH > 0;
    const double rel[3] = {mv / nv, mg / ng, mH / nH};
    const char* qn[3] = {"value", "gradient", "hessian"};
    for (int k = 0; k < 3; ++k) {
        const bool g = gated && c.full_b;
        std::printf("B_METRIC field=%s quantity=%s rel=%.6e gate=%s verdict=%s points=%zu\n", field,
                    qn[k], rel[k], g ? "1e-11" : "none",
                    g ? (rel[k] <= kGate ? "PASS" : "FAIL") : "RECORDED", gp.size());
        if (g)
            gate(rel[k] <= kGate, field);
    }
    gate(finite, "finite");

    // ---- profile and decay ------------------------------------------------------------------------
    std::vector<double> pg(lp.size()), ph(lp.size());
    for (std::size_t q = 0; q < lp.size(); ++q) {
        pg[q] = dg(lns[q], lps[q]) / ng;
        ph[q] = dH(lns[q], lps[q]) / nH;
    }
    for (int li = 0; li < 3; ++li) {
        const std::size_t o = static_cast<std::size_t>(li) * nx;
        const int lj = lines[li][0] % c.ny, lk = lines[li][1] % c.nz;
        for (int i = 0; i < nx; ++i) {
            const bool west = i < nx / 2;
            std::printf("B_PROFILE field=%s line=%d,%d face=%s knot=%d value=%.6e gradient=%.6e "
                        "hessian=%.6e\n",
                        field, lj, lk, west ? "west" : "east", west ? i : nx - 1 - i,
                        dv(lns[o + i], lps[o + i]) / nv, pg[o + i], ph[o + i]);
        }
        const int lo = std::min(64, nx / 4);
        for (int qi = 0; qi < 2; ++qi) {
            const std::vector<double>& pr = qi == 0 ? pg : ph;
            std::vector<double> mid(pr.begin() + o + lo, pr.begin() + o + nx - lo);
            std::nth_element(mid.begin(), mid.begin() + mid.size() / 2, mid.end());
            const double floor_ = mid[mid.size() / 2];
            for (int west = 1; west >= 0; --west) {
                auto at = [&](int d) { return pr[o + (west ? d : nx - 1 - d)]; };
                int k = 0;
                while (k < nx / 2 && k < 120 && at(k) > 1e3 * floor_)
                    ++k;
                double fac = std::nan("");
                if (k >= 4) { // least squares of log(diff) on the knot distance
                    double sx = 0, sy = 0, sxx = 0, sxy = 0;
                    for (int d = 0; d < k; ++d) {
                        const double ly = std::log(at(d));
                        sx += d;
                        sy += ly;
                        sxx += double(d) * d;
                        sxy += d * ly;
                    }
                    fac = std::exp((k * sxy - sx * sy) / (k * sxx - sx * sx));
                }
                std::printf("B_DECAY field=%s line=%d,%d face=%s factor=%.6f knots_used=%d "
                            "quantity=%s floor=%.3e\n",
                            field, lj, lk, west ? "west" : "east", fac, k,
                            qi == 0 ? "gradient" : "hessian", floor_);
                if (!c.full_b) // fast-tier gate: decay factor in [0.25, 0.29]
                    gate(fac >= 0.25 && fac <= 0.29, "decay_factor_range");
            }
        }
    }
    std::fflush(stdout);
}

} // namespace

int main(int argc, char** argv) {
    Cfg c;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--only") && i + 1 < argc && !std::strcmp(argv[i + 1], "B")) {
            c.full_b = true;
            c.nx = 1664;
            c.ny = c.nz = 128;
            ++i;
        } else {
            std::fprintf(stderr, "usage: %s [--only B]   (B = RR4 configuration, V100 only)\n",
                         argv[0]);
            return 2;
        }
    }
    CudaContext ctx(0);
    const std::size_t n = static_cast<std::size_t>(c.nx) * c.ny * c.nz;
    const double h = 1.0 / c.ny, L1 = c.nx * h;
    const Grid3D grid(c.nx, c.ny, c.nz, h, h, h);
    std::printf("B_CONFIG tier=%s grid=%dx%dx%d ell=%g lambda_over_h=%g eps=%g seed=%llu L1=%g\n",
                c.full_b ? "RR4" : "fast", c.nx, c.ny, c.nz, c.ell, c.ell / h, c.eps, c.seed, L1);

    // F: SF-18 field exactly as LongBoxStage phase F; S1: lnK = eps Y, K = exp(lnK) on the host.
    std::vector<real> lnK(n), K(n);
    {
        physics::PeriodicGaussianFieldConfig fc;
        fc.sigma2 = 1.0;
        fc.corr_length = c.ell;
        fc.seed = c.seed;
        fc.normalize_variance = true;
        DeviceBuffer<real> y(n);
        physics::PeriodicGaussianFieldWorkspace ws;
        physics::generate_periodic_gaussian_field(ctx, grid, fc, y.span(), ws);
        ctx.synchronize();
        const std::vector<real> Y = to_host(y.data(), n);
        for (std::size_t i = 0; i < n; ++i) {
            lnK[i] = c.eps * Y[i];
            K[i] = std::exp(lnK[i]);
        }
    }

    // S2: lean SF-19 solve exactly as LongBoxStage phase S2 (spec defaults: rtol 1e-10, level S+).
    std::vector<real> htilde;
    {
        physics::AffinePeriodicFlowConfig fcfg;
        fcfg.qbar[0] = 1.0;
        fcfg.qbar[1] = 0.0;
        fcfg.qbar[2] = 0.0;
        fcfg.linear.rtol = 1e-10;
        fcfg.mg.num_levels = lagrangian_labels::box_auto_mg_levels(c.nx, c.ny, c.nz);
        physics::AffinePeriodicFlowLeanOptions lo;
        lo.bhat_mapped_host = true;
        lo.keep_face_fluxes_host = false;
        auto flow = physics::solve_affine_periodic_flow_lean(ctx, grid, K, fcfg, lo);
        ctx.synchronize();
        std::vector<real>().swap(K);
        bool conv = true;
        for (int d = 0; d < 3; ++d) {
            const auto& r = flow.report.corrector_results[d];
            std::printf("B_SOLVE corrector=%d converged=%d status=%d iterations=%d initial=%.6e "
                        "final=%.6e rel=%.6e\n",
                        d + 1, int(r.converged), int(r.status), r.iterations,
                        r.initial_projected_residual, r.final_projected_residual,
                        r.relative_projected_residual);
            conv = conv && r.converged;
        }
        std::printf("B_SOLVE G=%.12e,%.12e,%.12e mg_levels=%d\n", flow.report.G[0],
                    flow.report.G[1], flow.report.G[2], fcfg.mg.num_levels);
        gate(conv, "sf19_converged");
        htilde = to_host(flow.h_tilde.data(), n); // host copy before the buffer is released
    }

    // 1e5 random points in [2 lambda, L1 - 2 lambda] x [0,1)^2
    std::vector<Pt> rnd(100000);
    {
        std::mt19937_64 rng(6001);
        std::uniform_real_distribution<double> ux(2.0 * c.ell, L1 - 2.0 * c.ell), u01(0.0, 1.0);
        for (auto& p : rnd) {
            p.x = ux(rng);
            p.y = u01(rng);
            p.z = u01(rng);
        }
    }
    compare(ctx, c, grid, "h_tilde", htilde, rnd, true);
    compare(ctx, c, grid, "lnK", lnK, rnd, false);

    std::printf("B_VERDICT all_pass=%d\n", int(g_ok));
    return g_ok ? 0 : 1;
}
