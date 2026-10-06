/**
 * @file SlabProductionSetup.cu
 * @brief SF-33 N3: production inputs (SF-18 / analytic field, spectral vertex values, SF-19 Darcy,
 *        D-1 inlet labels, SF-28 splines, reference vD). See SlabProductionSetup.cuh.
 */

#include "SlabProductionSetup.cuh"

#include "../../../runtime/cuda_check.cuh"
#include "SlabResidual.cuh"

#include "apps/closure_gate/closure_fields.hpp"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <utility>

#include <cufft.h>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

namespace {

constexpr real kPi = 3.141592653589793238462643383279502884;

void cufft_check(cufftResult r, const char* what) {
    if (r != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string("inlet_slab spectral evaluation: cuFFT failure in ") +
                                 what + " (code " + std::to_string(static_cast<int>(r)) + ")");
    }
}

__device__ inline int signed_mode(int i, int N) {
    return i < N / 2 ? i : i - N;
}

/// out_v = F * phase / N^3, out_g[d] = out_v * (i k_d); any Nyquist component -> 0.
__global__ void spectral_multiply_kernel(const cufftDoubleComplex* __restrict__ F, int N, int sx,
                                         int sy, int sz, cufftDoubleComplex* __restrict__ ov,
                                         cufftDoubleComplex* __restrict__ ogx,
                                         cufftDoubleComplex* __restrict__ ogy,
                                         cufftDoubleComplex* __restrict__ ogz) {
    const std::size_t n3 = static_cast<std::size_t>(N) * N * N;
    for (std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < n3; idx += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const int i = static_cast<int>(idx % N);
        const int j = static_cast<int>((idx / N) % N);
        const int k = static_cast<int>(idx / (static_cast<std::size_t>(N) * N));
        const bool nyq = (i == N / 2) || (j == N / 2) || (k == N / 2);
        cufftDoubleComplex zero;
        zero.x = 0.0;
        zero.y = 0.0;
        if (nyq) {
            ov[idx] = zero;
            ogx[idx] = zero;
            ogy[idx] = zero;
            ogz[idx] = zero;
            continue;
        }
        const int mx = signed_mode(i, N), my = signed_mode(j, N), mz = signed_mode(k, N);
        // phase exp(-i k h / 2) per shifted axis: k h / 2 = pi m / N
        const real th =
            -kPi *
            (static_cast<real>(sx * mx) + static_cast<real>(sy * my) + static_cast<real>(sz * mz)) /
            static_cast<real>(N);
        real s, c;
        sincos(th, &s, &c);
        const real inv = 1.0 / static_cast<real>(n3);
        const real vr = (F[idx].x * c - F[idx].y * s) * inv;
        const real vi = (F[idx].x * s + F[idx].y * c) * inv;
        ov[idx].x = vr;
        ov[idx].y = vi;
        const real kx = 2.0 * kPi * mx, ky = 2.0 * kPi * my, kz = 2.0 * kPi * mz;
        // (vr + i vi) * (i k) = -k vi + i k vr
        ogx[idx].x = -kx * vi;
        ogx[idx].y = kx * vr;
        ogy[idx].x = -ky * vi;
        ogy[idx].y = ky * vr;
        ogz[idx].x = -kz * vi;
        ogz[idx].y = kz * vr;
    }
}

std::vector<real> download_real_part(const DeviceBuffer<cufftDoubleComplex>& d, std::size_t n) {
    std::vector<cufftDoubleComplex> h(n);
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(h.data(), d.data(), n * sizeof(cufftDoubleComplex), cudaMemcpyDeviceToHost));
    std::vector<real> out(n);
    for (std::size_t i = 0; i < n; ++i)
        out[i] = h[i].x;
    return out;
}

std::vector<real> download(const real* d, std::size_t n) {
    std::vector<real> h(n);
    if (n > 0)
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(real), cudaMemcpyDeviceToHost));
    return h;
}

void upload(DeviceBuffer<real>& d, const std::vector<real>& h) {
    d.resize(h.size());
    if (!h.empty()) {
        MACROFLOW3D_CUDA_CHECK(
            cudaMemcpy(d.data(), h.data(), h.size() * sizeof(real), cudaMemcpyHostToDevice));
        // SF-33 C2: legacy-stream copy must land before ctx-stream work
        MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    }
}

DeviceSpan<const real> cspan(const DeviceBuffer<real>& b) {
    return DeviceSpan<const real>(b.data(), b.size());
}

const char* pcg_status_name(solvers::ProjectedPCGStatus s) {
    switch (s) {
    case solvers::ProjectedPCGStatus::converged:
        return "converged";
    case solvers::ProjectedPCGStatus::max_iterations:
        return "max_iterations";
    case solvers::ProjectedPCGStatus::invalid_configuration:
        return "invalid_configuration";
    case solvers::ProjectedPCGStatus::size_mismatch:
        return "size_mismatch";
    case solvers::ProjectedPCGStatus::aliasing:
        return "aliasing";
    case solvers::ProjectedPCGStatus::breakdown_pAp:
        return "breakdown_pAp";
    case solvers::ProjectedPCGStatus::breakdown_rz:
        return "breakdown_rz";
    case solvers::ProjectedPCGStatus::nonfinite_value:
        return "nonfinite_value";
    }
    return "unknown";
}

} // namespace

// ================================================================================================
// Analytic fields
// ================================================================================================

real analytic_log_conductivity_value(const std::string& field, real X, real Y, real Z) {
    return closure_gate::analytic_log_conductivity(closure_gate::analytic_field_from_name(field), X,
                                                   Y, Z);
}

void analytic_log_conductivity_gradient(const std::string& field, real X, real Y, real Z,
                                        real g[3]) {
    const double TP = 2.0 * 3.141592653589793; // = closure_fields.hpp TWO_PI
    switch (closure_gate::analytic_field_from_name(field)) {
    case closure_gate::AnalyticField::generic3d: {
        // cos(a) + cos(b) + 0.8 cos(c) + 0.6 sin(d)
        const double a = TP * (X + Y), b = TP * (X + Z) + 0.7, c = TP * (X - Y + Z) + 1.3,
                     d = TP * (2 * X + Y - Z);
        const double sa = std::sin(a), sb = std::sin(b), sc = std::sin(c), cd = std::cos(d);
        g[0] = -TP * sa - TP * sb - 0.8 * TP * sc + 0.6 * 2.0 * TP * cd;
        g[1] = -TP * sa + 0.8 * TP * sc + 0.6 * TP * cd;
        g[2] = -TP * sb - 0.8 * TP * sc - 0.6 * TP * cd;
        return;
    }
    case closure_gate::AnalyticField::two_mode: {
        const double sa = std::sin(TP * (X + Y)), sb = std::sin(TP * (X + Z));
        g[0] = -TP * (sa + sb);
        g[1] = -TP * sa;
        g[2] = -TP * sb;
        return;
    }
    case closure_gate::AnalyticField::control2d: {
        // cos(a) + 0.8 sin(e) + 0.5 cos(2 pi Y), e = 2 pi (2X - Y) + 0.4
        const double sa = std::sin(TP * (X + Y)), ce = std::cos(TP * (2 * X - Y) + 0.4);
        g[0] = -TP * sa + 0.8 * 2.0 * TP * ce;
        g[1] = -TP * sa - 0.8 * TP * ce - 0.5 * TP * std::sin(TP * Y);
        g[2] = 0.0;
        return;
    }
    case closure_gate::AnalyticField::lester2021:
    case closure_gate::AnalyticField::lester_brk: {
        // sin(2pi X) cos(2pi Y) sin(2pi Z) + 0.4 sin(2pi X + phi) sin(4 2pi Z)
        const double phi =
            closure_gate::analytic_field_from_name(field) == closure_gate::AnalyticField::lester_brk
                ? 0.9
                : 0.0;
        const double sx = std::sin(TP * X), cx = std::cos(TP * X);
        const double sy = std::sin(TP * Y), cy = std::cos(TP * Y);
        const double sz = std::sin(TP * Z), cz = std::cos(TP * Z);
        const double s4 = std::sin(4 * TP * Z), c4 = std::cos(4 * TP * Z);
        const double sxp = std::sin(TP * X + phi), cxp = std::cos(TP * X + phi);
        g[0] = TP * cx * cy * sz + 0.4 * TP * cxp * s4;
        g[1] = -TP * sx * sy * sz;
        g[2] = TP * sx * cy * cz + 0.4 * 4.0 * TP * sxp * c4;
        return;
    }
    }
    throw std::invalid_argument("analytic_log_conductivity_gradient: unknown field");
}

// ================================================================================================
// Spectral evaluation
// ================================================================================================

void spectral_periodic_evaluate(const CudaContext& ctx, int N, const std::vector<real>& cell,
                                const std::array<bool, 3>& to_vertex, std::vector<real>& value,
                                std::array<std::vector<real>, 3>& grad) {
    if (N < 4 || N % 2 != 0) {
        throw std::invalid_argument("spectral_periodic_evaluate: N must be even and >= 4");
    }
    const std::size_t n3 = static_cast<std::size_t>(N) * N * N;
    if (cell.size() != n3) {
        throw std::invalid_argument("spectral_periodic_evaluate: cell_samples.size() != N^3");
    }
    std::vector<cufftDoubleComplex> h(n3);
    for (std::size_t i = 0; i < n3; ++i) {
        h[i].x = cell[i];
        h[i].y = 0.0;
    }
    DeviceBuffer<cufftDoubleComplex> in(n3), F(n3), ov(n3),
        og[3] = {DeviceBuffer<cufftDoubleComplex>(n3), DeviceBuffer<cufftDoubleComplex>(n3),
                 DeviceBuffer<cufftDoubleComplex>(n3)};
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(in.data(), h.data(), n3 * sizeof(cufftDoubleComplex), cudaMemcpyHostToDevice));
    // SF-33 C2: legacy-stream copy must land before ctx-stream work
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    cufftHandle plan;
    cufft_check(cufftPlan3d(&plan, N, N, N, CUFFT_Z2Z), "cufftPlan3d");
    try {
        cufft_check(cufftSetStream(plan, ctx.cuda_stream()), "cufftSetStream");
        cufft_check(cufftExecZ2Z(plan, in.data(), F.data(), CUFFT_FORWARD), "forward");
        const int blocks = static_cast<int>(std::min<std::size_t>((n3 + 255) / 256, 4096));
        spectral_multiply_kernel<<<blocks, 256, 0, ctx.cuda_stream()>>>(
            F.data(), N, to_vertex[0] ? 1 : 0, to_vertex[1] ? 1 : 0, to_vertex[2] ? 1 : 0,
            ov.data(), og[0].data(), og[1].data(), og[2].data());
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        cufft_check(cufftExecZ2Z(plan, ov.data(), ov.data(), CUFFT_INVERSE), "inverse value");
        for (int d = 0; d < 3; ++d)
            cufft_check(cufftExecZ2Z(plan, og[d].data(), og[d].data(), CUFFT_INVERSE),
                        "inverse gradient");
        ctx.synchronize();
    } catch (...) {
        cufftDestroy(plan);
        throw;
    }
    cufftDestroy(plan);
    value = download_real_part(ov, n3);
    for (int d = 0; d < 3; ++d)
        grad[d] = download_real_part(og[d], n3);
}

void cell_lattice_to_slab_full(int N, const std::vector<real>& lattice, std::vector<real>& full) {
    const std::size_t n2 = static_cast<std::size_t>(N) * N;
    if (lattice.size() != n2 * N) {
        throw std::invalid_argument("cell_lattice_to_slab_full: lattice.size() != N^3");
    }
    full.assign(n2 * (N + 1), 0.0);
    for (int j = 0; j <= N; ++j) {
        const int i = j == N ? 0 : j; // vertex plane N = plane 0 (periodic field)
        for (int m2 = 0; m2 < N; ++m2)
            for (int m3 = 0; m3 < N; ++m3)
                full[static_cast<std::size_t>(m3) + N * (static_cast<std::size_t>(m2) + N * j)] =
                    lattice[static_cast<std::size_t>(i) +
                            N * (static_cast<std::size_t>(m2) + static_cast<std::size_t>(N) * m3)];
    }
}

int slab_auto_mg_levels(int N) {
    int levels = 1;
    int n = N;
    while (n % 2 == 0 && n / 2 >= 4 && (n / 2) % 2 == 0) {
        n /= 2;
        ++levels;
    }
    return levels;
}

// ================================================================================================
// Field sources
// ================================================================================================

void SlabFieldSource::fill_spectral(CudaContext& ctx) {
    const int N = n_;
    std::vector<real> v;
    std::array<std::vector<real>, 3> g;
    spectral_periodic_evaluate(ctx, N, cell_, {true, true, true}, v, g);
    cell_lattice_to_slab_full(N, v, vtx_);
    for (int d = 0; d < 3; ++d)
        cell_lattice_to_slab_full(N, g[d], vgrad_[d]);
    // inlet face centres: shift in x1 only, take the lattice plane i = 0
    spectral_periodic_evaluate(ctx, N, cell_, {true, false, false}, v, g);
    face_.assign(static_cast<std::size_t>(N) * N, 0.0);
    for (int m2 = 0; m2 < N; ++m2)
        for (int m3 = 0; m3 < N; ++m3)
            face_[static_cast<std::size_t>(m3) + static_cast<std::size_t>(N) * m2] =
                v[static_cast<std::size_t>(N) *
                  (static_cast<std::size_t>(m2) + static_cast<std::size_t>(N) * m3)];
}

SlabFieldSource SlabFieldSource::gaussian(CudaContext& ctx, const ProductionFieldSpec& spec) {
    if (spec.N < 4 || spec.N % 2 != 0) {
        throw std::invalid_argument("SlabFieldSource::gaussian: N must be even and >= 4");
    }
    SlabFieldSource s;
    s.kind_ = SlabFieldKind::gaussian;
    s.name_ = "gaussian";
    s.n_ = spec.N;
    const int N = spec.N;
    const real h = 1.0 / static_cast<real>(N);
    const Grid3D grid(N, N, N, h, h, h);
    physics::PeriodicGaussianFieldConfig cfg;
    cfg.sigma2 = spec.sigma2;
    cfg.corr_length = spec.ell;
    cfg.seed = spec.seed;
    cfg.normalize_variance = spec.normalize_variance;
    DeviceBuffer<real> y(grid.num_cells());
    physics::PeriodicGaussianFieldWorkspace ws;
    s.sf18_ = physics::generate_periodic_gaussian_field(ctx, grid, cfg, y.span(), ws);
    s.has_sf18_ = true;
    ctx.synchronize();
    s.cell_ = download(y.data(), grid.num_cells());
    s.fill_spectral(ctx);
    return s;
}

SlabFieldSource SlabFieldSource::spectral(CudaContext& ctx, const std::string& name, int N,
                                          std::vector<real> cell_samples) {
    if (N < 4 || N % 2 != 0) {
        throw std::invalid_argument("SlabFieldSource::spectral: N must be even and >= 4");
    }
    SlabFieldSource s;
    s.kind_ = SlabFieldKind::spectral;
    s.name_ = name;
    s.n_ = N;
    s.cell_ = std::move(cell_samples);
    s.fill_spectral(ctx);
    return s;
}

SlabFieldSource SlabFieldSource::analytic(const std::string& field, int N) {
    if (N < 4 || N % 2 != 0) {
        throw std::invalid_argument("SlabFieldSource::analytic: N must be even and >= 4");
    }
    const closure_gate::AnalyticField f = closure_gate::analytic_field_from_name(field);
    SlabFieldSource s;
    s.kind_ = SlabFieldKind::analytic;
    s.name_ = field;
    s.n_ = N;
    const real h = 1.0 / static_cast<real>(N);
    const Grid3D grid(N, N, N, h, h, h);
    closure_gate::fill_analytic_log_conductivity(grid, f, 1.0, s.cell_);
    const std::size_t n2 = static_cast<std::size_t>(N) * N;
    s.vtx_.assign(n2 * (N + 1), 0.0);
    for (auto& g : s.vgrad_)
        g.assign(n2 * (N + 1), 0.0);
    for (int j = 0; j <= N; ++j)
        for (int m2 = 0; m2 < N; ++m2)
            for (int m3 = 0; m3 < N; ++m3) {
                const real X = static_cast<real>(j) / static_cast<real>(N);
                const real Y = static_cast<real>(m2) / static_cast<real>(N);
                const real Z = static_cast<real>(m3) / static_cast<real>(N);
                const std::size_t idx =
                    static_cast<std::size_t>(m3) +
                    N * (static_cast<std::size_t>(m2) + N * static_cast<std::size_t>(j));
                s.vtx_[idx] = closure_gate::analytic_log_conductivity(f, X, Y, Z);
                real g[3];
                analytic_log_conductivity_gradient(field, X, Y, Z, g);
                for (int d = 0; d < 3; ++d)
                    s.vgrad_[d][idx] = g[d];
            }
    s.face_.assign(n2, 0.0);
    for (int m2 = 0; m2 < N; ++m2)
        for (int m3 = 0; m3 < N; ++m3)
            s.face_[static_cast<std::size_t>(m3) + static_cast<std::size_t>(N) * m2] =
                closure_gate::analytic_log_conductivity(f, 0.0, (static_cast<real>(m2) + 0.5) * h,
                                                        (static_cast<real>(m3) + 0.5) * h);
    return s;
}

// ================================================================================================
// Production stage
// ================================================================================================

const char* to_string(SlabProductionStatus s) {
    switch (s) {
    case SlabProductionStatus::ok:
        return "ok";
    case SlabProductionStatus::darcy_failed:
        return "darcy_failed";
    case SlabProductionStatus::inlet_backflow:
        return "inlet_backflow";
    }
    return "unknown";
}

std::string ProductionStageReport::summary() const {
    std::string out;
    char b[512];
    std::snprintf(b, sizeof(b), "STAGE field=%s eps=%g N=%d status=%s\n", field.c_str(), eps, N,
                  to_string(status));
    out += b;
    if (has_sf18) {
        std::snprintf(b, sizeof(b),
                      "  SF-18: raw_mean=%.6e raw_variance=%.12e applied_scale=%.12e "
                      "final_variance=%.12e active_modes=%zu\n",
                      sf18.raw_mean, sf18.raw_variance, sf18.applied_scale, sf18.final_variance,
                      sf18.active_mode_count);
        out += b;
    }
    std::snprintf(b, sizeof(b),
                  "  SF-19: mg_levels=%d converged=%d G=(%.12e, %.12e, %.12e) mean_flux=(%.3e, "
                  "%.3e, %.3e) div_max=%.3e div_rms=%.3e\n",
                  mg_levels, darcy_converged ? 1 : 0, darcy.G[0], darcy.G[1], darcy.G[2],
                  darcy.achieved_mean_flux[0], darcy.achieved_mean_flux[1],
                  darcy.achieved_mean_flux[2], darcy.div_max_abs, darcy.div_rms);
    out += b;
    for (int d = 0; d < 3; ++d) {
        const auto& r = darcy.corrector_results[d];
        std::snprintf(
            b, sizeof(b),
            "  SF-19 corrector %c: status=%s its=%d rel_residual=%.3e final_residual=%.3e\n",
            "xyz"[d], pcg_status_name(r.status), r.iterations, r.relative_projected_residual,
            r.final_projected_residual);
        out += b;
    }
    std::snprintf(b, sizeof(b),
                  "  inlet: vmin=%.6e mean(U plane 0)=%.15f (mean-1=%.3e) Q0-1=%.3e v_rms=%.6e\n",
                  inlet_vmin, inlet_mean, inlet_mean - 1.0, Q0 - 1.0, v_rms);
    out += b;
    std::snprintf(b, sizeof(b),
                  "  v1(U-face) vs spline flow k g1: rms_rel=%.6e max_rel=%.6e; spline potential "
                  "GPU vs host prefilter rel=%.3e\n",
                  v1_diff_rms_rel, v1_diff_max_rel, spline_potential_gpu_vs_host_rel);
    out += b;
    return out;
}

SlabSplineDirectionField ProductionStage::direction_field() const {
    if (potential_coefficients.empty() || logk_coefficients.empty()) {
        throw std::logic_error("ProductionStage::direction_field: no splines (stage not ok)");
    }
    SlabSplineDirectionField f;
    f.potential = interpolation::make_host_view(cell_grid, potential_coefficients.data());
    f.log_conductivity = interpolation::make_host_view(cell_grid, logk_coefficients.data());
    for (int d = 0; d < 3; ++d)
        f.G[d] = G[d];
    return f;
}

ProductionStage build_production_stage(CudaContext& ctx, const InletSlabGrid& grid,
                                       const SlabFieldSource& source, real eps,
                                       const ProductionStageOptions& opt) {
    require_valid_grid(grid, "build_production_stage");
    if (source.N() != grid.n) {
        throw std::invalid_argument("build_production_stage: source N != grid N");
    }
    if (!std::isfinite(eps)) {
        throw std::invalid_argument("build_production_stage: eps must be finite");
    }
    const int N = grid.n;
    const real h = grid.h;
    const std::size_t ncell = static_cast<std::size_t>(N) * N * N;
    const std::size_t n2 = grid.plane_size();
    const std::size_t nfull = grid.full_size();

    ProductionStage S;
    S.grid = grid;
    S.cell_grid = Grid3D(N, N, N, h, h, h);
    ProductionStageReport& R = S.report;
    R.field = source.name();
    R.eps = eps;
    R.N = N;
    R.has_sf18 = source.has_sf18();
    if (R.has_sf18)
        R.sf18 = source.sf18();

    // ---- (ii) SF-19 on K = exp(eps Y_cell)
    std::vector<real> lnK(ncell), K(ncell);
    for (std::size_t c = 0; c < ncell; ++c) {
        lnK[c] = eps * source.cell_samples()[c];
        K[c] = std::exp(lnK[c]);
    }
    DeviceBuffer<real> lnK_dev, K_dev;
    upload(lnK_dev, lnK);
    upload(K_dev, K);
    physics::AffinePeriodicFlowConfig fcfg;
    fcfg.qbar[0] = 1.0;
    fcfg.qbar[1] = 0.0;
    fcfg.qbar[2] = 0.0;
    fcfg.linear.rtol = opt.pcg_rtol;
    if (opt.pcg_max_iter != -1)
        fcfg.linear.max_iter = opt.pcg_max_iter;
    R.mg_levels = opt.mg_levels == 0 ? slab_auto_mg_levels(N) : opt.mg_levels;
    fcfg.mg.num_levels = R.mg_levels;
    physics::AffinePeriodicFlowWorkspace fws;
    DeviceBuffer<real> u(static_cast<std::size_t>(N + 1) * N * N),
        v(static_cast<std::size_t>(N) * (N + 1) * N), w(static_cast<std::size_t>(N) * N * (N + 1));
    R.darcy = physics::solve_affine_periodic_flow(
        ctx, S.cell_grid, cspan(K_dev), fcfg,
        physics::AffinePeriodicVelocityView{u.span(), v.span(), w.span()}, fws);
    ctx.synchronize();
    R.darcy_converged = R.darcy.corrector_results[0].converged &&
                        R.darcy.corrector_results[1].converged &&
                        R.darcy.corrector_results[2].converged;
    if (!R.darcy_converged) {
        R.status = SlabProductionStatus::darcy_failed;
        return S;
    }
    for (int d = 0; d < 3; ++d)
        S.G[d] = R.darcy.G[d];

    // ---- (iii) inlet v1 = U-face plane i = 0, face centres ((j + 1/2) h, (k + 1/2) h)
    {
        const std::vector<real> U = download(u.data(), u.size());
        S.inlet_v1_samples.assign(n2, 0.0);
        real sum = 0.0;
        for (int j = 0; j < N; ++j)
            for (int k = 0; k < N; ++k) {
                const real val = U[static_cast<std::size_t>(j) * (N + 1) +
                                   static_cast<std::size_t>(k) * (N + 1) * N];
                S.inlet_v1_samples[static_cast<std::size_t>(k) + static_cast<std::size_t>(N) * j] =
                    val;
                sum += val;
            }
        R.inlet_mean = sum / static_cast<real>(n2);
    }
    try {
        S.labels = InletLabels::build(S.inlet_v1_samples, N, 0.5, 0.5);
    } catch (const InletBackflowError& e) {
        R.inlet_vmin = e.vmin();
        R.status = SlabProductionStatus::inlet_backflow;
        return S;
    }
    R.inlet_vmin = S.labels.vmin();
    R.Q0 = S.labels.Q0();
    S.labels.prepare_device(opt.label_chunk);

    // ---- (iv) splines of h_tilde and of eps Y_cell
    {
        const DeviceSpan<const real> htilde = fws.potential_fluctuation();
        interpolation::PeriodicTricubicBSplineWorkspace sh_ws, sy_ws;
        interpolation::prefilter_periodic_tricubic_bspline(ctx, S.cell_grid, htilde, sh_ws);
        interpolation::prefilter_periodic_tricubic_bspline(ctx, S.cell_grid, cspan(lnK_dev), sy_ws);
        ctx.synchronize();
        S.potential_coefficients = download(sh_ws.coefficients.data(), ncell);
        S.logk_coefficients = download(sy_ws.coefficients.data(), ncell);
        const std::vector<real> h_host = download(htilde.data(), ncell);
        std::vector<real> c_host(ncell);
        interpolation::prefilter_periodic_tricubic_bspline_host(S.cell_grid, h_host.data(),
                                                                c_host.data());
        real dmax = 0.0, cmax = 0.0;
        for (std::size_t c = 0; c < ncell; ++c) {
            dmax = std::max(dmax, std::abs(S.potential_coefficients[c] - c_host[c]));
            cmax = std::max(cmax, std::abs(c_host[c]));
        }
        R.spline_potential_gpu_vs_host_rel = cmax > 0.0 ? dmax / cmax : dmax;

        // ---- (v) vD at every vertex: GPU batched evaluation of grad s_h
        std::vector<real> px(nfull), py(nfull), pz(nfull);
        for (int j = 0; j <= N; ++j)
            for (int m2 = 0; m2 < N; ++m2)
                for (int m3 = 0; m3 < N; ++m3) {
                    const std::size_t idx = grid.full_index(j, m2, m3);
                    px[idx] = grid.coord(j);
                    py[idx] = grid.coord(m2);
                    pz[idx] = grid.coord(m3);
                }
        DeviceBuffer<real> dpx, dpy, dpz, val(nfull), gx(nfull), gy(nfull), gz(nfull);
        upload(dpx, px);
        upload(dpy, py);
        upload(dpz, pz);
        interpolation::evaluate_periodic_tricubic_bspline(ctx, sh_ws.view(), cspan(dpx), cspan(dpy),
                                                          cspan(dpz), val.span(), gx.span(),
                                                          gy.span(), gz.span());
        ctx.synchronize();
        const std::vector<real> hg[3] = {download(gx.data(), nfull), download(gy.data(), nfull),
                                         download(gz.data(), nfull)};
        std::vector<real> lnk(nfull), glnk[3], vD[3];
        for (int d = 0; d < 3; ++d) {
            glnk[d].resize(nfull);
            vD[d].resize(nfull);
        }
        real sum_v2 = 0.0;
        for (std::size_t i = 0; i < nfull; ++i) {
            lnk[i] = eps * source.vertex_value()[i];
            const real kv = std::exp(lnk[i]);
            for (int d = 0; d < 3; ++d) {
                glnk[d][i] = eps * source.vertex_grad(d)[i];
                vD[d][i] = kv * (S.G[d] + hg[d][i]);
                sum_v2 += vD[d][i] * vD[d][i];
            }
        }
        R.v_rms = std::sqrt(sum_v2 / static_cast<real>(nfull));

        // ---- (vi) U-face v1 vs spline flow k g1 at the face centres
        const SlabSplineDirectionField fld = S.direction_field();
        real sd2 = 0.0, su2 = 0.0, dmx = 0.0;
        for (int m2 = 0; m2 < N; ++m2)
            for (int m3 = 0; m3 < N; ++m3) {
                const std::size_t p =
                    static_cast<std::size_t>(m3) + static_cast<std::size_t>(N) * m2;
                const real x[3] = {0.0, (static_cast<real>(m2) + 0.5) * h,
                                   (static_cast<real>(m3) + 0.5) * h};
                real g[3], kk;
                fld(x, g, kk);
                const real v1s = std::exp(eps * source.inlet_face_value()[p]) * g[0];
                const real Uv = S.inlet_v1_samples[p];
                const real d = Uv - v1s;
                sd2 += d * d;
                su2 += Uv * Uv;
                dmx = std::max(dmx, std::abs(d) / std::abs(Uv));
            }
        R.v1_diff_rms_rel = std::sqrt(sd2 / su2);
        R.v1_diff_max_rel = dmx;

        // ---- fill the stage inputs and the reference
        S.inputs.allocate(grid);
        S.inputs.field = source.name();
        S.inputs.eps = eps;
        S.inputs.v_rms = R.v_rms;
        auto put = [](DeviceBuffer<real>& d, const std::vector<real>& hv) {
            MACROFLOW3D_CUDA_CHECK(
                cudaMemcpy(d.data(), hv.data(), hv.size() * sizeof(real), cudaMemcpyHostToDevice));
            // SF-33 C2: legacy-stream copy must land before ctx-stream work
            MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
        };
        put(S.inputs.lnk, lnk);
        for (int d = 0; d < 3; ++d)
            put(S.inputs.grad_lnk[d], glnk[d]);
        fill_q_from_lnk(ctx, grid, S.inputs);
        std::vector<real> vp2(n2), vp3(n2);
        for (std::size_t p = 0; p < n2; ++p) {
            vp2[p] = vD[1][p]; // plane 0 occupies the first n2 entries of a full array
            vp3[p] = vD[2][p];
        }
        put(S.inputs.vperp_in[0], vp2);
        put(S.inputs.vperp_in[1], vp3);
        S.reference.allocate(grid, true, false);
        for (int d = 0; d < 3; ++d)
            put(S.reference.vD[d], vD[d]);

        // inlet periodic parts u0_i = psi0_i - coord at the inlet vertices (GPU label evaluation)
        DeviceBuffer<real> p1(n2), p2(n2);
        S.labels.evaluate_labels(ctx, DeviceSpan<const real>(dpy.data(), n2),
                                 DeviceSpan<const real>(dpz.data(), n2), p1.span(), p2.span());
        ctx.synchronize();
        std::vector<real> h1 = download(p1.data(), n2), h2 = download(p2.data(), n2);
        for (int m2 = 0; m2 < N; ++m2)
            for (int m3 = 0; m3 < N; ++m3) {
                const std::size_t p = grid.plane_index(m2, m3);
                h1[p] = h1[p] - grid.coord(m2);
                h2[p] = h2[p] - grid.coord(m3);
            }
        put(S.inputs.u0[0], h1);
        put(S.inputs.u0[1], h2);
        ctx.synchronize();
    }
    R.status = SlabProductionStatus::ok;
    return S;
}

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
