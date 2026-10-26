/**
 * @file DirichletLongBoxStage.cu
 * @brief SF-41 N2 lean medium (b) stage, flow part. See DirichletLongBoxStage.cuh.
 *
 * Compiled with the default flags like DirichletMedium.cu (whose file-local diagnostics kernels are copied
 * below verbatim, so they compile identically) and LongBoxStage.cu.
 */

#include "DirichletLongBoxStage.cuh"

#include "LagrangianTracker.cuh"
#include "../../../multigrid/coefficient_hierarchy.cuh"
#include "../../../multigrid/mg_types.hpp"
#include "../../../numerics/interpolation/NonPeriodicXTricubicBSpline.cuh"
#include "../../../numerics/operators/varcoeff_laplacian.cuh"
#include "../../../numerics/pin_spec.hpp"
#include "../../../numerics/solvers/mg_preconditioner.cuh"
#include "../../../numerics/solvers/pcg.cuh"
#include "../../../runtime/cuda_check.cuh"
#include "../../flow/rhs_head.cuh"
#include "../../flow/solve_head.cuh"
#include "../../flow/velocity_from_head.cuh"
#include "../../stochastic/PeriodicGaussianField.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cufft.h>
#include <exception>
#include <stdexcept>
#include <thread>

namespace macroflow3d {
namespace lagrangian_labels {

namespace {

using Clock = std::chrono::steady_clock;

double since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

std::size_t device_used_now() {
    std::size_t free_b = 0, total_b = 0;
    MACROFLOW3D_CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    return total_b - free_b;
}

std::vector<real> download(const real* d, std::size_t n) {
    std::vector<real> h(n);
    if (n > 0)
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(real), cudaMemcpyDeviceToHost));
    return h;
}

/// C-SYNC: legacy-stream upload followed by a device-wide synchronization.
template <class T> void upload_sync(T* dst, const T* src, std::size_t n) {
    if (n == 0)
        return;
    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(T), cudaMemcpyHostToDevice));
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
}

template <class F> void in_phase(const char* phase, F&& f) {
    try {
        f();
    } catch (const std::exception& e) {
        std::string m = e.what();
        if (m.find("phase=") == std::string::npos)
            m += std::string(" (phase=") + phase + ")";
        throw std::runtime_error("DirichletLongBoxStage: " + m);
    }
}

void release(std::vector<real>& v) {
    std::vector<real>().swap(v);
}

std::string fmt(const char* f, ...) __attribute__((format(printf, 1, 2)));
std::string fmt(const char* f, ...) {
    char buf[4096];
    va_list ap;
    va_start(ap, f);
    std::vsnprintf(buf, sizeof(buf), f, ap);
    va_end(ap);
    return std::string(buf);
}

// RAII pinned mapped host buffer (the device is synchronized first because queued kernels may still be
// touching the mapped memory when an exception unwinds).
class PinnedMappedHostBuffer {
  public:
    PinnedMappedHostBuffer() = default;
    PinnedMappedHostBuffer(const PinnedMappedHostBuffer&) = delete;
    PinnedMappedHostBuffer& operator=(const PinnedMappedHostBuffer&) = delete;
    ~PinnedMappedHostBuffer() { reset(); }
    real* allocate(std::size_t n) {
        MACROFLOW3D_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&host_), n * sizeof(real),
                                             cudaHostAllocMapped));
        void* dev = nullptr;
        MACROFLOW3D_CUDA_CHECK(cudaHostGetDevicePointer(&dev, host_, 0));
        return static_cast<real*>(dev);
    }
    const real* host() const { return host_; }
    void reset() {
        if (host_ != nullptr) {
            cudaDeviceSynchronize();
            cudaFreeHost(host_);
            host_ = nullptr;
        }
    }

  private:
    real* host_{nullptr};
};

// ---- diagnostics kernels: copied verbatim from DirichletMedium.cu (file-local there) ----------------------
constexpr int kDivBlock = 256;
constexpr int kDivMaxBlocks = 1024;

/// Gathers x-face planes i = 0 and i = nx of U (face index m = j + ny k) into out[0..nf) and out[nf..2nf).
__global__ void kernel_gather_x_planes(const real* __restrict__ U, real* __restrict__ out, int nx,
                                       std::size_t nf) {
    for (std::size_t m = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; m < nf;
         m += (std::size_t)gridDim.x * blockDim.x) {
        out[m] = U[(nx + 1) * m];
        out[nf + m] = U[nx + (nx + 1) * m];
    }
}

/// Per-block maximum of |sum of signed face differences| over the cells.
__global__ void kernel_div_block_max(const real* __restrict__ U, const real* __restrict__ V,
                                     const real* __restrict__ W, real* __restrict__ block_max,
                                     int nx, int ny, int nz) {
    __shared__ real sm[kDivBlock];
    const std::size_t n = (std::size_t)nx * ny * nz;
    real m = 0.0;
    for (std::size_t c = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; c < n;
         c += (std::size_t)gridDim.x * blockDim.x) {
        const int i = (int)(c % nx);
        const std::size_t jk = c / nx;
        const int j = (int)(jk % ny);
        const int k = (int)(jk / ny);
        const std::size_t u0 = i + (std::size_t)(nx + 1) * jk;
        const std::size_t v0 = i + (std::size_t)nx * (j + (std::size_t)(ny + 1) * k);
        const real d = (U[u0 + 1] - U[u0]) + (V[v0 + nx] - V[v0]) + (W[c + (std::size_t)nx * ny] - W[c]);
        m = fmax(m, fabs(d));
    }
    sm[threadIdx.x] = m;
    __syncthreads();
    for (int s = kDivBlock / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            sm[threadIdx.x] = fmax(sm[threadIdx.x], sm[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0)
        block_max[blockIdx.x] = sm[0];
}

/// In place: H -> f = ((-H) - G1 x) + G1 L1 with x = (i + 1/2) h; explicit round-to-nearest operations (no
/// FMA contraction), reproducible on the host.
__global__ void kernel_head_to_f(real* __restrict__ a, int nx, std::size_t n, real h, real G1, real head_drop) {
    for (std::size_t c = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; c < n;
         c += (std::size_t)gridDim.x * blockDim.x) {
        const int i = (int)(c % nx);
        const real x = __dmul_rn((real)i + 0.5, h);
        a[c] = __dadd_rn(__dsub_rn(-a[c], __dmul_rn(G1, x)), head_drop);
    }
}

/// S3c: {pot, logk} -> double2 (pure data movement).
__global__ void kernel_interleave(const real* __restrict__ pot, const real* __restrict__ logk,
                                  double2* __restrict__ out, std::size_t cnt) {
    for (std::size_t c = blockIdx.x * (std::size_t)blockDim.x + threadIdx.x; c < cnt;
         c += (std::size_t)gridDim.x * blockDim.x)
        out[c] = make_double2(pot[c], logk[c]);
}

std::string fmt_items(const std::vector<std::pair<std::string, std::size_t>>& items) {
    if (items.empty())
        return "-";
    std::string s;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i > 0)
            s += ",";
        s += items[i].first + ":" + std::to_string(items[i].second);
    }
    return s;
}

std::string phase_line(const LongBoxPhaseRecord& r) {
    std::string s = "MEM_PHASE phase=" + r.phase + " exact_bytes=" + std::to_string(r.exact_bytes()) +
                    " cufft_work=" + std::to_string(r.cufft_work);
    for (const auto& w : r.cufft_named)
        s += " " + w.first + "=" + std::to_string(w.second);
    s += " device_used=" + std::to_string(r.device_used);
    s += fmt(" seconds=%.6f", r.seconds);
    if (!r.categories.empty())
        s += " categories=" + r.categories;
    s += " items=" + fmt_items(r.items);
    return s;
}

} // namespace

// ---- solver record -----------------------------------------------------------------------------------------

DirichletSolverRecord make_dirichlet_solver_record(bool converged, int iterations, real initial, real final_res,
                                                   real qbar, real Q_in, real Q_out, real balance,
                                                   real max_div) {
    DirichletSolverRecord s;
    s.converged = converged;
    s.iterations = iterations;
    s.initial = initial;
    s.final = final_res;
    s.rel = initial > 0.0 ? final_res / initial : 0.0; // DirichletMedium.cu convention
    s.qbar = qbar;
    s.Q_in = Q_in;
    s.Q_out = Q_out;
    s.balance = balance;
    s.max_div = max_div;
    // the A/B application's rule (lagrangian_medium_ab_main.cu, "C0 / D8 record and the fallback rule")
    const bool finite_ok = std::isfinite(s.rel) && std::isfinite(s.balance) && std::isfinite(s.max_div) &&
                           std::isfinite(s.qbar) && std::isfinite(s.Q_in) && std::isfinite(s.Q_out);
    constexpr double kFallbackTol = 1e-8;
    if (!finite_ok) {
        s.verdict = "non_finite";
    } else if (s.converged) {
        s.usable = s.balance <= kFallbackTol;
        s.verdict = s.usable ? "converged" : "converged_balance_above_1e-8";
    } else if (s.rel <= kFallbackTol && s.balance <= kFallbackTol) {
        s.usable = s.fallback = true;
        s.verdict = "fallback";
    } else {
        s.verdict = "not_converged_beyond_fallback";
    }
    return s;
}

DirichletSolverRecord make_dirichlet_solver_record(const DirichletMediumReport& r) {
    return make_dirichlet_solver_record(r.converged, r.iterations, r.initial_residual, r.final_residual, r.qbar,
                                        r.Q_in, r.Q_out, r.balance, r.max_div);
}

std::string format_b_solver(int r, const DirichletSolverRecord& B) {
    return fmt("B_SOLVER r=%d converged=%d iterations=%d initial=%.6e final=%.6e rel=%.6e qbar=%.12e Q_in=%.12e "
               "Q_out=%.12e balance=%.6e max_div=%.6e fallback=%d",
               r, B.converged ? 1 : 0, B.iterations, B.initial, B.final, B.rel, B.qbar, B.Q_in, B.Q_out,
               B.balance, B.max_div, B.fallback ? 1 : 0);
}

std::size_t dirichlet_s2_closed_form_bytes(int nx, int ny, int nz, int mg_levels, bool head_on_device,
                                           std::size_t scratch_bytes) {
    const std::size_t n = (std::size_t)nx * ny * nz;
    std::size_t coarse = 0;
    for (int l = 1; l < mg_levels; ++l)
        coarse += (std::size_t)(nx >> l) * (ny >> l) * (nz >> l);
    return sizeof(real) * ((head_on_device ? 7 : 6) * n + 4 * coarse) + scratch_bytes;
}

std::size_t dirichlet_s3_closed_form_bytes(int nx, int ny, int nz, std::size_t cufft_work, bool fused,
                                           std::size_t sub[3]) {
    const std::size_t n = (std::size_t)nx * ny * nz;
    const std::size_t ne = (std::size_t)(nx + 4) * ny * nz;
    const std::size_t sp = (std::size_t)(nx + 4) * (ny / 2 + 1) * nz;
    const std::size_t table = (std::size_t)(nx - 4);
    const std::size_t a = 8 * n + 8 * ne + 16 * sp + 8 * table + cufft_work;
    const std::size_t b = 8 * ne + 8 * n + 8 * ne + 16 * sp + 8 * table + cufft_work;
    const std::size_t c = 32 * ne;
    if (sub != nullptr) {
        sub[0] = a;
        sub[1] = b;
        sub[2] = c;
    }
    return std::max(std::max(a, b), fused ? c : std::size_t(0));
}

namespace {
struct CufftHandleGuard {
    cufftHandle h = 0;
    CufftHandleGuard() {
        if (cufftCreate(&h) != CUFFT_SUCCESS)
            throw std::runtime_error("memory model: cufftCreate failed");
    }
    ~CufftHandleGuard() { cufftDestroy(h); }
};
} // namespace

DirichletStageMemoryModel dirichlet_long_box_stage_memory_model(int nx, int ny, int nz, int mg_levels,
                                                                bool head_on_device) {
    if (nx < 8 || ny < 4 || nz < 4 || ny != nz)
        throw std::invalid_argument("memory model: nx >= 8, ny == nz >= 4 required");
    DirichletStageMemoryModel M;
    M.nx = nx;
    M.ny = ny;
    M.nz = nz;
    M.mg_levels = mg_levels == 0 ? box_auto_mg_levels(nx, ny, nz) : mg_levels;
    M.head_on_device = head_on_device;
    const std::size_t n = (std::size_t)nx * ny * nz;
    const std::size_t nyz = (std::size_t)ny * nz;
    const std::size_t R = sizeof(real);
    auto blocks1d = [](std::size_t cnt, std::size_t cap) { return std::min<std::size_t>(cap, (cnt + 255) / 256); };

    { // F: y, SF-18 half spectrum, mode counter, reduce blocks, 3-D Z2D work
        CufftHandleGuard g;
        std::size_t w = 0;
        if (cufftGetSize3d(g.h, nz, ny, nx, CUFFT_Z2D, &w) != CUFFT_SUCCESS)
            throw std::runtime_error("memory model: cufftGetSize3d failed");
        M.cufft_f = w;
        M.F = R * n + sizeof(cufftDoubleComplex) * (std::size_t)(nx / 2 + 1) * ny * nz +
              sizeof(unsigned long long) + R * blocks1d(n, 256) + w;
    }
    M.S1 = 0; // host only
    M.S2 = dirichlet_s2_closed_form_bytes(nx, ny, nz, M.mg_levels, head_on_device, R); // scratch = d_scalar only (no CUB: pcg_solve_external uses cuBLAS)
    // V: K, head, U, V, W, planes, div_block_max
    M.V = R * (2 * n + (std::size_t)(nx + 1) * nyz + (std::size_t)nx * (ny + 1) * nz +
               (std::size_t)nx * ny * (nz + 1) + 2 * nyz + blocks1d(n, 1024));
    M.H = R * n;
    { // S3: W = D2Z + Z2D of the two batched rank-2 plans of N1 (nxe batches, stride nxe, distance 1)
        const int nxe = nx + 4, hy = ny / 2 + 1;
        int dims[2] = {nz, ny}, real_embed[2] = {nz, ny}, cplx_embed[2] = {nz, hy};
        CufftHandleGuard g1, g2;
        std::size_t w1 = 0, w2 = 0;
        if (cufftGetSizeMany(g1.h, 2, dims, real_embed, nxe, 1, cplx_embed, nxe, 1, CUFFT_D2Z, nxe, &w1) !=
                CUFFT_SUCCESS ||
            cufftGetSizeMany(g2.h, 2, dims, cplx_embed, nxe, 1, real_embed, nxe, 1, CUFFT_Z2D, nxe, &w2) !=
                CUFFT_SUCCESS)
            throw std::runtime_error("memory model: cufftGetSizeMany failed");
        M.cufft_prefilter = w1 + w2;
        std::size_t sub[3];
        dirichlet_s3_closed_form_bytes(nx, ny, nz, M.cufft_prefilter, true, sub);
        M.S3a = sub[0];
        M.S3b = sub[1];
        M.S3c = sub[2];
    }
    return M;
}

std::vector<std::string> DirichletLongBoxStage::mem_lines() const {
    std::vector<std::string> out;
    for (const LongBoxPhaseRecord* r : {&phase_F, &phase_S1, &phase_S2, &phase_V, &phase_H, &phase_S3}) {
        if (r->present)
            out.push_back(phase_line(*r));
    }
    return out;
}

std::vector<std::string> DirichletLongBoxStage::mem_lines(const LagrangianTracker* tracker,
                                                          double tracker_prepare_seconds) const {
    std::vector<std::string> out = mem_lines();
    if (tracker != nullptr && tracker->prepared() && ready()) {
        LongBoxPhaseRecord t;
        t.phase = "T";
        t.present = true;
        if (fused_coefficients) {
            t.items.emplace_back("coef_interleaved", coef_interleaved.capacity() * sizeof(double2));
        } else {
            t.items.emplace_back("coef_pot", coef_pot.capacity() * sizeof(real));
            t.items.emplace_back("coef_logk", coef_logk.capacity() * sizeof(real));
        }
        for (const auto& it : tracker->device_buffer_bytes())
            t.items.emplace_back("tracker_" + it.first, it.second);
        t.device_used = device_used_now();
        t.seconds = tracker_prepare_seconds;
        out.push_back(phase_line(t));
    }
    return out;
}

// ---- the stage ---------------------------------------------------------------------------------------------

DirichletLongBoxStage build_dirichlet_long_box_stage(CudaContext& ctx, const DirichletLongBoxSpec& spec) {
    const int nx = spec.nx, ny = spec.ny, nz = spec.nz;
    if (nx < 4 || ny < 4 || nz < 4 || nx % 2 != 0 || ny % 2 != 0 || nz % 2 != 0)
        throw std::invalid_argument("build_dirichlet_long_box_stage: nx, ny, nz must be even and >= 4");
    if (ny != nz)
        throw std::invalid_argument("build_dirichlet_long_box_stage: ny must equal nz (h = 1/ny)");
    if (spec.build_splines && nx < 8)
        throw std::invalid_argument("build_dirichlet_long_box_stage: build_splines needs nx >= 8 (C-SPLINE)");
    if (!std::isfinite(spec.eps))
        throw std::invalid_argument("build_dirichlet_long_box_stage: eps must be finite");
    if (!(std::isfinite(spec.G1) && spec.G1 > 0.0))
        throw std::invalid_argument("build_dirichlet_long_box_stage: G1 must be finite and > 0");
    if (!(spec.cg_rtol > 0.0) || spec.cg_max_iter < 1)
        throw std::invalid_argument("build_dirichlet_long_box_stage: cg_rtol > 0 and cg_max_iter >= 1 required");
    const std::size_t n = static_cast<std::size_t>(nx) * ny * nz;
    if (spec.source == LongBoxSourceKind::cells && spec.cells.size() != n)
        throw std::invalid_argument("build_dirichlet_long_box_stage: cells.size() != nx*ny*nz");
    const int levels = spec.mg_levels == 0 ? box_auto_mg_levels(nx, ny, nz) : spec.mg_levels;
    {
        const int f = levels >= 1 && levels <= 30 ? 1 << (levels - 1) : 0;
        if (f == 0 || nx % f != 0 || ny % f != 0 || nz % f != 0 || nx / f < 2 || ny / f < 2 || nz / f < 2)
            throw std::invalid_argument("build_dirichlet_long_box_stage: nx, ny, nz must be divisible by "
                                        "2^(mg_levels-1) with coarsest extents >= 2");
    }

    DirichletLongBoxStage S;
    S.h = 1.0 / static_cast<real>(ny);
    S.L1 = nx * S.h;
    S.grid = Grid3D(nx, ny, nz, S.h, S.h, S.h);
    S.G1 = spec.G1;
    S.head_drop = spec.G1 * S.L1;
    S.G[0] = spec.G1;
    S.fused_coefficients = spec.fused_coefficients;
    S.mg_levels = levels;
    S.head_mapped_host = spec.head_mapped_host;
    S.device_used_start = device_used_now();
    const Grid3D& grid = S.grid;

    BCSpec bc;
    bc.xmin = BCFace(BCType::Dirichlet, S.head_drop);
    bc.xmax = BCFace(BCType::Dirichlet, 0.0);
    bc.ymin = bc.ymax = BCFace(BCType::Periodic, 0.0);
    bc.zmin = bc.zmax = BCFace(BCType::Periodic, 0.0);

    // ---- F: SF-18 field (gaussian) -> host Y -------------------------------------------------------------
    std::vector<real> Y_gen;
    {
        LongBoxPhaseRecord& rec = S.phase_F;
        rec.phase = "F";
        rec.present = true;
        const auto t0 = Clock::now();
        if (spec.source == LongBoxSourceKind::gaussian) {
            in_phase("F", [&] {
                // exactly LongBoxStage phase F (= SlabFieldSource::gaussian's SF-18 call)
                physics::PeriodicGaussianFieldConfig cfg;
                cfg.sigma2 = spec.sigma2;
                cfg.corr_length = spec.ell;
                cfg.seed = spec.seed;
                cfg.normalize_variance = spec.normalize_variance;
                DeviceBuffer<real> y(n);
                physics::PeriodicGaussianFieldWorkspace ws;
                const auto sf18 = physics::generate_periodic_gaussian_field(ctx, grid, cfg, y.span(), ws);
                S.sf18 = sf18;
                ctx.synchronize();
                Y_gen = download(y.data(), n);
                rec.items.emplace_back("y", y.capacity() * sizeof(real));
                rec.items.emplace_back("sf18_spectrum", ws.spectrum.capacity() * sizeof(cufftDoubleComplex));
                rec.items.emplace_back("sf18_mode_counter",
                                       ws.mode_counter.capacity() * sizeof(unsigned long long));
                rec.items.emplace_back("sf18_reduce_blocks", ws.reduce_blocks.capacity() * sizeof(real));
                rec.cufft_work = sf18.cufft_work_area_bytes;
                rec.cufft_named.emplace_back("cufft_gen", sf18.cufft_work_area_bytes);
                rec.device_used = device_used_now();
            });
        } else {
            rec.device_used = S.device_used_start;
        }
        rec.seconds = since(t0);
    }
    const std::vector<real>& Y = spec.source == LongBoxSourceKind::cells ? spec.cells : Y_gen;

    // ---- S1: host lnK = eps Y (kept), K = exp(lnK) -------------------------------------------------------
    std::vector<real> K;
    {
        LongBoxPhaseRecord& rec = S.phase_S1;
        rec.phase = "S1";
        rec.present = true;
        const auto t0 = Clock::now();
        in_phase("S1", [&] {
            K.resize(n);
            S.host_lnk.resize(n);
            // Static contiguous chunks over host threads. Every element is the same scalar expression as
            // DirichletMedium K phase / LongBoxStage S1 (libm std::exp, no fast-math/SIMD/OpenMP), so the
            // result is bitwise independent of the thread count.
            const std::size_t nt = std::max<std::size_t>(
                1, std::min<std::size_t>({std::thread::hardware_concurrency(), 32, n}));
            real* const lnk = S.host_lnk.data();
            real* const Kd = K.data();
            const real* const Yd = Y.data();
            const real eps = spec.eps;
            auto chunk = [&](std::size_t t) {
                for (std::size_t c = n * t / nt, e = n * (t + 1) / nt; c < e; ++c) {
                    const real lnK = eps * Yd[c];
                    lnk[c] = lnK;
                    Kd[c] = std::exp(lnK);
                }
            };
            std::vector<std::thread> pool;
            for (std::size_t t = 1; t < nt; ++t)
                pool.emplace_back(chunk, t);
            chunk(0);
            for (auto& th : pool)
                th.join();
            rec.device_used = device_used_now();
        });
        release(Y_gen);
        rec.seconds = since(t0);
    }

    // ---- S2: lean solve ----------------------------------------------------------------------------------
    physics::KField Kf;       // the MG level-0 coefficient buffer, moved out after the solve
    physics::HeadField head;  // device head (unused until V in mapped mode)
    PinnedMappedHostBuffer head_host;
    {
        LongBoxPhaseRecord& rec = S.phase_S2;
        rec.phase = "S2";
        rec.present = true;
        const auto t0 = Clock::now();
        in_phase("S2", [&] {
            physics::HeadSolveConfig hc; // defaults of solve_head's MG smoothing and CG check cadence
            multigrid::MGHierarchy hier(grid, levels);
            if (hier.num_levels() != levels)
                throw std::invalid_argument("MG hierarchy has fewer levels than requested");
            DeviceBuffer<real> pcg_r, pcg_p;
            blas::ReductionWorkspace red;
            real* head_ptr = nullptr;
            if (spec.head_mapped_host) {
                head_ptr = head_host.allocate(n);
            } else {
                head.resize(grid);
                head_ptr = head.data.data();
            }
            pcg_r.resize(n);
            pcg_p.resize(n);
            auto& lvl0 = hier.levels.front();

            // K straight into the level-0 coefficient, then the coarsening loop of populate_coefficient_hierarchy
            upload_sync(lvl0.coefficient.data(), K.data(), n);
            release(K);
            for (int l = 0; l + 1 < hier.num_levels(); ++l) {
                auto& fine = hier.levels[l];
                auto& coarse = hier.levels[l + 1];
                multigrid::coarsen_coefficient_geometric(
                    ctx, coarse.grid, fine.grid,
                    DeviceSpan<const real>(fine.coefficient.data(), fine.coefficient.size()),
                    DeviceSpan<real>(coarse.coefficient.data(), coarse.coefficient.size()));
            }

            auto* trace = spec.debug_trace;
            if (trace) {
                for (std::size_t l = 0; l < hier.levels.size(); ++l)
                    trace->emplace_back("coef_l" + std::to_string(l), debug::fnv1a_device(hier.levels[l].coefficient.data(),
                                                                                       hier.levels[l].coefficient.size()));
            }
            // right-hand side into the dead level-0 residual buffer
            physics::build_rhs_head(DeviceSpan<real>(lvl0.r.data(), n),
                                    DeviceSpan<const real>(lvl0.coefficient.data(), n), grid, bc, ctx);
            const DeviceSpan<real> x(head_ptr, n);
            physics::init_head_guess(x, grid, bc, ctx);
            if (trace) {
                trace->emplace_back("rhs", debug::fnv1a_device(lvl0.r.data(), n));
                trace->emplace_back("head_init", debug::fnv1a_device(head_ptr, n));
            }

            const PinSpec pin_spec(pin_enabled(PinMode::Off, bc));
            operators::VarCoeffLaplacian A(grid, DeviceSpan<const real>(lvl0.coefficient.data(), n), bc, pin_spec);
            multigrid::MGConfig mg_cfg;
            mg_cfg.num_levels = hier.num_levels();
            mg_cfg.pre_smooth = hc.mg_pre_smooth;
            mg_cfg.post_smooth = hc.mg_post_smooth;
            mg_cfg.coarse_solve_iters = hc.mg_coarse_iters;
            mg_cfg.verbose = false;
            solvers::MultigridPreconditioner M(hier, bc, mg_cfg, pin_spec);

            solvers::PCGConfig pcg_cfg;
            pcg_cfg.max_iter = spec.cg_max_iter;
            pcg_cfg.check_every = hc.cg_check_every;
            pcg_cfg.rtol = spec.cg_rtol;
            pcg_cfg.verbose = false;
            solvers::PCGExternalVectors vec;
            vec.r = DeviceSpan<real>(pcg_r.data(), n);
            vec.p = DeviceSpan<real>(pcg_p.data(), n);
            vec.z = DeviceSpan<real>(lvl0.b.data(), n);
            vec.Ap = DeviceSpan<real>(lvl0.b.data(), n); // z == Ap == level-0 b (disjoint lifetimes)
            vec.red = &red;
            solvers::PCGResult r;
            if (trace) {
                const debug::TracedOp<operators::VarCoeffLaplacian> tA{A, trace, "A#", 0};
                const debug::TracedOp<solvers::MultigridPreconditioner> tM{M, trace, "M#", 0};
                r = solvers::pcg_solve_external(ctx, tA, tM, DeviceSpan<const real>(lvl0.r.data(), n), x, pcg_cfg, vec);
                trace->emplace_back("head_final", debug::fnv1a_device(head_ptr, n));
            } else {
                r = solvers::pcg_solve_external(ctx, A, M, DeviceSpan<const real>(lvl0.r.data(), n), x, pcg_cfg, vec);
            }
            ctx.synchronize();
            MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());

            // provisional solver record; the flow diagnostics are filled by V
            S.solver.converged = r.converged;
            S.solver.iterations = r.iterations;
            S.solver.initial = r.initial_residual;
            S.solver.final = r.final_residual;

            for (std::size_t l = 0; l < hier.levels.size(); ++l) {
                const auto& L = hier.levels[l];
                const std::string p = "mg_l" + std::to_string(l) + "_";
                rec.items.emplace_back(p + "x", L.x.capacity() * sizeof(real));
                rec.items.emplace_back(p + "b", L.b.capacity() * sizeof(real));
                rec.items.emplace_back(p + "r", L.r.capacity() * sizeof(real));
                rec.items.emplace_back(p + "coef", L.coefficient.capacity() * sizeof(real));
            }
            if (!spec.head_mapped_host)
                rec.items.emplace_back("head", head.data.capacity() * sizeof(real));
            rec.items.emplace_back("pcg_r", pcg_r.capacity() * sizeof(real));
            rec.items.emplace_back("pcg_p", pcg_p.capacity() * sizeof(real));
            rec.items.emplace_back("scratch_pcg_d_scalar", red.d_scalar.capacity() * sizeof(real));
            rec.items.emplace_back("scratch_pcg_red_temp", red.temp.capacity());
            rec.device_used = device_used_now(); // everything alive: the peak of the solve
            if (spec.head_mapped_host)
                S.phase_S2.categories = "head:mapped_host";

            // keep K (the level-0 coefficient) for V; MG, PCG buffers are freed at the end of this scope
            Kf.nx = nx;
            Kf.ny = ny;
            Kf.nz = nz;
            Kf.dx = Kf.dy = Kf.dz = S.h;
            Kf.data = std::move(lvl0.coefficient);
        });
        MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
        rec.seconds = since(t0);
    }

    // ---- V: velocity faces and the DirichletMedium diagnostics ------------------------------------------
    {
        LongBoxPhaseRecord& rec = S.phase_V;
        rec.phase = "V";
        rec.present = true;
        const auto t0 = Clock::now();
        in_phase("V", [&] {
            if (spec.head_mapped_host) {
                head.resize(grid);
                upload_sync(head.data.data(), head_host.host(), n);
                if (spec.keep_host_head)
                    S.host_head.assign(head_host.host(), head_host.host() + n);
                head_host.reset();
            } else if (spec.keep_host_head) {
                S.host_head = download(head.data.data(), n);
            }
            physics::VelocityField vel;
            vel.resize(grid);
            physics::compute_velocity_from_head(vel, head, Kf, grid, bc, ctx);
            ctx.synchronize();
            MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());

            const std::size_t nf = (std::size_t)ny * nz;
            DeviceBuffer<real> planes(2 * nf);
            const int nblk = (int)std::min<std::size_t>(kDivMaxBlocks, (n + kDivBlock - 1) / kDivBlock);
            DeviceBuffer<real> bmax((std::size_t)nblk);
            kernel_gather_x_planes<<<(int)std::min<std::size_t>(kDivMaxBlocks, (nf + 255) / 256), 256, 0,
                                     ctx.cuda_stream()>>>(vel.U.data(), planes.data(), nx, nf);
            MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
            kernel_div_block_max<<<nblk, kDivBlock, 0, ctx.cuda_stream()>>>(
                vel.U.data(), vel.V.data(), vel.W.data(), bmax.data(), nx, ny, nz);
            MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
            ctx.synchronize();
            const std::vector<real> hp = download(planes.data(), 2 * nf);
            const std::vector<real> hb = download(bmax.data(), (std::size_t)nblk);
            real s_in = 0.0, s_out = 0.0, dmax = 0.0;
            for (std::size_t m = 0; m < nf; ++m) {
                s_in += hp[m];
                s_out += hp[nf + m];
            }
            for (real v : hb)
                dmax = std::max(dmax, v);
            const real qbar = s_in / static_cast<real>(nf);
            const real Q_in = S.h * S.h * s_in;
            const real Q_out = S.h * S.h * s_out;
            const real balance = std::abs(Q_in - Q_out) / std::abs(Q_in);
            const real max_div = dmax / qbar;
            S.solver = make_dirichlet_solver_record(S.solver.converged, S.solver.iterations, S.solver.initial,
                                                    S.solver.final, qbar, Q_in, Q_out, balance, max_div);
            S.inlet_u_plane.assign(hp.begin(), hp.begin() + nf);

            auto cap = [](const DeviceBuffer<real>& b) { return b.capacity() * sizeof(real); };
            rec.items = {{"K", cap(Kf.data)},  {"head", cap(head.data)}, {"U", cap(vel.U)},
                         {"V", cap(vel.V)},    {"W", cap(vel.W)},        {"planes", cap(planes)},
                         {"div_block_max", cap(bmax)}};
            rec.device_used = device_used_now();
            if (spec.keep_host_faces) {
                S.face_u = download(vel.U.data(), vel.U.size());
                S.face_v = download(vel.V.data(), vel.V.size());
                S.face_w = download(vel.W.data(), vel.W.size());
            }
        }); // faces freed here
        Kf.data.reset(); // K is dead
        MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
        rec.seconds = since(t0);
    }

    // ---- H: head -> f in place -------------------------------------------------------------------------
    {
        LongBoxPhaseRecord& rec = S.phase_H;
        rec.phase = "H";
        rec.present = true;
        const auto t0 = Clock::now();
        in_phase("H", [&] {
            const int blocks = (int)std::min<std::size_t>(65535, (n + 255) / 256);
            kernel_head_to_f<<<blocks, 256, 0, ctx.cuda_stream()>>>(head.data.data(), nx, n, S.h, S.G1,
                                                                    S.head_drop);
            MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
            ctx.synchronize();
            MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
            rec.items.emplace_back("f", head.data.capacity() * sizeof(real));
            rec.device_used = device_used_now();
            S.f = std::move(head.data);
        });
        rec.seconds = since(t0);
    }

    // ---- S3: non-periodic spline coefficients of f and ln k ----------------------------------------------
    if (spec.build_splines) {
        LongBoxPhaseRecord& rec = S.phase_S3;
        rec.phase = "S3";
        rec.present = true;
        const auto t0 = Clock::now();
        in_phase("S3", [&] {
            using Items = std::vector<std::pair<std::string, std::size_t>>;
            const std::size_t ne = S.coefficient_count();
            auto capR = [](const DeviceBuffer<real>& b) { return b.capacity() * sizeof(real); };
            std::size_t sub_bytes[3] = {0, 0, 0}, used_peak = 0, work_peak = 0;
            Items peak_items;
            int peak_idx = -1;
            auto consider = [&](int idx, const Items& it, std::size_t w) {
                std::size_t b = w;
                for (const auto& x : it)
                    b += x.second;
                sub_bytes[idx] = b;
                if (peak_idx < 0 || b > sub_bytes[peak_idx]) {
                    peak_idx = idx;
                    peak_items = it;
                    work_peak = w;
                }
                used_peak = std::max(used_peak, device_used_now());
            };
            auto prefilter_items = [&](Items head, const interpolation::NonPeriodicXTricubicBSplineReport& r,
                                       const char* coef_name) {
                head.emplace_back(coef_name, r.coefficient_bytes);
                head.emplace_back("np_spectrum", r.spectrum_bytes);
                head.emplace_back("np_line_table", r.line_table_bytes);
                return head;
            };
            DeviceBuffer<real> pot, logk;
            {
                interpolation::NonPeriodicXTricubicBSplineWorkspace ws;
                // S3a: f -> coef_pot (coefficient buffer moved out, f freed)
                auto r = interpolation::prefilter_nonperiodic_x_tricubic_bspline(
                    ctx, grid, DeviceSpan<const real>(S.f.data(), n), ws);
                consider(0, prefilter_items({{"f", capR(S.f)}}, r, "coef_pot"), r.cufft_work_area_bytes);
                pot = std::move(ws.coefficients);
                S.f.reset();
                // S3b: ln k samples -> coef_logk in the same workspace (spectrum and pivot table reused)
                DeviceBuffer<real> d_lnk(n);
                upload_sync(d_lnk.data(), S.host_lnk.data(), n);
                r = interpolation::prefilter_nonperiodic_x_tricubic_bspline(
                    ctx, grid, DeviceSpan<const real>(d_lnk.data(), n), ws);
                consider(1, prefilter_items({{"coef_pot", capR(pot)}, {"lnk_samples", capR(d_lnk)}}, r, "coef_logk"),
                         r.cufft_work_area_bytes);
                logk = std::move(ws.coefficients);
            } // lnk samples and the workspace (spectrum, table) freed
            release(S.host_lnk);
            if (pot.size() != ne || logk.size() != ne)
                throw std::runtime_error("unexpected coefficient extent");
            if (spec.fused_coefficients) { // S3c
                DeviceBuffer<double2> inter(ne);
                const int blk = (int)std::min<std::size_t>(65535, (ne + 255) / 256);
                kernel_interleave<<<blk, 256, 0, ctx.cuda_stream()>>>(pot.data(), logk.data(), inter.data(), ne);
                MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
                ctx.synchronize();
                MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
                consider(2, {{"coef_pot", capR(pot)}, {"coef_logk", capR(logk)},
                             {"coef_interleaved", inter.capacity() * sizeof(double2)}}, 0);
                S.coef_interleaved = std::move(inter);
            } else {
                S.coef_pot = std::move(pot);
                S.coef_logk = std::move(logk);
            }
            S.coefficients_nonperiodic_x = true;
            rec.items = peak_items;
            rec.cufft_work = work_peak;
            rec.cufft_named.emplace_back("cufft_prefilter", work_peak);
            rec.categories = fmt("S3a:%zu,S3b:%zu,S3c:%zu,peak:S3%c", sub_bytes[0], sub_bytes[1], sub_bytes[2],
                                 "abc"[peak_idx]);
            rec.device_used = used_peak;
        });
        MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
        rec.seconds = since(t0);
    }
    return S;
}

} // namespace lagrangian_labels
} // namespace macroflow3d
