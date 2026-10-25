/**
 * @file DirichletLongBoxStage.cu
 * @brief SF-41 N2 lean medium (b) stage, flow part. See DirichletLongBoxStage.cuh.
 *
 * Compiled with the default flags like DirichletMedium.cu (whose file-local diagnostics kernels are copied
 * below verbatim, so they compile identically) and LongBoxStage.cu.
 */

#include "DirichletLongBoxStage.cuh"

#include "../../../multigrid/coefficient_hierarchy.cuh"
#include "../../../multigrid/mg_types.hpp"
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
#include <exception>
#include <stdexcept>

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

std::vector<std::string> DirichletLongBoxStage::mem_lines() const {
    std::vector<std::string> out;
    for (const LongBoxPhaseRecord* r : {&phase_F, &phase_S1, &phase_S2, &phase_V, &phase_H}) {
        if (r->present)
            out.push_back(phase_line(*r));
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
            for (std::size_t c = 0; c < n; ++c) { // DirichletMedium K phase / LongBoxStage S1 expression
                const real lnK = spec.eps * Y[c];
                S.host_lnk[c] = lnK;
                K[c] = std::exp(lnK);
            }
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

            // right-hand side into the dead level-0 residual buffer
            physics::build_rhs_head(DeviceSpan<real>(lvl0.r.data(), n),
                                    DeviceSpan<const real>(lvl0.coefficient.data(), n), grid, bc, ctx);
            const DeviceSpan<real> x(head_ptr, n);
            physics::init_head_guess(x, grid, bc, ctx);

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
            const solvers::PCGResult r =
                solvers::pcg_solve_external(ctx, A, M, DeviceSpan<const real>(lvl0.r.data(), n), x, pcg_cfg, vec);
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
    // SEAM(N4): host_lnk and f are the inputs of the non-periodic prefilters (phase S3 of N4).
    return S;
}

} // namespace lagrangian_labels
} // namespace macroflow3d
