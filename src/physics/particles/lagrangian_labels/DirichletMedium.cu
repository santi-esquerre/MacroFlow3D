/**
 * @file DirichletMedium.cu
 * @brief SF-38 N2 medium (b). See DirichletMedium.cuh.
 */

#include "DirichletMedium.cuh"

#include "../../../runtime/cuda_check.cuh"
#include "../../flow/solve_head.cuh"
#include "../../flow/velocity_from_head.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
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

std::vector<real> download(const real* d, std::size_t n) {
    std::vector<real> h(n);
    if (n > 0)
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(real), cudaMemcpyDeviceToHost));
    return h;
}

} // namespace

physics::particles::streamline_tracker::PeriodicFaceFluxView DirichletMedium::periodic_view() const {
    if (!periodic_available())
        throw std::logic_error("DirichletMedium::periodic_view: build_periodic_faces was not requested");
    physics::particles::streamline_tracker::PeriodicFaceFluxView v;
    v.u = pu_.data();
    v.v = pv_.data();
    v.w = pw_.data();
    v.nx = grid.nx;
    v.ny = grid.ny;
    v.nz = grid.nz;
    v.dx = v.dy = v.dz = h;
    v.Lx = grid.nx * h;
    v.Ly = grid.ny * h;
    v.Lz = grid.nz * h;
    return v;
}

std::vector<std::string> DirichletMedium::mem_lines() const {
    std::vector<std::string> out;
    for (const auto& p : phases) {
        std::string s = "MEM_PHASE_B phase=" + p.phase + " exact_bytes=" + std::to_string(p.exact_bytes()) +
                        " device_used_start=" + std::to_string(p.device_used_start) +
                        " device_used=" + std::to_string(p.device_used);
        char b[64];
        std::snprintf(b, sizeof(b), " seconds=%.6f", p.seconds);
        s += b;
        s += " items=";
        if (p.items.empty())
            s += "-";
        for (std::size_t i = 0; i < p.items.size(); ++i)
            s += (i ? "," : "") + p.items[i].first + ":" + std::to_string(p.items[i].second);
        out.push_back(s);
    }
    return out;
}

DirichletMedium build_dirichlet_medium(CudaContext& ctx, const DirichletMediumSpec& spec,
                                       const std::vector<real>& Y) {
    const int nx = spec.nx, ny = spec.ny, nz = spec.nz;
    if (nx < 2 || ny < 2 || nz < 2)
        throw std::invalid_argument("build_dirichlet_medium: nx, ny, nz must be >= 2");
    if (ny != nz)
        throw std::invalid_argument("build_dirichlet_medium: ny must equal nz (h = 1/ny)");
    if (!std::isfinite(spec.eps))
        throw std::invalid_argument("build_dirichlet_medium: eps must be finite");
    if (!(std::isfinite(spec.G1) && spec.G1 > 0.0))
        throw std::invalid_argument("build_dirichlet_medium: G1 must be finite and > 0");
    if (!(spec.cg_rtol > 0.0) || spec.cg_max_iter < 1)
        throw std::invalid_argument("build_dirichlet_medium: cg_rtol > 0 and cg_max_iter >= 1 required");
    if (spec.mg_levels < 1)
        throw std::invalid_argument("build_dirichlet_medium: mg_levels must be given explicitly (>= 1)");
    {
        const int f = 1 << (spec.mg_levels - 1);
        if (spec.mg_levels > 30 || nx % f != 0 || ny % f != 0 || nz % f != 0 || nx / f < 2 ||
            ny / f < 2 || nz / f < 2)
            throw std::invalid_argument("build_dirichlet_medium: nx, ny, nz must be divisible by "
                                        "2^(mg_levels-1) with coarsest extents >= 2");
    }
    const std::size_t n = (std::size_t)nx * ny * nz;
    if (Y.size() != n)
        throw std::invalid_argument("build_dirichlet_medium: Y.size() != nx*ny*nz");

    DirichletMedium M;
    M.h = 1.0 / static_cast<real>(ny);
    M.L1 = nx * M.h;
    M.grid = Grid3D(nx, ny, nz, M.h, M.h, M.h);
    const Grid3D& grid = M.grid;
    const real head_drop = spec.G1 * M.L1;
    M.report.head_drop = head_drop;

    BCSpec bc;
    bc.xmin = BCFace(BCType::Dirichlet, head_drop);
    bc.xmax = BCFace(BCType::Dirichlet, 0.0);
    bc.ymin = bc.ymax = BCFace(BCType::Periodic, 0.0);
    bc.zmin = bc.zmax = BCFace(BCType::Periodic, 0.0);

    auto cap = [](const DeviceBuffer<real>& b) { return b.capacity() * sizeof(real); };

    physics::KField Kf;
    physics::HeadField head;

    // ---- K: host K = exp(eps Y) (LongBoxStage S1 expression), upload ---------------------------
    {
        DirichletPhaseRecord rec;
        rec.phase = "K";
        rec.device_used_start = device_used_now();
        const auto t0 = Clock::now();
        Kf.resize(grid);
        head.resize(grid);
        std::vector<real> K(n);
        for (std::size_t c = 0; c < n; ++c) {
            const real lnK = spec.eps * Y[c];
            K[c] = std::exp(lnK);
        }
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(Kf.data.data(), K.data(), n * sizeof(real), cudaMemcpyHostToDevice));
        MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
        rec.items = {{"K", cap(Kf.data)}, {"head", cap(head.data)}};
        rec.device_used = device_used_now();
        rec.seconds = since(t0);
        M.phases.push_back(rec);
    }

    // ---- solve + release ----------------------------------------------------------------------
    {
        DirichletPhaseRecord rec;
        rec.phase = "solve";
        rec.device_used_start = device_used_now();
        const auto t0 = Clock::now();
        std::size_t ws_bytes = 0;
        {
            physics::FlowWorkspace ws;
            ws.allocate(grid); // rhs; MG hierarchy and PCG buffers are created by solve_head
            physics::HeadSolveConfig cfg;
            cfg.solver_type = physics::HeadSolverType::PCG_MG;
            cfg.mg_levels = spec.mg_levels;
            cfg.cg_rtol = spec.cg_rtol;
            cfg.cg_max_iter = spec.cg_max_iter;
            cfg.pin.mode = PinMode::Off;
            physics::init_head_guess(head.span(), grid, bc, ctx);
            const physics::HeadSolveResult r =
                physics::solve_head(head.span(), static_cast<const physics::KField&>(Kf).span(), grid, bc,
                                    cfg, ctx, ws);
            ctx.synchronize();
            MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
            DirichletMediumReport& R = M.report;
            R.converged = r.converged;
            R.iterations = r.num_iterations;
            R.initial_residual = r.initial_residual;
            R.final_residual = r.final_residual;
            R.rel = r.initial_residual > 0.0 ? r.final_residual / r.initial_residual : 0.0;

            rec.items = {{"K", cap(Kf.data)}, {"head", cap(head.data)}, {"rhs", cap(ws.rhs)}};
            ws_bytes = cap(ws.rhs);
            for (std::size_t l = 0; l < ws.mg_hierarchy.levels.size(); ++l) {
                const auto& L = ws.mg_hierarchy.levels[l];
                const std::size_t b = cap(L.x) + cap(L.b) + cap(L.r) + cap(L.coefficient);
                rec.items.emplace_back("mg_level" + std::to_string(l), b);
                ws_bytes += b;
            }
            const std::size_t pcg =
                cap(ws.pcg_workspace.r) + cap(ws.pcg_workspace.z) + cap(ws.pcg_workspace.p) + cap(ws.pcg_workspace.Ap);
            rec.items.emplace_back("pcg_r_z_p_Ap", pcg);
            ws_bytes += pcg;
            rec.device_used = device_used_now(); // everything alive: peak of the solve
            rec.seconds = since(t0);
            M.report.workspace_bytes = ws_bytes;
            M.phases.push_back(rec);

            DirichletPhaseRecord rel;
            rel.phase = "release";
            rel.device_used_start = device_used_now();
            const auto t1 = Clock::now();
            ws.clear(); // frees rhs, MG hierarchy, CG/PCG buffers
            MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
            rel.items = {{"K", cap(Kf.data)}, {"head", cap(head.data)}};
            rel.device_used = device_used_now();
            rel.seconds = since(t1);
            M.phases.push_back(rel);
        } // ws destroyed here: reduction workspace freed too
        MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- velocity + report --------------------------------------------------------------------
    physics::VelocityField vel;
    {
        DirichletPhaseRecord rec;
        rec.phase = "velocity";
        rec.device_used_start = device_used_now();
        const auto t0 = Clock::now();
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
        DirichletMediumReport& R = M.report;
        R.qbar = s_in / static_cast<real>(nf);
        R.Q_in = M.h * M.h * s_in;
        R.Q_out = M.h * M.h * s_out;
        R.balance = std::abs(R.Q_in - R.Q_out) / std::abs(R.Q_in);
        R.max_div = dmax / R.qbar;

        rec.items = {{"K", cap(Kf.data)},  {"head", cap(head.data)}, {"U", cap(vel.U)},
                     {"V", cap(vel.V)},    {"W", cap(vel.W)},        {"planes", cap(planes)},
                     {"div_block_max", cap(bmax)}};
        rec.device_used = device_used_now();
        rec.seconds = since(t0);
        M.phases.push_back(rec);
    }
    if (spec.keep_host_head)
        M.host_head = download(head.data.data(), n);
    Kf.data.reset(); // K and head are dead
    head.data.reset();
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());

    // ---- faces: optional host download, optional device periodic repack ------------------------
    {
        DirichletPhaseRecord rec;
        rec.phase = "faces";
        rec.device_used_start = device_used_now();
        const auto t0 = Clock::now();
        rec.items = {{"U", cap(vel.U)}, {"V", cap(vel.V)}, {"W", cap(vel.W)}};
        if (spec.keep_host_faces) {
            M.face_u = download(vel.U.data(), vel.U.size());
            M.face_v = download(vel.V.data(), vel.V.size());
            M.face_w = download(vel.W.data(), vel.W.size());
        }
        if (spec.build_periodic_faces) {
            M.pu_ = DeviceBuffer<real>(n);
            M.pv_ = DeviceBuffer<real>(n);
            M.pw_ = DeviceBuffer<real>(n);
            repack_compact_mac_to_periodic_faces(ctx, nx, ny, nz, vel.U.span(), vel.V.span(),
                                                 vel.W.span(), M.pu_.span(), M.pv_.span(),
                                                 M.pw_.span());
            ctx.synchronize();
            rec.items.emplace_back("periodic_u", cap(M.pu_));
            rec.items.emplace_back("periodic_v", cap(M.pv_));
            rec.items.emplace_back("periodic_w", cap(M.pw_));
        }
        rec.device_used = device_used_now(); // peak: CompactMAC + periodic alive
        rec.seconds = since(t0);
        M.phases.push_back(rec);
    }
    vel = physics::VelocityField(); // CompactMAC device arrays are dead
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
    return M;
}

} // namespace lagrangian_labels
} // namespace macroflow3d
