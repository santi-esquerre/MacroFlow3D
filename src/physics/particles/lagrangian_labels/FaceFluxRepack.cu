/**
 * @file FaceFluxRepack.cu
 * @brief SF-37 N3: CompactMAC -> periodic face layout repack (see FaceFluxRepack.cuh). Compiled with
 *        --fmad=false (CMakeLists.txt); the kernel is a pure copy.
 */

#include "FaceFluxRepack.cuh"

#include "../../../runtime/cuda_check.cuh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace macroflow3d {
namespace lagrangian_labels {

namespace {

constexpr int kBlock = 256;

/// dst[i + nx (j + ny k)] = src[i + rowlen (j + rows k)] for i < nx, j < ny, k < nz.
__global__ void repack_component_kernel(const real* __restrict__ src, real* __restrict__ dst, int nx,
                                        int ny, int nz, int rowlen, int rows) {
    const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                          static_cast<std::size_t>(nz);
    const std::size_t q = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (q >= n)
        return;
    const std::size_t i = q % static_cast<std::size_t>(nx);
    const std::size_t jk = q / static_cast<std::size_t>(nx);
    const std::size_t j = jk % static_cast<std::size_t>(ny);
    const std::size_t k = jk / static_cast<std::size_t>(ny);
    dst[q] = src[i + static_cast<std::size_t>(rowlen) * (j + static_cast<std::size_t>(rows) * k)];
}

void launch_component(CudaContext& ctx, const real* src, real* dst, int nx, int ny, int nz,
                      int rowlen, int rows) {
    const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                          static_cast<std::size_t>(nz);
    const unsigned blocks = static_cast<unsigned>((n + kBlock - 1) / kBlock);
    repack_component_kernel<<<blocks, kBlock, 0, ctx.cuda_stream()>>>(src, dst, nx, ny, nz, rowlen,
                                                                      rows);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void check_dims(int nx, int ny, int nz) {
    if (nx < 1 || ny < 1 || nz < 1) {
        throw std::invalid_argument("lagrangian_labels::repack: nx, ny, nz must be >= 1");
    }
}

void check_size(std::size_t got, std::size_t want, const char* what) {
    if (got != want) {
        throw std::invalid_argument(std::string("lagrangian_labels::repack: ") + what + " has " +
                                    std::to_string(got) + " entries, expected " +
                                    std::to_string(want));
    }
}

} // namespace

std::size_t compact_mac_u_size(int nx, int ny, int nz) {
    return static_cast<std::size_t>(nx + 1) * static_cast<std::size_t>(ny) *
           static_cast<std::size_t>(nz);
}
std::size_t compact_mac_v_size(int nx, int ny, int nz) {
    return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny + 1) *
           static_cast<std::size_t>(nz);
}
std::size_t compact_mac_w_size(int nx, int ny, int nz) {
    return static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
           static_cast<std::size_t>(nz + 1);
}

void repack_compact_mac_to_periodic_faces(CudaContext& ctx, int nx, int ny, int nz,
                                          DeviceSpan<const real> U, DeviceSpan<const real> V,
                                          DeviceSpan<const real> W, DeviceSpan<real> u_p,
                                          DeviceSpan<real> v_p, DeviceSpan<real> w_p) {
    check_dims(nx, ny, nz);
    const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                          static_cast<std::size_t>(nz);
    check_size(U.size(), compact_mac_u_size(nx, ny, nz), "U");
    check_size(V.size(), compact_mac_v_size(nx, ny, nz), "V");
    check_size(W.size(), compact_mac_w_size(nx, ny, nz), "W");
    check_size(u_p.size(), n, "u_p");
    check_size(v_p.size(), n, "v_p");
    check_size(w_p.size(), n, "w_p");
    launch_component(ctx, U.data(), u_p.data(), nx, ny, nz, nx + 1, ny);
    launch_component(ctx, V.data(), v_p.data(), nx, ny, nz, nx, ny + 1);
    launch_component(ctx, W.data(), w_p.data(), nx, ny, nz, nx, ny);
}

PeriodicFaceFluxes PeriodicFaceFluxes::upload_and_repack(CudaContext& ctx, const std::vector<real>& U,
                                                         const std::vector<real>& V,
                                                         const std::vector<real>& W, int nx, int ny,
                                                         int nz, real h) {
    check_dims(nx, ny, nz);
    check_size(U.size(), compact_mac_u_size(nx, ny, nz), "U");
    check_size(V.size(), compact_mac_v_size(nx, ny, nz), "V");
    check_size(W.size(), compact_mac_w_size(nx, ny, nz), "W");
    if (!(std::isfinite(h) && h > 0.0)) {
        throw std::invalid_argument("lagrangian_labels::repack: h must be finite and > 0");
    }
    const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                          static_cast<std::size_t>(nz);
    PeriodicFaceFluxes out;
    out.u_ = DeviceBuffer<real>(n);
    out.v_ = DeviceBuffer<real>(n);
    out.w_ = DeviceBuffer<real>(n);
    out.nx_ = nx;
    out.ny_ = ny;
    out.nz_ = nz;
    out.h_ = h;

    DeviceBuffer<real> stage(std::max({U.size(), V.size(), W.size()}));
    struct Comp {
        const std::vector<real>* host;
        real* dst;
        int rowlen, rows;
    };
    const Comp comps[3] = {{&U, out.u_.data(), nx + 1, ny},
                           {&V, out.v_.data(), nx, ny + 1},
                           {&W, out.w_.data(), nx, ny}};
    for (const Comp& c : comps) {
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(stage.data(), c.host->data(),
                                          c.host->size() * sizeof(real), cudaMemcpyHostToDevice));
        // SF-33 C2: the legacy-stream copy must land before ctx-stream work.
        MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
        launch_component(ctx, stage.data(), c.dst, nx, ny, nz, c.rowlen, c.rows);
        // The staging buffer is reused by the next component: complete the kernel first.
        ctx.synchronize();
    }
    return out;
}

physics::particles::streamline_tracker::PeriodicFaceFluxView PeriodicFaceFluxes::view() const {
    if (empty()) {
        throw std::logic_error("PeriodicFaceFluxes::view: empty");
    }
    physics::particles::streamline_tracker::PeriodicFaceFluxView f{};
    f.u = u_.data();
    f.v = v_.data();
    f.w = w_.data();
    f.nx = nx_;
    f.ny = ny_;
    f.nz = nz_;
    f.dx = f.dy = f.dz = h_;
    f.Lx = static_cast<real>(nx_) * h_;
    f.Ly = static_cast<real>(ny_) * h_;
    f.Lz = static_cast<real>(nz_) * h_;
    return f;
}

NonPeriodicXFaceFluxes NonPeriodicXFaceFluxes::upload(CudaContext& ctx, const std::vector<real>& U,
                                                      const std::vector<real>& V,
                                                      const std::vector<real>& W, int nx, int ny,
                                                      int nz, real h) {
    check_dims(nx, ny, nz);
    check_size(U.size(), compact_mac_u_size(nx, ny, nz), "U");
    check_size(V.size(), compact_mac_v_size(nx, ny, nz), "V");
    check_size(W.size(), compact_mac_w_size(nx, ny, nz), "W");
    if (!(std::isfinite(h) && h > 0.0)) {
        throw std::invalid_argument("lagrangian_labels::repack: h must be finite and > 0");
    }
    const std::size_t n = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                          static_cast<std::size_t>(nz);
    NonPeriodicXFaceFluxes out;
    out.u_ = DeviceBuffer<real>(U.size());
    out.v_ = DeviceBuffer<real>(n);
    out.w_ = DeviceBuffer<real>(n);
    out.nx_ = nx;
    out.ny_ = ny;
    out.nz_ = nz;
    out.h_ = h;

    MACROFLOW3D_CUDA_CHECK(cudaMemcpy(out.u_.data(), U.data(), U.size() * sizeof(real),
                                      cudaMemcpyHostToDevice));
    DeviceBuffer<real> stage(std::max(V.size(), W.size()));
    struct Comp {
        const std::vector<real>* host;
        real* dst;
        int rowlen, rows;
    };
    const Comp comps[2] = {{&V, out.v_.data(), nx, ny + 1}, {&W, out.w_.data(), nx, ny}};
    for (const Comp& c : comps) {
        MACROFLOW3D_CUDA_CHECK(cudaMemcpy(stage.data(), c.host->data(),
                                          c.host->size() * sizeof(real), cudaMemcpyHostToDevice));
        // SF-33 C2: the legacy-stream copies (U included) must land before ctx-stream work.
        MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
        launch_component(ctx, stage.data(), c.dst, nx, ny, nz, c.rowlen, c.rows);
        ctx.synchronize();
    }
    return out;
}

physics::particles::streamline_tracker::NonPeriodicXFaceFluxView
NonPeriodicXFaceFluxes::view() const {
    if (empty()) {
        throw std::logic_error("NonPeriodicXFaceFluxes::view: empty");
    }
    physics::particles::streamline_tracker::NonPeriodicXFaceFluxView f{};
    f.u = u_.data();
    f.v = v_.data();
    f.w = w_.data();
    f.nx = nx_;
    f.ny = ny_;
    f.nz = nz_;
    f.dx = f.dy = f.dz = h_;
    f.Lx = static_cast<real>(nx_) * h_;
    f.Ly = static_cast<real>(ny_) * h_;
    f.Lz = static_cast<real>(nz_) * h_;
    return f;
}

} // namespace lagrangian_labels
} // namespace macroflow3d
