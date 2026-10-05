/**
 * @file StreamlineTrackerCommon.cu
 * @brief SF-31 shared streamline-tracker bookkeeping - implementation
 *        (label-pair factory, injection and unwrapped-position kernels).
 *
 * See StreamlineTrackerCommon.cuh for the full specification (label
 * definition, evaluator concept, wrap rule, status codes, injection formula).
 */

#include "../../../runtime/cuda_check.cuh"
#include "StreamlineTrackerCommon.cuh"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace streamline_tracker {

namespace {

constexpr int kBlockSize = 256;

__global__ void kernel_inject_box(real* __restrict__ px, real* __restrict__ py,
                                  real* __restrict__ pz, uint8_t* __restrict__ status,
                                  int32_t* __restrict__ wx, int32_t* __restrict__ wy,
                                  int32_t* __restrict__ wz, int first, int count, real x0,
                                  real y0, real z0, real x1, real y1, real z1, uint64_t seed) {
    const int k = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (k >= count)
        return;
    const int i = first + k;
    const uint64_t index = static_cast<uint64_t>(i);
    px[i] = inject_coordinate(x0, x1, inject_uniform01(seed, index, 0));
    py[i] = inject_coordinate(y0, y1, inject_uniform01(seed, index, 1));
    pz[i] = inject_coordinate(z0, z1, inject_uniform01(seed, index, 2));
    status[i] = kStatusActive;
    wx[i] = 0;
    wy[i] = 0;
    wz[i] = 0;
}

__global__ void kernel_compute_unwrapped(const real* __restrict__ px, const real* __restrict__ py,
                                         const real* __restrict__ pz,
                                         const int32_t* __restrict__ wx,
                                         const int32_t* __restrict__ wy,
                                         const int32_t* __restrict__ wz, int n, real Lx, real Ly,
                                         real Lz, real* __restrict__ xu, real* __restrict__ yu,
                                         real* __restrict__ zu) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
    if (i >= n)
        return;
    xu[i] = fma(static_cast<real>(wx[i]), Lx, px[i]);
    yu[i] = fma(static_cast<real>(wy[i]), Ly, py[i]);
    zu[i] = fma(static_cast<real>(wz[i]), Lz, pz[i]);
}

int grid_size(int n) {
    return (n + kBlockSize - 1) / kBlockSize;
}

} // namespace

// ============================================================================
// Label-pair factory
// ============================================================================

SplineLabelPair make_spline_label_pair(const interpolation::PeriodicTricubicBSplineView& s1,
                                       const interpolation::PeriodicTricubicBSplineView& s2,
                                       const real gbar1[3], const real gbar2[3]) {
    if (s1.coeff == nullptr) {
        throw std::invalid_argument(
            "make_spline_label_pair: s1 coefficient pointer is null");
    }
    if (s2.coeff == nullptr) {
        throw std::invalid_argument(
            "make_spline_label_pair: s2 coefficient pointer is null");
    }
    if (s1.nx != s2.nx || s1.ny != s2.ny || s1.nz != s2.nz) {
        throw std::invalid_argument(
            "make_spline_label_pair: s1 and s2 differ in grid extents (nx, ny, nz)");
    }
    if (s1.hx != s2.hx || s1.hy != s2.hy || s1.hz != s2.hz) {
        throw std::invalid_argument(
            "make_spline_label_pair: s1 and s2 differ in grid spacings (hx, hy, hz)");
    }
    if (s1.Lx != s2.Lx || s1.Ly != s2.Ly || s1.Lz != s2.Lz) {
        throw std::invalid_argument(
            "make_spline_label_pair: s1 and s2 differ in periods (Lx, Ly, Lz)");
    }
    for (int d = 0; d < 3; ++d) {
        if (!std::isfinite(gbar1[d])) {
            throw std::invalid_argument(
                "make_spline_label_pair: a gbar1 component is not finite");
        }
    }
    for (int d = 0; d < 3; ++d) {
        if (!std::isfinite(gbar2[d])) {
            throw std::invalid_argument(
                "make_spline_label_pair: a gbar2 component is not finite");
        }
    }

    SplineLabelPair pair{};
    pair.s1 = s1;
    pair.s2 = s2;
    for (int d = 0; d < 3; ++d) {
        pair.gbar1[d] = gbar1[d];
        pair.gbar2[d] = gbar2[d];
    }
    pair.L[0] = s1.Lx;
    pair.L[1] = s1.Ly;
    pair.L[2] = s1.Lz;
    return pair;
}

// ============================================================================
// inject_box
// ============================================================================

void inject_box(cudaStream_t stream, const ParticlesSoA<real>& p, real x0, real y0, real z0,
                real x1, real y1, real z1, int first, int count, uint64_t seed) {
    if (p.x == nullptr || p.y == nullptr || p.z == nullptr) {
        throw std::invalid_argument("streamline_tracker::inject_box: a position pointer "
                                    "(x, y or z) is null");
    }
    if (p.status == nullptr) {
        throw std::invalid_argument("streamline_tracker::inject_box: status pointer is null");
    }
    if (p.wrapX == nullptr || p.wrapY == nullptr || p.wrapZ == nullptr) {
        throw std::invalid_argument("streamline_tracker::inject_box: a wrap-counter pointer "
                                    "(wrapX, wrapY or wrapZ) is null (wrap arrays are required)");
    }
    if (first < 0) {
        throw std::invalid_argument("streamline_tracker::inject_box: first < 0");
    }
    if (count < 0) {
        throw std::invalid_argument("streamline_tracker::inject_box: count < 0");
    }
    if (static_cast<int64_t>(first) + static_cast<int64_t>(count) >
        static_cast<int64_t>(p.n)) {
        throw std::invalid_argument("streamline_tracker::inject_box: first + count > p.n");
    }
    if (!std::isfinite(x0) || !std::isfinite(y0) || !std::isfinite(z0) || !std::isfinite(x1) ||
        !std::isfinite(y1) || !std::isfinite(z1)) {
        throw std::invalid_argument("streamline_tracker::inject_box: a box bound is not finite");
    }
    if (x1 < x0 || y1 < y0 || z1 < z0) {
        throw std::invalid_argument(
            "streamline_tracker::inject_box: upper box bound below lower bound (p1 < p0) on "
            "an axis");
    }
    if (count == 0)
        return;

    kernel_inject_box<<<grid_size(count), kBlockSize, 0, stream>>>(
        p.x, p.y, p.z, p.status, p.wrapX, p.wrapY, p.wrapZ, first, count, x0, y0, z0, x1, y1,
        z1, seed);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// compute_unwrapped
// ============================================================================

void compute_unwrapped(cudaStream_t stream, const ConstParticlesSoA<real>& p, const real L[3],
                       const UnwrappedSoA<real>& uw) {
    if (p.x == nullptr || p.y == nullptr || p.z == nullptr) {
        throw std::invalid_argument("streamline_tracker::compute_unwrapped: a position pointer "
                                    "(x, y or z) is null");
    }
    if (p.status == nullptr) {
        throw std::invalid_argument(
            "streamline_tracker::compute_unwrapped: status pointer is null");
    }
    if (p.wrapX == nullptr || p.wrapY == nullptr || p.wrapZ == nullptr) {
        throw std::invalid_argument(
            "streamline_tracker::compute_unwrapped: a wrap-counter pointer (wrapX, wrapY or "
            "wrapZ) is null (wrap arrays are required)");
    }
    if (p.n < 0) {
        throw std::invalid_argument("streamline_tracker::compute_unwrapped: p.n < 0");
    }
    for (int d = 0; d < 3; ++d) {
        if (!std::isfinite(L[d]) || !(L[d] > 0.0)) {
            throw std::invalid_argument(
                "streamline_tracker::compute_unwrapped: a period L[d] is not finite or not > 0");
        }
    }
    if (!uw.valid()) {
        throw std::invalid_argument("streamline_tracker::compute_unwrapped: unwrapped buffers "
                                    "are not valid (null pointer or capacity <= 0)");
    }
    if (uw.capacity < p.n) {
        throw std::invalid_argument(
            "streamline_tracker::compute_unwrapped: unwrapped capacity < p.n");
    }
    if (p.n == 0)
        return;

    kernel_compute_unwrapped<<<grid_size(p.n), kBlockSize, 0, stream>>>(
        p.x, p.y, p.z, p.wrapX, p.wrapY, p.wrapZ, p.n, L[0], L[1], L[2], uw.x_u, uw.y_u,
        uw.z_u);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

} // namespace streamline_tracker
} // namespace particles
} // namespace physics
} // namespace macroflow3d
