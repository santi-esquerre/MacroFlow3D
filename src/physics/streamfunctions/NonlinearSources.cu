#include "NonlinearSources.cuh"

#include "../../runtime/cuda_check.cuh"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace macroflow3d {
namespace streamfunctions {
namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 65535;
constexpr int kCounterFinite = 2; // fixed leading counters: [0] S1, [1] S2

struct NonlinearSourceKernelConfig {
    real epsilon;
    real v_rms;
    int num_degeneracy_thresholds;
    real degeneracy_thresholds[kMaxDegeneracyThresholds];
};

static_assert(std::is_trivially_copyable<NonlinearSourceKernelConfig>::value,
              "NonlinearSourceKernelConfig must be safe to copy into a kernel argument");

__global__ void streamfunction_nonlinear_sources_kernel(
    const real* psi1_x, const real* psi1_y, const real* psi1_z, const real* psi2_x,
    const real* psi2_y, const real* psi2_z, const real* b_x, const real* b_y, const real* b_z,
    real* s1, real* s2, unsigned long long* counters, std::size_t n,
    NonlinearSourceKernelConfig config) {
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;

    unsigned long long local_nonfinite_s1 = 0;
    unsigned long long local_nonfinite_s2 = 0;
    unsigned long long local_degeneracy[kMaxDegeneracyThresholds] = {};

    const real regularization = config.epsilon * config.v_rms;
    const real regularization_sq = regularization * regularization;

    for (std::size_t index = start; index < n; index += stride) {
        const real g1x = psi1_x[index];
        const real g1y = psi1_y[index];
        const real g1z = psi1_z[index];
        const real g2x = psi2_x[index];
        const real g2y = psi2_y[index];
        const real g2z = psi2_z[index];
        const real bx = b_x[index];
        const real by = b_y[index];
        const real bz = b_z[index];

        const real cx = g1y * g2z - g1z * g2y;
        const real cy = g1z * g2x - g1x * g2z;
        const real cz = g1x * g2y - g1y * g2x;
        const real c_sq = cx * cx + cy * cy + cz * cz;
        const real d = c_sq + regularization_sq;

        const real bxg1_x = by * g1z - bz * g1y;
        const real bxg1_y = bz * g1x - bx * g1z;
        const real bxg1_z = bx * g1y - by * g1x;
        const real bxg2_x = by * g2z - bz * g2y;
        const real bxg2_y = bz * g2x - bx * g2z;
        const real bxg2_z = bx * g2y - by * g2x;

        const real numerator_s1 = bxg1_x * cx + bxg1_y * cy + bxg1_z * cz;
        const real numerator_s2 = bxg2_x * cx + bxg2_y * cy + bxg2_z * cz;

        const real value_s1 = numerator_s1 / d;
        const real value_s2 = numerator_s2 / d;

        s1[index] = value_s1;
        s2[index] = value_s2;

        if (!isfinite(value_s1)) {
            ++local_nonfinite_s1;
        }
        if (!isfinite(value_s2)) {
            ++local_nonfinite_s2;
        }
        for (int t = 0; t < config.num_degeneracy_thresholds; ++t) {
            const real tau = config.degeneracy_thresholds[t];
            const real threshold_sq = (tau * config.v_rms) * (tau * config.v_rms);
            if (c_sq < threshold_sq) {
                ++local_degeneracy[t];
            }
        }
    }

    if (local_nonfinite_s1 != 0) {
        atomicAdd(&counters[0], local_nonfinite_s1);
    }
    if (local_nonfinite_s2 != 0) {
        atomicAdd(&counters[1], local_nonfinite_s2);
    }
    for (int t = 0; t < config.num_degeneracy_thresholds; ++t) {
        if (local_degeneracy[t] != 0) {
            atomicAdd(&counters[kCounterFinite + t], local_degeneracy[t]);
        }
    }
}

std::size_t require_valid_grid(const Grid3D& grid) {
    if (grid.nx <= 0 || grid.ny <= 0 || grid.nz <= 0) {
        throw std::invalid_argument("Streamfunction nonlinear sources require positive grid extents");
    }
    if (!std::isfinite(grid.dx) || !std::isfinite(grid.dy) || !std::isfinite(grid.dz) ||
        grid.dx <= real{0} || grid.dy <= real{0} || grid.dz <= real{0}) {
        throw std::invalid_argument(
            "Streamfunction nonlinear sources require finite positive direction-specific spacing");
    }

    const auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(real);
    const auto nx = static_cast<std::size_t>(grid.nx);
    const auto ny = static_cast<std::size_t>(grid.ny);
    const auto nz = static_cast<std::size_t>(grid.nz);
    if (nx > max_elements / ny || nx * ny > max_elements / nz) {
        throw std::invalid_argument(
            "Streamfunction nonlinear sources require a grid whose storage size does not overflow");
    }
    return nx * ny * nz;
}

void require_valid_config(const NonlinearSourceConfig& config) {
    if (!std::isfinite(config.epsilon) || config.epsilon < real{0}) {
        throw std::invalid_argument(
            "Streamfunction nonlinear sources require a finite, non-negative epsilon");
    }
    if (!std::isfinite(config.v_rms) || config.v_rms <= real{0}) {
        throw std::invalid_argument(
            "Streamfunction nonlinear sources require a finite, strictly positive v_rms");
    }
    if (config.num_degeneracy_thresholds < 0 ||
        config.num_degeneracy_thresholds > kMaxDegeneracyThresholds) {
        throw std::invalid_argument(
            "Streamfunction nonlinear sources require a degeneracy threshold count within bounds");
    }
    for (int t = 0; t < config.num_degeneracy_thresholds; ++t) {
        const real tau = config.degeneracy_thresholds[t];
        if (!std::isfinite(tau) || tau < real{0}) {
            throw std::invalid_argument(
                "Streamfunction nonlinear sources require finite, non-negative degeneracy thresholds");
        }
    }
}

bool byte_ranges_overlap(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) {
    const auto a_begin = reinterpret_cast<std::uintptr_t>(a);
    const auto b_begin = reinterpret_cast<std::uintptr_t>(b);
    if (a_begin <= b_begin) {
        return b_begin - a_begin < a_bytes;
    }
    return a_begin - b_begin < b_bytes;
}

template <typename T>
bool overlaps(const DeviceSpan<T>& a, const DeviceSpan<T>& b) {
    return byte_ranges_overlap(a.data(), a.size() * sizeof(T), b.data(), b.size() * sizeof(T));
}

void require_exact_nonoverlapping_spans(const TotalStreamfunctionGradientView& total_gradients,
                                        const StreamfunctionBFieldView& b,
                                        const NonlinearSourceOutput& output,
                                        const NonlinearSourceCounters& counters, std::size_t n,
                                        std::size_t num_counters) {
    const DeviceSpan<const real> inputs[] = {total_gradients.psi1_x, total_gradients.psi1_y,
                                             total_gradients.psi1_z, total_gradients.psi2_x,
                                             total_gradients.psi2_y, total_gradients.psi2_z,
                                             b.b_x, b.b_y, b.b_z};
    for (const auto input : inputs) {
        if (input.size() != n || input.data() == nullptr) {
            throw std::invalid_argument(
                "Streamfunction nonlinear sources require non-null inputs of exact grid size");
        }
    }

    const DeviceSpan<real> outputs[] = {output.s1, output.s2};
    for (const auto output_span : outputs) {
        if (output_span.size() != n || output_span.data() == nullptr) {
            throw std::invalid_argument(
                "Streamfunction nonlinear sources require non-null outputs of exact grid size");
        }
    }
    if (overlaps(output.s1, output.s2)) {
        throw std::invalid_argument(
            "Streamfunction nonlinear source output spans must not overlap");
    }
    for (const auto output_span : outputs) {
        for (const auto input : inputs) {
            if (byte_ranges_overlap(output_span.data(), output_span.size() * sizeof(real),
                                    input.data(), input.size() * sizeof(real))) {
                throw std::invalid_argument(
                    "Streamfunction nonlinear source outputs must not overlap any input");
            }
        }
    }

    if (counters.counters.size() != num_counters || counters.counters.data() == nullptr) {
        throw std::invalid_argument(
            "Streamfunction nonlinear sources require a non-null counter span of exact size");
    }
    const auto counters_bytes = counters.counters.size() * sizeof(unsigned long long);
    for (const auto output_span : outputs) {
        if (byte_ranges_overlap(output_span.data(), output_span.size() * sizeof(real),
                                counters.counters.data(), counters_bytes)) {
            throw std::invalid_argument(
                "Streamfunction nonlinear source counters must not overlap the source outputs");
        }
    }
    for (const auto input : inputs) {
        if (byte_ranges_overlap(counters.counters.data(), counters_bytes, input.data(),
                                input.size() * sizeof(real))) {
            throw std::invalid_argument(
                "Streamfunction nonlinear source counters must not overlap any input");
        }
    }
}

} // namespace

void enqueue_streamfunction_nonlinear_sources(
    CudaContext& ctx, const Grid3D& grid, const TotalStreamfunctionGradientView& total_gradients,
    const StreamfunctionBFieldView& b, const NonlinearSourceConfig& config,
    const NonlinearSourceOutput& output, const NonlinearSourceCounters& counters) {
    const std::size_t n = require_valid_grid(grid);
    require_valid_config(config);
    const std::size_t num_counters =
        static_cast<std::size_t>(kCounterFinite + config.num_degeneracy_thresholds);
    require_exact_nonoverlapping_spans(total_gradients, b, output, counters, n, num_counters);

    MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(counters.counters.data(), 0,
                                           num_counters * sizeof(unsigned long long),
                                           ctx.cuda_stream()));

    NonlinearSourceKernelConfig kernel_config{};
    kernel_config.epsilon = config.epsilon;
    kernel_config.v_rms = config.v_rms;
    kernel_config.num_degeneracy_thresholds = config.num_degeneracy_thresholds;
    for (int t = 0; t < config.num_degeneracy_thresholds; ++t) {
        kernel_config.degeneracy_thresholds[t] = config.degeneracy_thresholds[t];
    }

    const std::size_t requested_blocks = (n + kBlockSize - 1) / kBlockSize;
    const int blocks = static_cast<int>(
        requested_blocks < static_cast<std::size_t>(kMaxBlocks) ? requested_blocks : kMaxBlocks);
    streamfunction_nonlinear_sources_kernel<<<blocks, kBlockSize, 0, ctx.cuda_stream()>>>(
        total_gradients.psi1_x.data(), total_gradients.psi1_y.data(),
        total_gradients.psi1_z.data(), total_gradients.psi2_x.data(),
        total_gradients.psi2_y.data(), total_gradients.psi2_z.data(), b.b_x.data(), b.b_y.data(),
        b.b_z.data(), output.s1.data(), output.s2.data(), counters.counters.data(), n,
        kernel_config);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

} // namespace streamfunctions
} // namespace macroflow3d
