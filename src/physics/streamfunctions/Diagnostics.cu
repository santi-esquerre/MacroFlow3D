#include "Diagnostics.cuh"

#include "../../numerics/blas/dot.cuh"
#include "../../numerics/blas/nrm2.cuh"
#include "../../numerics/blas/sum.cuh"
#include "../../runtime/cuda_check.cuh"
#include "DifferentialOperators.cuh"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace macroflow3d {
namespace streamfunctions {
namespace {

constexpr int kBlockSize = 256;
constexpr int kMaxBlocks = 65535;

// Fixed layout within StreamfunctionDiagnosticsWorkspace::scalars_.
constexpr std::size_t kScalarNu = 0;
constexpr std::size_t kScalarNv = 1;
constexpr std::size_t kScalarNw = 2;
constexpr std::size_t kScalarL2DiffU = 3;
constexpr std::size_t kScalarL2DiffV = 4;
constexpr std::size_t kScalarL2DiffW = 5;
constexpr std::size_t kScalarLinfDiffU = 6;
constexpr std::size_t kScalarLinfDiffV = 7;
constexpr std::size_t kScalarLinfDiffW = 8;
// Correlation sums [Sx, Sy, Sxx, Syy, Sxy] per component, addressed by base
// offset (kScalarCorr{U,V,W}Sx + 0..4); only the base constants are named
// directly, the rest are documented here for readers.
constexpr std::size_t kScalarCorrUSx = 9;   // + 1 = Sy, +2 = Sxx, +3 = Syy, +4 = Sxy
constexpr std::size_t kScalarCorrVSx = 14;  // + 1 = Sy, +2 = Sxx, +3 = Syy, +4 = Sxy
constexpr std::size_t kScalarCorrWSx = 19;  // + 1 = Sy, +2 = Sxx, +3 = Syy, +4 = Sxy
constexpr std::size_t kScalarL2M = 24;
constexpr std::size_t kScalarLinfM = 25;
constexpr std::size_t kScalarL2Theta = 26;
constexpr std::size_t kScalarMaxTheta = 27;
constexpr std::size_t kScalarL2Dot1 = 28;
constexpr std::size_t kScalarL2Dot2 = 29;
constexpr std::size_t kScalarL2Div = 30;
constexpr std::size_t kScalarLinfDiv = 31;
constexpr std::size_t kScalarMinC = 32;
constexpr std::size_t kScalarMaxC = 33;
constexpr std::size_t kScalarSumC = 34;
constexpr std::size_t kScalarL2G1x = 35;
constexpr std::size_t kScalarL2G1y = 36;
constexpr std::size_t kScalarL2G1z = 37;
constexpr std::size_t kScalarL2G2x = 38;
constexpr std::size_t kScalarL2G2y = 39;
constexpr std::size_t kScalarL2G2z = 40;
constexpr std::size_t kNumScalars = 41;

// Fixed layout within StreamfunctionDiagnosticsWorkspace::counters_.
constexpr std::size_t kCounterIncluded = 0;
constexpr std::size_t kCounterExcluded = 1;
constexpr std::size_t kCounterLeading = 2;
constexpr std::size_t kNumCounters =
    kCounterLeading + static_cast<std::size_t>(2 * kMaxDegeneracyThresholds);

// CAS-based double atomicMax/atomicMin, valid for the non-negative
// quantities used in this module (|diff|, |m|, theta, |div|, |c|).
__device__ __forceinline__ void atomic_max_double(double* addr, double val) {
    unsigned long long* addr_ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long assumed;
    unsigned long long old = *addr_ull;
    do {
        assumed = old;
        const double current = __longlong_as_double(static_cast<long long>(assumed));
        if (val <= current) {
            break;
        }
        old = atomicCAS(addr_ull, assumed, static_cast<unsigned long long>(__double_as_longlong(val)));
    } while (assumed != old);
}

__device__ __forceinline__ void atomic_min_double(double* addr, double val) {
    unsigned long long* addr_ull = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long assumed;
    unsigned long long old = *addr_ull;
    do {
        assumed = old;
        const double current = __longlong_as_double(static_cast<long long>(assumed));
        if (val >= current) {
            break;
        }
        old = atomicCAS(addr_ull, assumed, static_cast<unsigned long long>(__double_as_longlong(val)));
    } while (assumed != old);
}

// Single-thread kernel: cudaMemsetAsync cannot express +infinity, so the
// |c| minimum accumulator is initialized by a dedicated kernel launch
// instead (not a CUDA runtime call).
__global__ void init_min_c_kernel(real* scalars) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        scalars[kScalarMinC] = static_cast<real>(INFINITY);
    }
}

struct DiagnosticsKernelConfig {
    real angle_exclusion_rel;
    real low_speed_rel;
    int num_degeneracy_thresholds;
    real degeneracy_thresholds[kMaxDegeneracyThresholds];
};

static_assert(std::is_trivially_copyable<DiagnosticsKernelConfig>::value,
              "DiagnosticsKernelConfig must be safe to copy into a kernel argument");

// Reconstructs v_psi on every CompactMAC face (interpolate-then-cross, see
// Diagnostics.cuh), writes the periodic duplicate planes, and materializes
// the unique-face reconstructed/Darcy/difference scratch arrays used by the
// per-component reduction metrics.
__global__ void reconstruct_and_face_diagnostics_kernel(
    const real* g1x, const real* g1y, const real* g1z, const real* g2x, const real* g2y,
    const real* g2z, const real* darcy_u, const real* darcy_v, const real* darcy_w, real* vpsi_u,
    real* vpsi_v, real* vpsi_w, real* unique_vpsi_u, real* unique_vpsi_v, real* unique_vpsi_w,
    real* unique_vd_u, real* unique_vd_v, real* unique_vd_w, real* diff_u, real* diff_v,
    real* diff_w, real* scalars, int nx, int ny, int nz) {
    const std::size_t n =
        static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const std::size_t y_stride = static_cast<std::size_t>(nx);
    const std::size_t z_stride = y_stride * static_cast<std::size_t>(ny);
    const std::size_t face_u_j_stride = static_cast<std::size_t>(nx + 1);
    const std::size_t face_u_k_stride = face_u_j_stride * static_cast<std::size_t>(ny);
    const std::size_t face_v_k_stride = y_stride * static_cast<std::size_t>(ny + 1);
    const std::size_t face_w_k_stride = y_stride * static_cast<std::size_t>(ny);

    real local_max_diff_u = real{0};
    real local_max_diff_v = real{0};
    real local_max_diff_w = real{0};

    for (std::size_t index = start; index < n; index += stride) {
        const int i = static_cast<int>(index % y_stride);
        const int j = static_cast<int>((index / y_stride) % static_cast<std::size_t>(ny));
        const int k = static_cast<int>(index / z_stride);

        // U-face: cells wrap(i-1), wrap(i)=i.
        {
            const std::size_t a_i = static_cast<std::size_t>(i == 0 ? nx - 1 : i - 1);
            const std::size_t cell_a = a_i + y_stride * (static_cast<std::size_t>(j) +
                                                          static_cast<std::size_t>(ny) *
                                                              static_cast<std::size_t>(k));
            const std::size_t cell_b = index;
            const real t1y = real{0.5} * (g1y[cell_a] + g1y[cell_b]);
            const real t1z = real{0.5} * (g1z[cell_a] + g1z[cell_b]);
            const real t2y = real{0.5} * (g2y[cell_a] + g2y[cell_b]);
            const real t2z = real{0.5} * (g2z[cell_a] + g2z[cell_b]);
            const real uval = t1y * t2z - t1z * t2y;
            const std::size_t face = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * face_u_j_stride +
                                      static_cast<std::size_t>(k) * face_u_k_stride;
            vpsi_u[face] = uval;
            unique_vpsi_u[index] = uval;
            const real ud = darcy_u[face];
            unique_vd_u[index] = ud;
            const real diffu = uval - ud;
            diff_u[index] = diffu;
            if (i == 0) {
                const std::size_t dup = static_cast<std::size_t>(nx) + static_cast<std::size_t>(j) * face_u_j_stride +
                                        static_cast<std::size_t>(k) * face_u_k_stride;
                vpsi_u[dup] = uval;
            }
            const real abs_diff_u = fabs(diffu);
            if (abs_diff_u > local_max_diff_u) {
                local_max_diff_u = abs_diff_u;
            }
        }

        // V-face: cells wrap(j-1), wrap(j)=j.
        {
            const std::size_t b_j = static_cast<std::size_t>(j == 0 ? ny - 1 : j - 1);
            const std::size_t cell_a = static_cast<std::size_t>(i) +
                                       y_stride * (b_j + static_cast<std::size_t>(ny) *
                                                             static_cast<std::size_t>(k));
            const std::size_t cell_b = index;
            const real t1z = real{0.5} * (g1z[cell_a] + g1z[cell_b]);
            const real t1x = real{0.5} * (g1x[cell_a] + g1x[cell_b]);
            const real t2z = real{0.5} * (g2z[cell_a] + g2z[cell_b]);
            const real t2x = real{0.5} * (g2x[cell_a] + g2x[cell_b]);
            const real vval = t1z * t2x - t1x * t2z;
            const std::size_t face = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * y_stride +
                                      static_cast<std::size_t>(k) * face_v_k_stride;
            vpsi_v[face] = vval;
            unique_vpsi_v[index] = vval;
            const real vd = darcy_v[face];
            unique_vd_v[index] = vd;
            const real diffv = vval - vd;
            diff_v[index] = diffv;
            if (j == 0) {
                const std::size_t dup = static_cast<std::size_t>(i) + static_cast<std::size_t>(ny) * y_stride +
                                        static_cast<std::size_t>(k) * face_v_k_stride;
                vpsi_v[dup] = vval;
            }
            const real abs_diff_v = fabs(diffv);
            if (abs_diff_v > local_max_diff_v) {
                local_max_diff_v = abs_diff_v;
            }
        }

        // W-face: cells wrap(k-1), wrap(k)=k.
        {
            const std::size_t c_k = static_cast<std::size_t>(k == 0 ? nz - 1 : k - 1);
            const std::size_t cell_a = static_cast<std::size_t>(i) +
                                       y_stride * (static_cast<std::size_t>(j) +
                                                   static_cast<std::size_t>(ny) * c_k);
            const std::size_t cell_b = index;
            const real t1x = real{0.5} * (g1x[cell_a] + g1x[cell_b]);
            const real t1y = real{0.5} * (g1y[cell_a] + g1y[cell_b]);
            const real t2x = real{0.5} * (g2x[cell_a] + g2x[cell_b]);
            const real t2y = real{0.5} * (g2y[cell_a] + g2y[cell_b]);
            const real wval = t1x * t2y - t1y * t2x;
            const std::size_t face = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * y_stride +
                                      static_cast<std::size_t>(k) * face_w_k_stride;
            vpsi_w[face] = wval;
            unique_vpsi_w[index] = wval;
            const real wd = darcy_w[face];
            unique_vd_w[index] = wd;
            const real diffw = wval - wd;
            diff_w[index] = diffw;
            if (k == 0) {
                const std::size_t dup = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * y_stride +
                                        static_cast<std::size_t>(nz) * face_w_k_stride;
                vpsi_w[dup] = wval;
            }
            const real abs_diff_w = fabs(diffw);
            if (abs_diff_w > local_max_diff_w) {
                local_max_diff_w = abs_diff_w;
            }
        }
    }

    if (local_max_diff_u > real{0}) {
        atomic_max_double(&scalars[kScalarLinfDiffU], local_max_diff_u);
    }
    if (local_max_diff_v > real{0}) {
        atomic_max_double(&scalars[kScalarLinfDiffV], local_max_diff_v);
    }
    if (local_max_diff_w > real{0}) {
        atomic_max_double(&scalars[kScalarLinfDiffW], local_max_diff_w);
    }
}

// Cell-centered face-to-center averaging (pinned), applied identically to
// v_psi and the Darcy input.
__global__ void face_to_center_kernel(const real* vpsi_u, const real* vpsi_v, const real* vpsi_w,
                                      const real* darcy_u, const real* darcy_v,
                                      const real* darcy_w, real* vpsi_cx, real* vpsi_cy,
                                      real* vpsi_cz, real* vd_cx, real* vd_cy, real* vd_cz, int nx,
                                      int ny, int nz) {
    const std::size_t n =
        static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const std::size_t y_stride = static_cast<std::size_t>(nx);
    const std::size_t z_stride = y_stride * static_cast<std::size_t>(ny);
    const std::size_t face_u_j_stride = static_cast<std::size_t>(nx + 1);
    const std::size_t face_u_k_stride = face_u_j_stride * static_cast<std::size_t>(ny);
    const std::size_t face_v_k_stride = y_stride * static_cast<std::size_t>(ny + 1);
    const std::size_t face_w_k_stride = y_stride * static_cast<std::size_t>(ny);

    for (std::size_t index = start; index < n; index += stride) {
        const int i = static_cast<int>(index % y_stride);
        const int j = static_cast<int>((index / y_stride) % static_cast<std::size_t>(ny));
        const int k = static_cast<int>(index / z_stride);

        const std::size_t face_u0 = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * face_u_j_stride +
                                    static_cast<std::size_t>(k) * face_u_k_stride;
        const std::size_t face_u1 = face_u0 + 1;
        vpsi_cx[index] = real{0.5} * (vpsi_u[face_u0] + vpsi_u[face_u1]);
        vd_cx[index] = real{0.5} * (darcy_u[face_u0] + darcy_u[face_u1]);

        const std::size_t face_v0 = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * y_stride +
                                    static_cast<std::size_t>(k) * face_v_k_stride;
        const std::size_t face_v1 = face_v0 + y_stride;
        vpsi_cy[index] = real{0.5} * (vpsi_v[face_v0] + vpsi_v[face_v1]);
        vd_cy[index] = real{0.5} * (darcy_v[face_v0] + darcy_v[face_v1]);

        const std::size_t face_w0 = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * y_stride +
                                    static_cast<std::size_t>(k) * face_w_k_stride;
        const std::size_t face_w1 = face_w0 + face_w_k_stride;
        vpsi_cz[index] = real{0.5} * (vpsi_w[face_w0] + vpsi_w[face_w1]);
        vd_cz[index] = real{0.5} * (darcy_w[face_w0] + darcy_w[face_w1]);
    }
}

// Cell-centered metrics: magnitude error, robust angle, Darcy invariance,
// reconstructed-flow MAC divergence, |c|, and Darcy-speed-split degeneracy.
__global__ void cell_metrics_kernel(const real* vpsi_cx, const real* vpsi_cy, const real* vpsi_cz,
                                    const real* vd_cx, const real* vd_cy, const real* vd_cz,
                                    const real* g1x, const real* g1y, const real* g1z,
                                    const real* g2x, const real* g2y, const real* g2z,
                                    const real* vpsi_u, const real* vpsi_v, const real* vpsi_w,
                                    real* magnitude_field, real* theta_field, real* dot1_field,
                                    real* dot2_field, real* divergence_field, real* abs_c_field,
                                    real* scalars, unsigned long long* counters, int nx, int ny,
                                    int nz, real dx, real dy, real dz, std::size_t n,
                                    DiagnosticsKernelConfig config) {
    const std::size_t start = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t stride = static_cast<std::size_t>(gridDim.x) * blockDim.x;
    const std::size_t y_stride = static_cast<std::size_t>(nx);
    const std::size_t z_stride = y_stride * static_cast<std::size_t>(ny);
    const std::size_t face_u_j_stride = static_cast<std::size_t>(nx + 1);
    const std::size_t face_u_k_stride = face_u_j_stride * static_cast<std::size_t>(ny);
    const std::size_t face_v_k_stride = y_stride * static_cast<std::size_t>(ny + 1);
    const std::size_t face_w_k_stride = y_stride * static_cast<std::size_t>(ny);

    const real nu = scalars[kScalarNu];
    const real nv = scalars[kScalarNv];
    const real nw = scalars[kScalarNw];
    const real n_real = static_cast<real>(n);
    const real v_rms = sqrt((nu * nu + nv * nv + nw * nw) / n_real);
    const real angle_threshold = config.angle_exclusion_rel * v_rms;
    const real low_speed_threshold = config.low_speed_rel * v_rms;

    real local_max_m = real{0};
    real local_max_theta = real{0};
    real local_max_div = real{0};
    real local_min_c = static_cast<real>(INFINITY);
    real local_max_c = real{0};
    unsigned long long local_included = 0;
    unsigned long long local_excluded = 0;
    unsigned long long local_total[kMaxDegeneracyThresholds] = {};
    unsigned long long local_low_speed[kMaxDegeneracyThresholds] = {};

    for (std::size_t index = start; index < n; index += stride) {
        const int i = static_cast<int>(index % y_stride);
        const int j = static_cast<int>((index / y_stride) % static_cast<std::size_t>(ny));
        const int k = static_cast<int>(index / z_stride);

        const real px = vpsi_cx[index];
        const real py = vpsi_cy[index];
        const real pz = vpsi_cz[index];
        const real qx = vd_cx[index];
        const real qy = vd_cy[index];
        const real qz = vd_cz[index];

        const real vpsi_mag = sqrt(px * px + py * py + pz * pz);
        const real vd_mag = sqrt(qx * qx + qy * qy + qz * qz);
        const real m = vpsi_mag - vd_mag;
        magnitude_field[index] = m;
        const real abs_m = fabs(m);
        if (abs_m > local_max_m) {
            local_max_m = abs_m;
        }

        const real g1xv = g1x[index];
        const real g1yv = g1y[index];
        const real g1zv = g1z[index];
        const real g2xv = g2x[index];
        const real g2yv = g2y[index];
        const real g2zv = g2z[index];

        const real dot1 = qx * g1xv + qy * g1yv + qz * g1zv;
        const real dot2 = qx * g2xv + qy * g2yv + qz * g2zv;
        dot1_field[index] = dot1;
        dot2_field[index] = dot2;

        const real cx = g1yv * g2zv - g1zv * g2yv;
        const real cy = g1zv * g2xv - g1xv * g2zv;
        const real cz = g1xv * g2yv - g1yv * g2xv;
        const real abs_c = sqrt(cx * cx + cy * cy + cz * cz);
        abs_c_field[index] = abs_c;
        if (abs_c < local_min_c) {
            local_min_c = abs_c;
        }
        if (abs_c > local_max_c) {
            local_max_c = abs_c;
        }

        const std::size_t face_u0 = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * face_u_j_stride +
                                    static_cast<std::size_t>(k) * face_u_k_stride;
        const std::size_t face_u1 = face_u0 + 1;
        const std::size_t face_v0 = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * y_stride +
                                    static_cast<std::size_t>(k) * face_v_k_stride;
        const std::size_t face_v1 = face_v0 + y_stride;
        const std::size_t face_w0 = static_cast<std::size_t>(i) + static_cast<std::size_t>(j) * y_stride +
                                    static_cast<std::size_t>(k) * face_w_k_stride;
        const std::size_t face_w1 = face_w0 + face_w_k_stride;
        const real div = (vpsi_u[face_u1] - vpsi_u[face_u0]) / dx +
                         (vpsi_v[face_v1] - vpsi_v[face_v0]) / dy +
                         (vpsi_w[face_w1] - vpsi_w[face_w0]) / dz;
        divergence_field[index] = div;
        const real abs_div = fabs(div);
        if (abs_div > local_max_div) {
            local_max_div = abs_div;
        }

        real theta;
        if (vpsi_mag >= angle_threshold && vd_mag >= angle_threshold) {
            const real dotv = px * qx + py * qy + pz * qz;
            const real crossx = py * qz - pz * qy;
            const real crossy = pz * qx - px * qz;
            const real crossz = px * qy - py * qx;
            const real cross_mag = sqrt(crossx * crossx + crossy * crossy + crossz * crossz);
            theta = atan2(cross_mag, dotv);
            if (theta > local_max_theta) {
                local_max_theta = theta;
            }
            ++local_included;
        } else {
            theta = real{0};
            ++local_excluded;
        }
        theta_field[index] = theta;

        for (int t = 0; t < config.num_degeneracy_thresholds; ++t) {
            const real tau_threshold = config.degeneracy_thresholds[t] * v_rms;
            if (abs_c < tau_threshold) {
                ++local_total[t];
                if (vd_mag < low_speed_threshold) {
                    ++local_low_speed[t];
                }
            }
        }
    }

    if (local_max_m > real{0}) {
        atomic_max_double(&scalars[kScalarLinfM], local_max_m);
    }
    if (local_max_theta > real{0}) {
        atomic_max_double(&scalars[kScalarMaxTheta], local_max_theta);
    }
    if (local_max_div > real{0}) {
        atomic_max_double(&scalars[kScalarLinfDiv], local_max_div);
    }
    if (local_min_c < static_cast<real>(INFINITY)) {
        atomic_min_double(&scalars[kScalarMinC], local_min_c);
    }
    if (local_max_c > real{0}) {
        atomic_max_double(&scalars[kScalarMaxC], local_max_c);
    }
    if (local_included != 0) {
        atomicAdd(&counters[kCounterIncluded], local_included);
    }
    if (local_excluded != 0) {
        atomicAdd(&counters[kCounterExcluded], local_excluded);
    }
    for (int t = 0; t < config.num_degeneracy_thresholds; ++t) {
        if (local_total[t] != 0) {
            atomicAdd(&counters[kCounterLeading + static_cast<std::size_t>(2 * t)], local_total[t]);
        }
        if (local_low_speed[t] != 0) {
            atomicAdd(&counters[kCounterLeading + static_cast<std::size_t>(2 * t) + 1],
                      local_low_speed[t]);
        }
    }
}

std::size_t require_valid_grid(const Grid3D& grid) {
    if (grid.nx < 2 || grid.ny < 2 || grid.nz < 2) {
        throw std::invalid_argument(
            "Streamfunction physical diagnostics require every grid dimension to be at least 2");
    }
    if (!std::isfinite(grid.dx) || !std::isfinite(grid.dy) || !std::isfinite(grid.dz) ||
        grid.dx <= real{0} || grid.dy <= real{0} || grid.dz <= real{0}) {
        throw std::invalid_argument(
            "Streamfunction physical diagnostics require finite positive direction-specific "
            "spacing (independent spacings are supported; see Diagnostics.cuh)");
    }

    const auto max_elements = std::numeric_limits<std::size_t>::max() / sizeof(real);
    const auto nx = static_cast<std::size_t>(grid.nx);
    const auto ny = static_cast<std::size_t>(grid.ny);
    const auto nz = static_cast<std::size_t>(grid.nz);
    if (nx > max_elements / ny || nx * ny > max_elements / nz) {
        throw std::invalid_argument(
            "Streamfunction physical diagnostics require a grid whose storage size does not "
            "overflow");
    }
    return nx * ny * nz;
}

void require_valid_config(const PhysicalDiagnosticsConfig& config) {
    if (!std::isfinite(config.angle_exclusion_rel) || config.angle_exclusion_rel <= real{0}) {
        throw std::invalid_argument(
            "Streamfunction physical diagnostics require a finite, strictly positive "
            "angle_exclusion_rel");
    }
    if (!std::isfinite(config.low_speed_rel) || config.low_speed_rel <= real{0}) {
        throw std::invalid_argument(
            "Streamfunction physical diagnostics require a finite, strictly positive "
            "low_speed_rel");
    }
    if (config.num_degeneracy_thresholds < 0 ||
        config.num_degeneracy_thresholds > kMaxDegeneracyThresholds) {
        throw std::invalid_argument(
            "Streamfunction physical diagnostics require a degeneracy threshold count within "
            "bounds");
    }
    real previous = real{0};
    for (int t = 0; t < config.num_degeneracy_thresholds; ++t) {
        const real tau = config.degeneracy_thresholds[t];
        if (!std::isfinite(tau) || tau <= real{0}) {
            throw std::invalid_argument(
                "Streamfunction physical diagnostics require finite, strictly positive "
                "degeneracy thresholds");
        }
        if (t > 0 && tau <= previous) {
            throw std::invalid_argument(
                "Streamfunction physical diagnostics require strictly increasing degeneracy "
                "thresholds");
        }
        previous = tau;
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

void require_exact_nonoverlapping_spans(const Grid3D& grid,
                                        const PeriodicStreamfunctionFluctuations& fluctuations,
                                        const CompactMacVelocityConstView& darcy,
                                        const CompactMacVelocityView& v_psi, std::size_t n) {
    const std::size_t size_u = static_cast<std::size_t>(grid.nx + 1) *
                               static_cast<std::size_t>(grid.ny) * static_cast<std::size_t>(grid.nz);
    const std::size_t size_v = static_cast<std::size_t>(grid.nx) *
                               static_cast<std::size_t>(grid.ny + 1) * static_cast<std::size_t>(grid.nz);
    const std::size_t size_w = static_cast<std::size_t>(grid.nx) * static_cast<std::size_t>(grid.ny) *
                               static_cast<std::size_t>(grid.nz + 1);

    if (fluctuations.u1.size() != n || fluctuations.u2.size() != n ||
        fluctuations.u1.data() == nullptr || fluctuations.u2.data() == nullptr) {
        throw std::invalid_argument(
            "Streamfunction physical diagnostics require non-null fluctuation spans of exact "
            "grid size");
    }
    if (darcy.u.size() != size_u || darcy.v.size() != size_v || darcy.w.size() != size_w ||
        darcy.u.data() == nullptr || darcy.v.data() == nullptr || darcy.w.data() == nullptr) {
        throw std::invalid_argument(
            "Streamfunction physical diagnostics require non-null Darcy CompactMAC spans of "
            "exact size");
    }
    if (v_psi.u.size() != size_u || v_psi.v.size() != size_v || v_psi.w.size() != size_w ||
        v_psi.u.data() == nullptr || v_psi.v.data() == nullptr || v_psi.w.data() == nullptr) {
        throw std::invalid_argument(
            "Streamfunction physical diagnostics require non-null v_psi CompactMAC outputs of "
            "exact size");
    }

    const DeviceSpan<const real> darcy_components[] = {darcy.u, darcy.v, darcy.w};
    for (std::size_t first = 0; first < 3; ++first) {
        for (std::size_t second = first + 1; second < 3; ++second) {
            if (overlaps(darcy_components[first], darcy_components[second])) {
                throw std::invalid_argument(
                    "Streamfunction physical diagnostics require Darcy u, v, w components not "
                    "to alias each other");
            }
        }
    }

    const DeviceSpan<const real> v_psi_components[] = {
        DeviceSpan<const real>(v_psi.u), DeviceSpan<const real>(v_psi.v),
        DeviceSpan<const real>(v_psi.w)};
    for (std::size_t first = 0; first < 3; ++first) {
        for (std::size_t second = first + 1; second < 3; ++second) {
            if (overlaps(v_psi_components[first], v_psi_components[second])) {
                throw std::invalid_argument(
                    "Streamfunction physical diagnostics require v_psi u, v, w outputs not to "
                    "alias each other");
            }
        }
    }

    const DeviceSpan<const real> other_inputs[] = {
        darcy.u, darcy.v, darcy.w, DeviceSpan<const real>(fluctuations.u1),
        DeviceSpan<const real>(fluctuations.u2)};
    for (const auto& output_span : v_psi_components) {
        for (const auto& other : other_inputs) {
            if (overlaps(output_span, other)) {
                throw std::invalid_argument(
                    "Streamfunction physical diagnostics v_psi outputs must not overlap Darcy "
                    "inputs or the fluctuation inputs");
            }
        }
    }
    // fluctuations.u1 and fluctuations.u2 are read-only and may overlap.
}

} // namespace

void StreamfunctionDiagnosticsWorkspace::prepare(const Grid3D& grid) {
    const std::size_t n = grid.num_cells();

    g1x_.resize(n);
    g1y_.resize(n);
    g1z_.resize(n);
    g2x_.resize(n);
    g2y_.resize(n);
    g2z_.resize(n);

    unique_vpsi_u_.resize(n);
    unique_vpsi_v_.resize(n);
    unique_vpsi_w_.resize(n);
    unique_vd_u_.resize(n);
    unique_vd_v_.resize(n);
    unique_vd_w_.resize(n);
    diff_u_.resize(n);
    diff_v_.resize(n);
    diff_w_.resize(n);

    vpsi_cx_.resize(n);
    vpsi_cy_.resize(n);
    vpsi_cz_.resize(n);
    vd_cx_.resize(n);
    vd_cy_.resize(n);
    vd_cz_.resize(n);

    magnitude_field_.resize(n);
    theta_field_.resize(n);
    dot1_field_.resize(n);
    dot2_field_.resize(n);
    divergence_field_.resize(n);
    abs_c_field_.resize(n);

    blas::prepare_sum_workspace(n, reduction_workspace_);

    scalars_.resize(kNumScalars);
    counters_.resize(kNumCounters);

    nx_ = grid.nx;
    ny_ = grid.ny;
    nz_ = grid.nz;
    n_ = n;
    has_last_results_ = false;
}

bool StreamfunctionDiagnosticsWorkspace::prepared_for(const Grid3D& grid) const noexcept {
    const std::size_t n = grid.num_cells();
    return nx_ == grid.nx && ny_ == grid.ny && nz_ == grid.nz && n_ == n && g1x_.size() == n &&
           g1y_.size() == n && g1z_.size() == n && g2x_.size() == n && g2y_.size() == n &&
           g2z_.size() == n && unique_vpsi_u_.size() == n && unique_vpsi_v_.size() == n &&
           unique_vpsi_w_.size() == n && unique_vd_u_.size() == n && unique_vd_v_.size() == n &&
           unique_vd_w_.size() == n && diff_u_.size() == n && diff_v_.size() == n &&
           diff_w_.size() == n && vpsi_cx_.size() == n && vpsi_cy_.size() == n &&
           vpsi_cz_.size() == n && vd_cx_.size() == n && vd_cy_.size() == n &&
           vd_cz_.size() == n && magnitude_field_.size() == n && theta_field_.size() == n &&
           dot1_field_.size() == n && dot2_field_.size() == n && divergence_field_.size() == n &&
           abs_c_field_.size() == n && blas::sum_workspace_ready(n, reduction_workspace_) &&
           scalars_.size() == kNumScalars && counters_.size() == kNumCounters;
}

void enqueue_streamfunction_physical_diagnostics(
    CudaContext& ctx, const Grid3D& grid, const PeriodicStreamfunctionFluctuations& fluctuations,
    const AffineGauge& gauge, const CompactMacVelocityConstView& darcy,
    const PhysicalDiagnosticsConfig& config, const CompactMacVelocityView& v_psi,
    StreamfunctionDiagnosticsWorkspace& workspace) {
    const std::size_t n = require_valid_grid(grid);
    require_valid_config(config);
    require_exact_nonoverlapping_spans(grid, fluctuations, darcy, v_psi, n);
    if (!workspace.prepared_for(grid)) {
        throw std::logic_error(
            "Streamfunction physical diagnostics require a workspace prepared for the exact "
            "grid");
    }

    // 1) SF-07 total streamfunction gradients.
    const TotalStreamfunctionGradientOutput grad_output{
        workspace.g1x_.span(), workspace.g1y_.span(), workspace.g1z_.span(),
        workspace.g2x_.span(), workspace.g2y_.span(), workspace.g2z_.span()};
    enqueue_total_streamfunction_gradients(ctx, grid, fluctuations, gauge, grad_output);

    // Zero the fixed-layout scalar buffer and the counter buffer, then set
    // the |c| minimum accumulator to +infinity via a dedicated kernel.
    MACROFLOW3D_CUDA_CHECK(
        cudaMemsetAsync(workspace.scalars_.data(), 0, kNumScalars * sizeof(real), ctx.cuda_stream()));
    MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(workspace.counters_.data(), 0,
                                           kNumCounters * sizeof(unsigned long long),
                                           ctx.cuda_stream()));
    init_min_c_kernel<<<1, 1, 0, ctx.cuda_stream()>>>(workspace.scalars_.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    const std::size_t requested_blocks = (n + kBlockSize - 1) / kBlockSize;
    const int blocks = static_cast<int>(
        requested_blocks < static_cast<std::size_t>(kMaxBlocks) ? requested_blocks : kMaxBlocks);

    // 2) Face reconstruction + unique-face scratch.
    reconstruct_and_face_diagnostics_kernel<<<blocks, kBlockSize, 0, ctx.cuda_stream()>>>(
        workspace.g1x_.data(), workspace.g1y_.data(), workspace.g1z_.data(), workspace.g2x_.data(),
        workspace.g2y_.data(), workspace.g2z_.data(), darcy.u.data(), darcy.v.data(),
        darcy.w.data(), v_psi.u.data(), v_psi.v.data(), v_psi.w.data(), workspace.unique_vpsi_u_.data(),
        workspace.unique_vpsi_v_.data(), workspace.unique_vpsi_w_.data(), workspace.unique_vd_u_.data(),
        workspace.unique_vd_v_.data(), workspace.unique_vd_w_.data(), workspace.diff_u_.data(),
        workspace.diff_v_.data(), workspace.diff_w_.data(), workspace.scalars_.data(), grid.nx,
        grid.ny, grid.nz);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    // 3) Face-to-center averaging (v_psi, Darcy).
    face_to_center_kernel<<<blocks, kBlockSize, 0, ctx.cuda_stream()>>>(
        v_psi.u.data(), v_psi.v.data(), v_psi.w.data(), darcy.u.data(), darcy.v.data(),
        darcy.w.data(), workspace.vpsi_cx_.data(), workspace.vpsi_cy_.data(), workspace.vpsi_cz_.data(),
        workspace.vd_cx_.data(), workspace.vd_cy_.data(), workspace.vd_cz_.data(), grid.nx, grid.ny,
        grid.nz);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    // 4) Darcy speed component L2 norms (stream-ordered; consumed in-kernel
    //    by cell_metrics_kernel to compute v_rms exactly, never on host
    //    before that kernel runs).
    blas::nrm2_device(ctx, DeviceSpan<const real>(workspace.vd_cx_.span()),
                      DeviceSpan<real>(workspace.scalars_.data() + kScalarNu, 1),
                      workspace.reduction_workspace_);
    blas::nrm2_device(ctx, DeviceSpan<const real>(workspace.vd_cy_.span()),
                      DeviceSpan<real>(workspace.scalars_.data() + kScalarNv, 1),
                      workspace.reduction_workspace_);
    blas::nrm2_device(ctx, DeviceSpan<const real>(workspace.vd_cz_.span()),
                      DeviceSpan<real>(workspace.scalars_.data() + kScalarNw, 1),
                      workspace.reduction_workspace_);

    // 5) Cell-centered metrics.
    DiagnosticsKernelConfig kernel_config{};
    kernel_config.angle_exclusion_rel = config.angle_exclusion_rel;
    kernel_config.low_speed_rel = config.low_speed_rel;
    kernel_config.num_degeneracy_thresholds = config.num_degeneracy_thresholds;
    for (int t = 0; t < config.num_degeneracy_thresholds; ++t) {
        kernel_config.degeneracy_thresholds[t] = config.degeneracy_thresholds[t];
    }
    cell_metrics_kernel<<<blocks, kBlockSize, 0, ctx.cuda_stream()>>>(
        workspace.vpsi_cx_.data(), workspace.vpsi_cy_.data(), workspace.vpsi_cz_.data(),
        workspace.vd_cx_.data(), workspace.vd_cy_.data(), workspace.vd_cz_.data(), workspace.g1x_.data(),
        workspace.g1y_.data(), workspace.g1z_.data(), workspace.g2x_.data(), workspace.g2y_.data(),
        workspace.g2z_.data(), v_psi.u.data(), v_psi.v.data(), v_psi.w.data(),
        workspace.magnitude_field_.data(), workspace.theta_field_.data(), workspace.dot1_field_.data(),
        workspace.dot2_field_.data(), workspace.divergence_field_.data(), workspace.abs_c_field_.data(),
        workspace.scalars_.data(), workspace.counters_.data(), grid.nx, grid.ny, grid.nz, grid.dx,
        grid.dy, grid.dz, n, kernel_config);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    // 6) Reductions.
    blas::nrm2_device(ctx, DeviceSpan<const real>(workspace.diff_u_.span()),
                      DeviceSpan<real>(workspace.scalars_.data() + kScalarL2DiffU, 1),
                      workspace.reduction_workspace_);
    blas::nrm2_device(ctx, DeviceSpan<const real>(workspace.diff_v_.span()),
                      DeviceSpan<real>(workspace.scalars_.data() + kScalarL2DiffV, 1),
                      workspace.reduction_workspace_);
    blas::nrm2_device(ctx, DeviceSpan<const real>(workspace.diff_w_.span()),
                      DeviceSpan<real>(workspace.scalars_.data() + kScalarL2DiffW, 1),
                      workspace.reduction_workspace_);

    const DeviceSpan<const real> unique_vpsi[3] = {DeviceSpan<const real>(workspace.unique_vpsi_u_.span()),
                                                   DeviceSpan<const real>(workspace.unique_vpsi_v_.span()),
                                                   DeviceSpan<const real>(workspace.unique_vpsi_w_.span())};
    const DeviceSpan<const real> unique_vd[3] = {DeviceSpan<const real>(workspace.unique_vd_u_.span()),
                                                 DeviceSpan<const real>(workspace.unique_vd_v_.span()),
                                                 DeviceSpan<const real>(workspace.unique_vd_w_.span())};
    const std::size_t corr_base[3] = {kScalarCorrUSx, kScalarCorrVSx, kScalarCorrWSx};
    for (int c = 0; c < 3; ++c) {
        const std::size_t base = corr_base[c];
        blas::sum_device(ctx, unique_vpsi[c], DeviceSpan<real>(workspace.scalars_.data() + base, 1),
                         workspace.reduction_workspace_);
        blas::sum_device(ctx, unique_vd[c], DeviceSpan<real>(workspace.scalars_.data() + base + 1, 1),
                         workspace.reduction_workspace_);
        blas::dot_device(ctx, unique_vpsi[c], unique_vpsi[c],
                         DeviceSpan<real>(workspace.scalars_.data() + base + 2, 1),
                         workspace.reduction_workspace_);
        blas::dot_device(ctx, unique_vd[c], unique_vd[c],
                         DeviceSpan<real>(workspace.scalars_.data() + base + 3, 1),
                         workspace.reduction_workspace_);
        blas::dot_device(ctx, unique_vpsi[c], unique_vd[c],
                         DeviceSpan<real>(workspace.scalars_.data() + base + 4, 1),
                         workspace.reduction_workspace_);
    }

    blas::nrm2_device(ctx, DeviceSpan<const real>(workspace.magnitude_field_.span()),
                      DeviceSpan<real>(workspace.scalars_.data() + kScalarL2M, 1),
                      workspace.reduction_workspace_);
    blas::nrm2_device(ctx, DeviceSpan<const real>(workspace.theta_field_.span()),
                      DeviceSpan<real>(workspace.scalars_.data() + kScalarL2Theta, 1),
                      workspace.reduction_workspace_);
    blas::nrm2_device(ctx, DeviceSpan<const real>(workspace.dot1_field_.span()),
                      DeviceSpan<real>(workspace.scalars_.data() + kScalarL2Dot1, 1),
                      workspace.reduction_workspace_);
    blas::nrm2_device(ctx, DeviceSpan<const real>(workspace.dot2_field_.span()),
                      DeviceSpan<real>(workspace.scalars_.data() + kScalarL2Dot2, 1),
                      workspace.reduction_workspace_);
    blas::nrm2_device(ctx, DeviceSpan<const real>(workspace.divergence_field_.span()),
                      DeviceSpan<real>(workspace.scalars_.data() + kScalarL2Div, 1),
                      workspace.reduction_workspace_);
    blas::sum_device(ctx, DeviceSpan<const real>(workspace.abs_c_field_.span()),
                     DeviceSpan<real>(workspace.scalars_.data() + kScalarSumC, 1),
                     workspace.reduction_workspace_);

    blas::nrm2_device(ctx, DeviceSpan<const real>(workspace.g1x_.span()),
                      DeviceSpan<real>(workspace.scalars_.data() + kScalarL2G1x, 1),
                      workspace.reduction_workspace_);
    blas::nrm2_device(ctx, DeviceSpan<const real>(workspace.g1y_.span()),
                      DeviceSpan<real>(workspace.scalars_.data() + kScalarL2G1y, 1),
                      workspace.reduction_workspace_);
    blas::nrm2_device(ctx, DeviceSpan<const real>(workspace.g1z_.span()),
                      DeviceSpan<real>(workspace.scalars_.data() + kScalarL2G1z, 1),
                      workspace.reduction_workspace_);
    blas::nrm2_device(ctx, DeviceSpan<const real>(workspace.g2x_.span()),
                      DeviceSpan<real>(workspace.scalars_.data() + kScalarL2G2x, 1),
                      workspace.reduction_workspace_);
    blas::nrm2_device(ctx, DeviceSpan<const real>(workspace.g2y_.span()),
                      DeviceSpan<real>(workspace.scalars_.data() + kScalarL2G2y, 1),
                      workspace.reduction_workspace_);
    blas::nrm2_device(ctx, DeviceSpan<const real>(workspace.g2z_.span()),
                      DeviceSpan<real>(workspace.scalars_.data() + kScalarL2G2z, 1),
                      workspace.reduction_workspace_);

    workspace.has_last_results_ = true;
}

PhysicalDiagnosticsReport synchronize_streamfunction_physical_diagnostics_report(
    CudaContext& ctx, const Grid3D& grid, const PhysicalDiagnosticsConfig& config,
    StreamfunctionDiagnosticsWorkspace& workspace) {
    const std::size_t n = require_valid_grid(grid);
    require_valid_config(config);
    if (!workspace.prepared_for(grid)) {
        throw std::logic_error(
            "Streamfunction physical diagnostics report requires a workspace prepared for the "
            "exact grid");
    }
    if (!workspace.has_last_results_) {
        throw std::logic_error(
            "Streamfunction physical diagnostics report requires a preceding "
            "enqueue_streamfunction_physical_diagnostics call");
    }

    real scalars_host[kNumScalars] = {};
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(scalars_host, workspace.scalars_.data(),
                                           kNumScalars * sizeof(real), cudaMemcpyDeviceToHost,
                                           ctx.cuda_stream()));

    unsigned long long counters_host[kNumCounters] = {};
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(counters_host, workspace.counters_.data(),
                                           kNumCounters * sizeof(unsigned long long),
                                           cudaMemcpyDeviceToHost, ctx.cuda_stream()));

    ctx.synchronize();

    PhysicalDiagnosticsReport report;
    const real n_real = static_cast<real>(n);
    const real sqrt_n = std::sqrt(n_real);

    const real nu = scalars_host[kScalarNu];
    const real nv = scalars_host[kScalarNv];
    const real nw = scalars_host[kScalarNw];
    const real v_d_rms = std::sqrt((nu * nu + nv * nv + nw * nw) / n_real);

    const real raw_l2_u = scalars_host[kScalarL2DiffU];
    const real raw_l2_v = scalars_host[kScalarL2DiffV];
    const real raw_l2_w = scalars_host[kScalarL2DiffW];
    report.rms_u = raw_l2_u / sqrt_n;
    report.rms_v = raw_l2_v / sqrt_n;
    report.rms_w = raw_l2_w / sqrt_n;
    report.linf_u = scalars_host[kScalarLinfDiffU];
    report.linf_v = scalars_host[kScalarLinfDiffV];
    report.linf_w = scalars_host[kScalarLinfDiffW];
    report.rms_u_rel = report.rms_u / v_d_rms;
    report.rms_v_rel = report.rms_v / v_d_rms;
    report.rms_w_rel = report.rms_w / v_d_rms;
    report.linf_u_rel = report.linf_u / v_d_rms;
    report.linf_v_rel = report.linf_v / v_d_rms;
    report.linf_w_rel = report.linf_w / v_d_rms;
    report.e_v = std::sqrt((raw_l2_u * raw_l2_u + raw_l2_v * raw_l2_v + raw_l2_w * raw_l2_w) /
                           (real{3} * n_real)) /
                v_d_rms;

    const std::size_t corr_base[3] = {kScalarCorrUSx, kScalarCorrVSx, kScalarCorrWSx};
    real* corr_out[3] = {&report.corr_u, &report.corr_v, &report.corr_w};
    for (int c = 0; c < 3; ++c) {
        const std::size_t base = corr_base[c];
        const real sx = scalars_host[base];
        const real sy = scalars_host[base + 1];
        const real sxx = scalars_host[base + 2];
        const real syy = scalars_host[base + 3];
        const real sxy = scalars_host[base + 4];
        const real numerator = n_real * sxy - sx * sy;
        const real denominator =
            std::sqrt((n_real * sxx - sx * sx) * (n_real * syy - sy * sy));
        *corr_out[c] = numerator / denominator;
    }

    const real raw_l2_m = scalars_host[kScalarL2M];
    report.rms_magnitude = raw_l2_m / sqrt_n;
    report.linf_magnitude = scalars_host[kScalarLinfM];
    report.rms_magnitude_rel = report.rms_magnitude / v_d_rms;
    report.linf_magnitude_rel = report.linf_magnitude / v_d_rms;

    report.angle_included_count = counters_host[kCounterIncluded];
    report.angle_excluded_count = counters_host[kCounterExcluded];
    const real included_real = static_cast<real>(report.angle_included_count);
    report.rms_theta = scalars_host[kScalarL2Theta] / std::sqrt(included_real);
    report.max_theta = scalars_host[kScalarMaxTheta];

    const real raw_l2_dot1 = scalars_host[kScalarL2Dot1];
    const real raw_l2_dot2 = scalars_host[kScalarL2Dot2];
    report.invariance_raw_rms_psi1 = raw_l2_dot1 / sqrt_n;
    report.invariance_raw_rms_psi2 = raw_l2_dot2 / sqrt_n;
    const real l2_g1x = scalars_host[kScalarL2G1x];
    const real l2_g1y = scalars_host[kScalarL2G1y];
    const real l2_g1z = scalars_host[kScalarL2G1z];
    const real l2_g2x = scalars_host[kScalarL2G2x];
    const real l2_g2y = scalars_host[kScalarL2G2y];
    const real l2_g2z = scalars_host[kScalarL2G2z];
    report.invariance_grad_rms_psi1 =
        std::sqrt((l2_g1x * l2_g1x + l2_g1y * l2_g1y + l2_g1z * l2_g1z) / n_real);
    report.invariance_grad_rms_psi2 =
        std::sqrt((l2_g2x * l2_g2x + l2_g2y * l2_g2y + l2_g2z * l2_g2z) / n_real);
    report.invariance_e_psi1 =
        report.invariance_raw_rms_psi1 / (v_d_rms * report.invariance_grad_rms_psi1);
    report.invariance_e_psi2 =
        report.invariance_raw_rms_psi2 / (v_d_rms * report.invariance_grad_rms_psi2);

    const real l_ref = std::cbrt(grid.Lx() * grid.Ly() * grid.Lz());
    report.rms_div = scalars_host[kScalarL2Div] / sqrt_n;
    report.linf_div = scalars_host[kScalarLinfDiv];
    report.e_div = l_ref * report.rms_div / v_d_rms;

    report.c_min = scalars_host[kScalarMinC];
    report.c_max = scalars_host[kScalarMaxC];
    report.c_mean = scalars_host[kScalarSumC] / n_real;

    report.num_degeneracy_thresholds = config.num_degeneracy_thresholds;
    for (int t = 0; t < config.num_degeneracy_thresholds; ++t) {
        report.degeneracy_thresholds[t] = config.degeneracy_thresholds[t];
        const unsigned long long total_t = counters_host[kCounterLeading + static_cast<std::size_t>(2 * t)];
        const unsigned long long low_speed_t =
            counters_host[kCounterLeading + static_cast<std::size_t>(2 * t) + 1];
        report.degeneracy_total[t] = total_t;
        report.degeneracy_low_speed[t] = low_speed_t;
        report.degeneracy_unexplained[t] = total_t - low_speed_t;
    }

    report.v_d_rms = v_d_rms;
    report.L_ref = l_ref;
    report.angle_exclusion_rel = config.angle_exclusion_rel;
    report.low_speed_rel = config.low_speed_rel;
    report.n = n;

    return report;
}

} // namespace streamfunctions
} // namespace macroflow3d
