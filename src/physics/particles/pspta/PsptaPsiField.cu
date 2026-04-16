/**
 * @file PsptaPsiField.cu
 * @brief Level A ψ precompute for PSPTA transport.
 *
 * Two kernels implement the semi-Lagrangian marching:
 *   kernel_init_inlet      – fills plane i=0 with ψ1=y, ψ2=z
 *   kernel_advance_plane   – marches one plane forward (i → i+1)
 *
 * A third kernel provides diagnostics:
 *   kernel_vdotgradpsi     – computes v·∇ψ residual fields
 *
 * All intermediate arithmetic is in double; results are stored as float32.
 *
 * Index convention (x-fastest, consistent with Grid3D::idx):
 *   psi_buf[i + nx*(j + ny*k)]
 *
 * CompactMAC face indices (x-fastest in each component):
 *   U(i,j,k) = U_ptr[ i + (nx+1)*j + (nx+1)*ny*k ]   i ∈ [0,nx]
 *   V(i,j,k) = V_ptr[ i + nx*j     + nx*(ny+1)*k  ]   j ∈ [0,ny]
 *   W(i,j,k) = W_ptr[ i + nx*j     + nx*ny*k      ]   k ∈ [0,nz]
 */

#include "../../../runtime/cuda_check.cuh"
#include "PsptaPsiField.cuh"

#include <cstdio>
#include <cstring> // for host-side memcpy/memset
#include <math.h>  // floor, round (device-side)

namespace macroflow3d {
namespace physics {
namespace particles {
namespace pspta {

// ============================================================================
// Device helpers
// ============================================================================

/// Cell-centered 3D index (x-fastest): i + nx*(j + ny*k)
__device__ __forceinline__ int idx3(int i, int j, int k, int nx, int ny) {
    return i + nx * (j + ny * k);
}

/// CompactMAC U-face index: i + (nx+1)*j + (nx+1)*ny*k
__device__ __forceinline__ int idxU(int i, int j, int k, int nx, int ny) {
    return i + (nx + 1) * j + (nx + 1) * ny * k;
}

/// CompactMAC V-face index: i + nx*j + nx*(ny+1)*k
__device__ __forceinline__ int idxV(int i, int j, int k, int nx, int ny_p1, int nx_) {
    // ny_p1 = ny+1
    return i + nx_ * j + nx_ * ny_p1 * k;
}

/// CompactMAC W-face index: i + nx*j + nx*ny*k
__device__ __forceinline__ int idxW(int i, int j, int k, int nx_, int ny_) {
    return i + nx_ * j + nx_ * ny_ * k;
}

/// Periodic wrap: bring x into [0, L)
__device__ __forceinline__ double wrap_periodic(double x, double L) {
    return x - floor(x / L) * L;
}

__device__ __forceinline__ double clamp_sym(double v, double lim) {
    if (v > lim)
        return lim;
    if (v < -lim)
        return -lim;
    return v;
}

/// Periodic modulo for integer index: ((n % N) + N) % N
__device__ __forceinline__ int imod(int n, int N) {
    return ((n % N) + N) % N;
}

// ============================================================================
// Kernel 1: init_inlet — fill plane i=0
// ============================================================================

/**
 * @brief Initialize ψ1, ψ2 on the inlet plane (i=0).
 *
 * ψ1(0,j,k) = (j + 0.5) * dy   (y coordinate of cell center)
 * ψ2(0,j,k) = (k + 0.5) * dz   (z coordinate of cell center)
 *
 * Thread layout: 2D over (j ∈ [0,ny), k ∈ [0,nz))
 */
__global__ void kernel_init_inlet(float* __restrict__ psi1, float* __restrict__ psi2, int nx,
                                  int ny, int nz, double dy, double dz) {
    const int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int k = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (j >= ny || k >= nz)
        return;

    const int idx = idx3(0, j, k, nx, ny); // i = 0

    // Cell-center coordinates (double math, cast to float for storage)
    const double y = (j + 0.5) * dy;
    const double z = (k + 0.5) * dz;

    psi1[idx] = static_cast<float>(y);
    psi2[idx] = static_cast<float>(z);
}

// ============================================================================
// Kernel 2: advance_plane — march from plane i to plane i+1
// ============================================================================

/**
 * @brief Advance ψ from plane i to plane i+1 via semi-Lagrangian backtrace.
 *
 * For each destination cell (i+1, j, k):
 *   1. Build x-midpoint cell-center velocity from CompactMAC faces.
 *   2. Clamp vx_avg to eps_vx; count clamp events via atomicAdd.
 *   3. Backtrace: y* = y - (vy_avg/vx_eff)*dx,  z* = z - (vz_avg/vx_eff)*dx
 *   4. Wrap y*, z* into [0, Ly) × [0, Lz).
 *   5. Bilinear sample ψ on plane i with periodic lifting.
 *   6. Wrap result back to [0, L) and store as float.
 *
 * @par Periodic lifting detail
 * All four corner values of the bilinear stencil are adjusted so that
 * p10, p01, p11 are the nearest representatives to p00 modulo L.
 * This prevents averaging across the periodic seam (e.g., ~0 and ~L).
 *
 * @param psi1     Full ψ1 buffer (read plane i, write plane i+1).
 * @param psi2     Full ψ2 buffer (read plane i, write plane i+1).
 * @param U        CompactMAC U-faces (const real*).
 * @param V        CompactMAC V-faces (const real*).
 * @param W        CompactMAC W-faces (const real*).
 * @param i_plane  Source plane index (i). Destination is i+1.
 * @param vx_clamped_cnt  Device counter; incremented when vx_avg <= eps_vx.
 *
 * Thread layout: 2D over (j ∈ [0,ny), k ∈ [0,nz))
 */
__global__ void kernel_advance_plane(float* __restrict__ psi1, float* __restrict__ psi2,
                                     const real* __restrict__ U, const real* __restrict__ V,
                                     const real* __restrict__ W, int nx, int ny, int nz, double dx,
                                     double dy, double dz, double Ly, double Lz, double eps_vx,
                                     int i_plane, int* __restrict__ vx_clamped_cnt) {
    const int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int k = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (j >= ny || k >= nz)
        return;

    const int i = i_plane;
    const int i1 = i_plane + 1;

    // ── Cell-center velocity at (i, j, k) ───────────────────────────────────
    // U(ix,iy,iz) = U[ ix + (nx+1)*iy + (nx+1)*ny*iz ]
    const double vx_i = 0.5 * (static_cast<double>(U[idxU(i, j, k, nx, ny)]) +
                               static_cast<double>(U[idxU(i + 1, j, k, nx, ny)]));
    const double vy_i = 0.5 * (static_cast<double>(V[i + nx * j + nx * (ny + 1) * k]) +
                               static_cast<double>(V[i + nx * (j + 1) + nx * (ny + 1) * k]));
    const double vz_i = 0.5 * (static_cast<double>(W[i + nx * j + nx * ny * k]) +
                               static_cast<double>(W[i + nx * j + nx * ny * (k + 1)]));

    // ── Cell-center velocity at (i+1, j, k) ─────────────────────────────────
    // Valid for i_plane <= nx-2: U(i+2,j,k) accesses ix=i+2 <= nx (U has nx+1
    // faces).
    const double vx_i1 = 0.5 * (static_cast<double>(U[idxU(i1, j, k, nx, ny)]) +
                                static_cast<double>(U[idxU(i1 + 1, j, k, nx, ny)]));
    const double vy_i1 = 0.5 * (static_cast<double>(V[i1 + nx * j + nx * (ny + 1) * k]) +
                                static_cast<double>(V[i1 + nx * (j + 1) + nx * (ny + 1) * k]));
    const double vz_i1 = 0.5 * (static_cast<double>(W[i1 + nx * j + nx * ny * k]) +
                                static_cast<double>(W[i1 + nx * j + nx * ny * (k + 1)]));

    // ── x-midpoint average ───────────────────────────────────────────────────
    const double vx_avg = 0.5 * (vx_i + vx_i1);
    const double vy_avg = 0.5 * (vy_i + vy_i1);
    const double vz_avg = 0.5 * (vz_i + vz_i1);

    // ── Clamp vx ─────────────────────────────────────────────────────────────
    const bool clamped = (vx_avg <= eps_vx);
    if (clamped) {
        atomicAdd(vx_clamped_cnt, 1);
    }
    const double vx_eff = clamped ? eps_vx : vx_avg;

    // ── Backtrace to plane i ─────────────────────────────────────────────────
    // Destination cell-center coordinates on plane i+1
    const double y_dest = (j + 0.5) * dy;
    const double z_dest = (k + 0.5) * dz;

    const double uy = vy_avg / vx_eff;
    const double uz = vz_avg / vx_eff;

    double y_star = y_dest - uy * dx;
    double z_star = z_dest - uz * dx;

    // ── Periodic wrap into [0, Ly) × [0, Lz) ─────────────────────────────────
    y_star = wrap_periodic(y_star, Ly);
    z_star = wrap_periodic(z_star, Lz);

    // ── Bilinear sample on plane i WITH periodic lifting ─────────────────────
    // Fractional cell-center indices
    const double jf = y_star / dy - 0.5;
    const double kf = z_star / dz - 0.5;

    const int j0_raw = static_cast<int>(floor(jf));
    const int k0_raw = static_cast<int>(floor(kf));
    const double t = jf - j0_raw; // ∈ [0, 1)
    const double s = kf - k0_raw; // ∈ [0, 1)

    // Periodic index wrap for the four stencil corners
    const int j0 = imod(j0_raw, ny);
    const int j1 = imod(j0_raw + 1, ny);
    const int k0 = imod(k0_raw, nz);
    const int k1 = imod(k0_raw + 1, nz);

    // Fetch ψ1 corners from plane i (double precision intermediates)
    double p1_00 = static_cast<double>(psi1[idx3(i, j0, k0, nx, ny)]);
    double p1_10 = static_cast<double>(psi1[idx3(i, j1, k0, nx, ny)]);
    double p1_01 = static_cast<double>(psi1[idx3(i, j0, k1, nx, ny)]);
    double p1_11 = static_cast<double>(psi1[idx3(i, j1, k1, nx, ny)]);

    // Periodic lifting for ψ1 (period = Ly):
    // Shift p10, p01, p11 to be the closest representative to p00 modulo Ly.
    {
        const double r = p1_00;
        p1_10 += Ly * round((r - p1_10) / Ly);
        p1_01 += Ly * round((r - p1_01) / Ly);
        p1_11 += Ly * round((r - p1_11) / Ly);
    }
    double p1 = (1.0 - s) * ((1.0 - t) * p1_00 + t * p1_10) + s * ((1.0 - t) * p1_01 + t * p1_11);
    // Wrap result back into [0, Ly)
    p1 = wrap_periodic(p1, Ly);

    // Fetch ψ2 corners from plane i
    double p2_00 = static_cast<double>(psi2[idx3(i, j0, k0, nx, ny)]);
    double p2_10 = static_cast<double>(psi2[idx3(i, j1, k0, nx, ny)]);
    double p2_01 = static_cast<double>(psi2[idx3(i, j0, k1, nx, ny)]);
    double p2_11 = static_cast<double>(psi2[idx3(i, j1, k1, nx, ny)]);

    // Periodic lifting for ψ2 (period = Lz):
    {
        const double r = p2_00;
        p2_10 += Lz * round((r - p2_10) / Lz);
        p2_01 += Lz * round((r - p2_01) / Lz);
        p2_11 += Lz * round((r - p2_11) / Lz);
    }
    double p2 = (1.0 - s) * ((1.0 - t) * p2_00 + t * p2_10) + s * ((1.0 - t) * p2_01 + t * p2_11);
    // Wrap result back into [0, Lz)
    p2 = wrap_periodic(p2, Lz);

    // ── Write to plane i+1 ───────────────────────────────────────────────────
    psi1[idx3(i1, j, k, nx, ny)] = static_cast<float>(p1);
    psi2[idx3(i1, j, k, nx, ny)] = static_cast<float>(p2);
}

// ============================================================================
// Kernel 3: v·∇ψ residuals (diagnostics)
// ============================================================================

/**
 * @brief Compute v·∇ψ residuals at all cell centers.
 *
 * v is reconstructed at cell centers from CompactMAC faces.
 * Gradients use central finite differences (periodic in y,z; one-sided at
 * x boundaries).
 *
 * @note  Near the y/z periodic seam, ∂ψ/∂y and ∂ψ/∂z are computed WITHOUT
 * periodic lifting, so residuals may be artificially large there.  This is
 * acceptable for a lightweight diagnostic.
 *
 * Thread layout: 3D over (i,j,k).
 */
__global__ void kernel_vdotgradpsi(const float* __restrict__ psi1, const float* __restrict__ psi2,
                                   const real* __restrict__ U, const real* __restrict__ V,
                                   const real* __restrict__ W, int nx, int ny, int nz, double dx,
                                   double dy, double dz, double Ly, double Lz,
                                   float* __restrict__ r1, float* __restrict__ r2) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int j = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
    const int k = static_cast<int>(blockIdx.z) * blockDim.z + threadIdx.z;
    if (i >= nx || j >= ny || k >= nz)
        return;

    // ── Cell-center velocity ─────────────────────────────────────────────────
    const double vx = 0.5 * (static_cast<double>(U[idxU(i, j, k, nx, ny)]) +
                             static_cast<double>(U[idxU(i + 1, j, k, nx, ny)]));
    const double vy = 0.5 * (static_cast<double>(V[i + nx * j + nx * (ny + 1) * k]) +
                             static_cast<double>(V[i + nx * (j + 1) + nx * (ny + 1) * k]));
    const double vz = 0.5 * (static_cast<double>(W[i + nx * j + nx * ny * k]) +
                             static_cast<double>(W[i + nx * j + nx * ny * (k + 1)]));

    // ── ∂ψ/∂x: central (one-sided at x boundaries) ──────────────────────────
    double dpsi1_dx, dpsi2_dx;
    if (i == 0) {
        dpsi1_dx = (static_cast<double>(psi1[idx3(1, j, k, nx, ny)]) -
                    static_cast<double>(psi1[idx3(0, j, k, nx, ny)])) /
                   dx;
        dpsi2_dx = (static_cast<double>(psi2[idx3(1, j, k, nx, ny)]) -
                    static_cast<double>(psi2[idx3(0, j, k, nx, ny)])) /
                   dx;
    } else if (i == nx - 1) {
        dpsi1_dx = (static_cast<double>(psi1[idx3(nx - 1, j, k, nx, ny)]) -
                    static_cast<double>(psi1[idx3(nx - 2, j, k, nx, ny)])) /
                   dx;
        dpsi2_dx = (static_cast<double>(psi2[idx3(nx - 1, j, k, nx, ny)]) -
                    static_cast<double>(psi2[idx3(nx - 2, j, k, nx, ny)])) /
                   dx;
    } else {
        dpsi1_dx = (static_cast<double>(psi1[idx3(i + 1, j, k, nx, ny)]) -
                    static_cast<double>(psi1[idx3(i - 1, j, k, nx, ny)])) /
                   (2.0 * dx);
        dpsi2_dx = (static_cast<double>(psi2[idx3(i + 1, j, k, nx, ny)]) -
                    static_cast<double>(psi2[idx3(i - 1, j, k, nx, ny)])) /
                   (2.0 * dx);
    }

    // ── ∂ψ/∂y: periodic central ─────────────────────────────────────────────
    const int jp1 = (j + 1) % ny;
    const int jm1 = (j - 1 + ny) % ny;
    const double p1_c = static_cast<double>(psi1[idx3(i, j, k, nx, ny)]);
    double p1_jp = static_cast<double>(psi1[idx3(i, jp1, k, nx, ny)]);
    double p1_jm = static_cast<double>(psi1[idx3(i, jm1, k, nx, ny)]);
    p1_jp += Ly * round((p1_c - p1_jp) / Ly);
    p1_jm += Ly * round((p1_c - p1_jm) / Ly);
    const double dpsi1_dy = (p1_jp - p1_jm) / (2.0 * dy);
    const double dpsi2_dy = (static_cast<double>(psi2[idx3(i, jp1, k, nx, ny)]) -
                             static_cast<double>(psi2[idx3(i, jm1, k, nx, ny)])) /
                            (2.0 * dy);

    // ── ∂ψ/∂z: periodic central ─────────────────────────────────────────────
    const int kp1 = (k + 1) % nz;
    const int km1 = (k - 1 + nz) % nz;
    const double dpsi1_dz = (static_cast<double>(psi1[idx3(i, j, kp1, nx, ny)]) -
                             static_cast<double>(psi1[idx3(i, j, km1, nx, ny)])) /
                            (2.0 * dz);
    const double p2_c = static_cast<double>(psi2[idx3(i, j, k, nx, ny)]);
    double p2_kp = static_cast<double>(psi2[idx3(i, j, kp1, nx, ny)]);
    double p2_km = static_cast<double>(psi2[idx3(i, j, km1, nx, ny)]);
    p2_kp += Lz * round((p2_c - p2_kp) / Lz);
    p2_km += Lz * round((p2_c - p2_km) / Lz);
    const double dpsi2_dz = (p2_kp - p2_km) / (2.0 * dz);

    // ── Residuals ────────────────────────────────────────────────────────────
    r1[idx3(i, j, k, nx, ny)] = static_cast<float>(vx * dpsi1_dx + vy * dpsi1_dy + vz * dpsi1_dz);
    r2[idx3(i, j, k, nx, ny)] = static_cast<float>(vx * dpsi2_dx + vy * dpsi2_dy + vz * dpsi2_dz);
}

/**
 * @brief March one x-plane for defect-correction fields delta_psi1, delta_psi2.
 *
 * Solves one explicit semi-Lagrangian step of:
 *   v·∇(delta_psi) = -r
 * using the same backtrace geometry as Level A.
 */
__global__ void kernel_refine_advance_plane(float* __restrict__ delta1, float* __restrict__ delta2,
                                            const float* __restrict__ r1,
                                            const float* __restrict__ r2,
                                            const real* __restrict__ U, const real* __restrict__ V,
                                            const real* __restrict__ W, int nx, int ny, int nz,
                                            double dx, double dy, double dz, double Ly, double Lz,
                                            double eps_vx, double src_clip_1, double src_clip_2,
                                            int i_plane, int* __restrict__ vx_clamped_cnt) {
    const int j = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int k = static_cast<int>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (j >= ny || k >= nz)
        return;

    const int i = i_plane;
    const int i1 = i_plane + 1;

    const double vx_i = 0.5 * (static_cast<double>(U[idxU(i, j, k, nx, ny)]) +
                               static_cast<double>(U[idxU(i + 1, j, k, nx, ny)]));
    const double vy_i = 0.5 * (static_cast<double>(V[i + nx * j + nx * (ny + 1) * k]) +
                               static_cast<double>(V[i + nx * (j + 1) + nx * (ny + 1) * k]));
    const double vz_i = 0.5 * (static_cast<double>(W[i + nx * j + nx * ny * k]) +
                               static_cast<double>(W[i + nx * j + nx * ny * (k + 1)]));

    const double vx_i1 = 0.5 * (static_cast<double>(U[idxU(i1, j, k, nx, ny)]) +
                                static_cast<double>(U[idxU(i1 + 1, j, k, nx, ny)]));
    const double vy_i1 = 0.5 * (static_cast<double>(V[i1 + nx * j + nx * (ny + 1) * k]) +
                                static_cast<double>(V[i1 + nx * (j + 1) + nx * (ny + 1) * k]));
    const double vz_i1 = 0.5 * (static_cast<double>(W[i1 + nx * j + nx * ny * k]) +
                                static_cast<double>(W[i1 + nx * j + nx * ny * (k + 1)]));

    const double vx_avg = 0.5 * (vx_i + vx_i1);
    const double vy_avg = 0.5 * (vy_i + vy_i1);
    const double vz_avg = 0.5 * (vz_i + vz_i1);

    const bool clamped = (vx_avg <= eps_vx);
    if (clamped)
        atomicAdd(vx_clamped_cnt, 1);
    const double vx_eff = clamped ? eps_vx : vx_avg;

    const double y_dest = (j + 0.5) * dy;
    const double z_dest = (k + 0.5) * dz;

    const double uy = vy_avg / vx_eff;
    const double uz = vz_avg / vx_eff;

    double y_star = wrap_periodic(y_dest - uy * dx, Ly);
    double z_star = wrap_periodic(z_dest - uz * dx, Lz);

    const double jf = y_star / dy - 0.5;
    const double kf = z_star / dz - 0.5;

    const int j0_raw = static_cast<int>(floor(jf));
    const int k0_raw = static_cast<int>(floor(kf));
    const double t = jf - j0_raw;
    const double s = kf - k0_raw;

    const int j0 = imod(j0_raw, ny);
    const int j1 = imod(j0_raw + 1, ny);
    const int k0 = imod(k0_raw, nz);
    const int k1 = imod(k0_raw + 1, nz);

    // delta_psi is a gauge-fixed correction: periodic indexing only, no value
    // lifting.
    const double d1_00 = static_cast<double>(delta1[idx3(i, j0, k0, nx, ny)]);
    const double d1_10 = static_cast<double>(delta1[idx3(i, j1, k0, nx, ny)]);
    const double d1_01 = static_cast<double>(delta1[idx3(i, j0, k1, nx, ny)]);
    const double d1_11 = static_cast<double>(delta1[idx3(i, j1, k1, nx, ny)]);
    const double d1_interp =
        (1.0 - s) * ((1.0 - t) * d1_00 + t * d1_10) + s * ((1.0 - t) * d1_01 + t * d1_11);

    const double d2_00 = static_cast<double>(delta2[idx3(i, j0, k0, nx, ny)]);
    const double d2_10 = static_cast<double>(delta2[idx3(i, j1, k0, nx, ny)]);
    const double d2_01 = static_cast<double>(delta2[idx3(i, j0, k1, nx, ny)]);
    const double d2_11 = static_cast<double>(delta2[idx3(i, j1, k1, nx, ny)]);
    const double d2_interp =
        (1.0 - s) * ((1.0 - t) * d2_00 + t * d2_10) + s * ((1.0 - t) * d2_01 + t * d2_11);

    const double rr1 = 0.5 * (static_cast<double>(r1[idx3(i, j, k, nx, ny)]) +
                              static_cast<double>(r1[idx3(i1, j, k, nx, ny)]));
    const double rr2 = 0.5 * (static_cast<double>(r2[idx3(i, j, k, nx, ny)]) +
                              static_cast<double>(r2[idx3(i1, j, k, nx, ny)]));

    const double src1 = clamp_sym((dx / vx_eff) * rr1, src_clip_1);
    const double src2 = clamp_sym((dx / vx_eff) * rr2, src_clip_2);

    const double d1_new = d1_interp - src1;
    const double d2_new = d2_interp - src2;

    delta1[idx3(i1, j, k, nx, ny)] = static_cast<float>(d1_new);
    delta2[idx3(i1, j, k, nx, ny)] = static_cast<float>(d2_new);
}

/**
 * @brief Build trial psi field from immutable base + alpha*delta.
 */
__global__ void
kernel_build_refine_trial(const float* __restrict__ psi1_base, const float* __restrict__ psi2_base,
                          const float* __restrict__ delta1, const float* __restrict__ delta2,
                          int nx, int ny, int nz, double Ly, double Lz, double alpha,
                          float* __restrict__ psi1_trial, float* __restrict__ psi2_trial) {
    const int c = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = nx * ny * nz;
    if (c >= total)
        return;

    const double p1 = static_cast<double>(psi1_base[c]) + alpha * static_cast<double>(delta1[c]);
    const double p2 = static_cast<double>(psi2_base[c]) + alpha * static_cast<double>(delta2[c]);

    psi1_trial[c] = static_cast<float>(wrap_periodic(p1, Ly));
    psi2_trial[c] = static_cast<float>(wrap_periodic(p2, Lz));
}

// ============================================================================
// PsptaPsiField host methods
// ============================================================================

void PsptaPsiField::resize(const Grid3D& grid) {
    nx = grid.nx;
    ny = grid.ny;
    nz = grid.nz;
    dx = static_cast<double>(grid.dx);
    dy = static_cast<double>(grid.dy);
    dz = static_cast<double>(grid.dz);

    const size_t n = static_cast<size_t>(nx) * ny * nz;
    psi1.resize(n);
    psi2.resize(n);
    d_vx_clamped_counter_.resize(1);
}

static inline double combined_rms(const PsiQualityReport& q) {
    return sqrt(0.5 * (q.rms_r1 * q.rms_r1 + q.rms_r2 * q.rms_r2));
}

PsptaPrecomputeReport PsptaPsiField::precompute_levelA(const VelocityField& vel, const Grid3D& grid,
                                                       cudaStream_t stream, double eps_vx) {
    // Resize / re-use existing allocations
    resize(grid);

    const int nx_ = nx, ny_ = ny, nz_ = nz;
    const double dx_ = dx, dy_ = dy, dz_ = dz;
    const double Ly = ny_ * dy_;
    const double Lz = nz_ * dz_;

    // ── Reset device counter ─────────────────────────────────────────────────
    MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(d_vx_clamped_counter_.data(), 0, sizeof(int), stream));

    // ── Kernel launch configuration ──────────────────────────────────────────
    // 2D: threads over (j, k) in each YZ plane
    const dim3 block2d(16, 16);
    const dim3 grid2d(static_cast<unsigned>((ny_ + 15) / 16),
                      static_cast<unsigned>((nz_ + 15) / 16));

    // ── Step 1: initialise inlet plane (i = 0) ───────────────────────────────
    kernel_init_inlet<<<grid2d, block2d, 0, stream>>>(psi1.data(), psi2.data(), nx_, ny_, nz_, dy_,
                                                      dz_);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    // ── Step 2: march in x, plane by plane ──────────────────────────────────
    for (int i = 0; i < nx_ - 1; ++i) {
        kernel_advance_plane<<<grid2d, block2d, 0, stream>>>(
            psi1.data(), psi2.data(), vel.U.data(), vel.V.data(), vel.W.data(), nx_, ny_, nz_, dx_,
            dy_, dz_, Ly, Lz, eps_vx, i, d_vx_clamped_counter_.data());
        MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
    }

    // ── Read back diagnostic counter ─────────────────────────────────────────
    // A single stream synchronization after all physics kernels are queued.
    // precompute_levelA is called once per realization; this sync is acceptable.
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream));

    int h_cnt = 0;
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(&h_cnt, d_vx_clamped_counter_.data(), sizeof(int), cudaMemcpyDeviceToHost));

    PsptaPrecomputeReport report;
    report.n_vx_clamped = static_cast<long long>(h_cnt);
    report.n_total = static_cast<long long>(nx_ - 1) * ny_ * nz_;
    return report;
}

void PsptaPsiField::compute_vdotgradpsi_norms(const VelocityField& vel, float* out_r1,
                                              float* out_r2, cudaStream_t stream) const {
    const double Ly = static_cast<double>(ny) * dy;
    const double Lz = static_cast<double>(nz) * dz;

    const dim3 block3d(8, 8, 8);
    const dim3 grid3d(static_cast<unsigned>((nx + 7) / 8), static_cast<unsigned>((ny + 7) / 8),
                      static_cast<unsigned>((nz + 7) / 8));

    kernel_vdotgradpsi<<<grid3d, block3d, 0, stream>>>(psi1.data(), psi2.data(), vel.U.data(),
                                                       vel.V.data(), vel.W.data(), nx, ny, nz, dx,
                                                       dy, dz, Ly, Lz, out_r1, out_r2);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// Diagnostic kernel: v·∇ψ quality reduction
// ============================================================================

/// CAS-based double atomicMax (for positive values only).
__device__ __forceinline__ void atomic_max_double(double* addr, double val) {
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long assumed, old = *addr_ull;
    do {
        assumed = old;
        const double cur = __longlong_as_double((long long)assumed);
        if (val <= cur)
            break;
        old = atomicCAS(addr_ull, assumed, (unsigned long long)__double_as_longlong(val));
    } while (assumed != old);
}

/// CAS-based double atomicAdd — works on all compute capabilities.
__device__ __forceinline__ void atomic_add_double(double* addr, double val) {
    unsigned long long* addr_ull = (unsigned long long*)addr;
    unsigned long long assumed, old = *addr_ull;
    do {
        assumed = old;
        const double sum = __longlong_as_double((long long)assumed) + val;
        old = atomicCAS(addr_ull, assumed, (unsigned long long)__double_as_longlong(sum));
    } while (assumed != old);
}

/**
 * @brief Reduce v·∇ψ residuals to {sumsq1, sumsq2, maxabs1, maxabs2, count}.
 *
 * Each thread handles one cell (1D index c = i + nx*(j + ny*k)).
 * Cell-center velocity = average of adjacent CompactMAC faces.
 * ∇ψ: central FDs with one-sided at x-boundaries; periodic-lifted central in
 * y,z.
 *
 * Output buf[5] (device double array):
 *   [0] sumsq1  [1] sumsq2  [2] maxabs1  [3] maxabs2  [4] count (as double)
 */
__global__ void
kernel_psi_quality_reduce(const float* __restrict__ psi1_buf, const float* __restrict__ psi2_buf,
                          const real* __restrict__ U, // (nx+1)*ny*nz
                          const real* __restrict__ V, // nx*(ny+1)*nz
                          const real* __restrict__ W, // nx*ny*(nz+1)
                          int nx, int ny, int nz, double dx, double dy, double dz, double Ly,
                          double Lz,
                          double* __restrict__ out_buf) // 5 doubles initialized to 0
{
    extern __shared__ double s[];
    // Shared layout: [0..bs-1]=sumsq1, [bs..2bs-1]=sumsq2,
    //                [2bs..3bs-1]=max1, [3bs..4bs-1]=max2
    const int bs = blockDim.x;
    double* s_ssq1 = s;
    double* s_ssq2 = s + bs;
    double* s_max1 = s + 2 * bs;
    double* s_max2 = s + 3 * bs;

    const int tid = threadIdx.x;
    s_ssq1[tid] = 0.0;
    s_ssq2[tid] = 0.0;
    s_max1[tid] = 0.0;
    s_max2[tid] = 0.0;
    __syncthreads();

    const int total = nx * ny * nz;
    int c = blockIdx.x * bs + tid;
    if (c < total) {
        // Decompose flat index
        const int i = c % nx;
        const int j = (c / nx) % ny;
        const int k = c / (nx * ny);

        // ── Cell-center velocity (CompactMAC average) ─────────────────
        // U faces at (i,j,k) and (i+1,j,k); V at (i,j,k) and (i,j+1,k); etc.
        const auto idx_U = [&](int ii, int jj, int kk) {
            return ii + (nx + 1) * jj + (nx + 1) * ny * kk;
        };
        const auto idx_V = [&](int ii, int jj, int kk) {
            return ii + nx * jj + nx * (ny + 1) * kk;
        };
        const auto idx_W = [&](int ii, int jj, int kk) { return ii + nx * jj + nx * ny * kk; };

        const double vx = 0.5 * (static_cast<double>(U[idx_U(i, j, k)]) +
                                 static_cast<double>(U[idx_U(i + 1, j, k)]));
        const double vy = 0.5 * (static_cast<double>(V[idx_V(i, j, k)]) +
                                 static_cast<double>(V[idx_V(i, j + 1, k)]));
        const double vz = 0.5 * (static_cast<double>(W[idx_W(i, j, k)]) +
                                 static_cast<double>(W[idx_W(i, j, k + 1)]));

        // ── ∂ψ1/∂x (one-sided at boundaries) ────────────────────────
        const auto idx_psi = [&](int ii, int jj, int kk) { return ii + nx * (jj + ny * kk); };

        const double p1_c = static_cast<double>(psi1_buf[idx_psi(i, j, k)]);

        double dp1_dx;
        if (i == 0)
            dp1_dx = (static_cast<double>(psi1_buf[idx_psi(1, j, k)]) - p1_c) / dx;
        else if (i == nx - 1)
            dp1_dx = (p1_c - static_cast<double>(psi1_buf[idx_psi(nx - 2, j, k)])) / dx;
        else
            dp1_dx = (static_cast<double>(psi1_buf[idx_psi(i + 1, j, k)]) -
                      static_cast<double>(psi1_buf[idx_psi(i - 1, j, k)])) /
                     (2.0 * dx);

        // ── ∂ψ1/∂y (periodic central with lifting; period = Ly) ──────
        const int j_p = (j + 1) % ny;
        const int j_m = (j - 1 + ny) % ny;
        double p1_jp = static_cast<double>(psi1_buf[idx_psi(i, j_p, k)]);
        double p1_jm = static_cast<double>(psi1_buf[idx_psi(i, j_m, k)]);
        p1_jp += Ly * round((p1_c - p1_jp) / Ly);
        p1_jm += Ly * round((p1_c - p1_jm) / Ly);
        const double dp1_dy = (p1_jp - p1_jm) / (2.0 * dy);

        // ── ∂ψ1/∂z (periodic central, no lifting needed for ψ1 in z) ─
        const int k_p = (k + 1) % nz;
        const int k_m = (k - 1 + nz) % nz;
        const double dp1_dz = (static_cast<double>(psi1_buf[idx_psi(i, j, k_p)]) -
                               static_cast<double>(psi1_buf[idx_psi(i, j, k_m)])) /
                              (2.0 * dz);

        const double r1 = vx * dp1_dx + vy * dp1_dy + vz * dp1_dz;

        // ── Same for ψ2 (period = Lz in z direction) ─────────────────
        const double p2_c = static_cast<double>(psi2_buf[idx_psi(i, j, k)]);

        double dp2_dx;
        if (i == 0)
            dp2_dx = (static_cast<double>(psi2_buf[idx_psi(1, j, k)]) - p2_c) / dx;
        else if (i == nx - 1)
            dp2_dx = (p2_c - static_cast<double>(psi2_buf[idx_psi(nx - 2, j, k)])) / dx;
        else
            dp2_dx = (static_cast<double>(psi2_buf[idx_psi(i + 1, j, k)]) -
                      static_cast<double>(psi2_buf[idx_psi(i - 1, j, k)])) /
                     (2.0 * dx);

        const double dp2_dy = (static_cast<double>(psi2_buf[idx_psi(i, j_p, k)]) -
                               static_cast<double>(psi2_buf[idx_psi(i, j_m, k)])) /
                              (2.0 * dy);

        double p2_kp = static_cast<double>(psi2_buf[idx_psi(i, j, k_p)]);
        double p2_km = static_cast<double>(psi2_buf[idx_psi(i, j, k_m)]);
        p2_kp += Lz * round((p2_c - p2_kp) / Lz);
        p2_km += Lz * round((p2_c - p2_km) / Lz);
        const double dp2_dz = (p2_kp - p2_km) / (2.0 * dz);

        const double r2 = vx * dp2_dx + vy * dp2_dy + vz * dp2_dz;

        s_ssq1[tid] = r1 * r1;
        s_ssq2[tid] = r2 * r2;
        s_max1[tid] = fabs(r1);
        s_max2[tid] = fabs(r2);
    }
    __syncthreads();

    // ── Block reduction (tree, power-of-2 stride) ─────────────────────
    for (int stride = bs / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_ssq1[tid] += s_ssq1[tid + stride];
            s_ssq2[tid] += s_ssq2[tid + stride];
            if (s_max1[tid + stride] > s_max1[tid])
                s_max1[tid] = s_max1[tid + stride];
            if (s_max2[tid + stride] > s_max2[tid])
                s_max2[tid] = s_max2[tid + stride];
        }
        __syncthreads();
    }

    // ── Atomically contribute block result to global buffer ───────────
    if (tid == 0) {
        atomic_add_double(&out_buf[0], s_ssq1[0]);
        atomic_add_double(&out_buf[1], s_ssq2[0]);
        atomic_max_double(&out_buf[2], s_max1[0]);
        atomic_max_double(&out_buf[3], s_max2[0]);
        atomic_add_double(&out_buf[4],
                          static_cast<double>(min(bs, total - (int)(blockIdx.x * bs))));
    }
}

// ── Host method ──────────────────────────────────────────────────────────────

PsiQualityReport PsptaPsiField::compute_psi_quality_from_buffers(const float* psi1_buf,
                                                                 const float* psi2_buf,
                                                                 const VelocityField& vel,
                                                                 const Grid3D& grid,
                                                                 cudaStream_t stream) const {
    PsiQualityReport report;
    const int total = nx * ny * nz;
    if (total <= 0)
        return report;

    // Reuse or allocate 5-element scratch buffer
    if (d_psi_quality_buf_.size() < 5)
        d_psi_quality_buf_.resize(5);

    MACROFLOW3D_CUDA_CHECK(
        cudaMemsetAsync(d_psi_quality_buf_.data(), 0, 5 * sizeof(double), stream));

    const double Ly = static_cast<double>(grid.ny) * static_cast<double>(grid.dy);
    const double Lz = static_cast<double>(grid.nz) * static_cast<double>(grid.dz);

    const int block_sz = 256;
    const int nblocks = (total + block_sz - 1) / block_sz;
    const size_t smem = 4 * block_sz * sizeof(double);

    kernel_psi_quality_reduce<<<nblocks, block_sz, smem, stream>>>(
        psi1_buf, psi2_buf, vel.U.data(), vel.V.data(), vel.W.data(), nx, ny, nz,
        static_cast<double>(grid.dx), static_cast<double>(grid.dy), static_cast<double>(grid.dz),
        Ly, Lz, d_psi_quality_buf_.data());
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    double host_buf[5] = {};
    MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(host_buf, d_psi_quality_buf_.data(), 5 * sizeof(double),
                                           cudaMemcpyDeviceToHost, stream));
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream));

    const double count = host_buf[4];
    report.n_cells = static_cast<long long>(count);
    if (count > 0.0) {
        report.rms_r1 = sqrt(host_buf[0] / count);
        report.rms_r2 = sqrt(host_buf[1] / count);
    }
    report.max_r1 = host_buf[2];
    report.max_r2 = host_buf[3];
    return report;
}

PsiQualityReport PsptaPsiField::compute_psi_quality(const VelocityField& vel, const Grid3D& grid,
                                                    cudaStream_t stream) const {
    return compute_psi_quality_from_buffers(psi1.data(), psi2.data(), vel, grid, stream);
}

PsiRefineReport PsptaPsiField::refine_psi(const VelocityField& vel, const Grid3D& grid,
                                          cudaStream_t stream, const PsiRefineConfig& cfg) {
    PsiRefineReport report;
    report.enabled = cfg.enabled;
    report.stop_reason = "max_iters";

    report.seed_quality = compute_psi_quality(vel, grid, stream);
    report.final_quality = report.seed_quality;

    if (!cfg.enabled) {
        report.stop_reason = "disabled";
        return report;
    }
    if (cfg.outer_iters <= 0) {
        report.stop_reason = "invalid_outer_iters";
        return report;
    }
    if (cfg.max_backtracks <= 0) {
        report.stop_reason = "invalid_max_backtracks";
        return report;
    }
    if (cfg.omega <= 0.0 || cfg.omega_min <= 0.0 || cfg.omega_min > cfg.omega) {
        report.stop_reason = "invalid_omega_range";
        return report;
    }
    if (cfg.source_clip_cells <= 0.0) {
        report.stop_reason = "invalid_source_clip";
        return report;
    }
    if (cfg.no_descent_patience <= 0) {
        report.stop_reason = "invalid_no_descent_patience";
        return report;
    }

    const size_t n = static_cast<size_t>(nx) * ny * nz;
    if (d_r1_.size() < n)
        d_r1_.resize(n);
    if (d_r2_.size() < n)
        d_r2_.resize(n);
    if (d_delta1_.size() < n)
        d_delta1_.resize(n);
    if (d_delta2_.size() < n)
        d_delta2_.resize(n);
    if (d_trial_psi1_.size() < n)
        d_trial_psi1_.resize(n);
    if (d_trial_psi2_.size() < n)
        d_trial_psi2_.resize(n);
    if (d_vx_clamped_counter_.size() < 1)
        d_vx_clamped_counter_.resize(1);

    const dim3 block2d(16, 16);
    const dim3 grid2d(static_cast<unsigned>((ny + 15) / 16), static_cast<unsigned>((nz + 15) / 16));

    const int block1d = 256;
    const int total = nx * ny * nz;
    const int grid1d = (total + block1d - 1) / block1d;

    const double Ly = static_cast<double>(grid.ny) * static_cast<double>(grid.dy);
    const double Lz = static_cast<double>(grid.nz) * static_cast<double>(grid.dz);
    const double src_clip_1 = cfg.source_clip_cells * dy;
    const double src_clip_2 = cfg.source_clip_cells * dz;
    const double seed_combo = combined_rms(report.seed_quality);

    if (cfg.print_every_iter) {
        std::printf("       [psi_refine] seed  rms_r1=%.3e rms_r2=%.3e combo=%.3e\n",
                    report.seed_quality.rms_r1, report.seed_quality.rms_r2, seed_combo);
        if (cfg.eq13_diagnostics) {
            std::printf("       [psi_refine] note: eq13_diagnostics reserved for "
                        "future V2\n");
        }
    }

    PsiQualityReport before = report.seed_quality;
    int no_descent_streak = 0;
    for (int iter = 1; iter <= cfg.outer_iters; ++iter) {
        PsiRefineIterReport it;
        it.iter = iter;
        it.omega_init = cfg.omega;
        it.omega_accepted = 0.0;
        it.backtracks = 0;
        it.accepted_step = false;
        it.step_status = "accepted";
        it.before = before;

        compute_vdotgradpsi_norms(vel, d_r1_.data(), d_r2_.data(), stream);

        MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(d_delta1_.data(), 0, n * sizeof(float), stream));
        MACROFLOW3D_CUDA_CHECK(cudaMemsetAsync(d_delta2_.data(), 0, n * sizeof(float), stream));
        MACROFLOW3D_CUDA_CHECK(
            cudaMemsetAsync(d_vx_clamped_counter_.data(), 0, sizeof(int), stream));

        for (int i = 0; i < nx - 1; ++i) {
            kernel_refine_advance_plane<<<grid2d, block2d, 0, stream>>>(
                d_delta1_.data(), d_delta2_.data(), d_r1_.data(), d_r2_.data(), vel.U.data(),
                vel.V.data(), vel.W.data(), nx, ny, nz, dx, dy, dz, Ly, Lz, cfg.eps_vx, src_clip_1,
                src_clip_2, i, d_vx_clamped_counter_.data());
            MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
        }

        const double before_combo = combined_rms(it.before);
        double alpha_try = cfg.omega;
        bool accepted = false;
        PsiQualityReport trial_quality = it.before;

        for (int bt = 0; bt < cfg.max_backtracks && alpha_try >= cfg.omega_min; ++bt) {
            it.backtracks = bt;

            kernel_build_refine_trial<<<grid1d, block1d, 0, stream>>>(
                psi1.data(), psi2.data(), d_delta1_.data(), d_delta2_.data(), nx, ny, nz, Ly, Lz,
                alpha_try, d_trial_psi1_.data(), d_trial_psi2_.data());
            MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

            kernel_init_inlet<<<grid2d, block2d, 0, stream>>>(
                d_trial_psi1_.data(), d_trial_psi2_.data(), nx, ny, nz, dy, dz);
            MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

            trial_quality = compute_psi_quality_from_buffers(
                d_trial_psi1_.data(), d_trial_psi2_.data(), vel, grid, stream);
            const double trial_combo = combined_rms(trial_quality);

            if (trial_combo <= before_combo * (1.0 + 1.0e-12)) {
                accepted = true;
                it.accepted_step = true;
                it.omega_accepted = alpha_try;
                it.step_status = (bt == 0) ? "accepted" : "accepted_backtrack";
                it.after = trial_quality;
                MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(psi1.data(), d_trial_psi1_.data(),
                                                       n * sizeof(float), cudaMemcpyDeviceToDevice,
                                                       stream));
                MACROFLOW3D_CUDA_CHECK(cudaMemcpyAsync(psi2.data(), d_trial_psi2_.data(),
                                                       n * sizeof(float), cudaMemcpyDeviceToDevice,
                                                       stream));
                break;
            }

            alpha_try *= 0.5;
        }

        if (!accepted) {
            it.step_status = "rejected_no_descent";
            it.after = it.before;
            it.omega_accepted = 0.0;
        }

        int h_cnt = 0;
        MACROFLOW3D_CUDA_CHECK(
            cudaMemcpy(&h_cnt, d_vx_clamped_counter_.data(), sizeof(int), cudaMemcpyDeviceToHost));
        it.n_vx_clamped = static_cast<long long>(h_cnt);
        it.n_total = static_cast<long long>(nx - 1) * ny * nz;
        it.vx_clamped_frac = (it.n_total > 0) ? static_cast<double>(it.n_vx_clamped) /
                                                    static_cast<double>(it.n_total)
                                              : 0.0;

        const double after_combo = combined_rms(it.after);
        it.rel_gain = (before_combo > 0.0) ? ((before_combo - after_combo) / before_combo) : 0.0;

        const bool stop_abs = (after_combo <= cfg.stop_abs_rms);
        const bool stop_rel =
            (seed_combo > 0.0) ? (after_combo <= cfg.stop_rel_rms * seed_combo) : false;

        if (it.accepted_step)
            no_descent_streak = 0;
        else
            no_descent_streak += 1;

        const bool stop_no_descent = (no_descent_streak >= cfg.no_descent_patience);
        const bool early_stop = stop_abs || stop_rel || stop_no_descent;
        it.converged = stop_abs || stop_rel;

        report.history.push_back(it);
        report.iters_done = iter;
        report.final_quality = it.after;

        if (cfg.print_every_iter) {
            std::printf("       [psi_refine] iter %d/%d omega_init=%.3e omega_acc=%.3e bt=%d "
                        "rms_r1 %.3e->%.3e rms_r2 %.3e->%.3e gain=%.3f clamp=%.3e "
                        "status=%s%s\n",
                        iter, cfg.outer_iters, it.omega_init, it.omega_accepted, it.backtracks,
                        it.before.rms_r1, it.after.rms_r1, it.before.rms_r2, it.after.rms_r2,
                        it.rel_gain, it.vx_clamped_frac, it.step_status,
                        early_stop ? " early-stop" : "");
        }

        if (early_stop) {
            report.converged = it.converged;
            if (stop_abs)
                report.stop_reason = "stop_abs_rms";
            else if (stop_rel)
                report.stop_reason = "stop_rel_rms";
            else
                report.stop_reason = "no_descent";
            break;
        }

        before = it.after;
    }

    if (cfg.print_every_iter) {
        std::printf("       [psi_refine] final rms_r1=%.3e rms_r2=%.3e stop=%s\n",
                    report.final_quality.rms_r1, report.final_quality.rms_r2, report.stop_reason);
    }
    return report;
}

} // namespace pspta
} // namespace particles
} // namespace physics
} // namespace macroflow3d
