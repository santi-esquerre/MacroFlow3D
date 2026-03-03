/**
 * @file PsptaEngine.cu
 * @brief PSPTA transport engine — Tier-2 advance + project implementation.
 *
 * This file contains:
 *   (A) Device utilities (velocity sampling, ψ trilinear + partials, Newton)
 *   (B) Four kernels:
 *         kernel_inject_box         – uniform injection
 *         kernel_init_invariants    – ψ sample at injection points
 *         kernel_pspta_step         – RK2-x + 3× Newton projection
 *         kernel_compute_unwrapped  – position + wrapCount → unwrapped coord
 *   (C) PsptaEngine host methods
 *
 * @note  All Newton arithmetic is in double (float positions → promoted).
 *        Per-particle buffers (psi_const, y/z_guess, fail_count) are float
 *        or uint32_t to minimise memory.
 */

#include "PsptaEngine.hpp"
#include "../../../runtime/cuda_check.cuh"
#include <math.h>     // floor, round (device)
#include <stdint.h>

namespace macroflow3d {
namespace physics {
namespace particles {
namespace pspta {

// ============================================================================
// A.1  Integer helpers
// ============================================================================

__device__ __forceinline__
int imod(int n, int N) { return ((n % N) + N) % N; }

__device__ __forceinline__
double clamp_d(double v, double lo, double hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// ============================================================================
// A.2  Deterministic hash RNG  (inject_box)
// ============================================================================

/// 64-bit Murmur-like finalizer mix.
__device__ __forceinline__
uint64_t hash64(uint64_t k) {
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdULL;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ULL;
    k ^= k >> 33;
    return k;
}

/// Hash to float in [0, 1).
__device__ __forceinline__
float hash_to_f01(uint64_t seed, uint64_t salt) {
    uint64_t h = hash64(seed ^ salt);
    // upper 32 bits → float
    return static_cast<float>(h >> 32) * (1.0f / 4294967296.0f);
}

// ============================================================================
// A.3  Periodic wrap (double) and periodic subtraction
// ============================================================================

__device__ __forceinline__
double wrap_to_L(double x, double L) {
    return x - floor(x / L) * L;
}

/// Nearest-image residual: r ∈ (-L/2, L/2]
__device__ __forceinline__
double wrap_diff(double f, double L) {
    return f - round(f / L) * L;
}

// ============================================================================
// A.4  CompactMAC face indices
// ============================================================================

/// U index: i + (nx+1)*j + (nx+1)*ny*k
__device__ __forceinline__
int idx_U(int i, int j, int k, int nx, int ny) {
    return i + (nx + 1) * j + (nx + 1) * ny * k;
}

/// V index: i + nx*j + nx*(ny+1)*k
__device__ __forceinline__
int idx_V(int i, int j, int k, int nx_v, int nyp1) {
    return i + nx_v * j + nx_v * nyp1 * k;
}

/// W index: i + nx*j + nx*ny*k
__device__ __forceinline__
int idx_W(int i, int j, int k, int nx_w, int ny_w) {
    return i + nx_w * j + nx_w * ny_w * k;
}

/// Cell-centered ψ index (x-fastest): i + nx*(j + ny*k)
__device__ __forceinline__
int idx_psi(int i, int j, int k, int nx, int ny) {
    return i + nx * (j + ny * k);
}

// ============================================================================
// A.5  Trilinear sample of ψ at 3D point WITH periodic lifting
//
//   Period is Ly for ψ1, Lz for ψ2.
//   "x" direction is NOT periodic but clamped to [0, Lx).
//   "y" and "z" are periodic with their respective periods.
//
//   The 8 corners are lifted so all are in the same periodic sheet as
//   corner_000.  This prevents interpolation artefacts at the periodic seam.
//
//   On return: *out_psi ∈ [0, L).
// ============================================================================

/// Trilinear sample + periodic lifting.
/// Returns ψ(x,y,z) in [0, L) (float).
/// Also writes analytic partials d_psi/dy and d_psi/dz (double) for Newton.
__device__
double sample_psi_and_partials(
    const float* __restrict__ psi_buf,
    double x, double y, double z,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double L_period_y,  ///< period in y dimension (Ly for ψ1, Ly for ψ2 too since x-fastest)
    double L_period_z,  ///< period in z dimension (Lz for ψ1, Lz for ψ2)
    double L_self,      ///< period of ψ values themselves (Ly for ψ1, Lz for ψ2)
    double* dpsi_dy,
    double* dpsi_dz)
{
    // ── Fractional cell-center indices ─────────────────────────────────────
    // x: non-periodic, clamp i0 to [0, nx-2]
    const double xf = x / dx - 0.5;
    int i0 = static_cast<int>(floor(xf));
    i0 = (i0 < 0) ? 0 : (i0 > nx - 2 ? nx - 2 : i0);
    const int i1 = i0 + 1;
    const double tx = clamp_d(xf - i0, 0.0, 1.0);

    // y: periodic
    const double yf = y / dy - 0.5;
    const int j0_raw = static_cast<int>(floor(yf));
    const int j0 = imod(j0_raw,     ny);
    const int j1 = imod(j0_raw + 1, ny);
    const double ty = yf - j0_raw;  // ∈ [0, 1)

    // z: periodic
    const double zf = z / dz - 0.5;
    const int k0_raw = static_cast<int>(floor(zf));
    const int k0 = imod(k0_raw,     nz);
    const int k1 = imod(k0_raw + 1, nz);
    const double tz = zf - k0_raw;  // ∈ [0, 1)

    // ── Fetch 8 corners (as double) ────────────────────────────────────────
    double c000 = static_cast<double>(psi_buf[idx_psi(i0, j0, k0, nx, ny)]);
    double c100 = static_cast<double>(psi_buf[idx_psi(i1, j0, k0, nx, ny)]);
    double c010 = static_cast<double>(psi_buf[idx_psi(i0, j1, k0, nx, ny)]);
    double c110 = static_cast<double>(psi_buf[idx_psi(i1, j1, k0, nx, ny)]);
    double c001 = static_cast<double>(psi_buf[idx_psi(i0, j0, k1, nx, ny)]);
    double c101 = static_cast<double>(psi_buf[idx_psi(i1, j0, k1, nx, ny)]);
    double c011 = static_cast<double>(psi_buf[idx_psi(i0, j1, k1, nx, ny)]);
    double c111 = static_cast<double>(psi_buf[idx_psi(i1, j1, k1, nx, ny)]);

    // ── Periodic lifting relative to c000 ─────────────────────────────────
    // All corners shifted by the multiple of L_self nearest to c000.
    // This makes the stencil "seam-free" before interpolation.
    const double ref = c000;
    c100 += L_self * round((ref - c100) / L_self);
    c010 += L_self * round((ref - c010) / L_self);
    c110 += L_self * round((ref - c110) / L_self);
    c001 += L_self * round((ref - c001) / L_self);
    c101 += L_self * round((ref - c101) / L_self);
    c011 += L_self * round((ref - c011) / L_self);
    c111 += L_self * round((ref - c111) / L_self);

    // ── Trilinear weights ──────────────────────────────────────────────────
    const double tx0 = 1.0 - tx, tx1 = tx;
    const double ty0 = 1.0 - ty, ty1 = ty;
    const double tz0 = 1.0 - tz, tz1 = tz;

    // ── ψ value ────────────────────────────────────────────────────────────
    const double psi =
        c000*tx0*ty0*tz0 + c100*tx1*ty0*tz0 +
        c010*tx0*ty1*tz0 + c110*tx1*ty1*tz0 +
        c001*tx0*ty0*tz1 + c101*tx1*ty0*tz1 +
        c011*tx0*ty1*tz1 + c111*tx1*ty1*tz1;

    // ── Analytic partial ∂ψ/∂y ─────────────────────────────────────────────
    // ∂ty/∂y = 1/dy, ∂ty0/∂y = -1/dy, ∂ty1/∂y = +1/dy
    // Corners with j=j0 contribute -1/dy; corners with j=j1 contribute +1/dy.
    *dpsi_dy = (1.0 / dy) * (
        - c000*tx0*tz0 - c100*tx1*tz0
        - c001*tx0*tz1 - c101*tx1*tz1
        + c010*tx0*tz0 + c110*tx1*tz0
        + c011*tx0*tz1 + c111*tx1*tz1);

    // ── Analytic partial ∂ψ/∂z ─────────────────────────────────────────────
    *dpsi_dz = (1.0 / dz) * (
        - c000*tx0*ty0 - c100*tx1*ty0
        - c010*tx0*ty1 - c110*tx1*ty1
        + c001*tx0*ty0 + c101*tx1*ty0
        + c011*tx0*ty1 + c111*tx1*ty1);

    // Wrap ψ output to [0, L_self)
    return wrap_to_L(psi, L_self);
}

// ============================================================================
// A.6  Trilinear sample of vx (U-faces) at arbitrary 3D point
//
//   U(i,j,k) is at x-face position (i*dx, (j+0.5)*dy, (k+0.5)*dz).
//   i ∈ [0,nx], j ∈ [0,ny-1], k ∈ [0,nz-1].
//   y,z are periodic; x is clamped to face-index range [0,nx].
// ============================================================================

__device__ __forceinline__
double sample_vx(const real* __restrict__ U,
                 double x, double y, double z,
                 int nx, int ny, int nz,
                 double dx, double dy, double dz)
{
    // ── x: fractional face index, clamped ─────────────────────────────────
    const double fx = x / dx;          // faces at 0,1,...,nx (in units of dx)
    int i0 = static_cast<int>(floor(fx));
    i0 = (i0 < 0) ? 0 : (i0 >= nx ? nx - 1 : i0);  // clamp to [0, nx-1] → i1=i0+1≤nx
    const int i1 = i0 + 1;
    const double tx = clamp_d(fx - i0, 0.0, 1.0);

    // ── y: fractional cell-center index, periodic ──────────────────────────
    const double fy = y / dy - 0.5;
    const int j0_raw = static_cast<int>(floor(fy));
    const int j0 = imod(j0_raw,     ny);
    const int j1 = imod(j0_raw + 1, ny);
    const double ty = fy - j0_raw;

    // ── z: fractional cell-center index, periodic ──────────────────────────
    const double fz = z / dz - 0.5;
    const int k0_raw = static_cast<int>(floor(fz));
    const int k0 = imod(k0_raw,     nz);
    const int k1 = imod(k0_raw + 1, nz);
    const double tz = fz - k0_raw;

    // ── Trilinear interpolation ────────────────────────────────────────────
    const double u000 = static_cast<double>(U[idx_U(i0, j0, k0, nx, ny)]);
    const double u100 = static_cast<double>(U[idx_U(i1, j0, k0, nx, ny)]);
    const double u010 = static_cast<double>(U[idx_U(i0, j1, k0, nx, ny)]);
    const double u110 = static_cast<double>(U[idx_U(i1, j1, k0, nx, ny)]);
    const double u001 = static_cast<double>(U[idx_U(i0, j0, k1, nx, ny)]);
    const double u101 = static_cast<double>(U[idx_U(i1, j0, k1, nx, ny)]);
    const double u011 = static_cast<double>(U[idx_U(i0, j1, k1, nx, ny)]);
    const double u111 = static_cast<double>(U[idx_U(i1, j1, k1, nx, ny)]);

    return u000*(1-tx)*(1-ty)*(1-tz) + u100*tx*(1-ty)*(1-tz)
         + u010*(1-tx)*ty*(1-tz)     + u110*tx*ty*(1-tz)
         + u001*(1-tx)*(1-ty)*tz     + u101*tx*(1-ty)*tz
         + u011*(1-tx)*ty*tz         + u111*tx*ty*tz;
}

// ============================================================================
// A.7  2×2 Newton solver for (y, z) given fixed x
//
//   Solves:  ψ1(x, y, z) = ψ1_c  and  ψ2(x, y, z) = ψ2_c
//   using at most PSPTA_N_NEWTON steps with trust-region clamping.
//
//   Returns: ok = true if converged, false if ill-conditioned (|det|<det_min)
//   On output: *y_out, *z_out hold the best iterate (successful or last).
// ============================================================================

__device__
bool newton_solve_yz(
    const float* __restrict__ psi1_buf,
    const float* __restrict__ psi2_buf,
    double x,
    double psi1_c, double psi2_c,
    double y0, double z0,
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    double Ly, double Lz,
    double* y_out, double* z_out)
{
    double y = y0;
    double z = z0;

    // Relative convergence tolerance: scales with grid spacing.
    // Float32 ψ storage limits resolution to ~1e-4 * cell, so 1e-4 is appropriate.
    const double tol = PSPTA_TOL_FACTOR * fmin(dy, dz);

    const double trust_y = PSPTA_TRUST_FACTOR * dy;
    const double trust_z = PSPTA_TRUST_FACTOR * dz;

    for (int it = 0; it < PSPTA_N_NEWTON; ++it) {
        // ── Evaluate ψ1, ψ2 with analytic partials ─────────────────────────
        double dp1_dy, dp1_dz, dp2_dy, dp2_dz;

        const double p1 = sample_psi_and_partials(
            psi1_buf, x, y, z,
            nx, ny, nz, dx, dy, dz,
            Ly, Lz, Ly,
            &dp1_dy, &dp1_dz);

        const double p2 = sample_psi_and_partials(
            psi2_buf, x, y, z,
            nx, ny, nz, dx, dy, dz,
            Ly, Lz, Lz,
            &dp2_dy, &dp2_dz);

        // ── Periodic residuals ─────────────────────────────────────────────
        const double f1 = wrap_diff(p1 - psi1_c, Ly);
        const double f2 = wrap_diff(p2 - psi2_c, Lz);

        // ── Convergence check ──────────────────────────────────────────────
        const double res = fabs(f1) > fabs(f2) ? fabs(f1) : fabs(f2);
        if (res < tol) break;

        // ── Jacobian: J = [[a,b],[c,d]] ────────────────────────────────────
        const double a = dp1_dy, b = dp1_dz;
        const double c = dp2_dy, d = dp2_dz;
        const double det = a * d - b * c;

        if (fabs(det) < PSPTA_DET_MIN || det != det) {
            // Ill-conditioned — skip update but do not overwrite current iterate
            break;
        }

        // ── Newton step: δ = -J^{-1} f ────────────────────────────────────
        const double inv_det = 1.0 / det;
        double dy_step = -(d * f1 - b * f2) * inv_det * PSPTA_DAMPING;
        double dz_step = -(-c * f1 + a * f2) * inv_det * PSPTA_DAMPING;

        // ── Trust-region clamp ─────────────────────────────────────────────
        if (dy_step >  trust_y) dy_step =  trust_y;
        if (dy_step < -trust_y) dy_step = -trust_y;
        if (dz_step >  trust_z) dz_step =  trust_z;
        if (dz_step < -trust_z) dz_step = -trust_z;

        // ── Update and wrap into [0, L) ────────────────────────────────────
        y = wrap_to_L(y + dy_step, Ly);
        z = wrap_to_L(z + dz_step, Lz);
    }

    *y_out = y;
    *z_out = z;

    // ── Final convergence assessment ───────────────────────────────────────
    // Recompute residual with (just) the ψ values to judge success.
    // (Avoids extra EvalPsiAndPartials; use sample_psi_and_partials with dummy partials.)
    double dp1_dy_dummy, dp1_dz_dummy;
    const double p1_fin = sample_psi_and_partials(
        psi1_buf, x, y, z,
        nx, ny, nz, dx, dy, dz, Ly, Lz, Ly,
        &dp1_dy_dummy, &dp1_dz_dummy);
    double dp2_dy_dummy, dp2_dz_dummy;
    const double p2_fin = sample_psi_and_partials(
        psi2_buf, x, y, z,
        nx, ny, nz, dx, dy, dz, Ly, Lz, Lz,
        &dp2_dy_dummy, &dp2_dz_dummy);

    const double f1_fin = fabs(wrap_diff(p1_fin - psi1_c, Ly));
    const double f2_fin = fabs(wrap_diff(p2_fin - psi2_c, Lz));
    const double res_fin = f1_fin > f2_fin ? f1_fin : f2_fin;

    return (res_fin < tol);
}

// ============================================================================
// B.1  Kernel: inject_box
// ============================================================================

/**
 * @brief Initialize particle positions uniformly in [p0,p1] axis-aligned box.
 *
 * For plane injection (e.g. x0==x1), dx_range=0 and all particles land on
 * that plane.  Uses 3 independent hash salts per spatial dimension.
 *
 * Thread layout: 1D over [first, first+count).
 */
__global__
void kernel_inject_box(
    real*     __restrict__ px,
    real*     __restrict__ py,
    real*     __restrict__ pz,
    uint8_t*  __restrict__ status,
    int32_t*               wrapX,
    int32_t*               wrapY,
    int32_t*               wrapZ,
    int    first,
    int    total,
    real   x0, real y0, real z0,
    real   x_range, real y_range, real z_range,
    uint64_t seed)
{
    int idx = first + static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= first + total) return;

    const uint64_t h = static_cast<uint64_t>(idx);
    px[idx] = x0 + static_cast<real>(hash_to_f01(seed ^ h, 0xA1B2C3D4ULL)) * x_range;
    py[idx] = y0 + static_cast<real>(hash_to_f01(seed ^ h, 0xDEADBEEFULL)) * y_range;
    pz[idx] = z0 + static_cast<real>(hash_to_f01(seed ^ h, 0xFEEDFACEULL)) * z_range;
    status[idx] = 0;  // active

    if (wrapX) wrapX[idx] = 0;
    if (wrapY) wrapY[idx] = 0;
    if (wrapZ) wrapZ[idx] = 0;
}

// ============================================================================
// B.2  Kernel: init_invariants  (called from prepare())
// ============================================================================

/**
 * @brief Sample ψ1, ψ2 at each particle's position; store as per-particle constants.
 *
 * For each active particle p:
 *   psi1_const[p] = ψ1(x[p], y[p], z[p])   (trilinear, periodic lifting in y)
 *   psi2_const[p] = ψ2(x[p], y[p], z[p])   (trilinear, periodic lifting in z)
 *   y_guess[p]    = y[p]
 *   z_guess[p]    = z[p]
 *
 * Thread layout: 1D over particles [0, N).
 */
__global__
void kernel_init_invariants(
    const real*   __restrict__ px,
    const real*   __restrict__ py,
    const real*   __restrict__ pz,
    const uint8_t* __restrict__ status,
    const float*  __restrict__ psi1_buf,
    const float*  __restrict__ psi2_buf,
    float*         __restrict__ psi1_const,
    float*         __restrict__ psi2_const,
    float*         __restrict__ y_guess,
    float*         __restrict__ z_guess,
    int    N,
    int    nx, int ny, int nz,
    double dx, double dy, double dz,
    double Ly, double Lz)
{
    int p = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= N) return;
    if (status[p] != 0) return;  // skip inactive

    const double x = static_cast<double>(px[p]);
    const double y = static_cast<double>(py[p]);
    const double z = static_cast<double>(pz[p]);

    // ψ1 — self-period is Ly
    double dp1_dy, dp1_dz;
    const double p1_val = sample_psi_and_partials(
        psi1_buf, x, y, z, nx, ny, nz, dx, dy, dz,
        Ly, Lz, Ly, &dp1_dy, &dp1_dz);

    // ψ2 — self-period is Lz
    double dp2_dy, dp2_dz;
    const double p2_val = sample_psi_and_partials(
        psi2_buf, x, y, z, nx, ny, nz, dx, dy, dz,
        Ly, Lz, Lz, &dp2_dy, &dp2_dz);

    psi1_const[p] = static_cast<float>(p1_val);
    psi2_const[p] = static_cast<float>(p2_val);
    y_guess[p]    = static_cast<float>(y);
    z_guess[p]    = static_cast<float>(z);
}

// ============================================================================
// B.3  Kernel: pspta_step  (HOT PATH — no allocations)
// ============================================================================

/**
 * @brief Tier-2 PSPTA advance: RK2 in x + 3× Newton projection in (y,z).
 *
 * Summary per active particle:
 *   1. Project (y,z) at x using Newton → (y0,z0)
 *   2. Sample vx at (x,y0,z0); advance x_mid = x + 0.5*dt*vx
 *   3. Project at x_mid → (y_mid,z_mid); sample vx at midpoint
 *   4. Advance x_new = x + dt*vx_mid
 *   5. Project at x_new → (y_new,z_new)
 *   6. Wrap x_new; detect y,z periodic crossings via wrap_diff; update wrapX/Y/Z
 *   7. Commit positions; update y_guess/z_guess
 *
 * Newton failure fallback: keep particle at current (x,y,z) (conservative),
 * increment fail_count.  The "do-nothing" policy avoids introducing spurious
 * drift when projection is numerically unreliable.
 *
 * Thread layout: 1D over particles [0, N).
 */
__global__
void kernel_pspta_step(
    real*     __restrict__ px,
    real*     __restrict__ py,
    real*     __restrict__ pz,
    uint8_t*  __restrict__ status,
    int32_t*               wrapX,
    int32_t*               wrapY,
    int32_t*               wrapZ,
    const float* __restrict__ psi1_buf,
    const float* __restrict__ psi2_buf,
    const real*  __restrict__ U,   // CompactMAC x-faces
    const float* __restrict__ psi1_const,
    const float* __restrict__ psi2_const,
    float*       __restrict__ y_guess,
    float*       __restrict__ z_guess,
    uint32_t*    __restrict__ fail_count,
    int    N,
    int    nx, int ny, int nz,
    double dx, double dy, double dz,
    double Lx, double Ly, double Lz,
    double dt,
    bool   x_periodic)
{
    int p = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= N) return;
    if (status[p] != 0) return;  // skip inactive

    // Store last committed positions once — used for wrap accounting (PATCH 2).
    const double x     = static_cast<double>(px[p]);
    const double y_old = static_cast<double>(py[p]);
    const double z_old = static_cast<double>(pz[p]);

    const double pc1 = static_cast<double>(psi1_const[p]);
    const double pc2 = static_cast<double>(psi2_const[p]);
    const double yg  = static_cast<double>(y_guess[p]);
    const double zg  = static_cast<double>(z_guess[p]);

    // ── 1. Project at x ───────────────────────────────────────────────────
    double y0, z0;
    const bool ok0 = newton_solve_yz(
        psi1_buf, psi2_buf,
        x, pc1, pc2, yg, zg,
        nx, ny, nz, dx, dy, dz, Ly, Lz,
        &y0, &z0);

    if (!ok0) { fail_count[p]++; return; }

    // ── 2. Sample vx at (x, y0, z0); advance to midpoint ─────────────────
    const double vx0 = sample_vx(U, x, y0, z0, nx, ny, nz, dx, dy, dz);
    // Fix D: vx ≤ 0 or non-finite is a physical failure — do nothing.
    if (vx0 <= 0.0 || !isfinite(vx0)) { fail_count[p]++; return; }
    const double x_mid = x + 0.5 * dt * vx0;

    // PATCH 1 — OPEN_X: exit immediately if x_mid is outside [0, Lx).
    // Newton projection at x_mid would sample ψ with clamped x, giving
    // inconsistent results; treat the particle as exited instead.
    if (!x_periodic) {
        if (x_mid < 0.0 || x_mid >= Lx) {
            status[p] = PSPTA_STATUS_EXITED;
            return;
        }
    }

    // ── 3. Project at x_mid ───────────────────────────────────────────────
    double y_mid, z_mid;
    const bool ok_mid = newton_solve_yz(
        psi1_buf, psi2_buf,
        x_mid, pc1, pc2, y0, z0,
        nx, ny, nz, dx, dy, dz, Ly, Lz,
        &y_mid, &z_mid);

    // Fix B: any mid-Newton failure → do nothing (Policy 1: all-or-nothing).
    if (!ok_mid) { fail_count[p]++; return; }

    // ── 4. Sample vx at midpoint; advance to x_new ────────────────────────
    const double vx_mid = sample_vx(U, x_mid, y_mid, z_mid,
                                    nx, ny, nz, dx, dy, dz);
    // Fix D: vx_mid ≤ 0 or non-finite → do nothing.
    if (vx_mid <= 0.0 || !isfinite(vx_mid)) { fail_count[p]++; return; }
    const double x_new_raw = x + dt * vx_mid;

    // ── 5. Handle x boundary, then project at x_new ──────────────────────
    double x_new_w;
    if (x_periodic) {
        // Periodic x: wrap into [0, Lx)
        x_new_w = wrap_to_L(x_new_raw, Lx);
    } else {
        // Fix A — OPEN_X: exit particle if it leaves the domain
        if (x_new_raw < 0.0 || x_new_raw >= Lx) {
            status[p] = PSPTA_STATUS_EXITED;
            return;
        }
        x_new_w = x_new_raw;
    }
    double y_new, z_new;
    const bool ok_new = newton_solve_yz(
        psi1_buf, psi2_buf,
        x_new_w, pc1, pc2, y_mid, z_mid,
        nx, ny, nz, dx, dy, dz, Ly, Lz,
        &y_new, &z_new);

    // Fix B: final Newton failure → do nothing (all-or-nothing policy).
    if (!ok_new) { fail_count[p]++; return; }

    // ── 6. Update wrap counters ────────────────────────────────────────────

    // x: count full Lx crossings in raw advance
    const int cross_x = static_cast<int>(floor(x_new_raw / Lx));

    // PATCH 2 — y, z: compute net periodic displacement relative to the last
    // *committed* position (y_old, z_old), NOT the Newton guess (yg, zg).
    // wrap_diff gives the minimal-image displacement in (-L/2, L/2].
    // The unwrapped target is y_old + dy_net; the wrap increment is the
    // integer number of full periods between that and the wrapped y_new.
    const double dy_net = wrap_diff(y_new - y_old, Ly);
    const double dz_net = wrap_diff(z_new - z_old, Lz);

    const double y_unrestricted = y_old + dy_net;
    const double z_unrestricted = z_old + dz_net;

    // round() gives the exact integer winding number change (robust for |disp| < Ly/2).
    const int delta_wrapY = static_cast<int>(round((y_unrestricted - y_new) / Ly));
    const int delta_wrapZ = static_cast<int>(round((z_unrestricted - z_new) / Lz));

    // ── 7. Commit ─────────────────────────────────────────────────────────
    px[p] = static_cast<real>(x_new_w);
    py[p] = static_cast<real>(y_new);
    pz[p] = static_cast<real>(z_new);

    y_guess[p] = static_cast<float>(y_new);
    z_guess[p] = static_cast<float>(z_new);

    if (wrapX && cross_x != 0)      wrapX[p] += cross_x;
    if (wrapY && delta_wrapY != 0)  wrapY[p] += delta_wrapY;
    if (wrapZ && delta_wrapZ != 0)  wrapZ[p] += delta_wrapZ;
}

// ============================================================================
// B.4  Kernel: count_stats  (diagnostics — called after transport, not in hot loop)
// ============================================================================

/**
 * @brief Accumulate per-particle status and fail counts into 4 ULL atomics.
 *
 * out[0] = n_active   (status == 0)
 * out[1] = n_exited   (status == PSPTA_STATUS_EXITED)
 * out[2] = n_other    (any other non-zero status)
 * out[3] = total_fail (sum of fail_count[])
 */
__global__
void kernel_count_stats(
    const uint8_t*  __restrict__ status,
    const uint32_t* __restrict__ fail_count,
    unsigned long long* __restrict__ out,
    int N)
{
    int p = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= N) return;

    const uint8_t s = status[p];
    if (s == 0)                        atomicAdd(&out[0], 1ULL);
    else if (s == PSPTA_STATUS_EXITED) atomicAdd(&out[1], 1ULL);
    else                               atomicAdd(&out[2], 1ULL);

    atomicAdd(&out[3], static_cast<unsigned long long>(fail_count[p]));
}

/**
 * @brief Map a fail_count value to one of PSPTA_FAIL_HIST_BINS histogram bins.
 *   Bin 0: f==0   Bin 1: f==1   Bin 2: f==2   Bin 3: f in [3,4]
 *   Bin 4: f in [5,8]   Bin 5: f in [9,16]   Bin 6: f >= 17
 */
__device__ __forceinline__
int pspta_fail_bin(uint32_t f) {
    if (f == 0)  return 0;
    if (f == 1)  return 1;
    if (f == 2)  return 2;
    if (f <= 4)  return 3;
    if (f <= 8)  return 4;
    if (f <= 16) return 5;
    return 6;
}

/**
 * @brief Compute n_nonzero_fail, max_fail_count, and 7-bin histogram.
 *
 * out[0] = n_nonzero_fail (uint32)
 * out[1] = max_fail_count (uint32, via atomicMax)
 * out[2..8] = hist[0..6]  (uint32, atomic block reduce)
 */
__global__
void kernel_fail_details(
    const uint32_t* __restrict__ fail_count,
    uint32_t* __restrict__ out,
    int N)
{
    __shared__ uint32_t s_hist[PSPTA_FAIL_HIST_BINS];
    if (threadIdx.x < PSPTA_FAIL_HIST_BINS) s_hist[threadIdx.x] = 0u;
    __syncthreads();

    const int p = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p < N) {
        const uint32_t f = fail_count[p];
        if (f > 0u) atomicAdd(&out[0], 1u);
        atomicMax(&out[1], f);
        atomicAdd(&s_hist[pspta_fail_bin(f)], 1u);
    }
    __syncthreads();

    if (threadIdx.x < PSPTA_FAIL_HIST_BINS)
        atomicAdd(&out[2 + threadIdx.x], s_hist[threadIdx.x]);
}

// ============================================================================
// B.5  Kernel: compute_unwrapped
// ============================================================================

/**
 * @brief Compute unwrapped positions: pos_u = pos_wrapped + wrapCount * L.
 * Thread layout: 1D over particles.
 */
__global__
void kernel_compute_unwrapped(
    const real*    __restrict__ px,
    const real*    __restrict__ py,
    const real*    __restrict__ pz,
    const int32_t* __restrict__ wrapX,
    const int32_t* __restrict__ wrapY,
    const int32_t* __restrict__ wrapZ,
    real* __restrict__ ux,
    real* __restrict__ uy,
    real* __restrict__ uz,
    int    N,
    double Lx, double Ly, double Lz)
{
    int p = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= N) return;

    ux[p] = px[p] + static_cast<real>(wrapX ? wrapX[p] : 0) * static_cast<real>(Lx);
    uy[p] = py[p] + static_cast<real>(wrapY ? wrapY[p] : 0) * static_cast<real>(Ly);
    uz[p] = pz[p] + static_cast<real>(wrapZ ? wrapZ[p] : 0) * static_cast<real>(Lz);
}

// ============================================================================
// C  PsptaEngine — host methods
// ============================================================================

PsptaEngine::PsptaEngine(const Grid3D& grid,
                         cudaStream_t  stream,
                         uint64_t      inject_seed)
    : grid_(grid)
    , Lx_(static_cast<double>(grid.nx) * static_cast<double>(grid.dx))
    , Ly_(static_cast<double>(grid.ny) * static_cast<double>(grid.dy))
    , Lz_(static_cast<double>(grid.nz) * static_cast<double>(grid.dz))
    , stream_(stream)
    , inject_seed_(inject_seed)
{}

void PsptaEngine::bind_velocity(const VelocityField* vel) {
    vel_ = vel;
}

void PsptaEngine::bind_psifield(const PsptaPsiField* psi) {
    psi_ = psi;
}

void PsptaEngine::bind_particles(ParticlesSoA<real>& p) {
    parts_ = p;
}

void PsptaEngine::ensure_tracking() {
    // Wrap arrays are managed by the caller's ParticlesSoA.
    // No operation needed here (interface compatibility stub).
}

void PsptaEngine::inject_box(real x0, real y0, real z0,
                              real x1, real y1, real z1,
                              int first, int count)
{
    if (count <= 0) return;

    const real x_range = x1 - x0;
    const real y_range = y1 - y0;
    const real z_range = z1 - z0;

    const dim3 block(256);
    const dim3 grid((static_cast<unsigned>(count) + 255u) / 256u);

    kernel_inject_box<<<grid, block, 0, stream_>>>(
        parts_.x, parts_.y, parts_.z,
        parts_.status,
        parts_.wrapX, parts_.wrapY, parts_.wrapZ,
        first, count,
        x0, y0, z0,
        x_range, y_range, z_range,
        inject_seed_);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void PsptaEngine::prepare()
{
    const int N = parts_.n;
    if (N <= 0) return;

    // Reallocate per-particle buffers only if capacity insufficient
    if (N > capacity_) {
        d_psi1_const_.resize(static_cast<size_t>(N));
        d_psi2_const_.resize(static_cast<size_t>(N));
        d_y_guess_.resize(static_cast<size_t>(N));
        d_z_guess_.resize(static_cast<size_t>(N));
        d_fail_count_.resize(static_cast<size_t>(N));
        capacity_ = N;
    }

    // Zero fail counters
    MACROFLOW3D_CUDA_CHECK(
        cudaMemsetAsync(d_fail_count_.data(), 0,
                        static_cast<size_t>(N) * sizeof(uint32_t), stream_));

    // Launch invariant initialisation kernel
    const dim3 block(256);
    const dim3 grid_k((static_cast<unsigned>(N) + 255u) / 256u);

    kernel_init_invariants<<<grid_k, block, 0, stream_>>>(
        parts_.x, parts_.y, parts_.z, parts_.status,
        psi_->psi1_ptr(), psi_->psi2_ptr(),
        d_psi1_const_.data(), d_psi2_const_.data(),
        d_y_guess_.data(), d_z_guess_.data(),
        N,
        psi_->nx, psi_->ny, psi_->nz,
        psi_->dx, psi_->dy, psi_->dz,
        Ly_, Lz_);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

void PsptaEngine::step(real dt)
{
    const int N = parts_.n;
    if (N <= 0) return;

    const dim3 block(256);
    const dim3 grid_k((static_cast<unsigned>(N) + 255u) / 256u);

    kernel_pspta_step<<<grid_k, block, 0, stream_>>>(
        parts_.x, parts_.y, parts_.z,
        parts_.status,
        parts_.wrapX, parts_.wrapY, parts_.wrapZ,
        psi_->psi1_ptr(), psi_->psi2_ptr(),
        vel_->U.data(),
        d_psi1_const_.data(), d_psi2_const_.data(),
        d_y_guess_.data(), d_z_guess_.data(),
        d_fail_count_.data(),
        N,
        grid_.nx, grid_.ny, grid_.nz,
        static_cast<double>(grid_.dx),
        static_cast<double>(grid_.dy),
        static_cast<double>(grid_.dz),
        Lx_, Ly_, Lz_,
        static_cast<double>(dt),
        x_periodic_);
    // No MACROFLOW3D_CUDA_CHECK here to keep the path non-synchronizing in release.
    // The pipeline's single sync point handles error detection.
}

void PsptaEngine::synchronize() {
    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream_));
}

ConstParticlesSoA<real> PsptaEngine::particles() const {
    ConstParticlesSoA<real> cv;
    cv.x      = parts_.x;
    cv.y      = parts_.y;
    cv.z      = parts_.z;
    cv.n      = parts_.n;
    cv.status = parts_.status;
    cv.wrapX  = parts_.wrapX;
    cv.wrapY  = parts_.wrapY;
    cv.wrapZ  = parts_.wrapZ;
    return cv;
}

void PsptaEngine::compute_unwrapped(UnwrappedSoA<real>& uw, cudaStream_t stream)
{
    const int N = parts_.n;
    if (!uw.valid() || N <= 0) return;

    const dim3 block(256);
    const dim3 grid_k((static_cast<unsigned>(N) + 255u) / 256u);

    kernel_compute_unwrapped<<<grid_k, block, 0, stream>>>(
        parts_.x, parts_.y, parts_.z,
        parts_.wrapX, parts_.wrapY, parts_.wrapZ,
        uw.x_u, uw.y_u, uw.z_u,
        N,
        Lx_, Ly_, Lz_);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());
}

PsptaEngine::TransportStats PsptaEngine::compute_transport_stats()
{
    const int N = parts_.n;
    TransportStats ts;
    if (N <= 0) return ts;

    const dim3 block(256);
    const dim3 grid_k((static_cast<unsigned>(N) + 255u) / 256u);

    // ── Pass 1: status + total_fail (ULL) ────────────────────────────────────
    if (d_stats_buf_.size() < 4) d_stats_buf_.resize(4);
    MACROFLOW3D_CUDA_CHECK(
        cudaMemsetAsync(d_stats_buf_.data(), 0, 4 * sizeof(unsigned long long), stream_));

    kernel_count_stats<<<grid_k, block, 0, stream_>>>(
        parts_.status, d_fail_count_.data(), d_stats_buf_.data(), N);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    unsigned long long host_buf[4] = {0, 0, 0, 0};
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpyAsync(host_buf, d_stats_buf_.data(),
                        4 * sizeof(unsigned long long),
                        cudaMemcpyDeviceToHost, stream_));

    // ── Pass 2: n_nonzero_fail + max_fail + 7-bin histogram (uint32) ─────────
    constexpr int DETAIL_SIZE = 2 + PSPTA_FAIL_HIST_BINS; // 9
    if (d_fail_detail_buf_.size() < DETAIL_SIZE) d_fail_detail_buf_.resize(DETAIL_SIZE);
    MACROFLOW3D_CUDA_CHECK(
        cudaMemsetAsync(d_fail_detail_buf_.data(), 0,
                        DETAIL_SIZE * sizeof(uint32_t), stream_));

    kernel_fail_details<<<grid_k, block, 0, stream_>>>(
        d_fail_count_.data(), d_fail_detail_buf_.data(), N);
    MACROFLOW3D_CUDA_CHECK(cudaGetLastError());

    uint32_t detail_buf[DETAIL_SIZE] = {};
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpyAsync(detail_buf, d_fail_detail_buf_.data(),
                        DETAIL_SIZE * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost, stream_));

    MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize(stream_));

    // ── Populate result ───────────────────────────────────────────────────────
    ts.n_active       = static_cast<int>(host_buf[0]);
    ts.n_exited       = static_cast<int>(host_buf[1]);
    ts.n_other        = static_cast<int>(host_buf[2]);
    ts.total_fail     = static_cast<long long>(host_buf[3]);
    ts.n_nonzero_fail = detail_buf[0];
    ts.max_fail_count = detail_buf[1];
    for (int b = 0; b < PSPTA_FAIL_HIST_BINS; ++b)
        ts.hist[b] = detail_buf[2 + b];
    return ts;
}

} // namespace pspta
} // namespace particles
} // namespace physics
} // namespace macroflow3d
