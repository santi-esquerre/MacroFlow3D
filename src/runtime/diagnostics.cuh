/**
 * @file diagnostics.cuh
 * @brief Compile-time flags for debug vs release behavior
 * 
 * This header centralizes all diagnostic/debug flags for the MacroFlow3D codebase.
 * In Release builds, these flags default to OFF to ensure zero-overhead.
 * 
 * ## Usage
 * 
 * Include this header and use the macros to guard expensive operations:
 * 
 *   #include "runtime/diagnostics.cuh"
 *   
 *   // Expensive sync + print
 *   MACROFLOW3D_DEBUG_SYNC(ctx);  // No-op in Release
 *   MACROFLOW3D_DEBUG_LOG("value = %f", val);  // Compiled out in Release
 *   
 *   // Field statistics (allocates + syncs)
 *   if (MACROFLOW3D_DIAGNOSTICS_ENABLED) {
 *       compute_field_stats(...);
 *   }
 * 
 * ## CMake Configuration
 * 
 * Add to CMakeLists.txt for Debug builds:
 *   target_compile_definitions(macroflow3d_lib PUBLIC MACROFLOW3D_ENABLE_DIAGNOSTICS=1)
 * 
 * For Release:
 *   target_compile_definitions(macroflow3d_lib PUBLIC MACROFLOW3D_ENABLE_DIAGNOSTICS=0)
 * 
 * ## Flags
 * 
 * - MACROFLOW3D_ENABLE_DIAGNOSTICS: Master switch for diagnostic code paths
 * - MACROFLOW3D_ENABLE_SYNC_CHECKS: Extra synchronizations for debugging
 * - MACROFLOW3D_ENABLE_NVTX: NVTX markers for Nsight Systems profiling
 * - MACROFLOW3D_ENABLE_PROFILING: Enable stage-by-stage timing
 */

#pragma once

#include "cuda_check.cuh"
#include <cstdio>

// ============================================================================
// Default flag values (can be overridden via -D at compile time)
// ============================================================================

// Master diagnostic switch: controls prints, sync checks, and stat dumps
#ifndef MACROFLOW3D_ENABLE_DIAGNOSTICS
#  ifdef NDEBUG
#    define MACROFLOW3D_ENABLE_DIAGNOSTICS 0
#  else
#    define MACROFLOW3D_ENABLE_DIAGNOSTICS 1
#  endif
#endif

// Extra synchronizations for debugging (catches async errors early)
#ifndef MACROFLOW3D_ENABLE_SYNC_CHECKS
#  define MACROFLOW3D_ENABLE_SYNC_CHECKS MACROFLOW3D_ENABLE_DIAGNOSTICS
#endif

// NVTX markers for Nsight Systems (requires linking -lnvToolsExt)
#ifndef MACROFLOW3D_ENABLE_NVTX
#  define MACROFLOW3D_ENABLE_NVTX 0
#endif

// Stage-by-stage profiling (event timers)
#ifndef MACROFLOW3D_ENABLE_PROFILING
#  define MACROFLOW3D_ENABLE_PROFILING 0
#endif

// ============================================================================
// Compile-time constants (for if-constexpr style guards)
// ============================================================================

namespace macroflow3d {
namespace diag {

static constexpr bool diagnostics_enabled = (MACROFLOW3D_ENABLE_DIAGNOSTICS != 0);
static constexpr bool sync_checks_enabled = (MACROFLOW3D_ENABLE_SYNC_CHECKS != 0);
static constexpr bool nvtx_enabled = (MACROFLOW3D_ENABLE_NVTX != 0);
static constexpr bool profiling_enabled = (MACROFLOW3D_ENABLE_PROFILING != 0);

} // namespace diag
} // namespace macroflow3d

// Alias for use in if statements (optimizer removes dead branches)
#define MACROFLOW3D_DIAGNOSTICS_ENABLED ::macroflow3d::diag::diagnostics_enabled
#define MACROFLOW3D_SYNC_CHECKS_ENABLED ::macroflow3d::diag::sync_checks_enabled
#define MACROFLOW3D_PROFILING_ENABLED   ::macroflow3d::diag::profiling_enabled

// ============================================================================
// Debug logging macro (compiled out in Release)
// ============================================================================

#if MACROFLOW3D_ENABLE_DIAGNOSTICS
#  define MACROFLOW3D_DEBUG_LOG(fmt, ...) \
      do { std::fprintf(stderr, "[RWPT DEBUG] " fmt "\n", ##__VA_ARGS__); } while(0)
#else
#  define MACROFLOW3D_DEBUG_LOG(fmt, ...) do {} while(0)
#endif

// ============================================================================
// Debug synchronization macro (compiled out in Release)
// ============================================================================

#if MACROFLOW3D_ENABLE_SYNC_CHECKS
#  define MACROFLOW3D_DEBUG_SYNC(ctx) \
      do { \
          MACROFLOW3D_CUDA_CHECK(cudaStreamSynchronize((ctx).cuda_stream())); \
          MACROFLOW3D_CUDA_CHECK(cudaGetLastError()); \
      } while(0)
#else
#  define MACROFLOW3D_DEBUG_SYNC(ctx) do {} while(0)
#endif

// ============================================================================
// NVTX range helper (include nvtx_range.cuh for full functionality)
// ============================================================================

#if MACROFLOW3D_ENABLE_NVTX
#  include "nvtx_range.cuh"
#  define MACROFLOW3D_NVTX_PUSH(name) MACROFLOW3D_NVTX_RANGE(name)
#else
#  define MACROFLOW3D_NVTX_PUSH(name) do {} while(0)
#endif

