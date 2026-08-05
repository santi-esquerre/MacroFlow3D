#pragma once

#include "../../core/DeviceSpan.cuh"
#include "../../core/Scalar.hpp"

#include <type_traits>

namespace macroflow3d {
namespace streamfunctions {

// Constant gradient of an affine streamfunction component.
struct AffineGradient {
    real x;
    real y;
    real z;
};

// Gauge constants for the affine parts only.  The corresponding scalar
// streamfunctions are psi1 = vbar*x2 + u1 and psi2 = x3 + u2, where
// mean(ui) = 0.  The affine scalars are never evaluated with wrapped
// coordinates; periodic wrapping applies only to the stored fluctuations.
struct AffineGauge {
    AffineGradient psi1_gradient{real{0}, real{1}, real{0}};
    AffineGradient psi2_gradient{real{0}, real{0}, real{1}};

    static constexpr AffineGauge benchmark(real vbar) {
        return AffineGauge{{real{0}, vbar, real{0}}, {real{0}, real{0}, real{1}}};
    }
};

// Caller-owned, periodic cell-centered fluctuation-only storage.  u1 and u2
// represent psi1_tilde and psi2_tilde respectively; neither span stores an
// affine ramp or owns/allocates device memory.
struct PeriodicStreamfunctionFluctuations {
    DeviceSpan<real> u1;
    DeviceSpan<real> u2;
};

static_assert(std::is_trivially_copyable<AffineGradient>::value,
              "AffineGradient must be safe to copy into a kernel argument");
static_assert(std::is_trivially_copyable<AffineGauge>::value,
              "AffineGauge must be safe to copy into a kernel argument");
static_assert(std::is_trivially_copyable<PeriodicStreamfunctionFluctuations>::value,
              "PeriodicStreamfunctionFluctuations must be safe to copy into a kernel argument");

} // namespace streamfunctions
} // namespace macroflow3d
