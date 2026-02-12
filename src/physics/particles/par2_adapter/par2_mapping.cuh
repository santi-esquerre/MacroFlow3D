#pragma once

/**
 * @file par2_mapping.cuh
 * @brief Internal Par2↔MacroFlow3D mapping helpers (grid, BC, velocity view)
 *
 * This file MUST only be included from .cu files inside par2_adapter/.
 * It is the single point where <par2_core/par2_core.hpp> is included for
 * mapping between MacroFlow3D types and Par2_Core types.
 */

#include <par2_core/par2_core.hpp>
#include "par2_views.hpp"
#include "../../common/fields.cuh"
#include "../../../core/BCSpec.hpp"
#include "../../../core/Grid3D.hpp"
#include "../../../core/Scalar.hpp"

namespace macroflow3d {
namespace physics {
namespace particles {
namespace detail {

// ============================================================================
// BC mapping
// ============================================================================

inline par2::BoundaryType to_par2_bc(BCType t) {
    switch (t) {
        case BCType::Neumann:   return par2::BoundaryType::Closed;
        case BCType::Periodic:  return par2::BoundaryType::Periodic;
        case BCType::Dirichlet: return par2::BoundaryType::Closed;
        default:                return par2::BoundaryType::Closed;
    }
}

inline par2::BoundaryConfig<real> make_par2_bc(const BCSpec& bc) {
    par2::BoundaryConfig<real> cfg;
    cfg.x.lo = to_par2_bc(bc.xmin.type);
    cfg.x.hi = to_par2_bc(bc.xmax.type);
    cfg.y.lo = to_par2_bc(bc.ymin.type);
    cfg.y.hi = to_par2_bc(bc.ymax.type);
    cfg.z.lo = to_par2_bc(bc.zmin.type);
    cfg.z.hi = to_par2_bc(bc.zmax.type);
    return cfg;
}

// ============================================================================
// Grid mapping
// ============================================================================

inline par2::GridDesc<real> make_par2_grid(const Grid3D& grid) {
    return par2::make_grid<real>(
        grid.nx, grid.ny, grid.nz,
        grid.dx, grid.dy, grid.dz);
}

// ============================================================================
// Velocity view (zero-copy from PaddedVelocityField)
// ============================================================================

inline par2::VelocityView<real> make_velocity_view(const PaddedVelocityField& vel) {
    par2::VelocityView<real> v;
    v.U    = vel.U_ptr();
    v.V    = vel.V_ptr();
    v.W    = vel.W_ptr();
    v.size = vel.field_size();
    return v;
}

// ============================================================================
// View conversions  (MacroFlow3D ↔ Par2_Core, zero-copy pointer remaps)
// ============================================================================

inline par2::ParticlesView<real> to_par2(ParticlesSoA<real>& p) {
    par2::ParticlesView<real> pv;
    pv.x      = p.x;
    pv.y      = p.y;
    pv.z      = p.z;
    pv.n      = p.n;
    pv.status = p.status;
    pv.wrapX  = p.wrapX;
    pv.wrapY  = p.wrapY;
    pv.wrapZ  = p.wrapZ;
    return pv;
}

inline ConstParticlesSoA<real> from_par2(const par2::ConstParticlesView<real>& cpv) {
    ConstParticlesSoA<real> out;
    out.x      = cpv.x;
    out.y      = cpv.y;
    out.z      = cpv.z;
    out.n      = cpv.n;
    out.status = cpv.status;
    out.wrapX  = cpv.wrapX;
    out.wrapY  = cpv.wrapY;
    out.wrapZ  = cpv.wrapZ;
    return out;
}

inline par2::ConstParticlesView<real> to_par2_const(const ConstParticlesSoA<real>& p) {
    par2::ConstParticlesView<real> cpv;
    cpv.x      = p.x;
    cpv.y      = p.y;
    cpv.z      = p.z;
    cpv.n      = p.n;
    cpv.status = p.status;
    cpv.wrapX  = p.wrapX;
    cpv.wrapY  = p.wrapY;
    cpv.wrapZ  = p.wrapZ;
    return cpv;
}

inline par2::UnwrappedPositionsView<real> to_par2(UnwrappedSoA<real>& uw) {
    par2::UnwrappedPositionsView<real> v;
    v.x_u      = uw.x_u;
    v.y_u      = uw.y_u;
    v.z_u      = uw.z_u;
    v.capacity = uw.capacity;
    return v;
}

inline par2::io::CsvSnapshotConfig to_par2(const SnapshotWriterConfig& c) {
    par2::io::CsvSnapshotConfig out;
    out.legacy_format       = c.legacy_format;
    out.include_time        = c.include_time;
    out.include_status      = c.include_status;
    out.include_wrap_counts = c.include_wrap_counts;
    out.include_unwrapped   = c.include_unwrapped;
    out.stride              = c.stride;
    out.max_particles       = c.max_particles;
    out.precision           = c.precision;
    return out;
}

} // namespace detail
} // namespace particles
} // namespace physics
} // namespace macroflow3d
