#pragma once

/**
 * @file physics_types.cuh
 * @brief Aggregated include for all physics types
 * 
 * Include this single header to get all physics data structures:
 * - Configuration structs (StochasticConfig, FlowConfig, TransportConfig)
 * - Field types (KField, HeadField, VelocityField)
 * - Workspace types (StochasticWorkspace, FlowWorkspace, ParticlesWorkspace)
 */

#include "physics_config.hpp"
#include "fields.cuh"
#include "workspaces.cuh"

namespace macroflow3d {
namespace physics {

// Version tag for physics module
constexpr int PHYSICS_VERSION_MAJOR = 5;
constexpr int PHYSICS_VERSION_MINOR = 0;

} // namespace physics
} // namespace macroflow3d
