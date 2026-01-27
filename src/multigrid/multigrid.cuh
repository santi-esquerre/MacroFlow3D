#pragma once

// Multigrid types
#include "mg_types.hpp"

// Transfer operators
#include "transfer/restrict_3d.cuh"
#include "transfer/prolong_3d.cuh"

// Smoothers
#include "smoothers/residual_3d.cuh"
#include "smoothers/gsrb_3d.cuh"

// V-cycle
#include "cycle/v_cycle.cuh"

// Note: No namespace re-export to avoid pollution.
// Use rwpt::multigrid::* explicitly.
