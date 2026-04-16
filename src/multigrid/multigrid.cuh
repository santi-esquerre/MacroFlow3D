#pragma once

// Multigrid conventions (mathematical definitions)
#include "common/mg_conventions.cuh"

// Multigrid types
#include "mg_types.hpp"

// Transfer operators
#include "transfer/prolong_3d.cuh"
#include "transfer/restrict_3d.cuh"

// Smoothers
#include "smoothers/gsrb_3d.cuh"
#include "smoothers/residual_3d.cuh"

// V-cycle
#include "cycle/v_cycle.cuh"

// Note: No namespace re-export to avoid pollution.
// Use macroflow3d::multigrid::* explicitly.
