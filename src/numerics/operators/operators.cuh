#pragma once

// Operator concept (template-based, no virtual dispatch)
#include "operator_concept.cuh"

// Operators for MacroFlow3D

// Constant-coefficient Laplacian (simple, K=1 everywhere)
#include "poisson3d_operator.cuh"

// Variable-coefficient Laplacian (same semantics as MG smoother/residual)
// Use this for CG solve to compare directly with MG solve
#include "varcoeff_laplacian.cuh"

// Negated operator wrapper (converts negative-definite to positive-definite)
// Essential for using CG/PCG with our negative Laplacian
#include "negated_operator.cuh"
