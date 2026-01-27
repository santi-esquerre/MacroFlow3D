#pragma once

#include "Scalar.hpp"

namespace rwpt {

enum class BCType {
    Dirichlet,
    Neumann,
    Periodic
};

struct BCFace {
    BCType type;
    real value;

    // Default: Dirichlet with zero value
    BCFace() : type(BCType::Dirichlet), value(0.0) {}
    
    BCFace(BCType t, real v) : type(t), value(v) {}
};

struct BCSpec {
    BCFace xmin, xmax;
    BCFace ymin, ymax;
    BCFace zmin, zmax;

    // Default constructor: all faces Dirichlet with zero value
    BCSpec()
        : xmin(BCType::Dirichlet, 0.0),
          xmax(BCType::Dirichlet, 0.0),
          ymin(BCType::Dirichlet, 0.0),
          ymax(BCType::Dirichlet, 0.0),
          zmin(BCType::Dirichlet, 0.0),
          zmax(BCType::Dirichlet, 0.0) {}
};

} // namespace rwpt
