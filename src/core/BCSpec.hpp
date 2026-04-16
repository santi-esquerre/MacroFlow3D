#pragma once

#include "Scalar.hpp"
#include <cmath>
#include <stdexcept>

namespace macroflow3d {

/**
 * @brief Boundary condition type for a single face.
 * @ingroup core
 */
enum class BCType { Dirichlet, Neumann, Periodic };

/**
 * @brief Per-face boundary condition (type + value).
 * @ingroup core
 */
struct BCFace {
    BCType type;
    real value;

    // Default: Dirichlet with zero value
    BCFace() : type(BCType::Dirichlet), value(0.0) {}

    BCFace(BCType t, real v) : type(t), value(v) {}
};

/**
 * @brief Full boundary condition specification for all 6 faces.
 * @ingroup core
 */
struct BCSpec {
    BCFace xmin, xmax;
    BCFace ymin, ymax;
    BCFace zmin, zmax;

    // Default constructor: all faces Dirichlet with zero value
    BCSpec()
        : xmin(BCType::Dirichlet, 0.0), xmax(BCType::Dirichlet, 0.0), ymin(BCType::Dirichlet, 0.0),
          ymax(BCType::Dirichlet, 0.0), zmin(BCType::Dirichlet, 0.0), zmax(BCType::Dirichlet, 0.0) {
    }

    // Query helpers
    bool is_periodic_x() const {
        return xmin.type == BCType::Periodic && xmax.type == BCType::Periodic;
    }
    bool is_periodic_y() const {
        return ymin.type == BCType::Periodic && ymax.type == BCType::Periodic;
    }
    bool is_periodic_z() const {
        return zmin.type == BCType::Periodic && zmax.type == BCType::Periodic;
    }

    bool is_all_homog_neumann() const {
        return xmin.type == BCType::Neumann && xmin.value == 0.0 && xmax.type == BCType::Neumann &&
               xmax.value == 0.0 && ymin.type == BCType::Neumann && ymin.value == 0.0 &&
               ymax.type == BCType::Neumann && ymax.value == 0.0 && zmin.type == BCType::Neumann &&
               zmin.value == 0.0 && zmax.type == BCType::Neumann && zmax.value == 0.0;
    }

    // Validation
    void validate() const {
        // Periodic must come in pairs
        if (xmin.type == BCType::Periodic || xmax.type == BCType::Periodic) {
            if (xmin.type != BCType::Periodic || xmax.type != BCType::Periodic) {
                throw std::runtime_error("Periodic BC must be specified on both xmin and xmax");
            }
        }
        if (ymin.type == BCType::Periodic || ymax.type == BCType::Periodic) {
            if (ymin.type != BCType::Periodic || ymax.type != BCType::Periodic) {
                throw std::runtime_error("Periodic BC must be specified on both ymin and ymax");
            }
        }
        if (zmin.type == BCType::Periodic || zmax.type == BCType::Periodic) {
            if (zmin.type != BCType::Periodic || zmax.type != BCType::Periodic) {
                throw std::runtime_error("Periodic BC must be specified on both zmin and zmax");
            }
        }

        // Values must be finite
        auto is_finite = [](real v) { return v == v && v != INFINITY && v != -INFINITY; };
        if (!is_finite(xmin.value) || !is_finite(xmax.value) || !is_finite(ymin.value) ||
            !is_finite(ymax.value) || !is_finite(zmin.value) || !is_finite(zmax.value)) {
            throw std::runtime_error("BC values must be finite");
        }
    }
};

} // namespace macroflow3d
