# Geometry/CartesianGrid.cuh

```cuh
/**
 * @file CellVariable.cuh
 * @brief Header file for grid.
 *        Data stuctures and functions of a Cartesian grid
 *
 * @author Calogero B. Rizzo
 *
 * @copyright This file is part of the PAR2 software.
 *            Copyright (C) 2018 Calogero B. Rizzo
 *
 * @license This program is free software: you can redistribute it and/or modify
 *          it under the terms of the GNU General Public License as published by
 *          the Free Software Foundation, either version 3 of the License, or
 *          (at your option) any later version.
 *
 *          This program is distributed in the hope that it will be useful,
 *          but WITHOUT ANY WARRANTY; without even the implied warranty of
 *          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *          GNU General Public License for more details.
 *
 *          You should have received a copy of the GNU General Public License
 *          along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

#ifndef PAR2_CARTESIANGRID_CUH
#define PAR2_CARTESIANGRID_CUH

#include "Point.cuh"
#include <stddef.h>

namespace par2 {

namespace grid {
/**
 * @struct Grid
 * @brief Struct containing the Cartesian grid parameters.
 * @tparam T Float number precision
 */
template <typename T> struct Grid {
  int nx, ny, nz;
  T dx, dy, dz;
  T px, py, pz;
  T xmax, ymax, zmax;
  T xmin = px;
  T ymin = py;
  T zmin = pz;


  Grid(){};

  __host__ __device__ Grid(const Grid &g)
      : nx(g.nx), ny(g.ny), nz(g.nz), dx(g.dx), dy(g.dy), dz(g.dz), px(g.px),
        py(g.py), pz(g.pz){};
};

/**
 * @brief Build the grid.
 * @param _nx Number of cells along x
 * @param _ny Number of cells along y
 * @param _nz Number of cells along z
 * @param _dx Size of cell along x
 * @param _dy Size of cell along y
 * @param _dz Size of cell along z
 * @param _px Origin along x
 * @param _py Origin along y
 * @param _pz Origin along z
 * @tparam T Float number precision
 */
template <typename T>
Grid<T> build(int _nx, int _ny, int _nz, T _dx, T _dy, T _dz, T _px = 0,
              T _py = 0, T _pz = 0) {
  Grid<T> g;
  g.nx = _nx;
  g.ny = _ny;
  g.nz = _nz;
  g.dx = _dx;
  g.dy = _dy;
  g.dz = _dz;
  g.px = _px;
  g.py = _py;
  g.pz = _pz;
  g.xmax = g.px + g.nx * g.dx;
  g.ymax = g.py + g.ny * g.dy;
  g.zmax = g.pz + g.nz * g.dz;
  g.xmin = _px;
  g.ymin = _py;
  g.zmin = _pz;
  return g;
}

/**
 * @brief Check if x is inside the grid.
 * @param g Grid parameters
 * @param x Coordinate
 * @tparam T Float number precision
 * @return True if x is inside the grid
 */
template <typename T>
__host__ __device__ bool validX(const Grid<T> &g, const T x) {
  return g.px < x && x < g.px + g.dx * g.nx;
}

/**
 * @brief Check if y is inside the grid.
 * @param g Grid parameters
 * @param y Coordinate
 * @tparam T Float number precision
 * @return True if y is inside the grid
 */
template <typename T>
__host__ __device__ bool validY(const Grid<T> &g, const T y) {
  return g.py < y && y < g.py + g.dy * g.ny;
}

/**
 * @brief Check if z is inside the grid.
 * @param g Grid parameters
 * @param z Coordinate
 * @tparam T Float number precision
 * @return True if z is inside the grid
 */
template <typename T>
__host__ __device__ bool validZ(const Grid<T> &g, const T z) {
  return g.pz < z && z < g.pz + g.dz * g.nz;
}

/**
 * @brief Get the total number of cells.
 * @param g Grid parameters
 * @tparam T Float number precision
 * @return Number of cells
 */
template <typename T> __host__ __device__ int numberOfCells(const Grid<T> &g) {
  return g.nx * g.ny * g.nz;
}

/**
 * @brief Get the total number of faces.
 * @param g Grid parameters
 * @tparam T Float number precision
 * @return Number of faces
 */
template <typename T> __host__ __device__ int numberOfFaces(const Grid<T> &g) {
  return g.nx * g.ny * g.nz + g.nx * g.ny + g.ny * g.nz + g.nx * g.nz;
}

/**
 * @brief Get the ID of the cell containing at given position
 * @param g Grid parameters
 * @param px Position x-coordinate
 * @param py Position y-coordinate
 * @param pz Position z-coordinate
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @tparam T Float number precision
 */
template <typename T>
__host__ __device__ void idPoint(const Grid<T> &g, T px, T py, T pz, int *idx,
                                 int *idy, int *idz) {
  *idx = floor((px - g.px) / g.dx);
  *idy = floor((py - g.py) / g.dy);
  *idz = floor((pz - g.pz) / g.dz);
}

/**
 * @brief Check if a given ID is valid (3 Ids)
 * @param g Grid parameters
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @tparam T Float number precision
 * @return True if the ID is valid
 */
template <typename T>
__host__ __device__ bool validId(const Grid<T> &g, int idx, int idy, int idz) {
  return idx >= 0 && idx < g.nx && idy >= 0 && idy < g.ny && idz >= 0 &&
         idz < g.nz;
}

/**
 * @brief Check if a given ID is valid (unique Id)
 * @param g Grid parameters
 * @param id Unique ID
 * @tparam T Float number precision
 * @return True if the ID is valid
 */
template <typename T>
__host__ __device__ bool validId(const Grid<T> &g, int id) {
  return id >= 0 && id < numberOfCells(g);
}

/**
 * @brief Given the unique ID of a cell, find the corresponding
 *        IDs along each principal direction.
 * @param g Grid parameters
 * @param id Unique ID
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @tparam T Float number precision
 */
template <typename T>
__host__ __device__ void splitId(const Grid<T> &g, int id, int *idx, int *idy,
                                 int *idz) {
  *idz = id / (g.nx * g.ny);
  id = id % (g.nx * g.ny);
  *idy = id / g.nx;
  id = id % g.nx;
  *idx = id;
}

/**
 * @brief Given the IDs along each principal direction,
 *        find the corresponding unique ID of a cell without
 *        performing validation.
 * @param g Grid parameters
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @tparam T Float number precision
 * @return Unique ID
 */
template <typename T>
__host__ __device__ int mergeId(const Grid<T> &g, int idx, int idy, int idz) {
  return idz * g.ny * g.nx + idy * g.nx + idx;
}

/**
 * @brief Given the IDs along each principal direction,
 *        find the corresponding unique ID of a cell
 *        performing validation. If the IDs are not valid,
 *        return the total number of cells.
 * @param g Grid parameters
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @tparam T Float number precision
 * @return Unique ID
 */
template <typename T>
__host__ __device__ int uniqueId(const Grid<T> &g, int idx, int idy, int idz) {
  return validId(g, idx, idy, idz) ? mergeId(g, idx, idy, idz)
                                   : numberOfCells(g);
}

/**
 * @brief Compute the volume of a single cell.
 * @param g Grid parameters
 * @tparam T Float number precision
 * @return Volume of a cell
 */
template <typename T> __host__ __device__ T volumeCell(const Grid<T> &g) {
  return g.dx * g.dy * g.dz;
}

/**
 * @brief Compute the volume of the whole grid.
 * @param g Grid parameters
 * @tparam T Float number precision
 * @return Volume of the grid
 */
template <typename T> __host__ __device__ T volumeGrid(const Grid<T> &g) {
  return volumeCell(g) * numberOfCells(g);
}

/**
 * @brief Compute the center point of a cell.
 * @param g Grid parameters
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @param px Center x-coordinate
 * @param py Center y-coordinate
 * @param pz Center z-coordinate
 * @tparam T Float number precision
 * @return Volume of the grid
 */
template <typename T>
__host__ __device__ void centerOfCell(const Grid<T> &g, int idx, int idy,
                                      int idz, T *px, T *py, T *pz) {
  *px = g.px + idx * g.dx + 0.5 * g.dx;
  *py = g.py + idy * g.dy + 0.5 * g.dy;
  *pz = g.pz + idz * g.dz + 0.5 * g.dz;
}

/**
 * @brief Direction needed to find the neighbors cells.
 */
enum Direction {
  XP = 0, // XP: cell increasing x
  XM,     // XM: cell decreasing x
  YP,
  YM,
  ZP,
  ZM
};

/**
 * @brief Find the ID of a neighbor cell.
 * @param g Grid parameters
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @tparam T Float number precision
 * @tparam dir Direction of the neighbor
 * @return Volume of the grid
 */
template <typename T, int dir>
__host__ __device__ int idNeighbor(const Grid<T> &g, int idx, int idy,
                                   int idz) {
  switch (dir) {
  case XP:
    return (idx != g.nx - 1) ? mergeId(g, idx + 1, idy, idz) : numberOfCells(g);
  case XM:
    return (idx != 0) ? mergeId(g, idx - 1, idy, idz) : numberOfCells(g);
  case YP:
    return (idy != g.ny - 1) ? mergeId(g, idx, idy + 1, idz) : numberOfCells(g);
  case YM:
    return (idy != 0) ? mergeId(g, idx, idy - 1, idz) : numberOfCells(g);
  case ZP:
    return (idz != g.nz - 1) ? mergeId(g, idx, idy, idz + 1) : numberOfCells(g);
  case ZM:
    return (idz != 0) ? mergeId(g, idx, idy, idz - 1) : numberOfCells(g);
  }
  return numberOfCells(g);
}

/**
 * @brief Find the cell containing a given point and check if it's valid
 * @param g Grid parameters
 * @param px Position x-coordinate
 * @param py Position y-coordinate
 * @param pz Position z-coordinate
 * @param idx Output: ID along x
 * @param idy Output: ID along y
 * @param idz Output: ID along z
 * @tparam T Float number precision
 * @return True if the point is inside a valid cell
 */
template <typename T>
__host__ __device__ bool findCell(const Grid<T> &g, T px, T py, T pz, int *idx,
                                  int *idy, int *idz) {
  // Calcular los índices de la celda
  idPoint(g, px, py, pz, idx, idy, idz);

  // Verificar si los índices son válidos
  return validId(g, *idx, *idy, *idz);
}
}; // namespace grid

} // namespace par2

#endif // PAR2_CARTESIANGRID_CUH

```

# Geometry/CellField.cuh

```cuh
/**
 * @file CellField.cuh
 * @brief Header file for cellfield.
 *        A cellfield is a field that is defined at the center of
 *        every cells of the grid.
 *
 * @author Calogero B. Rizzo
 *
 * @copyright This file is part of the PAR2 software.
 *            Copyright (C) 2018 Calogero B. Rizzo
 *
 * @license This program is free software: you can redistribute it and/or modify
 *          it under the terms of the GNU General Public License as published by
 *          the Free Software Foundation, either version 3 of the License, or
 *          (at your option) any later version.
 *
 *          This program is distributed in the hope that it will be useful,
 *          but WITHOUT ANY WARRANTY; without even the implied warranty of
 *          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *          GNU General Public License for more details.
 *
 *          You should have received a copy of the GNU General Public License
 *          along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

#ifndef PAR2_CELLFIELD_CUH
#define PAR2_CELLFIELD_CUH

#include <algorithm>
#include <fstream>
#include <iostream>
#include <thrust/device_vector.h>
// #include <thrust/raw_pointer_cast.h>

#include "CartesianGrid.cuh"
#include "FaceField.cuh"
#include "Vector.cuh"

namespace par2 {

namespace cellfield {
/**
 * @brief Build the vector containing a cellfield.
 * @param g Grid where the field is defined
 * @param data Vector that contains the cellfield values
 * @param v Init value for data
 * @tparam T Float number precision
 * @tparam Vector Container for data vectors
 */
template <typename T, class Vector>
void build(const grid::Grid<T> &g, Vector &data, T v = 0) {
  int size = g.nx * g.ny * g.nz;
  data.resize(size, v);
}

/**
 * @brief Compute drift correction in each cell.
 * @param g Grid where the field is defined
 * @param datax Vector that contains the values on the interfaces
 *        orthogonal to the x-axis
 * @param datay Vector that contains the values on the interfaces
 *        orthogonal to the y-axis
 * @param dataz Vector that contains the values on the interfaces
 *        orthogonal to the z-axis
 * @param driftx Vector that contains the x-values of the correction
 * @param drifty Vector that contains the y-values of the correction
 * @param driftz Vector that contains the z-values of the correction
 * @param Dm Effective molecular diffusion
 * @param alphaL Longitudinal dispersivity
 * @param alphaT Transverse dispersivity
 * @tparam T Float number precision
 * @tparam Vector Container for data vectors
 */
template <typename T, class Vector>
void computeDriftCorrection(const grid::Grid<T> &g, const Vector &datax,
                            const Vector &datay, const Vector &dataz,
                            Vector &driftx, Vector &drifty, Vector &driftz,
                            T Dm, T alphaL, T alphaT) {
  Vector D11, D22, D33;
  Vector D12, D13, D23;
  build(g, D11);
  build(g, D22);
  build(g, D33);
  build(g, D12);
  build(g, D13);
  build(g, D23);

  // Compute tensor D in all the cells
  for (auto idz = 0; idz < g.nz; idz++) {
    for (auto idy = 0; idy < g.ny; idy++) {
      for (auto idx = 0; idx < g.nx; idx++) {
        T cx, cy, cz;
        grid::centerOfCell<T>(g, idx, idy, idz, &cx, &cy, &cz);

        T vx, vy, vz;
        par2::facefield::in<T>(datax.data(), datay.data(), dataz.data(), g, idx,
                               idy, idz, true, cx, cy, cz, &vx, &vy, &vz);

        int id = grid::mergeId(g, idx, idy, idz);
        T vnorm = par2::vector::norm(vx, vy, vz);

        D11[id] = (alphaT * vnorm + Dm) + (alphaL - alphaT) * vx * vx / vnorm;
        D22[id] = (alphaT * vnorm + Dm) + (alphaL - alphaT) * vy * vy / vnorm;
        D33[id] = (alphaT * vnorm + Dm) + (alphaL - alphaT) * vz * vz / vnorm;
        D12[id] = (alphaL - alphaT) * vx * vy / vnorm;
        D13[id] = (alphaL - alphaT) * vx * vz / vnorm;
        D23[id] = (alphaL - alphaT) * vy * vz / vnorm;
      }
    }
  }

  // Compute drift correction term
  for (auto idz = 0; idz < g.nz; idz++) {
    for (auto idy = 0; idy < g.ny; idy++) {
      for (auto idx = 0; idx < g.nx; idx++) {
        int id1, id2;

        // Derivatives in x-direction
        T dD11x, dD12x, dD13x;

        T ddx;
        int idx1, idx2;
        if (idx == 0) {
          ddx = g.dx;
          idx1 = idx;
          idx2 = idx + 1;
        } else if (idx == g.nx - 1) {
          ddx = g.dx;
          idx1 = idx - 1;
          idx2 = idx;
        } else {
          ddx = 2 * g.dx;
          idx1 = idx - 1;
          idx2 = idx + 1;
        }

        id1 = grid::mergeId(g, idx1, idy, idz);
        id2 = grid::mergeId(g, idx2, idy, idz);

        dD11x = (D11[id2] - D11[id1]) / ddx;
        dD12x = (D12[id2] - D12[id1]) / ddx;
        dD13x = (D13[id2] - D13[id1]) / ddx;

        // Derivatives in y-direction
        T dD12y, dD22y, dD23y;

        T ddy;
        int idy1, idy2;
        if (idy == 0) {
          ddy = g.dy;
          idy1 = idy;
          idy2 = idy + 1;
        } else if (idy == g.ny - 1) {
          ddy = g.dy;
          idy1 = idy - 1;
          idy2 = idy;
        } else {
          ddy = 2 * g.dy;
          idy1 = idy - 1;
          idy2 = idy + 1;
        }

        id1 = grid::mergeId(g, idx, idy1, idz);
        id2 = grid::mergeId(g, idx, idy2, idz);

        dD12y = (D12[id2] - D12[id1]) / ddy;
        dD22y = (D22[id2] - D22[id1]) / ddy;
        dD23y = (D23[id2] - D23[id1]) / ddy;

        // Derivatives in y-direction
        T dD13z, dD23z, dD33z;

        // TODO fix 2D case
        T ddz;
        int idz1, idz2;
        if (idz == 0) {
          ddz = g.dz;
          idz1 = idz;
          idz2 = idz + 1;
        } else if (idz == g.nz - 1) {
          ddz = g.dz;
          idz1 = idz - 1;
          idz2 = idz;
        } else {
          ddz = 2 * g.dz;
          idz1 = idz - 1;
          idz2 = idz + 1;
        }

        id1 = grid::mergeId(g, idx, idy, idz1);
        id2 = grid::mergeId(g, idx, idy, idz2);

        dD13z = (D13[id2] - D13[id1]) / ddz;
        dD23z = (D23[id2] - D23[id1]) / ddz;
        dD33z = (D33[id2] - D33[id1]) / ddz;

        // Compute drift coefficient in the cell
        int id = grid::mergeId(g, idx, idy, idz);

        driftx[id] = dD11x + dD12y + dD13z;
        drifty[id] = dD12x + dD22y + dD23z;
        driftz[id] = dD13x + dD23y + dD33z;
      }
    }
  }
}

// Kernel para calcular el tensor de dispersión
template <typename T>
__global__ void
computeDispersionTensorKernel(const grid::Grid<T> g, const T *dataxPtr,
                              const T *datayPtr, const T *datazPtr, T *D11Ptr,
                              T *D22Ptr, T *D33Ptr, T *D12Ptr, T *D13Ptr,
                              T *D23Ptr, T Dm, T alphaL, T alphaT) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx < g.nx && idy < g.ny && idz < g.nz) {
    T cx, cy, cz;
    grid::centerOfCell<T>(g, idx, idy, idz, &cx, &cy, &cz);

    T vx, vy, vz;
    par2::facefield::in<T>(dataxPtr, datayPtr, datazPtr, g, idx, idy, idz, true,
                           cx, cy, cz, &vx, &vy, &vz);

    int id = grid::mergeId(g, idx, idy, idz);
    T vnorm = par2::vector::norm(vx, vy, vz);

    // Evitar división por cero si la velocidad es muy pequeña
    if (vnorm < 1e-10) {
      vnorm = 1e-10;
    }

    D11Ptr[id] = (alphaT * vnorm + Dm) + (alphaL - alphaT) * vx * vx / vnorm;
    D22Ptr[id] = (alphaT * vnorm + Dm) + (alphaL - alphaT) * vy * vy / vnorm;
    D33Ptr[id] = (alphaT * vnorm + Dm) + (alphaL - alphaT) * vz * vz / vnorm;
    D12Ptr[id] = (alphaL - alphaT) * vx * vy / vnorm;
    D13Ptr[id] = (alphaL - alphaT) * vx * vz / vnorm;
    D23Ptr[id] = (alphaL - alphaT) * vy * vz / vnorm;
  }
}

// Kernel para calcular la corrección de deriva
template <typename T>
__global__ void
computeDriftCorrectionKernel(const grid::Grid<T> g, const T *D11Ptr,
                             const T *D22Ptr, const T *D33Ptr, const T *D12Ptr,
                             const T *D13Ptr, const T *D23Ptr, T *driftxPtr,
                             T *driftyPtr, T *driftzPtr) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx < g.nx && idy < g.ny && idz < g.nz) {
    int id1, id2;

    // Derivadas en dirección x
    T dD11x, dD12x, dD13x;
    T ddx;
    int idx1, idx2;

    if (idx == 0) {
      ddx = g.dx;
      idx1 = idx;
      idx2 = idx + 1;
    } else if (idx == g.nx - 1) {
      ddx = g.dx;
      idx1 = idx - 1;
      idx2 = idx;
    } else {
      ddx = 2 * g.dx;
      idx1 = idx - 1;
      idx2 = idx + 1;
    }

    id1 = grid::mergeId(g, idx1, idy, idz);
    id2 = grid::mergeId(g, idx2, idy, idz);

    dD11x = (D11Ptr[id2] - D11Ptr[id1]) / ddx;
    dD12x = (D12Ptr[id2] - D12Ptr[id1]) / ddx;
    dD13x = (D13Ptr[id2] - D13Ptr[id1]) / ddx;

    // Derivadas en dirección y
    T dD12y, dD22y, dD23y;
    T ddy;
    int idy1, idy2;

    if (idy == 0) {
      ddy = g.dy;
      idy1 = idy;
      idy2 = idy + 1;
    } else if (idy == g.ny - 1) {
      ddy = g.dy;
      idy1 = idy - 1;
      idy2 = idy;
    } else {
      ddy = 2 * g.dy;
      idy1 = idy - 1;
      idy2 = idy + 1;
    }

    id1 = grid::mergeId(g, idx, idy1, idz);
    id2 = grid::mergeId(g, idx, idy2, idz);

    dD12y = (D12Ptr[id2] - D12Ptr[id1]) / ddy;
    dD22y = (D22Ptr[id2] - D22Ptr[id1]) / ddy;
    dD23y = (D23Ptr[id2] - D23Ptr[id1]) / ddy;

    // Derivadas en dirección z
    T dD13z, dD23z, dD33z;
    T ddz;
    int idz1, idz2;

    if (idz == 0) {
      ddz = g.dz;
      idz1 = idz;
      idz2 = idz + 1;
    } else if (idz == g.nz - 1) {
      ddz = g.dz;
      idz1 = idz - 1;
      idz2 = idz;
    } else {
      ddz = 2 * g.dz;
      idz1 = idz - 1;
      idz2 = idz + 1;
    }

    id1 = grid::mergeId(g, idx, idy, idz1);
    id2 = grid::mergeId(g, idx, idy, idz2);

    dD13z = (D13Ptr[id2] - D13Ptr[id1]) / ddz;
    dD23z = (D23Ptr[id2] - D23Ptr[id1]) / ddz;
    dD33z = (D33Ptr[id2] - D33Ptr[id1]) / ddz;

    // Calcular coeficiente de deriva
    int id = grid::mergeId(g, idx, idy, idz);

    driftxPtr[id] = dD11x + dD12y + dD13z;
    driftyPtr[id] = dD12x + dD22y + dD23z;
    driftzPtr[id] = dD13x + dD23y + dD33z;
  }
}

/**
 * @brief Compute drift correction in each cell (GPU version).
 * @param g Grid where the field is defined
 * @param datax Vector that contains the values on the interfaces
 *        orthogonal to the x-axis
 * @param datay Vector that contains the values on the interfaces
 *        orthogonal to the y-axis
 * @param dataz Vector that contains the values on the interfaces
 *        orthogonal to the z-axis
 * @param driftx Vector that contains the x-values of the correction
 * @param drifty Vector that contains the y-values of the correction
 * @param driftz Vector that contains the z-values of the correction
 * @param Dm Effective molecular diffusion
 * @param alphaL Longitudinal dispersivity
 * @param alphaT Transverse dispersivity
 * @tparam T Float number precision
 */
template <typename T>
void computeDriftCorrectionGPU(const grid::Grid<T> &g,
                               const thrust::device_vector<T> &datax,
                               const thrust::device_vector<T> &datay,
                               const thrust::device_vector<T> &dataz,
                               thrust::device_vector<T> &driftx,
                               thrust::device_vector<T> &drifty,
                               thrust::device_vector<T> &driftz, T Dm, T alphaL,
                               T alphaT) {
  // Crear vectores para el tensor de dispersión en device
  thrust::device_vector<T> D11, D22, D33;
  thrust::device_vector<T> D12, D13, D23;

  build(g, D11);
  build(g, D22);
  build(g, D33);
  build(g, D12);
  build(g, D13);
  build(g, D23);

  // Obtener punteros raw para usar en los kernels
  const T *dataxPtr = thrust::raw_pointer_cast(datax.data());
  const T *datayPtr = thrust::raw_pointer_cast(datay.data());
  const T *datazPtr = thrust::raw_pointer_cast(dataz.data());

  T *D11Ptr = thrust::raw_pointer_cast(D11.data());
  T *D22Ptr = thrust::raw_pointer_cast(D22.data());
  T *D33Ptr = thrust::raw_pointer_cast(D33.data());
  T *D12Ptr = thrust::raw_pointer_cast(D12.data());
  T *D13Ptr = thrust::raw_pointer_cast(D13.data());
  T *D23Ptr = thrust::raw_pointer_cast(D23.data());

  T *driftxPtr = thrust::raw_pointer_cast(driftx.data());
  T *driftyPtr = thrust::raw_pointer_cast(drifty.data());
  T *driftzPtr = thrust::raw_pointer_cast(driftz.data());

  // Configurar la dimensión de los bloques y la grid
  dim3 blockSize(8, 8, 8);
  dim3 gridSize((g.nx + blockSize.x - 1) / blockSize.x,
                (g.ny + blockSize.y - 1) / blockSize.y,
                (g.nz + blockSize.z - 1) / blockSize.z);

  // Lanzar kernel para calcular tensor de dispersión
  computeDispersionTensorKernel<<<gridSize, blockSize>>>(
      g, dataxPtr, datayPtr, datazPtr, D11Ptr, D22Ptr, D33Ptr, D12Ptr, D13Ptr,
      D23Ptr, Dm, alphaL, alphaT);

  // Comprobar errores
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error in computeDispersionTensorKernel: %s\n",
           cudaGetErrorString(error));
  }

  // Lanzar kernel para calcular corrección de deriva
  computeDriftCorrectionKernel<<<gridSize, blockSize>>>(
      g, D11Ptr, D22Ptr, D33Ptr, D12Ptr, D13Ptr, D23Ptr, driftxPtr, driftyPtr,
      driftzPtr);

  // Comprobar errores
  cudaDeviceSynchronize();
  error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("CUDA error in computeDriftCorrectionKernel: %s\n",
           cudaGetErrorString(error));
  }
}

}; // namespace cellfield
}; // namespace par2

#endif // PAR2_CELLFIELD_CUH

```

# Geometry/CornerField.cuh

```cuh
/**
 * @file CornerField.cuh
 * @brief Header file for cornerfield.
 *        A cornerfield is a field that is defined at the each corner
 *        of every cells of the grid.
 *
 * @author Calogero B. Rizzo
 *
 * @copyright This file is part of the PAR2 software.
 *            Copyright (C) 2018 Calogero B. Rizzo
 *
 * @license This program is free software: you can redistribute it and/or modify
 *          it under the terms of the GNU General Public License as published by
 *          the Free Software Foundation, either version 3 of the License, or
 *          (at your option) any later version.
 *
 *          This program is distributed in the hope that it will be useful,
 *          but WITHOUT ANY WARRANTY; without even the implied warranty of
 *          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *          GNU General Public License for more details.
 *
 *          You should have received a copy of the GNU General Public License
 *          along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

#ifndef PAR2_CORNERFIELD_CUH
#define PAR2_CORNERFIELD_CUH

#include <fstream>
#include <iostream>
#include <thrust/device_vector.h>

#include "CartesianGrid.cuh"
#include "FaceField.cuh"
#include "Vector.cuh"

namespace par2 {

namespace cornerfield {
/**
 * @brief Build the vector containing a cornerfield.
 * @param g Grid where the field is defined
 * @param data Vector that contains the cornerfield values
 * @param v Init value for data
 * @tparam T Float number precision
 * @tparam Vector Container for data vectors
 */
template <typename T, class Vector>
void build(const grid::Grid<T> &g, Vector &data, T v = 0) {
  int size = (g.nx + 1) * (g.ny + 1) * (g.nz + 1);
  data.resize(size, v);
}

/**
 * @brief Get the position on the data vector for the corner
 *        defined by the IDs in each direction.
 * @param g Grid where the field is defined
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @tparam T Float number precision
 * @return The position on the data vector
 */
template <typename T>
__host__ __device__ int mergeId(const grid::Grid<T> &g, int idx, int idy,
                                int idz) {
  return idz * (g.ny + 1) * (g.nx + 1) + idy * (g.nx + 1) + idx;
}

/**
 * @brief Identification for each corner. Each cell has
 *        8 corners.
 */
enum Direction {
  C000 = 0, // CXYZ
  C100,
  C010,
  C110,
  C001,
  C101,
  C011,
  C111
};

/**
 * @brief Get the value of the cornerfield at given corner.
 *        The location is found from the IDs of the cell and the
 *        direction of the corner.
 * @param data Vector that contains the cornerfield values
 * @param g Grid where the field is defined
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @tparam T Float number precision
 * @tparam dir Direction of the interface
 * @return The value of the cornerfield at the corner
 */
template <typename T, int dir>
__host__ __device__ T get(const T *data, const grid::Grid<T> &g, int idx,
                          int idy, int idz) {
  int id;
  switch (dir) {
  case Direction::C000:
    id = cornerfield::mergeId(g, idx, idy, idz);
    return data[id];
  case Direction::C100:
    id = cornerfield::mergeId(g, idx + 1, idy, idz);
    return data[id];
  case Direction::C010:
    id = cornerfield::mergeId(g, idx, idy + 1, idz);
    return data[id];
  case Direction::C110:
    id = cornerfield::mergeId(g, idx + 1, idy + 1, idz);
    return data[id];
  case Direction::C001:
    id = cornerfield::mergeId(g, idx, idy, idz + 1);
    return data[id];
  case Direction::C101:
    id = cornerfield::mergeId(g, idx + 1, idy, idz + 1);
    return data[id];
  case Direction::C011:
    id = cornerfield::mergeId(g, idx, idy + 1, idz + 1);
    return data[id];
  case Direction::C111:
    id = cornerfield::mergeId(g, idx + 1, idy + 1, idz + 1);
    return data[id];
  }
  return 0;
}

/**
 * @brief Get the value of the field at given point using trilinear
 *        interpolation.
 * @param cdatax Vector that contains the x-values of the field
 * @param cdatay Vector that contains the y-values of the field
 * @param cdataz Vector that contains the z-values of the field
 * @param g Grid where the field is defined
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @param idValid True if the position is inside the grid
 * @param px Position x-coordinate
 * @param py Position y-coordinate
 * @param pz Position z-coordinate
 * @param vx Result x-coordinate
 * @param vy Result y-coordinate
 * @param vz Result z-coordinate
 * @tparam T Float number precision
 */
template <typename T>
__host__ __device__ void in(const T *cdatax, const T *cdatay, const T *cdataz,
                            const grid::Grid<T> &g, int idx, int idy, int idz,
                            bool idValid, T px, T py, T pz, T *vx, T *vy,
                            T *vz) {
  using Direction = cornerfield::Direction;

  T cx, cy, cz;
  grid::centerOfCell<T>(g, idx, idy, idz, &cx, &cy, &cz);

  T x = (px - cx) / g.dx + 0.5;
  T y = (py - cy) / g.dy + 0.5;
  T z = (pz - cz) / g.dz + 0.5;

  *vx = idValid
            ? interpolation::trilinear(
                  x, y, z, get<T, Direction::C000>(cdatax, g, idx, idy, idz),
                  get<T, Direction::C100>(cdatax, g, idx, idy, idz),
                  get<T, Direction::C010>(cdatax, g, idx, idy, idz),
                  get<T, Direction::C110>(cdatax, g, idx, idy, idz),
                  get<T, Direction::C001>(cdatax, g, idx, idy, idz),
                  get<T, Direction::C101>(cdatax, g, idx, idy, idz),
                  get<T, Direction::C011>(cdatax, g, idx, idy, idz),
                  get<T, Direction::C111>(cdatax, g, idx, idy, idz))
            : 0;

  *vy = idValid
            ? interpolation::trilinear(
                  x, y, z, get<T, Direction::C000>(cdatay, g, idx, idy, idz),
                  get<T, Direction::C100>(cdatay, g, idx, idy, idz),
                  get<T, Direction::C010>(cdatay, g, idx, idy, idz),
                  get<T, Direction::C110>(cdatay, g, idx, idy, idz),
                  get<T, Direction::C001>(cdatay, g, idx, idy, idz),
                  get<T, Direction::C101>(cdatay, g, idx, idy, idz),
                  get<T, Direction::C011>(cdatay, g, idx, idy, idz),
                  get<T, Direction::C111>(cdatay, g, idx, idy, idz))
            : 0;

  *vz = idValid
            ? interpolation::trilinear(
                  x, y, z, get<T, Direction::C000>(cdataz, g, idx, idy, idz),
                  get<T, Direction::C100>(cdataz, g, idx, idy, idz),
                  get<T, Direction::C010>(cdataz, g, idx, idy, idz),
                  get<T, Direction::C110>(cdataz, g, idx, idy, idz),
                  get<T, Direction::C001>(cdataz, g, idx, idy, idz),
                  get<T, Direction::C101>(cdataz, g, idx, idy, idz),
                  get<T, Direction::C011>(cdataz, g, idx, idy, idz),
                  get<T, Direction::C111>(cdataz, g, idx, idy, idz))
            : 0;
}

/**
 * @brief Compute velocity correction for the particle tracking.
 * @param cdatax Vector that contains the x-values of the field
 * @param cdatay Vector that contains the y-values of the field
 * @param cdataz Vector that contains the z-values of the field
 * @param g Grid where the field is defined
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @param idValid True if the position is inside the grid
 * @param px Position x-coordinate
 * @param py Position y-coordinate
 * @param pz Position z-coordinate
 * @param Dm Effective molecular diffusion
 * @param alphaL Longitudinal dispersivity
 * @param alphaT Transverse dispersivity
 * @param vx Result x-coordinate
 * @param vy Result y-coordinate
 * @param vz Result z-coordinate
 * @tparam T Float number precision
 */
template <typename T>
__host__ __device__ void
velocityCorrection(const T *cdatax, const T *cdatay, const T *cdataz,
                   const grid::Grid<T> &g, int idx, int idy, int idz,
                   bool idValid, T px, T py, T pz, T Dm, T alphaL, T alphaT,
                   T *vx, T *vy, T *vz) {
  using Direction = cornerfield::Direction;

  T cx, cy, cz;
  grid::centerOfCell<T>(g, idx, idy, idz, &cx, &cy, &cz);

  T x = (px - cx) / g.dx + 0.5;
  T y = (py - cy) / g.dy + 0.5;
  T z = (pz - cz) / g.dz + 0.5;

  T vx000 = idValid ? get<T, Direction::C000>(cdatax, g, idx, idy, idz) : 1;
  T vx100 = idValid ? get<T, Direction::C100>(cdatax, g, idx, idy, idz) : 1;
  T vx010 = idValid ? get<T, Direction::C010>(cdatax, g, idx, idy, idz) : 1;
  T vx110 = idValid ? get<T, Direction::C110>(cdatax, g, idx, idy, idz) : 1;
  T vx001 = idValid ? get<T, Direction::C001>(cdatax, g, idx, idy, idz) : 1;
  T vx101 = idValid ? get<T, Direction::C101>(cdatax, g, idx, idy, idz) : 1;
  T vx011 = idValid ? get<T, Direction::C011>(cdatax, g, idx, idy, idz) : 1;
  T vx111 = idValid ? get<T, Direction::C111>(cdatax, g, idx, idy, idz) : 1;

  T vy000 = idValid ? get<T, Direction::C000>(cdatay, g, idx, idy, idz) : 1;
  T vy100 = idValid ? get<T, Direction::C100>(cdatay, g, idx, idy, idz) : 1;
  T vy010 = idValid ? get<T, Direction::C010>(cdatay, g, idx, idy, idz) : 1;
  T vy110 = idValid ? get<T, Direction::C110>(cdatay, g, idx, idy, idz) : 1;
  T vy001 = idValid ? get<T, Direction::C001>(cdatay, g, idx, idy, idz) : 1;
  T vy101 = idValid ? get<T, Direction::C101>(cdatay, g, idx, idy, idz) : 1;
  T vy011 = idValid ? get<T, Direction::C011>(cdatay, g, idx, idy, idz) : 1;
  T vy111 = idValid ? get<T, Direction::C111>(cdatay, g, idx, idy, idz) : 1;

  T vz000 = idValid ? get<T, Direction::C000>(cdataz, g, idx, idy, idz) : 1;
  T vz100 = idValid ? get<T, Direction::C100>(cdataz, g, idx, idy, idz) : 1;
  T vz010 = idValid ? get<T, Direction::C010>(cdataz, g, idx, idy, idz) : 1;
  T vz110 = idValid ? get<T, Direction::C110>(cdataz, g, idx, idy, idz) : 1;
  T vz001 = idValid ? get<T, Direction::C001>(cdataz, g, idx, idy, idz) : 1;
  T vz101 = idValid ? get<T, Direction::C101>(cdataz, g, idx, idy, idz) : 1;
  T vz011 = idValid ? get<T, Direction::C011>(cdataz, g, idx, idy, idz) : 1;
  T vz111 = idValid ? get<T, Direction::C111>(cdataz, g, idx, idy, idz) : 1;

  // If velocity is zero in one corner, add eps to vx component to avoid nan
  const T toll = 0.01 * Dm / alphaL;

  vx000 = (vx000 < toll && vy000 < toll && vz000 < toll) ? toll : vx000;
  vx100 = (vx100 < toll && vy100 < toll && vz100 < toll) ? toll : vx100;
  vx010 = (vx010 < toll && vy010 < toll && vz010 < toll) ? toll : vx010;
  vx110 = (vx110 < toll && vy110 < toll && vz110 < toll) ? toll : vx110;
  vx001 = (vx001 < toll && vy001 < toll && vz001 < toll) ? toll : vx001;
  vx101 = (vx101 < toll && vy101 < toll && vz101 < toll) ? toll : vx101;
  vx011 = (vx011 < toll && vy011 < toll && vz011 < toll) ? toll : vx011;
  vx111 = (vx111 < toll && vy111 < toll && vz111 < toll) ? toll : vx111;

  T vnorm000 = par2::vector::norm(vx000, vy000, vz000);
  T vnorm100 = par2::vector::norm(vx100, vy100, vz100);
  T vnorm010 = par2::vector::norm(vx010, vy010, vz010);
  T vnorm110 = par2::vector::norm(vx110, vy110, vz110);
  T vnorm001 = par2::vector::norm(vx001, vy001, vz001);
  T vnorm101 = par2::vector::norm(vx101, vy101, vz101);
  T vnorm011 = par2::vector::norm(vx011, vy011, vz011);
  T vnorm111 = par2::vector::norm(vx111, vy111, vz111);

  T dDxxx = interpolation::trilinearDevX(
      x, y, z, g.dx,
      (alphaT * vnorm000 + Dm) + (alphaL - alphaT) * vx000 * vx000 / vnorm000,
      (alphaT * vnorm100 + Dm) + (alphaL - alphaT) * vx100 * vx100 / vnorm100,
      (alphaT * vnorm010 + Dm) + (alphaL - alphaT) * vx010 * vx010 / vnorm010,
      (alphaT * vnorm110 + Dm) + (alphaL - alphaT) * vx110 * vx110 / vnorm110,
      (alphaT * vnorm001 + Dm) + (alphaL - alphaT) * vx001 * vx001 / vnorm001,
      (alphaT * vnorm101 + Dm) + (alphaL - alphaT) * vx101 * vx101 / vnorm101,
      (alphaT * vnorm011 + Dm) + (alphaL - alphaT) * vx011 * vx011 / vnorm011,
      (alphaT * vnorm111 + Dm) + (alphaL - alphaT) * vx111 * vx111 / vnorm111);

  T dDyyy = interpolation::trilinearDevY(
      x, y, z, g.dy,
      (alphaT * vnorm000 + Dm) + (alphaL - alphaT) * vy000 * vy000 / vnorm000,
      (alphaT * vnorm100 + Dm) + (alphaL - alphaT) * vy100 * vy100 / vnorm100,
      (alphaT * vnorm010 + Dm) + (alphaL - alphaT) * vy010 * vy010 / vnorm010,
      (alphaT * vnorm110 + Dm) + (alphaL - alphaT) * vy110 * vy110 / vnorm110,
      (alphaT * vnorm001 + Dm) + (alphaL - alphaT) * vy001 * vy001 / vnorm001,
      (alphaT * vnorm101 + Dm) + (alphaL - alphaT) * vy101 * vy101 / vnorm101,
      (alphaT * vnorm011 + Dm) + (alphaL - alphaT) * vy011 * vy011 / vnorm011,
      (alphaT * vnorm111 + Dm) + (alphaL - alphaT) * vy111 * vy111 / vnorm111);

  T dDzzz = interpolation::trilinearDevZ(
      x, y, z, g.dz,
      (alphaT * vnorm000 + Dm) + (alphaL - alphaT) * vz000 * vz000 / vnorm000,
      (alphaT * vnorm100 + Dm) + (alphaL - alphaT) * vz100 * vz100 / vnorm100,
      (alphaT * vnorm010 + Dm) + (alphaL - alphaT) * vz010 * vz010 / vnorm010,
      (alphaT * vnorm110 + Dm) + (alphaL - alphaT) * vz110 * vz110 / vnorm110,
      (alphaT * vnorm001 + Dm) + (alphaL - alphaT) * vz001 * vz001 / vnorm001,
      (alphaT * vnorm101 + Dm) + (alphaL - alphaT) * vz101 * vz101 / vnorm101,
      (alphaT * vnorm011 + Dm) + (alphaL - alphaT) * vz011 * vz011 / vnorm011,
      (alphaT * vnorm111 + Dm) + (alphaL - alphaT) * vz111 * vz111 / vnorm111);

  T dDxyx = interpolation::trilinearDevX(
      x, y, z, g.dx, (alphaL - alphaT) * vx000 * vy000 / vnorm000,
      (alphaL - alphaT) * vx100 * vy100 / vnorm100,
      (alphaL - alphaT) * vx010 * vy010 / vnorm010,
      (alphaL - alphaT) * vx110 * vy110 / vnorm110,
      (alphaL - alphaT) * vx001 * vy001 / vnorm001,
      (alphaL - alphaT) * vx101 * vy101 / vnorm101,
      (alphaL - alphaT) * vx011 * vy011 / vnorm011,
      (alphaL - alphaT) * vx111 * vy111 / vnorm111);

  T dDxyy = interpolation::trilinearDevY(
      x, y, z, g.dy, (alphaL - alphaT) * vx000 * vy000 / vnorm000,
      (alphaL - alphaT) * vx100 * vy100 / vnorm100,
      (alphaL - alphaT) * vx010 * vy010 / vnorm010,
      (alphaL - alphaT) * vx110 * vy110 / vnorm110,
      (alphaL - alphaT) * vx001 * vy001 / vnorm001,
      (alphaL - alphaT) * vx101 * vy101 / vnorm101,
      (alphaL - alphaT) * vx011 * vy011 / vnorm011,
      (alphaL - alphaT) * vx111 * vy111 / vnorm111);

  T dDxzx = interpolation::trilinearDevX(
      x, y, z, g.dx, (alphaL - alphaT) * vx000 * vz000 / vnorm000,
      (alphaL - alphaT) * vx100 * vz100 / vnorm100,
      (alphaL - alphaT) * vx010 * vz010 / vnorm010,
      (alphaL - alphaT) * vx110 * vz110 / vnorm110,
      (alphaL - alphaT) * vx001 * vz001 / vnorm001,
      (alphaL - alphaT) * vx101 * vz101 / vnorm101,
      (alphaL - alphaT) * vx011 * vz011 / vnorm011,
      (alphaL - alphaT) * vx111 * vz111 / vnorm111);

  T dDxzz = interpolation::trilinearDevZ(
      x, y, z, g.dz, (alphaL - alphaT) * vx000 * vz000 / vnorm000,
      (alphaL - alphaT) * vx100 * vz100 / vnorm100,
      (alphaL - alphaT) * vx010 * vz010 / vnorm010,
      (alphaL - alphaT) * vx110 * vz110 / vnorm110,
      (alphaL - alphaT) * vx001 * vz001 / vnorm001,
      (alphaL - alphaT) * vx101 * vz101 / vnorm101,
      (alphaL - alphaT) * vx011 * vz011 / vnorm011,
      (alphaL - alphaT) * vx111 * vz111 / vnorm111);

  T dDyzy = interpolation::trilinearDevY(
      x, y, z, g.dy, (alphaL - alphaT) * vy000 * vz000 / vnorm000,
      (alphaL - alphaT) * vy100 * vz100 / vnorm100,
      (alphaL - alphaT) * vy010 * vz010 / vnorm010,
      (alphaL - alphaT) * vy110 * vz110 / vnorm110,
      (alphaL - alphaT) * vy001 * vz001 / vnorm001,
      (alphaL - alphaT) * vy101 * vz101 / vnorm101,
      (alphaL - alphaT) * vy011 * vz011 / vnorm011,
      (alphaL - alphaT) * vy111 * vz111 / vnorm111);

  T dDyzz = interpolation::trilinearDevZ(
      x, y, z, g.dz, (alphaL - alphaT) * vy000 * vz000 / vnorm000,
      (alphaL - alphaT) * vy100 * vz100 / vnorm100,
      (alphaL - alphaT) * vy010 * vz010 / vnorm010,
      (alphaL - alphaT) * vy110 * vz110 / vnorm110,
      (alphaL - alphaT) * vy001 * vz001 / vnorm001,
      (alphaL - alphaT) * vy101 * vz101 / vnorm101,
      (alphaL - alphaT) * vy011 * vz011 / vnorm011,
      (alphaL - alphaT) * vy111 * vz111 / vnorm111);

  *vx = dDxxx + dDxyy + dDxzz;
  *vy = dDxyx + dDyyy + dDyzz;
  *vz = dDxzx + dDyzy + dDzzz;
}

/**
 * @brief Compute displacement matrix for the particle tracking.
 * @param cdatax Vector that contains the x-values of the field
 * @param cdatay Vector that contains the y-values of the field
 * @param cdataz Vector that contains the z-values of the field
 * @param g Grid where the field is defined
 * @param idx ID along x
 * @param idy ID along y
 * @param idz ID along z
 * @param idValid True if the position is inside the grid
 * @param px Position x-coordinate
 * @param py Position y-coordinate
 * @param pz Position z-coordinate
 * @param Dm Effective molecular diffusion
 * @param alphaL Longitudinal dispersivity
 * @param alphaT Transverse dispersivity
 * @param dt Time step
 * @param B00 Matrix component 00
 * @param B00 Matrix component 11
 * @param B00 Matrix component 22
 * @param B00 Matrix component 01
 * @param B00 Matrix component 02
 * @param B00 Matrix component 12
 * @tparam T Float number precision
 */
template <typename T>
__host__ __device__ void
displacementMatrix(const T *cdatax, const T *cdatay, const T *cdataz,
                   const grid::Grid<T> &g, int idx, int idy, int idz,
                   bool idValid, T px, T py, T pz, T Dm, T alphaL, T alphaT,
                   T dt, T *B00, T *B11, T *B22, T *B01, T *B02, T *B12) {
  T vx, vy, vz;
  cornerfield::in<T>(cdatax, cdatay, cdataz, g, idx, idy, idz, idValid, px, py,
                     pz, &vx, &vy, &vz);

  const T toll = 0.01 * Dm / alphaL;

  vx = (vx < toll) ? toll : vx;

  T vnorm2 = vector::norm2<T>(vx, vy, vz);
  T vnorm = sqrt(vnorm2);

  // Compute dispersion Matrix
  T alpha = alphaT * vnorm + Dm;
  T beta = (alphaL - alphaT) / vnorm; // Check threshold needed

  // Eigenvectors
  // vx0 == vx, etc.

  T vx1 = -vy;
  T vy1 = vx;
  T vz1 = 0.0;

  T vx2 = -vz * vx;
  T vy2 = -vz * vy;
  T vz2 = vx * vx + vy * vy;

  // vnorm2_0 == vnorm2
  T vnorm2_1 = vector::norm2<T>(vx1, vy1, vz1);
  T vnorm2_2 = vector::norm2<T>(vx2, vy2, vz2);

  T gamma0 = sqrt(alpha + beta * vnorm2) / vnorm2;
  T gamma1 = sqrt(alpha) / vnorm2_1;
  T gamma2 = sqrt(alpha) / vnorm2_2;

  // Pre-coefficient
  T coeff = sqrt(2.0 * dt);

  // Dispersion Matrix
  *B00 = coeff * (gamma0 * vx * vx + gamma1 * vx1 * vx1 + gamma2 * vx2 * vx2);

  *B11 = coeff * (gamma0 * vy * vy + gamma1 * vy1 * vy1 + gamma2 * vy2 * vy2);

  *B22 = coeff * (gamma0 * vz * vz + gamma1 * vz1 * vz1 + gamma2 * vz2 * vz2);

  *B01 = coeff * (gamma0 * vx * vy + gamma1 * vx1 * vy1 + gamma2 * vx2 * vy2);

  // B10 == B01

  *B02 = coeff * (gamma0 * vx * vz + gamma1 * vx1 * vz1 + gamma2 * vx2 * vz2);

  // B20 == B02

  *B12 = coeff * (gamma0 * vy * vz + gamma1 * vy1 * vz1 + gamma2 * vy2 * vz2);

  // B21 == B12
}

/**
 * @brief Compute cornerfield from facefield (CPU version).
 * @param g Grid where the field is defined
 * @param datax Vector that contains the values on the interfaces
 *        orthogonal to the x-axis
 * @param datay Vector that contains the values on the interfaces
 *        orthogonal to the y-axis
 * @param dataz Vector that contains the values on the interfaces
 *        orthogonal to the z-axis
 * @param cdatax Vector that contains the x-values of the cornerfield
 * @param cdatay Vector that contains the y-values of the cornerfield
 * @param cdataz Vector that contains the z-values of the cornerfield
 * @tparam T Float number precision
 * @tparam Vector Container for data vectors
 */
template <typename T, class Vector>
void computeCornerVelocities(const grid::Grid<T> &g, const Vector &datax,
                             const Vector &datay, const Vector &dataz,
                             Vector &cdatax, Vector &cdatay, Vector &cdataz) {
  using Direction = grid::Direction;

  auto dataxPtr = datax.data();
  auto datayPtr = datay.data();
  auto datazPtr = dataz.data();

  for (auto idz = 0; idz < g.nz + 1; idz++) {
    for (auto idy = 0; idy < g.ny + 1; idy++) {
      for (auto idx = 0; idx < g.nx + 1; idx++) {
        int id = cornerfield::mergeId(g, idx, idy, idz);

        T vx, vy, vz;
        int fx, fy, fz;
        int tx, ty, tz;

        // Velocity x-direction
        vx = 0;
        fx = 0;
        tx = 0;

        if (idx == g.nx) {
          tx = 1;
        }

        for (auto ty = 0; ty <= 1; ty++) {
          for (auto tz = 0; tz <= 1; tz++) {
            if (grid::validId(g, idx - tx, idy - ty, idz - tz)) {
              if (tx == 0) {
                vx += facefield::get<T, Direction::XM>(dataxPtr, datayPtr,
                                                       datazPtr, g, idx - tx,
                                                       idy - ty, idz - tz);
              } else {
                vx += facefield::get<T, Direction::XP>(dataxPtr, datayPtr,
                                                       datazPtr, g, idx - tx,
                                                       idy - ty, idz - tz);
              }
              fx += 1;
            }
          }
        }

        cdatax[id] = vx / fx;

        // Velocity y-direction
        vy = 0;
        fy = 0;
        ty = 0;

        if (idy == g.ny) {
          ty = 1;
        }

        for (auto tx = 0; tx <= 1; tx++) {
          for (auto tz = 0; tz <= 1; tz++) {
            if (grid::validId(g, idx - tx, idy - ty, idz - tz)) {
              if (ty == 0) {
                vy += facefield::get<T, Direction::YM>(dataxPtr, datayPtr,
                                                       datazPtr, g, idx - tx,
                                                       idy - ty, idz - tz);
              } else {
                vy += facefield::get<T, Direction::YP>(dataxPtr, datayPtr,
                                                       datazPtr, g, idx - tx,
                                                       idy - ty, idz - tz);
              }
              fy += 1;
            }
          }
        }
        cdatay[id] = vy / fy;

        // Velocity z-direction
        vz = 0;
        fz = 0;
        tz = 0;

        if (idz == g.nz) {
          tz = 1;
        }

        for (auto tx = 0; tx <= 1; tx++) {
          for (auto ty = 0; ty <= 1; ty++) {
            if (grid::validId(g, idx - tx, idy - ty, idz - tz)) {
              if (tz == 0) {
                vz += facefield::get<T, Direction::ZM>(dataxPtr, datayPtr,
                                                       datazPtr, g, idx - tx,
                                                       idy - ty, idz - tz);
              } else {
                vz += facefield::get<T, Direction::ZP>(dataxPtr, datayPtr,
                                                       datazPtr, g, idx - tx,
                                                       idy - ty, idz - tz);
              }
              fz += 1;
            }
          }
        }
        cdataz[id] = vz / fz;
      }
    }
  }
}

/**
 * @brief CUDA kernel para calcular velocidades en esquinas
 * @param g Grid donde el campo está definido
 * @param dataxPtr Puntero a los datos en x
 * @param datayPtr Puntero a los datos en y
 * @param datazPtr Puntero a los datos en z
 * @param cdataxPtr Puntero donde guardar resultados en x
 * @param cdatayPtr Puntero donde guardar resultados en y
 * @param cdatazPtr Puntero donde guardar resultados en z
 */
template <typename T>
__global__ void
computeCornerVelocitiesKernel(const grid::Grid<T> g, const T *dataxPtr,
                              const T *datayPtr, const T *datazPtr,
                              T *cdataxPtr, T *cdatayPtr, T *cdatazPtr) {
  using Direction = grid::Direction;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int idy = blockIdx.y * blockDim.y + threadIdx.y;
  int idz = blockIdx.z * blockDim.z + threadIdx.z;

  if (idx < g.nx + 1 && idy < g.ny + 1 && idz < g.nz + 1) {
    int id = cornerfield::mergeId(g, idx, idy, idz);

    T vx, vy, vz;
    int fx, fy, fz;
    int tx, ty, tz;

    // Velocity x-direction
    vx = 0;
    fx = 0;
    tx = 0;

    if (idx == g.nx) {
      tx = 1;
    }

    for (auto ty = 0; ty <= 1; ty++) {
      for (auto tz = 0; tz <= 1; tz++) {
        if (grid::validId(g, idx - tx, idy - ty, idz - tz)) {
          if (tx == 0) {
            vx += facefield::get<T, Direction::XM>(
                dataxPtr, datayPtr, datazPtr, g, idx - tx, idy - ty, idz - tz);
          } else {
            vx += facefield::get<T, Direction::XP>(
                dataxPtr, datayPtr, datazPtr, g, idx - tx, idy - ty, idz - tz);
          }
          fx += 1;
        }
      }
    }

    cdataxPtr[id] = vx / fx;

    // Velocity y-direction
    vy = 0;
    fy = 0;
    ty = 0;

    if (idy == g.ny) {
      ty = 1;
    }

    for (auto tx = 0; tx <= 1; tx++) {
      for (auto tz = 0; tz <= 1; tz++) {
        if (grid::validId(g, idx - tx, idy - ty, idz - tz)) {
          if (ty == 0) {
            vy += facefield::get<T, Direction::YM>(
                dataxPtr, datayPtr, datazPtr, g, idx - tx, idy - ty, idz - tz);
          } else {
            vy += facefield::get<T, Direction::YP>(
                dataxPtr, datayPtr, datazPtr, g, idx - tx, idy - ty, idz - tz);
          }
          fy += 1;
        }
      }
    }
    cdatayPtr[id] = vy / fy;

    // Velocity z-direction
    vz = 0;
    fz = 0;
    tz = 0;

    if (idz == g.nz) {
      tz = 1;
    }

    for (auto tx = 0; tx <= 1; tx++) {
      for (auto ty = 0; ty <= 1; ty++) {
        if (grid::validId(g, idx - tx, idy - ty, idz - tz)) {
          if (tz == 0) {
            vz += facefield::get<T, Direction::ZM>(
                dataxPtr, datayPtr, datazPtr, g, idx - tx, idy - ty, idz - tz);
          } else {
            vz += facefield::get<T, Direction::ZP>(
                dataxPtr, datayPtr, datazPtr, g, idx - tx, idy - ty, idz - tz);
          }
          fz += 1;
        }
      }
    }
    cdatazPtr[id] = vz / fz;
  }
}

/**
 * @brief Compute cornerfield from facefield (GPU version).
 * @param g Grid where the field is defined
 * @param datax Vector that contains the values on the interfaces
 *        orthogonal to the x-axis
 * @param datay Vector that contains the values on the interfaces
 *        orthogonal to the y-axis
 * @param dataz Vector that contains the values on the interfaces
 *        orthogonal to the z-axis
 * @param cdatax Vector that contains the x-values of the cornerfield
 * @param cdatay Vector that contains the y-values of the cornerfield
 * @param cdataz Vector that contains the z-values of the cornerfield
 * @tparam T Float number precision
 */
template <typename T>
void computeCornerVelocitiesGPU(const grid::Grid<T> &g,
                                const thrust::device_vector<T> &datax,
                                const thrust::device_vector<T> &datay,
                                const thrust::device_vector<T> &dataz,
                                thrust::device_vector<T> &cdatax,
                                thrust::device_vector<T> &cdatay,
                                thrust::device_vector<T> &cdataz) {
  using Direction = grid::Direction;

  // Obtener punteros raw para usar en el kernel
  const T *dataxPtr = thrust::raw_pointer_cast(datax.data());
  const T *datayPtr = thrust::raw_pointer_cast(datay.data());
  const T *datazPtr = thrust::raw_pointer_cast(dataz.data());
  T *cdataxPtr = thrust::raw_pointer_cast(cdatax.data());
  T *cdatayPtr = thrust::raw_pointer_cast(cdatay.data());
  T *cdatazPtr = thrust::raw_pointer_cast(cdataz.data());

  // Calcular dimensiones de la grid y lanzar el kernel
  dim3 block(8, 8, 8);
  dim3 grid((g.nx + 1 + block.x - 1) / block.x,
            (g.ny + 1 + block.y - 1) / block.y,
            (g.nz + 1 + block.z - 1) / block.z);

  // Lanzar el kernel
  computeCornerVelocitiesKernel<T><<<grid, block>>>(
      g, dataxPtr, datayPtr, datazPtr, cdataxPtr, cdatayPtr, cdatazPtr);

  cudaDeviceSynchronize();
}
} // namespace cornerfield

} // namespace par2

#endif // PAR2_CORNERFIELD_CUH

```

# Geometry/FaceField.cuh

```cuh
/**
* @file FaceField.cuh
* @brief Header file for facefield.
*        A facefield is a variable that is defined at the center of
*        of each cell interface of the grid.
*
* @author Calogero B. Rizzo
*
* @copyright This file is part of the PAR2 software.
*            Copyright (C) 2018 Calogero B. Rizzo
*
* @license This program is free software: you can redistribute it and/or modify
*          it under the terms of the GNU General Public License as published by
*          the Free Software Foundation, either version 3 of the License, or
*          (at your option) any later version.
*
*          This program is distributed in the hope that it will be useful,
*          but WITHOUT ANY WARRANTY; without even the implied warranty of
*          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*          GNU General Public License for more details.
*
*          You should have received a copy of the GNU General Public License
*          along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef PAR2_FACEFIELD_CUH
#define PAR2_FACEFIELD_CUH

#include <fstream>
#include <iostream>
#include <algorithm>

#include "CartesianGrid.cuh"
#include "Interpolation.cuh"

namespace par2
{

    namespace facefield
    {
        /**
        * @brief Build the vectors containing a facefield.
        * @param g Grid where the field is defined
        * @param datax Vector that contains the values on the interfaces
        *        orthogonal to the x-axis
        * @param datay Vector that contains the values on the interfaces
        *        orthogonal to the y-axis
        * @param dataz Vector that contains the values on the interfaces
        *        orthogonal to the z-axis
        * @param vx Init value for datax
        * @param vy Init value for datay
        * @param vz Init value for dataz
        * @tparam T Float number precision
        * @tparam Vector Container for data vectors
        */
        template<typename T, class Vector>
        void build(const grid::Grid<T>& g, Vector& datax, Vector& datay, Vector& dataz,
                               T vx = 0, T vy = 0, T vz = 0)
        {
            int size = (g.nx+1)*(g.ny+1)*(g.nz+1);
            datax.resize(size, vx);
            datay.resize(size, vy);
            dataz.resize(size, vz);
        }

        /**
        * @brief Get the position on the data vectors for the interface
        *        defined by the IDs in each direction.
        * @param g Grid where the field is defined
        * @param idx ID along x
        * @param idy ID along y
        * @param idz ID along z
        * @tparam T Float number precision
        * @return The position on the data vectors
        */
        template<typename T>
        __host__ __device__
        int mergeId(const grid::Grid<T>& g, int idx, int idy, int idz)
        {
            return idz*(g.ny+1)*(g.nx+1) + idy*(g.nx+1) + idx;
        }

        /**
        * @brief Get the value of the facefield at given interface.
        *        The interface is found from the IDs of the cell and the
        *        direction of the interface.
        * @param datax Vector that contains the values on the interfaces
        *        orthogonal to the x-axis
        * @param datay Vector that contains the values on the interfaces
        *        orthogonal to the y-axis
        * @param dataz Vector that contains the values on the interfaces
        *        orthogonal to the z-axis
        * @param g Grid where the field is defined
        * @param idx ID along x
        * @param idy ID along y
        * @param idz ID along z
        * @tparam T Float number precision
        * @tparam dir Direction of the interface
        * @return The value of the facefield at the interface
        */
        template<typename T, int dir>
        __host__ __device__
        T get(const T* datax, const T* datay, const T* dataz, const grid::Grid<T>& g,
              int idx, int idy, int idz)
        {
            int id;
            switch (dir)
            {
                case grid::XP:
                    id = facefield::mergeId(g, idx+1, idy, idz);
                    //id = (idz)*(g.nx+1)*(g.ny+1) + (idy)*(g.nx+1) + (idx+1);
                    return datax[id];
                case grid::XM:
                    id = facefield::mergeId(g, idx  , idy, idz);
                    //id = (idz)*(g.nx+1)*(g.ny+1) + (idy)*(g.nx+1) + (idx  );
                    return datax[id];
                case grid::YP:
                    id = facefield::mergeId(g, idx, idy+1, idz);
                    //id = (idz)*(g.nx+1)*(g.ny+1) + (idy+1)*(g.nx+1) + (idx);
                    return datay[id];
                case grid::YM:
                    id = facefield::mergeId(g, idx, idy  , idz);
                    //id = (idz)*(g.nx+1)*(g.ny+1) + (idy  )*(g.nx+1) + (idx);
                    return datay[id];
                case grid::ZP:
                    id = facefield::mergeId(g, idx, idy, idz+1);
                    //id = (idz+1)*(g.nx+1)*(g.ny+1) + (idy)*(g.nx+1) + (idx);
                    return dataz[id];
                case grid::ZM:
                    id = facefield::mergeId(g, idx, idy, idz  );
                    //id = (idz  )*(g.nx+1)*(g.ny+1) + (idy)*(g.nx+1) + (idx);
                    return dataz[id];
            }
            return 0;
        }

        /**
        * @brief Get the value of the field at given point using linear
        *        interpolation.
        * @param datax Vector that contains the values on the interfaces
        *        orthogonal to the x-axis
        * @param datay Vector that contains the values on the interfaces
        *        orthogonal to the y-axis
        * @param dataz Vector that contains the values on the interfaces
        *        orthogonal to the z-axis
        * @param g Grid where the field is defined
        * @param idx ID along x
        * @param idy ID along y
        * @param idz ID along z
        * @param idValid True if the position is inside the grid
        * @param px Position x-coordinate
        * @param py Position y-coordinate
        * @param pz Position z-coordinate
        * @param vx Result x-coordinate
        * @param vy Result y-coordinate
        * @param vz Result z-coordinate
        * @tparam T Float number precision
        */
        template<typename T>
        __host__ __device__
        void in(const T* datax, const T* datay, const T* dataz, const grid::Grid<T>& g,
                int idx, int idy, int idz, bool idValid, T px, T py, T pz, T* vx, T* vy, T* vz)
        {
            using Direction = grid::Direction;

            T cx, cy, cz;
            grid::centerOfCell<T>(g, idx, idy, idz, &cx, &cy, &cz);

            T Dx = px - cx;
            T Dy = py - cy;
            T Dz = pz - cz;

            T vxp = idValid ? get<T, Direction::XP>(datax, datay, dataz, g, idx, idy, idz) : 1;
            T vxm = idValid ? get<T, Direction::XM>(datax, datay, dataz, g, idx, idy, idz) : 1;
            T vyp = idValid ? get<T, Direction::YP>(datax, datay, dataz, g, idx, idy, idz) : 1;
            T vym = idValid ? get<T, Direction::YM>(datax, datay, dataz, g, idx, idy, idz) : 1;
            T vzp = idValid ? get<T, Direction::ZP>(datax, datay, dataz, g, idx, idy, idz) : 1;
            T vzm = idValid ? get<T, Direction::ZM>(datax, datay, dataz, g, idx, idy, idz) : 1;

            *vx = interpolation::linear<T>(Dx/g.dx + 0.5, vxm, vxp);
            *vy = interpolation::linear<T>(Dy/g.dy + 0.5, vym, vyp);
            *vz = interpolation::linear<T>(Dz/g.dz + 0.5, vzm, vzp);
        }

        /**
        * @brief Import velocity field from modflow.
        * @param g Grid where the field is defined
        * @param datax Vector that contains the values on the interfaces
        *        orthogonal to the x-axis
        * @param datay Vector that contains the values on the interfaces
        *        orthogonal to the y-axis
        * @param dataz Vector that contains the values on the interfaces
        *        orthogonal to the z-axis
        * @param fileName Path to the modflow file
        * @param rho Porosity
        * @tparam T Float number precision
        * @tparam Vector Container for data vectors
        */
        template<typename T, class Vector>
        void importFromModflow(const grid::Grid<T>& g,
                               Vector& datax, Vector& datay, Vector& dataz,
                               const std::string fileName,
                               const T rho)
        {
            std::ifstream inStream;
            inStream.open(fileName, std::ifstream::in);
            if (inStream.is_open())
            {
                std::string line;

                bool is2D = g.nz == 1;
                T area, val;
                int id;

                std::string prefix;

                prefix = " 'QXX";

                while (line.compare(0, prefix.size(), prefix) != 0)
                {
                    std::getline(inStream, line);
                }

                area = g.dy*g.dz;

                for (auto z = 0; z < g.nz; z++)
                {
                    for (auto y = 0; y < g.ny; y++)
                    {
                        for (auto x = 0; x < g.nx; x++)
                        {
                            id = z*(g.nx+1)*(g.ny+1) + y*(g.nx+1) + (x+1);
                            inStream >> val;
                            datax[id] = val/area/rho;   // velocity x
                        }
                    }
                }

                prefix = " 'QYY";

                while (line.compare(0, prefix.size(), prefix) != 0)
                {
                    std::getline(inStream, line);
                }

                area = g.dx*g.dz;

                for (auto z = 0; z < g.nz; z++)
                {
                    for (auto y = 0; y < g.ny; y++)
                    {
                        for (auto x = 0; x < g.nx; x++)
                        {
                            id = z*(g.nx+1)*(g.ny+1) + (y+1)*(g.nx+1) + x;
                            inStream >> val;
                            datay[id] = val/area/rho;   // velocity y
                        }
                    }
                }

                if (!is2D)
                {
                    prefix = " 'QZZ";

                    while (line.compare(0, prefix.size(), prefix) != 0)
                    {
                        std::getline(inStream, line);
                    }

                    area = g.dx*g.dy;

                    for (auto z = 0; z < g.nz; z++)
                    {
                        for (auto y = 0; y < g.ny; y++)
                        {
                            for (auto x = 0; x < g.nx; x++)
                            {
                                id = (z+1)*(g.nx+1)*(g.ny+1) + y*(g.nx+1) + x;
                                inStream >> val;
                                dataz[id] = val/area/rho;   // velocity z
                            }
                        }
                    }
                }
            }
            else
            {
                throw std::runtime_error(std::string("Could not open file ") + fileName);
            }
            inStream.close();
        }

        /**
        * @brief Export velocity field to VTK (unusued).
        * @param g Grid where the field is defined
        * @param datax Vector that contains the values on the interfaces
        *        orthogonal to the x-axis
        * @param datay Vector that contains the values on the interfaces
        *        orthogonal to the y-axis
        * @param dataz Vector that contains the values on the interfaces
        *        orthogonal to the z-axis
        * @param fileName Path to the vtk file
        * @tparam T Float number precision
        * @tparam Vector Container for data vectors
        */
        template<typename T, class Vector>
        void exportVTK(const grid::Grid<T>& g, Vector& datax, Vector& datay, Vector& dataz, const std::string fileName)
        {
            std::ofstream outStream;
            outStream.open(fileName);
            if (outStream.is_open())
            {
                outStream << "# vtk DataFile Version 2.0" << std::endl;
                outStream << "Velocity Field" << std::endl;
                outStream << "ASCII" << std::endl;
                outStream << "DATASET STRUCTURED_POINTS" << std::endl;
                outStream << "DIMENSIONS " << g.nx << " " << g.ny << " " << g.nz << std::endl;
                outStream << "ORIGIN " << g.dx << " " << g.dy << " " << g.dz << std::endl;
                outStream << "SPACING " << g.dx << " " << g.dy << " " << g.dz << std::endl;
                outStream << "POINT_DATA " << g.nx * g.ny * g.nz << std::endl;

                outStream << std::endl;
                outStream << "SCALARS velocityX double" << std::endl;
                outStream << "LOOKUP_TABLE default" << std::endl;
                for (auto idz = 0; idz < g.nz; idz++)
                {
                    for (auto idy = 0; idy < g.ny; idy++)
                    {
                        for (auto idx = 0; idx < g.nx; idx++)
                        {
                            T px, py, pz;
                            grid::centerOfCell(g, idx, idy, idz, &px, &py, &pz);

                            // Velocity term (linear interpolation)
                            T vlx, vly, vlz;
                            facefield::in<T>
                                        (datax.data(), datay.data(), dataz.data(),
                                         g, idx, idy, idz,
                                         true,
                                         px, py, pz,
                                         &vlx, &vly, &vlz);

                            outStream << vlx << std::endl;
                        }
                    }
                }

                outStream << std::endl;
                outStream << "SCALARS velocityY double" << std::endl;
                outStream << "LOOKUP_TABLE default" << std::endl;
                for (auto idz = 0; idz < g.nz; idz++)
                {
                    for (auto idy = 0; idy < g.ny; idy++)
                    {
                        for (auto idx = 0; idx < g.nx; idx++)
                        {
                            T px, py, pz;
                            grid::centerOfCell(g, idx, idy, idz, &px, &py, &pz);

                            // Velocity term (linear interpolation)
                            T vlx, vly, vlz;
                            facefield::in<T>
                                        (datax.data(), datay.data(), dataz.data(),
                                         g, idx, idy, idz,
                                         true,
                                         px, py, pz,
                                         &vlx, &vly, &vlz);

                            outStream << vly << std::endl;
                        }
                    }
                }

                outStream << std::endl;
                outStream << "SCALARS velocityZ double" << std::endl;
                outStream << "LOOKUP_TABLE default" << std::endl;
                for (auto idz = 0; idz < g.nz; idz++)
                {
                    for (auto idy = 0; idy < g.ny; idy++)
                    {
                        for (auto idx = 0; idx < g.nx; idx++)
                        {
                            T px, py, pz;
                            grid::centerOfCell(g, idx, idy, idz, &px, &py, &pz);

                            // Velocity term (linear interpolation)
                            T vlx, vly, vlz;
                            facefield::in<T>
                                        (datax.data(), datay.data(), dataz.data(),
                                         g, idx, idy, idz,
                                         true,
                                         px, py, pz,
                                         &vlx, &vly, &vlz);

                            outStream << vlz << std::endl;
                        }
                    }
                }
            }
            else
            {
                throw std::runtime_error(std::string("Could not open file ") + fileName);
            }
            outStream.close();
        }

    };
};


#endif //PAR2_FACEFIELD_CUH

```

# Geometry/Interpolation.cuh

```cuh
/**
* @file Point.cuh
* @brief Header file for interpolation.
*
* @author Calogero B. Rizzo
*
* @copyright This file is part of the PAR2 software.
*            Copyright (C) 2018 Calogero B. Rizzo
*
* @license This program is free software: you can redistribute it and/or modify
*          it under the terms of the GNU General Public License as published by
*          the Free Software Foundation, either version 3 of the License, or
*          (at your option) any later version.
*
*          This program is distributed in the hope that it will be useful,
*          but WITHOUT ANY WARRANTY; without even the implied warranty of
*          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*          GNU General Public License for more details.
*
*          You should have received a copy of the GNU General Public License
*          along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef PAR2_INTERPOLATION_CUH
#define PAR2_INTERPOLATION_CUH

namespace par2
{

    namespace interpolation
    {
        /**
        * @brief Normalized linear interpolation.
        * @param x Value between 0 and 1
        * @param v0 Value in 0
        * @param v1 Value in 1
        * @return Linear interpolation
        */
        template<typename T>
        __host__ __device__
        T linear(T x, T v0, T v1)
        {
            return v0*(1-x) + v1*x;
        }

        /**
        * @brief Normalized trilinear interpolation.
        * @param x Value between 0 and 1
        * @param y Value between 0 and 1
        * @param z Value between 0 and 1
        * @param v000 Value in 000
        * @param v100 Value in 100
        * @param v010 Value in 010
        * @param v110 Value in 110
        * @param v001 Value in 001
        * @param v101 Value in 101
        * @param v011 Value in 011
        * @param v111 Value in 111
        * @return Trilinear interpolation
        */
        template<typename T>
        __host__ __device__
        T trilinear(T x, T y, T z,
                    T v000, T v100, T v010, T v110,
                    T v001, T v101, T v011, T v111)
        {
            T v00 = linear(x, v000, v100);
            T v01 = linear(x, v001, v101);
            T v10 = linear(x, v010, v110);
            T v11 = linear(x, v011, v111);

            T v0 = linear(y, v00, v10);
            T v1 = linear(y, v01, v11);

            return linear(z, v0, v1);
        }

        /**
        * @brief Normalized trilinear interpolation of x-derivative.
        * @param x Value between 0 and 1
        * @param y Value between 0 and 1
        * @param z Value between 0 and 1
        * @param dx Block size in x-direction
        * @param v000 Value in 000
        * @param v100 Value in 100
        * @param v010 Value in 010
        * @param v110 Value in 110
        * @param v001 Value in 001
        * @param v101 Value in 101
        * @param v011 Value in 011
        * @param v111 Value in 111
        * @return Trilinear interpolation of x-derivative
        */
        template<typename T>
        __host__ __device__
        T trilinearDevX(T x, T y, T z, T dx,
                        T v000, T v100, T v010, T v110,
                        T v001, T v101, T v011, T v111)
        {
            // vYZ
            T v00 = (v100 - v000)/dx;
            T v01 = (v101 - v001)/dx;
            T v10 = (v110 - v010)/dx;
            T v11 = (v111 - v011)/dx;

            // vZ
            T v0 = linear(y, v00, v10);
            T v1 = linear(y, v01, v11);

            return linear(z, v0, v1);
        }

        /**
        * @brief Normalized trilinear interpolation of y-derivative.
        * @param x Value between 0 and 1
        * @param y Value between 0 and 1
        * @param z Value between 0 and 1
        * @param dy Block size in y-direction
        * @param v000 Value in 000
        * @param v100 Value in 100
        * @param v010 Value in 010
        * @param v110 Value in 110
        * @param v001 Value in 001
        * @param v101 Value in 101
        * @param v011 Value in 011
        * @param v111 Value in 111
        * @return Trilinear interpolation of y-derivative
        */
        template<typename T>
        __host__ __device__
        T trilinearDevY(T x, T y, T z, T dy,
                        T v000, T v100, T v010, T v110,
                        T v001, T v101, T v011, T v111)
        {
            // vXZ
            T v00 = (v010 - v000)/dy;
            T v01 = (v011 - v001)/dy;
            T v10 = (v110 - v100)/dy;
            T v11 = (v111 - v101)/dy;

            // vZ
            T v0 = linear(x, v00, v10);
            T v1 = linear(x, v01, v11);

            return linear(z, v0, v1);
        }

        /**
        * @brief Normalized trilinear interpolation of z-derivative.
        * @param x Value between 0 and 1
        * @param y Value between 0 and 1
        * @param z Value between 0 and 1
        * @param dz Block size in z-direction
        * @param v000 Value in 000
        * @param v100 Value in 100
        * @param v010 Value in 010
        * @param v110 Value in 110
        * @param v001 Value in 001
        * @param v101 Value in 101
        * @param v011 Value in 011
        * @param v111 Value in 111
        * @return Trilinear interpolation of z-derivative
        */
        template<typename T>
        __host__ __device__
        T trilinearDevZ(T x, T y, T z, T dz,
                        T v000, T v100, T v010, T v110,
                        T v001, T v101, T v011, T v111)
        {
            // vXY
            T v00 = (v001 - v000)/dz;
            T v01 = (v011 - v010)/dz;
            T v10 = (v101 - v100)/dz;
            T v11 = (v111 - v110)/dz;

            // vY
            T v0 = linear(x, v00, v10);
            T v1 = linear(x, v01, v11);

            return linear(y, v0, v1);
        }

    }
}


#endif //PAR2_INTERPOLATION_CUH

```

# Geometry/Point.cuh

```cuh
/**
* @file Point.cuh
* @brief Header file for point.
*
* @author Calogero B. Rizzo
*
* @copyright This file is part of the PAR2 software.
*            Copyright (C) 2018 Calogero B. Rizzo
*
* @license This program is free software: you can redistribute it and/or modify
*          it under the terms of the GNU General Public License as published by
*          the Free Software Foundation, either version 3 of the License, or
*          (at your option) any later version.
*
*          This program is distributed in the hope that it will be useful,
*          but WITHOUT ANY WARRANTY; without even the implied warranty of
*          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*          GNU General Public License for more details.
*
*          You should have received a copy of the GNU General Public License
*          along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef PAR2_POINT_CUH
#define PAR2_POINT_CUH

namespace par2
{

    namespace point
    {
        /**
        * @brief Compute the distance between two points.
        * @param px0 x-component of p0
        * @param py0 y-component of p0
        * @param pz0 z-component of p0
        * @param px1 x-component of p1
        * @param py1 y-component of p1
        * @param pz1 z-component of p1
        * @return Distance between p0 and p1
        */
        template<typename T>
        __host__ __device__
        T distance(T px0, T py0, T pz0, T px1, T py1, T pz1)
        {
            return sqrt((px0 - px1) * (px0 - px1) + (py0 - py1) * (py0 - py1) + (pz0 - pz1) * (pz0 - pz1));
        };

        /**
        * @brief Compute the square of the distance between two points.
        * @param px0 x-component of p0
        * @param py0 y-component of p0
        * @param pz0 z-component of p0
        * @param px1 x-component of p1
        * @param py1 y-component of p1
        * @param pz1 z-component of p1
        * @return Square of the distance between p0 and p1
        */
        template<typename T>
        __host__ __device__
        T distance2(T px0, T py0, T pz0, T px1, T py1, T pz1)
        {
            return (px0 - px1) * (px0 - px1) + (py0 - py1) * (py0 - py1) + (pz0 - pz1) * (pz0 - pz1);
        };

        /**
        * @brief Summation of the coordinates of two points.
        * @param px0 x-component of p0
        * @param py0 y-component of p0
        * @param pz0 z-component of p0
        * @param px1 x-component of p1
        * @param py1 y-component of p1
        * @param pz1 z-component of p1
        * @param rx x-component of p0 + p1
        * @param ry y-component of p0 + p1
        * @param rz z-component of p0 + p1
        */
        template<typename T>
        __host__ __device__
        void plus(T px0, T py0, T pz0, T px1, T py1, T pz1, T* rx, T* ry, T* rz)
        {
            *rx = px0 + px1;
            *ry = py0 + py1;
            *rz = pz0 + pz1;
        };

        /**
        * @brief Subtraction the coordinates of two points.
        * @param px0 x-component of p0
        * @param py0 y-component of p0
        * @param pz0 z-component of p0
        * @param px1 x-component of p1
        * @param py1 y-component of p1
        * @param pz1 z-component of p1
        * @param rx x-component of p0 - p1
        * @param ry y-component of p0 - p1
        * @param rz z-component of p0 - p1
        */
        template<typename T>
        __host__ __device__
        void minus(T px0, T py0, T pz0, T px1, T py1, T pz1, T* rx, T* ry, T* rz)
        {
            *rx = px0 - px1;
            *ry = py0 - py1;
            *rz = pz0 - pz1;
        };

        /**
        * @brief Multiplication between a point and a scalar.
        * @param px0 x-component of p0
        * @param py0 y-component of p0
        * @param pz0 z-component of p0
        * @param val scalar
        * @param rx x-component of val*p0
        * @param ry y-component of val*p0
        * @param rz z-component of val*p0
        */
        template<typename T>
        __host__ __device__
        void multiply(T px0, T py0, T pz0, T val, T* rx, T* ry, T* rz)
        {
            *rx = px0*val;
            *ry = py0*val;
            *rz = pz0*val;
        };

        /**
        * @brief Division between a point and a scalar.
        * @param px0 x-component of p0
        * @param py0 y-component of p0
        * @param pz0 z-component of p0
        * @param val scalar
        * @param rx x-component of p0/val
        * @param ry y-component of p0/val
        * @param rz z-component of p0/val
        */
        template<typename T>
        __host__ __device__
        void divide(T px0, T py0, T pz0, T val, T* rx, T* ry, T* rz)
        {
            *rx = px0/val;
            *ry = py0/val;
            *rz = pz0/val;
        };

    };

}

#endif //PAR2_POINT_CUH

```

# Geometry/Vector.cuh

```cuh
/**
* @file Vector.cuh
* @brief Header file for vector.
*
* @author Calogero B. Rizzo
*
* @copyright This file is part of the PAR2 software.
*            Copyright (C) 2018 Calogero B. Rizzo
*
* @license This program is free software: you can redistribute it and/or modify
*          it under the terms of the GNU General Public License as published by
*          the Free Software Foundation, either version 3 of the License, or
*          (at your option) any later version.
*
*          This program is distributed in the hope that it will be useful,
*          but WITHOUT ANY WARRANTY; without even the implied warranty of
*          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*          GNU General Public License for more details.
*
*          You should have received a copy of the GNU General Public License
*          along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef PAR2_VECTOR_CUH
#define PAR2_VECTOR_CUH

namespace par2
{

    namespace vector
    {
        /**
        * @brief Compute the square of the Euclidean norm.
        * @param vx x-component
        * @param vy y-component
        * @param vz z-component
        * @return Square of the Euclidean norm
        */
        template<typename T>
        __host__ __device__
        T norm2(T vx, T vy, T vz)
        {
            return vx*vx + vy*vy + vz*vz;
        }

        /**
        * @brief Compute the Euclidean norm.
        * @param vx x-component
        * @param vy y-component
        * @param vz z-component
        * @return Euclidean norm
        */
        template<typename T>
        __host__ __device__
        T norm(T vx, T vy, T vz)
        {
            return sqrt(norm2(vx, vy, vz));
        }

        /**
        * @brief Dot product between two vectors.
        * @param vx0 x-component of v0
        * @param vy0 y-component of v0
        * @param vz0 z-component of v0
        * @param vx1 x-component of v1
        * @param vy1 y-component of v1
        * @param vz1 z-component of v1
        * @return v0 dot v1
        */
        template<typename T>
        __host__ __device__
        T dot(T vx0, T vy0, T vz0, T vx1, T vy1, T vz1)
        {
            return vx0 * vx1 + vy0 * vy1 + vz0 * vz1;
        }

    }

}

#endif //PAR2_VECTOR_CUH

```

# Particles/MoveParticle.cuh

```cuh
/**
 * @file PParticles.cu
 * @brief MoveParticle functor.
 *
 * @author Calogero B. Rizzo
 *
 * @copyright This file is part of the PAR2 software.
 *            Copyright (C) 2018 Calogero B. Rizzo
 *
 * @license This program is free software: you can redistribute it and/or modify
 *          it under the terms of the GNU General Public License as published by
 *          the Free Software Foundation, either version 3 of the License, or
 *          (at your option) any later version.
 *
 *          This program is distributed in the hope that it will be useful,
 *          but WITHOUT ANY WARRANTY; without even the implied warranty of
 *          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *          GNU General Public License for more details.
 *
 *          You should have received a copy of the GNU General Public License
 *          along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

#ifndef PAR2_MOVEPARTICLE_CUH
#define PAR2_MOVEPARTICLE_CUH

#include "../Geometry/CartesianGrid.cuh"
#include "../Geometry/CornerField.cuh"
#include "../Geometry/FaceField.cuh"
#include "../Geometry/Vector.cuh"
#include <curand_kernel.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/random/normal_distribution.h>
#include <thrust/tuple.h>

namespace par2 {
/**
 * @struct MoveParticle
 * @brief Thrust functor for one step of the particle tracking method.
 * @tparam T Float number precision
 */
template <typename T> struct MoveParticle {
  // Raw pointer to velocity vectors (facefield)
  T *datax;
  T *datay;
  T *dataz;
  // Raw pointer to velocity vectors (cornerfield or cellfield)
  T *cdatax;
  T *cdatay;
  T *cdataz;
  // Grid and physical variables
  grid::Grid<T> grid;
  T dt;
  T molecularDiffusion;
  T alphaL;
  T alphaT;
  // Raw pointer to curand state vector
  curandState_t *states;
  // If true, use trilinear interpolation
  bool useTrilinearCorrection;

  // Contadores de imágenes (unwrapping) por partícula
  int *imgY = nullptr;
  int *imgZ = nullptr;

  // (opcional) buffers para guardar coordenadas unwrapped en tiempo real
  T *yUnwrap = nullptr;
  T *zUnwrap = nullptr;

  // Longitudes periódicas
  T Ly = 0;
  T Lz = 0;

  // Base index for current kernel chunk to compute global particle id
  unsigned int baseIndex = 0;

  /**
   * @brief Constructor.
   * @param _grid Grid where the velocity is defined
   */
  MoveParticle(const grid::Grid<T> &_grid) : grid(_grid), dt(0.0){};

  /**
   * @brief Initialize the functor.
   * @param _datax Velocity vector (x-direction)
   * @param _datay Velocity vector (y-direction)
   * @param _dataz Velocity vector (z-direction)
   * @param _molecularDiffusion Effective molecular diffusion
   * @param _alphaL Longitudinal dispersivity
   * @param _alphaT Transverse dispersivity
   * @param _states Curand states
   * @param _useTrilinearCorrection True if trilinear correction is used
   */
  void initialize(thrust::device_vector<T> &_datax,
                  thrust::device_vector<T> &_datay,
                  thrust::device_vector<T> &_dataz,
                  thrust::device_vector<T> &_cdatax,
                  thrust::device_vector<T> &_cdatay,
                  thrust::device_vector<T> &_cdataz, T _molecularDiffusion,
                  T _alphaL, T _alphaT,
                  thrust::device_vector<curandState_t> &_states,
                  bool _useTrilinearCorrection) {
    datax = thrust::raw_pointer_cast(_datax.data());
    datay = thrust::raw_pointer_cast(_datay.data());
    dataz = thrust::raw_pointer_cast(_dataz.data());

    cdatax = thrust::raw_pointer_cast(_cdatax.data());
    cdatay = thrust::raw_pointer_cast(_cdatay.data());
    cdataz = thrust::raw_pointer_cast(_cdataz.data());

    molecularDiffusion = _molecularDiffusion;
    alphaL = _alphaL;
    alphaT = _alphaT;
    states = thrust::raw_pointer_cast(_states.data());

    useTrilinearCorrection = _useTrilinearCorrection;
  }

  __host__ __device__ inline void setUnwrapBuffers(int *_imgY, int *_imgZ,
                                                   T *_yUnwrap, T *_zUnwrap,
                                                   T _Ly, T _Lz) {
    imgY = _imgY;
    imgZ = _imgZ;
    yUnwrap = _yUnwrap;
    zUnwrap = _zUnwrap;
    Ly = _Ly;
    Lz = _Lz;
  }

  /**
   * @brief Set time step.
   * @param _dt Time step
   */
  void setTimeStep(T _dt) { dt = _dt; }

  __host__ __device__ inline void setBaseIndex(unsigned int base) {
    baseIndex = base;
  }

  using Position = thrust::tuple<T, T, T, unsigned int>;

  /**
   * @brief Execute one step of the particle tracking method
   *        on one particle.
   * @param p Initial position of the particle
   * @return Final position of the particle
   */
  __device__ Position operator()(Position p) const {
    int idx, idy, idz;
    grid::idPoint(grid, thrust::get<0>(p), thrust::get<1>(p), thrust::get<2>(p),
                  &idx, &idy, &idz);

    int id = grid::mergeId(grid, idx, idy, idz);

    bool idValid = grid::validId<T>(grid, idx, idy, idz);

    // Velocity term (linear interpolation)
    T vlx, vly, vlz;
    facefield::in<T>(datax, datay, dataz, grid, idx, idy, idz, idValid,
                     thrust::get<0>(p), thrust::get<1>(p), thrust::get<2>(p),
                     &vlx, &vly, &vlz);

    T vcx, vcy, vcz;
    if (useTrilinearCorrection) {
      // Velocity correction div(D) (trilinear interpolation)
      cornerfield::velocityCorrection<T>(
          cdatax, cdatay, cdataz, grid, idx, idy, idz, idValid,
          thrust::get<0>(p), thrust::get<1>(p), thrust::get<2>(p),
          molecularDiffusion, alphaL, alphaT, &vcx, &vcy, &vcz);
    } else {
      // Velocity correction div(D) (block-centered finite difference)
      vcx = idValid ? cdatax[id] : 0;
      vcy = idValid ? cdatay[id] : 0;
      vcz = idValid ? cdataz[id] : 0;
    }

    // Displacement Matrix (trilinear interpolation)
    T B00, B11, B22, B01, B02, B12;
    cornerfield::displacementMatrix<T>(
        cdatax, cdatay, cdataz, grid, idx, idy, idz, idValid, thrust::get<0>(p),
        thrust::get<1>(p), thrust::get<2>(p), molecularDiffusion, alphaL,
        alphaT, dt, &B00, &B11, &B22, &B01, &B02, &B12);

    // Random Displacement
    T xi0 = curand_normal_double(&states[thrust::get<3>(p)]);
    T xi1 = curand_normal_double(&states[thrust::get<3>(p)]);
    T xi2 = curand_normal_double(&states[thrust::get<3>(p)]);

    // Update positions
    T dpx, dpy, dpz;

    dpx = (idValid ? ((vlx + vcx) * dt + (B00 * xi0 + B01 * xi1 + B02 * xi2))
                   : 0);
    dpy = (idValid ? ((vly + vcy) * dt + (B01 * xi0 + B11 * xi1 + B12 * xi2))
                   : 0);
    if (grid.nz != 1)
      dpz = (idValid ? ((vlz + vcz) * dt + (B02 * xi0 + B12 * xi1 + B22 * xi2))
                     : 0);

    // Closed condition: particles cannot exit the domain
    // thrust::get<0>(p) +=
    //     (grid::validX(grid, thrust::get<0>(p) + dpx) ? dpx : 0);
    // thrust::get<1>(p) += (grid::validY(grid, thrust::get<1>(p) + dpy) ?
    // dpy :
    // 0); if (grid.nz != 1)
    //     thrust::get<2>(p) += (grid::validZ(grid, thrust::get<2>(p) + dpz)
    // ?
    //     dpz : 0);

    // X: no periódico (conservar la lógica actual)
    const unsigned int pid = baseIndex + thrust::get<3>(p);
    T x_old = thrust::get<0>(p);
    T x_try = x_old + dpx;
    T x_new = grid::validX(grid, x_try) ? x_try : x_old;
    thrust::get<0>(p) = x_new;

    // Y: periódico + contadores
    T y_old = thrust::get<1>(p);
    T y_new = y_old + dpy;
    int wrapsY = 0;
    while (y_new < grid.py) {
      y_new += Ly;
      wrapsY--;
    }
    while (y_new >= grid.py + Ly) {
      y_new -= Ly;
      wrapsY++;
    }
    thrust::get<1>(p) = y_new;
    if (wrapsY != 0 && imgY)
      imgY[pid] += wrapsY;

    // Z: periódico + contadores (solo si nz>1)
    T z_new_local = thrust::get<2>(p);
    if (grid.nz != 1) {
      T z_try = z_new_local + dpz;
      int wrapsZ = 0;
      while (z_try < grid.pz) {
        z_try += Lz;
        wrapsZ--;
      }
      while (z_try >= grid.pz + Lz) {
        z_try -= Lz;
        wrapsZ++;
      }
      z_new_local = z_try;
      thrust::get<2>(p) = z_new_local;
      if (wrapsZ != 0 && imgZ)
        imgZ[pid] += wrapsZ;

      if (zUnwrap)
        zUnwrap[pid] = z_new_local + static_cast<T>(imgZ ? imgZ[pid] : 0) * Lz;
    }

    // Buffers unwrapped (Y)
    if (yUnwrap)
      yUnwrap[pid] = y_new + static_cast<T>(imgY ? imgY[pid] : 0) * Ly;

    return p;
  }
};
} // namespace par2

#endif // PAR2_MOVEPARTICLE_CUH

```

# Particles/PParticles.cu

```cu
/**
 * @file PParticles.cu
 * @brief Implementation file for PParticles class.
 *
 * @author Calogero B. Rizzo
 *
 * @copyright This file is part of the PAR2 software.
 *            Copyright (C) 2018 Calogero B. Rizzo
 *
 * @license This program is free software: you can redistribute it and/or modify
 *          it under the terms of the GNU General Public License as published by
 *          the Free Software Foundation, either version 3 of the License, or
 *          (at your option) any later version.
 *
 *          This program is distributed in the hope that it will be useful,
 *          but WITHOUT ANY WARRANTY; without even the implied warranty of
 *          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *          GNU General Public License for more details.
 *
 *          You should have received a copy of the GNU General Public License
 *          along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

#include "../Geometry/CornerField.cuh"
#include "../Geometry/FaceField.cuh"

#include <algorithm>
#include <fstream>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/sort.h>

namespace par2 {
template <typename T> struct InitCURAND {
  unsigned long long seed;
  curandState_t *states;
  InitCURAND(unsigned long long _seed,
             thrust::device_vector<curandState_t> &_states) {
    seed = _seed;
    states = thrust::raw_pointer_cast(_states.data());
  }

  __device__ void operator()(unsigned int i) {
    curand_init(seed, i, 0, &states[i]);
  }
};

template <typename T> struct InitVolume {
  curandState_t *states;
  T p1x, p1y, p1z;
  T p2x, p2y, p2z;
  InitVolume(thrust::device_vector<curandState_t> &_states, T _p1x, T _p1y,
             T _p1z, T _p2x, T _p2y, T _p2z) {
    states = thrust::raw_pointer_cast(_states.data());
    p1x = _p1x;
    p1y = _p1y;
    p1z = _p1z;
    p2x = _p2x;
    p2y = _p2y;
    p2z = _p2z;
  }

  using Position = thrust::tuple<T, T, T>;

  __device__ Position operator()(unsigned int i) const {
    Position p;

    thrust::get<0>(p) = p1x + (p2x - p1x) * curand_uniform(&states[i]);
    thrust::get<1>(p) = p1y + (p2y - p1y) * curand_uniform(&states[i]);
    thrust::get<2>(p) = p1z + (p2z - p1z) * curand_uniform(&states[i]);

    return p;
  }
};

// Estructura para almacenar información de cada celda y su probabilidad
template <typename T> struct CellProbability {
  int ix, iy, iz; // Índices de celda
  T probability;  // Probabilidad normalizada
};

// Función para generar la distribución de probabilidad basada en velocidades
template <typename T>
void generateVelocityDistribution(const thrust::device_vector<T> &datax,
                                  const thrust::device_vector<T> &datay,
                                  const thrust::device_vector<T> &dataz,
                                  const grid::Grid<T> &grid, T p1x, T p1y,
                                  T p1z, T p2x, T p2y, T p2z,
                                  std::vector<CellProbability<T>> &cellProbs,
                                  thrust::device_vector<T> &cdf) {
  // 1. Determinar los índices de celda que abarcan la caja
  int minCellX, minCellY, minCellZ;
  int maxCellX, maxCellY, maxCellZ;

  // Encontrar las celdas que contienen los puntos extremos
  bool valid1 =
      grid::findCell(grid, p1x, p1y, p1z, &minCellX, &minCellY, &minCellZ);
  bool valid2 =
      grid::findCell(grid, p2x, p2y, p2z, &maxCellX, &maxCellY, &maxCellZ);

  // Si algún punto está fuera de la malla, usar límites seguros
  if (!valid1 || !valid2) {
    minCellX = 0;
    minCellY = 0;
    minCellZ = 0;
    maxCellX = grid.nx - 1;
    maxCellY = grid.ny - 1;
    maxCellZ = grid.nz - 1;
  }

  // Asegurar que maxCell >= minCell (por si los puntos están en orden inverso)
  if (maxCellX < minCellX)
    std::swap(maxCellX, minCellX);
  if (maxCellY < minCellY)
    std::swap(maxCellY, minCellY);
  if (maxCellZ < minCellZ)
    std::swap(maxCellZ, minCellZ);

  // Ajustar límites para que estén dentro del dominio
  minCellX = std::max(0, minCellX);
  minCellY = std::max(0, minCellY);
  minCellZ = std::max(0, minCellZ);

  maxCellX = std::min(grid.nx - 1, maxCellX);
  maxCellY = std::min(grid.ny - 1, maxCellY);
  maxCellZ = std::min(grid.nz - 1, maxCellZ);

  // 2. Calcular velocidades para cada celda y su probabilidad
  cellProbs.clear();

  T sumVelocity = 0;

  // Punteros raw para acceso más eficiente
  const T *dataxPtr = thrust::raw_pointer_cast(datax.data());
  const T *datayPtr = thrust::raw_pointer_cast(datay.data());
  const T *datazPtr = thrust::raw_pointer_cast(dataz.data());

  for (int iz = minCellZ; iz <= maxCellZ; iz++) {
    for (int iy = minCellY; iy <= maxCellY; iy++) {
      for (int ix = minCellX; ix <= maxCellX; ix++) {
        // Obtener centro de celda
        T cx, cy, cz;
        grid::centerOfCell<T>(grid, ix, iy, iz, &cx, &cy, &cz);

        // Verificar si el centro está dentro de la caja de inyección
        if (cx >= p1x && cx <= p2x && cy >= p1y && cy <= p2y && cz >= p1z &&
            cz <= p2z) {
          // Calcular velocidad en el centro
          T vx, vy, vz;
          par2::facefield::in<T>(dataxPtr, datayPtr, datazPtr, grid, ix, iy, iz,
                                 true, cx, cy, cz, &vx, &vy, &vz);

          // Calcular magnitud de velocidad
          // T velocity = sqrt(vx * vx + vy * vy + vz * vz);
          T velocity2 = vx * vx + vy * vy + vz * vz;

          // Asignar probabilidad mínima para evitar celdas con probabilidad
          // cero
          if (velocity2 < 1e-10)
            velocity2 = 1e-10;

          // Añadir a la lista
          CellProbability<T> cellProb;
          cellProb.ix = ix;
          cellProb.iy = iy;
          cellProb.iz = iz;
          cellProb.probability = velocity2;
          cellProbs.push_back(cellProb);

          sumVelocity += velocity2;
        }
      }
    }
  }

  // 3. Normalizar probabilidades y construir CDF
  T cumulativeProb = 0;
  cdf.resize(cellProbs.size() + 1);
  cdf[0] = 0;

  for (size_t i = 0; i < cellProbs.size(); i++) {
    cellProbs[i].probability /= sumVelocity;
    cumulativeProb += cellProbs[i].probability;
    cdf[i + 1] = cumulativeProb;
  }
}

// Functor para generar partículas según la distribución de velocidades

template <typename T> struct InitVolumeByVelocityDistribution {
  curandState_t *states;
  grid::Grid<T> grid;
  const T *cdfPtr;
  int numCells;
  int *d_cellX_ptr;
  int *d_cellY_ptr;
  int *d_cellZ_ptr;

  InitVolumeByVelocityDistribution(
      thrust::device_vector<curandState_t> &_states,
      const thrust::device_vector<T> &_cdf, const grid::Grid<T> &_grid,
      int *_d_cellX_ptr, int *_d_cellY_ptr, int *_d_cellZ_ptr, int _numCells)
      : grid(_grid), numCells(_numCells) {
    states = thrust::raw_pointer_cast(_states.data());
    cdfPtr = thrust::raw_pointer_cast(_cdf.data());
    d_cellX_ptr = _d_cellX_ptr;
    d_cellY_ptr = _d_cellY_ptr;
    d_cellZ_ptr = _d_cellZ_ptr;
  }

  using Position = thrust::tuple<T, T, T>;

  __device__ Position operator()(unsigned int i) const {
    Position p;

    if (numCells <= 0) {
      // Si no hay celdas válidas, devolver posición inválida
      thrust::get<0>(p) = T(0);
      thrust::get<1>(p) = T(0);
      thrust::get<2>(p) = T(0);
      return p;
    }

    // 1. Generar un número aleatorio para seleccionar la celda
    T r = curand_uniform(&states[i]);

    // 2. Buscar en la CDF (búsqueda binaria)
    int low = 0;
    int high = numCells;

    while (low < high) {
      int mid = (low + high) / 2;
      if (cdfPtr[mid] < r)
        low = mid + 1;
      else
        high = mid;
    }

    // Obtener celda seleccionada (ajustar por posible error numérico)
    int selectedCell = min(low - 1, numCells - 1);
    if (selectedCell < 0)
      selectedCell = 0;

    // 3. Obtener índices de la celda seleccionada
    int ix = d_cellX_ptr[selectedCell];
    int iy = d_cellY_ptr[selectedCell];
    int iz = d_cellZ_ptr[selectedCell];

    // 4. Generar una posición aleatoria dentro de la celda
    T cx, cy, cz;
    grid::centerOfCell<T>(grid, ix, iy, iz, &cx, &cy, &cz);

    // Generar offset aleatorio dentro de la celda (entre -0.5 y 0.5 del tamaño
    // de celda)
    T offsetX = (curand_uniform(&states[i]) - 0.5) * grid.dx;
    T offsetY = (curand_uniform(&states[i]) - 0.5) * grid.dy;
    T offsetZ = (curand_uniform(&states[i]) - 0.5) * grid.dz;

    // Posición final
    thrust::get<0>(p) = cx + offsetX;
    thrust::get<1>(p) = cy + offsetY;
    thrust::get<2>(p) = cz + offsetZ;

    return p;
  }
};

template <typename T>
PParticles<T>::PParticles(const grid::Grid<T> &_grid,
                          thrust::device_vector<T> &&_datax,
                          thrust::device_vector<T> &&_datay,
                          thrust::device_vector<T> &&_dataz,
                          T _molecularDiffusion, T _alphaL, T _alphaT,
                          unsigned int _nParticles, long int _seed,
                          bool _useTrilinearCorrection)
    : nParticles(_nParticles), molecularDiffusion(_molecularDiffusion),
      alphaL(_alphaL), alphaT(_alphaT), grid(_grid), moveParticle(_grid),
      useTrilinearCorrection(_useTrilinearCorrection) {
  cx.resize(nParticles);
  cy.resize(nParticles);
  cz.resize(nParticles);

  datax = std::move(_datax);
  datay = std::move(_datay);
  dataz = std::move(_dataz);

  if (useTrilinearCorrection) {
    // Inicializar vectores de salida directamente en GPU
    par2::cornerfield::build(grid, cdatax);
    par2::cornerfield::build(grid, cdatay);
    par2::cornerfield::build(grid, cdataz);

    // Usar versión GPU que opera completamente en device
    par2::cornerfield::computeCornerVelocitiesGPU(grid, datax, datay, dataz,
                                                  cdatax, cdatay, cdataz);
  } else {
    // Inicializar directamente en device para evitar copia adicional
    par2::cellfield::build(grid, cdatax);
    par2::cellfield::build(grid, cdatay);
    par2::cellfield::build(grid, cdataz);

    // Usar versión GPU de la corrección de deriva
    par2::cellfield::computeDriftCorrectionGPU(
        grid, datax, datay, dataz, cdatax, cdatay, cdataz, molecularDiffusion,
        alphaL, alphaT);
  }

  states.resize(maxParticles);
  thrust::counting_iterator<unsigned int> count(0);
  thrust::for_each(count, count + maxParticles, InitCURAND<T>(_seed, states));

  moveParticle.initialize(datax, datay, dataz, cdatax, cdatay, cdataz,
                          molecularDiffusion, alphaL, alphaT, states,
                          useTrilinearCorrection);

  // Setup unwrapping buffers (allocated after knowing nParticles and grid)
  imgY_.assign(nParticles, 0);
  if (grid.nz > 1)
    imgZ_.assign(nParticles, 0);
  yU_.assign(nParticles, 0);
  if (grid.nz > 1)
    zU_.assign(nParticles, 0);

  Ly_ = grid.dy * grid.ny;
  Lz_ = grid.dz * grid.nz;

  moveParticle.setUnwrapBuffers(
      imgY_.empty() ? nullptr : thrust::raw_pointer_cast(imgY_.data()),
      imgZ_.empty() ? nullptr : thrust::raw_pointer_cast(imgZ_.data()),
      yU_.empty() ? nullptr : thrust::raw_pointer_cast(yU_.data()),
      zU_.empty() ? nullptr : thrust::raw_pointer_cast(zU_.data()), Ly_, Lz_);

  cudaDeviceSynchronize();
}

template <typename T> unsigned int PParticles<T>::size() const {
  return nParticles;
}

template <typename T>
void PParticles<T>::initializeBox(T p1x, T p1y, T p1z, T p2x, T p2y, T p2z,
                                  bool velocityBased) {
  thrust::counting_iterator<unsigned int> count(0);
  auto pBeg = thrust::make_zip_iterator(
      thrust::make_tuple(cx.begin(), cy.begin(), cz.begin()));

  if (velocityBased) {
    // Variables para la distribución
    std::vector<CellProbability<T>> cellProbs;
    thrust::device_vector<T> cdf;

    // Generar la distribución de probabilidad basada en velocidad
    generateVelocityDistribution(datax, datay, dataz, grid, p1x, p1y, p1z, p2x,
                                 p2y, p2z, cellProbs, cdf);

    // Si no hay celdas válidas, usar distribución uniforme
    if (cellProbs.empty()) {
      auto functor = InitVolume<T>(states, p1x, p1y, p1z, p2x, p2y, p2z);

      for (auto i = 0; i * maxParticles < nParticles; i++) {
        unsigned int kernelSize = maxParticles;
        if (kernelSize > nParticles - i * maxParticles) {
          kernelSize = nParticles - i * maxParticles;
        }
        thrust::transform(count, count + kernelSize, pBeg + i * maxParticles,
                          functor);
      }
    } else {
      // Copiar índices de celda a device y obtener punteros raw
      thrust::device_vector<int> d_cellX(cellProbs.size());
      thrust::device_vector<int> d_cellY(cellProbs.size());
      thrust::device_vector<int> d_cellZ(cellProbs.size());
      for (size_t i = 0; i < cellProbs.size(); i++) {
        d_cellX[i] = cellProbs[i].ix;
        d_cellY[i] = cellProbs[i].iy;
        d_cellZ[i] = cellProbs[i].iz;
      }
      int *d_cellX_ptr = thrust::raw_pointer_cast(d_cellX.data());
      int *d_cellY_ptr = thrust::raw_pointer_cast(d_cellY.data());
      int *d_cellZ_ptr = thrust::raw_pointer_cast(d_cellZ.data());
      auto functor = InitVolumeByVelocityDistribution<T>(
          states, cdf, grid, d_cellX_ptr, d_cellY_ptr, d_cellZ_ptr,
          cellProbs.size());
      for (auto i = 0; i * maxParticles < nParticles; i++) {
        unsigned int kernelSize = maxParticles;
        if (kernelSize > nParticles - i * maxParticles) {
          kernelSize = nParticles - i * maxParticles;
        }
        thrust::transform(count, count + kernelSize, pBeg + i * maxParticles,
                          functor);
      }
    }
  } else {
    // Comportamiento original con distribución uniforme
    auto functor = InitVolume<T>(states, p1x, p1y, p1z, p2x, p2y, p2z);

    for (auto i = 0; i * maxParticles < nParticles; i++) {
      unsigned int kernelSize = maxParticles;
      if (kernelSize > nParticles - i * maxParticles) {
        kernelSize = nParticles - i * maxParticles;
      }
      thrust::transform(count, count + kernelSize, pBeg + i * maxParticles,
                        functor);
    }
  }
}

template <typename T> void PParticles<T>::move(T dt) {
  thrust::counting_iterator<unsigned int> count(0);
  moveParticle.setTimeStep(dt);

  for (auto i = 0; i * maxParticles < nParticles; i++) {
    unsigned int kernelSize = maxParticles;
    if (kernelSize > nParticles - i * maxParticles) {
      kernelSize = nParticles - i * maxParticles;
    }

    moveParticle.setBaseIndex(i * maxParticles);

    // ensure id is the 4th element for the functor (local id per chunk)
    auto pBeg = thrust::make_zip_iterator(thrust::make_tuple(
        cx.begin() + i * maxParticles, cy.begin() + i * maxParticles,
        cz.begin() + i * maxParticles, count));
    // auto pEnd = thrust::make_zip_iterator(
    //     thrust::make_tuple(cx.end(),   cy.end()  , cz.end()  ,
    //     count+kernelSize));

    thrust::transform(pBeg, pBeg + kernelSize, pBeg, moveParticle);
  }
  cudaDeviceSynchronize();
}

template <typename T>
void PParticles<T>::exportCSV(const std::string &fileName) const {
  // Copy to host memory
  thrust::host_vector<T> hx = cx;
  thrust::host_vector<T> hy = cy;
  thrust::host_vector<T> hz = cz;

  std::ofstream outStream;
  outStream.open(fileName);
  if (outStream.is_open()) {
    outStream << "id,x coord,y coord,z coord" << std::endl;
    for (unsigned int i = 0; i < nParticles; i++) {
      outStream << i << "," << hx[i] << "," << hy[i] << "," << hz[i]
                << std::endl;
    }
  } else {
    throw std::runtime_error(std::string("Could not open file ") + fileName);
  }
  outStream.close();
}

template <typename T> struct isInside {
  T plane;

  T p1x, p1y, p1z;
  T p2x, p2y, p2z;
  isInside(T _p1x, T _p1y, T _p1z, T _p2x, T _p2y, T _p2z) {
    p1x = _p1x;
    p1y = _p1y;
    p1z = _p1z;
    p2x = _p2x;
    p2y = _p2y;
    p2z = _p2z;
  }

  using Position = thrust::tuple<T, T, T>;

  __device__ bool operator()(Position p) const {
    return (p1x <= thrust::get<0>(p) && thrust::get<0>(p) <= p2x) &&
           (p1y <= thrust::get<1>(p) && thrust::get<1>(p) <= p2y) &&
           (p1z <= thrust::get<2>(p) && thrust::get<2>(p) <= p2z);
  }
};

template <typename T>
T PParticles<T>::concentrationBox(T p1x, T p1y, T p1z, T p2x, T p2y,
                                  T p2z) const {
  auto pBeg = thrust::make_zip_iterator(
      thrust::make_tuple(cx.begin(), cy.begin(), cz.begin()));
  auto pEnd = thrust::make_zip_iterator(
      thrust::make_tuple(cx.end(), cy.end(), cz.end()));

  return thrust::count_if(pBeg, pEnd,
                          isInside<T>(p1x, p1y, p1z, p2x, p2y, p2z)) /
         T(nParticles);
}

template <typename T> struct isAfter {
  T plane;

  isAfter(T _plane) : plane(_plane){};

  __device__ bool operator()(T x) { return x > plane; }
};

template <typename T> T PParticles<T>::concentrationAfterX(T xplane) const {
  return thrust::count_if(cx.begin(), cx.end(), isAfter<T>(xplane)) /
         T(nParticles);
}

template <typename T> thrust::device_vector<T> PParticles<T>::getX() const {
  return cx;
}

template <typename T> thrust::device_vector<T> PParticles<T>::getY() const {
  return cy;
}

template <typename T> thrust::device_vector<T> PParticles<T>::getZ() const {
  return cz;
}

} // namespace par2

```

# Particles/PParticles.cuh

```cuh
/**
 * @file PParticles.cuh
 * @brief Header file for PParticles class.
 *
 * @author Calogero B. Rizzo
 *
 * @copyright This file is part of the PAR2 software.
 *            Copyright (C) 2018 Calogero B. Rizzo
 *
 * @license This program is free software: you can redistribute it and/or modify
 *          it under the terms of the GNU General Public License as published by
 *          the Free Software Foundation, either version 3 of the License, or
 *          (at your option) any later version.
 *
 *          This program is distributed in the hope that it will be useful,
 *          but WITHOUT ANY WARRANTY; without even the implied warranty of
 *          MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *          GNU General Public License for more details.
 *
 *          You should have received a copy of the GNU General Public License
 *          along with this program.  If not, see
 * <http://www.gnu.org/licenses/>.
 */

#ifndef PAR2_PPARTICLES_CUH
#define PAR2_PPARTICLES_CUH

#include "../Geometry/CartesianGrid.cuh"
#include "../Geometry/FaceField.cuh"
#include "MoveParticle.cuh"
#include <ctime>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace par2 {
/**
 * @class PParticles
 * @brief Class managing a cloud of particles. It provides functions
 *        to initialize and move the particles on device.
 * @tparam T Float number precision
 */
template <typename T> class PParticles {
public:
  /**
   * @brief Constructor.
   * @param _grid Grid where the velocity is defined
   * @param _datax Velocity vector (x-direction)
   * @param _datay Velocity vector (y-direction)
   * @param _dataz Velocity vector (z-direction)
   * @param _molecularDiffusion Effective molecular diffusion
   * @param _alphaL Longitudinal dispersivity
   * @param _alphaT Transverse dispersivity
   * @param _nParticles Total number of particles
   * @param _seed Seed for pseudo-random number generator
   * @param _useTrilinearCorrection True if trilinear correction is used
   */
  PParticles(const grid::Grid<T> &_grid, thrust::device_vector<T> &&_datax,
             thrust::device_vector<T> &&_datay,
             thrust::device_vector<T> &&_dataz, T _molecularDiffusion,
             T _alphaL, T _alphaT, unsigned int _nParticles,
             long int _seed = time(NULL), bool _useTrilinearCorrection = true);

  /**
   * @brief Number of particles in the system.
   * @return Number of particles
   */
  unsigned int size() const;

  /**
   * @brief Initialize particles inside a box defined by
   *        two points p1 and p2. The particles are uniformly
   *        distributed inside the box.
   * @param p1x x-component of p1
   * @param p1y y-component of p1
   * @param p1z z-component of p1
   * @param p2x x-component of p2
   * @param p2y y-component of p2
   * @param p2z z-component of p2
   */
  void initializeBox(T p1x, T p1y, T p1z, T p2x, T p2y, T p2z,
                     bool velocityBased = false);

  /**
   * @brief Execute one step of the particle tracking method using
   *        a time step dt.
   * @param dt Time step
   */
  void move(T dt);

  /**
   * @brief Export all the particles in a csv file.
   *        WARNING: this function is computationally expensive
   *        and should be avoided if computation time is critical.
   * @param fileName Path to the csv file
   */
  void exportCSV(const std::string &fileName) const;

  /**
   * @brief Compute the percentage of particles inside a box
   *        defined by two points p1 and p2.
   * @param p1x x-component of p1
   * @param p1y y-component of p1
   * @param p1z z-component of p1
   * @param p2x x-component of p2
   * @param p2y y-component of p2
   * @param p2z z-component of p2
   * @return Percentage of particles inside the box.
   */
  T concentrationBox(T p1x, T p1y, T p1z, T p2x, T p2y, T p2z) const;

  /**
   * @brief Compute the percentage of particles that crossed
   *        a plane orthogonal to the x-axis (i.e., px > xplane).
   * @param xplane Location of the plane
   * @return Percentage of particles after the plane.
   */
  T concentrationAfterX(T xplane) const;

  /**
   * @brief Getters for particle positions.
   * @return Particle positions (x,y,z) as thrust::device_vector
   */
  thrust::device_vector<T> getX() const;
  thrust::device_vector<T> getY() const;
  thrust::device_vector<T> getZ() const;

  const T *xPtr() const { return thrust::raw_pointer_cast(cx.data()); }
  const T *zPtr() const { return thrust::raw_pointer_cast(cz.data()); }
  const T *yPtr() const { return thrust::raw_pointer_cast(cy.data()); }

  // Unwrapping buffers (optional) getters
  const int *imgYPtr() const {
    return imgY_.empty() ? nullptr : thrust::raw_pointer_cast(imgY_.data());
  }
  const int *imgZPtr() const {
    return imgZ_.empty() ? nullptr : thrust::raw_pointer_cast(imgZ_.data());
  }
  const T *yUnwrapPtr() const {
    return yU_.empty() ? nullptr : thrust::raw_pointer_cast(yU_.data());
  }
  const T *zUnwrapPtr() const {
    return zU_.empty() ? nullptr : thrust::raw_pointer_cast(zU_.data());
  }

private:
  // Thrust functor for one step of particle tracking
  MoveParticle<T> moveParticle;
  // Grid and physical variables
  grid::Grid<T> grid;
  T molecularDiffusion;
  T alphaL;
  T alphaT;
  unsigned int nParticles;
  // Velocity field stored on device (facefield)
  thrust::device_vector<T> datax;
  thrust::device_vector<T> datay;
  thrust::device_vector<T> dataz;
  // Velocity field stored on device (cornerfield or cellfield)
  thrust::device_vector<T> cdatax;
  thrust::device_vector<T> cdatay;
  thrust::device_vector<T> cdataz;
  // Particle positions stored on device
  thrust::device_vector<T> cx;
  thrust::device_vector<T> cy;
  thrust::device_vector<T> cz;
  // Vector of curand states stored on device
  thrust::device_vector<curandState_t> states;
  // If true, use trilinear interpolation
  bool useTrilinearCorrection;
  // Max number of particles simulated in each kernel
  const int maxParticles = 65536;

  // Unwrapping state/buffers
  thrust::device_vector<int> imgY_, imgZ_;
  thrust::device_vector<T> yU_, zU_;
  T Ly_ = 0, Lz_ = 0;
};

} // namespace par2

#include "PParticles.cu"

#endif // PAR2_PPARTICLES_CUH

```
