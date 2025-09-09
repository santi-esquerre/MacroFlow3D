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
