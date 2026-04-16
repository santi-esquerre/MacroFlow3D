#pragma once

#include <cuda_runtime.h>

namespace macroflow3d {
namespace multigrid {
namespace bc_tags {

// Face tags (6 faces)
enum Face { XMIN = 0, XMAX = 1, YMIN = 2, YMAX = 3, ZMIN = 4, ZMAX = 5 };

// Edge tags (12 edges)
enum Edge {
    XMIN_YMIN = 0,
    XMIN_YMAX = 1,
    XMAX_YMIN = 2,
    XMAX_YMAX = 3,

    XMIN_ZMIN = 4,
    XMIN_ZMAX = 5,
    XMAX_ZMIN = 6,
    XMAX_ZMAX = 7,

    YMIN_ZMIN = 8,
    YMIN_ZMAX = 9,
    YMAX_ZMIN = 10,
    YMAX_ZMAX = 11
};

// Vertex tags (8 vertices)
enum Vertex {
    XMIN_YMIN_ZMIN = 0,
    XMIN_YMIN_ZMAX = 1,
    XMIN_YMAX_ZMIN = 2,
    XMIN_YMAX_ZMAX = 3,
    XMAX_YMIN_ZMIN = 4,
    XMAX_YMIN_ZMAX = 5,
    XMAX_YMAX_ZMIN = 6,
    XMAX_YMAX_ZMAX = 7
};

// Compile-time queries for faces
template <Face F> __host__ __device__ constexpr bool on_xmin() {
    return F == XMIN;
}
template <Face F> __host__ __device__ constexpr bool on_xmax() {
    return F == XMAX;
}
template <Face F> __host__ __device__ constexpr bool on_ymin() {
    return F == YMIN;
}
template <Face F> __host__ __device__ constexpr bool on_ymax() {
    return F == YMAX;
}
template <Face F> __host__ __device__ constexpr bool on_zmin() {
    return F == ZMIN;
}
template <Face F> __host__ __device__ constexpr bool on_zmax() {
    return F == ZMAX;
}

// Compile-time queries for edges
template <Edge E> __host__ __device__ constexpr bool on_xmin_edge() {
    return E == XMIN_YMIN || E == XMIN_YMAX || E == XMIN_ZMIN || E == XMIN_ZMAX;
}
template <Edge E> __host__ __device__ constexpr bool on_xmax_edge() {
    return E == XMAX_YMIN || E == XMAX_YMAX || E == XMAX_ZMIN || E == XMAX_ZMAX;
}
template <Edge E> __host__ __device__ constexpr bool on_ymin_edge() {
    return E == XMIN_YMIN || E == XMAX_YMIN || E == YMIN_ZMIN || E == YMIN_ZMAX;
}
template <Edge E> __host__ __device__ constexpr bool on_ymax_edge() {
    return E == XMIN_YMAX || E == XMAX_YMAX || E == YMAX_ZMIN || E == YMAX_ZMAX;
}
template <Edge E> __host__ __device__ constexpr bool on_zmin_edge() {
    return E == XMIN_ZMIN || E == XMAX_ZMIN || E == YMIN_ZMIN || E == YMAX_ZMIN;
}
template <Edge E> __host__ __device__ constexpr bool on_zmax_edge() {
    return E == XMIN_ZMAX || E == XMAX_ZMAX || E == YMIN_ZMAX || E == YMAX_ZMAX;
}

// Compile-time queries for vertices
template <Vertex V> __host__ __device__ constexpr bool on_xmin_vertex() {
    return V == XMIN_YMIN_ZMIN || V == XMIN_YMIN_ZMAX || V == XMIN_YMAX_ZMIN || V == XMIN_YMAX_ZMAX;
}
template <Vertex V> __host__ __device__ constexpr bool on_xmax_vertex() {
    return V == XMAX_YMIN_ZMIN || V == XMAX_YMIN_ZMAX || V == XMAX_YMAX_ZMIN || V == XMAX_YMAX_ZMAX;
}
template <Vertex V> __host__ __device__ constexpr bool on_ymin_vertex() {
    return V == XMIN_YMIN_ZMIN || V == XMIN_YMIN_ZMAX || V == XMAX_YMIN_ZMIN || V == XMAX_YMIN_ZMAX;
}
template <Vertex V> __host__ __device__ constexpr bool on_ymax_vertex() {
    return V == XMIN_YMAX_ZMIN || V == XMIN_YMAX_ZMAX || V == XMAX_YMAX_ZMIN || V == XMAX_YMAX_ZMAX;
}
template <Vertex V> __host__ __device__ constexpr bool on_zmin_vertex() {
    return V == XMIN_YMIN_ZMIN || V == XMIN_YMAX_ZMIN || V == XMAX_YMIN_ZMIN || V == XMAX_YMAX_ZMIN;
}
template <Vertex V> __host__ __device__ constexpr bool on_zmax_vertex() {
    return V == XMIN_YMIN_ZMAX || V == XMIN_YMAX_ZMAX || V == XMAX_YMIN_ZMAX || V == XMAX_YMAX_ZMAX;
}

} // namespace bc_tags
} // namespace multigrid
} // namespace macroflow3d
