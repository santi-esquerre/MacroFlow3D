/**
 * @file flow_workspace_bytes_tests.cu
 * @brief SF-40 L7: byte-count contract of FlowWorkspace::allocate(grid, L).
 *
 * After allocate the workspace holds rhs (8 n) and the MG hierarchy
 * (x, b, r, coefficient per level = 32 n_l); the CG/PCG workspaces are
 * allocated on demand in solve_head and must still be empty. The removed
 * dead members residual/aux1/aux2 (24 n bytes) must not come back.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/Grid3D.hpp"
#include "src/core/Scalar.hpp"
#include "src/physics/common/workspaces.cuh"

#include <cstddef>
#include <cstdio>

using namespace macroflow3d;

static int g_fail = 0;

static void check(const char* name, bool ok) {
    std::printf("CHECK %s %s\n", name, ok ? "PASS" : "FAIL");
    if (!ok)
        ++g_fail;
}

static size_t cap(const DeviceBuffer<real>& b) { return b.capacity() * sizeof(real); }

static void run(int nx, int ny, int nz, int L) {
    Grid3D grid;
    grid.nx = nx;
    grid.ny = ny;
    grid.nz = nz;
    grid.dx = grid.dy = grid.dz = 1.0;

    physics::FlowWorkspace ws;
    ws.allocate(grid, L);

    size_t n = grid.num_cells();
    size_t got = cap(ws.rhs);
    size_t expect = 8 * n;
    size_t sum_nl = 0;
    for (const auto& lv : ws.mg_hierarchy.levels) {
        got += cap(lv.x) + cap(lv.b) + cap(lv.r) + cap(lv.coefficient);
        sum_nl += lv.grid.num_cells();
    }
    expect += 32 * sum_nl;

    const auto& cg = ws.cg_workspace;
    const auto& pcg = ws.pcg_workspace;
    size_t od =
        cap(cg.r) + cap(cg.p) + cap(cg.Ap) + cap(pcg.r) + cap(pcg.z) + cap(pcg.p) + cap(pcg.Ap);

    std::printf("grid %dx%dx%d L=%d: levels=%d n=%zu sum_nl=%zu bytes=%zu expected=%zu "
                "on-demand=%zu (dead members would add %zu)\n",
                nx, ny, nz, L, ws.mg_hierarchy.num_levels(), n, sum_nl, got, expect, od, 24 * n);

    check("mg levels == requested", ws.mg_hierarchy.num_levels() == L);
    check("allocate bytes == 8 n + sum_l 32 n_l", got == expect);
    check("cg/pcg workspaces empty after allocate", od == 0);
}

int main() {
    run(32, 16, 16, 3);
    run(64, 32, 32, 3);
    std::printf(g_fail ? "FAILED\n" : "ALL PASSED\n");
    return g_fail ? 1 : 0;
}
