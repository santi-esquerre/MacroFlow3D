/**
 * @file inlet_slab_main.cu
 * @brief SF-33 N5 `inlet_slab` executable: entry point of the inlet-slab experiment driver
 *        (logic in `inlet_slab_driver.cuh`). Documented-experiment instrument, not a ctest entry.
 *
 * Exit codes: 0 converged (crosscheck: ok); 1 exception; 2 usage; 10 linesearch-fail;
 * 11 stagnation; 12 maxit; 13 linear_failure; 14 nan_inf; 15 continuation_floor;
 * 16 missing_stage_input; 17 inlet_backflow; 18 darcy_failed; 19 oracle_roundtrip_fail.
 */

#include "apps/inlet_slab/inlet_slab_driver.cuh"

int main(int argc, char** argv) {
    return macroflow3d::inlet_slab_app::run(argc, argv);
}
