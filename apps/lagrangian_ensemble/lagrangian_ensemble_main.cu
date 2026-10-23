/**
 * @file lagrangian_ensemble_main.cu
 * @brief SF-38 N3 `lagrangian_ensemble` executable: Lagrangian ensemble runner on the long box
 *        (medium (a), contract C-RUNNER). Logic in src/physics/particles/lagrangian_labels/
 *        EnsembleRunner.{cuh,cu}; options and exit codes: `lagrangian_ensemble --help`.
 *
 * Exit codes: 0 ok; 1 exception; 2 usage; 3 seed-face assertion (RR12); 4 stage failure (darcy
 * not converged); 5 track failure; 6 comparator failure; 7 DONE realization under different
 * options; 99 test-only abort.
 */

#include "src/physics/particles/lagrangian_labels/EnsembleRunner.cuh"

#include <cstdio>
#include <cstring>
#include <exception>

using namespace macroflow3d;
namespace ll = macroflow3d::lagrangian_labels;

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--help") == 0) {
            std::fputs(ll::ensemble_usage().c_str(), stdout);
            return ll::kExitOk;
        }
    ll::EnsembleConfig cfg;
    try {
        cfg = ll::parse_ensemble_args(argc, argv);
    } catch (const ll::UsageError& e) {
        std::fprintf(stderr, "lagrangian_ensemble: usage error: %s\n(see --help)\n", e.what());
        return ll::kExitUsage;
    }
    try {
        CudaContext ctx(0);
        return ll::run_ensemble(ctx, cfg);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "lagrangian_ensemble: ERROR: %s\n", e.what());
        return ll::kExitException;
    }
}
