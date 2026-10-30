// SF-42 N4: seed-lattice probe. Calls the PRODUCTION find_seed_on_face (which builds the production seed
// lattice with make_seed_lattice) on cases read from a file, one per line:  <ny> <w2a,w2b,w3a,w3b> <M>
// ('#' lines and blank lines are skipped). The window strings are parsed with strtod exactly as the runner's
// --window. One output line per case:  <ny> <window> <M> OK   |   <ny> <window> <M> ON_FACE m2=.. m3=.. axis=..
// Verification instrument for docs/experiments/artifacts/2026-10-30-sf42-confirmatory-tooling/scripts/seedlattice
// (not a production tool, no GPU work). Usage: seedlattice_probe <cases file>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "src/physics/particles/lagrangian_labels/EnsembleRunner.cuh"
#include "src/physics/particles/lagrangian_labels/LagrangianTrackerTypes.hpp"

namespace ll = macroflow3d::lagrangian_labels;

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <cases file>\n", argv[0]);
        return 2;
    }
    std::ifstream f(argv[1]);
    if (!f) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#')
            continue;
        std::istringstream is(line);
        int ny = 0, M = 0;
        std::string win;
        if (!(is >> ny >> win >> M)) {
            std::fprintf(stderr, "bad case line: %s\n", line.c_str());
            return 2;
        }
        double w[4];
        std::size_t a = 0;
        for (int k = 0; k < 4; ++k) {
            const std::size_t b = win.find(',', a);
            const std::string s = win.substr(a, b == std::string::npos ? b : b - a);
            char* end = nullptr;
            w[k] = std::strtod(s.c_str(), &end);
            if (s.empty() || end != s.c_str() + s.size() || (k < 3) != (b != std::string::npos)) {
                std::fprintf(stderr, "bad window: %s\n", win.c_str());
                return 2;
            }
            a = b + 1;
        }
        ll::InjectionSpec inj;
        inj.X0 = 0.5;
        inj.M = M;
        inj.w2a = w[0];
        inj.w2b = w[1];
        inj.w3a = w[2];
        inj.w3b = w[3];
        std::string what;
        if (!ll::find_seed_on_face(inj, ny, &what)) {
            std::printf("%d %s %d OK\n", ny, win.c_str(), M);
        } else {
            std::size_t p = 0, m2 = 0, m3 = 0;
            int axis = 0;
            if (std::sscanf(what.c_str(), "seed p=%zu (m2=%zu, m3=%zu) x%d", &p, &m2, &m3, &axis) != 4) {
                std::fprintf(stderr, "cannot parse runner message: %s\n", what.c_str());
                return 2;
            }
            std::printf("%d %s %d ON_FACE m2=%zu m3=%zu axis=%d\n", ny, win.c_str(), M, m2, m3, axis);
        }
    }
    return 0;
}
