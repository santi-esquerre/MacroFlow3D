/**
 * @file closure_gate_main.cu
 * @brief SF-30 `closure_gate` executable: argument parsing around
 *        `closure_gate::run_closure_gate` (see `closure_gate.cuh`).
 *
 * Documented-experiment instrument (not a ctest entry). Exit codes:
 * 0 success; 2 usage error; 3 Darcy PCG not converged; 1 any exception.
 */

#include "apps/closure_gate/closure_gate.cuh"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

using namespace macroflow3d;
using namespace macroflow3d::closure_gate;

namespace {

const char* kUsage =
    "usage: closure_gate --field <gaussian|gaussian2d|control2d|lester2021|lester_brk|two_mode|generic3d>\n"
    "                    --n <N> --out <DIR>\n"
    "                    [--sigma2 <s> --ell <l> --seed <u64>]      (gaussian, gaussian2d; required for them)\n"
    "                    [--eps <e>]                                (analytic fields; required for them)\n"
    "                    [--seeds <1024>] [--seed-rng <20261005>] [--seeds-file <csv y0,z0>]\n"
    "                    [--tols <1e-6,1e-8,1e-10>] [--working-tol <1e-8>] [--periods <1>]\n"
    "                    [--pcg-rtol <1e-10>] [--pcg-max-iter <int>] [--mg-levels <auto|int>]\n"
    "                    [--threads <min(hardware_concurrency, 32)>] [--device <0>]\n";

double parse_double(const std::string& opt, const std::string& s) {
    char* end = nullptr;
    errno = 0;
    const double v = std::strtod(s.c_str(), &end);
    if (s.empty() || *end != '\0' || errno == ERANGE) throw ConfigError(opt + ": invalid number '" + s + "'");
    return v;
}

long long parse_int(const std::string& opt, const std::string& s) {
    char* end = nullptr;
    errno = 0;
    const long long v = std::strtoll(s.c_str(), &end, 10);
    if (s.empty() || *end != '\0' || errno == ERANGE) throw ConfigError(opt + ": invalid integer '" + s + "'");
    return v;
}

int parse_int32(const std::string& opt, const std::string& s) {
    const long long v = parse_int(opt, s);
    if (v < -2147483647LL || v > 2147483647LL) throw ConfigError(opt + ": out of range '" + s + "'");
    return static_cast<int>(v);
}

std::uint64_t parse_u64(const std::string& opt, const std::string& s) {
    if (s.empty() || s[0] == '-' || s[0] == '+') throw ConfigError(opt + ": invalid unsigned integer '" + s + "'");
    char* end = nullptr;
    errno = 0;
    const unsigned long long v = std::strtoull(s.c_str(), &end, 10);
    if (*end != '\0' || errno == ERANGE) throw ConfigError(opt + ": invalid unsigned integer '" + s + "'");
    return static_cast<std::uint64_t>(v);
}

std::vector<double> parse_list(const std::string& opt, const std::string& s) {
    std::vector<double> out;
    std::size_t start = 0;
    while (true) {
        const std::size_t comma = s.find(',', start);
        out.push_back(parse_double(opt, s.substr(start, comma == std::string::npos ? std::string::npos : comma - start)));
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    return out;
}

ClosureGateConfig parse_args(int argc, char** argv) {
    ClosureGateConfig c;
    bool has_field = false, has_n = false, has_out = false;
    std::vector<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string opt = argv[i];
        if (opt.rfind("--", 0) != 0) throw ConfigError("unexpected argument '" + opt + "'");
        for (const auto& s : seen)
            if (s == opt) throw ConfigError("option given twice: " + opt);
        seen.push_back(opt);
        if (i + 1 >= argc) throw ConfigError(opt + " requires a value");
        const std::string val = argv[++i];
        if (opt == "--field") {
            c.field = val;
            has_field = true;
        } else if (opt == "--n") {
            c.n = parse_int32(opt, val);
            has_n = true;
        } else if (opt == "--out") {
            if (val.empty()) throw ConfigError("--out must not be empty");
            c.out_dir = val;
            has_out = true;
        } else if (opt == "--sigma2") {
            c.sigma2 = parse_double(opt, val);
            c.has_sigma2 = true;
        } else if (opt == "--ell") {
            c.ell = parse_double(opt, val);
            c.has_ell = true;
        } else if (opt == "--seed") {
            c.seed = parse_u64(opt, val);
            c.has_seed = true;
        } else if (opt == "--eps") {
            c.eps = parse_double(opt, val);
            c.has_eps = true;
        } else if (opt == "--seeds") {
            c.n_seeds = parse_int32(opt, val);
            c.has_n_seeds = true;
        } else if (opt == "--seed-rng") {
            c.seed_rng = parse_u64(opt, val);
            c.has_seed_rng = true;
        } else if (opt == "--seeds-file") {
            if (val.empty()) throw ConfigError("--seeds-file must not be empty");
            c.seeds_file = val;
        } else if (opt == "--tols") {
            c.tols = parse_list(opt, val);
        } else if (opt == "--working-tol") {
            c.working_tol = parse_double(opt, val);
        } else if (opt == "--periods") {
            c.periods = parse_int32(opt, val);
        } else if (opt == "--pcg-rtol") {
            c.pcg_rtol = parse_double(opt, val);
        } else if (opt == "--pcg-max-iter") {
            c.pcg_max_iter = parse_int32(opt, val);
            if (c.pcg_max_iter < 1) throw ConfigError("--pcg-max-iter must be >= 1");
        } else if (opt == "--mg-levels") {
            if (val == "auto") {
                c.mg_levels = 0;
            } else {
                c.mg_levels = parse_int32(opt, val);
                if (c.mg_levels < 1) throw ConfigError("--mg-levels must be 'auto' or an integer >= 1");
            }
        } else if (opt == "--threads") {
            c.threads = parse_int32(opt, val);
        } else if (opt == "--device") {
            c.device = parse_int32(opt, val);
        } else {
            throw ConfigError("unknown option " + opt);
        }
    }
    if (!has_field || !has_n || !has_out) throw ConfigError("--field, --n and --out are required");
    normalize_and_validate(c);
    return c;
}

} // namespace

int main(int argc, char** argv) {
    std::printf("command:");
    for (int i = 0; i < argc; ++i) std::printf(" %s", argv[i]);
    std::printf("\n");
    std::fflush(stdout);

    ClosureGateConfig config;
    try {
        config = parse_args(argc, argv);
    } catch (const ConfigError& e) {
        std::fprintf(stderr, "closure_gate: %s\n%s", e.what(), kUsage);
        return 2;
    }
    try {
        CudaContext ctx(config.device);
        const ClosureGateResult result = run_closure_gate(ctx, config);
        print_digest(result, stdout);
        std::printf("  outputs: %s/{summary.json,%stiming.json}  wall %.2f s\n", config.out_dir.c_str(),
                    result.darcy_converged ? "streamlines.csv," : "", result.t_total);
        std::fflush(stdout);
        if (!result.darcy_converged) {
            std::fprintf(stderr, "closure_gate: Darcy PCG did not converge (summary.json written; exit 3)\n");
            return 3;
        }
        return 0;
    } catch (const ConfigError& e) {
        std::fprintf(stderr, "closure_gate: %s\n%s", e.what(), kUsage);
        return 2;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "closure_gate: error: %s\n", e.what());
        return 1;
    }
}
