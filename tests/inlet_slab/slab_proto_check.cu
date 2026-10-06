/**
 * @file slab_proto_check.cu
 * @brief SF-33 N4: `inlet_slab_proto_check`, the prototype-input cross-validation of the N0
 *        residual and metrics on REAL SF-29 data. Built, NOT registered in ctest: it needs a
 *        Python export (docs/experiments/artifacts/2026-10-06-sf33-gpu-inlet-labels/scripts/
 *        export_proto.py).
 *
 * Modes (exit 0 iff every printed gate passes):
 *   --metrics-of-oracle <case_dir>
 *        loads the case, converts psi_or to periodic parts (labels_to_periodic_parts), runs
 *        evaluate_metrics against vD with psi_or as reference, prints the CASE line
 *        (cand=oracle_fd4) and every metric with 17 significant digits; if <case_dir>/
 *        ref_metrics.json exists (the prototype's metrics.fd_metrics(psi_or, order=4) at full
 *        precision), prints both values and the relative difference and gates <= 1e-10 for
 *        e_v, e_i1, e_i2, e_div, min_c, p0.1, p1, p5, p50, v_rms (e_psi* informational).
 *   --residual-of-solution <case_dir> <solution_dir>
 *        loads the case inputs at the target amplitude and the prototype's saved solution; the
 *        full planes are assembled exactly as the solver does (plane 0 = inputs.u0, planes 1..N =
 *        the solution); prints r_F, r_out (17 digits) next to the prototype's values and gates
 *        r_F <= 1e-12, r_out <= 1e-12. Also reports max |solution plane 0 - u0| and
 *        max_relative_difference(u, u).
 *   --metrics-of-solution <case_dir> <solution_dir>
 *        evaluate_metrics of the saved solution vs the prototype's own CASE metrics of that
 *        solution (solution.json metrics_json[cand]); gate <= 1e-10 relative on the same keys.
 *   --stages <case_dir>
 *        loads every stage/<amp> listed in case.json through ProtoStageProvider, checks that the
 *        stage at eps equals the main inputs bitwise and that an unlisted amplitude throws
 *        MissingStageInput.
 */

#include "src/core/DeviceBuffer.cuh"
#include "src/core/DeviceSpan.cuh"
#include "src/core/Scalar.hpp"
#include "src/external/nlohmann/json.hpp"
#include "src/physics/streamfunctions/inlet_slab/InletSlabGrid.cuh"
#include "src/physics/streamfunctions/inlet_slab/ProtoCase.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabMetrics.cuh"
#include "src/physics/streamfunctions/inlet_slab/SlabResidual.cuh"
#include "src/runtime/cuda_check.cuh"
#include "src/runtime/CudaContext.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

using namespace macroflow3d;
namespace sl = macroflow3d::streamfunctions::inlet_slab;
using json = nlohmann::json;

namespace {

constexpr double kMetricTol = 1e-10;
constexpr double kResidualTol = 1e-12;

DeviceSpan<const real> cspan(const DeviceBuffer<real>& b) {
    return DeviceSpan<const real>(b.data(), b.size());
}
DeviceSpan<real> mspan(DeviceBuffer<real>& b) {
    return DeviceSpan<real>(b.data(), b.size());
}

std::vector<real> download(const DeviceBuffer<real>& b) {
    std::vector<real> h(b.size());
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(h.data(), b.data(), h.size() * sizeof(real), cudaMemcpyDeviceToHost));
    return h;
}

void upload(DeviceBuffer<real>& b, const std::vector<real>& h) {
    b.resize(h.size());
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(b.data(), h.data(), h.size() * sizeof(real), cudaMemcpyHostToDevice));
}

bool read_json(const std::string& path, json& j) {
    std::ifstream f(path);
    if (!f)
        return false;
    j = json::parse(
        std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()));
    return true;
}

struct Gate {
    bool pass = true;
    void check(bool c, const std::string& name, const std::string& detail = "") {
        std::printf("[%s] %s%s%s\n", c ? "PASS" : "FAIL", name.c_str(), detail.empty() ? "" : "  ",
                    detail.c_str());
        pass = pass && c;
    }
};

std::vector<std::pair<std::string, double>> metric_list(const sl::SlabMetrics& m) {
    return {{"e_v", m.e_v},       {"e_i1", m.e_i1},     {"e_i2", m.e_i2},     {"e_div", m.e_div},
            {"min_c", m.min_c},   {"p0.1", m.p0_1},     {"p1", m.p1},         {"p5", m.p5},
            {"p50", m.p50},       {"v_rms", m.v_rms},   {"vD_min", m.vD_min}, {"e_psi", m.e_psi},
            {"e_psi1", m.e_psi1}, {"e_psi2", m.e_psi2}, {"a_psi1", m.a_psi1}, {"a_psi2", m.a_psi2}};
}

bool gated_key(const std::string& k) {
    static const char* keys[] = {"e_v",  "e_i1", "e_i2", "e_div", "min_c",
                                 "p0.1", "p1",   "p5",   "p50",   "v_rms"};
    for (const char* g : keys)
        if (k == g)
            return true;
    return false;
}

/// Prints the GPU metrics; compares with a prototype metrics dict if given.
void report_metrics(Gate& gate, const sl::SlabMetrics& m, const json* proto, const char* label) {
    std::printf("%-8s %-24s %-24s %s\n", "metric", "gpu (N0)", label, "rel diff");
    for (const auto& kv : metric_list(m)) {
        if (proto && proto->contains(kv.first) && (*proto)[kv.first].is_number()) {
            const double p = (*proto)[kv.first].get<double>();
            const double rel =
                p != 0.0 ? std::fabs(kv.second - p) / std::fabs(p) : std::fabs(kv.second - p);
            std::printf("%-8s %-24.17g %-24.17g %.3e%s\n", kv.first.c_str(), kv.second, p, rel,
                        gated_key(kv.first) ? "" : "  (informational)");
            if (gated_key(kv.first)) {
                char b[160];
                std::snprintf(b, sizeof(b), "gpu=%.17g proto=%.17g rel=%.3e", kv.second, p, rel);
                gate.check(rel <= kMetricTol, std::string("metric ") + kv.first + " <= 1e-10 rel",
                           b);
            }
        } else {
            std::printf("%-8s %-24.17g %-24s\n", kv.first.c_str(), kv.second, "-");
            if (proto && gated_key(kv.first))
                gate.check(false,
                           std::string("metric ") + kv.first + " present in the prototype dict");
        }
    }
    std::printf("vD_p     %.17g %.17g %.17g %.17g\n", m.vD_p[0], m.vD_p[1], m.vD_p[2], m.vD_p[3]);
    gate.check(m.nonfinite == 0, "no non-finite |c| / |vD|");
}

sl::InletSlabGrid grid_of(const std::string& case_dir) {
    return sl::InletSlabGrid::make(sl::read_proto_case_meta(case_dir).N);
}

int metrics_of_oracle(const std::string& case_dir) {
    Gate gate;
    CudaContext ctx;
    const auto g = grid_of(case_dir);
    sl::ProtoCase pc = sl::load_proto_case(ctx, case_dir, g);
    DeviceBuffer<real> U1(g.full_size()), U2(g.full_size());
    sl::labels_to_periodic_parts(ctx, g, cspan(pc.ref.psi_or[0]), cspan(pc.ref.psi_or[1]),
                                 mspan(U1), mspan(U2));
    sl::SlabMetricsWorkspace ws;
    ws.prepare(g);
    const sl::SlabMetrics m = sl::evaluate_metrics(ctx, g, cspan(U1), cspan(U2), pc.ref, ws);
    std::printf("CASE_DIR %s (field=%s eps=%g N=%d nphi=%d cache_hit=%d)\n", case_dir.c_str(),
                pc.meta.field.c_str(), pc.meta.eps, pc.meta.N, pc.meta.nphi,
                pc.meta.cache_hit ? 1 : 0);
    std::printf("%s\n",
                sl::format_case_line(pc.meta.field, pc.meta.eps, g.n, "oracle_fd4", m).c_str());
    json ref;
    const bool have = read_json(case_dir + "/ref_metrics.json", ref) && ref.contains("oracle_fd4");
    if (have) {
        const json& p = ref["oracle_fd4"];
        std::printf("PROTO %s\n", ref.value("case_line", std::string("?")).c_str());
        report_metrics(gate, m, &p, "prototype (oracle_fd4)");
        // Informational: e_psi of the oracle against itself is a roundoff quantity (N0 rebuilds the
        // labels as affine + (psi_or - affine), which is not bitwise psi_or), so the %.3e text of
        // e_psi may differ from the prototype's exact 0 while every gated metric agrees.
        const bool same = sl::format_case_line(pc.meta.field, pc.meta.eps, g.n, "oracle_fd4", m) ==
                          ref.value("case_line", std::string());
        std::printf("[INFO] CASE line %s the prototype's%s\n",
                    same ? "identical to" : "differs from",
                    same ? "" : " (see the e_psi* rows: roundoff of the label reconstruction)");
    } else {
        report_metrics(gate, m, nullptr, "-");
        std::printf("(no ref_metrics.json: values printed only)\n");
    }
    gate.check(std::fabs(m.v_rms - pc.inputs.v_rms) <= 1e-14 * pc.inputs.v_rms,
               "metrics v_rms == exported v_rms.npy");
    std::printf("%s\n", gate.pass ? "PROTO_CHECK PASS" : "PROTO_CHECK FAIL");
    return gate.pass ? 0 : 1;
}

struct LoadedSolution {
    DeviceBuffer<real> u_vec, U1, U2;
    sl::ProtoSolution sol;
};

void assemble_solution(CudaContext& ctx, const sl::InletSlabGrid& g, const sl::ProtoCase& pc,
                       LoadedSolution& ls, Gate& gate) {
    if (ls.sol.N != g.n)
        throw std::invalid_argument("solution N differs from the case N");
    // unknown vector = planes 1..N of the saved solution
    const std::size_t ps = g.plane_size(), nf = g.field_size();
    std::vector<real> hv(2 * nf);
    std::copy(ls.sol.u1.begin() + static_cast<std::ptrdiff_t>(ps), ls.sol.u1.end(), hv.begin());
    std::copy(ls.sol.u2.begin() + static_cast<std::ptrdiff_t>(ps), ls.sol.u2.end(),
              hv.begin() + static_cast<std::ptrdiff_t>(nf));
    upload(ls.u_vec, hv);
    ls.U1.resize(g.full_size());
    ls.U2.resize(g.full_size());
    sl::assemble_full_planes(ctx, g, cspan(ls.u_vec), pc.inputs, mspan(ls.U1), mspan(ls.U2));
    ctx.synchronize();
    const auto h1 = download(ls.U1), h2 = download(ls.U2);
    double d0 = 0.0;
    for (std::size_t i = 0; i < ps; ++i)
        d0 = std::fmax(d0,
                       std::fmax(std::fabs(h1[i] - ls.sol.u1[i]), std::fabs(h2[i] - ls.sol.u2[i])));
    char b[128];
    std::snprintf(b, sizeof(b), "max |solution plane 0 - u0(psi0 - x)| = %.3e", d0);
    gate.check(d0 == 0.0, "saved inlet plane == loaded u0 (bitwise)", b);
    const double self1 = sl::max_relative_difference(h1, ls.sol.u1);
    const double self2 = sl::max_relative_difference(h2, ls.sol.u2);
    std::snprintf(b, sizeof(b), "max_relative_difference(assembled, saved) = %.3e, %.3e", self1,
                  self2);
    gate.check(self1 == 0.0 && self2 == 0.0, "assembled full planes == saved full planes", b);
    gate.check(sl::max_relative_difference(ls.sol.u1, ls.sol.u1) == 0.0 &&
                   sl::max_relative_difference(ls.sol.u2, ls.sol.u2) == 0.0,
               "max_relative_difference(u, u) = 0");
}

int residual_of_solution(const std::string& case_dir, const std::string& sol_dir) {
    Gate gate;
    CudaContext ctx;
    const auto g = grid_of(case_dir);
    sl::ProtoCase pc = sl::load_proto_case(ctx, case_dir, g);
    LoadedSolution ls;
    ls.sol = sl::load_solution(sol_dir);
    std::printf("SOLUTION %s cand=%s status=%s its=%d path=%s (prototype r_F=%.17g r_out=%.17g)\n",
                sol_dir.c_str(), ls.sol.cand.c_str(), ls.sol.status.c_str(), ls.sol.its,
                ls.sol.path.c_str(), ls.sol.r_F, ls.sol.r_out);
    gate.check(ls.sol.field == pc.meta.field && ls.sol.eps == pc.meta.eps,
               "solution matches the case");
    assemble_solution(ctx, g, pc, ls, gate);
    sl::SlabResidualWorkspace ws;
    ws.prepare(g);
    DeviceBuffer<real> E(g.unknown_size());
    sl::SlabResidualNorms nrm;
    sl::evaluate_residual(ctx, g, pc.inputs, cspan(ls.U1), cspan(ls.U2), mspan(E), ws, &nrm);
    std::printf("RESIDUAL gpu(N0): r_F=%.17g r_out=%.17g | prototype: r_F=%.17g r_out=%.17g\n",
                nrm.r_F, nrm.r_out, ls.sol.r_F, ls.sol.r_out);
    char b[128];
    std::snprintf(b, sizeof(b), "r_F=%.3e", nrm.r_F);
    gate.check(nrm.r_F <= kResidualTol, "r_F <= 1e-12 at the prototype's solution", b);
    std::snprintf(b, sizeof(b), "r_out=%.3e", nrm.r_out);
    gate.check(nrm.r_out <= kResidualTol, "r_out <= 1e-12 at the prototype's solution", b);
    std::printf("%s\n", gate.pass ? "PROTO_CHECK PASS" : "PROTO_CHECK FAIL");
    return gate.pass ? 0 : 1;
}

int metrics_of_solution(const std::string& case_dir, const std::string& sol_dir) {
    Gate gate;
    CudaContext ctx;
    const auto g = grid_of(case_dir);
    sl::ProtoCase pc = sl::load_proto_case(ctx, case_dir, g);
    LoadedSolution ls;
    ls.sol = sl::load_solution(sol_dir);
    assemble_solution(ctx, g, pc, ls, gate);
    sl::SlabMetricsWorkspace ws;
    ws.prepare(g);
    const sl::SlabMetrics m = sl::evaluate_metrics(ctx, g, cspan(ls.U1), cspan(ls.U2), pc.ref, ws);
    sl::CaseLineExtras ex;
    ex.r_F = ls.sol.r_F;
    ex.its = ls.sol.its;
    std::printf("%s\n",
                sl::format_case_line(pc.meta.field, pc.meta.eps, g.n, ls.sol.cand, m, ex).c_str());
    const json sj = json::parse(ls.sol.json_text);
    const json& mj = sj.at("metrics_json");
    if (mj.contains(ls.sol.cand)) {
        report_metrics(gate, m, &mj.at(ls.sol.cand), ("prototype (" + ls.sol.cand + ")").c_str());
    } else {
        gate.check(false, "solution.json metrics_json has the candidate's metrics");
    }
    std::printf("%s\n", gate.pass ? "PROTO_CHECK PASS" : "PROTO_CHECK FAIL");
    return gate.pass ? 0 : 1;
}

int stages(const std::string& case_dir) {
    Gate gate;
    CudaContext ctx;
    const auto g = grid_of(case_dir);
    sl::ProtoCase pc = sl::load_proto_case(ctx, case_dir, g);
    sl::ProtoStageProvider prov(ctx, case_dir, g, pc.meta.field);
    for (const std::string& name : pc.meta.stage_amplitudes) {
        const double amp = std::strtod(name.c_str(), nullptr);
        const sl::SlabStageInputs& s = prov(amp);
        std::printf("STAGE amp=%s v_rms=%.17g\n", name.c_str(), s.v_rms);
        if (amp == pc.meta.eps) {
            bool same = s.v_rms == pc.inputs.v_rms;
            same = same && download(s.q) == download(pc.inputs.q) &&
                   download(s.lnk) == download(pc.inputs.lnk);
            for (int k = 0; k < 3; ++k)
                same = same && download(s.grad_lnk[k]) == download(pc.inputs.grad_lnk[k]);
            for (int i = 0; i < 2; ++i)
                same = same && download(s.u0[i]) == download(pc.inputs.u0[i]) &&
                       download(s.vperp_in[i]) == download(pc.inputs.vperp_in[i]);
            gate.check(same, "stage at eps == main inputs (bitwise)");
        }
    }
    gate.check(prov.cached_count() == pc.meta.stage_amplitudes.size(), "every listed stage loaded");
    try {
        prov(pc.meta.eps * 0.3);
        gate.check(false, "unlisted amplitude throws MissingStageInput");
    } catch (const sl::MissingStageInput& e) {
        gate.check(std::string(e.status()) == "missing_stage_input",
                   "unlisted amplitude throws MissingStageInput", e.what());
    }
    std::printf("%s\n", gate.pass ? "PROTO_CHECK PASS" : "PROTO_CHECK FAIL");
    return gate.pass ? 0 : 1;
}

int usage() {
    std::fprintf(stderr,
                 "usage: inlet_slab_proto_check --metrics-of-oracle <case_dir>\n"
                 "       inlet_slab_proto_check --residual-of-solution <case_dir> <solution_dir>\n"
                 "       inlet_slab_proto_check --metrics-of-solution <case_dir> <solution_dir>\n"
                 "       inlet_slab_proto_check --stages <case_dir>\n");
    return 2;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3)
        return usage();
    const std::string mode = argv[1];
    try {
        if (mode == "--metrics-of-oracle" && argc == 3)
            return metrics_of_oracle(argv[2]);
        if (mode == "--residual-of-solution" && argc == 4)
            return residual_of_solution(argv[2], argv[3]);
        if (mode == "--metrics-of-solution" && argc == 4)
            return metrics_of_solution(argv[2], argv[3]);
        if (mode == "--stages" && argc == 3)
            return stages(argv[2]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "inlet_slab_proto_check: %s\n", e.what());
        std::printf("PROTO_CHECK FAIL (exception)\n");
        return 1;
    }
    return usage();
}
