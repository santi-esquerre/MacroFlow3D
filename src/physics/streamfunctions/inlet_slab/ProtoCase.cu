/**
 * @file ProtoCase.cu
 * @brief SF-33 N4: SF-29 prototype-input loader (see ProtoCase.cuh).
 */

#include "ProtoCase.cuh"

#include "../../../external/nlohmann/json.hpp"
#include "../../../runtime/cuda_check.cuh"
#include "NpyIo.hpp"
#include "SlabResidual.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <sys/stat.h>
#include <utility>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

namespace {

using json = nlohmann::json;

std::string join(const std::string& dir, const std::string& name) {
    if (dir.empty() || dir.back() == '/')
        return dir + name;
    return dir + "/" + name;
}

bool is_dir(const std::string& p) {
    struct stat st;
    return ::stat(p.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

std::string read_text(const std::string& path) {
    std::ifstream f(path);
    if (!f)
        throw std::runtime_error("ProtoCase: cannot open " + path);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

template <class T> T json_get(const json& j, const char* key, const std::string& path) {
    if (!j.contains(key))
        throw std::runtime_error("ProtoCase: key '" + std::string(key) + "' missing in " + path);
    return j.at(key).get<T>();
}

std::vector<std::size_t> full_shape(const InletSlabGrid& g) {
    return {static_cast<std::size_t>(g.n + 1), static_cast<std::size_t>(g.n),
            static_cast<std::size_t>(g.n)};
}
std::vector<std::size_t> plane_shape(const InletSlabGrid& g) {
    return {static_cast<std::size_t>(g.n), static_cast<std::size_t>(g.n)};
}

void upload(DeviceBuffer<real>& b, const std::vector<real>& h) {
    if (b.size() != h.size())
        throw std::logic_error("ProtoCase: upload size mismatch");
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(b.data(), h.data(), h.size() * sizeof(real), cudaMemcpyHostToDevice));
    // SF-33 C2: legacy-stream copy must land before ctx-stream work
    MACROFLOW3D_CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<real> load(const std::string& dir, const std::string& name,
                       const std::vector<std::size_t>& shape) {
    return read_npy_shape(join(dir, name + ".npy"), shape).data;
}

real load_scalar(const std::string& dir, const std::string& name) {
    return read_npy_shape(join(dir, name + ".npy"), {}).data.at(0);
}

} // namespace

std::string stage_amplitude_name(real amp) {
    char b[64];
    std::snprintf(b, sizeof(b), "%g", amp);
    return b;
}

ProtoCaseMeta read_proto_case_meta(const std::string& dir) {
    const std::string path = join(dir, "case.json");
    ProtoCaseMeta m;
    m.dir = dir;
    m.json_text = read_text(path);
    json j;
    try {
        j = json::parse(m.json_text);
    } catch (const std::exception& e) {
        throw std::runtime_error("ProtoCase: cannot parse " + path + ": " + e.what());
    }
    m.field = json_get<std::string>(j, "field", path);
    m.eps = json_get<real>(j, "eps", path);
    m.N = json_get<int>(j, "N", path);
    m.nphi = json_get<int>(j, "nphi", path);
    m.Q0 = json_get<real>(j, "Q0", path);
    m.inlet_vmin = json_get<real>(j, "inlet_vmin", path);
    m.max_d1phi = json_get<real>(j, "max_d1phi", path);
    m.v_rms = json_get<real>(j, "v_rms", path);
    m.cache_hit = json_get<bool>(j, "cache_hit", path);
    if (j.contains("roundtrip_max") && j.at("roundtrip_max").is_number())
        m.roundtrip_max = j.at("roundtrip_max").get<real>();
    else
        m.roundtrip_max = std::nan("");
    m.stage_amplitudes = json_get<std::vector<std::string>>(j, "stage_amplitudes", path);
    return m;
}

std::vector<real> u0_from_psi0(const InletSlabGrid& grid, const std::vector<real>& psi0,
                               int which) {
    if (psi0.size() != grid.plane_size())
        throw std::invalid_argument("u0_from_psi0: psi0 size does not match the grid plane");
    if (which != 0 && which != 1)
        throw std::invalid_argument("u0_from_psi0: which must be 0 (psi1) or 1 (psi2)");
    std::vector<real> u(grid.plane_size());
    for (int m2 = 0; m2 < grid.n; ++m2)
        for (int m3 = 0; m3 < grid.n; ++m3) {
            const std::size_t i = grid.plane_index(m2, m3);
            u[i] = psi0[i] - grid.coord(which == 0 ? m2 : m3);
        }
    return u;
}

void load_stage_inputs_dir(CudaContext& ctx, const std::string& sd, const InletSlabGrid& grid,
                           const std::string& field, real amp, SlabStageInputs& out) {
    require_valid_grid(grid, "load_stage_inputs_dir");
    const auto fs = full_shape(grid);
    const auto ps = plane_shape(grid);
    out.allocate(grid);
    upload(out.lnk, load(sd, "lnk", fs));
    for (int k = 0; k < 3; ++k)
        upload(out.grad_lnk[k], load(sd, "grad_lnk_" + std::to_string(k + 1), fs));
    for (int i = 0; i < 2; ++i) {
        const std::vector<real> psi0 = load(sd, "psi0_" + std::to_string(i + 1), ps);
        upload(out.u0[i], u0_from_psi0(grid, psi0, i));
        upload(out.vperp_in[i], load(sd, "vperp_" + std::to_string(i + 2), ps));
    }
    out.v_rms = load_scalar(sd, "v_rms");
    out.field = field;
    out.eps = amp;
    out.N = grid.n;
    fill_q_from_lnk(ctx, grid, out);
    ctx.synchronize();
    out.check(grid);
}

ProtoCase load_proto_case(CudaContext& ctx, const std::string& dir, const InletSlabGrid& grid) {
    require_valid_grid(grid, "load_proto_case");
    ProtoCase pc;
    pc.meta = read_proto_case_meta(dir);
    if (pc.meta.N != grid.n)
        throw std::invalid_argument("load_proto_case: case N = " + std::to_string(pc.meta.N) +
                                    " differs from the grid N = " + std::to_string(grid.n) + " (" +
                                    dir + ")");
    load_stage_inputs_dir(ctx, dir, grid, pc.meta.field, pc.meta.eps, pc.inputs);
    const auto fs = full_shape(grid);
    pc.ref.allocate(grid, true, true);
    for (int k = 0; k < 3; ++k)
        upload(pc.ref.vD[k], load(dir, "vD_" + std::to_string(k + 1), fs));
    for (int i = 0; i < 2; ++i)
        upload(pc.ref.psi_or[i], load(dir, "psi_or_" + std::to_string(i + 1), fs));
    return pc;
}

// ------------------------------------------------------------------------------------------------

ProtoStageProvider::ProtoStageProvider(CudaContext& ctx, std::string case_dir,
                                       const InletSlabGrid& grid, std::string field)
    : ctx_(&ctx), case_dir_(std::move(case_dir)), grid_(grid), field_(std::move(field)) {
    require_valid_grid(grid_, "ProtoStageProvider");
}

std::string ProtoStageProvider::stage_dir(real amp) const {
    return join(join(case_dir_, "stage"), stage_amplitude_name(amp));
}

bool ProtoStageProvider::available(real amp) const {
    const std::string name = stage_amplitude_name(amp);
    return std::strtod(name.c_str(), nullptr) == amp && is_dir(stage_dir(amp));
}

const SlabStageInputs& ProtoStageProvider::operator()(real amp) {
    const std::string name = stage_amplitude_name(amp);
    auto it = cache_.find(name);
    if (it != cache_.end())
        return *it->second;
    if (std::strtod(name.c_str(), nullptr) != amp) {
        char b[64];
        std::snprintf(b, sizeof(b), "%.17g", amp);
        throw MissingStageInput(amp, std::string(kStatusMissingStageInput) + ": amplitude " + b +
                                         " is not exactly representable as a stage name ('" + name +
                                         "') under " + case_dir_);
    }
    const std::string sd = stage_dir(amp);
    if (!is_dir(sd))
        throw MissingStageInput(amp, std::string(kStatusMissingStageInput) +
                                         ": no exported stage input for amplitude " + name + " (" +
                                         sd + ")");
    auto in = std::make_unique<SlabStageInputs>();
    load_stage_inputs_dir(*ctx_, sd, grid_, field_, amp, *in);
    const SlabStageInputs& ref = *in;
    cache_.emplace(name, std::move(in));
    return ref;
}

// ------------------------------------------------------------------------------------------------

ProtoSolution load_solution(const std::string& dir) {
    const std::string path = join(dir, "solution.json");
    ProtoSolution s;
    s.dir = dir;
    s.json_text = read_text(path);
    json j;
    try {
        j = json::parse(s.json_text);
    } catch (const std::exception& e) {
        throw std::runtime_error("ProtoCase: cannot parse " + path + ": " + e.what());
    }
    s.N = json_get<int>(j, "N", path);
    s.field = json_get<std::string>(j, "field", path);
    s.cand = json_get<std::string>(j, "cand", path);
    s.status = json_get<std::string>(j, "status", path);
    s.path = json_get<std::string>(j, "path", path);
    s.eps = json_get<real>(j, "eps", path);
    s.its = json_get<int>(j, "its", path);
    s.r_F = json_get<real>(j, "r_F", path);
    s.r_out = json_get<real>(j, "r_out", path);
    const std::vector<std::size_t> fs = {static_cast<std::size_t>(s.N + 1),
                                         static_cast<std::size_t>(s.N),
                                         static_cast<std::size_t>(s.N)};
    s.u1 = load(dir, "u1", fs);
    s.u2 = load(dir, "u2", fs);
    return s;
}

real max_relative_difference(const std::vector<real>& a, const std::vector<real>& b) {
    if (a.size() != b.size())
        throw std::invalid_argument("max_relative_difference: size mismatch (" +
                                    std::to_string(a.size()) + " vs " + std::to_string(b.size()) +
                                    ")");
    real dmax = 0.0, bmax = 0.0;
    bool nonfinite = false;
    for (std::size_t i = 0; i < a.size(); ++i) {
        const real d = std::fabs(a[i] - b[i]);
        if (std::isnan(d) || std::isnan(b[i]))
            nonfinite = true;
        else if (d > dmax)
            dmax = d;
        bmax = std::fmax(bmax, std::fabs(b[i]));
    }
    if (nonfinite)
        return std::nan(""); // never hide a NaN in a comparison
    return bmax > 0.0 ? dmax / bmax : dmax;
}

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
