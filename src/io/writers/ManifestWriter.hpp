#pragma once

/**
 * @file ManifestWriter.hpp
 * @brief Writes a rich JSON manifest with run metadata + config.
 * @ingroup io_writers
 *
 * Etapa 8: uses nlohmann/json for proper JSON. Includes GPU info,
 * build info, git hash, full config snapshot, and effective_config path.
 *
 * Pure host-side. No hot-loop usage (called once at pipeline start).
 */

#include "../output_layout.hpp"
#include "../config/Config.hpp"
#include "../config/ConfigDefaults.hpp"
#include "BuildInfo.hpp"
#include "../../external/nlohmann/json.hpp"
#include <cstdio>
#include <ctime>
#include <fstream>
#include <string>

namespace macroflow3d {
namespace io {

struct ManifestWriter {

    /**
     * @brief Write manifest.json with full metadata.
     *
     * @param layout  Output layout (provides path).
     * @param cfg     Resolved config.
     * @param gpu     GPU info (from GPUInfo::query()).
     */
    static bool write(const OutputLayout& layout,
                      const AppConfig& cfg,
                      const GPUInfo& gpu)
    {
        using json = nlohmann::json;

        json j;

        // ── Versioning ──────────────────────────────────────────────
        j["format_version"]        = kOutputFormatVersion;
        j["config_schema_version"] = kConfigSchemaVersion;
        j["project"]               = "MacroFlow3D";

        // ── Timestamp ───────────────────────────────────────────────
        char ts[64];
        std::time_t now = std::time(nullptr);
        std::strftime(ts, sizeof(ts), "%Y-%m-%dT%H:%M:%S", std::localtime(&now));
        j["timestamp"] = ts;

        // ── Build info ──────────────────────────────────────────────
        j["build"] = {
            {"git_hash",           BuildInfo::git_hash},
            {"build_type",         BuildInfo::build_type},
            {"cuda_architectures", BuildInfo::cuda_architectures},
            {"nvcc_version",       BuildInfo::nvcc_version}
        };

        // ── GPU info ────────────────────────────────────────────────
        j["gpu"] = {
            {"device_name",         gpu.device_name},
            {"compute_capability",  gpu.compute_capability()},
            {"vram_mib",            gpu.vram_mib()},
            {"cuda_runtime_version", gpu.cuda_runtime_version},
            {"cuda_driver_version",  gpu.cuda_driver_version}
        };

        // ── Grid ────────────────────────────────────────────────────
        j["grid"] = {
            {"nx", cfg.grid.nx}, {"ny", cfg.grid.ny}, {"nz", cfg.grid.nz},
            {"dx", (double)cfg.grid.dx}
        };

        // ── Stochastic ──────────────────────────────────────────────
        j["stochastic"] = {
            {"sigma2",          (double)cfg.stochastic.sigma2},
            {"corr_length",     (double)cfg.stochastic.corr_length},
            {"n_modes",         cfg.stochastic.n_modes},
            {"covariance_type", cfg.stochastic.covariance_type},
            {"seed",            cfg.stochastic.seed},
            {"K_mean",          (double)cfg.stochastic.K_mean}
        };

        // ── Flow ────────────────────────────────────────────────────
        j["flow"] = {
            {"solver",         cfg.flow.solver},
            {"mg_levels",      cfg.flow.mg_levels},
            {"mg_max_cycles",  cfg.flow.mg_max_cycles},
            {"cg_max_iter",    cfg.flow.cg_max_iter},
            {"rtol",           (double)cfg.flow.rtol},
            {"verify_velocity", cfg.flow.verify_velocity}
        };

        // ── Transport ───────────────────────────────────────────────
        j["transport"] = {
            {"n_particles", cfg.transport.n_particles},
            {"dt",          (double)cfg.transport.dt},
            {"n_steps",     cfg.transport.n_steps},
            {"porosity",    (double)cfg.transport.porosity},
            {"diffusion",   (double)cfg.transport.diffusion},
            {"alpha_l",     (double)cfg.transport.alpha_l},
            {"alpha_t",     (double)cfg.transport.alpha_t},
            {"seed",        cfg.transport.seed}
        };

        // ── Analysis ────────────────────────────────────────────────
        const auto& mac = cfg.analysis.macrodispersion;
        j["analysis"]["macrodispersion"] = {
            {"enabled",       mac.enabled},
            {"NR",            mac.NR},
            {"lambda",        (double)mac.lambda},
            {"vmean_norm",    (double)mac.vmean_norm},
            {"sample_every",  mac.sample_every},
            {"var_estimator", mac.var_estimator}
        };

        const auto& snap = cfg.analysis.snapshots;
        j["analysis"]["snapshots"] = {
            {"enabled", snap.enabled},
            {"every",   snap.every},
            {"stride",  snap.stride}
        };

        // ── Output ──────────────────────────────────────────────────
        j["output_dir"]          = cfg.output.output_dir;
        j["effective_config"]    = layout.effective_config();

        // ── Write ───────────────────────────────────────────────────
        std::string path = layout.manifest();
        std::ofstream ofs(path);
        if (!ofs) return false;
        ofs << j.dump(2) << "\n";
        return true;
    }
};

} // namespace io
} // namespace macroflow3d
