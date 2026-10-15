/**
 * @file slab_npy_tests.cu
 * @brief SF-33 N4: fast contract tests of the `.npy` reader / writer (NpyIo.hpp) and of the
 *        prototype-case loader (ProtoCase.cuh) on a synthetic case directory written in C++.
 *        No Python is needed at test time.
 *
 * Cases:
 *   1  writer -> reader round trip, bitwise, for shapes (17,16,16), (16,16), (), (5,), (0,); both
 *      header versions (1.0 default, 2.0 forced); header length / 64-byte alignment / '\n'.
 *   2  numpy-written fixtures (tests/inlet_slab/fixtures/, written by numpy 2.5.0 with
 *        a = (np.arange(272.).reshape(17, 4, 4) + 1) / 7     -> np_f8_v1_17x4x4.npy  (np.save)
 *        (np.arange(15.).reshape(3, 5) - 7) / 3              -> np_f8_v2_3x5.npy     (format 2.0)
 *        np.float64(-1/3)                                    -> np_f8_0d.npy
 *        np.arange(5.) * 0.1                                 -> np_f8_1d_5.npy
 *        np.asfortranarray(np.arange(12.).reshape(3, 4))     -> np_f8_fortran_3x4.npy (rejected)
 *        np.arange(4, dtype=np.float32)                      -> np_f4_4.npy           (rejected)
 *        np.arange(4, dtype='>f8')                           -> np_be_f8_4.npy        (rejected))
 *      are read back bitwise, and the writer reproduces the numpy files byte for byte.
 *   3  rejections with distinct kinds / messages: bad magic, version 3.0, truncated header,
 *      malformed header (missing key, bad shape), fortran_order True, '<f4' / '>f8' descr, data
 * size mismatch (truncated and trailing), missing file. 4  host helpers: u0_from_psi0 (psi0 - m/N),
 * max_relative_difference, stage_amplitude_name. 5  load_proto_case + ProtoStageProvider on a
 * synthetic N = 8 case directory: device data equal to the files (q = 1/exp(lnk), u0 = psi0 - x),
 * cached stage loads, MissingStageInput with status "missing_stage_input" for an absent amplitude.
 * Option --emit DIR writes the round-trip arrays of case 1 to DIR (for an np.load check by hand).
 */

#include "src/core/Scalar.hpp"
#include "src/physics/streamfunctions/inlet_slab/InletSlabGrid.cuh"
#include "src/physics/streamfunctions/inlet_slab/NpyIo.hpp"
#include "src/physics/streamfunctions/inlet_slab/ProtoCase.cuh"
#include "src/runtime/cuda_check.cuh"
#include "src/runtime/CudaContext.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <unistd.h>
#include <vector>

#ifndef INLET_SLAB_FIXTURE_DIR
#error "INLET_SLAB_FIXTURE_DIR must be defined by the build (tests/inlet_slab/fixtures)"
#endif

using namespace macroflow3d;
namespace sl = macroflow3d::streamfunctions::inlet_slab;

namespace {

struct TestReport {
    bool overall_pass = true;
    int checks = 0;
    void check(bool cond, const std::string& name, const std::string& detail = "") {
        ++checks;
        std::printf("[%s] %s%s%s\n", cond ? "PASS" : "FAIL", name.c_str(),
                    detail.empty() ? "" : "  ", detail.c_str());
        overall_pass = overall_pass && cond;
    }
};

std::string fixture(const std::string& name) {
    return std::string(INLET_SLAB_FIXTURE_DIR) + "/" + name;
}

std::string read_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

bool bitwise_equal(const std::vector<double>& a, const std::vector<double>& b) {
    return a.size() == b.size() &&
           (a.empty() || std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

/// Deterministic non-trivial values, including signed zero, subnormal, huge, tiny.
std::vector<double> pattern(std::size_t n, double seed, bool specials = true) {
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i)
        v[i] = std::sin(seed + 0.37 * static_cast<double>(i)) * std::exp(0.01 * (i % 97));
    if (specials && n > 3) {
        v[0] = -0.0;
        v[1] = 4.9406564584124654e-324;
        v[2] = 1.7976931348623157e308;
        v[3] = -2.2250738585072014e-308;
    }
    return v;
}

/// Expect parse/read to throw NpyError of `kind` whose message starts with `prefix`.
template <class F>
void expect_error(TestReport& rep, const std::string& name, sl::NpyErrorKind kind,
                  const std::string& prefix, F&& f) {
    try {
        f();
        rep.check(false, name, "no exception");
    } catch (const sl::NpyError& e) {
        const std::string msg = e.what();
        rep.check(e.kind() == kind && msg.compare(0, prefix.size(), prefix) == 0, name,
                  "message: " + msg);
    } catch (const std::exception& e) {
        rep.check(false, name, std::string("wrong exception type: ") + e.what());
    }
}

std::string make_header_image(const std::string& dict, int major, std::size_t data_bytes) {
    std::string out = std::string("\x93NUMPY", 6);
    out.push_back(static_cast<char>(major));
    out.push_back('\0');
    std::string hdr = dict;
    const std::size_t pre = 8 + (major == 1 ? 2 : 4);
    std::size_t pad = (64 - (pre + hdr.size() + 1) % 64) % 64;
    hdr += std::string(pad, ' ') + "\n";
    const std::size_t hl = hdr.size();
    for (int i = 0; i < (major == 1 ? 2 : 4); ++i)
        out.push_back(static_cast<char>((hl >> (8 * i)) & 0xFF));
    out += hdr;
    out += std::string(data_bytes, '\0');
    return out;
}

// ------------------------------------------------------------------------------------------------
void case_roundtrip(TestReport& rep, const char* emit_dir) {
    std::printf("\n== case 1: writer -> reader round trip ==\n");
    const std::vector<std::vector<std::size_t>> shapes = {{17, 16, 16}, {16, 16}, {}, {5}, {0}};
    const char* names[] = {"a17x16x16", "a16x16", "a0d", "a5", "a0"};
    for (std::size_t s = 0; s < shapes.size(); ++s) {
        sl::NpyArray proto;
        proto.shape = shapes[s];
        const auto data = pattern(proto.count(), 0.1 + static_cast<double>(s));
        for (int v : {1, 2}) {
            const std::string img =
                sl::serialize_npy(shapes[s], data.data(), data.size(), v == 2 ? 2 : 0);
            const auto u = reinterpret_cast<const unsigned char*>(img.data());
            const std::size_t hl = v == 1 ? (u[8] | (u[9] << 8))
                                          : (u[8] | (u[9] << 8) | (u[10] << 16) |
                                             (static_cast<std::size_t>(u[11]) << 24));
            const std::size_t off = (v == 1 ? 10 : 12) + hl;
            const bool aligned = off % 64 == 0 && img[off - 1] == '\n' && u[6] == v && u[7] == 0;
            const sl::NpyArray r = sl::parse_npy(img, names[s]);
            rep.check(r.shape == shapes[s] && bitwise_equal(r.data, data) && aligned &&
                          r.major_version == v && r.data_offset == off,
                      std::string("round trip ") + names[s] + " " + r.shape_string() + " v" +
                          std::to_string(v) + ".0",
                      "header bytes " + std::to_string(off) + ", count " +
                          std::to_string(r.count()));
        }
        if (emit_dir) {
            const std::string p = std::string(emit_dir) + "/" + names[s] + ".npy";
            sl::write_npy(p, shapes[s], data.data(), data.size());
            const sl::NpyArray r = sl::read_npy(p);
            rep.check(bitwise_equal(r.data, data), std::string("emitted ") + p);
        }
    }
    // file round trip through write_npy / read_npy
    char tmpl[] = "/tmp/inlet_slab_npy_XXXXXX";
    const int fd = mkstemp(tmpl);
    if (fd >= 0)
        close(fd);
    const auto data = pattern(17 * 16 * 16, 3.0);
    sl::write_npy(tmpl, {17, 16, 16}, data.data(), data.size());
    const sl::NpyArray r = sl::read_npy(tmpl);
    rep.check(r.shape == std::vector<std::size_t>({17, 16, 16}) && bitwise_equal(r.data, data),
              "file round trip (17, 16, 16) via write_npy / read_npy");
    std::remove(tmpl);
    // header text exactly as numpy.lib.format writes it
    const std::string img = sl::serialize_npy({17, 16, 16}, data.data(), data.size());
    const std::string dict = "{'descr': '<f8', 'fortran_order': False, 'shape': (17, 16, 16), }";
    rep.check(img.compare(10, dict.size(), dict) == 0 && img[10 + dict.size()] == ' ',
              "header dict text identical to numpy's");
}

void case_fixtures(TestReport& rep) {
    std::printf("\n== case 2: numpy-written fixtures ==\n");
    {
        std::vector<double> want(272);
        for (int i = 0; i < 272; ++i)
            want[i] = (static_cast<double>(i) + 1.0) / 7.0;
        const sl::NpyArray a = sl::read_npy(fixture("np_f8_v1_17x4x4.npy"));
        rep.check(a.shape == std::vector<std::size_t>({17, 4, 4}) && a.major_version == 1 &&
                      bitwise_equal(a.data, want),
                  "numpy v1.0 (17, 4, 4) read bitwise");
        rep.check(sl::serialize_npy({17, 4, 4}, want.data(), want.size()) ==
                      read_bytes(fixture("np_f8_v1_17x4x4.npy")),
                  "writer reproduces the numpy v1.0 file byte for byte");
    }
    {
        std::vector<double> want(15);
        for (int i = 0; i < 15; ++i)
            want[i] = (static_cast<double>(i) - 7.0) / 3.0;
        const sl::NpyArray a = sl::read_npy(fixture("np_f8_v2_3x5.npy"));
        rep.check(a.shape == std::vector<std::size_t>({3, 5}) && a.major_version == 2 &&
                      bitwise_equal(a.data, want),
                  "numpy v2.0 (3, 5) read bitwise");
        rep.check(sl::serialize_npy({3, 5}, want.data(), want.size(), 2) ==
                      read_bytes(fixture("np_f8_v2_3x5.npy")),
                  "writer (forced 2.0) reproduces the numpy v2.0 file byte for byte");
    }
    {
        const std::vector<double> want = {-1.0 / 3.0};
        const sl::NpyArray a = sl::read_npy(fixture("np_f8_0d.npy"));
        rep.check(a.shape.empty() && bitwise_equal(a.data, want), "numpy 0-d () read bitwise");
        rep.check(sl::serialize_npy({}, want.data(), 1) == read_bytes(fixture("np_f8_0d.npy")),
                  "writer reproduces the numpy 0-d file byte for byte");
    }
    {
        std::vector<double> want(5);
        for (int i = 0; i < 5; ++i)
            want[i] = static_cast<double>(i) * 0.1;
        const sl::NpyArray a = sl::read_npy(fixture("np_f8_1d_5.npy"));
        rep.check(a.shape == std::vector<std::size_t>({5}) && bitwise_equal(a.data, want),
                  "numpy 1-d (5,) read bitwise");
        rep.check(sl::serialize_npy({5}, want.data(), 5) == read_bytes(fixture("np_f8_1d_5.npy")),
                  "writer reproduces the numpy 1-d file byte for byte");
    }
    expect_error(rep, "numpy Fortran-order file rejected", sl::NpyErrorKind::fortran_order,
                 "npy: fortran_order arrays are not supported",
                 [] { sl::read_npy(fixture("np_f8_fortran_3x4.npy")); });
    expect_error(rep, "numpy float32 file rejected", sl::NpyErrorKind::unsupported_dtype,
                 "npy: unsupported dtype", [] { sl::read_npy(fixture("np_f4_4.npy")); });
    expect_error(rep, "numpy big-endian float64 file rejected", sl::NpyErrorKind::unsupported_dtype,
                 "npy: unsupported dtype", [] { sl::read_npy(fixture("np_be_f8_4.npy")); });
}

void case_rejections(TestReport& rep) {
    std::printf("\n== case 3: rejections ==\n");
    const std::string ok = "{'descr': '<f8', 'fortran_order': False, 'shape': (2, 3), }";
    expect_error(rep, "bad magic", sl::NpyErrorKind::bad_magic, "npy: bad magic", [] {
        sl::parse_npy(std::string("\x93NUMPZ\x01\x00\x10\x00", 10) + std::string(64, ' '), "t");
    });
    expect_error(rep, "empty file", sl::NpyErrorKind::bad_magic, "npy: bad magic",
                 [] { sl::parse_npy("", "t"); });
    expect_error(rep, "version 3.0", sl::NpyErrorKind::unsupported_version,
                 "npy: unsupported format version 3.0",
                 [&] { sl::parse_npy(make_header_image(ok, 3, 48), "t"); });
    expect_error(rep, "version 1.1", sl::NpyErrorKind::unsupported_version,
                 "npy: unsupported format version 1.1", [&] {
                     std::string b = make_header_image(ok, 1, 48);
                     b[7] = 1;
                     sl::parse_npy(b, "t");
                 });
    expect_error(rep, "truncated header (no length)", sl::NpyErrorKind::truncated_header,
                 "npy: truncated header",
                 [] { sl::parse_npy(std::string("\x93NUMPY\x02\x00\x10", 9), "t"); });
    expect_error(rep, "truncated header (length beyond file)", sl::NpyErrorKind::truncated_header,
                 "npy: truncated header", [&] {
                     std::string b = make_header_image(ok, 1, 0);
                     sl::parse_npy(b.substr(0, 40), "t");
                 });
    expect_error(rep, "malformed header (missing shape)", sl::NpyErrorKind::malformed_header,
                 "npy: malformed header (missing key 'shape')", [] {
                     sl::parse_npy(
                         make_header_image("{'descr': '<f8', 'fortran_order': False, }", 1, 8),
                         "t");
                 });
    expect_error(rep, "malformed header (bad shape entry)", sl::NpyErrorKind::malformed_header,
                 "npy: malformed header (bad shape entry", [] {
                     sl::parse_npy(
                         make_header_image(
                             "{'descr': '<f8', 'fortran_order': False, 'shape': (2, x), }", 1, 16),
                         "t");
                 });
    expect_error(rep, "malformed header (not a dict)", sl::NpyErrorKind::malformed_header,
                 "npy: malformed header (not a dict literal)",
                 [] { sl::parse_npy(make_header_image("descr <f8", 1, 8), "t"); });
    expect_error(rep, "fortran_order True", sl::NpyErrorKind::fortran_order,
                 "npy: fortran_order arrays are not supported", [] {
                     sl::parse_npy(
                         make_header_image(
                             "{'descr': '<f8', 'fortran_order': True, 'shape': (2, 3), }", 1, 48),
                         "t");
                 });
    expect_error(rep, "descr '<f4'", sl::NpyErrorKind::unsupported_dtype,
                 "npy: unsupported dtype '<f4'", [] {
                     sl::parse_npy(
                         make_header_image(
                             "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }", 1, 24),
                         "t");
                 });
    expect_error(rep, "descr '>f8'", sl::NpyErrorKind::unsupported_dtype,
                 "npy: unsupported dtype '>f8'", [] {
                     sl::parse_npy(
                         make_header_image(
                             "{'descr': '>f8', 'fortran_order': False, 'shape': (2, 3), }", 1, 48),
                         "t");
                 });
    expect_error(rep, "data truncated", sl::NpyErrorKind::data_size_mismatch,
                 "npy: data size mismatch",
                 [&] { sl::parse_npy(make_header_image(ok, 1, 40), "t"); });
    expect_error(rep, "trailing data", sl::NpyErrorKind::data_size_mismatch,
                 "npy: data size mismatch",
                 [&] { sl::parse_npy(make_header_image(ok, 1, 56), "t"); });
    expect_error(rep, "missing file", sl::NpyErrorKind::io, "npy: cannot open",
                 [] { sl::read_npy("/nonexistent/inlet_slab/none.npy"); });
    // keys in another order and double quotes are accepted (Python literal)
    const std::string alt = "{\"shape\": (2, 3), \"fortran_order\": False, \"descr\": \"<f8\"}";
    const sl::NpyArray a = sl::parse_npy(make_header_image(alt, 1, 48), "t");
    rep.check(a.shape == std::vector<std::size_t>({2, 3}) && a.count() == 6,
              "reordered keys / double quotes accepted");
    try {
        const auto v = pattern(4, 1.0);
        sl::write_npy("/tmp/inlet_slab_shape_check.npy", {2, 2}, v.data(), 4);
        sl::read_npy_shape("/tmp/inlet_slab_shape_check.npy", {4});
        rep.check(false, "read_npy_shape rejects a wrong shape", "no exception");
    } catch (const sl::NpyError& e) {
        rep.check(false, "read_npy_shape rejects a wrong shape",
                  std::string("NpyError: ") + e.what());
    } catch (const std::runtime_error& e) {
        rep.check(std::string(e.what()).find("differs from the expected (4,)") != std::string::npos,
                  "read_npy_shape rejects a wrong shape", e.what());
    }
    std::remove("/tmp/inlet_slab_shape_check.npy");
}

void case_host_helpers(TestReport& rep) {
    std::printf("\n== case 4: host helpers ==\n");
    const auto g = sl::InletSlabGrid::make(8);
    std::vector<double> psi0(g.plane_size());
    for (int m2 = 0; m2 < 8; ++m2)
        for (int m3 = 0; m3 < 8; ++m3)
            psi0[g.plane_index(m2, m3)] = 0.3 * m2 - 0.7 * m3 + 0.01;
    const auto u1 = sl::u0_from_psi0(g, psi0, 0);
    const auto u2 = sl::u0_from_psi0(g, psi0, 1);
    bool ok = true;
    for (int m2 = 0; m2 < 8; ++m2)
        for (int m3 = 0; m3 < 8; ++m3) {
            const std::size_t i = g.plane_index(m2, m3);
            ok = ok && u1[i] == psi0[i] - static_cast<double>(m2) / 8.0 &&
                 u2[i] == psi0[i] - static_cast<double>(m3) / 8.0;
        }
    rep.check(ok, "u0_from_psi0: u0_1 = psi0_1 - m2/N, u0_2 = psi0_2 - m3/N (bitwise)");
    const std::vector<double> a = {1.0, -2.0, 3.0}, b = {1.0, -2.5, 4.0};
    rep.check(sl::max_relative_difference(a, a) == 0.0, "max_relative_difference(a, a) = 0");
    rep.check(sl::max_relative_difference(a, b) == 1.0 / 4.0,
              "max_relative_difference = max|a-b|/max|b|");
    rep.check(sl::max_relative_difference({1e-3}, {0.0}) == 1e-3,
              "max |b| = 0 -> absolute difference");
    rep.check(std::isnan(sl::max_relative_difference({std::nan("")}, {1.0})), "NaN propagates");
    bool threw = false;
    try {
        sl::max_relative_difference(a, {1.0});
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    rep.check(threw, "size mismatch throws");
    rep.check(sl::stage_amplitude_name(0.25) == "0.25" && sl::stage_amplitude_name(1.0) == "1" &&
                  sl::stage_amplitude_name(0.015625) == "0.015625" &&
                  sl::stage_amplitude_name(0.8125) == "0.8125",
              "stage_amplitude_name = Python '%g'");
}

// ------------------------------------------------------------------------------------------------
void write_arr(const std::string& dir, const std::string& name,
               const std::vector<std::size_t>& shape, const std::vector<double>& v) {
    sl::write_npy(dir + "/" + name + ".npy", shape, v.data(), v.size());
}

std::vector<double> download(const DeviceBuffer<real>& b) {
    std::vector<double> h(b.size());
    MACROFLOW3D_CUDA_CHECK(
        cudaMemcpy(h.data(), b.data(), h.size() * sizeof(double), cudaMemcpyDeviceToHost));
    return h;
}

std::vector<double> pattern_s(std::size_t n, double seed) {
    return pattern(n, seed, false);
}

void write_stage(const std::string& dir, const sl::InletSlabGrid& g, double amp,
                 std::vector<double>& lnk, std::vector<double>& psi01, std::vector<double>& vp2) {
    const std::vector<std::size_t> fs = {9, 8, 8}, ps = {8, 8};
    lnk = pattern_s(g.full_size(), amp);
    for (auto& x : lnk)
        x *= amp;
    write_arr(dir, "lnk", fs, lnk);
    for (int k = 1; k <= 3; ++k)
        write_arr(dir, "grad_lnk_" + std::to_string(k), fs,
                  pattern_s(g.full_size(), 10.0 * k + amp));
    psi01 = pattern_s(g.plane_size(), 20.0 + amp);
    write_arr(dir, "psi0_1", ps, psi01);
    write_arr(dir, "psi0_2", ps, pattern_s(g.plane_size(), 21.0 + amp));
    vp2 = pattern_s(g.plane_size(), 22.0 + amp);
    write_arr(dir, "vperp_2", ps, vp2);
    write_arr(dir, "vperp_3", ps, pattern_s(g.plane_size(), 23.0 + amp));
    const double vr = 1.0 + amp;
    sl::write_npy(dir + "/v_rms.npy", {}, &vr, 1);
}

void case_loader(TestReport& rep) {
    std::printf("\n== case 5: load_proto_case / ProtoStageProvider (synthetic N = 8 case) ==\n");
    char tmpl[] = "/tmp/inlet_slab_case_XXXXXX";
    if (!mkdtemp(tmpl)) {
        rep.check(false, "mkdtemp");
        return;
    }
    const std::string dir = tmpl;
    const auto g = sl::InletSlabGrid::make(8);
    std::vector<double> lnk, psi01, vp2, lnk_s, psi01_s, vp2_s;
    write_stage(dir, g, 0.25, lnk, psi01, vp2);
    const std::vector<std::size_t> fs = {9, 8, 8};
    const auto vD1 = pattern_s(g.full_size(), 30.0);
    write_arr(dir, "vD_1", fs, vD1);
    write_arr(dir, "vD_2", fs, pattern_s(g.full_size(), 31.0));
    write_arr(dir, "vD_3", fs, pattern_s(g.full_size(), 32.0));
    const auto por2 = pattern_s(g.full_size(), 33.0);
    write_arr(dir, "psi_or_1", fs, pattern_s(g.full_size(), 34.0));
    write_arr(dir, "psi_or_2", fs, por2);
    {
        std::ofstream f(dir + "/case.json");
        f << "{\"field\": \"synthetic\", \"eps\": 0.25, \"N\": 8, \"nphi\": 16, \"Q0\": 1.0, "
             "\"inlet_vmin\": 0.5, \"max_d1phi\": -0.1, \"v_rms\": 1.25, \"cache_hit\": true, "
             "\"roundtrip_max\": 1e-12, \"stage_amplitudes\": [\"0.125\", \"0.25\"]}";
    }
    const std::string sd = dir + "/stage/0.125";
    const std::string mk = "mkdir -p " + sd;
    rep.check(std::system(mk.c_str()) == 0, "create stage dir");
    write_stage(sd, g, 0.125, lnk_s, psi01_s, vp2_s);

    CudaContext ctx;
    sl::ProtoCase pc = sl::load_proto_case(ctx, dir, g);
    rep.check(pc.meta.field == "synthetic" && pc.meta.eps == 0.25 && pc.meta.N == 8 &&
                  pc.meta.stage_amplitudes.size() == 2 && pc.inputs.v_rms == 1.25,
              "case.json meta and v_rms");
    const auto hl = download(pc.inputs.lnk);
    const auto hq = download(pc.inputs.q);
    double qerr = 0.0;
    for (std::size_t i = 0; i < hq.size(); ++i)
        qerr =
            std::fmax(qerr, std::fabs(hq[i] - 1.0 / std::exp(lnk[i])) / (1.0 / std::exp(lnk[i])));
    rep.check(bitwise_equal(hl, lnk), "lnk uploaded bitwise");
    char b[96];
    std::snprintf(b, sizeof(b), "max rel |q - 1/exp(lnk)| = %.3e", qerr);
    rep.check(qerr <= 1e-15, "q = 1/exp(lnk) (fill_q_from_lnk)", b);
    const auto hu0 = download(pc.inputs.u0[0]);
    rep.check(bitwise_equal(hu0, sl::u0_from_psi0(g, psi01, 0)),
              "u0_1 = psi0_1 - m2/N on the device");
    rep.check(bitwise_equal(download(pc.inputs.vperp_in[0]), vp2), "vperp_in[0] = vperp_2 bitwise");
    rep.check(bitwise_equal(download(pc.ref.vD[0]), vD1) &&
                  bitwise_equal(download(pc.ref.psi_or[1]), por2) && pc.ref.has_vD &&
                  pc.ref.has_psi_or,
              "vD, psi_or uploaded bitwise");

    sl::ProtoStageProvider prov(ctx, dir, g, pc.meta.field);
    auto cb = prov.callback();
    const sl::SlabStageInputs& s1 = cb(0.125);
    const sl::SlabStageInputs& s1b = cb(0.125);
    rep.check(&s1 == &s1b && prov.cached_count() == 1 && s1.eps == 0.125 && s1.v_rms == 1.125 &&
                  bitwise_equal(download(s1.lnk), lnk_s) &&
                  bitwise_equal(download(s1.u0[0]), sl::u0_from_psi0(g, psi01_s, 0)),
              "provider: stage 0.125 loaded once, cached, data bitwise");
    rep.check(prov.available(0.125) && !prov.available(0.0625) && !prov.available(0.1),
              "provider: available() reflects stage/<amp:%g>");
    for (double amp : {0.0625, 0.1}) {
        try {
            cb(amp);
            rep.check(false, "MissingStageInput for an absent amplitude", "no exception");
        } catch (const sl::MissingStageInput& e) {
            rep.check(std::string(e.status()) == "missing_stage_input" && e.amplitude() == amp &&
                          std::string(e.what()).compare(0, 19, "missing_stage_input") == 0,
                      "MissingStageInput for amplitude " + sl::stage_amplitude_name(amp), e.what());
        }
    }
    bool threw = false;
    try {
        sl::load_proto_case(ctx, dir, sl::InletSlabGrid::make(10));
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    rep.check(threw, "load_proto_case rejects a grid of another N");
    const std::string rm = "rm -rf " + dir;
    (void)std::system(rm.c_str());
}

} // namespace

int main(int argc, char** argv) {
    const char* emit = nullptr;
    for (int i = 1; i < argc; ++i)
        if (std::string(argv[i]) == "--emit" && i + 1 < argc)
            emit = argv[++i];
    TestReport rep;
    try {
        case_roundtrip(rep, emit);
        case_fixtures(rep);
        case_rejections(rep);
        case_host_helpers(rep);
        case_loader(rep);
    } catch (const std::exception& e) {
        rep.check(false, "unexpected exception", e.what());
    }
    std::printf("\n%s: %d checks, %s\n", rep.overall_pass ? "PASS" : "FAIL", rep.checks,
                rep.overall_pass ? "all passed" : "FAILURES above");
    return rep.overall_pass ? 0 : 1;
}
