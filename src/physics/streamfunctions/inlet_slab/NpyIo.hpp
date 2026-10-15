#pragma once

/**
 * @file NpyIo.hpp
 * @brief SF-33 N4: minimal header-only reader / writer of NumPy `.npy` files holding C-order
 *        little-endian float64 (`<f8`) arrays. Used to load the SF-29 prototype inputs exported by
 *        `docs/experiments/artifacts/2026-10-06-sf33-gpu-inlet-labels/scripts/export_proto.py`.
 *
 * Format (NumPy NEP 1, "npy-format"): magic "\x93NUMPY", major and minor version bytes, header
 * length (uint16 little-endian for version 1.0, uint32 little-endian for 2.0), then an ASCII
 * Python-literal dict `{'descr': '<f8', 'fortran_order': False, 'shape': (17, 16, 16), }` padded
 * with spaces and terminated by '\n' so that the data offset is a multiple of 64 (NumPy >= 1.14;
 * older writers used 16 - the reader does not require any alignment), then the raw data.
 *
 * Accepted: versions 1.0 and 2.0; descr exactly '<f8' (what NumPy writes for float64 on a
 * little-endian host; any other descr, including '>f8', '<f4', '<i8', is rejected);
 * fortran_order False; any shape including 0-d `()` and 1-d `(n,)`. Everything else is rejected
 * with an NpyError whose kind() and message identify the cause (distinct messages per cause). The
 * data size must match the shape exactly (no truncated and no trailing bytes). Requires a
 * little-endian host (checked at run time).
 *
 * The writer emits version 1.0 (version 2.0 only if the header does not fit in 65535 bytes, or
 * when forced), 64-byte aligned, in the exact header style of numpy.lib.format, so the files load
 * with `np.load` (checked locally with numpy 2.x; the format is unchanged since numpy 1.x).
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace macroflow3d {
namespace streamfunctions {
namespace inlet_slab {

/// Causes of an `.npy` read failure (one distinct message prefix each).
enum class NpyErrorKind {
    io,                  ///< "npy: cannot open"/"npy: cannot write"
    bad_magic,           ///< "npy: bad magic"
    unsupported_version, ///< "npy: unsupported format version"
    truncated_header,    ///< "npy: truncated header"
    malformed_header,    ///< "npy: malformed header"
    fortran_order,       ///< "npy: fortran_order arrays are not supported"
    unsupported_dtype,   ///< "npy: unsupported dtype"
    data_size_mismatch,  ///< "npy: data size mismatch"
    big_endian_host,     ///< "npy: big-endian host"
};

class NpyError : public std::runtime_error {
  public:
    NpyError(NpyErrorKind kind, const std::string& msg) : std::runtime_error(msg), kind_(kind) {}
    NpyErrorKind kind() const noexcept { return kind_; }

  private:
    NpyErrorKind kind_;
};

/// A C-order float64 array: shape (empty for 0-d) and data (product of the shape, 1 for 0-d).
struct NpyArray {
    std::vector<std::size_t> shape;
    std::vector<double> data;
    int major_version = 0; ///< version of the file it was read from (0 if built in memory)
    std::size_t data_offset = 0;

    std::size_t ndim() const { return shape.size(); }
    std::size_t count() const {
        std::size_t n = 1;
        for (std::size_t s : shape)
            n *= s;
        return n;
    }
    std::string shape_string() const {
        std::string s = "(";
        for (std::size_t i = 0; i < shape.size(); ++i) {
            s += std::to_string(shape[i]);
            if (shape.size() == 1 || i + 1 < shape.size())
                s += shape.size() == 1 ? "," : ", ";
        }
        return s + ")";
    }
};

namespace npy_detail {

inline bool host_is_little_endian() {
    const std::uint16_t one = 1;
    unsigned char b[2];
    std::memcpy(b, &one, 2);
    return b[0] == 1;
}

inline void require_little_endian_host() {
    if (!host_is_little_endian())
        throw NpyError(NpyErrorKind::big_endian_host,
                       "npy: big-endian host is not supported (data are '<f8')");
}

inline const char* kMagic() {
    return "\x93NUMPY";
}

inline std::string trim(const std::string& s) {
    std::size_t a = 0, b = s.size();
    while (a < b && (s[a] == ' ' || s[a] == '\t' || s[a] == '\n' || s[a] == '\r'))
        ++a;
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\n' || s[b - 1] == '\r'))
        --b;
    return s.substr(a, b - a);
}

/// Value text of key `key` in the header dict (raw, trimmed); throws malformed_header if absent.
inline std::string dict_value(const std::string& hdr, const std::string& key,
                              const std::string& what) {
    std::size_t p = std::string::npos;
    for (const char q : {'\'', '"'}) {
        const std::string k = std::string(1, q) + key + std::string(1, q);
        p = hdr.find(k);
        if (p != std::string::npos) {
            p += k.size();
            break;
        }
    }
    if (p == std::string::npos)
        throw NpyError(NpyErrorKind::malformed_header,
                       "npy: malformed header (missing key '" + key + "') in " + what);
    while (p < hdr.size() && hdr[p] == ' ')
        ++p;
    if (p >= hdr.size() || hdr[p] != ':')
        throw NpyError(NpyErrorKind::malformed_header,
                       "npy: malformed header (no ':' after key '" + key + "') in " + what);
    ++p;
    while (p < hdr.size() && hdr[p] == ' ')
        ++p;
    std::size_t e = p;
    if (e < hdr.size() && hdr[e] == '(') {
        e = hdr.find(')', e);
        if (e == std::string::npos)
            throw NpyError(NpyErrorKind::malformed_header,
                           "npy: malformed header (unterminated shape tuple) in " + what);
        ++e;
    } else if (e < hdr.size() && (hdr[e] == '\'' || hdr[e] == '"')) {
        const char q = hdr[e];
        e = hdr.find(q, e + 1);
        if (e == std::string::npos)
            throw NpyError(NpyErrorKind::malformed_header,
                           "npy: malformed header (unterminated string for key '" + key + "') in " +
                               what);
        ++e;
    } else {
        while (e < hdr.size() && hdr[e] != ',' && hdr[e] != '}')
            ++e;
    }
    return trim(hdr.substr(p, e - p));
}

inline std::vector<std::size_t> parse_shape(const std::string& v, const std::string& what) {
    if (v.size() < 2 || v.front() != '(' || v.back() != ')')
        throw NpyError(NpyErrorKind::malformed_header,
                       "npy: malformed header (shape is not a tuple: '" + v + "') in " + what);
    std::vector<std::size_t> shape;
    std::string inner = v.substr(1, v.size() - 2);
    std::size_t p = 0;
    while (p < inner.size()) {
        std::size_t c = inner.find(',', p);
        const std::string tok =
            trim(inner.substr(p, c == std::string::npos ? std::string::npos : c - p));
        if (!tok.empty()) {
            std::size_t n = 0;
            for (char ch : tok) {
                if (ch < '0' || ch > '9')
                    throw NpyError(NpyErrorKind::malformed_header,
                                   "npy: malformed header (bad shape entry '" + tok + "') in " +
                                       what);
                n = n * 10 + static_cast<std::size_t>(ch - '0');
            }
            shape.push_back(n);
        } else if (c != std::string::npos) {
            throw NpyError(NpyErrorKind::malformed_header,
                           "npy: malformed header (empty shape entry in '" + v + "') in " + what);
        }
        if (c == std::string::npos)
            break;
        p = c + 1;
    }
    return shape;
}

inline std::uint32_t read_le(const unsigned char* b, int nbytes) {
    std::uint32_t v = 0;
    for (int i = nbytes - 1; i >= 0; --i)
        v = (v << 8) | b[i];
    return v;
}

} // namespace npy_detail

/// Parse an in-memory `.npy` image. `what` names the source in error messages.
inline NpyArray parse_npy(const std::string& bytes, const std::string& what = "<memory>") {
    using namespace npy_detail;
    require_little_endian_host();
    if (bytes.size() < 8 || std::memcmp(bytes.data(), kMagic(), 6) != 0)
        throw NpyError(NpyErrorKind::bad_magic, "npy: bad magic (not a .npy file): " + what);
    const auto* u = reinterpret_cast<const unsigned char*>(bytes.data());
    const int major = u[6], minor = u[7];
    if (!((major == 1 || major == 2) && minor == 0))
        throw NpyError(NpyErrorKind::unsupported_version,
                       "npy: unsupported format version " + std::to_string(major) + "." +
                           std::to_string(minor) + " (only 1.0 and 2.0): " + what);
    const std::size_t len_bytes = major == 1 ? 2 : 4;
    if (bytes.size() < 8 + len_bytes)
        throw NpyError(NpyErrorKind::truncated_header,
                       "npy: truncated header (no header length): " + what);
    const std::size_t hlen = read_le(u + 8, static_cast<int>(len_bytes));
    const std::size_t off = 8 + len_bytes + hlen;
    if (bytes.size() < off)
        throw NpyError(NpyErrorKind::truncated_header, "npy: truncated header (header length " +
                                                           std::to_string(hlen) +
                                                           " exceeds the file): " + what);
    const std::string hdr = bytes.substr(8 + len_bytes, hlen);
    const std::string th = trim(hdr);
    if (th.empty() || th.front() != '{' || th.back() != '}')
        throw NpyError(NpyErrorKind::malformed_header,
                       "npy: malformed header (not a dict literal): " + what);

    const std::string descr = dict_value(hdr, "descr", what);
    const std::string fo = dict_value(hdr, "fortran_order", what);
    const std::string shp = dict_value(hdr, "shape", what);
    if (fo == "True")
        throw NpyError(NpyErrorKind::fortran_order,
                       "npy: fortran_order arrays are not supported (C order required): " + what);
    if (fo != "False")
        throw NpyError(NpyErrorKind::malformed_header,
                       "npy: malformed header (fortran_order is '" + fo + "'): " + what);
    if (!(descr == "'<f8'" || descr == "\"<f8\""))
        throw NpyError(NpyErrorKind::unsupported_dtype,
                       "npy: unsupported dtype " + descr +
                           " (only '<f8' little-endian float64): " + what);

    NpyArray a;
    a.shape = parse_shape(shp, what);
    a.major_version = major;
    a.data_offset = off;
    const std::size_t n = a.count();
    const std::size_t have = bytes.size() - off;
    if (have != n * sizeof(double))
        throw NpyError(NpyErrorKind::data_size_mismatch,
                       "npy: data size mismatch (shape " + a.shape_string() + " needs " +
                           std::to_string(n * sizeof(double)) + " bytes, file has " +
                           std::to_string(have) + "): " + what);
    a.data.resize(n);
    if (n > 0)
        std::memcpy(a.data.data(), bytes.data() + off, n * sizeof(double));
    return a;
}

/// Read a `.npy` file (whole file into memory, then parse_npy).
inline NpyArray read_npy(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw NpyError(NpyErrorKind::io, "npy: cannot open " + path);
    std::string bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    return parse_npy(bytes, path);
}

/// Read and require an exact shape (throws std::runtime_error naming the file otherwise).
inline NpyArray read_npy_shape(const std::string& path, const std::vector<std::size_t>& shape) {
    NpyArray a = read_npy(path);
    if (a.shape != shape) {
        NpyArray want;
        want.shape = shape;
        throw std::runtime_error("npy: shape " + a.shape_string() + " of " + path +
                                 " differs from the expected " + want.shape_string());
    }
    return a;
}

/// The `.npy` image of a C-order float64 array (version 1.0 unless the header needs 2.0 or
/// force_version == 2).
inline std::string serialize_npy(const std::vector<std::size_t>& shape, const double* data,
                                 std::size_t count, int force_version = 0) {
    npy_detail::require_little_endian_host();
    NpyArray tmp;
    tmp.shape = shape;
    if (tmp.count() != count)
        throw std::invalid_argument("npy: serialize_npy: data count " + std::to_string(count) +
                                    " does not match shape " + tmp.shape_string());
    if (force_version != 0 && force_version != 1 && force_version != 2)
        throw std::invalid_argument("npy: serialize_npy: version must be 1 or 2");
    std::string dict =
        "{'descr': '<f8', 'fortran_order': False, 'shape': " + tmp.shape_string() + ", }";
    int major = force_version == 2 ? 2 : 1;
    auto padded = [&](int mj) {
        const std::size_t pre = 8 + (mj == 1 ? 2 : 4);
        std::size_t total = pre + dict.size() + 1; // + '\n'
        const std::size_t pad = (64 - total % 64) % 64;
        return dict + std::string(pad, ' ') + "\n";
    };
    std::string hdr = padded(major);
    if (major == 1 && hdr.size() > 65535) {
        major = 2;
        hdr = padded(2);
    }
    std::string out = std::string(npy_detail::kMagic(), 6);
    out.push_back(static_cast<char>(major));
    out.push_back('\0');
    const std::uint32_t hl = static_cast<std::uint32_t>(hdr.size());
    const int lb = major == 1 ? 2 : 4;
    for (int i = 0; i < lb; ++i)
        out.push_back(static_cast<char>((hl >> (8 * i)) & 0xFFu));
    out += hdr;
    const std::size_t off = out.size();
    out.resize(off + count * sizeof(double));
    if (count > 0)
        std::memcpy(&out[off], data, count * sizeof(double));
    return out;
}

/// Write a C-order float64 array as `.npy`.
inline void write_npy(const std::string& path, const std::vector<std::size_t>& shape,
                      const double* data, std::size_t count, int force_version = 0) {
    const std::string img = serialize_npy(shape, data, count, force_version);
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f)
        throw NpyError(NpyErrorKind::io, "npy: cannot write " + path);
    f.write(img.data(), static_cast<std::streamsize>(img.size()));
    if (!f)
        throw NpyError(NpyErrorKind::io, "npy: cannot write " + path + " (write failed)");
}

inline void write_npy(const std::string& path, const NpyArray& a, int force_version = 0) {
    write_npy(path, a.shape, a.data.data(), a.data.size(), force_version);
}

} // namespace inlet_slab
} // namespace streamfunctions
} // namespace macroflow3d
