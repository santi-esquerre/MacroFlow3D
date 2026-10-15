#pragma once

/**
 * @file json_writer.hpp
 * @brief SF-32 N2a: minimal ordered JSON value (writer + a small reader) for
 *        the `spurious_spreading` instrument. Host-only C++17, no third-party
 *        header.
 *
 * The writer (class JVal, helper jobj) is a copy of the one in
 * `apps/closure_gate/ev_ladder_main.cu` (SF-30 N3). It is copied, not
 * included, because that file is a `.cu` translation unit, and the vendored
 * nlohmann header must not be used here: it triggers an nvcc 11.4 internal
 * compiler error inside a CUDA translation unit (SF-30 finding C3).
 *
 * Writer conventions: objects keep insertion order; assigning an existing key
 * replaces its value in place; doubles are written with "%.17g" (lossless
 * round trip) and keep a ".0" when integral; NaN/inf become null; integers
 * are written exactly; 2-space indented pretty print.
 *
 * Additions for SF-32 (not in the SF-30 copy):
 *  - read accessors (kind(), is_number(), as_double(), as_int(), as_string(),
 *    size(), at(i), find(key), keys()) so the instrument can read the labels
 *    metadata it wrote itself;
 *  - parse_json(text): a small strict recursive-descent reader of standard
 *    JSON (objects, arrays, strings with the escapes the writer emits plus
 *    \uXXXX in the ASCII range, numbers, true/false/null). A number without
 *    '.', 'e' or 'E' becomes a signed (or, if too large, unsigned) integer,
 *    any other number a double parsed with strtod, so a metadata object read
 *    back and re-written reproduces the bytes the writer produced.
 */

#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace spurious_spreading {

class JVal {
  public:
    enum class Kind {
        null_value,
        boolean,
        signed_int,
        unsigned_int,
        number,
        string,
        array,
        object
    };

    JVal() = default;
    JVal(std::nullptr_t) {}
    JVal(bool b) : kind_(Kind::boolean), b_(b) {}
    JVal(const char* s) : kind_(Kind::string), s_(s) {}
    JVal(const std::string& s) : kind_(Kind::string), s_(s) {}
    template <class T,
              typename std::enable_if<std::is_arithmetic<T>::value && !std::is_same<T, bool>::value,
                                      int>::type = 0>
    JVal(T v) {
        if (std::is_floating_point<T>::value) {
            kind_ = Kind::number;
            d_ = static_cast<double>(v);
        } else if (std::is_signed<T>::value) {
            kind_ = Kind::signed_int;
            i_ = static_cast<long long>(v);
        } else {
            kind_ = Kind::unsigned_int;
            u_ = static_cast<unsigned long long>(v);
        }
    }

    static JVal array() {
        JVal j;
        j.kind_ = Kind::array;
        return j;
    }
    static JVal array(std::initializer_list<JVal> items) {
        JVal j = array();
        for (const JVal& v : items)
            j.items_.push_back(v);
        return j;
    }
    static JVal object() {
        JVal j;
        j.kind_ = Kind::object;
        return j;
    }

    // Object member access: inserts a null member at the end if absent.
    JVal& operator[](const std::string& key) {
        if (kind_ == Kind::null_value)
            kind_ = Kind::object;
        if (kind_ != Kind::object)
            throw std::logic_error("JVal: operator[] on a non-object");
        for (std::size_t i = 0; i < keys_.size(); ++i) {
            if (keys_[i] == key)
                return items_[i];
        }
        keys_.push_back(key);
        items_.push_back(JVal());
        return items_.back();
    }

    void push_back(const JVal& v) {
        if (kind_ == Kind::null_value)
            kind_ = Kind::array;
        if (kind_ != Kind::array)
            throw std::logic_error("JVal: push_back on a non-array");
        items_.push_back(v);
    }

    std::string dump(int indent) const {
        std::string out;
        write(out, indent, 0);
        return out;
    }

    // ---- read accessors (SF-32 addition) ----
    Kind kind() const { return kind_; }
    bool is_null() const { return kind_ == Kind::null_value; }
    bool is_number() const {
        return kind_ == Kind::number || kind_ == Kind::signed_int || kind_ == Kind::unsigned_int;
    }
    bool is_object() const { return kind_ == Kind::object; }
    bool is_array() const { return kind_ == Kind::array; }
    bool is_string() const { return kind_ == Kind::string; }
    double as_double() const {
        switch (kind_) {
        case Kind::number:
            return d_;
        case Kind::signed_int:
            return static_cast<double>(i_);
        case Kind::unsigned_int:
            return static_cast<double>(u_);
        default:
            throw std::logic_error("JVal: as_double on a non-number");
        }
    }
    long long as_int() const {
        if (kind_ == Kind::signed_int)
            return i_;
        if (kind_ == Kind::unsigned_int && u_ <= 9223372036854775807ULL)
            return static_cast<long long>(u_);
        throw std::logic_error("JVal: as_int on a non-integer");
    }
    const std::string& as_string() const {
        if (kind_ != Kind::string)
            throw std::logic_error("JVal: as_string on a non-string");
        return s_;
    }
    std::size_t size() const { return items_.size(); }
    const JVal& at(std::size_t i) const {
        if (kind_ != Kind::array && kind_ != Kind::object)
            throw std::logic_error("JVal: at on a scalar");
        if (i >= items_.size())
            throw std::out_of_range("JVal: index out of range");
        return items_[i];
    }
    /// Pointer to the member `key` of an object, or nullptr if absent / not an object.
    const JVal* find(const std::string& key) const {
        if (kind_ != Kind::object)
            return nullptr;
        for (std::size_t i = 0; i < keys_.size(); ++i) {
            if (keys_[i] == key)
                return &items_[i];
        }
        return nullptr;
    }
    const std::vector<std::string>& keys() const { return keys_; }

    // ---- construction helpers for the reader ----
    static JVal make_double(double d) {
        JVal j;
        j.kind_ = Kind::number;
        j.d_ = d;
        return j;
    }
    static JVal make_signed(long long v) {
        JVal j;
        j.kind_ = Kind::signed_int;
        j.i_ = v;
        return j;
    }
    static JVal make_unsigned(unsigned long long v) {
        JVal j;
        j.kind_ = Kind::unsigned_int;
        j.u_ = v;
        return j;
    }

  private:
    static void write_string(std::string& out, const std::string& s) {
        out += '"';
        for (char ch : s) {
            const unsigned char c = static_cast<unsigned char>(ch);
            switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\b':
                out += "\\b";
                break;
            case '\f':
                out += "\\f";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out += ch;
                }
            }
        }
        out += '"';
    }

    static void write_double(std::string& out, double d) {
        if (!std::isfinite(d)) {
            out += "null";
            return;
        }
        char buf[40];
        std::snprintf(buf, sizeof(buf), "%.17g", d);
        std::string s(buf);
        if (s.find_first_of(".eE") == std::string::npos)
            s += ".0";
        out += s;
    }

    void write(std::string& out, int indent, int depth) const {
        char buf[32];
        switch (kind_) {
        case Kind::null_value:
            out += "null";
            return;
        case Kind::boolean:
            out += b_ ? "true" : "false";
            return;
        case Kind::signed_int:
            std::snprintf(buf, sizeof(buf), "%lld", i_);
            out += buf;
            return;
        case Kind::unsigned_int:
            std::snprintf(buf, sizeof(buf), "%llu", u_);
            out += buf;
            return;
        case Kind::number:
            write_double(out, d_);
            return;
        case Kind::string:
            write_string(out, s_);
            return;
        case Kind::array:
        case Kind::object:
            break;
        }
        const bool is_obj = (kind_ == Kind::object);
        if (items_.empty()) {
            out += is_obj ? "{}" : "[]";
            return;
        }
        const std::string inner(static_cast<std::size_t>(indent * (depth + 1)), ' ');
        const std::string outer(static_cast<std::size_t>(indent * depth), ' ');
        out += is_obj ? "{\n" : "[\n";
        for (std::size_t i = 0; i < items_.size(); ++i) {
            out += inner;
            if (is_obj) {
                write_string(out, keys_[i]);
                out += ": ";
            }
            items_[i].write(out, indent, depth + 1);
            if (i + 1 < items_.size())
                out += ',';
            out += '\n';
        }
        out += outer;
        out += is_obj ? '}' : ']';
    }

    Kind kind_ = Kind::null_value;
    bool b_ = false;
    long long i_ = 0;
    unsigned long long u_ = 0;
    double d_ = 0.0;
    std::string s_;
    std::vector<std::string> keys_; // object only, parallel to items_
    std::vector<JVal> items_;       // array elements or object values
};

// Object literal helper: jobj({{"key", value}, ...}) preserves the order given.
struct JMember {
    JMember(const char* k, const JVal& v) : key(k), value(v) {}
    std::string key;
    JVal value;
};

inline JVal jobj(std::initializer_list<JMember> members) {
    JVal j = JVal::object();
    for (const JMember& m : members)
        j[m.key] = m.value;
    return j;
}

// ---------------------------------------------------------------------------
// Reader (SF-32 addition)
// ---------------------------------------------------------------------------

namespace json_detail {

class Parser {
  public:
    explicit Parser(const std::string& s) : s_(s) {}

    JVal parse_document() {
        skip_ws();
        JVal v = parse_value(0);
        skip_ws();
        if (pos_ != s_.size())
            fail("trailing characters");
        return v;
    }

  private:
    [[noreturn]] void fail(const char* what) const {
        throw std::runtime_error(std::string("json: ") + what + " at offset " +
                                 std::to_string(pos_));
    }
    void skip_ws() {
        while (pos_ < s_.size() &&
               (s_[pos_] == ' ' || s_[pos_] == '\n' || s_[pos_] == '\r' || s_[pos_] == '\t'))
            ++pos_;
    }
    bool consume(const char* lit) {
        std::size_t k = 0;
        while (lit[k] != '\0') {
            if (pos_ + k >= s_.size() || s_[pos_ + k] != lit[k])
                return false;
            ++k;
        }
        pos_ += k;
        return true;
    }
    JVal parse_value(int depth) {
        if (depth > 64)
            fail("nesting too deep");
        if (pos_ >= s_.size())
            fail("unexpected end");
        const char c = s_[pos_];
        if (c == '{')
            return parse_object(depth);
        if (c == '[')
            return parse_array(depth);
        if (c == '"')
            return JVal(parse_string());
        if (consume("true"))
            return JVal(true);
        if (consume("false"))
            return JVal(false);
        if (consume("null"))
            return JVal(nullptr);
        return parse_number();
    }
    JVal parse_object(int depth) {
        ++pos_; // '{'
        JVal obj = JVal::object();
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == '}') {
            ++pos_;
            return obj;
        }
        for (;;) {
            skip_ws();
            if (pos_ >= s_.size() || s_[pos_] != '"')
                fail("expected a key");
            const std::string key = parse_string();
            skip_ws();
            if (pos_ >= s_.size() || s_[pos_] != ':')
                fail("expected ':'");
            ++pos_;
            skip_ws();
            if (obj.find(key) != nullptr)
                fail("duplicate key");
            obj[key] = parse_value(depth + 1);
            skip_ws();
            if (pos_ < s_.size() && s_[pos_] == ',') {
                ++pos_;
                continue;
            }
            if (pos_ < s_.size() && s_[pos_] == '}') {
                ++pos_;
                return obj;
            }
            fail("expected ',' or '}'");
        }
    }
    JVal parse_array(int depth) {
        ++pos_; // '['
        JVal arr = JVal::array();
        skip_ws();
        if (pos_ < s_.size() && s_[pos_] == ']') {
            ++pos_;
            return arr;
        }
        for (;;) {
            skip_ws();
            arr.push_back(parse_value(depth + 1));
            skip_ws();
            if (pos_ < s_.size() && s_[pos_] == ',') {
                ++pos_;
                continue;
            }
            if (pos_ < s_.size() && s_[pos_] == ']') {
                ++pos_;
                return arr;
            }
            fail("expected ',' or ']'");
        }
    }
    std::string parse_string() {
        ++pos_; // '"'
        std::string out;
        for (;;) {
            if (pos_ >= s_.size())
                fail("unterminated string");
            const char c = s_[pos_++];
            if (c == '"')
                return out;
            if (c != '\\') {
                out += c;
                continue;
            }
            if (pos_ >= s_.size())
                fail("unterminated escape");
            const char e = s_[pos_++];
            switch (e) {
            case '"':
                out += '"';
                break;
            case '\\':
                out += '\\';
                break;
            case '/':
                out += '/';
                break;
            case 'b':
                out += '\b';
                break;
            case 'f':
                out += '\f';
                break;
            case 'n':
                out += '\n';
                break;
            case 'r':
                out += '\r';
                break;
            case 't':
                out += '\t';
                break;
            case 'u': {
                if (pos_ + 4 > s_.size())
                    fail("short \\u escape");
                const std::string hex = s_.substr(pos_, 4);
                pos_ += 4;
                char* end = nullptr;
                const long cp = std::strtol(hex.c_str(), &end, 16);
                if (end != hex.c_str() + 4 || cp < 0 || cp > 0x7f)
                    fail("unsupported \\u escape");
                out += static_cast<char>(cp);
                break;
            }
            default:
                fail("bad escape");
            }
        }
    }
    JVal parse_number() {
        const std::size_t start = pos_;
        bool is_float = false;
        if (pos_ < s_.size() && (s_[pos_] == '-' || s_[pos_] == '+'))
            ++pos_;
        while (pos_ < s_.size()) {
            const char c = s_[pos_];
            if (c >= '0' && c <= '9') {
                ++pos_;
            } else if (c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+') {
                is_float = true;
                ++pos_;
            } else {
                break;
            }
        }
        const std::string tok = s_.substr(start, pos_ - start);
        if (tok.empty() || tok == "-" || tok == "+")
            fail("bad value");
        char* end = nullptr;
        if (!is_float) {
            errno = 0;
            if (tok[0] == '-') {
                const long long v = std::strtoll(tok.c_str(), &end, 10);
                if (errno == 0 && end == tok.c_str() + tok.size())
                    return JVal::make_signed(v);
            } else {
                const unsigned long long v = std::strtoull(tok.c_str(), &end, 10);
                if (errno == 0 && end == tok.c_str() + tok.size()) {
                    if (v <= 9223372036854775807ULL)
                        return JVal::make_signed(static_cast<long long>(v));
                    return JVal::make_unsigned(v);
                }
            }
        }
        const double d = std::strtod(tok.c_str(), &end);
        if (end != tok.c_str() + tok.size())
            fail("bad number");
        return JVal::make_double(d);
    }

    const std::string& s_;
    std::size_t pos_ = 0;
};

} // namespace json_detail

/// Parse a complete JSON document (throws std::runtime_error on malformed input).
inline JVal parse_json(const std::string& text) {
    json_detail::Parser p(text);
    return p.parse_document();
}

} // namespace spurious_spreading
