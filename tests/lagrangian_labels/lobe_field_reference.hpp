#pragma once

/**
 * @file lobe_field_reference.hpp
 * @brief SF-41 N3a: host reference of the C4b control (inlet exit on a harmonic k = 1 field with a backflow
 *        lobe at the inlet). Self-contained, no device code; reused by the Pollock tests (N3a) and the
 *        forward-tracker tests (N3b).
 *
 * Field (G1 = 1, a = 0.5 by default; x3-independent, k = 1):
 *   Phi = G1 x1 + a cos(2 pi x2) exp(-2 pi x1)             (harmonic)
 *   v   = grad Phi = (G1 - 2 pi a cos(2 pi x2) e, -2 pi a sin(2 pi x2) e, 0),  e = exp(-2 pi x1)
 *   psi = G1 x2 - a e sin(2 pi x2)                          (v1 = d psi/d x2, v2 = -d psi/d x1)
 * The separatrix is psi in Z (periodic in x2: psi(x2 + 1) = psi + 1); a > G1/(2 pi) gives a stagnation
 * point at x1s = ln(2 pi a / G1)/(2 pi) and a backflow lobe at the inlet. Streamlines are the level sets
 * of psi, so the reference classification of a seed is the exit of an exact integral curve.
 *
 * Face fluxes (per unit area, exact averages of v over each cell face, h x h faces; x3-independence makes
 * the z average trivial):
 *   U(i; j) = G1 - a e(x_i) (sin(2 pi y_{j+1}) - sin(2 pi y_j)) / h          x-face at x = i h
 *   V(i; j) = -a sin(2 pi y_j) (e(x_i) - e(x_{i+1})) / h                    y-face at y = j h
 *   W = 0
 * returned in the CompactMAC layout of SF-19 (U (nx+1) ny nz, V nx (ny+1) nz, W nx ny (nz+1)).
 *
 * Reference classification (independent of every discretization): host adaptive Dormand-Prince 5(4)
 * (rtol 1e-12, atol 1e-14, h <= 1/256) on the analytic field from (X0, x2) until the first accepted step
 * that ends at x1 < 0 (exit_inlet) or x1 >= L1 (exit_outlet); the exit time is interpolated linearly in
 * x1 inside that step; `reached_plane` = max x1 along the path before an inlet exit >= Xp; the stream
 * function drift |psi - psi_0| at the integrated end-of-step state is returned for the check.
 */

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace macroflow3d {
namespace lobe_ref {

inline constexpr double kTwoPi = 6.283185307179586476925286766559;

struct LobeField {
    double G1 = 1.0;
    double a = 0.5;
    double v1(double x1, double x2) const { return G1 - kTwoPi * a * std::cos(kTwoPi * x2) * std::exp(-kTwoPi * x1); }
    double v2(double x1, double x2) const { return -kTwoPi * a * std::sin(kTwoPi * x2) * std::exp(-kTwoPi * x1); }
    double psi(double x1, double x2) const { return G1 * x2 - a * std::exp(-kTwoPi * x1) * std::sin(kTwoPi * x2); }
};

struct CompactMacHost {
    std::vector<double> U, V, W;
    double max_cell_divergence = 0.0; ///< max |(U[i+1]-U[i])/h + (V[j+1]-V[j])/h| over the cells
};

/// Exact face-flux integrals of the lobe field on an (nx, ny, nz) grid of spacing h (CompactMAC layout).
inline CompactMacHost lobe_face_fluxes(const LobeField& f, int nx, int ny, int nz, double h) {
    CompactMacHost c;
    const std::size_t sx = static_cast<std::size_t>(nx), sy = static_cast<std::size_t>(ny),
                      sz = static_cast<std::size_t>(nz);
    c.U.assign((sx + 1) * sy * sz, 0.0);
    c.V.assign(sx * (sy + 1) * sz, 0.0);
    c.W.assign(sx * sy * (sz + 1), 0.0);
    std::vector<double> e(sx + 1);
    for (int i = 0; i <= nx; ++i)
        e[static_cast<std::size_t>(i)] = std::exp(-kTwoPi * (static_cast<double>(i) * h));
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j < ny; ++j) {
            const double s0 = std::sin(kTwoPi * (static_cast<double>(j) * h));
            const double s1 = std::sin(kTwoPi * (static_cast<double>(j + 1) * h));
            for (int i = 0; i <= nx; ++i)
                c.U[static_cast<std::size_t>(i) + (sx + 1) * (static_cast<std::size_t>(j) + sy * static_cast<std::size_t>(k))] =
                    f.G1 - f.a * e[static_cast<std::size_t>(i)] * (s1 - s0) / h;
        }
    for (int k = 0; k < nz; ++k)
        for (int j = 0; j <= ny; ++j) {
            const double sj = std::sin(kTwoPi * (static_cast<double>(j) * h));
            for (int i = 0; i < nx; ++i)
                c.V[static_cast<std::size_t>(i) + sx * static_cast<std::size_t>(j) + sx * (sy + 1) * static_cast<std::size_t>(k)] =
                    -f.a * sj * (e[static_cast<std::size_t>(i)] - e[static_cast<std::size_t>(i) + 1]) / h;
        }
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i) {
            const double d = (c.U[static_cast<std::size_t>(i) + 1 + (sx + 1) * static_cast<std::size_t>(j)] -
                              c.U[static_cast<std::size_t>(i) + (sx + 1) * static_cast<std::size_t>(j)]) / h +
                             (c.V[static_cast<std::size_t>(i) + sx * (static_cast<std::size_t>(j) + 1)] -
                              c.V[static_cast<std::size_t>(i) + sx * static_cast<std::size_t>(j)]) / h;
            c.max_cell_divergence = std::max(c.max_cell_divergence, std::abs(d));
        }
    return c;
}

enum class ExitKind { inlet, outlet };

struct LobeReference {
    ExitKind exit = ExitKind::outlet;
    bool reached_plane = false; ///< max x1 before an inlet exit (or the outlet exit) >= Xp
    double t_exit = 0.0;        ///< exit clock (linear interpolation in x1 inside the exit step)
    double psi_drift = 0.0;     ///< |psi(end) - psi(start)|
};

/// Integral curve of v through (X0, x2): Dormand-Prince 5(4), see the file header.
inline LobeReference classify_lobe_seed(const LobeField& f, double L1, double X0, double x2, double Xp) {
    static const double a21 = 1.0 / 5.0, a31 = 3.0 / 40.0, a32 = 9.0 / 40.0, a41 = 44.0 / 45.0,
                        a42 = -56.0 / 15.0, a43 = 32.0 / 9.0, a51 = 19372.0 / 6561.0,
                        a52 = -25360.0 / 2187.0, a53 = 64448.0 / 6561.0, a54 = -212.0 / 729.0,
                        a61 = 9017.0 / 3168.0, a62 = -355.0 / 33.0, a63 = 46732.0 / 5247.0,
                        a64 = 49.0 / 176.0, a65 = -5103.0 / 18656.0, b1 = 35.0 / 384.0,
                        b3 = 500.0 / 1113.0, b4 = 125.0 / 192.0, b5 = -2187.0 / 6784.0,
                        b6 = 11.0 / 84.0, e1 = 71.0 / 57600.0, e3 = -71.0 / 16695.0,
                        e4 = 71.0 / 1920.0, e5 = -17253.0 / 339200.0, e6 = 22.0 / 525.0,
                        e7 = -1.0 / 40.0;
    const double rtol = 1e-12, atol = 1e-14, hmax = 1.0 / 256.0;
    double x = X0, y = x2, t = 0.0, h = 1e-3, xmax = X0;
    LobeReference ref;
    auto rhs = [&](double px, double py, double& d1, double& d2) {
        d1 = f.v1(px, py);
        d2 = f.v2(px, py);
    };
    for (long long step = 0; step < 50000000LL; ++step) {
        double k1x, k1y, k2x, k2y, k3x, k3y, k4x, k4y, k5x, k5y, k6x, k6y, k7x, k7y;
        rhs(x, y, k1x, k1y);
        rhs(x + h * a21 * k1x, y + h * a21 * k1y, k2x, k2y);
        rhs(x + h * (a31 * k1x + a32 * k2x), y + h * (a31 * k1y + a32 * k2y), k3x, k3y);
        rhs(x + h * (a41 * k1x + a42 * k2x + a43 * k3x), y + h * (a41 * k1y + a42 * k2y + a43 * k3y), k4x, k4y);
        rhs(x + h * (a51 * k1x + a52 * k2x + a53 * k3x + a54 * k4x),
            y + h * (a51 * k1y + a52 * k2y + a53 * k3y + a54 * k4y), k5x, k5y);
        rhs(x + h * (a61 * k1x + a62 * k2x + a63 * k3x + a64 * k4x + a65 * k5x),
            y + h * (a61 * k1y + a62 * k2y + a63 * k3y + a64 * k4y + a65 * k5y), k6x, k6y);
        const double xn = x + h * (b1 * k1x + b3 * k3x + b4 * k4x + b5 * k5x + b6 * k6x);
        const double yn = y + h * (b1 * k1y + b3 * k3y + b4 * k4y + b5 * k5y + b6 * k6y);
        rhs(xn, yn, k7x, k7y);
        const double ex = h * (e1 * k1x + e3 * k3x + e4 * k4x + e5 * k5x + e6 * k6x + e7 * k7x);
        const double ey = h * (e1 * k1y + e3 * k3y + e4 * k4y + e5 * k5y + e6 * k6y + e7 * k7y);
        const double sx = atol + rtol * std::max(std::abs(x), std::abs(xn));
        const double sy = atol + rtol * std::max(std::abs(y), std::abs(yn));
        const double err = std::max(std::abs(ex) / sx, std::abs(ey) / sy);
        const double fac = std::min(5.0, std::max(0.2, 0.9 * std::pow(std::max(err, 1e-10), -0.2)));
        if (err <= 1.0) {
            if (xn < 0.0 || xn >= L1) {
                const double bnd = (xn < 0.0) ? 0.0 : L1;
                ref.exit = (xn < 0.0) ? ExitKind::inlet : ExitKind::outlet;
                ref.t_exit = t + h * (bnd - x) / (xn - x);
                // drift at the integrated end-of-step state (the analytic field extends beyond the box)
                ref.psi_drift = std::abs(f.psi(xn, yn) - f.psi(X0, x2));
                xmax = std::max(xmax, std::min(xn, L1));
                ref.reached_plane = xmax >= Xp;
                return ref;
            }
            x = xn;
            y = yn;
            t += h;
            xmax = std::max(xmax, x);
        }
        h = std::min(hmax, h * fac);
        if (!(h > 1e-14))
            throw std::runtime_error("lobe_field_reference: step underflow");
    }
    throw std::runtime_error("lobe_field_reference: step cap exceeded");
}

/// One seed of the M x M lattice x2 = (m2 + 1/2)/M, x3 = (m3 + 1/2)/M (window [0,1)^2, index p = m2 M + m3).
struct LobeSeed {
    int m2 = 0, m3 = 0;
    double x2 = 0.0, x3 = 0.0;
    double v1 = 0.0;          ///< analytic v1 at (X0, x2)
    double psi_margin = 0.0;  ///< |psi0 - round(psi0)|
    bool included = false;    ///< v1 >= v1min and psi_margin >= margin_min
    LobeReference ref;        ///< valid iff included
};

struct LobeLattice {
    std::vector<LobeSeed> seeds; ///< M * M, index p = m2 M + m3
    int n_included = 0, n_inlet = 0, n_outlet = 0, n_plane = 0; ///< over included seeds
    double max_t_exit_outlet = 0.0;                              ///< over included outlet seeds
    double max_psi_drift = 0.0;
};

/// Lattice with the C4b inclusion rule (defaults of the specification: v1 >= 0.05, |psi0 - round(psi0)| >= 0.02)
/// and the reference classification of every included seed (computed once per distinct x2).
inline LobeLattice make_lobe_lattice(const LobeField& f, double L1, double X0, int M, double Xp,
                                     double v1min = 0.05, double margin_min = 0.02) {
    LobeLattice lat;
    lat.seeds.resize(static_cast<std::size_t>(M) * static_cast<std::size_t>(M));
    for (int m2 = 0; m2 < M; ++m2) {
        const double x2 = (static_cast<double>(m2) + 0.5) / M;
        const double psi0 = f.psi(X0, x2);
        const double v1 = f.v1(X0, x2);
        const double mg = std::abs(psi0 - std::round(psi0));
        const bool inc = (v1 >= v1min) && (mg >= margin_min);
        LobeReference ref;
        if (inc)
            ref = classify_lobe_seed(f, L1, X0, x2, Xp);
        for (int m3 = 0; m3 < M; ++m3) {
            LobeSeed& s = lat.seeds[static_cast<std::size_t>(m2) * static_cast<std::size_t>(M) + static_cast<std::size_t>(m3)];
            s.m2 = m2;
            s.m3 = m3;
            s.x2 = x2;
            s.x3 = (static_cast<double>(m3) + 0.5) / M;
            s.v1 = v1;
            s.psi_margin = mg;
            s.included = inc;
            s.ref = ref;
            if (inc) {
                ++lat.n_included;
                (ref.exit == ExitKind::inlet ? lat.n_inlet : lat.n_outlet) += 1;
                lat.n_plane += ref.reached_plane ? 1 : 0;
                lat.max_psi_drift = std::max(lat.max_psi_drift, ref.psi_drift);
                if (ref.exit == ExitKind::outlet)
                    lat.max_t_exit_outlet = std::max(lat.max_t_exit_outlet, ref.t_exit);
            }
        }
    }
    return lat;
}

/// Snapshot time of the time pass: 1.25 x the largest included outlet exit time, rounded up to 0.5.
inline double lobe_time_T(const LobeLattice& lat) {
    return std::ceil(1.25 * lat.max_t_exit_outlet * 2.0) / 2.0;
}

} // namespace lobe_ref
} // namespace macroflow3d
