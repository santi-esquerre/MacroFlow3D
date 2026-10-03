"""SF-29 N1 -- inlet labels in flow coordinates (normalized triangular construction, deviation D-1).

On the inlet face x1 = 0 (periodic in x2, x3), with v1 = v1(0, x2, x3) > 0:

    Q(x3)       = int_0^1 v1(0, s, x3) ds                         (> 0)
    psi2^0(x3)  = int_0^{x3} Q(t) dt            = Q0 x3 + periodic  (Q0 = face flux = 1)
    psi1^0(x2,x3) = int_0^{x2} v1(0, s, x3) ds / Q(x3) = x2 + periodic

so that d2 psi1^0 d3 psi2^0 - d3 psi1^0 d2 psi2^0 = (v1/Q) Q - 0 = v1 on the face.

Representation: v1 is sampled on an nf x nf face grid and replaced by its 2-D trigonometric
interpolant V (Nyquist lines dropped).  Then, exactly for V,
    Q(x3)       = sum_{m3} V(0, m3) e3,
    psi2^0      = Q0 x3 + sum_{m3 != 0} V(0, m3) (e3 - 1)/(i 2 pi m3),
    num(x2, x3) = sum_{m2 != 0, m3} V(m2, m3) (e2 - 1) e3 / (i 2 pi m2)   (periodic part of the x2-integral),
    psi1^0      = x2 + num(x2, x3) / Q(x3).
Both labels are evaluated at arbitrary (unwrapped) face points to the spectral accuracy of V.
"""
import numpy as np

TWO_PI = 2.0 * np.pi


def _modes(n):
    m = np.fft.fftfreq(n, d=1.0 / n)
    if n % 2 == 0:
        m[n // 2] = 0.0
    return m


class InletLabels:
    def __init__(self, ref, nf):
        self.nf = int(nf)
        s = np.arange(nf) / float(nf)
        Y, Z = np.meshgrid(s, s, indexing="ij")
        v = ref.velocity(0.0, Y.ravel(), Z.ravel())
        self.v1_samples = v[:, 0].reshape(nf, nf)
        V = np.fft.fft2(self.v1_samples) / float(nf * nf)       # V[m2, m3]
        if nf % 2 == 0:
            V[nf // 2, :] = 0.0
            V[:, nf // 2] = 0.0
        self.V = V
        self.m = _modes(nf)
        self.kk = TWO_PI * self.m
        self.Q0 = float(V[0, 0].real)                            # mean face flux
        self.Qh = V[0, :].copy()                                 # Q(x3) coefficients
        A = np.zeros_like(V)
        nz = self.m != 0
        A[nz, :] = V[nz, :] / (1j * self.kk[nz])[:, None]
        self.A = A                                               # num coefficients
        B = np.zeros_like(self.Qh)
        B[nz] = self.Qh[nz] / (1j * self.kk[nz])
        self.B = B                                               # psi2 periodic coefficients
        self.vmin = float(self.v1_samples.min())
        if not self.vmin > 0.0:
            raise RuntimeError("inlet face v1 not positive (min %.3e)" % self.vmin)

    # ---- elementary sums -------------------------------------------------------------------
    def _e(self, x):
        return np.exp(1j * np.outer(np.asarray(x, float).ravel(), self.kk))   # (P, nf)

    def Q(self, z, deriv=0):
        E3 = self._e(z)
        c = self.Qh * (1j * self.kk) ** deriv
        return (E3 @ c).real

    def _num(self, y, z, d2=0, d3=0):
        """periodic part of int_0^{x2} V ds (and its derivatives)."""
        y = np.asarray(y, float).ravel(); z = np.asarray(z, float).ravel()
        E2 = self._e(y); E3 = self._e(z)
        if d2 == 0:
            E2 = E2 - 1.0
        else:
            E2 = E2 * (1j * self.kk[None, :]) ** d2
        if d3:
            E3 = E3 * (1j * self.kk[None, :]) ** d3
        return np.einsum("pm,pm->p", E2 @ self.A, E3).real

    # ---- labels ----------------------------------------------------------------------------
    def psi1(self, y, z):
        y = np.asarray(y, float).ravel()
        return y + self._num(y, z) / self.Q(z)

    def psi2(self, z):
        z = np.asarray(z, float).ravel()
        E3 = self._e(z) - 1.0
        return self.Q0 * z + (E3 @ self.B).real

    def labels(self, y, z):
        return self.psi1(y, z), self.psi2(z)

    def V_eval(self, y, z):
        """2-D trigonometric interpolant of v1 on the face."""
        E2 = self._e(y); E3 = self._e(z)
        return np.einsum("pm,pm->p", E2 @ self.V, E3).real

    def jacobian(self, y, z):
        """d2psi1 d3psi2 - d3psi1 d2psi2 from the analytic derivatives of the representation."""
        y = np.asarray(y, float).ravel(); z = np.asarray(z, float).ravel()
        Qz = self.Q(z); dQ = self.Q(z, 1)
        num = self._num(y, z); n2 = self._num(y, z, d2=1); n3 = self._num(y, z, d3=1)
        d2p1 = 1.0 + n2 / Qz
        d3p1 = (n3 * Qz - num * dQ) / Qz**2
        d2p2 = 0.0 * y
        d3p2 = Qz
        return d2p1 * d3p2 - d3p1 * d2p2
