"""SF-18 periodic Gaussian generator, reimplemented in Python (SF-42 N3).

Line-by-line port of `src/physics/stochastic/PeriodicGaussianField.cu` (kernel_fill_spectrum + the
host reduction): stateless splitmix64 hash of (seed, m1, m2, m3) -> Box-Muller -> spectrum
`S = sigma2 pi^(3/2) l^3 exp(-k^2 l^2/4) / (Lx Ly Lz)`, zeroed zero mode and Nyquist planes, canonical
Hermitian rule on the mx = 0 plane, half-cell phase twist, cuFFT Z2D layout (half-spectrum array
(nz, ny, nx/2+1), x fastest), two-pass mean / variance and `applied_scale = sqrt(sigma2 / raw_variance)`.

Two products:
  (a) `generate(...)`: Y on the generator's own grid (cell centres, array (nz, ny, nx), element
      [k, j, i] = Y(i, j, k)); agrees with the GPU to round-off (the libm calls differ by an ulp).
  (b) `field_on_grid(f, M, offset)`: the SAME continuous band-limited field of the generating grid
      (its mode set |m_d| <= N_d/2 - 1, its applied scale) on any finer grid by zero-padding the
      spectrum; `offset` is the sample position in cells of the target grid (0.5: cell centres,
      0: vertices).  The coefficient of a mode depends only on (seed, m), so a finer generating grid
      adds modes but never changes the coarser ones (header of the CUDA file, item 4).

Only numpy + scipy (host stack: numpy 1.26 / scipy 1.11; current: numpy 2.5 / scipy 1.18).
"""
import numpy as np
import scipy.fft as sfft

_MASK = (1 << 64) - 1
_U = np.uint64
_GOLD = _U(0x9E3779B97F4A7C15)
_MIX1 = _U(0xBF58476D1CE4E5B9)
_MIX2 = _U(0x94D049BB133111EB)
_KM1 = _U(0xD1B54A32D192ED03)
_KM2 = _U(0xA24BAED4963EE407)
_KM3 = _U(0x9FB21C651E98DF25)
# `static_cast<real>(1ULL << 53) + 1.0` in double arithmetic: 2^53 + 1 is not representable, so this is
# 2^53 (the "strictly inside (0,1)" of the CUDA comment can be 1.0 with probability 2^-53; harmless).
_DENOM = float(1 << 53) + 1.0
PI = 3.141592653589793238462643383279502884


def _splitmix64(x):
    x = x + _GOLD
    x = (x ^ (x >> _U(30))) * _MIX1
    x = (x ^ (x >> _U(27))) * _MIX2
    return x ^ (x >> _U(31))


def _hash_mode(seed, m1, m2, m3, salt):
    """hash_mode of the CUDA file; m1, m2, m3 integer arrays of one shape (>= 1-d)."""
    h = np.full(m1.shape, _U((int(seed) ^ 0x9E3779B97F4A7C15) & _MASK), dtype=np.uint64)
    for m, k in ((m1, _KM1), (m2, _KM2), (m3, _KM3)):
        mu = np.ascontiguousarray(m, dtype=np.int64).view(np.uint64)  # two's complement, as the cast
        h = _splitmix64(h ^ (mu * k))
    return _splitmix64(h ^ _U(salt))


def _open_unit(x):
    return ((x >> _U(11)) + _U(1)).astype(np.float64) / _DENOM


def _box_muller(seed, m1, m2, m3):
    u1 = _open_unit(_hash_mode(seed, m1, m2, m3, 0))
    u2 = _open_unit(_hash_mode(seed, m1, m2, m3, 1))
    r = np.sqrt(-2.0 * np.log(u1))
    theta = (2.0 * PI) * u2
    return r * np.cos(theta), r * np.sin(theta)


class SF18Field:
    """One SF-18 realization. `c` is the UNtwisted half-spectrum (nz, ny, nx//2+1) of the unscaled
    field (zero on the zero mode and the Nyquist planes); `y` the cell-centred sample on the
    generating grid after the optional variance normalization."""

    def __init__(self, nx, ny, nz, c, y, raw_mean, raw_variance, applied_scale, sigma2, ell, seed, h):
        self.nx, self.ny, self.nz = nx, ny, nz
        self.c, self.y = c, y
        self.raw_mean, self.raw_variance, self.applied_scale = raw_mean, raw_variance, applied_scale
        self.sigma2, self.ell, self.seed, self.h = sigma2, ell, seed, h

    @property
    def active_mode_count(self):
        """Number of stored (half-spectrum) modes, as `PeriodicGaussianFieldReport::active_mode_count`."""
        return int(np.count_nonzero(self.c))


def spectrum(nx, ny, nz, sigma2, ell, seed, h):
    """Untwisted half-spectrum c[gz, gy, gx] of kernel_fill_spectrum, and the twist phase array."""
    if min(nx, ny, nz) <= 0 or nx % 2 or ny % 2 or nz % 2:
        raise ValueError("grid extents must be positive and even")
    hx = nx // 2 + 1
    gx = np.arange(hx)[None, None, :]
    gy = np.arange(ny)[None, :, None]
    gz = np.arange(nz)[:, None, None]
    mx = np.broadcast_to(gx, (nz, ny, hx))
    my = np.broadcast_to(np.where(gy < ny // 2, gy, gy - ny), (nz, ny, hx))
    mz = np.broadcast_to(np.where(gz < nz // 2, gz, gz - nz), (nz, ny, hx))
    nyq = (mx == nx // 2) | (gy == ny // 2) | (gz == nz // 2)
    zero = (mx == 0) & (my == 0) & (mz == 0)
    active = ~(nyq | zero)
    mx, my, mz = mx[active], my[active], mz[active]
    # canonical Hermitian representative on the mx == 0 plane
    flip = (mx == 0) & ~((my > 0) | ((my == 0) & (mz > 0)))
    cmy = np.where(flip, -my, my)
    cmz = np.where(flip, -mz, mz)
    re_std, im_std = _box_muller(seed, mx, cmy, cmz)
    Lx, Ly, Lz = nx * h, ny * h, nz * h
    kx = (2.0 * PI) * mx.astype(np.float64) / Lx
    ky = (2.0 * PI) * my.astype(np.float64) / Ly
    kz = (2.0 * PI) * mz.astype(np.float64) / Lz
    k2 = kx * kx + ky * ky + kz * kz
    S = sigma2 * PI ** 1.5 * ell * ell * ell * np.exp(-k2 * ell * ell * 0.25) / (Lx * Ly * Lz)
    amp = np.sqrt(S * 0.5)
    re = re_std * amp
    im = np.where(flip, -(im_std * amp), im_std * amp)
    c = np.zeros((nz, ny, hx), dtype=np.complex128)
    c[active] = re + 1j * im
    phase = np.zeros((nz, ny, hx))
    phase[active] = 0.5 * (kx * h + ky * h + kz * h)
    return c, phase


def generate(nx, ny=None, nz=None, sigma2=1.0, ell=0.25, seed=0, normalize=True, h=None, workers=None):
    """The GPU generator `generate_periodic_gaussian_field` on Grid3D(nx, ny, nz, h, h, h)
    (default h = 1/ny, the box-unit convention of the consumer and of `sf42_periodic_flow`)."""
    ny = nx if ny is None else ny
    nz = nx if nz is None else nz
    h = 1.0 / ny if h is None else h
    c, phase = spectrum(nx, ny, nz, sigma2, ell, seed, h)
    twisted = c * (np.cos(phase) + 1j * np.sin(phase))
    y = sfft.irfftn(twisted, s=(nz, ny, nx), norm="forward", workers=workers)  # cuFFT Z2D: unnormalized
    raw_mean = float(y.mean())
    raw_variance = float(np.mean((y - raw_mean) ** 2))
    scale = 1.0
    if normalize:
        if not raw_variance > 0.0:
            raise RuntimeError("raw_variance <= 0, cannot normalize")
        scale = float(np.sqrt(sigma2 / raw_variance))
        y *= scale
    return SF18Field(nx, ny, nz, c, y, raw_mean, raw_variance, scale, sigma2, ell, seed, h)


def embed(c, nx, ny, nz, Mx, My, Mz):
    """Re-index the half-spectrum of an (nz, ny, nx) generating grid on a (Mz, My, Mx//2+1) one: zero-padding when the
    target is finer, dropping the modes with |m_d| >= M_d/2 when it is coarser (their variance is <= 4e-17 of the
    total below M = 16 at ell = 1/4 and exactly representable as 0 beyond M = 64)."""
    gy, gz = np.arange(ny), np.arange(nz)
    my, mz = np.where(gy < ny // 2, gy, gy - ny), np.where(gz < nz // 2, gz, gz - nz)
    ky, kz = np.abs(my) <= My // 2 - 1, np.abs(mz) <= Mz // 2 - 1
    kx = np.arange(nx // 2 + 1) <= Mx // 2 - 1
    out = np.zeros((Mz, My, Mx // 2 + 1), dtype=np.complex128)
    out[np.ix_(mz[kz] % Mz, my[ky] % My, np.arange(nx // 2 + 1)[kx])] = c[np.ix_(np.flatnonzero(kz), np.flatnonzero(ky), np.flatnonzero(kx))]
    return out


def field_on_grid(f, M, offset=0.5, scale=None, workers=None):
    """The continuous field of generator `f` sampled at x_a = (a + offset) / M * L on an M^3 grid (or an
    (Mx, My, Mz) tuple), array (Mz, My, Mx).  `scale` defaults to the applied scale of `f` (1 when the
    generator was run without normalization).  offset = 0.5 reproduces `f.y` when M equals the
    generating grid (to round-off)."""
    Mx, My, Mz = (M, M, M) if np.isscalar(M) else M
    cM = embed(f.c, f.nx, f.ny, f.nz, Mx, My, Mz)
    if offset != 0.0:  # phase of the sample shift, per axis: exp(2 pi i m offset / M_axis)
        mx = np.arange(Mx // 2 + 1)
        my = np.fft.fftfreq(My, 1.0 / My)
        mz = np.fft.fftfreq(Mz, 1.0 / Mz)
        ph = (mx[None, None, :] / Mx + my[None, :, None] / My + mz[:, None, None] / Mz) * (2.0 * PI * offset)
        cM = cM * (np.cos(ph) + 1j * np.sin(ph))
    y = sfft.irfftn(cM, s=(Mz, My, Mx), norm="forward", workers=workers)
    return y * (f.applied_scale if scale is None else scale)


def mode_variance_outside(f, n_inner):
    """Continuum variance (after the applied scale) of the modes of `f` OUTSIDE the cube
    |m_d| <= n_inner/2 - 1: the mean-square difference between the field of the generating grid
    `f` and the field of the cubic generating grid n_inner (scale differences are O(1e-16))."""
    nx, ny, nz = f.nx, f.ny, f.nz
    mx = np.arange(nx // 2 + 1)[None, None, :]
    gy = np.arange(ny)[None, :, None]
    gz = np.arange(nz)[:, None, None]
    my = np.where(gy < ny // 2, gy, gy - ny)
    mz = np.where(gz < nz // 2, gz, gz - nz)
    lim = n_inner // 2 - 1
    outside = (mx > lim) | (np.abs(my) > lim) | (np.abs(mz) > lim)
    w = np.where(mx == 0, 1.0, 2.0)  # half-spectrum: every mx > 0 mode stands for a +-m pair
    return float(np.sum(w * np.abs(f.c) ** 2 * outside) * f.applied_scale ** 2)
