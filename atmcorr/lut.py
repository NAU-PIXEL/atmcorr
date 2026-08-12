"""
Reader for atmospheric-correction lookup tables (LUTs) — part of ``atmcorr``.

An atmospheric LUT is produced in two stages: RFMwrapper runs line-by-line RFM +
the MT_CKD continuum to build a filter-agnostic **master** of optical-depth
spectra (the only stage needing RFM), and :func:`atmcorr.build_instrument`
convolves an instrument's spectral response into the compact per-instrument LUT
read here. This reader — and the whole ``atmcorr`` package — has **no RFM
dependency**; only NumPy, SciPy and pandas. One LUT file describes one instrument
and may hold several named filters (e.g. the MMT-gasCam bands); a broadband LWIR
camera is simply the single-filter case.

``AtmosLUT.write`` (below) is the single definition of the on-disk format, shared
by the reader and by :func:`atmcorr.build_instrument` so the two cannot drift.

Representation — the k-distribution
-----------------------------------
For a *homogeneous* viewing path the band-integrated transmittance is

    tau_band(d) = (1 / sum_nu w_nu) * sum_nu w_nu * exp(-k(nu) * d)

where ``w_nu`` is the (SRF-weighted) spectral weight and ``k(nu)`` the monochromatic
absorption coefficient [km^-1]. Reordering the spectral sum by ascending ``k`` and
collapsing it onto a fixed set of ``g`` quadrature points gives the **k-distribution**:

    tau_band(d) = sum_g w_g * exp(-k_g * d)                              (exact)

This is *exact* for a homogeneous path (it is a reordering of the same integral,
not the correlated-k approximation), so a single stored ``k_g`` vector reproduces
transmittance at **any** distance with no distance axis and no distance-interpolation
error. The LUT therefore stores, per filter, the array ``k_g(T, P, RH)`` of shape
``(n_temp, n_pres, n_rh, n_g)`` plus the shared quadrature weights ``w_g``.

At query time the ambient state ``(T, P, RH)`` is fixed (one radiosonde / weather
reading per frame), the ``k_g`` vector is interpolated once, and transmittance is
evaluated in closed form and fully vectorised over a per-pixel distance map.

Symbol conventions
------------------
To avoid the classic radiative-transfer collision between *temperature* and
*transmittance* (both "T"), this module never uses a bare ``T``:

    ``temp``          ambient air temperature [K]        (grid axis)
    ``pres``          ambient pressure [mbar]            (grid axis)
    ``rh``            relative humidity [%]              (grid axis)
    ``dist``          path length [km native]           (query argument)
    ``tau`` / transmittance   band transmittance in [0, 1]   (primary output)
    ``k`` / ``k_g``   absorption coefficient [km^-1]     (stored; optical depth per km)

Optical depth is deliberately *not* exposed: the correction is applied with
transmittance, and ``optical_depth = -ln(transmittance)`` is a trivial conversion
that is never needed here. Note that per spectral slice the optical depth ``k_g * dist``
is exactly linear in distance (additive), but the *band* optical depth is not — which
is precisely why we sum slices rather than fit a single distance law.

On-disk format (``.npz``)
-------------------------
``format_version`` : str
``instrument``     : str
``wn_range``       : float (2,)  spectral range of the underlying RFM runs [cm^-1]
``temp``           : float (n_temp,)  grid axis [K]
``pres``           : float (n_pres,)  grid axis [mbar]
``rh``             : float (n_rh,)    grid axis [%]
``gweight``        : float (n_g,)     k-distribution quadrature weights (sum = 1)
``filters``        : str  (n_filt,)   filter names
``kdist__<name>``  : float (n_temp, n_pres, n_rh, n_g)  sorted k per node [km^-1]
``default_pres``   : float scalar     pressure used when a query omits it [mbar]
``meta``           : str              JSON provenance blob

The writer (:func:`AtmosLUT.write`) lives here so the factory and the reader share
one definition of the contract and cannot drift apart.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import nnls

from .registry import resolve_lut

# 2 — bands may be weighted by ``filter × detector`` rather than by the supplied
#     SRF alone, and ``meta`` records which, via 'weighting' and 'detector'. A
#     version-1 LUT carries no such record, so its weighting cannot be
#     established from the file at all: rebuild rather than assume.
FORMAT_VERSION = '2'

# Distance-map queries larger than this switch from direct evaluation to
# build-curve-then-interpolate; it also sets the curve's sample count.
_CURVE_POINTS = 1024

# Query-time convenience: arrays are accepted for the distance argument, scalars
# for the ambient state. Temperature/distance unit selectors keep the reader a
# drop-in for pipelines that carry Celsius / metres (e.g. the FLIR model).
ArrayLike = Union[float, np.ndarray]
TempUnit = str  # 'K' or 'C'
DistUnit = str  # 'km' or 'm'


def _to_kelvin(temp: ArrayLike, unit: TempUnit) -> ArrayLike:
    """Convert an ambient temperature to Kelvin for the internal grid."""
    if unit == 'K':
        return temp
    if unit == 'C':
        return np.asarray(temp) + 273.15
    raise ValueError(f"temp_unit must be 'K' or 'C', got {unit!r}")


def _to_km(dist: ArrayLike, unit: DistUnit) -> ArrayLike:
    """Convert a path length to kilometres for the internal k [km^-1]."""
    if unit == 'km':
        return dist
    if unit == 'm':
        return np.asarray(dist, dtype=float) / 1000.0
    raise ValueError(f"dist_unit must be 'km' or 'm', got {unit!r}")


def _scalar_ambient(value: ArrayLike, name: str) -> float:
    """
    Coerce an ambient parameter to a single float, rejecting per-pixel maps.

    The query reduction assumes one ambient state per frame (a single weather /
    radiosonde reading): the k-distribution is interpolated once and reused across
    all pixels. A per-pixel ambient *map* would force that interpolation at every
    pixel — an order-of-magnitude cost increase — so it is refused explicitly here
    rather than being silently collapsed to the first element.
    """
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(
            f"{name} must be a single ambient value (one reading per frame), got an "
            f"array of shape {arr.shape}. Per-pixel ambient maps are not supported: "
            f"the k-distribution would need interpolating at every pixel. Pass a "
            f"scalar; vary only the distance map per pixel."
        )
    return float(arr.reshape(-1)[0])


# International Standard Atmosphere (troposphere) constants for altitude→pressure.
STANDARD_MSL_PRES = 1013.25          # mbar, ISA mean-sea-level pressure
_ISA_T0 = 288.15                      # K, sea-level temperature
_ISA_LAPSE = 0.0065                   # K/m, tropospheric lapse rate
_ISA_EXP = 9.80665 * 0.0289644 / (8.314462618 * _ISA_LAPSE)   # g·M / (R·L) ≈ 5.2559


def altitude_to_pressure(altitude: float,
                         msl_pres: float = STANDARD_MSL_PRES) -> float:
    """
    Convert geometric altitude to pressure via the ISA barometric formula.

    Uses the International Standard Atmosphere tropospheric relation
    ``P = P₀·(1 − L·h/T₀)^(gM/RL)``, valid to ~11 km — well beyond the LUT's
    ~6000 m envelope. Supplying a *local* sea-level pressure (from a nearby weather
    report) via ``msl_pres`` corrects the largest source of error, the ±3 % daily
    variation in ``P₀``; the standard temperature profile is assumed otherwise.

    Parameters
    ----------
    altitude : float
        Geometric altitude above sea level [m].
    msl_pres : float, optional
        Local mean-sea-level pressure [mbar]. Default is the ISA standard 1013.25.

    Returns
    -------
    float
        Pressure [mbar].
    """
    h = _scalar_ambient(altitude, 'altitude')
    return float(msl_pres) * (1.0 - _ISA_LAPSE * h / _ISA_T0) ** _ISA_EXP


@dataclass
class AtmosLUT:
    """
    Reader for an RFM-generated atmospheric-correction LUT.

    Parameters
    ----------
    path : str or pathlib.Path
        Either an instrument name (e.g. ``'MMT-gasCam'``), resolved to a bundled or
        user LUT via :func:`atmcorr.registry.resolve_lut`, or an explicit path to a
        ``.npz`` LUT written by :func:`AtmosLUT.write`.

    Attributes
    ----------
    instrument : str
        Instrument name recorded in the file.
    filters : list of str
        Available filter names. For a broadband instrument this is length 1.
    temp, pres, rh : numpy.ndarray
        Grid axis vectors [K], [mbar], [%].
    gweight : numpy.ndarray
        k-distribution quadrature weights, shape ``(n_g,)``, summing to 1.
    wn_range : tuple of float
        Spectral range of the underlying RFM runs [cm^-1].
    default_pres : float
        Pressure assumed when a query omits ``pres`` [mbar].
    meta : dict
        Provenance recorded by the factory.

    Notes
    -----
    Ambient queries are clamped to the grid envelope rather than extrapolated:
    the k-distribution must stay non-negative, and field ambient conditions can
    stray slightly beyond the tabulated range. A warning is emitted on clamp.
    """

    path: Union[str, Path]

    def __post_init__(self) -> None:
        # Accept an instrument name ('MMT-gasCam') or an explicit .npz path.
        self.path = resolve_lut(self.path)
        with np.load(self.path, allow_pickle=False) as npz:
            version = str(npz['format_version'])
            if version != FORMAT_VERSION:
                extra = ''
                if version == '1':
                    extra = (" A version-1 LUT does not record how its bands "
                             "were weighted; if the instrument has filters in "
                             "front of a detector, this LUT is filter-only and "
                             "disagrees with a composite radiance. Rebuild with "
                             "build_instrument(..., detector=...).")
                warnings.warn(
                    f"LUT format version {version!r} != reader version "
                    f"{FORMAT_VERSION!r}; proceeding but fields may differ."
                    + extra,
                    RuntimeWarning,
                )
            self.instrument = str(npz['instrument'])
            self.wn_range = tuple(float(x) for x in npz['wn_range'])
            self.temp = npz['temp'].astype(float)
            self.pres = npz['pres'].astype(float)
            self.rh = npz['rh'].astype(float)
            self.gweight = npz['gweight'].astype(float)
            self.default_pres = float(npz['default_pres'])
            self.filters = [str(f) for f in npz['filters']]
            self.meta = json.loads(str(npz['meta'])) if 'meta' in npz else {}
            self._kdist = {
                name: npz[f'kdist__{name}'].astype(float) for name in self.filters
            }

        # Validate and build one interpolator per filter over (T, P, RH); the
        # trailing g-axis is carried through as vector-valued output.
        if not np.isclose(self.gweight.sum(), 1.0, atol=1e-6):
            raise ValueError(
                f"k-distribution weights sum to {self.gweight.sum():.6f}, expected 1."
            )
        self._interp: dict[str, RegularGridInterpolator] = {}
        for name, kd in self._kdist.items():
            expected = (self.temp.size, self.pres.size, self.rh.size, self.gweight.size)
            if kd.shape != expected:
                raise ValueError(
                    f"kdist__{name} shape {kd.shape} != expected {expected}."
                )
            if np.any(kd < 0):
                raise ValueError(f"kdist__{name} contains negative absorption coeffs.")
            self._interp[name] = RegularGridInterpolator(
                (self.temp, self.pres, self.rh), kd,
                bounds_error=False, fill_value=None,
            )

    # ------------------------------------------------------------------ helpers
    def _resolve_filter(self, filter: Optional[str]) -> str:
        """Return a valid filter name, defaulting to the sole filter if unique."""
        if filter is None:
            if len(self.filters) != 1:
                raise ValueError(
                    f"{self.instrument} has {len(self.filters)} filters "
                    f"{self.filters}; specify one via filter=..."
                )
            return self.filters[0]
        if filter not in self.filters:
            raise KeyError(f"Unknown filter {filter!r}; available: {self.filters}")
        return filter

    def _kvector(self, temp_k: float, pres: float, rh: float, filter: str) -> np.ndarray:
        """Interpolate the k-distribution to a scalar ambient state, clamped to grid."""
        clamped = (
            float(np.clip(temp_k, self.temp.min(), self.temp.max())),
            float(np.clip(pres, self.pres.min(), self.pres.max())),
            float(np.clip(rh, self.rh.min(), self.rh.max())),
        )
        if (clamped[0], clamped[1], clamped[2]) != (temp_k, pres, rh):
            warnings.warn(
                f"Ambient (T={temp_k:.1f} K, P={pres:.0f} mbar, RH={rh:.0f}%) "
                f"outside LUT envelope; clamped to {clamped}.",
                RuntimeWarning,
            )
        point = np.array([clamped], dtype=float)         # shape (1, 3)
        k_g = self._interp[filter](point)[0]             # shape (n_g,)
        return np.clip(k_g, 0.0, None)

    def _reduce_expsum(
        self, k_g: np.ndarray, n_terms: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compress a k-distribution vector to an ``n_terms`` exponential sum.

        The rates are a geometric ladder spanning the k-distribution; the weights
        are solved by non-negative least squares against the exact curve, which is
        stable (unlike a free non-linear exp-sum fit). Returns ``(w, k_nodes)``,
        each shape ``(n_terms,)``, with rates in km^-1.
        """
        k_lo = max(k_g.min(), 1e-6)
        k_hi = max(k_g.max(), k_lo * 10)
        k_nodes = np.geomspace(k_lo, k_hi, n_terms)
        # Fit weights on a distance grid that resolves the whole decay.
        d_km = np.linspace(0.0, -np.log(0.02) / k_lo, 400)
        target = (self.gweight[:, None] * np.exp(-k_g[:, None] * d_km[None, :])).sum(0)
        basis = np.exp(-k_nodes[None, :] * d_km[:, None])   # (n_d, n_terms)
        w, _ = nnls(basis, target)
        return w, k_nodes

    def _kvector_state(
        self,
        temp: ArrayLike,
        rh: ArrayLike,
        filter: Optional[str],
        pres: Optional[float],
        temp_unit: TempUnit,
        altitude: Optional[float] = None,
        msl_pres: float = STANDARD_MSL_PRES,
    ) -> tuple[str, np.ndarray]:
        """
        Resolve filter, guard/convert the (scalar) ambient state, and interpolate.

        Returns the resolved filter name and the ``(n_g,)`` k-distribution vector.
        Ambient parameters are coerced to scalars via :func:`_scalar_ambient`, which
        rejects per-pixel maps. Pressure may be given directly (``pres``) or as an
        ``altitude`` [m] converted through :func:`altitude_to_pressure`; supplying
        both is an error. If neither is given, ``default_pres`` is used.
        """
        name = self._resolve_filter(filter)
        temp_k = float(_to_kelvin(_scalar_ambient(temp, 'temp'), temp_unit))
        rh_val = _scalar_ambient(rh, 'rh')
        if altitude is not None:
            if pres is not None:
                raise ValueError("Give either pres or altitude, not both.")
            p = altitude_to_pressure(altitude, msl_pres)
        elif pres is not None:
            p = _scalar_ambient(pres, 'pres')
        else:
            p = self.default_pres
        return name, self._kvector(temp_k, p, rh_val, name)

    # -------------------------------------------------------------------- core
    def transmittance(
        self,
        temp: float,
        rh: float,
        dist: ArrayLike,
        filter: Optional[str] = None,
        pres: Optional[float] = None,
        temp_unit: TempUnit = 'K',
        dist_unit: DistUnit = 'km',
        altitude: Optional[float] = None,
        msl_pres: float = STANDARD_MSL_PRES,
    ) -> ArrayLike:
        """
        Band transmittance at fixed ambient state over a distance (map).

        Uses the exact closed form ``tau(d) = sum_g w_g exp(-k_g d)``. For a
        per-pixel distance map the curve is built once on a dense 1-D grid from the
        interpolated k-distribution and the map is then linearly interpolated
        against it — several-fold cheaper than evaluating the sum at every pixel,
        and exact to within a negligible interpolation error on the smooth curve.
        Small arrays and scalars are evaluated directly.

        Parameters
        ----------
        temp : float
            Ambient air temperature (unit set by ``temp_unit``).
        rh : float
            Relative humidity [%].
        dist : float or numpy.ndarray
            Path length(s); scalar or a per-pixel distance map (unit set by
            ``dist_unit``). NaNs (e.g. sky pixels) propagate as NaN.
        filter : str, optional
            Filter name; may be omitted for a single-filter instrument.
        pres : float, optional
            Ambient pressure [mbar]. Mutually exclusive with ``altitude``; if
            neither is given, the LUT's ``default_pres`` is used.
        temp_unit : {'K', 'C'}, optional
            Unit of ``temp``. Default ``'K'``.
        dist_unit : {'km', 'm'}, optional
            Unit of ``dist``. Default ``'km'``.
        altitude : float, optional
            Deployment altitude [m], converted to pressure via
            :func:`altitude_to_pressure` when ``pres`` is not supplied. Convenient
            when only altitude is known (a humidity sensor gives no pressure).
        msl_pres : float, optional
            Local mean-sea-level pressure [mbar] for the altitude conversion.
            Default is the ISA standard (1013.25).

        Returns
        -------
        float or numpy.ndarray
            Transmittance in [0, 1], same shape as ``dist``.
        """
        _, k_g = self._kvector_state(temp, rh, filter, pres, temp_unit, altitude, msl_pres)   # (n_g,)
        w = self.gweight
        d_km = np.asarray(_to_km(dist, dist_unit), dtype=float)

        if d_km.size <= _CURVE_POINTS:
            # Direct exact evaluation — cheap when there are few query points.
            flat = d_km.ravel()
            tau = (w[:, None] * np.exp(-k_g[:, None] * flat[None, :])).sum(0)
            tau = tau.reshape(d_km.shape)
        else:
            # Build tau(d) once on a dense grid, then interpolate the map. NaNs
            # (sky pixels) are excluded from the range and restored afterwards.
            finite = d_km[np.isfinite(d_km)]
            d_max = float(finite.max()) if finite.size else 1.0
            d_grid = np.linspace(0.0, d_max * (1.0 + 1e-6), _CURVE_POINTS)
            curve = (w[:, None] * np.exp(-k_g[:, None] * d_grid[None, :])).sum(0)
            tau = np.interp(d_km, d_grid, curve)
            tau[~np.isfinite(d_km)] = np.nan

        return float(tau) if np.ndim(dist) == 0 else tau

    def curve(
        self,
        temp: float,
        rh: float,
        filter: Optional[str] = None,
        pres: Optional[float] = None,
        dist: Optional[np.ndarray] = None,
        temp_unit: TempUnit = 'K',
        dist_unit: DistUnit = 'km',
        altitude: Optional[float] = None,
        msl_pres: float = STANDARD_MSL_PRES,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return a 1-D ``(dist, transmittance)`` curve at fixed ambient state.

        Convenience for plotting or for feeding an external fit. ``dist`` defaults
        to 200 points spanning 0 to a transmittance of ~0.05 (a physically useful
        range), expressed in ``dist_unit``.
        """
        _, k_g = self._kvector_state(temp, rh, filter, pres, temp_unit, altitude, msl_pres)
        if dist is None:
            # Span until the clearest channel reaches tau ~ 0.05.
            k_min = max(k_g.min(), 1e-6)
            d_max_km = -np.log(0.05) / k_min
            d_km = np.linspace(0.0, d_max_km, 200)
            dist_out = d_km if dist_unit == 'km' else d_km * 1000.0
        else:
            dist_out = np.asarray(dist, dtype=float)
            d_km = _to_km(dist_out, dist_unit)
        tau = (self.gweight[:, None] * np.exp(-k_g[:, None] * d_km[None, :])).sum(0)
        return dist_out, tau

    def expsum(
        self,
        temp: float,
        rh: float,
        n_terms: int = 4,
        filter: Optional[str] = None,
        pres: Optional[float] = None,
        temp_unit: TempUnit = 'K',
        altitude: Optional[float] = None,
        msl_pres: float = STANDARD_MSL_PRES,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Reduce the k-distribution to a compact ``n_terms`` exponential sum.

        Returns weights ``w`` and rates ``k`` [km^-1] such that
        ``tau(d) ~= sum_i w_i exp(-k_i d)`` with ``d`` in km. Useful when a
        downstream model wants a handful of coefficients rather than the reader.
        The rates are chosen as a geometric ladder spanning the k-distribution and
        the weights solved by non-negative least squares against the exact curve,
        which is stable (unlike a free non-linear exp-sum fit).

        Parameters
        ----------
        n_terms : int
            Number of exponential terms.

        Returns
        -------
        w, k : numpy.ndarray
            Weights (sum ~ 1) and rates [km^-1], each shape ``(n_terms,)``.
        """
        _, k_g = self._kvector_state(temp, rh, filter, pres, temp_unit, altitude, msl_pres)
        return self._reduce_expsum(k_g, n_terms)

    def as_callable(
        self,
        filter: Optional[str] = None,
        pres: Optional[float] = None,
        temp_unit: TempUnit = 'C',
        dist_unit: DistUnit = 'm',
        altitude: Optional[float] = None,
        msl_pres: float = STANDARD_MSL_PRES,
    ) -> Callable[[ArrayLike, ArrayLike, ArrayLike], ArrayLike]:
        """
        Build a ``tau(T_atm, RH, d)`` closure for a custom-atmosphere hook.

        The returned callable matches the signature expected by FLIR-style
        pipelines (``atm_model='custom'``). Defaults assume the pipeline carries
        **Celsius** temperatures and **metre** distances, as the FLIR model does;
        override ``temp_unit`` / ``dist_unit`` for other conventions. A fixed
        deployment altitude may be bound via ``altitude`` [m] (with optional local
        ``msl_pres``) instead of ``pres``.

        Examples
        --------
        >>> lut = AtmosLUT('flir_broadband_atmos_lut.npz')
        >>> tau = lut.as_callable(altitude=3200)   # tau(T_atm[C], RH[%], d[m]) at 3.2 km
        >>> raw_to_temperature(..., atm_model='custom', atm_tau=tau)
        """
        name = self._resolve_filter(filter)

        def tau(t_atm: ArrayLike, rh: ArrayLike, d: ArrayLike) -> ArrayLike:
            # t_atm / rh must be scalar (uniform ambient); a per-pixel map raises
            # in transmittance() rather than being silently collapsed. Only d
            # (the distance map) varies per pixel.
            return self.transmittance(
                t_atm, rh, d, filter=name, pres=pres,
                temp_unit=temp_unit, dist_unit=dist_unit,
                altitude=altitude, msl_pres=msl_pres,
            )

        return tau

    def __repr__(self) -> str:
        return (
            f"AtmosLUT(instrument={self.instrument!r}, filters={self.filters}, "
            f"grid={self.temp.size}x{self.pres.size}x{self.rh.size}, "
            f"n_g={self.gweight.size}, wn={self.wn_range[0]:.0f}-{self.wn_range[1]:.0f})"
        )

    # ------------------------------------------------------------------- writer
    @staticmethod
    def write(
        path: Union[str, Path],
        instrument: str,
        temp: np.ndarray,
        pres: np.ndarray,
        rh: np.ndarray,
        gweight: np.ndarray,
        kdist: dict[str, np.ndarray],
        wn_range: tuple[float, float],
        default_pres: float,
        meta: Optional[dict] = None,
    ) -> Path:
        """
        Write an atmospheric LUT in the canonical format (used by the factory).

        Parameters
        ----------
        path : str or pathlib.Path
            Output ``.npz`` path.
        instrument : str
            Instrument name.
        temp, pres, rh : numpy.ndarray
            Grid axis vectors [K], [mbar], [%].
        gweight : numpy.ndarray
            k-distribution quadrature weights, shape ``(n_g,)``, must sum to 1.
        kdist : dict of str -> numpy.ndarray
            Per-filter sorted absorption coefficients [km^-1], each of shape
            ``(n_temp, n_pres, n_rh, n_g)``.
        wn_range : tuple of float
            Spectral range of the underlying RFM runs [cm^-1].
        default_pres : float
            Pressure assumed when a query omits pressure [mbar].
        meta : dict, optional
            Provenance (RFM version, HITRAN file, reference path, date, ...).

        Returns
        -------
        pathlib.Path
            The path written.
        """
        temp = np.asarray(temp, float)
        pres = np.asarray(pres, float)
        rh = np.asarray(rh, float)
        gweight = np.asarray(gweight, float)
        if not np.isclose(gweight.sum(), 1.0, atol=1e-6):
            raise ValueError(f"gweight must sum to 1, got {gweight.sum():.6f}.")
        expected = (temp.size, pres.size, rh.size, gweight.size)
        arrays: dict[str, np.ndarray] = {}
        for name, kd in kdist.items():
            kd = np.asarray(kd, float)
            if kd.shape != expected:
                raise ValueError(
                    f"kdist[{name!r}] shape {kd.shape} != expected {expected}."
                )
            if np.any(kd < 0):
                raise ValueError(f"kdist[{name!r}] has negative coefficients.")
            arrays[f'kdist__{name}'] = kd

        path = Path(path)
        np.savez_compressed(
            path,
            format_version=FORMAT_VERSION,
            instrument=instrument,
            wn_range=np.asarray(wn_range, float),
            temp=temp, pres=pres, rh=rh, gweight=gweight,
            filters=np.array(list(kdist.keys())),
            default_pres=float(default_pres),
            meta=json.dumps(meta or {}),
            **arrays,
        )
        return path


if __name__ == '__main__':
    # Self-test with a synthetic LUT (no RFM needed): a grey-ish band whose
    # absorption grows with RH and temperature. Verifies the round-trip and the
    # exactness of the exp-sum reduction against the k-distribution.
    import tempfile

    n_g = 32
    g = (np.arange(n_g) + 0.5) / n_g
    gweight = np.full(n_g, 1.0 / n_g)
    temp = np.linspace(273.0, 313.0, 5)
    pres = np.array([950.0, 1013.0])
    rh = np.linspace(10.0, 90.0, 5)

    # k(nu) spread as a simple monotone ladder scaled by state (arbitrary, km^-1).
    base = np.geomspace(0.01, 3.0, n_g)
    T, P, R = np.meshgrid(temp, pres, rh, indexing='ij')
    scale = (0.5 + R / 100.0) * (0.8 + (T - 273.0) / 80.0)
    kd = scale[..., None] * base[None, None, None, :]

    with tempfile.TemporaryDirectory() as tmp:
        p = AtmosLUT.write(
            Path(tmp) / 'synthetic_atmos_lut.npz', instrument='synthetic',
            temp=temp, pres=pres, rh=rh, gweight=gweight,
            kdist={'band': kd}, wn_range=(700.0, 1400.0), default_pres=1013.0,
            meta={'note': 'self-test'},
        )
        lut = AtmosLUT(p)
        print(lut)

        # Scalar and array queries.
        print('tau(298 K, 55%, 2 km) =', lut.transmittance(298.0, 55.0, 2.0))
        dmap = np.array([[0.5, 1.0], [2.0, np.nan]])
        print('tau map (km):\n', lut.transmittance(298.0, 55.0, dmap))

        # exp-sum export (for handing coefficients to an external model).
        d, tau_exact = lut.curve(298.0, 55.0)
        w, k = lut.expsum(298.0, 55.0, n_terms=6)
        tau_fit = (w[None, :] * np.exp(-k[None, :] * d[:, None])).sum(1)
        print(f'exp-sum(6) export max abs err vs exact: {np.max(np.abs(tau_fit - tau_exact)):.2e}')

        # Large map (curve+interp path) must agree with a direct evaluation.
        big = np.random.uniform(0.1, 12.0, _CURVE_POINTS * 8)      # forces curve+interp
        tau_interp = lut.transmittance(298.0, 55.0, big)
        k0 = lut._kvector_state(298.0, 55.0, None, None, 'K')[1]
        tau_direct = (lut.gweight[:, None] * np.exp(-k0[:, None] * big[None, :])).sum(0)
        print(f'curve+interp vs direct max abs err    : {np.max(np.abs(tau_interp - tau_direct)):.2e}')

        # Altitude→pressure convenience: altitude=... must match the equivalent pres=.
        print('altitude_to_pressure(0/3000/6000 m) =',
              [round(altitude_to_pressure(h), 1) for h in (0, 3000, 6000)])
        p3000 = altitude_to_pressure(3000.0)
        t_alt = lut.transmittance(298.0, 55.0, 2.0, altitude=3000.0)
        t_pres = lut.transmittance(298.0, 55.0, 2.0, pres=p3000)
        print(f'altitude vs equiv-pres match: {abs(t_alt - t_pres):.2e}')
        try:
            lut.transmittance(298.0, 55.0, 2.0, pres=850.0, altitude=3000.0)
            print('altitude/pres guard: FAILED (both accepted)')
        except ValueError:
            print('altitude/pres guard: ok (both rejected)')

        # FLIR-style callable (Celsius, metres).
        tau_cb = lut.as_callable()
        print('callable tau(25 C, 55%, 2000 m) =', tau_cb(25.0, 55.0, 2000.0))

        # Guard: a per-pixel ambient MAP must raise; a scalar / size-1 array is fine.
        try:
            tau_cb(np.full(dmap.shape, 25.0), 55.0, 2000.0)
            print('guard: FAILED (map accepted)')
        except ValueError as exc:
            print(f'guard: ok, ambient map rejected ({str(exc).split(":")[0]})')
        print('size-1 array ambient ok       =',
              lut.transmittance(np.array([298.0]), np.array([55.0]), 2.0))
