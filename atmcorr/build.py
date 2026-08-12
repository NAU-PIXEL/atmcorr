"""
Stage 2 of the atmospheric-correction LUT factory — RFM-free.

Given a **master OPT-spectra file** (produced by RFMwrapper Stage 1) and an
instrument's spectral response function (SRF), build the portable per-instrument
``<instrument>_atmos_lut.npz`` consumed by :class:`atmcorr.AtmosLUT`.

This stage needs no RFM — only NumPy / SciPy / pandas and the master data file.
It resamples each SRF onto the master's wavenumber grid, then collapses the
per-node absorption spectrum ``k(ν) = OPT(ν)/L0`` into an ``n_g``-point
k-distribution (sorted, SRF-weighted, uniform-g quadrature). The k-distribution
is exact for a homogeneous path, so the resulting LUT reproduces transmittance at
any distance from a handful of coefficients per node.

The master is treated as an external data artifact (like a HITRAN binary): it is
passed in by path, never bundled with this package.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Union

import numpy as np

from .lut import AtmosLUT
from .registry import resolve_master, resolve_srf, select_master_for_srf


def _load_master(master_path: Union[str, Path]) -> tuple:
    """Load a master OPT-spectra file and its settings stamp."""
    with np.load(master_path, allow_pickle=False) as d:
        opt = d['opt']                       # (nT, nP, nR, nWn) float32, optical depth at L0
        wn = d['wn'].astype(np.float64)
        temp = d['temp'].astype(np.float64)
        pres = d['pres'].astype(np.float64)
        rh = d['rh'].astype(np.float64)
        stamp = json.loads(str(d['stamp']))
    return opt, wn, temp, pres, rh, stamp


def _load_srf(csv_path: Union[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    """
    Read one SRF CSV → (wn ascending [cm⁻¹], T).

    Accepts a ``wn`` [cm⁻¹] column or a ``wl`` [µm] column (converted via
    ``wn = 1e4 / wl``); the response column ``T`` may be any positive scale (it is
    normalised downstream). Column names are matched case-insensitively.
    """
    try:
        import pandas as pd
    except ImportError as exc:                       # pragma: no cover
        raise ImportError(
            "Reading SRF CSVs needs pandas (generation only). Install with "
            "`pip install atmcorr[build]`."
        ) from exc
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    if 'wn' in cols:
        wn = df[cols['wn']].to_numpy(dtype=float)
    elif 'wl' in cols:
        wn = 1e4 / df[cols['wl']].to_numpy(dtype=float)
    else:
        raise ValueError(
            f"{csv_path}: SRF needs a 'wn' [cm⁻¹] or 'wl' [µm] column; "
            f"got {list(df.columns)}."
        )
    if 't' not in cols:
        raise ValueError(
            f"{csv_path}: SRF needs a 'T' response column; got {list(df.columns)}."
        )
    T = df[cols['t']].to_numpy(dtype=float)
    order = np.argsort(wn)
    return wn[order], T[order]


def _srf_weights(srf_wn: np.ndarray, srf_T: np.ndarray,
                 master_wn: np.ndarray, name: str,
                 detector: Union[tuple, None] = None) -> np.ndarray:
    """
    Resample an SRF onto the master grid and normalise to unit sum.

    Response is linearly interpolated onto ``master_wn`` and zeroed outside the
    SRF's own support. A warning is raised if the SRF carries >1 % response beyond
    the master's wavenumber range (that part is clipped).

    Parameters
    ----------
    srf_wn, srf_T : numpy.ndarray
        The band's own response, ascending in wavenumber.
    master_wn : numpy.ndarray
        Master wavenumber grid.
    name : str
        Band name, for diagnostics.
    detector : tuple of numpy.ndarray, optional
        ``(wn, response)`` for the detector the band sits behind. When given the
        weights become the **product** of the two: light passes through the
        filter *and* the detector, so a band's response is neither alone. The
        filter's own support still bounds the result, since the product is zero
        outside it.

        Mirrors ``IRViewer``'s ``resolve_srf`` product form
        (``'<set>/<filter>*<detector>'``); the two packages must weight bands
        identically or a retrieval disagrees with the radiance it was given.
    """
    lo, hi = master_wn.min(), master_wn.max()
    sig = srf_T > 0.01 * srf_T.max()
    if srf_wn[sig].min() < lo - 1e-6 or srf_wn[sig].max() > hi + 1e-6:
        warnings.warn(
            f"SRF {name!r} has >1% response outside the master range "
            f"[{lo:.0f}, {hi:.0f}] cm⁻¹; the out-of-range part is clipped.",
            RuntimeWarning,
        )
    w = np.interp(master_wn, srf_wn, srf_T, left=0.0, right=0.0)
    w = np.clip(w, 0.0, None)
    if detector is not None:
        det_wn, det_T = detector
        d = np.clip(np.interp(master_wn, det_wn, det_T, left=0.0, right=0.0),
                    0.0, None)
        w = w * d
    total = w.sum()
    if total <= 0:
        raise ValueError(
            f"SRF {name!r} has no overlap with the master range "
            f"[{lo:.0f}, {hi:.0f}] cm⁻¹"
            + (" once multiplied by the detector response" if detector else "")
            + "."
        )
    return w / total


def _build_kdists(k: np.ndarray, weights: dict[str, np.ndarray],
                  n_g: int) -> dict[str, np.ndarray]:
    """
    Collapse per-node absorption spectra into per-filter k-distributions.

    The sort order of ``k(ν)`` is filter-independent, so it is computed once per
    node and reused across all filters; only the SRF-weighted CDF differs. For
    each node and filter, ``k`` is sampled at the uniform-g quadrature points
    ``g = (i + ½)/n_g`` of the SRF-weighted cumulative distribution.

    Parameters
    ----------
    k : numpy.ndarray
        Absorption coefficients, shape ``(nT, nP, nR, nWn)`` [km⁻¹].
    weights : dict of str -> numpy.ndarray
        Per-filter normalised SRF weights, each shape ``(nWn,)``.
    n_g : int
        Number of quadrature points.

    Returns
    -------
    dict of str -> numpy.ndarray
        Per-filter k-distributions, each ``(nT, nP, nR, n_g)`` float32.
    """
    shape = k.shape[:-1]
    n_wn = k.shape[-1]
    kf = k.reshape(-1, n_wn)
    n_nodes = kf.shape[0]
    g_q = (np.arange(n_g) + 0.5) / n_g
    out = {name: np.empty((n_nodes, n_g), dtype=np.float64) for name in weights}

    for i in range(n_nodes):
        order = np.argsort(kf[i])
        k_sorted = kf[i][order]
        for name, w in weights.items():
            cdf = np.cumsum(w[order])          # SRF-weighted CDF, 0..1
            out[name][i] = np.interp(g_q, cdf, k_sorted)

    return {name: v.reshape(*shape, n_g).astype(np.float32)
            for name, v in out.items()}


def _resolve_srf(srf: Union[str, Path, dict]) -> dict:
    """Normalise the ``srf`` argument to a ``{filter_name: csv_path}`` mapping."""
    if isinstance(srf, dict):
        return srf
    p = Path(srf)
    if p.is_dir():
        files = sorted(p.glob('*.csv'))
        if not files:
            raise ValueError(f"No .csv SRF files found in directory {p}.")
        return {f.stem: f for f in files}
    if p.is_file():
        return {p.stem: p}
    raise FileNotFoundError(f"SRF path {p} is neither a file nor a directory.")


def tophat_srf(
    wl_lo: float,
    wl_hi: float,
    out_path: Union[str, Path, None] = None,
    edge_um: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a flat top-hat SRF over a wavelength band, as a generic stand-in.

    T = 1 within ``[wl_lo, wl_hi]`` µm and drops to 0 within ``edge_um`` just
    outside. Useful when a camera's real spectral response is unavailable — but a
    top-hat spanning too wide a band is pessimistic (it includes the opaque window
    edges that a real, edge-tapered detector responds to only weakly), so keep the
    band to the detector's usable core (e.g. 8–12 µm for LWIR).

    Parameters
    ----------
    wl_lo, wl_hi : float
        Band limits [µm], ``wl_lo < wl_hi``.
    out_path : str or pathlib.Path, optional
        If given, also write a ``wn,wl,T`` CSV there (the SRF file format).
    edge_um : float, optional
        Width of the 1→0 transition just outside the band [µm]. Default 0.01.

    Returns
    -------
    wn, T : numpy.ndarray
        Wavenumbers [cm⁻¹] ascending and the top-hat response.
    """
    if not wl_lo < wl_hi:
        raise ValueError(f"need wl_lo < wl_hi, got {wl_lo} !< {wl_hi}.")
    wl = np.array([wl_hi + edge_um, wl_hi, wl_lo, wl_lo - edge_um])
    T = np.array([0.0, 1.0, 1.0, 0.0])
    wn = 1e4 / wl
    order = np.argsort(wn)
    wn, wl, T = wn[order], wl[order], T[order]
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = ['wn,wl,T'] + [f'{a:.4f},{b:.4f},{c:.1f}' for a, b, c in zip(wn, wl, T)]
        out_path.write_text('\n'.join(rows) + '\n')
    return wn, T


def build_instrument(
    instrument: str,
    srf: Union[str, Path, dict, None] = None,
    master_path: Union[str, Path, None] = None,
    window: Union[str, None] = None,
    out_path: Union[str, Path, None] = None,
    n_g: int = 32,
    default_pres: float = 1013.0,
    detector: Union[str, Path, None] = None,
) -> Path:
    """
    Build a per-instrument atmospheric-correction LUT from a master + SRF(s).

    Parameters
    ----------
    instrument : str
        Instrument name recorded in the LUT. Also used to resolve ``srf`` and
        name the output when those are omitted.
    srf : str, pathlib.Path, dict, or None
        SRF source: a single CSV (one band), a directory of CSVs (one band per
        file, named by filename stem), or a ``{name: csv_path}`` mapping. If None,
        the bundled SRF directory for ``instrument`` is used.
    master_path : str or pathlib.Path, optional
        Master OPT-spectra file. If None (and ``window`` is None), the master is
        auto-selected as the one whose spectral range best covers the SRF (see
        :func:`atmcorr.registry.select_master_for_srf`).
    window : str, optional
        Force a spectral window ('LWIR' / 'MWIR' / …) instead of auto-selecting.
        Ignored if ``master_path`` is given.
    out_path : str or pathlib.Path, optional
        Destination ``.npz``. Defaults to ``<instrument>_atmos_lut.npz`` in the
        current directory.
    n_g : int, optional
        k-distribution quadrature points. Default 32.
    default_pres : float, optional
        Pressure the reader assumes when a query omits it [mbar]. Default 1013.
    detector : str or pathlib.Path, optional
        Detector response the bands sit behind, as an instrument name or a CSV
        path. When given, every band's weights become ``filter × detector``,
        which is the response the light actually sees — a filter transmission
        alone is not a band's response.

        Recorded in the LUT metadata, and ``format_version`` becomes 2. A
        version-1 LUT is filter-only; the two are otherwise indistinguishable,
        which is why the field exists.

    Returns
    -------
    pathlib.Path
        The LUT path written.
    """
    if srf is None:
        srf = resolve_srf(instrument)
    elif isinstance(srf, (str, Path)):
        srf = resolve_srf(srf)
    srf_map = _resolve_srf(srf)
    loaded = {name: _load_srf(csv) for name, csv in srf_map.items()}   # (wn, T) per band

    # Pick the master: explicit path / window, else auto-select by SRF coverage.
    if master_path is not None or window is not None:
        master_path = resolve_master(master_path, window)
    else:
        all_wn = np.concatenate([swn for swn, _ in loaded.values()])
        all_t = np.concatenate([sT for _, sT in loaded.values()])
        master_path = select_master_for_srf(all_wn, all_t)

    opt, wn, temp, pres, rh, stamp = _load_master(master_path)
    l0 = float(stamp['L0_km'])
    wn_range = (float(stamp['wn_start']), float(stamp['wn_stop']))
    # k(ν) = OPT / L0, kept float32 to bound memory (L0 = 1 km ⇒ k == OPT).
    k = opt.astype(np.float32) / np.float32(l0)

    # The detector the bands sit behind, if any. Resolved the same way as an
    # SRF, so it may be a bundled instrument name or an explicit CSV.
    det_curve = None
    det_file = None
    if detector is not None:
        det_path = resolve_srf(detector)
        if det_path.is_dir():
            csvs = sorted(det_path.glob('*.csv'))
            if len(csvs) != 1:
                raise ValueError(
                    f"detector {detector!r} resolves to {det_path}, which holds "
                    f"{len(csvs)} CSVs; a detector is one curve, so name the "
                    f"file explicitly.")
            det_path = csvs[0]
        det_curve = _load_srf(det_path)
        det_file = str(det_path)

    weights = {
        name: _srf_weights(swn, sT, wn, name, detector=det_curve)
        for name, (swn, sT) in loaded.items()
    }
    kdist = _build_kdists(k, weights, n_g)

    gweight = np.full(n_g, 1.0 / n_g)
    meta = {
        'source': 'atmcorr.build_instrument',
        'n_g': n_g,
        'window': stamp.get('window'),
        'srf_files': {name: str(csv) for name, csv in srf_map.items()},
        # How the bands were weighted. Without this a composite LUT and a
        # filter-only one are indistinguishable — same keys, same shapes,
        # different meaning — which is worse than a name collision because
        # nothing surfaces it.
        # 'as-supplied' rather than 'filter-only': for a single-curve instrument
        # such as FLIR-microbolometer the CSV already *is* the whole response,
        # so there is nothing to compose and nothing missing.
        'weighting': 'composite' if det_curve is not None else 'as-supplied',
        'detector': str(detector) if detector is not None else None,
        'detector_file': det_file,
        'master_stamp': stamp,
    }
    if out_path is None:
        out_path = Path(f'{instrument}_atmos_lut.npz')
    return AtmosLUT.write(
        out_path, instrument, temp, pres, rh, gweight, kdist,
        wn_range, default_pres, meta=meta,
    )
