"""
Plotting helpers for atmcorr LUTs and masters.

Two entry points, split by whether the wavenumber axis survives:

* :func:`plotLUT` — band transmittance from a per-instrument LUT. The wavenumber
  axis is already integrated away, so the plot explores the four remaining
  variables ``(temp, pres, rh, dist)``: name one or two as free axes and the rest
  are pinned. One free axis gives line plots, two give a heatmap / contour /
  surface.
* :func:`plotMaster` — the spectral view of a master OPT-spectra file. Wavenumber
  is always the x-axis; an instrument SRF can be overlaid to show which absorption
  features a filter actually samples.

Both return ``(fig, ax)`` and never call ``plt.show()``, so they compose into
larger figures.

Matplotlib is an optional dependency (``pip install atmcorr[plot]``); nothing in
``atmcorr`` imports this module unless you ask for it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib import colormaps
except ImportError as exc:                                   # pragma: no cover
    raise ImportError(
        "atmcorr.plotting needs matplotlib. Install it with "
        "`pip install atmcorr[plot]` (or `pip install matplotlib`)."
    ) from exc

from .lut import AtmosLUT

# Free-axis names → (LUT attribute holding the grid, axis label, unit)
_AXES = {
    'temp': ('temp', 'Air temperature', 'K'),
    'pres': ('pres', 'Pressure', 'mbar'),
    'rh': ('rh', 'Relative humidity', '%'),
    'dist': (None, 'Distance', 'm'),          # not a LUT grid axis; user-ranged
}

_DEFAULTS = {'temp': 288.0, 'pres': None, 'rh': 50.0, 'dist': 2000.0}

# Sequential map for ordered quantities (2-D views, families over numeric values);
# qualitative map for discrete series (filters, SRFs).
CMAP_SEQUENTIAL = 'cividis'
CMAP_CATEGORICAL = 'tab20'


def _series_colors(n: int, cmap_name: str) -> list:
    """
    Pick ``n`` distinguishable colors from a colormap.

    Qualitative maps (tab20 and friends) are indexed by entry and cycled, so
    neighbouring series stay distinct; continuous maps are sampled evenly.
    """
    cm = colormaps[cmap_name]
    base = getattr(cm, 'colors', None)
    if base is not None and len(base) <= 32:                  # qualitative
        return [base[i % len(base)] for i in range(n)]
    return [cm(i / max(n - 1, 1)) for i in range(n)]


def _as_lut(lut: Union[AtmosLUT, str, Path]) -> AtmosLUT:
    """Accept an AtmosLUT, an instrument name, or a path."""
    return lut if isinstance(lut, AtmosLUT) else AtmosLUT(lut)


def _axis_values(lut: AtmosLUT, name: str, n: int,
                 dist_max: float) -> np.ndarray:
    """Sample values along a free axis: LUT grid span, or 0→dist_max for distance."""
    if name == 'dist':
        return np.linspace(0.0, dist_max, n)
    attr, _, _ = _AXES[name]
    grid = getattr(lut, attr)
    return np.linspace(float(grid.min()), float(grid.max()), n)


def _label(name: str) -> str:
    _, lab, unit = _AXES[name]
    return f'{lab} [{unit}]'


def _tau_grid(lut: AtmosLUT, filt: str, free: dict, pinned: dict) -> np.ndarray:
    """
    Evaluate transmittance over 1 or 2 free axes with the rest pinned.

    ``free`` maps axis name → value array (1 or 2 entries); ``pinned`` maps the
    remaining axis names → scalars. Returns shape ``(n,)`` or ``(ny, nx)``.
    """
    names = list(free)
    if len(names) == 1:
        (xn,), xv = names, free[names[0]]
        out = np.empty(xv.size)
        for i, v in enumerate(xv):
            kw = {**pinned, xn: v}
            out[i] = lut.transmittance(kw['temp'], kw['rh'], kw['dist'],
                                       filter=filt, pres=kw['pres'], dist_unit='m')
        return out
    xn, yn = names
    xv, yv = free[xn], free[yn]
    out = np.empty((yv.size, xv.size))
    for j, yval in enumerate(yv):
        for i, xval in enumerate(xv):
            kw = {**pinned, xn: xval, yn: yval}
            out[j, i] = lut.transmittance(kw['temp'], kw['rh'], kw['dist'],
                                          filter=filt, pres=kw['pres'], dist_unit='m')
    return out


def plotLUT(
    lut: Union[AtmosLUT, str, Path],
    x: str = 'dist',
    y: Optional[str] = None,
    hue: Optional[str] = None,
    filter: Union[str, Sequence[str], None] = None,
    kind: str = 'auto',
    layout: Optional[str] = None,
    temp: float = 288.0,
    rh: float = 50.0,
    pres: Optional[float] = None,
    dist: float = 2000.0,
    dist_max: float = 10000.0,
    n: int = 60,
    hue_values: Optional[Sequence[float]] = None,
    cmap: str = CMAP_SEQUENTIAL,
    line_cmap: str = CMAP_CATEGORICAL,
    ax=None,
    figsize: Optional[tuple] = None,
):
    """
    Plot band transmittance from a LUT against one or two chosen variables.

    Parameters
    ----------
    lut : AtmosLUT, str, or pathlib.Path
        A loaded LUT, an instrument name, or a path to a ``.npz``.
    x : {'dist', 'temp', 'rh', 'pres'}, optional
        Free axis on the abscissa. Default ``'dist'``.
    y : {'dist', 'temp', 'rh', 'pres'}, optional
        Second free axis. If given, a 2-D view is drawn (see ``kind``); if None,
        a line plot.
    hue : {'dist', 'temp', 'rh', 'pres'}, optional
        Line-plot only: draw a family of coloured lines over this variable.
    filter : str, sequence of str, or None, optional
        Which filter(s) to plot. None means all filters in the LUT.
    kind : {'auto', 'line', 'heatmap', 'contour', 'surface'}, optional
        ``'auto'`` picks ``'line'`` when ``y`` is None, else ``'heatmap'``.
    layout : {'overlay', 'grid'}, optional
        With several filters: overlay them on one axes (line plots only) or draw
        a subplot grid. Defaults to overlay for lines, grid for 2-D views.
    temp, rh, pres, dist : float, optional
        Pinned values for whichever variables are not free [K], [%], [mbar], [m].
        ``pres=None`` uses the LUT's ``default_pres``.
    dist_max : float, optional
        Upper limit when distance is a free axis [m]. Default 10000.
    n : int, optional
        Samples per free axis. Default 60.
    hue_values : sequence of float, optional
        Explicit values for the ``hue`` family (default: 5 across its range).
    cmap : str, optional
        Sequential colormap for ordered quantities: 2-D views and ``hue``
        families. Default ``'cividis'``.
    line_cmap : str, optional
        Qualitative colormap for discrete series (one colour per filter when
        several are overlaid). Default ``'tab20'``.
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axes (single-panel cases only).
    figsize : tuple, optional
        Figure size; a sensible default is chosen per layout.

    Returns
    -------
    fig, ax : matplotlib figure and axes (or array of axes for a grid)
    """
    lut = _as_lut(lut)
    for name in (x, y, hue):
        if name is not None and name not in _AXES:
            raise ValueError(f"axis must be one of {sorted(_AXES)}, got {name!r}")
    if y is not None and hue is not None:
        raise ValueError("hue applies to line plots only; drop hue or y.")
    if x == y:
        raise ValueError("x and y must differ.")

    filters = ([filter] if isinstance(filter, str)
               else list(filter) if filter is not None else list(lut.filters))
    for f in filters:
        if f not in lut.filters:
            raise KeyError(f"Unknown filter {f!r}; available: {lut.filters}")

    if kind == 'auto':
        kind = 'line' if y is None else 'heatmap'
    if kind == 'line' and y is not None:
        raise ValueError("kind='line' cannot take a second axis y.")
    if kind in ('heatmap', 'contour', 'surface') and y is None:
        raise ValueError(f"kind={kind!r} needs a second axis y.")
    if layout is None:
        layout = 'overlay' if kind == 'line' else 'grid'
    if layout == 'overlay' and kind != 'line':
        raise ValueError("overlay layout is only available for line plots.")

    pinned_all = {'temp': temp, 'rh': rh,
                  'pres': lut.default_pres if pres is None else pres, 'dist': dist}
    free_names = [a for a in (x, y) if a is not None]
    xv = _axis_values(lut, x, n, dist_max)
    yv = _axis_values(lut, y, n, dist_max) if y is not None else None

    # --- panels ---------------------------------------------------------
    grid_mode = layout == 'grid' and len(filters) > 1
    if grid_mode:
        ncol = int(np.ceil(np.sqrt(len(filters))))
        nrow = int(np.ceil(len(filters) / ncol))
        share = {} if kind == 'surface' else {'sharex': True, 'sharey': True}
        subkw = {'subplot_kw': {'projection': '3d'}} if kind == 'surface' else {}
        fig, axes = plt.subplots(nrow, ncol, figsize=figsize or (3.0 * ncol, 2.7 * nrow),
                                 squeeze=False, **share, **subkw)
        axes_flat = axes.ravel()
        for extra in axes_flat[len(filters):]:
            extra.set_visible(False)
    else:
        if ax is not None:
            fig, axes_flat = ax.figure, np.array([ax])
        else:
            subkw = {'subplot_kw': {'projection': '3d'}} if kind == 'surface' else {}
            fig, single = plt.subplots(figsize=figsize or (6.4, 4.4), **subkw)
            axes_flat = np.array([single])
        axes = axes_flat

    cmap_obj = colormaps[cmap]
    filter_colors = _series_colors(len(filters), line_cmap)
    mappable = None

    for k, filt in enumerate(filters):
        cax = axes_flat[k if grid_mode else 0]

        if kind == 'line':
            if hue is not None:
                hv = (np.asarray(hue_values, float) if hue_values is not None
                      else _axis_values(lut, hue, 5, dist_max))
                # hue spans an ordered quantity → sequential colours.
                for m, hval in enumerate(hv):
                    pinned = {a: v for a, v in pinned_all.items() if a != x}
                    pinned[hue] = hval
                    tau = _tau_grid(lut, filt, {x: xv}, pinned)
                    cax.plot(xv, tau, color=cmap_obj(m / max(len(hv) - 1, 1)),
                             label=f'{hval:g}')
            else:
                pinned = {a: v for a, v in pinned_all.items() if a != x}
                tau = _tau_grid(lut, filt, {x: xv}, pinned)
                # filters are discrete series → qualitative colours.
                colr = filter_colors[k] if (not grid_mode and len(filters) > 1) else None
                cax.plot(xv, tau, color=colr,
                         label=filt if (not grid_mode and len(filters) > 1) else None)
            cax.set_ylim(0, 1.02)
            cax.grid(alpha=0.3)
        else:
            pinned = {a: v for a, v in pinned_all.items() if a not in free_names}
            tau = _tau_grid(lut, filt, {x: xv, y: yv}, pinned)
            if kind == 'heatmap':
                mappable = cax.pcolormesh(xv, yv, tau, cmap=cmap, vmin=0, vmax=1,
                                          shading='auto')
            elif kind == 'contour':
                mappable = cax.contourf(xv, yv, tau, levels=12, cmap=cmap, vmin=0, vmax=1)
                cax.contour(xv, yv, tau, levels=12, colors='k', linewidths=0.3, alpha=0.4)
            else:                                            # surface
                X, Y = np.meshgrid(xv, yv)
                mappable = cax.plot_surface(X, Y, tau, cmap=cmap, vmin=0, vmax=1,
                                            linewidth=0, antialiased=True)
                cax.set_zlabel('transmittance')
        if grid_mode:
            cax.set_title(filt, fontsize=9)

    # --- labels / legends / colorbars ------------------------------------
    pin_txt = ', '.join(
        f'{a}={pinned_all[a]:g}{_AXES[a][2]}'
        for a in ('temp', 'rh', 'pres', 'dist')
        if a not in free_names and a != hue
    )
    ylab = 'band transmittance' if kind == 'line' else _label(y)
    if grid_mode:
        # Shared axes: label only the outer edges to keep panels legible.
        nrow_, ncol_ = axes.shape
        for idx in range(len(filters)):
            r, c = divmod(idx, ncol_)
            cax = axes[r, c]
            last_row = (idx + ncol_) >= len(filters)
            if last_row or kind == 'surface':
                cax.set_xlabel(_label(x), fontsize=8)
            if c == 0 or kind == 'surface':
                cax.set_ylabel(ylab, fontsize=8)
            cax.tick_params(labelsize=7)
    else:
        axes_flat[0].set_xlabel(_label(x))
        axes_flat[0].set_ylabel(ylab)

    if kind == 'line':
        lead = axes_flat[0]
        if hue is not None:
            lead.legend(title=_label(hue), fontsize=7, title_fontsize=7)
        elif not grid_mode and len(filters) > 1:
            lead.legend(fontsize=7, ncol=2 if len(filters) > 6 else 1)
    elif mappable is not None and kind != 'surface':
        # Reserve space on the right so the bar never overlaps the panels.
        target = list(axes_flat[:len(filters)]) if grid_mode else [axes_flat[0]]
        fig.colorbar(mappable, ax=target, label='band transmittance',
                     fraction=0.046, pad=0.02)

    title = f'{lut.instrument}'
    if pin_txt:
        title += f'  ({pin_txt})'
    fig.suptitle(title, fontsize=10)
    if kind == 'surface':
        # 3-D axes clip their z-label under tight_layout/bbox_inches='tight'.
        fig.subplots_adjust(left=0.02, right=0.88, top=0.9, bottom=0.06)
    elif not (grid_mode and mappable is not None):
        fig.tight_layout()
    return fig, (axes if grid_mode else axes_flat[0])


def plotMaster(
    master: Union[str, Path, None] = None,
    temp: float = 288.0,
    rh: float = 50.0,
    pres: Optional[float] = None,
    dist: Union[float, Sequence[float]] = 2000.0,
    window: Optional[str] = None,
    srf: Union[str, Path, None] = None,
    quantity: str = 'transmittance',
    wl_axis: bool = True,
    ascending: str = 'wl',
    xlim: Optional[tuple] = None,
    cmap: str = CMAP_SEQUENTIAL,
    srf_cmap: str = CMAP_CATEGORICAL,
    ax=None,
    figsize: tuple = (10.0, 4.2),
):
    """
    Plot spectra from a master OPT-spectra file, optionally with an SRF overlay.

    Parameters
    ----------
    master : str or pathlib.Path, optional
        Master ``.npz``; resolved automatically (by ``window``) when omitted.
    temp, rh, pres : float, optional
        Ambient state to plot [K], [%], [mbar]. Snapped to the nearest grid node.
    dist : float or sequence of float, optional
        Path length(s) [m]. Several values draw a coloured family of curves.
    window : str, optional
        Spectral window ('LWIR'/'MWIR') used to resolve the master when ``master``
        is not given.
    srf : str or pathlib.Path, optional
        Instrument name or SRF CSV to shade behind the spectrum, showing which
        features the band actually samples.
    quantity : {'transmittance', 'optical_depth'}, optional
        What to draw. Default transmittance.
    wl_axis : bool, optional
        Add a secondary top axis in µm. Default True.
    ascending : {'wl', 'wn'}, optional
        Which spectral axis increases left→right. ``'wl'`` (default) runs
        wavelength up and wavenumber down — matching how the IR-imaging community
        reads spectra, and the classic IR-spectroscopy convention of descending
        cm⁻¹. Use ``'wn'`` for increasing wavenumber.
    xlim : tuple, optional
        Wavenumber limits [cm⁻¹].
    cmap : str, optional
        Sequential colormap for the distance family. Default ``'cividis'``.
    srf_cmap : str, optional
        Qualitative colormap used when several SRF bands are overlaid (a
        multi-filter instrument). Default ``'tab20'``.
    ax : matplotlib.axes.Axes, optional
        Draw into an existing axes.
    figsize : tuple, optional
        Figure size when creating a new figure.

    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    from .build import _load_master, _load_srf, _resolve_srf, _srf_weights
    from .registry import resolve_master, resolve_srf

    path = resolve_master(master, window=window) if master is None else Path(master)
    opt, wn, temps, press, rhs, stamp = _load_master(path)

    it = int(np.argmin(np.abs(temps - temp)))
    ip = int(np.argmin(np.abs(press - (press.max() if pres is None else pres))))
    ir = int(np.argmin(np.abs(rhs - rh)))
    k = opt[it, ip, ir]                                   # [km^-1] (L0 = 1 km)

    dists = np.atleast_1d(np.asarray(dist, dtype=float))
    fig, cax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=figsize)
    cmap_obj = colormaps[cmap]

    if srf is not None:
        srf_map = _resolve_srf(resolve_srf(srf))
        # One SRF reads best as neutral shading; several are distinct bands, so
        # give each a qualitative colour and label it.
        srf_colors = _series_colors(len(srf_map), srf_cmap)
        for i, (name, csv) in enumerate(srf_map.items()):
            w = _srf_weights(*_load_srf(csv), wn, name)
            single = len(srf_map) == 1
            cax.fill_between(wn, 0, w / w.max(),
                             color='0.75' if single else srf_colors[i],
                             alpha=0.45 if single else 0.35, lw=0,
                             label=f'SRF: {name}' if single else name, zorder=0)

    for i, d in enumerate(dists):
        y = np.exp(-k * d / 1000.0)
        if quantity == 'optical_depth':
            y = k * d / 1000.0
        cax.plot(wn, y, lw=0.6, color=cmap_obj(i / max(len(dists) - 1, 1)),
                 label=f'{d:g} m', zorder=2)

    if ascending not in ('wl', 'wn'):
        raise ValueError(f"ascending must be 'wl' or 'wn', got {ascending!r}")
    cax.set_xlabel('Wavenumber [cm$^{-1}$]')
    cax.set_ylabel('Transmittance' if quantity == 'transmittance' else 'Optical depth')
    lo, hi = xlim or (wn.min(), wn.max())
    # Wavelength is the inverse of wavenumber, so ascending µm means descending cm⁻¹.
    cax.set_xlim((hi, lo) if ascending == 'wl' else (lo, hi))
    if quantity == 'transmittance':
        cax.set_ylim(0, 1.02)
    cax.grid(alpha=0.3)
    if len(dists) > 1 or srf is not None:
        cax.legend(fontsize=7)

    if wl_axis:
        top = cax.secondary_xaxis('top', functions=(lambda v: 1e4 / np.maximum(v, 1e-9),
                                                    lambda v: 1e4 / np.maximum(v, 1e-9)))
        top.set_xlabel('Wavelength [µm]')

    fig.suptitle(f"master {stamp.get('window', '?')}  "
                 f"(temp={temps[it]:g} K, pres={press[ip]:g} mbar, rh={rhs[ir]:g} %)",
                 fontsize=10)
    fig.tight_layout()
    return fig, cax
