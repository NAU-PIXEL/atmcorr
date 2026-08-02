"""
Locate bundled and user-supplied atmcorr data, and fetch the master on demand.

Three kinds of data are resolved by name so callers need not know file paths:

* **LUTs** — per-instrument ``<name>_atmos_lut.npz`` files. Searched in the user
  override dir(s) then the bundled ``data/luts/``; ``AtmosLUT('MMT-gasCam')`` uses
  this. Small, shipped in the git repo.
* **SRFs** — per-instrument spectral-response directories under ``data/srf/``.
  Small, shipped in the git repo. Curated to **ground-based** instruments only.
* **Master** — the large optical-depth-spectra file used only for *generating*
  LUTs. Too big for the git tree (>100 MB), so it is distributed as a GitHub
  Release asset and fetched on demand via :func:`fetch_master`.

Search order for LUTs puts a user override dir first (``ATMCORR_LUT_DIR`` env var,
then ``~/.local/share/atmcorr/luts``) so a user can add or replace instruments
without touching the installed package.
"""

from __future__ import annotations

import json
import os
import shutil
import warnings
from pathlib import Path
from typing import Optional, Union

import numpy as np

PathLike = Union[str, Path]

_DATA_DIR = Path(__file__).resolve().parent / 'data'
BUNDLED_LUT_DIR = _DATA_DIR / 'luts'
BUNDLED_SRF_DIR = _DATA_DIR / 'srf'
# Masters are named by spectral window: master_<WINDOW>.npz (e.g. master_LWIR.npz,
# master_MWIR.npz). Each is generated only for its window; build_instrument picks the
# one whose wn_range covers a given SRF (see select_master_for_srf).
_MASTER_DIR = _DATA_DIR / 'master'
_MASTER_PREFIX = 'master_'

_LUT_SUFFIX = '_atmos_lut.npz'

# Masters are published as Release / Zenodo assets, keyed by window (set once published):
#   {'LWIR': "https://github.com/NAU-PIXEL/atmcorr/releases/download/master-lwir-v1/master_LWIR.npz",
#    'MWIR': "…/master_MWIR.npz"}
MASTER_URLS: dict = {}


def _user_lut_dirs() -> list[Path]:
    """User-override LUT directories, highest priority first."""
    dirs: list[Path] = []
    env = os.environ.get('ATMCORR_LUT_DIR')
    if env:
        dirs.append(Path(env).expanduser())
    dirs.append(Path.home() / '.local' / 'share' / 'atmcorr' / 'luts')
    return dirs


def available_luts() -> list[str]:
    """Instrument names for which a LUT can be resolved (user dirs + bundled)."""
    names: set[str] = set()
    for d in [*_user_lut_dirs(), BUNDLED_LUT_DIR]:
        if d.is_dir():
            for f in d.glob(f'*{_LUT_SUFFIX}'):
                names.add(f.name[: -len(_LUT_SUFFIX)])
    return sorted(names)


def resolve_lut(name_or_path: PathLike) -> Path:
    """
    Resolve a LUT by instrument name or explicit path.

    An existing ``.npz`` path is returned as-is; otherwise the argument is treated
    as an instrument name and ``<name>_atmos_lut.npz`` is searched for in the user
    override dir(s) then the bundled ``data/luts/``.
    """
    p = Path(name_or_path)
    if p.suffix == '.npz' and p.is_file():
        return p
    fname = p.name if p.name.endswith(_LUT_SUFFIX) else f'{name_or_path}{_LUT_SUFFIX}'
    for d in [*_user_lut_dirs(), BUNDLED_LUT_DIR]:
        cand = d / fname
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"No LUT found for {name_or_path!r}. Available: {available_luts()}. "
        f"Pass an explicit path, place a LUT in $ATMCORR_LUT_DIR, or generate one "
        f"with atmcorr.build_instrument()."
    )


def resolve_srf(name_or_path: PathLike) -> Path:
    """
    Resolve an SRF source by instrument name or explicit path.

    An existing path (file or directory) is returned as-is; otherwise the argument
    is treated as an instrument name and looked up under the bundled ``data/srf/``.
    """
    p = Path(name_or_path)
    if p.exists():
        return p
    cand = BUNDLED_SRF_DIR / str(name_or_path)
    if cand.is_dir():
        return cand
    available = sorted(d.name for d in BUNDLED_SRF_DIR.iterdir() if d.is_dir()) \
        if BUNDLED_SRF_DIR.is_dir() else []
    raise FileNotFoundError(
        f"No SRF found for {name_or_path!r}. Bundled instruments: {available}. "
        f"Pass an explicit CSV/directory path."
    )


def _master_user_dirs() -> list[Path]:
    """User-override master directories (highest priority first)."""
    dirs: list[Path] = []
    env = os.environ.get('ATMCORR_MASTER_DIR')
    if env:
        dirs.append(Path(env).expanduser())
    dirs.append(Path.home() / '.local' / 'share' / 'atmcorr' / 'master')
    return dirs


def _read_master_stamp(path: PathLike) -> dict:
    """Read only the settings stamp from a master (the small stamp, not `opt`)."""
    with np.load(path, allow_pickle=False) as d:
        return json.loads(str(d['stamp']))


def available_masters() -> dict:
    """
    Map spectral window -> master path (user dirs first, then bundled).

    Each ``master_*.npz`` self-identifies via its stamp (``window``, ``wn_start/stop``);
    only the stamp is read, not the large ``opt`` array. An explicit ``$ATMCORR_MASTER``
    file is included under its stamped window and takes top priority.
    """
    found: dict = {}
    paths: list[Path] = []
    env = os.environ.get('ATMCORR_MASTER')
    if env:
        paths.append(Path(env).expanduser())
    for d in [*_master_user_dirs(), _MASTER_DIR]:
        if d.is_dir():
            paths.extend(sorted(d.glob(f'{_MASTER_PREFIX}*.npz')))
    for p in paths:
        if not p.is_file():
            continue
        try:
            window = _read_master_stamp(p).get('window')
        except Exception:
            continue
        if window and window not in found:      # first hit wins (user dirs first)
            found[window] = p
    return found


def master_for_window(window: str) -> Path:
    """Resolve the master for a spectral window, or raise pointing at fetch_master."""
    masters = available_masters()
    if window in masters:
        return masters[window]
    raise FileNotFoundError(
        f"No master for window {window!r} (available: {sorted(masters)}). Masters are "
        f"large Release/Zenodo assets, not in the git tree — run "
        f"atmcorr.fetch_master(window={window!r}) or pass master_path=."
    )


def select_master_for_srf(srf_wn, srf_T, coverage_threshold: float = 0.95) -> Path:
    """
    Pick the master whose spectral range best covers an SRF's response.

    Coverage = fraction of the (positive) SRF response falling inside a master's
    ``[wn_start, wn_stop]``. The highest-coverage master is returned; a warning is
    raised if even the best covers less than ``coverage_threshold`` — the SRF may
    straddle the gap between windows, or need a master that doesn't exist yet.
    """
    masters = available_masters()
    if not masters:
        raise FileNotFoundError(
            "No masters available. Run atmcorr.fetch_master(window=...) or pass "
            "master_path=/window=.")
    wn = np.asarray(srf_wn, dtype=float)
    w = np.clip(np.asarray(srf_T, dtype=float), 0.0, None)
    total = w.sum()
    scored = []
    for window, path in masters.items():
        st = _read_master_stamp(path)
        lo, hi = float(st['wn_start']), float(st['wn_stop'])
        cov = float(w[(wn >= lo) & (wn <= hi)].sum() / total) if total > 0 else 0.0
        scored.append((cov, window, path, (lo, hi)))
    scored.sort(key=lambda s: s[0], reverse=True)
    cov, window, path, _ = scored[0]
    if cov < coverage_threshold:
        ranges = ', '.join(f"{w_}[{r[0]:.0f}-{r[1]:.0f}]" for _, w_, _, r in scored)
        warnings.warn(
            f"Best master {window!r} covers only {cov:.0%} of the SRF response "
            f"(windows: {ranges} cm⁻¹). The SRF may straddle the gap between windows "
            f"or need a master that doesn't exist yet; pass window=/master_path= to "
            f"override, or split the instrument into per-window LUTs.", RuntimeWarning)
    return path


def resolve_master(master_path: Optional[PathLike] = None,
                   window: Optional[str] = None) -> Path:
    """
    Resolve a master by explicit path, by window, or (if unambiguous) the only one.

    For SRF-driven auto-selection use :func:`select_master_for_srf`.
    """
    if master_path is not None:
        return Path(master_path)
    if window is not None:
        return master_for_window(window)
    masters = available_masters()
    if len(masters) == 1:
        return next(iter(masters.values()))
    if not masters:
        raise FileNotFoundError(
            "No masters available. Run atmcorr.fetch_master(window=...) or pass "
            "master_path=/window=.")
    raise ValueError(
        f"Multiple masters available ({sorted(masters)}); specify window= or "
        f"master_path=, or use select_master_for_srf().")


def fetch_master(window: str = 'LWIR',
                 url: Optional[str] = None,
                 source: Optional[PathLike] = None,
                 dest: Optional[PathLike] = None) -> Path:
    """
    Place a window's master into the bundled data dir (download it, or copy a file).

    Parameters
    ----------
    window : str
        Spectral window to fetch (default 'LWIR'). Sets the default URL
        (``MASTER_URLS[window]``) and destination filename (``master_<window>.npz``).
    url : str, optional
        Download URL. Defaults to ``MASTER_URLS[window]``.
    source : str or pathlib.Path, optional
        Copy from this local file instead of downloading (takes precedence over url).
    dest : str or pathlib.Path, optional
        Destination path. Defaults to the bundled ``master_<window>.npz``.

    Returns
    -------
    pathlib.Path
        The master path now on disk.
    """
    dest = Path(dest) if dest is not None else _MASTER_DIR / f'{_MASTER_PREFIX}{window}.npz'
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source is not None:
        shutil.copyfile(Path(source), dest)
        return dest
    url = url or MASTER_URLS.get(window)
    if not url:
        raise ValueError(
            f"No URL for window {window!r} (MASTER_URLS not set / no Release published). "
            f"Pass source=<local master path> to copy one in, or url=<download url>.")
    import urllib.request
    urllib.request.urlretrieve(url, dest)
    return dest
