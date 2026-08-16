"""
atmcorr — portable atmospheric-correction lookup tables for thermal imaging.

RFM-free reader and Stage-2 generator for per-instrument atmospheric-correction
LUTs. The expensive line-by-line master (OPT spectra) is produced separately by
RFMwrapper; this package turns a master + an instrument SRF into a compact LUT
and reads it back to return band transmittance for a per-pixel distance map.

Typical use
-----------
Author a LUT (needs a master data file, one-off)::

    from atmcorr import build_instrument
    build_instrument('MyCam', srf='path/to/MyCam.csv')   # master auto-resolved

When the SRFs are filter transmissions rather than whole-band responses, give the
detector they sit behind so the weights become ``filter x detector``::

    build_instrument('MMT-gasCam', detector='FLIR-microbolometer')

Apply a LUT (imaging side, no master needed)::

    from atmcorr import AtmosLUT
    lut = AtmosLUT('MyCam')                               # by name, or an explicit .npz path
    tau = lut.transmittance(temp=25.0, rh=55.0, dist=distance_map_m,
                            temp_unit='C', dist_unit='m')
"""

from .lut import AtmosLUT, altitude_to_pressure, FORMAT_VERSION
from .build import LINESHAPES, build_instrument, lineshape_srf, tophat_srf
from .registry import (
    available_luts, available_masters, fetch_master, resolve_lut, resolve_srf,
    select_master_for_srf,
)

__all__ = [
    'AtmosLUT', 'LINESHAPES', 'build_instrument', 'lineshape_srf',
    'tophat_srf', 'altitude_to_pressure',
    'available_luts', 'available_masters', 'fetch_master', 'resolve_lut',
    'resolve_srf', 'select_master_for_srf', 'plotLUT', 'plotMaster',
    'FORMAT_VERSION',
]
__version__ = '0.3.0'

# Plotting is optional (needs matplotlib): resolve atmcorr.plotLUT / plotMaster
# lazily so importing atmcorr never requires matplotlib.
_LAZY = {'plotLUT', 'plotMaster'}


def __getattr__(name):
    if name in _LAZY:
        from . import plotting
        return getattr(plotting, name)
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')


def __dir__():
    return sorted(set(globals()) | _LAZY)
