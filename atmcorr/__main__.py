"""
Command-line interface for atmcorr.

    python -m atmcorr list                         # instruments with a resolvable LUT
    python -m atmcorr fetch-master                 # download the master (or --source a local copy)
    python -m atmcorr build MyCam --srf MyCam.csv  # build a per-instrument LUT

``build`` and ``fetch-master`` are the generation side (they need the master and,
for reading SRF CSVs, the ``[build]`` extra); ``list`` and applying LUTs need only
the reader.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='atmcorr', description='Atmospheric-correction LUT tools.')
    sub = p.add_subparsers(dest='cmd', required=True)

    sub.add_parser('list', help='List instruments with a resolvable LUT (bundled + user dirs).')

    b = sub.add_parser('build', help='Build a per-instrument LUT from an SRF + the master.')
    b.add_argument('instrument',
                   help='Instrument name. Names the output; if --srf is omitted, resolves a bundled SRF.')
    b.add_argument('--srf', default=None,
                   help='SRF source: a CSV, a directory of CSVs, or omit to use a bundled SRF for <instrument>.')
    b.add_argument('--master', default=None,
                   help='Master OPT-spectra file (default: auto-select by SRF coverage).')
    b.add_argument('--window', default=None,
                   help='Force a spectral window (LWIR/MWIR) instead of auto-selecting by SRF.')
    b.add_argument('--out', default=None,
                   help='Output .npz path, or a directory (default: <instrument>_atmos_lut.npz in the CWD).')
    b.add_argument('--detector', default=None,
                   help="Detector the bands sit behind (instrument name or CSV). Give this when "
                        "--srf holds filter transmissions: a band's response is filter x detector, "
                        "not the filter alone.")
    b.add_argument('--n-g', type=int, default=32, help='k-distribution quadrature points (default 32).')
    b.add_argument('--default-pres', type=float, default=1013.0,
                   help='Pressure assumed when a query omits it [mbar] (default 1013).')

    f = sub.add_parser('fetch-master', help='Download (or copy) a window master OPT-spectra file.')
    f.add_argument('window', nargs='?', default='LWIR', help="Spectral window (default 'LWIR').")
    f.add_argument('--url', default=None, help='Download URL (default: MASTER_URLS[window]).')
    f.add_argument('--source', default=None, help='Copy from a local master file instead of downloading.')
    f.add_argument('--dest', default=None, help='Destination path (default: bundled master_<window>.npz).')
    return p


def _cmd_list() -> None:
    from .lut import AtmosLUT
    from .registry import available_luts, available_masters
    masters = available_masters()
    print(f'masters: {sorted(masters) or "none (run fetch-master)"}')
    names = available_luts()
    if not names:
        print('No LUTs found.')
        return
    for name in names:
        lut = AtmosLUT(name)
        print(f'{name}: {len(lut.filters)} filter(s), '
              f'grid {lut.temp.size}x{lut.pres.size}x{lut.rh.size}, '
              f'wn {lut.wn_range[0]:.0f}-{lut.wn_range[1]:.0f} cm-1')


def _cmd_build(args: argparse.Namespace) -> None:
    from .build import build_instrument
    out = args.out
    if out is not None and Path(out).is_dir():
        out = str(Path(out) / f'{args.instrument}_atmos_lut.npz')
    written = build_instrument(args.instrument, srf=args.srf, master_path=args.master,
                               window=args.window, out_path=out, n_g=args.n_g,
                               default_pres=args.default_pres, detector=args.detector)
    print(f'Wrote {written}')
    print('To resolve it by name, place it in $ATMCORR_LUT_DIR or ~/.local/share/atmcorr/luts/ ; '
          'otherwise load it by path with AtmosLUT("<path>").')


def _cmd_fetch_master(args: argparse.Namespace) -> None:
    from .registry import fetch_master
    dest = fetch_master(window=args.window, url=args.url, source=args.source, dest=args.dest)
    print(f'Master ({args.window}) at {dest} ({dest.stat().st_size / 1024 / 1024:.0f} MB)')


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)
    try:
        if args.cmd == 'list':
            _cmd_list()
        elif args.cmd == 'build':
            _cmd_build(args)
        elif args.cmd == 'fetch-master':
            _cmd_fetch_master(args)
    except (ValueError, FileNotFoundError, ImportError, KeyError) as exc:
        print(f'atmcorr {args.cmd}: error: {exc}', file=sys.stderr)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
