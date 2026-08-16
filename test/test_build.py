"""
LUT generation: band weighting, lineshapes, and the version-2 metadata.

The hazard this suite exists for is silent. A filter-only LUT and a composite one
have the same keys, the same shapes and the same dtypes — they differ only in
what the numbers mean, so nothing surfaces a mismatch at the point of use. That
is why ``format_version`` moved to 2 and the metadata gained ``weighting``,
``detector`` and ``filter_centroids_um``; these tests hold those in place.

Tests needing the master OPT file skip when it is absent: it is a 163 MB Release
asset, not part of the git tree.
"""

from __future__ import annotations

import numpy as np
import pytest

from atmcorr import AtmosLUT, FORMAT_VERSION, build_instrument, lineshape_srf
from atmcorr.build import LINESHAPES, _centroid_um, _srf_weights
from atmcorr.registry import available_masters, resolve_srf


def _master_or_skip(window: str = 'LWIR'):
    """The master for *window*, or skip — it is a Release asset, not in git."""
    masters = available_masters()
    if window not in masters:
        pytest.skip(f"no {window} master on this machine; fetch_master() first")
    return masters[window]


class TestLineshapes:
    """Asserting a band response for an instrument that cannot supply one."""

    @pytest.mark.parametrize('shape', sorted(LINESHAPES))
    def test_the_fwhm_means_the_same_for_every_shape(self, shape):
        # Every lineshape is defined on the same half-width units, which is what
        # makes a shape comparison at fixed FWHM meaningful rather than a
        # comparison of widths in disguise.
        x = np.array([-0.5, 0.0, 0.5])
        response = LINESHAPES[shape](x)
        assert response[1] == pytest.approx(1.0)
        assert response[0] == pytest.approx(response[2])
        if shape != 'tophat':          # a boxcar is 1 right up to its edge
            assert response[0] == pytest.approx(0.5, abs=0.02)

    @pytest.mark.parametrize('shape', sorted(LINESHAPES))
    def test_it_is_centred_where_asked(self, shape, tmp_path):
        wn, response = lineshape_srf(10.0, 0.3, shape=shape)
        assert _centroid_um(wn, response) == pytest.approx(10.0, abs=0.02)

    def test_wavenumber_and_wavelength_domains_differ(self):
        # A Fabry-Perot is symmetric in wavenumber; building the shape on the
        # wrong axis skews the band, and the skew grows with band width.
        wl_wn, t_wn = lineshape_srf(13.0, 24.0, domain='wn')
        wl_wl, t_wl = lineshape_srf(13.0, 13.0 ** 2 * 24.0 / 1e4, domain='wl')
        # Same nominal width, but the samples do not coincide.
        assert not np.allclose(_centroid_um(wl_wn, t_wn),
                               _centroid_um(wl_wl, t_wl), atol=1e-6)

    def test_a_wavenumber_band_widens_with_wavelength(self):
        # Constant resolution in wavenumber is a width in um that scales as
        # lambda^2 -- the R100's 24 cm-1 is 0.12 um at 7 um and 0.41 at 13.
        widths = []
        for centre in (7.0, 13.0):
            wn, response = lineshape_srf(centre, 23.74, domain='wn')
            wl = 1e4 / wn
            half = wl[response >= 0.5]
            widths.append(half.max() - half.min())
        assert widths[1] / widths[0] == pytest.approx((13.0 / 7.0) ** 2, rel=0.05)

    def test_it_writes_a_readable_csv(self, tmp_path):
        path = tmp_path / 'band00.csv'
        lineshape_srf(9.0, 0.2, out_path=path)
        header = path.read_text().splitlines()[0]
        assert header == 'wn,wl,T'

    @pytest.mark.parametrize('kwargs, match', [
        ({'shape': 'airy'}, 'unknown lineshape'),
        ({'domain': 'freq'}, "domain must be"),
        ({'fwhm': -1.0}, 'fwhm must be positive'),
        ({'centre_um': 0.0}, 'centre_um must be positive')])
    def test_bad_parameters_are_refused(self, kwargs, match):
        call = {'centre_um': 10.0, 'fwhm': 0.2}
        call.update(kwargs)
        with pytest.raises(ValueError, match=match):
            lineshape_srf(**call)


class TestWeighting:
    """Composed against as-supplied — the distinction with no nominal symptom."""

    def _grid(self) -> np.ndarray:
        return np.linspace(700.0, 1450.0, 4001)

    def test_weights_are_normalised(self):
        wn, response = lineshape_srf(10.0, 0.3)
        weights = _srf_weights(wn, response, self._grid(), 'band')
        assert weights.sum() == pytest.approx(1.0)

    def test_a_detector_changes_the_weights(self):
        # The whole point: filter x detector is not the filter alone.
        grid = self._grid()
        wn, response = lineshape_srf(10.0, 0.6)
        detector_wn = np.linspace(700.0, 1450.0, 200)
        # A detector that falls off toward long wavelength (low wavenumber).
        detector = (detector_wn - 700.0) / 750.0
        plain = _srf_weights(wn, response, grid, 'band')
        composed = _srf_weights(wn, response, grid, 'band',
                                detector=(detector_wn, detector))
        assert not np.allclose(plain, composed)
        # Both still normalised, so the difference is shape, not scale.
        assert composed.sum() == pytest.approx(1.0)

    def test_a_flat_detector_changes_nothing(self):
        grid = self._grid()
        wn, response = lineshape_srf(10.0, 0.6)
        flat = (np.array([700.0, 1450.0]), np.array([1.0, 1.0]))
        assert _srf_weights(wn, response, grid, 'band') == pytest.approx(
            _srf_weights(wn, response, grid, 'band', detector=flat))

    def test_the_composite_centroid_shifts_toward_the_detector(self):
        grid = self._grid()
        wn, response = lineshape_srf(10.0, 1.0)
        detector_wn = np.linspace(700.0, 1450.0, 200)
        detector = (detector_wn - 700.0) / 750.0      # favours high wavenumber
        plain = _srf_weights(wn, response, grid, 'band')
        composed = _srf_weights(wn, response, grid, 'band',
                                detector=(detector_wn, detector))
        assert (composed * grid).sum() > (plain * grid).sum()

    def test_no_overlap_is_refused(self):
        wn, response = lineshape_srf(4.0, 0.1)        # MWIR band, LWIR grid
        with pytest.raises(ValueError, match='no overlap'):
            _srf_weights(wn, response, self._grid(), 'band')


class TestCentroid:
    """The field that lets a consumer verify a band pairing."""

    def test_a_symmetric_band_centres_on_itself(self):
        wn, response = lineshape_srf(10.0, 0.4, shape='gaussian')
        assert _centroid_um(wn, response) == pytest.approx(10.0, abs=0.01)

    def test_an_empty_response_is_nan_not_zero(self):
        assert np.isnan(_centroid_um(np.array([900.0, 1000.0]),
                                     np.array([0.0, 0.0])))


@pytest.mark.slow
class TestBuiltLuts:
    """End to end, against the real master."""

    def test_it_records_version_two_metadata(self, tmp_path):
        master = _master_or_skip()
        srf_dir = tmp_path / 'srf'
        srf_dir.mkdir()
        for index, centre in enumerate((9.0, 10.0, 11.0)):
            lineshape_srf(centre, 23.74, domain='wn',
                          out_path=srf_dir / f'band{index:02d}.csv')
        path = build_instrument('TestCam', srf=srf_dir, master_path=master,
                                out_path=tmp_path / 'lut.npz')
        lut = AtmosLUT(path)
        assert lut.filters == ['band00', 'band01', 'band02']
        assert lut.meta['weighting'] == 'as-supplied'
        assert lut.meta['detector'] is None
        centroids = lut.meta['filter_centroids_um']
        assert centroids['band00'] == pytest.approx(9.0, abs=0.02)
        assert centroids['band02'] == pytest.approx(11.0, abs=0.02)

    def test_a_detector_makes_it_composite(self, tmp_path):
        master = _master_or_skip()
        srf_dir = tmp_path / 'srf'
        srf_dir.mkdir()
        lineshape_srf(10.0, 0.5, out_path=srf_dir / 'band00.csv')
        path = build_instrument(
            'TestCam', srf=srf_dir, master_path=master,
            detector='FLIR-microbolometer', out_path=tmp_path / 'lut.npz')
        lut = AtmosLUT(path)
        assert lut.meta['weighting'] == 'composite'
        assert lut.meta['detector'] == 'FLIR-microbolometer'
        assert lut.meta['detector_file'] is not None

    def test_composing_moves_the_transmittance(self, tmp_path):
        # If it did not, recording the weighting would be pointless.
        master = _master_or_skip()
        srf_dir = tmp_path / 'srf'
        srf_dir.mkdir()
        lineshape_srf(12.5, 1.2, out_path=srf_dir / 'band00.csv')
        plain = AtmosLUT(build_instrument(
            'Plain', srf=srf_dir, master_path=master,
            out_path=tmp_path / 'plain.npz'))
        composed = AtmosLUT(build_instrument(
            'Composed', srf=srf_dir, master_path=master,
            detector='FLIR-microbolometer', out_path=tmp_path / 'composed.npz'))
        a = plain.transmittance(293.0, 60.0, 2.0)
        b = composed.transmittance(293.0, 60.0, 2.0)
        assert a != pytest.approx(b, abs=1e-6)


class TestBundledLuts:
    """The shipped library, which the imaging side resolves by name."""

    def test_every_bundled_lut_is_current(self):
        # All eight were rebuilt when filter_centroids_um was added, so the
        # field is universal rather than something a consumer must fall back
        # around. The k-distributions came back bit-identical, so the rebuild
        # changed metadata and nothing physical.
        from atmcorr.registry import available_luts
        for name in available_luts():
            lut = AtmosLUT(name)
            assert lut.meta.get('weighting') in ('composite', 'as-supplied'), name
            centroids = lut.meta.get('filter_centroids_um')
            assert centroids is not None, f"{name} predates filter_centroids_um"
            assert set(centroids) == set(lut.filters), name
            assert all(np.isfinite(v) for v in centroids.values()), name

    def test_centroids_sit_inside_the_master_window(self):
        # A centroid outside the window the LUT was built against would mean the
        # band was clipped, and its transmittance would describe only the part
        # that overlapped.
        from atmcorr.registry import available_luts
        for name in available_luts():
            lut = AtmosLUT(name)
            lo, hi = 1e4 / max(lut.wn_range), 1e4 / min(lut.wn_range)
            for band, centre in lut.meta['filter_centroids_um'].items():
                assert lo <= centre <= hi, f"{name}/{band} at {centre:.3f} um"

    def test_the_gascam_is_composite(self):
        lut = AtmosLUT('MMT-gasCam')
        assert lut.meta['weighting'] == 'composite'
        assert lut.meta['detector'] == 'FLIR-microbolometer'

    def test_the_r100_covers_its_cube(self):
        lut = AtmosLUT('SPI-R100')
        assert len(lut.filters) == 29
        centroids = lut.meta['filter_centroids_um']
        assert len(centroids) == 29
        values = np.array([centroids[f] for f in sorted(centroids)])
        assert values.min() == pytest.approx(7.02, abs=0.01)
        assert values.max() == pytest.approx(13.16, abs=0.01)
        # Evenly spaced in wavenumber, which is what the instrument does.
        steps = np.diff(1e4 / values)
        assert steps.std() < 0.05

    def test_a_reader_sees_the_declared_format_version(self):
        assert FORMAT_VERSION == '2'
