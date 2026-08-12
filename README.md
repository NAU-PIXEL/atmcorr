# atmcorr — atmospheric-correction LUTs for thermal imaging

`atmcorr` builds and applies physically-derived atmospheric-correction lookup
tables (LUTs) for thermal cameras, aimed at the long-distance viewing regime
where a camera's built-in correction (e.g. FLIR's) is unreliable. A LUT returns
the **band transmittance** of the atmosphere for given ambient conditions and a
per-pixel distance map.

The package has two sides, both **RFM-free**:

- **Reader** (`AtmosLUT`) — applies a precomputed LUT. Needs only NumPy and SciPy,
  so it drops into any imaging pipeline.
- **Generator** (`build_instrument`) — Stage 2 of the factory: turns a *master* of
  optical-depth spectra plus an instrument's spectral response (SRF) into a
  compact per-instrument LUT. Additionally needs pandas (to read SRF CSVs).

The expensive line-by-line **master** is produced separately by **RFMwrapper**
(Stage 1 — the only step needing RFM). At 163 MB it is too large for the git tree,
so it is distributed as a **GitHub Release asset** and fetched on demand (see
`fetch_master`); it is needed only for *generation*, never for applying a LUT. One
LUT file describes one instrument and may hold several named filters (a multi-band
camera); a broadband LWIR camera (e.g. a FLIR Boson or Duo) is the single-filter case.

> **Scope — ground-based instruments only.** The master models a *single
> homogeneous layer* (a horizontal path at uniform temperature, pressure and
> humidity). This is right for ground-based cameras viewing across the surface. It
> is **not** valid for satellite / nadir instruments, whose paths traverse a
> stratified atmosphere with strongly varying conditions — those need a layered
> forward model and are out of scope here. The shipped range (700–1450 cm⁻¹) is
> LWIR; a 3–5 µm (MWIR) master would be a separate build.

## Install

```bash
pip install -e .            # reader only (numpy, scipy)
pip install -e '.[build]'   # + generation (adds pandas)
pip install -e '.[plot]'    # + plotting  (adds matplotlib)
pip install -e '.[all]'     # everything
```

---

## What problem this solves

To turn radiance into a true surface temperature at range, you must divide out
how much the intervening air absorbed. That fraction is the **transmittance**
`tau ∈ [0, 1]`. It depends on the ambient air (temperature, pressure, humidity)
and, strongly, on the path length.

FLIR estimates `tau` with a compact empirical formula. It is tuned for short
industrial distances and degrades badly over kilometres — exactly the regime
this project cares about. `atmcorr` replaces that estimate with one computed
from first principles.

---

## How it differs from FLIR

FLIR's law (its atmospheric-correction stage) is a two-band empirical fit:

```
tau = X·exp(−√d·(α + β·√ω)) + (1−X)·exp(−√d·(α₂ + β₂·√ω))
```

with `ω` a water-vapour-pressure polynomial and `d` the distance. Two issues
dominate at long range:

1. **The `√d` distance law is a fudge.** Physically, absorption along a uniform
   path grows *linearly* with distance, not as `√d`. FLIR uses `√d` as a cheap
   one-parameter stand-in for the fact that the *band-averaged* transmittance
   falls off in a curved, non-exponential way. Fitted against a true
   line-by-line spectrum, this `√d` form is off by up to **~14 percentage
   points** over 0–15 km.

2. **It cannot capture the water-vapour self-continuum.** The self-continuum
   absorbs in proportion to water-vapour *density squared*. As a result, a short
   humid path and a long dry path with the *same total water column* are **not**
   equivalent — they differ by **~9 percentage points** of transmittance at
   representative long ranges. FLIR's `β·√ω` term cannot represent this; our LUT,
   built from RFM + the MT_CKD continuum, does so exactly.

The LUT keeps what FLIR got right — working in **optical depth** inside the
exponential — but fixes the `√d`: each spectral slice uses the exact
linear-in-distance optical depth, and the band curvature is recovered by summing
slices (see below), not by bending a single distance law.

---

## How it works: the k-distribution (in plain terms)

RFM produces a full transmittance *spectrum* for each atmospheric state. Your
camera, however, does not see a spectrum — its filter sums all wavelengths into a
single number. So the quantity we ultimately need is one transmittance value per
condition, with the wavelength axis integrated away.

The only remaining complication is distance, and it has a simple picture:

> Imagine the atmosphere as a stack of dirty windows of **different dirtiness**.
> The camera looks through all of them at once and averages. Over a **short**
> path, even the dirty windows pass some light. Over a **long** path, only the
> **clean** windows still matter — so the average transmittance stays higher than
> a naive "square it for double the distance" would predict.

A single window (one absorption level `k`) follows an exact, simple law:
`exp(−k·d)`. The *mixture* does not — that is the whole difficulty. So we split
the band into levels, each of which behaves perfectly, and recombine them by how
much of the band they represent. That summary — a short list of absorption
levels `k` with weights `w` — is the **k-distribution**:

```
tau(d) = Σ_g  w_g · exp(−k_g · d)
```

For a uniform (homogeneous) path this is **exact at every distance**, needs no
distance axis in the table, and extrapolates safely to any range. The LUT stores,
per filter, the array `k_g(temp, pres, rh)` plus the shared weights `w_g`. At
query time the ambient state fixes one `k_g` vector; distance is then analytic.

Contrast the two philosophies:

| | FLIR | This LUT |
|---|---|---|
| Source | empirical two-band fit | RFM line-by-line + MT_CKD continuum + real filter response |
| Distance law | `√d` (approximate) | exact `exp(−k·d)` per slice, summed |
| Self-continuum (humidity²) | not represented | exact |
| Range | short/industrial | long (km) |

---

## Symbol conventions

To avoid the classic collision between *temperature* and *transmittance* (both
"T"), no bare `T` is ever used:

| Symbol | Meaning | Unit |
|---|---|---|
| `temp` | ambient air temperature | K (or °C via `temp_unit='C'`) |
| `pres` | ambient pressure | mbar |
| `altitude` | deployment altitude (alternative to `pres`, ISA-converted) | m |
| `rh` | relative humidity | % |
| `dist` | path length | km (or m via `dist_unit='m'`) |
| `tau` / transmittance | band transmittance (primary output) | — (0–1) |
| `k`, `k_g` | absorption coefficient (stored; optical depth per km) | km⁻¹ |

Optical depth is intentionally **not** exposed: the correction is applied with
transmittance, and `optical_depth = −ln(transmittance)` is a trivial conversion
that is never needed here.

---

## Usage

### Applying a LUT (imaging side — no master needed)

```python
from atmcorr import AtmosLUT, available_luts

print(available_luts())          # instruments with a resolvable LUT, e.g. ['FLIR-microbolometer']
lut = AtmosLUT('FLIR-microbolometer')   # a FLIR VOx microbolometer (Boson/Duo); or an explicit .npz path
print(lut)                       # instrument, filters, grid size, wn range

# Direct query: ambient state + a per-pixel distance map (metres here).
tau = lut.transmittance(temp=25.0, rh=55.0, dist=distance_map_m,
                        temp_unit='C', dist_unit='m')

# When only altitude is known (no pressure sensor), supply altitude [m] instead
# of pres; it is converted via the ISA barometric formula. Pass msl_pres=<local
# sea-level pressure> to correct for daily weather.
tau = lut.transmittance(temp=25.0, rh=55.0, dist=distance_map_m,
                        altitude=3200.0, temp_unit='C', dist_unit='m')

# 1-D curve for plotting or inspection.
d, tau_curve = lut.curve(temp=25.0, rh=55.0, temp_unit='C')

# Export a compact closed form, if a downstream model wants coefficients rather
# than the reader (6–8 terms track a real band to <~0.1 %pts):
w, k = lut.expsum(temp=25.0, rh=55.0, n_terms=6, temp_unit='C')
```

A per-pixel distance map is evaluated exactly via build-curve-then-interpolate
(~18 ms for a 640×512 frame). Ambient state must be a single value per query
(one weather reading); only the distance map varies per pixel. Queries outside
the tabulated envelope clamp to the nearest grid edge with a warning.

**Generic fallbacks — `broadband-LWIR` / `broadband-MWIR`.** When you don't have a
camera's actual SRF, use a flat top-hat: **8–12 µm** for LWIR (the clean core) or
**3–5 µm** for MWIR. The MWIR one is a close stand-in — real cooled InSb detectors
are near-square over 3–5 µm, so it tracks the bundled InSb LUTs to ~1.4 pp; the
LWIR one is rougher, since real microbolometers taper at the band edges. Treat
either as a first-order approximation until the instrument's SRF is available.
Make your own generic band with the `tophat_srf` helper:

```python
from atmcorr import tophat_srf, build_instrument
tophat_srf(7.5, 13.5, out_path='data/srf/MyCam/broadband.csv')   # wl [µm]
build_instrument('MyCam')
```

Keep the band to the detector's usable *core*: too wide a top-hat spans the opaque
window edges (near 7 and 14 µm) that a real, edge-tapered detector barely sees, so
it reads pessimistically low.

### Generating a LUT (needs the master; one-off)

Generation needs the master, which is not in the repo — fetch it once:

```python
import atmcorr
atmcorr.fetch_master()                       # downloads the ~163 MB Release asset
# or, if you built one with RFMwrapper: atmcorr.fetch_master(source='…/master_opt_spectra.npz')
```

Then build. With a bundled SRF, the instrument name is enough:

```python
from atmcorr import build_instrument

build_instrument('FLIR-microbolometer')      # bundled SRF + bundled master → LUT

# A new instrument: point srf at your CSV(s) (name, path, directory, or {name: path}).
build_instrument('MyCam', srf='path/to/MyCam_srf/')
```

Each SRF CSV needs a `wn` [cm⁻¹] **or** `wl` [µm] column and a `T` response column
(any positive scale — it is normalised internally). One CSV per band; a directory
becomes a multi-filter LUT named by filename stem. Drop the result in
`$ATMCORR_LUT_DIR` (or the bundled `data/luts/`) and it resolves by name.

#### Filters in front of a detector

For a filter-wheel instrument the CSVs are usually **filter transmissions alone**,
but light passes through the filter *and* the detector, so a band's response is
the **product** of the two. Pass the detector and the weights are composed:

```python
build_instrument('MMT-gasCam', detector='FLIR-microbolometer')
```

`detector` takes an instrument name or a CSV path. Use it whenever your CSVs are
filter curves; omit it when each CSV already *is* a band's whole response, as for
a single-detector camera.

How much it matters, measured on the gasCam's twelve filters against the bundled
LWIR master: band transmittance shifts by **at most 0.0027 absolute** across the
full grid (2520 atmospheres × three path lengths), worth ≲0.8 K at 500 K. Small,
but a bias rather than noise, and it costs nothing to get right.

The choice is recorded in the LUT, because a composed LUT and a filter-only one
are otherwise indistinguishable — same keys, same shapes, different meaning:

```python
import json, numpy as np
meta = json.loads(str(np.load('MMT-gasCam_atmos_lut.npz', allow_pickle=True)['meta']))
meta['weighting']    # 'composite' or 'as-supplied'
meta['detector']     # the curve used, or None
```

`'as-supplied'` rather than `'filter-only'`: for a single-curve instrument the CSV
already is the whole response, so nothing is missing.

### Plugging into an imaging pipeline

A pipeline with a custom-atmosphere hook (e.g. a FLIR-style `atm_model='custom'`,
which accepts a `tau(T_atm, RH, d)` callable) can be driven by `as_callable`,
whose defaults assume the Celsius / metre conventions such pipelines carry:

```python
lut = AtmosLUT('FLIR-microbolometer')
tau_fn = lut.as_callable()                    # tau(T_atm[°C], RH[%], d[m])
raw_to_temperature(..., atm_model='custom', atm_tau=tau_fn)
```

Only the transmittance computation changes; all the surrounding radiative
bookkeeping (reflected, atmospheric self-emission, window terms, `split`/`single`
geometry) is untouched. For a `split` path, feed `d/2` as the FLIR model already
does — `tau_fn` accepts any distance.

### Multi-filter instruments

For a multi-band instrument, name the filter:

```python
tau = lut.transmittance(temp=25.0, rh=55.0, dist=dmap, filter='band_A',
                        temp_unit='C', dist_unit='m')
```

`lut.filters` lists the available names. Queries outside the tabulated ambient
envelope are clamped to the nearest grid edge (with a warning) rather than
extrapolated, to keep absorption non-negative.

---

## Add your own instrument (no code, no package changes)

Have a new camera? Generate a LUT for its SRF and use it by name — the reader
checks your user dir *before* the bundled set, so nothing in the installed package
changes. Everything below is available on the command line (after `pip install`,
the bare `atmcorr <cmd>` works as shorthand for `python -m atmcorr <cmd>`):

```bash
# 1. Get the master once (needs the atmcorr[build] extra for pandas):
python -m atmcorr fetch-master                 # from MASTER_URL
#   or from a local copy you already have:
python -m atmcorr fetch-master --source /path/to/master_opt_spectra.npz

# 2. Make an SRF CSV (columns: `wn` [cm⁻¹] or `wl` [µm], plus `T`; one CSV per band,
#    a folder of CSVs = a multi-band instrument). Then build the LUT:
python -m atmcorr build MyCam --srf MyCam.csv --out ~/.local/share/atmcorr/luts/

# 3. Use it by name:
python -c "from atmcorr import AtmosLUT; print(AtmosLUT('MyCam'))"
```

Other CLI commands:

```bash
python -m atmcorr list          # instruments with a resolvable LUT (bundled + user dirs)
python -m atmcorr build --help  # all build options (--master, --n-g, --default-pres, …)
```

**Constraints.** The bundled master is **LWIR, 700–1450 cm⁻¹ (≈ 6.9–14.3 µm)** — a
custom SRF must fall in that band (a MWIR 3–5 µm or SWIR instrument needs a
different master). And the whole model is **ground-based, single-homogeneous-layer**
(not valid for satellite / nadir geometry).

**Contributing an instrument upstream.** To make an instrument available to
everyone, add its SRF CSV to `atmcorr/data/srf/<name>/` and the built LUT to
`atmcorr/data/luts/`, and open a PR — both are small and commit fine; the master
stays external.

---

## Plotting

Two helpers in `atmcorr.plotting` (needs `atmcorr[plot]`). Both return
`(fig, ax)` and never call `plt.show()`, so they compose into larger figures.
They are also reachable lazily as `atmcorr.plotLUT` / `atmcorr.plotMaster` —
importing `atmcorr` itself never requires matplotlib.

**`plotLUT`** — band transmittance against any of the four variables
`temp, pres, rh, dist`. Name one or two as free axes; the rest are pinned by
keyword. One free axis gives lines, two give a heatmap (or `kind='contour'` /
`'surface'`).

```python
from atmcorr import plotLUT

# τ vs distance, one line per relative humidity
plotLUT('FLIR-microbolometer', x='dist', hue='rh', temp=298)

# all 12 bands overlaid (default layout for line plots)
plotLUT('MMT-gasCam', x='dist', temp=298, rh=60)

# τ over temperature × humidity at 5 km — one panel per filter
plotLUT('MMT-gasCam', x='temp', y='rh', dist=5000)

# 3-D surface, single band
plotLUT('FLIR-microbolometer', x='dist', y='rh', kind='surface')
```

With several filters, line plots **overlay** by default and 2-D views use a
**subplot grid**; override with `layout='grid'` / `'overlay'`, or select bands with
`filter='ANDV16931'` (or a list).

**`plotMaster`** — the spectral view: wavenumber on the x-axis with a secondary µm
axis on top. Overlay an instrument SRF to see which absorption features a band
actually samples — a quick way to sanity-check a new SRF against a window.

```python
from atmcorr import plotMaster

plotMaster(window='LWIR', dist=[500, 2000, 10000],
           srf='FLIR-microbolometer', temp=298, rh=60)
plotMaster(window='MWIR', srf='FLIR-A6700-MWIR', dist=2000)
plotMaster(window='LWIR', srf='MMT-gasCam')        # every band, one colour each
```

By default **wavelength ascends left→right** (`ascending='wl'`), so cm⁻¹ runs
descending — matching both how the IR-imaging community reads spectra and the
classic IR-spectroscopy convention. Pass `ascending='wn'` to flip it back.

Colours follow the data type: a **sequential** map (`cmap`, default `cividis`) for
ordered quantities — 2-D views, distance families, `hue` families — and a
**qualitative** map (`line_cmap` / `srf_cmap`, default `tab20`) for discrete series
such as filters and SRF bands.

---

## On-disk format (`.npz`)

Produced by `atmcorr.build_instrument`; the writer `AtmosLUT.write` lives in
`atmcorr/lut.py` so the reader and generator share one definition of the contract:

| Key | Type | Meaning |
|---|---|---|
| `format_version` | str | schema version — see below |
| `instrument` | str | instrument name |
| `wn_range` | float (2,) | spectral range of the RFM runs [cm⁻¹] |
| `temp`, `pres`, `rh` | float (n,) | grid axes [K], [mbar], [%] |
| `gweight` | float (n_g,) | k-distribution weights (sum = 1) |
| `filters` | str (n_filt,) | filter names |
| `kdist__<name>` | float (n_temp, n_pres, n_rh, n_g) | sorted absorption coefficients [km⁻¹] |
| `default_pres` | float | pressure assumed when a query omits it [mbar] |
| `meta` | str (JSON) | provenance (RFM/HITRAN version, reference path, date, …) plus `weighting`, `detector`, `detector_file` |

### Schema versions

| `format_version` | |
|---|---|
| `1` | original layout |
| `2` | `meta` records how bands were weighted: `weighting` (`'composite'` / `'as-supplied'`), `detector`, `detector_file` |

A version-1 LUT records **nothing** about its weighting, so it cannot be
established from the file. If its instrument has filters in front of a detector,
that LUT is filter-only and disagrees with a composite radiance — rebuild it with
`detector=` rather than assume. The reader says so on load.

Reading a LUT whose version differs from the reader's warns but proceeds; nothing
in the array layout changed between 1 and 2.
