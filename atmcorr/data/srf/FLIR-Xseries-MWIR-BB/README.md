# Broadband InSb SRF — reference only, no LUT

Digitized from FLIR support answer a_id/3634 (`BB_X-Series_A8500_RS8500.png`), the *broadband* InSb
response of the FLIR X-series / A8500 / RS8500 (~1.0–5.70 µm).

**No LUT is built for this SRF, and one should not be built from the MWIR master.**
The MWIR master covers 1800–3700 cm⁻¹ (2.70–5.56 µm), which contains only ~64 % of
this SRF's response — the rest lies in the SWIR/NIR, below the master's short-wave
edge. Building against the MWIR master would silently clip that portion and yield a
biased transmittance.

A correct broadband LUT would need a master extending into the SWIR/NIR, where the
dominant physics also differs (solar scattering, aerosol extinction), i.e. a
different modelling problem from the thermal-emission window this package targets.

The CSV is kept for completeness/reference (band shape, cutoffs).
