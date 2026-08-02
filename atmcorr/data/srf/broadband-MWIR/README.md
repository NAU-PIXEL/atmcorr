# Generic MWIR band — 3–5 µm top-hat

A flat 3–5 µm (2000–3333 cm⁻¹) response, for use when a MWIR camera's real SRF is
unavailable. Unlike the LWIR generic, this is a close stand-in rather than a rough
one: real cooled InSb detectors are near-square over 3–5 µm with sharp cutoffs, so
the top-hat tracks the digitized FLIR InSb responses to within ~1.4 percentage
points of transmittance at all distances.

**Note on the CO₂ band.** The 3–5 µm window is split by the strong CO₂ ν₃ band at
4.3 µm, which is effectively opaque over any appreciable path. That blackout sits
*inside* this band (and inside every real InSb response), so it is correctly
included rather than excluded — the k-distribution handles it. Expect noticeably
lower band transmittance than an LWIR instrument at the same conditions: the
useful signal comes from the 3.5–4.1 µm and 4.5–5.0 µm sub-windows either side.

Prefer a real instrument SRF when you have one — `FLIR-A6700-MWIR` and
`FLIR-Xseries-MWIR` are bundled.
