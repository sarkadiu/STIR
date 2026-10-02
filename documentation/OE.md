# Origin ensemble reconstruction

`OE` reconstructs PET list-mode data using the flat-prior minimum-mean-square-error
origin ensemble algorithm (MMSE-OE), including TOF information when available.
This first implementation corrects for attenuation
using a supplied 511 keV attenuation image. It processes prompt coincidences and ignores
delayed coincidences. Scatter, randoms, detector normalization, and quantitative activity
calibration are not implemented.

## Running a reconstruction

Build STIR with its reconstruction programs enabled, then run:

```text
OE OE.par
```

Start from [`examples/samples/OE.par`](../examples/samples/OE.par). Set `INPUT` to the
list-mode filename, `ATTENUATION_IMAGE` to the attenuation image filename, and
`OUTPUT` to the desired output prefix, or replace those environment-variable references
with literal filenames. The input must be readable by STIR's list-mode reader.
TOF acquisition metadata determines timing bins and the scanner timing resolution;
non-TOF input samples positions uniformly along each LOR inside the image. The attenuation
image must be readable by STIR, contain linear
attenuation coefficients at 511 keV in **cm^-1**, and be registered to the acquisition
coordinate system. A CT image in Hounsfield units is not an attenuation image; convert
and register it before running `OE`.

`start time` and `end time` select a frame `[start time, end time)` in seconds.
The default `end time := -1`
reads through the end of the list. `maximum segment number := -1` includes all segments
supported by the acquisition. A nonnegative value restricts both accepted events and
sensitivity computation to segments whose absolute segment number is at most that value.
An explicit finite end time sets the output image's frame metadata to that interval.

Image geometry uses the usual STIR parameters: `zoom`, `Z zoom`,
`XY output image size (in pixels)`, `Z output image size (in pixels)`, and
`Z offset (in mm)`. An image size of `-1` uses the scanner-derived default.
The base reconstruction options `output file format type`, `post-filter type`,
`disable output`, and `verbosity` are also available. Post-filtering changes the final
image; use `post-filter type := None` to retain the unfiltered ensemble estimate.

## Sampling rule and attenuation

Each retained prompt event has one current origin voxel. Let event `n` occupy voxel
`i`, let `j` be its proposed voxel, and let `c_i` and `c_j` be the current event counts
before moving that event. Each proposal draws a continuous position along the LOR,
then determines its voxel from the image origin and voxel spacing. No ray tracing,
voxel traversal, or projection-matrix row lookup occurs during event sampling.

For TOF data, let `[low, high]` be the event's timing-bin boundaries in distance
units and `sigma` be the scanner timing FWHM converted to distance and divided by
2.355. Draw `u` uniformly within that bin and `g` from a zero-mean Gaussian with
standard deviation `sigma`. The signed TOF position is `d = u + g`, and the
physical position is `LOR_midpoint - d * unit_vector_towards_second_detector`.
This convolution samples STIR's bin-integrated Gaussian TOF kernel, including
coarse or mashed timing bins. It follows STIR's canonical LOR direction and
scanner-to-image axial coordinate offset. The portable list-mode adapter uses
the canonical LOR of the event's projection bin; compressed spatial bins
therefore use their representative LOR rather than a sub-bin detector position.

Non-TOF proposals draw uniformly from the LOR's intersection with the image box.
The intersection is calculated analytically once per distinct event bin, without
traversing voxels. TOF proposals outside the image, or in a voxel with zero
sensitivity, leave the state unchanged. Initial origins are obtained by rejection
sampling; setup fails after a bounded number of unsuccessful draws rather than
changing the proposal distribution. The acceptance probability is

```text
A(i -> j) = min(1, (c_j + 1) / c_i * epsilon_i / epsilon_j).
```

Same-voxel proposals also leave the state unchanged. The continuous kernel integrated
over each voxel is the event model; its factors cancel against the proposal in
the Metropolis-Hastings ratio. This is the flat-prior MMSE-OE count rule of
[Wulker, Sitek, and Prevrhal (2015), equations 22-29](https://pmc.ncbi.nlm.nih.gov/articles/PMC6363359/).
It differs from the earlier count-power OE rule in
[Sitek (2011), equations 6-8](https://pmc.ncbi.nlm.nih.gov/articles/PMC3079437/).

For geometric projection bin `b`, the PET attenuation survival factor is

```text
a_b = exp(-integral_over_LOR_b mu ds).
epsilon_i = sum_over_all_geometric_bins_b (a_b * P_bi_non_TOF).
```

Ray tracing is used only during setup: first to integrate the mu-map along each
LOR and then to accumulate attenuation-weighted **non-TOF sensitivity** over all
spatial acquisition bins, including bins with no observed events. There is no
loop over timing bins in this calculation. This is also STIR's usual list-mode
sensitivity convention: a normalized TOF kernel integrates to one over the full
timing range. It assumes the acquisition timing window covers that kernel over
the reconstruction region. Data with a deliberately truncated timing window
require an additional timing-coverage factor, which is not implemented here.
The configured `projection matrix type` is used for sensitivity only.

Attenuation enters the acceptance rule through sensitivity. Its survival factor
is constant along a fixed PET LOR and cancels from event proposals. Voxels
without positive sensitivity cannot contain an event origin. Direct continuous
position sampling integrates the TOF kernel within each voxel rather than
using the old voxel-center TOF matrix approximation. The ray-traced sensitivity
retains STIR's geometric discretization; use one tangential ray per bin to match
the single-LOR event model most closely.

Attenuation ray tracing uses the mu-map's own supported square transaxial FOV,
including material outside the emission image's reconstruction FOV. Its voxel grid may
differ from the emission grid, subject to the existing ray-tracing matrix restrictions:
the x/y origin must be centered at zero, and the axial LOR sampling distance must be an
integer multiple of the mu-map's axial voxel spacing. The projector uses an inscribed
square and clips a small outer margin, so all attenuating material must fit inside that
region. Pad and register a compatible square grid with zero attenuation near its
outer margins when necessary; arbitrary native CT grids are not supported.
Unit conversion for projection is handled internally by
`BinNormalisationFromAttenuationImage`; supply the mu-map in cm^-1 without manually
multiplying it by a voxel size.

## Output and practical limits

The chain first performs `number of burn-in sweeps`. It then collects
`number of samples`, separated by `sweeps between samples` complete event sweeps.
The output voxel value is the sampled mean event count divided by that voxel's
attenuated sensitivity. The `+1` appears in the acceptance rule, not as an extra count
added to the output image. Values are in count/system-matrix-sensitivity units and are
not calibrated Bq/ml. This program does not infer or change attenuation-map calibration.
An attenuation image is required; an all-zero image permits an unattenuated comparison.

`random seed` permits repeatable sampling for a fixed input and configuration using
the same build and C++ standard library.
The sample file's sweep counts demonstrate the interface; they do not establish that
an arbitrary dataset has reached equilibrium. Sample progress messages do not diagnose
equilibrium; compare longer burn-in and sampling runs when assessing stability.
Consecutive samples can remain correlated.

This is a serial CPU reference implementation. It stores an origin per retained event
and one compact LOR sampler per distinct event bin. It does not store projection
matrix rows for proposals. Memory grows with the event count and distinct bin count.
Sensitivity setup traverses the full selected spatial acquisition geometry and can be expensive. Use a suitable frame and image grid when
evaluating large acquisitions.
