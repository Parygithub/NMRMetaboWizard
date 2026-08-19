Preprocessing
=============

The app stores full-resolution arrays for calculations and uses
peak-preserving downsampling only for display.

Default parameters
------------------

.. list-table::
   :header-rows: 1
   :widths: 24 26 50

   * - Step
     - Default
     - Notes
   * - Group delay
     - ``-1``
     - Uses Bruker ``GRPDLY``; manual override or skip is available.
   * - Solvent residual suppression
     - enabled, lambda = ``1e6``
     - Whittaker-type FID smoothing with second differences.
   * - Apodization
     - exponential, LB = ``1.0``
     - Gaussian is also available.
   * - Zero filling
     - 32768 extra points
     - Adds digital interpolation; does not add experimental information.
   * - Fourier transformation
     - required
     - FFT followed by frequency reordering and ppm-axis construction.
   * - Phase correction
     - automatic zero-order
     - Manual angle or skip is available.
   * - Referencing
     - disabled
     - Enable only when an appropriate reference peak and interval are known.
   * - Baseline
     - ALS, lambda ``1e6``, p ``0.01``, 12 iterations
     - arPLS and airPLS are also available.
   * - Baseline exclusion interval
     - disabled
     - A custom interval can be excluded when a dominant artifact distorts the fit.
   * - Alignment
     - disabled
     - Integer cross-correlation shift, not full icoshift.
   * - Negative zeroing
     - enabled
     - Sets negative spectral values to zero.
   * - Window
     - 0.2-10 ppm
     - Can be skipped to retain the full ppm range.
   * - Region removal
     - ``None``
     - Optional urine preset or custom interval.
   * - Binning
     - width 0.01 ppm, trapezoidal
     - May instead specify total bin count or rectangular integration.
   * - Normalization
     - PQN
     - Total area, SNV, and none are available.

Group delay
-----------

The group-delay step removes the initial complex FID points defined by
``GRPDLY`` or by a user override.

Solvent residual suppression
----------------------------

A smooth component is estimated separately from the real and imaginary FID
parts using penalized second differences and subtracted from the FID. The
lambda parameter controls smoothness. This optional operation requires visual
quality control because an unsuitable setting may affect broad sample signals.

Apodization and zero filling
----------------------------

Exponential and Gaussian weighting functions are available. Zero filling
appends complex zeros before Fourier transformation. It increases digital
sampling density but does not add experimental information or intrinsic
spectral resolution.

Fourier transformation and ppm axis
-----------------------------------

The time-domain FID is converted to the frequency domain using ``fft`` and
reordered with ``fftshift``. The ppm axis is calculated from ``SW_h``, ``O1``,
and ``SFO1``.

Phasing, referencing, and baseline
----------------------------------

Automatic phasing estimates an independent zero-order phase for each
spectrum. Referencing is disabled by default and should be enabled only when a
suitable reference signal is present. Baseline correction supports ALS,
arPLS, and airPLS. By default, the full spectrum contributes to baseline
estimation; users may optionally exclude a custom solvent or artifact
interval.

Alignment limitation
--------------------

The alignment method searches integer point shifts that maximize correlation
with a selected reference spectrum within one ppm interval. The interval
should avoid dominant solvent or artifact peaks. The method is not full
interval-correlation shifting or local warping.

Region removal
--------------

Region removal is optional and disabled by default. Users may choose:

- ``None`` - preserve the selected spectral window;
- ``Urine water/urea (4.5-6.1 ppm)`` - a urine-specific preset;
- ``Custom`` - a user-defined ppm interval.

The zero mode replaces values inside the interval with zero. The interpolate
mode reconstructs the interval from the signal values at its boundaries. A
sample-type-specific preset should be used only when scientifically justified.

Binning and normalization
-------------------------

Binning partitions each processed spectrum into chemical-shift intervals
defined by a fixed width or a total bin count. Signals are integrated using
the trapezoidal or rectangular method. PQN, total-area normalization, SNV, or
no normalization can then be applied.
Large-cohort memory handling
----------------------------

For cohorts containing 200 or more spectra, NMRMetaboWizard automatically
releases obsolete full-resolution arrays when the user moves from window
selection to region removal. The arrays required for region removal, binning,
normalization, EDA, and machine learning are preserved unchanged. Earlier
preprocessing plots are no longer retained after this transition, so required
quality-control plots should be inspected or downloaded first.

Binning uses a vectorized implementation that assigns spectral points and
trapezoidal segments to bins once, rather than constructing a full-length mask
for every bin. A progress indicator reports the number of processed spectra.
