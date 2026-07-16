Preprocessing
=============

The app stores the full-resolution data for calculations and uses
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
     - target 0 ppm, search -0.2 to 0.2 ppm
     - Can be skipped.
   * - Baseline
     - ALS, lambda ``1e6``, p ``0.01``, 12 iterations
     - arPLS and airPLS are also available.
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
     - 4.5-6.1 ppm, zero mode
     - Urine-focused default; interpolation is optional.
   * - Binning
     - width 0.01 ppm, trapezoidal
     - May instead specify total bin count or rectangular integration.
   * - Normalization
     - PQN
     - Total area, SNV, and none are available.

Group delay
-----------

The group-delay step removes the initial number of complex FID points defined
by ``GRPDLY`` or by a user override.

Solvent residual suppression
-----------------------------

A smooth component is estimated separately from the real and imaginary FID
parts using penalized second differences and subtracted from the FID.

Apodization and zero filling
----------------------------

Exponential and Gaussian windows are available. Zero filling appends complex
zeros to the time-domain signal before Fourier transformation.

Fourier transformation and ppm axis
-----------------------------------

The app uses ``fft`` and ``fftshift``. The ppm axis is calculated from
``SW_h``, ``O1``, and ``SFO1``.

Phasing, referencing, and baseline
----------------------------------

Automatic phasing estimates a separate zero-order phase for each spectrum.
Referencing searches for the largest peak in the selected interval. Baseline
methods include ALS, arPLS, and airPLS.

Alignment limitation
--------------------

The current alignment method searches integer point shifts that maximize
correlation with a selected reference spectrum within one ppm window. It is
not a full interval-correlation-shifting implementation.

Binning and normalization
-------------------------

The integrated matrix has samples as rows and ppm-bin centers as columns.
PQN uses the median spectrum as the reference. The plotted bin rectangles are
visual guides; integrated values are stored in the downloadable matrix.
