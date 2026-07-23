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

The group-delay step removes the initial FID points introduced by digital oversampling and filtering hardware (which do not contain spectral information) defined
by ``GRPDLY`` or by a user override.

Solvent residual suppression
-----------------------------
Solvent residual removal suppresses the intense residual solvent signal (usually water) in the NMR spectrum to improve visualization and analysis of weaker analyte peaks.
λ is used as the smoothing or roughness penalty parameter used in Whittaker-based solvent residual estimation: the higher λ is, the smoother the estimated solvent signal becomes. 

Apodization and zero filling
----------------------------

Apodization applies a weighting function to the time-domain FID before Fourier transformation. Exponential and Gaussian windows are available to improve signal-to-noise ratio, reduce truncation artifacts, or modify spectral line shapes, although these effects may involve a trade-off in spectral resolution. 
Zero filling appends complex zeros to the FID before Fourier transformation, increasing the number of frequency-domain data points and producing a smoother spectrum without adding new spectral information or improving the intrinsic resolution.

Fourier transformation and ppm axis
-----------------------------------

The time-domain FID is converted into a frequency-domain spectrum using ``fft``, while ``fftshift`` reorders the frequency components around the spectral center. The chemical-shift axis is then calculated in parts per million (ppm) from the spectral width (``SW_h``). transmitter offset (``O1``), and observation frequency (``SFO1``), allowing peak positions to be reported independently of the magnetic-field strength.

Phasing, referencing, and baseline
----------------------------------

Automatic phasing estimates and applies an independent zero-order phase correction to each spectrum, improving absorptive peak shapes and reducing dispersive distortions. Chemical-shift referencing identifies the largest peak within a selected interval and aligns it with the specified reference position. Baseline correction removes slowly varying background distortions using asymmetric least squares (ALS), asymmetrically reweighted penalized least squares (arPLS), or adaptive iteratively reweighted penalized least squares (airPLS), thereby improving peak visualization and quantification.

Alignment limitation
--------------------

The current alignment method searches integer point shifts that maximize
correlation with a selected reference spectrum within one ppm window. It is
not a full interval-correlation-shifting implementation.

Negative-value zeroing 
--------------------

Negative intensities may remain after phasing and baseline correction, particularly in noisy or low-signal regions. Setting negative values to zero produces a non-negative dataset that is better suited to integration, normalization, and multivariate modelling. By default, the app replaces all negative values with zero, although users may change this setting.

Spectral window selection
-------------------------

Limiting the analysis to a relevant ppm range removes unused portions of the spectrum and ensures that all samples are processed over the same domain. This reduces the size of the resulting dataset and prevents regions with little analytical value from contributing to subsequent region exclusion, binning, normalization, and statistical modelling. The lower and upper limits are entered in the 'Window min ppm' and 'Window max ppm' fields, and the retained region is displayed for visual inspection.

Region removal
-------------------------

This step removes an internal ppm interval that should not contribute to later analysis. For urine samples, the 4.5–6.1 ppm range is excluded to reduce the influence of intense water- and urea-associated signals in the spectra.
Two processing options are available. The zero option replaces all values inside the specified interval with zero, whereas interpolate reconstructs the interval from the signal values at its boundaries. The first option retains an explicitly empty region, while the second provides a continuous spectral profile. 

Binning and normalization
-------------------------

Binning partitions each processed NMR spectrum into chemical-shift intervals defined either by a fixed bin width or by a specified number of bins. The signal within each interval is integrated using either the 'trapezoidal** or **rectangular** method to generate a binned data matrix, with samples arranged as rows and ppm-bin centres as columns. The shaded vertical regions displayed across the spectrum indicate the bin boundaries, while the calculated integrals can be inspected in the table preview and exported as a CSV file.
The binned data can then be normalized using probabilistic quotient normalization (PQN), total-area normalization, standard normal variate normalization (SNV), or left unnormalized. PQN estimates sample-specific dilution factors relative to the median spectrum, total-area normalization scales each sample by its total integrated signal, and SNV centres and scales each binned spectrum using its mean and standard deviation. The resulting normalized matrix can be exported for exploratory analysis, statistical testing, and machine-learning applications.
