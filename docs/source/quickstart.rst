Quick start
===========

1. Prepare a ZIP file containing Bruker experiment folders with ``fid`` and
   ``acqus`` files.
2. Ensure that biological sample-folder names match the future clinical
   ``study_id`` values.
3. Remove dummy acquisition folders such as ``99999``.
4. Start the app and upload the ZIP.

*Try the synthetic example
-------------------------
At Step 1, the application provides two downloadable demonstration files:

- a ZIP archive containing 16 fully synthetic Bruker-like FIDs;
- a matching clinical metadata CSV containing 8 BPH-labelled and 8
  PCa-labelled samples.

Download both files from the application. Upload the NMR ZIP at Step 1, then
upload the matching clinical CSV at Step 17.

5. Inspect the raw FIDs and apply or skip each preprocessing step.
6. Create the binned table and apply normalization.
7. Upload a clinical file containing ``study_id`` and ``Class``.
8. Review the alignment summary and unmatched IDs.
9. Run EDA.
10. Review possible outliers and decide whether to retain or remove them.
11. Rerun EDA after the outlier decision.
12. Configure and run machine-learning models.
13. Download processing logs, plot data, feature matrices, predictions, and
    model-performance history.

Recommended first-pass settings for urine
-----------------------------------------

- group delay: use Bruker ``GRPDLY``;
- alignment: off unless visible drift is present;
- window: 0.2-10 ppm;
- region removal: 4.5-6.1 ppm;
- bin width: 0.01 ppm;
- integration: trapezoidal;
- normalization: PQN;
- ML PCA: off initially;
- outlier detection: groupwise Hotelling's T², followed by manual review.
