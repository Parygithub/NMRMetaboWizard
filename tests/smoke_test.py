from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from nmr_pipeline import (
    apply_group_delay,
    apply_solvent_residuals_removal,
    apply_apodization,
    apply_zero_filling,
    apply_fourier_transform,
    apply_phase_correction,
    apply_referencing,
    apply_baseline_correction,
    apply_peak_alignment,
    apply_negative_values_zeroing,
    apply_window_selection,
    apply_region_removal,
    apply_binning,
    apply_normalization,
)

from clinical_analysis import (
    merge_omics_clinical,
    pca_scores,
    plsda_scores,
    detect_pca_outliers,
    train_ml_model,
)


def synthetic_samples(n_samples: int = 16, n_points: int = 1024):
    rng = np.random.default_rng(123)
    samples = []

    t = np.arange(n_points) / 5000.0

    for i in range(n_samples):
        cls_shift = 0.15 if i >= n_samples // 2 else 0.0
        fid = (
            np.exp(-8 * t) * np.exp(1j * 2 * np.pi * (400 + cls_shift * 20) * t)
            + 0.5 * np.exp(-14 * t) * np.exp(1j * 2 * np.pi * 900 * t)
            + 0.02 * (rng.normal(size=n_points) + 1j * rng.normal(size=n_points))
        )
        samples.append(
            {
                "name": f"synthetic_{i:03d}",
                "sample_id": f"synthetic_{i:03d}",
                "folder": "synthetic",
                "acqus": {
                    "SW_h": 5000.0,
                    "O1": 3000.0,
                    "SFO1": 600.0,
                    "GRPDLY": 4.0,
                },
                "raw_fid": fid,
                "log": [],
            }
        )

    return samples


def main():
    samples = synthetic_samples()

    samples = apply_group_delay(samples)
    samples = apply_solvent_residuals_removal(samples, lam=1e4, enabled=False)
    samples = apply_apodization(samples, lb=1.0, kind="exponential")
    samples = apply_zero_filling(samples, extra_points=512)
    samples = apply_fourier_transform(samples)
    samples = apply_phase_correction(samples, auto=True)
    samples = apply_referencing(
        samples,
        use_reference=False,
        target_ppm=0.0,
        search_min=-0.2,
        search_max=0.2,
    )
    samples = apply_baseline_correction(
        samples,
        method="als",
        smoothness=1e4,
        asymmetry=0.01,
        max_iter=3,
        exclude_region_text="",
        max_points=300,
    )
    samples = apply_peak_alignment(samples, enabled=False)
    samples = apply_negative_values_zeroing(samples, enabled=True)
    samples = apply_window_selection(samples, ppm_min=0.2, ppm_max=10.0)
    samples = apply_region_removal(samples, region_text="4.5-6.1", mode="zero")
    samples, binned = apply_binning(samples, n_bins=40, method="trapezoidal")
    normalized = apply_normalization(samples, binned, method="PQN")

    clinical = pd.DataFrame(
        {
            "study_id": [f"synthetic_{i:03d}" for i in range(len(samples))],
            "Class": ["BPH"] * (len(samples) // 2) + ["PCa"] * (len(samples) // 2),
            "age": np.linspace(55, 75, len(samples)),
            "psa": np.r_[np.linspace(2, 6, len(samples) // 2), np.linspace(5, 12, len(samples) // 2)],
        }
    )

    aligned = merge_omics_clinical(
        normalized,
        clinical,
        clinical_id_col="study_id",
        class_col="Class",
    )

    assert aligned["X"].shape[0] == len(samples)
    assert aligned["X"].shape[1] == 40

    pca = pca_scores(aligned, n_components=3)
    pls = plsda_scores(aligned, n_components=2)
    outliers = detect_pca_outliers(aligned, n_components=3, groupwise=True)

    assert not pca["scores"].empty
    assert not pca["loadings"].empty
    assert not pls["scores"].empty
    assert "table" in outliers

    for model in ["LogisticRegression", "RandomForest", "LinearSVM"]:
        result = train_ml_model(
            aligned,
            model_name=model,
            test_size=0.25,
            cv_folds=2,
            use_pca=False,
            feature_mode="NMR only",
            use_cv=False,
        )
        assert "metrics" in result
        assert "feature_importance" in result

    ann = train_ml_model(
        aligned,
        model_name="ANN",
        test_size=0.25,
        cv_folds=2,
        use_pca=False,
        feature_mode="NMR only",
        ann_hidden_layers="8",
        ann_max_iter=30,
        ann_early_stopping=False,
        use_cv=False,
    )
    assert "metrics" in ann

    print("NMRMetaboWizard smoke test passed.")


if __name__ == "__main__":
    main()
