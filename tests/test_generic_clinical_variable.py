from __future__ import annotations

"""Check disease-neutral clinical-variable stratification and modelling."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from clinical_analysis import merge_omics_clinical, train_ml_model  # noqa: E402


def build_aligned() -> dict:
    rng = np.random.default_rng(123)
    sample_ids = [f"sample_{i:02d}" for i in range(24)]
    labels = np.array(["Control"] * 12 + ["Case"] * 12)

    spectra = pd.DataFrame(
        rng.normal(size=(24, 20)),
        index=sample_ids,
        columns=[f"{0.5 + i * 0.02:.4f}" for i in range(20)],
    )
    spectra.loc[labels == "Case", spectra.columns[:3]] += 0.8

    clinical = pd.DataFrame(
        {
            "study_id": sample_ids,
            "Class": labels,
            "biomarker_x": np.r_[
                rng.normal(2.0, 0.4, 12),
                rng.normal(5.0, 0.5, 12),
            ],
            "age": rng.integers(40, 76, size=24),
            "diagnosis_code": np.r_[np.zeros(12), np.ones(12)],
        }
    )

    return merge_omics_clinical(
        spectra,
        clinical,
        clinical_id_col="study_id",
        class_col="Class",
    )


def test_generic_selected_variable_and_threshold_rule() -> None:
    aligned = build_aligned()

    result = train_ml_model(
        aligned,
        model_name="LogisticRegression",
        feature_mode="Selected clinical variable only",
        selected_variable_col="biomarker_x",
        variable_cutoff=3.5,
        variable_subset="All samples",
        clinical_predictors=["age"],
        use_threshold_baseline=True,
        threshold_positive_class="Case",
        threshold_direction="At or above cutoff predicts positive class",
        use_cv=False,
    )

    assert result["metrics"]["selected_variable"] == "biomarker_x"
    assert result["metrics"]["variable_subset"] == "All samples"
    assert "psa_column" not in result["metrics"]

    baseline = result["threshold_baseline"]
    assert baseline["variable"] == "biomarker_x"
    assert baseline["positive_class"] == "Case"
    assert baseline["negative_class"] == "Control"
    assert not baseline["confusion_matrix"].empty


def test_explicit_clinical_predictor_selection() -> None:
    aligned = build_aligned()

    result = train_ml_model(
        aligned,
        model_name="LogisticRegression",
        feature_mode="Clinical only",
        clinical_predictors=["age", "biomarker_x"],
        selected_variable_col="biomarker_x",
        use_threshold_baseline=False,
        use_cv=False,
    )

    assert result["metrics"]["n_input_features_before_pca"] == 2
    assert result["metrics"]["clinical_predictors"] == ["age", "biomarker_x"]


def test_empty_clinical_predictor_selection_is_rejected() -> None:
    aligned = build_aligned()

    try:
        train_ml_model(
            aligned,
            model_name="LogisticRegression",
            feature_mode="Clinical only",
            clinical_predictors=[],
            use_cv=False,
        )
    except ValueError as exc:
        assert "No usable numeric clinical predictors" in str(exc)
    else:
        raise AssertionError("Empty clinical-predictor selection should be rejected.")


def main() -> None:
    test_generic_selected_variable_and_threshold_rule()
    test_explicit_clinical_predictor_selection()
    test_empty_clinical_predictor_selection_is_rejected()
    print("Generic clinical-variable test passed.")


if __name__ == "__main__":
    main()
