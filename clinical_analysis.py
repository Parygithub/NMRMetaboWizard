from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC


def read_clinical_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if path.suffix.lower() == ".xlsx":
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path, sep=None, engine="python")

    df.columns = [str(c).strip() for c in df.columns]

    # Remove fully empty rows that sometimes appear from spreadsheet exports.
    df = df.dropna(how="all").copy()

    return df


def _clean_sample_id(x) -> str:
    if pd.isna(x):
        return ""

    s = str(x).strip()

    if s.endswith(".0"):
        possible = s[:-2]
        if possible.isdigit():
            s = possible

    return s


def _match_key(x) -> str:
    s = _clean_sample_id(x)

    if s == "":
        return ""

    s = s.replace("\\", "/").split("/")[-1].strip().lower()

    for ext in [".zip", ".csv", ".tsv", ".txt"]:
        if s.endswith(ext):
            s = s[: -len(ext)]

    if s.isdigit():
        s = str(int(s))

    return s



def _standardize_class_labels(y: pd.Series) -> pd.Series:
    """
    Standardize class labels conservatively.

    Examples:
        PCa, Pca, pca -> the most frequent spelling among those values.

    Robust to Excel/pandas reading occasional labels as non-string values.
    Empty/NaN class labels are kept empty and are removed later by the alignment logic.
    """
    y = pd.Series(y).map(lambda value: "" if pd.isna(value) else str(value).strip())

    groups = {}
    for value in y:
        if value == "" or value.lower() == "nan":
            continue

        key = value.lower()
        if key not in groups:
            groups[key] = {}
        groups[key][value] = groups[key].get(value, 0) + 1

    mapping = {}
    for key, counts in groups.items():
        # Use most frequent spelling. Tie-break by earliest sorted value.
        canonical = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        for spelling in counts:
            mapping[spelling] = canonical

    return y.map(mapping).fillna(y)


def prepare_spectra_table(spectra: pd.DataFrame) -> pd.DataFrame:
    X = spectra.copy()

    possible_id_cols = ["sample_id", "study_id", "SampleID", "ID", "id"]
    for col in possible_id_cols:
        if col in X.columns:
            X = X.set_index(col)
            break

    X.index = [_clean_sample_id(i) for i in X.index]
    X = X[X.index != ""].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.dropna(axis=1, how="all")

    return X


def align_spectra_with_clinical(
    spectra: pd.DataFrame,
    clinical: pd.DataFrame,
    clinical_id_col: str,
    class_col: str,
) -> dict:
    if clinical_id_col not in clinical.columns:
        raise ValueError(f"Clinical ID column not found: {clinical_id_col}")

    if class_col not in clinical.columns:
        raise ValueError(f"Class column not found: {class_col}")

    X_all = prepare_spectra_table(spectra)

    clinical_df = clinical.copy()
    clinical_df[clinical_id_col] = clinical_df[clinical_id_col].map(_clean_sample_id)

    # Drop empty IDs and accidental repeated header rows inside the file.
    header_key = _match_key(clinical_id_col)
    clinical_df = clinical_df[clinical_df[clinical_id_col] != ""].copy()
    clinical_df = clinical_df[clinical_df[clinical_id_col].map(_match_key) != header_key].copy()

    spectra_key = pd.Series([_match_key(i) for i in X_all.index], index=X_all.index, name="_match_key")
    clinical_df["_match_key"] = clinical_df[clinical_id_col].map(_match_key)

    duplicated_clinical = clinical_df[clinical_df["_match_key"].duplicated(keep=False)]["_match_key"].unique().tolist()
    clinical_unique = clinical_df.drop_duplicates("_match_key", keep="first").copy()
    clinical_lookup = clinical_unique.set_index("_match_key", drop=False)

    rows = []
    matched_indices = []
    clinical_rows = []

    for nmr_id, key in spectra_key.items():
        if key in clinical_lookup.index:
            row = clinical_lookup.loc[key]
            matched_indices.append(nmr_id)
            clinical_rows.append(row)
            rows.append(
                {
                    "nmr_sample_id": nmr_id,
                    "clinical_id": row[clinical_id_col],
                    "match_key": key,
                    "class": row[class_col],
                }
            )

    if matched_indices:
        X = X_all.loc[matched_indices].copy()
        clinical_aligned = pd.DataFrame(clinical_rows).reset_index(drop=True)
        clinical_aligned.index = matched_indices
        clinical_aligned.index.name = "sample_id"

        y = clinical_aligned[class_col].astype(str).str.strip()
        y = _standardize_class_labels(y)
        clinical_aligned[class_col] = y.values

        non_empty = y.notna() & (y != "") & (y.str.lower() != "nan")

        X = X.loc[non_empty.values].copy()
        clinical_aligned = clinical_aligned.loc[non_empty.values].copy()
        y = y.loc[non_empty.values].copy()
    else:
        X = X_all.iloc[0:0].copy()
        clinical_aligned = clinical_unique.iloc[0:0].copy()
        y = pd.Series(dtype=str)

    sample_table = pd.DataFrame(rows)

    unmatched_spectra_keys = sorted(set(spectra_key.values) - set(clinical_unique["_match_key"]))
    unmatched_clinical_keys = sorted(set(clinical_unique["_match_key"]) - set(spectra_key.values))

    summary = {
        "n_spectra": int(len(X_all)),
        "n_clinical": int(len(clinical_df)),
        "n_matched_before_class_filter": int(len(rows)),
        "n_matched_with_class": int(len(X)),
        "n_features": int(X.shape[1]),
        "n_unmatched_spectra": int(len(unmatched_spectra_keys)),
        "n_unmatched_clinical": int(len(unmatched_clinical_keys)),
        "unmatched_spectra": unmatched_spectra_keys[:50],
        "unmatched_clinical": unmatched_clinical_keys[:50],
        "example_spectra_ids": list(X_all.index[:10]),
        "example_clinical_ids": list(clinical_df[clinical_id_col].head(10)),
        "duplicated_clinical_match_keys": duplicated_clinical[:50],
        "clinical_id_col": clinical_id_col,
        "class_col": class_col,
        "clinical_cols": [c for c in clinical_df.columns if c != "_match_key"],
        "feature_cols": list(X_all.columns),
    }

    return {
        "X": X,
        "y": y,
        "clinical_aligned": clinical_aligned,
        "sample_table": sample_table,
        "summary": summary,
        "feature_cols": list(X.columns),
    }


def merge_omics_clinical(
    omics: pd.DataFrame,
    clinical: pd.DataFrame,
    clinical_id_col: str,
    class_col: str,
) -> dict:
    return align_spectra_with_clinical(
        spectra=omics,
        clinical=clinical,
        clinical_id_col=clinical_id_col,
        class_col=class_col,
    )


def _require_aligned(aligned: dict):
    if aligned is None:
        raise ValueError("Clinical data have not been aligned yet.")

    X = aligned.get("X")
    y = aligned.get("y")

    if X is None or y is None:
        raise ValueError("Aligned object is missing X/y.")

    if len(X) == 0:
        summary = aligned.get("summary", {})
        raise ValueError(
            "There are 0 aligned samples with non-empty Class labels. "
            "Check that NMR sample IDs match clinical study_id values. "
            f"Example NMR IDs: {summary.get('example_spectra_ids', [])[:5]}; "
            f"Example clinical IDs: {summary.get('example_clinical_ids', [])[:5]}"
        )

    if X.shape[1] == 0:
        raise ValueError("There are 0 spectral features after alignment.")

    return X, y


def _impute_scale_X(X: pd.DataFrame):
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    Xs = pipe.fit_transform(X)
    return Xs, pipe


def _safe_n_components(requested: int, X: pd.DataFrame) -> int:
    requested = int(requested)
    max_allowed = min(int(X.shape[0]), int(X.shape[1]))
    if max_allowed < 1:
        raise ValueError("No usable samples/features.")
    return max(1, min(requested, max_allowed))


def class_counts(aligned: dict) -> pd.DataFrame:
    _X, y = _require_aligned(aligned)
    counts = y.astype(str).value_counts(dropna=False)
    return counts.rename_axis("class").reset_index(name="count")


def pca_scores(aligned: dict, n_components: int = 5) -> dict:
    X, y = _require_aligned(aligned)

    n_components = _safe_n_components(n_components, X)

    if n_components < 2:
        raise ValueError("PCA needs at least 2 components for score plotting.")

    Xs, prep = _impute_scale_X(X)

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(Xs)

    score_df = pd.DataFrame(
        scores,
        index=X.index,
        columns=[f"PC{i+1}" for i in range(n_components)],
    )
    score_df.insert(0, "sample_id", X.index.astype(str))
    score_df["class"] = y.values

    variance_df = pd.DataFrame(
        {
            "component": [f"PC{i+1}" for i in range(n_components)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "explained_variance_percent": pca.explained_variance_ratio_ * 100,
        }
    )

    loadings_df = pd.DataFrame(
        pca.components_.T,
        index=X.columns,
        columns=[f"PC{i+1}" for i in range(n_components)],
    ).reset_index(names="feature_ppm")

    return {
        "scores": score_df,
        "variance": variance_df,
        "loadings": loadings_df,
    }


def plsda_scores(aligned: dict, n_components: int = 3) -> dict:
    X, y = _require_aligned(aligned)

    le = LabelEncoder()
    y_int = le.fit_transform(y)

    if len(le.classes_) < 2:
        raise ValueError("PLS-DA needs at least 2 classes.")

    n_components = int(n_components)
    max_allowed = min(X.shape[0] - 1, X.shape[1], n_components)
    if max_allowed < 1:
        raise ValueError("PLS-DA needs at least 2 samples and 1 spectral feature.")
    n_components = max_allowed

    Xs, prep = _impute_scale_X(X)

    # One-hot Y for multi-class PLS-DA. For two classes, this also works.
    Y = np.eye(len(le.classes_))[y_int]

    pls = PLSRegression(n_components=n_components, scale=False)
    pls.fit(Xs, Y)

    scores = pls.x_scores_

    score_df = pd.DataFrame(
        scores,
        index=X.index,
        columns=[f"LV{i+1}" for i in range(n_components)],
    )
    score_df.insert(0, "sample_id", X.index.astype(str))
    score_df["class"] = y.values

    # Approximate X variance represented by each LV, useful for labeling but not identical to PCA variance.
    total_x_var = np.sum(np.var(Xs, axis=0, ddof=1))
    score_vars = np.var(scores, axis=0, ddof=1)
    if total_x_var == 0:
        x_var_ratio = np.zeros(n_components)
    else:
        x_var_ratio = score_vars / total_x_var

    variance_df = pd.DataFrame(
        {
            "component": [f"LV{i+1}" for i in range(n_components)],
            "approx_x_variance_ratio": x_var_ratio,
            "approx_x_variance_percent": x_var_ratio * 100,
        }
    )

    # Feature importance for PLS-DA.
    #
    # Different scikit-learn versions expose PLSRegression.coef_ with different
    # orientation for multi-output Y. To avoid shape errors, use x_weights_,
    # which is always shaped (n_features, n_components).
    weights = np.asarray(pls.x_weights_)

    if weights.ndim == 2 and weights.shape[0] == X.shape[1]:
        importance = np.mean(np.abs(weights), axis=1)
        importance_name = "plsda_abs_x_weight"
    else:
        # Fallback: try coef_ and orient it to match n_features.
        coef = np.asarray(pls.coef_)

        if coef.ndim == 2 and coef.shape[0] == X.shape[1]:
            importance = np.mean(np.abs(coef), axis=1)
        elif coef.ndim == 2 and coef.shape[1] == X.shape[1]:
            importance = np.mean(np.abs(coef), axis=0)
        else:
            importance = np.zeros(X.shape[1])

        importance_name = "plsda_abs_coefficient"

    vip_like = pd.DataFrame(
        {
            "feature_ppm": list(X.columns),
            importance_name: np.asarray(importance).ravel()[: X.shape[1]],
        }
    ).sort_values(importance_name, ascending=False)
    vip_like["rank"] = np.arange(1, len(vip_like) + 1)

    return {
        "scores": score_df,
        "variance": variance_df,
        "loadings": vip_like,
        "classes": list(le.classes_),
    }



def detect_pca_outliers(
    aligned: dict,
    n_components: int = 5,
    threshold: float = 3.0,
    method: str = "Hotelling T2",
    confidence: float = 0.95,
    groupwise: bool = True,
) -> dict:
    """
    Detect possible outliers from PCA scores.

    Methods:
    - "Hotelling T2": classic score-space T² statistic.
    - "Robust distance": PCA score distance converted to robust z-score.

    groupwise=True:
        Calculate thresholds within each class/cohort. This is often more useful
        when cohorts have real biological separation, because a valid sample from
        a minority/shifted class should not be flagged only because it differs
        from the global center.

    groupwise=False:
        Calculate one global threshold across all samples.
    """
    X, y = _require_aligned(aligned)

    n_components = _safe_n_components(n_components, X)
    n_components = min(n_components, max(1, X.shape[0] - 1))

    Xs, _prep = _impute_scale_X(X)
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(Xs)

    score_df = pd.DataFrame(scores, index=X.index, columns=[f"PC{i+1}" for i in range(n_components)])
    score_df.insert(0, "sample_id", X.index.astype(str))
    score_df["class"] = y.values

    method = str(method)
    confidence = float(confidence)
    threshold = float(threshold)

    rows = []
    outlier_score = pd.Series(index=X.index, dtype=float)
    outlier_flag = pd.Series(False, index=X.index, dtype=bool)

    if groupwise:
        groups = [(str(cls), np.where(y.values == cls)[0]) for cls in pd.Series(y.values).unique()]
    else:
        groups = [("all samples", np.arange(len(X)))]

    for group_name, idx in groups:
        idx = np.asarray(idx, dtype=int)
        n_group = len(idx)

        if n_group < 3:
            for ii in idx:
                rows.append(
                    {
                        "sample_id": str(X.index[ii]),
                        "class": str(y.iloc[ii]),
                        "group_used": group_name,
                        "method": method,
                        "outlier_score": np.nan,
                        "limit_or_threshold": np.nan,
                        "is_outlier": False,
                        "note": "Group too small for outlier statistics.",
                    }
                )
            continue

        group_scores_all = scores[idx, :]

        if method == "Hotelling T2":
            # Need group_n > a for the F limit. Reduce PCs if needed.
            a = min(n_components, max(1, n_group - 2))
            group_scores = group_scores_all[:, :a]
            center = np.mean(group_scores, axis=0)
            variances = np.var(group_scores, axis=0, ddof=1)
            variances[variances == 0] = np.nan

            t2 = np.nansum(((group_scores - center) ** 2) / variances, axis=1)

            if n_group > a:
                limit = (a * (n_group - 1) / (n_group - a)) * stats.f.ppf(confidence, a, n_group - a)
            else:
                limit = np.nan

            flagged = t2 > limit if np.isfinite(limit) else np.zeros_like(t2, dtype=bool)

            for local_i, ii in enumerate(idx):
                outlier_score.iloc[ii] = t2[local_i]
                outlier_flag.iloc[ii] = bool(flagged[local_i])
                rows.append(
                    {
                        "sample_id": str(X.index[ii]),
                        "class": str(y.iloc[ii]),
                        "group_used": group_name,
                        "method": "Hotelling T2",
                        "pcs_used": int(a),
                        "outlier_score": float(t2[local_i]),
                        "limit_or_threshold": float(limit) if np.isfinite(limit) else np.nan,
                        "is_outlier": bool(flagged[local_i]),
                        "note": "",
                    }
                )

        else:
            a = min(n_components, max(1, n_group - 1))
            group_scores = group_scores_all[:, :a]
            center = np.median(group_scores, axis=0)
            distances = np.sqrt(np.sum((group_scores - center) ** 2, axis=1))

            med = np.median(distances)
            mad = np.median(np.abs(distances - med))

            if mad == 0:
                robust_z = np.zeros_like(distances)
            else:
                robust_z = 0.6745 * (distances - med) / mad

            flagged = robust_z > threshold

            for local_i, ii in enumerate(idx):
                outlier_score.iloc[ii] = robust_z[local_i]
                outlier_flag.iloc[ii] = bool(flagged[local_i])
                rows.append(
                    {
                        "sample_id": str(X.index[ii]),
                        "class": str(y.iloc[ii]),
                        "group_used": group_name,
                        "method": "Robust distance",
                        "pcs_used": int(a),
                        "outlier_score": float(robust_z[local_i]),
                        "limit_or_threshold": float(threshold),
                        "pca_distance": float(distances[local_i]),
                        "is_outlier": bool(flagged[local_i]),
                        "note": "",
                    }
                )

    outlier_table = pd.DataFrame(rows)

    score_df["outlier_score"] = outlier_score.values
    score_df["is_outlier"] = outlier_flag.values

    sort_col = "outlier_score"
    outlier_table = outlier_table.sort_values(sort_col, ascending=False, na_position="last")

    outlier_ids = list(outlier_table.loc[outlier_table["is_outlier"], "sample_id"])

    return {
        "table": outlier_table,
        "scores": score_df,
        "threshold": confidence if method == "Hotelling T2" else threshold,
        "method": method,
        "groupwise": bool(groupwise),
        "n_components": int(n_components),
        "n_outliers": int(len(outlier_ids)),
        "outlier_ids": outlier_ids,
    }


def filter_aligned_samples(
    aligned: dict,
    outlier_ids: list[str],
) -> dict:
    """
    Return a copy of aligned object with outlier samples removed.
    """
    X, y = _require_aligned(aligned)

    outlier_ids = [str(x) for x in outlier_ids]
    keep_mask = ~X.index.astype(str).isin(outlier_ids)

    X_new = X.loc[keep_mask].copy()
    y_new = y.loc[keep_mask].copy()

    clinical_new = aligned["clinical_aligned"].copy()
    clinical_new = clinical_new.loc[X_new.index].copy()

    sample_table = aligned.get("sample_table", pd.DataFrame()).copy()
    if not sample_table.empty and "nmr_sample_id" in sample_table.columns:
        sample_table = sample_table[~sample_table["nmr_sample_id"].astype(str).isin(outlier_ids)].copy()

    summary = dict(aligned.get("summary", {}))
    summary["n_matched_with_class_before_outlier_removal"] = int(len(X))
    summary["n_removed_outliers"] = int(len(outlier_ids))
    summary["removed_outlier_ids"] = list(outlier_ids)
    summary["n_matched_with_class"] = int(len(X_new))
    summary["n_features"] = int(X_new.shape[1])

    return {
        "X": X_new,
        "y": y_new,
        "clinical_aligned": clinical_new,
        "sample_table": sample_table,
        "summary": summary,
        "feature_cols": list(X_new.columns),
    }



def univariate_feature_tests(aligned: dict, max_features: int = 50) -> pd.DataFrame:
    X, y = _require_aligned(aligned)
    classes = list(pd.Series(y).dropna().unique())

    if len(classes) < 2:
        return pd.DataFrame({"message": ["Need at least 2 classes for univariate tests."]})

    rows = []

    for feature in X.columns:
        values = pd.to_numeric(X[feature], errors="coerce")
        groups = [values[y == cls].dropna().values for cls in classes]
        groups = [g for g in groups if len(g) >= 2]

        if len(groups) < 2:
            continue

        try:
            if len(classes) == 2:
                stat, p_value = stats.ttest_ind(groups[0], groups[1], equal_var=False, nan_policy="omit")
                mean_1 = float(np.nanmean(groups[0]))
                mean_2 = float(np.nanmean(groups[1]))
                effect = mean_2 - mean_1
                test_name = "Welch t-test"
            else:
                stat, p_value = stats.f_oneway(*groups)
                means = [float(np.nanmean(g)) for g in groups]
                effect = max(means) - min(means)
                test_name = "ANOVA"
        except Exception:
            continue

        rows.append(
            {
                "feature_ppm": feature,
                "test": test_name,
                "statistic": float(stat) if np.isfinite(stat) else np.nan,
                "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                "effect_size_simple": effect,
                "abs_effect": abs(effect),
            }
        )

    out = pd.DataFrame(rows)

    if out.empty:
        return pd.DataFrame({"message": ["No valid features for univariate tests."]})

    out = out.sort_values(["p_value", "abs_effect"], ascending=[True, False])
    out["rank"] = np.arange(1, len(out) + 1)

    return out.head(int(max_features))


def clinical_correlation_matrix(aligned: dict) -> pd.DataFrame:
    _X, _y = _require_aligned(aligned)
    clinical = aligned["clinical_aligned"]
    class_col = aligned["summary"]["class_col"]

    use_cols = []

    for col in clinical.columns:
        if col in [class_col, "_match_key"]:
            continue

        numeric = pd.to_numeric(clinical[col], errors="coerce")
        if numeric.notna().sum() >= 3:
            use_cols.append(col)

    if len(use_cols) < 2:
        return pd.DataFrame()

    return clinical[use_cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")


def top_feature_clinical_correlations(aligned: dict, max_rows: int = 50) -> pd.DataFrame:
    X, _y = _require_aligned(aligned)
    clinical = aligned["clinical_aligned"]
    class_col = aligned["summary"]["class_col"]

    rows = []
    clinical_numeric_cols = []

    for col in clinical.columns:
        if col in [class_col, "_match_key"]:
            continue

        numeric = pd.to_numeric(clinical[col], errors="coerce")
        if numeric.notna().sum() >= 5:
            clinical_numeric_cols.append(col)

    for clinical_col in clinical_numeric_cols:
        c = pd.to_numeric(clinical[clinical_col], errors="coerce")

        for feature in X.columns:
            x = pd.to_numeric(X[feature], errors="coerce")
            valid = c.notna() & x.notna()

            if valid.sum() < 5:
                continue

            try:
                rho, p_value = stats.spearmanr(c[valid], x[valid])
            except Exception:
                continue

            if np.isfinite(rho):
                rows.append(
                    {
                        "clinical_variable": clinical_col,
                        "feature_ppm": feature,
                        "spearman_rho": float(rho),
                        "abs_rho": float(abs(rho)),
                        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                    }
                )

    out = pd.DataFrame(rows)

    if out.empty:
        return pd.DataFrame({"message": ["No valid clinical-feature correlations found."]})

    return out.sort_values(["abs_rho", "p_value"], ascending=[False, True]).head(int(max_rows))


def _parse_hidden_layers(hidden_layers) -> tuple:
    if hidden_layers is None:
        return (64, 32)

    if isinstance(hidden_layers, (list, tuple)):
        return tuple(int(x) for x in hidden_layers if int(x) > 0) or (64, 32)

    parts = str(hidden_layers).replace(";", ",").replace(" ", "").split(",")
    vals = []
    for part in parts:
        if part == "":
            continue
        try:
            value = int(part)
            if value > 0:
                vals.append(value)
        except Exception:
            pass

    return tuple(vals) if vals else (64, 32)


def make_classifier(
    model_name: str,
    random_state: int = 123,
    ann_hidden_layers="64,32",
    ann_activation: str = "relu",
    ann_alpha: float = 0.0001,
    ann_learning_rate: float = 0.001,
    ann_max_iter: int = 500,
    ann_early_stopping: bool = True,
):
    if model_name == "LogisticRegression":
        # lbfgs supports multiclass classification. Do not pass multi_class,
        # because newer scikit-learn versions removed/deprecated that argument.
        return LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="lbfgs",
        )

    if model_name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=500,
            random_state=random_state,
            class_weight="balanced_subsample",
        )

    if model_name == "LinearSVM":
        return LinearSVC(
            class_weight="balanced",
            random_state=random_state,
            max_iter=10000,
        )

    if model_name == "ANN":
        return MLPClassifier(
            hidden_layer_sizes=_parse_hidden_layers(ann_hidden_layers),
            activation=str(ann_activation),
            alpha=float(ann_alpha),
            learning_rate_init=float(ann_learning_rate),
            max_iter=int(ann_max_iter),
            early_stopping=bool(ann_early_stopping),
            random_state=random_state,
        )

    raise ValueError("Model must be LogisticRegression, RandomForest, LinearSVM, or ANN.")


def _feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    feature_cols = [str(c) for c in feature_cols]

    if hasattr(model, "coef_"):
        vals = np.mean(np.abs(model.coef_), axis=0)
    elif hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "coefs_") and len(getattr(model, "coefs_", [])) > 0:
        # Approximate ANN/MLP feature importance from absolute first-layer weights.
        # This is a heuristic, but it is more informative than returning zeros.
        first_layer = np.asarray(model.coefs_[0])
        if first_layer.ndim == 2 and first_layer.shape[0] == len(feature_cols):
            vals = np.mean(np.abs(first_layer), axis=1)
        else:
            vals = np.zeros(len(feature_cols))
    else:
        vals = np.zeros(len(feature_cols))

    vals = np.asarray(vals).ravel()

    if len(vals) != len(feature_cols):
        vals = np.resize(vals, len(feature_cols))

    origins = []
    ppm_values = []

    for feature in feature_cols:
        clean = str(feature)
        if clean.startswith("nmr__"):
            clean = clean.replace("nmr__", "", 1)

        try:
            float(clean)
            origins.append("NMR")
            ppm_values.append(clean)
        except Exception:
            origins.append("Clinical")
            ppm_values.append(clean)

    out = pd.DataFrame(
        {
            "feature_ppm": ppm_values,
            "feature": feature_cols,
            "feature_origin": origins,
            "importance": vals,
        }
    )

    out = out.sort_values("importance", ascending=False)
    out["rank"] = np.arange(1, len(out) + 1)

    return out


def _find_column_case_insensitive(columns, requested: str | None):
    if requested is None:
        requested = ""

    requested = str(requested).strip()

    if requested in columns:
        return requested

    lower_map = {str(c).lower(): c for c in columns}

    if requested.lower() in lower_map:
        return lower_map[requested.lower()]

    if requested.lower() in ["", "auto", "psa"]:
        for c in columns:
            if str(c).strip().lower() == "psa":
                return c

        for c in columns:
            if "psa" in str(c).strip().lower():
                return c

    return None


def _numeric_clinical_features(aligned: dict, psa_only: bool = False, psa_col: str | None = "psa") -> pd.DataFrame:
    _X, _y = _require_aligned(aligned)
    clinical = aligned.get("clinical_aligned", pd.DataFrame()).copy()
    class_col = aligned.get("summary", {}).get("class_col", "Class")
    clinical_id_col = aligned.get("summary", {}).get("clinical_id_col", "")

    exclude = {class_col, clinical_id_col, "_match_key"}

    if psa_only:
        found = _find_column_case_insensitive(clinical.columns, psa_col)
        if found is None:
            raise ValueError("PSA-only model requested, but no PSA column was found.")
        use_cols = [found]
    else:
        use_cols = []
        for col in clinical.columns:
            if col in exclude:
                continue
            numeric = pd.to_numeric(clinical[col], errors="coerce")
            if numeric.notna().sum() >= 2:
                use_cols.append(col)

    if not use_cols:
        raise ValueError("No usable numeric clinical variables were found.")

    out = pd.DataFrame(index=clinical.index)

    for col in use_cols:
        out[f"clinical__{col}"] = pd.to_numeric(clinical[col], errors="coerce")

    out = out.dropna(axis=1, how="all")

    if out.shape[1] == 0:
        raise ValueError("No usable numeric clinical variables remained after cleaning.")

    return out


def _subset_aligned_by_psa(aligned: dict, psa_col: str | None = "psa", psa_cutoff: float = 4.0, psa_subset: str = "All samples") -> dict:
    subset = str(psa_subset)

    if subset == "All samples":
        return aligned

    X, y = _require_aligned(aligned)
    clinical = aligned.get("clinical_aligned", pd.DataFrame()).copy()

    found = _find_column_case_insensitive(clinical.columns, psa_col)

    if found is None:
        raise ValueError("PSA subset was requested, but no PSA column was found.")

    psa = pd.to_numeric(clinical[found], errors="coerce")

    if "Low PSA" in subset:
        keep = psa < float(psa_cutoff)
    elif "High PSA" in subset:
        keep = psa >= float(psa_cutoff)
    else:
        keep = pd.Series(True, index=clinical.index)

    keep = keep.fillna(False)

    filtered = dict(aligned)
    filtered["X"] = X.loc[keep.values].copy()
    filtered["y"] = y.loc[keep.values].copy()
    filtered["clinical_aligned"] = clinical.loc[keep.values].copy()

    summary = dict(aligned.get("summary", {}))
    summary["psa_subset"] = subset
    summary["psa_col"] = found
    summary["psa_cutoff"] = float(psa_cutoff)
    summary["n_matched_with_class"] = int(len(filtered["X"]))
    filtered["summary"] = summary

    return filtered


def _make_model_features(
    aligned: dict,
    feature_mode: str = "NMR only",
    psa_col: str | None = "psa",
) -> pd.DataFrame:
    X, _y = _require_aligned(aligned)
    feature_mode = str(feature_mode)

    X_nmr = X.copy()
    X_nmr.columns = [str(c) for c in X_nmr.columns]

    if feature_mode == "NMR only":
        return X_nmr

    if feature_mode == "Clinical only":
        return _numeric_clinical_features(aligned, psa_only=False, psa_col=psa_col)

    if feature_mode == "PSA only":
        return _numeric_clinical_features(aligned, psa_only=True, psa_col=psa_col)

    if feature_mode == "NMR + clinical":
        clinical = _numeric_clinical_features(aligned, psa_only=False, psa_col=psa_col)
        nmr = X_nmr.copy()
        nmr.columns = [f"nmr__{c}" for c in nmr.columns]
        return pd.concat([nmr, clinical], axis=1)

    raise ValueError("Feature mode must be NMR only, Clinical only, NMR + clinical, or PSA only.")


def _psa_cutoff_baseline(
    aligned: dict,
    psa_col: str | None = "psa",
    psa_cutoff: float = 4.0,
) -> dict:
    """Simple PSA cutoff baseline for binary BPH/PCa-like labels."""
    _X, y = _require_aligned(aligned)
    clinical = aligned.get("clinical_aligned", pd.DataFrame()).copy()
    found = _find_column_case_insensitive(clinical.columns, psa_col)

    if found is None:
        return {"note": "No PSA column found."}

    if y.nunique() != 2:
        return {"note": "PSA cutoff baseline is shown only for two-class problems."}

    classes = list(pd.Series(y).astype(str).unique())

    positive = None
    for c in classes:
        cl = str(c).lower()
        if "pca" in cl or "cancer" in cl or cl == "pc":
            positive = c
            break

    if positive is None:
        positive = classes[-1]

    negative = [c for c in classes if c != positive][0]

    psa = pd.to_numeric(clinical[found], errors="coerce")
    valid = psa.notna() & pd.Series(y.values, index=y.index).notna()

    if valid.sum() < 2:
        return {"note": "Not enough valid PSA values for PSA cutoff baseline."}

    y_true = y.loc[valid.values].astype(str)
    y_pred = pd.Series(np.where(psa.loc[valid.values] >= float(psa_cutoff), positive, negative), index=y_true.index)

    le = LabelEncoder()
    le.fit([negative, positive])

    true_enc = le.transform(y_true)
    pred_enc = le.transform(y_pred)

    cm = pd.DataFrame(confusion_matrix(true_enc, pred_enc), index=le.classes_, columns=le.classes_)

    return {
        "psa_col": found,
        "psa_cutoff": float(psa_cutoff),
        "positive_class": positive,
        "negative_class": negative,
        "n_samples": int(valid.sum()),
        "accuracy": float(accuracy_score(true_enc, pred_enc)),
        "balanced_accuracy": float(balanced_accuracy_score(true_enc, pred_enc)),
        "confusion_matrix": cm,
    }


def _make_ml_pipeline(
    model_name: str,
    random_state: int = 123,
    use_pca: bool = False,
    pca_components: int = 10,
    n_samples: int | None = None,
    n_features: int | None = None,
    ann_hidden_layers="64,32",
    ann_activation: str = "relu",
    ann_alpha: float = 0.0001,
    ann_learning_rate: float = 0.001,
    ann_max_iter: int = 500,
    ann_early_stopping: bool = True,
):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]

    if use_pca:
        max_allowed = pca_components
        if n_samples is not None and n_features is not None:
            max_allowed = min(int(pca_components), max(1, int(n_samples) - 1), int(n_features))
        steps.append(("pca", PCA(n_components=max_allowed)))

    steps.append((
        "classifier",
        make_classifier(
            model_name,
            random_state=random_state,
            ann_hidden_layers=ann_hidden_layers,
            ann_activation=ann_activation,
            ann_alpha=ann_alpha,
            ann_learning_rate=ann_learning_rate,
            ann_max_iter=ann_max_iter,
            ann_early_stopping=ann_early_stopping,
        ),
    ))

    return Pipeline(steps=steps)


def train_ml_model(
    aligned: dict,
    model_name: str = "LogisticRegression",
    test_size: float = 0.25,
    cv_folds: int = 5,
    random_state: int = 123,
    use_pca: bool = False,
    pca_components: int = 10,
    feature_mode: str = "NMR only",
    psa_col: str | None = "psa",
    psa_cutoff: float = 4.0,
    psa_subset: str = "All samples",
    ann_hidden_layers="64,32",
    ann_activation: str = "relu",
    ann_alpha: float = 0.0001,
    ann_learning_rate: float = 0.001,
    ann_max_iter: int = 500,
    ann_early_stopping: bool = True,
    use_cv: bool = True,
) -> dict:
    aligned_used = _subset_aligned_by_psa(
        aligned,
        psa_col=psa_col,
        psa_cutoff=psa_cutoff,
        psa_subset=psa_subset,
    )

    _X_spectral, y = _require_aligned(aligned_used)
    X = _make_model_features(aligned_used, feature_mode=feature_mode, psa_col=psa_col)

    if len(y.unique()) < 2:
        raise ValueError("Need at least two classes for ML after filtering/subsetting.")

    if X.shape[1] == 0:
        raise ValueError("No usable features for ML.")

    class_counts_ = y.value_counts()
    min_class = int(class_counts_.min())

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    effective_pca_components = min(
        int(pca_components),
        max(1, int(X.shape[0]) - 1),
        int(X.shape[1]),
    )

    pipe = _make_ml_pipeline(
        model_name,
        random_state=random_state,
        use_pca=use_pca,
        pca_components=pca_components,
        n_samples=X.shape[0],
        n_features=X.shape[1],
        ann_hidden_layers=ann_hidden_layers,
        ann_activation=ann_activation,
        ann_alpha=ann_alpha,
        ann_learning_rate=ann_learning_rate,
        ann_max_iter=ann_max_iter,
        ann_early_stopping=ann_early_stopping,
    )

    metrics = {
        "model": model_name,
        "feature_mode": feature_mode,
        "n_samples": int(len(X)),
        "n_input_features_before_pca": int(X.shape[1]),
        "n_model_features": int(effective_pca_components if use_pca else X.shape[1]),
        "classes": list(le.classes_),
        "feature_reduction": f"PCA({effective_pca_components})" if use_pca else "none",
        "psa_subset": psa_subset,
        "psa_column": psa_col,
        "psa_cutoff": float(psa_cutoff),
        "cross_validation_requested": bool(use_cv),
    }

    if model_name == "ANN":
        metrics["ann_hidden_layers"] = str(ann_hidden_layers)
        metrics["ann_activation"] = str(ann_activation)
        metrics["ann_alpha"] = float(ann_alpha)
        metrics["ann_learning_rate"] = float(ann_learning_rate)
        metrics["ann_max_iter"] = int(ann_max_iter)
        metrics["ann_early_stopping"] = bool(ann_early_stopping)

    cm_df = pd.DataFrame()
    report = "Holdout test was skipped."
    test_predictions = pd.DataFrame()
    probability_df = pd.DataFrame()
    roc_df = pd.DataFrame()

    can_holdout = min_class >= 2 and len(X) >= 4

    if can_holdout:
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y_encoded,
                test_size=float(test_size),
                random_state=random_state,
                stratify=y_encoded,
            )

            pipe.fit(X_train, y_train)

            train_pred = pipe.predict(X_train)
            metrics["train_accuracy"] = float(accuracy_score(y_train, train_pred))
            metrics["train_balanced_accuracy"] = float(balanced_accuracy_score(y_train, train_pred))

            pred = pipe.predict(X_test)

            true_labels = le.inverse_transform(y_test)
            pred_labels = le.inverse_transform(pred)

            test_predictions = pd.DataFrame(
                {
                    "sample_id": X_test.index.astype(str),
                    "true_class": true_labels,
                    "predicted_class": pred_labels,
                    "correct": true_labels == pred_labels,
                }
            )

            metrics["test_accuracy"] = float(accuracy_score(y_test, pred))
            metrics["test_balanced_accuracy"] = float(balanced_accuracy_score(y_test, pred))

            if hasattr(pipe, "predict_proba"):
                try:
                    proba = pipe.predict_proba(X_test)

                    probability_df = pd.DataFrame(
                        proba,
                        columns=[f"prob_{cls}" for cls in le.classes_],
                    )
                    probability_df.insert(0, "sample_id", X_test.index.astype(str))
                    probability_df.insert(1, "true_class", true_labels)
                    probability_df.insert(2, "predicted_class", pred_labels)

                    if len(le.classes_) == 2:
                        metrics["test_roc_auc"] = float(roc_auc_score(y_test, proba[:, 1]))
                        fpr, tpr, _thresholds = roc_curve(y_test, proba[:, 1])
                        roc_df = pd.DataFrame(
                            {
                                "class": le.classes_[1],
                                "fpr": fpr,
                                "tpr": tpr,
                                "auc": auc(fpr, tpr),
                            }
                        )
                    else:
                        metrics["test_roc_auc_ovr"] = float(
                            roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
                        )

                        roc_rows = []
                        for class_index, class_label in enumerate(le.classes_):
                            y_binary = (y_test == class_index).astype(int)

                            if len(np.unique(y_binary)) < 2:
                                continue

                            fpr, tpr, _thresholds = roc_curve(y_binary, proba[:, class_index])
                            class_auc = auc(fpr, tpr)

                            for f, t in zip(fpr, tpr):
                                roc_rows.append(
                                    {
                                        "class": class_label,
                                        "fpr": f,
                                        "tpr": t,
                                        "auc": class_auc,
                                    }
                                )

                        roc_df = pd.DataFrame(roc_rows)
                except Exception as e:
                    metrics["roc_note"] = f"ROC/probability output skipped because: {e}"

            cm = confusion_matrix(y_test, pred)
            cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

            report = classification_report(
                y_test,
                pred,
                target_names=le.classes_,
                output_dict=False,
                zero_division=0,
            )
        except Exception as e:
            metrics["test_note"] = f"Holdout test skipped because: {e}"
            cm_df = pd.DataFrame()
            report = "Holdout test was skipped."

    else:
        metrics["test_note"] = "Holdout test skipped: not enough samples per class."

    if not use_cv:
        n_splits = 0
        cv_summary = {
            "cv_folds_used": 0,
            "note": "Cross-validation skipped by user.",
        }
    else:
        n_splits = min(int(cv_folds), min_class)

    if use_cv and n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        scoring = ["accuracy", "balanced_accuracy"]

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_results = cross_validate(
                    pipe,
                    X,
                    y_encoded,
                    cv=cv,
                    scoring=scoring,
                    n_jobs=None,
                    error_score=np.nan,
                )

            cv_summary = {
                "cv_folds_used": int(n_splits),
                "cv_accuracy_mean": float(np.nanmean(cv_results["test_accuracy"])),
                "cv_accuracy_sd": float(np.nanstd(cv_results["test_accuracy"])),
                "cv_balanced_accuracy_mean": float(np.nanmean(cv_results["test_balanced_accuracy"])),
                "cv_balanced_accuracy_sd": float(np.nanstd(cv_results["test_balanced_accuracy"])),
            }
        except Exception as e:
            cv_summary = {
                "cv_folds_used": int(n_splits),
                "note": f"Cross-validation skipped because: {e}",
            }
    elif use_cv:
        cv_summary = {
            "cv_folds_used": 0,
            "note": "Not enough samples per class for cross-validation.",
        }

    final_pipe = _make_ml_pipeline(
        model_name,
        random_state=random_state,
        use_pca=False,
        n_samples=X.shape[0],
        n_features=X.shape[1],
        ann_hidden_layers=ann_hidden_layers,
        ann_activation=ann_activation,
        ann_alpha=ann_alpha,
        ann_learning_rate=ann_learning_rate,
        ann_max_iter=ann_max_iter,
        ann_early_stopping=ann_early_stopping,
    )
    final_pipe.fit(X, y_encoded)

    final_model = final_pipe.named_steps["classifier"]
    importance = _feature_importance(final_model, list(X.columns))

    psa_baseline = _psa_cutoff_baseline(
        aligned_used,
        psa_col=psa_col,
        psa_cutoff=psa_cutoff,
    )

    return {
        "metrics": metrics,
        "cv_summary": cv_summary,
        "confusion_matrix": cm_df,
        "classification_report": report,
        "feature_importance": importance,
        "test_predictions": test_predictions,
        "probabilities": probability_df,
        "roc_curve": roc_df,
        "psa_baseline": psa_baseline,
    }


def metrics_to_text(result: dict) -> str:
    lines = []

    lines.append("Model metrics")
    lines.append("=============")

    for k, v in result["metrics"].items():
        lines.append(f"{k}: {v}")

    lines.append("")
    lines.append("Cross-validation")
    lines.append("================")

    for k, v in result["cv_summary"].items():
        lines.append(f"{k}: {v}")

    psa = result.get("psa_baseline", {})
    lines.append("")
    lines.append("PSA cutoff baseline")
    lines.append("===================")

    if psa:
        note = psa.get("note")
        if note:
            lines.append(note)
        else:
            for k, v in psa.items():
                if k == "confusion_matrix":
                    continue
                lines.append(f"{k}: {v}")

            cm = psa.get("confusion_matrix")
            if cm is not None and not cm.empty:
                lines.append("")
                lines.append("PSA baseline confusion matrix:")
                lines.append(cm.to_string())
    else:
        lines.append("No PSA baseline calculated.")

    lines.append("")
    lines.append("Classification report")
    lines.append("=====================")
    lines.append(result["classification_report"])

    return "\n".join(lines)

