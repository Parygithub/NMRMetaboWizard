from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from nmr_pipeline import (  # noqa: E402
    _integrate_bins,
    compact_samples_for_downstream,
)


def _reference_integrate(ppm, intensity, edges, method):
    order = np.argsort(ppm)
    ppm = np.asarray(ppm)[order]
    intensity = np.asarray(intensity)[order]
    values = []

    for left, right in zip(edges[:-1], edges[1:]):
        mask = (ppm >= left) & (ppm < right)
        if np.sum(mask) < 2:
            values.append(0.0)
        elif method == "rectangular":
            values.append(float(np.mean(intensity[mask]) * (right - left)))
        else:
            values.append(float(np.trapezoid(intensity[mask], ppm[mask])))

    return np.asarray(values)


def test_vectorized_binning_matches_reference():
    rng = np.random.default_rng(20260818)
    ppm = np.sort(rng.uniform(0.2, 10.0, 46_820))[::-1]
    intensity = rng.normal(size=ppm.size)
    edges = np.arange(0.2, 10.001, 0.01)

    for method in ("trapezoidal", "rectangular"):
        expected = _reference_integrate(ppm, intensity, edges, method)
        observed = _integrate_bins(ppm, intensity, edges, method)
        np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_large_cohort_compaction_retains_downstream_arrays():
    n = 1000
    sample = {
        "name": "sample_001",
        "sample_id": "sample_001",
        "folder": "synthetic",
        "acqus": {"SW_h": 8000.0},
        "raw_fid": np.zeros(n, dtype=np.complex128),
        "complex_spectrum": np.zeros(2 * n, dtype=np.complex128),
        "baseline_corrected": np.zeros(2 * n, dtype=float),
        "referenced_ppm": np.linspace(10, 0, 2 * n),
        "negative_zeroed": np.zeros(2 * n, dtype=float),
        "window_ppm": np.linspace(10, 0.2, n),
        "window_intensity": np.ones(n, dtype=float),
        "window_range": (0.2, 10.0),
        "log": [],
    }

    compact = compact_samples_for_downstream([sample])[0]

    assert compact["memory_compacted"] is True
    assert compact["raw_fid_points"] == n
    assert "raw_fid" not in compact
    assert "complex_spectrum" not in compact
    assert "baseline_corrected" not in compact
    assert compact["window_ppm"] is sample["window_ppm"]
    assert compact["window_intensity"] is sample["window_intensity"]
    assert compact["referenced_ppm"] is sample["referenced_ppm"]
    assert compact["negative_zeroed"] is sample["negative_zeroed"]


if __name__ == "__main__":
    test_vectorized_binning_matches_reference()
    test_large_cohort_compaction_retains_downstream_arrays()
    print("Large-cohort memory and vectorized-binning tests passed.")
