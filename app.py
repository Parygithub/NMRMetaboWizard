from __future__ import annotations

import traceback
import re
import io
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly

from nmr_pipeline import (
    load_zip_and_read_fids,
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
    time_axis,
)

from clinical_analysis import (
    read_clinical_file,
    merge_omics_clinical,
    class_counts,
    pca_scores,
    plsda_scores,
    univariate_feature_tests,
    clinical_correlation_matrix,
    top_feature_clinical_correlations,
    detect_pca_outliers,
    filter_aligned_samples,
    train_ml_model,
    metrics_to_text,
)


DEMO_RANDOM_SEED = 20260717
DEMO_SW_H = 7200.0
DEMO_O1 = 2820.0
DEMO_SFO1 = 600.13
DEMO_N_POINTS = 4096


def _demo_sample_rows() -> list[dict]:
    """Create deterministic, fully synthetic clinical metadata."""
    rng = np.random.default_rng(DEMO_RANDOM_SEED)
    rows = []

    for class_label in ["BPH", "PCa"]:
        for index in range(1, 9):
            sample_id = f"demo_{class_label}_{index:02d}"

            if class_label == "BPH":
                age = int(rng.integers(55, 79))
                psa = float(np.clip(rng.normal(5.1, 1.5), 1.0, 10.0))
                trus = float(np.clip(rng.normal(58.0, 12.0), 25.0, 100.0))
            else:
                age = int(rng.integers(57, 81))
                psa = float(np.clip(rng.normal(7.8, 2.0), 2.5, 15.0))
                trus = float(np.clip(rng.normal(42.0, 10.0), 20.0, 85.0))

            rows.append(
                {
                    "study_id": sample_id,
                    "Class": class_label,
                    "age": age,
                    "psa": round(psa, 2),
                    "height": int(rng.integers(165, 188)),
                    "weight": int(rng.integers(65, 101)),
                    "krea": int(rng.integers(65, 111)),
                    "trus_volume_v2": round(trus, 1),
                }
            )

    return rows


def _demo_fid_bytes(class_label: str, sample_index: int) -> bytes:
    """Generate one deterministic but visibly distinct synthetic complex FID.

    Each sample receives its own dilution factor, peak amplitudes, small
    chemical-shift offsets, linewidths, phases, water residual, and noise.
    The generated data are for software demonstration only.
    """
    seed_offset = 0 if class_label == "BPH" else 1000
    rng = np.random.default_rng(
        DEMO_RANDOM_SEED + seed_offset + int(sample_index)
    )

    time_s = np.arange(DEMO_N_POINTS, dtype=float) / DEMO_SW_H

    peak_ppm = np.array(
        [0.00, 0.92, 1.31, 1.65, 2.06, 2.36, 2.75, 3.03,
         3.22, 3.56, 4.12, 6.65, 7.15, 8.05],
        dtype=float,
    )
    base_amplitudes = np.array(
        [0.25, 0.55, 0.70, 0.12, 0.42, 0.32, 0.10, 0.58,
         0.48, 0.37, 0.26, 0.08, 0.14, 0.10],
        dtype=float,
    )
    base_linewidth_hz = np.array(
        [1.2, 2.0, 2.5, 2.8, 2.2, 2.0, 3.0, 2.4,
         2.1, 2.6, 2.0, 3.2, 3.0, 3.5],
        dtype=float,
    )

    # Distributed class effects create a useful but non-trivial example.
    if class_label == "PCa":
        class_effect = np.array(
            [1.00, 1.22, 0.78, 1.35, 1.28, 1.12, 1.30,
             0.76, 1.24, 0.88, 1.18, 1.25, 1.18, 0.82]
        )
    else:
        class_effect = np.array(
            [1.00, 0.82, 1.22, 0.72, 0.78, 0.94, 0.75,
             1.24, 0.80, 1.16, 0.86, 0.78, 0.84, 1.18]
        )

    dilution_factor = rng.uniform(0.65, 1.45)
    peak_variation = rng.lognormal(mean=0.0, sigma=0.22, size=peak_ppm.size)
    amplitudes = (
        base_amplitudes
        * class_effect
        * peak_variation
        * dilution_factor
    )

    # Make the optional variability peaks differ markedly among samples.
    amplitudes[[3, 6, 11]] *= rng.uniform(0.15, 2.8, size=3)

    global_shift_ppm = rng.normal(0.0, 0.006)
    local_shift_ppm = rng.normal(0.0, 0.004, size=peak_ppm.size)
    shifted_ppm = peak_ppm + global_shift_ppm + local_shift_ppm

    linewidth_hz = (
        base_linewidth_hz
        * rng.uniform(0.75, 1.45, size=peak_ppm.size)
    )
    zero_order_phase = rng.normal(0.0, 0.12)

    fid = np.zeros(DEMO_N_POINTS, dtype=np.complex128)

    for ppm_value, amplitude, linewidth in zip(
        shifted_ppm, amplitudes, linewidth_hz
    ):
        frequency_hz = ppm_value * DEMO_SFO1 - DEMO_O1
        peak_phase = zero_order_phase + rng.normal(0.0, 0.04)
        fid += (
            amplitude
            * np.exp(-np.pi * linewidth * time_s)
            * np.exp(
                1j
                * (
                    2.0 * np.pi * frequency_hz * time_s
                    + peak_phase
                )
            )
        )

    # Variable broad water-like residual near 4.75 ppm.
    water_ppm = 4.75 + rng.normal(0.0, 0.025)
    water_frequency_hz = water_ppm * DEMO_SFO1 - DEMO_O1
    water_amplitude = rng.uniform(0.45, 2.20)
    water_linewidth = rng.uniform(8.0, 20.0)
    fid += (
        water_amplitude
        * np.exp(-np.pi * water_linewidth * time_s)
        * np.exp(
            1j
            * (
                2.0 * np.pi * water_frequency_hz * time_s
                + rng.normal(0.0, 0.10)
            )
        )
    )

    # A broad low-frequency component gives sample-specific early-FID shape.
    broad_frequency_hz = rng.uniform(-450.0, 450.0)
    fid += (
        rng.uniform(0.04, 0.18)
        * np.exp(-np.pi * rng.uniform(18.0, 45.0) * time_s)
        * np.exp(
            1j
            * (
                2.0 * np.pi * broad_frequency_hz * time_s
                + rng.uniform(-np.pi, np.pi)
            )
        )
    )

    noise_sd = rng.uniform(0.004, 0.014)
    fid += noise_sd * (
        rng.normal(size=DEMO_N_POINTS)
        + 1j * rng.normal(size=DEMO_N_POINTS)
    )

    integer_scale = 2.5e7 / max(
        float(np.max(np.abs(fid.real))),
        float(np.max(np.abs(fid.imag))),
        1e-12,
    )
    real = np.rint(fid.real * integer_scale).astype("<i4")
    imag = np.rint(fid.imag * integer_scale).astype("<i4")

    interleaved = np.empty(real.size * 2, dtype="<i4")
    interleaved[0::2] = real
    interleaved[1::2] = imag
    return interleaved.tobytes()


def _demo_acqus_text(sample_id: str) -> str:
    """Return the minimal Bruker acquisition parameters used by the importer."""
    return (
        f"##TITLE= NMRMetaboWizard synthetic demonstration {sample_id}\n"
        "##JCAMPDX= 5.00 Bruker JCAMP library\n"
        "##DATATYPE= NMR FID\n"
        "##ORIGIN= NMRMetaboWizard synthetic demonstration\n"
        "##OWNER= Public synthetic example\n"
        "##$BYTORDA= 0\n"
        "##$DTYPA= 0\n"
        f"##$TD= {DEMO_N_POINTS * 2}\n"
        f"##$SW_h= {DEMO_SW_H}\n"
        f"##$O1= {DEMO_O1}\n"
        f"##$SFO1= {DEMO_SFO1}\n"
        "##$GRPDLY= 0\n"
        "##END=\n"
    )


def _build_demo_nmr_zip() -> bytes:
    """Build the complete synthetic Bruker cohort directly in memory."""
    buffer = io.BytesIO()

    with zipfile.ZipFile(
        buffer,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        for row in _demo_sample_rows():
            sample_id = str(row["study_id"])
            class_label = str(row["Class"])
            sample_index = int(sample_id.rsplit("_", 1)[-1])
            experiment_root = f"{sample_id}/1"

            archive.writestr(
                f"{experiment_root}/fid",
                _demo_fid_bytes(class_label, sample_index),
            )
            archive.writestr(
                f"{experiment_root}/acqus",
                _demo_acqus_text(sample_id),
            )

    return buffer.getvalue()


def _build_demo_clinical_csv() -> bytes:
    """Build matching synthetic clinical metadata directly in memory."""
    table = pd.DataFrame(_demo_sample_rows())
    return table.to_csv(index=False).encode("utf-8")


STEPS = [
    ("upload", "1 Upload"),
    ("raw", "2 Raw FID"),
    ("group", "3 Group delay"),
    ("solvent", "4 Solvent residuals"),
    ("apod", "5 Apodization"),
    ("zero", "6 Zero filling"),
    ("fft", "7 Fourier"),
    ("phase", "8 Phase"),
    ("ref", "9 Reference"),
    ("base", "10 Baseline"),
    ("align", "11 Alignment"),
    ("neg", "12 Negative zeroing"),
    ("window", "13 Window"),
    ("region", "14 Region removal"),
    ("bin", "15 Binning"),
    ("norm", "16 Normalization"),
    ("clinical", "17 Clinical"),
    ("eda", "18 EDA"),
    ("outlier", "19 Outliers"),
    ("eda_filtered", "20 EDA after outlier removal"),
    ("ml", "21 ML"),
]


def step_number(step_id: str) -> int:
    for i, (sid, _label) in enumerate(STEPS):
        if sid == step_id:
            return i
    return 0


def sample_choices(samples: list[dict]) -> dict[str, str]:
    return {
        str(i): f"{i} — {sample['name']}"
        for i, sample in enumerate(samples)
    }


def get_sample_index(input, samples: list[dict]) -> int:
    if not samples:
        return 0

    try:
        value = int(input.sample_index())
    except Exception:
        value = 0

    return max(0, min(value, len(samples) - 1))


def as_real(x):
    return np.real(np.asarray(x))


BEFORE_COLOR = "#1f77b4"
AFTER_COLOR = "#d62728"
AUXILIARY_COLOR = "#ff7f0e"

PLOT_STYLE = {
    "font_family": "Arial",
    "font_size": 14,
    "title_size": 18,
    "axis_title_size": 15,
    "tick_size": 13,
    "legend_size": 13,
    "width": 950,
    "height": 0,
}


def _safe_int(value, default):
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_str(value, default):
    try:
        value = str(value)
        return value if value else default
    except Exception:
        return default


def _update_plot_style_from_input(input):
    """Read plot appearance controls and store them in PLOT_STYLE."""
    try:
        PLOT_STYLE["font_family"] = _safe_str(input.plot_font_family(), "Arial")
        PLOT_STYLE["font_size"] = _safe_int(input.plot_font_size(), 14)
        PLOT_STYLE["title_size"] = _safe_int(input.plot_title_size(), 18)
        PLOT_STYLE["axis_title_size"] = _safe_int(input.plot_axis_title_size(), 15)
        PLOT_STYLE["tick_size"] = _safe_int(input.plot_tick_size(), 13)
        PLOT_STYLE["legend_size"] = _safe_int(input.plot_legend_size(), 13)
        PLOT_STYLE["width"] = _safe_int(input.plot_width(), 950)
        PLOT_STYLE["height"] = _safe_int(input.plot_height(), 0)
    except Exception:
        pass


def _apply_global_plot_style(fig):
    """Apply global plot font, tick, legend and size settings to a Plotly figure."""
    if fig is None:
        return fig

    layout_updates = dict(
        font=dict(
            family=PLOT_STYLE.get("font_family", "Arial"),
            size=PLOT_STYLE.get("font_size", 14),
        ),
        title=dict(
            font=dict(
                family=PLOT_STYLE.get("font_family", "Arial"),
                size=PLOT_STYLE.get("title_size", 18),
            )
        ),
        legend=dict(
            font=dict(
                family=PLOT_STYLE.get("font_family", "Arial"),
                size=PLOT_STYLE.get("legend_size", 13),
            ),
            title_font=dict(
                family=PLOT_STYLE.get("font_family", "Arial"),
                size=PLOT_STYLE.get("legend_size", 13),
            ),
        ),
        margin=dict(l=80, r=40, t=80, b=70),
    )

    if PLOT_STYLE.get("width", 0) and int(PLOT_STYLE["width"]) > 0:
        layout_updates["width"] = int(PLOT_STYLE["width"])

    if PLOT_STYLE.get("height", 0) and int(PLOT_STYLE["height"]) > 0:
        layout_updates["height"] = int(PLOT_STYLE["height"])

    fig.update_layout(**layout_updates)

    axis_title_font = dict(
        family=PLOT_STYLE.get("font_family", "Arial"),
        size=PLOT_STYLE.get("axis_title_size", 15),
    )
    tick_font = dict(
        family=PLOT_STYLE.get("font_family", "Arial"),
        size=PLOT_STYLE.get("tick_size", 13),
    )

    fig.update_xaxes(title_font=axis_title_font, tickfont=tick_font)
    fig.update_yaxes(title_font=axis_title_font, tickfont=tick_font)

    # Plotly subplot titles are stored as annotations. Update them too,
    # so cohort-specific outlier subplots follow the global appearance controls.
    try:
        fig.update_annotations(
            font=dict(
                family=PLOT_STYLE.get("font_family", "Arial"),
                size=PLOT_STYLE.get("title_size", 18),
            )
        )
    except Exception:
        pass

    return fig



def _trace_line_style(trace: dict):
    """Consistent colors for before/after plots, even when a single trace is displayed."""
    if trace is None:
        return None

    if "color" in trace and trace["color"]:
        return dict(color=trace["color"])

    name = str(trace.get("name", "")).lower()

    if "after" in name or "corrected" in name or "processed" in name:
        return dict(color=AFTER_COLOR)

    if "before" in name or "raw" in name:
        return dict(color=BEFORE_COLOR)

    if "estimated" in name or "baseline" in name or "solvent" in name:
        return dict(color=AUXILIARY_COLOR)

    return None


def thin_xy(x, y, max_points=6000):
    """
    Display-only downsampling that preserves sharp peaks.

    Instead of taking every nth point, each bin contributes its minimum and
    maximum y value. This avoids the visual artifact where a peak seems to
    change height after a shift/alignment step.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    n = len(y)
    if n <= max_points:
        return x, y

    # Use two points per bin: local min and local max.
    n_bins = max(1, int(max_points // 2))
    edges = np.linspace(0, n, n_bins + 1, dtype=int)

    x_out = []
    y_out = []

    for left, right in zip(edges[:-1], edges[1:]):
        if right <= left:
            continue

        ys = y[left:right]
        xs = x[left:right]

        if len(ys) == 0:
            continue

        i_min = int(np.argmin(ys))
        i_max = int(np.argmax(ys))

        # Preserve order along x-axis.
        for idx in sorted(set([i_min, i_max])):
            x_out.append(xs[idx])
            y_out.append(ys[idx])

    return np.asarray(x_out), np.asarray(y_out)


def fast_plot(ax, x, y, *args, max_points=6000, **kwargs):
    x2, y2 = thin_xy(x, y, max_points=max_points)
    # Call the original Matplotlib Axes.plot method directly.
    # Do NOT call fast_plot again, otherwise it recurses forever.
    return type(ax).plot(ax, x2, y2, *args, **kwargs)


def _blank_plotly_global(message: str, height: int = 640):
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    fig.update_layout(height=height)
    return _apply_global_plot_style(fig)


def _plotly_line_figure(
    traces,
    title: str,
    x_title: str = "",
    y_title: str = "Intensity",
    reverse_x: bool = False,
    height: int = 650,
    view_range=None,
    x_tickformat: str | None = None,
    y_tickformat: str | None = ".3e",
    x_hoverformat: str = "",
    y_hoverformat: str = ":.6e",
):
    fig = go.Figure()

    for trace in traces:
        x2, y2 = thin_xy(trace["x"], trace["y"], max_points=8000)
        fig.add_trace(
            go.Scattergl(
                x=x2,
                y=y2,
                mode="lines",
                name=trace.get("name", ""),
                hovertemplate=f"{x_title}: %{{x{x_hoverformat}}}<br>{y_title}: %{{y{y_hoverformat}}}<extra>%{{fullData.name}}</extra>",
                line=_trace_line_style(trace),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=height,
        hovermode="x unified",
    )

    if x_tickformat is not None:
        fig.update_xaxes(tickformat=x_tickformat)

    if y_tickformat is not None:
        fig.update_yaxes(tickformat=y_tickformat)

    if reverse_x:
        if view_range is not None:
            lo, hi = view_range
            fig.update_xaxes(range=[hi, lo])
        else:
            fig.update_xaxes(autorange="reversed")

    return _apply_global_plot_style(fig)


def _plotly_stack_figure(
    rows,
    title: str = "",
    x_title: str = "",
    y_title: str = "Intensity",
    reverse_x: bool = False,
    height_per_row: int = 360,
    view_range=None,
    x_tickformat: str | None = None,
    y_tickformat: str | None = ".3e",
    x_hoverformat: str = "",
    y_hoverformat: str = ":.6e",
):
    nrows = len(rows)

    fig = make_subplots(
        rows=nrows,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.08,
        subplot_titles=[r.get("title", "") for r in rows],
    )

    for row_i, row in enumerate(rows, start=1):
        for trace in row.get("traces", []):
            x2, y2 = thin_xy(trace["x"], trace["y"], max_points=7000)
            fig.add_trace(
                go.Scattergl(
                    x=x2,
                    y=y2,
                    mode="lines",
                    name=trace.get("name", ""),
                    hovertemplate=f"{x_title}: %{{x{x_hoverformat}}}<br>{y_title}: %{{y{y_hoverformat}}}<extra>%{{fullData.name}}</extra>",
                    line=_trace_line_style(trace),
                ),
                row=row_i,
                col=1,
            )

        fig.update_yaxes(title_text=y_title, row=row_i, col=1)

        if y_tickformat is not None:
            fig.update_yaxes(tickformat=y_tickformat, row=row_i, col=1)

        if x_tickformat is not None:
            fig.update_xaxes(tickformat=x_tickformat, row=row_i, col=1)

        if reverse_x:
            if view_range is not None:
                lo, hi = view_range
                fig.update_xaxes(range=[hi, lo], row=row_i, col=1)
            else:
                fig.update_xaxes(autorange="reversed", row=row_i, col=1)

    fig.update_xaxes(title_text=x_title, row=nrows, col=1)
    if x_tickformat is not None:
        fig.update_xaxes(tickformat=x_tickformat, row=nrows, col=1)
    fig.update_layout(
        title=title,
        height=max(420, int(height_per_row * nrows)),
        hovermode="x unified",
    )
    # Keep before/after subplots independent so the user can zoom one panel
    # without forcing the other panel to the same x/y range.
    fig.update_xaxes(matches=None)
    fig.update_yaxes(matches=None)

    return _apply_global_plot_style(fig)


def _add_bin_overlays(fig, bin_edges, view_range=None, max_shapes: int = 140):
    """Add faint alternating rectangles showing bin/bucket positions."""
    if bin_edges is None:
        return fig

    try:
        edges = np.asarray(bin_edges, dtype=float)
    except Exception:
        return fig

    if edges.size < 2:
        return fig

    intervals = list(zip(edges[:-1], edges[1:]))

    if view_range is not None:
        lo, hi = view_range
        low, high = min(lo, hi), max(lo, hi)
        intervals = [(a, b) for a, b in intervals if max(a, b) >= low and min(a, b) <= high]

    if not intervals:
        return fig

    step = max(1, int(np.ceil(len(intervals) / max_shapes)))

    for j, (a, b) in enumerate(intervals[::step]):
        fill = "rgba(160,160,160,0.12)" if j % 2 == 0 else "rgba(230,230,230,0.06)"
        fig.add_vrect(
            x0=float(a),
            x1=float(b),
            fillcolor=fill,
            line_width=0,
            layer="below",
        )

    return fig


def _fid_seconds(sample: dict, fid) -> np.ndarray:
    """Return the FID time axis in seconds from the Bruker spectral width."""
    return time_axis(sample.get("acqus", {}), len(fid))


def _fid_plot_kwargs():
    return {
        "x_title": "Time (s)",
        "y_title": "Intensity",
        "x_tickformat": ".6f",
        "y_tickformat": ".3e",
        "x_hoverformat": ":.6f",
        "y_hoverformat": ":.6e",
    }


def _comparison_view(input=None, default: str = "Both") -> str:
    try:
        if input is not None:
            return str(input.comparison_view())
    except Exception:
        pass
    return default


def _plotly_fid_stack(before_x, before_y, after_x, after_y, title_before, title_after, mode: str = "Both"):
    mode = str(mode)
    kwargs = _fid_plot_kwargs()

    if mode == "Before only":
        return _plotly_line_figure(
            [{"x": before_x, "y": before_y, "name": "before"}],
            title_before,
            reverse_x=False,
            **kwargs,
        )

    if mode == "After only":
        return _plotly_line_figure(
            [{"x": after_x, "y": after_y, "name": "after"}],
            title_after,
            reverse_x=False,
            **kwargs,
        )

    if mode == "Overlay":
        return _plotly_line_figure(
            [
                {"x": before_x, "y": before_y, "name": "before"},
                {"x": after_x, "y": after_y, "name": "after"},
            ],
            "Before/after overlay",
            reverse_x=False,
            **kwargs,
        )

    return _plotly_stack_figure(
        [
            {"title": title_before, "traces": [{"x": before_x, "y": before_y, "name": "before"}]},
            {"title": title_after, "traces": [{"x": after_x, "y": after_y, "name": "after"}]},
        ],
        reverse_x=False,
        **kwargs,
    )


def _plotly_spectrum_stack(input, before_x, before_y, after_x, after_y, title_before, title_after):
    mode = _comparison_view(input)

    if mode == "Before only":
        return _plotly_line_figure(
            [{"x": before_x, "y": before_y, "name": "before"}],
            title_before,
            x_title="ppm",
            y_title="Intensity",
            reverse_x=True,
            view_range=get_view_range(input),
        )

    if mode == "After only":
        return _plotly_line_figure(
            [{"x": after_x, "y": after_y, "name": "after"}],
            title_after,
            x_title="ppm",
            y_title="Intensity",
            reverse_x=True,
            view_range=get_view_range(input),
        )

    if mode == "Overlay":
        return _plotly_line_figure(
            [
                {"x": before_x, "y": before_y, "name": "before"},
                {"x": after_x, "y": after_y, "name": "after"},
            ],
            "Before/after overlay",
            x_title="ppm",
            y_title="Intensity",
            reverse_x=True,
            view_range=get_view_range(input),
        )

    return _plotly_stack_figure(
        [
            {"title": title_before, "traces": [{"x": before_x, "y": before_y, "name": "before"}]},
            {"title": title_after, "traces": [{"x": after_x, "y": after_y, "name": "after"}]},
        ],
        x_title="ppm",
        y_title="Intensity",
        reverse_x=True,
        view_range=get_view_range(input),
    )


def safe_float(value, default):
    try:
        return float(value)
    except Exception:
        return float(default)


def get_view_range(input):
    lo = safe_float(input.view_ppm_min(), 0.2)
    hi = safe_float(input.view_ppm_max(), 10.0)
    return min(lo, hi), max(lo, hi)


def set_ppm_axis(ax, input):
    lo, hi = get_view_range(input)
    ax.set_xlim(hi, lo)
    ax.set_xlabel("ppm")


def plot_height():
    return "720px"


def comparison_view_control():
    return ui.input_select(
        "comparison_view",
        "Before/after plot view",
        choices=["Both", "Before only", "After only", "Overlay"],
        selected="Both",
    )


def sample_and_ppm_controls(choices):
    return ui.div(
        ui.layout_columns(
            ui.input_select("sample_index", "Sample index", choices=choices, selected="0"),
            ui.input_numeric("view_ppm_min", "Display min ppm", value=0.2),
            ui.input_numeric("view_ppm_max", "Display max ppm", value=10.0),
            col_widths=[6, 3, 3],
        ),
        comparison_view_control(),
        output_widget("main_plot"),
        ui.download_button("download_current_plot_data", "Download current plot data CSV"),
    )


def sample_only_controls(choices):
    return ui.div(
        ui.input_select("sample_index", "Sample index", choices=choices, selected="0"),
        comparison_view_control(),
        output_widget("main_plot"),
        ui.download_button("download_current_plot_data", "Download current plot data CSV"),
    )


def clinical_variable_choices(aligned, include_class=True, include_psa_groups=True):
    choices = ["Class"] if include_class else []

    if include_psa_groups:
        choices.extend(["PSA group", "Class + PSA group"])

    if aligned is None:
        return choices

    clinical = aligned.get("clinical_aligned")
    summary = aligned.get("summary", {})

    if clinical is None or clinical.empty:
        return choices

    class_col = summary.get("class_col", "Class")
    clinical_id_col = summary.get("clinical_id_col", "")

    for col in clinical.columns:
        if col in [class_col, clinical_id_col, "_match_key"]:
            continue
        if str(col) not in choices:
            choices.append(str(col))

    return choices


def numeric_clinical_choices(aligned, prefer_psa=True):
    choices = []

    if aligned is None:
        return ["psa"]

    clinical = aligned.get("clinical_aligned")
    summary = aligned.get("summary", {})

    if clinical is None or clinical.empty:
        return ["psa"]

    class_col = summary.get("class_col", "Class")
    clinical_id_col = summary.get("clinical_id_col", "")

    for col in clinical.columns:
        if col in [class_col, clinical_id_col, "_match_key"]:
            continue

        numeric = pd.to_numeric(clinical[col], errors="coerce")
        if numeric.notna().sum() >= 2:
            choices.append(str(col))

    if prefer_psa:
        for c in choices:
            if c.lower() == "psa":
                choices.remove(c)
                choices.insert(0, c)
                break
        else:
            for c in choices:
                if "psa" in c.lower():
                    choices.remove(c)
                    choices.insert(0, c)
                    break

    return choices or ["psa"]


def stacked_fig(nrows=2):
    fig, axes = plt.subplots(
        nrows,
        1,
        figsize=(11, 3.4 * nrows),
        constrained_layout=True,
    )
    if nrows == 1:
        axes = [axes]
    return fig, axes


def plot_fid_stack(before_x, before_y, after_x, after_y, title_before, title_after):
    fig, axes = stacked_fig(2)

    fast_plot(axes[0], before_x, before_y)
    axes[0].set_title(title_before, pad=10)
    axes[0].set_ylabel("Intensity")

    fast_plot(axes[1], after_x, after_y)
    axes[1].set_title(title_after, pad=10)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Intensity")

    return fig


def plot_spectrum_stack(input, before_x, before_y, after_x, after_y, title_before, title_after):
    fig, axes = stacked_fig(2)

    fast_plot(axes[0], before_x, before_y)
    axes[0].set_title(title_before, pad=10)
    axes[0].set_ylabel("Intensity")
    set_ppm_axis(axes[0], input)

    fast_plot(axes[1], after_x, after_y)
    axes[1].set_title(title_after, pad=10)
    axes[1].set_ylabel("Intensity")
    set_ppm_axis(axes[1], input)

    return fig


app_ui = ui.page_fluid(
    ui.tags.style(
        """
        body {
            background: #f5f7fb;
            color: #1b2430;
        }

        .top-hero {
            position: relative;
            overflow: hidden;
            background:
                linear-gradient(135deg, rgba(23,50,77,0.96), rgba(57,95,143,0.92)),
                radial-gradient(circle at 82% 25%, rgba(255,255,255,0.24), transparent 28%),
                radial-gradient(circle at 72% 78%, rgba(80,210,255,0.20), transparent 32%);
            color: white;
            padding: 30px 34px;
            border-radius: 22px;
            margin-top: 18px;
            margin-bottom: 18px;
            box-shadow: 0 12px 30px rgba(30, 55, 90, 0.18);
        }

        .top-hero::after {
            content: "";
            position: absolute;
            right: -18px;
            top: -12px;
            width: 420px;
            height: 170px;
            opacity: 0.33;
            pointer-events: none;
            background-repeat: no-repeat;
            background-size: contain;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='520' height='210' viewBox='0 0 520 210'%3E%3Cg fill='none' stroke='white' stroke-width='3' stroke-linecap='round' stroke-linejoin='round' opacity='0.95'%3E%3Cpath d='M8 142 C42 142 49 142 70 142 C84 142 87 75 96 75 C104 75 108 142 121 142 C141 142 142 115 151 115 C160 115 161 142 182 142 C203 142 208 28 219 28 C232 28 237 142 258 142 C282 142 285 101 296 101 C307 101 313 142 335 142 C357 142 361 128 370 128 C380 128 383 142 408 142 C435 142 439 86 451 86 C462 86 466 142 512 142'/%3E%3Ccircle cx='78' cy='42' r='13'/%3E%3Ccircle cx='116' cy='32' r='10'/%3E%3Ccircle cx='150' cy='52' r='11'/%3E%3Cpath d='M90 39 L107 34 M126 37 L141 47 M151 63 L151 86'/%3E%3C/g%3E%3C/svg%3E");
        }

        .top-hero h1, .top-hero p {
            position: relative;
            z-index: 1;
        }

        .top-hero h1 {
            margin: 0;
            font-weight: 800;
            letter-spacing: -0.03em;
        }

        .top-hero p {
            margin: 8px 0 0 0;
            font-size: 16px;
            opacity: 0.92;
        }

        .progress-wrap {
            background: white;
            padding: 14px;
            border-radius: 18px;
            box-shadow: 0 8px 22px rgba(30, 55, 90, 0.08);
            margin-bottom: 18px;
        }

        .step-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }

        .step-pill {
            padding: 8px 12px;
            border-radius: 999px;
            background: #edf1f7;
            color: #58677a;
            font-weight: 700;
            font-size: 13px;
        }

        .step-pill.done {
            background: #dff5e8;
            color: #197141;
        }

        .step-pill.current {
            background: #17324d;
            color: white;
        }

        .card {
            background: white;
            border: 1px solid #e6ebf2;
            border-radius: 22px;
            padding: 24px;
            margin-bottom: 18px;
            box-shadow: 0 10px 24px rgba(30, 55, 90, 0.08);
        }

        .step-title {
            margin-top: 0;
            font-weight: 800;
            letter-spacing: -0.02em;
        }

        .note {
            background: #fff7df;
            border-left: 5px solid #f2b705;
            padding: 12px 14px;
            border-radius: 12px;
            margin-bottom: 16px;
        }

        .good-note {
            background: #e9f8ef;
            border-left: 5px solid #2b9348;
            padding: 12px 14px;
            border-radius: 12px;
            margin-bottom: 16px;
        }

        .btn-primary {
            border-radius: 999px;
            padding-left: 18px;
            padding-right: 18px;
            font-weight: 800;
        }

        pre {
            white-space: pre-wrap;
            word-break: break-word;
        }
        """
    ),

    ui.div(
        ui.h1("NMRMetaboWizard"),
        ui.p("Interactive ¹H NMR metabolomics preprocessing, EDA, outlier handling, and machine learning."),
        class_="top-hero",
    ),

    ui.div(
        ui.output_ui("progress_bar"),
        class_="progress-wrap",
    ),

    ui.div(
        ui.h4("Plot appearance"),
        ui.layout_columns(
            ui.input_select("plot_font_family", "Font", choices=["Arial", "Calibri", "DejaVu Sans", "Times New Roman"], selected="Arial"),
            ui.input_numeric("plot_width", "Plot width px (0 = auto)", value=950, min=0),
            ui.input_numeric("plot_height", "Plot height px (0 = auto)", value=0, min=0),
            col_widths=[4, 4, 4],
        ),
        ui.layout_columns(
            ui.input_numeric("plot_title_size", "Title size", value=18, min=8),
            ui.input_numeric("plot_axis_title_size", "Axis-label size", value=15, min=8),
            ui.input_numeric("plot_tick_size", "Tick-number size", value=13, min=8),
            ui.input_numeric("plot_legend_size", "Legend size", value=13, min=8),
            ui.input_numeric("plot_font_size", "General font size", value=14, min=8),
            col_widths=[2, 2, 2, 2, 2],
        ),
        class_="card",
    ),

    ui.output_ui("main_step"),

    ui.layout_columns(
        ui.div(
            ui.h4("Status"),
            ui.output_text_verbatim("status_text"),
            class_="card",
        ),
        ui.div(
            ui.h4("Applied settings / processing log"),
            ui.output_text_verbatim("processing_settings_text"),
            class_="card",
        ),
        ui.div(
            ui.h4("Error messages"),
            ui.output_text_verbatim("error_text"),
            class_="card",
        ),
        col_widths=[4, 4, 4],
    ),
)


def server(input, output, session):
    current_step = reactive.Value("upload")
    samples_state = reactive.Value([])

    binned_state = reactive.Value(None)
    normalized_state = reactive.Value(None)

    clinical_state = reactive.Value(None)
    combined_state = reactive.Value(None)
    eda_state = reactive.Value(None)
    outlier_state = reactive.Value(None)
    filtered_state = reactive.Value(None)
    eda_filtered_state = reactive.Value(None)
    ml_state = reactive.Value(None)
    ml_history_state = reactive.Value(pd.DataFrame())

    status_state = reactive.Value("NMRMetaboWizard is ready. Upload a Bruker ZIP file.")
    error_state = reactive.Value("No error.")

    def set_error():
        error_state.set(traceback.format_exc())
        status_state.set("Error. Check the error box.")

    def clear_error():
        error_state.set("No error.")

    def require_samples():
        samples = samples_state.get()
        if not samples:
            raise ValueError("No samples loaded. Go to Step 1 and read a ZIP first.")
        return samples

    def reset_downstream_analysis():
        binned_state.set(None)
        normalized_state.set(None)
        clinical_state.set(None)
        combined_state.set(None)
        eda_state.set(None)
        outlier_state.set(None)
        filtered_state.set(None)
        eda_filtered_state.set(None)
        ml_state.set(None)

    @render.ui
    def progress_bar():
        current = current_step.get()
        current_i = step_number(current)

        groups = [
            ("Preprocessing", ["upload", "raw", "group", "solvent", "apod", "zero", "fft", "phase", "ref", "base", "align", "neg", "window", "region", "bin", "norm"]),
            ("Clinical labels", ["clinical"]),
            ("EDA and outliers", ["eda", "outlier", "eda_filtered"]),
            ("Machine learning", ["ml"]),
        ]

        step_lookup = {sid: label for sid, label in STEPS}
        cards = []

        for group_name, step_ids in groups:
            pills = []
            for sid in step_ids:
                label = step_lookup.get(sid, sid)
                i = step_number(sid)
                cls = "step-pill"
                if i < current_i:
                    cls += " done"
                elif sid == current:
                    cls += " current"
                pills.append(ui.tags.span(label, class_=cls))

            cards.append(
                ui.div(
                    ui.tags.div(group_name, style="font-weight:800; margin-bottom:8px; color:#17324d;"),
                    ui.div(*pills, class_="step-row"),
                    style="background:#ffffff; border:1px solid #e6ebf2; border-radius:16px; padding:12px; margin-bottom:10px;",
                )
            )

        return ui.div(*cards)

    @render.ui
    def main_step():
        step = current_step.get()
        samples = samples_state.get()

        if step == "upload":
            return ui.div(
                ui.h2("Step 1 — Upload ZIP", class_="step-title"),
                ui.div(
                    "Choose a ZIP file containing Bruker experiment folders. "
                    "The app searches for folders containing both `fid` and `acqus`. "
                    "Important: the detected sample/folder names are used later as study_id values and must match the study_id column in the clinical metadata.",
                    class_="note",
                ),
                ui.div(
                    ui.h4("Try NMRMetaboWizard with synthetic example data"),
                    ui.p(
                        "Download both files below. The NMR archive contains 16 "
                        "fully synthetic Bruker-like FIDs (8 BPH-labelled and "
                        "8 PCa-labelled samples), and the CSV contains matching "
                        "study_id and Class values. The files contain no patient data."
                    ),
                    ui.layout_columns(
                        ui.download_button(
                            "download_example_nmr",
                            "Download example NMR ZIP",
                        ),
                        ui.download_button(
                            "download_example_clinical",
                            "Download example clinical CSV",
                        ),
                        col_widths=[6, 6],
                    ),
                    ui.p(
                        "Upload the NMR ZIP here. At Step 17, upload the matching "
                        "clinical CSV and keep study_id as the ID column and Class "
                        "as the class column.",
                        style="margin-top:10px; margin-bottom:0;",
                    ),
                    class_="good-note",
                ),
                ui.div(
                    "Processing note: reading many raw NMR folders can take a little while.",
                    class_="note",
                ),
                ui.input_file("data_zip", "Bruker ZIP file", accept=[".zip"], multiple=False),
                ui.input_action_button("read_zip", "Read ZIP", class_="btn-primary"),
                class_="card",
            )

        if not samples:
            return ui.div(
                ui.h2("No data loaded", class_="step-title"),
                ui.p("Go back to Step 1 and upload a ZIP file."),
                ui.input_action_button("go_upload_empty", "Back to upload"),
                class_="card",
            )

        choices = sample_choices(samples)
        combined_for_choices = combined_state.get()
        filtered_for_choices = filtered_state.get() if filtered_state.get() is not None else combined_for_choices
        eda_color_choices = clinical_variable_choices(combined_for_choices)
        eda2_color_choices = clinical_variable_choices(filtered_for_choices)
        psa_column_choices = numeric_clinical_choices(combined_for_choices)
        psa2_column_choices = numeric_clinical_choices(filtered_for_choices)

        if step == "raw":
            return ui.div(
                ui.h2("Step 2 — Inspect raw FIDs", class_="step-title"),
                ui.div(
                    "Choose a sample index and inspect the raw FID. "
                    "For speed, the plot is display-downsampled, but the full FID is still used for processing.",
                    class_="good-note",
                ),
                sample_only_controls(choices),
                ui.h4("Detected experiments"),
                ui.output_data_frame("sample_table"),
                ui.hr(),
                ui.input_action_button("back_to_upload", "Back"),
                ui.input_action_button("go_group", "Next: group delay", class_="btn-primary"),
                class_="card",
            )

        if step == "group":
            return ui.div(
                ui.h2("Step 3 — Group delay removal", class_="step-title"),
                ui.div(
                    "Use -1 to use Bruker's GRPDLY value. Use 0 to remove nothing.",
                    class_="note",
                ),
                ui.input_numeric("group_delay_override", "Override points", value=-1),
                ui.input_action_button("apply_group", "Apply group delay", class_="btn-primary"),
                ui.input_action_button("skip_group", "Skip group delay", class_="btn-primary"),
                ui.hr(),
                sample_only_controls(choices),
                ui.hr(),
                ui.input_action_button("back_raw", "Back"),
                ui.input_action_button("go_solvent", "Next: solvent residuals", class_="btn-primary"),
                class_="card",
            )

        if step == "solvent":
            return ui.div(
                ui.h2("Step 4 — Solvent residuals removal", class_="step-title"),
                ui.div(
                    "This is done in the FID domain, after group delay and before apodization.",
                    class_="note",
                ),
                ui.input_checkbox("solvent_enabled", "Apply solvent residual suppression", value=True),
                ui.input_numeric("solvent_lambda", "Solvent smoother lambda", value=1e6, min=1),
                ui.div("This step may take a while for long FIDs or many samples.", class_="note"),
                ui.input_action_button("apply_solvent", "Apply solvent residual removal", class_="btn-primary"),
                ui.input_action_button("skip_solvent", "Skip solvent residual removal", class_="btn-primary"),
                ui.hr(),
                sample_only_controls(choices),
                ui.hr(),
                ui.input_action_button("back_group", "Back"),
                ui.input_action_button("go_apod", "Next: apodization", class_="btn-primary"),
                class_="card",
            )

        if step == "apod":
            return ui.div(
                ui.h2("Step 5 — Apodization", class_="step-title"),
                ui.div("Apodization applies a window to the FID before Fourier transform.", class_="note"),
                ui.input_select("apod_kind", "Apodization type", choices=["exponential", "gaussian"], selected="exponential"),
                ui.input_numeric("apod_lb", "LB parameter", value=1.0, min=0),
                ui.input_action_button("apply_apod", "Apply apodization", class_="btn-primary"),
                ui.input_action_button("skip_apod", "Skip apodization", class_="btn-primary"),
                ui.hr(),
                sample_only_controls(choices),
                ui.hr(),
                ui.input_action_button("back_solvent", "Back"),
                ui.input_action_button("go_zero", "Next: zero filling", class_="btn-primary"),
                class_="card",
            )

        if step == "zero":
            return ui.div(
                ui.h2("Step 6 — Zero filling", class_="step-title"),
                ui.div("Zero filling adds this many zeros to the end of the FID.", class_="note"),
                ui.input_numeric("zero_extra_points", "Additional zero points", value=32768, min=0),
                ui.input_action_button("apply_zero", "Apply zero filling", class_="btn-primary"),
                ui.input_action_button("skip_zero", "Skip zero filling", class_="btn-primary"),
                ui.hr(),
                sample_only_controls(choices),
                ui.hr(),
                ui.input_action_button("back_apod", "Back"),
                ui.input_action_button("go_fft", "Next: Fourier transform", class_="btn-primary"),
                class_="card",
            )

        if step == "fft":
            return ui.div(
                ui.h2("Step 7 — Fourier transform", class_="step-title"),
                ui.div(
                    "The ppm window is fixed by the display controls below, so the spectrum does not jump between steps.",
                    class_="note",
                ),
                ui.input_action_button("apply_fft", "Apply Fourier transform", class_="btn-primary"),
                ui.hr(),
                sample_and_ppm_controls(choices),
                ui.hr(),
                ui.input_action_button("back_zero", "Back"),
                ui.input_action_button("go_phase", "Next: phase correction", class_="btn-primary"),
                class_="card",
            )

        if step == "phase":
            return ui.div(
                ui.h2("Step 8 — Phase correction", class_="step-title"),
                ui.div(
                    "Auto phase estimates a zero-order phase angle from the whole spectrum. "
                    "No fixed solvent/water region is assumed, so this step is more universal.",
                    class_="note",
                ),
                ui.input_checkbox("phase_auto", "Automatic phase correction", value=True),
                ui.input_numeric("manual_phase", "Manual angle, degrees", value=0.0),
                ui.input_action_button("apply_phase", "Apply phase correction", class_="btn-primary"),
                ui.input_action_button("skip_phase", "Skip phase correction", class_="btn-primary"),
                ui.hr(),
                sample_and_ppm_controls(choices),
                ui.hr(),
                ui.input_action_button("back_fft", "Back"),
                ui.input_action_button("go_ref", "Next: internal referencing", class_="btn-primary"),
                class_="card",
            )

        if step == "ref":
            return ui.div(
                ui.h2("Step 9 — Internal referencing", class_="step-title"),
                ui.div("Search around the reference region and shift the ppm axis to the target ppm.", class_="note"),
                ui.input_checkbox("use_reference", "Use reference peak", value=True),
                ui.input_numeric("reference_ppm", "Target ppm", value=0.0),
                ui.input_numeric("reference_search_min", "Search min ppm", value=-0.2),
                ui.input_numeric("reference_search_max", "Search max ppm", value=0.2),
                ui.input_action_button("apply_ref", "Apply referencing", class_="btn-primary"),
                ui.input_action_button("skip_ref", "Skip referencing", class_="btn-primary"),
                ui.hr(),
                sample_and_ppm_controls(choices),
                ui.hr(),
                ui.input_action_button("back_phase", "Back"),
                ui.input_action_button("go_base", "Next: baseline correction", class_="btn-primary"),
                class_="card",
            )

        if step == "base":
            return ui.div(
                ui.h2("Step 10 — Baseline correction", class_="step-title"),
                ui.div(
                    "The baseline estimator excludes/interpolates the water region while estimating baseline. "
                    "This avoids the water peak dragging the baseline.",
                    class_="note",
                ),
                ui.input_select("baseline_method", "Baseline method", choices=["als", "arpls", "airpls"], selected="als"),
                ui.input_numeric("baseline_smoothness", "Smoothness lambda", value=1e6, min=1),
                ui.input_numeric("baseline_asymmetry", "ALS asymmetry p", value=0.01, min=0, max=1),
                ui.input_numeric("baseline_iter", "Max iterations", value=12, min=1),
                ui.input_numeric("baseline_max_points", "Fast baseline points", value=3000, min=500),
                ui.div("Baseline correction may take a while for high-resolution spectra or many samples.", class_="note"),
                ui.input_action_button("apply_base", "Apply baseline correction", class_="btn-primary"),
                ui.input_action_button("skip_base", "Skip baseline correction", class_="btn-primary"),
                ui.hr(),
                sample_and_ppm_controls(choices),
                ui.hr(),
                ui.input_action_button("back_ref", "Back"),
                ui.input_action_button("go_align", "Next: peak alignment", class_="btn-primary"),
                class_="card",
            )

        if step == "align":
            return ui.div(
                ui.h2("Step 11 — Peak alignment", class_="step-title"),
                ui.div(
                    "This optional step applies a simple cross-correlation peak shift. "
                    "Keep it off unless you clearly see peak drift between samples. "
                    "For urine data, avoid letting the water/urea region drive alignment.",
                    class_="note",
                ),
                ui.input_checkbox("align_enabled", "Apply simple alignment", value=False),
                ui.input_numeric("align_reference", "Reference sample index", value=0, min=0),
                ui.input_numeric("align_min", "Alignment window min ppm", value=0.5),
                ui.input_numeric("align_max", "Alignment window max ppm", value=4.4),
                ui.input_numeric("align_max_shift", "Maximum shift, points", value=20, min=0),
                ui.input_action_button("apply_align", "Apply / skip alignment", class_="btn-primary"),
                ui.hr(),
                sample_and_ppm_controls(choices),
                ui.hr(),
                ui.input_action_button("back_base", "Back"),
                ui.input_action_button("go_neg", "Next: negative-value zeroing", class_="btn-primary"),
                class_="card",
            )

        if step == "neg":
            return ui.div(
                ui.h2("Step 12 — Negative-value zeroing", class_="step-title"),
                ui.div("Negative values after phase/baseline correction can be set to zero.", class_="note"),
                ui.input_checkbox("neg_enabled", "Set negative values to zero", value=True),
                ui.input_action_button("apply_neg", "Apply negative-value zeroing", class_="btn-primary"),
                ui.hr(),
                sample_and_ppm_controls(choices),
                ui.hr(),
                ui.input_action_button("back_align", "Back"),
                ui.input_action_button("go_window", "Next: window selection", class_="btn-primary"),
                class_="card",
            )

        if step == "window":
            return ui.div(
                ui.h2("Step 13 — Window selection", class_="step-title"),
                ui.div("Select the spectral window to keep for downstream analysis. Default: 0.2–10 ppm.", class_="note"),
                ui.input_numeric("window_min", "Window min ppm", value=0.2),
                ui.input_numeric("window_max", "Window max ppm", value=10.0),
                ui.input_action_button("apply_window", "Apply window selection", class_="btn-primary"),
                ui.input_action_button("skip_window", "Skip window selection", class_="btn-primary"),
                ui.hr(),
                sample_and_ppm_controls(choices),
                ui.hr(),
                ui.input_action_button("back_neg", "Back"),
                ui.input_action_button("go_region", "Next: region removal", class_="btn-primary"),
                class_="card",
            )

        if step == "region":
            return ui.div(
                ui.h2("Step 14 — Region removal", class_="step-title"),
                ui.div(
                    "Choose the ppm interval to remove before binning. "
                    "For urine data, the default is 4.5–6.1 ppm.",
                    class_="note",
                ),
                ui.layout_columns(
                    ui.input_numeric("region_min", "Region min ppm", value=4.5),
                    ui.input_numeric("region_max", "Region max ppm", value=6.1),
                    col_widths=[6, 6],
                ),
                ui.input_select("region_mode", "Removal mode", choices=["zero", "interpolate"], selected="zero"),
                ui.input_action_button("apply_region", "Apply region removal", class_="btn-primary"),
                ui.input_action_button("skip_region", "Skip region removal", class_="btn-primary"),
                ui.hr(),
                sample_and_ppm_controls(choices),
                ui.hr(),
                ui.input_action_button("back_window", "Back"),
                ui.input_action_button("go_bin", "Next: binning", class_="btn-primary"),
                class_="card",
            )

        if step == "bin":
            return ui.div(
                ui.h2("Step 15 — Binning", class_="step-title"),
                ui.div("Binning integrates the processed spectrum into ppm buckets.", class_="note"),
                ui.input_select("bin_definition", "Define buckets by", choices=["Bin width (ppm)", "Number of bins"], selected="Bin width (ppm)"),
                ui.layout_columns(
                    ui.input_numeric("bin_width", "Bin width ppm", value=0.01, min=0.001),
                    ui.input_numeric("bin_n_bins", "Number of bins", value=100, min=1),
                    col_widths=[6, 6],
                ),
                ui.input_select("bin_method", "Integration method", choices=["trapezoidal", "rectangular"], selected="trapezoidal"),
                ui.div("Creating the binned table may take a little while for many samples or very small bin widths.", class_="note"),
                ui.input_action_button("apply_bin", "Create binned table", class_="btn-primary"),
                ui.hr(),
                sample_and_ppm_controls(choices),
                ui.h4("Binned table preview"),
                ui.output_data_frame("binned_preview"),
                ui.hr(),
                ui.input_action_button("back_region", "Back"),
                ui.input_action_button("go_norm", "Next: normalization", class_="btn-primary"),
                class_="card",
            )

        if step == "norm":
            return ui.div(
                ui.h2("Step 16 — Normalization and export", class_="step-title"),
                ui.div("Normalize the binned table. PQN, TotalArea, SNV, or none are available.", class_="good-note"),
                ui.input_select("normalization_method", "Normalization method", choices=["PQN", "TotalArea", "SNV", "none"], selected="PQN"),
                ui.input_action_button("apply_norm", "Create normalized table", class_="btn-primary"),
                ui.input_action_button("skip_norm", "Skip normalization", class_="btn-primary"),
                ui.hr(),
                sample_and_ppm_controls(choices),
                ui.h4("Normalized table preview"),
                ui.output_data_frame("normalized_preview"),
                ui.download_button("download_binned", "Download binned CSV"),
                ui.download_button("download_normalized", "Download normalized CSV"),
                ui.download_button("download_log", "Download log"),
                ui.hr(),
                ui.input_action_button("back_bin", "Back"),
                ui.input_action_button("go_clinical", "Next: clinical data", class_="btn-primary"),
                class_="card",
            )

        if step == "clinical":
            return ui.div(
                ui.h2("Step 17 — Clinical labels and metadata", class_="step-title"),
                ui.div(
                    "Upload the clinical CSV/TSV/Excel file. The clinical table should contain a column named study_id and a column named Class for cohort labels. "
                    "The study_id values must match the sample/folder names detected during upload. "
                    "Clinical data are used for sample labels/classes, PCA/PLS-DA coloring, clinical correlations, and ML targets. By default, ML features are NMR bins only.",
                    class_="note",
                ),
                ui.div(
                    ui.strong("Using the synthetic example NMR cohort? "),
                    "Download its matching clinical labels here: ",
                    ui.download_button(
                        "download_example_clinical_step",
                        "Download example clinical CSV",
                    ),
                    class_="good-note",
                ),
                ui.input_file("clinical_file", "Clinical CSV/TSV/TXT/Excel file", accept=[".csv", ".tsv", ".txt", ".xlsx"], multiple=False),
                ui.layout_columns(
                    ui.input_text("clinical_id_col", "Sample ID column", value="study_id"),
                    ui.input_text("clinical_class_col", "Class column", value="Class"),
                    col_widths=[6, 6],
                ),
                ui.input_action_button("apply_clinical", "Read clinical data and align IDs", class_="btn-primary"),
                ui.hr(),
                ui.h4("Alignment summary"),
                ui.output_text_verbatim("merge_summary"),
                ui.h4("Clinical preview"),
                ui.output_data_frame("clinical_preview"),
                ui.h4("Aligned sample table"),
                ui.output_data_frame("merged_preview"),
                ui.download_button("download_combined", "Download aligned sample table CSV"),
                ui.hr(),
                ui.input_action_button("back_norm", "Back"),
                ui.input_action_button("go_eda", "Next: EDA", class_="btn-primary"),
                class_="card",
            )

        if step == "eda":
            return ui.div(
                ui.h2("Step 18 — Exploratory data analysis", class_="step-title"),
                ui.div(
                    "EDA uses X = normalized NMR bins and y = Class labels. "
                    "You can calculate multiple PCA/PLS-DA components and choose any component combination for 2D or 3D score plots. "
                    "The score-color menu is populated from the aligned clinical metadata, so users can color scores by Class, PSA, Gleason, age, or other available clinical variables.",
                    class_="note",
                ),
                ui.layout_columns(
                    ui.input_numeric("eda_top_n", "Top features/correlations to show", value=50, min=5),
                    ui.input_numeric("pca_n_components", "PCA components to calculate", value=5, min=2),
                    ui.input_numeric("pls_n_components", "PLS-DA components to calculate", value=3, min=2),
                    col_widths=[4, 4, 4],
                ),
                ui.layout_columns(
                    ui.input_select("eda_psa_column", "PSA column", choices=psa_column_choices, selected=psa_column_choices[0]),
                    ui.input_numeric("eda_psa_cutoff", "PSA cutoff", value=4.0, min=0.0),
                    col_widths=[6, 6],
                ),
                ui.div("EDA calculations may take a little while, especially PCA/PLS-DA with many bins.", class_="note"),
                ui.input_action_button("apply_eda", "Run EDA", class_="btn-primary"),
                ui.hr(),
                ui.input_select(
                    "eda_plot_type",
                    "Plot",
                    choices=["PCA scores", "PCA loadings", "PLS-DA scores", "Class counts", "Top univariate bins", "Clinical correlation heatmap", "PSA by Class", "Class by PSA group", "PSA-feature correlations"],
                    selected="PCA scores",
                ),
                ui.layout_columns(
                    ui.input_select("score_dimension", "Score plot dimension", choices=["2D", "3D"], selected="2D"),
                    ui.input_numeric("score_x_component", "X/start component", value=1, min=1),
                    ui.input_numeric("score_y_component", "Y component", value=2, min=1),
                    ui.input_numeric("score_z_component", "Z component", value=3, min=1),
                    ui.input_numeric("loading_n_components", "Loading PCs to show", value=2, min=1),
                    col_widths=[2, 2, 2, 2, 4],
                ),
                ui.layout_columns(
                    ui.input_select("score_color_by", "Score color by", choices=eda_color_choices, selected="Class"),
                    ui.input_select("score_palette", "Score color palette", choices=["Plotly", "Viridis", "Plasma", "Turbo", "Cividis"], selected="Plotly"),
                    ui.input_numeric("score_marker_size", "Score marker size", value=8, min=3, max=30),
                    col_widths=[4, 4, 4],
                ),
                output_widget("eda_plot"),
                ui.download_button("download_eda_plot_data", "Download current EDA plot data CSV"),
                ui.h4("PCA explained variance"),
                ui.output_data_frame("pca_variance_table"),
                ui.h4("PLS-DA component summary"),
                ui.output_data_frame("pls_variance_table"),
                ui.h4("Class counts"),
                ui.output_data_frame("class_counts_table"),
                ui.h4("Top univariate bins"),
                ui.output_data_frame("univariate_table"),
                ui.h4("Clinical correlation matrix"),
                ui.output_data_frame("clinical_corr_table"),
                ui.h4("Top clinical-feature correlations"),
                ui.output_data_frame("feature_clinical_corr_table"),
                ui.download_button("download_eda_univariate", "Download EDA univariate CSV"),
                ui.hr(),
                ui.input_action_button("back_clinical", "Back"),
                ui.input_action_button("go_outlier", "Next: outlier removal", class_="btn-primary"),
                class_="card",
            )

        if step == "outlier":
            return ui.div(
                ui.h2("Step 19 — Outlier detection/removal", class_="step-title"),
                ui.div(
                    "Detect possible spectral outliers, inspect them, then either remove flagged samples or skip removal. "
                    "For multiple cohorts, calculating limits within each class is usually safer because valid cohorts may naturally separate.",
                    class_="note",
                ),
                ui.layout_columns(
                    ui.input_select("outlier_method", "Outlier method", choices=["Hotelling T2", "Robust distance"], selected="Hotelling T2"),
                    ui.input_numeric("outlier_pcs", "PCs used for outlier detection", value=5, min=2),
                    col_widths=[6, 6],
                ),
                ui.layout_columns(
                    ui.input_select("outlier_confidence", "Hotelling T² confidence", choices=["0.95", "0.99"], selected="0.95"),
                    ui.input_numeric("outlier_threshold", "Robust-distance z threshold", value=3.0, min=1.0),
                    col_widths=[6, 6],
                ),
                ui.input_checkbox("outlier_groupwise", "Calculate limits within each Class/cohort", value=True),
                ui.input_text_area(
                    "manual_remove_ids",
                    "Sample IDs to remove manually",
                    placeholder="Optional: paste sample IDs separated by comma, space, or new line. Leave empty to remove all flagged outliers.",
                    rows=3,
                ),
                ui.input_action_button("detect_outliers", "Detect outliers", class_="btn-primary"),
                ui.input_action_button("remove_outliers", "Remove selected/flagged outliers", class_="btn-primary"),
                ui.input_action_button("skip_outliers", "Skip outlier removal", class_="btn-primary"),
                ui.hr(),
                ui.h4("Outliers by cohort/class"),
                output_widget("outlier_cohort_plot"),
                ui.h4("Outlier table"),
                ui.output_data_frame("outlier_table"),
                ui.download_button("download_outlier_table", "Download outlier table CSV"),
                ui.hr(),
                ui.input_action_button("back_eda", "Back"),
                ui.input_action_button("go_eda_filtered", "Next: EDA after outlier removal", class_="btn-primary"),
                class_="card",
            )

        if step == "eda_filtered":
            return ui.div(
                ui.h2("Step 20 — EDA after outlier removal", class_="step-title"),
                ui.div(
                    "Run the same EDA again after outlier removal. "
                    "If no outliers were removed, this uses the original aligned dataset.",
                    class_="note",
                ),
                ui.layout_columns(
                    ui.input_numeric("eda2_top_n", "Top features/correlations to show", value=50, min=5),
                    ui.input_numeric("pca2_n_components", "PCA components to calculate", value=5, min=2),
                    ui.input_numeric("pls2_n_components", "PLS-DA components to calculate", value=3, min=2),
                    col_widths=[4, 4, 4],
                ),
                ui.layout_columns(
                    ui.input_select("eda2_psa_column", "PSA column", choices=psa2_column_choices, selected=psa2_column_choices[0]),
                    ui.input_numeric("eda2_psa_cutoff", "PSA cutoff", value=4.0, min=0.0),
                    col_widths=[6, 6],
                ),
                ui.div("EDA after outlier removal may take a little while for large datasets.", class_="note"),
                ui.input_action_button("apply_eda_filtered", "Run EDA after outlier removal", class_="btn-primary"),
                ui.hr(),
                ui.input_select(
                    "eda2_plot_type",
                    "Plot",
                    choices=["PCA scores", "PCA loadings", "PLS-DA scores", "Class counts", "Top univariate bins", "Clinical correlation heatmap", "PSA by Class", "Class by PSA group", "PSA-feature correlations"],
                    selected="PCA scores",
                ),
                ui.layout_columns(
                    ui.input_select("score2_dimension", "Score plot dimension", choices=["2D", "3D"], selected="2D"),
                    ui.input_numeric("score2_x_component", "X/start component", value=1, min=1),
                    ui.input_numeric("score2_y_component", "Y component", value=2, min=1),
                    ui.input_numeric("score2_z_component", "Z component", value=3, min=1),
                    ui.input_numeric("loading2_n_components", "Loading PCs to show", value=2, min=1),
                    col_widths=[2, 2, 2, 2, 4],
                ),
                ui.layout_columns(
                    ui.input_select("score2_color_by", "Score color by", choices=eda2_color_choices, selected="Class"),
                    ui.input_select("score2_palette", "Score color palette", choices=["Plotly", "Viridis", "Plasma", "Turbo", "Cividis"], selected="Plotly"),
                    ui.input_numeric("score2_marker_size", "Score marker size", value=8, min=3, max=30),
                    col_widths=[4, 4, 4],
                ),
                output_widget("eda_filtered_plot"),
                ui.download_button("download_eda_filtered_plot_data", "Download current post-outlier EDA plot data CSV"),
                ui.h4("Class counts after outlier removal"),
                ui.output_data_frame("class_counts_filtered_table"),
                ui.h4("Top univariate bins after outlier removal"),
                ui.output_data_frame("univariate_filtered_table"),
                ui.download_button("download_eda_filtered_univariate", "Download filtered EDA univariate CSV"),
                ui.hr(),
                ui.input_action_button("back_outlier", "Back"),
                ui.input_action_button("go_ml", "Next: ML", class_="btn-primary"),
                class_="card",
            )

        if step == "ml":
            return ui.div(
                ui.h2("Step 21 — Machine learning", class_="step-title"),
                ui.div(
                    "ML can use NMR bins only, clinical variables only, NMR + clinical variables, or PSA only. "
                    "Imputation, scaling, and optional PCA reduction are fitted inside the scikit-learn Pipeline.",
                    class_="good-note",
                ),
                ui.layout_columns(
                    ui.input_select("ml_model", "Model", choices=["LogisticRegression", "RandomForest", "LinearSVM", "ANN"], selected="LogisticRegression"),
                    ui.input_numeric("ml_test_size", "Test size", value=0.25, min=0.1, max=0.5),
                    ui.input_checkbox("ml_use_cv", "Use cross-validation", value=True),
                    ui.input_numeric("ml_cv_folds", "CV folds", value=5, min=2),
                    col_widths=[3, 3, 3, 3],
                ),
                ui.layout_columns(
                    ui.input_select("ml_feature_mode", "ML feature set", choices=["NMR only", "Clinical only", "NMR + clinical", "PSA only"], selected="NMR only"),
                    ui.input_checkbox("ml_use_pca", "Use PCA reduction inside ML pipeline", value=False),
                    ui.input_numeric("ml_pca_components", "ML PCA components", value=10, min=2),
                    col_widths=[4, 4, 4],
                ),
                ui.layout_columns(
                    ui.input_select("ml_psa_column", "PSA column", choices=psa2_column_choices, selected=psa2_column_choices[0]),
                    ui.input_numeric("ml_psa_cutoff", "PSA cutoff", value=4.0, min=0.0),
                    ui.input_select("ml_psa_subset", "PSA subset", choices=["All samples", "Low PSA (< cutoff)", "High PSA (>= cutoff)"], selected="All samples"),
                    col_widths=[4, 4, 4],
                ),
                ui.h4("ANN settings"),
                ui.layout_columns(
                    ui.input_text("ann_hidden_layers", "Hidden layers", value="64,32"),
                    ui.input_select("ann_activation", "Activation", choices=["relu", "tanh", "logistic"], selected="relu"),
                    ui.input_checkbox("ann_early_stopping", "Early stopping", value=True),
                    col_widths=[4, 4, 4],
                ),
                ui.layout_columns(
                    ui.input_numeric("ann_alpha", "L2 alpha", value=0.0001, min=0),
                    ui.input_numeric("ann_learning_rate", "Learning rate", value=0.001, min=0.000001),
                    ui.input_numeric("ann_max_iter", "Max iterations", value=500, min=50),
                    col_widths=[4, 4, 4],
                ),
                ui.div("ML training and cross-validation may take a while depending on model, features, and sample size.", class_="note"),
                ui.input_action_button("apply_ml", "Run ML", class_="btn-primary"),
                ui.hr(),
                ui.h4("Metrics"),
                ui.output_text_verbatim("ml_metrics_text"),
                ui.input_select(
                    "ml_plot_type",
                    "ML plot",
                    choices=["Confusion matrix", "Feature importance", "ROC curve", "Predicted probabilities", "PSA baseline confusion matrix"],
                    selected="Confusion matrix",
                ),
                output_widget("ml_plot"),
                ui.download_button("download_ml_plot_data", "Download current ML plot data CSV"),
                ui.h4("Feature importance table"),
                ui.output_data_frame("feature_importance_table"),
                ui.h4("Test predictions"),
                ui.output_data_frame("test_predictions_table"),
                ui.download_button("download_ml_importance", "Download feature importance CSV"),
                ui.download_button("download_ml_predictions", "Download test predictions CSV"),
                ui.h4("Model performance history"),
                ui.output_data_frame("ml_history_table"),
                ui.download_button("download_ml_history", "Download model performance history CSV"),
                ui.hr(),
                ui.input_action_button("back_eda", "Back"),
                class_="card",
            )

        return ui.div("Unknown step.", class_="card")

    # Navigation and action events

    @reactive.effect
    @reactive.event(input.read_zip)
    def _read_zip():
        clear_error()
        try:
            uploaded = input.data_zip()
            if not uploaded:
                status_state.set("Please choose a ZIP file first.")
                return

            samples = load_zip_and_read_fids(uploaded[0]["datapath"])
            samples_state.set(samples)
            binned_state.set(None)
            normalized_state.set(None)
            clinical_state.set(None)
            combined_state.set(None)
            eda_state.set(None)
            outlier_state.set(None)
            filtered_state.set(None)
            eda_filtered_state.set(None)
            ml_state.set(None)
            ml_history_state.set(pd.DataFrame())
            current_step.set("raw")
            status_state.set(f"ZIP read successfully. Found {len(samples)} experiment(s).")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.go_upload_empty)
    def _go_upload_empty():
        current_step.set("upload")

    @reactive.effect
    @reactive.event(input.back_to_upload)
    def _back_to_upload():
        current_step.set("upload")

    @reactive.effect
    @reactive.event(input.go_group)
    def _go_group():
        current_step.set("group")

    @reactive.effect
    @reactive.event(input.back_raw)
    def _back_raw():
        current_step.set("raw")

    @reactive.effect
    @reactive.event(input.apply_group)
    def _apply_group():
        clear_error()
        try:
            samples_state.set(apply_group_delay(require_samples(), override_points=float(input.group_delay_override())))
            reset_downstream_analysis()
            status_state.set("Group delay applied.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.skip_group)
    def _skip_group():
        clear_error()
        try:
            out = []
            for sample in require_samples():
                s = dict(sample)
                s["group_delay_points"] = 0
                s["group_delay_fid"] = s["raw_fid"].copy()
                s["log"] = s.get("log", []) + ["Group delay removal skipped."]
                out.append(s)
            samples_state.set(out)
            reset_downstream_analysis()
            status_state.set("Group delay removal skipped.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.go_solvent)
    def _go_solvent():
        samples = samples_state.get()
        if samples and "group_delay_fid" in samples[0]:
            current_step.set("solvent")
        else:
            status_state.set("Click 'Apply group delay' first.")

    @reactive.effect
    @reactive.event(input.back_group)
    def _back_group():
        current_step.set("group")

    @reactive.effect
    @reactive.event(input.apply_solvent)
    def _apply_solvent():
        clear_error()
        try:
            samples_state.set(
                apply_solvent_residuals_removal(
                    require_samples(),
                    lam=float(input.solvent_lambda()),
                    enabled=bool(input.solvent_enabled()),
                )
            )
            reset_downstream_analysis()
            status_state.set("Solvent residual step applied.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.skip_solvent)
    def _skip_solvent():
        clear_error()
        try:
            out = []
            for sample in require_samples():
                if "group_delay_fid" not in sample:
                    raise ValueError("Apply or skip group delay first.")
                s = dict(sample)
                fid = s["group_delay_fid"]
                s["solvent_removed_fid"] = fid.copy()
                s["estimated_solvent"] = np.zeros_like(fid)
                s["log"] = s.get("log", []) + ["Solvent residual removal skipped."]
                out.append(s)
            samples_state.set(out)
            reset_downstream_analysis()
            status_state.set("Solvent residual removal skipped.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.go_apod)
    def _go_apod():
        samples = samples_state.get()
        if samples and "solvent_removed_fid" in samples[0]:
            current_step.set("apod")
        else:
            status_state.set("Click 'Apply solvent residual removal' first.")

    @reactive.effect
    @reactive.event(input.back_solvent)
    def _back_solvent():
        current_step.set("solvent")

    @reactive.effect
    @reactive.event(input.apply_apod)
    def _apply_apod():
        clear_error()
        try:
            samples_state.set(
                apply_apodization(
                    require_samples(),
                    lb=float(input.apod_lb()),
                    kind=input.apod_kind(),
                )
            )
            reset_downstream_analysis()
            status_state.set("Apodization applied.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.skip_apod)
    def _skip_apod():
        clear_error()
        try:
            out = []
            for sample in require_samples():
                if "solvent_removed_fid" not in sample:
                    raise ValueError("Apply or skip solvent residual removal first.")
                s = dict(sample)
                s["apodized_fid"] = s["solvent_removed_fid"].copy()
                s["apodization_kind"] = "none"
                s["apodization_lb"] = 0.0
                s["log"] = s.get("log", []) + ["Apodization skipped."]
                out.append(s)
            samples_state.set(out)
            reset_downstream_analysis()
            status_state.set("Apodization skipped.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.go_zero)
    def _go_zero():
        samples = samples_state.get()
        if samples and "apodized_fid" in samples[0]:
            current_step.set("zero")
        else:
            status_state.set("Click 'Apply apodization' first.")

    @reactive.effect
    @reactive.event(input.back_apod)
    def _back_apod():
        current_step.set("apod")

    @reactive.effect
    @reactive.event(input.apply_zero)
    def _apply_zero():
        clear_error()
        try:
            samples_state.set(apply_zero_filling(require_samples(), extra_points=int(input.zero_extra_points())))
            reset_downstream_analysis()
            status_state.set("Zero filling applied.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.skip_zero)
    def _skip_zero():
        clear_error()
        try:
            out = []
            for sample in require_samples():
                if "apodized_fid" not in sample:
                    raise ValueError("Apply or skip apodization first.")
                s = dict(sample)
                s["zero_filled_fid"] = s["apodized_fid"].copy()
                s["zero_fill_extra_points"] = 0
                s["log"] = s.get("log", []) + ["Zero filling skipped."]
                out.append(s)
            samples_state.set(out)
            reset_downstream_analysis()
            status_state.set("Zero filling skipped.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.go_fft)
    def _go_fft():
        samples = samples_state.get()
        if samples and "zero_filled_fid" in samples[0]:
            current_step.set("fft")
        else:
            status_state.set("Click 'Apply zero filling' first.")

    @reactive.effect
    @reactive.event(input.back_zero)
    def _back_zero():
        current_step.set("zero")

    @reactive.effect
    @reactive.event(input.apply_fft)
    def _apply_fft():
        clear_error()
        try:
            samples_state.set(apply_fourier_transform(require_samples()))
            reset_downstream_analysis()
            status_state.set("Fourier transform applied.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.go_phase)
    def _go_phase():
        samples = samples_state.get()
        if samples and "complex_spectrum" in samples[0]:
            current_step.set("phase")
        else:
            status_state.set("Click 'Apply Fourier transform' first.")

    @reactive.effect
    @reactive.event(input.back_fft)
    def _back_fft():
        current_step.set("fft")

    @reactive.effect
    @reactive.event(input.apply_phase)
    def _apply_phase():
        clear_error()
        try:
            samples_state.set(
                apply_phase_correction(
                    require_samples(),
                    auto=bool(input.phase_auto()),
                    manual_angle_deg=float(input.manual_phase()),
                    exclude_region_text="",
                )
            )
            reset_downstream_analysis()
            status_state.set("Phase correction applied.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.skip_phase)
    def _skip_phase():
        clear_error()
        try:
            out = []
            for sample in require_samples():
                if "spectrum_real" not in sample:
                    raise ValueError("Apply Fourier transform first.")
                s = dict(sample)
                s["phased"] = s["spectrum_real"].copy()
                s["phase_angle_deg"] = 0.0
                s["log"] = s.get("log", []) + ["Phase correction skipped."]
                out.append(s)
            samples_state.set(out)
            reset_downstream_analysis()
            status_state.set("Phase correction skipped.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.go_ref)
    def _go_ref():
        samples = samples_state.get()
        if samples and "phased" in samples[0]:
            current_step.set("ref")
        else:
            status_state.set("Click 'Apply phase correction' first.")

    @reactive.effect
    @reactive.event(input.back_phase)
    def _back_phase():
        current_step.set("phase")

    @reactive.effect
    @reactive.event(input.apply_ref)
    def _apply_ref():
        clear_error()
        try:
            samples_state.set(
                apply_referencing(
                    require_samples(),
                    use_reference=bool(input.use_reference()),
                    target_ppm=float(input.reference_ppm()),
                    search_min=float(input.reference_search_min()),
                    search_max=float(input.reference_search_max()),
                )
            )
            reset_downstream_analysis()
            status_state.set("Internal referencing applied.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.skip_ref)
    def _skip_ref():
        clear_error()
        try:
            out = []
            for sample in require_samples():
                if "ppm" not in sample or "phased" not in sample:
                    raise ValueError("Apply or skip phase correction first.")
                s = dict(sample)
                s["referenced_ppm"] = s["ppm"].copy()
                s["found_reference"] = None
                s["log"] = s.get("log", []) + ["Referencing skipped."]
                out.append(s)
            samples_state.set(out)
            reset_downstream_analysis()
            status_state.set("Referencing skipped.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.go_base)
    def _go_base():
        samples = samples_state.get()
        if samples and "referenced_ppm" in samples[0]:
            current_step.set("base")
        else:
            status_state.set("Click 'Apply referencing' first.")

    @reactive.effect
    @reactive.event(input.back_ref)
    def _back_ref():
        current_step.set("ref")

    @reactive.effect
    @reactive.event(input.apply_base)
    def _apply_base():
        clear_error()
        try:
            samples_state.set(
                apply_baseline_correction(
                    require_samples(),
                    method=input.baseline_method(),
                    smoothness=float(input.baseline_smoothness()),
                    asymmetry=float(input.baseline_asymmetry()),
                    max_iter=int(input.baseline_iter()),
                    exclude_region_text="",
                    max_points=int(input.baseline_max_points()),
                )
            )
            reset_downstream_analysis()
            status_state.set("Baseline correction applied.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.skip_base)
    def _skip_base():
        clear_error()
        try:
            out = []
            for sample in require_samples():
                if "referenced_ppm" not in sample or "phased" not in sample:
                    raise ValueError("Apply or skip referencing first.")
                s = dict(sample)
                y = s["phased"]
                s["baseline"] = np.zeros_like(y)
                s["baseline_corrected"] = y.copy()
                s["log"] = s.get("log", []) + ["Baseline correction skipped."]
                out.append(s)
            samples_state.set(out)
            reset_downstream_analysis()
            status_state.set("Baseline correction skipped.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.go_align)
    def _go_align():
        samples = samples_state.get()
        if samples and "baseline_corrected" in samples[0]:
            current_step.set("align")
        else:
            status_state.set("Click 'Apply baseline correction' first.")

    @reactive.effect
    @reactive.event(input.back_base)
    def _back_base():
        current_step.set("base")

    @reactive.effect
    @reactive.event(input.apply_align)
    def _apply_align():
        clear_error()
        try:
            samples_state.set(
                apply_peak_alignment(
                    require_samples(),
                    enabled=bool(input.align_enabled()),
                    reference_index=int(input.align_reference()),
                    align_min=float(input.align_min()),
                    align_max=float(input.align_max()),
                    max_shift_points=int(input.align_max_shift()),
                )
            )
            reset_downstream_analysis()
            status_state.set("Peak alignment step applied/skipped.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.go_neg)
    def _go_neg():
        samples = samples_state.get()
        if samples and "aligned" in samples[0]:
            current_step.set("neg")
        else:
            status_state.set("Click 'Apply / skip alignment' first.")

    @reactive.effect
    @reactive.event(input.back_align)
    def _back_align():
        current_step.set("align")

    @reactive.effect
    @reactive.event(input.apply_neg)
    def _apply_neg():
        clear_error()
        try:
            samples_state.set(apply_negative_values_zeroing(require_samples(), enabled=bool(input.neg_enabled())))
            reset_downstream_analysis()
            status_state.set("Negative-value zeroing step applied.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.go_window)
    def _go_window():
        samples = samples_state.get()
        if samples and "negative_zeroed" in samples[0]:
            current_step.set("window")
        else:
            status_state.set("Click 'Apply negative-value zeroing' first.")

    @reactive.effect
    @reactive.event(input.back_neg)
    def _back_neg():
        current_step.set("neg")

    @reactive.effect
    @reactive.event(input.apply_window)
    def _apply_window():
        clear_error()
        try:
            samples_state.set(
                apply_window_selection(
                    require_samples(),
                    ppm_min=float(input.window_min()),
                    ppm_max=float(input.window_max()),
                )
            )
            reset_downstream_analysis()
            status_state.set("Window selection applied.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.skip_window)
    def _skip_window():
        clear_error()
        try:
            out = []
            for sample in require_samples():
                if "negative_zeroed" not in sample or "referenced_ppm" not in sample:
                    raise ValueError("Apply negative-value zeroing first.")
                s = dict(sample)
                ppm = s["referenced_ppm"]
                y = s["negative_zeroed"]
                s["window_ppm"] = ppm.copy()
                s["window_intensity"] = y.copy()
                s["window_range"] = (float(np.nanmin(ppm)), float(np.nanmax(ppm)))
                s["log"] = s.get("log", []) + ["Window selection skipped; full ppm range kept."]
                out.append(s)
            samples_state.set(out)
            reset_downstream_analysis()
            status_state.set("Window selection skipped; full ppm range kept.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.go_region)
    def _go_region():
        samples = samples_state.get()
        if samples and "window_intensity" in samples[0]:
            current_step.set("region")
        else:
            status_state.set("Click 'Apply window selection' first.")

    @reactive.effect
    @reactive.event(input.back_window)
    def _back_window():
        current_step.set("window")

    @reactive.effect
    @reactive.event(input.apply_region)
    def _apply_region():
        clear_error()
        try:
            samples_state.set(
                apply_region_removal(
                    require_samples(),
                    region_text=f"{float(input.region_min())}-{float(input.region_max())}",
                    mode=input.region_mode(),
                )
            )
            binned_state.set(None)
            normalized_state.set(None)
            reset_downstream_analysis()
            status_state.set("Region removal applied.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.skip_region)
    def _skip_region():
        clear_error()
        try:
            out = []
            for sample in require_samples():
                if "window_intensity" not in sample:
                    raise ValueError("Apply or skip window selection first.")
                s = dict(sample)
                s["region_removed"] = s["window_intensity"].copy()
                s["region_text"] = ""
                s["region_mode"] = "none"
                s["log"] = s.get("log", []) + ["Region removal skipped."]
                out.append(s)
            samples_state.set(out)
            reset_downstream_analysis()
            status_state.set("Region removal skipped.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.go_bin)
    def _go_bin():
        samples = samples_state.get()
        if samples and "region_removed" in samples[0]:
            current_step.set("bin")
        else:
            status_state.set("Click 'Apply region removal' first.")

    @reactive.effect
    @reactive.event(input.back_region)
    def _back_region():
        current_step.set("region")

    @reactive.effect
    @reactive.event(input.apply_bin)
    def _apply_bin():
        clear_error()
        try:
            n_bins = int(input.bin_n_bins()) if input.bin_definition() == "Number of bins" else None
            new_samples, binned = apply_binning(
                require_samples(),
                bin_width=float(input.bin_width()),
                method=input.bin_method(),
                n_bins=n_bins,
            )
            samples_state.set(new_samples)
            binned_state.set(binned)
            normalized_state.set(None)
            clinical_state.set(None)
            combined_state.set(None)
            eda_state.set(None)
            outlier_state.set(None)
            filtered_state.set(None)
            eda_filtered_state.set(None)
            ml_state.set(None)
            status_state.set("Binned table created.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.go_norm)
    def _go_norm():
        if binned_state.get() is not None:
            current_step.set("norm")
        else:
            status_state.set("Click 'Create binned table' first.")

    @reactive.effect
    @reactive.event(input.back_bin)
    def _back_bin():
        current_step.set("bin")

    @reactive.effect
    @reactive.event(input.apply_norm)
    def _apply_norm():
        clear_error()
        try:
            binned = binned_state.get()
            if binned is None:
                raise ValueError("Create binned table first.")
            normalized_state.set(
                apply_normalization(
                    require_samples(),
                    binned,
                    method=input.normalization_method(),
                )
            )
            combined_state.set(None)
            eda_state.set(None)
            outlier_state.set(None)
            filtered_state.set(None)
            eda_filtered_state.set(None)
            ml_state.set(None)
            status_state.set("Normalized table created.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.skip_norm)
    def _skip_norm():
        clear_error()
        try:
            binned = binned_state.get()
            if binned is None:
                raise ValueError("Create binned table first.")
            normalized_state.set(binned.copy())
            combined_state.set(None)
            eda_state.set(None)
            outlier_state.set(None)
            filtered_state.set(None)
            eda_filtered_state.set(None)
            ml_state.set(None)
            status_state.set("Normalization skipped; binned table copied unchanged.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.go_clinical)
    def _go_clinical():
        if normalized_state.get() is not None:
            current_step.set("clinical")
        else:
            status_state.set("Click 'Create normalized table' first.")

    @reactive.effect
    @reactive.event(input.back_norm)
    def _back_norm():
        current_step.set("norm")

    @reactive.effect
    @reactive.event(input.apply_clinical)
    def _apply_clinical():
        clear_error()
        try:
            normalized = normalized_state.get()
            if normalized is None:
                raise ValueError("Create normalized table first.")

            uploaded = input.clinical_file()
            if not uploaded:
                raise ValueError("Please upload a clinical CSV/TSV/TXT file.")

            clinical = read_clinical_file(uploaded[0]["datapath"])
            clinical_state.set(clinical)

            merged_info = merge_omics_clinical(
                normalized,
                clinical,
                clinical_id_col=input.clinical_id_col(),
                class_col=input.clinical_class_col(),
            )

            combined_state.set(merged_info)
            eda_state.set(None)
            outlier_state.set(None)
            filtered_state.set(None)
            eda_filtered_state.set(None)
            ml_state.set(None)

            summary = merged_info["summary"]
            status_state.set(
                f"Clinical data aligned: {summary['n_matched_with_class']} samples with Class labels. "
                f"{summary['n_unmatched_spectra']} spectra unmatched; "
                f"{summary['n_unmatched_clinical']} clinical rows unmatched."
            )
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.go_eda)
    def _go_eda():
        if combined_state.get() is not None:
            current_step.set("eda")
        else:
            status_state.set("Click 'Read clinical data and align IDs' first.")

    @reactive.effect
    @reactive.event(input.back_clinical)
    def _back_clinical():
        current_step.set("clinical")

    @reactive.effect
    @reactive.event(input.apply_eda)
    def _apply_eda():
        clear_error()
        try:
            aligned = combined_state.get()
            if aligned is None:
                raise ValueError("Align clinical data first.")

            eda = {
                "aligned": aligned,
                "psa_col": input.eda_psa_column(),
                "psa_cutoff": float(input.eda_psa_cutoff()),
                "class_counts": class_counts(aligned),
                "pca": pca_scores(aligned, n_components=int(input.pca_n_components())),
                "plsda": plsda_scores(aligned, n_components=int(input.pls_n_components())),
                "univariate": univariate_feature_tests(
                    aligned,
                    max_features=int(input.eda_top_n()),
                ),
                "clinical_corr": clinical_correlation_matrix(aligned),
                "feature_clinical_corr": top_feature_clinical_correlations(
                    aligned,
                    max_rows=int(input.eda_top_n()),
                ),
            }

            eda_state.set(eda)
            status_state.set("EDA finished.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.go_outlier)
    def _go_outlier():
        if eda_state.get() is not None:
            current_step.set("outlier")
        else:
            status_state.set("Run EDA first.")

    @reactive.effect
    @reactive.event(input.back_eda)
    def _back_eda():
        current_step.set("eda")

    @reactive.effect
    @reactive.event(input.detect_outliers)
    def _detect_outliers():
        clear_error()
        try:
            aligned = combined_state.get()
            if aligned is None:
                raise ValueError("Align clinical data first.")

            result = detect_pca_outliers(
                aligned,
                n_components=int(input.outlier_pcs()),
                threshold=float(input.outlier_threshold()),
                method=input.outlier_method(),
                confidence=float(input.outlier_confidence()),
                groupwise=bool(input.outlier_groupwise()),
            )
            outlier_state.set(result)
            status_state.set(f"Outlier detection finished. Flagged {result['n_outliers']} sample(s).")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.remove_outliers)
    def _remove_outliers():
        clear_error()
        try:
            aligned = combined_state.get()
            result = outlier_state.get()

            if aligned is None:
                raise ValueError("Align clinical data first.")
            if result is None:
                raise ValueError("Detect outliers first.")

            manual_text = str(input.manual_remove_ids() or "").strip()

            if manual_text:
                outlier_ids = [
                    x.strip()
                    for x in re.split(r"[\s,;]+", manual_text)
                    if x.strip()
                ]
                source = "manually selected"
            else:
                outlier_ids = result.get("outlier_ids", [])
                source = "flagged"

            filtered = filter_aligned_samples(aligned, outlier_ids)
            filtered_state.set(filtered)
            eda_filtered_state.set(None)
            ml_state.set(None)
            status_state.set(f"Removed {len(outlier_ids)} {source} outlier(s).")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.skip_outliers)
    def _skip_outliers():
        clear_error()
        try:
            aligned = combined_state.get()
            if aligned is None:
                raise ValueError("Align clinical data first.")
            filtered_state.set(aligned)
            eda_filtered_state.set(None)
            ml_state.set(None)
            status_state.set("Outlier removal skipped. The next EDA uses all aligned samples.")
            current_step.set("eda_filtered")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.go_eda_filtered)
    def _go_eda_filtered():
        if filtered_state.get() is None:
            # Allow the step even without removal; it will use original aligned data.
            status_state.set("No outliers removed yet. Post-outlier EDA will use the original aligned dataset.")
        current_step.set("eda_filtered")

    @reactive.effect
    @reactive.event(input.back_outlier)
    def _back_outlier():
        current_step.set("outlier")

    @reactive.effect
    @reactive.event(input.apply_eda_filtered)
    def _apply_eda_filtered():
        clear_error()
        try:
            aligned = filtered_state.get()
            if aligned is None:
                aligned = combined_state.get()

            if aligned is None:
                raise ValueError("Align clinical data first.")

            eda = {
                "aligned": aligned,
                "psa_col": input.eda2_psa_column(),
                "psa_cutoff": float(input.eda2_psa_cutoff()),
                "class_counts": class_counts(aligned),
                "pca": pca_scores(aligned, n_components=int(input.pca2_n_components())),
                "plsda": plsda_scores(aligned, n_components=int(input.pls2_n_components())),
                "univariate": univariate_feature_tests(
                    aligned,
                    max_features=int(input.eda2_top_n()),
                ),
                "clinical_corr": clinical_correlation_matrix(aligned),
                "feature_clinical_corr": top_feature_clinical_correlations(
                    aligned,
                    max_rows=int(input.eda2_top_n()),
                ),
            }

            eda_filtered_state.set(eda)
            status_state.set("EDA after outlier removal finished.")
        except Exception:
            set_error()

    @reactive.effect
    @reactive.event(input.go_ml)
    def _go_ml():
        if filtered_state.get() is not None or combined_state.get() is not None:
            current_step.set("ml")
        else:
            status_state.set("Align clinical data first.")

    def _ml_history_row(result: dict):
        row = {
            "run_time": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        metrics = result.get("metrics", {})
        cv_summary = result.get("cv_summary", {})

        for key, value in metrics.items():
            if isinstance(value, (list, tuple)):
                row[key] = "; ".join(map(str, value))
            else:
                row[key] = value

        for key, value in cv_summary.items():
            row[f"cv_{key}"] = value

        # Keep the most useful app settings explicit, even if some are already
        # present in metrics.
        try:
            row["ui_model"] = input.ml_model()
            row["ui_feature_mode"] = input.ml_feature_mode()
            row["ui_use_pca"] = bool(input.ml_use_pca())
            row["ui_pca_components"] = int(input.ml_pca_components())
            row["ui_use_cv"] = bool(input.ml_use_cv())
            row["ui_cv_folds"] = int(input.ml_cv_folds())
            row["ui_test_size"] = float(input.ml_test_size())
            row["ui_psa_subset"] = input.ml_psa_subset()
        except Exception:
            pass

        return row

    def _append_ml_history(result: dict):
        row = pd.DataFrame([_ml_history_row(result)])
        history = ml_history_state.get()

        if history is None or history.empty:
            ml_history_state.set(row)
        else:
            ml_history_state.set(pd.concat([history, row], ignore_index=True, sort=False))

    @reactive.effect
    @reactive.event(input.apply_ml)
    def _apply_ml():
        clear_error()
        try:
            aligned = filtered_state.get()
            if aligned is None:
                aligned = combined_state.get()

            if aligned is None:
                raise ValueError("Align clinical data first.")

            result = train_ml_model(
                aligned,
                model_name=input.ml_model(),
                test_size=float(input.ml_test_size()),
                cv_folds=int(input.ml_cv_folds()),
                use_pca=bool(input.ml_use_pca()),
                pca_components=int(input.ml_pca_components()),
                feature_mode=input.ml_feature_mode(),
                psa_col=input.ml_psa_column(),
                psa_cutoff=float(input.ml_psa_cutoff()),
                psa_subset=input.ml_psa_subset(),
                ann_hidden_layers=input.ann_hidden_layers(),
                ann_activation=input.ann_activation(),
                ann_alpha=float(input.ann_alpha()),
                ann_learning_rate=float(input.ann_learning_rate()),
                ann_max_iter=int(input.ann_max_iter()),
                ann_early_stopping=bool(input.ann_early_stopping()),
                use_cv=bool(input.ml_use_cv()),
            )

            ml_state.set(result)
            _append_ml_history(result)
            status_state.set("ML finished.")
        except Exception:
            set_error()

    # Public synthetic example downloads.
    # Files are generated in memory so downloads work on local installations
    # and hosted deployments without relying on separately packaged files.

    @render.download(
        filename="NMRMetaboWizard_demo_cohort_bruker.zip",
        media_type="application/zip",
    )
    def download_example_nmr():
        yield _build_demo_nmr_zip()

    @render.download(
        filename="NMRMetaboWizard_demo_clinical_metadata.csv",
        media_type="text/csv",
    )
    def download_example_clinical():
        yield _build_demo_clinical_csv()

    @render.download(
        filename="NMRMetaboWizard_demo_clinical_metadata.csv",
        media_type="text/csv",
    )
    def download_example_clinical_step():
        yield _build_demo_clinical_csv()

    # Outputs

    @render.text
    def status_text():
        return status_state.get()

    @render.text
    def error_text():
        return error_state.get()

    @render.text
    def processing_settings_text():
        lines = []

        samples = samples_state.get()
        if samples:
            log = samples[0].get("log", [])
            if log:
                lines.append("Preprocessing settings from first sample:")
                for item in log[-12:]:
                    lines.append(f"- {item}")

        aligned = filtered_state.get()
        if aligned is None:
            aligned = combined_state.get()

        if aligned is not None:
            summary = aligned.get("summary", {})
            lines.append("")
            lines.append("Clinical/analysis state:")
            lines.append(f"- matched samples with Class: {summary.get('n_matched_with_class', 'n/a')}")
            lines.append(f"- spectral features: {summary.get('n_features', 'n/a')}")
            if summary.get("n_removed_outliers", 0):
                lines.append(f"- removed outliers: {summary.get('n_removed_outliers')}")
                lines.append(f"- removed IDs: {summary.get('removed_outlier_ids', [])}")

        if not lines:
            return "No settings applied yet."

        return "\n".join(lines)

    @render.data_frame
    def sample_table():
        samples = samples_state.get()

        if not samples:
            return render.DataGrid(pd.DataFrame({"message": ["No samples loaded."]}))

        rows = []

        for i, sample in enumerate(samples):
            rows.append(
                {
                    "index": i,
                    "name": sample["name"],
                    "fid_points": len(sample["raw_fid"]),
                    "folder": sample["folder"],
                }
            )

        return render.DataGrid(pd.DataFrame(rows))

    @render_plotly
    def main_plot():
        _update_plot_style_from_input(input)
        samples = samples_state.get()
        step = current_step.get()

        if not samples:
            return _blank_plotly_global("No samples loaded yet")

        i = get_sample_index(input, samples)
        sample = samples[i]

        try:
            if step == "raw":
                y = as_real(sample["raw_fid"])
                return _plotly_line_figure(
                    [{"x": _fid_seconds(sample, y), "y": y, "name": "raw FID"}],
                    f"Raw FID — sample {i}: {sample['name']}",
                    **_fid_plot_kwargs(),
                )

            if step == "group":
                raw = as_real(sample["raw_fid"])
                if "group_delay_fid" in sample:
                    after = as_real(sample["group_delay_fid"])
                    return _plotly_fid_stack(
                        _fid_seconds(sample, raw),
                        raw,
                        _fid_seconds(sample, after),
                        after,
                        "Before group delay removal",
                        f"After group delay removal — removed {sample.get('group_delay_points', 0)} point(s)",
                        mode=_comparison_view(input),
                    )

                return _plotly_line_figure(
                    [{"x": _fid_seconds(sample, raw), "y": raw, "name": "before"}],
                    "Before group delay removal",
                    **_fid_plot_kwargs(),
                )

            if step == "solvent":
                if "solvent_removed_fid" in sample:
                    before = as_real(sample["group_delay_fid"])
                    solvent = as_real(sample["estimated_solvent"])
                    after = as_real(sample["solvent_removed_fid"])

                    return _plotly_stack_figure(
                        [
                            {
                                "title": "Before solvent residual removal + estimated residual",
                                "traces": [
                                    {"x": _fid_seconds(sample, before), "y": before, "name": "FID before"},
                                    {"x": _fid_seconds(sample, solvent), "y": solvent, "name": "estimated residual"},
                                ],
                            },
                            {
                                "title": "After solvent residual removal",
                                "traces": [{"x": _fid_seconds(sample, after), "y": after, "name": "after"}],
                            },
                        ],
                        **_fid_plot_kwargs(),
                    )

                y = as_real(sample["group_delay_fid"])
                return _plotly_line_figure(
                    [{"x": _fid_seconds(sample, y), "y": y, "name": "before"}],
                    "Before solvent residual removal",
                    **_fid_plot_kwargs(),
                )

            if step == "apod":
                if "apodized_fid" in sample:
                    before = as_real(sample["solvent_removed_fid"])
                    after = as_real(sample["apodized_fid"])
                    return _plotly_fid_stack(
                        _fid_seconds(sample, before),
                        before,
                        _fid_seconds(sample, after),
                        after,
                        "Before apodization",
                        f"After apodization — {sample.get('apodization_kind', '?')}, LB={sample.get('apodization_lb', '?')}",
                        mode=_comparison_view(input),
                    )

                y = as_real(sample["solvent_removed_fid"])
                return _plotly_line_figure(
                    [{"x": _fid_seconds(sample, y), "y": y, "name": "before"}],
                    "Before apodization",
                    **_fid_plot_kwargs(),
                )

            if step == "zero":
                if "zero_filled_fid" in sample:
                    before = as_real(sample["apodized_fid"])
                    after = as_real(sample["zero_filled_fid"])
                    return _plotly_fid_stack(
                        _fid_seconds(sample, before),
                        before,
                        _fid_seconds(sample, after),
                        after,
                        "Before zero filling",
                        f"After zero filling — length {len(after)}",
                        mode=_comparison_view(input),
                    )

                y = as_real(sample["apodized_fid"])
                return _plotly_line_figure(
                    [{"x": _fid_seconds(sample, y), "y": y, "name": "before"}],
                    "Before zero filling",
                    **_fid_plot_kwargs(),
                )

            if step == "fft":
                if "spectrum_real" not in sample:
                    return _blank_plotly_global("Click Apply Fourier transform")

                return _plotly_line_figure(
                    [{"x": sample["ppm"], "y": sample["spectrum_real"], "name": "spectrum"}],
                    "Spectrum after Fourier transform",
                    x_title="ppm",
                    y_title="Intensity",
                    reverse_x=True,
                    view_range=get_view_range(input),
                )

            if step == "phase":
                if "phased" in sample:
                    return _plotly_spectrum_stack(
                        input,
                        sample["ppm"],
                        sample["spectrum_real"],
                        sample["ppm"],
                        sample["phased"],
                        "Before phase correction",
                        "After phase correction",
                    )

                return _plotly_line_figure(
                    [{"x": sample["ppm"], "y": sample["spectrum_real"], "name": "before"}],
                    "Before phase correction",
                    x_title="ppm",
                    reverse_x=True,
                    view_range=get_view_range(input),
                )

            if step == "ref":
                if "referenced_ppm" in sample:
                    found = sample.get("found_reference")
                    title = "After internal referencing"
                    if found is not None:
                        title += f" — found {found:.4f} ppm"

                    return _plotly_spectrum_stack(
                        input,
                        sample["ppm"],
                        sample["phased"],
                        sample["referenced_ppm"],
                        sample["phased"],
                        "Before internal referencing",
                        title,
                    )

                return _plotly_line_figure(
                    [{"x": sample["ppm"], "y": sample["phased"], "name": "before"}],
                    "Before internal referencing",
                    x_title="ppm",
                    reverse_x=True,
                    view_range=get_view_range(input),
                )

            if step == "base":
                if "baseline_corrected" in sample:
                    ppm = sample["referenced_ppm"]
                    return _plotly_stack_figure(
                        [
                            {
                                "title": "Before baseline correction",
                                "traces": [
                                    {"x": ppm, "y": sample["phased"], "name": "before"},
                                    {"x": ppm, "y": sample["baseline"], "name": "estimated baseline"},
                                ],
                            },
                            {
                                "title": "After baseline correction",
                                "traces": [{"x": ppm, "y": sample["baseline_corrected"], "name": "corrected"}],
                            },
                        ],
                        x_title="ppm",
                        y_title="Intensity",
                        reverse_x=True,
                        view_range=get_view_range(input),
                    )

                return _plotly_line_figure(
                    [{"x": sample["referenced_ppm"], "y": sample["phased"], "name": "before"}],
                    "Before baseline correction",
                    x_title="ppm",
                    reverse_x=True,
                    view_range=get_view_range(input),
                )

            if step == "align":
                if "aligned" in sample:
                    title_after = f"After alignment — shift {sample.get('alignment_shift_points', 0)} point(s)"
                    return _plotly_spectrum_stack(
                        input,
                        sample["referenced_ppm"],
                        sample["baseline_corrected"],
                        sample["referenced_ppm"],
                        sample["aligned"],
                        "Before peak alignment",
                        title_after,
                    )

                return _plotly_line_figure(
                    [{"x": sample["referenced_ppm"], "y": sample["baseline_corrected"], "name": "before"}],
                    "Before peak alignment",
                    x_title="ppm",
                    reverse_x=True,
                    view_range=get_view_range(input),
                )

            if step == "neg":
                source_key = "aligned" if "aligned" in sample else "baseline_corrected"
                if "negative_zeroed" in sample:
                    return _plotly_spectrum_stack(
                        input,
                        sample["referenced_ppm"],
                        sample[source_key],
                        sample["referenced_ppm"],
                        sample["negative_zeroed"],
                        "Before negative-value zeroing",
                        "After negative-value zeroing",
                    )

                return _plotly_line_figure(
                    [{"x": sample["referenced_ppm"], "y": sample[source_key], "name": "before"}],
                    "Before negative-value zeroing",
                    x_title="ppm",
                    reverse_x=True,
                    view_range=get_view_range(input),
                )

            if step == "window":
                if "window_intensity" in sample:
                    return _plotly_stack_figure(
                        [
                            {
                                "title": "Before window selection",
                                "traces": [{"x": sample["referenced_ppm"], "y": sample["negative_zeroed"], "name": "before"}],
                            },
                            {
                                "title": f"After window selection — {sample.get('window_range', ('?', '?'))[0]}-{sample.get('window_range', ('?', '?'))[1]} ppm",
                                "traces": [{"x": sample["window_ppm"], "y": sample["window_intensity"], "name": "after"}],
                            },
                        ],
                        x_title="ppm",
                        y_title="Intensity",
                        reverse_x=True,
                        view_range=get_view_range(input),
                    )

                return _plotly_line_figure(
                    [{"x": sample["referenced_ppm"], "y": sample["negative_zeroed"], "name": "before"}],
                    "Before window selection",
                    x_title="ppm",
                    reverse_x=True,
                    view_range=get_view_range(input),
                )

            if step == "region":
                if "region_removed" in sample:
                    return _plotly_spectrum_stack(
                        input,
                        sample["window_ppm"],
                        sample["window_intensity"],
                        sample["window_ppm"],
                        sample["region_removed"],
                        "Before region removal",
                        f"After region removal — {sample.get('region_text', '')}, mode={sample.get('region_mode', '')}",
                    )

                return _plotly_line_figure(
                    [{"x": sample["window_ppm"], "y": sample["window_intensity"], "name": "before"}],
                    "Before region removal",
                    x_title="ppm",
                    reverse_x=True,
                    view_range=get_view_range(input),
                )

            if step == "bin":
                if "region_removed" in sample:
                    fig = _plotly_line_figure(
                        [
                            {"x": sample["window_ppm"], "y": sample["region_removed"], "name": "processed spectrum before binning"},
                        ],
                        "Processed spectrum with bin intervals",
                        x_title="ppm",
                        y_title="Intensity",
                        reverse_x=True,
                        view_range=get_view_range(input),
                    )
                    return _add_bin_overlays(fig, sample.get("bin_edges"), view_range=get_view_range(input))

                return _blank_plotly_global("Apply region removal/window selection before binning")

            if step == "norm":
                if "region_removed" in sample:
                    fig = _plotly_line_figure(
                        [
                            {"x": sample["window_ppm"], "y": sample["region_removed"], "name": "processed spectrum used for table generation"},
                        ],
                        "Processed spectrum used for binning/normalization",
                        x_title="ppm",
                        y_title="Intensity",
                        reverse_x=True,
                        view_range=get_view_range(input),
                    )
                    return _add_bin_overlays(fig, sample.get("bin_edges"), view_range=get_view_range(input))

                return _blank_plotly_global("Create binned table first")

            return _blank_plotly_global("No plot for this step")

        except Exception:
            return _blank_plotly_global(traceback.format_exc(), height=650)

    def _stage_df(sample_name: str, step: str, stage: str, x_name: str, x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        n = min(len(x), len(y))
        return pd.DataFrame(
            {
                "sample_id": sample_name,
                "step": step,
                "stage": stage,
                x_name: x[:n],
                "intensity": y[:n],
            }
        )

    def _current_plot_data_frame():
        samples = samples_state.get()
        step = current_step.get()

        if not samples:
            return pd.DataFrame({"message": ["No samples loaded."]})

        i = get_sample_index(input, samples)
        sample = samples[i]
        sample_name = sample.get("name", str(i))

        try:
            if step == "raw":
                y = as_real(sample["raw_fid"])
                return _stage_df(sample_name, step, "raw FID", "time_s", _fid_seconds(sample, y), y)

            if step == "group":
                frames = [_stage_df(sample_name, step, "before", "time_s", _fid_seconds(sample, sample["raw_fid"]), as_real(sample["raw_fid"]))]
                if "group_delay_fid" in sample:
                    frames.append(_stage_df(sample_name, step, "after", "time_s", _fid_seconds(sample, sample["group_delay_fid"]), as_real(sample["group_delay_fid"])))
                return pd.concat(frames, ignore_index=True)

            if step == "solvent":
                frames = []
                if "group_delay_fid" in sample:
                    frames.append(_stage_df(sample_name, step, "before", "time_s", _fid_seconds(sample, sample["group_delay_fid"]), as_real(sample["group_delay_fid"])))
                if "estimated_solvent" in sample:
                    frames.append(_stage_df(sample_name, step, "estimated solvent", "time_s", _fid_seconds(sample, sample["estimated_solvent"]), as_real(sample["estimated_solvent"])))
                if "solvent_removed_fid" in sample:
                    frames.append(_stage_df(sample_name, step, "after", "time_s", _fid_seconds(sample, sample["solvent_removed_fid"]), as_real(sample["solvent_removed_fid"])))
                return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"message": ["No solvent-step data available."]})

            if step == "apod":
                frames = []
                if "solvent_removed_fid" in sample:
                    frames.append(_stage_df(sample_name, step, "before", "time_s", _fid_seconds(sample, sample["solvent_removed_fid"]), as_real(sample["solvent_removed_fid"])))
                if "apodized_fid" in sample:
                    frames.append(_stage_df(sample_name, step, "after", "time_s", _fid_seconds(sample, sample["apodized_fid"]), as_real(sample["apodized_fid"])))
                return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"message": ["No apodization data available."]})

            if step == "zero":
                frames = []
                if "apodized_fid" in sample:
                    frames.append(_stage_df(sample_name, step, "before", "time_s", _fid_seconds(sample, sample["apodized_fid"]), as_real(sample["apodized_fid"])))
                if "zero_filled_fid" in sample:
                    frames.append(_stage_df(sample_name, step, "after", "time_s", _fid_seconds(sample, sample["zero_filled_fid"]), as_real(sample["zero_filled_fid"])))
                return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"message": ["No zero-filling data available."]})

            if step == "fft":
                if "spectrum_real" in sample:
                    return _stage_df(sample_name, step, "spectrum", "ppm", sample["ppm"], sample["spectrum_real"])

            if step == "phase":
                frames = []
                if "spectrum_real" in sample:
                    frames.append(_stage_df(sample_name, step, "before", "ppm", sample["ppm"], sample["spectrum_real"]))
                if "phased" in sample:
                    frames.append(_stage_df(sample_name, step, "after", "ppm", sample["ppm"], sample["phased"]))
                return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"message": ["No phase-correction data available."]})

            if step == "ref":
                frames = []
                if "phased" in sample and "ppm" in sample:
                    frames.append(_stage_df(sample_name, step, "before", "ppm", sample["ppm"], sample["phased"]))
                if "referenced_ppm" in sample and "phased" in sample:
                    frames.append(_stage_df(sample_name, step, "after", "ppm", sample["referenced_ppm"], sample["phased"]))
                return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"message": ["No referencing data available."]})

            if step == "base":
                frames = []
                if "referenced_ppm" in sample and "phased" in sample:
                    frames.append(_stage_df(sample_name, step, "before", "ppm", sample["referenced_ppm"], sample["phased"]))
                if "baseline" in sample:
                    frames.append(_stage_df(sample_name, step, "estimated baseline", "ppm", sample["referenced_ppm"], sample["baseline"]))
                if "baseline_corrected" in sample:
                    frames.append(_stage_df(sample_name, step, "after", "ppm", sample["referenced_ppm"], sample["baseline_corrected"]))
                return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"message": ["No baseline data available."]})

            if step == "align":
                frames = []
                if "baseline_corrected" in sample:
                    frames.append(_stage_df(sample_name, step, "before", "ppm", sample["referenced_ppm"], sample["baseline_corrected"]))
                if "aligned" in sample:
                    frames.append(_stage_df(sample_name, step, "after", "ppm", sample["referenced_ppm"], sample["aligned"]))
                return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"message": ["No alignment data available."]})

            if step == "neg":
                source_key = "aligned" if "aligned" in sample else "baseline_corrected"
                frames = []
                if source_key in sample:
                    frames.append(_stage_df(sample_name, step, "before", "ppm", sample["referenced_ppm"], sample[source_key]))
                if "negative_zeroed" in sample:
                    frames.append(_stage_df(sample_name, step, "after", "ppm", sample["referenced_ppm"], sample["negative_zeroed"]))
                return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"message": ["No negative-zeroing data available."]})

            if step == "window":
                frames = []
                if "negative_zeroed" in sample:
                    frames.append(_stage_df(sample_name, step, "before", "ppm", sample["referenced_ppm"], sample["negative_zeroed"]))
                if "window_intensity" in sample:
                    frames.append(_stage_df(sample_name, step, "after", "ppm", sample["window_ppm"], sample["window_intensity"]))
                return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"message": ["No window-selection data available."]})

            if step == "region":
                frames = []
                if "window_intensity" in sample:
                    frames.append(_stage_df(sample_name, step, "before", "ppm", sample["window_ppm"], sample["window_intensity"]))
                if "region_removed" in sample:
                    frames.append(_stage_df(sample_name, step, "after", "ppm", sample["window_ppm"], sample["region_removed"]))
                return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame({"message": ["No region-removal data available."]})

            if step in ["bin", "norm"]:
                if "region_removed" in sample:
                    df = _stage_df(sample_name, step, "processed spectrum", "ppm", sample["window_ppm"], sample["region_removed"])
                    if "bin_edges" in sample:
                        edges = np.asarray(sample["bin_edges"])
                        df["note"] = "Bin edges are shown visually on the plot. Download the binned table for integrated bin values."
                    return df
                return pd.DataFrame({"message": ["No binning-spectrum data available."]})

            return pd.DataFrame({"message": [f"No exportable plot data for step: {step}"]})
        except Exception as e:
            return pd.DataFrame({"error": [str(e)]})

    @render.download(filename="current_plot_data.csv")
    def download_current_plot_data():
        yield _current_plot_data_frame().to_csv(index=False)

    @render.data_frame
    def binned_preview():
        df = binned_state.get()

        if df is None:
            return render.DataGrid(pd.DataFrame({"message": ["Click Create binned table first."]}))

        preview = df.head(20).reset_index().rename(columns={"index": "sample"})
        return render.DataGrid(preview)

    @render.data_frame
    def normalized_preview():
        df = normalized_state.get()

        if df is None:
            return render.DataGrid(pd.DataFrame({"message": ["Click Create normalized table first."]}))

        preview = df.head(20).reset_index().rename(columns={"index": "sample"})
        return render.DataGrid(preview)


    @render.text
    def merge_summary():
        aligned = combined_state.get()

        if aligned is None:
            return "No clinical data aligned yet."

        summary = aligned["summary"]

        lines = []
        lines.append("Alignment structure:")
        lines.append("  X = spectral bins only")
        lines.append("  y = Class labels")
        lines.append("  metadata = clinical variables aligned by sample ID")
        lines.append("")
        lines.append(f"Spectral samples: {summary['n_spectra']}")
        lines.append(f"Clinical rows: {summary['n_clinical']}")
        lines.append(f"Matched samples before class filtering: {summary['n_matched_before_class_filter']}")
        lines.append(f"Matched samples with non-empty Class: {summary['n_matched_with_class']}")
        lines.append(f"Spectral features: {summary['n_features']}")
        lines.append(f"Unmatched spectra: {summary['n_unmatched_spectra']}")
        lines.append(f"Unmatched clinical rows: {summary['n_unmatched_clinical']}")
        lines.append("")
        lines.append(f"Clinical ID column: {summary['clinical_id_col']}")
        lines.append(f"Class column: {summary['class_col']}")
        lines.append("")
        lines.append("Example NMR sample IDs:")
        lines.extend([f"  - {x}" for x in summary.get("example_spectra_ids", [])[:10]])
        lines.append("")
        lines.append("Example clinical IDs:")
        lines.extend([f"  - {x}" for x in summary.get("example_clinical_ids", [])[:10]])

        if summary.get("duplicated_clinical_match_keys"):
            lines.append("")
            lines.append("Duplicated clinical IDs after cleanup, first occurrence used:")
            lines.extend([f"  - {x}" for x in summary["duplicated_clinical_match_keys"][:10]])

        if summary["unmatched_spectra"]:
            lines.append("")
            lines.append("First unmatched spectra keys:")
            lines.extend([f"  - {x}" for x in summary["unmatched_spectra"][:10]])

        if summary["unmatched_clinical"]:
            lines.append("")
            lines.append("First unmatched clinical keys:")
            lines.extend([f"  - {x}" for x in summary["unmatched_clinical"][:10]])

        return "\n".join(lines)

    @render.data_frame
    def clinical_preview():
        clinical = clinical_state.get()

        if clinical is None:
            return render.DataGrid(pd.DataFrame({"message": ["Upload and align clinical data first."]}))

        return render.DataGrid(clinical.head(20))

    @render.data_frame
    def merged_preview():
        aligned = combined_state.get()

        if aligned is None:
            return render.DataGrid(pd.DataFrame({"message": ["Align clinical data first."]}))

        table = aligned["sample_table"].copy()

        if table.empty:
            return render.DataGrid(pd.DataFrame({"message": ["No sample IDs matched. Check NMR sample names and clinical study_id."]}))

        return render.DataGrid(table.head(50))

    @render.data_frame
    def pca_variance_table():
        eda = eda_state.get()

        if eda is None:
            return render.DataGrid(pd.DataFrame({"message": ["Run EDA first."]}))

        return render.DataGrid(eda["pca"]["variance"])

    @render.data_frame
    def pls_variance_table():
        eda = eda_state.get()

        if eda is None:
            return render.DataGrid(pd.DataFrame({"message": ["Run EDA first."]}))

        return render.DataGrid(eda["plsda"]["variance"])

    @render.data_frame
    def class_counts_table():
        eda = eda_state.get()

        if eda is None:
            return render.DataGrid(pd.DataFrame({"message": ["Run EDA first."]}))

        return render.DataGrid(eda["class_counts"])

    @render.data_frame
    def univariate_table():
        eda = eda_state.get()

        if eda is None:
            return render.DataGrid(pd.DataFrame({"message": ["Run EDA first."]}))

        return render.DataGrid(eda["univariate"])

    @render.data_frame
    def clinical_corr_table():
        eda = eda_state.get()

        if eda is None:
            return render.DataGrid(pd.DataFrame({"message": ["Run EDA first."]}))

        corr = eda["clinical_corr"]

        if corr is None or corr.empty:
            return render.DataGrid(pd.DataFrame({"message": ["Not enough numeric clinical variables for correlation matrix."]}))

        return render.DataGrid(corr.reset_index().rename(columns={"index": "variable"}))

    @render.data_frame
    def feature_clinical_corr_table():
        eda = eda_state.get()

        if eda is None:
            return render.DataGrid(pd.DataFrame({"message": ["Run EDA first."]}))

        return render.DataGrid(eda["feature_clinical_corr"])

    def _component_column(prefix: str, component_number: int) -> str:
        component_number = max(1, int(component_number))
        return f"{prefix}{component_number}"

    def _plot_scores_2d(ax, scores, x_col, y_col, title, x_label=None, y_label=None):
        classes = list(scores["class"].unique())

        for cls in classes:
            sub = scores[scores["class"] == cls]
            ax.scatter(sub[x_col], sub[y_col], label=str(cls), s=55, alpha=0.85)

        ax.set_xlabel(x_label or x_col)
        ax.set_ylabel(y_label or y_col)
        ax.set_title(title)
        ax.legend()

    def _plot_scores_3d(fig, scores, x_col, y_col, z_col, title, x_label=None, y_label=None, z_label=None):
        ax3d = fig.add_subplot(111, projection="3d")

        classes = list(scores["class"].unique())

        for cls in classes:
            sub = scores[scores["class"] == cls]
            ax3d.scatter(sub[x_col], sub[y_col], sub[z_col], label=str(cls), s=55, alpha=0.85)

        ax3d.set_xlabel(x_label or x_col)
        ax3d.set_ylabel(y_label or y_col)
        ax3d.set_zlabel(z_label or z_col)
        ax3d.set_title(title)
        ax3d.legend()

        return ax3d

    def _blank_plotly(message: str):
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        fig.update_layout(height=600)
        return _apply_global_plot_style(fig)

    def _score_axis_label(prefix: str, col: str, variance: pd.DataFrame):
        if prefix == "PC":
            row = variance[variance["component"] == col]
            if len(row) == 1 and "explained_variance_percent" in row.columns:
                return f"{col} ({row['explained_variance_percent'].iloc[0]:.1f}%)"
            return col

        row = variance[variance["component"] == col]
        if len(row) == 1 and "approx_x_variance_percent" in row.columns:
            return f"{col} (approx X var {row['approx_x_variance_percent'].iloc[0]:.1f}%)"
        return col

    def _find_eda_column(columns, requested):
        requested = str(requested or "").strip()

        if requested in columns:
            return requested

        lower_map = {str(c).lower(): c for c in columns}

        if requested.lower() in lower_map:
            return lower_map[requested.lower()]

        if requested.lower() == "psa":
            for c in columns:
                if str(c).lower() == "psa":
                    return c
            for c in columns:
                if "psa" in str(c).lower():
                    return c

        return None

    def _get_score_data_with_metadata(eda, plot_type, psa_col="psa", psa_cutoff=4.0):
        if plot_type == "PCA scores":
            scores = eda["pca"]["scores"].copy()
        else:
            scores = eda["plsda"]["scores"].copy()

        aligned = eda.get("aligned")

        if aligned is not None:
            clinical = aligned.get("clinical_aligned", pd.DataFrame()).copy()

            if clinical is not None and not clinical.empty:
                clinical = clinical.copy()
                clinical.index = clinical.index.astype(str)

                scores["sample_id"] = scores["sample_id"].astype(str)
                scores = scores.merge(
                    clinical.reset_index().rename(columns={"index": "sample_id"}),
                    on="sample_id",
                    how="left",
                    suffixes=("", "_clinical"),
                )

        scores["Class"] = scores["class"].astype(str)

        psa_found = _find_eda_column(scores.columns, psa_col)

        if psa_found is not None:
            psa_numeric = pd.to_numeric(scores[psa_found], errors="coerce")
            scores["PSA value"] = psa_numeric
            scores["PSA group"] = np.where(
                psa_numeric < float(psa_cutoff),
                f"PSA < {float(psa_cutoff):g}",
                f"PSA ≥ {float(psa_cutoff):g}",
            )
            scores.loc[psa_numeric.isna(), "PSA group"] = "PSA missing"
        else:
            scores["PSA value"] = np.nan
            scores["PSA group"] = "PSA unavailable"

        scores["Class + PSA group"] = scores["Class"].astype(str) + " | " + scores["PSA group"].astype(str)

        return scores

    def _palette_sequence(name: str):
        name = str(name)

        if name == "Viridis":
            return px.colors.sequential.Viridis
        if name == "Plasma":
            return px.colors.sequential.Plasma
        if name == "Turbo":
            return px.colors.sequential.Turbo
        if name == "Cividis":
            return px.colors.sequential.Cividis

        return px.colors.qualitative.Plotly

    def _score_plotly(
        eda,
        plot_type,
        dimension,
        x_component,
        y_component,
        z_component,
        title_suffix="",
        color_by="Class",
        marker_size=8,
        palette="Plotly",
        psa_col="psa",
        psa_cutoff=4.0,
    ):
        if plot_type == "PCA scores":
            base_scores = eda["pca"]["scores"]
            variance = eda["pca"]["variance"]
            prefix = "PC"
            title_base = "PCA scores"
        else:
            base_scores = eda["plsda"]["scores"]
            variance = eda["plsda"]["variance"]
            prefix = "LV"
            title_base = "PLS-DA scores"

        scores = _get_score_data_with_metadata(
            eda,
            plot_type,
            psa_col=psa_col,
            psa_cutoff=psa_cutoff,
        )

        x_col = f"{prefix}{int(x_component)}"
        y_col = f"{prefix}{int(y_component)}"
        z_col = f"{prefix}{int(z_component)}"

        available = [c for c in base_scores.columns if c.startswith(prefix)]

        if x_col not in scores.columns or y_col not in scores.columns:
            return _blank_plotly(f"Selected components are not available. Available: {', '.join(available)}")

        color_col = _find_eda_column(scores.columns, color_by) or "Class"
        marker_size = int(marker_size)

        fig = go.Figure()

        numeric_color = pd.to_numeric(scores[color_col], errors="coerce")
        is_numeric_color = numeric_color.notna().sum() >= 3 and color_col not in ["Class", "class", "PSA group", "Class + PSA group"]

        if dimension == "3D":
            if z_col not in scores.columns:
                return _blank_plotly(f"{z_col} is not available. Available: {', '.join(available)}")

            if is_numeric_color:
                fig.add_trace(
                    go.Scatter3d(
                        x=scores[x_col],
                        y=scores[y_col],
                        z=scores[z_col],
                        mode="markers",
                        text=scores["sample_id"],
                        marker=dict(
                            size=marker_size,
                            color=numeric_color,
                            colorscale=palette if palette != "Plotly" else "Viridis",
                            colorbar=dict(title=color_col),
                        ),
                        customdata=np.stack([scores["Class"].astype(str), scores[color_col].astype(str)], axis=-1),
                        hovertemplate=(
                            "Sample: %{text}<br>"
                            + f"{x_col}: " + "%{x:.3f}<br>"
                            + f"{y_col}: " + "%{y:.3f}<br>"
                            + f"{z_col}: " + "%{z:.3f}<br>"
                            "Class: %{customdata[0]}<br>"
                            + f"{color_col}: " + "%{customdata[1]}<extra></extra>"
                        ),
                    )
                )
            else:
                categories = scores[color_col].astype(str).fillna("missing")
                colors = _palette_sequence(palette)
                for idx_cat, cat in enumerate(pd.Series(categories).drop_duplicates()):
                    sub = scores[categories == cat]
                    color_value = colors[idx_cat % len(colors)]
                    fig.add_trace(
                        go.Scatter3d(
                            x=sub[x_col],
                            y=sub[y_col],
                            z=sub[z_col],
                            mode="markers",
                            name=str(cat),
                            text=sub["sample_id"],
                            marker=dict(size=marker_size, color=color_value),
                            customdata=np.stack([sub["Class"].astype(str), sub[color_col].astype(str)], axis=-1),
                            hovertemplate=(
                                "Sample: %{text}<br>"
                                + f"{x_col}: " + "%{x:.3f}<br>"
                                + f"{y_col}: " + "%{y:.3f}<br>"
                                + f"{z_col}: " + "%{z:.3f}<br>"
                                "Class: %{customdata[0]}<br>"
                                + f"{color_col}: " + "%{customdata[1]}<extra></extra>"
                            ),
                        )
                    )

            fig.update_layout(
                title=f"3D {title_base}{title_suffix} — colored by {color_col}",
                scene=dict(
                    xaxis_title=_score_axis_label(prefix, x_col, variance),
                    yaxis_title=_score_axis_label(prefix, y_col, variance),
                    zaxis_title=_score_axis_label(prefix, z_col, variance),
                ),
                height=650,
            )
            return _apply_global_plot_style(fig)

        if is_numeric_color:
            fig.add_trace(
                go.Scattergl(
                    x=scores[x_col],
                    y=scores[y_col],
                    mode="markers",
                    text=scores["sample_id"],
                    marker=dict(
                        size=marker_size,
                        color=numeric_color,
                        colorscale=palette if palette != "Plotly" else "Viridis",
                        colorbar=dict(title=color_col),
                    ),
                    customdata=np.stack([scores["Class"].astype(str), scores[color_col].astype(str)], axis=-1),
                    hovertemplate=(
                        "Sample: %{text}<br>"
                        + f"{x_col}: " + "%{x:.3f}<br>"
                        + f"{y_col}: " + "%{y:.3f}<br>"
                        "Class: %{customdata[0]}<br>"
                        + f"{color_col}: " + "%{customdata[1]}<extra></extra>"
                    ),
                )
            )
        else:
            categories = scores[color_col].astype(str).fillna("missing")
            colors = _palette_sequence(palette)
            for idx_cat, cat in enumerate(pd.Series(categories).drop_duplicates()):
                sub = scores[categories == cat]
                color_value = colors[idx_cat % len(colors)]
                fig.add_trace(
                    go.Scattergl(
                        x=sub[x_col],
                        y=sub[y_col],
                        mode="markers",
                        name=str(cat),
                        text=sub["sample_id"],
                        marker=dict(size=marker_size, color=color_value),
                        customdata=np.stack([sub["Class"].astype(str), sub[color_col].astype(str)], axis=-1),
                        hovertemplate=(
                            "Sample: %{text}<br>"
                            + f"{x_col}: " + "%{x:.3f}<br>"
                            + f"{y_col}: " + "%{y:.3f}<br>"
                            "Class: %{customdata[0]}<br>"
                            + f"{color_col}: " + "%{customdata[1]}<extra></extra>"
                        ),
                    )
                )

        fig.update_layout(
            title=f"2D {title_base}{title_suffix} — colored by {color_col}",
            xaxis_title=_score_axis_label(prefix, x_col, variance),
            yaxis_title=_score_axis_label(prefix, y_col, variance),
            height=650,
        )
        return _apply_global_plot_style(fig)

    def _psa_column_and_values(eda, psa_col="psa"):
        aligned = eda.get("aligned")

        if aligned is None:
            return None, pd.Series(dtype=float), pd.DataFrame()

        clinical = aligned.get("clinical_aligned", pd.DataFrame()).copy()

        if clinical is None or clinical.empty:
            return None, pd.Series(dtype=float), pd.DataFrame()

        found = _find_eda_column(clinical.columns, psa_col)

        if found is None:
            return None, pd.Series(dtype=float), clinical

        return found, pd.to_numeric(clinical[found], errors="coerce"), clinical

    def _plot_psa_by_class(eda, psa_col="psa", psa_cutoff=4.0, title_suffix=""):
        found, psa, clinical = _psa_column_and_values(eda, psa_col)

        if found is None:
            return _blank_plotly("No PSA column found.")

        y = eda["aligned"]["y"]
        df = pd.DataFrame(
            {
                "sample_id": y.index.astype(str),
                "Class": y.values,
                "PSA": psa.values,
            }
        ).dropna(subset=["PSA"])

        if df.empty:
            return _blank_plotly("No valid PSA values.")

        fig = go.Figure()

        for cls in df["Class"].astype(str).unique():
            sub = df[df["Class"].astype(str) == str(cls)]
            fig.add_trace(
                go.Box(
                    y=sub["PSA"],
                    x=sub["Class"],
                    name=str(cls),
                    boxpoints="all",
                    text=sub["sample_id"],
                    hovertemplate="Sample: %{text}<br>Class: %{x}<br>PSA: %{y:.3f}<extra></extra>",
                )
            )

        fig.add_hline(y=float(psa_cutoff), line_dash="dash", annotation_text=f"PSA cutoff {float(psa_cutoff):g}")
        fig.update_layout(title=f"PSA by Class{title_suffix}", xaxis_title="Class", yaxis_title=found, height=680)
        return _apply_global_plot_style(fig)

    def _plot_class_by_psa_group(eda, psa_col="psa", psa_cutoff=4.0, title_suffix=""):
        found, psa, clinical = _psa_column_and_values(eda, psa_col)

        if found is None:
            return _blank_plotly("No PSA column found.")

        y = eda["aligned"]["y"]
        df = pd.DataFrame(
            {
                "Class": y.values,
                "PSA": psa.values,
            }
        ).dropna(subset=["PSA"])

        if df.empty:
            return _blank_plotly("No valid PSA values.")

        df["PSA group"] = np.where(df["PSA"] < float(psa_cutoff), f"PSA < {float(psa_cutoff):g}", f"PSA ≥ {float(psa_cutoff):g}")
        counts = df.groupby(["Class", "PSA group"], dropna=False).size().reset_index(name="count")

        fig = go.Figure()

        for group in counts["PSA group"].unique():
            sub = counts[counts["PSA group"] == group]
            fig.add_trace(go.Bar(x=sub["Class"].astype(str), y=sub["count"], name=str(group)))

        fig.update_layout(
            title=f"Class counts by PSA group{title_suffix}",
            xaxis_title="Class",
            yaxis_title="Sample count",
            barmode="group",
            height=640,
        )
        return _apply_global_plot_style(fig)

    def _plot_psa_feature_correlations(eda, psa_col="psa", title_suffix=""):
        table = eda.get("feature_clinical_corr", pd.DataFrame())

        if table is None or table.empty or "clinical_variable" not in table.columns:
            return _blank_plotly("No clinical-feature correlations available.")

        found = _find_eda_column(table["clinical_variable"].astype(str).unique(), psa_col)

        if found is None:
            matches = table[table["clinical_variable"].astype(str).str.lower().str.contains("psa", na=False)]
        else:
            matches = table[table["clinical_variable"].astype(str).str.lower() == str(found).lower()]

        if matches.empty:
            return _blank_plotly("No PSA-feature correlations found.")

        matches = matches.sort_values("abs_rho", ascending=False).head(30)

        fig = go.Figure(
            data=[
                go.Bar(
                    x=matches["spearman_rho"],
                    y=matches["feature_ppm"].astype(str),
                    orientation="h",
                    text=matches["feature_ppm"].astype(str),
                    hovertemplate="ppm: %{y}<br>Spearman rho: %{x:.3f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title=f"Top PSA-feature Spearman correlations{title_suffix}",
            xaxis_title="Spearman rho",
            yaxis_title="ppm bin",
            height=650,
        )
        fig.update_yaxes(autorange="reversed")
        return _apply_global_plot_style(fig)


    def _pca_loading_plotly(eda, component_number=1, n_components=1, title_suffix=""):
        """PCA loading bar plot.

        Shows one or more PC loading vectors as bar plots. For numeric ppm bins,
        the x-axis is shown as ppm with integer tick labels (0, 1, 2, ...), and
        the axis is reversed in the conventional NMR direction.
        """
        pca = eda.get("pca", {}) if eda is not None else {}
        loadings = pca.get("loadings", pd.DataFrame()).copy()

        if loadings is None or loadings.empty:
            return _blank_plotly("No PCA loading data available. Run EDA first.")

        component_number = max(1, int(component_number))
        n_components = max(1, int(n_components))

        available_components = [c for c in loadings.columns if str(c).startswith("PC")]
        if not available_components:
            return _blank_plotly("No PCA loading columns are available.")

        selected_components = [f"PC{i}" for i in range(component_number, component_number + n_components)]
        selected_components = [c for c in selected_components if c in loadings.columns]

        if not selected_components:
            return _blank_plotly(
                f"No requested PCs are available. Available loading columns: {', '.join(available_components)}"
            )

        base = loadings[["feature_ppm"] + selected_components].copy()
        base["feature_ppm_numeric"] = pd.to_numeric(base["feature_ppm"], errors="coerce")

        use_numeric = base["feature_ppm_numeric"].notna().sum() >= 2
        if use_numeric:
            base = base.sort_values("feature_ppm_numeric")
            x_values = base["feature_ppm_numeric"].astype(float)
            x_title = "ppm"
            hover_x = base["feature_ppm"].astype(str)

            unique_x = np.sort(pd.Series(x_values).dropna().unique())
            if len(unique_x) >= 2:
                bar_width = float(np.nanmedian(np.abs(np.diff(unique_x)))) * 0.85
            else:
                bar_width = 0.01

            tick_start = int(np.floor(float(np.nanmin(unique_x))))
            tick_end = int(np.ceil(float(np.nanmax(unique_x))))
            tickvals = list(range(tick_start, tick_end + 1))
            ticktext = [str(v) for v in tickvals]
        else:
            x_values = base["feature_ppm"].astype(str)
            x_title = "feature / ppm bin"
            hover_x = base["feature_ppm"].astype(str)
            bar_width = None
            tickvals = None
            ticktext = None

        from plotly.subplots import make_subplots

        n = len(selected_components)
        cols = 2 if n > 1 else 1
        rows = int(np.ceil(n / cols))
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[f"{pc} loadings" for pc in selected_components],
            vertical_spacing=0.14,
            horizontal_spacing=0.08,
        )

        for idx, component in enumerate(selected_components):
            df = pd.DataFrame({
                "x": x_values,
                "hover_x": hover_x,
                "loading": pd.to_numeric(base[component], errors="coerce"),
            }).dropna(subset=["loading"])

            row = idx // cols + 1
            col = idx % cols + 1

            bar_kwargs = dict(
                x=df["x"],
                y=df["loading"],
                name=component,
                marker_color=BEFORE_COLOR,
                customdata=df["hover_x"],
                hovertemplate="ppm/bin: %{customdata}<br>Loading: %{y:.6g}<extra></extra>",
                showlegend=False,
            )
            if use_numeric and bar_width is not None:
                bar_kwargs["width"] = bar_width

            fig.add_trace(go.Bar(**bar_kwargs), row=row, col=col)
            fig.add_hline(y=0, line_dash="dash", line_color="gray", row=row, col=col)

            if use_numeric:
                fig.update_xaxes(
                    autorange="reversed",
                    tickmode="array",
                    tickvals=tickvals,
                    ticktext=ticktext,
                    title_text=x_title,
                    row=row,
                    col=col,
                )
            else:
                fig.update_xaxes(title_text=x_title, row=row, col=col)

            fig.update_yaxes(title_text="Loading", row=row, col=col)

        total_height = max(520, 330 * rows)
        if n == 1:
            title = f"PCA loading bar plot for {selected_components[0]}{title_suffix}"
        else:
            title = f"PCA loading bar plots for {selected_components[0]}–{selected_components[-1]}{title_suffix}"

        fig.update_layout(
            title=title,
            height=total_height,
            bargap=0.05,
        )
        return _apply_global_plot_style(fig)


    @render_plotly
    def eda_plot():
        _update_plot_style_from_input(input)
        eda = eda_state.get()

        if eda is None:
            return _blank_plotly("Run EDA first")

        plot_type = input.eda_plot_type()

        if plot_type in ["PCA scores", "PLS-DA scores"]:
            return _score_plotly(
                eda,
                plot_type,
                input.score_dimension(),
                input.score_x_component(),
                input.score_y_component(),
                input.score_z_component(),
                color_by=input.score_color_by(),
                marker_size=input.score_marker_size(),
                palette=input.score_palette(),
                psa_col=input.eda_psa_column(),
                psa_cutoff=float(input.eda_psa_cutoff()),
            )

        if plot_type == "PCA loadings":
            return _pca_loading_plotly(eda, component_number=input.score_x_component(), n_components=input.loading_n_components())

        if plot_type == "Class counts":
            counts = eda["class_counts"]
            fig = go.Figure(
                data=[
                    go.Bar(
                        x=counts["class"].astype(str),
                        y=counts["count"],
                    )
                ]
            )
            fig.update_layout(title="Class counts", xaxis_title="Class", yaxis_title="Count", height=600)
            return _apply_global_plot_style(fig)

        if plot_type == "Top univariate bins":
            uni = eda["univariate"].copy()
            if "p_value" not in uni.columns:
                return _blank_plotly("No univariate results available")

            uni = uni.head(20).copy()
            y_values = -np.log10(uni["p_value"].replace(0, np.nan))
            y_values = y_values.fillna(y_values.max() if y_values.notna().any() else 0)

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=y_values,
                        y=uni["feature_ppm"].astype(str),
                        orientation="h",
                    )
                ]
            )
            fig.update_layout(
                title="Top bins by univariate p-value",
                xaxis_title="-log10(p-value)",
                yaxis_title="ppm bin",
                height=650,
            )
            fig.update_yaxes(autorange="reversed")
            return _apply_global_plot_style(fig)

        if plot_type == "Clinical correlation heatmap":
            corr = eda["clinical_corr"]

            if corr is None or corr.empty:
                return _blank_plotly("Not enough numeric clinical variables")

            fig = go.Figure(
                data=[
                    go.Heatmap(
                        z=corr.values,
                        x=list(corr.columns),
                        y=list(corr.index),
                        zmin=-1,
                        zmax=1,
                        colorbar=dict(title="Spearman rho"),
                    )
                ]
            )
            fig.update_layout(title="Spearman correlation among clinical variables", height=650)
            return _apply_global_plot_style(fig)

        if plot_type == "PSA by Class":
            return _plot_psa_by_class(
                eda,
                psa_col=input.eda_psa_column(),
                psa_cutoff=float(input.eda_psa_cutoff()),
            )

        if plot_type == "Class by PSA group":
            return _plot_class_by_psa_group(
                eda,
                psa_col=input.eda_psa_column(),
                psa_cutoff=float(input.eda_psa_cutoff()),
            )

        if plot_type == "PSA-feature correlations":
            return _plot_psa_feature_correlations(
                eda,
                psa_col=input.eda_psa_column(),
            )

        return _blank_plotly("Unknown EDA plot")

    @render.text
    def ml_metrics_text():
        ml = ml_state.get()

        if ml is None:
            return "Run ML first."

        return metrics_to_text(ml)

    @render_plotly
    def ml_plot():
        _update_plot_style_from_input(input)
        ml = ml_state.get()

        if ml is None:
            return _blank_plotly("Run ML first.")

        plot_type = input.ml_plot_type()

        if plot_type == "Confusion matrix":
            cm = ml["confusion_matrix"]

            if cm is None or cm.empty:
                return _blank_plotly("No holdout confusion matrix available. Check the metrics box for the reason.")

            fig = go.Figure(
                data=[
                    go.Heatmap(
                        z=cm.values,
                        x=list(cm.columns),
                        y=list(cm.index),
                        text=cm.values,
                        texttemplate="%{text}",
                        colorbar=dict(title="Count"),
                    )
                ]
            )
            fig.update_layout(
                title="Confusion matrix",
                xaxis_title="Predicted",
                yaxis_title="True",
                height=620,
            )
            return _apply_global_plot_style(fig)

        if plot_type == "Feature importance":
            imp = ml.get("feature_importance", pd.DataFrame()).copy()

            if imp is None or imp.empty or "importance" not in imp.columns:
                return _blank_plotly("No feature importance available for this model.")

            imp["importance"] = pd.to_numeric(imp["importance"], errors="coerce").fillna(0.0)
            imp = imp.sort_values("importance", ascending=False).head(30).copy()

            if float(imp["importance"].abs().sum()) == 0:
                return _blank_plotly(
                    "Feature importance is zero or unavailable for this model/settings. "
                    "Try LogisticRegression, LinearSVM, or RandomForest for direct feature importance."
                )

            label_col = "feature_ppm" if "feature_ppm" in imp.columns else "feature"

            fig = go.Figure(
                data=[
                    go.Bar(
                        x=imp["importance"],
                        y=imp[label_col].astype(str),
                        orientation="h",
                        text=imp[label_col].astype(str),
                        hovertemplate="Feature: %{y}<br>Importance: %{x:.6g}<extra></extra>",
                    )
                ]
            )
            fig.update_layout(
                title="Top feature importances",
                xaxis_title="Importance",
                yaxis_title="Feature",
                height=650,
            )
            fig.update_yaxes(autorange="reversed")
            return _apply_global_plot_style(fig)

        if plot_type == "ROC curve":
            roc = ml.get("roc_curve", pd.DataFrame())

            if roc is None or roc.empty:
                return _blank_plotly("ROC curve is available only for models with probability output and a valid holdout test.")

            fig = go.Figure()

            for cls in roc["class"].unique():
                sub = roc[roc["class"] == cls]
                auc_value = sub["auc"].iloc[0] if "auc" in sub.columns and len(sub) else np.nan
                fig.add_trace(
                    go.Scatter(
                        x=sub["fpr"],
                        y=sub["tpr"],
                        mode="lines",
                        name=f"{cls} AUC={auc_value:.3f}",
                    )
                )

            fig.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[0, 1],
                    mode="lines",
                    name="chance",
                    line=dict(dash="dash"),
                )
            )
            fig.update_layout(
                title="ROC curve",
                xaxis_title="False positive rate",
                yaxis_title="True positive rate",
                height=620,
            )
            return _apply_global_plot_style(fig)

        if plot_type == "Predicted probabilities":
            probs = ml.get("probabilities", pd.DataFrame())

            if probs is None or probs.empty:
                return _blank_plotly("Predicted probabilities are available only for Logistic Regression and Random Forest with a valid holdout test.")

            prob_cols = [c for c in probs.columns if c.startswith("prob_")]

            fig = go.Figure()
            x_vals = probs["sample_id"].astype(str)

            for col in prob_cols:
                fig.add_trace(
                    go.Bar(
                        x=x_vals,
                        y=probs[col],
                        name=col.replace("prob_", ""),
                        hovertemplate="Sample: %{x}<br>Probability: %{y:.3f}<extra></extra>",
                    )
                )

            fig.update_layout(
                title="Predicted class probabilities on holdout samples",
                xaxis_title="Sample ID",
                yaxis_title="Predicted probability",
                barmode="stack",
                height=620,
            )
            return _apply_global_plot_style(fig)

        if plot_type == "PSA baseline confusion matrix":
            psa = ml.get("psa_baseline", {})
            cm = psa.get("confusion_matrix") if isinstance(psa, dict) else None

            if cm is None or getattr(cm, "empty", True):
                note = psa.get("note", "No PSA baseline confusion matrix available.") if isinstance(psa, dict) else "No PSA baseline confusion matrix available."
                return _blank_plotly(note)

            fig = go.Figure(
                data=[
                    go.Heatmap(
                        z=cm.values,
                        x=list(cm.columns),
                        y=list(cm.index),
                        text=cm.values,
                        texttemplate="%{text}",
                        colorbar=dict(title="Count"),
                    )
                ]
            )
            fig.update_layout(
                title=f"PSA cutoff baseline confusion matrix (cutoff {psa.get('psa_cutoff', '')})",
                xaxis_title="Predicted",
                yaxis_title="True",
                height=620,
            )
            return _apply_global_plot_style(fig)

        return _blank_plotly("Unknown ML plot.")

    def _ml_plot_data_frame():
        ml = ml_state.get()
        if ml is None:
            return pd.DataFrame({"message": ["No ML results yet."]})

        plot_type = input.ml_plot_type()

        if plot_type == "Confusion matrix":
            cm = ml.get("confusion_matrix", pd.DataFrame())
            if cm is None or cm.empty:
                return pd.DataFrame({"message": ["No confusion matrix available."]})
            return cm.reset_index().melt(id_vars="index", var_name="predicted_class", value_name="count").rename(columns={"index": "true_class"})

        if plot_type == "Feature importance":
            return ml.get("feature_importance", pd.DataFrame()).copy()

        if plot_type == "ROC curve":
            roc = ml.get("roc_curve", pd.DataFrame())
            if roc is None or roc.empty:
                return pd.DataFrame({"message": ["No ROC curve data available."]})
            return roc.copy()

        if plot_type == "Predicted probabilities":
            probs = ml.get("probabilities", pd.DataFrame())
            if probs is None or probs.empty:
                return pd.DataFrame({"message": ["No probability data available."]})
            return probs.copy()

        if plot_type == "PSA baseline confusion matrix":
            psa = ml.get("psa_baseline", {})
            cm = psa.get("confusion_matrix") if isinstance(psa, dict) else None
            if cm is None or cm.empty:
                return pd.DataFrame({"message": ["No PSA baseline confusion matrix available."]})
            return cm.reset_index().melt(id_vars="index", var_name="predicted_class", value_name="count").rename(columns={"index": "true_class"})

        return pd.DataFrame({"message": [f"No data export available for ML plot: {plot_type}"]})

    @render.download(filename="current_ml_plot_data.csv")
    def download_ml_plot_data():
        yield _ml_plot_data_frame().to_csv(index=False)

    @render.data_frame
    def feature_importance_table():
        ml = ml_state.get()

        if ml is None:
            return render.DataGrid(pd.DataFrame({"message": ["Run ML first."]}))

        return render.DataGrid(ml["feature_importance"].head(50))

    @render.data_frame
    def test_predictions_table():
        ml = ml_state.get()

        if ml is None:
            return render.DataGrid(pd.DataFrame({"message": ["Run ML first."]}))

        preds = ml.get("test_predictions", pd.DataFrame())

        if preds is None or preds.empty:
            return render.DataGrid(pd.DataFrame({"message": ["No holdout test predictions available."]}))

        return render.DataGrid(preds)

    @render.data_frame
    def ml_history_table():
        history = ml_history_state.get()

        if history is None or history.empty:
            return render.DataGrid(pd.DataFrame({"message": ["No ML runs have been recorded yet."]}))

        return render.DataGrid(history)



    @render_plotly
    def outlier_plot():
        _update_plot_style_from_input(input)
        result = outlier_state.get()

        if result is None:
            return _blank_plotly("Click Detect outliers first")

        scores = result["scores"]

        if "PC1" not in scores.columns or "PC2" not in scores.columns:
            return _blank_plotly("PC1/PC2 scores are not available")

        fig = go.Figure()

        for cls in scores["class"].unique():
            sub = scores[scores["class"] == cls]
            fig.add_trace(
                go.Scatter(
                    x=sub["PC1"],
                    y=sub["PC2"],
                    mode="markers",
                    name=str(cls),
                    text=sub["sample_id"],
                    customdata=np.stack([sub["outlier_score"], sub["is_outlier"]], axis=-1),
                    hovertemplate=(
                        "Sample: %{text}<br>"
                        "PC1: %{x:.3f}<br>"
                        "PC2: %{y:.3f}<br>"
                        "Outlier score: %{customdata[0]:.3f}<br>"
                        "Flagged: %{customdata[1]}<extra></extra>"
                    ),
                )
            )

        out = scores[scores["is_outlier"]]
        if len(out) > 0:
            fig.add_trace(
                go.Scatter(
                    x=out["PC1"],
                    y=out["PC2"],
                    mode="markers+text",
                    name="flagged outlier",
                    text=out["sample_id"],
                    textposition="top center",
                    marker=dict(symbol="circle-open", size=14, line=dict(width=2)),
                    hovertemplate="Flagged sample: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
                )
            )

        group_text = "within each class" if result.get("groupwise", False) else "globally"
        fig.update_layout(
            title=f"PCA outlier screening — {result.get('method', '')}, calculated {group_text}",
            xaxis_title="PC1",
            yaxis_title="PC2",
            height=680,
        )
        return _apply_global_plot_style(fig)

    @render_plotly
    def outlier_cohort_plot():
        _update_plot_style_from_input(input)
        result = outlier_state.get()

        if result is None:
            return _blank_plotly("Click Detect outliers first")

        scores = result["scores"].copy()

        if "PC1" not in scores.columns or "PC2" not in scores.columns:
            return _blank_plotly("PC1/PC2 scores are not available")

        classes = list(pd.Series(scores["class"].astype(str)).drop_duplicates())
        n = len(classes)

        if n == 0:
            return _blank_plotly("No classes available for cohort plots.")

        ncols = 2 if n > 1 else 1
        nrows = int(np.ceil(n / ncols))

        fig = make_subplots(
            rows=nrows,
            cols=ncols,
            subplot_titles=[str(c) for c in classes],
            horizontal_spacing=0.08,
            vertical_spacing=0.12,
        )

        for idx_cls, cls in enumerate(classes):
            row = int(idx_cls // ncols) + 1
            col = int(idx_cls % ncols) + 1

            sub = scores[scores["class"].astype(str) == str(cls)]
            flagged = sub[sub["is_outlier"]]
            kept = sub[~sub["is_outlier"]]

            fig.add_trace(
                go.Scatter(
                    x=kept["PC1"],
                    y=kept["PC2"],
                    mode="markers",
                    name=f"{cls} kept",
                    showlegend=True,
                    text=kept["sample_id"],
                    marker=dict(size=8),
                    customdata=np.stack([kept["outlier_score"], kept["is_outlier"]], axis=-1) if len(kept) else None,
                    hovertemplate="Sample: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>Outlier score: %{customdata[0]:.3f}<extra></extra>",
                ),
                row=row,
                col=col,
            )

            if len(flagged) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=flagged["PC1"],
                        y=flagged["PC2"],
                        mode="markers+text",
                        name=f"{cls} flagged",
                        showlegend=True,
                        text=flagged["sample_id"],
                        textposition="top center",
                        marker=dict(symbol="circle-open", size=14, line=dict(width=2)),
                        customdata=np.stack([flagged["outlier_score"], flagged["is_outlier"]], axis=-1),
                        hovertemplate="Flagged sample: %{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>Outlier score: %{customdata[0]:.3f}<extra></extra>",
                    ),
                    row=row,
                    col=col,
                )

            fig.update_xaxes(title_text="PC1", row=row, col=col)
            fig.update_yaxes(title_text="PC2", row=row, col=col)

        fig.update_layout(
            title="Outlier screening by cohort/class",
            height=max(500, 420 * nrows),
        )
        return _apply_global_plot_style(fig)

    @render.data_frame
    def outlier_table():
        result = outlier_state.get()

        if result is None:
            return render.DataGrid(pd.DataFrame({"message": ["Click Detect outliers first."]}))

        return render.DataGrid(result["table"])

    def _plot_score_data(fig, ax, eda, plot_type, dimension, x_comp, y_comp, z_comp, suffix=""):
        if plot_type == "PCA scores":
            scores = eda["pca"]["scores"]
            variance = eda["pca"]["variance"]
            prefix = "PC"
            x_col = f"PC{int(x_comp)}"
            y_col = f"PC{int(y_comp)}"
            z_col = f"PC{int(z_comp)}"

            def label(col):
                row = variance[variance["component"] == col]
                if len(row) == 1:
                    return f"{col} ({row['explained_variance_percent'].iloc[0]:.1f}%)"
                return col

            title_base = "PCA scores"

        elif plot_type == "PLS-DA scores":
            scores = eda["plsda"]["scores"]
            variance = eda["plsda"]["variance"]
            prefix = "LV"
            x_col = f"LV{int(x_comp)}"
            y_col = f"LV{int(y_comp)}"
            z_col = f"LV{int(z_comp)}"

            def label(col):
                row = variance[variance["component"] == col]
                if len(row) == 1:
                    return f"{col} (approx X var {row['approx_x_variance_percent'].iloc[0]:.1f}%)"
                return col

            title_base = "PLS-DA scores"

        else:
            return False

        available = [c for c in scores.columns if c.startswith(prefix)]

        if x_col not in scores.columns or y_col not in scores.columns:
            ax.text(0.5, 0.5, f"Selected components are not available. Available: {', '.join(available)}", ha="center", va="center")
            ax.set_axis_off()
            return True

        if dimension == "3D":
            if z_col not in scores.columns:
                ax3d = fig.add_subplot(111, projection="3d")
                ax3d.text2D(0.5, 0.5, f"{z_col} is not available. Available: {', '.join(available)}", ha="center", va="center", transform=ax3d.transAxes)
                ax3d.set_axis_off()
                return True

            ax3d = fig.add_subplot(111, projection="3d")
            for cls in scores["class"].unique():
                sub = scores[scores["class"] == cls]
                ax3d.scatter(sub[x_col], sub[y_col], sub[z_col], label=str(cls), s=55, alpha=0.85)

            ax3d.set_xlabel(label(x_col))
            ax3d.set_ylabel(label(y_col))
            ax3d.set_zlabel(label(z_col))
            ax3d.set_title(f"3D {title_base}{suffix}")
            ax3d.legend()
            return True

        for cls in scores["class"].unique():
            sub = scores[scores["class"] == cls]
            ax.scatter(sub[x_col], sub[y_col], label=str(cls), s=55, alpha=0.85)

        ax.set_xlabel(label(x_col))
        ax.set_ylabel(label(y_col))
        ax.set_title(f"2D {title_base}{suffix}")
        ax.legend()
        return True

    @render_plotly
    def eda_filtered_plot():
        _update_plot_style_from_input(input)
        eda = eda_filtered_state.get()

        if eda is None:
            return _blank_plotly("Run EDA after outlier removal first")

        plot_type = input.eda2_plot_type()

        if plot_type in ["PCA scores", "PLS-DA scores"]:
            return _score_plotly(
                eda,
                plot_type,
                input.score2_dimension(),
                input.score2_x_component(),
                input.score2_y_component(),
                input.score2_z_component(),
                title_suffix=" after outlier step",
                color_by=input.score2_color_by(),
                marker_size=input.score2_marker_size(),
                palette=input.score2_palette(),
                psa_col=input.eda2_psa_column(),
                psa_cutoff=float(input.eda2_psa_cutoff()),
            )

        if plot_type == "PCA loadings":
            return _pca_loading_plotly(eda, component_number=input.score2_x_component(), n_components=input.loading2_n_components(), title_suffix=" after outlier step")

        if plot_type == "Class counts":
            counts = eda["class_counts"]
            fig = go.Figure(data=[go.Bar(x=counts["class"].astype(str), y=counts["count"])])
            fig.update_layout(title="Class counts after outlier step", xaxis_title="Class", yaxis_title="Count", height=600)
            return _apply_global_plot_style(fig)

        if plot_type == "Top univariate bins":
            uni = eda["univariate"].copy()
            if "p_value" not in uni.columns:
                return _blank_plotly("No univariate results available")

            uni = uni.head(20).copy()
            y_values = -np.log10(uni["p_value"].replace(0, np.nan))
            y_values = y_values.fillna(y_values.max() if y_values.notna().any() else 0)

            fig = go.Figure(data=[go.Bar(x=y_values, y=uni["feature_ppm"].astype(str), orientation="h")])
            fig.update_layout(
                title="Top bins after outlier step",
                xaxis_title="-log10(p-value)",
                yaxis_title="ppm bin",
                height=650,
            )
            fig.update_yaxes(autorange="reversed")
            return _apply_global_plot_style(fig)

        if plot_type == "Clinical correlation heatmap":
            corr = eda["clinical_corr"]

            if corr is None or corr.empty:
                return _blank_plotly("Not enough numeric clinical variables")

            fig = go.Figure(
                data=[
                    go.Heatmap(
                        z=corr.values,
                        x=list(corr.columns),
                        y=list(corr.index),
                        zmin=-1,
                        zmax=1,
                        colorbar=dict(title="Spearman rho"),
                    )
                ]
            )
            fig.update_layout(title="Clinical correlation heatmap after outlier step", height=650)
            return _apply_global_plot_style(fig)

        if plot_type == "PSA by Class":
            return _plot_psa_by_class(
                eda,
                psa_col=input.eda2_psa_column(),
                psa_cutoff=float(input.eda2_psa_cutoff()),
                title_suffix=" after outlier step",
            )

        if plot_type == "Class by PSA group":
            return _plot_class_by_psa_group(
                eda,
                psa_col=input.eda2_psa_column(),
                psa_cutoff=float(input.eda2_psa_cutoff()),
                title_suffix=" after outlier step",
            )

        if plot_type == "PSA-feature correlations":
            return _plot_psa_feature_correlations(
                eda,
                psa_col=input.eda2_psa_column(),
                title_suffix=" after outlier step",
            )

        return _blank_plotly("Unknown plot")

    @render.data_frame
    def class_counts_filtered_table():
        eda = eda_filtered_state.get()

        if eda is None:
            return render.DataGrid(pd.DataFrame({"message": ["Run EDA after outlier removal first."]}))

        return render.DataGrid(eda["class_counts"])

    @render.data_frame
    def univariate_filtered_table():
        eda = eda_filtered_state.get()

        if eda is None:
            return render.DataGrid(pd.DataFrame({"message": ["Run EDA after outlier removal first."]}))

        return render.DataGrid(eda["univariate"])


    def _eda_plot_data_frame(eda, plot_type: str):
        if eda is None:
            return pd.DataFrame({"message": ["No EDA results available."]})

        if plot_type == "PCA scores":
            return eda.get("pca", {}).get("scores", pd.DataFrame()).copy()

        if plot_type == "PCA loadings":
            return eda.get("pca", {}).get("loadings", pd.DataFrame()).copy()

        if plot_type == "PLS-DA scores":
            return eda.get("plsda", {}).get("scores", pd.DataFrame()).copy()

        if plot_type == "Class counts":
            return eda.get("class_counts", pd.DataFrame()).copy()

        if plot_type == "Top univariate bins":
            return eda.get("univariate", pd.DataFrame()).copy()

        if plot_type == "Clinical correlation heatmap":
            corr = eda.get("clinical_corr", pd.DataFrame())
            if corr is None or corr.empty:
                return pd.DataFrame({"message": ["No clinical correlation matrix available."]})
            return corr.reset_index().melt(id_vars="index", var_name="variable_2", value_name="spearman_rho").rename(columns={"index": "variable_1"})

        if plot_type == "PSA-feature correlations":
            table = eda.get("feature_clinical_corr", pd.DataFrame()).copy()
            if table.empty:
                return pd.DataFrame({"message": ["No PSA-feature correlation table available."]})
            return table

        # For PSA by Class / Class by PSA group, the data are embedded in the aligned clinical metadata.
        aligned = combined_state.get()
        if aligned is None:
            aligned = filtered_state.get()
        if aligned is not None and aligned.get("clinical_aligned") is not None:
            return aligned["clinical_aligned"].copy().reset_index()

        return pd.DataFrame({"message": [f"No plot data available for: {plot_type}"]})

    @render.download(filename="current_eda_plot_data.csv")
    def download_eda_plot_data():
        yield _eda_plot_data_frame(eda_state.get(), input.eda_plot_type()).to_csv(index=False)

    @render.download(filename="current_post_outlier_eda_plot_data.csv")
    def download_eda_filtered_plot_data():
        yield _eda_plot_data_frame(eda_filtered_state.get(), input.eda2_plot_type()).to_csv(index=False)

    @render.download(filename="outlier_table.csv")
    def download_outlier_table():
        result = outlier_state.get()
        if result is None:
            yield "No outlier results yet.\n"
            return
        yield result.get("table", pd.DataFrame()).to_csv(index=False)

    @render.download(filename="binned_nmr.csv")
    def download_binned():
        df = binned_state.get()

        if df is None:
            yield "No binned table yet.\n"
            return

        yield df.to_csv(index=True)

    @render.download(filename="normalized_nmr.csv")
    def download_normalized():
        df = normalized_state.get()

        if df is None:
            yield "No normalized table yet.\n"
            return

        yield df.to_csv(index=True)

    @render.download(filename="processing_log.txt")
    def download_log():
        samples = samples_state.get()

        if not samples:
            yield "No samples processed yet.\n"
            return

        lines = []

        for i, sample in enumerate(samples):
            lines.append(f"Sample {i}: {sample['name']}")
            for item in sample.get("log", []):
                lines.append(f"  - {item}")
            lines.append("")

        yield "\n".join(lines)


    @render.download(filename="aligned_sample_table.csv")
    def download_combined():
        aligned = combined_state.get()

        if aligned is None:
            yield "No aligned sample table yet.\n"
            return

        yield aligned["sample_table"].to_csv(index=False)

    @render.download(filename="eda_univariate_bins.csv")
    def download_eda_univariate():
        eda = eda_state.get()

        if eda is None:
            yield "No EDA results yet.\n"
            return

        yield eda["univariate"].to_csv(index=False)

    @render.download(filename="ml_test_predictions.csv")
    def download_ml_predictions():
        ml = ml_state.get()

        if ml is None:
            yield "No ML results yet.\n"
            return

        preds = ml.get("test_predictions", pd.DataFrame())

        if preds is None or preds.empty:
            yield "No holdout test predictions available.\n"
            return

        yield preds.to_csv(index=False)

    @render.download(filename="ml_model_performance_history.csv")
    def download_ml_history():
        history = ml_history_state.get()

        if history is None or history.empty:
            yield "No ML runs have been recorded yet.\n"
            return

        yield history.to_csv(index=False)

    @render.download(filename="ml_feature_importance.csv")
    def download_ml_importance():
        ml = ml_state.get()

        if ml is None:
            yield "No ML results yet.\n"
            return

        yield ml["feature_importance"].to_csv(index=False)


    @render.download(filename="eda_univariate_bins_after_outlier_removal.csv")
    def download_eda_filtered_univariate():
        eda = eda_filtered_state.get()

        if eda is None:
            yield "No filtered EDA results yet.\n"
            return

        yield eda["univariate"].to_csv(index=False)


app = App(app_ui, server)
