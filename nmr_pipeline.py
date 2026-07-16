from __future__ import annotations

from pathlib import Path
import tempfile
import zipfile

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.linalg import pascal
from scipy.optimize import minimize_scalar
from scipy.sparse import eye, csr_matrix, csc_matrix
from scipy.sparse.linalg import factorized, splu, spsolve


# ============================================================
# Bruker reading
# ============================================================

def _parse_bruker_value(text: str):
    text = text.strip()

    if text.startswith("<") and text.endswith(">"):
        return text[1:-1]

    try:
        return int(text)
    except ValueError:
        pass

    try:
        return float(text)
    except ValueError:
        return text


def read_bruker_params(path: Path | str | None) -> dict:
    if path is None:
        return {}

    path = Path(path)

    if not path.exists():
        return {}

    params = {}

    with open(path, "r", errors="ignore") as f:
        for line in f:
            line = line.strip()

            if line.startswith("##$") and "=" in line:
                key, value = line[3:].split("=", 1)
                params[key.strip()] = _parse_bruker_value(value)

    return params


def find_bruker_experiments(root_folder: Path | str) -> list[Path]:
    root_folder = Path(root_folder)
    experiments = []

    for folder in root_folder.rglob("*"):
        if folder.is_dir() and (folder / "fid").exists() and (folder / "acqus").exists():
            experiments.append(folder)

    return sorted(experiments, key=lambda p: str(p))


def read_fid(fid_path: Path | str, acqus: dict) -> np.ndarray:
    fid_path = Path(fid_path)

    bytor = int(acqus.get("BYTORDA", 0))
    dtypa = int(acqus.get("DTYPA", 0))

    endian = "<" if bytor == 0 else ">"

    if dtypa == 0:
        dtype = np.dtype(endian + "i4")
    elif dtypa == 2:
        dtype = np.dtype(endian + "f8")
    else:
        dtype = np.dtype(endian + "i4")

    raw = np.fromfile(fid_path, dtype=dtype).astype(float)

    if len(raw) < 2:
        raise ValueError(f"FID file is too small or unreadable: {fid_path}")

    if len(raw) % 2 != 0:
        raw = raw[:-1]

    return raw[0::2] + 1j * raw[1::2]


def infer_sample_id(experiment_folder: Path) -> str:
    """
    Infer the biological/sample ID from a Bruker experiment folder.

    Bruker data are often structured like:
        study_id / 1 / fid
        study_id / 10 / fid

    In that case, the folder containing fid is only the Bruker experiment number,
    and the parent folder is the real sample/study ID. If the folder containing
    fid is not just an experiment number, use it directly.
    """
    experiment_folder = Path(experiment_folder)
    folder_name = experiment_folder.name
    parent_name = experiment_folder.parent.name

    # Bruker experiment folders are commonly numeric: 1, 2, 10, 11, ...
    if folder_name.isdigit() and parent_name not in ["", ".", "unzipped", "__MACOSX"]:
        return parent_name

    return folder_name


def dwell_time(acqus: dict) -> float:
    sw_h = float(acqus.get("SW_h", acqus.get("$SW_h", 1.0)))
    if sw_h <= 0:
        sw_h = 1.0
    return 1.0 / sw_h


def time_axis(acqus: dict, n_points: int) -> np.ndarray:
    return np.arange(n_points) * dwell_time(acqus)


def load_zip_and_read_fids(zip_path: Path | str) -> list[dict]:
    zip_path = Path(zip_path)

    temp_dir = Path(tempfile.mkdtemp(prefix="nmr_shiny_"))
    unzip_dir = temp_dir / "unzipped"
    unzip_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(unzip_dir)

    experiments = find_bruker_experiments(unzip_dir)

    if len(experiments) == 0:
        raise ValueError("No Bruker experiments found. I expected folders containing both 'fid' and 'acqus'.")

    samples = []

    for experiment in experiments:
        acqus = read_bruker_params(experiment / "acqus")
        raw_fid = read_fid(experiment / "fid", acqus)
        sample_id = infer_sample_id(experiment)

        samples.append(
            {
                "name": sample_id,
                "sample_id": sample_id,
                "experiment_folder_name": experiment.name,
                "folder": str(experiment),
                "acqus": acqus,
                "raw_fid": raw_fid,
                "log": [f"Loaded raw FID with {len(raw_fid)} complex point(s)."],
            }
        )

    return samples


# ============================================================
# FID-domain steps
# ============================================================

def apply_group_delay(samples: list[dict], override_points: float = -1) -> list[dict]:
    out = []

    for sample in samples:
        s = dict(sample)
        fid = s["raw_fid"]
        acqus = s["acqus"]

        if override_points is not None and float(override_points) >= 0:
            points = int(round(float(override_points)))
        else:
            points = int(round(float(acqus.get("GRPDLY", acqus.get("$GRPDLY", 0)))))

        points = max(0, points)

        if points >= len(fid):
            points = 0

        s["group_delay_points"] = points
        s["group_delay_fid"] = fid[points:].copy()
        s["log"] = s.get("log", []) + [f"Group delay removal: removed {points} point(s)."]
        out.append(s)

    return out


def _diff_matrix(m: int, d: int = 2):
    nums = pascal(d + 1, kind="lower")[-1].astype(float)
    minuses_from = d % 2 + 1 if d != 1 else d % 2
    nums[minuses_from::2] *= -1
    data = np.tile(nums, m - d)
    row_ind = (np.arange(d + 1) + np.arange(m - d).reshape(-1, 1)).flatten()
    col_ind = np.repeat(np.arange(m - d), d + 1)
    return csr_matrix((data, (row_ind, col_ind)), shape=(m, m - d)).T


def _solvent_smooth_1d(y: np.ndarray, lam: float = 1e6, d: int = 2) -> np.ndarray:
    m = len(y)

    if m < 10:
        return np.zeros_like(y)

    E = eye(m)
    D = _diff_matrix(m, d=d)
    A = E + float(lam) * D.T.dot(D)
    C = splu(csc_matrix(A))
    return C.solve(C.solve(y))


def apply_solvent_residuals_removal(samples: list[dict], lam: float = 1e6, enabled: bool = True) -> list[dict]:
    """
    Protomix/PepsNMR-style solvent residual suppression in the FID domain.

    A smooth residual signal is estimated separately for real and imaginary FID
    parts using penalized second differences, then subtracted.
    """
    out = []

    for sample in samples:
        s = dict(sample)

        if "group_delay_fid" not in s:
            raise ValueError("Group delay must be applied before solvent residual suppression.")

        fid = s["group_delay_fid"]

        if not enabled:
            s["solvent_removed_fid"] = fid.copy()
            s["estimated_solvent"] = np.zeros_like(fid)
            s["log"] = s.get("log", []) + ["Solvent residual suppression skipped."]
            out.append(s)
            continue

        solvent_re = _solvent_smooth_1d(fid.real, lam=lam, d=2)
        solvent_im = _solvent_smooth_1d(fid.imag, lam=lam, d=2)
        solvent = solvent_re + 1j * solvent_im

        corrected = fid - solvent

        s["solvent_removed_fid"] = corrected
        s["estimated_solvent"] = solvent
        s["log"] = s.get("log", []) + [f"Solvent residual suppression: lambda={lam:g}."]
        out.append(s)

    return out


def apply_apodization(samples: list[dict], lb: float = 1.0, kind: str = "exponential") -> list[dict]:
    out = []

    for sample in samples:
        s = dict(sample)

        if "solvent_removed_fid" not in s:
            raise ValueError("Solvent residual suppression must be applied before apodization.")

        fid = s["solvent_removed_fid"]
        t = time_axis(s["acqus"], len(fid))

        if kind == "gaussian":
            window = np.exp(-0.5 * (float(lb) * t) ** 2)
        elif kind == "exponential":
            window = np.exp(-float(lb) * t)
        else:
            raise ValueError("Apodization kind must be 'exponential' or 'gaussian'.")

        s["apodized_fid"] = fid * window
        s["apodization_lb"] = float(lb)
        s["apodization_kind"] = kind
        s["log"] = s.get("log", []) + [f"Apodization: {kind}, LB={lb:g}."]
        out.append(s)

    return out


def apply_zero_filling(samples: list[dict], extra_points: int = 32768) -> list[dict]:
    """
    Protomix-style zero filling: add `extra_points` zeros to the end of the FID.
    """
    out = []
    extra_points = int(extra_points)

    for sample in samples:
        s = dict(sample)

        if "apodized_fid" not in s:
            raise ValueError("Apodization must be applied before zero filling.")

        fid = s["apodized_fid"]

        if extra_points <= 0:
            zf = fid.copy()
        else:
            zf = np.concatenate([fid, np.zeros(extra_points, dtype=complex)])

        s["zero_filled_fid"] = zf
        s["zero_fill_extra_points"] = extra_points
        s["log"] = s.get("log", []) + [f"Zero filling: added {extra_points} zero point(s); final length={len(zf)}."]
        out.append(s)

    return out


# ============================================================
# Spectrum-domain steps
# ============================================================

def make_ppm_axis(acqus: dict, n_points: int) -> np.ndarray:
    """
    Protomix-style ppm conversion:
        dwell_time = 1 / SW_h
        freq = fftfreq(n, dwell_time)
        freq = fftshift(freq + O1)
        ppm = freq / SFO1
    """
    sw_h = float(acqus.get("SW_h", acqus.get("$SW_h", 1.0)))
    o1 = float(acqus.get("O1", acqus.get("$O1", 0.0)))
    sfo1 = float(acqus.get("SFO1", acqus.get("$SFO1", 1.0)))

    if sw_h <= 0:
        sw_h = 1.0
    if sfo1 == 0:
        sfo1 = 1.0

    dt = 1.0 / sw_h
    freq = np.fft.fftfreq(n_points, d=dt)
    freq = np.fft.fftshift(freq + o1)
    ppm = freq / sfo1

    return ppm


def apply_fourier_transform(samples: list[dict]) -> list[dict]:
    out = []

    for sample in samples:
        s = dict(sample)

        if "zero_filled_fid" not in s:
            raise ValueError("Zero filling must be applied before Fourier transform.")

        fid = s["zero_filled_fid"]
        spectrum = np.fft.fftshift(np.fft.fft(fid))
        ppm = make_ppm_axis(s["acqus"], len(spectrum))

        s["complex_spectrum"] = spectrum
        s["ppm"] = ppm
        s["spectrum_real"] = np.real(spectrum)
        s["log"] = s.get("log", []) + ["Fourier transform applied."]
        out.append(s)

    return out


def _phase_objective(angle_rad: float, y: np.ndarray) -> float:
    rotated = y * np.exp(1j * angle_rad)
    real_y = np.real(rotated)
    positive = real_y[real_y >= 0]
    pos_ss = np.sum(positive ** 2)
    total_ss = np.sum(real_y ** 2)

    if total_ss == 0:
        return 0.0

    return -pos_ss / total_ss


def _auto_phase_protomix_like(complex_spectrum: np.ndarray, ppm: np.ndarray, exclude_region=None) -> tuple[np.ndarray, float]:
    """
    Universal automatic zero-order phase correction.

    By default, no ppm region is excluded. This makes the app usable for samples
    that do not have a water peak or have a solvent peak in a different place.

    If exclude_region is supplied by advanced/custom code, that region is ignored
    during phase-angle estimation.
    """
    if exclude_region is None:
        mask = np.ones_like(ppm, dtype=bool)
    else:
        low, high = min(exclude_region), max(exclude_region)
        mask = (ppm <= low) | (ppm >= high)

        if np.sum(mask) < 10:
            mask = np.ones_like(ppm, dtype=bool)

    y_for_optim = complex_spectrum[mask]

    f0 = _phase_objective(0, y_for_optim)
    fpi = _phase_objective(np.pi, y_for_optim)
    bounds = (-np.pi, np.pi) if f0 < fpi else (0, 2 * np.pi)

    res = minimize_scalar(
        _phase_objective,
        args=(y_for_optim,),
        bounds=bounds,
        method="bounded",
    )

    angle = float(res.x)
    phased_complex = complex_spectrum * np.exp(1j * angle)
    phased_real = np.real(phased_complex)

    return phased_real, np.rad2deg(angle)


def apply_phase_correction(
    samples: list[dict],
    auto: bool = True,
    manual_angle_deg: float = 0.0,
    exclude_region_text: str = "",
) -> list[dict]:
    out = []

    if exclude_region_text is None or str(exclude_region_text).strip() == "":
        exclude_region = None
    else:
        exclude_region = parse_region_text(exclude_region_text)

    for sample in samples:
        s = dict(sample)

        if "complex_spectrum" not in s:
            raise ValueError("Fourier transform must be applied before phase correction.")

        spectrum = s["complex_spectrum"]
        ppm = s["ppm"]

        if auto:
            phased, angle_deg = _auto_phase_protomix_like(spectrum, ppm, exclude_region=exclude_region)
        else:
            angle_deg = float(manual_angle_deg)
            phased = np.real(spectrum * np.exp(1j * np.deg2rad(angle_deg)))

        s["phased"] = phased
        s["phase_angle_deg"] = float(angle_deg)
        if auto:
            s["log"] = s.get("log", []) + ["Phase correction: automatic."]
        else:
            s["log"] = s.get("log", []) + ["Phase correction: manual."]
        out.append(s)

    return out


def apply_referencing(
    samples: list[dict],
    use_reference: bool = True,
    target_ppm: float = 0.0,
    search_min: float = -0.2,
    search_max: float = 0.2,
) -> list[dict]:
    out = []

    for sample in samples:
        s = dict(sample)

        if "phased" not in s:
            raise ValueError("Phase correction must be applied before referencing.")

        ppm = s["ppm"]
        intensity = s["phased"]

        if not use_reference:
            s["referenced_ppm"] = ppm.copy()
            s["found_reference"] = None
            s["log"] = s.get("log", []) + ["Referencing skipped."]
            out.append(s)
            continue

        low = min(float(search_min), float(search_max))
        high = max(float(search_min), float(search_max))
        mask = (ppm >= low) & (ppm <= high)

        if np.sum(mask) < 3:
            s["referenced_ppm"] = ppm.copy()
            s["found_reference"] = None
            s["log"] = s.get("log", []) + ["Reference peak not found; ppm axis unchanged."]
            out.append(s)
            continue

        local_ppm = ppm[mask]
        local_intensity = intensity[mask]
        found_ppm = float(local_ppm[np.argmax(local_intensity)])
        shift = found_ppm - float(target_ppm)

        s["referenced_ppm"] = ppm - shift
        s["found_reference"] = found_ppm
        s["log"] = s.get("log", []) + [f"Referencing: found {found_ppm:.4f} ppm; shifted to {target_ppm:.4f} ppm."]
        out.append(s)

    return out


# ============================================================
# Baseline correction
# ============================================================

def parse_region_text(region_text: str) -> tuple[float, float]:
    text = region_text.strip().replace(" ", "")

    if text == "":
        raise ValueError("Region text is empty. Example: 4.5-5.1")

    if "-" in text:
        left, right = text.split("-", 1)
    elif "," in text:
        left, right = text.split(",", 1)
    else:
        raise ValueError("Use a region like 4.5-5.1")

    left = float(left)
    right = float(right)

    return min(left, right), max(left, right)


def _interpolate_region(ppm: np.ndarray, y: np.ndarray, region_text: str) -> np.ndarray:
    out = y.copy()

    if region_text.strip() == "":
        return out

    low, high = parse_region_text(region_text)
    mask = (ppm >= low) & (ppm <= high)

    if not np.any(mask):
        return out

    idx = np.arange(len(out))
    region_idx = np.where(mask)[0]
    left_idx = region_idx[0] - 1
    right_idx = region_idx[-1] + 1

    if left_idx < 0 or right_idx >= len(out):
        return out

    out[mask] = np.interp(
        idx[mask],
        [left_idx, right_idx],
        [out[left_idx], out[right_idx]],
    )

    return out


def _als_baseline(y: np.ndarray, smoothness: float = 1e7, asymmetry: float = 0.01, max_iter: int = 42) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    m = len(y)

    if m < 10:
        return np.zeros_like(y)

    D = sparse.eye(m, format="csc")
    D = D[1:] - D[:-1]
    D = D[1:] - D[:-1]
    D = D.T

    w = np.ones(m)

    for _ in range(int(max_iter)):
        W = sparse.diags(w, 0, shape=(m, m))
        Z = W + float(smoothness) * D @ D.T
        z = spsolve(Z, w * y)
        w = float(asymmetry) * (y > z) + (1 - float(asymmetry)) * (y < z)

    return z


def _arpls_baseline(y: np.ndarray, smoothness: float = 1e5, tolerance: float = 0.05, max_iter: int = 100) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    n = len(y)

    if n < 10:
        return np.zeros_like(y)

    D = sparse.eye(n, format="csc")
    D = D[1:] - D[:-1]
    D = D[1:] - D[:-1]
    H = float(smoothness) * (D.T @ D)

    w = np.ones(n)
    z = np.zeros(n)

    for _ in range(int(max_iter)):
        W = sparse.diags(w, 0, shape=(n, n))
        WH = W + H
        solver = factorized(WH)
        z = solver(w * y)
        d = y - z
        dn = d[d < 0]

        if len(dn) == 0:
            break

        m = np.mean(dn)
        s = np.std(dn)

        if s == 0 or np.isnan(s):
            break

        wt = 1.0 / (1 + np.exp(2 * (d - (2 * s - m)) / s))

        if np.linalg.norm(w - wt) / np.linalg.norm(w) < tolerance:
            break

        w = wt

    return z


def _whittaker_smooth(x, w, smoothness, diff_order=1):
    x = np.asarray(x, dtype=float)
    m = len(x)

    D = sparse.eye(m, format="csc")
    for _ in range(int(diff_order)):
        D = D[1:] - D[:-1]

    W = sparse.diags(w, 0, shape=(m, m))
    A = W + float(smoothness) * D.T @ D
    B = W @ x
    return spsolve(A, B)


def _airpls_baseline(x, smoothness=1e5, diff_order=1, max_iter=100):
    x = np.asarray(x, dtype=float)
    m = x.shape[0]
    w = np.ones(m)
    z = np.zeros(m)

    for i in range(1, int(max_iter) + 1):
        z = _whittaker_smooth(x, w, smoothness, diff_order)
        d = x - z
        dssn = np.abs(d[d < 0].sum())

        if dssn < 0.001 * np.abs(x).sum() or dssn == 0:
            break

        w[d >= 0] = 0
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = w[-1] = np.exp(i * np.max(d[d < 0]) / dssn)

    return z


def estimate_baseline(
    ppm: np.ndarray,
    y: np.ndarray,
    method: str = "als",
    smoothness: float = 1e6,
    asymmetry: float = 0.01,
    max_iter: int = 12,
    exclude_region_text: str = "4.5-5.1",
    max_points: int = 3000,
) -> np.ndarray:
    """
    Estimate baseline.

    Important speed fix:
    Baseline correction on 50,000–100,000 points can be very slow because ALS/arPLS/airPLS
    solve sparse linear systems repeatedly. For an interactive Shiny app, we estimate the
    smooth baseline on a downsampled spectrum and interpolate it back to full length.

    max_points:
        Number of points used for baseline fitting.
        2000-5000 is usually enough for a smooth baseline.
        Use 0 to disable downsampling, but this may be slow.
    """
    y_for_baseline = _interpolate_region(ppm, y, exclude_region_text)
    y_for_baseline = np.asarray(y_for_baseline, dtype=float)
    n = len(y_for_baseline)

    def fit_baseline(y_fit: np.ndarray) -> np.ndarray:
        if method == "als":
            return _als_baseline(
                y_fit,
                smoothness=smoothness,
                asymmetry=asymmetry,
                max_iter=max_iter,
            )

        if method == "arpls":
            return _arpls_baseline(
                y_fit,
                smoothness=smoothness,
                tolerance=0.05,
                max_iter=max_iter,
            )

        if method == "airpls":
            return _airpls_baseline(
                y_fit,
                smoothness=smoothness,
                diff_order=1,
                max_iter=max_iter,
            )

        raise ValueError("Baseline method must be 'als', 'arpls', or 'airpls'.")

    if max_points is not None and int(max_points) > 0 and n > int(max_points):
        max_points = int(max_points)
        idx = np.linspace(0, n - 1, max_points).astype(int)
        idx = np.unique(idx)

        y_small = y_for_baseline[idx]
        baseline_small = fit_baseline(y_small)

        baseline_full = np.interp(
            np.arange(n),
            idx,
            baseline_small,
        )
        return baseline_full

    return fit_baseline(y_for_baseline)


def apply_baseline_correction(
    samples: list[dict],
    method: str = "als",
    smoothness: float = 1e6,
    asymmetry: float = 0.01,
    max_iter: int = 12,
    exclude_region_text: str = "4.5-5.1",
    max_points: int = 3000,
) -> list[dict]:
    out = []

    for sample in samples:
        s = dict(sample)

        if "referenced_ppm" not in s:
            raise ValueError("Referencing must be applied before baseline correction.")

        ppm = s["referenced_ppm"]
        y = s["phased"]

        baseline = estimate_baseline(
            ppm,
            y,
            method=method,
            smoothness=smoothness,
            asymmetry=asymmetry,
            max_iter=max_iter,
            exclude_region_text=exclude_region_text,
            max_points=max_points,
        )

        corrected = y - baseline

        s["baseline"] = baseline
        s["baseline_corrected"] = corrected
        s["log"] = s.get("log", []) + [
            f"Baseline correction: method={method}, smoothness={smoothness:g}, asymmetry={asymmetry:g}, iterations={max_iter}, excluded={exclude_region_text}, max_points={max_points}."
        ]
        out.append(s)

    return out


def _integer_shift_fill(y: np.ndarray, shift: int) -> np.ndarray:
    """Shift a spectrum by an integer number of points and fill edges with zero."""
    y = np.asarray(y)
    out = np.zeros_like(y)

    if shift == 0:
        return y.copy()

    if shift > 0:
        out[shift:] = y[:-shift]
    else:
        out[:shift] = y[-shift:]

    return out


def apply_peak_alignment(
    samples: list[dict],
    enabled: bool = False,
    reference_index: int = 0,
    align_min: float = 0.5,
    align_max: float = 9.5,
    max_shift_points: int = 80,
) -> list[dict]:
    """
    Simple FFT/cross-correlation-style alignment placeholder.

    This is NOT full Icoshift. It is a practical first alignment step:
    each spectrum is integer-shifted to maximize correlation with the
    chosen reference spectrum inside a selected ppm interval.
    """
    if not samples:
        return []

    for sample in samples:
        if "baseline_corrected" not in sample:
            raise ValueError("Baseline correction must be applied before peak alignment.")

    if not enabled:
        out = []
        for sample in samples:
            s = dict(sample)
            s["aligned"] = s["baseline_corrected"].copy()
            s["alignment_shift_points"] = 0
            s["log"] = s.get("log", []) + ["Peak alignment skipped."]
            out.append(s)
        return out

    reference_index = int(reference_index)
    reference_index = max(0, min(reference_index, len(samples) - 1))

    ref_sample = samples[reference_index]
    ppm = ref_sample["referenced_ppm"]
    low = min(float(align_min), float(align_max))
    high = max(float(align_min), float(align_max))
    mask = (ppm >= low) & (ppm <= high)

    if np.sum(mask) < 10:
        raise ValueError("Alignment window is too small.")

    ref = np.asarray(ref_sample["baseline_corrected"], dtype=float)
    ref_window = ref[mask]
    ref_window = ref_window - np.mean(ref_window)

    out = []

    for sample in samples:
        s = dict(sample)
        y = np.asarray(s["baseline_corrected"], dtype=float)
        y_window = y[mask]
        y_window = y_window - np.mean(y_window)

        best_shift = 0
        best_score = -np.inf

        for shift in range(-int(max_shift_points), int(max_shift_points) + 1):
            shifted = _integer_shift_fill(y_window, shift)
            score = float(np.dot(ref_window, shifted))

            if score > best_score:
                best_score = score
                best_shift = shift

        aligned = _integer_shift_fill(y, best_shift)

        s["aligned"] = aligned
        s["alignment_shift_points"] = best_shift
        s["log"] = s.get("log", []) + [
            f"Peak alignment: enabled, reference={reference_index}, window={low}-{high} ppm, shift={best_shift} point(s)."
        ]
        out.append(s)

    return out


def apply_negative_values_zeroing(samples: list[dict], enabled: bool = True) -> list[dict]:
    out = []

    for sample in samples:
        s = dict(sample)

        if "baseline_corrected" not in s:
            raise ValueError("Baseline correction must be applied before negative-value zeroing.")

        source_key = "aligned" if "aligned" in s else "baseline_corrected"
        y = s[source_key].copy()

        if enabled:
            y[y < 0] = 0.0
            s["negative_zeroed"] = y
            s["log"] = s.get("log", []) + ["Negative-value zeroing applied."]
        else:
            s["negative_zeroed"] = y
            s["log"] = s.get("log", []) + ["Negative-value zeroing skipped."]

        out.append(s)

    return out


def apply_window_selection(samples: list[dict], ppm_min: float = 0.2, ppm_max: float = 10.0) -> list[dict]:
    out = []
    low = min(float(ppm_min), float(ppm_max))
    high = max(float(ppm_min), float(ppm_max))

    for sample in samples:
        s = dict(sample)

        if "negative_zeroed" not in s:
            raise ValueError("Negative-value zeroing must be applied before window selection.")

        ppm = s["referenced_ppm"]
        y = s["negative_zeroed"]

        mask = (ppm >= low) & (ppm <= high)

        if np.sum(mask) < 3:
            raise ValueError(f"Selected window {low}-{high} ppm is empty or too small.")

        s["window_ppm"] = ppm[mask]
        s["window_intensity"] = y[mask]
        s["window_range"] = (low, high)
        s["log"] = s.get("log", []) + [f"Window selection: kept {low}-{high} ppm."]
        out.append(s)

    return out


def _remove_region(ppm: np.ndarray, y: np.ndarray, region_text: str, mode: str = "zero") -> np.ndarray:
    out = y.copy()

    if region_text.strip() == "":
        return out

    low, high = parse_region_text(region_text)
    mask = (ppm >= low) & (ppm <= high)

    if not np.any(mask):
        return out

    if mode == "zero":
        out[mask] = 0.0
        return out

    if mode == "interpolate":
        return _interpolate_region(ppm, y, region_text)

    raise ValueError("Region removal mode must be 'zero' or 'interpolate'.")


def apply_region_removal(samples: list[dict], region_text: str = "4.5-5.1", mode: str = "zero") -> list[dict]:
    out = []

    for sample in samples:
        s = dict(sample)

        if "window_ppm" not in s:
            raise ValueError("Window selection must be applied before region removal.")

        ppm = s["window_ppm"]
        y = s["window_intensity"]

        rr = _remove_region(ppm, y, region_text, mode=mode)

        s["region_removed"] = rr
        s["region_text"] = region_text
        s["region_mode"] = mode
        s["log"] = s.get("log", []) + [f"Region removal: region={region_text}, mode={mode}."]
        out.append(s)

    return out


def _integrate_bins(ppm: np.ndarray, intensity: np.ndarray, edges: np.ndarray, method: str = "trapezoidal") -> np.ndarray:
    order = np.argsort(ppm)
    ppm = ppm[order]
    intensity = intensity[order]

    values = []

    for left, right in zip(edges[:-1], edges[1:]):
        mask = (ppm >= left) & (ppm < right)

        if np.sum(mask) < 2:
            values.append(0.0)
        elif method == "rectangular":
            width = right - left
            values.append(float(np.mean(intensity[mask]) * width))
        else:
            values.append(float(np.trapezoid(intensity[mask], ppm[mask])))

    return np.array(values)


def apply_binning(
    samples: list[dict],
    bin_width: float = 0.01,
    method: str = "trapezoidal",
    n_bins: int | None = None,
) -> tuple[list[dict], pd.DataFrame]:
    if not samples:
        raise ValueError("No samples available.")

    for sample in samples:
        if "region_removed" not in sample:
            raise ValueError("Region removal must be applied before binning.")

    all_ppm = np.concatenate([s["window_ppm"] for s in samples])
    ppm_min = float(np.min(all_ppm))
    ppm_max = float(np.max(all_ppm))

    if n_bins is not None and int(n_bins) > 0:
        n_bins = int(n_bins)
        edges = np.linspace(ppm_min, ppm_max, n_bins + 1)
        bin_description = f"n_bins={n_bins}"
    else:
        bin_width = float(bin_width)
        if bin_width <= 0:
            raise ValueError("Bin width must be positive.")
        edges = np.arange(ppm_min, ppm_max + bin_width, bin_width)
        bin_description = f"width={bin_width:g} ppm"

    centers = (edges[:-1] + edges[1:]) / 2

    out = []
    rows = []

    for sample in samples:
        s = dict(sample)
        values = _integrate_bins(s["window_ppm"], s["region_removed"], edges, method=method)

        s["bin_edges"] = edges
        s["bin_centers"] = centers
        s["binned_values"] = values
        s["log"] = s.get("log", []) + [f"Binning: {bin_description}, method={method}."]

        rows.append(
            pd.Series(values, index=[f"{x:.4f}" for x in centers], name=s["name"])
        )
        out.append(s)

    return out, pd.DataFrame(rows)


def normalize_table(df: pd.DataFrame, method: str = "PQN") -> pd.DataFrame:
    if method == "none":
        return df.copy()

    if method == "TotalArea":
        out = df.copy()
        total = out.sum(axis=1)
        total[total == 0] = 1.0
        return out.div(total, axis=0)

    if method == "SNV":
        out = df.copy()
        centered = out.sub(out.mean(axis=1), axis=0)
        std = centered.std(axis=1)
        std[std == 0] = 1.0
        return centered.div(std, axis=0)

    if method == "PQN":
        out = df.copy()
        reference = out.median(axis=0)
        reference = reference.replace(0, np.nan)
        quotients = out.divide(reference, axis=1)
        factors = quotients.median(axis=1, skipna=True)
        factors = factors.replace(0, 1.0).fillna(1.0)
        return out.div(factors, axis=0)

    raise ValueError("Normalization must be 'PQN', 'TotalArea', 'SNV', or 'none'.")


def apply_normalization(samples: list[dict], binned_table: pd.DataFrame, method: str = "PQN") -> pd.DataFrame:
    for sample in samples:
        sample["log"] = sample.get("log", []) + [f"Normalization prepared: method={method}."]

    return normalize_table(binned_table, method=method)
