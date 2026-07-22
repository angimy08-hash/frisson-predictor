"""Vassilakis (2001) inspired spectral roughness approximation."""
import numpy as np
from scipy.signal import find_peaks


def frame_roughness(magnitude: np.ndarray, frequencies: np.ndarray, max_peaks: int = 24) -> float:
    """Estimate sensory dissonance from pairwise spectral peak interactions.

    This is a practical approximation, not a calibrated psychoacoustic model: nearby
    strong partials contribute most, with a critical-band distance decay.
    """
    peaks, _ = find_peaks(magnitude)
    if peaks.size < 2:
        return 0.0
    peaks = peaks[np.argsort(magnitude[peaks])[-max_peaks:]]
    f, a = frequencies[peaks], magnitude[peaks]
    i, j = np.triu_indices(f.size, 1)
    low = np.minimum(f[i], f[j])
    # Critical-band-width approximation in Hz (Sethares/Vassilakis family).
    x = np.abs(f[i] - f[j]) / (0.021 * low + 19.0)
    dissonance = np.exp(-3.5 * x) - np.exp(-5.75 * x)
    amplitude = (a[i] * a[j]) ** 0.5
    return float(np.sum(dissonance * amplitude))


def roughness_series(stft_magnitude: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    return np.asarray([frame_roughness(stft_magnitude[:, i], frequencies) for i in range(stft_magnitude.shape[1])])
