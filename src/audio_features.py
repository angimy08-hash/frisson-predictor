"""Windowed audio descriptors used by the Frisson predictor."""
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from roughness import roughness_series


def extract_features(audio_path: str | Path, window_seconds: float = 0.5, overlap: float = 0.5, sample_rate: int | None = None) -> tuple[pd.DataFrame, np.ndarray, int]:
def extract_features(
    audio_path: str | Path,
    window_seconds: float = 0.5,
    overlap: float = 0.5,
    sample_rate: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray, int]:
    # soundfile avoids platform-specific stalls seen in librosa.load on some Windows builds;
    # all spectral/onset descriptors below remain librosa implementations.
    y, sr = sf.read(str(audio_path), always_2d=False)

    if y.ndim > 1:
        y = y.mean(axis=1)

    y = np.asarray(y, dtype=np.float32)

    if sample_rate and sr != sample_rate:
        y = librosa.resample(y, orig_sr=sr, target_sr=sample_rate)
        sr = sample_rate

    frame_length = max(256, int(window_seconds * sr))
    hop_length = max(1, int(frame_length * (1 - overlap)))
    stft = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length, center=True))
    times = librosa.frames_to_time(np.arange(stft.shape[1]), sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(S=stft)[0]
    centroid = librosa.feature.spectral_centroid(S=stft, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(S=stft, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=stft)[0]
    roughness = roughness_series(stft, librosa.fft_frequencies(sr=sr, n_fft=frame_length))
    onset_times = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, units='time')
    # Onset-density proxy: count onsets in each analysis window.
    onset_density = np.asarray([np.sum(np.abs(onset_times - t) <= window_seconds / 2) / window_seconds for t in times])
    diff = lambda x: np.r_[0.0, np.diff(x)]

    # STFT
    stft = np.abs(
        librosa.stft(
            y,
            n_fft=frame_length,
            hop_length=hop_length,
            center=True,
        )
    )

    # Time axis
    times = librosa.frames_to_time(
        np.arange(stft.shape[1]),
        sr=sr,
        hop_length=hop_length,
    )

    # RMS (compute from waveform to avoid frame_length mismatch)
    rms = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]

    # Spectral features
    centroid = librosa.feature.spectral_centroid(
        S=stft,
        sr=sr,
    )[0]

    bandwidth = librosa.feature.spectral_bandwidth(
        S=stft,
        sr=sr,
    )[0]

    flatness = librosa.feature.spectral_flatness(
        S=stft,
    )[0]

    # Roughness
    roughness = roughness_series(
        stft,
        librosa.fft_frequencies(sr=sr, n_fft=frame_length),
    )

    # Onset density
    onset_times = librosa.onset.onset_detect(
        y=y,
        sr=sr,
        hop_length=hop_length,
        units="time",
    )

    onset_density = np.asarray([
        np.sum(np.abs(onset_times - t) <= window_seconds / 2) / window_seconds
        for t in times
    ])

    def diff(x):
        return np.r_[0.0, np.diff(x)]

    return pd.DataFrame({
        'time_s': times, 'rms': rms, 'rms_change': diff(rms),
        'centroid_hz': centroid, 'brightness_change': diff(centroid),
        'bandwidth_hz': bandwidth, 'bandwidth_change': diff(bandwidth),
        'flatness': flatness, 'roughness': roughness, 'roughness_change': diff(roughness),
        'onset_density': onset_density,
        "time_s": times,
        "rms": rms,
        "rms_change": diff(rms),
        "centroid_hz": centroid,
        "brightness_change": diff(centroid),
        "bandwidth_hz": bandwidth,
        "bandwidth_change": diff(bandwidth),
        "flatness": flatness,
        "roughness": roughness,
        "roughness_change": diff(roughness),
        "onset_density": onset_density,
    }), y, sr
