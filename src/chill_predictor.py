"""Five-cue Chill Likelihood Score calculation."""
import numpy as np
import pandas as pd

# Fixed, literature-informed starting weights. These are not participant-facing
# tuning controls: loudness is weighted highest because it is the most consistently
# reported cue in the cited frisson literature.
DEFAULT_WEIGHTS = {'loudness': .30, 'brightness': .20, 'roughness': .20, 'bandwidth': .15, 'onset': .15}


def positive_z(values: pd.Series) -> np.ndarray:
    x = values.to_numpy(dtype=float)
    sd = x.std()
    z = np.zeros_like(x) if sd < 1e-12 else (x - x.mean()) / sd
    return np.maximum(z, 0)


def score_features(features: pd.DataFrame, weights: dict | None = None, top_percent: float = 10.0) -> pd.DataFrame:
    weights = {**DEFAULT_WEIGHTS, **(weights or {})}
    total = sum(max(0, w) for w in weights.values())
    if total == 0:
        raise ValueError('At least one cue weight must be positive.')
    w = {k: max(0, v) / total for k, v in weights.items()}
    out = features.copy()
    cues = {
        'loudness': positive_z(out['rms_change']),
        'brightness': positive_z(out['brightness_change']),
        'roughness': positive_z(out['roughness_change']),
        'bandwidth': positive_z(out['bandwidth_change']),
        'onset': positive_z(out['onset_density']),
    }
    for name, values in cues.items(): out[f'z_{name}'] = values
    raw = sum(w[name] * cues[name] for name in cues)
    # Monotonic 0–1 normalization makes scores interpretable within an individual song.
    out['chill_score'] = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)
    threshold = np.percentile(out['chill_score'], 100 - top_percent)
    out['is_predicted'] = out['chill_score'] >= threshold
    return out


def prediction_peaks(scored: pd.DataFrame) -> np.ndarray:
    score = scored['chill_score'].to_numpy()
    selected = scored['is_predicted'].to_numpy()
    peaks = []
    for start in np.where(np.diff(np.r_[False, selected, False].astype(int)) == 1)[0]:
        end = np.where(np.diff(np.r_[False, selected, False].astype(int)) == -1)[0]
        end = end[end > start][0]
        local = start + np.argmax(score[start:end])
        peaks.append(scored['time_s'].iloc[local])
    return np.asarray(peaks)
def frisson_score(scored: pd.DataFrame) -> int:
    """
    Convert chill_score (0~1) into a presentation-friendly 0~100 score.
    """

    if len(scored) == 0:
        return 0

    peak = scored["chill_score"].max()
    mean = scored["chill_score"].mean()

    # Peak를 조금 더 크게 반영
    score = int((0.7 * peak + 0.3 * mean) * 100)

    return max(0, min(score, 100))
