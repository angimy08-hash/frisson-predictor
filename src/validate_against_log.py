"""Comparison of predicted frisson regions against button-press logs."""
import numpy as np
import pandas as pd
from chill_predictor import prediction_peaks


def validate(scored: pd.DataFrame, chill_log: pd.DataFrame, tolerance_seconds: float = 3.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    needed = {'participant_id', 'condition', 'timestamp_ms'}
    missing = needed - set(chill_log.columns)
    if missing: raise ValueError(f'chill_log.csv에 필요한 열이 없습니다: {", ".join(sorted(missing))}')
    clicks = chill_log.copy(); clicks['timestamp_s'] = pd.to_numeric(clicks['timestamp_ms'], errors='coerce') / 1000
    clicks = clicks.dropna(subset=['timestamp_s'])
    peaks = prediction_peaks(scored)
    if peaks.size:
        clicks['nearest_peak_s'] = clicks['timestamp_s'].apply(lambda t: float(peaks[np.argmin(np.abs(peaks-t))]))
        clicks['time_difference_s'] = np.abs(clicks['timestamp_s'] - clicks['nearest_peak_s'])
        clicks['hit'] = clicks['time_difference_s'] <= tolerance_seconds
    else:
        clicks['nearest_peak_s'], clicks['time_difference_s'], clicks['hit'] = np.nan, np.nan, False
    summary = clicks.groupby('condition', dropna=False).agg(clicks=('hit','size'), hits=('hit','sum'), hit_rate=('hit','mean'), mean_time_difference_s=('time_difference_s','mean')).reset_index()
    overall = pd.DataFrame([{'condition':'전체', 'clicks':len(clicks), 'hits':int(clicks['hit'].sum()), 'hit_rate':clicks['hit'].mean() if len(clicks) else np.nan, 'mean_time_difference_s':clicks['time_difference_s'].mean()}])
    return pd.concat([summary, overall], ignore_index=True), clicks
