from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from audio_features import extract_features
from chill_predictor import (
    DEFAULT_WEIGHTS,
    prediction_peaks,
    score_features,
    frisson_score,
)
from validate_against_log import validate

st.set_page_config(page_title='Frisson 예측기', layout='wide')
st.title('음악 소름(Frisson) 유발 지점 예측기')
st.caption('음향 변화 5가지 단서를 기반으로 전율 발생 가능 구간을 예측하고, 버튼-프레스 로그와 비교합니다.')

with st.sidebar:
    st.header('표시·검증 옵션')
    st.caption('아래 값은 결과를 표시·비교하는 방법만 바꿉니다. 음향 단서 가중치는 연구근거에 따라 고정되어 있습니다.')
    window = st.slider('분석 창 길이(초)', .25, 2.0, .5, .25)
    overlap = st.slider('창 오버랩', .25, .75, .5, .25)
    top_percent = st.slider('예측 구간 상위 비율(%)', 1, 30, 10)
    tolerance = st.slider('Hit 허용 오차(초)', .5, 10.0, 3.0, .5)

audio_upload = st.file_uploader('분석할 오디오 파일', type=['wav', 'mp3', 'flac', 'ogg', 'm4a'])
st.info('예측 점수는 각 곡 내부에서 0–1로 정규화됩니다. 높은 점수는 확률의 임상적 추정치가 아니라 상대적 우선 분석 구간입니다.')

if audio_upload:
    suffix = Path(audio_upload.name).suffix or '.wav'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
        temp.write(audio_upload.getvalue()); audio_path = temp.name
    try:
        with st.spinner('음향 특성을 추출하고 있습니다…'):
            features, audio, sr = extract_features(audio_path, window, overlap)
            scored = score_features(features, DEFAULT_WEIGHTS, top_percent)
        peaks = prediction_peaks(scored)
        st.success(f'분석 완료 — {len(scored)}개 창, 예측 피크 {len(peaks)}개')
        st.markdown('''> **방법론 설명**  
> 본 예측은 Grewe et al.(2007), Guhn et al.(2007), Nagel et al.(2008), Huron & Margulis(2010)에서 보고된 5가지 음향적 상관요인을 근거로 한 것이며, 각 논문에서 확인된 상관관계 방향(양의 상관)만 반영한 탐색적 지표입니다.''')
        with st.expander('연구근거 기반 고정 가중치', expanded=False):
            st.markdown('''- 음량 급증: **0.30** — Grewe et al.(2007), Huron & Margulis(2010)에서 가장 일관되게 언급된 단서
- 밝기 증가: **0.20**
- 거칠기 증가: **0.20**
- 주파수 대역 확장: **0.15**
- onset/새 사건 밀도: **0.15**

가중치는 각 단서가 선행연구에서 보고된 빈도를 근거로 설정되었으며, 임의 조정값이 아닙니다. 밝기·거칠기·대역·새 사건 단서는 Guhn et al.(2007), Nagel et al.(2008)을 포함한 관련 연구에서의 상대적 언급을 반영했습니다.''')
        duration = len(audio) / sr
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, height_ratios=[1, 1.2])
        x = np.linspace(0, duration, len(audio))
        ax0.plot(x, audio, color='#566573', linewidth=.35); ax0.set_ylabel('파형'); ax0.set_title('오디오 파형과 Chill Likelihood Score')
        ax1.plot(scored.time_s, scored.chill_score, color='#126a8a', label='Chill Likelihood Score')
        ax1.fill_between(scored.time_s, 0, scored.chill_score, where=scored.is_predicted, color='#f0a33a', alpha=.35, label=f'상위 {top_percent}% 예측 구간')
        for p in peaks: ax1.axvline(p, color='#c94c3b', alpha=.5, linewidth=1)
        ax1.set(ylim=(0, 1.05), xlabel='시간 (초)', ylabel='점수'); ax1.legend(loc='upper right')

        st.pyplot(fig, clear_figure=True)
        st.subheader('내 청취 실험 결과와 비교하기 (선택사항)')
        st.caption('예측 결과는 로그 없이도 이미 표시됩니다. `chill_log.csv`를 추가하면 실제 버튼 클릭과의 일치도를 계산합니다.')
        log_upload = st.file_uploader('chill_log.csv 업로드', type=['csv'], key='chill_log')
        if log_upload:
            log = pd.read_csv(log_upload)
            summary, annotated = validate(scored, log, tolerance)
            # Recreate the plot here only when actual clicks are available.
            fig2, (ax0, ax1) = plt.subplots(2, 1, figsize=(13, 7), sharex=True, height_ratios=[1, 1.2])
            ax0.plot(x, audio, color='#566573', linewidth=.35); ax0.set_ylabel('파형'); ax0.set_title('오디오 파형·예측 점수·실측 클릭')
            ax1.plot(scored.time_s, scored.chill_score, color='#126a8a', label='Chill Likelihood Score')
            ax1.fill_between(scored.time_s, 0, scored.chill_score, where=scored.is_predicted, color='#f0a33a', alpha=.35, label=f'상위 {top_percent}% 예측 구간')
            for p in peaks: ax1.axvline(p, color='#c94c3b', alpha=.5, linewidth=1)
            colors = {'A': '#e67e22', 'B': '#8e44ad'}
            for condition, rows in annotated.groupby('condition'):
                ax0.vlines(rows.timestamp_s, -.98, .98, colors=colors.get(str(condition), '#222'), alpha=.65, linewidth=1, label=f'실측 {condition}')
            ax0.legend(loc='upper right')
            ax1.set(ylim=(0, 1.05), xlabel='시간 (초)', ylabel='점수'); ax1.legend(loc='upper right')
            st.pyplot(fig2, clear_figure=True)
            st.subheader('예측-실측 검증')
            view = summary.copy(); view['hit_rate'] = view['hit_rate'].map(lambda v: f'{v:.1%}' if pd.notna(v) else '—'); view['mean_time_difference_s'] = view['mean_time_difference_s'].map(lambda v: f'{v:.2f}' if pd.notna(v) else '—')
            st.dataframe(view, hide_index=True, use_container_width=True)
            st.caption(f'Hit: 실제 클릭이 예측 피크에서 ±{tolerance:.1f}초 이내인 경우. 조건 A/B 비교는 각 조건의 클릭을 별도 집계합니다.')
            st.download_button('검증 상세 CSV 다운로드', annotated.to_csv(index=False).encode('utf-8-sig'), 'frisson_validation_detail.csv', 'text/csv')
        st.subheader('구간별 분석값')
        st.dataframe(scored[['time_s','chill_score','is_predicted','rms_change','brightness_change','roughness_change','bandwidth_change','onset_density']], hide_index=True, use_container_width=True)
        st.download_button('예측 점수 CSV 다운로드', scored.to_csv(index=False).encode('utf-8-sig'), 'frisson_predictions.csv', 'text/csv')
    except Exception as exc:
        st.error(f'분석에 실패했습니다: {exc}')
        st.exception(exc)
else:
    st.markdown('''### 사용 방법
1. WAV, MP3 등 분석할 음원 파일을 업로드합니다.
2. 상위 예측 비율을 조정해 보고 싶은 후보 구간의 범위를 선택합니다. 다섯 단서 가중치는 연구근거에 따라 고정되어 있습니다.
3. 앞선 청취 실험에서 받은 `chill_log.csv`를 올리면 실측 클릭과의 일치도를 확인합니다.

**방법론 주석** — 본 도구는 Grewe et al. (2007), Guhn et al. (2007), Nagel et al. (2008), Huron & Margulis (2010)에서 논의된 음량, 밝기, 거칠기, 대역폭 및 새 사건 단서를 탐색적으로 조합합니다. 점수는 진단이나 개인별 전율을 확정하는 값이 아닙니다.''')
