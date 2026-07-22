# 음악 소름(Frisson) 유발 지점 예측기

임의의 오디오에서 전율과 연관된 음향 변화 단서를 추출하고, 버튼-프레스 실측 로그와 대조하는 Streamlit 도구입니다.

## 기능

- 0.5초 창(기본값, 50% 오버랩)에서 RMS 변화, 스펙트럴 중심주파수 변화, 대역폭 변화, Vassilakis 계열 근사 거칠기 변화, onset 밀도를 계산합니다.
- 다섯 지표를 양의 z-score로 정규화하고 가중합해 곡 내부 상대 `Chill Likelihood Score (0–1)`를 계산합니다.
- 상위 N% 구간과 그 지역 피크를 예측 지점으로 표시합니다.
- `chill_log.csv`의 클릭과 예측 피크 간 ±3초(조절 가능) hit rate, 평균 시간차를 전체 및 조건 A/B별로 계산합니다.

## 실행

```powershell
cd C:\Users\wksck\frisson-predictor
python -m pip install -r requirements.txt
streamlit run src/app.py
```

브라우저에서 표시된 로컬 URL을 열고 음원과 (선택) `chill_log.csv`를 업로드하세요. 로그는 `participant_id, condition, order, timestamp_ms` 열을 사용합니다. `timestamp_ms`는 오디오 재생 시작점 기준 상대 밀리초여야 합니다.

## 해석 주의

점수는 곡 안에서의 상대적 후보 구간이며 개인의 실제 전율을 확정하는 확률이 아닙니다. Vassilakis (2001)에 영감을 받은 거칠기 계산은 빠른 웹 분석을 위한 스펙트럼 피크 기반 근사입니다. 결과는 충분한 표본의 실측 버튼 로그 및 가능하다면 생리 데이터로 검증해야 합니다.

## 관련 배경

Grewe et al. (2007); Guhn, Hamm, & Zentner (2007); Nagel, Kopiez, Grewe, & Altenmüller (2008); Huron & Margulis (2010).
