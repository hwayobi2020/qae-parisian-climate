# D−2 대기강 지속 예보 캘리브레이션 모델

> 마지막 갱신: 2026-07-06. 이 문서는 "D−2(2일 전) 예보로 대기강 24시간 지속을 예측하는" 모델의 전체 설계를 기록한다.

## 1. 목적 / 핵심 질문
- 대기강(AR, Atmospheric River = 대기 중 수증기가 강처럼 좁고 길게 수송되는 현상)이 **발생 2일 전(D−2) 시점**에서, D일에 **24시간 이상 지속**될지 예측한다.
- 핵심 질문: **우리 모델이 기상청 원예보(raw forecast)를 ML로 캘리브레이션 해서 성능을 향상시킬 수 있는가?**
- 비교 기준(baseline)은 언제나 **원예보 그 자체**이지, 다른 모델이 아니다.

## 2. 데이터
| 요소 | 출처 | 기간 | 비고 |
|---|---|---|---|
| 예보 | GEFS v12 재예측(reforecast) | 2000~2019 (20년) | **D−2 00Z 발표, c00 컨트롤**, 리드 48~90h |
| env(환경, 관측/재분석) | IVT·순환지수·기압 등 | 1980~2023 (44년) | 인코더 학습용 |
| 지역 | 3곳 | — | CA(캘리포니아), UK(영국), Chile(칠레) |

- **왜 D-2인가**: 근접예보(0~18h)는 예보가 관측과 상관 0.9로 거의 정답을 측정 → 천장(예보만으로 이미 강함)이라 모델이 얹힐 여지가 작다. D−2는 예보가 약해(관측상관 0.55~0.66) 모델이 기여할 여지가 크다.
- 온셋까지 리드 = 48 + 온셋시각(OH) 시간 = **약 2~2.75일 전 예보**.

## 3. 타깃(정답 y)
- AR 판정: 6시간 IVT > THR (지역별 관측 85퍼센타일; **CA 249.8 / UK 457.1 / Chile 161.8**. 본문의 "250"은 CA 대표값, 실제 코드는 지역별 THR 사용)
- 온셋 s(대기강 시작 시점), 온셋시각 OH ∈ {0,6,12,18}시.
- **지속** = 온셋+6h·+12h·+18h의 관측 IVT가 모두 AR 수준(>250) 유지.
- **y = (지속시간 ≥ 24h) 이진 레이블.**

## 4. 기준선 (raw 예보)
- fcv = D−2 예보 IVT의 **온셋+6/+12/+18h 중 최소값**(min).
  - 왜 min: 24h 지속하려면 세 시점 IVT가 다 버텨야 하므로, 가장 낮은 지점(병목)이 "지속 여부"의 핵심 신호.
- **예보의 지속 예측 = `fcv ≥ THR`** (파라미터 없는 원예보 그대로, 고정 임계).

## 5. 우리 모델
- **입력 (메인 = C, 12개, 전부 c00 1멤버 예보 IVT):**
  1~8. IVT @ **고정 리드** 48·54·60·66·72·78·84·90h (정렬 없이 절대 리드 그대로 = **D8**)
  9~12. 예보 peak(onL) 지속창 요약 4개 = **min(=fcv) / mean / std** + **기울기**(onL+24 − onL)  (= D2의 요약부. onL = 리드 48~66h 중 예보 IVT 최대 시각)
- **이전 버전 = D2(peak정렬 9개)** — IVT@onL·+6·+12·+18·+24 + 요약4. **2026-07-07 ablation에서 C에 열등 → C로 교체 확정**: 고정리드(B_fix8/C) > peak정렬(A_peak9). 18h TabPFN CA ΔF1 +0.074→+0.117 / UK +0.029→+0.058(P0.94), 로컬 LGBM 24h 동일 순서. ("온셋"은 라벨 개념일 뿐 피처 아님 — 정렬은 항상 예보 기준.)
- **⚠️ §8 등 이 문서의 헤드라인 결과는 전부 아직 D2(peak9)로 뽑은 것 → C로 재산출 필요.** TabPFN Chile·24h·30h ablation 확인도 대기. 데이터=`build_op_denom_full.py`(D8 동봉), 셀=`colab_align_ablation.py`.
- 모델: **TabPFN 회귀(Regressor)** (사전학습된 트랜스포머, in-context fit — 그래디언트 학습 없음).
- 출력: **연속구간 관측 min IVT를 수치로 예측** (raw fcv와 같은 IVT 단위·의미). 회귀 타깃 = 관측 min IVT(온셋+6/12/18h).
- 판정: **예측 min IVT ≥ 250(THR) → 지속(발생), < 250 → 미지속.** raw와 완전히 같은 임계·규칙(확률 0.5 없음).
- 역할: TabPFN은 날씨를 예보하지 않는다. GEFS가 만든 예보값을 **어떻게 지속 판정으로 번역하느냐**만 학습한다(원예보 위의 후처리기/해석기). GEFS는 기준·모델 양쪽 밑에 공통으로 깔린다.

## 6. env 인코더 · 관측 env 추가 (검증된 폐기)
- 동기: 관측 env(온셋-시점 nowcast 프레임에서 leak-free AUC 0.66~0.74)의 지속 신호를 D−2 예보모델에 얹으려 함. env 44 = IVT 16일 웨이블릿·제트·블로킹·기압·IVT방향·다중스케일 IVT·격자 순환장·ENSO 등.
- **테스트 1 (예보온셋 분모, pre-2000 인코더)**: env 44를 D−2 컷으로 pre-2000 MLP(hid=8, leak-free) 학습 → 2000~2019 투입. 회귀 예보9+인코더8: CA F1 0.687→0.673(악화), UK·Chile 변화 없음.
- **테스트 2 (완전분모, 2000~2019 폴드별 인코더, 2026-07-06)**: env 44 → MLP(→8차원, 3클래스 헤드: 비온셋/온셋<24h/온셋≥24h) 폴드별 train만 학습(누수 없음) → 예보9 위에 8차원. **전 지역 F1 폭락**: CA +0.064→−0.233, UK −0.021→−0.175, Chile +0.168→−0.110. AUC는 유지·F1만 붕괴 = 무차원 인코더 8차원이 IVT-스케일 회귀의 THR 판정을 de-calibration(희석)함.
- **테스트 3 (완전분모, IVT 16일 웨이블릿만 직접, 인코더 없음)**: 예보9 + IVT웨이블릿6 = 15 → TabPFN. **전 지역 ΔF1 사실상 불변**: CA +0.064→+0.052, UK −0.021→−0.019, Chile +0.168→+0.167. AUC·AUPRC 동일. → 정보 무증가(해치지도 않음).
- **결론: D−2 예보가 관측 env(저주파 IVT 배경 포함)를 충분통계로 흡수** → env를 얹을 이득 0. 최종 헤드라인 = **예보피처9만(env 없음)**. 막연한 폐기가 아니라 세 테스트로 검증된 폐기. (스크립트: `build_op_denom_full.py`가 ENV 44·y3 동봉, `colab_opdenom_enc.py`가 테스트 2·3.)

## 7. 평가 방법
- **지표: F1** (양성=지속 클래스의 precision·recall 조화평균, TN 다수 제외). 
- **단일 임계 250(THR), 대칭**: raw `fcv ≥ 250` / 우리모델 `예측 min IVT ≥ 250`. 두 모델 다 IVT 수치를 내고 같은 250으로 자름 → 임의 임계(0.5) 없음.
- 교차검증: 워크포워드 폴드(테스트 5블록, 64일 임베고). 유의성: 지역별 부트스트랩 → 평균 Δ의 CI/P, 3지역 통합검정.

## 8. 운영 정합 분모 — 0.5THR 전체 (예보온셋 정렬)
- **운영 정합 수정 (핵심)**: 예측 시점(D−2)엔 **예보만** 있음 → 피처는 **예보에서 정렬**(관측 온셋 시각 사용 금지). 이전 버전(관측온셋 정렬)은 실전에서 만들 수 없어 폐기.
- **혼동행렬**: TT(관측O 예보O) / TF(관측O 예보X=예보미스) / FT(관측X 예보O=거짓경보) / FF(둘다X).
- **분모 = 0.5THR 전체(TT+TF+FT+FF)**: 관측 일최대 **≥0.5THR** 인 모든 후보일. 질문 = "**≥0.5THR 날에 2일 전 예보로 24h+ 지속 AR 발생을 맞히나**". 라벨 = 관측 온셋 지속(온셋: ivt[s+1:s+4]≥THR / near-miss: 0). 피처 = 예보 peak 정렬(그날 최대 예보 IVT 시각). `build_op_denom_full.py` → opdenom_full_{r}.npz.
  - CA 2882(온셋 801+nm 2081) 지속200 base0.069 rawF1 0.414(AUC 0.899) / UK 3297(863+2434) 지속128 base0.039 rawF1 0.293(AUC 0.860) / Chile 3017(653+2364) 지속233 base0.077 rawF1 0.414(AUC 0.894).
  - raw는 recall이 낮음 — 완전 분모엔 예보가 약한 TF가 많아 raw가 지속을 대거 놓침(FN). 모델의 기회 = recall 복원.

- **결과 (3지역 + 통합, 헤드라인 = TabPFN 회귀, 2026-07-06):**

  | 지역 | raw F1 | TabPFN F1 | ΔF1 | 95%CI | P(Δ>0) | 판정 |
  |---|---|---|---|---|---|---|
  | CA | 0.414 | 0.478 | +0.064 | [−0.006,+0.135] | 0.963 | marginal (단측 p=0.037) |
  | UK | 0.293 | 0.272 | −0.021 | [−0.112,+0.067] | 0.327 | null (약간 악화) |
  | Chile | 0.414 | 0.582 | +0.168 | [+0.107,+0.231] | 1.000 | 강함 |
  | **통합** | — | — | **+0.070** | **[+0.026,+0.116]** | **1.000** | **유의(양측 CI 0 배제)** |

- **5모델 통합검정 (지역평균 ΔF1 vs raw, 부트스트랩):** LR +0.003 [−0.040,+0.043] P0.570 / LGBM +0.063 [+0.021,+0.107] P1.000 / LSTM +0.064 [+0.016,+0.109] P0.996 / **TabPFN +0.070 [+0.026,+0.116] P1.000** / TabNet +0.041 [−0.005,+0.091] P0.955. → LGBM·LSTM·TabPFN 통합 유의, TabPFN 최고, LR flat.

- **판정 = 조건부 재현 (클린 3지역 재현은 아님)**: Chile 강함 / CA marginal / **UK 무효**(TabPFN −0.021, LR은 유의하게 음수 ΔF1 −0.111 p0.007 — 순진한 선형 보정은 UK를 오히려 해침). 통합(3지역 메타)은 TabPFN +0.070, 양측 95%CI가 0을 배제 → 유의하지만, **per-region은 예보 스킬에 조건부**.
  - **원칙 (예외 아님)**: 이 방법은 D−2 예보 스킬의 재보정(후처리)이라 **예보 자체에 지속 설명력이 있어야 작동**. UK raw F1 0.293·AUC 0.860 = 3지역 최약 → 보정할 지반 부족 → UK 무효는 방법의 전제이지 실패가 아니다. (이전 예보온셋 분모에서도 UK flat −0.003으로 일관.)
  - **이득 정체**: 완전분모에선 raw AUC 0.899→TabPFN 0.921(CA), 즉 보정 + 얇은 랭킹개선. env 추가는 예보에 흡수돼 이득 0(§6).

- **지속 horizon robustness (24h vs 30h, TabPFN, 2026-07-06):** 지속 임계를 30h(무손실 최대 — 온셋+24h 필요 ≤ 예보 리드 90h)로 올려도 이득 유지·강화.

  | 지역 | raw F1 24h→30h | TabPFN ΔF1 24h | TabPFN ΔF1 30h | 30h 95%CI | 30h P |
  |---|---|---|---|---|---|
  | CA | 0.414→0.382 | +0.064 (marginal) | **+0.097** | [+0.018,+0.181] | 0.990 |
  | UK | 0.293→0.341 | −0.021 | −0.025 | [−0.151,+0.089] | 0.331 |
  | Chile | 0.414→0.400 | +0.168 | **+0.173** | [+0.102,+0.244] | 1.000 |
  | **통합** | — | +0.070 | **+0.082** | **[+0.026,+0.135]** | 0.997 |

  - 통합 이득이 30h에서 **오히려 커지고**(+0.070→+0.082), CA가 marginal→**양측 유의로 승격**(CI 0 배제). 조건부 패턴(Chile 강·UK 무효)은 두 horizon **일관** → 임계 정의에 강건.
  - **메커니즘**: 엄격한 30h에서 raw는 더 떨어지고(CA 0.414→0.382) 모델은 유지(0.478→0.480) → 격차 확대. "보정은 임계가 엄격할수록 더 값지다."
  - 30h 5모델 통합: LR −0.011 / LGBM +0.061 / LSTM +0.061 / TabNet +0.061 / **TabPFN +0.082(P0.997)** — TabPFN이 30h서 타모델과 분리(24h는 상위3 군집).
  - **한계**: UK 30h 양성 70개(base 0.021)로 노이즈 큼(CI 넓음). 36h+는 온셋+30h(≤96h)로 예보 리드 90h 초과 → 후보 30~55% 드롭이라 무손실 불가.

- **모델 정체성 / 명명**: **D−2 AR 지속예보 보정기(statistical calibrator / post-processing / MOS 계열)**. 기여 = "damped D−2 예보가 THR에서 잃는 지속 스킬을 통계 보정으로 복원"; env는 예보에 흡수돼 추가 이득 없음.

## 9. 관련 연구 · 포지셔닝 (목표 = Q3급 open-access 빠른 게재)
- **가장 가까운 선행연구**: **Chapman et al. 2019 (GRL) / 2022 (MWR)** — 단일멤버 AR 예보를 ML/DL로 후처리. 우리와 **같은 계열**(단일멤버·reforecast·서해안). **단, 예측 대상 = IVT 세기/확률**(0~120h, CNN/NN/FCN, CRPS·Brier로 검증).
  - *검증 수준: CW3E 공지(저자 기관 요약)에 "focuses exclusively on IVT intensity/magnitude, **not duration or persistence** ... no mention of forecasting how long an AR lasts"라고 명시. AMS 전문 PDF는 미열람(403) — 인용은 공지 수준(Q3엔 충분).*
- **일반 배경**: 앙상블 후처리·MOS(강수·streamflow·바람·가시거리)는 확립된 분야 — "예보를 ML로 보정"이라는 큰 아이디어 자체는 새롭지 않음.
- **우리 차별점 (novelty는 방법이 아니라 대상·진단)**:
  1. **예측 대상 = 24h 지속(duration/persistence)**, IVT 세기 아님. **AR 스케일(Ralph)이 24h/48h 지속으로 등급을 조정**하므로 물리적으로 중요한 별개 축.
  2. **정직한 보정-vs-정보 진단** (ΔAUC≈0 → 이득은 damped 예보 재보정이지 새 랭킹 정보 아님).
  3. **운영 분모(TT/TF/FT, 거짓경보 포함)** 명시 구성·평가.
- **포지셔닝 한 줄**: "Chapman et al.과 같은 단일멤버 후처리 계열이되, **예측 대상을 IVT 세기 → 24h 지속으로 옮기고**, 이득이 정보가 아니라 보정임을 진단한다."
- **앙상블 = future work**: 5멤버는 초기섭동의 카오스 분산(노이즈 아님, 불확실성 정보). 단 이 **이진 지속 재보정** 과제엔 컨트롤(c00)로 충분하다는 입장 → 앙상블 확장은 향후과제로.

## 10. 참고문헌 (핵심 레퍼런스)
> ⚠️ 저자·연도 일부는 검색 스니펫 기준 — 최종 투고 전 원문으로 서지 확인 필요.

**AR ML 후처리 (직접 관련 — same family, 대상은 IVT 세기):**
- Chapman, W. E., et al. (2019). Improving Atmospheric River Forecasts With Machine Learning. *Geophysical Research Letters*. https://doi.org/10.1029/2019GL083662
- Chapman, W. E., Delle Monache, L., Alessandrini, S., Subramanian, A. C., Ralph, F. M., Xie, S., Lerch, S., & Hayatbini, N. (2022). Probabilistic Predictions from Deterministic Atmospheric River Forecasts with Deep Learning. *Monthly Weather Review*, 150(1). https://doi.org/10.1175/MWR-D-21-0106.1

**AR 스케일 · duration (우리 동기):**
- CW3E Atmospheric River Scale (Ralph et al. 2019; 지속시간 × 최대 IVT로 Cat 1–5, 24h/48h가 등급 조정). https://cw3e.ucsd.edu/arscale/

**AR ML 기타:**
- Higgins et al. (2023). Using Deep Learning for an Analysis of Atmospheric Rivers in a High-Resolution Large Ensemble Climate Data Set. *JAMES*. https://doi.org/10.1029/2022MS003495
- Global performance benchmarking of AI models in atmospheric river forecasting (2025). *Communications Earth & Environment*. https://www.nature.com/articles/s43247-025-02823-y
- Role of Machine Learning in Understanding and Managing Atmospheric Rivers. *Springer* (2024). https://link.springer.com/chapter/10.1007/978-3-031-63478-9_5

**AR subseasonal / S2S 예측:**
- Skillful empirical subseasonal prediction of landfalling AR activity using the MJO and QBO (2017). *npj Climate and Atmospheric Science*. https://www.nature.com/articles/s41612-017-0008-2
- S2S prediction of atmospheric rivers in the Northern Winter (2024). *npj Climate and Atmospheric Science*. https://www.nature.com/articles/s41612-024-00827-7

**앙상블 후처리 · MOS 배경:**
- Machine learning for postprocessing ensemble streamflow forecasts. *Journal of Hydroinformatics*, 25(1). https://iwaponline.com/jh/article/25/1/126/

## 파일 맵
- `export_transfer_feats.py` → transfer_{region}.npz (env 44, y, oday, s 온셋인덱스)
- `export_d2feat.py` → d2feat_{region}.npz (D−2 예보피처 9 + D2min)
- `export_d2env.py` → d2env_{region}.npz (D−2 컷 env 44, THR, omin 회귀타깃)
- `export_nearmiss.py` → nearmiss_{region}.npz (거짓경보 후보일: 관측 [0.5THR,THR), init_str=D−2 발표일)
- `gefs_ivt.py` (GEFS_NEARMISS=1) → gefs_ivt_{r}_d2c00fa.npz (near-miss 날 D−2 c00 예보)
- `build_op_denom.py` → opdenom_ca.npz (예보기준 분모 TT+TF+FT: D2 피처9, fcv, y, omin, oday, THR)
- `colab_opdenom_ca.py` → Colab 셀 (예보기준 분모에서 raw vs 5모델 F1/AUC/AUPRC)
- `build_op_denom_full.py` → opdenom_full_{r}.npz(24h; D2 9 + fcv/y/omin/oday/THR + **ENV 44 D−2컷** + **y3 3클래스**) + opdenom_full_{r}_30h.npz(30h 코어필드) — 3지역, HORIZONS=[24,30]
- `colab_opdenom.py` → Colab 셀 (완전분모 3지역 × **24h·30h** raw vs 5모델 F1/AUC/AUPRC + 통합검정) — §8 헤드라인·robustness
- `colab_opdenom_enc.py` → Colab 셀 (완전분모 env 추가 테스트: 인코더 44→8·3클래스 / IVT웨이블릿6 직접) — §6 테스트 2·3
- `colab_d2_tabpfn.py` → Colab 실행 셀 (기준 raw fcv≥THR / 우리모델 TabPFN회귀 예측 min IVT≥THR, F1, 통합검정) — 회귀 설계 완료·실행됨
