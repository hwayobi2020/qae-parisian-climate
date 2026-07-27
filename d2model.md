# D−2 대기강 지속 예보 캘리브레이션 모델

> 마지막 갱신: 2026-07-19. 이 문서는 "D−2(2일 전) 예보로 대기강 24시간 지속을 예측하는" 모델의 전체 설계를 기록한다.

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

**한 줄 요약**: 이틀 전(D−2) 발표된 GEFS 예보에서 뽑은 숫자 12개를 입력으로, "관측될 min IVT"를 회귀로 예측하고, 원예보와 똑같은 임계값으로 지속 여부를 판정한다.

### 입력 피처 8개 (메인 피처셋 = B: 고정 리드 원값. 전부 GEFS 컨트롤 멤버 c00 한 개의 예보 IVT에서 나옴)

| 번호 | 내용 |
|---|---|
| 1~8 | 발표 시점(D−2일 00시)부터 48·54·60·66·72·78·84·90시간 뒤, 8개 시점의 예보 IVT 값을 그대로 넣는다(= **D8**). 어떤 날이든 피처 1번은 항상 "발표+48시간" 값 — 날마다 시점을 밀지 않는다(정렬 없음). 피처에는 peak정렬·온셋 개념이 전혀 없다. |

- 용어: **fcv = forecast value(예보값)** — 예보 peak(발표 후 48~66시간 중 예보 IVT 최대 시각) 뒤 +6/+12/+18시간 예보값의 최소. **raw 기준선의 판정값이며(§4), 피처가 아니다.** ("온셋"도 라벨을 만들 때만 쓰는 개념이고 피처가 아니다.)
- **검토 후 제외한 피처 변형들 (2026-07-19 확정, `colab_align_ablation.py` 완전분모 6모델 × 18/24/30h):**
  - **A (peak정렬 9개, 구버전 D2)**: 예보 peak 기준 궤적 5점 + 요약 4개. 고정리드에 전 블록 열등 (예: 18h TabPFN 통합 ΔF1 +0.092 vs B +0.120) → 폐기.
  - **C (= B + peak정렬 요약 4개: min/mean/std/기울기)**: C−B paired 부트스트랩 직접검정 18칸(6모델×3horizon) 중 유의 4칸 — **3칸이 B 우세**(TabPFN 24h Δ−0.021 P0.026, TabNet 24h −0.047 P0.012 · 30h −0.049 P0.030), C 우세는 TabICL 30h(+0.036 P0.994) 1칸. 나머지 14칸 무차이. → 요약4의 일관된 이득 없음, 더 단순한 B 채택.
  - **D (C + env 인코더 8차원)**: 전 블록 유의한 악화 — §6의 2026-07-19 재검증 참조(과적합 아티팩트로 규명).
  - **E (C + IVT 16일 웨이블릿 6개 직접)**: 전 블록에서 C와 동일 수준(무증가·무해) — env 흡수 결론 유지(§6).

### 모델·출력·판정

- **모델**: TabPFN 회귀 — 표(tabular) 데이터용으로 사전학습된 트랜스포머. 우리 데이터로 그래디언트 학습(파인튜닝)을 하지 않고, train 데이터를 문맥(in-context)으로 주면 바로 예측한다.
- **출력**: 지속창의 **관측 min IVT를 수치로 예측** (raw의 fcv와 같은 단위·같은 의미). 회귀 타깃 = 관측 min IVT(온셋+6/12/18h).
- **판정**: **예측 min IVT ≥ 지역별 임계값(THR) → 지속, 미만 → 미지속.** 원예보와 완전히 같은 임계·같은 규칙이라 확률 0.5 같은 임의 임계가 없다.
- **역할**: 이 모델은 날씨를 예보하지 않는다. GEFS가 만든 예보값을 **지속 판정으로 어떻게 번역할지**만 학습하는 후처리기다. GEFS는 기준선(raw)과 우리 모델 양쪽 밑에 공통으로 깔려 있다.

## 6. env 인코더 · 관측 env 추가 (검증된 폐기)
- 동기: 관측 env(온셋-시점 nowcast 프레임에서 leak-free AUC 0.66~0.74)의 지속 신호를 D−2 예보모델에 얹으려 함. env 44 = IVT 16일 웨이블릿·제트·블로킹·기압·IVT방향·다중스케일 IVT·격자 순환장·ENSO 등.
- **테스트 1 (예보온셋 분모, pre-2000 인코더)**: env 44를 D−2 컷으로 pre-2000 MLP(hid=8, leak-free) 학습 → 2000~2019 투입. 회귀 예보9+인코더8: CA F1 0.687→0.673(악화), UK·Chile 변화 없음.
- **테스트 2 (완전분모, 2000~2019 폴드별 인코더, 2026-07-06)**: env 44 → MLP(→8차원, 3클래스 헤드: 비온셋/온셋<24h/온셋≥24h) 폴드별 train만 학습(누수 없음) → 예보9 위에 8차원. **전 지역 F1 폭락**: CA +0.064→−0.233, UK −0.021→−0.175, Chile +0.168→−0.110. AUC는 유지·F1만 붕괴 = 무차원 인코더 8차원이 IVT-스케일 회귀의 THR 판정을 de-calibration(희석)함.
- **테스트 3 (완전분모, IVT 16일 웨이블릿만 직접, 인코더 없음)**: 예보9 + IVT웨이블릿6 = 15 → TabPFN. **전 지역 ΔF1 사실상 불변**: CA +0.064→+0.052, UK −0.021→−0.019, Chile +0.168→+0.167. AUC·AUPRC 동일. → 정보 무증가(해치지도 않음).
- **결론: D−2 예보가 관측 env(저주파 IVT 배경 포함)를 충분통계로 흡수** → env를 얹을 이득 0. 최종 헤드라인 = **예보피처만(env 없음)**. (스크립트: `build_op_denom_full.py`가 ENV 44·y3 동봉, `colab_opdenom_enc.py`가 테스트 2·3.)
- **재검증 (2026-07-19, 고정리드 base·전 horizon)**: 테스트 2의 "인코더 F1 폭락"은 **env에 정보가 없다는 증거가 아니라 인코더 과적합 아티팩트**로 규명됨. 인코더(파라미터 883개, 폴드 train 수백 행에 300에폭)가 train 라벨을 사실상 암기(지속클래스 재현율 train 0.89~1.00 vs test 0.00~0.29)하고, 그 출력을 같은 train 행에 붙여 메인 모델을 학습시키니 메인 모델이 이 열들에 크게 의존(LGBM importance 24~35%)했다가 test에서 붕괴하는 구조. 파라미터를 45개(로지스틱 44→지속확률 1)로 줄이면 폭락이 사라지고 두 지역에서 소폭 양수(+0.01~0.02, 유의성 미검정). **폐기 결론 자체는 유지** — 근거는 "인코더 폭락"이 아니라, (i) 웨이블릿 직접 투입(테스트 3·E)이 고정리드 base에서도 전 블록 무증가, (ii) 공정한 초소형 인코더도 이득이 노이즈 수준이라는 것.

## 7. 평가 방법
- **지표: F1** (양성=지속 클래스의 precision·recall 조화평균, TN 다수 제외). 
- **단일 임계 250(THR), 대칭**: raw `fcv ≥ 250` / 우리모델 `예측 min IVT ≥ 250`. 두 모델 다 IVT 수치를 내고 같은 250으로 자름 → 임의 임계(0.5) 없음.
- 교차검증: 워크포워드 폴드(테스트 5블록, 64일 임베고). 유의성: 지역별 부트스트랩 → 평균 Δ의 CI/P, 3지역 통합검정.

## 8. 운영 정합 분모 — 0.5THR 전체 (예보온셋 정렬)
- **운영 정합 수정 (핵심)**: 예측 시점(D−2)엔 **예보만** 있음 → 피처는 **예보에서만** 생성(관측 온셋 시각 사용 금지). 이전 버전(관측온셋 정렬)은 실전에서 만들 수 없어 폐기. 현재 메인 피처 B는 정렬 자체가 없음(고정 리드 원값 8개, §5).
- **혼동행렬**: TT(관측O 예보O) / TF(관측O 예보X=예보미스) / FT(관측X 예보O=거짓경보) / FF(둘다X).
- **분모 = 0.5THR 전체(TT+TF+FT+FF)**: 관측 일최대 **≥0.5THR** 인 모든 후보일. 질문 = "**≥0.5THR 날에 2일 전 예보로 24h+ 지속 AR 발생을 맞히나**". 라벨 = 관측 온셋 지속(온셋: ivt[s+1:s+4]≥THR / near-miss: 0). 피처 = 고정 리드 8개(B, §5; raw 기준선 fcv만 예보 peak 정렬). `build_op_denom_full.py` → opdenom_full_{r}.npz.
  - CA 2882(온셋 801+nm 2081) 지속200 base0.069 rawF1 0.414(AUC 0.899) / UK 3297(863+2434) 지속128 base0.039 rawF1 0.293(AUC 0.860) / Chile 3017(653+2364) 지속233 base0.077 rawF1 0.414(AUC 0.894).
  - raw는 recall이 낮음 — 완전 분모엔 예보가 약한 TF가 많아 raw가 지속을 대거 놓침(FN). 모델의 기회 = recall 복원.

- **결과 (헤드라인 = 지역별 조건부, TabPFN 회귀, 피처 B, 2026-07-19):** ※ 지역별 CI는 미산출(P만), 통합만 CI 있음.

  | 지역 | raw F1 | TabPFN F1 | ΔF1 | P(Δ>0) | 판정 |
  |---|---|---|---|---|---|
  | CA | 0.414 | 0.502 | +0.087 | 0.99 | 성립 |
  | UK | 0.293 | 0.359 | +0.066 | 0.87 | 비유의 (부호 양수) |
  | Chile | 0.414 | 0.601 | +0.187 | 1.00 | 강함 |
  | (참고) 통합 | — | — | +0.114 | [+0.064,+0.163] P1.000 | 3지역 평균·Chile 주도 — 일반화 주장 아님 |

- **판정 = 지역별 조건부 재현이 헤드라인**: Chile 강함 / CA 성립 / **UK 비유의**. 통합검정은 UK 약세를 Chile 강세로 평균해 가리므로 성공 근거가 아니라 부차 참고로만 둔다.
  - **원칙 (예외 아님)**: 이 방법은 D−2 예보 스킬의 재보정(후처리)이라 **예보 자체에 지속 설명력이 있어야 작동**. UK raw F1 0.293·AUC 0.860 = 3지역 최약 → 보정할 지반이 얇다. 단, 구버전 peak정렬 피처에서 "UK 무효(−0.021, 악화)"였던 것이 **고정리드 B에서는 18h +0.044 / 24h +0.066으로 부호가 양수로 돌아섬**(비유의) → "악화"가 아니라 "이득이 유의 수준에 못 미침"이 정확한 서술.
- **6모델 통합검정 (지역평균 ΔF1 vs raw, 피처 B, 24h):** LR +0.051 [+0.001,+0.102] P0.977 / LGBM +0.105 P1.000 / LSTM +0.057 P0.989 / TabPFN +0.114 [+0.064,+0.163] P1.000 / TabICL +0.086 P1.000 / **TabNet +0.124 [+0.074,+0.174] P1.000**. → 6모델 전부 방향 일치·유의(LR만 경계선). **통합 수치는 TabNet이 근소 최고**(TabPFN 대비 +0.01 수준, 전 horizon 일관) — 헤드라인 모델 서술 시 유의.
- **이득 정체**: raw AUC가 이미 0.899(CA)로 천장 → 이득은 랭킹 개선이 아니라 재보정. (ΔAUC≈0 진단 수치는 구버전 peak9 피처로 측정한 것 — B 기준 AUC 재산출은 미실행. 결론 방향은 §6 env 흡수와 함께 유지.)

- **지속 horizon robustness (18h·24h·30h, TabPFN, 피처 B, 2026-07-19):** 30h = 무손실 최대(온셋+24h 필요 ≤ 예보 리드 90h). ※ 지역별은 P만, 통합만 CI.

  | 지역 | ΔF1 18h (P) | ΔF1 24h (P) | ΔF1 30h (P) |
  |---|---|---|---|
  | CA | +0.116 (1.00) | +0.087 (0.99) | +0.096 (0.98) |
  | UK | +0.044 (0.89) | +0.066 (0.87) | −0.003 (0.47) |
  | Chile | +0.200 (1.00) | +0.187 (1.00) | +0.181 (1.00) |
  | 통합 | +0.120 [+0.085,+0.157] | +0.114 [+0.064,+0.163] | +0.091 [+0.027,+0.153] |

  - **CA·Chile는 전 horizon 견고**(CA +0.087~+0.116, Chile +0.181~+0.200, 전부 P≥0.98). **UK는 18h·24h 양수(비유의)·30h 0** → 조건부 패턴(예보 스킬이 얇은 UK만 이득 없음)이 horizon에 강건.
  - 통합 이득 크기는 18h > 24h > 30h 순(+0.120→+0.114→+0.091, 전부 유의) — 30h로 갈수록 UK·양성 수 감소가 통합을 끌어내림. (구버전 peak9의 "30h에서 통합 이득 확대" 서술은 B 기준으로 성립하지 않아 폐기.)
  - 30h 6모델 통합: LR −0.004 P0.449 / LGBM +0.080 P0.994 / LSTM +0.080 P0.995 / TabPFN +0.091 P0.995 / TabICL +0.056 P0.963 / TabNet +0.100 P0.999.
  - **한계**: UK 30h 양성 70개(base 0.021)로 노이즈 큼. 36h+는 온셋+30h(≤96h)로 예보 리드 90h 초과 → 후보 30~55% 드롭이라 무손실 불가. LR은 UK에서 일관 음수(24h −0.057, 30h −0.192) — 순진한 선형 보정은 UK에 해롭다.

- **모델 정체성 / 명명**: **D−2 AR 지속예보 보정기(statistical calibrator / post-processing / MOS 계열)**. 기여 = "damped D−2 예보가 THR에서 잃는 지속 스킬을 통계 보정으로 복원"; env는 예보에 흡수돼 추가 이득 없음.

## 9. 관련 연구 · 포지셔닝 (목표 = Q3급 open-access 빠른 게재)
- **가장 가까운 선행연구**: **Chapman et al. 2019 (GRL) / 2022 (MWR)** — 단일멤버 AR 예보를 ML/DL로 후처리. 우리와 **같은 계열**(단일멤버·reforecast·서해안). **단, 예측 대상 = IVT 세기**이며 두 논문의 방법 계열이 다름:
  - **2019 (ARcnn)**: West-WRF/GFS **IVT 필드의 결정론적 오차 보정**(RMSE 9~17%↓, 리드 3h~7d). 값 자체를 더 정확한 값으로.
  - **2022**: 단일멤버 예보에 **확률분포(가우시안 μ·σ)를 씌우는 분포 보정**(CNN/NN/FCN, 0~120h, CRPS·Brier). 계보상 Rasp & Lerch 2018(신경망+지점임베딩으로 앙상블 평균·표준편차 → 분포 보정)의 단일멤버 변형.
  - *검증 수준: 2022는 저자 최종본 PDF 전문 확인(`chapman2022_한글번역.md`) — "focus remains exclusively on IVT magnitude, not duration/persistence". 2019는 CW3E 공지 수준.*
- **일반 배경**: 앙상블 후처리·MOS(강수·streamflow·바람·가시거리)는 확립된 분야 — "예보를 ML로 보정"이라는 큰 아이디어 자체는 새롭지 않음.
- **우리 차별점 (대상 + 방법 계열 둘 다)**:
  1. **예측 대상 = 24h 지속(duration/persistence)**, IVT 세기 아님. **AR 스케일(Ralph)이 24h/48h 지속으로 등급을 조정**하므로 물리적으로 중요한 별개 축.
  2. **방법 계열 = 분포 보정이 아니라 점 회귀 + 임계 판정.** Chapman/RL2018은 예보에 확률분포를 씌워 CRPS/Brier로 검증(분포 계열). 우리는 지속창 **최소 IVT를 점(point)으로 회귀 예측 → 지역 임계와 비교 → 지속 이진 판정**(F1). 이 "구간 최소값 예측 + 임계수준(threshold-level)" 접근은 수문학 저유량·가뭄 지속 진단(예: 연간 최소 D일 평균 유량, 임계수준법)에서 확립된 것으로, **방법 자체는 우리 발명이 아니라 대기강 지속으로의 이식(transplant)**이다. (수문학 근거를 명시해 "새 방법 발명" 과장을 피하고 리뷰어 반박을 선제 차단.)
  3. **정직한 보정-vs-정보 진단** (ΔAUC≈0 → 이득은 damped 예보 재보정이지 새 랭킹 정보 아님). ⚠️ 이 ΔAUC 수치는 구버전 peak9 기준 — B 기준 재산출 필요(§8).
  4. **운영 분모(TT/TF/FT, 거짓경보 포함)** 명시 구성·평가.
- **novelty 강도 (정직)**: AR에 이 접근을 쓴 선행연구는 다각도 웹검색(AR duration ML, min-IVT threshold, AR persistence 등)에서 **미발견 → "to our knowledge, the first"** 수준으로 서술(웹검색은 부재 증명 불가; 완곡 표현 관례 준수). "방법 최초"는 거짓(수문학 선행) — 최초성은 **대상(AR 지속)으로의 이식**에 한정.
- **포지셔닝 한 줄**: "Chapman et al.의 단일멤버 후처리 계열을 잇되, **분포 보정(IVT 세기) → 점회귀·임계 판정(24h 지속)**으로 옮기고 — 이는 수문학 저유량 지속 진단을 대기강에 이식한 것 — 이득이 정보가 아니라 보정임을 진단한다."
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
- `build_op_denom_full.py` → opdenom_full_{r}{,_18h,_30h}.npz — 3지역 × HORIZONS=[18,24,30], 전 파일에 D2 9 + **D8(고정리드 8, 메인 B)** + fcv/y/omin/oday/THR + **ENV 44 D−2컷** + **y3 3클래스** 동봉
- `colab_opdenom.py` → Colab 셀 (완전분모 3지역 × **24h·30h** raw vs 5모델 F1/AUC/AUPRC + 통합검정) — §8 헤드라인·robustness
- `colab_opdenom_enc.py` → Colab 셀 (완전분모 env 추가 테스트: 인코더 44→8·3클래스 / IVT웨이블릿6 직접) — §6 테스트 2·3
- `colab_d2_tabpfn.py` → Colab 실행 셀 (기준 raw fcv≥THR / 우리모델 TabPFN회귀 예측 min IVT≥THR, F1, 통합검정) — 회귀 설계 완료·실행됨
