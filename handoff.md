# Handoff — AR 지속기간 예측가능성 (2026-06-27)

> 이 프로젝트는 대기강(Atmospheric River, AR) **지속기간 예측가능성** 연구다.
> 양자컴(QAE) 단계는 종료·보존: `archive/`, `archive/CLAUDE_quantum_legacy.md`, 구 handoff 내용은 `chat/session_2026-04-*`.

## 0. 한 줄 상태

dmax[o] 데이터누수 제거 후, **onset 시점 종관환경(지점 해면기압 + IVT 수송방향)**으로 leak-free held-out AUC **0.66~0.74** 달성, **3지역(CA/UK/Chile) 독립 재현**. 누수 검증 완료. TabPFN(Colab) 확증 + 메커니즘 분석이 다음.

## 1. 확정 사항 (변경 금지)

### 1.1 데이터 누수 (해결됨)
- 기존 강도/base 피처 `dmax[o] = ivt.reshape(-1,4).max(1)[o]` (o=s//4)는 onset 당일 일별최대 → 시간축 00Z 시작(확정)이라 onset 이후 같은날 **0~18h를 엿봄**(intra-day look-ahead). onset 시각에 따라 누수량 들쭉(18Z=0h, 00Z=18h).
- 기존 AUC 0.77~0.81은 leak-inflated. **causal 강도 = `max(ivt[s-3:s+1])`** (onset에서 끝나는 직전 24h, 미래 無).
- **교훈**: "leak 빼니 onset 예측 ≈ null(0.51~0.55)"은 **성급한 negative였음** — intensity+circulation만 봤고 pressure/direction을 안 봤다.

### 1.2 핵심 결과 — 피처셋 E (held-out 5-fold OOF, paired bootstrap)
- A=강도(causal), B=+IVT 16d wavelet, C=+순환(제트/블로킹), D=full(A+B+C), **E=D+신규**.
- 신규 = **해면기압(pressure_msl)+24h추세, IVT 방향(viwve/viwvn→sin/cos), 운량, 다중스케일 IVT(48h~7d max/mean/std), 직전 AR활동(30/60d)**.

| 지역 | 24h | 36h | 48h | E vs D |
|---|---|---|---|---|
| CA | 0.60 | 0.63 | 0.59 | 24/36h *유의*, 48h 비유의(n=73) |
| UK | 0.67 | 0.66 | 0.66 | 전구간 +0.09~0.14 *유의* (D≈0.50→구제) |
| Chile | 0.74 | 0.73 | 0.73 | 전구간 +0.10~0.13 *유의* |

### 1.3 누수 검증 (완료)
- openmeteo ↔ IVT 시간정렬 3지역 불일치 **0건**.
- 점프 분해(36h): **기압**(UK+0.078, Chile+0.110)과 **IVT 방향**(UK+0.071, Chile+0.050)이 주동력. 둘 다 onset 종관상태(≤s).
- **직전 AR활동**(누수 의심했던 것)은 기여 ~0, 단독 AUC 0.47~0.54 → 누수 아님, 그냥 약함.
- 약함→E에서 제외: MJO, QBO, ENSO, SST, 제트 위도(jet_lat은 제트 강도보다도 약함).

## 2. 한계 (정직하게)

- **기압은 onset 동시상태(nowcast) 피처** — 미래는 안 보지만 "사전 전조"가 아님. 논문에 명시 요.
- 48h 양성 작음(CA 73 / UK 32 / Chile 152) → CA 48h 비유의. 핵심 근거는 n 큰 24·36h.
- 현재 **Logistic/LGBM held-out**만. TabPFN(Colab) 미확정.

## 3. 다음 작업 (우선순위)

1. **TabPFN 확증** — Colab에서 `cv_causal_6h_region.py`를 TabPFN 포함 실행. 데이터는 `data/raw/efeat_{region}.npz`(사전계산, force-add)로 전달 → 큰 raw 불필요.
2. **메커니즘 해석** — 기압/IVT방향이 어떤 종관패턴(저기압 깊이·수송 방위)으로 지속을 가르는지.
3. **persistence/기후 baseline 대비 skill** (미착수).
4. 출판: Q2 SCI (J. Hydrometeorology / NHESS / Int. J. Climatology).

## 4. 핵심 파일

| 경로 | 역할 |
|---|---|
| `cv_causal_6h_region.py [ca\|uk\|chile]` | A/B/C/D/E 피처셋 leak-free held-out CV. TabPFN 선택적 import(로컬=Logistic/LGBM). efeat npz 우선 로드 |
| `screen_features_region.py [region\|all]` | train-only 기초통계 후보 선별 (AUC/MI/point-biserial) |
| `build_efeat_npz.py [region\|all]` | E 신규피처(pmsl/cloud/dir_sin/dir_cos)를 data/raw/efeat_{region}.npz로 사전계산 |
| `data/raw/efeat_{ca,uk,chile}.npz` | Colab 전달용 사전계산 피처 (force-add) |

## 5. 데이터 정렬 (확정)
- 6h: ivt_{sf,uk,chile}_1980_2023.npy, times_*, openmeteo_*.npz, enso/sst → 길이 64284, 00Z 시작 [00,06,12,18], 인덱스 s 직접.
- 일별: circ_indices*.npz(jet/blocking/dates), 격자 nc → 길이 16071, o = s//4. causal은 ≤ o-1.
- 강수(openmeteo)는 "직전 1h 누적"이라 precip[s]도 causal.
