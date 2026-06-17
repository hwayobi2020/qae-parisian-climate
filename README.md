# Atmospheric River Duration Predictability

**대기강(Atmospheric River)이 발생한 시점의 대규모 환경만으로, 그 이벤트가 며칠 지속할지 예측 가능한가?**

> 이 저장소는 원래 양자 진폭추정 기반 기후 파생상품 가격결정 프로젝트였으나, 2026-06-17 위 주제로 전면 전환했다. 이전 양자 단계 자료는 `archive/`에 보존되어 있다.

---

## 왜 지속기간인가

대기강(AR)은 수증기를 대량 수송하는 좁고 긴 대기 흐름으로, 중위도 서해안의 호우·홍수 재해의 주범이다. 핵심은 **지속 시간**이다 — 같은 강도라도 짧게 지나가면 유익한 강수지만, 길게(≥수일) 머무르면 토양이 포화되어 대규모 홍수가 된다.

기존 AR 예보 연구는 **발생·위치·강도**를 며칠 앞서 맞추는 데 집중해왔고(예: DeFlorio et al. 2018; Nayak & Villarini 2014; Ramos et al. NHESS 2020), 개별 AR의 **지속성(duration)이 발생 시점 환경에 의해 얼마나 예측 가능한지는 정량화된 바가 거의 없다.** 본 연구는 이 빈 자리를 다룬다.

## 연구 질문과 정체성

- **질문**: AR이 막 시작된 시점(onset)의 대규모 순환(상층 제트, 블로킹)과 IVT 다중척도 구조만으로, 그 이벤트가 ≥N일 지속할지 예측 가능한가.
- **정체성**: 예보 도구가 아니라 **예측가능성·메커니즘 과학**이다. 수치예보(NWP)는 시간별 IVT 궤적을 적분해 지속시간을 부산물로 얻지만, 본 연구는 "지속성이 onset 시점 환경에 의해 이미 얼마나 조건지어져 있는가"와 그 물리적 원천을 묻는다.

## 방법

- **대상**: AR 이벤트 = 일별 최대 IVT가 그 지역의 **85번째 백분위수**를 연속으로 넘는 구간. onset = 그 시작일.
- **레이블**: 지속기간 ≥ N일 (기준 N=4).
- **특징(20개)**: onset 직전 IVT 16일 wavelet(다중척도 분해) + onset IVT + 상층 제트(U250) onset/64일 wavelet + 블로킹(Z500) 64일 wavelet.
- **모델**: Logistic / Random Forest / TabPFN. 시간순 60/20 홀드아웃, train-only 표준화, test AUC + 부트스트랩 신뢰구간.
- **검증 원칙**: 3지역(캘리포니아·영국·칠레)을 **완전 독립 모델**로 분석하고(풀링 금지), 같은 방법론의 **지역 간 재현성**으로 일반화를 입증한다.

## 현재까지 결과 (캘리포니아 SF)

| 항목 | 값 |
|---|---|
| 본 기준 | AR onset → ≥4일 지속 이진분류 |
| 성능 | test AUC ~0.77 (Logistic ≈ RF ≈ TabPFN, 통계적 동급) |
| 신호원 | IVT 16일 wavelet + U250 상층 제트 |
| 무효 변수 | Z500 블로킹, 해수면온도(SST) |

임계값을 올려 "더 오래 가는 극단 이벤트"로 좁힐수록 AUC가 오르며(≥2일 0.71 → ≥4일 0.77), 이는 *치명적으로 오래 갈 AR일수록 발생 직후에 더 잘 식별된다*는 것을 시사한다.

## 데이터

- **IVT**: ERA5 (`reanalysis-era5-single-levels`, viwve/viwvn → sqrt 합성), 1980-2023, 6시간 간격, 단일 지점.
- **순환지수**: NCEP/NCAR Reanalysis (Z500 500hPa 고도, U250 250hPa 동서바람) OPeNDAP, 지역별 영역.
- **지점**: 캘리포니아 SF(37.7°N), 영국 콘월(50.0°N), 칠레 발파라이소(33.0°S).

## 저장소 구조 (현행)

```
qae-parisian-climate/
├── scripts/
│   ├── download_era5_ivt_uk.py      # 영국 IVT 다운로드
│   ├── download_era5_ivt_chile.py   # 칠레 IVT 다운로드
│   └── build_circ_uk.py             # 영국 북대서양 순환지수
├── threshold_sweep_region.py        # 지역 일반화 임계값 스윕 + 모델 비교 (Colab)
├── data/raw/                        # ERA5 IVT, 순환지수 (gitignore)
├── archive/                         # 이전 양자 단계 소스·문서 보존
└── chat/                            # 세션 기록 (gitignore)
```

## 참고 문헌

- Ralph et al. (2019): Atmospheric River Scale (BAMS)
- DeFlorio et al. (2018): Global Assessment of Atmospheric River Prediction Skill (JHM)
- Nayak & Villarini (2014): Skill of NWP to forecast ARs over the central US (GRL)
- Ramos et al. (2020): Predictive skill for atmospheric rivers in the western Iberian Peninsula (NHESS)
- Guan & Waliser (2015): AR detection with percentile-based IVT thresholds (JGR)

## 진행 상황

| 단계 | 상태 |
|------|------|
| 캘리포니아 onset→지속 예측가능성 | 완료 (AUC ~0.77) |
| 신호원 규명 (ablation) | 완료 (IVT+제트 유효, Z500/SST 무효) |
| 영국·칠레 IVT 다운로드 | 진행 중 |
| 영국·칠레 독립 재현 검증 | 예정 |
| persistence baseline 비교 | 예정 |
| 메커니즘(합성장) 분석 | 예정 |
