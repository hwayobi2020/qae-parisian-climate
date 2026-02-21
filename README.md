# QAE-Parisian-Climate

**중요도 샘플링 기반 양자 진폭 추정을 활용한 파리지앵 기후 파생상품 가격 결정**

Importance-Sampled Quantum Amplitude Estimation for Parisian Climate Derivatives

---

## 왜 이 연구가 필요한가

### 대기강(Atmospheric River)과 기후 재해

캘리포니아는 연간 강수량의 30-50%를 대기강(AR)에서 얻는다. AR은 수증기를 대량으로 수송하는 좁고 긴 대기 흐름으로, 적당한 AR은 수자원 공급에 필수적이지만, 강한 AR이 **장시간 지속**되면 대규모 홍수와 산사태를 유발한다.

핵심은 **지속 시간**이다:
- AR이 12시간 지나가면: 유익한 강수
- AR이 48시간 이상 지속되면: 재앙적 홍수 (2017년 Oroville 댐 위기, 2023년 캘리포니아 대홍수)

AR로 인한 연간 피해액은 $5-7B (약 7-10조원)이며, 이 중 80%가 보험으로 커버되지 않는다. 기후변화로 AR의 강도와 빈도가 증가하면서 이 gap은 더 벌어지고 있다.

### 기존 보험 상품의 한계

전통적 기상 파생상품(weather derivatives)은 단순 조건에 기반한다:
- "강수량이 X mm 초과" (단일 시점 조건)
- "기온이 Y도 이하인 날이 Z일 이상" (누적 조건)

하지만 AR 재해의 핵심 메커니즘은 **연속 지속 시간**이다. IVT(수증기 수송량)가 높은 상태가 **끊기지 않고** 48시간 이상 유지되어야 토양이 포화되고, 유출이 누적되어 홍수가 발생한다. 중간에 잠깐이라도 IVT가 내려가면 피해 규모가 급격히 줄어든다.

이 "연속 지속" 조건은 금융공학에서 **파리지앵 옵션(Parisian option)**이라고 불리는 구조와 정확히 일치한다:

> **파리지앵 조건**: 기초자산이 배리어를 **연속으로** 일정 기간 이상 초과해야 트리거

### 가격 결정의 어려움

파리지앵 조건은 경로 의존적(path-dependent)이어서 해석적 해가 존재하지 않는다. 실무에서는 몬테카를로(MC) 시뮬레이션에 의존하는데:

1. **MC는 느리다**: 정확도 epsilon을 위해 O(1/epsilon^2)개의 경로가 필요
2. **희귀 사건 문제**: AR 트리거 확률이 1-5% 수준으로, 대부분의 시뮬레이션 경로가 "낭비"
3. **보험 상품 설계**: 다양한 (배리어, 윈도우) 조합에 대해 반복 계산 필요

### 본 연구의 접근

네 가지 방법을 점진적으로 개선하며 비교한다:

```
Naive MC  →  IS-MC (Glasserman)  →  QAE  →  IS-QAE
 (기본)     (분산 감소)          (양자 가속)  (분산 감소 + 양자 가속)
```

- **Importance Sampling (IS)**: Glasserman의 exponential tilting으로 희귀 사건 방향으로 경로를 편향시켜 분산을 줄임
- **Quantum Amplitude Estimation (QAE)**: MC의 O(1/epsilon^2)를 O(1/epsilon)으로 개선하는 양자 알고리즘
- **IS-QAE**: 두 개선이 곱해져서 최상의 점근적 복잡도 달성

핵심 기술적 기여는 **파리지앵 조건(연속 지속 시간)을 양자 회로로 구현한 최초의 oracle 설계**이다.

---

## 주요 결과

### 파리지앵 오라클 (검증 완료)
- Borrow-chain 비교기, 가역적 연속 카운터, OR-래치
- Bennett's pebble game 적용: **340 큐빗** (naive 1,543 큐빗 대비 4.5배 감소)
- Qiskit Aer 시뮬레이션 검증: 50.62% (이론값 50.00%)

### JD-OU 모델 (ERA5 캘리브레이션)
- 44년 IVT 데이터 (1980-2023), 샌프란시스코 해안, 6시간 간격
- Jump-Diffusion Ornstein-Uhlenbeck: 정상 레짐 + AR 이벤트 2개 레짐 모델
- 캘리브레이션 결과: kappa=0.767, theta=85.8, sigma=62, lambda=0.108/day (연 39.5회 AR)

### Glasserman IS (검증 완료)
- 3성분 exponential tilting: 확산(Girsanov) + 점프율(Poisson) + 점프지속(Exponential)
- Jump-only tilting이 최적 (확산 tilt는 가중치 퇴화 유발)
- 분산 감소: 1.2x (일반 이벤트) ~ 6.5x (희귀 이벤트), 희귀도에 비례

### QAE 수렴 (검증 완료)
- 18큐빗 파리지앵 오라클에서 IQAE (Iterative QAE) 실행
- MC vs IQAE 같은 호출 횟수에서 IQAE가 2-25배 정밀
- O(1/sqrt(N)) vs O(1/N) 수렴 속도 확인

### 파라미터 불확실성 (핵심 동기)
- Bootstrap으로 ERA5에서 K개 파라미터셋 생성
- 고전: K × O(1/ε²) → 양자: K × O(1/ε)
- K=30,000, ε=0.001 기준: MC 25분 vs QAE 24초 (64x speedup)

### 복잡도 비교

| 방법 | 큐빗 | 계산 비용 (K 파라미터셋) |
|------|-------|-----------|
| Naive MC | 0 | K × O(T / epsilon^2) |
| IS-MC (Glasserman) | 0 | K × O(T * sigma_IS^2 / epsilon^2) |
| QAE | O(sqrt(T)*n + T) | K × O(T^{3/2} / epsilon) |
| **IS-QAE** | O(sqrt(T)*n + T) | **K × O(T^{3/2} * sigma_IS / epsilon)** |

---

## 프로젝트 구조

```
qae-parisian-climate/
├── src/
│   ├── models/
│   │   ├── ornstein_uhlenbeck.py   # 기본 O-U 과정
│   │   ├── jump_diffusion_ou.py    # JD-OU (2레짐 AR 모델)
│   │   └── parisian.py             # 파리지앵 조건 판별 + MC 가격결정
│   ├── quantum/
│   │   ├── parisian_oracle.py      # 양자 오라클 + pebble game + 복잡도 분석
│   │   └── qae_convergence_demo.py # IQAE 수렴 데모 (MC vs QAE 호출횟수 비교)
│   ├── classical/
│   │   ├── glasserman_is.py        # Glasserman IS (3성분 exponential tilting)
│   │   ├── markov_chain.py         # 마르코프 체인 exact solver
│   │   ├── parameter_sweep.py      # 파라미터 불확실성 분석 (bootstrap + 배치 실행)
│   │   └── robust_mc.py            # 파라미터 범위에 대한 robust MC
│   └── generate_paper_figures.py   # 논문 figure 생성기
├── figures/                        # 생성된 논문 figure
├── data/
│   └── raw/                        # ERA5 IVT NetCDF 파일 (1980-2023)
├── docs/
│   └── paper_outline.md            # 논문 구조
├── notebooks/
├── tests/
└── requirements.txt
```

## 데이터

- **출처**: ERA5 CDS (Copernicus Climate Data Store)
- **변수**: viwve, viwvn (동서/남북 수증기 수송량)
- **지역**: 캘리포니아 해안 [40N-34N, 130W-120W]
- **기간**: 1980-2023, 6시간 간격 (SF 지점 64,284개 관측)
- **IVT**: sqrt(viwve^2 + viwvn^2), AR 판별 임계값 >= 250 kg/m/s

## 참고 문헌

- Chesney et al. (1997): Parisian barrier options
- Glasserman (2003): Monte Carlo Methods in Financial Engineering
- Stamatopoulos et al. (2020): Option pricing using quantum computers
- Bennett (1989): Time/space tradeoffs for reversible computation
- Merton (1976): Jump-diffusion option pricing
- Ralph et al. (2019): Atmospheric River Scale

## 진행 상황

| 단계 | 상태 |
|------|------|
| ERA5 데이터 수집 | 완료 |
| JD-OU 캘리브레이션 | 완료 |
| 파리지앵 오라클 설계 | 완료 |
| 오라클 검증 (Qiskit Aer) | 완료 |
| Glasserman IS 구현 | 완료 |
| 마르코프 체인 검증 | 완료 |
| QAE 수렴 데모 (IQAE) | 완료 |
| 파라미터 불확실성 분석 | 완료 |
| 논문 figure 생성 | 완료 |
| 논문 작성 | 예정 |
