# QAE-Parisian-Climate

Robust Quantum Pricing of Path-Dependent Climate Derivatives under Parameter Uncertainty

## Overview

기후 파생상품(특히 대기강 기반)의 가격 책정을 위한 양자 진폭 추정(QAE) 프레임워크.
Parisian 옵션 구조와 파라미터 불확실성을 동시에 다룸.

## Key Ideas

### 1. Parisian Option Structure
- 대기강(AR)은 **지속시간(duration)**이 피해 규모 결정
- IVT ≥ 250 kg/m/s가 **연속 48시간** 이상 → AR 트리거
- 이는 금융의 Parisian barrier option과 동일한 구조

### 2. Stochastic Model
- IVT를 Ornstein-Uhlenbeck 과정으로 모델링
- `dIVT = κ(θ(t) - IVT)dt + σdW`
- 평균 회귀(mean reversion) 특성 반영

### 3. Parameter Uncertainty
- 기후 데이터의 calibration 불확실성 존재
- 파라미터 범위 [κ_min, κ_max] × [σ_min, σ_max]
- **Robust pricing**: 가격의 상한/하한 동시 추정

### 4. Quantum Advantage
- Classical Monte Carlo: O(K × L × 1/ε²)
- QAE with parameter space: O(√(K×L) / ε) (목표)

## Project Structure

```
qae-parisian-climate/
├── src/
│   ├── models/           # Stochastic models (O-U, etc.)
│   ├── quantum/          # QAE circuits and oracles
│   ├── classical/        # Monte Carlo baseline
│   └── data/             # Data processing (IVT, CW3E)
├── notebooks/            # Experiments and analysis
├── tests/                # Unit tests
├── docs/                 # Documentation and paper drafts
└── requirements.txt
```

## Technical Challenges

### Oracle Design
- Parisian 조건 (연속 duration)을 양자 회로로 구현
- 연속 카운트 리셋 로직의 가역성(reversibility) 유지
- 게이트 복잡도: O(T) vs O(log T)?

### State Preparation
- O-U 과정의 비정규 분포 인코딩
- 파라미터 공간의 양자 상태 준비

## Data Sources

| Source | Description |
|--------|-------------|
| [CW3E](https://cw3e.ucsd.edu/) | AR Scale, IVT forecasts |
| ERA5 | Historical reanalysis (1979~) |
| MERRA-2 | NASA reanalysis |

## References

- Parisian Options: Chesney et al. (1997)
- QAE: Brassard et al. (2002)
- AR Scale: Ralph et al. (2019)
- Weather Derivatives O-U: Benth & Šaltytė-Benth (2007)
- Robust Option Pricing: Avellaneda et al. (1995)

## Status

🚧 Research in progress

## License

TBD
