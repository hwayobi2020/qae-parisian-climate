# [ARCHIVED 2026-06-17] 양자 단계 CLAUDE.md (Model-Free Quantum Pricing)

> 이 문서는 이 프로젝트의 **이전 정체성**(양자 진폭추정 기반 경로의존 기후 파생상품 가격결정)의 작업 지침이다.
> 2026-06-17 프로젝트가 대기강(atmospheric river) 지속기간 예측가능성 연구로 **전면 피벗**하면서
> 보존용으로 archive에 옮겼다. 현재 작업 지침은 루트 CLAUDE.md를 참조.
> 양자 단계 상세 결과/세션은 chat/session_2026-04-*.{jsonl,md} 에 있다.

---

# QAE-Parisian-Climate

## 현재 작업
논문 작성 — **Model-Free Quantum Pricing of Path-Dependent Climate Derivatives**

## 확정 아키텍처 (2026-04-01 확정)

### Qubit Path Encoding + Controlled State Prep + Grover AE
- **Cavity에서 qubit으로 전환**: cavity Fock M^(T+1) → qubit (T+1)×log₂(M)
- Path register: |t₀, t₁, ..., t_T⟩, 각 t_k ∈ {0,...,M-1}
- Transition: controlled state prep (매 step, conditioned on previous step)
- Barrier oracle: per-path phase flip (Parisian consecutive window W 지원)
- Grover AE: O(1/ε) precision

### 핵심: Model-Free
- Transition matrix P(j|i)를 **ERA5 data에서 직접 추출** (histogram counting)
- 모델 가정 불필요 (OU, GBM 등 필요 없음)
- 기존 양자 금융 논문은 전부 parametric model (GBM/Heston) 기반
- Model risk 정량화: OU vs Empirical → **52-59% 가격 차이**

### 왜 Path Encoding인가
Compact encoding (같은 register 재활용)은 모두 실패:
- 핵심 이유: barrier check가 경로(path) 수준 정보를 요구
- amplitude² ≠ probability (√P encoding 문제)
- Path encoding: 각 경로가 고유한 basis state → barrier = per-state property → 정확
- 10가지 이상의 compact encoding 시도 후 확정 (상세: 아래 실패 기록)

## 검증 완료 결과 (2026-04-01)

### OU CDF transition (barrier-aligned grid)
| M | T | P_ki quantum | MC 오차 | CQ match | Grover | qubits |
|---|---|---|---|---|---|---|
| 16 | 1 | 0.00389 | 6.1% | 9.1e-17 | OK | 8 |
| 16 | 2 | 0.00870 | 2.4% | 3.6e-17 | OK | 12 |
| 64 | 1 | 0.00410 | 1.2% | 8.6e-17 | OK | 12 |
| 64 | 2 | 0.00894 | 0.2% | - | OK | 18 |

### Empirical transition (model-free, ERA5)
| T | P_ki quantum | MC 오차 | CQ match | Grover |
|---|---|---|---|---|
| 1 | 0.01054 | **0.15%** | 2.4e-17 | OK |
| 2 | 0.03060 | **0.29%** | 1.0e-17 | OK |
| 4 | 0.07314 | **0.09%** | 8.3e-17 | OK |

### Parisian barrier (consecutive window W)
| T | W | P_ki quantum | Grover |
|---|---|---|---|
| 2 | 1 | 0.01303 | OK |
| 2 | 2 | 0.00021 | OK |
| 4 | 2 | 0.00091 | OK |
| 4 | 3 | 0.00002 | OK |

### MLE Amplitude Estimation
- M=16, T=2: 7,000 queries → 0.08% precision
- Classical MC 동등 정밀도: ~700,000 samples → **100x speedup**
- Roundtrip: 4.7×10⁻¹⁵

## 리소스 추정

| 시나리오 | M | T | Qubits | Quantum gates | MC ops | Speedup |
|----------|---|---|--------|---------------|--------|---------|
| Monthly Parisian W=1 | 8 | 4 | 19 | 960K | 10.7M | 11x |
| Monthly Parisian W=2 | 8 | 4 | 19 | 4.8M | 320M | **67x** |
| Quarterly W=1 | 8 | 13 | 47 | 1.9M | 10.4M | 5.6x |
| Annual W=1 | 8 | 52 | 166 | 5.0M | 13.9M | 2.8x |

## 논문 Contribution
1. **Model-free quantum pricing**: empirical transition으로 직접 pricing. 양자 금융에서 최초.
2. **기후 파생상품 양자 가격 산정**: Parisian barrier on climate data. 최초 적용.
3. **Model risk 정량화**: parametric vs empirical → 52-59% 차이. 모델 선택의 중요성.
4. **실증 검증**: ERA5 44년 데이터, CQ match 10⁻¹⁶, Grover 완벽 동작.

## Compact Encoding 실패 기록 (10+ 시도)

| # | 방법 | 실패 원인 |
|---|------|-----------|
| 1 | Flag subspace (10종) | column 겹침 (flag=0→1 output ∩ flag=1 identity) |
| 2 | SWAP flag | 양방향 swap → un-knock |
| 3 | On-the-fly PRNG | A† 불가 (noise uncompute 안 됨) |
| 4 | Scratch register | dirty 72% (discretized contraction many-to-one) |
| 5 | P_safe^T isometry | classical 전처리 의존 |
| 6 | HO + bath | 파라미터 부족 |
| 7 | Schrodingerisation | tail probability O(1) 비수렴 |
| 8 | QSVT block-encode | σ_max^T 지수적 감쇠 |
| 9 | Szegedy Walk | non-reversible P → Chebyshev 불성립 |
| 10 | Hermite + counter | amplitude² ≠ probability mismatch |
| **공통 원인** | **barrier check가 경로 수준 정보 요구 → compact encoding 불가** |

## 부가 발견: Hermite Basis
- OU FP 고유함수 = Hermite 다항식 → N=16에서 10⁻¹³ 정확도 (FD 대비 exponential)
- 하지만 multi-step barrier check에서 amplitude/probability mismatch로 사용 불가
- Single-step European에서만 완벽 동작 (논문에서 이론적 분석으로 언급 가능)

## 핵심 파일
- `data/empirical_transition_M8_dt7.npy` — ERA5 empirical transition
- `paper_draft_kr.md` — 논문 초고 (재작성 필요)
- `research_progress_20260328.md` — 진행 보고 (cavity 시절)
- `chat.txt` — 전체 대화 기록

## 주의사항
- Classical shortcut 절대 금지
- 완료 여부 주장 전 반드시 코드 실행으로 검증
- Classical 부분 있으면 즉시 밝힐 것
- 수치 주장 시 단위 확인 (percentage vs fraction)
- 실패 결과 숨기지 말 것
