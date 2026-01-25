# Paper Outline

## Title (Working)

**Robust Quantum Pricing of Path-Dependent Climate Derivatives under Parameter Uncertainty**

---

## Abstract

Parisian options, where barrier activation depends on the underlying asset remaining beyond a threshold for a *consecutive* duration, are widely used in convertible bonds, credit risk modeling, and emerging climate derivatives. Unlike standard barrier options, their path-dependent structure renders analytical solutions intractable for complex underlying dynamics, forcing practitioners to rely on Monte Carlo simulation with O(1/ε²) sample complexity. Moreover, climate derivatives face significant calibration uncertainty due to limited historical data on extreme events. We propose a Quantum Amplitude Estimation (QAE) framework for robust pricing of Parisian-type options under parameter uncertainty, achieving quadratic speedup in both the sampling and parameter space dimensions. We demonstrate the framework on Atmospheric River (AR) derivatives—a nascent climate risk market where payoffs depend on storm duration exceeding critical thresholds. Numerical experiments validate convergence and show significant advantage over classical Monte Carlo, particularly for rare-event tail probabilities.

---

## 1. Introduction

### 1.1 Motivation
- Climate risk increasing (AR damage exponentially growing)
- Market gap: $5-7B annual losses, 80% uninsured
- Duration matters: AR4-5 are multi-day events (3-11 days)

### 1.2 Problem Statement
- Pricing path-dependent climate derivatives
- Parisian structure needed (duration-based triggers)
- Calibration uncertainty due to rare events

### 1.3 Contributions
1. First QAE framework for Parisian-type options
2. Novel robust pricing under parameter uncertainty
3. Application to climate derivatives (AR)
4. Complexity analysis of Parisian oracle

---

## 2. Background

### 2.1 Parisian Options
- Definition and financial applications
- Pricing methods: Laplace, PDE, Monte Carlo
- Why MC dominates for complex underlyings

### 2.2 Quantum Amplitude Estimation
- QAE algorithm overview
- Quadratic speedup: O(1/ε) vs O(1/ε²)
- Existing applications in finance

### 2.3 Atmospheric Rivers
- AR scale (IVT + duration)
- Economic impacts
- Why Parisian structure is natural

---

## 3. Model Framework

### 3.1 IVT Dynamics
- Ornstein-Uhlenbeck process
- dIVT = κ(θ(t) - IVT)dt + σdW
- Seasonal mean θ(t)

### 3.2 Parisian Payoff
- Trigger: IVT ≥ barrier for consecutive duration ≥ window
- Connection to AR scale categories

### 3.3 Parameter Uncertainty
- Calibration challenges for climate data
- Parameter ranges: [κ_min, κ_max] × [σ_min, σ_max]
- Robust pricing: worst/best case bounds

---

## 4. Quantum Algorithm

### 4.1 State Preparation
- O-U distribution encoding
- Parameter space superposition

### 4.2 Parisian Oracle Design ⭐ (Technical Core)
- Consecutive counter in quantum state
- Reversible reset logic
- Gate complexity: O(T) vs O(log T)?
- Uncomputation strategy

### 4.3 Robust QAE
- Joint amplitude estimation over parameter space
- Extracting price bounds

### 4.4 Complexity Analysis
- Classical: O(K × L × T / ε²)
- Quantum: O(√(K×L) × T / ε) or better?

---

## 5. Experiments

### 5.1 Setup
- Synthetic O-U paths
- Historical IVT data (CW3E/ERA5)
- Parameter ranges from literature

### 5.2 Convergence Analysis
- QAE vs MC sample complexity
- Error scaling with ε

### 5.3 Rare Event Performance
- AR4-5 trigger probabilities (p < 0.01)
- Advantage amplification for rare events

### 5.4 Robust Pricing
- Price bounds under uncertainty
- Comparison: grid search vs QAE

---

## 6. Discussion

### 6.1 Practical Implications
- Pricing previously unhedgeable risks
- Risk management under uncertainty

### 6.2 Limitations
- NISQ constraints
- State preparation costs
- Oracle implementation challenges

### 6.3 Future Work
- Real quantum hardware experiments
- Multi-location derivatives
- Integration with climate models

---

## 7. Conclusion

---

## Key Technical Challenges (For Implementation)

1. **Oracle Gate Complexity**
   - Can we achieve O(log T) for Parisian counter?
   - Or prove O(T) is optimal?

2. **Reversible Reset**
   - Counter reset when IVT < barrier
   - Must maintain reversibility for QAE

3. **Parameter Superposition**
   - Encode κ, σ ranges in qubits
   - Joint state preparation

---

## Target Venues

- **Q1 Journals**: Quantum, npj Quantum Information, Nature Communications
- **Finance + CS**: Quantitative Finance, Journal of Computational Finance
- **Interdisciplinary**: Scientific Reports, PLOS ONE

---

## Timeline (TBD)

| Phase | Task | Status |
|-------|------|--------|
| 1 | Literature review | In progress |
| 2 | Classical baseline | Started |
| 3 | Oracle design | TODO |
| 4 | QAE implementation | TODO |
| 5 | Experiments | TODO |
| 6 | Writing | TODO |
