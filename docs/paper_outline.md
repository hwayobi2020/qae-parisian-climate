# Paper Outline

## Title (Working)

**Importance-Sampled Quantum Amplitude Estimation for Parisian Climate Derivatives**

---

## Abstract

Parisian options require the underlying to remain beyond a barrier for a *consecutive* duration, making them natural instruments for climate risk (e.g., Atmospheric River insurance where sustained storm conditions determine payouts). Classical Monte Carlo pricing scales as O(1/epsilon^2), which becomes expensive for rare-event tail probabilities. We present four progressively efficient pricing methods: (1) naive MC, (2) importance-sampled MC using Glasserman's exponential tilting, (3) Quantum Amplitude Estimation (QAE), and (4) IS-QAE combining both. Our key technical contribution is the first quantum oracle for Parisian (consecutive-duration) conditions, featuring a reversible consecutive counter with conditional reset, an OR-latch for first-trigger detection, and Bennett's pebble game for qubit reduction from O(T*n) to O(sqrt(T)*n + T). We calibrate a Jump-Diffusion Ornstein-Uhlenbeck model to 44 years of ERA5 reanalysis data and demonstrate that IS-QAE achieves the best asymptotic complexity, combining variance reduction with quadratic speedup.

---

## 1. Introduction

### 1.1 Motivation
- Climate risk: AR damage exponentially growing, $5-7B annual losses, 80% uninsured
- Duration matters: AR4-5 are multi-day events — Parisian structure is natural
- Pricing challenge: path-dependent + rare events + calibration uncertainty

### 1.2 Problem Statement
- Price Parisian-type climate derivatives efficiently
- Handle rare-event probabilities (trigger rates ~1-5%)
- Compare classical and quantum approaches fairly

### 1.3 Contributions
1. First quantum oracle for Parisian (consecutive-duration) conditions
2. IS-QAE framework combining Glasserman importance sampling with QAE
3. JD-OU model calibrated to 44 years ERA5 IVT data
4. Complete complexity analysis: naive MC → IS-MC → QAE → IS-QAE

---

## 2. Background

### 2.1 Parisian Options
- Definition: barrier + consecutive duration window
- Financial applications (convertible bonds, credit risk)
- Pricing methods: Laplace (Chesney 1997), PDE, Monte Carlo

### 2.2 Atmospheric Rivers & Climate Derivatives
- AR scale (IVT >= 250 kg/m/s + duration)
- Economic impacts (California flood risk)
- Why Parisian structure maps to AR insurance

### 2.3 Quantum Amplitude Estimation
- QAE algorithm: O(1/epsilon) vs MC's O(1/epsilon^2)
- Existing quantum finance: Stamatopoulos (2020), Rebentrost (2018)
- Gap: no path-dependent (Parisian) oracle exists

### 2.4 Importance Sampling for Rare Events
- Glasserman (2003): exponential tilting, optimal change of measure
- Variance reduction for rare-event MC
- Connection to quantum state preparation

---

## 3. Climate Model

### 3.1 IVT Dynamics: Jump-Diffusion O-U
- Normal regime: dIVT = kappa(theta - IVT)dt + sigma*dW
- AR events: Poisson(lambda) arrivals, peak ~ N(mu_peak, sigma_peak), duration ~ Exp(tau)
- Two-regime model: theta shifts during AR events

### 3.2 Calibration to ERA5 Data
- 44 years (1980-2023), 6-hourly, SF coast (37.75N, 122.5W)
- Parameters: kappa=0.767, theta=85.8, sigma=62, lambda=0.108/day
- Validation: MC trigger probabilities vs historical rates

### 3.3 Parisian Condition for AR Insurance
- Barrier B (IVT threshold), window W (consecutive duration)
- Contract: payout if IVT >= B for consecutive W hours during season
- Pricing = P(at least one trigger during contract period)

---

## 4. Classical Pricing Methods

### 4.1 Naive Monte Carlo
- Simulate JD-OU paths, check Parisian condition
- Complexity: O(T / epsilon^2)
- Baseline results for various (B, W) combinations

### 4.2 Importance-Sampled Monte Carlo (Glasserman)
- Exponential tilting: bias JD-OU toward barrier region
- Optimal tilt parameter selection
- Variance reduction ratio: sigma^2 / sigma^2_IS
- Complexity: O(T * sigma^2_IS / epsilon^2)

### 4.3 Markov Chain Formulation
- State: (IVT_level, consecutive_count) — Markov
- Transition matrix from JD-OU discretization
- Exact solution via absorption probability (validation)

---

## 5. Quantum Pricing Algorithm

### 5.1 Parisian Oracle Design (Technical Core)
- **Borrow-chain comparator**: IVT >= barrier, O(n) Toffoli gates
- **Reversible consecutive counter**: increment if above, reset if below
- **OR-latch**: first-trigger detection (trigger = cond AND NOT flag)
- **Bennett's pebble game**: qubits O(T*n) → O(sqrt(T)*n + T)

### 5.2 Importance-Sampled State Preparation
- Tilted distribution |psi_IS> encodes biased paths
- Controlled rotations for exponential tilting
- Likelihood ratio incorporated in oracle

### 5.3 IS-QAE Integration
- QAE with IS-prepared states
- Grover iterations on biased distribution
- Combined complexity: O(T^{3/2} * sigma_IS / epsilon)

### 5.4 Complexity Analysis
| Method | Qubits | Gates | Accuracy |
|--------|--------|-------|----------|
| Naive MC | 0 | O(T/epsilon^2) | O(sigma/sqrt(N)) |
| IS-MC | 0 | O(T*sigma^2_IS/epsilon^2) | O(sigma_IS/sqrt(N)) |
| QAE | O(sqrt(T)*n+T) | O(T^{3/2}/epsilon) | O(1/N) |
| IS-QAE | O(sqrt(T)*n+T) | O(T^{3/2}*sigma_IS/epsilon) | O(1/N) |

---

## 6. Numerical Experiments

### 6.1 Setup
- ERA5 calibrated JD-OU parameters
- Barrier/window configurations: (250,24h), (250,48h), (500,24h), (500,48h)
- Qiskit Aer verification of oracle (small-scale)

### 6.2 Oracle Verification
- 18-qubit demo: 50.62% vs expected 50% (validated)
- Comparator: all thresholds verified
- Counter + OR-latch: correct behavior confirmed

### 6.3 Variance Reduction Analysis
- IS variance reduction ratios for each (B,W)
- Rare events (B=500, W=48h): expected large reduction

### 6.4 Complexity Comparison
- Concrete gate/qubit counts for realistic parameters
- Crossover analysis: when does IS-QAE beat IS-MC?

---

## 7. Discussion

### 7.1 Practical Implications
- Near-term: IS-MC is best practical method
- Long-term: IS-QAE with fault-tolerant hardware

### 7.2 Limitations (Honest)
- IS-QAE cannot run on current hardware for realistic sizes
- Quadratic speedup may not overcome constant-factor overhead
- Climate derivative market is nascent

### 7.3 Future Work
- Multi-variate Parisian (IVT + soil moisture + wind)
- Real quantum hardware experiments (small-scale)
- Extension to other path-dependent climate payoffs

---

## 8. Conclusion

---

## Target Venues
- **Primary**: Quantum Science and Technology (Q1, IF ~5.6)
- **Alternative**: EPJ Quantum Technology, IEEE Trans. Quantum Engineering
- **Reach**: PRX Quantum (needs deeper theoretical contribution)

## Timeline

| Phase | Task | Status |
|-------|------|--------|
| 1 | ERA5 data acquisition | DONE |
| 2 | JD-OU calibration | DONE |
| 3 | Parisian oracle design | DONE |
| 4 | Oracle verification (Qiskit) | DONE |
| 5 | Glasserman IS implementation | TODO |
| 6 | IS-QAE state preparation | TODO |
| 7 | Numerical experiments | TODO |
| 8 | Paper writing | TODO |
