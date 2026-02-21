# QAE-Parisian-Climate

Importance-Sampled Quantum Amplitude Estimation for Parisian Climate Derivatives

## Overview

Atmospheric River (AR) insurance pricing using four progressively efficient methods:

**Naive MC → IS-MC (Glasserman) → QAE → IS-QAE**

The core contribution is the first quantum oracle for **Parisian conditions** (consecutive-duration barriers), combined with importance sampling for rare-event acceleration.

## Key Results

### Parisian Oracle (verified)
- Borrow-chain comparator, reversible consecutive counter, OR-latch
- Bennett's pebble game: **340 qubits** (vs 1,543 naive) for T=192 steps
- Qiskit Aer verification: 50.62% vs expected 50.00%

### JD-OU Model (calibrated to ERA5)
- 44 years of IVT data (1980-2023), SF coast, 6-hourly
- Jump-Diffusion O-U: kappa=0.767, theta=85.8, sigma=62
- AR events: lambda=0.108/day (39.5/yr), mu_peak=390.5, tau=0.69 days

### Complexity

| Method | Qubits | Cost |
|--------|--------|------|
| Naive MC | 0 | O(T / epsilon^2) |
| IS-MC | 0 | O(T * sigma_IS^2 / epsilon^2) |
| QAE | O(sqrt(T)*n + T) | O(T^{3/2} / epsilon) |
| IS-QAE | O(sqrt(T)*n + T) | O(T^{3/2} * sigma_IS / epsilon) |

## Project Structure

```
qae-parisian-climate/
├── src/
│   ├── models/
│   │   ├── ornstein_uhlenbeck.py   # Base O-U process
│   │   ├── jump_diffusion_ou.py    # JD-OU (two-regime AR model)
│   │   └── parisian.py             # Parisian condition checker + MC pricing
│   ├── quantum/
│   │   └── parisian_oracle.py      # Quantum oracle + pebble game + complexity
│   └── classical/
│       └── robust_mc.py            # Robust MC over parameter ranges
├── data/
│   └── raw/                        # ERA5 IVT NetCDF files (1980-2023)
├── docs/
│   └── paper_outline.md            # Full paper structure
├── notebooks/
├── tests/
└── requirements.txt
```

## Data

- **Source**: ERA5 CDS (Copernicus Climate Data Store)
- **Variables**: viwve, viwvn (eastward/northward vapor transport)
- **Region**: California coast [40N-34N, 130W-120W]
- **Period**: 1980-2023, 6-hourly (64,284 observations at SF point)
- **IVT**: sqrt(viwve^2 + viwvn^2), threshold >= 250 kg/m/s for AR

## References

- Chesney et al. (1997): Parisian barrier options
- Glasserman (2003): Monte Carlo Methods in Financial Engineering
- Stamatopoulos et al. (2020): Option pricing using quantum computers
- Bennett (1989): Time/space tradeoffs for reversible computation
- Merton (1976): Jump-diffusion option pricing
- Ralph et al. (2019): AR Scale

## Status

| Phase | Status |
|-------|--------|
| ERA5 data acquisition | DONE |
| JD-OU calibration | DONE |
| Parisian oracle design | DONE |
| Oracle verification (Qiskit) | DONE |
| Glasserman IS | TODO |
| IS-QAE state preparation | TODO |
| Numerical experiments | TODO |
| Paper writing | TODO |
