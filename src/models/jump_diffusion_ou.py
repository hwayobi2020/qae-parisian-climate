"""
Jump-Diffusion Ornstein-Uhlenbeck process for IVT modeling

Two-regime model:
    Normal:  dIVT = κ(θ - IVT)dt + σdW
    AR event: IVT jumps to elevated level, mean-reverts to θ_AR with κ_AR

AR events arrive as Poisson(λ), with peak IVT ~ N(μ_peak, σ_peak)
and duration ~ Exp(1/τ). During AR, mean reversion target shifts up.

References:
    - Merton (1976): Jump-diffusion option pricing
    - Cartea & Figueroa (2005): Energy price spikes via jump-diffusion
    - Weron (2008): Regime-switching for electricity prices
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class JDOUParameters:
    """Parameters for Jump-Diffusion O-U process"""
    # Normal regime
    kappa: float        # Mean reversion speed (/day)
    theta: float        # Long-term mean IVT (kg/m/s)
    sigma: float        # Diffusion volatility

    # AR events (jump component)
    lam: float          # AR arrival rate (events/day)
    mu_peak: float      # Mean peak IVT during AR (kg/m/s)
    sigma_peak: float   # Peak IVT std (kg/m/s)
    tau: float          # Mean AR duration (days)


def simulate_jdou_paths(
    n_paths: int,
    n_steps: int,
    dt: float,
    params: JDOUParameters,
    ivt_init: float = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate IVT paths using Jump-Diffusion O-U process

    When an AR event arrives:
    - IVT jumps toward peak level
    - Mean reversion temporarily targets elevated theta_AR
    - After AR duration expires, reverts back to normal theta

    Args:
        n_paths: Number of Monte Carlo paths
        n_steps: Number of time steps
        dt: Time step size (in days, 0.25 = 6 hours)
        params: JD-OU parameters
        ivt_init: Initial IVT value (defaults to theta)
        seed: Random seed

    Returns:
        Array of shape (n_paths, n_steps) with IVT values
    """
    if seed is not None:
        np.random.seed(seed)

    if ivt_init is None:
        ivt_init = params.theta

    ivt = np.zeros((n_paths, n_steps))
    ivt[:, 0] = ivt_init

    # Track AR state per path: remaining AR duration
    ar_remaining = np.zeros(n_paths)  # days remaining in AR
    ar_target = np.zeros(n_paths)     # target IVT during AR

    for i in range(1, n_steps):
        # Check for new AR arrivals (only if not already in AR)
        not_in_ar = ar_remaining <= 0
        new_ar = not_in_ar & (np.random.random(n_paths) < params.lam * dt)

        if new_ar.any():
            n_new = new_ar.sum()
            ar_remaining[new_ar] = np.random.exponential(params.tau, n_new)
            ar_target[new_ar] = np.random.normal(params.mu_peak, params.sigma_peak, n_new)
            ar_target[new_ar] = np.maximum(ar_target[new_ar], params.theta)

        # Determine effective theta and kappa per path
        in_ar = ar_remaining > 0
        theta_eff = np.where(in_ar, ar_target, params.theta)

        # O-U step
        dW = np.sqrt(dt) * np.random.randn(n_paths)
        drift = params.kappa * (theta_eff - ivt[:, i-1]) * dt
        diffusion = params.sigma * dW

        ivt[:, i] = ivt[:, i-1] + drift + diffusion
        ivt[:, i] = np.maximum(ivt[:, i], 0)

        # Decrement AR timer
        ar_remaining -= dt
        ar_remaining = np.maximum(ar_remaining, 0)

    return ivt


def calibrate_jdou_parameters(
    ivt_data: np.ndarray,
    dt: float = 0.25,
    ar_threshold: float = 250.0
) -> JDOUParameters:
    """
    Calibrate JD-OU parameters from historical IVT data

    Strategy: physically separate AR events from normal regime
    1. Normal regime: consecutive steps where IVT < threshold
    2. AR events: contiguous periods where IVT >= threshold
    3. O-U fitted to normal regime only
    4. AR peak and duration from identified events

    Args:
        ivt_data: Historical IVT time series
        dt: Time step (0.25 = 6 hours)
        ar_threshold: IVT threshold for AR detection (kg/m/s)

    Returns:
        Calibrated JDOUParameters
    """
    increments = np.diff(ivt_data)
    above = ivt_data >= ar_threshold

    # --- Normal regime O-U calibration ---
    normal_mask = (~above[:-1]) & (~above[1:])
    normal_inc = increments[normal_mask]
    normal_ivt = ivt_data[:-1][normal_mask]

    theta_normal = np.mean(normal_ivt)
    normal_x = normal_ivt - theta_normal

    kappa = -np.sum(normal_inc * normal_x) / (np.sum(normal_x**2) * dt)
    kappa = max(kappa, 0.01)

    residuals = normal_inc + kappa * normal_x * dt
    sigma = np.std(residuals) / np.sqrt(dt)

    # --- AR event identification ---
    ar_onsets = np.where(np.diff(above.astype(int)) == 1)[0]
    ar_ends = np.where(np.diff(above.astype(int)) == -1)[0]

    # Align onsets and ends
    if len(ar_ends) > 0 and len(ar_onsets) > 0:
        if ar_ends[0] < ar_onsets[0]:
            ar_ends = ar_ends[1:]

    n_events = min(len(ar_onsets), len(ar_ends))
    total_time = len(ivt_data) * dt

    peaks = []
    durations = []
    for k in range(n_events):
        start = ar_onsets[k]
        end = ar_ends[k]
        segment = ivt_data[start:end + 1]
        peaks.append(np.max(segment))
        durations.append((end - start) * dt)  # days

    peaks = np.array(peaks)
    durations = np.array(durations)

    lam = n_events / total_time
    mu_peak = np.mean(peaks) if len(peaks) > 0 else ar_threshold
    sigma_peak = np.std(peaks) if len(peaks) > 1 else 0
    tau = np.mean(durations) if len(durations) > 0 else 1.0

    return JDOUParameters(
        kappa=kappa,
        theta=theta_normal,
        sigma=sigma,
        lam=lam,
        mu_peak=mu_peak,
        sigma_peak=sigma_peak,
        tau=tau
    )


if __name__ == "__main__":
    ivt = np.load('D:/projects/qae-parisian-climate/data/raw/ivt_sf_1980_2023.npy')

    params = calibrate_jdou_parameters(ivt, dt=0.25)
    print("Calibrated JD-OU Parameters:")
    print(f"  kappa:      {params.kappa:.4f} /day (half-life: {np.log(2)/params.kappa:.1f} days)")
    print(f"  theta:      {params.theta:.1f} kg/m/s (normal regime)")
    print(f"  sigma:      {params.sigma:.1f}")
    print(f"  lambda:     {params.lam:.4f} /day ({params.lam*365:.1f} AR/year)")
    print(f"  mu_peak:    {params.mu_peak:.1f} kg/m/s")
    print(f"  sigma_peak: {params.sigma_peak:.1f} kg/m/s")
    print(f"  tau:        {params.tau:.2f} days ({params.tau*24:.0f} hours)")
