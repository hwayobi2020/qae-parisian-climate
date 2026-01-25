"""
Ornstein-Uhlenbeck process for IVT modeling

dIVT = κ(θ(t) - IVT)dt + σdW

where:
    κ: mean reversion speed
    θ(t): long-term mean (seasonal)
    σ: volatility
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class OUParameters:
    """Parameters for O-U process"""
    kappa: float          # Mean reversion speed
    theta_base: float     # Base long-term mean
    sigma: float          # Volatility

    # Optional: parameter ranges for robust pricing
    kappa_range: Optional[Tuple[float, float]] = None
    sigma_range: Optional[Tuple[float, float]] = None


def seasonal_theta(t: np.ndarray, theta_base: float,
                   amplitude: float = 100, phase: float = 0) -> np.ndarray:
    """
    Seasonal long-term mean for IVT

    θ(t) = θ_base + A * sin(2πt/365 + φ)

    Args:
        t: Time in days
        theta_base: Base mean IVT
        amplitude: Seasonal amplitude
        phase: Phase shift

    Returns:
        Seasonal theta values
    """
    return theta_base + amplitude * np.sin(2 * np.pi * t / 365 + phase)


def simulate_ou_paths(
    n_paths: int,
    n_steps: int,
    dt: float,
    params: OUParameters,
    ivt_init: float = 200.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate IVT paths using Ornstein-Uhlenbeck process

    Args:
        n_paths: Number of Monte Carlo paths
        n_steps: Number of time steps
        dt: Time step size (in days, e.g., 1/24 for hourly)
        params: O-U parameters
        ivt_init: Initial IVT value
        seed: Random seed

    Returns:
        Array of shape (n_paths, n_steps) with IVT values
    """
    if seed is not None:
        np.random.seed(seed)

    ivt = np.zeros((n_paths, n_steps))
    ivt[:, 0] = ivt_init

    # Time array for seasonal theta
    t = np.arange(n_steps) * dt
    theta_t = seasonal_theta(t, params.theta_base)

    # Simulate paths
    for i in range(1, n_steps):
        dW = np.sqrt(dt) * np.random.randn(n_paths)
        drift = params.kappa * (theta_t[i-1] - ivt[:, i-1]) * dt
        diffusion = params.sigma * dW
        ivt[:, i] = ivt[:, i-1] + drift + diffusion

        # IVT should be non-negative
        ivt[:, i] = np.maximum(ivt[:, i], 0)

    return ivt


def calibrate_ou_parameters(
    ivt_data: np.ndarray,
    dt: float = 1.0
) -> OUParameters:
    """
    Calibrate O-U parameters from historical IVT data

    Uses AR(1) regression: IVT(t) - IVT(t-1) = κ(θ - IVT(t-1))dt + noise

    Args:
        ivt_data: Historical IVT time series
        dt: Time step

    Returns:
        Calibrated OUParameters
    """
    # Remove seasonal component first (simplified)
    theta_base = np.mean(ivt_data)

    # AR(1) regression for kappa
    diff = np.diff(ivt_data)
    x = ivt_data[:-1] - theta_base

    # Linear regression: diff = -kappa * x * dt + noise
    kappa = -np.sum(diff * x) / (np.sum(x**2) * dt)
    kappa = max(kappa, 0.01)  # Ensure positive

    # Estimate sigma from residuals
    residuals = diff + kappa * x * dt
    sigma = np.std(residuals) / np.sqrt(dt)

    return OUParameters(
        kappa=kappa,
        theta_base=theta_base,
        sigma=sigma
    )


if __name__ == "__main__":
    # Quick test
    params = OUParameters(kappa=0.5, theta_base=300, sigma=50)
    paths = simulate_ou_paths(
        n_paths=1000,
        n_steps=100,
        dt=1/24,  # hourly
        params=params
    )
    print(f"Shape: {paths.shape}")
    print(f"Mean final IVT: {paths[:, -1].mean():.1f}")
    print(f"Std final IVT: {paths[:, -1].std():.1f}")
