"""
Parisian option payoff for AR derivatives

Parisian condition: IVT >= barrier for consecutive duration >= window
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class ParisianParams:
    """Parisian option parameters"""
    barrier: float = 250.0      # IVT threshold (kg/m/s)
    window: int = 48            # Consecutive hours required
    payoff_amount: float = 1.0  # Payoff if triggered


def check_parisian_condition(
    ivt_path: np.ndarray,
    params: ParisianParams
) -> Tuple[bool, int]:
    """
    Check if Parisian condition is met for a single path

    Args:
        ivt_path: 1D array of IVT values over time
        params: Parisian parameters

    Returns:
        (triggered, trigger_time): Whether triggered and when
    """
    consecutive_count = 0

    for t, ivt in enumerate(ivt_path):
        if ivt >= params.barrier:
            consecutive_count += 1
            if consecutive_count >= params.window:
                return True, t - params.window + 1
        else:
            consecutive_count = 0  # Reset!

    return False, -1


def compute_parisian_payoffs(
    ivt_paths: np.ndarray,
    params: ParisianParams
) -> np.ndarray:
    """
    Compute Parisian payoffs for multiple paths

    Args:
        ivt_paths: 2D array (n_paths, n_steps)
        params: Parisian parameters

    Returns:
        1D array of payoffs
    """
    n_paths = ivt_paths.shape[0]
    payoffs = np.zeros(n_paths)

    for i in range(n_paths):
        triggered, _ = check_parisian_condition(ivt_paths[i], params)
        if triggered:
            payoffs[i] = params.payoff_amount

    return payoffs


def monte_carlo_parisian_price(
    ivt_paths: np.ndarray,
    params: ParisianParams
) -> Tuple[float, float]:
    """
    Price Parisian AR derivative using Monte Carlo

    Args:
        ivt_paths: Simulated IVT paths
        params: Parisian parameters

    Returns:
        (price, std_error)
    """
    payoffs = compute_parisian_payoffs(ivt_paths, params)

    price = np.mean(payoffs)
    std_error = np.std(payoffs) / np.sqrt(len(payoffs))

    return price, std_error


if __name__ == "__main__":
    from ornstein_uhlenbeck import OUParameters, simulate_ou_paths

    # Simulate paths
    ou_params = OUParameters(kappa=0.3, theta_base=300, sigma=80)
    paths = simulate_ou_paths(
        n_paths=10000,
        n_steps=168,  # 1 week in hours
        dt=1/24,
        params=ou_params,
        ivt_init=200
    )

    # Price Parisian option
    parisian_params = ParisianParams(barrier=250, window=48)
    price, err = monte_carlo_parisian_price(paths, parisian_params)

    print(f"Parisian AR Option Price: {price:.4f} ± {err:.4f}")
    print(f"Trigger probability: {price*100:.2f}%")
