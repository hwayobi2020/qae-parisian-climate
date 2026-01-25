"""
Robust Monte Carlo pricing under parameter uncertainty

Classical baseline for comparison with QAE
"""

import numpy as np
from typing import Tuple, List
from dataclasses import dataclass
import sys
sys.path.append('..')

from models.ornstein_uhlenbeck import OUParameters, simulate_ou_paths
from models.parisian import ParisianParams, monte_carlo_parisian_price


@dataclass
class ParameterRange:
    """Parameter uncertainty ranges"""
    kappa_min: float
    kappa_max: float
    sigma_min: float
    sigma_max: float
    theta_base: float = 300.0


def robust_mc_pricing(
    param_range: ParameterRange,
    parisian_params: ParisianParams,
    n_kappa_grid: int = 10,
    n_sigma_grid: int = 10,
    n_paths_per_param: int = 10000,
    n_steps: int = 168,
    dt: float = 1/24
) -> Tuple[float, float, dict]:
    """
    Compute price bounds under parameter uncertainty

    Classical approach: Grid search over parameter space

    Complexity: O(n_kappa * n_sigma * n_paths / ε²)

    Args:
        param_range: Parameter uncertainty ranges
        parisian_params: Parisian option parameters
        n_kappa_grid: Grid points for kappa
        n_sigma_grid: Grid points for sigma
        n_paths_per_param: MC paths per parameter combination
        n_steps: Time steps
        dt: Time step size

    Returns:
        (price_min, price_max, details)
    """
    kappa_values = np.linspace(
        param_range.kappa_min,
        param_range.kappa_max,
        n_kappa_grid
    )
    sigma_values = np.linspace(
        param_range.sigma_min,
        param_range.sigma_max,
        n_sigma_grid
    )

    prices = np.zeros((n_kappa_grid, n_sigma_grid))

    total_simulations = 0

    for i, kappa in enumerate(kappa_values):
        for j, sigma in enumerate(sigma_values):
            ou_params = OUParameters(
                kappa=kappa,
                theta_base=param_range.theta_base,
                sigma=sigma
            )

            paths = simulate_ou_paths(
                n_paths=n_paths_per_param,
                n_steps=n_steps,
                dt=dt,
                params=ou_params
            )

            price, _ = monte_carlo_parisian_price(paths, parisian_params)
            prices[i, j] = price
            total_simulations += n_paths_per_param

    # Find worst and best case
    price_min = np.min(prices)
    price_max = np.max(prices)

    # Find which parameters give extremes
    min_idx = np.unravel_index(np.argmin(prices), prices.shape)
    max_idx = np.unravel_index(np.argmax(prices), prices.shape)

    details = {
        'price_grid': prices,
        'kappa_values': kappa_values,
        'sigma_values': sigma_values,
        'worst_case_params': {
            'kappa': kappa_values[min_idx[0]],
            'sigma': sigma_values[min_idx[1]]
        },
        'best_case_params': {
            'kappa': kappa_values[max_idx[0]],
            'sigma': sigma_values[max_idx[1]]
        },
        'total_simulations': total_simulations,
        'complexity': f'O({n_kappa_grid} × {n_sigma_grid} × {n_paths_per_param})'
    }

    return price_min, price_max, details


if __name__ == "__main__":
    # Example usage
    param_range = ParameterRange(
        kappa_min=0.2,
        kappa_max=0.8,
        sigma_min=40,
        sigma_max=120,
        theta_base=300
    )

    parisian_params = ParisianParams(barrier=250, window=48)

    print("Running robust MC pricing...")
    print("(This may take a while for classical baseline)\n")

    price_min, price_max, details = robust_mc_pricing(
        param_range=param_range,
        parisian_params=parisian_params,
        n_kappa_grid=5,      # Reduced for quick test
        n_sigma_grid=5,
        n_paths_per_param=1000
    )

    print(f"Price range: [{price_min:.4f}, {price_max:.4f}]")
    print(f"Worst case params: {details['worst_case_params']}")
    print(f"Best case params: {details['best_case_params']}")
    print(f"Total simulations: {details['total_simulations']:,}")
    print(f"Complexity: {details['complexity']}")
