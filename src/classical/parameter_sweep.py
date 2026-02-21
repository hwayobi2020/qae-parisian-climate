"""
Parameter Uncertainty Analysis for Parisian Climate Derivatives

Bootstrap-based parameter set generation and batch MC/IS-MC execution.
Demonstrates the K × O(1/ε²) classical cost that motivates QAE.

Workflow:
    1. Bootstrap resample historical IVT → K calibrated parameter sets
    2. Run MC and IS-MC for each parameter set
    3. Collect distribution of trigger probabilities across uncertainty
    4. Compare wall-clock cost: K/ε² (MC) vs K/ε (QAE)

References:
    - Efron & Tibshirani (1993): Bootstrap Methods
    - Glasserman (2003): Monte Carlo Methods in Financial Engineering, Ch. 7
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.jump_diffusion_ou import JDOUParameters, calibrate_jdou_parameters, simulate_jdou_paths
from models.parisian import ParisianParams, monte_carlo_parisian_price, check_parisian_condition
from classical.glasserman_is import (
    TiltParameters, simulate_jdou_paths_is, is_parisian_price, find_optimal_tilt
)


# ============================================================
# 1. Bootstrap Parameter Set Generator
# ============================================================

def generate_bootstrap_parameter_sets(
    ivt_data: np.ndarray,
    K: int,
    dt: float = 0.25,
    ar_threshold: float = 250.0,
    block_size_days: int = 30,
    seed: int = 42,
) -> List[JDOUParameters]:
    """
    Generate K parameter sets via block bootstrap of historical IVT data.

    Block bootstrap preserves temporal autocorrelation:
    1. Split IVT series into blocks of `block_size_days` days
    2. Resample blocks with replacement to form new series
    3. Calibrate JD-OU parameters on each resampled series

    Args:
        ivt_data: Historical IVT time series (6-hourly)
        K: Number of bootstrap parameter sets to generate
        dt: Time step (0.25 = 6 hours)
        ar_threshold: IVT threshold for AR event detection
        block_size_days: Block size in days for block bootstrap
        seed: Random seed

    Returns:
        List of K calibrated JDOUParameters
    """
    rng = np.random.RandomState(seed)
    n_obs = len(ivt_data)
    steps_per_day = int(1.0 / dt)
    block_size = block_size_days * steps_per_day  # in time steps
    n_blocks = n_obs // block_size

    # Split data into blocks
    blocks = []
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size
        blocks.append(ivt_data[start:end])

    param_sets = []
    for k in range(K):
        # Resample blocks with replacement
        chosen = rng.randint(0, n_blocks, size=n_blocks)
        resampled = np.concatenate([blocks[c] for c in chosen])

        # Calibrate on resampled series
        try:
            params = calibrate_jdou_parameters(resampled, dt=dt, ar_threshold=ar_threshold)
            param_sets.append(params)
        except Exception:
            # If calibration fails (e.g., no AR events in resample), retry
            alt_chosen = rng.randint(0, n_blocks, size=n_blocks)
            resampled_alt = np.concatenate([blocks[c] for c in alt_chosen])
            params = calibrate_jdou_parameters(resampled_alt, dt=dt, ar_threshold=ar_threshold)
            param_sets.append(params)

    return param_sets


def generate_parametric_sweep(
    base_params: JDOUParameters,
    n_kappa: int = 5,
    n_sigma: int = 5,
    n_lambda: int = 3,
    kappa_range: Tuple[float, float] = None,
    sigma_range: Tuple[float, float] = None,
    lambda_range: Tuple[float, float] = None,
) -> List[JDOUParameters]:
    """
    Generate parameter sets via deterministic grid sweep.

    Useful when bootstrap data is unavailable or for controlled experiments.

    Args:
        base_params: Reference JD-OU parameters
        n_kappa, n_sigma, n_lambda: Grid points per parameter
        kappa_range, sigma_range, lambda_range: Override ranges

    Returns:
        List of JDOUParameters on the grid
    """
    if kappa_range is None:
        kappa_range = (base_params.kappa * 0.5, base_params.kappa * 2.0)
    if sigma_range is None:
        sigma_range = (base_params.sigma * 0.5, base_params.sigma * 2.0)
    if lambda_range is None:
        lambda_range = (base_params.lam * 0.5, base_params.lam * 2.0)

    kappas = np.linspace(*kappa_range, n_kappa)
    sigmas = np.linspace(*sigma_range, n_sigma)
    lambdas = np.linspace(*lambda_range, n_lambda)

    param_sets = []
    for k in kappas:
        for s in sigmas:
            for l in lambdas:
                p = JDOUParameters(
                    kappa=k, theta=base_params.theta, sigma=s,
                    lam=l, mu_peak=base_params.mu_peak,
                    sigma_peak=base_params.sigma_peak, tau=base_params.tau,
                )
                param_sets.append(p)

    return param_sets


# ============================================================
# 2. Batch MC / IS-MC Runner
# ============================================================

@dataclass
class SingleRunResult:
    """Result for one parameter set"""
    param_idx: int
    params: JDOUParameters
    price_mc: float
    se_mc: float
    price_is: float
    se_is: float
    variance_mc: float
    variance_is: float
    vr_ratio: float
    trigger_rate_mc: float
    trigger_rate_is_Q: float
    time_mc: float
    time_is: float
    n_paths: int


def run_single_paramset(
    idx: int,
    params: JDOUParameters,
    parisian_params: ParisianParams,
    n_steps: int,
    dt: float,
    n_paths_mc: int,
    n_paths_is: int,
    tilt: TiltParameters = None,
    ivt_init: float = None,
    seed: int = 42,
) -> SingleRunResult:
    """
    Run MC and IS-MC for a single parameter set.

    Args:
        idx: Parameter set index
        params: JD-OU parameters
        parisian_params: Parisian condition
        n_steps: Time steps
        dt: Time step size
        n_paths_mc: Paths for naive MC
        n_paths_is: Paths for IS-MC
        tilt: IS tilt parameters (if None, uses light heuristic)
        ivt_init: Initial IVT
        seed: Random seed

    Returns:
        SingleRunResult with prices, errors, timings
    """
    if ivt_init is None:
        ivt_init = params.theta

    # --- Naive MC ---
    t0 = time.time()
    paths_mc = simulate_jdou_paths(
        n_paths=n_paths_mc, n_steps=n_steps, dt=dt,
        params=params, ivt_init=ivt_init, seed=seed,
    )
    price_mc, se_mc = monte_carlo_parisian_price(paths_mc, parisian_params)
    from models.parisian import compute_parisian_payoffs
    payoffs_mc = compute_parisian_payoffs(paths_mc, parisian_params)
    var_mc = float(np.var(payoffs_mc))
    trigger_rate_mc = float(np.mean(payoffs_mc > 0))
    time_mc = time.time() - t0

    # --- IS-MC ---
    if tilt is None:
        tilt = TiltParameters(mu_diffusion=0.0, lambda_factor=1.5, tau_factor=1.5)

    t0 = time.time()
    paths_is, log_lr = simulate_jdou_paths_is(
        n_paths=n_paths_is, n_steps=n_steps, dt=dt,
        params=params, tilt=tilt, ivt_init=ivt_init, seed=seed + 50000,
    )
    price_is, se_is, is_details = is_parisian_price(paths_is, log_lr, parisian_params)
    var_is = is_details['is_variance']
    trigger_rate_is_Q = is_details['trigger_rate_Q']
    time_is = time.time() - t0

    vr_ratio = var_mc / var_is if var_is > 0 else float('inf')

    return SingleRunResult(
        param_idx=idx,
        params=params,
        price_mc=price_mc, se_mc=se_mc,
        price_is=price_is, se_is=se_is,
        variance_mc=var_mc, variance_is=var_is,
        vr_ratio=vr_ratio,
        trigger_rate_mc=trigger_rate_mc,
        trigger_rate_is_Q=trigger_rate_is_Q,
        time_mc=time_mc, time_is=time_is,
        n_paths=n_paths_mc,
    )


def batch_parameter_sweep(
    param_sets: List[JDOUParameters],
    parisian_params: ParisianParams,
    n_steps: int = 120,
    dt: float = 0.25,
    n_paths_mc: int = 10000,
    n_paths_is: int = 10000,
    tilt: TiltParameters = None,
    seed: int = 42,
    verbose: bool = True,
) -> Dict:
    """
    Run MC and IS-MC across all K parameter sets.

    This is the classical baseline: K × O(1/ε²) total cost.
    QAE would reduce this to K × O(1/ε).

    Args:
        param_sets: K parameter sets from bootstrap or grid
        parisian_params: Parisian condition parameters
        n_steps: Time steps per simulation
        dt: Time step size
        n_paths_mc: MC paths per parameter set
        n_paths_is: IS-MC paths per parameter set
        tilt: IS tilt (shared across all sets)
        seed: Base random seed
        verbose: Print progress

    Returns:
        Dictionary with all results, statistics, and timing
    """
    K = len(param_sets)
    results = []
    total_time_mc = 0.0
    total_time_is = 0.0

    if verbose:
        print(f"\n{'='*70}")
        print(f"BATCH PARAMETER SWEEP: K={K} parameter sets")
        print(f"  Parisian: B={parisian_params.barrier}, W={parisian_params.window} steps")
        print(f"  Contract: {n_steps} steps ({n_steps*dt:.0f} days)")
        print(f"  MC paths: {n_paths_mc:,} per set, IS paths: {n_paths_is:,} per set")
        print(f"{'='*70}")

    t_total_start = time.time()

    for k in range(K):
        result = run_single_paramset(
            idx=k,
            params=param_sets[k],
            parisian_params=parisian_params,
            n_steps=n_steps, dt=dt,
            n_paths_mc=n_paths_mc,
            n_paths_is=n_paths_is,
            tilt=tilt,
            seed=seed + k * 1000,
        )
        results.append(result)
        total_time_mc += result.time_mc
        total_time_is += result.time_is

        if verbose and (k + 1) % max(1, K // 10) == 0:
            print(f"  [{k+1}/{K}] P_mc={result.price_mc:.4f}, "
                  f"P_is={result.price_is:.4f}, VR={result.vr_ratio:.1f}x, "
                  f"t_mc={result.time_mc:.2f}s, t_is={result.time_is:.2f}s")

    t_total = time.time() - t_total_start

    # Collect statistics
    prices_mc = np.array([r.price_mc for r in results])
    prices_is = np.array([r.price_is for r in results])
    vr_ratios = np.array([r.vr_ratio for r in results])
    times_mc = np.array([r.time_mc for r in results])
    times_is = np.array([r.time_is for r in results])

    # Worst-case analysis
    worst_idx_mc = int(np.argmax(prices_mc))
    worst_idx_is = int(np.argmax(prices_is))

    summary = {
        'K': K,
        'n_paths_mc': n_paths_mc,
        'n_paths_is': n_paths_is,
        'results': results,

        # Price distribution across parameter uncertainty
        'prices_mc': {
            'mean': float(np.mean(prices_mc)),
            'std': float(np.std(prices_mc)),
            'min': float(np.min(prices_mc)),
            'max': float(np.max(prices_mc)),
            'median': float(np.median(prices_mc)),
            'percentile_95': float(np.percentile(prices_mc, 95)),
            'percentile_99': float(np.percentile(prices_mc, 99)),
            'worst_case_params': {
                'kappa': param_sets[worst_idx_mc].kappa,
                'sigma': param_sets[worst_idx_mc].sigma,
                'lam': param_sets[worst_idx_mc].lam,
            },
        },
        'prices_is': {
            'mean': float(np.mean(prices_is)),
            'std': float(np.std(prices_is)),
            'min': float(np.min(prices_is)),
            'max': float(np.max(prices_is)),
            'median': float(np.median(prices_is)),
            'percentile_95': float(np.percentile(prices_is, 95)),
            'percentile_99': float(np.percentile(prices_is, 99)),
        },

        # Variance reduction
        'variance_reduction': {
            'mean_vr': float(np.mean(vr_ratios)),
            'median_vr': float(np.median(vr_ratios)),
            'min_vr': float(np.min(vr_ratios)),
            'max_vr': float(np.max(vr_ratios)),
        },

        # Timing
        'timing': {
            'total_mc_seconds': float(total_time_mc),
            'total_is_seconds': float(total_time_is),
            'total_seconds': float(t_total),
            'avg_mc_per_set': float(np.mean(times_mc)),
            'avg_is_per_set': float(np.mean(times_is)),
        },

        # Scaling projections
        'scaling': _compute_scaling_projections(
            K, n_paths_mc, float(np.mean(times_mc)), float(np.mean(prices_mc))
        ),
    }

    if verbose:
        _print_summary(summary)

    return summary


def _compute_scaling_projections(
    K: int,
    n_paths: int,
    avg_time_per_set: float,
    avg_price: float,
) -> Dict:
    """
    Project classical vs quantum costs for large-scale parameter sweeps.

    Classical MC: K × O(1/ε²) samples
    QAE:          K × O(1/ε) oracle calls
    """
    projections = {}

    for eps_label, eps in [('1e-2', 0.01), ('1e-3', 0.001), ('1e-4', 0.0001)]:
        # MC: samples needed for precision epsilon
        # Var(Bernoulli) ~ p(1-p), so samples = p(1-p)/ε²
        p = max(avg_price, 0.001)
        mc_samples = p * (1 - p) / eps**2
        mc_samples_per_set = mc_samples

        # Time estimate based on observed rate
        samples_per_sec = n_paths / avg_time_per_set if avg_time_per_set > 0 else 1e6
        mc_time_per_set = mc_samples_per_set / samples_per_sec
        mc_total_time = K * mc_time_per_set

        # QAE: O(1/ε) oracle calls
        qae_calls = int(np.pi / (4 * eps))
        qae_total_calls = K * qae_calls

        projections[f'epsilon={eps_label}'] = {
            'mc_samples_per_set': int(mc_samples_per_set),
            'mc_total_samples': int(K * mc_samples_per_set),
            'mc_time_seconds': mc_total_time,
            'mc_time_human': _format_time(mc_total_time),
            'qae_calls_per_set': qae_calls,
            'qae_total_calls': qae_total_calls,
            'speedup_in_calls': int(mc_samples_per_set) // qae_calls if qae_calls > 0 else 0,
        }

    return projections


def _format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    elif seconds < 86400:
        return f"{seconds/3600:.1f}h"
    else:
        return f"{seconds/86400:.1f}d"


def _print_summary(summary: Dict):
    """Print formatted summary of batch sweep results."""
    K = summary['K']
    pm = summary['prices_mc']
    pi = summary['prices_is']
    vr = summary['variance_reduction']
    tm = summary['timing']

    print(f"\n{'='*70}")
    print(f"RESULTS: {K} parameter sets")
    print(f"{'='*70}")

    print(f"\n  Price Distribution (MC):")
    print(f"    Mean:   {pm['mean']:.6f}")
    print(f"    Std:    {pm['std']:.6f}")
    print(f"    Range:  [{pm['min']:.6f}, {pm['max']:.6f}]")
    print(f"    95th:   {pm['percentile_95']:.6f}")
    print(f"    99th:   {pm['percentile_99']:.6f}")
    print(f"    Worst:  kappa={pm['worst_case_params']['kappa']:.3f}, "
          f"sigma={pm['worst_case_params']['sigma']:.1f}, "
          f"lam={pm['worst_case_params']['lam']:.3f}")

    print(f"\n  Price Distribution (IS-MC):")
    print(f"    Mean:   {pi['mean']:.6f}")
    print(f"    Std:    {pi['std']:.6f}")
    print(f"    Range:  [{pi['min']:.6f}, {pi['max']:.6f}]")

    print(f"\n  Variance Reduction:")
    print(f"    Mean VR:   {vr['mean_vr']:.1f}x")
    print(f"    Median VR: {vr['median_vr']:.1f}x")
    print(f"    Range:     [{vr['min_vr']:.1f}x, {vr['max_vr']:.1f}x]")

    print(f"\n  Timing:")
    print(f"    Total MC:  {_format_time(tm['total_mc_seconds'])}")
    print(f"    Total IS:  {_format_time(tm['total_is_seconds'])}")
    print(f"    Avg per set: MC={tm['avg_mc_per_set']:.2f}s, IS={tm['avg_is_per_set']:.2f}s")

    print(f"\n  Scaling Projections (K={K} parameter sets):")
    print(f"    {'Precision':<12} {'MC samples':<15} {'MC time':<12} {'QAE calls':<12} {'Speedup':<10}")
    print(f"    {'-'*60}")
    for eps_key, proj in summary['scaling'].items():
        print(f"    {eps_key:<12} {proj['mc_total_samples']:<15,} "
              f"{proj['mc_time_human']:<12} {proj['qae_total_calls']:<12,} "
              f"{proj['speedup_in_calls']}x")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Load historical data
    data_path = os.path.join(
        os.path.dirname(__file__), '..', '..', 'data', 'raw', 'ivt_sf_1980_2023.npy'
    )

    if os.path.exists(data_path):
        ivt_data = np.load(data_path)
        print(f"Loaded IVT data: {len(ivt_data):,} observations")

        # Generate bootstrap parameter sets
        print("\nGenerating bootstrap parameter sets...")
        K = 50  # Small K for demo (paper would use 1000+)
        param_sets = generate_bootstrap_parameter_sets(
            ivt_data, K=K, dt=0.25, seed=42
        )
        print(f"Generated {len(param_sets)} parameter sets")

        # Show parameter distribution
        kappas = [p.kappa for p in param_sets]
        sigmas = [p.sigma for p in param_sets]
        lams = [p.lam for p in param_sets]
        print(f"\n  kappa: {np.mean(kappas):.3f} ± {np.std(kappas):.3f} "
              f"[{np.min(kappas):.3f}, {np.max(kappas):.3f}]")
        print(f"  sigma: {np.mean(sigmas):.1f} ± {np.std(sigmas):.1f} "
              f"[{np.min(sigmas):.1f}, {np.max(sigmas):.1f}]")
        print(f"  lambda: {np.mean(lams):.4f} ± {np.std(lams):.4f} "
              f"[{np.min(lams):.4f}, {np.max(lams):.4f}]")

    else:
        print("Historical data not found. Using parametric sweep.")
        base_params = JDOUParameters(
            kappa=0.767, theta=85.8, sigma=62.0,
            lam=0.108, mu_peak=390.5, sigma_peak=80.0, tau=0.69
        )
        K = 50
        param_sets = generate_parametric_sweep(
            base_params, n_kappa=5, n_sigma=5, n_lambda=2
        )
        print(f"Generated {len(param_sets)} parameter sets (grid sweep)")

    # Batch sweep
    parisian = ParisianParams(barrier=250, window=8)  # 8 steps = 48h at dt=0.25

    summary = batch_parameter_sweep(
        param_sets=param_sets,
        parisian_params=parisian,
        n_steps=120,  # 30 days
        dt=0.25,
        n_paths_mc=5000,   # Small for demo
        n_paths_is=5000,
        seed=42,
        verbose=True,
    )
