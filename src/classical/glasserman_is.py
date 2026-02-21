"""
Importance Sampling for Parisian Climate Derivatives (Glasserman Framework)

Three-component exponential tilting for JD-OU process:
1. Diffusion tilt: shift Brownian noise mean (Girsanov theorem)
2. Jump rate tilt: increase AR event frequency
3. Jump duration tilt: lengthen AR events

The IS estimator: E_P[f(X)] = E_Q[f(X) * dP/dQ]
where Q is the tilted measure and dP/dQ is the likelihood ratio.

For rare Parisian events (trigger prob ~0.1-5%), IS dramatically reduces
variance by generating more trigger events under Q, then correcting
with the likelihood ratio.

References:
    - Glasserman (2003): Monte Carlo Methods in Financial Engineering, Ch. 4-5
    - Glasserman & Li (2005): Importance Sampling for Portfolio Credit Risk
    - Merton (1976): Jump-diffusion option pricing
"""

import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass, field
from scipy.stats import norm
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.jump_diffusion_ou import JDOUParameters, simulate_jdou_paths
from models.parisian import ParisianParams, check_parisian_condition


@dataclass
class TiltParameters:
    """Importance sampling tilt parameters for JD-OU process

    Three independent tilt components:
    - mu_diffusion: additive shift to each N(0,1) Brownian increment
      Under tilt, noise z_t ~ N(mu, 1) instead of N(0, 1)
      Effective O-U mean shifts to theta + sigma*mu/(kappa*sqrt(dt))
    - lambda_factor: multiplicative factor for AR arrival rate
      lambda_tilt = lambda * lambda_factor
    - tau_factor: multiplicative factor for AR mean duration
      tau_tilt = tau * tau_factor
    """
    mu_diffusion: float = 0.0
    lambda_factor: float = 1.0
    tau_factor: float = 1.0

    def __repr__(self):
        parts = []
        if self.mu_diffusion != 0:
            parts.append(f"mu_diff={self.mu_diffusion:.4f}")
        if self.lambda_factor != 1.0:
            parts.append(f"lam_x={self.lambda_factor:.2f}")
        if self.tau_factor != 1.0:
            parts.append(f"tau_x={self.tau_factor:.2f}")
        return f"Tilt({', '.join(parts) if parts else 'identity'})"


def compute_heuristic_tilt(
    params: JDOUParameters,
    parisian_params: ParisianParams,
    dt: float,
    strategy: str = 'moderate'
) -> TiltParameters:
    """
    Suggest initial tilt parameters based on model and contract structure

    Strategies:
    - 'light': minimal tilting, ~2-5x variance reduction
    - 'moderate': balanced tilting, ~10-50x variance reduction
    - 'aggressive': heavy tilting, >50x variance reduction (risk of weight degeneracy)

    Args:
        params: JD-OU model parameters
        parisian_params: Parisian contract parameters
        dt: time step size (days)
        strategy: 'light', 'moderate', or 'aggressive'

    Returns:
        Suggested TiltParameters
    """
    B = parisian_params.barrier
    W_hours = parisian_params.window
    W_days = W_hours * dt  # window duration in days (if window is in steps)

    # How far is barrier from normal mean?
    gap_ratio = (B - params.theta) / params.sigma

    # How does window compare to mean AR duration?
    duration_ratio = W_days / params.tau if params.tau > 0 else 1.0

    if strategy == 'light':
        mu = 0.1 * gap_ratio * np.sqrt(dt)
        lam_factor = 1.5
        tau_factor = max(1.0, duration_ratio * 0.3)
    elif strategy == 'moderate':
        mu = 0.2 * gap_ratio * np.sqrt(dt)
        lam_factor = 2.5
        tau_factor = max(1.0, duration_ratio * 0.5)
    elif strategy == 'aggressive':
        mu = 0.4 * gap_ratio * np.sqrt(dt)
        lam_factor = 5.0
        tau_factor = max(1.0, duration_ratio * 0.8)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Clamp to reasonable ranges
    mu = np.clip(mu, 0, 2.0)
    lam_factor = np.clip(lam_factor, 1.0, 10.0)
    tau_factor = np.clip(tau_factor, 1.0, 5.0)

    return TiltParameters(
        mu_diffusion=mu,
        lambda_factor=lam_factor,
        tau_factor=tau_factor,
    )


def simulate_jdou_paths_is(
    n_paths: int,
    n_steps: int,
    dt: float,
    params: JDOUParameters,
    tilt: TiltParameters,
    ivt_init: float = None,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate JD-OU paths under importance sampling (tilted) measure Q

    Under Q:
    - Brownian increments: Z_t = eps_t + mu (eps_t ~ N(0,1))
    - AR arrivals: Poisson(lambda * lambda_factor)
    - AR durations: Exp(tau * tau_factor)

    The log-likelihood ratio log(dP/dQ) is accumulated per path.

    Args:
        n_paths: number of Monte Carlo paths
        n_steps: number of time steps
        dt: time step size (days)
        params: JD-OU model parameters
        tilt: importance sampling tilt parameters
        ivt_init: initial IVT value
        seed: random seed

    Returns:
        (paths, log_lr):
            paths: (n_paths, n_steps) IVT values under Q
            log_lr: (n_paths,) log(dP/dQ) for each path
    """
    rng = np.random.RandomState(seed)

    if ivt_init is None:
        ivt_init = params.theta

    ivt = np.zeros((n_paths, n_steps))
    ivt[:, 0] = ivt_init
    log_lr = np.zeros(n_paths)

    # Tilted parameters
    mu = tilt.mu_diffusion
    lam_tilt = params.lam * tilt.lambda_factor
    tau_tilt = params.tau * tilt.tau_factor

    # AR state tracking
    ar_remaining = np.zeros(n_paths)
    ar_target = np.zeros(n_paths)

    for i in range(1, n_steps):
        # === Jump component ===
        not_in_ar = ar_remaining <= 0
        u_jump = rng.random(n_paths)
        new_ar = not_in_ar & (u_jump < lam_tilt * dt)

        # Log-LR for jump arrivals
        if new_ar.any():
            # LR = P(jump) / Q(jump) = (lam*dt) / (lam_tilt*dt)
            log_lr[new_ar] += np.log(params.lam) - np.log(lam_tilt)

            n_new = new_ar.sum()
            # AR duration under tilted measure
            ar_durations = rng.exponential(tau_tilt, n_new)
            ar_remaining[new_ar] = ar_durations

            # LR for AR duration: p(d)/q(d) = (tau_tilt/tau) * exp(d*(1/tau_tilt - 1/tau))
            if tilt.tau_factor != 1.0:
                log_lr[new_ar] += (
                    np.log(tau_tilt / params.tau)
                    + ar_durations * (1.0 / tau_tilt - 1.0 / params.tau)
                )

            # AR peak IVT (not tilted — could be extended)
            ar_target[new_ar] = rng.normal(params.mu_peak, params.sigma_peak, n_new)
            ar_target[new_ar] = np.maximum(ar_target[new_ar], params.theta)

        # LR for non-jump steps (paths eligible for jump but didn't jump)
        no_jump = not_in_ar & (~new_ar)
        if no_jump.any():
            # LR = P(no jump) / Q(no jump) = (1 - lam*dt) / (1 - lam_tilt*dt)
            log_p = np.log(1 - params.lam * dt)
            log_q = np.log(1 - lam_tilt * dt)
            log_lr[no_jump] += log_p - log_q

        # Effective theta
        in_ar = ar_remaining > 0
        theta_eff = np.where(in_ar, ar_target, params.theta)

        # === Diffusion component with tilting ===
        epsilon = rng.randn(n_paths)       # N(0,1) under Q
        z_tilted = epsilon + mu            # Actual noise: N(mu, 1) under Q

        # Log-LR for diffusion: log(p(z)/q(z)) = -mu*z + mu^2/2
        if mu != 0:
            log_lr += -mu * z_tilted + mu**2 / 2

        # O-U step with tilted noise
        drift = params.kappa * (theta_eff - ivt[:, i-1]) * dt
        diffusion = params.sigma * np.sqrt(dt) * z_tilted

        ivt[:, i] = ivt[:, i-1] + drift + diffusion
        ivt[:, i] = np.maximum(ivt[:, i], 0)

        # Decrement AR timer
        ar_remaining -= dt
        ar_remaining = np.maximum(ar_remaining, 0)

    return ivt, log_lr


def is_parisian_price(
    ivt_paths: np.ndarray,
    log_lr: np.ndarray,
    parisian_params: ParisianParams
) -> Tuple[float, float, Dict]:
    """
    Compute IS-weighted Parisian derivative price

    IS estimator: price = (1/N) * sum(1{triggered} * payoff * dP/dQ)

    Args:
        ivt_paths: (n_paths, n_steps) simulated under tilted measure Q
        log_lr: (n_paths,) log(dP/dQ) for each path
        parisian_params: Parisian barrier/window parameters

    Returns:
        (price, std_error, details)
    """
    n_paths = ivt_paths.shape[0]

    # Check Parisian condition for each path
    triggered = np.zeros(n_paths, dtype=bool)
    for i in range(n_paths):
        triggered[i], _ = check_parisian_condition(ivt_paths[i], parisian_params)

    # Likelihood ratios with numerical stability
    # Clip log_lr to prevent overflow/underflow
    log_lr_clipped = np.clip(log_lr, -500, 500)
    lr = np.exp(log_lr_clipped)

    # IS-weighted payoffs
    weighted_payoffs = triggered.astype(float) * parisian_params.payoff_amount * lr

    price = np.mean(weighted_payoffs)
    variance = np.var(weighted_payoffs)
    std_error = np.std(weighted_payoffs) / np.sqrt(n_paths)

    # Effective Sample Size (ESS)
    if triggered.any():
        w = lr[triggered]
        ess = (np.sum(w))**2 / np.sum(w**2)
    else:
        ess = 0.0

    # Self-normalized estimator (more robust but slightly biased)
    total_weight = np.sum(lr)
    if total_weight > 0:
        price_sn = np.sum(weighted_payoffs) / total_weight
    else:
        price_sn = 0.0

    details = {
        'trigger_rate_Q': float(np.mean(triggered)),
        'trigger_count': int(triggered.sum()),
        'mean_lr': float(np.mean(lr)),
        'std_lr': float(np.std(lr)),
        'max_lr': float(np.max(lr)),
        'min_lr': float(np.min(lr)),
        'mean_log_lr': float(np.mean(log_lr)),
        'std_log_lr': float(np.std(log_lr)),
        'ess': float(ess),
        'ess_ratio': float(ess / n_paths) if n_paths > 0 else 0,
        'is_variance': float(variance),
        'price_self_normalized': float(price_sn),
        'n_lr_clipped': int(np.sum(np.abs(log_lr) > 500)),
    }

    return price, std_error, details


def find_optimal_tilt(
    params: JDOUParameters,
    parisian_params: ParisianParams,
    n_steps: int,
    dt: float,
    search_space: Dict = None,
    n_paths_per: int = 5000,
    ivt_init: float = None,
    seed: int = 42,
    verbose: bool = False
) -> Tuple[TiltParameters, Dict]:
    """
    Find optimal tilt parameters via grid search over IS variance

    Searches over combinations of (mu_diffusion, lambda_factor, tau_factor)
    and selects the combination with minimum IS variance.

    Args:
        params: JD-OU model parameters
        parisian_params: Parisian condition parameters
        n_steps: number of time steps
        dt: time step size
        search_space: dict with 'mu', 'lam_factor', 'tau_factor' arrays
        n_paths_per: paths per grid point
        ivt_init: initial IVT
        seed: random seed
        verbose: print progress

    Returns:
        (optimal_tilt, search_details)
    """
    if search_space is None:
        # Jump-only tilting is most effective for JD-OU:
        # - Diffusion tilt accumulates LR over ALL steps → weight degeneracy
        # - Jump tilting only penalizes at jump events → bounded LR
        search_space = {
            'mu': np.array([0.0, 0.05]),  # minimal diffusion tilt
            'lam_factor': np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
            'tau_factor': np.array([1.0, 1.5, 2.0, 2.5, 3.0]),
        }

    mu_vals = search_space['mu']
    lam_vals = search_space['lam_factor']
    tau_vals = search_space['tau_factor']

    results = []
    best_variance = np.inf
    best_tilt = TiltParameters()
    trial = 0

    for mu in mu_vals:
        for lam_f in lam_vals:
            for tau_f in tau_vals:
                tilt = TiltParameters(
                    mu_diffusion=mu,
                    lambda_factor=lam_f,
                    tau_factor=tau_f,
                )

                paths, log_lr = simulate_jdou_paths_is(
                    n_paths=n_paths_per,
                    n_steps=n_steps,
                    dt=dt,
                    params=params,
                    tilt=tilt,
                    ivt_init=ivt_init,
                    seed=seed + trial,
                )

                price, se, details = is_parisian_price(paths, log_lr, parisian_params)

                entry = {
                    'mu': mu, 'lam_factor': lam_f, 'tau_factor': tau_f,
                    'price': price, 'std_error': se,
                    'variance': details['is_variance'],
                    'trigger_rate_Q': details['trigger_rate_Q'],
                    'ess_ratio': details['ess_ratio'],
                }
                results.append(entry)

                if details['trigger_rate_Q'] > 0 and details['is_variance'] < best_variance:
                    # Also check ESS is not too low (weight degeneracy)
                    if details['ess_ratio'] > 0.01:
                        best_variance = details['is_variance']
                        best_tilt = tilt

                if verbose:
                    print(f"  {tilt} → price={price:.6f}, var={details['is_variance']:.2e}, "
                          f"trig_Q={details['trigger_rate_Q']:.3f}, ESS={details['ess_ratio']:.3f}")

                trial += 1

    search_details = {
        'results': results,
        'n_combinations': len(results),
        'optimal_tilt': best_tilt,
        'optimal_variance': best_variance,
    }

    return best_tilt, search_details


def compare_naive_vs_is(
    params: JDOUParameters,
    parisian_params: ParisianParams,
    n_steps: int,
    dt: float,
    tilt: TiltParameters = None,
    n_paths: int = 100000,
    ivt_init: float = None,
    seed: int = 42,
    auto_optimize: bool = True
) -> Dict:
    """
    Full comparison of naive MC vs IS-MC

    Args:
        params: JD-OU parameters
        parisian_params: Parisian condition parameters
        n_steps: time steps
        dt: time step size
        tilt: tilt parameters (if None, auto-optimizes)
        n_paths: number of MC paths
        ivt_init: initial IVT
        seed: random seed
        auto_optimize: whether to auto-optimize tilt if not provided

    Returns:
        Comparison dictionary with naive MC, IS-MC results, and variance reduction
    """
    from models.parisian import monte_carlo_parisian_price, compute_parisian_payoffs

    # === Naive MC ===
    paths_naive = simulate_jdou_paths(
        n_paths=n_paths,
        n_steps=n_steps,
        dt=dt,
        params=params,
        ivt_init=ivt_init,
        seed=seed,
    )

    price_naive, se_naive = monte_carlo_parisian_price(paths_naive, parisian_params)
    payoffs_naive = compute_parisian_payoffs(paths_naive, parisian_params)
    var_naive = float(np.var(payoffs_naive))
    trigger_rate_naive = float(np.mean(payoffs_naive > 0))

    # === Find optimal tilt if needed ===
    if tilt is None and auto_optimize:
        tilt, opt_details = find_optimal_tilt(
            params, parisian_params, n_steps, dt,
            n_paths_per=min(5000, n_paths // 10),
            ivt_init=ivt_init,
            seed=seed + 10000,
        )

    if tilt is None:
        tilt = compute_heuristic_tilt(params, parisian_params, dt)

    # === IS-MC ===
    paths_is, log_lr = simulate_jdou_paths_is(
        n_paths=n_paths,
        n_steps=n_steps,
        dt=dt,
        params=params,
        tilt=tilt,
        ivt_init=ivt_init,
        seed=seed + 20000,
    )

    price_is, se_is, is_details = is_parisian_price(paths_is, log_lr, parisian_params)

    # === Comparison ===
    var_is = is_details['is_variance']
    vr_ratio = var_naive / var_is if var_is > 0 else float('inf')

    return {
        'naive_mc': {
            'price': price_naive,
            'std_error': se_naive,
            'variance': var_naive,
            'trigger_rate': trigger_rate_naive,
            'n_paths': n_paths,
        },
        'is_mc': {
            'price': price_is,
            'std_error': se_is,
            'variance': var_is,
            'trigger_rate_Q': is_details['trigger_rate_Q'],
            'trigger_count': is_details['trigger_count'],
            'tilt': str(tilt),
            'ess': is_details['ess'],
            'ess_ratio': is_details['ess_ratio'],
            'price_self_normalized': is_details['price_self_normalized'],
            'n_paths': n_paths,
        },
        'comparison': {
            'variance_reduction_ratio': vr_ratio,
            'speedup_equivalent': vr_ratio,
            'price_difference': abs(price_naive - price_is),
            'relative_price_diff': abs(price_naive - price_is) / max(price_naive, 1e-10),
        },
        'params': {
            'barrier': parisian_params.barrier,
            'window': parisian_params.window,
            'n_steps': n_steps,
            'dt': dt,
        },
    }


def run_experiment_suite(
    params: JDOUParameters,
    configs: list = None,
    n_paths: int = 50000,
    seed: int = 42,
    verbose: bool = True,
) -> list:
    """
    Run comparison for multiple (barrier, window, n_steps) configurations

    Args:
        params: JD-OU model parameters
        configs: list of (barrier, window_steps, n_steps, dt) tuples
        n_paths: paths per experiment
        seed: base random seed
        verbose: print results

    Returns:
        List of comparison results
    """
    if configs is None:
        dt = 0.25  # 6 hours
        configs = [
            # (barrier, window_steps, n_steps, dt, label)
            (250,  8, 120, dt, "B=250, W=48h, T=30d"),
            (250, 16, 120, dt, "B=250, W=96h, T=30d"),
            (250,  8, 360, dt, "B=250, W=48h, T=90d"),
            (500,  8, 120, dt, "B=500, W=48h, T=30d"),
            (500,  8, 360, dt, "B=500, W=48h, T=90d"),
            (500, 16, 360, dt, "B=500, W=96h, T=90d"),
        ]

    results = []

    for idx, (barrier, window, n_steps, dt, label) in enumerate(configs):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Config: {label}")
            print(f"  Barrier={barrier} kg/m/s, Window={window} steps ({window*dt*24:.0f}h)")
            print(f"  Contract={n_steps} steps ({n_steps*dt:.0f} days)")
            print(f"{'='*60}")

        pp = ParisianParams(barrier=barrier, window=window)

        result = compare_naive_vs_is(
            params=params,
            parisian_params=pp,
            n_steps=n_steps,
            dt=dt,
            n_paths=n_paths,
            seed=seed + idx * 100000,
        )
        result['label'] = label
        results.append(result)

        if verbose:
            naive = result['naive_mc']
            is_mc = result['is_mc']
            comp = result['comparison']
            print(f"\n  Naive MC:  price={naive['price']:.6f} ± {naive['std_error']:.6f}  "
                  f"(trigger rate: {naive['trigger_rate']:.4f})")
            print(f"  IS-MC:     price={is_mc['price']:.6f} ± {is_mc['std_error']:.6f}  "
                  f"(trigger rate Q: {is_mc['trigger_rate_Q']:.4f})")
            print(f"  Tilt: {is_mc['tilt']}")
            print(f"  Variance Reduction: {comp['variance_reduction_ratio']:.1f}x")
            print(f"  ESS ratio: {is_mc['ess_ratio']:.3f}")

    return results


if __name__ == "__main__":
    # Load calibrated parameters
    try:
        ivt_data = np.load('D:/projects/qae-parisian-climate/data/raw/ivt_sf_1980_2023.npy')
        from models.jump_diffusion_ou import calibrate_jdou_parameters
        params = calibrate_jdou_parameters(ivt_data, dt=0.25)
        print("Loaded ERA5 calibrated parameters:")
    except FileNotFoundError:
        # Fallback to hardcoded calibration results
        params = JDOUParameters(
            kappa=0.767, theta=85.8, sigma=62.0,
            lam=0.108, mu_peak=390.5, sigma_peak=80.0, tau=0.69
        )
        print("Using hardcoded calibrated parameters:")

    print(f"  kappa={params.kappa:.3f}, theta={params.theta:.1f}, sigma={params.sigma:.1f}")
    print(f"  lambda={params.lam:.3f}/day, mu_peak={params.mu_peak:.1f}, tau={params.tau:.2f}d")

    # Run experiment suite
    print("\n" + "="*60)
    print("GLASSERMAN IS vs NAIVE MC COMPARISON")
    print("="*60)

    results = run_experiment_suite(
        params=params,
        n_paths=20000,  # Reduced for quick demo
        seed=42,
        verbose=True,
    )

    # Summary table
    print("\n\n" + "="*60)
    print("SUMMARY TABLE")
    print("="*60)
    print(f"{'Config':<30} {'Naive MC':>12} {'IS-MC':>12} {'VR Ratio':>10}")
    print("-" * 64)
    for r in results:
        label = r['label']
        p_naive = r['naive_mc']['price']
        p_is = r['is_mc']['price']
        vr = r['comparison']['variance_reduction_ratio']
        print(f"{label:<30} {p_naive:>12.6f} {p_is:>12.6f} {vr:>10.1f}x")
