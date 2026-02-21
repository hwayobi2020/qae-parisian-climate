"""
Markov Chain Exact Solver for Parisian Trigger Probability

Discretizes O-U (or simplified JD-OU) process into a
(IVT_level, consecutive_counter) Markov chain.

Computes exact trigger probability via transition matrix powers.
Demonstrates numerical precision issues for rare events.

State space:
    - S = K * W + 1
    - K: number of IVT discretization bins
    - W: counter values {0, 1, ..., W-1}
    - Plus 1 absorbing "triggered" state

Transition rules:
    From (level, count):
        If next_level >= barrier_bin:
            If count + 1 >= W: -> absorbing state
            Else: -> (next_level, count + 1)
        If next_level < barrier_bin:
            -> (next_level, 0)

References:
    - Chesney et al. (1997): Parisian barrier options
    - Glasserman (2003): Monte Carlo Methods in Financial Engineering
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.jump_diffusion_ou import JDOUParameters


@dataclass
class MarkovChainConfig:
    """Configuration for Markov chain discretization"""
    n_levels: int = 256         # Number of IVT bins (K)
    ivt_min: float = 0.0        # Minimum IVT
    ivt_max: float = 1000.0     # Maximum IVT
    barrier: float = 250.0      # IVT threshold (kg/m/s)
    window: int = 8             # Consecutive steps required for trigger
    dt: float = 0.25            # Time step (days)

    @property
    def bin_width(self):
        return (self.ivt_max - self.ivt_min) / self.n_levels

    @property
    def bin_centers(self):
        return np.linspace(
            self.ivt_min + self.bin_width / 2,
            self.ivt_max - self.bin_width / 2,
            self.n_levels
        )

    @property
    def barrier_bin(self):
        """First bin index that is >= barrier"""
        centers = self.bin_centers
        above = np.where(centers >= self.barrier)[0]
        return above[0] if len(above) > 0 else self.n_levels

    @property
    def state_space_size(self):
        return self.n_levels * self.window + 1  # +1 for absorbing state

    @property
    def absorbing_idx(self):
        return self.n_levels * self.window  # Last index


def state_to_idx(level: int, counter: int, config: MarkovChainConfig) -> int:
    """Convert (level, counter) to flat index"""
    return counter * config.n_levels + level


def idx_to_state(idx: int, config: MarkovChainConfig) -> Tuple[int, int]:
    """Convert flat index to (level, counter)"""
    if idx == config.absorbing_idx:
        return (-1, -1)  # Absorbing state
    counter = idx // config.n_levels
    level = idx % config.n_levels
    return (level, counter)


def build_ou_transition_probs(
    config: MarkovChainConfig,
    kappa: float,
    theta: float,
    sigma: float,
) -> np.ndarray:
    """
    Build IVT level-to-level transition probabilities for pure O-U

    P(level' | level) via Gaussian CDF of O-U increment

    For each current level l with center x_l:
        Next IVT ~ N(x_l + kappa*(theta - x_l)*dt, sigma^2*dt)
        P(level' = l') = Phi(bin_max(l') - mu) / sigma_step) - Phi(bin_min(l') - mu) / sigma_step)

    Args:
        config: Markov chain configuration
        kappa: mean reversion speed
        theta: long-term mean
        sigma: volatility

    Returns:
        (K, K) transition probability matrix
    """
    K = config.n_levels
    centers = config.bin_centers
    bw = config.bin_width
    sigma_step = sigma * np.sqrt(config.dt)

    P_level = np.zeros((K, K))

    for l in range(K):
        x_l = centers[l]
        mu_next = x_l + kappa * (theta - x_l) * config.dt

        # CDF at bin edges
        edges = np.linspace(config.ivt_min, config.ivt_max, K + 1)
        cdf_vals = norm.cdf(edges, loc=mu_next, scale=sigma_step)

        # Probability of landing in each bin
        probs = np.diff(cdf_vals)

        # Handle boundary: probability below min goes to bin 0
        probs[0] += norm.cdf(config.ivt_min, loc=mu_next, scale=sigma_step)
        # Probability above max goes to last bin
        probs[-1] += 1 - norm.cdf(config.ivt_max, loc=mu_next, scale=sigma_step)

        P_level[l] = probs

    return P_level


def build_jdou_transition_probs(
    config: MarkovChainConfig,
    params: JDOUParameters,
) -> np.ndarray:
    """
    Build IVT level-to-level transition probabilities for JD-OU

    Mixture model:
        P(l' | l) = (1 - lambda*dt) * P_ou(l' | l) + lambda*dt * P_jump(l')

    where P_jump is the probability of landing in bin l' after a jump
    (approximation: jump directly samples from peak distribution)

    Args:
        config: Markov chain configuration
        params: JD-OU parameters

    Returns:
        (K, K) transition probability matrix
    """
    K = config.n_levels

    # O-U component
    P_ou = build_ou_transition_probs(
        config, params.kappa, params.theta, params.sigma
    )

    # Jump component: distribution of IVT after an AR event starts
    edges = np.linspace(config.ivt_min, config.ivt_max, K + 1)
    cdf_jump = norm.cdf(edges, loc=params.mu_peak, scale=params.sigma_peak)
    P_jump_vec = np.diff(cdf_jump)
    P_jump_vec[0] += norm.cdf(config.ivt_min, loc=params.mu_peak, scale=params.sigma_peak)
    P_jump_vec[-1] += 1 - norm.cdf(config.ivt_max, loc=params.mu_peak, scale=params.sigma_peak)

    # Mixture
    lam_dt = params.lam * config.dt
    P_level = (1 - lam_dt) * P_ou + lam_dt * P_jump_vec[np.newaxis, :]

    return P_level


def build_transition_matrix(
    config: MarkovChainConfig,
    P_level: np.ndarray,
) -> np.ndarray:
    """
    Build full Markov chain transition matrix including counter logic

    State space: K*W states (level, counter) + 1 absorbing

    Args:
        config: Markov chain configuration
        P_level: (K, K) level-to-level transition probabilities

    Returns:
        (S, S) transition matrix where S = K*W + 1
    """
    K = config.n_levels
    W = config.window
    S = config.state_space_size
    barrier_bin = config.barrier_bin

    M = np.zeros((S, S))

    # Absorbing state stays
    absorb = config.absorbing_idx
    M[absorb, absorb] = 1.0

    for c in range(W):
        for l in range(K):
            src = state_to_idx(l, c, config)

            for l_next in range(K):
                p = P_level[l, l_next]
                if p < 1e-15:
                    continue

                if l_next >= barrier_bin:
                    # Above barrier
                    c_next = c + 1
                    if c_next >= W:
                        # Trigger! -> absorbing
                        M[src, absorb] += p
                    else:
                        dst = state_to_idx(l_next, c_next, config)
                        M[src, dst] += p
                else:
                    # Below barrier -> reset counter
                    dst = state_to_idx(l_next, 0, config)
                    M[src, dst] += p

    return M


def compute_trigger_probability_naive(
    M: np.ndarray,
    n_steps: int,
    init_state_idx: int,
    config: MarkovChainConfig,
) -> float:
    """
    Compute trigger probability via matrix power (naive method)

    P(trigger in n_steps) = (M^n_steps)[init, absorbing]

    WARNING: For rare events, this suffers from catastrophic cancellation
    because P(trigger) = 1 - P(no trigger) where P(no trigger) ~ 1.

    Args:
        M: transition matrix
        n_steps: number of steps
        init_state_idx: initial state index
        config: Markov chain config

    Returns:
        Trigger probability
    """
    S = config.state_space_size
    absorb = config.absorbing_idx

    # Compute M^n_steps via repeated squaring
    p = np.zeros(S)
    p[init_state_idx] = 1.0

    # Power iteration step by step for stability tracking
    for _ in range(n_steps):
        p = p @ M

    return p[absorb]


def compute_trigger_probability_stable(
    M: np.ndarray,
    n_steps: int,
    init_state_idx: int,
    config: MarkovChainConfig,
) -> Tuple[float, np.ndarray]:
    """
    Compute trigger probability with numerical stability

    Instead of computing P(trigger) = [M^T]_{init, absorb},
    accumulate probability flow INTO the absorbing state at each step.

    P(trigger) = sum_{t=1}^{T} P(enter absorbing at step t)

    This avoids the catastrophic cancellation in 1 - P(no trigger).

    Args:
        M: transition matrix
        n_steps: number of steps
        init_state_idx: initial state index
        config: Markov chain config

    Returns:
        (trigger_probability, cumulative_trigger_by_step)
    """
    S = config.state_space_size
    absorb = config.absorbing_idx

    # Build sub-transition matrix (non-absorbing states only)
    non_absorb = list(range(absorb))  # 0 to S-2
    M_sub = M[np.ix_(non_absorb, non_absorb)]

    # Transition to absorbing state
    absorb_col = M[non_absorb, absorb]

    # Initial distribution (non-absorbing part)
    p = np.zeros(len(non_absorb))
    if init_state_idx < absorb:
        p[init_state_idx] = 1.0

    trigger_total = 0.0
    trigger_by_step = np.zeros(n_steps)

    for t in range(n_steps):
        # Probability of entering absorbing state at this step
        p_absorb = np.dot(p, absorb_col)
        trigger_total += p_absorb
        trigger_by_step[t] = trigger_total

        # Update non-absorbing distribution
        p = p @ M_sub

    return trigger_total, trigger_by_step


def demonstrate_precision_issues(
    params: JDOUParameters,
    configs: list = None,
    verbose: bool = True,
) -> Dict:
    """
    Demonstrate numerical precision issues of Markov chain
    for rare-event Parisian trigger probabilities

    Compares naive vs stable computation methods and shows
    precision loss for increasingly rare events.

    Args:
        params: JD-OU parameters
        configs: list of (barrier, window, n_steps, label) tuples
        verbose: print results

    Returns:
        Results dictionary
    """
    if configs is None:
        configs = [
            # (barrier, window_steps, n_steps, label)
            (250,  8,  120, "B=250, W=48h, T=30d (common)"),
            (250, 16,  120, "B=250, W=96h, T=30d"),
            (500,  8,  120, "B=500, W=48h, T=30d"),
            (500,  8,  360, "B=500, W=48h, T=90d"),
            (500, 16,  360, "B=500, W=96h, T=90d (rare)"),
            (700,  8,  120, "B=700, W=48h, T=30d (very rare)"),
        ]

    results = []
    dt = 0.25

    if verbose:
        print(f"\n{'='*70}")
        print("MARKOV CHAIN PRECISION ANALYSIS")
        print(f"{'='*70}")
        print(f"  Model: JD-OU with kappa={params.kappa:.3f}, theta={params.theta:.1f}")
        print(f"  Discretization: K=128 bins, IVT range [0, 1000]")
        print()

    for barrier, window, n_steps, label in configs:
        mc_config = MarkovChainConfig(
            n_levels=128,  # Moderate resolution for speed
            ivt_min=0, ivt_max=1000,
            barrier=barrier,
            window=window,
            dt=dt,
        )

        # Build transition matrix (with jump component)
        P_level = build_jdou_transition_probs(mc_config, params)
        M = build_transition_matrix(mc_config, P_level)

        # Initial state: IVT = theta, counter = 0
        init_level = np.argmin(np.abs(mc_config.bin_centers - params.theta))
        init_idx = state_to_idx(init_level, 0, mc_config)

        # Method 1: Naive (M^T)
        p_naive = compute_trigger_probability_naive(
            M, n_steps, init_idx, mc_config
        )

        # Method 2: Stable (accumulated flow)
        p_stable, p_by_step = compute_trigger_probability_stable(
            M, n_steps, init_idx, mc_config
        )

        # Precision metrics
        if p_stable > 0:
            relative_error = abs(p_naive - p_stable) / p_stable
        else:
            relative_error = 0.0

        entry = {
            'label': label,
            'barrier': barrier,
            'window': window,
            'n_steps': n_steps,
            'p_naive': p_naive,
            'p_stable': p_stable,
            'relative_error': relative_error,
            'state_space_size': mc_config.state_space_size,
            'log10_prob': np.log10(max(p_stable, 1e-300)),
        }
        results.append(entry)

        if verbose:
            print(f"  {label}")
            print(f"    State space: {mc_config.state_space_size:,} states")
            print(f"    P(trigger) naive:  {p_naive:.10e}")
            print(f"    P(trigger) stable: {p_stable:.10e}")
            print(f"    Relative error:    {relative_error:.2e}")
            if p_stable > 0:
                print(f"    log10(P):          {np.log10(p_stable):.2f}")
            print()

    return {
        'results': results,
        'conclusion': _precision_conclusion(results),
    }


def _precision_conclusion(results: list) -> str:
    """Generate conclusion about precision issues"""
    rare = [r for r in results if r['p_stable'] < 0.01]
    if not rare:
        return "No significant precision issues detected for these configurations."

    worst = max(rare, key=lambda r: r['relative_error'])
    return (
        f"For rare events (P < 1%), Markov chain precision degrades. "
        f"Worst case: {worst['label']} with relative error {worst['relative_error']:.2e}. "
        f"This motivates Monte Carlo with importance sampling for rare-event pricing."
    )


if __name__ == "__main__":
    # Load or use default parameters
    params = JDOUParameters(
        kappa=0.767, theta=85.8, sigma=62.0,
        lam=0.108, mu_peak=390.5, sigma_peak=80.0, tau=0.69
    )

    print("JD-OU Parameters:")
    print(f"  kappa={params.kappa:.3f}, theta={params.theta:.1f}, sigma={params.sigma:.1f}")
    print(f"  lambda={params.lam:.3f}/day, mu_peak={params.mu_peak:.1f}, tau={params.tau:.2f}d")

    # Run precision analysis
    result = demonstrate_precision_issues(params)

    print(f"\nConclusion: {result['conclusion']}")

    # Also show a simple example
    print(f"\n{'='*70}")
    print("SINGLE CONFIGURATION DETAIL")
    print(f"{'='*70}")

    mc_config = MarkovChainConfig(
        n_levels=128, ivt_min=0, ivt_max=1000,
        barrier=250, window=8, dt=0.25,
    )

    P_level = build_jdou_transition_probs(mc_config, params)
    M = build_transition_matrix(mc_config, P_level)

    init_level = np.argmin(np.abs(mc_config.bin_centers - params.theta))
    init_idx = state_to_idx(init_level, 0, mc_config)

    # Compute for increasing contract durations
    print(f"\nB=250, W=48h: Trigger probability vs contract duration")
    print(f"{'Duration':<15} {'P(trigger)':<15} {'Cumulative':<15}")
    print("-" * 45)

    for T_days in [7, 14, 30, 60, 90, 180, 365]:
        n_steps = int(T_days / 0.25)
        p, _ = compute_trigger_probability_stable(M, n_steps, init_idx, mc_config)
        print(f"{T_days:>3}d ({n_steps:>5} steps)  {p:>12.6f}")
