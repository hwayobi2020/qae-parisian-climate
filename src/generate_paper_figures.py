"""
Generate all paper figures: convergence, scaling, parameter uncertainty.

Three-panel figure for the paper:
    (a) MC vs IQAE convergence (oracle calls vs error)
    (b) Scaling projection (K parameter sets × precision)
    (c) Price distribution under parameter uncertainty
"""

import numpy as np
import os
import sys
import time
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

sys.path.insert(0, os.path.dirname(__file__))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# ============================================================
# Figure 1: QAE Convergence (from quantum demo)
# ============================================================

def generate_figure1(save_dir: str):
    """MC vs IQAE convergence on 18-qubit Parisian oracle."""
    from quantum.qae_convergence_demo import convergence_comparison, exact_probability

    print("Generating Figure 1: QAE Convergence...")
    conv = convergence_comparison(
        barrier=6, window=2, n_steps=4,
        n_mc_repeats=50, qae_max_depth=32, n_shots_qae=100,
    )

    p_exact = conv['p_exact']
    mc = conv['mc']
    iqae = conv['iqae']

    fig, ax = plt.subplots(figsize=(8, 6))

    # MC
    mc_calls = [r['oracle_calls'] for r in mc]
    mc_err = [r['mean_error'] for r in mc]
    mc_std = [r['std_error'] for r in mc]
    ax.errorbar(mc_calls, mc_err, yerr=mc_std,
                fmt='o-', color='#1976D2', label='Monte Carlo',
                markersize=7, capsize=3, linewidth=2, zorder=3)

    # IQAE
    iq_calls = [r['total_oracle_calls'] for r in iqae if r['error'] > 1e-6]
    iq_err = [r['error'] for r in iqae if r['error'] > 1e-6]
    if iq_calls:
        ax.plot(iq_calls, iq_err, 's-', color='#D32F2F', label='IQAE (Quantum)',
                markersize=8, linewidth=2, zorder=3)

    # Theory lines
    x = np.logspace(1, 4.5, 200)
    c_mc = mc_err[2] * np.sqrt(mc_calls[2])
    ax.plot(x, c_mc / np.sqrt(x), '--', color='#1976D2', alpha=0.4,
            label=r'$O(N^{-1/2})$ theory', linewidth=1.5)
    if iq_calls:
        c_q = iq_err[1] * iq_calls[1]
        ax.plot(x, c_q / x, '--', color='#D32F2F', alpha=0.4,
                label=r'$O(N^{-1})$ theory', linewidth=1.5)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Total Oracle Calls', fontsize=14)
    ax.set_ylabel('Estimation Error $|\\hat{p} - p|$', fontsize=14)
    ax.set_title(f'Convergence: MC vs QAE (p = {p_exact:.2f})', fontsize=15)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_xlim(8, 2e4)

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig1_convergence.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return conv


# ============================================================
# Figure 2: Scaling Projections
# ============================================================

def generate_figure2(save_dir: str):
    """Scaling of MC vs QAE for parameter sweep."""
    print("Generating Figure 2: Scaling Projections...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Panel (a): Total oracle calls vs precision
    epsilons = np.logspace(-1, -4, 30)
    p = 0.05  # Rare event probability (5%)

    mc_calls = p * (1 - p) / epsilons**2
    qae_calls = np.pi / (4 * epsilons)

    ax1.plot(epsilons, mc_calls, '-', color='#1976D2', linewidth=2.5, label='MC: $O(1/\\varepsilon^2)$')
    ax1.plot(epsilons, qae_calls, '-', color='#D32F2F', linewidth=2.5, label='QAE: $O(1/\\varepsilon)$')

    # Fill region showing speedup
    ax1.fill_between(epsilons, qae_calls, mc_calls, alpha=0.1, color='green')
    ax1.annotate('Quadratic\nSpeedup', xy=(0.003, 5e4),
                 fontsize=12, ha='center', color='green', fontweight='bold')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Precision ε', fontsize=13)
    ax1.set_ylabel('Oracle Calls (per parameter set)', fontsize=13)
    ax1.set_title('(a) Single Parameter Set', fontsize=14)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3, which='both')
    ax1.invert_xaxis()

    # Panel (b): K parameter sets × precision
    K_values = [10, 100, 1000, 10000, 30000]
    eps_values = [0.01, 0.001, 0.0001]
    bar_width = 0.25

    for i, eps in enumerate(eps_values):
        mc_total = [K * p * (1-p) / eps**2 for K in K_values]
        qae_total = [K * np.pi / (4 * eps) for K in K_values]

        x = np.arange(len(K_values))
        ax2.bar(x + i * bar_width - bar_width, mc_total, bar_width * 0.9,
                label=f'MC ε={eps}' if i == 0 else f'ε={eps}',
                color=plt.cm.Blues(0.4 + i * 0.2), edgecolor='white')

    # Show time equivalents on right axis
    ax2.set_yscale('log')
    ax2.set_xlabel('Number of Parameter Sets (K)', fontsize=13)
    ax2.set_ylabel('Total Oracle Calls', fontsize=13)
    ax2.set_title('(b) Parameter Sweep: K × Precision', fontsize=14)
    ax2.set_xticks(np.arange(len(K_values)))
    ax2.set_xticklabels([f'{K:,}' for K in K_values], rotation=15)
    ax2.grid(True, alpha=0.3, axis='y')

    # Add time reference lines
    t_per_call = 1e-6  # 1 microsecond per MC call
    for t_label, t_sec in [('1 min', 60), ('1 hour', 3600), ('1 day', 86400), ('1 year', 3.15e7)]:
        calls_equiv = t_sec / t_per_call
        ax2.axhline(y=calls_equiv, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
        ax2.text(len(K_values) - 0.5, calls_equiv * 1.3, t_label,
                 fontsize=8, color='gray', ha='right')

    # Legend for epsilon values
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=plt.cm.Blues(0.4), label='ε = 0.01'),
        Patch(facecolor=plt.cm.Blues(0.6), label='ε = 0.001'),
        Patch(facecolor=plt.cm.Blues(0.8), label='ε = 0.0001'),
    ]
    ax2.legend(handles=legend_elements, fontsize=10, title='MC total calls')

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig2_scaling.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


# ============================================================
# Figure 3: Parameter Uncertainty Distribution
# ============================================================

def generate_figure3(save_dir: str):
    """Price distribution under parameter uncertainty from bootstrap."""
    print("Generating Figure 3: Parameter Uncertainty...")

    from models.jump_diffusion_ou import JDOUParameters, calibrate_jdou_parameters
    from classical.parameter_sweep import (
        generate_bootstrap_parameter_sets, batch_parameter_sweep
    )
    from models.parisian import ParisianParams

    data_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'raw', 'ivt_sf_1980_2023.npy'
    )

    if not os.path.exists(data_path):
        print("  No IVT data found, using parametric sweep")
        base = JDOUParameters(
            kappa=0.767, theta=85.8, sigma=62.0,
            lam=0.108, mu_peak=390.5, sigma_peak=80.0, tau=0.69
        )
        from classical.parameter_sweep import generate_parametric_sweep
        param_sets = generate_parametric_sweep(base, n_kappa=6, n_sigma=6, n_lambda=3)
    else:
        ivt_data = np.load(data_path)
        param_sets = generate_bootstrap_parameter_sets(ivt_data, K=100, seed=42)

    K = len(param_sets)

    # Run sweep for two barrier levels
    configs = [
        (ParisianParams(barrier=250, window=8), "B=250, W=48h (common)"),
        (ParisianParams(barrier=500, window=8), "B=500, W=48h (rare)"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel (a): Parameter distribution
    ax = axes[0]
    kappas = [p.kappa for p in param_sets]
    sigmas = [p.sigma for p in param_sets]
    lams = [p.lam for p in param_sets]
    sc = ax.scatter(kappas, sigmas, c=lams, cmap='YlOrRd', s=30, alpha=0.7, edgecolors='gray', linewidth=0.3)
    plt.colorbar(sc, ax=ax, label='λ (AR rate)')
    ax.set_xlabel('κ (mean reversion)', fontsize=12)
    ax.set_ylabel('σ (volatility)', fontsize=12)
    ax.set_title(f'(a) Bootstrap Parameters (K={K})', fontsize=13)
    ax.grid(True, alpha=0.3)

    # Panels (b,c): Price distributions
    for panel_idx, (pp, label) in enumerate(configs):
        ax = axes[panel_idx + 1]

        print(f"\n  Running sweep: {label}...")
        summary = batch_parameter_sweep(
            param_sets, pp,
            n_steps=120, dt=0.25,
            n_paths_mc=3000, n_paths_is=3000,
            seed=42, verbose=False,
        )

        prices_mc = [r.price_mc for r in summary['results']]
        prices_is = [r.price_is for r in summary['results']]

        ax.hist(prices_mc, bins=20, alpha=0.6, color='#1976D2', label='Naive MC', edgecolor='white')
        ax.hist(prices_is, bins=20, alpha=0.6, color='#D32F2F', label='IS-MC', edgecolor='white')

        mean_mc = np.mean(prices_mc)
        p95_mc = np.percentile(prices_mc, 95)
        ax.axvline(mean_mc, color='#1976D2', linestyle='--', linewidth=1.5)
        ax.axvline(p95_mc, color='red', linestyle=':', linewidth=1.5,
                   label=f'95th = {p95_mc:.4f}')

        ax.set_xlabel('Trigger Probability', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        panel_letter = chr(98 + panel_idx)  # b, c
        ax.set_title(f'({panel_letter}) {label}', fontsize=13)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

        # Print scaling info
        avg_time = summary['timing']['avg_mc_per_set']
        print(f"    Mean price: {mean_mc:.4f}, 95th: {p95_mc:.4f}")
        print(f"    Avg time/set: {avg_time:.3f}s")
        print(f"    Scaling (K={K}, ε=0.001):")
        print(f"      MC:  {K * mean_mc*(1-mean_mc)/0.001**2:.0f} total calls")
        print(f"      QAE: {K * np.pi/(4*0.001):.0f} total calls")

    plt.tight_layout()
    path = os.path.join(save_dir, 'fig3_parameter_uncertainty.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")


# ============================================================
# Summary Table
# ============================================================

def print_summary_table():
    """Print the 4-method summary for the paper."""
    print(f"\n{'='*80}")
    print("PAPER SUMMARY: 4-METHOD COMPARISON")
    print(f"{'='*80}")

    print("""
    ┌─────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
    │  Method     │  Per-set     │  K=100       │  K=1000      │  K=30,000    │
    │             │  (ε=0.001)   │              │              │              │
    ├─────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
    │  Naive MC   │  50,000      │  5,000,000   │  50,000,000  │  1,500,000,000│
    │  IS-MC      │  10,000*     │  1,000,000   │  10,000,000  │  300,000,000 │
    │  QAE        │  785         │  78,500      │  785,000     │  23,550,000  │
    │  IS-QAE     │  157*        │  15,700      │  157,000     │  4,710,000   │
    ├─────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
    │  Speedup    │  MC/QAE=64x  │              │              │              │
    │  (calls)    │  MC/IS-QAE   │              │              │              │
    │             │  =318x       │              │              │              │
    └─────────────┴──────────────┴──────────────┴──────────────┴──────────────┘

    * IS reduces variance ~5x for rare events (VR ratio from Glasserman IS)

    Time estimates (1μs per MC call, K=30,000, ε=0.001):
    ┌─────────────┬──────────┬──────────┐
    │  Method     │  Calls   │  Time    │
    ├─────────────┼──────────┼──────────┤
    │  Naive MC   │  1.5B    │  25 min  │
    │  IS-MC      │  300M    │  5 min   │
    │  QAE*       │  23.6M   │  24 sec  │
    │  IS-QAE*    │  4.7M    │  5 sec   │
    └─────────────┴──────────┴──────────┘
    * QAE time assumes quantum hardware with ~1μs oracle call
    """)

    print("\n  Key insight: QAE advantage = K × (1/ε² → 1/ε)")
    print("  The larger K and smaller ε, the greater the quantum advantage.")
    print("  Parameter uncertainty (bootstrap) naturally creates large K.")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    save_dir = os.path.join(os.path.dirname(__file__), '..', 'figures')
    os.makedirs(save_dir, exist_ok=True)

    t0 = time.time()

    # Figure 1: QAE convergence
    generate_figure1(save_dir)

    # Figure 2: Scaling projections
    generate_figure2(save_dir)

    # Figure 3: Parameter uncertainty
    generate_figure3(save_dir)

    # Summary table
    print_summary_table()

    print(f"\nTotal time: {time.time() - t0:.1f}s")
    print(f"All figures saved to: {save_dir}")
