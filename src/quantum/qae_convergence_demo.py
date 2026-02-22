"""
QAE Convergence Demo: MC vs QAE Oracle Call Count Comparison

Demonstrates quadratic speedup on the 18-qubit Parisian oracle:
    MC:  O(1/ε²) samples for precision ε
    QAE: O(1/ε)  oracle calls for precision ε

Uses Iterative QAE (IQAE) from Grinko et al. (2021) on Qiskit Aer.
Wall-clock time favors MC on simulator (classical overhead of quantum simulation),
but oracle CALL COUNT clearly shows the quadratic advantage.

Key design choice: barrier=6 (p=0.25) instead of barrier=4 (p=0.5)
because p=0.5 is a degenerate case where Grover amplification oscillates.

References:
    - Brassard et al. (2002): Quantum Amplitude Amplification
    - Grinko et al. (2021): Iterative QAE without QPE
    - Stamatopoulos et al. (2020): Option pricing using quantum computers
"""

import numpy as np
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import GroverOperator
from qiskit_aer import AerSimulator

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from quantum.parisian_oracle import build_comparator, build_counter_update, build_or_latch


# ============================================================
# 1. Build Oracle + State Preparation for QAE
# ============================================================

def build_qae_circuit(
    n_ivt: int = 3,
    n_counter: int = 3,
    barrier: int = 6,
    window: int = 2,
    n_steps: int = 4,
) -> Tuple[QuantumCircuit, int]:
    """
    Build the A operator (state preparation + oracle) for QAE.

    A|0> = sqrt(1-p)|bad>|0> + sqrt(p)|good>|1>

    where good = states where Parisian condition is triggered.
    Static IVT in uniform superposition; flag qubit marks triggered states.

    Args:
        n_ivt: IVT qubits (3 = 8 levels)
        n_counter: Counter qubits
        barrier: IVT threshold level (6 -> p=2/8=0.25)
        window: Consecutive steps for trigger
        n_steps: Number of time steps

    Returns:
        (A_circuit, flag_qubit_index)
    """
    ivt = QuantumRegister(n_ivt, 'ivt')
    counter = QuantumRegister(n_counter, 'counter')
    above = QuantumRegister(1, 'above')
    flag = QuantumRegister(1, 'parisian')
    cmp_borrow = QuantumRegister(n_ivt, 'cmp_borr')
    cnt_scratch = QuantumRegister(n_counter, 'cnt_scratch')
    trigger = QuantumRegister(1, 'trigger')
    latch_borrow = QuantumRegister(n_counter, 'latch_borr')

    A = QuantumCircuit(ivt, counter, above, flag, cmp_borrow,
                       cnt_scratch, trigger, latch_borrow,
                       name='A_operator')

    # State preparation: uniform superposition over IVT
    for q in range(n_ivt):
        A.h(ivt[q])

    # Oracle: apply Parisian condition check over n_steps
    for step in range(n_steps):
        cmp = build_comparator(n_ivt, barrier)
        A.append(cmp, ivt[:] + cmp_borrow[:] + above[:])

        cnt_upd = build_counter_update(n_counter)
        A.append(cnt_upd, counter[:] + above[:] + cnt_scratch[:])

        latch = build_or_latch(n_counter, window)
        A.append(latch, counter[:] + flag[:] + trigger[:] + latch_borrow[:])

        A.append(cmp.inverse(), ivt[:] + cmp_borrow[:] + above[:])

    # Flag qubit index in the global circuit
    flag_idx = n_ivt + n_counter + 1  # after ivt, counter, above

    return A, flag_idx


# ============================================================
# 2. Exact and MC probability
# ============================================================

def exact_probability(
    n_ivt: int = 3, barrier: int = 6, window: int = 2, n_steps: int = 4,
) -> float:
    """Exact trigger probability for static IVT, uniform distribution."""
    n_levels = 2 ** n_ivt
    if n_steps >= window:
        return (n_levels - barrier) / n_levels
    return 0.0


def mc_estimate(
    n_ivt: int, barrier: int, window: int, n_steps: int,
    n_samples: int, seed: int = 42,
) -> float:
    """Classical MC estimate of trigger probability."""
    rng = np.random.RandomState(seed)
    n_levels = 2 ** n_ivt
    ivt_vals = rng.randint(0, n_levels, size=n_samples)
    triggered = (ivt_vals >= barrier).astype(float)
    return float(np.mean(triggered)) if n_steps >= window else 0.0


# ============================================================
# 3. QAE measurement
# ============================================================

def qae_measure(
    n_ivt: int = 3, n_counter: int = 3,
    barrier: int = 6, window: int = 2, n_steps: int = 4,
    n_grover_iters: int = 0, n_shots: int = 100,
) -> Tuple[float, int]:
    """
    Single QAE round: apply A, then Q^m, measure flag.

    Returns:
        (fraction_of_1s, oracle_calls_per_shot)
    """
    A, flag_idx = build_qae_circuit(n_ivt, n_counter, barrier, window, n_steps)
    n_qubits = A.num_qubits

    qc = QuantumCircuit(n_qubits, 1)
    qc.append(A, range(n_qubits))

    if n_grover_iters > 0:
        oracle_circ = QuantumCircuit(n_qubits, name='S_chi')
        oracle_circ.z(flag_idx)
        Q = GroverOperator(oracle=oracle_circ, state_preparation=A)
        for _ in range(n_grover_iters):
            qc.append(Q, range(n_qubits))

    qc.measure(flag_idx, 0)

    backend = AerSimulator()
    qc_t = qc.decompose(reps=5)
    counts = backend.run(qc_t, shots=n_shots).result().get_counts()

    p_flag = counts.get('1', 0) / n_shots
    oracle_calls = 1 + 2 * n_grover_iters

    return p_flag, oracle_calls


# ============================================================
# 4. IQAE (Iterative QAE)
# ============================================================

def iqae_estimate(
    n_ivt: int = 3, n_counter: int = 3,
    barrier: int = 6, window: int = 2, n_steps: int = 4,
    max_depth: int = 32, n_shots_per_round: int = 100,
) -> Dict:
    """
    Iterative QAE: run increasing Grover depths, combine via ML.

    Schedule: m = 0, 1, 2, 4, 8, 16, ...
    At each depth, measure P(flag=1) and accumulate data.
    Use maximum likelihood across all rounds to estimate theta.

    Returns dict with convergence trace (oracle calls vs error at each round).
    """
    p_exact = exact_probability(n_ivt, barrier, window, n_steps)

    # Grover depth schedule
    depths = [0]
    m = 1
    while m <= max_depth:
        depths.append(m)
        m *= 2

    all_data = []          # (depth, n_1s, n_shots)
    trace = []             # convergence trace
    total_oracle_calls = 0

    for m in depths:
        p_flag, oc_per_shot = qae_measure(
            n_ivt, n_counter, barrier, window, n_steps,
            n_grover_iters=m, n_shots=n_shots_per_round,
        )
        n_1s = int(round(p_flag * n_shots_per_round))
        total_oracle_calls += oc_per_shot * n_shots_per_round

        all_data.append((m, n_1s, n_shots_per_round))

        # ML estimate from all data so far
        theta_est = _ml_theta(all_data)
        p_est = np.sin(theta_est) ** 2
        error = abs(p_est - p_exact)

        trace.append({
            'depth': m,
            'p_flag': float(p_flag),
            'p_est': float(p_est),
            'error': float(error),
            'total_oracle_calls': total_oracle_calls,
        })

    return {
        'p_exact': float(p_exact),
        'trace': trace,
    }


def _ml_theta(data: List[Tuple[int, int, int]]) -> float:
    """ML estimate of theta from list of (depth, n_successes, n_trials)."""
    from scipy.optimize import minimize_scalar

    def neg_ll(theta):
        ll = 0.0
        for m, n1, n_tot in data:
            k = 2 * m + 1
            p = np.sin(k * theta) ** 2
            p = np.clip(p, 1e-12, 1 - 1e-12)
            ll += n1 * np.log(p) + (n_tot - n1) * np.log(1 - p)
        return -ll

    res = minimize_scalar(neg_ll, bounds=(0.001, np.pi / 2 - 0.001), method='bounded')
    return res.x


# ============================================================
# 5. Convergence Comparison
# ============================================================

def convergence_comparison(
    barrier: int = 6,
    window: int = 2,
    n_steps: int = 4,
    n_ivt: int = 3,
    n_counter: int = 3,
    mc_sizes: List[int] = None,
    n_mc_repeats: int = 50,
    qae_max_depth: int = 32,
    n_shots_qae: int = 100,
    seed: int = 42,
) -> Dict:
    """
    Compare MC vs IQAE convergence: error vs total oracle calls.

    Paper figure:
    - X-axis: total oracle calls (log)
    - Y-axis: estimation error (log)
    - MC:  slope ~ -1/2  (O(1/sqrt(N)))
    - IQAE: slope ~ -1   (O(1/N))
    """
    p_exact = exact_probability(n_ivt, barrier, window, n_steps)

    if mc_sizes is None:
        mc_sizes = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]

    print(f"\nExact probability: {p_exact:.4f}")
    print(f"Config: n_ivt={n_ivt}, barrier={barrier}, window={window}, steps={n_steps}")

    # --- MC ---
    print("\n  MC convergence:")
    mc_results = []
    for N in mc_sizes:
        errors = []
        for rep in range(n_mc_repeats):
            p_hat = mc_estimate(n_ivt, barrier, window, n_steps, N, seed=seed + rep * 7)
            errors.append(abs(p_hat - p_exact))
        mean_err = float(np.mean(errors))
        mc_results.append({
            'oracle_calls': N,
            'mean_error': mean_err,
            'std_error': float(np.std(errors)),
        })
        print(f"    N={N:>6}: error={mean_err:.4f}")

    # --- IQAE ---
    print("\n  IQAE convergence:")
    iqae_res = iqae_estimate(
        n_ivt, n_counter, barrier, window, n_steps,
        max_depth=qae_max_depth, n_shots_per_round=n_shots_qae,
    )
    for t in iqae_res['trace']:
        print(f"    depth={t['depth']:>3}: p_est={t['p_est']:.4f}, "
              f"error={t['error']:.4f}, calls={t['total_oracle_calls']}")

    return {
        'p_exact': p_exact,
        'mc': mc_results,
        'iqae': iqae_res['trace'],
    }


# ============================================================
# 6. 4-Method Comparison
# ============================================================

def four_method_comparison(
    barrier: int = 6, window: int = 2, n_steps: int = 4,
    n_ivt: int = 3, n_counter: int = 3,
    n_mc: int = 10000, n_is: int = 10000,
    qae_max_depth: int = 16, n_shots_qae: int = 200,
    seed: int = 42,
) -> Dict:
    """
    4-method comparison: Naive MC, IS-MC, QAE, IS-QAE
    """
    p_exact = exact_probability(n_ivt, barrier, window, n_steps)
    n_levels = 2 ** n_ivt

    print(f"\n{'='*60}")
    print(f"4-METHOD COMPARISON")
    print(f"  p_exact = {p_exact:.4f} (barrier={barrier}/{n_levels})")
    print(f"{'='*60}")

    results = {'p_exact': p_exact}

    # 1. Naive MC
    t0 = time.time()
    p_mc = mc_estimate(n_ivt, barrier, window, n_steps, n_mc, seed)
    t1 = time.time()
    results['naive_mc'] = {
        'price': p_mc, 'error': abs(p_mc - p_exact),
        'oracle_calls': n_mc, 'wall_time': t1 - t0,
    }

    # 2. IS-MC (tilt uniform distribution toward above-barrier)
    t0 = time.time()
    rng = np.random.RandomState(seed + 1000)
    q_high = 0.7  # tilt probability
    n_above = n_levels - barrier
    n_below = barrier
    payoffs = np.zeros(n_is)
    for i in range(n_is):
        if rng.random() < q_high:
            ivt_val = rng.randint(barrier, n_levels)
            lr = (1.0 / n_levels) / (q_high / n_above)
        else:
            ivt_val = rng.randint(0, barrier)
            lr = (1.0 / n_levels) / ((1 - q_high) / n_below)
        triggered = (ivt_val >= barrier) and (n_steps >= window)
        payoffs[i] = float(triggered) * lr
    p_is = float(np.mean(payoffs))
    se_is = float(np.std(payoffs) / np.sqrt(n_is))
    var_is = float(np.var(payoffs))
    t1 = time.time()

    # Naive MC variance for VR calculation
    naive_payoffs = ((np.random.RandomState(seed).randint(0, n_levels, n_mc) >= barrier)
                     & (n_steps >= window)).astype(float)
    var_naive = float(np.var(naive_payoffs))
    vr_ratio = var_naive / var_is if var_is > 0 else float('inf')

    results['is_mc'] = {
        'price': p_is, 'error': abs(p_is - p_exact),
        'oracle_calls': n_is, 'wall_time': t1 - t0,
        'vr_ratio': vr_ratio,
    }

    # 3. QAE (IQAE)
    t0 = time.time()
    iqae_res = iqae_estimate(
        n_ivt, n_counter, barrier, window, n_steps,
        max_depth=qae_max_depth, n_shots_per_round=n_shots_qae,
    )
    t1 = time.time()
    final = iqae_res['trace'][-1]
    results['qae'] = {
        'price': final['p_est'], 'error': final['error'],
        'oracle_calls': final['total_oracle_calls'], 'wall_time': t1 - t0,
    }

    # 4. IS-QAE: IS-tilted state prep + IQAE + classical correction
    # Grover-Rudolph circuit loads tilted distribution q(x) with
    # more amplitude on above-barrier states (p_IS >> p).
    # Then IQAE estimates p_IS, and classical correction gives p = p_IS * c.
    from quantum.is_qae import is_qae_full_pipeline
    t0 = time.time()
    is_qae_result = is_qae_full_pipeline(
        n_ivt=n_ivt, n_counter=n_counter,
        barrier=barrier, window=window, n_steps=n_steps,
        q_high=0.7,
        qae_max_depth=qae_max_depth,
        n_shots_qae=n_shots_qae,
        n_correction_samples=n_is,
        seed=seed + 2000,
    )
    t1 = time.time()
    results['is_qae'] = {
        'price': is_qae_result.p_final,
        'error': abs(is_qae_result.p_final - p_exact),
        'oracle_calls': is_qae_result.total_oracle_calls,
        'wall_time': t1 - t0,
        'p_is': is_qae_result.p_is,
        'correction': is_qae_result.correction_factor,
    }

    # Print table
    print(f"\n  {'Method':<12} {'Estimate':>10} {'Error':>10} {'Calls':>10} {'Time':>10}")
    print(f"  {'-'*52}")
    for method, key in [('Naive MC', 'naive_mc'), ('IS-MC', 'is_mc'),
                         ('QAE', 'qae'), ('IS-QAE', 'is_qae')]:
        r = results[key]
        print(f"  {method:<12} {r['price']:>10.4f} {r['error']:>10.4f} "
              f"{r['oracle_calls']:>10} {r['wall_time']:>10.3f}s")

    print(f"\n  Wall-clock: MC is faster (simulator overhead).")
    print(f"  Oracle calls: QAE uses ~{results['naive_mc']['oracle_calls'] // max(results['qae']['oracle_calls'], 1)}x fewer calls than MC.")

    return results


# ============================================================
# 7. Plotting
# ============================================================

def plot_convergence(results: Dict, save_path: str = None):
    """Plot MC vs QAE convergence (key paper figure)."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        matplotlib.rcParams['font.size'] = 12
    except ImportError:
        print("matplotlib not available")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    p_exact = results['p_exact']

    # Left: Error vs Oracle Calls (log-log)
    mc = results['mc']
    iqae = results['iqae']

    mc_calls = [r['oracle_calls'] for r in mc]
    mc_errors = [r['mean_error'] for r in mc]
    mc_stds = [r['std_error'] for r in mc]
    ax1.errorbar(mc_calls, mc_errors, yerr=mc_stds,
                 fmt='o-', color='#2196F3', label='Monte Carlo',
                 markersize=6, capsize=3, linewidth=2)

    iqae_calls = [r['total_oracle_calls'] for r in iqae if r['error'] > 1e-6]
    iqae_errors = [r['error'] for r in iqae if r['error'] > 1e-6]
    if iqae_calls:
        ax1.plot(iqae_calls, iqae_errors, 's-', color='#F44336',
                 label='IQAE (Quantum)', markersize=7, linewidth=2)

    # Theoretical slopes
    x = np.logspace(1, 4.5, 100)
    c_mc = mc_errors[0] * np.sqrt(mc_calls[0])
    c_qae = iqae_errors[0] * iqae_calls[0] if iqae_calls else 1.0
    ax1.plot(x, c_mc / np.sqrt(x), ':', color='#2196F3', alpha=0.5,
             label=r'$O(1/\sqrt{N})$')
    ax1.plot(x, c_qae / x, ':', color='#F44336', alpha=0.5,
             label=r'$O(1/N)$')

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('Total Oracle Calls', fontsize=13)
    ax1.set_ylabel('Estimation Error |p̂ - p|', fontsize=13)
    ax1.set_title('Convergence Rate: MC vs QAE', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: Required calls for target precision
    epsilons = [0.1, 0.05, 0.02, 0.01, 0.005]
    p = p_exact
    mc_needed = [p * (1 - p) / eps**2 for eps in epsilons]
    qae_needed = [np.pi / (4 * eps) for eps in epsilons]

    x_pos = np.arange(len(epsilons))
    w = 0.35
    ax2.bar(x_pos - w/2, mc_needed, w, label='MC', color='#2196F3', alpha=0.8)
    ax2.bar(x_pos + w/2, qae_needed, w, label='QAE', color='#F44336', alpha=0.8)

    ax2.set_yscale('log')
    ax2.set_xlabel('Target Precision (ε)', fontsize=13)
    ax2.set_ylabel('Oracle Calls Required', fontsize=13)
    ax2.set_title('Scaling: MC O(1/ε²) vs QAE O(1/ε)', fontsize=14)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([str(e) for e in epsilons])
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    for i, (mc_c, qae_c) in enumerate(zip(mc_needed, qae_needed)):
        ax2.text(i, max(mc_c, qae_c) * 2, f'{mc_c/qae_c:.0f}x',
                 ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved: {save_path}")
    plt.close()


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    print("=" * 60)
    print("QAE CONVERGENCE DEMO")
    print("  barrier=6/8 -> p=0.25 (non-degenerate)")
    print("=" * 60)

    # Convergence comparison
    conv = convergence_comparison(
        barrier=6, window=2, n_steps=4,
        n_mc_repeats=30, qae_max_depth=32, n_shots_qae=100,
    )

    # Plot
    plot_convergence(conv,
        save_path='D:/projects/qae-parisian-climate/figures/qae_convergence.png')

    # 4-method comparison
    four = four_method_comparison(
        barrier=6, window=2, n_steps=4,
        n_mc=10000, n_is=10000,
        qae_max_depth=16, n_shots_qae=200,
    )
