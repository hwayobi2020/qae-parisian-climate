"""
IS-QAE: Importance-Sampled Quantum Amplitude Estimation

Two-stage hybrid quantum-classical algorithm:
    Stage 1 (Quantum): IQAE with IS-tilted state prep estimates p_IS = P_Q(triggered)
    Stage 2 (Classical): Correction factor c = E_Q[w(X)|triggered] via MC
    Final:  p = p_IS * c

Mathematical justification:
    p = P_P(triggered)
      = E_Q[1{triggered} * w(X)]           (IS identity)
      = P_Q(triggered) * E_Q[w(X)|triggered]
      = p_IS * c

The IS tilting increases p_IS >> p, reducing Grover iterations from
O(1/(eps*sqrt(p))) to O(1/(eps*sqrt(p_IS))). The correction c is
an O(1) quantity estimated cheaply via classical MC.

References:
    - Miyamoto & Kubo (2022): Reduction of qubits by importance sampling
    - Herbert (2022): Quantum Monte Carlo Integration
    - Grinko et al. (2021): Iterative QAE without QPE
"""

import numpy as np
import time
from typing import Dict, Tuple, List
from dataclasses import dataclass

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import GroverOperator
from qiskit_aer import AerSimulator

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from quantum.parisian_oracle import build_comparator, build_counter_update, build_or_latch
from quantum.is_state_prep import (
    TiltedDistribution,
    compute_tilted_distribution_simple,
    build_state_prep_circuit,
)


# ============================================================
# Result dataclass
# ============================================================

@dataclass
class ISQAEResult:
    """Results from IS-QAE estimation."""
    p_is: float                   # QAE estimate of P_Q(triggered)
    p_is_exact: float             # Exact P_Q(triggered) for the static model
    correction_factor: float      # c = E_Q[w|triggered]
    correction_se: float          # Standard error of correction
    p_final: float                # p = p_is * c
    p_exact: float                # Exact P_P(triggered) (for comparison)
    p_is_error: float             # |p_is - p_is_exact|
    total_oracle_calls: int       # Quantum oracle calls
    classical_samples: int        # MC samples for correction
    iqae_trace: list              # IQAE convergence trace
    tilted_dist: TiltedDistribution


# ============================================================
# IS-QAE Circuit
# ============================================================

def build_is_qae_circuit(
    n_ivt: int,
    n_counter: int,
    barrier: int,
    window: int,
    n_steps: int,
    tilted_dist: TiltedDistribution,
) -> Tuple[QuantumCircuit, int]:
    """
    Build QAE A-operator with IS-tilted state preparation.

    Identical to build_qae_circuit() except H gates are replaced
    with Grover-Rudolph controlled-Ry rotations encoding the
    tilted distribution.

    A_IS|0> = sum_k sqrt(q_k)|k> (x) |oracle(k)>

    The oracle (comparator + counter + OR-latch) is unchanged.

    Returns:
        (A_IS_circuit, flag_qubit_index)
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
                       name='A_IS_operator')

    # IS-tilted state preparation (replaces H gates)
    state_prep = build_state_prep_circuit(tilted_dist)
    A.append(state_prep, ivt[:])

    # Oracle: standard Parisian condition (no change)
    for step in range(n_steps):
        cmp = build_comparator(n_ivt, barrier)
        A.append(cmp, ivt[:] + cmp_borrow[:] + above[:])

        cnt_upd = build_counter_update(n_counter)
        A.append(cnt_upd, counter[:] + above[:] + cnt_scratch[:])

        latch = build_or_latch(n_counter, window)
        A.append(latch, counter[:] + flag[:] + trigger[:] + latch_borrow[:])

        A.append(cmp.inverse(), ivt[:] + cmp_borrow[:] + above[:])

    flag_idx = n_ivt + n_counter + 1
    return A, flag_idx


# ============================================================
# IS-QAE Measurement and IQAE
# ============================================================

def is_qae_measure(
    n_ivt: int, n_counter: int,
    barrier: int, window: int, n_steps: int,
    tilted_dist: TiltedDistribution,
    n_grover_iters: int = 0, n_shots: int = 100,
) -> Tuple[float, int]:
    """
    Single IS-QAE round: apply A_IS, then Q^m, measure flag.

    Returns:
        (fraction_of_1s, oracle_calls_per_shot)
    """
    A, flag_idx = build_is_qae_circuit(
        n_ivt, n_counter, barrier, window, n_steps, tilted_dist
    )
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
    qc_t = qc.decompose(reps=10)
    counts = backend.run(qc_t, shots=n_shots).result().get_counts()

    p_flag = counts.get('1', 0) / n_shots
    oracle_calls = 1 + 2 * n_grover_iters

    return p_flag, oracle_calls


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

    res = minimize_scalar(neg_ll, bounds=(0.001, np.pi / 2 - 0.001),
                          method='bounded')
    return res.x


def is_iqae_estimate(
    n_ivt: int, n_counter: int,
    barrier: int, window: int, n_steps: int,
    tilted_dist: TiltedDistribution,
    max_depth: int = 32,
    n_shots_per_round: int = 100,
) -> Dict:
    """
    IQAE with IS-tilted state preparation.

    Estimates p_IS = P_Q(triggered) using increasing Grover depths.
    Since p_IS >> p, the amplitude theta_IS is larger and the ML
    estimator converges faster (fewer rounds needed).
    """
    p_is_exact = tilted_dist.prob_above_barrier() if n_steps >= window else 0.0

    depths = [0]
    m = 1
    while m <= max_depth:
        depths.append(m)
        m *= 2

    all_data = []
    trace = []
    total_oracle_calls = 0

    for m in depths:
        p_flag, oc_per_shot = is_qae_measure(
            n_ivt, n_counter, barrier, window, n_steps,
            tilted_dist, n_grover_iters=m, n_shots=n_shots_per_round,
        )
        n_1s = int(round(p_flag * n_shots_per_round))
        total_oracle_calls += oc_per_shot * n_shots_per_round

        all_data.append((m, n_1s, n_shots_per_round))

        theta_est = _ml_theta(all_data)
        p_est = np.sin(theta_est) ** 2
        error = abs(p_est - p_is_exact)

        trace.append({
            'depth': m,
            'p_flag': float(p_flag),
            'p_est': float(p_est),
            'error': float(error),
            'total_oracle_calls': total_oracle_calls,
        })

    return {
        'p_is_exact': float(p_is_exact),
        'trace': trace,
    }


# ============================================================
# Classical Correction Factor
# ============================================================

def compute_correction_factor(
    tilted_dist: TiltedDistribution,
    n_steps: int,
    window: int,
    n_samples: int = 10000,
    seed: int = 42,
) -> Tuple[float, float]:
    """
    Classical correction factor c = E_Q[w(X) | triggered].

    For static IVT: sample from tilted distribution, compute
    likelihood ratio w(k) = p_uniform(k) / q_tilted(k) for triggered paths.

    Under Q, triggering is common (p_IS ~ 0.7), so the conditional
    expectation converges fast.

    Returns:
        (correction_factor, standard_error)
    """
    rng = np.random.RandomState(seed)

    ivt_samples = rng.choice(
        tilted_dist.n_levels, size=n_samples, p=tilted_dist.probs
    )

    triggered = (ivt_samples >= tilted_dist.barrier) & (n_steps >= window)
    n_triggered = int(triggered.sum())

    if n_triggered == 0:
        return 0.0, float('inf')

    lr = tilted_dist.likelihood_ratios()
    lr_triggered = lr[ivt_samples[triggered]]

    correction = float(np.mean(lr_triggered))
    correction_se = float(np.std(lr_triggered) / np.sqrt(n_triggered))

    return correction, correction_se


def compute_correction_exact(
    tilted_dist: TiltedDistribution,
    n_steps: int,
    window: int,
) -> float:
    """
    Exact correction factor for the static-IVT model.

    c = E_Q[w(X) | triggered] = sum_{k>=barrier} q_k * w_k / p_IS
    """
    if n_steps < window:
        return 0.0

    lr = tilted_dist.likelihood_ratios()
    p_is = tilted_dist.prob_above_barrier()
    if p_is < 1e-15:
        return 0.0

    c = float(np.sum(tilted_dist.probs[tilted_dist.barrier:] *
                      lr[tilted_dist.barrier:])) / p_is
    return c


# ============================================================
# Full IS-QAE Pipeline
# ============================================================

def is_qae_full_pipeline(
    n_ivt: int = 3,
    n_counter: int = 3,
    barrier: int = 6,
    window: int = 2,
    n_steps: int = 4,
    q_high: float = 0.7,
    qae_max_depth: int = 16,
    n_shots_qae: int = 200,
    n_correction_samples: int = 10000,
    seed: int = 42,
) -> ISQAEResult:
    """
    Complete IS-QAE pipeline:
        1. Compute tilted distribution (q_high on above-barrier states)
        2. Run IQAE with IS state prep -> p_IS
        3. Compute classical correction factor -> c
        4. Return p = p_IS * c

    Args:
        n_ivt, n_counter, barrier, window, n_steps: Circuit parameters
        q_high: IS tilt strength (fraction of prob on above-barrier)
        qae_max_depth: Maximum Grover depth for IQAE
        n_shots_qae: Shots per IQAE round
        n_correction_samples: Classical MC samples for correction
        seed: Random seed

    Returns:
        ISQAEResult with all estimates and diagnostics
    """
    n_levels = 2 ** n_ivt
    p_exact = (n_levels - barrier) / n_levels if n_steps >= window else 0.0

    # Step 1: Tilted distribution
    tilted_dist = compute_tilted_distribution_simple(n_ivt, barrier, q_high)
    p_is_exact = tilted_dist.prob_above_barrier()

    # Step 2: IQAE with IS state prep
    iqae_res = is_iqae_estimate(
        n_ivt, n_counter, barrier, window, n_steps,
        tilted_dist,
        max_depth=qae_max_depth,
        n_shots_per_round=n_shots_qae,
    )

    final = iqae_res['trace'][-1]
    p_is = final['p_est']

    # Step 3: Classical correction
    correction, correction_se = compute_correction_factor(
        tilted_dist, n_steps, window,
        n_samples=n_correction_samples, seed=seed,
    )

    # Step 4: Final estimate
    p_final = p_is * correction

    return ISQAEResult(
        p_is=p_is,
        p_is_exact=p_is_exact,
        correction_factor=correction,
        correction_se=correction_se,
        p_final=p_final,
        p_exact=p_exact,
        p_is_error=final['error'],
        total_oracle_calls=final['total_oracle_calls'],
        classical_samples=n_correction_samples,
        iqae_trace=iqae_res['trace'],
        tilted_dist=tilted_dist,
    )


# ============================================================
# Complexity Analysis
# ============================================================

def complexity_comparison(
    p: float,
    p_is: float,
    epsilon_rel: float = 0.01,
    T: int = 192,
    n: int = 8,
) -> Dict:
    """
    Concrete complexity comparison for the 4 methods.

    Args:
        p: True trigger probability P_P(triggered)
        p_is: Tilted trigger probability P_Q(triggered)
        epsilon_rel: Target relative error
        T: Number of time steps
        n: IVT qubits
    """
    # Naive MC: N = (1-p) / (p * eps_rel^2)
    n_mc = int(np.ceil((1 - p) / (p * epsilon_rel ** 2)))

    # IS-MC: variance of w*1{trig} under Q
    # For static IVT: w = (1/N) / (q_high/n_above) = n_above / (N * q_high)
    # Var_Q = p_is * w^2 - p^2
    n_levels = 2 ** n
    n_above = n_levels - int(p * n_levels + 0.5)  # approximate
    w = 1.0 / (n_levels * p_is / n_above) if n_above > 0 else 1.0
    var_is = p_is * w ** 2 - p ** 2
    var_naive = p * (1 - p)
    vr = var_naive / max(var_is, 1e-15)
    n_is_mc = int(np.ceil(n_mc / vr))

    # QAE: M = pi / (4 * epsilon_rel * sqrt(p))  Grover iterations
    m_qae = int(np.ceil(np.pi / (4 * epsilon_rel * np.sqrt(p))))

    # IS-QAE: M_IS = pi / (4 * epsilon_rel * sqrt(p_IS))
    m_is_qae = int(np.ceil(np.pi / (4 * epsilon_rel * np.sqrt(p_is))))
    # + classical correction (cheap)
    n_correction = max(100, int(np.ceil(1 / (epsilon_rel ** 2))))

    # Gate cost per oracle call (pebble game)
    g_per_step = n ** 2 + 2 * n + 3 * 6 + 4 * 6  # rough estimate
    g_oracle = int(T ** 1.5) * g_per_step

    return {
        'naive_mc': {
            'samples': n_mc,
            'total_ops': n_mc * T,
        },
        'is_mc': {
            'samples': n_is_mc,
            'total_ops': n_is_mc * T,
            'variance_reduction': vr,
        },
        'qae': {
            'grover_iters': m_qae,
            'oracle_calls': m_qae,
            'total_gates': m_qae * 2 * g_oracle,
        },
        'is_qae': {
            'grover_iters': m_is_qae,
            'oracle_calls': m_is_qae,
            'correction_samples': n_correction,
            'total_gates': m_is_qae * 2 * g_oracle,
            'quantum_speedup_vs_qae': f'{m_qae / max(m_is_qae, 1):.1f}x',
        },
        'summary': {
            'p': p,
            'p_IS': p_is,
            'epsilon_rel': epsilon_rel,
            'QAE_iters': m_qae,
            'IS-QAE_iters': m_is_qae,
            'IS-QAE saves': f'{(1 - m_is_qae / m_qae) * 100:.0f}% fewer Grover iters',
        },
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore', category=DeprecationWarning)

    print("=" * 60)
    print("IS-QAE FULL PIPELINE TEST")
    print("=" * 60)

    # Run full pipeline
    print("\nRunning IS-QAE (n_ivt=3, barrier=6, q_high=0.7)...")
    result = is_qae_full_pipeline(
        n_ivt=3, n_counter=3, barrier=6, window=2, n_steps=4,
        q_high=0.7, qae_max_depth=8, n_shots_qae=200,
        n_correction_samples=10000, seed=42,
    )

    print(f"\n  Results:")
    print(f"    p_exact          = {result.p_exact:.4f}")
    print(f"    p_IS (QAE est)   = {result.p_is:.4f} (exact: {result.p_is_exact:.4f})")
    print(f"    correction c     = {result.correction_factor:.4f} +/- {result.correction_se:.4f}")
    print(f"    p_final = p_IS*c = {result.p_final:.4f}")
    print(f"    |p_final - p_exact| = {abs(result.p_final - result.p_exact):.4f}")
    print(f"    Oracle calls     = {result.total_oracle_calls}")

    # Exact correction for validation
    c_exact = compute_correction_exact(result.tilted_dist, n_steps=4, window=2)
    print(f"\n  Exact correction:  {c_exact:.6f}")
    print(f"  p_IS_exact * c_exact = {result.p_is_exact * c_exact:.6f} (should be {result.p_exact:.6f})")

    # IQAE trace
    print(f"\n  IQAE trace (IS-QAE):")
    print(f"    {'Depth':>6} {'p_flag':>8} {'p_est':>8} {'Error':>8} {'Calls':>8}")
    for t in result.iqae_trace:
        print(f"    {t['depth']:>6} {t['p_flag']:>8.4f} {t['p_est']:>8.4f} "
              f"{t['error']:>8.4f} {t['total_oracle_calls']:>8}")

    # Complexity comparison
    print(f"\n{'='*60}")
    print("COMPLEXITY COMPARISON")
    print(f"{'='*60}")
    comp = complexity_comparison(p=0.25, p_is=0.7, epsilon_rel=0.01)
    for key, val in comp.items():
        print(f"\n  {key}:")
        for k, v in val.items():
            print(f"    {k}: {v}")

    # Rare event scenario
    print(f"\n{'='*60}")
    print("RARE EVENT SCENARIO (p=0.01, p_IS=0.3)")
    print(f"{'='*60}")
    comp_rare = complexity_comparison(p=0.01, p_is=0.3, epsilon_rel=0.05)
    for key, val in comp_rare.items():
        print(f"\n  {key}:")
        for k, v in val.items():
            print(f"    {k}: {v}")
