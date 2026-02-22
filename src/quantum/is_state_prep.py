"""
IS-Informed State Preparation for QAE (Grover-Rudolph Algorithm)

Loads an arbitrary probability distribution |0> -> sum_k sqrt(q_k)|k>
using a binary tree of controlled-Ry rotations (Grover & Rudolph, 2002).

For IS-QAE, the tilted distribution q(x) puts more amplitude on
above-barrier IVT states, increasing trigger probability p_IS >> p.
This reduces the Grover iterations from O(1/(eps*sqrt(p))) to O(1/(eps*sqrt(p_IS))).

References:
    - Grover & Rudolph (2002): Creating superpositions that correspond
      to efficiently integrable probability distributions
    - Herbert (2022): Quantum Monte Carlo Integration
    - Miyamoto & Kubo (2022): Reduction of qubits by importance sampling
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import RYGate


# ============================================================
# Tilted Distribution
# ============================================================

@dataclass
class TiltedDistribution:
    """Discretized probability distribution for quantum state preparation."""
    n_qubits: int
    n_levels: int
    probs: np.ndarray       # Shape (n_levels,)
    q_high: float            # Total prob mass on above-barrier states
    barrier: int             # Barrier level
    source: str = 'simple'   # 'simple' or 'jdou_mc'

    def prob_above_barrier(self) -> float:
        """P_Q(IVT >= barrier)."""
        return float(np.sum(self.probs[self.barrier:]))

    def uniform_probs(self) -> np.ndarray:
        """Reference uniform distribution."""
        return np.full(self.n_levels, 1.0 / self.n_levels)

    def likelihood_ratios(self) -> np.ndarray:
        """w(k) = p_uniform(k) / q_tilted(k) for each level k."""
        uniform = self.uniform_probs()
        lr = np.zeros(self.n_levels)
        mask = self.probs > 1e-15
        lr[mask] = uniform[mask] / self.probs[mask]
        return lr


def compute_tilted_distribution_simple(
    n_ivt: int,
    barrier: int,
    q_high: float = 0.7,
) -> TiltedDistribution:
    """
    Simple tilted distribution for the static-IVT demo.

    Places probability q_high uniformly on above-barrier states
    and (1-q_high) uniformly on below-barrier states.

    Args:
        n_ivt: Number of IVT qubits (2^n_ivt levels)
        barrier: Integer barrier level
        q_high: Total probability mass on above-barrier states

    Returns:
        TiltedDistribution
    """
    n_levels = 2 ** n_ivt
    n_above = n_levels - barrier
    n_below = barrier

    if n_above <= 0 or n_below <= 0:
        raise ValueError(f"barrier={barrier} must be in (0, {n_levels})")
    if not 0 < q_high < 1:
        raise ValueError(f"q_high={q_high} must be in (0, 1)")

    probs = np.zeros(n_levels)
    probs[:barrier] = (1 - q_high) / n_below
    probs[barrier:] = q_high / n_above

    return TiltedDistribution(
        n_qubits=n_ivt,
        n_levels=n_levels,
        probs=probs,
        q_high=q_high,
        barrier=barrier,
        source='simple',
    )


def compute_tilted_distribution_jdou(
    n_ivt: int,
    barrier_frac: float,
    ivt_range: Tuple[float, float],
    jdou_params: dict,
    tilt_params: dict,
    n_steps: int,
    dt: float,
    n_mc_paths: int = 50000,
    seed: int = 42,
) -> TiltedDistribution:
    """
    Tilted distribution from JD-OU Monte Carlo simulation.

    Runs IS-MC paths under tilted measure Q, histograms the IVT values
    into 2^n_ivt bins to estimate the marginal distribution.

    Args:
        n_ivt: Number of IVT qubits
        barrier_frac: Barrier as fraction of IVT range (0-1)
        ivt_range: (min_ivt, max_ivt) for discretization
        jdou_params: JD-OU model parameters dict
        tilt_params: IS tilt parameters dict
        n_steps: Time steps
        dt: Time step size
        n_mc_paths: MC paths for histogram
        seed: Random seed

    Returns:
        TiltedDistribution
    """
    n_levels = 2 ** n_ivt
    barrier = int(barrier_frac * n_levels)
    bin_edges = np.linspace(ivt_range[0], ivt_range[1], n_levels + 1)

    # Import here to avoid circular dependency
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from models.jump_diffusion_ou import JDOUModel

    model = JDOUModel(**jdou_params)
    rng = np.random.RandomState(seed)

    # Simulate under tilted measure
    all_ivt = []
    ivt = np.full(n_mc_paths, jdou_params.get('theta', 85.8))

    for step in range(n_steps):
        # Simplified: normal O-U step with tilt
        mu_tilt = tilt_params.get('mu_diffusion', 0.0)
        z = rng.randn(n_mc_paths) + mu_tilt
        kappa = jdou_params.get('kappa', 0.767)
        theta = jdou_params.get('theta', 85.8)
        sigma = jdou_params.get('sigma', 62.0)
        ivt = ivt + kappa * (theta - ivt) * dt + sigma * np.sqrt(dt) * z
        ivt = np.maximum(ivt, 0)
        all_ivt.append(ivt.copy())

    all_ivt = np.concatenate(all_ivt)
    hist, _ = np.histogram(all_ivt, bins=bin_edges, density=True)
    probs = hist * np.diff(bin_edges)
    probs = probs / probs.sum()  # normalize

    q_high = float(np.sum(probs[barrier:]))

    return TiltedDistribution(
        n_qubits=n_ivt,
        n_levels=n_levels,
        probs=probs,
        q_high=q_high,
        barrier=barrier,
        source='jdou_mc',
    )


# ============================================================
# Grover-Rudolph State Preparation
# ============================================================

def _compute_rotation_angles(
    probs: np.ndarray,
    n_qubits: int,
) -> List[Tuple[int, List[int], Optional[int], float]]:
    """
    Compute Ry rotation angles for Grover-Rudolph state preparation.

    Binary tree decomposition: at each level, split the distribution
    based on the next most significant bit.

    Convention:
        - ivt[0] = LSB, ivt[n-1] = MSB
        - Level 0 targets MSB (ivt[n-1]), level n-1 targets LSB (ivt[0])
        - Control qubits at level d: [ivt[n-1], ..., ivt[n-d]]
        - Integer ctrl_state: bit i = required value of ctrl_qubits[i]

    Args:
        probs: Probability distribution, shape (2^n_qubits,)
        n_qubits: Number of qubits

    Returns:
        List of (target_qubit, ctrl_qubits, ctrl_state_int, angle)
    """
    angles = []

    for level in range(n_qubits):
        target = n_qubits - 1 - level
        n_groups = 2 ** level
        group_size = 2 ** (n_qubits - level)
        half = group_size // 2

        ctrl_qubits = [n_qubits - 1 - i for i in range(level)]

        for g in range(n_groups):
            start = g * group_size

            p_left = float(np.sum(probs[start:start + half]))
            p_total = float(np.sum(probs[start:start + group_size]))

            if p_total < 1e-15:
                theta = 0.0
            else:
                ratio = np.clip(p_left / p_total, 0.0, 1.0)
                theta = 2.0 * np.arccos(np.sqrt(ratio))

            # Compute integer ctrl_state from group index
            # bit i of ctrl_state = value of ctrl_qubits[i]
            # ctrl_qubits[i] = q_{n-1-i}, value = bit (level-1-i) of g
            if level == 0:
                ctrl_state = None
            else:
                ctrl_state = 0
                for i in range(level):
                    bit_val = (g >> (level - 1 - i)) & 1
                    ctrl_state |= bit_val << i

            angles.append((target, list(ctrl_qubits), ctrl_state, theta))

    return angles


def build_state_prep_circuit(
    dist: TiltedDistribution,
) -> QuantumCircuit:
    """
    Build Grover-Rudolph state preparation circuit.

    Maps |0...0> to sum_k sqrt(q_k) |k> using a binary tree
    of controlled-Ry rotations.

    Gate count: 2^n - 1 rotations (exact for arbitrary distribution).
    For n=3: 7 rotations, for n=8: 255 rotations.

    Args:
        dist: TiltedDistribution with probabilities

    Returns:
        QuantumCircuit on n_qubits qubits
    """
    n = dist.n_qubits
    qr = QuantumRegister(n, 'ivt')
    qc = QuantumCircuit(qr, name='IS_StatePrep')

    angles = _compute_rotation_angles(dist.probs, n)

    for target, ctrl_qubits, ctrl_state, theta in angles:
        if abs(theta) < 1e-12:
            continue
        if abs(theta - 2 * np.pi) < 1e-12:
            continue  # Full rotation = identity

        if ctrl_state is None:
            # No controls (root of tree)
            qc.ry(theta, qr[target])
        else:
            n_ctrls = len(ctrl_qubits)
            ry_gate = RYGate(theta).control(n_ctrls, ctrl_state=ctrl_state)
            qc.append(ry_gate, [qr[q] for q in ctrl_qubits] + [qr[target]])

    return qc


# ============================================================
# Verification utilities
# ============================================================

def verify_state_prep(
    dist: TiltedDistribution,
    verbose: bool = True,
) -> np.ndarray:
    """
    Verify state prep circuit using statevector simulation.

    Returns measured probabilities from statevector.
    """
    from qiskit_aer import AerSimulator

    qc = build_state_prep_circuit(dist)
    qc.save_statevector()

    backend = AerSimulator(method='statevector')
    qc_dec = qc.decompose(reps=10)
    result = backend.run(qc_dec).result()
    sv = result.get_statevector()
    probs_measured = np.abs(np.array(sv)) ** 2

    if verbose:
        max_err = np.max(np.abs(probs_measured - dist.probs))
        print(f"State prep verification (n={dist.n_qubits}):")
        print(f"  Target distribution: {dist.probs}")
        print(f"  Measured (SV sim):   {np.round(probs_measured, 6)}")
        print(f"  Max error: {max_err:.2e}")
        print(f"  P(above barrier):  target={dist.prob_above_barrier():.4f}, "
              f"measured={float(np.sum(probs_measured[dist.barrier:])):.4f}")

    return probs_measured


if __name__ == "__main__":
    print("=" * 60)
    print("IS State Preparation - Grover-Rudolph Verification")
    print("=" * 60)

    # Test 1: Simple tilted distribution
    print("\n--- Test 1: Tilted distribution (n=3, barrier=6, q_high=0.7) ---")
    dist = compute_tilted_distribution_simple(n_ivt=3, barrier=6, q_high=0.7)
    probs_sv = verify_state_prep(dist)
    assert np.allclose(probs_sv, dist.probs, atol=1e-10), "State prep failed!"
    print("  PASSED")

    # Test 2: Uniform distribution should match H gates
    print("\n--- Test 2: Uniform distribution (should match H gates) ---")
    n_levels = 8
    uniform_dist = TiltedDistribution(
        n_qubits=3, n_levels=n_levels,
        probs=np.full(n_levels, 1.0 / n_levels),
        q_high=0.25, barrier=6, source='uniform',
    )
    probs_sv = verify_state_prep(uniform_dist)
    assert np.allclose(probs_sv, 1.0 / n_levels, atol=1e-10), "Uniform test failed!"
    print("  PASSED")

    # Test 3: Extreme tilt
    print("\n--- Test 3: Extreme tilt (q_high=0.95) ---")
    dist3 = compute_tilted_distribution_simple(n_ivt=3, barrier=6, q_high=0.95)
    probs_sv = verify_state_prep(dist3)
    assert np.allclose(probs_sv, dist3.probs, atol=1e-10), "Extreme tilt failed!"
    print("  PASSED")

    # Test 4: Arbitrary distribution
    print("\n--- Test 4: Arbitrary 4-level distribution ---")
    arb_probs = np.array([0.1, 0.2, 0.3, 0.4])
    arb_dist = TiltedDistribution(
        n_qubits=2, n_levels=4, probs=arb_probs,
        q_high=0.7, barrier=2, source='arbitrary',
    )
    probs_sv = verify_state_prep(arb_dist)
    assert np.allclose(probs_sv, arb_probs, atol=1e-10), "Arbitrary dist failed!"
    print("  PASSED")

    # Test 5: Larger circuit (n=4, 16 levels)
    print("\n--- Test 5: n=4, 16 levels ---")
    dist5 = compute_tilted_distribution_simple(n_ivt=4, barrier=12, q_high=0.8)
    probs_sv = verify_state_prep(dist5)
    assert np.allclose(probs_sv, dist5.probs, atol=1e-10), "n=4 test failed!"
    print("  PASSED")

    print("\n" + "=" * 60)
    print("All state prep tests PASSED")
    print("=" * 60)
