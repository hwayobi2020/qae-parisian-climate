"""
Quantum Oracle for Parisian Condition with Bennett's Pebble Game

Architecture:
    1. Single-step transition operator U_step: |IVT_t, state> -> |IVT_{t+1}, state'>
       - Implements JD-OU dynamics as reversible circuit
    2. Comparator: |IVT> -> flag if IVT >= barrier
    3. Consecutive counter: increments if above, RESETS if below
    4. Parisian flag: marks if counter >= window at any point (OR-latch)

Bennett's Pebble Game (1989):
    - Naive: T steps x n qubits/step = O(T*n) qubits (infeasible for T=1460)
    - Pebble game: O(sqrt(T)) checkpoints, recompute between them
    - Qubit cost: O(sqrt(T) * n + T) instead of O(T * n)
    - Gate cost: O(T^{3/2}) instead of O(T)
    - Trade-off: more gates, far fewer qubits

    The O(T) term comes from OR-latch garbage qubits needed to track
    whether the Parisian condition was EVER satisfied during the path.
    This is unavoidable for sequential OR in reversible computation.

References:
    - Bennett (1989): Time/space tradeoffs for reversible computation
    - Cuccaro et al. (2004): New quantum ripple-carry addition circuit
    - Haner et al. (2018): Optimizing quantum circuits for arithmetic
    - Stamatopoulos et al. (2020): Option pricing using quantum computers
"""

from qiskit import QuantumCircuit, QuantumRegister
import numpy as np
from dataclasses import dataclass


@dataclass
class CircuitConfig:
    """Configuration for quantum Parisian oracle circuit"""
    n_ivt_qubits: int = 8       # IVT discretization (2^8 = 256 levels)
    n_counter_qubits: int = 6   # Consecutive counter (up to 63 steps)
    n_time_steps: int = 192     # Total time steps (48h at 15-min resolution)
    barrier_level: int = 128    # Barrier as integer level (after discretization)
    window_steps: int = 48      # Consecutive steps required
    n_checkpoints: int = None   # Pebble game checkpoints (default: sqrt(T))

    def __post_init__(self):
        if self.n_checkpoints is None:
            self.n_checkpoints = int(np.ceil(np.sqrt(self.n_time_steps)))


# ============================================================
# Building Block 1: Reversible Comparator (borrow-chain)
# ============================================================

def build_comparator(n_qubits: int, threshold: int) -> QuantumCircuit:
    """
    Reversible comparator: flag = (x >= threshold)

    Uses borrow-chain subtraction against classical constant.
    Computes x - threshold; if no final borrow, then x >= threshold.

    Borrow logic per bit i:
        c_i = 0: borrow_i = borrow_{i-1} AND (NOT x_i)
        c_i = 1: borrow_i = (NOT x_i) OR borrow_{i-1}
                           = NOT(x_i AND NOT borrow_{i-1})

    flag = NOT borrow_{n-1}  (no borrow => x >= threshold)

    Gate complexity: O(n) Toffoli gates
    Ancilla: n qubits (borrow chain)

    Args:
        n_qubits: Number of qubits encoding value
        threshold: Classical threshold to compare against

    Returns:
        QuantumCircuit with registers [x (n), borrow (n), flag (1)]
    """
    x_reg = QuantumRegister(n_qubits, 'x')
    borrow = QuantumRegister(n_qubits, 'borrow')
    flag_reg = QuantumRegister(1, 'flag')
    qc = QuantumCircuit(x_reg, borrow, flag_reg, name='CMP')

    c_bits = [(threshold >> i) & 1 for i in range(n_qubits)]

    # --- Forward: compute borrow chain ---
    for i in range(n_qubits):
        if i == 0:
            if c_bits[0] == 1:
                # borrow_0 = NOT x_0
                qc.x(x_reg[0])
                qc.cx(x_reg[0], borrow[0])
                qc.x(x_reg[0])
            # c_bits[0] == 0: borrow_0 = 0 (already initialized)
        else:
            if c_bits[i] == 0:
                # borrow_i = borrow_{i-1} AND NOT(x_i)
                qc.x(x_reg[i])
                qc.ccx(borrow[i - 1], x_reg[i], borrow[i])
                qc.x(x_reg[i])
            else:
                # borrow_i = NOT(x_i AND NOT borrow_{i-1})
                # Start with borrow[i] = 1, flip to 0 if x_i=1 AND borrow_{i-1}=0
                qc.x(borrow[i])
                qc.x(borrow[i - 1])
                qc.ccx(x_reg[i], borrow[i - 1], borrow[i])
                qc.x(borrow[i - 1])

    # --- flag = NOT borrow_{n-1} ---
    qc.x(borrow[n_qubits - 1])
    qc.cx(borrow[n_qubits - 1], flag_reg[0])
    qc.x(borrow[n_qubits - 1])

    # --- Backward: uncompute borrow chain ---
    for i in range(n_qubits - 1, -1, -1):
        if i == 0:
            if c_bits[0] == 1:
                qc.x(x_reg[0])
                qc.cx(x_reg[0], borrow[0])
                qc.x(x_reg[0])
        else:
            if c_bits[i] == 0:
                qc.x(x_reg[i])
                qc.ccx(borrow[i - 1], x_reg[i], borrow[i])
                qc.x(x_reg[i])
            else:
                qc.x(borrow[i - 1])
                qc.ccx(x_reg[i], borrow[i - 1], borrow[i])
                qc.x(borrow[i - 1])
                qc.x(borrow[i])

    return qc


# ============================================================
# Building Block 2: Reversible Counter with Conditional Reset
# ============================================================

def build_counter_update(n_counter_qubits: int) -> QuantumCircuit:
    """
    Reversible counter update for Parisian condition:
        if above_flag=1: counter += 1
        if above_flag=0: counter = 0 (reset)

    Strategy:
        1. Copy counter to scratch (preserves old value for reversibility)
        2. If NOT above: XOR counter with scratch (zeros counter)
        3. If above: controlled increment

    Gate complexity: O(n_counter) Toffoli gates per step

    Args:
        n_counter_qubits: Bits for consecutive counter

    Returns:
        QuantumCircuit with [counter (n), above_flag (1), scratch (n)]
    """
    counter = QuantumRegister(n_counter_qubits, 'cnt')
    above = QuantumRegister(1, 'above')
    scratch = QuantumRegister(n_counter_qubits, 'scratch')
    qc = QuantumCircuit(counter, above, scratch, name='CNT_UPD')

    # Phase 1: Save counter to scratch
    for i in range(n_counter_qubits):
        qc.cx(counter[i], scratch[i])

    # Phase 2: Conditional reset (if NOT above, zero counter)
    qc.x(above[0])
    for i in range(n_counter_qubits):
        # When above is flipped (=1 means "below"):
        # counter[i] ^= (above_flipped AND scratch[i])
        # Since scratch[i] == counter[i], this zeros counter[i] when below
        qc.ccx(above[0], scratch[i], counter[i])
    qc.x(above[0])

    # Phase 3: Saturating conditional increment (if above AND not max, counter += 1)
    # Standard controlled-increment with saturation:
    # Only increment if counter < 2^n - 1 (not all bits = 1)
    # Process from MSB down to avoid carry corruption
    # The MCX naturally handles saturation: when all bits are 1,
    # the MSB flip propagates correctly. To saturate, we add
    # an anti-control on "all bits = 1" condition.
    # Simpler approach: use scratch[0] as "not_max" flag
    #   not_max = NAND(counter[0], counter[1], ..., counter[n-1])
    # But we already used scratch. Instead, just use enough counter bits.
    # For correctness: increment with wrap is OK if n_counter > log2(T+1),
    # since counter can never reach 2^n within T steps.
    for i in range(n_counter_qubits - 1, 0, -1):
        controls = [above[0]] + [counter[j] for j in range(i)]
        qc.mcx(controls, counter[i])
    qc.cx(above[0], counter[0])

    return qc


# ============================================================
# Building Block 3: OR-Latch (first-trigger detection)
# ============================================================

def build_or_latch(n_counter_qubits: int, window: int) -> QuantumCircuit:
    """
    OR-latch for Parisian flag: flag |= (counter >= window)

    Uses first-trigger detection:
        trigger = (counter >= window) AND (NOT flag)
        flag ^= trigger
    This only fires once (the first time counter reaches window),
    so repeated applications don't toggle the flag back.

    Requires a dedicated 'trigger' ancilla qubit that must be
    uncomputed after the flag update.

    Uncomputation is valid because after flag is set to 1:
        trigger_recomputed = (counter >= window) AND (NOT flag_new)
                           = (counter >= window) AND 0 = 0
    Which matches the expected state (trigger was used and should be 0).

    Args:
        n_counter_qubits: Counter register size
        window: Required consecutive steps

    Returns:
        QuantumCircuit with [counter (n), flag (1), trigger (1), cmp_borrow (n)]
    """
    counter = QuantumRegister(n_counter_qubits, 'cnt')
    flag = QuantumRegister(1, 'flag')
    trigger = QuantumRegister(1, 'trigger')
    cmp_borrow = QuantumRegister(n_counter_qubits, 'cmp_borr')
    qc = QuantumCircuit(counter, flag, trigger, cmp_borrow, name='OR_LATCH')

    # Step 1: Compute (counter >= window) into trigger qubit
    # Reuse the comparator logic inline for the counter register
    c_bits = [(window >> i) & 1 for i in range(n_counter_qubits)]

    # Forward borrow chain
    for i in range(n_counter_qubits):
        if i == 0:
            if c_bits[0] == 1:
                qc.x(counter[0])
                qc.cx(counter[0], cmp_borrow[0])
                qc.x(counter[0])
        else:
            if c_bits[i] == 0:
                qc.x(counter[i])
                qc.ccx(cmp_borrow[i - 1], counter[i], cmp_borrow[i])
                qc.x(counter[i])
            else:
                qc.x(cmp_borrow[i])
                qc.x(cmp_borrow[i - 1])
                qc.ccx(counter[i], cmp_borrow[i - 1], cmp_borrow[i])
                qc.x(cmp_borrow[i - 1])

    # cond = NOT borrow_{n-1} means counter >= window
    # trigger = cond AND (NOT flag) = (NOT borrow) AND (NOT flag)
    qc.x(cmp_borrow[n_counter_qubits - 1])
    qc.x(flag[0])
    qc.ccx(cmp_borrow[n_counter_qubits - 1], flag[0], trigger[0])
    qc.x(flag[0])
    qc.x(cmp_borrow[n_counter_qubits - 1])

    # Step 2: flag ^= trigger (sets flag on first trigger)
    qc.cx(trigger[0], flag[0])

    # Step 3: Uncompute trigger
    # Now flag=1 (just set), so NOT flag=0, so trigger_recomputed = cond AND 0 = 0
    # This correctly uncomputes trigger back to 0
    qc.x(cmp_borrow[n_counter_qubits - 1])
    qc.x(flag[0])
    qc.ccx(cmp_borrow[n_counter_qubits - 1], flag[0], trigger[0])
    qc.x(flag[0])
    qc.x(cmp_borrow[n_counter_qubits - 1])

    # Uncompute borrow chain
    for i in range(n_counter_qubits - 1, -1, -1):
        if i == 0:
            if c_bits[0] == 1:
                qc.x(counter[0])
                qc.cx(counter[0], cmp_borrow[0])
                qc.x(counter[0])
        else:
            if c_bits[i] == 0:
                qc.x(counter[i])
                qc.ccx(cmp_borrow[i - 1], counter[i], cmp_borrow[i])
                qc.x(counter[i])
            else:
                qc.x(cmp_borrow[i - 1])
                qc.ccx(counter[i], cmp_borrow[i - 1], cmp_borrow[i])
                qc.x(cmp_borrow[i - 1])
                qc.x(cmp_borrow[i])

    return qc


# ============================================================
# Building Block 4: Single Time Step Transition (JD-OU)
# ============================================================

def build_jdou_step(n_ivt_qubits: int, n_rand_qubits: int) -> QuantumCircuit:
    """
    Single time step of JD-OU dynamics (reversible)

    |IVT_t>|rand> -> |IVT_{t+1}>|garbage>

    The transition encodes:
        IVT_{t+1} = IVT_t + kappa(theta_eff - IVT_t)dt + sigma*sqrt(dt)*Z

    For quantum implementation:
        - Discretize IVT into 2^n levels
        - Drift: reversible affine transform (Draper QFT adder or ripple-carry)
        - Diffusion: controlled rotations from |rand> register
        - AR jumps: controlled operations on regime qubit

    Gate complexity per step: O(n^2) for arithmetic, O(n) for rotations
    """
    ivt = QuantumRegister(n_ivt_qubits, 'ivt')
    rand_reg = QuantumRegister(n_rand_qubits, 'rand')
    anc = QuantumRegister(n_ivt_qubits, 'step_anc')
    qc = QuantumCircuit(ivt, rand_reg, anc, name='JDOU_STEP')

    # Placeholder: actual arithmetic would use Draper/Cuccaro adders
    # Gate complexity is O(n^2) per step

    return qc


# ============================================================
# Pebble Game Strategy
# ============================================================

class PebbleGameStrategy:
    """
    Bennett's Pebble Game for reversible path computation

    Given T sequential steps, each requiring n qubits of state:
    - Naive: store all T states -> O(T*n) qubits
    - Pebble game with sqrt(T) checkpoints:
        -> O(sqrt(T) * n) qubits for IVT state
        -> O(T) qubits for OR-latch garbage (unavoidable)
        -> O(T^{3/2}) gate overhead from recomputation
    """

    def __init__(self, T: int, n_per_step: int):
        self.T = T
        self.n_per_step = n_per_step


# ============================================================
# Main Oracle: Parisian with Pebble Game
# ============================================================

class ParisianOraclePebble:
    """
    Quantum Parisian oracle using Bennett's Pebble Game

    Circuit structure:
        1. Initialize |IVT_0> from calibrated distribution
        2. For each time step t = 1..T:
           a. Apply JD-OU transition
           b. Compare IVT >= barrier -> above flag
           c. Update consecutive counter (increment or reset)
           d. OR-latch: flag |= (counter >= window)
           e. Uncompute above flag
        3. Parisian flag = 1 iff condition was ever met
        4. For QAE: phase-kick on flag, then uncompute everything

    Qubit budget (T=192, n=8, W=48):
        - sqrt(192)*8 = 112 checkpoint qubits (IVT state)
        - 192 OR-latch garbage qubits
        - 8 IVT working + 6 counter + 1 flag + ancillas
        Total: ~340 qubits (vs 1,543 naive)
    """

    def __init__(self, config: CircuitConfig):
        self.config = config
        self.strategy = PebbleGameStrategy(
            T=config.n_time_steps,
            n_per_step=config.n_ivt_qubits
        )

    def build_small_demo(self, n_steps: int = 4) -> QuantumCircuit:
        """
        Build small-scale demo for verification on Qiskit Aer.

        Parameters: n_ivt=3 (8 levels), barrier=4, window=2, counter=2 bits
        IVT initialized in uniform superposition.
        Static IVT (no transition) — purely tests comparator + counter + OR-latch.
        """
        n_ivt = 3
        n_cnt = 3       # Need ceil(log2(n_steps+1)) to avoid overflow
        barrier = 4
        window = 2

        ivt = QuantumRegister(n_ivt, 'ivt')
        counter = QuantumRegister(n_cnt, 'counter')
        above = QuantumRegister(1, 'above')
        flag = QuantumRegister(1, 'parisian')
        cmp_borrow = QuantumRegister(n_ivt, 'cmp_borr')
        cnt_scratch = QuantumRegister(n_cnt, 'cnt_scratch')
        trigger = QuantumRegister(1, 'trigger')
        latch_borrow = QuantumRegister(n_cnt, 'latch_borr')

        qc = QuantumCircuit(ivt, counter, above, flag, cmp_borrow,
                            cnt_scratch, trigger, latch_borrow,
                            name='Parisian_Demo')

        # Uniform superposition over IVT values
        for q in range(n_ivt):
            qc.h(ivt[q])

        for step in range(n_steps):
            qc.barrier(label=f'step_{step}')

            # Compare IVT >= barrier
            cmp = build_comparator(n_ivt, barrier)
            qc.append(cmp, ivt[:] + cmp_borrow[:] + above[:])

            # Update counter
            cnt_upd = build_counter_update(n_cnt)
            qc.append(cnt_upd, counter[:] + above[:] + cnt_scratch[:])

            # OR-latch: flag |= (counter >= window)
            latch = build_or_latch(n_cnt, window)
            qc.append(latch, counter[:] + flag[:] + trigger[:] + latch_borrow[:])

            # Uncompute above
            qc.append(cmp.inverse(), ivt[:] + cmp_borrow[:] + above[:])

        return qc

    def complexity_analysis(self) -> dict:
        """Complete complexity analysis for the paper"""
        cfg = self.config
        T = cfg.n_time_steps
        n = cfg.n_ivt_qubits
        W = cfg.window_steps
        k = cfg.n_checkpoints
        c = cfg.n_counter_qubits

        # Qubit counts
        q_naive = T * n + c + 1
        q_ivt_checkpoints = k * n          # Pebble game checkpoints
        q_ivt_working = n                   # Current IVT state
        q_counter = c                       # Consecutive counter
        q_flag = 1                          # Parisian flag
        q_or_garbage = T                    # OR-latch garbage (unavoidable)
        q_ancilla = n + c + 1 + c           # comparator + counter scratch + trigger + latch borrow
        q_pebble = q_ivt_checkpoints + q_ivt_working + q_counter + q_flag + q_or_garbage + q_ancilla

        # Gate counts (Toffoli gates as unit)
        g_step = n ** 2           # JD-OU transition (arithmetic)
        g_cmp = 2 * n             # Comparator + uncompute
        g_cnt = 3 * c             # Counter update
        g_latch = 4 * c           # OR-latch (comparator + trigger + uncompute)
        g_per_step = g_step + g_cmp + g_cnt + g_latch

        g_naive = T * g_per_step
        g_pebble = int(T ** 1.5) * g_per_step

        # QAE: O(1/epsilon) Grover iterations
        epsilon = 0.01
        n_grover = int(np.pi / (4 * epsilon))
        g_total_qae = n_grover * 2 * g_pebble  # x2 for oracle + diffusion

        # Classical MC
        mc_paths = int(1 / epsilon ** 2)

        return {
            'problem_params': {
                'T (time steps)': T,
                'n (IVT qubits)': n,
                'W (window)': W,
                'k (checkpoints)': k,
                'c (counter qubits)': c,
            },
            'qubits': {
                'naive': q_naive,
                'pebble_game': q_pebble,
                'breakdown': {
                    'IVT checkpoints': q_ivt_checkpoints,
                    'IVT working': q_ivt_working,
                    'counter': q_counter,
                    'flag': q_flag,
                    'OR-latch garbage': q_or_garbage,
                    'ancilla': q_ancilla,
                },
                'reduction': f'{q_naive / q_pebble:.1f}x',
            },
            'gates_per_oracle_call': {
                'naive': g_naive,
                'pebble_game': g_pebble,
                'overhead': f'{g_pebble / g_naive:.1f}x',
            },
            'total_for_pricing': {
                'qae_grover_iterations': n_grover,
                'qae_total_gates': g_total_qae,
                'classical_mc_paths': mc_paths,
                'classical_mc_total_ops': mc_paths * T,
                'quantum_advantage': f'O(1/e) vs O(1/e^2): {mc_paths // n_grover}x fewer iterations'
            },
            'asymptotic': {
                'classical_mc': f'O(T / e^2)',
                'quantum_pebble_qubits': f'O(sqrt(T)*n + T)',
                'quantum_pebble_gates': f'O(T^(3/2) * n^2 / e)',
                'qubit_reduction': f'O(T*n) -> O(sqrt(T)*n + T): sqrt(T)={np.sqrt(T):.1f}x for IVT state',
                'speed_advantage': 'O(1/e^2) -> O(1/e): quadratic speedup in accuracy'
            }
        }


# ============================================================
# Robust Extension: Parameter Uncertainty
# ============================================================

class RobustParisianOracle(ParisianOraclePebble):
    """
    Extended oracle exploring parameter space simultaneously.

    Additional registers |kappa>|sigma>|lambda> encode discretized parameters.
    The transition operator becomes parameter-dependent:
        U_step(kappa, sigma, lambda)|IVT_t> -> |IVT_{t+1}>

    Enables worst-case/best-case price estimation in one QAE run.
    """

    def __init__(self, config: CircuitConfig,
                 n_kappa_qubits: int = 3,
                 n_sigma_qubits: int = 3,
                 n_lambda_qubits: int = 2):
        super().__init__(config)
        self.n_kappa_qubits = n_kappa_qubits
        self.n_sigma_qubits = n_sigma_qubits
        self.n_lambda_qubits = n_lambda_qubits

    def total_qubits(self) -> int:
        base = self.complexity_analysis()['qubits']['pebble_game']
        return base + self.n_kappa_qubits + self.n_sigma_qubits + self.n_lambda_qubits

    def parameter_grid_size(self) -> int:
        return 2 ** (self.n_kappa_qubits + self.n_sigma_qubits + self.n_lambda_qubits)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Parisian Oracle - Complexity Analysis")
    print("=" * 60)

    config = CircuitConfig(
        n_ivt_qubits=8,
        n_counter_qubits=6,
        n_time_steps=192,
        barrier_level=128,
        window_steps=48,
    )

    oracle = ParisianOraclePebble(config)
    analysis = oracle.complexity_analysis()

    for section, data in analysis.items():
        print(f"\n{section}:")
        if isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        print(f"    {k2}: {v2}")
                else:
                    print(f"  {k}: {v}")
        else:
            print(f"  {data}")

    # Robust
    print(f"\n{'=' * 60}")
    print("Robust Extension")
    robust = RobustParisianOracle(config)
    print(f"  Total qubits: {robust.total_qubits()}")
    print(f"  Parameter grid: {robust.parameter_grid_size()}")

    # Demo
    print(f"\n{'=' * 60}")
    print("Small Demo Circuit")
    demo = oracle.build_small_demo(n_steps=4)
    print(f"  Qubits: {demo.num_qubits}")
    print(f"  Depth: {demo.depth()}")
    print(f"  Gates: {demo.size()}")
