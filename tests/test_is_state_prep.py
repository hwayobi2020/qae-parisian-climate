"""
Tests for IS-informed State Preparation and IS-QAE pipeline.

Validates:
    1. Grover-Rudolph state prep produces correct statevector
    2. Uniform distribution matches H gates
    3. Correction factor is exact for static-IVT model
    4. Full IS-QAE pipeline gives unbiased estimate (p_IS * c ≈ p_exact)
"""

import numpy as np
import sys
import os
import warnings
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
warnings.filterwarnings('ignore', category=DeprecationWarning)

from quantum.is_state_prep import (
    TiltedDistribution,
    compute_tilted_distribution_simple,
    build_state_prep_circuit,
    verify_state_prep,
    _compute_rotation_angles,
)
from quantum.is_qae import (
    compute_correction_factor,
    compute_correction_exact,
    build_is_qae_circuit,
    is_qae_full_pipeline,
)


# ============================================================
# State Preparation Tests
# ============================================================

class TestGroverRudolphStatePrep:
    """Test Grover-Rudolph state preparation circuit."""

    def test_tilted_distribution_n3(self):
        """Tilted distribution (n=3, barrier=6, q_high=0.7) matches statevector."""
        dist = compute_tilted_distribution_simple(n_ivt=3, barrier=6, q_high=0.7)
        probs_sv = verify_state_prep(dist, verbose=False)
        np.testing.assert_allclose(probs_sv, dist.probs, atol=1e-10)

    def test_uniform_distribution(self):
        """Uniform distribution gives equal amplitudes (same as H gates)."""
        n_levels = 8
        dist = TiltedDistribution(
            n_qubits=3, n_levels=n_levels,
            probs=np.full(n_levels, 1.0 / n_levels),
            q_high=0.25, barrier=6, source='uniform',
        )
        probs_sv = verify_state_prep(dist, verbose=False)
        np.testing.assert_allclose(probs_sv, 1.0 / n_levels, atol=1e-10)

    def test_extreme_tilt(self):
        """Extreme tilt (q_high=0.95) works correctly."""
        dist = compute_tilted_distribution_simple(n_ivt=3, barrier=6, q_high=0.95)
        probs_sv = verify_state_prep(dist, verbose=False)
        np.testing.assert_allclose(probs_sv, dist.probs, atol=1e-10)

    def test_arbitrary_4_level(self):
        """Arbitrary 4-level distribution [0.1, 0.2, 0.3, 0.4]."""
        arb_probs = np.array([0.1, 0.2, 0.3, 0.4])
        dist = TiltedDistribution(
            n_qubits=2, n_levels=4, probs=arb_probs,
            q_high=0.7, barrier=2, source='arbitrary',
        )
        probs_sv = verify_state_prep(dist, verbose=False)
        np.testing.assert_allclose(probs_sv, arb_probs, atol=1e-10)

    def test_n4_16_levels(self):
        """Larger circuit with n=4 (16 levels)."""
        dist = compute_tilted_distribution_simple(n_ivt=4, barrier=12, q_high=0.8)
        probs_sv = verify_state_prep(dist, verbose=False)
        np.testing.assert_allclose(probs_sv, dist.probs, atol=1e-10)

    def test_different_barriers(self):
        """Various barrier levels produce correct distributions."""
        for barrier in [2, 4, 6]:
            dist = compute_tilted_distribution_simple(n_ivt=3, barrier=barrier, q_high=0.7)
            probs_sv = verify_state_prep(dist, verbose=False)
            np.testing.assert_allclose(probs_sv, dist.probs, atol=1e-10,
                                       err_msg=f"Failed for barrier={barrier}")

    def test_prob_above_barrier(self):
        """prob_above_barrier() returns correct value."""
        dist = compute_tilted_distribution_simple(n_ivt=3, barrier=6, q_high=0.7)
        assert abs(dist.prob_above_barrier() - 0.7) < 1e-10

    def test_probs_sum_to_one(self):
        """Tilted distribution sums to 1."""
        dist = compute_tilted_distribution_simple(n_ivt=3, barrier=6, q_high=0.7)
        assert abs(np.sum(dist.probs) - 1.0) < 1e-10


# ============================================================
# Rotation Angle Tests
# ============================================================

class TestRotationAngles:
    """Test the rotation angle computation."""

    def test_uniform_angles(self):
        """Uniform distribution should give all angles = pi/2."""
        probs = np.full(8, 1.0 / 8)
        angles = _compute_rotation_angles(probs, 3)
        for target, ctrl, ctrl_state, theta in angles:
            # All conditional splits should be 50/50 → theta = pi/2
            np.testing.assert_allclose(theta, np.pi / 2, atol=1e-10,
                err_msg=f"target={target}, ctrl_state={ctrl_state}")

    def test_angle_count(self):
        """Number of angles should be 2^n - 1."""
        for n in [2, 3, 4]:
            probs = np.ones(2 ** n) / (2 ** n)
            angles = _compute_rotation_angles(probs, n)
            assert len(angles) == 2 ** n - 1, f"n={n}: expected {2**n - 1}, got {len(angles)}"

    def test_zero_prob_angle(self):
        """Zero probability bins give theta=0 or pi."""
        probs = np.array([0.0, 0.0, 0.5, 0.5])
        angles = _compute_rotation_angles(probs, 2)
        # Level 0: P(MSB=0) = 0.0 → theta = 2*arccos(0) = pi
        target, _, _, theta = angles[0]
        np.testing.assert_allclose(theta, np.pi, atol=1e-10)


# ============================================================
# Likelihood Ratio Tests
# ============================================================

class TestLikelihoodRatio:
    """Test likelihood ratio and correction factor."""

    def test_lr_values(self):
        """Likelihood ratios computed correctly."""
        dist = compute_tilted_distribution_simple(n_ivt=3, barrier=6, q_high=0.7)
        lr = dist.likelihood_ratios()
        # Below barrier: w(k) = (1/8) / (0.3/6) = 0.125 / 0.05 = 2.5
        np.testing.assert_allclose(lr[:6], 2.5, atol=1e-10)
        # Above barrier: w(k) = (1/8) / (0.7/2) = 0.125 / 0.35 = 5/14
        np.testing.assert_allclose(lr[6:], 5.0 / 14.0, atol=1e-10)

    def test_exact_correction_factor(self):
        """Exact correction factor p_IS * c = p."""
        dist = compute_tilted_distribution_simple(n_ivt=3, barrier=6, q_high=0.7)
        c = compute_correction_exact(dist, n_steps=4, window=2)
        p = 0.25
        p_is = 0.7
        np.testing.assert_allclose(p_is * c, p, atol=1e-10)

    def test_mc_correction_converges(self):
        """MC correction factor converges to exact value."""
        dist = compute_tilted_distribution_simple(n_ivt=3, barrier=6, q_high=0.7)
        c_exact = compute_correction_exact(dist, n_steps=4, window=2)
        c_mc, c_se = compute_correction_factor(dist, n_steps=4, window=2,
                                                n_samples=50000, seed=42)
        # MC should be within 3 sigma of exact
        assert abs(c_mc - c_exact) < 3 * c_se + 1e-10, \
            f"MC correction {c_mc:.6f} too far from exact {c_exact:.6f}"

    def test_unbiasedness_identity(self):
        """E_Q[w(X)] = 1 (IS unbiasedness)."""
        dist = compute_tilted_distribution_simple(n_ivt=3, barrier=6, q_high=0.7)
        lr = dist.likelihood_ratios()
        # E_Q[w(X)] = sum_k q_k * w_k = sum_k q_k * (p_k / q_k) = sum_k p_k = 1
        e_w = np.sum(dist.probs * lr)
        np.testing.assert_allclose(e_w, 1.0, atol=1e-10)


# ============================================================
# IS-QAE Circuit Tests
# ============================================================

class TestISQAECircuit:
    """Test IS-QAE circuit construction."""

    def test_circuit_builds(self):
        """IS-QAE circuit builds without error."""
        dist = compute_tilted_distribution_simple(n_ivt=3, barrier=6, q_high=0.7)
        A, flag_idx = build_is_qae_circuit(
            n_ivt=3, n_counter=3, barrier=6, window=2, n_steps=4,
            tilted_dist=dist,
        )
        assert A.num_qubits == 18  # same as standard QAE
        assert flag_idx == 7  # n_ivt + n_counter + 1

    def test_flag_index_consistent(self):
        """Flag qubit index matches standard QAE."""
        from quantum.qae_convergence_demo import build_qae_circuit
        _, flag_std = build_qae_circuit(n_ivt=3, n_counter=3, barrier=6)
        dist = compute_tilted_distribution_simple(n_ivt=3, barrier=6, q_high=0.7)
        _, flag_is = build_is_qae_circuit(
            n_ivt=3, n_counter=3, barrier=6, window=2, n_steps=4,
            tilted_dist=dist,
        )
        assert flag_std == flag_is


# ============================================================
# Full Pipeline Test
# ============================================================

class TestISQAEPipeline:
    """Test the full IS-QAE pipeline."""

    def test_pipeline_correct(self):
        """IS-QAE pipeline gives answer close to p_exact."""
        result = is_qae_full_pipeline(
            n_ivt=3, n_counter=3, barrier=6, window=2, n_steps=4,
            q_high=0.7, qae_max_depth=8, n_shots_qae=200,
            n_correction_samples=10000, seed=42,
        )
        # Should be within 0.05 of exact (stochastic)
        assert abs(result.p_final - result.p_exact) < 0.05, \
            f"p_final={result.p_final:.4f} too far from p_exact={result.p_exact:.4f}"

    def test_p_is_larger_than_p(self):
        """IS-QAE trigger probability is larger than original."""
        result = is_qae_full_pipeline(
            n_ivt=3, n_counter=3, barrier=6, window=2, n_steps=4,
            q_high=0.7, qae_max_depth=4, n_shots_qae=100,
        )
        assert result.p_is > result.p_exact, \
            f"p_IS={result.p_is:.4f} should be > p_exact={result.p_exact:.4f}"


# ============================================================
# Run all tests
# ============================================================

if __name__ == "__main__":
    pytest.main([__file__, '-v', '--tb=short'])
