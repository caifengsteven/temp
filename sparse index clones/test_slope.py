"""
test_slope.py
Quick sanity checks and demo for the SLOPE implementation.
"""

import numpy as np
from slope_solver import prox_sorted_l1, solve_slope_fista, solve_lasso_pg, compute_lambda_sequence_bogdan


def test_prox_sorted_l1():
    """Test prox operator on simple cases with known solutions."""
    print("Testing prox_sorted_l1...")

    # Test 1: All-zero lambda -> prox is identity
    y = np.array([3.0, -1.0, 2.0])
    lam = np.zeros(3)
    v = prox_sorted_l1(y, lam)
    assert np.allclose(v, y), f"Identity failed: {v} vs {y}"
    print("  [PASS] Zero lambda (identity)")

    # Test 2: Very large lambda -> all zeros
    lam = np.ones(3) * 100.0
    v = prox_sorted_l1(y, lam)
    assert np.allclose(v, 0.0), f"Large lambda failed: {v}"
    print("  [PASS] Large lambda (all zeros)")

    # Test 3: LASSO-equivalent (constant lambda)
    y = np.array([3.0, -1.0, 0.5])
    lam = np.ones(3) * 1.0
    v = prox_sorted_l1(y, lam)
    # For constant lambda, SLOPE = LASSO prox = soft-thresholding
    v_lasso = np.sign(y) * np.maximum(np.abs(y) - 1.0, 0.0)
    assert np.allclose(v, v_lasso), f"LASSO equiv failed: {v} vs {v_lasso}"
    print("  [PASS] Constant lambda (LASSO equivalence)")

    # Test 4: Grouping property - similar values should be grouped
    # Set up y where two components are equal after thresholding
    y = np.array([2.0, 2.0, 0.1])
    lam = np.array([1.5, 0.5, 0.1])
    v = prox_sorted_l1(y, lam)
    # The two largest should be grouped (equal) after prox
    print(f"  Grouping test: y={y}, lam={lam}, v={v}")
    print("  [INFO] First two components should be close due to grouping")

    print("prox_sorted_l1 tests completed.\n")


def test_simulation_quick():
    """Run a tiny simulation to verify end-to-end."""
    print("Running quick simulation test (5 iterations, 3 alphas)...")

    from slope_simulation import run_simulation_scenario

    alpha_grid = np.logspace(-3.0, -2.0, 3)
    results = run_simulation_scenario(
        scenario=1, n_iter=5, alpha_grid=alpha_grid,
        K=99, T=500, seed_offset=42, verbose=False
    )

    print(f"  Scenario 1, 5 iters, 3 alphas")
    print(f"  Elapsed: {results['elapsed_sec']:.2f}s")
    print(f"  Mean non-zero (last alpha): {results['n_nonzero'][:, -1].mean():.1f}")
    print(f"  Mean groups   (last alpha): {results['n_groups'][:, -1].mean():.1f}")
    print(f"  Mean MSE      (last alpha): {results['mse'][:, -1].mean():.4f}")
    print("Quick simulation test completed.\n")


def test_constrained_solver():
    """Test SLOPE with simplex and box constraints."""
    print("Testing constrained SLOPE solvers...")

    rng = np.random.default_rng(123)
    T, K = 100, 10
    R = rng.standard_normal((T, K))
    w_true = np.zeros(K)
    w_true[2:5] = 0.3
    w_true[7:9] = 0.2
    Y = R @ w_true + rng.normal(0, 0.1, T)

    lam = compute_lambda_sequence_bogdan(K, alpha=0.01, theta=0.1)

    # Unconstrained
    w_unc, _ = solve_slope_fista(R, Y, lam, constraint='none', verbose=False)
    print(f"  Unconstrained: sum={np.sum(w_unc):.4f}, nz={np.sum(np.abs(w_unc)>1e-4)}")

    # Simplex
    w_sim, _ = solve_slope_fista(R, Y, lam, constraint='simplex', verbose=False)
    print(f"  Simplex:       sum={np.sum(w_sim):.4f}, min={np.min(w_sim):.4f}, nz={np.sum(np.abs(w_sim)>1e-4)}")
    assert np.abs(np.sum(w_sim) - 1.0) < 1e-4, "Simplex sum failed"
    assert np.min(w_sim) >= -1e-6, "Simplex non-negativity failed"
    print("  [PASS] Simplex constraints")

    # Box + hyperplane
    w_box, _ = solve_slope_fista(R, Y, lam, constraint='box_hyperplane', verbose=False)
    print(f"  Box+Hyper:     sum={np.sum(w_box):.4f}, min={np.min(w_box):.4f}, max={np.max(w_box):.4f}, nz={np.sum(np.abs(w_box)>1e-4)}")
    assert np.abs(np.sum(w_box) - 1.0) < 1e-4, "Box sum failed"
    assert np.min(w_box) >= -1.0 - 1e-6 and np.max(w_box) <= 1.0 + 1e-6, "Box bounds failed"
    print("  [PASS] Box+hyperplane constraints")

    print("Constrained solver tests completed.\n")


if __name__ == '__main__':
    test_prox_sorted_l1()
    test_constrained_solver()
    test_simulation_quick()
    print("All tests passed!")
