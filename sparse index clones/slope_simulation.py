"""
slope_simulation.py
Replication of Section 3 simulation study from Kremer et al. (2022).

Simulation design (Hidden Factor Model):
- T = 500 observations, K = 99 assets, S = 3 factors
- Factors: F ~ N(0, I_S)
- Loadings B: 33 copies each of [0.77,0.64,0]', [0.9,0,0.42]', [0,0.31,0.64]'
- Asset returns: R = F*B + epsilon,  epsilon ~ N(0, 0.05*I_K)
- Columns of R normalized to unit norm
- Index: Y = R*w_true + nu,  nu ~ N(0, sigma^2), sigma = 0.0015

Scenarios:
- Scenario 1: w = [0(33), 2(33), 3(33)]
- Scenario 2: w = [0(33), 1(16), 2(17), 3(33)]

Lambda sequences: 12 log-spaced alpha values between 10^-3.5 and 10^-1.7.
For each alpha: lambda_i = alpha * Phi^{-1}(1 - i*theta/(2K)), theta=0.1.

Metrics (per iteration and lambda sequence):
- Number of non-zero coefficients
- Number of groups (unique non-zero values)
- MSE  = ||w_true - w_hat||^2
- MSPE = ||R*w_true - R*w_hat||^2
"""

import numpy as np
import pandas as pd
from scipy import stats
from slope_solver import (solve_slope_fista, solve_lasso_pg,
                          compute_lambda_sequence_bogdan, count_groups)
import time
import json


def generate_data(T=500, K=99, S=3, sigma_nu=0.0015, seed=None):
    """
    Generate simulated data following the hidden factor model.

    Returns
    -------
    R : ndarray, shape (T, K)
    w_true : ndarray, shape (K,), placeholder (set by caller)
    B : ndarray, shape (S, K)
    """
    rng = np.random.default_rng(seed)

    # Factors: independent N(0, I_S)
    F = rng.standard_normal((T, S))

    # Loading matrix: 33 copies each of three factor exposures
    b1 = np.array([0.77, 0.64, 0.0])
    b2 = np.array([0.90, 0.0, 0.42])
    b3 = np.array([0.0, 0.31, 0.64])

    B = np.zeros((S, K))
    for i in range(33):
        B[:, i] = b1
        B[:, i + 33] = b2
        B[:, i + 66] = b3

    # Asset-specific noise
    epsilon = rng.normal(0.0, np.sqrt(0.05), (T, K))
    R = F @ B + epsilon

    # Normalize each column to have Euclidean norm = 1
    col_norms = np.linalg.norm(R, axis=0)
    col_norms[col_norms == 0] = 1.0  # guard
    R = R / col_norms

    return R, B


def run_simulation_scenario(scenario=1, n_iter=1000, alpha_grid=None,
                            K=99, T=500, seed_offset=0, verbose=True):
    """
    Run full Monte Carlo simulation for one scenario.

    Parameters
    ----------
    scenario : int, {1, 2}
    n_iter : int, number of Monte Carlo iterations
    alpha_grid : ndarray or None; if None uses paper's 12-point grid
    K, T : simulation dimensions
    seed_offset : int, added to iteration index for RNG seeding
    verbose : bool

    Returns
    -------
    results : dict with arrays of shape (n_iter, n_alpha)
    """
    if alpha_grid is None:
        # 12 log-spaced points between 10^-3.5 and 10^-1.7 (paper's grid)
        alpha_grid = np.logspace(-3.5, -1.7, 12)

    n_alpha = len(alpha_grid)

    # Pre-compute lambda sequences for all alphas
    lambda_sequences = []
    for alpha in alpha_grid:
        lam = compute_lambda_sequence_bogdan(K, alpha, theta=0.1)
        lambda_sequences.append(lam)

    # Pre-allocate result arrays
    n_nonzero = np.zeros((n_iter, n_alpha), dtype=int)
    n_groups = np.zeros((n_iter, n_alpha), dtype=int)
    mse = np.zeros((n_iter, n_alpha))
    mspe = np.zeros((n_iter, n_alpha))

    # LASSO comparison (lambda_1 = lambda_K = same alpha-based penalty)
    n_nonzero_lasso = np.zeros((n_iter, n_alpha), dtype=int)
    mse_lasso = np.zeros((n_iter, n_alpha))
    mspe_lasso = np.zeros((n_iter, n_alpha))

    t_start = time.time()

    for it in range(n_iter):
        if verbose and (it + 1) % 100 == 0:
            elapsed = time.time() - t_start
            print(f"  Iteration {it + 1}/{n_iter}  ({elapsed:.1f}s elapsed)")

        # Generate data
        R, B = generate_data(T=T, K=K, seed=seed_offset + it)

        # True weight vector
        w_true = np.zeros(K)
        if scenario == 1:
            w_true[33:66] = 2.0
            w_true[66:] = 3.0
        elif scenario == 2:
            w_true[33:49] = 1.0   # first half of group 2  (16 assets)
            w_true[49:66] = 2.0   # second half of group 2 (17 assets)
            w_true[66:] = 3.0
        else:
            raise ValueError("scenario must be 1 or 2")

        # Generate index returns
        rng = np.random.default_rng(seed_offset + it + 100000)
        nu = rng.normal(0.0, 0.0015, T)
        Y = R @ w_true + nu

        # Warm start: use previous alpha solution as initial guess
        w0 = None
        w0_lasso = None

        for ia, alpha in enumerate(alpha_grid):
            lam = lambda_sequences[ia]

            # --- SLOPE (unconstrained, as in paper simulation) ---
            w_hat, info = solve_slope_fista(
                R, Y, lam, w0=w0, constraint='none',
                max_iter=3000, tol=1e-7, verbose=False
            )

            # Threshold small weights (paper uses 0.05% = 0.0005)
            w_hat[np.abs(w_hat) < 0.0005] = 0.0

            # Metrics
            n_nonzero[it, ia] = int(np.sum(np.abs(w_hat) > 0))
            n_groups[it, ia] = count_groups(w_hat, tol=1e-4)
            mse[it, ia] = np.sum((w_true - w_hat) ** 2)
            mspe[it, ia] = np.sum((R @ (w_true - w_hat)) ** 2)

            w0 = w_hat.copy()  # warm start for next alpha

            # --- LASSO baseline (lambda = alpha * Phi^{-1}(1 - theta/(2K))) ---
            # Use the first lambda as constant LASSO penalty
            lambda_lasso = lam[0]
            w_lasso, info_l = solve_lasso_pg(
                R, Y, lambda_lasso, w0=w0_lasso, constraint='none',
                max_iter=3000, tol=1e-7, verbose=False
            )
            w_lasso[np.abs(w_lasso) < 0.0005] = 0.0

            n_nonzero_lasso[it, ia] = int(np.sum(np.abs(w_lasso) > 0))
            mse_lasso[it, ia] = np.sum((w_true - w_lasso) ** 2)
            mspe_lasso[it, ia] = np.sum((R @ (w_true - w_lasso)) ** 2)
            w0_lasso = w_lasso.copy()

    elapsed = time.time() - t_start
    if verbose:
        print(f"\nScenario {scenario} completed in {elapsed:.1f}s "
              f"({elapsed/n_iter:.2f}s per iteration)")

    return {
        'scenario': scenario,
        'n_iter': n_iter,
        'alpha_grid': alpha_grid,
        'K': K, 'T': T,
        # SLOPE
        'n_nonzero': n_nonzero,
        'n_groups': n_groups,
        'mse': mse,
        'mspe': mspe,
        # LASSO
        'n_nonzero_lasso': n_nonzero_lasso,
        'mse_lasso': mse_lasso,
        'mspe_lasso': mspe_lasso,
        'elapsed_sec': elapsed
    }


def print_simulation_summary(results):
    """Print summary statistics matching paper's exhibit format."""
    alpha_grid = results['alpha_grid']
    n_alpha = len(alpha_grid)

    print(f"\n{'='*80}")
    print(f"Simulation Results - Scenario {results['scenario']} "
          f"({results['n_iter']} iterations, K={results['K']}, T={results['T']})")
    print(f"{'='*80}")

    # SLOPE table
    print("\n--- SLOPE ---")
    print(f"{'Alpha':>12} {'Non-zero':>10} {'Groups':>8} "
          f"{'MSE':>14} {'MSPE':>14}")
    print("-" * 60)
    for ia in range(n_alpha):
        print(f"{alpha_grid[ia]:>12.6f} "
              f"{results['n_nonzero'][:,ia].mean():>10.2f} "
              f"{results['n_groups'][:,ia].mean():>8.2f} "
              f"{results['mse'][:,ia].mean():>14.4f} "
              f"{results['mspe'][:,ia].mean():>14.4f}")

    # LASSO table
    print("\n--- LASSO (baseline) ---")
    print(f"{'Alpha':>12} {'Non-zero':>10} {'Groups':>8} "
          f"{'MSE':>14} {'MSPE':>14}")
    print("-" * 60)
    for ia in range(n_alpha):
        print(f"{alpha_grid[ia]:>12.6f} "
              f"{results['n_nonzero_lasso'][:,ia].mean():>10.2f} "
              f"{results['n_nonzero_lasso'][:,ia].mean():>8.2f} "
              f"{results['mse_lasso'][:,ia].mean():>14.4f} "
              f"{results['mspe_lasso'][:,ia].mean():>14.4f}")


def save_results(results, filepath):
    """Save simulation results to NPZ file."""
    np.savez(filepath, **results)
    print(f"Results saved to {filepath}")


def load_results(filepath):
    """Load simulation results from NPZ file."""
    data = np.load(filepath, allow_pickle=True)
    results = {k: data[k] for k in data.files}
    # Convert arrays back to correct types
    for k in ['scenario', 'n_iter', 'K', 'T']:
        if k in results:
            results[k] = int(results[k])
    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='SLOPE Simulation Study')
    parser.add_argument('--scenario', type=int, default=1, choices=[1, 2])
    parser.add_argument('--n-iter', type=int, default=1000,
                        help='Number of Monte Carlo iterations')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 20 iterations and 6 alpha values')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save results (npz)')
    args = parser.parse_args()

    if args.quick:
        print("QUICK TEST MODE: 20 iterations, 6 alpha values\n")
        alpha_grid = np.logspace(-3.5, -1.7, 6)
        n_iter = 20
    else:
        alpha_grid = None
        n_iter = args.n_iter

    results = run_simulation_scenario(
        scenario=args.scenario,
        n_iter=n_iter,
        alpha_grid=alpha_grid,
        verbose=True
    )

    print_simulation_summary(results)

    if args.save:
        save_results(results, args.save)
