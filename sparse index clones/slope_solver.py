"""
slope_solver.py
Core SLOPE (Sorted L1 Penalized Estimator) optimization for index tracking.
Based on: Kremer et al. (2022), "Sparse index clones via the sorted l1-Norm"

Implements:
- Proximal operator of the sorted l1-norm via PAVA (Pool Adjacent Violators Algorithm)
- FISTA solver for SLOPE with optional constraints (simplex, box+hyperplane)
- LASSO baseline solver
- Lambda sequence generator following Bogdan et al. (2013)
"""

import numpy as np
from scipy import stats
import warnings


def prox_sorted_l1(y, lam):
    """
    Compute the proximal operator of the sorted l1-norm.

    J_lambda(w) = sum_{i=1}^n lambda_i * |w|_(i)

    Solves:  min_v  { 0.5 * ||v - y||^2 + J_lambda(v) }

    Algorithm: sort |y| descending, apply isotonic regression of (|y| - lam)
    onto the decreasing cone {v : v_1 >= ... >= v_n >= 0} via PAVA,
    then unsort and restore signs.

    Parameters
    ----------
    y : ndarray, shape (n,)
    lam : ndarray, shape (n,), non-increasing sequence (lam[0] >= lam[1] >= ...)

    Returns
    -------
    v : ndarray, shape (n,)
    """
    n = len(y)
    if len(lam) != n:
        raise ValueError("lam must have same length as y")

    # Signs and sort |y| in descending order
    s = np.sign(y)
    abs_y = np.abs(y)
    idx = np.argsort(-abs_y, kind='mergesort')
    y_sorted = abs_y[idx]
    lam_sorted = lam  # assumed already in descending order

    # Isotonic regression of (y_sorted - lam) onto decreasing cone.
    # Equivalent to: increasing PAVA on the reversed array.
    x = y_sorted - lam_sorted
    x_rev = x[::-1].copy()

    # PAVA for increasing isotonic regression on x_rev
    blocks = [[i, i + 1, float(x_rev[i])] for i in range(n)]  # [start, end, value)

    i = 0
    while i < len(blocks) - 1:
        if blocks[i][2] > blocks[i + 1][2]:
            # Merge blocks i and i+1
            start = blocks[i][0]
            end = blocks[i + 1][1]
            size_i = blocks[i][1] - blocks[i][0]
            size_j = blocks[i + 1][1] - blocks[i + 1][0]
            val = (blocks[i][2] * size_i + blocks[i + 1][2] * size_j) / (size_i + size_j)
            blocks[i] = [start, end, val]
            del blocks[i + 1]
            # Backtrack to maintain increasing order
            while i > 0 and blocks[i - 1][2] > blocks[i][2]:
                i -= 1
                start = blocks[i][0]
                end = blocks[i + 1][1]
                size_i = blocks[i][1] - blocks[i][0]
                size_j = blocks[i + 1][1] - blocks[i + 1][0]
                val = (blocks[i][2] * size_i + blocks[i + 1][2] * size_j) / (size_i + size_j)
                blocks[i] = [start, end, val]
                del blocks[i + 1]
        else:
            i += 1

    # Reconstruct solution
    v_rev = np.zeros(n)
    for start, end, val in blocks:
        v_rev[start:end] = val

    # Reverse back and project to non-negative orthant
    v_sorted = v_rev[::-1]
    v_sorted = np.maximum(v_sorted, 0.0)

    # Unsort to original order
    inv_idx = np.empty_like(idx)
    inv_idx[idx] = np.arange(n)
    v = v_sorted[inv_idx]

    # Apply original signs
    return s * v


def project_simplex(v, s=1.0):
    """
    Project v onto the simplex {w : w >= 0, sum(w) = s}.
    Algorithm from Duchi et al. (2008), O(n log n).
    """
    if s <= 0:
        raise ValueError("s must be positive")
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - s
    ind = np.arange(n) + 1
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.zeros(n)
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = np.maximum(v - theta, 0)
    return w


def project_box_hyperplane(v, a=-1.0, b=1.0, c=1.0, max_iter=200, tol=1e-12):
    """
    Project v onto {w : a <= w_i <= b, sum(w) = c}.
    Uses bisection on dual variable tau:  w_i = clip(v_i - tau, a, b).
    """
    def phi(tau):
        return np.sum(np.clip(v - tau, a, b)) - c

    # Find initial bracket
    tau_min = np.min(v) - b - 1.0
    tau_max = np.max(v) - a + 1.0

    # Expand bracket if needed
    for _ in range(50):
        if phi(tau_min) <= 0 and phi(tau_max) >= 0:
            break
        if phi(tau_min) > 0:
            tau_min *= 2.0
        if phi(tau_max) < 0:
            tau_max *= 2.0

    # Bisection
    for _ in range(max_iter):
        tau_mid = (tau_min + tau_max) / 2.0
        p = phi(tau_mid)
        if np.abs(p) < tol:
            return np.clip(v - tau_mid, a, b)
        if p > 0:
            tau_min = tau_mid
        else:
            tau_max = tau_mid
        if tau_max - tau_min < tol:
            break

    tau = (tau_min + tau_max) / 2.0
    return np.clip(v - tau, a, b)


def solve_slope_fista(R, Y, lambda_seq, w0=None, constraint='none',
                      max_iter=5000, tol=1e-8, verbose=False):
    """
    Solve SLOPE index tracking via FISTA.

    minimize   ||Y - R*w||^2_2 + sum(lambda_i * |w|_(i)) + I_C(w)

    Parameters
    ----------
    R : ndarray, shape (T, K)
    Y : ndarray, shape (T,)
    lambda_seq : ndarray, shape (K,), non-increasing penalty sequence
    w0 : ndarray, shape (K,), initial guess
    constraint : {'none', 'simplex', 'box_hyperplane'}
        - 'none'           : unconstrained
        - 'simplex'        : w >= 0, sum(w) = 1
        - 'box_hyperplane' : -1 <= w <= 1, sum(w) = 1
    max_iter : int
    tol : float, convergence tolerance on ||w_{k+1} - w_k||
    verbose : bool

    Returns
    -------
    w : ndarray, shape (K,)
    info : dict with keys 'iter', 'converged', 'final_obj'
    """
    T, K = R.shape
    if len(Y) != T:
        raise ValueError("Y must have length T")
    if len(lambda_seq) != K:
        raise ValueError("lambda_seq must have length K")
    if not np.all(lambda_seq[:-1] >= lambda_seq[1:] - 1e-12):
        warnings.warn("lambda_seq should be non-increasing; sorting descending.")
        lambda_seq = np.sort(lambda_seq)[::-1]

    if w0 is None:
        if constraint == 'simplex':
            w0 = np.ones(K) / K
        elif constraint == 'box_hyperplane':
            w0 = np.ones(K) / K
        else:
            w0 = np.zeros(K)

    # Precompute R'R and R'Y
    RtR = R.T @ R
    RtY = R.T @ Y

    # Lipschitz constant of gradient of f(w) = ||Y - R*w||^2_2
    L = 2.0 * np.linalg.norm(RtR, ord=2)
    if L < 1e-14:
        warnings.warn("R'R has near-zero spectral norm")
        L = 1e-14
    eta = 1.0 / L

    w = w0.copy()
    z = w.copy()  # FISTA auxiliary point
    t = 1.0

    for k in range(max_iter):
        w_prev = w.copy()

        # Gradient at z
        grad = 2.0 * (RtR @ z - RtY)

        # Proximal gradient step
        w_new = prox_sorted_l1(z - eta * grad, eta * lambda_seq)

        # Project onto constraint set
        if constraint == 'simplex':
            w_new = project_simplex(w_new)
        elif constraint == 'box_hyperplane':
            w_new = project_box_hyperplane(w_new, a=-1.0, b=1.0, c=1.0)

        # FISTA momentum update
        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = w_new + ((t - 1.0) / t_new) * (w_new - w_prev)

        # Convergence check
        if np.linalg.norm(w_new - w_prev) < tol:
            resid = Y - R @ w_new
            obj = np.dot(resid, resid)
            if verbose:
                print(f"SLOPE converged in {k+1} iterations, obj={obj:.6e}")
            return w_new, {'iter': k+1, 'converged': True, 'final_obj': obj}

        w = w_new
        t = t_new

    resid = Y - R @ w
    obj = np.dot(resid, resid)
    if verbose:
        print(f"SLOPE did not converge in {max_iter} iterations, obj={obj:.6e}")
    return w, {'iter': max_iter, 'converged': False, 'final_obj': obj}


def solve_lasso_pg(R, Y, lambda_lasso, w0=None, constraint='none',
                   max_iter=5000, tol=1e-8, verbose=False):
    """
    Solve LASSO index tracking via FISTA.

    minimize   ||Y - R*w||^2_2 + lambda * ||w||_1 + I_C(w)
    """
    T, K = R.shape
    if w0 is None:
        if constraint == 'simplex':
            w0 = np.ones(K) / K
        elif constraint == 'box_hyperplane':
            w0 = np.ones(K) / K
        else:
            w0 = np.zeros(K)

    RtR = R.T @ R
    RtY = R.T @ Y
    L = 2.0 * np.linalg.norm(RtR, ord=2)
    if L < 1e-14:
        L = 1e-14
    eta = 1.0 / L

    def prox_l1(v, lam):
        return np.sign(v) * np.maximum(np.abs(v) - lam, 0.0)

    w = w0.copy()
    z = w.copy()
    t = 1.0

    for k in range(max_iter):
        w_prev = w.copy()
        grad = 2.0 * (RtR @ z - RtY)
        w_new = prox_l1(z - eta * grad, eta * lambda_lasso)

        if constraint == 'simplex':
            w_new = project_simplex(w_new)
        elif constraint == 'box_hyperplane':
            w_new = project_box_hyperplane(w_new, a=-1.0, b=1.0, c=1.0)

        t_new = (1.0 + np.sqrt(1.0 + 4.0 * t * t)) / 2.0
        z = w_new + ((t - 1.0) / t_new) * (w_new - w_prev)

        if np.linalg.norm(w_new - w_prev) < tol:
            resid = Y - R @ w_new
            obj = np.dot(resid, resid)
            if verbose:
                print(f"LASSO converged in {k+1} iterations, obj={obj:.6e}")
            return w_new, {'iter': k+1, 'converged': True, 'final_obj': obj}

        w = w_new
        t = t_new

    resid = Y - R @ w
    obj = np.dot(resid, resid)
    if verbose:
        print(f"LASSO did not converge in {max_iter} iterations, obj={obj:.6e}")
    return w, {'iter': max_iter, 'converged': False, 'final_obj': obj}


def compute_lambda_sequence_bogdan(K, alpha, theta=0.1):
    """
    Generate SLOPE lambda sequence following Bogdan et al. (2013).

    lambda_i = alpha * Phi^{-1}(1 - q_i)
    q_i      = i * theta / (2*K)

    Parameters
    ----------
    K : int, number of assets
    alpha : float, scale parameter
    theta : float, default 0.1

    Returns
    -------
    lam : ndarray, shape (K,), non-increasing
    """
    i = np.arange(1, K + 1)
    q = i * theta / (2.0 * K)
    q = np.minimum(q, 0.99999)
    lam = alpha * stats.norm.ppf(1.0 - q)
    lam = np.maximum(lam, 0.0)  # ensure non-negative
    return lam


def count_groups(w, tol=1e-6):
    """Count number of unique non-zero coefficient values (groups)."""
    nz = w[np.abs(w) > tol]
    if len(nz) == 0:
        return 0
    return len(np.unique(np.round(nz, 6)))
