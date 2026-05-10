# SLOPE Index Tracking & Hedge Fund Replication

Comprehensive Python implementation of **Kremer et al. (2022)**, "Sparse index clones via the sorted l1-Norm", *Quantitative Finance*, Vol. 22, No. 2, 349-366.

---

## Files

| File | Description |
|------|-------------|
| `slope_solver.py` | Core SLOPE optimization: sorted-l1 prox operator (PAVA), FISTA solver, LASSO baseline, simplex/box projections |
| `slope_simulation.py` | Replication of Section 3 simulation study (hidden factor model, 2 scenarios, 1000 MC iterations) |
| `slope_empirical.py` | Rolling-window empirical framework for equity index tracking & hedge fund replication |
| `demo_empirical.py` | Self-contained demo with synthetic data |
| `test_slope.py` | Unit tests for prox operator, constraints, and quick simulation |
| `349_r_pacage.md` | Markdown extraction of the PDF paper |

---

## Quick Start

### 1. Run tests
```bash
python3 test_slope.py
```

### 2. Run demo with synthetic data
```bash
python3 demo_empirical.py
```

### 3. Run full simulation (Scenario 1, 1000 iterations)
```bash
python3 slope_simulation.py --scenario 1 --n-iter 1000 --save results_sc1.npz
```

Quick test mode (20 iterations, 6 alphas):
```bash
python3 slope_simulation.py --scenario 1 --quick
```

---

## Methodology

### SLOPE Problem

```
minimize   ||Y - R*w||^2_2 + sum(lambda_i * |w|_(i))
subject to 1'w = 1  (+ optional box constraints)
```

where `|w|_(i)` is the i-th largest absolute coefficient and `lambda_1 >= lambda_2 >= ... >= lambda_K >= 0`.

### Key Properties

- **Sparsity**: promotes zero coefficients (like LASSO)
- **Grouping**: assigns equal weights to assets with similar partial correlations
- **Convex**: globally optimizable (unlike SCAD/LOG)

### SLOPE-SLC Strategy

1. Solve SLOPE to get weights and implicit groups
2. For each group, compute median partial correlation with index residual
3. Keep only groups above a percentile threshold (75% for equity, 25% for hedge funds)
4. Rescale weights to sum to 1

---

## Simulation Design (Section 3)

**Data Generating Process**:
- Hidden factor model: `R = F*B + epsilon`
- T=500, K=99, S=3 factors
- Loadings: 33 copies each of `[0.77,0.64,0]`, `[0.9,0,0.42]`, `[0,0.31,0.64]`
- `epsilon ~ N(0, 0.05*I)`, columns of R normalized to unit norm
- `Y = R*w_true + nu`, `nu ~ N(0, 0.0015^2)`

**Scenarios**:
- Scenario 1: `w = [0(33), 2(33), 3(33)]`
- Scenario 2: `w = [0(33), 1(16), 2(17), 3(33)]`

**Lambda Sequences** (12 values, log-spaced):
- `alpha in [10^-3.5, 10^-1.7]`
- `lambda_i = alpha * Phi^{-1}(1 - i*theta/(2K))`, `theta=0.1`

**Metrics**: Non-zero count, groups, MSE, MSPE

---

## Empirical Framework (Section 4)

### Equity Index Tracking
- **Data**: S&P 100/200/500 constituents (daily)
- **Window**: tau = 750 days
- **Rebalance**: every 21 days (monthly)
- **Constraint**: simplex (`w >= 0`, `sum(w) = 1`)
- **Strategies**: SLOPE, SLOPE-SLC, LASSO

### Hedge Fund Replication
- **Data**: 26 HFR indices (monthly) + 17 risk factors
- **Window**: tau = 60 months
- **Constraint**: box + hyperplane (`-1 <= w <= 1`, `sum(w) = 1`)
- **Strategies**: SLOPE, SLOPE-LO, SLOPE-SLC, SLOPE-LO-SLC, LASSO

### Usage

```python
from slope_empirical import rolling_window_backtest
from slope_solver import compute_lambda_sequence_bogdan

# Your data: R (T x K constituent returns), Y (T index returns)
K = R.shape[1]
lambda_seq = compute_lambda_sequence_bogdan(K, alpha=0.01, theta=0.1)

results = rolling_window_backtest(
    R, Y, window_size=750, step_size=21,
    lambda_seq=lambda_seq, strategy='slope-slc',
    constraint='simplex', percentile=75.0
)

print(f"TE Volatility: {results['te_vol']:.4f}")
print(f"Mean Sparsity: {results['mean_sparsity']:.1f}")
```

---

## Solver Details

### Proximal Operator of Sorted l1-Norm

Implemented via **Pool Adjacent Violators Algorithm (PAVA)**:
1. Sort `|y|` descending
2. Compute `x = |y| - lambda`
3. Apply increasing PAVA on reversed `x` -> decreasing isotonic regression
4. Clip to non-negative, unsort, restore signs

Complexity: `O(K log K)` per prox evaluation.

### FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)

```
z_{k+1} = w_k + ((t_k - 1) / t_{k+1}) * (w_k - w_{k-1})
w_{k+1} = Proj_C( prox_{eta*lambda}(z_k - eta * grad_f(z_k)) )
```

Step size `eta = 1/L` where `L = 2*||R'R||_2`.

### Constraint Projections

- **Simplex**: Duchi et al. (2008) algorithm, `O(K log K)`
- **Box + Hyperplane**: Bisection on dual variable tau, `w_i = clip(v_i - tau, a, b)`

---

## Performance Notes

- Simulation: ~3.5s per iteration (K=99, T=500, 12 alphas)
  - 1000 iterations full replication ≈ 60 minutes
- Empirical: ~0.5-2s per rebalance depending on K and constraint

To speed up:
- Use warm starts across lambda grid or time steps
- Reduce `max_iter` or increase `tol` for prototyping
- Use `--quick` flag for simulation testing

---

## References

- Kremer, P.J., Brzyski, D., Bogdan, M., Paterlini, S. (2022). Sparse index clones via the sorted l1-Norm. *Quantitative Finance*, 22(2), 349-366.
- Bogdan, M., van den Berg, E., Sabatti, C., Su, W., Candes, E.J. (2015). SLOPE -- adaptive variable selection via convex optimization. *Annals of Applied Statistics*, 9(3), 1103-1140.
- Duchi, J., Shalev-Shwartz, S., Singer, Y., Chandra, T. (2008). Efficient projections onto the l1-ball for learning in high dimensions. *ICML*.

---

## Requirements

- Python 3.8+
- NumPy
- SciPy
- pandas
- matplotlib (optional, for demo plots)
