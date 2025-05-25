import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize_scalar, brentq
from scipy.integrate import quad
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')

class LeveragedETFReplicator:
    """
    Implements the optimal replication strategy for leveraged and inverse ETFs
    based on the Guasoni & Mayerhofer (2023) paper
    """
    
    def __init__(self, leverage_factor, spread, gamma, risk_premium_kappa=0):
        """
        Parameters:
        -----------
        leverage_factor : float
            Target leverage factor (L). Positive for leveraged, negative for inverse
        spread : float
            Bid-ask spread (epsilon)
        gamma : float
            Tracking error aversion parameter
        risk_premium_kappa : float
            Risk premium parameter (μ = κσ²)
        """
        self.L = leverage_factor
        self.epsilon = spread
        self.gamma = gamma
        self.kappa = risk_premium_kappa
        
        # Calculate optimal trading boundaries
        self.pi_minus, self.pi_plus = self._calculate_boundaries()
        
    def _calculate_boundaries(self):
        """
        Calculate optimal trading boundaries using asymptotic formulas
        """
        L = self.L
        epsilon = self.epsilon
        gamma = self.gamma
        kappa = self.kappa
        
        # Handle the calculation carefully to avoid complex numbers
        # First-order approximation from equation (16)
        # Note: For complex expressions, we need to handle the cube root carefully
        inner_term = 3 / (4 * gamma) * L**2 * (L - 1)**2
        
        # For negative values, we use the real cube root
        if inner_term >= 0:
            delta = inner_term**(1/3) * epsilon**(1/3)
        else:
            # For negative numbers, take the real cube root
            delta = -(-inner_term)**(1/3) * epsilon**(1/3)
        
        # Second-order correction
        correction_inner = (gamma - kappa) / gamma * (L - 1) / 6
        if correction_inner >= 0:
            correction = correction_inner**(1/3) * epsilon**(2/3)
        else:
            correction = -(-correction_inner)**(1/3) * epsilon**(2/3)
        
        pi_plus = L + delta - correction
        pi_minus = L - delta - correction
        
        # Ensure boundaries are properly ordered
        if pi_minus > pi_plus:
            pi_minus, pi_plus = pi_plus, pi_minus
        
        return pi_minus, pi_plus
    
    def simulate_index(self, T, dt, sigma, mu=0, S0=100):
        """
        Simulate index price path
        """
        n_steps = int(T / dt)
        times = np.linspace(0, T, n_steps + 1)
        
        # Generate Brownian motion
        dW = np.sqrt(dt) * np.random.randn(n_steps)
        
        # Simulate log prices
        log_S = np.zeros(n_steps + 1)
        log_S[0] = np.log(S0)
        
        for i in range(n_steps):
            log_S[i+1] = log_S[i] + (mu - 0.5 * sigma**2) * dt + sigma * dW[i]
        
        S = np.exp(log_S)
        returns = np.diff(log_S)
        
        return times, S, returns
    
    def replicate(self, times, S, F0=100):
        """
        Implement the optimal replication strategy
        """
        n_steps = len(times) - 1
        dt = times[1] - times[0]
        
        # Initialize arrays
        F = np.zeros(n_steps + 1)  # Fund value
        pi = np.zeros(n_steps + 1)  # Portfolio weight
        shares = np.zeros(n_steps + 1)  # Number of shares
        cash = np.zeros(n_steps + 1)  # Cash position
        trades = np.zeros(n_steps)  # Trading volume
        costs = np.zeros(n_steps)  # Transaction costs
        
        # Initial conditions
        F[0] = F0
        pi[0] = self.L
        shares[0] = self.L * F0 / S[0]
        cash[0] = F0 - shares[0] * S[0]
        
        for i in range(n_steps):
            # Update fund value before rebalancing
            F_before = cash[i] + shares[i] * S[i+1]
            
            # Handle the case where fund value becomes very small or negative
            if F_before <= 1e-6:
                # Fund is essentially bankrupt
                F[i+1:] = 0
                break
            
            pi_before = shares[i] * S[i+1] / F_before
            
            # Check if rebalancing is needed
            if pi_before < self.pi_minus:
                # Buy to reach lower boundary
                pi_target = self.pi_minus
            elif pi_before > self.pi_plus:
                # Sell to reach upper boundary
                pi_target = self.pi_plus
            else:
                # No trade needed
                pi_target = pi_before
            
            # Calculate target shares
            shares_target = pi_target * F_before / S[i+1]
            shares_traded = shares_target - shares[i]
            
            # Apply transaction costs
            cost = abs(shares_traded) * S[i+1] * self.epsilon
            
            costs[i] = cost
            trades[i] = abs(shares_traded) * S[i+1]
            
            # Update positions
            shares[i+1] = shares_target
            cash[i+1] = F_before - shares[i+1] * S[i+1] - cost
            F[i+1] = cash[i+1] + shares[i+1] * S[i+1]
            
            # Update portfolio weight
            if F[i+1] > 0:
                pi[i+1] = shares[i+1] * S[i+1] / F[i+1]
            else:
                pi[i+1] = 0
        
        return {
            'F': F,
            'pi': pi,
            'shares': shares,
            'cash': cash,
            'trades': trades,
            'costs': costs
        }
    
    def calculate_performance_metrics(self, times, S, F, pi, trades, costs):
        """
        Calculate tracking error, tracking difference, and other metrics
        """
        dt = times[1] - times[0]
        n_steps = len(times) - 1
        
        # Find valid data (before any bankruptcy)
        valid_idx = F > 0
        if not any(valid_idx[1:]):
            # Fund went bankrupt immediately
            return {
                'tracking_difference': -100,
                'tracking_error': 100,
                'average_exposure': 0,
                'turnover': 0,
                'cost_drag': 100,
                'r_squared': 0,
                'total_return': -100,
                'index_return': (S[-1] / S[0] - 1) * 100
            }
        
        # Calculate returns only for valid periods
        valid_periods = np.where(valid_idx[:-1] & valid_idx[1:])[0]
        if len(valid_periods) == 0:
            return {
                'tracking_difference': -100,
                'tracking_error': 100,
                'average_exposure': 0,
                'turnover': 0,
                'cost_drag': 100,
                'r_squared': 0,
                'total_return': -100,
                'index_return': (S[-1] / S[0] - 1) * 100
            }
        
        index_returns = np.diff(np.log(S))[valid_periods]
        fund_returns = np.diff(np.log(F[valid_idx]))[:(len(valid_periods))]
        
        # Tracking difference (annualized)
        diff_returns = fund_returns - self.L * index_returns
        tracking_diff = np.mean(diff_returns) * 252 / dt
        
        # Tracking error (annualized)
        tracking_error = np.std(diff_returns) * np.sqrt(252 / dt)
        
        # Average exposure
        avg_exposure = np.mean(pi[valid_idx][1:])
        
        # Turnover
        total_time = (times[valid_periods[-1]+1] - times[0])
        if total_time > 0 and np.mean(F[valid_idx]) > 0:
            turnover = np.sum(trades[:valid_periods[-1]+1]) / np.mean(F[valid_idx]) / total_time * 252
        else:
            turnover = 0
        
        # Total costs
        total_costs = np.sum(costs[:valid_periods[-1]+1])
        if F[0] > 0 and total_time > 0:
            cost_drag = total_costs / F[0] / total_time * 252
        else:
            cost_drag = 0
        
        # R-squared
        if len(fund_returns) > 1 and np.std(fund_returns) > 0 and np.std(index_returns) > 0:
            correlation = np.corrcoef(fund_returns, index_returns)[0, 1]
            r_squared = correlation ** 2
        else:
            r_squared = 0
        
        # Total returns
        last_valid_idx = np.where(valid_idx)[0][-1]
        total_return = (F[last_valid_idx] / F[0] - 1) * 100 if F[0] > 0 else -100
        index_return = (S[last_valid_idx] / S[0] - 1) * 100
        
        return {
            'tracking_difference': tracking_diff,
            'tracking_error': tracking_error,
            'average_exposure': avg_exposure,
            'turnover': turnover,
            'cost_drag': cost_drag,
            'r_squared': r_squared,
            'total_return': total_return,
            'index_return': index_return
        }
    
    def implied_spread(self, tracking_diff, tracking_error, sigma):
        """
        Calculate implied spread from equation (1)
        """
        if tracking_error == 0 or self.L == 0 or self.L == 1:
            return np.nan
        
        return 12 / np.sqrt(3) * (-tracking_diff) * tracking_error / (
            sigma**3 * self.L**2 * (1 - self.L)**2
        )


class LeveragedETFBacktest:
    """
    Backtesting framework for leveraged ETF strategies
    """
    
    def __init__(self, leverage_factors, spreads, gammas, sigma=0.16):
        self.leverage_factors = leverage_factors
        self.spreads = spreads
        self.gammas = gammas
        self.sigma = sigma
        
    def run_simulation(self, T=5, dt=1/252, n_paths=100):
        """
        Run Monte Carlo simulation
        """
        results = []
        
        for L in self.leverage_factors:
            for epsilon in self.spreads:
                for gamma in self.gammas:
                    print(f"Testing L={L}, ε={epsilon:.1%}, γ={gamma}")
                    
                    path_results = []
                    
                    for path in range(n_paths):
                        try:
                            # Create replicator
                            replicator = LeveragedETFReplicator(L, epsilon, gamma)
                            
                            # Simulate index
                            times, S, _ = replicator.simulate_index(T, dt, self.sigma)
                            
                            # Replicate
                            rep_result = replicator.replicate(times, S)
                            
                            # Calculate metrics
                            metrics = replicator.calculate_performance_metrics(
                                times, S, rep_result['F'], rep_result['pi'],
                                rep_result['trades'], rep_result['costs']
                            )
                            
                            metrics['leverage_factor'] = L
                            metrics['spread'] = epsilon
                            metrics['gamma'] = gamma
                            metrics['pi_minus'] = replicator.pi_minus
                            metrics['pi_plus'] = replicator.pi_plus
                            
                            path_results.append(metrics)
                            
                        except Exception as e:
                            print(f"  Error in path {path}: {e}")
                            continue
                    
                    if path_results:
                        # Average across paths
                        avg_metrics = pd.DataFrame(path_results).mean()
                        results.append(avg_metrics.to_dict())
        
        return pd.DataFrame(results)
    
    def plot_tracking_tradeoff(self, results_df):
        """
        Plot tracking error vs tracking difference tradeoff
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Filter out invalid results
        results_df = results_df[
            (results_df['tracking_error'] < 50) & 
            (results_df['tracking_difference'] > -50)
        ]
        
        # Group by leverage factor
        for L in sorted(results_df['leverage_factor'].unique()):
            df_L = results_df[results_df['leverage_factor'] == L]
            
            # Left plot: varying gamma
            ax = axes[0]
            for epsilon in sorted(df_L['spread'].unique()):
                df_eps = df_L[df_L['spread'] == epsilon]
                df_eps = df_eps.sort_values('tracking_error')
                
                ax.plot(df_eps['tracking_error'] * 100, 
                       -df_eps['tracking_difference'] * 100,
                       marker='o', 
                       label=f'L={L}, ε={epsilon:.1%}')
            
            # Right plot: theoretical curve
            ax = axes[1]
            te_range = np.linspace(0.1, 5, 50) / 100
            td_theory = []
            
            for te in te_range:
                # From equation (21)
                for epsilon in self.spreads:
                    td = -np.sqrt(3) / 12 * self.sigma**3 * L**2 * (1 - L)**2 * epsilon / te
                    td_theory.append(td)
                    break  # Just use first epsilon for theory
            
            ax.plot(te_range * 100, np.array(td_theory) * 100, 
                   '--', label=f'L={L} (theory)')
        
        axes[0].set_xlabel('Tracking Error (%)')
        axes[0].set_ylabel('Negative Tracking Difference (%)')
        axes[0].set_title('Empirical Tracking Trade-off')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0, 10)
        axes[0].set_ylim(0, 10)
        
        axes[1].set_xlabel('Tracking Error (%)')
        axes[1].set_ylabel('Negative Tracking Difference (%)')
        axes[1].set_title('Theoretical Tracking Trade-off')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(0, 5)
        axes[1].set_ylim(0, 5)
        
        plt.suptitle('Tracking Error vs Tracking Difference Trade-off', fontsize=14)
        plt.tight_layout()
        plt.show()
    
    def plot_underexposure(self, results_df):
        """
        Plot underexposure effect
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter for specific parameters
        df_filtered = results_df[
            (results_df['spread'] == 0.001) & 
            (results_df['gamma'] == 10)
        ]
        
        if len(df_filtered) == 0:
            print("No data available for underexposure plot")
            return
        
        leverage_factors = df_filtered['leverage_factor'].values
        avg_exposures = df_filtered['average_exposure'].values
        
        # Theoretical underexposure from equation (22)
        epsilon = 0.001
        gamma = 10
        
        theoretical_exposures = []
        for L in leverage_factors:
            # Handle the calculation carefully
            inner = gamma * (L - 1) / 6
            if inner >= 0:
                factor = inner**(1/3)
            else:
                factor = -(-inner)**(1/3)
            
            beta_bar = L - (2 * L - 1) / gamma * factor * epsilon**(2/3)
            theoretical_exposures.append(beta_bar)
        
        # Plot
        width = 0.35
        x = np.arange(len(leverage_factors))
        
        bars1 = ax.bar(x - width/2, leverage_factors, width, 
                       label='Target', alpha=0.7, color='blue')
        bars2 = ax.bar(x + width/2, avg_exposures, width, 
                       label='Realized', alpha=0.7, color='orange')
        
        ax.plot(x, theoretical_exposures, 'ro--', 
               label='Theoretical', markersize=8)
        
        ax.set_xlabel('Leverage Factor')
        ax.set_ylabel('Average Exposure')
        ax.set_title('Underexposure Effect in Leveraged ETFs')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{L:.0f}' for L in leverage_factors])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_boundary_behavior(self, L=2, epsilon=0.001):
        """
        Plot how boundaries change with tracking error aversion
        """
        gammas = np.logspace(-1, 3, 50)
        pi_minus_vals = []
        pi_plus_vals = []
        
        for gamma in gammas:
            try:
                replicator = LeveragedETFReplicator(L, epsilon, gamma)
                pi_minus_vals.append(replicator.pi_minus)
                pi_plus_vals.append(replicator.pi_plus)
            except:
                pi_minus_vals.append(np.nan)
                pi_plus_vals.append(np.nan)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Remove NaN values
        valid_idx = ~(np.isnan(pi_minus_vals) | np.isnan(pi_plus_vals))
        gammas_valid = gammas[valid_idx]
        pi_minus_valid = np.array(pi_minus_vals)[valid_idx]
        pi_plus_valid = np.array(pi_plus_vals)[valid_idx]
        
        ax.plot(gammas_valid, pi_plus_valid, 'b-', label='π+ (sell boundary)', linewidth=2)
        ax.plot(gammas_valid, pi_minus_valid, 'r-', label='π- (buy boundary)', linewidth=2)
        ax.axhline(y=L, color='black', linestyle='--', label=f'Target L={L}')
        
        ax.set_xscale('log')
        ax.set_xlabel('Tracking Error Aversion (γ)')
        ax.set_ylabel('Trading Boundaries')
        ax.set_title(f'Optimal Trading Boundaries vs Tracking Error Aversion (L={L}, ε={epsilon:.1%})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# Example: Single path simulation
def example_single_path():
    """
    Example of single path replication
    """
    # Parameters
    L = 2  # 2x leveraged fund
    epsilon = 0.001  # 10 bps spread
    gamma = 10  # Tracking error aversion
    sigma = 0.16  # 16% annual volatility
    T = 1  # 1 year
    dt = 1/252  # Daily rebalancing
    
    # Create replicator
    replicator = LeveragedETFReplicator(L, epsilon, gamma)
    print(f"Optimal boundaries: π- = {replicator.pi_minus:.4f}, π+ = {replicator.pi_plus:.4f}")
    
    # Simulate index
    times, S, _ = replicator.simulate_index(T, dt, sigma)
    
    # Replicate
    result = replicator.replicate(times, S)
    
    # Calculate metrics
    metrics = replicator.calculate_performance_metrics(
        times, S, result['F'], result['pi'], 
        result['trades'], result['costs']
    )
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Fund vs leveraged index
    ax = axes[0, 0]
    ax.plot(times, S / S[0], label='Index', linewidth=2)
    ax.plot(times, result['F'] / result['F'][0], label='Fund', linewidth=2)
    
    # Calculate theoretical leveraged index (continuous rebalancing, no costs)
    leveraged_index = np.zeros_like(S)
    leveraged_index[0] = 1
    for i in range(1, len(S)):
        leveraged_index[i] = leveraged_index[i-1] * (1 + L * (S[i]/S[i-1] - 1))
    
    ax.plot(times, leveraged_index, '--', label=f'{L}x Index (no costs)', linewidth=2)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Normalized Value')
    ax.set_title('Fund Performance vs Leveraged Index')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Portfolio weight
    ax = axes[0, 1]
    ax.plot(times, result['pi'], linewidth=2)
    ax.axhline(y=replicator.pi_plus, color='red', linestyle='--', 
               label=f'π+ = {replicator.pi_plus:.3f}')
    ax.axhline(y=replicator.pi_minus, color='green', linestyle='--', 
               label=f'π- = {replicator.pi_minus:.3f}')
    ax.axhline(y=L, color='black', linestyle=':', label=f'Target = {L}')
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Portfolio Weight')
    ax.set_title('Index Exposure Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Cumulative costs
    ax = axes[1, 0]
    cumulative_costs = np.cumsum(result['costs'])
    ax.plot(times[:-1], cumulative_costs / result['F'][0] * 100, linewidth=2)
    ax.set_xlabel('Time (years)')
    ax.set_ylabel('Cumulative Costs (% of initial value)')
    ax.set_title('Transaction Costs')
    ax.grid(True, alpha=0.3)
    
    # Performance summary
    ax = axes[1, 1]
    ax.axis('off')
    summary_text = f"""
    Performance Summary:
    
    Tracking Difference: {metrics['tracking_difference']:.2%}
    Tracking Error: {metrics['tracking_error']:.2%}
    Average Exposure: {metrics['average_exposure']:.3f}
    Turnover: {metrics['turnover']:.1%}
    Cost Drag: {metrics['cost_drag']:.2%}
    R-squared: {metrics['r_squared']:.3f}
    
    Fund Return: {metrics['total_return']:.2f}%
    Index Return: {metrics['index_return']:.2f}%
    {L}x Index Return: {(leveraged_index[-1] - 1)*100:.2f}%
    """
    ax.text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
            fontfamily='monospace')
    
    plt.suptitle(f'{L}x Leveraged ETF Replication (γ={gamma}, ε={epsilon:.1%})', 
                fontsize=14)
    plt.tight_layout()
    plt.show()
    
    return metrics


# Main execution
if __name__ == "__main__":
    print("="*70)
    print("LEVERAGED ETF REPLICATION STRATEGY")
    print("Based on Guasoni & Mayerhofer (2023)")
    print("="*70)
    
    # Example 1: Single path demonstration
    print("\nExample 1: Single Path Replication")
    metrics = example_single_path()
    
    # Example 2: Monte Carlo backtesting
    print("\nExample 2: Monte Carlo Backtesting")
    
    # Parameters for backtesting
    leverage_factors = [-3, -2, -1, 2, 3]
    spreads = [0.0005, 0.001, 0.002]  # 5, 10, 20 bps
    gammas = [1, 5, 10, 50]  # Different tracking error aversions
    
    # Run backtest
    backtest = LeveragedETFBacktest(leverage_factors, spreads, gammas)
    results_df = backtest.run_simulation(T=1, dt=1/252, n_paths=50)
    
    # Display results summary
    print("\nBacktest Results Summary:")
    summary_cols = ['leverage_factor', 'spread', 'gamma', 
                   'tracking_difference', 'tracking_error', 'average_exposure']
    print(results_df[summary_cols].round(4))
    
    # Plot results
    backtest.plot_tracking_tradeoff(results_df)
    backtest.plot_underexposure(results_df)
    backtest.plot_boundary_behavior(L=3, epsilon=0.001)
    
    # Example 3: Implied spread analysis
    print("\nExample 3: Implied Spread Analysis")
    
    # Calculate implied spreads for different funds
    implied_spreads = []
    
    for _, row in results_df.iterrows():
        replicator = LeveragedETFReplicator(
            row['leverage_factor'], 
            row['spread'], 
            row['gamma']
        )
        
        implied = replicator.implied_spread(
            row['tracking_difference'],
            row['tracking_error'],
            0.16  # sigma
        )
        
        if not np.isnan(implied):
            implied_spreads.append({
                'leverage_factor': row['leverage_factor'],
                'actual_spread_bps': row['spread'] * 10000,
                'implied_spread_bps': implied * 10000,
                'gamma': row['gamma']
            })
    
    if implied_spreads:
        implied_df = pd.DataFrame(implied_spreads)
        
        # Plot implied vs actual spreads
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for L in sorted(implied_df['leverage_factor'].unique()):
            df_L = implied_df[implied_df['leverage_factor'] == L]
            ax.scatter(df_L['actual_spread_bps'], df_L['implied_spread_bps'], 
                      label=f'L={L}', s=100, alpha=0.7)
        
        # Add 45-degree line
        max_spread = max(implied_df['actual_spread_bps'].max(), 
                        implied_df['implied_spread_bps'].max())
        ax.plot([0, max_spread], [0, max_spread], 'k--', alpha=0.5)
        
        ax.set_xlabel('Actual Spread (bps)')
        ax.set_ylabel('Implied Spread (bps)')
        ax.set_title('Implied vs Actual Spreads')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()