import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skewnorm, t, norm, jarque_bera
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class EnhancedFourMomentPortfolio:
    """
    Enhanced implementation with several performance improvements
    """
    
    def __init__(self, returns_data, y_data, z_data, lookback_window=252):
        self.returns = returns_data
        self.n_assets = returns_data.shape[1]
        self.asset_names = returns_data.columns
        self.lookback_window = lookback_window
        
        # Estimate parameters with robust methods
        self._estimate_parameters_robust(y_data, z_data)
        
        # Calculate rolling statistics for dynamic adjustment
        self._calculate_rolling_stats()
        
    def _estimate_parameters_robust(self, y, z):
        """Enhanced parameter estimation with robustness checks"""
        T = len(self.returns)
        
        # Use exponentially weighted estimation for more recent data importance
        decay_factor = 0.94
        weights = decay_factor ** np.arange(T-1, -1, -1)
        weights = weights / weights.sum() * T
        
        # Robust mean estimation (trimmed mean)
        trim_pct = 0.05
        self.mu = self.returns.apply(lambda x: self._trimmed_mean(x, trim_pct)).values
        
        # Robust OLS with outlier detection
        X = np.column_stack([np.ones(T), y, z])
        
        self.b = np.zeros(self.n_assets)
        self.t = np.zeros(self.n_assets)
        residuals = np.zeros((T, self.n_assets))
        
        for i in range(self.n_assets):
            # Weighted least squares
            W = np.diag(weights)
            coeffs = np.linalg.inv(X.T @ W @ X) @ X.T @ W @ self.returns.iloc[:, i].values
            
            self.b[i] = coeffs[1]
            self.t[i] = coeffs[2]
            
            # Calculate residuals
            fitted = X @ coeffs
            residuals[:, i] = self.returns.iloc[:, i].values - fitted
        
        # Robust covariance estimation (shrinkage)
        self.C = self._shrinkage_covariance(residuals)
        
        # Calculate D with regularization
        self.D = self.C + np.outer(self.b, self.b) + np.outer(self.t, self.t)
        
        # Regularization for numerical stability
        eigenvalues = np.linalg.eigvals(self.D)
        if np.min(eigenvalues) < 1e-6:
            self.D += np.eye(self.n_assets) * max(1e-6, abs(np.min(eigenvalues)) * 1.1)
    
    def _trimmed_mean(self, x, trim_pct):
        """Calculate trimmed mean to reduce outlier impact"""
        lower = x.quantile(trim_pct)
        upper = x.quantile(1 - trim_pct)
        return x[(x >= lower) & (x <= upper)].mean()
    
    def _shrinkage_covariance(self, residuals, shrinkage_target='diagonal'):
        """Ledoit-Wolf shrinkage estimator for covariance"""
        n, p = residuals.shape
        
        # Sample covariance
        S = np.cov(residuals.T)
        
        # Shrinkage target (diagonal matrix)
        if shrinkage_target == 'diagonal':
            F = np.diag(np.diag(S))
        else:
            F = np.eye(p) * np.trace(S) / p
        
        # Optimal shrinkage intensity
        c = np.linalg.norm(S - F, 'fro')**2 / n
        gamma = 0
        for i in range(n):
            gamma += np.linalg.norm(np.outer(residuals[i], residuals[i]) - S, 'fro')**2
        gamma = gamma / n
        
        kappa = gamma / c if c > 0 else 0
        shrinkage_intensity = max(0, min(1, kappa))
        
        # Shrunk covariance
        return shrinkage_intensity * F + (1 - shrinkage_intensity) * S
    
    def _calculate_rolling_stats(self):
        """Calculate rolling statistics for regime detection"""
        self.rolling_vol = self.returns.rolling(window=21).std()
        self.rolling_corr = self.returns.rolling(window=63).corr().mean()
        self.rolling_skew = self.returns.rolling(window=self.lookback_window).skew()
        self.rolling_kurt = self.returns.rolling(window=self.lookback_window).apply(lambda x: x.kurtosis() + 3)
    
    def detect_market_regime(self, current_idx):
        """Detect current market regime for adaptive allocation"""
        if current_idx < self.lookback_window:
            return 'normal'
        
        # Recent volatility vs historical
        recent_vol = self.rolling_vol.iloc[current_idx].mean()
        hist_vol = self.rolling_vol.iloc[:current_idx].mean().mean()
        
        # Recent correlation
        recent_corr = self.returns.iloc[current_idx-63:current_idx].corr().values
        avg_corr = np.mean(recent_corr[np.triu_indices_from(recent_corr, k=1)])
        
        # Regime classification
        if recent_vol > 1.5 * hist_vol and avg_corr > 0.6:
            return 'crisis'
        elif recent_vol > 1.2 * hist_vol:
            return 'volatile'
        elif recent_vol < 0.8 * hist_vol:
            return 'calm'
        else:
            return 'normal'
    
    def _calculate_matrices(self):
        """Calculate auxiliary matrices P, A, P2, etc."""
        ones = np.ones(self.n_assets)
        M = np.column_stack([self.mu, ones, self.b, self.t])
        
        D_inv = np.linalg.inv(self.D)
        P = M.T @ D_inv @ M
        
        self.A = P[:2, :2]
        self.P2 = P[:3, :3]
        self.P = P
        
        self.psi = P[:3, 3]
        self.s = P[3, 3]
        
        P2_inv = np.linalg.inv(self.P2)
        self.H = self.psi.T @ P2_inv @ self.psi
        
        self.D_inv = D_inv
        self.A_inv = np.linalg.inv(self.A)
        self.P2_inv = P2_inv
    
    def mean_variance_portfolio(self, target_return, long_only=False):
        """Calculate mean-variance optimal portfolio"""
        self._calculate_matrices()
        
        # x_mv = D^(-1) * [mu, 1] * A^(-1) * [target_return, 1]'
        M_mv = np.column_stack([self.mu, np.ones(self.n_assets)])
        target_vec = np.array([target_return, 1])
        
        x_mv = self.D_inv @ M_mv @ self.A_inv @ target_vec
        
        if long_only:
            x_mv = np.maximum(x_mv, 0)
            x_mv = x_mv / np.sum(x_mv)
        
        return x_mv
    
    def portfolio_moments(self, weights):
        """Calculate four moments of a portfolio"""
        mu_p = weights.T @ self.mu
        var_p = weights.T @ self.D @ weights
        skew_p = weights.T @ self.b
        kurt_p = weights.T @ self.t
        
        return mu_p, var_p, skew_p, kurt_p
    
    def four_moment_portfolio(self, target_return, target_variance, target_skewness, 
                            xi_y, k_z, long_only=False):
        """Base four-moment portfolio optimization"""
        self._calculate_matrices()
        
        # Mean-variance portfolio
        M_mv = np.column_stack([self.mu, np.ones(self.n_assets)])
        x_mv = self.D_inv @ M_mv @ self.A_inv @ np.array([target_return, 1])
        
        # Calculate variance levels
        var_A = x_mv.T @ self.D @ x_mv
        beta = np.array([target_return, 1, target_skewness/xi_y])
        var_P2 = beta.T @ self.P2_inv @ beta
        
        if target_variance < var_P2:
            target_variance = var_P2 * 1.1
        
        # Skewness component
        f, g = self.P2[0, 2], self.P2[1, 2]
        e = self.P2[2, 2]
        h = np.array([f, g]).T @ self.A_inv @ np.array([f, g])
        
        if var_P2 > var_A and e > h:
            x_sk_factor = np.sqrt((var_P2 - var_A) / (e - h))
            x_sk = (self.D_inv @ self.b - self.D_inv @ M_mv @ self.A_inv @ np.array([f, g])) * x_sk_factor
        else:
            x_sk = np.zeros(self.n_assets)
        
        # Kurtosis component
        if target_variance > var_P2 and self.s > self.H:
            x_k_factor = np.sqrt((target_variance - var_P2) / (self.s - self.H))
            pq = np.array([self.P[0, 3], self.P[1, 3]])
            if k_z < 0:
                x_k = -(self.D_inv @ self.t - self.D_inv @ M_mv @ self.A_inv @ pq) * x_k_factor
            else:
                x_k = (self.D_inv @ self.t - self.D_inv @ M_mv @ self.A_inv @ pq) * x_k_factor
        else:
            x_k = np.zeros(self.n_assets)
        
        x_optimal = x_mv + x_sk + x_k
        
        if long_only:
            x_optimal = np.maximum(x_optimal, 0)
            if np.sum(x_optimal) > 0:
                x_optimal = x_optimal / np.sum(x_optimal)
        
        return x_optimal, x_mv, x_sk, x_k
    
    def adaptive_four_moment_portfolio(self, target_return, base_target_variance, 
                                     target_skewness, xi_y, k_z, current_idx=None):
        """
        Adaptive portfolio that adjusts based on market regime
        """
        # Detect regime if index provided
        if current_idx is not None:
            regime = self.detect_market_regime(current_idx)
        else:
            regime = 'normal'
        
        # Adjust targets based on regime
        regime_adjustments = {
            'crisis': {'var_mult': 0.7, 'skew_mult': 1.5, 'kurt_importance': 1.5},
            'volatile': {'var_mult': 0.85, 'skew_mult': 1.2, 'kurt_importance': 1.3},
            'calm': {'var_mult': 1.2, 'skew_mult': 0.8, 'kurt_importance': 0.8},
            'normal': {'var_mult': 1.0, 'skew_mult': 1.0, 'kurt_importance': 1.0}
        }
        
        adj = regime_adjustments[regime]
        adjusted_variance = base_target_variance * adj['var_mult']
        adjusted_skewness = target_skewness * adj['skew_mult']
        
        # Get base portfolio
        x_opt, x_mv, x_sk, x_k = self.four_moment_portfolio(
            target_return, adjusted_variance, adjusted_skewness, xi_y, k_z
        )
        
        # Apply regime-based adjustments to components
        x_k = x_k * adj['kurt_importance']
        
        # Renormalize
        x_adjusted = x_mv + x_sk + x_k
        
        return x_adjusted, regime
    
    def four_moment_portfolio_with_constraints(self, target_return, target_variance, 
                                             target_skewness, xi_y, k_z, 
                                             max_weight=0.3, min_weight=-0.1,
                                             sector_constraints=None):
        """
        Four-moment portfolio with practical constraints
        """
        self._calculate_matrices()
        
        # Get unconstrained solution
        x_unconstrained, x_mv, x_sk, x_k = self.four_moment_portfolio(
            target_return, target_variance, target_skewness, xi_y, k_z
        )
        
        # Define optimization problem with constraints
        def objective(x):
            # Minimize negative skewness and positive kurtosis
            return -x @ self.b / xi_y + x @ self.t / abs(k_z)
        
        def variance_constraint(x):
            return target_variance - x @ self.D @ x
        
        def return_constraint(x):
            return x @ self.mu - target_return
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Budget
            {'type': 'eq', 'fun': return_constraint},        # Return
            {'type': 'eq', 'fun': variance_constraint}       # Variance
        ]
        
        # Bounds
        bounds = [(min_weight, max_weight) for _ in range(self.n_assets)]
        
        # Sector constraints if provided
        if sector_constraints is not None:
            for sector, (min_exp, max_exp) in sector_constraints.items():
                sector_assets = [i for i, name in enumerate(self.asset_names) 
                               if sector in name]
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=sector_assets: np.sum(x[idx]) - min_exp
                })
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda x, idx=sector_assets: max_exp - np.sum(x[idx])
                })
        
        # Optimize
        result = minimize(objective, x_unconstrained, method='SLSQP',
                        bounds=bounds, constraints=constraints,
                        options={'ftol': 1e-9, 'maxiter': 1000})
        
        if result.success:
            return result.x
        else:
            # Fall back to unconstrained with clipping
            x_clipped = np.clip(x_unconstrained, min_weight, max_weight)
            x_clipped = x_clipped / np.sum(x_clipped)
            return x_clipped
    
    def robust_portfolio_optimization(self, target_return, n_scenarios=100):
        """
        Robust optimization considering parameter uncertainty
        """
        # Generate scenarios for uncertain parameters
        scenarios_mu = []
        scenarios_D = []
        
        for _ in range(n_scenarios):
            # Perturb mean returns
            mu_noise = np.random.normal(0, self.mu.std() * 0.2, self.n_assets)
            scenarios_mu.append(self.mu + mu_noise)
            
            # Perturb covariance (maintain positive definiteness)
            D_perturb = self.D.copy()
            noise = np.random.normal(0, 0.05, self.D.shape)
            noise = (noise + noise.T) / 2  # Make symmetric
            D_perturb += noise * np.sqrt(np.diag(self.D).reshape(-1, 1) @ np.diag(self.D).reshape(1, -1))
            
            # Ensure positive definiteness
            eigenvalues, eigenvectors = np.linalg.eig(D_perturb)
            eigenvalues = np.real(eigenvalues)  # Take real part
            eigenvalues = np.maximum(eigenvalues, 1e-6)
            D_perturb = np.real(eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T)
            scenarios_D.append(D_perturb)
        
        # Solve for worst-case scenario
        def worst_case_objective(x):
            worst_sharpe = np.inf
            
            for mu_s, D_s in zip(scenarios_mu, scenarios_D):
                ret = x @ mu_s
                vol = np.sqrt(x @ D_s @ x)
                sharpe = ret / vol if vol > 0 else 0
                worst_sharpe = min(worst_sharpe, sharpe)
            
            return -worst_sharpe  # Minimize negative worst-case Sharpe
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'ineq', 'fun': lambda x: x @ self.mu - target_return}
        ]
        
        # Optimize
        bounds = [(-0.2, 0.5) for _ in range(self.n_assets)]
        x0 = np.ones(self.n_assets) / self.n_assets
        
        result = minimize(worst_case_objective, x0, method='SLSQP',
                        bounds=bounds, constraints=constraints,
                        options={'ftol': 1e-9, 'maxiter': 500})
        
        if result.success:
            return result.x
        else:
            # Fall back to equal weight
            return x0

# Generate simulated data
def generate_enhanced_simulated_data(n_assets=10, n_periods=2000):
    """Generate more realistic simulated data with regime changes"""
    np.random.seed(42)
    
    # Define market regimes
    regime_params = {
        'normal': {'vol_mult': 1.0, 'corr': 0.3, 'skew': 0},
        'volatile': {'vol_mult': 1.5, 'corr': 0.5, 'skew': -0.5},
        'crisis': {'vol_mult': 2.0, 'corr': 0.7, 'skew': -1.0},
        'calm': {'vol_mult': 0.7, 'corr': 0.2, 'skew': 0.2}
    }
    
    # Generate regime sequence
    regime_sequence = []
    current_regime = 'normal'
    regime_probs = {
        'normal': {'normal': 0.8, 'volatile': 0.15, 'crisis': 0.03, 'calm': 0.02},
        'volatile': {'normal': 0.4, 'volatile': 0.5, 'crisis': 0.08, 'calm': 0.02},
        'crisis': {'normal': 0.2, 'volatile': 0.4, 'crisis': 0.4, 'calm': 0.0},
        'calm': {'normal': 0.3, 'volatile': 0.1, 'crisis': 0.0, 'calm': 0.6}
    }
    
    for _ in range(n_periods):
        probs = regime_probs[current_regime]
        current_regime = np.random.choice(list(probs.keys()), p=list(probs.values()))
        regime_sequence.append(current_regime)
    
    # Generate returns based on regimes
    returns = []
    y_values = []
    z_values = []
    
    # Base parameters
    base_mu = np.random.uniform(0.0001, 0.001, n_assets)
    base_vol = np.random.uniform(0.01, 0.03, n_assets)
    b = np.random.uniform(-0.001, 0.001, n_assets)
    t_vec = np.random.uniform(0.0001, 0.0003, n_assets)
    
    for regime in regime_sequence:
        params = regime_params[regime]
        
        # Adjust parameters based on regime
        mu = base_mu * (1 + 0.2 * np.random.randn())
        vol = base_vol * params['vol_mult']
        
        # Generate correlated returns
        corr_matrix = np.full((n_assets, n_assets), params['corr'])
        np.fill_diagonal(corr_matrix, 1.0)
        cov_matrix = np.outer(vol, vol) * corr_matrix
        
        # Generate y and z with regime-dependent properties
        y = skewnorm.rvs(a=params['skew']*5, size=1)[0]
        z = t.rvs(df=5 if regime in ['volatile', 'crisis'] else 10, size=1)[0]
        
        y_values.append(y)
        z_values.append(z)
        
        # Generate returns
        epsilon = np.random.multivariate_normal(np.zeros(n_assets), cov_matrix)
        r = mu + epsilon + b * y + t_vec * z
        returns.append(r)
    
    # Standardize y and z
    y_values = np.array(y_values)
    z_values = np.array(z_values)
    y_values = (y_values - np.mean(y_values)) / np.std(y_values)
    z_values = (z_values - np.mean(z_values)) / np.std(z_values)
    
    # Create DataFrame
    returns_df = pd.DataFrame(returns, columns=[f'Asset_{i+1}' for i in range(n_assets)])
    
    return returns_df, y_values, z_values

def calculate_comprehensive_metrics(returns):
    """Calculate comprehensive performance metrics"""
    
    # Basic metrics
    annual_return = returns.mean() * 252
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
    
    # Downside metrics
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
    sortino = annual_return / downside_vol if downside_vol > 0 else 0
    
    # Higher moments
    skewness = returns.skew()
    kurtosis = returns.kurtosis() + 3
    
    # Risk metrics
    var_95 = returns.quantile(0.05)
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
    
    # Drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Tail ratio
    right_tail = returns[returns > returns.mean() + returns.std()]
    left_tail = returns[returns < returns.mean() - returns.std()]
    tail_ratio = abs(right_tail.mean() / left_tail.mean()) if len(left_tail) > 0 and left_tail.mean() != 0 else 1
    
    # Calmar ratio
    calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    return {
        'Annual Return': annual_return,
        'Annual Volatility': annual_vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'VaR 95%': var_95,
        'CVaR 95%': cvar_95,
        'Max Drawdown': max_drawdown,
        'Tail Ratio': tail_ratio,
        'Calmar Ratio': calmar
    }

def display_enhanced_results(results, regimes, optimizer):
    """Display enhanced results with improvements analysis"""
    
    print("\n" + "="*80)
    print("ENHANCED STRATEGY PERFORMANCE COMPARISON")
    print("="*80)
    
    # Create comparison DataFrame
    metrics_data = {}
    for strategy, data in results.items():
        metrics_data[strategy] = data['metrics']
    
    metrics_df = pd.DataFrame(metrics_data).T
    print("\nPerformance Metrics:")
    print(metrics_df.round(4))
    
    # Performance improvement analysis
    print("\n" + "="*50)
    print("PERFORMANCE IMPROVEMENTS")
    print("="*50)
    
    base_sharpe = metrics_df.loc['Mean-Variance', 'Sharpe Ratio']
    
    for strategy in metrics_df.index:
        if strategy != 'Mean-Variance':
            sharpe_imp = (metrics_df.loc[strategy, 'Sharpe Ratio'] - base_sharpe) / base_sharpe * 100
            skew_imp = metrics_df.loc[strategy, 'Skewness'] - metrics_df.loc['Mean-Variance', 'Skewness']
            kurt_imp = metrics_df.loc['Mean-Variance', 'Kurtosis'] - metrics_df.loc[strategy, 'Kurtosis']
            dd_imp = (metrics_df.loc['Mean-Variance', 'Max Drawdown'] - metrics_df.loc[strategy, 'Max Drawdown']) / abs(metrics_df.loc['Mean-Variance', 'Max Drawdown']) * 100
            
            print(f"\n{strategy} vs Mean-Variance:")
            print(f"  Sharpe Ratio: {sharpe_imp:+.1f}%")
            print(f"  Skewness: {skew_imp:+.3f}")
            print(f"  Kurtosis reduction: {kurt_imp:+.3f}")
            print(f"  Max Drawdown: {dd_imp:+.1f}%")
    
    # Regime analysis for adaptive strategy
    if regimes:
        print("\n" + "="*50)
        print("ADAPTIVE STRATEGY REGIME ANALYSIS")
        print("="*50)
        
        regime_counts = pd.Series(regimes).value_counts()
        print("\nRegime Distribution:")
        for regime, count in regime_counts.items():
            print(f"  {regime}: {count} periods ({count/len(regimes)*100:.1f}%)")

def create_enhanced_visualizations(results, returns_df, regimes):
    """Create enhanced visualizations for strategy comparison"""
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    # 1. Cumulative returns with regime shading
    ax = axes[0, 0]
    
    for strategy, data in results.items():
        if strategy != 'Adaptive':  # Handle adaptive separately
            cum_returns = (1 + data['returns']).cumprod()
            ax.plot(cum_returns.index, cum_returns.values, label=strategy, linewidth=2)
    
    # Add regime shading
    if regimes:
        regime_colors = {
            'normal': 'green', 'volatile': 'yellow', 
            'crisis': 'red', 'calm': 'blue'
        }
        
        current_regime = regimes[0]
        start_idx = 252
        
        for i, regime in enumerate(regimes[1:], 1):
            if regime != current_regime:
                color = regime_colors.get(current_regime, 'gray')
                ax.axvspan(start_idx + i - 1, start_idx + i, 
                          alpha=0.2, color=color)
                current_regime = regime
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Cumulative Return')
    ax.set_title('Cumulative Performance with Market Regimes')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Risk-Return Scatter with Skewness
    ax = axes[0, 1]
    
    returns = []
    vols = []
    skews = []
    names = []
    
    for strategy, data in results.items():
        metrics = data['metrics']
        returns.append(metrics['Annual Return'])
        vols.append(metrics['Annual Volatility'])
        skews.append(metrics['Skewness'])
        names.append(strategy)
    
    scatter = ax.scatter(vols, returns, c=skews, s=100, cmap='RdYlBu', 
                        edgecolor='black', linewidth=2)
    
    for i, name in enumerate(names):
        ax.annotate(name, (vols[i], returns[i]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=9)
    
    plt.colorbar(scatter, ax=ax, label='Skewness')
    ax.set_xlabel('Annual Volatility')
    ax.set_ylabel('Annual Return')
    ax.set_title('Risk-Return Profile with Skewness')
    ax.grid(True, alpha=0.3)
    
    # 3. Drawdown comparison
    ax = axes[0, 2]
    
    for strategy, data in results.items():
        if strategy != 'Adaptive':
            cum_returns = (1 + data['returns']).cumprod()
            running_max = cum_returns.expanding().max()
            drawdown = (cum_returns - running_max) / running_max
            ax.plot(drawdown.index, drawdown.values * 100, label=strategy, alpha=0.8)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Drawdown (%)')
    ax.set_title('Drawdown Analysis')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Rolling Sharpe Ratio
    ax = axes[1, 0]
    window = 126  # 6 months
    
    for strategy, data in results.items():
        if strategy != 'Adaptive' and len(data['returns']) > window:
            rolling_sharpe = (data['returns'].rolling(window).mean() / 
                            data['returns'].rolling(window).std() * np.sqrt(252))
            ax.plot(rolling_sharpe.index, rolling_sharpe.values, 
                   label=strategy, alpha=0.8)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Rolling Sharpe Ratio')
    ax.set_title(f'{window}-Day Rolling Sharpe Ratio')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Return distribution comparison
    ax = axes[1, 1]
    
    for strategy, data in results.items():
        if strategy in ['Mean-Variance', '4M-Constrained', 'Adaptive']:
            ax.hist(data['returns'], bins=50, alpha=0.5, density=True, 
                   label=strategy)
    
    ax.set_xlabel('Daily Returns')
    ax.set_ylabel('Density')
    ax.set_title('Return Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Higher moments evolution
    ax = axes[1, 2]
    window = 252
    
    if '4M-Constrained' in results:
        strategy_returns = results['4M-Constrained']['returns']
        rolling_skew = strategy_returns.rolling(window).skew()
        rolling_kurt = strategy_returns.rolling(window).apply(lambda x: x.kurtosis() + 3)
        
        ax2 = ax.twinx()
        line1 = ax.plot(rolling_skew.index, rolling_skew.values, 'b-', 
                        label='Skewness', alpha=0.8)
        line2 = ax2.plot(rolling_kurt.index, rolling_kurt.values, 'r-', 
                         label='Kurtosis', alpha=0.8)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Skewness', color='b')
        ax2.set_ylabel('Kurtosis', color='r')
        ax.set_title('Rolling Higher Moments (4M-Constrained)')
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best')
        ax.grid(True, alpha=0.3)
    
    # 7. Weight stability analysis
    ax = axes[2, 0]
    
    # Compare weight concentration
    strategies_to_compare = ['Mean-Variance', '4M-Standard', '4M-Constrained', 'Robust']
    herfindahl_indices = []
    max_weights = []
    
    for strategy in strategies_to_compare:
        if strategy in results:
            weights = results[strategy]['weights']
            herfindahl = np.sum(weights**2)
            max_weight = np.max(np.abs(weights))
            herfindahl_indices.append(herfindahl)
            max_weights.append(max_weight)
    
    if herfindahl_indices:
        x = np.arange(len(herfindahl_indices))
        width = 0.35
        
        ax.bar(x - width/2, herfindahl_indices, width, label='Herfindahl Index', alpha=0.8)
        ax.bar(x + width/2, max_weights, width, label='Max |Weight|', alpha=0.8)
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Value')
        ax.set_title('Portfolio Concentration Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies_to_compare[:len(herfindahl_indices)], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 8. Tail risk comparison
    ax = axes[2, 1]
    
    var_95s = []
    cvar_95s = []
    tail_ratios = []
    
    for strategy in ['Mean-Variance', '4M-Standard', '4M-Constrained', 'Robust']:
        if strategy in results:
            metrics = results[strategy]['metrics']
            var_95s.append(abs(metrics['VaR 95%']) * 100)
            cvar_95s.append(abs(metrics['CVaR 95%']) * 100)
            tail_ratios.append(metrics['Tail Ratio'])
    
    if var_95s:
        x = np.arange(len(var_95s))
        width = 0.25
        
        ax.bar(x - width, var_95s, width, label='VaR 95%', alpha=0.8)
        ax.bar(x, cvar_95s, width, label='CVaR 95%', alpha=0.8)
        ax.bar(x + width, tail_ratios, width, label='Tail Ratio', alpha=0.8)
        
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Value (%)')
        ax.set_title('Tail Risk Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(['MV', '4M-Std', '4M-Const', 'Robust'][:len(var_95s)], rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 9. Performance attribution
    ax = axes[2, 2]
    
    # Compare Sharpe ratios
    sharpe_ratios = []
    sortino_ratios = []
    calmar_ratios = []
    
    for strategy in ['Mean-Variance', '4M-Standard', '4M-Constrained', 'Robust', 'Adaptive']:
        if strategy in results:
            metrics = results[strategy]['metrics']
            sharpe_ratios.append(metrics['Sharpe Ratio'])
            sortino_ratios.append(metrics['Sortino Ratio'])
            calmar_ratios.append(metrics['Calmar Ratio'])
    
    # Normalize to mean-variance = 1
    if sharpe_ratios and sharpe_ratios[0] != 0:
        sharpe_ratios = np.array(sharpe_ratios) / sharpe_ratios[0]
        sortino_ratios = np.array(sortino_ratios) / sortino_ratios[0]
        calmar_ratios = np.array(calmar_ratios) / calmar_ratios[0]
        
        strategies_labels = ['MV', '4M-Std', '4M-Const', 'Robust', 'Adaptive'][:len(sharpe_ratios)]
        x = np.arange(len(strategies_labels))
        
        width = 0.25
        ax.bar(x - width, sharpe_ratios, width, label='Sharpe', alpha=0.8)
        ax.bar(x, sortino_ratios, width, label='Sortino', alpha=0.8)
        ax.bar(x + width, calmar_ratios, width, label='Calmar', alpha=0.8)
        
        ax.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('Strategy')
        ax.set_ylabel('Relative Performance')
        ax.set_title('Risk-Adjusted Performance (MV = 1.0)')
        ax.set_xticks(x)
        ax.set_xticklabels(strategies_labels)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_enhanced_results(returns_df, strategies, adaptive_weights, regimes, optimizer):
    """Analyze and visualize enhanced strategy results"""
    
    # Calculate performance metrics for all strategies
    results = {}
    
    for name, weights in strategies.items():
        port_returns = returns_df @ weights
        
        # Calculate comprehensive metrics
        metrics = calculate_comprehensive_metrics(port_returns)
        results[name] = {
            'returns': port_returns,
            'metrics': metrics,
            'weights': weights
        }
    
    # Add adaptive strategy
    if adaptive_weights:
        adaptive_returns = []
        for i, w in enumerate(adaptive_weights):
            ret = returns_df.iloc[252+i] @ w
            adaptive_returns.append(ret)
        
        adaptive_returns = pd.Series(adaptive_returns)
        results['Adaptive'] = {
            'returns': adaptive_returns,
            'metrics': calculate_comprehensive_metrics(adaptive_returns),
            'weights': np.mean(adaptive_weights, axis=0)
        }
    
    # Display results
    display_enhanced_results(results, regimes, optimizer)
    
    # Create enhanced visualizations
    create_enhanced_visualizations(results, returns_df, regimes)
    
    # Tail risk analysis
    print("\n" + "="*50)
    print("TAIL RISK ANALYSIS")
    print("="*50)
    
    for name, data in results.items():
        returns = data['returns']
        print(f"\n{name}:")
        print(f"  Best day: {returns.max():.4f}")
        print(f"  Worst day: {returns.min():.4f}")
        print(f"  Days with >2σ moves: {np.sum(np.abs(returns - returns.mean()) > 2*returns.std())}")
        print(f"  Negative skew days: {np.sum(returns < returns.mean() - returns.std())}")

# Test enhanced strategies
def test_enhanced_strategies():
    """Test enhanced four-moment strategies"""
    
    # Generate simulated data
    print("Generating enhanced simulated data...")
    returns_df, y, z = generate_enhanced_simulated_data()
    
    # Initialize enhanced optimizer
    optimizer = EnhancedFourMomentPortfolio(returns_df, y, z)
    
    # Calculate instrumental variable moments
    xi_y = np.mean(y**3)
    k_z = np.mean(z**4)
    
    print(f"\nInstrumental variable moments:")
    print(f"Skewness of y: {xi_y:.4f}")
    print(f"Kurtosis of z: {k_z:.4f}")
    
    # Display asset statistics
    print("\nAsset Statistics:")
    stats_df = pd.DataFrame({
        'Mean': returns_df.mean(),
        'Std': returns_df.std(),
        'Skewness': returns_df.skew(),
        'Kurtosis': returns_df.kurtosis() + 3
    })
    print(stats_df)
    
    # Test different enhanced strategies
    strategies = {}
    target_return = 0.0005
    
    # 1. Basic mean-variance
    print("\nCalculating Mean-Variance portfolio...")
    w_mv = optimizer.mean_variance_portfolio(target_return)
    strategies['Mean-Variance'] = w_mv
    mu_mv, var_mv, skew_mv, kurt_mv = optimizer.portfolio_moments(w_mv)
    print(f"  Expected return: {mu_mv:.6f}")
    print(f"  Variance: {var_mv:.6f}")
    print(f"  Skewness contribution: {skew_mv:.6f}")
    print(f"  Kurtosis contribution: {kurt_mv:.6f}")
    
    # 2. Standard four-moment
    print("\nCalculating Standard Four-Moment portfolio...")
    w_4m, _, _, _ = optimizer.four_moment_portfolio(
        target_return, var_mv * 1.2, 0.0001, xi_y, k_z
    )
    strategies['4M-Standard'] = w_4m
    mu_4m, var_4m, skew_4m, kurt_4m = optimizer.portfolio_moments(w_4m)
    print(f"  Expected return: {mu_4m:.6f}")
    print(f"  Variance: {var_4m:.6f}")
    print(f"  Skewness contribution: {skew_4m:.6f}")
    print(f"  Kurtosis contribution: {kurt_4m:.6f}")
    
    # 3. Constrained four-moment
    print("\nCalculating Constrained Four-Moment portfolio...")
    w_4m_const = optimizer.four_moment_portfolio_with_constraints(
        target_return, var_mv * 1.2, 0.0001, xi_y, k_z,
        max_weight=0.25, min_weight=-0.05
    )
    strategies['4M-Constrained'] = w_4m_const
    mu_4mc, var_4mc, skew_4mc, kurt_4mc = optimizer.portfolio_moments(w_4m_const)
    print(f"  Expected return: {mu_4mc:.6f}")
    print(f"  Variance: {var_4mc:.6f}")
    print(f"  Skewness contribution: {skew_4mc:.6f}")
    print(f"  Kurtosis contribution: {kurt_4mc:.6f}")
    
    # 4. Robust optimization
    print("\nCalculating Robust portfolio...")
    w_robust = optimizer.robust_portfolio_optimization(target_return)
    strategies['Robust'] = w_robust
    mu_rob, var_rob, skew_rob, kurt_rob = optimizer.portfolio_moments(w_robust)
    print(f"  Expected return: {mu_rob:.6f}")
    print(f"  Variance: {var_rob:.6f}")
    print(f"  Skewness contribution: {skew_rob:.6f}")
    print(f"  Kurtosis contribution: {kurt_rob:.6f}")
    
    # 5. Adaptive strategy (simulate over time)
    print("\nTesting adaptive strategy...")
    adaptive_weights = []
    regimes = []
    
    for i in range(252, min(len(returns_df), 1000)):  # Limit to first 1000 days after warmup
        w_adapt, regime = optimizer.adaptive_four_moment_portfolio(
            target_return, var_mv * 1.2, 0.0001, xi_y, k_z, current_idx=i
        )
        adaptive_weights.append(w_adapt)
        regimes.append(regime)
        
        if (i - 252) % 100 == 0:
            print(f"  Processed {i - 252} days...")
    
    print(f"  Total {len(adaptive_weights)} adaptive allocations computed")
    
    # Analyze results
    analyze_enhanced_results(returns_df, strategies, adaptive_weights, regimes, optimizer)

# Run the complete test
if __name__ == "__main__":
    test_enhanced_strategies()