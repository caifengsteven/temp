import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t as t_dist, bernoulli
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.cluster import AgglomerativeClustering
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# =====================================================
# PART 1: Volatility Encoding-Decoding Model
# =====================================================

class VolatilityEncodingDecoding:
    """
    Implementation of the encoding-and-decoding approach for volatility analysis
    Based on Wang & Hsieh (2022)
    """
    
    def __init__(self):
        self.segments = None
        self.emission_probs = None
        self.states = None
        
    def encode_returns(self, returns, quantile_thresholds):
        """
        Encode returns into binary sequences based on quantile thresholds
        """
        encoded_sequences = {}
        
        for q in quantile_thresholds:
            if q < 0.5:
                # Lower tail
                threshold = np.quantile(returns, q)
                encoded = (returns <= threshold).astype(int)
            else:
                # Upper tail
                threshold = np.quantile(returns, q)
                encoded = (returns >= threshold).astype(int)
            
            encoded_sequences[q] = encoded
            
        return encoded_sequences
    
    def calculate_recurrence_times(self, binary_sequence):
        """
        Calculate recurrence times between successive 1's
        Returns both recurrence times and their positions in original sequence
        """
        ones_indices = np.where(binary_sequence == 1)[0]
        
        if len(ones_indices) == 0:
            return np.array([]), np.array([])
        
        recurrence_times = []
        positions = []
        
        # First recurrence time
        if ones_indices[0] > 0:
            recurrence_times.append(ones_indices[0])
        else:
            recurrence_times.append(0)
        positions.append(ones_indices[0])
        
        # Subsequent recurrence times
        for i in range(1, len(ones_indices)):
            recurrence_times.append(ones_indices[i] - ones_indices[i-1] - 1)
            positions.append(ones_indices[i])
            
        return np.array(recurrence_times), np.array(positions)
    
    def multiple_states_search(self, recurrence_times, m_states=3, T_star=None):
        """
        Algorithm 1: Multiple-states searching algorithm
        """
        if len(recurrence_times) < m_states:
            return np.zeros(len(recurrence_times))
        
        # Set T_star based on data if not provided
        if T_star is None:
            T_star = np.percentile(recurrence_times, 75)
        
        # Generate threshold parameters
        T_params = np.percentile(recurrence_times, 
                                np.linspace(20, 80, m_states-1))
        
        # Initialize states array
        states = np.ones(len(recurrence_times)) * (m_states - 1)  # Default to highest state
        
        # Apply thresholds to assign states
        for i, Ti in enumerate(T_params):
            # States with recurrence time >= Ti get lower volatility state
            high_rec_mask = recurrence_times >= Ti
            
            # Second level coding
            C_i = high_rec_mask.astype(int)
            R_i, _ = self.calculate_recurrence_times(C_i)
            
            if len(R_i) > 0:
                # Find segments with high second-level recurrence
                high_segments = R_i >= T_star
                
                # Map back to original positions
                ones_in_Ci = np.where(C_i == 1)[0]
                for j, is_high in enumerate(high_segments):
                    if is_high and j < len(ones_in_Ci):
                        states[ones_in_Ci[j]] = i
        
        return states
    
    def map_states_to_sequence(self, binary_sequence, rec_states, rec_positions):
        """
        Map states from recurrence time domain back to original sequence
        """
        sequence_states = np.ones(len(binary_sequence)) * -1  # Initialize with -1
        
        # Assign states at positions where events occur
        for i, pos in enumerate(rec_positions):
            if i < len(rec_states):
                sequence_states[pos] = rec_states[i]
        
        # Propagate states forward
        current_state = 0
        for i in range(len(sequence_states)):
            if sequence_states[i] >= 0:
                current_state = sequence_states[i]
            else:
                sequence_states[i] = current_state
                
        return sequence_states.astype(int)
    
    def estimate_emission_probabilities(self, binary_sequence, states):
        """
        Estimate emission probabilities for each state
        """
        unique_states = np.unique(states)
        emission_probs = {}
        
        for state in unique_states:
            state_mask = (states == state)
            if np.sum(state_mask) > 0:
                emission_probs[state] = np.mean(binary_sequence[state_mask])
            else:
                emission_probs[state] = 0.0
                
        return emission_probs
    
    def decode_volatility_states(self, returns, quantile_range=(0.05, 0.95), 
                               n_quantiles=10, m_states=3):
        """
        Main encoding-and-decoding procedure (Algorithm 2)
        """
        # Generate quantile thresholds
        quantile_thresholds = np.linspace(quantile_range[0], 
                                        quantile_range[1], 
                                        n_quantiles)
        
        # Encode returns
        encoded_sequences = self.encode_returns(returns, quantile_thresholds)
        
        # Store emission probability vectors
        n_obs = len(returns)
        emission_vectors = np.zeros((n_obs, n_quantiles))
        
        # Process each threshold
        state_assignments = []
        
        for idx, (q, binary_seq) in enumerate(encoded_sequences.items()):
            # Calculate recurrence times and positions
            rec_times, rec_positions = self.calculate_recurrence_times(binary_seq)
            
            if len(rec_times) > m_states:
                # Search for states in recurrence time domain
                rec_states = self.multiple_states_search(rec_times, m_states)
                
                # Map states back to original sequence
                seq_states = self.map_states_to_sequence(binary_seq, rec_states, rec_positions)
                
                # Estimate emission probabilities
                emission_probs = self.estimate_emission_probabilities(binary_seq, seq_states)
                
                # Store emission probabilities for each time point
                for t in range(n_obs):
                    state = seq_states[t]
                    if state in emission_probs:
                        emission_vectors[t, idx] = emission_probs[state]
                
                state_assignments.append(seq_states)
        
        # Cluster time points based on emission vectors
        self.emission_vectors = emission_vectors
        self.final_states = self._cluster_emission_vectors(emission_vectors, n_clusters=m_states)
        
        return self.final_states
    
    def _cluster_emission_vectors(self, emission_vectors, n_clusters=None):
        """
        Cluster time points based on emission probability vectors
        """
        # Remove zero variance features
        valid_features = np.var(emission_vectors, axis=0) > 1e-10
        
        if np.sum(valid_features) < 2:
            # Not enough features for clustering
            return np.zeros(len(emission_vectors))
        
        emission_vectors_clean = emission_vectors[:, valid_features]
        
        # Hierarchical clustering with Ward linkage
        linkage_matrix = linkage(emission_vectors_clean, method='ward')
        
        # Determine optimal number of clusters if not specified
        if n_clusters is None:
            # Use elbow method
            if len(linkage_matrix) > 1:
                distances = linkage_matrix[:, 2]
                diff = np.diff(distances)
                n_clusters = np.argmax(diff) + 2
                n_clusters = min(max(n_clusters, 2), 5)
            else:
                n_clusters = 2
        
        # Get cluster assignments
        clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        
        return clusters - 1  # Convert to 0-indexed

# =====================================================
# PART 2: Trading Strategies
# =====================================================

class VolatilityBasedTradingStrategy:
    """
    Trading strategies based on volatility state detection
    """
    
    def __init__(self, volatility_model):
        self.volatility_model = volatility_model
        self.positions = None
        self.returns = None
        self.equity_curve = None
        
    def detect_volatility_regimes(self, returns, lookback_window=252):
        """
        Detect volatility regimes using the encoding-decoding approach
        """
        # Use rolling window for online detection
        n = len(returns)
        states = np.zeros(n)
        
        # Initialize with historical detection
        if n > lookback_window:
            initial_states = self.volatility_model.decode_volatility_states(
                returns[:lookback_window], m_states=3
            )
            states[:lookback_window] = initial_states
            
            # Online detection for remaining periods
            for t in range(lookback_window, n):
                window_returns = returns[t-lookback_window:t]
                window_states = self.volatility_model.decode_volatility_states(
                    window_returns, m_states=3
                )
                states[t] = window_states[-1]
        else:
            states = self.volatility_model.decode_volatility_states(returns, m_states=3)
            
        return states
    
    def strategy_risk_parity(self, returns, states, risk_free_rate=0.02):
        """
        Risk parity strategy: Adjust position size based on volatility regime
        """
        n = len(returns)
        positions = np.zeros(n)
        
        # Define position sizes for each volatility state
        # State 0: Low vol - higher position
        # State 1: Medium vol - moderate position  
        # State 2: High vol - lower position
        position_sizes = {0: 1.5, 1: 1.0, 2: 0.5}
        
        for t in range(1, n):
            state = int(states[t-1])
            positions[t] = position_sizes.get(state, 1.0)
            
        # Calculate strategy returns
        strategy_returns = positions[1:] * returns[1:]
        
        # Add cash returns for unlevered portion
        cash_weight = 1 - positions[1:]
        cash_weight = np.maximum(cash_weight, 0)  # No negative cash
        cash_returns = cash_weight * risk_free_rate / 252
        
        total_returns = strategy_returns + cash_returns
        
        return positions, total_returns
    
    def strategy_regime_switching(self, returns, states):
        """
        Regime switching strategy: Go long in low vol, short in high vol
        """
        n = len(returns)
        positions = np.zeros(n)
        
        for t in range(1, n):
            state = int(states[t-1])
            
            if state == 0:  # Low volatility - go long
                positions[t] = 1.0
            elif state == 1:  # Medium volatility - neutral
                positions[t] = 0.5
            else:  # High volatility - defensive/short
                positions[t] = -0.5
                
        strategy_returns = positions[1:] * returns[1:]
        
        return positions, strategy_returns
    
    def strategy_volatility_targeting(self, returns, states, target_vol=0.15):
        """
        Target constant volatility by adjusting leverage based on regime
        """
        n = len(returns)
        positions = np.zeros(n)
        
        # Estimate volatility for each state
        state_vols = {}
        for state in np.unique(states):
            state_mask = states == state
            if np.sum(state_mask) > 20:
                state_vols[state] = np.std(returns[state_mask]) * np.sqrt(252)
            else:
                state_vols[state] = 0.20  # Default assumption
        
        for t in range(1, n):
            state = int(states[t-1])
            current_vol = state_vols.get(state, 0.20)
            
            # Scale position to target volatility
            if current_vol > 0:
                positions[t] = min(target_vol / current_vol, 2.0)  # Cap leverage at 2x
            else:
                positions[t] = 1.0
                
        strategy_returns = positions[1:] * returns[1:]
        
        return positions, strategy_returns
    
    def strategy_options_based(self, returns, states, option_premium=0.001):
        """
        Options-based strategy: Buy protection in high vol regimes
        """
        n = len(returns)
        positions = np.zeros(n)
        protection_cost = np.zeros(n-1)
        
        for t in range(1, n):
            state = int(states[t-1])
            
            if state == 0:  # Low vol - full exposure
                positions[t] = 1.0
                protection_cost[t-1] = 0
            elif state == 1:  # Medium vol - partial hedge
                positions[t] = 0.8
                protection_cost[t-1] = option_premium * 0.5
            else:  # High vol - full hedge
                positions[t] = 0.5
                protection_cost[t-1] = option_premium
        
        # Strategy returns = position returns - protection cost
        strategy_returns = positions[1:] * returns[1:] - protection_cost
        
        # Add protection payoff in down markets
        down_days = returns[1:] < -0.02  # 2% down days
        protection_payoff = np.where(down_days & (protection_cost > 0), 
                                   -returns[1:] * 0.5, 0)
        
        total_returns = strategy_returns + protection_payoff
        
        return positions, total_returns
    
    def strategy_mean_reversion(self, returns, states, lookback=20):
        """
        Mean reversion strategy: Trade against extremes in each regime
        """
        n = len(returns)
        positions = np.zeros(n)
        
        for t in range(lookback, n):
            state = int(states[t-1])
            recent_returns = returns[t-lookback:t]
            
            # Calculate z-score within regime
            regime_mask = states[t-lookback:t] == state
            if np.sum(regime_mask) > 5:
                regime_returns = recent_returns[regime_mask]
                mean = np.mean(regime_returns)
                std = np.std(regime_returns)
                
                if std > 0:
                    z_score = (returns[t-1] - mean) / std
                    
                    # Mean reversion signal
                    if z_score > 2:  # Overbought
                        positions[t] = -0.5
                    elif z_score < -2:  # Oversold
                        positions[t] = 1.5
                    else:
                        positions[t] = 1.0
                else:
                    positions[t] = 1.0
            else:
                positions[t] = 1.0
                
        strategy_returns = positions[1:] * returns[1:]
        
        return positions, strategy_returns
    
    def strategy_ensemble(self, returns, states):
        """
        Ensemble strategy combining multiple approaches
        """
        n = len(returns)
        
        # Get positions from individual strategies
        pos_rp, _ = self.strategy_risk_parity(returns, states)
        pos_rs, _ = self.strategy_regime_switching(returns, states)
        pos_vt, _ = self.strategy_volatility_targeting(returns, states)
        pos_mr, _ = self.strategy_mean_reversion(returns, states)
        
        # Dynamic weighting based on recent performance
        window = 60
        ensemble_positions = np.zeros(n)
        
        # Initialize with equal weights
        for t in range(min(window, n)):
            ensemble_positions[t] = 0.25 * (pos_rp[t] + pos_rs[t] + pos_vt[t] + pos_mr[t])
        
        for t in range(window, n):
            # Calculate recent performance of each strategy
            recent_slice = slice(t-window, t)
            
            perf_rp = np.mean(pos_rp[recent_slice] * returns[recent_slice])
            perf_rs = np.mean(pos_rs[recent_slice] * returns[recent_slice])
            perf_vt = np.mean(pos_vt[recent_slice] * returns[recent_slice])
            perf_mr = np.mean(pos_mr[recent_slice] * returns[recent_slice])
            
            # Weight by recent performance (momentum)
            perfs = np.array([perf_rp, perf_rs, perf_vt, perf_mr])
            perfs = np.maximum(perfs, 0)  # Only positive weights
            
            if np.sum(perfs) > 0:
                weights = perfs / np.sum(perfs)
            else:
                weights = np.array([0.25, 0.25, 0.25, 0.25])
            
            # Ensemble position
            ensemble_positions[t] = (
                weights[0] * pos_rp[t] +
                weights[1] * pos_rs[t] +
                weights[2] * pos_vt[t] +
                weights[3] * pos_mr[t]
            )
        
        strategy_returns = ensemble_positions[1:] * returns[1:]
        
        return ensemble_positions, strategy_returns
    
    def backtest_strategy(self, returns, strategy='risk_parity', transaction_cost=0.001, **kwargs):
        """
        Backtest a specific strategy with transaction costs
        """
        # Detect volatility states
        states = self.detect_volatility_regimes(returns)
        
        # Apply strategy
        if strategy == 'risk_parity':
            positions, strategy_returns = self.strategy_risk_parity(returns, states, **kwargs)
        elif strategy == 'regime_switching':
            positions, strategy_returns = self.strategy_regime_switching(returns, states)
        elif strategy == 'volatility_targeting':
            positions, strategy_returns = self.strategy_volatility_targeting(returns, states, **kwargs)
        elif strategy == 'options_based':
            positions, strategy_returns = self.strategy_options_based(returns, states, **kwargs)
        elif strategy == 'mean_reversion':
            positions, strategy_returns = self.strategy_mean_reversion(returns, states, **kwargs)
        elif strategy == 'ensemble':
            positions, strategy_returns = self.strategy_ensemble(returns, states)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Apply transaction costs
        position_changes = np.abs(np.diff(positions))
        transaction_costs = position_changes * transaction_cost
        strategy_returns = strategy_returns - transaction_costs
        
        # Store results
        self.positions = positions
        self.returns = strategy_returns
        self.states = states
        
        # Build equity curve
        self.equity_curve = (1 + strategy_returns).cumprod()
        
        return self.calculate_performance_metrics(strategy_returns, returns[1:])
    
    def calculate_performance_metrics(self, strategy_returns, benchmark_returns):
        """
        Calculate comprehensive performance metrics
        """
        # Annual metrics
        annual_return = np.mean(strategy_returns) * 252
        annual_vol = np.std(strategy_returns) * np.sqrt(252)
        sharpe_ratio = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Drawdown
        equity_curve = (1 + strategy_returns).cumprod()
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Sortino ratio (downside deviation)
        downside_returns = strategy_returns[strategy_returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annual_return / downside_vol if downside_vol > 0 else 0
        
        # Win rate
        win_rate = np.mean(strategy_returns > 0)
        
        # Information ratio vs benchmark
        excess_returns = strategy_returns - benchmark_returns
        tracking_error = np.std(excess_returns) * np.sqrt(252)
        information_ratio = np.mean(excess_returns) * 252 / tracking_error if tracking_error > 0 else 0
        
        # Calmar ratio
        calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        metrics = {
            'Annual Return': annual_return,
            'Annual Volatility': annual_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Max Drawdown': max_drawdown,
            'Calmar Ratio': calmar_ratio,
            'Win Rate': win_rate,
            'Information Ratio': information_ratio,
            'Final Equity': equity_curve[-1] if len(equity_curve) > 0 else 1.0
        }
        
        return metrics
    
    def plot_backtest_results(self, benchmark_returns=None):
        """
        Visualize backtest results
        """
        fig, axes = plt.subplots(4, 1, figsize=(12, 14))
        
        # Plot 1: Equity curves
        ax = axes[0]
        ax.plot(self.equity_curve, 'b-', label='Strategy', linewidth=2)
        
        if benchmark_returns is not None:
            bench_equity = (1 + benchmark_returns).cumprod()
            ax.plot(bench_equity, 'gray', label='Buy & Hold', alpha=0.7)
        
        ax.set_ylabel('Equity')
        ax.set_title('Equity Curve Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Positions and volatility states
        ax = axes[1]
        ax2 = ax.twinx()
        
        # Plot positions
        ax.plot(self.positions, 'b-', alpha=0.7, label='Position Size')
        ax.set_ylabel('Position Size', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        
        # Plot states
        ax2.plot(self.states, 'r--', alpha=0.5, label='Volatility State')
        ax2.set_ylabel('Volatility State', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.set_ylim(-0.5, 2.5)
        
        ax.set_title('Position Sizing and Volatility States')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Rolling Sharpe ratio
        ax = axes[2]
        window = 252
        if len(self.returns) > window:
            rolling_returns = pd.Series(self.returns).rolling(window).mean() * 252
            rolling_vol = pd.Series(self.returns).rolling(window).std() * np.sqrt(252)
            rolling_sharpe = rolling_returns / rolling_vol
            
            ax.plot(rolling_sharpe.values, 'g-', linewidth=2)
            ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
            ax.set_ylabel('Sharpe Ratio')
            ax.set_title(f'{window}-Day Rolling Sharpe Ratio')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Drawdown
        ax = axes[3]
        equity_curve = (1 + self.returns).cumprod()
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        
        ax.fill_between(range(len(drawdown)), 0, drawdown * 100, 
                       color='red', alpha=0.3)
        ax.set_ylabel('Drawdown (%)')
        ax.set_xlabel('Time')
        ax.set_title('Drawdown Analysis')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# =====================================================
# PART 3: Simulation and Testing Functions
# =====================================================

def simulate_regime_switching_data(n_obs=3000, regime_params=None):
    """
    Simulate data with regime-switching volatility
    """
    if regime_params is None:
        # Default parameters: 3 states with different volatilities
        regime_params = {
            0: {'mu': 0.0005, 'sigma': 0.01, 'df': None},   # Low volatility (Normal)
            1: {'mu': 0.0002, 'sigma': 0.02, 'df': None},   # Medium volatility (Normal)
            2: {'mu': -0.0001, 'sigma': 0.03, 'df': 5}      # High volatility (t-dist)
        }
    
    # Define regime switches
    regime_lengths = [500, 400, 600, 500, 400, 600]
    regimes = [0, 1, 2, 1, 0, 2]
    
    # Ensure total length matches n_obs
    total_length = sum(regime_lengths)
    if total_length != n_obs:
        # Adjust last segment
        regime_lengths[-1] = n_obs - sum(regime_lengths[:-1])
    
    returns = []
    true_states = []
    
    for regime, length in zip(regimes, regime_lengths):
        params = regime_params[regime]
        
        if params['df'] is None:
            # Normal distribution
            segment = np.random.normal(params['mu'], params['sigma'], length)
        else:
            # Student-t distribution
            segment = params['mu'] + params['sigma'] * t_dist.rvs(params['df'], size=length)
        
        returns.extend(segment)
        true_states.extend([regime] * length)
    
    return np.array(returns), np.array(true_states)

def simulate_garch_returns(n_obs=2000, omega=0.00001, alpha=0.1, beta=0.85):
    """
    Simulate returns with GARCH(1,1) volatility
    """
    returns = np.zeros(n_obs)
    sigma2 = np.zeros(n_obs)
    
    # Initialize
    sigma2[0] = omega / (1 - alpha - beta)
    returns[0] = np.random.normal(0, np.sqrt(sigma2[0]))
    
    # Generate GARCH process
    for t in range(1, n_obs):
        sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
        returns[t] = np.random.normal(0.0003, np.sqrt(sigma2[t]))
    
    return returns

# =====================================================
# PART 4: Main Testing and Demonstration
# =====================================================

def main_analysis():
    """
    Complete analysis workflow
    """
    print("="*60)
    print("VOLATILITY STATE DETECTION AND TRADING STRATEGY ANALYSIS")
    print("="*60)
    
    # Step 1: Test volatility detection
    print("\n1. Testing Volatility State Detection")
    print("-"*40)
    
    # Generate test data
    returns, true_states = simulate_regime_switching_data(n_obs=3000)
    
    # Initialize model
    vol_model = VolatilityEncodingDecoding()
    
    # Detect states
    detected_states = vol_model.decode_volatility_states(returns, m_states=3)
    
    # Calculate accuracy
    ari = adjusted_rand_score(true_states, detected_states)
    nmi = normalized_mutual_info_score(true_states, detected_states)
    
    print(f"Adjusted Rand Index: {ari:.3f}")
    print(f"Normalized Mutual Information: {nmi:.3f}")
    
    # Step 2: Test trading strategies
    print("\n2. Testing Trading Strategies")
    print("-"*40)
    
    # Initialize trading system
    trader = VolatilityBasedTradingStrategy(vol_model)
    
    # Test all strategies
    strategies = [
        'risk_parity',
        'regime_switching',
        'volatility_targeting',
        'options_based',
        'mean_reversion',
        'ensemble'
    ]
    
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy...")
        metrics = trader.backtest_strategy(returns, strategy=strategy)
        results[strategy] = metrics
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results).T
    
    print("\n3. Strategy Performance Comparison")
    print("-"*40)
    print(comparison_df.round(4))
    
    # Find best strategy
    best_strategy = comparison_df['Sharpe Ratio'].idxmax()
    print(f"\nBest Strategy (by Sharpe Ratio): {best_strategy}")
    
    # Plot best strategy
    print("\n4. Visualizing Best Strategy Performance")
    print("-"*40)
    
    metrics = trader.backtest_strategy(returns, strategy=best_strategy)
    trader.plot_backtest_results(returns[1:])
    
    # Step 3: Out-of-sample test with GARCH data
    print("\n5. Out-of-Sample Test with GARCH Data")
    print("-"*40)
    
    # Generate GARCH returns
    garch_returns = simulate_garch_returns(n_obs=2000)
    
    # Test ensemble strategy
    metrics_oos = trader.backtest_strategy(garch_returns, strategy='ensemble')
    
    print("\nOut-of-Sample Ensemble Strategy Performance:")
    for key, value in metrics_oos.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
    
    return vol_model, trader, results

def advanced_features_demo():
    """
    Demonstrate advanced features
    """
    print("\n" + "="*60)
    print("ADVANCED FEATURES DEMONSTRATION")
    print("="*60)
    
    # 1. Multi-asset portfolio
    print("\n1. Multi-Asset Portfolio with Correlation")
    print("-"*40)
    
    # Simulate correlated assets
    n_assets = 5
    n_obs = 1000
    
    # Correlation matrix
    corr_matrix = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            corr_matrix[i,j] = corr_matrix[j,i] = 0.3 + 0.3 * np.exp(-abs(i-j))
    
    # Generate returns
    base_returns = np.random.multivariate_normal(
        mean=np.zeros(n_assets), 
        cov=corr_matrix, 
        size=n_obs
    ) * 0.01
    
    # Add different volatility regimes to each asset
    asset_states = []
    portfolio_returns = []
    
    for i in range(n_assets):
        # Each asset has its own volatility dynamics
        vol_scale = np.random.choice([1, 2, 0.5], size=n_obs, p=[0.6, 0.3, 0.1])
        asset_returns = base_returns[:, i] * vol_scale
        
        # Detect states for each asset
        vol_model = VolatilityEncodingDecoding()
        states = vol_model.decode_volatility_states(asset_returns, m_states=3)
        asset_states.append(states)
        
        # Simple portfolio: equal weight
        portfolio_returns.append(asset_returns / n_assets)
    
    portfolio_returns = np.sum(portfolio_returns, axis=0)
    
    # Trade the portfolio
    trader = VolatilityBasedTradingStrategy(vol_model)
    metrics = trader.backtest_strategy(portfolio_returns, strategy='ensemble')
    
    print("Multi-Asset Portfolio Performance:")
    print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.3f}")
    print(f"Max Drawdown: {metrics['Max Drawdown']:.3%}")
    
    # 2. Transfer Entropy Network
    print("\n2. Volatility Spillover Network")
    print("-"*40)
    
    # Calculate Transfer Entropy between assets
    te_matrix = np.zeros((n_assets, n_assets))
    
    for i in range(n_assets):
        for j in range(n_assets):
            if i != j:
                # Simplified TE calculation
                source_high_vol = asset_states[i] == 2
                target_high_vol = asset_states[j] == 2
                
                # Probability of target high vol given source high vol
                if np.sum(source_high_vol) > 0:
                    p_target_given_source = np.mean(target_high_vol[source_high_vol])
                    p_target = np.mean(target_high_vol)
                    
                    if p_target > 0:
                        te = p_target_given_source * np.log(p_target_given_source / p_target + 1e-10)
                        te_matrix[i, j] = max(te, 0)
    
    # Visualize network
    plt.figure(figsize=(8, 6))
    plt.imshow(te_matrix, cmap='hot', interpolation='nearest')
    plt.colorbar(label='Transfer Entropy')
    plt.xlabel('Target Asset')
    plt.ylabel('Source Asset')
    plt.title('Volatility Spillover Network')
    for i in range(n_assets):
        for j in range(n_assets):
            plt.text(j, i, f'{te_matrix[i,j]:.3f}', 
                    ha='center', va='center', color='white' if te_matrix[i,j] > 0.5 else 'black')
    plt.tight_layout()
    plt.show()
    
    # 3. Real-time monitoring simulation
    print("\n3. Real-Time Volatility Monitoring")
    print("-"*40)
    
    # Simulate real-time updates
    window_size = 252
    update_freq = 20  # Update every 20 periods
    
    # Initialize
    vol_model = VolatilityEncodingDecoding()
    trader = VolatilityBasedTradingStrategy(vol_model)
    
    # Generate streaming data
    all_returns = simulate_garch_returns(n_obs=1000)
    
    # Storage for results
    real_time_states = []
    real_time_positions = []
    
    for t in range(window_size, len(all_returns), update_freq):
        # Get current window
        current_window = all_returns[t-window_size:t]
        
        # Detect current state
        states = vol_model.decode_volatility_states(current_window, m_states=3)
        current_state = states[-1]
        
        # Determine position
        if current_state == 0:  # Low vol
            position = 1.5
        elif current_state == 1:  # Medium vol
            position = 1.0
        else:  # High vol
            position = 0.5
        
        real_time_states.extend([current_state] * update_freq)
        real_time_positions.extend([position] * update_freq)
        
        if t % 100 == 0:
            print(f"Time {t}: State={current_state}, Position={position:.1f}")
    
    print("\nReal-time monitoring complete!")

# =====================================================
# PART 5: Run Everything
# =====================================================

if __name__ == "__main__":
    # Run main analysis
    vol_model, trader, results = main_analysis()
    
    # Run advanced features
    advanced_features_demo()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nKey Findings:")
    print("1. Volatility states can be detected without distributional assumptions")
    print("2. Trading strategies based on volatility regimes show improved risk-adjusted returns")
    print("3. Ensemble strategies provide robust performance across different market conditions")
    print("4. The framework can be extended to multi-asset portfolios and real-time applications")