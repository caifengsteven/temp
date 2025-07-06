import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
NUM_FUNDS = 500  # Total number of funds in the universe
NUM_STRATEGIES = 9  # Number of strategies
NUM_MONTHS = 60  # Number of months to simulate
INDEX_FEE = 0.0095  # 95bp annual index fee
MONTHLY_FEE = INDEX_FEE / 12  # Monthly fee
MAX_FUND_WEIGHT = 0.01  # 1% maximum fund weight
STRATEGY_WEIGHT_CAP_FACTOR = 1.2  # Strategy weight cap at 120% of target
MIN_FUND_WEIGHT_AFTER_REDEMPTION = 0.0025  # 25bp minimum weight after redemption
MAX_FUND_WEIGHT_ALLOCATION = 0.0075  # 75bp maximum weight for new allocations
INDEX_TARGET_NUM_FUNDS = 250  # Target number of funds in the index

# Strategy names
STRATEGY_NAMES = [
    "Equity Long/Short",
    "Event Driven",
    "Macro",
    "Relative Value",
    "Multi-Strategy",
    "Credit",
    "Quantitative",
    "Emerging Markets",
    "Commodities"
]

# Function to generate simulated hedge fund data
def generate_hedge_fund_universe():
    """
    Generate a universe of hedge funds with different strategies and returns.
    
    Returns:
    --------
    fund_data: DataFrame
        DataFrame containing fund information
    """
    # Create fund IDs
    fund_ids = [f"Fund_{i}" for i in range(1, NUM_FUNDS + 1)]
    
    # Assign strategies (with some strategies more common than others)
    strategy_probabilities = [0.25, 0.15, 0.12, 0.10, 0.10, 0.08, 0.08, 0.07, 0.05]
    fund_strategies = np.random.choice(STRATEGY_NAMES, size=NUM_FUNDS, p=strategy_probabilities)
    
    # Generate initial AUM (log-normal distribution to reflect real-world skew)
    initial_aum = np.random.lognormal(mean=4.5, sigma=1.2, size=NUM_FUNDS) * 1e6  # in millions
    
    # Assign liquidity terms (monthly, quarterly, semi-annual)
    liquidity_terms = np.random.choice(["Monthly", "Quarterly", "Semi-Annual"], size=NUM_FUNDS, p=[0.5, 0.3, 0.2])
    
    # Notice periods (in days)
    notice_periods = np.random.choice([15, 30, 45, 60, 90], size=NUM_FUNDS, p=[0.2, 0.3, 0.2, 0.2, 0.1])
    
    # Eligibility (initially all eligible)
    eligible = np.ones(NUM_FUNDS, dtype=bool)
    
    # Create DataFrame
    fund_data = pd.DataFrame({
        'fund_id': fund_ids,
        'strategy': fund_strategies,
        'aum_initial': initial_aum,
        'liquidity': liquidity_terms,
        'notice_period': notice_periods,
        'eligible': eligible
    })
    
    # Set 10% of funds as ineligible initially
    ineligible_indices = np.random.choice(NUM_FUNDS, size=int(NUM_FUNDS * 0.1), replace=False)
    fund_data.loc[ineligible_indices, 'eligible'] = False
    
    return fund_data

# Function to simulate monthly fund returns
def simulate_fund_returns(fund_data, num_months):
    """
    Simulate monthly returns for all funds over the specified time period.
    
    Parameters:
    -----------
    fund_data: DataFrame
        DataFrame containing fund information
    num_months: int
        Number of months to simulate
    
    Returns:
    --------
    returns_data: DataFrame
        DataFrame containing monthly returns for each fund
    """
    # Create date range
    start_date = datetime(2018, 1, 1)
    dates = [start_date + timedelta(days=30*i) for i in range(num_months)]
    
    # Strategy-specific parameters (mean, volatility, correlation with market)
    strategy_params = {
        "Equity Long/Short": (0.008, 0.035, 0.7),
        "Event Driven": (0.007, 0.03, 0.5),
        "Macro": (0.006, 0.04, 0.2),
        "Relative Value": (0.005, 0.02, 0.3),
        "Multi-Strategy": (0.006, 0.025, 0.4),
        "Credit": (0.007, 0.03, 0.6),
        "Quantitative": (0.008, 0.045, 0.3),
        "Emerging Markets": (0.009, 0.05, 0.6),
        "Commodities": (0.006, 0.04, 0.4)
    }
    
    # Generate market returns (used for correlation)
    market_returns = np.random.normal(0.007, 0.04, num_months)
    
    # Initialize returns DataFrame
    returns_data = pd.DataFrame(index=dates)
    
    # Generate returns for each fund
    for idx, fund in fund_data.iterrows():
        fund_id = fund['fund_id']
        strategy = fund['strategy']
        mean, vol, corr = strategy_params[strategy]
        
        # Add some fund-specific variation to mean and volatility
        fund_mean = mean + np.random.normal(0, 0.001)
        fund_vol = vol + np.random.normal(0, 0.005)
        
        # Generate correlated returns
        idiosyncratic = np.random.normal(0, fund_vol, num_months)
        fund_returns = fund_mean + corr * (market_returns - 0.007) + idiosyncratic
        
        # Add some randomness to returns (some months might be particularly good/bad)
        fund_returns += np.random.normal(0, 0.01, num_months)
        
        # Store returns in DataFrame
        returns_data[fund_id] = fund_returns
    
    # Add market returns as a reference
    returns_data['Market'] = market_returns
    
    return returns_data

# Function to simulate monthly AUM changes
def simulate_aum_changes(fund_data, returns_data):
    """
    Simulate monthly AUM changes for all funds based on returns and flows.
    
    Parameters:
    -----------
    fund_data: DataFrame
        DataFrame containing fund information
    returns_data: DataFrame
        DataFrame containing monthly returns for each fund
    
    Returns:
    --------
    aum_data: DataFrame
        DataFrame containing monthly AUM for each fund
    """
    # Initialize AUM DataFrame with same index as returns
    aum_data = pd.DataFrame(index=returns_data.index)
    
    # Set initial AUM
    for idx, fund in fund_data.iterrows():
        fund_id = fund['fund_id']
        aum_data.loc[aum_data.index[0], fund_id] = fund['aum_initial']
    
    # Calculate AUM for each month based on returns and flows
    for i in range(1, len(aum_data)):
        for idx, fund in fund_data.iterrows():
            fund_id = fund['fund_id']
            prev_aum = aum_data.iloc[i-1][fund_id]
            fund_return = returns_data.iloc[i-1][fund_id]
            
            # Simulate fund flows (some random inflows/outflows)
            flow_pct = np.random.normal(0, 0.05)  # -5% to +5% typical flows
            
            # Calculate new AUM
            new_aum = prev_aum * (1 + fund_return) * (1 + flow_pct)
            aum_data.loc[aum_data.index[i], fund_id] = new_aum
    
    return aum_data

# Function to calculate 12-month moving average of strategy weights in the universe
def calculate_strategy_target_weights(fund_data, aum_data):
    """
    Calculate target strategy weights based on 12-month moving average of AUM.
    
    Parameters:
    -----------
    fund_data: DataFrame
        DataFrame containing fund information
    aum_data: DataFrame
        DataFrame containing monthly AUM for each fund
    
    Returns:
    --------
    target_weights: DataFrame
        DataFrame containing target strategy weights for each month
    """
    # Initialize target weights DataFrame
    target_weights = pd.DataFrame(index=aum_data.index, columns=STRATEGY_NAMES)
    
    # For each month
    for i in range(len(aum_data)):
        current_date = aum_data.index[i]
        
        # Calculate total AUM by strategy for the current month
        strategy_aum = {}
        for strategy in STRATEGY_NAMES:
            strategy_aum[strategy] = 0
        
        # Sum AUM for each strategy
        for idx, fund in fund_data.iterrows():
            fund_id = fund['fund_id']
            strategy = fund['strategy']
            
            if fund_id in aum_data.columns:
                strategy_aum[strategy] += aum_data.iloc[i][fund_id]
        
        # Calculate strategy weights
        total_aum = sum(strategy_aum.values())
        for strategy in STRATEGY_NAMES:
            strategy_weight = strategy_aum[strategy] / total_aum if total_aum > 0 else 0
            target_weights.loc[current_date, strategy] = strategy_weight
    
    # Calculate 12-month moving average (or available history if less than 12 months)
    ma_target_weights = pd.DataFrame(index=aum_data.index, columns=STRATEGY_NAMES)
    for i in range(len(aum_data)):
        for strategy in STRATEGY_NAMES:
            lookback_start = max(0, i - 11)  # Use up to 12 months of history
            ma_target_weights.iloc[i][strategy] = target_weights.iloc[lookback_start:i+1][strategy].mean()
    
    # Apply strategy weight limits (optional)
    # For simplicity, we'll cap each strategy at 25%
    for i in range(len(ma_target_weights)):
        for strategy in STRATEGY_NAMES:
            ma_target_weights.iloc[i][strategy] = min(ma_target_weights.iloc[i][strategy], 0.25)
        
        # Renormalize after capping
        ma_target_weights.iloc[i] = ma_target_weights.iloc[i] / ma_target_weights.iloc[i].sum()
    
    return ma_target_weights

# Class to implement the Hedge Fund Index
class HedgeFundIndex:
    def __init__(self, fund_universe, returns_data, aum_data, target_strategy_weights):
        """
        Initialize the Hedge Fund Index.
        
        Parameters:
        -----------
        fund_universe: DataFrame
            DataFrame containing fund information
        returns_data: DataFrame
            DataFrame containing monthly returns for each fund
        aum_data: DataFrame
            DataFrame containing monthly AUM for each fund
        target_strategy_weights: DataFrame
            DataFrame containing target strategy weights for each month
        """
        self.fund_universe = fund_universe
        self.returns_data = returns_data
        self.aum_data = aum_data
        self.target_strategy_weights = target_strategy_weights
        self.dates = returns_data.index
        
        # Initialize index tracking variables
        self.index_level = pd.Series(index=self.dates, dtype=float)
        self.index_level.iloc[0] = 1000.0  # Initial index level
        
        self.index_return = pd.Series(index=self.dates, dtype=float)
        self.index_return.iloc[0] = 0.0
        
        # Fund weights at beginning and end of each month
        self.fund_weights_begin = pd.DataFrame(index=self.dates, columns=fund_universe['fund_id'])
        self.fund_weights_end = pd.DataFrame(index=self.dates, columns=fund_universe['fund_id'])
        
        # Track residual weights (outstanding redemptions)
        self.residual_weights = pd.DataFrame(index=self.dates, columns=fund_universe['fund_id'], data=0.0)
        
        # Track index composition by strategy
        self.strategy_weights = pd.DataFrame(index=self.dates, columns=STRATEGY_NAMES, data=0.0)
        
        # Track notional amount of index
        self.notional = pd.Series(index=self.dates, dtype=float)
        self.notional.iloc[0] = 100e6  # Initial notional amount ($100M)
        
        # Track funds in the index
        self.index_funds = set()
        
        # Initialize index with equal-weighted eligible funds
        self._initialize_index()
    
    def _initialize_index(self):
        """
        Initialize the index with eligible funds based on target strategy weights.
        """
        # Get target strategy weights for first month
        first_target_weights = self.target_strategy_weights.iloc[0]
        
        # Calculate target number of funds per strategy
        target_funds_per_strategy = {}
        for strategy in STRATEGY_NAMES:
            target_funds_per_strategy[strategy] = int(INDEX_TARGET_NUM_FUNDS * first_target_weights[strategy])
        
        # Ensure at least one fund per strategy if target weight > 0
        for strategy in STRATEGY_NAMES:
            if first_target_weights[strategy] > 0 and target_funds_per_strategy[strategy] == 0:
                target_funds_per_strategy[strategy] = 1
        
        # Select eligible funds for each strategy
        selected_funds = []
        for strategy in STRATEGY_NAMES:
            # Get eligible funds for this strategy
            eligible_funds = self.fund_universe[
                (self.fund_universe['strategy'] == strategy) & 
                (self.fund_universe['eligible'] == True)
            ]
            
            # Sort by AUM (larger funds first)
            eligible_funds = eligible_funds.sort_values('aum_initial', ascending=False)
            
            # Select target number of funds (or all eligible funds if fewer)
            num_to_select = min(target_funds_per_strategy[strategy], len(eligible_funds))
            selected_strategy_funds = eligible_funds.iloc[:num_to_select]['fund_id'].tolist()
            
            # Add to selected funds
            selected_funds.extend(selected_strategy_funds)
        
        # Set initial weights (equal weight within each strategy based on target weights)
        self.index_funds = set(selected_funds)
        
        # Calculate initial weights
        for fund_id in selected_funds:
            strategy = self.fund_universe.loc[self.fund_universe['fund_id'] == fund_id, 'strategy'].iloc[0]
            num_funds_in_strategy = len([f for f in selected_funds if self.fund_universe.loc[self.fund_universe['fund_id'] == f, 'strategy'].iloc[0] == strategy])
            
            if num_funds_in_strategy > 0:
                weight = first_target_weights[strategy] / num_funds_in_strategy
                self.fund_weights_begin.loc[self.dates[0], fund_id] = weight
        
        # Fill NAs with 0
        self.fund_weights_begin.fillna(0, inplace=True)
        
        # Calculate strategy weights
        self._update_strategy_weights(0)
    
    def _update_strategy_weights(self, month_idx):
        """
        Update strategy weights based on current fund weights.
        
        Parameters:
        -----------
        month_idx: int
            Index of the current month
        """
        # Reset strategy weights
        for strategy in STRATEGY_NAMES:
            self.strategy_weights.iloc[month_idx][strategy] = 0.0
        
        # Calculate strategy weights based on fund weights
        for fund_id in self.fund_universe['fund_id']:
            if self.fund_weights_begin.iloc[month_idx][fund_id] > 0:
                strategy = self.fund_universe.loc[self.fund_universe['fund_id'] == fund_id, 'strategy'].iloc[0]
                self.strategy_weights.iloc[month_idx][strategy] += self.fund_weights_begin.iloc[month_idx][fund_id]
    
    def _calculate_monthly_return(self, month_idx):
        """
        Calculate index return for the given month.
        
        Parameters:
        -----------
        month_idx: int
            Index of the current month
        
        Returns:
        --------
        index_return: float
            Return of the index for the month
        """
        index_return = 0.0
        
        # Calculate weighted sum of fund returns
        for fund_id in self.fund_universe['fund_id']:
            fund_weight = self.fund_weights_begin.iloc[month_idx][fund_id]
            residual_weight = self.residual_weights.iloc[month_idx][fund_id]
            active_weight = fund_weight - residual_weight
            
            if active_weight > 0:
                fund_return = self.returns_data.iloc[month_idx][fund_id]
                index_return += active_weight * fund_return
        
        # Apply index fee
        index_return -= MONTHLY_FEE
        
        return index_return
    
    def _update_end_weights(self, month_idx):
        """
        Update fund weights at the end of the month based on returns.
        
        Parameters:
        -----------
        month_idx: int
            Index of the current month
        """
        index_return = self.index_return.iloc[month_idx]
        
        # Update end weights based on fund returns
        for fund_id in self.fund_universe['fund_id']:
            begin_weight = self.fund_weights_begin.iloc[month_idx][fund_id]
            residual_weight = self.residual_weights.iloc[month_idx][fund_id]
            active_weight = begin_weight - residual_weight
            
            if active_weight > 0:
                fund_return = self.returns_data.iloc[month_idx][fund_id]
                end_weight = active_weight * (1 + fund_return) / (1 + index_return)
                self.fund_weights_end.iloc[month_idx][fund_id] = end_weight + residual_weight
            elif begin_weight > 0:
                # Carry forward residual weight
                self.fund_weights_end.iloc[month_idx][fund_id] = residual_weight
            else:
                self.fund_weights_end.iloc[month_idx][fund_id] = 0
    
    def _adjust_weights(self, month_idx):
        """
        Adjust weights for the next month based on strategy targets and fund constraints.
        
        Parameters:
        -----------
        month_idx: int
            Index of the current month
        """
        if month_idx >= len(self.dates) - 1:
            return  # No next month to adjust for
        
        # Copy end weights to beginning weights for next month
        for fund_id in self.fund_universe['fund_id']:
            self.fund_weights_begin.iloc[month_idx + 1][fund_id] = self.fund_weights_end.iloc[month_idx][fund_id]
        
        # Reset residual weights for next month
        for fund_id in self.fund_universe['fund_id']:
            self.residual_weights.iloc[month_idx + 1][fund_id] = 0
        
        # Check if rebalancing is needed based on strategy weight caps
        target_weights = self.target_strategy_weights.iloc[month_idx]
        current_weights = self.strategy_weights.iloc[month_idx]
        
        # Calculate aggregate reallocation weight (ARW)
        arw = 0.0
        
        # Simulate some inflow/outflow (randomly varying between -5% and +7%)
        notional_change_pct = np.random.normal(0.01, 0.03)  # Mean 1%, SD 3%
        self.notional.iloc[month_idx + 1] = self.notional.iloc[month_idx] * (1 + notional_change_pct)
        
        if notional_change_pct != 0:
            arw = notional_change_pct / (1 + notional_change_pct)
        
        # Check for strategy weight cap breaches
        strategy_excesses = {}
        strategy_shortfalls = {}
        
        for strategy in STRATEGY_NAMES:
            target = target_weights[strategy]
            current = current_weights[strategy]
            cap = target * STRATEGY_WEIGHT_CAP_FACTOR
            
            if current > cap:
                strategy_excesses[strategy] = current - target
            elif current < target:
                strategy_shortfalls[strategy] = target - current
        
        # Handle redemptions for strategies over cap
        if strategy_excesses:
            # Sort strategies by excess (largest first)
            sorted_excesses = sorted(strategy_excesses.items(), key=lambda x: x[1], reverse=True)
            
            for strategy, excess in sorted_excesses:
                # Calculate redemption amount
                redemption_amount = excess * 0.5  # Redeem half of the excess
                
                # Identify funds in this strategy
                strategy_funds = [
                    fund_id for fund_id in self.index_funds 
                    if self.fund_universe.loc[self.fund_universe['fund_id'] == fund_id, 'strategy'].iloc[0] == strategy
                ]
                
                # Sort funds by weight (largest first)
                strategy_funds.sort(
                    key=lambda fund_id: self.fund_weights_begin.iloc[month_idx + 1][fund_id], 
                    reverse=True
                )
                
                # Redeem from funds in order of weight
                remaining_redemption = redemption_amount
                for fund_id in strategy_funds:
                    fund_weight = self.fund_weights_begin.iloc[month_idx + 1][fund_id]
                    
                    # Skip funds with zero weight
                    if fund_weight <= 0:
                        continue
                    
                    # Calculate amount to redeem from this fund
                    fund_redemption = min(remaining_redemption, fund_weight - MIN_FUND_WEIGHT_AFTER_REDEMPTION)
                    
                    if fund_redemption > 0:
                        # Update weights
                        self.fund_weights_begin.iloc[month_idx + 1][fund_id] -= fund_redemption
                        self.residual_weights.iloc[month_idx + 1][fund_id] += fund_redemption
                        
                        # Update remaining redemption
                        remaining_redemption -= fund_redemption
                        
                        # Break if redemption complete
                        if remaining_redemption <= 0:
                            break
        
        # Handle allocations (inflows and rebalancing)
        if arw > 0 or strategy_shortfalls:
            # Determine allocation to each strategy
            allocation_weights = {}
            
            if arw > 0:
                # Allocate inflow proportionally to strategy shortfalls
                total_shortfall = sum(strategy_shortfalls.values())
                
                if total_shortfall > 0:
                    for strategy, shortfall in strategy_shortfalls.items():
                        allocation_weights[strategy] = arw * (shortfall / total_shortfall)
                else:
                    # If no shortfalls, allocate proportionally to target weights
                    for strategy in STRATEGY_NAMES:
                        allocation_weights[strategy] = arw * target_weights[strategy]
            
            # Implement allocations
            for strategy, allocation in allocation_weights.items():
                if allocation <= 0:
                    continue
                
                # Identify eligible funds in this strategy
                eligible_funds = self.fund_universe[
                    (self.fund_universe['strategy'] == strategy) & 
                    (self.fund_universe['eligible'] == True)
                ]
                
                # Get fund IDs
                eligible_fund_ids = eligible_funds['fund_id'].tolist()
                
                # Separate existing and new funds
                existing_funds = [fund_id for fund_id in eligible_fund_ids if fund_id in self.index_funds]
                new_funds = [fund_id for fund_id in eligible_fund_ids if fund_id not in self.index_funds]
                
                # Sort existing funds by weight (smallest first)
                existing_funds.sort(
                    key=lambda fund_id: self.fund_weights_begin.iloc[month_idx + 1][fund_id]
                )
                
                # Sort new funds by AUM (largest first)
                new_funds.sort(
                    key=lambda fund_id: self.aum_data.iloc[month_idx][fund_id] 
                    if fund_id in self.aum_data.columns else 0, 
                    reverse=True
                )
                
                # Allocate to existing funds first
                remaining_allocation = allocation
                for fund_id in existing_funds:
                    current_weight = self.fund_weights_begin.iloc[month_idx + 1][fund_id]
                    
                    # Calculate allocation to this fund (cap at MAX_FUND_WEIGHT_ALLOCATION)
                    fund_allocation = min(
                        remaining_allocation,
                        MAX_FUND_WEIGHT_ALLOCATION - current_weight,
                        MAX_FUND_WEIGHT - current_weight
                    )
                    
                    if fund_allocation > 0:
                        # Update weight
                        self.fund_weights_begin.iloc[month_idx + 1][fund_id] += fund_allocation
                        
                        # Update remaining allocation
                        remaining_allocation -= fund_allocation
                        
                        # Break if allocation complete
                        if remaining_allocation <= 0:
                            break
                
                # If there's still allocation remaining, add new funds
                if remaining_allocation > 0 and new_funds:
                    for fund_id in new_funds:
                        # Calculate allocation to this fund (cap at MAX_FUND_WEIGHT_ALLOCATION)
                        fund_allocation = min(
                            remaining_allocation,
                            MAX_FUND_WEIGHT_ALLOCATION
                        )
                        
                        if fund_allocation > 0:
                            # Add fund to index
                            self.index_funds.add(fund_id)
                            
                            # Set weight
                            self.fund_weights_begin.iloc[month_idx + 1][fund_id] = fund_allocation
                            
                            # Update remaining allocation
                            remaining_allocation -= fund_allocation
                            
                            # Break if allocation complete
                            if remaining_allocation <= 0:
                                break
        
        # Ensure weights sum to 1
        total_weight = sum(self.fund_weights_begin.iloc[month_idx + 1])
        if abs(total_weight - 1.0) > 1e-10:  # Allow for small numerical errors
            # Normalize weights
            for fund_id in self.fund_universe['fund_id']:
                if self.fund_weights_begin.iloc[month_idx + 1][fund_id] > 0:
                    self.fund_weights_begin.iloc[month_idx + 1][fund_id] /= total_weight
        
        # Update strategy weights for next month
        self._update_strategy_weights(month_idx + 1)
    
    def run_simulation(self):
        """
        Run the index simulation for all months.
        """
        for month_idx in tqdm(range(len(self.dates)), desc="Simulating Hedge Fund Index"):
            if month_idx > 0:
                # Calculate index return for the month
                self.index_return.iloc[month_idx] = self._calculate_monthly_return(month_idx)
                
                # Update index level
                self.index_level.iloc[month_idx] = self.index_level.iloc[month_idx - 1] * (1 + self.index_return.iloc[month_idx])
            
            # Update end weights based on returns
            self._update_end_weights(month_idx)
            
            # Adjust weights for next month
            self._adjust_weights(month_idx)
            
            # Every quarter, randomly make some funds ineligible and others eligible
            if month_idx % 3 == 0 and month_idx > 0:
                # Make 2% of funds ineligible
                eligible_funds = self.fund_universe[self.fund_universe['eligible'] == True]
                num_to_change = max(1, int(0.02 * len(eligible_funds)))
                funds_to_make_ineligible = eligible_funds.sample(num_to_change)
                
                for idx, fund in funds_to_make_ineligible.iterrows():
                    self.fund_universe.loc[idx, 'eligible'] = False
                
                # Make 2% of ineligible funds eligible
                ineligible_funds = self.fund_universe[self.fund_universe['eligible'] == False]
                num_to_change = max(1, int(0.02 * len(ineligible_funds)))
                funds_to_make_eligible = ineligible_funds.sample(num_to_change)
                
                for idx, fund in funds_to_make_eligible.iterrows():
                    self.fund_universe.loc[idx, 'eligible'] = True
    
    def get_performance_metrics(self):
        """
        Calculate performance metrics for the index.
        
        Returns:
        --------
        metrics: dict
            Dictionary containing performance metrics
        """
        # Calculate metrics
        metrics = {}
        
        # Annualized return
        metrics['annualized_return'] = (
            (1 + self.index_return).prod() ** (12 / len(self.index_return)) - 1
        )
        
        # Annualized volatility
        metrics['annualized_volatility'] = self.index_return.std() * np.sqrt(12)
        
        # Sharpe ratio (assuming risk-free rate of 2%)
        risk_free_rate = 0.02
        metrics['sharpe_ratio'] = (
            (metrics['annualized_return'] - risk_free_rate) / metrics['annualized_volatility']
        )
        
        # Maximum drawdown
        cumulative_returns = (1 + self.index_return).cumprod()
        rolling_max = cumulative_returns.cummax()
        drawdowns = (cumulative_returns / rolling_max) - 1
        metrics['max_drawdown'] = drawdowns.min()
        
        # Sortino ratio
        negative_returns = self.index_return[self.index_return < 0]
        downside_deviation = negative_returns.std() * np.sqrt(12)
        metrics['sortino_ratio'] = (
            (metrics['annualized_return'] - risk_free_rate) / downside_deviation
            if downside_deviation > 0 else np.nan
        )
        
        return metrics
    
    def plot_index_performance(self):
        """
        Plot index level and returns.
        """
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot index level
        axes[0].plot(self.dates, self.index_level, 'b-', linewidth=2)
        axes[0].set_title('Hedge Fund Index Level')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Index Level')
        axes[0].grid(True)
        
        # Plot monthly returns
        axes[1].bar(self.dates, self.index_return * 100, color='g')
        axes[1].set_title('Hedge Fund Index Monthly Returns (%)')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Return (%)')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_strategy_weights(self):
        """
        Plot strategy weights over time.
        """
        plt.figure(figsize=(15, 8))
        
        # Plot strategy weights as stacked area chart
        self.strategy_weights.plot(kind='area', stacked=True, colormap='viridis')
        
        plt.title('Hedge Fund Index Strategy Weights')
        plt.xlabel('Date')
        plt.ylabel('Weight')
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()
    
    def plot_fund_concentration(self):
        """
        Plot fund concentration (number of funds and top 10 weights).
        """
        # Calculate number of funds in index each month
        num_funds = []
        for month_idx in range(len(self.dates)):
            count = sum(1 for fund_id in self.fund_universe['fund_id'] 
                       if self.fund_weights_begin.iloc[month_idx][fund_id] > 0)
            num_funds.append(count)
        
        # Calculate top 10 fund weights each month
        top10_weights = []
        for month_idx in range(len(self.dates)):
            weights = [w for w in self.fund_weights_begin.iloc[month_idx].values if w > 0]
            weights.sort(reverse=True)
            top10_sum = sum(weights[:10]) if len(weights) >= 10 else sum(weights)
            top10_weights.append(top10_sum)
        
        # Plot
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot number of funds
        axes[0].plot(self.dates, num_funds, 'b-', linewidth=2)
        axes[0].set_title('Number of Funds in Index')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel('Count')
        axes[0].grid(True)
        
        # Plot top 10 weights
        axes[1].plot(self.dates, [w * 100 for w in top10_weights], 'r-', linewidth=2)
        axes[1].set_title('Top 10 Fund Weights (%)')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel('Weight (%)')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.show()
    
    def plot_turnover(self):
        """
        Plot index turnover over time.
        """
        # Calculate monthly turnover
        turnover = []
        for month_idx in range(1, len(self.dates)):
            # Sum the absolute weight changes
            month_turnover = 0
            for fund_id in self.fund_universe['fund_id']:
                prev_weight = self.fund_weights_begin.iloc[month_idx - 1][fund_id]
                curr_weight = self.fund_weights_begin.iloc[month_idx][fund_id]
                month_turnover += abs(curr_weight - prev_weight)
            
            # Divide by 2 (since each sell is matched by a buy)
            month_turnover /= 2
            turnover.append(month_turnover)
        
        # Plot
        plt.figure(figsize=(15, 6))
        plt.plot(self.dates[1:], [t * 100 for t in turnover], 'g-', linewidth=2)
        plt.title('Monthly Index Turnover (%)')
        plt.xlabel('Date')
        plt.ylabel('Turnover (%)')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def plot_strategy_performance(self):
        """
        Plot performance by strategy.
        """
        # Calculate strategy returns
        strategy_returns = pd.DataFrame(index=self.dates[1:], columns=STRATEGY_NAMES)
        
        for month_idx in range(1, len(self.dates)):
            for strategy in STRATEGY_NAMES:
                strategy_return = 0
                strategy_weight = 0
                
                # Get all funds in this strategy
                for fund_id in self.fund_universe['fund_id']:
                    if (self.fund_universe.loc[self.fund_universe['fund_id'] == fund_id, 'strategy'].iloc[0] == strategy and
                        self.fund_weights_begin.iloc[month_idx - 1][fund_id] > 0):
                        
                        fund_weight = self.fund_weights_begin.iloc[month_idx - 1][fund_id]
                        fund_return = self.returns_data.iloc[month_idx - 1][fund_id]
                        
                        strategy_return += fund_weight * fund_return
                        strategy_weight += fund_weight
                
                if strategy_weight > 0:
                    strategy_returns.iloc[month_idx - 1][strategy] = strategy_return / strategy_weight
                else:
                    strategy_returns.iloc[month_idx - 1][strategy] = 0
        
        # Calculate cumulative returns
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # Plot
        plt.figure(figsize=(15, 8))
        cumulative_returns.plot(colormap='viridis', linewidth=2)
        plt.title('Cumulative Returns by Strategy')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.show()
    
    def compare_to_market(self):
        """
        Compare index performance to market.
        """
        # Calculate cumulative returns
        index_cum_return = (1 + self.index_return).cumprod()
        market_cum_return = (1 + self.returns_data['Market']).cumprod()
        
        # Plot
        plt.figure(figsize=(15, 8))
        plt.plot(self.dates, index_cum_return, 'b-', linewidth=2, label='Hedge Fund Index')
        plt.plot(self.dates, market_cum_return, 'r-', linewidth=2, label='Market')
        plt.title('Cumulative Returns: Hedge Fund Index vs Market')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        # Calculate correlation
        correlation = np.corrcoef(self.index_return[1:], self.returns_data['Market'][1:])[0, 1]
        print(f"Correlation with market: {correlation:.4f}")
        
        # Calculate beta
        market_var = np.var(self.returns_data['Market'][1:])
        covariance = np.cov(self.index_return[1:], self.returns_data['Market'][1:])[0, 1]
        beta = covariance / market_var
        print(f"Beta to market: {beta:.4f}")
    
    def generate_summary_report(self):
        """
        Generate a summary report of the hedge fund index.
        """
        # Get performance metrics
        metrics = self.get_performance_metrics()
        
        # Print summary
        print("=" * 50)
        print("HEDGE FUND INDEX SUMMARY REPORT")
        print("=" * 50)
        
        print("\nPerformance Metrics:")
        print(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
        print(f"Annualized Volatility: {metrics['annualized_volatility']*100:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']*100:.2f}%")
        
        # Calculate final strategy weights
        final_strategy_weights = self.strategy_weights.iloc[-1]
        
        print("\nFinal Strategy Weights:")
        for strategy in STRATEGY_NAMES:
            print(f"{strategy}: {final_strategy_weights[strategy]*100:.2f}%")
        
        # Calculate number of funds in index
        final_fund_count = sum(1 for fund_id in self.fund_universe['fund_id'] 
                              if self.fund_weights_begin.iloc[-1][fund_id] > 0)
        
        print(f"\nFinal Number of Funds in Index: {final_fund_count}")
        
        # Calculate average monthly turnover
        turnover = []
        for month_idx in range(1, len(self.dates)):
            month_turnover = 0
            for fund_id in self.fund_universe['fund_id']:
                prev_weight = self.fund_weights_begin.iloc[month_idx - 1][fund_id]
                curr_weight = self.fund_weights_begin.iloc[month_idx][fund_id]
                month_turnover += abs(curr_weight - prev_weight)
            month_turnover /= 2
            turnover.append(month_turnover)
        
        avg_turnover = np.mean(turnover)
        print(f"Average Monthly Turnover: {avg_turnover*100:.2f}%")
        
        # Calculate top 10 fund concentration
        final_weights = [w for w in self.fund_weights_begin.iloc[-1].values if w > 0]
        final_weights.sort(reverse=True)
        top10_concentration = sum(final_weights[:10]) if len(final_weights) >= 10 else sum(final_weights)
        
        print(f"Top 10 Fund Concentration: {top10_concentration*100:.2f}%")
        
        # Calculate correlation with market
        correlation = np.corrcoef(self.index_return[1:], self.returns_data['Market'][1:])[0, 1]
        print(f"Correlation with Market: {correlation:.4f}")
        
        print("=" * 50)


# Main execution
if __name__ == "__main__":
    print("Generating hedge fund universe...")
    fund_universe = generate_hedge_fund_universe()
    
    print("Simulating fund returns...")
    returns_data = simulate_fund_returns(fund_universe, NUM_MONTHS)
    
    print("Simulating AUM changes...")
    aum_data = simulate_aum_changes(fund_universe, returns_data)
    
    print("Calculating target strategy weights...")
    target_strategy_weights = calculate_strategy_target_weights(fund_universe, aum_data)
    
    print("Initializing and running hedge fund index simulation...")
    index = HedgeFundIndex(fund_universe, returns_data, aum_data, target_strategy_weights)
    index.run_simulation()
    
    # Generate visualizations and report
    index.plot_index_performance()
    index.plot_strategy_weights()
    index.plot_fund_concentration()
    index.plot_turnover()
    index.plot_strategy_performance()
    index.compare_to_market()
    index.generate_summary_report()