import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gurobipy as gp
from gurobipy import GRB
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)

class StockDataSimulator:
    """Simulate stock price data for testing the portfolio transition strategies"""
    
    def __init__(self, num_stocks=20, num_days=60, volatility=0.02, drift=0.001):
        """
        Initialize the stock data simulator
        
        Parameters:
        - num_stocks: Number of stocks to simulate
        - num_days: Number of days to simulate
        - volatility: Daily volatility of the stocks
        - drift: Daily drift (average return) of the stocks
        """
        self.num_stocks = num_stocks
        self.num_days = num_days
        self.volatility = volatility
        self.drift = drift
        
        # Stock names
        self.stock_names = [f"Stock_{i}" for i in range(num_stocks)]
        
    def generate_data(self):
        """
        Generate simulated stock price data
        
        Returns:
        - DataFrame with stock prices
        """
        # Initialize price matrix
        prices = np.zeros((self.num_days, self.num_stocks))
        
        # Set initial prices (random between $10 and $100)
        prices[0] = np.random.uniform(10, 100, self.num_stocks)
        
        # Generate price series with Geometric Brownian Motion
        for t in range(1, self.num_days):
            # Generate random shocks
            shocks = np.random.normal(
                self.drift, 
                self.volatility, 
                self.num_stocks
            )
            
            # Update prices
            prices[t] = prices[t-1] * np.exp(shocks)
        
        # Convert to DataFrame
        df = pd.DataFrame(prices, columns=self.stock_names)
        
        return df
    
    def generate_portfolio(self, prices, budget, target_stocks=None):
        """
        Generate a random portfolio with a given budget
        
        Parameters:
        - prices: DataFrame with stock prices
        - budget: Total budget for the portfolio
        - target_stocks: List of specific stocks to include (optional)
        
        Returns:
        - Dictionary with portfolio holdings and cash
        """
        # Get latest prices
        current_prices = prices.iloc[0].values
        
        # Initialize portfolio
        portfolio = {stock: 0 for stock in self.stock_names}
        
        # Available cash
        cash = budget
        
        # If target_stocks is provided, allocate budget to those stocks
        if target_stocks is not None:
            # Randomly distribute budget among target stocks
            weights = np.random.dirichlet(np.ones(len(target_stocks)))
            
            for i, stock in enumerate(target_stocks):
                # Calculate number of shares
                allocation = budget * weights[i]
                shares = int(allocation / current_prices[self.stock_names.index(stock)])
                
                # Update portfolio and cash
                portfolio[stock] = shares
                cash -= shares * current_prices[self.stock_names.index(stock)]
        else:
            # Randomly select stocks to include in portfolio
            num_stocks_in_portfolio = np.random.randint(5, self.num_stocks)
            selected_stocks = np.random.choice(self.stock_names, num_stocks_in_portfolio, replace=False)
            
            # Randomly distribute budget among selected stocks
            weights = np.random.dirichlet(np.ones(num_stocks_in_portfolio))
            
            for i, stock in enumerate(selected_stocks):
                # Calculate number of shares
                allocation = budget * weights[i]
                shares = int(allocation / current_prices[self.stock_names.index(stock)])
                
                # Update portfolio and cash
                portfolio[stock] = shares
                cash -= shares * current_prices[self.stock_names.index(stock)]
        
        return portfolio, cash

class PriceForecaster:
    """Simple price forecasting model for stock prices"""
    
    def __init__(self, lookback=10, forecast_horizon=30, error_rate=0.05):
        """
        Initialize the price forecaster
        
        Parameters:
        - lookback: Number of historical days to use for forecasting
        - forecast_horizon: Number of days to forecast
        - error_rate: Artificial error rate to add to forecasts (0.05 = 5% error)
        """
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.error_rate = error_rate
        self.models = {}
        
    def forecast(self, historical_data):
        """
        Generate price forecasts for all stocks
        
        Parameters:
        - historical_data: DataFrame with historical stock prices
        
        Returns:
        - DataFrame with forecasted stock prices
        """
        # Get the stocks
        stocks = historical_data.columns
        
        # Initialize DataFrame for forecasts
        forecasts = pd.DataFrame(index=range(self.forecast_horizon), columns=stocks)
        
        # Generate forecasts for each stock
        for stock in stocks:
            # Get historical data for this stock
            history = historical_data[stock].values
            
            if len(history) < self.lookback:
                # If we don't have enough history, use simple exponential growth forecast
                forecasts[stock] = self._simple_forecast(history)
            else:
                # Use linear regression on log prices
                forecasts[stock] = self._regression_forecast(history)
                
            # Add artificial error to make it more realistic
            error = np.random.normal(0, self.error_rate, self.forecast_horizon)
            forecasts[stock] *= (1 + error)
        
        return forecasts
    
    def _simple_forecast(self, history):
        """Simple exponential growth forecast"""
        last_price = history[-1]
        avg_return = np.mean(np.diff(history) / history[:-1])
        
        # Generate forecast
        forecast = np.zeros(self.forecast_horizon)
        forecast[0] = last_price * (1 + avg_return)
        
        for i in range(1, self.forecast_horizon):
            forecast[i] = forecast[i-1] * (1 + avg_return)
        
        return forecast
    
    def _regression_forecast(self, history):
        """Forecast using linear regression on log prices"""
        # Use last 'lookback' days
        history = history[-self.lookback:]
        
        # Transform to log prices
        log_prices = np.log(history)
        
        # Create features (time index)
        X = np.arange(self.lookback).reshape(-1, 1)
        
        # Fit linear regression
        model = LinearRegression()
        model.fit(X, log_prices)
        
        # Predict future log prices
        future_X = np.arange(self.lookback, self.lookback + self.forecast_horizon).reshape(-1, 1)
        future_log_prices = model.predict(future_X)
        
        # Transform back to prices
        forecast = np.exp(future_log_prices)
        
        return forecast

class PortfolioTransitionOptimizer:
    """Portfolio transition optimizer with different trading policies"""
    
    def __init__(self, stock_names, trading_fee=5.0):
        """
        Initialize the portfolio transition optimizer
        
        Parameters:
        - stock_names: List of stock names
        - trading_fee: Fixed fee per trade
        """
        self.stock_names = stock_names
        self.trading_fee = trading_fee
        self.num_stocks = len(stock_names)
        
    def naive_policy(self, current_portfolio, target_portfolio, prices, cash):
        """
        Implement naive trading policy (single period)
        
        Parameters:
        - current_portfolio: Dictionary with current holdings
        - target_portfolio: Dictionary with target holdings
        - prices: Current stock prices (Series)
        - cash: Available cash
        
        Returns:
        - Dictionary with trades to execute
        - Updated cash after trades
        - Dictionary with optimizer metrics
        """
        start_time = time.time()
        
        # Convert portfolios to arrays for easier manipulation
        current_holdings = np.array([current_portfolio[stock] for stock in self.stock_names])
        target_holdings = np.array([target_portfolio[stock] for stock in self.stock_names])
        price_array = np.array([prices[stock] for stock in self.stock_names])
        
        # Create optimization model
        model = gp.Model("Naive_Portfolio_Transition")
        model.setParam('OutputFlag', 0)  # Suppress output
        
        # Decision variables
        z_plus = model.addVars(self.num_stocks, vtype=GRB.INTEGER, name="z_plus", lb=0)
        z_minus = model.addVars(self.num_stocks, vtype=GRB.INTEGER, name="z_minus", lb=0)
        w_plus = model.addVars(self.num_stocks, vtype=GRB.BINARY, name="w_plus")
        w_minus = model.addVars(self.num_stocks, vtype=GRB.BINARY, name="w_minus")
        
        # Final portfolio and cash variables
        p = model.addVars(self.num_stocks, vtype=GRB.INTEGER, name="p", lb=0)
        c = model.addVar(vtype=GRB.CONTINUOUS, name="c", lb=0)
        
        # Big M value - upper bound on number of shares that can be bought
        M = int(np.ceil((cash + np.sum(current_holdings * price_array)) / np.min(price_array)))
        
        # Constraints
        
        # Final portfolio calculation
        for i in range(self.num_stocks):
            model.addConstr(p[i] == current_holdings[i] + z_plus[i] - z_minus[i])
        
        # Cash calculation
        model.addConstr(
            c == cash - gp.quicksum(price_array[i] * (z_plus[i] - z_minus[i]) for i in range(self.num_stocks)) - 
            self.trading_fee * gp.quicksum(w_plus[i] + w_minus[i] for i in range(self.num_stocks))
        )
        
        # Linking z and w variables (buy)
        for i in range(self.num_stocks):
            model.addConstr(z_plus[i] <= M * w_plus[i])
        
        # Linking z and w variables (sell)
        for i in range(self.num_stocks):
            model.addConstr(z_minus[i] <= M * w_minus[i])
        
        # Target portfolio constraint
        for i in range(self.num_stocks):
            model.addConstr(p[i] >= target_holdings[i])
        
        # Objective: Maximize final portfolio value minus trading costs
        model.setObjective(
            gp.quicksum(price_array[i] * p[i] for i in range(self.num_stocks)) + c -
            self.trading_fee * gp.quicksum(w_plus[i] + w_minus[i] for i in range(self.num_stocks)), 
            GRB.MAXIMIZE
        )
        
        # Solve the model
        model.optimize()
        
        # Extract results
        trades = {}
        for i in range(self.num_stocks):
            stock = self.stock_names[i]
            buy = z_plus[i].X
            sell = z_minus[i].X
            trades[stock] = int(buy - sell)
        
        # Calculate updated cash
        trading_costs = self.trading_fee * sum(1 for stock in trades if trades[stock] != 0)
        updated_cash = cash
        for i, stock in enumerate(self.stock_names):
            updated_cash -= trades[stock] * price_array[i]
        updated_cash -= trading_costs
        
        # Metrics
        solve_time = time.time() - start_time
        num_trades = sum(1 for stock in trades if trades[stock] != 0)
        trading_cost = self.trading_fee * num_trades
        
        metrics = {
            "solve_time": solve_time,
            "num_trades": num_trades,
            "trading_cost": trading_cost,
            "obj_value": model.ObjVal if model.Status == GRB.OPTIMAL else None
        }
        
        return trades, updated_cash, metrics
    
    def directional_policy(self, current_portfolio, target_portfolio, prices, cash, forecast_prices, 
                          lookahead=5, current_time=0):
        """
        Implement directional trading policy
        
        Parameters:
        - current_portfolio: Dictionary with current holdings
        - target_portfolio: Dictionary with target holdings
        - prices: Current stock prices (Series)
        - cash: Available cash
        - forecast_prices: DataFrame with forecasted prices
        - lookahead: Number of days to look ahead
        - current_time: Current time period
        
        Returns:
        - Dictionary with trades to execute
        - Updated cash after trades
        - Dictionary with optimizer metrics
        """
        start_time = time.time()
        
        # Convert portfolios to arrays for easier manipulation
        current_holdings = np.array([current_portfolio[stock] for stock in self.stock_names])
        target_holdings = np.array([target_portfolio[stock] for stock in self.stock_names])
        
        # Determine buy and sell sets
        A_plus = [i for i, stock in enumerate(self.stock_names) 
                 if current_portfolio[stock] < target_portfolio[stock]]
        A_minus = [i for i, stock in enumerate(self.stock_names) 
                  if current_portfolio[stock] >= target_portfolio[stock]]
        
        # Extract price data
        price_array = np.array([prices[stock] for stock in self.stock_names])
        
        # Prepare forecast prices for the lookahead period
        forecast_array = np.zeros((lookahead, self.num_stocks))
        for t in range(lookahead):
            if t == 0:
                # Use current prices for the first step
                forecast_array[t] = price_array
            else:
                # Use forecasted prices for future steps
                forecast_idx = current_time + t
                if forecast_idx < len(forecast_prices):
                    forecast_array[t] = [forecast_prices.iloc[forecast_idx][stock] for stock in self.stock_names]
                else:
                    # Use last forecast if we're beyond the forecast horizon
                    forecast_array[t] = forecast_array[t-1]
        
        # Create optimization model
        model = gp.Model("Directional_Portfolio_Transition")
        model.setParam('OutputFlag', 0)  # Suppress output
        
        # Decision variables - for each time step in the lookahead
        z_plus = {}
        z_minus = {}
        w_plus = {}
        w_minus = {}
        p = {}
        c = {}
        
        for t in range(lookahead):
            z_plus[t] = model.addVars(self.num_stocks, vtype=GRB.INTEGER, name=f"z_plus_{t}", lb=0)
            z_minus[t] = model.addVars(self.num_stocks, vtype=GRB.INTEGER, name=f"z_minus_{t}", lb=0)
            w_plus[t] = model.addVars(self.num_stocks, vtype=GRB.BINARY, name=f"w_plus_{t}")
            w_minus[t] = model.addVars(self.num_stocks, vtype=GRB.BINARY, name=f"w_minus_{t}")
            p[t] = model.addVars(self.num_stocks, vtype=GRB.INTEGER, name=f"p_{t}", lb=0)
            c[t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"c_{t}", lb=0)
        
        # Upper bounds for the number of shares that can be bought
        U = {}
        M = {}
        
        U[0] = cash + np.sum(current_holdings * price_array)
        for t in range(1, lookahead):
            # Calculate maximum possible value growth (perfect foresight bound)
            max_return = max(forecast_array[t] / forecast_array[t-1])
            U[t] = U[t-1] * max_return
        
        for t in range(lookahead):
            M[t] = int(np.ceil(U[t] / np.min(forecast_array[t])))
        
        # Constraints
        
        # Initial portfolio
        for i in range(self.num_stocks):
            model.addConstr(p[0][i] == current_holdings[i] + z_plus[0][i] - z_minus[0][i])
        
        # Initial cash
        model.addConstr(
            c[0] == cash - gp.quicksum(price_array[i] * (z_plus[0][i] - z_minus[0][i]) for i in range(self.num_stocks)) - 
            self.trading_fee * gp.quicksum(w_plus[0][i] + w_minus[0][i] for i in range(self.num_stocks))
        )
        
        # Portfolio evolution for t > 0
        for t in range(1, lookahead):
            for i in range(self.num_stocks):
                model.addConstr(p[t][i] == p[t-1][i] + z_plus[t][i] - z_minus[t][i])
        
        # Cash evolution for t > 0
        for t in range(1, lookahead):
            model.addConstr(
                c[t] == c[t-1] - gp.quicksum(forecast_array[t][i] * (z_plus[t][i] - z_minus[t][i]) for i in range(self.num_stocks)) - 
                self.trading_fee * gp.quicksum(w_plus[t][i] + w_minus[t][i] for i in range(self.num_stocks))
            )
        
        # Linking z and w variables (buy)
        for t in range(lookahead):
            for i in range(self.num_stocks):
                model.addConstr(z_plus[t][i] <= M[t] * w_plus[t][i])
        
        # Linking z and w variables (sell)
        for t in range(lookahead):
            for i in range(self.num_stocks):
                model.addConstr(z_minus[t][i] <= M[t] * w_minus[t][i])
        
        # Target portfolio constraint at the end of the horizon
        for i in range(self.num_stocks):
            model.addConstr(p[lookahead-1][i] >= target_holdings[i])
        
        # Directional constraints
        for t in range(lookahead):
            for i in A_minus:
                model.addConstr(z_plus[t][i] == 0)  # Cannot buy stocks in the sell set
            
            for i in A_plus:
                model.addConstr(z_minus[t][i] == 0)  # Cannot sell stocks in the buy set
        
        # Objective: Maximize final portfolio value minus trading costs
        model.setObjective(
            gp.quicksum(forecast_array[lookahead-1][i] * p[lookahead-1][i] for i in range(self.num_stocks)) + 
            c[lookahead-1], 
            GRB.MAXIMIZE
        )
        
        # Solve the model
        model.optimize()
        
        # Extract results for the current time step (t=0)
        trades = {}
        for i in range(self.num_stocks):
            stock = self.stock_names[i]
            buy = z_plus[0][i].X if model.Status == GRB.OPTIMAL else 0
            sell = z_minus[0][i].X if model.Status == GRB.OPTIMAL else 0
            trades[stock] = int(buy - sell)
        
        # Calculate updated cash
        trading_costs = self.trading_fee * sum(1 for stock in trades if trades[stock] != 0)
        updated_cash = cash
        for i, stock in enumerate(self.stock_names):
            updated_cash -= trades[stock] * price_array[i]
        updated_cash -= trading_costs
        
        # Metrics
        solve_time = time.time() - start_time
        num_trades = sum(1 for stock in trades if trades[stock] != 0)
        trading_cost = self.trading_fee * num_trades
        
        metrics = {
            "solve_time": solve_time,
            "num_trades": num_trades,
            "trading_cost": trading_cost,
            "obj_value": model.ObjVal if model.Status == GRB.OPTIMAL else None
        }
        
        return trades, updated_cash, metrics
    
    def directional_penalty_policy(self, current_portfolio, target_portfolio, prices, cash, forecast_prices, 
                                  penalty_factor=0.5, lookahead=5, current_time=0):
        """
        Implement directional penalty trading policy
        
        Parameters:
        - current_portfolio: Dictionary with current holdings
        - target_portfolio: Dictionary with target holdings
        - prices: Current stock prices (Series)
        - cash: Available cash
        - forecast_prices: DataFrame with forecasted prices
        - penalty_factor: Penalty factor for trades not moving towards target (between 0 and 1)
        - lookahead: Number of days to look ahead
        - current_time: Current time period
        
        Returns:
        - Dictionary with trades to execute
        - Updated cash after trades
        - Dictionary with optimizer metrics
        """
        start_time = time.time()
        
        # Convert portfolios to arrays for easier manipulation
        current_holdings = np.array([current_portfolio[stock] for stock in self.stock_names])
        target_holdings = np.array([target_portfolio[stock] for stock in self.stock_names])
        
        # Determine buy and sell sets
        A_plus = [i for i, stock in enumerate(self.stock_names) 
                 if current_portfolio[stock] < target_portfolio[stock]]
        A_minus = [i for i, stock in enumerate(self.stock_names) 
                  if current_portfolio[stock] >= target_portfolio[stock]]
        
        # Extract price data
        price_array = np.array([prices[stock] for stock in self.stock_names])
        
        # Prepare forecast prices for the lookahead period
        forecast_array = np.zeros((lookahead, self.num_stocks))
        for t in range(lookahead):
            if t == 0:
                # Use current prices for the first step
                forecast_array[t] = price_array
            else:
                # Use forecasted prices for future steps
                forecast_idx = current_time + t
                if forecast_idx < len(forecast_prices):
                    forecast_array[t] = [forecast_prices.iloc[forecast_idx][stock] for stock in self.stock_names]
                else:
                    # Use last forecast if we're beyond the forecast horizon
                    forecast_array[t] = forecast_array[t-1]
        
        # Create optimization model
        model = gp.Model("Directional_Penalty_Portfolio_Transition")
        model.setParam('OutputFlag', 0)  # Suppress output
        
        # Decision variables - for each time step in the lookahead
        z_plus = {}
        z_minus = {}
        w_plus = {}
        w_minus = {}
        p = {}
        c = {}
        
        for t in range(lookahead):
            z_plus[t] = model.addVars(self.num_stocks, vtype=GRB.INTEGER, name=f"z_plus_{t}", lb=0)
            z_minus[t] = model.addVars(self.num_stocks, vtype=GRB.INTEGER, name=f"z_minus_{t}", lb=0)
            w_plus[t] = model.addVars(self.num_stocks, vtype=GRB.BINARY, name=f"w_plus_{t}")
            w_minus[t] = model.addVars(self.num_stocks, vtype=GRB.BINARY, name=f"w_minus_{t}")
            p[t] = model.addVars(self.num_stocks, vtype=GRB.INTEGER, name=f"p_{t}", lb=0)
            c[t] = model.addVar(vtype=GRB.CONTINUOUS, name=f"c_{t}", lb=0)
        
        # Upper bounds for the number of shares that can be bought
        U = {}
        M = {}
        
        U[0] = cash + np.sum(current_holdings * price_array)
        for t in range(1, lookahead):
            # Calculate maximum possible value growth (perfect foresight bound)
            max_return = max(forecast_array[t] / forecast_array[t-1])
            U[t] = U[t-1] * max_return
        
        for t in range(lookahead):
            M[t] = int(np.ceil(U[t] / np.min(forecast_array[t])))
        
        # Constraints
        
        # Initial portfolio
        for i in range(self.num_stocks):
            model.addConstr(p[0][i] == current_holdings[i] + z_plus[0][i] - z_minus[0][i])
        
        # Initial cash
        model.addConstr(
            c[0] == cash - gp.quicksum(price_array[i] * (z_plus[0][i] - z_minus[0][i]) for i in range(self.num_stocks)) - 
            self.trading_fee * gp.quicksum(w_plus[0][i] + w_minus[0][i] for i in range(self.num_stocks))
        )
        
        # Portfolio evolution for t > 0
        for t in range(1, lookahead):
            for i in range(self.num_stocks):
                model.addConstr(p[t][i] == p[t-1][i] + z_plus[t][i] - z_minus[t][i])
        
        # Cash evolution for t > 0
        for t in range(1, lookahead):
            model.addConstr(
                c[t] == c[t-1] - gp.quicksum(forecast_array[t][i] * (z_plus[t][i] - z_minus[t][i]) for i in range(self.num_stocks)) - 
                self.trading_fee * gp.quicksum(w_plus[t][i] + w_minus[t][i] for i in range(self.num_stocks))
            )
        
        # Linking z and w variables (buy)
        for t in range(lookahead):
            for i in range(self.num_stocks):
                model.addConstr(z_plus[t][i] <= M[t] * w_plus[t][i])
        
        # Linking z and w variables (sell)
        for t in range(lookahead):
            for i in range(self.num_stocks):
                model.addConstr(z_minus[t][i] <= M[t] * w_minus[t][i])
        
        # Target portfolio constraint at the end of the horizon
        for i in range(self.num_stocks):
            model.addConstr(p[lookahead-1][i] >= target_holdings[i])
        
        # No simultaneous buy and sell of the same asset
        for t in range(lookahead):
            for i in range(self.num_stocks):
                model.addConstr(w_plus[t][i] + w_minus[t][i] <= 1)
        
        # Objective: Maximize final portfolio value minus trading costs and penalties
        objective = gp.quicksum(forecast_array[lookahead-1][i] * p[lookahead-1][i] for i in range(self.num_stocks)) + c[lookahead-1]
        
        # Add penalty for trades not moving towards target
        for t in range(lookahead):
            # Penalty for buying stocks in the sell set
            for i in A_minus:
                objective -= penalty_factor * forecast_array[t][i] * z_plus[t][i]
            
            # Penalty for selling stocks in the buy set
            for i in A_plus:
                objective -= penalty_factor * forecast_array[t][i] * z_minus[t][i]
        
        model.setObjective(objective, GRB.MAXIMIZE)
        
        # Solve the model
        model.optimize()
        
        # Extract results for the current time step (t=0)
        trades = {}
        for i in range(self.num_stocks):
            stock = self.stock_names[i]
            buy = z_plus[0][i].X if model.Status == GRB.OPTIMAL else 0
            sell = z_minus[0][i].X if model.Status == GRB.OPTIMAL else 0
            trades[stock] = int(buy - sell)
        
        # Calculate updated cash
        trading_costs = self.trading_fee * sum(1 for stock in trades if trades[stock] != 0)
        updated_cash = cash
        for i, stock in enumerate(self.stock_names):
            updated_cash -= trades[stock] * price_array[i]
        updated_cash -= trading_costs
        
        # Metrics
        solve_time = time.time() - start_time
        num_trades = sum(1 for stock in trades if trades[stock] != 0)
        trading_cost = self.trading_fee * num_trades
        
        metrics = {
            "solve_time": solve_time,
            "num_trades": num_trades,
            "trading_cost": trading_cost,
            "obj_value": model.ObjVal if model.Status == GRB.OPTIMAL else None
        }
        
        return trades, updated_cash, metrics

class PortfolioTransitionSimulator:
    """Simulator for testing portfolio transition strategies"""
    
    def __init__(self, stock_data, optimizer, forecaster, initial_portfolio, target_portfolio, initial_cash, 
                trading_horizon=30, lookahead=5):
        """
        Initialize the portfolio transition simulator
        
        Parameters:
        - stock_data: DataFrame with stock price data
        - optimizer: PortfolioTransitionOptimizer instance
        - forecaster: PriceForecaster instance
        - initial_portfolio: Dictionary with initial holdings
        - target_portfolio: Dictionary with target holdings
        - initial_cash: Initial cash
        - trading_horizon: Number of days for transition
        - lookahead: Number of days to look ahead for multi-period strategies
        """
        self.stock_data = stock_data
        self.optimizer = optimizer
        self.forecaster = forecaster
        self.initial_portfolio = initial_portfolio
        self.target_portfolio = target_portfolio
        self.initial_cash = initial_cash
        self.trading_horizon = min(trading_horizon, len(stock_data))
        self.lookahead = min(lookahead, len(stock_data))
        self.stock_names = list(stock_data.columns)
    
    def run_naive_strategy(self):
        """
        Run naive (single-period) trading strategy
        
        Returns:
        - Dictionary with simulation results
        """
        # Initialize
        current_portfolio = self.initial_portfolio.copy()
        cash = self.initial_cash
        
        # Results tracking
        portfolio_values = []
        trades_history = []
        metrics_history = []
        
        # Current prices
        prices = self.stock_data.iloc[0]
        
        # Calculate initial portfolio value
        initial_value = cash + sum(current_portfolio[stock] * prices[stock] for stock in self.stock_names)
        portfolio_values.append(initial_value)
        
        # Execute trades at time 0
        trades, cash, metrics = self.optimizer.naive_policy(current_portfolio, self.target_portfolio, prices, cash)
        
        # Update portfolio
        for stock, amount in trades.items():
            current_portfolio[stock] += amount
        
        trades_history.append(trades)
        metrics_history.append(metrics)
        
        # Track portfolio value over time
        for t in range(1, self.trading_horizon):
            prices = self.stock_data.iloc[t]
            value = cash + sum(current_portfolio[stock] * prices[stock] for stock in self.stock_names)
            portfolio_values.append(value)
        
        # Final portfolio check
        final_prices = self.stock_data.iloc[self.trading_horizon - 1]
        final_value = cash + sum(current_portfolio[stock] * final_prices[stock] for stock in self.stock_names)
        
        # Calculate summary metrics
        total_trades = sum(metrics['num_trades'] for metrics in metrics_history)
        total_trading_cost = sum(metrics['trading_cost'] for metrics in metrics_history)
        percent_change = (final_value - initial_value) / initial_value * 100
        
        results = {
            "strategy": "Naive",
            "portfolio_values": portfolio_values,
            "trades_history": trades_history,
            "metrics_history": metrics_history,
            "final_portfolio": current_portfolio,
            "final_cash": cash,
            "initial_value": initial_value,
            "final_value": final_value,
            "percent_change": percent_change,
            "total_trades": total_trades,
            "total_trading_cost": total_trading_cost
        }
        
        return results
    
    def run_directional_strategy(self):
        """
        Run directional trading strategy
        
        Returns:
        - Dictionary with simulation results
        """
        # Initialize
        current_portfolio = self.initial_portfolio.copy()
        cash = self.initial_cash
        
        # Results tracking
        portfolio_values = []
        trades_history = []
        metrics_history = []
        
        # Current prices
        prices = self.stock_data.iloc[0]
        
        # Calculate initial portfolio value
        initial_value = cash + sum(current_portfolio[stock] * prices[stock] for stock in self.stock_names)
        portfolio_values.append(initial_value)
        
        # Run multi-period transition
        for t in range(self.trading_horizon):
            # Get current prices
            prices = self.stock_data.iloc[t]
            
            # Generate forecasts
            historical_window = min(t + 1, 10)  # Use up to 10 days of history
            historical_data = self.stock_data.iloc[max(0, t+1-historical_window):t+1]
            forecast_prices = self.forecaster.forecast(historical_data)
            
            # Execute trades
            trades, cash, metrics = self.optimizer.directional_policy(
                current_portfolio, 
                self.target_portfolio, 
                prices, 
                cash, 
                forecast_prices,
                lookahead=min(self.lookahead, self.trading_horizon - t),
                current_time=t
            )
            
            # Update portfolio
            for stock, amount in trades.items():
                current_portfolio[stock] += amount
            
            trades_history.append(trades)
            metrics_history.append(metrics)
            
            # Track portfolio value
            value = cash + sum(current_portfolio[stock] * prices[stock] for stock in self.stock_names)
            portfolio_values.append(value)
            
            # Check if target portfolio has been reached
            target_reached = all(current_portfolio[stock] >= self.target_portfolio[stock] for stock in self.stock_names)
            if target_reached and t > 0:  # Skip the first day to ensure we have at least one transition
                break
        
        # Continue tracking portfolio value until the end of the horizon
        for t in range(len(portfolio_values), self.trading_horizon):
            prices = self.stock_data.iloc[t]
            value = cash + sum(current_portfolio[stock] * prices[stock] for stock in self.stock_names)
            portfolio_values.append(value)
        
        # Final portfolio check
        final_prices = self.stock_data.iloc[self.trading_horizon - 1]
        final_value = cash + sum(current_portfolio[stock] * final_prices[stock] for stock in self.stock_names)
        
        # Calculate summary metrics
        total_trades = sum(metrics['num_trades'] for metrics in metrics_history)
        total_trading_cost = sum(metrics['trading_cost'] for metrics in metrics_history)
        percent_change = (final_value - initial_value) / initial_value * 100
        
        results = {
            "strategy": "Directional",
            "portfolio_values": portfolio_values,
            "trades_history": trades_history,
            "metrics_history": metrics_history,
            "final_portfolio": current_portfolio,
            "final_cash": cash,
            "initial_value": initial_value,
            "final_value": final_value,
            "percent_change": percent_change,
            "total_trades": total_trades,
            "total_trading_cost": total_trading_cost
        }
        
        return results
    
    def run_directional_penalty_strategy(self, penalty_factor=0.5):
        """
        Run directional penalty trading strategy
        
        Parameters:
        - penalty_factor: Penalty factor for trades not moving towards target
        
        Returns:
        - Dictionary with simulation results
        """
        # Initialize
        current_portfolio = self.initial_portfolio.copy()
        cash = self.initial_cash
        
        # Results tracking
        portfolio_values = []
        trades_history = []
        metrics_history = []
        
        # Current prices
        prices = self.stock_data.iloc[0]
        
        # Calculate initial portfolio value
        initial_value = cash + sum(current_portfolio[stock] * prices[stock] for stock in self.stock_names)
        portfolio_values.append(initial_value)
        
        # Run multi-period transition
        for t in range(self.trading_horizon):
            # Get current prices
            prices = self.stock_data.iloc[t]
            
            # Generate forecasts
            historical_window = min(t + 1, 10)  # Use up to 10 days of history
            historical_data = self.stock_data.iloc[max(0, t+1-historical_window):t+1]
            forecast_prices = self.forecaster.forecast(historical_data)
            
            # Execute trades
            trades, cash, metrics = self.optimizer.directional_penalty_policy(
                current_portfolio, 
                self.target_portfolio, 
                prices, 
                cash, 
                forecast_prices,
                penalty_factor=penalty_factor,
                lookahead=min(self.lookahead, self.trading_horizon - t),
                current_time=t
            )
            
            # Update portfolio
            for stock, amount in trades.items():
                current_portfolio[stock] += amount
            
            trades_history.append(trades)
            metrics_history.append(metrics)
            
            # Track portfolio value
            value = cash + sum(current_portfolio[stock] * prices[stock] for stock in self.stock_names)
            portfolio_values.append(value)
            
            # Check if target portfolio has been reached
            target_reached = all(current_portfolio[stock] >= self.target_portfolio[stock] for stock in self.stock_names)
            if target_reached and t > 0:  # Skip the first day to ensure we have at least one transition
                break
        
        # Continue tracking portfolio value until the end of the horizon
        for t in range(len(portfolio_values), self.trading_horizon):
            prices = self.stock_data.iloc[t]
            value = cash + sum(current_portfolio[stock] * prices[stock] for stock in self.stock_names)
            portfolio_values.append(value)
        
        # Final portfolio check
        final_prices = self.stock_data.iloc[self.trading_horizon - 1]
        final_value = cash + sum(current_portfolio[stock] * final_prices[stock] for stock in self.stock_names)
        
        # Calculate summary metrics
        total_trades = sum(metrics['num_trades'] for metrics in metrics_history)
        total_trading_cost = sum(metrics['trading_cost'] for metrics in metrics_history)
        percent_change = (final_value - initial_value) / initial_value * 100
        
        results = {
            "strategy": f"DirP_{penalty_factor*100:.1f}",
            "portfolio_values": portfolio_values,
            "trades_history": trades_history,
            "metrics_history": metrics_history,
            "final_portfolio": current_portfolio,
            "final_cash": cash,
            "initial_value": initial_value,
            "final_value": final_value,
            "percent_change": percent_change,
            "total_trades": total_trades,
            "total_trading_cost": total_trading_cost
        }
        
        return results
    
def generate_report(analysis):
    """
    Generate a comprehensive report on the findings
    
    Parameters:
    - analysis: Analysis output from analyze_results
    
    Returns:
    - None (prints report)
    """
    summary = analysis["summary"]
    win_loss_tie = analysis["win_loss_tie"]
    results_df = analysis["results_df"]
    
    print("=" * 80)
    print("                   PORTFOLIO TRANSITION STRATEGY ANALYSIS                   ")
    print("=" * 80)
    print("\n1. PERFORMANCE SUMMARY\n")
    
    # Format summary statistics
    percent_change = summary["percent_change"]
    print("Percent Change in Portfolio Value:")
    print("-" * 60)
    print(f"{'Strategy':<15} {'Mean':<10} {'Std Dev':<10} {'Median':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)
    
    for strategy in results_df["strategy"].unique():
        mean = percent_change.loc[strategy, "mean"]
        std = percent_change.loc[strategy, "std"]
        median = percent_change.loc[strategy, "median"]
        min_val = percent_change.loc[strategy, "min"]
        max_val = percent_change.loc[strategy, "max"]
        
        print(f"{strategy:<15} {mean:10.2f} {std:10.2f} {median:10.2f} {min_val:10.2f} {max_val:10.2f}")
    
    print("\nNumber of Trades Executed:")
    print("-" * 60)
    print(f"{'Strategy':<15} {'Mean':<10} {'Std Dev':<10} {'Median':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)
    
    trades = summary["total_trades"]
    for strategy in results_df["strategy"].unique():
        mean = trades.loc[strategy, "mean"]
        std = trades.loc[strategy, "std"]
        median = trades.loc[strategy, "median"]
        min_val = trades.loc[strategy, "min"]
        max_val = trades.loc[strategy, "max"]
        
        print(f"{strategy:<15} {mean:10.2f} {std:10.2f} {median:10.2f} {min_val:10.2f} {max_val:10.2f}")
    
    print("\nTrading Costs Incurred:")
    print("-" * 60)
    print(f"{'Strategy':<15} {'Mean':<10} {'Std Dev':<10} {'Median':<10} {'Min':<10} {'Max':<10}")
    print("-" * 60)
    
    costs = summary["total_trading_cost"]
    for strategy in results_df["strategy"].unique():
        mean = costs.loc[strategy, "mean"]
        std = costs.loc[strategy, "std"]
        median = costs.loc[strategy, "median"]
        min_val = costs.loc[strategy, "min"]
        max_val = costs.loc[strategy, "max"]
        
        print(f"{strategy:<15} {mean:10.2f} {std:10.2f} {median:10.2f} {min_val:10.2f} {max_val:10.2f}")
    
    print("\n\n2. WIN/LOSS/TIE ANALYSIS\n")
    print("-" * 60)
    print(f"{'Strategy':<15} {'Wins':<10} {'Losses':<10} {'Ties':<10} {'Win Rate':<10}")
    print("-" * 60)
    
    for strategy, (wins, losses, ties) in win_loss_tie.items():
        win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0
        print(f"{strategy:<15} {wins:<10} {losses:<10} {ties:<10} {win_rate:10.2f}%")
    
    print("\n\n3. KEY FINDINGS\n")
    
    # Identify best strategy for each metric
    best_return = results_df.groupby("strategy")["percent_change"].mean().idxmax()
    lowest_trades = results_df.groupby("strategy")["total_trades"].mean().idxmin()
    lowest_cost = results_df.groupby("strategy")["total_trading_cost"].mean().idxmin()
    
    print(f"1. The {best_return} strategy achieved the highest average return ({percent_change.loc[best_return, 'mean']:.2f}%).")
    print(f"2. The {lowest_trades} strategy required the fewest trades ({trades.loc[lowest_trades, 'mean']:.2f} on average).")
    print(f"3. The {lowest_cost} strategy incurred the lowest trading costs (${costs.loc[lowest_cost, 'mean']:.2f} on average).")
    
    # Compare directional penalty strategies
    dir_penalties = [s for s in results_df["strategy"].unique() if s.startswith("DirP")]
    dir_penalties.sort(key=lambda x: float(x.split("_")[1]) if "_" in x else 0)
    
    if len(dir_penalties) > 1:
        print("\n4. Directional Penalty Analysis:")
        print(f"   - As the penalty factor increases from {dir_penalties[0]} to {dir_penalties[-1]}:")
        
        returns_trend = results_df.groupby("strategy")["percent_change"].mean()[dir_penalties].values
        if returns_trend[0] > returns_trend[-1]:
            print(f"     * Average returns decrease from {returns_trend[0]:.2f}% to {returns_trend[-1]:.2f}%")
        else:
            print(f"     * Average returns increase from {returns_trend[0]:.2f}% to {returns_trend[-1]:.2f}%")
        
        trades_trend = results_df.groupby("strategy")["total_trades"].mean()[dir_penalties].values
        if trades_trend[0] > trades_trend[-1]:
            print(f"     * Average number of trades decrease from {trades_trend[0]:.2f} to {trades_trend[-1]:.2f}")
        else:
            print(f"     * Average number of trades increase from {trades_trend[0]:.2f} to {trades_trend[-1]:.2f}")
    
    print("\n\n4. RECOMMENDATIONS\n")
    
    # Make recommendations based on analysis
    print("Based on the analysis, we recommend:")
    
    # Best overall strategy
    best_overall = None
    max_score = -float('inf')
    
    for strategy in results_df["strategy"].unique():
        # Simple scoring: normalize and combine return, trades, and costs
        return_score = (percent_change.loc[strategy, "mean"] - percent_change["mean"].min()) / (percent_change["mean"].max() - percent_change["mean"].min() + 1e-10)
        trade_score = 1 - (trades.loc[strategy, "mean"] - trades["mean"].min()) / (trades["mean"].max() - trades["mean"].min() + 1e-10)
        cost_score = 1 - (costs.loc[strategy, "mean"] - costs["mean"].min()) / (costs["mean"].max() - costs["mean"].min() + 1e-10)
        
        score = return_score * 0.5 + trade_score * 0.25 + cost_score * 0.25
        
        if score > max_score:
            max_score = score
            best_overall = strategy
    
    print(f"1. For most investors, the {best_overall} strategy offers the best balance of returns and efficiency.")
    
    # For risk-averse investors
    risk_averse = results_df.groupby("strategy")["percent_change"].std().idxmin()
    print(f"2. For risk-averse investors, the {risk_averse} strategy provides the most stable performance.")
    
    # For cost-sensitive investors
    print(f"3. For cost-sensitive investors with smaller portfolios, the {lowest_cost} strategy minimizes trading fees.")
    
    # For aggressive investors
    aggressive = results_df.groupby("strategy")["percent_change"].max().idxmax()
    print(f"4. For aggressive investors seeking maximum potential returns, the {aggressive} strategy has shown the highest upside.")
    
    print("\n5. LIMITATIONS AND FUTURE WORK\n")
    
    print("This analysis has several limitations:")
    print("1. Limited simulation scenarios - results may vary with more extensive testing.")
    print("2. Simplified price forecasting model - real-world forecasting errors may be larger.")
    print("3. Fixed trading fees - real brokers may have more complex fee structures.")
    print("4. No consideration of market impact or liquidity constraints.")
    
    print("\nFuture work could include:")
    print("1. Testing with real market data over longer periods.")
    print("2. Incorporating more sophisticated forecasting models.")
    print("3. Exploring the column generation algorithm mentioned in the paper for larger portfolios.")
    print("4. Adding risk constraints to the optimization models.")
    
    print("\n" + "=" * 80)

# Generate comprehensive report
generate_report(analysis)


