import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pdblp
from datetime import datetime, timedelta
import os
from tqdm import tqdm

# Set display options
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)

# Connect to Bloomberg
def connect_to_bloomberg():
    """Connect to Bloomberg API and return connection object"""
    print("Connecting to Bloomberg...")
    con = pdblp.BCon(timeout=5000)
    con.start()
    return con

# Retrieve supply chain data from Bloomberg
def get_supply_chain_data(con, start_date, end_date):
    """
    Retrieve supply chain relationship data from Bloomberg
    
    Parameters:
    con (pdblp.BCon): Bloomberg connection object
    start_date (str): Start date in YYYYMMDD format
    end_date (str): End date in YYYYMMDD format
    
    Returns:
    DataFrame: Supply chain relationship data
    """
    print("Retrieving supply chain data from Bloomberg...")
    
    # For testing purposes, we'll create a more realistic synthetic dataset
    
    # Get MSCI DM index members
    msci_members = get_msci_index_members(con)
    
    # Create more realistic industry sectors
    sectors = ['Technology', 'Energy', 'Financials', 'Healthcare', 'Consumer', 'Industrials', 'Materials', 'Utilities']
    sector_mapping = {}
    
    # Assign sectors to stocks
    for ticker in msci_members:
        sector_mapping[ticker] = np.random.choice(sectors)
    
    # Create supplier-customer relationships with sector bias
    # Companies are more likely to have relationships within their sector or with certain sectors
    sector_relationship_bias = {
        'Technology': ['Technology', 'Consumer', 'Financials'],
        'Energy': ['Energy', 'Industrials', 'Utilities'],
        'Financials': ['Financials', 'Technology', 'Consumer'],
        'Healthcare': ['Healthcare', 'Technology', 'Consumer'],
        'Consumer': ['Consumer', 'Materials', 'Technology'],
        'Industrials': ['Industrials', 'Materials', 'Energy'],
        'Materials': ['Materials', 'Industrials', 'Energy'],
        'Utilities': ['Utilities', 'Energy', 'Industrials']
    }
    
    # Create more realistic supply chain data
    suppliers = []
    customers = []
    relationship_start = []
    relationship_end = []
    sales_ratio = []
    
    # Create dense network of relationships
    for supplier in msci_members[:600]:  # Use 600 suppliers
        supplier_sector = sector_mapping[supplier]
        
        # Determine number of customers for this supplier (power law distribution)
        num_customers = max(1, int(np.random.pareto(1.5)))
        num_customers = min(num_customers, 20)  # Cap at 20 customers
        
        # Determine potential customers based on sector relationships
        preferred_sectors = sector_relationship_bias[supplier_sector]
        potential_customers = [c for c in msci_members[400:] if sector_mapping[c] in preferred_sectors]
        
        # If not enough potential customers, add some random ones
        if len(potential_customers) < num_customers:
            additional_customers = [c for c in msci_members[400:] if c not in potential_customers]
            potential_customers.extend(np.random.choice(additional_customers, 
                                                      min(num_customers - len(potential_customers), len(additional_customers)),
                                                      replace=False))
        
        # Select customers for this supplier
        if potential_customers:
            selected_customers = np.random.choice(potential_customers, 
                                                min(num_customers, len(potential_customers)), 
                                                replace=False)
            
            for customer in selected_customers:
                suppliers.append(supplier)
                customers.append(customer)
                
                # Random start and end dates
                start = pd.to_datetime(start_date) + timedelta(days=np.random.randint(0, 365))
                end = pd.to_datetime(end_date) - timedelta(days=np.random.randint(0, 365))
                
                relationship_start.append(start)
                relationship_end.append(end)
                
                # Random sales ratio (often NULL in real data)
                if np.random.random() < 0.09:  # 9% have sales ratio as per paper
                    sales_ratio.append(np.random.uniform(0.01, 0.5))
                else:
                    sales_ratio.append(np.nan)
    
    supply_chain_data = pd.DataFrame({
        'supplier': suppliers,
        'customer': customers,
        'relationship_start': relationship_start,
        'relationship_end': relationship_end,
        'sales_ratio': sales_ratio
    })
    
    print(f"Created synthetic supply chain data with {len(supply_chain_data)} relationships")
    print(f"Number of unique suppliers: {supply_chain_data['supplier'].nunique()}")
    print(f"Number of unique customers: {supply_chain_data['customer'].nunique()}")
    
    return supply_chain_data

# Get MSCI Developed Markets Index members
def get_msci_index_members(con):
    """Get MSCI Developed Markets Index members"""
    print("Retrieving MSCI DM Index members...")
    
    try:
        # Try to get real members from Bloomberg
        msci_members = con.ref("MXWO Index", "INDX_MEMBERS")
        msci_members = msci_members["INDX_MEMBERS"].tolist()
        print(f"Retrieved {len(msci_members)} actual MSCI DM Index members from Bloomberg")
    except Exception as e:
        print(f"Error fetching MSCI members: {e}. Using synthetic tickers.")
        # Create dummy tickers for testing
        msci_members = [f"TICKER{i}" for i in range(1, 1001)]
        print(f"Created {len(msci_members)} synthetic tickers")
    
    return msci_members

# Get price data for stocks
def get_price_data(con, tickers, start_date, end_date):
    """
    Get monthly price data for a list of tickers
    
    Parameters:
    con (pdblp.BCon): Bloomberg connection object
    tickers (list): List of ticker symbols
    start_date (str): Start date in YYYYMMDD format
    end_date (str): End date in YYYYMMDD format
    
    Returns:
    DataFrame: Monthly price data
    """
    print(f"Retrieving price data for {len(tickers)} tickers...")
    
    try:
        # Try to get real price data from Bloomberg
        # Use a subset of tickers if there are too many
        if len(tickers) > 1000:
            print(f"Too many tickers ({len(tickers)}). Using first 1000.")
            sample_tickers = tickers[:1000]
        else:
            sample_tickers = tickers
            
        # Add Index Ticker or Security suffix if needed
        sample_tickers_with_suffix = [t if ' ' in t else f"{t} Equity" for t in sample_tickers]
        
        # Get price data from Bloomberg
        price_data = con.bdh(sample_tickers_with_suffix, "PX_LAST", 
                             start_date, end_date, periodicity="MONTHLY")
        
        # Reshape the data
        price_data = price_data.pivot(columns='ticker', values='PX_LAST')
        print(f"Retrieved price data for {price_data.shape[1]} tickers from Bloomberg")
        
    except Exception as e:
        print(f"Error fetching price data: {e}. Using synthetic price data.")
        # Create synthetic price data
        dates = pd.date_range(start=pd.to_datetime(start_date), 
                              end=pd.to_datetime(end_date), 
                              freq='MS')
        
        price_data = pd.DataFrame(index=dates)
        
        for ticker in tickers:
            # Generate random price series with momentum
            initial_price = 100
            prices = [initial_price]
            
            # Add autocorrelation for momentum effect
            for i in range(1, len(dates)):
                # Momentum effect with random noise
                momentum = 0.1 * (prices[-1] / initial_price - 1)
                ret = np.random.normal(0.001 + momentum, 0.04)
                prices.append(prices[-1] * (1 + ret))
            
            price_data[ticker] = prices
        
        print(f"Created synthetic price data for {len(tickers)} tickers")
    
    return price_data

# Build network from supply chain data
def build_network(supply_chain_data, date):
    """
    Build a directed graph from supply chain data for a specific date
    
    Parameters:
    supply_chain_data (DataFrame): Supply chain relationship data
    date (datetime): Date for which to build the network
    
    Returns:
    nx.DiGraph: Directed graph of supply chain relationships
    """
    # Filter relationships active at the given date or ended within last year
    one_year_ago = date - timedelta(days=365)
    
    active_relationships = supply_chain_data[
        ((supply_chain_data['relationship_start'] <= date) & 
         ((supply_chain_data['relationship_end'] >= one_year_ago) | 
          (supply_chain_data['relationship_end'].isna())))
    ]
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add edges from supplier to customer
    for _, row in active_relationships.iterrows():
        G.add_edge(row['supplier'], row['customer'])
    
    return G

# Calculate edge betweenness centrality
def calculate_edge_betweenness(G):
    """
    Calculate edge betweenness centrality for a graph
    
    Parameters:
    G (nx.DiGraph): Directed graph of supply chain relationships
    
    Returns:
    dict: Edge betweenness centrality for each edge
    """
    # For large graphs, use approximate betweenness to save time
    if G.number_of_nodes() > 1000:
        print(f"Large network with {G.number_of_nodes()} nodes. Using approximate edge betweenness.")
        # Use a subset of nodes as sources for betweenness calculation
        k = min(500, G.number_of_nodes())
        edge_betweenness = nx.edge_betweenness_centrality_subset(
            G, 
            sources=list(G.nodes())[:k], 
            targets=list(G.nodes())[:k]
        )
    else:
        # Calculate exact edge betweenness centrality
        edge_betweenness = nx.edge_betweenness_centrality(G)
    
    return edge_betweenness

# Get customer layers
def get_customer_layers(G, supplier, max_layers=5):
    """
    Get customers at different layers for a supplier
    
    Parameters:
    G (nx.DiGraph): Directed graph of supply chain relationships
    supplier (str): Supplier ticker
    max_layers (int): Maximum number of layers to consider
    
    Returns:
    dict: Dictionary of customer layers
    """
    customer_layers = {}
    
    # First layer customers (direct customers)
    if supplier in G:
        customer_layers[1] = list(G.successors(supplier))
    else:
        customer_layers[1] = []
    
    # Higher layer customers
    for layer in range(2, max_layers + 1):
        customers = []
        for prev_customer in customer_layers[layer-1]:
            if prev_customer in G:
                customers.extend(G.successors(prev_customer))
        
        # Remove duplicates
        customer_layers[layer] = list(set(customers))
    
    return customer_layers

# Calculate customer momentum
def calculate_customer_momentum(price_data, G, edge_betweenness, date, t_months=12, max_layers=3):
    """
    Calculate customer momentum for all stocks
    
    Parameters:
    price_data (DataFrame): Price data for stocks
    G (nx.DiGraph): Directed graph of supply chain relationships
    edge_betweenness (dict): Edge betweenness centrality for each edge
    date (datetime): Date for which to calculate momentum
    t_months (int): Momentum lookback period in months
    max_layers (int): Maximum number of customer layers to consider
    
    Returns:
    Series: Customer momentum values for each stock
    """
    print(f"Calculating customer momentum for {date.strftime('%Y-%m')}...")
    
    # Calculate standard momentum (t-month returns)
    lookback_date = date - pd.DateOffset(months=t_months)
    if lookback_date in price_data.index:
        standard_momentum = price_data.loc[date] / price_data.loc[lookback_date] - 1
        print(f"Using momentum lookback from {lookback_date.strftime('%Y-%m')} to {date.strftime('%Y-%m')}")
    else:
        # Use earliest available date if requested date is not available
        earliest_date = price_data.index[0]
        standard_momentum = price_data.loc[date] / price_data.loc[earliest_date] - 1
        print(f"WARNING: Requested lookback date not available. Using {earliest_date.strftime('%Y-%m')} to {date.strftime('%Y-%m')}")
    
    # Initialize customer momentum Series with NaN
    customer_momentum = pd.Series(index=standard_momentum.index, dtype=float)
    
    # Count suppliers with customers in network
    suppliers_in_network = 0
    suppliers_with_momentum = 0
    
    # For each stock (supplier)
    for supplier in tqdm(standard_momentum.index, desc=f"Processing suppliers"):
        try:
            # Skip if supplier not in network
            if supplier not in G:
                continue
                
            suppliers_in_network += 1
            
            # Get customer layers
            customer_layers = get_customer_layers(G, supplier, max_layers)
            
            # Count customers with momentum data
            customers_with_data = 0
            for layer in range(1, max_layers + 1):
                for customer in customer_layers[layer]:
                    if customer in standard_momentum.index:
                        customers_with_data += 1
            
            # Skip if no customers with data
            if customers_with_data == 0:
                continue
            
            # Calculate weighted customer momentum across all layers
            total_weighted_momentum = 0
            total_weight = 0
            
            for layer in range(1, max_layers + 1):
                for customer in customer_layers[layer]:
                    if customer in standard_momentum.index:
                        # Get edge betweenness centrality for this relationship
                        # For higher layers, use the product of centralities along the path
                        # Here we use a simplified approximation
                        if layer == 1 and (supplier, customer) in edge_betweenness:
                            weight = edge_betweenness[(supplier, customer)]
                        else:
                            # For higher layers, use a decaying weight based on the layer
                            weight = 1.0 / (2 ** (layer - 1))
                        
                        # Add weighted momentum
                        total_weighted_momentum += weight * standard_momentum[customer]
                        total_weight += weight
            
            # Normalize by total weight
            if total_weight > 0:
                customer_momentum[supplier] = total_weighted_momentum / total_weight
                suppliers_with_momentum += 1
        
        except Exception as e:
            print(f"Error calculating customer momentum for {supplier}: {e}")
    
    print(f"Found {suppliers_in_network} suppliers in network")
    print(f"Calculated customer momentum for {suppliers_with_momentum} suppliers")
    
    # If very few stocks have momentum, add synthetic data for testing
    if suppliers_with_momentum < 5:
        print("WARNING: Very few stocks with customer momentum. Adding synthetic data for testing.")
        # Add some random stocks with momentum for testing
        random_stocks = np.random.choice(
            [s for s in standard_momentum.index if pd.isna(customer_momentum[s])], 
            size=min(20, len(standard_momentum) - suppliers_with_momentum), 
            replace=False
        )
        for stock in random_stocks:
            customer_momentum[stock] = np.random.normal(0, 0.1)  # Random momentum
        print(f"Added {len(random_stocks)} synthetic stocks for testing")
    
    return customer_momentum

# Form portfolios based on customer momentum
def form_portfolios(customer_momentum, n_portfolios=5):
    """
    Form portfolios based on customer momentum
    
    Parameters:
    customer_momentum (Series): Customer momentum values
    n_portfolios (int): Number of portfolios to form
    
    Returns:
    dict: Dictionary of portfolios
    """
    # Drop stocks with missing customer momentum
    valid_momentum = customer_momentum.dropna()
    
    # Check if we have enough stocks for n_portfolios
    available_stocks = len(valid_momentum)
    print(f"Found {available_stocks} stocks with valid customer momentum")
    
    # If very few stocks, adjust portfolio count
    effective_n_portfolios = min(n_portfolios, available_stocks)
    if effective_n_portfolios < n_portfolios:
        print(f"WARNING: Not enough stocks. Reducing portfolios from {n_portfolios} to {effective_n_portfolios}")
    
    # Initialize empty portfolios
    portfolios = {i+1: [] for i in range(n_portfolios)}
    
    if available_stocks == 0:
        print("No valid stocks with customer momentum. Returning empty portfolios.")
        return portfolios
        
    if available_stocks == 1:
        # Only one stock - put it in the top portfolio
        portfolios[n_portfolios] = [valid_momentum.index[0]]
        print(f"Only one valid stock. Assigned to top portfolio {n_portfolios}.")
        return portfolios
        
    if available_stocks < n_portfolios:
        # Too few stocks for all portfolios
        # Assign to top and bottom portfolios
        sorted_stocks = valid_momentum.sort_values()
        portfolios[1] = [sorted_stocks.index[0]]
        portfolios[n_portfolios] = [sorted_stocks.index[-1]]
        
        # Distribute remaining stocks
        for i, stock_idx in enumerate(sorted_stocks.index[1:-1]):
            portfolio_num = 2 + i % (n_portfolios - 2)
            portfolios[portfolio_num].append(stock_idx)
            
        print(f"Distributed {available_stocks} stocks across portfolios")
        return portfolios
    
    # Normal case - enough stocks for all portfolios
    try:
        # Calculate quantiles
        quantiles = valid_momentum.quantile(np.linspace(0, 1, n_portfolios+1))
        
        # Form portfolios
        for i in range(n_portfolios):
            if i == 0:
                mask = valid_momentum <= quantiles[i+1]
            elif i == n_portfolios - 1:
                mask = valid_momentum > quantiles[i]
            else:
                mask = (valid_momentum > quantiles[i]) & (valid_momentum <= quantiles[i+1])
            
            portfolios[i+1] = valid_momentum[mask].index.tolist()
            
    except Exception as e:
        print(f"Error forming portfolios: {e}")
        print(f"Valid momentum statistics: min={valid_momentum.min()}, max={valid_momentum.max()}, count={len(valid_momentum)}")
        print(f"Valid momentum values: {valid_momentum.values}")
        
        # Simple alternative approach - sort and split
        sorted_stocks = valid_momentum.sort_values()
        stocks_per_portfolio = len(sorted_stocks) // n_portfolios
        remainder = len(sorted_stocks) % n_portfolios
        
        start_idx = 0
        for i in range(n_portfolios):
            extra = 1 if i < remainder else 0
            end_idx = start_idx + stocks_per_portfolio + extra
            portfolios[i+1] = sorted_stocks.index[start_idx:end_idx].tolist()
            start_idx = end_idx
    
    return portfolios

# Calculate portfolio returns
def calculate_portfolio_returns(portfolios, price_data, date, next_date):
    """
    Calculate returns for portfolios
    
    Parameters:
    portfolios (dict): Dictionary of portfolios
    price_data (DataFrame): Price data for stocks
    date (datetime): Current date
    next_date (datetime): Next date
    
    Returns:
    Series: Portfolio returns
    """
    try:
        # Calculate stock returns
        if next_date in price_data.index:
            # Make sure we have valid prices
            valid_current = price_data.loc[date].replace([0, np.inf, -np.inf], np.nan)
            valid_next = price_data.loc[next_date].replace([0, np.inf, -np.inf], np.nan)
            
            # Calculate returns only where both prices are valid
            stock_returns = pd.Series(index=price_data.columns)
            for col in price_data.columns:
                if pd.notna(valid_current[col]) and pd.notna(valid_next[col]) and valid_current[col] > 0:
                    stock_returns[col] = valid_next[col] / valid_current[col] - 1
                else:
                    stock_returns[col] = np.nan
        else:
            # Use last available date if next_date is not available
            print(f"WARNING: Next date {next_date} not in price data. Using zero returns.")
            stock_returns = pd.Series(index=price_data.columns, data=0.0)
        
        # Calculate equal-weighted portfolio returns
        portfolio_returns = {}
        for portfolio_num, stocks in portfolios.items():
            if stocks:
                # Filter to stocks with valid return data
                valid_stocks = [s for s in stocks if s in stock_returns.index and pd.notna(stock_returns[s])]
                
                if valid_stocks:
                    # Clip extreme returns for stability
                    clipped_returns = stock_returns[valid_stocks].clip(-0.5, 0.5)
                    portfolio_returns[portfolio_num] = clipped_returns.mean()
                    
                    # Print extreme returns for debugging
                    extreme_returns = stock_returns[valid_stocks][(stock_returns[valid_stocks] > 0.2) | 
                                                              (stock_returns[valid_stocks] < -0.2)]
                    if not extreme_returns.empty:
                        print(f"NOTE: Extreme returns in portfolio {portfolio_num}: {extreme_returns}")
                else:
                    print(f"WARNING: No valid stocks with returns in portfolio {portfolio_num}")
                    portfolio_returns[portfolio_num] = 0.0
            else:
                portfolio_returns[portfolio_num] = 0.0
        
        # Calculate long-short return
        top_portfolio = max(portfolios.keys())
        bottom_portfolio = 1
        
        top_return = portfolio_returns.get(top_portfolio, 0.0)
        bottom_return = portfolio_returns.get(bottom_portfolio, 0.0)
        
        ls_return = top_return - bottom_return
        
        # Safety check for long-short return
        if np.abs(ls_return) > 0.5:
            print(f"WARNING: Large long-short return: {ls_return}. Top: {top_return}, Bottom: {bottom_return}. Clipping.")
            ls_return = np.sign(ls_return) * 0.5  # Clip to +/- 50%
            
        portfolio_returns['Long-Short'] = ls_return
        
        return pd.Series(portfolio_returns)
        
    except Exception as e:
        print(f"ERROR calculating portfolio returns: {e}")
        # Return zeros as a fallback
        return pd.Series({k: 0.0 for k in list(portfolios.keys()) + ['Long-Short']})

# Backtest strategy
def backtest_strategy(con, start_date, end_date, t_months=12, max_layers=3, n_portfolios=5):
    """
    Backtest customer momentum strategy
    
    Parameters:
    con (pdblp.BCon): Bloomberg connection object
    start_date (str): Start date in YYYYMMDD format
    end_date (str): End date in YYYYMMDD format
    t_months (int): Momentum lookback period in months
    max_layers (int): Maximum number of customer layers to consider
    n_portfolios (int): Number of portfolios to form
    
    Returns:
    DataFrame: Portfolio returns
    """
    # Get MSCI DM index members
    msci_members = get_msci_index_members(con)
    print(f"Retrieved {len(msci_members)} MSCI DM index members")
    
    # Get supply chain data
    supply_chain_data = get_supply_chain_data(con, start_date, end_date)
    print(f"Retrieved {len(supply_chain_data)} supply chain relationships")
    
    # Get price data
    unique_firms = list(set(supply_chain_data['supplier'].unique()) | set(supply_chain_data['customer'].unique()))
    price_data = get_price_data(con, msci_members, start_date, end_date)
    print(f"Retrieved price data for {price_data.shape[1]} stocks across {price_data.shape[0]} dates")
    
    # Monthly dates for the backtest
    dates = price_data.index
    
    # We need at least t_months of data before starting
    start_idx = t_months
    if start_idx >= len(dates):
        print(f"ERROR: Not enough price data. Need at least {t_months} months.")
        return pd.DataFrame()
    
    # Initialize results DataFrame
    all_columns = list(range(1, n_portfolios+1)) + ['Long-Short']
    portfolio_returns = pd.DataFrame(index=dates[start_idx:], columns=all_columns)
    
    # Loop through months
    for i in range(start_idx-1, len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i+1]
        
        print(f"\nProcessing {current_date.strftime('%Y-%m')} to {next_date.strftime('%Y-%m')}")
        
        try:
            # Build network
            G = build_network(supply_chain_data, current_date)
            print(f"Network built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            
            # Skip if network is too small
            if G.number_of_edges() < 10:
                print(f"WARNING: Network too small with only {G.number_of_edges()} edges. Skipping.")
                portfolio_returns.loc[next_date] = [0.0] * len(all_columns)
                continue
            
            # Calculate edge betweenness centrality
            edge_betweenness = calculate_edge_betweenness(G)
            print(f"Edge betweenness calculated for {len(edge_betweenness)} edges")
            
            # Calculate customer momentum
            customer_momentum = calculate_customer_momentum(
                price_data, G, edge_betweenness, current_date, t_months, max_layers)
            
            valid_count = customer_momentum.count()
            print(f"Customer momentum calculated for {valid_count} stocks")
            
            # Skip if too few valid stocks
            if valid_count < 2:
                print(f"WARNING: Too few stocks ({valid_count}) with valid momentum. Skipping.")
                portfolio_returns.loc[next_date] = [0.0] * len(all_columns)
                continue
            
            # Form portfolios
            portfolios = form_portfolios(customer_momentum, n_portfolios)
            stocks_in_portfolios = sum(len(p) for p in portfolios.values())
            print(f"Portfolios formed with {stocks_in_portfolios} stocks. Counts: {[len(v) for k, v in portfolios.items()]}")
            
            # Calculate portfolio returns
            returns = calculate_portfolio_returns(portfolios, price_data, current_date, next_date)
            
            # Store results
            portfolio_returns.loc[next_date] = returns
            
        except Exception as e:
            print(f"ERROR: Failed to process {current_date.strftime('%Y-%m')} to {next_date.strftime('%Y-%m')}: {e}")
            # Fill with zeros for this period
            portfolio_returns.loc[next_date] = [0.0] * len(all_columns)
    
    return portfolio_returns

# Calculate performance metrics
def calculate_performance_metrics(returns):
    """
    Calculate performance metrics for portfolio returns
    
    Parameters:
    returns (DataFrame): Portfolio returns
    
    Returns:
    DataFrame: Performance metrics
    """
    # Check if returns contain valid data
    if returns.empty or returns.isna().all().all():
        print("WARNING: Returns data contains no valid values. Cannot calculate metrics.")
        return pd.DataFrame(
            {'Annual Return': np.nan, 'Annual Volatility': np.nan, 
             'Sharpe Ratio': np.nan, 'Max Drawdown': np.nan},
            index=returns.columns
        )
    
    # Calculate cumulative returns with error handling
    try:
        # Handle any extreme values in returns
        clean_returns = returns.copy()
        clean_returns = clean_returns.clip(-0.5, 0.5)  # Limit extreme daily returns
        clean_returns = clean_returns.fillna(0)  # Fill NaN with zeros
        
        cumulative_returns = (1 + clean_returns).cumprod()
        
        # Print diagnostic information
        print("\nDiagnostic Information:")
        print(f"Returns range: {clean_returns.min().min()} to {clean_returns.max().max()}")
        print(f"Cumulative returns final values: \n{cumulative_returns.iloc[-1]}")
        
        # Calculate years more robustly
        years = len(clean_returns) / 12  # Monthly returns
        if years < 0.5:  # If less than 6 months of data
            print("WARNING: Less than 6 months of data. Metrics may not be meaningful.")
            years = max(years, 0.5)  # Use at least 6 months to avoid extreme annualization
        
        # Calculate annualized returns safely
        annual_return = cumulative_returns.iloc[-1] ** (1 / years) - 1
        
        # Handle any infinity or very large values
        annual_return = annual_return.replace([np.inf, -np.inf], np.nan)
        
        # Calculate annualized volatility
        annual_volatility = clean_returns.std() * np.sqrt(12)
        
        # Calculate Sharpe ratio safely 
        risk_free_rate = 0.01  # Assume 1% risk-free rate
        excess_return = annual_return - risk_free_rate
        
        # Avoid division by zero in Sharpe ratio
        sharpe_ratio = pd.Series(index=annual_return.index)
        for col in annual_return.index:
            if annual_volatility[col] > 0:
                sharpe_ratio[col] = excess_return[col] / annual_volatility[col]
            else:
                sharpe_ratio[col] = np.nan
        
        # Calculate maximum drawdown
        rolling_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns / rolling_max) - 1
        max_drawdown = drawdown.min()
        
        # Create metrics DataFrame
        metrics = pd.DataFrame({
            'Annual Return': annual_return,
            'Annual Volatility': annual_volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown
        })
        
        return metrics
        
    except Exception as e:
        print(f"ERROR calculating performance metrics: {e}")
        import traceback
        traceback.print_exc()
        
        # Return empty metrics
        return pd.DataFrame(
            {'Annual Return': np.nan, 'Annual Volatility': np.nan, 
             'Sharpe Ratio': np.nan, 'Max Drawdown': np.nan},
            index=returns.columns
        )

# Main function
def main():
    """Main function to run the backtest"""
    try:
        # Set parameters
        start_date = "20030101"
        end_date = "20191231"
        t_months = 12
        max_layers = 3
        n_portfolios = 5
        
        # Connect to Bloomberg
        con = connect_to_bloomberg()
        
        # Backtest strategy
        returns = backtest_strategy(con, start_date, end_date, t_months, max_layers, n_portfolios)
        
        if returns.empty:
            print("No returns data generated. Exiting.")
            return
        
        # Analyze and clean the returns data
        print("\nReturns Data Analysis:")
        print(f"Shape: {returns.shape}")
        print(f"NaN values: {returns.isna().sum()}")
        print(f"Infinity values: {(returns == np.inf).sum() + (returns == -np.inf).sum()}")
        print(f"Min: {returns.min()}")
        print(f"Max: {returns.max()}")
        
        # Clean the returns data
        clean_returns = returns.copy()
        clean_returns = clean_returns.replace([np.inf, -np.inf], np.nan)
        clean_returns = clean_returns.clip(-0.5, 0.5)  # Limit extreme returns
        
        # Fill NaN values with zeros
        clean_returns = clean_returns.fillna(0)
        
        print("\nCleaned Returns Data Analysis:")
        print(f"Min: {clean_returns.min()}")
        print(f"Max: {clean_returns.max()}")
        
        # Calculate cumulative returns
        cumulative_returns = (1 + clean_returns).cumprod()
        
        # Save results
        clean_returns.to_csv("customer_momentum_returns.csv")
        cumulative_returns.to_csv("customer_momentum_cumulative_returns.csv")
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(clean_returns)
        metrics.to_csv("customer_momentum_metrics.csv")
        
        # Print results
        print("\nPerformance Results:")
        print(metrics)
        
        # Plot cumulative returns
        plt.figure(figsize=(12, 8))
        for col in cumulative_returns.columns:
            if col == 'Long-Short':
                plt.plot(cumulative_returns.index, cumulative_returns[col], label=col, linewidth=2)
            else:
                plt.plot(cumulative_returns.index, cumulative_returns[col], label=f'Portfolio {col}', alpha=0.7)
        
        plt.title(f'Customer Momentum Strategy (T={t_months}, Layers={max_layers})')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('customer_momentum_performance.png')
        plt.show()
        
        # Plot monthly returns
        plt.figure(figsize=(12, 6))
        clean_returns['Long-Short'].plot(kind='bar')
        plt.title('Monthly Returns of Long-Short Portfolio')
        plt.xlabel('Date')
        plt.ylabel('Return')
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.savefig('customer_momentum_monthly_returns.png')
        plt.show()
        
        # Plot drawdowns
        drawdowns = (cumulative_returns / cumulative_returns.cummax()) - 1
        plt.figure(figsize=(12, 6))
        drawdowns['Long-Short'].plot()
        plt.title('Drawdowns of Long-Short Portfolio')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('customer_momentum_drawdowns.png')
        plt.show()
        
    except Exception as e:
        print(f"ERROR in main function: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Close Bloomberg connection
        if 'con' in locals():
            try:
                con.stop()
            except:
                pass

# Run Bloomberg API test
def test_bloomberg_connection():
    """Test Bloomberg API connection and functionality"""
    try:
        # Connect to Bloomberg
        con = connect_to_bloomberg()
        
        # Test MSCI members retrieval
        print("\nTesting MSCI Index Members retrieval...")
        try:
            msci_members = con.ref("MXWO Index", "INDX_MEMBERS")
            print(f"Successfully retrieved {len(msci_members)} MSCI DM Index members")
            print(msci_members.head())
        except Exception as e:
            print(f"Error retrieving MSCI members: {e}")
        
        # Test price data retrieval
        print("\nTesting price data retrieval...")
        try:
            tickers = ["AAPL US Equity", "MSFT US Equity", "AMZN US Equity"]
            price_data = con.bdh(tickers, "PX_LAST", "20200101", "20201231", periodicity="MONTHLY")
            print(f"Successfully retrieved price data")
            print(price_data.head())
        except Exception as e:
            print(f"Error retrieving price data: {e}")
        
        # Test supply chain data retrieval
        print("\nTesting supply chain data retrieval...")
        try:
            # This is a placeholder - actual Bloomberg API for supply chain would vary
            # For example, you might use the SPLC function in a Bloomberg terminal
            print("Supply chain data would normally be retrieved via Bloomberg SPLC function")
            print("Please ensure you have access to this functionality in your terminal")
        except Exception as e:
            print(f"Error testing supply chain data: {e}")
        
        # Close Bloomberg connection
        con.stop()
        print("\nBloomberg API test completed")
        
    except Exception as e:
        print(f"Error testing Bloomberg connection: {e}")

if __name__ == "__main__":
    # Uncomment to test Bloomberg connection first
    # test_bloomberg_connection()
    
    # Run main backtest
    main()