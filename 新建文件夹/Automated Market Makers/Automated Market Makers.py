import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.dates as mdates
from scipy.stats import norm
from tqdm import tqdm

# Set random seed for reproducibility
np.random.seed(42)

# Parameters based on the paper
def simulate_amm_dynamics(T=1000, dt=0.01, gamma=0.997, mu=0, sigma=0.2, p0=100, plot=True):
    """
    Simulate the AMM price dynamics as described in the paper.
    
    Parameters:
    - T: Time horizon
    - dt: Time step
    - gamma: Fee parameter (1-gamma is the fee percentage)
    - mu: Drift of the reference price (geometric Brownian motion)
    - sigma: Volatility of the reference price
    - p0: Initial price
    - plot: Whether to plot the results
    
    Returns:
    - DataFrame containing simulation results
    """
    # Calculate bounds based on the fee parameter
    c = np.log(1/gamma)
    lower_bound_factor = gamma
    upper_bound_factor = 1/gamma
    
    # Number of time steps
    n_steps = int(T/dt)
    
    # Initialize arrays
    times = np.linspace(0, T, n_steps)
    dW = np.random.normal(0, np.sqrt(dt), n_steps)  # Brownian increments
    
    # Reference price follows geometric Brownian motion
    ref_log_price = np.zeros(n_steps)
    ref_price = np.zeros(n_steps)
    
    # AMM price
    amm_log_price = np.zeros(n_steps)
    amm_price = np.zeros(n_steps)
    
    # Initialize prices
    ref_log_price[0] = np.log(p0)
    ref_price[0] = p0
    amm_log_price[0] = np.log(p0)
    amm_price[0] = p0
    
    # Lower and upper bounds
    lower_bound_log = np.zeros(n_steps)
    upper_bound_log = np.zeros(n_steps)
    lower_bound = np.zeros(n_steps)
    upper_bound = np.zeros(n_steps)
    
    lower_bound_log[0] = ref_log_price[0] - c
    upper_bound_log[0] = ref_log_price[0] + c
    lower_bound[0] = ref_price[0] * lower_bound_factor
    upper_bound[0] = ref_price[0] * upper_bound_factor
    
    # Track arbitrage opportunities
    arbitrage_events = []
    
    # Simulate price dynamics
    for i in range(1, n_steps):
        # Update reference price (geometric Brownian motion)
        ref_log_price[i] = ref_log_price[i-1] + (mu - 0.5 * sigma**2) * dt + sigma * dW[i-1]
        ref_price[i] = np.exp(ref_log_price[i])
        
        # Calculate bounds
        lower_bound_log[i] = ref_log_price[i] - c
        upper_bound_log[i] = ref_log_price[i] + c
        lower_bound[i] = ref_price[i] * lower_bound_factor
        upper_bound[i] = ref_price[i] * upper_bound_factor
        
        # Update AMM price based on arbitrage constraints
        if ref_log_price[i] - c > amm_log_price[i-1]:
            # Arbitrage pushes AMM price up
            amm_log_price[i] = ref_log_price[i] - c
            arbitrage_events.append((times[i], "up", ref_price[i], np.exp(amm_log_price[i])))
        elif ref_log_price[i] + c < amm_log_price[i-1]:
            # Arbitrage pushes AMM price down
            amm_log_price[i] = ref_log_price[i] + c
            arbitrage_events.append((times[i], "down", ref_price[i], np.exp(amm_log_price[i])))
        else:
            # No arbitrage, AMM price remains unchanged
            amm_log_price[i] = amm_log_price[i-1]
        
        amm_price[i] = np.exp(amm_log_price[i])
    
    # Create DataFrame with results
    results = pd.DataFrame({
        'time': times,
        'ref_price': ref_price,
        'amm_price': amm_price,
        'ref_log_price': ref_log_price,
        'amm_log_price': amm_log_price,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'lower_bound_log': lower_bound_log,
        'upper_bound_log': upper_bound_log
    })
    
    # Convert arbitrage events to DataFrame
    arb_df = pd.DataFrame(arbitrage_events, 
                         columns=['time', 'direction', 'ref_price', 'amm_price']) if arbitrage_events else None
    
    # Plot results if requested
    if plot:
        plot_simulation_results(results, arb_df, gamma)
    
    return results, arb_df

def plot_simulation_results(results, arb_df, gamma):
    """
    Plot the simulation results
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 12), sharex=True)
    
    # Plot 1: Price dynamics in original scale
    ax1 = axes[0]
    ax1.plot(results['time'], results['ref_price'], label='Reference Price', color='blue')
    ax1.plot(results['time'], results['amm_price'], label='AMM Price', color='red')
    ax1.plot(results['time'], results['lower_bound'], label=f'Lower Bound (γ={gamma:.4f})', color='green', linestyle='--')
    ax1.plot(results['time'], results['upper_bound'], label=f'Upper Bound (1/γ={1/gamma:.4f})', color='purple', linestyle='--')
    
    # Add arbitrage points if any
    if arb_df is not None:
        for direction in ['up', 'down']:
            subset = arb_df[arb_df['direction'] == direction]
            marker = '^' if direction == 'up' else 'v'
            color = 'green' if direction == 'up' else 'red'
            if not subset.empty:
                ax1.scatter(subset['time'], subset['amm_price'], marker=marker, color=color, s=80, 
                           label=f'Arbitrage ({direction})')
    
    ax1.set_title('Reference and AMM Price Dynamics')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Log price dynamics (as shown in the paper)
    ax2 = axes[1]
    ax2.plot(results['time'], results['ref_log_price'], label='Log Reference Price', color='blue')
    ax2.plot(results['time'], results['amm_log_price'], label='Log AMM Price', color='red')
    ax2.plot(results['time'], results['lower_bound_log'], label='Log Lower Bound', color='green', linestyle='--')
    ax2.plot(results['time'], results['upper_bound_log'], label='Log Upper Bound', color='purple', linestyle='--')
    
    ax2.set_title('Log Price Dynamics (as in Fig. 1 of the paper)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Log Price')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def analyze_amm_performance(gamma_values, n_simulations=100, T=1000, dt=0.01, p0=100, sigma=0.2):
    """
    Analyze AMM performance for different fee levels
    
    Parameters:
    - gamma_values: List of fee parameters to test
    - n_simulations: Number of simulations for each gamma value
    - T, dt, p0, sigma: Simulation parameters
    
    Returns:
    - DataFrame with performance metrics
    """
    results = []
    
    for gamma in tqdm(gamma_values, desc="Testing fee levels"):
        # Metrics across simulations
        arbitrage_counts = []
        lp_revenues = []
        price_deviations = []
        
        for sim in range(n_simulations):
            # Run simulation
            sim_data, arb_df = simulate_amm_dynamics(T=T, dt=dt, gamma=gamma, 
                                                   sigma=sigma, p0=p0, plot=False)
            
            # Count arbitrage events
            arb_count = 0 if arb_df is None else len(arb_df)
            arbitrage_counts.append(arb_count)
            
            # Calculate LP revenue from fees
            if arb_df is not None and not arb_df.empty:
                # Simplified fee revenue model based on price impact
                fee_percentage = 1 - gamma
                total_volume = arb_df['ref_price'].sum()  # Simplification - actual volume would be more complex
                revenue = total_volume * fee_percentage
                lp_revenues.append(revenue)
            else:
                lp_revenues.append(0)
            
            # Calculate average price deviation (as percentage)
            price_deviation = np.mean(np.abs(sim_data['amm_price'] - sim_data['ref_price']) / sim_data['ref_price'])
            price_deviations.append(price_deviation)
        
        # Record results for this gamma
        results.append({
            'gamma': gamma,
            'fee_percentage': (1-gamma)*100,
            'avg_arbitrage_events': np.mean(arbitrage_counts),
            'avg_lp_revenue': np.mean(lp_revenues),
            'avg_price_deviation': np.mean(price_deviations),
            'std_price_deviation': np.std(price_deviations)
        })
    
    return pd.DataFrame(results)

def plot_amm_performance(performance_df):
    """
    Plot the performance metrics
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Sort by fee_percentage for better visualization
    df = performance_df.sort_values('fee_percentage')
    
    # Plot 1: Arbitrage Events vs Fee
    ax1 = axes[0]
    ax1.bar(df['fee_percentage'], df['avg_arbitrage_events'], color='purple')
    ax1.set_title('Average Number of Arbitrage Events vs Fee')
    ax1.set_xlabel('Fee Percentage')
    ax1.set_ylabel('Avg. Arbitrage Events')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(PercentFormatter())
    
    # Plot 2: LP Revenue vs Fee
    ax2 = axes[1]
    ax2.bar(df['fee_percentage'], df['avg_lp_revenue'], color='green')
    ax2.set_title('Average LP Revenue vs Fee')
    ax2.set_xlabel('Fee Percentage')
    ax2.set_ylabel('Avg. LP Revenue')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(PercentFormatter())
    
    # Plot 3: Price Deviation vs Fee
    ax3 = axes[2]
    ax3.errorbar(df['fee_percentage'], df['avg_price_deviation']*100, 
                yerr=df['std_price_deviation']*100, fmt='o-', color='blue', capsize=5)
    ax3.set_title('Average Price Deviation vs Fee')
    ax3.set_xlabel('Fee Percentage')
    ax3.set_ylabel('Avg. Price Deviation (%)')
    ax3.grid(True, alpha=0.3)
    ax3.xaxis.set_major_formatter(PercentFormatter())
    ax3.yaxis.set_major_formatter(PercentFormatter())
    
    plt.tight_layout()
    plt.show()

def simulate_amm_excursions(T=1000, dt=0.01, gamma=0.997, mu=0, sigma=0.3, p0=100):
    """
    Simulate the AMM price and analyze the excursions as described in the paper
    
    Parameters:
    - T: Time horizon
    - dt: Time step
    - gamma: Fee parameter
    - mu, sigma: GBM parameters
    - p0: Initial price
    
    Returns:
    - DataFrame with simulation results and excursion statistics
    """
    # First run the simulation
    results, arb_df = simulate_amm_dynamics(T, dt, gamma, mu, sigma, p0, plot=False)
    
    # Find excursions (periods between arbitrage events)
    if arb_df is None or len(arb_df) < 2:
        print("Not enough arbitrage events to analyze excursions")
        return results, None
    
    # Get arbitrage times
    arb_times = arb_df['time'].values
    
    # Calculate excursion durations
    excursion_durations = np.diff(arb_times)
    
    # Analyze the distribution of excursion durations
    mean_duration = np.mean(excursion_durations)
    median_duration = np.median(excursion_durations)
    
    # Create histogram of excursion durations
    plt.figure(figsize=(10, 6))
    plt.hist(excursion_durations, bins=30, alpha=0.7, color='blue')
    plt.axvline(mean_duration, color='red', linestyle='--', label=f'Mean: {mean_duration:.2f}')
    plt.axvline(median_duration, color='green', linestyle='--', label=f'Median: {median_duration:.2f}')
    plt.title('Distribution of Excursion Durations between Arbitrage Events')
    plt.xlabel('Duration')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Theoretical expected duration based on the paper
    c = np.log(1/gamma)
    theoretical_mean = 4 * c**2
    
    print(f"Theoretical expected excursion duration (based on paper): {theoretical_mean:.2f}")
    print(f"Empirical mean excursion duration: {mean_duration:.2f}")
    
    # Analyze excursion heights
    if len(arb_df) >= 2:
        # Calculate price changes during excursions
        excursion_heights = []
        directions = []
        
        for i in range(len(arb_df) - 1):
            height = arb_df.iloc[i+1]['amm_price'] - arb_df.iloc[i]['amm_price']
            excursion_heights.append(height)
            directions.append("up" if height > 0 else "down")
        
        # Plot excursion heights
        plt.figure(figsize=(10, 6))
        plt.hist(excursion_heights, bins=30, alpha=0.7, color='purple')
        plt.title('Distribution of AMM Price Changes during Excursions')
        plt.xlabel('Price Change')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Check if heights follow normal distribution as suggested by the paper
        plt.figure(figsize=(10, 6))
        plt.hist(excursion_heights, bins=30, density=True, alpha=0.7, color='blue', 
                label='Empirical distribution')
        
        # Fit a normal distribution
        mu_fit, std_fit = norm.fit(excursion_heights)
        x = np.linspace(min(excursion_heights), max(excursion_heights), 100)
        plt.plot(x, norm.pdf(x, mu_fit, std_fit), 'r-', lw=2, 
                label=f'Normal fit: μ={mu_fit:.2f}, σ={std_fit:.2f}')
        
        plt.title('Excursion Heights vs. Normal Distribution')
        plt.xlabel('Price Change')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Analyze the relationship between excursion heights and durations
        plt.figure(figsize=(10, 6))
        plt.scatter(excursion_durations, excursion_heights, alpha=0.7, c=directions, 
                   cmap='coolwarm')
        plt.title('Excursion Heights vs. Durations')
        plt.xlabel('Duration')
        plt.ylabel('Price Change')
        plt.grid(True, alpha=0.3)
        plt.colorbar(label='Direction')
        plt.show()
        
        # Proposition 4 from the paper: Test if V_H / sqrt(H) follows N(0,1)
        # Using the price changes scaled by the square root of duration
        scaled_heights = np.array(excursion_heights) / np.sqrt(excursion_durations)
        
        plt.figure(figsize=(10, 6))
        plt.hist(scaled_heights, bins=30, density=True, alpha=0.7, color='blue', 
                label='Empirical scaled heights')
        
        # Standard normal distribution
        x = np.linspace(min(scaled_heights), max(scaled_heights), 100)
        plt.plot(x, norm.pdf(x, 0, 1), 'r-', lw=2, label='N(0,1)')
        
        plt.title('Scaled Excursion Heights vs. Standard Normal (Proposition 4)')
        plt.xlabel('Price Change / √Duration')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # Return excursion data
        excursion_data = pd.DataFrame({
            'duration': excursion_durations,
            'height': excursion_heights,
            'direction': directions,
            'scaled_height': scaled_heights
        })
        
        return results, excursion_data
    
    return results, None

# Main simulation with parameters based on real-world AMMs
print("Running main simulation...")
gamma = 0.997  # Corresponds to a 0.3% fee (30 bps), common on Uniswap
simulation_results, arb_events = simulate_amm_dynamics(
    T=1000,       # Time horizon
    dt=0.01,      # Time step
    gamma=gamma,  # Fee parameter
    mu=0.05,      # Small positive drift
    sigma=0.5,    # Volatility
    p0=100        # Initial price
)

# Print summary statistics
if arb_events is not None:
    print(f"\nNumber of arbitrage events: {len(arb_events)}")
    up_arbs = arb_events[arb_events['direction'] == 'up']
    down_arbs = arb_events[arb_events['direction'] == 'down']
    print(f"Up arbitrages: {len(up_arbs)}, Down arbitrages: {len(down_arbs)}")
    
    # Calculate theoretical no-arbitrage bounds width
    bounds_width_pct = (1/gamma - gamma) * 100
    print(f"No-arbitrage bounds width: {bounds_width_pct:.2f}%")
    
    # Calculate fee revenue (simplified model)
    fee_pct = (1 - gamma) * 100
    print(f"Fee percentage: {fee_pct:.2f}%")
else:
    print("No arbitrage events detected in this simulation.")

# Compare different fee levels
print("\nComparing different fee levels...")
gamma_values = [0.999, 0.997, 0.995, 0.99, 0.98, 0.97, 0.95]  # Corresponds to fees from 0.1% to 5%
performance_df = analyze_amm_performance(gamma_values, n_simulations=20)
plot_amm_performance(performance_df)

# Analyze excursions
print("\nAnalyzing excursions between arbitrage events...")
_, excursion_data = simulate_amm_excursions(
    T=5000,      # Longer time horizon
    dt=0.01,
    gamma=0.997,
    mu=0,
    sigma=0.4    # Higher volatility to generate more arbitrage events
)

if excursion_data is not None:
    print("\nExcursion statistics:")
    print(f"Total excursions: {len(excursion_data)}")
    print(f"Mean duration: {excursion_data['duration'].mean():.2f}")
    print(f"Mean height: {excursion_data['height'].mean():.2f}")
    
    # Test normality of scaled heights (Proposition 4)
    from scipy import stats
    _, p_value = stats.normaltest(excursion_data['scaled_height'])
    print(f"Test of scaled heights normality (p-value): {p_value:.4f}")
    print("Low p-value suggests rejection of normality hypothesis")