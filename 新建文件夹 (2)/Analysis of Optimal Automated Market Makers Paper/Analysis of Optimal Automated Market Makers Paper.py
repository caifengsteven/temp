import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

class OptimalMarketMaker:
    """
    Implementation of the optimal market maker for two goods with adverse selection
    as described in Section 5.2 of the paper.
    """
    
    def __init__(self, c=(0.5, 0.5), lambda_val=1.0):
        """
        Initialize the market maker with initial beliefs and adverse selection parameter.
        
        Args:
            c: Initial belief about good values (default: (0.5, 0.5))
            lambda_val: Adverse selection parameter (default: 1.0)
                        λ=1 means no adverse selection, λ=0 means full adverse selection
        """
        self.c = c
        self.lambda_val = lambda_val
        
        # Compute the octagon parameters based on lambda
        self.a = (1 + np.sqrt(2)) * self.lambda_val / (4 * self.lambda_val + 2)
        self.b = self.lambda_val / (4 * self.lambda_val + 2)
        
        # Compute menu prices based on lambda
        self._compute_menu_prices()
    
    def _compute_menu_prices(self):
        """
        Compute the menu prices for the optimal mechanism based on lambda.
        Using the formula from Table 2 in the paper.
        """
        lambda_val = self.lambda_val
        
        # Menu items and their prices
        self.menu = {
            (1, 0): (3*lambda_val + 2) / (4*lambda_val + 2),  # Buy good 1
            (0, 1): (3*lambda_val + 2) / (4*lambda_val + 2),  # Buy good 2
            (-1, 0): -lambda_val / (4*lambda_val + 2),        # Sell good 1
            (0, -1): -lambda_val / (4*lambda_val + 2),        # Sell good 2
            (1, 1): (-np.sqrt(2)*lambda_val + 6*lambda_val + 4) / (4*lambda_val + 2),  # Buy both
            (-1, -1): -((2+np.sqrt(2))*lambda_val) / (4*lambda_val + 2),              # Sell both
            (1, -1): (-np.sqrt(2)*lambda_val + 2*lambda_val + 2) / (4*lambda_val + 2),  # Buy 1, sell 2
            (-1, 1): (-np.sqrt(2)*lambda_val + 2*lambda_val + 2) / (4*lambda_val + 2),  # Sell 1, buy 2
            (0, 0): 0  # No trade
        }
        
    def get_allocation_and_payment(self, x):
        """
        Get the optimal allocation and payment for a trader with valuation x.
        
        Args:
            x: Trader's valuation for the two goods as a tuple (x1, x2)
            
        Returns:
            tuple: (allocation, payment)
        """
        # Check if point is in no-trade region (octagon)
        if self._in_no_trade_region(x):
            return (0, 0), 0
        
        # Compute utilities for all menu items
        utilities = {}
        for alloc, price in self.menu.items():
            utility = alloc[0] * x[0] + alloc[1] * x[1] - price
            utilities[alloc] = utility
        
        # Find the allocation that maximizes utility
        best_alloc = max(utilities.items(), key=lambda item: item[1])[0]
        
        return best_alloc, self.menu[best_alloc]
    
    def _in_no_trade_region(self, x):
        """
        Check if a point is in the no-trade region (octagon in the center).
        
        Args:
            x: Point to check (x1, x2)
            
        Returns:
            bool: True if point is in no-trade region
        """
        # For simplified implementation, we'll approximate the octagon
        # with the following conditions based on the octagon geometry
        c1, c2 = self.c
        
        # Boundaries of the octagon
        upper_bound = c2 + self.a
        lower_bound = c2 - self.a
        left_bound = c1 - self.a
        right_bound = c1 + self.a
        
        # Diagonal boundaries
        upper_left = lambda x1: c2 + (x1 - (c1 - self.b)) * (self.a - self.b) / self.b
        upper_right = lambda x1: c2 + ((c1 + self.b) - x1) * (self.a - self.b) / self.b
        lower_left = lambda x1: c2 - ((c1 - self.b) - x1) * (self.a - self.b) / self.b
        lower_right = lambda x1: c2 - (x1 - (c1 + self.b)) * (self.a - self.b) / self.b
        
        x1, x2 = x
        
        # Check vertical and horizontal boundaries
        if x1 < left_bound or x1 > right_bound or x2 < lower_bound or x2 > upper_bound:
            return False
        
        # Check diagonal boundaries
        if c1 - self.b <= x1 <= c1 and x2 > upper_left(x1):
            return False
        if c1 <= x1 <= c1 + self.b and x2 > upper_right(x1):
            return False
        if c1 - self.b <= x1 <= c1 and x2 < lower_left(x1):
            return False
        if c1 <= x1 <= c1 + self.b and x2 < lower_right(x1):
            return False
        
        return True
    
    def compute_expected_profit(self, num_samples=100000):
        """
        Compute the expected profit of the market maker on uniform distribution.
        
        Args:
            num_samples: Number of samples to use for the Monte Carlo estimation
            
        Returns:
            float: Expected profit
        """
        # Generate samples from uniform distribution [0,1]^2
        samples = np.random.uniform(0, 1, (num_samples, 2))
        
        total_profit = 0
        
        for sample in samples:
            x = (sample[0], sample[1])
            alloc, payment = self.get_allocation_and_payment(x)
            
            # Update belief based on linear model
            updated_belief = tuple(self.lambda_val * c_i + (1 - self.lambda_val) * x_i 
                                  for c_i, x_i in zip(self.c, x))
            
            # Compute profit: payment - allocation·updated_belief
            profit = payment - (alloc[0] * updated_belief[0] + alloc[1] * updated_belief[1])
            total_profit += profit
        
        return total_profit / num_samples
    
    def plot_allocation_regions(self, resolution=100):
        """
        Plot the allocation regions for the market maker.
        
        Args:
            resolution: Grid resolution for the plot
        """
        # Create a grid of points
        x1 = np.linspace(0, 1, resolution)
        x2 = np.linspace(0, 1, resolution)
        X1, X2 = np.meshgrid(x1, x2)
        
        # Compute allocations for each point
        alloc1 = np.zeros_like(X1)
        alloc2 = np.zeros_like(X2)
        payments = np.zeros_like(X1)
        
        for i in range(resolution):
            for j in range(resolution):
                x = (X1[i, j], X2[i, j])
                alloc, payment = self.get_allocation_and_payment(x)
                alloc1[i, j] = alloc[0]
                alloc2[i, j] = alloc[1]
                payments[i, j] = payment
        
        # Create a figure with 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot allocation for good 1
        im1 = axes[0].imshow(alloc1, origin='lower', extent=[0, 1, 0, 1], 
                            cmap='viridis', vmin=-1, vmax=1)
        axes[0].set_title('Allocation for Good 1')
        axes[0].set_xlabel('x₁ (Trader Value for Good 1)')
        axes[0].set_ylabel('x₂ (Trader Value for Good 2)')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot allocation for good 2
        im2 = axes[1].imshow(alloc2, origin='lower', extent=[0, 1, 0, 1], 
                            cmap='viridis', vmin=-1, vmax=1)
        axes[1].set_title('Allocation for Good 2')
        axes[1].set_xlabel('x₁ (Trader Value for Good 1)')
        axes[1].set_ylabel('x₂ (Trader Value for Good 2)')
        plt.colorbar(im2, ax=axes[1])
        
        # Plot payment
        im3 = axes[2].imshow(payments, origin='lower', extent=[0, 1, 0, 1], 
                            cmap='coolwarm')
        axes[2].set_title('Payment')
        axes[2].set_xlabel('x₁ (Trader Value for Good 1)')
        axes[2].set_ylabel('x₂ (Trader Value for Good 2)')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        plt.show()
        
        # Add a 3D plot for the utility function
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        utility = np.zeros_like(X1)
        for i in range(resolution):
            for j in range(resolution):
                x = (X1[i, j], X2[i, j])
                alloc, payment = self.get_allocation_and_payment(x)
                utility[i, j] = alloc[0] * x[0] + alloc[1] * x[1] - payment
        
        surf = ax.plot_surface(X1, X2, utility, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_title('Trader Utility Function')
        ax.set_xlabel('x₁ (Trader Value for Good 1)')
        ax.set_ylabel('x₂ (Trader Value for Good 2)')
        ax.set_zlabel('Utility')
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.show()
        
    def compare_with_separate_market_making(self, num_samples=100000):
        """
        Compare the profit of the optimal market maker with a market maker
        that makes markets for each good separately.
        
        Args:
            num_samples: Number of samples for Monte Carlo estimation
            
        Returns:
            tuple: (optimal_profit, separate_profit, improvement_percentage)
        """
        # Compute optimal profit
        optimal_profit = self.compute_expected_profit(num_samples)
        
        # Compute profit for separate market making
        # Using the single-good optimal prices from Section 4
        samples = np.random.uniform(0, 1, (num_samples, 2))
        separate_profit = 0
        
        # Single-good optimal prices based on Section 4
        buy_price = (1 + self.lambda_val * self.c[0]) / (self.lambda_val + 1)
        sell_price = (self.lambda_val * self.c[0]) / (self.lambda_val + 1)
        
        for sample in samples:
            x = (sample[0], sample[1])
            profit_good1 = 0
            profit_good2 = 0
            
            # Good 1
            if x[0] > buy_price:
                # Trader buys good 1
                updated_belief1 = self.lambda_val * self.c[0] + (1 - self.lambda_val) * x[0]
                profit_good1 = buy_price - updated_belief1
            elif x[0] < sell_price:
                # Trader sells good 1
                updated_belief1 = self.lambda_val * self.c[0] + (1 - self.lambda_val) * x[0]
                profit_good1 = sell_price - updated_belief1
            
            # Good 2 (symmetric)
            if x[1] > buy_price:
                # Trader buys good 2
                updated_belief2 = self.lambda_val * self.c[1] + (1 - self.lambda_val) * x[1]
                profit_good2 = buy_price - updated_belief2
            elif x[1] < sell_price:
                # Trader sells good 2
                updated_belief2 = self.lambda_val * self.c[1] + (1 - self.lambda_val) * x[1]
                profit_good2 = sell_price - updated_belief2
            
            separate_profit += profit_good1 + profit_good2
        
        separate_profit /= num_samples
        
        # Compute improvement percentage
        improvement = (optimal_profit - separate_profit) / separate_profit * 100
        
        return optimal_profit, separate_profit, improvement

# Function to analyze market maker properties for different values of lambda
def analyze_market_maker_properties(lambda_values=[1.0, 0.5, 0.1]):
    """
    Analyze the properties of the market maker for different values of lambda.
    
    Args:
        lambda_values: List of lambda values to analyze
    """
    # Create a single market maker instance
    mm = OptimalMarketMaker(lambda_val=1.0)
    
    # Display pricing for different combinations
    print("Price menu for λ=1.0 (no adverse selection):")
    for alloc, price in mm.menu.items():
        print(f"Allocation {alloc}: Price {price:.4f}")
    
    # Show bundling discount example
    buy_good1 = mm.menu[(1, 0)]
    sell_good2 = mm.menu[(0, -1)]
    bundle = mm.menu[(1, -1)]
    discount = buy_good1 - sell_good2 - bundle
    
    print(f"\nBundling discount example:")
    print(f"Buy good 1: {buy_good1:.4f}")
    print(f"Sell good 2: {sell_good2:.4f}")
    print(f"Buy 1 and sell 2 together: {bundle:.4f}")
    print(f"Discount for bundle: {discount:.4f}")
    
    # Plot allocation regions for different lambda values
    for lambda_val in lambda_values:
        mm = OptimalMarketMaker(lambda_val=lambda_val)
        print(f"\nExpected profit for λ={lambda_val}:")
        opt_profit, sep_profit, improvement = mm.compare_with_separate_market_making(num_samples=50000)
        print(f"Optimal market maker: {opt_profit:.6f}")
        print(f"Separate market making: {sep_profit:.6f}")
        print(f"Improvement: {improvement:.2f}%")
        
        print(f"\nPlotting allocation regions for λ={lambda_val}...")
        mm.plot_allocation_regions(resolution=50)
        
# Function to simulate trading with the market maker
def simulate_trading(num_traders=1000, lambda_val=1.0):
    """
    Simulate trading with the market maker and analyze the results.
    
    Args:
        num_traders: Number of traders to simulate
        lambda_val: Adverse selection parameter
    """
    mm = OptimalMarketMaker(lambda_val=lambda_val)
    
    # Generate trader types from uniform distribution
    trader_types = np.random.uniform(0, 1, (num_traders, 2))
    
    # Collect data on trades
    allocations = []
    payments = []
    trader_utilities = []
    mm_profits = []
    
    for trader_type in trader_types:
        x = (trader_type[0], trader_type[1])
        alloc, payment = mm.get_allocation_and_payment(x)
        
        # Calculate trader utility
        utility = alloc[0] * x[0] + alloc[1] * x[1] - payment
        
        # Calculate market maker profit
        updated_belief = tuple(lambda_val * c_i + (1 - lambda_val) * x_i 
                              for c_i, x_i in zip(mm.c, x))
        profit = payment - (alloc[0] * updated_belief[0] + alloc[1] * updated_belief[1])
        
        allocations.append(alloc)
        payments.append(payment)
        trader_utilities.append(utility)
        mm_profits.append(profit)
    
    # Count trades by type
    trade_counts = {}
    for alloc in allocations:
        if alloc in trade_counts:
            trade_counts[alloc] += 1
        else:
            trade_counts[alloc] = 1
    
    # Display trade statistics
    print(f"Trading simulation results (λ={lambda_val}):")
    print(f"Number of traders: {num_traders}")
    print("\nTrade counts by allocation:")
    for alloc, count in sorted(trade_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = count / num_traders * 100
        print(f"Allocation {alloc}: {count} trades ({percentage:.1f}%)")
    
    print("\nMarket maker statistics:")
    print(f"Total profit: {sum(mm_profits):.4f}")
    print(f"Average profit per trade: {np.mean(mm_profits):.4f}")
    print(f"Standard deviation of profit: {np.std(mm_profits):.4f}")
    
    print("\nTrader statistics:")
    print(f"Average utility: {np.mean(trader_utilities):.4f}")
    print(f"Standard deviation of utility: {np.std(trader_utilities):.4f}")
    
    # Plot distributions
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot profit distribution
    sns.histplot(mm_profits, kde=True, ax=axes[0])
    axes[0].set_title('Market Maker Profit Distribution')
    axes[0].set_xlabel('Profit')
    axes[0].set_ylabel('Frequency')
    
    # Plot utility distribution
    sns.histplot(trader_utilities, kde=True, ax=axes[1])
    axes[1].set_title('Trader Utility Distribution')
    axes[1].set_xlabel('Utility')
    axes[1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    # Plot a scatter plot of trader types colored by allocation
    allocation_map = {
        (0, 0): 0,
        (1, 0): 1,
        (0, 1): 2,
        (-1, 0): 3,
        (0, -1): 4,
        (1, 1): 5,
        (-1, -1): 6,
        (1, -1): 7,
        (-1, 1): 8
    }
    
    allocation_labels = {
        0: "No Trade",
        1: "Buy Good 1",
        2: "Buy Good 2",
        3: "Sell Good 1",
        4: "Sell Good 2",
        5: "Buy Both",
        6: "Sell Both",
        7: "Buy 1, Sell 2",
        8: "Sell 1, Buy 2"
    }
    
    allocation_codes = [allocation_map.get(alloc, 0) for alloc in allocations]
    
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(trader_types[:, 0], trader_types[:, 1], 
                         c=allocation_codes, cmap='tab10', alpha=0.7)
    
    # Add a legend
    handles, _ = scatter.legend_elements()
    legend_labels = [allocation_labels.get(i, f"Unknown {i}") 
                    for i in range(len(handles))]
    plt.legend(handles, legend_labels, title="Allocation")
    
    plt.title('Trader Types and Allocations')
    plt.xlabel('Trader Value for Good 1')
    plt.ylabel('Trader Value for Good 2')
    plt.colorbar(scatter, label='Allocation Type')
    plt.grid(True, alpha=0.3)
    plt.show()

# Run the analysis
analyze_market_maker_properties()

# Run the trading simulation
simulate_trading(num_traders=5000, lambda_val=1.0)

# Test the profit improvement claim in the paper
def test_profit_improvement():
    """
    Test the profit improvement claim in the paper.
    The paper states that the maximum improvement is 11.4% when λ = sqrt(2)/3.
    """
    lambda_val = np.sqrt(2)/3
    mm = OptimalMarketMaker(lambda_val=lambda_val)
    
    opt_profit, sep_profit, improvement = mm.compare_with_separate_market_making(num_samples=100000)
    
    print(f"Testing profit improvement for λ = sqrt(2)/3 = {lambda_val:.4f}:")
    print(f"Optimal market maker profit: {opt_profit:.6f}")
    print(f"Separate market making profit: {sep_profit:.6f}")
    print(f"Improvement: {improvement:.2f}%")
    print(f"Expected improvement from paper: 11.4%")

test_profit_improvement()