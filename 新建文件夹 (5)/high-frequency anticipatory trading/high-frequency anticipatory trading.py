import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from tqdm import tqdm

class MarketSimulation:
    def __init__(self, 
                 sigma_v=1.0,  # volatility of asset value
                 sigma_1=0.5,  # noise trading volatility at t=1
                 sigma_1plus=0.1,  # noise trading volatility at t=1+
                 sigma_2=0.5,  # noise trading volatility at t=2
                 sigma_epsilon=0.1,  # signal noise
                 theta_z=0.2,  # IT's order randomization intensity
                 num_small_ITs=2,  # number of Small-ITs 
                 num_round_trippers=2,  # number of Round-Trippers
                 gamma=0.0  # inventory aversion parameter
                ):
        
        # Market parameters
        self.sigma_v = sigma_v
        self.sigma_1 = sigma_1
        self.sigma_1plus = sigma_1plus
        self.sigma_2 = sigma_2
        self.sigma_epsilon = sigma_epsilon
        self.theta_z = theta_z
        
        # Dimensionless parameters (from paper)
        self.theta_1plus = (sigma_1plus/sigma_1)**2  # relative size of time-1+ market
        self.theta_2 = (sigma_2/sigma_1)**2  # relative size of time-2 market
        self.theta_epsilon = (sigma_epsilon/sigma_1)**2  # signal accuracy
        self.Gamma = gamma/(sigma_v/sigma_1)  # dimensionless inventory aversion
        
        # Number of traders
        self.J1 = num_small_ITs  # Small-ITs
        self.J2 = num_round_trippers  # Round-Trippers
        self.J = self.J1 + self.J2  # Total number of HFTs
        
        # Trader parameters (to be calibrated from equilibrium)
        self.A1 = None  # IT's first period trading intensity
        self.beta11 = None  # Small-IT first period trading intensity
        self.beta12 = None  # Round-Tripper first period trading intensity
        self.beta21 = None  # Small-IT second period trading coefficient
        self.beta22 = None  # Small-IT second period adjustment coefficient
        self.beta23 = None  # Small-IT second period inventory coefficient
        
        # Market prices
        self.p0 = 100.0  # initial price
        self.Lambda1 = None  # price impact at t=1
        self.Lambda1plus = None  # price impact at t=1+
        self.Lambda21 = None  # first component of price impact at t=2
        self.Lambda22 = None  # second component of price impact at t=2
        
    def calibrate_model(self, mixed_strategy=True):
        """
        Calibrate model parameters based on simplified equilibrium values from the paper
        """
        # These values would normally be computed by solving the equilibrium conditions
        # But for simplicity, we'll use reasonable approximations based on the paper
        
        # For mixed strategy
        if mixed_strategy:
            # IT parameters
            self.A1 = 1.0 / np.sqrt(1 + self.theta_z)
            
            # HFT parameters - Small-ITs
            self.beta11 = 0.5 * self.A1
            self.beta21 = 0.3 * self.A1
            self.beta22 = -0.1
            self.beta23 = -0.2
            
            # HFT parameters - Round-Trippers
            self.beta12 = 0.4 * self.A1
            
            # Price impact coefficients
            self.Lambda1 = self.A1 / (1 + self.A1**2 + self.theta_z)
            self.Lambda1plus = (self.beta11 * self.J1 + self.beta12 * self.J2) * self.A1 / (
                (self.beta11 * self.J1 + self.beta12 * self.J2)**2 * self.theta_epsilon + self.theta_1plus)
            self.Lambda22 = self.Lambda1
            self.Lambda21 = self.Lambda1plus / 2
            
        # For pure strategy
        else:
            # IT parameters
            self.A1 = 1.0
            self.theta_z = 0  # No randomization in pure strategy
            
            # HFT parameters - Small-ITs
            self.beta11 = 0.4 * self.A1
            self.beta21 = 0.25 * self.A1
            self.beta22 = -0.05
            self.beta23 = -0.3
            
            # HFT parameters - Round-Trippers
            self.beta12 = 0.35 * self.A1
            
            # Price impact coefficients
            self.Lambda1 = self.A1 / (1 + self.A1**2)
            self.Lambda1plus = (self.beta11 * self.J1 + self.beta12 * self.J2) * self.A1 / (
                (self.beta11 * self.J1 + self.beta12 * self.J2)**2 * self.theta_epsilon + self.theta_1plus)
            self.Lambda22 = self.Lambda1
            self.Lambda21 = self.Lambda1plus / 2
    
    def simulate_trading(self, num_simulations=1000):
        """
        Simulate trading between IT and HFTs
        """
        # Make sure model is calibrated
        if self.A1 is None:
            self.calibrate_model()
            
        # Arrays to store results
        v_values = np.zeros(num_simulations)
        i1_values = np.zeros(num_simulations)
        i2_values = np.zeros(num_simulations)
        x1_small_values = np.zeros((num_simulations, self.J1))
        x2_small_values = np.zeros((num_simulations, self.J1))
        x1_round_values = np.zeros((num_simulations, self.J2))
        x2_round_values = np.zeros((num_simulations, self.J2))
        p1_values = np.zeros(num_simulations)
        p1plus_values = np.zeros(num_simulations)
        p2_values = np.zeros(num_simulations)
        
        # IT profits
        IT_profit_values = np.zeros(num_simulations)
        
        # HFT profits
        small_IT_profit_values = np.zeros((num_simulations, self.J1))
        round_tripper_profit_values = np.zeros((num_simulations, self.J2))
        
        # Run simulations
        for sim in tqdm(range(num_simulations)):
            # Generate true asset value
            v = self.p0 + self.sigma_v * np.random.randn()
            v_values[sim] = v
            
            # IT's trading at t=1
            i1_deterministic = self.A1 * (v - self.p0)
            z = self.sigma_1 * np.sqrt(self.theta_z) * np.random.randn() if self.theta_z > 0 else 0
            i1 = i1_deterministic + z
            i1_values[sim] = i1
            
            # Noise trading at t=1
            u1 = self.sigma_1 * np.random.randn()
            
            # Order flow at t=1
            y1 = i1 + u1
            
            # Price at t=1
            p1 = self.p0 + self.Lambda1 * y1
            p1_values[sim] = p1
            
            # HFTs observe signal about i1
            i1_signals = np.zeros(self.J)
            for j in range(self.J):
                i1_signals[j] = i1 + self.sigma_epsilon * np.random.randn()
            
            # Expected i1 given y1
            E_i1_given_y1 = self.Lambda1 * y1
            
            # Small-ITs trading at t=1+
            for j in range(self.J1):
                x1_small_values[sim, j] = self.beta11 * (i1_signals[j] - E_i1_given_y1)
            
            # Round-Trippers trading at t=1+
            for j in range(self.J2):
                x1_round_values[sim, j] = self.beta12 * (i1_signals[j+self.J1] - E_i1_given_y1)
            
            # Noise trading at t=1+
            u1plus = self.sigma_1plus * np.random.randn()
            
            # Order flow at t=1+
            y1plus = np.sum(x1_small_values[sim]) + np.sum(x1_round_values[sim]) + u1plus
            
            # Price at t=1+
            p1plus = p1 + self.Lambda1plus * y1plus
            p1plus_values[sim] = p1plus
            
            # IT's trading at t=2
            # In this simplified model, we'll assume IT's second trade is proportional to private information
            alpha21 = 1 / (2 * self.Lambda22)
            alpha22 = -(1 + self.A1**2 + self.theta_z) / 2
            
            i2 = alpha21 * (v - p1) + alpha22 * (i1 - E_i1_given_y1)
            i2_values[sim] = i2
            
            # Small-ITs trading at t=2
            for j in range(self.J1):
                signal_diff = i1_signals[j] - E_i1_given_y1
                others_flow = np.sum(x1_small_values[sim]) + np.sum(x1_round_values[sim]) - x1_small_values[sim, j]
                x2_small_values[sim, j] = (
                    self.beta21 * signal_diff + 
                    self.beta22 * (others_flow + u1plus) + 
                    self.beta23 * x1_small_values[sim, j]
                )
            
            # Round-Trippers trading at t=2 (reverse their positions)
            for j in range(self.J2):
                x2_round_values[sim, j] = -x1_round_values[sim, j]
            
            # Noise trading at t=2
            u2 = self.sigma_2 * np.random.randn()
            
            # Order flow at t=2
            y2 = i2 + np.sum(x2_small_values[sim]) + np.sum(x2_round_values[sim]) + u2
            
            # Price at t=2
            p2 = p1 + self.Lambda21 * y1plus + self.Lambda22 * y2
            p2_values[sim] = p2
            
            # Calculate profits
            # IT profit
            IT_profit = i1 * (v - p1) + i2 * (v - p2)
            IT_profit_values[sim] = IT_profit
            
            # Small-IT profits
            for j in range(self.J1):
                profit1 = x1_small_values[sim, j] * (v - p1plus)
                profit2 = x2_small_values[sim, j] * (v - p2)
                inventory_penalty = self.Gamma * (x1_small_values[sim, j] + x2_small_values[sim, j])**2
                small_IT_profit_values[sim, j] = profit1 + profit2 - inventory_penalty
            
            # Round-Tripper profits
            for j in range(self.J2):
                profit1 = x1_round_values[sim, j] * (v - p1plus)
                profit2 = x2_round_values[sim, j] * (v - p2)
                # Round-Trippers have infinite inventory aversion so x2 = -x1
                round_tripper_profit_values[sim, j] = profit1 + profit2
            
        # Prepare results
        results = {
            'v': v_values,
            'i1': i1_values,
            'i2': i2_values,
            'x1_small': x1_small_values,
            'x2_small': x2_small_values,
            'x1_round': x1_round_values,
            'x2_round': x2_round_values,
            'p1': p1_values,
            'p1plus': p1plus_values,
            'p2': p2_values,
            'IT_profit': IT_profit_values,
            'small_IT_profit': small_IT_profit_values,
            'round_tripper_profit': round_tripper_profit_values
        }
        
        return results

    def analyze_results(self, results):
        """
        Analyze simulation results
        """
        # Calculate average profits
        avg_IT_profit = np.mean(results['IT_profit'])
        avg_small_IT_profit = np.mean(np.mean(results['small_IT_profit'], axis=1))
        avg_round_tripper_profit = np.mean(np.mean(results['round_tripper_profit'], axis=1))
        
        print(f"Average IT Profit: {avg_IT_profit:.4f}")
        print(f"Average Small-IT Profit: {avg_small_IT_profit:.4f}")
        print(f"Average Round-Tripper Profit: {avg_round_tripper_profit:.4f}")
        
        # Plot price evolution
        plt.figure(figsize=(12, 6))
        plt.plot(np.arange(3), 
                 [np.mean(results['p1']), np.mean(results['p1plus']), np.mean(results['p2'])], 
                 'o-', linewidth=2)
        plt.xticks([0, 1, 2], ['t=1', 't=1+', 't=2'])
        plt.ylabel('Average Price')
        plt.title('Price Evolution')
        plt.grid(True)
        plt.show()
        
        # Plot profit distributions
        plt.figure(figsize=(12, 6))
        plt.hist(results['IT_profit'], bins=30, alpha=0.5, label='IT')
        plt.hist(np.mean(results['small_IT_profit'], axis=1), bins=30, alpha=0.5, label='Small-IT')
        plt.hist(np.mean(results['round_tripper_profit'], axis=1), bins=30, alpha=0.5, label='Round-Tripper')
        plt.xlabel('Profit')
        plt.ylabel('Frequency')
        plt.title('Profit Distributions')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot trading volumes
        plt.figure(figsize=(12, 6))
        avg_volumes = [
            np.mean(np.abs(results['i1'])),
            np.mean(np.abs(np.sum(results['x1_small'], axis=1)) + np.abs(np.sum(results['x1_round'], axis=1))),
            np.mean(np.abs(results['i2']) + np.abs(np.sum(results['x2_small'], axis=1)) + np.abs(np.sum(results['x2_round'], axis=1)))
        ]
        plt.bar([0, 1, 2], avg_volumes)
        plt.xticks([0, 1, 2], ['t=1', 't=1+', 't=2'])
        plt.ylabel('Average Trading Volume')
        plt.title('Trading Volumes')
        plt.grid(True)
        plt.show()
        
        return {
            'avg_IT_profit': avg_IT_profit,
            'avg_small_IT_profit': avg_small_IT_profit,
            'avg_round_tripper_profit': avg_round_tripper_profit
        }

    def run_parameter_study(self):
        """
        Study how changing parameters affects profits
        """
        # Parameters to vary
        theta_z_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]  # IT's randomization intensity
        theta_epsilon_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]  # Signal accuracy
        theta_1plus_values = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]  # Size of high-speed market
        
        # Study effect of randomization intensity
        it_profits_z = []
        small_it_profits_z = []
        round_tripper_profits_z = []
        
        for theta_z in theta_z_values:
            self.theta_z = theta_z
            self.calibrate_model(mixed_strategy=(theta_z > 0))
            results = self.simulate_trading(num_simulations=500)
            
            it_profits_z.append(np.mean(results['IT_profit']))
            small_it_profits_z.append(np.mean(np.mean(results['small_IT_profit'], axis=1)))
            round_tripper_profits_z.append(np.mean(np.mean(results['round_tripper_profit'], axis=1)))
        
        # Study effect of signal accuracy
        it_profits_epsilon = []
        small_it_profits_epsilon = []
        round_tripper_profits_epsilon = []
        
        self.theta_z = 0.2  # Reset randomization
        
        for theta_epsilon in theta_epsilon_values:
            self.theta_epsilon = theta_epsilon
            self.calibrate_model()
            results = self.simulate_trading(num_simulations=500)
            
            it_profits_epsilon.append(np.mean(results['IT_profit']))
            small_it_profits_epsilon.append(np.mean(np.mean(results['small_IT_profit'], axis=1)))
            round_tripper_profits_epsilon.append(np.mean(np.mean(results['round_tripper_profit'], axis=1)))
        
        # Study effect of high-speed market size
        it_profits_1plus = []
        small_it_profits_1plus = []
        round_tripper_profits_1plus = []
        
        self.theta_epsilon = 0.1  # Reset signal accuracy
        
        for theta_1plus in theta_1plus_values:
            self.theta_1plus = theta_1plus
            self.sigma_1plus = np.sqrt(theta_1plus) * self.sigma_1
            self.calibrate_model()
            results = self.simulate_trading(num_simulations=500)
            
            it_profits_1plus.append(np.mean(results['IT_profit']))
            small_it_profits_1plus.append(np.mean(np.mean(results['small_IT_profit'], axis=1)))
            round_tripper_profits_1plus.append(np.mean(np.mean(results['round_tripper_profit'], axis=1)))
        
        # Plot results
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot effect of randomization
        axes[0].plot(theta_z_values, it_profits_z, 'o-', linewidth=2, label='IT')
        axes[0].plot(theta_z_values, small_it_profits_z, 's-', linewidth=2, label='Small-IT')
        axes[0].plot(theta_z_values, round_tripper_profits_z, '^-', linewidth=2, label='Round-Tripper')
        axes[0].set_xlabel('θz (Randomization Intensity)')
        axes[0].set_ylabel('Average Profit')
        axes[0].set_title('Effect of Randomization')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot effect of signal accuracy
        axes[1].plot(theta_epsilon_values, it_profits_epsilon, 'o-', linewidth=2, label='IT')
        axes[1].plot(theta_epsilon_values, small_it_profits_epsilon, 's-', linewidth=2, label='Small-IT')
        axes[1].plot(theta_epsilon_values, round_tripper_profits_epsilon, '^-', linewidth=2, label='Round-Tripper')
        axes[1].set_xlabel('θε (Signal Noise)')
        axes[1].set_ylabel('Average Profit')
        axes[1].set_title('Effect of Signal Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot effect of high-speed market size
        axes[2].plot(theta_1plus_values, it_profits_1plus, 'o-', linewidth=2, label='IT')
        axes[2].plot(theta_1plus_values, small_it_profits_1plus, 's-', linewidth=2, label='Small-IT')
        axes[2].plot(theta_1plus_values, round_tripper_profits_1plus, '^-', linewidth=2, label='Round-Tripper')
        axes[2].set_xlabel('θ1+ (High-Speed Market Size)')
        axes[2].set_ylabel('Average Profit')
        axes[2].set_title('Effect of High-Speed Market Size')
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return {
            'theta_z': {
                'values': theta_z_values,
                'it_profits': it_profits_z,
                'small_it_profits': small_it_profits_z,
                'round_tripper_profits': round_tripper_profits_z
            },
            'theta_epsilon': {
                'values': theta_epsilon_values,
                'it_profits': it_profits_epsilon,
                'small_it_profits': small_it_profits_epsilon,
                'round_tripper_profits': round_tripper_profits_epsilon
            },
            'theta_1plus': {
                'values': theta_1plus_values,
                'it_profits': it_profits_1plus,
                'small_it_profits': small_it_profits_1plus,
                'round_tripper_profits': round_tripper_profits_1plus
            }
        }

    def compare_strategies(self):
        """
        Compare different strategic scenarios
        """
        # Define different market scenarios
        scenarios = {
            'Pure Strategy (no randomization)': {
                'mixed_strategy': False,
                'J1': self.J1,
                'J2': self.J2
            },
            'Mixed Strategy (with randomization)': {
                'mixed_strategy': True,
                'J1': self.J1,
                'J2': self.J2
            },
            'Only Small-ITs': {
                'mixed_strategy': True,
                'J1': self.J1 + self.J2,
                'J2': 0
            },
            'Only Round-Trippers': {
                'mixed_strategy': True,
                'J1': 0,
                'J2': self.J1 + self.J2
            }
        }
        
        # Store results
        scenario_results = {}
        
        # Run each scenario
        for name, params in scenarios.items():
            # Store original values to restore later
            orig_J1, orig_J2 = self.J1, self.J2
            
            # Set scenario parameters
            self.J1 = params['J1']
            self.J2 = params['J2']
            
            # Calibrate and run simulation
            self.calibrate_model(mixed_strategy=params['mixed_strategy'])
            results = self.simulate_trading(num_simulations=500)
            
            # Store results
            scenario_results[name] = {
                'IT_profit': np.mean(results['IT_profit']),
                'IT_profit_std': np.std(results['IT_profit']),
                'HFT_profit': np.mean(np.concatenate([
                    np.mean(results['small_IT_profit'], axis=1) if self.J1 > 0 else np.array([]),
                    np.mean(results['round_tripper_profit'], axis=1) if self.J2 > 0 else np.array([])
                ])),
                'HFT_profit_std': np.std(np.concatenate([
                    np.mean(results['small_IT_profit'], axis=1) if self.J1 > 0 else np.array([]),
                    np.mean(results['round_tripper_profit'], axis=1) if self.J2 > 0 else np.array([])
                ])),
                'small_IT_profit': np.mean(np.mean(results['small_IT_profit'], axis=1)) if self.J1 > 0 else 0,
                'round_tripper_profit': np.mean(np.mean(results['round_tripper_profit'], axis=1)) if self.J2 > 0 else 0
            }
            
            # Restore original values
            self.J1, self.J2 = orig_J1, orig_J2
        
        # Plot comparison
        plt.figure(figsize=(14, 7))
        
        # IT profits
        plt.subplot(1, 2, 1)
        scenario_names = list(scenario_results.keys())
        it_profits = [scenario_results[name]['IT_profit'] for name in scenario_names]
        it_profit_errors = [scenario_results[name]['IT_profit_std'] / np.sqrt(500) for name in scenario_names]
        
        plt.bar(scenario_names, it_profits, yerr=it_profit_errors, capsize=5)
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Average Profit')
        plt.title('IT Profits by Scenario')
        plt.grid(True)
        
        # HFT profits
        plt.subplot(1, 2, 2)
        hft_profits = []
        hft_types = []
        scenario_labels = []
        
        for name in scenario_names:
            scenario = scenario_results[name]
            
            if scenario['small_IT_profit'] > 0:
                hft_profits.append(scenario['small_IT_profit'])
                hft_types.append('Small-IT')
                scenario_labels.append(name)
            
            if scenario['round_tripper_profit'] > 0:
                hft_profits.append(scenario['round_tripper_profit'])
                hft_types.append('Round-Tripper')
                scenario_labels.append(name)
        
        # Create HFT profit plot
        x_pos = np.arange(len(hft_profits))
        plt.bar(x_pos, hft_profits)
        plt.xticks(x_pos, [f"{label}\n({type})" for label, type in zip(scenario_labels, hft_types)], 
                  rotation=45, ha='right')
        plt.ylabel('Average Profit')
        plt.title('HFT Profits by Scenario and Type')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return scenario_results
    
    def run_hft_composition_study(self):
        """
        Study how changing the composition of HFTs affects profits
        """
        total_HFTs = self.J1 + self.J2
        small_IT_proportions = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        
        # Store results
        it_profits = []
        small_it_profits = []
        round_tripper_profits = []
        
        # Run simulations for different compositions
        for prop in small_IT_proportions:
            # Calculate numbers of each type
            self.J1 = int(total_HFTs * prop)
            self.J2 = total_HFTs - self.J1
            
            # Calibrate model and run simulation
            self.calibrate_model(mixed_strategy=True)
            results = self.simulate_trading(num_simulations=500)
            
            # Store results
            it_profits.append(np.mean(results['IT_profit']))
            small_it_profits.append(np.mean(np.mean(results['small_IT_profit'], axis=1)) if self.J1 > 0 else 0)
            round_tripper_profits.append(np.mean(np.mean(results['round_tripper_profit'], axis=1)) if self.J2 > 0 else 0)
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(small_IT_proportions, it_profits, 'o-', linewidth=2, label='IT')
        plt.plot(small_IT_proportions, small_it_profits, 's-', linewidth=2, label='Small-IT')
        plt.plot(small_IT_proportions, round_tripper_profits, '^-', linewidth=2, label='Round-Tripper')
        plt.xlabel('Proportion of Small-ITs')
        plt.ylabel('Average Profit')
        plt.title(f'Effect of HFT Composition (Total HFTs = {total_HFTs})')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        return {
            'proportions': small_IT_proportions,
            'it_profits': it_profits,
            'small_it_profits': small_it_profits,
            'round_tripper_profits': round_tripper_profits
        }
    
    def implement_trading_strategy(self):
        """
        Implement a trading strategy based on the paper's findings
        """
        # This is a practical trading strategy an investor might implement
        # based on the insights from the paper
        
        # For an informed trader who wants to minimize anticipatory trading impact:
        # 1. Randomize order size to confuse HFTs
        # 2. Split orders into two parts with the optimal ratio
        # 3. Trade when high-speed noise trading is most active
        
        # Let's simulate this strategy vs a naive strategy
        
        # Parameters
        num_simulations = 500
        order_size = 1.0  # Normalized order size
        
        # Strategies
        strategies = {
            'Naive': {
                'randomize': False,
                'split_ratio': 0.5,  # Equal splitting
                'high_speed_market': 'low'  # Trade when high-speed market is quiet
            },
            'Smart': {
                'randomize': True,
                'split_ratio': 0.3,  # Optimal splitting from paper
                'high_speed_market': 'high'  # Trade when high-speed market is active
            }
        }
        
        # Results
        strategy_results = {}
        
        for name, params in strategies.items():
            # Set parameters based on strategy
            self.theta_z = 0.3 if params['randomize'] else 0.0
            
            # Set high-speed market size
            if params['high_speed_market'] == 'high':
                self.theta_1plus = 1.0
                self.sigma_1plus = np.sqrt(self.theta_1plus) * self.sigma_1
            else:
                self.theta_1plus = 0.1
                self.sigma_1plus = np.sqrt(self.theta_1plus) * self.sigma_1
            
            # Calibrate model
            self.calibrate_model(mixed_strategy=params['randomize'])
            
            # Modify IT's trading parameters
            # A1 represents first period trading intensity
            original_A1 = self.A1
            self.A1 = original_A1 * params['split_ratio'] / 0.5  # Adjust for split ratio
            
            # Run simulation
            results = self.simulate_trading(num_simulations=num_simulations)
            
            # Store IT profit
            strategy_results[name] = {
                'profit': np.mean(results['IT_profit']),
                'profit_std': np.std(results['IT_profit']),
                'profit_by_sim': results['IT_profit']
            }
            
            # Restore original A1
            self.A1 = original_A1
        
        # Plot strategy comparison
        plt.figure(figsize=(14, 6))
        
        # Bar chart of average profits
        plt.subplot(1, 2, 1)
        plt.bar(list(strategy_results.keys()), 
                [strategy_results[name]['profit'] for name in strategy_results],
                yerr=[strategy_results[name]['profit_std'] / np.sqrt(num_simulations) for name in strategy_results],
                capsize=5)
        plt.ylabel('Average Profit')
        plt.title('IT Strategy Comparison')
        plt.grid(True)
        
        # Profit distribution
        plt.subplot(1, 2, 2)
        for name in strategy_results:
            sns.kdeplot(strategy_results[name]['profit_by_sim'], label=name)
        plt.xlabel('Profit')
        plt.ylabel('Density')
        plt.title('IT Profit Distribution')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        return strategy_results

# Run the simulation
if __name__ == "__main__":
    # Create a simulation with default parameters
    sim = MarketSimulation(
        sigma_v=1.0,        # Asset value volatility
        sigma_1=0.5,        # Noise trading volatility at t=1
        sigma_1plus=0.1,    # Noise trading volatility at t=1+
        sigma_2=0.5,        # Noise trading volatility at t=2
        sigma_epsilon=0.1,  # Signal noise
        theta_z=0.2,        # IT's order randomization intensity
        num_small_ITs=2,    # Number of Small-ITs
        num_round_trippers=2 # Number of Round-Trippers
    )
    
    # Calibrate the model
    sim.calibrate_model(mixed_strategy=True)
    
    # Run simulations
    print("Running basic simulation...")
    results = sim.simulate_trading(num_simulations=1000)
    
    # Analyze results
    print("\nAnalyzing simulation results:")
    sim.analyze_results(results)
    
    # Run parameter study
    print("\nRunning parameter study...")
    param_study = sim.run_parameter_study()
    
    # Compare strategies
    print("\nComparing different market scenarios:")
    scenario_results = sim.compare_strategies()
    
    # Study HFT composition
    print("\nStudying effect of HFT composition:")
    composition_study = sim.run_hft_composition_study()
    
    # Implement trading strategy
    print("\nImplementing practical trading strategy:")
    strategy_results = sim.implement_trading_strategy()