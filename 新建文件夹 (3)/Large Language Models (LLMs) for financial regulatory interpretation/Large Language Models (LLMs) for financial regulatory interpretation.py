import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class BaselIIICalculator:
    """
    A class to calculate capital requirements for market risk under Basel III's Standardized Approach
    based on the framework described in the paper.
    """
    
    def __init__(self):
        # Risk weights for different asset classes based on Basel III
        self.equity_risk_weights = {
            1: 0.30,  # Consumer goods and services, transportation and storage, etc.
            2: 0.35,  # Telecommunications, industrials
            3: 0.40,  # Basic materials, energy, agriculture
            4: 0.50,  # Financials
            5: 0.70,  # Technology, health care
            6: 0.35,  # Telecommunications
            7: 0.40,  # Energy, Oil and Gas
            8: 0.35,  # Other advanced economy sectors
            9: 0.50,  # Emerging market economy sectors
            10: 0.70,  # Volatility indices
            11: 0.15,  # Qualified indices
            12: 0.50   # Other sectors/unclassified
        }
        
        # Correlation matrix for equity buckets
        self.equity_correlations = np.ones((12, 12)) * 0.15  # Default correlation of 15%
        np.fill_diagonal(self.equity_correlations, 1.0)  # Correlation of 100% within the same bucket
        
        # Risk weights for general interest rate risk
        self.girr_risk_weights = {
            '0-1y': 0.0100,
            '1-3y': 0.0100,
            '3-5y': 0.0125,
            '5-10y': 0.0150,
            '>10y': 0.0175
        }
        
        # Correlation matrix for interest rate buckets
        self.girr_correlations = np.ones((5, 5)) * 0.25  # Default correlation of 25%
        np.fill_diagonal(self.girr_correlations, 1.0)  # Correlation of 100% within the same bucket
        
        # Risk weights for FX risk
        self.fx_risk_weight = 0.15  # 15% for all FX exposures
        
        # Risk weights for commodity risk
        self.commodity_risk_weights = {
            1: 0.30,  # Energy - including electricity
            2: 0.35,  # Metals - including precious and non-precious metals
            3: 0.60,  # Agricultural - including grains, livestock, forestry, etc.
            4: 0.80   # Other - including industrial minerals, bulk shipping, etc.
        }
        
        # Correlation matrix for commodity buckets
        self.commodity_correlations = np.ones((4, 4)) * 0.20  # Default correlation of 20%
        np.fill_diagonal(self.commodity_correlations, 1.0)  # Correlation of 100% within the same bucket
    
    def calculate_equity_delta_sensitivities(self, equity_holdings):
        """
        Calculate delta sensitivities for equity positions.
        
        Parameters:
        -----------
        equity_holdings : list of dict
            List of dictionaries containing equity position information.
            Each dictionary should have:
            - 'name': Name of the equity
            - 'bucket': Bucket number (1-12)
            - 'market_value': Current market value
            
        Returns:
        --------
        dict
            Dictionary with equity names as keys and sensitivities as values
        """
        sensitivities = {}
        
        for equity in equity_holdings:
            # Delta sensitivity is defined as the change in market value for a 1% change in equity price
            # As per Basel III paragraph 67, this is calculated as:
            # sk = (Vi(1.01 * EQk) - Vi(EQk)) / 0.01
            # For a direct equity position, this simplifies to the market value
            sensitivities[equity['name']] = {
                'sensitivity': equity['market_value'],
                'bucket': equity['bucket']
            }
            
        return sensitivities
    
    def calculate_girr_delta_sensitivities(self, bond_holdings):
        """
        Calculate delta sensitivities for General Interest Rate Risk (GIRR) positions.
        
        Parameters:
        -----------
        bond_holdings : list of dict
            List of dictionaries containing bond position information.
            Each dictionary should have:
            - 'name': Name of the bond
            - 'maturity': Maturity of the bond in years
            - 'market_value': Current market value
            - 'duration': Modified duration of the bond
            
        Returns:
        --------
        dict
            Dictionary with bond names as keys and sensitivities as values
        """
        sensitivities = {}
        
        # Map maturities to buckets
        bucket_mapping = {
            (0, 1): '0-1y',
            (1, 3): '1-3y',
            (3, 5): '3-5y',
            (5, 10): '5-10y',
            (10, float('inf')): '>10y'
        }
        
        for bond in bond_holdings:
            # Determine the bucket based on maturity
            bucket = None
            for (lower, upper), bucket_name in bucket_mapping.items():
                if lower <= bond['maturity'] < upper:
                    bucket = bucket_name
                    break
            
            if bucket is None:
                bucket = '>10y'  # Default to the longest maturity bucket
            
            # Delta sensitivity for interest rate risk is defined as the change in market value
            # for a 1 basis point parallel shift in the yield curve
            # This is approximated by duration times market value times 0.0001
            sensitivity = bond['market_value'] * bond['duration'] * 0.0001
            
            sensitivities[bond['name']] = {
                'sensitivity': sensitivity,
                'bucket': bucket
            }
            
        return sensitivities
    
    def calculate_fx_delta_sensitivities(self, fx_holdings):
        """
        Calculate delta sensitivities for FX positions.
        
        Parameters:
        -----------
        fx_holdings : list of dict
            List of dictionaries containing FX position information.
            Each dictionary should have:
            - 'name': Name of the currency pair
            - 'base_currency': Base currency
            - 'quote_currency': Quote currency
            - 'position': Position size in base currency
            - 'exchange_rate': Current exchange rate
            
        Returns:
        --------
        dict
            Dictionary with currency names as keys and sensitivities as values
        """
        sensitivities = {}
        
        for fx in fx_holdings:
            # Convert position to USD (or reporting currency)
            position_value = fx['position'] * fx['exchange_rate']
            
            # For FX risk, sensitivity is the market value of the position
            sensitivity = position_value
            
            sensitivities[fx['name']] = {
                'sensitivity': sensitivity,
                'base_currency': fx['base_currency'],
                'quote_currency': fx['quote_currency']
            }
            
        return sensitivities
    
    def calculate_commodity_delta_sensitivities(self, commodity_holdings):
        """
        Calculate delta sensitivities for commodity positions.
        
        Parameters:
        -----------
        commodity_holdings : list of dict
            List of dictionaries containing commodity position information.
            Each dictionary should have:
            - 'name': Name of the commodity
            - 'bucket': Bucket number (1-4)
            - 'quantity': Quantity of the commodity
            - 'price': Price per unit
            
        Returns:
        --------
        dict
            Dictionary with commodity names as keys and sensitivities as values
        """
        sensitivities = {}
        
        for commodity in commodity_holdings:
            # Market value is quantity times price
            market_value = commodity['quantity'] * commodity['price']
            
            # For commodity risk, sensitivity is the market value of the position
            sensitivity = market_value
            
            sensitivities[commodity['name']] = {
                'sensitivity': sensitivity,
                'bucket': commodity['bucket']
            }
            
        return sensitivities
    
    def calculate_equity_delta_risk_charge(self, equity_sensitivities):
        """
        Calculate delta risk charge for equity risk.
        
        Parameters:
        -----------
        equity_sensitivities : dict
            Dictionary with equity sensitivities.
            
        Returns:
        --------
        float
            Delta risk charge for equity risk
        """
        # Step 1: Calculate weighted sensitivities
        weighted_sensitivities = {}
        for name, data in equity_sensitivities.items():
            bucket = data['bucket']
            sensitivity = data['sensitivity']
            risk_weight = self.equity_risk_weights[bucket]
            
            weighted_sensitivities[name] = {
                'bucket': bucket,
                'weighted_sensitivity': sensitivity * risk_weight
            }
        
        # Step 2: Calculate risk positions for each bucket
        bucket_risk_positions = {}
        bucket_sensitivities = {}
        
        # Group sensitivities by bucket
        for name, data in weighted_sensitivities.items():
            bucket = data['bucket']
            ws = data['weighted_sensitivity']
            
            if bucket not in bucket_sensitivities:
                bucket_sensitivities[bucket] = {}
            
            bucket_sensitivities[bucket][name] = ws
        
        # Calculate risk position for each bucket
        for bucket, sensitivities in bucket_sensitivities.items():
            # Sum of weighted sensitivities in the bucket
            Sb = sum(sensitivities.values())
            bucket_sensitivities[bucket]['sum'] = Sb
            
            # Calculate the risk position Kb
            sum_squared = sum(ws**2 for ws in sensitivities.values() if isinstance(ws, (int, float)))
            
            # Calculate the cross-terms
            cross_terms = 0
            sensitivity_list = [(name, ws) for name, ws in sensitivities.items() if name != 'sum']
            
            for i, (name1, ws1) in enumerate(sensitivity_list):
                for name2, ws2 in sensitivity_list[i+1:]:
                    # Apply correlation within the bucket (assumed to be 1.0 for simplicity)
                    cross_terms += 1.0 * ws1 * ws2
            
            # Kb = sqrt(sum of squared weighted sensitivities + cross terms)
            Kb = np.sqrt(sum_squared + 2 * cross_terms)
            bucket_risk_positions[bucket] = Kb
        
        # Step 3: Calculate delta risk charge
        # Sum of squared bucket risk positions
        sum_squared_Kb = sum(Kb**2 for Kb in bucket_risk_positions.values())
        
        # Calculate cross-bucket terms
        cross_bucket_terms = 0
        bucket_list = list(bucket_risk_positions.keys())
        
        for i, bucket1 in enumerate(bucket_list):
            for bucket2 in bucket_list[i+1:]:
                correlation = self.equity_correlations[bucket1-1, bucket2-1]
                Sb1 = bucket_sensitivities[bucket1]['sum']
                Sb2 = bucket_sensitivities[bucket2]['sum']
                
                cross_bucket_terms += correlation * Sb1 * Sb2
        
        # Delta = sqrt(sum of squared bucket risk positions + cross-bucket terms)
        delta_risk_charge = np.sqrt(sum_squared_Kb + cross_bucket_terms)
        
        return delta_risk_charge
    
    def calculate_girr_delta_risk_charge(self, girr_sensitivities):
        """
        Calculate delta risk charge for General Interest Rate Risk (GIRR).
        
        Parameters:
        -----------
        girr_sensitivities : dict
            Dictionary with GIRR sensitivities.
            
        Returns:
        --------
        float
            Delta risk charge for GIRR
        """
        # Step 1: Group sensitivities by bucket
        bucket_sensitivities = {}
        
        for name, data in girr_sensitivities.items():
            bucket = data['bucket']
            sensitivity = data['sensitivity']
            
            if bucket not in bucket_sensitivities:
                bucket_sensitivities[bucket] = {}
            
            bucket_sensitivities[bucket][name] = sensitivity
        
        # Step 2: Calculate weighted sensitivities
        weighted_bucket_sensitivities = {}
        
        for bucket, sensitivities in bucket_sensitivities.items():
            weighted_bucket_sensitivities[bucket] = {}
            
            for name, sensitivity in sensitivities.items():
                risk_weight = self.girr_risk_weights[bucket]
                weighted_bucket_sensitivities[bucket][name] = sensitivity * risk_weight
        
        # Step 3: Calculate risk positions for each bucket
        bucket_risk_positions = {}
        
        for bucket, sensitivities in weighted_bucket_sensitivities.items():
            # Sum of weighted sensitivities in the bucket
            Sb = sum(sensitivities.values())
            weighted_bucket_sensitivities[bucket]['sum'] = Sb
            
            # Calculate the risk position Kb
            sum_squared = sum(ws**2 for ws in sensitivities.values() if isinstance(ws, (int, float)))
            
            # Calculate the cross-terms
            cross_terms = 0
            sensitivity_list = [(name, ws) for name, ws in sensitivities.items() if name != 'sum']
            
            for i, (name1, ws1) in enumerate(sensitivity_list):
                for name2, ws2 in sensitivity_list[i+1:]:
                    # Apply correlation within the bucket (assumed to be 1.0 for simplicity)
                    cross_terms += 1.0 * ws1 * ws2
            
            # Kb = sqrt(sum of squared weighted sensitivities + cross terms)
            Kb = np.sqrt(sum_squared + 2 * cross_terms)
            bucket_risk_positions[bucket] = Kb
        
        # Step 4: Calculate delta risk charge
        # Map buckets to indices for the correlation matrix
        bucket_to_index = {
            '0-1y': 0,
            '1-3y': 1,
            '3-5y': 2,
            '5-10y': 3,
            '>10y': 4
        }
        
        # Sum of squared bucket risk positions
        sum_squared_Kb = sum(Kb**2 for Kb in bucket_risk_positions.values())
        
        # Calculate cross-bucket terms
        cross_bucket_terms = 0
        bucket_list = list(bucket_risk_positions.keys())
        
        for i, bucket1 in enumerate(bucket_list):
            for bucket2 in bucket_list[i+1:]:
                idx1 = bucket_to_index[bucket1]
                idx2 = bucket_to_index[bucket2]
                correlation = self.girr_correlations[idx1, idx2]
                
                Sb1 = weighted_bucket_sensitivities[bucket1]['sum']
                Sb2 = weighted_bucket_sensitivities[bucket2]['sum']
                
                cross_bucket_terms += correlation * Sb1 * Sb2
        
        # Delta = sqrt(sum of squared bucket risk positions + cross-bucket terms)
        delta_risk_charge = np.sqrt(sum_squared_Kb + cross_bucket_terms)
        
        return delta_risk_charge
    
    def calculate_fx_delta_risk_charge(self, fx_sensitivities):
        """
        Calculate delta risk charge for FX risk.
        
        Parameters:
        -----------
        fx_sensitivities : dict
            Dictionary with FX sensitivities.
            
        Returns:
        --------
        float
            Delta risk charge for FX risk
        """
        # For FX risk, each currency forms its own risk class
        # Step 1: Calculate weighted sensitivities
        weighted_sensitivities = {}
        
        for name, data in fx_sensitivities.items():
            sensitivity = data['sensitivity']
            weighted_sensitivities[name] = sensitivity * self.fx_risk_weight
        
        # Step 2: Calculate the risk position
        # Since correlation between different currencies is typically set to 0.5 in Basel III
        # when no specific correlation is provided
        correlation = 0.5
        
        # Sum of squared weighted sensitivities
        sum_squared = sum(ws**2 for ws in weighted_sensitivities.values())
        
        # Calculate the cross-terms
        cross_terms = 0
        sensitivity_list = list(weighted_sensitivities.items())
        
        for i, (name1, ws1) in enumerate(sensitivity_list):
            for name2, ws2 in sensitivity_list[i+1:]:
                cross_terms += correlation * ws1 * ws2
        
        # Delta = sqrt(sum of squared weighted sensitivities + cross terms)
        delta_risk_charge = np.sqrt(sum_squared + 2 * cross_terms)
        
        return delta_risk_charge
    
    def calculate_commodity_delta_risk_charge(self, commodity_sensitivities):
        """
        Calculate delta risk charge for commodity risk.
        
        Parameters:
        -----------
        commodity_sensitivities : dict
            Dictionary with commodity sensitivities.
            
        Returns:
        --------
        float
            Delta risk charge for commodity risk
        """
        # Step 1: Calculate weighted sensitivities
        weighted_sensitivities = {}
        for name, data in commodity_sensitivities.items():
            bucket = data['bucket']
            sensitivity = data['sensitivity']
            risk_weight = self.commodity_risk_weights[bucket]
            
            weighted_sensitivities[name] = {
                'bucket': bucket,
                'weighted_sensitivity': sensitivity * risk_weight
            }
        
        # Step 2: Calculate risk positions for each bucket
        bucket_risk_positions = {}
        bucket_sensitivities = {}
        
        # Group sensitivities by bucket
        for name, data in weighted_sensitivities.items():
            bucket = data['bucket']
            ws = data['weighted_sensitivity']
            
            if bucket not in bucket_sensitivities:
                bucket_sensitivities[bucket] = {}
            
            bucket_sensitivities[bucket][name] = ws
        
        # Calculate risk position for each bucket
        for bucket, sensitivities in bucket_sensitivities.items():
            # Sum of weighted sensitivities in the bucket
            Sb = sum(sensitivities.values())
            bucket_sensitivities[bucket]['sum'] = Sb
            
            # Calculate the risk position Kb
            sum_squared = sum(ws**2 for ws in sensitivities.values() if isinstance(ws, (int, float)))
            
            # Calculate the cross-terms
            cross_terms = 0
            sensitivity_list = [(name, ws) for name, ws in sensitivities.items() if name != 'sum']
            
            for i, (name1, ws1) in enumerate(sensitivity_list):
                for name2, ws2 in sensitivity_list[i+1:]:
                    # Apply correlation within the bucket (assumed to be 1.0 for simplicity)
                    cross_terms += 1.0 * ws1 * ws2
            
            # Kb = sqrt(sum of squared weighted sensitivities + cross terms)
            Kb = np.sqrt(sum_squared + 2 * cross_terms)
            bucket_risk_positions[bucket] = Kb
        
        # Step 3: Calculate delta risk charge
        # Sum of squared bucket risk positions
        sum_squared_Kb = sum(Kb**2 for Kb in bucket_risk_positions.values())
        
        # Calculate cross-bucket terms
        cross_bucket_terms = 0
        bucket_list = list(bucket_risk_positions.keys())
        
        for i, bucket1 in enumerate(bucket_list):
            for bucket2 in bucket_list[i+1:]:
                correlation = self.commodity_correlations[bucket1-1, bucket2-1]
                Sb1 = bucket_sensitivities[bucket1]['sum']
                Sb2 = bucket_sensitivities[bucket2]['sum']
                
                cross_bucket_terms += correlation * Sb1 * Sb2
        
        # Delta = sqrt(sum of squared bucket risk positions + cross-bucket terms)
        delta_risk_charge = np.sqrt(sum_squared_Kb + cross_bucket_terms)
        
        return delta_risk_charge
    
    def calculate_total_capital_requirement(self, equity_holdings, bond_holdings, fx_holdings, commodity_holdings):
        """
        Calculate the total capital requirement for market risk.
        
        Parameters:
        -----------
        equity_holdings : list of dict
            List of dictionaries containing equity position information.
        bond_holdings : list of dict
            List of dictionaries containing bond position information.
        fx_holdings : list of dict
            List of dictionaries containing FX position information.
        commodity_holdings : list of dict
            List of dictionaries containing commodity position information.
            
        Returns:
        --------
        dict
            Dictionary with capital requirements for different risk types and total
        """
        # Calculate sensitivities
        equity_sensitivities = self.calculate_equity_delta_sensitivities(equity_holdings)
        girr_sensitivities = self.calculate_girr_delta_sensitivities(bond_holdings)
        fx_sensitivities = self.calculate_fx_delta_sensitivities(fx_holdings)
        commodity_sensitivities = self.calculate_commodity_delta_sensitivities(commodity_holdings)
        
        # Calculate risk charges
        equity_risk_charge = self.calculate_equity_delta_risk_charge(equity_sensitivities)
        girr_risk_charge = self.calculate_girr_delta_risk_charge(girr_sensitivities)
        fx_risk_charge = self.calculate_fx_delta_risk_charge(fx_sensitivities)
        commodity_risk_charge = self.calculate_commodity_delta_risk_charge(commodity_sensitivities)
        
        # Total capital requirement is the sum of all risk charges
        total_capital_requirement = equity_risk_charge + girr_risk_charge + fx_risk_charge + commodity_risk_charge
        
        return {
            'equity_risk_charge': equity_risk_charge,
            'girr_risk_charge': girr_risk_charge,
            'fx_risk_charge': fx_risk_charge,
            'commodity_risk_charge': commodity_risk_charge,
            'total_capital_requirement': total_capital_requirement
        }


# Define a function to simulate a portfolio based on the paper's example
def simulate_portfolio():
    """
    Create a simulated portfolio similar to the one described in the paper.
    
    Returns:
    --------
    tuple
        Tuple containing equity_holdings, bond_holdings, fx_holdings, and commodity_holdings
    """
    # Define equity holdings
    equity_holdings = [
        {
            'name': 'Exxon Mobil',
            'bucket': 7,  # Energy, Oil and Gas
            'market_value': 1100000  # $1,100,000
        },
        {
            'name': 'AT&T',
            'bucket': 6,  # Telecommunications
            'market_value': 170000  # $170,000
        },
        {
            'name': 'Apple',
            'bucket': 5,  # Technology
            'market_value': 500000  # $500,000
        },
        {
            'name': 'JPMorgan Chase',
            'bucket': 4,  # Financials
            'market_value': 300000  # $300,000
        }
    ]
    
    # Define bond holdings
    bond_holdings = [
        {
            'name': 'US Treasury 5Y',
            'maturity': 5,
            'market_value': 10000,  # $10,000
            'duration': 4.5  # Modified duration
        },
        {
            'name': 'US Treasury 10Y',
            'maturity': 10,
            'market_value': 10000,  # $10,000
            'duration': 8.7  # Modified duration
        },
        {
            'name': 'Corporate Bond A',
            'maturity': 3,
            'market_value': 5000,  # $5,000
            'duration': 2.8  # Modified duration
        },
        {
            'name': 'Corporate Bond B',
            'maturity': 7,
            'market_value': 7500,  # $7,500
            'duration': 6.2  # Modified duration
        }
    ]
    
    # Define FX holdings
    fx_holdings = [
        {
            'name': 'EUR/USD',
            'base_currency': 'EUR',
            'quote_currency': 'USD',
            'position': 100000,  # 100,000 EUR
            'exchange_rate': 1.09  # 1 EUR = 1.09 USD
        },
        {
            'name': 'USD/JPY',
            'base_currency': 'USD',
            'quote_currency': 'JPY',
            'position': -10000,  # Short 10,000 USD
            'exchange_rate': 151.5  # 1 USD = 151.5 JPY
        },
        {
            'name': 'GBP/USD',
            'base_currency': 'GBP',
            'quote_currency': 'USD',
            'position': 50000,  # 50,000 GBP
            'exchange_rate': 1.27  # 1 GBP = 1.27 USD
        }
    ]
    
    # Define commodity holdings
    commodity_holdings = [
        {
            'name': 'Gold',
            'bucket': 2,  # Metals
            'quantity': 600,  # 600 ounces
            'price': 2380  # $2,380 per ounce
        },
        {
            'name': 'Crude Oil',
            'bucket': 1,  # Energy
            'quantity': 2000,  # 2,000 barrels
            'price': 74  # $74 per barrel
        },
        {
            'name': 'Wheat',
            'bucket': 3,  # Agricultural
            'quantity': 10000,  # 10,000 bushels
            'price': 6  # $6 per bushel
        }
    ]
    
    return equity_holdings, bond_holdings, fx_holdings, commodity_holdings

# Define a function to compare prompt engineering approaches
def simulate_prompt_engineering_impact():
    """
    Simulate the impact of prompt engineering on capital requirement calculation accuracy.
    
    Returns:
    --------
    dict
        Dictionary with results comparing naive prompt vs. detailed prompt
    """
    # Simulate different levels of accuracy based on the paper's findings
    naive_prompt_results = {
        'bucket_identification_accuracy': 0.655,
        'risk_weight_accuracy': 1.0,
        'correlation_accuracy': 0.30
    }
    
    detailed_prompt_results = {
        'bucket_identification_accuracy': 0.85,
        'risk_weight_accuracy': 1.0,
        'correlation_accuracy': 0.965
    }
    
    # Calculate potential error in capital requirements based on these accuracies
    # Generate 1000 simulated capital calculations with errors
    np.random.seed(42)
    n_simulations = 1000
    
    # Baseline capital requirement (assuming perfect accuracy)
    baseline_capital = 1000000  # $1,000,000
    
    # Simulate capital calculations with naive prompt errors
    naive_prompt_capitals = []
    for _ in range(n_simulations):
        # Errors in bucket identification affect risk weights
        bucket_error = np.random.binomial(1, 1 - naive_prompt_results['bucket_identification_accuracy'])
        # Errors in correlation affect aggregation
        correlation_error = np.random.binomial(1, 1 - naive_prompt_results['correlation_accuracy'])
        
        # Calculate capital with errors
        capital = baseline_capital
        if bucket_error:
            # Misidentified buckets can lead to wrong risk weights
            capital *= np.random.uniform(0.8, 1.2)
        if correlation_error:
            # Misidentified correlations can lead to wrong aggregation
            capital *= np.random.uniform(0.7, 1.3)
        
        naive_prompt_capitals.append(capital)
    
    # Simulate capital calculations with detailed prompt errors
    detailed_prompt_capitals = []
    for _ in range(n_simulations):
        # Errors in bucket identification affect risk weights
        bucket_error = np.random.binomial(1, 1 - detailed_prompt_results['bucket_identification_accuracy'])
        # Errors in correlation affect aggregation
        correlation_error = np.random.binomial(1, 1 - detailed_prompt_results['correlation_accuracy'])
        
        # Calculate capital with errors
        capital = baseline_capital
        if bucket_error:
            # Misidentified buckets can lead to wrong risk weights
            capital *= np.random.uniform(0.8, 1.2)
        if correlation_error:
            # Misidentified correlations can lead to wrong aggregation
            capital *= np.random.uniform(0.7, 1.3)
        
        detailed_prompt_capitals.append(capital)
    
    return {
        'naive_prompt': {
            'capitals': naive_prompt_capitals,
            'mean': np.mean(naive_prompt_capitals),
            'std': np.std(naive_prompt_capitals),
            'min': np.min(naive_prompt_capitals),
            'max': np.max(naive_prompt_capitals),
            'accuracies': naive_prompt_results
        },
        'detailed_prompt': {
            'capitals': detailed_prompt_capitals,
            'mean': np.mean(detailed_prompt_capitals),
            'std': np.std(detailed_prompt_capitals),
            'min': np.min(detailed_prompt_capitals),
            'max': np.max(detailed_prompt_capitals),
            'accuracies': detailed_prompt_results
        },
        'baseline_capital': baseline_capital
    }

# Define a function to compare different LLMs
def simulate_llm_comparison():
    """
    Simulate the performance of different LLMs based on the paper's findings.
    
    Returns:
    --------
    dict
        Dictionary with performance metrics for different LLMs
    """
    # Performance metrics based on the paper's Table 2 and Table 3
    llm_performance = {
        'GPT-4': {
            'bucket_identification': 0.85,
            'risk_weight_identification': 1.0,
            'correlation_identification': 0.965,
            'mcr_calculation_accuracy': 0.95
        },
        'GPT-3.5': {
            'bucket_identification': 0.10,
            'risk_weight_identification': 0.30,
            'correlation_identification': 0.0,
            'mcr_calculation_accuracy': 0.0
        },
        'Claude-3-Opus': {
            'bucket_identification': 0.825,
            'risk_weight_identification': 1.0,
            'correlation_identification': 0.975,
            'mcr_calculation_accuracy': 0.38
        },
        'Gemini-1.5-Pro': {
            'bucket_identification': 0.275,
            'risk_weight_identification': 0.75,
            'correlation_identification': 0.80,
            'mcr_calculation_accuracy': 0.58
        }
    }
    
    # Simulate capital calculations based on these accuracies
    baseline_capital = 1000000  # $1,000,000
    n_simulations = 1000
    
    llm_results = {}
    
    for llm_name, metrics in llm_performance.items():
        capitals = []
        for _ in range(n_simulations):
            # Errors in different aspects of capital calculation
            bucket_error = np.random.binomial(1, 1 - metrics['bucket_identification'])
            risk_weight_error = np.random.binomial(1, 1 - metrics['risk_weight_identification'])
            correlation_error = np.random.binomial(1, 1 - metrics['correlation_identification'])
            calculation_error = np.random.binomial(1, 1 - metrics['mcr_calculation_accuracy'])
            
            # Calculate capital with errors
            capital = baseline_capital
            if bucket_error:
                capital *= np.random.uniform(0.8, 1.2)
            if risk_weight_error:
                capital *= np.random.uniform(0.7, 1.3)
            if correlation_error:
                capital *= np.random.uniform(0.85, 1.15)
            if calculation_error:
                capital *= np.random.uniform(0.5, 1.5)
            
            capitals.append(capital)
        
        llm_results[llm_name] = {
            'capitals': capitals,
            'mean': np.mean(capitals),
            'std': np.std(capitals),
            'min': np.min(capitals),
            'max': np.max(capitals),
            'metrics': metrics
        }
    
    return {
        'llm_results': llm_results,
        'baseline_capital': baseline_capital
    }

# Define a function to compare document loading methods
def simulate_document_loading_comparison():
    """
    Simulate the impact of document loading methods on capital requirement calculations.
    
    Returns:
    --------
    dict
        Dictionary with results comparing PDF vs. image loading
    """
    # Accuracy in identifying correlations based on the paper's Table 4
    loading_accuracy = {
        'PDF': {
            'Claude-3-Opus': 0.765,
            'GPT-4': 0.68
        },
        'IMAGE': {
            'Claude-3-Opus': 0.975,
            'GPT-4': 0.965
        }
    }
    
    # Simulate capital calculations based on these accuracies
    baseline_capital = 1000000  # $1,000,000
    n_simulations = 1000
    
    loading_results = {}
    
    for loading_method, llm_accuracies in loading_accuracy.items():
        llm_results = {}
        
        for llm_name, accuracy in llm_accuracies.items():
            capitals = []
            for _ in range(n_simulations):
                # Errors in correlation identification affect capital calculation
                correlation_error = np.random.binomial(1, 1 - accuracy)
                
                # Calculate capital with errors
                capital = baseline_capital
                if correlation_error:
                    # Misidentified correlations can lead to wrong aggregation
                    capital *= np.random.uniform(0.7, 1.3)
                
                capitals.append(capital)
            
            llm_results[llm_name] = {
                'capitals': capitals,
                'mean': np.mean(capitals),
                'std': np.std(capitals),
                'min': np.min(capitals),
                'max': np.max(capitals),
                'accuracy': accuracy
            }
        
        loading_results[loading_method] = llm_results
    
    return {
        'loading_results': loading_results,
        'baseline_capital': baseline_capital
    }

# Run the simulations and present the results
def main():
    # Create an instance of the Basel III calculator
    calculator = BaselIIICalculator()
    
    # Simulate a portfolio
    equity_holdings, bond_holdings, fx_holdings, commodity_holdings = simulate_portfolio()
    
    # Calculate capital requirements
    capital_requirements = calculator.calculate_total_capital_requirement(
        equity_holdings, bond_holdings, fx_holdings, commodity_holdings
    )
    
    # Print portfolio details
    print("Portfolio Holdings:")
    print("-------------------")
    print("\nEquity Holdings:")
    equity_df = pd.DataFrame(equity_holdings)
    print(tabulate(equity_df, headers='keys', tablefmt='pretty', showindex=False))
    
    print("\nBond Holdings:")
    bond_df = pd.DataFrame(bond_holdings)
    print(tabulate(bond_df, headers='keys', tablefmt='pretty', showindex=False))
    
    print("\nFX Holdings:")
    fx_df = pd.DataFrame(fx_holdings)
    print(tabulate(fx_df, headers='keys', tablefmt='pretty', showindex=False))
    
    print("\nCommodity Holdings:")
    commodity_df = pd.DataFrame(commodity_holdings)
    print(tabulate(commodity_df, headers='keys', tablefmt='pretty', showindex=False))
    
    # Print capital requirements
    print("\nCapital Requirements:")
    print("--------------------")
    for risk_type, amount in capital_requirements.items():
        print(f"{risk_type.replace('_', ' ').title()}: ${amount:,.2f}")
    
    # Print sensitivities
    print("\nCalculated Sensitivities:")
    print("-----------------------")
    
    print("\nEquity Sensitivities:")
    equity_sensitivities = calculator.calculate_equity_delta_sensitivities(equity_holdings)
    equity_sens_df = pd.DataFrame([
        {'Name': name, 'Sensitivity': data['sensitivity'], 'Bucket': data['bucket']}
        for name, data in equity_sensitivities.items()
    ])
    print(tabulate(equity_sens_df, headers='keys', tablefmt='pretty', showindex=False))
    
    print("\nGIRR Sensitivities:")
    girr_sensitivities = calculator.calculate_girr_delta_sensitivities(bond_holdings)
    girr_sens_df = pd.DataFrame([
        {'Name': name, 'Sensitivity': data['sensitivity'], 'Bucket': data['bucket']}
        for name, data in girr_sensitivities.items()
    ])
    print(tabulate(girr_sens_df, headers='keys', tablefmt='pretty', showindex=False))
    
    print("\nFX Sensitivities:")
    fx_sensitivities = calculator.calculate_fx_delta_sensitivities(fx_holdings)
    fx_sens_df = pd.DataFrame([
        {'Name': name, 'Sensitivity': data['sensitivity'], 
         'Base Currency': data['base_currency'], 'Quote Currency': data['quote_currency']}
        for name, data in fx_sensitivities.items()
    ])
    print(tabulate(fx_sens_df, headers='keys', tablefmt='pretty', showindex=False))
    
    print("\nCommodity Sensitivities:")
    commodity_sensitivities = calculator.calculate_commodity_delta_sensitivities(commodity_holdings)
    commodity_sens_df = pd.DataFrame([
        {'Name': name, 'Sensitivity': data['sensitivity'], 'Bucket': data['bucket']}
        for name, data in commodity_sensitivities.items()
    ])
    print(tabulate(commodity_sens_df, headers='keys', tablefmt='pretty', showindex=False))
    
    # Simulate and visualize prompt engineering impact
    prompt_results = simulate_prompt_engineering_impact()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(prompt_results['naive_prompt']['capitals'], kde=True, color='red', alpha=0.5, label='Naive Prompt')
    sns.histplot(prompt_results['detailed_prompt']['capitals'], kde=True, color='blue', alpha=0.5, label='Detailed Prompt')
    plt.axvline(prompt_results['baseline_capital'], color='black', linestyle='--', label='Baseline Capital')
    plt.title('Impact of Prompt Engineering on Capital Requirement Calculation')
    plt.xlabel('Capital Requirement ($)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Print prompt engineering results
    print("\nPrompt Engineering Impact:")
    print("--------------------------")
    print(f"Baseline Capital: ${prompt_results['baseline_capital']:,.2f}")
    print("\nNaive Prompt:")
    print(f"  Bucket Identification Accuracy: {prompt_results['naive_prompt']['accuracies']['bucket_identification_accuracy']:.2%}")
    print(f"  Risk Weight Accuracy: {prompt_results['naive_prompt']['accuracies']['risk_weight_accuracy']:.2%}")
    print(f"  Correlation Accuracy: {prompt_results['naive_prompt']['accuracies']['correlation_accuracy']:.2%}")
    print(f"  Mean Capital: ${prompt_results['naive_prompt']['mean']:,.2f}")
    print(f"  Standard Deviation: ${prompt_results['naive_prompt']['std']:,.2f}")
    print(f"  Range: ${prompt_results['naive_prompt']['min']:,.2f} - ${prompt_results['naive_prompt']['max']:,.2f}")
    
    print("\nDetailed Prompt:")
    print(f"  Bucket Identification Accuracy: {prompt_results['detailed_prompt']['accuracies']['bucket_identification_accuracy']:.2%}")
    print(f"  Risk Weight Accuracy: {prompt_results['detailed_prompt']['accuracies']['risk_weight_accuracy']:.2%}")
    print(f"  Correlation Accuracy: {prompt_results['detailed_prompt']['accuracies']['correlation_accuracy']:.2%}")
    print(f"  Mean Capital: ${prompt_results['detailed_prompt']['mean']:,.2f}")
    print(f"  Standard Deviation: ${prompt_results['detailed_prompt']['std']:,.2f}")
    print(f"  Range: ${prompt_results['detailed_prompt']['min']:,.2f} - ${prompt_results['detailed_prompt']['max']:,.2f}")
    
    # Simulate and visualize LLM comparison
    llm_comparison = simulate_llm_comparison()
    
    plt.figure(figsize=(12, 8))
    
    llm_names = list(llm_comparison['llm_results'].keys())
    means = [llm_comparison['llm_results'][llm]['mean'] for llm in llm_names]
    stds = [llm_comparison['llm_results'][llm]['std'] for llm in llm_names]
    
    bars = plt.bar(llm_names, means, yerr=stds, capsize=10, alpha=0.7)
    
    # Add baseline
    plt.axhline(llm_comparison['baseline_capital'], color='black', linestyle='--', label='Baseline Capital')
    
    plt.title('Comparison of Different LLMs for Capital Requirement Calculation')
    plt.xlabel('LLM')
    plt.ylabel('Mean Capital Requirement ($)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Create a heatmap of LLM performance metrics
    metrics = ['bucket_identification', 'risk_weight_identification', 
               'correlation_identification', 'mcr_calculation_accuracy']
    
    performance_data = np.zeros((len(llm_names), len(metrics)))
    
    for i, llm in enumerate(llm_names):
        for j, metric in enumerate(metrics):
            performance_data[i, j] = llm_comparison['llm_results'][llm]['metrics'][metric]
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(performance_data, annot=True, fmt='.2%', cmap='viridis',
                xticklabels=[m.replace('_', ' ').title() for m in metrics],
                yticklabels=llm_names)
    plt.title('LLM Performance Metrics')
    plt.tight_layout()
    plt.show()
    
    # Simulate and visualize document loading comparison
    loading_comparison = simulate_document_loading_comparison()
    
    plt.figure(figsize=(10, 6))
    
    loading_methods = list(loading_comparison['loading_results'].keys())
    llm_names = list(loading_comparison['loading_results'][loading_methods[0]].keys())
    
    x = np.arange(len(llm_names))
    width = 0.35
    
    for i, method in enumerate(loading_methods):
        means = [loading_comparison['loading_results'][method][llm]['mean'] for llm in llm_names]
        stds = [loading_comparison['loading_results'][method][llm]['std'] for llm in llm_names]
        
        plt.bar(x + (i - 0.5) * width, means, width, label=method,
                yerr=stds, capsize=5, alpha=0.7)
    
    plt.axhline(loading_comparison['baseline_capital'], color='black', linestyle='--', label='Baseline Capital')
    
    plt.title('Impact of Document Loading Method on Capital Requirement Calculation')
    plt.xlabel('LLM')
    plt.ylabel('Mean Capital Requirement ($)')
    plt.xticks(x, llm_names)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Create a heatmap of correlation identification accuracy by document loading method
    accuracy_data = np.zeros((len(loading_methods), len(llm_names)))
    
    for i, method in enumerate(loading_methods):
        for j, llm in enumerate(llm_names):
            accuracy_data[i, j] = loading_comparison['loading_results'][method][llm]['accuracy']
    
    plt.figure(figsize=(8, 4))
    sns.heatmap(accuracy_data, annot=True, fmt='.2%', cmap='viridis',
                xticklabels=llm_names, yticklabels=loading_methods)
    plt.title('Correlation Identification Accuracy by Document Loading Method')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()