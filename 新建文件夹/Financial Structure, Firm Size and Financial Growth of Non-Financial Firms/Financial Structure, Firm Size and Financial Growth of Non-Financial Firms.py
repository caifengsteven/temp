import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set seed for reproducibility
np.random.seed(42)

# Set up plotting
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def generate_simulated_data(n_firms=45, n_years=10):
    """
    Generate simulated data for non-financial firms based on the parameters 
    described in the study.
    
    Parameters:
    n_firms (int): Number of firms
    n_years (int): Number of years
    
    Returns:
    DataFrame: Simulated financial data
    """
    # Generate firm IDs and years
    firms = [f'Firm_{i+1}' for i in range(n_firms)]
    years = list(range(2008, 2008 + n_years))
    
    # Create empty DataFrame
    data = []
    
    for firm in firms:
        # Base values for financial structure components
        # Using the means and standard deviations from the paper
        base_std = np.random.normal(0.29, 0.26, 1)[0]  # Short-term debt
        base_ltd = np.random.normal(0.20, 0.19, 1)[0]  # Long-term debt
        base_re = np.random.normal(0.28, 0.33, 1)[0]   # Retained earnings
        base_sc = np.random.normal(0.10, 0.16, 1)[0]   # Share capital
        
        # Base firm size (in million KES)
        base_size = np.random.lognormal(mean=10, sigma=1, size=1)[0]
        
        for year in years:
            # Add random variation over time
            std = max(0.008, min(2.5, base_std + np.random.normal(0, 0.05)))
            ltd = max(0, min(1.1, base_ltd + np.random.normal(0, 0.04)))
            re = max(-1.6, min(1.0, base_re + np.random.normal(0, 0.06)))
            sc = max(0.001, min(1.1, base_sc + np.random.normal(0, 0.03)))
            
            # Firm size grows over time with some randomness
            firm_size = base_size * (1 + (year - 2008) * 0.08 + np.random.normal(0, 0.1))
            
            # Calculate financial growth measures based on financial structure
            # Parameters are derived from regression coefficients in the paper
            # Growth in EPS
            eps_growth = (0.024 * std + 0.864 * ltd + 0.952 * re + 0.007 * sc + 
                         0.00001 * firm_size + np.random.normal(0, 3.5))
            
            # Growth in market capitalization
            market_cap = firm_size * (1 + (0.029 * std + 0.959 * ltd + 0.044 * re + 0.096 * sc + 
                        np.random.normal(0, 0.2)))
            
            # Actual EPS value (not just growth)
            eps = max(-46, min(100, 6.5 + eps_growth + np.random.normal(0, 2)))
            
            # Add row to data
            data.append({
                'Firm': firm,
                'Year': year,
                'Short_Term_Debt': std,
                'Long_Term_Debt': ltd,
                'Retained_Earnings': re,
                'Share_Capital': sc,
                'Firm_Size': firm_size,
                'EPS': eps,
                'Market_Cap': market_cap,
                'EPS_Growth': eps_growth,
                'Market_Cap_Growth': (market_cap / firm_size) - 1
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Calculate lagged values for growth calculations
    df = df.sort_values(['Firm', 'Year'])
    
    return df

# Generate the simulated data
df = generate_simulated_data()

# Display basic statistics
print("Descriptive Statistics of Simulated Data:")
print(df[['Short_Term_Debt', 'Long_Term_Debt', 'Retained_Earnings', 
          'Share_Capital', 'Firm_Size', 'EPS', 'Market_Cap']].describe())

# Correlation analysis
print("\nCorrelation Matrix:")
correlation_matrix = df[['Short_Term_Debt', 'Long_Term_Debt', 'Retained_Earnings', 
                         'Share_Capital', 'EPS_Growth', 'Market_Cap_Growth']].corr()
print(correlation_matrix.round(4))

# Visualize the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.4f')
plt.title('Correlation Matrix of Financial Structure and Growth Measures')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Function to run panel regression
def run_panel_regression(df, dependent_var, independent_vars, entity_effects=True):
    """
    Run a panel regression on the simulated data.
    
    Parameters:
    df (DataFrame): The dataset
    dependent_var (str): The dependent variable name
    independent_vars (list): List of independent variable names
    entity_effects (bool): Whether to include entity (firm) fixed effects
    
    Returns:
    Results: Regression results
    """
    # Create entity and time dummies
    entities = pd.Categorical(df['Firm']).codes
    times = pd.Categorical(df['Year']).codes
    
    # Prepare the data
    Y = df[dependent_var]
    X = df[independent_vars]
    
    # Add constant
    X = add_constant(X)
    
    # Run regression with entity effects
    if entity_effects:
        # Use entity (firm) fixed effects
        model = sm.OLS(Y, pd.concat([X, pd.get_dummies(entities, drop_first=True)], axis=1))
    else:
        model = sm.OLS(Y, X)
    
    results = model.fit()
    return results

# Test models as described in the paper
# Model 1: Direct effect of financial structure on EPS growth
model1_eps = run_panel_regression(
    df, 
    'EPS_Growth', 
    ['Short_Term_Debt', 'Long_Term_Debt', 'Retained_Earnings', 'Share_Capital']
)

# Model 1: Direct effect of financial structure on market cap growth
model1_mcap = run_panel_regression(
    df, 
    'Market_Cap_Growth', 
    ['Short_Term_Debt', 'Long_Term_Debt', 'Retained_Earnings', 'Share_Capital']
)

# Create a composite measure of financial structure (total finance)
df['Total_Finance'] = df['Short_Term_Debt'] + df['Long_Term_Debt'] + df['Retained_Earnings'] + df['Share_Capital']

# Testing the intervening effect of firm size using Baron & Kenny approach
# Step 1: Independent variable (Total Finance) -> Dependent variable (EPS Growth)
step1_eps = run_panel_regression(df, 'EPS_Growth', ['Total_Finance'])

# Step 2: Independent variable (Total Finance) -> Intervening variable (Firm Size)
step2 = run_panel_regression(df, 'Firm_Size', ['Total_Finance'])

# Step 3: Intervening variable (Firm Size) -> Dependent variable (EPS Growth)
step3_eps = run_panel_regression(df, 'EPS_Growth', ['Firm_Size'])

# Step 4: Independent variable (Total Finance) and Intervening variable (Firm Size) -> Dependent variable (EPS Growth)
step4_eps = run_panel_regression(df, 'EPS_Growth', ['Total_Finance', 'Firm_Size'])

# Same steps for Market Cap Growth
step1_mcap = run_panel_regression(df, 'Market_Cap_Growth', ['Total_Finance'])
step3_mcap = run_panel_regression(df, 'Market_Cap_Growth', ['Firm_Size'])
step4_mcap = run_panel_regression(df, 'Market_Cap_Growth', ['Total_Finance', 'Firm_Size'])

# Print results
print("\nModel 1: Effect of Financial Structure on EPS Growth")
print(model1_eps.summary().tables[1])

print("\nModel 1: Effect of Financial Structure on Market Cap Growth")
print(model1_mcap.summary().tables[1])

print("\nBaron & Kenny Approach for Intervening Effect - EPS Growth")
print("Step 1: Total Finance -> EPS Growth")
print(f"Coefficient: {step1_eps.params['Total_Finance']:.6f}, p-value: {step1_eps.pvalues['Total_Finance']:.6f}")
print("Step 2: Total Finance -> Firm Size")
print(f"Coefficient: {step2.params['Total_Finance']:.6f}, p-value: {step2.pvalues['Total_Finance']:.6f}")
print("Step 3: Firm Size -> EPS Growth")
print(f"Coefficient: {step3_eps.params['Firm_Size']:.6f}, p-value: {step3_eps.pvalues['Firm_Size']:.6f}")
print("Step 4: Total Finance + Firm Size -> EPS Growth")
print(f"Total Finance coefficient: {step4_eps.params['Total_Finance']:.6f}, p-value: {step4_eps.pvalues['Total_Finance']:.6f}")
print(f"Firm Size coefficient: {step4_eps.params['Firm_Size']:.6f}, p-value: {step4_eps.pvalues['Firm_Size']:.6f}")

print("\nBaron & Kenny Approach for Intervening Effect - Market Cap Growth")
print("Step 1: Total Finance -> Market Cap Growth")
print(f"Coefficient: {step1_mcap.params['Total_Finance']:.6f}, p-value: {step1_mcap.pvalues['Total_Finance']:.6f}")
print("Step 3: Firm Size -> Market Cap Growth")
print(f"Coefficient: {step3_mcap.params['Firm_Size']:.6f}, p-value: {step3_mcap.pvalues['Firm_Size']:.6f}")
print("Step 4: Total Finance + Firm Size -> Market Cap Growth")
print(f"Total Finance coefficient: {step4_mcap.params['Total_Finance']:.6f}, p-value: {step4_mcap.pvalues['Total_Finance']:.6f}")
print(f"Firm Size coefficient: {step4_mcap.params['Firm_Size']:.6f}, p-value: {step4_mcap.pvalues['Firm_Size']:.6f}")

# Visualize the relationships
# 1. Financial structure components vs EPS Growth
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

components = ['Short_Term_Debt', 'Long_Term_Debt', 'Retained_Earnings', 'Share_Capital']
for i, component in enumerate(components):
    sns.regplot(x=component, y='EPS_Growth', data=df, scatter_kws={'alpha':0.5}, ax=axes[i])
    axes[i].set_title(f'{component} vs EPS Growth')
    axes[i].set_xlabel(component)
    axes[i].set_ylabel('EPS Growth')

plt.tight_layout()
plt.savefig('financial_structure_vs_eps_growth.png')
plt.close()

# 2. Financial structure components vs Market Cap Growth
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for i, component in enumerate(components):
    sns.regplot(x=component, y='Market_Cap_Growth', data=df, scatter_kws={'alpha':0.5}, ax=axes[i])
    axes[i].set_title(f'{component} vs Market Cap Growth')
    axes[i].set_xlabel(component)
    axes[i].set_ylabel('Market Cap Growth')

plt.tight_layout()
plt.savefig('financial_structure_vs_mcap_growth.png')
plt.close()

# 3. Firm Size as intervening variable
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.regplot(x='Total_Finance', y='Firm_Size', data=df, scatter_kws={'alpha':0.5})
plt.title('Total Finance vs Firm Size')
plt.xlabel('Total Finance')
plt.ylabel('Firm Size')

plt.subplot(1, 2, 2)
sns.regplot(x='Firm_Size', y='EPS_Growth', data=df, scatter_kws={'alpha':0.5})
plt.title('Firm Size vs EPS Growth')
plt.xlabel('Firm Size')
plt.ylabel('EPS Growth')

plt.tight_layout()
plt.savefig('intervening_effect.png')
plt.close()

# Sector comparison (simulate different sectors)
df['Sector'] = np.random.choice(['Manufacturing', 'Energy', 'Consumer Goods', 'Technology', 'Services'], 
                               size=len(df))

# Compare financial structure across sectors
sector_comparison = df.groupby('Sector')[['Short_Term_Debt', 'Long_Term_Debt', 
                                         'Retained_Earnings', 'Share_Capital']].mean()
print("\nFinancial Structure by Sector:")
print(sector_comparison)

# Visualize sector comparison
plt.figure(figsize=(12, 8))
sector_comparison.plot(kind='bar', ax=plt.gca())
plt.title('Average Financial Structure Components by Sector')
plt.xlabel('Sector')
plt.ylabel('Proportion')
plt.legend(title='Financial Structure Component')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('sector_comparison.png')
plt.close()

# Compare growth metrics across sectors
growth_by_sector = df.groupby('Sector')[['EPS_Growth', 'Market_Cap_Growth']].mean()
print("\nGrowth Metrics by Sector:")
print(growth_by_sector)

plt.figure(figsize=(12, 6))
growth_by_sector.plot(kind='bar', ax=plt.gca())
plt.title('Average Growth Metrics by Sector')
plt.xlabel('Sector')
plt.ylabel('Growth Rate')
plt.legend(title='Growth Metric')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('growth_by_sector.png')
plt.close()

# Recommendations based on simulation results
print("\nRECOMMENDATIONS BASED ON SIMULATION RESULTS:")
print("1. Financial Structure Composition:")
for component in components:
    coef_eps = model1_eps.params[component]
    coef_mcap = model1_mcap.params[component]
    print(f"   - {component}: EPS Growth Impact = {coef_eps:.4f}, Market Cap Growth Impact = {coef_mcap:.4f}")

# Determine optimal financial structure based on simulation
# Create a grid of possible financial structure combinations
def find_optimal_structure():
    """
    Find the optimal financial structure combination that maximizes growth metrics
    """
    # Create grid of possible combinations
    grid_size = 10
    std_range = np.linspace(0.01, 0.5, grid_size)
    ltd_range = np.linspace(0.01, 0.4, grid_size)
    re_range = np.linspace(0.01, 0.5, grid_size)
    sc_range = np.linspace(0.01, 0.3, grid_size)
    
    # Assuming average firm size
    avg_firm_size = df['Firm_Size'].mean()
    
    best_eps_growth = -np.inf
    best_mcap_growth = -np.inf
    best_structure_eps = None
    best_structure_mcap = None
    
    # Get model coefficients
    coef_std_eps = model1_eps.params['Short_Term_Debt']
    coef_ltd_eps = model1_eps.params['Long_Term_Debt']
    coef_re_eps = model1_eps.params['Retained_Earnings']
    coef_sc_eps = model1_eps.params['Share_Capital']
    
    coef_std_mcap = model1_mcap.params['Short_Term_Debt']
    coef_ltd_mcap = model1_mcap.params['Long_Term_Debt']
    coef_re_mcap = model1_mcap.params['Retained_Earnings']
    coef_sc_mcap = model1_mcap.params['Share_Capital']
    
    # Constraint: components must sum to approximately 1
    for std in std_range:
        for ltd in ltd_range:
            for re in re_range:
                for sc in sc_range:
                    total = std + ltd + re + sc
                    if 0.8 <= total <= 1.2:  # Allow some flexibility
                        # Predict growth
                        eps_growth = coef_std_eps * std + coef_ltd_eps * ltd + coef_re_eps * re + coef_sc_eps * sc
                        mcap_growth = coef_std_mcap * std + coef_ltd_mcap * ltd + coef_re_mcap * re + coef_sc_mcap * sc
                        
                        if eps_growth > best_eps_growth:
                            best_eps_growth = eps_growth
                            best_structure_eps = (std, ltd, re, sc)
                        
                        if mcap_growth > best_mcap_growth:
                            best_mcap_growth = mcap_growth
                            best_structure_mcap = (std, ltd, re, sc)
    
    return best_structure_eps, best_eps_growth, best_structure_mcap, best_mcap_growth

best_structure_eps, best_eps_growth, best_structure_mcap, best_mcap_growth = find_optimal_structure()

print("\n2. Optimal Financial Structure:")
print(f"   For maximizing EPS Growth:")
print(f"   - Short-Term Debt: {best_structure_eps[0]:.4f}")
print(f"   - Long-Term Debt: {best_structure_eps[1]:.4f}")
print(f"   - Retained Earnings: {best_structure_eps[2]:.4f}")
print(f"   - Share Capital: {best_structure_eps[3]:.4f}")
print(f"   - Predicted EPS Growth: {best_eps_growth:.4f}")

print(f"\n   For maximizing Market Cap Growth:")
print(f"   - Short-Term Debt: {best_structure_mcap[0]:.4f}")
print(f"   - Long-Term Debt: {best_structure_mcap[1]:.4f}")
print(f"   - Retained Earnings: {best_structure_mcap[2]:.4f}")
print(f"   - Share Capital: {best_structure_mcap[3]:.4f}")
print(f"   - Predicted Market Cap Growth: {best_mcap_growth:.4f}")

print("\n3. Firm Size Considerations:")
if step4_eps.pvalues['Total_Finance'] > 0.05 and step4_eps.pvalues['Firm_Size'] < 0.05:
    print("   - Firm size significantly mediates the relationship between financial structure and EPS growth")
    print("   - Larger firms should focus more on optimizing their financial structure for EPS growth")
else:
    print("   - Firm size does not strongly mediate the relationship between financial structure and EPS growth")
    
if step4_mcap.pvalues['Total_Finance'] < 0.05:
    print("   - Firm size does not mediate the relationship between financial structure and market cap growth")
    print("   - Financial structure directly impacts market cap growth regardless of firm size")

print("\n4. Sector-Specific Recommendations:")
# Find best performing sector
best_eps_sector = growth_by_sector['EPS_Growth'].idxmax()
best_mcap_sector = growth_by_sector['Market_Cap_Growth'].idxmax()

print(f"   - {best_eps_sector} sector shows highest EPS growth, with average financial structure:")
print(f"     {sector_comparison.loc[best_eps_sector].to_dict()}")
print(f"   - {best_mcap_sector} sector shows highest market cap growth, with average financial structure:")
print(f"     {sector_comparison.loc[best_mcap_sector].to_dict()}")
print("   - Firms should consider sector-specific financial structures as optimal ratios vary by industry")

# Final summary visualization
# Create a composite chart showing optimal financial structures
plt.figure(figsize=(12, 8))

# Current average structure
current_avg = df[['Short_Term_Debt', 'Long_Term_Debt', 'Retained_Earnings', 'Share_Capital']].mean()

# Data for bar chart
structures = pd.DataFrame({
    'Current Average': current_avg,
    'Optimal for EPS Growth': pd.Series({
        'Short_Term_Debt': best_structure_eps[0],
        'Long_Term_Debt': best_structure_eps[1],
        'Retained_Earnings': best_structure_eps[2],
        'Share_Capital': best_structure_eps[3]
    }),
    'Optimal for Market Cap Growth': pd.Series({
        'Short_Term_Debt': best_structure_mcap[0],
        'Long_Term_Debt': best_structure_mcap[1],
        'Retained_Earnings': best_structure_mcap[2],
        'Share_Capital': best_structure_mcap[3]
    }),
    f'Best Sector ({best_eps_sector})': sector_comparison.loc[best_eps_sector]
})

structures.plot(kind='bar', ax=plt.gca())
plt.title('Financial Structure Comparison: Current vs Optimal')
plt.xlabel('Financial Structure Component')
plt.ylabel('Proportion')
plt.legend(title='Structure Type')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('optimal_structure_comparison.png')
plt.close()

print("\nAnalysis complete. Results and visualizations have been saved.")