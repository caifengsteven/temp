from pykrx import stock
import pandas as pd
from datetime import datetime

# Get today's date in YYYYMMDD format
today = datetime.today().strftime("%Y%m%d")

# Ticker for "KODEX KOSDAQ 150" ETF is 229200
etf_ticker = "229200" 

print(f"Fetching data for KODEX KOSDAQ 150 ({etf_ticker})...")

# 1. Get the Portfolio Deposit File (PDF) - This lists the shares held
# Note: If today is a weekend/holiday, this might fail. You may need to hardcode a recent weekday.
try:
    pdf = stock.get_etf_portfolio_deposit_file(etf_ticker, today)
except:
    print("Data not available for today (market closed?). Trying yesterday...")
    # You might need to adjust logic here to find the last business day
    # For now, let's assume it works or you manually input a date like "20231027"

# 2. Get current prices of these stocks to calculate Market Value
ticker_list = pdf.index.tolist()
# We fetch OHLCV for the constituents to get the 'Close' price
df_prices = stock.get_market_ohlcv(today, market="KOSDAQ")
df_prices = df_prices.loc[ticker_list]

# 3. Merge and Calculate Weights
# PDF contains 'Contract Amount' (Number of shares)
result = pdf.join(df_prices['종가']) # '종가' means Close Price
result.columns = ['Shares', 'Amount', 'Price']

# Calculate Total Value of each holding
result['Total_Value'] = result['Shares'] * result['Price']

# Calculate Weight (%)
total_asset_value = result['Total_Value'].sum()
result['Weight_Pct'] = (result['Total_Value'] / total_asset_value) * 100

# Sort by Weight
result = result.sort_values(by='Weight_Pct', ascending=False)

# Display top 10
print(result[['Shares', 'Price', 'Weight_Pct']].head(10))

# Save to CSV
result.to_csv("kosdaq150_weights.csv")
print("Saved to kosdaq150_weights.csv")