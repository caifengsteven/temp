"""
Estimate market-making parameters from actual quote data:
1. Hedge ratio (beta) between AAPL and MSFT
2. Queue depletion rate (delta) for fill probability
3. Volatility for inventory penalty (gamma)
"""
import pandas as pd
import numpy as np
import struct
import os

def load_binary_quotes(filename, max_records=10000000):
    """Load quotes from binary file"""
    quotes = []
    with open(filename, 'rb') as f:
        num_records = struct.unpack('<Q', f.read(8))[0]
        num_to_read = min(num_records, max_records)
        print(f"Loading {num_to_read:,} of {num_records:,} quotes from {filename}...")
        
        for i in range(num_to_read):
            data = f.read(36)  # 8+8+4+8+4+4 = 36 bytes
            if len(data) < 36:
                break
            ts, bid, bid_sz, ask, ask_sz, exch = struct.unpack('<qdidii', data)
            quotes.append({
                'timestamp': ts,
                'bid': bid,
                'ask': ask,
                'mid': (bid + ask) / 2
            })
    return pd.DataFrame(quotes)

# Load sample of data (first 10M quotes each)
aapl = load_binary_quotes('aapl_quotes.bin', 10000000)
msft = load_binary_quotes('msft_quotes.bin', 10000000)

print(f"\nAAPL: {len(aapl):,} quotes, price range: ${aapl['mid'].min():.2f} - ${aapl['mid'].max():.2f}")
print(f"MSFT: {len(msft):,} quotes, price range: ${msft['mid'].min():.2f} - ${msft['mid'].max():.2f}")

# Resample to 1-second bars for cleaner analysis
aapl['ts_sec'] = aapl['timestamp'] // 1000000000
msft['ts_sec'] = msft['timestamp'] // 1000000000

aapl_bars = aapl.groupby('ts_sec').agg({'mid': 'last'}).reset_index()
msft_bars = msft.groupby('ts_sec').agg({'mid': 'last'}).reset_index()

# Merge on timestamp
merged = pd.merge(aapl_bars, msft_bars, on='ts_sec', suffixes=('_aapl', '_msft'))
print(f"\nMerged bars: {len(merged):,}")

# Calculate returns
merged['ret_aapl'] = merged['mid_aapl'].pct_change()
merged['ret_msft'] = merged['mid_msft'].pct_change()
merged = merged.dropna()

# ============ 1. HEDGE RATIO (BETA) ============
cov = merged['ret_aapl'].cov(merged['ret_msft'])
var_msft = merged['ret_msft'].var()
beta = cov / var_msft

# Also calculate correlation
corr = merged['ret_aapl'].corr(merged['ret_msft'])

print("\n" + "="*50)
print("1. HEDGE RATIO (BETA)")
print("="*50)
print(f"   Covariance: {cov:.2e}")
print(f"   MSFT Variance: {var_msft:.2e}")
print(f"   Beta (AAPL vs MSFT): {beta:.4f}")
print(f"   Correlation: {corr:.4f}")
print(f"   --> Recommended hedge_ratio = {beta:.2f}")

# ============ 2. VOLATILITY & GAMMA ============
# Annualized volatility (assuming 6.5 hours/day, 252 days/year)
seconds_per_year = 6.5 * 3600 * 252
aapl_vol = merged['ret_aapl'].std() * np.sqrt(seconds_per_year)
msft_vol = merged['ret_msft'].std() * np.sqrt(seconds_per_year)

# Gamma from Avellaneda-Stoikov: gamma = 1 / (sigma^2 * T)
# Where T is trading horizon. For market making, T ~ seconds to minutes
T = 60  # 1 minute horizon
aapl_price = merged['mid_aapl'].mean()
sigma_per_sec = merged['ret_aapl'].std()
gamma_theoretical = 1 / (sigma_per_sec**2 * T * aapl_price**2)

print("\n" + "="*50)
print("2. VOLATILITY & INVENTORY PENALTY (GAMMA)")
print("="*50)
print(f"   AAPL 1-sec return std: {merged['ret_aapl'].std():.2e}")
print(f"   AAPL annualized vol: {aapl_vol*100:.1f}%")
print(f"   MSFT annualized vol: {msft_vol*100:.1f}%")
print(f"   AAPL avg price: ${aapl_price:.2f}")
print(f"   Theoretical gamma (T=60s): {gamma_theoretical:.2e}")
print(f"   --> Recommended gamma = {gamma_theoretical:.1e}")

# ============ 3. QUEUE DEPLETION RATE ============
# Estimate how often the best bid/ask changes
aapl['bid_change'] = (aapl['bid'] != aapl['bid'].shift(1)).astype(int)
aapl['ask_change'] = (aapl['ask'] != aapl['ask'].shift(1)).astype(int)

# Time between quotes (in seconds)
aapl['dt'] = aapl['timestamp'].diff() / 1e9

# Events per second
total_time = (aapl['timestamp'].max() - aapl['timestamp'].min()) / 1e9
bid_changes = aapl['bid_change'].sum()
ask_changes = aapl['ask_change'].sum()

delta_bid = bid_changes / total_time
delta_ask = ask_changes / total_time

print("\n" + "="*50)
print("3. QUEUE DEPLETION RATE (DELTA)")
print("="*50)
print(f"   Total time: {total_time:.0f} seconds ({total_time/3600:.1f} hours)")
print(f"   Bid changes: {bid_changes:,}")
print(f"   Ask changes: {ask_changes:,}")
print(f"   Bid depletion rate: {delta_bid:.2f} events/sec")
print(f"   Ask depletion rate: {delta_ask:.2f} events/sec")

# Fill horizon tau: time for queue to deplete
avg_queue_size = 1000  # shares (rough estimate)
order_size = 100
queue_position = avg_queue_size / 2  # assume middle of queue
tau_estimate = queue_position / (delta_bid * order_size)

print(f"   Estimated tau for 50% fill prob: {tau_estimate:.1f} seconds")

print("\n" + "="*50)
print("SUMMARY - RECOMMENDED PARAMETERS")
print("="*50)
print(f"   hedge_ratio = {beta:.2f}")
print(f"   gamma = {gamma_theoretical:.1e}")
print(f"   tau = {min(tau_estimate, 10):.1f} seconds")

