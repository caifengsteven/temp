"""
Export AAPL and MSFT quote data to a binary format for C++ processing.
Process day by day to avoid memory issues.
"""
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import struct
import os
import glob

def process_day_file(filepath):
    """Process a single day's parquet file and return filtered AAPL/MSFT data"""
    filters = [('ticker', 'in', ['AAPL', 'MSFT'])]
    table = pq.read_table(filepath, filters=filters)
    df = table.to_pandas()

    # Filter to market hours and valid quotes
    df['ts_sec'] = df['sip_timestamp'] / 1e9
    df['hour'] = pd.to_datetime(df['ts_sec'], unit='s').dt.hour
    df = df[(df['bid_price'] > 0) & (df['ask_price'] > 0)]
    df = df[(df['hour'] >= 14) & (df['hour'] < 21)]

    # Split by ticker
    aapl = df[df['ticker'] == 'AAPL'].copy()
    msft = df[df['ticker'] == 'MSFT'].copy()

    # Sort by timestamp within each ticker
    aapl = aapl.sort_values('sip_timestamp')
    msft = msft.sort_values('sip_timestamp')

    return aapl, msft

def append_binary(data, f):
    """Append quote data to binary file - using numpy for speed"""
    if len(data) == 0:
        return

    timestamps = data['sip_timestamp'].values.astype(np.int64)
    bid_prices = data['bid_price'].values.astype(np.float64)
    bid_sizes = data['bid_size'].values.astype(np.int32)
    ask_prices = data['ask_price'].values.astype(np.float64)
    ask_sizes = data['ask_size'].values.astype(np.int32)
    exchanges = data['bid_exchange'].values.astype(np.int32)

    # Write all records using structured array
    chunk_data = np.zeros(len(data), dtype=[
        ('timestamp', '<i8'),
        ('bid_price', '<f8'),
        ('bid_size', '<i4'),
        ('ask_price', '<f8'),
        ('ask_size', '<i4'),
        ('exchange', '<i4')
    ])
    chunk_data['timestamp'] = timestamps
    chunk_data['bid_price'] = bid_prices
    chunk_data['bid_size'] = bid_sizes
    chunk_data['ask_price'] = ask_prices
    chunk_data['ask_size'] = ask_sizes
    chunk_data['exchange'] = exchanges

    f.write(chunk_data.tobytes())

# Main processing
base_dir = 'W:/us_stock_quotes_parquet/2024'
months = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])

# First pass: count total records and collect all file paths
print("=== Pass 1: Counting records ===")
all_files = []
aapl_total = 0
msft_total = 0

for month in months:
    month_dir = os.path.join(base_dir, month)
    files = sorted([f for f in os.listdir(month_dir) if f.endswith('.parquet')])
    for fname in files:
        all_files.append(os.path.join(month_dir, fname))

print(f"Total files to process: {len(all_files)}")

# Second pass: process and write
print("\n=== Pass 2: Processing and exporting ===")

# Open output files
aapl_file = open('aapl_quotes.bin', 'wb')
msft_file = open('msft_quotes.bin', 'wb')

# Write placeholder headers (will update at end)
aapl_file.write(struct.pack('<Q', 0))
msft_file.write(struct.pack('<Q', 0))

for i, filepath in enumerate(all_files):
    fname = os.path.basename(filepath)
    print(f"[{i+1}/{len(all_files)}] {fname}...", end=" ", flush=True)

    try:
        aapl_day, msft_day = process_day_file(filepath)

        append_binary(aapl_day, aapl_file)
        append_binary(msft_day, msft_file)

        aapl_total += len(aapl_day)
        msft_total += len(msft_day)

        print(f"AAPL:{len(aapl_day):,} MSFT:{len(msft_day):,}")
    except Exception as e:
        print(f"ERROR: {e}")

# Update headers with actual counts
aapl_file.seek(0)
aapl_file.write(struct.pack('<Q', aapl_total))
aapl_file.close()

msft_file.seek(0)
msft_file.write(struct.pack('<Q', msft_total))
msft_file.close()

print(f"\n=== DONE ===")
print(f"Total AAPL quotes: {aapl_total:,}")
print(f"Total MSFT quotes: {msft_total:,}")

