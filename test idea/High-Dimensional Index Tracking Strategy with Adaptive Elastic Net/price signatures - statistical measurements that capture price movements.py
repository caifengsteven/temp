import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pdblp
from sklearn.cluster import KMeans

# Connect to Bloomberg
con = pdblp.BCon(debug=False, port=8194)
con.start()

# Define parameters
symbols = ['EURUSD Curncy']
start_date = '20240101'
end_date = '20240501'

try:
    # Get daily data just to verify Bloomberg connection
    df_daily = con.bdh(symbols, 
                ['PX_LAST', 'PX_VOLUME'], 
                start_date, 
                end_date)

    print("Successfully retrieved daily data")
    print(df_daily.head())
    
    # Access the latest price (handling multi-index structure)
    # Print the structure to understand the data
    print("\nDataFrame Structure:")
    print(f"Columns: {df_daily.columns}")
    print(f"Index: {df_daily.index}")
    
    # Use a default price if we can't extract it from Bloomberg
    try:
        if isinstance(df_daily.columns, pd.MultiIndex):
            # For multi-index case
            latest_price = float(df_daily.xs('PX_LAST', level=1, axis=1).iloc[-1, 0])
        else:
            # Try direct access
            latest_price = float(df_daily['PX_LAST'].iloc[-1])
    except:
        print("Could not extract latest price from Bloomberg data, using default")
        latest_price = 1.0950  # Default EUR/USD price
    
    print(f"Using base price: {latest_price}")
    
    # Function to calculate price signature
    def calculate_signature(trades_df, horizon=5, direction_col='direction', 
                            price_col='price', size_col='size', time_col='timestamp'):
        """
        Calculate price signature as defined in Oomen (2019)
        
        Parameters:
        -----------
        trades_df : DataFrame with trade data
        horizon : Signature horizon in time units
        direction_col : Column with trade direction (1 for buy, -1 for sell)
        price_col : Column with price data
        size_col : Column with trade size
        time_col : Column with timestamps
        
        Returns:
        --------
        Signature values as a function of time relative to execution
        """
        # If dataframe is empty, return empty signature
        if len(trades_df) == 0:
            return pd.Series(dtype=float)
            
        # Convert prices to logarithmic scale
        trades_df['log_price'] = np.log(trades_df[price_col])
        
        # Create time points for signature
        times = np.arange(-horizon, horizon+1)
        
        signature = pd.Series(index=times, dtype=float)
        signature.loc[0] = 0  # S(0) = 0 by construction
        
        # Total volume for normalization
        total_volume = trades_df[size_col].sum()
        if total_volume == 0:
            print("Warning: Total volume is zero. Using count instead.")
            total_volume = len(trades_df)
        
        # For each time offset
        for t in times:
            if t == 0:
                continue
                
            weighted_price_diffs = []
            
            # For each trade, find price t time units away
            for idx, trade in trades_df.iterrows():
                trade_time = trade[time_col]
                
                # Calculate target time
                target_time = trade_time + pd.Timedelta(minutes=t)
                
                # Find closest price observation to target time
                time_diffs = (trades_df[time_col] - target_time).abs()
                if not time_diffs.empty and time_diffs.min().total_seconds() <= 300:  # Within 5 minutes
                    closest_idx = time_diffs.idxmin()
                    target_price = trades_df.loc[closest_idx, 'log_price']
                    
                    # Calculate price difference and weight by volume and direction
                    log_price_diff = target_price - trade['log_price']
                    weighted_diff = trade[direction_col] * trade[size_col] * log_price_diff
                    weighted_price_diffs.append(weighted_diff)
            
            if weighted_price_diffs:
                signature.loc[t] = sum(weighted_price_diffs) / total_volume
            else:
                signature.loc[t] = np.nan
        
        return signature.dropna()

    # Generate simulated intraday data
    def generate_simulated_intraday_data(base_price=1.0950, n_days=5, trades_per_day=100):
        """Generate simulated intraday data for testing the signature calculation"""
        np.random.seed(42)
        
        # Start from 5 days ago
        start_date = datetime.now() - timedelta(days=n_days)
        
        # Generate timestamps
        timestamps = []
        for day in range(n_days):
            day_start = start_date + timedelta(days=day)
            # Trading hours 8am to 5pm
            day_start = day_start.replace(hour=8, minute=0, second=0, microsecond=0)
            
            # Generate random times within trading hours
            for _ in range(trades_per_day):
                minute_offset = np.random.randint(0, 9 * 60)  # 9 hours of trading
                timestamps.append(day_start + timedelta(minutes=minute_offset))
        
        # Sort timestamps
        timestamps.sort()
        
        # Generate price data
        # Simple Brownian motion with drift
        drift = 0.0001
        volatility = 0.001
        
        prices = [base_price]
        for i in range(1, len(timestamps)):
            time_diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600  # in hours
            price_change = np.random.normal(drift * time_diff, volatility * np.sqrt(time_diff))
            prices.append(prices[-1] * np.exp(price_change))
        
        # Generate trade directions (with some autocorrelation)
        rho = 0.7  # Autocorrelation parameter
        directions = np.zeros(len(timestamps), dtype=int)
        directions[0] = np.random.choice([1, -1])
        
        for i in range(1, len(timestamps)):
            if np.random.random() < rho:
                directions[i] = directions[i-1]
            else:
                directions[i] = -directions[i-1]
        
        # Generate trade sizes (log-normal distribution)
        sizes = np.exp(np.random.normal(14, 0.5, len(timestamps)))
        
        # Create DataFrame
        df = pd.DataFrame({
            'timestamp': timestamps,
            'price': prices,
            'direction': directions,
            'size': sizes,
            'lp': np.random.choice(['LP1', 'LP2', 'LP3'], len(timestamps))
        })
        
        return df

    # Simulate intraday data
    print("\nGenerating simulated intraday data...")
    intraday_data = generate_simulated_intraday_data(
        base_price=latest_price,
        n_days=5,
        trades_per_day=200
    )
    
    print(f"Generated {len(intraday_data)} simulated trades")
    print(intraday_data.head())
    
    # Calculate signature
    print("\nCalculating price signature...")
    signature = calculate_signature(
        intraday_data, 
        horizon=30,  # 30-minute horizon
        direction_col='direction',
        price_col='price',
        size_col='size',
        time_col='timestamp'
    )
    
    # Plot the signature
    plt.figure(figsize=(12, 6))
    plt.plot(signature.index, signature.values, marker='o')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Price Signature (Simulated Intraday Data)')
    plt.xlabel('Time relative to execution (minutes)')
    plt.ylabel('Signature Value (log price change)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('price_signature_simulated.png')
    print("\nSignature calculated and saved to price_signature_simulated.png")
    
    # Calculate and compare signatures by LP
    print("\nCalculating signatures by liquidity provider...")
    lp_groups = intraday_data.groupby('lp')
    signatures_by_lp = {}
    
    for lp, group_data in lp_groups:
        signatures_by_lp[lp] = calculate_signature(
            group_data,
            horizon=30,
            direction_col='direction',
            price_col='price',
            size_col='size',
            time_col='timestamp'
        )
    
    # Plot LP signatures
    plt.figure(figsize=(12, 6))
    for lp, sig in signatures_by_lp.items():
        plt.plot(sig.index, sig.values, marker='o', label=f'LP: {lp}')
    
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Price Signatures by Liquidity Provider (Simulated)')
    plt.xlabel('Time relative to execution (minutes)')
    plt.ylabel('Signature Value (log price change)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('price_signatures_by_lp.png')
    print("\nLP signatures calculated and saved to price_signatures_by_lp.png")
    
    # Implement statistical tests from the paper
    print("\nPerforming statistical tests...")
    
    # L2-norm test for differences between LPs
    def l2_norm_test(signatures_dict):
        """
        Simple implementation of the L2-norm test from equation (13) in the paper
        """
        # Calculate overall mean signature
        all_sigs = pd.concat([s for s in signatures_dict.values()])
        overall_mean = all_sigs.groupby(level=0).mean()
        
        # Calculate SSH (between group sum of squares)
        SSH = pd.Series(0.0, index=overall_mean.index)
        
        for lp, sig in signatures_dict.items():
            # Weight by number of observations
            weight = len(sig)
            diff = sig - overall_mean
            SSH += weight * diff * diff
        
        # Integrate across signature horizon
        l2_stat = SSH.sum()
        
        return l2_stat
    
    l2_stat = l2_norm_test(signatures_by_lp)
    print(f"L2-norm test statistic: {l2_stat}")
    
    # Basic bootstrap implementation
    def simple_bootstrap(signatures_dict, n_bootstrap=1000):
        """
        Simple bootstrap to estimate p-value for L2-norm test
        """
        # Collect all data
        all_data = []
        lp_sizes = {}
        
        for lp, sig in signatures_dict.items():
            lp_sizes[lp] = len(sig)
            all_data.extend([(t, v, lp) for t, v in sig.items()])
        
        # Original test statistic
        original_stat = l2_norm_test(signatures_dict)
        
        # Bootstrap samples
        exceed_count = 0
        
        for _ in range(n_bootstrap):
            # Sample with replacement
            bootstrap_sample = np.random.choice(len(all_data), size=len(all_data), replace=True)
            
            # Create new signature groups with same LP sizes
            bootstrap_sigs = {}
            current_idx = 0
            
            for lp, size in lp_sizes.items():
                lp_data = [all_data[i] for i in bootstrap_sample[current_idx:current_idx+size]]
                current_idx += size
                
                # Convert to Series
                bootstrap_sigs[lp] = pd.Series({t: v for t, v, _ in lp_data})
            
            # Calculate test statistic
            bootstrap_stat = l2_norm_test(bootstrap_sigs)
            
            if bootstrap_stat >= original_stat:
                exceed_count += 1
        
        p_value = exceed_count / n_bootstrap
        return p_value
    
    # Run bootstrap test
    p_value = simple_bootstrap(signatures_by_lp, n_bootstrap=1000)
    print(f"Bootstrap p-value: {p_value}")
    
    if p_value < 0.05:
        print("Conclusion: There are significant differences between LP signatures")
    else:
        print("Conclusion: No significant differences between LP signatures")
        
    # Adaptive block bootstrap implementation (simplified)
    def adaptive_block_bootstrap(data, signature_horizon=30, n_bootstrap=1000):
        """
        Simplified implementation of adaptive block bootstrap from the paper
        """
        print("Implementing adaptive block bootstrap...")
        
        # Sort by time
        data = data.sort_values('timestamp')
        
        # Create blocks based on time gaps
        block_starts = [0]
        
        for i in range(1, len(data)):
            time_gap = (data.iloc[i]['timestamp'] - data.iloc[i-1]['timestamp']).total_seconds() / 60
            if time_gap > signature_horizon:
                block_starts.append(i)
        
        block_starts.append(len(data))
        
        # Create blocks
        blocks = []
        for i in range(len(block_starts) - 1):
            start = block_starts[i]
            end = block_starts[i+1]
            blocks.append(data.iloc[start:end])
        
        print(f"Created {len(blocks)} adaptive blocks")
        
        return blocks
    
    # Run adaptive block bootstrap
    blocks = adaptive_block_bootstrap(intraday_data, signature_horizon=30)
    
    # Print block statistics
    block_sizes = [len(block) for block in blocks]
    print(f"Block statistics: min={min(block_sizes)}, max={max(block_sizes)}, avg={np.mean(block_sizes):.1f}")
    
    # Summary
    print("\nSummary of results:")
    print(f"- Total trades analyzed: {len(intraday_data)}")
    print(f"- Number of LPs: {len(signatures_by_lp)}")
    print(f"- L2-norm test statistic: {l2_stat:.4f}")
    print(f"- Bootstrap p-value: {p_value:.4f}")
    
    # Create different LP profiles to test detection
    print("\nSimulating different LP profiles...")
    
    # Create a new dataset with distinct LP behaviors
    def generate_lp_profiles():
        """Generate data with distinct LP behaviors"""
        np.random.seed(43)
        
        # Start from 5 days ago
        start_date = datetime.now() - timedelta(days=5)
        
        # Common parameters
        n_trades = 1000
        base_price = latest_price
        
        # Generate timestamps
        timestamps = []
        for _ in range(n_trades):
            day_offset = np.random.randint(0, 5)
            minute_offset = np.random.randint(0, 9 * 60)  # 9 hours of trading
            ts = start_date + timedelta(days=day_offset, minutes=minute_offset)
            timestamps.append(ts)
        
        timestamps.sort()
        
        # Base price series (common to all LPs)
        prices = [base_price]
        for i in range(1, len(timestamps)):
            time_diff = (timestamps[i] - timestamps[i-1]).total_seconds() / 3600
            price_change = np.random.normal(0.0001 * time_diff, 0.001 * np.sqrt(time_diff))
            prices.append(prices[-1] * np.exp(price_change))
        
        # Generate LP data with different behaviors
        all_data = []
        
        # LP1: Internalizer (minimal market impact)
        lp1_data = pd.DataFrame({
            'timestamp': timestamps[:int(n_trades/3)],
            'price': prices[:int(n_trades/3)],
            'direction': np.random.choice([1, -1], size=int(n_trades/3)),
            'size': np.exp(np.random.normal(14, 0.5, int(n_trades/3))),
            'lp': 'Internalizer'
        })
        
        # Add minimal post-trade impact
        for i in range(len(lp1_data)):
            # Only slight price decay (5 bps over 30 min)
            decay_factor = 0.0005 * lp1_data.loc[i, 'direction']
            
            # Find trades within next 30 minutes
            start_time = lp1_data.loc[i, 'timestamp']
            end_time = start_time + pd.Timedelta(minutes=30)
            
            affected_indices = lp1_data[
                (lp1_data['timestamp'] > start_time) & 
                (lp1_data['timestamp'] <= end_time)
            ].index
            
            # Apply decay proportional to time difference
            for j in affected_indices:
                time_diff = (lp1_data.loc[j, 'timestamp'] - start_time).total_seconds() / 1800  # normalize to 30 min
                lp1_data.loc[j, 'price'] *= (1 - decay_factor * time_diff)
        
        # LP2: Externalizer (significant market impact)
        lp2_data = pd.DataFrame({
            'timestamp': timestamps[int(n_trades/3):int(2*n_trades/3)],
            'price': prices[int(n_trades/3):int(2*n_trades/3)],
            'direction': np.random.choice([1, -1], size=int(n_trades/3)),
            'size': np.exp(np.random.normal(14, 0.5, int(n_trades/3))),
            'lp': 'Externalizer'
        })
        
        # Add significant post-trade impact
        for i in range(len(lp2_data)):
            # Strong price decay (30 bps over 30 min)
            decay_factor = 0.003 * lp2_data.loc[i, 'direction']
            
            # Find trades within next 30 minutes
            start_time = lp2_data.loc[i, 'timestamp']
            end_time = start_time + pd.Timedelta(minutes=30)
            
            affected_indices = lp2_data[
                (lp2_data['timestamp'] > start_time) & 
                (lp2_data['timestamp'] <= end_time)
            ].index
            
            # Apply decay proportional to time difference
            for j in affected_indices:
                time_diff = (lp2_data.loc[j, 'timestamp'] - start_time).total_seconds() / 1800
                lp2_data.loc[j, 'price'] *= (1 - decay_factor * time_diff)
        
        # LP3: Mixed strategy
        lp3_data = pd.DataFrame({
            'timestamp': timestamps[int(2*n_trades/3):],
            'price': prices[int(2*n_trades/3):],
            'direction': np.random.choice([1, -1], size=n_trades-int(2*n_trades/3)),
            'size': np.exp(np.random.normal(14, 0.5, n_trades-int(2*n_trades/3))),
            'lp': 'Mixed'
        })
        
        # Add moderate post-trade impact
        for i in range(len(lp3_data)):
            # Moderate price decay (15 bps over 30 min)
            decay_factor = 0.0015 * lp3_data.loc[i, 'direction']
            
            # Find trades within next 30 minutes
            start_time = lp3_data.loc[i, 'timestamp']
            end_time = start_time + pd.Timedelta(minutes=30)
            
            affected_indices = lp3_data[
                (lp3_data['timestamp'] > start_time) & 
                (lp3_data['timestamp'] <= end_time)
            ].index
            
            # Apply decay proportional to time difference
            for j in affected_indices:
                time_diff = (lp3_data.loc[j, 'timestamp'] - start_time).total_seconds() / 1800
                lp3_data.loc[j, 'price'] *= (1 - decay_factor * time_diff)
        
        # Combine data
        profile_data = pd.concat([lp1_data, lp2_data, lp3_data], ignore_index=True)
        profile_data = profile_data.sort_values('timestamp').reset_index(drop=True)
        
        return profile_data

    # Generate LP profile data
    profile_data = generate_lp_profiles()
    
    # Calculate signatures by LP profile
    profile_groups = profile_data.groupby('lp')
    profile_signatures = {}
    
    for lp, group_data in profile_groups:
        profile_signatures[lp] = calculate_signature(
            group_data,
            horizon=30,
            direction_col='direction',
            price_col='price',
            size_col='size',
            time_col='timestamp'
        )
    
    # Plot LP profile signatures
    plt.figure(figsize=(12, 6))
    for lp, sig in profile_signatures.items():
        plt.plot(sig.index, sig.values, marker='o', label=f'{lp}')
    
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='r', linestyle='-', alpha=0.3)
    plt.title('Price Signatures by LP Profile (Simulated)')
    plt.xlabel('Time relative to execution (minutes)')
    plt.ylabel('Signature Value (log price change)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('price_signatures_by_profile.png')
    print("\nLP profile signatures calculated and saved to price_signatures_by_profile.png")
    
    # Calculate L2-norm test for profiles
    l2_stat_profiles = l2_norm_test(profile_signatures)
    p_value_profiles = simple_bootstrap(profile_signatures, n_bootstrap=1000)
    
    print(f"\nL2-norm test for LP profiles: {l2_stat_profiles:.4f}")
    print(f"Bootstrap p-value: {p_value_profiles:.4f}")
    
    if p_value_profiles < 0.05:
        print("Conclusion: There are significant differences between LP profiles")
    else:
        print("Conclusion: No significant differences between LP profiles")
    
    # ================= TRADING STRATEGY IMPLEMENTATION =================
    
    def price_signature_trading_strategy(historical_data, incoming_trade_size=1000000, direction=1):
        """
        Smart order routing strategy based on price signatures
        
        Parameters:
        -----------
        historical_data : DataFrame of historical trades with LP information
        incoming_trade_size : Size of trade to execute
        direction : Trade direction (1=buy, -1=sell)
        
        Returns:
        --------
        Allocation of trade across different LPs to minimize market impact
        """
        print("\n======= Price Signature Trading Strategy =======")
        print(f"Executing {'BUY' if direction == 1 else 'SELL'} order of size {incoming_trade_size:,.0f}")
        
        # 1. Calculate price signatures for each LP
        lp_groups = historical_data.groupby('lp')
        signatures_by_lp = {}
        
        for lp, group_data in lp_groups:
            signatures_by_lp[lp] = calculate_signature(
                group_data,
                horizon=30,
                direction_col='direction',
                price_col='price',
                size_col='size',
                time_col='timestamp'
            )
        
        # 2. Calculate post-trade impact measure for each LP
        lp_impact = {}
        for lp, sig in signatures_by_lp.items():
            # Measure post-trade impact (average signature over first 5 minutes)
            # Multiply by direction to make it comparable across buys and sells
            post_impact = sig[sig.index > 0].mean() * direction
            lp_impact[lp] = post_impact
        
        # 3. Group LPs into categories based on impact
        lp_names = list(lp_impact.keys())
        lp_impact_values = np.array([lp_impact[lp] for lp in lp_names]).reshape(-1, 1)
        
        # Use K-means to cluster LPs (with k=3 for internalizer, mixed, externalizer)
        if len(lp_names) >= 3:
            kmeans = KMeans(n_clusters=3, random_state=42).fit(lp_impact_values)
            clusters = kmeans.labels_
            
            # Sort clusters by impact level (0=low, 1=medium, 2=high impact)
            cluster_centers = kmeans.cluster_centers_.flatten()
            cluster_order = np.argsort(cluster_centers)
            
            # Assign LP types
            lp_types = {}
            for i, lp in enumerate(lp_names):
                cluster_id = clusters[i]
                impact_rank = np.where(cluster_order == cluster_id)[0][0]
                
                if impact_rank == 0:
                    lp_types[lp] = "Internalizer"
                elif impact_rank == 1:
                    lp_types[lp] = "Mixed"
                else:
                    lp_types[lp] = "Externalizer" 
        else:
            # If we have fewer than 3 LPs, use simple thresholds
            lp_types = {}
            for lp in lp_names:
                impact = lp_impact[lp]
                if impact < 0.0001:  # very small impact
                    lp_types[lp] = "Internalizer"
                elif impact < 0.0005:  # moderate impact
                    lp_types[lp] = "Mixed"
                else:
                    lp_types[lp] = "Externalizer"
        
        # Print LP classification
        print("\nLP Classification:")
        for lp, lp_type in lp_types.items():
            print(f"  {lp}: {lp_type} (Impact: {lp_impact[lp]:.6f})")
        
        # 4. Create execution strategy based on LP types
        # Strategy:
        # 1. Prioritize internalizers for majority of flow
        # 2. Include some mixed LPs for price competition
        # 3. Minimize or avoid externalizers except for necessary liquidity
        
        internalizers = [lp for lp, tp in lp_types.items() if tp == "Internalizer"]
        mixed_lps = [lp for lp, tp in lp_types.items() if tp == "Mixed"]
        externalizers = [lp for lp, tp in lp_types.items() if tp == "Externalizer"]
        
        # Calculate average trade sizes by LP type to understand liquidity
        avg_size_by_lp = historical_data.groupby('lp')['size'].mean()
        
        # Calculate allocation percentages
        allocation = {}
        
        if internalizers:
            # Prioritize internalizers (70-80% of flow)
            internalizer_alloc = 0.75
            
            # Distribute based on relative impact within the group
            impacts = np.array([lp_impact[lp] for lp in internalizers])
            # Invert and normalize (lower impact gets more allocation)
            if np.max(impacts) > 0:
                weights = (np.max(impacts) - impacts) / np.sum(np.max(impacts) - impacts)
            else:
                weights = np.ones_like(impacts) / len(impacts)
            
            for i, lp in enumerate(internalizers):
                allocation[lp] = internalizer_alloc * weights[i]
        
        if mixed_lps:
            # Give mixed LPs some share (15-25% of flow)
            mixed_alloc = 0.20
            
            # Distribute based on relative impact
            impacts = np.array([lp_impact[lp] for lp in mixed_lps])
            if np.max(impacts) > 0:
                weights = (np.max(impacts) - impacts) / np.sum(np.max(impacts) - impacts)
            else:
                weights = np.ones_like(impacts) / len(impacts)
            
            for i, lp in enumerate(mixed_lps):
                allocation[lp] = mixed_alloc * weights[i]
        
        if externalizers:
            # Minimize externalizers (0-10% of flow)
            externalizer_alloc = 0.05
            
            # Distribute based on relative impact
            impacts = np.array([lp_impact[lp] for lp in externalizers])
            if np.max(impacts) > 0:
                weights = (np.max(impacts) - impacts) / np.sum(np.max(impacts) - impacts)
            else:
                weights = np.ones_like(impacts) / len(impacts)
            
            for i, lp in enumerate(externalizers):
                allocation[lp] = externalizer_alloc * weights[i]
        
        # Normalize to ensure percentages sum to 1
        total_alloc = sum(allocation.values())
        for lp in allocation:
            allocation[lp] /= total_alloc
        
        # 5. Convert percentages to actual trade sizes
        trade_sizes = {}
        for lp, percentage in allocation.items():
            trade_sizes[lp] = incoming_trade_size * percentage
        
        # 6. Consider typical LP liquidity constraints
        # If allocation exceeds typical trade size significantly, redistribute excess
        excess = 0
        for lp in trade_sizes.copy():
            typical_size = avg_size_by_lp.get(lp, incoming_trade_size / len(allocation))
            if trade_sizes[lp] > 3 * typical_size:  # If exceeds 3x typical size
                excess += trade_sizes[lp] - 3 * typical_size
                trade_sizes[lp] = 3 * typical_size
        
        # Redistribute excess according to initial allocation ratios
        if excess > 0:
            total_original = sum(allocation.values())
            for lp in trade_sizes:
                trade_sizes[lp] += excess * (allocation[lp] / total_original)
        
        # 7. Round to nice numbers for FX trading
        for lp in trade_sizes:
            trade_sizes[lp] = round(trade_sizes[lp] / 10000) * 10000  # Round to nearest 10K
        
        # 8. Split large trades over time if needed
        time_splits = {}
        for lp, size in trade_sizes.items():
            typical_size = avg_size_by_lp.get(lp, incoming_trade_size / len(allocation))
            
            if size > 5 * typical_size:
                # Split into multiple trades over time
                n_splits = min(10, int(size / typical_size))
                split_size = size / n_splits
                time_splits[lp] = [(split_size, f"+{i*5}min") for i in range(n_splits)]
            else:
                time_splits[lp] = [(size, "now")]
        
        # Print execution strategy
        print("\nExecution Strategy:")
        for lp, splits in time_splits.items():
            for size, timing in splits:
                print(f"  {lp}: {'BUY' if direction == 1 else 'SELL'} {size:,.0f} @ {timing}")
        
        # 9. Calculate expected execution cost based on LP signatures
        expected_impact = 0
        for lp, splits in time_splits.items():
            lp_total_size = sum(size for size, _ in splits)
            lp_weight = lp_total_size / incoming_trade_size
            expected_impact += lp_impact[lp] * lp_weight
        
        # Compare to worst-case (all to highest impact LP)
        worst_lp = max(lp_impact.items(), key=lambda x: x[1])[0]
        worst_impact = lp_impact[worst_lp]
        
        print("\nExpected Execution Performance:")
        print(f"  Optimized Impact: {expected_impact*10000:.2f} bps")
        print(f"  Worst-case Impact: {worst_impact*10000:.2f} bps")
        print(f"  Improvement: {(worst_impact-expected_impact)*10000:.2f} bps ({(1-expected_impact/worst_impact)*100:.1f}%)")
        
        # 10. Implement execution monitoring
        print("\nExecution Monitoring Plan:")
        print("  1. Record actual execution prices and timestamps")
        print("  2. Calculate realized price signature after execution")
        print("  3. Compare actual vs. expected impact")
        print("  4. Update LP profiles based on actual performance")
        
        return time_splits

    # Simulation of strategy execution
    def simulate_strategy_execution(trade_plan, historical_data):
        """Simulate execution of the trade plan"""
        print("\n======= Execution Simulation =======")
        
        # Get current market conditions
        current_time = datetime.now()
        current_price = historical_data['price'].iloc[-1]
        
        print(f"Current time: {current_time}")
        print(f"Current price: {current_price:.6f}")
        
        # Create execution report
        execution_report = []
        
        for lp, splits in trade_plan.items():
            for size, timing in splits:
                # Parse timing
                if timing == "now":
                    execution_time = current_time
                else:
                    minutes = int(timing.replace("+", "").replace("min", ""))
                    execution_time = current_time + timedelta(minutes=minutes)
                
                # Simulate execution price with slippage based on LP profile
                lp_data = historical_data[historical_data['lp'] == lp]
                avg_impact = calculate_signature(lp_data)
                
                # Get impact at 1-minute mark as rough measure of immediate slippage
                if 1 in avg_impact.index:
                    slippage_factor = avg_impact[1] 
                else:
                    slippage_factor = 0.0001  # Default slippage
                
                # Adjustment for trade size (larger trades have more impact)
                size_adjustment = np.log1p(size / lp_data['size'].mean()) / 10
                
                # Calculate execution price with slippage
                execution_price = current_price * (1 + slippage_factor * size_adjustment)
                
                # Add to execution report
                execution_report.append({
                    "LP": lp,
                    "Size": size,
                    "Time": execution_time,
                    "Price": execution_price,
                    "Slippage (bps)": (execution_price/current_price - 1) * 10000
                })
        
        # Create DataFrame from report
        execution_df = pd.DataFrame(execution_report)
        
        # Calculate volume-weighted average execution price
        vwap = (execution_df['Price'] * execution_df['Size']).sum() / execution_df['Size'].sum()
        
        # Calculate overall slippage
        overall_slippage_bps = (vwap/current_price - 1) * 10000
        
        print("\nExecution Summary:")
        print(f"  Volume-Weighted Average Price: {vwap:.6f}")
        print(f"  Overall Slippage: {overall_slippage_bps:.2f} bps")
        print("\nDetailed Execution Report:")
        print(execution_df.to_string(index=False))
        
        return execution_df

    # Implement the full trading strategy
    def implement_trading_strategy(profile_data):
        """Implement the full trading strategy"""
        # Calculate optimal trade routing
        trade_plan = price_signature_trading_strategy(
            profile_data, 
            incoming_trade_size=5000000,  # 5 million
            direction=1  # Buy
        )
        
        # Simulate execution
        execution_results = simulate_strategy_execution(trade_plan, profile_data)
        
        # Plot execution performance
        plt.figure(figsize=(12, 8))
        
        # 1. Bar chart of slippage by LP
        plt.subplot(2, 1, 1)
        slippage_by_lp = execution_results.groupby('LP')['Slippage (bps)'].mean()
        slippage_by_lp.plot(kind='bar', color='skyblue')
        plt.title('Average Slippage by Liquidity Provider')
        plt.ylabel('Slippage (bps)')
        plt.grid(axis='y', alpha=0.3)
        
        # 2. Execution timeline
        plt.subplot(2, 1, 2)
        for lp in execution_results['LP'].unique():
            lp_data = execution_results[execution_results['LP'] == lp]
            plt.scatter(lp_data['Time'], lp_data['Price'], 
                       s=lp_data['Size']/50000, label=lp, alpha=0.7)
        
        plt.axhline(y=profile_data['price'].iloc[-1], color='r', linestyle='--', label='Market price')
        plt.title('Execution Timeline')
        plt.xlabel('Time')
        plt.ylabel('Execution Price')
        plt.legend()
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('execution_performance.png')
        print("\nExecution performance chart saved to execution_performance.png")
        
        return execution_results
    
    # Apply trading strategy to our profile data
    if 'profile_data' in locals():
        implement_trading_strategy(profile_data)
    else:
        print("No profile data available for strategy implementation")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Close Bloomberg connection
    con.stop()