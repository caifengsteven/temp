#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Automatic 1-2-3 Trading Strategy

This implementation is based on the paper:
"Automatic One Two Three" by Stanislaus Maier-Paape (2015)

The strategy automatically identifies market-technical trends by detecting
sequences of alternating minima and maxima, and uses these to generate
trading signals.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
from typing import List, Dict, Tuple, Optional, Union
import logging
from bloomberg_data_fetcher import BloombergDataFetcher

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configure matplotlib
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 12


class AutomaticOneTwoThree:
    """
    Implementation of the Automatic 1-2-3 trading strategy as described in:
    "Automatic One Two Three" by Stanislaus Maier-Paape (2015)
    
    The strategy detects market-technical trends by identifying sequences of
    alternating minima and maxima, using a stop and reverse (SAR) process.
    """
    
    def __init__(self, 
                 sar_method: str = 'macd',
                 macd_fast: int = 12, 
                 macd_slow: int = 26, 
                 macd_signal: int = 9,
                 delta_coef: float = 0.001,
                 choice: int = 0,
                 time_scale: float = 1.0):
        """
        Initialize the Automatic 1-2-3 strategy.
        
        Args:
            sar_method: Method for the stop and reverse process ('macd', 'dist_macd', 'integ_macd', 'renko')
            macd_fast: Fast period for MACD calculation
            macd_slow: Slow period for MACD calculation
            macd_signal: Signal period for MACD calculation
            delta_coef: Coefficient for the threshold in dist_macd and integ_macd methods
            choice: Parameter controlling whether minima and maxima can occur in the same period (0 or 1)
            time_scale: Scaling factor for MACD parameters
        """
        self.sar_method = sar_method
        self.macd_fast = int(macd_fast * time_scale)
        self.macd_slow = int(macd_slow * time_scale)
        self.macd_signal = int(macd_signal * time_scale)
        self.delta_coef = delta_coef
        self.choice = choice
        
        # Initialize series for the algorithm
        self.direction = None  # SAR process direction
        self.status = None     # Status series (Direction modified by exceptional situations)
        self.excep = None      # Exceptional situations
        
        # Variables for tracking extrema
        self.lastminbar = None  # Period of last confirmed minimum
        self.lastmaxbar = None  # Period of last confirmed maximum
        self.tempminbar = None  # Period of temporary minimum
        self.tempmaxbar = None  # Period of temporary maximum
        
        # MinMax process
        self.minmaxvalue = None  # Series containing all relevant minima and maxima
        
        # Trend indicators
        self.currenttrend = None    # Current trend status
        self.upprephase = None      # Indicates possible upcoming up-trend
        self.downprephase = None    # Indicates possible upcoming down-trend
        self.uptrigger = None       # Point 2 level in up-trend
        self.downtrigger = None     # Point 2 level in down-trend
        self.stoptrigger = None     # Point 3 level (stop level)
        self.movementtrigger = None # Signals when point 2 is reached
        
    def compute_macd_direction(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute the basic MACD-based direction process.
        
        Args:
            data: DataFrame with OHLC price data
            
        Returns:
            Series with direction values (+1 or -1)
        """
        # Calculate MACD
        exp1 = data['close'].ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = data['close'].ewm(span=self.macd_slow, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        
        # Determine direction based on MACD vs Signal line
        direction = pd.Series(index=data.index)
        direction[macd >= signal] = 1    # Bullish
        direction[macd < signal] = -1    # Bearish
        
        return direction
    
    def compute_dist_macd_direction(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute the distance MACD direction process with threshold.
        
        Args:
            data: DataFrame with OHLC price data
            
        Returns:
            Series with direction values (+1 or -1)
        """
        # Calculate MACD
        exp1 = data['close'].ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = data['close'].ewm(span=self.macd_slow, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        
        # Calculate threshold delta (as percentage of closing price)
        delta = self.delta_coef * data['close']
        
        # Initialize direction series
        direction = pd.Series(index=data.index)
        direction.iloc[0] = 1 if macd.iloc[0] >= signal.iloc[0] else -1
        
        # Process the series
        for i in range(1, len(direction)):
            diff = macd.iloc[i] - signal.iloc[i]
            
            # Check for direction change with threshold
            if direction.iloc[i-1] == 1:
                if diff <= -delta.iloc[i]:  # Change from +1 to -1
                    direction.iloc[i] = -1
                else:
                    direction.iloc[i] = 1
            else:  # direction.iloc[i-1] == -1
                if diff >= delta.iloc[i]:   # Change from -1 to +1
                    direction.iloc[i] = 1
                else:
                    direction.iloc[i] = -1
        
        return direction
    
    def compute_integ_macd_direction(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute the integral MACD direction process with threshold.
        
        Args:
            data: DataFrame with OHLC price data
            
        Returns:
            Series with direction values (+1 or -1)
        """
        # Calculate MACD
        exp1 = data['close'].ewm(span=self.macd_fast, adjust=False).mean()
        exp2 = data['close'].ewm(span=self.macd_slow, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=self.macd_signal, adjust=False).mean()
        
        # Calculate difference
        diff = macd - signal
        
        # Calculate threshold delta (as percentage of closing price)
        delta = self.delta_coef * data['close']
        
        # Initialize direction series
        direction = pd.Series(index=data.index)
        direction.iloc[0] = 1 if macd.iloc[0] >= signal.iloc[0] else -1
        
        # Process the series
        for i in range(1, len(direction)):
            # Calculate integral
            if diff.iloc[i] * diff.iloc[i-1] >= 0:  # Same sign
                # Find the index where the sign started
                j = i - 1
                while j > 0 and diff.iloc[j] * diff.iloc[j-1] >= 0:
                    j -= 1
                
                # Calculate integral from j to i
                integral = diff.iloc[j:i+1].sum()
            else:
                integral = diff.iloc[i]
                
            # Check for direction change with threshold
            if direction.iloc[i-1] == 1:
                if integral <= -delta.iloc[i]:  # Change from +1 to -1
                    direction.iloc[i] = -1
                else:
                    direction.iloc[i] = 1
            else:  # direction.iloc[i-1] == -1
                if integral >= delta.iloc[i]:   # Change from -1 to +1
                    direction.iloc[i] = 1
                else:
                    direction.iloc[i] = -1
        
        return direction
    
    def compute_renko_direction(self, data: pd.DataFrame, box_size_pct: float = 0.5, rev_amount: int = 2) -> pd.Series:
        """
        Compute the Range Renko direction process.
        
        Args:
            data: DataFrame with OHLC price data
            box_size_pct: Box size as percentage of close price
            rev_amount: Minimum number of boxes for reversal
            
        Returns:
            Series with direction values (+1 or -1)
        """
        # Initialize direction and brick levels
        direction = pd.Series(index=data.index)
        brick_high = pd.Series(index=data.index)
        brick_low = pd.Series(index=data.index)
        
        # Set initial values
        direction.iloc[0] = 1  # Assume up-trend initially
        brick_high.iloc[0] = data['high'].iloc[0]
        brick_low.iloc[0] = data['low'].iloc[0]
        
        # Process the series
        for i in range(1, len(direction)):
            # Calculate box size for this period
            box_size = box_size_pct / 100 * data['close'].iloc[i]
            min_reversal = box_size * rev_amount
            
            if direction.iloc[i-1] == 1:  # Previous direction is up
                # Update brick high if higher
                if data['high'].iloc[i] > brick_high.iloc[i-1]:
                    brick_high.iloc[i] = data['high'].iloc[i]
                else:
                    brick_high.iloc[i] = brick_high.iloc[i-1]
                
                # Check for reversal
                if brick_high.iloc[i] - data['low'].iloc[i] >= box_size * (1 + rev_amount):
                    direction.iloc[i] = -1  # Reverse to down-trend
                    brick_low.iloc[i] = data['low'].iloc[i]
                else:
                    direction.iloc[i] = 1
                    brick_low.iloc[i] = brick_low.iloc[i-1]
            
            else:  # Previous direction is down
                # Update brick low if lower
                if data['low'].iloc[i] < brick_low.iloc[i-1]:
                    brick_low.iloc[i] = data['low'].iloc[i]
                else:
                    brick_low.iloc[i] = brick_low.iloc[i-1]
                
                # Check for reversal
                if data['high'].iloc[i] - brick_low.iloc[i] >= box_size * (1 + rev_amount):
                    direction.iloc[i] = 1  # Reverse to up-trend
                    brick_high.iloc[i] = data['high'].iloc[i]
                else:
                    direction.iloc[i] = -1
                    brick_high.iloc[i] = brick_high.iloc[i-1]
        
        return direction
    
    def compute_direction(self, data: pd.DataFrame) -> pd.Series:
        """
        Compute the direction series using the selected SAR method.
        
        Args:
            data: DataFrame with OHLC price data
            
        Returns:
            Series with direction values (+1 or -1)
        """
        if self.sar_method == 'macd':
            return self.compute_macd_direction(data)
        elif self.sar_method == 'dist_macd':
            return self.compute_dist_macd_direction(data)
        elif self.sar_method == 'integ_macd':
            return self.compute_integ_macd_direction(data)
        elif self.sar_method == 'renko':
            return self.compute_renko_direction(data)
        else:
            raise ValueError(f"Unknown SAR method: {self.sar_method}")
    
    def compute_excep_and_status(self, data: pd.DataFrame, direction: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        Compute the exceptional situations and status series.
        
        Args:
            data: DataFrame with OHLC price data
            direction: Direction series from SAR process
            
        Returns:
            Tuple of (excep, status) series
        """
        # Initialize series
        excep = pd.Series(1, index=data.index)  # Default is 1 (no exception)
        
        # Initialize lastminbar and lastmaxbar if not already set
        if self.lastminbar is None:
            self.lastminbar = data['low'].idxmin()
        if self.lastmaxbar is None:
            self.lastmaxbar = data['high'].idxmax()
        
        # Process the series
        for i in range(1, len(excep)):
            if excep.iloc[i-1] == -1:  # Exceptional process was already active
                # Check if we can return to normal
                if direction.iloc[i-1] * direction.iloc[i] == -1:  # Direction changed
                    excep.iloc[i] = 1
                elif direction.iloc[i-1] == 1 and data['high'].iloc[i] >= data.loc[self.lastmaxbar, 'high']:
                    excep.iloc[i] = 1
                elif direction.iloc[i-1] == -1 and data['low'].iloc[i] <= data.loc[self.lastminbar, 'low']:
                    excep.iloc[i] = 1
                else:
                    excep.iloc[i] = -1  # Stay in exceptional state
            
            elif direction.iloc[i-1] == direction.iloc[i]:  # Check for exceptional situation
                if direction.iloc[i] == 1 and data['low'].iloc[i] <= data.loc[self.lastminbar, 'low']:
                    excep.iloc[i] = -1  # Enter exceptional state
                elif direction.iloc[i] == -1 and data['high'].iloc[i] >= data.loc[self.lastmaxbar, 'high']:
                    excep.iloc[i] = -1  # Enter exceptional state
                else:
                    excep.iloc[i] = 1  # Normal state
            else:
                excep.iloc[i] = 1  # Normal state after direction change
        
        # Calculate status = excep * direction
        status = excep * direction
        
        return excep, status
    
    def compute_minmax_process(self, data: pd.DataFrame, direction: pd.Series, status: pd.Series) -> pd.Series:
        """
        Compute the minmax process (alternating minima and maxima).
        
        Args:
            data: DataFrame with OHLC price data
            direction: Direction series from SAR process
            status: Status series
            
        Returns:
            Series with minmax values
        """
        # Initialize the minmax value series
        minmaxvalue = pd.Series(index=data.index)
        
        # Initialize temporary extrema if not set yet
        if self.tempminbar is None:
            self.tempminbar = self.lastminbar
        if self.tempmaxbar is None:
            self.tempmaxbar = self.lastmaxbar
        
        # Process the series
        for i in range(1, len(status)):
            if status.iloc[i-1] == 1:  # Looking for maximum
                # Update temporary maximum if higher
                if data['high'].iloc[i] > data.loc[self.tempmaxbar, 'high']:
                    self.tempmaxbar = data.index[i]
                
                # Check if status has changed
                if status.iloc[i] == -1:
                    # Fix the last maximum
                    self.lastmaxbar = self.tempmaxbar
                    
                    # Find the lowest low after lastmaxbar
                    if self.choice == 1 or (self.lastminbar is not None and self.lastmaxbar == self.lastminbar):
                        alpha = 1
                    else:
                        alpha = 0
                    
                    # Find tempminbar (lowest low after lastmaxbar + alpha)
                    mask = data.index > self.lastmaxbar
                    if alpha > 0:
                        mask = mask & (data.index > self.lastmaxbar)
                    
                    if mask.any():
                        self.tempminbar = data.loc[mask, 'low'].idxmin()
                    else:
                        # Use the next bar as tempminbar if no other option
                        next_idx = data.index.get_loc(self.lastmaxbar) + 1
                        if next_idx < len(data):
                            self.tempminbar = data.index[next_idx]
            
            elif status.iloc[i-1] == -1:  # Looking for minimum
                # Update temporary minimum if lower
                if data['low'].iloc[i] < data.loc[self.tempminbar, 'low']:
                    self.tempminbar = data.index[i]
                
                # Check if status has changed
                if status.iloc[i] == 1:
                    # Fix the last minimum
                    self.lastminbar = self.tempminbar
                    
                    # Find the highest high after lastminbar
                    if self.choice == 1 or (self.lastmaxbar is not None and self.lastmaxbar == self.lastminbar):
                        alpha = 1
                    else:
                        alpha = 0
                    
                    # Find tempmaxbar (highest high after lastminbar + alpha)
                    mask = data.index > self.lastminbar
                    if alpha > 0:
                        mask = mask & (data.index > self.lastminbar)
                    
                    if mask.any():
                        self.tempmaxbar = data.loc[mask, 'high'].idxmax()
                    else:
                        # Use the next bar as tempmaxbar if no other option
                        next_idx = data.index.get_loc(self.lastminbar) + 1
                        if next_idx < len(data):
                            self.tempmaxbar = data.index[next_idx]
            
            # Fill the minmaxvalue series
            if self.lastminbar is not None and self.lastmaxbar is not None:
                # Determine if current index is after a minimum or maximum
                if self.lastminbar > self.lastmaxbar:  # Last extremum was a minimum
                    if data.index[i] > self.lastminbar:
                        if status.iloc[i] == 1 and self.tempmaxbar is not None:
                            minmaxvalue.iloc[i] = data.loc[self.tempmaxbar, 'high']
                        else:
                            minmaxvalue.iloc[i] = data.loc[self.lastminbar, 'low']
                    elif data.index[i] > self.lastmaxbar:
                        minmaxvalue.iloc[i] = data.loc[self.lastminbar, 'low']
                    else:
                        minmaxvalue.iloc[i] = data.loc[self.lastmaxbar, 'high']
                else:  # Last extremum was a maximum
                    if data.index[i] > self.lastmaxbar:
                        if status.iloc[i] == -1 and self.tempminbar is not None:
                            minmaxvalue.iloc[i] = data.loc[self.tempminbar, 'low']
                        else:
                            minmaxvalue.iloc[i] = data.loc[self.lastmaxbar, 'high']
                    elif data.index[i] > self.lastminbar:
                        minmaxvalue.iloc[i] = data.loc[self.lastmaxbar, 'high']
                    else:
                        minmaxvalue.iloc[i] = data.loc[self.lastminbar, 'low']
            
            # Handle exception for bars containing both minimum and maximum
            if self.lastminbar is not None and self.lastmaxbar is not None:
                if self.lastminbar == self.lastmaxbar:
                    # Add both levels
                    next_idx = data.index.get_loc(self.lastminbar) + 1
                    if next_idx < len(data) and data.index[i] == data.index[next_idx]:
                        if i % 2 == 0:  # Alternate showing high/low
                            minmaxvalue.iloc[i] = data.loc[self.lastminbar, 'low']
                        else:
                            minmaxvalue.iloc[i] = data.loc[self.lastmaxbar, 'high']
        
        return minmaxvalue
    
    def detect_trends(self, data: pd.DataFrame, minmaxvalue: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Detect market-technical trends from the minmax process.
        
        Args:
            data: DataFrame with OHLC price data
            minmaxvalue: Series with minmax values
            
        Returns:
            Tuple containing (currenttrend, upprephase, downprephase, uptrigger, downtrigger, 
                               stoptrigger, movementtrigger)
        """
        # Extract the extrema series from minmaxvalue
        temp_extremum = pd.Series(index=data.index)
        last_extremum = pd.Series(index=data.index)
        before_last_extremum = pd.Series(index=data.index)
        third_last_extremum = pd.Series(index=data.index)
        forth_last_extremum = pd.Series(index=data.index)
        
        # Initialize trend series
        currenttrend = pd.Series(0, index=data.index)  # Default is no trend
        upprephase = pd.Series(0, index=data.index)
        downprephase = pd.Series(0, index=data.index)
        uptrigger = pd.Series(np.nan, index=data.index)
        downtrigger = pd.Series(np.nan, index=data.index)
        stoptrigger = pd.Series(np.nan, index=data.index)
        movementtrigger = pd.Series(0, index=data.index)
        
        # Extract extrema series
        extrema = []
        for i in range(len(minmaxvalue)):
            if not pd.isna(minmaxvalue.iloc[i]) and (i == 0 or minmaxvalue.iloc[i] != minmaxvalue.iloc[i-1]):
                extrema.append((data.index[i], minmaxvalue.iloc[i]))
        
        # Detect trends
        if len(extrema) >= 4:
            for i in range(len(data)):
                # Fill extrema series
                if i >= len(extrema):
                    # Use the last available extrema
                    temp_extremum.iloc[i] = extrema[-1][1]
                    last_extremum.iloc[i] = extrema[-2][1]
                    before_last_extremum.iloc[i] = extrema[-3][1]
                    third_last_extremum.iloc[i] = extrema[-4][1]
                    if len(extrema) >= 5:
                        forth_last_extremum.iloc[i] = extrema[-5][1]
                else:
                    # Fill based on the current position
                    idx = min(i, len(extrema) - 1)
                    if idx >= 0:
                        temp_extremum.iloc[i] = extrema[idx][1]
                    if idx - 1 >= 0:
                        last_extremum.iloc[i] = extrema[idx - 1][1]
                    if idx - 2 >= 0:
                        before_last_extremum.iloc[i] = extrema[idx - 2][1]
                    if idx - 3 >= 0:
                        third_last_extremum.iloc[i] = extrema[idx - 3][1]
                    if idx - 4 >= 0:
                        forth_last_extremum.iloc[i] = extrema[idx - 4][1]
                
                # Detect up-trend
                if (not pd.isna(last_extremum.iloc[i]) and not pd.isna(before_last_extremum.iloc[i]) and 
                    not pd.isna(third_last_extremum.iloc[i]) and not pd.isna(forth_last_extremum.iloc[i])):
                    
                    # Check for up-trend (rising minima and maxima)
                    is_uptrend = (
                        forth_last_extremum.iloc[i] <= before_last_extremum.iloc[i] and
                        third_last_extremum.iloc[i] < last_extremum.iloc[i]
                    )
                    
                    # Check for up-trend in question
                    is_uptrend_in_question = (
                        forth_last_extremum.iloc[i] <= before_last_extremum.iloc[i] and
                        third_last_extremum.iloc[i] >= last_extremum.iloc[i] and
                        data['close'].iloc[i] > third_last_extremum.iloc[i]
                    )
                    
                    # Check for down-trend (falling minima and maxima)
                    is_downtrend = (
                        forth_last_extremum.iloc[i] >= before_last_extremum.iloc[i] and
                        third_last_extremum.iloc[i] > last_extremum.iloc[i]
                    )
                    
                    # Check for down-trend in question
                    is_downtrend_in_question = (
                        forth_last_extremum.iloc[i] >= before_last_extremum.iloc[i] and
                        third_last_extremum.iloc[i] <= last_extremum.iloc[i] and
                        data['close'].iloc[i] < third_last_extremum.iloc[i]
                    )
                    
                    # Set current trend
                    if is_uptrend:
                        currenttrend.iloc[i] = 1
                    elif is_uptrend_in_question:
                        currenttrend.iloc[i] = 0.5
                    elif is_downtrend:
                        currenttrend.iloc[i] = -1
                    elif is_downtrend_in_question:
                        currenttrend.iloc[i] = -0.5
                    else:
                        currenttrend.iloc[i] = 0
                    
                    # Check for pre-phase trends
                    is_up_prephase = (
                        currenttrend.iloc[i] == 0 and
                        forth_last_extremum.iloc[i] <= before_last_extremum.iloc[i] and
                        third_last_extremum.iloc[i] < data['close'].iloc[i]
                    )
                    
                    is_down_prephase = (
                        currenttrend.iloc[i] == 0 and
                        forth_last_extremum.iloc[i] >= before_last_extremum.iloc[i] and
                        third_last_extremum.iloc[i] > data['close'].iloc[i]
                    )
                    
                    # Set pre-phase flags
                    upprephase.iloc[i] = 0.5 if is_up_prephase else 0
                    downprephase.iloc[i] = -0.5 if is_down_prephase else 0
                    
                    # Set triggers
                    if currenttrend.iloc[i] == 1 or currenttrend.iloc[i] == 0.5 or upprephase.iloc[i] == 0.5:
                        uptrigger.iloc[i] = before_last_extremum.iloc[i]  # Point 2 level
                    
                    if currenttrend.iloc[i] == -1 or currenttrend.iloc[i] == -0.5 or downprephase.iloc[i] == -0.5:
                        downtrigger.iloc[i] = before_last_extremum.iloc[i]  # Point 2 level
                    
                    if currenttrend.iloc[i] == 1 or currenttrend.iloc[i] == -1:
                        stoptrigger.iloc[i] = last_extremum.iloc[i]  # Point 3 level
                    
                    # Set movement trigger
                    if i > 0 and (
                        (currenttrend.iloc[i] == 1 and currenttrend.iloc[i-1] != 1) or
                        (currenttrend.iloc[i] == -1 and currenttrend.iloc[i-1] != -1)
                    ):
                        movementtrigger.iloc[i] = currenttrend.iloc[i]
        
        return currenttrend, upprephase, downprephase, uptrigger, downtrigger, stoptrigger, movementtrigger
    
    def fit(self, data: pd.DataFrame) -> 'AutomaticOneTwoThree':
        """
        Process the data to generate trading signals.
        
        Args:
            data: DataFrame with OHLC price data
            
        Returns:
            Self (processed strategy)
        """
        # Initialize extrema points
        self.lastminbar = data['low'].idxmin()
        self.lastmaxbar = data['high'].idxmax()
        self.tempminbar = self.lastminbar
        self.tempmaxbar = self.lastmaxbar
        
        # Compute direction using the selected SAR method
        self.direction = self.compute_direction(data)
        
        # Compute exceptional situations and status
        self.excep, self.status = self.compute_excep_and_status(data, self.direction)
        
        # Compute minmax process
        self.minmaxvalue = self.compute_minmax_process(data, self.direction, self.status)
        
        # Detect trends
        (self.currenttrend, self.upprephase, self.downprephase, 
         self.uptrigger, self.downtrigger, self.stoptrigger, 
         self.movementtrigger) = self.detect_trends(data, self.minmaxvalue)
        
        return self
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on the 1-2-3 trend indicator.
        
        Args:
            data: DataFrame with OHLC price data
            
        Returns:
            DataFrame with trading signals
        """
        # Initialize signals DataFrame
        signals = pd.DataFrame(index=data.index)
        signals['position'] = 0
        signals['stop_level'] = np.nan
        
        # Generate signals based on trend direction
        for i in range(1, len(signals)):
            # Trend trading signals
            if self.currenttrend.iloc[i] == 1 and self.currenttrend.iloc[i-1] != 1:
                # New up-trend - enter long
                signals.iloc[i, signals.columns.get_loc('position')] = 1
            elif self.currenttrend.iloc[i] == -1 and self.currenttrend.iloc[i-1] != -1:
                # New down-trend - enter short
                signals.iloc[i, signals.columns.get_loc('position')] = -1
            elif self.currenttrend.iloc[i] == 0.5 and self.currenttrend.iloc[i-1] == 1:
                # Up-trend in question - exit long
                signals.iloc[i, signals.columns.get_loc('position')] = 0
            elif self.currenttrend.iloc[i] == -0.5 and self.currenttrend.iloc[i-1] == -1:
                # Down-trend in question - exit short
                signals.iloc[i, signals.columns.get_loc('position')] = 0
            elif self.currenttrend.iloc[i] == 0 and (self.currenttrend.iloc[i-1] == 0.5 or self.currenttrend.iloc[i-1] == -0.5):
                # Trend in question resolved to no trend - maintain exit
                signals.iloc[i, signals.columns.get_loc('position')] = 0
            else:
                # Maintain previous position
                signals.iloc[i, signals.columns.get_loc('position')] = signals.iloc[i-1, signals.columns.get_loc('position')]
            
            # Add stop triggers
            if not pd.isna(self.stoptrigger.iloc[i]):
                if signals.iloc[i, signals.columns.get_loc('position')] == 1:
                    signals.iloc[i, signals.columns.get_loc('stop_level')] = self.stoptrigger.iloc[i]
                elif signals.iloc[i, signals.columns.get_loc('position')] == -1:
                    signals.iloc[i, signals.columns.get_loc('stop_level')] = self.stoptrigger.iloc[i]
            
            # Check if price hit stop level
            if i > 0 and not pd.isna(signals.iloc[i-1, signals.columns.get_loc('stop_level')]):
                if signals.iloc[i-1, signals.columns.get_loc('position')] == 1:
                    if data.iloc[i]['low'] <= signals.iloc[i-1, signals.columns.get_loc('stop_level')]:
                        signals.iloc[i, signals.columns.get_loc('position')] = 0  # Stop triggered, exit position
                
                elif signals.iloc[i-1, signals.columns.get_loc('position')] == -1:
                    if data.iloc[i]['high'] >= signals.iloc[i-1, signals.columns.get_loc('stop_level')]:
                        signals.iloc[i, signals.columns.get_loc('position')] = 0  # Stop triggered, exit position
        
        # Add movement trigger signals (point 2 breakout)
        signals['movement_trigger'] = self.movementtrigger
        
        return signals
    
    def backtest(self, data: pd.DataFrame, signals: pd.DataFrame, initial_capital: float = 10000.0) -> pd.DataFrame:
        """
        Backtest the trading strategy.
        
        Args:
            data: DataFrame with OHLC price data
            signals: DataFrame with trading signals
            initial_capital: Initial capital for the backtest
            
        Returns:
            DataFrame with backtest results
        """
        # Initialize portfolio DataFrame
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['position'] = signals['position']
        portfolio['close'] = data['close']
        
        # Calculate returns based on position changes
        portfolio['returns'] = data['close'].pct_change()
        portfolio['strategy_returns'] = portfolio['position'].shift(1) * portfolio['returns']
        
        # Calculate cumulative returns
        portfolio['cumulative_returns'] = (1 + portfolio['returns']).cumprod()
        portfolio['cumulative_strategy_returns'] = (1 + portfolio['strategy_returns']).cumprod()
        
        # Calculate equity curve
        portfolio['equity_curve'] = initial_capital * portfolio['cumulative_strategy_returns']
        
        # Calculate drawdown
        portfolio['peak'] = portfolio['equity_curve'].cummax()
        portfolio['drawdown'] = (portfolio['equity_curve'] - portfolio['peak']) / portfolio['peak']
        
        return portfolio
    
    def plot_results(self, data: pd.DataFrame, signals: pd.DataFrame, portfolio: pd.DataFrame) -> None:
        """
        Plot the backtest results.
        
        Args:
            data: DataFrame with OHLC price data
            signals: DataFrame with trading signals
            portfolio: DataFrame with backtest results
        """
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 15), gridspec_kw={'height_ratios': [3, 1, 1]})
        
        # Plot 1: Price chart with MinMax process and signals
        axes[0].plot(data.index, data['close'], label='Close Price', alpha=0.6)
        
        # Plot MinMax process
        valid_minmax = self.minmaxvalue.dropna()
        axes[0].plot(valid_minmax.index, valid_minmax, 'r-', label='MinMax Process', linewidth=1.5)
        
        # Plot entry and exit points
        for i in range(1, len(signals)):
            if signals['position'].iloc[i] == 1 and signals['position'].iloc[i-1] != 1:
                axes[0].plot(signals.index[i], data['close'].iloc[i], '^', markersize=10, color='g', label='Buy' if i == 1 else '')
            elif signals['position'].iloc[i] == -1 and signals['position'].iloc[i-1] != -1:
                axes[0].plot(signals.index[i], data['close'].iloc[i], 'v', markersize=10, color='r', label='Sell' if i == 1 else '')
            elif signals['position'].iloc[i] == 0 and signals['position'].iloc[i-1] != 0:
                axes[0].plot(signals.index[i], data['close'].iloc[i], 'o', markersize=8, color='black', alpha=0.7, label='Exit' if i == 1 else '')
        
        # Plot stop levels
        stop_levels = signals['stop_level'].dropna()
        
        # Process stop levels
        if not stop_levels.empty:
            prev_level = None
            for i, (timestamp, level) in enumerate(stop_levels.items()):
                # Only plot if it's a new level
                if prev_level is None or level != prev_level:
                    # Find how long this stop level was active
                    if i < len(stop_levels) - 1:
                        next_timestamp = stop_levels.index[i + 1]
                    else:
                        next_timestamp = data.index[-1]
                    
                    axes[0].plot([timestamp, next_timestamp], [level, level], 'k--', alpha=0.5)
                    prev_level = level
        
        # Plot trend state
        trend_colors = {
            1: 'green',     # Up-trend
            0.5: 'lightgreen',  # Up-trend in question
            0: 'gray',      # No trend
            -0.5: 'lightcoral',  # Down-trend in question
            -1: 'red'       # Down-trend
        }
        
        for i in range(1, len(self.currenttrend)):
            if self.currenttrend.iloc[i] != self.currenttrend.iloc[i-1]:
                # Plot shaded areas for trend state
                start_idx = self.currenttrend.index[i]
                if i < len(self.currenttrend) - 1:
                    end_idx = self.currenttrend.index[i+1]
                else:
                    end_idx = self.currenttrend.index[-1]
                
                trend_value = self.currenttrend.iloc[i]
                if trend_value in trend_colors:
                    axes[0].axvspan(start_idx, end_idx, alpha=0.2, color=trend_colors[trend_value])
        
        axes[0].set_title('Price Chart with 1-2-3 Trend Analysis')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot 2: Current trend
        axes[1].plot(self.currenttrend.index, self.currenttrend, 'b-', label='Current Trend')
        axes[1].fill_between(self.currenttrend.index, self.currenttrend, 0, where=self.currenttrend > 0, color='green', alpha=0.3, label='Up-trend')
        axes[1].fill_between(self.currenttrend.index, self.currenttrend, 0, where=self.currenttrend < 0, color='red', alpha=0.3, label='Down-trend')
        axes[1].set_ylabel('Trend State')
        axes[1].set_yticks([-1, -0.5, 0, 0.5, 1])
        axes[1].set_yticklabels(['Down', 'Down?', 'None', 'Up?', 'Up'])
        axes[1].grid(True)
        axes[1].legend()
        
        # Plot 3: Equity curve and drawdown
        axes[2].plot(portfolio.index, portfolio['equity_curve'], 'g-', label='Equity Curve')
        axes[2].set_ylabel('Equity ($)')
        axes[2].grid(True)
        axes[2].legend(loc='upper left')
        
        # Add drawdown as secondary axis
        ax_drawdown = axes[2].twinx()
        ax_drawdown.fill_between(portfolio.index, 0, portfolio['drawdown'], color='red', alpha=0.3)
        ax_drawdown.set_ylabel('Drawdown')
        ax_drawdown.set_ylim(-1, 0)
        
        # Format x-axis dates
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.savefig('automatic_123_backtest.png')
        plt.show()
    
    def calculate_performance_metrics(self, portfolio: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics for the strategy.
        
        Args:
            portfolio: DataFrame with backtest results
            
        Returns:
            Dictionary of performance metrics
        """
        # Calculate performance metrics
        total_return = portfolio['cumulative_strategy_returns'].iloc[-1] - 1
        
        # Annualized return (assuming 252 trading days per year)
        days = (portfolio.index[-1] - portfolio.index[0]).days
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        # Volatility and Sharpe ratio
        daily_returns = portfolio['strategy_returns'].dropna()
        volatility = daily_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        max_drawdown = portfolio['drawdown'].min()
        
        # Win rate
        trades = portfolio['position'].diff().fillna(0) != 0
        if trades.sum() > 0:
            wins = ((portfolio['strategy_returns'][trades] > 0).sum() / trades.sum())
        else:
            wins = 0
        
        # Calculate profit factor
        positive_returns = portfolio['strategy_returns'][portfolio['strategy_returns'] > 0].sum()
        negative_returns = abs(portfolio['strategy_returns'][portfolio['strategy_returns'] < 0].sum())
        profit_factor = positive_returns / negative_returns if negative_returns > 0 else float('inf')
        
        return {
            'Total Return': total_return,
            'Annual Return': annual_return,
            'Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Win Rate': wins,
            'Profit Factor': profit_factor,
            'Number of Trades': trades.sum()
        }


class MockBloombergDataFetcher:
    """
    Mock Bloomberg data fetcher for testing when Bloomberg is not available.
    """
    def __init__(self):
        pass
    
    def start_session(self):
        return False
    
    def stop_session(self):
        pass
    
    def get_intraday_bars(self, security, event_type, interval, start_date, end_date):
        return pd.DataFrame()


# Use the mock fetcher when the real one is not available
try:
    from bloomberg_data_fetcher import BloombergDataFetcher
except ImportError:
    logger.warning("Bloomberg API not available. Using mock data fetcher.")
    BloombergDataFetcher = MockBloombergDataFetcher


def fetch_bloomberg_data(tickers: List[str], 
                        start_date: datetime.datetime,
                        end_date: datetime.datetime,
                        interval: Optional[int] = None) -> Dict[str, pd.DataFrame]:
    """
    Fetch data from Bloomberg.
    
    Args:
        tickers: List of Bloomberg security identifiers
        start_date: Start date for data
        end_date: End date for data
        interval: Data interval in minutes (if None, daily data is fetched)
        
    Returns:
        Dictionary mapping tickers to DataFrames with OHLC data
    """
    logger.info(f"Fetching Bloomberg data for {len(tickers)} securities")
    
    # Initialize the Bloomberg data fetcher
    fetcher = BloombergDataFetcher()
    
    # Start Bloomberg session
    if not fetcher.start_session():
        logger.error("Failed to connect to Bloomberg. Using synthetic data instead.")
        return generate_synthetic_data(tickers, start_date, end_date)
    
    try:
        # Container for OHLC data
        all_data = {}
        
        # Fetch data for each ticker
        for ticker in tickers:
            if interval is None:
                # For daily data, use 1440 minute intervals
                df = fetcher.get_intraday_bars(
                    security=ticker,
                    event_type="TRADE",
                    interval=1440,  # Daily data
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                # For intraday data, use the specified interval
                df = fetcher.get_intraday_bars(
                    security=ticker,
                    event_type="TRADE",
                    interval=interval,
                    start_date=start_date,
                    end_date=end_date
                )
            
            if not df.empty:
                # Format the data
                ohlc_data = pd.DataFrame({
                    'open': df['open'],
                    'high': df['high'],
                    'low': df['low'],
                    'close': df['close'],
                    'volume': df['volume']
                }, index=df['time'])
                
                # Store in dictionary
                all_data[ticker] = ohlc_data
                logger.info(f"Retrieved {len(ohlc_data)} data points for {ticker}")
            else:
                logger.warning(f"No data retrieved for {ticker}")
        
        # If no data was retrieved, use synthetic data
        if not all_data:
            logger.warning("No data retrieved from Bloomberg. Using synthetic data instead.")
            return generate_synthetic_data(tickers, start_date, end_date)
        
        return all_data
    
    except Exception as e:
        logger.error(f"Error fetching Bloomberg data: {e}")
        return generate_synthetic_data(tickers, start_date, end_date)
    
    finally:
        # Stop Bloomberg session
        fetcher.stop_session()


def generate_synthetic_data(tickers: List[str],
                           start_date: datetime.datetime,
                           end_date: datetime.datetime) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic data for testing when Bloomberg is unavailable.
    
    Args:
        tickers: List of tickers
        start_date: Start date for data
        end_date: End date for data
        
    Returns:
        Dictionary mapping tickers to DataFrames with OHLC data
    """
    logger.info(f"Generating synthetic data for {len(tickers)} securities")
    
    # Create date range (business days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    
    # Generate synthetic data for each ticker
    all_data = {}
    
    for ticker in tickers:
        # Set random seed for reproducibility
        np.random.seed(hash(ticker) % 2**32)
        
        # Generate price data with trends, cycles, and noise
        n = len(dates)
        t = np.arange(n)
        
        # Base price depends on ticker type
        if 'SX5E' in ticker or 'SPX' in ticker:
            base_price = 3000  # For indices
        elif 'USD' in ticker or 'EUR' in ticker:
            base_price = 1.0  # For currencies
        else:
            base_price = 100  # For other securities
        
        # Trend component
        trend = 0.0001 * t
        
        # Cycle components
        annual_cycle = 0.05 * np.sin(2 * np.pi * t / 252)
        monthly_cycle = 0.02 * np.sin(2 * np.pi * t / 21)
        
        # Random walk component
        random_walk = np.random.normal(0, 0.01, n).cumsum()
        
        # Combine components
        close_prices = base_price * (1 + trend + annual_cycle + monthly_cycle + random_walk)
        
        # Generate OHLC data
        data = pd.DataFrame(index=dates)
        data['close'] = close_prices
        
        # Add some daily volatility for high, low, and open
        volatility = 0.005
        data['high'] = data['close'] * (1 + np.abs(np.random.normal(0, volatility, n)))
        data['low'] = data['close'] * (1 - np.abs(np.random.normal(0, volatility, n)))
        data['open'] = data['close'].shift(1)
        
        # Fill first open price
        data.loc[data.index[0], 'open'] = data.loc[data.index[0], 'close'] * (1 - np.random.normal(0, volatility))
        
        # Add volume
        data['volume'] = np.random.normal(1000000, 200000, n)
        
        # Store in dictionary
        all_data[ticker] = data
    
    return all_data


def main():
    """Main function to test the Automatic 1-2-3 trading strategy with multiple parameters."""
    logger.info("Testing Automatic 1-2-3 Trading Strategy")
    
    # Define test parameters
    ticker = "SPX Index"  # S&P 500 Index
    start_date = datetime.datetime(2015, 1, 1)
    end_date = datetime.datetime(2023, 12, 31)
    
    # Fetch data from Bloomberg (or generate synthetic if unavailable)
    data_dict = fetch_bloomberg_data([ticker], start_date, end_date)
    
    if ticker in data_dict:
        data = data_dict[ticker]
        logger.info(f"Retrieved {len(data)} data points for {ticker}")
        
        # Define parameter combinations to test
        parameter_sets = [
            # Method, fast, slow, signal, delta_coef, time_scale
            ('macd', 12, 26, 9, 0.001, 1.0),
            ('macd', 8, 17, 9, 0.001, 1.0),
            ('macd', 5, 35, 5, 0.001, 1.0),
            ('dist_macd', 12, 26, 9, 0.0005, 1.0),
            ('dist_macd', 12, 26, 9, 0.002, 1.0),
            ('integ_macd', 12, 26, 9, 0.001, 1.0),
            ('renko', 12, 26, 9, 0.001, 1.0),
            # Test time scaling
            ('macd', 12, 26, 9, 0.001, 0.5),
            ('macd', 12, 26, 9, 0.001, 2.0),
        ]
        
        # Results container
        results = []
        
        # Test each parameter set
        for params in parameter_sets:
            method, fast, slow, signal, delta, time_scale = params
            name = f"{method} ({fast},{slow},{signal}) delta={delta} scale={time_scale}"
            
            logger.info(f"\nTesting strategy: {name}")
            
            # Initialize strategy with these parameters
            strategy = AutomaticOneTwoThree(
                sar_method=method,
                macd_fast=fast,
                macd_slow=slow,
                macd_signal=signal,
                delta_coef=delta,
                time_scale=time_scale
            )
            
            # Fit the strategy to the data
            strategy.fit(data)
            
            # Generate trading signals
            signals = strategy.generate_signals(data)
            
            # Backtest the strategy
            portfolio = strategy.backtest(data, signals)
            
            # Calculate performance metrics
            metrics = strategy.calculate_performance_metrics(portfolio)
            
            # Add to results
            metrics['Name'] = name
            results.append(metrics)
            
            # Print performance summary
            print(f"\nPerformance Summary for {name}:")
            print(f"Total Return: {metrics['Total Return']:.2%}")
            print(f"Annual Return: {metrics['Annual Return']:.2%}")
            print(f"Sharpe Ratio: {metrics['Sharpe Ratio']:.2f}")
            print(f"Max Drawdown: {metrics['Max Drawdown']:.2%}")
            print(f"Win Rate: {metrics['Win Rate']:.2%}")
            print(f"Profit Factor: {metrics['Profit Factor']:.2f}")
            print(f"Number of Trades: {metrics['Number of Trades']}")
        
        # Find the best parameter set
        if results:
            results_df = pd.DataFrame(results)
            
            # Sort by Sharpe ratio
            results_df = results_df.sort_values('Sharpe Ratio', ascending=False)
            
            print("\nParameter Optimization Results (sorted by Sharpe Ratio):")
            print(results_df[['Name', 'Total Return', 'Annual Return', 'Sharpe Ratio', 'Max Drawdown', 'Profit Factor', 'Number of Trades']])
            
            if not results_df.empty:
                # Plot the best strategy
                best_idx = results_df.index[0]
                best_params = parameter_sets[best_idx]
                method, fast, slow, signal, delta, time_scale = best_params
                
                best_strategy = AutomaticOneTwoThree(
                    sar_method=method,
                    macd_fast=fast,
                    macd_slow=slow,
                    macd_signal=signal,
                    delta_coef=delta,
                    time_scale=time_scale
                )
                
                best_strategy.fit(data)
                signals = best_strategy.generate_signals(data)
                portfolio = best_strategy.backtest(data, signals)
                
                print(f"\nPlotting results for best strategy: {results_df.iloc[0]['Name']}")
                best_strategy.plot_results(data, signals, portfolio)
    else:
        logger.error(f"No data available for {ticker}")


if __name__ == "__main__":
    main()