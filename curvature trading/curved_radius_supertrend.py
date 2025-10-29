"""
Curved Radius Supertrend Indicator

This indicator implements a curvature-based supertrend that models trend acceleration
using radius dynamics instead of simple linear ATR bands.

Theoretical Foundation:
- Standard Supertrend uses linear ATR envelopes
- This version adds radius-based acceleration to model parabolic trend evolution
- Curvature increases over time, creating dynamic arcs that anticipate price movement

Key Components:
1. Baseline Supertrend Core: ATR-derived upper/lower bands
2. Curvature Acceleration Engine: Parabolic displacement using radiusStrength
3. Adaptive Smoothing Layer: SMA smoothing for visual coherence
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional


class CurvedRadiusSupertrend:
    """
    Curved Radius Supertrend Indicator
    
    Parameters:
    -----------
    atr_period : int
        Period for ATR calculation (default: 10)
    atr_multiplier : float
        Multiplier for ATR bands (default: 3.0)
    radius_strength : float
        Controls curvature acceleration aggressiveness (default: 0.5)
        - Lower values: Tighter, more reactive curves (scalping)
        - Higher values: Wider arcs (swing/position trading)
    smoothness : int
        SMA period for smoothing the curved bands (default: 3)
    """
    
    def __init__(
        self,
        atr_period: int = 10,
        atr_multiplier: float = 3.0,
        radius_strength: float = 0.5,
        smoothness: int = 3
    ):
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.radius_strength = radius_strength
        self.smoothness = smoothness
    
    def calculate_atr(self, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
        """
        Calculate Average True Range

        Parameters:
        -----------
        high, low, close : np.ndarray
            Price arrays

        Returns:
        --------
        np.ndarray : ATR values
        """
        # True Range calculation
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))

        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value uses high-low

        # ATR using simple moving average with min_periods=1 to avoid NaN
        atr = pd.Series(tr).rolling(window=self.atr_period, min_periods=1).mean().values

        return atr
    
    def calculate_basic_supertrend(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray,
        atr: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate basic Supertrend bands and direction
        
        Returns:
        --------
        upper_band, lower_band, direction : Tuple[np.ndarray, np.ndarray, np.ndarray]
            - upper_band: Upper volatility band
            - lower_band: Lower volatility band
            - direction: 1 for uptrend, -1 for downtrend
        """
        n = len(close)
        hl_avg = (high + low) / 2
        
        # Initial bands
        basic_upper = hl_avg + (self.atr_multiplier * atr)
        basic_lower = hl_avg - (self.atr_multiplier * atr)
        
        # Final bands with persistence logic
        upper_band = np.zeros(n)
        lower_band = np.zeros(n)
        direction = np.zeros(n)
        
        # Initialize first values
        upper_band[0] = basic_upper[0]
        lower_band[0] = basic_lower[0]
        direction[0] = 1
        
        for i in range(1, n):
            # Upper band: use previous if current is higher and price was above
            if basic_upper[i] < upper_band[i-1] or close[i-1] > upper_band[i-1]:
                upper_band[i] = basic_upper[i]
            else:
                upper_band[i] = upper_band[i-1]
            
            # Lower band: use previous if current is lower and price was below
            if basic_lower[i] > lower_band[i-1] or close[i-1] < lower_band[i-1]:
                lower_band[i] = basic_lower[i]
            else:
                lower_band[i] = lower_band[i-1]
            
            # Determine direction
            if close[i] > upper_band[i-1]:
                direction[i] = 1  # Uptrend
            elif close[i] < lower_band[i-1]:
                direction[i] = -1  # Downtrend
            else:
                direction[i] = direction[i-1]  # Maintain previous direction
        
        return upper_band, lower_band, direction
    
    def apply_curvature_acceleration(
        self,
        close: np.ndarray,
        upper_band: np.ndarray,
        lower_band: np.ndarray,
        direction: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply smooth parabolic curvature to create support/resistance curves

        Key principles:
        1. Curves are SMOOTH (no jagged movements)
        2. Uptrend curve stays BELOW price (support)
        3. Downtrend curve stays ABOVE price (resistance)
        4. Curves are parabolic arcs, not following price action

        Returns:
        --------
        curved_upper, curved_lower : Tuple[np.ndarray, np.ndarray]
            Smooth parabolic curves
        """
        if self.radius_strength == 0:
            return upper_band, lower_band

        n = len(close)
        curved_upper = np.zeros(n)
        curved_lower = np.zeros(n)

        # Find trend change points (anchor points for parabolas)
        trend_changes = [0]  # Start is always an anchor
        for i in range(1, n):
            if direction[i] != direction[i-1]:
                trend_changes.append(i)
        trend_changes.append(n - 1)  # End is always an anchor

        # For each trend segment, create a smooth parabolic curve
        for seg_idx in range(len(trend_changes) - 1):
            start_idx = trend_changes[seg_idx]
            end_idx = trend_changes[seg_idx + 1]
            segment_len = end_idx - start_idx

            if segment_len <= 0:
                continue

            # Determine if this is uptrend or downtrend
            is_uptrend = direction[start_idx] == 1

            # Calculate average ATR for this segment
            segment_atr = np.mean(np.abs(upper_band[start_idx:end_idx+1] -
                                         lower_band[start_idx:end_idx+1])) / (2 * self.atr_multiplier)

            # Get the starting anchor point for the curve
            # This should be close to price but not touching
            if is_uptrend:
                # For uptrend, start slightly below the low of first candle
                anchor_level = lower_band[start_idx]
            else:
                # For downtrend, start slightly above the high of first candle
                anchor_level = upper_band[start_idx]

            # Calculate the slope based on overall trend direction
            # Use a gentle slope that doesn't follow every price wiggle
            price_start = close[start_idx]
            price_end = close[min(end_idx, n-1)]

            # Determine natural slope direction
            if is_uptrend:
                # Uptrend should slope upward
                # Use actual price movement to determine slope
                natural_slope = max(0, (price_end - price_start) * 0.6) / max(segment_len, 1)
            else:
                # Downtrend should slope downward
                # Use actual price movement to determine slope
                natural_slope = min(0, (price_end - price_start) * 0.6) / max(segment_len, 1)

            # Create smooth parabolic curve for this segment
            for i in range(start_idx, end_idx + 1):
                t = i - start_idx  # Time from anchor

                # Normalized time (0 to 1)
                t_norm = t / max(segment_len, 1)

                # Smooth parabolic curve formula:
                # curve(t) = anchor + linear_slope * t + parabolic_term * t^2

                # Linear component (slope)
                linear_component = natural_slope * t

                # Parabolic component (creates the pronounced curve)
                # Increase the multiplier for more visible curvature
                parabolic_component = self.radius_strength * segment_atr * (t_norm ** 2) * 3.0

                if is_uptrend:
                    # Uptrend: curve below price, sloping upward with pronounced arc
                    # Parabolic term makes it curve upward (accelerating rise)
                    curved_lower[i] = anchor_level + linear_component + parabolic_component
                    curved_upper[i] = upper_band[i]
                else:
                    # Downtrend: curve above price, sloping downward with pronounced arc
                    # Parabolic term makes it curve downward (accelerating fall)
                    curved_upper[i] = anchor_level + linear_component - parabolic_component
                    curved_lower[i] = lower_band[i]

        return curved_upper, curved_lower
    
    def apply_smoothing(self, band: np.ndarray) -> np.ndarray:
        """
        Apply exponential smoothing for ultra-smooth curves

        Parameters:
        -----------
        band : np.ndarray
            Band values to smooth

        Returns:
        --------
        np.ndarray : Smoothed band values
        """
        if self.smoothness <= 1:
            return band

        # Use exponential moving average for smoother curves
        smoothed = pd.Series(band).ewm(span=self.smoothness, adjust=False).mean().values
        return smoothed
    
    def calculate(
        self,
        high: np.ndarray,
        low: np.ndarray,
        close: np.ndarray
    ) -> pd.DataFrame:
        """
        Calculate the complete Curved Radius Supertrend
        
        Parameters:
        -----------
        high, low, close : np.ndarray
            Price arrays
            
        Returns:
        --------
        pd.DataFrame with columns:
            - curved_upper: Upper curved band
            - curved_lower: Lower curved band
            - direction: Trend direction (1=up, -1=down)
            - trend_line: The active trend line (upper for downtrend, lower for uptrend)
        """
        # Step 1: Calculate ATR
        atr = self.calculate_atr(high, low, close)
        
        # Step 2: Calculate basic Supertrend
        upper_band, lower_band, direction = self.calculate_basic_supertrend(
            high, low, close, atr
        )
        
        # Step 3: Apply curvature acceleration
        curved_upper, curved_lower = self.apply_curvature_acceleration(
            close, upper_band, lower_band, direction
        )
        
        # Step 4: Apply smoothing
        curved_upper = self.apply_smoothing(curved_upper)
        curved_lower = self.apply_smoothing(curved_lower)
        
        # Step 5: Determine active trend line
        trend_line = np.where(direction == 1, curved_lower, curved_upper)
        
        # Create result DataFrame
        result = pd.DataFrame({
            'curved_upper': curved_upper,
            'curved_lower': curved_lower,
            'direction': direction,
            'trend_line': trend_line
        })
        
        return result


def example_usage():
    """
    Example usage of the Curved Radius Supertrend indicator
    """
    # Generate sample data (replace with real market data)
    np.random.seed(42)
    n = 200
    
    # Simulate price data with trend
    trend = np.linspace(100, 150, n) + np.sin(np.linspace(0, 4*np.pi, n)) * 10
    noise = np.random.randn(n) * 2
    close = trend + noise
    high = close + np.abs(np.random.randn(n)) * 1.5
    low = close - np.abs(np.random.randn(n)) * 1.5
    
    # Create indicator instance
    indicator = CurvedRadiusSupertrend(
        atr_period=10,
        atr_multiplier=3.0,
        radius_strength=0.5,
        smoothness=3
    )
    
    # Calculate indicator
    result = indicator.calculate(high, low, close)
    
    print("Curved Radius Supertrend Results:")
    print(result.tail(10))
    print(f"\nCurrent Trend: {'UPTREND' if result['direction'].iloc[-1] == 1 else 'DOWNTREND'}")
    
    return close, high, low, result


if __name__ == "__main__":
    example_usage()

