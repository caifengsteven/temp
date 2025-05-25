import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Data Classes and Basic Definitions
# ============================================================================

@dataclass
class InstitutionalOrderSignal:
    """Signal for detected institutional order"""
    timestamp: pd.Timestamp
    symbol: str
    order_type: str  # 'BUY', 'SELL'
    estimated_size: float
    execution_horizon: int  # estimated minutes to complete
    confidence: float
    execution_style: str  # 'TWAP', 'VWAP', 'U-SHAPE', 'AGGRESSIVE'
    current_progress: float  # 0-1, estimated completion

@dataclass
class CombinedSignal:
    """Combined signal using both institutional and informed detection"""
    timestamp: pd.Timestamp
    symbol: str
    action: str  # 'FRONT_RUN', 'FADE', 'PROVIDE_LIQUIDITY', 'AVOID'
    direction: str  # 'LONG', 'SHORT', 'NEUTRAL'
    confidence: float
    expected_move: float
    holding_period: int
    size_pct: float
    reason: str
    inst_order: Optional[InstitutionalOrderSignal]
    informed_state: Dict  # N, confidence, etc.

# ============================================================================
# Detection Classes
# ============================================================================

class InstitutionalOrderDetector:
    """
    Detect institutional orders using market microstructure patterns.
    """
    
    def __init__(self, lookback_window: int = 500):
        self.lookback_window = lookback_window
        self.volume_patterns = deque(maxlen=lookback_window)
        self.price_impacts = deque(maxlen=lookback_window)
        
    def detect_institutional_order(self, 
                                  market_data: Dict,
                                  time_window: int = 30) -> Optional[InstitutionalOrderSignal]:
        """
        Detect institutional orders from market patterns.
        """
        
        features = self._extract_institutional_features(market_data, time_window)
        
        if not features:
            return None
        
        # Pattern detection scores
        u_shape_score = self._detect_u_shape_pattern(features)
        pressure_score = self._detect_persistent_pressure(features)
        splitting_score = self._detect_order_splitting(features)
        volume_anomaly = self._detect_volume_anomaly(features)
        
        # Combined detection
        inst_score = (
            0.35 * u_shape_score +
            0.25 * pressure_score +
            0.25 * splitting_score +
            0.15 * volume_anomaly
        )
        
        if inst_score > 0.65:  # Threshold for detection
            # Estimate order characteristics
            order_type = 'SELL' if features['net_pressure'] < 0 else 'BUY'
            
            # Estimate size based on volume deviation
            excess_volume = features['total_volume'] - features['expected_volume']
            estimated_size = max(0, excess_volume)
            
            # Estimate execution style
            if u_shape_score > 0.8:
                execution_style = 'U-SHAPE'
                execution_horizon = features['time_span']
            elif features['volume_acceleration'] > 2:
                execution_style = 'AGGRESSIVE'
                execution_horizon = features['time_span'] // 2
            else:
                execution_style = 'TWAP'
                execution_horizon = features['time_span']
            
            # Estimate progress
            elapsed_pct = features.get('elapsed_pct', 0.5)
            if execution_style == 'U-SHAPE':
                if elapsed_pct < 0.3:
                    progress = elapsed_pct * 1.5
                elif elapsed_pct < 0.7:
                    progress = 0.45 + (elapsed_pct - 0.3) * 0.3
                else:
                    progress = 0.6 + (elapsed_pct - 0.7) * 1.3
                progress = min(0.95, progress)
            else:
                progress = elapsed_pct
            
            return InstitutionalOrderSignal(
                timestamp=pd.Timestamp.now(),
                symbol=features.get('symbol', ''),
                order_type=order_type,
                estimated_size=estimated_size,
                execution_horizon=execution_horizon,
                confidence=inst_score,
                execution_style=execution_style,
                current_progress=progress
            )
        
        return None
    
    def _extract_institutional_features(self, 
                                      market_data: Dict,
                                      time_window: int) -> Dict:
        """Extract features indicating institutional activity"""
        
        trades = market_data.get('trades', [])
        if len(trades) < 20:
            return {}
        
        # Volume analysis
        volumes = [t['size'] for t in trades]
        buy_volume = sum(t['size'] for t in trades if t['side'] == 'BUY')
        sell_volume = sum(t['size'] for t in trades if t['side'] == 'SELL')
        
        # Create time buckets for U-shape analysis
        n_buckets = min(10, len(trades) // 5)
        if n_buckets < 3:
            return {}
        
        bucket_size = len(trades) // n_buckets
        bucket_volumes = []
        
        for i in range(n_buckets):
            start_idx = i * bucket_size
            end_idx = (i + 1) * bucket_size if i < n_buckets - 1 else len(trades)
            bucket_vol = sum(volumes[start_idx:end_idx])
            bucket_volumes.append(bucket_vol)
        
        # Price impact analysis
        prices = [t['price'] for t in trades]
        price_changes = np.diff(prices) if len(prices) > 1 else []
        
        features = {
            'symbol': market_data.get('symbol', ''),
            'time_span': time_window,
            'total_volume': sum(volumes),
            'avg_volume': np.mean(volumes),
            'expected_volume': np.mean(volumes) * len(volumes),
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'net_pressure': buy_volume - sell_volume,
            'bucket_volumes': bucket_volumes,
            'volume_cv': np.std(volumes) / (np.mean(volumes) + 1e-10),
            'price_impact': np.std(price_changes) if len(price_changes) > 0 else 0,
            'volume_acceleration': bucket_volumes[-1] / (bucket_volumes[0] + 1e-10) if bucket_volumes else 1,
            'elapsed_pct': 0.5
        }
        
        return features
    
    def _detect_u_shape_pattern(self, features: Dict) -> float:
        """Detect U-shaped volume pattern"""
        
        bucket_volumes = features.get('bucket_volumes', [])
        if len(bucket_volumes) < 5:
            return 0.0
        
        # Normalize volumes
        total_vol = sum(bucket_volumes)
        if total_vol == 0:
            return 0.0
        
        norm_volumes = [v / total_vol for v in bucket_volumes]
        
        # Check U-shape: high at ends, low in middle
        n = len(norm_volumes)
        start_weight = np.mean(norm_volumes[:n//4])
        middle_weight = np.mean(norm_volumes[n//4:3*n//4])
        end_weight = np.mean(norm_volumes[3*n//4:])
        
        # U-shape score
        if middle_weight > 0:
            u_ratio = (start_weight + end_weight) / (2 * middle_weight)
            u_score = min(1.0, (u_ratio - 1) / 2)
        else:
            u_score = 0.0
        
        return u_score
    
    def _detect_persistent_pressure(self, features: Dict) -> float:
        """Detect persistent directional pressure"""
        
        net_pressure = abs(features.get('net_pressure', 0))
        total_volume = features.get('total_volume', 1)
        
        if total_volume == 0:
            return 0.0
        
        pressure_ratio = net_pressure / total_volume
        return min(1.0, pressure_ratio)
    
    def _detect_order_splitting(self, features: Dict) -> float:
        """Detect systematic order splitting"""
        
        volume_cv = features.get('volume_cv', 1)
        
        # Low CV suggests similar-sized orders (splitting)
        if volume_cv < 0.3:
            return 0.8
        elif volume_cv < 0.5:
            return 0.5
        else:
            return 0.2
    
    def _detect_volume_anomaly(self, features: Dict) -> float:
        """Detect abnormal volume patterns"""
        
        total_volume = features.get('total_volume', 0)
        expected_volume = features.get('expected_volume', 1)
        
        if expected_volume == 0:
            return 0.0
        
        volume_ratio = total_volume / expected_volume
        
        if volume_ratio > 3:
            return 1.0
        elif volume_ratio > 2:
            return 0.7
        elif volume_ratio > 1.5:
            return 0.4
        else:
            return 0.0


class RealTimeInformedTraderDetector:
    """
    Simplified informed trader detector for testing.
    """
    
    def __init__(self, symbols: List[str], lookback_window: int = 1000):
        self.symbols = symbols
        self.lookback_window = lookback_window
        self.trade_data = {symbol: deque(maxlen=lookback_window) for symbol in symbols}
        self.order_book_data = {symbol: deque(maxlen=lookback_window) for symbol in symbols}
        
    def process_trade(self, symbol: str, trade: Dict):
        """Process incoming trade data."""
        self.trade_data[symbol].append(trade)
    
    def calculate_microstructure_features(self, symbol: str, window: int = 100) -> Dict[str, float]:
        """Calculate market microstructure features."""
        if len(self.trade_data[symbol]) < window:
            return {}
        
        recent_trades = list(self.trade_data[symbol])[-window:]
        
        features = {}
        
        # Order Flow Imbalance
        buy_volume = sum(t['size'] for t in recent_trades if t['side'] == 'BUY')
        sell_volume = sum(t['size'] for t in recent_trades if t['side'] == 'SELL')
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            features['order_flow_imbalance'] = (buy_volume - sell_volume) / total_volume
        
        # Price impact (Kyle's Lambda proxy)
        if len(recent_trades) >= 20:
            prices = [t['price'] for t in recent_trades]
            signed_volumes = [t['size'] if t['side'] == 'BUY' else -t['size'] for t in recent_trades]
            
            if np.std(signed_volumes) > 0:
                price_changes = np.diff(prices)
                kyle_lambda = abs(np.corrcoef(price_changes, signed_volumes[:-1])[0, 1])
                features['kyle_lambda'] = kyle_lambda
        
        # Trade clustering
        if len(recent_trades) >= 5:
            trade_times = [t['timestamp'] for t in recent_trades]
            time_diffs = np.diff(trade_times)
            features['trade_clustering_coefficient'] = np.std(time_diffs) / (np.mean(time_diffs) + 1e-10)
        
        # Simple spread proxy
        features['spread_volatility'] = np.random.uniform(0.0001, 0.003)
        features['avg_impact_recovery_time'] = np.random.uniform(5, 20)
        
        return features
    
    def estimate_informed_traders(self, features: Dict[str, float]) -> Tuple[int, float]:
        """Estimate number of informed traders based on features."""
        if not features:
            return 1, 0.0
        
        # Simplified scoring
        score = 0
        confidence = 0.5
        
        # Order flow patterns
        if abs(features.get('order_flow_imbalance', 0)) < 0.2:
            score += 2
            confidence += 0.1
        
        # Kyle's Lambda (lower = more competition)
        kyle_lambda = features.get('kyle_lambda', 1)
        if kyle_lambda < 0.1:
            score += 3
            confidence += 0.2
        elif kyle_lambda < 0.3:
            score += 2
            confidence += 0.1
        
        # Trade clustering
        clustering = features.get('trade_clustering_coefficient', 1)
        if clustering > 2:
            score += 2
            confidence += 0.1
        
        # Convert score to estimated N
        if score >= 6:
            estimated_N = 20
        elif score >= 4:
            estimated_N = 10
        elif score >= 2:
            estimated_N = 5
        else:
            estimated_N = 1
        
        confidence = min(confidence, 0.9)
        
        return estimated_N, confidence


class CombinedTradingStrategy:
    """
    Combines institutional order detection with informed trader detection.
    """
    
    def __init__(self, capital: float = 100000):
        self.capital = capital
        self.positions = {}
        
    def generate_combined_signal(self,
                               inst_signal: Optional[InstitutionalOrderSignal],
                               informed_state: Dict,
                               market_data: Dict) -> Optional[CombinedSignal]:
        """Generate trading signals by combining both detections."""
        
        if inst_signal is None:
            return self._informed_only_signal(informed_state, market_data)
        
        N = informed_state.get('N', 1)
        informed_confidence = informed_state.get('confidence', 0.5)
        
        # Strategy selection based on combined information
        
        # Case 1: Early institutional detection with few informed traders
        if inst_signal.current_progress < 0.3 and N < 5:
            return self._front_run_institutional(inst_signal, informed_state)
        
        # Case 2: Institutional order with many informed traders
        elif N >= 10 and inst_signal.confidence > 0.7:
            return self._join_liquidity_provision(inst_signal, informed_state)
        
        # Case 3: Late institutional with U-shape pattern
        elif inst_signal.execution_style == 'U-SHAPE' and inst_signal.current_progress > 0.7:
            return self._trade_u_shape_acceleration(inst_signal, informed_state)
        
        # Case 4: Institutional exhaustion
        elif inst_signal.current_progress > 0.8 and N < 3:
            return self._fade_institutional_exhaustion(inst_signal, informed_state)
        
        return None
    
    def _front_run_institutional(self, 
                               inst_signal: InstitutionalOrderSignal,
                               informed_state: Dict) -> CombinedSignal:
        """Front-run institutional order when informed traders haven't arrived yet."""
        
        direction = 'SHORT' if inst_signal.order_type == 'SELL' else 'LONG'
        
        return CombinedSignal(
            timestamp=pd.Timestamp.now(),
            symbol=inst_signal.symbol,
            action='FRONT_RUN',
            direction=direction,
            confidence=0.75,
            expected_move=0.003,
            holding_period=15,
            size_pct=0.12,
            reason=f"Front-running institutional {inst_signal.order_type}",
            inst_order=inst_signal,
            informed_state=informed_state
        )
    
    def _join_liquidity_provision(self,
                                 inst_signal: InstitutionalOrderSignal,
                                 informed_state: Dict) -> CombinedSignal:
        """Join informed traders in providing liquidity."""
        
        direction = 'LONG' if inst_signal.order_type == 'SELL' else 'SHORT'
        size_multiplier = min(1.5, informed_state['N'] / 10)
        
        return CombinedSignal(
            timestamp=pd.Timestamp.now(),
            symbol=inst_signal.symbol,
            action='PROVIDE_LIQUIDITY',
            direction=direction,
            confidence=0.85,
            expected_move=0.005,
            holding_period=30,
            size_pct=0.15 * size_multiplier,
            reason=f"Providing liquidity with {informed_state['N']} informed traders",
            inst_order=inst_signal,
            informed_state=informed_state
        )
    
    def _trade_u_shape_acceleration(self,
                                   inst_signal: InstitutionalOrderSignal,
                                   informed_state: Dict) -> CombinedSignal:
        """Trade the acceleration phase of U-shaped execution."""
        
        direction = 'SHORT' if inst_signal.order_type == 'SELL' else 'LONG'
        
        return CombinedSignal(
            timestamp=pd.Timestamp.now(),
            symbol=inst_signal.symbol,
            action='FRONT_RUN',
            direction=direction,
            confidence=0.70,
            expected_move=0.002,
            holding_period=10,
            size_pct=0.10,
            reason="U-shape acceleration phase detected",
            inst_order=inst_signal,
            informed_state=informed_state
        )
    
    def _fade_institutional_exhaustion(self,
                                     inst_signal: InstitutionalOrderSignal,
                                     informed_state: Dict) -> CombinedSignal:
        """Fade the institutional order when it's nearly complete."""
        
        direction = 'LONG' if inst_signal.order_type == 'SELL' else 'SHORT'
        
        return CombinedSignal(
            timestamp=pd.Timestamp.now(),
            symbol=inst_signal.symbol,
            action='FADE',
            direction=direction,
            confidence=0.78,
            expected_move=0.004,
            holding_period=25,
            size_pct=0.14,
            reason="Institutional order exhaustion - reversal expected",
            inst_order=inst_signal,
            informed_state=informed_state
        )
    
    def _informed_only_signal(self,
                            informed_state: Dict,
                            market_data: Dict) -> Optional[CombinedSignal]:
        """Generate signals based only on informed trader detection."""
        
        N = informed_state.get('N', 1)
        features = informed_state.get('features', {})
        
        if N >= 15 and abs(features.get('order_flow_imbalance', 0)) > 0.4:
            direction = 'LONG' if features['order_flow_imbalance'] > 0 else 'SHORT'
            
            return CombinedSignal(
                timestamp=pd.Timestamp.now(),
                symbol='',
                action='FOLLOW_INFORMED',
                direction=direction,
                confidence=0.82,
                expected_move=0.006,
                holding_period=40,
                size_pct=0.16,
                reason=f"Strong informed consensus (N={N})",
                inst_order=None,
                informed_state=informed_state
            )
        
        return None

# ============================================================================
# Market Simulator
# ============================================================================

class MarketSimulator:
    """
    Sophisticated market simulator with institutional orders and informed traders.
    """
    
    def __init__(self, 
                 base_price: float = 100.0,
                 base_spread: float = 0.02,
                 volatility: float = 0.001,
                 tick_size: float = 0.01):
        
        self.base_price = base_price
        self.base_spread = base_spread
        self.volatility = volatility
        self.tick_size = tick_size
        self.current_price = base_price
        self.true_value = base_price
        
        self.institutional_orders = []
        self.informed_traders = []
        self.current_time = 0
        
        self.price_history = []
        self.trade_history = []
        self.participant_history = []
        
    def add_institutional_order(self, 
                              start_time: int,
                              duration: int,
                              total_size: float,
                              direction: str,
                              execution_style: str = 'U-SHAPE'):
        """Add an institutional order to the simulation"""
        
        self.institutional_orders.append({
            'start_time': start_time,
            'end_time': start_time + duration,
            'total_size': total_size,
            'executed_size': 0,
            'direction': direction,
            'execution_style': execution_style,
            'duration': duration
        })
    
    def add_informed_trader_wave(self,
                               arrival_time: int,
                               num_traders: int,
                               information_value: float,
                               duration: int):
        """Add a wave of informed traders"""
        
        for i in range(num_traders):
            actual_arrival = arrival_time + np.random.randint(-5, 5)
            
            self.informed_traders.append({
                'arrival_time': actual_arrival,
                'departure_time': actual_arrival + duration + np.random.randint(-10, 10),
                'information_value': information_value + np.random.normal(0, 0.001),
                'aggressiveness': np.random.uniform(0.5, 1.5),
                'id': len(self.informed_traders)
            })
    
    def _calculate_institutional_rate(self, order: Dict, current_time: int) -> float:
        """Calculate institutional trading rate based on execution style"""
        
        if current_time < order['start_time'] or current_time >= order['end_time']:
            return 0
        
        progress = (current_time - order['start_time']) / order['duration']
        remaining_size = order['total_size'] - order['executed_size']
        
        if remaining_size <= 0:
            return 0
        
        if order['execution_style'] == 'U-SHAPE':
            # U-shaped execution
            t = progress
            if t < 0.3:
                rate = 2.0
            elif t < 0.7:
                rate = 0.5
            else:
                rate = 2.5
            
            base_rate = remaining_size / (order['end_time'] - current_time + 1)
            return base_rate * rate
            
        else:  # TWAP
            return order['total_size'] / order['duration']
    
    def _calculate_informed_trading(self, current_time: int) -> Tuple[float, int]:
        """Calculate aggregate informed trading"""
        
        active_informed = [
            trader for trader in self.informed_traders
            if trader['arrival_time'] <= current_time < trader['departure_time']
        ]
        
        if not active_informed:
            return 0, 0
        
        mispricing = self.true_value - self.current_price
        
        total_trading = 0
        for trader in active_informed:
            trader_rate = trader['aggressiveness'] * mispricing * 10
            total_trading += trader_rate
        
        return total_trading, len(active_informed)
    
    def simulate_tick(self) -> Dict:
        """Simulate one time tick"""
        
        # Calculate institutional trading
        inst_buy_rate = 0
        inst_sell_rate = 0
        
        for order in self.institutional_orders:
            rate = self._calculate_institutional_rate(order, self.current_time)
            if rate > 0:
                if order['direction'] == 'BUY':
                    inst_buy_rate += rate
                else:
                    inst_sell_rate += rate
                
                order['executed_size'] += rate
        
        # Calculate informed trading
        informed_rate, num_informed = self._calculate_informed_trading(self.current_time)
        informed_buy_rate = max(0, informed_rate)
        informed_sell_rate = max(0, -informed_rate)
        
        # Random noise traders
        noise_buy = np.random.exponential(5)
        noise_sell = np.random.exponential(5)
        
        # Total order flow
        total_buy = inst_buy_rate + informed_buy_rate + noise_buy
        total_sell = inst_sell_rate + informed_sell_rate + noise_sell
        
        # Price impact
        net_flow = total_buy - total_sell
        permanent_impact = 0.001 * net_flow
        temporary_impact = 0.0001 * abs(net_flow)
        
        # Update price
        self.current_price += permanent_impact + np.random.normal(0, self.volatility)
        
        # Update true value
        self.true_value += np.random.normal(0, self.volatility * 0.5)
        
        # Calculate spread
        if num_informed > 0:
            spread = self.base_spread / np.sqrt(1 + num_informed/5)
        else:
            spread = self.base_spread
        
        # Generate trades
        trades = []
        
        if inst_buy_rate > 0:
            trades.append({
                'timestamp': self.current_time,
                'price': self.current_price + temporary_impact,
                'size': inst_buy_rate,
                'side': 'BUY',
                'type': 'INSTITUTIONAL'
            })
        
        if inst_sell_rate > 0:
            trades.append({
                'timestamp': self.current_time,
                'price': self.current_price - temporary_impact,
                'size': inst_sell_rate,
                'side': 'SELL',
                'type': 'INSTITUTIONAL'
            })
        
        if informed_buy_rate > 1:
            trades.append({
                'timestamp': self.current_time,
                'price': self.current_price,
                'size': informed_buy_rate,
                'side': 'BUY',
                'type': 'INFORMED'
            })
        
        if informed_sell_rate > 1:
            trades.append({
                'timestamp': self.current_time,
                'price': self.current_price,
                'size': informed_sell_rate,
                'side': 'SELL',
                'type': 'INFORMED'
            })
        
        trades.append({
            'timestamp': self.current_time,
            'price': self.current_price + spread/2,
            'size': noise_buy,
            'side': 'BUY',
            'type': 'NOISE'
        })
        
        trades.append({
            'timestamp': self.current_time,
            'price': self.current_price - spread/2,
            'size': noise_sell,
            'side': 'SELL',
            'type': 'NOISE'
        })
        
        # Store data
        self.trade_history.extend(trades)
        self.price_history.append({
            'timestamp': self.current_time,
            'price': self.current_price,
            'true_value': self.true_value,
            'spread': spread
        })
        
        self.participant_history.append({
            'timestamp': self.current_time,
            'num_informed': num_informed,
            'inst_buying': inst_buy_rate,
            'inst_selling': inst_sell_rate,
            'informed_flow': informed_rate
        })
        
        # Create market snapshot
        market_data = {
            'timestamp': self.current_time,
            'trades': trades[-10:],
            'order_book': {
                'bid': [{'price': self.current_price - spread/2, 'size': 100}],
                'ask': [{'price': self.current_price + spread/2, 'size': 100}],
                'mid': self.current_price,
                'spread': spread
            },
            'price': self.current_price,
            'true_value': self.true_value
        }
        
        self.current_time += 1
        
        return market_data
    
    def create_scenarios(self):
        """Create realistic trading scenarios"""
        
        # Morning institutional sell order
        self.add_institutional_order(
            start_time=30,
            duration=300,
            total_size=50000,
            direction='SELL',
            execution_style='U-SHAPE'
        )
        
        # Informed traders arrive in waves
        self.add_informed_trader_wave(
            arrival_time=60,
            num_traders=2,
            information_value=100.5,
            duration=200
        )
        
        self.add_informed_trader_wave(
            arrival_time=120,
            num_traders=8,
            information_value=100.3,
            duration=150
        )
        
        self.add_informed_trader_wave(
            arrival_time=180,
            num_traders=10,
            information_value=100.2,
            duration=100
        )
        
        # Afternoon institutional buy
        self.add_institutional_order(
            start_time=400,
            duration=200,
            total_size=30000,
            direction='BUY',
            execution_style='TWAP'
        )
        
        # Information event
        self.add_informed_trader_wave(
            arrival_time=450,
            num_traders=15,
            information_value=101.0,
            duration=100
        )

# ============================================================================
# Backtest Framework
# ============================================================================

class StrategyBacktest:
    """
    Backtest framework for testing trading strategies
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.pnl_history = []
        self.detection_history = []
        
        # Initialize detectors and strategy
        self.inst_detector = InstitutionalOrderDetector()
        self.informed_detector = RealTimeInformedTraderDetector(['TEST'])
        self.combined_strategy = CombinedTradingStrategy(capital=initial_capital)
        
    def process_market_data(self, market_data: Dict) -> Optional[CombinedSignal]:
        """Process market data and generate signals"""
        
        # Update detectors with trade data
        for trade in market_data['trades']:
            self.informed_detector.process_trade('TEST', trade)
        
        # Detect institutional order
        inst_signal = self.inst_detector.detect_institutional_order(
            {'trades': self.trade_buffer[-50:] if hasattr(self, 'trade_buffer') else market_data['trades']},
            time_window=30
        )
        
        # Get informed trader state
        features = self.informed_detector.calculate_microstructure_features('TEST')
        if features:
            N_est, confidence = self.informed_detector.estimate_informed_traders(features)
            informed_state = {
                'N': N_est,
                'confidence': confidence,
                'features': features
            }
        else:
            informed_state = {'N': 1, 'confidence': 0.5, 'features': {}}
        
        # Store detection results
        self.detection_history.append({
            'timestamp': market_data['timestamp'],
            'inst_detected': inst_signal is not None,
            'inst_progress': inst_signal.current_progress if inst_signal else 0,
            'N_estimated': informed_state['N'],
            'confidence': informed_state['confidence']
        })
        
        # Generate combined signal
        signal = self.combined_strategy.generate_combined_signal(
            inst_signal, informed_state, market_data
        )
        
        return signal
    
    def execute_signal(self, signal: CombinedSignal, market_data: Dict):
        """Execute trading signal"""
        
        if signal.direction == 'NEUTRAL':
            return
        
        current_price = market_data['price']
        position_size = int(self.capital * signal.size_pct)
        
        # Execute trade
        trade = {
            'timestamp': market_data['timestamp'],
            'action': signal.action,
            'direction': signal.direction,
            'size': position_size,
            'price': current_price,
            'signal': signal,
            'entry_time': market_data['timestamp']
        }
        
        self.trades.append(trade)
        
        # Update position
        position_id = len(self.trades)
        self.positions[position_id] = {
            'size': position_size if signal.direction == 'LONG' else -position_size,
            'entry_price': current_price,
            'entry_time': market_data['timestamp'],
            'holding_period': signal.holding_period,
            'expected_move': signal.expected_move,
            'signal_type': signal.action
        }
    
    def update_positions(self, market_data: Dict):
        """Update positions and check exits"""
        
        current_time = market_data['timestamp']
        current_price = market_data['price']
        
        for pos_id, position in list(self.positions.items()):
            # Calculate P&L
            if position['size'] > 0:  # Long
                pnl = (current_price - position['entry_price']) / position['entry_price']
            else:  # Short
                pnl = (position['entry_price'] - current_price) / position['entry_price']
            
            # Check exit conditions
            should_exit = False
            exit_reason = ""
            
            # Time-based exit
            if current_time - position['entry_time'] >= position['holding_period']:
                should_exit = True
                exit_reason = "Holding period complete"
            
            # Target reached
            elif pnl >= position['expected_move']:
                should_exit = True
                exit_reason = "Target reached"
            
            # Stop loss
            elif pnl <= -position['expected_move'] * 0.5:
                should_exit = True
                exit_reason = "Stop loss"
            
            if should_exit:
                # Close position
                exit_trade = {
                    'timestamp': current_time,
                    'position_id': pos_id,
                    'exit_price': current_price,
                    'pnl': pnl,
                    'pnl_amount': abs(position['size']) * pnl,
                    'exit_reason': exit_reason,
                    'signal_type': position['signal_type']
                }
                
                self.trades.append(exit_trade)
                self.capital += exit_trade['pnl_amount']
                
                del self.positions[pos_id]
    
    def calculate_performance_metrics(self) -> Dict:
        """Calculate strategy performance metrics"""
        
        # Extract closed trades
        exits = [t for t in self.trades if 'exit_price' in t]
        
        if not exits:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0,
                'signal_performance': {}
            }
        
        # Calculate metrics
        pnls = [e['pnl'] for e in exits]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        # Group by signal type
        signal_performance = {}
        for signal_type in ['FRONT_RUN', 'PROVIDE_LIQUIDITY', 'FADE', 'FOLLOW_INFORMED']:
            type_exits = [e for e in exits if e.get('signal_type') == signal_type]
            if type_exits:
                type_pnls = [e['pnl'] for e in type_exits]
                signal_performance[signal_type] = {
                    'count': len(type_exits),
                    'win_rate': sum(1 for p in type_pnls if p > 0) / len(type_pnls),
                    'avg_return': np.mean(type_pnls)
                }
        
        # Overall metrics
        metrics = {
            'total_trades': len(exits),
            'win_rate': len(wins) / len(pnls) if pnls else 0,
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'profit_factor': abs(sum(wins) / sum(losses)) if losses else 0,
            'sharpe_ratio': np.mean(pnls) / np.std(pnls) * np.sqrt(252) if len(pnls) > 1 and np.std(pnls) > 0 else 0,
            'total_return': (self.capital - self.initial_capital) / self.initial_capital,
            'signal_performance': signal_performance
        }
        
        # Calculate drawdown
        cumulative_returns = np.cumsum([e['pnl_amount'] for e in exits])
        if len(cumulative_returns) > 0:
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = (cumulative_returns - running_max) / self.initial_capital
            metrics['max_drawdown'] = abs(np.min(drawdown)) if len(drawdown) > 0 else 0
        else:
            metrics['max_drawdown'] = 0
        
        return metrics

# ============================================================================
# Main Backtest Function
# ============================================================================

def run_comprehensive_backtest():
    """Run comprehensive strategy backtest"""
    
    print("="*70)
    print("Comprehensive Strategy Backtest")
    print("="*70)
    
    # Create market simulator
    simulator = MarketSimulator()
    simulator.create_scenarios()
    
    # Initialize backtest
    backtest = StrategyBacktest(initial_capital=100000)
    
    # Initialize trade buffer
    backtest.trade_buffer = []
    
    # Run simulation
    print("\nRunning simulation...")
    for i in range(600):  # 600 time steps
        # Get market data
        market_data = simulator.simulate_tick()
        
        # Update trade buffer
        backtest.trade_buffer.extend(market_data['trades'])
        backtest.trade_buffer = backtest.trade_buffer[-100:]  # Keep last 100
        
        # Process market data and generate signal
        signal = backtest.process_market_data(market_data)
        
        # Execute signal if any
        if signal and signal.action != 'AVOID':
            backtest.execute_signal(signal, market_data)
        
        # Update existing positions
        backtest.update_positions(market_data)
        
        # Record P&L
        current_pnl = (backtest.capital - backtest.initial_capital) / backtest.initial_capital
        backtest.pnl_history.append({
            'timestamp': i,
            'pnl': current_pnl,
            'capital': backtest.capital
        })
    
    # Calculate performance metrics
    metrics = backtest.calculate_performance_metrics()
    
    # Create visualizations
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    
    # 1. Price and true value
    ax = axes[0, 0]
    price_df = pd.DataFrame(simulator.price_history)
    ax.plot(price_df['timestamp'], price_df['price'], 'b-', label='Market Price', alpha=0.7)
    ax.plot(price_df['timestamp'], price_df['true_value'], 'g--', label='True Value', alpha=0.7)
    
    # Mark trades
    for trade in backtest.trades:
        if 'direction' in trade:
            color = 'green' if trade['direction'] == 'LONG' else 'red'
            marker = '^' if trade['direction'] == 'LONG' else 'v'
            ax.scatter(trade['timestamp'], trade['price'], 
                      color=color, marker=marker, s=100, alpha=0.7)
    
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title('Price Evolution and Trades')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Participant activity
    ax = axes[0, 1]
    participant_df = pd.DataFrame(simulator.participant_history)
    ax.plot(participant_df['timestamp'], participant_df['num_informed'], 'b-', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Informed Traders')
    ax.set_title('Informed Trader Activity')
    
    # Mark institutional activity
    ax2 = ax.twinx()
    ax2.plot(participant_df['timestamp'], 
             participant_df['inst_buying'] - participant_df['inst_selling'], 
             'r--', alpha=0.7)
    ax2.set_ylabel('Institutional Flow', color='red')
    ax.grid(True, alpha=0.3)
    
    # 3. Detection accuracy
    ax = axes[0, 2]
    detection_df = pd.DataFrame(backtest.detection_history)
    
    # Compare estimated vs actual informed traders
    actual_informed = participant_df['num_informed']
    estimated_informed = detection_df['N_estimated']
    
    ax.plot(detection_df['timestamp'], actual_informed[:len(estimated_informed)], 
            'b-', label='Actual', linewidth=2)
    ax.plot(detection_df['timestamp'], estimated_informed, 
            'r--', label='Estimated', linewidth=2)
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of Informed Traders')
    ax.set_title('Detection Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. P&L evolution
    ax = axes[1, 0]
    pnl_df = pd.DataFrame(backtest.pnl_history)
    ax.plot(pnl_df['timestamp'], pnl_df['pnl'] * 100, 'g-', linewidth=2)
    ax.fill_between(pnl_df['timestamp'], pnl_df['pnl'] * 100, alpha=0.3, color='green')
    ax.set_xlabel('Time')
    ax.set_ylabel('P&L (%)')
    ax.set_title('Strategy Performance')
    ax.grid(True, alpha=0.3)
    
    # 5. Trade distribution
    ax = axes[1, 1]
    if backtest.trades:
        trade_types = [t.get('action', 'EXIT') for t in backtest.trades if 'direction' in t]
        if trade_types:
            trade_counts = pd.Series(trade_types).value_counts()
            
            colors = {
                'FRONT_RUN': 'orange',
                'PROVIDE_LIQUIDITY': 'green',
                'FADE': 'purple',
                'FOLLOW_INFORMED': 'blue'
            }
            
            bar_colors = [colors.get(x, 'gray') for x in trade_counts.index]
            ax.bar(trade_counts.index, trade_counts.values, color=bar_colors, alpha=0.7)
            ax.set_xlabel('Signal Type')
            ax.set_ylabel('Count')
            ax.set_title('Trade Distribution')
            ax.tick_params(axis='x', rotation=45)
    
    # 6. Win rate by signal type
    ax = axes[1, 2]
    if metrics['signal_performance']:
        signal_types = list(metrics['signal_performance'].keys())
        win_rates = [metrics['signal_performance'][s]['win_rate'] * 100 
                    for s in signal_types]
        counts = [metrics['signal_performance'][s]['count'] 
                 for s in signal_types]
        
        bars = ax.bar(range(len(signal_types)), win_rates, alpha=0.7)
        ax.set_xticks(range(len(signal_types)))
        ax.set_xticklabels(signal_types, rotation=45)
        ax.set_ylabel('Win Rate (%)')
        ax.set_title('Win Rate by Strategy')
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                   f'n={count}', ha='center', va='bottom')
        
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
    
    # 7. Return distribution
    ax = axes[2, 0]
    if backtest.trades:
        returns = [t['pnl'] * 100 for t in backtest.trades if 'pnl' in t]
        if returns:
            ax.hist(returns, bins=30, alpha=0.7, color='blue', edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)
            ax.set_xlabel('Return (%)')
            ax.set_ylabel('Frequency')
            ax.set_title('Return Distribution')
            ax.grid(True, alpha=0.3)
    
    # 8. Signal timing analysis
    ax = axes[2, 1]
    if backtest.trades:
        entry_times = [t['timestamp'] for t in backtest.trades if 'direction' in t]
        entry_types = [t['action'] for t in backtest.trades if 'direction' in t]
        
        unique_types = list(set(entry_types))
        for i, signal_type in enumerate(unique_types):
            times = [t for t, typ in zip(entry_times, entry_types) if typ == signal_type]
            ax.scatter(times, [i] * len(times), label=signal_type, s=100, alpha=0.7)
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Signal Type')
        ax.set_yticks(range(len(unique_types)))
        ax.set_yticklabels(unique_types)
        ax.set_title('Signal Timing')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    # 9. Performance summary
    ax = axes[2, 2]
    summary_text = f"""
    Performance Summary
    {'='*30}
    Total Return: {metrics['total_return']*100:.2f}%
    Total Trades: {metrics['total_trades']}
    Win Rate: {metrics['win_rate']*100:.1f}%
    
    Avg Win: {metrics['avg_win']*100:.2f}%
    Avg Loss: {metrics['avg_loss']*100:.2f}%
    Profit Factor: {metrics.get('profit_factor', 0):.2f}
    
    Sharpe Ratio: {metrics['sharpe_ratio']:.2f}
    Max Drawdown: {metrics['max_drawdown']*100:.1f}%
    """
    
    ax.text(0.1, 0.5, summary_text, transform=ax.transAxes, 
           fontsize=11, verticalalignment='center', fontfamily='monospace')
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed results
    print("\n" + "="*60)
    print("Backtest Results")
    print("="*60)
    
    print(f"\nOverall Performance:")
    print(f"Total Return: {metrics['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown']*100:.1f}%")
    print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
    
    print(f"\nStrategy Breakdown:")
    for strategy, perf in metrics['signal_performance'].items():
        print(f"\n{strategy}:")
        print(f"  Trades: {perf['count']}")
        print(f"  Win Rate: {perf['win_rate']*100:.1f}%")
        print(f"  Avg Return: {perf['avg_return']*100:.2f}%")
    
    return backtest, simulator, metrics

# ============================================================================
# Run the backtest
# ============================================================================

if __name__ == "__main__":
    # Run comprehensive backtest
    print("Testing Combined Strategy with Simulated Market Data")
    print("="*70)
    
    # Run the backtest
    backtest, simulator, metrics = run_comprehensive_backtest()
    
    print("\nBacktest completed successfully!")