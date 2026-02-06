"""
State Encoder: Converts market data into feature vectors for RL agent.
This is what the AI "sees" - regime, momentum, and risk.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class StateEncoder:
    """
    Encodes market state into feature vector for RL agent.
    
    State vector includes:
    - Price returns (multiple timeframes)
    - Volatility metrics (ATR, std dev)
    - Trend indicators (EMA crossovers)
    - Momentum (RSI)
    - VWAP distance
    - Time features
    - Position state
    - Risk metrics
    """
    
    def __init__(self, 
                 lookback_periods: List[int] = [1, 5, 15, 60],
                 ema_fast: int = 12,
                 ema_slow: int = 26,
                 rsi_period: int = 14,
                 atr_period: int = 14):
        """
        Args:
            lookback_periods: Periods for return calculations (in bars)
            ema_fast: Fast EMA period
            ema_slow: Slow EMA period
            rsi_period: RSI calculation period
            atr_period: ATR calculation period
        """
        self.lookback_periods = lookback_periods
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        
        # Cache for indicators
        self.price_history = []
        self.volume_history = []
        self.vwap_history = []
        
    def calculate_returns(self, prices: np.ndarray, periods: List[int]) -> np.ndarray:
        """Calculate returns over multiple periods."""
        returns = []
        for period in periods:
            if len(prices) > period:
                ret = (prices[-1] - prices[-period-1]) / prices[-period-1]
                returns.append(ret)
            else:
                returns.append(0.0)
        return np.array(returns)
    
    def calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return prices[-1] if len(prices) > 0 else 0.0
        
        alpha = 2.0 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
    
    def calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_atr(self, highs: np.ndarray, lows: np.ndarray, 
                     closes: np.ndarray, period: int) -> float:
        """Calculate Average True Range."""
        if len(highs) < period + 1:
            return 0.0
        
        tr_list = []
        for i in range(1, len(highs[-period-1:])):
            tr1 = highs[-period-1+i] - lows[-period-1+i]
            tr2 = abs(highs[-period-1+i] - closes[-period-2+i])
            tr3 = abs(lows[-period-1+i] - closes[-period-2+i])
            tr = max(tr1, tr2, tr3)
            tr_list.append(tr)
        
        return np.mean(tr_list) if tr_list else 0.0
    
    def calculate_vwap_distance(self, current_price: float, vwap: float) -> float:
        """Calculate normalized distance from VWAP."""
        if vwap == 0:
            return 0.0
        return (current_price - vwap) / vwap
    
    def calculate_volatility(self, prices: np.ndarray, period: int = 20) -> float:
        """Calculate standard deviation of returns."""
        if len(prices) < period + 1:
            return 0.0
        
        returns = np.diff(prices[-period-1:]) / prices[-period-1:-1]
        return np.std(returns) if len(returns) > 0 else 0.0
    
    def encode_time_features(self, timestamp: Optional[pd.Timestamp] = None) -> np.ndarray:
        """Encode time of day features (hour, day of week)."""
        if timestamp is None:
            return np.array([0.0, 0.0])
        
        hour = timestamp.hour / 24.0  # Normalize to [0, 1]
        day_of_week = timestamp.dayofweek / 7.0  # Normalize to [0, 1]
        return np.array([hour, day_of_week])
    
    def encode_state(self,
                    current_price: float,
                    high: float,
                    low: float,
                    volume: float,
                    vwap: Optional[float] = None,
                    position_size: float = 0.0,
                    unrealized_pnl: float = 0.0,
                    account_equity: float = 10000.0,
                    peak_equity: float = 10000.0,
                    timestamp: Optional[pd.Timestamp] = None) -> np.ndarray:
        """
        Encode complete market state into feature vector.
        
        Returns:
            state_vector: np.ndarray of shape (STATE_SIZE,)
        """
        # Update history
        self.price_history.append(current_price)
        self.volume_history.append(volume)
        if vwap is not None:
            self.vwap_history.append(vwap)
        
        # Keep history manageable (need max lookback)
        max_history = max(self.lookback_periods) + self.atr_period + 10
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.volume_history = self.volume_history[-max_history:]
            if len(self.vwap_history) > max_history:
                self.vwap_history = self.vwap_history[-max_history:]
        
        prices = np.array(self.price_history)
        volumes = np.array(self.volume_history)
        
        # 1. Price returns (multiple timeframes)
        returns = self.calculate_returns(prices, self.lookback_periods)
        
        # 2. Volatility metrics
        # For ATR, we need high/low arrays - use current values if history insufficient
        if len(prices) >= self.atr_period + 1:
            price_slice = prices[-self.atr_period-1:]
            highs_array = np.array([high] * len(price_slice))
            lows_array = np.array([low] * len(price_slice))
            atr = self.calculate_atr(highs_array, lows_array, price_slice, self.atr_period)
        else:
            atr = 0.0
        volatility = self.calculate_volatility(prices)
        normalized_atr = atr / current_price if current_price > 0 else 0.0
        
        # 3. Trend indicators
        ema_fast_val = self.calculate_ema(prices, self.ema_fast)
        ema_slow_val = self.calculate_ema(prices, self.ema_slow)
        ema_crossover = (ema_fast_val - ema_slow_val) / current_price if current_price > 0 else 0.0
        
        # 4. Momentum
        rsi = self.calculate_rsi(prices, self.rsi_period)
        normalized_rsi = (rsi - 50) / 50.0  # Normalize to [-1, 1]
        
        # 5. VWAP distance
        vwap_val = vwap if vwap is not None else current_price
        vwap_dist = self.calculate_vwap_distance(current_price, vwap_val)
        
        # 6. Volume features
        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else volume
        volume_ratio = volume / avg_volume if avg_volume > 0 else 1.0
        
        # 7. Time features
        time_features = self.encode_time_features(timestamp)
        
        # 8. Position state
        normalized_position = position_size  # Already normalized
        normalized_pnl = unrealized_pnl / account_equity if account_equity > 0 else 0.0
        
        # 9. Risk metrics
        drawdown = (peak_equity - account_equity) / peak_equity if peak_equity > 0 else 0.0
        equity_ratio = account_equity / peak_equity if peak_equity > 0 else 1.0
        
        # Combine all features
        state_vector = np.concatenate([
            returns,                    # len(lookback_periods) features
            [normalized_atr, volatility],  # 2 features
            [ema_crossover],           # 1 feature
            [normalized_rsi],          # 1 feature
            [vwap_dist],               # 1 feature
            [volume_ratio],            # 1 feature
            time_features,             # 2 features
            [normalized_position, normalized_pnl],  # 2 features
            [drawdown, equity_ratio]   # 2 features
        ])
        
        return state_vector.astype(np.float32)
    
    def get_state_size(self) -> int:
        """Calculate total state vector size."""
        return (len(self.lookback_periods) +  # returns
                2 +  # volatility (atr, std)
                1 +  # ema crossover
                1 +  # rsi
                1 +  # vwap distance
                1 +  # volume ratio
                2 +  # time features
                2 +  # position state
                2)   # risk metrics


if __name__ == "__main__":
    # Test state encoder
    encoder = StateEncoder()
    
    # Simulate some price data
    prices = np.linspace(100, 110, 100)
    for i, price in enumerate(prices):
        state = encoder.encode_state(
            current_price=price,
            high=price * 1.001,
            low=price * 0.999,
            volume=1000.0,
            vwap=price * 0.998,
            timestamp=pd.Timestamp.now()
        )
        if i == len(prices) - 1:
            print(f"State vector size: {len(state)}")
            print(f"State vector: {state[:10]}...")  # Show first 10 features

