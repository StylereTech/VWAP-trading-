"""
Enhanced VWAP Control Flip Strategy
- Stretched targets for higher R:R (let winners run to opposite band)
- Stronger filters (trend alignment, RSI divergence, FVG, tighter ATR)
- Symbol-specific optimizations
- Better trailing stops (breakeven after 1R, then trail)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import math
import pandas as pd
import numpy as np
from datetime import datetime, time

# Import base strategy components
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capital_allocation_ai.vwap_control_flip_strategy import (
    ema, rsi, atr, vwma, vwap_and_sigma, StrategyParams as BaseStrategyParams
)


@dataclass
class EnhancedStrategyParams(BaseStrategyParams):
    """Enhanced parameters with R:R optimization."""
    # R:R Optimization
    stretch_targets: bool = True  # Let winners run to opposite band
    min_rr_ratio: float = 2.5  # Minimum reward:risk ratio target
    breakeven_after_rr: float = 1.0  # Move to breakeven after 1R profit
    
    # Enhanced Filters
    require_trend_alignment: bool = True  # Price must be above/below EMA-90
    require_rsi_divergence: bool = False  # RSI hidden divergence (optional override)
    require_fvg_fill: bool = True  # Fair Value Gap fill confirmation
    atr_cap_tighter: float = 2.0  # Tighter ATR cap (was 2.5)
    
    # Symbol-specific
    symbol_type: str = "FOREX"  # FOREX, CRYPTO, METAL
    use_ema200_filter: bool = False  # For Gold (long-term filter)
    
    # Volume Delta (simplified - using volume vs average)
    require_volume_imbalance: bool = True
    volume_imbalance_threshold: float = 1.2  # 20% above average
    
    # Note: volume_filter_type and related params inherited from BaseStrategyParams


class EnhancedVWAPControlFlipStrategy:
    """Enhanced control flip strategy with R:R optimization."""
    
    def __init__(self, params: EnhancedStrategyParams):
        self.p = params
        
        # State tracking
        self.retest_count_since_cross = 0
        self.last_cross_dir = None
        self.last_cross_bar = -999
        
        # History
        self.price_history = []
        self.volume_history = []
        self.high_history = []
        self.low_history = []
        self.close_history = []
        self.timestamp_history = []
        
        # Indicators cache
        self.vwap_history = []
        self.sigma_history = []
        self.ema90_history = []
        self.ema200_history = []
        self.vwma10_history = []
        self.atr20_history = []
        self.atr20_ema5_history = []
        self.rsi5_history = []
        self.vol_sma_history = []
        
        # Position
        self.position: Optional[Any] = None
        
        # FVG tracking
        self.fvg_list = []
        
    def update(self, current_price: float, high: float, low: float, close: float,
               volume: float, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Update strategy with new bar."""
        # Update history
        self.price_history.append(current_price)
        self.high_history.append(high)
        self.low_history.append(low)
        self.close_history.append(close)
        self.volume_history.append(volume)
        self.timestamp_history.append(timestamp or datetime.now())
        
        # Need minimum bars
        min_bars = max(self.p.ema_trend_len, self.p.atr_len, self.p.vol_sma_len) + 10
        if len(self.close_history) < min_bars:
            return {
                'signals': {
                    'enter_long': False, 'enter_short': False,
                    'exit': False, 'hold': True
                },
                'indicators': {}
            }
        
        # Calculate indicators
        df = pd.DataFrame({
            'timestamp': self.timestamp_history,
            'open': self.price_history,
            'high': self.high_history,
            'low': self.low_history,
            'close': self.close_history,
            'volume': self.volume_history
        })
        
        df['vwap'], df['sigma'] = vwap_and_sigma(df, reset_hour_utc=self.p.reset_hour_utc)
        df['ema90'] = ema(df['close'], self.p.ema_trend_len)
        if self.p.use_ema200_filter:
            df['ema200'] = ema(df['close'], 200)
        df['vwma10'] = vwma(df['close'], df['volume'], self.p.vwma_len)
        df['atr20'] = atr(df, self.p.atr_len)
        df['atr20_ema5'] = ema(df['atr20'], self.p.atr_ema_len)
        df['rsi5'] = rsi(df['close'], self.p.rsi_len)
        df['vol_sma'] = df['volume'].rolling(self.p.vol_sma_len).mean()
        
        # Bands
        for k in set(self.p.band_k_list + (self.p.band_k,)):
            df[f'ub_{k}'] = df['vwap'] + k * df['sigma']
            df[f'lb_{k}'] = df['vwap'] - k * df['sigma']
        
        row = df.iloc[-1]
        row_prev = df.iloc[-2] if len(df) > 1 else row
        
        # Generate signal
        signal = self._generate_signal(row_prev, row, df)
        
        # Update position management
        exit_signal = False
        if self.position is not None:
            self.position, exit_action = self._update_position_management_enhanced(self.position, row, row_prev)
            if exit_action == "exit":
                exit_signal = True
        
        return {
            'signals': {
                'enter_long': signal == 'enter_long',
                'enter_short': signal == 'enter_short',
                'exit': exit_signal,
                'hold': signal == 'hold' and not exit_signal
            },
            'indicators': {
                'vwap': row['vwap'],
                'sigma': row['sigma'],
                'ema90': row['ema90'],
                'rsi5': row['rsi5']
            }
        }
    
    def _check_trend_alignment(self, row: pd.Series, row_prev: pd.Series, direction: str) -> bool:
        """Check if price aligns with trend."""
        if not self.p.require_trend_alignment:
            return True
        
        # EMA-90 filter
        if direction == "long":
            return row['close'] > row['ema90'] and row['ema90'] >= row_prev['ema90']
        else:
            return row['close'] < row['ema90'] and row['ema90'] <= row_prev['ema90']
    
    def _check_rsi_divergence(self, df: pd.DataFrame, direction: str) -> bool:
        """Check for RSI hidden divergence."""
        if not self.p.require_rsi_divergence or len(df) < 10:
            return False
        
        # Simple divergence check
        recent_lows = df['low'].tail(10).values
        recent_rsi = df['rsi5'].tail(10).values
        
        if direction == "long":
            # Price makes lower low but RSI makes higher low
            price_low_idx = np.argmin(recent_lows)
            rsi_low_idx = np.argmin(recent_rsi)
            if price_low_idx == len(recent_lows) - 1 and rsi_low_idx < len(recent_rsi) - 1:
                if recent_lows[-1] < recent_lows[rsi_low_idx] and recent_rsi[-1] > recent_rsi[rsi_low_idx]:
                    return True
        
        return False
    
    def _check_fvg_fill(self, df: pd.DataFrame, direction: str) -> bool:
        """Check if Fair Value Gap is filled."""
        if not self.p.require_fvg_fill or len(df) < 3:
            return True  # Default to true if not enough data
        
        # Simple FVG: gap between candle 1 high and candle 3 low (bullish)
        # or candle 1 low and candle 3 high (bearish)
        if len(df) >= 3:
            c1 = df.iloc[-3]
            c2 = df.iloc[-2]
            c3 = df.iloc[-1]
            
            if direction == "long":
                # Bullish FVG: c1 high < c3 low (gap up)
                if c1['high'] < c3['low']:
                    # Check if filled (price touched the gap)
                    return c2['low'] <= (c1['high'] + c3['low']) / 2
            else:
                # Bearish FVG: c1 low > c3 high (gap down)
                if c1['low'] > c3['high']:
                    return c2['high'] >= (c1['low'] + c3['high']) / 2
        
        return True
    
    def _check_volume_imbalance(self, row: pd.Series) -> bool:
        """Check volume imbalance (simplified)."""
        if not self.p.require_volume_imbalance:
            return True
        
        # Volume should be above threshold
        return row['volume'] >= self.p.volume_imbalance_threshold * (row['vol_sma'] + 1e-12)
    
    def _cross_above(self, prev_close, prev_vwap, close, vwap) -> bool:
        return close > vwap and prev_close <= prev_vwap
    
    def _cross_below(self, prev_close, prev_vwap, close, vwap) -> bool:
        return close < vwap and prev_close >= prev_vwap
    
    def _touch(self, value: float, target: float, tol: float) -> bool:
        return abs(value - target) <= tol
    
    def _generate_signal(self, row_prev: pd.Series, row: pd.Series, df: pd.DataFrame) -> str:
        """Generate trading signal with enhanced filters."""
        ts = row['timestamp']
        
        # Session filter
        if self.p.require_session_filter:
            from capital_allocation_ai.vwap_control_flip_strategy import in_sessions
            if not in_sessions(ts, self.p.sessions_utc):
                return "hold"
        
        # Tighter ATR cap
        if row['atr20'] > self.p.atr_cap_tighter * row['atr20_ema5']:
            return "hold"
        
        tol = self.p.touch_tol_atr_frac * row['atr20']
        
        # Detect crosses
        crossed_up = self._cross_above(row_prev['close'], row_prev['vwap'], row['close'], row['vwap'])
        crossed_down = self._cross_below(row_prev['close'], row_prev['vwap'], row['close'], row['vwap'])
        
        current_bar = len(self.close_history) - 1
        
        if crossed_up:
            self.last_cross_dir = "up"
            self.retest_count_since_cross = 0
            self.last_cross_bar = current_bar
        elif crossed_down:
            self.last_cross_dir = "down"
            self.retest_count_since_cross = 0
            self.last_cross_bar = current_bar
        
        cross_recent = (current_bar - self.last_cross_bar) <= self.p.cross_lookback_bars
        
        # Retest detection
        retest_hold = self._touch(row['low'], row['vwap'], tol) and (row['close'] > row['vwap'])
        retest_reject = self._touch(row['high'], row['vwap'], tol) and (row['close'] < row['vwap'])
        
        if retest_hold and self.last_cross_dir == "up":
            self.retest_count_since_cross += 1
        if retest_reject and self.last_cross_dir == "down":
            self.retest_count_since_cross += 1
        
        # Volume confirmation
        vol_ok = (row['volume'] >= self.p.vol_mult * (row['vol_sma'] + 1e-12))
        vol_imbalance_ok = self._check_volume_imbalance(row)
        
        # Trend filters
        ema_up_or_flat = row['ema90'] >= row_prev['ema90']
        ema_down_or_flat = row['ema90'] <= row_prev['ema90']
        
        # VWMA slope
        vwma_up = row['vwma10'] >= row_prev['vwma10']
        vwma_down = row['vwma10'] <= row_prev['vwma10']
        
        # Nth retest
        nth_ok = True
        if self.p.require_nth_retest is not None:
            nth_ok = (self.retest_count_since_cross == self.p.require_nth_retest)
        
        # Enhanced filters
        trend_ok_long = self._check_trend_alignment(row, row_prev, "long")
        trend_ok_short = self._check_trend_alignment(row, row_prev, "short")
        
        rsi_div_long = self._check_rsi_divergence(df, "long") if self.p.require_rsi_divergence else True
        rsi_div_short = self._check_rsi_divergence(df, "short") if self.p.require_rsi_divergence else True
        
        fvg_ok_long = self._check_fvg_fill(df, "long")
        fvg_ok_short = self._check_fvg_fill(df, "short")
        
        # EMA-200 filter for Gold
        ema200_ok = True
        if self.p.use_ema200_filter and 'ema200' in row:
            if self.last_cross_dir == "up":
                ema200_ok = row['close'] > row['ema200']
            elif self.last_cross_dir == "down":
                ema200_ok = row['close'] < row['ema200']
        
        # Entries
        if self.position is None:
            if (self.last_cross_dir == "up" and retest_hold and vol_ok and vol_imbalance_ok and
                ema_up_or_flat and vwma_up and nth_ok and cross_recent and
                trend_ok_long and rsi_div_long and fvg_ok_long and ema200_ok):
                return "enter_long"
            
            if (self.last_cross_dir == "down" and retest_reject and vol_ok and vol_imbalance_ok and
                ema_down_or_flat and vwma_down and nth_ok and cross_recent and
                trend_ok_short and rsi_div_short and fvg_ok_short and ema200_ok):
                return "enter_short"
        
        return "hold"
    
    def _update_position_management_enhanced(self, pos: Any, row: pd.Series, row_prev: pd.Series) -> Tuple[Any, str]:
        """Enhanced position management with R:R optimization."""
        price = row['close']
        entry_price = pos.entry_price
        
        if pos.side == "long":
            pos.highest = max(pos.highest, price)
            
            # Calculate R:R
            risk = abs(entry_price - (pos.stop_price or entry_price * 0.99))
            reward = price - entry_price
            current_rr = reward / risk if risk > 0 else 0
            
            # Move to breakeven after 1R
            if not hasattr(pos, 'breakeven_set') and current_rr >= self.p.breakeven_after_rr:
                pos.stop_price = entry_price
                pos.breakeven_set = True
            
            # Trailing stop
            if not pos.trail_active and (price >= entry_price * (1 + self.p.start_trail_profit_pct)):
                pos.trail_active = True
            
            if pos.trail_active:
                trail_stop = pos.highest * (1 - self.p.trail_pct)
                pos.stop_price = max(pos.stop_price or -math.inf, trail_stop)
            
            # Stretched targets - let winners run to opposite band
            if self.p.stretch_targets:
                # Check momentum for continuation
                momentum_ok = (row['ema90'] >= row_prev['ema90'] and 
                              row['vwma10'] >= row_prev['vwma10'] and
                              row['volume'] >= row_prev['volume'] * 0.9)
                
                if momentum_ok:
                    # Target opposite band (upper band for longs)
                    for k in sorted(self.p.band_k_list, reverse=True):
                        ub = row.get(f'ub_{k}', None)
                        if ub is not None and price >= ub:
                            if not hasattr(pos, 'last_band_reached') or pos.last_band_reached < k:
                                pos.last_band_reached = k
                                # Don't exit immediately - let it run
                                pass
                else:
                    # No momentum - exit at first band touch
                    for k in sorted(self.p.band_k_list, reverse=True):
                        ub = row.get(f'ub_{k}', None)
                        if ub is not None and price >= ub:
                            if not hasattr(pos, 'last_band_reached') or pos.last_band_reached < k:
                                pos.last_band_reached = k
                                return pos, "exit"
            else:
                # Original logic - exit at first band
                for k in sorted(self.p.band_k_list, reverse=True):
                    ub = row.get(f'ub_{k}', None)
                    if ub is not None and price >= ub:
                        if not hasattr(pos, 'last_band_reached') or pos.last_band_reached < k:
                            pos.last_band_reached = k
                            return pos, "exit"
            
            # Stop loss
            if pos.stop_price is not None and price <= pos.stop_price:
                return pos, "exit"
        
        else:  # short
            pos.lowest = min(pos.lowest, price)
            
            risk = abs(entry_price - (pos.stop_price or entry_price * 1.01))
            reward = entry_price - price
            current_rr = reward / risk if risk > 0 else 0
            
            if not hasattr(pos, 'breakeven_set') and current_rr >= self.p.breakeven_after_rr:
                pos.stop_price = entry_price
                pos.breakeven_set = True
            
            if not pos.trail_active and (price <= entry_price * (1 - self.p.start_trail_profit_pct)):
                pos.trail_active = True
            
            if pos.trail_active:
                trail_stop = pos.lowest * (1 + self.p.trail_pct)
                pos.stop_price = min(pos.stop_price or math.inf, trail_stop)
            
            # Stretched targets for shorts
            if self.p.stretch_targets:
                momentum_ok = (row['ema90'] <= row_prev['ema90'] and 
                              row['vwma10'] <= row_prev['vwma10'] and
                              row['volume'] >= row_prev['volume'] * 0.9)
                
                if momentum_ok:
                    for k in sorted(self.p.band_k_list, reverse=True):
                        lb = row.get(f'lb_{k}', None)
                        if lb is not None and price <= lb:
                            if not hasattr(pos, 'last_band_reached') or pos.last_band_reached < k:
                                pos.last_band_reached = k
                else:
                    for k in sorted(self.p.band_k_list, reverse=True):
                        lb = row.get(f'lb_{k}', None)
                        if lb is not None and price <= lb:
                            if not hasattr(pos, 'last_band_reached') or pos.last_band_reached < k:
                                pos.last_band_reached = k
                                return pos, "exit"
            else:
                for k in sorted(self.p.band_k_list, reverse=True):
                    lb = row.get(f'lb_{k}', None)
                    if lb is not None and price <= lb:
                        if not hasattr(pos, 'last_band_reached') or pos.last_band_reached < k:
                            pos.last_band_reached = k
                            return pos, "exit"
            
            if pos.stop_price is not None and price >= pos.stop_price:
                return pos, "exit"
        
        return pos, "hold"
    
    def enter_long(self, price: float, qty: float = 1.0):
        """Enter long position."""
        from capital_allocation_ai.vwap_control_flip_strategy import PositionState
        self.position = PositionState(
            side="long", entry_price=price, qty=qty,
            highest=price, lowest=price
        )
    
    def enter_short(self, price: float, qty: float = 1.0):
        """Enter short position."""
        from capital_allocation_ai.vwap_control_flip_strategy import PositionState
        self.position = PositionState(
            side="short", entry_price=price, qty=qty,
            highest=price, lowest=price
        )
    
    def exit_position(self):
        """Exit current position."""
        self.position = None
