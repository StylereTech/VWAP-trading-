"""
VWAP Control Flip Strategy
Based on precise rulebook: control flips, band progression, filters
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple
import math
import pandas as pd
import numpy as np
from datetime import datetime, time

# Risk management imports
try:
    from .risk import (
        DrawdownGovernor,
        GovernorConfig,
        compute_position_units,
        compute_stop_distance,
        InstrumentSpec,
        SizingConfig
    )
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from capital_allocation_ai.risk import (
        DrawdownGovernor,
        GovernorConfig,
        compute_position_units,
        compute_stop_distance,
        InstrumentSpec,
        SizingConfig
    )


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    rs = ema(up, length) / (ema(down, length) + 1e-12)
    return 100 - (100 / (1 + rs))


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return ema(tr, length)


def vwma(close: pd.Series, volume: pd.Series, length: int) -> pd.Series:
    """Volume-Weighted Moving Average."""
    pv = close * volume
    return pv.rolling(length).sum() / (volume.rolling(length).sum() + 1e-12)


def _reset_day_id(ts: pd.Series, reset_hour_utc: int = 23) -> pd.Series:
    """Calculate day ID that resets at specified UTC hour."""
    if not isinstance(ts.iloc[0], pd.Timestamp):
        ts = pd.to_datetime(ts)
    shifted = ts - pd.to_timedelta(reset_hour_utc, unit="h")
    return shifted.dt.floor("D")


def vwap_and_sigma(df: pd.DataFrame, reset_hour_utc: int = 23) -> Tuple[pd.Series, pd.Series]:
    """
    Daily-reset VWAP + volume-weighted stdev of price around VWAP.
    Price for VWAP uses typical price (H+L+C)/3.
    """
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].astype(float)

    if "timestamp" not in df.columns:
        df = df.copy()
        df["timestamp"] = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.date_range(start='2024-01-01', periods=len(df), freq='5min')

    day_id = _reset_day_id(df["timestamp"], reset_hour_utc=reset_hour_utc)

    vwap = pd.Series(index=df.index, dtype=float)
    sigma = pd.Series(index=df.index, dtype=float)

    for d, idx in df.groupby(day_id).groups.items():
        i = df.index[df.index.isin(idx)]
        tp_d = tp.loc[i]
        vol_d = vol.loc[i]

        cum_vol = vol_d.cumsum()
        cum_pv = (tp_d * vol_d).cumsum()
        vwap_d = cum_pv / (cum_vol + 1e-12)

        # Volume-weighted variance around VWAP
        diff = tp_d - vwap_d
        cum_w = cum_vol
        cum_wdiff2 = (vol_d * diff.pow(2)).cumsum()
        var_d = cum_wdiff2 / (cum_w + 1e-12)

        vwap.loc[i] = vwap_d
        sigma.loc[i] = np.sqrt(np.maximum(var_d, 0.0))

    return vwap, sigma


@dataclass
class StrategyParams:
    """Strategy parameters - optimized by quantum optimizer."""
    reset_hour_utc: int = 23
    band_k: float = 2.0
    band_k_list: Tuple[float, ...] = (1.0, 2.0, 3.0)

    ema_trend_len: int = 90
    vwma_len: int = 10
    atr_len: int = 20
    atr_ema_len: int = 5
    rsi_len: int = 5

    touch_tol_atr_frac: float = 0.05
    cross_lookback_bars: int = 12

    vol_sma_len: int = 20
    vol_mult: float = 1.1
    atr_cap_mult: float = 2.5
    
    # Volume filter options
    volume_filter_type: str = "multiplier"  # "multiplier", "percentile", "impulse", "reclaim_quality"
    vol_percentile_L: int = 50  # Lookback for percentile
    vol_percentile_p: float = 60.0  # Percentile threshold (55-70)
    body_atr_thresh: float = 0.8  # For impulse type (0.6-1.0)
    reclaim_atr_thresh: float = 0.05  # For reclaim_quality (close-vwap >= this * ATR) - loosened from 0.15
    body_ratio_thresh: float = 0.45  # For reclaim_quality (body ratio threshold) - loosened from 0.55
    
    # Flexible retest window (override require_nth_retest)
    retest_min: Optional[int] = None  # If set, use flexible window
    retest_max: Optional[int] = None  # If set, use flexible window

    require_session_filter: bool = True
    sessions_utc: Tuple[Tuple[str, str], ...] = (("08:00", "10:00"), ("13:30", "17:00"))

    require_nth_retest: Optional[int] = 3

    trail_pct: float = 0.007
    start_trail_profit_pct: float = 0.007
    
    # Risk management parameters (prop-firm-safe defaults)
    risk_per_trade_frac: float = 0.003  # 0.30% risk per trade
    atr_mult_stop: float = 2.0  # Stop distance multiplier
    enable_drawdown_governor: bool = True
    enable_enhanced_exits: bool = True
    
    # Enhanced exit parameters
    time_stop_bars: Optional[int] = None  # Exit if trade doesn't reach +0.5R within N bars
    break_even_r: float = 1.0  # Move stop to entry at +1R
    partial_tp_r: float = 1.0  # Take partial profit at +1R
    partial_tp_pct: float = 0.4  # Take 40% at partial TP
    loss_duration_cap_bars: Optional[int] = None  # Exit losers after N bars


def in_sessions(ts: pd.Timestamp, sessions: Tuple[Tuple[str, str], ...]) -> bool:
    """Check if timestamp is within trading sessions."""
    if isinstance(ts, str):
        ts = pd.to_datetime(ts)
    t = ts.time() if hasattr(ts, 'time') else pd.to_datetime(ts).time()
    
    for start_s, end_s in sessions:
        start = pd.to_datetime(start_s).time()
        end = pd.to_datetime(end_s).time()
        if start <= t <= end:
            return True
    return False


@dataclass
class PositionState:
    """Current position state."""
    side: str  # "long" or "short"
    entry_price: float
    qty: float
    highest: float
    lowest: float
    trail_active: bool = False
    stop_price: Optional[float] = None
    last_band_reached: float = 0.0


class VWAPControlFlipStrategy:
    """VWAP Control Flip Strategy with band progression."""
    
    def __init__(self, params: StrategyParams):
        self.p = params
        self.retest_count_since_cross = 0
        self.last_cross_dir = None  # "up" or "down"
        self.last_cross_bar = -999
        
        # History for indicators
        self.price_history = []
        self.volume_history = []
        self.high_history = []
        self.low_history = []
        self.close_history = []
        self.timestamp_history = []
        
        # Calculated indicators
        self.vwap_history = []
        self.sigma_history = []
        self.ema90_history = []
        self.vwma10_history = []
        self.atr20_history = []
        self.atr20_ema5_history = []
        self.rsi5_history = []
        self.vol_sma_history = []
        
        # Position
        self.position: Optional[PositionState] = None
        
    def update(self, current_price: float, high: float, low: float, close: float,
               volume: float, timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """Update strategy with new bar."""
        ts = timestamp or datetime.now()
        
        # Update history
        self.price_history.append(current_price)
        self.high_history.append(high)
        self.low_history.append(low)
        self.close_history.append(close)
        self.volume_history.append(volume)
        self.timestamp_history.append(ts)
        
        self.current_bar_index = len(self.close_history) - 1
        
        # Update drawdown governor
        if self.governor is not None:
            self.governor.on_bar(self.equity, ts)
        
        # Need minimum bars for indicators
        if len(self.close_history) < max(self.p.ema_trend_len, self.p.atr_len, self.p.vol_sma_len) + 10:
            return {
                'signals': {
                    'enter_long': False,
                    'enter_short': False,
                    'exit': False,
                    'hold': True
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
        df['vwma10'] = vwma(df['close'], df['volume'], self.p.vwma_len)
        df['atr20'] = atr(df, self.p.atr_len)
        df['atr20_ema5'] = ema(df['atr20'], self.p.atr_ema_len)
        df['rsi5'] = rsi(df['close'], self.p.rsi_len)
        df['vol_sma'] = df['volume'].rolling(self.p.vol_sma_len).mean()
        
        # Update bands
        for k in set(self.p.band_k_list + (self.p.band_k,)):
            df[f'ub_{k}'] = df['vwap'] + k * df['sigma']
            df[f'lb_{k}'] = df['vwap'] - k * df['sigma']
        
        # Get current row
        row = df.iloc[-1]
        row_prev = df.iloc[-2] if len(df) > 1 else row
        
        # Generate signal
        signal = self._generate_signal(row_prev, row)
        
        # Update position management
        exit_signal = False
        if self.position is not None:
            self.position, exit_action = self._update_position_management(self.position, row)
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
                'vwma10': row['vwma10'],
                'atr20': row['atr20']
            }
        }
    
    def _cross_above(self, prev_close, prev_vwap, close, vwap) -> bool:
        """Detect cross above VWAP."""
        return close > vwap and prev_close <= prev_vwap
    
    def _cross_below(self, prev_close, prev_vwap, close, vwap) -> bool:
        """Detect cross below VWAP."""
        return close < vwap and prev_close >= prev_vwap
    
    def _touch(self, value: float, target: float, tol: float) -> bool:
        """Check if value touches target within tolerance."""
        return abs(value - target) <= tol
    
    def _check_volume_filter(self, row: pd.Series, row_prev: pd.Series) -> bool:
        """Check volume filter based on configured type."""
        if self.p.volume_filter_type == "multiplier":
            # Original: vol >= mult * SMA(vol)
            return row['volume'] >= self.p.vol_mult * (row['vol_sma'] + 1e-12)
        
        elif self.p.volume_filter_type == "percentile":
            # Relative volume percentile within rolling window
            if len(self.volume_history) < self.p.vol_percentile_L:
                return True  # Not enough history
            
            vol_window = self.volume_history[-self.p.vol_percentile_L:]
            percentile_threshold = np.percentile(vol_window, self.p.vol_percentile_p)
            return row['volume'] >= percentile_threshold
        
        elif self.p.volume_filter_type == "impulse":
            # Volume OR range expansion (institutional initiative)
            vol_ok = row['volume'] >= row['vol_sma']
            body_size = abs(row['close'] - row['open'])
            range_expansion = body_size >= self.p.body_atr_thresh * row['atr20']
            return vol_ok or range_expansion
        
        elif self.p.volume_filter_type == "reclaim_quality":
            # VWAP participation filter - quality of reclaim
            # Only apply if reclaim just occurred (within lookback)
            if self.last_cross_bar < 0 or len(self.close_history) - self.last_cross_bar > self.p.cross_lookback_bars:
                return True  # Not a recent reclaim, skip this filter
            
            # Distance above/below VWAP (looser threshold)
            vwap_dist = abs(row['close'] - row['vwap'])
            vwap_dist_ok = vwap_dist >= self.p.reclaim_atr_thresh * row['atr20']
            
            # Body ratio (relaxed)
            body_size = abs(row['close'] - row['open'])
            candle_range = row['high'] - row['low']
            body_ratio = body_size / (candle_range + 1e-12)
            body_ok = body_ratio >= self.p.body_ratio_thresh
            
            return vwap_dist_ok and body_ok
        
        else:
            # Default: no filter
            return True
    
    def _generate_signal(self, row_prev: pd.Series, row: pd.Series) -> str:
        """Generate trading signal."""
        ts = row['timestamp']
        
        # Session filter
        if self.p.require_session_filter:
            if not in_sessions(ts, self.p.sessions_utc):
                return "hold"
        
        # Volatility cap
        if row['atr20'] > self.p.atr_cap_mult * row['atr20_ema5']:
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
        
        # Check if cross is within lookback
        cross_recent = (current_bar - self.last_cross_bar) <= self.p.cross_lookback_bars
        
        # Retest detection
        retest_hold = self._touch(row['low'], row['vwap'], tol) and (row['close'] > row['vwap'])
        retest_reject = self._touch(row['high'], row['vwap'], tol) and (row['close'] < row['vwap'])
        
        if retest_hold and self.last_cross_dir == "up":
            self.retest_count_since_cross += 1
        if retest_reject and self.last_cross_dir == "down":
            self.retest_count_since_cross += 1
        
        # Volume confirmation - multiple options
        vol_ok = self._check_volume_filter(row, row_prev)
        
        # Trend filters
        ema_up_or_flat = row['ema90'] >= row_prev['ema90']
        ema_down_or_flat = row['ema90'] <= row_prev['ema90']
        
        # VWMA slope
        vwma_up = row['vwma10'] >= row_prev['vwma10']
        vwma_down = row['vwma10'] <= row_prev['vwma10']
        
        # Nth retest filter - flexible window instead of exact match
        nth_ok = True
        if self.p.require_nth_retest is not None:
            # Use custom retest window if provided, otherwise use default flexible window
            if self.p.retest_min is not None and self.p.retest_max is not None:
                # Custom flexible window
                nth_ok = (self.retest_count_since_cross >= self.p.retest_min and 
                         self.retest_count_since_cross <= self.p.retest_max)
            elif self.p.require_nth_retest == 3:
                # Default flexible: allow 2nd or 3rd retest
                nth_ok = (self.retest_count_since_cross >= 2 and self.retest_count_since_cross <= 4)
            else:
                # For other values, use exact match or window
                nth_ok = (self.retest_count_since_cross == self.p.require_nth_retest)
        
        # Entries (only if flat)
        if self.position is None:
            if (self.last_cross_dir == "up" and retest_hold and vol_ok and 
                ema_up_or_flat and vwma_up and nth_ok and cross_recent):
                return "enter_long"
            
            if (self.last_cross_dir == "down" and retest_reject and vol_ok and 
                ema_down_or_flat and vwma_down and nth_ok and cross_recent):
                return "enter_short"
        
        return "hold"
    
    def _update_position_management(self, pos: PositionState, row: pd.Series) -> Tuple[PositionState, str]:
        """Update position management (trailing stops, band exits, enhanced exits)."""
        price = row['close']
        bars_in_trade = self.current_bar_index - pos.entry_bar
        
        # Calculate R (risk units) for enhanced exits
        if pos.initial_stop_distance > 0:
            if pos.side == "long":
                r_units = (price - pos.entry_price) / pos.initial_stop_distance
            else:
                r_units = (pos.entry_price - price) / pos.initial_stop_distance
        else:
            r_units = 0.0
        
        if pos.side == "long":
            pos.highest = max(pos.highest, price)
            
            # Enhanced exits (if enabled)
            if self.p.enable_enhanced_exits:
                # Time stop: exit if trade doesn't reach +0.5R within N bars
                if self.p.time_stop_bars is not None and bars_in_trade >= self.p.time_stop_bars:
                    if r_units < 0.5:
                        return pos, "exit"
                
                # Break-even: move stop to entry at +1R
                if not pos.break_even_set and r_units >= self.p.break_even_r:
                    pos.stop_price = pos.entry_price
                    pos.break_even_set = True
                
                # Partial TP: take 40% at +1R
                if not pos.partial_tp_taken and r_units >= self.p.partial_tp_r:
                    # In backtest, we simulate partial TP by reducing position size
                    # For now, we'll mark it and adjust exit logic
                    pos.partial_tp_taken = True
                    pos.qty = pos.qty * (1.0 - self.p.partial_tp_pct)
                
                # Loss duration cap: exit losers after N bars
                if self.p.loss_duration_cap_bars is not None and r_units < 0:
                    if bars_in_trade >= self.p.loss_duration_cap_bars:
                        return pos, "exit"
            
            # Activate trailing after threshold
            if not pos.trail_active and (price >= pos.entry_price * (1 + self.p.start_trail_profit_pct)):
                pos.trail_active = True
            
            if pos.trail_active:
                trail_stop = pos.highest * (1 - self.p.trail_pct)
                pos.stop_price = max(pos.stop_price or -math.inf, trail_stop)
            
            # Band progression exits
            for k in sorted(self.p.band_k_list, reverse=True):
                ub = row.get(f'ub_{k}', None)
                if ub is not None and price >= ub and pos.last_band_reached < k:
                    pos.last_band_reached = k
                    return pos, "exit"
            
            # Stop loss
            if pos.stop_price is not None and price <= pos.stop_price:
                return pos, "exit"
        
        else:  # short
            pos.lowest = min(pos.lowest, price)
            
            # Enhanced exits (if enabled)
            if self.p.enable_enhanced_exits:
                # Time stop
                if self.p.time_stop_bars is not None and bars_in_trade >= self.p.time_stop_bars:
                    if r_units < 0.5:
                        return pos, "exit"
                
                # Break-even
                if not pos.break_even_set and r_units >= self.p.break_even_r:
                    pos.stop_price = pos.entry_price
                    pos.break_even_set = True
                
                # Partial TP
                if not pos.partial_tp_taken and r_units >= self.p.partial_tp_r:
                    pos.partial_tp_taken = True
                    pos.qty = pos.qty * (1.0 - self.p.partial_tp_pct)
                
                # Loss duration cap
                if self.p.loss_duration_cap_bars is not None and r_units < 0:
                    if bars_in_trade >= self.p.loss_duration_cap_bars:
                        return pos, "exit"
            
            if not pos.trail_active and (price <= pos.entry_price * (1 - self.p.start_trail_profit_pct)):
                pos.trail_active = True
            
            if pos.trail_active:
                trail_stop = pos.lowest * (1 + self.p.trail_pct)
                pos.stop_price = min(pos.stop_price or math.inf, trail_stop)
            
            # Band progression exits
            for k in sorted(self.p.band_k_list, reverse=True):
                lb = row.get(f'lb_{k}', None)
                if lb is not None and price <= lb and pos.last_band_reached < k:
                    pos.last_band_reached = k
                    return pos, "exit"
            
            # Stop loss
            if pos.stop_price is not None and price >= pos.stop_price:
                return pos, "exit"
        
        return pos, "hold"
    
    def enter_long(self, price: float, qty: float = 1.0):
        """Enter long position."""
        self.position = PositionState(
            side="long",
            entry_price=price,
            qty=qty,
            highest=price,
            lowest=price
        )
    
    def enter_short(self, price: float, qty: float = 1.0):
        """Enter short position."""
        self.position = PositionState(
            side="short",
            entry_price=price,
            qty=qty,
            highest=price,
            lowest=price
        )
    
    def exit_position(self):
        """Exit current position."""
        self.position = None
