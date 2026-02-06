"""
VWAP Pro Strategy Implementation
Based on Quantum Traders VWAP Pro with advanced filters and indicators.

Strategy Rules:
- Entry: Buy when price tags lower band (2.0σ) and closes through it (no wick)
- Exit: When price hits new band level (reposition, avoid pullback)
- Reposition: When price crosses back above VWAP and retests (buy 3rd touch)
- Filters: Volume confirmation, 3-touch rule, volatility filter, session filter
- Indicators: FVG, VWMA, EMA-90, RSI-5 with hidden divergence
- Trailing stop: 0.7% trailing stop
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, List
from datetime import datetime, time
from .vwap_bands import VWAPBandsCalculator


class VWAPProStrategy:
    """
    VWAP Pro Strategy with all filters and indicators.
    """
    
    def __init__(self,
                 sigma_multiplier: float = 2.0,
                 lookback_periods: int = 20,
                 vwma_period: int = 10,
                 ema_period: int = 90,
                 rsi_period: int = 5,
                 atr_period: int = 14,
                 trailing_stop_pct: float = 0.007,  # 0.7%
                 volatility_threshold: float = 2.5,
                 session_start_utc: int = 8,  # London open
                 session_end_utc: int = 17,  # NY close
                 min_volume_multiplier: float = 1.2,  # Minimum volume multiplier
                 volume_spike_multiplier: float = 1.5,  # Volume spike requirement
                 min_touches_before_entry: int = 3):  # 3-touch rule
        """
        Initialize VWAP Pro Strategy.
        
        Args:
            sigma_multiplier: Standard deviation multiplier for bands (default 2.0)
            lookback_periods: Lookback for VWAP calculation
            vwma_period: Volume-weighted moving average period
            ema_period: EMA period for trend filter
            rsi_period: RSI period for divergence detection
            atr_period: ATR period for volatility filter
            trailing_stop_pct: Trailing stop percentage (0.7%)
            volatility_threshold: ATR multiplier threshold (2.5x)
            session_start_utc: Trading session start (UTC)
            session_end_utc: Trading session end (UTC)
        """
        self.sigma_multiplier = sigma_multiplier
        self.lookback_periods = lookback_periods
        self.vwma_period = vwma_period
        self.ema_period = ema_period
        self.rsi_period = rsi_period
        self.atr_period = atr_period
        self.trailing_stop_pct = trailing_stop_pct
        self.volatility_threshold = volatility_threshold
        self.session_start_utc = session_start_utc
        self.session_end_utc = session_end_utc
        self.min_volume_multiplier = min_volume_multiplier
        self.volume_spike_multiplier = volume_spike_multiplier
        self.min_touches_before_entry = min_touches_before_entry
        
        # Initialize calculators
        self.vwap_calc = VWAPBandsCalculator(stdev1=sigma_multiplier, stdev2=sigma_multiplier*1.5)
        
        # History tracking
        self.price_history = []
        self.volume_history = []
        self.high_history = []
        self.low_history = []
        self.close_history = []
        
        # Strategy state
        self.position = 0  # 0=flat, 1=long, -1=short
        self.entry_price = 0.0
        self.trailing_stop_price = 0.0
        self.band_touch_count = {}  # Track touches per band
        self.last_swing_low = None
        self.last_swing_high = None
        self.last_rsi_low = None
        self.last_rsi_high = None
        
        # FVG tracking
        self.fvg_list = []  # List of Fair Value Gaps
        
    def update(self,
              current_price: float,
              high: float,
              low: float,
              close: float,
              volume: float,
              timestamp: Optional[datetime] = None) -> Dict:
        """
        Update strategy with new bar data.
        
        Returns:
            Dictionary with signals and state
        """
        # Update history
        self.price_history.append(current_price)
        self.volume_history.append(volume)
        self.high_history.append(high)
        self.low_history.append(low)
        self.close_history.append(close)
        
        # Keep history manageable
        max_history = max(self.lookback_periods, self.ema_period, 100)
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.volume_history = self.volume_history[-max_history:]
            self.high_history = self.high_history[-max_history:]
            self.low_history = self.low_history[-max_history:]
            self.close_history = self.close_history[-max_history:]
        
        # Calculate indicators
        vwap_data = self.vwap_calc.calculate(high, low, close, volume, timestamp)
        vwap = vwap_data['vwap']
        lower_band = vwap_data['s1']  # Lower band (sigma multiplier)
        upper_band = vwap_data['r1']  # Upper band
        
        # Calculate additional indicators
        indicators = self._calculate_indicators()
        
        # Check filters
        filters = self._check_filters(timestamp, indicators)
        
        # Detect FVG
        fvg = self._detect_fvg()
        
        # Update FVG list
        if fvg:
            self.fvg_list.append(fvg)
            # Clean old FVGs
            self.fvg_list = [g for g in self.fvg_list if g['still_open']]
        
        # Check for band touches
        band_touch = self._check_band_touch(low, high, lower_band, upper_band, vwap)
        
        # Check RSI divergence
        rsi_divergence = self._check_rsi_divergence(indicators['rsi'], low, high)
        
        # Generate signals
        signals = self._generate_signals(
            close, high, low, vwap, lower_band, upper_band,
            band_touch, indicators, filters, rsi_divergence, fvg
        )
        
        # Update trailing stop
        if self.position != 0:
            self._update_trailing_stop(close, high, low)
        
        return {
            'vwap': vwap,
            'lower_band': lower_band,
            'upper_band': upper_band,
            'indicators': indicators,
            'filters': filters,
            'signals': signals,
            'position': self.position,
            'band_touch': band_touch,
            'rsi_divergence': rsi_divergence,
            'fvg': fvg
        }
    
    def _calculate_indicators(self) -> Dict:
        """Calculate all indicators."""
        if len(self.close_history) < max(self.vwma_period, self.ema_period, self.rsi_period):
            return {
                'vwma': self.close_history[-1] if self.close_history else 0.0,
                'ema': self.close_history[-1] if self.close_history else 0.0,
                'rsi': 50.0,
                'atr': 0.0,
                'vwma_slope': 0.0,
                'ema_slope': 0.0
            }
        
        closes = np.array(self.close_history)
        volumes = np.array(self.volume_history)
        highs = np.array(self.high_history)
        lows = np.array(self.low_history)
        
        # VWMA (Volume-Weighted Moving Average)
        if len(volumes) >= self.vwma_period:
            recent_closes = closes[-self.vwma_period:]
            recent_volumes = volumes[-self.vwma_period:]
            vwma = np.sum(recent_closes * recent_volumes) / np.sum(recent_volumes)
            vwma_slope = (vwma - closes[-self.vwma_period-1]) / closes[-self.vwma_period-1] if len(closes) > self.vwma_period else 0.0
        else:
            vwma = closes[-1]
            vwma_slope = 0.0
        
        # EMA
        ema = self._calculate_ema(closes, self.ema_period)
        ema_slope = (ema - closes[-self.ema_period-1]) / closes[-self.ema_period-1] if len(closes) > self.ema_period else 0.0
        
        # RSI
        rsi = self._calculate_rsi(closes, self.rsi_period)
        
        # ATR
        atr = self._calculate_atr(highs, lows, closes, self.atr_period)
        
        return {
            'vwma': vwma,
            'ema': ema,
            'rsi': rsi,
            'atr': atr,
            'vwma_slope': vwma_slope,
            'ema_slope': ema_slope
        }
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate EMA."""
        if len(prices) < period:
            return prices[-1]
        alpha = 2.0 / (period + 1)
        ema = prices[-period]
        for price in prices[-period+1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_atr(self, highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float:
        """Calculate ATR."""
        if len(highs) < period + 1:
            return 0.0
        tr_list = []
        for i in range(len(highs) - period, len(highs)):
            if i > 0:
                tr1 = highs[i] - lows[i]
                tr2 = abs(highs[i] - closes[i-1])
                tr3 = abs(lows[i] - closes[i-1])
                tr = max(tr1, tr2, tr3)
                tr_list.append(tr)
        return np.mean(tr_list) if tr_list else 0.0
    
    def _check_filters(self, timestamp: Optional[datetime], indicators: Dict) -> Dict:
        """Check all filters."""
        filters = {
            'volatility_ok': True,
            'session_ok': True,
            'volume_ok': True,
            'ema_trend_ok': True
        }
        
        # Volatility filter (stricter - avoid extreme volatility)
        if len(self.close_history) >= self.atr_period * 2:
            recent_atr = indicators['atr']
            atr_history = []
            for i in range(len(self.high_history) - self.atr_period * 2, len(self.high_history) - self.atr_period):
                if i >= self.atr_period:
                    atr_val = self._calculate_atr(
                        np.array(self.high_history[i-self.atr_period:i+1]),
                        np.array(self.low_history[i-self.atr_period:i+1]),
                        np.array(self.close_history[i-self.atr_period:i+1]),
                        self.atr_period
                    )
                    atr_history.append(atr_val)
            
            if atr_history:
                avg_atr = np.mean(atr_history)
                # Stricter: avoid if volatility is too high OR too low (choppy)
                if recent_atr > self.volatility_threshold * avg_atr or recent_atr < avg_atr * 0.5:
                    filters['volatility_ok'] = False
        
        # Session filter
        if timestamp:
            hour_utc = timestamp.hour
            if not (self.session_start_utc <= hour_utc <= self.session_end_utc):
                filters['session_ok'] = False
        
        # Volume filter (TIGHTER - require significant volume spike)
        min_vol_mult = getattr(self, 'min_volume_multiplier', 1.2)  # Default 1.2 (20% above)
        volume_spike_mult = getattr(self, 'volume_spike_multiplier', 1.5)  # Default 1.5x spike
        
        if len(self.volume_history) >= 20:
            recent_vol = self.volume_history[-1]
            avg_vol = np.mean(self.volume_history[-20:])  # 20-bar average
            if recent_vol < avg_vol * min_vol_mult:
                filters['volume_ok'] = False
            # Also check for volume spike (current vs recent 3 bars)
            if len(self.volume_history) >= 4:
                recent_3_avg = np.mean(self.volume_history[-4:-1])
                if recent_vol < recent_3_avg * volume_spike_mult:
                    filters['volume_ok'] = False
        elif len(self.volume_history) >= 3:
            recent_vol = self.volume_history[-1]
            avg_vol = np.mean(self.volume_history[-3:-1])
            if recent_vol < avg_vol * min_vol_mult * 1.2:  # Even stricter for short history
                filters['volume_ok'] = False
        
        # EMA trend filter (stricter - must be flat or up)
        if indicators['ema_slope'] < 0.0005:  # EMA must be flat or sloping up (was -0.001)
            filters['ema_trend_ok'] = False
        
        return filters
    
    def _detect_fvg(self) -> Optional[Dict]:
        """Detect Fair Value Gap (3-candle imbalance)."""
        if len(self.close_history) < 3:
            return None
        
        # Check for bullish FVG (gap up)
        if (self.low_history[-1] > self.high_history[-3] and
            self.close_history[-2] > self.high_history[-3]):
            return {
                'type': 'bullish',
                'top': self.low_history[-1],
                'bottom': self.high_history[-3],
                'filled': False,
                'still_open': True
            }
        
        # Check for bearish FVG (gap down)
        if (self.high_history[-1] < self.low_history[-3] and
            self.close_history[-2] < self.low_history[-3]):
            return {
                'type': 'bearish',
                'top': self.low_history[-3],
                'bottom': self.high_history[-1],
                'filled': False,
                'still_open': True
            }
        
        return None
    
    def _check_band_touch(self, low: float, high: float, lower_band: float, upper_band: float, vwap: float) -> Dict:
        """Check if price touched bands."""
        touch = {
            'lower_band': low <= lower_band,
            'upper_band': high >= upper_band,
            'vwap_cross': False,
            'vwap_above': False
        }
        
        # Check VWAP cross
        if len(self.close_history) >= 2:
            prev_close = self.close_history[-2]
            curr_close = self.close_history[-1]
            if prev_close < vwap and curr_close > vwap:
                touch['vwap_cross'] = True
            touch['vwap_above'] = curr_close > vwap
        
        # Track touch count
        if touch['lower_band']:
            key = 'lower'
            if key not in self.band_touch_count:
                self.band_touch_count[key] = 0
            self.band_touch_count[key] += 1
        
        if touch['upper_band']:
            key = 'upper'
            if key not in self.band_touch_count:
                self.band_touch_count[key] = 0
            self.band_touch_count[key] += 1
        
        return touch
    
    def _check_rsi_divergence(self, rsi: float, low: float, high: float) -> Dict:
        """Check for RSI hidden divergence."""
        divergence = {
            'bullish': False,
            'bearish': False
        }
        
        if len(self.close_history) < 10:
            return divergence
        
        # Track swing lows/highs
        if len(self.close_history) >= 5:
            # Simple swing detection
            recent_lows = self.low_history[-5:]
            recent_highs = self.high_history[-5:]
            
            current_low_idx = np.argmin(recent_lows)
            current_high_idx = np.argmax(recent_highs)
            
            if current_low_idx == len(recent_lows) - 1:  # New swing low
                if self.last_swing_low is not None and low < self.last_swing_low:
                    # Lower low in price
                    if self.last_rsi_low is not None and rsi > self.last_rsi_low:
                        # Higher low in RSI = bullish divergence
                        divergence['bullish'] = True
                self.last_swing_low = low
                self.last_rsi_low = rsi
            
            if current_high_idx == len(recent_highs) - 1:  # New swing high
                if self.last_swing_high is not None and high > self.last_swing_high:
                    # Higher high in price
                    if self.last_rsi_high is not None and rsi < self.last_rsi_high:
                        # Lower high in RSI = bearish divergence
                        divergence['bearish'] = True
                self.last_swing_high = high
                self.last_rsi_high = rsi
        
        return divergence
    
    def _generate_signals(self,
                         close: float,
                         high: float,
                         low: float,
                         vwap: float,
                         lower_band: float,
                         upper_band: float,
                         band_touch: Dict,
                         indicators: Dict,
                         filters: Dict,
                         rsi_divergence: Dict,
                         fvg: Optional[Dict]) -> Dict:
        """Generate trading signals based on strategy rules."""
        signals = {
            'entry_long': False,
            'entry_short': False,
            'exit': False,
            'reposition_long': False,
            'stop_hit': False
        }
        
        # Check trailing stop
        if self.position != 0 and self.trailing_stop_price > 0:
            if self.position == 1 and low <= self.trailing_stop_price:
                signals['stop_hit'] = True
                signals['exit'] = True
            elif self.position == -1 and high >= self.trailing_stop_price:
                signals['stop_hit'] = True
                signals['exit'] = True
        
        # Exit if price hits new band level (reposition logic)
        if self.position == 1 and band_touch['upper_band']:
            signals['exit'] = True  # Reposition, don't get dragged into pullback
        
        if self.position == -1 and band_touch['lower_band']:
            signals['exit'] = True
        
        # Entry: Buy when price tags lower band and closes through it (no wick)
        # STRICTER CONDITIONS
        if (self.position == 0 and
            band_touch['lower_band'] and
            close > lower_band * 1.001 and  # Closed well through band (0.1% buffer)
            low <= lower_band and  # Tagged band
            filters['volatility_ok'] and
            filters['session_ok'] and
            filters['volume_ok'] and
            filters['ema_trend_ok']):
            
            # N-touch rule: Only enter on Nth touch (configurable)
            touch_count = self.band_touch_count.get('lower', 0)
            min_touches = getattr(self, 'min_touches_before_entry', 3)
            if touch_count >= min_touches:
                # Check VWMA slope (must be positive)
                if indicators['vwma_slope'] > 0:
                    # RSI divergence preferred but not required
                    if rsi_divergence['bullish'] or touch_count >= min_touches:
                        # Additional: Check candle body (avoid dojis)
                        if len(self.close_history) >= 2:
                            prev_close = self.close_history[-2]
                            body_size = abs(close - prev_close)
                            total_range = high - low
                            if total_range > 0 and body_size / total_range >= 0.3:
                                signals['entry_long'] = True
                        else:
                            signals['entry_long'] = True
        
        # Reposition: Buy when price crosses back above VWAP and retests
        if (self.position == 0 and
            band_touch['vwap_cross'] and
            band_touch['vwap_above'] and
            low <= vwap and  # Retested VWAP from above
            close > vwap and  # Closed above VWAP
            filters['volatility_ok'] and
            filters['session_ok'] and
            filters['volume_ok']):
            
            # Check if FVG is filled (preferred)
            fvg_filled = True
            if fvg and fvg['type'] == 'bullish':
                if low <= fvg['bottom']:
                    fvg_filled = True
            
            if fvg_filled:
                signals['reposition_long'] = True
        
        return signals
    
    def _update_trailing_stop(self, close: float, high: float, low: float):
        """Update trailing stop."""
        if self.position == 1:  # Long position
            # Calculate new high
            if close > self.entry_price * (1 + self.trailing_stop_pct):
                # In profit, start trailing
                new_stop = close * (1 - self.trailing_stop_pct)
                if new_stop > self.trailing_stop_price:
                    self.trailing_stop_price = new_stop
            elif self.trailing_stop_price == 0:
                # Initialize trailing stop once in profit
                if close > self.entry_price * (1 + self.trailing_stop_pct):
                    self.trailing_stop_price = close * (1 - self.trailing_stop_pct)
        
        elif self.position == -1:  # Short position
            if close < self.entry_price * (1 - self.trailing_stop_pct):
                new_stop = close * (1 + self.trailing_stop_pct)
                if new_stop < self.trailing_stop_price or self.trailing_stop_price == 0:
                    self.trailing_stop_price = new_stop
    
    def enter_long(self, price: float):
        """Enter long position."""
        self.position = 1
        self.entry_price = price
        self.trailing_stop_price = 0.0
    
    def enter_short(self, price: float):
        """Enter short position."""
        self.position = -1
        self.entry_price = price
        self.trailing_stop_price = 0.0
    
    def exit_position(self):
        """Exit current position."""
        self.position = 0
        self.entry_price = 0.0
        self.trailing_stop_price = 0.0
