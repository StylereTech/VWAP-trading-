import backtrader as bt
import math

class VWAPWithBands(bt.Indicator):
    """
    VWAP indicator with volume-weighted standard deviation bands (session-reset).
    """
    lines = ('vwap', 'r1', 's1', 'r2', 's2', 'r3', 's3')
    params = (('stdev1', 1.0), ('stdev2', 2.0), ('stdev3', 3.0))
    
    def __init__(self):
        self.current_day = None
        self.cum_typ_vol = 0.0
        self.cum_vol = 0.0
        self.day_typical_prices = []
        self.day_volumes = []
        self.addminperiod(1)
    
    def next(self):
        if len(self.data) == 0:
            return

        dt = self.data.datetime.date(0)
        if self.current_day != dt:
            self.current_day = dt
            self.cum_typ_vol = 0.0
            self.cum_vol = 0.0
            self.day_typical_prices = []
            self.day_volumes = []

        typical_price = (self.data.high[0] + self.data.low[0] + self.data.close[0]) / 3.0
        volume = float(self.data.volume[0]) if self.data.volume[0] is not None else 0.0

        self.day_typical_prices.append(typical_price)
        self.day_volumes.append(volume)

        self.cum_typ_vol += typical_price * volume
        self.cum_vol += volume

        vwap = (self.cum_typ_vol / self.cum_vol) if self.cum_vol > 0 else typical_price
        self.lines.vwap[0] = vwap

        if self.cum_vol > 0 and len(self.day_typical_prices) > 1:
            wssd = 0.0
            for tp, vol in zip(self.day_typical_prices, self.day_volumes):
                wssd += ((tp - vwap) ** 2) * vol
            std_dev = math.sqrt(wssd / self.cum_vol)
        else:
            std_dev = 0.0

        self.lines.r1[0] = vwap + self.p.stdev1 * std_dev
        self.lines.s1[0] = vwap - self.p.stdev1 * std_dev
        self.lines.r2[0] = vwap + self.p.stdev2 * std_dev
        self.lines.s2[0] = vwap - self.p.stdev2 * std_dev
        self.lines.r3[0] = vwap + self.p.stdev3 * std_dev
        self.lines.s3[0] = vwap - self.p.stdev3 * std_dev


class TradeLockerVWAPStrategy(bt.Strategy):
    """
    TradeLocker optimized VWAP strategy with adaptive risk management.
    Optimized for GBP/JPY 1-minute timeframe with increased trade frequency.
    """
    params = dict(
        use_band_level=1,            # 1=R1/S1 (closer bands = more touches)
        touch_tolerance=0.002,       # 0.2% tolerance for relaxed entries (increased from 0.001)
        relaxed_entry=True,          # Enable relaxed entry conditions
        use_volume_filter=True,      # Enable volume confirmation
        vol_period=20,               # Volume SMA period
        min_vol_mult=0.70,           # Minimum volume multiplier (lowered from 0.80 to allow more entries)
        use_volatility_filter=True,  # Enable ATR-based volatility filter
        atr_period=14,               # ATR period
        atr_multiplier=1.50,         # ATR multiplier for volatility filter
        min_volatility_threshold=0.004,   # 0.4% minimum volatility (lowered from 0.005)
        risk_per_trade=0.02,         # 2% risk per trade (not used when fixed_lot_size is set)
        max_position_pct=0.10,       # 10% maximum equity per position
        use_opposite_band_target=True,   # Target opposite VWAP band on exit
        use_vwap_early_exit=True,         # Exit if price crosses VWAP unfavorably
        atr_stop_multiplier=1.4,     # Stop loss ATR multiplier (tightened from 1.5)
        max_concurrent_trades=1,     # Only one position at a time
        min_warmup_bars=30,          # Minimum bars before trading
        enable_logging=True,
        log_frequency=20,            # How often to log status messages
        use_session_filter=False,    # Disabled by default; add session logic if needed
        performance_window=20,       # Rolling window for performance adaptation
        optimization_frequency=50,   # Bars between parameter adjustments
        partial_profit_taking=True,  # Enable partial profit taking at 50% target move
        fixed_lot_size=0.10          # Fixed lot size (mini lot) - overrides risk-based sizing
    )
    
    def __init__(self):
        # Track initial equity for drawdown and ROI calculations
        self.initial_equity = None
        self.peak_equity = None
        
        # Order and position management
        self.order = None
        self.position_side = None
        self.entry_price = None
        self.stop_price = None
        self.target_price = None
        
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.long_trades = 0
        self.short_trades = 0
        
        # Adaptive performance tracking
        self.recent_trades = []         # 1 for win, 0 for loss
        self.performance_window = self.p.performance_window
        self.optimization_frequency = self.p.optimization_frequency
        self.bars_since_optimization = 0
        self.trading_halted = False
        self.halt_reason = ""
        
        # Initialize VWAP bands indicator
        self.vwap_bands = VWAPWithBands(self.data)
        self.vwap = self.vwap_bands.lines.vwap
        if self.p.use_band_level == 1:
            self.upper_band = self.vwap_bands.lines.r1
            self.lower_band = self.vwap_bands.lines.s1
        else:
            self.upper_band = self.vwap_bands.lines.r2
            self.lower_band = self.vwap_bands.lines.s2
        
        # Additional indicators
        self.volume_sma = bt.indicators.SMA(self.data.volume, period=self.p.vol_period)
        self.atr = bt.indicators.ATR(self.data, period=self.p.atr_period)
        
        self.log(f"Strategy initialized: Using band level {self.p.use_band_level}")
    
    def get_initial_equity(self):
        """Get and cache initial equity on first call."""
        if self.initial_equity is None:
            self.initial_equity = float(self.broker.getvalue())
            self.peak_equity = self.initial_equity
            self.log(f"Initial equity set: ${self.initial_equity:.2f}")
        return self.initial_equity
    
    def update_peak_equity(self):
        """Track peak equity for drawdown calculation."""
        current_equity = float(self.broker.getvalue())
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
    
    def calculate_drawdown(self):
        """Calculate current drawdown percentage."""
        if self.peak_equity is None or self.peak_equity == 0:
            return 0.0
        current_equity = float(self.broker.getvalue())
        drawdown = ((self.peak_equity - current_equity) / self.peak_equity) * 100
        return max(0.0, drawdown)
    
    def is_ready_to_trade(self):
        if len(self.data) < self.p.min_warmup_bars:
            return False, "Insufficient warmup bars"
        if len(self.vwap) == 0:
            return False, "VWAP not ready"
        if self.p.use_volume_filter and len(self.volume_sma) < self.p.vol_period:
            return False, "Volume SMA not ready"
        if len(self.atr) < self.p.atr_period:
            return False, "ATR not ready"
        try:
            upper = self.upper_band[0]
            lower = self.lower_band[0]
            vwap_val = self.vwap[0]
            if any(x is None or math.isnan(x) for x in [upper, lower, vwap_val]):
                return False, "Invalid band values"
            if not (lower < vwap_val < upper):
                return False, "Invalid band order"
        except Exception as e:
            return False, f"Band validation error: {e}"
        return True, "Ready"
    
    def volume_filter_check(self):
        if not self.p.use_volume_filter:
            return True, "Volume filter disabled"
        if self.volume_sma[0] in (None, 0):
            return False, "Volume SMA invalid"
        current_vol = float(self.data.volume[0])
        avg_vol = float(self.volume_sma[0])
        vol_ratio = current_vol / avg_vol
        return vol_ratio >= self.p.min_vol_mult, f"Vol ratio: {vol_ratio:.2f}"
    
    def volatility_filter_check(self):
        if not self.p.use_volatility_filter:
            return True, "Volatility filter disabled"
        if self.atr[0] in (None, 0):
            return False, "ATR invalid"
        atr_value = float(self.atr[0])
        price = self.data.close[0]
        volatility_pct = (atr_value * self.p.atr_multiplier) / price
        return volatility_pct >= self.p.min_volatility_threshold, f"Volatility: {volatility_pct*100:.2f}%"
    
    def calculate_position_size(self, entry_price, stop_price):
        # Fixed lot size takes precedence
        if self.p.fixed_lot_size is not None and self.p.fixed_lot_size > 0:
            return self.p.fixed_lot_size
        
        # Fallback to risk-based sizing if fixed_lot_size not set
        try:
            equity = float(self.broker.getvalue())
            risk_amount = equity * self.p.risk_per_trade
            stop_distance = abs(entry_price - stop_price)
            if stop_distance <= 0:
                return 0.01
            risk_size = risk_amount / stop_distance
            max_size = (equity * self.p.max_position_pct) / entry_price
            final_size = max(0.01, min(risk_size, max_size))
            return round(final_size, 2)
        except Exception:
            return 0.01
        
    def get_stop_and_target_levels(self, direction, entry_price):
        try:
            vwap_level = float(self.vwap[0])
            atr_value = float(self.atr[0]) if self.atr[0] else 0
            if direction == 'long':
                stop_loss = entry_price - (atr_value * self.p.atr_stop_multiplier)
                take_profit = float(self.upper_band[0]) if self.p.use_opposite_band_target else vwap_level
            else:
                stop_loss = entry_price + (atr_value * self.p.atr_stop_multiplier)
                take_profit = float(self.lower_band[0]) if self.p.use_opposite_band_target else vwap_level
            return stop_loss, take_profit
        except Exception as e:
            self.log(f"Error calculating levels: {e}")
            return None, None
        
    def check_entry_signals(self):
        price = self.data.close[0]
        high = self.data.high[0]
        low = self.data.low[0]
        upper = self.upper_band[0]
        lower = self.lower_band[0]
        if self.p.relaxed_entry:
            tol = self.p.touch_tolerance
            long_signal = (low <= lower * (1 + tol)) and (price > lower * (1 - tol/2))
            short_signal = (high >= upper * (1 - tol)) and (price < upper * (1 + tol/2))
        else:
            long_signal = (low <= lower and price > lower)
            short_signal = (high >= upper and price < upper)
        return long_signal, short_signal
    
    def check_partial_profit_conditions(self):
        if not self.p.partial_profit_taking or not self.position:
            return False, "Partial profits disabled"
        try:
            current_price = self.data.close[0]
            entry_price = self.entry_price
            target_price = self.target_price
            if self.position_side == 'long':
                halfway = entry_price + (target_price - entry_price) * 0.5
                return current_price >= halfway, "50% target reached"
            else:
                halfway = entry_price - (entry_price - target_price) * 0.5
                return current_price <= halfway, "50% target reached"
        except Exception:
            return False, "Partial profit check failed"
        
    def check_early_exit_conditions(self):
        if not self.position:
            return False, "No position"
        if self.p.use_vwap_early_exit:
            price = self.data.close[0]
            vwap_level = self.vwap[0]
            if self.position_side == 'long' and price < vwap_level:
                return True, "Long position: price below VWAP"
            if self.position_side == 'short' and price > vwap_level:
                return True, "Short position: price above VWAP"
        should_partial, reason = self.check_partial_profit_conditions()
        if should_partial:
            if abs(self.position.size) > 0.02:
                partial_size = round(abs(self.position.size) * 0.5, 2)
                if self.position_side == 'long':
                    self.sell(size=partial_size)
                else:
                    self.buy(size=partial_size)
                self.log(f"Partial profit taken: {reason}")
        return False, "No full exit condition"
    
    def check_risk_limits(self):
        """Check risk limits using actual equity values."""
        # Initialize equity tracking on first call
        self.get_initial_equity()
        self.update_peak_equity()
        
        current_equity = float(self.broker.getvalue())
        initial_equity = self.initial_equity
        
        # Drawdown threshold: 20% from peak
        drawdown_threshold = self.peak_equity * 0.80
        if current_equity <= drawdown_threshold:
            self.trading_halted = True
            drawdown_pct = self.calculate_drawdown()
            self.halt_reason = f"DRAWDOWN LIMIT: {drawdown_pct:.1f}% drawdown (Equity ${current_equity:.2f} below ${drawdown_threshold:.2f})"
            return False
        
        # Consecutive losses check
        if len(self.recent_trades) >= 5 and sum(self.recent_trades[-5:]) == 0:
            self.trading_halted = True
            self.halt_reason = "CONSECUTIVE LOSSES: 5 losing trades in a row"
            return False
        
        # Low win rate check
        if len(self.recent_trades) >= 15 and (sum(self.recent_trades[-15:]) / 15) < 0.5:
            self.trading_halted = True
            self.halt_reason = f"LOW WIN RATE: {(sum(self.recent_trades[-15:]) / 15)*100:.1f}% over last 15 trades"
            return False
        
        # Poor performance check
        if len(self.recent_trades) >= 10 and sum(self.recent_trades[-10:]) <= 2:
            self.trading_halted = True
            self.halt_reason = f"POOR PERFORMANCE: {sum(self.recent_trades[-10:])}/10 wins in recent trades"
            return False
        
        # Resume trading if conditions improve
        if self.trading_halted:
            if (current_equity > drawdown_threshold * 1.1 and
                len(self.recent_trades) >= 3 and
                sum(self.recent_trades[-3:]) >= 2):
                self.trading_halted = False
                self.halt_reason = ""
                self.log("TRADING RESUMED: Risk conditions improved")
                return True
            else:
                return False
        
        return True
    
    def adapt_parameters(self):
        if len(self.recent_trades) < 10:
            return
        recent_win_rate = sum(self.recent_trades[-self.performance_window:]) / float(min(len(self.recent_trades), self.performance_window))
        original_band = self.p.use_band_level
        original_vol_mult = self.p.min_vol_mult
        if recent_win_rate < 0.65:
            self.p.use_band_level = 2
            self.p.min_vol_mult = min(1.2, self.p.min_vol_mult + 0.1)
            self.p.atr_stop_multiplier = min(2.5, self.p.atr_stop_multiplier + 0.2)
        elif recent_win_rate > 0.85:
            self.p.use_band_level = 1
            self.p.min_vol_mult = max(0.5, self.p.min_vol_mult - 0.1)
            self.p.atr_stop_multiplier = max(1.0, self.p.atr_stop_multiplier - 0.1)
        
        # Update band references if level changed
        if self.p.use_band_level == 1:
            self.upper_band = self.vwap_bands.lines.r1
            self.lower_band = self.vwap_bands.lines.s1
        else:
            self.upper_band = self.vwap_bands.lines.r2
            self.lower_band = self.vwap_bands.lines.s2
        
        current_equity = float(self.broker.getvalue())
        initial_equity = self.get_initial_equity()
        equity_multiplier = current_equity / initial_equity if initial_equity > 0 else 1.0
        
        if equity_multiplier > 1.5:
            self.p.risk_per_trade = min(0.03, self.p.risk_per_trade + 0.002)
        elif equity_multiplier < 0.8:
            self.p.risk_per_trade = max(0.01, self.p.risk_per_trade - 0.002)
        
        if hasattr(self, 'atr') and len(self.atr) > 0:
            current_atr = float(self.atr[0])
            price = self.data.close[0]
            volatility_pct = (current_atr / price) * 100
            if volatility_pct > 1.5:
                self.p.touch_tolerance = min(0.003, self.p.touch_tolerance + 0.0005)
            elif volatility_pct < 0.5:
                self.p.touch_tolerance = max(0.0005, self.p.touch_tolerance - 0.0005)
        
        if (self.p.use_band_level != original_band or 
            abs(self.p.min_vol_mult - original_vol_mult) > 0.05):
            self.log("PARAMETER ADAPTATION:")
            self.log(f"  Recent win rate: {recent_win_rate*100:.1f}%")
            self.log(f"  Band level: {original_band} -> {self.p.use_band_level}")
            self.log(f"  Volume mult: {original_vol_mult:.2f} -> {self.p.min_vol_mult:.2f}")
            self.log(f"  Risk per trade: {self.p.risk_per_trade*100:.1f}%")
            self.log(f"  Touch tolerance: {self.p.touch_tolerance*100:.2f}%")
    
    def place_bracket_trade(self, direction, entry_price, stop_price, target_price):
        try:
            size = self.calculate_position_size(entry_price, stop_price)
            if size <= 0:
                self.log(f"Invalid position size: {size}")
                return False
            if direction == 'long':
                parent, stop_order, limit_order = self.buy_bracket(
                    price=entry_price, size=size,
                    stopprice=stop_price, limitprice=target_price
                )
                self.long_trades += 1
            else:
                parent, stop_order, limit_order = self.sell_bracket(
                    price=entry_price, size=size,
                    stopprice=stop_price, limitprice=target_price
                )
                self.short_trades += 1
            if parent:
                self.order = parent
                self.position_side = direction
                self.entry_price = entry_price
                self.stop_price = stop_price
                self.target_price = target_price
                self.total_trades += 1
                self.log(f"{direction.upper()} ENTRY:")
                self.log(f"  Price: {entry_price:.5f} | Size: {size}")
                self.log(f"  Stop: {stop_price:.5f} | Target: {target_price:.5f}")
                risk = abs(entry_price - stop_price) * size
                reward = abs(target_price - entry_price) * size
                rr = reward / risk if risk > 0 else 0
                self.log(f"  Risk: ${risk:.2f} | R:R = 1:{rr:.2f}")
                return True
            else:
                self.log(f"Failed to place {direction} bracket order")
                return False
        except Exception as e:
            self.log(f"Error placing {direction} trade: {e}")
            return False

    def next(self):
        self.bars_since_optimization += 1
        
        # Update peak equity tracking
        self.update_peak_equity()
        
        if self.order:
            return
        
        if not self.check_risk_limits():
            if self.p.enable_logging and len(self.data) % 100 == 0:
                self.log(f"TRADING HALTED: {self.halt_reason}")
            return
        
        if self.bars_since_optimization >= self.optimization_frequency:
            self.adapt_parameters()
            self.bars_since_optimization = 0
        
        ready, status = self.is_ready_to_trade()
        if not ready:
            if len(self.data) % self.p.log_frequency == 0:
                self.log(f"Not ready: {status}")
            return
        
        if self.position:
            should_exit, reason = self.check_early_exit_conditions()
            if should_exit:
                self.close()
                self.log(f"Early exit: {reason}")
                self.position_side = None
                return
        
        if self.position and self.p.max_concurrent_trades <= 1:
            return
        
        vol_ok, vol_msg = self.volume_filter_check()
        if not vol_ok:
            return
        
        volatility_ok, vol_status = self.volatility_filter_check()
        if not volatility_ok:
            return
        
        long_signal, short_signal = self.check_entry_signals()
        if (long_signal or short_signal) and self.p.enable_logging:
            self.log(f"SIGNAL: Long={long_signal}, Short={short_signal}")
            self.log(f"Filters: {vol_msg}, {vol_status}")
        
        entry_price = float(self.data.close[0])
        if long_signal and not self.position:
            stop_price, target_price = self.get_stop_and_target_levels('long', entry_price)
            if stop_price and target_price and stop_price < entry_price < target_price:
                self.place_bracket_trade('long', entry_price, stop_price, target_price)
        elif short_signal and not self.position:
            stop_price, target_price = self.get_stop_and_target_levels('short', entry_price)
            if stop_price and target_price and target_price < entry_price < stop_price:
                self.place_bracket_trade('short', entry_price, stop_price, target_price)
    
    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            if order.status == order.Completed:
                side = 'BUY' if order.isbuy() else 'SELL'
                self.log(f"FILLED: {side} {order.executed.size} @ {order.executed.price:.5f}")
            elif order.status in [order.Canceled, order.Rejected]:
                self.log(f"ORDER FAILED: {order.getstatusname()}")
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            pnl = trade.pnlcomm
            pnl_pct = (pnl / abs(trade.value)) * 100 if trade.value != 0 else 0
            outcome = 1 if pnl > 0 else 0
            self.recent_trades.append(outcome)
            if len(self.recent_trades) > self.performance_window:
                self.recent_trades.pop(0)
            if pnl > 0:
                self.winning_trades += 1
                result = "WIN"
            else:
                self.losing_trades += 1
                result = "LOSS"
            self.log(f"{result}: PnL=${pnl:.2f} ({pnl_pct:.1f}%)")
            if len(self.recent_trades) >= 5:
                recent_wins = sum(self.recent_trades[-5:])
                self.log(f"Recent performance: {recent_wins}/5 wins")
            self.position_side = None

    def stop(self):
        if self.p.enable_logging:
            initial_equity = self.get_initial_equity()
            final_value = self.broker.getvalue()
            roi = ((final_value - initial_equity) / initial_equity) * 100 if initial_equity > 0 else 0.0
            max_drawdown = self.calculate_drawdown()
            
            self.log("\n" + "="*60)
            self.log("TRADELOCKER VWAP STRATEGY - FINAL REPORT")
            self.log("="*60)
            self.log(f"Initial equity: ${initial_equity:.2f}")
            self.log(f"Final portfolio: ${final_value:.2f}")
            self.log(f"Total ROI: {roi:.1f}%")
            self.log(f"Maximum drawdown: {max_drawdown:.1f}%")
            self.log(f"Peak equity: ${self.peak_equity:.2f}")
            self.log("-"*60)
            self.log(f"Total trades: {self.total_trades}")
            self.log(f"Winning trades: {self.winning_trades}")
            self.log(f"Losing trades: {self.losing_trades}")
            self.log(f"Long trades: {self.long_trades}")
            self.log(f"Short trades: {self.short_trades}")
            
            if self.total_trades > 0:
                win_rate = (self.winning_trades / self.total_trades) * 100
                self.log(f"Win rate: {win_rate:.1f}%")
                long_pct = (self.long_trades / self.total_trades) * 100
                short_pct = (self.short_trades / self.total_trades) * 100
                self.log(f"Direction split: {long_pct:.1f}% Long, {short_pct:.1f}% Short")
                
                self.log("\n" + "-"*40)
                self.log("PERFORMANCE vs TARGETS:")
                self.log(f"Win Rate: {win_rate:.1f}% (Target: 55%+) {'✓' if win_rate>=55 else '✗'}")
                self.log(f"Max Drawdown: {max_drawdown:.1f}% (Target: ≤20%) {'✓' if max_drawdown<=20 else '✗'}")
                self.log(f"ROI: {roi:.1f}%")
                
                self.log("\n" + "-"*40)
                self.log("OPTIMIZATION RECOMMENDATIONS:")
                if win_rate < 55:
                    self.log("- Consider increasing band level (R2/S2) for quality")
                    self.log("- Tighten volume filter (min_vol_mult=0.80)")
                if max_drawdown > 20:
                    self.log("- Consider reducing risk per trade")
                    self.log("- Tighten stop loss (atr_stop_multiplier=1.2)")
                if self.long_trades > self.short_trades * 2:
                    self.log("- Strategy favoring longs - monitor GBP/JPY bias")
                elif self.short_trades > self.long_trades * 2:
                    self.log("- Strategy favoring shorts - good for trending down markets")
                
                if self.trading_halted:
                    self.log(f"TRADING STATUS: HALTED - {self.halt_reason}")
                else:
                    self.log("TRADING STATUS: ACTIVE")
                
                self.log(f"\nCurrent parameters:")
                self.log(f"  Band level: {self.p.use_band_level}")
                self.log(f"  Volume multiplier: {self.p.min_vol_mult:.2f}")
                self.log(f"  Risk per trade: {self.p.risk_per_trade*100:.1f}%")
                self.log(f"  Touch tolerance: {self.p.touch_tolerance*100:.2f}%")
                self.log(f"  Fixed lot size: {self.p.fixed_lot_size}")
            self.log("="*60)

    def log(self, message):
        if self.p.enable_logging:
            try:
                if len(self.data) > 0:
                    dt = self.data.datetime.date(0)
                    tm = self.data.datetime.time(0)
                    timestamp = f"{dt} {tm}"
                else:
                    timestamp = "INIT"
                clean_msg = str(message).encode('ascii', 'ignore').decode('ascii')
                print(f"{timestamp} | {clean_msg}")
            except Exception:
                clean_msg = str(message).encode('ascii', 'ignore').decode('ascii')
                print(f"LOG | {clean_msg}")

