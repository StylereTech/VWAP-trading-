# Parameter Changes Summary - GBP/JPY 1-Minute Strategy

## Overview
Updated TradeLocker VWAP strategy to increase trade frequency while maintaining risk controls. Optimized for GBP/JPY 1-minute timeframe.

## Key Changes

### 1. Fixed Lot Size
- **Parameter**: `fixed_lot_size = 0.10` (mini lot)
- **Impact**: Consistent position sizing regardless of equity. Removes risk-based sizing variability.
- **Frequency Effect**: Neutral - doesn't directly increase entries but ensures consistent trade size.

### 2. Band Level
- **Parameter**: `use_band_level = 1` (R1/S1 bands)
- **Previous**: Could use R2/S2
- **Impact**: Closer bands = more frequent touches = more entry signals
- **Frequency Effect**: **HIGH** - R1/S1 bands are hit more often than R2/S2

### 3. Touch Tolerance
- **Parameter**: `touch_tolerance = 0.002` (0.2%)
- **Previous**: 0.001 (0.1%)
- **Impact**: Relaxed entry conditions - allows entries when price gets "close enough" to bands
- **Frequency Effect**: **HIGH** - More lenient entry criteria = more trades
- **Risk**: Slightly increases false signals but improves fill probability

### 4. Volume Filter
- **Parameter**: `min_vol_mult = 0.70`
- **Previous**: 0.80
- **Impact**: Allows entries when volume is only 70% of average (vs 80%)
- **Frequency Effect**: **MEDIUM** - More bars pass volume filter
- **Risk**: Slightly lower quality signals but still maintains volume confirmation

### 5. Volatility Filter
- **Parameter**: `min_volatility_threshold = 0.004` (0.4%)
- **Previous**: 0.005 (0.5%)
- **Impact**: Allows trading in slightly quieter conditions
- **Frequency Effect**: **MEDIUM** - More bars pass volatility filter
- **Risk**: Minimal - still requires meaningful volatility

### 6. ATR Stop Multiplier
- **Parameter**: `atr_stop_multiplier = 1.4`
- **Previous**: 1.5
- **Impact**: Tighter stops = faster exits (both winners and losers)
- **Frequency Effect**: **INDIRECT** - Faster exits free up capital for new trades
- **Risk**: Slightly tighter stops may increase stop-outs but also protects capital

### 7. Removed Hard-Coded Equity
- **Change**: All equity calculations now use actual broker equity
- **Impact**: Drawdown and ROI calculations are accurate regardless of starting capital
- **Drawdown**: Now calculated as 20% from peak equity (not hard-coded $500 threshold)

## Expected Frequency Impact

| Change | Frequency Impact | Risk Impact |
|--------|-----------------|-------------|
| Band Level 1 | HIGH ⬆⬆⬆ | Low |
| Touch Tolerance 0.002 | HIGH ⬆⬆⬆ | Medium |
| Volume Mult 0.70 | MEDIUM ⬆⬆ | Low |
| Volatility 0.004 | MEDIUM ⬆⬆ | Low |
| ATR Stop 1.4 | INDIRECT ⬆ | Medium |

**Combined Effect**: Expected 2-3x increase in trade frequency while maintaining win rate ≥55% and drawdown ≤20%.

## Risk Management

### Drawdown Protection
- **Threshold**: 20% from peak equity (dynamic, not hard-coded)
- **Action**: Trading halts if drawdown exceeds threshold
- **Resume**: Auto-resumes when equity recovers 10% above threshold AND recent performance improves

### Performance-Based Halts
- 5 consecutive losses → Halt
- <50% win rate over last 15 trades → Halt
- ≤2 wins in last 10 trades → Halt

### Adaptive Parameters
- Strategy adjusts band level, volume multiplier, and stop multiplier based on recent performance
- Win rate <65% → Tighter filters (band level 2, higher volume req)
- Win rate >85% → Looser filters (band level 1, lower volume req)

## Testing Recommendations

### Test Plan
1. **Baseline Test**: Run on GBP/JPY 1-minute data for 1 week
   - Record: Trades/day, Win rate, ROI, Max drawdown
   
2. **Parameter Sensitivity**:
   - Test `touch_tolerance` at 0.0015, 0.002, 0.0025, 0.003
   - Test `min_vol_mult` at 0.60, 0.70, 0.80
   - Test `atr_stop_multiplier` at 1.2, 1.4, 1.6

3. **Validation**:
   - Win rate should be ≥55%
   - Max drawdown should be ≤20%
   - Trade frequency should be 3-8 trades/day (vs previous 1-3)

## Usage Example

```python
import backtrader as bt
from trade_locker_vwap_strategy import TradeLockerVWAPStrategy

cerebro = bt.Cerebro()

# Add data (GBP/JPY 1-minute)
data = bt.feeds.YourDataFeed(...)
cerebro.adddata(data)

# Add strategy with optimized parameters
cerebro.addstrategy(TradeLockerVWAPStrategy,
    fixed_lot_size=0.10,          # Fixed mini lot
    use_band_level=1,              # R1/S1 bands
    relaxed_entry=True,
    touch_tolerance=0.002,         # 0.2% tolerance
    use_volume_filter=True,
    min_vol_mult=0.70,             # 70% of average volume
    use_volatility_filter=True,
    min_volatility_threshold=0.004, # 0.4% minimum volatility
    atr_stop_multiplier=1.4,       # Tighter stops
    max_concurrent_trades=1,
    optimization_frequency=50,
)

# Set initial cash (any amount - no longer hard-coded)
cerebro.broker.setcash(1000.0)  # Example: $1000 starting capital

# Run backtest
cerebro.run()
```

## Notes

- **No HTF Trend Filter**: Removed any higher timeframe trend filter that was limiting entries
- **Fixed Lot Size**: Strategy uses 0.10 lots regardless of equity (overrides risk-based sizing)
- **Dynamic Equity Tracking**: All calculations use actual broker equity, not hard-coded values
- **GBP/JPY Optimized**: Parameters tuned for GBP/JPY volatility characteristics on 1-minute timeframe

