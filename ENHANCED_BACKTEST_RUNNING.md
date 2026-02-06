# Enhanced Backtest - Currently Running

## ✅ Status

**Backtest**: Running in background
**File**: `backtest_enhanced_fast.py`
**Duration**: Estimated 15-25 minutes for all 3 symbols

## 🎯 What's Being Tested

### Enhanced Strategy Features:
1. **Stretched Targets** - Winners run to opposite band (2.5-3:1 R:R target)
2. **Trend Alignment** - Only trade with trend (EMA-90 filter)
3. **FVG Fill** - Fair Value Gap confirmation required
4. **Tighter ATR Cap** - 2.0x (was 2.5x) to avoid extreme volatility
5. **Volume Imbalance** - 20-30% above average required
6. **Breakeven After 1R** - Locks in profits
7. **Enhanced Trailing** - 0.7% trail after profit threshold

### Symbol-Specific Settings:

#### GBP/JPY (FOREX):
- Band K: 2.0σ (tighter)
- Volume Mult: 1.5x (stronger requirement)
- R:R Target: 2.5:1
- ATR Cap: 2.0x

#### BTC/USD (CRYPTO):
- Band K: 2.5σ (wider for volatility)
- Volume Mult: 1.6x (30% threshold)
- R:R Target: 3.0:1 (higher for crypto trends)
- ATR Cap: 2.0x

#### Gold/XAUUSD (METAL):
- Band K: 1.9σ
- EMA-200 Filter: Enabled (long-term trend)
- R:R Target: 2.5:1
- Volume Mult: 1.3x

## 📊 Expected Improvements

### Comparison to Original Strategy:

| Metric | Original | Enhanced (Target) |
|--------|----------|-------------------|
| Win Rate | 30-33% | 35-45% |
| Return | -1% to -4% | +2% to +5% |
| Sharpe | -7 to -8 | +1 to +3 |
| R:R | ~1:1 | 2.5:1 to 3:1 |
| Trades/Day | 2-3 | 1-2 (more selective) |

### Key Changes:
- **R:R Improvement**: Winners now run to opposite band instead of exiting at VWAP
- **Win Rate Boost**: Stronger filters (trend, FVG, volume) improve entry quality
- **Risk Management**: Breakeven after 1R prevents winners from turning into losers

## ⏱️ Progress

The backtest processes:
- **17,280 bars per symbol** (60 days × 288 bars/day)
- **Enhanced calculations** (FVG, RSI divergence, trend alignment)
- **Chunked processing** (1000 bars at a time for efficiency)

**Estimated Time**:
- GBP/JPY: ~5-8 minutes
- BTC/USD: ~5-8 minutes  
- Gold/USD: ~5-8 minutes
- **Total**: ~15-25 minutes

## 📝 What to Look For

### Success Metrics:
1. **Win Rate**: Should improve to 35-45% (from 30-33%)
2. **R:R Ratio**: Should average 2.5:1 to 3:1 (from ~1:1)
3. **Returns**: Should flip positive (+2% to +5%)
4. **Sharpe Ratio**: Should improve to positive (from -7 to -8)
5. **Trade Frequency**: May decrease slightly (1-2/day vs 2-3/day)

### If Results Are Still Negative:
- May need to loosen filters slightly (reduce nth_retest from 3 to 2)
- May need to adjust R:R targets
- May need further parameter optimization

## 🔍 Monitoring

Check progress by looking at:
- Terminal output (progress percentages)
- Results will print when complete
- Summary table will show all metrics

## ✅ Next Steps After Completion

1. **Review Results**: Compare enhanced vs original
2. **Analyze R:R**: Check if average R:R meets 2.5:1+ target
3. **Check Win Rate**: Verify improvement to 35-45%
4. **Optimize Further**: If needed, adjust parameters based on results
5. **Run Quantum Optimization**: Use results to fine-tune parameters

---

**Status**: ⏳ Running in background - Results will appear when complete!
