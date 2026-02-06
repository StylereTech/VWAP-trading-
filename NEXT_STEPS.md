# Next Steps - Production Readiness

## ✅ Completed

1. **Ablation Ladder** - Identified volume filter as choke point
2. **Volume Filter Variants** - Implemented 4 new filter types
3. **Flexible Retest Count** - Changed from exact match to window (2-4)
4. **Truth Prints** - Validated VWAP calculation and cross detection

## 📊 Current Status

### Synthetic Data Results:
- **Baseline (flexible retest, no volume filter)**: 18 trades
- **Best Volume Filter**: Percentile L=50, p=55 → 21 trades
- **Event Detection**: ✅ Working (682 crosses, 1341 retests)

### Key Improvements:
- **Retest Count**: Flexible window (2-4) instead of exact match
- **Volume Filter**: Percentile-based (adaptive, scale-stable)
- **MIN_TRADES Gate**: Already in optimizer (prevents 0-trade configs)

## 🎯 Immediate Next Steps

### 1. Test on Real Data (REQUIRED)
**Before optimization**, run ablation ladder on real market data:

```bash
# Place CSV file: gbpjpy_5m.csv
# OR set: export TRADERLOCKER_API_KEY=your_key
python run_ablation_real_data.py
```

**Acceptance Criteria**:
- Trades ≥ 40 per 30 days
- Max drawdown reasonable
- Payoff ratio not absurd (>10 = bug)
- Win rate not suspiciously high (98% = bug)

### 2. Update PARAM_SPACE for Optimization
Once real data validates, update `vwap_qubo_tuner_production.py`:

```python
PARAM_SPACE = {
    "sigma_level": [1.5, 2.0, 2.5, 3.0],
    "retest_count": [2, 3],  # Flexible window handled in code
    "volume_filter_type": ["percentile", "impulse"],  # New
    "vol_percentile_p": [55, 60, 65, 70],  # New
    "atr_cap_mult": [2.0, 2.5, 3.0],
    "trail_pct": [0.005, 0.007, 0.010],
    "session_filter": [0, 1],
}
```

### 3. Run Optimization (After Real Data Validation)
Only after real data shows non-degenerate trades:

```bash
python run_optimization_with_progress.py
```

## 📝 Files Ready

- ✅ `run_ablation_volume_filters.py` - Volume filter variants test
- ✅ `run_ablation_real_data.py` - Real data ablation (ready)
- ✅ `load_real_market_data.py` - CSV + TraderLocker loader
- ✅ `capital_allocation_ai/vwap_control_flip_strategy.py` - Updated with new filters

## ⚠️ Do NOT Do Yet

- ❌ Run full QUBO optimization (wait for real data)
- ❌ Deploy to live trading (needs real data validation)
- ❌ Trust synthetic data results (only for logic debugging)

## ✅ What to Do Right Now

1. **Get real market data** (CSV or TraderLocker API)
2. **Run ablation on real data** - Validate filters work on real patterns
3. **Check acceptance criteria** - Ensure ≥40 trades/30 days
4. **Then optimize** - With confidence that filters are correct
