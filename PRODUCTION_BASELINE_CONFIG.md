# Production Baseline Configuration

## Locked-In Baseline (Before Real Data Tests)

Based on ablation ladder results, use this as production baseline:

```python
BASELINE_CONFIG = {
    "sigma_level": 2.0,
    "retest_min": 2,  # Flexible window (2-4)
    "retest_max": 4,
    "volume_filter_type": "percentile",
    "vol_percentile_p": 60,  # Middle of good band (55-65)
    "vol_percentile_L": 50,
    "atr_cap_mult": 3.0,  # Start loose, tighten later
    "trail_pct": 0.007,
    "session_filter": 0  # Off for baseline; turn on after validation
}
```

## Rationale

### Retest Window (2-4)
- **Before**: Exact match retest_count=3 → 3 trades
- **After**: Flexible window (2-4) → 18 trades
- **Impact**: 6x increase, less brittle

### Volume Filter: Percentile p=60
- **p=55**: More trades (21), lower Sharpe (3.46)
- **p=60**: Balanced (18 trades, Sharpe 4.86) - **CHOSEN**
- **p=65**: Fewer trades (16), higher Sharpe (11.30)
- **Why p=60**: Most stable starting point, middle of good band

### ATR Cap: 3.0
- Start loose to ensure trade generation
- Tighten after validation (2.5-3.5 range)

### Session Filter: Off
- Turn off for baseline to maximize trade generation
- Turn on only after trade count is healthy

## Updated PARAM_SPACE

```python
PARAM_SPACE = {
    "sigma_level": [1.5, 2.0, 2.5, 3.0],
    "retest_min": [1, 2],
    "retest_max": [3, 4, 5],  # Ensure retest_min < retest_max
    "volume_filter_type": ["percentile"],  # Fixed for now
    "vol_percentile_p": [55, 60, 65],
    "vol_percentile_L": [30, 50, 80],
    "atr_cap_mult": [2.5, 3.0, 3.5],
    "trail_pct": [0.005, 0.007, 0.010],
    "session_filter": [0, 1],
}
```

## Reclaim Quality Filter - Fixed

**Before**: Too strict (0 trades)
- `reclaim_atr_thresh = 0.15` → `0.05` (loosened)
- `body_ratio_thresh = 0.55` → `0.45` (loosened)
- **Scope**: Only apply after recent reclaim (within lookback)

**Status**: Not included in optimizer yet - needs real data validation first

## Next Steps

1. ✅ **Run real data ablation** - Test baseline on real OHLCV
2. ✅ **Check acceptance criteria**:
   - Trades ≥ 40 / 30 days
   - Max DD < 15-20%
   - Payoff ratio not >6-8
   - Win rate not >85-90%
3. ✅ **Then optimize** - With MIN_TRADES gate in place

## Files Updated

- ✅ `capital_allocation_ai/vwap_control_flip_strategy.py` - New volume filters + flexible retest
- ✅ `vwap_qubo_tuner_production.py` - Updated PARAM_SPACE and BASELINE_CONFIG
- ✅ `run_ablation_real_data.py` - Production baseline config ready
