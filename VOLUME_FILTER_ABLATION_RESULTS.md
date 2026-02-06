# Volume Filter Ablation Results

## Summary

**Baseline**: Flexible retest_count (2-4) produces **18 trades** (vs 3 with exact match)

**Best Volume Filter**: **Percentile L=50, p=55** - **21 trades** (increases trade count)

## Results Table

| Rung | Vol_Filter | Vol_Param | Trades | Confirms | Sharpe | DD |
|------|------------|-----------|--------|----------|--------|-----|
| V0 | none | N/A | 18 | 18 | 4.86 | 0.03% |
| V1 | multiplier | 1.00 | 18 | 18 | 4.86 | 0.03% |
| V2 | percentile | L=50,p=55 | **21** | 21 | 3.46 | 0.03% |
| V3 | percentile | L=50,p=65 | 16 | 16 | **11.30** | 0.01% |
| V4 | impulse | body>=0.8*ATR | 18 | 18 | 4.86 | 0.03% |
| V5 | reclaim_quality | dist>=0.15*ATR,body>=0.55 | 0 | 0 | 0.00 | 0.00% |

## Key Findings

### 1. Flexible Retest Count Works ✅
- **Before**: Exact match retest_count=3 → 3 trades
- **After**: Flexible window (2-4) → 18 trades
- **Impact**: 6x increase in trade frequency

### 2. Percentile Filter is Best ✅
- **V2 (p=55)**: 21 trades, Sharpe 3.46 - **Best trade count**
- **V3 (p=65)**: 16 trades, Sharpe 11.30 - **Best quality**
- **Advantage**: Adaptive, scale-stable, symbol-agnostic

### 3. Reclaim Quality Too Strict ❌
- **V5**: 0 trades - Filter is too restrictive
- **Issue**: Requires both distance AND body ratio, kills all signals

### 4. Impulse Filter Neutral
- **V4**: 18 trades - Same as baseline
- **Use case**: Good fallback if percentile doesn't work on real data

## Recommendations

### For Production:
1. **Use Percentile Filter** (L=50, p=55-65)
   - Start with p=55 for more trades
   - Optimize p between 55-70 based on real data
   
2. **Keep Flexible Retest Window** (2-4)
   - Less brittle than exact match
   - Preserves "third poke" thesis

3. **Test on Real Data**
   - Synthetic data may not reflect real volume patterns
   - Validate percentile thresholds on real market data

### Next Steps:
1. ✅ Run ablation on real data (when available)
2. ✅ Update PARAM_SPACE to include percentile parameters
3. ✅ Run optimization only after real data shows ≥40 trades/30 days

## Implementation Status

- ✅ Volume filter variants implemented
- ✅ Flexible retest_count implemented
- ✅ Ablation ladder completed
- ⏳ Real data testing (pending data)
- ⏳ Parameter optimization (pending real data validation)
