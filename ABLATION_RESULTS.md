# Ablation Ladder Results

## Synthetic Data Results

| Rung | Retest | Vol_Mult | ATR_Cap | EMA | Session | Trades | Sharpe | DD |
|------|--------|----------|---------|-----|---------|--------|--------|-----|
| R0 | 1 | 1.00 | 999.0 | False | False | 75 | 4.23 | 0.05% |
| R1a | 2 | 1.00 | 999.0 | False | False | 14 | 4.28 | 0.04% |
| R1b | 3 | 1.00 | 999.0 | False | False | 3 | 7.04 | 0.00% |
| R2-1.05 | 3 | 1.05 | 999.0 | False | False | 0 | 0.00 | 0.00% |

## Detailed Metrics (Synthetic)

| Rung | Crosses | Retests | Confirms | Trades | Kill_Ratio |
|------|---------|---------|----------|--------|------------|
| R0 | 682 | 1341 | 75 | 75 | 1.00 |
| R1a | 682 | 1341 | 14 | 14 | 1.00 |
| R1b | 682 | 1341 | 3 | 3 | 1.00 |
| R2-1.05 | 682 | 1341 | 0 | 0 | 0.00 |

## Key Findings

### Primary Culprit: Volume Filter
- **Volume filter at 1.05x kills all trades** when combined with retest_count=3
- Even a 5% volume requirement is too strict for synthetic data
- Crosses and retests remain constant (682/1341) - detection works fine

### Secondary Issue: Retest Count
- **retest_count=3 reduces trades from 75 → 3** (96% reduction)
- Still produces trades, but very selective
- Quality improves (Sharpe 4.23 → 7.04)

### Event Detection: ✅ Working
- Crosses detected: 682 (consistent across all tests)
- Retests detected: 1341 (consistent across all tests)
- Kill ratio = 1.00 (all confirmations become trades when filters pass)

## Next Steps

1. **Run on Real Data** - Synthetic volume model may not match real market behavior
2. **Adjust Volume Filter** - Consider:
   - Volume percentile (60th percentile) instead of multiplier
   - Range expansion (abs(close-open) >= ATR) instead of volume
   - Lower threshold (1.02x or 1.0x baseline)
3. **Flexible Retest Count** - Use "retest_count >= 2 and <= 4" instead of exact match

## Files Created

- ✅ `run_ablation_ladder.py` - Synthetic data ablation
- ✅ `run_ablation_real_data.py` - Real data ablation (ready when data available)
- ✅ `load_real_market_data.py` - Real data loader (CSV + TraderLocker)

## To Run Real Data Ablation

```bash
# Option 1: Place CSV file
# gbpjpy_5m.csv with columns: timestamp, open, high, low, close, volume
python run_ablation_real_data.py

# Option 2: Use TraderLocker API
export TRADERLOCKER_API_KEY=your_key
python run_ablation_real_data.py
```
