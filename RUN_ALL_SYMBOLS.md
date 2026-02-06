# Run All Symbols Backtest - Guide

## Quick Start

### Step 1: Check Available Data
```bash
python check_data_files.py
```

This will show:
- Which symbols have CSV files ready
- Which symbols are missing data
- Whether TraderLocker API is configured

### Step 2: Run All Tests
```bash
python run_all_symbols_tests.py
```

This will:
- Run tests A, B, C for each symbol that has data
- Output parseable results table
- Show acceptance criteria for each symbol/test
- Provide summary of which symbols are ready for optimization

## Required CSV Files

Place CSV files in project root or `data/` folder:

| Symbol | Filename Options |
|--------|------------------|
| GBPJPY | `gbpjpy_5m.csv` or `GBPJPY_5m.csv` |
| BTCUSD | `btcusd_5m.csv`, `BTCUSD_5m.csv`, `btc_5m.csv`, or `BTC_5m.csv` |
| XAUUSD | `xauusd_5m.csv`, `XAUUSD_5m.csv`, `gold_5m.csv`, `xau_5m.csv`, or `XAU_5m.csv` |

## CSV Format

Each CSV needs these columns (case-insensitive):
- `timestamp` (or `time`, `date`, `datetime`)
- `open`, `high`, `low`, `close`
- `volume` (or `vol`, `tick_volume`)

## Test Configurations

Each symbol will be tested with 3 configurations:

### Test A - Baseline
- Volume Filter: percentile (p=60, L=50)
- Retest Window: 2-4
- ATR Cap: 3.0
- Session Filter: OFF

### Test B - Higher Frequency
- Volume Filter: percentile (p=55, L=50)
- Retest Window: 1-4
- ATR Cap: 3.5
- Session Filter: OFF

### Test C - Higher Quality
- Volume Filter: percentile (p=65, L=50)
- Retest Window: 2-4
- ATR Cap: 2.5
- Session Filter: ON

## Output Format

The script outputs:

1. **Parseable Results Table:**
   ```
   symbol|test|rung|retest_min-retest_max|vol_filter(p,L)|atr_cap|session|trades|sharpe|dd|total_bars
   ```

2. **Acceptance Criteria Check:**
   - Trades >= 40 per 30 days
   - Sharpe < 8 (to avoid backtest bugs)
   - Max DD < 20%

3. **Detailed Summary:**
   - Bars loaded
   - Timestamp range
   - Key metrics (trades, Sharpe, DD) per test

## Example Output

```
================================================================================
SUMMARY
================================================================================

GBPJPY:
  Test A: Loaded bars: 8640
           Range: 2026-01-01 00:00:00+00:00 -> 2026-01-31 23:55:00+00:00
           Trades: 45, Sharpe: 2.34, DD: 5.2%
  Test B: ...
  Test C: ...

ACCEPTANCE CRITERIA CHECK
================================================================================

GBPJPY:
  Test A:
    Trades >= 40: [PASS] (45)
    Sharpe < 8: [PASS] (2.34)
    DD < 20%: [PASS] (5.2%)
    Overall: [PASS]
```

## Troubleshooting

### No CSV Files Found
- Export data from TradingView or your broker
- Place files in project root or `data/` folder
- Verify filename matches exactly (case-sensitive on some systems)

### TraderLocker API
- Set environment variable: `export TRADERLOCKER_API_KEY=your_key`
- Script will attempt to fetch data for missing symbols
- Ensure API supports historical OHLCV (not just quotes)

### Test Fails for Specific Symbol
- Check CSV file loads correctly: `python test_csv_loader.py`
- Verify timestamp format and timezone
- Ensure volume column exists (or use `add_proxy_volume.py` for validation)

## Next Steps After Tests

Once tests complete:
1. Review parseable results table
2. Check which symbols pass acceptance criteria
3. Proceed with optimization for passing symbols
4. Adjust filters for symbols that fail (loosen retest_min, percentile p, or ATR cap)

## Individual Symbol Testing

To test a single symbol:
```bash
python run_ablation_real_data.py --test A --symbol GBPJPY --days 30
python run_ablation_real_data.py --test B --symbol GBPJPY --days 30
python run_ablation_real_data.py --test C --symbol GBPJPY --days 30
```

Replace `GBPJPY` with `BTCUSD` or `XAUUSD` as needed.
