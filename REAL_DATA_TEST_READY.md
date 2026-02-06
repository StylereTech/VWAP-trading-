# Real Data Test Scripts - Ready

## Status: ✅ Scripts Ready, Awaiting Real Data

The real data test infrastructure is complete and ready to run. You need to provide real market data before executing the tests.

## What's Ready

### 1. Test Script: `run_ablation_real_data.py`
- ✅ Supports command-line arguments for all 3 test configurations
- ✅ Test presets (A, B, C) configured correctly
- ✅ Proper backtest function with metrics calculation
- ✅ Parseable output format as requested
- ✅ Acceptance criteria checking

### 2. Strategy Updates
- ✅ Flexible retest window (`retest_min`, `retest_max`) support
- ✅ Volume filter types (percentile, impulse, reclaim_quality)
- ✅ All filters properly integrated

### 3. Convenience Script: `run_real_data_tests.py`
- ✅ Runs all 3 tests sequentially
- ✅ Captures output for analysis

## How to Run

### Option 1: Run Individual Tests

**Test A - Baseline:**
```bash
python run_ablation_real_data.py --test A --symbol GBPJPY --days 30
```

**Test B - Higher Frequency:**
```bash
python run_ablation_real_data.py --test B --symbol GBPJPY --days 30
```

**Test C - Higher Quality:**
```bash
python run_ablation_real_data.py --test C --symbol GBPJPY --days 30
```

### Option 2: Run All Tests at Once
```bash
python run_real_data_tests.py
```

## Data Requirements

You need to provide real OHLCV data in one of these formats:

### Option 1: CSV File
Place a CSV file with columns: `timestamp`, `open`, `high`, `low`, `close`, `volume`

**Accepted file locations:**
- `gbpjpy_5m.csv` (in project root)
- `GBPJPY_5m.csv` (in project root)
- `data/gbpjpy_5m.csv`
- `data/GBPJPY_5m.csv`

**CSV Format Example:**
```csv
timestamp,open,high,low,close,volume
2024-01-01 00:00:00,150.123,150.456,150.100,150.345,1250
2024-01-01 00:05:00,150.345,150.500,150.200,150.400,1180
...
```

### Option 2: TraderLocker API
Set environment variable:
```bash
export TRADERLOCKER_API_KEY=your_api_key_here
```

Then the script will automatically fetch data from TraderLocker.

## Expected Output Format

Each test will output:

1. **Configuration Summary** - Shows the test parameters
2. **Results Table** - Key metrics (trades, Sharpe, DD, etc.)
3. **Acceptance Criteria Check** - Pass/Fail for each criterion
4. **Parseable Table** - Pipe-delimited format:
   ```
   rung|retest_min-retest_max|vol_filter(p,L)|atr_cap|session|trades|sharpe|dd|total_bars
   ```

## Test Configurations

### Test A - Baseline (Production Target)
- Volume Filter: percentile (p=60, L=50)
- Retest Window: 2-4
- ATR Cap: 3.0
- Session Filter: OFF

### Test B - Higher Frequency (If A < 40 trades)
- Volume Filter: percentile (p=55, L=50)
- Retest Window: 1-4
- ATR Cap: 3.5
- Session Filter: OFF

### Test C - Higher Quality (If A is noisy/high DD)
- Volume Filter: percentile (p=65, L=50)
- Retest Window: 2-4
- ATR Cap: 2.5
- Session Filter: ON

## Acceptance Criteria

Each test must meet:
- ✅ Trades ≥ 40 / 30 days
- ✅ Sharpe < 8 (to avoid backtest bugs)
- ✅ Max DD < 20%
- ✅ Win Rate: 40-70%
- ✅ Payoff Ratio: 0.8-2.5

## Next Steps

1. **Obtain real data** (CSV or TraderLocker API)
2. **Run Test A** first
3. **If Test A fails** (trades < 40), run Test B
4. **If Test A passes but DD is high**, run Test C
5. **Paste all results** for analysis and parameter space finalization

## Notes

- Synthetic data is NOT suitable for these tests (only for logic debugging)
- Real data must be 5-minute bars, minimum 30 days
- Script will automatically handle data preparation (VWAP calculation, indicators, etc.)
