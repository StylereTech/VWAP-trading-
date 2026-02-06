# Data Setup Guide - Quick Reference

## Current Status
✅ Scripts hardened and ready  
✅ CSV loader robust (handles TradingView, broker exports)  
⏳ **Waiting for real market data**

## Quick Test (Before Running Full Tests)

Run this to verify your CSV is ready:
```bash
python test_csv_loader.py
```

**Pass conditions:**
- ✅ CSV file found in expected location
- ✅ Rows > 1000 (roughly 30 days @ 5m = ~8,640 bars)
- ✅ Timestamps sorted (ascending)
- ✅ Volume non-zero

## Where to Place Your CSV

Place your CSV file in one of these exact locations:

1. **`gbpjpy_5m.csv`** (project root) ← **Recommended**
2. **`data/gbpjpy_5m.csv`** (if you create a `data/` folder)

## CSV Format Requirements

### Required Columns (case-insensitive, any order):
- `timestamp` (or `time`, `date`, `datetime`)
- `open`
- `high`
- `low`
- `close`
- `volume` (or `vol`, `tick_volume`)

### Timestamp Format Examples (all work):
- `2026-01-02 13:35:00`
- `2026-01-02T13:35:00Z`
- `01/02/2026 13:35`
- `2026-01-02 13:35:00+00:00`

**Note:** If timestamps are timezone-naive, the loader will localize to UTC automatically.

## How to Get the Data

### Option 1: TradingView Export (Easiest)
1. Open GBPJPY chart
2. Set timeframe to **5 minutes**
3. Use "Export chart data" (available in most plans)
4. Ensure export includes **volume**
5. Save as `gbpjpy_5m.csv` in project root

### Option 2: Broker Platform Export
1. Open your broker's terminal/history center
2. Select GBPJPY
3. Export **5-minute bars** for last **30-45 days**
4. Ensure columns: timestamp, open, high, low, close, volume
5. Save as `gbpjpy_5m.csv` in project root

### Option 3: TraderLocker API
1. Set environment variable: `TRADERLOCKER_API_KEY=your_key`
2. Script will automatically fetch data
3. Ensure API endpoint supports historical OHLCV (not just quotes)

## Once You Have Data

### Step 1: Verify CSV Loads
```bash
python test_csv_loader.py
```

Expected output:
```
[OK] Found CSV: gbpjpy_5m.csv
Rows: 8640
Range: 2026-01-01 00:00:00+00:00 -> 2026-01-31 23:55:00+00:00
[SUCCESS] CSV is ready for backtesting!
```

### Step 2: Run the 3 Tests
```bash
python run_ablation_real_data.py --test A --symbol GBPJPY --days 30
python run_ablation_real_data.py --test B --symbol GBPJPY --days 30
python run_ablation_real_data.py --test C --symbol GBPJPY --days 30
```

Or run all at once:
```bash
python run_real_data_tests.py
```

### Step 3: Paste Results
For each test, paste:
1. **Parseable table line:**
   ```
   rung|retest_min-retest_max|vol_filter(p,L)|atr_cap|session|trades|sharpe|dd|total_bars
   ```
2. **Data sanity line:**
   ```
   Loaded bars: XXXX
   Range: YYYY-MM-DD HH:MM:SS -> YYYY-MM-DD HH:MM:SS
   ```

## Troubleshooting

### "CSV missing columns"
- Check column names match required format (case-insensitive)
- Ensure volume column exists (may be named `vol` or `tick_volume`)

### "Failed to parse timestamps"
- Check timestamp format matches examples above
- Ensure no empty timestamp cells

### "No CSV file found"
- Verify file is named exactly `gbpjpy_5m.csv` (lowercase)
- Check file is in project root or `data/` folder
- Run `python test_csv_loader.py` to see exact paths checked

## Next Steps After Data is Ready

Once you paste the 3 test results, I will:
1. ✅ Analyze which configuration produces best baseline
2. ✅ Recommend final PARAM_SPACE ranges for optimization
3. ✅ Determine if session filter should be included
4. ✅ Set up two-stage optimizer (fast screen → full validation)

**Ready when you are!** 🚀
