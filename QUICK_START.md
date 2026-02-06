# Quick Start - Real Data Tests

## 🎯 Goal
Run 3 real-data tests (A, B, C) to validate strategy before optimization.

## ⚡ Fastest Path

### 1. Get Your CSV Data
- Export 30 days of GBPJPY 5-minute bars from TradingView or your broker
- Save as: **`gbpjpy_5m.csv`** in project root

### 2. Test CSV Loads
```bash
python test_csv_loader.py
```
Should show: `[SUCCESS] CSV is ready for backtesting!`

### 3. Run Tests
```bash
python run_ablation_real_data.py --test A --symbol GBPJPY --days 30
python run_ablation_real_data.py --test B --symbol GBPJPY --days 30
python run_ablation_real_data.py --test C --symbol GBPJPY --days 30
```

### 4. Paste Results
Copy the parseable table lines from each test output and paste here.

## 📋 CSV Format (Quick Check)

Your CSV needs these columns (any order, case-insensitive):
- `timestamp` (or `time`, `date`, `datetime`)
- `open`, `high`, `low`, `close`
- `volume` (or `vol`, `tick_volume`)

Example:
```csv
timestamp,open,high,low,close,volume
2026-01-01 00:00:00,150.123,150.456,150.100,150.345,1250
2026-01-01 00:05:00,150.345,150.500,150.200,150.400,1180
```

## ✅ That's It!

Once you have the CSV and run the tests, paste the results and we'll finalize the optimization setup.
