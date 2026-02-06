# Data Acquisition Checklist

## ✅ Engineering Side: Complete
- ✅ CSV loader hardened (handles TradingView, broker exports)
- ✅ Scripts fail fast with clear messages
- ✅ `data/` directory created
- ✅ CSV presence test script ready
- ✅ Zero CSVs detected (waiting for data)

## 📥 Data Acquisition: Next Steps

### Route A: Broker Platform Export (Recommended)
**Best because:** Volume/tick_volume included, timestamps match your trading feed

1. Open your broker platform (MT4/MT5/TradingView broker integration)
2. Select **GBPJPY**
3. Set timeframe to **M5** (5 minutes)
4. Export last **30-45 days**
5. Save as CSV
6. Place in: `gbpjpy_5m.csv` (project root) or `data/gbpjpy_5m.csv`

### Route B: TradingView Export
**Works if:** Your plan allows enough history export

1. Open GBPJPY chart
2. Set to **5m** timeframe
3. Export chart data → Download CSV
4. **Note:** If TradingView limits bars, export what's available (even 10-14 days works for validation)
5. Place in: `gbpjpy_5m.csv` (project root) or `data/gbpjpy_5m.csv`

## 🔧 Edge Case Helpers

### If CSV Has No Volume Column

**Option 1 (Preferred):** Use tick_volume
- Many FX exports include `tick_volume` instead of `volume`
- Your loader already supports this (automatically maps `tick_volume` → `volume`)

**Option 2:** Add proxy volume (validation only)
```bash
python add_proxy_volume.py your_file.csv --method body
```
- Creates `volume = abs(close - open)` (body size)
- Use `--method range` for `volume = high - low` (range size)
- **Warning:** Only for validation! Switch to real tick_volume before optimization.

### If Timestamps Are Wrong Timezone

If your CSV has local timezone timestamps (not UTC):

```bash
python convert_timezone.py your_file.csv --output gbpjpy_5m.csv --from "America/New_York" --to UTC
```

Common timezones:
- `America/New_York` (EST/EDT)
- `Europe/London` (GMT/BST)
- `Asia/Tokyo` (JST)
- `UTC` (if already UTC)

## ✅ Verification Steps

### Step 1: Test CSV Loads
```bash
python test_csv_loader.py
```

**Expected output:**
```
[OK] Found CSV: gbpjpy_5m.csv
Rows: 8640
Range: 2026-01-01 00:00:00+00:00 -> 2026-01-31 23:55:00+00:00
[SUCCESS] CSV is ready for backtesting!
```

### Step 2: Run Tests
```bash
python run_ablation_real_data.py --test A --symbol GBPJPY --days 30
python run_ablation_real_data.py --test B --symbol GBPJPY --days 30
python run_ablation_real_data.py --test C --symbol GBPJPY --days 30
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

## ⚠️ Common Gotchas

### File Naming
- Windows often adds `(1)` suffix → Rename to exactly `gbpjpy_5m.csv`
- Case-sensitive on some systems → Use lowercase: `gbpjpy_5m.csv`

### Timezone Issues
- If VWAP reset/session rules seem off, timestamps might be wrong timezone
- Run everything in UTC (loader defaults to UTC)
- If your CSV is local time, use `convert_timezone.py` first

### Volume Column Missing
- Check if column is named `tick_volume` (loader handles this automatically)
- If truly missing, use `add_proxy_volume.py` for validation only
- **Important:** Get real tick_volume before final optimization

## 🎯 Success Criteria

You're ready when:
- ✅ `test_csv_loader.py` shows `[SUCCESS]`
- ✅ Rows ≈ 8,000+ (for 30 days @ 5m)
- ✅ Timestamps sorted ascending
- ✅ Volume/tick_volume column present and non-zero

## 📋 Quick Reference

**File locations:**
- `gbpjpy_5m.csv` (project root) ← **Recommended**
- `data/gbpjpy_5m.csv` (data folder)

**Required columns:**
- `timestamp` (or `time`, `date`, `datetime`)
- `open`, `high`, `low`, `close`
- `volume` (or `vol`, `tick_volume`)

**Helper scripts:**
- `test_csv_loader.py` - Verify CSV loads correctly
- `add_proxy_volume.py` - Add proxy volume if missing
- `convert_timezone.py` - Convert timestamps to UTC

**Ready to proceed once CSV is in place!** 🚀
