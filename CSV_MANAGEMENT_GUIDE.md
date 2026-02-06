# CSV Data Management Guide

## Quick Reference

### Update CSV File (Fetch Latest Data)

**From MetaTrader5:**
```bash
python update_csv_data.py update --symbol GBPJPY --source mt5 --days 30
```

**From TraderLocker API:**
```bash
python update_csv_data.py update --symbol GBPJPY --source api --days 30 --api-key YOUR_KEY
```

**Auto-detect source:**
```bash
python update_csv_data.py update --symbol GBPJPY --days 30
```

### Validate CSV File

```bash
python update_csv_data.py validate --file gbpjpy_5m.csv
```

### List Available CSV Files

```bash
python update_csv_data.py list
```

### Append to Existing File (Update with Latest Data)

```bash
python update_csv_data.py update --symbol GBPJPY --append --days 7
```

## Usage Examples

### Example 1: Create New CSV for GBPJPY

```bash
python update_csv_data.py update --symbol GBPJPY --source mt5 --days 30
```

This will:
- Fetch 30 days of 5-minute bars for GBPJPY
- Save to `gbpjpy_5m.csv` (or `data/gbpjpy_5m.csv` if data/ folder exists)
- Validate the file after creation

### Example 2: Update Existing CSV (Append Latest Data)

```bash
python update_csv_data.py update --symbol GBPJPY --append --days 7
```

This will:
- Load existing `gbpjpy_5m.csv`
- Fetch last 7 days of new data
- Merge and remove duplicates
- Save updated file

### Example 3: Update All Symbols

```bash
python update_csv_data.py update --symbol GBPJPY --days 30
python update_csv_data.py update --symbol BTCUSD --days 30
python update_csv_data.py update --symbol XAUUSD --days 30
```

### Example 4: Validate Before Running Tests

```bash
python update_csv_data.py validate --file gbpjpy_5m.csv
```

## Command-Line Options

### Update Action

```
python update_csv_data.py update [OPTIONS]

Required:
  --symbol SYMBOL      Symbol to update (GBPJPY, BTCUSD, XAUUSD)

Optional:
  --file PATH          CSV file path (default: {symbol}_5m.csv)
  --days N             Days of history (default: 30)
  --timeframe TF       Timeframe: 5m, 15m, 1h, 1d (default: 5m)
  --source SOURCE      Data source: mt5, api, auto (default: auto)
  --api-key KEY        API key for REST API
  --append             Append to existing file instead of overwriting
```

### Validate Action

```
python update_csv_data.py validate --file PATH
```

### List Action

```
python update_csv_data.py list
```

## Data Sources

### MetaTrader5 (MT5)

**Requirements:**
- MetaTrader5 terminal installed and running
- `pip install MetaTrader5`
- Symbol available in your broker's MT5

**Usage:**
```bash
python update_csv_data.py update --symbol GBPJPY --source mt5
```

### TraderLocker REST API

**Requirements:**
- API key from TraderLocker
- `pip install requests`
- Set `TRADERLOCKER_API_KEY` environment variable OR pass `--api-key`

**Usage:**
```bash
# Option 1: Environment variable
export TRADERLOCKER_API_KEY=your_key
python update_csv_data.py update --symbol GBPJPY --source api

# Option 2: Command-line argument
python update_csv_data.py update --symbol GBPJPY --source api --api-key your_key
```

### Auto-Detect

**Usage:**
```bash
python update_csv_data.py update --symbol GBPJPY
```

**Behavior:**
- Tries MT5 first (if available)
- Falls back to API (if MT5 not available)
- Raises error if neither available

## File Locations

The script looks for CSV files in these locations (in order):
1. `{symbol.lower()}_5m.csv` (project root)
2. `data/{symbol.lower()}_5m.csv` (data folder)

When creating new files, it uses the first location that exists, or project root if data/ doesn't exist.

## Validation Checks

The validation checks:
- ✅ File exists and is readable
- ✅ Required columns present (timestamp, open, high, low, close, volume)
- ✅ Timestamps are sorted (ascending)
- ✅ No duplicate timestamps
- ✅ No null values
- ✅ Price and volume ranges are reasonable

## Troubleshooting

### "MT5 initialization failed"
- Make sure MetaTrader5 terminal is running
- Check that you're logged into your broker account
- Verify symbol is available in MT5

### "API key required"
- Set environment variable: `export TRADERLOCKER_API_KEY=your_key`
- Or pass `--api-key` argument

### "No data returned"
- Check date range (may be outside available history)
- Verify symbol name matches your broker's format
- Check timeframe is supported

### "File not found" (on validate)
- Check file path is correct
- Use `list` action to see available files

## Integration with Backtesting

After updating CSV files:

1. **Validate:**
   ```bash
   python update_csv_data.py validate --file gbpjpy_5m.csv
   ```

2. **Test loader:**
   ```bash
   python test_csv_loader.py
   ```

3. **Run tests:**
   ```bash
   python run_all_symbols_tests.py
   ```

## Scheduled Updates

You can schedule regular updates using cron (Linux/Mac) or Task Scheduler (Windows):

**Daily update (Linux/Mac):**
```bash
# Add to crontab: 0 2 * * * cd /path/to/project && python update_csv_data.py update --symbol GBPJPY --append --days 1
```

**Daily update (Windows Task Scheduler):**
- Create task to run: `python update_csv_data.py update --symbol GBPJPY --append --days 1`
- Schedule daily at 2 AM

This keeps your CSV files up-to-date automatically.
