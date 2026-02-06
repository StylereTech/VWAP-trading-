# TraderLocker Script - Copy/Paste Guide

## Quick Copy/Paste Scripts

I've created two scripts for TraderLocker:

### Option 1: Full Script (`traderlocker_fetch_data.py`)
- Handles both MT5 and REST API
- More error handling
- Better logging

### Option 2: Simple Script (`traderlocker_simple_fetch.py`)
- Minimal code
- Easier to customize
- Two API options (MT5 or REST)

## How to Use

### Step 1: Choose Your Script

**If TraderLocker uses MetaTrader5:**
- Use the MT5 section from `traderlocker_simple_fetch.py`
- Or use `traderlocker_fetch_data.py` (it auto-detects MT5)

**If TraderLocker uses REST API:**
- Use the REST API section from `traderlocker_simple_fetch.py`
- Set your API key: `TRADERLOCKER_API_KEY=your_key`

### Step 2: Copy Script into TraderLocker

1. Open TraderLocker's script editor
2. Copy the entire script from `traderlocker_simple_fetch.py`
3. Paste into TraderLocker
4. Adjust configuration at the top:
   ```python
   SYMBOLS = ['GBPJPY', 'BTCUSD', 'XAUUSD']  # Your symbols
   TIMEFRAME = '5m'  # 5-minute bars
   DAYS = 30  # Days of history
   ```

### Step 3: Adjust API Calls

**For MT5:**
- Script uses `mt5.copy_rates_range()`
- Adjust symbol names if your broker uses different names
- Example: BTCUSD might be 'BTCUSD' or 'BTC/USD' depending on broker

**For REST API:**
- Update the API endpoint URL
- Adjust request parameters based on TraderLocker API docs
- Set API key in environment or hardcode (not recommended for production)

### Step 4: Run Script

Execute the script in TraderLocker. It will:
1. Fetch historical OHLCV data for each symbol
2. Save as CSV files: `gbpjpy_5m.csv`, `btcusd_5m.csv`, `xauusd_5m.csv`
3. Files will be saved in the script's working directory

### Step 5: Download CSV Files

After script completes:
1. Download the CSV files from TraderLocker
2. Place them in your project root or `data/` folder
3. Run: `python test_csv_loader.py` to verify

## Common TraderLocker API Patterns

### Pattern 1: MT5/MetaTrader5
```python
import MetaTrader5 as mt5
mt5.initialize()
rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_date, end_date)
```

### Pattern 2: REST API
```python
import requests
response = requests.get(f"https://api.traderlocker.com/v1/historical/{symbol}", 
                       params={'apikey': key, 'start': start, 'end': end})
```

### Pattern 3: TraderLocker SDK
```python
from traderlocker import TraderLocker
client = TraderLocker(api_key='your_key')
data = client.get_historical_data(symbol, timeframe='5m', days=30)
```

## Troubleshooting

### "MT5 initialization failed"
- Check if MetaTrader5 is installed
- Verify broker connection is active
- Check symbol names match your broker's format

### "API key required"
- Set environment variable: `TRADERLOCKER_API_KEY=your_key`
- Or hardcode in script (temporary, for testing only)

### "Symbol not found"
- Check symbol names match your broker's format
- Some brokers use: 'BTC/USD', 'XAU/USD', 'GBP/JPY'
- Adjust symbol mapping in script

### "No data returned"
- Check date range (may be outside available history)
- Verify timeframe is supported (5m, 15m, 1h, etc.)
- Check broker's data availability

## Expected Output

After running, you should see:
```
Processing GBPJPY...
[OK] GBPJPY: 8640 bars saved to gbpjpy_5m.csv
Processing BTCUSD...
[OK] BTCUSD: 8640 bars saved to btcusd_5m.csv
Processing XAUUSD...
[OK] XAUUSD: 8640 bars saved to xauusd_5m.csv

Done! CSV files created in current directory.
```

## CSV Format

The script creates CSV files with this format:
```csv
timestamp,open,high,low,close,volume
2026-01-01 00:00:00,150.123,150.456,150.100,150.345,1250
2026-01-01 00:05:00,150.345,150.500,150.200,150.400,1180
```

This matches exactly what our loader expects!

## Next Steps

1. Run script in TraderLocker
2. Download CSV files
3. Place in project root
4. Run: `python test_csv_loader.py`
5. Run: `python run_all_symbols_tests.py`
