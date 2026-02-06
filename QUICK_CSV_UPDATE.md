# Quick CSV Update Guide

## 🚀 Fastest Way to Get CSV Files

### Option 1: Use Update Script (Recommended)

```bash
# Update single symbol
python update_csv_data.py update --symbol GBPJPY --source mt5 --days 30

# Update all symbols
python update_csv_data.py update --symbol GBPJPY --source mt5 --days 30
python update_csv_data.py update --symbol BTCUSD --source mt5 --days 30
python update_csv_data.py update --symbol XAUUSD --source mt5 --days 30
```

### Option 2: Copy-Paste into TraderLocker

**File:** `COPY_PASTE_TRADERLOCKER_UPDATED.py`

1. Open the file
2. Copy entire contents
3. Paste into TraderLocker script editor
4. Run script
5. Download CSV files
6. Place in project root

## 📋 Complete Script (Copy This)

```python
import pandas as pd
from datetime import datetime, timedelta
import os

SYMBOLS = ['GBPJPY', 'BTCUSD', 'XAUUSD']
TIMEFRAME = '5m'
DAYS = 30

print("="*80)
print("TRADERLOCKER DATA FETCHER")
print("="*80)

for symbol in SYMBOLS:
    print(f"\nProcessing {symbol}...")
    
    try:
        import MetaTrader5 as mt5
        
        if not mt5.initialize():
            print(f"  [ERROR] MT5 init failed: {mt5.last_error()}")
            continue
        
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"  [ERROR] Symbol {symbol} not found")
            mt5.shutdown()
            continue
        
        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=DAYS)
        
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            print(f"  [WARN] No data for {symbol}")
            mt5.shutdown()
            continue
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.rename(columns={
            'time': 'timestamp',
            'tick_volume': 'volume'
        })
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
        
        filename = f"{symbol.lower()}_5m.csv"
        df.to_csv(filename, index=False)
        
        print(f"  [OK] Saved {len(df)} bars to {filename}")
        print(f"       Range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
        
        mt5.shutdown()
        
    except ImportError:
        import requests
        api_key = os.getenv('TRADERLOCKER_API_KEY')
        if not api_key:
            print(f"  [ERROR] API key not set")
            continue
        
        url = f"https://api.traderlocker.com/v1/historical/{symbol}"
        end_date = datetime.now()
        start_date = end_date - timedelta(days=DAYS)
        
        params = {
            'apikey': api_key,
            'start': start_date.isoformat(),
            'end': end_date.isoformat(),
            'timeframe': TIMEFRAME
        }
        
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        df = pd.DataFrame(data.get('data', []))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
        
        filename = f"{symbol.lower()}_5m.csv"
        df.to_csv(filename, index=False)
        
        print(f"  [OK] Saved {len(df)} bars to {filename}")

print("\nDone! CSV files created.")
```

## ✅ After Getting CSV Files

1. **Validate:**
   ```bash
   python test_csv_loader.py
   ```

2. **Run tests:**
   ```bash
   python run_all_symbols_tests.py
   ```

3. **Run backtest with risk management:**
   ```bash
   python backtest_with_risk_management.py
   ```

## 📁 File Locations

Place CSV files in:
- `gbpjpy_5m.csv` (project root) ← **Recommended**
- `data/gbpjpy_5m.csv` (data folder)

Same for `btcusd_5m.csv` and `xauusd_5m.csv`
