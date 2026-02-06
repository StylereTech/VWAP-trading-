"""
================================================================================
COPY THIS ENTIRE SCRIPT INTO TRADERLOCKER SCRIPT EDITOR
================================================================================
This script fetches historical OHLCV data and saves as CSV files
for GBPJPY, BTCUSD, and XAUUSD (5-minute bars, 30 days)
================================================================================
"""

import pandas as pd
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURATION - Adjust if needed
# ============================================================================
SYMBOLS = ['GBPJPY', 'BTCUSD', 'XAUUSD']
TIMEFRAME = '5m'
DAYS = 30

# ============================================================================
# SCRIPT STARTS HERE - Copy everything below
# ============================================================================

print("="*80)
print("TRADERLOCKER DATA FETCHER")
print("="*80)
print(f"Symbols: {', '.join(SYMBOLS)}")
print(f"Timeframe: {TIMEFRAME}")
print(f"Days: {DAYS}")
print("="*80)

for symbol in SYMBOLS:
    print(f"\nProcessing {symbol}...")
    
    try:
        # ================================================================
        # OPTION 1: MetaTrader5 (MT5) - Most common
        # ================================================================
        import MetaTrader5 as mt5
        
        # Initialize MT5
        if not mt5.initialize():
            print(f"  [ERROR] MT5 initialization failed: {mt5.last_error()}")
            continue
        
        # Check if symbol exists
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"  [ERROR] Symbol {symbol} not found in MT5")
            mt5.shutdown()
            continue
        
        # Enable symbol if not available
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                print(f"  [ERROR] Failed to enable symbol {symbol}")
                mt5.shutdown()
                continue
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=DAYS)
        
        print(f"  Fetching data from {start_date} to {end_date}...")
        
        # Fetch 5-minute bars
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            print(f"  [WARN] No data returned for {symbol}")
            mt5.shutdown()
            continue
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Rename columns to match our format
        df = df.rename(columns={
            'time': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume'  # MT5 uses tick_volume
        })
        
        # Select required columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Sort by timestamp and remove duplicates
        df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
        
        # Save to CSV
        filename = f"{symbol.lower()}_5m.csv"
        df.to_csv(filename, index=False)
        
        print(f"  [OK] Saved {len(df)} bars to {filename}")
        print(f"       Range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
        
        mt5.shutdown()
        
    except ImportError:
        # ================================================================
        # OPTION 2: REST API (if TraderLocker uses REST API)
        # ================================================================
        import requests
        import os
        
        api_key = os.getenv('TRADERLOCKER_API_KEY', 'YOUR_API_KEY_HERE')
        
        if api_key == 'YOUR_API_KEY_HERE':
            print(f"  [ERROR] Please set TRADERLOCKER_API_KEY environment variable")
            continue
        
        # API endpoint (adjust URL based on TraderLocker API docs)
        url = f"https://api.traderlocker.com/v1/historical/{symbol}"
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=DAYS)
        
        params = {
            'apikey': api_key,
            'start': start_date.isoformat(),
            'end': end_date.isoformat(),
            'timeframe': TIMEFRAME
        }
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame (adjust based on API response format)
            df = pd.DataFrame(data.get('data', []))
            
            if len(df) == 0:
                print(f"  [WARN] No data returned for {symbol}")
                continue
            
            # Ensure timestamp column
            if 'timestamp' not in df.columns:
                if 'time' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['time'])
                elif 'date' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['date'])
                else:
                    print(f"  [ERROR] No timestamp column found in API response")
                    continue
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Ensure required columns exist
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                print(f"  [ERROR] Missing columns: {missing}")
                continue
            
            # Select and reorder columns
            df = df[required_cols]
            
            # Sort and remove duplicates
            df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
            
            # Save to CSV
            filename = f"{symbol.lower()}_5m.csv"
            df.to_csv(filename, index=False)
            
            print(f"  [OK] Saved {len(df)} bars to {filename}")
            print(f"       Range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
            
        except Exception as e:
            print(f"  [ERROR] API request failed: {e}")
            continue
            
    except Exception as e:
        print(f"  [ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "="*80)
print("COMPLETE")
print("="*80)
print("\nCSV files created in current directory.")
print("Download them and place in your project root or data/ folder.")
print("="*80)
