"""
================================================================================
UPDATED COPY-PASTE SCRIPT FOR TRADERLOCKER
================================================================================
This script fetches historical OHLCV data and saves as CSV files
for GBPJPY, BTCUSD, and XAUUSD (5-minute bars, 30 days)

UPDATED VERSION: Includes error handling and better output
================================================================================
"""

import pandas as pd
from datetime import datetime, timedelta
import os

# ============================================================================
# CONFIGURATION - Adjust if needed
# ============================================================================
SYMBOLS = ['GBPJPY', 'BTCUSD', 'XAUUSD']
TIMEFRAME = '5m'
DAYS = 30
OUTPUT_DIR = '.'  # Current directory (change to 'data' if you want data/ folder)

# ============================================================================
# SCRIPT STARTS HERE - Copy everything below into TraderLocker
# ============================================================================

print("="*80)
print("TRADERLOCKER DATA FETCHER - UPDATED")
print("="*80)
print(f"Symbols: {', '.join(SYMBOLS)}")
print(f"Timeframe: {TIMEFRAME}")
print(f"Days: {DAYS}")
print(f"Output directory: {OUTPUT_DIR}")
print("="*80)

results = {}

for symbol in SYMBOLS:
    print(f"\n{'='*80}")
    print(f"Processing {symbol}...")
    print('='*80)
    
    try:
        # ================================================================
        # OPTION 1: MetaTrader5 (MT5) - Most common
        # ================================================================
        import MetaTrader5 as mt5
        
        # Initialize MT5
        if not mt5.initialize():
            print(f"  [ERROR] MT5 initialization failed: {mt5.last_error()}")
            results[symbol] = {'success': False, 'error': f"MT5 init: {mt5.last_error()}"}
            continue
        
        # Check if symbol exists
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"  [ERROR] Symbol {symbol} not found in MT5")
            mt5.shutdown()
            results[symbol] = {'success': False, 'error': 'Symbol not found'}
            continue
        
        # Enable symbol if not available
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                print(f"  [ERROR] Failed to enable symbol {symbol}")
                mt5.shutdown()
                results[symbol] = {'success': False, 'error': 'Failed to enable symbol'}
                continue
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=DAYS)
        
        print(f"  Fetching data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        
        # Fetch 5-minute bars
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            print(f"  [WARN] No data returned for {symbol}")
            mt5.shutdown()
            results[symbol] = {'success': False, 'error': 'No data returned'}
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
        if OUTPUT_DIR != '.':
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            filepath = os.path.join(OUTPUT_DIR, filename)
        else:
            filepath = filename
        
        df.to_csv(filepath, index=False)
        
        print(f"  [OK] Saved {len(df)} bars to {filepath}")
        print(f"       Range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
        print(f"       Close: {df['close'].min():.5f} - {df['close'].max():.5f}")
        print(f"       Volume: {df['volume'].min():.0f} - {df['volume'].max():.0f}")
        
        results[symbol] = {
            'success': True,
            'filepath': filepath,
            'bars': len(df),
            'range': (str(df['timestamp'].iloc[0]), str(df['timestamp'].iloc[-1]))
        }
        
        mt5.shutdown()
        
    except ImportError:
        # ================================================================
        # OPTION 2: REST API (if TraderLocker uses REST API)
        # ================================================================
        import requests
        
        api_key = os.getenv('TRADERLOCKER_API_KEY', 'YOUR_API_KEY_HERE')
        
        if api_key == 'YOUR_API_KEY_HERE':
            print(f"  [ERROR] Please set TRADERLOCKER_API_KEY environment variable")
            print(f"          Or modify script to include your API key")
            results[symbol] = {'success': False, 'error': 'API key not set'}
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
            print(f"  Fetching from API...")
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame (adjust based on API response format)
            df = pd.DataFrame(data.get('data', []))
            
            if len(df) == 0:
                print(f"  [WARN] No data returned for {symbol}")
                results[symbol] = {'success': False, 'error': 'No data in API response'}
                continue
            
            # Ensure timestamp column
            if 'timestamp' not in df.columns:
                if 'time' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['time'])
                elif 'date' in df.columns:
                    df['timestamp'] = pd.to_datetime(df['date'])
                else:
                    print(f"  [ERROR] No timestamp column found in API response")
                    print(f"          Available columns: {list(df.columns)}")
                    results[symbol] = {'success': False, 'error': 'No timestamp column'}
                    continue
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Ensure required columns exist
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                print(f"  [ERROR] Missing columns: {missing}")
                print(f"          Available columns: {list(df.columns)}")
                results[symbol] = {'success': False, 'error': f'Missing columns: {missing}'}
                continue
            
            # Select and reorder columns
            df = df[required_cols]
            
            # Sort and remove duplicates
            df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
            
            # Save to CSV
            filename = f"{symbol.lower()}_5m.csv"
            if OUTPUT_DIR != '.':
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                filepath = os.path.join(OUTPUT_DIR, filename)
            else:
                filepath = filename
            
            df.to_csv(filepath, index=False)
            
            print(f"  [OK] Saved {len(df)} bars to {filepath}")
            print(f"       Range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
            print(f"       Close: {df['close'].min():.5f} - {df['close'].max():.5f}")
            print(f"       Volume: {df['volume'].min():.0f} - {df['volume'].max():.0f}")
            
            results[symbol] = {
                'success': True,
                'filepath': filepath,
                'bars': len(df),
                'range': (str(df['timestamp'].iloc[0]), str(df['timestamp'].iloc[-1]))
            }
            
        except requests.exceptions.RequestException as e:
            print(f"  [ERROR] API request failed: {e}")
            results[symbol] = {'success': False, 'error': f'API error: {str(e)}'}
            continue
            
    except Exception as e:
        print(f"  [ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        results[symbol] = {'success': False, 'error': str(e)}
        continue

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

success_count = sum(1 for r in results.values() if r.get('success'))
total_count = len(results)

for symbol, result in results.items():
    if result.get('success'):
        print(f"[OK] {symbol}: {result['bars']} bars -> {result['filepath']}")
    else:
        print(f"[FAIL] {symbol}: {result.get('error', 'Unknown error')}")

print(f"\nSuccess: {success_count}/{total_count}")

if success_count > 0:
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("1. Download CSV files from TraderLocker")
    print("2. Place in project root or data/ folder")
    print("3. Run: python test_csv_loader.py")
    print("4. Run: python run_all_symbols_tests.py")
    print("="*80)
else:
    print("\n[ERROR] No CSV files were created.")
    print("Check error messages above and verify:")
    print("  - MT5 is running and logged in")
    print("  - Symbols are available in MT5")
    print("  - OR API key is set correctly")
    print("="*80)
