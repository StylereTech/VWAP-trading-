"""
CSV Data Management Script
Fetch, update, validate, and manage OHLCV CSV files for backtesting
"""

import sys
import os
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import argparse

# Try to import TraderLocker/MetaTrader5 modules
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

from load_real_market_data import load_csv_ohlcv


SYMBOLS = {
    'GBPJPY': {'default_days': 30, 'timeframe': '5m'},
    'BTCUSD': {'default_days': 30, 'timeframe': '5m'},
    'XAUUSD': {'default_days': 30, 'timeframe': '5m'},
}


def fetch_from_mt5(symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Fetch data from MetaTrader5."""
    if not MT5_AVAILABLE:
        raise ImportError("MetaTrader5 not available. Install: pip install MetaTrader5")
    
    if not mt5.initialize():
        raise Exception(f"MT5 initialization failed: {mt5.last_error()}")
    
    try:
        # Check symbol
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            raise Exception(f"Symbol {symbol} not found in MT5")
        
        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Map timeframe
        tf_map = {
            '5m': mt5.TIMEFRAME_M5,
            '15m': mt5.TIMEFRAME_M15,
            '1h': mt5.TIMEFRAME_H1,
            '1d': mt5.TIMEFRAME_D1
        }
        mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_M5)
        
        # Fetch rates
        rates = mt5.copy_rates_range(symbol, mt5_tf, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            raise Exception(f"No data returned for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Rename columns
        df = df.rename(columns={
            'time': 'timestamp',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'tick_volume': 'volume'
        })
        
        # Select required columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
        
        return df
        
    finally:
        mt5.shutdown()


def fetch_from_api(symbol: str, timeframe: str, days: int, api_key: str = None) -> pd.DataFrame:
    """Fetch data from REST API."""
    if not REQUESTS_AVAILABLE:
        raise ImportError("requests not available. Install: pip install requests")
    
    if api_key is None:
        api_key = os.getenv('TRADERLOCKER_API_KEY')
        if not api_key:
            raise ValueError("API key required. Set TRADERLOCKER_API_KEY or pass api_key parameter")
    
    # API endpoint (adjust based on your API)
    url = f"https://api.traderlocker.com/v1/historical/{symbol}"
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    params = {
        'apikey': api_key,
        'start': start_date.isoformat(),
        'end': end_date.isoformat(),
        'timeframe': timeframe
    }
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    response = requests.get(url, params=params, headers=headers)
    response.raise_for_status()
    data = response.json()
    
    # Convert to DataFrame (adjust based on API response format)
    df = pd.DataFrame(data.get('data', []))
    
    if len(df) == 0:
        raise Exception(f"No data returned for {symbol}")
    
    # Ensure timestamp column
    if 'timestamp' not in df.columns:
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'])
        elif 'date' in df.columns:
            df['timestamp'] = pd.to_datetime(df['date'])
        else:
            raise ValueError("No timestamp column found in API response")
    else:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ensure required columns
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    
    df = df[required_cols]
    df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
    
    return df


def update_csv_file(
    symbol: str,
    csv_path: str,
    days: int = 30,
    timeframe: str = '5m',
    source: str = 'mt5',
    api_key: str = None,
    append: bool = False
) -> pd.DataFrame:
    """
    Update or create CSV file with latest data.
    
    Args:
        symbol: Trading symbol (GBPJPY, BTCUSD, XAUUSD)
        csv_path: Path to CSV file
        days: Number of days of history to fetch
        timeframe: Bar timeframe (5m, 15m, 1h, 1d)
        source: Data source ('mt5', 'api', or 'auto')
        api_key: API key for REST API (if using 'api' source)
        append: If True, append to existing file; if False, overwrite
    
    Returns:
        DataFrame with updated data
    """
    print(f"\n{'='*80}")
    print(f"UPDATING CSV: {symbol}")
    print(f"{'='*80}")
    print(f"File: {csv_path}")
    print(f"Days: {days}, Timeframe: {timeframe}")
    print(f"Source: {source}, Append: {append}")
    
    # Fetch new data
    print(f"\nFetching data from {source}...")
    try:
        if source == 'mt5' or (source == 'auto' and MT5_AVAILABLE):
            df_new = fetch_from_mt5(symbol, timeframe, days)
            print(f"[OK] Fetched {len(df_new)} bars from MT5")
        elif source == 'api' or (source == 'auto' and REQUESTS_AVAILABLE and not MT5_AVAILABLE):
            df_new = fetch_from_api(symbol, timeframe, days, api_key)
            print(f"[OK] Fetched {len(df_new)} bars from API")
        else:
            raise ValueError(f"Source '{source}' not available. MT5: {MT5_AVAILABLE}, API: {REQUESTS_AVAILABLE}")
    except Exception as e:
        print(f"[ERROR] Failed to fetch data: {e}")
        raise
    
    # Handle existing file
    if append and os.path.exists(csv_path):
        print(f"\nLoading existing file: {csv_path}")
        try:
            df_existing = load_csv_ohlcv(csv_path)
            print(f"[OK] Loaded {len(df_existing)} existing bars")
            
            # Merge: remove duplicates, keep latest
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            df_combined = df_combined.sort_values('timestamp').drop_duplicates('timestamp', keep='last').reset_index(drop=True)
            
            print(f"[OK] Combined: {len(df_combined)} total bars (removed duplicates)")
            df_final = df_combined
        except Exception as e:
            print(f"[WARN] Failed to load existing file: {e}")
            print(f"[INFO] Creating new file instead")
            df_final = df_new
    else:
        df_final = df_new
    
    # Save to CSV
    print(f"\nSaving to: {csv_path}")
    df_final.to_csv(csv_path, index=False)
    print(f"[OK] Saved {len(df_final)} bars")
    print(f"     Range: {df_final['timestamp'].iloc[0]} -> {df_final['timestamp'].iloc[-1]}")
    
    return df_final


def validate_csv(csv_path: str) -> dict:
    """Validate CSV file."""
    print(f"\n{'='*80}")
    print(f"VALIDATING CSV: {csv_path}")
    print(f"{'='*80}")
    
    if not os.path.exists(csv_path):
        return {'valid': False, 'error': 'File not found'}
    
    try:
        df = load_csv_ohlcv(csv_path)
        
        # Basic checks
        checks = {
            'valid': True,
            'rows': len(df),
            'columns': list(df.columns),
            'timestamp_range': (str(df['timestamp'].iloc[0]), str(df['timestamp'].iloc[-1])),
            'has_nulls': df.isnull().any().any(),
            'sorted': df['timestamp'].is_monotonic_increasing,
            'duplicates': df['timestamp'].duplicated().sum(),
            'price_range': {
                'close_min': float(df['close'].min()),
                'close_max': float(df['close'].max())
            },
            'volume_range': {
                'volume_min': float(df['volume'].min()),
                'volume_max': float(df['volume'].max())
            }
        }
        
        # Validation results
        print(f"\nValidation Results:")
        print(f"  Rows: {checks['rows']}")
        print(f"  Timestamp range: {checks['timestamp_range'][0]} -> {checks['timestamp_range'][1]}")
        print(f"  Sorted: {checks['sorted']}")
        print(f"  Duplicates: {checks['duplicates']}")
        print(f"  Has nulls: {checks['has_nulls']}")
        print(f"  Close range: {checks['price_range']['close_min']:.5f} - {checks['price_range']['close_max']:.5f}")
        print(f"  Volume range: {checks['volume_range']['volume_min']:.0f} - {checks['volume_range']['volume_max']:.0f}")
        
        if not checks['sorted']:
            print(f"  [WARN] Timestamps are not sorted!")
        if checks['duplicates'] > 0:
            print(f"  [WARN] Found {checks['duplicates']} duplicate timestamps!")
        if checks['has_nulls']:
            print(f"  [WARN] Found null values!")
        
        return checks
        
    except Exception as e:
        return {'valid': False, 'error': str(e)}


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description='Update and manage CSV data files')
    parser.add_argument('action', choices=['update', 'validate', 'list'], help='Action to perform')
    parser.add_argument('--symbol', type=str, help='Symbol to update (GBPJPY, BTCUSD, XAUUSD)')
    parser.add_argument('--file', type=str, help='CSV file path (default: {symbol}_5m.csv)')
    parser.add_argument('--days', type=int, default=30, help='Days of history to fetch')
    parser.add_argument('--timeframe', type=str, default='5m', help='Timeframe (5m, 15m, 1h, 1d)')
    parser.add_argument('--source', type=str, default='auto', choices=['mt5', 'api', 'auto'],
                       help='Data source (mt5, api, or auto-detect)')
    parser.add_argument('--api-key', type=str, help='API key for REST API')
    parser.add_argument('--append', action='store_true', help='Append to existing file instead of overwriting')
    
    args = parser.parse_args()
    
    if args.action == 'list':
        print("="*80)
        print("AVAILABLE CSV FILES")
        print("="*80)
        
        csv_files = []
        for symbol in SYMBOLS.keys():
            for path in [f"{symbol.lower()}_5m.csv", f"data/{symbol.lower()}_5m.csv"]:
                if os.path.exists(path):
                    csv_files.append((symbol, path))
        
        if csv_files:
            print(f"\nFound {len(csv_files)} CSV file(s):")
            for symbol, path in csv_files:
                size = os.path.getsize(path) / 1024  # KB
                print(f"  {symbol}: {path} ({size:.1f} KB)")
        else:
            print("\nNo CSV files found.")
            print("\nTo create CSV files, run:")
            print("  python update_csv_data.py update --symbol GBPJPY --source mt5")
        
        return
    
    if args.action == 'validate':
        if not args.file:
            print("[ERROR] --file required for validate action")
            return
        
        result = validate_csv(args.file)
        if result['valid']:
            print("\n[OK] CSV file is valid")
        else:
            print(f"\n[FAIL] CSV file is invalid: {result.get('error', 'Unknown error')}")
        return
    
    if args.action == 'update':
        if not args.symbol:
            print("[ERROR] --symbol required for update action")
            print("Available symbols:", ', '.join(SYMBOLS.keys()))
            return
        
        symbol = args.symbol.upper()
        if symbol not in SYMBOLS:
            print(f"[ERROR] Unknown symbol: {symbol}")
            print(f"Available symbols: {', '.join(SYMBOLS.keys())}")
            return
        
        # Determine file path
        if args.file:
            csv_path = args.file
        else:
            csv_path = f"{symbol.lower()}_5m.csv"
            # Check if data/ folder exists
            if os.path.exists('data'):
                csv_path = f"data/{csv_path}"
        
        # Update CSV
        try:
            df = update_csv_file(
                symbol=symbol,
                csv_path=csv_path,
                days=args.days,
                timeframe=args.timeframe,
                source=args.source,
                api_key=args.api_key,
                append=args.append
            )
            
            # Validate after update
            print("\n" + "="*80)
            print("VALIDATING UPDATED FILE")
            print("="*80)
            result = validate_csv(csv_path)
            
            if result['valid']:
                print("\n[SUCCESS] CSV file updated and validated!")
                print(f"\nReady for backtesting:")
                print(f"  python test_csv_loader.py")
                print(f"  python run_all_symbols_tests.py")
            else:
                print(f"\n[WARN] File updated but validation failed: {result.get('error')}")
                
        except Exception as e:
            print(f"\n[ERROR] Failed to update CSV: {e}")
            import traceback
            traceback.print_exc()
            return


if __name__ == "__main__":
    main()
