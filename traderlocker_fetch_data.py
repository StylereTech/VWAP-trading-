"""
TraderLocker Script - Fetch Historical OHLCV Data
Copy and paste this script into TraderLocker's script editor to fetch data
"""

import pandas as pd
from datetime import datetime, timedelta
import os

# Configuration
SYMBOLS = ['GBPJPY', 'BTCUSD', 'XAUUSD']  # Add or remove symbols as needed
TIMEFRAME = '5m'  # 5-minute bars
DAYS = 30  # Number of days of history to fetch
OUTPUT_DIR = '.'  # Output directory (current directory)

def fetch_traderlocker_data(symbol, timeframe, days):
    """
    Fetch historical OHLCV data from TraderLocker API.
    
    Adjust this function based on your TraderLocker API documentation.
    Common patterns:
    - TraderLocker may use mt5, mt4, or custom API
    - May require authentication via API key or session
    """
    try:
        # Option 1: If TraderLocker uses MT5/MetaTrader5
        try:
            import MetaTrader5 as mt5
            
            if not mt5.initialize():
                raise Exception(f"MT5 initialization failed: {mt5.last_error()}")
            
            # Convert symbol format if needed
            mt5_symbol = symbol
            if symbol == 'BTCUSD':
                mt5_symbol = 'BTCUSD'  # Adjust based on your broker's symbol name
            elif symbol == 'XAUUSD':
                mt5_symbol = 'XAUUSD'  # or 'GOLD' depending on broker
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Fetch rates
            rates = mt5.copy_rates_range(mt5_symbol, 
                                        mt5.TIMEFRAME_M5 if timeframe == '5m' else mt5.TIMEFRAME_M1,
                                        start_date, end_date)
            
            mt5.shutdown()
            
            if rates is None or len(rates) == 0:
                raise Exception(f"No data returned for {symbol}")
            
            # Convert to DataFrame
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.rename(columns={
                'time': 'timestamp',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'tick_volume': 'volume'  # MT5 uses tick_volume
            })
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except ImportError:
            # Option 2: If TraderLocker uses REST API
            import requests
            
            api_key = os.getenv('TRADERLOCKER_API_KEY')
            if not api_key:
                raise Exception("TRADERLOCKER_API_KEY not set")
            
            # Adjust URL and parameters based on TraderLocker API docs
            url = f"https://api.traderlocker.com/v1/historical/{symbol}"
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            params = {
                'apikey': api_key,
                'start': start_date.isoformat(),
                'end': end_date.isoformat(),
                'timeframe': timeframe
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame (adjust based on API response format)
            df = pd.DataFrame(data.get('data', []))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        raise

def save_csv(df, symbol, output_dir):
    """Save DataFrame to CSV file."""
    filename = f"{symbol.lower()}_5m.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Ensure timestamp is in correct format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(filepath, index=False)
    print(f"Saved: {filepath} ({len(df)} bars)")
    
    return filepath

def main():
    """Main function to fetch and save data for all symbols."""
    print("="*80)
    print("TRADERLOCKER DATA FETCHER")
    print("="*80)
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Timeframe: {TIMEFRAME}")
    print(f"Days: {DAYS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*80)
    
    results = {}
    
    for symbol in SYMBOLS:
        print(f"\nFetching {symbol}...")
        try:
            df = fetch_traderlocker_data(symbol, TIMEFRAME, DAYS)
            filepath = save_csv(df, symbol, OUTPUT_DIR)
            results[symbol] = {'success': True, 'filepath': filepath, 'bars': len(df)}
            print(f"[OK] {symbol}: {len(df)} bars saved to {filepath}")
        except Exception as e:
            results[symbol] = {'success': False, 'error': str(e)}
            print(f"[FAIL] {symbol}: {e}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for symbol, result in results.items():
        if result['success']:
            print(f"[OK] {symbol}: {result['bars']} bars -> {result['filepath']}")
        else:
            print(f"[FAIL] {symbol}: {result['error']}")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Verify CSV files were created")
    print("2. Run: python test_csv_loader.py")
    print("3. Run: python run_all_symbols_tests.py")
    print("="*80)

if __name__ == "__main__":
    main()
