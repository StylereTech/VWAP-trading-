"""
SIMPLE TraderLocker Script - Minimal version for copy/paste
Use this if TraderLocker has a simpler API or script interface
"""

import pandas as pd
from datetime import datetime, timedelta

# ============================================================================
# CONFIGURATION - Adjust these values
# ============================================================================
SYMBOLS = ['GBPJPY', 'BTCUSD', 'XAUUSD']
TIMEFRAME = '5m'  # 5-minute bars
DAYS = 30  # Days of history

# ============================================================================
# MAIN SCRIPT - Copy everything below into TraderLocker
# ============================================================================

def fetch_and_save_data():
    """Fetch data from TraderLocker and save as CSV."""
    
    for symbol in SYMBOLS:
        print(f"Processing {symbol}...")
        
        try:
            # ================================================================
            # OPTION 1: If TraderLocker uses MT5/MetaTrader5
            # ================================================================
            import MetaTrader5 as mt5
            
            if not mt5.initialize():
                print(f"MT5 init failed for {symbol}")
                continue
            
            # Get symbol info
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                print(f"Symbol {symbol} not found")
                mt5.shutdown()
                continue
            
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=DAYS)
            
            # Fetch rates (5-minute timeframe)
            rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M5, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                print(f"No data for {symbol}")
                mt5.shutdown()
                continue
            
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
                'tick_volume': 'volume'  # MT5 uses tick_volume
            })
            
            # Select and reorder columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            # Sort and remove duplicates
            df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
            
            # Save to CSV
            filename = f"{symbol.lower()}_5m.csv"
            df.to_csv(filename, index=False)
            
            print(f"[OK] {symbol}: {len(df)} bars saved to {filename}")
            
            mt5.shutdown()
            
        except ImportError:
            # ================================================================
            # OPTION 2: If TraderLocker uses REST API
            # ================================================================
            import requests
            import os
            
            api_key = os.getenv('TRADERLOCKER_API_KEY', 'YOUR_API_KEY_HERE')
            
            # Adjust URL based on TraderLocker API
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
            
            # Convert to DataFrame (adjust based on API response)
            df = pd.DataFrame(data.get('data', []))
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Ensure required columns
            required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df = df[required_cols]
            
            # Sort and remove duplicates
            df = df.sort_values('timestamp').drop_duplicates('timestamp').reset_index(drop=True)
            
            # Save to CSV
            filename = f"{symbol.lower()}_5m.csv"
            df.to_csv(filename, index=False)
            
            print(f"[OK] {symbol}: {len(df)} bars saved to {filename}")
            
        except Exception as e:
            print(f"[ERROR] {symbol}: {e}")
            continue
    
    print("\nDone! CSV files created in current directory.")

# Run the script
if __name__ == "__main__":
    fetch_and_save_data()
