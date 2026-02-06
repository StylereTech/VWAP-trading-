"""
Real Market Data Loader
Supports CSV and prepares for TraderLocker API
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import os


def load_csv_ohlcv(path: str, tz: str = "UTC") -> pd.DataFrame:
    """
    Robust CSV loader that handles common variants from TradingView, brokers, etc.
    
    Required columns (final internal names): timestamp, open, high, low, close, volume
    Accepts common variants: time, date, datetime, Open, High, Low, Close, Volume, vol, tick_volume
    """
    df = pd.read_csv(path)
    
    # 1) Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    rename_map = {
        "time": "timestamp",
        "date": "timestamp",
        "datetime": "timestamp",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
        "vol": "volume",
        "tick_volume": "volume",
    }
    for k, v in list(rename_map.items()):
        if k in df.columns:
            df = df.rename(columns={k: v})
    
    required = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing columns: {missing}. Found: {list(df.columns)}")
    
    # 2) Timestamp parse
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=False)
    if df["timestamp"].isna().any():
        bad = int(df["timestamp"].isna().sum())
        raise ValueError(f"Failed to parse {bad} timestamps. Check CSV timestamp format.")
    
    # 3) Force timezone
    # If timestamps are naive, localize; if aware, convert.
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(tz)
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
    
    # 4) Types + sort
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).sort_values("timestamp").drop_duplicates("timestamp")
    
    return df[required].reset_index(drop=True)


def load_from_csv(filepath: str, symbol: str = None) -> pd.DataFrame:
    """
    Load OHLCV data from CSV (wrapper for backward compatibility).
    Uses the robust load_csv_ohlcv function.
    """
    return load_csv_ohlcv(filepath)


def load_from_traderlocker(symbol: str, days: int = 60, api_key: Optional[str] = None) -> pd.DataFrame:
    """
    Load historical data from TraderLocker API.
    
    Requires API key and symbol.
    """
    import requests
    
    if api_key is None:
        api_key = os.getenv('TRADERLOCKER_API_KEY')
        if api_key is None:
            raise ValueError("TraderLocker API key required. Set TRADERLOCKER_API_KEY env var or pass api_key parameter")
    
    # API endpoint (adjust based on actual TraderLocker API)
    url = f"https://api.traderlocker.com/v1/historical/{symbol}"
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    params = {
        'apikey': api_key,
        'start': start_date.isoformat(),
        'end': end_date.isoformat(),
        'timeframe': '5m'  # 5-minute bars
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Convert to DataFrame (adjust based on actual API response format)
        df = pd.DataFrame(data.get('data', []))
        
        # Map columns
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.rename(columns={
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        
        return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].sort_values('timestamp').reset_index(drop=True)
    
    except Exception as e:
        raise ValueError(f"Failed to load data from TraderLocker: {e}")


def prepare_data_for_backtest(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Prepare real market data for backtesting.
    Adds basic validation and filtering.
    """
    # Filter to weekdays only (if needed)
    df = df[df['timestamp'].dt.weekday < 5].copy()
    
    # Ensure timezone-aware (convert to UTC if needed)
    if df['timestamp'].dt.tz is None:
        # Assume data is in UTC or local timezone
        # For now, keep as-is but log warning
        print("Warning: Timestamps are timezone-naive. Assuming UTC.")
    
    # Remove any remaining NaNs
    df = df.dropna().reset_index(drop=True)
    
    # Validate data quality
    print(f"\nData Summary for {symbol}:")
    print(f"  Bars: {len(df)}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Close range: {df['close'].min():.5f} to {df['close'].max():.5f}")
    print(f"  Volume range: {df['volume'].min()} to {df['volume'].max()}")
    print(f"  Missing values: {df.isna().sum().sum()}")
    
    return df


if __name__ == "__main__":
    # Example usage
    print("Real Market Data Loader")
    print("="*80)
    print("\nUsage:")
    print("1. Load from CSV:")
    print("   df = load_from_csv('gbpjpy_5m.csv')")
    print("2. Load from TraderLocker:")
    print("   df = load_from_traderlocker('GBPJPY', days=60, api_key='your_key')")
    print("3. Prepare for backtest:")
    print("   df = prepare_data_for_backtest(df, 'GBPJPY')")
    print("="*80)
