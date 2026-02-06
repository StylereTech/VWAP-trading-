"""
Add proxy volume column to CSV if volume column is missing.
Use this ONLY for validation/testing - not for final optimization.

Proxy volume options:
- abs(close - open) - body size
- high - low - range size
"""

import pandas as pd
import sys
from pathlib import Path

def add_proxy_volume(input_csv: str, output_csv: str = None, method: str = "body"):
    """
    Add proxy volume column to CSV.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV (default: overwrites input)
        method: "body" (abs(close-open)) or "range" (high-low)
    """
    if output_csv is None:
        output_csv = input_csv
    
    print(f"Loading: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Check if volume already exists
    vol_cols = [c for c in df.columns if c.lower() in ['volume', 'vol', 'tick_volume']]
    if vol_cols:
        print(f"[WARN] Volume column already exists: {vol_cols}")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        # Remove existing volume columns
        for col in vol_cols:
            df = df.drop(columns=[col])
    
    # Normalize column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]
    
    # Ensure required columns exist
    required = ['open', 'high', 'low', 'close']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")
    
    # Calculate proxy volume
    if method == "body":
        df['volume'] = abs(df['close'] - df['open'])
        print("[INFO] Using proxy volume: abs(close - open) [body size]")
    elif method == "range":
        df['volume'] = df['high'] - df['low']
        print("[INFO] Using proxy volume: high - low [range size]")
    else:
        raise ValueError(f"Unknown method: {method}. Use 'body' or 'range'")
    
    # Ensure volume is positive and non-zero
    df['volume'] = df['volume'].abs()
    df.loc[df['volume'] == 0, 'volume'] = 0.0001  # Minimum to avoid division issues
    
    # Reorder columns (timestamp first, volume last)
    cols = [c for c in df.columns if c != 'volume']
    if 'timestamp' in cols or 'time' in cols or 'date' in cols:
        ts_col = next((c for c in ['timestamp', 'time', 'date', 'datetime'] if c in cols), None)
        if ts_col:
            cols = [ts_col] + [c for c in cols if c != ts_col]
    cols.append('volume')
    
    df = df[cols]
    
    # Save
    print(f"Saving: {output_csv}")
    df.to_csv(output_csv, index=False)
    
    print(f"\n[SUCCESS] Proxy volume added!")
    print(f"  Rows: {len(df)}")
    print(f"  Volume min/max: {df['volume'].min():.6f} / {df['volume'].max():.6f}")
    print(f"  Volume mean: {df['volume'].mean():.6f}")
    print(f"\n[WARN] This is proxy volume for validation only!")
    print(f"       Use real tick_volume or volume for final optimization.")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Add proxy volume to CSV')
    parser.add_argument('input_csv', type=str, help='Input CSV file path')
    parser.add_argument('--output', type=str, default=None, help='Output CSV path (default: overwrite input)')
    parser.add_argument('--method', type=str, default='body', choices=['body', 'range'],
                       help='Proxy volume method: body (abs(close-open)) or range (high-low)')
    
    args = parser.parse_args()
    
    if not Path(args.input_csv).exists():
        print(f"[ERROR] File not found: {args.input_csv}")
        sys.exit(1)
    
    add_proxy_volume(args.input_csv, args.output, args.method)
