"""
Convert CSV timestamps to UTC timezone.
Useful if your CSV has local timezone timestamps.
"""

import pandas as pd
import sys
from pathlib import Path
from pytz import timezone, UTC

def convert_csv_timezone(input_csv: str, output_csv: str, from_tz: str, to_tz: str = "UTC"):
    """
    Convert CSV timestamp column to different timezone.
    
    Args:
        input_csv: Path to input CSV
        output_csv: Path to output CSV
        from_tz: Source timezone (e.g., "America/New_York", "Europe/London", "UTC")
        to_tz: Target timezone (default: "UTC")
    """
    print(f"Loading: {input_csv}")
    df = pd.read_csv(input_csv)
    
    # Find timestamp column
    ts_cols = ['timestamp', 'time', 'datetime', 'date']
    ts_col = None
    for col in ts_cols:
        if col in df.columns:
            ts_col = col
            break
    
    if ts_col is None:
        raise ValueError(f"No timestamp column found. Expected one of: {ts_cols}")
    
    print(f"Found timestamp column: {ts_col}")
    
    # Parse timestamps
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    
    # Check if already timezone-aware
    if df[ts_col].dt.tz is None:
        print(f"Timestamps are naive. Localizing from: {from_tz}")
        from_tz_obj = timezone(from_tz)
        df[ts_col] = df[ts_col].dt.tz_localize(from_tz_obj)
    else:
        print(f"Timestamps are timezone-aware. Converting from current timezone")
    
    # Convert to target timezone
    if to_tz == "UTC":
        to_tz_obj = UTC
    else:
        to_tz_obj = timezone(to_tz)
    
    print(f"Converting to: {to_tz}")
    df[ts_col] = df[ts_col].dt.tz_convert(to_tz_obj)
    
    # Save
    print(f"Saving: {output_csv}")
    df.to_csv(output_csv, index=False)
    
    print(f"\n[SUCCESS] Timezone conversion complete!")
    print(f"  Rows: {len(df)}")
    print(f"  Range: {df[ts_col].iloc[0]} -> {df[ts_col].iloc[-1]}")
    print(f"  Timezone: {to_tz}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert CSV timestamps to UTC')
    parser.add_argument('input_csv', type=str, help='Input CSV file path')
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    parser.add_argument('--from', type=str, dest='from_tz', required=True,
                       help='Source timezone (e.g., America/New_York, Europe/London, UTC)')
    parser.add_argument('--to', type=str, dest='to_tz', default='UTC',
                       help='Target timezone (default: UTC)')
    
    args = parser.parse_args()
    
    if not Path(args.input_csv).exists():
        print(f"[ERROR] File not found: {args.input_csv}")
        sys.exit(1)
    
    try:
        convert_csv_timezone(args.input_csv, args.output, args.from_tz, args.to_tz)
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
