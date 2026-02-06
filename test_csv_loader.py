"""
Quick CSV presence test - verify data can be loaded before running full tests
"""

import os
import sys
from pathlib import Path
import pandas as pd
from load_real_market_data import load_csv_ohlcv

def test_csv_presence():
    """Test if CSV file exists and can be loaded."""
    print("="*80)
    print("CSV PRESENCE TEST")
    print("="*80)
    
    # Check candidate paths
    candidates = [
        Path("gbpjpy_5m.csv"),
        Path("data/gbpjpy_5m.csv"),
        Path("GBPJPY_5m.csv"),
        Path("data/GBPJPY_5m.csv"),
    ]
    
    csv_path = None
    for p in candidates:
        if p.exists():
            csv_path = p
            print(f"[OK] Found CSV: {csv_path}")
            break
    
    if csv_path is None:
        print("\n[FAIL] No CSV file found in expected locations:")
        for p in candidates:
            print(f"  - {p} (exists: {p.exists()})")
        print("\nPlease place GBPJPY 5-minute OHLCV data in one of these locations:")
        print("  - gbpjpy_5m.csv (project root)")
        print("  - data/gbpjpy_5m.csv")
        return False
    
    # Try to load
    print(f"\nAttempting to load: {csv_path}")
    try:
        df = load_csv_ohlcv(str(csv_path))
        
        print("\n" + "="*80)
        print("LOAD SUCCESS")
        print("="*80)
        print(f"Rows: {len(df)}")
        print(f"Expected (~30 days @ 5m): ~8,640 bars")
        print(f"Range: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
        print(f"Close min/max: {df['close'].min():.5f} / {df['close'].max():.5f}")
        print(f"Vol min/max: {df['volume'].min()} / {df['volume'].max()}")
        
        # Check if timestamps are increasing
        is_sorted = df['timestamp'].is_monotonic_increasing
        print(f"Timestamps sorted: {is_sorted}")
        
        # Check for gaps
        time_diffs = df['timestamp'].diff()
        median_diff = time_diffs.median()
        print(f"Median time difference: {median_diff}")
        
        print("\n" + "="*80)
        print("FIRST 3 ROWS")
        print("="*80)
        print(df.head(3).to_string())
        
        print("\n" + "="*80)
        print("LAST 3 ROWS")
        print("="*80)
        print(df.tail(3).to_string())
        
        # Pass conditions
        print("\n" + "="*80)
        print("PASS CONDITIONS CHECK")
        print("="*80)
        rows_ok = len(df) > 1000  # At least some reasonable amount
        timestamps_ok = is_sorted
        volume_ok = (df['volume'] > 0).all()
        
        print(f"Rows > 1000: {'[PASS]' if rows_ok else '[FAIL]'} ({len(df)})")
        print(f"Timestamps sorted: {'[PASS]' if timestamps_ok else '[FAIL]'}")
        print(f"Volume non-zero: {'[PASS]' if volume_ok else '[FAIL]'}")
        
        if rows_ok and timestamps_ok and volume_ok:
            print("\n[SUCCESS] CSV is ready for backtesting!")
            return True
        else:
            print("\n[WARN] CSV loaded but some checks failed. Review above.")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Failed to load CSV: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_csv_presence()
    sys.exit(0 if success else 1)
