"""
Check what data files are available for backtesting
"""

from pathlib import Path
import os

SYMBOLS = {
    'GBPJPY': ['gbpjpy_5m.csv', 'GBPJPY_5m.csv'],
    'BTCUSD': ['btcusd_5m.csv', 'BTCUSD_5m.csv', 'btc_5m.csv', 'BTC_5m.csv'],
    'XAUUSD': ['xauusd_5m.csv', 'XAUUSD_5m.csv', 'gold_5m.csv', 'xau_5m.csv', 'XAU_5m.csv']
}

def check_data_files():
    """Check what CSV files are available."""
    print("="*80)
    print("DATA FILES CHECK")
    print("="*80)
    
    found_files = {}
    missing_symbols = []
    
    for symbol, variants in SYMBOLS.items():
        found = None
        for variant in variants:
            # Check project root
            p1 = Path(variant)
            # Check data folder
            p2 = Path("data") / variant
            
            if p1.exists():
                found = p1
                break
            elif p2.exists():
                found = p2
                break
        
        if found:
            print(f"[OK] {symbol}: {found}")
            found_files[symbol] = found
        else:
            print(f"[MISSING] {symbol}: No CSV found")
            print(f"         Looking for: {', '.join(variants[:2])}...")
            missing_symbols.append(symbol)
    
    # Check TraderLocker API
    print("\n" + "="*80)
    print("TRADERLOCKER API CHECK")
    print("="*80)
    api_key = os.getenv('TRADERLOCKER_API_KEY')
    if api_key:
        print(f"[OK] TRADERLOCKER_API_KEY is set")
        print(f"     Will attempt to fetch data for missing symbols")
    else:
        print("[MISSING] TRADERLOCKER_API_KEY not set")
        print("          Set it with: export TRADERLOCKER_API_KEY=your_key")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    if found_files:
        print(f"\n[READY] {len(found_files)} symbol(s) have CSV files:")
        for symbol, path in found_files.items():
            print(f"  - {symbol}: {path}")
    
    if missing_symbols:
        print(f"\n[MISSING] {len(missing_symbols)} symbol(s) need CSV files:")
        for symbol in missing_symbols:
            variants = SYMBOLS[symbol]
            print(f"  - {symbol}: Place {variants[0]} in project root or data/ folder")
    
    if api_key and missing_symbols:
        print(f"\n[INFO] TraderLocker API will attempt to fetch data for: {', '.join(missing_symbols)}")
    
    # Next steps
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    
    if found_files:
        print("\nTo run tests for available symbols:")
        print("  python run_all_symbols_tests.py")
        print("\nTo run tests for a specific symbol:")
        symbols_list = ', '.join(found_files.keys())
        print(f"  python run_ablation_real_data.py --test A --symbol <SYMBOL> --days 30")
        print(f"  Available symbols: {symbols_list}")
    else:
        print("\nNo CSV files found. To proceed:")
        print("1. Export CSV files from TradingView or your broker:")
        print("   - GBPJPY: gbpjpy_5m.csv")
        print("   - BTCUSD: btcusd_5m.csv")
        print("   - XAUUSD: xauusd_5m.csv")
        print("\n2. Place files in project root or data/ folder")
        print("\n3. Or set TRADERLOCKER_API_KEY to fetch data via API")
        print("\n4. Then run: python run_all_symbols_tests.py")
    
    return len(found_files) > 0 or api_key is not None

if __name__ == "__main__":
    import sys
    has_data = check_data_files()
    sys.exit(0 if has_data else 1)
