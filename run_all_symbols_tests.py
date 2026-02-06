"""
Run real data tests (A, B, C) for all symbols: GBPJPY, BTCUSD, XAUUSD
"""

import sys
import os
import subprocess
from pathlib import Path
import pandas as pd

SYMBOLS = ['GBPJPY', 'BTCUSD', 'XAUUSD']
TESTS = ['A', 'B', 'C']

def find_csv_for_symbol(symbol):
    """Find CSV file for a symbol."""
    candidates = [
        Path(f"{symbol.lower()}_5m.csv"),
        Path("data") / f"{symbol.lower()}_5m.csv",
        Path(f"{symbol}_5m.csv"),
        Path("data") / f"{symbol}_5m.csv",
    ]
    
    # Also check for common variations
    if symbol == 'BTCUSD':
        candidates.extend([
            Path("btc_5m.csv"),
            Path("data/btc_5m.csv"),
            Path("BTC_5m.csv"),
            Path("data/BTC_5m.csv"),
        ])
    elif symbol == 'XAUUSD':
        candidates.extend([
            Path("gold_5m.csv"),
            Path("data/gold_5m.csv"),
            Path("xau_5m.csv"),
            Path("data/xau_5m.csv"),
            Path("XAU_5m.csv"),
            Path("data/XAU_5m.csv"),
        ])
    
    for p in candidates:
        if p.exists():
            return p
    return None

def run_test(symbol, test):
    """Run a single test for a symbol."""
    print("\n" + "="*80)
    print(f"RUNNING: {symbol} - Test {test}")
    print("="*80)
    
    cmd = [
        sys.executable,
        'run_ablation_real_data.py',
        '--test', test,
        '--symbol', symbol,
        '--days', '30'
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 minute timeout
            encoding='utf-8',
            errors='replace'
        )
        
        output = result.stdout + result.stderr
        
        # Extract key information
        parseable_line = None
        bars_info = None
        timestamp_range = None
        
        for line in output.split('\n'):
            if 'rung|retest_min-retest_max|vol_filter' in line:
                # Header line
                continue
            elif '|' in line and 'Test' in line:
                # Parseable result line
                parseable_line = line.strip()
            elif 'Loaded bars:' in line or 'Total bars:' in line:
                bars_info = line.strip()
            elif 'Range:' in line and '->' in line:
                timestamp_range = line.strip()
        
        return {
            'success': result.returncode == 0,
            'output': output,
            'parseable_line': parseable_line,
            'bars_info': bars_info,
            'timestamp_range': timestamp_range,
            'error': result.stderr if result.returncode != 0 else None
        }
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'output': '',
            'parseable_line': None,
            'bars_info': None,
            'timestamp_range': None,
            'error': 'Timeout (10 minutes exceeded)'
        }
    except Exception as e:
        return {
            'success': False,
            'output': '',
            'parseable_line': None,
            'bars_info': None,
            'timestamp_range': None,
            'error': str(e)
        }

def main():
    """Run all tests for all symbols."""
    print("="*80)
    print("MULTI-SYMBOL REAL DATA TESTS")
    print("="*80)
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Tests: {', '.join(TESTS)}")
    print("="*80)
    
    # Check for CSV files
    print("\n" + "="*80)
    print("CHECKING FOR CSV FILES")
    print("="*80)
    symbol_files = {}
    for symbol in SYMBOLS:
        csv_path = find_csv_for_symbol(symbol)
        if csv_path:
            print(f"[OK] {symbol}: {csv_path}")
            symbol_files[symbol] = csv_path
        else:
            print(f"[MISSING] {symbol}: No CSV found")
            symbol_files[symbol] = None
    
    # Check if any data is available
    available_symbols = [s for s, f in symbol_files.items() if f is not None]
    if not available_symbols:
        print("\n" + "="*80)
        print("ERROR: No CSV files found for any symbol")
        print("="*80)
        print("Please provide CSV files:")
        for symbol in SYMBOLS:
            print(f"  - {symbol.lower()}_5m.csv or data/{symbol.lower()}_5m.csv")
        print("\nOr set TRADERLOCKER_API_KEY environment variable")
        sys.exit(1)
    
    print(f"\n[INFO] Will test {len(available_symbols)} symbol(s): {', '.join(available_symbols)}")
    
    # Run tests
    results = {}
    for symbol in available_symbols:
        results[symbol] = {}
        for test in TESTS:
            print(f"\n{'='*80}")
            print(f"PROCESSING: {symbol} - Test {test}")
            print('='*80)
            result = run_test(symbol, test)
            results[symbol][test] = result
            
            if result['success']:
                print(f"[OK] {symbol} - Test {test} completed")
            else:
                print(f"[FAIL] {symbol} - Test {test} failed")
                if result['error']:
                    print(f"Error: {result['error']}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Parseable results table
    print("\nPARSEABLE RESULTS TABLE")
    print("="*80)
    print("symbol|test|rung|retest_min-retest_max|vol_filter(p,L)|atr_cap|session|trades|sharpe|dd|total_bars")
    print("-"*100)
    
    for symbol in available_symbols:
        for test in TESTS:
            r = results[symbol][test]
            if r['success'] and r['parseable_line']:
                # Extract symbol and test from parseable line
                parts = r['parseable_line'].split('|')
                if len(parts) >= 8:
                    # Insert symbol and test at the beginning
                    print(f"{symbol}|{test}|{r['parseable_line']}")
            else:
                # Failed test
                print(f"{symbol}|{test}|FAILED|{r.get('error', 'Unknown error')}")
    
    # Detailed summary
    print("\n" + "="*80)
    print("DETAILED SUMMARY")
    print("="*80)
    
    for symbol in available_symbols:
        print(f"\n{symbol}:")
        for test in TESTS:
            r = results[symbol][test]
            if r['success']:
                if r['bars_info']:
                    print(f"  Test {test}: {r['bars_info']}")
                if r['timestamp_range']:
                    print(f"           {r['timestamp_range']}")
                if r['parseable_line']:
                    # Extract trades, sharpe, dd
                    parts = r['parseable_line'].split('|')
                    if len(parts) >= 8:
                        trades = parts[6] if len(parts) > 6 else 'N/A'
                        sharpe = parts[7] if len(parts) > 7 else 'N/A'
                        dd = parts[8] if len(parts) > 8 else 'N/A'
                        print(f"           Trades: {trades}, Sharpe: {sharpe}, DD: {dd}")
            else:
                print(f"  Test {test}: FAILED - {r.get('error', 'Unknown error')}")
    
    # Acceptance criteria check
    print("\n" + "="*80)
    print("ACCEPTANCE CRITERIA CHECK")
    print("="*80)
    
    for symbol in available_symbols:
        print(f"\n{symbol}:")
        for test in TESTS:
            r = results[symbol][test]
            if r['success'] and r['parseable_line']:
                parts = r['parseable_line'].split('|')
                if len(parts) >= 8:
                    try:
                        trades = int(parts[6])
                        sharpe = float(parts[7])
                        dd_str = parts[8].replace('%', '')
                        dd = float(dd_str) / 100 if '%' not in parts[8] else float(dd_str)
                        
                        trades_ok = trades >= 40
                        sharpe_ok = sharpe < 8
                        dd_ok = dd < 0.20
                        
                        print(f"  Test {test}:")
                        print(f"    Trades >= 40: {'[PASS]' if trades_ok else '[FAIL]'} ({trades})")
                        print(f"    Sharpe < 8: {'[PASS]' if sharpe_ok else '[FAIL]'} ({sharpe:.2f})")
                        print(f"    DD < 20%: {'[PASS]' if dd_ok else '[FAIL]'} ({dd:.2%})")
                        print(f"    Overall: {'[PASS]' if (trades_ok and sharpe_ok and dd_ok) else '[FAIL]'}")
                    except (ValueError, IndexError):
                        print(f"  Test {test}: Could not parse results")
            else:
                print(f"  Test {test}: [SKIP] - Test failed or no data")
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review parseable results table above")
    print("2. Check acceptance criteria for each symbol/test")
    print("3. Proceed with optimization for symbols that pass criteria")
    print("="*80)

if __name__ == "__main__":
    main()
