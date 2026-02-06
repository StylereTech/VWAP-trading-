"""
Demo script showing what run_all_symbols_tests.py output looks like
This helps visualize the expected output format before running with real data
"""

def print_demo_output():
    """Print example output format."""
    print("="*80)
    print("MULTI-SYMBOL REAL DATA TESTS - DEMO OUTPUT")
    print("="*80)
    print("Symbols: GBPJPY, BTCUSD, XAUUSD")
    print("Tests: A, B, C")
    print("="*80)
    
    print("\n" + "="*80)
    print("CHECKING FOR CSV FILES")
    print("="*80)
    print("[OK] GBPJPY: gbpjpy_5m.csv")
    print("[OK] BTCUSD: btcusd_5m.csv")
    print("[OK] XAUUSD: xauusd_5m.csv")
    
    print("\n[INFO] Will test 3 symbol(s): GBPJPY, BTCUSD, XAUUSD")
    
    print("\n" + "="*80)
    print("PARSEABLE RESULTS TABLE")
    print("="*80)
    print("symbol|test|rung|retest_min-retest_max|vol_filter(p,L)|atr_cap|session|trades|sharpe|dd|total_bars")
    print("-"*100)
    
    # Example results
    demo_results = [
        ("GBPJPY", "A", "TestA", "2-4", "percentile(p=60.0,L=50)", "3.0", "0", "45", "2.34", "5.2%", "8640"),
        ("GBPJPY", "B", "TestB", "1-4", "percentile(p=55.0,L=50)", "3.5", "0", "62", "1.89", "7.1%", "8640"),
        ("GBPJPY", "C", "TestC", "2-4", "percentile(p=65.0,L=50)", "2.5", "1", "28", "3.12", "3.8%", "8640"),
        ("BTCUSD", "A", "TestA", "2-4", "percentile(p=60.0,L=50)", "3.0", "0", "38", "1.45", "12.3%", "8640"),
        ("BTCUSD", "B", "TestB", "1-4", "percentile(p=55.0,L=50)", "3.5", "0", "51", "1.23", "15.6%", "8640"),
        ("BTCUSD", "C", "TestC", "2-4", "percentile(p=65.0,L=50)", "2.5", "1", "22", "2.01", "8.9%", "8640"),
        ("XAUUSD", "A", "TestA", "2-4", "percentile(p=60.0,L=50)", "3.0", "0", "42", "2.67", "6.4%", "8640"),
        ("XAUUSD", "B", "TestB", "1-4", "percentile(p=55.0,L=50)", "3.5", "0", "58", "2.12", "9.2%", "8640"),
        ("XAUUSD", "C", "TestC", "2-4", "percentile(p=65.0,L=50)", "2.5", "1", "31", "3.45", "4.1%", "8640"),
    ]
    
    for symbol, test, rung, retest, vol_filter, atr_cap, session, trades, sharpe, dd, bars in demo_results:
        print(f"{symbol}|{test}|{rung}|{retest}|{vol_filter}|{atr_cap}|{session}|{trades}|{sharpe}|{dd}|{bars}")
    
    print("\n" + "="*80)
    print("DETAILED SUMMARY")
    print("="*80)
    
    symbols_data = {
        "GBPJPY": [
            ("A", "8640", "2026-01-01 00:00:00+00:00 -> 2026-01-31 23:55:00+00:00", "45", "2.34", "5.2%"),
            ("B", "8640", "2026-01-01 00:00:00+00:00 -> 2026-01-31 23:55:00+00:00", "62", "1.89", "7.1%"),
            ("C", "8640", "2026-01-01 00:00:00+00:00 -> 2026-01-31 23:55:00+00:00", "28", "3.12", "3.8%"),
        ],
        "BTCUSD": [
            ("A", "8640", "2026-01-01 00:00:00+00:00 -> 2026-01-31 23:55:00+00:00", "38", "1.45", "12.3%"),
            ("B", "8640", "2026-01-01 00:00:00+00:00 -> 2026-01-31 23:55:00+00:00", "51", "1.23", "15.6%"),
            ("C", "8640", "2026-01-01 00:00:00+00:00 -> 2026-01-31 23:55:00+00:00", "22", "2.01", "8.9%"),
        ],
        "XAUUSD": [
            ("A", "8640", "2026-01-01 00:00:00+00:00 -> 2026-01-31 23:55:00+00:00", "42", "2.67", "6.4%"),
            ("B", "8640", "2026-01-01 00:00:00+00:00 -> 2026-01-31 23:55:00+00:00", "58", "2.12", "9.2%"),
            ("C", "8640", "2026-01-01 00:00:00+00:00 -> 2026-01-31 23:55:00+00:00", "31", "3.45", "4.1%"),
        ]
    }
    
    for symbol, tests in symbols_data.items():
        print(f"\n{symbol}:")
        for test, bars, ts_range, trades, sharpe, dd in tests:
            print(f"  Test {test}:")
            print(f"    Loaded bars: {bars}")
            print(f"    Range: {ts_range}")
            print(f"    Trades: {trades}, Sharpe: {sharpe}, DD: {dd}")
    
    print("\n" + "="*80)
    print("ACCEPTANCE CRITERIA CHECK")
    print("="*80)
    
    criteria_data = {
        "GBPJPY": [
            ("A", 45, 2.34, 0.052, True, True, True),
            ("B", 62, 1.89, 0.071, True, True, True),
            ("C", 28, 3.12, 0.038, False, True, True),  # Fails trades >= 40
        ],
        "BTCUSD": [
            ("A", 38, 1.45, 0.123, False, True, True),  # Fails trades >= 40
            ("B", 51, 1.23, 0.156, True, True, True),
            ("C", 22, 2.01, 0.089, False, True, True),  # Fails trades >= 40
        ],
        "XAUUSD": [
            ("A", 42, 2.67, 0.064, True, True, True),
            ("B", 58, 2.12, 0.092, True, True, True),
            ("C", 31, 3.45, 0.041, False, True, True),  # Fails trades >= 40
        ],
    }
    
    for symbol, tests in criteria_data.items():
        print(f"\n{symbol}:")
        for test, trades, sharpe, dd, trades_ok, sharpe_ok, dd_ok in tests:
            print(f"  Test {test}:")
            print(f"    Trades >= 40: {'[PASS]' if trades_ok else '[FAIL]'} ({trades})")
            print(f"    Sharpe < 8: {'[PASS]' if sharpe_ok else '[FAIL]'} ({sharpe:.2f})")
            print(f"    DD < 20%: {'[PASS]' if dd_ok else '[FAIL]'} ({dd:.2%})")
            overall = trades_ok and sharpe_ok and dd_ok
            print(f"    Overall: {'[PASS]' if overall else '[FAIL]'}")
    
    print("\n" + "="*80)
    print("READY SYMBOLS SUMMARY")
    print("="*80)
    print("\nSymbols ready for optimization (passing at least one test):")
    print("  - GBPJPY: Tests A, B pass (Test C fails trades >= 40)")
    print("  - BTCUSD: Test B passes (Tests A, C fail trades >= 40)")
    print("  - XAUUSD: Tests A, B pass (Test C fails trades >= 40)")
    print("\nRecommendations:")
    print("  - GBPJPY: Use Test A (baseline) or Test B (higher frequency)")
    print("  - BTCUSD: Use Test B (higher frequency) - may need to loosen filters further")
    print("  - XAUUSD: Use Test A (baseline) or Test B (higher frequency)")
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review parseable results table above")
    print("2. Check acceptance criteria for each symbol/test")
    print("3. Proceed with optimization for symbols that pass criteria")
    print("4. Adjust filters for symbols that fail (loosen retest_min, percentile p, or ATR cap)")
    print("="*80)

if __name__ == "__main__":
    print_demo_output()
