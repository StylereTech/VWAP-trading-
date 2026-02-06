"""
Run QUBO/QAOA Optimization for All Symbols
Optimizes parameters using quantum-inspired methods
"""

import subprocess
import sys
import time

def run_optimization(symbol: str, days: int = 30, samples: int = 150):
    """Run QUBO optimization for a symbol."""
    print(f"\n{'='*80}")
    print(f"OPTIMIZING {symbol}")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable,
        "vwap_qubo_tuner.py",
        "--symbol", symbol,
        "--days", str(days),
        "--samples", str(samples)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    return result.returncode == 0

def main():
    """Run optimization for all symbols."""
    print("="*80)
    print("QUBO/QAOA OPTIMIZATION - ALL SYMBOLS")
    print("="*80)
    
    symbols = ['GBPJPY', 'BTCUSD', 'XAUUSD']
    
    # Use smaller dataset (30 days) and fewer samples for speed
    # Can increase if you want more thorough optimization
    days = 30
    samples = 150
    
    print(f"\nSettings:")
    print(f"  Backtest Days: {days}")
    print(f"  Samples per Symbol: {samples}")
    print(f"  Estimated Time: ~{len(symbols) * samples * 2 / 60:.0f} minutes")
    
    results = {}
    start_time = time.time()
    
    for symbol in symbols:
        success = run_optimization(symbol, days=days, samples=samples)
        results[symbol] = success
    
    total_time = time.time() - start_time
    
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    
    for symbol, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"{symbol}: {status}")
        if success:
            print(f"  Config saved: vwap_params_{symbol.lower()}.json")
    
    print(f"\nTotal Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    print("="*80)
    
    print("\nNext Steps:")
    print("1. Review optimized parameters in JSON files")
    print("2. Run full 60-day backtest with optimized params")
    print("3. Load params in live bot: params = json.load(open('vwap_params_gbpjpy.json'))")

if __name__ == "__main__":
    main()
