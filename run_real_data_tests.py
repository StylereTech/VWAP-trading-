"""
Run All 3 Real Data Tests (A, B, C)
Convenience script to run all tests sequentially
"""

import subprocess
import sys
import os

def run_test(test_name, symbol='GBPJPY', days=30):
    """Run a single test."""
    print("\n" + "="*80)
    print(f"RUNNING TEST {test_name}")
    print("="*80)
    
    cmd = [
        sys.executable,
        'run_ablation_real_data.py',
        '--symbol', symbol,
        '--days', str(days),
        '--test', test_name
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0

def main():
    """Run all 3 tests."""
    print("="*80)
    print("RUNNING ALL 3 REAL DATA TESTS")
    print("="*80)
    print("\nTest A: Baseline (percentile p=60, retest 2-4, ATR 3.0, session off)")
    print("Test B: Higher frequency (p=55, retest 1-4, ATR 3.5, session off)")
    print("Test C: Higher quality (p=65, retest 2-4, ATR 2.5, session on)")
    print("="*80)
    
    results = {}
    
    for test in ['A', 'B', 'C']:
        success = run_test(test)
        results[test] = success
        if not success:
            print(f"\n[WARN] Test {test} failed or timed out")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for test, success in results.items():
        status = "[OK]" if success else "[FAIL]"
        print(f"Test {test}: {status}")
    
    print("\nCheck output above for parseable results tables")
    print("="*80)

if __name__ == "__main__":
    main()
