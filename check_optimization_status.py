"""
Check Optimization Status and Progress Monitor
"""

import os
import sys
import json
import time
from datetime import datetime

def check_status():
    """Check if optimization has completed."""
    output_file = "optimization_output_gbpjpy.json"
    
    print("="*80)
    print("OPTIMIZATION STATUS CHECK")
    print("="*80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    if os.path.exists(output_file):
        print(f"[OK] Output file found: {output_file}")
        file_size = os.path.getsize(output_file)
        print(f"   File size: {file_size:,} bytes")
        
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            print("\n[RESULTS] OPTIMIZATION RESULTS:")
            print(f"   Symbol: {data.get('symbol', 'N/A')}")
            print(f"   Date: {data.get('optimization_date', 'N/A')}")
            print(f"   Method: {data.get('method', 'N/A')}")
            print(f"   Samples: {data.get('n_samples', 'N/A')}")
            print(f"   Seed: {data.get('seed', 'N/A')}")
            print(f"   Lambda: {data.get('onehot_lambda_used', 'N/A')}")
            
            print("\n[CONFIG] CHOSEN CONFIG:")
            chosen = data.get('chosen_config', {})
            for k, v in chosen.items():
                print(f"   {k}: {v}")
            
            print("\n[METRICS] TRAIN METRICS:")
            train = data.get('train_metrics', {})
            print(f"   Sharpe: {train.get('sharpe', 0):.4f}")
            print(f"   Max DD: {train.get('max_drawdown', 0):.4f}")
            print(f"   Trades: {train.get('trades', 0)}")
            print(f"   Win Rate: {train.get('win_rate', 0):.2%}")
            
            print("\n[VALIDATION] VALIDATION METRICS:")
            wf = data.get('walk_forward', {})
            print(f"   Train Sharpe: {wf.get('train_sharpe', 0):.4f}")
            print(f"   Test Sharpe: {wf.get('test_sharpe', 0):.4f}")
            print(f"   Sharpe Drop: {wf.get('sharpe_drop', 0):.4f}")
            print(f"   DD Stable: {wf.get('dd_stable', False)}")
            print(f"   Trades Stable: {wf.get('trades_stable', False)}")
            
            print("\n[SURROGATE] SURROGATE QUALITY:")
            sq = data.get('surrogate_quality', {})
            print(f"   R2: {sq.get('r2', 0):.4f}")
            print(f"   Spearman: {sq.get('spearman_correlation', 0):.4f}")
            print(f"   P-value: {sq.get('spearman_pvalue', 1):.4f}")
            
            print("\n[CONSTRAINTS] CONSTRAINTS:")
            print(f"   Valid: {data.get('constraint_valid', False)}")
            
            print("\n[BASELINE] BASELINE COMPARISON:")
            bc = data.get('baseline_comparison', {})
            if bc:
                print(f"   Baseline Sharpe: {bc.get('baseline_sharpe', 0):.4f}")
                print(f"   Optimized Sharpe: {bc.get('optimized_sharpe', 0):.4f}")
                print(f"   Improvement: {bc.get('sharpe_improvement', 0):.4f}")
            
            print("\n[TOP-K] TOP-K CONFIGS:")
            top_k = data.get('top_k_configs', [])
            for i, cfg in enumerate(top_k[:3], 1):
                print(f"   Rank {i}: Sharpe={cfg.get('sharpe', 0):.4f}, Trades={cfg.get('trades', 0)}")
            
            print("\n" + "="*80)
            print("✅ OPTIMIZATION COMPLETE!")
            print("="*80)
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"[WARN] File exists but JSON is invalid: {e}")
            return False
        except Exception as e:
            print(f"[WARN] Error reading file: {e}")
            return False
    else:
        print(f"[WAIT] Output file not found: {output_file}")
        print("   Optimization may still be running...")
        
        # Check for Python processes
        print("\n[CHECK] Checking for running Python processes...")
        try:
            import subprocess
            result = subprocess.run(
                ["powershell", "-Command", "Get-Process python -ErrorAction SilentlyContinue | Select-Object Id,CPU,WorkingSet"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.stdout.strip():
                print("   Active Python processes found:")
                print(result.stdout)
            else:
                print("   No active Python processes found")
        except:
            print("   Could not check processes")
        
        return False

def monitor_progress(interval=30, max_checks=20):
    """Monitor optimization progress."""
    print("="*80)
    print("PROGRESS MONITOR")
    print("="*80)
    print(f"Checking every {interval} seconds (max {max_checks} checks)...")
    print("Press Ctrl+C to stop monitoring")
    print("="*80)
    
    for i in range(max_checks):
        print(f"\n[{i+1}/{max_checks}] Check at {datetime.now().strftime('%H:%M:%S')}")
        
        if check_status():
            print("\n[OK] Optimization completed!")
            break
        
        if i < max_checks - 1:
            print(f"\nWaiting {interval} seconds...")
            time.sleep(interval)
    else:
        print("\n[WAIT] Monitoring stopped. Optimization may still be running.")
        print("   Check manually later or increase max_checks")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check optimization status")
    parser.add_argument("--monitor", action="store_true", help="Monitor progress continuously")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--max-checks", type=int, default=20, help="Maximum number of checks")
    
    args = parser.parse_args()
    
    if args.monitor:
        monitor_progress(args.interval, args.max_checks)
    else:
        check_status()
