"""
Ablation Ladder on Real Market Data
Same ladder as synthetic, but uses real OHLCV data
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from load_real_market_data import load_csv_ohlcv, load_from_traderlocker, prepare_data_for_backtest
from capital_allocation_ai.vwap_control_flip_strategy import StrategyParams, VWAPControlFlipStrategy
import time


def run_backtest(data, params):
    """Run backtest and return metrics."""
    strategy = VWAPControlFlipStrategy(params)
    equity = 10000.0
    trades = []
    
    # Event counters
    crosses = 0
    retests = 0
    confirmations = 0
    
    prev_close = None
    prev_vwap = None
    
    for idx, row in data.iterrows():
        try:
            result = strategy.update(
                current_price=row['close'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                timestamp=row.get('timestamp')
            )
            
            current_vwap = result['indicators'].get('vwap', 0)
            current_atr = result['indicators'].get('atr20', 0)
            
            # Count crosses
            if prev_close is not None and prev_vwap is not None and current_vwap > 0:
                if (row['close'] > current_vwap and prev_close <= prev_vwap) or \
                   (row['close'] < current_vwap and prev_close >= prev_vwap):
                    crosses += 1
                
                # Count retests
                tol = params.touch_tol_atr_frac * current_atr if current_atr > 0 else 0.001 * row['close']
                if (row['low'] <= current_vwap + tol and row['close'] > current_vwap) or \
                   (row['high'] >= current_vwap - tol and row['close'] < current_vwap):
                    retests += 1
            
            prev_close = row['close']
            prev_vwap = current_vwap
            
            # Count confirmations
            if result['signals']['enter_long'] or result['signals']['enter_short']:
                confirmations += 1
            
            if result['signals']['enter_long'] and strategy.position is None:
                entry_price = row['close']
                direction = 'long'
                stop_loss = result['signals'].get('stop_loss', entry_price * 0.99)
                trades.append({
                    'entry': entry_price,
                    'direction': direction,
                    'stop_loss': stop_loss,
                    'entry_time': row.get('timestamp', idx)
                })
                strategy.position = {'direction': direction, 'entry': entry_price, 'stop': stop_loss}
            
            elif result['signals']['enter_short'] and strategy.position is None:
                entry_price = row['close']
                direction = 'short'
                stop_loss = result['signals'].get('stop_loss', entry_price * 1.01)
                trades.append({
                    'entry': entry_price,
                    'direction': direction,
                    'stop_loss': stop_loss,
                    'entry_time': row.get('timestamp', idx)
                })
                strategy.position = {'direction': direction, 'entry': entry_price, 'stop': stop_loss}
            
            elif strategy.position is not None:
                # Check exit conditions
                exit_price = None
                if result['signals']['exit_long'] or result['signals']['exit_short']:
                    exit_price = row['close']
                elif strategy.position['direction'] == 'long' and row['low'] <= strategy.position['stop']:
                    exit_price = strategy.position['stop']
                elif strategy.position['direction'] == 'short' and row['high'] >= strategy.position['stop']:
                    exit_price = strategy.position['stop']
                
                if exit_price is not None:
                    trade = trades[-1]
                    if strategy.position['direction'] == 'long':
                        pnl = (exit_price - trade['entry']) / trade['entry']
                    else:
                        pnl = (trade['entry'] - exit_price) / trade['entry']
                    
                    trade['exit'] = exit_price
                    trade['exit_time'] = row.get('timestamp', idx)
                    trade['pnl'] = pnl
                    trade['return'] = pnl
                    strategy.position = None
        
        except Exception as e:
            print(f"Error at row {idx}: {e}")
            continue
    
    # Close any open trades
    if strategy.position is not None and len(trades) > 0:
        trade = trades[-1]
        if not trade.get('exit'):
            exit_price = data.iloc[-1]['close']
            if strategy.position['direction'] == 'long':
                pnl = (exit_price - trade['entry']) / trade['entry']
            else:
                pnl = (trade['entry'] - exit_price) / trade['entry']
            trade['exit'] = exit_price
            trade['exit_time'] = data.iloc[-1].get('timestamp', len(data)-1)
            trade['pnl'] = pnl
            trade['return'] = pnl
    
    # Calculate metrics
    completed_trades = [t for t in trades if 'pnl' in t]
    n_trades = len(completed_trades)
    
    if n_trades == 0:
        return {
            'crosses': crosses,
            'retests': retests,
            'confirmations': confirmations,
            'trades': 0,
            'kill_ratio': 0.0,
            'sharpe': 0.0,
            'max_dd': 0.0,
            'win_rate': 0.0,
            'return': 0.0,
            'trades_list': []
        }
    
    returns = [t['pnl'] for t in completed_trades]
    total_return = sum(returns)
    win_rate = sum(1 for r in returns if r > 0) / n_trades
    
    # Sharpe ratio
    if len(returns) > 1:
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0
    else:
        sharpe = 0.0
    
    # Max drawdown
    cumulative = np.cumsum(returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_dd = abs(np.min(drawdown)) if len(drawdown) > 0 else 0.0
    
    # Kill ratio
    kill_ratio = confirmations / n_trades if n_trades > 0 else 0.0
    
    return {
        'crosses': crosses,
        'retests': retests,
        'confirmations': confirmations,
        'trades': n_trades,
        'kill_ratio': kill_ratio,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'return': total_return,
        'trades_list': completed_trades
    }


def main():
    """Run ablation ladder on real data."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ablation ladder on real market data')
    parser.add_argument('--symbol', type=str, default='GBPJPY', help='Symbol to test')
    parser.add_argument('--days', type=int, default=30, help='Days of data')
    parser.add_argument('--timeframe', type=str, default='5m', help='Timeframe (5m, 15m, etc.)')
    parser.add_argument('--vol_filter', type=str, default='percentile', choices=['percentile', 'impulse', 'multiplier'], help='Volume filter type')
    parser.add_argument('--p', type=float, default=60.0, help='Percentile threshold (55-70)')
    parser.add_argument('--L', type=int, default=50, help='Percentile lookback window')
    parser.add_argument('--retest_min', type=int, default=2, help='Minimum retest count')
    parser.add_argument('--retest_max', type=int, default=4, help='Maximum retest count')
    parser.add_argument('--atr_cap', type=float, default=3.0, help='ATR cap multiplier')
    parser.add_argument('--session', type=int, default=0, choices=[0, 1], help='Session filter (0=off, 1=on)')
    parser.add_argument('--test', type=str, default='A', choices=['A', 'B', 'C'], help='Test preset (A=baseline, B=high freq, C=high quality)')
    
    args = parser.parse_args()
    
    # Apply test presets
    if args.test == 'A':
        # Baseline
        args.vol_filter = 'percentile'
        args.p = 60.0
        args.L = 50
        args.retest_min = 2
        args.retest_max = 4
        args.atr_cap = 3.0
        args.session = 0
    elif args.test == 'B':
        # Higher frequency
        args.vol_filter = 'percentile'
        args.p = 55.0
        args.L = 50
        args.retest_min = 1
        args.retest_max = 4
        args.atr_cap = 3.5
        args.session = 0
    elif args.test == 'C':
        # Higher quality
        args.vol_filter = 'percentile'
        args.p = 65.0
        args.L = 50
        args.retest_min = 2
        args.retest_max = 4
        args.atr_cap = 2.5
        args.session = 1
    
    print("="*80)
    print(f"REAL DATA TEST - {args.test} ({args.symbol}, {args.days} days)")
    print("="*80)
    print(f"Config:")
    print(f"  Volume Filter: {args.vol_filter} (p={args.p}, L={args.L})")
    print(f"  Retest Window: {args.retest_min}-{args.retest_max}")
    print(f"  ATR Cap: {args.atr_cap}")
    print(f"  Session Filter: {args.session}")
    print("="*80)
    
    symbol = args.symbol
    
    # Try to load real data - fail fast with explicit messages
    data = None
    
    # Option 1: Try CSV file (explicit paths)
    csv_candidates = [
        Path(f"{symbol.lower()}_5m.csv"),
        Path("data") / f"{symbol.lower()}_5m.csv",
        Path(f"{symbol}_5m.csv"),
        Path("data") / f"{symbol}_5m.csv",
    ]
    
    csv_path = next((p for p in csv_candidates if p.exists()), None)
    if csv_path:
        print(f"\n[DATA] Using CSV: {csv_path}")
        try:
            data = load_csv_ohlcv(str(csv_path))
            # Sanity print
            print(f"Loaded bars: {len(data)}")
            print(f"Range: {data['timestamp'].iloc[0]} -> {data['timestamp'].iloc[-1]}")
            print(f"Close min/max: {data['close'].min():.5f} / {data['close'].max():.5f}")
            print(f"Vol min/max: {data['volume'].min()} / {data['volume'].max()}")
            data = prepare_data_for_backtest(data, symbol)
        except Exception as e:
            print(f"[ERROR] Failed to load CSV {csv_path}: {e}")
            sys.exit(2)
    
    # Option 2: Try TraderLocker API
    elif os.getenv("TRADERLOCKER_API_KEY"):
        print("\n[DATA] Using TraderLocker API")
        try:
            data = load_from_traderlocker(symbol, days=args.days)
            data = prepare_data_for_backtest(data, symbol)
            print(f"TraderLocker fetch OK: {len(data)} bars")
        except Exception as e:
            print(f"[ERROR] Failed to load from TraderLocker: {e}")
            sys.exit(2)
    
    # No data source available
    else:
        print("\n" + "="*80)
        print("[ERROR] No data source found")
        print("="*80)
        print("Please provide:")
        print("1. CSV file with OHLCV data (timestamp, open, high, low, close, volume)")
        print(f"   Place in: {symbol.lower()}_5m.csv or data/{symbol.lower()}_5m.csv")
        print("2. OR set TRADERLOCKER_API_KEY environment variable")
        print("="*80)
        sys.exit(2)
    
    print(f"\nLoaded {len(data)} bars of real data")
    print(f"Total bars: {len(data)}")
    
    # Build params from arguments
    # Map retest_min/max to require_nth_retest (strategy uses flexible window if retest_count==3)
    retest_value = args.retest_max  # Strategy will use flexible window
    
    params = StrategyParams(
        band_k=2.0,
        volume_filter_type=args.vol_filter,
        vol_percentile_L=args.L,
        vol_percentile_p=args.p,
        vol_mult=1.0,  # For multiplier type
        atr_cap_mult=args.atr_cap,
        require_nth_retest=retest_value,  # Will use flexible window
        retest_min=args.retest_min,  # Custom flexible window
        retest_max=args.retest_max,
        require_session_filter=bool(args.session),
        cross_lookback_bars=20,
        touch_tol_atr_frac=0.20
    )
    
    print("\nRunning backtest...")
    start = time.time()
    result = run_backtest(data, params)
    elapsed = time.time() - start
    
    # Build result record
    result['rung'] = f"Test{args.test}"
    result['retest'] = f"{args.retest_min}-{args.retest_max}"
    result['vol_mult'] = f"{args.vol_filter}(p={args.p},L={args.L})"
    result['atr_cap'] = args.atr_cap
    result['session'] = bool(args.session)
    result['total_bars'] = len(data)
    result['time'] = elapsed
    
    # Output results
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    r = result
    print(f"Total Bars: {r['total_bars']}")
    print(f"Crosses: {r['crosses']}")
    print(f"Retests: {r['retests']}")
    print(f"Confirmations: {r['confirmations']}")
    print(f"Trades: {r['trades']}")
    print(f"Trades per day: {r['trades'] / args.days:.2f}")
    print(f"Win Rate: {r.get('win_rate', 0)*100:.1f}%")
    print(f"Total Return: {r.get('return', 0)*100:+.2f}%")
    print(f"Sharpe: {r['sharpe']:.2f}")
    print(f"Max DD: {r['max_dd']:.2%}")
    print(f"Payoff Ratio: {r.get('payoff_ratio', 0):.2f}")
    print(f"Time: {r['time']:.1f}s")
    
    # Calculate payoff ratio if we have trades
    payoff_ratio = 0.0
    if r['trades'] > 0 and 'trades_list' in r and len(r['trades_list']) > 0:
        trades_df = pd.DataFrame(r['trades_list'])
        if len(trades_df) > 0:
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if (trades_df['pnl'] > 0).any() else 0
            avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if (trades_df['pnl'] < 0).any() else 1
            payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0
            r['payoff_ratio'] = payoff_ratio
    else:
        r['payoff_ratio'] = 0.0
    
    # Acceptance criteria
    print("\n" + "="*80)
    print("ACCEPTANCE CRITERIA CHECK")
    print("="*80)
    print(f"Trades >= 40: {'[PASS]' if r['trades'] >= 40 else '[FAIL]'} ({r['trades']})")
    print(f"Sharpe < 8: {'[PASS]' if r['sharpe'] < 8 else '[FAIL]'} ({r['sharpe']:.2f})")
    print(f"Max DD < 20%: {'[PASS]' if r['max_dd'] < 0.20 else '[FAIL]'} ({r['max_dd']:.2%})")
    print(f"Win Rate 40-70%: {'[PASS]' if 0.40 <= r.get('win_rate', 0) <= 0.70 else '[FAIL]'} ({r.get('win_rate', 0)*100:.1f}%)")
    print(f"Payoff Ratio 0.8-2.5: {'[PASS]' if 0.8 <= r.get('payoff_ratio', 0) <= 2.5 else '[FAIL]'} ({r.get('payoff_ratio', 0):.2f})")
    
    # Parseable output table
    print("\n" + "="*80)
    print("PARSEABLE RESULTS TABLE")
    print("="*80)
    print("rung|retest_min-retest_max|vol_filter(p,L)|atr_cap|session|trades|sharpe|dd|total_bars")
    print("-"*95)
    vol_str = f"{args.vol_filter}(p={args.p},L={args.L})"
    session_str = '1' if args.session else '0'
    print(f"{r['rung']}|{args.retest_min}-{args.retest_max}|{vol_str}|{args.atr_cap:.1f}|{session_str}|{r['trades']}|{r['sharpe']:.2f}|{r['max_dd']:.2%}|{r['total_bars']}")
    
    print("="*80)


if __name__ == "__main__":
    main()
