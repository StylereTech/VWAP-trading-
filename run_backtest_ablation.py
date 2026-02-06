"""
Filter Ablation Ladder - Diagnose Which Filter Kills Trades
Tracks: cross_events, retest_events, confirm_events, trades
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capital_allocation_ai.vwap_control_flip_strategy import VWAPControlFlipStrategy, StrategyParams


def generate_market_data(symbol: str, days: int = 7) -> pd.DataFrame:
    """Generate market data."""
    bars_per_day = 288
    total_bars = days * bars_per_day
    
    np.random.seed(42 if symbol == 'GBPJPY' else (43 if symbol == 'BTCUSD' else 44))
    
    if symbol == 'GBPJPY':
        base_price, vol_base = 185.0, 0.0005
    elif symbol == 'BTCUSD':
        base_price, vol_base = 60000.0, 0.002
    else:
        base_price, vol_base = 2650.0, 0.0008
    
    volatility = np.full(total_bars, vol_base)
    returns = np.random.randn(total_bars) * volatility
    prices = base_price + np.cumsum(returns)
    
    if symbol == 'GBPJPY':
        prices = np.clip(prices, 175, 200)
    elif symbol == 'BTCUSD':
        prices = np.clip(prices, 50000, 70000)
    else:
        prices = np.clip(prices, 2550, 2750)
    
    opens = prices.copy()
    highs = opens + abs(np.random.randn(total_bars) * 0.002 * opens)
    lows = opens - abs(np.random.randn(total_bars) * 0.002 * opens)
    closes = opens + np.random.randn(total_bars) * 0.001 * opens
    
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))
    
    base_vol = 5000 if symbol != 'BTCUSD' else 100
    volumes = (base_vol * (1 + abs(returns) * 5)).astype(int)
    volumes = np.clip(volumes, base_vol * 0.2, base_vol * 10)
    
    start_date = datetime.now() - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, periods=total_bars, freq='5min')
    timestamps = [ts for ts in timestamps if ts.weekday() < 5][:total_bars]
    
    min_len = min(len(timestamps), len(opens))
    
    return pd.DataFrame({
        'timestamp': timestamps[:min_len],
        'open': opens[:min_len],
        'high': highs[:min_len],
        'low': lows[:min_len],
        'close': closes[:min_len],
        'volume': volumes[:min_len]
    })


def backtest_with_diagnostics(data: pd.DataFrame, symbol: str, params: StrategyParams, test_name: str):
    """Backtest with full event tracking."""
    strategy = VWAPControlFlipStrategy(params)
    
    equity = 10000.0
    trades = []
    max_dd = 0.0
    peak = equity
    
    # Event counters - track what's happening
    cross_events = 0
    retest_events = 0
    confirm_events = 0
    entered_trades = 0
    
    # Track previous state to detect events
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
            
            # Get current VWAP from indicators
            current_vwap = result['indicators'].get('vwap', 0)
            current_atr = result['indicators'].get('atr20', 0)
            
            # Detect cross events (by comparing current vs previous)
            if prev_close is not None and prev_vwap is not None:
                # Cross above VWAP
                if row['close'] > current_vwap and prev_close <= prev_vwap:
                    cross_events += 1
                # Cross below VWAP
                elif row['close'] < current_vwap and prev_close >= prev_vwap:
                    cross_events += 1
                
                # Detect retest events (using tolerance)
                tol = params.touch_tol_atr_frac * current_atr if current_atr > 0 else 0.001 * row['close']
                
                # Bullish retest: low touches VWAP from below, close above VWAP
                if row['low'] <= current_vwap + tol and row['close'] > current_vwap:
                    retest_events += 1
                
                # Bearish retest: high touches VWAP from above, close below VWAP
                if row['high'] >= current_vwap - tol and row['close'] < current_vwap:
                    retest_events += 1
            
            prev_close = row['close']
            prev_vwap = current_vwap
            
            # Track signal generation (confirmation events)
            if result['signals']['enter_long'] or result['signals']['enter_short']:
                confirm_events += 1
            
            if result['signals']['enter_long']:
                if strategy.position is None:
                    strategy.enter_long(row['close'])
                    entered_trades += 1
                    trades.append({
                        'pnl': 0,
                        'entry': row['close'],
                        'side': 'long',
                        'bar': idx
                    })
            elif result['signals']['enter_short']:
                if strategy.position is None:
                    strategy.enter_short(row['close'])
                    entered_trades += 1
                    trades.append({
                        'pnl': 0,
                        'entry': row['close'],
                        'side': 'short',
                        'bar': idx
                    })
            elif result['signals']['exit'] and strategy.position is not None:
                pos = strategy.position
                size = equity / pos.entry_price * 0.1
                pnl = (row['close'] - pos.entry_price) * size if pos.side == "long" else (pos.entry_price - row['close']) * size
                pnl -= abs(size * row['close']) * 0.0001
                equity += pnl
                
                if trades:
                    trades[-1]['pnl'] = pnl
                    trades[-1]['exit'] = row['close']
                
                strategy.exit_position()
            
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0)
        except Exception as e:
            continue
    
    # Calculate metrics
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        win_rate = (trades_df['pnl'] > 0).sum() / len(trades)
        total_return = (equity - 10000) / 10000
        returns = trades_df['pnl'] / 10000
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0
        trades_per_day = len(trades) / (len(data) / 288)
    else:
        win_rate = 0
        total_return = 0
        sharpe = 0
        trades_per_day = 0
    
    return {
        'test_name': test_name,
        'trades': len(trades),
        'cross_events': cross_events,
        'retest_events': retest_events,
        'confirm_events': confirm_events,
        'entered_trades': entered_trades,
        'win_rate': win_rate,
        'return': total_return,
        'sharpe': sharpe,
        'trades_per_day': trades_per_day,
        'max_dd': max_dd,
        'final_equity': equity
    }


def main():
    """Run ablation ladder."""
    print("="*80)
    print("FILTER ABLATION LADDER")
    print("Diagnosing Which Filter Kills Trades")
    print("="*80)
    
    symbol = 'GBPJPY'
    days = 7
    
    print(f"\nSymbol: {symbol}")
    print(f"Days: {days}")
    print(f"Generating data...")
    data = generate_market_data(symbol, days=days)
    print(f"{len(data)} bars generated")
    
    # Ablation ladder - run in order
    tests = [
        {
            'name': '1. Core Flip Only (NO filters, retest=1)',
            'params': StrategyParams(
                band_k=2.0,
                vol_mult=1.0,  # Disabled (multiplier = 1.0 means no filter)
                atr_cap_mult=999.0,  # Effectively disabled
                require_nth_retest=1,  # 1st retest only
                require_session_filter=False,
                cross_lookback_bars=20,
                touch_tol_atr_frac=0.20  # Loose tolerance (20% of ATR)
            )
        },
        {
            'name': '2. Add Retest Count = 2',
            'params': StrategyParams(
                band_k=2.0,
                vol_mult=1.0,
                atr_cap_mult=999.0,
                require_nth_retest=2,
                require_session_filter=False,
                cross_lookback_bars=20,
                touch_tol_atr_frac=0.20
            )
        },
        {
            'name': '3. Add Retest Count = 3',
            'params': StrategyParams(
                band_k=2.0,
                vol_mult=1.0,
                atr_cap_mult=999.0,
                require_nth_retest=3,
                require_session_filter=False,
                cross_lookback_bars=20,
                touch_tol_atr_frac=0.20
            )
        },
        {
            'name': '4. Add ATR Cap = 4.0',
            'params': StrategyParams(
                band_k=2.0,
                vol_mult=1.0,
                atr_cap_mult=4.0,
                require_nth_retest=3,
                require_session_filter=False,
                cross_lookback_bars=20,
                touch_tol_atr_frac=0.20
            )
        },
        {
            'name': '5. Add ATR Cap = 3.0',
            'params': StrategyParams(
                band_k=2.0,
                vol_mult=1.0,
                atr_cap_mult=3.0,
                require_nth_retest=3,
                require_session_filter=False,
                cross_lookback_bars=20,
                touch_tol_atr_frac=0.20
            )
        },
        {
            'name': '6. Add ATR Cap = 2.5',
            'params': StrategyParams(
                band_k=2.0,
                vol_mult=1.0,
                atr_cap_mult=2.5,
                require_nth_retest=3,
                require_session_filter=False,
                cross_lookback_bars=20,
                touch_tol_atr_frac=0.20
            )
        },
        {
            'name': '7. Add Volume Filter = 1.1',
            'params': StrategyParams(
                band_k=2.0,
                vol_mult=1.1,
                atr_cap_mult=2.5,
                require_nth_retest=3,
                require_session_filter=False,
                cross_lookback_bars=20,
                touch_tol_atr_frac=0.20
            )
        },
        {
            'name': '8. Add Volume Filter = 1.3',
            'params': StrategyParams(
                band_k=2.0,
                vol_mult=1.3,
                atr_cap_mult=2.5,
                require_nth_retest=3,
                require_session_filter=False,
                cross_lookback_bars=20,
                touch_tol_atr_frac=0.20
            )
        },
        {
            'name': '9. Add Volume Filter = 1.5 (FULL)',
            'params': StrategyParams(
                band_k=2.0,
                vol_mult=1.5,
                atr_cap_mult=2.5,
                require_nth_retest=3,
                require_session_filter=False,
                cross_lookback_bars=20,
                touch_tol_atr_frac=0.20
            )
        }
    ]
    
    results = []
    
    for test in tests:
        print(f"\n{'='*80}")
        print(f"Running: {test['name']}")
        print(f"{'='*80}")
        print(f"  Parameters:")
        print(f"    Volume Mult: {test['params'].vol_mult}")
        print(f"    ATR Cap: {test['params'].atr_cap_mult}")
        print(f"    Retest Count: {test['params'].require_nth_retest}")
        print(f"    Touch Tolerance: {test['params'].touch_tol_atr_frac * 100:.0f}% of ATR")
        print(f"    Session Filter: {test['params'].require_session_filter}")
        
        start = time.time()
        stats = backtest_with_diagnostics(data, symbol, test['params'], test['name'])
        elapsed = time.time() - start
        
        stats['time'] = elapsed
        results.append(stats)
        
        print(f"\n  Results:")
        print(f"    Cross Events: {stats['cross_events']}")
        print(f"    Retest Events: {stats['retest_events']}")
        print(f"    Confirm Events: {stats['confirm_events']}")
        print(f"    Entered Trades: {stats['entered_trades']}")
        print(f"    Trades: {stats['trades']}")
        print(f"    Time: {elapsed:.1f}s")
        
        if stats['trades'] > 0:
            print(f"    Win Rate: {stats['win_rate']*100:.1f}%")
            print(f"    Return: {stats['return']*100:+.2f}%")
            print(f"    Sharpe: {stats['sharpe']:.2f}")
    
    # Summary
    print("\n" + "="*80)
    print("ABLATION SUMMARY")
    print("="*80)
    print(f"{'Test':<45} {'Cross':<8} {'Retest':<8} {'Confirm':<8} {'Trades':<8}")
    print("-"*80)
    for r in results:
        print(f"{r['test_name']:<45} {r['cross_events']:<8} {r['retest_events']:<8} {r['confirm_events']:<8} {r['trades']:<8}")
    
    print("\n" + "="*80)
    print("DIAGNOSIS:")
    print("="*80)
    
    # Find where trades drop to 0
    first_zero = None
    for i, r in enumerate(results):
        if r['trades'] == 0 and first_zero is None:
            first_zero = i
            if i > 0:
                print(f"Trades dropped to 0 at: {r['test_name']}")
                print(f"Previous test had {results[i-1]['trades']} trades")
                print(f"Previous: Cross={results[i-1]['cross_events']}, Retest={results[i-1]['retest_events']}, Confirm={results[i-1]['confirm_events']}")
                print(f"Current: Cross={r['cross_events']}, Retest={r['retest_events']}, Confirm={r['confirm_events']}")
                print(f"Likely culprit: Filter added in this step")
            else:
                print(f"Trades = 0 from the start: {r['test_name']}")
                print(f"Cross Events: {r['cross_events']}")
                print(f"Retest Events: {r['retest_events']}")
                print(f"Confirm Events: {r['confirm_events']}")
                if r['cross_events'] == 0:
                    print("NO VWAP CROSSES DETECTED - Check VWAP calculation or data patterns")
                elif r['retest_events'] == 0:
                    print("NO RETESTS DETECTED - Check retest logic or tolerance")
                elif r['confirm_events'] == 0:
                    print("NO CONFIRMATIONS - Filters are blocking all signals")
    
    if first_zero is None:
        print("All tests generated trades - filters are not the issue")
        print("Check signal logic or data patterns")
    
    print("="*80)


if __name__ == "__main__":
    main()
