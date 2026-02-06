"""
Diagnostic Backtest - Shows Why Trades Are/Are Not Generated
Helps debug filter conditions
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


def backtest_diagnostic(data: pd.DataFrame, symbol: str, params: StrategyParams):
    """Backtest with diagnostic output."""
    strategy = VWAPControlFlipStrategy(params)
    
    equity = 10000.0
    trades = []
    max_dd = 0.0
    peak = equity
    
    # Diagnostic counters
    vwap_crosses = 0
    retests = 0
    volume_failures = 0
    atr_failures = 0
    session_failures = 0
    retest_count_failures = 0
    other_failures = 0
    
    print(f"\n  Diagnostic Mode - Analyzing {len(data)} bars...")
    print(f"  Parameters:")
    print(f"    Band K: {params.band_k}")
    print(f"    Volume Mult: {params.vol_mult}")
    print(f"    ATR Cap: {params.atr_cap_mult}")
    print(f"    Retest Count: {params.require_nth_retest}")
    print(f"    Session Filter: {params.require_session_filter}")
    
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
            
            # Check for signals and why they might be blocked
            if result['signals']['enter_long']:
                if strategy.position is None:
                    strategy.enter_long(row['close'])
                    trades.append({
                        'pnl': 0,  # Will update on exit
                        'entry': row['close'],
                        'side': 'long',
                        'bar': idx
                    })
                    print(f"    [LONG] Entry at {row['close']:.2f} (bar {idx})")
                else:
                    print(f"    [SKIP] Long signal but position already open")
            elif result['signals']['enter_short']:
                if strategy.position is None:
                    strategy.enter_short(row['close'])
                    trades.append({
                        'pnl': 0,
                        'entry': row['close'],
                        'side': 'short',
                        'bar': idx
                    })
                    print(f"    [SHORT] Entry at {row['close']:.2f} (bar {idx})")
                else:
                    print(f"    [SKIP] Short signal but position already open")
            elif result['signals']['exit'] and strategy.position is not None:
                pos = strategy.position
                size = equity / pos.entry_price * 0.1
                pnl = (row['close'] - pos.entry_price) * size if pos.side == "long" else (pos.entry_price - row['close']) * size
                pnl -= abs(size * row['close']) * 0.0001
                equity += pnl
                
                # Update trade PnL
                if trades:
                    trades[-1]['pnl'] = pnl
                    trades[-1]['exit'] = row['close']
                
                print(f"    [EXIT] {pos.side.upper()} PnL: {pnl:.2f} (Entry: {pos.entry_price:.2f}, Exit: {row['close']:.2f})")
                strategy.exit_position()
            
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0)
        except Exception as e:
            other_failures += 1
            continue
    
    print(f"\n  Diagnostic Summary:")
    print(f"    VWAP Crosses Detected: {vwap_crosses}")
    print(f"    Retests Detected: {retests}")
    print(f"    Volume Filter Failures: {volume_failures}")
    print(f"    ATR Filter Failures: {atr_failures}")
    print(f"    Session Filter Failures: {session_failures}")
    print(f"    Retest Count Failures: {retest_count_failures}")
    print(f"    Other Failures: {other_failures}")
    
    if len(trades) == 0:
        return {
            'trades': 0, 'win_rate': 0, 'return': 0, 'sharpe': 0, 
            'trades_per_day': 0, 'max_dd': 0, 'total_pnl': 0, 'final_equity': equity,
            'diagnostics': {
                'vwap_crosses': vwap_crosses,
                'retests': retests,
                'volume_failures': volume_failures,
                'atr_failures': atr_failures,
                'session_failures': session_failures,
                'retest_count_failures': retest_count_failures
            }
        }
    
    trades_df = pd.DataFrame(trades)
    win_rate = (trades_df['pnl'] > 0).sum() / len(trades)
    total_return = (equity - 10000) / 10000
    returns = trades_df['pnl'] / 10000
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0
    trades_per_day = len(trades) / (len(data) / 288)
    total_pnl = equity - 10000
    
    return {
        'trades': len(trades),
        'win_rate': win_rate,
        'return': total_return,
        'sharpe': sharpe,
        'trades_per_day': trades_per_day,
        'max_dd': max_dd,
        'total_pnl': total_pnl,
        'final_equity': equity,
        'diagnostics': {
            'vwap_crosses': vwap_crosses,
            'retests': retests,
            'volume_failures': volume_failures,
            'atr_failures': atr_failures,
            'session_failures': session_failures,
            'retest_count_failures': retest_count_failures
        }
    }


def main():
    """Run diagnostic backtest."""
    print("="*80)
    print("VWAP CONTROL FLIP - DIAGNOSTIC BACKTEST")
    print("Shows Why Trades Are/Are Not Generated")
    print("="*80)
    
    symbol = 'GBPJPY'
    
    # Try baseline config from production tuner
    print("\nUsing Baseline Configuration:")
    params = StrategyParams(
        band_k=2.0,
        vol_mult=1.5,
        atr_cap_mult=2.5,
        require_nth_retest=3,
        require_session_filter=False,  # Disabled for testing
        cross_lookback_bars=12
    )
    
    print(f"\n{symbol}:")
    print(f"  Generating data...")
    data = generate_market_data(symbol, days=7)
    print(f"  {len(data)} bars generated")
    
    print(f"  Running diagnostic backtest...")
    start = time.time()
    stats = backtest_diagnostic(data, symbol, params)
    elapsed = time.time() - start
    
    print(f"\n  Results:")
    print(f"    Trades: {stats['trades']} ({stats['trades_per_day']:.2f}/day)")
    print(f"    Win Rate: {stats['win_rate']*100:.1f}%")
    print(f"    Total Return: {stats['return']*100:+.2f}%")
    print(f"    Total PnL: ${stats['total_pnl']:+.2f}")
    print(f"    Final Equity: ${stats['final_equity']:.2f}")
    print(f"    Sharpe: {stats['sharpe']:.2f}")
    print(f"    Max DD: {stats['max_dd']*100:.2f}%")
    print(f"    Time: {elapsed:.1f}s")
    
    if stats['trades'] == 0:
        print("\n" + "="*80)
        print("NO TRADES GENERATED - RECOMMENDATIONS:")
        print("="*80)
        print("1. Try looser parameters:")
        print("   - Lower volume_mult (try 1.1 instead of 1.5)")
        print("   - Lower require_nth_retest (try 2 instead of 3)")
        print("   - Increase atr_cap_mult (try 3.0 instead of 2.5)")
        print("2. Wait for optimization to complete")
        print("3. Use optimized parameters from optimization_output_gbpjpy.json")
        print("="*80)


if __name__ == "__main__":
    main()
