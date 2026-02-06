"""
Run Backtest - 7 Days (Quick Validation)
Then scale up to 60 days once validated
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


def backtest(data: pd.DataFrame, symbol: str, params: StrategyParams):
    """Backtest strategy."""
    strategy = VWAPControlFlipStrategy(params)
    
    equity = 10000.0
    trades = []
    max_dd = 0.0
    peak = equity
    
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
            
            if result['signals']['enter_long'] and strategy.position is None:
                strategy.enter_long(row['close'])
            elif result['signals']['enter_short'] and strategy.position is None:
                strategy.enter_short(row['close'])
            elif result['signals']['exit'] and strategy.position is not None:
                pos = strategy.position
                pnl = (row['close'] - pos.entry_price) * (equity / pos.entry_price * 0.1) if pos.side == "long" else (pos.entry_price - row['close']) * (equity / pos.entry_price * 0.1)
                equity += pnl
                trades.append({'pnl': pnl})
                strategy.exit_position()
            
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0)
        except:
            continue
    
    if len(trades) == 0:
        return {'trades': 0, 'win_rate': 0, 'return': 0, 'sharpe': 0, 'trades_per_day': 0, 'max_dd': 0}
    
    trades_df = pd.DataFrame(trades)
    win_rate = (trades_df['pnl'] > 0).sum() / len(trades)
    total_return = (equity - 10000) / 10000
    returns = trades_df['pnl'] / 10000
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 1 else 0
    trades_per_day = len(trades) / (len(data) / 288)
    
    return {
        'trades': len(trades),
        'win_rate': win_rate,
        'return': total_return,
        'sharpe': sharpe,
        'trades_per_day': trades_per_day,
        'max_dd': max_dd
    }


def main():
    """Run short backtest."""
    print("="*80)
    print("VWAP CONTROL FLIP - SHORT BACKTEST (7 Days)")
    print("="*80)
    
    symbols = ['GBPJPY', 'BTCUSD', 'XAUUSD']
    params_map = {
        'GBPJPY': StrategyParams(band_k=2.1, vol_mult=1.4, atr_cap_mult=2.8, require_nth_retest=3, require_session_filter=False),
        'BTCUSD': StrategyParams(band_k=2.3, vol_mult=1.6, atr_cap_mult=3.0, require_nth_retest=3, require_session_filter=False),
        'XAUUSD': StrategyParams(band_k=1.9, vol_mult=1.3, atr_cap_mult=2.5, require_nth_retest=3, require_session_filter=False)
    }
    
    results = []
    for symbol in symbols:
        print(f"\n{symbol}: Generating data...")
        data = generate_market_data(symbol, days=7)
        print(f"  {len(data)} bars generated")
        
        print(f"  Running backtest...")
        start = time.time()
        stats = backtest(data, symbol, params_map[symbol])
        elapsed = time.time() - start
        
        print(f"  Results:")
        print(f"    Trades: {stats['trades']} ({stats['trades_per_day']:.2f}/day)")
        print(f"    Win Rate: {stats['win_rate']*100:.1f}%")
        print(f"    Return: {stats['return']*100:+.2f}%")
        print(f"    Sharpe: {stats['sharpe']:.2f}")
        print(f"    Max DD: {stats['max_dd']*100:.2f}%")
        print(f"    Time: {elapsed:.1f}s")
        
        results.append((symbol, stats))
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Symbol':<10} {'Trades':<8} {'Trades/Day':<12} {'Win Rate':<10} {'Return':<10} {'Sharpe':<8}")
    print("-"*80)
    for symbol, stats in results:
        print(f"{symbol:<10} {stats['trades']:<8} {stats['trades_per_day']:<12.2f} "
              f"{stats['win_rate']*100:<9.1f}% {stats['return']*100:<9.2f}% {stats['sharpe']:<8.2f}")
    
    print("\n" + "="*80)
    print("NOTE: This is a 7-day test. For full 60-day backtest,")
    print("run: python run_backtest_all_symbols.py")
    print("(May take 10-30 minutes due to large dataset)")
    print("="*80)


if __name__ == "__main__":
    main()
