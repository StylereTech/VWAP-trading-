"""
Backtest Using Optimized Parameters
Loads parameters from optimization_output_gbpjpy.json if available
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capital_allocation_ai.vwap_control_flip_strategy import VWAPControlFlipStrategy, StrategyParams

def load_optimized_config():
    """Load optimized config from JSON if available."""
    config_file = "optimization_output_gbpjpy.json"
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            chosen_config = data.get('chosen_config', {})
            print(f"  Loaded optimized config from {config_file}")
            return chosen_config
        except Exception as e:
            print(f"  Warning: Could not load {config_file}: {e}")
            return None
    else:
        print(f"  {config_file} not found, using baseline config")
        return None

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
                print(f"    [LONG] Entry at {row['close']:.2f} (bar {idx})")
            elif result['signals']['enter_short'] and strategy.position is None:
                strategy.enter_short(row['close'])
                print(f"    [SHORT] Entry at {row['close']:.2f} (bar {idx})")
            elif result['signals']['exit'] and strategy.position is not None:
                pos = strategy.position
                size = equity / pos.entry_price * 0.1
                pnl = (row['close'] - pos.entry_price) * size if pos.side == "long" else (pos.entry_price - row['close']) * size
                pnl -= abs(size * row['close']) * 0.0001
                equity += pnl
                trades.append({
                    'pnl': pnl,
                    'entry': pos.entry_price,
                    'exit': row['close'],
                    'side': pos.side
                })
                print(f"    [EXIT] {pos.side.upper()} PnL: {pnl:.2f} (Entry: {pos.entry_price:.2f}, Exit: {row['close']:.2f})")
                strategy.exit_position()
            
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0)
        except:
            continue
    
    if len(trades) == 0:
        return {
            'trades': 0, 'win_rate': 0, 'return': 0, 'sharpe': 0, 
            'trades_per_day': 0, 'max_dd': 0, 'total_pnl': 0, 'final_equity': equity
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
        'final_equity': equity
    }


def main():
    """Run backtest with optimized parameters."""
    print("="*80)
    print("VWAP CONTROL FLIP - BACKTEST WITH OPTIMIZED PARAMETERS")
    print("="*80)
    
    symbol = 'GBPJPY'
    
    # Try to load optimized config
    print("\nLoading configuration...")
    opt_config = load_optimized_config()
    
    if opt_config:
        # Map optimized config to StrategyParams
        params = StrategyParams(
            band_k=opt_config.get('sigma_level', 2.0),
            vol_mult=1.5,  # Not in optimization space, use default
            atr_cap_mult=opt_config.get('atr_cap_mult', 2.5),
            require_nth_retest=opt_config.get('retest_count', 3),
            require_session_filter=bool(opt_config.get('session_filter', 1)),
            cross_lookback_bars=12
        )
        print(f"  Using OPTIMIZED parameters:")
        print(f"    Sigma Level: {opt_config.get('sigma_level', 2.0)}")
        print(f"    Retest Count: {opt_config.get('retest_count', 3)}")
        print(f"    ATR Cap Mult: {opt_config.get('atr_cap_mult', 2.5)}")
        print(f"    Session Filter: {opt_config.get('session_filter', 1)}")
    else:
        # Use baseline config
        params = StrategyParams(
            band_k=2.0,
            vol_mult=1.5,
            atr_cap_mult=2.5,
            require_nth_retest=3,
            require_session_filter=False,
            cross_lookback_bars=12
        )
        print(f"  Using BASELINE parameters")
    
    print(f"\n{symbol}:")
    print(f"  Generating data...")
    data = generate_market_data(symbol, days=7)
    print(f"  {len(data)} bars generated")
    
    print(f"  Running backtest...")
    start = time.time()
    stats = backtest(data, symbol, params)
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
    
    print("\n" + "="*80)
    if opt_config:
        print("Using optimized parameters from QUBO optimization.")
    else:
        print("Using baseline parameters. Run optimization to get better parameters:")
        print("  python run_optimization_with_progress.py")
    print("="*80)


if __name__ == "__main__":
    main()
