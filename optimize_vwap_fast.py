"""
Fast VWAP Optimization & Training
- Tighter filters to reduce trade frequency
- Quantum-inspired parameter optimization
- Strategy training to improve signals
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capital_allocation_ai.vwap_pro_strategy import VWAPProStrategy


def generate_data(symbol: str, days: int = 60) -> pd.DataFrame:
    """Generate market data."""
    trading_days = days
    bars_per_day = 288
    total_bars = trading_days * bars_per_day
    
    np.random.seed(42 if symbol == 'GBPJPY' else (43 if symbol == 'BTCUSD' else 44))
    
    if symbol == 'GBPJPY':
        base_price, vol_base = 185.0, 0.0005
    elif symbol == 'BTCUSD':
        base_price, vol_base = 60000.0, 0.002
    else:
        base_price, vol_base = 2650.0, 0.0008
    
    volatility = np.ones(total_bars) * vol_base
    for i in range(1, total_bars):
        volatility[i] = vol_base * 0.6 + 0.7 * volatility[i-1] + 0.2 * abs(np.random.randn() * vol_base)
    
    returns = np.random.randn(total_bars) * volatility
    prices = base_price + np.cumsum(returns)
    
    opens = prices.copy()
    highs = opens + abs(np.random.randn(total_bars) * volatility * opens * 2)
    lows = opens - abs(np.random.randn(total_bars) * volatility * opens * 2)
    closes = opens + np.random.randn(total_bars) * volatility * opens
    
    for i in range(total_bars):
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
    
    volumes = (5000 * (1 + abs(returns) * 10)).astype(int)
    volumes = np.clip(volumes, 1000, 50000)
    
    start_date = datetime.now() - timedelta(days=days)
    timestamps = [start_date + timedelta(minutes=5*i) for i in range(total_bars)]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })


def backtest_with_params(data: pd.DataFrame, symbol: str, params: Dict) -> Dict:
    """Backtest with given parameters."""
    strategy = VWAPProStrategy(**params)
    
    equity = 10000.0
    trades = []
    
    for idx, row in data.iterrows():
        result = strategy.update(
            current_price=row['close'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            timestamp=row.get('timestamp')
        )
        
        signals = result['signals']
        
        if signals['entry_long'] and strategy.position == 0:
            strategy.enter_long(row['close'])
        elif signals['reposition_long'] and strategy.position == 0:
            strategy.enter_long(row['close'])
        elif (signals['exit'] or signals['stop_hit']) and strategy.position != 0:
            exit_price = row['close']
            position_size = equity / strategy.entry_price * 0.1
            pnl = (exit_price - strategy.entry_price) * position_size if strategy.position == 1 else (strategy.entry_price - exit_price) * position_size
            pnl -= abs(position_size * exit_price) * 0.0001
            equity += pnl
            trades.append({'pnl': pnl})
            strategy.exit_position()
    
    if len(trades) == 0:
        return {'total_trades': 0, 'win_rate': 0.0, 'sharpe': 0.0, 'trades_per_day': 0.0}
    
    trades_df = pd.DataFrame(trades)
    win_rate = (trades_df['pnl'] > 0).sum() / len(trades)
    days = len(data) / 288
    trades_per_day = len(trades) / days
    
    # Simple Sharpe approximation
    avg_return = trades_df['pnl'].mean()
    std_return = trades_df['pnl'].std()
    sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0.0
    
    return {
        'total_trades': len(trades),
        'win_rate': win_rate,
        'sharpe': sharpe,
        'trades_per_day': trades_per_day,
        'avg_pnl': avg_return
    }


def optimize_quantum(symbol: str, data: pd.DataFrame) -> Dict:
    """Quantum-inspired optimization."""
    print(f"Optimizing {symbol}...")
    
    # Parameter ranges
    ranges = {
        'sigma_multiplier': (1.8, 2.5),
        'min_volume_multiplier': (1.2, 2.0),
        'volume_spike_multiplier': (1.3, 2.0),
        'min_touches_before_entry': (3, 5),
        'volatility_threshold': (2.0, 3.5)
    }
    
    best_params = None
    best_score = -999.0
    
    # Simulated annealing (quantum-inspired)
    current_params = {
        'sigma_multiplier': 2.0,
        'lookback_periods': 20,
        'min_volume_multiplier': 1.3,
        'volume_spike_multiplier': 1.6,
        'min_touches_before_entry': 3,
        'volatility_threshold': 2.5,
        'trailing_stop_pct': 0.007
    }
    
    temp = 10.0
    for iteration in range(30):  # Faster optimization
        # Test current
        stats = backtest_with_params(data, symbol, current_params)
        
        # Score: Sharpe - penalty for too many trades
        score = stats['sharpe'] - max(0, (stats['trades_per_day'] - 3.0) * 2)
        
        if score > best_score:
            best_score = score
            best_params = current_params.copy()
        
        # Generate neighbor
        param_to_change = np.random.choice(list(ranges.keys()))
        min_val, max_val = ranges[param_to_change]
        new_val = np.clip(
            current_params[param_to_change] + np.random.normal(0, (max_val - min_val) * 0.1),
            min_val, max_val
        )
        
        neighbor_params = current_params.copy()
        neighbor_params[param_to_change] = new_val
        
        neighbor_stats = backtest_with_params(data, symbol, neighbor_params)
        neighbor_score = neighbor_stats['sharpe'] - max(0, (neighbor_stats['trades_per_day'] - 3.0) * 2)
        
        # Accept or reject
        delta = neighbor_score - score
        if delta > 0 or np.random.random() < np.exp(delta / temp):
            current_params = neighbor_params
        
        temp *= 0.95
        
        if (iteration + 1) % 10 == 0:
            print(f"  Iter {iteration+1}: Sharpe={stats['sharpe']:.2f}, "
                  f"Trades/day={stats['trades_per_day']:.1f}, Score={score:.2f}")
    
    return best_params, best_score


def train_strategy(symbol: str, data: pd.DataFrame) -> Dict:
    """Train strategy iteratively."""
    print(f"Training {symbol}...")
    
    split_idx = int(len(data) * 0.7)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    # Start with optimized params
    params, _ = optimize_quantum(symbol, train_data)
    
    # Fine-tune on training data
    best_params = params.copy()
    best_sharpe = -999.0
    
    for epoch in range(20):
        stats = backtest_with_params(train_data, symbol, params)
        
        if stats['sharpe'] > best_sharpe:
            best_sharpe = stats['sharpe']
            best_params = params.copy()
        
        # Adjust based on performance
        if stats['trades_per_day'] > 4:
            params['min_volume_multiplier'] = min(2.0, params['min_volume_multiplier'] + 0.1)
            params['min_touches_before_entry'] = min(5, params['min_touches_before_entry'] + 1)
        elif stats['trades_per_day'] < 1:
            params['min_volume_multiplier'] = max(1.2, params['min_volume_multiplier'] - 0.1)
        
        if stats['win_rate'] < 0.50:
            params['sigma_multiplier'] = min(2.5, params['sigma_multiplier'] + 0.1)
    
    # Test on out-of-sample
    test_stats = backtest_with_params(test_data, symbol, best_params)
    
    return {
        'params': best_params,
        'train_sharpe': best_sharpe,
        'test_stats': test_stats
    }


def main():
    """Main optimization and training."""
    print("="*80)
    print("VWAP PRO - OPTIMIZATION & TRAINING")
    print("GBP/JPY, BTC/USD, Gold/USD - Last 60 Days")
    print("="*80)
    
    symbols = ['GBPJPY', 'BTCUSD', 'XAUUSD']
    results = []
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"{symbol}")
        print(f"{'='*80}")
        
        data = generate_data(symbol, days=60)
        
        # Optimize
        opt_params, opt_score = optimize_quantum(symbol, data)
        
        # Train
        train_result = train_strategy(symbol, data)
        
        # Final backtest
        final_stats = backtest_with_params(data, symbol, train_result['params'])
        
        print(f"\nFinal Results:")
        print(f"  Trades: {final_stats['total_trades']} ({final_stats['trades_per_day']:.1f}/day)")
        print(f"  Win Rate: {final_stats['win_rate']*100:.1f}%")
        print(f"  Sharpe: {final_stats['sharpe']:.2f}")
        
        results.append({
            'symbol': symbol,
            'params': train_result['params'],
            'stats': final_stats
        })
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    for r in results:
        print(f"{r['symbol']}: {r['stats']['trades_per_day']:.1f} trades/day, "
              f"{r['stats']['win_rate']*100:.1f}% win rate, "
              f"Sharpe: {r['stats']['sharpe']:.2f}")


if __name__ == "__main__":
    main()
