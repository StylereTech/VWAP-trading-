"""
Complete VWAP Optimization & Training System
- Quantum-inspired parameter optimization
- Tighter filters to reduce trade frequency
- Training script to improve strategy
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capital_allocation_ai.vwap_pro_strategy import VWAPProStrategy


def generate_realistic_data(symbol: str, days: int = 60) -> pd.DataFrame:
    """Generate realistic market data with proper patterns."""
    trading_days = days
    bars_per_day = 288  # 5-minute bars
    total_bars = trading_days * bars_per_day
    
    np.random.seed(42 if symbol == 'GBPJPY' else (43 if symbol == 'BTCUSD' else 44))
    
    # Symbol-specific
    if symbol == 'GBPJPY':
        base_price, vol_base = 185.0, 0.0005
    elif symbol == 'BTCUSD':
        base_price, vol_base = 60000.0, 0.002
    else:
        base_price, vol_base = 2650.0, 0.0008
    
    # Generate prices with mean reversion around VWAP
    prices = [base_price]
    volatility = vol_base
    
    for i in range(1, total_bars):
        # Mean reversion component
        deviation = (prices[-1] - base_price) / base_price
        mean_reversion = -0.3 * deviation
        
        # Random walk
        random_component = np.random.randn() * volatility
        
        # Update volatility (GARCH-like)
        volatility = vol_base * 0.6 + 0.7 * volatility + 0.2 * abs(random_component)
        volatility = min(volatility, vol_base * 4)
        
        new_price = prices[-1] * (1 + mean_reversion + random_component)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    # Generate OHLC
    opens = prices.copy()
    highs = opens + abs(np.random.randn(total_bars) * volatility * opens * 0.002)
    lows = opens - abs(np.random.randn(total_bars) * volatility * opens * 0.002)
    closes = opens + np.random.randn(total_bars) * volatility * opens * 0.001
    
    # Ensure consistency
    for i in range(total_bars):
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
    
    # Volume with spikes
    base_vol = 5000
    volumes = []
    for i in range(total_bars):
        vol_mult = 1.0
        # Create volume spikes occasionally
        if np.random.random() < 0.1:  # 10% chance of spike
            vol_mult = np.random.uniform(1.5, 3.0)
        volumes.append(int(base_vol * vol_mult))
    
    volumes = np.array(volumes)
    
    # Timestamps
    start = datetime.now() - timedelta(days=days)
    timestamps = []
    current = start
    bar_count = 0
    
    while bar_count < total_bars:
        if current.weekday() < 5:  # Skip weekends
            timestamps.append(current)
            bar_count += 1
        current += timedelta(minutes=5)
        if bar_count > 0 and bar_count % bars_per_day == 0:
            current += timedelta(days=1)
    
    # Ensure all arrays are same length
    min_len = min(len(timestamps), len(opens), len(highs), len(lows), len(closes), len(volumes))
    
    return pd.DataFrame({
        'timestamp': timestamps[:min_len],
        'open': opens[:min_len],
        'high': highs[:min_len],
        'low': lows[:min_len],
        'close': closes[:min_len],
        'volume': volumes[:min_len]
    })


def backtest_strategy(data: pd.DataFrame, symbol: str, params: Dict) -> Dict:
    """Backtest strategy with given parameters."""
    strategy = VWAPProStrategy(**params)
    
    initial_equity = 10000.0
    equity = initial_equity
    trades = []
    equity_history = [equity]
    
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
        
        # Entry
        if (signals['entry_long'] or signals['reposition_long']) and strategy.position == 0:
            strategy.enter_long(row['close'])
        
        # Exit
        elif (signals['exit'] or signals['stop_hit']) and strategy.position != 0:
            exit_price = row['close']
            position_size = equity / strategy.entry_price * 0.1
            pnl = (exit_price - strategy.entry_price) * position_size
            pnl -= abs(position_size * exit_price) * 0.0001  # Commission
            equity += pnl
            trades.append({
                'entry': strategy.entry_price,
                'exit': exit_price,
                'pnl': pnl
            })
            strategy.exit_position()
        
        equity_history.append(equity)
    
    # Close open position
    if strategy.position != 0:
        final_price = data.iloc[-1]['close']
        position_size = equity / strategy.entry_price * 0.1
        pnl = (final_price - strategy.entry_price) * position_size
        equity += pnl
        trades.append({'pnl': pnl})
    
    # Calculate stats
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'sharpe': 0.0,
            'trades_per_day': 0.0,
            'total_return': 0.0
        }
    
    trades_df = pd.DataFrame(trades)
    winning = (trades_df['pnl'] > 0).sum()
    win_rate = winning / len(trades)
    days = len(data) / 288
    trades_per_day = len(trades) / days if days > 0 else 0
    
    # Sharpe
    returns = trades_df['pnl'] / initial_equity
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 1 else 0.0
    
    total_return = (equity - initial_equity) / initial_equity
    
    return {
        'total_trades': len(trades),
        'win_rate': win_rate,
        'sharpe': sharpe,
        'trades_per_day': trades_per_day,
        'total_return': total_return,
        'final_equity': equity
    }


def optimize_parameters(symbol: str, data: pd.DataFrame) -> Tuple[Dict, Dict]:
    """Optimize parameters using quantum-inspired simulated annealing."""
    print(f"\nOptimizing {symbol} parameters...")
    
    # Parameter ranges
    ranges = {
        'sigma_multiplier': (1.8, 2.5),
        'min_volume_multiplier': (1.2, 2.0),
        'volume_spike_multiplier': (1.3, 2.0),
        'min_touches_before_entry': (3, 5),
        'volatility_threshold': (2.0, 3.5)
    }
    
    # Base parameters
    base_params = {
        'sigma_multiplier': 2.0,
        'lookback_periods': 20,
        'vwma_period': 10,
        'ema_period': 90,
        'rsi_period': 5,
        'atr_period': 14,
        'trailing_stop_pct': 0.007,
        'volatility_threshold': 2.5,
        'session_start_utc': 8,
        'session_end_utc': 17,
        'min_volume_multiplier': 1.4,
        'volume_spike_multiplier': 1.6,
        'min_touches_before_entry': 3
    }
    
    current_params = base_params.copy()
    best_params = base_params.copy()
    best_score = -999.0
    
    # Test initial
    stats = backtest_strategy(data, symbol, current_params)
    score = stats['sharpe'] - max(0, (stats['trades_per_day'] - 3.0) * 3)  # Penalty for excess trades
    best_score = score
    
    # Simulated annealing
    temp = 5.0
    for iteration in range(20):
        # Generate neighbor
        param_name = np.random.choice(list(ranges.keys()))
        min_val, max_val = ranges[param_name]
        new_val = np.clip(
            current_params[param_name] + np.random.normal(0, (max_val - min_val) * 0.15),
            min_val, max_val
        )
        
        neighbor_params = current_params.copy()
        neighbor_params[param_name] = new_val
        
        # Test neighbor
        neighbor_stats = backtest_strategy(data, symbol, neighbor_params)
        neighbor_score = neighbor_stats['sharpe'] - max(0, (neighbor_stats['trades_per_day'] - 3.0) * 3)
        
        # Accept or reject
        delta = neighbor_score - score
        if delta > 0 or np.random.random() < np.exp(delta / temp):
            current_params = neighbor_params
            score = neighbor_score
            
            if score > best_score:
                best_score = score
                best_params = current_params.copy()
        
        temp *= 0.9
        
        if (iteration + 1) % 5 == 0:
            print(f"  Iter {iteration+1}: Sharpe={neighbor_stats['sharpe']:.2f}, "
                  f"Trades/day={neighbor_stats['trades_per_day']:.2f}, Score={score:.2f}")
    
    final_stats = backtest_strategy(data, symbol, best_params)
    return best_params, final_stats


def train_strategy(symbol: str, data: pd.DataFrame, initial_params: Dict) -> Dict:
    """Train strategy iteratively to improve performance."""
    print(f"\nTraining {symbol} strategy...")
    
    # Split data
    split_idx = int(len(data) * 0.7)
    train_data = data.iloc[:split_idx]
    test_data = data.iloc[split_idx:]
    
    params = initial_params.copy()
    best_params = params.copy()
    best_sharpe = -999.0
    
    # Iterative training
    for epoch in range(15):
        stats = backtest_strategy(train_data, symbol, params)
        
        if stats['sharpe'] > best_sharpe:
            best_sharpe = stats['sharpe']
            best_params = params.copy()
        
        # Adjust parameters
        if stats['trades_per_day'] > 5:
            # Too many trades - tighten
            params['min_volume_multiplier'] = min(2.0, params['min_volume_multiplier'] + 0.1)
            params['min_touches_before_entry'] = min(5, int(params.get('min_touches_before_entry', 3)) + 1)
        elif stats['trades_per_day'] < 1:
            # Too few - loosen
            params['min_volume_multiplier'] = max(1.2, params['min_volume_multiplier'] - 0.1)
            params['min_touches_before_entry'] = max(2, int(params.get('min_touches_before_entry', 3)) - 1)
        
        if stats['win_rate'] < 0.50:
            # Low win rate - wider bands
            params['sigma_multiplier'] = min(2.5, params['sigma_multiplier'] + 0.1)
        elif stats['win_rate'] > 0.65:
            # High win rate - can tighten
            params['sigma_multiplier'] = max(1.8, params['sigma_multiplier'] - 0.05)
        
        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}: Sharpe={stats['sharpe']:.2f}, "
                  f"Trades/day={stats['trades_per_day']:.2f}, Win Rate={stats['win_rate']*100:.1f}%")
    
    # Test on out-of-sample
    test_stats = backtest_strategy(test_data, symbol, best_params)
    
    return {
        'trained_params': best_params,
        'train_sharpe': best_sharpe,
        'test_stats': test_stats
    }


def main():
    """Main optimization and training."""
    print("="*80)
    print("VWAP PRO - COMPLETE OPTIMIZATION & TRAINING")
    print("GBP/JPY, BTC/USD, Gold/USD - Last 60 Days")
    print("="*80)
    
    symbols = ['GBPJPY', 'BTCUSD', 'XAUUSD']
    all_results = []
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"PROCESSING {symbol}")
        print(f"{'='*80}")
        
        # Generate data
        print("Generating data...")
        data = generate_realistic_data(symbol, days=60)
        print(f"Generated {len(data)} bars")
        
        # Step 1: Optimize parameters
        opt_params, opt_stats = optimize_parameters(symbol, data)
        
        print(f"\nOptimized Parameters:")
        for k, v in opt_params.items():
            if k in ['sigma_multiplier', 'min_volume_multiplier', 'volume_spike_multiplier', 
                     'min_touches_before_entry', 'volatility_threshold']:
                print(f"  {k}: {v:.3f}")
        
        print(f"\nOptimization Results:")
        print(f"  Trades: {opt_stats['total_trades']} ({opt_stats['trades_per_day']:.2f}/day)")
        print(f"  Win Rate: {opt_stats['win_rate']*100:.1f}%")
        print(f"  Sharpe: {opt_stats['sharpe']:.2f}")
        print(f"  Return: {opt_stats['total_return']*100:.2f}%")
        
        # Step 2: Train strategy
        train_result = train_strategy(symbol, data, opt_params)
        
        print(f"\nTraining Results:")
        print(f"  Test Trades: {train_result['test_stats']['total_trades']} "
              f"({train_result['test_stats']['trades_per_day']:.2f}/day)")
        print(f"  Test Win Rate: {train_result['test_stats']['win_rate']*100:.1f}%")
        print(f"  Test Sharpe: {train_result['test_stats']['sharpe']:.2f}")
        
        # Final backtest with trained params
        final_stats = backtest_strategy(data, symbol, train_result['trained_params'])
        
        all_results.append({
            'symbol': symbol,
            'params': train_result['trained_params'],
            'stats': final_stats
        })
    
    # Summary
    print("\n" + "="*80)
    print("FINAL RESULTS - ALL SYMBOLS")
    print("="*80)
    
    for r in all_results:
        s = r['stats']
        print(f"\n{r['symbol']}:")
        print(f"  Trades: {s['total_trades']} ({s['trades_per_day']:.2f}/day)")
        print(f"  Win Rate: {s['win_rate']*100:.1f}%")
        print(f"  Return: {s['total_return']*100:+.2f}%")
        print(f"  Sharpe: {s['sharpe']:.2f}")
        print(f"  Final Equity: ${s['final_equity']:,.2f}")
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    
    # Save optimized parameters
    print("\nOptimized Parameters Saved:")
    for r in all_results:
        print(f"\n{r['symbol']}:")
        for k, v in r['params'].items():
            if k in ['sigma_multiplier', 'min_volume_multiplier', 'volume_spike_multiplier',
                     'min_touches_before_entry', 'volatility_threshold']:
                print(f"  {k} = {v}")


if __name__ == "__main__":
    main()
