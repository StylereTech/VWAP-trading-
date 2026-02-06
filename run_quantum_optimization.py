"""
Run Quantum Optimization for VWAP Control Flip Strategy
Optimizes parameters to improve R:R and win rate
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capital_allocation_ai.vwap_control_flip_strategy import VWAPControlFlipStrategy, StrategyParams
from capital_allocation_ai.quantum_optimizer import QuantumInspiredOptimizer


def generate_test_data(symbol: str, days: int = 14) -> pd.DataFrame:
    """Generate test data for optimization (smaller dataset)."""
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
    timestamps = []
    current = start_date
    bar_count = 0
    
    while bar_count < total_bars:
        if current.weekday() < 5:
            timestamps.append(current)
            bar_count += 1
        current += timedelta(minutes=5)
        if bar_count > 0 and bar_count % bars_per_day == 0:
            current += timedelta(days=1)
    
    min_len = min(len(timestamps), len(opens))
    
    return pd.DataFrame({
        'timestamp': timestamps[:min_len],
        'open': opens[:min_len],
        'high': highs[:min_len],
        'low': lows[:min_len],
        'close': closes[:min_len],
        'volume': volumes[:min_len]
    })


def quick_backtest(data: pd.DataFrame, symbol: str, params: StrategyParams) -> dict:
    """Quick backtest for optimization."""
    strategy = VWAPControlFlipStrategy(params)
    
    equity = 10000.0
    trades = []
    
    # Process every 10th bar for speed
    for idx, row in data.iloc[::10].iterrows():
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
                trades.append({'pnl': pnl, 'entry': pos.entry_price, 'exit': row['close']})
                strategy.exit_position()
        except:
            continue
    
    if len(trades) == 0:
        return {'sharpe': -10, 'trades_per_day': 0, 'win_rate': 0, 'rr_ratio': 0}
    
    trades_df = pd.DataFrame(trades)
    win_rate = (trades_df['pnl'] > 0).sum() / len(trades)
    
    # Calculate R:R ratio
    wins = trades_df[trades_df['pnl'] > 0]['pnl']
    losses = trades_df[trades_df['pnl'] < 0]['pnl']
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = abs(losses.mean()) if len(losses) > 0 else 1
    rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    
    returns = trades_df['pnl'] / 10000
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else -10
    
    days_traded = len(data) / 288
    trades_per_day = len(trades) / days_traded if days_traded > 0 else 0
    
    return {
        'sharpe': sharpe,
        'trades_per_day': trades_per_day,
        'win_rate': win_rate,
        'rr_ratio': rr_ratio
    }


def optimize_symbol(symbol: str):
    """Optimize parameters for a symbol."""
    print(f"\n{'='*60}")
    print(f"Optimizing {symbol}")
    print(f"{'='*60}")
    
    # Generate test data
    print("Generating test data...")
    data = generate_test_data(symbol, days=14)  # 14 days for faster optimization
    print(f"Generated {len(data)} bars")
    
    # Parameter ranges
    param_ranges = {
        'band_k': (1.8, 2.5),
        'vol_mult': (1.1, 1.8),
        'atr_cap_mult': (2.0, 3.5),
        'require_nth_retest': (2, 4),
        'touch_tol_atr_frac': (0.03, 0.08),
        'trail_pct': (0.005, 0.012),
        'cross_lookback_bars': (10, 20)
    }
    
    def cost_function(params_dict: dict) -> float:
        """Cost function: maximize Sharpe + R:R, minimize trade frequency."""
        try:
            params = StrategyParams(
                band_k=params_dict['band_k'],
                band_k_list=(1.0, params_dict['band_k'], 3.0),
                vol_mult=params_dict['vol_mult'],
                atr_cap_mult=params_dict['atr_cap_mult'],
                require_nth_retest=int(params_dict['require_nth_retest']),
                touch_tol_atr_frac=params_dict['touch_tol_atr_frac'],
                trail_pct=params_dict['trail_pct'],
                cross_lookback_bars=int(params_dict['cross_lookback_bars']),
                require_session_filter=False
            )
            
            stats = quick_backtest(data, symbol, params)
            
            # Penalties
            trade_penalty = max(0, (stats['trades_per_day'] - 3.0) * 2) if stats['trades_per_day'] > 3.0 else 0
            sharpe_penalty = abs(stats['sharpe']) * 5 if stats['sharpe'] < 0 else 0
            rr_penalty = max(0, (2.0 - stats['rr_ratio']) * 3) if stats['rr_ratio'] < 2.0 else 0
            
            # Reward for high R:R
            rr_reward = stats['rr_ratio'] * 0.5 if stats['rr_ratio'] >= 2.5 else 0
            
            # Cost (minimize)
            cost = -stats['sharpe'] + trade_penalty + sharpe_penalty + rr_penalty - rr_reward
            
            return cost
        except Exception as e:
            return 1000.0
    
    # Optimize
    print("Running quantum-inspired optimization...")
    optimizer = QuantumInspiredOptimizer(param_ranges)
    result = optimizer.optimize(cost_function, method='simulated_annealing', max_iterations=30)
    
    # Create final params
    opt_params = StrategyParams(
        band_k=result['params']['band_k'],
        band_k_list=(1.0, result['params']['band_k'], 3.0),
        vol_mult=result['params']['vol_mult'],
        atr_cap_mult=result['params']['atr_cap_mult'],
        require_nth_retest=int(result['params']['require_nth_retest']),
        touch_tol_atr_frac=result['params']['touch_tol_atr_frac'],
        trail_pct=result['params']['trail_pct'],
        cross_lookback_bars=int(result['params']['cross_lookback_bars']),
        require_session_filter=False
    )
    
    # Test optimized params
    print("Testing optimized parameters...")
    final_stats = quick_backtest(data, symbol, opt_params)
    
    print(f"\nOptimized Parameters:")
    print(f"  band_k: {opt_params.band_k:.3f}")
    print(f"  vol_mult: {opt_params.vol_mult:.3f}")
    print(f"  atr_cap_mult: {opt_params.atr_cap_mult:.3f}")
    print(f"  require_nth_retest: {opt_params.require_nth_retest}")
    print(f"  touch_tol_atr_frac: {opt_params.touch_tol_atr_frac:.3f}")
    print(f"  trail_pct: {opt_params.trail_pct:.3f}")
    print(f"  cross_lookback_bars: {opt_params.cross_lookback_bars}")
    
    print(f"\nOptimized Results:")
    print(f"  Sharpe Ratio: {final_stats['sharpe']:.2f}")
    print(f"  Trades/Day: {final_stats['trades_per_day']:.2f}")
    print(f"  Win Rate: {final_stats['win_rate']*100:.1f}%")
    print(f"  R:R Ratio: {final_stats['rr_ratio']:.2f}")
    
    return opt_params, final_stats


def main():
    """Run optimization for all symbols."""
    print("="*80)
    print("QUANTUM OPTIMIZATION - VWAP CONTROL FLIP STRATEGY")
    print("Optimizing for Higher R:R and Better Win Rate")
    print("="*80)
    
    symbols = ['GBPJPY', 'BTCUSD', 'XAUUSD']
    all_results = []
    
    start_time = time.time()
    
    for symbol in symbols:
        opt_params, stats = optimize_symbol(symbol)
        all_results.append({
            'symbol': symbol,
            'params': opt_params,
            'stats': stats
        })
    
    total_time = time.time() - start_time
    
    # Summary
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    print(f"{'Symbol':<10} {'Sharpe':<8} {'Trades/Day':<12} {'Win Rate':<10} {'R:R':<8}")
    print("-"*80)
    
    for r in all_results:
        s = r['stats']
        print(f"{r['symbol']:<10} {s['sharpe']:<8.2f} {s['trades_per_day']:<12.2f} "
              f"{s['win_rate']*100:<9.1f}% {s['rr_ratio']:<8.2f}")
    
    print(f"\nTotal Optimization Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)
    print("\nNext: Implement enhanced strategy with:")
    print("  - Stretched targets (opposite band exits)")
    print("  - Stronger filters (trend alignment, RSI divergence)")
    print("  - Better trailing stops")
    print("  - Symbol-specific tweaks")


if __name__ == "__main__":
    main()
