"""
Quantum-Inspired Optimization for VWAP Control Flip Strategy
Optimizes all key parameters using simulated annealing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capital_allocation_ai.vwap_control_flip_strategy import VWAPControlFlipStrategy, StrategyParams
from capital_allocation_ai.quantum_optimizer import QuantumInspiredOptimizer


def generate_market_data(symbol: str, days: int = 60) -> pd.DataFrame:
    """Generate realistic market data."""
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
        volatility[i] = min(volatility[i], vol_base * 4)
    
    returns = np.random.randn(total_bars) * volatility
    for i in range(1, total_bars):
        if abs(returns[i-1]) > vol_base * 2:
            returns[i] -= 0.3 * returns[i-1]
    
    prices = base_price + np.cumsum(returns)
    
    if symbol == 'GBPJPY':
        prices = np.clip(prices, 175, 200)
    elif symbol == 'BTCUSD':
        prices = np.clip(prices, 50000, 70000)
    else:
        prices = np.clip(prices, 2550, 2750)
    
    opens = prices.copy()
    highs = opens + abs(np.random.randn(total_bars) * volatility * opens * 2)
    lows = opens - abs(np.random.randn(total_bars) * volatility * opens * 2)
    closes = opens + np.random.randn(total_bars) * volatility * opens
    
    for i in range(total_bars):
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
    
    base_vol = 5000 if symbol != 'BTCUSD' else 100
    volumes = (base_vol * (1 + abs(returns) * 10)).astype(int)
    volumes = np.clip(volumes, base_vol * 0.2, base_vol * 10)
    
    spike_indices = np.random.choice(total_bars, size=int(total_bars * 0.1), replace=False)
    volumes[spike_indices] = volumes[spike_indices] * np.random.uniform(1.5, 3.0, len(spike_indices))
    
    start_date = datetime.now() - timedelta(days=days)
    timestamps = []
    current_date = start_date
    bar_count = 0
    
    while bar_count < total_bars:
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        timestamps.append(current_date)
        bar_count += 1
        if bar_count % bars_per_day == 0:
            current_date += timedelta(days=1)
        else:
            current_date += timedelta(minutes=5)
    
    return pd.DataFrame({
        'timestamp': timestamps[:total_bars],
        'open': opens[:total_bars],
        'high': highs[:total_bars],
        'low': lows[:total_bars],
        'close': closes[:total_bars],
        'volume': volumes[:total_bars]
    })


def backtest_control_flip(data: pd.DataFrame, symbol: str, params: StrategyParams) -> Dict:
    """Backtest control flip strategy."""
    strategy = VWAPControlFlipStrategy(params)
    
    initial_equity = 10000.0
    equity = initial_equity
    equity_history = [equity]
    trades = []
    max_drawdown = 0.0
    peak_equity = initial_equity
    
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
        if signals['enter_long'] and strategy.position is None:
            strategy.enter_long(row['close'], qty=1.0)
        elif signals['enter_short'] and strategy.position is None:
            strategy.enter_short(row['close'], qty=1.0)
        
        # Exit
        elif signals['exit'] and strategy.position is not None:
            exit_price = row['close']
            pos = strategy.position
            position_size = equity / pos.entry_price * 0.1
            
            if pos.side == "long":
                pnl = (exit_price - pos.entry_price) * position_size
            else:
                pnl = (pos.entry_price - exit_price) * position_size
            
            commission = abs(position_size * exit_price) * 0.0001
            pnl -= commission
            equity += pnl
            
            trades.append({
                'entry': pos.entry_price,
                'exit': exit_price,
                'pnl': pnl,
                'side': pos.side
            })
            
            strategy.exit_position()
        
        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        max_drawdown = max(max_drawdown, drawdown)
        equity_history.append(equity)
    
    # Close open position
    if strategy.position is not None:
        final_price = data.iloc[-1]['close']
        pos = strategy.position
        position_size = equity / pos.entry_price * 0.1
        if pos.side == "long":
            pnl = (final_price - pos.entry_price) * position_size
        else:
            pnl = (pos.entry_price - final_price) * position_size
        equity += pnl
        trades.append({'pnl': pnl})
        strategy.exit_position()
    
    # Calculate stats
    if len(trades) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'sharpe': 0.0,
            'trades_per_day': 0.0,
            'total_return': 0.0,
            'max_drawdown': 0.0
        }
    
    trades_df = pd.DataFrame(trades)
    winning = (trades_df['pnl'] > 0).sum()
    win_rate = winning / len(trades)
    days = len(data) / 288
    trades_per_day = len(trades) / days if days > 0 else 0
    
    returns = trades_df['pnl'] / initial_equity
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0.0
    
    total_return = (equity - initial_equity) / initial_equity
    
    return {
        'total_trades': len(trades),
        'win_rate': win_rate,
        'sharpe': sharpe,
        'trades_per_day': trades_per_day,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'final_equity': equity
    }


def optimize_parameters_quantum(symbol: str, data: pd.DataFrame) -> Tuple[StrategyParams, Dict]:
    """Optimize parameters using quantum-inspired optimizer."""
    print(f"\nOptimizing {symbol} parameters...")
    
    # Parameter ranges for optimization
    param_ranges = {
        'band_k': (1.5, 2.5),
        'vol_mult': (1.0, 2.0),
        'atr_cap_mult': (2.0, 3.5),
        'require_nth_retest': (2, 5),  # Will be converted to int
        'touch_tol_atr_frac': (0.03, 0.10),
        'trail_pct': (0.005, 0.012),
        'cross_lookback_bars': (8, 20)
    }
    
    def cost_function(params_dict: Dict) -> float:
        """Cost function: negative Sharpe + penalty for too many trades."""
        try:
            # Convert params to StrategyParams
            params = StrategyParams(
                band_k=params_dict['band_k'],
                band_k_list=(1.0, params_dict['band_k'], 3.0),
                vol_mult=params_dict['vol_mult'],
                atr_cap_mult=params_dict['atr_cap_mult'],
                require_nth_retest=int(params_dict['require_nth_retest']),
                touch_tol_atr_frac=params_dict['touch_tol_atr_frac'],
                trail_pct=params_dict['trail_pct'],
                cross_lookback_bars=int(params_dict['cross_lookback_bars'])
            )
            
            stats = backtest_control_flip(data, symbol, params)
            
            # Penalty for too many trades
            trades_per_day = stats.get('trades_per_day', 100)
            trade_penalty = 0.0
            if trades_per_day > 5.0:
                trade_penalty = (trades_per_day - 5.0) * 5
            
            # Penalty for negative Sharpe
            sharpe_penalty = 0.0
            if stats['sharpe'] < 0:
                sharpe_penalty = abs(stats['sharpe']) * 10
            
            # Penalty for high drawdown
            dd_penalty = 0.0
            if stats['max_drawdown'] > 0.20:
                dd_penalty = (stats['max_drawdown'] - 0.20) * 30
            
            # Penalty for low win rate
            wr_penalty = 0.0
            if stats['win_rate'] < 0.50:
                wr_penalty = (0.50 - stats['win_rate']) * 20
            
            # Total cost (minimize)
            cost = -stats['sharpe'] + trade_penalty + sharpe_penalty + dd_penalty + wr_penalty
            
            return cost
        except Exception as e:
            print(f"Error in cost function: {e}")
            return 1000.0
    
    # Use quantum-inspired optimizer
    optimizer = QuantumInspiredOptimizer(param_ranges)
    result = optimizer.optimize(cost_function, method='simulated_annealing', max_iterations=50)
    
    # Create StrategyParams from optimized values
    opt_params = StrategyParams(
        band_k=result['params']['band_k'],
        band_k_list=(1.0, result['params']['band_k'], 3.0),
        vol_mult=result['params']['vol_mult'],
        atr_cap_mult=result['params']['atr_cap_mult'],
        require_nth_retest=int(result['params']['require_nth_retest']),
        touch_tol_atr_frac=result['params']['touch_tol_atr_frac'],
        trail_pct=result['params']['trail_pct'],
        cross_lookback_bars=int(result['params']['cross_lookback_bars'])
    )
    
    # Get final stats
    final_stats = backtest_control_flip(data, symbol, opt_params)
    
    return opt_params, final_stats


def main():
    """Main optimization."""
    print("="*80)
    print("VWAP CONTROL FLIP STRATEGY - QUANTUM OPTIMIZATION")
    print("60 Days: GBP/JPY, BTC/USD, Gold/USD")
    print("="*80)
    
    symbols = ['GBPJPY', 'BTCUSD', 'XAUUSD']
    all_results = []
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"OPTIMIZING {symbol}")
        print(f"{'='*80}")
        
        # Generate data
        print("Generating 60 days of data...")
        data = generate_market_data(symbol, days=60)
        print(f"Generated {len(data)} bars")
        
        # Optimize
        opt_params, stats = optimize_parameters_quantum(symbol, data)
        
        print(f"\nOptimized Parameters:")
        print(f"  band_k: {opt_params.band_k:.3f}")
        print(f"  vol_mult: {opt_params.vol_mult:.3f}")
        print(f"  atr_cap_mult: {opt_params.atr_cap_mult:.3f}")
        print(f"  require_nth_retest: {opt_params.require_nth_retest}")
        print(f"  touch_tol_atr_frac: {opt_params.touch_tol_atr_frac:.3f}")
        print(f"  trail_pct: {opt_params.trail_pct:.3f}")
        print(f"  cross_lookback_bars: {opt_params.cross_lookback_bars}")
        
        print(f"\nOptimized Results:")
        print(f"  Trades: {stats['total_trades']} ({stats['trades_per_day']:.2f}/day)")
        print(f"  Win Rate: {stats['win_rate']*100:.1f}%")
        print(f"  Return: {stats['total_return']*100:+.2f}%")
        print(f"  Sharpe: {stats['sharpe']:.2f}")
        print(f"  Max DD: {stats['max_drawdown']*100:.2f}%")
        
        all_results.append({
            'symbol': symbol,
            'params': opt_params,
            'stats': stats
        })
    
    # Summary
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    print(f"{'Symbol':<10} {'Trades':<8} {'Trades/Day':<12} {'Win Rate':<10} {'Return':<10} {'Sharpe':<8}")
    print("-"*80)
    
    for r in all_results:
        s = r['stats']
        print(f"{r['symbol']:<10} {s['total_trades']:<8} {s['trades_per_day']:<12.2f} "
              f"{s['win_rate']*100:<9.1f}% {s['total_return']*100:<9.2f}% {s['sharpe']:<8.2f}")
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
