"""
Quantum-Inspired Parameter Optimization for VWAP Pro Strategy
Optimizes parameters to maximize Sharpe ratio and reduce trade frequency
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capital_allocation_ai.vwap_pro_strategy import VWAPProStrategy
from capital_allocation_ai.quantum_optimizer import optimize_vwap_params, QuantumInspiredOptimizer


def generate_market_data(symbol: str, days: int = 60) -> pd.DataFrame:
    """Generate market data (same as backtest_multi_symbol.py)."""
    trading_days = days
    bars_per_day = 288
    total_bars = trading_days * bars_per_day
    
    np.random.seed(42 if symbol == 'GBPJPY' else (43 if symbol == 'BTCUSD' else 44))
    
    if symbol == 'GBPJPY':
        base_price = 185.0
        volatility_base = 0.0005
        trend_range = (-2, 3)
    elif symbol == 'BTCUSD':
        base_price = 60000.0
        volatility_base = 0.002
        trend_range = (-5000, 8000)
    else:  # XAUUSD
        base_price = 2650.0
        volatility_base = 0.0008
        trend_range = (-50, 80)
    
    trend = np.linspace(trend_range[0], trend_range[1], total_bars)
    volatility = np.ones(total_bars) * volatility_base
    
    for i in range(1, total_bars):
        volatility[i] = volatility_base * 0.6 + 0.7 * volatility[i-1] + 0.2 * abs(np.random.randn() * volatility_base)
        volatility[i] = min(volatility[i], volatility_base * 4)
    
    returns = np.random.randn(total_bars) * volatility
    for i in range(1, total_bars):
        if abs(returns[i-1]) > volatility_base * 2:
            returns[i] -= 0.3 * returns[i-1]
    
    returns += trend / total_bars
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
    
    if symbol == 'BTCUSD':
        base_volume = 100
        volume_multiplier = 1 + volatility / volatility.mean() * 3
    elif symbol == 'XAUUSD':
        base_volume = 5000
        volume_multiplier = 1 + volatility / volatility.mean() * 2
    else:
        base_volume = 5000
        volume_multiplier = 1 + volatility / volatility.mean() * 2
    
    volumes = (base_volume * volume_multiplier * (1 + abs(returns) * 10)).astype(int)
    volumes = np.clip(volumes, base_volume * 0.2, base_volume * 10)
    
    from datetime import datetime, timedelta
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


def backtest_with_params(data: pd.DataFrame, symbol: str, params: Dict) -> Tuple[float, Dict]:
    """Backtest strategy with given parameters."""
    from backtest_multi_symbol import backtest_strategy
    return backtest_strategy(data, symbol, params)


def optimize_symbol(symbol: str, days: int = 60, target_trades_per_day: float = 5.0):
    """
    Optimize parameters for a specific symbol.
    
    Args:
        symbol: Trading symbol
        days: Days of data to use
        target_trades_per_day: Target number of trades per day (to reduce frequency)
    """
    print(f"\n{'='*80}")
    print(f"OPTIMIZING {symbol}")
    print(f"{'='*80}")
    
    # Generate data
    data = generate_market_data(symbol, days)
    
    # Define parameter ranges
    if symbol == 'BTCUSD':
        param_ranges = {
            'sigma_multiplier': (2.0, 3.5),
            'lookback_periods': (20, 40),
            'vwma_period': (10, 20),
            'volatility_threshold': (2.5, 4.0),
            'trailing_stop_pct': (0.008, 0.015)
        }
    elif symbol == 'XAUUSD':
        param_ranges = {
            'sigma_multiplier': (1.5, 2.5),
            'lookback_periods': (20, 35),
            'vwma_period': (10, 18),
            'volatility_threshold': (2.0, 3.5),
            'trailing_stop_pct': (0.005, 0.010)
        }
    else:  # GBPJPY
        param_ranges = {
            'sigma_multiplier': (1.8, 3.0),
            'lookback_periods': (20, 35),
            'vwma_period': (10, 20),
            'volatility_threshold': (2.5, 4.0),
            'trailing_stop_pct': (0.006, 0.012)
        }
    
    def cost_function(params: Dict) -> float:
        """Cost function: negative Sharpe + penalty for too many trades."""
        try:
            sharpe, stats = backtest_with_params(data, symbol, params)
            
            # Calculate trades per day
            trades_per_day = stats['total_trades'] / days
            
            # Cost = negative Sharpe + penalty for too many trades
            cost = -sharpe
            
            # Penalty for too many trades
            if trades_per_day > target_trades_per_day:
                excess_trades = trades_per_day - target_trades_per_day
                cost += excess_trades * 10  # Heavy penalty
            
            # Penalty for negative returns
            if stats['total_return'] < 0:
                cost += abs(stats['total_return']) * 5
            
            # Penalty for high drawdown
            if stats['max_drawdown'] > 0.20:
                cost += (stats['max_drawdown'] - 0.20) * 20
            
            # Bonus for good win rate
            if stats['win_rate'] < 0.50:
                cost += (0.50 - stats['win_rate']) * 5
            
            return cost
        except Exception as e:
            print(f"Error in cost function: {e}")
            return 1000.0
    
    # Run optimization
    optimizer = QuantumInspiredOptimizer(param_ranges)
    
    print(f"Running quantum-inspired optimization...")
    print(f"Target: <{target_trades_per_day} trades/day, maximize Sharpe")
    print(f"Parameter ranges: {param_ranges}")
    
    result = optimizer.optimize(cost_function, method='simulated_annealing', max_iterations=150)
    
    # Get final stats with optimized params
    sharpe, stats = backtest_with_params(data, symbol, result['params'])
    
    trades_per_day = stats['total_trades'] / days
    
    print(f"\nOPTIMIZATION RESULTS:")
    print(f"{'='*80}")
    print(f"Optimal Parameters:")
    for param, value in result['params'].items():
        print(f"  {param}: {value:.4f}")
    print(f"\nPerformance:")
    print(f"  Total Trades: {stats['total_trades']} ({trades_per_day:.1f} per day)")
    print(f"  Win Rate: {stats['win_rate']*100:.2f}%")
    print(f"  Total Return: {stats['total_return']*100:.2f}%")
    print(f"  Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {stats['max_drawdown']*100:.2f}%")
    print(f"  Profit Factor: {stats.get('profit_factor', 0):.2f}")
    
    return result['params'], stats


def main():
    """Main optimization function."""
    print("="*80)
    print("QUANTUM-INSPIRED PARAMETER OPTIMIZATION")
    print("VWAP Pro Strategy - Multi-Symbol")
    print("="*80)
    
    symbols = ['GBPJPY', 'BTCUSD', 'XAUUSD']
    optimized_params = {}
    optimized_stats = {}
    
    # Optimize each symbol
    for symbol in symbols:
        params, stats = optimize_symbol(symbol, days=60, target_trades_per_day=5.0)
        optimized_params[symbol] = params
        optimized_stats[symbol] = stats
    
    # Summary
    print("\n" + "="*80)
    print("OPTIMIZATION SUMMARY")
    print("="*80)
    
    summary_data = []
    for symbol in symbols:
        stats = optimized_stats[symbol]
        params = optimized_params[symbol]
        trades_per_day = stats['total_trades'] / 60
        
        summary_data.append({
            'Symbol': symbol,
            'Trades/Day': f"{trades_per_day:.1f}",
            'Total Trades': stats['total_trades'],
            'Win Rate': f"{stats['win_rate']*100:.1f}%",
            'Return': f"{stats['total_return']*100:+.1f}%",
            'Sharpe': f"{stats['sharpe_ratio']:.2f}",
            'Max DD': f"{stats['max_drawdown']*100:.1f}%"
        })
        
        print(f"\n{symbol} Optimal Parameters:")
        for param, value in params.items():
            print(f"  {param}: {value:.4f}")
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Save optimized parameters
    import json
    with open('optimized_params.json', 'w') as f:
        json.dump(optimized_params, f, indent=2)
    print(f"\nOptimized parameters saved to: optimized_params.json")
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
