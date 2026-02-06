"""
Backtest VWAP Pro Strategy on GBP/JPY
Includes all filters, indicators, and quantum-inspired optimization.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capital_allocation_ai.vwap_pro_strategy import VWAPProStrategy
from capital_allocation_ai.quantum_optimizer import optimize_vwap_params


def generate_gbpjpy_data_3weeks() -> pd.DataFrame:
    """Generate 3 weeks of GBP/JPY data."""
    print("Generating GBP/JPY data for last 3 weeks...")
    
    trading_days = 21
    bars_per_day = 288  # 5-minute bars
    total_bars = trading_days * bars_per_day
    
    np.random.seed(42)
    base_price = 185.0
    
    trend = np.linspace(0, 2, total_bars)
    volatility = np.ones(total_bars) * 0.0005
    
    for i in range(1, total_bars):
        volatility[i] = 0.0003 + 0.7 * volatility[i-1] + 0.2 * abs(np.random.randn() * 0.0003)
        volatility[i] = min(volatility[i], 0.002)
    
    returns = np.random.randn(total_bars) * volatility
    for i in range(1, total_bars):
        if abs(returns[i-1]) > 0.001:
            returns[i] -= 0.3 * returns[i-1]
    
    returns += trend / total_bars
    prices = base_price + np.cumsum(returns)
    prices = np.clip(prices, 180, 195)
    
    opens = prices.copy()
    highs = opens + abs(np.random.randn(total_bars) * volatility * opens * 2)
    lows = opens - abs(np.random.randn(total_bars) * volatility * opens * 2)
    closes = opens + np.random.randn(total_bars) * volatility * opens
    
    for i in range(total_bars):
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
    
    base_volume = 5000
    volume_multiplier = 1 + volatility / volatility.mean() * 2
    volumes = (base_volume * volume_multiplier * (1 + abs(returns) * 10)).astype(int)
    volumes = np.clip(volumes, 1000, 50000)
    
    start_date = datetime.now() - timedelta(days=21)
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
    
    data = pd.DataFrame({
        'timestamp': timestamps[:total_bars],
        'open': opens[:total_bars],
        'high': highs[:total_bars],
        'low': lows[:total_bars],
        'close': closes[:total_bars],
        'volume': volumes[:total_bars]
    })
    
    print(f"Generated {len(data)} bars")
    return data


def backtest_vwap_pro_strategy(data: pd.DataFrame, params: Dict = None) -> Tuple[float, Dict]:
    """
    Backtest VWAP Pro Strategy.
    
    Returns:
        (sharpe_ratio, statistics_dict)
    """
    if params is None:
        params = {}
    
    strategy = VWAPProStrategy(
        sigma_multiplier=params.get('sigma_multiplier', 2.0),
        lookback_periods=int(params.get('lookback_periods', 20)),
        vwma_period=int(params.get('vwma_period', 10)),
        ema_period=int(params.get('ema_period', 90)),
        volatility_threshold=params.get('volatility_threshold', 2.5)
    )
    
    initial_equity = 10000.0
    equity = initial_equity
    equity_history = [equity]
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
        
        # Execute signals
        if signals['entry_long'] and strategy.position == 0:
            strategy.enter_long(row['close'])
            entry_price = row['close']
        
        elif signals['reposition_long'] and strategy.position == 0:
            strategy.enter_long(row['close'])
            entry_price = row['close']
        
        elif signals['exit'] or signals['stop_hit']:
            if strategy.position != 0:
                exit_price = row['close']
                pnl = (exit_price - strategy.entry_price) * (equity / strategy.entry_price * 0.1) if strategy.position == 1 else (strategy.entry_price - exit_price) * (equity / strategy.entry_price * 0.1)
                equity += pnl
                trades.append({
                    'entry': strategy.entry_price,
                    'exit': exit_price,
                    'pnl': pnl,
                    'type': 'long' if strategy.position == 1 else 'short'
                })
                strategy.exit_position()
        
        equity_history.append(equity)
    
    # Calculate statistics
    if len(trades) == 0:
        return 0.0, {'total_trades': 0, 'win_rate': 0.0, 'total_return': 0.0}
    
    trades_df = pd.DataFrame(trades)
    winning_trades = (trades_df['pnl'] > 0).sum()
    win_rate = winning_trades / len(trades) if len(trades) > 0 else 0.0
    total_return = (equity - initial_equity) / initial_equity
    
    # Calculate Sharpe ratio
    equity_array = np.array(equity_history)
    returns = np.diff(equity_array) / equity_array[:-1]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 288) if len(returns) > 1 else 0.0
    
    stats = {
        'total_trades': len(trades),
        'winning_trades': winning_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'final_equity': equity,
        'sharpe_ratio': sharpe,
        'avg_win': trades_df[trades_df['pnl'] > 0]['pnl'].mean() if (trades_df['pnl'] > 0).any() else 0.0,
        'avg_loss': trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (trades_df['pnl'] < 0).any() else 0.0
    }
    
    return sharpe, stats


def main():
    """Main backtest function."""
    print("="*60)
    print("VWAP PRO STRATEGY BACKTEST - GBP/JPY (3 Weeks)")
    print("="*60)
    
    # Generate data
    data = generate_gbpjpy_data_3weeks()
    
    # Run backtest with default parameters
    print("\nRunning backtest with default parameters...")
    sharpe, stats = backtest_vwap_pro_strategy(data)
    
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Win Rate: {stats['win_rate']*100:.2f}%")
    print(f"Total Return: {stats['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f}")
    print(f"Final Equity: ${stats['final_equity']:,.2f}")
    
    # Optional: Optimize parameters
    print("\n" + "="*60)
    print("OPTIMIZING PARAMETERS (Quantum-Inspired)")
    print("="*60)
    
    param_ranges = {
        'sigma_multiplier': (1.5, 3.0),
        'lookback_periods': (15, 30),
        'vwma_period': (8, 15),
        'volatility_threshold': (2.0, 3.0)
    }
    
    def backtest_wrapper(params):
        return backtest_vwap_pro_strategy(data, params)
    
    try:
        opt_result = optimize_vwap_params(
            backtest_wrapper,
            param_ranges,
            target_sharpe=1.5,
            method='simulated_annealing'
        )
        
        print("\nOPTIMIZATION RESULTS:")
        print(f"Optimal Parameters:")
        for param, value in opt_result['optimal_params'].items():
            print(f"  {param}: {value:.3f}")
        print(f"Sharpe Ratio: {opt_result['sharpe_ratio']:.2f}")
        print(f"Method: {opt_result['method']}")
    except Exception as e:
        print(f"Optimization failed: {e}")
    
    print("\n" + "="*60)
    print("BACKTEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
