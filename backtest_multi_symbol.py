"""
Multi-Symbol Backtest - VWAP Pro Strategy
Tests on GBP/JPY, BTC/USD, and Gold/USD for last 60 days
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capital_allocation_ai.vwap_pro_strategy import VWAPProStrategy
from capital_allocation_ai.quantum_optimizer import optimize_vwap_params


def generate_market_data(symbol: str, days: int = 60) -> pd.DataFrame:
    """
    Generate realistic market data for different symbols.
    
    Args:
        symbol: 'GBPJPY', 'BTCUSD', or 'XAUUSD'
        days: Number of days
    
    Returns:
        DataFrame with OHLCV data
    """
    print(f"Generating {days} days of {symbol} data...")
    
    trading_days = days
    bars_per_day = 288  # 5-minute bars
    total_bars = trading_days * bars_per_day
    
    np.random.seed(42 if symbol == 'GBPJPY' else (43 if symbol == 'BTCUSD' else 44))
    
    # Symbol-specific parameters
    if symbol == 'GBPJPY':
        base_price = 185.0
        volatility_base = 0.0005
        trend_range = (-2, 3)
    elif symbol == 'BTCUSD':
        base_price = 60000.0
        volatility_base = 0.002  # Higher volatility for crypto
        trend_range = (-5000, 8000)
    else:  # XAUUSD (Gold)
        base_price = 2650.0
        volatility_base = 0.0008
        trend_range = (-50, 80)
    
    # Generate trend
    trend = np.linspace(trend_range[0], trend_range[1], total_bars)
    
    # Volatility clustering (GARCH-like)
    volatility = np.ones(total_bars) * volatility_base
    for i in range(1, total_bars):
        volatility[i] = volatility_base * 0.6 + 0.7 * volatility[i-1] + 0.2 * abs(np.random.randn() * volatility_base)
        volatility[i] = min(volatility[i], volatility_base * 4)
    
    # Generate returns
    returns = np.random.randn(total_bars) * volatility
    
    # Add mean reversion
    for i in range(1, total_bars):
        if abs(returns[i-1]) > volatility_base * 2:
            returns[i] -= 0.3 * returns[i-1]
    
    # Add trend
    returns += trend / total_bars
    
    # Generate prices
    prices = base_price + np.cumsum(returns)
    
    # Clip to reasonable ranges
    if symbol == 'GBPJPY':
        prices = np.clip(prices, 175, 200)
    elif symbol == 'BTCUSD':
        prices = np.clip(prices, 50000, 70000)
    else:  # XAUUSD
        prices = np.clip(prices, 2550, 2750)
    
    # Generate OHLC
    opens = prices.copy()
    highs = opens + abs(np.random.randn(total_bars) * volatility * opens * 2)
    lows = opens - abs(np.random.randn(total_bars) * volatility * opens * 2)
    closes = opens + np.random.randn(total_bars) * volatility * opens
    
    # Ensure OHLC consistency
    for i in range(total_bars):
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
    
    # Generate volume (symbol-specific)
    if symbol == 'BTCUSD':
        base_volume = 100  # BTC volume
        volume_multiplier = 1 + volatility / volatility.mean() * 3
    elif symbol == 'XAUUSD':
        base_volume = 5000  # Gold contracts
        volume_multiplier = 1 + volatility / volatility.mean() * 2
    else:  # GBPJPY
        base_volume = 5000
        volume_multiplier = 1 + volatility / volatility.mean() * 2
    
    volumes = (base_volume * volume_multiplier * (1 + abs(returns) * 10)).astype(int)
    volumes = np.clip(volumes, base_volume * 0.2, base_volume * 10)
    
    # Generate timestamps
    start_date = datetime.now() - timedelta(days=days)
    timestamps = []
    current_date = start_date
    bar_count = 0
    
    while bar_count < total_bars:
        # Skip weekends
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        
        timestamps.append(current_date)
        bar_count += 1
        
        # Move to next 5-minute bar
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
    
    print(f"Generated {len(data)} bars for {symbol}")
    print(f"Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    return data


def backtest_strategy(data: pd.DataFrame, symbol: str, params: Dict = None) -> Tuple[float, Dict]:
    """
    Backtest VWAP Pro Strategy on given data.
    
    Returns:
        (sharpe_ratio, statistics_dict)
    """
    if params is None:
        params = {}
    
    # Symbol-specific parameter adjustments
    if symbol == 'BTCUSD':
        # Crypto needs tighter stops and filters
        default_params = {
            'sigma_multiplier': 2.5,  # Wider bands for crypto volatility
            'lookback_periods': 20,
            'volatility_threshold': 3.0,  # Higher threshold for crypto
            'trailing_stop_pct': 0.01  # 1% trailing stop for crypto
        }
    elif symbol == 'XAUUSD':
        # Gold is less volatile
        default_params = {
            'sigma_multiplier': 1.8,
            'lookback_periods': 20,
            'volatility_threshold': 2.0,
            'trailing_stop_pct': 0.005  # 0.5% trailing stop
        }
    else:  # GBPJPY
        default_params = {
            'sigma_multiplier': 2.0,
            'lookback_periods': 20,
            'volatility_threshold': 2.5,
            'trailing_stop_pct': 0.007  # 0.7% trailing stop
        }
    
    # Merge with provided params
    final_params = {**default_params, **params}
    
    strategy = VWAPProStrategy(
        sigma_multiplier=final_params.get('sigma_multiplier', 2.0),
        lookback_periods=int(final_params.get('lookback_periods', 20)),
        vwma_period=int(final_params.get('vwma_period', 10)),
        ema_period=int(final_params.get('ema_period', 90)),
        volatility_threshold=final_params.get('volatility_threshold', 2.5),
        trailing_stop_pct=final_params.get('trailing_stop_pct', 0.007)
    )
    
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
                # Calculate position size (10% of equity)
                position_size = equity / entry_price * 0.1
                
                if strategy.position == 1:  # Long
                    pnl = (exit_price - strategy.entry_price) * position_size
                else:  # Short
                    pnl = (strategy.entry_price - exit_price) * position_size
                
                # Apply commission (0.01%)
                commission = abs(position_size * exit_price) * 0.0001
                pnl -= commission
                
                equity += pnl
                
                trades.append({
                    'entry': strategy.entry_price,
                    'exit': exit_price,
                    'pnl': pnl,
                    'type': 'long' if strategy.position == 1 else 'short',
                    'entry_time': row.get('timestamp'),
                    'exit_time': row.get('timestamp')
                })
                
                strategy.exit_position()
        
        # Track drawdown
        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        max_drawdown = max(max_drawdown, drawdown)
        
        equity_history.append(equity)
    
    # Close any open positions
    if strategy.position != 0:
        final_price = data.iloc[-1]['close']
        position_size = equity / strategy.entry_price * 0.1
        if strategy.position == 1:
            pnl = (final_price - strategy.entry_price) * position_size
        else:
            pnl = (strategy.entry_price - final_price) * position_size
        commission = abs(position_size * final_price) * 0.0001
        pnl -= commission
        equity += pnl
        equity_history.append(equity)
        strategy.exit_position()
    
    # Calculate statistics
    if len(trades) == 0:
        return 0.0, {
            'symbol': symbol,
            'total_trades': 0,
            'win_rate': 0.0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0
        }
    
    trades_df = pd.DataFrame(trades)
    winning_trades = (trades_df['pnl'] > 0).sum()
    win_rate = winning_trades / len(trades) if len(trades) > 0 else 0.0
    total_return = (equity - initial_equity) / initial_equity
    
    # Calculate Sharpe ratio
    equity_array = np.array(equity_history)
    if len(equity_array) > 1:
        returns = np.diff(equity_array) / equity_array[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 288) if len(returns) > 1 and np.std(returns) > 0 else 0.0
    else:
        sharpe = 0.0
    
    stats = {
        'symbol': symbol,
        'total_trades': len(trades),
        'winning_trades': winning_trades,
        'losing_trades': len(trades) - winning_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'final_equity': equity,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'avg_win': trades_df[trades_df['pnl'] > 0]['pnl'].mean() if (trades_df['pnl'] > 0).any() else 0.0,
        'avg_loss': trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (trades_df['pnl'] < 0).any() else 0.0,
        'profit_factor': abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if (trades_df['pnl'] < 0).any() and trades_df[trades_df['pnl'] < 0]['pnl'].sum() != 0 else 0.0,
        'equity_history': equity_history
    }
    
    return sharpe, stats


def print_results(all_results: List[Dict]):
    """Print comprehensive results for all symbols."""
    print("\n" + "="*80)
    print("MULTI-SYMBOL BACKTEST RESULTS - LAST 60 DAYS")
    print("="*80)
    
    # Individual results
    for stats in all_results:
        print(f"\n{stats['symbol']} RESULTS")
        print("-"*80)
        print(f"Total Trades:           {stats['total_trades']}")
        print(f"Winning Trades:         {stats['winning_trades']}")
        print(f"Losing Trades:          {stats['losing_trades']}")
        print(f"Win Rate:               {stats['win_rate']*100:.2f}%")
        print(f"Total Return:           {stats['total_return']*100:+.2f}%")
        print(f"Final Equity:           ${stats['final_equity']:,.2f}")
        print(f"Sharpe Ratio:           {stats['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:           {stats['max_drawdown']*100:.2f}%")
        if stats['total_trades'] > 0:
            print(f"Average Win:            ${stats['avg_win']:.2f}")
            print(f"Average Loss:           ${stats['avg_loss']:.2f}")
            print(f"Profit Factor:          {stats['profit_factor']:.2f}")
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON")
    print("="*80)
    
    summary_data = []
    for stats in all_results:
        summary_data.append({
            'Symbol': stats['symbol'],
            'Trades': stats['total_trades'],
            'Win Rate': f"{stats['win_rate']*100:.1f}%",
            'Return': f"{stats['total_return']*100:+.1f}%",
            'Sharpe': f"{stats['sharpe_ratio']:.2f}",
            'Max DD': f"{stats['max_drawdown']*100:.1f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    # Best performer
    best_sharpe = max(all_results, key=lambda x: x['sharpe_ratio'])
    best_return = max(all_results, key=lambda x: x['total_return'])
    
    print(f"\nBest Sharpe Ratio: {best_sharpe['symbol']} ({best_sharpe['sharpe_ratio']:.2f})")
    print(f"Best Return: {best_return['symbol']} ({best_return['total_return']*100:.2f}%)")
    
    # Combined portfolio (equal weight)
    print("\n" + "="*80)
    print("COMBINED PORTFOLIO (Equal Weight)")
    print("="*80)
    
    combined_return = np.mean([s['total_return'] for s in all_results])
    combined_sharpe = np.mean([s['sharpe_ratio'] for s in all_results])
    combined_dd = np.mean([s['max_drawdown'] for s in all_results])
    total_trades = sum([s['total_trades'] for s in all_results])
    
    print(f"Combined Return:       {combined_return*100:+.2f}%")
    print(f"Combined Sharpe:        {combined_sharpe:.2f}")
    print(f"Combined Max DD:        {combined_dd*100:.2f}%")
    print(f"Total Trades:           {total_trades}")


def main():
    """Main backtest function."""
    print("="*80)
    print("VWAP PRO STRATEGY - MULTI-SYMBOL BACKTEST")
    print("Last 60 Days: GBP/JPY, BTC/USD, Gold/USD")
    print("="*80)
    
    symbols = ['GBPJPY', 'BTCUSD', 'XAUUSD']
    all_results = []
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"BACKTESTING {symbol}")
        print(f"{'='*80}")
        
        # Generate data
        data = generate_market_data(symbol, days=60)
        
        # Run backtest
        sharpe, stats = backtest_strategy(data, symbol)
        all_results.append(stats)
        
        print(f"\n{symbol} Quick Stats:")
        print(f"  Trades: {stats['total_trades']}")
        print(f"  Win Rate: {stats['win_rate']*100:.1f}%")
        print(f"  Return: {stats['total_return']*100:+.2f}%")
        print(f"  Sharpe: {stats['sharpe_ratio']:.2f}")
    
    # Print comprehensive results
    print_results(all_results)
    
    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
