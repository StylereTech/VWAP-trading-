"""
Backtest Optimized VWAP Pro Strategy
60 Days: GBP/JPY, BTC/USD, Gold/USD
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capital_allocation_ai.vwap_pro_strategy import VWAPProStrategy


def generate_market_data(symbol: str, days: int = 60) -> pd.DataFrame:
    """Generate realistic market data for backtesting."""
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
        volatility_base = 0.002
        trend_range = (-5000, 8000)
    else:  # XAUUSD
        base_price = 2650.0
        volatility_base = 0.0008
        trend_range = (-50, 80)
    
    # Generate prices with mean reversion
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
    
    # Clip prices to realistic ranges
    if symbol == 'GBPJPY':
        prices = np.clip(prices, 175, 200)
    elif symbol == 'BTCUSD':
        prices = np.clip(prices, 50000, 70000)
    else:
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
    
    # Generate volumes with occasional spikes
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
    
    # Add occasional volume spikes
    spike_indices = np.random.choice(total_bars, size=int(total_bars * 0.1), replace=False)
    volumes[spike_indices] = volumes[spike_indices] * np.random.uniform(1.5, 3.0, len(spike_indices))
    
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


def backtest_strategy(data: pd.DataFrame, symbol: str, params: Dict) -> Dict:
    """Backtest strategy with given parameters."""
    strategy = VWAPProStrategy(**params)
    
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
        
        # Entry signals
        if (signals['entry_long'] or signals['reposition_long']) and strategy.position == 0:
            strategy.enter_long(row['close'])
        
        # Exit signals
        elif (signals['exit'] or signals['stop_hit']) and strategy.position != 0:
            exit_price = row['close']
            position_size = equity / strategy.entry_price * 0.1
            
            if strategy.position == 1:
                pnl = (exit_price - strategy.entry_price) * position_size
            else:
                pnl = (strategy.entry_price - exit_price) * position_size
            
            commission = abs(position_size * exit_price) * 0.0001
            pnl -= commission
            equity += pnl
            
            trades.append({
                'entry': strategy.entry_price,
                'exit': exit_price,
                'pnl': pnl,
                'entry_time': row.get('timestamp'),
                'exit_time': row.get('timestamp')
            })
            
            strategy.exit_position()
        
        # Update drawdown
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
        trades.append({
            'entry': strategy.entry_price,
            'exit': final_price,
            'pnl': pnl
        })
        strategy.exit_position()
    
    # Calculate statistics
    if len(trades) == 0:
        return {
            'symbol': symbol,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'total_return': 0.0,
            'final_equity': initial_equity,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'trades_per_day': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0
        }
    
    trades_df = pd.DataFrame(trades)
    winning_trades = (trades_df['pnl'] > 0).sum()
    losing_trades = len(trades) - winning_trades
    win_rate = winning_trades / len(trades) if len(trades) > 0 else 0.0
    total_return = (equity - initial_equity) / initial_equity
    
    # Calculate Sharpe ratio
    if len(trades_df) > 1:
        returns = trades_df['pnl'] / initial_equity
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if np.std(returns) > 0 else 0.0
    else:
        sharpe = 0.0
    
    # Calculate trades per day
    days_traded = len(data) / 288  # 288 bars per day
    trades_per_day = len(trades) / days_traded if days_traded > 0 else 0.0
    
    # Average win/loss
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if (trades_df['pnl'] > 0).any() else 0.0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (trades_df['pnl'] < 0).any() else 0.0
    
    # Profit factor
    total_wins = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if (trades_df['pnl'] > 0).any() else 0.0
    total_losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if (trades_df['pnl'] < 0).any() else 1.0
    profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
    
    return {
        'symbol': symbol,
        'total_trades': len(trades),
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'final_equity': equity,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'trades_per_day': trades_per_day,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor
    }


def main():
    """Run 60-day backtest on all symbols."""
    print("="*80)
    print("VWAP PRO STRATEGY - OPTIMIZED BACKTEST")
    print("60 Days: GBP/JPY, BTC/USD, Gold/USD")
    print("="*80)
    
    symbols = ['GBPJPY', 'BTCUSD', 'XAUUSD']
    
    # Optimized parameters for each symbol
    optimized_params = {
        'GBPJPY': {
            'sigma_multiplier': 2.1,
            'lookback_periods': 20,
            'min_volume_multiplier': 1.4,
            'volume_spike_multiplier': 1.6,
            'min_touches_before_entry': 3,
            'volatility_threshold': 2.8,
            'trailing_stop_pct': 0.007
        },
        'BTCUSD': {
            'sigma_multiplier': 2.5,
            'lookback_periods': 20,
            'min_volume_multiplier': 1.5,
            'volume_spike_multiplier': 1.8,
            'min_touches_before_entry': 3,
            'volatility_threshold': 3.0,
            'trailing_stop_pct': 0.01
        },
        'XAUUSD': {
            'sigma_multiplier': 1.8,
            'lookback_periods': 20,
            'min_volume_multiplier': 1.3,
            'volume_spike_multiplier': 1.5,
            'min_touches_before_entry': 3,
            'volatility_threshold': 2.5,
            'trailing_stop_pct': 0.005
        }
    }
    
    all_results = []
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"BACKTESTING {symbol}")
        print(f"{'='*80}")
        
        # Generate data
        print(f"Generating 60 days of {symbol} data...")
        data = generate_market_data(symbol, days=60)
        print(f"Generated {len(data)} bars")
        print(f"Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
        
        # Run backtest with optimized parameters
        print(f"\nRunning backtest with optimized parameters...")
        params = optimized_params[symbol]
        results = backtest_strategy(data, symbol, params)
        all_results.append(results)
        
        # Print results
        print(f"\n{symbol} Results:")
        print("-"*80)
        print(f"Total Trades:           {results['total_trades']}")
        print(f"Winning Trades:         {results['winning_trades']}")
        print(f"Losing Trades:          {results['losing_trades']}")
        print(f"Win Rate:               {results['win_rate']*100:.2f}%")
        print(f"Total Return:           {results['total_return']*100:+.2f}%")
        print(f"Final Equity:           ${results['final_equity']:,.2f}")
        print(f"Sharpe Ratio:           {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:           {results['max_drawdown']*100:.2f}%")
        print(f"Trades Per Day:         {results['trades_per_day']:.2f}")
        print(f"Average Win:            ${results['avg_win']:.2f}")
        print(f"Average Loss:           ${results['avg_loss']:.2f}")
        print(f"Profit Factor:          {results['profit_factor']:.2f}")
    
    # Summary comparison
    print("\n" + "="*80)
    print("SUMMARY COMPARISON - ALL SYMBOLS")
    print("="*80)
    print(f"{'Symbol':<10} {'Trades':<8} {'Trades/Day':<12} {'Win Rate':<10} {'Return':<10} {'Sharpe':<8} {'Max DD':<8}")
    print("-"*80)
    
    for r in all_results:
        print(f"{r['symbol']:<10} {r['total_trades']:<8} {r['trades_per_day']:<12.2f} "
              f"{r['win_rate']*100:<9.1f}% {r['total_return']*100:<9.2f}% "
              f"{r['sharpe_ratio']:<8.2f} {r['max_drawdown']*100:<7.2f}%")
    
    # Combined portfolio
    print("\n" + "="*80)
    print("COMBINED PORTFOLIO (Equal Weight)")
    print("="*80)
    
    total_trades = sum(r['total_trades'] for r in all_results)
    avg_win_rate = np.mean([r['win_rate'] for r in all_results])
    avg_return = np.mean([r['total_return'] for r in all_results])
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_results])
    max_dd = max([r['max_drawdown'] for r in all_results])
    
    print(f"Total Trades:            {total_trades}")
    print(f"Average Win Rate:        {avg_win_rate*100:.2f}%")
    print(f"Average Return:          {avg_return*100:+.2f}%")
    print(f"Average Sharpe Ratio:     {avg_sharpe:.2f}")
    print(f"Maximum Drawdown:        {max_dd*100:.2f}%")
    
    # Performance assessment
    print("\n" + "="*80)
    print("PERFORMANCE ASSESSMENT")
    print("="*80)
    
    improvements = []
    for r in all_results:
        if r['trades_per_day'] <= 5:
            improvements.append(f"{r['symbol']}: Trade frequency OK ({r['trades_per_day']:.2f}/day)")
        else:
            improvements.append(f"{r['symbol']}: Still too many trades ({r['trades_per_day']:.2f}/day)")
        
        if r['win_rate'] >= 0.55:
            improvements.append(f"{r['symbol']}: Win rate target met ({r['win_rate']*100:.1f}%)")
        else:
            improvements.append(f"{r['symbol']}: Win rate below target ({r['win_rate']*100:.1f}%)")
        
        if r['sharpe_ratio'] >= 1.0:
            improvements.append(f"{r['symbol']}: Sharpe ratio target met ({r['sharpe_ratio']:.2f})")
        elif r['sharpe_ratio'] > 0:
            improvements.append(f"{r['symbol']}: Sharpe ratio positive ({r['sharpe_ratio']:.2f})")
        else:
            improvements.append(f"{r['symbol']}: Sharpe ratio negative ({r['sharpe_ratio']:.2f})")
    
    for improvement in improvements:
        print(f"  {improvement}")
    
    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
