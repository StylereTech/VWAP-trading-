"""
Fast Control Flip Backtest - 60 Days
Optimized for speed
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capital_allocation_ai.vwap_control_flip_strategy import VWAPControlFlipStrategy, StrategyParams


def generate_market_data(symbol: str, days: int = 60) -> pd.DataFrame:
    """Generate realistic market data - optimized."""
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
    
    # Faster generation
    volatility = np.ones(total_bars) * vol_base
    for i in range(1, min(100, total_bars)):  # Limit volatility calculation
        volatility[i] = vol_base * 0.6 + 0.7 * volatility[i-1] + 0.2 * abs(np.random.randn() * vol_base)
    volatility = np.clip(volatility, vol_base * 0.5, vol_base * 4)
    
    returns = np.random.randn(total_bars) * volatility
    prices = base_price + np.cumsum(returns)
    
    # Clip prices
    if symbol == 'GBPJPY':
        prices = np.clip(prices, 175, 200)
    elif symbol == 'BTCUSD':
        prices = np.clip(prices, 50000, 70000)
    else:
        prices = np.clip(prices, 2550, 2750)
    
    opens = prices.copy()
    highs = opens + abs(np.random.randn(total_bars) * volatility * opens * 0.002)
    lows = opens - abs(np.random.randn(total_bars) * volatility * opens * 0.002)
    closes = opens + np.random.randn(total_bars) * volatility * opens * 0.001
    
    # Ensure OHLC consistency
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))
    
    # Volumes
    base_vol = 5000 if symbol != 'BTCUSD' else 100
    volumes = (base_vol * (1 + abs(returns) * 5)).astype(int)
    volumes = np.clip(volumes, base_vol * 0.2, base_vol * 10)
    
    # Timestamps
    start_date = datetime.now() - timedelta(days=days)
    timestamps = []
    current_date = start_date
    bar_count = 0
    
    while bar_count < total_bars:
        if current_date.weekday() < 5:  # Skip weekends
            timestamps.append(current_date)
            bar_count += 1
        current_date += timedelta(minutes=5)
        if bar_count > 0 and bar_count % bars_per_day == 0:
            current_date += timedelta(days=1)
    
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


def backtest_strategy(data: pd.DataFrame, symbol: str, params: StrategyParams) -> Dict:
    """Backtest control flip strategy - optimized."""
    strategy = VWAPControlFlipStrategy(params)
    
    initial_equity = 10000.0
    equity = initial_equity
    trades = []
    max_drawdown = 0.0
    peak_equity = initial_equity
    
    # Process in batches to avoid memory issues
    batch_size = 1000
    for batch_start in range(0, len(data), batch_size):
        batch_end = min(batch_start + batch_size, len(data))
        batch_data = data.iloc[batch_start:batch_end]
        
        for idx, row in batch_data.iterrows():
            try:
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
                
                # Update drawdown
                peak_equity = max(peak_equity, equity)
                drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                max_drawdown = max(max_drawdown, drawdown)
                
            except Exception as e:
                print(f"Error at bar {idx}: {e}")
                continue
    
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
    
    # Sharpe ratio
    if len(trades_df) > 1:
        returns = trades_df['pnl'] / initial_equity
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if np.std(returns) > 0 else 0.0
    else:
        sharpe = 0.0
    
    # Trades per day
    days_traded = len(data) / 288
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
    """Run fast backtest."""
    print("="*80)
    print("VWAP CONTROL FLIP STRATEGY - FAST BACKTEST")
    print("60 Days: GBP/JPY, BTC/USD, Gold/USD")
    print("="*80)
    
    symbols = ['GBPJPY', 'BTCUSD', 'XAUUSD']
    
    # Optimized parameters
    optimized_params = {
        'GBPJPY': StrategyParams(
            band_k=2.1,
            band_k_list=(1.0, 2.1, 3.0),
            vol_mult=1.4,
            atr_cap_mult=2.8,
            require_nth_retest=3,
            touch_tol_atr_frac=0.05,
            trail_pct=0.007,
            cross_lookback_bars=12
        ),
        'BTCUSD': StrategyParams(
            band_k=2.3,
            band_k_list=(1.0, 2.3, 3.0),
            vol_mult=1.6,
            atr_cap_mult=3.0,
            require_nth_retest=3,
            touch_tol_atr_frac=0.06,
            trail_pct=0.01,
            cross_lookback_bars=15
        ),
        'XAUUSD': StrategyParams(
            band_k=1.9,
            band_k_list=(1.0, 1.9, 3.0),
            vol_mult=1.3,
            atr_cap_mult=2.5,
            require_nth_retest=3,
            touch_tol_atr_frac=0.04,
            trail_pct=0.006,
            cross_lookback_bars=10
        )
    }
    
    all_results = []
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"BACKTESTING {symbol}")
        print(f"{'='*80}")
        
        print(f"Generating data...")
        data = generate_market_data(symbol, days=60)
        print(f"Generated {len(data)} bars")
        
        print(f"Running backtest...")
        params = optimized_params[symbol]
        results = backtest_strategy(data, symbol, params)
        all_results.append(results)
        
        print(f"\n{symbol} Results:")
        print("-"*80)
        print(f"Total Trades:           {results['total_trades']}")
        print(f"Winning Trades:         {results['winning_trades']}")
        print(f"Losing Trades:          {results['losing_trades']}")
        print(f"Win Rate:               {results['win_rate']*100:.2f}%")
        print(f"Total Return:            {results['total_return']*100:+.2f}%")
        print(f"Final Equity:           ${results['final_equity']:,.2f}")
        print(f"Sharpe Ratio:            {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:            {results['max_drawdown']*100:.2f}%")
        print(f"Trades Per Day:          {results['trades_per_day']:.2f}")
        print(f"Profit Factor:           {results['profit_factor']:.2f}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY - ALL SYMBOLS")
    print("="*80)
    print(f"{'Symbol':<10} {'Trades':<8} {'Trades/Day':<12} {'Win Rate':<10} {'Return':<10} {'Sharpe':<8} {'Max DD':<8}")
    print("-"*80)
    
    for r in all_results:
        print(f"{r['symbol']:<10} {r['total_trades']:<8} {r['trades_per_day']:<12.2f} "
              f"{r['win_rate']*100:<9.1f}% {r['total_return']*100:<9.2f}% "
              f"{r['sharpe_ratio']:<8.2f} {r['max_drawdown']*100:<7.2f}%")
    
    # Assessment
    print("\n" + "="*80)
    print("PERFORMANCE ASSESSMENT")
    print("="*80)
    
    needs_optimization = False
    for r in all_results:
        if r['win_rate'] < 0.50 or r['sharpe_ratio'] < 0.5 or r['total_return'] < 0:
            needs_optimization = True
            print(f"{r['symbol']}: Needs optimization (Win Rate: {r['win_rate']*100:.1f}%, Sharpe: {r['sharpe_ratio']:.2f})")
        else:
            print(f"{r['symbol']}: Performance OK")
    
    if needs_optimization:
        print("\n" + "="*80)
        print("RECOMMENDATION: Run quantum optimization")
        print("="*80)
        print("Command: python optimize_control_flip_quantum.py")
    
    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
