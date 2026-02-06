"""
Run Backtest on All Symbols - Optimized Version
GBP/JPY, BTC/USD, Gold/USD - 60 Days
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict
import sys
import os
import time

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
    
    # Faster volatility generation
    volatility = np.full(total_bars, vol_base)
    for i in range(1, min(200, total_bars)):
        volatility[i] = vol_base * 0.6 + 0.7 * volatility[i-1] + 0.2 * abs(np.random.randn() * vol_base)
    volatility = np.clip(volatility, vol_base * 0.5, vol_base * 4)
    
    returns = np.random.randn(total_bars) * volatility
    for i in range(1, min(1000, total_bars)):
        if abs(returns[i-1]) > vol_base * 2:
            returns[i] -= 0.3 * returns[i-1]
    
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
    
    # Add some volume spikes
    spike_count = int(total_bars * 0.05)
    spike_indices = np.random.choice(total_bars, size=spike_count, replace=False)
    volumes[spike_indices] = volumes[spike_indices] * np.random.uniform(1.5, 2.5, spike_count)
    
    # Timestamps - optimized
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
    
    # Process bars with progress updates
    total_bars = len(data)
    last_progress = 0
    
    for idx, (_, row) in enumerate(data.iterrows()):
        # Progress update every 10%
        progress = int((idx / total_bars) * 100)
        if progress >= last_progress + 10:
            print(f"  Progress: {progress}%", end='\r')
            last_progress = progress
        
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
            print(f"\nError at bar {idx}: {e}")
            continue
    
    print("  Progress: 100%")
    
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
    """Run backtest on all symbols."""
    print("="*80)
    print("VWAP CONTROL FLIP STRATEGY - BACKTEST ALL SYMBOLS")
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
    start_time = time.time()
    
    for symbol_idx, symbol in enumerate(symbols):
        print(f"\n{'='*80}")
        print(f"BACKTESTING {symbol} ({symbol_idx+1}/{len(symbols)})")
        print(f"{'='*80}")
        
        symbol_start = time.time()
        
        print(f"Generating 60 days of {symbol} data...")
        data = generate_market_data(symbol, days=60)
        print(f"Generated {len(data)} bars")
        print(f"Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
        
        print(f"\nRunning backtest with optimized parameters...")
        params = optimized_params[symbol]
        results = backtest_strategy(data, symbol, params)
        all_results.append(results)
        
        symbol_time = time.time() - symbol_start
        
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
        print(f"Time Taken:             {symbol_time:.1f} seconds")
    
    total_time = time.time() - start_time
    
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
    
    for r in all_results:
        status = []
        if r['trades_per_day'] <= 5:
            status.append("Trade frequency OK")
        else:
            status.append("Too many trades")
        
        if r['win_rate'] >= 0.55:
            status.append("Win rate target met")
        elif r['win_rate'] >= 0.45:
            status.append("Win rate acceptable")
        else:
            status.append("Win rate below target")
        
        if r['sharpe_ratio'] >= 1.0:
            status.append("Sharpe excellent")
        elif r['sharpe_ratio'] >= 0.5:
            status.append("Sharpe acceptable")
        elif r['sharpe_ratio'] > 0:
            status.append("Sharpe positive")
        else:
            status.append("Sharpe negative")
        
        print(f"{r['symbol']}: {', '.join(status)}")
    
    print(f"\nTotal Backtest Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    print("\n" + "="*80)
    print("BACKTEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
