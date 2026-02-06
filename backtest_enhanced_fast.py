"""
Fast Enhanced Backtest - Optimized for Speed
Processes bars efficiently with progress tracking
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_control_flip_strategy import EnhancedVWAPControlFlipStrategy, EnhancedStrategyParams


def generate_market_data(symbol: str, days: int = 60) -> pd.DataFrame:
    """Generate market data - optimized."""
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


def backtest_fast(data: pd.DataFrame, symbol: str, params: EnhancedStrategyParams):
    """Fast backtest with progress updates."""
    strategy = EnhancedVWAPControlFlipStrategy(params)
    
    equity = 10000.0
    trades = []
    max_dd = 0.0
    peak = equity
    
    total_bars = len(data)
    chunk_size = 1000  # Process in chunks
    
    print(f"  Processing {total_bars} bars in chunks of {chunk_size}...")
    
    for chunk_start in range(0, total_bars, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_bars)
        chunk_data = data.iloc[chunk_start:chunk_end]
        
        progress = int((chunk_end / total_bars) * 100)
        if chunk_start % (chunk_size * 3) == 0:  # Update every 3 chunks
            print(f"  Progress: {progress}%", end='\r')
        
        for idx, row in chunk_data.iterrows():
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
                    size = equity / pos.entry_price * 0.1
                    pnl = (row['close'] - pos.entry_price) * size if pos.side == "long" else (pos.entry_price - row['close']) * size
                    pnl -= abs(size * row['close']) * 0.0001
                    equity += pnl
                    
                    risk = abs(pos.entry_price - (pos.stop_price or pos.entry_price * 0.99))
                    reward = abs(row['close'] - pos.entry_price) if pos.side == "long" else abs(pos.entry_price - row['close'])
                    rr = reward / risk if risk > 0 else 0
                    
                    trades.append({'pnl': pnl, 'rr': rr})
                    strategy.exit_position()
                
                peak = max(peak, equity)
                max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0)
            except Exception as e:
                if chunk_start == 0:  # Only print first error
                    print(f"\n  Warning: {e}")
                continue
    
    print(f"  Progress: 100%")
    
    if strategy.position is not None:
        final_price = data.iloc[-1]['close']
        pos = strategy.position
        size = equity / pos.entry_price * 0.1
        pnl = (final_price - pos.entry_price) * size if pos.side == "long" else (pos.entry_price - final_price) * size
        equity += pnl
        trades.append({'pnl': pnl, 'rr': 0})
    
    if len(trades) == 0:
        return {
            'total_trades': 0, 'winning_trades': 0, 'losing_trades': 0,
            'win_rate': 0.0, 'total_return': 0.0, 'final_equity': 10000.0,
            'sharpe_ratio': 0.0, 'max_drawdown': 0.0, 'trades_per_day': 0.0,
            'avg_win': 0.0, 'avg_loss': 0.0, 'profit_factor': 0.0, 'avg_rr': 0.0
        }
    
    trades_df = pd.DataFrame(trades)
    winning = (trades_df['pnl'] > 0).sum()
    losing = len(trades) - winning
    win_rate = winning / len(trades)
    total_return = (equity - 10000) / 10000
    
    returns = trades_df['pnl'] / 10000
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0.0
    
    days_traded = len(data) / 288
    trades_per_day = len(trades) / days_traded if days_traded > 0 else 0.0
    
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if (trades_df['pnl'] > 0).any() else 0.0
    avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (trades_df['pnl'] < 0).any() else 0.0
    
    total_wins = trades_df[trades_df['pnl'] > 0]['pnl'].sum() if (trades_df['pnl'] > 0).any() else 0.0
    total_losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if (trades_df['pnl'] < 0).any() else 1.0
    profit_factor = total_wins / total_losses if total_losses > 0 else 0.0
    
    avg_rr = trades_df['rr'].mean() if 'rr' in trades_df.columns else 0.0
    
    return {
        'symbol': symbol,
        'total_trades': len(trades),
        'winning_trades': winning,
        'losing_trades': losing,
        'win_rate': win_rate,
        'total_return': total_return,
        'final_equity': equity,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'trades_per_day': trades_per_day,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'avg_rr': avg_rr
    }


def main():
    """Run fast enhanced backtest."""
    print("="*80)
    print("ENHANCED VWAP CONTROL FLIP - FAST 60 DAY BACKTEST")
    print("With R:R Optimization & Stronger Filters")
    print("="*80)
    
    symbols = ['GBPJPY', 'BTCUSD', 'XAUUSD']
    
    params_map = {
        'GBPJPY': EnhancedStrategyParams(
            band_k=2.0, vol_mult=1.5, atr_cap_mult=2.0,
            require_nth_retest=3, touch_tol_atr_frac=0.05,
            trail_pct=0.007, cross_lookback_bars=12,
            require_session_filter=False,
            stretch_targets=True, min_rr_ratio=2.5, breakeven_after_rr=1.0,
            require_trend_alignment=True, require_fvg_fill=True,
            atr_cap_tighter=2.0, require_volume_imbalance=True,
            volume_imbalance_threshold=1.2, symbol_type="FOREX"
        ),
        'BTCUSD': EnhancedStrategyParams(
            band_k=2.5, vol_mult=1.6, atr_cap_mult=2.0,
            require_nth_retest=3, touch_tol_atr_frac=0.06,
            trail_pct=0.01, cross_lookback_bars=15,
            require_session_filter=False,
            stretch_targets=True, min_rr_ratio=3.0, breakeven_after_rr=1.0,
            require_trend_alignment=True, require_fvg_fill=True,
            atr_cap_tighter=2.0, require_volume_imbalance=True,
            volume_imbalance_threshold=1.3, symbol_type="CRYPTO"
        ),
        'XAUUSD': EnhancedStrategyParams(
            band_k=1.9, vol_mult=1.3, atr_cap_mult=2.0,
            require_nth_retest=3, touch_tol_atr_frac=0.04,
            trail_pct=0.006, cross_lookback_bars=10,
            require_session_filter=False,
            stretch_targets=True, min_rr_ratio=2.5, breakeven_after_rr=1.0,
            require_trend_alignment=True, require_fvg_fill=True,
            atr_cap_tighter=2.0, require_volume_imbalance=True,
            volume_imbalance_threshold=1.2, use_ema200_filter=True,
            symbol_type="METAL"
        )
    }
    
    all_results = []
    total_start = time.time()
    
    for idx, symbol in enumerate(symbols):
        print(f"\n{'='*80}")
        print(f"{symbol} ({idx+1}/{len(symbols)})")
        print(f"{'='*80}")
        
        symbol_start = time.time()
        
        print(f"Generating 60 days of data...")
        data = generate_market_data(symbol, days=60)
        print(f"Generated {len(data)} bars")
        
        print(f"\nRunning enhanced backtest...")
        results = backtest_fast(data, symbol, params_map[symbol])
        all_results.append(results)
        
        elapsed = time.time() - symbol_start
        
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
        print(f"Average R:R:            {results['avg_rr']:.2f}")
        print(f"Time Taken:             {elapsed:.1f}s ({elapsed/60:.1f}m)")
    
    total_time = time.time() - total_start
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY - ENHANCED STRATEGY (60 DAYS)")
    print("="*80)
    print(f"{'Symbol':<10} {'Trades':<8} {'Trades/Day':<12} {'Win Rate':<10} {'Return':<10} {'Sharpe':<8} {'Avg R:R':<8}")
    print("-"*80)
    
    for r in all_results:
        print(f"{r['symbol']:<10} {r['total_trades']:<8} {r['trades_per_day']:<12.2f} "
              f"{r['win_rate']*100:<9.1f}% {r['total_return']*100:<9.2f}% "
              f"{r['sharpe_ratio']:<8.2f} {r['avg_rr']:<8.2f}")
    
    # Combined
    print("\n" + "="*80)
    print("COMBINED PORTFOLIO")
    print("="*80)
    
    total_trades = sum(r['total_trades'] for r in all_results)
    avg_win_rate = np.mean([r['win_rate'] for r in all_results])
    avg_return = np.mean([r['total_return'] for r in all_results])
    avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_results])
    avg_rr = np.mean([r['avg_rr'] for r in all_results])
    max_dd = max([r['max_drawdown'] for r in all_results])
    
    print(f"Total Trades:            {total_trades}")
    print(f"Average Win Rate:        {avg_win_rate*100:.2f}%")
    print(f"Average Return:          {avg_return*100:+.2f}%")
    print(f"Average Sharpe Ratio:     {avg_sharpe:.2f}")
    print(f"Average R:R:              {avg_rr:.2f}")
    print(f"Maximum Drawdown:        {max_dd*100:.2f}%")
    
    print(f"\nTotal Backtest Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    print("\n" + "="*80)
    print("ENHANCEMENTS APPLIED:")
    print("="*80)
    print("✓ Stretched targets (winners run to opposite band)")
    print("✓ Trend alignment filter (EMA-90)")
    print("✓ FVG fill confirmation")
    print("✓ Tighter ATR cap (2.0x)")
    print("✓ Volume imbalance requirement")
    print("✓ Breakeven after 1R, then trail")
    print("✓ Symbol-specific optimizations")
    print("="*80)
    print("BACKTEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
