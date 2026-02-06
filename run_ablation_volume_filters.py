"""
Ablation Ladder - Volume Filter Variants
Tests new volume filter designs to replace brittle multiplier
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capital_allocation_ai.vwap_control_flip_strategy import VWAPControlFlipStrategy, StrategyParams


def generate_market_data(symbol: str, days: int = 7) -> pd.DataFrame:
    """Generate market data."""
    bars_per_day = 288
    total_bars = days * bars_per_day
    np.random.seed(42)
    base_price, vol_base = 185.0, 0.0005
    volatility = np.full(total_bars, vol_base)
    returns = np.random.randn(total_bars) * volatility
    prices = base_price + np.cumsum(returns)
    prices = np.clip(prices, 175, 200)
    opens = prices.copy()
    highs = opens + abs(np.random.randn(total_bars) * 0.002 * opens)
    lows = opens - abs(np.random.randn(total_bars) * 0.002 * opens)
    closes = opens + np.random.randn(total_bars) * 0.001 * opens
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))
    base_vol = 5000
    volumes = (base_vol * (1 + abs(returns) * 5)).astype(int)
    volumes = np.clip(volumes, base_vol * 0.2, base_vol * 10)
    start_date = datetime.now() - timedelta(days=days)
    timestamps = pd.date_range(start=start_date, periods=total_bars, freq='5min')
    timestamps = [ts for ts in timestamps if ts.weekday() < 5][:total_bars]
    min_len = min(len(timestamps), len(opens))
    return pd.DataFrame({
        'timestamp': timestamps[:min_len],
        'open': opens[:min_len],
        'high': highs[:min_len],
        'low': lows[:min_len],
        'close': closes[:min_len],
        'volume': volumes[:min_len]
    })


def run_backtest(data, params):
    """Run backtest and return metrics."""
    strategy = VWAPControlFlipStrategy(params)
    equity = 10000.0
    trades = []
    
    crosses = 0
    retests = 0
    confirmations = 0
    
    prev_close = None
    prev_vwap = None
    
    for idx, row in data.iterrows():
        try:
            result = strategy.update(
                current_price=row['close'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                timestamp=row.get('timestamp')
            )
            
            current_vwap = result['indicators'].get('vwap', 0)
            current_atr = result['indicators'].get('atr20', 0)
            
            if prev_close is not None and prev_vwap is not None and current_vwap > 0:
                if (row['close'] > current_vwap and prev_close <= prev_vwap) or \
                   (row['close'] < current_vwap and prev_close >= prev_vwap):
                    crosses += 1
                
                tol = params.touch_tol_atr_frac * current_atr if current_atr > 0 else 0.001 * row['close']
                if (row['low'] <= current_vwap + tol and row['close'] > current_vwap) or \
                   (row['high'] >= current_vwap - tol and row['close'] < current_vwap):
                    retests += 1
            
            prev_close = row['close']
            prev_vwap = current_vwap
            
            if result['signals']['enter_long'] or result['signals']['enter_short']:
                confirmations += 1
            
            if result['signals']['enter_long'] and strategy.position is None:
                strategy.enter_long(row['close'])
                trades.append({'pnl': 0, 'entry': row['close'], 'side': 'long'})
            elif result['signals']['enter_short'] and strategy.position is None:
                strategy.enter_short(row['close'])
                trades.append({'pnl': 0, 'entry': row['close'], 'side': 'short'})
            elif result['signals']['exit'] and strategy.position is not None:
                pos = strategy.position
                size = equity / pos.entry_price * 0.1
                pnl = (row['close'] - pos.entry_price) * size if pos.side == "long" else (pos.entry_price - row['close']) * size
                pnl -= abs(size * row['close']) * 0.0001
                equity += pnl
                if trades:
                    trades[-1]['pnl'] = pnl
                strategy.exit_position()
        except:
            continue
    
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        win_rate = (trades_df['pnl'] > 0).sum() / len(trades)
        returns = trades_df['pnl'] / 10000
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0
        total_return = (equity - 10000) / 10000
        peak = 10000
        max_dd = 0
        equity_curve = [10000]
        for t in trades:
            equity_curve.append(equity_curve[-1] + t['pnl'])
            peak = max(peak, equity_curve[-1])
            max_dd = max(max_dd, (peak - equity_curve[-1]) / peak if peak > 0 else 0)
    else:
        win_rate = 0
        sharpe = 0
        total_return = 0
        max_dd = 0
    
    kill_ratio = confirmations / len(trades) if len(trades) > 0 else (confirmations if confirmations > 0 else 0)
    
    return {
        'crosses': crosses,
        'retests': retests,
        'confirmations': confirmations,
        'trades': len(trades),
        'kill_ratio': kill_ratio,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'return': total_return
    }


def main():
    """Run volume filter ablation."""
    print("="*80)
    print("ABLATION LADDER - VOLUME FILTER VARIANTS")
    print("="*80)
    
    symbol = 'GBPJPY'
    days = 7
    
    print(f"\nSymbol: {symbol}, Days: {days}")
    print("Generating data...")
    data = generate_market_data(symbol, days=days)
    print(f"{len(data)} bars generated")
    
    results = []
    
    # Baseline: retest=3, no volume filter
    print("\n" + "="*80)
    print("V0 - BASELINE (retest=3, no volume filter)")
    print("="*80)
    params_v0 = StrategyParams(
        band_k=2.0,
        volume_filter_type="multiplier",
        vol_mult=1.0,  # Effectively disabled
        atr_cap_mult=999.0,
        require_nth_retest=3,
        require_session_filter=False,
        cross_lookback_bars=20,
        touch_tol_atr_frac=0.20
    )
    v0 = run_backtest(data, params_v0)
    v0['rung'] = 'V0'
    v0['vol_filter'] = 'none'
    v0['vol_param'] = 'N/A'
    results.append(v0)
    print(f"Crosses: {v0['crosses']}, Retests: {v0['retests']}, Confirmations: {v0['confirmations']}, Trades: {v0['trades']}")
    print(f"Kill Ratio: {v0['kill_ratio']:.2f}, Sharpe: {v0['sharpe']:.2f}, DD: {v0['max_dd']:.2%}")
    
    # V1 - Original multiplier at 1.00 (should still be ~3)
    print("\n" + "="*80)
    print("V1 - Multiplier mult=1.00")
    print("="*80)
    params_v1 = StrategyParams(
        band_k=2.0,
        volume_filter_type="multiplier",
        vol_mult=1.00,
        atr_cap_mult=999.0,
        require_nth_retest=3,
        require_session_filter=False,
        cross_lookback_bars=20,
        touch_tol_atr_frac=0.20
    )
    v1 = run_backtest(data, params_v1)
    v1['rung'] = 'V1'
    v1['vol_filter'] = 'multiplier'
    v1['vol_param'] = '1.00'
    results.append(v1)
    print(f"Crosses: {v1['crosses']}, Retests: {v1['retests']}, Confirmations: {v1['confirmations']}, Trades: {v1['trades']}")
    print(f"Kill Ratio: {v1['kill_ratio']:.2f}, Sharpe: {v1['sharpe']:.2f}, DD: {v1['max_dd']:.2%}")
    
    # V2 - Percentile L=50, p=55
    print("\n" + "="*80)
    print("V2 - Percentile L=50, p=55")
    print("="*80)
    params_v2 = StrategyParams(
        band_k=2.0,
        volume_filter_type="percentile",
        vol_percentile_L=50,
        vol_percentile_p=55.0,
        atr_cap_mult=999.0,
        require_nth_retest=3,
        require_session_filter=False,
        cross_lookback_bars=20,
        touch_tol_atr_frac=0.20
    )
    v2 = run_backtest(data, params_v2)
    v2['rung'] = 'V2'
    v2['vol_filter'] = 'percentile'
    v2['vol_param'] = 'L=50,p=55'
    results.append(v2)
    print(f"Crosses: {v2['crosses']}, Retests: {v2['retests']}, Confirmations: {v2['confirmations']}, Trades: {v2['trades']}")
    print(f"Kill Ratio: {v2['kill_ratio']:.2f}, Sharpe: {v2['sharpe']:.2f}, DD: {v2['max_dd']:.2%}")
    
    # V3 - Percentile L=50, p=65
    print("\n" + "="*80)
    print("V3 - Percentile L=50, p=65")
    print("="*80)
    params_v3 = StrategyParams(
        band_k=2.0,
        volume_filter_type="percentile",
        vol_percentile_L=50,
        vol_percentile_p=65.0,
        atr_cap_mult=999.0,
        require_nth_retest=3,
        require_session_filter=False,
        cross_lookback_bars=20,
        touch_tol_atr_frac=0.20
    )
    v3 = run_backtest(data, params_v3)
    v3['rung'] = 'V3'
    v3['vol_filter'] = 'percentile'
    v3['vol_param'] = 'L=50,p=65'
    results.append(v3)
    print(f"Crosses: {v3['crosses']}, Retests: {v3['retests']}, Confirmations: {v3['confirmations']}, Trades: {v3['trades']}")
    print(f"Kill Ratio: {v3['kill_ratio']:.2f}, Sharpe: {v3['sharpe']:.2f}, DD: {v3['max_dd']:.2%}")
    
    # V4 - Impulse (vol>=SMA OR body>=0.8*ATR)
    print("\n" + "="*80)
    print("V4 - Impulse (vol>=SMA OR body>=0.8*ATR)")
    print("="*80)
    params_v4 = StrategyParams(
        band_k=2.0,
        volume_filter_type="impulse",
        body_atr_thresh=0.8,
        atr_cap_mult=999.0,
        require_nth_retest=3,
        require_session_filter=False,
        cross_lookback_bars=20,
        touch_tol_atr_frac=0.20
    )
    v4 = run_backtest(data, params_v4)
    v4['rung'] = 'V4'
    v4['vol_filter'] = 'impulse'
    v4['vol_param'] = 'body>=0.8*ATR'
    results.append(v4)
    print(f"Crosses: {v4['crosses']}, Retests: {v4['retests']}, Confirmations: {v4['confirmations']}, Trades: {v4['trades']}")
    print(f"Kill Ratio: {v4['kill_ratio']:.2f}, Sharpe: {v4['sharpe']:.2f}, DD: {v4['max_dd']:.2%}")
    
    # V5 - Reclaim Quality (close-vwap>=0.15*ATR AND body_ratio>=0.55)
    print("\n" + "="*80)
    print("V5 - Reclaim Quality (close-vwap>=0.15*ATR AND body_ratio>=0.55)")
    print("="*80)
    params_v5 = StrategyParams(
        band_k=2.0,
        volume_filter_type="reclaim_quality",
        reclaim_atr_thresh=0.15,
        body_ratio_thresh=0.55,
        atr_cap_mult=999.0,
        require_nth_retest=3,
        require_session_filter=False,
        cross_lookback_bars=20,
        touch_tol_atr_frac=0.20
    )
    v5 = run_backtest(data, params_v5)
    v5['rung'] = 'V5'
    v5['vol_filter'] = 'reclaim_quality'
    v5['vol_param'] = 'dist>=0.15*ATR,body>=0.55'
    results.append(v5)
    print(f"Crosses: {v5['crosses']}, Retests: {v5['retests']}, Confirmations: {v5['confirmations']}, Trades: {v5['trades']}")
    print(f"Kill Ratio: {v5['kill_ratio']:.2f}, Sharpe: {v5['sharpe']:.2f}, DD: {v5['max_dd']:.2%}")
    
    # Summary table
    print("\n" + "="*80)
    print("VOLUME FILTER ABLATION RESULTS")
    print("="*80)
    print(f"{'Rung':<6} {'Vol_Filter':<15} {'Vol_Param':<20} {'Trades':<8} {'Confirms':<10} {'Sharpe':<8} {'DD':<8}")
    print("-"*80)
    for r in results:
        print(f"{r['rung']:<6} {r['vol_filter']:<15} {r['vol_param']:<20} {r['trades']:<8} "
              f"{r['confirmations']:<10} {r['sharpe']:<8.2f} {r['max_dd']:<8.2%}")
    
    print("\n" + "="*80)
    print("DETAILED METRICS")
    print("="*80)
    print(f"{'Rung':<6} {'Crosses':<10} {'Retests':<10} {'Confirms':<10} {'Trades':<8} {'Kill_Ratio':<12}")
    print("-"*80)
    for r in results:
        print(f"{r['rung']:<6} {r['crosses']:<10} {r['retests']:<10} {r['confirmations']:<10} "
              f"{r['trades']:<8} {r['kill_ratio']:<12.2f}")
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    baseline_trades = results[0]['trades']
    print(f"Baseline (no volume filter): {baseline_trades} trades")
    
    best = max([r for r in results if r['trades'] > 0], key=lambda x: x['trades'], default=None)
    if best:
        print(f"Best volume filter: {best['rung']} ({best['vol_filter']}) with {best['trades']} trades")
        print(f"  Sharpe: {best['sharpe']:.2f}, DD: {best['max_dd']:.2%}")
    
    print("\nRecommendation:")
    print("  - Use filter that produces non-zero trades")
    print("  - Prefer stable trade count across symbols")
    print("  - Test on real data before optimization")
    print("="*80)


if __name__ == "__main__":
    main()
