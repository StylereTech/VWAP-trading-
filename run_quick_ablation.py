"""
Quick Ablation - Test key filter combinations to find bottleneck
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


def quick_backtest(data, params):
    """Quick backtest returning key metrics."""
    strategy = VWAPControlFlipStrategy(params)
    trades = 0
    cross_events = 0
    retest_events = 0
    confirm_events = 0
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
                if (row['close'] > current_vwap and prev_close <= prev_vwap) or (row['close'] < current_vwap and prev_close >= prev_vwap):
                    cross_events += 1
                tol = params.touch_tol_atr_frac * current_atr if current_atr > 0 else 0.001 * row['close']
                if (row['low'] <= current_vwap + tol and row['close'] > current_vwap) or (row['high'] >= current_vwap - tol and row['close'] < current_vwap):
                    retest_events += 1
            
            prev_close = row['close']
            prev_vwap = current_vwap
            
            if result['signals']['enter_long'] or result['signals']['enter_short']:
                confirm_events += 1
            
            if result['signals']['enter_long'] and strategy.position is None:
                strategy.enter_long(row['close'])
                trades += 1
            elif result['signals']['enter_short'] and strategy.position is None:
                strategy.enter_short(row['close'])
                trades += 1
            elif result['signals']['exit'] and strategy.position is not None:
                strategy.exit_position()
        except:
            continue
    
    return {'trades': trades, 'cross_events': cross_events, 'retest_events': retest_events, 'confirm_events': confirm_events}


def main():
    """Run quick ablation."""
    print("="*80)
    print("QUICK ABLATION TEST")
    print("="*80)
    
    data = generate_market_data('GBPJPY', days=7)
    print(f"Data: {len(data)} bars")
    
    tests = [
        ('1. Base (retest=1, no filters)', StrategyParams(vol_mult=1.0, atr_cap_mult=999.0, require_nth_retest=1, require_session_filter=False, touch_tol_atr_frac=0.20)),
        ('2. Add retest=2', StrategyParams(vol_mult=1.0, atr_cap_mult=999.0, require_nth_retest=2, require_session_filter=False, touch_tol_atr_frac=0.20)),
        ('3. Add retest=3', StrategyParams(vol_mult=1.0, atr_cap_mult=999.0, require_nth_retest=3, require_session_filter=False, touch_tol_atr_frac=0.20)),
        ('4. Add ATR cap=3.0', StrategyParams(vol_mult=1.0, atr_cap_mult=3.0, require_nth_retest=3, require_session_filter=False, touch_tol_atr_frac=0.20)),
        ('5. Add ATR cap=2.5', StrategyParams(vol_mult=1.0, atr_cap_mult=2.5, require_nth_retest=3, require_session_filter=False, touch_tol_atr_frac=0.20)),
        ('6. Add volume=1.1', StrategyParams(vol_mult=1.1, atr_cap_mult=2.5, require_nth_retest=3, require_session_filter=False, touch_tol_atr_frac=0.20)),
        ('7. Add volume=1.3', StrategyParams(vol_mult=1.3, atr_cap_mult=2.5, require_nth_retest=3, require_session_filter=False, touch_tol_atr_frac=0.20)),
        ('8. Add volume=1.5', StrategyParams(vol_mult=1.5, atr_cap_mult=2.5, require_nth_retest=3, require_session_filter=False, touch_tol_atr_frac=0.20)),
    ]
    
    results = []
    for name, params in tests:
        r = quick_backtest(data, params)
        r['name'] = name
        results.append(r)
        print(f"{name}: trades={r['trades']}, cross={r['cross_events']}, retest={r['retest_events']}, confirm={r['confirm_events']}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"{'Test':<35} {'Trades':<8} {'Cross':<8} {'Retest':<8} {'Confirm':<8}")
    print("-"*80)
    for r in results:
        print(f"{r['name']:<35} {r['trades']:<8} {r['cross_events']:<8} {r['retest_events']:<8} {r['confirm_events']:<8}")
    
    # Find where trades drop
    for i in range(1, len(results)):
        if results[i]['trades'] == 0 and results[i-1]['trades'] > 0:
            print(f"\nTrades dropped to 0 at: {results[i]['name']}")
            print(f"Previous had {results[i-1]['trades']} trades")
            print(f"Filter added in this step is likely the culprit")


if __name__ == "__main__":
    main()
