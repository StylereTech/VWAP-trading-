"""
Test Production Baseline Config
Uses synthetic data as fallback if real data not available
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capital_allocation_ai.vwap_control_flip_strategy import VWAPControlFlipStrategy, StrategyParams
from load_real_market_data import load_from_csv, load_from_traderlocker, prepare_data_for_backtest


def generate_market_data(symbol: str, days: int = 30) -> pd.DataFrame:
    """Generate synthetic data (fallback)."""
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
    trades_list = []
    
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
                if (row['close'] > current_vwap and prev_close <= prev_vwap) or (row['close'] < current_vwap and prev_close >= prev_vwap):
                    crosses += 1
                tol = params.touch_tol_atr_frac * current_atr if current_atr > 0 else 0.001 * row['close']
                if (row['low'] <= current_vwap + tol and row['close'] > current_vwap) or (row['high'] >= current_vwap - tol and row['close'] < current_vwap):
                    retests += 1
            
            prev_close = row['close']
            prev_vwap = current_vwap
            
            if result['signals']['enter_long'] or result['signals']['enter_short']:
                confirmations += 1
            
            if result['signals']['enter_long'] and strategy.position is None:
                strategy.enter_long(row['close'])
                trades_list.append({'pnl': 0, 'entry': row['close'], 'side': 'long'})
            elif result['signals']['enter_short'] and strategy.position is None:
                strategy.enter_short(row['close'])
                trades_list.append({'pnl': 0, 'entry': row['close'], 'side': 'short'})
            elif result['signals']['exit'] and strategy.position is not None:
                pos = strategy.position
                size = equity / pos.entry_price * 0.1
                pnl = (row['close'] - pos.entry_price) * size if pos.side == "long" else (pos.entry_price - row['close']) * size
                pnl -= abs(size * row['close']) * 0.0001
                equity += pnl
                if trades_list:
                    trades_list[-1]['pnl'] = pnl
                strategy.exit_position()
        except:
            continue
    
    if len(trades_list) > 0:
        trades_df = pd.DataFrame(trades_list)
        win_rate = (trades_df['pnl'] > 0).sum() / len(trades_df)
        returns = trades_df['pnl'] / 10000
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0
        total_return = (equity - 10000) / 10000
        peak = 10000
        max_dd = 0
        equity_curve = [10000]
        for t in trades_list:
            equity_curve.append(equity_curve[-1] + t['pnl'])
            peak = max(peak, equity_curve[-1])
            max_dd = max(max_dd, (peak - equity_curve[-1]) / peak if peak > 0 else 0)
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if (trades_df['pnl'] > 0).any() else 0
        avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if (trades_df['pnl'] < 0).any() else 1
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0
    else:
        win_rate = 0
        sharpe = 0
        total_return = 0
        max_dd = 0
        payoff_ratio = 0
    
    kill_ratio = confirmations / len(trades_list) if len(trades_list) > 0 else (confirmations if confirmations > 0 else 0)
    
    return {
        'crosses': crosses,
        'retests': retests,
        'confirmations': confirmations,
        'trades': len(trades_list),
        'kill_ratio': kill_ratio,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'win_rate': win_rate,
        'return': total_return,
        'payoff_ratio': payoff_ratio,
        'trades_list': trades_list
    }


def main():
    """Test production baseline config."""
    print("="*80)
    print("PRODUCTION BASELINE CONFIG TEST")
    print("="*80)
    
    symbol = 'GBPJPY'
    days = 30
    
    # Try to load real data
    data = None
    csv_paths = [
        f'{symbol.lower()}_5m.csv',
        f'{symbol}_5m.csv',
        f'data/{symbol.lower()}_5m.csv',
        f'data/{symbol}_5m.csv'
    ]
    
    for csv_path in csv_paths:
        if os.path.exists(csv_path):
            print(f"\nLoading REAL data from CSV: {csv_path}")
            try:
                data = load_from_csv(csv_path, symbol)
                data = prepare_data_for_backtest(data, symbol)
                # Limit to 30 days if more
                if len(data) > days * 288:
                    data = data.tail(days * 288).reset_index(drop=True)
                break
            except Exception as e:
                print(f"Failed to load {csv_path}: {e}")
                continue
    
    if data is None:
        print("\n[WARN] No real data found - using SYNTHETIC data for testing")
        print("   NOTE: Results are for logic validation only, not production")
        data = generate_market_data(symbol, days=days)
    
    print(f"\nData: {len(data)} bars ({days} days)")
    
    # Production Baseline Config
    print("\n" + "="*80)
    print("PRODUCTION BASELINE CONFIGURATION")
    print("="*80)
    print("retest_window = (2, 4)")
    print("volume_filter_type = 'percentile'")
    print("vol_percentile_L = 50")
    print("vol_percentile_p = 60")
    print("atr_cap_mult = 3.0")
    print("session_filter = False")
    print("="*80)
    
    params = StrategyParams(
        band_k=2.0,
        volume_filter_type="percentile",
        vol_percentile_L=50,
        vol_percentile_p=60.0,
        atr_cap_mult=3.0,
        require_nth_retest=3,  # Will use flexible window (2-4) in code
        require_session_filter=False,
        cross_lookback_bars=20,
        touch_tol_atr_frac=0.20
    )
    
    print("\nRunning backtest...")
    start = time.time()
    stats = run_backtest(data, params)
    elapsed = time.time() - start
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"Crosses: {stats['crosses']}")
    print(f"Retests: {stats['retests']}")
    print(f"Confirmations: {stats['confirmations']}")
    print(f"Trades: {stats['trades']}")
    print(f"Trades per day: {stats['trades'] / days:.2f}")
    print(f"Win Rate: {stats['win_rate']*100:.1f}%")
    print(f"Total Return: {stats['return']*100:+.2f}%")
    print(f"Sharpe: {stats['sharpe']:.2f}")
    print(f"Max DD: {stats['max_dd']*100:.2f}%")
    print(f"Payoff Ratio: {stats['payoff_ratio']:.2f}")
    print(f"Kill Ratio: {stats['kill_ratio']:.2f}")
    print(f"Time: {elapsed:.1f}s")
    
    print("\n" + "="*80)
    print("ACCEPTANCE CRITERIA CHECK")
    print("="*80)
    print(f"Trades >= 40: {'[PASS]' if stats['trades'] >= 40 else '[FAIL]'} ({stats['trades']})")
    print(f"Max DD < 20%: {'[PASS]' if stats['max_dd'] < 0.20 else '[FAIL]'} ({stats['max_dd']*100:.2f}%)")
    print(f"Payoff Ratio < 8: {'[PASS]' if stats['payoff_ratio'] < 8 else '[FAIL]'} ({stats['payoff_ratio']:.2f})")
    print(f"Win Rate < 90%: {'[PASS]' if stats['win_rate'] < 0.90 else '[FAIL]'} ({stats['win_rate']*100:.1f}%)")
    
    if stats['trades'] < 40:
        print("\n[WARN] Trade count below target - may need to loosen filters")
        print("   Try: percentile p=55, or retest window (1-4)")
    
    print("="*80)


if __name__ == "__main__":
    main()
