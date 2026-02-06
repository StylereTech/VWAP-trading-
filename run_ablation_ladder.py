"""
Ablation Ladder - Systematic Filter Testing
Identifies which filter kills trades
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
    
    # Event counters
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
            
            # Count crosses
            if prev_close is not None and prev_vwap is not None and current_vwap > 0:
                if (row['close'] > current_vwap and prev_close <= prev_vwap) or \
                   (row['close'] < current_vwap and prev_close >= prev_vwap):
                    crosses += 1
                
                # Count retests
                tol = params.touch_tol_atr_frac * current_atr if current_atr > 0 else 0.001 * row['close']
                if (row['low'] <= current_vwap + tol and row['close'] > current_vwap) or \
                   (row['high'] >= current_vwap - tol and row['close'] < current_vwap):
                    retests += 1
            
            prev_close = row['close']
            prev_vwap = current_vwap
            
            # Count confirmations
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
    
    # Calculate metrics
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
    """Run ablation ladder."""
    print("="*80)
    print("ABLATION LADDER - FILTER TESTING")
    print("="*80)
    
    symbol = 'GBPJPY'
    days = 7
    
    print(f"\nSymbol: {symbol}, Days: {days}")
    print("Generating data...")
    data = generate_market_data(symbol, days=days)
    print(f"{len(data)} bars generated")
    
    results = []
    
    # R0 - Core only
    print("\n" + "="*80)
    print("R0 - CORE ONLY (baseline)")
    print("="*80)
    params_r0 = StrategyParams(
        band_k=2.0,
        vol_mult=1.0,  # No volume filter
        atr_cap_mult=999.0,  # No ATR cap
        require_nth_retest=1,
        require_session_filter=False,
        cross_lookback_bars=20,
        touch_tol_atr_frac=0.20
    )
    r0 = run_backtest(data, params_r0)
    r0['rung'] = 'R0'
    r0['retest'] = 1
    r0['vol_mult'] = 1.0
    r0['atr_cap'] = 999.0
    r0['ema_filter'] = False
    r0['session'] = False
    results.append(r0)
    print(f"Crosses: {r0['crosses']}, Retests: {r0['retests']}, Confirmations: {r0['confirmations']}, Trades: {r0['trades']}")
    print(f"Kill Ratio: {r0['kill_ratio']:.2f}, Sharpe: {r0['sharpe']:.2f}, DD: {r0['max_dd']:.2%}")
    
    # R1 - Add retest_count = 2
    print("\n" + "="*80)
    print("R1a - Add retest_count = 2")
    print("="*80)
    params_r1a = StrategyParams(
        band_k=2.0,
        vol_mult=1.0,
        atr_cap_mult=999.0,
        require_nth_retest=2,
        require_session_filter=False,
        cross_lookback_bars=20,
        touch_tol_atr_frac=0.20
    )
    r1a = run_backtest(data, params_r1a)
    r1a['rung'] = 'R1a'
    r1a['retest'] = 2
    r1a['vol_mult'] = 1.0
    r1a['atr_cap'] = 999.0
    r1a['ema_filter'] = False
    r1a['session'] = False
    results.append(r1a)
    print(f"Crosses: {r1a['crosses']}, Retests: {r1a['retests']}, Confirmations: {r1a['confirmations']}, Trades: {r1a['trades']}")
    print(f"Kill Ratio: {r1a['kill_ratio']:.2f}, Sharpe: {r1a['sharpe']:.2f}, DD: {r1a['max_dd']:.2%}")
    
    # R1 - Add retest_count = 3
    print("\n" + "="*80)
    print("R1b - Add retest_count = 3")
    print("="*80)
    params_r1b = StrategyParams(
        band_k=2.0,
        vol_mult=1.0,
        atr_cap_mult=999.0,
        require_nth_retest=3,
        require_session_filter=False,
        cross_lookback_bars=20,
        touch_tol_atr_frac=0.20
    )
    r1b = run_backtest(data, params_r1b)
    r1b['rung'] = 'R1b'
    r1b['retest'] = 3
    r1b['vol_mult'] = 1.0
    r1b['atr_cap'] = 999.0
    r1b['ema_filter'] = False
    r1b['session'] = False
    results.append(r1b)
    print(f"Crosses: {r1b['crosses']}, Retests: {r1b['retests']}, Confirmations: {r1b['confirmations']}, Trades: {r1b['trades']}")
    print(f"Kill Ratio: {r1b['kill_ratio']:.2f}, Sharpe: {r1b['sharpe']:.2f}, DD: {r1b['max_dd']:.2%}")
    
    # R2 - Add volume filter (starting from retest=3 baseline)
    base_params = params_r1b
    for vol_mult in [1.05, 1.10, 1.20, 1.30]:
        print(f"\n{'='*80}")
        print(f"R2 - Add volume_mult = {vol_mult}")
        print("="*80)
        params = StrategyParams(
            band_k=2.0,
            vol_mult=vol_mult,
            atr_cap_mult=999.0,
            require_nth_retest=3,
            require_session_filter=False,
            cross_lookback_bars=20,
            touch_tol_atr_frac=0.20
        )
        r = run_backtest(data, params)
        r['rung'] = f'R2-{vol_mult}'
        r['retest'] = 3
        r['vol_mult'] = vol_mult
        r['atr_cap'] = 999.0
        r['ema_filter'] = False
        r['session'] = False
        results.append(r)
        print(f"Crosses: {r['crosses']}, Retests: {r['retests']}, Confirmations: {r['confirmations']}, Trades: {r['trades']}")
        print(f"Kill Ratio: {r['kill_ratio']:.2f}, Sharpe: {r['sharpe']:.2f}, DD: {r['max_dd']:.2%}")
        if r['trades'] == 0:
            print("*** TRADES DROPPED TO ZERO - Volume filter is the culprit ***")
            break
    
    # R3 - Add ATR cap (using last working volume_mult)
    last_working_vol = 1.30 if results[-1]['trades'] > 0 else 1.20
    for atr_cap in [4.0, 3.5, 3.0, 2.5]:
        print(f"\n{'='*80}")
        print(f"R3 - Add ATR cap = {atr_cap}")
        print("="*80)
        params = StrategyParams(
            band_k=2.0,
            vol_mult=last_working_vol,
            atr_cap_mult=atr_cap,
            require_nth_retest=3,
            require_session_filter=False,
            cross_lookback_bars=20,
            touch_tol_atr_frac=0.20
        )
        r = run_backtest(data, params)
        r['rung'] = f'R3-{atr_cap}'
        r['retest'] = 3
        r['vol_mult'] = last_working_vol
        r['atr_cap'] = atr_cap
        r['ema_filter'] = False
        r['session'] = False
        results.append(r)
        print(f"Crosses: {r['crosses']}, Retests: {r['retests']}, Confirmations: {r['confirmations']}, Trades: {r['trades']}")
        print(f"Kill Ratio: {r['kill_ratio']:.2f}, Sharpe: {r['sharpe']:.2f}, DD: {r['max_dd']:.2%}")
        if r['trades'] == 0:
            print("*** TRADES DROPPED TO ZERO - ATR cap is the culprit ***")
            break
    
    # Summary table
    print("\n" + "="*80)
    print("ABLATION LADDER RESULTS TABLE")
    print("="*80)
    print(f"{'Rung':<10} {'Retest':<8} {'Vol_Mult':<10} {'ATR_Cap':<10} {'EMA':<6} {'Session':<8} {'Trades':<8} {'Sharpe':<8} {'DD':<8}")
    print("-"*80)
    for r in results:
        print(f"{r['rung']:<10} {r['retest']:<8} {r['vol_mult']:<10.2f} {r['atr_cap']:<10.1f} "
              f"{str(r['ema_filter']):<6} {str(r['session']):<8} {r['trades']:<8} "
              f"{r['sharpe']:<8.2f} {r['max_dd']:<8.2%}")
    
    print("\n" + "="*80)
    print("DETAILED METRICS")
    print("="*80)
    print(f"{'Rung':<10} {'Crosses':<10} {'Retests':<10} {'Confirms':<10} {'Trades':<8} {'Kill_Ratio':<12}")
    print("-"*80)
    for r in results:
        print(f"{r['rung']:<10} {r['crosses']:<10} {r['retests']:<10} {r['confirmations']:<10} "
              f"{r['trades']:<8} {r['kill_ratio']:<12.2f}")
    
    # Find choke point
    print("\n" + "="*80)
    print("CHOKE POINT ANALYSIS")
    print("="*80)
    for i in range(1, len(results)):
        if results[i]['trades'] == 0 and results[i-1]['trades'] > 0:
            print(f"TRADES DROPPED TO ZERO at: {results[i]['rung']}")
            print(f"Previous rung ({results[i-1]['rung']}) had {results[i-1]['trades']} trades")
            print(f"Filter added: {results[i]['rung']}")
            print("This is the PRIMARY CULPRIT")
            break
    else:
        print("No single filter killed all trades - combination effect")
    
    print("="*80)


if __name__ == "__main__":
    main()
