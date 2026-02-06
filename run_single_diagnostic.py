"""
Single Diagnostic Test - Get cross_events, retest_events, confirm_events, trades
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


def main():
    """Run single diagnostic test."""
    print("="*80)
    print("SINGLE DIAGNOSTIC TEST")
    print("="*80)
    
    symbol = 'GBPJPY'
    days = 7
    
    print(f"\nSymbol: {symbol}, Days: {days}")
    print("Generating data...")
    data = generate_market_data(symbol, days=days)
    print(f"{len(data)} bars generated")
    
    # Very loose parameters - core flip only
    params = StrategyParams(
        band_k=2.0,
        vol_mult=1.0,  # No volume filter
        atr_cap_mult=999.0,  # No ATR cap
        require_nth_retest=1,  # 1st retest only
        require_session_filter=False,
        cross_lookback_bars=20,
        touch_tol_atr_frac=0.20  # 20% of ATR tolerance
    )
    
    print("\nParameters:")
    print(f"  Volume Mult: {params.vol_mult} (disabled)")
    print(f"  ATR Cap: {params.atr_cap_mult} (disabled)")
    print(f"  Retest Count: {params.require_nth_retest}")
    print(f"  Touch Tolerance: {params.touch_tol_atr_frac * 100:.0f}% of ATR")
    
    # First, compute indicators on full dataframe for truth prints
    print("\nComputing indicators for truth prints...")
    import pandas as pd
    from capital_allocation_ai.vwap_control_flip_strategy import vwap_and_sigma, ema, atr
    
    df = data.copy()
    df['vwap'], df['sigma'] = vwap_and_sigma(df, reset_hour_utc=params.reset_hour_utc)
    df['atr20'] = atr(df, params.atr_len)
    
    # TRUTH PRINTS - Basic Data Sanity
    print("\n" + "="*80)
    print("TRUTH PRINTS - DATA SANITY")
    print("="*80)
    print(f"BARS: {len(df)}")
    print(f"TS: {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}")
    print(f"CLOSE min/max: {df['close'].min():.5f} / {df['close'].max():.5f}")
    print(f"VOL min/max: {df['volume'].min()} / {df['volume'].max()}")
    
    # TRUTH PRINTS - VWAP Sanity
    vwap = df['vwap']
    close = df['close']
    
    print("\n" + "="*80)
    print("TRUTH PRINTS - VWAP SANITY")
    print("="*80)
    print(f"VWAP NaN %: {float(vwap.isna().mean()) * 100:.2f}%")
    print(f"VWAP min/max: {vwap.min():.5f} / {vwap.max():.5f}")
    print(f"Mean |close-vwap|: {float((close - vwap).abs().mean()):.5f}")
    print(f"Median |close-vwap|: {float((close - vwap).abs().median()):.5f}")
    
    # TRUTH PRINTS - Cross Sanity (raw, independent of strategy)
    print("\n" + "="*80)
    print("TRUTH PRINTS - CROSS SANITY (RAW)")
    print("="*80)
    cross_up = ((close > vwap) & (close.shift(1) <= vwap.shift(1))).sum()
    cross_dn = ((close < vwap) & (close.shift(1) >= vwap.shift(1))).sum()
    print(f"RAW crosses up: {int(cross_up)}")
    print(f"RAW crosses down: {int(cross_dn)}")
    print(f"RAW crosses total: {int(cross_up + cross_dn)}")
    print("="*80)
    
    # Now run strategy backtest
    strategy = VWAPControlFlipStrategy(params)
    
    equity = 10000.0
    trades = []
    
    # Event counters
    cross_events = 0
    retest_events = 0
    confirm_events = 0
    entered_trades = 0
    
    prev_close = None
    prev_vwap = None
    
    print("\nProcessing bars with strategy...")
    start_time = time.time()
    
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
            
            # Detect cross events
            if prev_close is not None and prev_vwap is not None and current_vwap > 0:
                if row['close'] > current_vwap and prev_close <= prev_vwap:
                    cross_events += 1
                elif row['close'] < current_vwap and prev_close >= prev_vwap:
                    cross_events += 1
                
                # Detect retest events
                tol = params.touch_tol_atr_frac * current_atr if current_atr > 0 else 0.001 * row['close']
                
                if row['low'] <= current_vwap + tol and row['close'] > current_vwap:
                    retest_events += 1
                if row['high'] >= current_vwap - tol and row['close'] < current_vwap:
                    retest_events += 1
            
            prev_close = row['close']
            prev_vwap = current_vwap
            
            # Track confirmations
            if result['signals']['enter_long'] or result['signals']['enter_short']:
                confirm_events += 1
            
            if result['signals']['enter_long'] and strategy.position is None:
                strategy.enter_long(row['close'])
                entered_trades += 1
                trades.append({'pnl': 0, 'entry': row['close'], 'side': 'long'})
            elif result['signals']['enter_short'] and strategy.position is None:
                strategy.enter_short(row['close'])
                entered_trades += 1
                trades.append({'pnl': 0, 'entry': row['close'], 'side': 'short'})
            elif result['signals']['exit'] and strategy.position is not None:
                pos = strategy.position
                size = equity / pos.entry_price * 0.1
                pnl = (row['close'] - pos.entry_price) * size if pos.side == "long" else (pos.entry_price - row['close']) * size
                equity += pnl
                if trades:
                    trades[-1]['pnl'] = pnl
                strategy.exit_position()
        except:
            continue
    
    elapsed = time.time() - start_time
    
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    print(f"cross_events: {cross_events}")
    print(f"retest_events: {retest_events}")
    print(f"confirm_events: {confirm_events}")
    print(f"trades: {len(trades)}")
    print(f"Time: {elapsed:.1f}s")
    print("="*80)
    
    if len(trades) == 0:
        print("\nDIAGNOSIS:")
        if cross_events == 0:
            print("NO VWAP CROSSES - Check VWAP calculation or data patterns")
        elif retest_events == 0:
            print("NO RETESTS - Check retest logic or tolerance (try increasing touch_tol_atr_frac)")
        elif confirm_events == 0:
            print("NO CONFIRMATIONS - Filters are blocking signals despite retests")
        else:
            print(f"Has {cross_events} crosses, {retest_events} retests, {confirm_events} confirmations but 0 trades")
            print("Check entry conditions in _generate_signal")


if __name__ == "__main__":
    main()
