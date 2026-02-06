"""
Quick Test - Control Flip Strategy
Tests with smaller dataset to validate functionality
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capital_allocation_ai.vwap_control_flip_strategy import VWAPControlFlipStrategy, StrategyParams


def generate_test_data(symbol: str, bars: int = 500) -> pd.DataFrame:
    """Generate small test dataset."""
    np.random.seed(42)
    
    if symbol == 'GBPJPY':
        base_price = 185.0
    elif symbol == 'BTCUSD':
        base_price = 60000.0
    else:
        base_price = 2650.0
    
    # Simple price generation
    returns = np.random.randn(bars) * 0.001
    prices = base_price + np.cumsum(returns)
    
    opens = prices.copy()
    highs = opens + abs(np.random.randn(bars) * 0.002 * opens)
    lows = opens - abs(np.random.randn(bars) * 0.002 * opens)
    closes = opens + np.random.randn(bars) * 0.001 * opens
    
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))
    
    volumes = np.random.randint(3000, 8000, bars)
    
    timestamps = pd.date_range(start=datetime.now() - timedelta(days=2), periods=bars, freq='5min')
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })


def quick_test():
    """Quick functionality test."""
    print("="*60)
    print("QUICK CONTROL FLIP STRATEGY TEST")
    print("="*60)
    
    # Test parameters
    params = StrategyParams(
        band_k=2.0,
        vol_mult=1.2,
        atr_cap_mult=2.5,
        require_nth_retest=2,  # Lower for testing
        touch_tol_atr_frac=0.05,
        trail_pct=0.007,
        cross_lookback_bars=10,
        require_session_filter=False  # Disable for testing
    )
    
    strategy = VWAPControlFlipStrategy(params)
    
    # Generate test data
    print("\nGenerating test data...")
    data = generate_test_data('GBPJPY', bars=500)
    print(f"Generated {len(data)} bars")
    
    # Process bars
    print("\nProcessing bars...")
    signals_count = {'enter_long': 0, 'enter_short': 0, 'exit': 0}
    trades = 0
    
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
            
            signals = result['signals']
            
            if signals['enter_long']:
                signals_count['enter_long'] += 1
                if strategy.position is None:
                    strategy.enter_long(row['close'])
                    trades += 1
            
            if signals['enter_short']:
                signals_count['enter_short'] += 1
                if strategy.position is None:
                    strategy.enter_short(row['close'])
                    trades += 1
            
            if signals['exit'] and strategy.position is not None:
                signals_count['exit'] += 1
                strategy.exit_position()
        
        except Exception as e:
            print(f"Error at bar {idx}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Bars Processed:        {len(data)}")
    print(f"Entry Long Signals:    {signals_count['enter_long']}")
    print(f"Entry Short Signals:   {signals_count['enter_short']}")
    print(f"Exit Signals:          {signals_count['exit']}")
    print(f"Trades Executed:       {trades}")
    print(f"Final Position:        {strategy.position is not None}")
    
    if signals_count['enter_long'] > 0 or signals_count['enter_short'] > 0:
        print("\n✅ Strategy is generating signals!")
    else:
        print("\nWARNING: No signals generated - may need parameter adjustment")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    quick_test()
