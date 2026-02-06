"""
Test Optimized VWAP Strategy
Quick test to verify optimized parameters work correctly
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capital_allocation_ai.vwap_pro_strategy import VWAPProStrategy


def generate_test_data(n_bars=1000):
    """Generate test data."""
    np.random.seed(42)
    base_price = 185.0
    
    # Create price with mean reversion
    prices = [base_price]
    for i in range(1, n_bars):
        deviation = (prices[-1] - base_price) / base_price
        mean_reversion = -0.2 * deviation
        random_walk = np.random.randn() * 0.0005
        new_price = prices[-1] * (1 + mean_reversion + random_walk)
        prices.append(new_price)
    
    prices = np.array(prices)
    
    opens = prices.copy()
    highs = opens + abs(np.random.randn(n_bars) * 0.001 * opens)
    lows = opens - abs(np.random.randn(n_bars) * 0.001 * opens)
    closes = opens + np.random.randn(n_bars) * 0.0005 * opens
    
    for i in range(n_bars):
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
    
    volumes = np.random.randint(3000, 8000, n_bars)
    
    timestamps = [datetime.now() - timedelta(minutes=5*(n_bars-i)) for i in range(n_bars)]
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    })


def test_strategy(params, data):
    """Test strategy with given parameters."""
    strategy = VWAPProStrategy(**params)
    
    trades = 0
    signals_count = {'entry': 0, 'exit': 0, 'reposition': 0}
    
    for _, row in data.iterrows():
        result = strategy.update(
            current_price=row['close'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            timestamp=row.get('timestamp')
        )
        
        if result['signals']['entry_long']:
            signals_count['entry'] += 1
            if strategy.position == 0:
                strategy.enter_long(row['close'])
                trades += 1
        
        if result['signals']['reposition_long']:
            signals_count['reposition'] += 1
        
        if result['signals']['exit'] and strategy.position != 0:
            signals_count['exit'] += 1
            strategy.exit_position()
    
    return {
        'total_trades': trades,
        'signals': signals_count,
        'filters_working': signals_count['entry'] < len(data) / 10  # Should be much less
    }


def main():
    """Test optimized strategy."""
    print("="*60)
    print("TESTING OPTIMIZED VWAP STRATEGY")
    print("="*60)
    
    data = generate_test_data(2000)
    print(f"Generated {len(data)} bars of test data")
    
    # Test 1: Default parameters (loose)
    print("\nTest 1: Default Parameters (Loose Filters)")
    print("-"*60)
    default_params = {
        'sigma_multiplier': 2.0,
        'min_volume_multiplier': 1.0,  # Loose
        'min_touches_before_entry': 1  # Loose
    }
    default_result = test_strategy(default_params, data)
    print(f"Total Trades: {default_result['total_trades']}")
    print(f"Entry Signals: {default_result['signals']['entry']}")
    print(f"Exit Signals: {default_result['signals']['exit']}")
    
    # Test 2: Optimized parameters (tight)
    print("\nTest 2: Optimized Parameters (Tight Filters)")
    print("-"*60)
    optimized_params = {
        'sigma_multiplier': 2.1,
        'min_volume_multiplier': 1.4,      # Tighter
        'volume_spike_multiplier': 1.6,    # Requires spike
        'min_touches_before_entry': 3,     # 3-touch rule
        'volatility_threshold': 2.8        # Stricter
    }
    optimized_result = test_strategy(optimized_params, data)
    print(f"Total Trades: {optimized_result['total_trades']}")
    print(f"Entry Signals: {optimized_result['signals']['entry']}")
    print(f"Exit Signals: {optimized_result['signals']['exit']}")
    
    # Comparison
    print("\n" + "="*60)
    print("COMPARISON")
    print("="*60)
    reduction = ((default_result['total_trades'] - optimized_result['total_trades']) / 
                 default_result['total_trades'] * 100) if default_result['total_trades'] > 0 else 0
    print(f"Trade Reduction: {reduction:.1f}%")
    print(f"Default: {default_result['total_trades']} trades")
    print(f"Optimized: {optimized_result['total_trades']} trades")
    
    if optimized_result['total_trades'] < default_result['total_trades']:
        print("\nSUCCESS: Optimized parameters reduce trade frequency!")
    else:
        print("\nNote: May need further tuning based on your data")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
