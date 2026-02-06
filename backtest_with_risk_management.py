"""
Example Backtest with Risk Management Integration
Shows how to use the new risk management modules in a backtest
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capital_allocation_ai.vwap_control_flip_strategy import StrategyParams, VWAPControlFlipStrategy
from capital_allocation_ai.risk import InstrumentSpec, PropRules, monte_carlo_paths, worst_case_sequence
from load_real_market_data import load_csv_ohlcv, prepare_data_for_backtest


def backtest_with_risk_management(
    data: pd.DataFrame,
    symbol: str,
    initial_equity: float = 10000.0,
    use_risk_management: bool = True
) -> dict:
    """
    Run backtest with integrated risk management.
    
    Args:
        data: OHLCV DataFrame with timestamp, open, high, low, close, volume
        symbol: Trading symbol (GBPJPY, BTCUSD, XAUUSD)
        initial_equity: Starting equity
        use_risk_management: Enable risk management modules
    
    Returns:
        Dictionary with metrics and trade list
    """
    # Create strategy params with risk management
    params = StrategyParams(
        band_k=2.0,
        band_k_list=(1.0, 2.0, 3.0),
        require_nth_retest=3,
        retest_min=2,
        retest_max=4,
        volume_filter_type="percentile",
        vol_percentile_L=50,
        vol_percentile_p=60.0,
        atr_cap_mult=3.0,
        require_session_filter=False,
        trail_pct=0.007,
        start_trail_profit_pct=0.007,
        # Risk management parameters
        risk_per_trade_frac=0.003 if use_risk_management else 0.01,  # 0.30% with RM, 1% without
        atr_mult_stop=2.0,
        enable_drawdown_governor=use_risk_management,
        enable_enhanced_exits=use_risk_management,
        time_stop_bars=20 if use_risk_management else None,
        break_even_r=1.0,
        partial_tp_r=1.0,
        partial_tp_pct=0.4,
        loss_duration_cap_bars=30 if use_risk_management else None
    )
    
    # Create instrument spec (adjust for your broker)
    if symbol == "XAUUSD":
        inst_spec = InstrumentSpec(dollar_per_price_unit=1.0)  # $1 per $1 move per oz
    elif symbol == "GBPJPY":
        inst_spec = InstrumentSpec(dollar_per_price_unit=0.01)  # $0.01 per pip per lot
    elif symbol == "BTCUSD":
        inst_spec = InstrumentSpec(dollar_per_price_unit=1.0)  # $1 per $1 move per contract
    else:
        inst_spec = InstrumentSpec(dollar_per_price_unit=1.0)  # Default
    
    # Create strategy
    strategy = VWAPControlFlipStrategy(
        params=params,
        initial_equity=initial_equity,
        instrument_spec=inst_spec
    )
    
    # Backtest loop
    equity = float(initial_equity)
    trades = []
    equity_curve = [equity]
    peak_equity = equity
    
    for idx, row in data.iterrows():
        # Update strategy
        result = strategy.update(
            current_price=row['close'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            timestamp=row.get('timestamp', datetime.now())
        )
        
        # Update equity tracking (for governor)
        strategy.equity = equity
        
        # Entry signals
        if result['signals']['enter_long'] and strategy.position is None:
            atr_value = result['indicators'].get('atr20', row['close'] * 0.01)
            strategy.enter_long(price=row['close'], atr_value=atr_value)
            
        elif result['signals']['enter_short'] and strategy.position is None:
            atr_value = result['indicators'].get('atr20', row['close'] * 0.01)
            strategy.enter_short(price=row['close'], atr_value=atr_value)
        
        # Exit signals
        elif result['signals']['exit'] and strategy.position is not None:
            pos = strategy.position
            
            # Calculate PnL
            if pos.side == "long":
                pnl = (row['close'] - pos.entry_price) * pos.qty
            else:
                pnl = (pos.entry_price - row['close']) * pos.qty
            
            # Update equity
            equity += pnl
            peak_equity = max(peak_equity, equity)
            
            # Record trade
            trades.append({
                'entry': pos.entry_price,
                'exit': row['close'],
                'side': pos.side,
                'qty': pos.qty,
                'pnl': pnl,
                'entry_time': row.get('timestamp'),
                'exit_time': row.get('timestamp'),
                'bars_held': strategy.current_bar_index - pos.entry_bar
            })
            
            strategy.exit_position()
        
        # Track equity curve
        equity_curve.append(equity)
    
    # Calculate metrics
    if len(trades) == 0:
        return {
            'trades': 0,
            'equity': equity,
            'return': 0.0,
            'max_drawdown': 0.0,
            'sharpe': 0.0,
            'win_rate': 0.0,
            'trade_list': []
        }
    
    trades_df = pd.DataFrame(trades)
    trade_pnls = trades_df['pnl'].tolist()
    returns = np.array(trade_pnls) / initial_equity
    
    # Sharpe ratio
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Max drawdown
    equity_array = np.array(equity_curve)
    peak_array = np.maximum.accumulate(equity_array)
    drawdowns = (peak_array - equity_array) / peak_array
    max_dd = float(np.max(drawdowns))
    
    # Win rate
    win_rate = (trades_df['pnl'] > 0).sum() / len(trades)
    
    # Stress test (if enabled and we have trades)
    breach_prob = 0.0
    if use_risk_management and len(trade_pnls) >= 10:
        try:
            days = (data['timestamp'].iloc[-1] - data['timestamp'].iloc[0]).days
            trades_per_day = max(1, len(trade_pnls) // max(days, 1))
            
            stress_result = monte_carlo_paths(
                start_equity=initial_equity,
                trade_pnls=trade_pnls,
                trades_per_day=trades_per_day,
                days=max(days, 1),
                rules=PropRules(),
                n_paths=1000,
                seed=42
            )
            breach_prob = stress_result.breach_prob
        except Exception as e:
            print(f"Stress test failed: {e}")
    
    return {
        'trades': len(trades),
        'equity': equity,
        'return': (equity - initial_equity) / initial_equity,
        'max_drawdown': max_dd,
        'sharpe': float(sharpe),
        'win_rate': float(win_rate),
        'breach_prob': breach_prob,
        'trade_list': trades,
        'equity_curve': equity_curve
    }


def main():
    """Example usage."""
    print("="*80)
    print("BACKTEST WITH RISK MANAGEMENT")
    print("="*80)
    
    # Try to load real data
    symbol = 'GBPJPY'
    data = None
    
    # Check for CSV
    csv_paths = [
        f'{symbol.lower()}_5m.csv',
        f'data/{symbol.lower()}_5m.csv'
    ]
    
    for path in csv_paths:
        if os.path.exists(path):
            print(f"\nLoading data from: {path}")
            try:
                data = load_csv_ohlcv(path)
                data = prepare_data_for_backtest(data, symbol)
                break
            except Exception as e:
                print(f"Failed to load {path}: {e}")
                continue
    
    if data is None:
        print("\n[ERROR] No data found. Please provide CSV file.")
        print("Expected: gbpjpy_5m.csv or data/gbpjpy_5m.csv")
        return
    
    print(f"\nLoaded {len(data)} bars")
    print(f"Range: {data['timestamp'].iloc[0]} -> {data['timestamp'].iloc[-1]}")
    
    # Run backtest WITH risk management
    print("\n" + "="*80)
    print("BACKTEST WITH RISK MANAGEMENT")
    print("="*80)
    results_rm = backtest_with_risk_management(data, symbol, use_risk_management=True)
    
    print(f"\nResults (WITH Risk Management):")
    print(f"  Trades: {results_rm['trades']}")
    print(f"  Return: {results_rm['return']:.2%}")
    print(f"  Max DD: {results_rm['max_drawdown']:.2%}")
    print(f"  Sharpe: {results_rm['sharpe']:.2f}")
    print(f"  Win Rate: {results_rm['win_rate']:.2%}")
    print(f"  Breach Prob: {results_rm['breach_prob']:.2%}")
    
    # Run backtest WITHOUT risk management (for comparison)
    print("\n" + "="*80)
    print("BACKTEST WITHOUT RISK MANAGEMENT (Comparison)")
    print("="*80)
    results_no_rm = backtest_with_risk_management(data, symbol, use_risk_management=False)
    
    print(f"\nResults (WITHOUT Risk Management):")
    print(f"  Trades: {results_no_rm['trades']}")
    print(f"  Return: {results_no_rm['return']:.2%}")
    print(f"  Max DD: {results_no_rm['max_drawdown']:.2%}")
    print(f"  Sharpe: {results_no_rm['sharpe']:.2f}")
    print(f"  Win Rate: {results_no_rm['win_rate']:.2%}")
    
    # Comparison
    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print(f"{'Metric':<20} {'With RM':<15} {'Without RM':<15} {'Change':<15}")
    print("-"*65)
    print(f"{'Return':<20} {results_rm['return']:>14.2%} {results_no_rm['return']:>14.2%} "
          f"{(results_rm['return'] - results_no_rm['return']):>14.2%}")
    print(f"{'Max DD':<20} {results_rm['max_drawdown']:>14.2%} {results_no_rm['max_drawdown']:>14.2%} "
          f"{(results_rm['max_drawdown'] - results_no_rm['max_drawdown']):>14.2%}")
    print(f"{'Sharpe':<20} {results_rm['sharpe']:>14.2f} {results_no_rm['sharpe']:>14.2f} "
          f"{(results_rm['sharpe'] - results_no_rm['sharpe']):>14.2f}")
    print(f"{'Trades':<20} {results_rm['trades']:>14} {results_no_rm['trades']:>14} "
          f"{(results_rm['trades'] - results_no_rm['trades']):>14}")
    
    print("\n" + "="*80)
    print("KEY TAKEAWAY")
    print("="*80)
    print("Risk management should:")
    print("  ✅ Reduce max drawdown significantly")
    print("  ✅ Maintain or improve Sharpe ratio")
    print("  ✅ Keep breach probability < 5%")
    print("  ⚠️  May reduce return (but it's REAL, not leveraged)")
    print("="*80)


if __name__ == "__main__":
    main()
