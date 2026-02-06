"""
Backtest Capital Allocation AI on GBP/JPY - Last 3 Weeks
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add capital_allocation_ai to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capital_allocation_ai.system import CapitalAllocationAI
from capital_allocation_ai.training import TrainingEnvironment


def generate_gbpjpy_data_3weeks() -> pd.DataFrame:
    """
    Generate realistic GBP/JPY data for last 3 weeks.
    Uses 5-minute bars for faster processing.
    """
    print("Generating GBP/JPY data for last 3 weeks...")
    
    # 3 weeks = 21 trading days
    # 5-minute bars: 288 bars per day (24 hours * 12 bars/hour)
    trading_days = 21
    bars_per_day = 288
    total_bars = trading_days * bars_per_day
    
    np.random.seed(42)
    
    # GBP/JPY typically trades around 185-200
    base_price = 185.0
    
    # Generate realistic price movements
    # Add some trend and volatility clustering
    trend = np.linspace(0, 2, total_bars)  # 2 yen trend over 3 weeks
    
    # Volatility clustering (GARCH-like)
    volatility = np.ones(total_bars) * 0.0005
    for i in range(1, total_bars):
        volatility[i] = 0.0003 + 0.7 * volatility[i-1] + 0.2 * abs(np.random.randn() * 0.0003)
        volatility[i] = min(volatility[i], 0.002)
    
    # Generate returns
    returns = np.random.randn(total_bars) * volatility
    
    # Add mean reversion
    for i in range(1, total_bars):
        if abs(returns[i-1]) > 0.001:
            returns[i] -= 0.3 * returns[i-1]
    
    # Add trend
    returns += trend / total_bars
    
    # Generate prices
    prices = base_price + np.cumsum(returns)
    prices = np.clip(prices, 180, 195)  # Reasonable range
    
    # Generate OHLC
    opens = prices.copy()
    highs = opens + abs(np.random.randn(total_bars) * volatility * opens * 2)
    lows = opens - abs(np.random.randn(total_bars) * volatility * opens * 2)
    closes = opens + np.random.randn(total_bars) * volatility * opens
    
    # Ensure OHLC consistency
    for i in range(total_bars):
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
    
    # Generate volume
    base_volume = 5000
    volume_multiplier = 1 + volatility / volatility.mean() * 2
    volumes = (base_volume * volume_multiplier * (1 + abs(returns) * 10)).astype(int)
    volumes = np.clip(volumes, 1000, 50000)
    
    # Generate timestamps (5-minute bars, skip weekends)
    start_date = datetime.now() - timedelta(days=21)
    timestamps = []
    current_date = start_date
    bar_count = 0
    
    while bar_count < total_bars:
        # Skip weekends
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        
        timestamps.append(current_date)
        bar_count += 1
        
        # Move to next 5-minute bar
        if bar_count % bars_per_day == 0:
            current_date += timedelta(days=1)
        else:
            current_date += timedelta(minutes=5)
    
    # Calculate VWAP (session-based, resets daily)
    vwap_values = []
    current_day = None
    day_prices = []
    day_volumes = []
    
    for i, ts in enumerate(timestamps[:total_bars]):
        if current_day != ts.date():
            current_day = ts.date()
            day_prices = []
            day_volumes = []
        
        day_prices.append(closes[i])
        day_volumes.append(volumes[i])
        
        if sum(day_volumes) > 0:
            vwap = sum(p * v for p, v in zip(day_prices, day_volumes)) / sum(day_volumes)
        else:
            vwap = closes[i]
        
        vwap_values.append(vwap)
    
    # Create DataFrame
    data = pd.DataFrame({
        'timestamp': timestamps[:total_bars],
        'open': opens[:total_bars],
        'high': highs[:total_bars],
        'low': lows[:total_bars],
        'close': closes[:total_bars],
        'volume': volumes[:total_bars],
        'vwap': vwap_values[:total_bars]
    })
    
    print(f"Generated {len(data)} bars")
    print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    return data


def backtest_capital_allocation_ai(data: pd.DataFrame,
                                    initial_equity: float = 10000.0,
                                    max_risk: float = 0.5) -> dict:
    """
    Backtest Capital Allocation AI system.
    
    Returns:
        Dictionary with backtest results
    """
    print("\n" + "="*60)
    print("CAPITAL ALLOCATION AI BACKTEST - GBP/JPY (3 Weeks)")
    print("="*60)
    
    # Initialize AI system
    ai = CapitalAllocationAI(
        symbol="GBPJPY",
        initial_equity=initial_equity,
        max_risk=max_risk,
        broker_type="mock"  # Use mock execution for backtesting
    )
    
    # Track results
    equity_history = [initial_equity]
    drawdown_history = [0.0]
    action_history = []
    position_history = []
    
    print(f"\nInitial Equity: ${initial_equity:,.2f}")
    print(f"Max Risk per Trade: {max_risk*100:.0f}%")
    print(f"Data Bars: {len(data)}")
    print("-"*60)
    
    # Run backtest
    for idx, row in data.iterrows():
        # Update AI with market data
        result = ai.update_market_data(
            current_price=row['close'],
            high=row['high'],
            low=row['low'],
            volume=row['volume'],
            vwap=row.get('vwap', row['close']),
            timestamp=row.get('timestamp')
        )
        
        # Check if halted
        if result.get('action') == 'halted':
            print(f"\nWARNING: TRADING HALTED: {result.get('reason')}")
            break
        
        # Record history
        equity_history.append(result['equity'])
        drawdown_history.append(result['drawdown'])
        action_history.append(result['action'])
        position_history.append(result['position_size'])
        
        # Print progress every 500 bars
        if (idx + 1) % 500 == 0:
            print(f"Bar {idx+1}/{len(data)} | "
                  f"Equity: ${result['equity']:,.2f} | "
                  f"DD: {result['drawdown']*100:.1f}% | "
                  f"Action: {result['action']}")
    
    # Close any open positions
    if ai.position_direction != 0:
        final_price = data.iloc[-1]['close']
        ai._close_position(final_price)
        equity_history.append(ai.account_equity)
    
    # Get statistics
    stats = ai.get_statistics()
    stats['equity_history'] = equity_history
    stats['drawdown_history'] = drawdown_history
    stats['action_history'] = action_history
    stats['position_history'] = position_history
    
    return stats


def print_backtest_results(stats: dict, initial_equity: float):
    """Print comprehensive backtest results."""
    print("\n" + "="*60)
    print("BACKTEST RESULTS - GBP/JPY (3 Weeks)")
    print("="*60)
    
    print(f"\nPERFORMANCE METRICS")
    print(f"{'-'*60}")
    print(f"Initial Equity:        ${initial_equity:,.2f}")
    print(f"Final Equity:           ${stats['final_equity']:,.2f}")
    print(f"Total Return:           {stats['total_return']*100:+.2f}%")
    print(f"Max Drawdown:           {stats['max_drawdown']*100:.2f}%")
    
    # Calculate additional metrics
    equity_array = np.array(stats['equity_history'])
    if len(equity_array) > 1:
        returns = np.diff(equity_array) / equity_array[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 288)  # Annualized
        volatility = np.std(returns) * np.sqrt(252 * 288) * 100
        print(f"Sharpe Ratio:           {sharpe:.2f}")
        print(f"Equity Volatility:      {volatility:.2f}%")
        
        # Best/Worst day
        if len(equity_array) >= 288:  # At least 1 day
            daily_returns = []
            for i in range(0, len(equity_array) - 288, 288):
                day_return = (equity_array[i+288] - equity_array[i]) / equity_array[i]
                daily_returns.append(day_return)
            
            if daily_returns:
                best_day = max(daily_returns) * 100
                worst_day = min(daily_returns) * 100
                print(f"Best Day:              {best_day:+.2f}%")
                print(f"Worst Day:             {worst_day:.2f}%")
    
    print(f"\nTRADING STATISTICS")
    print(f"{'-'*60}")
    print(f"Total Trades:           {stats['total_trades']}")
    print(f"Winning Trades:         {stats['winning_trades']}")
    print(f"Losing Trades:          {stats['losing_trades']}")
    print(f"Win Rate:               {stats['win_rate']*100:.2f}%")
    
    if stats['total_trades'] > 0:
        trades_df = pd.DataFrame(stats.get('trade_history', []))
        if len(trades_df) > 0:
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if (trades_df['pnl'] > 0).any() else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (trades_df['pnl'] < 0).any() else 0
            print(f"Average Win:            ${avg_win:.2f}")
            print(f"Average Loss:           ${avg_loss:.2f}")
            if avg_loss != 0:
                profit_factor = abs(avg_win * stats['winning_trades'] / (avg_loss * stats['losing_trades'])) if stats['losing_trades'] > 0 else float('inf')
                print(f"Profit Factor:          {profit_factor:.2f}")
    
    # Action distribution
    if stats.get('action_history'):
        actions = pd.Series(stats['action_history'])
        action_counts = actions.value_counts()
        print(f"\nACTION DISTRIBUTION")
        print(f"{'-'*60}")
        for action, count in action_counts.items():
            pct = count / len(actions) * 100
            print(f"{action:15s}: {count:5d} ({pct:5.1f}%)")
    
    print(f"\nTARGET COMPARISON")
    print(f"{'-'*60}")
    win_check = "PASS" if stats['win_rate']*100 >= 55 else "FAIL"
    dd_check = "PASS" if stats['max_drawdown']*100 <= 20 else "FAIL"
    return_check = "PASS" if stats['total_return']*100 > 0 else "FAIL"
    
    print(f"Win Rate:     {stats['win_rate']*100:.1f}% (Target: >=55%) [{win_check}]")
    print(f"Max Drawdown: {stats['max_drawdown']*100:.1f}% (Target: <=20%) [{dd_check}]")
    print(f"Total Return: {stats['total_return']*100:.2f}% (Target: >0%) [{return_check}]")
    
    # Equity curve summary
    if len(equity_array) > 0:
        peak_equity = np.max(equity_array)
        final_equity = equity_array[-1]
        print(f"\nEQUITY CURVE")
        print(f"{'-'*60}")
        print(f"Peak Equity:            ${peak_equity:,.2f}")
        print(f"Final Equity:           ${final_equity:,.2f}")
        print(f"From Peak:              {(final_equity - peak_equity) / peak_equity * 100:.2f}%")


def plot_results(stats: dict, initial_equity: float, save_path: str = 'backtest_results.png'):
    """Plot equity curve and drawdown."""
    try:
        import matplotlib.pyplot as plt
        
        equity_history = stats['equity_history']
        drawdown_history = stats['drawdown_history']
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Equity curve
        axes[0].plot(equity_history, linewidth=2, color='blue', label='Equity')
        axes[0].axhline(y=initial_equity, color='gray', linestyle='--', label='Initial Equity')
        axes[0].set_title('Equity Curve - GBP/JPY (3 Weeks)', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Equity ($)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Drawdown
        drawdown_pct = [d * 100 for d in drawdown_history]
        axes[1].fill_between(range(len(drawdown_pct)), drawdown_pct, 0, 
                            color='red', alpha=0.3, label='Drawdown')
        axes[1].plot(drawdown_pct, linewidth=1, color='red')
        axes[1].axhline(y=-20, color='orange', linestyle='--', label='20% Drawdown Limit')
        axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Bar', fontsize=12)
        axes[1].set_ylabel('Drawdown (%)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nCharts saved to: {save_path}")
        plt.close()
        
    except ImportError:
        print("\nmatplotlib not available, skipping charts")
    except Exception as e:
        print(f"\nError plotting: {e}")


def main():
    """Main backtest function."""
    print("="*60)
    print("CAPITAL ALLOCATION AI BACKTEST")
    print("GBP/JPY - Last 3 Weeks")
    print("="*60)
    
    # Parameters
    initial_equity = 10000.0
    max_risk = 0.5  # 50% max position size
    
    # Generate data
    data = generate_gbpjpy_data_3weeks()
    
    # Run backtest
    stats = backtest_capital_allocation_ai(
        data=data,
        initial_equity=initial_equity,
        max_risk=max_risk
    )
    
    # Print results
    print_backtest_results(stats, initial_equity)
    
    # Plot results
    plot_results(stats, initial_equity)
    
    print("\n" + "="*60)
    print("BACKTEST COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
