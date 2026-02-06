"""
Backtest RL Trading System for Past 6 Months
Generates realistic market data and runs comprehensive backtest.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import sys
import os

# Add rl_trading_system to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_trading_system.ppo_agent import PPOAgent
from rl_trading_system.trading_environment import TradingEnvironment
from rl_trading_system.risk_governor import RiskGovernor


def generate_realistic_gbpjpy_data(start_date: str, n_months: int = 6) -> pd.DataFrame:
    """
    Generate realistic GBP/JPY 1-minute data for backtesting.
    Simulates realistic price movements with trends, volatility clusters, and volume patterns.
    """
    print(f"Generating {n_months} months of GBP/JPY 1-minute data...")
    
    # Calculate number of bars (6 months of 1-minute data)
    # For faster backtesting, we'll use 5-minute bars instead of 1-minute
    # Trading days: ~252 per year, ~21 per month
    # Trading hours: 24 hours/day for forex
    # Bars per day: 288 (24 hours * 12 five-minute bars)
    trading_days_per_month = 21
    bars_per_day = 288  # 5-minute bars
    total_bars = trading_days_per_month * n_months * bars_per_day
    
    # Start date
    start = pd.to_datetime(start_date)
    
    # Generate timestamps (skip weekends)
    timestamps = []
    current_date = start
    bar_count = 0
    
    while bar_count < total_bars:
        # Skip weekends (Saturday=5, Sunday=6)
        if current_date.weekday() < 5:
            # Generate bars for this day
            for hour in range(24):
                for minute in range(60):
                    if bar_count >= total_bars:
                        break
                    # 5-minute bars
                    if minute % 5 == 0:
                        timestamps.append(current_date + timedelta(hours=hour, minutes=minute))
                        bar_count += 1
                if bar_count >= total_bars:
                    break
        current_date += timedelta(days=1)
        if bar_count >= total_bars:
            break
    
    n_bars = len(timestamps)
    
    # Generate realistic price data
    np.random.seed(42)  # For reproducibility
    
    # Base price (GBP/JPY typically trades around 150-200)
    base_price = 185.0
    
    # Create trend component (slow drift)
    trend = np.linspace(0, 5, n_bars)  # 5 yen trend over period
    
    # Create volatility clusters (GARCH-like behavior)
    volatility = np.ones(n_bars) * 0.0005  # Base volatility
    for i in range(1, n_bars):
        # Volatility clustering
        volatility[i] = 0.0003 + 0.7 * volatility[i-1] + 0.2 * abs(np.random.randn() * 0.0003)
        volatility[i] = min(volatility[i], 0.002)  # Cap volatility
    
    # Generate returns with volatility clustering
    returns = np.random.randn(n_bars) * volatility
    
    # Add some mean reversion
    for i in range(1, n_bars):
        if abs(returns[i-1]) > 0.001:
            returns[i] -= 0.3 * returns[i-1]  # Mean reversion
    
    # Add trend
    returns += trend / n_bars
    
    # Generate prices
    prices = base_price + np.cumsum(returns)
    
    # Ensure prices stay in reasonable range
    prices = np.clip(prices, 150, 220)
    
    # Generate OHLC
    opens = prices.copy()
    highs = opens + abs(np.random.randn(n_bars) * volatility * opens * 2)
    lows = opens - abs(np.random.randn(n_bars) * volatility * opens * 2)
    closes = opens + np.random.randn(n_bars) * volatility * opens
    
    # Ensure high >= max(open, close) and low <= min(open, close)
    for i in range(n_bars):
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
    
    # Generate volume (higher during volatile periods)
    base_volume = 5000
    volume_multiplier = 1 + volatility / volatility.mean() * 2
    volumes = (base_volume * volume_multiplier * (1 + abs(returns) * 10)).astype(int)
    volumes = np.clip(volumes, 1000, 50000)
    
    # Calculate VWAP (session-based, resets daily)
    vwap_values = []
    current_day = None
    day_prices = []
    day_volumes = []
    
    for i, ts in enumerate(timestamps):
        if current_day != ts.date():
            # New day, reset VWAP
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
        'timestamp': timestamps[:n_bars],
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes,
        'vwap': vwap_values
    })
    
    print(f"Generated {len(data)} bars")
    print(f"Date range: {data['timestamp'].min()} to {data['timestamp'].max()}")
    print(f"Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
    
    return data


def split_data(data: pd.DataFrame, train_ratio: float = 0.7) -> tuple:
    """Split data into training and testing sets."""
    split_idx = int(len(data) * train_ratio)
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    return train_data, test_data


def train_agent(env: TradingEnvironment, episodes: int = 500) -> PPOAgent:
    """Train PPO agent."""
    print(f"\n{'='*60}")
    print("TRAINING PHASE")
    print(f"{'='*60}")
    
    agent = PPOAgent(
        state_size=env.state_size,
        lr=3e-4,
        gamma=0.99
    )
    
    episode_rewards = []
    episode_returns = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            direction, pos_size, value, log_prob = agent.act(state)
            action = (direction, pos_size)
            
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Train periodically
            if steps % 200 == 0 and len(agent.memory['states']) > 0:
                agent.train(epochs=4)
        
        # Get statistics
        stats = env.get_statistics()
        episode_rewards.append(total_reward)
        episode_returns.append(stats['total_return'])
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_return = np.mean(episode_returns[-50:])
            print(f"Episode {episode+1}/{episodes} | "
                  f"Avg Reward: {avg_reward:.4f} | "
                  f"Avg Return: {avg_return*100:.2f}% | "
                  f"Trades: {stats['total_trades']} | "
                  f"Win Rate: {stats['win_rate']*100:.1f}%")
    
    return agent


def run_backtest(env: TradingEnvironment, agent: PPOAgent, risk_gov: RiskGovernor) -> dict:
    """Run backtest with trained agent."""
    print(f"\n{'='*60}")
    print("BACKTEST PHASE")
    print(f"{'='*60}")
    
    agent.model.eval()
    risk_gov.initialize(env.initial_equity)
    
    state = env.reset()
    done = False
    step = 0
    
    equity_history = []
    drawdown_history = []
    position_history = []
    
    while not done:
        # Check risk limits
        current_equity = env.account_equity + env.unrealized_pnl
        risk_gov.update_equity(current_equity)
        allowed, reason = risk_gov.check_risk_limits(current_equity)
        
        if not allowed:
            print(f"\nTrading halted at step {step}: {reason}")
            if env.position_size != 0:
                current_price = env.data.iloc[env.current_step]['close']
                env._close_position(current_price)
            break
        
        # Agent selects action
        direction, pos_size, value, _ = agent.act(state, deterministic=False)
        
        # Adjust position size
        adjusted_size = risk_gov.adjust_position_size(pos_size, current_equity)
        action = (direction, adjusted_size)
        
        # Execute
        next_state, reward, done, info = env.step(action)
        
        # Record history
        equity_history.append(info['equity'])
        drawdown_history.append(info['drawdown'])
        position_history.append(info['position_size'])
        
        state = next_state
        step += 1
        
        if step % 1000 == 0:
            print(f"Step {step} | Equity: ${info['equity']:.2f} | "
                  f"DD: {info['drawdown']*100:.1f}% | "
                  f"Position: {info['position_size']:.2f}")
    
    # Get final statistics
    stats = env.get_statistics()
    stats['equity_history'] = equity_history
    stats['drawdown_history'] = drawdown_history
    stats['position_history'] = position_history
    
    return stats


def print_results(stats: dict, initial_equity: float):
    """Print comprehensive backtest results."""
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS - 6 MONTHS")
    print(f"{'='*60}")
    
    print(f"\n📊 PERFORMANCE METRICS")
    print(f"{'-'*60}")
    print(f"Initial Equity:        ${initial_equity:,.2f}")
    print(f"Final Equity:           ${stats['final_equity']:,.2f}")
    print(f"Total Return:           {stats['total_return']*100:+.2f}%")
    print(f"Max Drawdown:           {stats['max_drawdown']*100:.2f}%")
    print(f"Sharpe Ratio:           {stats['sharpe_ratio']:.2f}")
    
    print(f"\n📈 TRADING STATISTICS")
    print(f"{'-'*60}")
    print(f"Total Trades:           {stats['total_trades']}")
    print(f"Winning Trades:         {stats['winning_trades']}")
    print(f"Losing Trades:          {stats['losing_trades']}")
    print(f"Win Rate:               {stats['win_rate']*100:.2f}%")
    print(f"Average Trade PnL:      ${stats['avg_trade_pnl']:.2f}")
    
    # Calculate additional metrics
    if len(stats['equity_history']) > 0:
        equity_array = np.array(stats['equity_history'])
        returns = np.diff(equity_array) / equity_array[:-1]
        
        # Calculate metrics
        positive_trades = stats['winning_trades']
        negative_trades = stats['losing_trades']
        
        if positive_trades > 0 and negative_trades > 0:
            # Profit factor
            # This would require individual trade PnL, simplified here
            pass
        
        # Volatility
        volatility = np.std(returns) * np.sqrt(252 * 1440) * 100  # Annualized %
        print(f"Equity Volatility:      {volatility:.2f}%")
        
        # Best/Worst day
        daily_returns = []
        if len(equity_array) >= 1440:  # At least 1 day of data
            for i in range(0, len(equity_array) - 1440, 1440):
                day_return = (equity_array[i+1440] - equity_array[i]) / equity_array[i]
                daily_returns.append(day_return)
            
            if daily_returns:
                best_day = max(daily_returns) * 100
                worst_day = min(daily_returns) * 100
                print(f"Best Day:              {best_day:+.2f}%")
                print(f"Worst Day:             {worst_day:.2f}%")
    
    print(f"\n🎯 TARGET COMPARISON")
    print(f"{'-'*60}")
    win_rate_target = 55.0
    drawdown_target = 20.0
    sharpe_target = 1.0
    
    print(f"Win Rate:     {stats['win_rate']*100:.1f}% (Target: ≥{win_rate_target}%) {'✓' if stats['win_rate']*100 >= win_rate_target else '✗'}")
    print(f"Max Drawdown: {stats['max_drawdown']*100:.1f}% (Target: ≤{drawdown_target}%) {'✓' if stats['max_drawdown']*100 <= drawdown_target else '✗'}")
    print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f} (Target: ≥{sharpe_target}) {'✓' if stats['sharpe_ratio'] >= sharpe_target else '✗'}")
    
    # Risk-adjusted return
    if stats['max_drawdown'] > 0:
        calmar_ratio = stats['total_return'] / stats['max_drawdown']
        print(f"Calmar Ratio:  {calmar_ratio:.2f}")


def plot_results(stats: dict, initial_equity: float, save_path: str = 'backtest_results.png'):
    """Plot equity curve and drawdown."""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        equity_history = stats['equity_history']
        drawdown_history = stats['drawdown_history']
        position_history = stats['position_history']
        
        # Equity curve
        axes[0].plot(equity_history, linewidth=2, color='blue')
        axes[0].axhline(y=initial_equity, color='gray', linestyle='--', label='Initial Equity')
        axes[0].set_title('Equity Curve', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Equity ($)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Drawdown
        drawdown_pct = [d * 100 for d in drawdown_history]
        axes[1].fill_between(range(len(drawdown_pct)), drawdown_pct, 0, 
                            color='red', alpha=0.3)
        axes[1].plot(drawdown_pct, linewidth=1, color='red')
        axes[1].axhline(y=-20, color='orange', linestyle='--', label='20% Drawdown Limit')
        axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Drawdown (%)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Position size
        axes[2].plot(position_history, linewidth=1, color='green', alpha=0.6)
        axes[2].set_title('Position Size Over Time', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Step', fontsize=12)
        axes[2].set_ylabel('Position Size', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n📊 Charts saved to: {save_path}")
        plt.close()
        
    except ImportError:
        print("\n⚠️  matplotlib not available, skipping charts")


def main():
    """Main backtest function."""
    print("="*60)
    print("RL TRADING SYSTEM - 6 MONTH BACKTEST")
    print("="*60)
    
    # Parameters
    initial_equity = 10000.0
    n_months = 6
    start_date = (datetime.now() - timedelta(days=180)).strftime('%Y-%m-%d')
    
    # Generate data
    data = generate_realistic_gbpjpy_data(start_date, n_months)
    
    # Split into train/test (70/30)
    train_data, test_data = split_data(data, train_ratio=0.7)
    
    print(f"\nData split:")
    print(f"  Training: {len(train_data)} bars ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  Testing:  {len(test_data)} bars ({len(test_data)/len(data)*100:.1f}%)")
    
    # Create training environment
    train_env = TradingEnvironment(
        data=train_data,
        initial_equity=initial_equity,
        commission=0.0001,
        slippage=0.0001
    )
    
    # Train agent
    agent = train_agent(train_env, episodes=200)  # Reduced for faster demo
    
    # Create test environment
    test_env = TradingEnvironment(
        data=test_data,
        initial_equity=initial_equity,
        commission=0.0001,
        slippage=0.0001
    )
    
    # Create risk governor
    risk_gov = RiskGovernor(
        max_drawdown_pct=0.20,
        max_position_size_pct=0.50,
        daily_loss_limit_pct=0.05,
        consecutive_loss_limit=5
    )
    
    # Run backtest
    stats = run_backtest(test_env, agent, risk_gov)
    
    # Print results
    print_results(stats, initial_equity)
    
    # Plot results
    plot_results(stats, initial_equity)
    
    print(f"\n{'='*60}")
    print("BACKTEST COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

