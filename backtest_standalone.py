"""
Standalone Backtest Script - RL Trading System
6 Month Backtest with Results
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import RL components
from rl_trading_system.ppo_agent import PPOAgent
from rl_trading_system.trading_environment import TradingEnvironment
from rl_trading_system.risk_governor import RiskGovernor


def generate_data(n_months=6):
    """Generate 6 months of realistic GBP/JPY data."""
    print(f"Generating {n_months} months of GBP/JPY data...")
    
    # Use 5-minute bars for faster processing
    trading_days_per_month = 21
    bars_per_day = 288  # 5-minute bars
    total_bars = trading_days_per_month * n_months * bars_per_day
    
    np.random.seed(42)
    base_price = 185.0
    
    # Generate realistic price movements
    trend = np.linspace(0, 5, total_bars)
    volatility = np.ones(total_bars) * 0.0005
    
    for i in range(1, total_bars):
        volatility[i] = 0.0003 + 0.7 * volatility[i-1] + 0.2 * abs(np.random.randn() * 0.0003)
        volatility[i] = min(volatility[i], 0.002)
    
    returns = np.random.randn(total_bars) * volatility
    for i in range(1, total_bars):
        if abs(returns[i-1]) > 0.001:
            returns[i] -= 0.3 * returns[i-1]
    
    returns += trend / total_bars
    prices = base_price + np.cumsum(returns)
    prices = np.clip(prices, 150, 220)
    
    opens = prices.copy()
    highs = opens + abs(np.random.randn(total_bars) * volatility * opens * 2)
    lows = opens - abs(np.random.randn(total_bars) * volatility * opens * 2)
    closes = opens + np.random.randn(total_bars) * volatility * opens
    
    for i in range(total_bars):
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
    
    base_volume = 5000
    volume_multiplier = 1 + volatility / volatility.mean() * 2
    volumes = (base_volume * volume_multiplier * (1 + abs(returns) * 10)).astype(int)
    volumes = np.clip(volumes, 1000, 50000)
    
    # Calculate VWAP and generate timestamps
    vwap_values = []
    current_day = None
    day_prices = []
    day_volumes = []
    
    start_date = datetime.now() - timedelta(days=180)
    timestamps = []
    current_date = start_date
    
    bar_count = 0
    for i in range(total_bars):
        # Skip weekends
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        
        timestamps.append(current_date)
        
        if current_day != current_date.date():
            current_day = current_date.date()
            day_prices = []
            day_volumes = []
        
        day_prices.append(closes[i])
        day_volumes.append(volumes[i])
        
        if sum(day_volumes) > 0:
            vwap = sum(p * v for p, v in zip(day_prices, day_volumes)) / sum(day_volumes)
        else:
            vwap = closes[i]
        
        vwap_values.append(vwap)
        
        # Move to next 5-minute bar
        bar_count += 1
        if bar_count % 288 == 0:
            current_date += timedelta(days=1)
        else:
            current_date += timedelta(minutes=5)
    
    # Ensure all arrays have same length
    min_len = min(len(timestamps), len(closes), len(opens), len(highs), len(lows), len(volumes), len(vwap_values))
    
    data = pd.DataFrame({
        'timestamp': timestamps[:min_len],
        'open': opens[:min_len],
        'high': highs[:min_len],
        'low': lows[:min_len],
        'close': closes[:min_len],
        'volume': volumes[:min_len],
        'vwap': vwap_values[:min_len]
    })
    
    print(f"Generated {len(data)} bars")
    return data


def train_agent(env, episodes=150):
    """Train PPO agent."""
    print(f"\n{'='*60}")
    print("TRAINING PHASE")
    print(f"{'='*60}")
    
    agent = PPOAgent(state_size=env.state_size, lr=3e-4, gamma=0.99)
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        
        while not done:
            direction, pos_size, value, log_prob = agent.act(state)
            action = (direction, pos_size)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, value, log_prob, done)
            state = next_state
            steps += 1
            
            if steps % 200 == 0 and len(agent.memory['states']) > 0:
                agent.train(epochs=3)
        
        if (episode + 1) % 25 == 0:
            stats = env.get_statistics()
            print(f"Episode {episode+1}/{episodes} | "
                  f"Return: {stats['total_return']*100:.2f}% | "
                  f"Trades: {stats['total_trades']} | "
                  f"Win Rate: {stats['win_rate']*100:.1f}%")
    
    return agent


def run_backtest(env, agent, risk_gov):
    """Run backtest."""
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
    
    while not done:
        current_equity = env.account_equity + env.unrealized_pnl
        risk_gov.update_equity(current_equity)
        allowed, reason = risk_gov.check_risk_limits(current_equity)
        
        if not allowed:
            print(f"\nTrading halted: {reason}")
            break
        
        direction, pos_size, value, _ = agent.act(state, deterministic=False)
        adjusted_size = risk_gov.adjust_position_size(pos_size, current_equity)
        action = (direction, adjusted_size)
        
        next_state, reward, done, info = env.step(action)
        
        equity_history.append(info['equity'])
        drawdown_history.append(info['drawdown'])
        
        state = next_state
        step += 1
        
        if step % 500 == 0:
            print(f"Step {step} | Equity: ${info['equity']:.2f} | "
                  f"DD: {info['drawdown']*100:.1f}%")
    
    stats = env.get_statistics()
    stats['equity_history'] = equity_history
    stats['drawdown_history'] = drawdown_history
    return stats


def print_results(stats, initial_equity):
    """Print results."""
    print(f"\n{'='*60}")
    print("BACKTEST RESULTS - 6 MONTHS")
    print(f"{'='*60}")
    
    print(f"\nPERFORMANCE METRICS")
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
    
    if len(stats['equity_history']) > 0:
        equity_array = np.array(stats['equity_history'])
        returns = np.diff(equity_array) / equity_array[:-1]
        volatility = np.std(returns) * np.sqrt(252 * 288) * 100
        print(f"Equity Volatility:      {volatility:.2f}%")
    
    print(f"\nTARGET COMPARISON")
    print(f"{'-'*60}")
    win_check = "PASS" if stats['win_rate']*100 >= 55 else "FAIL"
    dd_check = "PASS" if stats['max_drawdown']*100 <= 20 else "FAIL"
    sharpe_check = "PASS" if stats['sharpe_ratio'] >= 1.0 else "FAIL"
    print(f"Win Rate:     {stats['win_rate']*100:.1f}% (Target: >=55%) [{win_check}]")
    print(f"Max Drawdown: {stats['max_drawdown']*100:.1f}% (Target: <=20%) [{dd_check}]")
    print(f"Sharpe Ratio: {stats['sharpe_ratio']:.2f} (Target: >=1.0) [{sharpe_check}]")


def main():
    print("="*60)
    print("RL TRADING SYSTEM - 6 MONTH BACKTEST")
    print("="*60)
    
    initial_equity = 10000.0
    
    # Generate data
    data = generate_data(n_months=6)
    
    # Split 70/30
    split_idx = int(len(data) * 0.7)
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    print(f"\nData split: Train={len(train_data)}, Test={len(test_data)}")
    
    # Train
    train_env = TradingEnvironment(data=train_data, initial_equity=initial_equity)
    agent = train_agent(train_env, episodes=150)
    
    # Test
    test_env = TradingEnvironment(data=test_data, initial_equity=initial_equity)
    risk_gov = RiskGovernor(max_drawdown_pct=0.20, max_position_size_pct=0.50)
    stats = run_backtest(test_env, agent, risk_gov)
    
    # Results
    print_results(stats, initial_equity)
    
    print(f"\n{'='*60}")
    print("BACKTEST COMPLETE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

