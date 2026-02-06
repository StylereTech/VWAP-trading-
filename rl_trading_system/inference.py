"""
Inference script for RL Trading Agent
Runs trained agent on live or historical data.
"""

import numpy as np
import pandas as pd
import torch
import argparse
import os
from datetime import datetime

from .ppo_agent import PPOAgent
from .trading_environment import TradingEnvironment
from .risk_governor import RiskGovernor
from .traderlocker_executor import TraderLockerExecutor, MockExecutor


def load_data(filepath: str) -> pd.DataFrame:
    """Load historical data."""
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            raise ValueError(f"Data must contain: {required}")
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def run_inference(env: TradingEnvironment,
                 agent: PPOAgent,
                 executor,
                 risk_gov: RiskGovernor,
                 live: bool = False,
                 symbol: str = 'GBPJPY'):
    """
    Run inference with trained agent.
    
    Args:
        env: Trading environment
        agent: Trained PPO agent
        executor: Execution engine (TraderLocker or Mock)
        risk_gov: Risk governor
        live: Whether running live trading
        symbol: Trading symbol
    """
    agent.model.eval()  # Set to evaluation mode
    
    state = env.reset()
    risk_gov.initialize(env.initial_equity)
    
    done = False
    step = 0
    
    print("=" * 60)
    print("RUNNING INFERENCE")
    print("=" * 60)
    print(f"Mode: {'LIVE' if live else 'BACKTEST'}")
    print(f"Symbol: {symbol}")
    print(f"Initial Equity: ${env.initial_equity:.2f}")
    print("-" * 60)
    
    while not done:
        # Check risk limits
        current_equity = env.account_equity + env.unrealized_pnl
        risk_gov.update_equity(current_equity)
        
        allowed, reason = risk_gov.check_risk_limits(current_equity)
        if not allowed:
            print(f"\nTRADING HALTED: {reason}")
            if env.position_size != 0:
                # Close positions
                current_price = env.data.iloc[env.current_step]['close']
                env._close_position(current_price)
            break
        
        # Agent selects action (deterministic for inference)
        direction, pos_size, value, _ = agent.act(state, deterministic=False)
        
        # Adjust position size based on risk governor
        adjusted_size = risk_gov.adjust_position_size(pos_size, current_equity)
        action = (direction, adjusted_size)
        
        # Execute in environment
        next_state, reward, done, info = env.step(action)
        
        # Execute trade if not flat
        if direction != 0 and live:
            current_price = env.data.iloc[env.current_step]['close']
            
            # Calculate stop loss and take profit (simplified)
            atr = env.state_encoder.calculate_atr(
                np.array([env.data.iloc[env.current_step]['high']]),
                np.array([env.data.iloc[env.current_step]['low']]),
                np.array([current_price]),
                14
            )
            
            stop_loss = None
            take_profit = None
            if direction == 1:  # Long
                stop_loss = current_price - atr * 1.5
                take_profit = current_price + atr * 2.0
            else:  # Short
                stop_loss = current_price + atr * 1.5
                take_profit = current_price - atr * 2.0
            
            # Execute via TraderLocker
            result = executor.execute_trade(
                symbol=symbol,
                action=action,
                current_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            
            if 'error' in result:
                print(f"Execution error: {result['error']}")
        
        state = next_state
        step += 1
        
        # Print progress
        if step % 100 == 0:
            print(f"Step {step} | Equity: ${info['equity']:.2f} | "
                  f"DD: {info['drawdown']*100:.1f}% | "
                  f"Position: {info['position_size']:.2f} | "
                  f"Reward: {reward:.4f}")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print("=" * 60)
    stats = env.get_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            if 'rate' in key or 'return' in key or 'drawdown' in key:
                print(f"{key}: {value*100:.2f}%")
            else:
                print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")


def main():
    parser = argparse.ArgumentParser(description='Run RL Trading Agent Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, help='Path to historical data (for backtest)')
    parser.add_argument('--live', action='store_true', help='Run live trading')
    parser.add_argument('--symbol', type=str, default='GBPJPY', help='Trading symbol')
    parser.add_argument('--api-key', type=str, help='TraderLocker API key (for live trading)')
    parser.add_argument('--initial-equity', type=float, default=10000.0, help='Initial equity')
    
    args = parser.parse_args()
    
    if args.live and not args.api_key:
        raise ValueError("API key required for live trading")
    
    # Load agent
    print(f"Loading model from {args.model}...")
    state_size = 15  # Should match training state size
    agent = PPOAgent(state_size=state_size)
    agent.load(args.model)
    print("Model loaded successfully")
    
    # Create executor
    if args.live:
        executor = TraderLockerExecutor(api_key=args.api_key, sandbox=True)
    else:
        executor = MockExecutor()
    
    # Create environment
    if args.data:
        data = load_data(args.data)
        env = TradingEnvironment(data=data, initial_equity=args.initial_equity)
    else:
        # For live trading, you'd fetch data from API
        # This is a placeholder - implement data fetching for live trading
        raise ValueError("Data file required for backtesting")
    
    # Create risk governor
    risk_gov = RiskGovernor()
    
    # Run inference
    run_inference(
        env=env,
        agent=agent,
        executor=executor,
        risk_gov=risk_gov,
        live=args.live,
        symbol=args.symbol
    )


if __name__ == "__main__":
    main()

