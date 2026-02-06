"""
Example usage of RL Trading System
Demonstrates how to use the system for training and inference.
"""

import pandas as pd
import numpy as np
from ppo_agent import PPOAgent
from trading_environment import TradingEnvironment
from risk_governor import RiskGovernor
from traderlocker_executor import MockExecutor


def create_sample_data(n_bars: int = 1000) -> pd.DataFrame:
    """Create sample market data for testing."""
    dates = pd.date_range('2024-01-01', periods=n_bars, freq='5min')
    
    # Generate realistic price data with trend
    np.random.seed(42)
    returns = np.random.randn(n_bars) * 0.001
    prices = 150.0 + np.cumsum(returns)
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices + np.random.randn(n_bars) * 0.0001,
        'high': prices + abs(np.random.randn(n_bars) * 0.0005),
        'low': prices - abs(np.random.randn(n_bars) * 0.0005),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_bars),
    })
    
    # Calculate simple VWAP
    data['vwap'] = (data['high'] + data['low'] + data['close']) / 3
    
    return data


def example_training():
    """Example: Train the RL agent."""
    print("=" * 60)
    print("EXAMPLE: Training RL Agent")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data(n_bars=500)
    print(f"Created {len(data)} bars of sample data")
    
    # Create environment
    env = TradingEnvironment(
        data=data,
        initial_equity=10000.0,
        commission=0.0001,
        slippage=0.0001
    )
    print(f"State size: {env.state_size}")
    
    # Create agent
    agent = PPOAgent(
        state_size=env.state_size,
        lr=3e-4,
        gamma=0.99
    )
    
    # Train for a few episodes (short example)
    print("\nTraining agent...")
    for episode in range(10):
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
        
        # Train every 50 steps
        if len(agent.memory['states']) >= 50:
            loss = agent.train(epochs=2)
            print(f"Episode {episode+1}: Reward={total_reward:.2f}, Loss={loss:.4f}, Steps={steps}")
    
    # Get final statistics
    stats = env.get_statistics()
    print("\nFinal Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            if 'rate' in key or 'return' in key or 'drawdown' in key:
                print(f"  {key}: {value*100:.2f}%")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Save model
    agent.save('example_model.pth')
    print("\nModel saved to 'example_model.pth'")


def example_inference():
    """Example: Run inference with trained agent."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Running Inference")
    print("=" * 60)
    
    # Create sample data
    data = create_sample_data(n_bars=200)
    
    # Create environment
    env = TradingEnvironment(
        data=data,
        initial_equity=10000.0
    )
    
    # Load agent (or create new one if no saved model)
    try:
        agent = PPOAgent(state_size=env.state_size)
        agent.load('example_model.pth')
        print("Loaded saved model")
    except:
        print("No saved model found, using new agent")
        agent = PPOAgent(state_size=env.state_size)
    
    # Create risk governor
    risk_gov = RiskGovernor()
    risk_gov.initialize(env.initial_equity)
    
    # Create executor (mock for example)
    executor = MockExecutor()
    
    # Run inference
    state = env.reset()
    done = False
    step = 0
    
    print("\nRunning inference...")
    while not done and step < 100:  # Limit steps for example
        # Check risk limits
        current_equity = env.account_equity + env.unrealized_pnl
        risk_gov.update_equity(current_equity)
        allowed, reason = risk_gov.check_risk_limits(current_equity)
        
        if not allowed:
            print(f"Trading halted: {reason}")
            break
        
        # Agent selects action (deterministic)
        direction, pos_size, value, _ = agent.act(state, deterministic=True)
        
        # Adjust position size
        adjusted_size = risk_gov.adjust_position_size(pos_size, current_equity)
        action = (direction, adjusted_size)
        
        # Execute
        next_state, reward, done, info = env.step(action)
        
        if step % 20 == 0:
            print(f"Step {step} | Equity: ${info['equity']:.2f} | "
                  f"DD: {info['drawdown']*100:.1f}% | "
                  f"Position: {info['position_size']:.2f} | "
                  f"Direction: {['Flat', 'Long', 'Short'][direction]}")
        
        state = next_state
        step += 1
    
    # Final statistics
    stats = env.get_statistics()
    print("\nInference Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            if 'rate' in key or 'return' in key or 'drawdown' in key:
                print(f"  {key}: {value*100:.2f}%")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")


def example_state_encoding():
    """Example: State encoding."""
    print("\n" + "=" * 60)
    print("EXAMPLE: State Encoding")
    print("=" * 60)
    
    from state_encoder import StateEncoder
    
    encoder = StateEncoder()
    
    # Simulate some price data
    prices = np.linspace(150, 155, 100)
    
    for i, price in enumerate(prices):
        state = encoder.encode_state(
            current_price=price,
            high=price * 1.001,
            low=price * 0.999,
            volume=1000.0,
            vwap=price * 0.998,
            timestamp=pd.Timestamp.now()
        )
        
        if i == len(prices) - 1:
            print(f"State vector size: {len(state)}")
            print(f"First 10 features: {state[:10]}")
            print(f"Feature breakdown:")
            print(f"  Returns (4): {state[0:4]}")
            print(f"  Volatility (2): {state[4:6]}")
            print(f"  Trend (1): {state[6]}")
            print(f"  RSI (1): {state[7]}")
            print(f"  VWAP dist (1): {state[8]}")
            print(f"  Volume ratio (1): {state[9]}")


if __name__ == "__main__":
    # Run examples
    example_state_encoding()
    example_training()
    example_inference()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)

