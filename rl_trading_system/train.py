"""
Training script for RL Trading Agent
Trains PPO agent on historical data.
"""

import numpy as np
import pandas as pd
import torch
import argparse
import os
from datetime import datetime
import json

from .ppo_agent import PPOAgent
from .trading_environment import TradingEnvironment
from .risk_governor import RiskGovernor


def load_data(filepath: str) -> pd.DataFrame:
    """Load historical data from CSV or other format."""
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required):
            raise ValueError(f"Data must contain: {required}")
        
        # Convert timestamp if exists
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df
    else:
        raise ValueError(f"Unsupported file format: {filepath}")


def train_agent(env: TradingEnvironment,
                agent: PPOAgent,
                episodes: int = 1000,
                update_frequency: int = 200,
                save_frequency: int = 100,
                save_dir: str = './models'):
    """
    Train RL agent.
    
    Args:
        env: Trading environment
        agent: PPO agent
        episodes: Number of training episodes
        update_frequency: Steps between training updates
        save_frequency: Episodes between model saves
        save_dir: Directory to save models
    """
    os.makedirs(save_dir, exist_ok=True)
    
    episode_rewards = []
    episode_returns = []
    episode_drawdowns = []
    
    print(f"Starting training for {episodes} episodes...")
    print(f"State size: {agent.state_size}")
    print(f"Update frequency: {update_frequency} steps")
    print("-" * 60)
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            # Agent selects action
            direction, pos_size, value, log_prob = agent.act(state)
            action = (direction, pos_size)
            
            # Execute action in environment
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            agent.store_transition(state, action, reward, value, log_prob, done)
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Train agent periodically
            if steps % update_frequency == 0 and len(agent.memory['states']) > 0:
                loss = agent.train(epochs=4)
                if steps % (update_frequency * 5) == 0:
                    print(f"  Step {steps}: Training loss = {loss:.4f}")
        
        # Get episode statistics
        stats = env.get_statistics()
        episode_rewards.append(total_reward)
        episode_returns.append(stats['total_return'])
        episode_drawdowns.append(stats['max_drawdown'])
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            avg_return = np.mean(episode_returns[-10:])
            avg_dd = np.mean(episode_drawdowns[-10:])
            
            print(f"Episode {episode+1}/{episodes} | "
                  f"Avg Reward: {avg_reward:.4f} | "
                  f"Avg Return: {avg_return*100:.2f}% | "
                  f"Avg DD: {avg_dd*100:.2f}% | "
                  f"Trades: {stats['total_trades']} | "
                  f"Win Rate: {stats['win_rate']*100:.1f}%")
        
        # Save model periodically
        if (episode + 1) % save_frequency == 0:
            model_path = os.path.join(save_dir, f"ppo_model_ep{episode+1}.pth")
            agent.save(model_path)
            print(f"Model saved: {model_path}")
    
    # Final statistics
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final Episode Stats:")
    final_stats = env.get_statistics()
    for key, value in final_stats.items():
        if isinstance(value, float):
            if 'rate' in key or 'return' in key or 'drawdown' in key:
                print(f"  {key}: {value*100:.2f}%")
            else:
                print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Save final model
    final_model_path = os.path.join(save_dir, "ppo_model_final.pth")
    agent.save(final_model_path)
    print(f"\nFinal model saved: {final_model_path}")
    
    # Save training history
    history = {
        'episode_rewards': episode_rewards,
        'episode_returns': episode_returns,
        'episode_drawdowns': episode_drawdowns,
        'final_stats': final_stats
    }
    history_path = os.path.join(save_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved: {history_path}")


def main():
    parser = argparse.ArgumentParser(description='Train RL Trading Agent')
    parser.add_argument('--data', type=str, required=True, help='Path to historical data CSV')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--update-freq', type=int, default=200, help='Steps between training updates')
    parser.add_argument('--save-freq', type=int, default=100, help='Episodes between model saves')
    parser.add_argument('--initial-equity', type=float, default=10000.0, help='Initial equity')
    parser.add_argument('--save-dir', type=str, default='./models', help='Directory to save models')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.data}...")
    data = load_data(args.data)
    print(f"Loaded {len(data)} bars")
    
    # Create environment
    env = TradingEnvironment(
        data=data,
        initial_equity=args.initial_equity
    )
    
    # Create agent
    state_size = env.state_size
    agent = PPOAgent(
        state_size=state_size,
        lr=args.lr,
        gamma=args.gamma
    )
    
    # Train
    train_agent(
        env=env,
        agent=agent,
        episodes=args.episodes,
        update_frequency=args.update_freq,
        save_frequency=args.save_freq,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()

