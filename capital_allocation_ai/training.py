"""
Training script for Capital Allocation AI.
Trains the system to learn compounding behavior.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from .system import CapitalAllocationAI


class TrainingEnvironment:
    """
    Training environment with reward function:
    reward = Δ(account_equity) - λ × drawdown - γ × volatility_of_equity
    """
    
    def __init__(self, data: pd.DataFrame, initial_equity: float = 10000.0):
        self.data = data.reset_index(drop=True)
        self.initial_equity = initial_equity
        
        # Reward parameters
        self.lambda_drawdown = 2.0  # Drawdown penalty multiplier
        self.gamma_volatility = 0.5  # Volatility penalty multiplier
        
        # Initialize AI system
        self.ai = CapitalAllocationAI(
            symbol="GBPJPY",
            initial_equity=initial_equity,
            max_risk=0.5,
            broker_type="mock"
        )
        
        self.current_step = 0
        self.previous_equity = initial_equity
    
    def reset(self):
        """Reset environment."""
        self.current_step = 0
        self.previous_equity = self.initial_equity
        
        # Reset AI system
        self.ai = CapitalAllocationAI(
            symbol="GBPJPY",
            initial_equity=self.initial_equity,
            max_risk=0.5,
            broker_type="mock"
        )
        
        return self._get_state()
    
    def _get_state(self):
        """Get current state."""
        if self.current_step >= len(self.data):
            return None
        
        row = self.data.iloc[self.current_step]
        return {
            'price': row['close'],
            'high': row['high'],
            'low': row['low'],
            'volume': row['volume'],
            'vwap': row.get('vwap', row['close'])
        }
    
    def step(self) -> Tuple[Dict, float, bool]:
        """
        Execute one step.
        
        Returns:
            (state, reward, done)
        """
        if self.current_step >= len(self.data) - 1:
            # Close position at end
            if self.ai.position_direction != 0:
                row = self.data.iloc[self.current_step]
                self.ai._close_position(row['close'])
            
            return None, 0.0, True
        
        row = self.data.iloc[self.current_step]
        
        # Update AI with market data
        result = self.ai.update_market_data(
            current_price=row['close'],
            high=row['high'],
            low=row['low'],
            volume=row['volume'],
            vwap=row.get('vwap', row['close']),
            timestamp=row.get('timestamp')
        )
        
        # Calculate reward
        current_equity = result['equity']
        
        # Equity change
        equity_change = current_equity - self.previous_equity
        equity_change_pct = equity_change / self.previous_equity if self.previous_equity > 0 else 0.0
        
        # Drawdown penalty
        drawdown = result['drawdown']
        drawdown_penalty = self.lambda_drawdown * drawdown
        
        # Volatility penalty
        if len(self.ai.equity_history) >= 10:
            equity_returns = np.diff(self.ai.equity_history[-10:]) / self.ai.equity_history[-10:-1]
            equity_volatility = np.std(equity_returns) if len(equity_returns) > 0 else 0.0
            volatility_penalty = self.gamma_volatility * equity_volatility
        else:
            volatility_penalty = 0.0
        
        # Total reward
        reward = equity_change_pct - drawdown_penalty - volatility_penalty
        
        self.previous_equity = current_equity
        self.current_step += 1
        
        # Get next state
        next_state = self._get_state()
        done = self.current_step >= len(self.data) - 1
        
        return next_state, reward, done
    
    def get_statistics(self) -> Dict:
        """Get training statistics."""
        return self.ai.get_statistics()


def train_capital_allocation_ai(data: pd.DataFrame,
                                episodes: int = 1000,
                                initial_equity: float = 10000.0,
                                update_frequency: int = 200) -> CapitalAllocationAI:
    """
    Train Capital Allocation AI.
    
    Args:
        data: Market data DataFrame
        episodes: Number of training episodes
        initial_equity: Starting equity
        update_frequency: Steps between training updates
    
    Returns:
        Trained AI system
    """
    env = TrainingEnvironment(data, initial_equity)
    ai = env.ai
    
    print("=" * 60)
    print("CAPITAL ALLOCATION AI TRAINING")
    print("=" * 60)
    print(f"Episodes: {episodes}")
    print(f"Data bars: {len(data)}")
    print(f"Initial equity: ${initial_equity:,.2f}")
    print("-" * 60)
    
    episode_rewards = []
    episode_returns = []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done and state is not None:
            # Get action from agent
            direction, pos_size, value, log_prob = ai.agent.act(ai.state_encoder.encode_state(
                current_price=state['price'],
                high=state['high'],
                low=state['low'],
                volume=state['volume'],
                vwap=state['vwap']
            ))
            
            # Store transition
            ai.agent.store_transition(
                state=ai.state_encoder.encode_state(
                    current_price=state['price'],
                    high=state['high'],
                    low=state['low'],
                    volume=state['volume'],
                    vwap=state['vwap']
                ),
                action=(direction, pos_size),
                reward=0.0,  # Will be updated
                value=value,
                log_prob=log_prob,
                done=False
            )
            
            # Execute step
            next_state, reward, done = env.step()
            
            # Update last transition with reward
            if len(ai.agent.memory['rewards']) > 0:
                ai.agent.memory['rewards'][-1] = reward
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Train periodically
            if steps % update_frequency == 0 and len(ai.agent.memory['states']) > 0:
                loss = ai.agent.train(epochs=4)
        
        # Final training
        if len(ai.agent.memory['states']) > 0:
            ai.agent.train(epochs=4)
        
        # Get episode statistics
        stats = env.get_statistics()
        episode_rewards.append(total_reward)
        episode_returns.append(stats['total_return'])
        
        # Print progress
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            avg_return = np.mean(episode_returns[-50:])
            print(f"Episode {episode+1}/{episodes} | "
                  f"Avg Reward: {avg_reward:.4f} | "
                  f"Avg Return: {avg_return*100:.2f}% | "
                  f"Trades: {stats['total_trades']} | "
                  f"Win Rate: {stats['win_rate']*100:.1f}%")
    
    # Final statistics
    final_stats = env.get_statistics()
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total Trades: {final_stats['total_trades']}")
    print(f"Win Rate: {final_stats['win_rate']*100:.2f}%")
    print(f"Total Return: {final_stats['total_return']*100:.2f}%")
    print(f"Max Drawdown: {final_stats['max_drawdown']*100:.2f}%")
    print(f"Final Equity: ${final_stats['final_equity']:,.2f}")
    
    return ai


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python training.py <data_file.csv> [episodes]")
        sys.exit(1)
    
    data_file = sys.argv[1]
    episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    # Load data
    data = pd.read_csv(data_file)
    
    # Train
    ai = train_capital_allocation_ai(data, episodes=episodes)
    
    # Save model
    model_path = f"capital_allocation_ai_{datetime.now().strftime('%Y%m%d')}.pth"
    ai.agent.save(model_path)
    print(f"\nModel saved to: {model_path}")
