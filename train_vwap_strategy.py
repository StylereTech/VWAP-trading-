"""
Training Script for VWAP Pro Strategy
Uses reinforcement learning to improve entry/exit timing
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capital_allocation_ai.vwap_pro_strategy import VWAPProStrategy
from capital_allocation_ai.system import CapitalAllocationAI, CapitalAllocationAgent


class VWAPStrategyTrainer:
    """
    Trains the VWAP strategy using RL to improve entry/exit decisions.
    """
    
    def __init__(self, symbol: str, initial_equity: float = 10000.0):
        self.symbol = symbol
        self.initial_equity = initial_equity
        
        # Initialize RL agent
        from capital_allocation_ai.system import CapitalAllocationStateEncoder
        encoder = CapitalAllocationStateEncoder()
        state_size = encoder.get_state_size()
        
        self.agent = CapitalAllocationAgent(
            state_size=state_size,
            max_risk=0.3,  # Conservative risk
            lr=3e-4,
            gamma=0.99
        )
        
        self.state_encoder = encoder
        
        # Training history
        self.training_history = []
    
    def train_on_data(self, data: pd.DataFrame, episodes: int = 100, update_frequency: int = 200):
        """
        Train RL agent on historical data.
        
        Args:
            data: Historical market data
            episodes: Number of training episodes
            update_frequency: Steps between training updates
        """
        print(f"\n{'='*80}")
        print(f"TRAINING VWAP STRATEGY - {self.symbol}")
        print(f"{'='*80}")
        print(f"Episodes: {episodes}")
        print(f"Data bars: {len(data)}")
        print(f"Update frequency: {update_frequency} steps")
        print("-"*80)
        
        episode_returns = []
        episode_sharpes = []
        episode_trades = []
        
        for episode in range(episodes):
            # Reset for new episode
            equity = self.initial_equity
            peak_equity = self.initial_equity
            position = 0
            entry_price = 0.0
            trades = []
            
            # Initialize strategy
            strategy = VWAPProStrategy()
            
            # Reset state encoder
            self.state_encoder = type(self.state_encoder)()
            
            total_reward = 0
            steps = 0
            
            for idx, row in data.iterrows():
                # Update strategy
                result = strategy.update(
                    current_price=row['close'],
                    high=row['high'],
                    low=row['low'],
                    close=row['close'],
                    volume=row['volume'],
                    timestamp=row.get('timestamp')
                )
                
                # Encode state
                state = self.state_encoder.encode_state(
                    current_price=row['close'],
                    high=row['high'],
                    low=row['low'],
                    volume=row['volume'],
                    vwap=result.get('vwap', row['close']),
                    position_size=0.1 if position == 1 else (-0.1 if position == -1 else 0.0),
                    unrealized_pnl=0.0,
                    account_equity=equity,
                    peak_equity=peak_equity,
                    timestamp=row.get('timestamp')
                )
                
                # Get RL action
                direction, pos_size, value, log_prob = self.agent.act(state)
                
                # Combine strategy signal with RL decision
                signals = result['signals']
                strategy_signal = 0
                if signals['entry_long'] or signals['reposition_long']:
                    strategy_signal = 1
                elif signals['exit'] or signals['stop_hit']:
                    strategy_signal = 0
                
                # RL can override or confirm strategy signal
                # If RL says long and strategy says long = strong signal
                # If RL says flat but strategy says long = weak signal (reduce size)
                final_direction = direction
                if strategy_signal == 1 and direction == 0:
                    # Strategy says enter but RL says wait - reduce confidence
                    final_direction = 0  # Don't enter
                elif strategy_signal == 1 and direction == 1:
                    # Both agree - strong signal
                    final_direction = 1
                elif strategy_signal == 0 and direction == 1:
                    # RL says enter but strategy doesn't - use RL but smaller size
                    final_direction = 1
                    pos_size *= 0.5  # Reduce size
                
                # Execute action
                if final_direction == 1 and position == 0:
                    position = 1
                    entry_price = row['close']
                elif final_direction == 0 and position != 0:
                    # Exit
                    exit_price = row['close']
                    pnl = (exit_price - entry_price) * (equity / entry_price * pos_size)
                    equity += pnl
                    trades.append({'entry': entry_price, 'exit': exit_price, 'pnl': pnl})
                    position = 0
                    entry_price = 0.0
                
                # Calculate reward
                current_equity = equity
                if position == 1:
                    unrealized_pnl = (row['close'] - entry_price) * (equity / entry_price * pos_size)
                    current_equity += unrealized_pnl
                
                peak_equity = max(peak_equity, current_equity)
                equity_change = current_equity - equity
                equity_change_pct = equity_change / equity if equity > 0 else 0.0
                drawdown = (peak_equity - current_equity) / peak_equity if peak_equity > 0 else 0.0
                
                # Reward: equity change - drawdown penalty - trade frequency penalty
                reward = equity_change_pct - 2.0 * drawdown
                if len(trades) > steps * 0.1:  # Penalize too many trades
                    reward -= 0.001
                
                # Store transition
                next_state = self.state_encoder.encode_state(
                    current_price=row['close'],
                    high=row['high'],
                    low=row['low'],
                    volume=row['volume'],
                    vwap=result.get('vwap', row['close']),
                    position_size=pos_size if position == 1 else 0.0,
                    unrealized_pnl=unrealized_pnl if position == 1 else 0.0,
                    account_equity=current_equity,
                    peak_equity=peak_equity,
                    timestamp=row.get('timestamp')
                )
                
                self.agent.store_transition(
                    state, (final_direction, pos_size), reward, value, log_prob, False
                )
                
                total_reward += reward
                steps += 1
                
                # Train periodically
                if steps % update_frequency == 0 and len(self.agent.memory['states']) > 0:
                    loss = self.agent.train(epochs=4)
            
            # Final training
            if len(self.agent.memory['states']) > 0:
                self.agent.train(epochs=4)
            
            # Calculate episode statistics
            total_return = (equity - self.initial_equity) / self.initial_equity
            
            # Calculate Sharpe
            if len(trades) > 0:
                trades_df = pd.DataFrame(trades)
                returns = trades_df['pnl'] / self.initial_equity
                sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 1 else 0.0
            else:
                sharpe = 0.0
            
            episode_returns.append(total_return)
            episode_sharpes.append(sharpe)
            episode_trades.append(len(trades))
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_return = np.mean(episode_returns[-10:])
                avg_sharpe = np.mean(episode_sharpes[-10:])
                avg_trades = np.mean(episode_trades[-10:])
                print(f"Episode {episode+1}/{episodes} | "
                      f"Return: {avg_return*100:.2f}% | "
                      f"Sharpe: {avg_sharpe:.2f} | "
                      f"Trades: {avg_trades:.1f}")
        
        # Final statistics
        print(f"\n{'='*80}")
        print("TRAINING COMPLETE")
        print(f"{'='*80}")
        print(f"Final Episode Return: {episode_returns[-1]*100:.2f}%")
        print(f"Final Episode Sharpe: {episode_sharpes[-1]:.2f}")
        print(f"Final Episode Trades: {episode_trades[-1]}")
        print(f"Average Return (last 10): {np.mean(episode_returns[-10:])*100:.2f}%")
        print(f"Average Sharpe (last 10): {np.mean(episode_sharpes[-10:]):.2f}")
        
        return self.agent
    
    def save_model(self, filepath: str):
        """Save trained model."""
        self.agent.save(filepath)
        print(f"Model saved to: {filepath}")


def generate_market_data(symbol: str, days: int = 60) -> pd.DataFrame:
    """Generate market data (same as optimize script)."""
    from optimize_vwap_params import generate_market_data as gen_data
    return gen_data(symbol, days)


def main():
    """Main training function."""
    print("="*80)
    print("VWAP PRO STRATEGY TRAINING")
    print("Training RL agent to improve entry/exit timing")
    print("="*80)
    
    symbols = ['GBPJPY', 'BTCUSD', 'XAUUSD']
    trained_models = {}
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"TRAINING ON {symbol}")
        print(f"{'='*80}")
        
        # Generate training data
        data = generate_market_data(symbol, days=60)
        
        # Create trainer
        trainer = VWAPStrategyTrainer(symbol)
        
        # Train
        agent = trainer.train_on_data(data, episodes=50, update_frequency=200)
        
        # Save model
        model_path = f"trained_vwap_{symbol.lower()}.pth"
        trainer.save_model(model_path)
        trained_models[symbol] = model_path
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE - ALL SYMBOLS")
    print("="*80)
    print("Trained models saved:")
    for symbol, path in trained_models.items():
        print(f"  {symbol}: {path}")
    
    print("\nNext steps:")
    print("1. Use optimized parameters from optimize_vwap_params.py")
    print("2. Use trained models for improved entry/exit timing")
    print("3. Run backtest with both optimizations")


if __name__ == "__main__":
    main()
