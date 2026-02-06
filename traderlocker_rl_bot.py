"""
TraderLocker RL Trading Bot - Integrated Version
Combines prototype simplicity with production features
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import requests
import time
import os
from datetime import datetime

# Import our existing components
from rl_trading_system.state_encoder import StateEncoder
from rl_trading_system.risk_governor import RiskGovernor

# -------------------
# Hyperparameters
# -------------------
STATE_SIZE = 15       # Will be set by StateEncoder
ACTION_SIZE = 2       # [direction, position_size]
HIDDEN_SIZE = 128
LR = 0.0003
GAMMA = 0.99
EPSILON = 0.2         # PPO clip
BATCH_SIZE = 64
UPDATE_EVERY = 200
MAX_EPISODES = 1000

# -------------------
# PPO Actor-Critic Network
# -------------------
class ActorCritic(nn.Module):
    def __init__(self, state_size, action_size, hidden_size):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        # Actor: direction (3 actions: flat, long, short)
        self.actor_direction = nn.Linear(hidden_size // 2, 3)
        # Actor: position size (continuous 0-1)
        self.actor_position = nn.Sequential(
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()
        )
        # Critic: state value
        self.critic = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        x = self.shared(x)
        direction_logits = self.actor_direction(x)
        position_size = self.actor_position(x)
        state_value = self.critic(x)
        return direction_logits, position_size, state_value

# -------------------
# PPO Agent
# -------------------
class PPOAgent:
    def __init__(self, state_size, action_size=2, hidden_size=128, lr=0.0003):
        self.state_size = state_size
        self.model = ActorCritic(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.memory = []
        self.gamma = GAMMA
        self.epsilon = EPSILON
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def act(self, state, deterministic=False):
        """Select action given state."""
        self.model.eval()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            direction_logits, position_size, value = self.model(state_tensor)
            
            # Sample direction
            if deterministic:
                direction = torch.argmax(direction_logits, dim=-1).item()
            else:
                direction_probs = torch.softmax(direction_logits, dim=-1)
                direction = torch.multinomial(direction_probs, 1).item()
            
            pos_size = position_size.item()
            log_prob = torch.log_softmax(direction_logits, dim=-1)[0, direction].item()
        
        return direction, pos_size, value.item(), log_prob

    def store(self, transition):
        """Store transition: (state, action, reward, next_state, done, value, log_prob)"""
        self.memory.append(transition)

    def train(self):
        """Train PPO agent on stored transitions."""
        if len(self.memory) < BATCH_SIZE:
            return
        
        # Convert to tensors
        states = torch.FloatTensor([t[0] for t in self.memory]).to(self.device)
        actions = [t[1] for t in self.memory]  # (direction, position_size)
        rewards = [t[2] for t in self.memory]
        next_states = torch.FloatTensor([t[3] for t in self.memory]).to(self.device)
        dones = [t[4] for t in self.memory]
        old_values = torch.FloatTensor([t[5] for t in self.memory]).to(self.device)
        old_log_probs = torch.FloatTensor([t[6] for t in self.memory]).to(self.device)
        
        # Compute returns and advantages
        returns = []
        advantages = []
        gae = 0
        
        for step in reversed(range(len(rewards))):
            if dones[step]:
                next_value = 0
            else:
                with torch.no_grad():
                    _, _, next_value = self.model(next_states[step:step+1])
                    next_value = next_value.item()
            
            delta = rewards[step] + self.gamma * next_value - old_values[step].item()
            gae = delta + self.gamma * 0.95 * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + old_values[step].item())
        
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        self.model.train()
        direction_logits, position_sizes, values = self.model(states)
        
        # Direction probabilities
        direction_probs = torch.softmax(direction_logits, dim=-1)
        direction_tensor = torch.LongTensor([a[0] for a in actions]).to(self.device)
        new_log_probs = torch.log_softmax(direction_logits, dim=-1)
        new_log_probs = new_log_probs.gather(1, direction_tensor.unsqueeze(1)).squeeze(1)
        
        # PPO clipped objective
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = nn.MSELoss()(values.squeeze(), returns)
        
        # Entropy bonus
        entropy = -(direction_probs * torch.log_softmax(direction_logits, dim=-1)).sum(dim=1).mean()
        
        # Total loss
        loss = actor_loss + 0.5 * value_loss - 0.01 * entropy
        
        # Update
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        self.optimizer.step()
        
        # Clear memory
        self.memory = []
        
        return loss.item()

# -------------------
# Trading Environment (TraderLocker)
# -------------------
class TradingEnv:
    def __init__(self, symbol, traderlocker_api_key=None, data=None):
        """
        Initialize trading environment.
        
        Args:
            symbol: Trading symbol (e.g., 'GBPJPY')
            traderlocker_api_key: API key for live data (optional)
            data: DataFrame with OHLCV data (optional, for backtesting)
        """
        self.symbol = symbol
        self.api_key = traderlocker_api_key
        self.data = data if data is not None else self.load_historical_data()
        self.current_step = 0
        self.account_balance = 10000.0
        self.peak_balance = 10000.0
        self.position = 0  # 0=flat, 1=long, 2=short
        self.position_size = 0.0
        self.entry_price = 0.0
        
        # State encoder
        self.state_encoder = StateEncoder()
        self.state_size = self.state_encoder.get_state_size()
        
        # History
        self.equity_history = [self.account_balance]
        self.trade_history = []

    def load_historical_data(self):
        """Load historical data from TraderLocker API."""
        if not self.api_key:
            raise ValueError("API key required for loading historical data")
        
        url = f"https://api.traderlocker.com/v1/historical/{self.symbol}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        params = {"timeframe": "1m", "limit": 10000}
        
        try:
            resp = requests.get(url, headers=headers, params=params)
            resp.raise_for_status()
            data = resp.json()
            df = pd.DataFrame(data.get('data', []))
            
            # Ensure required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required):
                raise ValueError(f"Missing required columns: {required}")
            
            # Add VWAP if not present
            if 'vwap' not in df.columns:
                df['vwap'] = (df['high'] + df['low'] + df['close']) / 3
            
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            # Return empty DataFrame for testing
            return pd.DataFrame()

    def reset(self):
        """Reset environment."""
        self.current_step = 0
        self.account_balance = 10000.0
        self.peak_balance = 10000.0
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.equity_history = [self.account_balance]
        self.trade_history = []
        self.state_encoder = StateEncoder()  # Reset encoder
        return self._get_state()

    def _get_state(self):
        """Build state vector."""
        if self.current_step >= len(self.data):
            return np.zeros(self.state_size)
        
        row = self.data.iloc[self.current_step]
        current_price = row['close']
        
        # Calculate unrealized PnL
        unrealized_pnl = 0.0
        if self.position == 1:  # Long
            unrealized_pnl = (current_price - self.entry_price) * self.position_size if self.entry_price > 0 else 0
        elif self.position == 2:  # Short
            unrealized_pnl = (self.entry_price - current_price) * self.position_size if self.entry_price > 0 else 0
        
        current_equity = self.account_balance + unrealized_pnl
        self.peak_balance = max(self.peak_balance, current_equity)
        
        # Encode state using StateEncoder
        state = self.state_encoder.encode_state(
            current_price=current_price,
            high=row['high'],
            low=row['low'],
            volume=row['volume'],
            vwap=row.get('vwap', current_price),
            position_size=self.position_size if self.position == 1 else (-self.position_size if self.position == 2 else 0.0),
            unrealized_pnl=unrealized_pnl,
            account_equity=current_equity,
            peak_equity=self.peak_balance,
            timestamp=row.get('timestamp')
        )
        
        return state

    def step(self, action, position_size):
        """
        Execute one step.
        
        Args:
            action: 0=flat, 1=long, 2=short
            position_size: fraction of equity to allocate [0, 1]
        
        Returns:
            next_state, reward, done, info
        """
        if self.current_step >= len(self.data) - 1:
            # Close position at end
            if self.position != 0:
                self._close_position(self.data.iloc[self.current_step]['close'])
            return self._get_state(), 0.0, True, {}
        
        previous_equity = self.account_balance
        
        row = self.data.iloc[self.current_step]
        current_price = row['close']
        
        # Execute action
        if action == 0:  # Go flat
            if self.position != 0:
                self._close_position(current_price)
        elif action == 1:  # Long
            if self.position != 1:
                if self.position == 2:
                    self._close_position(current_price)  # Close short first
                self._open_position(1, position_size, current_price)
        elif action == 2:  # Short
            if self.position != 2:
                if self.position == 1:
                    self._close_position(current_price)  # Close long first
                self._open_position(2, position_size, current_price)
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        current_equity = self.account_balance
        if self.position != 0:
            unrealized_pnl = self._calculate_unrealized_pnl(current_price)
            current_equity += unrealized_pnl
        
        self.peak_balance = max(self.peak_balance, current_equity)
        self.equity_history.append(current_equity)
        
        # Reward: equity change - drawdown penalty
        equity_change = current_equity - previous_equity
        equity_change_pct = equity_change / previous_equity if previous_equity > 0 else 0.0
        drawdown = (self.peak_balance - current_equity) / self.peak_balance if self.peak_balance > 0 else 0.0
        reward = equity_change_pct - 2.0 * drawdown  # Penalize drawdown
        
        done = self.current_step >= len(self.data) - 1
        
        info = {
            'equity': current_equity,
            'drawdown': drawdown,
            'position': self.position,
            'balance': self.account_balance
        }
        
        next_state = self._get_state()
        return next_state, reward, done, info

    def _open_position(self, direction, size_fraction, price):
        """Open position."""
        position_value = self.account_balance * size_fraction
        self.position_size = position_value / price
        self.position = direction
        self.entry_price = price

    def _close_position(self, price):
        """Close current position."""
        if self.position == 0:
            return
        
        # Calculate PnL
        if self.position == 1:  # Long
            pnl = (price - self.entry_price) * self.position_size
        else:  # Short
            pnl = (self.entry_price - price) * self.position_size
        
        # Apply commission (0.01%)
        commission = abs(self.position_size * price) * 0.0001
        pnl -= commission
        
        self.account_balance += pnl
        
        # Record trade
        self.trade_history.append({
            'direction': 'long' if self.position == 1 else 'short',
            'entry_price': self.entry_price,
            'exit_price': price,
            'size': self.position_size,
            'pnl': pnl
        })
        
        # Reset position
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0

    def _calculate_unrealized_pnl(self, current_price):
        """Calculate unrealized PnL."""
        if self.position == 1:  # Long
            return (current_price - self.entry_price) * self.position_size if self.entry_price > 0 else 0
        elif self.position == 2:  # Short
            return (self.entry_price - current_price) * self.position_size if self.entry_price > 0 else 0
        return 0.0

# -------------------
# TraderLocker Execution
# -------------------
def execute_trade_traderlocker(symbol, direction, size, api_key):
    """
    Send trade to TraderLocker live API
    
    Args:
        symbol: Trading symbol
        direction: 'long', 'short', or 'flat'
        size: position size (lot size or fraction)
        api_key: TraderLocker API key
    """
    url = f"https://api.traderlocker.com/v1/order"
    payload = {
        "symbol": symbol,
        "direction": direction,
        "size": size,
        "type": "market"
    }
    headers = {"Authorization": f"Bearer {api_key}"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error executing trade: {e}")
        return {"error": str(e)}

# -------------------
# Main Training Loop
# -------------------
def train_agent(symbol, api_key=None, data=None, episodes=1000):
    """Train RL agent."""
    # Create environment
    env = TradingEnv(symbol, api_key, data)
    state_size = env.state_size
    
    # Create agent
    agent = PPOAgent(state_size, ACTION_SIZE, HIDDEN_SIZE, LR)
    
    print(f"Training on {symbol}")
    print(f"State size: {state_size}")
    print(f"Data bars: {len(env.data)}")
    print("-" * 60)
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            direction, position_size, value, log_prob = agent.act(state)
            next_state, reward, done, info = env.step(direction, position_size)
            
            # Store transition
            agent.store((state, direction, reward, next_state, done, value, log_prob))
            
            state = next_state
            total_reward += reward
            steps += 1
            
            # Train periodically
            if steps % UPDATE_EVERY == 0 and len(agent.memory) >= BATCH_SIZE:
                loss = agent.train()
        
        # Final training on episode end
        if len(agent.memory) > 0:
            agent.train()
        
        # Print progress
        if (episode + 1) % 50 == 0:
            stats = env.get_statistics() if hasattr(env, 'get_statistics') else {}
            print(f"Episode {episode+1}/{episodes}: "
                  f"Reward={total_reward:.2f}, "
                  f"Balance=${env.account_balance:.2f}, "
                  f"Trades={len(env.trade_history)}")
    
    return agent, env

def get_statistics(env):
    """Get trading statistics."""
    if len(env.trade_history) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'total_return': 0.0
        }
    
    trades = pd.DataFrame(env.trade_history)
    winning_trades = (trades['pnl'] > 0).sum()
    total_trades = len(trades)
    win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
    
    total_return = (env.account_balance - 10000.0) / 10000.0
    
    return {
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': total_trades - winning_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'final_balance': env.account_balance
    }

# -------------------
# Main Entry Point
# -------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='TraderLocker RL Trading Bot')
    parser.add_argument('--symbol', type=str, default='GBPJPY', help='Trading symbol')
    parser.add_argument('--api-key', type=str, help='TraderLocker API key')
    parser.add_argument('--episodes', type=int, default=1000, help='Training episodes')
    parser.add_argument('--data', type=str, help='Path to CSV data file (for backtesting)')
    parser.add_argument('--train', action='store_true', help='Train agent')
    parser.add_argument('--live', action='store_true', help='Run live trading')
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("TRADERLOCKER_API_KEY")
    
    # Load data if provided
    data = None
    if args.data:
        data = pd.read_csv(args.data)
        print(f"Loaded {len(data)} bars from {args.data}")
    
    if args.train or not args.live:
        # Training mode
        print("=" * 60)
        print("TRAINING MODE")
        print("=" * 60)
        
        agent, env = train_agent(
            symbol=args.symbol,
            api_key=api_key,
            data=data,
            episodes=args.episodes
        )
        
        # Print statistics
        stats = get_statistics(env)
        print("\n" + "=" * 60)
        print("TRAINING RESULTS")
        print("=" * 60)
        print(f"Total Trades: {stats['total_trades']}")
        print(f"Win Rate: {stats['win_rate']*100:.2f}%")
        print(f"Total Return: {stats['total_return']*100:.2f}%")
        print(f"Final Balance: ${stats['final_balance']:.2f}")
        
        # Save model
        model_path = f"ppo_model_{args.symbol}_{datetime.now().strftime('%Y%m%d')}.pth"
        torch.save({
            'model_state_dict': agent.model.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'state_size': agent.state_size
        }, model_path)
        print(f"\nModel saved to: {model_path}")
    
    elif args.live:
        # Live trading mode
        print("=" * 60)
        print("LIVE TRADING MODE")
        print("=" * 60)
        print("Live trading not yet implemented. Use traderlocker_setup.py instead.")
        print("Or implement live trading loop here.")
