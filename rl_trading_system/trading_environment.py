"""
Trading Environment for RL Agent
Simulates trading environment with proper reward shaping.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, List
from .state_encoder import StateEncoder
import math


class TradingEnvironment:
    """
    Trading environment for RL agent.
    Handles position management, PnL calculation, and reward shaping.
    """
    
    def __init__(self,
                 data: pd.DataFrame,
                 initial_equity: float = 10000.0,
                 commission: float = 0.0001,  # 0.01% commission
                 slippage: float = 0.0001,    # 0.01% slippage
                 max_position_size: float = 0.5,  # Max 50% of equity per trade
                 drawdown_penalty: float = 2.0,  # Penalty multiplier for drawdown
                 volatility_penalty: float = 0.5):  # Penalty for equity volatility
        """
        Args:
            data: DataFrame with columns: open, high, low, close, volume, vwap (optional)
            initial_equity: Starting capital
            commission: Commission rate per trade
            slippage: Slippage rate
            max_position_size: Maximum position size as fraction of equity
            drawdown_penalty: Multiplier for drawdown penalty in reward
            volatility_penalty: Multiplier for equity volatility penalty
        """
        self.data = data.reset_index(drop=True)
        self.initial_equity = initial_equity
        self.commission = commission
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.drawdown_penalty = drawdown_penalty
        self.volatility_penalty = volatility_penalty
        
        # State encoder
        self.state_encoder = StateEncoder()
        self.state_size = self.state_encoder.get_state_size()
        
        # Environment state
        self.current_step = 0
        self.account_equity = initial_equity
        self.peak_equity = initial_equity
        self.position_size = 0.0  # Positive = long, Negative = short, 0 = flat
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        
        # History tracking
        self.equity_history = [initial_equity]
        self.trade_history = []
        self.reward_history = []
        
        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in self.data.columns for col in required_cols):
            raise ValueError(f"Data must contain columns: {required_cols}")
        
        if 'vwap' not in self.data.columns:
            # Calculate simple VWAP if not provided
            self.data['vwap'] = (self.data['high'] + self.data['low'] + self.data['close']) / 3
    
    def reset(self) -> np.ndarray:
        """Reset environment to initial state."""
        self.current_step = 0
        self.account_equity = self.initial_equity
        self.peak_equity = self.initial_equity
        self.position_size = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        self.equity_history = [self.initial_equity]
        self.trade_history = []
        self.reward_history = []
        
        # Reset state encoder
        self.state_encoder = StateEncoder()
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Get current state vector."""
        if self.current_step >= len(self.data):
            return np.zeros(self.state_size)
        
        row = self.data.iloc[self.current_step]
        timestamp = row.get('timestamp') if 'timestamp' in row else None
        
        state = self.state_encoder.encode_state(
            current_price=row['close'],
            high=row['high'],
            low=row['low'],
            volume=row['volume'],
            vwap=row.get('vwap'),
            position_size=self.position_size,
            unrealized_pnl=self.unrealized_pnl,
            account_equity=self.account_equity,
            peak_equity=self.peak_equity,
            timestamp=timestamp
        )
        
        return state
    
    def _update_unrealized_pnl(self):
        """Update unrealized PnL based on current price."""
        if self.position_size == 0:
            self.unrealized_pnl = 0.0
            return
        
        current_price = self.data.iloc[self.current_step]['close']
        
        if self.position_size > 0:  # Long position
            self.unrealized_pnl = (current_price - self.entry_price) * abs(self.position_size)
        else:  # Short position
            self.unrealized_pnl = (self.entry_price - current_price) * abs(self.position_size)
    
    def _close_position(self, current_price: float) -> float:
        """Close current position and return realized PnL."""
        if self.position_size == 0:
            return 0.0
        
        # Calculate PnL
        if self.position_size > 0:  # Long
            pnl = (current_price - self.entry_price) * abs(self.position_size)
        else:  # Short
            pnl = (self.entry_price - current_price) * abs(self.position_size)
        
        # Apply slippage and commission
        trade_value = abs(self.position_size) * current_price
        slippage_cost = trade_value * self.slippage
        commission_cost = trade_value * self.commission
        
        realized_pnl = pnl - slippage_cost - commission_cost
        
        # Update equity
        self.account_equity += realized_pnl
        self.peak_equity = max(self.peak_equity, self.account_equity)
        
        # Record trade
        self.trade_history.append({
            'entry_price': self.entry_price,
            'exit_price': current_price,
            'position_size': self.position_size,
            'pnl': realized_pnl,
            'step': self.current_step
        })
        
        # Reset position
        self.position_size = 0.0
        self.entry_price = 0.0
        self.unrealized_pnl = 0.0
        
        return realized_pnl
    
    def _open_position(self, direction: int, position_size_fraction: float, current_price: float):
        """
        Open new position.
        
        Args:
            direction: 0=flat, 1=long, 2=short
            position_size_fraction: Position size as fraction of equity [0, 1]
            current_price: Current market price
        """
        if direction == 0:  # Flat
            return
        
        # Close existing position if any
        if self.position_size != 0:
            self._close_position(current_price)
        
        # Calculate position size in currency
        position_value = self.account_equity * min(position_size_fraction, self.max_position_size)
        position_size = position_value / current_price
        
        # Apply slippage and commission
        slippage_cost = position_value * self.slippage
        commission_cost = position_value * self.commission
        self.account_equity -= (slippage_cost + commission_cost)
        
        # Set position
        if direction == 1:  # Long
            self.position_size = position_size
        else:  # Short
            self.position_size = -position_size
        
        self.entry_price = current_price * (1 + self.slippage)  # Account for slippage
        self._update_unrealized_pnl()
    
    def _calculate_reward(self, previous_equity: float, current_equity: float) -> float:
        """
        Calculate reward with proper shaping.
        
        Reward = equity_change - drawdown_penalty - volatility_penalty
        
        This encourages:
        - Equity growth
        - Low drawdown
        - Smooth equity curve
        """
        # Equity change
        equity_change = current_equity - previous_equity
        equity_change_pct = equity_change / previous_equity if previous_equity > 0 else 0.0
        
        # Drawdown penalty
        drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        drawdown_penalty = self.drawdown_penalty * drawdown
        
        # Volatility penalty (penalize large equity swings)
        if len(self.equity_history) >= 10:
            recent_equity = self.equity_history[-10:]
            if len(recent_equity) > 1:
                equity_returns = np.diff(recent_equity) / recent_equity[:-1]
                equity_volatility = np.std(equity_returns) if len(equity_returns) > 0 else 0.0
                volatility_penalty = self.volatility_penalty * equity_volatility
            else:
                volatility_penalty = 0.0
        else:
            volatility_penalty = 0.0
        
        # Total reward
        reward = equity_change_pct - drawdown_penalty - volatility_penalty
        
        return reward
    
    def step(self, action: Tuple[int, float]) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in environment.
        
        Args:
            action: (direction, position_size_fraction)
                direction: 0=flat, 1=long, 2=short
                position_size_fraction: Position size [0, 1]
        
        Returns:
            next_state: Next state vector
            reward: Reward for this step
            done: Whether episode is finished
            info: Additional information
        """
        if self.current_step >= len(self.data) - 1:
            # Close any open positions at end
            if self.position_size != 0:
                current_price = self.data.iloc[self.current_step]['close']
                self._close_position(current_price)
            
            return self._get_state(), 0.0, True, {'episode_end': True}
        
        previous_equity = self.account_equity
        
        # Get current market data
        row = self.data.iloc[self.current_step]
        current_price = row['close']
        
        # Execute action
        direction, position_size_fraction = action
        
        if direction == 0:  # Go flat
            if self.position_size != 0:
                self._close_position(current_price)
        else:
            # Check if we need to change position
            if (direction == 1 and self.position_size <= 0) or \
               (direction == 2 and self.position_size >= 0):
                self._open_position(direction, position_size_fraction, current_price)
        
        # Update unrealized PnL
        self._update_unrealized_pnl()
        
        # Move to next step
        self.current_step += 1
        
        # Update equity (including unrealized PnL)
        current_equity = self.account_equity + self.unrealized_pnl
        self.peak_equity = max(self.peak_equity, current_equity)
        self.equity_history.append(current_equity)
        
        # Calculate reward
        reward = self._calculate_reward(previous_equity, current_equity)
        self.reward_history.append(reward)
        
        # Check if done
        done = self.current_step >= len(self.data) - 1
        
        # Additional info
        info = {
            'equity': current_equity,
            'drawdown': (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0,
            'position_size': self.position_size,
            'unrealized_pnl': self.unrealized_pnl
        }
        
        next_state = self._get_state()
        return next_state, reward, done, info
    
    def get_statistics(self) -> Dict:
        """Get trading statistics."""
        if len(self.trade_history) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0
            }
        
        trades = pd.DataFrame(self.trade_history)
        winning_trades = (trades['pnl'] > 0).sum()
        total_trades = len(trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        total_return = (self.account_equity - self.initial_equity) / self.initial_equity
        
        # Calculate max drawdown
        equity_array = np.array(self.equity_history)
        running_max = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))
        
        # Calculate Sharpe ratio
        if len(self.reward_history) > 1:
            returns = np.diff(self.equity_history) / self.equity_history[:-1]
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)  # Annualized
        else:
            sharpe = 0.0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'final_equity': self.account_equity,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'avg_trade_pnl': trades['pnl'].mean() if len(trades) > 0 else 0.0
        }


if __name__ == "__main__":
    # Test environment
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=100, freq='5min')
    data = pd.DataFrame({
        'timestamp': dates,
        'open': np.random.randn(100).cumsum() + 100,
        'high': np.random.randn(100).cumsum() + 101,
        'low': np.random.randn(100).cumsum() + 99,
        'close': np.random.randn(100).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 100),
        'vwap': np.random.randn(100).cumsum() + 100
    })
    
    env = TradingEnvironment(data, initial_equity=10000.0)
    state = env.reset()
    
    # Test a few steps
    for _ in range(10):
        action = (1, 0.1)  # Long, 10% position
        next_state, reward, done, info = env.step(action)
        print(f"Step: {env.current_step}, Equity: {info['equity']:.2f}, Reward: {reward:.4f}")
        if done:
            break
    
    stats = env.get_statistics()
    print(f"\nStatistics: {stats}")

