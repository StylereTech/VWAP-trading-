"""
Capital Allocation AI System
Not a trading bot - a capital-allocation AI that decides:
- When to trade
- How large to trade  
- When to exit
- When to stop trading
- How aggressively to compound

The market is the environment.
Your account balance is the reward.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from datetime import datetime
import math

# Import VWAP bands calculator
try:
    from .vwap_bands import VWAPBandsCalculator
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from vwap_bands import VWAPBandsCalculator


# ================================
# 1. STATE ENCODER
# ================================
class CapitalAllocationStateEncoder:
    """
    Encodes market state into regime, momentum, and risk features.
    The AI doesn't see charts - it sees regime, momentum, and risk.
    """
    
    def __init__(self):
        self.price_history = []
        self.volume_history = []
        self.vwap_history = []
        self.vwap_bands_calc = VWAPBandsCalculator()
        
    def encode_state(self,
                    current_price: float,
                    high: float,
                    low: float,
                    volume: float,
                    vwap: Optional[float] = None,
                    position_size: float = 0.0,
                    unrealized_pnl: float = 0.0,
                    account_equity: float = 10000.0,
                    peak_equity: float = 10000.0,
                    timestamp: Optional[pd.Timestamp] = None) -> np.ndarray:
        """
        Encode complete state: regime, momentum, risk.
        
        Returns state vector:
        - price returns (1m, 5m, 15m, 1h)
        - volatility (ATR, stdev)
        - trend (EMA fast - EMA slow)
        - RSI
        - distance from VWAP
        - time of day
        - open position size
        - unrealized PnL
        - drawdown
        """
        # Update history
        self.price_history.append(current_price)
        self.volume_history.append(volume)
        if vwap:
            self.vwap_history.append(vwap)
        
        # Keep history manageable
        max_history = 100
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.volume_history = self.volume_history[-max_history:]
            if len(self.vwap_history) > max_history:
                self.vwap_history = self.vwap_history[-max_history:]
        
        prices = np.array(self.price_history)
        
        # 1. Price returns (1m, 5m, 15m, 1h equivalent)
        returns_1m = (prices[-1] - prices[-2]) / prices[-2] if len(prices) > 1 else 0.0
        returns_5m = (prices[-1] - prices[-6]) / prices[-6] if len(prices) > 5 else 0.0
        returns_15m = (prices[-1] - prices[-16]) / prices[-16] if len(prices) > 15 else 0.0
        returns_1h = (prices[-1] - prices[-61]) / prices[-61] if len(prices) > 60 else 0.0
        
        # 2. Volatility (ATR, stdev)
        atr = self._calculate_atr(prices, high, low) if len(prices) > 14 else 0.0
        volatility = np.std(np.diff(prices[-20:]) / prices[-20:-1]) if len(prices) > 20 else 0.0
        normalized_atr = atr / current_price if current_price > 0 else 0.0
        
        # 3. Trend (EMA fast - EMA slow)
        ema_fast = self._calculate_ema(prices, 12) if len(prices) >= 12 else current_price
        ema_slow = self._calculate_ema(prices, 26) if len(prices) >= 26 else current_price
        trend = (ema_fast - ema_slow) / current_price if current_price > 0 else 0.0
        
        # 4. RSI
        rsi = self._calculate_rsi(prices, 14) if len(prices) > 14 else 50.0
        normalized_rsi = (rsi - 50) / 50.0  # Normalize to [-1, 1]
        
        # 5. VWAP bands and distance
        # Calculate VWAP bands
        bands = self.vwap_bands_calc.calculate(
            high=high,
            low=low,
            close=current_price,
            volume=volume,
            timestamp=timestamp if isinstance(timestamp, datetime) else None
        )
        vwap_val = bands['vwap']
        r1 = bands['r1']
        s1 = bands['s1']
        r2 = bands['r2']
        s2 = bands['s2']
        
        # VWAP distance (normalized)
        vwap_distance = (current_price - vwap_val) / vwap_val if vwap_val > 0 else 0.0
        
        # Distance to bands (normalized)
        distance_to_r1 = (r1 - current_price) / current_price if current_price > 0 else 0.0
        distance_to_s1 = (current_price - s1) / current_price if current_price > 0 else 0.0
        distance_to_r2 = (r2 - current_price) / current_price if current_price > 0 else 0.0
        distance_to_s2 = (current_price - s2) / current_price if current_price > 0 else 0.0
        
        # Band touch signals
        touches_s1 = 1.0 if low <= s1 else 0.0
        touches_r1 = 1.0 if high >= r1 else 0.0
        touches_s2 = 1.0 if low <= s2 else 0.0
        touches_r2 = 1.0 if high >= r2 else 0.0
        
        # 6. Time of day
        if timestamp:
            hour = timestamp.hour / 24.0
            day_of_week = timestamp.dayofweek / 7.0
        else:
            hour = 0.0
            day_of_week = 0.0
        
        # 7. Open position size (normalized)
        normalized_position = position_size
        
        # 8. Unrealized PnL (as % of equity)
        unrealized_pnl_pct = unrealized_pnl / account_equity if account_equity > 0 else 0.0
        
        # 9. Drawdown
        drawdown = (peak_equity - account_equity) / peak_equity if peak_equity > 0 else 0.0
        
        # Combine into state vector (expanded with VWAP bands)
        state = np.array([
            returns_1m,
            returns_5m,
            returns_15m,
            returns_1h,
            normalized_atr,
            volatility,
            trend,
            normalized_rsi,
            vwap_distance,          # Distance from VWAP
            distance_to_r1,        # Distance to R1 band
            distance_to_s1,        # Distance to S1 band
            distance_to_r2,        # Distance to R2 band
            distance_to_s2,        # Distance to S2 band
            touches_r1,            # Touching R1 band
            touches_s1,            # Touching S1 band
            touches_r2,            # Touching R2 band
            touches_s2,            # Touching S2 band
            hour,
            day_of_week,
            normalized_position,
            unrealized_pnl_pct,
            drawdown
        ], dtype=np.float32)
        
        return state
    
    def _calculate_atr(self, prices: np.ndarray, high: float, low: float, period: int = 14) -> float:
        """Calculate Average True Range."""
        if len(prices) < period + 1:
            return abs(high - low)
        
        tr_list = []
        for i in range(len(prices) - period, len(prices)):
            if i > 0:
                tr1 = abs(high - low)
                tr2 = abs(high - prices[i-1])
                tr3 = abs(low - prices[i-1])
                tr = max(tr1, tr2, tr3)
                tr_list.append(tr)
        
        return np.mean(tr_list) if tr_list else abs(high - low)
    
    def _calculate_ema(self, prices: np.ndarray, period: int) -> float:
        """Calculate Exponential Moving Average."""
        if len(prices) < period:
            return prices[-1]
        
        alpha = 2.0 / (period + 1)
        ema = prices[-period]
        for price in prices[-period+1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema
    
    def _calculate_rsi(self, prices: np.ndarray, period: int) -> float:
        """Calculate Relative Strength Index."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains) if len(gains) > 0 else 0.0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.0
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def get_state_size(self) -> int:
        """Return state vector size."""
        return 22  # Expanded with VWAP bands


# ================================
# 2. RL POLICY NETWORK (PPO)
# ================================
class CapitalAllocationPolicy(nn.Module):
    """
    PPO Policy Network for capital allocation.
    Outputs: direction (long/short/flat) + position_size [0, max_risk]
    """
    
    def __init__(self, state_size: int, hidden_size: int = 128):
        super(CapitalAllocationPolicy, self).__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU()
        )
        
        # Actor: direction (3 actions: flat, long, short)
        self.actor_direction = nn.Linear(hidden_size // 2, 3)
        
        # Actor: position size [0, max_risk] - this is the key to compounding
        self.actor_position_size = nn.Sequential(
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Outputs [0, 1], scaled by max_risk
        )
        
        # Critic: state value estimate
        self.critic = nn.Linear(hidden_size // 2, 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            direction_logits: Logits for direction action
            position_size: Position size in [0, 1]
            state_value: Estimated state value
        """
        x = self.shared(state)
        
        direction_logits = self.actor_direction(x)
        position_size = self.actor_position_size(x)
        state_value = self.critic(x)
        
        return direction_logits, position_size, state_value


class CapitalAllocationAgent:
    """
    PPO Agent for capital allocation.
    Learns when to bet hard and when to go flat.
    """
    
    def __init__(self,
                 state_size: int,
                 max_risk: float = 0.5,  # Max 50% of equity per trade
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon: float = 0.2):
        self.state_size = state_size
        self.max_risk = max_risk
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.model = CapitalAllocationPolicy(state_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float, float]:
        """
        Select action: (direction, position_size, value, log_prob)
        
        direction: 0=flat, 1=long, 2=short
        position_size: Fraction of equity [0, max_risk]
        """
        self.model.eval()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            direction_logits, position_size_raw, value = self.model(state_tensor)
            
            # Sample direction
            if deterministic:
                direction = torch.argmax(direction_logits, dim=-1).item()
            else:
                direction_probs = F.softmax(direction_logits, dim=-1)
                direction = torch.multinomial(direction_probs, 1).item()
            
            # Scale position size by max_risk
            pos_size = position_size_raw.item() * self.max_risk
            
            # Get log probability
            log_prob = F.log_softmax(direction_logits, dim=-1)[0, direction].item()
        
        return direction, pos_size, value.item(), log_prob
    
    def store_transition(self, state, action, reward, value, log_prob, done):
        """Store transition for training."""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value)
        self.memory['log_probs'].append(log_prob)
        self.memory['dones'].append(done)
    
    def train(self, epochs: int = 4):
        """Train PPO agent."""
        if len(self.memory['states']) < 10:
            return 0.0
        
        self.model.train()
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.memory['states'])).to(self.device)
        actions = self.memory['actions']
        old_log_probs = torch.FloatTensor(self.memory['log_probs']).to(self.device)
        old_values = torch.FloatTensor(self.memory['values']).to(self.device)
        
        # Compute returns and advantages
        returns, advantages = self._compute_gae(
            self.memory['rewards'],
            self.memory['values'],
            self.memory['dones']
        )
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        total_loss = 0.0
        for epoch in range(epochs):
            # Forward pass
            direction_logits, position_sizes, values = self.model(states)
            
            # Direction probabilities
            direction_tensor = torch.LongTensor([a[0] for a in actions]).to(self.device)
            new_log_probs = F.log_softmax(direction_logits, dim=-1)
            new_log_probs = new_log_probs.gather(1, direction_tensor.unsqueeze(1)).squeeze(1)
            
            # PPO clipped objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Entropy bonus
            entropy = -(F.softmax(direction_logits, dim=-1) * new_log_probs.unsqueeze(1)).sum(dim=1).mean()
            
            # Total loss
            loss = actor_loss + 0.5 * value_loss - 0.01 * entropy
            
            # Update
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Clear memory
        self.clear_memory()
        
        return total_loss / epochs
    
    def _compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        returns = []
        advantages = []
        gae = 0
        
        for step in reversed(range(len(rewards))):
            if dones[step]:
                next_value = 0
            else:
                next_value = values[step]
            
            delta = rewards[step] + self.gamma * next_value - values[step]
            gae = delta + self.gamma * 0.95 * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
        
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def clear_memory(self):
        """Clear stored transitions."""
        for key in self.memory:
            self.memory[key] = []
    
    def save(self, filepath: str):
        """Save model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'max_risk': self.max_risk
        }, filepath)
    
    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'max_risk' in checkpoint:
            self.max_risk = checkpoint['max_risk']


# ================================
# 3. POSITION SIZING ENGINE
# ================================
class PositionSizingEngine:
    """
    Position sizing engine.
    Scales RL output by max_risk and applies risk adjustments.
    """
    
    def __init__(self, max_risk: float = 0.5):
        self.max_risk = max_risk
    
    def calculate_size(self,
                      rl_output: float,
                      current_equity: float,
                      drawdown: float,
                      volatility: float) -> float:
        """
        Calculate position size from RL output.
        
        Args:
            rl_output: Position size from RL agent [0, 1]
            current_equity: Current account equity
            drawdown: Current drawdown [0, 1]
            volatility: Current market volatility
        
        Returns:
            Position size in currency
        """
        # Base size from RL
        base_size_fraction = rl_output * self.max_risk
        
        # Reduce size if drawdown is high
        if drawdown > 0.10:  # If drawdown > 10%
            reduction = 1.0 - (drawdown / 0.20)  # Scale down as drawdown increases
            base_size_fraction *= max(0.1, reduction)  # At least 10% of base
        
        # Reduce size if volatility is extreme
        if volatility > 0.02:  # If volatility > 2%
            reduction = 0.5  # Cut position size in half
            base_size_fraction *= reduction
        
        # Calculate position value
        position_value = current_equity * base_size_fraction
        
        return max(0.0, position_value)


# ================================
# 4. EXECUTION ENGINE
# ================================
class ExecutionEngine:
    """
    Execution engine interface.
    Can be connected to any broker API.
    """
    
    def __init__(self, broker_type: str = "traderlocker", api_key: Optional[str] = None):
        self.broker_type = broker_type
        self.api_key = api_key
    
    def execute_order(self,
                     symbol: str,
                     direction: int,  # 0=flat, 1=long, 2=short
                     size: float,
                     current_price: float,
                     stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None) -> Dict:
        """
        Execute order via broker API.
        
        Returns:
            Execution result
        """
        if direction == 0:
            # Close positions
            return self._close_positions(symbol)
        
        direction_str = "long" if direction == 1 else "short"
        
        if self.broker_type == "traderlocker":
            return self._execute_traderlocker(symbol, direction_str, size, stop_loss, take_profit)
        elif self.broker_type == "mock":
            return self._execute_mock(symbol, direction_str, size, current_price)
        else:
            raise ValueError(f"Unknown broker type: {self.broker_type}")
    
    def _execute_traderlocker(self, symbol, direction, size, stop_loss, take_profit):
        """Execute via TraderLocker API."""
        import requests
        
        url = f"https://api.traderlocker.com/v1/order"
        payload = {
            "symbol": symbol,
            "direction": direction,
            "size": size,
            "type": "market"
        }
        
        if stop_loss:
            payload["stop_loss"] = stop_loss
        if take_profit:
            payload["take_profit"] = take_profit
        
        headers = {"Authorization": f"Bearer {self.api_key}"}
        
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"error": str(e)}
    
    def _execute_mock(self, symbol, direction, size, current_price):
        """Mock execution for testing."""
        return {
            "status": "filled",
            "symbol": symbol,
            "direction": direction,
            "size": size,
            "price": current_price
        }
    
    def _close_positions(self, symbol):
        """Close all positions for symbol."""
        if self.broker_type == "traderlocker":
            import requests
            url = f"https://api.traderlocker.com/v1/positions/close-all"
            headers = {"Authorization": f"Bearer {self.api_key}"}
            payload = {"symbol": symbol}
            try:
                response = requests.post(url, json=payload, headers=headers)
                return response.json()
            except Exception as e:
                return {"error": str(e)}
        return {"status": "closed"}


# ================================
# 5. RISK GOVERNOR
# ================================
class RiskGovernor:
    """
    Risk governor - enforces hard limits.
    Hard max-drawdown kill switch.
    """
    
    def __init__(self,
                 max_drawdown: float = 0.20,  # 20% max drawdown
                 daily_loss_limit: float = 0.05,  # 5% daily loss
                 consecutive_loss_limit: int = 5):
        self.max_drawdown = max_drawdown
        self.daily_loss_limit = daily_loss_limit
        self.consecutive_loss_limit = consecutive_loss_limit
        
        self.initial_equity = None
        self.peak_equity = None
        self.daily_start_equity = None
        self.consecutive_losses = 0
        self.trading_halted = False
        self.halt_reason = ""
    
    def initialize(self, initial_equity: float):
        """Initialize risk governor."""
        self.initial_equity = initial_equity
        self.peak_equity = initial_equity
        self.daily_start_equity = initial_equity
        self.consecutive_losses = 0
        self.trading_halted = False
        self.halt_reason = ""
    
    def check_limits(self, current_equity: float) -> Tuple[bool, str]:
        """
        Check if trading should be halted.
        
        Returns:
            (allowed, reason)
        """
        if self.peak_equity is None:
            self.peak_equity = current_equity
        
        self.peak_equity = max(self.peak_equity, current_equity)
        
        # Check max drawdown
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if drawdown > self.max_drawdown:
            self.trading_halted = True
            self.halt_reason = f"Max drawdown exceeded: {drawdown*100:.1f}%"
            return False, self.halt_reason
        
        # Check daily loss
        if self.daily_start_equity:
            daily_loss = (self.daily_start_equity - current_equity) / self.daily_start_equity
            if daily_loss > self.daily_loss_limit:
                self.trading_halted = True
                self.halt_reason = f"Daily loss limit exceeded: {daily_loss*100:.1f}%"
                return False, self.halt_reason
        
        # Check consecutive losses
        if self.consecutive_losses >= self.consecutive_loss_limit:
            self.trading_halted = True
            self.halt_reason = f"Consecutive loss limit: {self.consecutive_losses} losses"
            return False, self.halt_reason
        
        return True, ""
    
    def record_trade_result(self, pnl: float):
        """Record trade result."""
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
    
    def reset_daily(self, current_equity: float):
        """Reset daily tracking."""
        self.daily_start_equity = current_equity


# ================================
# MAIN SYSTEM
# ================================
class CapitalAllocationAI:
    """
    Complete Capital Allocation AI System.
    
    Five engines:
    1. State Encoder
    2. RL Policy Network
    3. Position Sizing Engine
    4. Execution Engine
    5. Risk Governor
    """
    
    def __init__(self,
                 symbol: str,
                 initial_equity: float = 10000.0,
                 max_risk: float = 0.5,
                 broker_type: str = "mock",
                 api_key: Optional[str] = None):
        self.symbol = symbol
        self.initial_equity = initial_equity
        self.account_equity = initial_equity
        self.peak_equity = initial_equity
        
        # Initialize engines
        self.state_encoder = CapitalAllocationStateEncoder()
        state_size = self.state_encoder.get_state_size()
        
        self.agent = CapitalAllocationAgent(state_size, max_risk=max_risk)
        self.position_sizer = PositionSizingEngine(max_risk=max_risk)
        self.executor = ExecutionEngine(broker_type, api_key)
        self.risk_gov = RiskGovernor()
        self.risk_gov.initialize(initial_equity)
        
        # Position tracking
        self.position_direction = 0  # 0=flat, 1=long, 2=short
        self.position_size = 0.0
        self.entry_price = 0.0
        
        # History
        self.equity_history = [initial_equity]
        self.trade_history = []
    
    def update_market_data(self,
                          current_price: float,
                          high: float,
                          low: float,
                          volume: float,
                          vwap: Optional[float] = None,
                          timestamp: Optional[pd.Timestamp] = None) -> Dict:
        """
        Update with new market data and make capital allocation decision.
        
        Returns:
            Decision and execution result
        """
        # Calculate unrealized PnL
        unrealized_pnl = self._calculate_unrealized_pnl(current_price)
        current_equity = self.account_equity + unrealized_pnl
        self.peak_equity = max(self.peak_equity, current_equity)
        
        # Check risk limits
        allowed, reason = self.risk_gov.check_limits(current_equity)
        if not allowed:
            # Halt trading
            if self.position_direction != 0:
                self._close_position(current_price)
            return {
                "action": "halted",
                "reason": reason,
                "equity": current_equity
            }
        
        # Encode state
        state = self.state_encoder.encode_state(
            current_price=current_price,
            high=high,
            low=low,
            volume=volume,
            vwap=vwap,
            position_size=self.position_size if self.position_direction == 1 else (-self.position_size if self.position_direction == 2 else 0.0),
            unrealized_pnl=unrealized_pnl,
            account_equity=current_equity,
            peak_equity=self.peak_equity,
            timestamp=timestamp
        )
        
        # Get action from RL agent
        direction, rl_position_size, value, log_prob = self.agent.act(state)
        
        # Calculate position size
        drawdown = (self.peak_equity - current_equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        volatility = abs(state[4])  # ATR from state
        position_value = self.position_sizer.calculate_size(
            rl_position_size / self.agent.max_risk,  # Normalize back to [0, 1]
            current_equity,
            drawdown,
            volatility
        )
        position_size = position_value / current_price if current_price > 0 else 0.0
        
        # Execute action
        if direction == 0:  # Go flat
            if self.position_direction != 0:
                result = self._close_position(current_price)
            else:
                result = {"action": "hold"}
        elif direction == 1:  # Long
            if self.position_direction != 1:
                if self.position_direction == 2:
                    self._close_position(current_price)  # Close short first
                result = self._open_position(1, position_size, current_price)
            else:
                result = {"action": "hold_long"}
        else:  # Short
            if self.position_direction != 2:
                if self.position_direction == 1:
                    self._close_position(current_price)  # Close long first
                result = self._open_position(2, position_size, current_price)
            else:
                result = {"action": "hold_short"}
        
        # Update equity
        self.equity_history.append(current_equity)
        
        return {
            "action": ["flat", "long", "short"][direction],
            "position_size": position_size,
            "equity": current_equity,
            "drawdown": drawdown,
            "result": result
        }
    
    def _open_position(self, direction: int, size: float, price: float) -> Dict:
        """Open new position."""
        self.position_direction = direction
        self.position_size = size
        self.entry_price = price
        
        # Calculate stop loss and take profit
        atr = abs(self.state_encoder.price_history[-1] - self.state_encoder.price_history[-2]) if len(self.state_encoder.price_history) > 1 else price * 0.01
        
        if direction == 1:  # Long
            stop_loss = price - atr * 1.5
            take_profit = price + atr * 2.0
        else:  # Short
            stop_loss = price + atr * 1.5
            take_profit = price - atr * 2.0
        
        # Execute via broker
        direction_str = "long" if direction == 1 else "short"
        result = self.executor.execute_order(
            self.symbol,
            direction,
            size,
            price,
            stop_loss,
            take_profit
        )
        
        return result
    
    def _close_position(self, price: float) -> Dict:
        """Close current position."""
        if self.position_direction == 0:
            return {"action": "no_position"}
        
        # Calculate PnL
        if self.position_direction == 1:  # Long
            pnl = (price - self.entry_price) * self.position_size
        else:  # Short
            pnl = (self.entry_price - price) * self.position_size
        
        # Apply commission
        commission = abs(self.position_size * price) * 0.0001
        pnl -= commission
        
        # Update equity
        self.account_equity += pnl
        
        # Record trade
        self.trade_history.append({
            "direction": "long" if self.position_direction == 1 else "short",
            "entry_price": self.entry_price,
            "exit_price": price,
            "size": self.position_size,
            "pnl": pnl
        })
        
        # Record in risk governor
        self.risk_gov.record_trade_result(pnl)
        
        # Execute close via broker
        result = self.executor.execute_order(self.symbol, 0, 0, price)
        
        # Reset position
        self.position_direction = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        
        return result
    
    def _calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized PnL."""
        if self.position_direction == 0:
            return 0.0
        
        if self.position_direction == 1:  # Long
            return (current_price - self.entry_price) * self.position_size
        else:  # Short
            return (self.entry_price - current_price) * self.position_size
    
    def get_statistics(self) -> Dict:
        """Get system statistics."""
        if len(self.trade_history) == 0:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_return": 0.0,
                "max_drawdown": 0.0
            }
        
        trades = pd.DataFrame(self.trade_history)
        winning_trades = (trades['pnl'] > 0).sum()
        total_trades = len(trades)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.0
        
        total_return = (self.account_equity - self.initial_equity) / self.initial_equity
        
        # Max drawdown
        equity_array = np.array(self.equity_history)
        running_max = np.maximum.accumulate(equity_array)
        drawdowns = (equity_array - running_max) / running_max
        max_drawdown = abs(np.min(drawdowns))
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": total_trades - winning_trades,
            "win_rate": win_rate,
            "total_return": total_return,
            "final_equity": self.account_equity,
            "max_drawdown": max_drawdown,
            "equity_history": self.equity_history
        }
