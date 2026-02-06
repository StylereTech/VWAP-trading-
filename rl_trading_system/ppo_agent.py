"""
PPO (Proximal Policy Optimization) Agent for Trading
Implements actor-critic architecture with proper PPO training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Tuple, List, Dict
import random


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    Actor outputs action probabilities (direction + position size).
    Critic outputs state value estimate.
    """
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        super(ActorCritic, self).__init__()
        
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
        
        # Actor head: outputs action logits
        # Action space: [direction (3: long/flat/short), position_size (continuous 0-1)]
        self.actor_direction = nn.Linear(hidden_size // 2, 3)  # long, flat, short
        self.actor_position_size = nn.Sequential(
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid()  # Position size in [0, 1]
        )
        
        # Critic head: outputs state value
        self.critic = nn.Linear(hidden_size // 2, 1)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            direction_logits: Logits for direction action (long/flat/short)
            position_size: Position size in [0, 1]
            state_value: Estimated state value
        """
        x = self.shared(state)
        
        direction_logits = self.actor_direction(x)
        position_size = self.actor_position_size(x)
        state_value = self.critic(x)
        
        return direction_logits, position_size, state_value


class PPOAgent:
    """
    PPO Agent for trading.
    Uses clipped surrogate objective and value function loss.
    """
    
    def __init__(self,
                 state_size: int,
                 action_size: int = 2,  # direction + position_size
                 hidden_size: int = 128,
                 lr: float = 3e-4,
                 gamma: float = 0.99,
                 epsilon: float = 0.2,  # PPO clip parameter
                 value_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5):
        """
        Args:
            state_size: Size of state vector
            action_size: Size of action space (not used directly, kept for compatibility)
            hidden_size: Hidden layer size
            lr: Learning rate
            gamma: Discount factor
            epsilon: PPO clip parameter
            value_coef: Value function loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Gradient clipping threshold
        """
        self.state_size = state_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        
        # Initialize networks
        self.model = ActorCritic(state_size, action_size, hidden_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        # Memory for PPO updates
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
    def act(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float]:
        """
        Select action given state.
        
        Args:
            state: State vector
            deterministic: If True, select best action. If False, sample from policy.
        
        Returns:
            direction: 0=flat, 1=long, 2=short
            position_size: Position size fraction [0, 1]
        """
        self.model.eval()
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            direction_logits, position_size, value = self.model(state_tensor)
            
            # Sample direction
            if deterministic:
                direction = torch.argmax(direction_logits, dim=-1).item()
            else:
                direction_probs = F.softmax(direction_logits, dim=-1)
                direction = torch.multinomial(direction_probs, 1).item()
            
            # Get position size
            pos_size = position_size.item()
            
            # Get log probability for training
            log_prob = F.log_softmax(direction_logits, dim=-1)[0, direction]
        
        return direction, pos_size, value.item(), log_prob.item()
    
    def store_transition(self, state: np.ndarray, action: Tuple[int, float],
                        reward: float, value: float, log_prob: float, done: bool):
        """Store transition in memory."""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value)
        self.memory['log_probs'].append(log_prob)
        self.memory['dones'].append(done)
    
    def compute_returns(self, rewards: List[float], values: List[float],
                       dones: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute discounted returns and advantages using GAE (Generalized Advantage Estimation).
        
        Returns:
            returns: Discounted returns
            advantages: Advantages (returns - values)
        """
        returns = []
        advantages = []
        gae = 0
        
        # Compute returns backwards
        next_value = 0
        for step in reversed(range(len(rewards))):
            if dones[step]:
                next_value = 0
            
            delta = rewards[step] + self.gamma * next_value - values[step]
            gae = delta + self.gamma * 0.95 * gae  # GAE lambda = 0.95
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step])
            next_value = values[step]
        
        returns = np.array(returns)
        advantages = np.array(advantages)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def train(self, epochs: int = 4):
        """
        Train PPO agent on stored transitions.
        
        Args:
            epochs: Number of training epochs over the batch
        """
        if len(self.memory['states']) < 10:
            return
        
        self.model.train()
        
        # Convert to tensors
        states = torch.FloatTensor(np.array(self.memory['states'])).to(self.device)
        actions = self.memory['actions']
        old_log_probs = torch.FloatTensor(self.memory['log_probs']).to(self.device)
        old_values = torch.FloatTensor(self.memory['values']).to(self.device)
        
        # Compute returns and advantages
        returns, advantages = self.compute_returns(
            self.memory['rewards'],
            self.memory['values'],
            self.memory['dones']
        )
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Training loop
        total_loss = 0
        for epoch in range(epochs):
            # Forward pass
            direction_logits, position_sizes, values = self.model(states)
            
            # Compute action probabilities
            action_probs = F.softmax(direction_logits, dim=-1)
            directions = [a[0] for a in actions]
            direction_tensor = torch.LongTensor(directions).to(self.device)
            
            # New log probabilities
            new_log_probs = F.log_softmax(direction_logits, dim=-1)
            new_log_probs = new_log_probs.gather(1, direction_tensor.unsqueeze(1)).squeeze(1)
            
            # PPO clipped surrogate objective
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Value function loss
            value_loss = F.mse_loss(values.squeeze(), returns)
            
            # Entropy bonus (encourage exploration)
            entropy = -(action_probs * F.log_softmax(direction_logits, dim=-1)).sum(dim=1).mean()
            
            # Total loss
            loss = actor_loss + self.value_coef * value_loss - self.entropy_coef * entropy
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        # Clear memory
        self.clear_memory()
        
        return total_loss / epochs
    
    def clear_memory(self):
        """Clear stored transitions."""
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
    
    def save(self, filepath: str):
        """Save model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str):
        """Load model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


if __name__ == "__main__":
    # Test PPO agent
    state_size = 15
    agent = PPOAgent(state_size)
    
    # Test action selection
    test_state = np.random.randn(state_size)
    direction, pos_size, value, log_prob = agent.act(test_state)
    print(f"Direction: {direction}, Position Size: {pos_size:.3f}, Value: {value:.3f}")
    
    # Test training
    for _ in range(20):
        direction, pos_size, value, log_prob = agent.act(test_state)
        reward = np.random.randn()
        agent.store_transition(test_state, (direction, pos_size), reward, value, log_prob, False)
    
    loss = agent.train()
    print(f"Training loss: {loss:.4f}")

