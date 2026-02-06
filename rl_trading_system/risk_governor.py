"""
Risk Governor: Enforces risk limits and position sizing rules.
Acts as a safety layer between RL agent and execution.
"""

from typing import Dict, Optional, Tuple
import numpy as np


class RiskGovernor:
    """
    Risk management system that enforces:
    - Maximum drawdown limits
    - Position size limits
    - Daily loss limits
    - Consecutive loss limits
    """
    
    def __init__(self,
                 max_drawdown_pct: float = 0.20,  # 20% max drawdown
                 max_position_size_pct: float = 0.50,  # Max 50% of equity per position
                 daily_loss_limit_pct: float = 0.05,  # 5% daily loss limit
                 consecutive_loss_limit: int = 5,  # Stop after 5 consecutive losses
                 min_equity_threshold: float = 0.50):  # Stop if equity drops below 50% of initial
        """
        Args:
            max_drawdown_pct: Maximum allowed drawdown from peak equity
            max_position_size_pct: Maximum position size as fraction of equity
            daily_loss_limit_pct: Maximum daily loss as fraction of equity
            consecutive_loss_limit: Maximum consecutive losses before halt
            min_equity_threshold: Minimum equity as fraction of initial equity
        """
        self.max_drawdown_pct = max_drawdown_pct
        self.max_position_size_pct = max_position_size_pct
        self.daily_loss_limit_pct = daily_loss_limit_pct
        self.consecutive_loss_limit = consecutive_loss_limit
        self.min_equity_threshold = min_equity_threshold
        
        # State tracking
        self.initial_equity = None
        self.peak_equity = None
        self.daily_start_equity = None
        self.consecutive_losses = 0
        self.trading_halted = False
        self.halt_reason = ""
        
    def initialize(self, initial_equity: float):
        """Initialize risk governor with starting equity."""
        self.initial_equity = initial_equity
        self.peak_equity = initial_equity
        self.daily_start_equity = initial_equity
        self.consecutive_losses = 0
        self.trading_halted = False
        self.halt_reason = ""
    
    def update_equity(self, current_equity: float):
        """Update equity tracking."""
        if self.peak_equity is None:
            self.peak_equity = current_equity
        
        self.peak_equity = max(self.peak_equity, current_equity)
    
    def check_risk_limits(self, current_equity: float) -> Tuple[bool, str]:
        """
        Check if trading should be halted due to risk limits.
        
        Returns:
            (allowed, reason): Whether trading is allowed and reason if not
        """
        if self.peak_equity is None:
            return True, ""
        
        # Check drawdown limit
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if drawdown > self.max_drawdown_pct:
            self.trading_halted = True
            self.halt_reason = f"Max drawdown exceeded: {drawdown*100:.1f}%"
            return False, self.halt_reason
        
        # Check minimum equity threshold
        if self.initial_equity is not None:
            equity_ratio = current_equity / self.initial_equity
            if equity_ratio < self.min_equity_threshold:
                self.trading_halted = True
                self.halt_reason = f"Equity below threshold: {equity_ratio*100:.1f}% of initial"
                return False, self.halt_reason
        
        # Check daily loss limit
        if self.daily_start_equity is not None:
            daily_loss = (self.daily_start_equity - current_equity) / self.daily_start_equity
            if daily_loss > self.daily_loss_limit_pct:
                self.trading_halted = True
                self.halt_reason = f"Daily loss limit exceeded: {daily_loss*100:.1f}%"
                return False, self.halt_reason
        
        # Check consecutive losses
        if self.consecutive_losses >= self.consecutive_loss_limit:
            self.trading_halted = True
            self.halt_reason = f"Consecutive loss limit: {self.consecutive_losses} losses"
            return False, self.halt_reason
        
        # If we were halted but conditions improved, resume
        if self.trading_halted:
            # Resume if equity recovered and we have some wins
            if drawdown < self.max_drawdown_pct * 0.8 and self.consecutive_losses < 3:
                self.trading_halted = False
                self.halt_reason = ""
                return True, "Trading resumed"
        
        return True, ""
    
    def adjust_position_size(self, requested_size: float, current_equity: float) -> float:
        """
        Adjust position size based on risk limits.
        
        Args:
            requested_size: Requested position size fraction [0, 1]
            current_equity: Current account equity
        
        Returns:
            Adjusted position size fraction
        """
        # Cap at maximum position size
        adjusted_size = min(requested_size, self.max_position_size_pct)
        
        # Reduce size if drawdown is high
        if self.peak_equity is not None:
            drawdown = (self.peak_equity - current_equity) / self.peak_equity
            if drawdown > self.max_drawdown_pct * 0.5:  # Reduce size if drawdown > 10%
                reduction_factor = 1.0 - (drawdown / self.max_drawdown_pct)
                adjusted_size *= max(0.1, reduction_factor)  # At least 10% of requested
        
        # Reduce size after consecutive losses
        if self.consecutive_losses > 0:
            reduction = 1.0 - (self.consecutive_losses / self.consecutive_loss_limit) * 0.5
            adjusted_size *= max(0.2, reduction)  # At least 20% of requested
        
        return max(0.0, min(adjusted_size, self.max_position_size_pct))
    
    def record_trade_result(self, pnl: float):
        """Record trade result for consecutive loss tracking."""
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
    
    def reset_daily(self, current_equity: float):
        """Reset daily tracking (call at start of each day)."""
        self.daily_start_equity = current_equity
    
    def get_status(self) -> Dict:
        """Get current risk governor status."""
        return {
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason,
            'consecutive_losses': self.consecutive_losses,
            'peak_equity': self.peak_equity,
            'initial_equity': self.initial_equity
        }


class PositionSizingEngine:
    """
    Position sizing engine that determines optimal position size.
    Can use fixed sizing, risk-based sizing, or RL agent output.
    """
    
    def __init__(self,
                 method: str = 'rl',  # 'rl', 'fixed', 'risk_based'
                 fixed_size: float = 0.10,  # Fixed position size fraction
                 risk_per_trade: float = 0.02,  # 2% risk per trade for risk-based
                 max_size: float = 0.50):  # Maximum position size
        """
        Args:
            method: Sizing method ('rl', 'fixed', 'risk_based')
            fixed_size: Fixed position size (if method='fixed')
            risk_per_trade: Risk per trade as fraction of equity (if method='risk_based')
            max_size: Maximum position size fraction
        """
        self.method = method
        self.fixed_size = fixed_size
        self.risk_per_trade = risk_per_trade
        self.max_size = max_size
    
    def calculate_size(self,
                      rl_output: Optional[float] = None,
                      current_equity: float = 10000.0,
                      entry_price: float = 100.0,
                      stop_loss_price: Optional[float] = None) -> float:
        """
        Calculate position size based on selected method.
        
        Args:
            rl_output: Position size from RL agent [0, 1] (if method='rl')
            current_equity: Current account equity
            entry_price: Entry price
            stop_loss_price: Stop loss price (for risk-based sizing)
        
        Returns:
            Position size fraction [0, 1]
        """
        if self.method == 'fixed':
            return min(self.fixed_size, self.max_size)
        
        elif self.method == 'risk_based':
            if stop_loss_price is None:
                return min(self.fixed_size, self.max_size)
            
            risk_amount = current_equity * self.risk_per_trade
            stop_distance = abs(entry_price - stop_loss_price)
            if stop_distance == 0:
                return min(self.fixed_size, self.max_size)
            
            position_value = risk_amount / (stop_distance / entry_price)
            position_size_fraction = position_value / current_equity
            return min(position_size_fraction, self.max_size)
        
        else:  # 'rl'
            if rl_output is None:
                return min(self.fixed_size, self.max_size)
            return min(rl_output, self.max_size)


if __name__ == "__main__":
    # Test risk governor
    risk_gov = RiskGovernor()
    risk_gov.initialize(10000.0)
    
    # Simulate some equity changes
    risk_gov.update_equity(11000.0)  # Peak
    risk_gov.update_equity(9000.0)   # 18% drawdown
    
    allowed, reason = risk_gov.check_risk_limits(9000.0)
    print(f"Trading allowed: {allowed}, Reason: {reason}")
    
    # Test position sizing
    sizing = PositionSizingEngine(method='rl')
    size = sizing.calculate_size(rl_output=0.3, current_equity=10000.0)
    print(f"Position size: {size:.2%}")

