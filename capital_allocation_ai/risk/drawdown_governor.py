"""
Drawdown Governor Module - Hard Risk Control
Tracks equity peak + daily start equity. Provides:
  - can_trade(): hard block on prop constraints
  - risk_scale(): [0..1] scaling factor as drawdown increases
"""

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, date
from typing import Optional


@dataclass
class GovernorConfig:
    # Prop-style constraints
    max_daily_loss_frac: float = 0.02     # 2% daily loss => stop trading
    max_total_loss_frac: float = 0.05     # 5% overall loss => stop trading
    peak_trailing_dd_hard: float = 0.045  # 4.5% peak-to-trough => stop trading

    # Risk scaling region (soft->hard)
    dd_soft: float = 0.02                # start scaling risk at 2% dd from peak
    dd_hard: float = 0.045               # stop trading at 4.5% dd from peak


@dataclass
class GovernorState:
    start_equity: float
    peak_equity: float
    day_start_equity: float
    current_day: date
    halted: bool = False
    halt_reason: Optional[str] = None


class DrawdownGovernor:
    """
    Tracks equity peak + daily start equity. Provides:
      - can_trade(): hard block on prop constraints
      - risk_scale(): [0..1] scaling factor as drawdown increases
    """

    def __init__(self, cfg: GovernorConfig, start_equity: float, ts: datetime):
        d = ts.date()
        self.cfg = cfg
        self.state = GovernorState(
            start_equity=float(start_equity),
            peak_equity=float(start_equity),
            day_start_equity=float(start_equity),
            current_day=d,
        )

    def on_bar(self, equity: float, ts: datetime) -> None:
        """Update governor state on each bar."""
        if self.state.halted:
            return

        # Daily reset
        if ts.date() != self.state.current_day:
            self.state.current_day = ts.date()
            self.state.day_start_equity = float(equity)

        # Peak tracking
        if equity > self.state.peak_equity:
            self.state.peak_equity = float(equity)

        # Hard checks
        if self._daily_loss_frac(equity) >= self.cfg.max_daily_loss_frac:
            self._halt("max_daily_loss")
        elif self._total_loss_frac(equity) >= self.cfg.max_total_loss_frac:
            self._halt("max_total_loss")
        elif self._peak_dd_frac(equity) >= self.cfg.peak_trailing_dd_hard:
            self._halt("peak_trailing_dd_hard")

    def can_trade(self) -> bool:
        """Check if trading is allowed."""
        return not self.state.halted

    def risk_scale(self, equity: float) -> float:
        """
        Linear scaling between dd_soft and dd_hard from peak.
        Returns value in [0..1] where 1.0 = full risk, 0.0 = no risk.
        """
        if self.state.halted:
            return 0.0

        dd = self._peak_dd_frac(equity)
        if dd <= self.cfg.dd_soft:
            return 1.0
        if dd >= self.cfg.dd_hard:
            return 0.0

        # Linear ramp down
        return (self.cfg.dd_hard - dd) / (self.cfg.dd_hard - self.cfg.dd_soft)

    def _daily_loss_frac(self, equity: float) -> float:
        """Calculate daily loss fraction."""
        return max(0.0, (self.state.day_start_equity - equity) / self.state.day_start_equity)

    def _total_loss_frac(self, equity: float) -> float:
        """Calculate total loss fraction from start."""
        return max(0.0, (self.state.start_equity - equity) / self.state.start_equity)

    def _peak_dd_frac(self, equity: float) -> float:
        """Calculate peak-to-trough drawdown fraction."""
        return max(0.0, (self.state.peak_equity - equity) / self.state.peak_equity)

    def _halt(self, reason: str) -> None:
        """Halt trading with reason."""
        self.state.halted = True
        self.state.halt_reason = reason

    def get_state(self) -> dict:
        """Get current state for logging/debugging."""
        return {
            'halted': self.state.halted,
            'halt_reason': self.state.halt_reason,
            'peak_equity': self.state.peak_equity,
            'current_equity': self.state.day_start_equity,  # approximate
            'peak_dd_frac': self._peak_dd_frac(self.state.day_start_equity),
        }
