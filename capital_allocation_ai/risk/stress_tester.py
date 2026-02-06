"""
Stress Tester - Monte Carlo + Worst-Case Sequence Analysis
Tests strategy robustness under adversarial conditions
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
import math


@dataclass
class PropRules:
    max_daily_loss_frac: float = 0.02
    max_total_loss_frac: float = 0.05
    peak_trailing_dd_hard: float = 0.045


@dataclass
class StressResult:
    breach_prob: float
    breach_prob_daily: float
    breach_prob_total: float
    breach_prob_peakdd: float
    avg_max_dd: float
    p95_max_dd: float
    worst_equity: float
    worst_max_dd: float


def _max_drawdown(equity_curve: List[float]) -> float:
    """Calculate maximum drawdown from equity curve."""
    peak = equity_curve[0]
    max_dd = 0.0
    for e in equity_curve:
        if e > peak:
            peak = e
        dd = (peak - e) / peak if peak > 0 else 0.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def worst_case_sequence(
    start_equity: float,
    trade_pnls: List[float],
) -> Dict[str, float]:
    """
    Apply worst losses first (deterministic nightmare ordering).
    """
    eq = float(start_equity)
    curve = [eq]
    for pnl in sorted(trade_pnls):
        eq += pnl
        curve.append(eq)
    return {
        "worst_end_equity": eq,
        "worst_max_dd": _max_drawdown(curve),
        "trades": len(trade_pnls),
    }


def monte_carlo_paths(
    start_equity: float,
    trade_pnls: List[float],
    trades_per_day: int,
    days: int,
    rules: PropRules,
    n_paths: int = 5000,
    seed: int = 7,
) -> StressResult:
    """
    Bootstrap trade PnLs with replacement into many paths.
    Checks:
      - daily loss limit
      - total loss limit
      - peak trailing dd limit
    """
    rng = random.Random(seed)
    pnls = list(trade_pnls)
    if not pnls:
        return StressResult(
            breach_prob=1.0, breach_prob_daily=1.0, breach_prob_total=1.0, breach_prob_peakdd=1.0,
            avg_max_dd=1.0, p95_max_dd=1.0, worst_equity=0.0, worst_max_dd=1.0
        )

    daily_breaches = 0
    total_breaches = 0
    peakdd_breaches = 0
    any_breaches = 0
    max_dds: List[float] = []
    worst_equity = float("inf")
    worst_max_dd = 0.0

    for _ in range(n_paths):
        eq = float(start_equity)
        peak = eq
        start = eq
        breached = False
        breached_daily = False
        breached_total = False
        breached_peak = False
        curve = [eq]

        for _day in range(days):
            day_start = eq
            for _t in range(trades_per_day):
                eq += rng.choice(pnls)
                curve.append(eq)
                if eq > peak:
                    peak = eq

                # total loss
                if (start - eq) / start >= rules.max_total_loss_frac:
                    breached = True
                    breached_total = True
                    break

                # peak trailing dd
                if peak > 0 and (peak - eq) / peak >= rules.peak_trailing_dd_hard:
                    breached = True
                    breached_peak = True
                    break

            # daily loss check at end of day
            if day_start > 0 and (day_start - eq) / day_start >= rules.max_daily_loss_frac:
                breached = True
                breached_daily = True

            if breached:
                break

        max_dd = _max_drawdown(curve)
        max_dds.append(max_dd)
        worst_equity = min(worst_equity, eq)
        worst_max_dd = max(worst_max_dd, max_dd)

        if breached:
            any_breaches += 1
        if breached_daily:
            daily_breaches += 1
        if breached_total:
            total_breaches += 1
        if breached_peak:
            peakdd_breaches += 1

    max_dds.sort()
    p95 = max_dds[int(0.95 * (len(max_dds) - 1))] if max_dds else 1.0

    return StressResult(
        breach_prob=any_breaches / n_paths,
        breach_prob_daily=daily_breaches / n_paths,
        breach_prob_total=total_breaches / n_paths,
        breach_prob_peakdd=peakdd_breaches / n_paths,
        avg_max_dd=sum(max_dds) / len(max_dds) if max_dds else 1.0,
        p95_max_dd=p95,
        worst_equity=worst_equity,
        worst_max_dd=worst_max_dd,
    )
