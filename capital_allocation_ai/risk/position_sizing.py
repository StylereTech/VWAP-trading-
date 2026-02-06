"""
Position Sizing Logic - Exact Formula (Production Safe)
ATR-based risk-per-trade sizing with drawdown scaling
"""

from __future__ import annotations
from dataclasses import dataclass
from math import floor
from typing import Optional


@dataclass
class InstrumentSpec:
    """
    dollar_per_price_unit:
      - If XAUUSD and you trade 1 oz per 1.00 move => $1 per $1 move => 1.0
      - If you trade 100 oz contract => 100.0
      - If FX lots, set to pip value mapping (or approximate using broker's spec).
    """
    dollar_per_price_unit: float
    min_units: int = 0
    max_units: Optional[int] = None


@dataclass
class SizingConfig:
    risk_frac_per_trade: float = 0.003  # 0.30% default (prop-safe range 0.25–0.50%)
    atr_mult_stop: float = 2.0
    min_stop_price: float = 0.0         # optional absolute minimum stop distance
    round_down_to: int = 1              # round units to multiple (e.g., 1, 10, 100)


def compute_stop_distance(atr: float, cfg: SizingConfig) -> float:
    """
    Compute stop distance in price units.
    Uses max of ATR-based stop and minimum stop.
    """
    d_atr = cfg.atr_mult_stop * float(atr)
    d_min = float(cfg.min_stop_price)
    return max(d_atr, d_min)


def compute_position_units(
    equity: float,
    stop_distance_price: float,
    inst: InstrumentSpec,
    cfg: SizingConfig,
    dd_scale: float = 1.0,
) -> int:
    """
    Exact formula:
      R$ = equity * risk_frac * dd_scale
      units = floor( R$ / (stop_distance * dollar_per_price_unit) )
    
    Args:
        equity: Current account equity
        stop_distance_price: Stop distance in price units (e.g., $2.50 for XAUUSD)
        inst: Instrument specification
        cfg: Sizing configuration
        dd_scale: Drawdown scaling factor [0..1] from governor
    
    Returns:
        Number of units (lots/contracts) to trade
    """
    if stop_distance_price <= 0:
        return 0

    risk_dollars = float(equity) * float(cfg.risk_frac_per_trade) * float(dd_scale)
    denom = float(stop_distance_price) * float(inst.dollar_per_price_unit)
    raw_units = floor(risk_dollars / denom)

    # round down to multiple
    if cfg.round_down_to > 1:
        raw_units = (raw_units // cfg.round_down_to) * cfg.round_down_to

    # clamp
    raw_units = max(inst.min_units, raw_units)
    if inst.max_units is not None:
        raw_units = min(raw_units, inst.max_units)

    return int(raw_units)
