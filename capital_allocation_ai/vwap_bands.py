"""
VWAP Bands Calculator
Calculates VWAP with volume-weighted standard deviation bands (session-reset).
Same as TradeLockerVWAPStrategy but standalone.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime


class VWAPBandsCalculator:
    """
    Calculates VWAP with bands (R1/S1, R2/S2, R3/S3).
    Resets daily like the original strategy.
    """
    
    def __init__(self, stdev1: float = 1.0, stdev2: float = 2.0, stdev3: float = 3.0):
        self.stdev1 = stdev1
        self.stdev2 = stdev2
        self.stdev3 = stdev3
        
        self.current_day = None
        self.cum_typ_vol = 0.0
        self.cum_vol = 0.0
        self.day_typical_prices = []
        self.day_volumes = []
    
    def calculate(self,
                 high: float,
                 low: float,
                 close: float,
                 volume: float,
                 timestamp: Optional[datetime] = None) -> Dict[str, float]:
        """
        Calculate VWAP and bands for current bar.
        
        Returns:
            Dictionary with vwap, r1, s1, r2, s2, r3, s3
        """
        # Check if new day
        if timestamp:
            current_day = timestamp.date()
        else:
            current_day = None
        
        if self.current_day != current_day and self.current_day is not None:
            # Reset for new day
            self.current_day = current_day
            self.cum_typ_vol = 0.0
            self.cum_vol = 0.0
            self.day_typical_prices = []
            self.day_volumes = []
        elif self.current_day is None:
            self.current_day = current_day
        
        # Calculate typical price
        typical_price = (high + low + close) / 3.0
        vol = float(volume) if volume > 0 else 0.0
        
        # Update day's data
        self.day_typical_prices.append(typical_price)
        self.day_volumes.append(vol)
        
        # Update cumulative values
        self.cum_typ_vol += typical_price * vol
        self.cum_vol += vol
        
        # Calculate VWAP
        vwap = (self.cum_typ_vol / self.cum_vol) if self.cum_vol > 0 else typical_price
        
        # Calculate volume-weighted standard deviation
        if self.cum_vol > 0 and len(self.day_typical_prices) > 1:
            wssd = 0.0
            for tp, v in zip(self.day_typical_prices, self.day_volumes):
                wssd += ((tp - vwap) ** 2) * v
            std_dev = np.sqrt(wssd / self.cum_vol)
        else:
            std_dev = 0.0
        
        # Calculate bands
        r1 = vwap + self.stdev1 * std_dev
        s1 = vwap - self.stdev1 * std_dev
        r2 = vwap + self.stdev2 * std_dev
        s2 = vwap - self.stdev2 * std_dev
        r3 = vwap + self.stdev3 * std_dev
        s3 = vwap - self.stdev3 * std_dev
        
        return {
            'vwap': vwap,
            'r1': r1,
            's1': s1,
            'r2': r2,
            's2': s2,
            'r3': r3,
            's3': s3,
            'std_dev': std_dev
        }
    
    def reset(self):
        """Reset calculator (for new session)."""
        self.current_day = None
        self.cum_typ_vol = 0.0
        self.cum_vol = 0.0
        self.day_typical_prices = []
        self.day_volumes = []
