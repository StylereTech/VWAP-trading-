"""
Capital Allocation AI System
"""

from .system import CapitalAllocationAI, CapitalAllocationStateEncoder, CapitalAllocationAgent
from .vwap_bands import VWAPBandsCalculator
from .training import train_capital_allocation_ai, TrainingEnvironment

__version__ = "1.0.0"
__all__ = [
    'CapitalAllocationAI',
    'CapitalAllocationStateEncoder',
    'CapitalAllocationAgent',
    'VWAPBandsCalculator',
    'train_capital_allocation_ai',
    'TrainingEnvironment'
]
