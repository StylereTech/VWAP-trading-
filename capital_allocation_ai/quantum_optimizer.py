"""
Quantum-Inspired Optimization for VWAP Parameters
Uses QAOA-inspired optimization to tune strategy parameters.
"""

import numpy as np
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


class QuantumInspiredOptimizer:
    """
    Quantum-inspired optimizer for VWAP strategy parameters.
    Uses simulated annealing and QUBO-like formulation.
    """
    
    def __init__(self, param_ranges: Dict[str, Tuple[float, float]]):
        """
        Initialize optimizer.
        
        Args:
            param_ranges: Dictionary of parameter names to (min, max) ranges
                Example: {'sigma_multiplier': (1.0, 3.0), 'lookback': (10, 50)}
        """
        self.param_ranges = param_ranges
        self.param_names = list(param_ranges.keys())
        self.n_params = len(param_ranges)
    
    def optimize(self,
                cost_function,
                method: str = 'simulated_annealing',
                max_iterations: int = 100) -> Dict:
        """
        Optimize parameters using quantum-inspired methods.
        
        Args:
            cost_function: Function that takes params dict and returns cost
            method: 'simulated_annealing' or 'qubo_like'
            max_iterations: Maximum iterations
        
        Returns:
            Dictionary with optimal parameters and cost
        """
        if method == 'simulated_annealing':
            return self._simulated_annealing(cost_function, max_iterations)
        elif method == 'qubo_like':
            return self._qubo_like_optimization(cost_function, max_iterations)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _simulated_annealing(self, cost_function, max_iterations: int) -> Dict:
        """Simulated annealing optimization."""
        # Initialize
        current_params = {}
        for name, (min_val, max_val) in self.param_ranges.items():
            current_params[name] = np.random.uniform(min_val, max_val)
        
        current_cost = cost_function(current_params)
        best_params = current_params.copy()
        best_cost = current_cost
        
        # Temperature schedule
        initial_temp = 100.0
        final_temp = 0.1
        temp = initial_temp
        
        for iteration in range(max_iterations):
            # Generate neighbor
            neighbor_params = current_params.copy()
            param_to_change = np.random.choice(self.param_names)
            min_val, max_val = self.param_ranges[param_to_change]
            neighbor_params[param_to_change] = np.clip(
                neighbor_params[param_to_change] + np.random.normal(0, (max_val - min_val) * 0.1),
                min_val, max_val
            )
            
            neighbor_cost = cost_function(neighbor_params)
            
            # Accept or reject
            delta = neighbor_cost - current_cost
            if delta < 0 or np.random.random() < np.exp(-delta / temp):
                current_params = neighbor_params
                current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best_params = current_params.copy()
                    best_cost = current_cost
            
            # Cool down
            temp = initial_temp * (final_temp / initial_temp) ** (iteration / max_iterations)
        
        return {
            'params': best_params,
            'cost': best_cost,
            'method': 'simulated_annealing'
        }
    
    def _qubo_like_optimization(self, cost_function, max_iterations: int) -> Dict:
        """
        QUBO-like optimization using binary encoding.
        Discretizes continuous parameters into binary variables.
        """
        # Discretize parameters (4 bits per parameter = 16 levels)
        bits_per_param = 4
        n_bits = self.n_params * bits_per_param
        
        def binary_to_params(binary_vector: np.ndarray) -> Dict:
            """Convert binary vector to parameter dict."""
            params = {}
            for i, name in enumerate(self.param_names):
                start_bit = i * bits_per_param
                end_bit = start_bit + bits_per_param
                binary_slice = binary_vector[start_bit:end_bit]
                
                # Convert binary to integer
                param_int = int(''.join(map(str, binary_slice.astype(int))), 2)
                
                # Map to parameter range
                min_val, max_val = self.param_ranges[name]
                n_levels = 2 ** bits_per_param
                params[name] = min_val + (param_int / (n_levels - 1)) * (max_val - min_val)
            
            return params
        
        # Initialize binary vector
        current_binary = np.random.randint(0, 2, n_bits)
        current_params = binary_to_params(current_binary)
        current_cost = cost_function(current_params)
        
        best_binary = current_binary.copy()
        best_params = current_params.copy()
        best_cost = current_cost
        
        # QUBO-like search
        for iteration in range(max_iterations):
            # Flip random bits (quantum-inspired exploration)
            neighbor_binary = current_binary.copy()
            n_flips = np.random.randint(1, min(4, n_bits // 2))
            flip_indices = np.random.choice(n_bits, n_flips, replace=False)
            neighbor_binary[flip_indices] = 1 - neighbor_binary[flip_indices]
            
            neighbor_params = binary_to_params(neighbor_binary)
            neighbor_cost = cost_function(neighbor_params)
            
            # Accept if better
            if neighbor_cost < current_cost:
                current_binary = neighbor_binary
                current_params = neighbor_params
                current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best_binary = neighbor_binary.copy()
                    best_params = neighbor_params.copy()
                    best_cost = current_cost
        
        return {
            'params': best_params,
            'cost': best_cost,
            'method': 'qubo_like'
        }


def optimize_vwap_params(backtest_function,
                        param_ranges: Dict[str, Tuple[float, float]],
                        target_sharpe: float = 1.5,
                        method: str = 'simulated_annealing') -> Dict:
    """
    Optimize VWAP strategy parameters to maximize Sharpe ratio.
    
    Args:
        backtest_function: Function that takes params dict, returns (sharpe, stats)
        param_ranges: Parameter ranges
        target_sharpe: Target Sharpe ratio
        method: Optimization method
    
    Returns:
        Optimal parameters and results
    """
    def cost_function(params: Dict) -> float:
        """Cost function: negative Sharpe ratio (we want to maximize Sharpe)."""
        try:
            sharpe, stats = backtest_function(params)
            # Penalize if below target
            cost = -sharpe
            if sharpe < target_sharpe:
                cost += (target_sharpe - sharpe) * 2  # Extra penalty
            return cost
        except Exception as e:
            return 1000.0  # High cost for invalid params
    
    optimizer = QuantumInspiredOptimizer(param_ranges)
    result = optimizer.optimize(cost_function, method=method, max_iterations=100)
    
    # Get final stats
    sharpe, stats = backtest_function(result['params'])
    
    return {
        'optimal_params': result['params'],
        'sharpe_ratio': sharpe,
        'stats': stats,
        'method': result['method']
    }
