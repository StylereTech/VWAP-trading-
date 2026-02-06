"""
Optimize VWAP Pro Strategy Parameters
- Quantum-inspired optimization for parameter tuning
- Tighter filters to reduce trade frequency
- Training script to improve strategy performance
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Tuple, List
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from capital_allocation_ai.vwap_pro_strategy import VWAPProStrategy
from capital_allocation_ai.quantum_optimizer import QuantumInspiredOptimizer, optimize_vwap_params


def generate_market_data(symbol: str, days: int = 60) -> pd.DataFrame:
    """Generate realistic market data."""
    print(f"Generating {days} days of {symbol} data...")
    
    trading_days = days
    bars_per_day = 288  # 5-minute bars
    total_bars = trading_days * bars_per_day
    
    np.random.seed(42 if symbol == 'GBPJPY' else (43 if symbol == 'BTCUSD' else 44))
    
    # Symbol-specific parameters
    if symbol == 'GBPJPY':
        base_price = 185.0
        volatility_base = 0.0005
        trend_range = (-2, 3)
    elif symbol == 'BTCUSD':
        base_price = 60000.0
        volatility_base = 0.002
        trend_range = (-5000, 8000)
    else:  # XAUUSD
        base_price = 2650.0
        volatility_base = 0.0008
        trend_range = (-50, 80)
    
    trend = np.linspace(trend_range[0], trend_range[1], total_bars)
    volatility = np.ones(total_bars) * volatility_base
    
    for i in range(1, total_bars):
        volatility[i] = volatility_base * 0.6 + 0.7 * volatility[i-1] + 0.2 * abs(np.random.randn() * volatility_base)
        volatility[i] = min(volatility[i], volatility_base * 4)
    
    returns = np.random.randn(total_bars) * volatility
    for i in range(1, total_bars):
        if abs(returns[i-1]) > volatility_base * 2:
            returns[i] -= 0.3 * returns[i-1]
    
    returns += trend / total_bars
    prices = base_price + np.cumsum(returns)
    
    if symbol == 'GBPJPY':
        prices = np.clip(prices, 175, 200)
    elif symbol == 'BTCUSD':
        prices = np.clip(prices, 50000, 70000)
    else:
        prices = np.clip(prices, 2550, 2750)
    
    opens = prices.copy()
    highs = opens + abs(np.random.randn(total_bars) * volatility * opens * 2)
    lows = opens - abs(np.random.randn(total_bars) * volatility * opens * 2)
    closes = opens + np.random.randn(total_bars) * volatility * opens
    
    for i in range(total_bars):
        highs[i] = max(highs[i], opens[i], closes[i])
        lows[i] = min(lows[i], opens[i], closes[i])
    
    if symbol == 'BTCUSD':
        base_volume = 100
        volume_multiplier = 1 + volatility / volatility.mean() * 3
    elif symbol == 'XAUUSD':
        base_volume = 5000
        volume_multiplier = 1 + volatility / volatility.mean() * 2
    else:
        base_volume = 5000
        volume_multiplier = 1 + volatility / volatility.mean() * 2
    
    volumes = (base_volume * volume_multiplier * (1 + abs(returns) * 10)).astype(int)
    volumes = np.clip(volumes, base_volume * 0.2, base_volume * 10)
    
    start_date = datetime.now() - timedelta(days=days)
    timestamps = []
    current_date = start_date
    bar_count = 0
    
    while bar_count < total_bars:
        while current_date.weekday() >= 5:
            current_date += timedelta(days=1)
        timestamps.append(current_date)
        bar_count += 1
        if bar_count % bars_per_day == 0:
            current_date += timedelta(days=1)
        else:
            current_date += timedelta(minutes=5)
    
    data = pd.DataFrame({
        'timestamp': timestamps[:total_bars],
        'open': opens[:total_bars],
        'high': highs[:total_bars],
        'low': lows[:total_bars],
        'close': closes[:total_bars],
        'volume': volumes[:total_bars]
    })
    
    return data


class OptimizedVWAPStrategy(VWAPProStrategy):
    """
    Enhanced VWAP Pro Strategy with tighter filters.
    Reduces trade frequency while maintaining quality.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Tighter filter parameters
        self.min_volume_multiplier = kwargs.get('min_volume_multiplier', 1.2)  # Require 20% above average
        self.min_touches_before_entry = kwargs.get('min_touches_before_entry', 3)  # 3-touch rule
        self.require_volume_spike = kwargs.get('require_volume_spike', True)
        self.volume_spike_multiplier = kwargs.get('volume_spike_multiplier', 1.5)  # 50% volume spike
        self.require_ema_confirmation = kwargs.get('require_ema_confirmation', True)
        self.require_vwma_confirmation = kwargs.get('require_vwma_confirmation', True)
        self.require_rsi_divergence = kwargs.get('require_rsi_divergence', False)  # Optional but preferred
        
        # Additional filters
        self.min_candle_body_ratio = kwargs.get('min_candle_body_ratio', 0.3)  # Avoid dojis
        self.max_wick_ratio = kwargs.get('max_wick_ratio', 0.5)  # Avoid long wicks on entry
        
        # Track consecutive touches
        self.consecutive_touches = 0
        self.last_touch_bar = None
    
    def _check_filters(self, timestamp, indicators):
        """Enhanced filter checking with tighter requirements."""
        filters = super()._check_filters(timestamp, indicators)
        
        # Tighter volume filter
        if len(self.volume_history) >= 20:
            recent_vol = self.volume_history[-1]
            avg_vol = np.mean(self.volume_history[-20:])
            vol_ratio = recent_vol / avg_vol if avg_vol > 0 else 0
            
            if vol_ratio < self.min_volume_multiplier:
                filters['volume_ok'] = False
            
            # Volume spike requirement
            if self.require_volume_spike:
                if vol_ratio < self.volume_spike_multiplier:
                    filters['volume_ok'] = False
        
        # EMA confirmation
        if self.require_ema_confirmation:
            if indicators['ema_slope'] < -0.0005:  # EMA must be flat or rising
                filters['ema_trend_ok'] = False
        
        # VWMA confirmation
        if self.require_vwma_confirmation:
            if indicators['vwma_slope'] < 0:  # VWMA must be rising
                filters['volume_ok'] = False
        
        # Candle body filter
        if len(self.close_history) >= 2:
            current_close = self.close_history[-1]
            current_open = self.price_history[-1] if len(self.price_history) > 0 else current_close
            current_high = self.high_history[-1] if len(self.high_history) > 0 else current_close
            current_low = self.low_history[-1] if len(self.low_history) > 0 else current_close
            
            body_size = abs(current_close - current_open)
            total_range = current_high - current_low
            
            if total_range > 0:
                body_ratio = body_size / total_range
                if body_ratio < self.min_candle_body_ratio:
                    filters['candle_ok'] = False
                else:
                    filters['candle_ok'] = True
            else:
                filters['candle_ok'] = False
        else:
            filters['candle_ok'] = True
        
        return filters
    
    def _generate_signals(self, close, high, low, vwap, lower_band, upper_band,
                          band_touch, indicators, filters, rsi_divergence, fvg):
        """Enhanced signal generation with tighter entry requirements."""
        signals = super()._generate_signals(close, high, low, vwap, lower_band, upper_band,
                                           band_touch, indicators, filters, rsi_divergence, fvg)
        
        # Track consecutive touches for 3-touch rule
        if band_touch['lower_band']:
            if self.last_touch_bar != len(self.close_history) - 1:
                self.consecutive_touches += 1
                self.last_touch_bar = len(self.close_history) - 1
        elif band_touch['upper_band']:
            if self.last_touch_bar != len(self.close_history) - 1:
                self.consecutive_touches += 1
                self.last_touch_bar = len(self.close_history) - 1
        else:
            # Reset if no touch for several bars
            if len(self.close_history) - self.last_touch_bar > 10:
                self.consecutive_touches = 0
        
        # Override entry signals with tighter requirements
        if signals['entry_long']:
            # Require 3 touches
            if self.consecutive_touches < self.min_touches_before_entry:
                signals['entry_long'] = False
            
            # Require all filters pass
            if not all([filters['volatility_ok'], filters['session_ok'], 
                       filters['volume_ok'], filters['ema_trend_ok'],
                       filters.get('candle_ok', True)]):
                signals['entry_long'] = False
            
            # Prefer RSI divergence but don't require it
            if self.require_rsi_divergence and not rsi_divergence['bullish']:
                signals['entry_long'] = False
        
        if signals['reposition_long']:
            # Same tight filters for reposition
            if not all([filters['volatility_ok'], filters['session_ok'], 
                       filters['volume_ok'], filters.get('candle_ok', True)]):
                signals['reposition_long'] = False
        
        return signals


def backtest_optimized_strategy(data: pd.DataFrame, symbol: str, params: Dict = None) -> Tuple[float, Dict]:
    """
    Backtest optimized strategy with tighter filters.
    
    Returns:
        (sharpe_ratio, statistics_dict)
    """
    if params is None:
        params = {}
    
    # Symbol-specific defaults
    if symbol == 'BTCUSD':
        default_params = {
            'sigma_multiplier': 2.5,
            'lookback_periods': 20,
            'volatility_threshold': 3.0,
            'trailing_stop_pct': 0.01,
            'min_volume_multiplier': 1.3,
            'volume_spike_multiplier': 1.8
        }
    elif symbol == 'XAUUSD':
        default_params = {
            'sigma_multiplier': 1.8,
            'lookback_periods': 20,
            'volatility_threshold': 2.0,
            'trailing_stop_pct': 0.005,
            'min_volume_multiplier': 1.2,
            'volume_spike_multiplier': 1.5
        }
    else:  # GBPJPY
        default_params = {
            'sigma_multiplier': 2.0,
            'lookback_periods': 20,
            'volatility_threshold': 2.5,
            'trailing_stop_pct': 0.007,
            'min_volume_multiplier': 1.2,
            'volume_spike_multiplier': 1.5
        }
    
    final_params = {**default_params, **params}
    
    strategy = OptimizedVWAPStrategy(**final_params)
    
    initial_equity = 10000.0
    equity = initial_equity
    equity_history = [equity]
    trades = []
    max_drawdown = 0.0
    peak_equity = initial_equity
    
    for idx, row in data.iterrows():
        result = strategy.update(
            current_price=row['close'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
            timestamp=row.get('timestamp')
        )
        
        signals = result['signals']
        
        # Execute signals
        if signals['entry_long'] and strategy.position == 0:
            strategy.enter_long(row['close'])
        
        elif signals['reposition_long'] and strategy.position == 0:
            strategy.enter_long(row['close'])
        
        elif signals['exit'] or signals['stop_hit']:
            if strategy.position != 0:
                exit_price = row['close']
                position_size = equity / strategy.entry_price * 0.1
                
                if strategy.position == 1:
                    pnl = (exit_price - strategy.entry_price) * position_size
                else:
                    pnl = (strategy.entry_price - exit_price) * position_size
                
                commission = abs(position_size * exit_price) * 0.0001
                pnl -= commission
                equity += pnl
                
                trades.append({
                    'entry': strategy.entry_price,
                    'exit': exit_price,
                    'pnl': pnl,
                    'type': 'long' if strategy.position == 1 else 'short'
                })
                
                strategy.exit_position()
        
        peak_equity = max(peak_equity, equity)
        drawdown = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        max_drawdown = max(max_drawdown, drawdown)
        equity_history.append(equity)
    
    # Close open positions
    if strategy.position != 0:
        final_price = data.iloc[-1]['close']
        position_size = equity / strategy.entry_price * 0.1
        if strategy.position == 1:
            pnl = (final_price - strategy.entry_price) * position_size
        else:
            pnl = (strategy.entry_price - final_price) * position_size
        commission = abs(position_size * final_price) * 0.0001
        pnl -= commission
        equity += pnl
        equity_history.append(equity)
        strategy.exit_position()
    
    # Calculate statistics
    if len(trades) == 0:
        return 0.0, {
            'symbol': symbol,
            'total_trades': 0,
            'win_rate': 0.0,
            'total_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'trades_per_day': 0.0
        }
    
    trades_df = pd.DataFrame(trades)
    winning_trades = (trades_df['pnl'] > 0).sum()
    win_rate = winning_trades / len(trades) if len(trades) > 0 else 0.0
    total_return = (equity - initial_equity) / initial_equity
    
    equity_array = np.array(equity_history)
    if len(equity_array) > 1:
        returns = np.diff(equity_array) / equity_array[:-1]
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 288) if len(returns) > 1 and np.std(returns) > 0 else 0.0
    else:
        sharpe = 0.0
    
    # Calculate trades per day
    days_traded = len(data) / 288  # 288 bars per day
    trades_per_day = len(trades) / days_traded if days_traded > 0 else 0.0
    
    stats = {
        'symbol': symbol,
        'total_trades': len(trades),
        'winning_trades': winning_trades,
        'losing_trades': len(trades) - winning_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'final_equity': equity,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'trades_per_day': trades_per_day,
        'avg_win': trades_df[trades_df['pnl'] > 0]['pnl'].mean() if (trades_df['pnl'] > 0).any() else 0.0,
        'avg_loss': trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (trades_df['pnl'] < 0).any() else 0.0,
        'profit_factor': abs(trades_df[trades_df['pnl'] > 0]['pnl'].sum() / trades_df[trades_df['pnl'] < 0]['pnl'].sum()) if (trades_df['pnl'] < 0).any() and trades_df[trades_df['pnl'] < 0]['pnl'].sum() != 0 else 0.0
    }
    
    return sharpe, stats


def optimize_parameters_quantum(symbol: str, data: pd.DataFrame, target_trades_per_day: float = 3.0) -> Dict:
    """
    Optimize parameters using quantum-inspired optimizer.
    Optimizes for Sharpe ratio while constraining trade frequency.
    """
    print(f"\nOptimizing {symbol} parameters using quantum-inspired optimizer...")
    
    # Parameter ranges for optimization
    param_ranges = {
        'sigma_multiplier': (1.5, 3.0),
        'lookback_periods': (15, 30),
        'volatility_threshold': (2.0, 4.0),
        'min_volume_multiplier': (1.0, 2.0),
        'volume_spike_multiplier': (1.2, 2.5),
        'min_touches_before_entry': (2, 5)
    }
    
    def cost_function(params: Dict) -> float:
        """Cost function: negative Sharpe + penalty for too many trades."""
        try:
            sharpe, stats = backtest_optimized_strategy(data, symbol, params)
            
            # Penalty for too many trades
            trades_per_day = stats.get('trades_per_day', 100)
            trade_penalty = 0.0
            if trades_per_day > target_trades_per_day * 1.5:
                trade_penalty = (trades_per_day - target_trades_per_day * 1.5) * 10
            
            # Penalty for negative Sharpe
            sharpe_penalty = 0.0
            if sharpe < 0:
                sharpe_penalty = abs(sharpe) * 5
            
            # Penalty for high drawdown
            dd_penalty = 0.0
            if stats.get('max_drawdown', 0) > 0.20:
                dd_penalty = (stats['max_drawdown'] - 0.20) * 20
            
            # Total cost (we want to minimize this)
            cost = -sharpe + trade_penalty + sharpe_penalty + dd_penalty
            
            return cost
        except Exception as e:
            print(f"Error in cost function: {e}")
            return 1000.0
    
    optimizer = QuantumInspiredOptimizer(param_ranges)
    result = optimizer.optimize(cost_function, method='simulated_annealing', max_iterations=50)
    
    # Get final stats with optimized params
    sharpe, stats = backtest_optimized_strategy(data, symbol, result['params'])
    
    return {
        'optimal_params': result['params'],
        'sharpe_ratio': sharpe,
        'stats': stats,
        'method': result['method']
    }


def train_strategy_improvement(symbol: str, data: pd.DataFrame, episodes: int = 100):
    """
    Train strategy to improve entry/exit signals.
    Uses iterative optimization to refine parameters.
    """
    print(f"\nTraining strategy for {symbol}...")
    
    # Split data into train/test
    split_idx = int(len(data) * 0.7)
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    
    best_params = None
    best_sharpe = -999.0
    best_stats = None
    
    # Iterative improvement
    current_params = {
        'sigma_multiplier': 2.0,
        'lookback_periods': 20,
        'volatility_threshold': 2.5,
        'min_volume_multiplier': 1.2,
        'volume_spike_multiplier': 1.5,
        'min_touches_before_entry': 3
    }
    
    for episode in range(episodes):
        # Test current parameters
        sharpe, stats = backtest_optimized_strategy(train_data, symbol, current_params)
        
        # Update best if improved
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = current_params.copy()
            best_stats = stats
        
        # Adjust parameters based on performance
        if stats['trades_per_day'] > 5:
            # Too many trades - tighten filters
            current_params['min_volume_multiplier'] = min(2.0, current_params['min_volume_multiplier'] + 0.05)
            current_params['min_touches_before_entry'] = min(5, current_params['min_touches_before_entry'] + 1)
        elif stats['trades_per_day'] < 1:
            # Too few trades - loosen filters
            current_params['min_volume_multiplier'] = max(1.0, current_params['min_volume_multiplier'] - 0.05)
            current_params['min_touches_before_entry'] = max(2, current_params['min_touches_before_entry'] - 1)
        
        if stats['win_rate'] < 0.50:
            # Low win rate - tighten entry
            current_params['sigma_multiplier'] = min(3.0, current_params['sigma_multiplier'] + 0.1)
        elif stats['win_rate'] > 0.65:
            # High win rate - can loosen slightly
            current_params['sigma_multiplier'] = max(1.5, current_params['sigma_multiplier'] - 0.05)
        
        if (episode + 1) % 20 == 0:
            print(f"Episode {episode+1}/{episodes}: Sharpe={sharpe:.2f}, "
                  f"Trades/day={stats['trades_per_day']:.1f}, Win Rate={stats['win_rate']*100:.1f}%")
    
    # Test on out-of-sample data
    print(f"\nTesting optimized parameters on out-of-sample data...")
    test_sharpe, test_stats = backtest_optimized_strategy(test_data, symbol, best_params)
    
    return {
        'trained_params': best_params,
        'train_sharpe': best_sharpe,
        'train_stats': best_stats,
        'test_sharpe': test_sharpe,
        'test_stats': test_stats
    }


def main():
    """Main optimization and training function."""
    print("="*80)
    print("VWAP PRO STRATEGY - OPTIMIZATION & TRAINING")
    print("GBP/JPY, BTC/USD, Gold/USD - Last 60 Days")
    print("="*80)
    
    symbols = ['GBPJPY', 'BTCUSD', 'XAUUSD']
    all_results = []
    
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"PROCESSING {symbol}")
        print(f"{'='*80}")
        
        # Generate data
        data = generate_market_data(symbol, days=60)
        
        # Step 1: Optimize parameters using quantum-inspired optimizer
        print(f"\nStep 1: Quantum-Inspired Parameter Optimization")
        opt_result = optimize_parameters_quantum(symbol, data, target_trades_per_day=3.0)
        
        print(f"\nOptimized Parameters:")
        for param, value in opt_result['optimal_params'].items():
            print(f"  {param}: {value:.3f}")
        print(f"Sharpe Ratio: {opt_result['sharpe_ratio']:.2f}")
        print(f"Trades/Day: {opt_result['stats']['trades_per_day']:.2f}")
        print(f"Win Rate: {opt_result['stats']['win_rate']*100:.1f}%")
        
        # Step 2: Train strategy to improve signals
        print(f"\nStep 2: Training Strategy")
        train_result = train_strategy_improvement(symbol, data, episodes=100)
        
        print(f"\nTrained Parameters:")
        for param, value in train_result['trained_params'].items():
            print(f"  {param}: {value:.3f}")
        print(f"Train Sharpe: {train_result['train_sharpe']:.2f}")
        print(f"Test Sharpe: {train_result['test_sharpe']:.2f}")
        print(f"Test Trades/Day: {train_result['test_stats']['trades_per_day']:.2f}")
        print(f"Test Win Rate: {train_result['test_stats']['win_rate']*100:.1f}%")
        
        # Use best parameters (from training)
        final_params = train_result['trained_params']
        final_sharpe, final_stats = backtest_optimized_strategy(data, symbol, final_params)
        
        all_results.append({
            'symbol': symbol,
            'params': final_params,
            'stats': final_stats
        })
    
    # Print summary
    print("\n" + "="*80)
    print("OPTIMIZATION & TRAINING SUMMARY")
    print("="*80)
    
    for result in all_results:
        stats = result['stats']
        print(f"\n{result['symbol']}:")
        print(f"  Trades: {stats['total_trades']} ({stats['trades_per_day']:.1f}/day)")
        print(f"  Win Rate: {stats['win_rate']*100:.1f}%")
        print(f"  Return: {stats['total_return']*100:+.2f}%")
        print(f"  Sharpe: {stats['sharpe_ratio']:.2f}")
        print(f"  Max DD: {stats['max_drawdown']*100:.1f}%")
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
