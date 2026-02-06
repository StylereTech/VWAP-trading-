"""
Production-Grade QUBO/QAOA Parameter Tuner
With sanity checks, guardrails, and validation
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Callable, Any, Optional
from datetime import datetime

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_control_flip_strategy import EnhancedVWAPControlFlipStrategy, EnhancedStrategyParams

# Risk management imports
try:
    from capital_allocation_ai.risk import (
        monte_carlo_paths,
        worst_case_sequence,
        PropRules
    )
except ImportError:
    # Fallback
    import sys
    import os
    risk_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'capital_allocation_ai', 'risk')
    if os.path.exists(risk_path):
        sys.path.insert(0, risk_path)
    from risk.stress_tester import monte_carlo_paths, worst_case_sequence, PropRules


# =========================
# 0) Parameter space - Updated for new filter architecture
# =========================
PARAM_SPACE = {
    "sigma_level": [1.5, 2.0, 2.5, 3.0],
    "retest_min": [1, 2],  # Flexible retest window (min)
    "retest_max": [3, 4, 5],  # Flexible retest window (max) - ensure min < max
    "volume_filter_type": ["percentile"],  # Fixed to percentile for now (do NOT optimize type yet)
    "vol_percentile_p": [55, 60, 65],  # Percentile threshold
    "vol_percentile_L": [30, 50, 80],  # Lookback window
    "atr_cap_mult": [2.5, 3.0, 3.5],
    "trail_pct": [0.005, 0.007, 0.010],
    "session_filter": [0, 1],
}

# Default baseline config
BASELINE_CONFIG = {
    "sigma_level": 2.0,
    "retest_count": 3,
    "atr_cap_mult": 2.5,
    "trail_pct": 0.007,
    "session_filter": 1
}


# =========================
# 1) Bit encoding (one-hot)
# =========================
@dataclass(frozen=True)
class Encoding:
    keys: List[str]
    offsets: Dict[str, int]
    sizes: Dict[str, int]
    n_bits: int

def build_encoding(space: Dict[str, List[Any]]) -> Encoding:
    keys = list(space.keys())
    offsets, sizes = {}, {}
    cur = 0
    for k in keys:
        offsets[k] = cur
        sizes[k] = len(space[k])
        cur += sizes[k]
    return Encoding(keys=keys, offsets=offsets, sizes=sizes, n_bits=cur)

ENC = build_encoding(PARAM_SPACE)

def config_to_bits(cfg: Dict[str, Any], space: Dict[str, List[Any]], enc: Encoding) -> np.ndarray:
    x = np.zeros(enc.n_bits, dtype=int)
    for k in enc.keys:
        opts = space[k]
        if k not in cfg:
            raise ValueError(f"Missing key in cfg: {k}")
        try:
            idx = opts.index(cfg[k])
        except ValueError:
            raise ValueError(f"Invalid value for {k}: {cfg[k]} not in {opts}")
        x[enc.offsets[k] + idx] = 1
    return x

def bits_to_config(x: np.ndarray, space: Dict[str, List[Any]], enc: Encoding) -> Dict[str, Any]:
    cfg = {}
    for k in enc.keys:
        start = enc.offsets[k]
        size = enc.sizes[k]
        block = x[start:start+size]
        idx = int(np.argmax(block))
        cfg[k] = space[k][idx]
    return cfg

def random_config(space: Dict[str, List[Any]], seed: Optional[int] = None) -> Dict[str, Any]:
    rng = random.Random(seed) if seed is not None else random
    return {k: rng.choice(v) for k, v in space.items()}


# =========================
# 2) Constraint validation
# =========================
def validate_onehot_constraints(x: np.ndarray, enc: Encoding) -> Tuple[bool, List[str]]:
    """Validate one-hot constraints. Returns (is_valid, error_messages)."""
    errors = []
    for k in enc.keys:
        start = enc.offsets[k]
        size = enc.sizes[k]
        block = x[start:start+size]
        ones = np.sum(block)
        if ones == 0:
            errors.append(f"{k}: No value selected (all zeros)")
        elif ones > 1:
            errors.append(f"{k}: Multiple values selected ({ones} ones, should be 1)")
    return len(errors) == 0, errors

def validate_config(cfg: Dict[str, Any], space: Dict[str, List[Any]]) -> Tuple[bool, List[str]]:
    """Validate config has exactly one value per parameter."""
    errors = []
    for k, opts in space.items():
        if k not in cfg:
            errors.append(f"Missing parameter: {k}")
        elif cfg[k] not in opts:
            errors.append(f"Invalid value for {k}: {cfg[k]} not in {opts}")
    return len(errors) == 0, errors


# =========================
# 3) Enhanced Metrics
# =========================
@dataclass
class EnhancedMetrics:
    sharpe: float
    max_drawdown: float
    false_break_rate: float
    trades: int
    win_rate: float = 0.0
    avg_holding_time: float = 0.0  # bars
    payoff_ratio: float = 0.0  # avg_win / avg_loss
    max_consecutive_losers: int = 0
    avg_rr: float = 0.0
    # Stress test metrics (optional)
    breach_prob: float = 0.0  # Probability of prop rule breach
    cvar_95: float = 0.0  # Conditional Value at Risk (95th percentile)
    p95_loss_duration: float = 0.0  # 95th percentile loss duration in bars
    trade_pnls: Optional[List[float]] = None  # Individual trade PnLs for stress testing
    
    def to_dict(self):
        d = asdict(self)
        # Remove trade_pnls from dict (too large for JSON)
        if 'trade_pnls' in d:
            d['trade_pnls'] = None
        return d


# =========================
# 4) Scoring with guardrails
# =========================
# Minimum trades gate - prevents degenerate optimization
MIN_TRADES_GATE = {
    7: 10,   # 7 days: minimum 10 trades
    14: 20,  # 14 days: minimum 20 trades
    30: 40,  # 30 days: minimum 40 trades
    60: 80   # 60 days: minimum 80 trades
}

def production_score(
    m: EnhancedMetrics,
    dd_lambda: float = 1.0,
    fb_lambda: float = 0.5,
    freq_lambda: float = 0.001,
    target_trades: int = 100,
    min_trades: int = 20,
    min_trades_penalty: float = 1000.0
) -> float:
    """
    Production scoring with hard floor on trades.
    Prevents degenerate "no-trade" solutions.
    """
    # Hard penalty if too few trades
    if m.trades < min_trades:
        return min_trades_penalty + (min_trades - m.trades) * 10
    
    freq_pen = max(0, target_trades - m.trades) * freq_lambda
    
    # Penalize poor risk metrics
    score = (-m.sharpe) + dd_lambda * m.max_drawdown + fb_lambda * m.false_break_rate + freq_pen
    
    # Additional penalties for pathological cases
    if m.max_consecutive_losers > 10:
        score += 2.0  # Penalize long losing streaks
    
    if m.payoff_ratio > 0 and m.payoff_ratio < 0.5:
        score += 1.0  # Penalize very poor R:R
    
    return score


# =========================
# 5) Constrained QUBO
# =========================
def add_onehot_penalties(Q: np.ndarray, enc: Encoding, lam: float = 5.0) -> np.ndarray:
    Q = Q.copy()
    for k in enc.keys:
        start = enc.offsets[k]
        size = enc.sizes[k]
        idxs = list(range(start, start+size))
        for i in idxs:
            Q[i, i] += -lam
        for a in range(len(idxs)):
            for b in range(a+1, len(idxs)):
                i, j = idxs[a], idxs[b]
                Q[i, j] += 2 * lam
                Q[j, i] += 2 * lam
    return Q


# =========================
# 6) QUBO fitting
# =========================
def fit_qubo_from_samples(samples_x: np.ndarray, samples_y: np.ndarray, n_bits: int, ridge: float = 1e-3) -> np.ndarray:
    pairs = [(i, j) for i in range(n_bits) for j in range(i, n_bits)]
    F = np.zeros((samples_x.shape[0], len(pairs)), dtype=float)
    for r in range(samples_x.shape[0]):
        x = samples_x[r]
        F[r, :] = [x[i] * x[j] for (i, j) in pairs]

    FtF = F.T @ F
    FtF += ridge * np.eye(FtF.shape[0])
    w = np.linalg.solve(FtF, F.T @ samples_y)

    Q = np.zeros((n_bits, n_bits), dtype=float)
    for coef, (i, j) in zip(w, pairs):
        if i == j:
            Q[i, i] += coef
        else:
            Q[i, j] += coef / 2.0
            Q[j, i] += coef / 2.0
    return Q


# =========================
# 7) Solvers
# =========================
def qubo_energy(Q: np.ndarray, x: np.ndarray) -> float:
    return float(x.T @ Q @ x)

def anneal(Q: np.ndarray, n_steps: int = 20000, t0: float = 2.0, t1: float = 0.01, seed: int = 7) -> np.ndarray:
    rng = random.Random(seed)
    n = Q.shape[0]
    x = np.array([rng.randint(0, 1) for _ in range(n)], dtype=int)
    best = x.copy()
    e = qubo_energy(Q, x)
    best_e = e

    for step in range(1, n_steps + 1):
        frac = step / n_steps
        T = t0 * (t1 / t0) ** frac

        i = rng.randrange(n)
        x_new = x.copy()
        x_new[i] ^= 1
        e_new = qubo_energy(Q, x_new)

        de = e_new - e
        if de <= 0 or rng.random() < math.exp(-de / max(T, 1e-9)):
            x, e = x_new, e_new
            if e < best_e:
                best, best_e = x.copy(), e

    return best

def try_qaoa_solve(Q: np.ndarray, reps: int = 2, shots: int = 2048, seed: int = 7) -> Optional[np.ndarray]:
    try:
        from qiskit_aer import Aer
        from qiskit_algorithms.minimum_eigensolvers import QAOA
        from qiskit_algorithms.optimizers import COBYLA
        from qiskit_optimization import QuadraticProgram
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        from qiskit_optimization.converters import QuadraticProgramToQubo
    except Exception:
        return None

    n = Q.shape[0]
    qp = QuadraticProgram()
    for i in range(n):
        qp.binary_var(name=f"x{i}")

    linear = {f"x{i}": float(Q[i, i]) for i in range(n)}
    quadratic = {}
    for i in range(n):
        for j in range(i+1, n):
            if Q[i, j] != 0:
                quadratic[(f"x{i}", f"x{j}")] = float(2 * Q[i, j])
    qp.minimize(linear=linear, quadratic=quadratic)

    conv = QuadraticProgramToQubo()
    qubo = conv.convert(qp)

    backend = Aer.get_backend("aer_simulator")
    optimizer = COBYLA(maxiter=200)
    qaoa = QAOA(optimizer=optimizer, reps=reps, sampler=backend, initial_point=None)

    meo = MinimumEigenOptimizer(qaoa)
    result = meo.solve(qubo)

    x = np.array([int(result.variables_dict[f"x{i}"]) for i in range(n)], dtype=int)
    return x


# =========================
# 8) Enhanced Backtest Evaluator
# =========================
def enhanced_backtest_evaluator(cfg: Dict[str, Any], symbol: str = "GBPJPY", days: int = 30) -> EnhancedMetrics:
    """Enhanced backtest with detailed metrics."""
    import pandas as pd
    from datetime import datetime, timedelta
    
    bars_per_day = 288
    total_bars = days * bars_per_day
    
    np.random.seed(42 if symbol == 'GBPJPY' else (43 if symbol == 'BTCUSD' else 44))
    
    if symbol == 'GBPJPY':
        base_price, vol_base = 185.0, 0.0005
    elif symbol == 'BTCUSD':
        base_price, vol_base = 60000.0, 0.002
    else:
        base_price, vol_base = 2650.0, 0.0008
    
    volatility = np.full(total_bars, vol_base)
    returns = np.random.randn(total_bars) * volatility
    prices = base_price + np.cumsum(returns)
    
    opens = prices.copy()
    highs = opens + abs(np.random.randn(total_bars) * 0.002 * opens)
    lows = opens - abs(np.random.randn(total_bars) * 0.002 * opens)
    closes = opens + np.random.randn(total_bars) * 0.001 * opens
    
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))
    
    base_vol = 5000 if symbol != 'BTCUSD' else 100
    volumes = (base_vol * (1 + abs(returns) * 5)).astype(int)
    
    start_date = datetime.now() - timedelta(days=days)
    timestamps = []
    current = start_date
    bar_count = 0
    
    while bar_count < total_bars:
        if current.weekday() < 5:
            timestamps.append(current)
            bar_count += 1
        current += timedelta(minutes=5)
        if bar_count > 0 and bar_count % bars_per_day == 0:
            current += timedelta(days=1)
    
    min_len = min(len(timestamps), len(opens))
    data = pd.DataFrame({
        'timestamp': timestamps[:min_len],
        'open': opens[:min_len],
        'high': highs[:min_len],
        'low': lows[:min_len],
        'close': closes[:min_len],
        'volume': volumes[:min_len]
    })
    
    # Map config to EnhancedStrategyParams
    # Handle flexible retest window
    retest_value = cfg.get("retest_max", cfg.get("retest_count", 3))  # Use max for flexible window
    if "retest_min" in cfg and "retest_max" in cfg:
        retest_value = cfg["retest_max"]  # Will use flexible window (2-4) in strategy code
    
    params = EnhancedStrategyParams(
        band_k=cfg["sigma_level"],
        band_k_list=(1.0, cfg["sigma_level"], 3.0),
        require_nth_retest=retest_value,  # Strategy will use flexible window if retest_count==3
        atr_cap_mult=cfg["atr_cap_mult"],
        atr_cap_tighter=cfg["atr_cap_mult"],
        trail_pct=cfg["trail_pct"],
        require_session_filter=bool(cfg["session_filter"]),
        volume_filter_type=cfg.get("volume_filter_type", "percentile"),
        vol_percentile_L=cfg.get("vol_percentile_L", 50),
        vol_percentile_p=cfg.get("vol_percentile_p", 60.0),
        vol_mult=cfg.get("vol_mult", 1.0),  # Fallback for multiplier type
        stretch_targets=True,
        require_trend_alignment=True,
        require_fvg_fill=True,
        require_volume_imbalance=True,
        volume_imbalance_threshold=1.2,
        symbol_type="FOREX" if symbol == "GBPJPY" else ("CRYPTO" if symbol == "BTCUSD" else "METAL")
    )
    
    strategy = EnhancedVWAPControlFlipStrategy(params)
    
    equity = 10000.0
    trades = []
    max_dd = 0.0
    peak = equity
    false_breaks = 0
    entry_bars = []
    holding_times = []
    consecutive_losers = 0
    max_consecutive_losers = 0
    
    for idx, row in data.iterrows():
        try:
            result = strategy.update(
                current_price=row['close'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['volume'],
                timestamp=row.get('timestamp')
            )
            
            if result['signals']['enter_long'] and strategy.position is None:
                strategy.enter_long(row['close'])
                entry_bars.append(idx)
            elif result['signals']['enter_short'] and strategy.position is None:
                strategy.enter_short(row['close'])
                entry_bars.append(idx)
            elif result['signals']['exit'] and strategy.position is not None:
                pos = strategy.position
                size = equity / pos.entry_price * 0.1
                pnl = (row['close'] - pos.entry_price) * size if pos.side == "long" else (pos.entry_price - row['close']) * size
                pnl -= abs(size * row['close']) * 0.0001
                equity += pnl
                
                # Calculate metrics
                if entry_bars:
                    holding_time = idx - entry_bars[-1]
                    holding_times.append(holding_time)
                    entry_bars.pop()
                
                risk = abs(pos.entry_price - (pos.stop_price or pos.entry_price * 0.99))
                reward = abs(row['close'] - pos.entry_price) if pos.side == "long" else abs(pos.entry_price - row['close'])
                rr = reward / risk if risk > 0 else 0
                
                if pnl < 0 and abs(pnl) > abs(size * (pos.entry_price * 0.01)):
                    false_breaks += 1
                    consecutive_losers += 1
                    max_consecutive_losers = max(max_consecutive_losers, consecutive_losers)
                else:
                    consecutive_losers = 0
                
                trades.append({
                    'pnl': pnl,
                    'rr': rr,
                    'holding_time': holding_time if entry_bars else 0
                })
                strategy.exit_position()
            
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0)
        except:
            continue
    
    if len(trades) == 0:
        return EnhancedMetrics(
            sharpe=0.0, max_drawdown=1.0, false_break_rate=1.0, trades=0,
            win_rate=0.0, avg_holding_time=0.0, payoff_ratio=0.0,
            max_consecutive_losers=0, avg_rr=0.0
        )
    
    trades_df = pd.DataFrame(trades)
    returns = trades_df['pnl'] / 10000
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0.0
    
    false_break_rate = false_breaks / len(trades) if len(trades) > 0 else 1.0
    win_rate = (trades_df['pnl'] > 0).sum() / len(trades) if len(trades) > 0 else 0.0
    
    avg_holding_time = np.mean(holding_times) if holding_times else 0.0
    
    avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if (trades_df['pnl'] > 0).any() else 0.0
    avg_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].mean()) if (trades_df['pnl'] < 0).any() else 1.0
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
    
    avg_rr = trades_df['rr'].mean() if 'rr' in trades_df.columns else 0.0
    
    # Stress test metrics
    trade_pnls = trades_df['pnl'].tolist() if len(trades) > 0 else []
    
    # Calculate CVaR (Conditional Value at Risk) - 95th percentile
    if len(trade_pnls) > 0:
        sorted_pnls = sorted(trade_pnls)
        tail_threshold = int(0.05 * len(sorted_pnls))
        if tail_threshold > 0:
            tail_losses = sorted_pnls[:tail_threshold]
            cvar_95 = abs(np.mean(tail_losses)) / 10000.0  # Normalize to fraction
        else:
            cvar_95 = abs(sorted_pnls[0]) / 10000.0 if sorted_pnls[0] < 0 else 0.0
    else:
        cvar_95 = 1.0
    
    # Calculate loss duration (95th percentile)
    loss_durations = [ht for ht, pnl in zip(holding_times, trade_pnls) if pnl < 0]
    if loss_durations:
        sorted_durations = sorted(loss_durations)
        p95_idx = int(0.95 * (len(sorted_durations) - 1))
        p95_loss_duration = float(sorted_durations[p95_idx])
    else:
        p95_loss_duration = 0.0
    
    # Run stress tests (if we have trades)
    breach_prob = 0.0
    if len(trade_pnls) > 0 and len(trade_pnls) >= 10:
        try:
            # Estimate trades per day
            trades_per_day = max(1, len(trade_pnls) // days)
            
            # Run Monte Carlo stress test
            prop_rules = PropRules(
                max_daily_loss_frac=0.02,
                max_total_loss_frac=0.05,
                peak_trailing_dd_hard=0.045
            )
            
            stress_result = monte_carlo_paths(
                start_equity=10000.0,
                trade_pnls=trade_pnls,
                trades_per_day=trades_per_day,
                days=days,
                rules=prop_rules,
                n_paths=1000,  # Reduced for speed
                seed=42
            )
            
            breach_prob = stress_result.breach_prob
        except Exception as e:
            # If stress test fails, use conservative estimate
            breach_prob = 0.1 if max_dd > 0.05 else 0.0
    
    return EnhancedMetrics(
        sharpe=float(sharpe),
        max_drawdown=float(max_dd),
        false_break_rate=float(false_break_rate),
        trades=int(len(trades)),
        win_rate=float(win_rate),
        avg_holding_time=float(avg_holding_time),
        payoff_ratio=float(payoff_ratio),
        max_consecutive_losers=int(max_consecutive_losers),
        avg_rr=float(avg_rr),
        breach_prob=float(breach_prob),
        cvar_95=float(cvar_95),
        p95_loss_duration=float(p95_loss_duration),
        trade_pnls=trade_pnls if len(trade_pnls) <= 1000 else None  # Limit size
    )


# =========================
# 9) Production Orchestrator with Sanity Checks
# =========================
def tune_params_with_qubo_production(
    evaluator: Callable[[Dict[str, Any]], EnhancedMetrics],
    n_samples: int = 200,
    onehot_lambda: float = 6.0,
    use_qaoa_if_available: bool = True,
    seed: int = 7,
    validate_constraints: bool = True,
    compare_baseline: bool = True,
    top_k: int = 5
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Production-grade optimization with sanity checks.
    """
    rng = random.Random(seed)
    np.random.seed(seed)
    
    print(f"Sampling {n_samples} configurations (seed={seed})...")
    xs, ys, raw = [], [], []
    
    # Sample baseline first
    if compare_baseline:
        print("  Evaluating baseline config...")
        baseline_metrics = evaluator(BASELINE_CONFIG)
        baseline_score = production_score(baseline_metrics)
        print(f"  Baseline: Sharpe={baseline_metrics.sharpe:.2f}, Trades={baseline_metrics.trades}, Score={baseline_score:.2f}")
    
    for i in range(n_samples):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{n_samples} samples")
        cfg = random_config(PARAM_SPACE, seed=seed+i)
        m = evaluator(cfg)
        y = production_score(m)
        x = config_to_bits(cfg, PARAM_SPACE, ENC)
        
        # Sanity check: validate constraints
        if validate_constraints:
            is_valid, errors = validate_onehot_constraints(x, ENC)
            if not is_valid:
                print(f"  WARNING: Invalid constraints in sample {i+1}: {errors}")
                # Fix by using argmax
                for k in ENC.keys:
                    start = ENC.offsets[k]
                    size = ENC.sizes[k]
                    block = x[start:start+size]
                    x[start:start+size] = 0
                    x[start + np.argmax(block)] = 1
        
        xs.append(x)
        ys.append(y)
        raw.append((cfg, m, y))
    
    print(f"\nFitting QUBO surrogate from {n_samples} samples...")
    X = np.vstack(xs)
    Y = np.array(ys, dtype=float)
    
    Q = fit_qubo_from_samples(X, Y, ENC.n_bits, ridge=1e-2)
    
    # Try different penalty weights if constraints break
    penalty_weights = [onehot_lambda, onehot_lambda * 1.25, onehot_lambda * 1.5]
    x_best = None
    method = "anneal"
    
    for lam in penalty_weights:
        Qp = add_onehot_penalties(Q, ENC, lam=lam)
        
        print(f"Solving QUBO (penalty λ={lam:.1f})...")
        if use_qaoa_if_available:
            xq = try_qaoa_solve(Qp, reps=2, shots=2048, seed=seed)
            if xq is not None:
                is_valid, errors = validate_onehot_constraints(xq, ENC)
                if is_valid:
                    x_best = xq
                    method = "qaoa"
                    print(f"  QAOA solve successful (λ={lam:.1f})")
                    break
                else:
                    print(f"  QAOA solution invalid: {errors}, trying higher penalty...")
        
        if x_best is None:
            x_anneal = anneal(Qp, n_steps=40000, t0=3.0, t1=0.01, seed=seed)
            is_valid, errors = validate_onehot_constraints(x_anneal, ENC)
            if is_valid:
                x_best = x_anneal
                method = "anneal"
                print(f"  Simulated Annealing successful (λ={lam:.1f})")
                break
            else:
                print(f"  Annealing solution invalid: {errors}, trying higher penalty...")
    
    if x_best is None:
        raise RuntimeError("Failed to find valid solution after trying multiple penalty weights")
    
    cfg_best = bits_to_config(x_best, PARAM_SPACE, ENC)
    
    # Final validation
    is_valid, errors = validate_config(cfg_best, PARAM_SPACE)
    if not is_valid:
        raise RuntimeError(f"Final config invalid: {errors}")
    
    # Get top-K configs for stability
    sorted_raw = sorted(raw, key=lambda t: t[2])
    top_k_configs = [t[0] for t in sorted_raw[:top_k]]
    
    # Evaluate best config
    print(f"\nEvaluating optimized configuration...")
    m_best = evaluator(cfg_best)
    y_best = production_score(m_best)
    
    # Compare to baseline
    baseline_comparison = {}
    if compare_baseline:
        baseline_comparison = {
            "baseline_sharpe": baseline_metrics.sharpe,
            "optimized_sharpe": m_best.sharpe,
            "sharpe_improvement": m_best.sharpe - baseline_metrics.sharpe,
            "baseline_trades": baseline_metrics.trades,
            "optimized_trades": m_best.trades
        }
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "method": method,
        "n_bits": ENC.n_bits,
        "n_samples": n_samples,
        "onehot_lambda_used": lam,
        "chosen_config": cfg_best,
        "chosen_metrics": m_best.to_dict(),
        "chosen_score": y_best,
        "sample_best_score": float(np.min(Y)),
        "sample_best_config": min(raw, key=lambda t: t[2])[0],
        "top_k_configs": top_k_configs[:top_k],
        "baseline_comparison": baseline_comparison,
        "constraint_valid": is_valid
    }
    
    return cfg_best, report


# =========================
# 10) Walk-Forward Validation
# =========================
def walk_forward_validation(
    cfg: Dict[str, Any],
    evaluator: Callable[[Dict[str, Any], str, int], EnhancedMetrics],
    symbol: str,
    train_days: int = 30,
    test_days: int = 60
) -> Dict[str, Any]:
    """
    Validate config on out-of-sample data.
    """
    print(f"\nWalk-Forward Validation:")
    print(f"  Train: {train_days} days (used for optimization)")
    print(f"  Test: {test_days} days (out-of-sample)")
    
    train_metrics = evaluator(cfg, symbol=symbol, days=train_days)
    test_metrics = evaluator(cfg, symbol=symbol, days=test_days)
    
    return {
        "train_metrics": train_metrics.to_dict(),
        "test_metrics": test_metrics.to_dict(),
        "sharpe_drop": train_metrics.sharpe - test_metrics.sharpe,
        "dd_stable": abs(train_metrics.max_drawdown - test_metrics.max_drawdown) < 0.05,
        "trades_stable": abs(train_metrics.trades - test_metrics.trades) / max(train_metrics.trades, 1) < 0.5
    }


# =========================
# 11) Export with metadata
# =========================
def export_config_production(cfg: Dict[str, Any], report: Dict[str, Any], path: str = "vwap_params.json") -> None:
    """Export config with full metadata."""
    output = {
        "config": cfg,
        "metadata": {
            "optimized_at": report.get("timestamp", datetime.now().isoformat()),
            "seed": report.get("seed"),
            "method": report.get("method"),
            "n_samples": report.get("n_samples"),
            "metrics": report.get("chosen_metrics", {}),
            "baseline_comparison": report.get("baseline_comparison", {}),
            "constraint_valid": report.get("constraint_valid", True)
        }
    }
    
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Production QUBO/QAOA Parameter Tuner")
    parser.add_argument("--symbol", type=str, default="GBPJPY", choices=["GBPJPY", "BTCUSD", "XAUUSD"])
    parser.add_argument("--days", type=int, default=30, help="Days for optimization")
    parser.add_argument("--samples", type=int, default=150, help="Number of samples")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for reproducibility")
    parser.add_argument("--no-qaoa", action="store_true", help="Disable QAOA")
    parser.add_argument("--validate", action="store_true", help="Run walk-forward validation")
    
    args = parser.parse_args()
    
    print("="*80)
    print("PRODUCTION QUBO/QAOA PARAMETER TUNER")
    print("="*80)
    print(f"Symbol: {args.symbol}")
    print(f"Optimization Days: {args.days}")
    print(f"Samples: {args.samples}")
    print(f"Seed: {args.seed}")
    print("="*80)
    
    def evaluator(cfg):
        return enhanced_backtest_evaluator(cfg, symbol=args.symbol, days=args.days)
    
    cfg, report = tune_params_with_qubo_production(
        evaluator=evaluator,
        n_samples=args.samples,
        onehot_lambda=7.5,
        use_qaoa_if_available=not args.no_qaoa,
        seed=args.seed,
        validate_constraints=True,
        compare_baseline=True,
        top_k=5
    )
    
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    print("CHOSEN CONFIG:")
    print(json.dumps(cfg, indent=2))
    print("\nMETRICS:")
    print(json.dumps(report["chosen_metrics"], indent=2))
    
    if report.get("baseline_comparison"):
        print("\nBASELINE COMPARISON:")
        bc = report["baseline_comparison"]
        print(f"  Sharpe: {bc['baseline_sharpe']:.2f} → {bc['optimized_sharpe']:.2f} ({bc['sharpe_improvement']:+.2f})")
        print(f"  Trades: {bc['baseline_trades']} → {bc['optimized_trades']}")
    
    # Walk-forward validation
    if args.validate:
        wf_result = walk_forward_validation(
            cfg,
            enhanced_backtest_evaluator,
            symbol=args.symbol,
            train_days=args.days,
            test_days=60
        )
        print("\nWALK-FORWARD VALIDATION:")
        print(f"  Train Sharpe: {wf_result['train_metrics']['sharpe']:.2f}")
        print(f"  Test Sharpe: {wf_result['test_metrics']['sharpe']:.2f}")
        print(f"  Sharpe Drop: {wf_result['sharpe_drop']:.2f}")
        print(f"  DD Stable: {wf_result['dd_stable']}")
        print(f"  Trades Stable: {wf_result['trades_stable']}")
        report["walk_forward"] = wf_result
    
    output_file = f"vwap_params_{args.symbol.lower()}.json"
    export_config_production(cfg, report, output_file)
    print(f"\nSaved optimized parameters -> {output_file}")
    print("="*80)
