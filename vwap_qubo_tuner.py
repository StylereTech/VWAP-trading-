"""
QUBO/QAOA Parameter Tuner for VWAP Strategy
Fits quadratic surrogate from backtest samples, optimizes with Simulated Annealing/QAOA
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Callable, Any, Optional

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from enhanced_control_flip_strategy import EnhancedVWAPControlFlipStrategy, EnhancedStrategyParams


# =========================
# 0) Parameter space
# =========================
PARAM_SPACE = {
    "sigma_level": [1.5, 2.0, 2.5, 3.0],         # VWAP stdev band multiplier
    "retest_count": [1, 2, 3],                   # 1/2/3 retests after VWAP reclaim/reject
    "atr_cap_mult": [2.0, 2.5, 3.0],             # ATR cap multiplier
    "trail_pct": [0.005, 0.007, 0.010],          # 0.5%, 0.7%, 1.0%
    "session_filter": [0, 1],                    # off/on
}


# =========================
# 1) Bit encoding (one-hot)
#    Each parameter is one-hot encoded.
#    Total bits = sum(len(options))
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

def random_config(space: Dict[str, List[Any]]) -> Dict[str, Any]:
    return {k: random.choice(v) for k, v in space.items()}


# =========================
# 2) Constrained QUBO via penalties
#    One-hot constraint per parameter:
#      (sum(block) - 1)^2 = sum(xi) + 2*sum_{i<j} xi xj - 2*sum(xi) + 1
#                         = 2*sum_{i<j} xi xj - sum(xi) + 1
# =========================
def add_onehot_penalties(Q: np.ndarray, enc: Encoding, lam: float = 5.0) -> np.ndarray:
    Q = Q.copy()
    for k in enc.keys:
        start = enc.offsets[k]
        size = enc.sizes[k]
        idxs = list(range(start, start+size))
        # -lam * sum(xi) on diagonal
        for i in idxs:
            Q[i, i] += -lam
        # +2*lam * sum_{i<j} xi xj
        for a in range(len(idxs)):
            for b in range(a+1, len(idxs)):
                i, j = idxs[a], idxs[b]
                Q[i, j] += 2 * lam
                Q[j, i] += 2 * lam
    return Q


# =========================
# 3) Build a quadratic surrogate Q from sampled backtests
#    We fit: score(x) ≈ x^T Q x   (minimize)
#    where score = -Sharpe + dd_penalty + false_break_penalty + freq_penalty
#    You provide an evaluator(cfg)->metrics.
# =========================
@dataclass
class Metrics:
    sharpe: float
    max_drawdown: float     # e.g., 0.18 for 18%
    false_break_rate: float # 0..1
    trades: int

def default_score(m: Metrics,
                  dd_lambda: float = 1.0,
                  fb_lambda: float = 0.5,
                  freq_lambda: float = 0.001,
                  target_trades: int = 100) -> float:
    # Minimization objective:
    # -Sharpe is good (lower is better), penalties add cost.
    freq_pen = max(0, target_trades - m.trades)  # penalize too few trades
    return (-m.sharpe) + dd_lambda * m.max_drawdown + fb_lambda * m.false_break_rate + freq_lambda * freq_pen

def fit_qubo_from_samples(samples_x: np.ndarray, samples_y: np.ndarray, n_bits: int, ridge: float = 1e-3) -> np.ndarray:
    """
    Fit quadratic form: y ≈ x^T Q x
    Parameterize Q as symmetric.
    Create features for i<=j: f_ij = x_i * x_j
    Solve ridge regression to get q_ij.
    """
    # build feature matrix
    pairs = [(i, j) for i in range(n_bits) for j in range(i, n_bits)]
    F = np.zeros((samples_x.shape[0], len(pairs)), dtype=float)
    for r in range(samples_x.shape[0]):
        x = samples_x[r]
        F[r, :] = [x[i] * x[j] for (i, j) in pairs]

    # ridge regression: w = (F^T F + ridge I)^-1 F^T y
    FtF = F.T @ F
    FtF += ridge * np.eye(FtF.shape[0])
    w = np.linalg.solve(FtF, F.T @ samples_y)

    Q = np.zeros((n_bits, n_bits), dtype=float)
    for coef, (i, j) in zip(w, pairs):
        if i == j:
            Q[i, i] += coef
        else:
            # because x^T Q x includes both Qij and Qji; keep symmetric
            Q[i, j] += coef / 2.0
            Q[j, i] += coef / 2.0
    return Q


# =========================
# 4) Solvers
# =========================
def qubo_energy(Q: np.ndarray, x: np.ndarray) -> float:
    return float(x.T @ Q @ x)

def anneal(Q: np.ndarray, n_steps: int = 20000, t0: float = 2.0, t1: float = 0.01, seed: int = 7) -> np.ndarray:
    """
    Simulated annealing on binary vector.
    Note: constraints are enforced by penalties already in Q.
    """
    rng = random.Random(seed)
    n = Q.shape[0]
    x = np.array([rng.randint(0, 1) for _ in range(n)], dtype=int)
    best = x.copy()
    e = qubo_energy(Q, x)
    best_e = e

    for step in range(1, n_steps + 1):
        # temperature schedule (log-ish)
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
    """
    Optional: requires qiskit + qiskit-aer.
    Solves QUBO approximately via QAOA on simulator.
    """
    try:
        from qiskit_aer import Aer
        from qiskit import QuantumCircuit
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

    # QUBO objective: minimize x^T Q x
    linear = {f"x{i}": float(Q[i, i]) for i in range(n)}
    quadratic = {}
    for i in range(n):
        for j in range(i+1, n):
            if Q[i, j] != 0:
                quadratic[(f"x{i}", f"x{j}")] = float(2 * Q[i, j])  # qp uses (i,j) once
    qp.minimize(linear=linear, quadratic=quadratic)

    # ensure QUBO form
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
# 5) Orchestrator
# =========================
def tune_params_with_qubo(
    evaluator: Callable[[Dict[str, Any]], Metrics],
    n_samples: int = 200,
    onehot_lambda: float = 6.0,
    use_qaoa_if_available: bool = True,
    seed: int = 7,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    - Sample configs, backtest via evaluator
    - Build surrogate Q
    - Add one-hot penalties
    - Solve via QAOA (optional) else anneal
    - Return best config + report
    """
    rng = random.Random(seed)

    xs, ys, raw = [], [], []
    print(f"Sampling {n_samples} configurations...")
    for i in range(n_samples):
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{n_samples} samples")
        cfg = random_config(PARAM_SPACE)
        m = evaluator(cfg)
        y = default_score(m)
        x = config_to_bits(cfg, PARAM_SPACE, ENC)
        xs.append(x)
        ys.append(y)
        raw.append((cfg, m, y))

    print(f"\nFitting QUBO surrogate from {n_samples} samples...")
    X = np.vstack(xs)
    Y = np.array(ys, dtype=float)

    Q = fit_qubo_from_samples(X, Y, ENC.n_bits, ridge=1e-2)
    Qp = add_onehot_penalties(Q, ENC, lam=onehot_lambda)

    print("Solving QUBO...")
    x_best = None
    method = "anneal"
    if use_qaoa_if_available:
        print("  Attempting QAOA solve...")
        xq = try_qaoa_solve(Qp, reps=2, shots=2048, seed=seed)
        if xq is not None:
            x_best = xq
            method = "qaoa"
            print("  QAOA solve successful!")
        else:
            print("  QAOA not available, using Simulated Annealing")

    if x_best is None:
        print("  Using Simulated Annealing...")
        x_best = anneal(Qp, n_steps=40000, t0=3.0, t1=0.01, seed=seed)

    cfg_best = bits_to_config(x_best, PARAM_SPACE, ENC)

    # Evaluate the chosen config for a final honest score
    print(f"\nEvaluating optimized configuration...")
    m_best = evaluator(cfg_best)
    y_best = default_score(m_best)

    report = {
        "method": method,
        "n_bits": ENC.n_bits,
        "n_samples": n_samples,
        "chosen_config": cfg_best,
        "chosen_metrics": m_best.__dict__,
        "chosen_score": y_best,
        "sample_best_score": float(np.min(Y)),
        "sample_best_config": min(raw, key=lambda t: t[2])[0],
    }
    return cfg_best, report


# =========================
# 6) Output config your bot can load
# =========================
def export_config(cfg: Dict[str, Any], path: str = "vwap_params.json") -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


# =========================
# 7) Real evaluator - integrates with our backtest system
# =========================
def real_backtest_evaluator(cfg: Dict[str, Any], symbol: str = "GBPJPY", days: int = 30) -> Metrics:
    """
    Real backtest evaluator - runs actual strategy backtest.
    Returns Metrics for QUBO optimization.
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Generate test data (smaller dataset for speed)
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
    
    # Convert QUBO config to EnhancedStrategyParams
    params = EnhancedStrategyParams(
        band_k=cfg["sigma_level"],
        band_k_list=(1.0, cfg["sigma_level"], 3.0),
        require_nth_retest=cfg["retest_count"],
        atr_cap_mult=cfg["atr_cap_mult"],
        atr_cap_tighter=cfg["atr_cap_mult"],
        trail_pct=cfg["trail_pct"],
        require_session_filter=bool(cfg["session_filter"]),
        stretch_targets=True,
        require_trend_alignment=True,
        require_fvg_fill=True,
        require_volume_imbalance=True,
        volume_imbalance_threshold=1.2,
        symbol_type="FOREX" if symbol == "GBPJPY" else ("CRYPTO" if symbol == "BTCUSD" else "METAL")
    )
    
    # Run backtest
    strategy = EnhancedVWAPControlFlipStrategy(params)
    
    equity = 10000.0
    trades = []
    max_dd = 0.0
    peak = equity
    false_breaks = 0
    
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
            elif result['signals']['enter_short'] and strategy.position is None:
                strategy.enter_short(row['close'])
            elif result['signals']['exit'] and strategy.position is not None:
                pos = strategy.position
                size = equity / pos.entry_price * 0.1
                pnl = (row['close'] - pos.entry_price) * size if pos.side == "long" else (pos.entry_price - row['close']) * size
                pnl -= abs(size * row['close']) * 0.0001
                equity += pnl
                
                # Track false breaks (trades that lose quickly)
                if pnl < 0 and abs(pnl) > abs(size * (pos.entry_price * 0.01)):  # Lost more than 1%
                    false_breaks += 1
                
                trades.append({'pnl': pnl})
                strategy.exit_position()
            
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak if peak > 0 else 0)
        except:
            continue
    
    if len(trades) == 0:
        return Metrics(sharpe=0.0, max_drawdown=1.0, false_break_rate=1.0, trades=0)
    
    trades_df = pd.DataFrame(trades)
    returns = trades_df['pnl'] / 10000
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252) if len(returns) > 1 and np.std(returns) > 0 else 0.0
    
    false_break_rate = false_breaks / len(trades) if len(trades) > 0 else 1.0
    
    return Metrics(
        sharpe=float(sharpe),
        max_drawdown=float(max_dd),
        false_break_rate=float(false_break_rate),
        trades=int(len(trades))
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="QUBO/QAOA Parameter Tuner for VWAP Strategy")
    parser.add_argument("--symbol", type=str, default="GBPJPY", choices=["GBPJPY", "BTCUSD", "XAUUSD"])
    parser.add_argument("--days", type=int, default=30, help="Days of data for backtest (default: 30)")
    parser.add_argument("--samples", type=int, default=200, help="Number of samples (default: 200)")
    parser.add_argument("--no-qaoa", action="store_true", help="Disable QAOA, use only Simulated Annealing")
    
    args = parser.parse_args()
    
    print("="*80)
    print("QUBO/QAOA PARAMETER TUNER FOR VWAP STRATEGY")
    print("="*80)
    print(f"Symbol: {args.symbol}")
    print(f"Backtest Days: {args.days}")
    print(f"Samples: {args.samples}")
    print("="*80)
    
    # Create evaluator function for this symbol
    def evaluator(cfg):
        return real_backtest_evaluator(cfg, symbol=args.symbol, days=args.days)
    
    cfg, report = tune_params_with_qubo(
        evaluator=evaluator,
        n_samples=args.samples,
        onehot_lambda=7.5,
        use_qaoa_if_available=not args.no_qaoa,
        seed=7,
    )
    
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS")
    print("="*80)
    print("CHOSEN CONFIG:")
    print(json.dumps(cfg, indent=2))
    print("\nMETRICS:")
    print(json.dumps(report["chosen_metrics"], indent=2))
    print(f"\nScore: {report['chosen_score']:.4f}")
    print(f"Method: {report['method']}")
    
    output_file = f"vwap_params_{args.symbol.lower()}.json"
    export_config(cfg, output_file)
    print(f"\nSaved optimized parameters -> {output_file}")
    print("="*80)
