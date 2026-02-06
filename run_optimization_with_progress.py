"""
Run Optimization with Progress Tracking
Shows progress as it runs
"""

import json
import sys
import os
import numpy as np
from datetime import datetime
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vwap_qubo_tuner_production import (
    tune_params_with_qubo_production,
    enhanced_backtest_evaluator,
    walk_forward_validation,
    PARAM_SPACE,
    BASELINE_CONFIG,
    ENC,
    config_to_bits,
    fit_qubo_from_samples,
    production_score,
    random_config
)
import random


def spearmanr(x, y):
    """Compute Spearman correlation manually."""
    if len(x) < 2:
        return 0.0, 1.0
    x_ranked = np.argsort(np.argsort(x))
    y_ranked = np.argsort(np.argsort(y))
    x_mean = np.mean(x_ranked)
    y_mean = np.mean(y_ranked)
    numerator = np.sum((x_ranked - x_mean) * (y_ranked - y_mean))
    x_std = np.sqrt(np.sum((x_ranked - x_mean) ** 2))
    y_std = np.sqrt(np.sum((y_ranked - y_mean) ** 2))
    if x_std == 0 or y_std == 0:
        return 0.0, 1.0
    corr = numerator / (x_std * y_std)
    p_value = 2 * (1 - abs(corr)) if abs(corr) < 1 else 0.0
    return corr, p_value


def compute_surrogate_quality(samples_x, samples_y, Q):
    """Compute surrogate fit quality metrics."""
    pred_scores = np.array([x.T @ Q @ x for x in samples_x])
    ss_res = np.sum((samples_y - pred_scores) ** 2)
    ss_tot = np.sum((samples_y - np.mean(samples_y)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    spearman_corr, spearman_p = spearmanr(samples_y, pred_scores)
    return {
        "r2": float(r2),
        "spearman_correlation": float(spearman_corr),
        "spearman_pvalue": float(spearman_p)
    }


def main():
    """Generate single optimization output with progress tracking."""
    print("="*80)
    print("OPTIMIZATION WITH PROGRESS TRACKING")
    print("="*80)
    
    symbol = "GBPJPY"
    days = 3  # Ultra-minimal: 3 days = ~864 bars
    samples = 10  # Ultra-minimal samples
    seed = 7
    
    print(f"Symbol: {symbol}")
    print(f"Days: {days} (ultra-minimal for speed)")
    print(f"Samples: {samples} (ultra-minimal)")
    print(f"Seed: {seed}")
    print("="*80)
    
    start_time = time.time()
    
    def evaluator(cfg):
        return enhanced_backtest_evaluator(cfg, symbol=symbol, days=days)
    
    # Get baseline metrics
    print("\n[1/6] Evaluating baseline config...")
    baseline_start = time.time()
    baseline_metrics = enhanced_backtest_evaluator(BASELINE_CONFIG, symbol=symbol, days=days)
    baseline_time = time.time() - baseline_start
    print(f"   Baseline Sharpe: {baseline_metrics.sharpe:.4f}, Trades: {baseline_metrics.trades}")
    print(f"   Time: {baseline_time:.1f}s")
    
    # Run optimization
    print(f"\n[2/6] Running optimization ({samples} samples)...")
    opt_start = time.time()
    cfg, report = tune_params_with_qubo_production(
        evaluator=evaluator,
        n_samples=samples,
        seed=seed,
        use_qaoa_if_available=False,
        validate_constraints=True,
        compare_baseline=True,
        top_k=5
    )
    opt_time = time.time() - opt_start
    print(f"   Chosen config: {cfg}")
    print(f"   Chosen Sharpe: {report['chosen_metrics']['sharpe']:.4f}")
    print(f"   Time: {opt_time:.1f}s")
    
    # Walk-forward validation
    print("\n[3/6] Running walk-forward validation...")
    wf_start = time.time()
    wf_result = walk_forward_validation(
        cfg,
        enhanced_backtest_evaluator,
        symbol=symbol,
        train_days=days,
        test_days=7  # Minimal
    )
    wf_time = time.time() - wf_start
    print(f"   Test Sharpe: {wf_result['test_metrics']['sharpe']:.4f}")
    print(f"   Time: {wf_time:.1f}s")
    
    # Compute surrogate quality
    print("\n[4/6] Computing surrogate quality...")
    sq_start = time.time()
    random.seed(seed)
    np.random.seed(seed)
    
    xs, ys = [], []
    for i in range(samples):
        print(f"   Sample {i+1}/{samples}...", end="\r")
        cfg_sample = random_config(PARAM_SPACE, seed=seed+i)
        m = evaluator(cfg_sample)
        y = production_score(m)
        x = config_to_bits(cfg_sample, PARAM_SPACE, ENC)
        xs.append(x)
        ys.append(y)
    print(f"   Sample {samples}/{samples} complete")
    
    X = np.vstack(xs)
    Y = np.array(ys)
    Q = fit_qubo_from_samples(X, Y, ENC.n_bits, ridge=1e-2)
    surrogate_quality = compute_surrogate_quality(X, Y, Q)
    sq_time = time.time() - sq_start
    print(f"   Surrogate R2: {surrogate_quality['r2']:.4f}")
    print(f"   Surrogate Spearman: {surrogate_quality['spearman_correlation']:.4f}")
    print(f"   Time: {sq_time:.1f}s")
    
    # Get top-K configs
    print("\n[5/6] Evaluating top-K configs...")
    topk_start = time.time()
    top_k_configs_raw = report.get("top_k_configs", [])
    top_k = []
    for i, cfg_dict in enumerate(top_k_configs_raw[:5]):
        m = evaluator(cfg_dict)
        score = production_score(m)
        top_k.append({
            "rank": i+1,
            "config": cfg_dict,
            "score": float(score),
            "sharpe": float(m.sharpe),
            "trades": int(m.trades),
            "max_drawdown": float(m.max_drawdown)
        })
    topk_time = time.time() - topk_start
    print(f"   Time: {topk_time:.1f}s")
    
    # Build output
    print("\n[6/6] Building output JSON...")
    output = {
        "symbol": symbol,
        "optimization_date": datetime.now().isoformat(),
        "seed": seed,
        "method": report.get("method", "anneal"),
        "n_samples": samples,
        "onehot_lambda_used": report.get("onehot_lambda_used", 7.5),
        
        "chosen_config": cfg,
        
        "baseline_config": BASELINE_CONFIG,
        "baseline_metrics": baseline_metrics.to_dict(),
        
        "train_metrics": report["chosen_metrics"],
        
        "validation_metrics": wf_result["test_metrics"],
        "walk_forward": {
            "train_sharpe": wf_result["train_metrics"]["sharpe"],
            "test_sharpe": wf_result["test_metrics"]["sharpe"],
            "sharpe_drop": wf_result["sharpe_drop"],
            "dd_stable": wf_result["dd_stable"],
            "trades_stable": wf_result["trades_stable"]
        },
        
        "top_k_configs": top_k,
        
        "surrogate_quality": surrogate_quality,
        
        "constraint_valid": report.get("constraint_valid", True),
        
        "baseline_comparison": report.get("baseline_comparison", {}),
        
        "timing": {
            "baseline_seconds": baseline_time,
            "optimization_seconds": opt_time,
            "walk_forward_seconds": wf_time,
            "surrogate_quality_seconds": sq_time,
            "top_k_seconds": topk_time,
            "total_seconds": time.time() - start_time
        }
    }
    
    # Save and print
    output_file = "optimization_output_gbpjpy.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    total_time = time.time() - start_time
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE!")
    print("="*80)
    print(f"Total time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
    print(f"\nSaved to: {output_file}")
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Chosen Config: {cfg}")
    print(f"Train Sharpe: {report['chosen_metrics']['sharpe']:.4f}")
    print(f"Test Sharpe: {wf_result['test_metrics']['sharpe']:.4f}")
    print(f"Surrogate R2: {surrogate_quality['r2']:.4f}")
    print(f"Surrogate Spearman: {surrogate_quality['spearman_correlation']:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
