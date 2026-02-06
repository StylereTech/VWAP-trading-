"""
Generate Single Optimization Output JSON
As requested for production validation
"""

import json
import sys
import os
import numpy as np
from datetime import datetime

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
    x_ranked = np.argsort(np.argsort(x))
    y_ranked = np.argsort(np.argsort(y))
    n = len(x)
    if n < 2:
        return 0.0, 1.0
    
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
    """Generate single optimization output with all requested metrics."""
    print("="*80)
    print("GENERATING OPTIMIZATION OUTPUT")
    print("="*80)
    
    symbol = "GBPJPY"
    days = 30
    samples = 100
    seed = 7
    
    print(f"Symbol: {symbol}")
    print(f"Days: {days}")
    print(f"Samples: {samples}")
    print(f"Seed: {seed}")
    print("="*80)
    
    def evaluator(cfg):
        return enhanced_backtest_evaluator(cfg, symbol=symbol, days=days)
    
    # Get baseline metrics
    print("\nEvaluating baseline config...")
    baseline_metrics = enhanced_backtest_evaluator(BASELINE_CONFIG, symbol=symbol, days=days)
    
    # Run optimization
    print(f"\nRunning optimization ({samples} samples)...")
    cfg, report = tune_params_with_qubo_production(
        evaluator=evaluator,
        n_samples=samples,
        seed=seed,
        use_qaoa_if_available=False,
        validate_constraints=True,
        compare_baseline=True,
        top_k=5
    )
    
    # Walk-forward validation
    print("\nRunning walk-forward validation...")
    wf_result = walk_forward_validation(
        cfg,
        enhanced_backtest_evaluator,
        symbol=symbol,
        train_days=days,
        test_days=60
    )
    
    # Compute surrogate quality
    print("\nComputing surrogate quality...")
    random.seed(seed)
    np.random.seed(seed)
    
    xs, ys = [], []
    for i in range(samples):
        cfg_sample = random_config(PARAM_SPACE, seed=seed+i)
        m = evaluator(cfg_sample)
        y = production_score(m)
        x = config_to_bits(cfg_sample, PARAM_SPACE, ENC)
        xs.append(x)
        ys.append(y)
    
    X = np.vstack(xs)
    Y = np.array(ys)
    Q = fit_qubo_from_samples(X, Y, ENC.n_bits, ridge=1e-2)
    surrogate_quality = compute_surrogate_quality(X, Y, Q)
    
    # Get top-K configs
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
    
    # Build output
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
        
        "baseline_comparison": report.get("baseline_comparison", {})
    }
    
    # Save and print
    output_file = "optimization_output_gbpjpy.json"
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "="*80)
    print("OPTIMIZATION OUTPUT")
    print("="*80)
    print(json.dumps(output, indent=2))
    print("="*80)
    print(f"\nSaved to: {output_file}")


if __name__ == "__main__":
    main()
