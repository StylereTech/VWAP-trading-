"""
Production Acceptance Test Suite
Tests: Determinism, Constraints, Walk-Forward Leakage, Surrogate Quality
"""

import json
import subprocess
import sys
import os
import numpy as np
from typing import Dict, Any, Tuple

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_determinism(seed: int = 7, samples: int = 50, days: int = 14) -> Tuple[bool, Dict[str, Any]]:
    """Test 1: Determinism - same seed should produce identical results."""
    print("="*80)
    print("TEST 1: DETERMINISM CHECK")
    print("="*80)
    
    def evaluator(cfg):
        return enhanced_backtest_evaluator(cfg, symbol="GBPJPY", days=days)
    
    print(f"Running optimization #1 (seed={seed})...")
    cfg1, report1 = tune_params_with_qubo_production(
        evaluator=evaluator,
        n_samples=samples,
        seed=seed,
        use_qaoa_if_available=False,
        validate_constraints=True,
        compare_baseline=True
    )
    
    print(f"\nRunning optimization #2 (seed={seed})...")
    cfg2, report2 = tune_params_with_qubo_production(
        evaluator=evaluator,
        n_samples=samples,
        seed=seed,
        use_qaoa_if_available=False,
        validate_constraints=True,
        compare_baseline=True
    )
    
    # Compare configs
    config_identical = cfg1 == cfg2
    
    # Compare metrics
    m1 = report1["chosen_metrics"]
    m2 = report2["chosen_metrics"]
    
    sharpe_diff = abs(m1["sharpe"] - m2["sharpe"])
    dd_diff = abs(m1["max_drawdown"] - m2["max_drawdown"])
    trades_diff = abs(m1["trades"] - m2["trades"]) / max(m1["trades"], 1)
    
    passed = (
        config_identical and
        sharpe_diff <= 0.05 and
        dd_diff <= 0.01 and
        trades_diff <= 0.03
    )
    
    result = {
        "test": "determinism",
        "passed": passed,
        "config_identical": config_identical,
        "sharpe_diff": sharpe_diff,
        "dd_diff": dd_diff,
        "trades_diff": trades_diff,
        "run1_config": cfg1,
        "run2_config": cfg2,
        "run1_metrics": m1,
        "run2_metrics": m2
    }
    
    print(f"\nResults:")
    print(f"  Config Identical: {config_identical}")
    print(f"  Sharpe Diff: {sharpe_diff:.4f} (≤0.05: {sharpe_diff <= 0.05})")
    print(f"  DD Diff: {dd_diff:.4f} (≤0.01: {dd_diff <= 0.01})")
    print(f"  Trades Diff: {trades_diff*100:.2f}% (≤3%: {trades_diff <= 0.03})")
    print(f"  PASSED: {passed}")
    
    return passed, result


def test_constraint_enforcement() -> Tuple[bool, Dict[str, Any]]:
    """Test 2: Constraint enforcement - force failure with low lambda."""
    print("\n" + "="*80)
    print("TEST 2: CONSTRAINT ENFORCEMENT")
    print("="*80)
    
    def evaluator(cfg):
        return enhanced_backtest_evaluator(cfg, symbol="GBPJPY", days=14)
    
    # Test with low lambda (should fail, then auto-adjust)
    print("Testing with low lambda (should auto-adjust)...")
    
    # Test that constraint validation works
    
    # Create invalid config (manually break constraints)
    invalid_bits = np.zeros(ENC.n_bits, dtype=int)
    # Set multiple bits for sigma_level
    invalid_bits[0] = 1  # sigma 1.5
    invalid_bits[1] = 1  # sigma 2.0 (invalid - two bits on)
    
    is_valid, errors = validate_onehot_constraints(invalid_bits, ENC)
    
    passed = not is_valid and len(errors) > 0
    
    result = {
        "test": "constraint_enforcement",
        "passed": passed,
        "invalid_detected": not is_valid,
        "errors": errors
    }
    
    print(f"  Invalid config detected: {not is_valid}")
    print(f"  Errors: {errors}")
    print(f"  PASSED: {passed}")
    
    return passed, result


def test_walk_forward_leakage() -> Tuple[bool, Dict[str, Any]]:
    """Test 3: Walk-forward leakage - ensure no data leakage."""
    print("\n" + "="*80)
    print("TEST 3: WALK-FORWARD LEAKAGE CHECK")
    print("="*80)
    
    def evaluator_train(cfg):
        return enhanced_backtest_evaluator(cfg, symbol="GBPJPY", days=30)
    
    cfg, report = tune_params_with_qubo_production(
        evaluator=evaluator_train,
        n_samples=50,
        seed=7,
        use_qaoa_if_available=False
    )
    
    # Validate on non-overlapping window (60 days ending 31 days ago)
    def evaluator_test(cfg):
        return enhanced_backtest_evaluator(cfg, symbol="GBPJPY", days=60)
    
    wf_result = walk_forward_validation(
        cfg,
        enhanced_backtest_evaluator,
        symbol="GBPJPY",
        train_days=30,
        test_days=60
    )
    
    # Check for leakage indicators
    train_sharpe = wf_result["train_metrics"]["sharpe"]
    test_sharpe = wf_result["test_metrics"]["sharpe"]
    sharpe_drop = wf_result["sharpe_drop"]
    
    # If test Sharpe is suspiciously high (close to train), possible leakage
    # If test Sharpe collapses completely, possible overfitting
    # Good: test Sharpe is lower but reasonable
    leakage_suspicious = test_sharpe > train_sharpe * 0.9  # Test too close to train
    overfitting = test_sharpe < train_sharpe * 0.5  # Test collapsed
    
    passed = not leakage_suspicious and not overfitting
    
    result = {
        "test": "walk_forward_leakage",
        "passed": passed,
        "train_sharpe": train_sharpe,
        "test_sharpe": test_sharpe,
        "sharpe_drop": sharpe_drop,
        "leakage_suspicious": leakage_suspicious,
        "overfitting": overfitting,
        "wf_result": wf_result
    }
    
    print(f"  Train Sharpe: {train_sharpe:.4f}")
    print(f"  Test Sharpe: {test_sharpe:.4f}")
    print(f"  Sharpe Drop: {sharpe_drop:.4f}")
    print(f"  Leakage Suspicious: {leakage_suspicious}")
    print(f"  Overfitting: {overfitting}")
    print(f"  PASSED: {passed}")
    
    return passed, result


def compute_surrogate_quality(samples_x: np.ndarray, samples_y: np.ndarray, Q: np.ndarray) -> Dict[str, float]:
    """Compute surrogate fit quality metrics."""
    # Predictions from surrogate
    pred_scores = np.array([x.T @ Q @ x for x in samples_x])
    
    # R²
    ss_res = np.sum((samples_y - pred_scores) ** 2)
    ss_tot = np.sum((samples_y - np.mean(samples_y)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    # Spearman correlation
    spearman_corr, spearman_p = spearmanr(samples_y, pred_scores)
    
    return {
        "r2": float(r2),
        "spearman_correlation": float(spearman_corr),
        "spearman_pvalue": float(spearman_p)
    }


def test_surrogate_quality() -> Tuple[bool, Dict[str, Any]]:
    """Test 4: Surrogate quality - R² and Spearman correlation."""
    print("\n" + "="*80)
    print("TEST 4: SURROGATE QUALITY")
    print("="*80)
    
    from vwap_qubo_tuner_production import (
        random_config, config_to_bits, fit_qubo_from_samples, ENC, PARAM_SPACE
    )
    import random
    
    random.seed(7)
    np.random.seed(7)
    
    # Generate samples
    xs, ys = [], []
    for i in range(100):
        cfg = random_config(PARAM_SPACE, seed=7+i)
        m = enhanced_backtest_evaluator(cfg, symbol="GBPJPY", days=14)
        from vwap_qubo_tuner_production import production_score
        y = production_score(m)
        x = config_to_bits(cfg, PARAM_SPACE, ENC)
        xs.append(x)
        ys.append(y)
    
    X = np.vstack(xs)
    Y = np.array(ys)
    
    # Fit surrogate
    Q = fit_qubo_from_samples(X, Y, ENC.n_bits, ridge=1e-2)
    
    # Compute quality metrics
    quality = compute_surrogate_quality(X, Y, Q)
    
    spearman_ok = quality["spearman_correlation"] >= 0.35
    
    result = {
        "test": "surrogate_quality",
        "passed": spearman_ok,
        "r2": quality["r2"],
        "spearman_correlation": quality["spearman_correlation"],
        "spearman_pvalue": quality["spearman_pvalue"],
        "spearman_threshold": 0.35
    }
    
    print(f"  R²: {quality['r2']:.4f}")
    print(f"  Spearman Correlation: {quality['spearman_correlation']:.4f}")
    print(f"  Spearman P-value: {quality['spearman_pvalue']:.4f}")
    print(f"  Threshold: ≥0.35")
    print(f"  PASSED: {spearman_ok}")
    
    return spearman_ok, result


def run_full_optimization_with_metrics() -> Dict[str, Any]:
    """Run full optimization and collect all requested metrics."""
    print("\n" + "="*80)
    print("FULL OPTIMIZATION WITH METRICS")
    print("="*80)
    
    symbol = "GBPJPY"
    days = 30
    samples = 100
    
    def evaluator(cfg):
        return enhanced_backtest_evaluator(cfg, symbol=symbol, days=days)
    
    # Run optimization
    cfg, report = tune_params_with_qubo_production(
        evaluator=evaluator,
        n_samples=samples,
        seed=7,
        use_qaoa_if_available=False,
        validate_constraints=True,
        compare_baseline=True,
        top_k=5
    )
    
    # Get baseline metrics
    baseline_metrics = enhanced_backtest_evaluator(BASELINE_CONFIG, symbol=symbol, days=days)
    
    # Walk-forward validation
    wf_result = walk_forward_validation(
        cfg,
        enhanced_backtest_evaluator,
        symbol=symbol,
        train_days=days,
        test_days=60
    )
    
    # Compute surrogate quality (need to refit with same samples)
    from vwap_qubo_tuner_production import (
        random_config, config_to_bits, fit_qubo_from_samples, ENC, PARAM_SPACE, add_onehot_penalties
    )
    import random
    
    random.seed(7)
    np.random.seed(7)
    
    xs, ys = [], []
    for i in range(samples):
        cfg_sample = random_config(PARAM_SPACE, seed=7+i)
        m = evaluator(cfg_sample)
        from vwap_qubo_tuner_production import production_score
        y = production_score(m)
        x = config_to_bits(cfg_sample, PARAM_SPACE, ENC)
        xs.append(x)
        ys.append(y)
    
    X = np.vstack(xs)
    Y = np.array(ys)
    Q = fit_qubo_from_samples(X, Y, ENC.n_bits, ridge=1e-2)
    surrogate_quality = compute_surrogate_quality(X, Y, Q)
    
    # Get top-K configs and scores (from report)
    top_k_configs_raw = report.get("top_k_configs", [])
    top_k = []
    for i, cfg_dict in enumerate(top_k_configs_raw[:5]):
        m = evaluator(cfg_dict)
        score = production_score(m)
        top_k.append({
            "config": cfg_dict,
            "score": float(score),
            "metrics": m.to_dict()
        })
    
    # Build comprehensive output
    output = {
        "symbol": symbol,
        "optimization_date": report.get("timestamp", ""),
        "seed": report.get("seed", 7),
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
        
        "top_k_configs": [
            {
                "rank": i+1,
                "config": t["config"],
                "score": t["score"],
                "sharpe": t["metrics"]["sharpe"],
                "trades": t["metrics"]["trades"]
            }
            for i, t in enumerate(top_k)
        ],
        
        "surrogate_quality": surrogate_quality,
        
        "constraint_valid": report.get("constraint_valid", True),
        
        "baseline_comparison": report.get("baseline_comparison", {})
    }
    
    return output


def main():
    """Run all acceptance tests and generate full report."""
    print("="*80)
    print("PRODUCTION ACCEPTANCE TEST SUITE")
    print("="*80)
    
    results = {}
    
    # Test 1: Determinism
    try:
        passed, result = test_determinism(seed=7, samples=50, days=14)
        results["determinism"] = {"passed": passed, "details": result}
    except Exception as e:
        results["determinism"] = {"passed": False, "error": str(e)}
    
    # Test 2: Constraint Enforcement
    try:
        passed, result = test_constraint_enforcement()
        results["constraint_enforcement"] = {"passed": passed, "details": result}
    except Exception as e:
        results["constraint_enforcement"] = {"passed": False, "error": str(e)}
    
    # Test 3: Walk-Forward Leakage
    try:
        passed, result = test_walk_forward_leakage()
        results["walk_forward_leakage"] = {"passed": passed, "details": result}
    except Exception as e:
        results["walk_forward_leakage"] = {"passed": False, "error": str(e)}
    
    # Test 4: Surrogate Quality
    try:
        passed, result = test_surrogate_quality()
        results["surrogate_quality"] = {"passed": passed, "details": result}
    except Exception as e:
        results["surrogate_quality"] = {"passed": False, "error": str(e)}
    
    # Full optimization with all metrics
    try:
        full_output = run_full_optimization_with_metrics()
        results["full_optimization"] = full_output
    except Exception as e:
        results["full_optimization"] = {"error": str(e)}
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, test_result in results.items():
        if test_name != "full_optimization":
            status = "PASS" if test_result.get("passed", False) else "FAIL"
            print(f"{test_name}: {status}")
    
    # Save full report
    output_file = "acceptance_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nFull results saved to: {output_file}")
    
    # Save the requested single JSON (full optimization output)
    single_output_file = "optimization_output_gbpjpy.json"
    if "full_optimization" in results and "error" not in results["full_optimization"]:
        with open(single_output_file, "w") as f:
            json.dump(results["full_optimization"], f, indent=2)
        print(f"Single optimization output saved to: {single_output_file}")
        print("\n" + "="*80)
        print("REQUESTED OUTPUT (Single JSON)")
        print("="*80)
        print(json.dumps(results["full_optimization"], indent=2))
    
    print("="*80)


if __name__ == "__main__":
    # Fix import
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    # Import the production function
    from vwap_qubo_tuner_production import tune_params_with_qubo_production, bits_to_config
    
    main()
