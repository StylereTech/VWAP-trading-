# Production Acceptance Test Suite

## Overview

This document describes the acceptance test suite for validating the production-grade QUBO optimization system. The suite includes 4 critical tests plus a full optimization output generator.

## Tests Implemented

### Test 1: Determinism Check
**Purpose**: Verify that the same seed produces identical results.

**Command**:
```bash
python acceptance_test_suite.py
```

**Pass Conditions**:
- `chosen_config` identical between runs
- `train_metrics.sharpe` diff ≤ 0.05
- `train_metrics.dd` diff ≤ 0.01
- `trades` diff ≤ 3%

**What it tests**:
- Data windowing consistency (timestamps/timezones)
- Indicator calculation consistency (rolling NaNs)
- Backtest randomness/slippage simulation

### Test 2: Constraint Enforcement
**Purpose**: Verify that one-hot encoding constraints are properly enforced.

**How it works**:
- Creates an invalid config (multiple bits set for same parameter)
- Verifies the system detects the violation
- Tests auto-adjustment of penalty λ

**Pass Conditions**:
- Invalid config detected
- System auto-adjusts λ and reruns
- Final JSON shows `constraint_valid=true`

### Test 3: Walk-Forward Leakage Test
**Purpose**: Ensure no data leakage between train and test sets.

**How it works**:
- Optimizes on 30 days ending yesterday
- Validates on 60 days ending 31 days ago (non-overlapping)
- Checks that indicators are computed strictly within each window

**Pass Conditions**:
- No leakage indicators (test Sharpe not suspiciously close to train)
- No overfitting (test Sharpe doesn't collapse completely)
- Indicators computed only within their respective windows

### Test 4: Surrogate Quality Test
**Purpose**: Verify that the quadratic surrogate actually predicts well.

**Metrics Reported**:
- R² on held-out samples
- Spearman rank correlation between predicted and true scores

**Pass Conditions**:
- Spearman ≥ 0.35 (rough heuristic)
- If Spearman ~ 0, surrogate is optimizing noise

**If surrogate is weak**:
- Increase samples
- Reduce parameter space
- Add interaction features
- Or switch to Bayesian/GA

## Full Optimization Output

The `generate_output_ultra_fast.py` script generates a single JSON file (`optimization_output_gbpjpy.json`) containing:

1. **chosen_config**: The optimized parameter configuration
2. **baseline_config** + **baseline_metrics**: Default configuration for comparison
3. **train_metrics**: Performance on training data
4. **validation_metrics** + **walk_forward**: Out-of-sample validation results
5. **top_k_configs**: Top 5 configurations for stability analysis
6. **surrogate_quality**: R² and Spearman correlation metrics
7. **constraint_valid**: Whether constraints were satisfied
8. **baseline_comparison**: Direct comparison to baseline

## Running the Tests

### Quick Test (Ultra-Fast)
```bash
python generate_output_ultra_fast.py
```
- Uses 3 days of data, 10 samples
- Generates JSON structure quickly
- Good for validating format

### Standard Test
```bash
python generate_optimization_output_fast.py
```
- Uses 14 days of data, 50 samples
- More reliable results
- Takes longer but still reasonable

### Full Test Suite
```bash
python acceptance_test_suite.py
```
- Runs all 4 acceptance tests
- Full optimization with comprehensive metrics
- Takes longest but most thorough

## Expected Output Format

The JSON output follows this structure:

```json
{
  "symbol": "GBPJPY",
  "optimization_date": "2026-01-31T...",
  "seed": 7,
  "method": "anneal",
  "n_samples": 10,
  "onehot_lambda_used": 7.5,
  
  "chosen_config": {
    "sigma_level": 2.0,
    "retest_count": 3,
    "atr_cap_mult": 2.5,
    "trail_pct": 0.007,
    "session_filter": 1
  },
  
  "baseline_config": {...},
  "baseline_metrics": {...},
  
  "train_metrics": {
    "sharpe": 0.45,
    "max_drawdown": 0.12,
    "trades": 15,
    "win_rate": 0.40,
    ...
  },
  
  "validation_metrics": {...},
  "walk_forward": {
    "train_sharpe": 0.45,
    "test_sharpe": 0.38,
    "sharpe_drop": 0.07,
    "dd_stable": true,
    "trades_stable": true
  },
  
  "top_k_configs": [
    {
      "rank": 1,
      "config": {...},
      "score": -2.34,
      "sharpe": 0.45,
      "trades": 15
    },
    ...
  ],
  
  "surrogate_quality": {
    "r2": 0.65,
    "spearman_correlation": 0.42,
    "spearman_pvalue": 0.001
  },
  
  "constraint_valid": true,
  "baseline_comparison": {
    "baseline_sharpe": 0.30,
    "optimized_sharpe": 0.45,
    "sharpe_improvement": 0.15,
    ...
  }
}
```

## Additional Failure Mode Protections

Beyond the 4 core tests, the system includes protections against:

1. **Meta-stability**: Top-K configs tracked to prevent single "lucky" sample dominance
2. **Over-penalized false breaks**: Opportunity cost penalties prevent late entries
3. **Regime fragility**: Configs validated across different market regimes

## Troubleshooting

### Timeout Issues
If optimization times out:
- Reduce `days` parameter (try 3-7 days)
- Reduce `n_samples` (try 10-20)
- Check that backtest evaluator isn't stuck in infinite loop

### Low Surrogate Quality
If Spearman < 0.35:
- Increase `n_samples` (try 50-100)
- Check parameter space isn't too large
- Consider reducing number of parameters

### Constraint Violations
If constraints fail:
- System auto-adjusts λ (penalty weight)
- Check that `onehot_lambda_used` increased
- Verify `constraint_valid: true` in final output

## Next Steps

Once you have the JSON output:
1. Review surrogate quality metrics (R², Spearman)
2. Check walk-forward validation (no leakage, stable performance)
3. Compare to baseline (should show improvement)
4. Validate top-K configs are similar (stability check)
5. Deploy if all checks pass
