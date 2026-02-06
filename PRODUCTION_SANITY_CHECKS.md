# Production Sanity Checks & Guardrails

## ✅ 5 Critical Sanity Checks Implemented

### ✅ A. Reproducibility Check
**Implemented**: Fixed seed logging and validation

**How to Run**:
```bash
# Run twice with same seed
python vwap_qubo_tuner_production.py --symbol GBPJPY --seed 7
python vwap_qubo_tuner_production.py --symbol GBPJPY --seed 7
```

**Expected**: Near-identical best config + similar metrics

**What's Logged**:
- Seed stored in JSON metadata
- Timestamp of optimization
- Method used (QAOA or Annealing)

### ✅ B. Constraint Check (One-Hot Validity)
**Implemented**: Automatic validation and penalty adjustment

**Checks**:
- Exactly one value selected per parameter
- No multiple bits on for same parameter
- Auto-fixes invalid samples during optimization

**Penalty Adjustment**:
- Starts with λ = 7.5
- If constraints break, tries λ = 9.4, then λ = 11.25
- Reports which penalty weight succeeded

**Output**: `"constraint_valid": true` in JSON metadata

### ✅ C. Overfitting Check (Walk-Forward)
**Implemented**: Separate train/test validation

**How to Run**:
```bash
python vwap_qubo_tuner_production.py --symbol GBPJPY --validate
```

**Process**:
- Optimize on: Last 30 days
- Validate on: Prior 60 days (out-of-sample)

**Checks**:
- Sharpe drop (should be <50% drop)
- Drawdown stability (should be <5% difference)
- Trade count stability (should be <50% difference)

**Expected**:
- Sharpe may drop some (normal)
- Drawdown stable
- Trade count doesn't collapse

### ✅ D. Baseline Check
**Implemented**: Automatic baseline comparison

**Baseline Config**:
```python
{
    "sigma_level": 2.0,
    "retest_count": 3,
    "atr_cap_mult": 2.5,
    "trail_pct": 0.007,
    "session_filter": 1
}
```

**What's Compared**:
- Baseline Sharpe vs Optimized Sharpe
- Baseline Trades vs Optimized Trades
- Improvement metrics

**Expected**: Optimized config consistently beats baseline on risk-adjusted metrics

### ✅ E. Stability Check (Top-K Configs)
**Implemented**: Top-K config tracking

**Process**:
- Tracks top 5-10 configurations
- Reports median/consensus config
- Prevents "one lucky sample" from dominating

**Output**: `"top_k_configs"` in report (can deploy median instead of single best)

## 📊 Enhanced Logging

### Metrics Now Logged:
1. **Sharpe Ratio** - Risk-adjusted return
2. **Max Drawdown** - Maximum equity drop
3. **False Break Rate** - % of trades that lose quickly
4. **Total Trades** - Trade count
5. **Win Rate** - % of winning trades
6. **Avg Holding Time** - Average bars held
7. **Payoff Ratio** - Avg win / Avg loss
8. **Max Consecutive Losers** - Longest losing streak
9. **Avg R:R** - Average reward:risk ratio

### Why These Matter:
- **Holding Time**: If too short → overtrading, if too long → missing opportunities
- **Payoff Ratio**: Should be >1.0 (winners bigger than losers)
- **Max Consecutive Losers**: If >10 → risk management issue
- **Avg R:R**: Should match target (2.5:1 to 3:1)

## 🔧 Scoring Improvements

### Production Score Function:
```python
def production_score(m: EnhancedMetrics):
    # Hard penalty if too few trades (prevents "no-trade" solutions)
    if m.trades < min_trades:
        return big_penalty
    
    # Balanced scoring
    score = -Sharpe + dd_penalty + false_break_penalty + freq_penalty
    
    # Additional penalties for pathological cases
    if max_consecutive_losers > 10:
        score += 2.0
    
    if payoff_ratio < 0.5:
        score += 1.0
```

### Key Improvements:
1. **Hard Floor on Trades**: Prevents degenerate "no-trade" solutions
2. **Balanced Weights**: Not too Sharpe-heavy, not too DD-heavy
3. **Pathology Penalties**: Catches edge cases

## 🎯 Risk Governor Check

### Current Implementation:
**Only optimizes**:
- sigma_level
- retest_count
- atr_cap_mult
- trail_pct
- session_filter

**Does NOT optimize**:
- Position size (fixed 0.1 lot)
- Risk per trade (fixed)
- Leverage (not changed)

### Risk Governor (If Needed):
```python
# Hard limits (not optimized)
MAX_RISK_PER_TRADE = 0.01  # 1% of equity
MAX_DAILY_DRAWDOWN = 0.05   # 5% kill switch
MAX_CONCURRENT_EXPOSURE = 1  # One position at a time
```

**Status**: ✅ Safe - Only strategy parameters optimized, not risk settings

## 📝 Output Format

### JSON Structure:
```json
{
  "config": {
    "sigma_level": 2.0,
    "retest_count": 3,
    "atr_cap_mult": 2.5,
    "trail_pct": 0.007,
    "session_filter": 1
  },
  "metadata": {
    "optimized_at": "2026-02-01T17:30:00",
    "seed": 7,
    "method": "anneal",
    "n_samples": 150,
    "metrics": {
      "sharpe": 1.23,
      "max_drawdown": 0.12,
      "trades": 85,
      "win_rate": 0.42,
      "avg_rr": 2.3
    },
    "baseline_comparison": {
      "baseline_sharpe": 0.85,
      "optimized_sharpe": 1.23,
      "sharpe_improvement": 0.38
    },
    "constraint_valid": true
  }
}
```

## 🚀 Usage

### Run with All Checks:
```bash
python vwap_qubo_tuner_production.py --symbol GBPJPY --days 30 --samples 150 --seed 7 --validate
```

### Quick Reproducibility Test:
```bash
# Run twice, compare outputs
python vwap_qubo_tuner_production.py --symbol GBPJPY --seed 7 > run1.txt
python vwap_qubo_tuner_production.py --symbol GBPJPY --seed 7 > run2.txt
diff run1.txt run2.txt  # Should be nearly identical
```

## ✅ Validation Checklist

Before deploying optimized config:

- [ ] **Reproducibility**: Same seed produces same results
- [ ] **Constraints**: One value per parameter (check JSON)
- [ ] **Walk-Forward**: Test Sharpe doesn't collapse
- [ ] **Baseline**: Beats default config
- [ ] **Stability**: Top-K configs are similar
- [ ] **Metrics**: All metrics reasonable (win rate, R:R, etc.)
- [ ] **Risk**: No position sizing changes

## 🎯 Expected Behavior

### Good Optimization:
- ✅ Sharpe improves vs baseline
- ✅ Constraints always valid
- ✅ Walk-forward Sharpe drop <50%
- ✅ Trade count reasonable (50-150 for 30 days)
- ✅ Win rate 35-45%
- ✅ Avg R:R 2.0-3.0

### Red Flags:
- ❌ Constraints break (penalty too low)
- ❌ Walk-forward Sharpe collapses (>50% drop)
- ❌ Trade count collapses (<10 trades)
- ❌ Max consecutive losers >10
- ❌ Payoff ratio <0.5

## 📊 Sample Output Interpretation

### If You See:
```
BASELINE COMPARISON:
  Sharpe: 0.85 → 1.23 (+0.38)  ✅ Good improvement
  Trades: 120 → 85  ✅ Reasonable (not collapsed)

WALK-FORWARD VALIDATION:
  Train Sharpe: 1.23
  Test Sharpe: 0.95  ⚠️ Some drop (normal)
  Sharpe Drop: 0.28  ✅ <50% drop, acceptable
  DD Stable: True  ✅
  Trades Stable: True  ✅
```

**Verdict**: ✅ **GOOD** - Optimization is working, not overfitting

### If You See:
```
BASELINE COMPARISON:
  Sharpe: 0.85 → 2.50 (+1.65)  ⚠️ Suspiciously high
  Trades: 120 → 5  ❌ Collapsed!

WALK-FORWARD VALIDATION:
  Train Sharpe: 2.50
  Test Sharpe: -0.50  ❌ Collapsed!
  Sharpe Drop: 3.00  ❌ >50% drop
```

**Verdict**: ❌ **OVERFITTING** - Increase samples, check scoring weights

---

**Status**: ✅ **All sanity checks implemented and ready!**

The production tuner now includes all 5 critical checks plus enhanced logging and validation.
