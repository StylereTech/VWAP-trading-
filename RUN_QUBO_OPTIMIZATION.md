# How to Run QUBO Optimization

## ✅ System Status

**QUBO/QAOA Optimization**: ✅ **FULLY INTEGRATED & READY**

All files are in place:
- ✅ `vwap_qubo_tuner.py` - Main optimization script
- ✅ `run_qubo_optimization.py` - Multi-symbol runner
- ✅ `enhanced_control_flip_strategy.py` - Strategy integration
- ✅ Real backtest evaluator - Connected

## 🚀 Running the Optimization

### Option 1: Quick Test (Recommended First)
Test with smaller dataset to verify it works:

```bash
python vwap_qubo_tuner.py --symbol GBPJPY --days 7 --samples 30 --no-qaoa
```

**Time**: ~5-10 minutes
**Purpose**: Verify system works before full optimization

### Option 2: Standard Optimization
Full optimization for one symbol:

```bash
python vwap_qubo_tuner.py --symbol GBPJPY --days 30 --samples 150
```

**Time**: ~15-30 minutes per symbol
**Purpose**: Find optimal parameters

### Option 3: All Symbols
Optimize all three symbols:

```bash
python run_qubo_optimization.py
```

**Time**: ~45-90 minutes total
**Purpose**: Complete optimization for all symbols

## ⏱️ Why It Takes Time

Each sample requires:
1. Generate random parameter configuration
2. Run full backtest (7-30 days of data)
3. Calculate metrics (Sharpe, drawdown, trades, false breaks)
4. Repeat 30-200 times

**Example**: 30 samples × 7-day backtest = ~210 days of backtesting

## 📊 What Happens During Optimization

### Phase 1: Sampling (5-15 minutes)
```
Sampling 30 configurations...
  Progress: 10/30 samples
  Progress: 20/30 samples
  Progress: 30/30 samples
```

### Phase 2: Learning (1-2 minutes)
```
Fitting QUBO surrogate from 30 samples...
```

### Phase 3: Solving (1-2 minutes)
```
Solving QUBO...
  Attempting QAOA solve...
  QAOA not available, using Simulated Annealing
  Using Simulated Annealing...
```

### Phase 4: Validation (1-2 minutes)
```
Evaluating optimized configuration...
```

### Phase 5: Output
```
CHOSEN CONFIG: {
  "sigma_level": 2.0,
  "retest_count": 3,
  "atr_cap_mult": 2.5,
  "trail_pct": 0.007,
  "session_filter": 1
}

METRICS: {
  "sharpe": 1.23,
  "max_drawdown": 0.12,
  "false_break_rate": 0.25,
  "trades": 85
}

Saved optimized parameters -> vwap_params_gbpjpy.json
```

## 🎯 Expected Results

After optimization, you'll get:

1. **Optimized Parameters** (JSON file):
   - Best combination of sigma_level, retest_count, atr_cap_mult, trail_pct, session_filter
   - Symbol-specific (different for GBP/JPY vs BTC/USD vs Gold)

2. **Performance Metrics**:
   - Sharpe ratio (target: >1.0)
   - Max drawdown (target: <20%)
   - False break rate (target: <30%)
   - Trade count (target: 50-150 for 30 days)

3. **Comparison**:
   - Best sample score vs optimized score
   - Shows improvement from optimization

## 🔧 Troubleshooting

### If It Times Out:
- Use smaller dataset: `--days 7` instead of `--days 30`
- Use fewer samples: `--samples 30` instead of `--samples 150`
- Run in background or on faster machine

### If No Trades Generated:
- Parameters may be too strict
- Check backtest evaluator is working
- Try looser parameter space

### If QAOA Fails:
- That's OK - Simulated Annealing will be used
- QAOA requires Qiskit installation (optional)
- Use `--no-qaoa` flag to skip QAOA attempt

## 📝 After Optimization

1. **Review Results**:
   ```bash
   cat vwap_params_gbpjpy.json
   ```

2. **Validate**:
   Run 60-day backtest with optimized params:
   ```python
   import json
   from enhanced_control_flip_strategy import EnhancedVWAPControlFlipStrategy, EnhancedStrategyParams
   
   cfg = json.load(open("vwap_params_gbpjpy.json"))
   params = EnhancedStrategyParams(
       band_k=cfg["sigma_level"],
       require_nth_retest=cfg["retest_count"],
       atr_cap_mult=cfg["atr_cap_mult"],
       trail_pct=cfg["trail_pct"],
       require_session_filter=bool(cfg["session_filter"])
   )
   # Run backtest...
   ```

3. **Deploy**:
   Load JSON config in live bot at session start

## ✅ Ready to Run

The system is fully integrated and ready. Choose your option:

**Quick Test** (verify it works):
```bash
python vwap_qubo_tuner.py --symbol GBPJPY --days 7 --samples 30 --no-qaoa
```

**Full Optimization** (find best params):
```bash
python run_qubo_optimization.py
```

---

**Note**: Optimization will take time (15-30 min per symbol) because it runs real backtests. This is normal and expected. The QUBO approach is much more efficient than brute-force grid search!
