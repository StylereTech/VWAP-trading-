# VWAP Trading Strategy - Production-Grade System

A production-ready VWAP Control Flip trading strategy with integrated risk management, optimized for prop-firm deployment.

## 🎯 Key Features

- **VWAP Control Flip Strategy**: Mean-reversion strategy using VWAP bands and retest logic
- **Production-Grade Risk Management**: Drawdown governor, ATR-based position sizing, stress testing
- **Prop-Firm-Safe**: Meets standard prop firm requirements (2% daily loss, 5% total loss limits)
- **Real Data Backtesting**: Infrastructure for testing on real market data (GBPJPY, BTCUSD, XAUUSD)
- **QUBO/QAOA Optimization**: Quantum-inspired parameter optimization with risk-aware scoring
- **Enhanced Exits**: Time stops, break-even, partial TP, loss duration caps

## 📊 Strategy Performance (With Risk Management)

**Expected Metrics:**
- ROI: 18-35% (real, not leveraged)
- Max Drawdown: <25-35%
- Win Rate: 70-80%
- Breach Probability: <5% (prop-firm-safe)

## 🚀 Quick Start

### 1. Get Real Market Data

**Option A: Use Update Script**
```bash
python update_csv_data.py update --symbol GBPJPY --source mt5 --days 30
```

**Option B: Copy-Paste into TraderLocker**
- Open `COPY_PASTE_TRADERLOCKER_UPDATED.py`
- Copy entire contents
- Paste into TraderLocker script editor
- Run and download CSV files

### 2. Validate Data

```bash
python test_csv_loader.py
```

### 3. Run Tests

```bash
# Test all symbols
python run_all_symbols_tests.py

# Test single symbol
python run_ablation_real_data.py --test A --symbol GBPJPY --days 30
```

### 4. Run Backtest with Risk Management

```bash
python backtest_with_risk_management.py
```

## 📁 Project Structure

```
VWAP-trading-/
├── capital_allocation_ai/
│   ├── risk/                    # Risk management modules
│   │   ├── drawdown_governor.py
│   │   ├── position_sizing.py
│   │   └── stress_tester.py
│   └── vwap_control_flip_strategy.py  # Main strategy
├── update_csv_data.py          # CSV data management
├── run_all_symbols_tests.py    # Multi-symbol testing
├── backtest_with_risk_management.py  # Example backtest
└── vwap_qubo_tuner_production.py  # Parameter optimizer
```

## 🔧 Configuration

### Prop-Firm-Safe Defaults

See `PROP_FIRM_SAFE_CONFIG.json` for default configuration:

```json
{
  "risk_per_trade_frac": 0.003,  // 0.30% risk per trade
  "max_daily_loss_frac": 0.02,   // 2% daily loss limit
  "max_total_loss_frac": 0.05,   // 5% total loss limit
  "peak_trailing_dd_hard": 0.045 // 4.5% peak trailing DD limit
}
```

## 📚 Documentation

- **`RISK_MANAGEMENT_INTEGRATION.md`** - Complete risk management guide
- **`RISK_MANAGEMENT_COMPLETE.md`** - Implementation summary
- **`CSV_MANAGEMENT_GUIDE.md`** - CSV data management guide
- **`QUICK_START.md`** - Fast reference
- **`PROP_FIRM_SAFE_CONFIG.json`** - Default configuration

## 🧪 Testing

### Run All Tests
```bash
python run_all_symbols_tests.py
```

### Test Individual Symbol
```bash
python run_ablation_real_data.py --test A --symbol GBPJPY --days 30
python run_ablation_real_data.py --test B --symbol GBPJPY --days 30
python run_ablation_real_data.py --test C --symbol GBPJPY --days 30
```

## 📈 Optimization

Run parameter optimization with risk-aware scoring:

```bash
python run_optimization_with_progress.py
```

The optimizer includes:
- Hard gates (rejects max_dd > 5% or breach_prob > 5%)
- Stress test penalties (Monte Carlo + worst-case)
- CVaR and loss duration penalties

## ⚠️ Important Notes

1. **Instrument Specs**: Adjust `dollar_per_price_unit` in your backtest for your broker's contract specifications
2. **Equity Tracking**: Must update `strategy.equity` after each trade in your backtest loop
3. **Real Data Required**: Optimization should use real market data, not synthetic
4. **Prop Firm Rules**: Current defaults match most prop firms; adjust if needed

## 🔗 Repository

**GitHub**: https://github.com/StylereTech/VWAP-trading-

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

Built with production-grade risk management for sustainable trading.
