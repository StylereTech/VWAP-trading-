# Capital Allocation AI System

## Philosophy

**You are not building a trading bot.**
**You are building a capital-allocation AI.**

It decides:
- **When to trade**
- **How large to trade**
- **When to exit**
- **When to stop trading**
- **How aggressively to compound**

The market is the environment.
Your account balance is the reward.

## Architecture

Five engines working together:

```
Market Data → State Encoder → RL Policy Network
                                    ↓
                            Position Sizing Engine
                                    ↓
                            Execution Engine
                                    ↓
                           Risk Governor
```

## 1. State Encoder

**What the AI "sees"** - regime, momentum, and risk:

```python
state = [
    price returns (1m, 5m, 15m, 1h),
    volatility (ATR, stdev),
    trend (EMA fast - EMA slow),
    RSI,
    distance from VWAP,
    time of day,
    open position size,
    unrealized PnL,
    drawdown
]
```

The model doesn't see charts - it sees **regime, momentum, and risk**.

## 2. RL Policy Network

**What the AI can do**:

```python
Action = {
    direction ∈ { long, short, flat },
    position_size ∈ [0 … max_risk]
}
```

This lets it:
- Go **all-in** when confidence is high
- **Scale down** when conditions worsen
- Go **flat** during bad regimes

This is how 600% years happen - the AI learns when to bet hard.

## 3. Reward Function

**The secret sauce**:

```python
reward = Δ(account_equity) - λ × drawdown - γ × volatility_of_equity
```

This forces the agent to:
- Prefer **smooth growth**
- Avoid **blowing up**
- Learn **compounding behavior**

This is how professionals think.

## 4. Position Sizing Engine

Scales RL output by:
- **Max risk** (e.g., 50% of equity)
- **Drawdown adjustments** (reduce size if drawdown high)
- **Volatility adjustments** (reduce size if volatility extreme)

## 5. Risk Governor

**Hard limits**:
- Max drawdown: 20%
- Daily loss limit: 5%
- Consecutive loss limit: 5 trades

**Hard max-drawdown kill switch** - trading halts if limits exceeded.

## 6. Execution Engine

Connects to any broker:
- TraderLocker
- OANDA
- Interactive Brokers
- MT5
- Or any broker API

## Usage

### Training

```python
from capital_allocation_ai.training import train_capital_allocation_ai
import pandas as pd

# Load data
data = pd.read_csv('gbpjpy_1m.csv')

# Train
ai = train_capital_allocation_ai(
    data=data,
    episodes=1000,
    initial_equity=10000.0
)

# Save model
ai.agent.save('capital_allocation_model.pth')
```

### Live Trading

```python
from capital_allocation_ai.system import CapitalAllocationAI

# Initialize
ai = CapitalAllocationAI(
    symbol="GBPJPY",
    initial_equity=10000.0,
    max_risk=0.5,
    broker_type="traderlocker",
    api_key="YOUR_API_KEY"
)

# Trading loop
while True:
    # Get market data
    market_data = get_market_data()  # Your data source
    
    # Update and get decision
    result = ai.update_market_data(
        current_price=market_data['close'],
        high=market_data['high'],
        low=market_data['low'],
        volume=market_data['volume'],
        vwap=market_data['vwap']
    )
    
    # Result contains action, position_size, equity, drawdown
    print(f"Action: {result['action']}, Equity: ${result['equity']:.2f}")
```

## Key Features

### Compounding Behavior
The AI learns to:
- Increase position size when winning
- Compound gains aggressively
- Scale back when conditions worsen

### Regime Detection
The AI detects:
- Trending markets → bet hard
- Choppy markets → reduce size
- High volatility → go flat

### Dynamic Position Sizing
- RL agent outputs position size [0, max_risk]
- Position sizer adjusts based on:
  - Current drawdown
  - Market volatility
  - Risk limits

### Risk Management
- Hard drawdown limit (20%)
- Daily loss limit (5%)
- Consecutive loss limit (5)
- Automatic trading halt

## Performance Targets

Based on similar systems:
- **Win Rate**: 55-65%
- **Sharpe Ratio**: 1.0-2.0
- **Max Drawdown**: ≤20%
- **Annual Return**: 20-100%+ (depends on market)

## Why This Works

600% annual returns require:
- ✅ **Dynamic position sizing** - bet hard when confident
- ✅ **Regime detection** - know when to trade
- ✅ **Momentum exploitation** - ride trends
- ✅ **Automated discipline** - no emotions

**Humans cannot do this consistently.**
**AI agents can.**

This is exactly the type of system that wins year-long trading championships.

## Files

- `system.py` - Core Capital Allocation AI system
- `training.py` - Training script with reward function
- `README.md` - This file

---

**Ready to build your capital-allocation AI?** Start training and let it learn compounding behavior! 🚀
