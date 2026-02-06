# TraderLocker RL Trading Bot - Integrated Version

## Overview

This is an integrated version that combines:
- ✅ Your prototype's simplicity
- ✅ Our production RL system's features
- ✅ TraderLocker API integration
- ✅ Comprehensive state encoding
- ✅ Risk management

## Quick Start

### 1. Training Mode (Backtest)

```bash
# Using CSV data
python traderlocker_rl_bot.py --train --symbol GBPJPY --data your_data.csv --episodes 500

# Using TraderLocker API
python traderlocker_rl_bot.py --train --symbol GBPJPY --api-key YOUR_KEY --episodes 500
```

### 2. Live Trading

```bash
# Use the existing traderlocker_setup.py
python traderlocker_setup.py --live --model ./models/ppo_model_final.pth --symbol GBPJPY
```

## Key Features

### 1. **State Encoding**
- Price returns (multiple timeframes)
- Volatility metrics (ATR, std dev)
- Trend indicators (EMA crossovers)
- Momentum (RSI)
- VWAP distance
- Volume features
- Time features
- Position state
- Risk metrics

### 2. **PPO Agent**
- Actor-Critic architecture
- Direction: 3 actions (flat, long, short)
- Position size: Continuous [0, 1]
- Proper PPO training with clipping

### 3. **Trading Environment**
- Realistic PnL calculation
- Commission handling
- Drawdown tracking
- Reward shaping (equity growth - drawdown penalty)

### 4. **TraderLocker Integration**
- Historical data loading
- Live order execution
- Position management

## Architecture

```
┌─────────────────┐
│  Market Data    │ (TraderLocker API or CSV)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TradingEnv      │ → State encoding, PnL, rewards
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PPO Agent      │ → Action (direction, size)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ TraderLocker    │ → Execute orders
│   Execution     │
└─────────────────┘
```

## Training Parameters

```python
STATE_SIZE = 15       # Set by StateEncoder
ACTION_SIZE = 2       # [direction, position_size]
HIDDEN_SIZE = 128
LR = 0.0003
GAMMA = 0.99
EPSILON = 0.2         # PPO clip
UPDATE_EVERY = 200    # Steps between training
```

## Reward Function

```
reward = equity_change_pct - 2.0 × drawdown
```

This encourages:
- Equity growth
- Low drawdown
- Smooth performance

## Example Usage

### Training on Historical Data

```python
from traderlocker_rl_bot import train_agent
import pandas as pd

# Load your data
data = pd.read_csv('gbpjpy_1m.csv')

# Train
agent, env = train_agent(
    symbol='GBPJPY',
    data=data,
    episodes=1000
)

# Get statistics
from traderlocker_rl_bot import get_statistics
stats = get_statistics(env)
print(f"Win Rate: {stats['win_rate']*100:.2f}%")
print(f"Return: {stats['total_return']*100:.2f}%")
```

### Live Trading

```python
from traderlocker_rl_bot import TradingEnv, PPOAgent, execute_trade_traderlocker
import torch

# Load trained model
agent = PPOAgent(state_size=15)
checkpoint = torch.load('ppo_model_GBPJPY_20240131.pth')
agent.model.load_state_dict(checkpoint['model_state_dict'])

# Create environment with live data
env = TradingEnv('GBPJPY', api_key='YOUR_KEY')

# Trading loop
state = env.reset()
while True:
    direction, pos_size, _, _ = agent.act(state, deterministic=True)
    
    # Execute via TraderLocker
    direction_str = ['flat', 'long', 'short'][direction]
    result = execute_trade_traderlocker('GBPJPY', direction_str, pos_size, 'YOUR_KEY')
    
    next_state, reward, done, info = env.step(direction, pos_size)
    state = next_state
    
    if done:
        break
```

## Differences from Prototype

| Prototype | Integrated Version |
|-----------|-------------------|
| Simple state (OHLC + volume) | Comprehensive state (15+ features) |
| Basic PPO | Full PPO with proper training |
| Simple reward | Reward shaping with drawdown penalty |
| No risk management | Risk governor integration ready |
| Basic position sizing | Adaptive position sizing |

## Next Steps

1. **Train on your data**: Use `--train` mode with your CSV data
2. **Test in sandbox**: Use TraderLocker sandbox mode
3. **Monitor performance**: Track win rate, drawdown, returns
4. **Go live**: Switch to live mode when ready

## Files

- `traderlocker_rl_bot.py` - Main integrated bot
- `rl_trading_system/` - Production RL components
- `traderlocker_setup.py` - Live trading setup
- `backtest_standalone.py` - Backtesting script

---

**Ready to train?** Run: `python traderlocker_rl_bot.py --train --symbol GBPJPY --data your_data.csv`
