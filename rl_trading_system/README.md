# RL Trading System

Production-grade Reinforcement Learning trading system using PPO (Proximal Policy Optimization) for TraderLocker.

## Architecture

```
Market Data → State Encoder → RL Policy Network → Position Sizing Engine → TraderLocker Execution Engine → Risk Governor
```

## Components

### 1. State Encoder (`state_encoder.py`)
Converts market data into feature vectors:
- Price returns (multiple timeframes)
- Volatility metrics (ATR, std dev)
- Trend indicators (EMA crossovers)
- Momentum (RSI)
- VWAP distance
- Time features
- Position state
- Risk metrics

### 2. PPO Agent (`ppo_agent.py`)
Actor-Critic neural network:
- **Actor**: Outputs action probabilities (direction + position size)
- **Critic**: Estimates state value
- Uses clipped surrogate objective for stable training

### 3. Trading Environment (`trading_environment.py`)
Simulates trading environment:
- Position management
- PnL calculation
- Reward shaping: `reward = equity_change - drawdown_penalty - volatility_penalty`
- Encourages smooth equity growth with drawdown control

### 4. Risk Governor (`risk_governor.py`)
Enforces risk limits:
- Maximum drawdown limits
- Position size limits
- Daily loss limits
- Consecutive loss limits

### 5. TraderLocker Executor (`traderlocker_executor.py`)
Handles live trading via TraderLocker API:
- Order placement
- Position management
- Account queries
- Market data fetching

## Installation

```bash
pip install torch numpy pandas requests
```

## Usage

### Training

```bash
python train.py \
    --data data/gbpjpy_1m.csv \
    --episodes 1000 \
    --update-freq 200 \
    --save-freq 100 \
    --initial-equity 10000 \
    --save-dir ./models
```

### Inference (Backtest)

```bash
python inference.py \
    --model ./models/ppo_model_final.pth \
    --data data/gbpjpy_1m.csv \
    --initial-equity 10000
```

### Inference (Live Trading)

```bash
python inference.py \
    --model ./models/ppo_model_final.pth \
    --live \
    --symbol GBPJPY \
    --api-key YOUR_API_KEY
```

## Data Format

CSV file with columns:
- `open`, `high`, `low`, `close`: OHLC prices
- `volume`: Trading volume
- `vwap`: Volume-weighted average price (optional, will be calculated if missing)
- `timestamp`: Timestamp (optional)

## Reward Function

The reward function encourages:
- **Equity growth**: Positive reward for increasing equity
- **Low drawdown**: Penalty for drawdown from peak equity
- **Smooth equity curve**: Penalty for high equity volatility

```
reward = equity_change_pct - λ × drawdown - γ × volatility
```

This forces the agent to:
- Prefer smooth growth over volatile gains
- Avoid blowing up
- Learn compounding behavior

## Action Space

At each step, the agent chooses:
- **Direction**: `{0: flat, 1: long, 2: short}`
- **Position Size**: Fraction of equity `[0, 1]`

This allows the agent to:
- Go all-in when confidence is high
- Scale down when conditions worsen
- Go flat during bad regimes

## Training Tips

1. **Start with historical data**: Train on at least 3-6 months of 1-minute data
2. **Monitor metrics**: Watch win rate (target: ≥55%), drawdown (target: ≤20%), Sharpe ratio
3. **Adjust hyperparameters**: 
   - Learning rate: `3e-4` (default)
   - Discount factor: `0.99` (default)
   - PPO clip: `0.2` (default)
4. **Risk management**: Always use Risk Governor in live trading
5. **Start small**: Use sandbox mode and small position sizes initially

## Performance Targets

- **Win Rate**: ≥55%
- **Max Drawdown**: ≤20%
- **Sharpe Ratio**: ≥1.0
- **Total Return**: Varies by market conditions

## File Structure

```
rl_trading_system/
├── __init__.py
├── state_encoder.py      # State encoding
├── ppo_agent.py          # PPO agent
├── trading_environment.py # Trading environment
├── risk_governor.py      # Risk management
├── traderlocker_executor.py # Execution engine
├── train.py              # Training script
├── inference.py          # Inference script
└── README.md             # This file
```

## Notes

- The system is designed for **GBP/JPY 1-minute** trading but can be adapted
- Always test in sandbox mode before live trading
- Monitor the agent's behavior and adjust risk limits as needed
- The reward function is critical - it shapes what the agent learns
- Use proper position sizing - the agent outputs fractions, you control the actual size

## License

MIT License

