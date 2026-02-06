# TraderLocker Integration Guide

## ✅ What's Ready

Your RL Trading System is now ready for TraderLocker integration! Here's what you have:

### 1. **TraderLocker Executor** (`rl_trading_system/traderlocker_executor.py`)
- ✅ API connection handling
- ✅ Order placement (market orders)
- ✅ Position management
- ✅ Account information retrieval
- ✅ Market data fetching
- ✅ Sandbox mode support

### 2. **Setup Script** (`traderlocker_setup.py`)
- ✅ Connection testing
- ✅ Live trading wrapper
- ✅ Model loading integration
- ✅ Risk management integration

## 🚀 Quick Start

### Step 1: Get Your API Key
1. Log into TraderLocker dashboard
2. Navigate to API settings
3. Generate an API key
4. Copy the key

### Step 2: Set Environment Variable
```bash
# Windows PowerShell
$env:TRADERLOCKER_API_KEY = "your_api_key_here"

# Windows CMD
set TRADERLOCKER_API_KEY=your_api_key_here

# Linux/Mac
export TRADERLOCKER_API_KEY="your_api_key_here"
```

### Step 3: Test Connection
```bash
python traderlocker_setup.py
```

### Step 4: Run Live Trading
```bash
python traderlocker_setup.py --live --model ./models/ppo_model_final.pth --symbol GBPJPY
```

## 📋 API Endpoints Used

The system uses these TraderLocker API endpoints:

1. **GET /account** - Get account information
2. **GET /positions** - Get current positions
3. **GET /market-data** - Get historical market data
4. **POST /order** - Place new order
5. **POST /position/{id}/close** - Close specific position
6. **POST /positions/close-all** - Close all positions

## 🔧 Configuration

### Sandbox Mode (Recommended for Testing)
```python
executor = TraderLockerExecutor(
    api_key="your_key",
    sandbox=True  # Use sandbox environment
)
```

### Live Mode
```python
executor = TraderLockerExecutor(
    api_key="your_key",
    sandbox=False  # Live trading
)
```

## 📊 Order Execution

The system executes trades like this:

```python
# Long position
executor.execute_trade(
    symbol="GBPJPY",
    action=(1, 0.10),  # (direction=long, position_size=10% of equity)
    current_price=185.50,
    stop_loss=185.00,
    take_profit=186.50
)

# Short position
executor.execute_trade(
    symbol="GBPJPY",
    action=(2, 0.10),  # (direction=short, position_size=10% of equity)
    current_price=185.50,
    stop_loss=186.00,
    take_profit=184.50
)

# Close position (go flat)
executor.execute_trade(
    symbol="GBPJPY",
    action=(0, 0.0),  # (direction=flat)
    current_price=185.50
)
```

## ⚠️ Safety Features

### 1. Risk Governor
- Max drawdown: 20%
- Position size limits: 50% of equity
- Daily loss limit: 5%
- Consecutive loss limit: 5 trades

### 2. Sandbox Mode
- Always test in sandbox first!
- Sandbox uses virtual money
- Perfect for testing before going live

### 3. Position Sizing
- Fixed lot size: 0.10 (mini lot)
- Or risk-based: 2% risk per trade
- Or RL agent output (adaptive)

## 📝 Example: Complete Live Trading Setup

```python
from rl_trading_system.ppo_agent import PPOAgent
from rl_trading_system.traderlocker_executor import TraderLockerExecutor
from rl_trading_system.risk_governor import RiskGovernor
import os

# 1. Setup
API_KEY = os.getenv("TRADERLOCKER_API_KEY")
executor = TraderLockerExecutor(API_KEY, sandbox=True)

# 2. Load trained model
agent = PPOAgent(state_size=15)
agent.load("./models/ppo_model_final.pth")

# 3. Setup risk management
risk_gov = RiskGovernor(
    max_drawdown_pct=0.20,
    max_position_size_pct=0.50
)

# 4. Get market data
market_data = executor.get_market_data("GBPJPY", "1m", limit=1000)

# 5. Run trading loop
# (See traderlocker_setup.py for complete example)
```

## 🔍 Monitoring

### Check Account Status
```python
account = executor.get_account_info()
print(f"Equity: ${account['equity']}")
print(f"Balance: ${account['balance']}")
```

### Check Positions
```python
positions = executor.get_positions()
for pos in positions['positions']:
    print(f"{pos['symbol']}: {pos['size']} @ {pos['price']}")
```

## 🎯 Next Steps

1. **Test in Sandbox**: Run `traderlocker_setup.py` with sandbox=True
2. **Monitor Performance**: Watch equity, drawdown, win rate
3. **Adjust Parameters**: Tune risk limits based on results
4. **Go Live**: Switch to sandbox=False when ready

## 📞 Support

If you encounter issues:
1. Check API key is correct
2. Verify sandbox/live mode setting
3. Check network connection
4. Review TraderLocker API documentation

## ⚡ Quick Commands

```bash
# Test connection
python traderlocker_setup.py

# Run live trading (sandbox)
python traderlocker_setup.py --live --model ./models/ppo_model_final.pth

# Run live trading (specific symbol)
python traderlocker_setup.py --live --model ./models/ppo_model_final.pth --symbol GBPJPY
```

---

**Ready to trade?** Start with sandbox mode and test thoroughly before going live! 🚀
