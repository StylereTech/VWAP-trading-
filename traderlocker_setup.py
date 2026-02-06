"""
TraderLocker Integration Setup
Quick setup guide and example for connecting to TraderLocker API
"""

import os
from rl_trading_system.traderlocker_executor import TraderLockerExecutor
from rl_trading_system.inference import run_inference
from rl_trading_system.trading_environment import TradingEnvironment
from rl_trading_system.ppo_agent import PPOAgent
from rl_trading_system.risk_governor import RiskGovernor


def setup_traderlocker(api_key: str, sandbox: bool = True):
    """
    Setup TraderLocker executor.
    
    Args:
        api_key: Your TraderLocker API key
        sandbox: Use sandbox environment (recommended for testing)
    """
    executor = TraderLockerExecutor(
        api_key=api_key,
        base_url="https://api.traderlocker.com/v1",
        sandbox=sandbox
    )
    
    # Test connection
    print("Testing TraderLocker connection...")
    account_info = executor.get_account_info()
    
    if account_info:
        print("✓ Connected successfully!")
        print(f"Account Equity: ${account_info.get('equity', 'N/A')}")
        return executor
    else:
        print("✗ Connection failed. Check your API key.")
        return None


def run_live_trading(model_path: str, api_key: str, symbol: str = "GBPJPY"):
    """
    Run live trading with TraderLocker.
    
    Args:
        model_path: Path to trained PPO model
        api_key: TraderLocker API key
        symbol: Trading symbol (default: GBPJPY)
    """
    print("="*60)
    print("TRADERLOCKER LIVE TRADING")
    print("="*60)
    
    # Setup executor
    executor = setup_traderlocker(api_key, sandbox=True)
    if not executor:
        return
    
    # Load agent
    print(f"\nLoading model from {model_path}...")
    state_size = 15  # Match your training state size
    agent = PPOAgent(state_size=state_size)
    agent.load(model_path)
    print("Model loaded successfully")
    
    # Get market data from TraderLocker
    print(f"\nFetching market data for {symbol}...")
    market_data = executor.get_market_data(symbol=symbol, timeframe='1m', limit=1000)
    
    if not market_data or 'data' not in market_data:
        print("Failed to fetch market data")
        return
    
    # Convert to DataFrame format
    import pandas as pd
    df = pd.DataFrame(market_data['data'])
    
    # Ensure required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        print(f"Missing required columns. Got: {df.columns.tolist()}")
        return
    
    # Create environment
    env = TradingEnvironment(data=df, initial_equity=10000.0)
    
    # Create risk governor
    risk_gov = RiskGovernor(
        max_drawdown_pct=0.20,
        max_position_size_pct=0.50,
        daily_loss_limit_pct=0.05,
        consecutive_loss_limit=5
    )
    
    # Run inference
    print("\nStarting live trading...")
    print("Press Ctrl+C to stop")
    
    try:
        run_inference(
            env=env,
            agent=agent,
            executor=executor,
            risk_gov=risk_gov,
            live=True,
            symbol=symbol
        )
    except KeyboardInterrupt:
        print("\n\nTrading stopped by user")
        # Close all positions
        executor.close_all_positions(symbol=symbol)
        print("All positions closed")


if __name__ == "__main__":
    # Example usage
    API_KEY = os.getenv("TRADERLOCKER_API_KEY", "YOUR_API_KEY_HERE")
    
    if API_KEY == "YOUR_API_KEY_HERE":
        print("="*60)
        print("TRADERLOCKER SETUP")
        print("="*60)
        print("\nTo use TraderLocker:")
        print("1. Get your API key from TraderLocker dashboard")
        print("2. Set environment variable:")
        print("   export TRADERLOCKER_API_KEY='your_key_here'")
        print("3. Or edit this file and replace YOUR_API_KEY_HERE")
        print("\nFor live trading:")
        print("  python traderlocker_setup.py --live --model ./models/ppo_model_final.pth")
    else:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--live', action='store_true', help='Run live trading')
        parser.add_argument('--model', type=str, help='Path to model file')
        parser.add_argument('--symbol', type=str, default='GBPJPY', help='Trading symbol')
        args = parser.parse_args()
        
        if args.live and args.model:
            run_live_trading(args.model, API_KEY, args.symbol)
        else:
            setup_traderlocker(API_KEY, sandbox=True)
