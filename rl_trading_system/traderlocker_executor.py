"""
TraderLocker Execution Engine
Handles live trading via TraderLocker API.
"""

import requests
import time
from typing import Dict, Optional, Tuple
import logging


class TraderLockerExecutor:
    """
    Execution engine for TraderLocker API.
    Handles order placement, position management, and account queries.
    """
    
    def __init__(self,
                 api_key: str,
                 base_url: str = "https://api.traderlocker.com/v1",
                 sandbox: bool = True):
        """
        Args:
            api_key: TraderLocker API key
            base_url: API base URL
            sandbox: Whether to use sandbox environment
        """
        self.api_key = api_key
        self.base_url = base_url
        self.sandbox = sandbox
        
        if sandbox:
            self.base_url = base_url.replace('api', 'api-sandbox')
        
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        })
        
        self.logger = logging.getLogger(__name__)
    
    def get_account_info(self) -> Dict:
        """Get account information."""
        try:
            response = self.session.get(f"{self.base_url}/account")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_positions(self) -> Dict:
        """Get current positions."""
        try:
            response = self.session.get(f"{self.base_url}/positions")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return {}
    
    def get_market_data(self, symbol: str, timeframe: str = '1m', limit: int = 100) -> Dict:
        """Get market data for symbol."""
        try:
            params = {
                'symbol': symbol,
                'timeframe': timeframe,
                'limit': limit
            }
            response = self.session.get(f"{self.base_url}/market-data", params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}")
            return {}
    
    def place_order(self,
                   symbol: str,
                   direction: str,  # 'long' or 'short'
                   size: float,
                   order_type: str = 'market',
                   stop_loss: Optional[float] = None,
                   take_profit: Optional[float] = None) -> Dict:
        """
        Place order via TraderLocker API.
        
        Args:
            symbol: Trading symbol (e.g., 'GBPJPY')
            direction: 'long' or 'short'
            size: Position size (lot size or currency amount)
            order_type: 'market' or 'limit'
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
        
        Returns:
            Order response from API
        """
        payload = {
            'symbol': symbol,
            'direction': direction,
            'size': size,
            'type': order_type
        }
        
        if stop_loss is not None:
            payload['stop_loss'] = stop_loss
        
        if take_profit is not None:
            payload['take_profit'] = take_profit
        
        try:
            response = self.session.post(f"{self.base_url}/order", json=payload)
            response.raise_for_status()
            result = response.json()
            self.logger.info(f"Order placed: {direction} {size} {symbol}")
            return result
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return {'error': str(e)}
    
    def close_position(self, position_id: str) -> Dict:
        """Close a specific position."""
        try:
            response = self.session.post(f"{self.base_url}/position/{position_id}/close")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
            return {'error': str(e)}
    
    def close_all_positions(self, symbol: Optional[str] = None) -> Dict:
        """Close all positions (optionally filtered by symbol)."""
        try:
            payload = {}
            if symbol:
                payload['symbol'] = symbol
            
            response = self.session.post(f"{self.base_url}/positions/close-all", json=payload)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            self.logger.error(f"Error closing all positions: {e}")
            return {'error': str(e)}
    
    def execute_trade(self,
                     symbol: str,
                     action: Tuple[int, float],
                     current_price: float,
                     stop_loss: Optional[float] = None,
                     take_profit: Optional[float] = None) -> Dict:
        """
        Execute trade based on RL agent action.
        
        Args:
            symbol: Trading symbol
            action: (direction, position_size_fraction)
                direction: 0=flat, 1=long, 2=short
                position_size_fraction: Position size [0, 1]
            current_price: Current market price
            stop_loss: Stop loss price
            take_profit: Take profit price
        
        Returns:
            Execution result
        """
        direction_code, position_size_fraction = action
        
        # Get account info to calculate position size
        account_info = self.get_account_info()
        equity = account_info.get('equity', 10000.0)
        
        # Calculate lot size (assuming 0.10 lot = $10 per pip for GBP/JPY)
        # Adjust based on your broker's specifications
        position_value = equity * position_size_fraction
        lot_size = position_value / 100000.0  # Rough conversion (adjust for your broker)
        lot_size = round(lot_size, 2)  # Round to 2 decimal places
        
        if direction_code == 0:  # Flat - close positions
            return self.close_all_positions(symbol)
        
        elif direction_code == 1:  # Long
            return self.place_order(
                symbol=symbol,
                direction='long',
                size=lot_size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
        
        else:  # Short
            return self.place_order(
                symbol=symbol,
                direction='short',
                size=lot_size,
                stop_loss=stop_loss,
                take_profit=take_profit
            )


class MockExecutor:
    """
    Mock executor for testing without live API.
    Simulates order execution for backtesting.
    """
    
    def __init__(self):
        self.positions = {}
        self.order_history = []
        self.account_equity = 10000.0
    
    def get_account_info(self) -> Dict:
        return {'equity': self.account_equity, 'balance': self.account_equity}
    
    def get_positions(self) -> Dict:
        return {'positions': list(self.positions.values())}
    
    def execute_trade(self, symbol: str, action: Tuple[int, float],
                     current_price: float, **kwargs) -> Dict:
        """Mock trade execution."""
        direction_code, position_size_fraction = action
        
        order = {
            'symbol': symbol,
            'direction': 'flat' if direction_code == 0 else ('long' if direction_code == 1 else 'short'),
            'size': position_size_fraction,
            'price': current_price,
            'timestamp': time.time()
        }
        
        self.order_history.append(order)
        
        if direction_code == 0:
            # Close positions
            self.positions.clear()
        else:
            # Open position
            self.positions[symbol] = order
        
        return {'status': 'filled', 'order': order}


if __name__ == "__main__":
    # Test mock executor
    executor = MockExecutor()
    
    result = executor.execute_trade('GBPJPY', (1, 0.1), 150.0)
    print(f"Mock execution result: {result}")
    
    account = executor.get_account_info()
    print(f"Account info: {account}")

