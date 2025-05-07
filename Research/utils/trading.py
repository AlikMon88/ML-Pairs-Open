import pandas as pd
import numpy as np
import random

class Portfolio:
    def __init__(self, cash=100000):
        self.cash = cash
        self.holdings = {}  # Stores security holdings (key: symbol)
        self.total_value = cash  # Total portfolio value (cash + holdings value)
        
    def update(self, symbol: str, price: float):
        """Update portfolio values based on current security prices"""
        if symbol in self.holdings:
            position = self.holdings[symbol]
            position['market_value'] = position['quantity'] * price
            self.total_value = self.cash + sum(
                pos['market_value'] for pos in self.holdings.values()
            )

class Trading:
    def __init__(self, portfolio: Portfolio):
        self.portfolio = portfolio
        self.current_prices = {}  # Stores current market prices
        
    def set_holding(self, symbol: str, target_percent: float, price: float):
        """
        Set target percentage of portfolio for a security
        Args:
            symbol: Security identifier
            target_percent: Decimal percentage (e.g., 0.5 for 50%)
            price: Current market price
        """
        # Calculate target value
        target_value = self.portfolio.total_value * target_percent
        
        # Get current position value
        current_value = self.portfolio.holdings.get(symbol, {}).get('market_value', 0)
        
        # Calculate required order value
        delta_value = target_value - current_value
        order_quantity = delta_value / price
        
        # Execute order
        self._place_order(symbol, order_quantity, price)
        
    def liquidate(self, symbol: str, price: float):
        """Close entire position for a security"""
        if symbol in self.portfolio.holdings:
            position = self.portfolio.holdings[symbol]
            order_quantity = -position['quantity']
            self._place_order(symbol, order_quantity, price)
            del self.portfolio.holdings[symbol]
            
    def _place_order(self, symbol: str, quantity: float, price: float):
        """Core order execution logic"""
        if quantity == 0:
            return
            
        # Calculate order cost
        order_cost = quantity * price
        
        # Update portfolio cash
        self.portfolio.cash -= order_cost
        
        # Update holdings
        if symbol in self.portfolio.holdings:
            self.portfolio.holdings[symbol]['quantity'] += quantity
        else:
            self.portfolio.holdings[symbol] = {
                'quantity': quantity,
                'average_price': price,
                'market_value': quantity * price
            }
            
        # Update portfolio values
        self.portfolio.update(symbol, price)
        
    def update_prices(self, prices: dict):
        """Update current market prices for securities"""
        self.current_prices.update(prices)
        for symbol in prices:
            if symbol in self.portfolio.holdings:
                self.portfolio.update(symbol, prices[symbol])



if __name__ == '__main__':
    print('running ...__trading.py__...')