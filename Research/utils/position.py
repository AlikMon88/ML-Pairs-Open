import numpy as np
import pandas as pd


def CalculateMarketVolatility(market_volatility, market_volatility_window):
    
    # Calculate rolling volatility of SPY using EMA
    history = _history(spy, market_volatility_window, Resolution.Daily)
    
    if history.empty:
        return market_volatility
    
    returns = history['close'].pct_change().dropna()
    
    if len(returns) == 0:
        return market_volatility
    
    ema_variance = returns.ewm(span=market_volatility_window).var().iloc[-1]
    ema_std = np.sqrt(ema_variance)
    
    return ema_std

class ManagePosition:

    def __init__(self, market_volatility, transaction_cost, entry_threshold, exit_threshold, stop_loss, take_profit):
        
        self.market_volatility = market_volatility 
        self.transaction_cost = transaction_cost 
        self.entry_threshold = entry_threshold 
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit

    def positions(self, Securities, open_positions):
        # Calculate market volatility scaling factor
        if self.market_volatility > 0:
            scaling_factor = 1.0 / self.market_volatility
        else:
            scaling_factor = 1.0  # Default scaling
        
        # Monitor open positions and generate signals
        for pair_key, position in open_positions.items():
            symbol_y = position['symbol_y']
            symbol_x = position['symbol_x']
            hedge_ratio = position['hedge_ratio']
            spread_mean = position['spread_mean']
            spread_std = position['spread_std']
            
            price_y = Securities[symbol_y].Price
            price_x = Securities[symbol_x].Price
            
            if price_y == 0 or price_x == 0:
                continue
            
            # Calculate current spread and z-score
            spread = price_y - hedge_ratio * price_x
            z_score = (spread - spread_mean) / spread_std
            
            # Adjust z-score for transaction costs
            adjusted_z = abs(z_score) - self.transaction_cost * 100  # Normalize transaction cost
            
            # Entry signal
            if not position['is_open']:
                if adjusted_z > self.entry_threshold:
                    # Position sizing
                    allocation = scaling_factor * (1.0 / spread_std) / len(pairs)
                    allocation = min(allocation, 0.05)  # Max 5% per position
                    
                    # Go long spread: short x, long y
                    if z_score > self.entry_threshold:
                        SetHoldings(symbol_y, allocation)
                        SetHoldings(symbol_x, -allocation * hedge_ratio)
                        print(f"Entering long spread for pair {pair_key} at z-score {z_score:.2f}")
                    # Go short spread: long x, short y
                    elif z_score < - self.entry_threshold:
                        SetHoldings(symbol_y, -allocation)
                        SetHoldings(symbol_x, allocation * hedge_ratio)
                        print(f"Entering short spread for pair {pair_key} at z-score {z_score:.2f}")
                    position['is_open'] = True
                    position['entry_z'] = z_score
            else:
                # Exit conditions
                exit_condition = abs(z_score) < self.exit_threshold
                stop_loss_condition = abs(z_score) > self.stop_loss
                take_profit_condition = (self.take_profit > 0) and (abs(z_score) > self.take_profit)
                
                if exit_condition or stop_loss_condition or take_profit_condition:
                    Liquidate(symbol_y)
                    Liquidate(symbol_x)
                    Debug(f"Exiting position for pair {pair_key} at z-score {z_score:.2f}")
                    position['is_open'] = False

    def LiquidatePositionsNotInPairs(self, pairs):
        # Liquidate positions that are not in the current pairs
        current_symbols = set()
        for pair_info in pairs.values():
            current_symbols.add(pair_info['symbol_y'])
            current_symbols.add(pair_info['symbol_x'])
        
        for kvp in Portfolio:
            symbol = kvp.Key
            if symbol not in current_symbols and Portfolio[symbol].Invested:
                Liquidate(symbol)
                Debug(f"Liquidating position for symbol {symbol.Value} not in current pairs.")
