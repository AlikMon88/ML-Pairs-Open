import numpy as np
import pandas as pd
from .trading import Trading
from .trading import Portfolio

def CalculateMarketVolatility(benchmark_history, market_volatility = 0.0, market_volatility_window = 63): ### implied-volatility based on history --> approximating realised volatility
    
    if benchmark_history.empty:
        return market_volatility
    
    returns = benchmark_history['TRDPRC_1'].pct_change().dropna()
    
    if len(returns) == 0:
        return market_volatility
    
    ema_variance = returns.ewm(span=market_volatility_window).var().iloc[-1]
    ema_std = np.sqrt(ema_variance)
    
    return ema_std

class ManagePosition:

    def __init__(self, market_volatility, transaction_cost = 0.01, entry_threshold = 2.0, exit_threshold = 0.3, stop_loss = 3.0, take_profit = 0.0): ### based on std. deviation from a std normal distribution
        
        self.market_volatility = market_volatility
        self.transaction_cost = transaction_cost 
        self.entry_threshold = entry_threshold 
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.rebalanceFlag = True

    def create_position(self, pairs, return_position=False):

        if not self.rebalanceFlag:
            self.update_positions()
            return
        
        self.rebalanceFlag = False
        print("Rebalancing triggered.")        
        
        # Prepare for trading / Trade Initialization
        open_positions = {}
        for pair_key, pair_info in pairs.items():
            symbol_y = pair_info['symbol_y']
            symbol_x = pair_info['symbol_x']
            open_positions[pair_key] = {
                'symbol_y': symbol_y,
                'symbol_x': symbol_x,
                'hedge_ratio': pair_info['hedge_ratio'],
                'spread_mean': pair_info['spread_mean'],
                'spread_std': pair_info['spread_std'],
                'is_open': False
            }

        # Calculate Risk-Parity Weights based on inverse spread volatility
        inverse_volatilities = {pair: 1.0 / info['spread_std'] for pair, info in self.pairs.items()}
        total_inverse_vol = sum(inverse_volatilities.values())
        weights = {pair: vol / total_inverse_vol for pair, vol in inverse_volatilities.items()}

        # Calculate market volatility scaling factor
        if self.market_volatility > 0:
            self.scaling_factor = 1.0 / self.market_volatility
        else:
            self.scaling_factor = 1.0 
        
        # Allocate holdings based on weights and scaling factor
        for pair_key, weight in weights.items():
            position = open_positions[pair_key]
            symbol_y = position['symbol_y']
            symbol_x = position['symbol_x']

            allocation = weight * self.scaling_factor
            
            # Limit allocation to prevent overexposure
            allocation = min(allocation, 0.05)  # Max 5% per position
            
            # Initialize positions with zero holdings
            Trading.set_holding(symbol_y, 0)
            Trading.set_holding(symbol_x, 0)

        if return_position:
            return open_positions

    def update_positions(self, Securities, open_positions, pairs):

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
                    allocation = self.scaling_factor * (1.0 / spread_std) / len(pairs)
                    allocation = min(allocation, 0.05)  # Max 5% per position
                    
                    # Go long spread: short x, long y
                    if z_score > self.entry_threshold:
                        Trading.set_holding(symbol_y, allocation)
                        Trading.set_holding(symbol_x, -allocation * hedge_ratio)
                        print(f"Entering long spread for pair {pair_key} at z-score {z_score:.2f}")

                    # Go short spread: long x, short y
                    elif z_score < - self.entry_threshold:
                        Trading.set_holding(symbol_y, -allocation)
                        Trading.set_holding(symbol_x, allocation * hedge_ratio)
                        print(f"Entering short spread for pair {pair_key} at z-score {z_score:.2f}")
                    position['is_open'] = True
                    position['entry_z'] = z_score
            else:
                # Exit conditions
                exit_condition = abs(z_score) < self.exit_threshold
                stop_loss_condition = abs(z_score) > self.stop_loss
                take_profit_condition = (self.take_profit > 0) and (abs(z_score) > self.take_profit)
                
                if exit_condition or stop_loss_condition or take_profit_condition:
                    Trading.liquidate(symbol_y)
                    Trading.liquidate(symbol_x)
                    print(f"Exiting position for pair {pair_key} at z-score {z_score:.2f}")
                    position['is_open'] = False

    def LiquidatePositionsNotInPairs(self, pairs):
        # Liquidate positions that are not in the current pairs
        current_symbols = set()
        for pair_info in pairs.values():
            current_symbols.add(pair_info['symbol_y'])
            current_symbols.add(pair_info['symbol_x'])
        
        ### Need to correct it
        portfoilo_arr = Portfolio.portfolio()
        for kvp in portfoilo_arr:
            symbol = kvp.Key
            if symbol not in current_symbols and Portfoilo.portfolio[symbol].Invested:
                Trading.liquidate(symbol)
                print(f"Liquidating position for symbol {symbol.Value} not in current pairs.")
