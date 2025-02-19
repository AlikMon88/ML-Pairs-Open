import numpy as np
import pandas as pd


def CalculateMarketVolatility():
    # Calculate rolling volatility of SPY using EMA
    history = History(spy, market_volatility_window, Resolution.Daily)
    if history.empty:
        return market_volatility
    returns = history['close'].pct_change().dropna()
    if len(returns) == 0:
        return market_volatility
    ema_variance = returns.ewm(span=market_volatility_window).var().iloc[-1]
    ema_std = np.sqrt(ema_variance)
    return ema_std

def ManagePositions():
    # Calculate market volatility scaling factor
    if market_volatility > 0:
        scaling_factor = 1.0 / market_volatility
    else:
        scaling_factor = 1.0  # Default scaling
    
    # Monitor open positions and generate signals
    for pair_key, position in open_positions.items():
        symbol_y = position['symbol_y']
        symbol_x = position['symbol_x']
        hedge_ratio = position['hedge_ratio']
        spread_mean = position['spread_mean']
        spread_std = position['spread_std']
        
        # Check if data is available
        if not (Securities.ContainsKey(symbol_y) and Securities.ContainsKey(symbol_x)):
            continue
        price_y = Securities[symbol_y].Price
        price_x = Securities[symbol_x].Price
        
        if price_y == 0 or price_x == 0:
            continue
        
        # Calculate current spread and z-score
        spread = price_y - hedge_ratio * price_x
        z_score = (spread - spread_mean) / spread_std
        
        # Adjust z-score for transaction costs
        adjusted_z = abs(z_score) - transaction_cost * 100  # Normalize transaction cost
        
        # Entry signal
        if not position['is_open']:
            if adjusted_z > entry_threshold:
                # Position sizing
                allocation = scaling_factor * (1.0 / spread_std) / len(pairs)
                allocation = min(allocation, 0.05)  # Max 5% per position
                
                # Go long spread: short x, long y
                if z_score > entry_threshold:
                    SetHoldings(symbol_y, allocation)
                    SetHoldings(symbol_x, -allocation * hedge_ratio)
                    Debug(f"Entering long spread for pair {pair_key} at z-score {z_score:.2f}")
                # Go short spread: long x, short y
                elif z_score < -entry_threshold:
                    SetHoldings(symbol_y, -allocation)
                    SetHoldings(symbol_x, allocation * hedge_ratio)
                    Debug(f"Entering short spread for pair {pair_key} at z-score {z_score:.2f}")
                position['is_open'] = True
                position['entry_z'] = z_score
        else:
            # Exit conditions
            exit_condition = abs(z_score) < exit_threshold
            stop_loss_condition = abs(z_score) > stop_loss
            take_profit_condition = (take_profit > 0) and (abs(z_score) > take_profit)
            
            if exit_condition or stop_loss_condition or take_profit_condition:
                Liquidate(symbol_y)
                Liquidate(symbol_x)
                Debug(f"Exiting position for pair {pair_key} at z-score {z_score:.2f}")
                position['is_open'] = False

def LiquidatePositionsNotInPairs(self):
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

def GetFundamentalData(self, symbol):
    # Retrieve fundamental data (P/E and P/B ratios) for the given symbol
    if symbol in Securities:
        security = Securities[symbol]
        if security.HasData and security.Fundamentals is not None:
            pe_ratio = security.Fundamentals.ValuationRatios.PERatio
            pb_ratio = security.Fundamentals.ValuationRatios.PBRatio
            if pe_ratio is not None and pb_ratio is not None:
                return {'PE': pe_ratio, 'PB': pb_ratio}
    return None

def GetSectorCode(self, symbol):
    # Get sector code for the symbol
    if symbol in Securities:
        security = Securities[symbol]
        if security.HasData and security.Fundamentals is not None:
            sector_code = security.Fundamentals.AssetClassification.MorningstarSectorCode
            return sector_code
    return 0
