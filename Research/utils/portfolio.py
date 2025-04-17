import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class VolatilityWeightedPortfolioManager:
    def __init__(self, history_provider, portfolio, broker):
        """
        Initialize the portfolio manager with history, portfolio and broker objects
        
        Args:
            history_provider: Object to get historical price data
            portfolio: Object containing portfolio information
            broker: Object to execute trades
        """
        self.history_provider = history_provider
        self.portfolio = portfolio
        self.broker = broker
        self.current_allocations = {}
        self.portfolio_history = []
        
    def calculate_volatility(self, symbol, lookback_days=30):
        """
        Calculate historical volatility for a symbol
        
        Args:
            symbol: The asset symbol
            lookback_days: Number of days to use for volatility calculation
            
        Returns:
            volatility: Annualized volatility
        """
        # Get historical daily prices
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days * 1.5)  # Extra days for calculation
        
        history = self.history_provider.get_price_history(
            symbol, 
            start_date, 
            end_date, 
            resolution='daily'
        )
        
        if history is None or len(history) < lookback_days:
            raise ValueError(f"Insufficient price history for {symbol}")
        
        # Calculate daily returns
        history['returns'] = history['close'].pct_change().dropna()
        
        # Calculate volatility (standard deviation of returns)
        daily_volatility = history['returns'].tail(lookback_days).std()
        
        # Annualize volatility (assuming 252 trading days)
        annualized_volatility = daily_volatility * np.sqrt(252)
        
        return annualized_volatility
    
    def create_inverse_volatility_portfolio(self, symbols, max_allocation=0.30, min_allocation=0.05):
        """
        Create a portfolio weighted by inverse volatility
        
        Args:
            symbols: List of symbols to include in portfolio
            max_allocation: Maximum allocation to any single asset
            min_allocation: Minimum allocation to include an asset
            
        Returns:
            allocations: Dictionary of {symbol: allocation}
        """
        if not symbols:
            return {}
            
        # Calculate volatility for each symbol
        volatilities = {}
        valid_symbols = []
        
        for symbol in symbols:
            try:
                vol = self.calculate_volatility(symbol)
                volatilities[symbol] = vol
                valid_symbols.append(symbol)
            except Exception as e:
                print(f"Error calculating volatility for {symbol}: {e}")
        
        if not valid_symbols:
            return {}
            
        # Calculate inverse volatility
        inverse_vols = {symbol: 1.0/vol for symbol, vol in volatilities.items()}
        
        # Calculate weights based on inverse volatility
        total_inverse_vol = sum(inverse_vols.values())
        weights = {symbol: inv_vol/total_inverse_vol for symbol, inv_vol in inverse_vols.items()}
        
        # Apply maximum and minimum allocation constraints
        allocations = self._apply_allocation_constraints(weights, max_allocation, min_allocation)
        
        return allocations
    
    def _apply_allocation_constraints(self, weights, max_allocation, min_allocation):
        """
        Apply maximum and minimum allocation constraints
        
        Args:
            weights: Initial weights dictionary
            max_allocation: Maximum allocation to any single asset
            min_allocation: Minimum allocation to include an asset
            
        Returns:
            allocations: Adjusted allocation dictionary
        """
        # Apply maximum constraint and remove small allocations
        filtered_weights = {}
        excluded_weight = 0
        
        for symbol, weight in weights.items():
            if weight > max_allocation:
                filtered_weights[symbol] = max_allocation
                excluded_weight += weight - max_allocation
            elif weight < min_allocation:
                excluded_weight += weight
            else:
                filtered_weights[symbol] = weight
        
        if not filtered_weights:
            # If all weights were filtered out, use the original with just max constraint
            filtered_weights = {s: min(w, max_allocation) for s, w in weights.items()}
        
        # Redistribute excluded weight
        if excluded_weight > 0 and filtered_weights:
            total_filtered_weight = sum(filtered_weights.values())
            for symbol in filtered_weights:
                # Proportionally redistribute excess weight
                filtered_weights[symbol] += excluded_weight * (filtered_weights[symbol] / total_filtered_weight)
        
        # Normalize to ensure sum is 1.0
        total_weight = sum(filtered_weights.values())
        normalized_weights = {s: w/total_weight for s, w in filtered_weights.items()}
        
        return normalized_weights
    
    def create_equal_risk_contribution_portfolio(self, symbols, target_risk=0.10):
        """
        Create a portfolio where each asset contributes equal risk
        
        Args:
            symbols: List of symbols to include in portfolio
            target_risk: Target portfolio risk level (annualized volatility)
            
        Returns:
            allocations: Dictionary of {symbol: allocation}
        """
        # This is a simplified implementation of ERC
        # A full implementation would use optimization and correlation matrix
        
        # Calculate volatility for each symbol
        volatilities = {}
        for symbol in symbols:
            try:
                vol = self.calculate_volatility(symbol)
                volatilities[symbol] = vol
            except Exception as e:
                print(f"Error calculating volatility for {symbol}: {e}")
        
        # Calculate weights (proportional to 1/volatility)
        inv_vols = {symbol: 1.0/vol for symbol, vol in volatilities.items()}
        total_inv_vol = sum(inv_vols.values())
        weights = {symbol: inv_vol/total_inv_vol for symbol, inv_vol in inv_vols.items()}
        
        # Scale weights to match target risk (assuming no correlation)
        port_vol = np.sqrt(sum([(weights[s]**2) * (volatilities[s]**2) for s in weights]))
        scaling_factor = target_risk / port_vol
        
        scaled_weights = {s: w * scaling_factor for s, w in weights.items()}
        
        # If sum > 1.0, normalize
        total_weight = sum(scaled_weights.values())
        if total_weight > 1.0:
            scaled_weights = {s: w/total_weight for s, w in scaled_weights.items()}
            
        return scaled_weights
    
    def apply_portfolio_allocations(self, allocations, tolerance=0.005):
        """
        Apply the calculated allocations to the portfolio
        
        Args:
            allocations: Dictionary of {symbol: allocation}
            tolerance: Small buffer to prevent unnecessary trades
            
        Returns:
            orders: List of orders executed
        """
        if not allocations:
            return []
            
        portfolio_value = self.portfolio.get_total_portfolio_value()
        orders = []
        
        for symbol, target_pct in allocations.items():
            # Get current position information
            current_position = self.portfolio.get_position(symbol)
            current_shares = current_position.quantity if current_position else 0
            current_price = self.broker.get_last_price(symbol)
            
            if current_price <= 0:
                print(f"Invalid price for {symbol}: {current_price}")
                continue
                
            current_value = current_shares * current_price
            current_pct = current_value / portfolio_value if portfolio_value > 0 else 0
            
            # Check if adjustment is within tolerance
            if abs(current_pct - target_pct) <= tolerance:
                continue
                
            # Calculate the value difference and required shares
            target_value = portfolio_value * target_pct
            value_difference = target_value - current_value
            shares_to_trade = int(value_difference / current_price)
            
            # Execute the order if shares_to_trade is not zero
            if shares_to_trade != 0:
                # Positive: buy, Negative: sell
                if shares_to_trade > 0:
                    order_id = self.broker.place_market_order(symbol, shares_to_trade, "buy")
                else:
                    order_id = self.broker.place_market_order(symbol, abs(shares_to_trade), "sell")
                
                orders.append({
                    'symbol': symbol,
                    'order_id': order_id,
                    'shares': shares_to_trade,
                    'target_pct': target_pct
                })
                
                # Update current allocations
                self.current_allocations[symbol] = target_pct
        
        # Record portfolio state
        self._record_portfolio_state(allocations)
        
        return orders
    
    def _record_portfolio_state(self, target_allocations):
        """
        Record current portfolio state for tracking
        """
        portfolio_value = self.portfolio.get_total_portfolio_value()
        
        portfolio_state = {
            'timestamp': datetime.now(),
            'portfolio_value': portfolio_value,
            'cash': self.portfolio.get_cash(),
            'positions': []
        }
        
        # Record all positions
        for symbol in set(list(target_allocations.keys()) + list(self.current_allocations.keys())):
            position = self.portfolio.get_position(symbol)
            
            if position:
                current_price = self.broker.get_last_price(symbol)
                current_value = position.quantity * current_price
                current_pct = current_value / portfolio_value if portfolio_value > 0 else 0
                
                vol = self.calculate_volatility(symbol)
                
                position_data = {
                    'symbol': symbol,
                    'quantity': position.quantity,
                    'price': current_price,
                    'value': current_value,
                    'current_allocation': current_pct,
                    'target_allocation': target_allocations.get(symbol, 0),
                    'volatility': vol,
                }
                
                portfolio_state['positions'].append(position_data)
        
        self.portfolio_history.append(portfolio_state)
    
    def get_current_portfolio_df(self):
        """
        Get current portfolio as a DataFrame
        
        Returns:
            df: DataFrame with current portfolio information
        """
        if not self.portfolio_history:
            return pd.DataFrame()
            
        current_state = self.portfolio_history[-1]
        
        # Create positions dataframe
        positions_df = pd.DataFrame(current_state['positions'])
        
        if positions_df.empty:
            return pd.DataFrame()
            
        # Add portfolio summary
        positions_df['portfolio_value'] = current_state['portfolio_value']
        positions_df['timestamp'] = current_state['timestamp']
        positions_df['cash'] = current_state['cash']
        positions_df['cash_allocation'] = current_state['cash'] / current_state['portfolio_value']
        
        # Calculate risk contribution (simplified)
        if 'volatility' in positions_df.columns and 'current_allocation' in positions_df.columns:
            positions_df['risk_contribution'] = positions_df['volatility'] * positions_df['current_allocation']
            total_risk = positions_df['risk_contribution'].sum()
            positions_df['risk_pct'] = positions_df['risk_contribution'] / total_risk if total_risk > 0 else 0
        
        return positions_df
    
    def get_portfolio_history_df(self):
        """
        Get historical portfolio data as a DataFrame
        
        Returns:
            df: DataFrame with portfolio history
        """
        if not self.portfolio_history:
            return pd.DataFrame()
            
        all_positions = []
        
        for state in self.portfolio_history:
            for position in state['positions']:
                position_data = {
                    'timestamp': state['timestamp'],
                    'portfolio_value': state['portfolio_value'],
                    'cash': state['cash'],
                    'symbol': position['symbol'],
                    'quantity': position['quantity'],
                    'price': position['price'],
                    'value': position['value'],
                    'allocation': position['current_allocation'],
                    'target_allocation': position['target_allocation'],
                    'volatility': position.get('volatility', None)
                }
                all_positions.append(position_data)
        
        return pd.DataFrame(all_positions)
    
    def export_portfolio_to_csv(self, filename=None):
        """
        Export current portfolio to CSV file
        
        Args:
            filename: CSV filename, defaults to 'portfolio_{timestamp}.csv'
            
        Returns:
            path: Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'portfolio_{timestamp}.csv'
            
        df = self.get_current_portfolio_df()
        
        if not df.empty:
            df.to_csv(filename, index=False)
            return filename
        
        return None
    
    def liquidate_all(self):
        """
        Liquidate all positions
        
        Returns:
            orders: List of order IDs
        """
        order_ids = []
        
        for position in self.portfolio.get_positions():
            if position.quantity != 0:
                direction = "sell" if position.quantity > 0 else "buy"
                order_id = self.broker.place_market_order(
                    position.symbol, 
                    abs(position.quantity), 
                    direction
                )
                order_ids.append(order_id)
        
        # Clear current allocations
        self.current_allocations = {}
        
        # Record portfolio state
        self._record_portfolio_state({})
        
        return order_ids

# # Example usage in a QC algorithm
# class VolatilityWeightedAlgorithm:
#     def Initialize(self):
#         self.SetStartDate(2020, 1, 1)
#         self.SetEndDate(2023, 1, 1)
#         self.SetCash(100000)
        
#         # Add symbols
#         self.symbols = ['SPY', 'QQQ', 'TLT', 'GLD', 'VNQ']
#         for symbol in self.symbols:
#             self.AddEquity(symbol, Resolution.Daily)
        
#         # Create portfolio manager
#         self.portfolio_manager = VolatilityWeightedPortfolioManager(
#             history_provider=self,  # QC algorithm has history methods
#             portfolio=self.Portfolio,
#             broker=self
#         )
        
#         # Schedule portfolio rebalancing
#         self.Schedule.On(self.DateRules.MonthStart(), 
#                          self.TimeRules.AfterMarketOpen('SPY'), 
#                          self.Rebalance)
    
#     def OnData(self, data):
#         # Main algorithm logic
#         pass
    
#     def Rebalance(self):
#         # Create inverse volatility portfolio
#         allocations = self.portfolio_manager.create_inverse_volatility_portfolio(
#             self.symbols,
#             max_allocation=0.30,
#             min_allocation=0.05
#         )
        
#         # Apply the allocations
#         orders = self.portfolio_manager.apply_portfolio_allocations(allocations)
        
#         # Log the portfolio
#         portfolio_df = self.portfolio_manager.get_current_portfolio_df()
#         self.Log(f"Portfolio rebalanced. Current allocations: {allocations}")
        
#         # Export to CSV (in live trading)
#         if self.LiveMode:
#             csv_path = self.portfolio_manager.export_portfolio_to_csv()
#             self.Log(f"Portfolio exported to {csv_path}")