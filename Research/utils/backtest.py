# backtest.py

'''
A 'Generalized' BackTesting Framework that interfaces with any strategy from 'Porfolio' class
'''

import pandas as pd

### Generalized Equitites Fees-Model
class BacktestingEngine:
    def run_backtest(self, market_data, alpha_module, beta_module, constructor_module, risk_module):
        """
        Orchestrates the full Core-Satellite backtest.
        """
        # ... setup ...

        for timestamp, current_prices in market_data.iterrows():
            
            # Station 1 & 2: Generate independent signals
            ideal_alpha_weights = alpha_module.generate_signals(current_prices)
            beta_signal_strength = beta_module.generate_beta_signal(market_data.loc[:timestamp]) # Pass historical data
            
            # Station 3: Blend the signals into a target portfolio
            blended_target_weights = constructor_module.construct_target_portfolio(
                ideal_alpha_weights, 
                beta_signal_strength
            )
            
            # Station 4: Apply final risk overlays
            final_trade_weights = risk_module.manage_portfolio(
                blended_target_weights, 
                current_portfolio_state
            )
            
            # Station 5: Execute and calculate P&L
            # ... update portfolio based on final_trade_weights ...
            
        return portfolio_history

    
### Fees-Model for BinanceEx/HyperLiquid Perp. Contract 
class PerpetualFuturesBacktester:
    
    """
    Orchestrates a backtest for a perpetual futures strategy,
    simulating trade execution, P&L, funding, and fees.
    """
    
    def __init__(self, initial_capital: float, fee_config: dict):
        self.initial_capital = initial_capital
        self.fees = fee_config  # e.g., {'maker': 0.00018, 'taker': 0.00045}
        self.execution_fee = self.fees['taker'] # Assume worst-case execution for backtesting

    def run(self, 
            market_data: pd.DataFrame, 
            funding_data: pd.DataFrame, 
            orchestrator: callable):
        
        """
        Loops through time to simulate the full strategy.

        Args:
            market_data: Multi-level column DataFrame of prices, e.g., ('BTCUSDT', 'close').
            funding_data: DataFrame of funding rates for each asset.
            orchestrator: A function that wires together all the previous modules.
        """
        
        # --- State Initialization ---
        portfolio_state = {
            "equity": self.initial_capital,
            "cash": self.initial_capital,
            "positions": {}, # e.g., {'BTCUSDT': {'size': 1.5, 'value': 90000}}
            "high_water_mark": self.initial_capital,
            "drawdown": 0.0
        }
        
        equity_curve = pd.Series(index=market_data.index, dtype=float)
        equity_curve.iloc[0] = self.initial_capital

        # --- Main Backtesting Loop ---
        for i in range(1, len(market_data.index)):
            timestamp = market_data.index[i]
            prev_timestamp = market_data.index[i-1]
            
            current_equity = portfolio_state['equity']
            pnl_from_price_change = 0.0
            pnl_from_funding = 0.0
            total_fees = 0.0

            # 1. Calculate P&L from existing positions (Mark-to-Market)
            for asset, pos_data in portfolio_state['positions'].items():
                current_price = market_data.loc[timestamp][(asset, 'close')]
                prev_price = market_data.loc[prev_timestamp][(asset, 'close')]
                pnl_from_price_change += (current_price - prev_price) * pos_data['size']

            current_equity += pnl_from_price_change

            # 2. Calculate and subtract funding costs
            if timestamp in funding_data.index:
                for asset, pos_data in portfolio_state['positions'].items():
                    position_value = pos_data['size'] * market_data.loc[timestamp][(asset, 'close')]
                    funding_rate = funding_data.loc[timestamp][asset]
                    # You PAY funding on long positions if rate is positive
                    # You RECEIVE funding on short positions if rate is positive
                    funding_payment = position_value * funding_rate
                    pnl_from_funding -= funding_payment # Subtract the payment

            current_equity += pnl_from_funding
            
            # Update portfolio state *before* calling the orchestrator
            portfolio_state['equity'] = current_equity
            portfolio_state['high_water_mark'] = max(portfolio_state['high_water_mark'], current_equity)
            portfolio_state['drawdown'] = 1 - (current_equity / portfolio_state['high_water_mark'])

            # --- Orchestration: Get Final Target Weights ---
            # The orchestrator function calls alpha, beta, constructor, and risk modules
            final_target_weights = orchestrator(timestamp, portfolio_state, market_data.loc[:timestamp])

            # --- Rebalancing Logic ---
            # 3. Determine trades needed to rebalance to final_target_weights
            for asset in set(list(portfolio_state['positions'].keys()) + list(final_target_weights.keys())):
                current_size = portfolio_state['positions'].get(asset, {}).get('size', 0.0)
                current_price = market_data.loc[timestamp][(asset, 'close')]
                
                target_weight = final_target_weights.get(asset, 0.0)
                target_value = target_weight * current_equity
                target_size = target_value / current_price if current_price != 0 else 0
                
                trade_size = target_size - current_size
                
                if abs(trade_size) > 1e-9: # Avoid tiny floating point trades
                    trade_value = abs(trade_size) * current_price
                    total_fees += trade_value * self.execution_fee

                    # Update position
                    if abs(target_size) < 1e-9: # If target is to be flat
                        portfolio_state['positions'].pop(asset, None) # Remove if exists
                    else:
                        portfolio_state['positions'][asset] = {'size': target_size, 'value': target_value}

            # 5. Update equity with fees and save the final value for the day
            current_equity -= total_fees
            portfolio_state['equity'] = current_equity
            equity_curve.iloc[i] = current_equity

        return equity_curve
