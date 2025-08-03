# backtest.py

'''
A 'Generalized' BackTesting Framework that interfaces with any strategy from 'Porfolio' class
'''

import pandas as pd
import random 
import numpy as np
    
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
            history_data: pd.DataFrame, 
            funding_data: pd.DataFrame, 
            orchestrator: callable):
        
        """
        Loops through time to simulate the full strategy.

        Args:
            history_data: Multi-level column DataFrame of prices, e.g., ('BTCUSDT', 'close').
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
        
        equity_curve = pd.Series(index=history_data.index, dtype=float)
        equity_curve.iloc[0] = self.initial_capital

        rn = random.randint(0, len(history_data) - 1)
        
        print('Backtesting ...')
        print('start-timestamp: ', history_data.index[rn][-1])
        print('end-timestamp: ', history_data.index[-1][-1])
        
        alpha_signals, beta_signal = {}, 0
        
        # --- Main Backtesting Loop ---
        for i in range(rn, len(history_data.index)):
            timestamp = history_data.index[i][-1]
            prev_timestamp = history_data.index[i-1][-1]
            
            current_equity = portfolio_state['equity']
            pnl_from_price_change = 0.0
            pnl_from_funding = 0.0
            total_fees = 0.0

            # 1. Calculate P&L from existing positions (Mark-to-Market)
            for asset, pos_data in portfolio_state['positions'].items():
                # current_price = history_dataloc[timestamp][(asset, 'close')]
                current_price = history_data.loc[(asset, timestamp)].close
                
                # prev_price = history_dataloc[prev_timestamp][(asset, 'close')]
                prev_price = history_data.loc[(asset, prev_timestamp)].close
                
                pnl_from_price_change += (current_price - prev_price) * pos_data['size']

            current_equity += pnl_from_price_change

            # 2. Calculate and subtract funding costs
            if timestamp in funding_data.index:
                for asset, pos_data in portfolio_state['positions'].items():
                    # position_value = pos_data['size'] * history_dataloc[timestamp][(asset, 'close')]
                    position_value = pos_data['size'] * history_data.loc[(asset, timestamp)].close
                    
                    # funding_rate = funding_data.loc[timestamp][asset]
                    funding_rate = funding_data.loc[(asset, timestamp)].fundingRate
                    
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
            final_target_weights, alpha_signals, beta_signal = orchestrator(timestamp, portfolio_state, alpha_signals, beta_signal)

            # --- Rebalancing Logic ---
            # 3. Determine trades needed to rebalance to final_target_weights
            for asset in set(list(portfolio_state['positions'].keys()) + list(final_target_weights.keys())):
                current_size = portfolio_state['positions'].get(asset, {}).get('size', 0.0)
                # current_price = history_dataloc[timestamp][(asset, 'close')]
                current_price = history_data.loc[(asset, timestamp)].close
                
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
