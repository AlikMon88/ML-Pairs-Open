# risk_engine.py 

'''
Risk-Engine: 
We put constraints (Volume, Liquidity) / Target-Locking / Drawdown exits 
'''
  
class RiskManager:
    """
    Applies portfolio-level risk constraints to the *total proposed portfolio*,
    considering both existing positions and new target weights.
    This module acts as a final safety check.
    """
    def __init__(self, max_leverage: float, max_concentration: float, max_portfolio_drawdown: float):
        self.max_leverage = max_leverage
        self.max_concentration = max_concentration
        self.max_drawdown = max_portfolio_drawdown

    def manage_weights(self, target_weights: dict, portfolio_state: dict):
        """
        Adjusts target weights to comply with risk rules by evaluating the final combined portfolio.

        Args:
            target_weights: The ideal weights for assets targeted by new signals.
            portfolio_state: The complete current state of the portfolio (equity, positions, etc.).

        Returns:
            dict: The final, risk-approved target weights for the entire portfolio.
        """
        
        # print('Drawdown: ', portfolio_state['drawdown'])
        
        # Rule 1: Max Drawdown "Kill Switch" - This rule is stateful and correct as is.
        # At Kill-Switch max-drawdown we expire the strategy (No-More trade)
        if portfolio_state.get('drawdown', 0) > self.max_drawdown:
            # Generate zero-weight targets for ALL existing positions to signal a full exit.
            all_assets_to_close = list(portfolio_state.get('positions', {}).keys())
            return {asset: 0.0 for asset in all_assets_to_close}

        # --- CORE CORRECTION: Create a complete picture of the proposed portfolio ---
        # 1. Start with the current positions, converted to weights.
        equity = portfolio_state.get('equity', 1) # Avoid division by zero
        proposed_weights = {
            asset: pos_data['value'] / equity
            for asset, pos_data in portfolio_state.get('positions', {}).items()
        }

        # 2. Update with new targets. This correctly reflects the desired *final* state.
        # If an asset is already in a position, its weight is overwritten by the new target.
        # If the target is for a new asset, it's added to the portfolio.
        proposed_weights.update(target_weights)

        # Now, apply all subsequent risk checks to this complete 'proposed_weights' dictionary.
        final_weights = proposed_weights.copy()
        
        # Rule 2: Concentration Limit (Applied to the total proposed position)
        for asset, weight in final_weights.items():
            if abs(weight) > self.max_concentration:
                final_weights[asset] = self.max_concentration * (1 if weight > 0 else -1)
        
        # Rule 3: Leverage Limit (Applied to the total proposed portfolio)
        total_leverage = sum(abs(w) for w in final_weights.values())
        if total_leverage > self.max_leverage:
            # If leverage is breached, scale down the *entire* proposed portfolio proportionally.
            scaling_factor = self.max_leverage / total_leverage
            final_weights = {asset: w * scaling_factor for asset, w in final_weights.items()}
            
        # print('hedged-port-weights: ', final_weights)
        return final_weights
