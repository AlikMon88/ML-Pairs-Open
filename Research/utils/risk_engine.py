# risk_engine.py 

'''
Risk-Engine: 
We put constraints (Volume, Liquidity) / Target-Locking / Drawdown exits 
'''

class RiskManager:
    def __init__(self, risk_params):
        self.max_drawdown = risk_params['max_drawdown']
        self.max_concentration = risk_params['max_concentration']
        self.volatility_target = risk_params['volatility_target']

    def manage_portfolio(self, ideal_weights, current_portfolio_state):
        """
        Takes ideal weights and applies risk rules to produce final target weights.
        
        Args:
            ideal_weights (dict): The output from the Alpha module.
            current_portfolio_state (dict): Contains current equity, P&L, open positions, etc.
            
        Returns:
            dict: The final, risk-adjusted target weights.
        """
        # 1. Check for portfolio-level kill switch
        if current_portfolio_state['drawdown'] > self.max_drawdown:
            return {}  # Return empty dict, meaning "exit all positions"

        final_weights = ideal_weights.copy()
        
        # 2. Apply concentration limits
        for asset, weight in final_weights.items():
            if abs(weight) > self.max_concentration:
                final_weights[asset] = self.max_concentration * (1 if weight > 0 else -1)
                
        # 3. Apply volatility targeting (simplified example)
        # This is where you would calculate position sizes based on volatility.
        # For now, we'll just use the concentration-capped weights.
        
        # ... more sophisticated logic here ...
        
        return final_weights
    
class RiskManager:
    """
    Applies portfolio-level risk constraints to target weights.
    This module acts as a final safety check.
    """
    def __init__(self, max_leverage: float, max_concentration: float, max_portfolio_drawdown: float):
        self.max_leverage = max_leverage
        self.max_concentration = max_concentration
        self.max_drawdown = max_portfolio_drawdown

    def manage_weights(self, target_weights: dict, portfolio_state: dict):
        
        """
        Adjusts target weights to comply with risk rules.

        Args:
            target_weights: The blended weights from the Portfolio Constructor.
            portfolio_state: Contains current equity, drawdown, positions, etc.

        Returns:
            dict: The final, risk-approved target weights.
        """
        
        # Rule 1: Max Drawdown "Kill Switch"
        if portfolio_state.get('drawdown', 0) > self.max_drawdown:
            return {asset: 0.0 for asset in target_weights} # Exit all positions

        final_weights = target_weights.copy()
        
        # Rule 2: Concentration Limit
        for asset, weight in final_weights.items():
            if abs(weight) > self.max_concentration:
                final_weights[asset] = self.max_concentration * (1 if weight > 0 else -1)
        
        # Rule 3: Leverage Limit
        total_leverage = sum(abs(w) for w in final_weights.values())
        if total_leverage > self.max_leverage:
            scaling_factor = self.max_leverage / total_leverage
            final_weights = {asset: w * scaling_factor for asset, w in final_weights.items()}
            
        return final_weights