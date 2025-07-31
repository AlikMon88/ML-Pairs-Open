''' 
we create Syntheic Indexes for 'market-proxy' - to hedge any heavy reliance on single asset
'''

from .retrieve import get_top_100_liquid_binance_pairs 
import numpy as np
import pandas as pd

class MarketProxy():
    def __init__(self, df_uni, top_n=3):
        self.df_uni = df_uni
        self.top_n = top_n
    
    def call_n_tokens(self):
        top_n_list = get_top_100_liquid_binance_pairs()
        # return top_n_list[:self.top_n]
        return top_n_list[:3]
    
    def generate_geometric_weights(self, ratio):
        """
        Generates a series of n weights that decrease geometrically and sum to 1.

        Args:
            n (int): The number of assets to generate weights for.
            ratio (float): The geometric ratio between consecutive weights (e.g., 0.9).
                        Must be between 0 and 1.

        Returns:
            np.ndarray: A numpy array of n weights that sums to 1.0.
        """
        
        if not 0 < ratio < 1:
            raise ValueError("Ratio must be between 0 and 1.")
        if self.top_n <= 0:
            return np.array([])

        # Generate the raw geometric series: 1, r, r^2, r^3, ...
        raw_weights = np.array([ratio**i for i in range(self.top_n)])
        
        # Normalize the weights so they sum to 1
        normalized_weights = raw_weights / np.sum(raw_weights)
        
        return normalized_weights

    def get_syn_index(self, ret_beta_basket=False):
        
        ### For now I have fixed top_n=3
        BETA_BASKET_WEIGHTS = {'BTCUSDT': 0.6, 'ETHUSDT': 0.3, 'SOLUSDT': 0.1} ## weighting for market-capitalization | or maybe a geometric series
        
        ## Geometrics-Degression / ratio = 0.8 / subs. asset to have 80% of weight of prev. asset
        weights = self.generate_geometric_weights(ratio=0.8)
        tokens = self.call_n_tokens()
        BETA_BASKET_WEIGHTS = {key:value for key, value in zip(tokens, weights)}
        
        market_proxy_series = 0
        for asset, weight in BETA_BASKET_WEIGHTS.items():
            market_proxy_series += self.df_uni.loc[asset]['close'] * weight
            
        if ret_beta_basket:
            return market_proxy_series, tokens 
        
        return market_proxy_series

                
        
