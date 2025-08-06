# src/beta_generator.py
import pandas as pd

class SimpleMarketBeta:
    """
    Generates a stateful {1, 0, -1} signal for MARKET_REGIME (Bull, Neutral, Bear)
    based on a dual moving average crossover system with two layers of timing.
    """
    def __init__(self, market_proxy: pd.Series, market_regime_fast: int, market_regime_slow: int, exit_fast_ma_period: int, exit_slow_ma_period: int):
        """
        Args:
            market_proxy: A pd.Series representing the overall market (e.g., a custom index).
            market_regime_fast: The fast MA for determining the long-term market regime.
            market_regime_slow: The slow MA for determining the long-term market regime.
            exit_fast_ma_period: The fast MA for timing entries and exits within a regime.
            exit_slow_ma_period: The slow MA for timing entries and exits within a regime.
        """
        if (exit_fast_ma_period >= exit_slow_ma_period) or (market_regime_fast >= market_regime_slow):
            raise ValueError("Fast MA period must be less than Slow MA period.")
            
        self.market_proxy = market_proxy
    
        # Pre-compute moving averages for efficiency
        self.exit_fast_ma = self.market_proxy.rolling(window=exit_fast_ma_period).mean()
        self.exit_slow_ma = self.market_proxy.rolling(window=exit_slow_ma_period).mean()

        # Pre-compute moving averages for efficiency
        self.market_regime_fast_ma = self.market_proxy.rolling(window=market_regime_fast).mean()
        self.market_regime_slow_ma = self.market_proxy.rolling(window=market_regime_slow).mean()

    def generate_signal(self, timestamp: pd.Timestamp, prev_signal: float):
        """
        Determines the current market regime based on the MA crossover.
        Its better to switch ONLY long/short based on -> Long MA regime-crossover +
        Entry/Exit positiion based lowerEMA crossover
        """
        # --- Data Check ---
        if (timestamp not in self.market_regime_slow_ma.index or 
            pd.isna(self.market_regime_slow_ma.loc[timestamp]) or
            pd.isna(self.exit_slow_ma.loc[timestamp])):
            return 0.0 # Not enough data, remain neutral

        # Get the MA values for the current timestamp
        exit_fast_ma_value = self.exit_fast_ma.loc[timestamp]
        exit_slow_ma_value = self.exit_slow_ma.loc[timestamp]
        
        market_regime_fast_ma_value = self.market_regime_fast_ma.loc[timestamp]
        market_regime_slow_ma_value = self.market_regime_slow_ma.loc[timestamp]
        
        target_signal = prev_signal # By default, hold the previous state

        # This check happens every time to see what "mode" we are in today.
        market_is_bullish = market_regime_fast_ma_value > market_regime_slow_ma_value
        market_is_bearish = market_regime_fast_ma_value < market_regime_slow_ma_value

        # --- State Machine Logic (Refactored for clarity and correctness) ---

        # === CHECKING FOR EXITS FIRST (HIGHEST PRIORITY) ===
        ## 2% consolidation-period
        is_choppy = (abs(exit_fast_ma_value - exit_slow_ma_value) / exit_slow_ma_value) < 0.005
        
        if prev_signal == 1: # If we are currently LONG
            # Exit if the fast MA crosses below the slow MA OR if the market becomes choppy
            if exit_fast_ma_value < exit_slow_ma_value or is_choppy:
                target_signal = 0.0
        
        elif prev_signal == -1: # If we are currently SHORT
            # Exit if the fast MA crosses above the slow MA OR if the market becomes choppy
            if exit_fast_ma_value > exit_slow_ma_value or is_choppy:
                target_signal = 0.0
                
        # === CHECKING FOR ENTRIES (ONLY IF WE ARE CURRENTLY FLAT) ===
        elif prev_signal == 0:
            # Condition for entering a LONG position (Golden Cross)
            if market_is_bullish and exit_fast_ma_value > exit_slow_ma_value:
                target_signal = 1.0
            
            # Condition for entering a SHORT position (Death Cross)
            elif market_is_bearish and exit_fast_ma_value < exit_slow_ma_value:
                # Note: Shorting is costlier. The decision to take this signal
                target_signal = -1.0
        
        return target_signal