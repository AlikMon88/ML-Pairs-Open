# src/beta_generator.py
import pandas as pd

class SimpleMarketBeta:
    """
    Generates a stateful {1, 0, -1} signal for MARKET_REGIME (Bull, Neutral, Bear)
    based on a dual moving average crossover system.
    """
    def __init__(self, market_proxy: pd.Series, fast_ma_period: int, slow_ma_period: int):
        """
        Args:
            market_proxy: A pd.Series representing the overall market (e.g., a custom index).
            fast_ma_period: The lookback period for the shorter-term moving average.
            slow_ma_period: The lookback period for the longer-term moving average.
        """
        if fast_ma_period >= slow_ma_period:
            raise ValueError("Fast MA period must be less than Slow MA period.")
            
        self.market_proxy = market_proxy
        # Pre-compute moving averages for efficiency
        self.fast_ma = self.market_proxy.rolling(window=fast_ma_period).mean()
        self.slow_ma = self.market_proxy.rolling(window=slow_ma_period).mean()

    def generate_signal(self, timestamp: pd.Timestamp, prev_signal: float):
        """
        Determines the current market regime based on the MA crossover.

        Args:
            timestamp: The current time point of the backtest.
            prev_signal: The signal from the previous timestamp (the state).

        Returns:
            float: 1.0 for Bull, -1.0 for Bear, or 0.0 for Neutral regime.
        """
        # --- Data Check ---
        # Check if we have enough historical data to compute the slow MA
        if timestamp not in self.slow_ma.index or pd.isna(self.slow_ma.loc[timestamp]):
            return 0.0 # Not enough data, remain neutral

        fast_ma_value = self.fast_ma.loc[timestamp]
        slow_ma_value = self.slow_ma.loc[timestamp]
        
        target_signal = prev_signal # By default, hold the previous state

        # --- State Machine Logic ---
        
        # Condition for entering a BULL market (Golden Cross)
        if fast_ma_value > slow_ma_value and prev_signal <= 0:
            target_signal = 1.0
        
        # Condition for entering a BEAR market (Death Cross)
        elif fast_ma_value < slow_ma_value and prev_signal >= 0:
            target_signal = -1.0
        
        # Optional: A rule to go neutral if the MAs are very close (choppy market)
        # This helps reduce whipsaws in a sideways market.
        # For example, if the MAs are within 0.5% of each other.
        if abs(fast_ma_value - slow_ma_value) / slow_ma_value < 0.005:
            target_signal = 0.0
            
        return target_signal