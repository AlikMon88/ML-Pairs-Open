# src/beta_generator.py
import pandas as pd

class SimpleMarketBeta:
    """
    Generates a simple binary signal for long-term market beta exposure.
    """
    def __init__(self, market_proxy: pd.Series, long_term_ma: int):
        self.market_proxy = market_proxy
        self.long_term_ma = long_term_ma

    def generate_signal(self, timestamp: pd.Timestamp):
        """
        Determines if the market is in a long-term uptrend.

        Args:
            timestamp: The current time point of the backtest.

        Returns:
            float: 1.0 for beta exposure, 0.0 for no beta exposure.
        """
        market_series = self.market_proxy.loc[:timestamp]
        if len(market_series) < self.long_term_ma:
            return 0.0

        current_price = market_series.iloc[-1]
        moving_average = market_series.rolling(window=self.long_term_ma).mean().iloc[-1]

        return 1.0 if current_price > moving_average else 0.0