import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from IPython.display import display

class BenchmarkPortfolio:
    def __init__(self, pairs, data_universe, market_data):
        """
        Args:
            pairs (dict): Dictionary of filtered cointegrated pairs with spread series, hedge ratio, etc.
            market_data (pd.Series): Market index price series.
            price_series_dict (dict): Dictionary with keys as tickers and values as pd.Series of price data.
        """
        self.pairs = pairs
        self.data_universe = data_universe
        self.market_data = market_data
        self.returns = {}
        self.weights = {}
        self.adjusted_weights = {}
        self.portfolio_returns = None
        self.portfolio_cumulative_returns = None

    def compute_daily_returns(self):
        for pair, data in self.pairs.items():
            self.returns[pair] = data['spread_series'].pct_change().dropna()

    def compute_inverse_volatility_weights(self):
        volatilities = {pair: np.std(ret) for pair, ret in self.returns.items()}
        inv_vol = {pair: 1 / vol for pair, vol in volatilities.items()}
        total_inv_vol = sum(inv_vol.values())
        self.weights = {pair: inv_vol[pair] / total_inv_vol for pair in inv_vol}

    ## Linear-Regression of assest-returns v/s market-returns | asset = alpha + beta * market 
    def compute_beta(self, price_series):
        asset_returns = price_series.pct_change().dropna()
        market_returns = self.market_data.pct_change().dropna()
        min_len = min(len(asset_returns), len(market_returns))
        X = market_returns[-min_len:].values.reshape(-1, 1)
        y = asset_returns[-min_len:].values
        model = LinearRegression().fit(X, y)
        return model.coef_[0]

    def adjust_weights_for_beta_neutrality(self):
        beta_x = {}
        beta_y = {}

        for pair, data in self.pairs.items():
            px = self.data_universe[data['symbol_x']]['TRDPRC_1']
            py = self.data_universe[data['symbol_y']]['TRDPRC_1']
            beta_x[pair] = self.compute_beta(px)
            beta_y[pair] = self.compute_beta(py)

        for pair, weight in self.weights.items():
            hedge_ratio = self.pairs[pair]['hedge_ratio']
            beta_adj = beta_x[pair] - hedge_ratio * beta_y[pair]
            self.adjusted_weights[pair] = weight * beta_adj

        total_weight = sum(abs(w) for w in self.adjusted_weights.values())
        self.adjusted_weights = {pair: w / total_weight for pair, w in self.adjusted_weights.items()}

    def backtest_portfolio(self):
        dates = self.returns[list(self.returns.keys())[0]].index
        self.portfolio_returns = pd.Series(0, index=dates)

        for pair, weight in self.adjusted_weights.items():
            aligned_returns = self.returns[pair].reindex(dates).fillna(0)
            self.portfolio_returns += weight * aligned_returns

        ## assumes you reinvest all profits each day into the portfolio
        self.portfolio_cumulative_returns = (1 + self.portfolio_returns).cumprod()

    def evaluate_performance(self):
        sharpe_ratio = self.portfolio_returns.mean() / self.portfolio_returns.std() * np.sqrt(252)
        max_drawdown = (self.portfolio_cumulative_returns / self.portfolio_cumulative_returns.cummax() - 1).min()
        return sharpe_ratio, max_drawdown

    def plot_performance(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.portfolio_cumulative_returns, label='Portfolio')
        plt.title('Beta-Neutral, Inverse-Volatility Weighted Pairs Portfolio')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Returns')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def run(self):
        self.compute_daily_returns()
        self.compute_inverse_volatility_weights()
        self.adjust_weights_for_beta_neutrality()
        self.backtest_portfolio()
        self.plot_performance()
        sharpe_ratio, max_drawdown = self.evaluate_performance()
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        return {
            'cumulative_returns': self.portfolio_cumulative_returns,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
