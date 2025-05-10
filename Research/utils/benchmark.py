import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from IPython.display import display
from scipy.optimize import minimize

class BenchmarkPortfolio:
    def __init__(self, initial_amount, pairs, data_universe, market_data):
        """
        Args:
            pairs (dict): Dictionary of filtered cointegrated pairs with spread series, hedge ratio, etc.
            market_data (pd.Series): Market index price series.
            price_series_dict (dict): Dictionary with keys as tickers and values as pd.Series of price data.
        """
        self.initial_amount = initial_amount
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
        """
        Optimize weights to enforce beta neutrality and ADV-based position limits.
        """
        
        target_weights = np.array([self.weights[pair] for pair in self.pairs])
        pair_list = list(self.pairs.keys())

        # Compute effective betas and ADV-based limits
        effective_betas = []
        
        for pair in pair_list:
            data = self.pairs[pair]
            px = self.data_universe[data['symbol_x']]['TRDPRC_1']
            py = self.data_universe[data['symbol_y']]['TRDPRC_1']
            beta_x = self.compute_beta(px)
            beta_y = self.compute_beta(py)
            beta_spread = beta_x - data['hedge_ratio'] * beta_y
            effective_betas.append(beta_spread)

        effective_betas = np.array(effective_betas)

        # Objective: minimize deviation from target weights
        def objective(w):
            return np.sum((w - target_weights) ** 2)

        # Constraint: total gross exposure <= 1
        def leverage_constraint(w):
            return 1.0 - np.sum(np.abs(w))

        # Constraint: portfolio beta neutrality = 0
        def beta_constraint(w):
            return np.dot(w, effective_betas)

        constraints = [
            {'type': 'ineq', 'fun': leverage_constraint},
            {'type': 'eq',   'fun': beta_constraint}
        ]

        # Initial guess
        x0 = target_weights.copy()
        result = minimize(objective, x0, method='SLSQP', constraints=constraints)
        if not result.success:
            print("Optimization failed:", result.message)

        optimized_weights = result.x
        self.adjusted_weights = dict(zip(pair_list, optimized_weights))


    def backtest_portfolio(self):
        dates = self.returns[list(self.returns.keys())[0]].index
        self.portfolio_returns = pd.Series(0, index=dates)

        for pair, weight in self.adjusted_weights.items():
            aligned_returns = self.returns[pair].reindex(dates).fillna(0)
            self.portfolio_returns += weight * aligned_returns

        self.portfolio_cumulative_returns = (1 + self.portfolio_returns).cumprod()
        self.portfolio_value = self.portfolio_cumulative_returns * self.initial_amount

    def evaluate_performance(self):
        sharpe_ratio = self.portfolio_returns.mean() / self.portfolio_returns.std() * np.sqrt(252)
        max_drawdown = (self.portfolio_cumulative_returns / self.portfolio_cumulative_returns.cummax() - 1).min()
        return sharpe_ratio, max_drawdown

    def plot_performance(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.portfolio_value, label='Portfolio Value (USD)')
        plt.title('Beta-Neutral Portfolio (Benchmark) (No Fees, No Trade-Limit)')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def beta_neutral_check(self):
        market_returns = self.market_data.pct_change().dropna()
        min_len = min(len(self.portfolio_returns), len(market_returns))
        X = market_returns[-min_len:].values.reshape(-1, 1)
        y = self.portfolio_returns[-min_len:].values
        model = LinearRegression().fit(X, y)
        print('Portfolio-Beta: ', model.coef_[0])

    def run(self):
        self.compute_daily_returns()
        self.compute_inverse_volatility_weights()
        self.adjust_weights_for_beta_neutrality()
        self.backtest_portfolio()
        self.plot_performance()
        sharpe_ratio, max_drawdown = self.evaluate_performance()
        print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"Max Drawdown: {max_drawdown:.2%}")
        print(f"Portfolio Value: ${list(self.portfolio_value)[-1]:.2f}")
        return {
            'portfolio_value': list(self.portfolio_value)[-1],
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
