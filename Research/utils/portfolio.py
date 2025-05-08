import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

class SignalPortfolio:
    def __init__(self, initial_amount, pairs, data_universe, market_data):
        
        self.initial_amount = initial_amount
        self.pairs = pairs
        self.data_universe = data_universe  # dict of symbol: dataframe
        self.market_data = market_data
        self.returns = {}
        self.weights = {}
        self.adjusted_weights = {}
        self.portfolio_returns = None
        self.portfolio_cumulative_returns = None

    def compute_daily_returns(self):
        for pair, data in self.pairs.items():
            self.returns[pair] = data['spread_series'].pct_change().dropna()

    ### lower portfolio weightage to highly volatile assets (pair-spread based)
    def compute_inverse_volatility_weights(self):
        volatilities = {pair: np.std(ret) for pair, ret in self.returns.items()}
        inv_vol = {pair: 1 / vol for pair, vol in volatilities.items()}
        total_inv_vol = sum(inv_vol.values())
        self.weights = {pair: inv_vol[pair] / total_inv_vol for pair in inv_vol}

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
            # Effective beta of the spread = beta_x - hedge * beta_y
            beta_adj = beta_x[pair] - hedge_ratio * beta_y[pair]
            self.adjusted_weights[pair] = weight * beta_adj

        total_weight = sum(abs(w) for w in self.adjusted_weights.values())
        self.adjusted_weights = {pair: w / total_weight for pair, w in self.adjusted_weights.items()}

    def backtest_with_signals(self, entry_z=1.0, exit_z=0.2):
        dates = self.returns[list(self.returns.keys())[0]].index
        self.portfolio_returns = pd.Series(0, index=dates)
        position_tracker = {pair: 0 for pair in self.pairs}
        z_scores = {}

        for pair, data in self.pairs.items():
            spread = data['spread_series']
            z = (spread - data['spread_mean']) / data['spread_std']
            z_scores[pair] = z.reindex(dates).fillna(0)

        for t in range(1, len(dates)):
            daily_return = 0
            for pair in self.pairs:
                z = z_scores[pair].iloc[t]
                ret = self.returns[pair].iloc[t]
                pos = position_tracker[pair]
                weight = self.adjusted_weights[pair]

                if pos == 0:
                    if z > entry_z: ## Short spread (synthetic asset)
                        position_tracker[pair] = -1
                    elif z < -entry_z: ## Long spread (synthetic asset)
                        position_tracker[pair] = 1
                elif pos == 1 and z > -exit_z:
                    position_tracker[pair] = 0
                elif pos == -1 and z < exit_z:
                    position_tracker[pair] = 0

                daily_return += pos * weight * ret

            self.portfolio_returns.iloc[t] = daily_return

        self.portfolio_cumulative_returns = (1 + self.portfolio_returns).cumprod()
        self.portfolio_value = self.portfolio_cumulative_returns * self.initial_amount

    def evaluate_performance(self):
        sharpe = self.portfolio_returns.mean() / self.portfolio_returns.std() * np.sqrt(252)
        drawdown = (self.portfolio_cumulative_returns / self.portfolio_cumulative_returns.cummax() - 1).min()
        return sharpe, drawdown

    def plot_performance(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.portfolio_value, label="Portfolio")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.title("Beta-Neutral Signal-Based Portfolio")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_pair_signals(self, pair_name, entry_z=1.0, exit_z=0.2):
        pair_data = self.pairs[pair_name]
        spread = pair_data['spread_series']
        mean = pair_data['spread_mean']
        std = pair_data['spread_std']

        z_score = (spread - mean) / std
        position = 0
        entries, exits = [], []

        for t in range(1, len(z_score)):
            z = z_score.iloc[t]

            if position == 0:
                if z > entry_z:
                    entries.append((z_score.index[t], spread.iloc[t], 'short'))
                    position = -1
                elif z < -entry_z:
                    entries.append((z_score.index[t], spread.iloc[t], 'long'))
                    position = 1
            elif position == 1 and z > -exit_z:
                exits.append((z_score.index[t], spread.iloc[t]))
                position = 0
            elif position == -1 and z < exit_z:
                exits.append((z_score.index[t], spread.iloc[t]))
                position = 0

        plt.figure(figsize=(8, 4))
        plt.plot(spread, label='Spread')
        # plt.axhline(mean, color='gray', linestyle='--', label='Mean')
        # plt.axhline(mean + entry_z * std, color='red', linestyle='--', label=f'+{entry_z}σ')
        # plt.axhline(mean - entry_z * std, color='green', linestyle='--', label=f'-{entry_z}σ')
        # plt.axhline(mean + exit_z * std, color='orange', linestyle=':', label=f'+{exit_z}σ exit')
        # plt.axhline(mean - exit_z * std, color='orange', linestyle=':', label=f'-{exit_z}σ exit')

        for dt, val, signal in entries:
            plt.plot(dt, val, 'go' if signal == 'long' else 'ro', label=f'Entry ({signal})')
        for dt, val in exits:
            plt.plot(dt, val, 'kx', label='Exit')

        plt.title(f'Trading Signals for Pair: {pair_name}')
        plt.xlabel('Date')
        plt.ylabel('Spread')
        # plt.legend()
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
        self.backtest_with_signals()
        self.plot_performance()
        sharpe, mdd = self.evaluate_performance()
        print(f"Sharpe Ratio: {sharpe:.4f}")
        print(f"Max Drawdown: {mdd:.2%}")
        print(f"Portfolio Value: ${list(self.portfolio_value)[-1]:.2f}")
        return {
            'portfolio_value': list(self.portfolio_value)[-1],
            'sharpe_ratio': sharpe,
            'max_drawdown': mdd
        }


#### --------------------------------------------------------------------------------------------------------- ######
#### --------------------------------------------------------------------------------------------------------- ######
#### --------------------------------------------------------------------------------------------------------- ######


class SignalPortfolioConstrained:
    def __init__(self, initial_amount, pairs, data_universe, market_data):
        self.initial_amount = initial_amount
        self.pairs = pairs
        self.data_universe = data_universe
        self.market_data = market_data
        self.returns = {}
        self.weights = {}
        self.adjusted_weights = {}
        self.portfolio_returns = None
        self.portfolio_cumulative_returns = None
        self.portfolio_value = None
        self.var_series = None
        self.es_series = None

    def compute_daily_returns(self):
        for pair, data in self.pairs.items():
            self.returns[pair] = data['spread_series'].pct_change().dropna()

    def compute_inverse_volatility_weights(self):
        volatilities = {pair: np.std(ret) for pair, ret in self.returns.items()}
        inv_vol = {pair: 1 / vol for pair, vol in volatilities.items() if vol != 0}
        total_inv_vol = sum(inv_vol.values())
        self.weights = {pair: inv_vol[pair] / total_inv_vol for pair in inv_vol}

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

    def apply_constraints(self, date):
        constrained_weights = {}
        gmv = self.initial_amount

        for pair, weight in self.adjusted_weights.items():
            data = self.pairs[pair]
            x = data['symbol_x']
            y = data['symbol_y']

            # adv_x = self.adv_data.get(x, 1e9)
            # adv_y = self.adv_data.get(y, 1e9)
            
            ### Average daily Volume ?? 
            adv_x = np.mean(list(self.data_universe[x]['ACVOL_UNS']))
            adv_y = np.mean(list(self.data_universe[y]['ACVOL_UNS']))
            
            max_trade_limit = 0.025 * min(adv_x, adv_y)
            pos_limit = 0.025 * min(adv_x, adv_y)

            position_value = abs(weight * gmv)
            if position_value > pos_limit:
                weight = np.sign(weight) * pos_limit / gmv

            constrained_weights[pair] = weight

        total = sum(abs(w) for w in constrained_weights.values())
        return {pair: w / total for pair, w in constrained_weights.items()}

    def backtest_with_signals(self, entry_z=1.0, exit_z=0.2):
        dates = self.returns[list(self.returns.keys())[0]].index
        self.portfolio_returns = pd.Series(0, index=dates)
        position_tracker = {pair: 0 for pair in self.pairs}
        z_scores = {}

        for pair, data in self.pairs.items():
            spread = data['spread_series']
            z = (spread - data['spread_mean']) / data['spread_std']
            z_scores[pair] = z.reindex(dates).fillna(0)

        for t in range(1, len(dates)):
            date = dates[t]
            daily_return = 0
            weights_today = self.apply_constraints(date)

            for pair in self.pairs:
                z = z_scores[pair].iloc[t]
                ret = self.returns[pair].iloc[t]
                pos = position_tracker[pair]
                weight = weights_today[pair]

                if pos == 0:
                    if z > entry_z:
                        position_tracker[pair] = -1
                    elif z < -entry_z:
                        position_tracker[pair] = 1
                elif pos == 1 and z > -exit_z:
                    position_tracker[pair] = 0
                elif pos == -1 and z < exit_z:
                    position_tracker[pair] = 0

                holding_days = 1 / 252
                execution_cost = 0.0002 * abs(pos * weight)
                financing_cost = 0.005 * holding_days * abs(pos * weight)

                daily_return += pos * weight * ret - execution_cost - financing_cost

            self.portfolio_returns.iloc[t] = daily_return

        self.portfolio_cumulative_returns = (1 + self.portfolio_returns).cumprod()
        self.portfolio_value = self.portfolio_cumulative_returns * self.initial_amount

    def evaluate_performance(self):
        sharpe = self.portfolio_returns.mean() / self.portfolio_returns.std() * np.sqrt(252)
        drawdown = (self.portfolio_cumulative_returns / self.portfolio_cumulative_returns.cummax() - 1).min()
        return sharpe, drawdown

    def compute_var_es(self, alpha=0.05):
        if self.portfolio_returns is None:
            raise ValueError("Run the backtest first to generate portfolio returns.")
        
        returns = self.portfolio_returns.dropna()
        var_list = []
        es_list = []

        for i in range(len(returns)):
            window = returns[max(0, i - 59): i + 1]
            if len(window) < 10:
                var_list.append(0)
                es_list.append(0)
                continue

            sorted_returns = window.sort_values()
            var = sorted_returns.quantile(alpha)
            es = sorted_returns[sorted_returns <= var].mean()

            var_list.append(var)
            es_list.append(es)

        self.var_series = pd.Series(var_list, index=returns.index)
        self.es_series = pd.Series(es_list, index=returns.index)

    def plot_var_es_curve(self, alpha=0.05, risk_limit=500_000):
        if self.var_series is None or self.es_series is None:
            self.compute_var_es(alpha)

        var_usd = self.var_series * self.initial_amount
        es_usd = self.es_series * self.initial_amount

        plt.figure(figsize=(8, 4))
        sns.lineplot(data=var_usd, label=f'VaR ({int((1 - alpha) * 100)}%)')
        sns.lineplot(data=es_usd, label=f'Expected Shortfall ({int((1 - alpha) * 100)}%)')
        plt.axhline(-risk_limit, color='red', linestyle='--', label=f'Risk Limit (${risk_limit:,.0f})')
        plt.title('Value at Risk (VaR) and Expected Shortfall (ES) Over Time')
        plt.ylabel('Potential Loss in USD')
        plt.xlabel('Date')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_performance(self):
        plt.figure(figsize=(8, 4))
        plt.plot(self.portfolio_value, label="Portfolio")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.title("Beta-Neutral Signal-Based Portfolio")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_pair_signals(self, pair_name, entry_z=1.0, exit_z=0.2):
        pair_data = self.pairs[pair_name]
        spread = pair_data['spread_series']
        mean = pair_data['spread_mean']
        std = pair_data['spread_std']

        z_score = (spread - mean) / std
        position = 0
        entries, exits = [], []

        for t in range(1, len(z_score)):
            z = z_score.iloc[t]

            if position == 0:
                if z > entry_z:
                    entries.append((z_score.index[t], spread.iloc[t], 'short'))
                    position = -1
                elif z < -entry_z:
                    entries.append((z_score.index[t], spread.iloc[t], 'long'))
                    position = 1
            elif position == 1 and z > -exit_z:
                exits.append((z_score.index[t], spread.iloc[t]))
                position = 0
            elif position == -1 and z < exit_z:
                exits.append((z_score.index[t], spread.iloc[t]))
                position = 0

        plt.figure(figsize=(8, 4))
        plt.plot(spread, label='Spread')
        # plt.axhline(mean, color='gray', linestyle='--', label='Mean')
        # plt.axhline(mean + entry_z * std, color='red', linestyle='--', label=f'+{entry_z}σ')
        # plt.axhline(mean - entry_z * std, color='green', linestyle='--', label=f'-{entry_z}σ')
        # plt.axhline(mean + exit_z * std, color='orange', linestyle=':', label=f'+{exit_z}σ exit')
        # plt.axhline(mean - exit_z * std, color='orange', linestyle=':', label=f'-{exit_z}σ exit')

        for dt, val, signal in entries:
            plt.plot(dt, val, 'go' if signal == 'long' else 'ro', label=f'Entry ({signal})')
        for dt, val in exits:
            plt.plot(dt, val, 'kx', label='Exit')

        plt.title(f'Trading Signals for Pair: {pair_name}')
        plt.xlabel('Date')
        plt.ylabel('Spread')
        # plt.legend()
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
        self.backtest_with_signals()
        self.plot_performance()
        sharpe, mdd = self.evaluate_performance()
        print(f"Sharpe Ratio: {sharpe:.4f}")
        print(f"Max Drawdown: {mdd:.2%}")
        print(f"Portfolio Value: ${list(self.portfolio_value)[-1]:.2f}")
        return {
            'portfolio_value': list(self.portfolio_value)[-1],
            'sharpe_ratio': sharpe,
            'max_drawdown': mdd
        }
    

if __name__ == '__main__':
    print('running __portfolio.py__')