import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
from collections import defaultdict


'''
Portfolio: Constructs a 'Core-Satellite' based portfolio combining alpha + beta to maximize returns
'''


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

    # def adjust_weights_for_beta_neutrality(self):
    #     beta_x = {}
    #     beta_y = {}

    #     for pair, data in self.pairs.items():
    #         px = self.data_universe[data['symbol_x']]['TRDPRC_1']
    #         py = self.data_universe[data['symbol_y']]['TRDPRC_1']
    #         beta_x[pair] = self.compute_beta(px)
    #         beta_y[pair] = self.compute_beta(py)

    #     for pair, weight in self.weights.items():
    #         hedge_ratio = self.pairs[pair]['hedge_ratio']
    #         # Effective beta of the spread = beta_x - hedge * beta_y
    #         beta_adj = beta_x[pair] - hedge_ratio * beta_y[pair]
    #         self.adjusted_weights[pair] = weight * beta_adj

    #     total_weight = sum(abs(w) for w in self.adjusted_weights.values())
    #     self.adjusted_weights = {pair: w / total_weight for pair, w in self.adjusted_weights.items()}

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
            px = self.data_universe.loc[data['symbol_x']]['close']
            py = self.data_universe.loc[data['symbol_y']]['close']
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
        plt.title("Beta-Neutral Signal-Based Portfolio (No Fees, No Trade-Limit)")
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
    def __init__(self, initial_amount, pairs, data_universe, market_data, past_window = 30, entry_z = 1.0, exit_z = 0.2):
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
        self.z_scores = {}
        self.past_window = past_window
        self.entry_z = entry_z
        self.exit_z = exit_z

    def compute_daily_returns(self):
        for pair, data in self.pairs.items():
            self.returns[pair] = data['spread_series'].pct_change().dropna()

    def compute_rolling_z_scores(self):
        self.z_scores = {}
        for pair, data in self.pairs.items():
            spread = data['spread_series']
            rolling_mean = spread.rolling(window=self.past_window).mean()
            rolling_std = spread.rolling(window=self.past_window).std()
            z = (spread - rolling_mean) / rolling_std
            self.z_scores[pair] = z

    def compute_inverse_volatility_weights(self, t):
        volatilities = {pair: np.std(ret[t - self.past_window: t]) for pair, ret in self.returns.items()}
        inv_vol = {pair: 1 / vol for pair, vol in volatilities.items() if vol != 0}
        total_inv_vol = sum(inv_vol.values())
        return {pair: inv_vol[pair] / total_inv_vol for pair in inv_vol}

    def compute_beta(self, price_series):
        asset_returns = price_series.pct_change().dropna()
        market_returns = self.market_data.pct_change().dropna()
        min_len = min(len(asset_returns), len(market_returns))
        X = market_returns[-min_len:].values.reshape(-1, 1)
        y = asset_returns[-min_len:].values
        model = LinearRegression().fit(X, y)
        return model.coef_[0]

    def adjust_weights_for_beta_neutrality(self, weights):
        target_weights = np.array([weights[pair] for pair in self.pairs])
        pair_list = list(self.pairs.keys())

        effective_betas = []
        adv_limits = []
        for pair in pair_list:
            data = self.pairs[pair]
            px = self.data_universe.loc[data['symbol_x']]['close']
            py = self.data_universe.loc[data['symbol_y']]['close']
            beta_x = self.compute_beta(px)
            beta_y = self.compute_beta(py)
            beta_spread = beta_x - data['hedge_ratio'] * beta_y
            effective_betas.append(beta_spread)

            x, y = data['symbol_x'], data['symbol_y']
            adv_x = np.mean(list(self.data_universe.loc[x]['volume']))
            adv_y = np.mean(list(self.data_universe.loc[y]['volume']))
            max_trade_val = 0.025 * min(adv_x, adv_y)
            adv_limits.append(max_trade_val / self.initial_amount)

        effective_betas = np.array(effective_betas)
        adv_limits = np.array(adv_limits)

        def objective(w):
            return np.sum((w - target_weights) ** 2)

        def leverage_constraint(w):
            return 1.0 - np.sum(np.abs(w))

        def beta_constraint(w):
            return np.dot(w, effective_betas)

        constraints = [
            {'type': 'ineq', 'fun': leverage_constraint},
            {'type': 'eq',   'fun': beta_constraint}
        ]

        bounds = [(-lim, lim) for lim in adv_limits]
        x0 = target_weights.copy()

        result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
        if not result.success:
            print("Optimization failed:", result.message)

        optimized_weights = result.x
        return dict(zip(pair_list, optimized_weights))

    def backtest_with_signals(self):

        self.compute_rolling_z_scores()
        dates = self.returns[list(self.returns.keys())[0]].index
        self.portfolio_returns = pd.Series(0.0, index=dates)
        position_tracker = {pair: 0 for pair in self.pairs} ## -1: short, 0: No position, 1: Long
        current_weights = {pair: 0.0 for pair in self.pairs}
        z_scores = {pair: z.reindex(dates).fillna(0) for pair, z in self.z_scores.items()}

        for t in range(self.past_window, len(dates)):
            date = dates[t]
            daily_return = 0.0

            ### Daily Rebalancing
            weights_today = self.compute_inverse_volatility_weights(t)
            weights_today = self.adjust_weights_for_beta_neutrality(weights_today)

            for pair in self.pairs:
                z = z_scores[pair].iloc[t]
                ret = self.returns[pair].iloc[t]
                prev_pos = position_tracker[pair]
                target_pos = prev_pos
                weight = weights_today[pair]

                if prev_pos == 0:
                    if z < - self.entry_z: ## Long Spread
                        target_pos = 1
                    # elif z > self.entry_z: ## Short Spread
                    #     target_pos = -1 
                elif prev_pos == 1 and z > - self.exit_z:
                    target_pos = 0
                # elif prev_pos == -1 and z < self.exit_z:
                #     target_pos = 0

                execution_cost = 0.0002 * abs(weight * (target_pos - prev_pos)) if target_pos != prev_pos else 0.0
                financing_cost = 0.005 * (1 / 252) * abs(target_pos * weight)
                pnl = prev_pos * weight * ret
                daily_return += pnl - execution_cost - financing_cost

                position_tracker[pair] = target_pos
                current_weights[pair] = weight * target_pos

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
        plt.axhline(-risk_limit/252, color='red', linestyle='--', label=f'Risk Limit (daily) (${risk_limit/252:,.0f})')
        plt.title('Value at Risk (VaR) and Expected Shortfall (ES) Over Time')
        plt.ylabel('Potential Loss ($)')
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

    def plot_pair_signals(self, pair_name):

        pair_data = self.pairs[pair_name]
        spread = pair_data['spread_series']
        rolling_mean = spread.rolling(window=self.past_window).mean()
        rolling_std = spread.rolling(window=self.past_window).std()
        z_score = (spread - rolling_mean) / rolling_std

        position = 0
        entries, exits = [], []

        for t in range(self.past_window, len(z_score)):

            z = z_score.iloc[t]
            if position == 0:
                if z < -self.entry_z:
                    entries.append((z_score.index[t], spread.iloc[t], 'long'))
                    position = 1
                # elif z > self.entry_z:
                #     entries.append((z_score.index[t], spread.iloc[t], 'short'))
                #     position = -1
            elif position == 1 and z > - self.exit_z:
                exits.append((z_score.index[t], spread.iloc[t]))
                position = 0
            # elif position == -1 and z < self.exit_z:
            #     exits.append((z_score.index[t], spread.iloc[t]))
            #     position = 0

        plt.figure(figsize=(8, 4))
        plt.plot(spread, label='Spread')

        for dt, val, signal in entries:
            plt.plot(dt, val, 'go' if signal == 'long' else 'ro', label=f'Entry ({signal})')
        for dt, val in exits:
            plt.plot(dt, val, 'kx', label='Exit')

        plt.title(f'Trading Signals for Pair: {pair_name}')
        plt.xlabel('Date')
        plt.ylabel('Spread')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def beta_neutral_check(self):
        market_returns = self.market_data.pct_change().dropna()
        min_len = min(len(self.portfolio_returns), len(market_returns))
        X = market_returns[-min_len:].values.reshape(-1, 1)
        y = self.portfolio_returns[-min_len:].values
        model = LinearRegression().fit(X, y)
        print('Portfolio-Beta (Daily): ', model.coef_[0])
        print('Portfolio-Alpha (Daily) (%): ', model.intercept_ * 100, ' %')
        alpha_annualized = ((1 + model.intercept_) ** 252) - 1
        print('Portfolio-Alpha (Annual) (%): ', alpha_annualized * 100, ' %')

    def run(self):
        self.compute_daily_returns()
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
    
class PortfolioConstructor:
    """
    Constructs a complete portfolio by generating alpha signals, optimizing them
    for beta-neutrality and other constraints, and then blending them with a
    strategic beta overlay.
    """
    
    def __init__(self, pairs_data: dict, market_data: pd.DataFrame, config: dict):
        """
        Initializes the constructor with all necessary data and configuration.

        Args:
            pairs_data: Dict containing spread_series and hedge_ratios for each pair.
            market_data: DataFrame of asset prices and volumes.
            config: A dictionary containing all parameters.
        """
        
        # Data
        self.pairs_data = pairs_data
        self.market_data = market_data
        self.returns = self._compute_spread_returns()

        # Alpha Config
        self.past_window = config['alpha']['past_window']
        self.z_scores = self._compute_rolling_z_scores()

        # Optimization Config
        self.max_adv_fraction = config['optimizer']['max_adv_fraction']
        self.target_beta = config['optimizer']['target_beta']
        self.max_leverage_alpha = config['optimizer']['max_leverage_alpha']
        
        # Strategic Allocation Config
        self.alpha_alloc = config['strategy']['alpha_allocation']
        self.beta_alloc = config['strategy']['beta_allocation']
        self.beta_basket = config['strategy']['beta_basket']
        assert 0.0 <= self.alpha_alloc + self.beta_alloc, "Allocations must be valid."

    # --- Step 1: Internal Helper Methods for Data Preparation ---
    def _compute_spread_returns(self):
        returns = {}
        for pair, data in self.pairs_data.items():
            returns[pair] = data['spread_series'].pct_change()
        return returns

    def _compute_rolling_z_scores(self):
        z_scores = {}
        for pair, data in self.pairs_data.items():
            spread = data['spread_series']
            mean = spread.rolling(window=self.past_window).mean()
            std = spread.rolling(window=self.past_window).std()
            z_scores[pair] = (spread - mean) / std
        return z_scores

    # --- Step 2: Internal Helper Methods for Optimization ---
    def _compute_inverse_volatility_weights(self, active_pairs: list, timestamp: pd.Timestamp):
        """Calculates initial weights for active pairs based on inverse volatility."""
        if not active_pairs:
            return {}
        
        volatilities = {}
        for pair in active_pairs:
            # Get returns in the lookback window up to the current timestamp
            window_returns = self.returns[pair].loc[:timestamp].tail(self.past_window)
            vol = window_returns.std()
            volatilities[pair] = vol

        inv_vol = {pair: 1 / vol for pair, vol in volatilities.items() if vol > 0}
        total_inv_vol = sum(inv_vol.values())
        
        return {pair: inv_vol[pair] / total_inv_vol for pair in inv_vol} if total_inv_vol > 0 else {}

    def _compute_asset_beta(self, asset_symbol: str, timestamp: pd.Timestamp):
        """Computes beta for a single asset against the market proxy."""
        # For simplicity, using a fixed market proxy. A more advanced version could use a dynamic index.
        market_returns = self.market_data[('BTCUSDT', 'close')].pct_change()
        asset_returns = self.market_data[(asset_symbol, 'close')].pct_change()
        
        # Align data up to the current timestamp
        combined = pd.concat([asset_returns, market_returns], axis=1).dropna()
        combined = combined.loc[:timestamp].tail(self.past_window) # Use rolling window for beta
        
        if len(combined) < 20: return 1.0 # Default to beta of 1 if not enough data
        
        X = combined.iloc[:, 1].values.reshape(-1, 1)
        y = combined.iloc[:, 0].values
        model = LinearRegression().fit(X, y)
        return model.coef_[0]

    def _optimize_alpha_weights(self, initial_weights: dict, timestamp: pd.Timestamp):
        """
        Runs the SLSQP optimizer to find beta-neutral weights that respect constraints.
        This is the core of your 'adjust_weights_for_beta_neutrality' logic.
        """
        if not initial_weights:
            return {}

        active_pairs = list(initial_weights.keys())
        target_weights = np.array([initial_weights[pair] for pair in active_pairs])

        # --- Gather Constraints ---
        effective_betas = []
        adv_limits = []
        for pair in active_pairs:
            pair_info = self.pairs_data[pair]
            asset_x, asset_y = pair_info['symbol_x'], pair_info['symbol_y']
            
            # Beta
            beta_x = self._compute_asset_beta(asset_x, timestamp)
            beta_y = self._compute_asset_beta(asset_y, timestamp)
            beta_spread = beta_x - pair_info['hedge_ratio'] * beta_y
            effective_betas.append(beta_spread)

            # ADV
            adv_x = self.market_data[(asset_x, 'volume')].loc[:timestamp].tail(30).mean()
            adv_y = self.market_data[(asset_y, 'volume')].loc[:timestamp].tail(30).mean()
            # Note: initial_amount should be passed in or stored in config
            max_trade_val = self.max_adv_fraction * min(adv_x, adv_y)
            adv_limits.append(max_trade_val / 100000) # Placeholder equity

        # --- Define Optimization Problem ---
        def objective(w):
            return np.sum((w - target_weights)**2)

        def leverage_constraint(w):
            return self.max_leverage_alpha - np.sum(np.abs(w))

        def beta_constraint(w):
            return self.target_beta - np.dot(w, effective_betas)

        constraints = [{'type': 'ineq', 'fun': leverage_constraint},
                       {'type': 'eq', 'fun': beta_constraint}]
        bounds = [(-lim, lim) for lim in adv_limits]
        
        result = minimize(objective, target_weights, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if not result.success:
            # Fallback strategy: return an empty dict if optimization fails
            print(f"Optimization failed at {timestamp}: {result.message}")
            return {}

        return dict(zip(active_pairs, result.x))

    # --- Step 3: The Main Orchestrator Method ---
    def construct_target_weights(self, timestamp: pd.Timestamp, alpha_signals: dict, beta_signal: float):
        """
        The main public method that orchestrates the entire portfolio construction process.
        """
        final_weights = defaultdict(float)

        # === ALPHA PORTFOLIO CONSTRUCTION ===
        active_alpha_pairs = [pair for pair, signal in alpha_signals.items() if signal != 0]
        
        initial_alpha_weights = self._compute_inverse_volatility_weights(active_alpha_pairs, timestamp)        
        optimized_pair_weights = self._optimize_alpha_weights(initial_alpha_weights, timestamp)
        
        # 3. Scale and decompose the final alpha portfolio
        for pair, pair_weight in optimized_pair_weights.items():
            pair_signal = alpha_signals.get(pair, 0)
            final_pair_weight = self.alpha_alloc * pair_weight * pair_signal
           
            # Decompose into asset legs (simplified equal-dollar split)
            asset1, asset2 = self.pairs_data.loc[pair]['symbol_x'], self.pairs_data.loc[pair]['symbol_y']
            final_weights[asset1] += final_pair_weight / 2
            final_weights[asset2] -= final_pair_weight / 2

        # === BETA PORTFOLIO CONSTRUCTION ===
        if beta_signal > 0:
            total_beta_weight = self.beta_alloc * beta_signal
            beta_weight_per_asset = total_beta_weight / len(self.beta_basket)
            for asset in self.beta_basket:
                final_weights[asset] += beta_weight_per_asset
                
        ## Let's NOT trade spreads
        ## Basically, a dict of different assests (not pair/spread) with adjusted portfolio-weights
        return dict(final_weights)


if __name__ == '__main__':
    print('running __portfolio.py__')