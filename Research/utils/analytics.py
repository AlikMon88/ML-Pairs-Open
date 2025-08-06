# src/analytics.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

### NEED: VaR and CVaR
class PortfolioAnalytics:
    """
    Calculates and visualizes portfolio performance metrics based on an equity curve.
    This class is strategy-agnostic.
    """
    def __init__(self, equity_curve, market_data, periods_per_year):
        """
        Args:
            equity_curve: A pandas Series representing the portfolio value over time.
            periods_per_year: The number of trading periods in a year for annualization.
                              (e.g., 252 for daily, 252*24 for hourly, 12 for monthly).
        """
        if equity_curve.empty or len(equity_curve) < 2:
            raise ValueError("Equity curve must not be empty and must have at least two data points.")
            
        self.equity_curve = equity_curve.dropna()
        self.market_data = market_data.dropna()
        
        self.returns = self.equity_curve.pct_change().dropna()
        self.market_returns = self.market_data.pct_change().dropna()
       
        self.periods_per_year = periods_per_year

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0):
        """Calculates the annualized Sharpe ratio."""
        if self.returns.std() == 0:
            return 0.0
        
        excess_returns = self.returns - (risk_free_rate / self.periods_per_year)
        return np.sqrt(self.periods_per_year) * excess_returns.mean() / excess_returns.std()

    def calculate_sortino_ratio(self, risk_free_rate: float = 0.0):
        """Calculates the annualized Sortino ratio."""
        excess_returns = self.returns - (risk_free_rate / self.periods_per_year)
        downside_std = excess_returns[excess_returns < 0].std()
        
        if downside_std == 0:
            return np.inf
        
        return np.sqrt(self.periods_per_year) * excess_returns.mean() / downside_std

    def calculate_max_drawdown(self):
        """Calculates the maximum drawdown."""
        running_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - running_max) / running_max
        return drawdown.min()
    
    def calculate_cagr(self):
        """Calculates the Compound Annual Growth Rate."""
        start_value = self.equity_curve.iloc[0]
        end_value = self.equity_curve.iloc[-1]
        num_years = len(self.equity_curve) / self.periods_per_year
        return (end_value / start_value) ** (1 / num_years) - 1

    def calculate_beta_and_alpha(self):
        """Calculates the portfolio's beta and annualized alpha against a market benchmark."""
        # Align the portfolio returns with the market returns
        df = pd.concat([self.returns, self.market_returns], axis=1).dropna()
        
        X = df.iloc[:, 1].values.reshape(-1, 1) # Market returns
        y = df.iloc[:, 0].values                # Portfolio returns
        
        model = LinearRegression().fit(X, y)
        beta = model.coef_[0]
        
        # Alpha is the intercept, annualized
        daily_alpha = model.intercept_
        annualized_alpha = (1 + daily_alpha)**self.periods_per_year - 1
        
        return beta, annualized_alpha

    def display_summary(self, is_ensmb=False):
        """Prints a comprehensive summary of key performance metrics."""
        sharpe = self.calculate_sharpe_ratio()
        sortino = self.calculate_sortino_ratio()
        mdd = self.calculate_max_drawdown()
        cagr = self.calculate_cagr()

        if not is_ensmb:
        
            print("--- Performance Summary ---")
            # print(f"Start Date: {self.equity_curve.index[0].strftime('%Y-%m-%d')}")
            # print(f"End Date: {self.equity_curve.index[-1].strftime('%Y-%m-%d')}")
            print(f"Final Portfolio Value: ${self.equity_curve.iloc[-1]:,.2f}")
            print("-" * 27)
            print(f"Compound Annual Growth Rate (CAGR): {cagr:.2%}")
            print(f"Annualized Sharpe Ratio: {sharpe:.2f}")
            print(f"Annualized Sortino Ratio: {sortino:.2f}")
            print(f"Maximum Drawdown: {mdd:.2%}")

            if self.market_returns is not None:
                beta, alpha = self.calculate_beta_and_alpha()
                print("-" * 27)
                print(f"Market Beta: {beta:.2f}")
                print(f"Annualized Alpha: {alpha:.2%}")

            print("---------------------------")
            self.plot_equity_curve()
        
        else:
            _ensmb_dict = {
                'port_value': self.equity_curve.iloc[-1],
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': mdd,
                'annualizd_compound_return': cagr
            }
            
            if self.market_returns is not None:
                beta, alpha = self.calculate_beta_and_alpha()
                _ensmb_dict['beta'] = beta
                _ensmb_dict['ann_alpha'] = alpha

            return _ensmb_dict
            

    def plot_equity_curve(self):
        """Plots the portfolio value over time."""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 4))
        
        self.equity_curve.plot(ax=ax, label='Portfolio Equity')
        
        ax.set_title('Portfolio Performance', fontsize=16)
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_rolling_sharpe(self, window: int = None):
        """Plots the rolling annualized Sharpe ratio."""
        if window is None:
            window = self.periods_per_year # Default to a 1-year rolling window
            
        rolling_sharpe = self.returns.rolling(window=window).apply(
            lambda x: np.sqrt(self.periods_per_year) * x.mean() / x.std() if x.std() != 0 else 0,
            raw=True
        )
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 4))
        
        rolling_sharpe.plot(ax=ax, label=f'{window}-Period Rolling Sharpe')
        ax.axhline(0, color='grey', linestyle='--')
        ax.set_title('Rolling Sharpe Ratio', fontsize=16)
        ax.set_xlabel('Date')
        ax.set_ylabel('Annualized Sharpe Ratio')
        ax.legend()
        plt.tight_layout()
        plt.show()
        
        
if __name__ == '__main__':
    print('running __analytics.py__')