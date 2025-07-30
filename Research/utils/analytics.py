# src/analytics.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class PortfolioAnalytics:
    """
    Calculates and visualizes portfolio performance metrics.
    """
    def __init__(self, equity_curve: pd.Series):
        self.equity_curve = equity_curve
        self.returns = self.equity_curve.pct_change().dropna()

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """Calculates the annualized Sharpe ratio."""
        # Assuming hourly returns, 252 * 24 trading hours in a year
        trading_hours_per_year = 252 * 24
        excess_returns = self.returns - risk_free_rate / trading_hours_per_year
        return np.sqrt(trading_hours_per_year) * excess_returns.mean() / excess_returns.std()

    def calculate_max_drawdown(self) -> float:
        """Calculates the maximum drawdown."""
        running_max = self.equity_curve.cummax()
        drawdown = (self.equity_curve - running_max) / running_max
        return drawdown.min()

    def plot_equity_curve(self):
        """Plots the portfolio value over time."""
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=self.equity_curve)
        plt.title('Portfolio Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True)
        plt.show()

    def display_summary(self):
        """Prints a summary of key performance metrics."""
        sharpe = self.calculate_sharpe_ratio()
        mdd = self.calculate_max_drawdown()
        
        print("--- Performance Summary ---")
        print(f"Final Portfolio Value: ${self.equity_curve.iloc[-1]:,.2f}")
        print(f"Annualized Sharpe Ratio: {sharpe:.2f}")
        print(f"Maximum Drawdown: {mdd:.2%}")
        print("---------------------------")
        self.plot_equity_curve()