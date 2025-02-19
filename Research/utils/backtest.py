import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# class Backtest():
#     def __init__(self):
#         pass

#     def tester(self):
#         pass

class Backtest:
    def __init__(self, data, strategy, initial_capital=1000, commission=0.001, name='Default'):
        """
        <Backtest> Class does the position entry and exit based on binary signals --> know the pct_change of every stick and count the '1' sticks --> find the cumulative return in % 
        """
        self.data = data.copy()
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = None
        self.name = name

    def run(self):
        """Runs the backtest and computes performance metrics."""
        self.data["Signal"] = self.strategy(self.data)
        self.data["Returns"] = self.data['TRDPRC_1'].pct_change() * self.data["Signal"].shift(1)

        # Calculate equity curve | no Concurrent trade + going with 100% capital all the time
        self.data["Equity"] = self.initial_capital * (1 + self.data["Returns"]).cumprod()

        # Apply trading cost
        self.data["Trades"] = self.data["Signal"].diff().abs()  # Count changes in position
        self.data["Equity"] -= self.data["Trades"] * self.commission * self.initial_capital

        self.results = self.data
        return self.results

    def performance(self):
        """Calculates and returns performance metrics."""
        if self.results is None:
            raise ValueError("Run the backtest first using `.run()`.")

        total_return = self.results["Equity"].iloc[-1] / self.initial_capital - 1
        sharpe_ratio = self.results["Returns"].mean() / self.results["Returns"].std() * np.sqrt(252)
        max_drawdown = (self.results["Equity"] / self.results["Equity"].cummax() - 1).min()

        return {
            "Total Return": f"{total_return:.2%}",
            "Sharpe Ratio": f"{sharpe_ratio:.2f}",
            "Max Drawdown": f"{max_drawdown:.2%}"
        }

    def plot(self):
        """Plots the equity curve."""
        if self.results is None:
            raise ValueError("Run the backtest first using `.run()`.")

        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        
        axs[0].plot(self.results.index, self.results["Equity"], label="Equity Curve", color='black')
        axs[1].hist(self.results["Returns"], bins=70)
        
        axs[0].set_xlabel("Date")
        axs[0].set_ylabel("Equity")
        axs[0].set_title(f"Backtest-Equity-Curve | {self.name}")
        
        axs[1].set_title(f"Return-Volatility | {self.name} | std. dev: {self.results["Returns"].std():.2f} ")
        
        plt.legend()
        plt.show()

# Example: Simple Moving Average Crossover Strategy
def simple_sma_strategy(data, short_window=25, long_window=60):
    
    data["SMA_Short"] = data['TRDPRC_1'].rolling(short_window).mean()
    data["SMA_Long"] = data['TRDPRC_1'].rolling(long_window).mean()

    position_binary = np.where(data["SMA_Short"] > data["SMA_Long"], 1, 0)
    
    return position_binary

# Example usage
if __name__ == "__main__":
    
    print('Running __backtest.py__ main now ...')    
    
    df = pd.read_csv("your_data.csv")  # Ensure it has 'Date' and 'Close' columns
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Run backtest
    backtest = Backtest(df, simple_sma_strategy)
    backtest.run()
    
    # Print performance
    print(backtest.performance())
    
    # Plot equity curve
    backtest.plot()