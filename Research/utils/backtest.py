import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm



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

### ------------------------ Pairs-Trading-Bactester --------------------------------

class PairsBacktester:
    def __init__(self, history_provider):
        """
        Initialize the backtest framework
        
        Args:
            history_provider: Object to get historical price data
        """
        self.history_provider = history_provider
        self.pairs = []
        self.results = {}
        self.portfolio_equity = []
        self.trades = []
        
    def add_pair(self, symbol1, symbol2, lookback_period=60, z_score_threshold=2.0, 
                 formation_period=252, hedge_ratio_method='rolling'):
        """
        Add a pair to the backtest
        
        Args:
            symbol1: First asset in the pair
            symbol2: Second asset in the pair
            lookback_period: Period for calculating the z-score
            z_score_threshold: Threshold for entry and exit
            formation_period: Period for calculating the hedge ratio
            hedge_ratio_method: Method for calculating hedge ratio ('rolling' or 'fixed')
        """
        self.pairs.append({
            'symbol1': symbol1,
            'symbol2': symbol2,
            'lookback_period': lookback_period,
            'z_score_threshold': z_score_threshold,
            'formation_period': formation_period,
            'hedge_ratio_method': hedge_ratio_method
        })
        
    # def calculate_hedge_ratio(self, price_series1, price_series2, method='rolling', window=None):
    #     """
    #     Calculate the hedge ratio between two price series
        
    #     Args:
    #         price_series1: Price series of first asset
    #         price_series2: Price series of second asset
    #         method: 'rolling' or 'fixed'
    #         window: Window size for rolling regression
            
    #     Returns:
    #         hedge_ratios: Series or single value representing hedge ratio(s)
    #     """
    #     if method == 'fixed':
    #         # Use the entire series for a fixed ratio
    #         model = OLS(price_series1, sm.add_constant(price_series2))
    #         results = model.fit()
    #         return results.params[1]  # Return the slope coefficient
    #     elif method == 'rolling':
    #         # Rolling window regression
    #         if window is None:
    #             window = len(price_series1) // 4  # Default to 1/4 of the data
                
    #         hedge_ratios = np.zeros(len(price_series1))
            
    #         for i in range(window, len(price_series1)):
    #             y = price_series1.iloc[i-window:i]
    #             X = price_series2.iloc[i-window:i]
    #             X = sm.add_constant(X)
                
    #             model = OLS(y, X)
    #             results = model.fit()
    #             hedge_ratios[i] = results.params[1]
                
    #         return pd.Series(hedge_ratios, index=price_series1.index)
    #     else:
    #         raise ValueError(f"Unknown hedge ratio method: {method}")
    
    # def calculate_spread(self, price_series1, price_series2, hedge_ratio):
    #     """
    #     Calculate the spread between two price series
        
    #     Args:
    #         price_series1: Price series of first asset
    #         price_series2: Price series of second asset
    #         hedge_ratio: Hedge ratio (fixed or series)
            
    #     Returns:
    #         spread: The calculated spread
    #     """
    #     if isinstance(hedge_ratio, pd.Series):
    #         # If hedge_ratio is a Series, we need to align and multiply
    #         spread = price_series1 - hedge_ratio * price_series2
    #         return spread
    #     else:
    #         # If hedge_ratio is a single value
    #         return price_series1 - hedge_ratio * price_series2
    
    # def calculate_zscore(self, spread, lookback_period):
    #     """
    #     Calculate the z-score of a spread
        
    #     Args:
    #         spread: Spread between two assets
    #         lookback_period: Period for calculating mean and std
            
    #     Returns:
    #         z_score: Z-score series
    #     """
    #     rolling_mean = spread.rolling(window=lookback_period).mean()
    #     rolling_std = spread.rolling(window=lookback_period).std()
    #     z_score = (spread - rolling_mean) / rolling_std
    #     return z_score
    
    def run_backtest(self, start_date, end_date, initial_capital=100000, 
                     transaction_cost_pct=0.001, position_size_pct=0.1):
        """
        Run the backtest for all pairs
        
        Args:
            start_date: Start date for the backtest
            end_date: End date for the backtest
            initial_capital: Initial capital
            transaction_cost_pct: Transaction cost as percentage
            position_size_pct: Position size as percentage of portfolio
            
        Returns:
            results: Dictionary of backtest results
        """
        # Initialize portfolio and positions
        portfolio_value = initial_capital
        cash = initial_capital
        positions = {}
        daily_equity = []
        all_pair_results = {}
        
        # Get the longest formation period from all pairs
        max_formation_period = max([p['formation_period'] for p in self.pairs])
        
        # Adjust start date to account for formation period
        adjusted_start_date = start_date - timedelta(days=max_formation_period * 1.5)
        
        # Create date range for the backtest
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')
        
        # Store all price data to avoid repeated API calls
        price_data = {}
        
        # Get historical data for all symbols
        for pair in self.pairs:
            for symbol in [pair['symbol1'], pair['symbol2']]:
                if symbol not in price_data:
                    history = self.history_provider.get_price_history(
                        symbol, adjusted_start_date, end_date, resolution='daily'
                    )
                    
                    if history is None or len(history) < max_formation_period:
                        raise ValueError(f"Insufficient price history for {symbol}")
                    
                    price_data[symbol] = history['close']
        
        # Ensure all symbols have data for the same dates
        all_symbols = list(price_data.keys())
        common_index = price_data[all_symbols[0]].index
        for symbol in all_symbols[1:]:
            common_index = common_index.intersection(price_data[symbol].index)
        
        # Reindex all price data to common index
        for symbol in all_symbols:
            price_data[symbol] = price_data[symbol].reindex(common_index)
        
        # Process each pair
        for pair_idx, pair in enumerate(self.pairs):
            symbol1 = pair['symbol1']
            symbol2 = pair['symbol2']
            lookback_period = pair['lookback_period']
            z_score_threshold = pair['z_score_threshold']
            formation_period = pair['formation_period']
            hedge_ratio_method = pair['hedge_ratio_method']
            
            # Get price data for the pair
            price1 = price_data[symbol1]
            price2 = price_data[symbol2]
            
            # Calculate hedge ratio
            hedge_ratio = self.calculate_hedge_ratio(
                price1[:formation_period], 
                price2[:formation_period], 
                method=hedge_ratio_method,
                window=formation_period
            )
            
            # Calculate spread and z-score
            spread = self.calculate_spread(price1, price2, hedge_ratio)
            z_score = self.calculate_zscore(spread, lookback_period)
            
            # Initialize pair state
            pair_positions = {symbol1: 0, symbol2: 0}
            pair_state = "flat"  # flat, long_spread, short_spread
            pair_trades = []
            pair_equity = []
            entry_price1 = 0
            entry_price2 = 0
            entry_date = None
            
            # Only start trading after formation period and lookback period
            trading_start_idx = max(formation_period, lookback_period)
            
            # Loop through each day in the trading period
            for i in range(trading_start_idx, len(price1)):
                current_date = price1.index[i]
                if current_date < start_date:
                    continue
                    
                current_z = z_score[i]
                current_price1 = price1[i]
                current_price2 = price2[i]
                
                # Update hedge ratio if using rolling method
                if hedge_ratio_method == 'rolling' and isinstance(hedge_ratio, pd.Series):
                    current_hedge_ratio = hedge_ratio[i]
                else:
                    current_hedge_ratio = hedge_ratio
                
                # Trading logic
                # 1. If we're flat and z-score exceeds threshold, enter position
                if pair_state == "flat":
                    if current_z > z_score_threshold:  # Spread is too wide, go short the spread
                        # Short symbol1, long symbol2
                        position_value = portfolio_value * position_size_pct
                        
                        # Calculate position sizes to maintain hedge ratio
                        pair_positions[symbol1] = -int(position_value / (2 * current_price1))
                        pair_positions[symbol2] = int((position_value / 2) / current_price2 * current_hedge_ratio)
                        
                        # Record entry
                        entry_price1 = current_price1
                        entry_price2 = current_price2
                        entry_date = current_date
                        pair_state = "short_spread"
                        
                        # Apply transaction costs
                        transaction_cost = abs(pair_positions[symbol1]) * current_price1 * transaction_cost_pct
                        transaction_cost += abs(pair_positions[symbol2]) * current_price2 * transaction_cost_pct
                        cash -= transaction_cost
                        
                        # Record trade
                        pair_trades.append({
                            'entry_date': entry_date,
                            'entry_z': current_z,
                            'entry_price1': entry_price1,
                            'entry_price2': entry_price2,
                            'position1': pair_positions[symbol1],
                            'position2': pair_positions[symbol2],
                            'type': 'short_spread',
                            'transaction_cost': transaction_cost
                        })
                        
                    elif current_z < -z_score_threshold:  # Spread is too narrow, go long the spread
                        # Long symbol1, short symbol2
                        position_value = portfolio_value * position_size_pct
                        
                        # Calculate position sizes to maintain hedge ratio
                        pair_positions[symbol1] = int(position_value / (2 * current_price1))
                        pair_positions[symbol2] = -int((position_value / 2) / current_price2 * current_hedge_ratio)
                        
                        # Record entry
                        entry_price1 = current_price1
                        entry_price2 = current_price2
                        entry_date = current_date
                        pair_state = "long_spread"
                        
                        # Apply transaction costs
                        transaction_cost = abs(pair_positions[symbol1]) * current_price1 * transaction_cost_pct
                        transaction_cost += abs(pair_positions[symbol2]) * current_price2 * transaction_cost_pct
                        cash -= transaction_cost
                        
                        # Record trade
                        pair_trades.append({
                            'entry_date': entry_date,
                            'entry_z': current_z,
                            'entry_price1': entry_price1,
                            'entry_price2': entry_price2,
                            'position1': pair_positions[symbol1],
                            'position2': pair_positions[symbol2],
                            'type': 'long_spread',
                            'transaction_cost': transaction_cost
                        })
                
                # 2. If we're in a position and z-score reverts, exit position
                elif pair_state == "short_spread" and current_z < 0:
                    # Close positions
                    exit_price1 = current_price1
                    exit_price2 = current_price2
                    
                    # Calculate P&L
                    pnl1 = pair_positions[symbol1] * (entry_price1 - exit_price1)
                    pnl2 = pair_positions[symbol2] * (exit_price2 - entry_price2)
                    total_pnl = pnl1 + pnl2
                    
                    # Apply transaction costs
                    transaction_cost = abs(pair_positions[symbol1]) * exit_price1 * transaction_cost_pct
                    transaction_cost += abs(pair_positions[symbol2]) * exit_price2 * transaction_cost_pct
                    total_pnl -= transaction_cost
                    
                    # Update cash and portfolio value
                    cash += total_pnl
                    
                    # Update trade record
                    pair_trades[-1].update({
                        'exit_date': current_date,
                        'exit_z': current_z,
                        'exit_price1': exit_price1,
                        'exit_price2': exit_price2,
                        'pnl': total_pnl,
                        'duration': (current_date - entry_date).days
                    })
                    
                    # Reset state
                    pair_positions = {symbol1: 0, symbol2: 0}
                    pair_state = "flat"
                    
                elif pair_state == "long_spread" and current_z > 0:
                    # Close positions
                    exit_price1 = current_price1
                    exit_price2 = current_price2
                    
                    # Calculate P&L
                    pnl1 = pair_positions[symbol1] * (exit_price1 - entry_price1)
                    pnl2 = pair_positions[symbol2] * (entry_price2 - exit_price2)
                    total_pnl = pnl1 + pnl2
                    
                    # Apply transaction costs
                    transaction_cost = abs(pair_positions[symbol1]) * exit_price1 * transaction_cost_pct
                    transaction_cost += abs(pair_positions[symbol2]) * exit_price2 * transaction_cost_pct
                    total_pnl -= transaction_cost
                    
                    # Update cash and portfolio value
                    cash += total_pnl
                    
                    # Update trade record
                    pair_trades[-1].update({
                        'exit_date': current_date,
                        'exit_z': current_z,
                        'exit_price1': exit_price1,
                        'exit_price2': exit_price2,
                        'pnl': total_pnl,
                        'duration': (current_date - entry_date).days
                    })
                    
                    # Reset state
                    pair_positions = {symbol1: 0, symbol2: 0}
                    pair_state = "flat"
                
                # Calculate current portfolio value
                position_value = 0
                if pair_positions[symbol1] != 0:
                    position_value += pair_positions[symbol1] * current_price1
                if pair_positions[symbol2] != 0:
                    position_value += pair_positions[symbol2] * current_price2
                
                daily_portfolio_value = cash + position_value
                
                # Record daily equity for this pair
                pair_equity.append({
                    'date': current_date,
                    'portfolio_value': daily_portfolio_value,
                    'cash': cash,
                    'position_value': position_value,
                    'z_score': current_z,
                    'spread': spread[i]
                })
            
            # Close any open positions at the end of the backtest
            if pair_state != "flat":
                last_date = price1.index[-1]
                last_price1 = price1[-1]
                last_price2 = price2[-1]
                
                # Calculate P&L
                if pair_state == "short_spread":
                    pnl1 = pair_positions[symbol1] * (entry_price1 - last_price1)
                    pnl2 = pair_positions[symbol2] * (last_price2 - entry_price2)
                else:  # long_spread
                    pnl1 = pair_positions[symbol1] * (last_price1 - entry_price1)
                    pnl2 = pair_positions[symbol2] * (entry_price2 - last_price2)
                
                total_pnl = pnl1 + pnl2
                
                # Apply transaction costs
                transaction_cost = abs(pair_positions[symbol1]) * last_price1 * transaction_cost_pct
                transaction_cost += abs(pair_positions[symbol2]) * last_price2 * transaction_cost_pct
                total_pnl -= transaction_cost
                
                # Update cash
                cash += total_pnl
                
                # Update trade record
                pair_trades[-1].update({
                    'exit_date': last_date,
                    'exit_z': z_score[-1],
                    'exit_price1': last_price1,
                    'exit_price2': last_price2,
                    'pnl': total_pnl,
                    'duration': (last_date - entry_date).days,
                    'note': 'Closed at end of backtest'
                })
            
            # Store pair results
            pair_equity_df = pd.DataFrame(pair_equity)
            
            # Calculate performance metrics
            if not pair_equity_df.empty:
                pair_equity_df['daily_returns'] = pair_equity_df['portfolio_value'].pct_change()
                
                # Calculate metrics
                total_return = (pair_equity_df['portfolio_value'].iloc[-1] / initial_capital) - 1
                annualized_return = ((1 + total_return) ** (252 / len(pair_equity_df))) - 1
                annualized_volatility = pair_equity_df['daily_returns'].std() * np.sqrt(252)
                sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
                
                max_drawdown = 0
                peak = pair_equity_df['portfolio_value'].iloc[0]
                for value in pair_equity_df['portfolio_value']:
                    if value > peak:
                        peak = value
                    drawdown = (peak - value) / peak
                    max_drawdown = max(max_drawdown, drawdown)
                
                # Count trades and calculate win rate
                winning_trades = sum(1 for trade in pair_trades if trade.get('pnl', 0) > 0)
                total_trades = len(pair_trades)
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                
                # Store in results
                all_pair_results[f"{symbol1}_{symbol2}"] = {
                    'equity_curve': pair_equity_df,
                    'trades': pair_trades,
                    'total_return': total_return,
                    'annualized_return': annualized_return,
                    'annualized_volatility': annualized_volatility,
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'trade_count': total_trades,
                    'win_rate': win_rate,
                    'hedge_ratio': hedge_ratio,
                    'z_scores': z_score,
                    'spread': spread
                }
        
        # Calculate portfolio level metrics by combining all pairs
        combined_equity = pd.DataFrame()
        
        for pair_key, pair_result in all_pair_results.items():
            if combined_equity.empty:
                combined_equity = pair_result['equity_curve'][['date', 'portfolio_value']].copy()
                combined_equity.set_index('date', inplace=True)
                combined_equity.rename(columns={'portfolio_value': 'total_value'}, inplace=True)
            else:
                # Add this pair's contribution
                temp_df = pair_result['equity_curve'][['date', 'portfolio_value']].copy()
                temp_df.set_index('date', inplace=True)
                temp_df.rename(columns={'portfolio_value': pair_key}, inplace=True)
                
                combined_equity = pd.merge(
                    combined_equity, 
                    temp_df, 
                    left_index=True, 
                    right_index=True, 
                    how='outer'
                )
                combined_equity[pair_key].fillna(method='ffill', inplace=True)
        
        # Calculate portfolio metrics
        if not combined_equity.empty:
            combined_equity['daily_returns'] = combined_equity['total_value'].pct_change()
            
            total_return = (combined_equity['total_value'].iloc[-1] / initial_capital) - 1
            annualized_return = ((1 + total_return) ** (252 / len(combined_equity))) - 1
            annualized_volatility = combined_equity['daily_returns'].std() * np.sqrt(252)
            sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
            
            max_drawdown = 0
            peak = combined_equity['total_value'].iloc[0]
            for value in combined_equity['total_value']:
                if value > peak:
                    peak = value
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
            
            # Compile all trades
            all_trades = []
            for pair_key, pair_result in all_pair_results.items():
                for trade in pair_result['trades']:
                    trade['pair'] = pair_key
                    all_trades.append(trade)
            
            # Sort trades by entry date
            all_trades.sort(key=lambda x: x['entry_date'])
            
            # Calculate overall win rate
            winning_trades = sum(1 for trade in all_trades if trade.get('pnl', 0) > 0)
            total_trades = len(all_trades)
            win_rate = winning_trades / total_trades if total_trades > 0 else 0
            
            # Store portfolio results
            portfolio_results = {
                'equity_curve': combined_equity,
                'trades': all_trades,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'annualized_volatility': annualized_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'trade_count': total_trades,
                'win_rate': win_rate
            }
            
            # Store all results
            self.results = {
                'portfolio': portfolio_results,
                'pairs': all_pair_results,
                'initial_capital': initial_capital,
                'start_date': start_date,
                'end_date': end_date,
                'transaction_cost_pct': transaction_cost_pct,
                'position_size_pct': position_size_pct
            }
            
            return self.results
    
    def plot_results(self, include_pairs=True, figsize=(14, 8)):
        """
        Plot backtest results
        
        Args:
            include_pairs: Whether to include individual pair plots
            figsize: Figure size
            
        Returns:
            None (displays plots)
        """
        if not self.results:
            print("No backtest results to plot")
            return
        
        # Create figure
        fig = plt.figure(figsize=figsize)
        
        # Plot portfolio equity curve
        ax1 = fig.add_subplot(211)
        ax1.set_title('Portfolio Equity Curve')
        ax1.plot(self.results['portfolio']['equity_curve'].index, 
                 self.results['portfolio']['equity_curve']['total_value'])
        ax1.set_ylabel('Portfolio Value')
        ax1.grid(True)
        
        # Plot drawdown
        ax2 = fig.add_subplot(212)
        ax2.set_title('Portfolio Drawdown')
        
        # Calculate drawdown
        equity_curve = self.results['portfolio']['equity_curve']['total_value']
        rolling_max = equity_curve.cummax()
        drawdown = (rolling_max - equity_curve) / rolling_max
        
        ax2.fill_between(equity_curve.index, 0, drawdown, color='red', alpha=0.3)
        ax2.set_ylabel('Drawdown')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Plot individual pairs if requested
        if include_pairs:
            for pair_key, pair_result in self.results['pairs'].items():
                fig = plt.figure(figsize=figsize)
                
                # Plot pair equity curve
                ax1 = fig.add_subplot(211)
                ax1.set_title(f'Pair Equity Curve: {pair_key}')
                ax1.plot(pair_result['equity_curve']['date'], 
                         pair_result['equity_curve']['portfolio_value'])
                ax1.set_ylabel('Value')
                ax1.grid(True)
                
                # Plot z-score
                ax2 = fig.add_subplot(212)
                ax2.set_title(f'Z-Score: {pair_key}')
                
                # Get z-score series
                z_score = pair_result['z_scores']
                
                # Add horizontal lines for thresholds
                threshold = self.pairs[list(self.results['pairs'].keys()).index(pair_key)]['z_score_threshold']
                ax2.axhline(y=threshold, color='r', linestyle='--', alpha=0.5)
                ax2.axhline(y=-threshold, color='r', linestyle='--', alpha=0.5)
                ax2.axhline(y=0, color='k', linestyle='-', alpha=0.2)
                
                # Plot z-score
                ax2.plot(z_score.index, z_score, color='blue')
                
                # Highlight trades
                for trade in pair_result['trades']:
                    if trade['type'] == 'long_spread':
                        color = 'green'
                    else:
                        color = 'red'
                    
                    # Mark entry and exit
                    ax2.plot(trade['entry_date'], trade['entry_z'], 'o', color=color, markersize=6)
                    if 'exit_date' in trade:
                        ax2.plot(trade['exit_date'], trade['exit_z'], 'x', color=color, markersize=6)
                
                ax2.set_ylabel('Z-Score')
                ax2.grid(True)
                
                plt.tight_layout()
                plt.show()
    
    def get_performance_summary(self):
        """
        Get a summary of backtest performance
        
        Returns:
            DataFrame: Performance summary
        """
        if not self.results:
            print("No backtest results to summarize")
            return None
        
        # Create summary DataFrame
        summary = pd.DataFrame({
            'Metric': [
                'Initial Capital',
                'Final Capital',
                'Total Return',
                'Annualized Return',
                'Annualized Volatility',
                'Sharpe Ratio',
                'Max Drawdown',
                'Number of Trades',
                'Win Rate',
                'Transaction Cost %',
                'Position Size %',
                'Start Date',
                'End Date',
                'Duration (days)'
            ],
            'Value': [
                f"${self.results['initial_capital']:,.2f}",
                f"${self.results['portfolio']['equity_curve']['total_value'].iloc[-1]:,.2f}",
                f"{self.results['portfolio']['total_return']:.2%}",
                f"{self.results['portfolio']['annualized_return']:.2%}",
                f"{self.results['portfolio']['annualized_volatility']:.2%}",
                f"{self.results['portfolio']['sharpe_ratio']:.2f}",
                f"{self.results['portfolio']['max_drawdown']:.2%}",
                f"{self.results['portfolio']['trade_count']}",
                f"{self.results['portfolio']['win_rate']:.2%}",
                f"{self.results['transaction_cost_pct']:.2%}",
                f"{self.results['position_size_pct']:.2%}",
                f"{self.results['start_date'].strftime('%Y-%m-%d')}",
                f"{self.results['end_date'].strftime('%Y-%m-%d')}",
                f"{(self.results['end_date'] - self.results['start_date']).days}"
            ]
        })
        
        return summary
    
    def get_trades_summary(self):
        """
        Get a summary of all trades
        
        Returns:
            DataFrame: Trades summary
        """
        if not self.results:
            print("No backtest results to summarize")
            return None
        
        trades_df = pd.DataFrame(self.results['portfolio']['trades'])
        
        if trades_df.empty:
            return pd.DataFrame()
        
        # Add more metrics
        trades_df['return'] = trades_df['pnl'] / self.results['initial_capital']
        trades_df['winning'] = trades_df['pnl'] > 0
        
        # Calculate summary by pair
        pair_summary = trades_df.groupby('pair').agg({
            'pnl': ['sum', 'mean', 'std', 'min', 'max'],
            'winning': 'mean',
            'duration': ['mean', 'min', 'max'],
            'entry_date': 'count'
        })
        
        # Rename columns
        pair_summary.columns = [
            'Total P&L', 'Avg P&L', 'Std P&L', 'Min P&L', 'Max P&L',
            'Win Rate', 'Avg Duration', 'Min Duration', 'Max Duration',
            'Trade Count'
        ]
        
        return pair_summary
    
    def export_results_to_csv(self, folder_path='.'):
        """
        Export backtest results to CSV files
        
        Args:
            folder_path: Folder to save CSV files
            
        Returns:
            None
        """
        if not self.results:
            print("No backtest results to export")
            return
        
        # Create timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export portfolio equity curve
        self.results['portfolio']['equity_curve'].to_csv(
            f"{folder_path}/portfolio_equity_{timestamp}.csv"
        )
        
        # Export trades
        trades_df = pd.DataFrame(self.results['portfolio']['trades'])
        if not trades_df.empty:
            trades_df.to_csv(f"{folder_path}/trades_{timestamp}.csv", index=False)
        
        # Export pair results
        for pair_key, pair_result in self.results['pairs'].items():
            pair_equity = pair_result['equity_curve']
            pair_equity.to_csv(f"{folder_path}/pair_{pair_key}_equity_{timestamp}.csv", index=False)
            
            pair_trades = pd.DataFrame(pair_result['trades'])
            if not pair_trades.empty:
                pair_trades.to_csv(f"{folder_path}/pair_{pair_key}_

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