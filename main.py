from AlgorithmImports import *
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from statsmodels.tsa.stattools import coint
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

class CointegratedPairsTradingStrategy(QCAlgorithm):
    
    def Initialize(self):
        # Algorithm start and end date, and initial cash
        self.SetStartDate(2022, 1, 1) 
        self.SetEndDate(2023, 1, 1)
        self.SetCash(100000)
        
        # Add SPY for scheduling and market volatility reference
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # Universe of US stocks from NYSE, NASDAQ, and AMEX
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelection, self.FineSelection)
        
        # Monthly rebalancing on the last trading day of each month
        self.Schedule.On(self.DateRules.MonthEnd(self.spy), 
                         self.TimeRules.AfterMarketOpen(self.spy, 30), 
                         self.SetRebalanceFlag)
        
        # Parameters for PCA and clustering
        self.lookbackDays = 252  # 12-month history
        self.pca_variance_threshold = 0.9  # 90% variance explained in PCA
        self.cluster_distance_threshold = None
        self.cluster_percentile = 20  # Lower percentile for more clusters
        self.correlation_threshold = 0.3  # Maximum allowed correlation between pairs
        
        self.pairs = {}  # Holds information about pairs
        self.rebalanceFlag = False  # Rebalance flag
        
        # Trading parameters
        self.entry_threshold = 2.0  # Z-score to enter a trade
        self.exit_threshold = 0.5   # Z-score to exit a trade
        self.stop_loss = 3.0        # Z-score stop-loss level
        self.take_profit = 0.0      # Z-score take-profit level (optional)
        
        # Transaction cost parameters
        self.transaction_cost = 0.001  # 0.1% per trade
        
        # For tracking open positions
        self.open_positions = {}
        
        # Debugging: Track the number of symbols at each stage
        self.symbols_count = {
            "CoarseSelected": 0,
            "FineSelected": 0,
            "AfterHistory": 0,
            "AfterFeaturePrep": 0
        }
        
        # Hurst exponent threshold
        self.hurst_threshold = 0.5  # Mean-reverting if H < 0.5
        
        # Volatility scaling parameters
        self.market_volatility_window = 63  # Approx. 3 months
        self.market_volatility = 0.0
        
    def CoarseSelection(self, coarse):
        # Lower the minimum price to $5 and reduce the minimum dollar volume to $1 million
        selected = [x.Symbol for x in coarse 
                   if x.HasFundamentalData and 
                      x.Price > 5 and 
                      x.DollarVolume > 1e6]
        self.symbols_count["CoarseSelected"] = len(selected)
        self.Debug(f"Coarse Selection: {self.symbols_count['CoarseSelected']} symbols selected.")
        return selected
    
    def FineSelection(self, fine):
        # Relaxed filters to include more stocks
        fine_filtered = [x for x in fine 
                         if x.MarketCap > 5e8 and  # Reduced MarketCap to 500M
                            x.EarningReports.BasicAverageShares.ThreeMonths > 5e6 and  # Reduced minimum shares filter
                            x.ValuationRatios.PERatio > 0 and x.ValuationRatios.PERatio < 100 and  # Loosened P/E
                            x.ValuationRatios.PBRatio > 0 and x.ValuationRatios.PBRatio < 10 and  # Loosened P/B
                            (x.OperationRatios.ROE.Value if x.OperationRatios.ROE.Value is not None else 0) > 0 and  # Loosened ROE to > 0
                            x.AssetClassification.MorningstarSectorCode != 0]  # Ensure valid sector code
        
        # Ensure sufficient history is available
        fine_selected = []
        for x in fine_filtered:
            history = self.History(x.Symbol, self.lookbackDays, Resolution.Daily)
            if not history.empty and len(history) >= self.lookbackDays:
                fine_selected.append(x.Symbol)
        
        self.symbols_count["FineSelected"] = len(fine_selected)
        self.Debug(f"Fine Selection: {self.symbols_count['FineSelected']} symbols selected.")
        return fine_selected
    
    def SetRebalanceFlag(self):
        # Set the rebalance flag at the scheduled time
        self.rebalanceFlag = True
        self.Debug("Rebalance flag set.")
        
    def OnData(self, data):
        if not self.rebalanceFlag:
            self.ManagePositions()
            return
        
        self.rebalanceFlag = False
        self.Debug("Rebalancing triggered.")
        
        # Retrieve historical data for selected universe
        symbols = list(self.ActiveSecurities.Keys)
        self.Debug(f"Active Securities Count: {len(symbols)}")
        
        if len(symbols) < 2:
            self.Debug("Not enough symbols after universe selection.")
            return
        
        history = self.History(symbols, self.lookbackDays, Resolution.Daily)
        self.symbols_count["AfterHistory"] = len(history.groupby(level=0))
        self.Debug(f"History retrieved for {self.symbols_count['AfterHistory']} symbols.")
        
        if history.empty or self.symbols_count["AfterHistory"] < 2:
            self.Debug("Not enough data or symbols for clustering.")
            return
        
        # Prepare data for clustering
        stock_data = {}
        for symbol in symbols:
            if symbol not in history.index.levels[0]:
                continue  # Skip symbols with no history
            
            stock_history = history.loc[symbol].close
            returns = stock_history.pct_change().dropna()
            sector_code = self.GetSectorCode(symbol)
            fundamental_data = self.GetFundamentalData(symbol)
            
            if fundamental_data and not returns.empty:
                # Calculate monthly returns (approx. 21 trading days per month)
                momentum_features = [returns[-i*21:].sum() for i in range(1, 13)]
                # Include sector as a numerical feature
                firm_features = [fundamental_data['PE'], fundamental_data['PB'], sector_code]
                stock_data[symbol] = momentum_features + firm_features
            else:
                self.Debug(f"Skipping symbol {symbol.Value} due to missing data.")
        
        self.symbols_count["AfterFeaturePrep"] = len(stock_data)
        self.Debug(f"Prepared features for {self.symbols_count['AfterFeaturePrep']} symbols.")
        
        if not stock_data or self.symbols_count["AfterFeaturePrep"] < 2:
            self.Debug("Insufficient data after preparing stock features.")
            return
        
        # Create DataFrame for clustering
        stock_df = pd.DataFrame(stock_data).T
        self.Debug(f"Stock DataFrame shape before dropna: {stock_df.shape}")
        stock_df = stock_df.dropna()
        self.Debug(f"Stock DataFrame shape after dropna: {stock_df.shape}")
        
        if stock_df.empty or stock_df.shape[0] < 2:
            self.Debug("Insufficient data after dropping NaNs.")
            return
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(stock_df)
        
        # Apply PCA
        pca = PCA(n_components=self.pca_variance_threshold, svd_solver='full')
        stock_pca = pca.fit_transform(scaled_features)
        self.Debug(f"PCA reduced dimensions to: {stock_pca.shape[1]}")
        
        # Check for NaNs or Infs in stock_pca
        if not np.all(np.isfinite(stock_pca)):
            self.Debug("PCA transformed data contains non-finite values. Skipping clustering.")
            return
        
        # Determine distance threshold if not set
        if self.cluster_distance_threshold is None:
            # Calculate pairwise distances and set threshold based on percentile
            pairwise_distances = np.linalg.norm(stock_pca[:, np.newaxis] - stock_pca, axis=2)
            upper_tri_indices = np.triu_indices_from(pairwise_distances, k=1)
            distances = pairwise_distances[upper_tri_indices]
            
            # Debug: Log min and max distances
            if len(distances) == 0:
                self.Debug("No pairwise distances computed. Skipping clustering.")
                return
            
            min_distance = distances.min()
            max_distance = distances.max()
            self.Debug(f"Pairwise distances min: {min_distance}, max: {max_distance}")
            
            # Ensure distances are non-negative
            if min_distance < 0:
                self.Debug(f"Minimum distance {min_distance} is negative. Setting to 0.")
                min_distance = 0.0
            if max_distance < 0:
                self.Debug(f"Maximum distance {max_distance} is negative. Setting to 1.0 as default.")
                max_distance = 1.0
            
            # Compute the cluster distance threshold
            self.cluster_distance_threshold = np.percentile(distances, self.cluster_percentile)
            self.Debug(f"Computed cluster_distance_threshold before validation: {self.cluster_distance_threshold}")
            
            # Validate cluster_distance_threshold
            if not np.isfinite(self.cluster_distance_threshold) or self.cluster_distance_threshold < 0:
                self.Debug(f"Invalid cluster_distance_threshold {self.cluster_distance_threshold}. Setting to default 1.0")
                self.cluster_distance_threshold = 1.0  # Default positive value
            
            self.Debug(f"Cluster distance threshold set to: {self.cluster_distance_threshold}")
        
        # Agglomerative Clustering
        try:
            aggl_cluster = AgglomerativeClustering(
                linkage='average',
                n_clusters=None,
                distance_threshold=self.cluster_distance_threshold
            )
            cluster_labels = aggl_cluster.fit_predict(stock_pca)
            self.Debug(f"Number of clusters formed: {len(set(cluster_labels))}")
        except Exception as e:
            self.Debug(f"AgglomerativeClustering failed with error: {e}. Setting distance_threshold to default 1.0 and retrying.")
            # Set to default positive value and retry
            self.cluster_distance_threshold = 1.0
            try:
                aggl_cluster = AgglomerativeClustering(
                    linkage='average',
                    n_clusters=None,
                    distance_threshold=self.cluster_distance_threshold
                )
                cluster_labels = aggl_cluster.fit_predict(stock_pca)
                self.Debug(f"Number of clusters formed after retry: {len(set(cluster_labels))}")
            except Exception as e:
                self.Debug(f"AgglomerativeClustering failed again with error: {e}. Skipping this rebalance period.")
                return
        
        # Pair selection using cointegration
        self.pairs = self.SelectCointegratedPairs(cluster_labels, history)
        self.Debug(f"Number of cointegrated pairs found: {len(self.pairs)}")
        
        if not self.pairs:
            self.Debug("No cointegrated pairs found in this rebalance period.")
            return
        
        # Multi-Pair Correlation Control
        self.pairs = self.FilterHighCorrelationPairs()
        self.Debug(f"Number of pairs after correlation filtering: {len(self.pairs)}")
        
        if not self.pairs:
            self.Debug("All pairs were filtered out due to high correlation.")
            return
        
        # Liquidate positions not in new pairs
        self.LiquidatePositionsNotInPairs()
        
        # Prepare for trading
        self.open_positions = {}
        for pair_key, pair_info in self.pairs.items():
            symbol_y = pair_info['symbol_y']
            symbol_x = pair_info['symbol_x']
            self.open_positions[pair_key] = {
                'symbol_y': symbol_y,
                'symbol_x': symbol_x,
                'hedge_ratio': pair_info['hedge_ratio'],
                'spread_mean': pair_info['spread_mean'],
                'spread_std': pair_info['spread_std'],
                'is_open': False
            }
        
        # Calculate Risk-Parity Weights based on inverse spread volatility
        inverse_volatilities = {pair: 1.0 / info['spread_std'] for pair, info in self.pairs.items()}
        total_inverse_vol = sum(inverse_volatilities.values())
        weights = {pair: vol / total_inverse_vol for pair, vol in inverse_volatilities.items()}
        
        # Adjust allocation based on market volatility (Volatility Scaling)
        self.market_volatility = self.CalculateMarketVolatility()
        if self.market_volatility > 0:
            scaling_factor = 1.0 / self.market_volatility
        else:
            scaling_factor = 1.0  # Default scaling
        
        # Allocate holdings based on weights and scaling factor
        for pair_key, weight in weights.items():
            position = self.open_positions[pair_key]
            symbol_y = position['symbol_y']
            symbol_x = position['symbol_x']
            hedge_ratio = position['hedge_ratio']
            
            allocation = weight * scaling_factor
            # Limit allocation to prevent overexposure
            allocation = min(allocation, 0.05)  # Max 5% per position
            
            # Initialize positions with zero holdings
            self.SetHoldings(symbol_y, 0)
            self.SetHoldings(symbol_x, 0)
    
    def SelectCointegratedPairs(self, cluster_labels, history):
        pairs = {}
        symbols = list(self.ActiveSecurities.Keys)
        clusters = {}
        
        # Group symbols by clusters
        for symbol, label in zip(symbols, cluster_labels):
            clusters.setdefault(label, []).append(symbol)
        
        # For each cluster, test for cointegration
        for cluster_id, cluster_symbols in clusters.items():
            if len(cluster_symbols) < 2:
                continue  # Need at least two symbols to form a pair
            
            self.Debug(f"Testing cointegration in cluster {cluster_id} with {len(cluster_symbols)} symbols.")
            
            for i in range(len(cluster_symbols)):
                for j in range(i+1, len(cluster_symbols)):
                    sym_i = cluster_symbols[i]
                    sym_j = cluster_symbols[j]
                    
                    try:
                        prices_i = history.loc[sym_i].close
                        prices_j = history.loc[sym_j].close
                        
                        # Ensure both series have no missing values
                        if prices_i.isnull().any() or prices_j.isnull().any():
                            self.Debug(f"Missing data for pair {sym_i.Value}-{sym_j.Value}. Skipping.")
                            continue
                        
                        # Cointegration test
                        coint_t, pvalue, _ = coint(prices_i, prices_j)
                        if pvalue < 0.05:
                            # Calculate hedge ratio using rolling regression
                            hedge_ratio = self.CalculateHedgeRatio(prices_j, prices_i)
                            
                            # Calculate spread
                            spread = prices_i - hedge_ratio * prices_j
                            spread_mean, spread_std = self.CalculateRollingSpreadStats(spread)
                            
                            # Calculate Hurst Exponent
                            hurst = self.CalculateHurstExponent(spread)
                            if hurst >= self.hurst_threshold:
                                self.Debug(f"Pair {sym_i.Value}-{sym_j.Value} rejected due to Hurst exponent {hurst:.2f}")
                                continue  # Not mean-reverting
                            
                            pair_key = f"{sym_i.Value}_{sym_j.Value}"
                            pairs[pair_key] = {
                                'symbol_y': sym_i,
                                'symbol_x': sym_j,
                                'hedge_ratio': hedge_ratio,
                                'spread_mean': spread_mean,
                                'spread_std': spread_std,
                                'hurst': hurst,
                                'spread_series': spread  # Store spread series
                            }
                            self.Debug(f"Cointegrated pair found: {pair_key} with p-value {pvalue} and Hurst {hurst:.2f}")
                    except Exception as e:
                        self.Debug(f"Error testing pair {sym_i.Value}-{sym_j.Value}: {e}")
                        continue
        return pairs
    
    def CalculateHedgeRatio(self, x, y):
        # Rolling window regression to calculate dynamic hedge ratio
        window = 60  # 60 days rolling window
        if len(x) < window:
            window = len(x)
        try:
            hedge_ratio = np.polyfit(x[-window:], y[-window:], 1)[0]
            return hedge_ratio
        except:
            return 1.0  # Default hedge ratio if regression fails
    
    def CalculateRollingSpreadStats(self, spread):
        # Calculate rolling mean and std using EMA
        lambda_val = 0.94
        spread_series = spread.values
        mu = spread_series[0]
        sigma = spread_series.std()
        for s in spread_series:
            mu = lambda_val * mu + (1 - lambda_val) * s
            sigma = lambda_val * sigma + (1 - lambda_val) * (s - mu)**2
        sigma = np.sqrt(sigma)
        return mu, sigma
    
    def CalculateHurstExponent(self, time_series):
        # Simple Hurst exponent calculation
        N = len(time_series)
        if N < 20:
            return 0.5  # Default value
        lag_max = min(20, N//2)
        tau = []
        for lag in range(2, lag_max):
            pp = np.polyfit(range(len(time_series)), time_series, 1)
            trend = np.polyval(pp, range(len(time_series)))
            detrended = time_series - trend
            lag_detrended = detrended[lag:]
            r = np.max(lag_detrended) - np.min(lag_detrended)
            s = np.std(detrended)
            if s != 0:
                tau.append(r/s)
        if len(tau) == 0:
            return 0.5
        hurst = np.polyfit(np.log(range(2, lag_max)), np.log(tau), 1)[0]
        return hurst
    
    def FilterHighCorrelationPairs(self):
        # Calculate correlation between spreads and filter out highly correlated pairs
        pair_keys = list(self.pairs.keys())
        
        # Create a DataFrame with spread time series
        spread_dict = {pair: self.pairs[pair]['spread_series'] for pair in pair_keys}
        
        spreads_df = pd.DataFrame(spread_dict)
        if spreads_df.empty:
            return self.pairs  # No pairs to filter
        
        # Compute the correlation matrix
        corr_matrix = spreads_df.corr()
        
        # Identify pairs to remove
        pairs_to_remove = set()
        for i in range(len(pair_keys)):
            for j in range(i + 1, len(pair_keys)):
                pair_i = pair_keys[i]
                pair_j = pair_keys[j]
                if pair_i in pairs_to_remove or pair_j in pairs_to_remove:
                    continue  # Already marked for removal
                correlation = corr_matrix.loc[pair_i, pair_j]
                if correlation > self.correlation_threshold:
                    # Exclude the pair with higher spread_std
                    if self.pairs[pair_i]['spread_std'] > self.pairs[pair_j]['spread_std']:
                        self.Debug(f"Excluding pair {pair_j} due to high correlation ({correlation:.2f}) with {pair_i}")
                        pairs_to_remove.add(pair_j)
                    else:
                        self.Debug(f"Excluding pair {pair_i} due to high correlation ({correlation:.2f}) with {pair_j}")
                        pairs_to_remove.add(pair_i)
        
        # Remove the identified pairs
        for pair in pairs_to_remove:
            del self.pairs[pair]
        
        return self.pairs
    
    def CalculateMarketVolatility(self):
        # Calculate rolling volatility of SPY using EMA
        history = self.History(self.spy, self.market_volatility_window, Resolution.Daily)
        if history.empty:
            return self.market_volatility
        returns = history['close'].pct_change().dropna()
        if len(returns) == 0:
            return self.market_volatility
        ema_variance = returns.ewm(span=self.market_volatility_window).var().iloc[-1]
        ema_std = np.sqrt(ema_variance)
        return ema_std
    
    def ManagePositions(self):
        # Calculate market volatility scaling factor
        if self.market_volatility > 0:
            scaling_factor = 1.0 / self.market_volatility
        else:
            scaling_factor = 1.0  # Default scaling
        
        # Monitor open positions and generate signals
        for pair_key, position in self.open_positions.items():
            symbol_y = position['symbol_y']
            symbol_x = position['symbol_x']
            hedge_ratio = position['hedge_ratio']
            spread_mean = position['spread_mean']
            spread_std = position['spread_std']
            
            # Check if data is available
            if not (self.Securities.ContainsKey(symbol_y) and self.Securities.ContainsKey(symbol_x)):
                continue
            price_y = self.Securities[symbol_y].Price
            price_x = self.Securities[symbol_x].Price
            
            if price_y == 0 or price_x == 0:
                continue
            
            # Calculate current spread and z-score
            spread = price_y - hedge_ratio * price_x
            z_score = (spread - spread_mean) / spread_std
            
            # Adjust z-score for transaction costs
            adjusted_z = abs(z_score) - self.transaction_cost * 100  # Normalize transaction cost
            
            # Entry signal
            if not position['is_open']:
                if adjusted_z > self.entry_threshold:
                    # Position sizing
                    allocation = scaling_factor * (1.0 / spread_std) / len(self.pairs)
                    allocation = min(allocation, 0.05)  # Max 5% per position
                    
                    # Go long spread: short x, long y
                    if z_score > self.entry_threshold:
                        self.SetHoldings(symbol_y, allocation)
                        self.SetHoldings(symbol_x, -allocation * hedge_ratio)
                        self.Debug(f"Entering long spread for pair {pair_key} at z-score {z_score:.2f}")
                    # Go short spread: long x, short y
                    elif z_score < -self.entry_threshold:
                        self.SetHoldings(symbol_y, -allocation)
                        self.SetHoldings(symbol_x, allocation * hedge_ratio)
                        self.Debug(f"Entering short spread for pair {pair_key} at z-score {z_score:.2f}")
                    position['is_open'] = True
                    position['entry_z'] = z_score
            else:
                # Exit conditions
                exit_condition = abs(z_score) < self.exit_threshold
                stop_loss_condition = abs(z_score) > self.stop_loss
                take_profit_condition = (self.take_profit > 0) and (abs(z_score) > self.take_profit)
                
                if exit_condition or stop_loss_condition or take_profit_condition:
                    self.Liquidate(symbol_y)
                    self.Liquidate(symbol_x)
                    self.Debug(f"Exiting position for pair {pair_key} at z-score {z_score:.2f}")
                    position['is_open'] = False
    
    def LiquidatePositionsNotInPairs(self):
        # Liquidate positions that are not in the current pairs
        current_symbols = set()
        for pair_info in self.pairs.values():
            current_symbols.add(pair_info['symbol_y'])
            current_symbols.add(pair_info['symbol_x'])
        
        for kvp in self.Portfolio:
            symbol = kvp.Key
            if symbol not in current_symbols and self.Portfolio[symbol].Invested:
                self.Liquidate(symbol)
                self.Debug(f"Liquidating position for symbol {symbol.Value} not in current pairs.")
    
    def GetFundamentalData(self, symbol):
        # Retrieve fundamental data (P/E and P/B ratios) for the given symbol
        if symbol in self.Securities:
            security = self.Securities[symbol]
            if security.HasData and security.Fundamentals is not None:
                pe_ratio = security.Fundamentals.ValuationRatios.PERatio
                pb_ratio = security.Fundamentals.ValuationRatios.PBRatio
                if pe_ratio is not None and pb_ratio is not None:
                    return {'PE': pe_ratio, 'PB': pb_ratio}
        return None
    
    def GetSectorCode(self, symbol):
        # Get sector code for the symbol
        if symbol in self.Securities:
            security = self.Securities[symbol]
            if security.HasData and security.Fundamentals is not None:
                sector_code = security.Fundamentals.AssetClassification.MorningstarSectorCode
                return sector_code
        return 0